//! Gemma 4 26B-A4B MTP (Multi-Token Prediction) drafter.
//!
//! Source checkpoint: `mlx-community/gemma-4-26B-A4B-it-assistant-bf16`,
//! ported from `google/gemma-4-26B-A4B-it-assistant` (Apache 2.0, May 5 2026).
//!
//! Architecture (see memory `gemma4_mtp_drafter_architecture.md`):
//! - 4-layer mini-Gemma4 (3 sliding + 1 full attn, dense MLP, hidden=1024).
//! - NO own KV cache — shares K/V with the trunk's last full-attention AND
//!   last sliding-attention layers (`attention_k_eq_v=true`,
//!   `num_kv_shared_layers=4`).
//! - Per-step input: `concat(trunk_embed(last_tok)[2816], trunk_last_h[2816])
//!   = [B, 1, 5632]` → `pre_projection [1024, 5632]` → drafter forward →
//!   `model.norm` → `post_projection [2816, 1024]` → trunk hidden space →
//!   trunk tied lm_head + softcap → vocab logits.
//!
//! Implementation phases (this file = Phase 1):
//! - **Phase 1** (this file): config + typed weight loader. Validates the 48
//!   safetensors keys load and dimensions match. No forward yet.
//! - **Phase 2** (next): `drafter_layer_forward` + `drafter_forward`.
//! - **Phase 3**: integration with `NativeGemma4Model::generate()`.
//! - **Phase 4**: verify + rollback on partial reject.
//! - **Phase 5**: bench + 3-pair cool A/B.

#[cfg(feature = "mlx-native")]
pub mod imp {
    use anyhow::{Context, Result, anyhow};
    use mlx_rs::Array;
    use serde::Deserialize;
    use std::collections::HashMap;
    use std::path::Path;

    // ───────────────────────── config.json ─────────────────────────

    /// Top-level config for `Gemma4AssistantForCausalLM`. Wraps the inner
    /// `text_config` plus the `backbone_hidden_size` (trunk hidden dim that
    /// drafter consumes via `pre_projection`).
    #[derive(Debug, Clone, Deserialize)]
    pub struct NativeGemma4MtpConfig {
        pub backbone_hidden_size: usize,
        pub text_config: NativeGemma4MtpTextConfig,
        #[serde(default)]
        pub centroid_intermediate_top_k: Option<usize>,
        #[serde(default)]
        pub num_centroids: Option<usize>,
        #[serde(default)]
        pub tie_word_embeddings: bool,
    }

    /// Inner text_config for the drafter. Mirrors `Gemma4TextConfig` but with
    /// drafter-specific fields (`enable_moe_block=false`, `attention_k_eq_v=
    /// true`, `num_kv_shared_layers=4`).
    #[derive(Debug, Clone, Deserialize)]
    pub struct NativeGemma4MtpTextConfig {
        pub hidden_size: usize,
        pub intermediate_size: usize,
        pub num_hidden_layers: usize,
        pub num_attention_heads: usize,
        pub num_key_value_heads: usize,
        pub head_dim: usize,
        pub num_global_key_value_heads: usize,
        pub global_head_dim: usize,
        pub sliding_window: usize,
        pub layer_types: Vec<String>,
        pub rms_norm_eps: f32,
        pub vocab_size: usize,
        #[serde(default)]
        pub attention_k_eq_v: bool,
        #[serde(default)]
        pub num_kv_shared_layers: usize,
        pub rope_parameters: NativeGemma4MtpRopeParams,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct NativeGemma4MtpRopeParams {
        pub full_attention: NativeGemma4MtpRopeKind,
        pub sliding_attention: NativeGemma4MtpRopeKind,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct NativeGemma4MtpRopeKind {
        pub rope_theta: f64,
        #[serde(default)]
        pub rope_type: String,
        #[serde(default)]
        pub partial_rotary_factor: Option<f64>,
    }

    impl NativeGemma4MtpConfig {
        pub fn load(path: &Path) -> Result<Self> {
            let raw = std::fs::read_to_string(path)
                .with_context(|| format!("NativeGemma4MtpConfig::load: read {}", path.display()))?;
            serde_json::from_str(&raw).with_context(|| {
                format!("NativeGemma4MtpConfig::load: parse JSON {}", path.display())
            })
        }
    }

    // ───────────────────────── layer kind ─────────────────────────

    /// Per-layer attention discriminant. Mirrors `NativeGemma4LayerType` but
    /// kept local so the MTP module compiles standalone.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum NativeGemma4MtpLayerType {
        Sliding,
        Full,
    }

    impl NativeGemma4MtpLayerType {
        fn from_str(s: &str) -> Result<Self> {
            match s {
                "sliding_attention" => Ok(Self::Sliding),
                "full_attention" => Ok(Self::Full),
                other => Err(anyhow!(
                    "NativeGemma4MtpLayerType::from_str: unknown layer kind `{other}`"
                )),
            }
        }
    }

    // ───────────────────────── resolved weights ─────────────────────────

    /// Drafter attention weights. Q-only (no K/V projection — K/V come from
    /// the trunk's shared cache). bf16 dense linear, NOT quantized.
    pub struct ResolvedGemma4MtpAttnWeights {
        /// `[H * head_dim, drafter_hidden]` bf16.
        pub q_proj: Array,
        /// `[H * head_dim, drafter_hidden]` bf16 (back-projection).
        pub o_proj: Array,
        /// `[head_dim]` bf16 — Gemma-convention RMSNorm scale (weight has +1
        /// baked in if the source did so; verify on load).
        pub q_norm: Array,
    }

    /// Drafter dense MLP weights. bf16, NOT quantized.
    pub struct ResolvedGemma4MtpMlpWeights {
        /// `[intermediate, hidden]` bf16.
        pub gate_proj: Array,
        /// `[intermediate, hidden]` bf16.
        pub up_proj: Array,
        /// `[hidden, intermediate]` bf16.
        pub down_proj: Array,
    }

    pub struct ResolvedGemma4MtpLayerWeights {
        pub kind: NativeGemma4MtpLayerType,
        pub attn: ResolvedGemma4MtpAttnWeights,
        pub mlp: ResolvedGemma4MtpMlpWeights,
        pub input_layernorm: Array,
        pub post_attention_layernorm: Array,
        pub pre_feedforward_layernorm: Array,
        pub post_feedforward_layernorm: Array,
        /// Shape `[1]` — per-layer output scalar.
        pub layer_scalar: Array,
    }

    /// Full resolved drafter: input/output projections, embedding table,
    /// 4 layers, final norm. Ready to dispatch a forward pass.
    pub struct ResolvedGemma4MtpDrafter {
        pub config: NativeGemma4MtpConfig,
        /// `[vocab, drafter_hidden]` bf16 — drafter's OWN embed. Stored for
        /// completeness; the spec uses **trunk's** embed for input, so the
        /// live decode path may not consume this directly.
        pub embed_tokens: Array,
        /// `[drafter_hidden, 2 * backbone_hidden]` bf16. Maps
        /// `concat(trunk_embed(tok), trunk_h)` → drafter hidden space.
        pub pre_projection: Array,
        /// `[backbone_hidden, drafter_hidden]` bf16. Maps drafter hidden
        /// back into trunk hidden space for tied lm_head.
        pub post_projection: Array,
        pub layers: Vec<ResolvedGemma4MtpLayerWeights>,
        pub final_norm: Array,
        /// Pre-built proportional-RoPE freqs for full-attn layers. Built once
        /// at load time per `gemma4_moe.rs` trunk convention (inf-tail freqs
        /// at indices `[rope_dim/2 .. head_dim/2)` with rotated values at
        /// `[0..rope_dim/2)`). For sliding-attn layers we use the freqs-less
        /// path since `partial_rotary_factor=1.0` (full rotation).
        pub rope_freqs_full: Array,
        /// `rope_base` for sliding-attn layers (full rotation).
        pub rope_base_sliding: f32,
    }

    // ───────────────────────── weight loader ─────────────────────────

    /// Gemma RMSNorm convention: trunk's MXFP4 export bakes `1 + raw_w`
    /// into its safetensors so `rms_norm(x, baked_w, eps)` gives the right
    /// scale. But the drafter checkpoint
    /// (`mlx-community/gemma-4-26B-A4B-it-assistant-bf16`) is stored with
    /// learned weights of mean ~3-5 and is loaded by mlx-vlm via plain
    /// `nn.RMSNorm` (no `+1` added at runtime). Adding `+1` on top breaks
    /// the drafter's effective scale.
    ///
    /// Default OFF (2026-05-18). Empirical: with PLUS_ONE=1, drafter
    /// outputs collapse to a token attractor (1595/1852/1902 family) — 0%
    /// acceptance. With PLUS_ONE=0 (and the lm_head + embed_scale +
    /// constant-position fixes), drafter produces diverse, real-looking
    /// drafts with acceptance > 0%.
    ///
    /// Kept env-gated for future drafter checkpoints with different
    /// conventions.
    fn add_one_to_norm(w: Array) -> Result<Array> {
        let on = std::env::var("LUMEN_GEMMA4_MTP_NORM_PLUS_ONE")
            .map(|v| v == "1")
            .unwrap_or(false);
        if !on {
            return Ok(w);
        }
        let one = Array::from_f32(1.0)
            .as_dtype(w.dtype())
            .context("add_one_to_norm: as_dtype")?;
        mlx_rs::ops::add(&w, &one).context("add_one_to_norm: add scalar 1")
    }

    /// Load every drafter weight from a directory containing `config.json`
    /// + `model.safetensors`. Validates that all expected keys are present
    ///   and dimensions match the config.
    pub fn load_drafter(drafter_dir: &Path) -> Result<ResolvedGemma4MtpDrafter> {
        let config_path = drafter_dir.join("config.json");
        let config = NativeGemma4MtpConfig::load(&config_path)
            .with_context(|| format!("load_drafter: parse {}", config_path.display()))?;

        let n_layers = config.text_config.num_hidden_layers;
        if config.text_config.layer_types.len() != n_layers {
            return Err(anyhow!(
                "load_drafter: layer_types.len()={} != num_hidden_layers={}",
                config.text_config.layer_types.len(),
                n_layers
            ));
        }
        if n_layers != 4 {
            // 26B-A4B drafter is always 4 layers per the May 2026 spec; bail
            // loudly if a future variant changes this so the forward path
            // doesn't silently mis-route KV sharing.
            return Err(anyhow!(
                "load_drafter: expected num_hidden_layers=4 for Gemma4Assistant, got {n_layers}"
            ));
        }

        let mut tensors = load_safetensors_dir(drafter_dir)?;

        // Top-level tensors.
        let embed_tokens = require_take(&mut tensors, "model.embed_tokens.weight")?;
        let pre_projection = require_take(&mut tensors, "pre_projection.weight")?;
        let post_projection = require_take(&mut tensors, "post_projection.weight")?;
        let final_norm = add_one_to_norm(require_take(&mut tensors, "model.norm.weight")?)?;

        // Per-layer.
        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let kind = NativeGemma4MtpLayerType::from_str(&config.text_config.layer_types[l])?;
            let base = format!("model.layers.{l}");
            let attn = ResolvedGemma4MtpAttnWeights {
                q_proj: require_take(&mut tensors, &format!("{base}.self_attn.q_proj.weight"))?,
                o_proj: require_take(&mut tensors, &format!("{base}.self_attn.o_proj.weight"))?,
                q_norm: add_one_to_norm(require_take(
                    &mut tensors,
                    &format!("{base}.self_attn.q_norm.weight"),
                )?)?,
            };
            let mlp = ResolvedGemma4MtpMlpWeights {
                gate_proj: require_take(&mut tensors, &format!("{base}.mlp.gate_proj.weight"))?,
                up_proj: require_take(&mut tensors, &format!("{base}.mlp.up_proj.weight"))?,
                down_proj: require_take(&mut tensors, &format!("{base}.mlp.down_proj.weight"))?,
            };
            let input_layernorm = add_one_to_norm(require_take(
                &mut tensors,
                &format!("{base}.input_layernorm.weight"),
            )?)?;
            let post_attention_layernorm = add_one_to_norm(require_take(
                &mut tensors,
                &format!("{base}.post_attention_layernorm.weight"),
            )?)?;
            let pre_feedforward_layernorm = add_one_to_norm(require_take(
                &mut tensors,
                &format!("{base}.pre_feedforward_layernorm.weight"),
            )?)?;
            let post_feedforward_layernorm = add_one_to_norm(require_take(
                &mut tensors,
                &format!("{base}.post_feedforward_layernorm.weight"),
            )?)?;
            let layer_scalar = require_take(&mut tensors, &format!("{base}.layer_scalar"))?;
            layers.push(ResolvedGemma4MtpLayerWeights {
                kind,
                attn,
                mlp,
                input_layernorm,
                post_attention_layernorm,
                pre_feedforward_layernorm,
                post_feedforward_layernorm,
                layer_scalar,
            });
        }

        // Validate no unexpected residual keys. Drafter checkpoint should
        // have exactly 48 keys; if anything's left over, the caller probably
        // pointed at the wrong model and we want to surface that early.
        if !tensors.is_empty() {
            let leftover: Vec<&String> = tensors.keys().collect();
            return Err(anyhow!(
                "load_drafter: {} unexpected leftover keys: {:?}",
                tensors.len(),
                leftover
            ));
        }

        // Dimensional sanity checks (drafter hidden = config.text_config.hidden_size).
        let dh = config.text_config.hidden_size as i32;
        let bh = config.backbone_hidden_size as i32;
        let pre = pre_projection.shape();
        if pre != [dh, (2 * bh)] {
            return Err(anyhow!(
                "load_drafter: pre_projection shape {pre:?} != [{dh}, {}]",
                2 * bh
            ));
        }
        let post = post_projection.shape();
        if post != [bh, dh] {
            return Err(anyhow!(
                "load_drafter: post_projection shape {post:?} != [{bh}, {dh}]"
            ));
        }
        let embed_shape = embed_tokens.shape();
        if embed_shape != [config.text_config.vocab_size as i32, dh] {
            return Err(anyhow!(
                "load_drafter: embed_tokens shape {embed_shape:?} != [{}, {dh}]",
                config.text_config.vocab_size
            ));
        }

        // Pre-build proportional RoPE freqs for the full-attn layer per the
        // trunk's convention (see `gemma4_moe.rs` ProportionalRoPE block).
        // For sliding layers (rope_dim == head_dim) the freqs-less path is
        // used at forward time, so we only precompute the full-attn version.
        let rope_full = &config.text_config.rope_parameters.full_attention;
        let global_head_dim = config.text_config.global_head_dim as i32;
        let partial = rope_full.partial_rotary_factor.unwrap_or(1.0) as f32;
        let rope_dim_full = (global_head_dim as f32 * partial) as i32;
        let rope_base_full = rope_full.rope_theta as f32;
        if rope_dim_full % 2 != 0 || global_head_dim % 2 != 0 {
            return Err(anyhow!(
                "load_drafter: rope_dim ({rope_dim_full}) and head_dim ({global_head_dim}) must be even"
            ));
        }
        let half_rot = (rope_dim_full / 2) as usize;
        let half_inf = ((global_head_dim - rope_dim_full) / 2) as usize;
        let mut freqs_vals: Vec<f32> = Vec::with_capacity(half_rot + half_inf);
        for i in 0..half_rot {
            let exp = (2 * i) as f32 / global_head_dim as f32;
            freqs_vals.push(rope_base_full.powf(exp));
        }
        for _ in 0..half_inf {
            freqs_vals.push(f32::INFINITY);
        }
        let rope_freqs_full = Array::from_slice(&freqs_vals, &[freqs_vals.len() as i32]);

        let rope_base_sliding = config
            .text_config
            .rope_parameters
            .sliding_attention
            .rope_theta as f32;

        Ok(ResolvedGemma4MtpDrafter {
            config,
            embed_tokens,
            pre_projection,
            post_projection,
            layers,
            final_norm,
            rope_freqs_full,
            rope_base_sliding,
        })
    }

    // ───────────────────────── Phase 2 — drafter forward ─────────────────────────

    impl ResolvedGemma4MtpDrafter {
        /// Run a SINGLE draft step. Caller-managed inputs (Phase 3 orchestrates):
        ///
        /// - `trunk_embed_of_last`: trunk's embed lookup for the previously
        ///   committed/drafted token, shape `[1, 1, backbone_hidden]` bf16.
        /// - `last_hidden_state`: trunk's last-layer hidden at the previous
        ///   position (or the drafter's recurrent state from the prior draft
        ///   step), shape `[1, 1, backbone_hidden]` bf16.
        /// - `trunk_k_full` / `trunk_v_full`: trunk's last full-attn KV cache
        ///   slice, each shape `[1, num_global_kv=2, T, global_head_dim=512]`.
        /// - `trunk_k_sliding` / `trunk_v_sliding`: trunk's last sliding-attn
        ///   KV cache slice, each shape `[1, num_kv=8, T_sliding, head_dim=256]`.
        /// - `position`: absolute position of this Q in the sequence (used as
        ///   RoPE offset). For draft step `k`, position = N + (k - 1) where N
        ///   is the trunk's commit offset.
        ///
        /// Returns: the drafter's output projected into trunk hidden space,
        /// shape `[1, 1, backbone_hidden]` bf16. The caller applies trunk's
        /// tied lm_head + softcap to obtain logits, and feeds the returned
        /// hidden back as `last_hidden_state` for the next draft step.
        #[allow(clippy::too_many_arguments)]
        pub fn draft_step(
            &self,
            trunk_embed_of_last: &Array,
            last_hidden_state: &Array,
            trunk_k_full: &Array,
            trunk_v_full: &Array,
            trunk_k_sliding: &Array,
            trunk_v_sliding: &Array,
            position: i32,
        ) -> Result<(Array, Array)> {
            let eps = self.config.text_config.rms_norm_eps;

            // (1) Input fusion: concat → pre_projection → drafter hidden.
            // Order gated by `LUMEN_GEMMA4_MTP_CONCAT_HE` (default 0):
            //   0 → [target_embed, last_hidden]  (architecture spec memo)
            //   1 → [last_hidden, target_embed]  (try-flip diagnostic)
            // The drafter's pre_projection weight is [drafter_hidden,
            // 2*backbone_hidden] — the slicing convention for which half
            // is embed vs hidden was inferred from documentation; mlx-vlm
            // / lilting.ch reference would disambiguate definitively.
            let concat_he = std::env::var("LUMEN_GEMMA4_MTP_CONCAT_HE")
                .map(|v| v == "1")
                .unwrap_or(false);
            let x = if concat_he {
                mlx_rs::ops::concatenate_axis(&[last_hidden_state, trunk_embed_of_last], -1)
                    .context("draft_step: concat(last_hidden, target_embed)")?
            } else {
                mlx_rs::ops::concatenate_axis(&[trunk_embed_of_last, last_hidden_state], -1)
                    .context("draft_step: concat(target_embed, last_hidden)")?
            };
            // pre_projection.weight is [out=drafter_hidden, in=2*backbone_hidden].
            // y = x @ W.T  →  matmul(x, W.transpose())
            let pre_w_t = mlx_rs::ops::transpose(&self.pre_projection)
                .context("draft_step: transpose(pre_projection)")?;
            let mut h =
                mlx_rs::ops::matmul(&x, &pre_w_t).context("draft_step: pre_projection matmul")?;

            // (2) 4 decoder layers (Q-only attention against trunk-shared KV).
            for lw in self.layers.iter() {
                let (k_shared, v_shared) = match lw.kind {
                    NativeGemma4MtpLayerType::Sliding => (trunk_k_sliding, trunk_v_sliding),
                    NativeGemma4MtpLayerType::Full => (trunk_k_full, trunk_v_full),
                };
                h = self.draft_layer_forward(&h, lw, k_shared, v_shared, position, eps)?;
            }

            // (3) Final norm + post_projection to trunk hidden space.
            //
            // Two outputs (mlx-vlm reference 2026-05-18):
            //   - `h_drafter_space` (= post-final_norm, shape [B,1,
            //     drafter_hidden=1024]) — feed to drafter's OWN embed_tokens
            //     as lm_head to obtain draft-token logits.
            //   - `h_backbone_space` (= post-post_projection, shape [B,1,
            //     backbone_hidden=2816]) — recurrent state for next
            //     draft step's `last_hidden_state` input.
            //
            // Earlier our code returned only `h_backbone_space` and used the
            // TRUNK's quantized lm_head + softcap on it. That was wrong on
            // three counts (mlx_vlm 0.5.0
            // `speculative.drafters.gemma4_assistant.Gemma4AssistantDraftModel.__call__`):
            //   (a) lm_head should be drafter's own `embed_tokens.as_linear`
            //       (tie_word_embeddings, drafter's bf16 vocab embed in 1024-dim),
            //       NOT trunk's lm_head (2816-dim quantized embed).
            //   (b) lm_head input is drafter-hidden `h_drafter_space`,
            //       NOT backbone-hidden `h_backbone_space`.
            //   (c) No softcap (reference omits it for the drafter logits).
            //
            // Net result of the prior bug: drafter logits ranged over a
            // 1024-dimensional subspace of the trunk's lm_head — collapsed
            // to a small set of "default" tokens (1595, 1852, 1902, 2889
            // attractor), producing 0% acceptance regardless of input.
            let h_drafter_space = mlx_rs::fast::rms_norm(&h, &self.final_norm, eps)
                .context("draft_step: model.norm")?;
            let post_w_t = mlx_rs::ops::transpose(&self.post_projection)
                .context("draft_step: transpose(post_projection)")?;
            let h_backbone_space = mlx_rs::ops::matmul(&h_drafter_space, &post_w_t)
                .context("draft_step: post_projection matmul")?;
            Ok((h_drafter_space, h_backbone_space))
        }

        /// Drafter's lm_head — applies the drafter's tied embed weights as
        /// a linear projection (tie_word_embeddings convention). No softcap
        /// (matches mlx-vlm reference). Input `h` must be in drafter hidden
        /// space (output of `draft_step`'s first tuple element).
        pub fn lm_head(&self, h_drafter_space: &Array) -> Result<Array> {
            // self.embed_tokens shape = [vocab, drafter_hidden]. as_linear:
            // logits = h @ embed.T → matmul(h, transpose(embed)).
            let embed_t = mlx_rs::ops::transpose(&self.embed_tokens)
                .context("lm_head: transpose(embed_tokens)")?;
            mlx_rs::ops::matmul(h_drafter_space, &embed_t).context("lm_head: matmul(h, embed.T)")
        }

        /// Single drafter layer forward. Q-only attention; K/V come from the
        /// caller (trunk-shared cache slice).
        fn draft_layer_forward(
            &self,
            h: &Array,
            lw: &ResolvedGemma4MtpLayerWeights,
            k_shared: &Array,
            v_shared: &Array,
            position: i32,
            eps: f32,
        ) -> Result<Array> {
            let tc = &self.config.text_config;
            let n_heads = tc.num_attention_heads as i32;
            // `n_kv` (sliding=8, full=2) isn't directly consumed: mlx
            // fast::scaled_dot_product_attention handles GQA when
            // n_heads / n_kv is an integer broadcast factor. Captured here
            // for documentation; underscore-prefixed to silence unused-warn.
            let (head_dim, _n_kv) = match lw.kind {
                NativeGemma4MtpLayerType::Sliding => {
                    (tc.head_dim as i32, tc.num_key_value_heads as i32)
                }
                NativeGemma4MtpLayerType::Full => (
                    tc.global_head_dim as i32,
                    tc.num_global_key_value_heads as i32,
                ),
            };

            // (a) input_layernorm → q_proj
            let residual = h.clone();
            let h_in = mlx_rs::fast::rms_norm(h, &lw.input_layernorm, eps)
                .context("draft_layer: input_layernorm")?;
            let q_w_t = mlx_rs::ops::transpose(&lw.attn.q_proj)
                .context("draft_layer: transpose(q_proj)")?;
            let q_flat =
                mlx_rs::ops::matmul(&h_in, &q_w_t).context("draft_layer: q_proj matmul")?;

            // (b) reshape Q to [B, 1, H, head_dim] then transpose to [B, H, 1, head_dim]
            let q_4d = mlx_rs::ops::reshape(&q_flat, &[1, 1, n_heads, head_dim])
                .context("draft_layer: reshape Q to 4D")?;
            let q_n = mlx_rs::fast::rms_norm(&q_4d, &lw.attn.q_norm, eps)
                .context("draft_layer: q_norm")?;
            let q_t = mlx_rs::ops::transpose_axes(&q_n, &[0, 2, 1, 3])
                .context("draft_layer: transpose Q for SDPA")?;

            // (c) RoPE
            let q_rope = match lw.kind {
                NativeGemma4MtpLayerType::Sliding => {
                    // Full rotation, freqs-less path (rope_dim == head_dim).
                    crate::native_rope::rope(
                        &q_t,
                        head_dim,
                        false,
                        self.rope_base_sliding,
                        1.0,
                        position,
                    )
                    .context("draft_layer: rope(Q) sliding")?
                }
                NativeGemma4MtpLayerType::Full => {
                    // Proportional RoPE — pre-built inf-tail freqs.
                    crate::native_rope::rope_with_freqs(
                        &q_t,
                        head_dim,
                        false,
                        1.0,
                        position,
                        &self.rope_freqs_full,
                    )
                    .context("draft_layer: rope_with_freqs(Q) full")?
                }
            };

            // (d) SDPA(Q, K_trunk, V_trunk). mlx fast::sdpa handles GQA when
            //     n_heads / n_kv is an integer (Sliding: 16/8=2, Full: 16/2=8).
            //     Decode S=1 with full cache → no mask needed (Q at position
            //     `position` attends to all of K/V which are at positions
            //     0..T-1 ≤ position).
            // Gemma 4 hard-codes scale=1.0 (NOT 1/sqrt(head_dim)) —
            // mlx-vlm reference `Attention.__init__: self.scale = 1.0`.
            // The q_norm RMSNorm absorbs the sqrt(head_dim) factor.
            let scale = 1.0_f32;
            let _ = head_dim; // silence unused if we kept the var name above
            let attn = mlx_rs::fast::scaled_dot_product_attention(
                &q_rope, k_shared, v_shared, scale, None, /* mask */
                None, /* sinks */
            )
            .context("draft_layer: scaled_dot_product_attention")?;

            // (e) transpose back + reshape + o_proj
            let attn_t = mlx_rs::ops::transpose_axes(&attn, &[0, 2, 1, 3])
                .context("draft_layer: transpose attn back")?;
            let attn_flat = mlx_rs::ops::reshape(&attn_t, &[1, 1, n_heads * head_dim])
                .context("draft_layer: reshape attn flat")?;
            let o_w_t = mlx_rs::ops::transpose(&lw.attn.o_proj)
                .context("draft_layer: transpose(o_proj)")?;
            let attn_out =
                mlx_rs::ops::matmul(&attn_flat, &o_w_t).context("draft_layer: o_proj matmul")?;

            // (f) post_attention_layernorm + residual
            let post_attn_n = mlx_rs::fast::rms_norm(&attn_out, &lw.post_attention_layernorm, eps)
                .context("draft_layer: post_attention_layernorm")?;
            let h = mlx_rs::ops::add(&residual, &post_attn_n)
                .context("draft_layer: +residual (attn)")?;
            let residual = h.clone();

            // (g) Dense MLP (pre_ff_norm → gate/up GeGLU → down → post_ff_norm)
            let h_n2 = mlx_rs::fast::rms_norm(&h, &lw.pre_feedforward_layernorm, eps)
                .context("draft_layer: pre_feedforward_layernorm")?;
            let gate_w_t = mlx_rs::ops::transpose(&lw.mlp.gate_proj)
                .context("draft_layer: transpose(gate_proj)")?;
            let up_w_t = mlx_rs::ops::transpose(&lw.mlp.up_proj)
                .context("draft_layer: transpose(up_proj)")?;
            let down_w_t = mlx_rs::ops::transpose(&lw.mlp.down_proj)
                .context("draft_layer: transpose(down_proj)")?;
            let gate = mlx_rs::ops::matmul(&h_n2, &gate_w_t).context("draft_layer: gate matmul")?;
            let up = mlx_rs::ops::matmul(&h_n2, &up_w_t).context("draft_layer: up matmul")?;
            // GeGLU = gelu_approx(gate) * up  (mirrors trunk dense_mlp).
            let dt = gate.dtype();
            let half = Array::from_f32(0.5)
                .as_dtype(dt)
                .context("draft_layer: const half")?;
            let one = Array::from_f32(1.0)
                .as_dtype(dt)
                .context("draft_layer: const one")?;
            let c3 = Array::from_f32(0.044715)
                .as_dtype(dt)
                .context("draft_layer: const c3")?;
            let coeff = Array::from_f32(0.797_884_6_f32)
                .as_dtype(dt)
                .context("draft_layer: const coeff")?;
            let x_sq = mlx_rs::ops::multiply(&gate, &gate).context("draft_layer: gelu x_sq")?;
            let x_cubed =
                mlx_rs::ops::multiply(&x_sq, &gate).context("draft_layer: gelu x_cubed")?;
            let inner_add = mlx_rs::ops::add(
                &gate,
                &mlx_rs::ops::multiply(&x_cubed, &c3).context("draft_layer: gelu cubed*c3")?,
            )
            .context("draft_layer: gelu inner_add")?;
            let scaled =
                mlx_rs::ops::multiply(&coeff, &inner_add).context("draft_layer: gelu scaled")?;
            let t = mlx_rs::ops::tanh(&scaled).context("draft_layer: gelu tanh")?;
            let one_plus_t = mlx_rs::ops::add(&one, &t).context("draft_layer: gelu 1+tanh")?;
            let half_gate =
                mlx_rs::ops::multiply(&half, &gate).context("draft_layer: gelu 0.5*gate")?;
            let gelu_g = mlx_rs::ops::multiply(&half_gate, &one_plus_t)
                .context("draft_layer: gelu (0.5g)(1+t)")?;
            let activated =
                mlx_rs::ops::multiply(&gelu_g, &up).context("draft_layer: gelu * up")?;
            let down =
                mlx_rs::ops::matmul(&activated, &down_w_t).context("draft_layer: down matmul")?;
            let mlp_out = mlx_rs::fast::rms_norm(&down, &lw.post_feedforward_layernorm, eps)
                .context("draft_layer: post_feedforward_layernorm")?;

            // (h) +residual, ×layer_scalar
            let h =
                mlx_rs::ops::add(&residual, &mlp_out).context("draft_layer: +residual (mlp)")?;
            let h = mlx_rs::ops::multiply(&h, &lw.layer_scalar)
                .context("draft_layer: × layer_scalar")?;
            Ok(h)
        }
    }

    // ───────────────────────── helpers ─────────────────────────

    fn load_safetensors_dir(dir: &Path) -> Result<HashMap<String, Array>> {
        let mut shard_paths = std::fs::read_dir(dir)
            .with_context(|| format!("load_safetensors_dir: read_dir {}", dir.display()))?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|p| p.is_file() && p.extension().is_some_and(|ext| ext == "safetensors"))
            .collect::<Vec<_>>();
        shard_paths.sort();
        if shard_paths.is_empty() {
            return Err(anyhow!(
                "load_safetensors_dir: no *.safetensors under {}",
                dir.display()
            ));
        }
        let mut tensors: HashMap<String, Array> = HashMap::new();
        for shard in shard_paths {
            let map = Array::load_safetensors(&shard)
                .map_err(|err| anyhow!("load_safetensors({}) failed: {err}", shard.display()))?;
            tensors.reserve(map.len());
            for (k, v) in map {
                if tensors.insert(k.clone(), v).is_some() {
                    return Err(anyhow!(
                        "load_safetensors_dir: duplicate key `{k}` in {}",
                        dir.display()
                    ));
                }
            }
        }
        Ok(tensors)
    }

    fn require_take(map: &mut HashMap<String, Array>, key: &str) -> Result<Array> {
        map.remove(key)
            .ok_or_else(|| anyhow!("require_take: missing key `{key}` in drafter checkpoint"))
    }
}

#[cfg(not(feature = "mlx-native"))]
pub mod imp {}
