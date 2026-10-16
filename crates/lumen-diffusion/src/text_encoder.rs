//! Native FLUX.2 text encoder: Mistral-Small-3.2-24B, text-only (Phase 1).
//!
//! A faithful mechanical port of the Mistral decoder stack used by FLUX.2-dev
//! to embed the prompt. FLUX does **not** run the LM head; instead it captures
//! the raw hidden states *after* decoder layers 10, 20 and 30, stacks them, and
//! flattens to `[1, T, 3 * hidden]`. We reproduce exactly that.
//!
//! Config (Mistral-Small-3.2-24B): `hidden = 5120`, `n_layers = 40`,
//! `n_heads = 32`, `n_kv_heads = 8` (GQA), `head_dim = 128`,
//! `rope_theta = 1e9`, `rms_norm_eps = 1e-5`, `intermediate = 32768`,
//! `vocab = 131072`, no sliding window.
//!
//! Weights come in two forms, auto-detected per tensor by the presence of a
//! `.scales` sibling (exactly like [`crate::dit`]):
//!
//! - **mlx-lm affine 4-bit** (`group_size = 64`, `bits = 4`): every projection
//!   stays in its packed `{weight(U32),scales,biases}` form and is applied via
//!   [`crate::ops::qlinear`]; `embed_tokens` is dequantized and gathered.
//!   Source: `mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit`.
//! - **dense BF16** (the official `black-forest-labs/FLUX.2-dev/text_encoder/`):
//!   each projection is a single BF16 `.weight` `[out,in]` applied as
//!   `x @ Wᵀ`; `embed_tokens.weight` is a dense BF16 `[vocab,hidden]` table
//!   gathered directly by token id (no dequant). For 128 GB machines.
//!
//! Both forms share identical key names and architecture, spread across
//! safetensors shards keyed by `model.safetensors.index.json`. The 4-bit vs
//! bf16 choice is purely `.scales`-presence — the same loader handles both.
//!
//! ## Mask (causal + left-pad)
//!
//! The 512-token prompt is **left-padded** (pad id 11) with the real tokens at
//! `first_real..T`. The additive attention mask allows position `i` to attend
//! to `j` iff `first_real <= j <= i`, else `-1e9`. A plain causal (or
//! bidirectional) mask does *not* match the validated Swift reference.

#[cfg(feature = "mlx-native")]
mod imp {
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};

    use anyhow::{Context, Result};
    use mlx_rs::Array;

    use crate::ops::qlinear::{BITS, GROUP_SIZE, Linear};
    use crate::ops::silu::silu;
    use crate::weights::Weights;

    // ── model config ──────────────────────────────────────────────────────────
    pub const HIDDEN: i32 = 5120;
    pub const N_HEADS: i32 = 32;
    pub const N_KV_HEADS: i32 = 8;
    pub const HEAD_DIM: i32 = 128;
    pub const ROPE_THETA: f32 = 1e9;
    pub const RMS_EPS: f32 = 1e-5;
    /// Hidden states captured after these (1-based) decoder layers.
    pub const CAPTURE_LAYERS: [usize; 3] = [10, 20, 30];
    /// Only the layers up to the deepest capture point are needed — FLUX.2
    /// extracts hidden states after layer 30, so decoder layers 31..=40, the
    /// final `norm`, and `lm_head` are never used and are skipped at load. This
    /// trims the encoder's resident footprint (~13 GB → ~10 GB) and its forward
    /// (40 → 30 layers), enabling encoder + DiT co-residence on a 36 GB Mac.
    /// = `max(CAPTURE_LAYERS)`.
    pub const LOAD_LAYERS: usize = 30;

    // ── weight bundles ──────────────────────────────────────────────────────────

    struct LayerW {
        input_layernorm: Array, // [hidden]
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        post_attention_layernorm: Array, // [hidden]
        gate_proj: Linear,
        up_proj: Linear,
        down_proj: Linear,
    }

    /// All weights for the Mistral text encoder.
    pub struct MistralEncoder {
        /// Token embedding table `[vocab, hidden]` (f32, eval'd). For the 4-bit
        /// encoder this is the dequantized table; for bf16 it is the dense table
        /// loaded directly. Either way `embed` just gathers rows by token id.
        embed_table: Array,
        layers: Vec<LayerW>,
    }

    impl MistralEncoder {
        /// Load the encoder from a snapshot directory. Works for BOTH the
        /// mlx-lm 4-bit dir (`mlx-community/...-4bit`, sharded U32 + `.scales`)
        /// and the official bf16 dir
        /// (`black-forest-labs/FLUX.2-dev/text_encoder/`, sharded BF16) — the
        /// quantized/dense choice is per-tensor by `.scales` presence.
        pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
            let m = Weights::open_sharded(dir)?;
            let pfx = "language_model.model";

            // Embedding table → [vocab, hidden] f32. 4-bit: dequantize the
            // packed `{weight,scales,biases}`. bf16: the dense `.weight` is
            // already the [vocab,hidden] table — load it as-is (no dequant).
            // `embed` gathers rows by token id from whichever form.
            let embed_table = if m.has(&format!("{pfx}.embed_tokens.scales")) {
                let embed_w = m.u32(&format!("{pfx}.embed_tokens.weight"))?;
                let embed_s = m.f32(&format!("{pfx}.embed_tokens.scales"))?;
                let embed_b = m.f32(&format!("{pfx}.embed_tokens.biases"))?;
                mlx_rs::ops::dequantize(&embed_w, &embed_s, &embed_b, GROUP_SIZE, BITS)
                    .context("dequantize embed_tokens")?
            } else {
                m.f32(&format!("{pfx}.embed_tokens.weight"))
                    .context("load dense bf16 embed_tokens")?
            };
            embed_table.eval().context("eval embed_table")?;

            // Load only layers 0..LOAD_LAYERS (deepest capture = layer 30); the
            // remaining layers / final norm / lm_head are never read. The
            // `linear()` loader auto-detects 4-bit vs dense bf16 per projection.
            let mut layers = Vec::with_capacity(LOAD_LAYERS);
            for li in 0..LOAD_LAYERS {
                let lp = format!("{pfx}.layers.{li}");
                layers.push(LayerW {
                    input_layernorm: m.f32(&format!("{lp}.input_layernorm.weight"))?,
                    q_proj: m.linear(&format!("{lp}.self_attn.q_proj"))?,
                    k_proj: m.linear(&format!("{lp}.self_attn.k_proj"))?,
                    v_proj: m.linear(&format!("{lp}.self_attn.v_proj"))?,
                    o_proj: m.linear(&format!("{lp}.self_attn.o_proj"))?,
                    post_attention_layernorm: m
                        .f32(&format!("{lp}.post_attention_layernorm.weight"))?,
                    gate_proj: m.linear(&format!("{lp}.mlp.gate_proj"))?,
                    up_proj: m.linear(&format!("{lp}.mlp.up_proj"))?,
                    down_proj: m.linear(&format!("{lp}.mlp.down_proj"))?,
                });
            }
            Ok(Self {
                embed_table,
                layers,
            })
        }

        /// Embedding lookup: gather rows of the dequantized table by token id.
        /// `ids`: `&[i32]` length `T` → `[1, T, hidden]`.
        fn embed(&self, ids: &[i32]) -> Result<Array> {
            let idx = Array::from_slice(ids, &[ids.len() as i32]);
            // take(table, idx, axis=0) -> [T, hidden]
            let rows = mlx_rs::ops::indexing::take_axis(&self.embed_table, &idx, 0)
                .context("embed: take rows")?;
            rows.reshape(&[1, ids.len() as i32, HIDDEN])
                .context("embed: reshape to [1,T,hidden]")
        }

        /// Build the additive causal + left-pad mask `[1, 1, T, T]`:
        /// 0.0 where `first_real <= j <= i`, else `-1e9`.
        fn build_mask(t: usize, first_real: usize) -> Array {
            let neg = -1e9f32;
            let mut data = vec![neg; t * t];
            for i in 0..t {
                for j in first_real..=i {
                    data[i * t + j] = 0.0;
                }
            }
            Array::from_slice(&data, &[1, 1, t as i32, t as i32])
        }

        /// Full encode. `ids`: length-`T` token ids; `first_real`: first
        /// real-token index for the left-pad mask. Returns the FLUX text
        /// embedding `[1, T, 3 * hidden]` (layers 10/20/30 stacked & flattened),
        /// and (for staged debugging) the hidden state after layer 10.
        pub fn encode_with_stage(&self, ids: &[i32], first_real: usize) -> Result<(Array, Array)> {
            let t = ids.len();
            let ti = t as i32;
            let mask = Self::build_mask(t, first_real);
            mask.eval().context("eval mask")?;
            let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();

            let mut h = self.embed(ids)?; // [1, T, hidden]
            // hidden_states[0] = embeddings; hidden_states[k] = after layer k.
            let mut captured: HashMap<usize, Array> = HashMap::new();
            let mut h10: Option<Array> = None;

            for (li, lw) in self.layers.iter().enumerate() {
                let layer_no = li + 1; // 1-based: output AFTER this layer

                // ── self-attention ──
                let r = h.clone();
                let x = mlx_rs::fast::rms_norm(&h, &lw.input_layernorm, RMS_EPS)
                    .context("attn rms_norm")?;

                let q = lw.q_proj.apply(&x)?; // [1,T,4096]
                let k = lw.k_proj.apply(&x)?; // [1,T,1024]
                let v = lw.v_proj.apply(&x)?; // [1,T,1024]

                // [1,T,H,D] -> [1,H,T,D]
                let q = q
                    .reshape(&[1, ti, N_HEADS, HEAD_DIM])?
                    .transpose_axes(&[0, 2, 1, 3])?;
                let k = k
                    .reshape(&[1, ti, N_KV_HEADS, HEAD_DIM])?
                    .transpose_axes(&[0, 2, 1, 3])?;
                let v = v
                    .reshape(&[1, ti, N_KV_HEADS, HEAD_DIM])?
                    .transpose_axes(&[0, 2, 1, 3])?;

                // RoPE (non-traditional, base 1e9, offset 0, scale 1).
                let q = mlx_rs::fast::rope(&q, HEAD_DIM, false, Some(ROPE_THETA), 1.0, 0, None)
                    .context("rope q")?;
                let k = mlx_rs::fast::rope(&k, HEAD_DIM, false, Some(ROPE_THETA), 1.0, 0, None)
                    .context("rope k")?;

                // GQA: 32 q heads, 8 kv heads — mlx sdpa broadcasts kv.
                let attn =
                    mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, scale, &mask, None)
                        .context("sdpa")?;

                // [1,H,T,D] -> [1,T,H,D] -> [1,T,4096]
                let attn =
                    attn.transpose_axes(&[0, 2, 1, 3])?
                        .reshape(&[1, ti, N_HEADS * HEAD_DIM])?;
                let attn = lw.o_proj.apply(&attn)?; // [1,T,hidden]
                h = r.add(&attn).context("attn residual add")?;

                // ── SwiGLU MLP ──
                let r2 = h.clone();
                let x = mlx_rs::fast::rms_norm(&h, &lw.post_attention_layernorm, RMS_EPS)
                    .context("mlp rms_norm")?;
                let g = silu(&lw.gate_proj.apply(&x)?)?;
                let u = lw.up_proj.apply(&x)?;
                let gu = g.multiply(&u).context("swiglu gate*up")?;
                let down = lw.down_proj.apply(&gu)?;
                h = r2.add(&down).context("mlp residual add")?;

                // Eval each layer's hidden to bound the lazy graph / memory.
                h.eval().context("eval layer hidden")?;

                if layer_no == 10 {
                    h10 = Some(h.clone());
                }
                if CAPTURE_LAYERS.contains(&layer_no) {
                    captured.insert(layer_no, h.clone());
                }
            }

            // stack [10],[20],[30] on axis=1 -> [1,3,T,hidden]
            let stacked = mlx_rs::ops::stack_axis(
                &[
                    captured.remove(&10).context("missing layer 10")?,
                    captured.remove(&20).context("missing layer 20")?,
                    captured.remove(&30).context("missing layer 30")?,
                ],
                1,
            )
            .context("stack captured layers")?;
            // transpose (0,2,1,3) -> [1,T,3,hidden] -> reshape [1,T,3*hidden]
            let out = stacked
                .transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[1, ti, 3 * HIDDEN])?;
            out.eval().context("eval encoder output")?;

            Ok((out, h10.context("layer 10 not captured")?))
        }

        /// Convenience: encode and return only the `[1, T, 3*hidden]` embedding.
        pub fn encode(&self, ids: &[i32], first_real: usize) -> Result<Array> {
            Ok(self.encode_with_stage(ids, first_real)?.0)
        }
    }

    /// Default text-encoder dir: the 4-bit MLX Mistral-Small-3.2-24B snapshot
    /// resolved from the local HuggingFace cache. Falls back to the repo id (so
    /// load fails with a clear message) if it isn't downloaded.
    pub fn default_weights_dir() -> PathBuf {
        crate::hf_cache::snapshot_dir_or_id(crate::repos::ENCODER_4BIT)
    }
}

#[cfg(feature = "mlx-native")]
pub use imp::{MistralEncoder, default_weights_dir};

// ── parity test against the validated golden reference (/tmp/mistral_*.bin) ───
#[cfg(all(test, feature = "mlx-native"))]
mod parity_tests {
    use super::imp::MistralEncoder;
    use crate::test_util::{read_f32_bin, read_i32_bin};

    /// max-abs, mean-abs, cosine over the slice of `[T, D]` rows in
    /// `[row_lo, row_hi)` (real-token positions only).
    fn compare_rows(
        label: &str,
        obs: &[f32],
        exp: &[f32],
        t: usize,
        d: usize,
        row_lo: usize,
        row_hi: usize,
    ) -> (f32, f32, f64) {
        assert_eq!(obs.len(), t * d, "{label}: obs size");
        assert_eq!(exp.len(), t * d, "{label}: exp size");
        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f64;
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        let mut n = 0usize;
        for row in row_lo..row_hi {
            for c in 0..d {
                let g = obs[row * d + c];
                let e = exp[row * d + c];
                let diff = (g - e).abs();
                if diff > max_abs {
                    max_abs = diff;
                }
                sum_abs += diff as f64;
                dot += (g as f64) * (e as f64);
                na += (g as f64) * (g as f64);
                nb += (e as f64) * (e as f64);
                n += 1;
            }
        }
        let mean_abs = (sum_abs / n as f64) as f32;
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
        println!(
            "{label}: rows[{row_lo}..{row_hi}) max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} cosine={cos:.6}"
        );
        (max_abs, mean_abs, cos)
    }

    #[test]
    #[ignore = "needs a Metal device AND the /tmp/mistral_*.bin dump set (tokens, \
                firstreal, h10, embed_ref); no committed script produces them — \
                see crates/lumen-mlx/tests/golden/README.md"]
    fn mistral_encoder_matches_reference() {
        const T: usize = 512;
        let ids = read_i32_bin("/tmp/mistral_tokens.bin");
        assert_eq!(ids.len(), T, "token count");
        let first_real = read_i32_bin("/tmp/mistral_firstreal.bin")[0] as usize;
        assert_eq!(first_real, 465, "first_real");

        let enc =
            MistralEncoder::load(super::imp::default_weights_dir()).expect("load encoder weights");
        let (out, h10) = enc.encode_with_stage(&ids, first_real).expect("encode");
        assert_eq!(out.shape(), &[1, T as i32, 3 * 5120], "output shape");

        // ── staged: hidden after layer 10 vs /tmp/mistral_h10.bin ──
        h10.eval().expect("eval h10");
        let h10_obs: &[f32] = h10.as_slice();
        let h10_exp = read_f32_bin("/tmp/mistral_h10.bin");
        let (_, _, h10_cos) = compare_rows("h10", h10_obs, &h10_exp, T, 5120, first_real, T);
        assert!(
            h10_cos > 0.999,
            "staged layer-10 cosine {h10_cos:.6} <= 0.999 — divergence before/at layer 10"
        );

        // ── final: full embedding vs /tmp/mistral_embed_ref.bin ──
        out.eval().expect("eval out");
        let obs: &[f32] = out.as_slice();
        let exp = read_f32_bin("/tmp/mistral_embed_ref.bin");
        let (max_abs, mean_abs, cos) = compare_rows("embed", obs, &exp, T, 3 * 5120, first_real, T);

        // outlier-dim max-abs may be a few units (cross-impl bf16/quant noise);
        // do NOT gate on it. Gate on cosine + mean-abs.
        assert!(cos > 0.999, "embedding cosine {cos:.6} <= 0.999");
        assert!(
            mean_abs < 0.05,
            "embedding mean-abs {mean_abs:.6e} >= 0.05 (max_abs {max_abs:.6e})"
        );
        println!(
            "Phase 1 gate PASSED: cosine={cos:.6} mean_abs={mean_abs:.6e} max_abs={max_abs:.6e}"
        );
    }
}
