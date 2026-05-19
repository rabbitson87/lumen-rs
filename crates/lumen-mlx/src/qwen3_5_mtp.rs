//! Multi-Token Prediction (MTP) head for Qwen3.5/3.6 — mlx-native port.
//!
//! Phase 2 Session 1 of the MTP rework: defines the block structure +
//! `MtpBlock::forward` on the mlx-native runner. Runner integration
//! (`mtp_step` orchestration in `qwen3_5_moe.rs`), the HF-native loader, and
//! the drafter lifecycle land in subsequent sessions.
//!
//! ## Architecture (HF original Qwen3.6 `mtp.*` namespace)
//!
//! ```text
//!   tok_embd = embed_tokens(x_{p+1})           ── caller-supplied (trunk's)
//!   e_norm   = enorm(tok_embd)
//!   h_norm   = hnorm(h_p)                      ── trunk's pre-final-norm
//!   concat   = cat([e_norm, h_norm], -1)       ── [B, T, 2*hidden]
//!   cur      = eh_proj @ concat                ── [B, T, hidden]
//!   res_attn = cur
//!   cur      = input_layernorm(cur)
//!   cur      = self_attention(cur)             ── owns its OWN KV cache
//!   cur     += res_attn
//!   res_ffn  = cur
//!   cur      = mlp(post_attention_layernorm(cur))
//!   cur     += res_ffn
//!   new_h_pre = cur                            ── seeds next AR draft step
//!   norm_out = (shared_head_norm || trunk_final_norm)(cur)
//!   logits   = trunk_lm_head(norm_out)         ── always shared on Qwen3.6
//! ```
//!
//! Differences from `gemma4_mtp.rs`:
//! - Qwen3.6 MTP has its **own** full self-attention with **its own KV
//!   cache** (Gemma4's drafter shares the trunk's KV via Q-only attention).
//! - eh_proj reduces `2*hidden -> hidden` (Gemma4's pre_proj goes from
//!   `drafter_hidden*2 -> drafter_hidden` where drafter_hidden differs from
//!   trunk hidden).
//! - Output head always shares trunk's `lm_head` on the Qwen3.6 checkpoints
//!   we ship (`mtp_use_dedicated_embeddings=false`).

use anyhow::{Context, Result, anyhow};
use mlx_rs::Array;
use std::ffi::CStr;

use crate::native_attention::sdpa;
use crate::native_cache::NativeKvCache;
use crate::native_norm::rms_norm;
use crate::native_quant::quantized_matmul_with_mode;
use crate::native_rope::rope;

// ─────────────────────────────────────────────────────────────────────────────
// Config + weight bundles
// ─────────────────────────────────────────────────────────────────────────────

/// Static dimensions sourced from the trunk's `text_config` plus the MTP head
/// specifics. The MTP block reuses trunk attention dims (num_heads, num_kv,
/// head_dim) — `mtp.layers.0.self_attn.*` weights are validated against these
/// at load time.
#[derive(Debug, Clone, Copy)]
pub struct Qwen35MtpDims {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    /// Same as trunk's. RoPE base.
    pub rope_theta: f32,
    /// `partial_rotary_factor * head_dim`. Qwen3.6 uses partial-rotary
    /// GPT-NeoX RoPE — only the first `rope_dim` features rotate.
    pub rope_dim: usize,
    pub rms_norm_eps: f32,
    /// True when the trunk Q proj has the doubled output (queries+gate
    /// halves) and an output gate is applied. Same convention as
    /// `qwen3_5_moe.rs` full-attn layer.
    pub attn_output_gate: bool,
}

/// Pre-resolved (load-time) linear projection. Mirrors
/// `qwen3_5_moe::ResolvedLinear` so MTP load paths can reuse the same
/// dequant pipeline.
pub struct Qwen35MtpLinear {
    pub weight: Array,
    pub scales: Array,
    pub biases: Option<Array>,
    pub group_size: i32,
    pub bits: i32,
    pub mode: &'static CStr,
}

/// Self-attention weights for the MTP layer. Layout matches the trunk's
/// full-attn layer — `mtp.layers.0.self_attn.{q,k,v,o}_proj.weight` plus
/// per-head q_norm / k_norm.
pub struct Qwen35MtpAttnWeights {
    pub q_proj: Qwen35MtpLinear,
    pub k_proj: Qwen35MtpLinear,
    pub v_proj: Qwen35MtpLinear,
    pub o_proj: Qwen35MtpLinear,
    pub q_norm_weight: Array, // [head_dim]
    pub k_norm_weight: Array, // [head_dim]
}

/// Dense MLP weights — SwiGLU with separate gate/up projections.
///
/// 27B Qwen3.6 uses Dense in the MTP block (and in every trunk layer).
/// 35B-A3B uses MoE in the trunk's middle layers but its MTP block is
/// also Dense (per HF checkpoint inspection — the MTP head is intentionally
/// simpler than a full trunk layer to keep cycle latency down).
pub struct Qwen35MtpDenseMlp {
    pub gate_proj: Qwen35MtpLinear,
    pub up_proj: Qwen35MtpLinear,
    pub down_proj: Qwen35MtpLinear,
}

/// MLP variant. Phase 2 ships the Dense path first (covers both 27B-style
/// and the 35B-A3B MTP head). MoE variant slot is reserved for any future
/// checkpoint that publishes a routed-MoE MTP head.
pub enum Qwen35MtpMlp {
    Dense(Qwen35MtpDenseMlp),
    // Reserved: Moe(ResolvedMoeWeights),
}

/// Full resolved MTP block — input, attention, MLP, output norm/head.
/// Constructed once at server startup by the loader from the HF-original
/// `mtp.*` shards (see future `qwen3_5_mtp_loader.rs`).
pub struct Qwen35MtpBlock {
    pub dims: Qwen35MtpDims,
    // ── MTP-specific projections + norms (pre-attention) ─────────────────
    /// `mtp.fc.weight` — [hidden, 2*hidden]. concat(e_norm,h_norm) -> hidden.
    pub eh_proj: Qwen35MtpLinear,
    /// `mtp.pre_fc_norm_embedding.weight` — [hidden]
    pub enorm: Array,
    /// `mtp.pre_fc_norm_hidden.weight` — [hidden]
    pub hnorm: Array,
    // ── Standard decoder block (1 layer) ─────────────────────────────────
    /// `mtp.layers.0.input_layernorm.weight` — [hidden]
    pub input_layernorm: Array,
    pub attention: Qwen35MtpAttnWeights,
    /// `mtp.layers.0.post_attention_layernorm.weight` — [hidden]
    pub post_attention_layernorm: Array,
    pub mlp: Qwen35MtpMlp,
    // ── Output head ──────────────────────────────────────────────────────
    /// `mtp.norm.weight` — [hidden] — used instead of trunk's final_norm
    /// when the checkpoint provides it. `None` falls back to trunk's.
    pub shared_head_norm: Option<Array>,
    // shared_head_head is **always** the trunk's lm_head on Qwen3.6 —
    // the checkpoint never ships `mtp.shared_head_head.*`. Caller threads
    // the trunk's lm_head through `forward()`.
}

// ─────────────────────────────────────────────────────────────────────────────
// Forward
// ─────────────────────────────────────────────────────────────────────────────

impl Qwen35MtpBlock {
    /// One MTP block forward pass.
    ///
    /// * `embeds`     — `embed_tokens(x_{p+1})`, shape `[B, T, hidden]`. Caller
    ///                  pre-embeds via the trunk's embedding table.
    /// * `h_pre`      — trunk's pre-final-norm hidden at the same T positions,
    ///                  shape `[B, T, hidden]`. Same dtype as `embeds`.
    /// * `cache`      — the MTP block's own KV cache (lifetime: per seq).
    ///                  Position offset is read from `cache.offset()`.
    /// * `causal`     — `true` when `T > 1` (verify-batch or prefill-mirror);
    ///                  `false` for AR decode (single-row queries).
    /// * `trunk_final_norm_weight` — trunk's `final_norm.weight`, used iff
    ///                  this block lacks its own `shared_head_norm`.
    /// * `trunk_lm_head` — trunk's lm_head linear. Always used (Qwen3.6
    ///                  doesn't ship a dedicated MTP head).
    ///
    /// Returns `(logits, new_h_pre)`:
    ///   * `logits[..., :, :vocab]` — predictions for `x_{p+2}` per row.
    ///   * `new_h_pre[..., :, hidden]` — post-residual hidden that becomes the
    ///     next `h_p` for the AR draft loop.
    pub fn forward(
        &self,
        embeds: &Array,
        h_pre: &Array,
        cache: &mut NativeKvCache,
        causal: bool,
        trunk_final_norm_weight: &Array,
        trunk_lm_head: &Qwen35MtpLinear,
    ) -> Result<(Array, Array)> {
        // ── shape sanity ─────────────────────────────────────────────────
        let e_shape = embeds.shape();
        let h_shape = h_pre.shape();
        if e_shape != h_shape {
            return Err(anyhow!(
                "Qwen35MtpBlock::forward: embeds.shape {:?} != h_pre.shape {:?}",
                e_shape,
                h_shape,
            ));
        }
        if embeds.ndim() != 3 {
            return Err(anyhow!(
                "Qwen35MtpBlock::forward: expected rank-3 [B, T, hidden], got ndim={}",
                embeds.ndim(),
            ));
        }
        let b = embeds.shape()[0];
        let t = embeds.shape()[1];
        let hidden = self.dims.hidden_size as i32;
        let eps = self.dims.rms_norm_eps;

        // ── (1) enorm + hnorm + concat ───────────────────────────────────
        let e_n = rms_norm(embeds, &self.enorm, eps)
            .context("Qwen35MtpBlock: enorm failed")?;
        let h_n = rms_norm(h_pre, &self.hnorm, eps)
            .context("Qwen35MtpBlock: hnorm failed")?;
        // DeepSeek-V3 / Qwen3.6 convention: [embed_norm, hidden_norm].
        let concat = mlx_rs::ops::concatenate_axis(&[&e_n, &h_n], -1)
            .context("Qwen35MtpBlock: concat(e_norm, h_norm) failed")?;

        // ── (2) eh_proj: [2*hidden] -> [hidden] ──────────────────────────
        let mut cur = self.linear_pre(&self.eh_proj, &concat)
            .context("Qwen35MtpBlock: eh_proj failed")?;
        cur = self.to_f32(cur, "eh_proj_out")?;

        // ── (3) attention sub-block: pre-norm + self_attn + residual ─────
        let res_attn = cur.clone();
        let cur_n = rms_norm(&cur, &self.input_layernorm, eps)
            .context("Qwen35MtpBlock: input_layernorm failed")?;
        let attn_out = self
            .self_attention_forward(&cur_n, b, t, causal, cache)
            .context("Qwen35MtpBlock: self_attention_forward failed")?;
        cur = mlx_rs::ops::add(&attn_out, &res_attn)
            .context("Qwen35MtpBlock: residual add (attn) failed")?;

        // ── (4) MLP sub-block: pre-norm + mlp + residual ─────────────────
        let res_ffn = cur.clone();
        let cur_n = rms_norm(&cur, &self.post_attention_layernorm, eps)
            .context("Qwen35MtpBlock: post_attention_layernorm failed")?;
        let mlp_out = match &self.mlp {
            Qwen35MtpMlp::Dense(d) => self
                .dense_mlp_forward(&cur_n, d)
                .context("Qwen35MtpBlock: dense_mlp_forward failed")?,
        };
        cur = mlx_rs::ops::add(&mlp_out, &res_ffn)
            .context("Qwen35MtpBlock: residual add (ffn) failed")?;

        // ── (5) stash new_h_pre BEFORE head norm ─────────────────────────
        // The AR draft loop pairs this with the next sampled token id.
        let new_h_pre = cur.clone();

        // ── (6) head norm + lm_head ──────────────────────────────────────
        let norm_w = match &self.shared_head_norm {
            Some(w) => w,
            None => trunk_final_norm_weight,
        };
        let norm_out = rms_norm(&cur, norm_w, eps)
            .context("Qwen35MtpBlock: head norm failed")?;
        let logits = self
            .linear_pre(trunk_lm_head, &norm_out)
            .context("Qwen35MtpBlock: lm_head failed")?;

        let _ = hidden; // dims sanity check above already locked it
        Ok((logits, new_h_pre))
    }

    /// Self-attention with the block's own KV cache. Same structure as the
    /// trunk's `layer_full_attn_forward` in `qwen3_5_moe.rs` — same q/gate
    /// split + q_norm/k_norm + partial-rotary RoPE + GQA SDPA.
    fn self_attention_forward(
        &self,
        x: &Array,
        b: i32,
        t: i32,
        causal: bool,
        cache: &mut NativeKvCache,
    ) -> Result<Array> {
        let num_heads = self.dims.num_attention_heads as i32;
        let num_kv = self.dims.num_key_value_heads as i32;
        let head_dim = self.dims.head_dim as i32;
        let rope_dim = self.dims.rope_dim as i32;
        let scale = (self.dims.head_dim as f32).powf(-0.5);
        let base_theta = self.dims.rope_theta;
        let eps = self.dims.rms_norm_eps;
        let attn = &self.attention;

        // (1) q/k/v projections.
        let q_proj_raw = self.linear_pre(&attn.q_proj, x)?;
        let q_proj = self.to_f32(q_proj_raw, "mtp.q_proj")?;
        let k_proj_raw = self.linear_pre(&attn.k_proj, x)?;
        let k_proj = self.to_f32(k_proj_raw, "mtp.k_proj")?;
        let v_proj_raw = self.linear_pre(&attn.v_proj, x)?;
        let v_proj = self.to_f32(v_proj_raw, "mtp.v_proj")?;

        // (2a) Reshape q to [B, T, num_heads, head_dim * (gate?2:1)] and
        //      split into (queries, gate) when attn_output_gate is on.
        let q_last_dim = if self.dims.attn_output_gate {
            head_dim * 2
        } else {
            head_dim
        };
        let q_4d = mlx_rs::ops::reshape(&q_proj, &[b, t, num_heads, q_last_dim])
            .context("mtp.q reshape")?;
        let (queries, gate_opt) = if self.dims.attn_output_gate {
            let parts = mlx_rs::ops::split(&q_4d, 2, -1)
                .context("mtp.q split (queries|gate)")?;
            if parts.len() != 2 {
                return Err(anyhow!(
                    "mtp.q split returned {} parts (want 2)",
                    parts.len()
                ));
            }
            let mut it = parts.into_iter();
            let q = it.next().expect("mtp.q split[0] missing");
            let g_4d = it.next().expect("mtp.q split[1] missing");
            let g = mlx_rs::ops::reshape(&g_4d, &[b, t, num_heads * head_dim])
                .context("mtp.gate reshape")?;
            (q, Some(g))
        } else {
            (q_4d, None)
        };

        // (3) Per-head q_norm + transpose to [B, num_heads, T, head_dim].
        let queries_n = rms_norm(&queries, &attn.q_norm_weight, eps)?;
        let queries_t = mlx_rs::ops::transpose_axes(&queries_n, &[0, 2, 1, 3])
            .context("mtp.queries transpose")?;

        // (2b) k/v: [B, T, num_kv, head_dim] -> k_norm -> transpose.
        let k_4d = mlx_rs::ops::reshape(&k_proj, &[b, t, num_kv, head_dim])
            .context("mtp.k reshape")?;
        let k_n = rms_norm(&k_4d, &attn.k_norm_weight, eps)?;
        let k_t = mlx_rs::ops::transpose_axes(&k_n, &[0, 2, 1, 3])
            .context("mtp.k transpose")?;

        let v_4d = mlx_rs::ops::reshape(&v_proj, &[b, t, num_kv, head_dim])
            .context("mtp.v reshape")?;
        let v_t = mlx_rs::ops::transpose_axes(&v_4d, &[0, 2, 1, 3])
            .context("mtp.v transpose")?;

        // (4) Partial-rotary RoPE on queries + keys.
        let offset = cache.offset() as i32;
        let q_rope = rope(&queries_t, rope_dim, false, base_theta, 1.0, offset)?;
        let k_rope = rope(&k_t, rope_dim, false, base_theta, 1.0, offset)?;

        // (5) Append to MTP block's own KV cache + fetch full history.
        let (k_full, v_full) = cache.update_and_fetch(&k_rope, &v_t)?;

        // (6) GQA SDPA.
        let attn_out = sdpa(&q_rope, &k_full, &v_full, scale, causal)?;

        // (7) Reshape to [B, T, num_heads * head_dim] and apply optional gate.
        let attn_back = mlx_rs::ops::transpose_axes(&attn_out, &[0, 2, 1, 3])
            .context("mtp.attn_out transpose")?;
        let attn_flat =
            mlx_rs::ops::reshape(&attn_back, &[b, t, num_heads * head_dim])
                .context("mtp.attn_out reshape")?;
        let gated = match gate_opt {
            Some(gate) => {
                let gate_sig = mlx_rs::ops::sigmoid(&gate)
                    .context("mtp.sigmoid(gate)")?;
                mlx_rs::ops::multiply(&attn_flat, &gate_sig)
                    .context("mtp.gated multiply")?
            }
            None => attn_flat,
        };

        // (8) o_proj.
        let o_proj_raw = self.linear_pre(&attn.o_proj, &gated)?;
        self.to_f32(o_proj_raw, "mtp.o_proj")
    }

    /// Dense SwiGLU MLP. Same dispatch shape as the trunk's `shared_mlp`
    /// path in `native_moe::shared_mlp`: separate gate_proj + up_proj +
    /// silu(gate) * up + down_proj.
    fn dense_mlp_forward(
        &self,
        x: &Array,
        w: &Qwen35MtpDenseMlp,
    ) -> Result<Array> {
        let gate = self.linear_pre(&w.gate_proj, x)?;
        let gate = self.to_f32(gate, "mtp.mlp.gate")?;
        let up = self.linear_pre(&w.up_proj, x)?;
        let up = self.to_f32(up, "mtp.mlp.up")?;
        // SwiGLU: silu(gate) * up == sigmoid(gate) * gate * up. Mirrors the
        // expansion used in `native_moe::swiglu_compiled_inner` so a future
        // fuse pass can swap in the compiled-graph variant trivially.
        let sig = mlx_rs::ops::sigmoid(&gate)
            .context("mtp.mlp sigmoid(gate)")?;
        let silu_gate = mlx_rs::ops::multiply(&gate, &sig)
            .context("mtp.mlp gate * sig")?;
        let activated = mlx_rs::ops::multiply(&silu_gate, &up)
            .context("mtp.mlp silu(gate) * up")?;
        let down = self.linear_pre(&w.down_proj, &activated)?;
        self.to_f32(down, "mtp.mlp.down")
    }

    fn linear_pre(&self, lin: &Qwen35MtpLinear, x: &Array) -> Result<Array> {
        quantized_matmul_with_mode(
            x,
            &lin.weight,
            &lin.scales,
            lin.biases.as_ref(),
            true,
            lin.group_size,
            lin.bits,
            lin.mode,
        )
        .context("Qwen35MtpBlock::linear_pre matmul failed")
    }

    fn to_f32(&self, arr: Array, what: &str) -> Result<Array> {
        arr.as_dtype(mlx_rs::Dtype::Float32)
            .with_context(|| format!("{what}: cast to f32 failed"))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Drafter lifecycle — skeleton only (Phase 2 Session 2)
// ─────────────────────────────────────────────────────────────────────────────
//
// `Qwen35MtpDrafter` will drive the block across a sequence's lifetime:
// init_sequence -> ... -> process_after_trunk -> draft -> accept -> ...
// All state currently lives in the runner (per-seq KV cache + position) —
// the drafter only needs to track:
//   * its own per-seq KV cache (one `NativeKvCache` per active seq)
//   * `pending_h: HashMap<u64, Array>` — `[1, 1, hidden]` right-shift seed
//   * `verify_h: HashMap<u64, Array>` — `[T, hidden]` snapshot from the most
//     recent process_after_trunk, consulted by `accept(n_accepted)`
//   * `last_n_drafted: HashMap<u64, usize>` — rewind bookkeeping
//
// Mirrors `crates/lumen-model/src/qwen3_5_moe/mtp_drafter.rs` (Candle, now
// deleted at 8e3d495) — port the same state machine to mlx-native idioms in
// the next session.

#[allow(dead_code)]
pub struct Qwen35MtpDrafter {
    pub(crate) block: Qwen35MtpBlock,
    // Per-seq KV cache instances live here in a `HashMap<u64, NativeKvCache>`.
    // Filled in next session.
}

impl Qwen35MtpDrafter {
    pub fn new(block: Qwen35MtpBlock) -> Self {
        Self { block }
    }

    pub fn block(&self) -> &Qwen35MtpBlock {
        &self.block
    }
}
