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
pub struct Qwen35MtpDenseMlp {
    pub gate_proj: Qwen35MtpLinear,
    pub up_proj: Qwen35MtpLinear,
    pub down_proj: Qwen35MtpLinear,
}

/// MoE MLP weights — `gate` router + per-expert switch_glu tiles + a dense
/// `shared_expert` with its own sigmoid gating. Layout mirrors the trunk's
/// `ResolvedMoeWeights` so we can hand a `MoeBlockWeights<'_>` view to
/// `native_moe::moe_block_forward`.
///
/// Used by Qwen3.6-35B-A3B (`mtp.layers.0.mlp.experts.gate_up_proj` etc).
pub struct Qwen35MtpMoeMlp {
    // Router gate (AFFINE 8-bit, with biases — convention from native_moe).
    pub gate_w: Array,
    pub gate_s: Array,
    pub gate_b: Array,
    pub gate_gs: i32,
    pub gate_bits: i32,
    // Shared expert sigmoid gate (AFFINE 8-bit, with biases).
    pub sg_w: Array,
    pub sg_s: Array,
    pub sg_b: Array,
    pub sg_gs: i32,
    pub sg_bits: i32,
    // Switch GLU experts — 3D quantized [E, I, H_packed].
    pub sw_gate_w: Array,
    pub sw_gate_s: Array,
    pub sw_up_w: Array,
    pub sw_up_s: Array,
    pub sw_down_w: Array,
    pub sw_down_s: Array,
    // Optional affine biases (Some for MTPLX-format affine-4bit experts,
    // None for MXFP4 which is bias-free).
    pub sw_gate_b: Option<Array>,
    pub sw_up_b: Option<Array>,
    pub sw_down_b: Option<Array>,
    pub sw_gs: i32,
    pub sw_bits: i32,
    pub sw_mode: &'static CStr,
    // Shared dense expert (MXFP4 group 32 — matches trunk).
    pub se_gate_w: Array,
    pub se_gate_s: Array,
    pub se_up_w: Array,
    pub se_up_s: Array,
    pub se_down_w: Array,
    pub se_down_s: Array,
    pub se_gate_b: Option<Array>,
    pub se_up_b: Option<Array>,
    pub se_down_b: Option<Array>,
    pub se_gs: i32,
    pub se_bits: i32,
    pub se_mode: &'static CStr,
    // Routing meta.
    pub top_k: i32,
    pub norm_topk_prob: bool,
}

/// MLP variant. Dense for 27B Qwen3.6, MoE for 35B-A3B.
pub enum Qwen35MtpMlp {
    Dense(Qwen35MtpDenseMlp),
    Moe(Qwen35MtpMoeMlp),
}

/// Config for the MoE MTP variant — passed by the caller (parsed from
/// `config.json` text_config).
#[derive(Debug, Clone, Copy)]
pub struct MtpMoeConfig {
    pub num_experts: i32,
    pub num_experts_per_tok: i32,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub norm_topk_prob: bool,
}

/// MLP topology + size config for `load_block_from_hf`. Dense for 27B, MoE
/// for 35B-A3B.
#[derive(Debug, Clone, Copy)]
pub enum MtpMlpConfig {
    Dense { intermediate_size: usize },
    Moe(MtpMoeConfig),
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

/// Which projection inside the MTP block a head-internal LoRA targets. Drives
/// the position-expansion experiment (eh-only vs attention / MLP). PoC-only.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MtpLoraPos {
    Eh,
    Q,
    K,
    V,
    O,
    Gate,
    Up,
    Down,
}

impl MtpLoraPos {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.trim().to_ascii_lowercase().as_str() {
            "eh" | "eh_proj" | "fc" => Self::Eh,
            "q" | "q_proj" => Self::Q,
            "k" | "k_proj" => Self::K,
            "v" | "v_proj" => Self::V,
            "o" | "o_proj" => Self::O,
            "gate" | "gate_proj" => Self::Gate,
            "up" | "up_proj" => Self::Up,
            "down" | "down_proj" => Self::Down,
            _ => return None,
        })
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Eh => "eh",
            Self::Q => "q",
            Self::K => "k",
            Self::V => "v",
            Self::O => "o",
            Self::Gate => "gate",
            Self::Up => "up",
            Self::Down => "down",
        }
    }
}

/// A set of optional rank-`r` LoRA factors, one per projection. The delta added
/// to each projection's output is `y += (x · Aᵀ) · Bᵀ` with `A:[r, in]`,
/// `B:[out, r]`; `None` leaves that projection frozen. Used ONLY by the
/// head-internal training PoC (`forward_train_lora_multi`); the serving forward
/// always passes `None`.
#[derive(Default)]
pub struct MtpLoraSet {
    pub eh_proj: Option<(Array, Array)>,
    pub q: Option<(Array, Array)>,
    pub k: Option<(Array, Array)>,
    pub v: Option<(Array, Array)>,
    pub o: Option<(Array, Array)>,
    pub gate: Option<(Array, Array)>,
    pub up: Option<(Array, Array)>,
    pub down: Option<(Array, Array)>,
}

/// Config for the head-internal MTP LoRA training PoC. `positions` selects the
/// projections that receive a LoRA; `kl` switches the objective from CE-to-
/// argmax to top-`topk` distillation KL against the trunk's verify distribution.
#[derive(Clone)]
pub struct HiTrainCfg {
    pub rank: usize,
    pub steps: usize,
    pub lr: f32,
    pub weight_decay: f32,
    pub positions: Vec<MtpLoraPos>,
    pub kl: bool,
    pub topk: usize,
    /// LoRA-init RNG seed. Vary to probe whether a marginal held-out lift is
    /// seed-stable (real) or noise (flips sign across seeds).
    pub seed: u64,
}

impl Default for HiTrainCfg {
    fn default() -> Self {
        Self {
            rank: 8,
            steps: 120,
            lr: 0.02,
            weight_decay: 0.0,
            positions: vec![MtpLoraPos::Eh],
            kl: false,
            topk: 16,
            seed: 1234,
        }
    }
}

/// `y += (x · Aᵀ) · Bᵀ` LoRA delta. x:[B,T,in] A:[r,in] B:[out,r] → [B,T,out].
fn lora_delta(x: &Array, a: &Array, b: &Array) -> Result<Array> {
    Ok(x.matmul(&a.transpose_axes(&[1, 0])?)?
        .matmul(&b.transpose_axes(&[1, 0])?)?)
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

        let dbg = std::env::var("LUMEN_MTP_DEBUG")
            .map(|v| v == "1")
            .unwrap_or(false);
        let stat = |tag: &str, a: &Array| {
            if !dbg {
                return;
            }
            let f = a.as_dtype(mlx_rs::Dtype::Float32).unwrap();
            let amax = mlx_rs::ops::abs(&f).unwrap().max(None).unwrap();
            let amean = mlx_rs::ops::abs(&f).unwrap().mean(None).unwrap();
            amax.eval().ok();
            amean.eval().ok();
            eprintln!(
                "[mtp.block] {tag:<16} shape={:?} |max|={:.4} |mean|={:.4}",
                a.shape(),
                amax.as_slice::<f32>()[0],
                amean.as_slice::<f32>()[0]
            );
        };

        // ── (1) enorm + hnorm + concat ───────────────────────────────────
        stat("embeds", embeds);
        stat("h_pre", h_pre);
        let e_n = rms_norm(embeds, &self.enorm, eps).context("Qwen35MtpBlock: enorm failed")?;
        let h_n = rms_norm(h_pre, &self.hnorm, eps).context("Qwen35MtpBlock: hnorm failed")?;
        stat("e_n", &e_n);
        stat("h_n", &h_n);
        // DeepSeek-V3 / Qwen3.6 convention: [embed_norm, hidden_norm].
        let concat = mlx_rs::ops::concatenate_axis(&[&e_n, &h_n], -1)
            .context("Qwen35MtpBlock: concat(e_norm, h_norm) failed")?;

        // ── (2) eh_proj: [2*hidden] -> [hidden] ──────────────────────────
        stat("concat", &concat);
        let mut cur = self
            .linear_pre(&self.eh_proj, &concat)
            .context("Qwen35MtpBlock: eh_proj failed")?;
        cur = self.to_f32(cur, "eh_proj_out")?;
        stat("after_eh_proj", &cur);

        // ── (3) attention sub-block: pre-norm + self_attn + residual ─────
        let res_attn = cur.clone();
        let cur_n = rms_norm(&cur, &self.input_layernorm, eps)
            .context("Qwen35MtpBlock: input_layernorm failed")?;
        let attn_out = self
            .self_attention_forward(&cur_n, b, t, causal, cache)
            .context("Qwen35MtpBlock: self_attention_forward failed")?;
        stat("attn_out", &attn_out);
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
            Qwen35MtpMlp::Moe(m) => self
                .moe_mlp_forward(&cur_n, m)
                .context("Qwen35MtpBlock: moe_mlp_forward failed")?,
        };
        stat("mlp_out", &mlp_out);
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
        let norm_out = rms_norm(&cur, norm_w, eps).context("Qwen35MtpBlock: head norm failed")?;
        stat("norm_out", &norm_out);
        let logits = self
            .linear_pre(trunk_lm_head, &norm_out)
            .context("Qwen35MtpBlock: lm_head failed")?;
        stat("logits", &logits);

        // Chained hidden for the next draft (k>=1). MTPLX's `_mtp_core` with
        // mtp_hidden_variant="post_norm" feeds the NEXT draft the post-head-norm
        // hidden (`mtp.norm(x)`), not the pre-norm residual. lumen historically
        // returned pre-norm `cur`. Gated for A/B (LUMEN_MTP_CHAINED_POSTNORM).
        // Default ON: A/B showed accept 0.333->0.362, speedup 0.845->0.873.
        let chained_hidden = if std::env::var("LUMEN_MTP_CHAINED_POSTNORM")
            .map(|v| v != "0")
            .unwrap_or(true)
        {
            norm_out.clone()
        } else {
            new_h_pre
        };

        let _ = hidden; // dims sanity check above already locked it
        Ok((logits, chained_hidden))
    }

    /// TRAINING forward for the head-internal C4 LoRA PoC (accept-lever probe).
    ///
    /// Identical math to [`forward`] but (a) adds a rank-`r` LoRA delta to the
    /// `eh_proj` (mtp.fc) output — `cur += (concat·Aᵀ)·Bᵀ`, A`[r,2h]` B`[h,r]` —
    /// so a `value_and_grad` over `(a, b)` trains a correction INSIDE the block
    /// (gradients flow back through attention + MLP, unlike the dead frozen-hidden
    /// lm_head-only C4); (b) uses a FRESH per-call KV cache so each calib row is
    /// independent (k=0 draft: `[N, 1, hidden]`, self-attention only); (c) returns
    /// only the logits. Frozen quantized weights in `self` are constants w.r.t.
    /// the LoRA args. Autograd through the dense op chain proven by
    /// `examples/mtp_autograd_probe.rs`. PoC-only; not on the serving path.
    pub fn forward_train_lora(
        &self,
        embeds: &Array,
        h_pre: &Array,
        lora_a: &Array, // [r, 2*hidden]
        lora_b: &Array, // [hidden, r]
        trunk_final_norm_weight: &Array,
        trunk_lm_head: &Qwen35MtpLinear,
    ) -> Result<Array> {
        let b = embeds.shape()[0];
        let t = embeds.shape()[1];
        let eps = self.dims.rms_norm_eps;

        let e_n = rms_norm(embeds, &self.enorm, eps)?;
        let h_n = rms_norm(h_pre, &self.hnorm, eps)?;
        let concat = mlx_rs::ops::concatenate_axis(&[&e_n, &h_n], -1)?;

        // eh_proj base (frozen quant) + LoRA delta.
        let base = self.linear_pre(&self.eh_proj, &concat)?;
        let base = self.to_f32(base, "eh_proj_out")?;
        let lo = concat
            .matmul(&lora_a.transpose_axes(&[1, 0])?)? // [B,T,r]
            .matmul(&lora_b.transpose_axes(&[1, 0])?)?; // [B,T,hidden]
        let mut cur = base.add(&lo)?;

        // attention sub-block (fresh cache → each row is a standalone k=0 draft).
        let mut cache = NativeKvCache::new();
        let res_attn = cur.clone();
        let cur_n = rms_norm(&cur, &self.input_layernorm, eps)?;
        let attn_out = self.self_attention_forward(&cur_n, b, t, t > 1, &mut cache)?;
        cur = mlx_rs::ops::add(&attn_out, &res_attn)?;

        // MLP sub-block.
        let res_ffn = cur.clone();
        let cur_n = rms_norm(&cur, &self.post_attention_layernorm, eps)?;
        let mlp_out = match &self.mlp {
            Qwen35MtpMlp::Dense(d) => self.dense_mlp_forward(&cur_n, d)?,
            Qwen35MtpMlp::Moe(m) => self.moe_mlp_forward(&cur_n, m)?,
        };
        cur = mlx_rs::ops::add(&mlp_out, &res_ffn)?;

        // head norm + lm_head.
        let norm_w = self
            .shared_head_norm
            .as_ref()
            .unwrap_or(trunk_final_norm_weight);
        let norm_out = rms_norm(&cur, norm_w, eps)?;
        self.linear_pre(trunk_lm_head, &norm_out)
    }

    /// Position-expanded training forward — like [`forward_train_lora`] but the
    /// LoRA delta can land on ANY subset of {eh_proj, q, k, v, o, gate, up,
    /// down} (the position-expansion experiment). Gradients flow through the
    /// whole block; frozen quant weights are constants w.r.t. the deltas. Fresh
    /// per-call KV cache → each calib row is a standalone k=0 draft. Returns the
    /// logits `[B, T, vocab]`. PoC-only; never on the serving path.
    pub fn forward_train_lora_multi(
        &self,
        embeds: &Array,
        h_pre: &Array,
        lora: &MtpLoraSet,
        trunk_final_norm_weight: &Array,
        trunk_lm_head: &Qwen35MtpLinear,
    ) -> Result<Array> {
        let b = embeds.shape()[0];
        let t = embeds.shape()[1];
        let eps = self.dims.rms_norm_eps;

        let e_n = rms_norm(embeds, &self.enorm, eps)?;
        let h_n = rms_norm(h_pre, &self.hnorm, eps)?;
        let concat = mlx_rs::ops::concatenate_axis(&[&e_n, &h_n], -1)?;

        let base = self.to_f32(self.linear_pre(&self.eh_proj, &concat)?, "eh_proj_out")?;
        let mut cur = self.apply_lora_opt(base, &concat, lora.eh_proj.as_ref())?;

        let mut cache = NativeKvCache::new();
        let res_attn = cur.clone();
        let cur_n = rms_norm(&cur, &self.input_layernorm, eps)?;
        let attn_out =
            self.self_attention_forward_lora(&cur_n, b, t, t > 1, &mut cache, Some(lora))?;
        cur = mlx_rs::ops::add(&attn_out, &res_attn)?;

        let res_ffn = cur.clone();
        let cur_n = rms_norm(&cur, &self.post_attention_layernorm, eps)?;
        let mlp_out = match &self.mlp {
            Qwen35MtpMlp::Dense(d) => self.dense_mlp_forward_lora(&cur_n, d, Some(lora))?,
            Qwen35MtpMlp::Moe(m) => self.moe_mlp_forward(&cur_n, m)?,
        };
        cur = mlx_rs::ops::add(&mlp_out, &res_ffn)?;

        let norm_w = self
            .shared_head_norm
            .as_ref()
            .unwrap_or(trunk_final_norm_weight);
        let norm_out = rms_norm(&cur, norm_w, eps)?;
        self.linear_pre(trunk_lm_head, &norm_out)
    }

    /// `base + (x·Aᵀ)·Bᵀ` when a LoRA pair is present; identity otherwise.
    fn apply_lora_opt(
        &self,
        base: Array,
        x: &Array,
        lora: Option<&(Array, Array)>,
    ) -> Result<Array> {
        match lora {
            Some((a, b)) => base.add(&lora_delta(x, a, b)?).map_err(Into::into),
            None => Ok(base),
        }
    }

    // ── dims accessors for the LoRA-shape bookkeeping in the train fn ─────────
    /// q projection output width = num_heads * (head_dim * (gate ? 2 : 1)).
    pub fn q_out_dim(&self) -> i32 {
        let q_last = if self.dims.attn_output_gate {
            self.dims.head_dim * 2
        } else {
            self.dims.head_dim
        };
        (self.dims.num_attention_heads * q_last) as i32
    }
    /// k / v projection output width = num_kv * head_dim.
    pub fn kv_out_dim(&self) -> i32 {
        (self.dims.num_key_value_heads * self.dims.head_dim) as i32
    }
    /// o projection input width = num_heads * head_dim.
    pub fn o_in_dim(&self) -> i32 {
        (self.dims.num_attention_heads * self.dims.head_dim) as i32
    }
    /// Dense MLP intermediate size (gate_proj out-dim from scales rows); `None`
    /// for MoE blocks (MLP-position LoRA unsupported there).
    pub fn dense_intermediate(&self) -> Option<i32> {
        match &self.mlp {
            Qwen35MtpMlp::Dense(d) => Some(d.gate_proj.scales.shape()[0]),
            Qwen35MtpMlp::Moe(_) => None,
        }
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
        self.self_attention_forward_lora(x, b, t, causal, cache, None)
    }

    /// LoRA-aware self-attention. `lora=None` (serving path) is byte-identical
    /// to the original — only an `Option` check per projection differs. When
    /// `Some`, adds the q/k/v/o LoRA deltas (position-expansion PoC).
    fn self_attention_forward_lora(
        &self,
        x: &Array,
        b: i32,
        t: i32,
        causal: bool,
        cache: &mut NativeKvCache,
        lora: Option<&MtpLoraSet>,
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
        let q_proj = self.apply_lora_opt(q_proj, x, lora.and_then(|l| l.q.as_ref()))?;
        let k_proj_raw = self.linear_pre(&attn.k_proj, x)?;
        let k_proj = self.to_f32(k_proj_raw, "mtp.k_proj")?;
        let k_proj = self.apply_lora_opt(k_proj, x, lora.and_then(|l| l.k.as_ref()))?;
        let v_proj_raw = self.linear_pre(&attn.v_proj, x)?;
        let v_proj = self.to_f32(v_proj_raw, "mtp.v_proj")?;
        let v_proj = self.apply_lora_opt(v_proj, x, lora.and_then(|l| l.v.as_ref()))?;

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
            let parts = mlx_rs::ops::split(&q_4d, 2, -1).context("mtp.q split (queries|gate)")?;
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
        let k_4d =
            mlx_rs::ops::reshape(&k_proj, &[b, t, num_kv, head_dim]).context("mtp.k reshape")?;
        let k_n = rms_norm(&k_4d, &attn.k_norm_weight, eps)?;
        let k_t = mlx_rs::ops::transpose_axes(&k_n, &[0, 2, 1, 3]).context("mtp.k transpose")?;

        let v_4d =
            mlx_rs::ops::reshape(&v_proj, &[b, t, num_kv, head_dim]).context("mtp.v reshape")?;
        let v_t = mlx_rs::ops::transpose_axes(&v_4d, &[0, 2, 1, 3]).context("mtp.v transpose")?;

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
        let attn_flat = mlx_rs::ops::reshape(&attn_back, &[b, t, num_heads * head_dim])
            .context("mtp.attn_out reshape")?;
        let gated = match gate_opt {
            Some(gate) => {
                let gate_sig = mlx_rs::ops::sigmoid(&gate).context("mtp.sigmoid(gate)")?;
                mlx_rs::ops::multiply(&attn_flat, &gate_sig).context("mtp.gated multiply")?
            }
            None => attn_flat,
        };

        // (8) o_proj.
        let o_proj_raw = self.linear_pre(&attn.o_proj, &gated)?;
        let o_proj = self.to_f32(o_proj_raw, "mtp.o_proj")?;
        self.apply_lora_opt(o_proj, &gated, lora.and_then(|l| l.o.as_ref()))
    }

    /// Dense SwiGLU MLP. Same dispatch shape as the trunk's `shared_mlp`
    /// path in `native_moe::shared_mlp`: separate gate_proj + up_proj +
    /// silu(gate) * up + down_proj.
    fn dense_mlp_forward(&self, x: &Array, w: &Qwen35MtpDenseMlp) -> Result<Array> {
        self.dense_mlp_forward_lora(x, w, None)
    }

    /// LoRA-aware dense MLP. `lora=None` is byte-identical to the serving path.
    fn dense_mlp_forward_lora(
        &self,
        x: &Array,
        w: &Qwen35MtpDenseMlp,
        lora: Option<&MtpLoraSet>,
    ) -> Result<Array> {
        let gate = self.linear_pre(&w.gate_proj, x)?;
        let gate = self.to_f32(gate, "mtp.mlp.gate")?;
        let gate = self.apply_lora_opt(gate, x, lora.and_then(|l| l.gate.as_ref()))?;
        let up = self.linear_pre(&w.up_proj, x)?;
        let up = self.to_f32(up, "mtp.mlp.up")?;
        let up = self.apply_lora_opt(up, x, lora.and_then(|l| l.up.as_ref()))?;
        // SwiGLU: silu(gate) * up == sigmoid(gate) * gate * up. Mirrors the
        // expansion used in `native_moe::swiglu_compiled_inner` so a future
        // fuse pass can swap in the compiled-graph variant trivially.
        let sig = mlx_rs::ops::sigmoid(&gate).context("mtp.mlp sigmoid(gate)")?;
        let silu_gate = mlx_rs::ops::multiply(&gate, &sig).context("mtp.mlp gate * sig")?;
        let activated =
            mlx_rs::ops::multiply(&silu_gate, &up).context("mtp.mlp silu(gate) * up")?;
        let down = self.linear_pre(&w.down_proj, &activated)?;
        let down = self.to_f32(down, "mtp.mlp.down")?;
        self.apply_lora_opt(down, &activated, lora.and_then(|l| l.down.as_ref()))
    }

    /// MoE MLP — wraps `native_moe::moe_block_forward` with the MTP block's
    /// resolved weights. Matches the trunk's middle-layer MoE dispatch.
    fn moe_mlp_forward(&self, x: &Array, w: &Qwen35MtpMoeMlp) -> Result<Array> {
        use crate::native_moe::{
            AffineGateWeights, MoeBlockWeights, SharedMlpWeights, SwitchExpertWeights,
            SwitchGluWeights, moe_block_forward,
        };
        let view = MoeBlockWeights {
            gate: AffineGateWeights {
                weight: &w.gate_w,
                scales: &w.gate_s,
                biases: &w.gate_b,
                group_size: w.gate_gs,
                bits: w.gate_bits,
            },
            switch_glu: SwitchGluWeights {
                gate_proj: SwitchExpertWeights {
                    weight: &w.sw_gate_w,
                    scales: &w.sw_gate_s,
                    biases: w.sw_gate_b.as_ref(),
                },
                up_proj: SwitchExpertWeights {
                    weight: &w.sw_up_w,
                    scales: &w.sw_up_s,
                    biases: w.sw_up_b.as_ref(),
                },
                down_proj: SwitchExpertWeights {
                    weight: &w.sw_down_w,
                    scales: &w.sw_down_s,
                    biases: w.sw_down_b.as_ref(),
                },
                group_size: w.sw_gs,
                bits: w.sw_bits,
                mode: w.sw_mode,
            },
            shared_expert: SharedMlpWeights {
                gate_proj_w: &w.se_gate_w,
                gate_proj_s: &w.se_gate_s,
                up_proj_w: &w.se_up_w,
                up_proj_s: &w.se_up_s,
                down_proj_w: &w.se_down_w,
                down_proj_s: &w.se_down_s,
                group_size: w.se_gs,
                bits: w.se_bits,
                mode: w.se_mode,
            },
            shared_gate: AffineGateWeights {
                weight: &w.sg_w,
                scales: &w.sg_s,
                biases: &w.sg_b,
                group_size: w.sg_gs,
                bits: w.sg_bits,
            },
            top_k: w.top_k,
            norm_topk_prob: w.norm_topk_prob,
        };
        moe_block_forward(x, &view)
    }

    /// Debug: run only enorm/hnorm/concat/eh_proj on explicit inputs.
    /// Used by the numeric reference-comparison harness to isolate the
    /// through-fc path. embeds/h_pre are [1, 1, hidden].
    pub fn debug_through_fc(&self, embeds: &Array, h_pre: &Array) -> Result<(Array, Array, Array)> {
        let eps = self.dims.rms_norm_eps;
        let e_n = rms_norm(embeds, &self.enorm, eps)?;
        let h_n = rms_norm(h_pre, &self.hnorm, eps)?;
        let concat = mlx_rs::ops::concatenate_axis(&[&e_n, &h_n], -1)?;
        let fc_out = self.linear_pre(&self.eh_proj, &concat)?;
        let fc_out = self.to_f32(fc_out, "debug_fc_out")?;
        Ok((e_n, h_n, fc_out))
    }

    /// Debug: run the FULL block forward on explicit (embeds, h_pre) and
    /// return every intermediate stage, for layer-by-layer comparison against
    /// the mlx Python reference. Mirrors `forward` exactly (steps 1-6) but uses
    /// a fresh internal KV cache (offset 0, like a first draft). Order:
    /// [e_n, h_n, fc_out, attn_out, post_attn, mlp_out, post_mlp, norm_out].
    pub fn debug_block_stages(
        &self,
        embeds: &Array,
        h_pre: &Array,
    ) -> Result<Vec<(&'static str, Array)>> {
        let eps = self.dims.rms_norm_eps;
        let mut cache = crate::native_cache::NativeKvCache::new();

        let e_n = rms_norm(embeds, &self.enorm, eps)?;
        let h_n = rms_norm(h_pre, &self.hnorm, eps)?;
        let concat = mlx_rs::ops::concatenate_axis(&[&e_n, &h_n], -1)?;
        let fc_out = self.to_f32(self.linear_pre(&self.eh_proj, &concat)?, "dbg.fc")?;

        let b = embeds.shape()[0];
        let t = embeds.shape()[1];

        let res_attn = fc_out.clone();
        let cur_n = rms_norm(&fc_out, &self.input_layernorm, eps)?;
        let attn_out = self.self_attention_forward(&cur_n, b, t, false, &mut cache)?;
        let post_attn = mlx_rs::ops::add(&attn_out, &res_attn)?;

        let res_ffn = post_attn.clone();
        let cur_n2 = rms_norm(&post_attn, &self.post_attention_layernorm, eps)?;
        let mlp_out = match &self.mlp {
            Qwen35MtpMlp::Dense(d) => self.dense_mlp_forward(&cur_n2, d)?,
            Qwen35MtpMlp::Moe(m) => self.moe_mlp_forward(&cur_n2, m)?,
        };
        let post_mlp = mlx_rs::ops::add(&mlp_out, &res_ffn)?;

        let norm_w = self
            .shared_head_norm
            .as_ref()
            .ok_or_else(|| anyhow!("debug_block_stages: shared_head_norm (mtp.norm) not loaded"))?;
        let norm_out = rms_norm(&post_mlp, norm_w, eps)?;

        Ok(vec![
            ("e_n", e_n),
            ("h_n", h_n),
            ("fc_out", fc_out),
            ("attn_out", attn_out),
            ("post_attn", post_attn),
            ("mlp_out", mlp_out),
            ("post_mlp", post_mlp),
            ("norm_out", norm_out),
        ])
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

// ─────────────────────────────────────────────────────────────────────────────
// HF-native loader (Qwen3.6 `mtp.*` shards -> Qwen35MtpBlock)
// ─────────────────────────────────────────────────────────────────────────────
//
// The MLX-converted snapshots ship by mlx-community DROP the `mtp.*` tensors
// at conversion time (mlx-lm never wired MTP). To recover them we read from
// the HuggingFace original snapshot — the `Qwen/Qwen3.6-*` repos under the
// user's HF cache — which carries the 15 `mtp.*` tensors as bf16. We
// quantize at load time to match the trunk's quant convention (AFFINE
// 4-bit or MXFP4) so the dispatch graph in `Qwen35MtpBlock::forward`
// stays homogeneous.
//
// HF original tensor names (verified against Qwen/Qwen3.6-27B's
// model.safetensors.index.json):
//
//   mtp.fc.weight                                  [hidden, 2*hidden]   eh_proj
//   mtp.pre_fc_norm_embedding.weight               [hidden]             enorm
//   mtp.pre_fc_norm_hidden.weight                  [hidden]             hnorm
//   mtp.norm.weight                                [hidden]             shared_head_norm
//   mtp.layers.0.input_layernorm.weight            [hidden]             input_layernorm
//   mtp.layers.0.post_attention_layernorm.weight   [hidden]             post_attention_layernorm
//   mtp.layers.0.self_attn.q_proj.weight           [q_out, hidden]      attention.q_proj
//   mtp.layers.0.self_attn.k_proj.weight           [kv_out, hidden]     attention.k_proj
//   mtp.layers.0.self_attn.v_proj.weight           [kv_out, hidden]     attention.v_proj
//   mtp.layers.0.self_attn.o_proj.weight           [hidden, v_dim]      attention.o_proj
//   mtp.layers.0.self_attn.q_norm.weight           [head_dim]           attention.q_norm_weight
//   mtp.layers.0.self_attn.k_norm.weight           [head_dim]           attention.k_norm_weight
//   mtp.layers.0.mlp.gate_proj.weight              [intermediate, hidden] mlp.gate_proj
//   mtp.layers.0.mlp.up_proj.weight                [intermediate, hidden] mlp.up_proj
//   mtp.layers.0.mlp.down_proj.weight              [hidden, intermediate] mlp.down_proj
//
// where:
//   q_out = num_heads * head_dim * (attn_output_gate ? 2 : 1)
//   kv_out = num_kv_heads * head_dim
//   v_dim = num_heads * head_dim

/// Quant mode to use at load time when materializing `mtp.*` bf16 tensors
/// into mlx-native quantized linears.
///
/// `Mxfp4` matches the trunk's production quant — fastest at inference but
/// requires mlx-c's `mlx_quantize(mode=mxfp4)` to be available (lands in
/// mlx-rs >= 0.30). `Affine4` is the universal fallback (every mlx build
/// supports it) and matches the Phase 2 S1.5 synth bench numbers
/// (Step B = 2.12 ms).
#[derive(Debug, Clone, Copy)]
pub enum MtpLoadQuant {
    /// AFFINE 4-bit, group_size=64 (or 32 to halve scale memory).
    Affine4 { group_size: i32 },
    /// MXFP4 block-scaled exponents, group_size=32. Production trunk format.
    Mxfp4,
    /// Keep bf16. Slower per step but no quant-time at load.
    Bf16,
}

impl MtpLoadQuant {
    fn group_size(&self) -> i32 {
        match self {
            MtpLoadQuant::Affine4 { group_size } => *group_size,
            MtpLoadQuant::Mxfp4 => 32,
            MtpLoadQuant::Bf16 => 0,
        }
    }
    fn bits(&self) -> i32 {
        match self {
            MtpLoadQuant::Affine4 { .. } | MtpLoadQuant::Mxfp4 => 4,
            MtpLoadQuant::Bf16 => 16,
        }
    }
    fn mode_cstr(&self) -> Option<&'static std::ffi::CStr> {
        use crate::native_quant::{MODE_AFFINE, MODE_MXFP4};
        match self {
            MtpLoadQuant::Affine4 { .. } => Some(MODE_AFFINE),
            MtpLoadQuant::Mxfp4 => Some(MODE_MXFP4),
            MtpLoadQuant::Bf16 => None,
        }
    }
}

/// Load a Qwen3.5/3.6 MTP block from an HF original snapshot directory.
///
/// `hf_dir` is the snapshot root holding `model.safetensors.index.json`
/// plus the multi-shard safetensors files. Typically:
///   `~/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/<hash>`
///
/// `dims` describes the **target** trunk's attention shape — `mtp.*` tensors
/// are validated against these. For 35B-A3B the loader pairs the HF-original
/// `Qwen/Qwen3.6-35B-A3B` snapshot's bf16 mtp.* against the mlx-community
/// 35B-A3B-mxfp4 trunk dims.
///
/// `intermediate_size` and `vocab_size` come from the trunk's `text_config`.
///
/// `quant` controls the load-time quant. Phase 2 S2 ships with `Affine4 { 64 }`
/// because it's universally supported and matches the synth bench cost
/// estimate; users can flip to `Mxfp4` once the trunk path verifies it.
pub fn load_block_from_hf(
    hf_dir: &std::path::Path,
    dims: Qwen35MtpDims,
    mlp_config: MtpMlpConfig,
    quant: MtpLoadQuant,
) -> Result<Qwen35MtpBlock> {
    use crate::native_quant::quantize_with_mode;
    use std::collections::HashMap;

    // 1) Parse `model.safetensors.index.json` to find which shard holds each
    //    `mtp.*` tensor. The HF index also exists for some MLX-converted
    //    snapshots without mtp.* in which case we error out with a hint.
    let index_path = hf_dir.join("model.safetensors.index.json");
    let index_str = std::fs::read_to_string(&index_path).with_context(|| {
        format!(
            "load_block_from_hf: reading {} failed (does the HF original snapshot exist?)",
            index_path.display()
        )
    })?;
    let index: serde_json::Value = serde_json::from_str(&index_str)
        .with_context(|| format!("parse {} as JSON", index_path.display()))?;
    let weight_map = index
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| anyhow!("missing `weight_map` object in {}", index_path.display()))?;

    let mut mtp_to_shard: HashMap<String, String> = HashMap::new();
    for (key, val) in weight_map {
        if key.starts_with("mtp.") {
            if let Some(shard) = val.as_str() {
                mtp_to_shard.insert(key.clone(), shard.to_string());
            }
        }
    }
    // 3) Load `mtp.*` tensors into a single key->Array map. Two layouts:
    //    (a) mtp.* listed in the index → load from the named shards.
    //    (b) standalone `mtp.safetensors` sidecar (MTPLX 9B/27B Speed bundles,
    //        and Forge heads) → load the whole file directly.
    let mut tensors: HashMap<String, Array> = HashMap::new();
    if !mtp_to_shard.is_empty() {
        // (a) Group keys by shard so we mmap each safetensors file exactly once.
        let mut shard_to_keys: HashMap<String, Vec<String>> = HashMap::new();
        for (key, shard) in &mtp_to_shard {
            shard_to_keys
                .entry(shard.clone())
                .or_default()
                .push(key.clone());
        }
        for (shard, keys) in shard_to_keys {
            let path = hf_dir.join(&shard);
            let map = Array::load_safetensors(&path)
                .map_err(|err| anyhow!("load_safetensors({}) failed: {err}", path.display()))?;
            for k in keys {
                if let Some(arr) = map.get(&k) {
                    tensors.insert(k, arr.clone());
                } else {
                    return Err(anyhow!(
                        "expected key `{k}` in {} (per index.json) but it was absent",
                        path.display(),
                    ));
                }
            }
        }
    } else {
        // (b) Standalone sidecar. MTPLX bundles ship the head as either a flat
        //     `mtp.safetensors` (9B) or an `mtp/weights.safetensors` subdir
        //     (27B); `mtplx_runtime.json.mtp_file` names the latter.
        let mut candidates = vec![
            hf_dir.join("mtp.safetensors"),
            hf_dir.join("mtp").join("weights.safetensors"),
        ];
        if let Ok(rt) = std::fs::read_to_string(hf_dir.join("mtplx_runtime.json"))
            && let Ok(v) = serde_json::from_str::<serde_json::Value>(&rt)
            && let Some(f) = v.get("mtp_file").and_then(|x| x.as_str())
        {
            candidates.insert(0, hf_dir.join(f));
        }
        let sidecar = candidates
            .into_iter()
            .find(|p| p.is_file())
            .ok_or_else(|| {
                anyhow!(
                    "no `mtp.*` in {} and no sidecar (mtp.safetensors / mtp/weights.safetensors) in {}",
                    index_path.display(),
                    hf_dir.display()
                )
            })?;
        let map = Array::load_safetensors(&sidecar)
            .map_err(|err| anyhow!("load_safetensors({}) failed: {err}", sidecar.display()))?;
        for (k, arr) in map.iter() {
            if k.starts_with("mtp.") {
                tensors.insert(k.to_string(), arr.clone());
            }
        }
        if tensors.is_empty() {
            return Err(anyhow!(
                "standalone {} held no `mtp.*` tensors",
                sidecar.display()
            ));
        }
    }

    // 4) Helpers for fetching + validating named tensors.
    let take_tensor =
        |tensors: &mut HashMap<String, Array>, name: &str, expect_shape: &[i32]| -> Result<Array> {
            let arr = tensors
                .remove(name)
                .ok_or_else(|| anyhow!("HF mtp.* missing key `{name}`"))?;
            let got: Vec<i32> = arr.shape().to_vec();
            if got != expect_shape {
                return Err(anyhow!(
                    "{name}: shape mismatch (got {got:?}, want {expect_shape:?})"
                ));
            }
            Ok(arr)
        };

    // Norm tensors stay bf16 (or whatever the original ships) -- no quant.
    let take_norm = |tensors: &mut HashMap<String, Array>, name: &str, dim: i32| -> Result<Array> {
        let arr = take_tensor(tensors, name, &[dim])?;
        Ok(arr)
    };

    // Linear tensors: bf16 -> mlx_rs::ops::quantize at requested mode.
    // `expect_shape` is the f32 logical shape [out, in].
    let take_linear = |tensors: &mut HashMap<String, Array>,
                       name: &str,
                       expect_shape: [i32; 2]|
     -> Result<Qwen35MtpLinear> {
        use crate::native_quant::MODE_AFFINE;
        // Pre-quantized affine head (27B Speed): the weight ships packed (U32)
        // alongside `.scales` + `.biases`. Load directly and infer the layout
        // from shapes instead of quantizing a bf16 weight. `name` ends in
        // `.weight`; the sibling keys swap the suffix.
        let base = name.strip_suffix(".weight").unwrap_or(name);
        let scales_key = format!("{base}.scales");
        if tensors.contains_key(&scales_key) {
            let weight = tensors
                .remove(name)
                .ok_or_else(|| anyhow!("prequant mtp.* missing `{name}`"))?;
            let scales = tensors
                .remove(&scales_key)
                .ok_or_else(|| anyhow!("prequant mtp.* missing `{scales_key}`"))?;
            let biases = tensors.remove(&format!("{base}.biases"));
            let out = expect_shape[0];
            let in_features = expect_shape[1];
            if weight.shape()[0] != out {
                return Err(anyhow!(
                    "{name}: prequant out-dim {} != expected {out}",
                    weight.shape()[0]
                ));
            }
            // packed_in = in * bits / 32 ; n_groups = in / group_size.
            let packed_in = weight.shape()[1];
            let bits = (32 * packed_in) / in_features.max(1);
            let n_groups = scales.shape()[1];
            let group_size = in_features / n_groups.max(1);
            return Ok(Qwen35MtpLinear {
                weight,
                scales,
                biases,
                group_size,
                bits,
                mode: MODE_AFFINE,
            });
        }
        let w = take_tensor(tensors, name, &expect_shape)?;
        if let Some(mode) = quant.mode_cstr() {
            let (packed, scales, biases) =
                quantize_with_mode(&w, quant.group_size(), quant.bits(), mode)
                    .with_context(|| format!("{name}: quantize at load failed"))?;
            Ok(Qwen35MtpLinear {
                weight: packed,
                scales,
                biases,
                group_size: quant.group_size(),
                bits: quant.bits(),
                mode,
            })
        } else {
            Err(anyhow!(
                "load_block_from_hf: Bf16 carrier path not yet implemented \
                 (Qwen35MtpLinear assumes a quantized weight layout); use \
                 Affine4 or Mxfp4"
            ))
        }
    };

    let hidden = dims.hidden_size as i32;
    let num_heads = dims.num_attention_heads as i32;
    let num_kv = dims.num_key_value_heads as i32;
    let head_dim = dims.head_dim as i32;
    let q_out = num_heads * head_dim * if dims.attn_output_gate { 2 } else { 1 };
    let kv_out = num_kv * head_dim;
    let v_dim = num_heads * head_dim;

    // 5) Build linears + norms.
    //
    // ── RMSNorm (1 + weight) convention fix ──────────────────────────────
    // The HF-original Qwen3.6 `mtp.*` snapshot stores RMSNorm weights in the
    // `(1 + w)` convention (effective scale = 1 + w). lumen's `rms_norm`
    // applies the weight directly, so we must fold `+1` at load. Confirmed
    // empirically against the MTPLX Forge-matched head (`mtp.safetensors`):
    // matched == raw + 1 to rel-MAE ~1e-4 for pre_fc_norm_{embedding,hidden},
    // mtp.norm, input_layernorm, q_norm, k_norm — but NOT
    // post_attention_layernorm (already folded in the HF export). The mxfp4
    // trunk works without this because mlx-community's conversion already
    // folded the trunk norms. Without the fold, e.g. pre_fc_norm_embedding
    // (raw mean -0.73) multiplies by a negative scale → garbage drafts,
    // 0% accept. Gated for A/B; default ON once validated.
    let fold_plus_one = std::env::var("LUMEN_MTP_NORM_PLUS_ONE")
        .map(|v| v != "0")
        .unwrap_or(true);
    let p1 = |a: Array| -> Result<Array> {
        if fold_plus_one {
            let one = Array::from_f32(1.0).as_dtype(a.dtype())?;
            Ok(mlx_rs::ops::add(&a, &one)?)
        } else {
            Ok(a)
        }
    };

    let eh_proj = take_linear(&mut tensors, "mtp.fc.weight", [hidden, 2 * hidden])?;
    let enorm = p1(take_norm(
        &mut tensors,
        "mtp.pre_fc_norm_embedding.weight",
        hidden,
    )?)?;
    let hnorm = p1(take_norm(
        &mut tensors,
        "mtp.pre_fc_norm_hidden.weight",
        hidden,
    )?)?;
    let shared_head_norm = match take_norm(&mut tensors, "mtp.norm.weight", hidden) {
        Ok(t) => Some(p1(t)?),
        // `mtp.norm.weight` is optional per llama.cpp / DeepSeek-V3 spec; if
        // the checkpoint omits it the block falls back to trunk's final_norm.
        Err(_) => None,
    };
    let input_layernorm = p1(take_norm(
        &mut tensors,
        "mtp.layers.0.input_layernorm.weight",
        hidden,
    )?)?;
    // NOTE: post_attention_layernorm is NOT folded — already absolute in the
    // HF export (matched == raw, rel-MAE 0).
    let post_attention_layernorm = take_norm(
        &mut tensors,
        "mtp.layers.0.post_attention_layernorm.weight",
        hidden,
    )?;

    let attention = Qwen35MtpAttnWeights {
        q_proj: take_linear(
            &mut tensors,
            "mtp.layers.0.self_attn.q_proj.weight",
            [q_out, hidden],
        )?,
        k_proj: take_linear(
            &mut tensors,
            "mtp.layers.0.self_attn.k_proj.weight",
            [kv_out, hidden],
        )?,
        v_proj: take_linear(
            &mut tensors,
            "mtp.layers.0.self_attn.v_proj.weight",
            [kv_out, hidden],
        )?,
        o_proj: take_linear(
            &mut tensors,
            "mtp.layers.0.self_attn.o_proj.weight",
            [hidden, v_dim],
        )?,
        q_norm_weight: p1(take_norm(
            &mut tensors,
            "mtp.layers.0.self_attn.q_norm.weight",
            head_dim,
        )?)?,
        k_norm_weight: p1(take_norm(
            &mut tensors,
            "mtp.layers.0.self_attn.k_norm.weight",
            head_dim,
        )?)?,
    };

    let mlp = match mlp_config {
        MtpMlpConfig::Dense { intermediate_size } => {
            let inter = intermediate_size as i32;
            Qwen35MtpMlp::Dense(Qwen35MtpDenseMlp {
                gate_proj: take_linear(
                    &mut tensors,
                    "mtp.layers.0.mlp.gate_proj.weight",
                    [inter, hidden],
                )?,
                up_proj: take_linear(
                    &mut tensors,
                    "mtp.layers.0.mlp.up_proj.weight",
                    [inter, hidden],
                )?,
                down_proj: take_linear(
                    &mut tensors,
                    "mtp.layers.0.mlp.down_proj.weight",
                    [hidden, inter],
                )?,
            })
        }
        MtpMlpConfig::Moe(moe) => {
            Qwen35MtpMlp::Moe(load_moe_mlp_from_tensors(&mut tensors, hidden, moe)?)
        }
    };

    // 6) Drain check: if any mtp.* tensors are left over we likely missed
    //    an intentionally optional field — warn but don't fail.
    if !tensors.is_empty() {
        let leftover: Vec<&str> = tensors.keys().map(String::as_str).collect();
        eprintln!(
            "[qwen3_5_mtp loader] note: {} unused mtp.* tensors in checkpoint: {leftover:?}",
            leftover.len()
        );
    }

    Ok(Qwen35MtpBlock {
        dims,
        eh_proj,
        enorm,
        hnorm,
        input_layernorm,
        attention,
        post_attention_layernorm,
        mlp,
        shared_head_norm,
    })
}

/// Helper for `load_block_from_hf`: pull the MoE MTP tensors out of the
/// `mtp.*` bag, quantize the bf16 originals into trunk-matching layouts, and
/// produce a `Qwen35MtpMoeMlp` ready for `moe_block_forward`.
///
/// Conventions (matching the trunk's middle-layer MoE in `qwen3_5_moe.rs`):
///   * `mtp.layers.0.mlp.gate` + `shared_expert_gate` — AFFINE 8-bit, group 64,
///     biases generated by `mlx_quantize`.
///   * `mtp.layers.0.mlp.shared_expert.{gate,up,down}_proj` — MXFP4 group 32.
///   * `mtp.layers.0.mlp.experts.gate_up_proj` — bf16 fused
///     `[E, 2*I, H]`; split along axis 1 into 2x `[E, I, H]` then MXFP4 group 32.
///   * `mtp.layers.0.mlp.experts.down_proj` — bf16 `[E, H, I]`; MXFP4 group 32.
fn load_moe_mlp_from_tensors(
    tensors: &mut std::collections::HashMap<String, Array>,
    hidden: i32,
    moe: MtpMoeConfig,
) -> Result<Qwen35MtpMoeMlp> {
    use crate::native_quant::{MODE_AFFINE, MODE_MXFP4, quantize_with_mode};

    let e = moe.num_experts;
    let i_per_expert = moe.moe_intermediate_size as i32;
    let i_shared = moe.shared_expert_intermediate_size as i32;

    // Helpers for fetching/validating named tensors out of the bag.
    let take = |tensors: &mut std::collections::HashMap<String, Array>,
                name: &str,
                expect: &[i32]|
     -> Result<Array> {
        let arr = tensors
            .remove(name)
            .ok_or_else(|| anyhow!("MoE MTP loader: missing key `{name}`"))?;
        let got: Vec<i32> = arr.shape().to_vec();
        if got != expect {
            return Err(anyhow!(
                "{name}: shape mismatch (got {got:?}, want {expect:?})"
            ));
        }
        Ok(arr)
    };

    // ── Router gate (AFFINE 8-bit, group 64) ──────────────────────────────
    let gate_w_bf16 = take(tensors, "mtp.layers.0.mlp.gate.weight", &[e, hidden])?;
    let (gate_w, gate_s, gate_b_opt) = quantize_with_mode(&gate_w_bf16, 64, 8, MODE_AFFINE)
        .context("MoE MTP: quantize gate.weight")?;
    let gate_b = gate_b_opt
        .ok_or_else(|| anyhow!("MoE MTP: gate quant must produce biases (AFFINE mode)"))?;

    // ── Shared-expert sigmoid gate (AFFINE 8-bit, group 64, output 1) ─────
    let sg_w_bf16 = take(
        tensors,
        "mtp.layers.0.mlp.shared_expert_gate.weight",
        &[1, hidden],
    )?;
    let (sg_w, sg_s, sg_b_opt) = quantize_with_mode(&sg_w_bf16, 64, 8, MODE_AFFINE)
        .context("MoE MTP: quantize shared_expert_gate.weight")?;
    let sg_b =
        sg_b_opt.ok_or_else(|| anyhow!("MoE MTP: shared_expert_gate quant must produce biases"))?;

    // ── Switch GLU experts (MXFP4 group 32) ───────────────────────────────
    // Fused `experts.gate_up_proj` ships as bf16 [E, 2*I, H]. We split along
    // axis 1 — first I rows are gate_proj, next I rows are up_proj — then
    // quantize each 3D tile independently. Matches what mlx-lm's
    // sanitize() + MXFP4 conversion script would have produced for the
    // trunk's middle layers.
    let gu_fused_bf16 = take(
        tensors,
        "mtp.layers.0.mlp.experts.gate_up_proj",
        &[e, 2 * i_per_expert, hidden],
    )?;
    let halves =
        mlx_rs::ops::split(&gu_fused_bf16, 2, 1).context("MoE MTP: split fused gate_up_proj")?;
    if halves.len() != 2 {
        return Err(anyhow!(
            "MoE MTP: split(gate_up_proj) returned {} parts (want 2)",
            halves.len()
        ));
    }
    let mut halves_iter = halves.into_iter();
    let gate_tile_bf16 = halves_iter.next().expect("gate half present");
    let up_tile_bf16 = halves_iter.next().expect("up half present");
    let (sw_gate_w, sw_gate_s, _) = quantize_with_mode(&gate_tile_bf16, 32, 4, MODE_MXFP4)
        .context("MoE MTP: quantize experts.gate tile (MXFP4)")?;
    let (sw_up_w, sw_up_s, _) = quantize_with_mode(&up_tile_bf16, 32, 4, MODE_MXFP4)
        .context("MoE MTP: quantize experts.up tile (MXFP4)")?;

    let down_3d_bf16 = take(
        tensors,
        "mtp.layers.0.mlp.experts.down_proj",
        &[e, hidden, i_per_expert],
    )?;
    let (sw_down_w, sw_down_s, _) = quantize_with_mode(&down_3d_bf16, 32, 4, MODE_MXFP4)
        .context("MoE MTP: quantize experts.down (MXFP4)")?;

    // ── Shared dense expert (MXFP4 group 32) ──────────────────────────────
    let se_gate_w_bf16 = take(
        tensors,
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
        &[i_shared, hidden],
    )?;
    let se_up_w_bf16 = take(
        tensors,
        "mtp.layers.0.mlp.shared_expert.up_proj.weight",
        &[i_shared, hidden],
    )?;
    let se_down_w_bf16 = take(
        tensors,
        "mtp.layers.0.mlp.shared_expert.down_proj.weight",
        &[hidden, i_shared],
    )?;
    let (se_gate_w, se_gate_s, _) = quantize_with_mode(&se_gate_w_bf16, 32, 4, MODE_MXFP4)
        .context("MoE MTP: quantize shared_expert.gate_proj")?;
    let (se_up_w, se_up_s, _) = quantize_with_mode(&se_up_w_bf16, 32, 4, MODE_MXFP4)
        .context("MoE MTP: quantize shared_expert.up_proj")?;
    let (se_down_w, se_down_s, _) = quantize_with_mode(&se_down_w_bf16, 32, 4, MODE_MXFP4)
        .context("MoE MTP: quantize shared_expert.down_proj")?;

    Ok(Qwen35MtpMoeMlp {
        gate_w,
        gate_s,
        gate_b,
        gate_gs: 64,
        gate_bits: 8,
        sg_w,
        sg_s,
        sg_b,
        sg_gs: 64,
        sg_bits: 8,
        sw_gate_w,
        sw_gate_s,
        sw_up_w,
        sw_up_s,
        sw_down_w,
        sw_down_s,
        sw_gate_b: None,
        sw_up_b: None,
        sw_down_b: None,
        sw_gs: 32,
        sw_bits: 4,
        sw_mode: MODE_MXFP4,
        se_gate_w,
        se_gate_s,
        se_up_w,
        se_up_s,
        se_down_w,
        se_down_s,
        se_gate_b: None,
        se_up_b: None,
        se_down_b: None,
        se_gs: 32,
        se_bits: 4,
        se_mode: MODE_MXFP4,
        top_k: moe.num_experts_per_tok,
        norm_topk_prob: moe.norm_topk_prob,
    })
}

/// Smoke-test the MTP block forward with synthetic trunk inputs.
///
/// Used by `examples/bench_qwen35_mtp_loader_smoke.rs` to validate that a
/// real-weight `Qwen35MtpBlock` (constructed via [`load_block_from_hf`])
/// dispatches end-to-end without a trunk hookup. The trunk's `lm_head` +
/// `final_norm` are synthesized at the loader's quant convention so the
/// dispatch graph is realistic but the argmax content is arbitrary.
///
/// Returns `(argmax_token_id, max_logit_value)` for the last row at T=1.
pub fn smoke_forward_with_synth_trunk(
    block: &Qwen35MtpBlock,
    vocab_size: usize,
) -> Result<(u32, f32)> {
    use crate::native_cache::NativeKvCache;
    use crate::native_quant::{MODE_AFFINE, quantize_with_mode};

    let hidden = block.dims.hidden_size as i32;

    // Synthetic trunk lm_head at the block's quant convention (AFFINE 4-bit,
    // group_size matching eh_proj). For mature wiring the real trunk lm_head
    // gets threaded in here.
    let group_size = block.eh_proj.group_size;
    let lm_head_w_f32 =
        mlx_rs::random::uniform::<_, f32>(-0.05f32, 0.05f32, &[vocab_size as i32, hidden], None)
            .context("smoke_forward: random lm_head failed")?;
    let lm_head_w_bf16 = lm_head_w_f32
        .as_dtype(mlx_rs::Dtype::Bfloat16)
        .context("smoke_forward: bf16 cast")?;
    let (lm_packed, lm_scales, lm_biases) =
        quantize_with_mode(&lm_head_w_bf16, group_size, 4, MODE_AFFINE)
            .context("smoke_forward: quantize lm_head")?;
    let trunk_lm_head = Qwen35MtpLinear {
        weight: lm_packed,
        scales: lm_scales,
        biases: lm_biases,
        group_size,
        bits: 4,
        mode: MODE_AFFINE,
    };
    let trunk_final_norm_w = mlx_rs::ops::ones::<f32>(&[hidden])
        .context("smoke_forward: ones final_norm")?
        .as_dtype(mlx_rs::Dtype::Bfloat16)
        .context("smoke_forward: final_norm bf16 cast")?;

    // Synthetic inputs at [1, 1, hidden].
    let embeds = mlx_rs::random::uniform::<_, f32>(-0.05f32, 0.05f32, &[1, 1, hidden], None)
        .context("smoke_forward: random embeds")?;
    let h_pre = mlx_rs::random::uniform::<_, f32>(-0.05f32, 0.05f32, &[1, 1, hidden], None)
        .context("smoke_forward: random h_pre")?;
    let mut cache = NativeKvCache::new();

    let (logits, _new_h) = block
        .forward(
            &embeds,
            &h_pre,
            &mut cache,
            false,
            &trunk_final_norm_w,
            &trunk_lm_head,
        )
        .context("smoke_forward: block.forward")?;
    logits.eval().context("smoke_forward: eval logits")?;

    // Pull max logit + argmax from the single output row.
    let logits_flat = logits
        .reshape(&[vocab_size as i32])
        .context("smoke_forward: reshape logits")?;
    let argmax = mlx_rs::ops::indexing::argmax_axis(&logits_flat, 0, false)
        .context("smoke_forward: argmax")?;
    argmax.eval().context("smoke_forward: argmax eval")?;
    let argmax_val: u32 = argmax
        .try_item::<i32>()
        .context("smoke_forward: argmax item")? as u32;
    let max_val_arr =
        mlx_rs::ops::max_axis(&logits_flat, 0, false).context("smoke_forward: max")?;
    max_val_arr.eval().context("smoke_forward: max eval")?;
    let max_val: f32 = max_val_arr
        .try_item::<f32>()
        .context("smoke_forward: max item")?;

    Ok((argmax_val, max_val))
}

// ─────────────────────────────────────────────────────────────────────────────
// Step B synthetic-weight microbench
// ─────────────────────────────────────────────────────────────────────────────
//
// Phase 1 (94ff5c2) measured trunk S-scaling at 1.12x and estimated Step B
// (MTP block forward) cost at ~10 ms. Step B is the swing factor for the K=2
// vs K=3 cycle decision:
//
//   K=2 cycle = trunk_decode(14.4) + mtp_draft(2 * step_B) + trunk_verify(19.2)
//             + ~3 (snap/restore/accept)
//             = ~47 ms + 2 * step_B_marginal
//   K=3 cycle = trunk_decode(14.4) + mtp_draft(3 * step_B) + trunk_verify(22.6)
//             + ~3
//             = ~50 ms + 3 * step_B_marginal
//
// where step_B_marginal is the AR-step cost (T=1 forward, post-warmup).
//
// This synth bench measures step_B at the real Qwen3.6-35B-A3B-mxfp4 shapes
// before we invest in the HF-native loader. Weights are random + MXFP4
// quantized via `mlx_rs::ops::quantize` (mode=MXFP4) — the dispatch graph
// matches the production block exactly; only the values differ.

/// Per-T benchmark result. `t` is the row count fed to `forward`; `median_ms`
/// is the post-warmup-drop median over `runs - 1` measurements.
#[derive(Debug, Clone)]
pub struct StepBBenchPoint {
    pub t: i32,
    pub min_ms: f64,
    pub median_ms: f64,
    pub max_ms: f64,
}

/// Run a synthetic-weight Step B latency probe at Qwen3.6-35B-A3B-mxfp4
/// shapes. Returns one [`StepBBenchPoint`] per requested `T` row count.
///
/// `t_values`: input row counts to probe. AR draft: T=1; verify-with-K=1: T=2.
/// `runs`: number of forward calls per T. First is dropped as warmup.
pub fn run_step_b_synthetic_bench(t_values: &[i32], runs: usize) -> Result<Vec<StepBBenchPoint>> {
    use crate::native_cache::NativeKvCache;
    use crate::native_quant::{MODE_AFFINE, quantize_with_mode};

    // Qwen3.6-35B-A3B-mxfp4 production dims (sourced from the HF config.json
    // text_config block at the snapshot we ship).
    let dims = Qwen35MtpDims {
        hidden_size: 2048,
        num_attention_heads: 16,
        num_key_value_heads: 2,
        head_dim: 256,
        rope_theta: 10_000_000.0,
        rope_dim: 64, // partial_rotary_factor 0.25 * head_dim 256
        rms_norm_eps: 1e-6,
        attn_output_gate: true,
    };
    // intermediate_size from config.json is 4304 which isn't divisible by 64
    // (the AFFINE group_size). Pad up to 4352 (= 64*68 = 32*136) for the
    // synth bench -- adds <1.2% to MLP matmul cost vs. real 4304.
    let intermediate_size: usize = 4352;
    let vocab_size: usize = 248320;

    // NOTE: synthetic bench uses AFFINE 4-bit instead of the production MXFP4
    // because `mlx_quantize` with MXFP4 returns only 2 arrays (no biases),
    // which our `quantize_with_mode` wrapper currently asserts == 3. The
    // dispatch graph (one `quantized_matmul_with_mode` per linear) is the
    // same shape; only the mode bytes differ. Real-weight bench in Phase 2
    // S2+ will hit the production MXFP4 path.
    let group_size: i32 = 64;
    let bits: i32 = 4;
    let mode = MODE_AFFINE;

    let hidden = dims.hidden_size as i32;
    let num_heads = dims.num_attention_heads as i32;
    let num_kv = dims.num_key_value_heads as i32;
    let head_dim = dims.head_dim as i32;
    let q_out = num_heads * head_dim * if dims.attn_output_gate { 2 } else { 1 };
    let kv_out = num_kv * head_dim;
    let value_dim = num_heads * head_dim;
    let inter = intermediate_size as i32;

    // Helper: random f32 array of given shape -> quantized Linear at mode.
    let make_linear = |out: i32, in_: i32, label: &str| -> Result<Qwen35MtpLinear> {
        // mlx_rs `random::uniform` returns a small-range array suitable as
        // a synthetic weight payload. The exact distribution doesn't affect
        // latency (FFI graph is shape-only) — but matching real-weight
        // magnitudes (small) avoids overflow paths in MXFP4 dequant.
        let w_f32 = mlx_rs::random::uniform::<_, f32>(-0.05f32, 0.05f32, &[out, in_], None)
            .with_context(|| format!("{label}: random.uniform failed"))?;
        let w_bf16 = w_f32
            .as_dtype(mlx_rs::Dtype::Bfloat16)
            .with_context(|| format!("{label}: cast bf16 failed"))?;
        let (packed, scales, biases) = quantize_with_mode(&w_bf16, group_size, bits, mode)
            .with_context(|| format!("{label}: quantize_with_mode failed"))?;
        Ok(Qwen35MtpLinear {
            weight: packed,
            scales,
            biases,
            group_size,
            bits,
            mode,
        })
    };

    // Helper: bf16 ones vector for RMSNorm weights.
    let ones_bf16 = |dim: i32, label: &str| -> Result<Array> {
        let v =
            mlx_rs::ops::ones::<f32>(&[dim]).with_context(|| format!("{label}: ones failed"))?;
        v.as_dtype(mlx_rs::Dtype::Bfloat16)
            .with_context(|| format!("{label}: bf16 cast failed"))
    };

    // ── Build synthetic MTP block ─────────────────────────────────────────
    let eh_proj = make_linear(hidden, 2 * hidden, "eh_proj")?;
    let enorm = ones_bf16(hidden, "enorm")?;
    let hnorm = ones_bf16(hidden, "hnorm")?;
    let input_layernorm = ones_bf16(hidden, "input_layernorm")?;
    let post_attention_layernorm = ones_bf16(hidden, "post_attn_layernorm")?;
    let shared_head_norm = Some(ones_bf16(hidden, "shared_head_norm")?);

    let attention = Qwen35MtpAttnWeights {
        q_proj: make_linear(q_out, hidden, "q_proj")?,
        k_proj: make_linear(kv_out, hidden, "k_proj")?,
        v_proj: make_linear(kv_out, hidden, "v_proj")?,
        o_proj: make_linear(hidden, value_dim, "o_proj")?,
        q_norm_weight: ones_bf16(head_dim, "q_norm")?,
        k_norm_weight: ones_bf16(head_dim, "k_norm")?,
    };
    let mlp = Qwen35MtpMlp::Dense(Qwen35MtpDenseMlp {
        gate_proj: make_linear(inter, hidden, "gate_proj")?,
        up_proj: make_linear(inter, hidden, "up_proj")?,
        down_proj: make_linear(hidden, inter, "down_proj")?,
    });
    let block = Qwen35MtpBlock {
        dims,
        eh_proj,
        enorm,
        hnorm,
        input_layernorm,
        attention,
        post_attention_layernorm,
        mlp,
        shared_head_norm,
    };

    // trunk_lm_head is the largest matmul in Step B's output stage; synth at
    // real vocab size so cost matches production.
    let trunk_lm_head = make_linear(vocab_size as i32, hidden, "trunk_lm_head")?;
    let trunk_final_norm_w = ones_bf16(hidden, "trunk_final_norm")?;

    // ── Run probes ────────────────────────────────────────────────────────
    let mut report = Vec::with_capacity(t_values.len());
    for &t in t_values {
        // Fresh cache per T so the offset starts at 0 (deterministic input
        // shape). Cache lifetime is per-T inside this loop.
        let mut cache = NativeKvCache::new();

        // Random embeds + h_pre at [1, T, hidden]. f32 to match the trunk
        // hidden's working dtype (post mlx-fast.rms_norm).
        let embeds = mlx_rs::random::uniform::<_, f32>(-0.05f32, 0.05f32, &[1, t, hidden], None)
            .context("random embeds")?;
        let h_pre = mlx_rs::random::uniform::<_, f32>(-0.05f32, 0.05f32, &[1, t, hidden], None)
            .context("random h_pre")?;

        // causal=true when T>1 (matches the verify-batch path).
        let causal = t > 1;

        let mut times_ms = Vec::with_capacity(runs);
        for _ in 0..runs {
            let t0 = std::time::Instant::now();
            let (logits, _new_h) = block
                .forward(
                    &embeds,
                    &h_pre,
                    &mut cache,
                    causal,
                    &trunk_final_norm_w,
                    &trunk_lm_head,
                )
                .context("block.forward in bench")?;
            // Force eval — mlx is lazy; without this the timer measures
            // only graph construction, not execution.
            logits.eval().context("eval(logits)")?;
            times_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
            // Reset cache between iters so each call starts from offset=0
            // (otherwise position grows, KV length grows, latency drifts).
            cache = NativeKvCache::new();
        }
        // Drop first run as warmup; sort remaining and pick min/median/max.
        let warm = &times_ms[1..];
        let mut sorted: Vec<f64> = warm.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        report.push(StepBBenchPoint {
            t,
            min_ms: sorted[0],
            median_ms: sorted[n / 2],
            max_ms: sorted[n - 1],
        });
    }

    Ok(report)
}
