//! Multi-Token Prediction (MTP) head for Qwen3.5/3.6 — one extra decoder block
//! that consumes the pair `(x_{p+1}, h_p)` and predicts `x_{p+2}`. Used as a
//! native speculative draft.
//!
//! Reference: llama.cpp PR #22673 (`common_speculative_state_draft_mtp` +
//! `llama_model_qwen35moe::graph_mtp`). The block architecture matches a
//! standard Qwen3.5 full-attention + Dense/MoE FFN block, prepended with three
//! MTP-specific bits:
//!
//! ```text
//!   tok_embd = embed_tokens(x_{p+1})        ── caller-supplied
//!   e_norm   = enorm(tok_embd)
//!   h_norm   = hnorm(h_p)                   ── h_p from trunk's pre-final-norm
//!   concat   = cat([e_norm, h_norm], -1)    ── [B, T, 2*hidden]
//!   cur      = eh_proj @ concat             ── [B, T, hidden]
//!   inpSA    = cur
//!   cur      = attn(input_layernorm(cur))
//!   cur     += inpSA                        ── attn residual
//!   ffn_res  = cur
//!   cur      = mlp(post_attention_layernorm(cur))
//!   cur     += ffn_res                      ── ffn residual
//!   new_h_pre = cur                          ── seeds the AR draft loop
//!   norm_out = (shared_head_norm || fallback_norm)(cur)
//!   logits   = (shared_head_head || fallback_lm_head)(norm_out)
//! ```
//!
//! The block owns its own attention KV cache (separate from the trunk's). The
//! `MtpDrafter` (Phase 4) drives this block via `process_after_trunk`/`draft`/
//! `accept` to mirror llama.cpp's `process`/`draft`/`accept` lifecycle.

use candle_core::{Result as CandleResult, Tensor, D};
use candle_nn::{Module, RmsNorm};

use super::moe::MlpBlock;
use super::proj::ProjLinear;
use super::self_attn::SelfAttention;

/// One Qwen3.5/3.6 MTP block. Always full-attention (never linear/Mamba) and
/// always paired with an FFN — Dense SwiGLU for 27B, routed MoE for 35B-A3B.
pub struct MtpBlock {
    // MTP-specific projection: concat(e_norm, h_norm) → hidden.
    pub(crate) eh_proj: ProjLinear,
    /// RMSNorm on the embedded next-token id (pre-`eh_proj`).
    pub(crate) enorm: RmsNorm,
    /// RMSNorm on the trunk's pre-final-norm hidden (pre-`eh_proj`).
    pub(crate) hnorm: RmsNorm,

    // Standard decoder block — same layout as Qwen3.5 full-attention layers.
    pub(crate) input_layernorm: RmsNorm,
    pub(crate) attention: SelfAttention,
    pub(crate) post_attention_layernorm: RmsNorm,
    pub(crate) mlp: MlpBlock,

    // Output head. Both `None` → fall back to trunk's final_norm + lm_head
    // (the `mtp_use_dedicated_embeddings=false` case shared by Qwen3.6-27B
    // and 35B-A3B). When `shared_head_norm = Some(...)` and the checkpoint
    // omits `mtp.shared_head_head`, we still use the trunk's lm_head.
    pub(crate) shared_head_norm: Option<RmsNorm>,
    pub(crate) shared_head_head: Option<ProjLinear>,
}

impl MtpBlock {
    pub fn new(
        eh_proj: ProjLinear,
        enorm: RmsNorm,
        hnorm: RmsNorm,
        input_layernorm: RmsNorm,
        attention: SelfAttention,
        post_attention_layernorm: RmsNorm,
        mlp: MlpBlock,
        shared_head_norm: Option<RmsNorm>,
        shared_head_head: Option<ProjLinear>,
    ) -> Self {
        Self {
            eh_proj,
            enorm,
            hnorm,
            input_layernorm,
            attention,
            post_attention_layernorm,
            mlp,
            shared_head_norm,
            shared_head_head,
        }
    }

    pub fn attention(&self) -> &SelfAttention {
        &self.attention
    }

    pub fn attention_mut(&mut self) -> &mut SelfAttention {
        &mut self.attention
    }

    // ── KV cache lifecycle (delegated to inner SelfAttention) ────────────────
    pub fn enable_kv_cache(&mut self, max_seq_len: usize) {
        self.attention.enable_kv_cache(max_seq_len);
    }
    pub fn reset_cache(&mut self) {
        self.attention.reset_kv_cache();
    }
    pub fn set_current_seq_id(&mut self, seq_id: u64) {
        self.attention.set_current_seq_id(seq_id);
    }
    pub fn init_seq_kv_cache(&mut self, seq_id: u64) {
        self.attention.init_seq_kv_cache(seq_id);
    }
    pub fn remove_seq_kv_cache(&mut self, seq_id: u64) {
        self.attention.remove_seq_kv_cache(seq_id);
    }
    pub fn truncate_kv_cache(&mut self, n_keep: usize) {
        self.attention.truncate_kv_cache(n_keep);
    }

    /// Forward one MTP step.
    ///
    /// * `embeds`        — `embed_tokens(x_{p+1})`, shape `[B, T, hidden]`.
    ///                     Caller pre-embeds so the block doesn't need to hold
    ///                     a reference to the trunk's [`Embedding`].
    /// * `h_pre`         — trunk's pre-final-norm hidden at the same `T`
    ///                     positions, shape `[B, T, hidden]`. For the first
    ///                     draft of a sequence this is the trunk's last
    ///                     captured `h_pre_norm` (see Phase 1).
    /// * `fallback_norm` — trunk's `final_norm`. Used iff `shared_head_norm`
    ///                     is `None`.
    /// * `fallback_lm_head` — trunk's `lm_head`. Used iff `shared_head_head`
    ///                     is `None` (true for Qwen3.6-27B / 35B-A3B).
    /// * `pos_offset`    — position of the first row of `embeds`/`h_pre` in
    ///                     the MTP block's KV cache.
    /// * `mask`          — optional attention mask. `None` for AR decode
    ///                     (single-row queries); causal mask required for
    ///                     verify batches with `T > 1`.
    ///
    /// Returns `(logits, new_h_pre)`:
    ///   * `logits[..., :, :vocab]` — predictions for `x_{p+2}` per row.
    ///   * `new_h_pre[..., :, hidden]` — post-residual hidden that becomes the
    ///     next `h_p` for the AR draft loop.
    pub fn forward(
        &mut self,
        embeds: &Tensor,
        h_pre: &Tensor,
        fallback_norm: &RmsNorm,
        fallback_lm_head: &ProjLinear,
        pos_offset: usize,
        mask: Option<&Tensor>,
    ) -> CandleResult<(Tensor, Tensor)> {
        // Shape sanity. Caller violations here are bugs in the drafter, not
        // user-facing errors, so we hard-fail rather than recover.
        let e_dims = embeds.dims();
        let h_dims = h_pre.dims();
        if e_dims != h_dims {
            return Err(candle_core::Error::Msg(format!(
                "MtpBlock::forward: embeds.dims {e_dims:?} != h_pre.dims {h_dims:?}",
            )));
        }
        if e_dims.len() != 3 {
            return Err(candle_core::Error::Msg(format!(
                "MtpBlock::forward: expected rank-3 [B,T,hidden], got {e_dims:?}",
            )));
        }

        let trace = std::env::var("LUMEN_MTP_BLOCK_TRACE").is_ok();
        let stat = |t: &Tensor, name: &str| -> CandleResult<()> {
            if !trace {
                return Ok(());
            }
            let f = t.to_dtype(candle_core::DType::F32)?.flatten_all()?;
            let v: Vec<f32> = f.to_vec1()?;
            let n = v.len() as f32;
            let mean: f32 = v.iter().sum::<f32>() / n;
            let var: f32 = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
            let std = var.sqrt();
            let max_abs = v.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
            let nan_count = v.iter().filter(|x| x.is_nan()).count();
            let inf_count = v.iter().filter(|x| x.is_infinite()).count();
            eprintln!(
                "[mtp_block] {name}: dtype={:?} shape={:?} mean={mean:.4e} std={std:.4e} max_abs={max_abs:.4e} nan={nan_count} inf={inf_count}",
                t.dtype(), t.dims(),
            );
            Ok(())
        };

        stat(embeds, "embeds_in")?;
        stat(h_pre, "h_pre_in")?;

        // 1) Normalise both inputs along hidden, then concat along hidden.
        let e_norm = self.enorm.forward(embeds)?;
        let h_norm = self.hnorm.forward(h_pre)?;
        stat(&e_norm, "e_norm")?;
        stat(&h_norm, "h_norm")?;
        // DeepSeek-V3 MTP convention: concat = [embedding_norm, hidden_norm].
        // `LUMEN_MTP_CONCAT_HE=1` swaps to hidden-first for A/B comparison.
        let concat_he_first =
            std::env::var("LUMEN_MTP_CONCAT_HE").map(|v| v == "1").unwrap_or(false);
        let concat = if concat_he_first {
            Tensor::cat(&[&h_norm, &e_norm], D::Minus1)?
        } else {
            Tensor::cat(&[&e_norm, &h_norm], D::Minus1)?
        }; // [B, T, 2*hidden]

        // 2) eh_proj: [2*hidden] → [hidden].
        let mut cur = self.eh_proj.forward(&concat)?; // [B, T, hidden]
        stat(&cur, "after_eh_proj")?;

        // Diagnostic: `LUMEN_MTP_BYPASS_BLOCK=1` skips attention + MLP. Lets us
        // isolate whether eh_proj alone produces a hidden the lm_head
        // understands.
        let bypass_block =
            std::env::var("LUMEN_MTP_BYPASS_BLOCK").map(|v| v == "1").unwrap_or(false);

        if !bypass_block {
            // 3) Standard decoder block: pre-norm attention + residual.
            let res_attn = cur.clone();
            cur = self.input_layernorm.forward(&cur)?;
            cur = self.attention.forward(&cur, pos_offset, mask)?;
            stat(&cur, "after_attn")?;
            cur = (&cur + &res_attn)?;

            // 4) Pre-norm FFN + residual.
            let res_ffn = cur.clone();
            let cur_normed = self.post_attention_layernorm.forward(&cur)?;
            let ffn_out = self.mlp.forward(&cur_normed)?;
            stat(&ffn_out, "after_mlp")?;
            cur = (&ffn_out + &res_ffn)?;
        }

        // 5) Stash the new pre-final-norm hidden BEFORE we apply the head norm.
        //    The AR draft loop pairs this with the next sampled token.
        let new_h_pre = cur.clone();
        stat(&new_h_pre, "new_h_pre")?;

        // 6) Head norm + lm_head. Either may fall back to the trunk's.
        // `LUMEN_MTP_FORCE_TRUNK_NORM=1` ignores MTP's own `shared_head_norm`
        // and uses trunk's `final_norm` — diagnostic for "is mtp.norm.weight
        // wrong" hypothesis.
        let force_trunk_norm =
            std::env::var("LUMEN_MTP_FORCE_TRUNK_NORM").map(|v| v == "1").unwrap_or(false);
        let norm_out = if force_trunk_norm {
            fallback_norm.forward(&cur)?
        } else {
            match &self.shared_head_norm {
                Some(n) => n.forward(&cur)?,
                None => fallback_norm.forward(&cur)?,
            }
        };
        stat(&norm_out, "norm_out")?;
        let logits = match &self.shared_head_head {
            Some(h) => h.forward(&norm_out)?,
            None => fallback_lm_head.forward(&norm_out)?,
        };
        stat(&logits, "logits")?;

        // Dump top-5 for the LAST row (the prediction the caller will use).
        if trace && logits.dims().len() == 3 {
            let dims = logits.dims();
            let t_last = dims[1] - 1;
            let row = logits.narrow(1, t_last, 1)?.squeeze(0)?.squeeze(0)?;
            let row_f32 = row.to_dtype(candle_core::DType::F32)?;
            let v: Vec<f32> = row_f32.to_vec1()?;
            let mut idx: Vec<usize> = (0..v.len()).collect();
            idx.sort_by(|a, b| v[*b].partial_cmp(&v[*a]).unwrap_or(std::cmp::Ordering::Equal));
            let top5: Vec<(usize, f32)> = idx.iter().take(5).map(|&i| (i, v[i])).collect();
            eprintln!("[mtp_block] top5 (last row): {top5:?}");
        }

        Ok((logits, new_h_pre))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen3_5_moe::moe::{DenseMlp, SharedExpert};
    use crate::qwen3_5_moe::self_attn::{SelfAttention, SelfAttnDims, SelfAttnRuntime};

    use candle_core::{Device, Tensor};
    use candle_nn::{Linear, RmsNorm};
    use rand::{rngs::StdRng, RngExt, SeedableRng};

    const HIDDEN: usize = 16;
    const VOCAB: usize = 32;
    const INTERMEDIATE: usize = 24;

    fn rnd(shape: &[usize], rng: &mut StdRng, device: &Device) -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| rng.random_range(-0.05..0.05)).collect();
        Tensor::from_vec(data, shape, device).unwrap()
    }

    fn ones_norm(dim: usize, device: &Device) -> RmsNorm {
        let w = Tensor::from_vec(vec![1f32; dim], (dim,), device).unwrap();
        RmsNorm::new(w, 1e-6)
    }

    fn tiny_self_attn(rng: &mut StdRng, device: &Device) -> SelfAttention {
        let d = SelfAttnDims {
            hidden_size: HIDDEN,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            attn_output_gate: true,
            rotary_dim: 4,
        };
        let combined_out = d.q_out_dim() + 2 * d.kv_out_dim();
        let qkv = Linear::new(rnd(&[combined_out, d.hidden_size], rng, device), None);
        let o = Linear::new(
            rnd(&[d.hidden_size, d.attn_value_dim()], rng, device),
            None,
        );
        let head_ones = Tensor::from_vec(vec![1f32; d.head_dim], (d.head_dim,), device).unwrap();
        SelfAttention::new(
            SelfAttnRuntime {
                dims: d,
                rope_theta: 10_000.0,
                rms_norm_eps: 1e-6,
            },
            qkv.into(),
            o.into(),
            RmsNorm::new(head_ones.clone(), 1e-6),
            RmsNorm::new(head_ones, 1e-6),
        )
    }

    fn tiny_dense_mlp(rng: &mut StdRng, device: &Device) -> DenseMlp {
        // Option J: pre-fused gate+up into [2*inter, hidden].
        let gate_up = Linear::new(rnd(&[2 * INTERMEDIATE, HIDDEN], rng, device), None);
        let down = Linear::new(rnd(&[HIDDEN, INTERMEDIATE], rng, device), None);
        DenseMlp::new(gate_up.into(), down.into(), INTERMEDIATE)
    }

    fn build_tiny_mtp(device: &Device) -> MtpBlock {
        let mut r = StdRng::seed_from_u64(0xDA47_BEEF);
        let eh_proj = Linear::new(rnd(&[HIDDEN, 2 * HIDDEN], &mut r, device), None);
        let mlp_dense = tiny_dense_mlp(&mut r, device);
        let attn = tiny_self_attn(&mut r, device);
        MtpBlock::new(
            eh_proj.into(),
            ones_norm(HIDDEN, device),  // enorm
            ones_norm(HIDDEN, device),  // hnorm
            ones_norm(HIDDEN, device),  // input_layernorm
            attn,
            ones_norm(HIDDEN, device),  // post_attention_layernorm
            MlpBlock::Dense(mlp_dense),
            Some(ones_norm(HIDDEN, device)),  // shared_head_norm (mtp.norm.weight)
            None,                              // shared_head_head — falls back to trunk lm_head
        )
    }

    fn is_finite(t: &Tensor) -> bool {
        t.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite())
    }

    /// Forward must produce two tensors of the documented shapes when fed
    /// matching `[B, T, hidden]` inputs.
    #[test]
    fn mtp_block_forward_returns_documented_shapes_and_is_finite() {
        let device = Device::Cpu;
        let mut mtp = build_tiny_mtp(&device);
        mtp.enable_kv_cache(8);
        mtp.init_seq_kv_cache(0);
        mtp.set_current_seq_id(0);

        // Inputs: B=1, T=3, hidden.
        let mut r = StdRng::seed_from_u64(0xFEEDFACE);
        let embeds = rnd(&[1, 3, HIDDEN], &mut r, &device);
        let h_pre = rnd(&[1, 3, HIDDEN], &mut r, &device);

        let fb_norm = ones_norm(HIDDEN, &device);
        let fb_lm_head: ProjLinear =
            Linear::new(rnd(&[VOCAB, HIDDEN], &mut r, &device), None).into();

        let (logits, new_h_pre) = mtp
            .forward(&embeds, &h_pre, &fb_norm, &fb_lm_head, 0, None)
            .unwrap();

        assert_eq!(logits.dims(), &[1, 3, VOCAB], "logits shape contract");
        assert_eq!(new_h_pre.dims(), &[1, 3, HIDDEN], "new_h_pre shape contract");
        assert!(is_finite(&logits), "logits should be finite");
        assert!(is_finite(&new_h_pre), "new_h_pre should be finite");
    }

    /// Shape-mismatch between `embeds` and `h_pre` must be rejected with a
    /// clear error rather than silently broadcasting / panicking deep inside
    /// the attention layer.
    #[test]
    fn mtp_block_forward_rejects_shape_mismatch() {
        let device = Device::Cpu;
        let mut mtp = build_tiny_mtp(&device);
        mtp.enable_kv_cache(8);
        mtp.init_seq_kv_cache(0);
        mtp.set_current_seq_id(0);

        let mut r = StdRng::seed_from_u64(0xBADBAD);
        let embeds = rnd(&[1, 3, HIDDEN], &mut r, &device);
        let h_pre = rnd(&[1, 2, HIDDEN], &mut r, &device); // mismatched T

        let fb_norm = ones_norm(HIDDEN, &device);
        let fb_lm_head: ProjLinear =
            Linear::new(rnd(&[VOCAB, HIDDEN], &mut r, &device), None).into();

        let err = mtp
            .forward(&embeds, &h_pre, &fb_norm, &fb_lm_head, 0, None)
            .expect_err("mismatched shapes must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("embeds.dims") && msg.contains("h_pre.dims"),
            "error must name both tensors: {msg}",
        );
    }

    /// The AR draft loop calls `forward` repeatedly with `T=1`, advancing the
    /// KV cache by one position each call. The second call's `pos_offset`
    /// must accept any non-negative value and produce a finite output.
    #[test]
    fn mtp_block_supports_autoregressive_single_token_steps() {
        let device = Device::Cpu;
        let mut mtp = build_tiny_mtp(&device);
        mtp.enable_kv_cache(8);
        mtp.init_seq_kv_cache(0);
        mtp.set_current_seq_id(0);

        let mut r = StdRng::seed_from_u64(0xA0A0);
        let fb_norm = ones_norm(HIDDEN, &device);
        let fb_lm_head: ProjLinear =
            Linear::new(rnd(&[VOCAB, HIDDEN], &mut r, &device), None).into();

        // Step 0: T=1 at pos_offset=0. Use a stub h_pre (would normally come
        // from the trunk's `take_h_pre_norm()`).
        let embeds_0 = rnd(&[1, 1, HIDDEN], &mut r, &device);
        let h_pre_0 = rnd(&[1, 1, HIDDEN], &mut r, &device);
        let (logits_0, h_next) = mtp
            .forward(&embeds_0, &h_pre_0, &fb_norm, &fb_lm_head, 0, None)
            .unwrap();
        assert_eq!(logits_0.dims(), &[1, 1, VOCAB]);
        assert_eq!(h_next.dims(), &[1, 1, HIDDEN]);

        // Step 1: pair the freshly-emitted h_next with another embedded token
        // at pos_offset=1. This is the canonical AR-draft sequence.
        let embeds_1 = rnd(&[1, 1, HIDDEN], &mut r, &device);
        let (logits_1, h_next_1) = mtp
            .forward(&embeds_1, &h_next, &fb_norm, &fb_lm_head, 1, None)
            .unwrap();
        assert_eq!(logits_1.dims(), &[1, 1, VOCAB]);
        assert_eq!(h_next_1.dims(), &[1, 1, HIDDEN]);
        assert!(is_finite(&logits_1));
        assert!(is_finite(&h_next_1));
    }
}
