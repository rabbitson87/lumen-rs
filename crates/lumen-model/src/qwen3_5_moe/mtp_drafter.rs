//! Speculative draft state machine driving an [`MtpBlock`].
//!
//! Mirrors llama.cpp PR #22673's `common_speculative_state_draft_mtp` —
//! the per-sequence lifecycle is `init_sequence` → ...repeat: trunk decode/verify
//! → `process_after_trunk` → `draft` → `accept`... → `remove_sequence`.
//!
//! ## The contract
//!
//! Two invariants hold across an entire generation, per sequence:
//!
//! 1. **MTP KV ≡ trunk KV (committed prefix)**: after every successful
//!    `accept(seq, n)`, the MTP block's KV cache for `seq` covers exactly the
//!    same positions as the trunk's KV for `seq` — no more, no less. The
//!    `process_after_trunk` hook re-mirrors any trunk ubatch into MTP before
//!    the next `draft`.
//!
//! 2. **`pending_h[seq]` ≡ trunk's pre-final-norm hidden at the seq's last
//!    committed position**. This is what the next `draft` call pairs with the
//!    last committed token to predict draft #1, and what `process_after_trunk`
//!    uses as the right-shifted seed for position 0 of the next mirror.
//!
//! Together these guarantee that draft #1 is generated from exactly the same
//! `(token, h)` pair the trunk would consume on its own next step — so when the
//! trunk verifies a long draft, the prefix it agrees with is correct *because*
//! the MTP block ran a parallel computation, not a guess.
//!
//! ## Math: why the "shift right by 1" is in `process_after_trunk`
//!
//! MTP's job is to predict `x_{p+2}` from `(x_{p+1}, h_p)`. So at the MTP-side
//! "position p+1", the input pair is `(x_{p+1}, h_p)` — i.e. the token at
//! position p+1 paired with the hidden state from position p. When the trunk
//! verifies tokens at positions `[P, P+1, ..., P+N-1]` and we want to mirror
//! that into MTP, the MTP input pairs are
//!
//! ```text
//!   pos P:   (x_P,     pending_h    )   ← pending_h was h_{P-1} from prev round
//!   pos P+1: (x_{P+1}, h_P          )
//!   pos P+2: (x_{P+2}, h_{P+1}      )
//!     ...
//!   pos P+N-1: (x_{P+N-1}, h_{P+N-2})
//! ```
//!
//! The h column is therefore `h_trunk` *shifted right by 1*, with the first
//! slot filled from `pending_h`. We then stash `h_{P+N-1}` as the new
//! `pending_h` for the next round.
//!
//! ## Greedy-only (v1)
//!
//! `draft` samples top-1 from the MTP logits. The engine accepts a draft token
//! iff the trunk's verify-batch argmax matches — i.e. greedy speculative
//! decoding. Temperature > 0 → engine must skip the MTP path and fall back
//! to single-token decode (Phase 5 enforces this).

use std::collections::HashMap;

use candle_core::{Result as CandleResult, Tensor, D};
use candle_nn::{Embedding, Module, RmsNorm};

use super::mtp::MtpBlock;
use super::proj::ProjLinear;

/// Speculative draft driver for an [`MtpBlock`]. Owns the per-seq
/// carryover/verify hidden tensors and the underlying MTP block (which in turn
/// owns its own KV cache, kept in lock-step with the trunk's via `accept` +
/// `process_after_trunk`).
pub struct MtpDrafter {
    block: MtpBlock,
    n_embd: usize,
    /// Per-seq RoPE/KV position cursor for the MTP block — the position the
    /// NEXT `forward` row will write to. Bumped by `process_after_trunk` and
    /// `draft`; rewound by `accept` when the trunk rejected some draft tail.
    mtp_pos: HashMap<u64, usize>,
    /// Per-seq pre-norm hidden carryover for the right-shift seeding. Holds
    /// `h_{p-1}` where `p` is the position of the NEXT row we'll feed MTP.
    /// Always shape `[1, 1, n_embd]` for storage convenience.
    pending_h: HashMap<u64, Tensor>,
    /// Per-seq snapshot of the trunk's pre-norm hidden across the most recent
    /// `process_after_trunk` call. Shape `[T, n_embd]` (no batch axis). The
    /// engine's `accept(seq, k)` updates `pending_h` from row `k` of this.
    verify_h: HashMap<u64, Tensor>,
    /// Per-seq number of rows in the last draft (so `accept` can rewind MTP KV
    /// past the rejected tail).
    last_n_drafted: HashMap<u64, usize>,
}

impl MtpDrafter {
    pub fn new(block: MtpBlock, n_embd: usize) -> Self {
        Self {
            block,
            n_embd,
            mtp_pos: HashMap::new(),
            pending_h: HashMap::new(),
            verify_h: HashMap::new(),
            last_n_drafted: HashMap::new(),
        }
    }

    pub fn block(&self) -> &MtpBlock {
        &self.block
    }

    pub fn block_mut(&mut self) -> &mut MtpBlock {
        &mut self.block
    }

    pub fn n_embd(&self) -> usize {
        self.n_embd
    }

    /// Allocate per-seq state. `max_seq_len` sizes the MTP block's KV cache
    /// (matches what `enable_kv_cache` does for the trunk's SelfAttention).
    pub fn init_sequence(&mut self, seq_id: u64, max_seq_len: usize, device: &candle_core::Device) {
        // Lazily enable the MTP block's KV cache on first sequence; safe to
        // re-call across multiple seqs since `enable_kv_cache` is idempotent
        // when the requested length doesn't grow.
        self.block.enable_kv_cache(max_seq_len);
        self.block.init_seq_kv_cache(seq_id);
        self.mtp_pos.insert(seq_id, 0);
        // Initial pending_h is zero: matches llama.cpp's
        // `pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f))`. The
        // first `process_after_trunk` call's row-0 input will therefore pair
        // the prompt's first token with zeros, which is exactly what MTP's
        // attention sees at position 0 (no prior hidden to depend on).
        let zeros = Tensor::zeros(&[1, 1, self.n_embd], candle_core::DType::F32, device)
            .expect("zero tensor alloc must not fail");
        self.pending_h.insert(seq_id, zeros);
        self.verify_h.remove(&seq_id);
        self.last_n_drafted.insert(seq_id, 0);
    }

    pub fn remove_sequence(&mut self, seq_id: u64) {
        self.block.remove_seq_kv_cache(seq_id);
        self.mtp_pos.remove(&seq_id);
        self.pending_h.remove(&seq_id);
        self.verify_h.remove(&seq_id);
        self.last_n_drafted.remove(&seq_id);
    }

    /// Mirror a trunk batch into the MTP block's KV.
    ///
    /// * `tokens` are the trunk's input token IDs for the batch (length `T`).
    /// * `h_pre_trunk` is `[1, T, n_embd]` from the trunk's
    ///   [`Qwen3_5MoeTextModel::take_h_pre_norm`] right after the same batch
    ///   ran. Caller is responsible for taking it BEFORE the next trunk
    ///   forward overwrites it.
    /// * `embed_tokens` / `fallback_norm` / `fallback_lm_head` are the trunk's
    ///   own components borrowed for the MTP forward (MTP shares the trunk's
    ///   embedding + LM head when `mtp_use_dedicated_embeddings=false`, which
    ///   is true for Qwen3.6).
    ///
    /// After this call: MTP KV for `seq_id` has advanced by `T` positions,
    /// `verify_h[seq_id]` holds `h_pre_trunk` (without batch axis) for the
    /// engine's `accept` to consult, and `pending_h[seq_id]` is the last row
    /// of `h_pre_trunk` (the new right-shift seed for the next round).
    pub fn process_after_trunk(
        &mut self,
        seq_id: u64,
        tokens: &[u32],
        h_pre_trunk: &Tensor,
        embed_tokens: &Embedding,
        fallback_norm: &RmsNorm,
        fallback_lm_head: &ProjLinear,
    ) -> CandleResult<()> {
        let device = h_pre_trunk.device();
        let dims = h_pre_trunk.dims();
        if dims.len() != 3 || dims[0] != 1 || dims[2] != self.n_embd {
            return Err(candle_core::Error::Msg(format!(
                "process_after_trunk: expected [1, T, {}], got {:?}",
                self.n_embd, dims,
            )));
        }
        let n_tokens = dims[1];
        if n_tokens != tokens.len() {
            return Err(candle_core::Error::Msg(format!(
                "process_after_trunk: tokens.len()={} but h_pre_trunk T={}",
                tokens.len(),
                n_tokens
            )));
        }
        if n_tokens == 0 {
            return Ok(());
        }

        let pos = *self.mtp_pos.get(&seq_id).ok_or_else(|| {
            candle_core::Error::Msg(format!("process_after_trunk: seq {seq_id} not initialised"))
        })?;
        self.block.set_current_seq_id(seq_id);

        // Build the shift-right h column: [pending_h[seq], h_trunk[0..T-1]].
        let pending = self.pending_h.get(&seq_id).expect("init_sequence sets pending_h").clone();
        let h_shifted = if n_tokens == 1 {
            pending
        } else {
            // pending: [1, 1, n_embd]; we want [1, T, n_embd] = cat([pending, h[0..T-1]], axis=1).
            let h_head = h_pre_trunk.narrow(1, 0, n_tokens - 1)?;
            Tensor::cat(&[&pending, &h_head], 1)?
        };

        // Embed tokens: [T] → [1, T, n_embd].
        let toks = Tensor::from_vec(tokens.to_vec(), (1, n_tokens), device)?;
        let embeds = embed_tokens.forward(&toks)?;

        // Forward MTP at the current cursor. We don't need the output logits
        // here — the only effect we want is the KV cache advancing T positions
        // and (for the engine) the h_pre_trunk getting stashed for accept().
        let _ = self.block.forward(
            &embeds,
            &h_shifted,
            fallback_norm,
            fallback_lm_head,
            pos,
            None,
        )?;

        // Stash trunk hidden trajectory + advance bookkeeping.
        let h_2d = h_pre_trunk.squeeze(0)?; // [T, n_embd]
        self.verify_h.insert(seq_id, h_2d);

        let last_row = h_pre_trunk.narrow(1, n_tokens - 1, 1)?; // [1, 1, n_embd]
        self.pending_h.insert(seq_id, last_row);

        self.mtp_pos.insert(seq_id, pos + n_tokens);
        Ok(())
    }

    /// Autoregressively draft up to `n_max` tokens from the MTP block.
    ///
    /// `last_token` is the token the engine just sampled from the trunk's most
    /// recent decode — the first draft is *its* successor. `embed_tokens` etc.
    /// are the trunk's components (same as `process_after_trunk`).
    ///
    /// `pending_h` provides the first step's `h_input`; thereafter each step
    /// pairs the freshly sampled draft token with the previous step's
    /// `new_h_pre` (MTP's own pre-norm output). The MTP KV advances by exactly
    /// `n_max` positions on success (or fewer if a forward call errors mid-way
    /// — partial drafts are still returned).
    pub fn draft(
        &mut self,
        seq_id: u64,
        last_token: u32,
        n_max: usize,
        embed_tokens: &Embedding,
        fallback_norm: &RmsNorm,
        fallback_lm_head: &ProjLinear,
    ) -> CandleResult<Vec<u32>> {
        if n_max == 0 {
            self.last_n_drafted.insert(seq_id, 0);
            return Ok(Vec::new());
        }
        let mut drafts = Vec::with_capacity(n_max);

        let mut pos = *self.mtp_pos.get(&seq_id).ok_or_else(|| {
            candle_core::Error::Msg(format!("draft: seq {seq_id} not initialised"))
        })?;
        self.block.set_current_seq_id(seq_id);

        let mut cur_token = last_token;
        let mut h_input = self
            .pending_h
            .get(&seq_id)
            .expect("init_sequence sets pending_h")
            .clone();
        let device = h_input.device().clone();

        for _ in 0..n_max {
            // Embed current token → [1, 1, n_embd].
            let toks = Tensor::from_vec(vec![cur_token], (1, 1), &device)?;
            let embeds = embed_tokens.forward(&toks)?;

            let (logits, new_h_pre) = self.block.forward(
                &embeds,
                &h_input,
                fallback_norm,
                fallback_lm_head,
                pos,
                None,
            )?;

            // Greedy argmax over the vocab axis. logits: [1, 1, vocab].
            let next_tok = argmax_last_axis_u32(&logits)?;
            drafts.push(next_tok);

            // Roll: next step pairs the sampled token with MTP's own new_h_pre.
            cur_token = next_tok;
            h_input = new_h_pre;
            pos += 1;
        }

        // Stash the draft tail for `accept` to know how far to rewind.
        self.last_n_drafted.insert(seq_id, drafts.len());
        self.mtp_pos.insert(seq_id, pos);
        // pending_h is intentionally NOT updated here — it stays anchored to
        // the trunk's hidden, which only `process_after_trunk` mutates. After
        // the engine verifies + calls `accept`, that hook resets pending_h
        // from verify_h[n_accepted].
        Ok(drafts)
    }

    /// Reconcile MTP state with the trunk's verification result.
    ///
    /// `n_accepted` is how many of the most recent draft tokens the trunk
    /// agreed with (0 ≤ n_accepted ≤ last_n_drafted). The engine commits
    /// `1 + n_accepted` tokens total (the trunk's own sample plus the
    /// accepted prefix of MTP's drafts) — MTP's KV needs to:
    ///
    /// 1. Have its cursor rewound to the position right after the last
    ///    accepted draft (so the rejected tail's KV entries don't pollute
    ///    future drafts).
    /// 2. Have `pending_h` reset to the trunk's hidden at the same position,
    ///    so the next round's right-shift seed is correct.
    ///
    /// `n_committed_after` is the trunk's seq length after commit — used to
    /// validate that MTP and trunk agree on cursor placement. The engine
    /// computes it from its own prompt + commit bookkeeping.
    pub fn accept(
        &mut self,
        seq_id: u64,
        n_accepted: usize,
        n_committed_after: usize,
    ) -> CandleResult<()> {
        let n_drafted = self.last_n_drafted.get(&seq_id).copied().unwrap_or(0);
        if n_accepted > n_drafted {
            return Err(candle_core::Error::Msg(format!(
                "accept: n_accepted={n_accepted} > n_drafted={n_drafted}",
            )));
        }

        // 1) Rewind MTP block's KV to keep only the committed prefix.
        //
        //    During `draft` we advanced MTP KV by `n_drafted` rows; the trunk
        //    accepted `n_accepted` of those. The total committed prefix for
        //    this seq is `n_committed_after` rows; anything past that must go.
        self.block.set_current_seq_id(seq_id);
        self.block.truncate_kv_cache(n_committed_after);
        self.mtp_pos.insert(seq_id, n_committed_after);

        // 2) Reset pending_h to the trunk's hidden at row `n_accepted` of
        //    verify_h. That is the trunk's pre-norm hidden at the last
        //    committed position — the right-shift seed the next mirror needs.
        if let Some(verify) = self.verify_h.get(&seq_id) {
            let n_rows = verify.dims()[0];
            let idx = n_accepted.min(n_rows.saturating_sub(1));
            // verify: [T, n_embd] → row `idx` as [1, 1, n_embd].
            let row = verify.narrow(0, idx, 1)?.unsqueeze(0)?;
            self.pending_h.insert(seq_id, row);
        }

        self.last_n_drafted.insert(seq_id, 0);
        Ok(())
    }
}

/// Greedy argmax along the last axis of a `[1, T, vocab]` tensor with `T == 1`.
/// Returns the vocab index as u32.
fn argmax_last_axis_u32(logits: &Tensor) -> CandleResult<u32> {
    // Flatten to [vocab] then argmax. Cheaper than `argmax(D::Minus1)?` then
    // unsqueeze/squeeze dances on a 3D shape we know has [1, 1, V].
    let flat = logits.flatten_all()?.to_vec1::<f32>()?;
    let (idx, _) = flat
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
            if v > bv { (i, v) } else { (bi, bv) }
        });
    Ok(idx as u32)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen3_5_moe::moe::{DenseMlp, MlpBlock};
    use crate::qwen3_5_moe::mtp::MtpBlock;
    use crate::qwen3_5_moe::self_attn::{SelfAttention, SelfAttnDims, SelfAttnRuntime};

    use candle_core::{Device, Tensor};
    use candle_nn::{Embedding, Linear, RmsNorm};
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

    fn build_drafter(device: &Device) -> (MtpDrafter, Embedding, RmsNorm, ProjLinear) {
        let mut r = StdRng::seed_from_u64(0xDA47_DA47);
        let eh_proj = Linear::new(rnd(&[HIDDEN, 2 * HIDDEN], &mut r, device), None);
        let gate_up = Linear::new(rnd(&[2 * INTERMEDIATE, HIDDEN], &mut r, device), None);
        let down = Linear::new(rnd(&[HIDDEN, INTERMEDIATE], &mut r, device), None);
        let mlp = MlpBlock::Dense(DenseMlp::new(gate_up.into(), down.into(), INTERMEDIATE));
        let attn = tiny_self_attn(&mut r, device);
        let block = MtpBlock::new(
            eh_proj.into(),
            ones_norm(HIDDEN, device),
            ones_norm(HIDDEN, device),
            ones_norm(HIDDEN, device),
            attn,
            ones_norm(HIDDEN, device),
            mlp,
            Some(ones_norm(HIDDEN, device)),
            None,
        );

        let embed_w = rnd(&[VOCAB, HIDDEN], &mut r, device);
        let embed = Embedding::new(embed_w, HIDDEN);
        let fb_norm = ones_norm(HIDDEN, device);
        let fb_lm_head: ProjLinear =
            Linear::new(rnd(&[VOCAB, HIDDEN], &mut r, device), None).into();
        (MtpDrafter::new(block, HIDDEN), embed, fb_norm, fb_lm_head)
    }

    /// Full lifecycle: init → process_after_trunk on a 3-token "prompt" →
    /// draft 2 tokens → accept(1) → process_after_trunk again. Every step
    /// must succeed and leave the state machine in a usable position for
    /// the next call.
    #[test]
    fn mtp_drafter_full_lifecycle_runs_without_errors() {
        let device = Device::Cpu;
        let (mut drafter, embed, fb_norm, fb_lm_head) = build_drafter(&device);

        drafter.init_sequence(0, /*max_seq_len=*/ 16, &device);
        assert_eq!(drafter.n_embd(), HIDDEN);

        // Simulate the trunk producing a 3-token prefill: pretend it consumed
        // [x_0, x_1, x_2] and emitted h_pre_trunk of shape [1, 3, hidden].
        let mut r = StdRng::seed_from_u64(0xC0FFEE);
        let tokens_prefill = vec![1u32, 7, 11];
        let h_prefill = rnd(&[1, 3, HIDDEN], &mut r, &device);
        drafter
            .process_after_trunk(0, &tokens_prefill, &h_prefill, &embed, &fb_norm, &fb_lm_head)
            .expect("process_after_trunk must succeed on the initial prefill");

        // Draft 2 tokens. The engine just sampled some x_3 from the trunk; we
        // pretend it's token 5.
        let drafts = drafter
            .draft(0, /*last_token=*/ 5, /*n_max=*/ 2, &embed, &fb_norm, &fb_lm_head)
            .expect("draft must succeed");
        assert_eq!(drafts.len(), 2, "should return exactly n_max tokens");
        for d in &drafts {
            assert!((*d as usize) < VOCAB, "draft tokens must be in vocab");
        }

        // Engine accepts one of the two drafts. Final committed length:
        // 3 prefill + 1 (the sampled x_3) + 1 (accepted draft) = 5.
        drafter
            .accept(0, /*n_accepted=*/ 1, /*n_committed_after=*/ 5)
            .expect("accept must succeed");

        // Next round: simulate the trunk processing [x_3, x_{accepted_draft},
        // x_5] (verify of 3 rows). MTP must mirror those without error.
        let tokens_verify = vec![5u32, drafts[0], 9];
        let h_verify = rnd(&[1, 3, HIDDEN], &mut r, &device);
        drafter
            .process_after_trunk(0, &tokens_verify, &h_verify, &embed, &fb_norm, &fb_lm_head)
            .expect("process_after_trunk must succeed after accept");

        drafter.remove_sequence(0);
    }

    /// n_max=0 returns an empty draft and is a no-op on KV.
    #[test]
    fn mtp_drafter_draft_zero_is_noop() {
        let device = Device::Cpu;
        let (mut drafter, embed, fb_norm, fb_lm_head) = build_drafter(&device);
        drafter.init_sequence(7, 16, &device);

        let drafts = drafter
            .draft(7, 3, /*n_max=*/ 0, &embed, &fb_norm, &fb_lm_head)
            .expect("draft(n_max=0) must succeed");
        assert!(drafts.is_empty());
    }

    /// `accept` rejects an out-of-range `n_accepted` rather than silently
    /// rewinding past the prompt.
    #[test]
    fn mtp_drafter_accept_rejects_overrun() {
        let device = Device::Cpu;
        let (mut drafter, embed, fb_norm, fb_lm_head) = build_drafter(&device);
        drafter.init_sequence(0, 16, &device);
        let mut r = StdRng::seed_from_u64(0xBEEFFACE);
        let tokens = vec![3u32];
        let h = rnd(&[1, 1, HIDDEN], &mut r, &device);
        drafter
            .process_after_trunk(0, &tokens, &h, &embed, &fb_norm, &fb_lm_head)
            .unwrap();

        let drafts = drafter
            .draft(0, 5, 2, &embed, &fb_norm, &fb_lm_head)
            .unwrap();
        assert_eq!(drafts.len(), 2);

        // n_accepted = 99 is larger than n_drafted = 2.
        let err = drafter
            .accept(0, /*n_accepted=*/ 99, /*n_committed_after=*/ 100)
            .expect_err("accept must reject n_accepted > n_drafted");
        assert!(format!("{err}").contains("n_accepted"));
    }

    /// Calling lifecycle methods on an uninitialized seq_id must error
    /// rather than panicking deep inside the block forward.
    #[test]
    fn mtp_drafter_rejects_uninitialised_seq() {
        let device = Device::Cpu;
        let (mut drafter, embed, fb_norm, fb_lm_head) = build_drafter(&device);

        // No init_sequence call.
        let mut r = StdRng::seed_from_u64(0xDEAD);
        let tokens = vec![0u32];
        let h = rnd(&[1, 1, HIDDEN], &mut r, &device);
        let err = drafter
            .process_after_trunk(42, &tokens, &h, &embed, &fb_norm, &fb_lm_head)
            .expect_err("must error on uninitialised seq");
        let msg = format!("{err}");
        assert!(msg.contains("42") && msg.contains("not initialised"));
    }
}
