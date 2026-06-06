//! MLX-side bridge for the shared `lumen_core::sampling` pipeline.
//!
//! The actual sampling math (penalty / temperature / softmax / top-p /
//! multinomial draw) lives in `lumen-core` so the Candle / Qwen / Gemma
//! legacy backends can reuse it. This module just pulls the last-
//! position logits off the GPU into a CPU `Vec<f32>` and forwards the
//! rest to `lumen_core::sampling::sample_from_logits`.

#[cfg(feature = "mlx-native")]
pub(crate) mod imp {
    use anyhow::{Context, Result};
    use mlx_rs::Array;

    pub use lumen_core::sampling::{SamplingConfig, Xorshift64, sample_from_logits};

    use crate::gemma4_critical_correction::CorrectionTable;
    use crate::grammar::Gemma4GrammarState;

    /// Phase B (v0.6.0 tool-call robustness) — context for the optional
    /// per-token logit correction step that runs immediately after the
    /// CPU pull, before any masks (grammar, EOS guard, DRY) modify the
    /// CPU logit buffer. See
    /// [`crate::gemma4_critical_correction::CorrectionTable`] for the
    /// math and binary format.
    ///
    /// Caller pre-builds this once per decode step (after the model's
    /// forward returns) by:
    ///   1. Taking the captured `h_for_lm_head` via
    ///      `NativeGemma4Model::take_captured_correction_h()`.
    ///   2. Pulling it to CPU as an f32 slice.
    ///   3. Borrowing the cached `CorrectionTable` from the backend's
    ///      OnceLock.
    ///
    /// `softcap` is the Gemma 4 `final_logit_softcapping` value (30.0 in
    /// every shipped variant), passed in so the sampler doesn't need to
    /// know about model-side config.
    pub struct LogitCorrectionCtx<'a> {
        pub table: &'a CorrectionTable,
        pub hidden_f32: &'a [f32],
        pub softcap: f32,
    }

    /// Apply `LogitCorrectionCtx` to a CPU-pulled logit buffer in-place.
    /// No-op when `ctx` is `None`. Errors propagate from
    /// `CorrectionTable::apply_to_logits` (mainly hidden-dim mismatch).
    #[inline]
    fn apply_optional_correction(
        buf: &mut [f32],
        ctx: Option<&LogitCorrectionCtx<'_>>,
    ) -> Result<()> {
        if let Some(c) = ctx {
            c.table
                .apply_to_logits(buf, c.hidden_f32, c.softcap)
                .context("sample: apply logit correction")?;
        }
        Ok(())
    }

    /// Pull the last-position logits out of a `[1, L, V]` Array as a CPU
    /// `Vec<f32>`. Caller takes ownership so subsequent mutations
    /// (penalty / temperature / softmax) stay on the CPU. Casts to f32
    /// (most decode paths leave logits in f32 already; bf16 prefill
    /// paths get one conversion here).
    pub fn last_logits_to_cpu_f32(logits: &Array) -> Result<Vec<f32>> {
        let l = logits.shape()[1];
        let last_pos = l - 1;
        let last_idx = Array::from_slice(&[last_pos], &[1]);
        let last_logits = mlx_rs::ops::indexing::take_axis(logits, &last_idx, 1)
            .context("sample: take_axis(L-1)")?;
        let f32_view = last_logits
            .as_dtype(mlx_rs::Dtype::Float32)
            .context("sample: cast to f32")?;
        f32_view.eval().context("sample: eval")?;
        Ok(f32_view.as_slice::<f32>().to_vec())
    }

    /// One-shot helper: take the [1, L, V] logits array, sample the
    /// next token id using the supplied recent-token window and
    /// `lumen_core` sampling config. Hides the GPU→CPU pull so callers
    /// in the decode loop stay one line.
    pub fn sample_next_token(
        logits: &Array,
        recent_tokens: &[u32],
        cfg: &SamplingConfig,
        rng: &mut Xorshift64,
        correction: Option<&LogitCorrectionCtx<'_>>,
    ) -> Result<u32> {
        let mut buf = last_logits_to_cpu_f32(logits)?;
        apply_optional_correction(&mut buf, correction)?;
        Ok(sample_from_logits(&mut buf, recent_tokens, cfg, rng))
    }

    /// Grammar-constrained variant. When `grammar` is `Some` and the
    /// matcher is in its active (constraining) state, this masks logits
    /// to the set of tokens allowed by the grammar before sampling, then
    /// notifies the matcher of the chosen token so subsequent calls reflect
    /// the new parser state.
    ///
    /// `grammar = None` is identical to [`sample_next_token`] — callers
    /// can blind-pass `None` when `tool_choice` is `"none"` or the request
    /// has no tools.
    ///
    /// In lazy mode the first call before `<|tool_call>` (id 48) is
    /// emitted will see `state.is_active() == false`, so the mask is
    /// skipped and the model samples freely. `observe()` runs every step
    /// regardless so the lazy trigger fires the moment the model emits 48.
    pub fn sample_next_token_with_grammar(
        logits: &Array,
        recent_tokens: &[u32],
        cfg: &SamplingConfig,
        rng: &mut Xorshift64,
        grammar: Option<&mut Gemma4GrammarState>,
        correction: Option<&LogitCorrectionCtx<'_>>,
    ) -> Result<u32> {
        let mut buf = last_logits_to_cpu_f32(logits)?;
        apply_optional_correction(&mut buf, correction)?;
        let next = match grammar {
            Some(state) => {
                if state.is_active() {
                    let _ = state.apply_mask_to_logits(&mut buf)?;
                }
                let tok = sample_from_logits(&mut buf, recent_tokens, cfg, rng);
                state.observe(tok)?;
                tok
            }
            None => sample_from_logits(&mut buf, recent_tokens, cfg, rng),
        };
        Ok(next)
    }

    /// Variant that masks "soft EOS" tokens (token id 106 = `<turn|>` for
    /// Gemma 4) using three complementary guards so the soft EOS is only
    /// honored when the model actually wants to end its turn:
    ///
    /// 1. **Hard min-tokens guard** (`min_tokens`) — for the first N
    ///    decoded tokens, `<turn|>` is unconditionally masked. Use when
    ///    the expected response is at least N tokens long.
    ///
    /// 2. **Soft top-K outlier guard** (`eos_top_k_guard`) — at every
    ///    step (after the min-tokens window), `<turn|>` must be in the
    ///    top-K of raw logits. Catches "sampling outlier" cases where
    ///    the random multinomial draw lands on a low-rank EOS token.
    ///
    /// 3. **Logit-margin confidence guard** (`eos_min_logit_margin`) —
    ///    even when `<turn|>` IS top-1, its logit must exceed the best
    ///    non-EOS continuation by at least N units. Equivalent to
    ///    "softmax(EOS) / softmax(top_non_EOS) >= exp(N)". Catches
    ///    "model is conflicted between ending and continuing" cases at
    ///    structural list boundaries (e.g. between bullet items) where
    ///    the model legitimately rates EOS as top-1 but with weak
    ///    confidence (margin ≈ 0.1).
    ///
    /// Both soft guards fire INDEPENDENTLY — if either condition is
    /// violated, EOS is masked. Pass 0 to either to disable that guard.
    /// Margin units are natural-log-prob-ratio: 0.7 ≈ 2× ratio, 1.4 ≈ 4×,
    /// 2.3 ≈ 10×, 3.0 ≈ 20× confidence over the runner-up.
    ///
    /// Hard EOS (id 1) and tool-response signal (id 50) are NEVER
    /// masked — only `<turn|>` (id 106).
    pub fn sample_next_token_with_eos_guard(
        logits: &Array,
        recent_tokens: &[u32],
        cfg: &SamplingConfig,
        rng: &mut Xorshift64,
        min_tokens: usize,
        eos_top_k_guard: usize,
        eos_min_logit_margin: f32,
        eos_tokens: &[u32],
        correction: Option<&LogitCorrectionCtx<'_>>,
    ) -> Result<u32> {
        let mut buf = last_logits_to_cpu_f32(logits)?;
        apply_optional_correction(&mut buf, correction)?;
        let vocab = buf.len();

        // Filter caller-supplied EOS ids to in-vocab positions. Empty set
        // = no guarding (matches plain `sample_next_token`).
        let valid_eos: Vec<usize> = eos_tokens
            .iter()
            .map(|&id| id as usize)
            .filter(|&id| id < vocab)
            .collect();
        if valid_eos.is_empty() {
            return Ok(sample_from_logits(&mut buf, recent_tokens, cfg, rng));
        }

        let verbose = std::env::var("LUMEN_EOS_GUARD_VERBOSE")
            .map(|v| !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(false);

        // Snapshot original EOS logits for diagnostic logging (cheap —
        // bounded by valid_eos.len(), usually 3).
        let original_eos_logits: Vec<f32> = if verbose {
            valid_eos.iter().map(|&id| buf[id]).collect()
        } else {
            Vec::new()
        };

        let eos_set: std::collections::HashSet<usize> = valid_eos.iter().copied().collect();

        // Compute `max_non_eos` (highest logit OUTSIDE the EOS set) once.
        // Needed by both the margin guard and the verbose logger.
        let need_max_non_eos = eos_top_k_guard > 0 || eos_min_logit_margin > 0.0 || verbose;
        let max_non_eos: f32 = if need_max_non_eos {
            buf.iter()
                .enumerate()
                .filter(|(i, _)| !eos_set.contains(i))
                .map(|(_, &v)| v)
                .fold(f32::NEG_INFINITY, f32::max)
        } else {
            f32::NEG_INFINITY
        };

        // (1) Hard guard: mask every EOS token in early window.
        if min_tokens > 0 && recent_tokens.len() < min_tokens {
            for &id in &valid_eos {
                buf[id] = f32::NEG_INFINITY;
            }
        } else if eos_top_k_guard > 0 || eos_min_logit_margin > 0.0 {
            // (2)+(3) Soft guards applied independently per EOS token.
            // An EOS token is masked if EITHER its rank in raw logits
            // is below top-K OR its margin over max_non_eos is below
            // the threshold.
            for &id in &valid_eos {
                let eos_logit = buf[id];
                if !eos_logit.is_finite() {
                    continue;
                }
                let fails_top_k = if eos_top_k_guard > 0 {
                    let mut higher = 0usize;
                    for (i, &v) in buf.iter().enumerate() {
                        if i == id {
                            continue;
                        }
                        if v > eos_logit {
                            higher += 1;
                            if higher >= eos_top_k_guard {
                                break;
                            }
                        }
                    }
                    higher >= eos_top_k_guard
                } else {
                    false
                };
                let fails_margin =
                    eos_min_logit_margin > 0.0 && (eos_logit - max_non_eos) < eos_min_logit_margin;
                if fails_top_k || fails_margin {
                    buf[id] = f32::NEG_INFINITY;
                }
            }
        }

        let next = sample_from_logits(&mut buf, recent_tokens, cfg, rng);

        // Diagnostic: when an EOS token is actually sampled, dump the
        // full stats so we can see exactly which guard would have caught
        // it (and why it didn't, if any).
        if verbose && eos_set.contains(&(next as usize)) {
            eprintln!(
                "[eos-guard] step={} sampled_eos={} max_non_eos={:.3} guards: min_tok={} top_k={} margin={:.2}",
                recent_tokens.len(),
                next,
                max_non_eos,
                min_tokens,
                eos_top_k_guard,
                eos_min_logit_margin,
            );
            for (i, &id) in valid_eos.iter().enumerate() {
                let logit = original_eos_logits[i];
                let margin = logit - max_non_eos;
                let final_logit = buf[id];
                let masked = !final_logit.is_finite();
                eprintln!(
                    "  eos[id={id}] orig_logit={logit:.3} margin={margin:.3} masked={masked}{}",
                    if id == next as usize {
                        " <-- SAMPLED"
                    } else {
                        ""
                    }
                );
            }
        }

        Ok(next)
    }

    /// Combined variant for the chat-streaming decode loop: grammar-
    /// constrained when active, EOS-guarded otherwise.
    ///
    /// Rules:
    ///   - If `grammar.is_active()` returns true, the grammar mask wins:
    ///     non-allowed positions are set to `-inf` and the EOS guard is
    ///     skipped (grammar already constrains termination to the
    ///     tool-call schema's accepting states).
    ///   - Otherwise (no grammar OR lazy pre-trigger), behaviour matches
    ///     [`sample_next_token_with_eos_guard`] exactly so bit-identical
    ///     decode is preserved for the no-tools / `tool_choice="none"`
    ///     path.
    ///
    /// In either case, after sampling the chosen token is fed back into
    /// `grammar.observe()` so the lazy trigger fires when `<|tool_call>`
    /// (id 48) appears and the matcher advances on subsequent steps.
    pub fn sample_next_token_with_eos_guard_and_grammar(
        logits: &Array,
        recent_tokens: &[u32],
        cfg: &SamplingConfig,
        rng: &mut Xorshift64,
        min_tokens: usize,
        eos_top_k_guard: usize,
        eos_min_logit_margin: f32,
        eos_tokens: &[u32],
        mut grammar: Option<&mut Gemma4GrammarState>,
        correction: Option<&LogitCorrectionCtx<'_>>,
    ) -> Result<u32> {
        let grammar_active = grammar.as_ref().is_some_and(|g| g.is_active());
        if grammar_active {
            let mut buf = last_logits_to_cpu_f32(logits)?;
            apply_optional_correction(&mut buf, correction)?;
            if let Some(state) = grammar.as_mut() {
                let _ = state.apply_mask_to_logits(&mut buf)?;
            }
            let next = sample_from_logits(&mut buf, recent_tokens, cfg, rng);
            if let Some(state) = grammar.as_mut() {
                state.observe(next)?;
            }
            return Ok(next);
        }
        // Grammar idle (None OR lazy-pre-trigger). Fall back to the EOS-
        // guarded path verbatim, then notify the grammar state so the
        // lazy trigger can fire when `<|tool_call>` is sampled.
        let next = sample_next_token_with_eos_guard(
            logits,
            recent_tokens,
            cfg,
            rng,
            min_tokens,
            eos_top_k_guard,
            eos_min_logit_margin,
            eos_tokens,
            correction,
        )?;
        if let Some(state) = grammar.as_mut() {
            state.observe(next)?;
        }
        Ok(next)
    }
}
