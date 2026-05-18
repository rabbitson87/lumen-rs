use anyhow::{Context, Result};
use candle_core::{DType, Tensor};

/// GPU-resident sampler. Keeps the full logit tensor on device, applies
/// optional repeat-penalty via index_add (log-space), then samples via
/// Gumbel-max (`argmax(logits/T + G)` where `G ~ -ln(-ln(U))`). Only a
/// single `u32` crosses the GPU↔CPU boundary — skips the 1 MB F16→F32
/// materialization that dominates CPU-side sampling at 262k vocab.
///
/// Supports: greedy (`temperature<=0`), stochastic (`temperature>0`),
/// and log-space repeat penalty (if `repeat_penalty>1`). Skips top-p
/// (nucleus) filtering and n-gram penalty — callers that need those
/// must stay on `sample_token_cpu`.
///
/// `logits` must be 1D `[vocab]` on the target device, dtype F16 or F32.
pub fn sample_token_gpu(
    logits: &Tensor,
    temperature: f32,
    repeat_penalty: f32,
    generated_tokens: &[u32],
) -> Result<u32> {
    let device = logits.device();
    let logits = if logits.dtype() == DType::F32 {
        logits.clone()
    } else {
        logits.to_dtype(DType::F32)?
    };

    // Log-space repeat penalty: subtract ln(penalty)*count from each
    // recently-seen token's logit. Equivalent to dividing its probability
    // by penalty^count under softmax. Only applied when rp>1.
    let logits = if repeat_penalty > 1.0 && !generated_tokens.is_empty() {
        let window = generated_tokens.len().min(256);
        let recent = &generated_tokens[generated_tokens.len() - window..];
        let mut counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        for &t in recent {
            *counts.entry(t).or_insert(0) += 1;
        }
        let ln_rp = repeat_penalty.ln();
        let (idxs, vals): (Vec<u32>, Vec<f32>) = counts
            .into_iter()
            .map(|(tok, c)| (tok, -(ln_rp * c as f32)))
            .unzip();
        let n = vals.len();
        let idxs_t = Tensor::from_vec(idxs, (n,), device)?;
        let vals_t = Tensor::from_vec(vals, (n,), device)?;
        logits.index_add(&idxs_t, &vals_t, 0)?
    } else {
        logits
    };

    if temperature <= 0.0 {
        // Greedy: single argmax kernel.
        return logits
            .argmax(0)?
            .to_scalar::<u32>()
            .context("gpu argmax failed");
    }

    // Gumbel-max sampling: `argmax(logits/T + G)` where G_i ~ Gumbel(0,1).
    // Gumbel sample: G = -ln(-ln(U)), U ~ Uniform(eps, 1).
    let scaled = (logits / temperature as f64)?;
    let u = Tensor::rand(1e-7f32, 1.0f32, scaled.shape(), device)?;
    let gumbel = u.log()?.neg()?.log()?.neg()?;
    let perturbed = (scaled + gumbel)?;
    perturbed
        .argmax(0)?
        .to_scalar::<u32>()
        .context("gpu gumbel-argmax failed")
}

/// Sample next token from logits with temperature and top-p.
pub fn sample_token(logits: &Tensor, temperature: f32, top_p: f32) -> Result<u32> {
    let logits = logits.squeeze(0)?.squeeze(0)?; // [vocab_size]
    let logits = logits.to_dtype(DType::F32)?;

    if temperature <= 0.0 {
        let token = logits
            .argmax(0)?
            .to_scalar::<u32>()
            .context("argmax failed")?;
        return Ok(token);
    }

    let logits = (&logits / temperature as f64)?;
    let probs = candle_nn::ops::softmax_last_dim(&logits.unsqueeze(0)?)?.squeeze(0)?;
    let probs_vec: Vec<f32> = probs.to_vec1()?;

    if top_p >= 1.0 {
        return sample_from_probs(&probs_vec);
    }

    // Top-p (nucleus) sampling
    let mut indexed: Vec<(usize, f32)> = probs_vec.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut cumulative = 0.0;
    let mut filtered = Vec::new();
    for (idx, p) in indexed {
        cumulative += p;
        filtered.push((idx, p));
        if cumulative >= top_p {
            break;
        }
    }

    let sum: f32 = filtered.iter().map(|(_, p)| p).sum();
    let normalized: Vec<(usize, f32)> = filtered.iter().map(|(i, p)| (*i, p / sum)).collect();
    sample_from_indexed(&normalized)
}

fn sample_from_probs(probs: &[f32]) -> Result<u32> {
    use rand::RngExt;
    let mut rng = rand::rng();
    let r: f32 = rng.random();
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return Ok(i as u32);
        }
    }
    Ok((probs.len() - 1) as u32)
}

fn sample_from_indexed(indexed: &[(usize, f32)]) -> Result<u32> {
    use rand::RngExt;
    let mut rng = rand::rng();
    let r: f32 = rng.random();
    let mut cumulative = 0.0;
    for &(idx, p) in indexed {
        cumulative += p;
        if r < cumulative {
            return Ok(idx as u32);
        }
    }
    Ok(indexed.last().map(|(i, _)| *i as u32).unwrap_or(0))
}

/// CPU-only sampling with frequency penalty and n-gram repetition penalty.
///
/// - `repeat_penalty`: frequency-based penalty over sliding window (1.0 = disabled)
/// - N-gram penalty: prevents repeating 4-token sequences (always active)
pub fn sample_token_cpu(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    repeat_penalty: f32,
    generated_tokens: &[u32],
) -> Result<u32> {
    let mut logits = logits.to_vec();

    apply_repeat_penalty_cpu(&mut logits, repeat_penalty, generated_tokens);

    // N-gram repetition penalty: prevent repeating 4-token sequences
    apply_ngram_penalty(&mut logits, generated_tokens, 4, 5.0);

    sample_token_cpu_inner(&logits, temperature, top_p)
}

/// CPU sampling with top-k + top-p + repeat penalty. Used by Qwen3 chat path
/// where generation_config specifies top_k=20, top_p=0.95, temperature=1.0.
///
/// `top_k = 0` disables the top-k filter (falls back to `sample_token_cpu`).
pub fn sample_token_cpu_full(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    top_k: usize,
    repeat_penalty: f32,
    generated_tokens: &[u32],
) -> Result<u32> {
    if top_k == 0 {
        return sample_token_cpu(logits, temperature, top_p, repeat_penalty, generated_tokens);
    }
    sample_token_cpu_full_owned(
        logits.to_vec(),
        temperature,
        top_p,
        top_k,
        repeat_penalty,
        generated_tokens,
    )
}

/// point. Avoids the
/// `logits.to_vec()` clone the borrowed-slice variant pays — caller can
/// pass an already-owned `Vec<f32>` (e.g. straight out of
/// `tensor_to_vec_f32`) and we mutate it in place for penalties.
///
/// Three sub-optimizations vs the prior implementation, all token bit-
/// identical for the top_k > 0 branch:
///
///   (a) Skip the full-vocab `-inf` mask sweep. After locating the K-th
///       value via `select_nth_unstable_by`, collect ONLY the `>= kth`
///       survivors into an `(idx, val)` Vec of length ~K. Downstream ops
///       run on this small slice instead of the 248K-elem masked array.
///   (b) Take `logits` by value (no internal clone of the 248K-elem vec).
///   (c) Softmax + top-p over K survivors only (~20 elements) instead of
///       the full vocab — the exp+sum and the sort previously ran over
///       248K elements where 248,300 of them were `-inf` placeholders.
///
/// Empirically on 35B (vocab=248320, top_k=20) this dropped the sampling
/// time well below the prior 0.83 ms baseline established by the
/// select_nth_unstable_by single-step refactor.
pub fn sample_token_cpu_full_owned(
    mut logits: Vec<f32>,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    repeat_penalty: f32,
    generated_tokens: &[u32],
) -> Result<u32> {
    if top_k == 0 {
        return sample_token_cpu_inner(&logits, temperature, top_p);
    }

    apply_repeat_penalty_cpu(&mut logits, repeat_penalty, generated_tokens);

    apply_ngram_penalty(&mut logits, generated_tokens, 4, 5.0);

    // (a) Find the K-th largest value via partial selection. Clone is
    // unavoidable because penalties already mutated specific slots and we
    // need original positions intact for the survivor-collection step.
    if top_k >= logits.len() {
        return sample_token_cpu_inner(&logits, temperature, top_p);
    }
    let kth = {
        let k_idx = top_k - 1;
        let mut scratch = logits.clone();
        let (_, kth_ref, _) = scratch.select_nth_unstable_by(k_idx, |a, b| {
            b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
        });
        *kth_ref
    };

    // (a)+(c) Collect survivors (>= kth) with their original indices and
    // run softmax/top-p over just this small slice. Ties at kth are kept
    // — same as the prior "mask < kth with -inf" semantics (exp(-inf)=0
    // contributed 0 to the softmax sum).
    let top_pairs: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .filter(|&(_, v)| *v >= kth)
        .map(|(i, v)| (i, *v))
        .collect();

    sample_from_top_pairs(&top_pairs, temperature, top_p)
}

/// softmax + optional top-p over a pre-filtered
/// `(token_index, logit)` list (typically ~K=20 elements). All loops run
/// over `pairs.len()`, never over the full vocab.
fn sample_from_top_pairs(pairs: &[(usize, f32)], temperature: f32, top_p: f32) -> Result<u32> {
    if pairs.is_empty() {
        return Err(anyhow::anyhow!("sample_from_top_pairs: empty pairs"));
    }

    if temperature <= 0.0 {
        let (idx, _) = pairs
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
            .ok_or_else(|| anyhow::anyhow!("empty pairs"))?;
        return Ok(idx as u32);
    }

    let max_v = pairs
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = pairs
        .iter()
        .map(|(_, v)| ((v - max_v) / temperature).exp())
        .collect();
    let exp_sum: f32 = exps.iter().sum();
    if exp_sum == 0.0 || !exp_sum.is_finite() {
        return Err(anyhow::anyhow!("sample_from_top_pairs: degenerate exp_sum"));
    }
    let probs: Vec<(usize, f32)> = pairs
        .iter()
        .zip(exps.iter())
        .map(|(&(i, _), &e)| (i, e / exp_sum))
        .collect();

    if top_p >= 1.0 {
        return sample_from_indexed(&probs);
    }

    // Top-p (nucleus) over K elements — cheap to sort even by full sort.
    let mut sorted = probs;
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut cumulative = 0.0;
    let mut filtered: Vec<(usize, f32)> = Vec::new();
    for (idx, p) in sorted {
        cumulative += p;
        filtered.push((idx, p));
        if cumulative >= top_p {
            break;
        }
    }
    let sum: f32 = filtered.iter().map(|(_, p)| p).sum();
    if sum == 0.0 {
        return Err(anyhow::anyhow!(
            "sample_from_top_pairs: degenerate top-p sum"
        ));
    }
    let normalized: Vec<(usize, f32)> = filtered.iter().map(|(i, p)| (*i, p / sum)).collect();
    sample_from_indexed(&normalized)
}

fn apply_repeat_penalty_cpu(logits: &mut [f32], repeat_penalty: f32, generated_tokens: &[u32]) {
    if repeat_penalty == 1.0 || generated_tokens.is_empty() {
        return;
    }

    let window = generated_tokens.len().min(256);
    let recent = &generated_tokens[generated_tokens.len() - window..];
    let mut tokens = recent.to_vec();
    tokens.sort_unstable();

    let mut i = 0;
    while i < tokens.len() {
        let tok = tokens[i];
        let mut count = 1usize;
        while i + count < tokens.len() && tokens[i + count] == tok {
            count += 1;
        }

        let idx = tok as usize;
        if idx < logits.len() {
            let penalty = repeat_penalty.powi(count as i32);
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }

        i += count;
    }
}

/// Penalize tokens that would form a repeated n-gram.
fn apply_ngram_penalty(logits: &mut [f32], tokens: &[u32], n: usize, penalty: f32) {
    if tokens.len() < n {
        return;
    }
    let prefix = &tokens[tokens.len() - (n - 1)..];
    let window = tokens.len().min(256);
    let start = tokens.len() - window;
    let end = tokens.len() - (n - 1);

    for i in start..end {
        if i + n > tokens.len() {
            break;
        }
        if tokens[i..i + n - 1] == *prefix {
            let next_tok = tokens[i + n - 1] as usize;
            if next_tok < logits.len() {
                if logits[next_tok] > 0.0 {
                    logits[next_tok] /= penalty;
                } else {
                    logits[next_tok] *= penalty;
                }
            }
        }
    }
}

fn sample_token_cpu_inner(logits: &[f32], temperature: f32, top_p: f32) -> Result<u32> {
    if temperature <= 0.0 {
        let (max_idx, _) = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .ok_or_else(|| anyhow::anyhow!("empty logits"))?;
        return Ok(max_idx as u32);
    }

    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits
        .iter()
        .map(|&x| ((x - max_logit) / temperature).exp())
        .sum();
    let probs: Vec<f32> = logits
        .iter()
        .map(|&x| ((x - max_logit) / temperature).exp() / exp_sum)
        .collect();

    if top_p >= 1.0 {
        return sample_from_probs(&probs);
    }

    // Top-p (nucleus) sampling
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut cumulative = 0.0;
    let mut filtered = Vec::new();
    for (idx, p) in indexed {
        cumulative += p;
        filtered.push((idx, p));
        if cumulative >= top_p {
            break;
        }
    }

    let sum: f32 = filtered.iter().map(|(_, p)| p).sum();
    let normalized: Vec<(usize, f32)> = filtered.iter().map(|(i, p)| (*i, p / sum)).collect();
    sample_from_indexed(&normalized)
}

#[cfg(test)]
mod tests {
    use super::apply_repeat_penalty_cpu;

    #[test]
    fn repeat_penalty_counts_recent_tokens_once_per_unique_token() {
        let mut logits = vec![8.0, -2.0, 4.0, 1.0];
        apply_repeat_penalty_cpu(&mut logits, 2.0, &[0, 1, 0, 5]);

        assert_eq!(logits[0], 2.0);
        assert_eq!(logits[1], -4.0);
        assert_eq!(logits[2], 4.0);
        assert_eq!(logits[3], 1.0);
    }

    #[test]
    fn repeat_penalty_uses_last_256_tokens() {
        let mut logits = vec![16.0, 16.0];
        let mut generated = vec![0; 10];
        generated.extend(std::iter::repeat_n(1, 256));

        apply_repeat_penalty_cpu(&mut logits, 2.0, &generated);

        assert_eq!(logits[0], 16.0);
        assert_eq!(logits[1], 0.0);
    }
}
