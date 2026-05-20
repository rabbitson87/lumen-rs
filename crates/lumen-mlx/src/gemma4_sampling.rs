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
    ) -> Result<u32> {
        let mut buf = last_logits_to_cpu_f32(logits)?;
        Ok(sample_from_logits(&mut buf, recent_tokens, cfg, rng))
    }
}
