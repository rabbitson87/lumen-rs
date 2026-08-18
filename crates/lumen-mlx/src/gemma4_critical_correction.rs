//! Runtime logit-correction kernel for Gemma 4 quantized variants.
//!
//! Restores bf16-equivalent logit values for a small set of critical
//! token ids (tool-call, channel, turn markers) on uniformly-quantized
//! 4-bit IT models. Designed to address the rank-degradation gap measured
//! in Phase A of the v0.6.0 tool-calling robustness work:
//!
//!   bf16            tool-required median rank ~412   (top-100 40%)
//!   uniform 4bit IT tool-required median rank ~4860  (top-100 0%)  — 11.8× damage
//!   imatrix-AWQ Q   tool-required median rank ~1738  (top-100 20%) — 4.2× damage
//!
//! The math: Gemma 4 has tied embedding, so the lm_head matmul is
//!   `logit_raw[i] = h · W[i, :]`
//! followed by a tanh softcap. Quantization perturbs each row of `W`
//! independently. For a small set of N critical ids, we precompute
//! offline (see `scripts/quant/precompute_logit_corrections.py`):
//!
//!   Δ[k, :] = W_bf16[critical_ids[k], :] - W_q_dq[critical_ids[k], :]
//!
//! Stored as a sidecar `logit_corrections.bin` in the quantized model
//! directory. At decode time:
//!
//!   1. Capture `h` at the lm_head input position (see
//!      [`NativeGemma4Model::take_captured_correction_h`]).
//!   2. Pull post-softcap logits to CPU (the sampler already does this).
//!   3. For each critical id k:
//!      raw_q   = softcap · atanh(logit[k] / softcap)
//!      delta_k = h · Δ[k, :]
//!      logit[k] = softcap · tanh((raw_q + delta_k) / softcap)
//!
//! Step 3 is N matrix-vector products plus 2 atanh/tanh ops — negligible
//! cost (~20 µs per decode step on M3 Max).
//!
//! Sidecar binary format (little-endian):
//!
//!   [u32]                magic = 0x4C544343 ("LTCC")
//!   [u32]                version = 1
//!   [u32]                n_critical
//!   [u32]                hidden
//!   [u32]                dtype_marker  (0 = bf16, 1 = f16)
//!   [u32 × 3]            reserved (zero)
//!   [u32 × n_critical]   critical_ids
//!   [bf16 × n_critical × hidden]   Δ matrix, row-major

use std::fs::File;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result, bail};

/// Sidecar filename inside a quantized model directory.
pub const SIDECAR_FILENAME: &str = "logit_corrections.bin";

const MAGIC: u32 = 0x4C544343; // "LTCC"
const VERSION: u32 = 1;
const HEADER_LEN: usize = 32;
const DTYPE_BF16: u32 = 0;
const DTYPE_F16: u32 = 1;

/// Per-token corrective delta in raw-logit space, stored as f32 on CPU
/// for fast `h · Δ[k, :]` evaluation during decode.
#[derive(Debug)]
pub struct CorrectionTable {
    /// Token ids that receive correction.
    pub critical_ids: Vec<u32>,
    /// Hidden dimension (must match the model's lm_head input width).
    pub hidden: usize,
    /// Δ matrix flat: `delta[k * hidden + i] = (W_bf16 - W_q_dq)[critical_ids[k], i]`.
    /// Stored as f32 to skip per-dot-product conversion at runtime.
    pub delta_f32: Vec<f32>,
}

impl CorrectionTable {
    /// Load from explicit path. Returns the parsed table or an error
    /// describing the validation failure (magic, version, size).
    pub fn load(path: &Path) -> Result<Self> {
        let mut f = File::open(path)
            .with_context(|| format!("open logit-correction sidecar: {}", path.display()))?;
        let mut header = [0u8; HEADER_LEN];
        f.read_exact(&mut header)
            .context("logit-correction: read header")?;
        let magic = u32::from_le_bytes(header[0..4].try_into().unwrap());
        if magic != MAGIC {
            bail!(
                "logit-correction sidecar magic mismatch: got 0x{:08X}, want 0x{:08X}",
                magic,
                MAGIC
            );
        }
        let version = u32::from_le_bytes(header[4..8].try_into().unwrap());
        if version != VERSION {
            bail!(
                "logit-correction sidecar version mismatch: got {}, want {}",
                version,
                VERSION
            );
        }
        let n_critical = u32::from_le_bytes(header[8..12].try_into().unwrap()) as usize;
        let hidden = u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;
        let dtype = u32::from_le_bytes(header[16..20].try_into().unwrap());
        if dtype != DTYPE_BF16 && dtype != DTYPE_F16 {
            bail!(
                "logit-correction sidecar unsupported dtype marker {}: only bf16(0)/f16(1)",
                dtype
            );
        }

        let mut id_bytes = vec![0u8; n_critical * 4];
        f.read_exact(&mut id_bytes)
            .context("logit-correction: read critical ids")?;
        let critical_ids: Vec<u32> = id_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        let delta_bytes_len = n_critical * hidden * 2;
        let mut delta_raw = vec![0u8; delta_bytes_len];
        f.read_exact(&mut delta_raw)
            .context("logit-correction: read delta matrix")?;

        // Convert bf16/f16 → f32 once. delta is small (~50 KB) so this
        // upfront cost is fine; runtime gets bare f32 dot products.
        let delta_f32: Vec<f32> = match dtype {
            DTYPE_BF16 => delta_raw
                .chunks_exact(2)
                .map(|c| {
                    let u16_val = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((u16_val as u32) << 16)
                })
                .collect(),
            DTYPE_F16 => delta_raw
                .chunks_exact(2)
                .map(|c| half_decode(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            _ => unreachable!(),
        };
        if delta_f32.len() != n_critical * hidden {
            bail!(
                "logit-correction delta length mismatch: parsed {} expected {}",
                delta_f32.len(),
                n_critical * hidden
            );
        }

        Ok(Self {
            critical_ids,
            hidden,
            delta_f32,
        })
    }

    /// Convenience: load `<model_dir>/logit_corrections.bin` if present.
    /// Returns `Ok(None)` (not an error) when the sidecar is absent —
    /// the backend treats that as "no correction" and degrades gracefully.
    pub fn load_from_model_dir(model_dir: &Path) -> Result<Option<Self>> {
        let path = model_dir.join(SIDECAR_FILENAME);
        if !path.exists() {
            return Ok(None);
        }
        Self::load(&path).map(Some)
    }

    /// Apply correction to a CPU-side logit slice in-place.
    ///
    /// `logits` is the post-softcap last-position vocab vector pulled
    /// to CPU (e.g. via `last_logits_to_cpu_f32`). `hidden_state` is the
    /// f32-converted lm_head input vector captured at the same forward
    /// step. `softcap` is Gemma 4's `final_logit_softcapping` (typically
    /// 30.0).
    ///
    /// For each critical id k:
    ///   1. `raw_q = softcap · atanh(logit[critical_ids[k]] / softcap)`
    ///   2. `delta_k = h · Δ[k, :]`
    ///   3. `logit[critical_ids[k]] = softcap · tanh((raw_q + delta_k) / softcap)`
    ///
    /// Total cost: `n_critical × (hidden FLOPs + ~10 cycles for atanh/tanh)`.
    /// At N=7, hidden=2816 this is ~20K FLOPs — under 5 µs per call.
    pub fn apply_to_logits(
        &self,
        logits: &mut [f32],
        hidden_state: &[f32],
        softcap: f32,
    ) -> Result<()> {
        if hidden_state.len() != self.hidden {
            bail!(
                "hidden state width {} does not match correction table hidden {}",
                hidden_state.len(),
                self.hidden
            );
        }
        if softcap <= 0.0 {
            bail!("softcap must be positive, got {softcap}");
        }
        let inv_softcap = 1.0 / softcap;

        for (k, &tid) in self.critical_ids.iter().enumerate() {
            let idx = tid as usize;
            if idx >= logits.len() {
                bail!(
                    "critical id {} out of bounds for logits len {}",
                    tid,
                    logits.len()
                );
            }
            // Step 1: unwrap softcap. Clamp ratio to avoid atanh singularity
            // — Gemma 4's softcap=30 means saturated raw logits would map
            // to ±30 post-softcap. Float atanh(±1) = ±inf; we clamp the
            // ratio just inside the open interval (-1, +1).
            let ratio = (logits[idx] * inv_softcap).clamp(-0.999999, 0.999999);
            let raw_q = atanh_f32(ratio) * softcap;

            // Step 2: dot product h · Δ[k, :].
            let row_start = k * self.hidden;
            let row = &self.delta_f32[row_start..row_start + self.hidden];
            let delta_k = dot_f32(hidden_state, row);

            // Step 3: re-apply softcap on corrected raw.
            let raw_corrected = raw_q + delta_k;
            logits[idx] = (raw_corrected * inv_softcap).tanh() * softcap;
        }
        Ok(())
    }
}

#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    // Unrolled by 4 for autovectorization hint; compiler will fold to
    // NEON FMA on AArch64.
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;
    let chunks = a.chunks_exact(4).zip(b.chunks_exact(4));
    let mut leftover_a: &[f32] = &[];
    let mut leftover_b: &[f32] = &[];
    for (ac, bc) in chunks {
        acc0 += ac[0] * bc[0];
        acc1 += ac[1] * bc[1];
        acc2 += ac[2] * bc[2];
        acc3 += ac[3] * bc[3];
    }
    let rem = a.len() % 4;
    if rem != 0 {
        leftover_a = &a[a.len() - rem..];
        leftover_b = &b[b.len() - rem..];
    }
    let mut tail = 0.0f32;
    for (x, y) in leftover_a.iter().zip(leftover_b.iter()) {
        tail += x * y;
    }
    acc0 + acc1 + acc2 + acc3 + tail
}

#[inline]
fn atanh_f32(x: f32) -> f32 {
    // Standard library provides `f32::atanh`. Kept as a thin wrapper so
    // we can swap in a domain-clamped variant if needed.
    x.atanh()
}

/// IEEE 754 binary16 (f16) decoder. Used only when sidecar carries
/// `dtype_marker = 1` (f16); current Python precompute emits bf16 by
/// default but the format reserves room for f16.
fn half_decode(u: u16) -> f32 {
    let sign = (u >> 15) & 0x1;
    let exp = ((u >> 10) & 0x1F) as i32;
    let frac = (u & 0x3FF) as u32;
    let (e, f) = if exp == 0 {
        if frac == 0 {
            (0u32, 0u32)
        } else {
            // Subnormal.
            let mut shift = 1u32;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                shift += 1;
            }
            let e = 127 - 15 - shift + 1;
            (e, (f & 0x3FF) << 13)
        }
    } else if exp == 0x1F {
        (0xFF, frac << 13)
    } else {
        ((exp + (127 - 15)) as u32, frac << 13)
    };
    let bits = ((sign as u32) << 31) | (e << 23) | f;
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;

    fn write_minimal_sidecar(tmp_dir: &Path, ids: &[u32], hidden: usize, fill: f32) -> PathBuf {
        let path = tmp_dir.join(SIDECAR_FILENAME);
        let mut f = File::create(&path).unwrap();
        let n = ids.len();
        f.write_all(&MAGIC.to_le_bytes()).unwrap();
        f.write_all(&VERSION.to_le_bytes()).unwrap();
        f.write_all(&(n as u32).to_le_bytes()).unwrap();
        f.write_all(&(hidden as u32).to_le_bytes()).unwrap();
        f.write_all(&DTYPE_BF16.to_le_bytes()).unwrap();
        for _ in 0..3 {
            f.write_all(&0u32.to_le_bytes()).unwrap();
        }
        for &id in ids {
            f.write_all(&id.to_le_bytes()).unwrap();
        }
        // Δ filled with `fill` bf16
        let bf16 = (fill.to_bits() >> 16) as u16;
        for _ in 0..(n * hidden) {
            f.write_all(&bf16.to_le_bytes()).unwrap();
        }
        path
    }

    #[test]
    fn roundtrip_load() {
        let tmp = tempfile::tempdir().unwrap();
        let _ = write_minimal_sidecar(tmp.path(), &[48, 100, 106], 64, 0.5);
        let t = CorrectionTable::load_from_model_dir(tmp.path())
            .unwrap()
            .unwrap();
        assert_eq!(t.critical_ids, vec![48, 100, 106]);
        assert_eq!(t.hidden, 64);
        assert_eq!(t.delta_f32.len(), 3 * 64);
        // bf16(0.5) rounds back to 0.5 exactly.
        for &v in &t.delta_f32 {
            assert!((v - 0.5).abs() < 1e-3, "value {v} not ~0.5");
        }
    }

    #[test]
    fn missing_sidecar_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let opt = CorrectionTable::load_from_model_dir(tmp.path()).unwrap();
        assert!(opt.is_none());
    }

    #[test]
    fn apply_recovers_correction_magnitude() {
        // Hand-built sanity case: Δ all 1.0, hidden all 0.1, hidden=10
        // → delta_k = 1.0 (10 × 0.1).
        // Pre-softcap logit at critical id = 0.0 → atanh(0) = 0 → raw_q = 0.
        // raw_corrected = 1.0 → softcap(1) = 30·tanh(1/30) ≈ 0.9989.
        let tmp = tempfile::tempdir().unwrap();
        let _ = write_minimal_sidecar(tmp.path(), &[5], 10, 1.0);
        let t = CorrectionTable::load_from_model_dir(tmp.path())
            .unwrap()
            .unwrap();
        let mut logits = vec![0.0f32; 32];
        let hidden = vec![0.1f32; 10];
        t.apply_to_logits(&mut logits, &hidden, 30.0).unwrap();
        let got = logits[5];
        let expected = 30.0_f32 * (1.0 / 30.0_f32).tanh();
        assert!(
            (got - expected).abs() < 1e-4,
            "got {got} expected {expected}"
        );
        // Other positions untouched.
        for (i, &v) in logits.iter().enumerate() {
            if i != 5 {
                assert_eq!(v, 0.0);
            }
        }
    }

    #[test]
    fn rejects_bad_magic() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join(SIDECAR_FILENAME);
        let mut f = File::create(&path).unwrap();
        f.write_all(&[0xFFu8; HEADER_LEN]).unwrap();
        let err = CorrectionTable::load(&path).err().unwrap();
        assert!(format!("{err}").contains("magic mismatch"));
    }

    #[test]
    fn rejects_hidden_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let _ = write_minimal_sidecar(tmp.path(), &[42], 8, 0.0);
        let t = CorrectionTable::load_from_model_dir(tmp.path())
            .unwrap()
            .unwrap();
        let mut logits = vec![0.0f32; 64];
        let hidden_wrong = vec![0.0f32; 16];
        let err = t
            .apply_to_logits(&mut logits, &hidden_wrong, 30.0)
            .err()
            .unwrap();
        assert!(format!("{err}").contains("hidden state width"));
    }
}
