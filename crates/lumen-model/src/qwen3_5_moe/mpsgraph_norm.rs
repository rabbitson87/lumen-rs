//! Process-wide MPSGraph RmsNorm singleton.
//!
//! Gated behind `LUMEN_MPSGRAPH=1`. The singleton owns one
//! `MpsGraphContext` and a JIT cache of compiled executables keyed by
//! `(m, hidden)`. Used by both `model::Qwen3_5MoeTextModel::forward_with_offset`
//! (final norm) and `layer::DecoderLayer::forward_with_tq`
//! (input/post-attention norms) so all 81 RmsNorm callsites share one cache.

use std::sync::OnceLock;
use lumen_metal::mpsgraph::MpsRmsNorm;

/// Qwen3.5-MoE config has all RmsNorm ops at eps=1e-6 (verified via
/// `qwen3_5_moe::config`). If a future checkpoint uses a different eps the
/// bit-identical token check will catch the drift.
const RMS_NORM_EPS: f32 = 1e-6;

static MPS_RMS_NORM: OnceLock<Option<MpsRmsNorm>> = OnceLock::new();

/// Borrow the process-wide MpsRmsNorm if `LUMEN_MPSGRAPH=1` is set and
/// the runtime initialised successfully. Returns `None` otherwise (callers
/// fall back to candle's own `RmsNorm::forward`).
pub(crate) fn get() -> Option<&'static MpsRmsNorm> {
    MPS_RMS_NORM
        .get_or_init(|| {
            if std::env::var("LUMEN_MPSGRAPH").ok().as_deref() == Some("1") {
                match MpsRmsNorm::new(RMS_NORM_EPS) {
                    Ok(m) => Some(m),
                    Err(e) => {
                        eprintln!("LUMEN_MPSGRAPH=1 but MpsRmsNorm init failed: {e}");
                        None
                    }
                }
            } else {
                None
            }
        })
        .as_ref()
}
