//! Validation: load Qwen3.6-27B with ZERO MTP env vars and confirm the MTP head
//! auto-enables from the self-contained `mtp/weights.safetensors` sidecar.
//!
//! Run: MODEL_ID=~/models/Qwen3.6-27B-MTPLX-Speed \
//!   cargo run --release -p lumen-mlx --features mlx-native --example validate_27b_auto_mtp

use anyhow::Result;
use lumen_mlx::MlxBackend;

fn main() -> Result<()> {
    unsafe {
        std::env::set_var("LUMEN_MLX_BACKEND", "native");
        // Explicitly ensure NONE of the manual MTP env vars are set — this proves
        // the auto-enable path.
        std::env::remove_var("LUMEN_QWEN35_MTP");
        std::env::remove_var("LUMEN_QWEN35_HF_ORIGINAL");
        std::env::remove_var("LUMEN_SPEC");
    }
    let home = std::env::var("HOME").unwrap_or_default();
    let model_id = std::env::var("MODEL_ID")
        .unwrap_or_else(|_| format!("{home}/models/Qwen3.6-27B-MTPLX-Speed"));
    println!("loading (no MTP env vars): {model_id}");
    let _backend = MlxBackend::load(&model_id)?;
    // Success criterion: the loader emits `[mlx] qwen3.5 MTP ENABLED ...` to
    // stderr from the auto-enable path. Grep the run output for that line.
    println!("loaded — check stderr above for `[mlx] qwen3.5 MTP ENABLED` (auto-enable).");
    Ok(())
}
