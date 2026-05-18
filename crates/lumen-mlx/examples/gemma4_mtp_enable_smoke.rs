//! load trunk + enable MTP via `try_enable_mtp`.
//!
//! Verifies that the drafter's `backbone_hidden_size` matches the trunk's
//! `hidden_size`, that ownership transfers cleanly, and that
//! `mtp_enabled()` flips to true. Does NOT run any drafting yet —
//! `mtp_step()` orchestration lands in subsequent Phase 3 work.
//!
//! Run:
//!   MLX_LOCAL_SOURCE_DIR=/path/to/Documents/GitHub/mlx \
//!   MODEL_ID=/path/to/models/gemma-4-26b-a4b-mlx-4bit \
//!   DRAFTER_DIR=/path/to/models/gemma-4-26B-A4B-it-assistant-bf16 \
//!   cargo run --release -p lumen-mlx \
//!       --example gemma4_mtp_enable_smoke --features mlx-native

use std::path::Path;

use anyhow::{Context, Result};

#[cfg(feature = "mlx-native")]
fn main() -> Result<()> {
    use lumen_mlx::gemma4::NativeGemma4Model;

    let model_id = std::env::var("MODEL_ID")
        .unwrap_or_else(|_| "/path/to/models/gemma-4-26b-a4b-mlx-4bit".into());
    let drafter_dir = std::env::var("DRAFTER_DIR").unwrap_or_else(|_| {
        "/path/to/models/gemma-4-26B-A4B-it-assistant-bf16".into()
    });

    eprintln!("[mtp-enable-smoke] loading trunk {model_id}");
    let mut model = NativeGemma4Model::load(Path::new(&model_id)).context("trunk load")?;
    eprintln!("[mtp-enable-smoke] trunk loaded; mtp_enabled? {}", model.mtp_enabled());
    assert!(!model.mtp_enabled(), "expected mtp_enabled=false before try_enable_mtp");

    eprintln!("[mtp-enable-smoke] try_enable_mtp({drafter_dir})");
    let ok = model
        .try_enable_mtp(Path::new(&drafter_dir))
        .context("try_enable_mtp")?;
    eprintln!("[mtp-enable-smoke] enabled = {ok}, mtp_enabled? {}", model.mtp_enabled());
    assert!(ok, "try_enable_mtp returned false");
    assert!(model.mtp_enabled(), "expected mtp_enabled=true after try_enable_mtp");

    println!("\n=== Phase 3 (scaffold) try_enable_mtp: PASS ===");
    println!("  trunk hidden_size matched drafter backbone_hidden_size");
    println!("  drafter ownership transferred");
    println!("  mtp_enabled() flips on");
    println!("\nNext: implement Step A/B/C/D/E orchestration in `mtp_step()`.");
    Ok(())
}

#[cfg(not(feature = "mlx-native"))]
fn main() -> Result<()> {
    eprintln!("This example requires --features mlx-native");
    Ok(())
}
