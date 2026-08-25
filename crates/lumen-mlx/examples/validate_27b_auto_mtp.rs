//! Validation: load a 27B checkpoint with ZERO MTP env vars and confirm the MTP
//! head auto-enables from the self-contained `mtp/weights.safetensors` sidecar,
//! at the shape derived from the trunk's own `config.json`.
//!
//! Run (either 27B — the shape is read, never assumed):
//!   MODEL_ID=~/models/Qwen3.8-27B-MTPLX-Speed \
//!     cargo run --release -p lumen-mlx --features mlx-native --example validate_27b_auto_mtp
//!
//! This used to print "check stderr above for `MTP ENABLED`" and exit 0 either
//! way, which is how the id-substring dim selection stayed invisible: a 3.8 id
//! missed the `"qwen3.6" | "qwen3_5"` test and fell into the 35B-A3B **MoE**
//! branch, so the head loaded at the wrong shape or not at all and the run still
//! reported success. Every check below is now an assertion.

use anyhow::{Result, anyhow, bail};
use lumen_mlx::{MlxBackend, MtpMlpConfig, mtp_shape_from_text_config, qwen35_config};
use std::path::PathBuf;

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

    // ── 1. Derived shape, from the checkpoint's own config ──
    // Done BEFORE the load so a shape regression is reported even if the load
    // itself fails, and so the numbers printed are provably the config's and not
    // an echo of whatever the loader decided.
    let dir = PathBuf::from(shellexpand(&model_id));
    let cfg_path = dir.join("config.json");
    if !cfg_path.exists() {
        bail!(
            "MODEL_ID must be a local checkpoint directory for this validation \
             (no config.json at {}); it asserts the DERIVED dims, which needs the \
             config on disk",
            cfg_path.display()
        );
    }
    let cfg = qwen35_config::NativeModelConfig::load(&cfg_path)?;
    let (dims, mlp) = mtp_shape_from_text_config(&cfg.text_config)?;
    let tc = &cfg.text_config;
    println!(
        "derived MTP shape: hidden={} heads={}/{} head_dim={} rope_dim={} mlp={}",
        dims.hidden_size,
        dims.num_attention_heads,
        dims.num_key_value_heads,
        dims.head_dim,
        dims.rope_dim,
        match &mlp {
            MtpMlpConfig::Dense { intermediate_size } => format!("Dense({intermediate_size})"),
            MtpMlpConfig::Moe(_) => "MoE".to_string(),
        }
    );

    // The derivation must be the trunk's numbers verbatim. Asserting against
    // hardcoded 27B constants would re-introduce exactly what the fix removed.
    let mismatches: Vec<String> = [
        ("hidden_size", dims.hidden_size, tc.hidden_size),
        (
            "num_attention_heads",
            dims.num_attention_heads,
            tc.num_attention_heads,
        ),
        (
            "num_key_value_heads",
            dims.num_key_value_heads,
            tc.num_key_value_heads,
        ),
        ("head_dim", dims.head_dim, tc.head_dim),
        ("rope_dim", dims.rope_dim, tc.rope_dim()),
    ]
    .iter()
    .filter(|(_, got, want)| got != want)
    .map(|(name, got, want)| format!("{name}: derived {got} != config {want}"))
    .collect();
    if !mismatches.is_empty() {
        bail!("MTP dims do not match the trunk config: {mismatches:?}");
    }

    // A 27B checkpoint is dense. Catching this here is the point: the defect
    // this example missed was a dense 27B being given the 35B-A3B MoE shape.
    match mlp {
        MtpMlpConfig::Dense { intermediate_size } => {
            if intermediate_size != tc.intermediate_size {
                bail!(
                    "dense MTP intermediate_size {intermediate_size} != config {}",
                    tc.intermediate_size
                );
            }
        }
        MtpMlpConfig::Moe(_) => bail!(
            "derived a MoE MTP MLP (config says num_experts={:?}). Either \
             MODEL_ID is not a 27B — this validation is dense-only, point it at \
             a 27B checkpoint — or a dense 27B is being handed the 35B-A3B \
             shape, which is the id-substring defect returning",
            tc.num_experts
        ),
    }

    // ── 2. Auto-enable, asserted rather than eyeballed ──
    let backend = MlxBackend::load(&dir.to_string_lossy())?;
    let qwen = backend
        .as_qwen35()
        .ok_or_else(|| anyhow!("{model_id} did not load as a Qwen3.5-family backend"))?;
    if !qwen.qwen35_mtp_enabled() {
        bail!(
            "MTP did NOT auto-enable for {model_id}. With no MTP env vars set, a \
             checkpoint shipping mtp/weights.safetensors must come up with the \
             head installed — see `try_enable_qwen35_mtp_from_env`"
        );
    }

    println!("OK: MTP auto-enabled at the config-derived shape (no MTP env vars).");
    Ok(())
}

/// Minimal `~` expansion — `MODEL_ID` is routinely pasted with a tilde, and a
/// literal `~` directory is the confusing failure that produces.
fn shellexpand(p: &str) -> String {
    match p.strip_prefix("~/") {
        Some(rest) => format!("{}/{rest}", std::env::var("HOME").unwrap_or_default()),
        None => p.to_string(),
    }
}
