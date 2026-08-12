//! Staged numeric reference comparison for the MTP block forward.
//! Loads fixed (embeds, h_pre) + per-stage references (ref_e_n .. ref_norm_out)
//! produced by /tmp/mtp_ref/ref_full.py (mlx, bf16 weights), runs lumen's
//! block.debug_block_stages on the SAME input, and prints rel-MAE per stage.
//! The FIRST stage whose rel-MAE >> quant-noise floor localizes the bug.
//!
//! Note: lumen quantizes the block linears (affine4/mxfp4) while the Python
//! ref uses bf16 — so each linear adds ~0.1 quant noise that COMPOUNDS
//! downstream. A structural bug shows as a sharp jump (e.g. 0.1 -> 0.9), not
//! gradual growth.
//!
//!   LUMEN_QWEN35_HF_ORIGINAL=~/models/Qwen3.6-35B-A3B-mtp-orig \
//!     cargo run --release -p lumen-mlx --features mlx-native --example bench_mtp_ref_compare

use anyhow::{Result, anyhow};
use lumen_mlx::{MtpLoadQuant, MtpMlpConfig, MtpMoeConfig, Qwen35MtpDims, load_block_from_hf};
use mlx_rs::Array;

fn rel_mae(a: &Array, b: &Array) -> f32 {
    let af = a.as_dtype(mlx_rs::Dtype::Float32).unwrap();
    let bf = b.as_dtype(mlx_rs::Dtype::Float32).unwrap();
    let num = mlx_rs::ops::abs(mlx_rs::ops::subtract(&af, &bf).unwrap())
        .unwrap()
        .sum(None)
        .unwrap();
    let den = mlx_rs::ops::abs(&bf).unwrap().sum(None).unwrap();
    num.eval().ok();
    den.eval().ok();
    num.as_slice::<f32>()[0] / den.as_slice::<f32>()[0].max(1e-9)
}

fn main() -> Result<()> {
    let hf_dir = std::env::var("LUMEN_QWEN35_HF_ORIGINAL")
        .map_err(|_| anyhow!("set LUMEN_QWEN35_HF_ORIGINAL"))?;
    let dims = Qwen35MtpDims {
        hidden_size: 2048,
        num_attention_heads: 16,
        num_key_value_heads: 2,
        head_dim: 256,
        rope_theta: 10_000_000.0,
        rope_dim: 64,
        rms_norm_eps: 1e-6,
        attn_output_gate: true,
    };
    let mlp_cfg = MtpMlpConfig::Moe(MtpMoeConfig {
        num_experts: 256,
        num_experts_per_tok: 8,
        moe_intermediate_size: 512,
        shared_expert_intermediate_size: 512,
        norm_topk_prob: true,
    });
    let block = load_block_from_hf(
        std::path::Path::new(&hf_dir),
        dims,
        mlp_cfg,
        MtpLoadQuant::Affine4 { group_size: 64 },
    )?;

    let io = Array::load_safetensors("/tmp/mtp_ref/io.safetensors")
        .map_err(|e| anyhow!("load io.safetensors: {e}"))?;
    let get = |k: &str| io.get(k).cloned().ok_or_else(|| anyhow!("io missing {k}"));
    let r3 = |a: Array| a.reshape(&[1, 1, 2048]).unwrap();
    let embeds = r3(get("embeds")?).as_dtype(mlx_rs::Dtype::Float32).unwrap();
    let h_pre = r3(get("h_pre")?).as_dtype(mlx_rs::Dtype::Float32).unwrap();

    let stages = block.debug_block_stages(&embeds, &h_pre)?;

    println!("=== MTP full block: lumen vs mlx Python reference (per stage) ===");
    println!("  {:<11} {:>10}", "stage", "rel-MAE");
    let mut flagged = false;
    for (name, val) in &stages {
        let refk = format!("ref_{name}");
        let r = r3(get(&refk)?);
        let rel = rel_mae(val, &r);
        let mark = if rel > 0.5 && !flagged {
            flagged = true;
            "   <<< FIRST BIG DIVERGENCE"
        } else if rel > 0.5 {
            "   <<< diverged"
        } else {
            ""
        };
        println!("  {name:<11} {rel:>10.5}{mark}");
    }
    Ok(())
}
