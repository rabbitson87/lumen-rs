//! Phase 2 S2 — env-gated smoke test for the Qwen3.6 MTP HF loader.
//!
//! Reads the bf16 `mtp.*` tensors from a Qwen3.6 HF original snapshot,
//! quantizes them to AFFINE 4-bit at load time, builds a `Qwen35MtpBlock`,
//! and runs one forward at T=1 with a synthetic trunk lm_head to verify
//! the full dispatch graph end-to-end with real MTP weights.
//!
//! Validates on either the 27B Dense HF original (`Qwen/Qwen3.6-27B`) or
//! the 35B-A3B HF original (`Qwen/Qwen3.6-35B-A3B`) when cached. Trunk
//! integration lands in Phase 2 S3 — this smoke only proves the loader
//! + block dispatch are correct; argmax content is meaningless without
//!   the real trunk lm_head.
//!
//! Required env:
//!   LUMEN_QWEN35_HF_ORIGINAL = snapshot directory holding the HF-original
//!                              `model.safetensors.index.json` + shards
//!                              with `mtp.*` tensors.
//!
//! Usage (27B Dense — what most users have cached):
//!   LUMEN_QWEN35_HF_ORIGINAL=~/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/<hash> \
//!     cargo run --release -p lumen-mlx --features mlx-native \
//!       --example bench_qwen35_mtp_loader_smoke -- --model 27b
//!
//! Usage (35B-A3B MoE — needs HF original separately downloaded):
//!   LUMEN_QWEN35_HF_ORIGINAL=~/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/<hash> \
//!     cargo run --release -p lumen-mlx --features mlx-native \
//!       --example bench_qwen35_mtp_loader_smoke -- --model 35b

use anyhow::{Result, anyhow};
use lumen_mlx::{
    MtpLoadQuant, MtpMlpConfig, MtpMoeConfig, Qwen35MtpDims, load_block_from_hf,
    smoke_forward_with_synth_trunk,
};
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut model = String::from("27b");
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--model" {
            model = args
                .get(i + 1)
                .cloned()
                .ok_or_else(|| anyhow!("--model needs a value (27b or 35b)"))?;
            i += 2;
        } else {
            i += 1;
        }
    }

    let hf_dir = std::env::var("LUMEN_QWEN35_HF_ORIGINAL").map_err(|_| {
        anyhow!(
            "LUMEN_QWEN35_HF_ORIGINAL must point at an HF original snapshot \
             directory (Qwen/Qwen3.6-27B or Qwen/Qwen3.6-35B-A3B)"
        )
    })?;
    let hf_path = std::path::PathBuf::from(&hf_dir);
    if !hf_path.is_dir() {
        return Err(anyhow!("{hf_dir} is not a directory"));
    }

    // Dims + MLP config sourced from each repo's config.json text_config block.
    let (dims, mlp_cfg, label) = match model.as_str() {
        "27b" => {
            let d = Qwen35MtpDims {
                // Qwen3.6-27B Dense
                hidden_size: 5120,
                num_attention_heads: 24,
                num_key_value_heads: 4,
                head_dim: 256,
                rope_theta: 10_000_000.0,
                rope_dim: 64, // partial_rotary_factor 0.25 * head_dim 256
                rms_norm_eps: 1e-6,
                attn_output_gate: true,
            };
            (
                d,
                MtpMlpConfig::Dense {
                    intermediate_size: 17_408,
                },
                "Qwen3.6-27B (Dense)",
            )
        }
        "35b" => {
            let d = Qwen35MtpDims {
                // Qwen3.6-35B-A3B MoE
                hidden_size: 2048,
                num_attention_heads: 16,
                num_key_value_heads: 2,
                head_dim: 256,
                rope_theta: 10_000_000.0,
                rope_dim: 64, // partial_rotary_factor 0.25 * head_dim 256
                rms_norm_eps: 1e-6,
                attn_output_gate: true,
            };
            let moe = MtpMoeConfig {
                num_experts: 256,
                num_experts_per_tok: 8,
                moe_intermediate_size: 512,
                shared_expert_intermediate_size: 512,
                norm_topk_prob: true,
            };
            (d, MtpMlpConfig::Moe(moe), "Qwen3.6-35B-A3B (MoE)")
        }
        other => {
            return Err(anyhow!("--model must be `27b` or `35b`, got `{other}`"));
        }
    };

    println!("--- Phase 2 S2 Qwen3.5 MTP loader smoke ({label}) ---");
    println!("hf_dir = {hf_dir}");
    println!(
        "dims  = hidden={}, heads={}/{}, head_dim={}, rope_dim={}, attn_output_gate={}",
        dims.hidden_size,
        dims.num_attention_heads,
        dims.num_key_value_heads,
        dims.head_dim,
        dims.rope_dim,
        dims.attn_output_gate,
    );
    match &mlp_cfg {
        MtpMlpConfig::Dense { intermediate_size } => {
            println!("mlp   = Dense intermediate_size={intermediate_size}");
        }
        MtpMlpConfig::Moe(m) => {
            println!(
                "mlp   = MoE E={} top_k={} moe_intermediate={} shared_intermediate={} norm_topk={}",
                m.num_experts,
                m.num_experts_per_tok,
                m.moe_intermediate_size,
                m.shared_expert_intermediate_size,
                m.norm_topk_prob,
            );
        }
    }

    // AFFINE 4-bit group_size=64 — matches Phase 2 S1.5 synth bench. For MoE
    // the eh_proj/attn/lm_head paths use this; the MoE experts internally
    // pin MXFP4 group 32 to match the trunk's middle layers.
    let quant = MtpLoadQuant::Affine4 { group_size: 64 };

    let t_load = Instant::now();
    let block = load_block_from_hf(&hf_path, dims, mlp_cfg, quant)?;
    let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;
    println!("loaded MTP block in {load_ms:.1} ms (attn/eh_proj=AFFINE4 gs=64)");

    println!();
    println!("forward smoke (synthetic trunk lm_head, T=1, vocab=248320):");
    let vocab_size: usize = 248_320;
    let t_fwd = Instant::now();
    let (argmax_tok, max_logit) = smoke_forward_with_synth_trunk(&block, vocab_size)?;
    let fwd_ms = t_fwd.elapsed().as_secs_f64() * 1000.0;
    println!(
        "  forward({load_ms:.0}+{fwd_ms:.1} ms total) -> argmax={argmax_tok} max_logit={max_logit:.4}"
    );

    if !max_logit.is_finite() {
        return Err(anyhow!(
            "FAIL: max_logit {max_logit} is not finite — block forward broken"
        ));
    }
    if argmax_tok >= vocab_size as u32 {
        return Err(anyhow!("FAIL: argmax {argmax_tok} >= vocab {vocab_size}"));
    }

    println!();
    println!("VERDICT: PASS");
    println!("  - all mtp.* tensors located + quantized + validated");
    println!("  - block.forward dispatch graph runs without error");
    println!("  - logits are finite, argmax within vocab range");
    println!();
    println!("Next (Phase 2 S3): wire `Qwen35MtpBlock::forward` against the trunk's");
    println!("real lm_head + per-seq KV cache via `runner_native::mtp_step`.");

    Ok(())
}
