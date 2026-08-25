//! Phase 2 S3 — end-to-end MTP decode smoke against a real Qwen3.6 trunk.
//!
//! Loads the mlx-community MXFP4 trunk via the standard `MlxBackend` path,
//! loads the matching MTP head from an HF-original snapshot (bf16 -> AFFINE4
//! at load time), installs the drafter onto the trunk via `enable_qwen35_mtp`,
//! prefills a short synthetic prompt, and then alternates between baseline
//! `decode_step` (single token) and `qwen35_mtp_step` (1 + n_accepted + 1
//! tokens) to validate the orchestration end-to-end against production weights.
//!
//! Required env:
//!   LUMEN_MLX_BACKEND=native         (forced — MTP path is native-only)
//!   LUMEN_QWEN35_HF_ORIGINAL=...     HF-original snapshot directory with the
//!                                    `mtp.*` shards (Qwen/Qwen3.6-27B or
//!                                    Qwen/Qwen3.6-35B-A3B).
//!
//! Optional env:
//!   MODEL_ID    trunk model id (default mlx-community/Qwen3.6-35B-A3B-mxfp4)
//!   PROMPT_LEN  synthetic prompt length (default 8)
//!   STEPS       MTP cycles to run (default 32)
//!   K           draft count per cycle (default 2)
//!   MTP_MODEL   27b | 35b — dims preset (default 35b)
//!
//! Usage (35B-A3B MoE):
//!   LUMEN_MLX_BACKEND=native \
//!   LUMEN_QWEN35_HF_ORIGINAL=~/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/<hash> \
//!     cargo run --release -p lumen-mlx --features mlx-native \
//!       --example bench_qwen35_mtp_decode_smoke
//!
//! The smoke verifies:
//!   * `enable_qwen35_mtp` accepts the loaded block
//!   * `qwen35_mtp_step` advances trunk cache + position correctly across
//!     accept / reject paths
//!   * Per-cycle latency and accept rate land in the same ballpark as the
//!     Phase 1 S-scaling + S1.5 Step B synth estimates (sanity envelope, not
//!     a perf gate — Phase 2 S4 owns the production A/B numbers).

use std::time::Instant;

use anyhow::{Result, anyhow};
use lumen_mlx::{
    MlxBackend, MtpLoadQuant, MtpMlpConfig, MtpMoeConfig, Qwen35MtpDims, load_block_from_hf,
};

fn main() -> Result<()> {
    // The MTP path is native-only; force the backend selection up front so
    // the user gets a clear error rather than a runtime mismatch later.
    if std::env::var("LUMEN_MLX_BACKEND")
        .map(|v| v.trim().to_ascii_lowercase())
        .unwrap_or_default()
        != "native"
    {
        unsafe {
            std::env::set_var("LUMEN_MLX_BACKEND", "native");
        }
        eprintln!("[bench] forced LUMEN_MLX_BACKEND=native (MTP requires native runner)");
    }

    let model_id =
        std::env::var("MODEL_ID").unwrap_or_else(|_| "mlx-community/Qwen3.6-35B-A3B-mxfp4".into());
    let prompt_len: usize = std::env::var("PROMPT_LEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let steps: usize = std::env::var("STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let k: usize = std::env::var("K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let mtp_model = std::env::var("MTP_MODEL").unwrap_or_else(|_| "35b".into());

    let hf_dir = std::env::var("LUMEN_QWEN35_HF_ORIGINAL").map_err(|_| {
        anyhow!(
            "LUMEN_QWEN35_HF_ORIGINAL must point at an HF-original snapshot directory \
             with `mtp.*` shards (Qwen/Qwen3.6-27B or Qwen/Qwen3.6-35B-A3B)"
        )
    })?;
    let hf_path = std::path::PathBuf::from(&hf_dir);
    if !hf_path.is_dir() {
        return Err(anyhow!("{hf_dir} is not a directory"));
    }

    let (dims, mlp_cfg, label) = match mtp_model.as_str() {
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
            return Err(anyhow!("MTP_MODEL must be `27b` or `35b`, got `{other}`"));
        }
    };

    println!("=== Phase 2 S3 Qwen3.5 MTP decode smoke ({label}) ===");
    println!("trunk model_id: {model_id}");
    println!("mtp hf_dir:     {hf_dir}");
    println!("prompt_len={prompt_len}  steps={steps}  K={k}");

    // 1) Load trunk (mlx-community MXFP4 — no `mtp.*` here).
    let t_load = Instant::now();
    let mut backend = MlxBackend::load(&model_id)?;
    println!(
        "trunk loaded in {:.0} ms",
        t_load.elapsed().as_secs_f64() * 1000.0
    );

    let qwen = backend
        .as_qwen35_mut()
        .ok_or_else(|| anyhow!("backend is not the Qwen3.5 family — MTP requires Qwen3.5/3.6"))?;

    // 2) Load MTP block from the HF original (bf16) and quantize to AFFINE4
    //    at load. Matches Phase 2 S2.5 loader smoke convention.
    let quant = MtpLoadQuant::Affine4 { group_size: 64 };
    let t_mtp = Instant::now();
    let block = load_block_from_hf(&hf_path, dims, mlp_cfg, quant)?;
    println!(
        "mtp head loaded + quantized in {:.0} ms (AFFINE4 gs=64)",
        t_mtp.elapsed().as_secs_f64() * 1000.0
    );

    // 3) Install drafter onto the trunk.
    qwen.enable_qwen35_mtp(block)?;
    if !qwen.qwen35_mtp_enabled() {
        return Err(anyhow!(
            "enable_qwen35_mtp returned Ok but qwen35_mtp_enabled() is false"
        ));
    }
    println!("MTP enabled on trunk.");

    // 4) Prefill a synthetic prompt.
    let prompt: Vec<u32> = (0..prompt_len).map(|i| 10 + (i as u32 * 7) % 200).collect();
    let t_pre = Instant::now();
    let (mut last, mut pos) = qwen.prefill(1, &prompt)?;
    println!(
        "prefill: {} tokens, next_tok={last} pos={pos} in {:.0} ms",
        prompt.len(),
        t_pre.elapsed().as_secs_f64() * 1000.0
    );

    // 5) Warmup — one MTP cycle (drops JIT-compile + cache-grow costs).
    let warm = qwen.qwen35_mtp_step(1, last, k, &Default::default())?;
    println!(
        "warmup cycle: emitted {} tokens (n_acc={}/{}); last_tok={}",
        warm.committed.len(),
        warm.n_accepted,
        warm.n_attempted,
        warm.committed.last().copied().unwrap_or(0)
    );
    last = *warm
        .committed
        .last()
        .ok_or_else(|| anyhow!("warmup: empty committed list"))?;
    pos += warm.committed.len();

    // 6) Timed MTP loop.
    let mut cycle_ms: Vec<f64> = Vec::with_capacity(steps);
    let mut accepted_total: usize = 0;
    let mut attempted_total: usize = 0;
    let mut emitted_total: usize = 0;
    for s in 0..steps {
        let t0 = Instant::now();
        let out = qwen.qwen35_mtp_step(1, last, k, &Default::default())?;
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        cycle_ms.push(dt);
        accepted_total += out.n_accepted;
        attempted_total += out.n_attempted;
        emitted_total += out.committed.len();
        last = *out
            .committed
            .last()
            .ok_or_else(|| anyhow!("cycle {s}: empty committed list"))?;
        pos += out.committed.len();
        if std::env::var("LUMEN_DUMP_CYCLE_MS")
            .map(|v| v != "0")
            .unwrap_or(false)
        {
            println!(
                "[cycle {s}] {dt:.2} ms  emitted={} n_acc={}/{}",
                out.committed.len(),
                out.n_accepted,
                out.n_attempted,
            );
        }
    }

    qwen.remove_seq(1)?;

    // 7) Aggregate report.
    cycle_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let total: f64 = cycle_ms.iter().sum();
    let mean = total / steps as f64;
    let p50 = cycle_ms[steps / 2];
    let p95_idx = ((steps as f64) * 0.95) as usize;
    let p95 = cycle_ms[p95_idx.min(steps - 1)];
    let p99_idx = ((steps as f64) * 0.99) as usize;
    let p99 = cycle_ms[p99_idx.min(steps - 1)];
    let accept_rate = (accepted_total as f64) / (attempted_total as f64);
    let emit_mean = (emitted_total as f64) / (steps as f64);
    let tok_per_s = (emitted_total as f64) / (total / 1000.0);

    println!();
    println!("--- MTP decode report (steps={steps}, K={k}) ---");
    println!(
        "cycle latency:  mean={mean:.2} ms  p50={p50:.2} ms  p95={p95:.2} ms  p99={p99:.2} ms"
    );
    println!("accept rate:    {accept_rate:.3}  ({accepted_total}/{attempted_total})");
    println!("emit per cycle: {emit_mean:.3}  (max possible = {})", k + 1);
    println!("throughput:     {tok_per_s:.1} tok/s");
    println!("final pos:      {pos}");

    // 8) Sanity gates — early exit if structurally broken.
    if !mean.is_finite() {
        return Err(anyhow!("mean cycle latency is not finite"));
    }
    if !(0.0..=1.0).contains(&accept_rate) {
        return Err(anyhow!("accept_rate {accept_rate} out of [0,1]"));
    }
    if emit_mean < 1.0 || emit_mean > (k + 1) as f64 {
        return Err(anyhow!("emit_mean {emit_mean} out of [1, {}]", k + 1));
    }

    println!();
    println!("VERDICT: PASS (smoke contract upheld; no perf judgment — see Phase 2 S4 for A/B)");
    Ok(())
}
