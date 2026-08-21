//! Phase 2 S4 — Qwen3.5 MTP vs baseline decode A/B bench.
//!
//! Loads the mlx-community Qwen3.6 MXFP4 trunk via `MlxBackend`, optionally
//! enables the Qwen3.5 MTP head from an HF-original snapshot (Phase 2 S3
//! wiring), then runs two timed passes with the SAME prompt:
//!
//!   Pass A (baseline)   — `MlxBackend::decode_step` single-token loop.
//!   Pass B (MTP)        — `MlxQwen35Backend::qwen35_mtp_step(seq, last, K)`
//!                         cycle loop.
//!
//! Reports per-pass tok/s, mean step / cycle ms, MTP accept rate, and
//! emit_mean. The bench is greedy + deterministic so the two passes always
//! generate the SAME tokens (modulo MTP's verify-only argmax which is
//! bit-identical to baseline argmax by construction — same logits, same
//! token).
//!
//! Required env:
//!   LUMEN_MLX_BACKEND=native             (MTP path is native-only)
//!   LUMEN_QWEN35_HF_ORIGINAL=...         HF-original snapshot directory
//!                                        holding `mtp.*` shards.
//!
//! Optional env:
//!   MODEL_ID          trunk model id (default mlx-community/Qwen3.6-35B-A3B-mxfp4)
//!   PROMPT_LEN        synthetic prompt length (default 32)
//!   STEPS             baseline single-step decode count (default 64)
//!   K                 MTP draft count per cycle (default 2)
//!   WARMUP            warmup steps before timing (default 4)
//!
//! Usage:
//!   LUMEN_MLX_BACKEND=native \
//!   LUMEN_QWEN35_HF_ORIGINAL=~/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/<hash> \
//!     cargo run --release -p lumen-mlx --features mlx-native \
//!       --example bench_qwen35_mtp_ab
//!
//! Interpretation:
//!   - MTP wins net iff   (MTP_tok_per_s / baseline_tok_per_s) > 1.0
//!   - For K=2 with cycle_ms ≈ baseline_step_ms × 2.5-3.5, MTP needs
//!     accept_rate ≥ ~0.5 to break even (per Phase 2 S1.5 cycle math).
//!   - Synthetic gibberish prompt drives accept rate near 0 — exercise both
//!     the structural smoke (full reject path) AND the perf characterization
//!     (cycle_ms baseline). Real prompts will lift accept_rate above gibberish.

use std::time::Instant;

use anyhow::{Result, anyhow};
use lumen_mlx::{
    MlxBackend, MtpLoadQuant, MtpMlpConfig, MtpMoeConfig, Qwen35MtpDims, load_block_from_hf,
};

fn main() -> Result<()> {
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
        .unwrap_or(32);
    let steps: usize = std::env::var("STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let k: usize = std::env::var("K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let warmup: usize = std::env::var("WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    let hf_dir = std::env::var("LUMEN_QWEN35_HF_ORIGINAL").map_err(|_| {
        anyhow!(
            "LUMEN_QWEN35_HF_ORIGINAL must point at an HF-original snapshot directory holding `mtp.*` shards"
        )
    })?;
    let hf_path = std::path::PathBuf::from(&hf_dir);
    if !hf_path.is_dir() {
        return Err(anyhow!("{hf_dir} is not a directory"));
    }

    // Detect 27B vs 35B-A3B for dims (same convention as the loader smoke
    // bench + the server-side auto-enable).
    let lower = model_id.to_lowercase();
    let is_27b_dense = (lower.contains("qwen3.6") || lower.contains("qwen3_5"))
        && !lower.contains("a3b")
        && !lower.contains("moe")
        && (lower.contains("27b") || lower.contains("-27-") || lower.contains("-dense"));
    let (dims, mlp_cfg, mtp_label) = if is_27b_dense {
        (
            Qwen35MtpDims {
                hidden_size: 5120,
                num_attention_heads: 24,
                num_key_value_heads: 4,
                head_dim: 256,
                rope_theta: 10_000_000.0,
                rope_dim: 64,
                rms_norm_eps: 1e-6,
                attn_output_gate: true,
            },
            MtpMlpConfig::Dense {
                intermediate_size: 17_408,
            },
            "Qwen3.6-27B (Dense)",
        )
    } else {
        (
            Qwen35MtpDims {
                hidden_size: 2048,
                num_attention_heads: 16,
                num_key_value_heads: 2,
                head_dim: 256,
                rope_theta: 10_000_000.0,
                rope_dim: 64,
                rms_norm_eps: 1e-6,
                attn_output_gate: true,
            },
            MtpMlpConfig::Moe(MtpMoeConfig {
                num_experts: 256,
                num_experts_per_tok: 8,
                moe_intermediate_size: 512,
                shared_expert_intermediate_size: 512,
                norm_topk_prob: true,
            }),
            "Qwen3.6-35B-A3B (MoE)",
        )
    };

    println!("=== Phase 2 S4 Qwen3.5 MTP vs baseline A/B ({mtp_label}) ===");
    println!("trunk model_id: {model_id}");
    println!("mtp hf_dir:     {hf_dir}");
    println!("prompt_len={prompt_len}  baseline_steps={steps}  K={k}  warmup={warmup}");

    let t_load = Instant::now();
    let mut backend = MlxBackend::load(&model_id)?;
    println!(
        "trunk loaded in {:.0} ms",
        t_load.elapsed().as_secs_f64() * 1000.0
    );

    let qwen = backend
        .as_qwen35_mut()
        .ok_or_else(|| anyhow!("backend is not the Qwen3.5 family"))?;

    let quant = MtpLoadQuant::Affine4 { group_size: 64 };
    let t_mtp = Instant::now();
    let block = load_block_from_hf(&hf_path, dims, mlp_cfg, quant)?;
    qwen.enable_qwen35_mtp(block)?;
    println!(
        "mtp head loaded + enabled in {:.0} ms",
        t_mtp.elapsed().as_secs_f64() * 1000.0
    );
    if !qwen.qwen35_mtp_enabled() {
        return Err(anyhow!(
            "MTP enable claimed Ok but qwen35_mtp_enabled()=false"
        ));
    }

    let prompt: Vec<u32> = (0..prompt_len).map(|i| 10 + (i as u32 * 7) % 200).collect();

    // ───────────── Pass A — baseline decode_step ─────────────
    println!();
    println!("--- Pass A (baseline single-token decode) ---");
    let seq_a: u64 = 1001;
    let t_pre_a = Instant::now();
    let (mut last_a, mut pos_a) = qwen.prefill(seq_a, &prompt)?;
    println!(
        "prefill: {} tokens, next_tok={last_a} in {:.0} ms",
        prompt.len(),
        t_pre_a.elapsed().as_secs_f64() * 1000.0
    );
    for _ in 0..warmup {
        let (n, p) = qwen.decode_step(seq_a, last_a, pos_a)?;
        last_a = n;
        pos_a = p;
    }
    let mut step_ms: Vec<f64> = Vec::with_capacity(steps);
    let mut baseline_tokens: Vec<u32> = Vec::with_capacity(steps);
    for _ in 0..steps {
        let t0 = Instant::now();
        let (n, p) = qwen.decode_step(seq_a, last_a, pos_a)?;
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        step_ms.push(dt);
        baseline_tokens.push(n);
        last_a = n;
        pos_a = p;
    }
    qwen.remove_seq(seq_a)?;

    step_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let base_total: f64 = step_ms.iter().sum();
    let base_mean = base_total / steps as f64;
    let base_p50 = step_ms[steps / 2];
    let base_p95_idx = ((steps as f64) * 0.95) as usize;
    let base_p95 = step_ms[base_p95_idx.min(steps - 1)];
    let base_tps = steps as f64 / (base_total / 1000.0);
    println!(
        "baseline: {steps} steps in {base_total:.0} ms  mean={base_mean:.2}  p50={base_p50:.2}  p95={base_p95:.2}  {base_tps:.1} tok/s"
    );

    // ───────────── Pass B — MTP cycles ─────────────
    println!();
    println!("--- Pass B (MTP qwen35_mtp_step, K={k}) ---");
    let seq_b: u64 = 1002;
    let t_pre_b = Instant::now();
    let (mut last_b, _pos_b) = qwen.prefill(seq_b, &prompt)?;
    println!(
        "prefill: {} tokens, next_tok={last_b} in {:.0} ms",
        prompt.len(),
        t_pre_b.elapsed().as_secs_f64() * 1000.0
    );

    // Warmup cycle (drops JIT-compile + cache-grow on the MTP path).
    for _ in 0..warmup {
        let w = qwen.qwen35_mtp_step(seq_b, last_b, k, &Default::default())?;
        last_b = *w.committed.last().expect("warmup: non-empty commit");
    }

    // Target same TOTAL emitted token count as baseline so the two passes
    // share their wall-clock comparison axis. baseline emitted `steps`
    // tokens; MTP emits 1 + n_accepted + 1 per cycle, so we run until we've
    // emitted at least `steps` post-warmup committed tokens.
    let target_emit = steps;
    let mut emitted: usize = 0;
    let mut cycle_ms: Vec<f64> = Vec::new();
    let mut accepted_total: usize = 0;
    let mut attempted_total: usize = 0;
    let mut mtp_tokens: Vec<u32> = Vec::new();
    while emitted < target_emit {
        let t0 = Instant::now();
        let out = qwen.qwen35_mtp_step(seq_b, last_b, k, &Default::default())?;
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        cycle_ms.push(dt);
        accepted_total += out.n_accepted;
        attempted_total += out.n_attempted;
        for t in &out.committed {
            mtp_tokens.push(*t);
            emitted += 1;
            if emitted >= target_emit {
                break;
            }
        }
        last_b = *out.committed.last().expect("non-empty commit");
    }
    qwen.remove_seq(seq_b)?;

    let cycles = cycle_ms.len();
    cycle_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mtp_total: f64 = cycle_ms.iter().sum();
    let mtp_mean = mtp_total / cycles as f64;
    let mtp_p50 = cycle_ms[cycles / 2];
    let mtp_p95_idx = ((cycles as f64) * 0.95) as usize;
    let mtp_p95 = cycle_ms[mtp_p95_idx.min(cycles - 1)];
    let mtp_tps = emitted as f64 / (mtp_total / 1000.0);
    let accept_rate = if attempted_total > 0 {
        (accepted_total as f64) / (attempted_total as f64)
    } else {
        0.0
    };
    let emit_per_cycle = emitted as f64 / cycles as f64;
    println!(
        "mtp: {cycles} cycles emit={emitted} in {mtp_total:.0} ms  mean={mtp_mean:.2}  p50={mtp_p50:.2}  p95={mtp_p95:.2}  {mtp_tps:.1} tok/s"
    );
    println!(
        "  accept_rate={accept_rate:.3} ({accepted_total}/{attempted_total})  emit/cycle={emit_per_cycle:.2} (max={})",
        k + 1
    );

    // ───────────── Summary ─────────────
    let speedup = mtp_tps / base_tps;
    let cycle_ratio = mtp_mean / base_mean;
    println!();
    println!("=== Summary ===");
    println!("baseline:  {base_tps:>6.1} tok/s  (mean step {base_mean:.2} ms)");
    println!(
        "mtp K={k}:   {mtp_tps:>6.1} tok/s  (mean cycle {mtp_mean:.2} ms; ratio vs baseline {cycle_ratio:.2}x)"
    );
    println!("speedup: {speedup:.3}x  (>= 1.0 = MTP wins; current accept_rate {accept_rate:.3})");

    // NOTE: baseline_tokens vs mtp_tokens cannot be directly compared
    // position-by-position because each pass uses its own warmup loop that
    // advances the seq's cache by a DIFFERENT number of tokens (baseline:
    // `warmup` single-step decodes; MTP: `warmup` cycles × 2 emits at 0%
    // accept). The token sequences are at different absolute positions in
    // the generated trajectory when timing starts. To compare token content
    // bit-identically, run the two passes with synchronized warmups OR
    // compare post-fact via decoded text similarity — out of scope for the
    // perf A/B harness.

    Ok(())
}
