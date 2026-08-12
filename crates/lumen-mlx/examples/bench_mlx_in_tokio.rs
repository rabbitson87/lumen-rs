//! Same as `bench_mlx_e2e` but runs inside a tokio runtime — to test the
//! hypothesis that tokio context causes the MLX subprocess pipe I/O to be
//! 4-10× slower than in plain sync (e2e).

use std::time::Instant;

use anyhow::{Result, anyhow};
use lumen_mlx::MlxBackend;

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<()> {
    // Move the backend ownership into a spawn_blocking so the whole bench
    // runs on a dedicated blocking-pool thread, isolated from tokio's async
    // workers and from any QoS / scheduling interference they might cause.
    tokio::task::spawn_blocking(run_sync).await??;
    Ok(())
}

fn run_sync() -> Result<()> {
    let model_id =
        std::env::var("MODEL_ID").unwrap_or_else(|_| "mlx-community/Qwen3.6-35B-A3B-mxfp4".into());
    let prompt_len: usize = std::env::var("PROMPT_LEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let steps: usize = std::env::var("STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let warmup: usize = std::env::var("WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    // Load on the main tokio task (synchronously — same as the engine does).
    let mut b = MlxBackend::load(&model_id)?;
    let b = b
        .as_qwen35_mut()
        .ok_or_else(|| anyhow!("bench requires Qwen35-family backend"))?;

    let prompt: Vec<u32> = (0..prompt_len).map(|i| 10 + (i as u32 * 7) % 200).collect();

    let t_prefill = Instant::now();
    let (mut last, mut pos) = b.prefill(1, &prompt)?;
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
    println!("prefill: {} tokens in {prefill_ms:.0}ms", prompt.len());

    for _ in 0..warmup {
        let (n, p) = b.decode_step(1, last, pos)?;
        last = n;
        pos = p;
    }

    let mut step_ms: Vec<f64> = Vec::with_capacity(steps);
    for _ in 0..steps {
        let t0 = Instant::now();
        let (n, p) = b.decode_step(1, last, pos)?;
        last = n;
        pos = p;
        step_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    b.remove_seq(1)?;

    step_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let total: f64 = step_ms.iter().sum();
    let mean = total / steps as f64;
    let p50 = step_ms[steps / 2];
    let p95 = step_ms[(steps as f64 * 0.95) as usize];
    let tps = steps as f64 / (total / 1000.0);

    println!("decode (in tokio): {steps} steps in {total:.0}ms");
    println!("  step latency: mean={mean:.2}ms p50={p50:.2}ms p95={p95:.2}ms");
    println!("  throughput:   {tps:.1} tok/s");
    Ok(())
}
