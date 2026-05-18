//! isolate raw mlx::fast::sdpa perf at full-attention shapes.
//!
//! Gemma 4 26B-A4B full-attention layer at PROMPT_LEN=4096 decode:
//!   q : [1, 16, 1, 512] bf16
//!   k : [1, 2, 4096, 512] bf16
//!   v : [1, 2, 4096, 512] bf16
//!   mask=None (decode), GQA ratio 8, head_dim=512
//!
//! E2E A/B showed lumen-rs +25 ms/step vs mlx-lm; PROMPT_LEN sweep localizes
//! +17.7 ms to the 5 full-attn layers between KV=1024 → 4096. This microbench
//! isolates whether the regression is in raw mlx::fast::sdpa or somewhere else
//! in our wrapping (norms, RoPE, KVCache slice view, etc.).
//!
//! Run:
//!   MLX_LOCAL_SOURCE_DIR=/path/to/mlx \
//!     cargo run --release --features mlx-native --no-default-features \
//!       --example bench_full_attn_microbench

#[cfg(not(feature = "mlx-native"))]
fn main() {
    eprintln!("rebuild with --features mlx-native");
}

#[cfg(feature = "mlx-native")]
fn main() -> anyhow::Result<()> {
    use mlx_rs::{Array, Dtype};
    use std::time::Instant;

    println!("=== Phase 1.8 post-M4.8: full-attention SDPA microbench ===");

    // Gemma 4 26B-A4B full attention shapes.
    let n_heads: i32 = 16;
    let n_kv: i32 = 2;
    let d: i32 = 512;
    let scale: f32 = 1.0;

    let kv_cases: &[i32] = &[128, 512, 1024, 2048, 4096];
    let warmup = 5;
    let iters = 50;

    // Helper to generate deterministic bf16 data shaped (n,).
    let make_bf16 = |n: usize, seed: f32| -> anyhow::Result<Vec<f32>> {
        Ok((0..n)
            .map(|i| {
                let t = (i as f32) * 0.011 + seed;
                t.sin() * 0.4 + (t * 1.7).cos() * 0.25
            })
            .collect())
    };

    println!(
        "shape: q=[1,{n_heads},1,{d}] k=v=[1,{n_kv},KV,{d}] bf16, GQA={}",
        n_heads / n_kv
    );
    println!("warmup={warmup} iters={iters}");
    println!();
    println!("{:>6} | {:>14} | {:>14}", "KV", "mean us/call", "throughput");

    for &skv in kv_cases {
        let q_data = make_bf16((n_heads * 1 * d) as usize, 0.0)?;
        let k_data = make_bf16((n_kv * skv * d) as usize, 1.0)?;
        let v_data = make_bf16((n_kv * skv * d) as usize, 2.0)?;

        let q_f32 = Array::from_slice(&q_data, &[1, n_heads, 1, d]);
        let k_f32 = Array::from_slice(&k_data, &[1, n_kv, skv, d]);
        let v_f32 = Array::from_slice(&v_data, &[1, n_kv, skv, d]);

        let q = q_f32.as_dtype(Dtype::Bfloat16)?;
        let k = k_f32.as_dtype(Dtype::Bfloat16)?;
        let v = v_f32.as_dtype(Dtype::Bfloat16)?;
        // Materialize inputs once.
        q.eval()?;
        k.eval()?;
        v.eval()?;

        // Warmup.
        for _ in 0..warmup {
            let out = mlx_rs::fast::scaled_dot_product_attention(
                &q, &k, &v, scale, None, None,
            )?;
            out.eval()?;
        }

        // Timed.
        let t0 = Instant::now();
        for _ in 0..iters {
            let out = mlx_rs::fast::scaled_dot_product_attention(
                &q, &k, &v, scale, None, None,
            )?;
            out.eval()?;
        }
        let total_us = t0.elapsed().as_micros() as f64;
        let mean_us = total_us / iters as f64;
        let calls_per_sec = 1e6 / mean_us;
        println!(
            "{:>6} | {:>14.2} | {:>10.0} c/s",
            skv, mean_us, calls_per_sec
        );
    }

    Ok(())
}
