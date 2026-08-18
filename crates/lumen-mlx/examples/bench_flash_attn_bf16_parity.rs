//! flash_attn_bf16 parity vs mlx::fast::sdpa.
//!
//! Decode-shape sweep (Sq=1, no mask, causal — matches the call site at
//! `gemma4_moe.rs:2068-2069` `sdpa(...)` path for full-attention layers).
//!
//! Threshold: bf16 with f32-internal softmax accumulation. Allow
//! max|Δ| ≤ 3e-2 and per-row cosine ≥ 0.9999.
//!
//! Run:
//!   MLX_LOCAL_SOURCE_DIR=/path/to/mlx \
//!     cargo run --release --features mlx-native --example bench_flash_attn_bf16_parity

#[cfg(feature = "mlx-native")]
fn main() -> anyhow::Result<()> {
    use lumen_mlx::native_metal_bridge::run_flash_attn_bf16;
    use mlx_rs::{Array, Dtype};

    println!("=== Phase 1.8 M4.2: flash_attn_bf16 parity vs mlx::fast::sdpa ===");

    // Gemma 4: n_heads=32 (Q), n_kv=8 (K/V), head_dim=256, GQA group=4.
    let n_heads: usize = 32;
    let n_kv: usize = 8;
    let d: usize = 256;
    let scale: f32 = 1.0 / (d as f32).sqrt();

    // Sweep KV lengths covering decode @ various offsets.
    let kv_cases: &[usize] = &[128, 1024, 2048, 4096];

    let mut max_diff_overall = 0.0f32;
    let mut min_cos_overall = 1.0f32;

    for &skv in kv_cases {
        // Deterministic, representative-magnitude inputs.
        let mk = |n: usize, seed: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let t = (i as f32) * 0.011 + seed;
                    t.sin() * 0.4 + (t * 1.7).cos() * 0.25
                })
                .collect()
        };
        let q_data = mk(n_heads * d, 0.0);
        let k_data = mk(n_kv * skv * d, 1.0);
        let v_data = mk(n_kv * skv * d, 2.0);

        let q_f32 = Array::from_slice(&q_data, &[1, n_heads as i32, 1, d as i32]);
        let k_f32 = Array::from_slice(&k_data, &[1, n_kv as i32, skv as i32, d as i32]);
        let v_f32 = Array::from_slice(&v_data, &[1, n_kv as i32, skv as i32, d as i32]);

        let q = q_f32.as_dtype(Dtype::Bfloat16)?;
        let k = k_f32.as_dtype(Dtype::Bfloat16)?;
        let v = v_f32.as_dtype(Dtype::Bfloat16)?;
        q.eval()?;
        k.eval()?;
        v.eval()?;

        // Reference: mlx::fast::scaled_dot_product_attention with no mask.
        // At Sq=1 decode every query attends to all KV (causal == no-mask).
        let ref_out = mlx_rs::fast::scaled_dot_product_attention(
            &q, &k, &v, scale, None, // no mask
            None, // no sinks
        )?;
        ref_out.eval()?;

        let got = run_flash_attn_bf16(&q, &k, &v, scale, None)?;
        got.eval()?;

        // Reference output shape: [1, n_heads, 1, d]. mlx may keep it as a
        // view → force materialization via dtype-cast to f32 (same shape).
        let ref_f32 = ref_out.as_dtype(Dtype::Float32)?;
        let got_f32 = got.as_dtype(Dtype::Float32)?;
        ref_f32.eval()?;
        got_f32.eval()?;

        let ref_vec: Vec<f32> = ref_f32.as_slice::<f32>().to_vec();
        let got_vec: Vec<f32> = got_f32.as_slice::<f32>().to_vec();
        assert_eq!(ref_vec.len(), got_vec.len());

        let max_diff = ref_vec
            .iter()
            .zip(&got_vec)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);

        // Per-head cosine (each head is one D-vector at Sq=1).
        let mut min_cos = 1.0f32;
        for h in 0..n_heads {
            let a = &ref_vec[h * d..(h + 1) * d];
            let b = &got_vec[h * d..(h + 1) * d];
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            let cos = if na > 0.0 && nb > 0.0 {
                (dot / (na * nb)).clamp(-1.0, 1.0)
            } else {
                1.0
            };
            min_cos = min_cos.min(cos);
        }

        max_diff_overall = max_diff_overall.max(max_diff);
        min_cos_overall = min_cos_overall.min(min_cos);

        let status = if max_diff <= 3.0e-2 && min_cos >= 0.9999 {
            "OK"
        } else {
            "FAIL"
        };
        println!(
            "Sq=1 H={n_heads} H_kv={n_kv} D={d} Skv={skv:>4}: max|Δ|={max_diff:.3e} min_cos={min_cos:.6} {status}"
        );
    }

    println!();
    println!("=== overall: max|Δ|={max_diff_overall:.3e} min_cos={min_cos_overall:.6} ===");
    if max_diff_overall > 3.0e-2 || min_cos_overall < 0.9999 {
        eprintln!("PARITY FAIL");
        std::process::exit(1);
    }
    println!("PARITY OK — flash_attn_bf16 matches mlx::fast::sdpa within bf16 epsilon");

    // ── M4.3: sliding-window additive mask ────────────────────────────────
    println!();
    println!("=== M4.3: flash_attn_bf16 with sliding-window additive mask ===");

    let window: usize = 1024;
    let mut m43_max = 0.0f32;
    let mut m43_cos = 1.0f32;

    for &skv in kv_cases {
        // Reuse the same Q/K/V from above. Re-create here for clarity.
        let mk = |n: usize, seed: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let t = (i as f32) * 0.011 + seed;
                    t.sin() * 0.4 + (t * 1.7).cos() * 0.25
                })
                .collect()
        };
        let q_data = mk(n_heads * d, 0.0);
        let k_data = mk(n_kv * skv * d, 1.0);
        let v_data = mk(n_kv * skv * d, 2.0);

        let q = Array::from_slice(&q_data, &[1, n_heads as i32, 1, d as i32])
            .as_dtype(Dtype::Bfloat16)?;
        let k = Array::from_slice(&k_data, &[1, n_kv as i32, skv as i32, d as i32])
            .as_dtype(Dtype::Bfloat16)?;
        let v = Array::from_slice(&v_data, &[1, n_kv as i32, skv as i32, d as i32])
            .as_dtype(Dtype::Bfloat16)?;
        q.eval()?;
        k.eval()?;
        v.eval()?;

        // Build sliding mask: query at position skv-1 attends to keys in
        // [skv-window, skv-1]; earlier keys get -inf.
        // mask shape: [Sq=1, Skv]. Additive in logits space.
        let qpos = skv - 1; // last position at decode
        let start_allowed = qpos.saturating_sub(window - 1);
        let mut mask_f32 = vec![0.0f32; skv];
        for j in 0..skv {
            if j < start_allowed {
                mask_f32[j] = f32::NEG_INFINITY;
            }
        }
        let mask = Array::from_slice(&mask_f32, &[1, skv as i32]).as_dtype(Dtype::Bfloat16)?;
        mask.eval()?;

        let ref_out = mlx_rs::fast::scaled_dot_product_attention(
            &q,
            &k,
            &v,
            scale,
            Some(mlx_rs::fast::ScaledDotProductAttentionMask::Array(&mask)),
            None,
        )?;
        ref_out.eval()?;

        let got = run_flash_attn_bf16(&q, &k, &v, scale, Some(&mask))?;
        got.eval()?;

        let ref_f32 = ref_out.as_dtype(Dtype::Float32)?;
        let got_f32 = got.as_dtype(Dtype::Float32)?;
        ref_f32.eval()?;
        got_f32.eval()?;

        let r: Vec<f32> = ref_f32.as_slice::<f32>().to_vec();
        let g: Vec<f32> = got_f32.as_slice::<f32>().to_vec();
        let max_diff = r
            .iter()
            .zip(&g)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let mut min_cos = 1.0f32;
        for h in 0..n_heads {
            let a = &r[h * d..(h + 1) * d];
            let b = &g[h * d..(h + 1) * d];
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            let cos = if na > 0.0 && nb > 0.0 {
                (dot / (na * nb)).clamp(-1.0, 1.0)
            } else {
                1.0
            };
            min_cos = min_cos.min(cos);
        }
        m43_max = m43_max.max(max_diff);
        m43_cos = m43_cos.min(min_cos);
        let status = if max_diff <= 3.0e-2 && min_cos >= 0.9999 {
            "OK"
        } else {
            "FAIL"
        };
        println!(
            "Sq=1 H={n_heads} D={d} Skv={skv:>4} window={window}: max|Δ|={max_diff:.3e} min_cos={min_cos:.6} {status}"
        );
    }

    println!();
    println!("=== sliding overall: max|Δ|={m43_max:.3e} min_cos={m43_cos:.6} ===");
    if m43_max > 3.0e-2 || m43_cos < 0.9999 {
        eprintln!("SLIDING PARITY FAIL");
        std::process::exit(1);
    }
    println!("M4.3 PARITY OK — sliding-window mask matches");
    Ok(())
}

#[cfg(not(feature = "mlx-native"))]
fn main() {
    eprintln!("requires --features mlx-native");
    std::process::exit(2);
}
