//! bit/epsilon parity for fused RMSNorm bf16 kernel.
//!
//! Compares `lumen_mlx::native_metal_bridge::run_rms_norm_bf16`
//! against `mlx_rs::fast::rms_norm` on the shapes Gemma 4 attention uses:
//!
//!   - Decode:  `[1, 1,  32, 256]`  (Q at single-token step)
//!   - Decode:  `[1, 1,   8, 256]`  (K/V at single-token step, GQA n_kv=8)
//!   - Prefill: `[1, 64, 32, 256]`  (Q at 64-token chunk)
//!   - Prefill: `[1, 4096, 32, 256]` (Q at full 4K prompt)
//!
//! Threshold: bf16 mantissa is 7 bits → ULP ≈ ~1e-2 of value magnitude.
//! Allow `max|Δ| ≤ 3e-2` and `cosine ≥ 0.9999` per row.
//!
//! Run:
//!   MLX_LOCAL_SOURCE_DIR=/path/to/mlx \
//!     cargo run --release --features mlx-native --example bench_rms_norm_bf16_parity

#[cfg(feature = "mlx-native")]
fn main() -> anyhow::Result<()> {
    use mlx_rs::{Array, Dtype};
    use lumen_mlx::native_metal_bridge::run_rms_norm_bf16;

    println!("=== Phase 1.8 M2.3: rms_norm_bf16 parity vs mlx::fast::rms_norm ===");

    let eps: f32 = 1e-6;

    let cases: &[(&str, &[i32])] = &[
        ("decode-Q   [1, 1, 32, 256]", &[1, 1, 32, 256]),
        ("decode-KV  [1, 1,  8, 256]", &[1, 1, 8, 256]),
        ("prefill-64 [1, 64, 32, 256]", &[1, 64, 32, 256]),
        ("prefill-4K [1, 4096, 32, 256]", &[1, 4096, 32, 256]),
    ];

    let mut overall_max_diff = 0.0f32;
    let mut overall_min_cos = 1.0f32;

    for (label, shape) in cases {
        let n: usize = shape.iter().product::<i32>() as usize;
        let d: usize = *shape.last().unwrap() as usize;

        // Reproducible random input + weight. Values near unit-norm to be
        // representative of post-projection activations.
        let input_data: Vec<f32> = (0..n)
            .map(|i| {
                let t = (i as f32) * 0.013;
                t.sin() * 0.5 + (t * 1.7).cos() * 0.3
            })
            .collect();
        let weight_data: Vec<f32> = (0..d).map(|i| 1.0 + 0.1 * ((i as f32) * 0.05).sin()).collect();

        let input_f32 = Array::from_slice(&input_data, shape);
        let weight_f32 = Array::from_slice(&weight_data, &[d as i32]);
        let input_bf16 = input_f32.as_dtype(Dtype::Bfloat16)?;
        let weight_bf16 = weight_f32.as_dtype(Dtype::Bfloat16)?;
        input_bf16.eval()?;
        weight_bf16.eval()?;

        // Reference path via mlx_rs::fast::rms_norm.
        let ref_out = mlx_rs::fast::rms_norm(&input_bf16, &weight_bf16, eps)?;
        ref_out.eval()?;

        // Our path.
        let got = run_rms_norm_bf16(&input_bf16, &weight_bf16, eps)?;
        got.eval()?;

        // Compare in f32.
        let ref_f32 = ref_out.as_dtype(Dtype::Float32)?;
        let got_f32 = got.as_dtype(Dtype::Float32)?;
        ref_f32.eval()?;
        got_f32.eval()?;

        let ref_vec: Vec<f32> = ref_f32.as_slice::<f32>().to_vec();
        let got_vec: Vec<f32> = got_f32.as_slice::<f32>().to_vec();
        assert_eq!(ref_vec.len(), got_vec.len(), "len mismatch for {label}");

        // Max abs diff overall.
        let max_diff = ref_vec
            .iter()
            .zip(&got_vec)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);

        // Per-row cosine (row = innermost D-vector).
        let n_rows = n / d;
        let mut min_cos = 1.0f32;
        for r in 0..n_rows {
            let r_a = &ref_vec[r * d..(r + 1) * d];
            let r_b = &got_vec[r * d..(r + 1) * d];
            let dot: f32 = r_a.iter().zip(r_b).map(|(x, y)| x * y).sum();
            let na: f32 = r_a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = r_b.iter().map(|x| x * x).sum::<f32>().sqrt();
            let cos = if na > 0.0 && nb > 0.0 {
                (dot / (na * nb)).clamp(-1.0, 1.0)
            } else {
                1.0
            };
            min_cos = min_cos.min(cos);
        }

        overall_max_diff = overall_max_diff.max(max_diff);
        overall_min_cos = overall_min_cos.min(min_cos);

        println!(
            "{label}: max|Δ|={max_diff:.3e}  min_cos={min_cos:.6}  ({})",
            if max_diff <= 3.0e-2 && min_cos >= 0.9999 {
                "OK"
            } else {
                "FAIL"
            }
        );
    }

    println!();
    println!(
        "=== overall: max|Δ|={overall_max_diff:.3e}  min_cos={overall_min_cos:.6} ===",
    );

    if overall_max_diff > 3.0e-2 || overall_min_cos < 0.9999 {
        eprintln!("PARITY FAIL — see per-shape numbers above");
        std::process::exit(1);
    }
    println!("PARITY OK — fused rms_norm_bf16 matches mlx::fast::rms_norm within bf16 epsilon");

    // ── Fused rms_norm + transpose [B,L,H,D] → [B,H,L,D] ────────────────
    use lumen_mlx::native_metal_bridge::run_rms_norm_transpose_bf16;
    println!();
    println!("=== M2.4: rms_norm_transpose_bf16 vs (rms_norm + transpose_axes) ===");

    let mut tr_max_diff = 0.0f32;
    let mut tr_min_cos = 1.0f32;

    for (label, shape) in cases {
        let n: usize = shape.iter().product::<i32>() as usize;
        let b = shape[0] as usize;
        let l = shape[1] as usize;
        let h = shape[2] as usize;
        let d = shape[3] as usize;

        let input_data: Vec<f32> = (0..n)
            .map(|i| {
                let t = (i as f32) * 0.013;
                t.sin() * 0.5 + (t * 1.7).cos() * 0.3
            })
            .collect();
        let weight_data: Vec<f32> = (0..d).map(|i| 1.0 + 0.1 * ((i as f32) * 0.05).sin()).collect();

        let input_f32 = Array::from_slice(&input_data, shape);
        let weight_f32 = Array::from_slice(&weight_data, &[d as i32]);
        let input_bf16 = input_f32.as_dtype(Dtype::Bfloat16)?;
        let weight_bf16 = weight_f32.as_dtype(Dtype::Bfloat16)?;
        input_bf16.eval()?;
        weight_bf16.eval()?;

        // Reference: rms_norm in mlx, then transpose ON CPU to avoid any
        // mlx lazy-stride ambiguity (transpose_axes returns a view; .as_slice
        // would yield the original physical layout).
        let ref_norm = mlx_rs::fast::rms_norm(&input_bf16, &weight_bf16, eps)?;
        ref_norm.eval()?;
        let ref_norm_f32 = ref_norm.as_dtype(Dtype::Float32)?;
        ref_norm_f32.eval()?;
        let ref_blhd: Vec<f32> = ref_norm_f32.as_slice::<f32>().to_vec();

        // CPU transpose [B, L, H, D] → [B, H, L, D].
        let mut ref_bhld = vec![0.0f32; b * h * l * d];
        for bi in 0..b {
            for li in 0..l {
                for hi in 0..h {
                    let src = ((bi * l + li) * h + hi) * d;
                    let dst = ((bi * h + hi) * l + li) * d;
                    ref_bhld[dst..dst + d].copy_from_slice(&ref_blhd[src..src + d]);
                }
            }
        }

        // Our fused path.
        let got = run_rms_norm_transpose_bf16(&input_bf16, &weight_bf16, eps)?;
        got.eval()?;

        let got_shape = got.shape();
        assert_eq!(
            got_shape,
            &[b as i32, h as i32, l as i32, d as i32],
            "output shape mismatch for {label}"
        );

        let got_f32 = got.as_dtype(Dtype::Float32)?;
        got_f32.eval()?;

        let ref_vec = ref_bhld;
        let got_vec: Vec<f32> = got_f32.as_slice::<f32>().to_vec();

        let max_diff = ref_vec
            .iter()
            .zip(&got_vec)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);

        let n_rows = n / d;
        let mut min_cos = 1.0f32;
        for r in 0..n_rows {
            let r_a = &ref_vec[r * d..(r + 1) * d];
            let r_b = &got_vec[r * d..(r + 1) * d];
            let dot: f32 = r_a.iter().zip(r_b).map(|(x, y)| x * y).sum();
            let na: f32 = r_a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = r_b.iter().map(|x| x * x).sum::<f32>().sqrt();
            let cos = if na > 0.0 && nb > 0.0 {
                (dot / (na * nb)).clamp(-1.0, 1.0)
            } else {
                1.0
            };
            min_cos = min_cos.min(cos);
        }

        tr_max_diff = tr_max_diff.max(max_diff);
        tr_min_cos = tr_min_cos.min(min_cos);

        println!(
            "{label} ->  max|Δ|={max_diff:.3e}  min_cos={min_cos:.6}  ({})",
            if max_diff <= 3.0e-2 && min_cos >= 0.9999 { "OK" } else { "FAIL" }
        );
    }

    println!();
    println!("=== fused-transpose overall: max|Δ|={tr_max_diff:.3e} min_cos={tr_min_cos:.6} ===");
    if tr_max_diff > 3.0e-2 || tr_min_cos < 0.9999 {
        eprintln!("FUSED-TRANSPOSE PARITY FAIL");
        std::process::exit(1);
    }
    println!("PARITY OK — rms_norm_transpose_bf16 matches (rms_norm + transpose_axes) within bf16 epsilon");
    Ok(())
}

#[cfg(not(feature = "mlx-native"))]
fn main() {
    eprintln!("requires --features mlx-native");
    std::process::exit(2);
}
