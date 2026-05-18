//! end-to-end round-trip smoke test.
//!
//!   mlx Array ─► metal_buffer_ptr() ─► [Metal kernel: out = in * 2] ─►
//!   MTL::Buffer ─► array_from_metal_buffer() ─► mlx Array
//!
//! Compares bit-identical against `mlx_rs::ops::multiply(input, 2.0)`.
//!
//! Run:
//!   MLX_LOCAL_SOURCE_DIR=/path/to/mlx \
//!     cargo run --release --features mlx-native --example bench_mul_by_2_roundtrip
//!
//! Expected: 3 sub-tests, all OK, max|Δ| = 0.0 exact.

#[cfg(feature = "mlx-native")]
fn main() -> anyhow::Result<()> {
    use mlx_rs::{Array, Dtype};
    use lumen_mlx::native_metal_bridge::{run_mul_by_2, run_mul_by_2_bf16};

    println!("=== Phase 1.8 M1.5 + M2 bf16: native Metal kernel round-trip ===");

    // ── (1) Tiny [2,2] f32 ───────────────────────────────────────────────
    {
        let input = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
        input.eval()?;
        let expected = mlx_rs::ops::multiply(&input, Array::from_f32(2.0))?;
        expected.eval()?;

        let got = run_mul_by_2(&input)?;
        got.eval()?;

        let got_vec: Vec<f32> = got.as_slice::<f32>().to_vec();
        let exp_vec: Vec<f32> = expected.as_slice::<f32>().to_vec();

        let max_abs_diff = got_vec
            .iter()
            .zip(&exp_vec)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);

        println!("(1) [2,2] f32:");
        println!("    got      = {:?}", got_vec);
        println!("    expected = {:?}", exp_vec);
        println!("    max|Δ|   = {:.3e}", max_abs_diff);
        assert!(max_abs_diff == 0.0, "FAIL: max|Δ| should be exactly 0.0");
        println!("    OK (bit-identical)");
    }

    // ── (2) Mid-sized [4,4] f32 ──────────────────────────────────────────
    {
        let data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.5 - 3.0).collect();
        let input = Array::from_slice(&data, &[4, 4]);
        input.eval()?;
        let expected = mlx_rs::ops::multiply(&input, Array::from_f32(2.0))?;
        expected.eval()?;

        let got = run_mul_by_2(&input)?;
        got.eval()?;

        let got_vec: Vec<f32> = got.as_slice::<f32>().to_vec();
        let exp_vec: Vec<f32> = expected.as_slice::<f32>().to_vec();
        let max_abs_diff = got_vec
            .iter()
            .zip(&exp_vec)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);

        println!("(2) [4,4] f32 (16 elems, mixed signs):");
        println!("    max|Δ|   = {:.3e}", max_abs_diff);
        assert!(max_abs_diff == 0.0, "FAIL: max|Δ| should be exactly 0.0");
        println!("    OK (bit-identical)");
    }

    // ── (3) Stress [128,128] f32 ─────────────────────────────────────────
    {
        let n = 128 * 128;
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.137).sin()).collect();
        let input = Array::from_slice(&data, &[128, 128]);
        input.eval()?;
        let expected = mlx_rs::ops::multiply(&input, Array::from_f32(2.0))?;
        expected.eval()?;

        let got = run_mul_by_2(&input)?;
        got.eval()?;

        let got_vec: Vec<f32> = got.as_slice::<f32>().to_vec();
        let exp_vec: Vec<f32> = expected.as_slice::<f32>().to_vec();
        let max_abs_diff = got_vec
            .iter()
            .zip(&exp_vec)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);

        println!("(3) [128,128] f32 (16384 elems, sin pattern):");
        println!("    max|Δ|   = {:.3e}", max_abs_diff);
        assert!(max_abs_diff == 0.0, "FAIL: max|Δ| should be exactly 0.0");
        println!("    OK (bit-identical)");
    }

    // ── (4) [4,4] bf16 ───────────────────────────────────────────────────
    {
        let data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.5 - 3.0).collect();
        let input_f32 = Array::from_slice(&data, &[4, 4]);
        let input = input_f32.as_dtype(Dtype::Bfloat16)?;
        input.eval()?;

        let expected = mlx_rs::ops::multiply(&input, Array::from_f32(2.0).as_dtype(Dtype::Bfloat16)?)?;
        expected.eval()?;

        let got = run_mul_by_2_bf16(&input)?;
        got.eval()?;

        let got_f32 = got.as_dtype(Dtype::Float32)?;
        let exp_f32 = expected.as_dtype(Dtype::Float32)?;
        got_f32.eval()?;
        exp_f32.eval()?;

        let got_vec: Vec<f32> = got_f32.as_slice::<f32>().to_vec();
        let exp_vec: Vec<f32> = exp_f32.as_slice::<f32>().to_vec();
        let max_abs_diff = got_vec
            .iter()
            .zip(&exp_vec)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);

        println!("(4) [4,4] bf16 (16 elems, mixed signs):");
        println!("    got      = {:?}", got_vec);
        println!("    expected = {:?}", exp_vec);
        println!("    max|Δ|   = {:.3e}", max_abs_diff);
        assert!(max_abs_diff == 0.0, "FAIL: bf16 mul-by-2 should be exact for all values");
        println!("    OK (bit-identical)");
    }

    // ── (5) [128,128] bf16 sin pattern ───────────────────────────────────
    {
        let n = 128 * 128;
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.137).sin()).collect();
        let input_f32 = Array::from_slice(&data, &[128, 128]);
        let input = input_f32.as_dtype(Dtype::Bfloat16)?;
        input.eval()?;

        let expected = mlx_rs::ops::multiply(&input, Array::from_f32(2.0).as_dtype(Dtype::Bfloat16)?)?;
        expected.eval()?;

        let got = run_mul_by_2_bf16(&input)?;
        got.eval()?;

        let got_f32 = got.as_dtype(Dtype::Float32)?;
        let exp_f32 = expected.as_dtype(Dtype::Float32)?;
        got_f32.eval()?;
        exp_f32.eval()?;

        let got_vec: Vec<f32> = got_f32.as_slice::<f32>().to_vec();
        let exp_vec: Vec<f32> = exp_f32.as_slice::<f32>().to_vec();
        let max_abs_diff = got_vec
            .iter()
            .zip(&exp_vec)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);

        println!("(5) [128,128] bf16 (16384 elems, sin pattern):");
        println!("    max|Δ|   = {:.3e}", max_abs_diff);
        assert!(
            max_abs_diff == 0.0,
            "FAIL: bf16 mul-by-2 should be exact (2x is power-of-2)"
        );
        println!("    OK (bit-identical)");
    }

    println!();
    println!("=== ALL 5 sub-tests passed — M1.5 (f32) + M2 (bf16) round-trips LIVE ===");
    println!();
    println!("Next: M2 real kernel — fused RMSNorm + scale (Q/K-norm) in bf16.");
    Ok(())
}

#[cfg(not(feature = "mlx-native"))]
fn main() {
    eprintln!("bench_mul_by_2_roundtrip requires --features mlx-native");
    std::process::exit(2);
}
