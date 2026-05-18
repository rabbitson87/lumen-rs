//! parity test for `Affine4Linear::forward_*_icb`.
//!
//! Verifies that the ICB record-once-replay-many path produces output
//! bit-identical to the existing `forward_*` non-ICB path. Runs N
//! decode-shaped iterations to exercise the cached-replay fast path
//! (after iter 0's record).
//!
//! Run:
//!   LUMEN_ICB=1 cargo test --test affine4_icb_parity \
//!     -p lumen-metal --features model-integration --release -- --nocapture

#![cfg(feature = "model-integration")]

use candle_core::{DType, Device, Tensor};
use std::sync::Arc;
use lumen_metal::affine4_gpu::{Affine4Context, Affine4Weight};
use lumen_metal::affine4_linear::Affine4Linear;

const OUT: usize = 5120;
const IN: usize = 5120;
const ITERS: usize = 5;

fn synth_packed(out: usize, ins: usize, seed: u32) -> Vec<u32> {
    let n = out * ins / 8;
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            s
        })
        .collect()
}

fn synth_scales_or_biases(out: usize, ins: usize, seed: u32, neg: bool) -> Vec<u16> {
    let n = out * ins / 64;
    let mut s = seed;
    let off = if neg { -0.005 } else { 0.01 };
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            let f = ((s >> 8) & 0xff) as f32 / 256.0 * 0.01 + off;
            (f.to_bits() >> 16) as u16
        })
        .collect()
}

fn synth_x(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 8) & 0xff) as f32 / 256.0 - 0.5
        })
        .collect()
}

fn metal_device() -> Option<Device> {
    Device::new_metal(0).ok()
}

#[test]
fn forward_bf16_in_bf16_out_icb_matches_standard() {
    // Force the env-gate ON for the duration of this test.
    unsafe { std::env::set_var("LUMEN_ICB", "1"); }

    let dev = match metal_device() {
        Some(d) => d,
        None => {
            eprintln!("[skip] no Metal device");
            return;
        }
    };
    let ctx = match Affine4Context::new() {
        Ok(c) => Arc::new(c),
        Err(_) => {
            eprintln!("[skip] no Affine4 context");
            return;
        }
    };

    let packed = synth_packed(OUT, IN, 0xDEADBEEF);
    let scales = synth_scales_or_biases(OUT, IN, 0xCAFEBABE, false);
    let biases = synth_scales_or_biases(OUT, IN, 0x12345678, true);

    let weight = Affine4Weight::from_host(&ctx.ctx, &packed, &scales, &biases, OUT, IN)
        .expect("weight upload");
    let linear = Affine4Linear::new(weight, None, ctx.clone());

    // Decode-shaped input.
    let x_data = synth_x(IN, 0xFADEFADE);
    let x = Tensor::from_vec(x_data, &[1, 1, IN], &dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .contiguous()
        .unwrap();

    // Drain the env-gate and call the standard path for the reference.
    unsafe { std::env::set_var("LUMEN_ICB", "0"); }
    let y_ref = linear.forward_bf16_in_bf16_out(&x).unwrap();

    // Now flip the env-gate back ON and call the ICB path N times.
    unsafe { std::env::set_var("LUMEN_ICB", "1"); }
    let mut diffs_total = 0usize;
    let mut compared = 0usize;
    for i in 0..ITERS {
        let y_icb = linear.forward_bf16_in_bf16_out_icb(&x).unwrap();
        let ref_cpu = y_ref.flatten_all().unwrap().to_dtype(DType::F32).unwrap()
            .to_vec1::<f32>().unwrap();
        let icb_cpu = y_icb.flatten_all().unwrap().to_dtype(DType::F32).unwrap()
            .to_vec1::<f32>().unwrap();
        assert_eq!(ref_cpu.len(), icb_cpu.len());
        let mut diffs = 0usize;
        for (a, b) in ref_cpu.iter().zip(icb_cpu.iter()) {
            if a.to_bits() != b.to_bits() {
                diffs += 1;
            }
        }
        diffs_total += diffs;
        compared += ref_cpu.len();
        eprintln!("  iter {i}: diffs={diffs} / {} {}",
            ref_cpu.len(),
            if diffs == 0 { "✓" } else { "✗" });
    }

    eprintln!();
    eprintln!("=== Phase 17.C parity (forward_bf16_in_bf16_out_icb) ===");
    eprintln!("Iterations:       {ITERS}");
    eprintln!("Total compared:   {compared}");
    eprintln!("Total diffs:      {diffs_total}");
    if diffs_total > 0 {
        panic!("ICB path diverged from standard path");
    }
}

/// Exercises the re-record fast-path miss: forces buffer-identity changes
/// across calls by pre-allocating two distinct input tensors and alternating
/// between them. Each alternation should trigger `needs_record = true` and
/// re-record at slot 0 with the new buffer baked in. Output must remain
/// bit-identical to the standard path for both inputs.
#[test]
fn forward_bf16_in_bf16_out_icb_re_records_on_buffer_change() {
    unsafe { std::env::set_var("LUMEN_ICB", "1"); }

    let dev = match metal_device() {
        Some(d) => d,
        None => {
            eprintln!("[skip] no Metal device");
            return;
        }
    };
    let ctx = match Affine4Context::new() {
        Ok(c) => Arc::new(c),
        Err(_) => {
            eprintln!("[skip] no Affine4 context");
            return;
        }
    };

    let packed = synth_packed(OUT, IN, 0xDEADBEEF);
    let scales = synth_scales_or_biases(OUT, IN, 0xCAFEBABE, false);
    let biases = synth_scales_or_biases(OUT, IN, 0x12345678, true);

    let weight = Affine4Weight::from_host(&ctx.ctx, &packed, &scales, &biases, OUT, IN)
        .expect("weight upload");
    let linear = Affine4Linear::new(weight, None, ctx.clone());

    // Two long-lived input tensors with different content. Long-lived =>
    // they each have distinct underlying MTLBuffers held alive throughout
    // the test, forcing the cache to re-record on every alternation.
    let xa_data = synth_x(IN, 0x1111_1111);
    let xa = Tensor::from_vec(xa_data, &[1, 1, IN], &dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .contiguous()
        .unwrap();
    let xb_data = synth_x(IN, 0x2222_2222);
    let xb = Tensor::from_vec(xb_data, &[1, 1, IN], &dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .contiguous()
        .unwrap();

    // References computed via standard path.
    unsafe { std::env::set_var("LUMEN_ICB", "0"); }
    let ya_ref = linear.forward_bf16_in_bf16_out(&xa).unwrap();
    let yb_ref = linear.forward_bf16_in_bf16_out(&xb).unwrap();
    unsafe { std::env::set_var("LUMEN_ICB", "1"); }

    let to_bits = |t: &Tensor| -> Vec<u32> {
        t.flatten_all().unwrap().to_dtype(DType::F32).unwrap()
            .to_vec1::<f32>().unwrap().iter().map(|f| f.to_bits()).collect()
    };
    let ya_bits = to_bits(&ya_ref);
    let yb_bits = to_bits(&yb_ref);

    // Sanity — distinct inputs must produce distinct outputs (otherwise the
    // test would pass trivially).
    let differing = ya_bits.iter().zip(yb_bits.iter()).filter(|(a, b)| a != b).count();
    assert!(
        differing > 0,
        "test invalid: synthetic inputs xa/xb produced identical outputs"
    );

    // Alternate xa / xb / xa / xb / xa across the ICB path. The cache
    // must re-record at each alternation and produce the matching output.
    let mut diffs_total = 0usize;
    for i in 0..6 {
        let (label, x_in, ref_bits) = if i % 2 == 0 {
            ("xa", &xa, &ya_bits)
        } else {
            ("xb", &xb, &yb_bits)
        };
        let y_icb = linear.forward_bf16_in_bf16_out_icb(x_in).unwrap();
        let icb_bits: Vec<u32> = y_icb.flatten_all().unwrap().to_dtype(DType::F32).unwrap()
            .to_vec1::<f32>().unwrap().iter().map(|f| f.to_bits()).collect();
        let diffs = icb_bits.iter().zip(ref_bits.iter()).filter(|(a, b)| a != b).count();
        diffs_total += diffs;
        eprintln!("  iter {i} ({label}): diffs={diffs} {}",
            if diffs == 0 { "✓" } else { "✗" });
    }

    if diffs_total > 0 {
        panic!("re-record path produced incorrect output: {diffs_total} total diffs");
    }
}

/// Validates `batch_buf` re-allocation + cache invalidation when the batch
/// size changes between calls. First call uses batch=2 (seq dimension), then
/// batch=1; the cache must re-record both variants on the batch change.
#[test]
fn forward_bf16_in_bf16_out_icb_handles_batch_change() {
    unsafe { std::env::set_var("LUMEN_ICB", "1"); }

    let dev = match metal_device() {
        Some(d) => d,
        None => {
            eprintln!("[skip] no Metal device");
            return;
        }
    };
    let ctx = match Affine4Context::new() {
        Ok(c) => Arc::new(c),
        Err(_) => {
            eprintln!("[skip] no Affine4 context");
            return;
        }
    };

    let packed = synth_packed(OUT, IN, 0xDEADBEEF);
    let scales = synth_scales_or_biases(OUT, IN, 0xCAFEBABE, false);
    let biases = synth_scales_or_biases(OUT, IN, 0x12345678, true);

    let weight = Affine4Weight::from_host(&ctx.ctx, &packed, &scales, &biases, OUT, IN)
        .expect("weight upload");
    let linear = Affine4Linear::new(weight, None, ctx.clone());

    // batch = 2 first (prefill-shape), then batch = 1 (decode-shape).
    let x2_data = synth_x(2 * IN, 0x3333_3333);
    let x2 = Tensor::from_vec(x2_data, &[1, 2, IN], &dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .contiguous()
        .unwrap();
    let x1_data = synth_x(IN, 0x4444_4444);
    let x1 = Tensor::from_vec(x1_data, &[1, 1, IN], &dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .contiguous()
        .unwrap();

    unsafe { std::env::set_var("LUMEN_ICB", "0"); }
    let y2_ref = linear.forward_bf16_in_bf16_out(&x2).unwrap();
    let y1_ref = linear.forward_bf16_in_bf16_out(&x1).unwrap();
    unsafe { std::env::set_var("LUMEN_ICB", "1"); }

    let to_bits = |t: &Tensor| -> Vec<u32> {
        t.flatten_all().unwrap().to_dtype(DType::F32).unwrap()
            .to_vec1::<f32>().unwrap().iter().map(|f| f.to_bits()).collect()
    };
    let y2_bits = to_bits(&y2_ref);
    let y1_bits = to_bits(&y1_ref);

    let y2_icb = linear.forward_bf16_in_bf16_out_icb(&x2).unwrap();
    let y2_icb_bits = to_bits(&y2_icb);
    let d2 = y2_icb_bits.iter().zip(y2_bits.iter()).filter(|(a, b)| a != b).count();
    eprintln!("  batch=2: diffs={d2} / {} {}", y2_bits.len(), if d2 == 0 { "✓" } else { "✗" });

    let y1_icb = linear.forward_bf16_in_bf16_out_icb(&x1).unwrap();
    let y1_icb_bits = to_bits(&y1_icb);
    let d1 = y1_icb_bits.iter().zip(y1_bits.iter()).filter(|(a, b)| a != b).count();
    eprintln!("  batch=1: diffs={d1} / {} {}", y1_bits.len(), if d1 == 0 { "✓" } else { "✗" });

    // And back to batch=2 to validate symmetric re-record.
    let y2_again = linear.forward_bf16_in_bf16_out_icb(&x2).unwrap();
    let y2_again_bits = to_bits(&y2_again);
    let d2b = y2_again_bits.iter().zip(y2_bits.iter()).filter(|(a, b)| a != b).count();
    eprintln!("  batch=2 (return): diffs={d2b} / {} {}", y2_bits.len(), if d2b == 0 { "✓" } else { "✗" });

    if d1 > 0 || d2 > 0 || d2b > 0 {
        panic!("batch-change re-record produced incorrect output: d1={d1} d2={d2} d2b={d2b}");
    }
}

#[test]
fn forward_with_residual_bf16_in_bf16_out_icb_matches_standard() {
    unsafe { std::env::set_var("LUMEN_ICB", "1"); }

    let dev = match metal_device() {
        Some(d) => d,
        None => {
            eprintln!("[skip] no Metal device");
            return;
        }
    };
    let ctx = match Affine4Context::new() {
        Ok(c) => Arc::new(c),
        Err(_) => {
            eprintln!("[skip] no Affine4 context");
            return;
        }
    };

    let packed = synth_packed(OUT, IN, 0xDEADBEEF);
    let scales = synth_scales_or_biases(OUT, IN, 0xCAFEBABE, false);
    let biases = synth_scales_or_biases(OUT, IN, 0x12345678, true);

    let weight = Affine4Weight::from_host(&ctx.ctx, &packed, &scales, &biases, OUT, IN)
        .expect("weight upload");
    let linear = Affine4Linear::new(weight, None, ctx.clone());

    let x_data = synth_x(IN, 0xFADEFADE);
    let x = Tensor::from_vec(x_data, &[1, 1, IN], &dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .contiguous()
        .unwrap();
    let r_data = synth_x(OUT, 0xBEEFBEEF);
    let residual = Tensor::from_vec(r_data, &[1, 1, OUT], &dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .contiguous()
        .unwrap();

    unsafe { std::env::set_var("LUMEN_ICB", "0"); }
    let y_ref = linear
        .forward_with_residual_bf16_in_bf16_out(&x, &residual)
        .unwrap();

    unsafe { std::env::set_var("LUMEN_ICB", "1"); }
    let mut diffs_total = 0usize;
    let mut compared = 0usize;
    for i in 0..ITERS {
        let y_icb = linear
            .forward_with_residual_bf16_in_bf16_out_icb(&x, &residual)
            .unwrap();
        let ref_cpu = y_ref.flatten_all().unwrap().to_dtype(DType::F32).unwrap()
            .to_vec1::<f32>().unwrap();
        let icb_cpu = y_icb.flatten_all().unwrap().to_dtype(DType::F32).unwrap()
            .to_vec1::<f32>().unwrap();
        let mut diffs = 0usize;
        for (a, b) in ref_cpu.iter().zip(icb_cpu.iter()) {
            if a.to_bits() != b.to_bits() {
                diffs += 1;
            }
        }
        diffs_total += diffs;
        compared += ref_cpu.len();
        eprintln!("  iter {i}: diffs={diffs} / {} {}",
            ref_cpu.len(),
            if diffs == 0 { "✓" } else { "✗" });
    }

    eprintln!();
    eprintln!("=== Phase 17.C parity (forward_with_residual_*_icb) ===");
    eprintln!("Iterations:       {ITERS}");
    eprintln!("Total compared:   {compared}");
    eprintln!("Total diffs:      {diffs_total}");
    if diffs_total > 0 {
        panic!("ICB residual path diverged from standard path");
    }
}
