//! `SiluMulBf16InBf16Out` parity vs Candle reference + ICB
//! parity vs standalone path.
//!
//! Two tests:
//!   1. `silu_mul_standalone_matches_candle_reference` — kernel's elementwise
//!      math matches Candle's `silu(narrow0) * narrow1` chain to within bf16
//!      rounding tolerance (per-element max-abs ≤ a few ulp).
//!   2. `silu_mul_icb_matches_standalone` — ICB-recorded dispatch produces
//!      bit-identical output to the standalone path.
//!
//! Run:
//!   cargo test --test silu_mul_parity -p lumen-metal \
//!     --features model-integration --release -- --nocapture

#![cfg(feature = "model-integration")]

use candle_core::backend::BackendDevice;
use candle_core::{DType, Device, Tensor};
use lumen_metal::metal::CommandBufferExt;
use lumen_metal::metal::IndirectCommandBuffer;
use lumen_metal::silu_mul::SiluMulBf16InBf16Out;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice, MTLResourceUsage};

const INTER: usize = 5120;

fn synth(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            (((s >> 8) & 0xff) as f32 / 256.0 - 0.5) * scale
        })
        .collect()
}

fn metal_device() -> Option<Device> {
    Device::new_metal(0).ok()
}

fn candle_silu_mul_reference(combined: &Tensor, inter: usize) -> Tensor {
    // Mirror the production silu*mul chain exactly:
    //   combined_bf16 → f32 → narrow gate / up → silu*up → bf16
    let combined_f32 = combined.to_dtype(DType::F32).unwrap();
    let last = combined_f32.dims().len() - 1;
    let gate = combined_f32
        .narrow(last, 0, inter)
        .unwrap()
        .contiguous()
        .unwrap();
    let up = combined_f32
        .narrow(last, inter, inter)
        .unwrap()
        .contiguous()
        .unwrap();
    let h = (candle_nn::ops::silu(&gate).unwrap() * up).unwrap();
    h.to_dtype(DType::BF16).unwrap()
}

#[test]
fn silu_mul_standalone_matches_candle_reference() {
    let dev = match metal_device() {
        Some(d) => d,
        None => {
            eprintln!("[skip] no Metal device");
            return;
        }
    };
    let kernel = match SiluMulBf16InBf16Out::new() {
        Ok(k) => k,
        Err(e) => {
            eprintln!("[skip] cannot init SiluMulBf16InBf16Out: {e}");
            return;
        }
    };

    let m = 1; // decode-shape
    let combined_data = synth(m * 2 * INTER, 0xFADE_FADE, 1.0);
    let combined = Tensor::from_vec(combined_data, &[m, 2 * INTER], &dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .contiguous()
        .unwrap();

    let y_ref = candle_silu_mul_reference(&combined, INTER);
    let y_kernel = kernel.forward(&combined).unwrap();

    let ref_v: Vec<f32> = y_ref
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let kernel_v: Vec<f32> = y_kernel
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    assert_eq!(ref_v.len(), kernel_v.len());
    let mut max_abs: f32 = 0.0;
    let mut bit_diffs = 0usize;
    for (a, b) in ref_v.iter().zip(kernel_v.iter()) {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
        if a.to_bits() != b.to_bits() {
            bit_diffs += 1;
        }
    }
    eprintln!();
    eprintln!("=== silu_mul standalone vs Candle reference ===");
    eprintln!("Compared:    {} elements", ref_v.len());
    eprintln!("Max-abs:     {:.6e}", max_abs);
    eprintln!("Bit diffs:   {bit_diffs} (bf16 rounding allowed)");

    // bf16 rounding: silu involves exp() so a few ulp drift expected.
    // Tolerance: max-abs ≤ 1e-2 (generous, well within bf16 mantissa).
    assert!(
        max_abs < 1e-2,
        "silu_mul kernel diverges from Candle reference: max-abs {max_abs:.6e} > 1e-2"
    );
}

#[test]
fn silu_mul_icb_matches_standalone() {
    let dev = match metal_device() {
        Some(d) => d,
        None => {
            eprintln!("[skip] no Metal device");
            return;
        }
    };
    let kernel = match SiluMulBf16InBf16Out::new() {
        Ok(k) => k,
        Err(e) => {
            eprintln!("[skip] cannot init SiluMulBf16InBf16Out: {e}");
            return;
        }
    };

    let m = 1;
    let combined_data = synth(m * 2 * INTER, 0x1234_5678, 1.0);
    let combined = Tensor::from_vec(combined_data, &[m, 2 * INTER], &dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .contiguous()
        .unwrap();

    // Standalone reference.
    let y_std = kernel.forward(&combined).unwrap();
    let std_bits: Vec<u32> = y_std
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .map(|f| f.to_bits())
        .collect();

    // ICB path.
    let metal_dev = match combined.device() {
        Device::Metal(m) => m,
        _ => unreachable!(),
    };
    let raw_device: Retained<ProtocolObject<dyn MTLDevice>> =
        Retained::from(metal_dev.metal_device().as_ref());
    let icb = IndirectCommandBuffer::new(&raw_device, 1, 3).expect("ICB alloc");
    let dims_buf = kernel.make_dims_buf(INTER);

    let y_icb_tensor = Tensor::zeros(&[m, INTER], DType::BF16, &dev).unwrap();

    let extract = |t: &Tensor| -> (lumen_metal::metal::Buffer, u64) {
        let (storage, layout) = t.storage_and_layout();
        match &*storage {
            candle_core::Storage::Metal(ms) => {
                let off = (layout.start_offset() * t.dtype().size_in_bytes()) as u64;
                (ms.buffer().clone(), off)
            }
            _ => panic!("not metal"),
        }
    };
    let (c_buf, c_off) = extract(&combined);
    let (y_buf, y_off) = extract(&y_icb_tensor);

    kernel.record_icb(&icb, 0, &c_buf, c_off, &y_buf, y_off, &dims_buf, m, INTER);

    // Drain Candle's pending zero-fill BEFORE submitting our custom CB
    // (lesson learned in 17.D-1c).
    let _ = metal_dev.synchronize();

    let cmd = lumen_metal::metal::new_command_buffer(&metal_dev.command_queue().unwrap());
    let enc = cmd.auto_compute_encoder();
    let usage = MTLResourceUsage(MTLResourceUsage::Read.0 | MTLResourceUsage::Write.0);
    enc.use_buffers_for_icb(&[&c_buf, &y_buf, &dims_buf], usage);
    enc.execute_commands_in_buffer(&icb, 1);
    drop(enc);
    cmd.commit();
    cmd.wait_until_completed();

    let icb_bits: Vec<u32> = y_icb_tensor
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .map(|f| f.to_bits())
        .collect();

    assert_eq!(std_bits.len(), icb_bits.len());
    let diffs = std_bits
        .iter()
        .zip(icb_bits.iter())
        .filter(|(a, b)| a != b)
        .count();

    eprintln!();
    eprintln!("=== silu_mul ICB vs standalone ===");
    eprintln!("Compared:  {} elements", std_bits.len());
    eprintln!("Diffs:     {diffs} {}", if diffs == 0 { "✓" } else { "✗" });

    if diffs > 0 {
        panic!("silu_mul ICB diverged from standalone");
    }
}
