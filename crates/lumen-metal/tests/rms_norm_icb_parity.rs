//! RmsNormBf16InBf16Out ICB parity.
//!
//! Validates that an ICB-recorded `rms_norm_bf16in_bf16out` dispatch
//! produces output bit-identical to the standard `forward()` path. This is
//! the foundation step for per-MLP-block ICB (Phase 17.D-1c) — RmsNorm is
//! the per-layer norm at the input + post-attention sites.
//!
//! Run:
//!   cargo test --test rms_norm_icb_parity -p lumen-metal \
//!     --features model-integration --release -- --nocapture

#![cfg(feature = "model-integration")]

use candle_core::{DType, Device, Tensor};
use lumen_metal::metal::CommandBufferExt;
use lumen_metal::metal::IndirectCommandBuffer;
use lumen_metal::rms_norm::RmsNormBf16InBf16Out;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice, MTLResourceUsage};

const HIDDEN: usize = 5120;
const EPS: f32 = 1e-6;

fn synth_x(n: usize, seed: u32, scale: f32) -> Vec<f32> {
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

#[test]
fn rms_norm_bf16in_bf16out_icb_matches_standard() {
    let dev = match metal_device() {
        Some(d) => d,
        None => {
            eprintln!("[skip] no Metal device");
            return;
        }
    };

    let rms = match RmsNormBf16InBf16Out::new(EPS) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[skip] cannot init RmsNormBf16InBf16Out: {e}");
            return;
        }
    };

    let m = 1; // decode-shape: one row
    let x_data = synth_x(m * HIDDEN, 0xFADE_FADE, 1.0);
    let w_data = synth_x(HIDDEN, 0xCAFE_BABE, 0.5);

    let x = Tensor::from_vec(x_data, &[m, HIDDEN], &dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .contiguous()
        .unwrap();
    let w = Tensor::from_vec(w_data, &[HIDDEN], &dev)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .contiguous()
        .unwrap();

    // ── Standard path reference ────────────────────────────────────────
    let y_ref = rms.forward(&x, &w).unwrap();
    let ref_bits: Vec<u32> = y_ref
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .map(|f| f.to_bits())
        .collect();

    // ── ICB path ───────────────────────────────────────────────────────
    use candle_core::backend::BackendDevice;
    let metal_dev = match x.device() {
        Device::Metal(m) => m,
        _ => unreachable!(),
    };
    let raw_device: Retained<ProtocolObject<dyn MTLDevice>> =
        Retained::from(metal_dev.metal_device().as_ref());

    // 4 buffers: x, weight, y, dims.
    let icb = IndirectCommandBuffer::new(&raw_device, 1, 4).expect("ICB alloc");
    let dims_buf = rms.make_dims_buf(HIDDEN);

    let y_icb_tensor = Tensor::zeros(&[m, HIDDEN], DType::BF16, &dev).unwrap();

    // Extract underlying buffers from the Candle Tensors.
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
    let (x_buf, x_off) = extract(&x);
    let (w_buf, w_off) = extract(&w);
    let (y_buf, y_off) = extract(&y_icb_tensor);

    rms.record_icb(
        &icb, 0, &x_buf, x_off, &w_buf, w_off, &y_buf, y_off, &dims_buf, m,
    );

    // Drain any Candle pending command buffers so they don't race with our
    // ICB submission on the shared queue (Tensor::zeros enqueues an async
    // zero-fill on the same queue; without this drain it can land AFTER
    // our ICB and clobber the kernel output).
    let _ = metal_dev.synchronize();

    let cmd = lumen_metal::metal::new_command_buffer(&metal_dev.command_queue().unwrap());
    let enc = cmd.auto_compute_encoder();
    let usage = MTLResourceUsage(MTLResourceUsage::Read.0 | MTLResourceUsage::Write.0);
    enc.use_buffers_for_icb(&[&x_buf, &w_buf, &y_buf, &dims_buf], usage);
    enc.execute_commands_in_buffer(&icb, 1);
    drop(enc);
    cmd.commit();
    cmd.wait_until_completed();

    // Compare bits.
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

    assert_eq!(ref_bits.len(), icb_bits.len());
    let diffs = ref_bits
        .iter()
        .zip(icb_bits.iter())
        .filter(|(a, b)| a != b)
        .count();

    // Debug output — first 8 elements of each.
    let ref_f: Vec<f32> = ref_bits
        .iter()
        .take(8)
        .map(|b| f32::from_bits(*b))
        .collect();
    let icb_f: Vec<f32> = icb_bits
        .iter()
        .take(8)
        .map(|b| f32::from_bits(*b))
        .collect();
    eprintln!();
    eprintln!("=== Phase 17.D-1 — RmsNormBf16InBf16Out ICB parity ===");
    eprintln!("Hidden:   {HIDDEN}");
    eprintln!("Compared: {} elements", ref_bits.len());
    eprintln!("Diffs:    {diffs} {}", if diffs == 0 { "✓" } else { "✗" });
    eprintln!("ref[0..8] = {ref_f:?}");
    eprintln!("icb[0..8] = {icb_f:?}");

    if diffs > 0 {
        panic!("RmsNorm ICB diverged from standard");
    }
}
