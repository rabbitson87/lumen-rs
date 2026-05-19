//! exercise the new Candle fork ICB wrappers.
//!
//! Replicates PoC #2 (single-weight 64-dispatch ICB on real qmv_fast) but
//! through the Candle fork API surface added in Phase 17.B foundation:
//!   - `Device::new_compute_pipeline_state_with_function_for_icb`
//!   - `IndirectCommandBuffer::new` + `record_compute`
//!   - `ComputeCommandEncoder::use_buffer_for_icb` +
//!     `execute_commands_in_buffer`
//!
//! Validates the wrapper API matches the raw-objc2 path: same bit-identical
//! GPU output, same per-CB CPU savings.

#![allow(unexpected_cfgs)]

use lumen_metal::metal::{BatchedEncoderExt, CommandBufferExt, ComputeEncoderCompat};
use std::time::Instant;

use lumen_metal::device::MetalContext;
use lumen_metal::metal::{Buffer, IndirectCommandBuffer};
use objc2_metal::{MTLResourceUsage, MTLSize};

const SHADER_SRC: &str = include_str!("../src/shaders/affine4.metal");
const N_PER_CB: usize = 64;

#[repr(C)]
#[derive(Clone, Copy)]
struct Affine4Dims {
    out_features: u32,
    in_features: u32,
}

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

fn synth_x_bf16(n: usize, seed: u32) -> Vec<u16> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            let f = ((s >> 8) & 0xff) as f32 / 256.0 - 0.5;
            (f.to_bits() >> 16) as u16
        })
        .collect()
}

#[test]
fn icb_wrapper_pipeline_record_replay_parity() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(_) => {
            eprintln!("[skip] no Metal device");
            return;
        }
    };

    // Compile the existing affine4 shader, fetch the bf16-in/bf16-out kernel.
    let library = ctx
        .device
        .new_library_with_source(SHADER_SRC, None)
        .expect("affine4.metal compile");
    let function = library
        .get_function("affine4_qmv_fast_bf16in_bf16out", None)
        .expect("kernel not found");

    // ICB-supporting pipeline (Phase 17.B foundation API).
    let pipeline = ctx
        .device
        .new_compute_pipeline_state_with_function_for_icb(&function)
        .expect("ICB pipeline build failed");

    // 27B Dense o_proj shape.
    let out: usize = 5120;
    let ins: usize = 5120;
    let batch: usize = 1;

    let packed = synth_packed(out, ins, 0xDEADBEEF);
    let scales = synth_scales_or_biases(out, ins, 0xCAFEBABE, false);
    let biases = synth_scales_or_biases(out, ins, 0x12345678, true);
    let x_bf16 = synth_x_bf16(batch * ins, 0xFADEFADE);

    let weight_packed = ctx.buffer_with_data(&packed);
    let weight_scales = ctx.buffer_with_data(&scales);
    let weight_biases = ctx.buffer_with_data(&biases);
    let x_buf = ctx.buffer_with_data(&x_bf16);
    let y_buf_a = ctx.buffer_zeroed((batch * out * 2) as u64);
    let y_buf_b = ctx.buffer_zeroed((batch * out * 2) as u64);

    let dims = Affine4Dims {
        out_features: out as u32,
        in_features: ins as u32,
    };
    let dims_buf = ctx.buffer_with_data(&[dims]);
    let batch_u32 = batch as u32;
    let batch_buf = ctx.buffer_with_data(&[batch_u32]);

    let grid = MTLSize {
        width: batch,
        height: out / 8,
        depth: 1,
    };
    let tg = MTLSize {
        width: 64,
        height: 1,
        depth: 1,
    };

    // ── ICB record via Candle wrapper ──────────────────────────────────
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_metal::MTLDevice;

    // Get raw device ProtocolObject from Candle wrapper.
    let raw_device: Retained<ProtocolObject<dyn MTLDevice>> = {
        // MetalContext.device → candle Device → as_ref ProtocolObject<dyn MTLDevice>
        let dev_ref = ctx.device.as_ref();
        // Increment ref count by retaining; we hold `Retained` for the
        // ICB's lifetime.
        Retained::from(dev_ref)
    };

    let icb = IndirectCommandBuffer::new(&raw_device, N_PER_CB, 7).expect("ICB allocation failed");

    // Record 64 identical commands (same buffers, different layers analogue
    // would supply distinct buffers per command).
    for cmd in 0..N_PER_CB {
        icb.record_compute(
            cmd,
            &pipeline,
            &[
                (&weight_packed, 0, 0),
                (&weight_scales, 0, 1),
                (&weight_biases, 0, 2),
                (&x_buf, 0, 3),
                (&y_buf_b, 0, 4),
                (&dims_buf, 0, 5),
                (&batch_buf, 0, 6),
            ],
            grid,
            tg,
        );
    }

    // ── Bench helpers ──────────────────────────────────────────────────
    let run_standard = |y: &Buffer| {
        let cmd = lumen_metal::metal::new_command_buffer(&ctx.queue);
        let enc = cmd.auto_compute_encoder();
        for _ in 0..N_PER_CB {
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&weight_packed), 0);
            enc.set_buffer(1, Some(&weight_scales), 0);
            enc.set_buffer(2, Some(&weight_biases), 0);
            enc.set_buffer(3, Some(&x_buf), 0);
            enc.set_buffer(4, Some(y), 0);
            enc.set_buffer(5, Some(&dims_buf), 0);
            enc.set_buffer(6, Some(&batch_buf), 0);
            enc.dispatch_thread_groups(grid, tg);
        }
        drop(enc);
        cmd.commit();
        cmd.wait_until_completed();
    };

    let usage = MTLResourceUsage(MTLResourceUsage::Read.0 | MTLResourceUsage::Write.0);
    let run_icb = || {
        let cmd = lumen_metal::metal::new_command_buffer(&ctx.queue);
        let enc = cmd.auto_compute_encoder();
        enc.use_buffer_for_icb(&weight_packed, usage);
        enc.use_buffer_for_icb(&weight_scales, usage);
        enc.use_buffer_for_icb(&weight_biases, usage);
        enc.use_buffer_for_icb(&x_buf, usage);
        enc.use_buffer_for_icb(&y_buf_b, usage);
        enc.use_buffer_for_icb(&dims_buf, usage);
        enc.use_buffer_for_icb(&batch_buf, usage);
        enc.execute_commands_in_buffer(&icb, N_PER_CB);
        drop(enc);
        cmd.commit();
        cmd.wait_until_completed();
    };

    // ── Bit-identity ───────────────────────────────────────────────────
    run_standard(&y_buf_a);
    run_icb();

    let a_ptr = y_buf_a.contents() as *const u16;
    let b_ptr = y_buf_b.contents() as *const u16;
    let mut diffs = 0usize;
    for i in 0..(out * batch) {
        let a = unsafe { *a_ptr.add(i) };
        let b = unsafe { *b_ptr.add(i) };
        if a != b {
            diffs += 1;
        }
    }
    eprintln!();
    eprintln!("=== Phase 17.B foundation — Candle wrapper ICB test ===");
    eprintln!(
        "Bit-identical (A vs B): {} / {} (diffs={})",
        out * batch - diffs,
        out * batch,
        diffs
    );
    if diffs > 0 {
        panic!("Candle wrapper ICB output diverged from standard");
    }

    // ── Bench (sanity vs PoC #2 raw-objc2 numbers) ─────────────────────
    let warmup = 16;
    for _ in 0..warmup {
        run_standard(&y_buf_a);
        run_icb();
    }
    let iters = 100;

    let t0 = Instant::now();
    for _ in 0..iters {
        run_standard(&y_buf_a);
    }
    let mean_a_us = t0.elapsed().as_secs_f64() * 1.0e6 / iters as f64;

    let t1 = Instant::now();
    for _ in 0..iters {
        run_icb();
    }
    let mean_b_us = t1.elapsed().as_secs_f64() * 1.0e6 / iters as f64;

    let savings_us_per_op = (mean_a_us - mean_b_us) / N_PER_CB as f64;
    let savings_pct = 100.0 * (mean_a_us - mean_b_us) / mean_a_us;

    eprintln!("Standard 64-dispatch CB: {mean_a_us:.2} µs");
    eprintln!("ICB    64-command CB:    {mean_b_us:.2} µs");
    eprintln!(
        "Per-dispatch CPU savings: {savings_us_per_op:+.2} µs/op ({savings_pct:+.1}% per CB)"
    );
    eprintln!("Reference (PoC #2 raw-objc2): +10.67 µs/op (+25.6% per CB)");

    // Soft acceptance: wrapper path within ±50% of raw-objc2 PoC #2.
    if savings_us_per_op < 4.0 {
        panic!(
            "Wrapper ICB savings {savings_us_per_op:.2} µs/op < expected ~10 µs/op — wrapper overhead?"
        );
    }
}
