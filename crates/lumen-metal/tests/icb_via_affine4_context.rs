//! lumen-rs API path for ICB.
//!
//! Replicates PoC #2 (single-weight 64-dispatch ICB on
//! `affine4_qmv_fast_bf16in_bf16out_residual`) but exclusively through
//! `Affine4Context::record_qmv_fast_bf16in_bf16out_residual_icb` —
//! i.e., the public API that production code paths will use.
//!
//! Validates that the entire lumen-rs ICB stack (Affine4Context →
//! IndirectCommandBuffer → encoder.execute_commands_in_buffer) produces
//! bit-identical output and matches the raw-objc2 PoC #2 savings.

#![allow(unexpected_cfgs)]

use lumen_metal::metal::CommandBufferExt;
use std::time::Instant;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice, MTLResourceUsage};
use lumen_metal::affine4_gpu::{Affine4Context, Affine4Weight};
use lumen_metal::metal::{Buffer, IndirectCommandBuffer};

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
    (0..n).map(|_| { s = s.wrapping_mul(1103515245).wrapping_add(12345); s }).collect()
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
fn affine4_context_icb_record_replay_residual() {
    let ctx = match Affine4Context::new() {
        Ok(c) => c,
        Err(_) => {
            eprintln!("[skip] no Metal device");
            return;
        }
    };

    let out: usize = 5120;
    let ins: usize = 5120;
    let batch: usize = 1;

    let packed = synth_packed(out, ins, 0xDEADBEEF);
    let scales = synth_scales_or_biases(out, ins, 0xCAFEBABE, false);
    let biases = synth_scales_or_biases(out, ins, 0x12345678, true);

    let weight = Affine4Weight::from_host(&ctx.ctx, &packed, &scales, &biases, out, ins)
        .expect("weight upload");

    let x_bf16 = synth_x_bf16(batch * ins, 0xFADEFADE);
    let residual_bf16 = synth_x_bf16(batch * out, 0xBEEFBEEF);

    let x_buf = ctx.ctx.buffer_with_data(&x_bf16);
    let residual_buf = ctx.ctx.buffer_with_data(&residual_bf16);
    let y_buf_a = ctx.ctx.buffer_zeroed((batch * out * 2) as u64);
    let y_buf_b = ctx.ctx.buffer_zeroed((batch * out * 2) as u64);

    let dims = Affine4Dims { out_features: out as u32, in_features: ins as u32 };
    let dims_buf = ctx.ctx.buffer_with_data(&[dims]);
    let batch_buf = ctx.ctx.buffer_with_data(&[batch as u32]);

    // Build ICB via Candle wrapper.
    let raw_device: Retained<ProtocolObject<dyn MTLDevice>> =
        Retained::from(ctx.ctx.device.as_ref());
    // 8 buffers for residual variant (packed/scales/biases/x/residual/y/dims/batch).
    let icb = IndirectCommandBuffer::new(&raw_device, N_PER_CB, 8)
        .expect("ICB alloc");

    for slot in 0..N_PER_CB {
        ctx.record_qmv_fast_bf16in_bf16out_residual_icb(
            &icb,
            slot,
            &weight,
            &x_buf,
            0,
            &residual_buf,
            0,
            &y_buf_b,
            0,
            &dims_buf,
            &batch_buf,
            batch,
        );
    }

    // ── Bench helpers ──────────────────────────────────────────────────
    let run_standard = || {
        let cmd = lumen_metal::metal::new_command_buffer(&ctx.ctx.queue);
        let enc = cmd.auto_compute_encoder();
        for _ in 0..N_PER_CB {
            ctx.encode_qmv_fast_bf16in_bf16out_residual_dispatch(
                &enc,
                &weight,
                &x_buf,
                0,
                &residual_buf,
                0,
                &y_buf_a,
                0,
                batch,
            );
        }
        drop(enc);
        cmd.commit();
        cmd.wait_until_completed();
    };

    let usage =
        MTLResourceUsage(MTLResourceUsage::Read.0 | MTLResourceUsage::Write.0);
    let run_icb = || {
        let cmd = lumen_metal::metal::new_command_buffer(&ctx.ctx.queue);
        let enc = cmd.auto_compute_encoder();
        // Production ICB integration would track the buffer set per layer;
        // here all commands share the same buffers so 1× useResource each.
        let (wp, ws, wb) = weight.buffers();
        for buf in [wp, ws, wb] as [&Buffer; 3] {
            enc.use_buffer_for_icb(buf, usage);
        }
        enc.use_buffer_for_icb(&x_buf, usage);
        enc.use_buffer_for_icb(&residual_buf, usage);
        enc.use_buffer_for_icb(&y_buf_b, usage);
        enc.use_buffer_for_icb(&dims_buf, usage);
        enc.use_buffer_for_icb(&batch_buf, usage);
        enc.execute_commands_in_buffer(&icb, N_PER_CB);
        drop(enc);
        cmd.commit();
        cmd.wait_until_completed();
    };

    // ── Bit-identity ───────────────────────────────────────────────────
    run_standard();
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
    eprintln!("=== Phase 17.B lumen API ICB test (residual variant) ===");
    eprintln!("Bit-identical (A vs B): {} / {} (diffs={})",
        out * batch - diffs, out * batch, diffs);
    if diffs > 0 {
        panic!("lumen API ICB output diverged from standard");
    }

    // ── Bench ──────────────────────────────────────────────────────────
    let warmup = 16;
    for _ in 0..warmup {
        run_standard();
        run_icb();
    }
    let iters = 100;

    let t0 = Instant::now();
    for _ in 0..iters {
        run_standard();
    }
    let mean_a_us = t0.elapsed().as_secs_f64() * 1.0e6 / iters as f64;

    let t1 = Instant::now();
    for _ in 0..iters {
        run_icb();
    }
    let mean_b_us = t1.elapsed().as_secs_f64() * 1.0e6 / iters as f64;

    let savings_us_per_op = (mean_a_us - mean_b_us) / N_PER_CB as f64;
    let savings_pct = 100.0 * (mean_a_us - mean_b_us) / mean_a_us;
    let projected_per_token_ms = savings_us_per_op * 960.0 / 1000.0;
    let new_ms = (67.0 - projected_per_token_ms).max(1.0);
    let new_tps = 1000.0 / new_ms;

    eprintln!("Standard 64-dispatch CB: {mean_a_us:.2} µs");
    eprintln!("ICB    64-command CB:    {mean_b_us:.2} µs");
    eprintln!(
        "Per-dispatch CPU savings: {savings_us_per_op:+.2} µs/op ({savings_pct:+.1}% per CB)"
    );
    eprintln!(
        "Projected 27B Dense decode: 67 ms → {new_ms:.2} ms = {new_tps:.2} tok/s"
    );

    if savings_us_per_op < 4.0 {
        panic!(
            "lumen API ICB savings {savings_us_per_op:.2} µs/op < expected ~10 µs/op"
        );
    }
}
