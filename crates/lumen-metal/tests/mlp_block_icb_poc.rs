//! per-MLP-block ICB at N=3.
//!
//! **Decision gate** for the Phase 17.D-2/-3/-4 escalation. Measures σ
//! between:
//!   - Standard path: gate_up + silu*mul (separate dispatches) + down_proj_residual
//!     (5-7 dispatches in flight depending on Candle's narrow/contiguous fusion)
//!   - ICB path: ONE `executeCommandsInBuffer(0..3)` covering all three
//!     fused into a single ICB.
//!
//! PoC #1 N curve outlook: N=3 expected ~+3-5%. If σ ≥ +2 here → continue
//! to 17.D-2 (RoPE + KV write ICB-compat) toward N=10. If σ ≤ 0 → ICB
//! granularity gain saturates earlier than projected; redesign needed.
//!
//! Run:
//!   cargo test --test mlp_block_icb_poc -p lumen-metal \
//!     --features model-integration --release -- --nocapture --test-threads=1

#![cfg(feature = "model-integration")]

use lumen_metal::metal::CommandBufferExt;
use candle_core::backend::BackendDevice;
use candle_core::{DType, Device, Tensor};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice, MTLResourceUsage};
use std::sync::Arc;
use std::time::Instant;
use lumen_metal::affine4_gpu::{Affine4Context, Affine4Weight};
use lumen_metal::affine4_linear::Affine4Linear;
use lumen_metal::metal::{Buffer, IndirectCommandBuffer};
use lumen_metal::silu_mul::SiluMulBf16InBf16Out;

const HIDDEN: usize = 5120;
// Shape source: mlx-community/Qwen3.6-27B-4bit config.json text_config.intermediate_size
// (was 25600 — fictional, see anti-pattern #25). Architecture is HYBRID (48 lin + 16 ful).
const INTER: usize = 17408;
const ITERS: usize = 100;
const WARMUP: usize = 30;

fn synth_packed(out: usize, ins: usize, seed: u32) -> Vec<u32> {
    let n = out * ins / 8;
    let mut s = seed;
    (0..n).map(|_| { s = s.wrapping_mul(1103515245).wrapping_add(12345); s }).collect()
}

fn synth_scales_or_biases(out: usize, ins: usize, seed: u32, neg: bool) -> Vec<u16> {
    let n = out * ins / 64;
    let mut s = seed;
    let off = if neg { -0.005 } else { 0.01 };
    (0..n).map(|_| {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        let f = ((s >> 8) & 0xff) as f32 / 256.0 * 0.01 + off;
        (f.to_bits() >> 16) as u16
    }).collect()
}

fn synth_x(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        ((s >> 8) & 0xff) as f32 / 256.0 - 0.5
    }).collect()
}

fn welchs_t(a: &[f64], b: &[f64]) -> f64 {
    let na = a.len() as f64;
    let nb = b.len() as f64;
    let ma = a.iter().sum::<f64>() / na;
    let mb = b.iter().sum::<f64>() / nb;
    let va = a.iter().map(|x| (x - ma).powi(2)).sum::<f64>() / (na - 1.0);
    let vb = b.iter().map(|x| (x - mb).powi(2)).sum::<f64>() / (nb - 1.0);
    let se = (va / na + vb / nb).sqrt();
    if se == 0.0 { 0.0 } else { (ma - mb) / se }
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|x, y| x.partial_cmp(y).unwrap());
    s[s.len() / 2]
}

#[test]
fn mlp_block_icb_poc_n3() {
    let dev = match Device::new_metal(0) {
        Ok(d) => d,
        Err(_) => {
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
    let silu_kernel = match SiluMulBf16InBf16Out::new() {
        Ok(k) => k,
        Err(e) => {
            eprintln!("[skip] cannot init SiluMulBf16InBf16Out: {e}");
            return;
        }
    };

    // Build fixture weights:
    //   gate_up_proj : [2*INTER, HIDDEN]
    //   down_proj    : [HIDDEN,  INTER]
    let gate_up_packed = synth_packed(2 * INTER, HIDDEN, 0xDEAD_BEEF);
    let gate_up_scales = synth_scales_or_biases(2 * INTER, HIDDEN, 0xCAFE_BABE, false);
    let gate_up_biases = synth_scales_or_biases(2 * INTER, HIDDEN, 0x1234_5678, true);
    let gate_up_w = Affine4Weight::from_host(
        &ctx.ctx, &gate_up_packed, &gate_up_scales, &gate_up_biases,
        2 * INTER, HIDDEN,
    ).expect("gate_up weight");

    let down_packed = synth_packed(HIDDEN, INTER, 0xFADE_FADE);
    let down_scales = synth_scales_or_biases(HIDDEN, INTER, 0xBEEF_BEEF, false);
    let down_biases = synth_scales_or_biases(HIDDEN, INTER, 0xC0DE_C0DE, true);
    let down_w = Affine4Weight::from_host(
        &ctx.ctx, &down_packed, &down_scales, &down_biases,
        HIDDEN, INTER,
    ).expect("down weight");

    let gate_up_lin = Affine4Linear::new(gate_up_w, None, ctx.clone());
    let down_lin = Affine4Linear::new(down_w, None, ctx.clone());

    // Inputs.
    let x_data = synth_x(HIDDEN, 0xAAAA_BBBB);
    let x = Tensor::from_vec(x_data, &[1, 1, HIDDEN], &dev).unwrap()
        .to_dtype(DType::BF16).unwrap()
        .contiguous().unwrap();
    let residual_data = synth_x(HIDDEN, 0x9999_8888);
    let residual = Tensor::from_vec(residual_data, &[1, 1, HIDDEN], &dev).unwrap()
        .to_dtype(DType::BF16).unwrap()
        .contiguous().unwrap();

    // ── Standard path closure (5-dispatch silu*mul chain) ─────────────
    // Mirrors `DenseMlp::forward_with_residual_bf16_in_bf16_out`.
    let run_standard = || -> Tensor {
        let combined = gate_up_lin.forward_bf16_in_bf16_out(&x).unwrap();
        let combined_f32 = combined.to_dtype(DType::F32).unwrap();
        let last = combined_f32.dims().len() - 1;
        let gate = combined_f32.narrow(last, 0, INTER).unwrap().contiguous().unwrap();
        let up = combined_f32.narrow(last, INTER, INTER).unwrap().contiguous().unwrap();
        let hidden_f32 = (candle_nn::ops::silu(&gate).unwrap() * up).unwrap();
        down_lin.forward_with_residual_bf16_in_bf16_out(&hidden_f32, &residual).unwrap()
    };

    // ── Fused-chain closure (no ICB, only the dispatch-count reduction) ─
    // Same 3 logical operations as the ICB path but via standard Candle
    // dispatches. Disambiguates whether the +13% gain comes from ICB
    // amortization or purely from cutting 8 dispatches → 3.
    let run_fused_no_icb = || -> Tensor {
        let combined = gate_up_lin.forward_bf16_in_bf16_out(&x).unwrap();
        let hidden = silu_kernel.forward(&combined).unwrap();
        down_lin.forward_with_residual_bf16_in_bf16_out(&hidden, &residual).unwrap()
    };

    // ── ICB path setup ────────────────────────────────────────────────
    // 3-slot ICB containing: gate_up_proj | silu*mul | down_proj_residual.
    // Persistent intermediate buffers (combined, hidden) — allocated via
    // synchronous `buffer_zeroed` to avoid the Candle queue race observed
    // in 17.D-1c.
    let metal_dev = match x.device() {
        Device::Metal(m) => m,
        _ => unreachable!(),
    };
    let raw_device: Retained<ProtocolObject<dyn MTLDevice>> =
        Retained::from(metal_dev.metal_device().as_ref());
    let icb = IndirectCommandBuffer::new(&raw_device, 3, 8).expect("ICB alloc");

    let combined_buf: Buffer = ctx.ctx.buffer_zeroed((2 * INTER * 2) as u64);
    let hidden_buf: Buffer = ctx.ctx.buffer_zeroed((INTER * 2) as u64);

    // Constant buffers for each slot.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Affine4Dims { out_features: u32, in_features: u32 }
    let gate_up_dims = ctx.ctx.buffer_with_data(&[Affine4Dims {
        out_features: (2 * INTER) as u32,
        in_features: HIDDEN as u32,
    }]);
    let down_dims = ctx.ctx.buffer_with_data(&[Affine4Dims {
        out_features: HIDDEN as u32,
        in_features: INTER as u32,
    }]);
    let batch_buf = ctx.ctx.buffer_with_data(&[1u32]);
    let silu_dims = silu_kernel.make_dims_buf(INTER);

    // Final output buffer (residual ICB target). Allocated once.
    let y_buf: Buffer = ctx.ctx.buffer_zeroed((HIDDEN * 2) as u64);

    // Extract raw input buffers from Candle Tensors (one-time at setup).
    let extract = |t: &Tensor| -> (Buffer, u64) {
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
    let (r_buf, r_off) = extract(&residual);

    // Slot 0: gate_up_proj qmv_fast
    ctx.record_qmv_fast_bf16in_bf16out_icb(
        &icb, 0, gate_up_lin.weight(),
        &x_buf, x_off,
        &combined_buf, 0,
        &gate_up_dims, &batch_buf, 1,
    );
    // Slot 1: silu*mul
    silu_kernel.record_icb(
        &icb, 1, &combined_buf, 0, &hidden_buf, 0, &silu_dims, 1, INTER,
    );
    // Slot 2: down_proj qmv_fast bf16-residual
    ctx.record_qmv_fast_bf16in_bf16out_residual_icb(
        &icb, 2, down_lin.weight(),
        &hidden_buf, 0,
        &r_buf, r_off,
        &y_buf, 0,
        &down_dims, &batch_buf, 1,
    );

    // ICB run closure (committed via Candle's queue + drain protocol).
    //
    // NB: a single `executeCommandsInBuffer(0..3)` was tried first; even
    // though our commands have data dependencies (slot 0 writes
    // combined_buf which slot 1 reads, etc.) `MTLIndirectCommandType::
    // ConcurrentDispatch` does NOT insert implicit barriers between
    // commands inside one execute. We therefore split into 3 separate
    // execute calls — within one encoder, sequential `executeCommandsInBuffer`
    // calls are serialized via the encoder's built-in hazard tracking.
    let run_icb = || {
        let _ = metal_dev.synchronize();
        let cmd = lumen_metal::metal::new_command_buffer(
            &metal_dev.command_queue().unwrap()
        );
        let enc = cmd.auto_compute_encoder();
        let usage = MTLResourceUsage(MTLResourceUsage::Read.0 | MTLResourceUsage::Write.0);
        let (gp, gs, gb) = gate_up_lin.weight().buffers();
        let (dp, ds, db) = down_lin.weight().buffers();
        enc.use_buffers_for_icb(
            &[
                gp, gs, gb,                     // gate_up weights
                &x_buf,                          // gate_up input
                &combined_buf,                   // gate_up output / silu*mul input
                &silu_dims,                      // silu*mul dims
                &hidden_buf,                     // silu*mul output / down input
                dp, ds, db,                     // down weights
                &r_buf,                          // down residual
                &y_buf,                          // down output
                &gate_up_dims, &down_dims, &batch_buf,
            ],
            usage,
        );
        // Serialized executes — each call boundary is an implicit barrier.
        // Slot 0 (gate_up) finishes before slot 1 (silu*mul) starts, etc.
        enc.execute_commands_in_buffer_range(&icb, 0, 1);
        enc.execute_commands_in_buffer_range(&icb, 1, 1);
        enc.execute_commands_in_buffer_range(&icb, 2, 1);
        drop(enc);
        cmd.commit();
        cmd.wait_until_completed();
    };

    // ── Bit-identity guard ────────────────────────────────────────────
    // First, run standard once to capture ref output. Then run ICB once;
    // y_buf should match the standard-path's output Tensor bit-for-bit
    // (kernel-level parity already proven; this checks chain integrity).
    let y_ref = run_standard();
    run_icb();

    let ref_bits: Vec<u32> = y_ref.flatten_all().unwrap()
        .to_dtype(DType::F32).unwrap()
        .to_vec1::<f32>().unwrap()
        .iter().map(|f| f.to_bits()).collect();
    // Read y_buf back via raw pointer.
    let y_ptr = y_buf.contents() as *const u16;
    let mut icb_bits: Vec<u32> = Vec::with_capacity(HIDDEN);
    for i in 0..HIDDEN {
        let bf16_bits = unsafe { *y_ptr.add(i) };
        let f32_bits = (bf16_bits as u32) << 16;
        icb_bits.push(f32_bits);
    }

    let diffs = ref_bits.iter().zip(icb_bits.iter()).filter(|(a, b)| a != b).count();
    eprintln!();
    eprintln!("=== bit-identity gate (standard chain vs 3-cmd ICB) ===");
    eprintln!("Compared:  {HIDDEN} elements");
    eprintln!("Diffs:     {diffs} {}", if diffs == 0 { "✓" } else { "✗" });
    if diffs > 0 {
        // Don't panic — surface the data first; downstream measurement
        // still informs the σ direction even if numerics drift.
        eprintln!("WARN: ICB chain output differs — production wiring would need");
        eprintln!("      to investigate before flipping default ON.");
    }

    // ── Bench (3-way A/B/C interleaved) ───────────────────────────────
    for _ in 0..WARMUP { run_standard(); }
    if let Device::Metal(md) = &dev { let _ = md.synchronize(); }
    for _ in 0..WARMUP { run_icb(); }
    if let Device::Metal(md) = &dev { let _ = md.synchronize(); }
    for _ in 0..WARMUP { run_fused_no_icb(); }
    if let Device::Metal(md) = &dev { let _ = md.synchronize(); }

    let mut t_std: Vec<f64> = Vec::with_capacity(ITERS);
    let mut t_fused: Vec<f64> = Vec::with_capacity(ITERS);
    let mut t_icb: Vec<f64> = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t0 = Instant::now();
        let y = run_standard();
        if let Device::Metal(md) = y.device() { let _ = md.synchronize(); }
        t_std.push(t0.elapsed().as_secs_f64() * 1e6);

        let t1 = Instant::now();
        let y = run_fused_no_icb();
        if let Device::Metal(md) = y.device() { let _ = md.synchronize(); }
        t_fused.push(t1.elapsed().as_secs_f64() * 1e6);

        let t2 = Instant::now();
        run_icb();
        t_icb.push(t2.elapsed().as_secs_f64() * 1e6);
    }

    let mean_std = t_std.iter().sum::<f64>() / ITERS as f64;
    let mean_fused = t_fused.iter().sum::<f64>() / ITERS as f64;
    let mean_icb = t_icb.iter().sum::<f64>() / ITERS as f64;
    let med_std = median(&t_std);
    let med_fused = median(&t_fused);
    let med_icb = median(&t_icb);
    // Sign convention: positive σ = second arg faster than first arg.
    let sigma_std_vs_icb = welchs_t(&t_std, &t_icb);
    let sigma_std_vs_fused = welchs_t(&t_std, &t_fused);
    let sigma_fused_vs_icb = welchs_t(&t_fused, &t_icb);
    let pct_icb = (med_icb - med_std) / med_std * 100.0;
    let pct_fused = (med_fused - med_std) / med_std * 100.0;
    let pct_icb_over_fused = (med_icb - med_fused) / med_fused * 100.0;

    eprintln!();
    eprintln!("=== Phase 17.D-1e — 3-way A/B/C: standard vs fused-no-ICB vs ICB-N=3 ===");
    eprintln!("Iterations:                {ITERS} per variant ({WARMUP} warmup)");
    eprintln!("X) Standard 8-dispatch:    µ {mean_std:.0} / med {med_std:.0} µs");
    eprintln!("Z) Fused-no-ICB 3-disp:    µ {mean_fused:.0} / med {med_fused:.0} µs");
    eprintln!("Y) ICB N=3 (serialized):   µ {mean_icb:.0} / med {med_icb:.0} µs");
    eprintln!();
    eprintln!("Δ med vs standard:");
    eprintln!("  Z (fused, no ICB) :  {pct_fused:+.2}%   σ {sigma_std_vs_fused:+.2}");
    eprintln!("  Y (ICB N=3)       :  {pct_icb:+.2}%   σ {sigma_std_vs_icb:+.2}");
    eprintln!();
    eprintln!("Δ med Y vs Z (ICB's own contribution beyond fusion):");
    eprintln!("  {pct_icb_over_fused:+.2}%   σ_(fused→icb) {sigma_fused_vs_icb:+.2}");
    eprintln!();
    eprintln!("Disambiguation gate:");
    eprintln!("  |Z - Y| < 2%  AND  |σ_(fused→icb)| < 1.5  → ICB itself ≈ 0 contribution");
    eprintln!("                                              ALL gain = dispatch reduction");
    eprintln!("  Y - Z ≤ -2%   AND  σ_(fused→icb) ≥ +2     → ICB adds real value beyond fusion");
    eprintln!("  Y - Z ≥ +2%   AND  σ_(fused→icb) ≤ -2     → ICB regresses vs plain fused");
}
