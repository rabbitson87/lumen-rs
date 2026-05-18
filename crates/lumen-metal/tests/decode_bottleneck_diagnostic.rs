//! decompose per-token decode time into:
//!
//!   A) CPU encoding time     (setBuffer + setBytes + dispatchThreadgroups, no commit)
//!   B) GPU execution time    (commit + wait, single CB with N dispatches)
//!   C) Per-call sync time    (the wall-clock surplus of synchronize-per-call)
//!
//! Question this answers: where does the 67 ms/token actually go on M3 Max?
//! If A is small and B is large → GPU bound (memory BW or compute), ICB
//! can't help. If A is large vs B → CPU dispatch overhead, ICB helps. If
//! C is large → synchronization barriers dominate.
//!
//! Configuration mirrors one MLP block of mlx-community/Qwen3.6-27B-4bit
//! (gate_up + silu*mul + down) repeated N times in one CB. Scaling N
//! reveals which of A/B/C grows.
//!
//! Shape source (verified from config.json):
//!   text_config.hidden_size = 5120
//!   text_config.intermediate_size = 17408
//!   text_config.head_dim = 256
//!   text_config.layer_types = [linear_attention × 3, full_attention × 1] × 16
//!   (i.e. 48 lin + 16 ful — model is HYBRID, not pure Dense)
//!
//! Run:
//!   cargo test --test decode_bottleneck_diagnostic -p lumen-metal \
//!     --features model-integration --release -- --nocapture --test-threads=1

#![cfg(feature = "model-integration")]

use candle_core::{DType, Device, Tensor};
use std::sync::Arc;
use std::time::Instant;
use lumen_metal::affine4_gpu::{Affine4Context, Affine4Weight};
use lumen_metal::affine4_linear::Affine4Linear;
use lumen_metal::silu_mul::SiluMulBf16InBf16Out;

const HIDDEN: usize = 5120;
const INTER: usize = 17408;

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

#[test]
fn decode_bottleneck_decomposition() {
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
        Err(_) => {
            eprintln!("[skip] cannot init silu kernel");
            return;
        }
    };

    // Build fixture weights matching 27B Dense MLP shapes.
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

    let x_data = synth_x(HIDDEN, 0xAAAA_BBBB);
    let x = Tensor::from_vec(x_data, &[1, 1, HIDDEN], &dev).unwrap()
        .to_dtype(DType::BF16).unwrap()
        .contiguous().unwrap();
    let r_data = synth_x(HIDDEN, 0x9999_8888);
    let residual = Tensor::from_vec(r_data, &[1, 1, HIDDEN], &dev).unwrap()
        .to_dtype(DType::BF16).unwrap()
        .contiguous().unwrap();

    // ── Single MLP block forward closure (matches DenseMlp standard path) ─
    let run_one = || {
        let combined = gate_up_lin.forward_bf16_in_bf16_out(&x).unwrap();
        let hidden = silu_kernel.forward(&combined).unwrap();
        down_lin.forward_with_residual_bf16_in_bf16_out(&hidden, &residual).unwrap()
    };

    use candle_core::backend::BackendDevice;
    let metal_dev = match x.device() {
        Device::Metal(m) => m,
        _ => unreachable!(),
    };

    // Warmup.
    for _ in 0..30 { let _ = run_one(); }
    let _ = metal_dev.synchronize();

    // ── Measurement A: CPU encode time per MLP block (no GPU execution) ─
    // We can't easily isolate "CPU encode only" because Candle's encoder
    // commits implicitly. Approximation: measure total wall time WITHOUT
    // synchronize. The function returns once CPU has handed work to GPU.
    // GPU work continues asynchronously.
    const ITERS: usize = 100;
    let t0 = Instant::now();
    for _ in 0..ITERS { let _ = run_one(); }
    let cpu_handoff_total = t0.elapsed().as_secs_f64() * 1e6;
    let cpu_handoff_per = cpu_handoff_total / ITERS as f64;
    // After this loop, GPU still has pending work. Drain it.
    let drain_t = Instant::now();
    let _ = metal_dev.synchronize();
    let drain_us = drain_t.elapsed().as_secs_f64() * 1e6;

    // ── Measurement B: synchronize-per-call (current production behavior) ─
    let t1 = Instant::now();
    for _ in 0..ITERS {
        let y = run_one();
        if let Device::Metal(md) = y.device() { let _ = md.synchronize(); }
    }
    let sync_per_call_total = t1.elapsed().as_secs_f64() * 1e6;
    let sync_per_call_per = sync_per_call_total / ITERS as f64;

    // ── Measurement C: bulk synchronize (encode N then sync once) ───────
    // This isolates pure GPU execution + queue throughput from per-call
    // sync overhead.
    let t2 = Instant::now();
    for _ in 0..ITERS { let _ = run_one(); }
    let _ = metal_dev.synchronize();
    let bulk_sync_total = t2.elapsed().as_secs_f64() * 1e6;
    let bulk_sync_per = bulk_sync_total / ITERS as f64;

    // ── Measurement D: long-run pipelined throughput ────────────────────
    // Fire many forwards back to back, sync once. Per-iteration time is
    // dominated by GPU steady-state throughput.
    const PIPELINE_ITERS: usize = 1000;
    let t3 = Instant::now();
    for _ in 0..PIPELINE_ITERS { let _ = run_one(); }
    let _ = metal_dev.synchronize();
    let pipeline_total = t3.elapsed().as_secs_f64() * 1e6;
    let pipeline_per = pipeline_total / PIPELINE_ITERS as f64;

    eprintln!();
    eprintln!("=== Decode bottleneck decomposition (per MLP block) ===");
    eprintln!("Shape: hidden={HIDDEN}, inter={INTER}, batch=1");
    eprintln!();
    eprintln!("A) CPU handoff total  ({ITERS} forwards, no per-call sync): {cpu_handoff_per:.1} µs/forward (queue handoff)");
    eprintln!("   (drain wait after the loop: {drain_us:.0} µs total)");
    eprintln!();
    eprintln!("B) Per-call synchronize (current measurement):              {sync_per_call_per:.1} µs/forward (CPU+GPU+sync barrier)");
    eprintln!();
    eprintln!("C) Bulk synchronize ({ITERS} forwards, 1 final sync):       {bulk_sync_per:.1} µs/forward (CPU+GPU pipelined)");
    eprintln!();
    eprintln!("D) Pipelined throughput ({PIPELINE_ITERS} forwards, 1 final sync): {pipeline_per:.1} µs/forward (steady-state throughput)");
    eprintln!();
    eprintln!("Synchronization overhead (B - C):  {:.1} µs/forward (per-call sync waste)", sync_per_call_per - bulk_sync_per);
    eprintln!("CPU-vs-GPU overlap (B - A):        {:.1} µs/forward (GPU work hidden behind CPU)", sync_per_call_per - cpu_handoff_per);
    eprintln!();
    eprintln!("Interpretation:");
    eprintln!("  - If D ≈ B: GPU dominates total time, CPU dispatch is fully overlapped.");
    eprintln!("    → Optimizing CPU encoding (ICB) gives near-zero gain.");
    eprintln!("  - If D << B: per-call sync barriers force CPU/GPU serialization.");
    eprintln!("    → Removing per-call sync (model decode usually has no per-call sync)");
    eprintln!("    → recovers the gap. ICB then helps for the remaining CPU portion.");
    eprintln!("  - If A ≈ D: CPU is the bottleneck even in pipelined mode.");
    eprintln!("    → ICB is the right lever (but likely it isn't on M3 Max).");

    // ── BW utilization of the MLP block ──────────────────────────────
    // Total weight bytes read per forward:
    //   gate_up: out_features × in_features × 0.5 bytes (4-bit)
    //          + (out_features × in_features / group=64) × (2+2) bytes (bf16 scale + bias)
    //   down  : same formula, smaller shape.
    let gate_up_packed_bytes = (2 * INTER * HIDDEN) / 2;
    let gate_up_meta_bytes = (2 * INTER * HIDDEN / 64) * 4; // bf16 scale + bf16 bias
    let down_packed_bytes = (HIDDEN * INTER) / 2;
    let down_meta_bytes = (HIDDEN * INTER / 64) * 4;
    let total_read_bytes = gate_up_packed_bytes + gate_up_meta_bytes
        + down_packed_bytes + down_meta_bytes;
    let bw_gbs = (total_read_bytes as f64) / (pipeline_per * 1e-6) / 1e9;
    let m3_max_peak_bw_gbs = 400.0;
    let bw_util_pct = bw_gbs / m3_max_peak_bw_gbs * 100.0;
    eprintln!();
    eprintln!("=== BW analysis (steady-state pipelined throughput) ===");
    eprintln!("Bytes read per MLP forward: {:.1} MB", total_read_bytes as f64 / 1e6);
    eprintln!("Effective BW (D):          {bw_gbs:.1} GB/s");
    eprintln!("M3 Max peak BW:            {m3_max_peak_bw_gbs:.0} GB/s");
    eprintln!("BW utilization:            {bw_util_pct:.1}%");
}

// (Pipelined ICB-vs-Standard test moved to lumen-model crate to access
// `DenseMlp` without circular dependency: see
// `crates/lumen-model/tests/mlp_icb_pipelined_microbench.rs`.)
