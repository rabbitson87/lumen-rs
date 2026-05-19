//! Affine3 (3-bit weight quant) BW utilization test.
//!
//! Compares pipelined µs/call of:
//!   - Affine3 matvec_bf16in_bf16out (1-thread/row, bit-plane decode)
//!   - Affine4 matvec_f32 reference (1-thread/row, nibble decode)  ← apples-to-apples topology
//!
//! Both are simplest-topology kernels (no threadgroup memory, no qmv_fast
//! cooperative reduction). The comparison isolates the per-element decode
//! compute cost difference (Affine3 ~9 ops/elem vs Affine4 ~4 ops/elem).
//!
//! Decision rule (from phase18a_affine3_design.md):
//!   - saving ≥ 15%: full Affine3 implementation green-lit
//!   - saving 5-15%: redesign packing (try Option B / triplet) and re-run
//!   - saving < 5%: 3-bit lever NEGATIVE, abandon → Phase 18.B only
//!
//! NOTE: this POC uses the simplest matvec topology (1-thread/row), NOT
//! qmv_fast. Production projection (+17.8%) was based on qmv_fast topology
//! which has different compute/BW balance. POC.0 derisks the dequant cost
//! direction; POC.1 (qmv_fast topology) confirms production projection.
//!
//! Run:
//!   cargo test --test affine3_poc_bw -p lumen-metal --release \
//!     -- --nocapture --test-threads=1

use lumen_metal::affine3_gpu::{AFFINE3_GROUP_SIZE, Affine3Context, Affine3Weight};
use lumen_metal::affine4_gpu::{Affine4Context, Affine4Weight};
use std::time::Instant;

const ITERS: usize = 1000;
const WARMUP: usize = 50;
const M3_MAX_PEAK_BW_GBS: f64 = 400.0;

fn bf16_from_f32(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

fn synth_3bit_codes(n: usize, seed: u32) -> Vec<u8> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 8) & 0x7) as u8 // [0..7]
        })
        .collect()
}

fn synth_4bit_packed(out: usize, ins: usize, seed: u32) -> Vec<u32> {
    let n = out * ins / 8;
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            s
        })
        .collect()
}

fn synth_meta(groups: usize, seed: u32, neg: bool) -> Vec<u16> {
    let mut s = seed;
    let off = if neg { -0.005 } else { 0.01 };
    (0..groups)
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
            bf16_from_f32(f)
        })
        .collect()
}

#[test]
fn affine3_poc_gate_up() {
    // Gate_up shape: out=34816, in=5120 — biggest single matmul cost (23 ms/token).
    const OUT: usize = 34816;
    const IN: usize = 5120;

    let ctx3 = match Affine3Context::new() {
        Ok(c) => c,
        Err(_) => {
            eprintln!("[skip] no Affine3 context");
            return;
        }
    };
    let ctx4 = match Affine4Context::new() {
        Ok(c) => c,
        Err(_) => {
            eprintln!("[skip] no Affine4 context");
            return;
        }
    };

    eprintln!();
    eprintln!("=== Phase 18.A.0 POC — Affine3 vs Affine4 matvec (gate_up shape) ===");
    eprintln!("Shape: out={OUT}, in={IN}, batch=1");
    eprintln!("Iterations: {ITERS} per arm ({WARMUP} warmup, single drain at end)");
    eprintln!("Topology: 1-thread/row matvec (no qmv_fast — POC.0 derisks dequant cost)");
    eprintln!();

    // ── Build Affine3 weight ──────────────────────────────────────────────
    let groups = OUT * IN / AFFINE3_GROUP_SIZE;
    let codes = synth_3bit_codes(OUT * IN, 0xDEAD_BEEF);
    let scales3 = synth_meta(groups, 0xCAFE_BABE, false);
    let biases3 = synth_meta(groups, 0x1234_5678, true);
    let weight3 = Affine3Weight::from_codes_3bit(&ctx3.ctx, &codes, &scales3, &biases3, OUT, IN)
        .expect("affine3 weight");

    let x3 = synth_x_bf16(IN, 0xAAAA_BBBB);
    let x3_buf = ctx3.ctx.buffer_with_data(&x3);
    let y3_buf = ctx3.ctx.buffer_for::<u16>(OUT);

    // Warmup.
    for _ in 0..WARMUP {
        ctx3.matvec_bf16in_bf16out_pipelined(&weight3, &x3_buf, &y3_buf)
            .unwrap();
    }
    // Drain warmup.
    let drain = lumen_metal::metal::new_command_buffer(&ctx3.ctx.queue);
    drain.commit();
    drain.wait_until_completed();

    // Pipelined timing.
    let t0 = Instant::now();
    for _ in 0..ITERS {
        ctx3.matvec_bf16in_bf16out_pipelined(&weight3, &x3_buf, &y3_buf)
            .unwrap();
    }
    let drain = lumen_metal::metal::new_command_buffer(&ctx3.ctx.queue);
    drain.commit();
    drain.wait_until_completed();
    let aff3_us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    let aff3_packed_bytes = OUT * IN * 3 / 8;
    let aff3_meta_bytes = groups * 2 * 2; // 2 bf16 per group
    let aff3_act_bytes = IN * 2;
    let aff3_out_bytes = OUT * 2;
    let aff3_total_bytes = aff3_packed_bytes + aff3_meta_bytes + aff3_act_bytes + aff3_out_bytes;
    let aff3_bw_gbs = aff3_total_bytes as f64 / aff3_us / 1e3;
    let aff3_bw_pct = aff3_bw_gbs / M3_MAX_PEAK_BW_GBS * 100.0;

    eprintln!("Affine3 matvec_bf16in_bf16out:");
    eprintln!(
        "  packed: {:.2} MB, meta: {:.2} MB, total: {:.2} MB",
        aff3_packed_bytes as f64 / 1e6,
        aff3_meta_bytes as f64 / 1e6,
        aff3_total_bytes as f64 / 1e6
    );
    eprintln!(
        "  pipelined µs/call: {aff3_us:.1}, effective BW: {aff3_bw_gbs:.1} GB/s ({aff3_bw_pct:.1}% of {M3_MAX_PEAK_BW_GBS} GB/s)"
    );
    eprintln!();

    // ── Build Affine4 weight (same shape, comparable matvec_f32) ──────────
    // NOTE: affine4_matvec_f32 reads f32 activation, so this isn't strictly
    // bf16in/bf16out apples-to-apples. The closer comparison is to Affine4's
    // qmv_fast which is bf16in/bf16out. We use matvec_f32 here for topology
    // parity (both are 1-thread/row, no threadgroup memory). The packed-byte
    // comparison is still the meaningful axis.
    let packed4 = synth_4bit_packed(OUT, IN, 0xFADE_FADE);
    let scales4 = synth_meta(groups, 0xCAFE_BABE, false);
    let biases4 = synth_meta(groups, 0x1234_5678, true);
    let weight4 = Affine4Weight::from_host(&ctx4.ctx, &packed4, &scales4, &biases4, OUT, IN)
        .expect("affine4 weight");

    let x4_f32: Vec<f32> = (0..IN)
        .map(|i| ((i as u32).wrapping_mul(0x9E3779B9) as f32) / (u32::MAX as f32) - 0.5)
        .collect();

    // Warmup + bench using the existing Affine4Context::matvec_with_weight
    // (synchronous; for fair pipelined comparison we'd need a pipelined
    // variant — but matvec_with_weight is the simplest pre-existing fn).
    // We use sync here for both arms' BW upper bound; pipelined variants
    // would shave per-call sync barrier off both equally.
    for _ in 0..WARMUP {
        let _ = ctx4.matvec_with_weight(&weight4, &x4_f32).unwrap();
    }
    let t1 = Instant::now();
    for _ in 0..ITERS {
        let _ = ctx4.matvec_with_weight(&weight4, &x4_f32).unwrap();
    }
    let aff4_us = t1.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    let aff4_packed_bytes = OUT * IN / 2; // 0.5 B/elem
    let aff4_meta_bytes = groups * 2 * 2;
    let aff4_act_bytes = IN * 4; // f32 activation
    let aff4_out_bytes = OUT * 4; // f32 output
    let aff4_total_bytes = aff4_packed_bytes + aff4_meta_bytes + aff4_act_bytes + aff4_out_bytes;
    let aff4_bw_gbs = aff4_total_bytes as f64 / aff4_us / 1e3;
    let aff4_bw_pct = aff4_bw_gbs / M3_MAX_PEAK_BW_GBS * 100.0;

    eprintln!("Affine4 matvec_f32 (reference, f32 act/out):");
    eprintln!(
        "  packed: {:.2} MB, meta: {:.2} MB, total: {:.2} MB",
        aff4_packed_bytes as f64 / 1e6,
        aff4_meta_bytes as f64 / 1e6,
        aff4_total_bytes as f64 / 1e6
    );
    eprintln!(
        "  per-call (sync) µs: {aff4_us:.1}, effective BW: {aff4_bw_gbs:.1} GB/s ({aff4_bw_pct:.1}% of {M3_MAX_PEAK_BW_GBS} GB/s)"
    );
    eprintln!();

    // ── Comparison summary ────────────────────────────────────────────────
    eprintln!("=== Comparison ===");
    eprintln!(
        "Packed bytes:  Affine3 {:.1} MB vs Affine4 {:.1} MB  (Affine3 = {:.1}% of Affine4)",
        aff3_packed_bytes as f64 / 1e6,
        aff4_packed_bytes as f64 / 1e6,
        aff3_packed_bytes as f64 / aff4_packed_bytes as f64 * 100.0
    );
    eprintln!(
        "Pipelined µs:  Affine3 {aff3_us:.1} vs Affine4 {aff4_us:.1}  (Affine3 = {:.1}% of Affine4)",
        aff3_us / aff4_us * 100.0
    );

    // POC decision logic (saving = how much faster Affine3 is vs Affine4)
    let saving_pct = (aff4_us - aff3_us) / aff4_us * 100.0;
    eprintln!("Saving (Affine3 vs Affine4): {saving_pct:+.1}%   (positive = Affine3 faster)");
    eprintln!();
    eprintln!("=== POC.0 Decision ===");
    if saving_pct >= 15.0 {
        eprintln!(
            "✅ Saving ≥ 15% — full Affine3 implementation green-lit. Proceed to POC.1 (qmv_fast topology)."
        );
    } else if saving_pct >= 5.0 {
        eprintln!(
            "⚠️  Saving 5-15% — packing redesign candidate. Try Option B (triplet ushort) or qmv_fast topology directly."
        );
    } else if saving_pct >= -5.0 {
        eprintln!(
            "❌ Saving ≈ 0% — dequant compute exhausts BW headroom. Lever 보류, focus on Phase 18.B (KV compression)."
        );
    } else {
        eprintln!(
            "❌ NEGATIVE — Affine3 slower than Affine4. Definitively not a viable lever for this hardware. Skip to 18.B."
        );
    }
    eprintln!();
    eprintln!("Caveat: this POC uses the SIMPLEST topology (1-thread/row matvec).");
    eprintln!(
        "Production qmv_fast (NSG=2 RPS=4 VPT=16) has different compute/BW balance — POC.1 needed for production projection."
    );
}

#[test]
fn affine3_poc1_qmv_fast_gate_up() {
    // Same shape as POC.0 — gate_up: out=34816, in=5120.
    const OUT: usize = 34816;
    const IN: usize = 5120;
    const BATCH: usize = 1;

    let ctx3 = match Affine3Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[skip] no Affine3 context: {e}");
            return;
        }
    };
    let ctx4 = match Affine4Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[skip] no Affine4 context: {e}");
            return;
        }
    };

    eprintln!();
    eprintln!("=== Phase 18.A POC.1 — Affine3 qmv_fast vs Affine4 qmv_fast (gate_up) ===");
    eprintln!("Shape: out={OUT}, in={IN}, batch={BATCH}");
    eprintln!("Iterations: {ITERS} per arm ({WARMUP} warmup, single drain at end)");
    eprintln!("Topology: qmv_fast (NSG=2, RPS=4, VPT=16, BLK=512) — production decode kernel");
    eprintln!();

    // ── Affine3 qmv_fast ────────────────────────────────────────────────────
    let groups = OUT * IN / AFFINE3_GROUP_SIZE;
    let codes = synth_3bit_codes(OUT * IN, 0xDEAD_BEEF);
    let scales3 = synth_meta(groups, 0xCAFE_BABE, false);
    let biases3 = synth_meta(groups, 0x1234_5678, true);
    let weight3 = Affine3Weight::from_codes_3bit(&ctx3.ctx, &codes, &scales3, &biases3, OUT, IN)
        .expect("affine3 weight");

    let x_bf16 = synth_x_bf16(IN, 0xAAAA_BBBB);
    let x_buf = ctx3.ctx.buffer_with_data(&x_bf16);
    let y3_buf = ctx3.ctx.buffer_for::<u16>(OUT);

    // Warmup.
    for _ in 0..WARMUP {
        ctx3.qmv_fast_bf16in_bf16out_pipelined(&weight3, &x_buf, &y3_buf, BATCH)
            .unwrap();
    }
    let drain = lumen_metal::metal::new_command_buffer(&ctx3.ctx.queue);
    drain.commit();
    drain.wait_until_completed();

    // Pipelined timing.
    let t0 = Instant::now();
    for _ in 0..ITERS {
        ctx3.qmv_fast_bf16in_bf16out_pipelined(&weight3, &x_buf, &y3_buf, BATCH)
            .unwrap();
    }
    let drain = lumen_metal::metal::new_command_buffer(&ctx3.ctx.queue);
    drain.commit();
    drain.wait_until_completed();
    let aff3_us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    let aff3_packed_bytes = OUT * IN * 3 / 8;
    let aff3_meta_bytes = groups * 2 * 2;
    let aff3_act_bytes = IN * 2;
    let aff3_out_bytes = OUT * 2;
    let aff3_total_bytes = aff3_packed_bytes + aff3_meta_bytes + aff3_act_bytes + aff3_out_bytes;
    let aff3_bw_gbs = aff3_total_bytes as f64 / aff3_us / 1e3;
    let aff3_bw_pct = aff3_bw_gbs / M3_MAX_PEAK_BW_GBS * 100.0;

    eprintln!("Affine3 qmv_fast_bf16in_bf16out:");
    eprintln!(
        "  total bytes: {:.2} MB  (packed {:.1} + meta {:.1} + act+out KB)",
        aff3_total_bytes as f64 / 1e6,
        aff3_packed_bytes as f64 / 1e6,
        aff3_meta_bytes as f64 / 1e6,
    );
    eprintln!(
        "  pipelined µs/call: {aff3_us:.1}, effective BW: {aff3_bw_gbs:.1} GB/s ({aff3_bw_pct:.1}% of {M3_MAX_PEAK_BW_GBS} GB/s)"
    );
    eprintln!();

    // ── Affine4 qmv_fast (production reference) ─────────────────────────────
    let packed4 = synth_4bit_packed(OUT, IN, 0xFADE_FADE);
    let scales4 = synth_meta(groups, 0xCAFE_BABE, false);
    let biases4 = synth_meta(groups, 0x1234_5678, true);
    let weight4 = Affine4Weight::from_host(&ctx4.ctx, &packed4, &scales4, &biases4, OUT, IN)
        .expect("affine4 weight");

    use lumen_metal::affine4_linear::Affine4Linear;
    use std::sync::Arc;
    let lin4 = Affine4Linear::new(weight4, None, Arc::new(ctx4));

    // Build candle bf16 tensor for the activation.
    use candle_core::{DType, Device, Tensor};
    let dev = match Device::new_metal(0) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("[skip] no Metal device for Affine4 reference path");
            return;
        }
    };
    // Source f32 to keep activation values consistent (same as Stage 2).
    let x_f32_for_a4: Vec<f32> = x_bf16
        .iter()
        .map(|b| f32::from_bits((*b as u32) << 16))
        .collect();
    let x_tensor = Tensor::from_vec(x_f32_for_a4, &[1, 1, IN], &dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .contiguous()
        .unwrap();

    use candle_core::backend::BackendDevice as _;
    let metal_dev = match x_tensor.device() {
        Device::Metal(m) => m,
        _ => unreachable!(),
    };

    // Warmup.
    for _ in 0..WARMUP {
        let _ = lin4.forward_bf16_in_bf16_out(&x_tensor).unwrap();
    }
    let _ = metal_dev.synchronize();

    // Pipelined timing.
    let t1 = Instant::now();
    for _ in 0..ITERS {
        let _ = lin4.forward_bf16_in_bf16_out(&x_tensor).unwrap();
    }
    let _ = metal_dev.synchronize();
    let aff4_us = t1.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    let aff4_packed_bytes = OUT * IN / 2;
    let aff4_meta_bytes = groups * 2 * 2;
    let aff4_act_bytes = IN * 2;
    let aff4_out_bytes = OUT * 2;
    let aff4_total_bytes = aff4_packed_bytes + aff4_meta_bytes + aff4_act_bytes + aff4_out_bytes;
    let aff4_bw_gbs = aff4_total_bytes as f64 / aff4_us / 1e3;
    let aff4_bw_pct = aff4_bw_gbs / M3_MAX_PEAK_BW_GBS * 100.0;

    eprintln!("Affine4 qmv_fast_bf16in_bf16out (production):");
    eprintln!(
        "  total bytes: {:.2} MB  (packed {:.1} + meta {:.1} + act+out KB)",
        aff4_total_bytes as f64 / 1e6,
        aff4_packed_bytes as f64 / 1e6,
        aff4_meta_bytes as f64 / 1e6,
    );
    eprintln!(
        "  pipelined µs/call: {aff4_us:.1}, effective BW: {aff4_bw_gbs:.1} GB/s ({aff4_bw_pct:.1}% of {M3_MAX_PEAK_BW_GBS} GB/s)"
    );
    eprintln!();

    // ── Comparison ──────────────────────────────────────────────────────────
    let saving_pct = (aff4_us - aff3_us) / aff4_us * 100.0;
    let bw_target_pct = aff4_total_bytes as f64 / aff3_total_bytes as f64;
    let theoretical_us_at_aff4_bw = aff3_total_bytes as f64 / (aff4_bw_gbs) / 1e3;
    let theoretical_saving_pct = (aff4_us - theoretical_us_at_aff4_bw) / aff4_us * 100.0;

    eprintln!("=== Comparison ===");
    eprintln!(
        "Total bytes:    Affine3 {:.2} MB vs Affine4 {:.2} MB  (Affine3 = {:.1}% of Affine4)",
        aff3_total_bytes as f64 / 1e6,
        aff4_total_bytes as f64 / 1e6,
        aff3_total_bytes as f64 / aff4_total_bytes as f64 * 100.0
    );
    eprintln!(
        "Pipelined µs:   Affine3 {aff3_us:.1} vs Affine4 {aff4_us:.1}  (Affine3 / Affine4 = {:.2}×)",
        aff3_us / aff4_us
    );
    eprintln!("BW utilization: Affine3 {aff3_bw_pct:.1}%  vs Affine4 {aff4_bw_pct:.1}%");
    eprintln!();
    eprintln!("Saving (Affine3 vs Affine4): {saving_pct:+.2}%   (positive = Affine3 faster)");
    eprintln!(
        "Theoretical max saving (if Affine3 hit Affine4's BW%): {theoretical_saving_pct:+.2}%"
    );
    eprintln!(
        "(theoretical = {theoretical_us_at_aff4_bw:.1} µs at {aff4_bw_pct:.1}% BW utilization)"
    );

    // ── Production projection ───────────────────────────────────────────────
    let saving_per_call_us = aff4_us - aff3_us;
    let calls_per_token = 64; // gate_up appears in all 64 layers
    let prod_saving_ms = saving_per_call_us * calls_per_token as f64 / 1000.0;
    let new_decode_ms = (67.0 - prod_saving_ms).max(1.0);
    let new_tps = 1000.0 / new_decode_ms;
    eprintln!();
    eprintln!("=== Production projection (gate_up only, 64 calls/token) ===");
    eprintln!("  saving per call:   {saving_per_call_us:+.1} µs");
    eprintln!("  saving per token:  {prod_saving_ms:+.2} ms (from gate_up alone)");
    eprintln!(
        "  decode 67 ms → {new_decode_ms:.2} ms = 15.04 → {new_tps:.2} tok/s ({:+.1}%)",
        (new_tps / 15.04 - 1.0) * 100.0
    );

    eprintln!();
    eprintln!("=== POC.1 Decision ===");
    if saving_pct >= 15.0 {
        eprintln!(
            "✅ Saving ≥ 15% — Affine3 qmv_fast achieves production-level BW saving. PROCEED to full impl (18.A.1+)."
        );
    } else if saving_pct >= 5.0 {
        eprintln!(
            "⚠️  Saving 5-15% — meaningful but below 17.8% projection. Consider full impl OR optimization (kernel tuning)."
        );
    } else if saving_pct >= -5.0 {
        eprintln!(
            "❌ Saving ≈ 0% — dequant compute exhausts BW headroom even at qmv_fast topology. ABANDON 18.A. Pivot to 18.B (KV compression)."
        );
    } else {
        eprintln!(
            "❌ NEGATIVE — Affine3 SLOWER than Affine4 production. Lever DEFINITIVELY DEAD. Skip to 18.B."
        );
    }
}

#[test]
fn affine3_qmv_fast_parity_small() {
    // Smaller shape that still satisfies qmv_fast constraints (in % 512 == 0, out % 8 == 0).
    const OUT: usize = 256;
    const IN: usize = 512;

    let ctx3 = match Affine3Context::new() {
        Ok(c) => c,
        Err(_) => return,
    };

    // Identity dequant: scale=1, bias=0 → reconstructed value == raw 3-bit code.
    let codes: Vec<u8> = (0..OUT * IN).map(|i| (i % 8) as u8).collect();
    let groups = OUT * IN / AFFINE3_GROUP_SIZE;
    let scales: Vec<u16> = vec![bf16_from_f32(1.0); groups];
    let biases: Vec<u16> = vec![bf16_from_f32(0.0); groups];

    let weight = Affine3Weight::from_codes_3bit(&ctx3.ctx, &codes, &scales, &biases, OUT, IN)
        .expect("affine3 weight");

    // Activation: all 1.0 in bf16.
    let x_data: Vec<u16> = vec![bf16_from_f32(1.0); IN];
    let x_buf = ctx3.ctx.buffer_with_data(&x_data);
    let y_buf = ctx3.ctx.buffer_for::<u16>(OUT);

    ctx3.qmv_fast_bf16in_bf16out_pipelined(&weight, &x_buf, &y_buf, 1)
        .unwrap();
    let drain = lumen_metal::metal::new_command_buffer(&ctx3.ctx.queue);
    drain.commit();
    drain.wait_until_completed();

    let y_bf16 = ctx3.ctx.read_buffer::<u16>(&y_buf, OUT);
    let y_f32: Vec<f32> = y_bf16
        .iter()
        .map(|&b| f32::from_bits((b as u32) << 16))
        .collect();

    // CPU reference.
    let mut max_rel_err: f32 = 0.0;
    let mut sum_rel_err: f32 = 0.0;
    for row in 0..OUT {
        let cpu_sum: f32 = (0..IN).map(|k| codes[row * IN + k] as f32).sum();
        let gpu = y_f32[row];
        let rel_err = (gpu - cpu_sum).abs() / cpu_sum.abs().max(1e-6);
        max_rel_err = max_rel_err.max(rel_err);
        sum_rel_err += rel_err;
    }
    let avg_rel_err = sum_rel_err / OUT as f32;
    eprintln!(
        "affine3 qmv_fast parity: max_rel_err={max_rel_err:.4e}, avg_rel_err={avg_rel_err:.4e}"
    );
    assert!(
        max_rel_err < 0.02,
        "qmv_fast parity violated: max_rel_err={max_rel_err}"
    );
}
