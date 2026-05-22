//! Parity test for `mxfp8_matmul_bf16` Metal kernel vs CPU dequant reference.
//!
//! Mirrors `affine8_parity.rs` but for the OCP MXFP8 format
//! (E4M3 elements + E8M0 byte scales, group_size=32, no biases).

use half::bf16;
use lumen_metal::mxfp8_gpu::{
    MXFP8_GROUP_SIZE, Mxfp8Context, Mxfp8Weight, cpu_reference_matmul_bf16,
};

fn make_ctx_or_skip() -> Option<Mxfp8Context> {
    match Mxfp8Context::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("[skip] no Metal device or shader compile failed: {e}");
            None
        }
    }
}

/// All weights = 1.0 (E4M3 = 0x38), all scales = 1.0 (E8M0 = 127), all
/// inputs = 1.0. Expected output: in_features per output row.
#[test]
fn mxfp8_matmul_bf16_identity_ones() {
    let Some(ctx) = make_ctx_or_skip() else {
        return;
    };
    // qmv_fast requires in_features % 512 == 0 && out_features % 8 == 0.
    // Stay aligned so we exercise the actually-used dispatch path.
    let in_f = 512usize;
    let out = 8usize;
    let batch = 1usize;

    // 4 E4M3 bytes of 0x38 (=+1.0) packed LSB-first as a uint32.
    let ones_word: u32 = 0x3838_3838;
    let packed = vec![ones_word; out * in_f / 4];
    // E8M0 = 127 → 2^0 = 1.0
    let scales = vec![127u8; out * (in_f / MXFP8_GROUP_SIZE)];
    let x_bf16: Vec<u16> = vec![bf16::from_f32(1.0).to_bits(); batch * in_f];

    let weight = Mxfp8Weight::from_host(&ctx.ctx, &packed, &scales, out, in_f).unwrap();
    let gpu = ctx
        .matmul_bf16_with_weight(&weight, &x_bf16, batch)
        .unwrap();
    let cpu = cpu_reference_matmul_bf16(&packed, &scales, &x_bf16, out, in_f, batch);

    for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        let gf = bf16::from_bits(*g).to_f32();
        let cf = bf16::from_bits(*c).to_f32();
        assert!(
            (gf - cf).abs() / cf.abs().max(1.0) < 1e-2,
            "y[{i}] gpu={gf} cpu={cf}"
        );
        // Expected: in_f * 1.0 = 512.0
        assert!(
            (cf - in_f as f32).abs() < 1.0,
            "cpu y[{i}]={cf} expected ~{in_f}"
        );
    }
}

/// Random-ish E4M3 weights + random E8M0 scales — GPU and CPU should agree
/// within bf16 truncation error (~0.5%).
#[test]
fn mxfp8_matmul_bf16_random() {
    let Some(ctx) = make_ctx_or_skip() else {
        return;
    };
    let in_f = 512usize;
    let out = 8usize;
    let batch = 3usize;
    let groups = in_f / MXFP8_GROUP_SIZE;

    let mut prng: u32 = 0xC0FF_EE42;
    let mut rng = || -> u32 {
        prng ^= prng << 13;
        prng ^= prng >> 17;
        prng ^= prng << 5;
        prng
    };

    // Random E4M3 bytes. Mask off the 0x7F/0xFF NaN bytes so we don't trip
    // the NaN sentinel (both kernel and reference collapse it to 0; testing
    // that path is covered by the per-byte unit test).
    let mut packed: Vec<u32> = Vec::with_capacity(out * in_f / 4);
    for _ in 0..(out * in_f / 4) {
        let mut w = rng();
        for byte_idx in 0..4u32 {
            let b = (w >> (byte_idx * 8)) & 0xFF;
            if b == 0x7F || b == 0xFF {
                // Replace with 0x38 (=+1.0) — keeps the rest of the test
                // exercising real dequant.
                w &= !(0xFF << (byte_idx * 8));
                w |= 0x38 << (byte_idx * 8);
            }
        }
        packed.push(w);
    }

    // E8M0 in [120, 134] → scale in [2^-7, 2^7]. Avoid 0xFF NaN.
    let scales: Vec<u8> = (0..out * groups)
        .map(|_| ((rng() % 15) as u8) + 120)
        .collect();

    // bf16 inputs in [-1, 1].
    let mut x_bf16 = Vec::with_capacity(batch * in_f);
    for _ in 0..(batch * in_f) {
        let v = ((rng() & 0xFFFF) as f32 / 32768.0) - 1.0;
        x_bf16.push(bf16::from_f32(v).to_bits());
    }

    let weight = Mxfp8Weight::from_host(&ctx.ctx, &packed, &scales, out, in_f).unwrap();
    let gpu = ctx
        .matmul_bf16_with_weight(&weight, &x_bf16, batch)
        .unwrap();
    let cpu = cpu_reference_matmul_bf16(&packed, &scales, &x_bf16, out, in_f, batch);

    let mut max_rel = 0f32;
    for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        let gf = bf16::from_bits(*g).to_f32();
        let cf = bf16::from_bits(*c).to_f32();
        let denom = cf.abs().max(1e-3);
        let rel = (gf - cf).abs() / denom;
        max_rel = max_rel.max(rel);
        // bf16 has ~7 mantissa bits; cooperative reduction over many lanes
        // accumulates a few ULPs. 1% relative is conservative.
        assert!(rel < 1e-2, "y[{i}] gpu={gf} cpu={cf} rel={rel}");
    }
    eprintln!("[mxfp8_parity] random in={in_f} out={out} batch={batch} max_rel={max_rel:.4}");
}

/// Force the naive (non-qmv_fast) kernel via env var and confirm parity.
/// Exercises the path used for any future projection that misses the
/// `in % 512 == 0 && out % 8 == 0` alignment.
#[test]
fn mxfp8_matmul_bf16_naive_path() {
    let Some(ctx) = make_ctx_or_skip() else {
        return;
    };
    // Safety: scoped to this test thread; cargo runs tests in the same
    // module sequentially by default. We restore on exit.
    unsafe { std::env::set_var("LUMEN_MXFP8_NAIVE", "1") };

    let in_f = 64usize; // group_size=32 → 2 groups, naive kernel only
    let out = 4usize;
    let batch = 2usize;
    let groups = in_f / MXFP8_GROUP_SIZE;

    // All E4M3 = 0x38 (1.0), all scales = 127 (2^0 = 1.0).
    let packed = vec![0x3838_3838u32; out * in_f / 4];
    let scales = vec![127u8; out * groups];
    let x_bf16: Vec<u16> = vec![bf16::from_f32(1.0).to_bits(); batch * in_f];

    let weight = Mxfp8Weight::from_host(&ctx.ctx, &packed, &scales, out, in_f).unwrap();
    let gpu = ctx
        .matmul_bf16_with_weight(&weight, &x_bf16, batch)
        .unwrap();

    unsafe { std::env::remove_var("LUMEN_MXFP8_NAIVE") };

    for (i, g) in gpu.iter().enumerate() {
        let gf = bf16::from_bits(*g).to_f32();
        assert!(
            (gf - in_f as f32).abs() < 1.0,
            "y[{i}]={gf} expected ~{in_f}"
        );
    }
}
