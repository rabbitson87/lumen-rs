//! Parity check between `loader::dequant_int8_affine` and MLX's reference
//! `mx.dequantize(..., bits=8, mode="affine")`. The fixture is produced by
//! `scripts/dump_int8_affine_ref.py` and covers layer 0 MoE `gate.weight` —
//! the first int8-affine tensor the shipped Qwen3.6 MoE checkpoint uses.
//!
//! The dequant helper is private to the crate, so this test depends on a
//! `pub(crate)` re-export. To run:
//!     python scripts/dump_int8_affine_ref.py
//!     cargo test -p lumen-model --test int8_affine_ref_parity --release -- --nocapture

use std::fs::File;
use std::io::Read;
use std::path::Path;

use lumen_model::qwen3_5_moe::loader::debug_dequant_int8_affine;

const REF_PATH: &str = "/tmp/int8_affine_ref.bin";

fn read_u32(r: &mut impl Read) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

#[test]
fn int8_affine_dequant_matches_mlx_reference() {
    if !Path::new(REF_PATH).exists() {
        eprintln!("skip: {REF_PATH} missing — run scripts/dump_int8_affine_ref.py");
        return;
    }

    let mut f = File::open(REF_PATH).unwrap();
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).unwrap();
    assert_eq!(&magic, b"TQI8");

    let out_dim = read_u32(&mut f).unwrap() as usize;
    let in_dim = read_u32(&mut f).unwrap() as usize;
    let group_size = read_u32(&mut f).unwrap() as usize;
    let bits = read_u32(&mut f).unwrap();
    let packed_len = read_u32(&mut f).unwrap() as usize;
    let scales_len = read_u32(&mut f).unwrap() as usize;
    let biases_len = read_u32(&mut f).unwrap() as usize;
    let ref_len = read_u32(&mut f).unwrap() as usize;
    assert_eq!(bits, 8);
    assert_eq!(ref_len, out_dim * in_dim);

    let mut packed_bytes = vec![0u8; packed_len * 4];
    f.read_exact(&mut packed_bytes).unwrap();
    let packed: Vec<u32> = packed_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let mut scales_bytes = vec![0u8; scales_len * 2];
    f.read_exact(&mut scales_bytes).unwrap();
    let scales: Vec<u16> = scales_bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();

    let mut biases_bytes = vec![0u8; biases_len * 2];
    f.read_exact(&mut biases_bytes).unwrap();
    let biases: Vec<u16> = biases_bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();

    let mut ref_bytes = vec![0u8; ref_len * 4];
    f.read_exact(&mut ref_bytes).unwrap();
    let expected: Vec<f32> = ref_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let got = debug_dequant_int8_affine(&packed, &scales, &biases, group_size)
        .expect("Rust dequant");
    assert_eq!(got.len(), expected.len());

    // L2, max abs, cosine.
    let mut l2_err_sq = 0.0f64;
    let mut l2_ref_sq = 0.0f64;
    let mut max_abs = 0f32;
    let mut dot = 0.0f64;
    let mut ng = 0.0f64;
    let mut nr = 0.0f64;
    for (g, r) in got.iter().zip(expected.iter()) {
        let d = g - r;
        l2_err_sq += (d as f64) * (d as f64);
        l2_ref_sq += (*r as f64) * (*r as f64);
        max_abs = max_abs.max(d.abs());
        dot += (*g as f64) * (*r as f64);
        ng += (*g as f64) * (*g as f64);
        nr += (*r as f64) * (*r as f64);
    }
    let l2_rel = (l2_err_sq / l2_ref_sq.max(1e-12)).sqrt();
    let cos = dot / (ng.sqrt() * nr.sqrt() + 1e-12);

    eprintln!("── int8-affine dequant parity ────────────────────────────");
    eprintln!("  shape:      out={out_dim} in={in_dim}  group={group_size}");
    eprintln!("  L2 rel err: {l2_rel:.6e}");
    eprintln!("  max |err|:  {max_abs:.6e}");
    eprintln!("  cosine:     {cos:.8}");
    // First 8 side-by-side.
    for i in 0..8 {
        eprintln!(
            "  w[0,{i:>4}]: got={:>+9.6}  ref={:>+9.6}  Δ={:+.3e}",
            got[i], expected[i], got[i] - expected[i]
        );
    }
    // Also check one sample from the middle and the end to rule out packing drift.
    for row in [0, 1, 127, 255] {
        for col in [0, 63, 64, in_dim - 1] {
            let idx = row * in_dim + col;
            if (got[idx] - expected[idx]).abs() > 1e-3 {
                eprintln!(
                    "  MISMATCH w[{row},{col}]: got={:+.6}  ref={:+.6}  Δ={:+.3e}",
                    got[idx], expected[idx], got[idx] - expected[idx]
                );
            }
        }
    }

    assert!(l2_rel < 1e-3, "int8-affine dequant diverges from MLX");
    assert!(cos > 0.9999, "cosine sim too low");
}
