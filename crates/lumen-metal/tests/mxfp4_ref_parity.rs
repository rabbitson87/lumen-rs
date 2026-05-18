//! Numerical parity check between our MXFP4 Metal kernel and MLX's
//! `mx.quantized_matmul(..., mode="mxfp4", bits=4, group_size=32)` reference.
//!
//! The fixture file is produced by `scripts/dump_mxfp4_ref.py` and lives at
//! `/tmp/mxfp4_ref.bin`. It contains real weights from a shipped shard
//! (`layer 3 self_attn.q_proj` of `mlx-community/Qwen3.6-35B-A3B-mxfp4`), a
//! fixed activation, and MLX's matmul output. If the shard cache + MLX aren't
//! available, the test skips rather than failing.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use lumen_metal::mxfp4_gpu::{MxFp4Context, Mxfp4Weight};

const REF_PATH: &str = "/tmp/mxfp4_ref.bin";

struct RefBlob {
    out_features: usize,
    in_features: usize,
    group_size: u32,
    bits: u32,
    x: Vec<f32>,
    packed: Vec<u32>,
    scales: Vec<u8>,
    expected_y: Vec<f32>,
}

fn read_u32(r: &mut impl Read) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_blob() -> std::io::Result<RefBlob> {
    let mut f = File::open(REF_PATH)?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    assert_eq!(&magic, b"TQMM", "ref file magic mismatch");

    let out_features = read_u32(&mut f)? as usize;
    let in_features = read_u32(&mut f)? as usize;
    let group_size = read_u32(&mut f)?;
    let bits = read_u32(&mut f)?;
    let x_len = read_u32(&mut f)? as usize;
    let packed_len = read_u32(&mut f)? as usize;
    let scales_len = read_u32(&mut f)? as usize;
    let y_len = read_u32(&mut f)? as usize;

    let mut x_bytes = vec![0u8; x_len * 4];
    f.read_exact(&mut x_bytes)?;
    let x: Vec<f32> = x_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let mut packed_bytes = vec![0u8; packed_len * 4];
    f.read_exact(&mut packed_bytes)?;
    let packed: Vec<u32> = packed_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let mut scales = vec![0u8; scales_len];
    f.read_exact(&mut scales)?;

    let mut y_bytes = vec![0u8; y_len * 4];
    f.read_exact(&mut y_bytes)?;
    let expected_y: Vec<f32> = y_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    Ok(RefBlob {
        out_features,
        in_features,
        group_size,
        bits,
        x,
        packed,
        scales,
        expected_y,
    })
}

#[test]
fn mxfp4_matmul_matches_mlx_reference() {
    if !Path::new(REF_PATH).exists() {
        eprintln!(
            "skip: {} missing — run `python scripts/dump_mxfp4_ref.py` first",
            REF_PATH
        );
        return;
    }

    let blob = read_blob().expect("read ref blob");
    assert_eq!(blob.group_size, 32);
    assert_eq!(blob.bits, 4);
    assert_eq!(blob.x.len(), blob.in_features);
    assert_eq!(blob.expected_y.len(), blob.out_features);
    assert_eq!(
        blob.packed.len(),
        blob.out_features * blob.in_features / 8
    );
    assert_eq!(
        blob.scales.len(),
        blob.out_features * blob.in_features / 32
    );

    let ctx = MxFp4Context::new().expect("Metal ctx");
    let weight = Mxfp4Weight::from_host(
        &ctx.ctx,
        &blob.packed,
        &blob.scales,
        blob.out_features,
        blob.in_features,
    )
    .expect("upload weight");

    let got = ctx
        .matmul_with_weight(&weight, &blob.x, 1)
        .expect("matmul");
    assert_eq!(got.len(), blob.out_features);

    // Compare: L2 relative error, max absolute error, cosine similarity.
    let mut l2_err_sq = 0.0f64;
    let mut l2_ref_sq = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut dot = 0.0f64;
    let mut ng = 0.0f64;
    let mut nr = 0.0f64;
    for (g, r) in got.iter().zip(blob.expected_y.iter()) {
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

    eprintln!("── MXFP4 Rust-vs-MLX parity ──────────────────────────────");
    eprintln!("  shape:       out={} in={}", blob.out_features, blob.in_features);
    eprintln!("  L2 rel err:  {l2_rel:.6e}");
    eprintln!("  max |err|:   {max_abs:.6e}");
    eprintln!("  cosine sim:  {cos:.6}");
    eprintln!("  ref range:   [{:.4}, {:.4}]  std={:.4}",
              blob.expected_y.iter().cloned().fold(f32::INFINITY, f32::min),
              blob.expected_y.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
              {
                  let mean = blob.expected_y.iter().sum::<f32>() / blob.expected_y.len() as f32;
                  let v = blob.expected_y.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
                      / blob.expected_y.len() as f32;
                  v.sqrt()
              });
    eprintln!("  got range:   [{:.4}, {:.4}]",
              got.iter().cloned().fold(f32::INFINITY, f32::min),
              got.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

    // First 8 values side-by-side for eyeballing when things drift.
    for i in 0..8 {
        eprintln!("  y[{i:>4}]: got={:>+10.5}  ref={:>+10.5}  Δ={:+.4e}",
                  got[i], blob.expected_y[i], got[i] - blob.expected_y[i]);
    }

    // Tolerance: MXFP4 is exact dequant + fp32 accumulate, so the only source of drift
    // is fp32 FMA ordering. L2 rel ≤ 1e-4 is a comfortable pass; cos ≥ 1 - 1e-6.
    assert!(l2_rel < 1e-3, "MXFP4 matmul diverges from MLX reference");
    assert!(cos > 0.9999, "cosine sim too low — kernel numerically wrong");
}
