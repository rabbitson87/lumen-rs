//! MXFP4 Metal matvec GPU output verified against CPU reference.
//!
//! Reference is computed from MLX-dequantized weight (the `.ref` fixture) times a
//! deterministic input vector. GPU path runs the `mxfp4_matvec_f32` kernel on the
//! exact same packed/scales bytes MLX produced — any mismatch beyond f32 summation
//! rounding indicates a kernel bug (wrong nibble order, wrong scale decode,
//! misindexed packed buffer, etc.).

use std::path::{Path, PathBuf};

use lumen_metal::mxfp4_gpu::MxFp4Context;

const MAGIC: &[u8; 4] = b"MXFP";
const HEADER_LEN: usize = 12;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

fn read_fixture_payload(path: &Path) -> Option<(usize, usize, Vec<u8>)> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() < HEADER_LEN || &bytes[..4] != MAGIC {
        panic!("fixture {:?} has bad header", path);
    }
    let rows = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let cols = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    Some((rows, cols, bytes[HEADER_LEN..].to_vec()))
}

fn load_fixture(name: &str) -> Option<(usize, usize, Vec<u32>, Vec<u8>, Vec<f32>)> {
    let dir = fixture_dir();
    let (rows, cols, packed_bytes) =
        read_fixture_payload(&dir.join(format!("mxfp4_{name}.weight")))?;
    let (srows, scols, scale_bytes) =
        read_fixture_payload(&dir.join(format!("mxfp4_{name}.scales")))?;
    let (rrows, rcols, ref_bytes) = read_fixture_payload(&dir.join(format!("mxfp4_{name}.ref")))?;
    assert_eq!((rows, cols), (srows, scols));
    assert_eq!((rows, cols), (rrows, rcols));

    let packed: Vec<u32> = packed_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let reference: Vec<f32> = ref_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Some((rows, cols, packed, scale_bytes, reference))
}

fn cpu_matvec_f32(w_dequant: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows];
    for r in 0..rows {
        let mut acc = 0.0f32;
        for c in 0..cols {
            acc += w_dequant[r * cols + c] * x[c];
        }
        out[r] = acc;
    }
    out
}

fn run_matvec_fixture(name: &str) {
    let Some((rows, cols, packed, scales, ref_dequant)) = load_fixture(name) else {
        eprintln!(
            "SKIP {name}: fixtures not found under {}. Run `.venv/bin/python scripts/generate_mxfp4_fixture.py`.",
            fixture_dir().display()
        );
        return;
    };

    // Deterministic input with mixed signs and small magnitudes.
    let x: Vec<f32> = (0..cols)
        .map(|i| ((i as f32 * 0.37).sin() * 0.5) - 0.1)
        .collect();

    let expected = cpu_matvec_f32(&ref_dequant, &x, rows, cols);

    let gpu = MxFp4Context::new().expect("Metal device unavailable");
    let got = gpu
        .matvec_f32(&packed, &scales, &x, rows, cols)
        .expect("matvec kernel failed");
    assert_eq!(got.len(), expected.len());

    for i in 0..rows {
        // Tolerance: rows have up to `cols` terms summed; f32 has ~7 decimal digits;
        // allow 4e-5 relative + 1e-5 absolute (very loose, would flag real bugs).
        let err = (got[i] - expected[i]).abs();
        let tol = expected[i].abs() * 4e-5 + 1e-5;
        assert!(
            err <= tol,
            "{name} row {i}: got {} vs expected {} (|err|={err}, tol={tol})",
            got[i],
            expected[i]
        );
    }
}

#[test]
fn multi_group_matvec_matches_cpu() {
    run_matvec_fixture("multi_group");
}

#[test]
fn dynamic_range_matvec_matches_cpu() {
    run_matvec_fixture("dynamic_range");
}

#[test]
fn single_group_matvec_matches_cpu() {
    run_matvec_fixture("single_group");
}
