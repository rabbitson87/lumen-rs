//! Bit-exact fixture verification: our MXFP4 decoder vs MLX's `mx.dequantize`.
//!
//! Fixtures are produced by `scripts/generate_mxfp4_fixture.py` (requires Apple MLX).
//! If fixtures are absent, the test is skipped with an informative message so CI on
//! non-Apple-Silicon machines does not fail.
//!
//! Binary format for each fixture triple (`.weight` / `.scales` / `.ref`):
//!   byte  0.. 3 : magic ASCII "MXFP"
//!   byte  4.. 7 : u32 little-endian  — rows (dim 0)
//!   byte  8..11 : u32 little-endian  — cols (dim 1)
//!   byte 12..   : payload
//!     .weight → rows * cols / 8  u32 little-endian
//!     .scales → rows * cols / 32 u8
//!     .ref    → rows * cols      f32 little-endian

use std::path::{Path, PathBuf};

use lumen_metal::mxfp4;

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

    assert_eq!((rows, cols), (srows, scols), "weight/scales shape mismatch");
    assert_eq!((rows, cols), (rrows, rcols), "weight/ref shape mismatch");

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

fn run_fixture(name: &str) {
    let Some((rows, cols, packed, scales, reference)) = load_fixture(name) else {
        eprintln!(
            "SKIP {name}: fixtures not found under {}. Run `python scripts/generate_mxfp4_fixture.py`.",
            fixture_dir().display()
        );
        return;
    };

    let n = rows * cols;
    assert_eq!(packed.len() * mxfp4::NIBBLES_PER_WORD, n);
    assert_eq!(scales.len() * mxfp4::MXFP4_GROUP_SIZE, n);
    assert_eq!(reference.len(), n);

    let mut out = vec![0.0f32; n];
    mxfp4::dequantize_f32(&packed, &scales, &mut out).expect("dequant should succeed");

    for (i, (&got, &expected)) in out.iter().zip(reference.iter()).enumerate() {
        assert!(
            got.to_bits() == expected.to_bits(),
            "{name}[{i}]: got {got} (bits=0x{:08x}) vs MLX {expected} (bits=0x{:08x})",
            got.to_bits(),
            expected.to_bits()
        );
    }
}

#[test]
fn single_group_matches_mlx() {
    run_fixture("single_group");
}

#[test]
fn multi_group_matches_mlx() {
    run_fixture("multi_group");
}

#[test]
fn dynamic_range_matches_mlx() {
    run_fixture("dynamic_range");
}
