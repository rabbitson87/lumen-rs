//! Fault sweep over the safetensors weight loader (005 Phase 3).
//!
//! `NativeWeights::load_dir` is what a downloaded checkpoint directory meets
//! first, and the failure mode that matters here is not a crash — it is
//! **silence**. Upstream `Array::load_safetensors` rejects a truncated header,
//! an absurd length prefix and a garbled header cleanly; what it does *not*
//! reject is a file truncated inside its **data** section, which is precisely
//! the shape an interrupted download leaves behind. Measured before the fix:
//! a 4×4 f32 tensor missing its last 16 bytes loaded without error and read
//! back `3.0` where the file had written `15.0`.
//!
//! A model that loads and serves wrong weights forever is the exact failure
//! class task 005 exists for — invisible correctness. So `load_dir` now
//! validates that the bytes the header promises are actually present, and this
//! sweep pins both halves: every truncation is rejected, and every *intact*
//! file still loads.

use lumen_mlx::qwen3_5_moe_config::NativeWeights;
use lumen_testkit::faults::{
    Corruption, TensorSpec, build_safetensors, corrupt, minimal_safetensors, truncation_offsets,
};

/// Write `bytes` as the only shard in a fresh directory and load it.
fn load_shard(bytes: &[u8]) -> (tempfile::TempDir, anyhow::Result<NativeWeights>) {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("model.safetensors"), bytes).expect("write shard");
    let res = NativeWeights::load_dir(dir.path());
    (dir, res)
}

#[test]
fn an_intact_shard_loads() {
    let (_d, res) = load_shard(&minimal_safetensors());
    let w = res.expect("the unmutated fixture must load — otherwise the sweep tests nothing");
    assert_eq!(w.len(), 2);
    assert!(w.get("weight").is_some() && w.get("bias").is_some());
}

/// The defect this sweep was written for: a file cut inside the data section
/// used to load silently with wrong values.
#[test]
fn data_section_truncation_is_rejected_not_silently_wrong() {
    let good = minimal_safetensors();
    // Data section is the tail; cut into it without touching the header.
    for cut in [1usize, 4, 16, 32, 60] {
        let bad = corrupt(&good, Corruption::TruncateAt(good.len() - cut));
        let (_d, res) = load_shard(&bad);
        let Err(err) = res else {
            panic!(
                "a shard missing its last {cut} data bytes loaded successfully — this is the \
                 silent-corruption path: it yields wrong weights with no error anywhere"
            )
        };
        let msg = format!("{err:#}");
        assert!(
            msg.contains("truncated") && msg.contains("short"),
            "the error must say the file is short so the fix is obvious (re-download): {msg}"
        );
    }
}

/// Every truncation offset, not the handful someone thought of. Each must be
/// an `Err`, and the loader must stay usable afterwards.
#[test]
fn every_truncation_errors_and_leaves_the_loader_usable() {
    let good = minimal_safetensors();
    let header_len = u64::from_le_bytes(good[..8].try_into().unwrap()) as usize;
    let boundaries = [8, 8 + header_len, good.len()];
    let offsets = truncation_offsets(good.len(), 1, &boundaries);
    assert!(
        offsets.len() > 100,
        "sweep should be dense: {}",
        offsets.len()
    );

    for off in offsets {
        let bad = corrupt(&good, Corruption::TruncateAt(off));
        let (_d, res) = load_shard(&bad);
        assert!(
            res.is_err(),
            "truncation at {off}/{} loaded as a complete checkpoint",
            good.len()
        );
        // Post-failure integrity.
        let (_d2, good_res) = load_shard(&good);
        assert!(good_res.is_ok(), "loader broke after a truncation at {off}");
    }
}

/// Header-level corruptions were already clean errors upstream; the sweep
/// asserts they stay that way rather than regressing into panics.
#[test]
fn header_corruptions_error_cleanly() {
    let good = minimal_safetensors();
    for c in [
        Corruption::HeaderLen(u64::MAX),
        Corruption::HeaderLen(0),
        Corruption::GarbageHeader,
        Corruption::FlipByte(0),
        Corruption::FlipByte(9),
    ] {
        let (_d, res) = load_shard(&corrupt(&good, c));
        assert!(res.is_err(), "header corruption {c:?} loaded as valid");
    }
}

/// A header whose `data_offsets` claim more bytes than the file could ever
/// hold is the allocation-bomb shape for this format. It must be rejected on
/// the size check, before any reader tries to honour the claim.
#[test]
fn implausible_data_offsets_are_rejected() {
    let bytes = build_safetensors(&[TensorSpec {
        name: "weight",
        dtype: "F32",
        shape: vec![4, 4],
        // Declared length is honest here; the corruption is applied below by
        // rewriting the header's offsets to a preposterous end.
        data: (0..16u32).flat_map(|i| (i as f32).to_le_bytes()).collect(),
    }]);
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value =
        serde_json::from_slice(&bytes[8..8 + header_len]).expect("header parses");
    let mut header = header.as_object().expect("object").clone();
    header.insert(
        "weight".into(),
        serde_json::json!({
            "dtype": "F32",
            "shape": [4, 4],
            "data_offsets": [0u64, u64::MAX / 2],
        }),
    );
    let new_header = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut bad = Vec::new();
    bad.extend_from_slice(&(new_header.len() as u64).to_le_bytes());
    bad.extend_from_slice(&new_header);
    bad.extend_from_slice(&bytes[8 + header_len..]);

    let (_d, res) = load_shard(&bad);
    assert!(
        res.is_err(),
        "a header claiming {} data bytes in a {}-byte file must be rejected",
        u64::MAX / 2,
        bad.len()
    );
}

/// A directory with no shards must say so, rather than loading an empty bag
/// that fails much later with a missing-tensor error.
#[test]
fn empty_directory_is_named_as_the_problem() {
    let dir = tempfile::tempdir().expect("tempdir");
    let Err(err) = NativeWeights::load_dir(dir.path()) else {
        panic!("an empty directory must error, not load an empty weight bag")
    };
    let msg = format!("{err:#}");
    assert!(msg.contains("safetensors"), "got: {msg}");
}
