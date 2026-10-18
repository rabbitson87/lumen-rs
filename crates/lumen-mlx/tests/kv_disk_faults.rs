//! Fault sweep over the LKV2 KV-disk format (005 Phase 3; LKV2 in task 023).
//!
//! `read_lkv` parses attacker-adjacent bytes: a KV snapshot is written by a
//! previous run and read by this one, so a partial write (power loss mid-flush,
//! a full disk, a killed process) hands it a truncated file, and disk rot hands
//! it a flipped byte. Both must produce a typed `Err` naming the problem —
//! never a panic, and never an allocation sized from an unvalidated field.
//!
//! Sweep shape is SQLite's: truncate at *every* offset (plus ±1 around each
//! structural boundary), not at three offsets someone thought of. `read_lkv` is
//! pure `Read` + parse, so a full sweep of a small file costs milliseconds and
//! belongs in tier 0.
//!
//! The last invariant is the one that keeps a sweep honest: after every
//! corrupt input, a known-good input must still round-trip. A reader that
//! poisons a global on the first bad file would pass a
//! corruption-only assertion.

use lumen_mlx::kv_disk::{
    ArrayRecord, DtypeTag, KvManifest, LayerKindTag, LayerMeta, read_lkv, write_lkv,
};
use lumen_testkit::faults::{Corruption, corrupt, truncation_offsets};

/// A small but structurally complete snapshot: two layers, two records,
/// mixed dtypes — enough that the record loop runs more than once and a
/// truncation can land inside the second record's payload.
fn sample() -> (KvManifest, Vec<ArrayRecord>) {
    let manifest = KvManifest {
        model_fingerprint: "test-fingerprint".into(),
        created_at_unix: 1_700_000_000,
        position: 42,
        prefix_tokens: vec![1, 2, 3, 4],
        last_token: Some(5),
        is_deep: true,
        layers: vec![
            LayerMeta::new(LayerKindTag::Full, 0),
            LayerMeta::new(LayerKindTag::Linear, 1),
        ],
    };
    let records = vec![
        ArrayRecord {
            dtype: DtypeTag::F32,
            shape: vec![2, 4],
            data: (0..8u32).flat_map(|i| (i as f32).to_le_bytes()).collect(),
        },
        ArrayRecord {
            dtype: DtypeTag::BF16,
            // BF16 records are stored in their native 2 bytes per element.
            shape: vec![3],
            data: (0..3u16).flat_map(|i| (0x3f80 + i).to_le_bytes()).collect(),
        },
    ];
    (manifest, records)
}

fn encode(manifest: &KvManifest, records: &[ArrayRecord]) -> Vec<u8> {
    let mut buf = Vec::new();
    write_lkv(&mut buf, manifest, records).expect("valid snapshot must encode");
    buf
}

/// Round-trips cleanly, and — the part that matters for the sweep below — this
/// is the "known good" probe re-run after every corrupt input.
#[test]
fn valid_snapshot_round_trips() {
    let (m, r) = sample();
    let bytes = encode(&m, &r);
    let (m2, r2) = read_lkv(&mut bytes.as_slice()).expect("round trip");
    assert_eq!(m2.model_fingerprint, m.model_fingerprint);
    assert_eq!(m2.position, m.position);
    assert_eq!(m2.prefix_tokens, m.prefix_tokens);
    assert_eq!(m2.layers.len(), m.layers.len());
    assert_eq!(r2.len(), r.len());
    assert_eq!(r2[0].data, r[0].data);
    assert_eq!(r2[1].shape, r[1].shape);
}

/// Every truncation offset must be a typed `Err`, and the reader must stay
/// usable afterwards.
#[test]
fn every_truncation_errors_without_panicking() {
    let (m, r) = sample();
    let bytes = encode(&m, &r);

    // Structural boundaries of LKV2: magic(4) version(4) json_len(4) json
    // json_crc(4) record_count(4) then records. ±1 around each is where length
    // checks actually go wrong.
    let json_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let boundaries = [4, 8, 12, 12 + json_len, 16 + json_len, 20 + json_len];
    let offsets = truncation_offsets(bytes.len(), 1, &boundaries);
    assert!(
        offsets.len() > 100,
        "sweep should be dense over a {}-byte file, got {} offsets",
        bytes.len(),
        offsets.len()
    );

    lumen_testkit::cases(offsets.len(), "kv-disk truncation sweep");
    for off in offsets {
        let bad = corrupt(&bytes, Corruption::TruncateAt(off));
        let res = read_lkv(&mut bad.as_slice());
        assert!(
            res.is_err(),
            "truncation at {off}/{} parsed as valid — a partial write must never \
             read back as a complete snapshot",
            bytes.len()
        );
        // Post-failure integrity: the good input still works.
        assert!(
            read_lkv(&mut bytes.as_slice()).is_ok(),
            "reader broke after a truncated input at {off}"
        );
    }
}

/// Every single flipped byte must be a typed error. LKV1 had no checksum, so a
/// flip inside a manifest string or a payload read back as a different but
/// "valid" snapshot — restored KV that silently disagrees with the prompt. LKV2
/// covers the manifest and every record with a CRC-32 and rejects trailing
/// bytes, which leaves no position where a flip is benign.
#[test]
fn every_single_byte_flip_is_rejected() {
    let (m, r) = sample();
    let bytes = encode(&m, &r);
    lumen_testkit::cases(bytes.len(), "kv-disk single-byte-flip sweep");
    for at in 0..bytes.len() {
        let bad = corrupt(&bytes, Corruption::FlipByte(at));
        assert!(
            read_lkv(&mut bad.as_slice()).is_err(),
            "a flip at byte {at}/{} read back as a valid snapshot",
            bytes.len()
        );
    }
}

/// Bytes after the last record mean the record count was corrupted downward
/// (or two writes interleaved); either way the file is not the snapshot its
/// header describes.
#[test]
fn trailing_bytes_are_rejected() {
    let (m, r) = sample();
    let mut bytes = encode(&m, &r);
    bytes.push(0);
    let msg = format!(
        "{:#}",
        read_lkv(&mut bytes.as_slice()).expect_err("trailing bytes must error")
    );
    assert!(msg.contains("trailing"), "got: {msg}");
}

/// A garbage manifest region must be a clean deserialize error.
#[test]
fn garbage_manifest_errors() {
    let (m, r) = sample();
    let bytes = encode(&m, &r);
    // LKV2's json_len is at offset 8, not 0, so drive the corruption manually
    // rather than through the safetensors-shaped `GarbageHeader` helper.
    let json_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let mut bad = bytes.clone();
    bad[12..12 + json_len].fill(b'#');
    let err = read_lkv(&mut bad.as_slice()).expect_err("garbage manifest must error");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("manifest"),
        "error should name the manifest, got: {msg}"
    );
}

/// Bad magic and bad version are the two cheapest rejections, and both must
/// name what they saw so a stale-format file is diagnosable from the log alone.
#[test]
fn bad_magic_and_version_are_named() {
    let (m, r) = sample();
    let bytes = encode(&m, &r);

    let mut bad_magic = bytes.clone();
    bad_magic[..4].copy_from_slice(b"XXXX");
    let msg = format!(
        "{:#}",
        read_lkv(&mut bad_magic.as_slice()).expect_err("bad magic must error")
    );
    assert!(msg.contains("magic"), "got: {msg}");

    let mut bad_version = bytes.clone();
    bad_version[4..8].copy_from_slice(&99u32.to_le_bytes());
    let msg = format!(
        "{:#}",
        read_lkv(&mut bad_version.as_slice()).expect_err("bad version must error")
    );
    assert!(msg.contains("version") && msg.contains("99"), "got: {msg}");
}

/// **Allocation-bomb guard.** `ArrayRecord::read_from` reads a `u64` payload
/// length off the wire and allocates it. `ndim` is already bounded (`> 8` is
/// rejected); `data_len` must be too, or a corrupt 8-byte field turns into a
/// multi-exabyte allocation request — an abort or an OOM-kill instead of a
/// typed error, on a path whose whole job is surviving a bad file.
///
/// The bound has to be a *format* limit rather than a heuristic: the payload
/// cannot exceed the bytes actually remaining in the file, and this test pins
/// that reasoning by declaring a preposterous length in a small file.
#[test]
fn implausible_record_length_is_rejected_not_allocated() {
    let (m, r) = sample();
    let bytes = encode(&m, &r);
    let json_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;

    // First record starts after: magic(4) version(4) json_len(4) json json_crc(4)
    // record_count(4). Layout within a record: dtype(1) ndim(4) shape(ndim*4)
    // data_len(8) data crc(4).
    let rec0 = 12 + json_len + 8;
    let ndim = u32::from_le_bytes(bytes[rec0 + 1..rec0 + 5].try_into().unwrap()) as usize;
    let data_len_at = rec0 + 1 + 4 + ndim * 4;

    for claimed in [u64::MAX, u64::MAX / 2, 1 << 40] {
        let mut bad = bytes.clone();
        bad[data_len_at..data_len_at + 8].copy_from_slice(&claimed.to_le_bytes());
        let res = read_lkv(&mut bad.as_slice());
        assert!(
            res.is_err(),
            "a record claiming {claimed} bytes inside a {}-byte file must be rejected \
             before any allocation",
            bytes.len()
        );
    }
}
