//! Write-side fault sweep over the KV-disk path (005 Phase 3, `FailingDisk`).
//!
//! `kv_disk_faults.rs` corrupts bytes that are already on disk. This asks the
//! other half of the question: **what does this code put on disk when the write
//! itself fails?** A full disk, an exceeded quota, an ejected volume and a
//! killed process all cut a write somewhere in the middle, and the invariant
//! SQLite states for that case is the one that matters here too — *committed or
//! completely rolled back*. Applied to a cache rather than a database, that
//! reads: after a failed write, a later read must find either the previous
//! snapshot or nothing, never a half-written one.
//!
//! Two levels are swept, because they can fail independently:
//!
//! * **Format** — `write_lkv` into a sink that dies after N bytes, for every N.
//!   Each partial output is fed back to `read_lkv`, which must reject all of
//!   them. This is the differential that makes the truncation sweep in
//!   `kv_disk_faults.rs` mean something: that file proves the reader rejects
//!   *truncations*, this one proves the writer cannot produce anything outside
//!   that rejected set.
//! * **Store** — `DiskKvStore::put` with its temp path blocked by a directory,
//!   which is a genuine `IsADirectory` failure from the real filesystem rather
//!   than an injected mock. The store must report it, keep the prior value
//!   readable, and stay usable for other keys.

use lumen_mlx::kv_disk::{
    ArrayRecord, DiskKvStore, DtypeTag, KvManifest, LayerKindTag, LayerMeta, read_lkv, write_lkv,
};
use lumen_testkit::faults::{FailingReader, FailingWriter, block_path_with_directory};

/// The store keys its directory on this AND `get` discards any snapshot whose
/// manifest disagrees with it, so the fixture must use one value for both. A
/// mismatch is not a fault worth injecting here: it makes every read a miss
/// with no error at all, which would make the sweeps below silently vacuous.
const FINGERPRINT: &str = "write-fault-fingerprint";

fn sample() -> (KvManifest, Vec<ArrayRecord>) {
    let manifest = KvManifest {
        model_fingerprint: FINGERPRINT.into(),
        created_at_unix: 1_700_000_000,
        position: 7,
        prefix_tokens: vec![10, 20, 30],
        last_token: Some(40),
        is_deep: false,
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
            dtype: DtypeTag::F32,
            shape: vec![3],
            data: (0..3u32).flat_map(|i| (i as f32).to_le_bytes()).collect(),
        },
    ];
    (manifest, records)
}

fn encoded_len() -> usize {
    let (m, r) = sample();
    let mut buf = Vec::new();
    write_lkv(&mut buf, &m, &r).expect("valid snapshot encodes");
    buf.len()
}

/// The sweep: fail the write after every possible byte count.
///
/// Two assertions per N, and the second is the load-bearing one — an `Err` from
/// the writer is worth little if the bytes it already emitted would read back
/// as a complete snapshot on the next run.
#[test]
fn every_partial_write_errors_and_is_unreadable() {
    let (m, r) = sample();
    let full = encoded_len();
    assert!(full > 100, "fixture too small to sweep: {full} bytes");

    lumen_testkit::cases(full, "kv-disk partial-write sweep");
    for budget in 0..full {
        let mut w = FailingWriter::new(budget);
        let res = write_lkv(&mut w, &m, &r);
        assert!(
            res.is_err(),
            "a sink that died after {budget}/{full} bytes reported success"
        );

        let partial = w.into_bytes();
        assert!(
            partial.len() <= budget,
            "the writer emitted {} bytes past a {budget}-byte budget",
            partial.len()
        );
        assert!(
            read_lkv(&mut partial.as_slice()).is_err(),
            "the {} bytes left behind by a write that failed at {budget} read back as a \
             COMPLETE snapshot — a partial write would be served as cache data",
            partial.len()
        );
    }

    // The full budget must still succeed, or the sweep proves nothing.
    let mut w = FailingWriter::new(full);
    write_lkv(&mut w, &m, &r).expect("a sink with room for the whole snapshot must succeed");
    assert!(read_lkv(&mut w.into_bytes().as_slice()).is_ok());
}

/// A mid-stream read **error** (as opposed to EOF) must propagate as a typed
/// `Err`. `read_exact` reports a short read and an I/O error differently, and
/// only one of them is covered by the truncation sweep.
#[test]
fn a_failing_reader_propagates_instead_of_hanging() {
    let (m, r) = sample();
    let mut buf = Vec::new();
    write_lkv(&mut buf, &m, &r).expect("encode");

    for budget in [0usize, 1, 4, 8, 12, 40, buf.len() / 2, buf.len() - 1] {
        let mut rd = FailingReader::new(buf.clone(), budget);
        assert!(
            read_lkv(&mut rd).is_err(),
            "a reader that failed after {budget} bytes produced a valid snapshot"
        );
    }
}

/// Store level: `put` cannot write its temp file, so the entry must not appear
/// and the store must survive.
#[test]
fn a_failed_put_leaves_no_entry_and_the_store_usable() {
    let root = tempfile::tempdir().expect("tempdir");
    let (m, r) = sample();
    let mut store = DiskKvStore::open(root.path(), FINGERPRINT, 0, 0).expect("open");

    // Establish a good entry first — this is the value that must survive.
    store.put("good", &m, &r).expect("first put");
    assert!(store.get("good").expect("get").is_some());

    // Block the temp path `put` writes through. Real filesystem, real error.
    let blocked_key = "blocked";
    let dir = root.path().join(FINGERPRINT);
    block_path_with_directory(dir.join(format!("{blocked_key}.lkv.tmp")));

    let res = store.put(blocked_key, &m, &r);
    assert!(
        res.is_err(),
        "put succeeded despite being unable to write its temp file"
    );
    let msg = format!("{:#}", res.unwrap_err());
    assert!(
        msg.contains("kv_disk") && msg.contains("write"),
        "the error must name the failing write so a full disk is diagnosable: {msg}"
    );

    // Committed-or-rolled-back: the failed key must not be readable, and the
    // previously stored one must be untouched.
    assert!(
        store
            .get(blocked_key)
            .expect("get after failed put")
            .is_none(),
        "a key whose write failed is readable — the cache would serve a snapshot \
         that was never fully written"
    );
    let (m2, r2) = store
        .get("good")
        .expect("get good")
        .expect("the pre-existing entry must survive an unrelated failed put");
    assert_eq!(m2.position, m.position);
    assert_eq!(r2.len(), r.len());

    // Post-failure integrity: the store still works for a fresh key.
    store
        .put("after", &m, &r)
        .expect("store unusable after a failed put");
    assert!(store.get("after").expect("get after").is_some());
}

/// A failed `put` must not silently double-count against the LRU budget, and
/// must not leave the index claiming bytes that are not there.
#[test]
fn a_failed_put_does_not_corrupt_the_byte_accounting() {
    let root = tempfile::tempdir().expect("tempdir");
    let (m, r) = sample();
    let mut store = DiskKvStore::open(root.path(), FINGERPRINT, 0, 0).expect("open");
    store.put("a", &m, &r).expect("put a");
    let bytes_before = store.total_bytes();
    let len_before = store.len();

    let dir = root.path().join(FINGERPRINT);
    block_path_with_directory(dir.join("b.lkv.tmp"));
    assert!(store.put("b", &m, &r).is_err());

    assert_eq!(
        store.len(),
        len_before,
        "a failed put added an index entry for a file that does not exist"
    );
    assert_eq!(
        store.total_bytes(),
        bytes_before,
        "a failed put charged bytes against the LRU budget that were never written"
    );

    // And the accounting still reflects reality after reopening from disk.
    drop(store);
    let mut reopened = DiskKvStore::open(root.path(), FINGERPRINT, 0, 0).expect("reopen");
    assert_eq!(reopened.len(), len_before);
    assert!(reopened.get("a").expect("get a").is_some());
    assert!(reopened.get("b").expect("get b").is_none());
}

/// The index is the store's own metadata; when *it* cannot be written the
/// failure must surface rather than leaving memory and disk disagreeing
/// forever.
///
/// This case also pins the one asymmetry in `put`: the index is saved *after*
/// the payload has been renamed into place, so an index write that fails leaves
/// a `.lkv` file on disk that the next process will not know about. The
/// assertion below records that as the deliberate ordering it is — the payload
/// must land before it is advertised, and the alternative (advertise first)
/// would let a reader chase an entry whose file does not exist. The orphan is
/// bounded: `get` on an unknown key is a plain miss, and the file is
/// overwritten the next time the same key is stored.
#[test]
fn a_failed_index_write_is_reported_and_orphans_at_most_the_payload() {
    let root = tempfile::tempdir().expect("tempdir");
    let (m, r) = sample();
    let mut store = DiskKvStore::open(root.path(), FINGERPRINT, 0, 0).expect("open");

    let dir = root.path().join(FINGERPRINT);
    block_path_with_directory(dir.join("index.json.tmp"));

    let res = store.put("k", &m, &r);
    assert!(
        res.is_err(),
        "put reported success while its index write was impossible — the entry would \
         vanish on restart with no indication anything went wrong"
    );

    // Reopening reads the on-disk index, which never got the entry.
    drop(store);
    let mut reopened = DiskKvStore::open(root.path(), FINGERPRINT, 0, 0).expect("reopen");
    assert!(
        reopened.get("k").expect("get k").is_none(),
        "the entry survived an index write that failed — memory and disk disagree"
    );
    assert_eq!(reopened.len(), 0);
    assert_eq!(reopened.total_bytes(), 0);

    // Storing the same key again must succeed once the obstruction is gone and
    // must not be confused by the orphaned payload.
    std::fs::remove_dir(dir.join("index.json.tmp")).expect("unblock");
    reopened
        .put("k", &m, &r)
        .expect("store usable after unblocking");
    assert!(reopened.get("k").expect("get k").is_some());
    assert_eq!(reopened.len(), 1);
}
