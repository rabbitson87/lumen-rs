//! Store-policy coverage for the KV-disk cache (005 Phase 4.1).
//!
//! `kv_disk_faults.rs` and `kv_disk_write_faults.rs` cover the *format* — what
//! happens to corrupt bytes and interrupted writes. What neither touches is the
//! **policy** around it: TTL expiry, LRU eviction, key sanitization and the
//! fingerprint check. Those were the module's remaining one-sided branches, and
//! they share a property that makes them worth the tests: every one fails
//! *quietly*.
//!
//! A cache that never evicts fills the disk. A cache that evicts the wrong
//! entry is slow, not wrong, so nothing reports it. A fingerprint check that
//! stops working serves one model's KV to another — the output stays fluent and
//! is nonsense. And `file_for` maps a cache key onto a filename, so if its
//! sanitization lapses a crafted key writes outside the cache directory.
//!
//! None of these produce an error anywhere. They are exactly the class this
//! task exists to make visible.

use lumen_mlx::kv_disk::{ArrayRecord, DiskKvStore, DtypeTag, KvManifest, LayerKindTag, LayerMeta};

const FP: &str = "policy-fingerprint";

fn manifest_for(fp: &str, position: usize) -> KvManifest {
    KvManifest {
        model_fingerprint: fp.into(),
        created_at_unix: 1_700_000_000,
        position,
        prefix_tokens: vec![1, 2, 3],
        last_token: Some(4),
        is_deep: false,
        layers: vec![LayerMeta::new(LayerKindTag::Full, 0)],
    }
}

/// `n_words` f32 words of payload — the knob for making an entry big enough to
/// blow a byte budget.
fn records(n_words: u32) -> Vec<ArrayRecord> {
    vec![ArrayRecord {
        dtype: DtypeTag::F32,
        shape: vec![n_words as i32],
        data: (0..n_words)
            .flat_map(|i| (i as f32).to_le_bytes())
            .collect(),
    }]
}

// ─────────────────────────── record shape ───────────────────────────

/// An empty shape is a scalar, and `elem_count` must report 0 rather than the
/// empty product of 1 — the length check derives the expected payload from it,
/// so a 1 here would demand four bytes that a scalar record does not carry.
#[test]
fn an_empty_shape_counts_zero_elements() {
    let rec = ArrayRecord {
        dtype: DtypeTag::F32,
        shape: Vec::new(),
        data: vec![],
    };
    assert_eq!(rec.elem_count(), 0);

    // And it survives a round-trip rather than being rejected as malformed.
    let m = manifest_for(FP, 1);
    let mut buf = Vec::new();
    lumen_mlx::kv_disk::write_lkv(&mut buf, &m, std::slice::from_ref(&rec)).expect("encode");
    let (_, back) = lumen_mlx::kv_disk::read_lkv(&mut buf.as_slice()).expect("decode");
    assert_eq!(back.len(), 1);
    assert!(back[0].shape.is_empty());
}

// ─────────────────────────── key sanitization ───────────────────────────

/// A cache key becomes a filename, so anything that is not
/// `[A-Za-z0-9_-]` has to be replaced. Without that a key containing `/` or
/// `..` writes outside the cache directory — a path traversal driven by
/// whatever the caller chose as a key.
#[test]
fn hostile_keys_cannot_escape_the_cache_directory() {
    let root = tempfile::tempdir().expect("tempdir");
    let mut store = DiskKvStore::open(root.path(), FP, 0, 0).expect("open");
    let dir = root.path().join(FP);

    let hostile = [
        "../../etc/passwd",
        "/absolute/path",
        "with spaces",
        "semi;colon",
        "quote\"quote",
        "새로운-키",
        "dot.dot",
    ];
    for key in hostile {
        store
            .put(key, &manifest_for(FP, 1), &records(4))
            .unwrap_or_else(|e| panic!("put({key:?}) must succeed: {e:#}"));
        assert!(
            store.get(key).expect("get").is_some(),
            "a sanitized key must still round-trip: {key:?}"
        );
    }

    // Every file the store created lives directly in its own directory.
    for entry in std::fs::read_dir(&dir).expect("read cache dir") {
        let p = entry.expect("entry").path();
        assert_eq!(
            p.parent(),
            Some(dir.as_path()),
            "the store wrote outside its directory: {}",
            p.display()
        );
        assert!(p.is_file(), "unexpected subdirectory: {}", p.display());
    }
    // Nothing landed above the cache root either.
    assert!(!root.path().join("etc").exists());
}

// ─────────────────────────── fingerprint ───────────────────────────

/// A snapshot whose manifest names a different model must be discarded on read.
/// Serving it would splice one model's KV into another's attention — fluent,
/// wrong output with nothing logged.
#[test]
fn a_snapshot_from_another_model_is_discarded_not_served() {
    let root = tempfile::tempdir().expect("tempdir");
    let mut store = DiskKvStore::open(root.path(), FP, 0, 0).expect("open");

    store
        .put("k", &manifest_for("a-different-model", 7), &records(4))
        .expect("put");
    assert!(
        store.get("k").expect("get").is_none(),
        "a foreign-fingerprint snapshot must read as a miss"
    );
    // ...and the bad entry is dropped rather than retried forever.
    assert_eq!(store.len(), 0);
    assert_eq!(store.total_bytes(), 0);

    // The matching fingerprint still works, so the check is not just "always
    // miss".
    store
        .put("k", &manifest_for(FP, 7), &records(4))
        .expect("put");
    let (m, _) = store
        .get("k")
        .expect("get")
        .expect("matching fingerprint hits");
    assert_eq!(m.position, 7);
}

// ─────────────────────────── TTL ───────────────────────────

/// TTL is keyed on last access and pruned at open, which is the path that runs
/// after a restart. Both arms matter: expired entries go, fresh ones stay, and
/// a store that pruned everything would silently make the cache useless.
#[test]
fn ttl_prunes_only_the_stale_entries_and_does_it_across_a_reopen() {
    let root = tempfile::tempdir().expect("tempdir");
    let ttl = 3600u64;

    // Write with TTL disabled so nothing is pruned on the way in.
    {
        let mut store = DiskKvStore::open(root.path(), FP, 0, 0).expect("open");
        store
            .put("fresh", &manifest_for(FP, 1), &records(4))
            .expect("put");
        store
            .put("stale", &manifest_for(FP, 2), &records(4))
            .expect("put");
        assert_eq!(store.len(), 2);
    }

    // Backdate one entry's last access past the TTL, in the index the store
    // will read at open. Editing the sidecar is how a real restart-after-a-day
    // presents itself, and it needs no clock manipulation.
    let index_path = root.path().join(FP).join("index.json");
    let mut index: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&index_path).expect("read index")).expect("parse");
    let now = index["entries"][0]["last_access_unix"]
        .as_u64()
        .expect("ts");
    for e in index["entries"].as_array_mut().expect("entries") {
        if e["key"] == "stale" {
            e["last_access_unix"] = serde_json::json!(now.saturating_sub(ttl * 2));
        }
    }
    std::fs::write(&index_path, serde_json::to_vec(&index).unwrap()).expect("write index");

    // Reopening with the TTL enabled prunes exactly one.
    let mut store = DiskKvStore::open(root.path(), FP, 0, ttl).expect("reopen");
    assert_eq!(store.len(), 1, "exactly the stale entry should have gone");
    assert!(store.get("fresh").expect("get").is_some());
    assert!(store.get("stale").expect("get").is_none());

    // The pruned entry's file is gone too, or the budget would count bytes
    // nothing can ever read.
    let files: Vec<_> = std::fs::read_dir(root.path().join(FP))
        .expect("read dir")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.ends_with(".lkv"))
        .collect();
    assert_eq!(
        files.len(),
        1,
        "the stale payload should be deleted: {files:?}"
    );
}

/// TTL of 0 disables expiry entirely — an entry older than any plausible TTL
/// must survive. This is the `ttl_secs == 0` early return, and getting it wrong
/// would make every restart drop the whole cache.
#[test]
fn a_zero_ttl_never_expires_anything() {
    let root = tempfile::tempdir().expect("tempdir");
    {
        let mut store = DiskKvStore::open(root.path(), FP, 0, 0).expect("open");
        store
            .put("k", &manifest_for(FP, 1), &records(4))
            .expect("put");
    }
    let index_path = root.path().join(FP).join("index.json");
    let mut index: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&index_path).expect("read")).expect("parse");
    index["entries"][0]["last_access_unix"] = serde_json::json!(1u64);
    std::fs::write(&index_path, serde_json::to_vec(&index).unwrap()).expect("write");

    let mut store = DiskKvStore::open(root.path(), FP, 0, 0).expect("reopen");
    assert_eq!(store.len(), 1, "ttl=0 must not expire a 1970 timestamp");
    assert!(store.get("k").expect("get").is_some());
}

// ─────────────────────────── LRU eviction ───────────────────────────

/// The byte budget evicts least-recently-used entries until the total fits.
/// Untested, a cache that never evicts fills the disk and nothing reports it.
#[test]
fn the_byte_budget_evicts_the_least_recently_used_entry() {
    let root = tempfile::tempdir().expect("tempdir");

    // Measure one entry rather than guessing: the on-disk size is payload plus
    // a manifest whose length depends on this fixture, and a hard-coded budget
    // would make the test assert about the fixture instead of the policy.
    let probe = tempfile::tempdir().expect("tempdir");
    let entry_bytes = {
        let mut s = DiskKvStore::open(probe.path(), FP, 0, 0).expect("open probe");
        s.put("x", &manifest_for(FP, 1), &records(256))
            .expect("put");
        s.total_bytes()
    };
    assert!(entry_bytes > 0);
    let budget = entry_bytes * 2 + entry_bytes / 2; // room for exactly two

    let mut store = DiskKvStore::open(root.path(), FP, budget, 0).expect("open");
    store
        .put("a", &manifest_for(FP, 1), &records(256))
        .expect("put a");
    store
        .put("b", &manifest_for(FP, 2), &records(256))
        .expect("put b");
    assert_eq!(store.len(), 2, "a {budget}-byte budget should hold two");

    store
        .put("c", &manifest_for(FP, 3), &records(256))
        .expect("put c");
    assert!(
        store.total_bytes() <= budget,
        "eviction must bring the store back under {budget}, got {}",
        store.total_bytes()
    );
    assert_eq!(store.len(), 2, "exactly one entry should have been evicted");
    assert!(
        store.get("c").expect("get c").is_some(),
        "the entry just written must survive its own eviction pass"
    );
    assert!(
        store.get("a").expect("get a").is_none(),
        "the least-recently-used entry is the one that goes"
    );
}

/// A budget of 0 means unbounded — the early return. A store that treated 0 as
/// "evict everything" would make the disk tier a no-op that still costs writes.
#[test]
fn a_zero_budget_is_unbounded_not_empty() {
    let root = tempfile::tempdir().expect("tempdir");
    let mut store = DiskKvStore::open(root.path(), FP, 0, 0).expect("open");
    for i in 0..6u32 {
        store
            .put(
                &format!("k{i}"),
                &manifest_for(FP, i as usize),
                &records(256),
            )
            .expect("put");
    }
    assert_eq!(store.len(), 6, "budget 0 must not evict");
    assert!(store.total_bytes() > 2_500);
}

/// The eviction loop stops at one entry even when that entry alone exceeds the
/// budget: emptying the cache to satisfy a budget it cannot satisfy would throw
/// away work for nothing. Pinned so the `> 1` guard is not "simplified" away.
#[test]
fn eviction_keeps_the_last_entry_even_when_it_busts_the_budget() {
    let root = tempfile::tempdir().expect("tempdir");
    let mut store = DiskKvStore::open(root.path(), FP, 100, 0).expect("open");
    store
        .put("big", &manifest_for(FP, 1), &records(256))
        .expect("put");
    assert_eq!(store.len(), 1, "a single oversized entry is kept");
    assert!(store.total_bytes() > 100, "and it does exceed the budget");
    assert!(store.get("big").expect("get").is_some());
}

// ─────────────────────────── index recovery ───────────────────────────

/// A missing index is a fresh cache, not an error. A *corrupt* one is treated
/// the same way — better an empty cache than a failed open, because the payload
/// files are regenerable and the process needs to start.
#[test]
fn a_missing_or_corrupt_index_opens_as_an_empty_cache() {
    let root = tempfile::tempdir().expect("tempdir");
    {
        let mut store = DiskKvStore::open(root.path(), FP, 0, 0).expect("open");
        store
            .put("k", &manifest_for(FP, 1), &records(4))
            .expect("put");
    }
    let index_path = root.path().join(FP).join("index.json");

    std::fs::write(&index_path, b"{ not json").expect("corrupt the index");
    let store = DiskKvStore::open(root.path(), FP, 0, 0).expect("a corrupt index must still open");
    assert_eq!(store.len(), 0);

    std::fs::remove_file(&index_path).expect("remove");
    let store = DiskKvStore::open(root.path(), FP, 0, 0).expect("a missing index must still open");
    assert!(store.is_empty());
}

/// An index path that cannot be read *for a reason other than absence* is a
/// real I/O failure and must surface, not be swallowed as "empty cache" —
/// silently discarding a cache the operator can see on disk is the confusing
/// outcome. A directory in the index's place produces exactly that error.
#[test]
fn an_unreadable_index_is_an_error_rather_than_a_silent_empty_cache() {
    let root = tempfile::tempdir().expect("tempdir");
    let dir = root.path().join(FP);
    std::fs::create_dir_all(dir.join("index.json")).expect("block the index path");

    let Err(err) = DiskKvStore::open(root.path(), FP, 0, 0) else {
        panic!("an unreadable index must not be mistaken for an absent one")
    };
    let msg = format!("{err:#}");
    assert!(
        msg.contains("index"),
        "the error must name the index so the cause is findable: {msg}"
    );
}

/// Key sanitization keeps `-` and `_` verbatim — they are the two
/// non-alphanumeric characters the real key format uses (`auto-<hex>`), so a
/// sanitizer that replaced them would collapse distinct keys onto one filename
/// and make two conversations share a KV snapshot.
#[test]
fn sanitization_preserves_the_characters_real_keys_use() {
    let root = tempfile::tempdir().expect("tempdir");
    let mut store = DiskKvStore::open(root.path(), FP, 0, 0).expect("open");

    // Two keys that differ ONLY in a preserved character must not collide.
    for (a, b) in [("auto-abc", "auto_abc"), ("k-1", "k-2"), ("x_y", "x-y")] {
        store
            .put(a, &manifest_for(FP, 1), &records(4))
            .expect("put a");
        store
            .put(b, &manifest_for(FP, 2), &records(4))
            .expect("put b");
        let (ma, _) = store.get(a).expect("get a").expect("a present");
        let (mb, _) = store.get(b).expect("get b").expect("b present");
        assert_ne!(
            ma.position, mb.position,
            "{a:?} and {b:?} collapsed onto one file — two conversations would \
             share a KV snapshot"
        );
    }
}

/// Removing a key that was never stored is a no-op, not an error. The eviction
/// and miss paths both call it speculatively, so an error here would turn a
/// cache miss into a failed request.
#[test]
fn removing_an_absent_key_is_a_no_op() {
    let root = tempfile::tempdir().expect("tempdir");
    let mut store = DiskKvStore::open(root.path(), FP, 0, 0).expect("open");
    store
        .remove("never-stored")
        .expect("removing an absent key must succeed");
    assert!(store.is_empty());

    store
        .put("k", &manifest_for(FP, 1), &records(4))
        .expect("put");
    store
        .remove("other")
        .expect("still a no-op with entries present");
    assert_eq!(
        store.len(),
        1,
        "an unrelated remove must not touch the store"
    );
    assert!(store.get("k").expect("get").is_some());

    // And clear() on an empty store is equally quiet. The tempdir must be
    // BOUND: `tempfile::tempdir().path()` drops the guard at the end of the
    // statement and deletes the directory out from under the store.
    let fresh_root = tempfile::tempdir().expect("tempdir");
    let mut fresh = DiskKvStore::open(fresh_root.path(), FP, 0, 0).expect("open");
    fresh.clear().expect("clear on an empty store");
    assert!(fresh.is_empty());
}
