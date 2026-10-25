//! P0 — on-disk KV-cache persistence primitives (shared core).
//!
//! This module is the backend-agnostic foundation for persisting prompt-cache
//! snapshots to disk (Phase 1: persistence across restart/idle/evict; Phase 3:
//! spill under memory pressure). It is intentionally split into two layers:
//!
//! * **Pure layer** (no MLX) — the on-disk binary format (`LKV2`), the
//!   `KvManifest`/`LayerMeta` metadata, the `ArrayRecord` byte container, and
//!   the `DiskKvStore` (directory + LRU budget + `index.json`). These compile
//!   and unit-test under the crate's default features (no Metal device needed),
//!   mirroring the `lloyd_max.rs` / `rotation.rs` TQCB/TQRM save-load pattern.
//!
//! * **MLX layer** (`#[cfg(feature = "mlx-native")]`) — conversion between an
//!   mlx-rs `Array` and an `ArrayRecord`. Every dtype is stored in its native
//!   little-endian width, so a bf16 KV cache costs 2 bytes per element on disk
//!   and in the read, not 4. These paths require a live Metal device, so their
//!   tests are `#[ignore]` like the rest of the native suite.
//!
//! The manifest and every record carry a CRC-32, and the reader rejects
//! trailing bytes, so any single corrupted byte is a typed error rather than a
//! silently different snapshot. A file in the older `LKV1` layout (f32-widened
//! floats, no checksums) fails the magic check and is dropped by the store like
//! any other unreadable entry; this is a cache, so that only costs a re-prefill.
//!
//! Snapshot coupling (turning a `PromptCacheSnapshot` into a `(KvManifest,
//! Vec<ArrayRecord>)` and back) lives in `native_snapshot.rs` (Qwen) and
//! `gemma4_moe.rs` (Gemma).

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};

/// File magic — `b"LKV2"` (Lumen KV, format generation 2: native dtypes and
/// CRC-32 per manifest and record).
pub const LKV_MAGIC: [u8; 4] = *b"LKV2";
/// Format version, bumped on incompatible layout changes.
pub const LKV_VERSION: u32 = 2;

// Records are raw little-endian bytes copied straight from (and into) native
// memory; a big-endian host would need a byte swap on every element.
const _: () = assert!(cfg!(target_endian = "little"));

// ---------------------------------------------------------------------------
// Dtype tagging
// ---------------------------------------------------------------------------

/// Element dtype recorded for an array: both the dtype it is reconstructed as
/// and its on-disk encoding (native width, little-endian).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DtypeTag {
    Bool,
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F16,
    F32,
    F64,
    BF16,
}

impl DtypeTag {
    /// Bytes per element on disk: the dtype's native width.
    #[inline]
    pub fn stored_elem_size(self) -> usize {
        match self {
            DtypeTag::Bool | DtypeTag::U8 | DtypeTag::I8 => 1,
            DtypeTag::U16 | DtypeTag::I16 | DtypeTag::F16 | DtypeTag::BF16 => 2,
            DtypeTag::F32 | DtypeTag::U32 | DtypeTag::I32 => 4,
            DtypeTag::U64 | DtypeTag::I64 | DtypeTag::F64 => 8,
        }
    }

    fn to_u8(self) -> u8 {
        match self {
            DtypeTag::Bool => 0,
            DtypeTag::U8 => 1,
            DtypeTag::U16 => 2,
            DtypeTag::U32 => 3,
            DtypeTag::U64 => 4,
            DtypeTag::I8 => 5,
            DtypeTag::I16 => 6,
            DtypeTag::I32 => 7,
            DtypeTag::I64 => 8,
            DtypeTag::F16 => 9,
            DtypeTag::F32 => 10,
            DtypeTag::F64 => 11,
            DtypeTag::BF16 => 12,
        }
    }

    fn from_u8(b: u8) -> Result<Self> {
        Ok(match b {
            0 => DtypeTag::Bool,
            1 => DtypeTag::U8,
            2 => DtypeTag::U16,
            3 => DtypeTag::U32,
            4 => DtypeTag::U64,
            5 => DtypeTag::I8,
            6 => DtypeTag::I16,
            7 => DtypeTag::I32,
            8 => DtypeTag::I64,
            9 => DtypeTag::F16,
            10 => DtypeTag::F32,
            11 => DtypeTag::F64,
            12 => DtypeTag::BF16,
            other => bail!("kv_disk: unknown dtype tag byte {other}"),
        })
    }
}

// ---------------------------------------------------------------------------
// ArrayRecord — a single tensor's physical bytes + shape + logical dtype
// ---------------------------------------------------------------------------

/// One serialized tensor. `data` holds the elements in `dtype`'s native width,
/// little-endian, in logical row-major order of `shape`.
#[derive(Debug, Clone, PartialEq)]
pub struct ArrayRecord {
    pub dtype: DtypeTag,
    pub shape: Vec<i32>,
    pub data: Vec<u8>,
}

impl ArrayRecord {
    /// Number of logical elements (product of shape, or 0 for empty shape).
    pub fn elem_count(&self) -> usize {
        if self.shape.is_empty() {
            return 0;
        }
        self.shape.iter().map(|&d| d.max(0) as usize).product()
    }

    /// Sanity: stored byte length must match shape × stored element size.
    fn validate_len(&self) -> Result<()> {
        let expected = self.elem_count() * self.dtype.stored_elem_size();
        if self.data.len() != expected {
            bail!(
                "kv_disk: ArrayRecord byte length {} != expected {} (shape {:?}, dtype {:?})",
                self.data.len(),
                expected,
                self.shape,
                self.dtype
            );
        }
        Ok(())
    }

    /// `[dtype][ndim][shape][data_len][data]` followed by a CRC-32 of all of it.
    fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        self.validate_len()?;
        let mut w = CrcWriter::new(w);
        w.write_all(&[self.dtype.to_u8()])?;
        w.write_all(&(self.shape.len() as u32).to_le_bytes())?;
        for &d in &self.shape {
            w.write_all(&d.to_le_bytes())?;
        }
        w.write_all(&(self.data.len() as u64).to_le_bytes())?;
        w.write_all(&self.data)?;
        w.finish()
    }

    fn read_from<R: Read>(r: &mut R) -> Result<Self> {
        let mut r = CrcReader::new(r);
        let rec = Self::read_body(&mut r)?;
        r.verify("record")?;
        Ok(rec)
    }

    fn read_body<R: Read>(r: &mut R) -> Result<Self> {
        let mut b1 = [0u8; 1];
        r.read_exact(&mut b1)?;
        let dtype = DtypeTag::from_u8(b1[0])?;

        let ndim = read_u32(r)? as usize;
        if ndim > 8 {
            bail!("kv_disk: implausible ndim {ndim} (corrupt record)");
        }
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            let mut sb = [0u8; 4];
            r.read_exact(&mut sb)?;
            let d = i32::from_le_bytes(sb);
            if d < 0 {
                bail!("kv_disk: negative dimension {d} (corrupt record)");
            }
            shape.push(d);
        }

        // Derive the payload size from shape × dtype and require the stored
        // length to match it EXACTLY, before allocating anything.
        //
        // This is a memory-safety boundary, not a tidiness check. `data_len`
        // arrives from a file that a previous process may have written
        // partially (power loss, full disk, SIGKILL mid-flush) or that disk rot
        // has touched; allocating it unvalidated turns a corrupt 8-byte field
        // into a multi-terabyte request, and an allocation failure *aborts* —
        // it is not a catchable panic, so no amount of caller-side error
        // handling saves the process. Observed as SIGABRT on a 280 TB request.
        //
        // `ndim` was already bounded above; this closes the same hole for the
        // payload. No heuristic cap is needed because the format is
        // self-describing: shape and dtype determine the size exactly, and a
        // record whose declared length disagrees is corrupt by definition.
        let expected = shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
            .and_then(|n| n.checked_mul(dtype.stored_elem_size()))
            .filter(|_| !shape.is_empty())
            .unwrap_or(0);
        let data_len = read_u64(r)? as usize;
        if data_len != expected {
            bail!(
                "kv_disk: record declares {data_len} payload bytes but shape {shape:?} \
                 with dtype {dtype:?} implies {expected} (corrupt record)"
            );
        }
        let mut data = vec![0u8; data_len];
        r.read_exact(&mut data)?;

        let rec = ArrayRecord { dtype, shape, data };
        rec.validate_len()?;
        Ok(rec)
    }
}

// ---------------------------------------------------------------------------
// Manifest — describes a persisted prompt-cache snapshot
// ---------------------------------------------------------------------------

/// Per-layer cache kind tag (mirrors `NativeLayerCache`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerKindTag {
    Full,
    Linear,
    FullTurboquant,
    /// Gemma4 sliding-window rotating cache (`NativeRotatingKvCache`). The
    /// full ring buffer is stored; `max_size`/`keep`/`idx` carry the rotation
    /// bookkeeping needed to reproduce circular reads.
    Sliding,
    /// Gemma4 affine-quantized full-attn cache (`NativeKvCacheQuantized`). K/V
    /// each a `(packed_uint32, scales, biases)` 3-tuple; `group_size`/`bits`
    /// carry the quant params. `arrays = [kp, ks, kb, vp, vs, vb]`.
    FullQuantized,
    /// Gemma4 affine-quantized sliding rotating cache
    /// (`NativeRotatingKvCacheQuantized`). Same 3-tuple layout + rotation
    /// bookkeeping (`max_size`/`keep`/`idx`) + `group_size`/`bits`.
    SlidingQuantized,
    /// Gemma4 TurboQuant sliding rotating cache (`NativeRotatingKvCacheTurboQuant`,
    /// same type as `FullTurboquant`, different attention window).
    SlidingTurboquant,
}

/// Metadata for a single cache layer. The actual tensors live in the blob's
/// `Vec<ArrayRecord>`; the `*_id` / `arrays` fields hold indices into it
/// (`None` = the corresponding Array was absent in the snapshot). Scalar
/// fields capture the bookkeeping a layer needs to be reconstructed.
///
/// Positional conventions per kind:
/// * `Full` — `arrays = [keys?, values?]`.
/// * `Linear` — `arrays = SSM state slots` (each may be `None`);
///   `lengths_id` / `left_padding_id` for the metadata arrays.
/// * `FullTurboquant` — `arrays = [keys_codes, keys_sigma, values_codes,
///   values_sigma, keys_signs?, keys_residual_norm?]`; rotating bookkeeping in
///   `max_size`/`keep`/`idx`/`bits`/`qjl_m` (wired in a later phase).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerMeta {
    pub kind: LayerKindTag,
    pub offset: usize,
    #[serde(default)]
    pub base_offset: usize,
    /// Per-slot record indices (positional per `kind`).
    #[serde(default)]
    pub arrays: Vec<Option<usize>>,
    /// Linear only — `lengths` metadata array index.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lengths_id: Option<usize>,
    /// Linear only — `left_padding` metadata array index.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub left_padding_id: Option<usize>,
    /// TurboQuant rotating window size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_size: Option<usize>,
    /// TurboQuant anchor-prefix size (never evicted).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keep: Option<usize>,
    /// TurboQuant circular write index.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idx: Option<usize>,
    /// TurboQuant Lloyd-Max bit width, or affine-quant bit width.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bits: Option<u32>,
    /// TurboQuant QJL stage-2 width.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qjl_m: Option<usize>,
    /// Affine-quant group size (`NativeKvCacheQuantized` / rotating quantized).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_size: Option<i32>,
}

impl LayerMeta {
    /// Construct a metadata stub for `kind` with all optional fields empty.
    pub fn new(kind: LayerKindTag, offset: usize) -> Self {
        Self {
            kind,
            offset,
            base_offset: 0,
            arrays: Vec::new(),
            lengths_id: None,
            left_padding_id: None,
            max_size: None,
            keep: None,
            idx: None,
            bits: None,
            qjl_m: None,
            group_size: None,
        }
    }
}

/// Top-level manifest persisted alongside the tensor blob.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvManifest {
    /// Hash of model weights + tokenizer + key config. A mismatch on load
    /// means the snapshot was produced by a different model ⇒ must be discarded.
    pub model_fingerprint: String,
    /// Wall-clock creation time (unix seconds) for LRU/TTL bookkeeping.
    pub created_at_unix: u64,
    /// Logical sequence position captured (snapshot boundary).
    pub position: usize,
    /// Token ids forming the cached prefix (boundary tokens).
    pub prefix_tokens: Vec<u32>,
    /// Cached argmax token at the prefix end, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_token: Option<u32>,
    /// Whether the snapshot held independent (deep) buffers when captured.
    #[serde(default)]
    pub is_deep: bool,
    /// Per-layer metadata, one entry per model layer.
    pub layers: Vec<LayerMeta>,
}

// ---------------------------------------------------------------------------
// File format: header + manifest(JSON) + record blob
// ---------------------------------------------------------------------------

/// Serialize a `(KvManifest, records)` pair to the `LKV2` binary layout.
///
/// Layout:
/// ```text
/// [4]  magic "LKV2"
/// [4]  version  u32 LE
/// [4]  manifest_json_len  u32 LE
/// [N]  manifest JSON (utf-8)
/// [4]  manifest CRC-32 (over the JSON bytes)
/// [4]  record_count  u32 LE
/// records[]: [1] dtype | [4] ndim | [ndim*4] shape i32 | [8] data_len u64 | [data_len] bytes
///            | [4] CRC-32 (over the record's preceding bytes)
/// ```
/// Nothing follows the last record; the reader rejects trailing bytes.
pub fn write_lkv<W: Write>(
    w: &mut W,
    manifest: &KvManifest,
    records: &[ArrayRecord],
) -> Result<()> {
    w.write_all(&LKV_MAGIC)?;
    w.write_all(&LKV_VERSION.to_le_bytes())?;

    let json = serde_json::to_vec(manifest).context("kv_disk: serialize manifest")?;
    w.write_all(&(json.len() as u32).to_le_bytes())?;
    w.write_all(&json)?;
    w.write_all(&crc32fast::hash(&json).to_le_bytes())?;

    w.write_all(&(records.len() as u32).to_le_bytes())?;
    for rec in records {
        rec.write_to(w)?;
    }
    Ok(())
}

/// Deserialize the `LKV2` binary layout back into `(KvManifest, records)`.
pub fn read_lkv<R: Read>(r: &mut R) -> Result<(KvManifest, Vec<ArrayRecord>)> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).context("kv_disk: read magic")?;
    if magic != LKV_MAGIC {
        bail!("kv_disk: bad magic {magic:?} (expected {LKV_MAGIC:?})");
    }
    let version = read_u32(r)?;
    if version != LKV_VERSION {
        bail!("kv_disk: unsupported version {version} (expected {LKV_VERSION})");
    }

    let json_len = read_u32(r)? as usize;
    let mut json = vec![0u8; json_len];
    r.read_exact(&mut json).context("kv_disk: read manifest")?;
    if read_u32(r)? != crc32fast::hash(&json) {
        bail!("kv_disk: manifest CRC mismatch (corrupt file)");
    }
    let manifest: KvManifest =
        serde_json::from_slice(&json).context("kv_disk: deserialize manifest")?;

    let n = read_u32(r)? as usize;
    let mut records = Vec::with_capacity(n.min(1024));
    for _ in 0..n {
        records.push(ArrayRecord::read_from(r)?);
    }
    let mut extra = [0u8; 1];
    if r.read(&mut extra).context("kv_disk: check end of file")? != 0 {
        bail!("kv_disk: trailing bytes after {n} records (corrupt record count)");
    }
    Ok((manifest, records))
}

/// Forwards writes and keeps a CRC-32 of everything written; `finish` appends it.
struct CrcWriter<'a, W: Write> {
    inner: &'a mut W,
    hasher: crc32fast::Hasher,
}

impl<'a, W: Write> CrcWriter<'a, W> {
    fn new(inner: &'a mut W) -> Self {
        Self {
            inner,
            hasher: crc32fast::Hasher::new(),
        }
    }

    fn finish(self) -> Result<()> {
        self.inner
            .write_all(&self.hasher.finalize().to_le_bytes())?;
        Ok(())
    }
}

impl<W: Write> Write for CrcWriter<'_, W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.hasher.update(&buf[..n]);
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Forwards reads and keeps a CRC-32 of everything read; `verify` reads the
/// stored CRC that follows and compares.
struct CrcReader<'a, R: Read> {
    inner: &'a mut R,
    hasher: crc32fast::Hasher,
}

impl<'a, R: Read> CrcReader<'a, R> {
    fn new(inner: &'a mut R) -> Self {
        Self {
            inner,
            hasher: crc32fast::Hasher::new(),
        }
    }

    fn verify(self, what: &str) -> Result<()> {
        let computed = self.hasher.finalize();
        if read_u32(self.inner)? != computed {
            bail!("kv_disk: {what} CRC mismatch (corrupt file)");
        }
        Ok(())
    }
}

impl<R: Read> Read for CrcReader<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.hasher.update(&buf[..n]);
        Ok(n)
    }
}

#[inline]
fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

#[inline]
fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

// ---------------------------------------------------------------------------
// DiskKvStore — directory + LRU budget + index.json
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DiskEntry {
    key: String,
    file: String,
    bytes: u64,
    last_access_unix: u64,
    hits: u64,
    /// The background write that makes this entry's file, until the writer
    /// reports it landed (see [`DiskKvStore::reconcile`]). In memory only.
    #[serde(skip)]
    pending: Option<u64>,
    /// The entry that replaces this one once its file is on the disk (see
    /// [`DiskKvStore::supersede`]). Kept in the index, so a process that stops
    /// before the replacement lands leaves the decision to the next open.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    superseded_by: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct DiskIndex {
    entries: Vec<DiskEntry>,
}

/// Disk-backed snapshot store rooted at `<root>/<fingerprint>/`. Enforces a
/// byte budget via LRU eviction. The `index.json` sidecar tracks per-entry
/// size and access time so eviction does not require statting every file.
///
/// This is the shared write/read path used by BOTH persistence (flush on
/// prefix-cache eviction) and spill (flush under memory pressure) — only the
/// trigger differs.
pub struct DiskKvStore {
    dir: PathBuf,
    max_bytes: u64,
    /// Time-to-live for entries, in seconds, keyed on last access. `0` disables
    /// TTL (entries kept until LRU/byte-budget eviction). Pruned on open + put.
    ttl_secs: u64,
    fingerprint: String,
    index: DiskIndex,
    /// Where file writes and deletes run: `None` performs them inline (the
    /// default, and what tests rely on); `Some` hands them to a background
    /// thread so a request never waits on the disk (see
    /// [`Self::write_in_background`]).
    writer: Option<FileWriter>,
    /// How files are created: the filesystem, or in a test a disk that fills up.
    create: CreateFile,
    /// What the writer thread reports back: which writes landed, and the
    /// failures of a disk that stopped taking them.
    report: std::sync::Arc<std::sync::Mutex<WriterReport>>,
    /// Number of the last write handed to the writer thread.
    last_seq: u64,
    /// Free space a write must leave on the store's volume, in bytes; `0`
    /// (what [`Self::open`] sets) means no floor. See [`Self::put`].
    min_free_bytes: u64,
    /// How the volume's free space is read: `statvfs`, or in a test a number.
    free_space: FreeSpace,
    /// Writes the floor skipped, for logging once per run of them.
    floor: FloorReport,
}

impl DiskKvStore {
    /// Open (creating if needed) a store under `root/<fingerprint>/`.
    /// `max_bytes` is the LRU budget (0 = unbounded). `ttl_secs` is the
    /// last-access TTL (0 = no expiry). Stale entries are pruned on open.
    /// No free-space floor until [`Self::set_min_free_bytes`] sets one.
    pub fn open(
        root: impl AsRef<Path>,
        fingerprint: impl Into<String>,
        max_bytes: u64,
        ttl_secs: u64,
    ) -> Result<Self> {
        let fingerprint = fingerprint.into();
        let dir = root.as_ref().join(&fingerprint);
        fs::create_dir_all(&dir)
            .with_context(|| format!("kv_disk: create dir {}", dir.display()))?;
        let index = Self::load_index(&dir)?;
        let mut store = Self {
            dir,
            max_bytes,
            ttl_secs,
            fingerprint,
            index,
            writer: None,
            create: real_files(),
            report: Default::default(),
            last_seq: 0,
            min_free_bytes: 0,
            free_space: std::sync::Arc::new(volume_free_bytes),
            floor: FloorReport::default(),
        };
        // Drop entries that expired while the process was down, and whatever a
        // process that died mid-write left behind: half-written temp files,
        // index rows whose file never landed, and files whose index row never
        // did (no key leads to them, so they would only take the space).
        let mut changed = store.prune_ttl();
        let before = store.index.entries.len();
        let dir = store.dir.clone();
        store.index.entries.retain(|e| dir.join(&e.file).exists());
        changed |= store.index.entries.len() != before;
        // A replacement that landed retires the entry it supersedes; one that
        // never did leaves it be.
        let listed: std::collections::HashSet<String> =
            store.index.entries.iter().map(|e| e.key.clone()).collect();
        let before = store.index.entries.len();
        store.index.entries.retain_mut(|e| match &e.superseded_by {
            Some(k) if listed.contains(k) => false,
            Some(_) => {
                e.superseded_by = None;
                changed = true;
                true
            }
            None => true,
        });
        changed |= store.index.entries.len() != before;
        if let Ok(files) = fs::read_dir(&store.dir) {
            let listed: std::collections::HashSet<&str> = store
                .index
                .entries
                .iter()
                .map(|e| e.file.as_str())
                .collect();
            for e in files.flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                let unlisted = name.ends_with(".lkv") && !listed.contains(name.as_str());
                if name.ends_with(".tmp") || unlisted {
                    let _ = fs::remove_file(e.path());
                }
            }
        }
        if changed {
            let _ = store.save_index();
        }
        Ok(store)
    }

    /// From now on, write and delete files on a background thread, in the
    /// order they were requested. A `get` for an entry whose file has not
    /// landed yet is a miss, as for any missing file. Dropping the store
    /// finishes the queued writes first.
    pub fn write_in_background(&mut self) -> Result<()> {
        if self.writer.is_none() {
            self.writer = Some(FileWriter::spawn(self.create.clone(), self.report.clone())?);
        }
        Ok(())
    }

    /// Tests: create files through `create` (a disk that fills up) from now on.
    /// Call before [`Self::write_in_background`], whose thread keeps the opener
    /// it started with.
    #[cfg(test)]
    fn set_create_for_test(&mut self, create: CreateFile) {
        assert!(
            self.writer.is_none(),
            "set the opener before the writer starts"
        );
        self.create = create;
    }

    /// Keep at least `bytes` free on the store's volume: a [`Self::put`] that
    /// would leave less is skipped (`0` turns the floor off). Entries already
    /// stored are never removed for it, and reads go on as before.
    pub fn set_min_free_bytes(&mut self, bytes: u64) {
        self.min_free_bytes = bytes;
    }

    /// Tests: read the volume's free space through `free_space` from now on.
    #[cfg(test)]
    fn set_free_space_for_test(&mut self, free_space: FreeSpace) {
        self.free_space = free_space;
    }

    /// Tests: `(writes skipped since the floor last let one through, lines logged)`.
    #[cfg(test)]
    fn floor_report_for_test(&self) -> (u64, u64) {
        (self.floor.skipped, self.floor.lines)
    }

    /// Write `bytes` to `path` through a temp file and a rename, so a reader
    /// never sees a partial file — inline, or queued on the background thread.
    /// Returns the number of a queued write, whose outcome the writer reports.
    fn write_file(&mut self, path: PathBuf, bytes: Vec<u8>) -> Result<Option<u64>> {
        match &self.writer {
            Some(w) => {
                self.last_seq += 1;
                let seq = self.last_seq;
                w.submit(FileJob::Write {
                    path,
                    bytes,
                    seq: Some(seq),
                });
                Ok(Some(seq))
            }
            None => write_via_tmp(&self.create, &path, &bytes).map(|()| None),
        }
    }

    /// Delete `path` (a missing file is fine) — inline, or queued behind any
    /// write of the same file so the delete cannot overtake it.
    fn delete_file(&self, path: PathBuf) {
        match &self.writer {
            Some(w) => w.submit(FileJob::Remove(path)),
            None => {
                let _ = fs::remove_file(path);
            }
        }
    }

    /// Remove entries whose last access is older than `ttl_secs` (deletes the
    /// `.lkv` files too). No-op when TTL is disabled. Returns whether anything
    /// was removed (so the caller can persist the index).
    fn prune_ttl(&mut self) -> bool {
        if self.ttl_secs == 0 {
            return false;
        }
        let now = now_unix();
        let ttl = self.ttl_secs;
        let before = self.index.entries.len();
        let mut removed_files = Vec::new();
        self.index.entries.retain(|e| {
            let expired = now.saturating_sub(e.last_access_unix) > ttl;
            if expired {
                removed_files.push(e.file.clone());
            }
            !expired
        });
        for f in &removed_files {
            self.delete_file(self.dir.join(f));
        }
        self.index.entries.len() != before
    }

    /// Default cache root: `$LUMEN_KV_DISK_DIR` or `~/.cache/lumen/kv`.
    pub fn default_root() -> PathBuf {
        if let Ok(dir) = std::env::var("LUMEN_KV_DISK_DIR") {
            return PathBuf::from(dir);
        }
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(".cache").join("lumen").join("kv");
        }
        PathBuf::from(".lumen-kv")
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub fn len(&self) -> usize {
        self.index.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.entries.is_empty()
    }

    pub fn total_bytes(&self) -> u64 {
        self.index.entries.iter().map(|e| e.bytes).sum()
    }

    fn index_path(dir: &Path) -> PathBuf {
        dir.join("index.json")
    }

    fn load_index(dir: &Path) -> Result<DiskIndex> {
        let path = Self::index_path(dir);
        match fs::read(&path) {
            Ok(bytes) => Ok(serde_json::from_slice(&bytes).unwrap_or_default()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(DiskIndex::default()),
            Err(e) => Err(anyhow!("kv_disk: read index {}: {e}", path.display())),
        }
    }

    /// Write the index through a temp file and a rename — on the background
    /// thread when it runs, after the payload writes queued before it.
    fn save_index(&self) -> Result<()> {
        let path = Self::index_path(&self.dir);
        let bytes = serde_json::to_vec(&self.index).context("kv_disk: serialize index")?;
        match &self.writer {
            Some(w) => {
                w.submit(FileJob::Write {
                    path,
                    bytes,
                    seq: None,
                });
                Ok(())
            }
            None => write_via_tmp(&self.create, &path, &bytes),
        }
    }

    /// Sanitize a cache key into a filesystem-safe basename. Keys are typically
    /// `auto-<hex>` already, but guard against separators.
    fn file_for(key: &str) -> String {
        let safe: String = key
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        format!("{safe}.lkv")
    }

    /// Whether writing `bytes` more leaves the store's volume at least
    /// `min_free_bytes` free. Writes still queued for the writer thread count
    /// as written, since their bytes are about to land. A volume whose free
    /// space cannot be read is not held back: a write that does not fit then
    /// fails the way a full disk does. Logs the first write of a run that is
    /// held back, and the first that is not again.
    fn leaves_floor_free(&mut self, bytes: u64) -> bool {
        if self.min_free_bytes == 0 {
            return true;
        }
        let Ok(free) = (self.free_space)(&self.dir) else {
            return true;
        };
        let queued: u64 = self
            .index
            .entries
            .iter()
            .filter(|e| e.pending.is_some())
            .map(|e| e.bytes)
            .sum();
        let fits = free.saturating_sub(queued).saturating_sub(bytes) >= self.min_free_bytes;
        self.floor.note(fits, free, self.min_free_bytes, &self.dir);
        fits
    }

    /// Persist a snapshot under `key`, replacing any existing entry. Evicts LRU
    /// victims afterward to honor the byte budget.
    ///
    /// Under the free-space floor ([`Self::set_min_free_bytes`]) the write is
    /// skipped instead, before anything is serialized, and `Ok` returned: the
    /// store keeps what it has (an entry under `key` included) and serves it.
    /// The check counts the records' bytes; the file adds a few KB of
    /// manifest.
    pub fn put(&mut self, key: &str, manifest: &KvManifest, records: &[ArrayRecord]) -> Result<()> {
        self.reconcile();
        let payload: u64 = records.iter().map(|r| r.data.len() as u64).sum();
        if !self.leaves_floor_free(payload) {
            return Ok(());
        }
        let file = Self::file_for(key);
        let path = self.dir.join(&file);

        let mut buf = Vec::new();
        write_lkv(&mut buf, manifest, records)?;
        let bytes = buf.len() as u64;

        let pending = self.write_file(path, buf)?;

        let now = now_unix();
        self.prune_ttl();
        self.index.entries.retain(|e| e.key != key);
        self.index.entries.push(DiskEntry {
            key: key.to_string(),
            file,
            bytes,
            last_access_unix: now,
            hits: 0,
            pending,
            superseded_by: None,
        });

        self.evict_to_fit()?;
        self.save_index()?;
        Ok(())
    }

    /// Whether `key` is in the index. Reads no file, so probing many
    /// candidate keys (see [`point_key`]) costs nothing per miss.
    pub fn contains(&self, key: &str) -> bool {
        let report = lock(&self.report);
        self.index.entries.iter().any(|e| {
            e.key == key
                && !e
                    .pending
                    .is_some_and(|seq| report.outcomes.contains(&(seq, false)))
        })
    }

    /// Mark `key` as used now, so the last-access TTL and the LRU budget keep an
    /// entry that is read from memory instead of from here. Returns whether the
    /// key is stored. Reads no file.
    pub fn touch(&mut self, key: &str) -> Result<bool> {
        self.reconcile();
        let Some(e) = self.index.entries.iter_mut().find(|e| e.key == key) else {
            return Ok(false);
        };
        e.last_access_unix = now_unix();
        self.save_index()?;
        Ok(true)
    }

    /// Load a snapshot by `key`. Returns `None` on miss, on fingerprint
    /// mismatch, or on a corrupt/unreadable file (treated as a miss, never a
    /// crash). Bumps the entry's access time + hit count on success.
    pub fn get(&mut self, key: &str) -> Result<Option<(KvManifest, Vec<ArrayRecord>)>> {
        self.reconcile();
        let Some(pos) = self.index.entries.iter().position(|e| e.key == key) else {
            return Ok(None);
        };
        let path = self.dir.join(&self.index.entries[pos].file);

        let bytes = match fs::read(&path) {
            Ok(b) => b,
            // File not present yet (async write in flight) or externally removed
            // — miss WITHOUT dropping the index entry so it self-heals once the
            // worker lands the payload.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(_) => {
                self.remove(key)?;
                return Ok(None);
            }
        };

        let (manifest, records) = match read_lkv(&mut bytes.as_slice()) {
            Ok(v) => v,
            Err(_) => {
                // Present but corrupt — drop the entry and miss.
                self.remove(key)?;
                return Ok(None);
            }
        };

        if manifest.model_fingerprint != self.fingerprint {
            // Stale snapshot from a different model — discard.
            self.remove(key)?;
            return Ok(None);
        }

        let now = now_unix();
        let e = &mut self.index.entries[pos];
        e.last_access_unix = now;
        e.hits += 1;
        self.save_index()?;
        Ok(Some((manifest, records)))
    }

    /// Remove a single entry (file + index row). Missing file is not an error.
    pub fn remove(&mut self, key: &str) -> Result<()> {
        self.reconcile();
        if self.drop_entry(key) {
            self.save_index()?;
        }
        Ok(())
    }

    /// Remove `key`'s row and queue the deletion of its file; whether it was
    /// there. Does not save the index.
    fn drop_entry(&mut self, key: &str) -> bool {
        let Some(pos) = self.index.entries.iter().position(|e| e.key == key) else {
            return false;
        };
        let e = self.index.entries.remove(pos);
        self.delete_file(self.dir.join(&e.file));
        true
    }

    /// `newer` holds everything `older` does (a later state of the same
    /// conversation), so `older` goes — but only once `newer` is on the disk.
    /// At once when it already is; when its background write is still queued,
    /// after the writer reports it landed. If that write fails (a full disk),
    /// `older` stays: the last state that made it to the disk is kept rather
    /// than losing both. A `newer` that is not stored at all keeps `older` too.
    pub fn supersede(&mut self, older: &str, newer: &str) -> Result<()> {
        self.reconcile();
        if older == newer {
            return Ok(());
        }
        match self.index.entries.iter().find(|e| e.key == newer) {
            None => Ok(()),
            Some(e) if e.pending.is_none() => self.remove(older),
            Some(_) => {
                let Some(e) = self.index.entries.iter_mut().find(|e| e.key == older) else {
                    return Ok(());
                };
                e.superseded_by = Some(newer.to_string());
                self.save_index()
            }
        }
    }

    /// Apply what the writer thread reported since the last call. A write
    /// that failed (a full disk) loses its index row, which would otherwise
    /// read as a miss forever, count against the byte budget and stop the key
    /// from being written again; the file it would have replaced, if any,
    /// goes too, since no row leads to it any more. A write that landed lets
    /// the entries it supersedes go.
    fn reconcile(&mut self) {
        let outcomes = std::mem::take(&mut lock(&self.report).outcomes);
        let mut changed = false;
        for (seq, landed) in outcomes {
            let Some(pos) = self
                .index
                .entries
                .iter()
                .position(|e| e.pending == Some(seq))
            else {
                continue;
            };
            let key = self.index.entries[pos].key.clone();
            if landed {
                self.index.entries[pos].pending = None;
            } else {
                changed |= self.drop_entry(&key);
            }
            let retired: Vec<String> = self
                .index
                .entries
                .iter_mut()
                .filter(|e| e.superseded_by.as_deref() == Some(key.as_str()))
                .filter_map(|e| {
                    if landed {
                        Some(e.key.clone())
                    } else {
                        e.superseded_by = None;
                        None
                    }
                })
                .collect();
            changed |= !landed;
            for older in retired {
                changed |= self.drop_entry(&older);
            }
        }
        if changed {
            let _ = self.save_index();
        }
    }

    /// Drop all entries and their files.
    pub fn clear(&mut self) -> Result<()> {
        self.reconcile();
        for e in std::mem::take(&mut self.index.entries) {
            self.delete_file(self.dir.join(&e.file));
        }
        self.save_index()?;
        Ok(())
    }

    /// Tests: `(writes failed since the last one that landed, lines logged)`.
    #[cfg(test)]
    fn write_report_for_test(&self) -> (u64, u64) {
        let r = lock(&self.report);
        (r.failing, r.lines)
    }

    /// Evict least-recently-used entries until under the byte budget.
    fn evict_to_fit(&mut self) -> Result<()> {
        if self.max_bytes == 0 {
            return Ok(());
        }
        while self.total_bytes() > self.max_bytes && self.index.entries.len() > 1 {
            // Find LRU victim (oldest last_access_unix).
            // Unreachable: the loop guard requires `entries.len() > 1`, so a
            // minimum exists. `break` rather than `unwrap` keeps a future
            // change to that guard from aborting an eviction pass.
            let Some(victim) = self
                .index
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last_access_unix)
                .map(|(i, _)| i)
            else {
                break;
            };
            let e = self.index.entries.remove(victim);
            self.delete_file(self.dir.join(&e.file));
        }
        Ok(())
    }
}

/// Opens a file for writing: the store's only way of putting bytes on the disk,
/// so a test can stand in a disk that fills up part-way through a file.
type CreateFile =
    std::sync::Arc<dyn Fn(&Path) -> std::io::Result<Box<dyn Write + Send>> + Send + Sync>;

/// The filesystem.
fn real_files() -> CreateFile {
    std::sync::Arc::new(|p| Ok(Box::new(fs::File::create(p)?) as Box<dyn Write + Send>))
}

/// Write `bytes` to `<path>.tmp`, then rename it onto `path`, so a reader
/// never sees a partial file. A write that fails — a full disk stops it
/// part-way — takes its temp file with it rather than leave the space taken
/// until the next open.
fn write_via_tmp(create: &CreateFile, path: &Path, bytes: &[u8]) -> Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    let written = create(&tmp)
        .and_then(|mut f| {
            f.write_all(bytes)?;
            f.flush()
        })
        .with_context(|| format!("kv_disk: write {}", tmp.display()))
        .and_then(|()| {
            fs::rename(&tmp, path).with_context(|| format!("kv_disk: rename {}", path.display()))
        });
    if written.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    written
}

/// Reads the bytes free on the volume holding a path: the filesystem, or in a
/// test a disk whose free space the test sets.
type FreeSpace = std::sync::Arc<dyn Fn(&Path) -> std::io::Result<u64> + Send + Sync>;

/// Bytes an unprivileged process may still write on the volume holding `path`
/// (`statvfs`: `f_bavail × f_frsize`).
#[cfg(unix)]
fn volume_free_bytes(path: &Path) -> std::io::Result<u64> {
    let c = std::ffi::CString::new(path.as_os_str().as_encoded_bytes())?;
    // SAFETY: `c` is NUL-terminated and `st` is a valid out-pointer, zeroed
    // because `statvfs` is plain old data.
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c.as_ptr(), &mut st) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    // The fields are 32-bit on some platforms and 64-bit on others.
    #[allow(clippy::unnecessary_cast)]
    Ok((st.f_bavail as u64).saturating_mul(st.f_frsize as u64))
}

#[cfg(not(unix))]
fn volume_free_bytes(_path: &Path) -> std::io::Result<u64> {
    Err(std::io::ErrorKind::Unsupported.into())
}

/// The free-space floor's account of the writes it skipped.
#[derive(Default)]
struct FloorReport {
    /// Writes skipped since the floor last let one through. A volume that
    /// stays short skips every write, so only the first of a run is logged,
    /// and the line that says writes resume gives the count.
    skipped: u64,
    /// Lines logged about the floor.
    lines: u64,
}

impl FloorReport {
    fn note(&mut self, fits: bool, free: u64, floor: u64, dir: &Path) {
        let gb = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
        if fits {
            if self.skipped > 0 {
                eprintln!(
                    "[kv-disk] writes resume: {:.1} GB free on the volume of {} ({} skipped \
                     below the {:.1} GB floor)",
                    gb(free),
                    dir.display(),
                    self.skipped,
                    gb(floor),
                );
                self.skipped = 0;
                self.lines += 1;
            }
            return;
        }
        if self.skipped == 0 {
            eprintln!(
                "[kv-disk] not writing: {:.1} GB free on the volume of {}, and a write may not \
                 leave less than {:.1} GB (LUMEN_KV_DISK_MIN_FREE_GB). States already on disk \
                 are kept and still read; further skips are counted, not logged, until there \
                 is room again",
                gb(free),
                dir.display(),
                gb(floor),
            );
            self.lines += 1;
        }
        self.skipped += 1;
    }
}

/// What a store's writer thread reports back to it.
#[derive(Default)]
struct WriterReport {
    /// `(write, landed)` for every numbered write since the store last looked.
    outcomes: Vec<(u64, bool)>,
    /// Writes failed since the last one that landed. A full disk fails every
    /// write, so only the first failure of a run is logged, and the line that
    /// says writes land again gives the count.
    failing: u64,
    /// Lines logged about failing writes.
    lines: u64,
}

impl WriterReport {
    fn note(&mut self, seq: Option<u64>, result: &Result<()>) {
        if let Some(seq) = seq {
            self.outcomes.push((seq, result.is_ok()));
        }
        match result {
            Ok(()) if self.failing > 0 => {
                eprintln!(
                    "[kv-disk] writes land again ({} failed before this one)",
                    self.failing
                );
                self.failing = 0;
                self.lines += 1;
            }
            Ok(()) => {}
            Err(e) => {
                if self.failing == 0 {
                    eprintln!(
                        "[kv-disk] write failed, so that state is not kept on disk: {e:#}. \
                         Serving goes on; further failures are counted, not logged, until a \
                         write lands again"
                    );
                    self.lines += 1;
                }
                self.failing += 1;
            }
        }
    }
}

/// `m`'s guard, whether or not a holder panicked (the data is counters).
fn lock<T>(m: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// File jobs submitted to any store's writer thread and not finished yet.
static IN_FLIGHT: std::sync::LazyLock<(std::sync::Mutex<usize>, std::sync::Condvar)> =
    std::sync::LazyLock::new(|| (std::sync::Mutex::new(0), std::sync::Condvar::new()));

fn in_flight_add(n: usize) {
    let (count, _) = &*IN_FLIGHT;
    *count.lock().unwrap_or_else(|e| e.into_inner()) += n;
}

fn in_flight_done() {
    let (count, cv) = &*IN_FLIGHT;
    let mut c = count.lock().unwrap_or_else(|e| e.into_inner());
    *c = c.saturating_sub(1);
    if *c == 0 {
        cv.notify_all();
    }
}

/// Block until every file write and delete queued by any [`DiskKvStore`] has
/// finished, or `timeout` passes. Returns whether the queue drained.
///
/// A process that is told to stop calls this first: the writer threads are
/// detached from its exit path, so a termination signal otherwise kills a
/// write halfway through its temp file and the newest state never lands (the
/// previous one stays valid, which is why this is a courtesy, not a repair).
pub fn drain_writes(timeout: std::time::Duration) -> bool {
    let (count, cv) = &*IN_FLIGHT;
    let guard = count.lock().unwrap_or_else(|e| e.into_inner());
    let (guard, result) = cv
        .wait_timeout_while(guard, timeout, |c| *c > 0)
        .unwrap_or_else(|e| e.into_inner());
    *guard == 0 && !result.timed_out()
}

/// A file mutation queued for a store's writer thread.
enum FileJob {
    /// `seq` numbers a payload write, whose outcome goes back to the store.
    Write {
        path: PathBuf,
        bytes: Vec<u8>,
        seq: Option<u64>,
    },
    Remove(PathBuf),
}

/// One thread that performs a store's file writes and deletes in submission
/// order: a request never waits on the disk, and a delete can never overtake
/// the write of the same file. Dropping it finishes the queue.
struct FileWriter {
    tx: Option<std::sync::mpsc::Sender<FileJob>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl FileWriter {
    fn spawn(
        create: CreateFile,
        report: std::sync::Arc<std::sync::Mutex<WriterReport>>,
    ) -> Result<Self> {
        let (tx, rx) = std::sync::mpsc::channel::<FileJob>();
        let handle = std::thread::Builder::new()
            .name("lumen-kv-disk-writer".into())
            .spawn(move || {
                for job in rx {
                    match job {
                        FileJob::Write { path, bytes, seq } => {
                            // A failed write leaves nothing behind (see
                            // `write_via_tmp`); the store drops its row when it
                            // reads the report.
                            let result = write_via_tmp(&create, &path, &bytes);
                            lock(&report).note(seq, &result);
                        }
                        FileJob::Remove(path) => {
                            let _ = fs::remove_file(path);
                        }
                    }
                    in_flight_done();
                }
            })
            .context("kv_disk: spawn writer thread")?;
        Ok(Self {
            tx: Some(tx),
            handle: Some(handle),
        })
    }

    fn submit(&self, job: FileJob) {
        if let Some(tx) = &self.tx {
            in_flight_add(1);
            if tx.send(job).is_err() {
                in_flight_done();
            }
        }
    }
}

impl Drop for FileWriter {
    fn drop(&mut self) {
        drop(self.tx.take());
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// The store key of a session state persisted after `tokens` on a prefill
/// grid of `chunk`: a content hash, so it is the same in every process (unlike
/// session ids, which reset on restart) and a lookup needs only the prompt.
/// The chunk is part of it because a state is only exact to resume on the grid
/// it was computed on (see `session_feed`).
pub fn point_key(chunk: usize, tokens: &[u32]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update((chunk as u64).to_le_bytes());
    for t in tokens {
        h.update(t.to_le_bytes());
    }
    let hex: String = h.finalize()[..16]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    format!("pt-{hex}")
}

/// The store key of a prefix-cache head (the state after a system prompt and
/// its tools) whose tokens are `tokens`: a content hash, the same in every
/// process, so a restart finds the head of each system prompt it served. The
/// prefill grid the head was computed on is part of the store's identity.
pub fn head_key(tokens: &[u32]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update((tokens.len() as u64).to_le_bytes());
    for t in tokens {
        h.update(t.to_le_bytes());
    }
    let hex: String = h.finalize()[..16]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    format!("hd-{hex}")
}

// ---------------------------------------------------------------------------
// Identity + opening the store from env
// ---------------------------------------------------------------------------

/// Env-var name prefixes whose values can change a persisted snapshot's bytes
/// or the numerics of the forward that extends it: KV dtype and compression,
/// weight requantization, the prefill pass grid, shared-prefix dedup, routing
/// and sparse attention. Over-inclusion only costs extra misses; leaving one
/// out lets a snapshot built under one setting be extended under another.
const IDENTITY_ENV_PREFIXES: &[&str] = &[
    "LUMEN_MLX_KV_",
    "LUMEN_QWEN35_TQ_",
    "LUMEN_QWEN35_QUANT_KV",
    "LUMEN_QWEN35_PREFILL_CHUNK",
    "LUMEN_QWEN35_PREFILL_SCORES_GB",
    "LUMEN_NATIVE_REQUANT_",
    "LUMEN_MLX_SHARED_PREFIX",
    "LUMEN_GEMMA4_TQ_",
    "LUMEN_GEMMA4_QUANT_KV_",
    "LUMEN_GEMMA4_PREFILL_CHUNK",
    "LUMEN_GEMMA4_PREFILL_SCORES_GB",
    "LUMEN_GEMMA4_TOP_K",
    "LUMEN_SPARSE_DECODE",
];

/// Env names that match a prefix above but only steer logging.
const IDENTITY_ENV_IGNORED_SUFFIXES: &[&str] = &["_LOG", "_TIMING"];

/// The settings a snapshot's identity covers, as sorted `(name, value)` pairs:
/// every env var under [`IDENTITY_ENV_PREFIXES`], plus every flag registered as
/// `Behavior` (it changes output by definition) at its effective value.
pub fn identity_settings_from_env() -> Vec<(String, String)> {
    let mut out: std::collections::BTreeMap<String, String> = std::env::vars()
        .filter(|(k, _)| {
            IDENTITY_ENV_PREFIXES.iter().any(|p| k.starts_with(p))
                && !IDENTITY_ENV_IGNORED_SUFFIXES.iter().any(|s| k.ends_with(s))
        })
        .collect();
    for f in lumen_flags::registry_sorted() {
        if f.kind == lumen_flags::FlagKind::Behavior {
            let v = std::env::var(f.env).unwrap_or_else(|_| format!("default={}", f.default));
            out.insert(f.env.to_string(), v);
        }
    }
    out.into_iter().collect()
}

/// The namespace persisted snapshots live under: a readable model name plus a
/// hash of everything their bytes depend on — this format's version, the
/// weights (shard names and sizes, `config.json`, the shard index) and
/// `settings` (see [`identity_settings_from_env`]; the MLX build goes here
/// too). A snapshot is only ever served to a process whose identity matches;
/// any change is a different directory, i.e. a miss, never a request error.
pub fn kv_identity(model_id: &str, model_dir: &Path, settings: &[(String, String)]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    let mut field = |name: &str, value: &[u8]| {
        h.update((name.len() as u64).to_le_bytes());
        h.update(name.as_bytes());
        h.update((value.len() as u64).to_le_bytes());
        h.update(value);
    };
    field(
        "format",
        &[LKV_MAGIC.as_slice(), &LKV_VERSION.to_le_bytes()].concat(),
    );
    field("model_id", model_id.as_bytes());

    let mut shards: Vec<(String, u64)> = fs::read_dir(model_dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            let len = e.metadata().ok()?.len();
            name.ends_with(".safetensors").then_some((name, len))
        })
        .collect();
    shards.sort();
    for (name, len) in &shards {
        field(name, &len.to_le_bytes());
    }
    for file in ["config.json", "model.safetensors.index.json"] {
        if let Ok(bytes) = fs::read(model_dir.join(file)) {
            field(file, &bytes);
        }
    }
    for (k, v) in settings {
        field(k, v.as_bytes());
    }

    let digest = h.finalize();
    let hex: String = digest[..8].iter().map(|b| format!("{b:02x}")).collect();
    let name: String = model_id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .take(64)
        .collect();
    format!("{name}-{hex}")
}

lumen_flags::flag! {
    /// The disk KV tier (`kv_disk.rs`): prompt and conversation states, and
    /// Flash-Next agents' system-prompt heads, written under
    /// `LUMEN_KV_DISK_DIR` (default `~/.cache/lumen/kv`) and read back after a
    /// restart or after memory pressure dropped them, instead of prefilling
    /// them again. A state read back is the cold computation exactly, so only
    /// speed changes.
    ///
    /// On by default, within three limits: `LUMEN_KV_DISK_MAX_GB` (default
    /// 10; least recently used entries go past it; `0` = unlimited),
    /// `LUMEN_KV_DISK_TTL_SECS` (default 2592000, 30 days, since an entry was
    /// last used; `0` = entries never expire) and `LUMEN_KV_DISK_MIN_FREE_GB`
    /// (default 10: no writes while the store's volume has less than that
    /// free, though stored entries stay and are still read; `0` = no floor).
    /// `LUMEN_KV_DISK=0` (or `false`, `off`, `no`) turns the tier off.
    pub kv_disk_tier {
        env: "LUMEN_KV_DISK",
        default: true,
        kind: Optimization,
    }
}

const GIB: u64 = 1024 * 1024 * 1024;

/// The disk tier's limits (see [`kv_disk_tier`]), as [`open_store_from_env`]
/// reads them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiskTierLimits {
    /// LRU byte budget (`LUMEN_KV_DISK_MAX_GB`); `0` = unbounded.
    pub max_bytes: u64,
    /// Last-access TTL in seconds (`LUMEN_KV_DISK_TTL_SECS`); `0` = entries
    /// never expire.
    pub ttl_secs: u64,
    /// Free space a write must leave on the store's volume
    /// (`LUMEN_KV_DISK_MIN_FREE_GB`); `0` = no floor.
    pub min_free_bytes: u64,
}

impl DiskTierLimits {
    /// `LUMEN_KV_DISK_MAX_GB` when unset.
    pub const DEFAULT_MAX_GB: u64 = 10;
    /// `LUMEN_KV_DISK_TTL_SECS` when unset: 30 days.
    pub const DEFAULT_TTL_SECS: u64 = 30 * 24 * 60 * 60;
    /// `LUMEN_KV_DISK_MIN_FREE_GB` when unset.
    pub const DEFAULT_MIN_FREE_GB: u64 = 10;

    pub fn from_env() -> Self {
        Self::parse(
            std::env::var("LUMEN_KV_DISK_MAX_GB").ok().as_deref(),
            std::env::var("LUMEN_KV_DISK_TTL_SECS").ok().as_deref(),
            std::env::var("LUMEN_KV_DISK_MIN_FREE_GB").ok().as_deref(),
        )
    }

    /// The limits from the variables' raw values (`None` = unset). An unset
    /// or empty value takes the default, and so does one that is not a whole
    /// number (with a line on stderr); an explicit `0` keeps its meaning.
    fn parse(max_gb: Option<&str>, ttl_secs: Option<&str>, min_free_gb: Option<&str>) -> Self {
        let whole = |name: &str, raw: Option<&str>, default: u64| {
            let raw = raw.map(str::trim).unwrap_or_default();
            if raw.is_empty() {
                return default;
            }
            raw.parse::<u64>().unwrap_or_else(|_| {
                eprintln!("[kv-disk] {name}={raw:?} is not a whole number; using {default}");
                default
            })
        };
        Self {
            max_bytes: whole("LUMEN_KV_DISK_MAX_GB", max_gb, Self::DEFAULT_MAX_GB)
                .saturating_mul(GIB),
            ttl_secs: whole("LUMEN_KV_DISK_TTL_SECS", ttl_secs, Self::DEFAULT_TTL_SECS),
            min_free_bytes: whole(
                "LUMEN_KV_DISK_MIN_FREE_GB",
                min_free_gb,
                Self::DEFAULT_MIN_FREE_GB,
            )
            .saturating_mul(GIB),
        }
    }
}

/// The L2 disk tier under `identity`, configured from env (see
/// [`kv_disk_tier`]). `None` when the tier is off, or when the store cannot
/// be opened (logged): the disk tier is an accelerator, never a reason to
/// fail a load.
pub fn open_store_from_env(identity: &str, log_tag: &str) -> Option<DiskKvStore> {
    if !kv_disk_tier::get() {
        return None;
    }
    open_store(
        DiskKvStore::default_root(),
        identity,
        log_tag,
        DiskTierLimits::from_env(),
    )
}

/// [`open_store_from_env`] under `root` with `limits`, its writes on the
/// background thread.
fn open_store(
    root: impl AsRef<Path>,
    identity: &str,
    log_tag: &str,
    limits: DiskTierLimits,
) -> Option<DiskKvStore> {
    let opened = DiskKvStore::open(root, identity, limits.max_bytes, limits.ttl_secs).and_then(
        |mut store| {
            store.set_min_free_bytes(limits.min_free_bytes);
            store.write_in_background().map(|()| store)
        },
    );
    match opened {
        Ok(store) => {
            eprintln!(
                "[{log_tag}] kv-disk tier ON: identity={identity:?} entries={} max_gb={} \
                 ttl_secs={} min_free_gb={} (LUMEN_KV_DISK=0 turns it off)",
                store.len(),
                limits.max_bytes / GIB,
                limits.ttl_secs,
                limits.min_free_bytes / GIB,
            );
            Some(store)
        }
        Err(e) => {
            eprintln!("[{log_tag}] kv-disk tier disabled (open failed): {e:#}");
            None
        }
    }
}

/// L2 spill trigger (opt-in via `LUMEN_KV_SPILL_MEM_GB`): true when MLX active
/// memory exceeds the configured GB threshold, so the caller should drop cold
/// in-memory prefix snapshots to relieve pressure; a prompt that needed one
/// resumes from whatever state the disk tier holds for it. `0` / unset
/// disables spill.
#[cfg(feature = "mlx-native")]
pub fn under_memory_pressure() -> bool {
    let thr_gb = std::env::var("LUMEN_KV_SPILL_MEM_GB")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|&g| g > 0.0);
    match thr_gb {
        Some(g) => match crate::metal_memory::get_active_memory() {
            Ok(bytes) => (bytes as f64 / (1024.0 * 1024.0 * 1024.0)) > g,
            Err(_) => false,
        },
        None => false,
    }
}

/// Number of in-memory prefix snapshots to keep when spilling under pressure
/// (`LUMEN_KV_SPILL_KEEP`, default 2). The rest are dropped from RAM (kept on
/// disk).
pub fn spill_keep_floor() -> usize {
    std::env::var("LUMEN_KV_SPILL_KEEP")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(2)
}

/// Current unix time in seconds (0 on clock error — only affects LRU ordering).
fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// MLX layer — Array <-> ArrayRecord (requires a live Metal device)
// ---------------------------------------------------------------------------

#[cfg(feature = "mlx-native")]
mod mlx_conv {
    use super::*;
    use crate::native_cache::NativeRotatingKvCacheTurboQuant;
    use mlx_rs::{Array, Dtype};

    impl DtypeTag {
        pub fn from_mlx(d: Dtype) -> Result<Self> {
            Ok(match d {
                Dtype::Bool => DtypeTag::Bool,
                Dtype::Uint8 => DtypeTag::U8,
                Dtype::Uint16 => DtypeTag::U16,
                Dtype::Uint32 => DtypeTag::U32,
                Dtype::Uint64 => DtypeTag::U64,
                Dtype::Int8 => DtypeTag::I8,
                Dtype::Int16 => DtypeTag::I16,
                Dtype::Int32 => DtypeTag::I32,
                Dtype::Int64 => DtypeTag::I64,
                Dtype::Float16 => DtypeTag::F16,
                Dtype::Float32 => DtypeTag::F32,
                Dtype::Float64 => DtypeTag::F64,
                Dtype::Bfloat16 => DtypeTag::BF16,
                other => bail!("kv_disk: unsupported mlx dtype {other:?}"),
            })
        }

        pub fn to_mlx(self) -> Dtype {
            match self {
                DtypeTag::Bool => Dtype::Bool,
                DtypeTag::U8 => Dtype::Uint8,
                DtypeTag::U16 => Dtype::Uint16,
                DtypeTag::U32 => Dtype::Uint32,
                DtypeTag::U64 => Dtype::Uint64,
                DtypeTag::I8 => Dtype::Int8,
                DtypeTag::I16 => Dtype::Int16,
                DtypeTag::I32 => Dtype::Int32,
                DtypeTag::I64 => Dtype::Int64,
                DtypeTag::F16 => Dtype::Float16,
                DtypeTag::F32 => Dtype::Float32,
                DtypeTag::F64 => Dtype::Float64,
                DtypeTag::BF16 => Dtype::Bfloat16,
            }
        }
    }

    /// The linked MLX build's version string (part of the snapshot identity:
    /// kernels, and so the numerics a restored cache is extended with, change
    /// between MLX releases).
    pub fn mlx_version() -> String {
        // SAFETY: mlx-c owns the string; we copy it out before freeing it, and
        // a null data pointer (only on failure) is handled.
        unsafe {
            let mut s = mlx_sys::mlx_string_new();
            let rc = mlx_sys::mlx_version(&mut s);
            let data = mlx_sys::mlx_string_data(s);
            let out = if rc == 0 && !data.is_null() {
                std::ffi::CStr::from_ptr(data)
                    .to_string_lossy()
                    .into_owned()
            } else {
                "unknown".to_string()
            };
            mlx_sys::mlx_string_free(s);
            out
        }
    }

    /// The elements of an evaluated, contiguous array, as raw bytes.
    fn raw_bytes<T: mlx_rs::ArrayElement>(a: &Array) -> Result<Vec<u8>> {
        let s = a
            .try_as_slice::<T>()
            .map_err(|e| anyhow!("kv_disk: as_slice {:?}: {e:?}", T::DTYPE))?;
        // SAFETY: `T` is a plain numeric element type (no padding, every bit
        // pattern valid), so viewing its storage as bytes is sound; the slice
        // stays borrowed from `a` for the duration of the copy.
        let bytes = unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<u8>(), size_of_val(s)) };
        Ok(bytes.to_vec())
    }

    /// Raw little-endian element bytes copied into an aligned `Vec<T>`.
    fn elems_from_raw<T: Copy + Default>(bytes: &[u8]) -> Vec<T> {
        let mut v = vec![T::default(); bytes.len() / size_of::<T>()];
        // SAFETY: `v` owns exactly `bytes.len()` bytes of plain numeric `T`
        // (checked by `validate_len` against the dtype width), the regions do
        // not overlap, and copying through `u8` sidesteps the alignment `bytes`
        // may not have.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), v.as_mut_ptr().cast::<u8>(), bytes.len())
        };
        v
    }

    /// An mlx array of `$t` built from raw little-endian element bytes.
    /// (A macro because mlx-rs keeps `from_slice`'s element bound private.)
    macro_rules! from_raw_bytes {
        ($t:ty, $bytes:expr, $shape:expr) => {
            Array::from_slice(&elems_from_raw::<$t>($bytes), $shape)
        };
    }

    /// Read an mlx `Array` into an `ArrayRecord`, in the array's own dtype.
    ///
    /// bf16 and f16 are read through a `u16` bit view, which keeps them 2 bytes
    /// on disk and makes the round trip bit-exact by construction. The array is
    /// `eval`'d before readback so lazy graphs are materialized contiguously.
    pub fn record_from_array(a: &Array) -> Result<ArrayRecord> {
        let tag = DtypeTag::from_mlx(a.dtype())?;
        if tag == DtypeTag::F64 {
            bail!("kv_disk: f64 arrays are not supported (KV caches never use f64)");
        }
        let shape = a.shape().to_vec();

        // CRITICAL: force a contiguous, logical-row-major materialization before
        // any `try_as_slice` readback. Cache backing buffers can be STRIDED views
        // (observed: a Gemma4 sliding cache's `values` is a strided/transposed
        // view from the V projection — `keys` is contiguous). `try_as_slice` on a
        // strided array returns the underlying buffer in PHYSICAL order, silently
        // permuting values vs the logical shape (a sum-preserving corruption that
        // decodes to garbage). `add(a, 0)` is NOT sufficient (it inherits the same
        // stride pathology for some views). Reshaping to a flat 1-D array forces
        // MLX to emit a true row-major contiguous copy in LOGICAL order for any
        // input layout — the reliable readback primitive.
        let numel: i32 = a.shape().iter().product();
        let a_flat = mlx_rs::ops::reshape(a, &[numel])
            .context("kv_disk: reshape to flat for contiguous logical readback")?;

        let flat = match tag {
            DtypeTag::BF16 | DtypeTag::F16 => a_flat
                .view::<u16>()
                .context("kv_disk: u16 bit view of a 16-bit float array")?,
            DtypeTag::Bool => a_flat
                .as_dtype(Dtype::Uint8)
                .context("kv_disk: bool → u8 for readback")?,
            _ => a_flat,
        };
        flat.eval().context("kv_disk: eval array for readback")?;
        let data = match tag {
            DtypeTag::Bool | DtypeTag::U8 => raw_bytes::<u8>(&flat)?,
            DtypeTag::I8 => raw_bytes::<i8>(&flat)?,
            DtypeTag::U16 | DtypeTag::BF16 | DtypeTag::F16 => raw_bytes::<u16>(&flat)?,
            DtypeTag::I16 => raw_bytes::<i16>(&flat)?,
            DtypeTag::U32 => raw_bytes::<u32>(&flat)?,
            DtypeTag::I32 => raw_bytes::<i32>(&flat)?,
            DtypeTag::F32 => raw_bytes::<f32>(&flat)?,
            DtypeTag::U64 => raw_bytes::<u64>(&flat)?,
            DtypeTag::I64 => raw_bytes::<i64>(&flat)?,
            DtypeTag::F64 => unreachable!("rejected above"),
        };

        let rec = ArrayRecord {
            dtype: tag,
            shape,
            data,
        };
        rec.validate_len()?;
        Ok(rec)
    }

    /// Reconstruct an mlx `Array` from an `ArrayRecord`, bit-exact in its own
    /// dtype (bf16/f16 come back through a `u16` bit view).
    pub fn record_to_array(rec: &ArrayRecord) -> Result<Array> {
        rec.validate_len()?;
        let (d, s) = (rec.data.as_slice(), rec.shape.as_slice());
        let arr = match rec.dtype {
            DtypeTag::U8 => from_raw_bytes!(u8, d, s),
            DtypeTag::Bool => from_raw_bytes!(u8, d, s)
                .as_dtype(Dtype::Bool)
                .context("kv_disk: u8 → bool")?,
            DtypeTag::I8 => from_raw_bytes!(i8, d, s),
            DtypeTag::U16 => from_raw_bytes!(u16, d, s),
            DtypeTag::BF16 | DtypeTag::F16 => from_raw_bytes!(u16, d, s)
                .view_dtype(rec.dtype.to_mlx())
                .context("kv_disk: 16-bit float from its u16 bit view")?,
            DtypeTag::I16 => from_raw_bytes!(i16, d, s),
            DtypeTag::U32 => from_raw_bytes!(u32, d, s),
            DtypeTag::I32 => from_raw_bytes!(i32, d, s),
            DtypeTag::F32 => from_raw_bytes!(f32, d, s),
            DtypeTag::U64 => from_raw_bytes!(u64, d, s),
            DtypeTag::I64 => from_raw_bytes!(i64, d, s),
            DtypeTag::F64 => bail!("kv_disk: f64 arrays are not supported"),
        };
        // Materialize before returning. The 16-bit float and bool paths end in a
        // lazy view/cast node; a rotating KV cache's in-place `slice_update` fast
        // path (Gemma4 sliding decode) misbehaves when its backing buffer is an
        // unevaluated graph node rather than a concrete array. Forcing eval here
        // makes reconstructed caches behave like cloned (already-evaluated) ones.
        arr.eval().context("kv_disk: eval reconstructed array")?;
        Ok(arr)
    }

    /// Append a present `Array` to the record blob and return its index;
    /// `None` for an absent one.
    pub fn push_record(records: &mut Vec<ArrayRecord>, a: Option<&Array>) -> Result<Option<usize>> {
        match a {
            None => Ok(None),
            Some(arr) => {
                let rec = record_from_array(arr)?;
                let id = records.len();
                records.push(rec);
                Ok(Some(id))
            }
        }
    }

    /// Rebuild the `Array` that record `id` holds; `None` for an absent slot.
    pub fn array_at(records: &[ArrayRecord], id: Option<usize>) -> Result<Option<Array>> {
        match id {
            None => Ok(None),
            Some(i) => {
                let rec = records
                    .get(i)
                    .ok_or_else(|| anyhow!("kv_disk: record index {i} out of range"))?;
                Ok(Some(record_to_array(rec)?))
            }
        }
    }

    /// Persist a TurboQuant rotating cache (Qwen's full-attention layers,
    /// Gemma 4's full and sliding ones): its arrays in the fixed order
    /// `[keys_codes, keys_sigma, values_codes, values_sigma, keys_signs?,
    /// keys_residual_norm?]` — uint8 codes, f32 sigma / residual norm, bf16
    /// signs — and the ring bookkeeping in the returned `LayerMeta`.
    pub fn turboquant_meta(
        records: &mut Vec<ArrayRecord>,
        c: &NativeRotatingKvCacheTurboQuant,
        kind: LayerKindTag,
    ) -> Result<LayerMeta> {
        let arrays = vec![
            push_record(records, c.keys_codes())?,
            push_record(records, c.keys_sigma())?,
            push_record(records, c.values_codes())?,
            push_record(records, c.values_sigma())?,
            push_record(records, c.keys_signs())?,
            push_record(records, c.keys_residual_norm())?,
        ];
        let mut m = LayerMeta::new(kind, c.offset());
        m.arrays = arrays;
        m.max_size = Some(c.max_size());
        m.keep = Some(c.keep());
        m.idx = Some(c.idx());
        m.bits = Some(c.bits());
        m.qjl_m = c.qjl_m();
        Ok(m)
    }

    /// Inverse of [`turboquant_meta`].
    pub fn turboquant_from_meta(
        m: &LayerMeta,
        records: &[ArrayRecord],
    ) -> Result<NativeRotatingKvCacheTurboQuant> {
        let kind = m.kind;
        let max_size = m
            .max_size
            .ok_or_else(|| anyhow!("kv_disk: {kind:?} layer missing max_size"))?;
        let keep = m
            .keep
            .ok_or_else(|| anyhow!("kv_disk: {kind:?} layer missing keep"))?;
        let idx = m
            .idx
            .ok_or_else(|| anyhow!("kv_disk: {kind:?} layer missing idx"))?;
        let bits = m
            .bits
            .ok_or_else(|| anyhow!("kv_disk: {kind:?} layer missing bits"))?;
        let slot = |i: usize| array_at(records, m.arrays.get(i).copied().flatten());
        Ok(NativeRotatingKvCacheTurboQuant::from_parts(
            slot(0)?, // keys_codes
            slot(1)?, // keys_sigma
            slot(2)?, // values_codes
            slot(3)?, // values_sigma
            slot(4)?, // keys_signs
            slot(5)?, // keys_residual_norm
            m.offset,
            max_size,
            keep,
            idx,
            bits,
            m.qjl_m,
        ))
    }
}

#[cfg(feature = "mlx-native")]
pub use mlx_conv::{
    array_at, mlx_version, push_record, record_from_array, record_to_array, turboquant_from_meta,
    turboquant_meta,
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_manifest() -> KvManifest {
        KvManifest {
            model_fingerprint: "fp-abc123".into(),
            created_at_unix: 1_700_000_000,
            position: 42,
            prefix_tokens: vec![1, 2, 3, 4, 5],
            last_token: Some(7),
            is_deep: true,
            layers: vec![
                LayerMeta {
                    arrays: vec![Some(0), Some(1)],
                    ..LayerMeta::new(LayerKindTag::Full, 42)
                },
                LayerMeta {
                    arrays: vec![Some(2), Some(3)],
                    max_size: Some(1024),
                    keep: Some(0),
                    idx: Some(42),
                    bits: Some(3),
                    qjl_m: Some(64),
                    ..LayerMeta::new(LayerKindTag::FullTurboquant, 42)
                },
            ],
        }
    }

    fn f32_record(shape: &[i32], fill: f32) -> ArrayRecord {
        let n: usize = shape.iter().map(|&d| d as usize).product();
        let mut data = Vec::with_capacity(n * 4);
        for i in 0..n {
            data.extend_from_slice(&(fill + i as f32).to_le_bytes());
        }
        ArrayRecord {
            dtype: DtypeTag::F32,
            shape: shape.to_vec(),
            data,
        }
    }

    fn u8_record(shape: &[i32]) -> ArrayRecord {
        let n: usize = shape.iter().map(|&d| d as usize).product();
        let data: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
        ArrayRecord {
            dtype: DtypeTag::U8,
            shape: shape.to_vec(),
            data,
        }
    }

    #[test]
    fn dtype_tag_byte_roundtrip() {
        for tag in [
            DtypeTag::Bool,
            DtypeTag::U8,
            DtypeTag::U16,
            DtypeTag::U32,
            DtypeTag::U64,
            DtypeTag::I8,
            DtypeTag::I16,
            DtypeTag::I32,
            DtypeTag::I64,
            DtypeTag::F16,
            DtypeTag::F32,
            DtypeTag::F64,
            DtypeTag::BF16,
        ] {
            assert_eq!(DtypeTag::from_u8(tag.to_u8()).unwrap(), tag);
        }
    }

    #[test]
    fn manifest_serde_roundtrip() {
        let m = sample_manifest();
        let json = serde_json::to_vec(&m).unwrap();
        let back: KvManifest = serde_json::from_slice(&json).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn lkv_file_roundtrip_is_byte_faithful() {
        let m = sample_manifest();
        let records = vec![
            f32_record(&[1, 2, 4, 8], 0.0),
            f32_record(&[1, 2, 4, 8], 100.0),
            u8_record(&[1, 4, 16, 128]),
            f32_record(&[1, 4, 16, 1], 3.5),
        ];

        let mut buf = Vec::new();
        write_lkv(&mut buf, &m, &records).unwrap();

        let (m2, r2) = read_lkv(&mut buf.as_slice()).unwrap();
        assert_eq!(m, m2);
        assert_eq!(records, r2);
    }

    #[test]
    fn read_lkv_rejects_bad_magic() {
        let mut buf = b"XXXX".to_vec();
        buf.extend_from_slice(&LKV_VERSION.to_le_bytes());
        let err = read_lkv(&mut buf.as_slice()).unwrap_err();
        assert!(err.to_string().contains("bad magic"));
    }

    #[test]
    fn array_record_rejects_length_mismatch() {
        let bad = ArrayRecord {
            dtype: DtypeTag::F32,
            shape: vec![2, 2],
            data: vec![0u8; 4], // expects 16
        };
        assert!(bad.validate_len().is_err());
    }

    #[test]
    fn disk_store_put_get_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let mut store = DiskKvStore::open(tmp.path(), "fp-abc123", 0, 0).unwrap();
        let m = sample_manifest();
        let records = vec![f32_record(&[1, 2, 2, 4], 1.0), u8_record(&[1, 2, 2, 4])];

        store.put("auto-deadbeef", &m, &records).unwrap();
        assert_eq!(store.len(), 1);

        let (m2, r2) = store.get("auto-deadbeef").unwrap().unwrap();
        assert_eq!(m, m2);
        assert_eq!(records, r2);

        // Miss on unknown key.
        assert!(store.get("auto-missing").unwrap().is_none());
    }

    /// A fake model directory: two shards of the given sizes plus a config.
    fn fake_model_dir(shard_sizes: [usize; 2], config: &str) -> tempfile::TempDir {
        let tmp = tempfile::tempdir().unwrap();
        for (i, n) in shard_sizes.iter().enumerate() {
            fs::write(
                tmp.path().join(format!("model-0000{i}.safetensors")),
                vec![0u8; *n],
            )
            .unwrap();
        }
        fs::write(tmp.path().join("config.json"), config).unwrap();
        fs::write(tmp.path().join("tokenizer.json"), "{}").unwrap();
        tmp
    }

    /// The identity must be a pure function of the weights and the settings,
    /// and must move when any one of them does — a snapshot served across a
    /// change it depends on is restored KV that silently disagrees with the
    /// forward that extends it.
    #[test]
    fn kv_identity_tracks_weights_and_settings_and_nothing_else() {
        let settings = vec![("LUMEN_MLX_KV_BF16".to_string(), "1".to_string())];
        let a = fake_model_dir([10, 20], r#"{"hidden_size": 8}"#);
        let id = kv_identity("org/Model-4bit", a.path(), &settings);

        // Deterministic, and a valid single path component with a readable name.
        assert_eq!(id, kv_identity("org/Model-4bit", a.path(), &settings));
        assert!(id.starts_with("org_Model-4bit-"), "got {id}");
        assert!(!id.contains('/') && id.len() <= 64 + 1 + 16, "got {id}");

        // Same content in another directory: same identity (a copied or
        // re-downloaded model keeps its cache).
        let b = fake_model_dir([10, 20], r#"{"hidden_size": 8}"#);
        assert_eq!(id, kv_identity("org/Model-4bit", b.path(), &settings));

        // A file the weights don't depend on doesn't move it.
        fs::write(b.path().join("tokenizer.json"), r#"{"x": 1}"#).unwrap();
        assert_eq!(id, kv_identity("org/Model-4bit", b.path(), &settings));

        // Each dependency moves it.
        let shard = fake_model_dir([10, 21], r#"{"hidden_size": 8}"#);
        let config = fake_model_dir([10, 20], r#"{"hidden_size": 9}"#);
        let mut setting = settings.clone();
        setting[0].1 = "0".into();
        let mut added = settings.clone();
        added.push(("mlx".into(), "0.32.3".into()));
        for (what, other) in [
            (
                "shard size",
                kv_identity("org/Model-4bit", shard.path(), &settings),
            ),
            (
                "config",
                kv_identity("org/Model-4bit", config.path(), &settings),
            ),
            (
                "setting value",
                kv_identity("org/Model-4bit", a.path(), &setting),
            ),
            (
                "added setting",
                kv_identity("org/Model-4bit", a.path(), &added),
            ),
            (
                "model id",
                kv_identity("org/Model-8bit", a.path(), &settings),
            ),
        ] {
            assert_ne!(id, other, "{what} must change the identity");
        }
    }

    /// With the background writer on, everything queued must be on disk once
    /// the store is dropped — in order, so a key written, deleted and written
    /// again ends up present — and a reopen must read it all back.
    #[test]
    fn background_writes_land_in_order_and_survive_drop() {
        let tmp = tempfile::tempdir().unwrap();
        let m = sample_manifest();
        let rec = |v: f32| vec![f32_record(&[1, 2, 2, 4], v)];
        {
            let mut store = DiskKvStore::open(tmp.path(), "fp-abc123", 0, 0).unwrap();
            store.write_in_background().unwrap();
            store.put("a", &m, &rec(1.0)).unwrap();
            store.put("b", &m, &rec(2.0)).unwrap();
            store.remove("a").unwrap();
            store.put("a", &m, &rec(3.0)).unwrap();
            assert_eq!(store.len(), 2);
        } // drop joins the writer

        let mut store = DiskKvStore::open(tmp.path(), "fp-abc123", 0, 0).unwrap();
        assert_eq!(store.len(), 2);
        assert_eq!(store.get("a").unwrap().unwrap().1, rec(3.0));
        assert_eq!(store.get("b").unwrap().unwrap().1, rec(2.0));
        let leftovers = fs::read_dir(tmp.path().join("fp-abc123"))
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .count();
        assert_eq!(leftovers, 0, "no temp file may survive a clean shutdown");
    }

    /// `drain_writes` must not return until the queued work is on disk, with
    /// the store still alive (a signal handler has no way to drop it first).
    #[test]
    fn drain_writes_waits_for_the_queue_without_dropping_the_store() {
        let tmp = tempfile::tempdir().unwrap();
        let m = sample_manifest();
        // Several multi-MB records so the writer thread is genuinely behind.
        let big = vec![f32_record(&[1, 1024, 1024, 2], 1.0)];
        let mut store = DiskKvStore::open(tmp.path(), "fp-abc123", 0, 0).unwrap();
        store.write_in_background().unwrap();
        for i in 0..4 {
            store.put(&format!("k{i}"), &m, &big).unwrap();
        }
        assert!(
            drain_writes(std::time::Duration::from_secs(30)),
            "queue did not drain"
        );
        let dir = tmp.path().join("fp-abc123");
        for i in 0..4 {
            assert!(
                dir.join(DiskKvStore::file_for(&format!("k{i}"))).exists(),
                "k{i} not on disk after drain"
            );
        }
        assert!(
            !fs::read_dir(&dir)
                .unwrap()
                .flatten()
                .any(|e| e.file_name().to_string_lossy().ends_with(".tmp")),
            "a temp file survived the drain"
        );
        // Nothing of ours is queued any more (the counter is process-wide, so
        // allow parallel tests' writes to finish too).
        assert!(drain_writes(std::time::Duration::from_secs(10)));
        drop(store);
    }

    /// A process that died mid-write leaves a half-written temp file and,
    /// possibly, an index row whose file never landed. Opening the store
    /// sweeps both, and the survivors still load.
    #[test]
    fn open_sweeps_temp_files_and_rows_without_a_file() {
        let tmp = tempfile::tempdir().unwrap();
        let m = sample_manifest();
        let records = vec![f32_record(&[1, 2, 2, 4], 1.0)];
        let dir = tmp.path().join("fp-abc123");
        {
            let mut store = DiskKvStore::open(tmp.path(), "fp-abc123", 0, 0).unwrap();
            store.put("kept", &m, &records).unwrap();
            store.put("lost", &m, &records).unwrap();
        }
        fs::remove_file(dir.join(DiskKvStore::file_for("lost"))).unwrap();
        fs::write(dir.join("half.lkv.tmp"), b"LKV2 partial").unwrap();

        let mut store = DiskKvStore::open(tmp.path(), "fp-abc123", 0, 0).unwrap();
        assert!(
            !dir.join("half.lkv.tmp").exists(),
            "temp file must be swept"
        );
        assert!(
            !store.contains("lost"),
            "a row without its file must be dropped"
        );
        assert_eq!(store.get("kept").unwrap().unwrap().1, records);
        // The sweep was persisted: a second open sees the same.
        drop(store);
        assert_eq!(
            DiskKvStore::open(tmp.path(), "fp-abc123", 0, 0)
                .unwrap()
                .len(),
            1
        );
    }

    // --- A full disk (ENOSPC part-way through a file) ---

    /// A disk with `room` bytes left: a write that does not fit puts what fits
    /// on the disk and then fails with ENOSPC, as a real full disk does, and
    /// the space comes back when the failed file goes (the store deletes it).
    fn disk_with_room(room: std::sync::Arc<std::sync::atomic::AtomicU64>) -> CreateFile {
        struct Filling {
            file: fs::File,
            room: std::sync::Arc<std::sync::atomic::AtomicU64>,
            written: u64,
            failed: bool,
        }
        impl Write for Filling {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                use std::sync::atomic::Ordering::SeqCst;
                let left = self.room.load(SeqCst).min(buf.len() as u64) as usize;
                if left == 0 && !buf.is_empty() {
                    self.failed = true;
                    return Err(std::io::Error::from_raw_os_error(28)); // ENOSPC
                }
                let n = self.file.write(&buf[..left])?;
                self.room.fetch_sub(n as u64, SeqCst);
                self.written += n as u64;
                Ok(n)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                self.file.flush()
            }
        }
        impl Drop for Filling {
            fn drop(&mut self) {
                if self.failed {
                    self.room
                        .fetch_add(self.written, std::sync::atomic::Ordering::SeqCst);
                }
            }
        }
        std::sync::Arc::new(move |p| {
            Ok(Box::new(Filling {
                file: fs::File::create(p)?,
                room: room.clone(),
                written: 0,
                failed: false,
            }) as Box<dyn Write + Send>)
        })
    }

    /// File names in `dir`, sorted.
    fn names_in(dir: &Path) -> Vec<String> {
        let mut v: Vec<String> = fs::read_dir(dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        v.sort();
        v
    }

    /// The server's mode (writes on the background thread) on a disk that
    /// fills up. Writes the disk refuses are given up: nothing truncated is
    /// left on the disk or in the index, the snapshot they would have replaced
    /// survives, and the store keeps serving — reads at once, writes as soon
    /// as there is room again. A restart then finds exactly what landed.
    #[test]
    fn a_full_disk_gives_up_the_write_and_keeps_serving() {
        use std::sync::atomic::Ordering::SeqCst;
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("fp-abc123");
        let m = sample_manifest();
        let rec = |v: f32| vec![f32_record(&[1, 2, 2, 4], v)];
        let room = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(u64::MAX));
        let drained = || assert!(drain_writes(std::time::Duration::from_secs(30)));

        let mut store = DiskKvStore::open(tmp.path(), "fp-abc123", 0, 0).unwrap();
        store.set_create_for_test(disk_with_room(room.clone()));
        store.write_in_background().unwrap();
        store.put("kept", &m, &rec(1.0)).unwrap();
        drained();
        let kept_bytes = store.total_bytes();

        // The disk fills: less room than one snapshot. Queued, so `put` says
        // nothing; a newer state of the same conversation supersedes `kept`
        // the way the runners do it.
        room.store(100, SeqCst);
        store.put("lost", &m, &rec(2.0)).unwrap();
        store.put("newer", &m, &rec(3.0)).unwrap();
        store.supersede("kept", "newer").unwrap();
        drained();

        // Every broken promise at once, not just the first.
        let mut broken = Vec::new();
        let names = names_in(&dir);
        if names.iter().any(|n| n.ends_with(".tmp")) {
            broken.push(format!(
                "a half-written file was left on a full disk: {names:?}"
            ));
        }
        for k in ["lost", "newer"] {
            if names.contains(&DiskKvStore::file_for(k)) {
                broken.push(format!("{k}: a refused write left a file: {names:?}"));
            }
            if store.contains(k) {
                broken.push(format!("{k}: a refused write is still listed"));
            }
            if store.get(k).unwrap().is_some() {
                broken.push(format!("{k}: served"));
            }
        }
        if store.get("kept").unwrap().map(|(_, r)| r) != Some(rec(1.0)) {
            broken.push(
                "the last state that reached the disk was deleted for one that never did".into(),
            );
        }
        if store.len() != 1 {
            broken.push(format!("{} entries listed, 1 on the disk", store.len()));
        }
        if store.total_bytes() != kept_bytes {
            broken.push(format!(
                "refused writes count against the budget: {} bytes listed, {kept_bytes} on the disk",
                store.total_bytes()
            ));
        }
        assert!(broken.is_empty(), "{broken:#?}");
        let (failing, lines) = store.write_report_for_test();
        assert!(failing >= 2, "{failing} failed writes counted");
        assert_eq!(lines, 1, "a full disk must be logged once, not per write");

        // Room again: writes land, and a newer state now does replace `kept`.
        room.store(u64::MAX, SeqCst);
        store.put("after", &m, &rec(4.0)).unwrap();
        store.supersede("kept", "after").unwrap();
        drained();
        assert_eq!(store.get("after").unwrap().map(|(_, r)| r), Some(rec(4.0)));
        assert!(store.get("kept").unwrap().is_none());
        assert_eq!(
            store.write_report_for_test(),
            (0, 2),
            "one more line: writes land again"
        );
        drop(store);

        // A restart: the index and the files agree, and hold what landed.
        let mut store = DiskKvStore::open(tmp.path(), "fp-abc123", 0, 0).unwrap();
        assert_eq!(store.len(), 1);
        assert_eq!(store.get("after").unwrap().map(|(_, r)| r), Some(rec(4.0)));
        assert_eq!(
            names_in(&dir),
            vec![DiskKvStore::file_for("after"), "index.json".to_string()]
        );
    }

    /// The full disk for real: an 8 MB APFS volume mounted from a file (no root
    /// needed). APFS fails a `write` part-way with ENOSPC and gives the space
    /// back when the file goes, which is what `disk_with_room` models, and the
    /// store behaves the same: the write is given up and leaves nothing behind,
    /// the state it would have replaced is served, and writes that fit go on.
    #[test]
    #[ignore = "mounts a disk image with hdiutil: macOS, outside a sandbox"]
    fn a_real_full_volume_gives_up_the_write_and_keeps_serving() {
        use std::process::Command;
        let tmp = tempfile::tempdir().unwrap();
        let (img, mnt) = (tmp.path().join("kv.dmg"), tmp.path().join("mnt"));
        fs::create_dir(&mnt).unwrap();
        let hdiutil = |args: &[&std::ffi::OsStr]| {
            assert!(
                Command::new("hdiutil")
                    .args(args)
                    .status()
                    .unwrap()
                    .success(),
                "hdiutil {args:?}"
            );
        };
        let os = |s: &'static str| std::ffi::OsStr::new(s);
        hdiutil(&[
            os("create"),
            os("-quiet"),
            os("-size"),
            os("8m"),
            os("-fs"),
            os("APFS"),
            img.as_os_str(),
        ]);
        hdiutil(&[
            os("attach"),
            os("-quiet"),
            os("-nobrowse"),
            os("-noverify"),
            os("-mountpoint"),
            mnt.as_os_str(),
            img.as_os_str(),
        ]);
        struct Detach(PathBuf);
        impl Drop for Detach {
            fn drop(&mut self) {
                let _ = Command::new("hdiutil")
                    .args(["detach", "-quiet", "-force"])
                    .arg(&self.0)
                    .status();
            }
        }
        let _detach = Detach(mnt.clone());
        let drained = || assert!(drain_writes(std::time::Duration::from_secs(30)));
        let m = sample_manifest();
        let small = vec![f32_record(&[1, 2, 2, 4], 1.0)];
        let big = vec![f32_record(&[1, 1024, 1024, 4], 0.0)]; // 16 MB: more than the volume
        let dir = mnt.join("fp-abc123");

        let mut store = DiskKvStore::open(&mnt, "fp-abc123", 0, 0).unwrap();
        store.write_in_background().unwrap();
        store.put("kept", &m, &small).unwrap();
        drained();
        store.put("big", &m, &big).unwrap();
        store.supersede("kept", "big").unwrap();
        drained();
        assert!(
            !store.contains("big"),
            "a write the volume refused is listed"
        );
        assert!(store.get("big").unwrap().is_none());
        assert_eq!(
            store.get("kept").unwrap().map(|(_, r)| r),
            Some(small.clone())
        );
        drained(); // the reads queued index writes
        assert_eq!(
            names_in(&dir),
            vec!["index.json".to_string(), DiskKvStore::file_for("kept")]
        );
        // The payload failed and the index writes after it landed: one line for
        // the failure, one for writes landing again.
        assert_eq!(store.write_report_for_test(), (0, 2));
        store.put("after", &m, &small).unwrap();
        drained();
        assert!(store.get("after").unwrap().is_some());
        drop(store);

        let store = DiskKvStore::open(&mnt, "fp-abc123", 0, 0).unwrap();
        assert_eq!(store.len(), 2);
        assert_eq!(
            names_in(&dir),
            vec![
                DiskKvStore::file_for("after"),
                "index.json".to_string(),
                DiskKvStore::file_for("kept")
            ]
        );
    }

    /// A server stops right after a turn: the newer state's write is queued
    /// and the store is never asked again, so only the index records that it
    /// supersedes the older one. The next open finishes it — retires the older
    /// state when the newer one is on the disk, and keeps it when the newer
    /// one never landed.
    #[test]
    fn a_supersede_still_pending_at_exit_is_finished_by_the_next_open() {
        use std::sync::atomic::Ordering::SeqCst;
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("fp-abc123");
        let m = sample_manifest();
        let rec = |v: f32| vec![f32_record(&[1, 2, 2, 4], v)];
        let room = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(u64::MAX));
        let session = |steps: &dyn Fn(&mut DiskKvStore)| {
            let mut store = DiskKvStore::open(tmp.path(), "fp-abc123", 0, 0).unwrap();
            store.set_create_for_test(disk_with_room(room.clone()));
            store.write_in_background().unwrap();
            steps(&mut store);
            // Dropped without another call: the writer finishes its queue.
        };

        session(&|s| {
            s.put("a", &m, &rec(1.0)).unwrap();
            assert!(drain_writes(std::time::Duration::from_secs(30)));
            s.put("b", &m, &rec(2.0)).unwrap();
            s.supersede("a", "b").unwrap();
        });
        let mut store = DiskKvStore::open(tmp.path(), "fp-abc123", 0, 0).unwrap();
        assert_eq!(
            store.len(),
            1,
            "the superseded state outlived its replacement"
        );
        assert_eq!(store.get("b").unwrap().map(|(_, r)| r), Some(rec(2.0)));
        assert!(!dir.join("a.lkv").exists());
        drop(store);

        // Room for the index, not for the newer state.
        let mut probe = Vec::new();
        write_lkv(&mut probe, &m, &rec(3.0)).unwrap();
        room.store(probe.len() as u64 - 1, SeqCst);
        session(&|s| {
            s.put("c", &m, &rec(3.0)).unwrap();
            s.supersede("b", "c").unwrap();
        });
        let index = fs::read_to_string(dir.join("index.json")).unwrap();
        assert!(
            index.contains("superseded_by"),
            "the index never recorded the supersede, so this proves nothing: {index}"
        );
        let mut store = DiskKvStore::open(tmp.path(), "fp-abc123", 0, 0).unwrap();
        assert_eq!(
            store.get("b").unwrap().map(|(_, r)| r),
            Some(rec(2.0)),
            "the last state on the disk went for one that never got there"
        );
        assert!(!store.contains("c"));
        assert_eq!(
            names_in(&dir),
            vec![DiskKvStore::file_for("b"), "index.json".to_string()]
        );
    }

    /// Inline writes (tests, tools) report a full disk to the caller, and the
    /// half-written temp file does not stay behind taking the space.
    #[test]
    fn a_full_disk_fails_an_inline_put_without_leaving_its_temp_file() {
        use std::sync::atomic::Ordering::SeqCst;
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("fp-abc123");
        let m = sample_manifest();
        let records = vec![f32_record(&[1, 2, 2, 4], 1.0)];
        let room = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(100));
        let mut store = DiskKvStore::open(tmp.path(), "fp-abc123", 0, 0).unwrap();
        store.set_create_for_test(disk_with_room(room.clone()));

        let err = store.put("k", &m, &records).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("kv_disk: write"), "{msg}");
        assert!(store.get("k").unwrap().is_none());
        assert_eq!(names_in(&dir), Vec::<String>::new(), "something was left");

        room.store(u64::MAX, SeqCst);
        store.put("k", &m, &records).unwrap();
        assert_eq!(store.get("k").unwrap().map(|(_, r)| r), Some(records));
    }

    /// A payload whose index row never reached the disk (the index write
    /// failed, or the process stopped between the two) is unreachable: no key
    /// leads to it. Opening the store deletes it rather than keep the space.
    #[test]
    fn open_deletes_payloads_the_index_does_not_list() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("fp-abc123");
        let m = sample_manifest();
        let records = vec![f32_record(&[1, 2, 2, 4], 1.0)];
        {
            let mut store = DiskKvStore::open(tmp.path(), "fp-abc123", 0, 0).unwrap();
            store.put("a", &m, &records).unwrap();
        }
        fs::copy(dir.join("a.lkv"), dir.join("orphan.lkv")).unwrap();
        fs::write(dir.join("index.json.tmp"), b"{").unwrap();
        let mut store = DiskKvStore::open(tmp.path(), "fp-abc123", 0, 0).unwrap();
        assert_eq!(
            names_in(&dir),
            vec!["a.lkv".to_string(), "index.json".to_string()]
        );
        assert_eq!(store.get("a").unwrap().map(|(_, r)| r), Some(records));
    }

    #[test]
    fn disk_store_index_persists_across_reopen() {
        let tmp = tempfile::tempdir().unwrap();
        let m = sample_manifest();
        let records = vec![f32_record(&[1, 1, 1, 4], 2.0)];
        {
            let mut store = DiskKvStore::open(tmp.path(), "fp-abc123", 0, 0).unwrap();
            store.put("k1", &m, &records).unwrap();
        }
        let mut store = DiskKvStore::open(tmp.path(), "fp-abc123", 0, 0).unwrap();
        assert_eq!(store.len(), 1);
        assert!(store.get("k1").unwrap().is_some());
    }

    #[test]
    fn disk_store_fingerprint_mismatch_is_miss() {
        let tmp = tempfile::tempdir().unwrap();
        let m = sample_manifest(); // fingerprint fp-abc123
        let records = vec![f32_record(&[1, 1, 1, 4], 2.0)];
        {
            let mut store = DiskKvStore::open(tmp.path(), "fp-abc123", 0, 0).unwrap();
            store.put("k1", &m, &records).unwrap();
        }
        // Reopen under a DIFFERENT fingerprint dir — different model.
        let mut other = DiskKvStore::open(tmp.path(), "fp-DIFFERENT", 0, 0).unwrap();
        assert!(other.get("k1").unwrap().is_none());
    }

    // --- MLX Array <-> ArrayRecord conversion (requires a Metal device) ---

    #[cfg(feature = "mlx-native")]
    mod mlx {
        use super::super::{DtypeTag, record_from_array, record_to_array};
        use mlx_rs::{Array, Dtype};

        #[test]
        #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
        fn f32_array_record_roundtrip_is_bit_identical() {
            let data: Vec<f32> = (0..(2 * 3 * 4)).map(|i| i as f32 * 0.25 - 1.0).collect();
            let a = Array::from_slice(&data, &[2, 3, 4]);
            let rec = record_from_array(&a).unwrap();
            assert_eq!(rec.dtype, DtypeTag::F32);
            assert_eq!(rec.shape, vec![2, 3, 4]);
            let b = record_to_array(&rec).unwrap();
            b.eval().unwrap();
            assert_eq!(b.try_as_slice::<f32>().unwrap(), data.as_slice());
        }

        #[test]
        #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
        fn bf16_array_record_roundtrip_is_bit_identical() {
            // Arbitrary bit patterns, not just "nice" values: zeros of both
            // signs, a subnormal, ±inf, a NaN with a payload, extremes. With
            // no cast in the path, every one must come back bit for bit.
            let bits: Vec<u16> = vec![
                0x0000, 0x8000, 0x0001, 0x3f80, 0xc000, 0x7f80, 0xff80, 0x7fc1, 0x7f7f, 0x1234,
                0xabcd, 0x0080,
            ];
            for (dtype, tag) in [
                (Dtype::Bfloat16, DtypeTag::BF16),
                (Dtype::Float16, DtypeTag::F16),
            ] {
                let a = Array::from_slice(&bits, &[1, 1, 3, 4])
                    .view_dtype(dtype)
                    .unwrap();
                let rec = record_from_array(&a).unwrap();
                assert_eq!(rec.dtype, tag);
                assert_eq!(
                    rec.data.len(),
                    bits.len() * 2,
                    "{tag:?} must stay 2 bytes per element"
                );
                let b = record_to_array(&rec).unwrap();
                assert_eq!(b.dtype(), dtype);
                assert_eq!(b.shape(), &[1, 1, 3, 4]);
                let back = b.view::<u16>().unwrap();
                back.eval().unwrap();
                assert_eq!(
                    back.try_as_slice::<u16>().unwrap(),
                    bits.as_slice(),
                    "{tag:?}"
                );
            }
        }

        #[test]
        #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
        fn bool_array_record_roundtrip() {
            let data: Vec<u8> = vec![1, 0, 0, 1, 1, 1, 0, 0];
            let a = Array::from_slice(&data, &[2, 4])
                .as_dtype(Dtype::Bool)
                .unwrap();
            let rec = record_from_array(&a).unwrap();
            assert_eq!(rec.dtype, DtypeTag::Bool);
            let b = record_to_array(&rec).unwrap();
            assert_eq!(b.dtype(), Dtype::Bool);
            let b8 = b.as_dtype(Dtype::Uint8).unwrap();
            b8.eval().unwrap();
            assert_eq!(b8.try_as_slice::<u8>().unwrap(), data.as_slice());
        }

        #[test]
        #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
        fn u8_array_record_roundtrip() {
            let data: Vec<u8> = (0..64u32).map(|i| (i % 251) as u8).collect();
            let a = Array::from_slice(&data, &[1, 2, 4, 8]);
            let rec = record_from_array(&a).unwrap();
            assert_eq!(rec.dtype, DtypeTag::U8);
            let b = record_to_array(&rec).unwrap();
            b.eval().unwrap();
            assert_eq!(b.try_as_slice::<u8>().unwrap(), data.as_slice());
        }

        #[test]
        #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
        fn u32_array_record_roundtrip() {
            let data: Vec<u32> = (0..32u32).map(|i| i.wrapping_mul(2_654_435_761)).collect();
            let a = Array::from_slice(&data, &[1, 1, 4, 8]);
            let rec = record_from_array(&a).unwrap();
            assert_eq!(rec.dtype, DtypeTag::U32);
            let b = record_to_array(&rec).unwrap();
            b.eval().unwrap();
            assert_eq!(b.try_as_slice::<u32>().unwrap(), data.as_slice());
        }
    }

    #[test]
    fn disk_store_lru_evicts_oldest_over_budget() {
        let tmp = tempfile::tempdir().unwrap();
        let m = sample_manifest();
        // Each record blob is a few hundred bytes; set a tiny budget so only
        // ~1-2 entries fit.
        let big = vec![f32_record(&[1, 4, 16, 16], 0.0)]; // 16384 bytes of f32
        let one_size = {
            let mut buf = Vec::new();
            write_lkv(&mut buf, &m, &big).unwrap();
            buf.len() as u64
        };
        let mut store = DiskKvStore::open(tmp.path(), "fp-abc123", one_size + 10, 0).unwrap();

        store.put("oldest", &m, &big).unwrap();
        // bump clock ordering is by insertion since now_unix may be equal within
        // a second; rely on insertion order + min_by_key stability.
        store.put("newest", &m, &big).unwrap();

        assert!(store.total_bytes() <= one_size + 10);
        // Only the newest should survive (budget fits ~1).
        assert!(store.get("newest").unwrap().is_some());
        assert!(store.get("oldest").unwrap().is_none());
    }

    #[test]
    fn disk_store_ttl_prunes_stale_entries() {
        let tmp = tempfile::tempdir().unwrap();
        let m = sample_manifest();
        let records = vec![f32_record(&[1, 1, 1, 4], 1.0)];
        let mut store = DiskKvStore::open(tmp.path(), "fp-abc123", 0, 100).unwrap();
        store.put("k1", &m, &records).unwrap();
        assert_eq!(store.len(), 1);
        // Backdate the entry well beyond the 100s TTL, then prune.
        store.index.entries[0].last_access_unix = 1;
        assert!(store.prune_ttl());
        assert_eq!(store.len(), 0);
        assert!(store.get("k1").unwrap().is_none());
    }

    #[test]
    fn disk_store_ttl_zero_keeps_forever() {
        let tmp = tempfile::tempdir().unwrap();
        let m = sample_manifest();
        let records = vec![f32_record(&[1, 1, 1, 4], 1.0)];
        let mut store = DiskKvStore::open(tmp.path(), "fp-abc123", 0, 0).unwrap();
        store.put("k1", &m, &records).unwrap();
        store.index.entries[0].last_access_unix = 1; // ancient
        assert!(!store.prune_ttl(), "ttl=0 must never prune");
        assert_eq!(store.len(), 1);
    }

    // --- Defaults: on, 10 GB, 30 days, a 10 GB free-space floor ---

    /// Unset is on: the tier is a registered flag whose default is on, and
    /// `with(false)` (what `LUMEN_KV_DISK=0` gives) opens nothing, before
    /// anything is read from or made on the disk.
    #[test]
    fn the_disk_tier_is_on_unless_turned_off() {
        let d = kv_disk_tier::describe();
        assert_eq!(d.env, "LUMEN_KV_DISK");
        assert!(d.default, "unset LUMEN_KV_DISK must mean on");
        assert_eq!(d.kind, lumen_flags::FlagKind::Optimization);
        kv_disk_tier::with(false, || {
            assert!(open_store_from_env("never-opened", "test").is_none());
        });
    }

    #[test]
    fn the_limits_default_to_10_gb_30_days_and_a_10_gb_floor() {
        let unset = DiskTierLimits::parse(None, None, None);
        assert_eq!(
            unset,
            DiskTierLimits {
                max_bytes: 10 * GIB,
                ttl_secs: 30 * 24 * 60 * 60,
                min_free_bytes: 10 * GIB,
            }
        );
        // Empty or unreadable is the default too, not zero (which means
        // unlimited / never / no floor).
        assert_eq!(DiskTierLimits::parse(Some(""), Some(" "), Some("")), unset);
        assert_eq!(
            DiskTierLimits::parse(Some("ten"), Some("-1"), Some("1.5")),
            unset
        );
    }

    #[test]
    fn explicit_limits_win_and_an_explicit_zero_keeps_its_meaning() {
        assert_eq!(
            DiskTierLimits::parse(Some("100"), Some("3600"), Some(" 5 ")),
            DiskTierLimits {
                max_bytes: 100 * GIB,
                ttl_secs: 3600,
                min_free_bytes: 5 * GIB,
            }
        );
        // 0 = unlimited budget, entries that never expire, no floor.
        assert_eq!(
            DiskTierLimits::parse(Some("0"), Some("0"), Some("0")),
            DiskTierLimits {
                max_bytes: 0,
                ttl_secs: 0,
                min_free_bytes: 0,
            }
        );
        // A size too large to count in bytes saturates instead of wrapping
        // to a tiny budget.
        let huge = DiskTierLimits::parse(Some(&u64::MAX.to_string()), None, None);
        assert_eq!(huge.max_bytes, u64::MAX);
    }

    /// What the server opens is what the limits say: the budget, the TTL and
    /// the floor all reach the store.
    #[test]
    fn the_opened_store_carries_every_limit() {
        let tmp = tempfile::tempdir().unwrap();
        let limits = DiskTierLimits::parse(None, None, None);
        let store = open_store(tmp.path(), "fp-abc123", "test", limits).expect("opens");
        assert_eq!(store.max_bytes, 10 * GIB);
        assert_eq!(store.ttl_secs, 30 * 24 * 60 * 60);
        assert_eq!(store.min_free_bytes, 10 * GIB);
        assert!(
            store.writer.is_some(),
            "the server writes in the background"
        );
    }

    // --- The free-space floor ---

    /// A volume whose free space is whatever `free` holds.
    fn volume_with_free(free: std::sync::Arc<std::sync::atomic::AtomicU64>) -> FreeSpace {
        std::sync::Arc::new(move |_| Ok(free.load(std::sync::atomic::Ordering::SeqCst)))
    }

    /// Below the floor a write is skipped, and nothing else moves: the entry
    /// it would have replaced and every other one stay, on disk and readable.
    /// Above it writes go on. A run of skips is logged once, and so is the
    /// write that ends it.
    #[test]
    fn the_floor_skips_writes_below_it_and_keeps_what_is_stored() {
        use std::sync::atomic::Ordering::SeqCst;
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("fp-abc123");
        let m = sample_manifest();
        let rec = |v: f32| vec![f32_record(&[1, 2, 2, 4], v)];
        let bytes = rec(0.0)[0].data.len() as u64;
        let floor = 1_000_000;
        let free = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(u64::MAX));

        let mut store = DiskKvStore::open(tmp.path(), "fp-abc123", 0, 0).unwrap();
        store.set_free_space_for_test(volume_with_free(free.clone()));
        store.set_min_free_bytes(floor);
        store.put("kept", &m, &rec(1.0)).unwrap();
        store.put("other", &m, &rec(2.0)).unwrap();
        let on_disk = names_in(&dir);

        // One byte short of leaving the floor free.
        free.store(floor + bytes - 1, SeqCst);
        store.put("skipped", &m, &rec(3.0)).unwrap();
        store.put("kept", &m, &rec(4.0)).unwrap();
        store.put("skipped-too", &m, &rec(5.0)).unwrap();
        assert!(!store.contains("skipped") && !store.contains("skipped-too"));
        assert_eq!(names_in(&dir), on_disk, "a skipped write touched the disk");
        assert_eq!(store.len(), 2);
        assert_eq!(
            store.get("kept").unwrap().map(|(_, r)| r),
            Some(rec(1.0)),
            "the stored state must survive a newer one the floor held back"
        );
        assert_eq!(store.get("other").unwrap().map(|(_, r)| r), Some(rec(2.0)));
        assert_eq!(
            store.floor_report_for_test(),
            (3, 1),
            "three skips, one line"
        );

        // Exactly the floor left free is allowed.
        free.store(floor + bytes, SeqCst);
        store.put("fits", &m, &rec(6.0)).unwrap();
        assert_eq!(store.get("fits").unwrap().map(|(_, r)| r), Some(rec(6.0)));
        assert_eq!(
            store.floor_report_for_test(),
            (0, 2),
            "one more line: writes resume"
        );
        store.put("kept", &m, &rec(7.0)).unwrap();
        assert_eq!(store.get("kept").unwrap().map(|(_, r)| r), Some(rec(7.0)));
        assert_eq!(store.floor_report_for_test(), (0, 2));
        drop(store);

        // A restart finds exactly what was written.
        let mut store = DiskKvStore::open(tmp.path(), "fp-abc123", 0, 0).unwrap();
        assert_eq!(store.len(), 3);
        assert_eq!(store.get("kept").unwrap().map(|(_, r)| r), Some(rec(7.0)));
        assert!(!store.contains("skipped"));
    }

    /// The server's mode: a write still queued for the writer thread counts
    /// as written, so two writes in a row cannot both pass a check that only
    /// one of them fits. The free space here is a disk that fills as the
    /// writer lands bytes on it, so either way the second write is held back.
    #[test]
    fn the_floor_counts_writes_still_queued() {
        use std::sync::atomic::Ordering::SeqCst;
        let tmp = tempfile::tempdir().unwrap();
        let m = sample_manifest();
        let big = |v: f32| vec![f32_record(&[1, 64, 64, 4], v)]; // 64 KiB
        let bytes = big(0.0)[0].data.len() as u64;
        let floor = 1_000_000;
        let room = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(floor + bytes * 3 / 2));

        let mut store = DiskKvStore::open(tmp.path(), "fp-abc123", 0, 0).unwrap();
        store.set_create_for_test(disk_with_room(room.clone()));
        store.set_free_space_for_test(volume_with_free(room.clone()));
        store.set_min_free_bytes(floor);
        store.write_in_background().unwrap();
        store.put("first", &m, &big(1.0)).unwrap();
        store.put("second", &m, &big(2.0)).unwrap();
        assert!(drain_writes(std::time::Duration::from_secs(30)));
        assert!(store.get("first").unwrap().is_some());
        assert!(!store.contains("second"), "both writes passed the floor");
        assert_eq!(store.floor_report_for_test(), (1, 1));

        room.store(u64::MAX, SeqCst);
        store.put("second", &m, &big(2.0)).unwrap();
        assert!(drain_writes(std::time::Duration::from_secs(30)));
        assert_eq!(store.get("second").unwrap().map(|(_, r)| r), Some(big(2.0)));
        assert_eq!(store.floor_report_for_test(), (0, 2));
    }

    /// No floor reads no free space at all, and a volume whose free space
    /// cannot be read is not held back (a write that does not fit then fails
    /// as on a full disk).
    #[test]
    fn no_floor_or_an_unreadable_volume_does_not_hold_writes_back() {
        let tmp = tempfile::tempdir().unwrap();
        let m = sample_manifest();
        let records = vec![f32_record(&[1, 2, 2, 4], 1.0)];
        let mut store = DiskKvStore::open(tmp.path(), "fp-abc123", 0, 0).unwrap();
        store.set_free_space_for_test(std::sync::Arc::new(|_| -> std::io::Result<u64> {
            panic!("no floor must not read the free space")
        }));
        store.put("no-floor", &m, &records).unwrap();
        assert!(store.contains("no-floor"));

        store.set_min_free_bytes(u64::MAX);
        store.set_free_space_for_test(std::sync::Arc::new(|_| {
            Err(std::io::ErrorKind::Other.into())
        }));
        store.put("unreadable", &m, &records).unwrap();
        assert!(store.contains("unreadable"));
        assert_eq!(store.floor_report_for_test(), (0, 0));
    }

    /// The real reading: a temp directory's volume reports some free space,
    /// and a path that does not exist is an error, not zero.
    #[test]
    fn the_volume_free_space_reads_statvfs() {
        let tmp = tempfile::tempdir().unwrap();
        let free = volume_free_bytes(tmp.path()).expect("statvfs on a temp dir");
        assert!(free > 0, "a writable temp dir on a volume with no space");
        assert!(volume_free_bytes(&tmp.path().join("missing")).is_err());
    }
}
