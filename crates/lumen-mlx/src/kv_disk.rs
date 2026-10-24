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
}

impl DiskKvStore {
    /// Open (creating if needed) a store under `root/<fingerprint>/`.
    /// `max_bytes` is the LRU budget (0 = unbounded). `ttl_secs` is the
    /// last-access TTL (0 = no expiry). Stale entries are pruned on open.
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
        };
        // Drop entries that expired while the process was down, and whatever a
        // process that died mid-write left behind: half-written temp files,
        // and index rows whose file never landed.
        let mut changed = store.prune_ttl();
        if let Ok(dir) = fs::read_dir(&store.dir) {
            for e in dir.flatten() {
                if e.file_name().to_string_lossy().ends_with(".lkv.tmp") {
                    let _ = fs::remove_file(e.path());
                }
            }
        }
        let before = store.index.entries.len();
        let dir = store.dir.clone();
        store.index.entries.retain(|e| dir.join(&e.file).exists());
        changed |= store.index.entries.len() != before;
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
            self.writer = Some(FileWriter::spawn()?);
        }
        Ok(())
    }

    /// Write `bytes` to `path` through a temp file and a rename, so a reader
    /// never sees a partial file — inline, or queued on the background thread.
    fn write_file(&self, path: PathBuf, bytes: Vec<u8>) -> Result<()> {
        match &self.writer {
            Some(w) => {
                w.submit(FileJob::Write { path, bytes });
                Ok(())
            }
            None => write_via_tmp(&path, &bytes),
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

    fn save_index(&self) -> Result<()> {
        let path = Self::index_path(&self.dir);
        let bytes = serde_json::to_vec(&self.index).context("kv_disk: serialize index")?;
        // Atomic-ish: write to temp then rename.
        let tmp = path.with_extension("json.tmp");
        fs::write(&tmp, &bytes).with_context(|| format!("kv_disk: write {}", tmp.display()))?;
        fs::rename(&tmp, &path)
            .with_context(|| format!("kv_disk: rename index {}", path.display()))?;
        Ok(())
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

    /// Persist a snapshot under `key`, replacing any existing entry. Evicts LRU
    /// victims afterward to honor the byte budget.
    pub fn put(&mut self, key: &str, manifest: &KvManifest, records: &[ArrayRecord]) -> Result<()> {
        let file = Self::file_for(key);
        let path = self.dir.join(&file);

        let mut buf = Vec::new();
        write_lkv(&mut buf, manifest, records)?;
        let bytes = buf.len() as u64;

        self.write_file(path, buf)?;

        let now = now_unix();
        self.prune_ttl();
        self.index.entries.retain(|e| e.key != key);
        self.index.entries.push(DiskEntry {
            key: key.to_string(),
            file,
            bytes,
            last_access_unix: now,
            hits: 0,
        });

        self.evict_to_fit()?;
        self.save_index()?;
        Ok(())
    }

    /// Whether `key` is in the index. Reads no file, so probing many
    /// candidate keys (see [`point_key`]) costs nothing per miss.
    pub fn contains(&self, key: &str) -> bool {
        self.index.entries.iter().any(|e| e.key == key)
    }

    /// Mark `key` as used now, so the last-access TTL and the LRU budget keep an
    /// entry that is read from memory instead of from here. Returns whether the
    /// key is stored. Reads no file.
    pub fn touch(&mut self, key: &str) -> Result<bool> {
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
        if let Some(pos) = self.index.entries.iter().position(|e| e.key == key) {
            let path = self.dir.join(&self.index.entries[pos].file);
            self.delete_file(path);
            self.index.entries.remove(pos);
            self.save_index()?;
        }
        Ok(())
    }

    /// Drop all entries and their files.
    pub fn clear(&mut self) -> Result<()> {
        for e in std::mem::take(&mut self.index.entries) {
            self.delete_file(self.dir.join(&e.file));
        }
        self.save_index()?;
        Ok(())
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

/// Write `bytes` to `<path>.tmp`, then rename it onto `path`.
fn write_via_tmp(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("lkv.tmp");
    fs::write(&tmp, bytes).with_context(|| format!("kv_disk: write {}", tmp.display()))?;
    fs::rename(&tmp, path).with_context(|| format!("kv_disk: rename {}", path.display()))?;
    Ok(())
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
    Write { path: PathBuf, bytes: Vec<u8> },
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
    fn spawn() -> Result<Self> {
        let (tx, rx) = std::sync::mpsc::channel::<FileJob>();
        let handle = std::thread::Builder::new()
            .name("lumen-kv-disk-writer".into())
            .spawn(move || {
                for job in rx {
                    match job {
                        FileJob::Write { path, bytes } => {
                            if let Err(e) = write_via_tmp(&path, &bytes) {
                                // The index already lists the entry; a missing
                                // file reads as a miss and is dropped on the
                                // next open.
                                eprintln!("[kv-disk] background write failed: {e:#}");
                                let _ = fs::remove_file(path.with_extension("lkv.tmp"));
                            }
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

/// The opt-in L2 disk tier, opened from env under `identity`. Off unless
/// `LUMEN_KV_DISK` is truthy; `LUMEN_KV_DISK_MAX_GB` bounds the LRU budget
/// (0 / unset = unbounded) and `LUMEN_KV_DISK_TTL_SECS` the last-access TTL
/// (default 1 day, 0 = never). Open failure is logged and yields `None`: the
/// disk tier is an accelerator, never a reason to fail a load.
pub fn open_store_from_env(identity: &str, log_tag: &str) -> Option<DiskKvStore> {
    let enabled = matches!(
        std::env::var("LUMEN_KV_DISK")
            .ok()
            .map(|s| s.trim().to_ascii_lowercase())
            .as_deref(),
        Some("1" | "true" | "yes" | "on")
    );
    if !enabled {
        return None;
    }
    let max_bytes = std::env::var("LUMEN_KV_DISK_MAX_GB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(|gb| gb * 1024 * 1024 * 1024)
        .unwrap_or(0);
    let ttl_secs = std::env::var("LUMEN_KV_DISK_TTL_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(86_400);
    match DiskKvStore::open(DiskKvStore::default_root(), identity, max_bytes, ttl_secs)
        .and_then(|mut store| store.write_in_background().map(|()| store))
    {
        Ok(store) => {
            eprintln!(
                "[{log_tag}] kv-disk tier ON: identity={identity:?} entries={} max_gb={}",
                store.len(),
                max_bytes / (1024 * 1024 * 1024),
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
/// in-memory prefix snapshots (which are already persisted to disk and will
/// rehydrate on next access) to relieve pressure. `0` / unset disables spill.
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
}
