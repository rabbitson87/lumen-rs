//! P0 — on-disk KV-cache persistence primitives (shared core).
//!
//! This module is the backend-agnostic foundation for persisting prompt-cache
//! snapshots to disk (Phase 1: persistence across restart/idle/evict; Phase 3:
//! spill under memory pressure). It is intentionally split into two layers:
//!
//! * **Pure layer** (no MLX) — the on-disk binary format (`LKV1`), the
//!   `KvManifest`/`LayerMeta` metadata, the `ArrayRecord` byte container, and
//!   the `DiskKvStore` (directory + LRU budget + `index.json`). These compile
//!   and unit-test under the crate's default features (no Metal device needed),
//!   mirroring the `lloyd_max.rs` / `rotation.rs` TQCB/TQRM save-load pattern.
//!
//! * **MLX layer** (`#[cfg(feature = "mlx-native")]`) — conversion between an
//!   mlx-rs `Array` and an `ArrayRecord`. Float dtypes (bf16/f16/f32) are
//!   stored as little-endian f32; because bf16/f16 are exact subsets of f32,
//!   the round trip is bit-identical. Integer dtypes are stored in their
//!   native little-endian width. These paths require a live Metal device, so
//!   their tests are `#[ignore]` like the rest of the native suite.
//!
//! Snapshot coupling (turning a `PromptCacheSnapshot` into a `(KvManifest,
//! Vec<ArrayRecord>)` and back) and the `PrefixCacheStore` disk tier are
//! deliberately NOT here yet — that is Phase 1. P0 only provides the proven,
//! independently-testable primitives those phases build on.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};

/// File magic — `b"LKV1"` (Lumen KV, format generation 1).
pub const LKV_MAGIC: [u8; 4] = *b"LKV1";
/// Format version, bumped on incompatible layout changes.
pub const LKV_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Dtype tagging
// ---------------------------------------------------------------------------

/// Logical element dtype recorded for an array. This is the dtype the array is
/// *reconstructed* as; the physical on-disk encoding may differ (float dtypes
/// are stored as f32 — see `stored_elem_size`).
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
    /// Whether the logical dtype is a floating-point type stored via f32.
    #[inline]
    pub fn is_float(self) -> bool {
        matches!(
            self,
            DtypeTag::F16 | DtypeTag::F32 | DtypeTag::F64 | DtypeTag::BF16
        )
    }

    /// Physical bytes-per-element on disk. Float dtypes ≤ f32 are normalized to
    /// f32 (4 bytes) — bf16/f16 are exact subsets of f32, so this is lossless.
    /// f64 keeps its native width (not used by KV caches; encoder bails). All
    /// integers/bool use their native width.
    #[inline]
    pub fn stored_elem_size(self) -> usize {
        match self {
            DtypeTag::Bool | DtypeTag::U8 | DtypeTag::I8 => 1,
            DtypeTag::U16 | DtypeTag::I16 => 2,
            DtypeTag::F16 | DtypeTag::BF16 | DtypeTag::F32 | DtypeTag::U32 | DtypeTag::I32 => 4,
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

/// One serialized tensor. `data` holds the physical little-endian bytes in the
/// encoding implied by `dtype` (float dtypes ⇒ f32 words; integer dtypes ⇒
/// native width). `shape` is the logical mlx shape (row-major).
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

    fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        self.validate_len()?;
        w.write_all(&[self.dtype.to_u8()])?;
        w.write_all(&(self.shape.len() as u32).to_le_bytes())?;
        for &d in &self.shape {
            w.write_all(&d.to_le_bytes())?;
        }
        w.write_all(&(self.data.len() as u64).to_le_bytes())?;
        w.write_all(&self.data)?;
        Ok(())
    }

    fn read_from<R: Read>(r: &mut R) -> Result<Self> {
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
            shape.push(i32::from_le_bytes(sb));
        }
        let data_len = read_u64(r)? as usize;
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

/// Serialize a `(KvManifest, records)` pair to the `LKV1` binary layout.
///
/// Layout:
/// ```text
/// [4]  magic "LKV1"
/// [4]  version  u32 LE
/// [4]  manifest_json_len  u32 LE
/// [N]  manifest JSON (utf-8)
/// [4]  record_count  u32 LE
/// records[]: [1] dtype | [4] ndim | [ndim*4] shape i32 | [8] data_len u64 | [data_len] bytes
/// ```
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

    w.write_all(&(records.len() as u32).to_le_bytes())?;
    for rec in records {
        rec.write_to(w)?;
    }
    Ok(())
}

/// Deserialize the `LKV1` binary layout back into `(KvManifest, records)`.
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
    let manifest: KvManifest =
        serde_json::from_slice(&json).context("kv_disk: deserialize manifest")?;

    let n = read_u32(r)? as usize;
    let mut records = Vec::with_capacity(n);
    for _ in 0..n {
        records.push(ArrayRecord::read_from(r)?);
    }
    Ok((manifest, records))
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
        };
        // Drop entries that expired while the process was down.
        if store.prune_ttl() {
            let _ = store.save_index();
        }
        Ok(store)
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
            let _ = fs::remove_file(self.dir.join(f));
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

        let tmp = path.with_extension("lkv.tmp");
        fs::write(&tmp, &buf).with_context(|| format!("kv_disk: write {}", tmp.display()))?;
        fs::rename(&tmp, &path).with_context(|| format!("kv_disk: rename {}", path.display()))?;

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
            let _ = fs::remove_file(&path);
            self.index.entries.remove(pos);
            self.save_index()?;
        }
        Ok(())
    }

    /// Drop all entries and their files.
    pub fn clear(&mut self) -> Result<()> {
        for e in std::mem::take(&mut self.index.entries) {
            let _ = fs::remove_file(self.dir.join(&e.file));
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
            let _ = fs::remove_file(self.dir.join(&e.file));
        }
        Ok(())
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

    /// Read an mlx `Array` into an `ArrayRecord`.
    ///
    /// Float dtypes (bf16/f16/f32) are materialized through f32 and stored as
    /// little-endian f32 words — lossless because bf16/f16 are exact subsets of
    /// f32. Integer dtypes are read in their native width. The array is `eval`'d
    /// before readback so lazy graphs are materialized contiguously.
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

        let data = if tag.is_float() {
            let f = a_flat
                .as_dtype(Dtype::Float32)
                .context("kv_disk: cast array to f32 for readback")?;
            f.eval().context("kv_disk: eval f32 array")?;
            let s = f
                .try_as_slice::<f32>()
                .map_err(|e| anyhow!("kv_disk: as_slice f32: {e:?}"))?;
            let mut data = Vec::with_capacity(s.len() * 4);
            for v in s {
                data.extend_from_slice(&v.to_le_bytes());
            }
            data
        } else {
            let e = a_flat;
            e.eval().context("kv_disk: eval int array")?;
            match tag {
                DtypeTag::U8 | DtypeTag::Bool => e
                    .try_as_slice::<u8>()
                    .map_err(|e| anyhow!("kv_disk: as_slice u8: {e:?}"))?
                    .to_vec(),
                DtypeTag::U32 => {
                    let s = e
                        .try_as_slice::<u32>()
                        .map_err(|e| anyhow!("kv_disk: as_slice u32: {e:?}"))?;
                    let mut d = Vec::with_capacity(s.len() * 4);
                    for v in s {
                        d.extend_from_slice(&v.to_le_bytes());
                    }
                    d
                }
                DtypeTag::I32 => {
                    let s = e
                        .try_as_slice::<i32>()
                        .map_err(|e| anyhow!("kv_disk: as_slice i32: {e:?}"))?;
                    let mut d = Vec::with_capacity(s.len() * 4);
                    for v in s {
                        d.extend_from_slice(&v.to_le_bytes());
                    }
                    d
                }
                other => bail!("kv_disk: int readback for {other:?} not implemented in P0"),
            }
        };

        let rec = ArrayRecord {
            dtype: tag,
            shape,
            data,
        };
        rec.validate_len()?;
        Ok(rec)
    }

    /// Reconstruct an mlx `Array` from an `ArrayRecord`. Float records are read
    /// as f32 then cast back to the recorded dtype (bit-identical for bf16/f16).
    pub fn record_to_array(rec: &ArrayRecord) -> Result<Array> {
        rec.validate_len()?;
        let arr = if rec.dtype.is_float() {
            let n = rec.data.len() / 4;
            let mut v = Vec::with_capacity(n);
            for c in rec.data.chunks_exact(4) {
                v.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
            let a = Array::from_slice(&v, &rec.shape);
            if rec.dtype == DtypeTag::F32 {
                a
            } else {
                a.as_dtype(rec.dtype.to_mlx())
                    .context("kv_disk: cast f32 record back to target dtype")?
            }
        } else {
            match rec.dtype {
                DtypeTag::U8 | DtypeTag::Bool => Array::from_slice(&rec.data, &rec.shape),
                DtypeTag::U32 => {
                    let n = rec.data.len() / 4;
                    let mut v = Vec::with_capacity(n);
                    for c in rec.data.chunks_exact(4) {
                        v.push(u32::from_le_bytes([c[0], c[1], c[2], c[3]]));
                    }
                    Array::from_slice(&v, &rec.shape)
                }
                DtypeTag::I32 => {
                    let n = rec.data.len() / 4;
                    let mut v = Vec::with_capacity(n);
                    for c in rec.data.chunks_exact(4) {
                        v.push(i32::from_le_bytes([c[0], c[1], c[2], c[3]]));
                    }
                    Array::from_slice(&v, &rec.shape)
                }
                other => bail!("kv_disk: int reconstruct for {other:?} not implemented in P0"),
            }
        };
        // Materialize before returning. The float path ends in a lazy `as_dtype`
        // node over a fresh `from_slice` buffer; a rotating KV cache's in-place
        // `slice_update` fast path (Gemma4 sliding decode) misbehaves when its
        // backing buffer is an unevaluated graph node rather than a concrete
        // array. Forcing eval here makes reconstructed caches behave like cloned
        // (already-evaluated) ones. Cheap — this is the cold reload path.
        arr.eval().context("kv_disk: eval reconstructed array")?;
        Ok(arr)
    }
}

#[cfg(feature = "mlx-native")]
pub use mlx_conv::{record_from_array, record_to_array};

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
            // Values exactly representable in bf16 so the cast chain is lossless.
            let data: Vec<f32> = vec![0.0, 1.0, -2.0, 0.5, -0.25, 128.0, -1.5, 3.0];
            let a32 = Array::from_slice(&data, &[1, 1, 2, 4]);
            let a = a32.as_dtype(Dtype::Bfloat16).unwrap();
            let rec = record_from_array(&a).unwrap();
            assert_eq!(rec.dtype, DtypeTag::BF16);
            let b = record_to_array(&rec).unwrap();
            assert_eq!(b.dtype(), Dtype::Bfloat16);
            let b32 = b.as_dtype(Dtype::Float32).unwrap();
            b32.eval().unwrap();
            assert_eq!(b32.try_as_slice::<f32>().unwrap(), data.as_slice());
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
