//! The contiguous expert pack: every expert's bytes in one record.
//!
//! In the checkpoint an expert is nine ranges (three projections, each weight, scales and
//! biases) 50 KB to 800 KB apiece, in nine different tensors, so reading a missing expert is
//! nine scattered `pread`s. The pack stores the same bytes as one 2.7 MB record per expert,
//! laid out layer by layer, so a miss is one sequential read. On Flash-Next that is 16% less
//! time per loading layer call in an isolated read benchmark, and it costs another 68 GB of
//! disk: the pack is an option, off unless asked for.
//!
//! The bytes are the checkpoint's own, never changed or recompressed, so everything computed
//! from them is bit-identical with or without the pack.
//!
//! File layout (`experts.lpack` next to the shards):
//!
//! ```text
//! [0, 8)      magic  "LUMENPK1"
//! [8, 16)     u64 LE  length of the JSON header that follows
//! [16, ..)    JSON    { version, fingerprint, stride, data_start, roles[], layers[] }
//! [data_start, ..)    records: layer after layer, expert after expert, `stride` bytes each
//!                     (the roles' bytes back to back, zero padded to a 16 KiB multiple)
//! ```
//!
//! The fingerprint hashes the shards' names and sizes and every expert range found in them, so
//! a pack made for other weights is rejected instead of serving plausible wrong ones.

use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};

use super::expert_store::{ExpertStore, LayerRanges, RoleRange};

/// File name of the pack inside a checkpoint directory.
pub(crate) const PACK_FILE: &str = "experts.lpack";
const MAGIC: &[u8; 8] = b"LUMENPK1";
/// Records start on this boundary (the Apple silicon page size).
const ALIGN: usize = 16 * 1024;
/// Room reserved for the header; the data starts right after it.
const HEADER_RESERVE: usize = 64 * 1024;
/// Free space to leave beyond the pack itself when building.
const SPARE_BYTES: u64 = 1 << 30;

/// Whether a checkpoint's expert pack is used, and whether a missing one is built.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum PackMode {
    /// Use the pack when it is there and matches the weights; never build one.
    #[default]
    Auto,
    /// Read the shards, ignoring any pack.
    Off,
    /// Use the pack, building it first (a few minutes, 68 GB) when it is missing.
    Build,
}

impl PackMode {
    /// `LUMEN_MOE_PACK`: `auto` (default), `off` or `build`.
    pub(crate) fn from_env() -> Result<Self> {
        Self::parse(std::env::var("LUMEN_MOE_PACK").ok().as_deref())
    }

    pub(crate) fn parse(raw: Option<&str>) -> Result<Self> {
        match raw.map(str::trim) {
            None | Some("") | Some("auto") => Ok(Self::Auto),
            Some("off") | Some("0") | Some("false") => Ok(Self::Off),
            Some("build") => Ok(Self::Build),
            Some(other) => bail!("LUMEN_MOE_PACK={other:?} is not `auto`, `off` or `build`"),
        }
    }
}

pub(crate) fn pack_path(dir: &Path) -> PathBuf {
    dir.join(PACK_FILE)
}

/// FNV-1a over a byte stream: stable across builds, which `DefaultHasher` is not.
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
    fn bytes(&mut self, b: &[u8]) {
        for &x in b {
            self.0 = (self.0 ^ u64::from(x)).wrapping_mul(0x0100_0000_01b3);
        }
    }
    fn num(&mut self, n: u64) {
        self.bytes(&n.to_le_bytes());
    }
}

/// What the pack is a copy of: the shards (name, size) and every expert range in them.
pub(crate) fn fingerprint(source: &ExpertStore) -> Result<String> {
    let mut h = Fnv::new();
    for path in &source.files {
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned());
        h.bytes(name.unwrap_or_default().as_bytes());
        h.num(
            std::fs::metadata(path)
                .with_context(|| format!("pack: stat {}", path.display()))?
                .len(),
        );
    }
    for (layer, l) in &source.layers {
        h.num(u64::from(*layer));
        h.num(l.n_experts as u64);
        for (role, r) in &l.roles {
            h.bytes(role.as_bytes());
            h.num(r.file as u64);
            h.num(r.offset);
            h.num(r.bytes as u64);
        }
    }
    Ok(format!("{:016x}", h.0))
}

/// Bytes of one record: the first layer's roles back to back, padded.
fn record_bytes(source: &ExpertStore) -> usize {
    source.expert_bytes()
}

fn stride_for(record: usize) -> usize {
    record.next_multiple_of(ALIGN)
}

/// Bypass the page cache for the file's reads and writes: a 68 GB pass would evict everything.
fn no_cache(f: &File) {
    // SAFETY: a plain fcntl on an fd we own.
    unsafe {
        libc::fcntl(f.as_raw_fd(), libc::F_NOCACHE, 1);
    }
}

fn free_bytes(dir: &Path) -> Result<u64> {
    let c = std::ffi::CString::new(dir.as_os_str().as_encoded_bytes())
        .map_err(|_| anyhow!("pack: path with a NUL byte"))?;
    // SAFETY: `c` is NUL-terminated and `st` is a valid out-pointer.
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::statvfs(c.as_ptr(), &mut st) };
    if rc != 0 {
        bail!("pack: statvfs({}) failed", dir.display());
    }
    Ok(st.f_bavail as u64 * st.f_frsize)
}

/// The pack of `dir`, as an [`ExpertStore`] whose every expert is one contiguous record, if it
/// exists and was made from `source`'s weights. `None` (with a line on stderr when a file is
/// there but unusable) otherwise.
pub(crate) fn open_pack(dir: &Path, source: &ExpertStore) -> Result<Option<ExpertStore>> {
    let path = pack_path(dir);
    let Ok(file) = File::open(&path) else {
        return Ok(None);
    };
    match read_header(&file, &path, source) {
        Ok(store) => Ok(Some(store)),
        Err(e) => {
            eprintln!(
                "[offload] ignoring the expert pack {}: {e:#}",
                path.display()
            );
            Ok(None)
        }
    }
}

fn read_header(file: &File, path: &Path, source: &ExpertStore) -> Result<ExpertStore> {
    let mut prefix = [0u8; 16];
    file.read_exact_at(&mut prefix, 0)
        .context("too short to hold a header")?;
    if &prefix[..8] != MAGIC {
        bail!("not an expert pack (bad magic)");
    }
    let json_len = u64::from_le_bytes(prefix[8..16].try_into().expect("8 bytes")) as usize;
    if json_len == 0 || json_len > HEADER_RESERVE - 16 {
        bail!("implausible header length {json_len}");
    }
    let mut json = vec![0u8; json_len];
    file.read_exact_at(&mut json, 16)
        .context("truncated header")?;
    let v: Value = serde_json::from_slice(&json).context("header is not JSON")?;
    let want = fingerprint(source)?;
    let got = v["fingerprint"].as_str().unwrap_or_default();
    if got != want {
        bail!("it was made for other weights (fingerprint {got}, these are {want})");
    }
    let stride = v["stride"].as_u64().ok_or_else(|| anyhow!("no stride"))? as usize;
    let data_start = v["data_start"]
        .as_u64()
        .ok_or_else(|| anyhow!("no data_start"))?;
    let roles = v["roles"].as_array().ok_or_else(|| anyhow!("no roles"))?;
    let layers = v["layers"].as_array().ok_or_else(|| anyhow!("no layers"))?;
    let mut role_info = Vec::new();
    for r in roles {
        role_info.push((
            r["name"].as_str().unwrap_or_default().to_string(),
            r["at"].as_u64().unwrap_or(0) as usize,
            r["bytes"].as_u64().unwrap_or(0) as usize,
            r["dtype"].as_str().unwrap_or_default().to_string(),
            r["tail_shape"]
                .as_array()
                .map(|a| a.iter().filter_map(Value::as_i64).collect::<Vec<_>>())
                .unwrap_or_default(),
        ));
    }
    let mut out_layers = std::collections::BTreeMap::new();
    let mut records = 0u64;
    for l in layers {
        let layer = l["layer"].as_u64().ok_or_else(|| anyhow!("layer id"))? as u16;
        let n_experts = l["n_experts"]
            .as_u64()
            .ok_or_else(|| anyhow!("n_experts"))? as usize;
        let first = l["first"].as_u64().ok_or_else(|| anyhow!("first record"))?;
        records = records.max(first + n_experts as u64);
        let base = data_start + first * stride as u64;
        let mut roles = std::collections::BTreeMap::new();
        for (name, at, bytes, dtype, tail_shape) in &role_info {
            roles.insert(
                name.clone(),
                RoleRange {
                    file: 0,
                    offset: base + *at as u64,
                    bytes: *bytes,
                    stride,
                    dtype: dtype.clone(),
                    tail_shape: tail_shape.clone(),
                },
            );
        }
        out_layers.insert(layer, LayerRanges { n_experts, roles });
    }
    let expected_len = data_start + records * stride as u64;
    let actual_len = file.metadata()?.len();
    if actual_len != expected_len {
        bail!(
            "it is {actual_len} bytes, its header promises {expected_len} (an interrupted build?)"
        );
    }
    // The pack must describe exactly what the shards do, role by role.
    for (layer, l) in &source.layers {
        let Some(p) = out_layers.get(layer) else {
            bail!("layer {layer} is missing");
        };
        if p.n_experts != l.n_experts
            || p.roles.len() != l.roles.len()
            || l.roles.iter().any(|(k, r)| {
                p.roles.get(k).is_none_or(|q| {
                    (q.bytes, &q.dtype, &q.tail_shape) != (r.bytes, &r.dtype, &r.tail_shape)
                })
            })
        {
            bail!("layer {layer} differs from the shards");
        }
    }
    Ok(ExpertStore {
        files: vec![path.to_path_buf()],
        layers: out_layers,
    })
}

/// Build `dir`'s pack from `source` (the shards' own ranges). `progress(done, total)` is called
/// after every layer. Writes `experts.lpack.tmp` and renames it, so a stopped build leaves no
/// pack that looks complete.
pub(crate) fn build(
    dir: &Path,
    source: &ExpertStore,
    mut progress: impl FnMut(usize, usize),
) -> Result<PathBuf> {
    let first = source
        .layers
        .values()
        .next()
        .ok_or_else(|| anyhow!("pack: the checkpoint has no routed experts"))?;
    let record = record_bytes(source);
    let stride = stride_for(record);
    let n_records: u64 = source.layers.values().map(|l| l.n_experts as u64).sum();
    let total = HEADER_RESERVE as u64 + n_records * stride as u64;
    let free = free_bytes(dir)?;
    if free < total + SPARE_BYTES {
        bail!(
            "pack: building it needs {:.1} GB and {} has {:.1} GB free",
            (total + SPARE_BYTES) as f64 / 1e9,
            dir.display(),
            free as f64 / 1e9
        );
    }
    // Where each role sits inside a record.
    let mut at = 0usize;
    let mut role_layout = Vec::new();
    for (name, r) in &first.roles {
        role_layout.push((
            name.clone(),
            at,
            r.bytes,
            r.dtype.clone(),
            r.tail_shape.clone(),
        ));
        at += r.bytes;
    }
    let layout_json: Vec<Value> = role_layout
        .iter()
        .map(|(name, at, bytes, dtype, tail)| {
            json!({"name": name, "at": at, "bytes": bytes, "dtype": dtype, "tail_shape": tail})
        })
        .collect();
    let mut layer_json = Vec::new();
    let mut next_record = 0u64;
    for (layer, l) in &source.layers {
        layer_json.push(json!({"layer": layer, "n_experts": l.n_experts, "first": next_record}));
        next_record += l.n_experts as u64;
    }
    let header = json!({
        "version": 1,
        "fingerprint": fingerprint(source)?,
        "stride": stride,
        "data_start": HEADER_RESERVE,
        "roles": layout_json,
        "layers": layer_json,
    })
    .to_string();
    if header.len() > HEADER_RESERVE - 16 {
        bail!(
            "pack: the header ({} bytes) does not fit its reserve",
            header.len()
        );
    }

    let tmp = dir.join(format!("{PACK_FILE}.tmp"));
    let out = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&tmp)
        .with_context(|| format!("pack: create {}", tmp.display()))?;
    no_cache(&out);
    out.set_len(total)?;
    let sources: Vec<File> = source
        .files
        .iter()
        .map(|p| {
            let f = File::open(p).with_context(|| format!("pack: open {}", p.display()))?;
            no_cache(&f);
            Ok(f)
        })
        .collect::<Result<_>>()?;

    let total_layers = source.layers.len();
    let mut record_index = 0u64;
    for (done, (layer, l)) in source.layers.iter().enumerate() {
        // One tensor per role, read whole and in parallel (they are contiguous in the shard).
        let bufs: Vec<Vec<u8>> = std::thread::scope(|s| {
            let handles: Vec<_> = role_layout
                .iter()
                .map(|(name, ..)| {
                    let r = &l.roles[name];
                    let file = &sources[r.file];
                    s.spawn(move || -> Result<Vec<u8>> {
                        let mut buf = vec![0u8; r.bytes * l.n_experts];
                        file.read_exact_at(&mut buf, r.offset)
                            .with_context(|| format!("pack: read layer {layer} {}", r.dtype))?;
                        Ok(buf)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().map_err(|_| anyhow!("pack: a reader panicked"))?)
                .collect::<Result<Vec<_>>>()
        })?;
        // Interleave them into records.
        let mut records = vec![0u8; l.n_experts * stride];
        let workers = 8usize.min(l.n_experts).max(1);
        let per = l.n_experts.div_ceil(workers);
        std::thread::scope(|s| {
            for (w, chunk) in records.chunks_mut(per * stride).enumerate() {
                let (bufs, layout) = (&bufs, &role_layout);
                s.spawn(move || {
                    for (i, rec) in chunk.chunks_mut(stride).enumerate() {
                        let e = w * per + i;
                        for (ri, (_, at, bytes, ..)) in layout.iter().enumerate() {
                            rec[*at..*at + *bytes]
                                .copy_from_slice(&bufs[ri][e * bytes..(e + 1) * bytes]);
                        }
                    }
                });
            }
        });
        drop(bufs);
        out.write_all_at(
            &records,
            HEADER_RESERVE as u64 + record_index * stride as u64,
        )
        .with_context(|| format!("pack: write layer {layer}"))?;
        record_index += l.n_experts as u64;
        progress(done + 1, total_layers);
    }

    let mut head = Vec::with_capacity(16 + header.len());
    head.extend_from_slice(MAGIC);
    head.extend_from_slice(&(header.len() as u64).to_le_bytes());
    head.extend_from_slice(header.as_bytes());
    out.write_all_at(&head, 0)?;
    out.sync_all()?;
    drop(out);
    let dest = pack_path(dir);
    std::fs::rename(&tmp, &dest)
        .with_context(|| format!("pack: move {} into place", tmp.display()))?;
    Ok(dest)
}
