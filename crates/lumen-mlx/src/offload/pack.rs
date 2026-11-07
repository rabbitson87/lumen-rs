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
//!
//! A low-bit copy (`experts.2b64.lsq.lpack`, see [`super::lowbit`]) is a file of the same shape
//! whose records were fitted again at fewer bits. Its header names its quantization and the
//! weights it was made from as `source_fingerprint`. It is a different model, not a cache of
//! this one, so nothing opens it unless asked to. When the checkpoint has a draft head, the copy
//! holds the head's experts too, as one more layer, and its header says `draft_head`. The head's
//! file is one of the shards `source_fingerprint` covers, so another head is other weights.

use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};

use super::expert_store::{ExpertStore, LayerRanges, MTP_LAYER, RoleRange};
use super::lowbit::{self, Fidelity, LowbitSpec, PackedProj, PackedProjMut, ScaleType};

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

/// A pack file's header, and the store its records describe, checked only against itself.
fn read_pack(file: &File, path: &Path) -> Result<(Value, ExpertStore)> {
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
    let store = ExpertStore {
        files: vec![path.to_path_buf()],
        layers: out_layers,
    };
    Ok((v, store))
}

fn read_header(file: &File, path: &Path, source: &ExpertStore) -> Result<ExpertStore> {
    let (v, store) = read_pack(file, path)?;
    let want = fingerprint(source)?;
    let got = v["fingerprint"].as_str().unwrap_or_default();
    if got != want {
        bail!("it was made for other weights (fingerprint {got}, these are {want})");
    }
    // The pack must describe exactly what the shards do, role by role.
    for (layer, l) in &source.layers {
        let Some(p) = store.layers.get(layer) else {
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
    Ok(store)
}

/// Where one role's bytes sit inside a record.
struct RoleSlot {
    name: String,
    at: usize,
    bytes: usize,
    dtype: String,
    tail_shape: Vec<i64>,
}

impl RoleSlot {
    /// Roles back to back, in the order given.
    fn lay_out(roles: impl IntoIterator<Item = (String, usize, String, Vec<i64>)>) -> Vec<Self> {
        let mut at = 0usize;
        roles
            .into_iter()
            .map(|(name, bytes, dtype, tail_shape)| {
                let slot = Self {
                    name,
                    at,
                    bytes,
                    dtype,
                    tail_shape,
                };
                at += bytes;
                slot
            })
            .collect()
    }

    fn to_json(&self) -> Value {
        json!({
            "name": self.name,
            "at": self.at,
            "bytes": self.bytes,
            "dtype": self.dtype,
            "tail_shape": self.tail_shape,
        })
    }
}

/// Write a pack of `source`'s experts to `dest`. `slots` lay a record out, `head` is the header's
/// own fields, and `fill(tensors, expert, record)` writes an expert's record from its layer's
/// whole tensors (in `LayerRanges::roles` order). Goes through `<dest>.tmp` and a rename, so a
/// stopped build leaves no pack that looks complete.
fn write_pack(
    dest: &Path,
    source: &ExpertStore,
    slots: &[RoleSlot],
    mut head: serde_json::Map<String, Value>,
    fill: impl Fn(&[Vec<u8>], usize, &mut [u8]) -> Result<()> + Sync,
    mut progress: impl FnMut(usize, usize),
) -> Result<()> {
    let dir = dest
        .parent()
        .ok_or_else(|| anyhow!("pack: {} has no directory", dest.display()))?;
    let record = slots.iter().map(|r| r.bytes).sum::<usize>();
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
    let mut layer_json = Vec::new();
    let mut next_record = 0u64;
    for (layer, l) in &source.layers {
        layer_json.push(json!({"layer": layer, "n_experts": l.n_experts, "first": next_record}));
        next_record += l.n_experts as u64;
    }
    head.insert("version".into(), json!(1));
    head.insert("stride".into(), json!(stride));
    head.insert("data_start".into(), json!(HEADER_RESERVE));
    head.insert(
        "roles".into(),
        slots.iter().map(RoleSlot::to_json).collect(),
    );
    head.insert("layers".into(), Value::Array(layer_json));
    let header = Value::Object(head).to_string();
    if header.len() > HEADER_RESERVE - 16 {
        bail!(
            "pack: the header ({} bytes) does not fit its reserve",
            header.len()
        );
    }

    let name = dest
        .file_name()
        .ok_or_else(|| anyhow!("pack: {} has no file name", dest.display()))?
        .to_string_lossy();
    let tmp = dir.join(format!("{name}.tmp"));
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
            let handles: Vec<_> = l
                .roles
                .values()
                .map(|r| {
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
        // Build the records.
        let mut records = vec![0u8; l.n_experts * stride];
        let workers = 8usize.min(l.n_experts).max(1);
        let per = l.n_experts.div_ceil(workers);
        std::thread::scope(|s| {
            let handles: Vec<_> = records
                .chunks_mut(per * stride)
                .enumerate()
                .map(|(w, chunk)| {
                    let (bufs, fill) = (&bufs, &fill);
                    s.spawn(move || -> Result<()> {
                        for (i, rec) in chunk.chunks_mut(stride).enumerate() {
                            fill(bufs, w * per + i, rec)?;
                        }
                        Ok(())
                    })
                })
                .collect();
            handles
                .into_iter()
                .try_for_each(|h| h.join().map_err(|_| anyhow!("pack: a worker panicked"))?)
        })
        .with_context(|| format!("pack: layer {layer}"))?;
        drop(bufs);
        out.write_all_at(
            &records,
            HEADER_RESERVE as u64 + record_index * stride as u64,
        )
        .with_context(|| format!("pack: write layer {layer}"))?;
        record_index += l.n_experts as u64;
        progress(done + 1, total_layers);
    }

    let mut bytes = Vec::with_capacity(16 + header.len());
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    out.write_all_at(&bytes, 0)?;
    out.sync_all()?;
    drop(out);
    std::fs::rename(&tmp, dest).with_context(|| format!("pack: move {} into place", tmp.display()))
}

/// Build `dir`'s pack from `source` (the shards' own ranges). `progress(done, total)` is called
/// after every layer.
pub(crate) fn build(
    dir: &Path,
    source: &ExpertStore,
    progress: impl FnMut(usize, usize),
) -> Result<PathBuf> {
    let first = source
        .layers
        .values()
        .next()
        .ok_or_else(|| anyhow!("pack: the checkpoint has no routed experts"))?;
    debug_assert_eq!(
        first.roles.values().map(|r| r.bytes).sum::<usize>(),
        record_bytes(source)
    );
    let slots = RoleSlot::lay_out(
        first
            .roles
            .iter()
            .map(|(k, r)| (k.clone(), r.bytes, r.dtype.clone(), r.tail_shape.clone())),
    );
    let mut head = serde_json::Map::new();
    head.insert("fingerprint".into(), json!(fingerprint(source)?));
    let dest = pack_path(dir);
    write_pack(
        &dest,
        source,
        &slots,
        head,
        |tensors, e, rec| {
            // The roles' bytes unchanged: `tensors` and `slots` are both in role order.
            for (buf, slot) in tensors.iter().zip(&slots) {
                rec[slot.at..slot.at + slot.bytes]
                    .copy_from_slice(&buf[e * slot.bytes..(e + 1) * slot.bytes]);
            }
            Ok(())
        },
        progress,
    )?;
    Ok(dest)
}

/// File name of `spec`'s low-bit copy inside a checkpoint directory.
pub(crate) fn lowbit_path(dir: &Path, spec: LowbitSpec) -> PathBuf {
    dir.join(format!(
        "experts.{}b{}.{}.lpack",
        spec.bits,
        spec.group_size,
        spec.recipe.name()
    ))
}

/// One projection of a layer: its three roles, in the source and in the copy.
struct ProjPlan {
    name: String,
    rows: usize,
    cols: usize,
    src_group: usize,
    scale_type: ScaleType,
    /// Index into the role-ordered tensors, and bytes of one expert, of weight, scales, biases.
    src: [(usize, usize); 3],
    /// Index into the role-ordered record slots.
    dst: [usize; 3],
}

const PARTS: [&str; 3] = ["weight", "scales", "biases"];

/// What a low-bit copy of `first`'s roles looks like: the plan per projection, and the record.
fn plan_lowbit(
    first: &LayerRanges,
    src_bits: u32,
    spec: LowbitSpec,
) -> Result<(Vec<ProjPlan>, Vec<RoleSlot>)> {
    let names: Vec<&String> = first.roles.keys().collect();
    let index = |role: &str| {
        names
            .iter()
            .position(|n| n.as_str() == role)
            .ok_or_else(|| anyhow!("pack: the checkpoint has no role {role}"))
    };
    let mut plans = Vec::new();
    let mut tails: Vec<Option<(usize, Vec<i64>)>> = vec![None; names.len()];
    for name in first.roles.keys().filter_map(|k| k.strip_suffix(".weight")) {
        let [w, s, b] = PARTS.map(|p| format!("{name}.{p}"));
        let (wi, si, bi) = (index(&w)?, index(&s)?, index(&b)?);
        let (wr, sr, br) = (&first.roles[&w], &first.roles[&s], &first.roles[&b]);
        if wr.dtype != "U32" || sr.dtype != br.dtype {
            bail!(
                "pack: {name} is {}, {}, {}; a low-bit copy needs U32 weights and one float type",
                wr.dtype,
                sr.dtype,
                br.dtype
            );
        }
        let scale_type = ScaleType::parse(&sr.dtype)?;
        let [rows, words] = wr.tail_shape[..] else {
            bail!("pack: {name}.weight is not a matrix");
        };
        let [_, groups] = sr.tail_shape[..] else {
            bail!("pack: {name}.scales is not a matrix");
        };
        let (rows, words, groups) = (rows as usize, words as usize, groups as usize);
        let cols = words * 32 / src_bits as usize;
        let dst_groups = cols / spec.group_size;
        let dst_words = lowbit::row_bytes(cols, spec.bits)? / 4;
        tails[wi] = Some((rows * dst_words * 4, vec![rows as i64, dst_words as i64]));
        for i in [si, bi] {
            let bytes = rows * dst_groups * scale_type.bytes();
            tails[i] = Some((bytes, vec![rows as i64, dst_groups as i64]));
        }
        plans.push(ProjPlan {
            name: name.to_string(),
            rows,
            cols,
            src_group: cols / groups,
            scale_type,
            src: [(wi, wr.bytes), (si, sr.bytes), (bi, br.bytes)],
            dst: [wi, si, bi],
        });
    }
    let roles = first
        .roles
        .iter()
        .zip(tails)
        .map(|((name, r), tail)| {
            let (bytes, shape) =
                tail.ok_or_else(|| anyhow!("pack: role {name} belongs to no projection"))?;
            Ok((name.clone(), bytes, r.dtype.clone(), shape))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((plans, RoleSlot::lay_out(roles)))
}

/// Build `spec`'s low-bit copy of `source`, whose experts are `src_bits` bits, and of the draft
/// head's next to it when there is one. Returns where it went and how close each projection
/// stayed.
pub(crate) fn build_lowbit(
    dir: &Path,
    source: &ExpertStore,
    src_bits: u32,
    spec: LowbitSpec,
    progress: impl FnMut(usize, usize),
) -> Result<(PathBuf, Vec<(String, Fidelity)>)> {
    let first = source
        .layers
        .values()
        .next()
        .ok_or_else(|| anyhow!("pack: the checkpoint has no routed experts"))?;
    let (plans, slots) = plan_lowbit(first, src_bits, spec)?;
    let mut head = serde_json::Map::new();
    head.insert("source_fingerprint".into(), json!(fingerprint(source)?));
    let mut copied = source.clone();
    copied.add_mtp_layer(dir)?;
    if copied.layers.contains_key(&MTP_LAYER) {
        head.insert("draft_head".into(), json!(true));
    }
    head.insert(
        "lowbit".into(),
        json!({"bits": spec.bits, "group_size": spec.group_size, "recipe": spec.recipe.name()}),
    );
    let fidelity: std::sync::Mutex<Vec<Fidelity>> =
        std::sync::Mutex::new(vec![Fidelity::default(); plans.len()]);
    let dest = lowbit_path(dir, spec);
    write_pack(
        &dest,
        &copied,
        &slots,
        head,
        |tensors, e, rec| {
            for (p, plan) in plans.iter().enumerate() {
                let [w, s, b] = plan
                    .src
                    .map(|(i, bytes)| &tensors[i][e * bytes..(e + 1) * bytes]);
                let [dw, ds, db] = plan.dst.map(|i| &slots[i]);
                let mut out = [dw, ds, db].map(|d| vec![0u8; d.bytes]);
                let [ow, os, ob] = &mut out;
                let f = lowbit::requantize(
                    PackedProj {
                        weight: w,
                        scales: s,
                        biases: b,
                        scale_type: plan.scale_type,
                        rows: plan.rows,
                        cols: plan.cols,
                        bits: src_bits,
                        group_size: plan.src_group,
                    },
                    spec,
                    PackedProjMut {
                        weight: ow,
                        scales: os,
                        biases: ob,
                    },
                )?;
                for (slot, bytes) in [dw, ds, db].into_iter().zip(&out) {
                    rec[slot.at..slot.at + slot.bytes].copy_from_slice(bytes);
                }
                fidelity.lock().unwrap_or_else(|e| e.into_inner())[p].add(f);
            }
            Ok(())
        },
        progress,
    )?;
    let totals = fidelity.into_inner().unwrap_or_else(|e| e.into_inner());
    let named = plans.iter().map(|p| p.name.clone()).zip(totals).collect();
    Ok((dest, named))
}

/// Does `spec`'s copy in `dir` hold the draft head's experts? `false` when there is no copy.
pub(crate) fn lowbit_holds_head(dir: &Path, spec: LowbitSpec) -> Result<bool> {
    let path = lowbit_path(dir, spec);
    let Ok(file) = File::open(&path) else {
        return Ok(false);
    };
    let (v, _) = read_pack(&file, &path).with_context(|| format!("pack {}", path.display()))?;
    Ok(v["draft_head"] == json!(true))
}

/// `spec`'s low-bit copy of `source`, as a store, if the file exists; an error if it does but
/// was not made from these weights at this quantization. The store has layer [`MTP_LAYER`] only
/// `with_mtp`, and then the copy has to hold it.
pub(crate) fn open_lowbit(
    dir: &Path,
    source: &ExpertStore,
    spec: LowbitSpec,
    with_mtp: bool,
) -> Result<Option<ExpertStore>> {
    let path = lowbit_path(dir, spec);
    let Ok(file) = File::open(&path) else {
        return Ok(None);
    };
    let (v, mut store) =
        read_pack(&file, &path).with_context(|| format!("pack {}", path.display()))?;
    let want = fingerprint(source)?;
    let got = v["source_fingerprint"].as_str().unwrap_or_default();
    if got != want {
        bail!(
            "{} was made for other weights (fingerprint {got}, these are {want})",
            path.display()
        );
    }
    let tier = &v["lowbit"];
    if tier["bits"].as_u64() != Some(u64::from(spec.bits))
        || tier["group_size"].as_u64() != Some(spec.group_size as u64)
        || tier["recipe"].as_str() != Some(spec.recipe.name())
    {
        bail!("{} holds {tier}, not {spec:?}", path.display());
    }
    for (layer, l) in &source.layers {
        let Some(p) = store.layers.get(layer) else {
            bail!("{}: layer {layer} is missing", path.display());
        };
        if p.n_experts != l.n_experts || !p.roles.keys().eq(l.roles.keys()) {
            bail!("{}: layer {layer} differs from the shards", path.display());
        }
    }
    if !with_mtp {
        store.layers.remove(&MTP_LAYER);
    } else if !store.layers.contains_key(&MTP_LAYER) {
        bail!(
            "{} holds no draft head: build the copy again",
            path.display()
        );
    }
    Ok(Some(store))
}
