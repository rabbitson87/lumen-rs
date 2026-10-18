//! Where each routed expert's bytes live inside the original safetensors shards.
//!
//! The MLX checkpoints stack a layer's experts into one tensor per projection
//! and quantization part, expert index first (`[E, out, in/pack]` weights,
//! `[E, out, groups]` scales, ...). So expert `e` of such a tensor is the
//! contiguous byte range `[e * per_expert, (e + 1) * per_expert)` of that
//! tensor's data, and no repacking is needed to read one expert: a read per
//! tensor ("role") at a computed offset. That is the layout Slotstream reads in
//! place too, and it needs no extra disk.
//!
//! This module only *finds* those ranges. It reads the shard headers and
//! nothing else; the bytes are read by [`super::io`]. Because the ranges are
//! later read straight off disk, the parse is as strict as the loader's
//! completeness check (`qwen3_5_moe::validate_safetensors_complete`): the
//! header must be exactly one JSON object, and every range it promises must lie
//! inside the file. A silently truncated or mis-sized shard would otherwise
//! serve plausible wrong weights forever.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};

/// The byte range of one tensor role (say `gate_proj.weight`) of one layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RoleRange {
    /// Index into [`ExpertStore::files`].
    pub file: usize,
    /// Absolute file offset of expert 0's bytes.
    pub offset: u64,
    /// Bytes of one expert.
    pub bytes: usize,
    /// Distance between expert `e`'s bytes and expert `e + 1`'s: `bytes` in a stacked
    /// checkpoint tensor, a whole record in the expert pack.
    pub stride: usize,
    /// safetensors dtype string, e.g. `U32`, `BF16`, `U8`.
    pub dtype: String,
    /// Shape of one expert (the stacked shape without its leading axis).
    pub tail_shape: Vec<i64>,
}

impl RoleRange {
    /// Absolute file offset of expert `e`.
    pub(crate) fn offset_of(&self, e: usize) -> u64 {
        self.offset + (e * self.stride) as u64
    }
}

/// Every role of one layer's routed experts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LayerRanges {
    pub n_experts: usize,
    /// Keyed by role: `gate_proj.weight`, `gate_proj.scales`, `up_proj.biases`, ...
    pub roles: BTreeMap<String, RoleRange>,
}

/// The layer id the draft head's routed experts are registered under: past every trunk layer
/// (Flash-Next has 48), so layer order, staging and `layer_after` never reach it.
pub(crate) const MTP_LAYER: u16 = 1000;

/// The shards of one checkpoint and the expert ranges found in them.
#[derive(Debug)]
pub(crate) struct ExpertStore {
    pub files: Vec<PathBuf>,
    pub layers: BTreeMap<u16, LayerRanges>,
}

impl ExpertStore {
    /// Scan every `*.safetensors` file of `dir`.
    pub(crate) fn open(dir: &Path) -> Result<Self> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .with_context(|| format!("expert store: read_dir({})", dir.display()))?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            bail!("expert store: no .safetensors files in {}", dir.display());
        }
        Self::from_files(files)
    }

    /// [`Self::open`], then the directory's expert pack if `mode` allows it: when `mode` is
    /// `Build` and there is no usable pack, the pack is built first. With `with_mtp` the draft
    /// head's experts are added as layer [`MTP_LAYER`] (they are never part of the pack).
    pub(crate) fn open_with(
        dir: &Path,
        mode: super::pack::PackMode,
        with_mtp: bool,
    ) -> Result<Self> {
        let mut store = Self::open_pack_or_shards(dir, mode)?;
        if with_mtp {
            store.add_mtp_layer(dir)?;
        }
        Ok(store)
    }

    fn open_pack_or_shards(dir: &Path, mode: super::pack::PackMode) -> Result<Self> {
        use super::pack::{self, PackMode};
        let source = Self::open(dir)?;
        if mode == PackMode::Off {
            return Ok(source);
        }
        if let Some(packed) = pack::open_pack(dir, &source)? {
            return Ok(packed);
        }
        if mode == PackMode::Build {
            let started = std::time::Instant::now();
            eprintln!(
                "[offload] building the expert pack in {} ...",
                dir.display()
            );
            let path = pack::build(dir, &source, |done, total| {
                if done % 6 == 0 || done == total {
                    eprintln!("[offload] expert pack: layer {done} of {total}");
                }
            })?;
            eprintln!(
                "[offload] expert pack ready in {:.0} s: {}",
                started.elapsed().as_secs_f64(),
                path.display()
            );
            if let Some(packed) = pack::open_pack(dir, &source)? {
                return Ok(packed);
            }
        }
        Ok(source)
    }

    pub(crate) fn from_files(files: Vec<PathBuf>) -> Result<Self> {
        Self::from_files_with(files, expert_role)
    }

    /// Add the draft head's routed experts (`mtp.safetensors` in `dir`, when there is one) as
    /// layer [`MTP_LAYER`], so they are served from the slot pool like any layer's.
    pub(crate) fn add_mtp_layer(&mut self, dir: &Path) -> Result<()> {
        let path = dir.join("mtp.safetensors");
        if !path.is_file() {
            return Ok(());
        }
        let head = Self::from_files_with(vec![path.clone()], mtp_expert_role)?;
        let Some(ranges) = head.layers.into_values().next() else {
            return Ok(());
        };
        let file = self.files.len();
        self.files.push(path);
        let roles = ranges
            .roles
            .into_iter()
            .map(|(k, r)| (k, RoleRange { file, ..r }))
            .collect();
        self.layers.insert(
            MTP_LAYER,
            LayerRanges {
                n_experts: ranges.n_experts,
                roles,
            },
        );
        self.check_uniform()
    }

    fn from_files_with(
        files: Vec<PathBuf>,
        extract: fn(&str) -> Option<(u16, String)>,
    ) -> Result<Self> {
        let mut layers: BTreeMap<u16, LayerRanges> = BTreeMap::new();
        for (file_idx, path) in files.iter().enumerate() {
            for t in read_header(path)? {
                let Some((layer, role)) = extract(&t.name) else {
                    continue;
                };
                let (n_experts, tail) = match t.shape.split_first() {
                    Some((&e, tail)) if e > 0 => (e as usize, tail.to_vec()),
                    _ => bail!(
                        "{}: {} has no expert axis (shape {:?})",
                        path.display(),
                        t.name,
                        t.shape
                    ),
                };
                let total = t.end - t.begin;
                if total % n_experts as u64 != 0 {
                    bail!(
                        "{}: {} is {total} bytes, not a multiple of its {n_experts} experts",
                        path.display(),
                        t.name
                    );
                }
                let bytes = (total / n_experts as u64) as usize;
                if bytes == 0 {
                    bail!("{}: {} holds zero bytes per expert", path.display(), t.name);
                }
                let entry = layers.entry(layer).or_insert_with(|| LayerRanges {
                    n_experts,
                    roles: BTreeMap::new(),
                });
                if entry.n_experts != n_experts {
                    bail!(
                        "{}: layer {layer} role {role} has {n_experts} experts but another role has {}",
                        path.display(),
                        entry.n_experts
                    );
                }
                let range = RoleRange {
                    file: file_idx,
                    offset: t.data_start + t.begin,
                    bytes,
                    stride: bytes,
                    dtype: t.dtype,
                    tail_shape: tail,
                };
                if entry.roles.insert(role.clone(), range).is_some() {
                    bail!("layer {layer} role {role} appears in more than one place");
                }
            }
        }
        let store = Self { files, layers };
        store.check_uniform()?;
        Ok(store)
    }

    /// Every layer must offer the same roles with the same shapes: the pool
    /// holds one stacked tensor per role, shared by all layers.
    fn check_uniform(&self) -> Result<()> {
        let mut it = self.layers.iter();
        let Some((first_layer, first)) = it.next() else {
            return Ok(());
        };
        for (layer, l) in it {
            if l.n_experts != first.n_experts {
                bail!(
                    "layer {layer} has {} experts, layer {first_layer} has {}",
                    l.n_experts,
                    first.n_experts
                );
            }
            if l.roles.len() != first.roles.len() {
                bail!(
                    "layer {layer} has roles {:?}, layer {first_layer} has {:?}",
                    l.roles.keys().collect::<Vec<_>>(),
                    first.roles.keys().collect::<Vec<_>>()
                );
            }
            for (role, r) in &l.roles {
                let Some(f) = first.roles.get(role) else {
                    bail!("layer {layer} has role {role}, layer {first_layer} does not");
                };
                if (f.bytes, &f.dtype, &f.tail_shape) != (r.bytes, &r.dtype, &r.tail_shape) {
                    bail!(
                        "layer {layer} role {role} is {:?} {:?}, layer {first_layer} has {:?} {:?}",
                        r.dtype,
                        r.tail_shape,
                        f.dtype,
                        f.tail_shape
                    );
                }
            }
        }
        Ok(())
    }

    pub(crate) fn layer(&self, layer: u16) -> Option<&LayerRanges> {
        self.layers.get(&layer)
    }

    /// Total bytes one expert occupies across all roles.
    pub(crate) fn expert_bytes(&self) -> usize {
        self.layers
            .values()
            .next()
            .map_or(0, |l| l.roles.values().map(|r| r.bytes).sum())
    }
}

/// `(layer, role)` for a tensor name that is a routed expert's stacked
/// projection; `None` for everything else (attention, router, shared expert...).
///
/// Matches `...layers.<L>.mlp.switch_mlp.<proj>.<part>` (Qwen) and
/// `...layers.<L>.experts.switch_glu.<proj>.<part>` (Gemma), where `<proj>` is
/// `gate_proj`, `up_proj` or `down_proj` and `<part>` is `weight`, `scales` or
/// `biases`. The shared expert (`shared_expert.gate_proj...`) and the router
/// (`mlp.gate.weight`) do not match.
fn expert_role(name: &str) -> Option<(u16, String)> {
    // The MTP head has its own `layers.0` with its own routed experts; it must
    // not be mistaken for layer 0 of the trunk.
    if name.contains("mtp.") {
        return None;
    }
    let after = name.split("layers.").nth(1)?;
    let (digits, rest) = after.split_once('.')?;
    let layer: u16 = digits.parse().ok()?;
    let rest = rest
        .strip_prefix("mlp.switch_mlp.")
        .or_else(|| rest.strip_prefix("experts.switch_glu."))?;
    let (proj, part) = rest.split_once('.')?;
    let proj_ok = matches!(proj, "gate_proj" | "up_proj" | "down_proj");
    let part_ok = matches!(part, "weight" | "scales" | "biases");
    (proj_ok && part_ok).then(|| (layer, format!("{proj}.{part}")))
}

/// `(MTP_LAYER, role)` for the draft head's stacked expert tensors
/// (`mtp.layers.0.mlp.switch_mlp.<proj>.<part>`); `None` for everything else.
fn mtp_expert_role(name: &str) -> Option<(u16, String)> {
    let rest = name.strip_prefix("mtp.layers.0.mlp.switch_mlp.")?;
    let (proj, part) = rest.split_once('.')?;
    let proj_ok = matches!(proj, "gate_proj" | "up_proj" | "down_proj");
    let part_ok = matches!(part, "weight" | "scales" | "biases");
    (proj_ok && part_ok).then(|| (MTP_LAYER, format!("{proj}.{part}")))
}

pub(crate) struct TensorEntry {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<i64>,
    /// Relative to the start of the data section.
    pub begin: u64,
    pub end: u64,
    /// Absolute file offset where the data section starts.
    pub data_start: u64,
}

impl TensorEntry {
    /// Absolute file offset of the tensor's first byte.
    pub(crate) fn offset(&self) -> u64 {
        self.data_start + self.begin
    }
}

/// Parse a shard's header, rejecting anything the loader's own check would.
pub(crate) fn read_header(path: &Path) -> Result<Vec<TensorEntry>> {
    let mut f = File::open(path).with_context(|| format!("open({})", path.display()))?;
    let file_len = f
        .metadata()
        .with_context(|| format!("metadata({})", path.display()))?
        .len();
    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf)
        .with_context(|| format!("{}: too short to hold a safetensors header", path.display()))?;
    let header_len = u64::from_le_bytes(len_buf);
    // The reference parser caps headers at 100 MB so a tiny prefix cannot ask
    // for an attacker-sized allocation; a bad prefix must not shift the data
    // section either (the loader's guard documents that failure).
    if header_len == 0 || header_len > 100_000_000 || 8 + header_len > file_len {
        bail!(
            "{}: header length {header_len} does not fit the {file_len}-byte file",
            path.display()
        );
    }
    let mut header = vec![0u8; header_len as usize];
    f.read_exact(&mut header)
        .with_context(|| format!("{}: read header", path.display()))?;
    let value: serde_json::Value = serde_json::from_slice(&header).map_err(|e| {
        anyhow!(
            "{}: the {header_len}-byte header region is not a single JSON object ({e}); \
             the declared length is wrong or the file is corrupt. Re-download the shard.",
            path.display()
        )
    })?;
    let obj = value.as_object().ok_or_else(|| {
        anyhow!(
            "{}: safetensors header is not a JSON object",
            path.display()
        )
    })?;

    let data_start = 8 + header_len;
    let mut out = Vec::with_capacity(obj.len());
    for (name, spec) in obj {
        if name == "__metadata__" {
            continue;
        }
        let offsets = spec
            .get("data_offsets")
            .and_then(|o| o.as_array())
            .filter(|o| o.len() == 2)
            .and_then(|o| Some((o[0].as_u64()?, o[1].as_u64()?)))
            .ok_or_else(|| anyhow!("{}: {name} has no valid data_offsets", path.display()))?;
        let (begin, end) = offsets;
        if end < begin || data_start + end > file_len {
            bail!(
                "{} is truncated or inconsistent: {name} needs bytes up to {} but the file is \
                 {file_len} bytes. A partial download loads without error and silently yields \
                 wrong weights. Re-download the shard.",
                path.display(),
                data_start + end
            );
        }
        let dtype = spec
            .get("dtype")
            .and_then(|d| d.as_str())
            .ok_or_else(|| anyhow!("{}: {name} has no dtype", path.display()))?
            .to_string();
        let shape = spec
            .get("shape")
            .and_then(|s| s.as_array())
            .ok_or_else(|| anyhow!("{}: {name} has no shape", path.display()))?
            .iter()
            .map(|d| {
                d.as_i64()
                    .ok_or_else(|| anyhow!("{}: {name} has a non-integer dim", path.display()))
            })
            .collect::<Result<Vec<i64>>>()?;
        out.push(TensorEntry {
            name: name.clone(),
            dtype,
            shape,
            begin,
            end,
            data_start,
        });
    }
    Ok(out)
}

#[cfg(test)]
pub(crate) mod test_files {
    //! Hand-built safetensors files for tests (no dependency on the crate that
    //! normally writes them, so the byte layout under test is explicit).
    use std::io::Write;
    use std::path::Path;

    pub(crate) struct T<'a> {
        pub name: &'a str,
        pub dtype: &'a str,
        pub shape: Vec<i64>,
        pub data: Vec<u8>,
    }

    /// Write `tensors` in the given order; returns each tensor's absolute data offset.
    pub(crate) fn write(path: &Path, tensors: &[T]) -> Vec<u64> {
        let mut header = serde_json::Map::new();
        let mut cursor = 0u64;
        let mut rel = Vec::new();
        for t in tensors {
            let end = cursor + t.data.len() as u64;
            header.insert(
                t.name.to_string(),
                serde_json::json!({"dtype": t.dtype, "shape": t.shape, "data_offsets": [cursor, end]}),
            );
            rel.push(cursor);
            cursor = end;
        }
        let mut json = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        while !json.len().is_multiple_of(8) {
            json.push(b' ');
        }
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&(json.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&json).unwrap();
        for t in tensors {
            f.write_all(&t.data).unwrap();
        }
        let data_start = 8 + json.len() as u64;
        rel.into_iter().map(|r| data_start + r).collect()
    }

    /// `n` bytes where byte `i` of tensor `seed` is a function of both, so a
    /// read from the wrong place cannot match by accident.
    pub(crate) fn bytes(n: usize, seed: u8) -> Vec<u8> {
        (0..n)
            .map(|i| {
                (i as u8)
                    .wrapping_mul(31)
                    .wrapping_add(seed.wrapping_mul(97))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::test_files::{T, bytes, write};
    use super::*;

    const E: usize = 4;

    /// One layer's six roles, shuffled in the file so offsets are not in role order.
    fn layer_tensors(layer: u16, seed: u8) -> Vec<(String, &'static str, Vec<i64>, Vec<u8>)> {
        let n = |l: i64, c: i64| (E as i64 * l * c) as usize;
        vec![
            (
                format!("language_model.model.layers.{layer}.mlp.switch_mlp.up_proj.weight"),
                "U32",
                vec![E as i64, 2, 3],
                bytes(n(2, 3) * 4, seed),
            ),
            (
                format!("language_model.model.layers.{layer}.mlp.switch_mlp.down_proj.scales"),
                "U8",
                vec![E as i64, 3, 2],
                bytes(n(3, 2), seed + 1),
            ),
            (
                format!("language_model.model.layers.{layer}.mlp.switch_mlp.gate_proj.weight"),
                "U32",
                vec![E as i64, 2, 3],
                bytes(n(2, 3) * 4, seed + 2),
            ),
            (
                format!("language_model.model.layers.{layer}.mlp.switch_mlp.gate_proj.scales"),
                "U8",
                vec![E as i64, 2, 1],
                bytes(n(2, 1), seed + 3),
            ),
            (
                format!("language_model.model.layers.{layer}.mlp.switch_mlp.down_proj.weight"),
                "U32",
                vec![E as i64, 3, 2],
                bytes(n(3, 2) * 4, seed + 4),
            ),
            (
                format!("language_model.model.layers.{layer}.mlp.switch_mlp.up_proj.scales"),
                "U8",
                vec![E as i64, 2, 1],
                bytes(n(2, 1), seed + 5),
            ),
        ]
    }

    fn noise() -> Vec<(String, &'static str, Vec<i64>, Vec<u8>)> {
        vec![
            (
                "language_model.model.layers.0.self_attn.q_proj.weight".into(),
                "U32",
                vec![8, 8],
                bytes(256, 90),
            ),
            (
                "language_model.model.layers.0.mlp.gate.weight".into(),
                "U32",
                vec![E as i64, 4],
                bytes(64, 91),
            ),
            (
                "language_model.model.layers.0.mlp.shared_expert.gate_proj.weight".into(),
                "U32",
                vec![8, 8],
                bytes(256, 92),
            ),
            (
                "language_model.model.embed_tokens.weight".into(),
                "U32",
                vec![16, 8],
                bytes(512, 93),
            ),
        ]
    }

    fn write_all(path: &Path, parts: &[(String, &'static str, Vec<i64>, Vec<u8>)]) -> Vec<u64> {
        let ts: Vec<T> = parts
            .iter()
            .map(|(n, d, s, b)| T {
                name: n,
                dtype: d,
                shape: s.clone(),
                data: b.clone(),
            })
            .collect();
        write(path, &ts)
    }

    #[test]
    fn expert_ranges_point_at_exactly_the_experts_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let mut parts = noise();
        parts.extend(layer_tensors(0, 10));
        parts.extend(layer_tensors(1, 40));
        let path = dir.path().join("model-00001.safetensors");
        let abs = write_all(&path, &parts);

        let store = ExpertStore::open(dir.path()).expect("open");
        assert_eq!(store.layers.keys().copied().collect::<Vec<_>>(), vec![0, 1]);
        let l1 = store.layer(1).unwrap();
        assert_eq!(l1.n_experts, E);
        assert_eq!(
            l1.roles.keys().map(String::as_str).collect::<Vec<_>>(),
            [
                "down_proj.scales",
                "down_proj.weight",
                "gate_proj.scales",
                "gate_proj.weight",
                "up_proj.scales",
                "up_proj.weight"
            ]
        );
        // The shared expert, the router, attention and embeddings are not experts.
        assert_eq!(store.layer(0).unwrap().roles.len(), 6);

        // For every role of layer 1, every expert's range holds that expert's slice.
        let file = std::fs::read(&path).unwrap();
        for (i, (name, _, _, data)) in parts.iter().enumerate() {
            let Some((layer, role)) = expert_role(name) else {
                continue;
            };
            let r = &store.layer(layer).unwrap().roles[&role];
            assert_eq!(r.offset, abs[i], "{name}: offset of expert 0");
            assert_eq!(r.bytes, data.len() / E);
            for e in 0..E {
                let at = r.offset_of(e) as usize;
                assert_eq!(
                    &file[at..at + r.bytes],
                    &data[e * r.bytes..(e + 1) * r.bytes],
                    "{name} expert {e}"
                );
            }
        }
        assert_eq!(store.expert_bytes(), 24 + 6 + 24 + 2 + 24 + 2);
    }

    #[test]
    fn layers_split_across_shards_are_found_in_each() {
        let dir = tempfile::tempdir().unwrap();
        write_all(
            &dir.path().join("model-00001.safetensors"),
            &layer_tensors(0, 1),
        );
        write_all(
            &dir.path().join("model-00002.safetensors"),
            &layer_tensors(1, 2),
        );
        let store = ExpertStore::open(dir.path()).expect("open");
        assert_eq!(store.files.len(), 2);
        assert_eq!(store.layer(0).unwrap().roles["up_proj.weight"].file, 0);
        assert_eq!(store.layer(1).unwrap().roles["up_proj.weight"].file, 1);
    }

    #[test]
    fn a_truncated_shard_is_an_error_not_wrong_weights() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model-00001.safetensors");
        write_all(&path, &layer_tensors(0, 1));
        let len = std::fs::metadata(&path).unwrap().len();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(len - 5)
            .unwrap();
        let err = ExpertStore::open(dir.path()).unwrap_err();
        assert!(format!("{err:#}").contains("truncated"), "got: {err:#}");
    }

    #[test]
    fn a_wrong_header_length_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model-00001.safetensors");
        write_all(&path, &layer_tensors(0, 1));
        let mut raw = std::fs::read(&path).unwrap();
        let declared = u64::from_le_bytes(raw[..8].try_into().unwrap());
        raw[..8].copy_from_slice(&(declared + 15).to_le_bytes()); // the loader guard's documented case
        std::fs::write(&path, raw).unwrap();
        let err = ExpertStore::open(dir.path()).unwrap_err();
        assert!(
            format!("{err:#}").contains("single JSON object"),
            "got: {err:#}"
        );
    }

    #[test]
    fn a_layer_whose_roles_differ_from_the_others_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let mut parts = layer_tensors(0, 1);
        let mut other = layer_tensors(1, 2);
        other.pop(); // layer 1 loses a role
        parts.extend(other);
        write_all(&dir.path().join("model-00001.safetensors"), &parts);
        let err = ExpertStore::open(dir.path()).unwrap_err();
        assert!(format!("{err:#}").contains("layer 1"), "got: {err:#}");
    }

    #[test]
    fn an_expert_axis_that_does_not_divide_the_data_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let parts = vec![(
            "m.layers.0.mlp.switch_mlp.up_proj.weight".to_string(),
            "U32",
            vec![E as i64, 2],
            bytes(4 * 2 * 4 + 1, 3), // one stray byte
        )];
        write_all(&dir.path().join("model-00001.safetensors"), &parts);
        let err = ExpertStore::open(dir.path()).unwrap_err();
        assert!(format!("{err:#}").contains("multiple"), "got: {err:#}");
    }

    /// Reads the real shard headers of the checkpoint in `LUMEN_EXPERT_STORE_DIR`
    /// and prints what it found, so a new checkpoint layout shows up as a number
    /// here before it shows up as a wrong answer there.
    #[test]
    #[ignore = "needs a real checkpoint; set LUMEN_EXPERT_STORE_DIR"]
    fn real_checkpoint_layout() {
        let Ok(dir) = std::env::var("LUMEN_EXPERT_STORE_DIR") else {
            return;
        };
        let store = ExpertStore::open(Path::new(&dir)).expect("open real checkpoint");
        let first = store
            .layers
            .values()
            .next()
            .expect("at least one MoE layer");
        eprintln!(
            "{dir}\n  shards={} moe_layers={} experts/layer={} bytes/expert={} ({:.2} MB)",
            store.files.len(),
            store.layers.len(),
            first.n_experts,
            store.expert_bytes(),
            store.expert_bytes() as f64 / 1e6
        );
        for (role, r) in &first.roles {
            eprintln!(
                "  {role:<20} {:>4} {:?} {:?} {} B/expert",
                r.dtype, r.tail_shape, r.file, r.bytes
            );
        }
        let all: u64 =
            store.layers.len() as u64 * first.n_experts as u64 * store.expert_bytes() as u64;
        eprintln!("  all routed experts: {:.2} GB", all as f64 / 1e9);
        assert!(store.layers.len() > 1 && first.n_experts > 1);
    }

    #[test]
    fn only_routed_expert_projections_match() {
        for (name, want) in [
            (
                "a.layers.12.mlp.switch_mlp.gate_proj.weight",
                Some((12, "gate_proj.weight")),
            ),
            (
                "a.layers.3.mlp.switch_mlp.down_proj.biases",
                Some((3, "down_proj.biases")),
            ),
            (
                "a.layers.7.experts.switch_glu.up_proj.scales",
                Some((7, "up_proj.scales")),
            ),
            ("a.layers.7.mlp.shared_expert.gate_proj.weight", None),
            ("a.layers.7.mlp.gate.weight", None),
            ("a.layers.7.mlp.shared_expert_gate.weight", None),
            ("a.layers.7.mlp.switch_mlp.gate_proj.bias", None),
            ("mtp.layers.0.mlp.switch_mlp.gate_proj.weight", None), // the trunk matcher leaves the head alone
            (
                "language_model.mtp.layers.0.mlp.switch_mlp.up_proj.scales",
                None,
            ),
            ("a.layers.x.mlp.switch_mlp.gate_proj.weight", None),
            ("model.embed_tokens.weight", None),
        ] {
            assert_eq!(
                expert_role(name),
                want.map(|(l, r)| (l, r.to_string())),
                "{name}"
            );
        }
    }

    // ---- the expert pack -------------------------------------------------------------------

    use super::super::io::{IoMode, IoPool, ReadJob};
    use super::super::pack::{self, PackMode};

    /// A two-layer checkpoint in `dir`; returns its tensors' bytes by (layer, role) for comparison.
    fn two_layer_checkpoint(dir: &Path) -> std::collections::HashMap<(u16, String), Vec<u8>> {
        let mut parts = noise();
        parts.extend(layer_tensors(0, 10));
        parts.extend(layer_tensors(1, 40));
        let mut by_role = std::collections::HashMap::new();
        for (name, _, _, data) in &parts {
            if let Some((layer, role)) = expert_role(name) {
                by_role.insert((layer, role), data.clone());
            }
        }
        write_all(&dir.join("model-00001.safetensors"), &parts);
        by_role
    }

    #[test]
    fn a_pack_holds_each_expert_as_one_record_made_of_the_shards_own_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let tensors = two_layer_checkpoint(dir.path());
        let source = ExpertStore::open(dir.path()).unwrap();
        let path = pack::build(dir.path(), &source, |_, _| {}).unwrap();
        assert_eq!(path, pack::pack_path(dir.path()));
        assert!(!dir.path().join("experts.lpack.tmp").exists());

        let packed = pack::open_pack(dir.path(), &source)
            .unwrap()
            .expect("usable pack");
        assert_eq!(packed.files, vec![path.clone()]);
        assert_eq!(packed.expert_bytes(), source.expert_bytes());
        let file = std::fs::read(&path).unwrap();
        for (layer, l) in &packed.layers {
            for (role, r) in &l.roles {
                let want = &tensors[&(*layer, role.clone())];
                for e in 0..l.n_experts {
                    let at = r.offset_of(e) as usize;
                    assert_eq!(
                        file[at..at + r.bytes],
                        want[e * r.bytes..(e + 1) * r.bytes],
                        "layer {layer} {role} expert {e}"
                    );
                }
            }
            // The roles of one expert are back to back: one read gets all of them.
            let mut spans: Vec<(u64, usize)> = l
                .roles
                .values()
                .map(|r| (r.offset_of(1), r.bytes))
                .collect();
            spans.sort();
            for w in spans.windows(2) {
                assert_eq!(w[0].0 + w[0].1 as u64, w[1].0);
            }
        }
    }

    #[test]
    fn reading_a_packed_expert_is_one_read_that_lands_in_every_roles_slot() {
        let dir = tempfile::tempdir().unwrap();
        let tensors = two_layer_checkpoint(dir.path());
        let source = ExpertStore::open(dir.path()).unwrap();
        pack::build(dir.path(), &source, |_, _| {}).unwrap();
        let packed = pack::open_pack(dir.path(), &source).unwrap().unwrap();
        let l = packed.layer(1).unwrap();
        let mut dsts: Vec<(String, Vec<u8>)> = l
            .roles
            .iter()
            .map(|(k, r)| (k.clone(), vec![0u8; r.bytes]))
            .collect();
        let jobs: Vec<ReadJob> = dsts
            .iter_mut()
            .map(|(k, d)| {
                let r = &l.roles[k];
                ReadJob::single(r.file, r.offset_of(2), r.bytes, d.as_mut_ptr())
            })
            .collect();
        let pool = IoPool::new(&packed.files, 2, IoMode::Direct).unwrap();
        // SAFETY: the destinations are distinct vectors that outlive the call.
        unsafe { pool.run(jobs) }.unwrap();
        for (k, d) in &dsts {
            let r = &l.roles[k];
            assert_eq!(
                d[..],
                tensors[&(1, k.clone())][2 * r.bytes..3 * r.bytes],
                "{k}"
            );
        }
        let reads = pool
            .profile()
            .jobs
            .load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(reads, 1, "six ranges of one record should be one read");
    }

    #[test]
    fn a_pack_for_other_weights_or_cut_short_is_ignored_and_off_never_reads_it() {
        let dir = tempfile::tempdir().unwrap();
        two_layer_checkpoint(dir.path());
        let source = ExpertStore::open(dir.path()).unwrap();
        pack::build(dir.path(), &source, |_, _| {}).unwrap();

        // Auto uses it, Off does not.
        assert!(
            ExpertStore::open_with(dir.path(), PackMode::Auto, false)
                .unwrap()
                .files[0]
                .ends_with("experts.lpack")
        );
        assert!(
            ExpertStore::open_with(dir.path(), PackMode::Off, false)
                .unwrap()
                .files[0]
                .ends_with("model-00001.safetensors")
        );

        // A pack cut short (an interrupted copy) is not used.
        let path = pack::pack_path(dir.path());
        let full = std::fs::read(&path).unwrap();
        std::fs::write(&path, &full[..full.len() - 1000]).unwrap();
        assert!(pack::open_pack(dir.path(), &source).unwrap().is_none());
        std::fs::write(&path, &full).unwrap();
        assert!(pack::open_pack(dir.path(), &source).unwrap().is_some());

        // Weights that changed (a shard re-downloaded with other contents) do not match.
        let mut parts = noise();
        parts.extend(layer_tensors(0, 11));
        parts.extend(layer_tensors(1, 40));
        parts.push((
            "language_model.model.layers.1.self_attn.extra".into(),
            "U32",
            vec![2],
            bytes(8, 1),
        ));
        write_all(&dir.path().join("model-00001.safetensors"), &parts);
        let changed = ExpertStore::open(dir.path()).unwrap();
        assert!(pack::open_pack(dir.path(), &changed).unwrap().is_none());
        // Build refuses without room, and builds when the pack is missing.
        std::fs::remove_file(&path).unwrap();
        let built = ExpertStore::open_with(dir.path(), PackMode::Build, false).unwrap();
        assert!(built.files[0].ends_with("experts.lpack"));
    }
}
