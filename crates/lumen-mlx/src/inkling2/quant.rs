//! Quantization of the `inkling2_loop` checkpoint: the `quantization` block of
//! `config.json` as a table of specs, and a shape check of every quantized
//! module against the packed tensors the shards actually declare.
//!
//! A module is a tensor name minus `.weight`, `.scales`, `.biases` or
//! `.global_scale`. A module is quantized iff it has a `.scales` tensor. The
//! check reads names, dtypes and shapes only; values are never inspected.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use anyhow::{Result, anyhow, bail, ensure};
use serde_json::Value;

use super::safetensors_header::TensorInfo;

/// Packing mode of a quantized module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantMode {
    /// Per-group bf16 `.scales` and `.biases`.
    Affine,
    /// Per-group U8 `.scales`, one F32 `.global_scale` per expert, no biases.
    Nvfp4,
}

impl QuantMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Affine => "affine",
            Self::Nvfp4 => "nvfp4",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantSpec {
    pub mode: QuantMode,
    pub bits: u32,
    pub group_size: u32,
}

impl QuantSpec {
    /// Accepted: nvfp4 with 4 bits and group 16; affine with 4 or 8 bits and group 64.
    fn new(what: &str, mode: QuantMode, bits: u32, group_size: u32) -> Result<Self> {
        let accepted = match mode {
            QuantMode::Nvfp4 => bits == 4 && group_size == 16,
            QuantMode::Affine => matches!(bits, 4 | 8) && group_size == 64,
        };
        ensure!(
            accepted,
            "{what}: {} with {bits} bits and group_size {group_size} is not accepted \
             (nvfp4: 4 bits, group 16; affine: 4 or 8 bits, group 64)",
            mode.name()
        );
        Ok(Self {
            mode,
            bits,
            group_size,
        })
    }
}

/// Parses one spec from the three fields as they appear in the JSON object.
fn parse_spec(
    what: &str,
    bits: Option<&Value>,
    group_size: Option<&Value>,
    mode: Option<&Value>,
) -> Result<QuantSpec> {
    let number = |v: Option<&Value>, field: &str| -> Result<u32> {
        v.and_then(Value::as_u64)
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| anyhow!("{what}: `{field}` is missing or not a small integer"))
    };
    let mode = match mode.and_then(Value::as_str) {
        Some("affine") => QuantMode::Affine,
        Some("nvfp4") => QuantMode::Nvfp4,
        other => bail!("{what}: mode {other:?} is neither affine nor nvfp4"),
    };
    QuantSpec::new(
        what,
        mode,
        number(bits, "bits")?,
        number(group_size, "group_size")?,
    )
}

/// The default spec and the per-module overrides of one checkpoint.
#[derive(Debug, Clone)]
pub struct QuantTable {
    default: QuantSpec,
    overrides: BTreeMap<String, QuantSpec>,
}

impl QuantTable {
    /// Reads a `quantization` block. Every entry that is not one of the three
    /// default keys must be an object giving one module's spec.
    pub fn from_block(block: &BTreeMap<String, Value>) -> Result<Self> {
        let default = parse_spec(
            "quantization default",
            block.get("bits"),
            block.get("group_size"),
            block.get("mode"),
        )?;
        let mut overrides = BTreeMap::new();
        for (module, value) in block {
            if matches!(module.as_str(), "bits" | "group_size" | "mode") {
                continue;
            }
            let obj = value
                .as_object()
                .ok_or_else(|| anyhow!("quantization.{module} is not an object"))?;
            let what = format!("quantization override {module}");
            let spec = parse_spec(
                &what,
                obj.get("bits"),
                obj.get("group_size"),
                obj.get("mode"),
            )?;
            overrides.insert(module.clone(), spec);
        }
        Ok(Self { default, overrides })
    }

    pub fn default_spec(&self) -> QuantSpec {
        self.default
    }

    /// The spec of `module` (its name without the tensor suffix): its override
    /// if it has one, otherwise the default.
    pub fn resolve(&self, module: &str) -> QuantSpec {
        self.overrides.get(module).copied().unwrap_or(self.default)
    }

    pub fn is_override(&self, module: &str) -> bool {
        self.overrides.contains_key(module)
    }

    fn overridden_modules(&self) -> impl Iterator<Item = &str> {
        self.overrides.keys().map(String::as_str)
    }
}

/// How many quantized modules of each mode passed `check_packed_shapes`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PackedSummary {
    pub affine: usize,
    pub nvfp4: usize,
}

/// Checks every quantized module against its packed tensors, and every packed
/// tensor against a quantized module. `num_experts` is the length the
/// `global_scale` of an nvfp4 module must have. The first violation is returned,
/// and it names the module.
pub fn check_packed_shapes(
    tensors: &[TensorInfo],
    table: &QuantTable,
    num_experts: usize,
) -> Result<PackedSummary> {
    let by_name: HashMap<&str, &TensorInfo> =
        tensors.iter().map(|t| (t.name.as_str(), t)).collect();
    let quantized: BTreeSet<&str> = tensors
        .iter()
        .filter_map(|t| t.name.strip_suffix(".scales"))
        .collect();

    let mut summary = PackedSummary::default();
    for &module in &quantized {
        let spec = table.resolve(module);
        check_module(module, spec, &by_name, num_experts)?;
        match spec.mode {
            QuantMode::Affine => summary.affine += 1,
            QuantMode::Nvfp4 => summary.nvfp4 += 1,
        }
    }

    for t in tensors {
        let packed_part =
            t.dtype == "U32" || t.name.ends_with(".biases") || t.name.ends_with(".global_scale");
        if packed_part {
            let owner = [".weight", ".biases", ".global_scale", ".scales"]
                .iter()
                .find_map(|suffix| t.name.strip_suffix(suffix));
            ensure!(
                owner.is_some_and(|m| quantized.contains(m)),
                "{}: packed tensor of no quantized module",
                t.name
            );
        }
    }

    for module in table.overridden_modules() {
        ensure!(
            quantized.contains(module),
            "quantization override {module} names no quantized module (no {module}.scales)"
        );
    }
    Ok(summary)
}

fn check_module(
    module: &str,
    spec: QuantSpec,
    by_name: &HashMap<&str, &TensorInfo>,
    num_experts: usize,
) -> Result<()> {
    let find = |suffix: &str| by_name.get(format!("{module}.{suffix}").as_str()).copied();
    let need =
        |suffix: &str| find(suffix).ok_or_else(|| anyhow!("module {module}: no {module}.{suffix}"));

    let weight = need("weight")?;
    let scales = need("scales")?;
    ensure!(
        weight.dtype == "U32",
        "module {module}: weight is {}, expected U32",
        weight.dtype
    );
    let packed = last_dim(module, weight)?;
    let bits = spec.bits as usize;
    let unpacked = packed * 32;
    ensure!(
        unpacked % bits == 0,
        "module {module}: {packed} packed 32-bit words do not split into {bits}-bit values"
    );
    let in_features = unpacked / bits;
    let groups = last_dim(module, scales)?;
    let covered = groups * spec.group_size as usize;
    ensure!(
        covered == in_features,
        "module {module}: {groups} scales x group {} cover {covered} input features, \
         but the weight unpacks to {in_features}",
        spec.group_size
    );
    let lead = weight.shape.len() - 1;
    ensure!(
        scales.shape.len() == weight.shape.len() && scales.shape[..lead] == weight.shape[..lead],
        "module {module}: scales {:?} and weight {:?} differ in their leading dims",
        scales.shape,
        weight.shape
    );

    match spec.mode {
        QuantMode::Affine => {
            ensure!(
                scales.dtype == "BF16",
                "module {module}: affine scales are {}, expected BF16",
                scales.dtype
            );
            let biases = need("biases")?;
            ensure!(
                biases.dtype == "BF16" && biases.shape == scales.shape,
                "module {module}: affine biases must be BF16 of the scales' shape {:?}, found {} {:?}",
                scales.shape,
                biases.dtype,
                biases.shape
            );
            ensure!(
                find("global_scale").is_none(),
                "module {module}: affine module has a global_scale"
            );
        }
        QuantMode::Nvfp4 => {
            ensure!(
                scales.dtype == "U8",
                "module {module}: nvfp4 scales are {}, expected U8",
                scales.dtype
            );
            ensure!(
                find("biases").is_none(),
                "module {module}: nvfp4 module has biases"
            );
            let global = need("global_scale")?;
            ensure!(
                global.dtype == "F32" && global.shape == [num_experts],
                "module {module}: nvfp4 global_scale must be F32 of shape [{num_experts}], found {} {:?}",
                global.dtype,
                global.shape
            );
        }
    }
    Ok(())
}

fn last_dim(module: &str, t: &TensorInfo) -> Result<usize> {
    t.shape
        .last()
        .copied()
        .ok_or_else(|| anyhow!("module {module}: {} is a scalar", t.name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn block(value: Value) -> BTreeMap<String, Value> {
        serde_json::from_value(value).expect("object")
    }

    fn tensor(name: &str, dtype: &str, shape: &[usize]) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            dtype: dtype.to_string(),
            shape: shape.to_vec(),
            file: "t.safetensors".to_string(),
        }
    }

    #[test]
    fn only_the_accepted_mode_and_bit_group_pairs_are_taken() {
        let ok = |mode, bits, group| QuantSpec::new("t", mode, bits, group).is_ok();
        assert!(ok(QuantMode::Affine, 4, 64) && ok(QuantMode::Affine, 8, 64));
        assert!(ok(QuantMode::Nvfp4, 4, 16));
        assert!(!ok(QuantMode::Affine, 2, 64) && !ok(QuantMode::Affine, 4, 32));
        assert!(!ok(QuantMode::Nvfp4, 8, 16) && !ok(QuantMode::Nvfp4, 4, 64));
    }

    #[test]
    fn resolve_returns_the_override_or_the_default() {
        let table = QuantTable::from_block(&block(json!({
            "bits": 4, "group_size": 64, "mode": "affine",
            "lm_head": {"bits": 8, "group_size": 64, "mode": "affine"}
        })))
        .expect("block parses");
        assert_eq!(table.resolve("lm_head").bits, 8);
        assert_eq!(table.resolve("model.embed").bits, 4);
        assert!(table.is_override("lm_head") && !table.is_override("model.embed"));
    }

    #[test]
    fn a_scalar_entry_or_an_unknown_mode_is_rejected() {
        let scalar = QuantTable::from_block(&block(json!({
            "bits": 4, "group_size": 64, "mode": "affine", "quant_method": "x"
        })));
        assert!(scalar.is_err());
        let mode = QuantTable::from_block(&block(json!({
            "bits": 4, "group_size": 64, "mode": "mxfp4"
        })));
        assert!(mode.unwrap_err().to_string().contains("mxfp4"));
    }

    /// One affine 4-bit module `m` of 64 input features, 2 output rows.
    fn affine_module(scales_groups: usize) -> Vec<TensorInfo> {
        vec![
            tensor("m.weight", "U32", &[2, 8]),
            tensor("m.scales", "BF16", &[2, scales_groups]),
            tensor("m.biases", "BF16", &[2, scales_groups]),
        ]
    }

    fn affine4() -> QuantTable {
        QuantTable::from_block(&block(
            json!({"bits": 4, "group_size": 64, "mode": "affine"}),
        ))
        .expect("block parses")
    }

    #[test]
    fn a_consistent_affine_module_passes_and_a_bad_scale_count_names_it() {
        let ok = check_packed_shapes(&affine_module(1), &affine4(), 1).expect("consistent");
        assert_eq!(
            ok,
            PackedSummary {
                affine: 1,
                nvfp4: 0
            }
        );
        let err = check_packed_shapes(&affine_module(2), &affine4(), 1).unwrap_err();
        assert!(err.to_string().contains("module m"), "{err}");
    }

    #[test]
    fn a_packed_tensor_with_no_scales_is_an_orphan() {
        let tensors = vec![tensor("x.weight", "U32", &[2, 8])];
        let err = check_packed_shapes(&tensors, &affine4(), 1).unwrap_err();
        assert!(err.to_string().contains("x.weight"), "{err}");
    }
}
