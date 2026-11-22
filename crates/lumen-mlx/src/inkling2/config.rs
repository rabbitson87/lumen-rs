//! `config.json` of the `inkling2_loop` checkpoint, reduced to the fields the
//! manifest and the quantization table read.
//!
//! Everything else in the file (widths, attention shapes, the vision and MTP
//! architectures) is ignored, not validated. What the loop fields mean is
//! unverified: needs the reference. Only the layer ranges are used here.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::Value;

const MODEL_TYPE: &str = "inkling2_loop";
const ARCHITECTURE: &str = "Inkling2LoopForConditionalGeneration";

#[derive(Debug, Clone, Deserialize)]
pub struct Inkling2Config {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub eos_token_id: u32,
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
    pub mtp_config: MtpConfig,
    /// The `quantization` block, raw: `bits`, `group_size` and `mode` at the top
    /// are the default; every other entry is one module's override.
    pub quantization: BTreeMap<String, Value>,
    /// Must equal `quantization`; checked on load.
    pub quantization_config: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub num_hidden_layers: usize,
    pub local_layer_ids: Vec<usize>,
    pub n_routed_experts: usize,
    pub loop_config: LoopConfig,
}

/// The loop's layer range `[core_start, core_end)`.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct LoopConfig {
    pub core_start: usize,
    pub core_end: usize,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct VisionConfig {
    pub num_hidden_layers: usize,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct MtpConfig {
    pub num_hidden_layers: usize,
}

/// Attention kind of a text layer. The config lists the local layers; every
/// other index below `num_hidden_layers` is global.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    Local,
    Global,
}

impl Inkling2Config {
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let text =
            std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        Self::from_json_str(&text).with_context(|| format!("config {}", path.display()))
    }

    pub fn from_json_str(text: &str) -> Result<Self> {
        let cfg: Self = serde_json::from_str(text).context("parse config.json")?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.model_type == MODEL_TYPE,
            "model_type is {:?}, expected {MODEL_TYPE:?}",
            self.model_type
        );
        ensure!(
            self.architectures == [ARCHITECTURE],
            "architectures is {:?}, expected [{ARCHITECTURE:?}]",
            self.architectures
        );
        let n = self.text_config.num_hidden_layers;
        let mut seen = BTreeSet::new();
        for &id in &self.text_config.local_layer_ids {
            ensure!(
                id < n,
                "local_layer_ids: {id} is not below num_hidden_layers {n}"
            );
            ensure!(seen.insert(id), "local_layer_ids: {id} is listed twice");
        }
        let LoopConfig {
            core_start,
            core_end,
        } = self.text_config.loop_config;
        ensure!(
            core_start < core_end && core_end <= n,
            "loop_config: core range {core_start}..{core_end} is not a non-empty range inside 0..{n}"
        );
        ensure!(
            self.quantization == self.quantization_config,
            "quantization and quantization_config differ"
        );
        Ok(())
    }

    /// Kind of text layer `layer`, which must be below `num_hidden_layers`.
    pub fn layer_kind(&self, layer: usize) -> LayerKind {
        if self.text_config.local_layer_ids.contains(&layer) {
            LayerKind::Local
        } else {
            LayerKind::Global
        }
    }

    /// The global layers: the complement of `local_layer_ids` in `0..num_hidden_layers`.
    pub fn global_layer_ids(&self) -> Vec<usize> {
        (0..self.text_config.num_hidden_layers)
            .filter(|&l| self.layer_kind(l) == LayerKind::Global)
            .collect()
    }

    /// Whether text layer `layer` lies in the loop's core range.
    pub fn is_core(&self, layer: usize) -> bool {
        let LoopConfig {
            core_start,
            core_end,
        } = self.text_config.loop_config;
        (core_start..core_end).contains(&layer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("../../tests/fixtures/inkling2_loop/config.json");

    /// The fixture with `edit` applied, validated as a whole.
    fn mutated(edit: impl FnOnce(&mut Value)) -> Result<Inkling2Config> {
        let mut value: Value = serde_json::from_str(FIXTURE).expect("fixture is JSON");
        edit(&mut value);
        Inkling2Config::from_json_str(&value.to_string())
    }

    #[test]
    fn the_fixture_validates_and_its_global_layers_are_the_complement() {
        let cfg = Inkling2Config::from_json_str(FIXTURE).expect("fixture validates");
        assert_eq!(cfg.global_layer_ids(), vec![5, 11, 17, 23]);
        assert!(cfg.is_core(6) && cfg.is_core(17));
        assert!(!cfg.is_core(5) && !cfg.is_core(18));
    }

    #[test]
    fn a_wrong_model_type_or_architecture_is_rejected() {
        let err = mutated(|v| v["model_type"] = Value::from("qwen3_5_moe")).unwrap_err();
        assert!(err.to_string().contains("model_type"), "{err}");
        let err = mutated(|v| v["architectures"] = Value::from(vec!["Other"])).unwrap_err();
        assert!(err.to_string().contains("architectures"), "{err}");
    }

    #[test]
    fn local_layer_ids_must_be_unique_and_in_range() {
        let err = mutated(|v| v["text_config"]["local_layer_ids"][1] = Value::from(0)).unwrap_err();
        assert!(err.to_string().contains("listed twice"), "{err}");
        let err =
            mutated(|v| v["text_config"]["local_layer_ids"][0] = Value::from(24)).unwrap_err();
        assert!(err.to_string().contains("not below"), "{err}");
    }

    #[test]
    fn the_core_range_must_lie_inside_the_layers() {
        let err =
            mutated(|v| v["text_config"]["loop_config"]["core_end"] = Value::from(25)).unwrap_err();
        assert!(err.to_string().contains("core range"), "{err}");
    }

    #[test]
    fn a_mismatched_pair_of_quantization_blocks_is_rejected() {
        let err = mutated(|v| v["quantization_config"]["bits"] = Value::from(8)).unwrap_err();
        assert!(err.to_string().contains("differ"), "{err}");
    }
}
