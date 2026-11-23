//! The text model's hyper-parameters, from the `text_config` block of `config.json`, and the
//! schedule of layer visits the loop produces.
//!
//! Derived from the `TextArgs` of oMLX's Inkling 2 port (`omlx/patches/inkling2/model.py`,
//! Apache-2.0). The field names are the checkpoint's. Unlike the manifest's [`super::Inkling2Config`],
//! which checks the whole checkpoint, this reads only what the forward pass needs, so the small
//! test replica (no vision tower, no draft head) parses too.

use std::ops::Range;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use serde_json::Value;

/// The most passes of the core this port takes. The checkpoint runs 3; every pass is a layer visit
/// with a cache entry, so a number that is not a loop count (a corrupt config) is refused before
/// the schedule is built from it.
const MAX_PASSES: usize = 32;

/// The `loop_config` block: `core_start..core_end` run `num_passes` times per token.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct LoopArgs {
    pub core_start: usize,
    pub core_end: usize,
    pub num_passes: usize,
    /// Width of the low-rank projection inside each loop controller.
    pub gate_rank: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextArgs {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    /// Token ids at or past this have no embedding row of their own; their logits are `-inf`.
    pub unpadded_vocab_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub swa_num_attention_heads: usize,
    pub swa_num_key_value_heads: usize,
    pub swa_head_dim: usize,
    pub sliding_window_size: usize,
    pub d_rel: usize,
    pub rel_extent: usize,
    pub log_scaling_n_floor: usize,
    pub log_scaling_alpha: f32,
    pub rms_norm_eps: f32,
    pub qk_l2_clip_threshold: f32,
    pub rel_l2_clip_threshold: f32,
    pub local_layer_ids: Vec<usize>,
    pub sconv_kernel_size: usize,
    pub n_layers_per_attn_res_block: usize,
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    pub intermediate_size: usize,
    pub mlp_gate_softcap: f32,
    pub mlp_up_softcap: f32,
    pub route_scale: f32,
    pub final_logit_softcapping: f32,
    pub loop_config: LoopArgs,
    /// The reference lets `layer_types` override `local_layer_ids`. This checkpoint has none, and
    /// the port does not read it, so a config that carries one is refused rather than guessed at.
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    // Switches the reference does not read: it hard-codes projections without bias, a value norm
    // without gamma, unit-RMS attention-residual sources, a router bias and no shared experts. They
    // are kept to refuse a checkpoint that sets one the other way, which would load and run wrong.
    #[serde(default)]
    pub q_bias: Option<bool>,
    #[serde(default)]
    pub o_bias: Option<bool>,
    #[serde(default)]
    pub use_value_norm_no_gamma: Option<bool>,
    #[serde(default)]
    pub attn_res_normalize_values: Option<bool>,
    #[serde(default)]
    pub use_gate_bias: Option<bool>,
    #[serde(default)]
    pub n_shared_experts: Option<usize>,
}

/// Attention geometry of one layer kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttnDims {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Keys a query can see: the window on a local layer, unlimited (`0`) on a global one.
    pub window: usize,
    /// Backward distances the relative bias covers: the window on a local layer.
    pub rel_extent: usize,
}

impl TextArgs {
    /// Reads `config.json` in `dir`: the `text_config` block, validated.
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let text =
            std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        Self::from_config_str(&text).with_context(|| format!("config {}", path.display()))
    }

    pub fn from_config_str(text: &str) -> Result<Self> {
        let root: Value = serde_json::from_str(text).context("parse config.json")?;
        let model_type = root.get("model_type").and_then(Value::as_str);
        ensure!(
            model_type == Some("inkling2_loop"),
            "model_type is {model_type:?}, expected \"inkling2_loop\""
        );
        let block = root
            .get("text_config")
            .context("config.json has no text_config")?;
        let args: Self = serde_json::from_value(block.clone()).context("parse text_config")?;
        args.validate()?;
        Ok(args)
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.layer_types.is_none(),
            "text_config.layer_types is not supported; this port reads local_layer_ids"
        );
        for (field, set, wanted) in [
            ("q_bias", self.q_bias, false),
            ("o_bias", self.o_bias, false),
            (
                "use_value_norm_no_gamma",
                self.use_value_norm_no_gamma,
                true,
            ),
            (
                "attn_res_normalize_values",
                self.attn_res_normalize_values,
                true,
            ),
            ("use_gate_bias", self.use_gate_bias, true),
        ] {
            ensure!(
                set.is_none_or(|v| v == wanted),
                "text_config.{field} is {}; this port implements the {wanted} setting only",
                !wanted
            );
        }
        ensure!(
            self.n_shared_experts.is_none_or(|n| n == 0),
            "text_config.n_shared_experts is {:?}; this port has no shared experts",
            self.n_shared_experts
        );
        let n = self.num_hidden_layers;
        for &id in &self.local_layer_ids {
            ensure!(
                id < n,
                "local_layer_ids: {id} is not below num_hidden_layers {n}"
            );
        }
        ensure!(
            self.sconv_kernel_size >= 2,
            "sconv_kernel_size must be at least 2"
        );
        // The global layers divide the position by this floor.
        ensure!(
            self.log_scaling_n_floor >= 1,
            "log_scaling_n_floor must be at least 1"
        );
        ensure!(
            self.num_experts_per_tok >= 1 && self.num_experts_per_tok <= self.n_routed_experts,
            "num_experts_per_tok {} is not within 1..={}",
            self.num_experts_per_tok,
            self.n_routed_experts
        );
        for (heads, kv_heads, what) in [
            (self.num_attention_heads, self.num_key_value_heads, "global"),
            (
                self.swa_num_attention_heads,
                self.swa_num_key_value_heads,
                "local",
            ),
        ] {
            ensure!(
                kv_heads > 0 && heads % kv_heads == 0,
                "{what} layers: {heads} query heads is not a multiple of {kv_heads} key/value heads"
            );
        }
        self.layer_schedule()?;
        Ok(())
    }

    pub fn is_sliding(&self, layer: usize) -> bool {
        self.local_layer_ids.contains(&layer)
    }

    /// Whether `layer` is in the loop's core.
    pub fn in_core(&self, layer: usize) -> bool {
        self.core().contains(&layer)
    }

    pub fn core(&self) -> Range<usize> {
        self.loop_config.core_start..self.loop_config.core_end
    }

    pub fn dims(&self, sliding: bool) -> AttnDims {
        if sliding {
            AttnDims {
                heads: self.swa_num_attention_heads,
                kv_heads: self.swa_num_key_value_heads,
                head_dim: self.swa_head_dim,
                window: self.sliding_window_size,
                rel_extent: self.sliding_window_size,
            }
        } else {
            AttnDims {
                heads: self.num_attention_heads,
                kv_heads: self.num_key_value_heads,
                head_dim: self.head_dim,
                window: 0,
                rel_extent: self.rel_extent,
            }
        }
    }

    /// `(layer, pass)` for every layer visit, in execution order: the layers before the core once,
    /// the core `num_passes` times, the layers after it once. Each visit owns a cache entry.
    ///
    /// The message bank of a pass is the blocks the pass before it wrote, so the core has to start
    /// and end on an attention-residual block boundary.
    pub fn layer_schedule(&self) -> Result<Vec<(usize, usize)>> {
        let n = self.num_hidden_layers;
        let core = self.core();
        let block = self.n_layers_per_attn_res_block;
        let passes = self.loop_config.num_passes;
        if block == 0
            || core.start == 0
            || core.start >= core.end
            || core.end > n
            || !core.start.is_multiple_of(block)
            || !core.end.is_multiple_of(block)
            || !(2..=MAX_PASSES).contains(&passes)
        {
            bail!(
                "unsupported loop_config: core {}..{} with {passes} passes in {n} layers of \
                 blocks of {block} (the core must start and end on a block boundary, start above \
                 0, with 2 to {MAX_PASSES} passes)",
                core.start,
                core.end
            );
        }
        let mut visits = Vec::with_capacity(n + (passes - 1) * core.len());
        visits.extend((0..core.start).map(|i| (i, 0)));
        for pass in 0..passes {
            visits.extend(core.clone().map(|i| (i, pass)));
        }
        visits.extend((core.end..n).map(|i| (i, 0)));
        Ok(visits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real checkpoint's config.
    const REAL: &str = include_str!("../../tests/fixtures/inkling2_loop/config.json");

    fn edited(edit: impl FnOnce(&mut Value)) -> Result<TextArgs> {
        let mut value: Value = serde_json::from_str(REAL).expect("fixture is JSON");
        edit(&mut value);
        TextArgs::from_config_str(&value.to_string())
    }

    #[test]
    fn the_real_config_parses_and_has_48_visits() {
        let args = TextArgs::from_config_str(REAL).expect("real config");
        assert_eq!(args.hidden_size, 2304);
        assert_eq!(args.log_scaling_n_floor, 8192);
        assert_eq!(args.dims(true).window, 1024);
        assert_eq!(args.dims(false).window, 0);
        assert_eq!(args.dims(false).rel_extent, 1024);
        let visits = args.layer_schedule().unwrap();
        assert_eq!(visits.len(), 48);
        assert_eq!(visits[5], (5, 0));
        assert_eq!(visits[6], (6, 0));
        assert_eq!(visits[18], (6, 1));
        assert_eq!(visits[30], (6, 2));
        assert_eq!(visits[41], (17, 2));
        assert_eq!(visits[42], (18, 0));
        assert_eq!(visits[47], (23, 0));
        let global: Vec<usize> = (0..24).filter(|&i| !args.is_sliding(i)).collect();
        assert_eq!(global, vec![5, 11, 17, 23]);
    }

    #[test]
    fn a_core_off_a_block_boundary_is_refused() {
        let err =
            edited(|v| v["text_config"]["loop_config"]["core_start"] = Value::from(7)).unwrap_err();
        assert!(format!("{err:#}").contains("block boundary"), "{err:#}");
    }

    #[test]
    fn a_single_pass_or_a_core_from_layer_zero_is_refused() {
        for (field, value) in [("num_passes", 1), ("core_start", 0)] {
            let err = edited(|v| v["text_config"]["loop_config"][field] = Value::from(value))
                .unwrap_err();
            assert!(
                format!("{err:#}").contains("unsupported loop_config"),
                "{field}: {err:#}"
            );
        }
    }

    /// RED before: a pass count nobody means (a damaged config) sized the schedule's allocation,
    /// and a floor of 0 reached the position scaling as a division by zero.
    #[test]
    fn a_pass_count_that_is_no_loop_and_a_zero_floor_are_refused() {
        let err =
            edited(|v| v["text_config"]["loop_config"]["num_passes"] = Value::from(1u64 << 60))
                .unwrap_err();
        assert!(
            format!("{err:#}").contains("unsupported loop_config"),
            "{err:#}"
        );
        let err = edited(|v| v["text_config"]["log_scaling_n_floor"] = Value::from(0)).unwrap_err();
        assert!(
            format!("{err:#}").contains("log_scaling_n_floor"),
            "{err:#}"
        );
    }

    #[test]
    fn layer_types_are_refused_not_guessed_at() {
        let err = edited(|v| v["text_config"]["layer_types"] = Value::from(vec!["hybrid_sliding"]))
            .unwrap_err();
        assert!(format!("{err:#}").contains("layer_types"), "{err:#}");
    }

    #[test]
    fn a_switch_the_reference_hard_codes_the_other_way_is_refused() {
        for (field, value) in [
            ("q_bias", Value::from(true)),
            ("o_bias", Value::from(true)),
            ("use_value_norm_no_gamma", Value::from(false)),
            ("attn_res_normalize_values", Value::from(false)),
            ("use_gate_bias", Value::from(false)),
            ("n_shared_experts", Value::from(1)),
        ] {
            let err = edited(|v| v["text_config"][field] = value).unwrap_err();
            assert!(format!("{err:#}").contains(field), "{field}: {err:#}");
        }
    }

    #[test]
    fn the_wrong_model_type_is_refused() {
        let err = edited(|v| v["model_type"] = Value::from("inkling2")).unwrap_err();
        assert!(format!("{err:#}").contains("inkling2_loop"), "{err:#}");
    }

    #[test]
    fn heads_must_group_evenly_and_experts_must_cover_top_k() {
        let err = edited(|v| v["text_config"]["num_key_value_heads"] = Value::from(7)).unwrap_err();
        assert!(format!("{err:#}").contains("multiple"), "{err:#}");
        let err =
            edited(|v| v["text_config"]["num_experts_per_tok"] = Value::from(129)).unwrap_err();
        assert!(
            format!("{err:#}").contains("num_experts_per_tok"),
            "{err:#}"
        );
    }
}
