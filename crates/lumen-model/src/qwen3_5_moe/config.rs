//! Qwen3.5-VL-MoE configuration types, deserialized from the HuggingFace `config.json`.
//!
//! The canonical document ships inside the MLX-community MXFP4 checkpoint. Field coverage below
//! mirrors that document exactly; unknown fields are tolerated so newer HF revisions don't break
//! loading. See [`tests/qwen3_5_moe_config.rs`] for a fixture-based round-trip test against the
//! real upstream file.
//!
//! Key structural facts validated here:
//!   - `layer_types` length == `text_config.num_hidden_layers` (40)
//!   - `layer_types` follows a `3×linear + 1×full` pattern (10 full attention layers total)
//!   - `quantization_config` carries per-weight overrides for the 80 gate/shared_expert_gate
//!     projections (int8-affine) on top of the global MXFP4 default
//!   - `vision_config` is present (multimodal), but vision weights remain BF16 plain — separate
//!     from the quantization pipeline

use std::collections::BTreeMap;

use serde::Deserialize;

/// Top-level config mirroring `config.json` of `mlx-community/Qwen3.6-35B-A3B-mxfp4`.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3_5MoeConfig {
    pub architectures: Vec<String>,
    pub model_type: String,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub transformers_version: Option<String>,

    /// Top-level EOS list (text generation halts on any match). 35B-A3B-mxfp4 publishes
    /// `[248046, 248044]`; 27B publishes the same; `Qwen3.5-4B-MLX-bf16` omits it entirely
    /// at top level and only carries the scalar `eos_token_id` inside `text_config`.
    /// `validate()` canonicalises the empty default by promoting `text_config.eos_token_id`
    /// into a single-element list when needed.
    #[serde(default)]
    pub eos_token_id: Vec<u32>,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,

    pub text_config: TextConfig,
    pub vision_config: VisionConfig,

    /// Quantization descriptor. MLX writes the same block under two keys (`quantization` and
    /// `quantization_config`); we parse the canonical HF name and absorb the MLX duplicate
    /// below. **Optional** because pure-bf16 MLX checkpoints (e.g. `Qwen3.5-4B-MLX-bf16`)
    /// ship without any quantization block — the loader falls back to the dense f32 path
    /// for such models.
    #[serde(default)]
    pub quantization_config: Option<QuantizationConfig>,

    /// Absorbs the MLX-specific `quantization` duplicate so serde doesn't reject it as unknown.
    /// Not exposed publicly; callers read `quantization_config` only.
    #[serde(default, rename = "quantization", skip_serializing)]
    #[allow(dead_code)]
    _mlx_duplicate_quantization: serde::de::IgnoredAny,
}

impl Qwen3_5MoeConfig {
    /// Validate cross-field invariants that serde alone cannot express, normalising
    /// fields that vary in placement across MLX checkpoint variants:
    ///   * `text_config.partial_rotary_factor` is duplicated at top-level on most
    ///     checkpoints (35B-A3B-mxfp4, 27B) but only present inside `rope_parameters`
    ///     on `Qwen3.5-4B-MLX-bf16`. We canonicalise to the top-level by copying from
    ///     `rope_parameters` when the top-level is 0 (the serde default).
    ///   * `text_config.bos_token_id` is absent on 4B but populated on 35B/27B; we
    ///     accept it as `Option<u32>` directly so no normalisation is needed.
    pub fn validate(&mut self) -> Result<(), ConfigError> {
        if self.text_config.partial_rotary_factor == 0.0
            && self.text_config.rope_parameters.partial_rotary_factor > 0.0
        {
            self.text_config.partial_rotary_factor =
                self.text_config.rope_parameters.partial_rotary_factor;
        }
        if self.eos_token_id.is_empty() {
            // Promote `text_config.eos_token_id` (scalar) into the top-level list so
            // downstream consumers don't have to special-case 4B-style configs.
            self.eos_token_id.push(self.text_config.eos_token_id);
        }
        if self.text_config.layer_types.len() != self.text_config.num_hidden_layers {
            return Err(ConfigError::LayerTypesLengthMismatch {
                num_hidden_layers: self.text_config.num_hidden_layers,
                layer_types_len: self.text_config.layer_types.len(),
            });
        }
        if self.text_config.rope_parameters.mrope_section.len() != 3 {
            return Err(ConfigError::MropeSectionNotTriplet {
                len: self.text_config.rope_parameters.mrope_section.len(),
            });
        }
        Ok(())
    }

    /// Indices of the 10 full-attention layers (rest are Mamba2 linear-attention).
    pub fn full_attention_layers(&self) -> Vec<usize> {
        self.text_config
            .layer_types
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| matches!(t, LayerType::FullAttention).then_some(i))
            .collect()
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub model_type: String,

    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub attention_dropout: f32,
    /// Qwen3-Next style gated attention output (sigmoid gate on attention values before o_proj).
    #[serde(default)]
    pub attn_output_gate: bool,

    /// Some MLX checkpoints (e.g. `Qwen3.5-4B-MLX-bf16`) omit `bos_token_id` entirely
    /// from `text_config`. Tokenizer JSON carries the canonical value in those cases.
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    pub eos_token_id: u32,
    pub pad_token_id: Option<u32>,

    pub dtype: String,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub initializer_range: f32,
    pub rms_norm_eps: f32,

    pub head_dim: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,

    pub num_hidden_layers: usize,
    /// Per-layer dispatch: `linear_attention` (Mamba2 SSM) or `full_attention` (GQA).
    pub layer_types: Vec<LayerType>,
    /// `N` in the "1 full per N layers" pattern (MLX reports 4 for 3:1).
    pub full_attention_interval: usize,

    // ─── Mamba2 linear-attention parameters ───
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub mamba_ssm_dtype: String,

    // ─── Rotary / RoPE ───
    pub max_position_embeddings: usize,
    /// Top-level mirror of `rope_parameters.partial_rotary_factor`. Some MLX checkpoints
    /// (e.g. `Qwen3.5-4B-MLX-bf16`) only carry it inside `rope_parameters`, so we tolerate
    /// it being absent at the top level. Use [`TextConfig::effective_partial_rotary_factor`]
    /// to read the resolved value (top-level wins, falls back to `rope_parameters`).
    #[serde(default)]
    pub partial_rotary_factor: f32,
    pub rope_parameters: RopeParameters,

    // ─── MLP variant ───
    //
    // Two configurations of the Qwen3.5 family share the same hybrid linear+full-attention
    // backbone and KV cache infrastructure but differ only in the per-layer MLP block:
    //
    //   * MoE (35B-A3B-mxfp4):  router + 256 routed experts + shared expert. The fields
    //                           `num_experts`, `moe_intermediate_size`,
    //                           `shared_expert_intermediate_size` are populated.
    //   * Dense (27B):          standard SwiGLU MLP with `intermediate_size` and no router.
    //                           The MoE-* fields are absent in the upstream config.json.
    //
    // All MoE fields default to 0 / false so a Dense config parses cleanly. Use
    // [`TextConfig::mlp_kind`] (presence check) rather than reading these directly.
    #[serde(default)]
    pub num_experts: usize,
    #[serde(default)]
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub moe_intermediate_size: usize,
    #[serde(default)]
    pub shared_expert_intermediate_size: usize,
    #[serde(default)]
    pub router_aux_loss_coef: f32,
    #[serde(default)]
    pub output_router_logits: bool,

    /// Dense SwiGLU intermediate dim (Qwen3.6-27B: 17408). Absent from MoE configs (the
    /// per-expert size is given by `moe_intermediate_size` instead).
    #[serde(default)]
    pub intermediate_size: usize,

    // ─── Multi-Token Prediction (weights absent in current checkpoint, see plan §6.5) ───
    #[serde(default)]
    pub mtp_num_hidden_layers: usize,
    #[serde(default)]
    pub mtp_use_dedicated_embeddings: bool,

    pub vocab_size: usize,
    pub tie_word_embeddings: bool,
    pub use_cache: bool,
}

/// Layer dispatch enum. JSON values: `"linear_attention"` | `"full_attention"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    LinearAttention,
    FullAttention,
}

/// MLP block variant. Inferred from presence of MoE-specific fields in `text_config`:
/// MoE configs carry `num_experts > 0`; Dense configs leave those fields at default and
/// instead populate `intermediate_size`. Callers branch on this enum to dispatch the
/// per-layer MLP forward (router + experts vs standard SwiGLU).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlpKind {
    /// 256-expert routed MoE with a shared expert (Qwen3.6-35B-A3B-mxfp4).
    Moe,
    /// Standard SwiGLU MLP (Qwen3.6-27B).
    Dense,
}

impl TextConfig {
    /// Infer the MLP variant from which family of size fields the config carries.
    /// Treats `num_experts > 0` as the MoE marker since both `moe_intermediate_size`
    /// and `shared_expert_intermediate_size` are tied to that count.
    pub fn mlp_kind(&self) -> MlpKind {
        if self.num_experts > 0 {
            MlpKind::Moe
        } else {
            MlpKind::Dense
        }
    }

    /// Dense variant intermediate dim. Panics if called on a MoE config — guard with
    /// [`Self::mlp_kind`] first.
    pub fn dense_intermediate_size(&self) -> usize {
        debug_assert_eq!(
            self.mlp_kind(),
            MlpKind::Dense,
            "dense_intermediate_size on a MoE TextConfig",
        );
        self.intermediate_size
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub rope_type: String,
    pub rope_theta: f32,
    #[serde(default)]
    pub partial_rotary_factor: f32,
    /// Multimodal-aware RoPE — interleaves rotary across temporal/spatial dims.
    #[serde(default)]
    pub mrope_interleaved: bool,
    /// Per-axis section widths for mRoPE. Canonical value `[11, 11, 10]` (text/image/video).
    #[serde(default)]
    pub mrope_section: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub model_type: String,
    pub depth: usize,
    pub hidden_size: usize,
    pub hidden_act: String,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub num_position_embeddings: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub out_hidden_size: usize,
    pub initializer_range: f32,
    #[serde(default)]
    pub deepstack_visual_indexes: Vec<usize>,
}

/// Quantization descriptor.
///
/// The document carries a default quantization mode (`mxfp4`, group_size=32, bits=4) at the
/// top level, plus per-weight overrides keyed by fully-qualified weight path. In the shipped
/// checkpoint the only overrides are 80 entries covering `mlp.gate` and `mlp.shared_expert_gate`
/// across all 40 text layers (all use `group_size=64, bits=8` int8-affine).
#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub group_size: usize,
    pub bits: usize,
    pub mode: String,
    /// Per-weight overrides. Key is the flat weight path (e.g.
    /// `"language_model.model.layers.0.mlp.gate"`).
    #[serde(flatten)]
    pub overrides: BTreeMap<String, QuantOverride>,
}

impl QuantizationConfig {
    /// Lookup the quantization parameters effective for a specific weight path.
    /// Falls back to the global default when no override applies.
    pub fn for_weight(&self, path: &str) -> QuantParams<'_> {
        if let Some(ovr) = self.overrides.get(path) {
            QuantParams {
                group_size: ovr.group_size,
                bits: ovr.bits,
                // Overrides change precision (4→8) but not format kind; we retain the global
                // mode label since MLX writes no per-override mode. Callers dispatch on
                // `bits`: 4 → MXFP4 path, 8 → int8-affine path.
                mode: &self.mode,
            }
        } else {
            QuantParams {
                group_size: self.group_size,
                bits: self.bits,
                mode: &self.mode,
            }
        }
    }
}

/// Per-weight quantization override. Only `group_size` and `bits` vary across entries.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct QuantOverride {
    pub group_size: usize,
    pub bits: usize,
}

/// Resolved quantization parameters for a single weight (global default or override).
#[derive(Debug, Clone, Copy)]
pub struct QuantParams<'a> {
    pub group_size: usize,
    pub bits: usize,
    pub mode: &'a str,
}

/// Errors from cross-field validation.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error(
        "layer_types length ({layer_types_len}) != num_hidden_layers ({num_hidden_layers})"
    )]
    LayerTypesLengthMismatch {
        num_hidden_layers: usize,
        layer_types_len: usize,
    },
    #[error("mrope_section must have 3 entries (text/image/video), got {len}")]
    MropeSectionNotTriplet { len: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE_JSON: &str = include_str!(
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/qwen3_5_moe_config.json")
    );

    #[test]
    fn parses_real_mlx_config() {
        let mut cfg: Qwen3_5MoeConfig = serde_json::from_str(FIXTURE_JSON)
            .expect("real mlx-community config should deserialize");
        cfg.validate().expect("cross-field invariants should hold");

        assert_eq!(cfg.model_type, "qwen3_5_moe");
        assert_eq!(cfg.architectures, vec!["Qwen3_5MoeForConditionalGeneration"]);
        assert!(!cfg.tie_word_embeddings);
        assert_eq!(cfg.eos_token_id, vec![248046, 248044]);
        assert_eq!(cfg.image_token_id, 248056);
        assert_eq!(cfg.video_token_id, 248057);
    }

    #[test]
    fn text_config_invariants_match_plan() {
        let cfg: Qwen3_5MoeConfig = serde_json::from_str(FIXTURE_JSON).unwrap();
        let t = &cfg.text_config;
        assert_eq!(t.num_hidden_layers, 40);
        assert_eq!(t.hidden_size, 2048);
        assert_eq!(t.num_attention_heads, 16);
        assert_eq!(t.num_key_value_heads, 2);
        assert_eq!(t.head_dim, 256);
        assert_eq!(t.vocab_size, 248320);
        assert_eq!(t.max_position_embeddings, 262144);
        assert_eq!(t.num_experts, 256);
        assert_eq!(t.num_experts_per_tok, 8);
        assert_eq!(t.moe_intermediate_size, 512);
        assert_eq!(t.shared_expert_intermediate_size, 512);
        assert!(t.attn_output_gate, "Qwen3-Next gated output should be enabled");
        assert_eq!(t.linear_conv_kernel_dim, 4);
        assert_eq!(t.linear_num_key_heads, 16);
        assert_eq!(t.linear_num_value_heads, 32);
        assert_eq!(t.partial_rotary_factor, 0.25);
    }

    #[test]
    fn layer_types_follow_3_linear_1_full_pattern() {
        let cfg: Qwen3_5MoeConfig = serde_json::from_str(FIXTURE_JSON).unwrap();
        let full = cfg.full_attention_layers();
        assert_eq!(full, vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39]);

        for (idx, ty) in cfg.text_config.layer_types.iter().enumerate() {
            let expected = if (idx + 1) % 4 == 0 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            };
            assert_eq!(*ty, expected, "layer {idx} type mismatch");
        }
    }

    #[test]
    fn rope_parameters_match_mrope_convention() {
        let cfg: Qwen3_5MoeConfig = serde_json::from_str(FIXTURE_JSON).unwrap();
        let r = &cfg.text_config.rope_parameters;
        assert_eq!(r.rope_type, "default");
        assert_eq!(r.rope_theta, 10_000_000.0);
        assert!(r.mrope_interleaved);
        assert_eq!(r.mrope_section, vec![11, 11, 10]);
        assert_eq!(r.partial_rotary_factor, 0.25);
    }

    #[test]
    fn vision_config_is_present_and_non_quantized() {
        let cfg: Qwen3_5MoeConfig = serde_json::from_str(FIXTURE_JSON).unwrap();
        let v = &cfg.vision_config;
        assert_eq!(v.depth, 27);
        assert_eq!(v.hidden_size, 1152);
        assert_eq!(v.out_hidden_size, cfg.text_config.hidden_size);
        assert_eq!(v.patch_size, 16);
        assert_eq!(v.spatial_merge_size, 2);
        assert_eq!(v.temporal_patch_size, 2);
        assert_eq!(v.num_position_embeddings, 2304);
    }

    #[test]
    fn quantization_global_is_mxfp4_group_32_bits_4() {
        let cfg: Qwen3_5MoeConfig = serde_json::from_str(FIXTURE_JSON).unwrap();
        let q = cfg
            .quantization_config
            .as_ref()
            .expect("MoE fixture has quantization_config");
        assert_eq!(q.mode, "mxfp4");
        assert_eq!(q.group_size, 32);
        assert_eq!(q.bits, 4);
    }

    #[test]
    fn quantization_overrides_cover_all_gates_with_int8_affine() {
        let cfg: Qwen3_5MoeConfig = serde_json::from_str(FIXTURE_JSON).unwrap();
        let q = cfg
            .quantization_config
            .as_ref()
            .expect("MoE fixture has quantization_config");
        // 40 layers × 2 gate paths = 80 override entries
        assert_eq!(q.overrides.len(), 80);
        for (path, ovr) in &q.overrides {
            assert!(
                path.ends_with(".mlp.gate") || path.ends_with(".mlp.shared_expert_gate"),
                "unexpected override path: {path}"
            );
            assert_eq!(ovr.group_size, 64, "gate override group_size");
            assert_eq!(ovr.bits, 8, "gate override bits");
        }
    }

    #[test]
    fn for_weight_returns_overrides_and_defaults() {
        let cfg: Qwen3_5MoeConfig = serde_json::from_str(FIXTURE_JSON).unwrap();
        let q = cfg
            .quantization_config
            .as_ref()
            .expect("MoE fixture has quantization_config");

        let gate = q.for_weight("language_model.model.layers.3.mlp.gate");
        assert_eq!(gate.bits, 8);
        assert_eq!(gate.group_size, 64);

        let regular = q.for_weight("language_model.model.layers.3.mlp.switch_mlp.down_proj");
        assert_eq!(regular.bits, 4);
        assert_eq!(regular.group_size, 32);
        assert_eq!(regular.mode, "mxfp4");
    }

    #[test]
    fn rejects_layer_types_length_mismatch() {
        let mut cfg: Qwen3_5MoeConfig = serde_json::from_str(FIXTURE_JSON).unwrap();
        cfg.text_config.layer_types.pop();
        let err = cfg.validate().unwrap_err();
        assert!(matches!(err, ConfigError::LayerTypesLengthMismatch { .. }));
    }

    #[test]
    fn tolerates_mlx_quantization_duplicate_key() {
        // Construct a minimal config that has BOTH `quantization` and `quantization_config`
        // (mirroring MLX's redundant serialization) and ensure we absorb the extra without error.
        let json = r#"{
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "model_type": "qwen3_5_moe",
            "eos_token_id": [1, 2],
            "image_token_id": 3, "video_token_id": 4,
            "vision_start_token_id": 5, "vision_end_token_id": 6,
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "bos_token_id": 0, "eos_token_id": 1,
                "dtype": "bfloat16", "hidden_act": "silu",
                "hidden_size": 8, "initializer_range": 0.02, "rms_norm_eps": 1e-6,
                "head_dim": 4, "num_attention_heads": 2, "num_key_value_heads": 1,
                "num_hidden_layers": 1, "layer_types": ["full_attention"],
                "full_attention_interval": 1,
                "linear_conv_kernel_dim": 4, "linear_key_head_dim": 8, "linear_value_head_dim": 8,
                "linear_num_key_heads": 1, "linear_num_value_heads": 1, "mamba_ssm_dtype": "float32",
                "max_position_embeddings": 32, "partial_rotary_factor": 1.0,
                "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
                "num_experts": 2, "num_experts_per_tok": 1,
                "moe_intermediate_size": 4, "shared_expert_intermediate_size": 4,
                "vocab_size": 100, "tie_word_embeddings": false, "use_cache": true
            },
            "vision_config": {
                "model_type": "qwen3_5_moe", "depth": 1, "hidden_size": 8, "hidden_act": "gelu",
                "intermediate_size": 16, "num_heads": 1, "in_channels": 3,
                "patch_size": 16, "num_position_embeddings": 1,
                "spatial_merge_size": 2, "temporal_patch_size": 2,
                "out_hidden_size": 8, "initializer_range": 0.02
            },
            "quantization":        {"group_size": 32, "bits": 4, "mode": "mxfp4"},
            "quantization_config": {"group_size": 32, "bits": 4, "mode": "mxfp4"}
        }"#;
        let cfg: Qwen3_5MoeConfig = serde_json::from_str(json).expect("duplicate key tolerated");
        assert_eq!(
            cfg.quantization_config
                .as_ref()
                .expect("MLX duplicate test fixture has quantization_config")
                .mode,
            "mxfp4"
        );
    }

    #[test]
    fn moe_fixture_reports_moe_mlp_kind() {
        let cfg: Qwen3_5MoeConfig = serde_json::from_str(FIXTURE_JSON).unwrap();
        assert_eq!(cfg.text_config.mlp_kind(), MlpKind::Moe);
        assert_eq!(cfg.text_config.num_experts, 256);
        assert_eq!(cfg.text_config.intermediate_size, 0, "MoE config has no top-level intermediate_size");
    }

    #[test]
    fn parses_dense_27b_style_config() {
        // Mirrors `mlx-community/Qwen3.6-27B-4bit/config.json`: same hybrid linear+full backbone
        // as 35B-A3B but a Dense SwiGLU MLP (no `num_experts`, has `intermediate_size`).
        // Quantization is uniform 4-bit affine (no mxfp4 per-weight overrides).
        let json = r#"{
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "eos_token_id": [248046, 248044],
            "image_token_id": 248056, "video_token_id": 248057,
            "vision_start_token_id": 248053, "vision_end_token_id": 248054,
            "text_config": {
                "model_type": "qwen3_5_text",
                "attn_output_gate": true,
                "bos_token_id": 248044, "eos_token_id": 248044,
                "dtype": "bfloat16", "hidden_act": "silu",
                "hidden_size": 5120, "initializer_range": 0.02, "rms_norm_eps": 1e-6,
                "head_dim": 256, "num_attention_heads": 24, "num_key_value_heads": 4,
                "num_hidden_layers": 4,
                "layer_types": ["linear_attention","linear_attention","linear_attention","full_attention"],
                "full_attention_interval": 4,
                "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
                "linear_num_key_heads": 16, "linear_num_value_heads": 48, "mamba_ssm_dtype": "float32",
                "max_position_embeddings": 262144, "partial_rotary_factor": 0.25,
                "rope_parameters": {
                    "rope_type": "default", "rope_theta": 10000000.0,
                    "partial_rotary_factor": 0.25, "mrope_interleaved": true,
                    "mrope_section": [11, 11, 10]
                },
                "intermediate_size": 17408,
                "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": false,
                "vocab_size": 248320, "tie_word_embeddings": false, "use_cache": true
            },
            "vision_config": {
                "model_type": "qwen3_5", "depth": 27, "hidden_size": 1152, "hidden_act": "gelu_pytorch_tanh",
                "intermediate_size": 4304, "num_heads": 16, "in_channels": 3,
                "patch_size": 16, "num_position_embeddings": 2304,
                "spatial_merge_size": 2, "temporal_patch_size": 2,
                "out_hidden_size": 5120, "initializer_range": 0.02
            },
            "quantization":        {"group_size": 64, "bits": 4, "mode": "affine"},
            "quantization_config": {"group_size": 64, "bits": 4, "mode": "affine"}
        }"#;
        let mut cfg: Qwen3_5MoeConfig =
            serde_json::from_str(json).expect("dense 27B-style config should parse");
        cfg.validate().expect("cross-field invariants for dense");

        let t = &cfg.text_config;
        assert_eq!(t.mlp_kind(), MlpKind::Dense);
        assert_eq!(t.dense_intermediate_size(), 17408);
        assert_eq!(t.num_experts, 0);
        assert_eq!(t.moe_intermediate_size, 0);
        assert_eq!(t.shared_expert_intermediate_size, 0);
        assert!(t.attn_output_gate, "27B Dense keeps gated output");
        assert_eq!(t.partial_rotary_factor, 0.25);
        assert_eq!(t.mtp_num_hidden_layers, 1, "27B has built-in MTP head");

        // Quantization: uniform 4-bit affine, no per-weight overrides like MoE has.
        let q = cfg
            .quantization_config
            .as_ref()
            .expect("dense 27B fixture carries quantization_config");
        assert_eq!(q.mode, "affine");
        assert_eq!(q.bits, 4);
        assert_eq!(q.group_size, 64);
        assert!(q.overrides.is_empty(), "27B has no per-weight quant overrides");
    }

    #[test]
    fn parses_dense_4b_bf16_config_without_quantization_block() {
        // `mlx-community/Qwen3.5-4B-MLX-bf16` ships pure-bf16 (no quantization_config block),
        // dense MLP, hybrid linear+full-attn 3:1 (32 layers = 24 linear + 8 full).
        let json = r#"{
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "eos_token_id": [248044],
            "image_token_id": 248056, "video_token_id": 248057,
            "vision_start_token_id": 248053, "vision_end_token_id": 248054,
            "text_config": {
                "model_type": "qwen3_5_text",
                "attn_output_gate": true,
                "bos_token_id": 248044, "eos_token_id": 248044,
                "dtype": "bfloat16", "hidden_act": "silu",
                "hidden_size": 2560, "initializer_range": 0.02, "rms_norm_eps": 1e-6,
                "head_dim": 256, "num_attention_heads": 16, "num_key_value_heads": 4,
                "num_hidden_layers": 4,
                "layer_types": ["linear_attention","linear_attention","linear_attention","full_attention"],
                "full_attention_interval": 4,
                "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
                "linear_num_key_heads": 16, "linear_num_value_heads": 32, "mamba_ssm_dtype": "float32",
                "max_position_embeddings": 262144, "partial_rotary_factor": 0.25,
                "rope_parameters": {
                    "rope_type": "default", "rope_theta": 10000000.0,
                    "partial_rotary_factor": 0.25, "mrope_interleaved": true,
                    "mrope_section": [11, 11, 10]
                },
                "intermediate_size": 9216,
                "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": false,
                "vocab_size": 248320, "tie_word_embeddings": true, "use_cache": true
            },
            "vision_config": {
                "model_type": "qwen3_5", "depth": 24, "hidden_size": 1024, "hidden_act": "gelu_pytorch_tanh",
                "intermediate_size": 4096, "num_heads": 16, "in_channels": 3,
                "patch_size": 16, "num_position_embeddings": 2304,
                "spatial_merge_size": 2, "temporal_patch_size": 2,
                "out_hidden_size": 2560, "initializer_range": 0.02
            }
        }"#;
        let mut cfg: Qwen3_5MoeConfig =
            serde_json::from_str(json).expect("4B bf16 config should parse without quantization block");
        cfg.validate().expect("cross-field invariants for 4B bf16");
        // After validate(), top-level partial_rotary_factor is canonicalised from
        // rope_parameters when the upstream checkpoint omits the duplicate (4B case).
        assert_eq!(cfg.text_config.partial_rotary_factor, 0.25);
        assert_eq!(cfg.text_config.mlp_kind(), MlpKind::Dense);
        assert_eq!(cfg.text_config.dense_intermediate_size(), 9216);
        assert!(cfg.text_config.tie_word_embeddings, "4B uses tied embedding");
        assert!(
            cfg.quantization_config.is_none(),
            "4B bf16 has no quantization_config block"
        );
    }
}
