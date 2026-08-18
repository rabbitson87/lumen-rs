//! Gemma 4 `config.json` parsing.
//!
//! Ungated on purpose, for the same reason `qwen35_config` is: this is pure
//! `serde` + `std::fs` with no MLX dependency, and while it sat inside
//! `gemma4_moe`'s `#[cfg(feature = "mlx-native")] mod imp` a config the loader
//! chokes on could only be tested by building the whole GPU stack. The
//! `gemma4-config-null-moe-fields` defect lived here, and its sweep should not
//! need a Metal toolchain to run. `imp` re-exports everything, so call sites
//! are untouched.

use anyhow::{Result, anyhow};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

// ───────────────────────── config.json parsing ─────────────────────────

/// Top-level config.json wrapper for `gemma4` (text-only deploy ignores
/// `vision_config`, `audio_config`, vision/image token ids).
#[derive(Debug, Clone, Deserialize)]
pub struct NativeGemma4Config {
    pub model_type: String,
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(
        default,
        rename = "eos_token_id",
        deserialize_with = "deserialize_token_ids"
    )]
    pub eos_token_ids: Vec<u32>,
    pub text_config: NativeGemma4TextConfig,
    /// `quantization` and `quantization_config` are duplicates in
    /// lmstudio's MLX shards; we accept either, with `quantization`
    /// taking precedence when both are present.
    #[serde(default)]
    pub quantization: Option<NativeGemma4QuantizationConfig>,
    #[serde(default)]
    pub quantization_config: Option<NativeGemma4QuantizationConfig>,
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,

    // ── multimodal (image) ──
    // Present on `Gemma4ForConditionalGeneration` checkpoints. All optional
    // so text-only deploys and vision-stripped quantizations still parse.
    #[serde(default)]
    pub vision_config: Option<NativeGemma4VisionConfig>,
    /// Placeholder token whose embedding rows get replaced by vision soft
    /// tokens (258880 on 26B-A4B).
    #[serde(default)]
    pub image_token_id: Option<u32>,
    /// `<start_of_image>` / `<end_of_image>` sentinels around each run.
    #[serde(default)]
    pub boi_token_id: Option<u32>,
    #[serde(default)]
    pub eoi_token_id: Option<u32>,
    /// Soft tokens the processor budgets per image (280 on 26B-A4B).
    #[serde(default)]
    pub vision_soft_tokens_per_image: Option<usize>,
}

/// `text_config` block — every field the forward path needs.
///
/// Fields absent in 26B-A4B (per-layer input embeddings: 2B/4B-only) are
/// kept with serde defaults so the same struct handles other Gemma 4
/// variants once we get there.
#[derive(Debug, Clone, Deserialize)]
pub struct NativeGemma4TextConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// Per-layer KV-head count for full attention layers when k_eq_v.
    /// 26B-A4B sets this to 2; absent in smaller variants.
    #[serde(default, deserialize_with = "crate::config_serde::null_as_default")]
    pub num_global_key_value_heads: Option<usize>,
    /// Head dim for sliding attention layers (and the default fallback).
    pub head_dim: usize,
    /// Head dim for full attention layers (26B/31B). 0 means "use head_dim".
    #[serde(default)]
    pub global_head_dim: usize,
    pub vocab_size: usize,
    #[serde(default = "default_vocab_size_per_layer_input")]
    pub vocab_size_per_layer_input: usize,
    pub rms_norm_eps: f32,
    pub layer_types: Vec<NativeGemma4LayerType>,
    pub sliding_window: usize,
    #[serde(default = "default_sliding_window_pattern")]
    pub sliding_window_pattern: usize,
    pub max_position_embeddings: usize,

    // RoPE
    pub rope_parameters: NativeGemma4RopeParameters,
    #[serde(default = "default_rope_traditional")]
    pub rope_traditional: bool,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,

    // Attention behavior
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub attention_dropout: f32,
    #[serde(default, deserialize_with = "crate::config_serde::null_as_default")]
    pub attention_k_eq_v: bool,

    // MoE
    #[serde(default, deserialize_with = "crate::config_serde::null_as_default")]
    pub enable_moe_block: bool,
    #[serde(default, deserialize_with = "crate::config_serde::null_as_default")]
    pub num_experts: usize,
    #[serde(default, deserialize_with = "crate::config_serde::null_as_default")]
    pub top_k_experts: usize,
    #[serde(default, deserialize_with = "crate::config_serde::null_as_default")]
    pub moe_intermediate_size: usize,

    // Dense MLP (always present in 26B; sized via intermediate_size).
    pub intermediate_size: usize,
    #[serde(default, deserialize_with = "crate::config_serde::null_as_default")]
    pub use_double_wide_mlp: bool,

    // Per-layer input embedding (2B/4B Gemma 4; 0 for 26B-A4B).
    #[serde(default, deserialize_with = "crate::config_serde::null_as_default")]
    pub hidden_size_per_layer_input: usize,
    #[serde(default, deserialize_with = "crate::config_serde::null_as_default")]
    pub num_kv_shared_layers: usize,

    // Activation + logit softcap
    #[serde(default = "default_hidden_activation")]
    pub hidden_activation: String,
    pub final_logit_softcapping: f32,

    // Tied embedding → lm_head reuses embed_tokens.
    #[serde(default)]
    pub tie_word_embeddings: bool,

    // Tokens
    #[serde(default)]
    pub pad_token_id: u32,
    #[serde(default = "default_bos_token_id")]
    pub bos_token_id: u32,
    #[serde(
        default,
        rename = "eos_token_id",
        deserialize_with = "deserialize_token_ids"
    )]
    pub eos_token_ids: Vec<u32>,
}

/// RoPE parameter block, keyed per attention layer kind. mlx-lm's
/// `gemma4_text.py` looks up `rope_parameters["sliding_attention"|"full_attention"]`.
#[derive(Debug, Clone, Deserialize)]
pub struct NativeGemma4RopeParameters {
    pub full_attention: NativeGemma4RopePerKind,
    pub sliding_attention: NativeGemma4RopePerKind,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NativeGemma4RopePerKind {
    pub rope_theta: f32,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    #[serde(default = "default_rope_type")]
    pub rope_type: String,
}

/// Layer kind discriminant. mlx-lm config.json string values match
/// `snake_case` form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeGemma4LayerType {
    SlidingAttention,
    FullAttention,
}

impl NativeGemma4LayerType {
    pub fn is_sliding(self) -> bool {
        matches!(self, NativeGemma4LayerType::SlidingAttention)
    }

    pub fn is_full(self) -> bool {
        matches!(self, NativeGemma4LayerType::FullAttention)
    }
}

/// Quantization block — uniform `(group_size, bits, mode)` default plus
/// per-tensor 8-bit overrides for `mlp.{gate,up,down}_proj` and
/// `router.proj` (and any future override entries).
#[derive(Debug, Clone, Deserialize)]
pub struct NativeGemma4QuantizationConfig {
    pub group_size: usize,
    pub bits: usize,
    pub mode: String,
    #[serde(flatten)]
    pub overrides: BTreeMap<String, NativeGemma4QuantizationOverride>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NativeGemma4QuantizationOverride {
    pub group_size: usize,
    pub bits: usize,
    /// Optional per-tensor mode override. When absent the override
    /// inherits MODE_AFFINE (Qwen3.6 reference convention: overrides
    /// encode AFFINE exceptions inside an otherwise non-AFFINE model,
    /// e.g. 8-bit AFFINE gate layers inside an MXFP4 model). When
    /// present, must be one of `"affine" | "mxfp4"`.
    #[serde(default)]
    pub mode: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TokenIdValue {
    One(u32),
    Many(Vec<u32>),
}

fn deserialize_token_ids<'de, D>(deserializer: D) -> std::result::Result<Vec<u32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<TokenIdValue>::deserialize(deserializer)?;
    Ok(match value {
        Some(TokenIdValue::One(id)) => vec![id],
        Some(TokenIdValue::Many(ids)) => ids,
        None => Vec::new(),
    })
}

fn default_sliding_window_pattern() -> usize {
    6
}

fn default_partial_rotary_factor() -> f32 {
    1.0
}

fn default_rope_traditional() -> bool {
    false
}

fn default_rope_type() -> String {
    "default".to_string()
}

fn default_hidden_activation() -> String {
    "gelu_pytorch_tanh".to_string()
}

fn default_bos_token_id() -> u32 {
    2
}

fn default_vocab_size_per_layer_input() -> usize {
    0
}

impl NativeGemma4Config {
    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|err| anyhow!("config.json read failed at {}: {err}", path.display()))?;
        let mut cfg: Self = serde_json::from_str(&raw)
            .map_err(|err| anyhow!("config.json parse failed at {}: {err}", path.display()))?;
        // `LUMEN_SLIDING_WINDOW` (desktop CONTEXT card → server) overrides the
        // model's built-in sliding window size. 0 means "no override".
        if let Ok(s) = std::env::var("LUMEN_SLIDING_WINDOW")
            && let Ok(n) = s.parse::<usize>()
            && n > 0
        {
            eprintln!(
                "[gemma4] sliding_window override via LUMEN_SLIDING_WINDOW: {} → {n}",
                cfg.text_config.sliding_window
            );
            cfg.text_config.sliding_window = n;
        }
        // `LUMEN_MAX_CTX` caps the maximum position embeddings the model
        // advertises — useful to keep KV cache pool sizing predictable
        // when the model config claims e.g. 128K but the host RAM can't
        // hold it.
        if let Ok(s) = std::env::var("LUMEN_MAX_CTX")
            && let Ok(n) = s.parse::<usize>()
            && n > 0
            && n < cfg.text_config.max_position_embeddings
        {
            eprintln!(
                "[gemma4] max_position_embeddings capped via LUMEN_MAX_CTX: {} → {n}",
                cfg.text_config.max_position_embeddings
            );
            cfg.text_config.max_position_embeddings = n;
        }
        // `LUMEN_GEMMA4_TOP_K` overrides the MoE router's top-k expert
        // count at load time. Quality knob — model was trained at k=8;
        // lowering to k=4 ~halves expert FFN compute per token but may
        // degrade output. Use for A/B measurement; ship only after
        // multi-axis quality eval (HAERAE / KMMLU / GSM8K).
        if let Ok(s) = std::env::var("LUMEN_GEMMA4_TOP_K")
            && let Ok(n) = s.parse::<usize>()
        {
            if n > 0 && n <= cfg.text_config.num_experts {
                if n != cfg.text_config.top_k_experts {
                    eprintln!(
                        "[gemma4] top_k_experts overridden via LUMEN_GEMMA4_TOP_K: {} → {n}",
                        cfg.text_config.top_k_experts
                    );
                    cfg.text_config.top_k_experts = n;
                }
            } else {
                eprintln!(
                    "[gemma4] LUMEN_GEMMA4_TOP_K={n} ignored (must be 1..={}, got {n})",
                    cfg.text_config.num_experts
                );
            }
        }
        Ok(cfg)
    }

    /// Returns whichever of `quantization` / `quantization_config` is
    /// present (preferring the former, which is what lmstudio's MLX
    /// shards reference at runtime).
    pub fn effective_quantization(&self) -> Option<&NativeGemma4QuantizationConfig> {
        self.quantization
            .as_ref()
            .or(self.quantization_config.as_ref())
    }

    /// Validate that this config belongs to the Gemma 4 family with a
    /// reachable text-only forward path.
    pub fn validate_gemma4_family(&self) -> Result<()> {
        if self.model_type != "gemma4" {
            return Err(anyhow!(
                "expected model_type='gemma4', got '{}'",
                self.model_type
            ));
        }
        if !self.architectures.is_empty()
            && !self
                .architectures
                .iter()
                .any(|a| a == "Gemma4ForConditionalGeneration")
        {
            return Err(anyhow!(
                "expected architectures to include 'Gemma4ForConditionalGeneration', got {:?}",
                self.architectures
            ));
        }
        self.text_config.validate()?;
        if let Some(quant) = self.effective_quantization() {
            if !matches!(quant.mode.as_str(), "affine" | "mxfp4" | "mxfp8" | "nvfp4")
                || quant.group_size == 0
            {
                return Err(anyhow!(
                    "quantization default must be mode∈{{affine, mxfp4, mxfp8, nvfp4}} with non-zero group, got mode='{}' bits={} group={}",
                    quant.mode,
                    quant.bits,
                    quant.group_size
                ));
            }
            if !matches!(quant.bits, 2 | 3 | 4 | 5 | 6 | 8) {
                return Err(anyhow!(
                    "quantization default bits must be in {{2,3,4,5,6,8}} (mlx-supported), got {}",
                    quant.bits
                ));
            }
            // (Sanity probe removed: it asserted the specific lmstudio
            // mixed 4/8 packaging, which doesn't hold for in-house
            // conversions via `mlx_lm.convert` — those may keep MLPs at
            // the default bit-width or pick different mixed recipes.
            // Per-key dispatch via `quant_params_for` handles whatever
            // overrides the config actually contains.)
            // Override `group_size` is only required to match the default
            // when the override's mode matches the default mode — when
            // modes differ (e.g. MXFP4 g=32 default with AFFINE g=64 embed
            // override), per-tensor dispatch through `quant_params_for`
            // routes each tensor to a kernel that consumes its own
            // `(group_size, bits, mode)` triple, so cross-mode group_size
            // mismatches are safe. This mirrors Qwen3.5's loader which has
            // no mixed-group-size check at all and ships in production
            // (Qwen3.6 MXFP4 g=32 default + AFFINE g=64 gate overrides).
            let top_mode = quant.mode.as_str();
            for (k, ov) in &quant.overrides {
                let ov_mode = ov.mode.as_deref().unwrap_or("affine");
                let modes_match = ov_mode == top_mode;
                if modes_match && ov.group_size != quant.group_size {
                    return Err(anyhow!(
                        "override '{k}' has group_size={} but default is {} (same mode={top_mode}) — mixed group_size within one mode not supported",
                        ov.group_size,
                        quant.group_size
                    ));
                }
                if !matches!(ov.bits, 2 | 3 | 4 | 5 | 6 | 8) {
                    return Err(anyhow!(
                        "override '{k}' bits must be in {{2,3,4,5,6,8}} (mlx-supported), got {}",
                        ov.bits
                    ));
                }
            }
        }
        Ok(())
    }
}

impl NativeGemma4TextConfig {
    pub fn validate(&self) -> Result<()> {
        if self.model_type != "gemma4_text" {
            return Err(anyhow!(
                "expected text_config.model_type='gemma4_text', got '{}'",
                self.model_type
            ));
        }
        if self.hidden_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.head_dim == 0
            || self.vocab_size == 0
            || self.intermediate_size == 0
        {
            return Err(anyhow!(
                "text_config has zero-valued core dims: hidden={} layers={} q_heads={} kv_heads={} head_dim={} vocab={} mlp_inter={}",
                self.hidden_size,
                self.num_hidden_layers,
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
                self.vocab_size,
                self.intermediate_size,
            ));
        }
        if self.layer_types.len() != self.num_hidden_layers {
            return Err(anyhow!(
                "layer_types length {} != num_hidden_layers {}",
                self.layer_types.len(),
                self.num_hidden_layers
            ));
        }
        if self.sliding_window == 0 {
            return Err(anyhow!("sliding_window must be > 0"));
        }
        if self.final_logit_softcapping <= 0.0 {
            return Err(anyhow!(
                "final_logit_softcapping must be > 0, got {}",
                self.final_logit_softcapping
            ));
        }
        // 26B-A4B has enable_moe_block=true + num_experts=128 + top_k=8.
        // For now we keep this validator strict so unsupported variants
        // surface early.
        if self.enable_moe_block {
            if self.num_experts == 0 {
                return Err(anyhow!("enable_moe_block=true but num_experts=0"));
            }
            if self.top_k_experts == 0 || self.top_k_experts > self.num_experts {
                return Err(anyhow!(
                    "top_k_experts {} invalid against num_experts {}",
                    self.top_k_experts,
                    self.num_experts
                ));
            }
            if self.moe_intermediate_size == 0 {
                return Err(anyhow!(
                    "moe_intermediate_size must be > 0 when enable_moe_block=true"
                ));
            }
        }
        // sliding/full split sanity: every num_hidden_layers/sliding_window_pattern
        // positions should be a full attention layer (5 sliding + 1 full
        // for 26B-A4B's pattern=6).
        let n_full = self.layer_types.iter().filter(|t| t.is_full()).count();
        let expected_full = self.num_hidden_layers / self.sliding_window_pattern;
        // `n_full > num_hidden_layers` is unreachable — it counts entries of
        // a list whose length was just checked equal to `num_hidden_layers` —
        // but it keeps the message honest if that check ever moves.
        if n_full == 0 || lumen_core::never!(n_full > self.num_hidden_layers) {
            return Err(anyhow!(
                "layer_types has {n_full} full_attention entries, expected ~{expected_full} given pattern={}",
                self.sliding_window_pattern,
            ));
        }
        Ok(())
    }

    /// Resolved head dim for a given layer kind. Sliding layers use
    /// `head_dim`; full layers use `global_head_dim` when set.
    pub fn head_dim_for(&self, kind: NativeGemma4LayerType) -> usize {
        match kind {
            NativeGemma4LayerType::FullAttention if self.global_head_dim != 0 => {
                self.global_head_dim
            }
            _ => self.head_dim,
        }
    }

    /// Resolved KV-head count for a given layer kind. When
    /// `attention_k_eq_v` is set and the layer is full attention,
    /// `num_global_key_value_heads` (if present) overrides
    /// `num_key_value_heads`.
    pub fn n_kv_heads_for(&self, kind: NativeGemma4LayerType) -> usize {
        match kind {
            NativeGemma4LayerType::FullAttention
                if self.attention_k_eq_v && self.num_global_key_value_heads.is_some() =>
            {
                self.num_global_key_value_heads.unwrap()
            }
            _ => self.num_key_value_heads,
        }
    }

    /// Returns true iff this layer should drop the `v_proj` tensor and
    /// reuse `k_proj` as `values` (full attention layers only when
    /// `attention_k_eq_v` is set).
    pub fn use_k_eq_v_for(&self, kind: NativeGemma4LayerType) -> bool {
        self.attention_k_eq_v && kind.is_full()
    }

    /// Resolved RoPE block for a given layer kind.
    pub fn rope_for(&self, kind: NativeGemma4LayerType) -> &NativeGemma4RopePerKind {
        match kind {
            NativeGemma4LayerType::FullAttention => &self.rope_parameters.full_attention,
            NativeGemma4LayerType::SlidingAttention => &self.rope_parameters.sliding_attention,
        }
    }
}

// ─────────────────── vision_config block ───────────────────

/// `vision_config` block of Gemma 4's config.json.
#[derive(Debug, Clone, Deserialize)]
pub struct NativeGemma4VisionConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    pub position_embedding_size: usize,
    pub rms_norm_eps: f32,
    pub rope_parameters: NativeGemma4VisionRope,
    #[serde(default)]
    pub standardize: bool,
    #[serde(default)]
    pub use_clipped_linears: bool,
    #[serde(default = "default_vision_activation")]
    pub hidden_activation: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NativeGemma4VisionRope {
    pub rope_theta: f32,
}

fn default_vision_activation() -> String {
    "gelu_pytorch_tanh".to_string()
}

impl NativeGemma4VisionConfig {
    pub fn validate(&self) -> Result<()> {
        if self.model_type != "gemma4_vision" {
            return Err(anyhow!(
                "expected vision_config.model_type='gemma4_vision', got '{}'",
                self.model_type
            ));
        }
        if self.hidden_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.head_dim == 0
            || self.patch_size == 0
            || self.pooling_kernel_size == 0
        {
            return Err(anyhow!("vision_config has zero-valued core dims"));
        }
        if !self.head_dim.is_multiple_of(4) {
            // 2-D RoPE splits head_dim into 2 halves and each half into
            // sin/cos pairs, so head_dim must be a multiple of 4.
            return Err(anyhow!(
                "vision head_dim ({}) must be a multiple of 4 for 2-D RoPE",
                self.head_dim
            ));
        }
        if self.num_attention_heads * self.head_dim != self.hidden_size {
            return Err(anyhow!(
                "vision heads×head_dim ({}×{}) != hidden_size ({})",
                self.num_attention_heads,
                self.head_dim,
                self.hidden_size
            ));
        }
        if self.use_clipped_linears {
            // The checkpoint would ship input_min/max buffers we don't read.
            return Err(anyhow!(
                "vision_config.use_clipped_linears=true is not supported"
            ));
        }
        if self.hidden_activation != "gelu_pytorch_tanh" {
            return Err(anyhow!(
                "unsupported vision hidden_activation '{}'",
                self.hidden_activation
            ));
        }
        Ok(())
    }
}
