//! Qwen 3.5 / 3.6 `config.json` parsing.
//!
//! Ungated on purpose. This is pure `serde` + `std::fs` — it has no MLX
//! dependency at all — and keeping it inside `qwen3_5_moe`'s
//! `#[cfg(feature = "mlx-native")] mod imp` meant a config the loader chokes
//! on could only be tested by building the whole GPU stack. Two of the
//! recorded defects (`config-null-moe-fields`, `safetensors-silent-truncation`)
//! live on exactly this surface, so it belongs where a tier-0 sweep can reach
//! it. `imp` re-exports everything, so call sites are untouched.

use anyhow::{Result, anyhow};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

/// Null-tolerant field decoding, shared with the Gemma 4 parser. See
/// [`crate::config_serde::null_as_default`] for why plain `#[serde(default)]`
/// is not enough.
use crate::config_serde::null_as_default;

// ───────────────────────── config.json parsing ─────────────────────────

/// Top-level config.json wrapper for `qwen3_5_moe`.
#[derive(Debug, Clone, Deserialize)]
pub struct NativeModelConfig {
    pub model_type: String,
    #[serde(
        default,
        rename = "eos_token_id",
        deserialize_with = "deserialize_token_ids"
    )]
    pub eos_token_ids: Vec<u32>,
    pub text_config: NativeTextConfig,
    #[serde(default)]
    pub quantization_config: Option<NativeQuantizationConfig>,

    // ── multimodal (image) ──
    // Present on every `Qwen3_5*ForConditionalGeneration` checkpoint. All
    // optional so a text-only conversion still parses.
    #[serde(default)]
    pub vision_config: Option<NativeQwen36VisionConfig>,
    /// Placeholder token whose embedding rows the vision features replace
    /// (`<|image_pad|>`, 248056 on Qwen3.6).
    #[serde(default)]
    pub image_token_id: Option<u32>,
    /// `<|vision_start|>` / `<|vision_end|>` sentinels around each run.
    #[serde(default)]
    pub vision_start_token_id: Option<u32>,
    #[serde(default)]
    pub vision_end_token_id: Option<u32>,
}

/// `text_config` block — all fields the forward path needs.
#[derive(Debug, Clone, Deserialize)]
pub struct NativeTextConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub head_dim: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    #[serde(default = "default_full_attn_interval")]
    pub full_attention_interval: usize,
    pub layer_types: Vec<NativeLayerType>,

    // RoPE
    //
    // NOTE: shipped Qwen3.6 configs carry `rope_theta` only inside
    // `rope_parameters`, so this flat field always falls back to its
    // default — which happens to be the same 10_000_000 the nested block
    // specifies. Left as-is rather than re-plumbed, because changing where
    // theta comes from changes decode numerics for every existing
    // checkpoint; `rope_parameters` below is additive.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    #[serde(default = "default_max_pos_emb")]
    pub max_position_embeddings: usize,
    /// Nested rope block. Present on Qwen3.6; read for its MRoPE fields.
    #[serde(default)]
    pub rope_parameters: Option<NativeRopeParameters>,

    // Linear (delta-net) attention dims
    #[serde(default, deserialize_with = "null_as_default")]
    pub linear_num_value_heads: usize,
    #[serde(default, deserialize_with = "null_as_default")]
    pub linear_num_key_heads: usize,
    #[serde(default, deserialize_with = "null_as_default")]
    pub linear_key_head_dim: usize,
    #[serde(default, deserialize_with = "null_as_default")]
    pub linear_value_head_dim: usize,
    #[serde(default, deserialize_with = "null_as_default")]
    pub linear_conv_kernel_dim: usize,

    // MoE
    #[serde(default, deserialize_with = "null_as_default")]
    pub num_experts: usize,
    #[serde(default, deserialize_with = "null_as_default")]
    pub num_experts_per_tok: usize,
    #[serde(default, deserialize_with = "null_as_default")]
    pub moe_intermediate_size: usize,
    #[serde(default, deserialize_with = "null_as_default")]
    pub shared_expert_intermediate_size: usize,
    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,

    // Dense SwiGLU intermediate dim (Qwen3.6-27B: 17408). Absent from MoE configs
    // (35B-A3B uses moe_intermediate_size + shared_expert_intermediate_size instead).
    #[serde(default, deserialize_with = "null_as_default")]
    pub intermediate_size: usize,

    // Tied embedding → lm_head reuses embed_tokens. Defaults to false for
    // Qwen3.5-MoE (35B); checked at load time.
    #[serde(default)]
    pub tie_word_embeddings: bool,
    // Other unknown fields (`attn_output_gate`, …) are silently dropped by serde
    // — no allow-list needed.
}

/// `text_config.rope_parameters` — the nested rope block Qwen3.6 ships.
///
/// Only the MRoPE fields are consumed today. They are inert for text-only
/// prompts (all three axes carry the same position, so MRoPE degenerates to
/// ordinary 1-D RoPE) and become load-bearing once an image block gives its
/// tokens a constant `t` with an `h`/`w` grid.
#[derive(Debug, Clone, Deserialize)]
pub struct NativeRopeParameters {
    /// Frequency channels allocated to `(t, h, w)`. Sums to
    /// `rope_dim / 2` — `[11, 11, 10]` for Qwen3.6's 64-wide rotary span.
    #[serde(default)]
    pub mrope_section: Option<Vec<usize>>,
    /// Spread the axes across the spectrum instead of assigning contiguous
    /// blocks. See `native_rope::mrope_axis_of_channel`.
    #[serde(default)]
    pub mrope_interleaved: bool,
}

/// Per-token `(t, h, w)` positions for the tokens in one forward pass.
///
/// `None` — the overwhelmingly common case — means "contiguous text
/// positions starting at the cache offset", which is what
/// `mlx::fast::rope`'s scalar `offset` already expresses and what every
/// text-only prompt wants.
///
/// `Shifted(delta)` is still contiguous, just offset — decode *after* an
/// image block, where the axes have realigned but the position counter
/// advanced more slowly than the token count did (an image occupies `h·w`
/// slots yet advances by only `max(h, w)`), so the running position is
/// `cache.offset() + delta` with `delta` negative. Upstream calls this
/// `mrope_position_deltas`. Still fused.
///
/// `Explicit` carries per-token `(t, h, w)` triples. Only the prefill of an
/// image-bearing prompt needs it, and only it pays for the unfused path.
#[derive(Debug, Clone, Copy)]
pub(crate) enum RopePlan<'a> {
    Sequential,
    Shifted(i32),
    Explicit(&'a [[i32; 3]]),
}

/// MLP variant within the qwen3_5 family. Dense (27B) carries `intermediate_size`
/// and zero `num_experts`; MoE (35B-A3B) carries the per-expert sizes and `num_experts > 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlpKind {
    Moe,
    Dense,
}

fn default_full_attn_interval() -> usize {
    4
}
fn default_rope_theta() -> f32 {
    10_000_000.0
}
fn default_partial_rotary_factor() -> f32 {
    0.25
}
fn default_max_pos_emb() -> usize {
    262_144
}
fn default_norm_topk_prob() -> bool {
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeLayerType {
    LinearAttention,
    FullAttention,
}

impl NativeLayerType {
    pub fn is_linear(self) -> bool {
        matches!(self, NativeLayerType::LinearAttention)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct NativeQuantizationConfig {
    pub group_size: usize,
    pub bits: usize,
    pub mode: String,
    // Per-tensor overrides (e.g. mlp.gate, shared_expert_gate at 8-bit).
    #[serde(flatten)]
    pub overrides: BTreeMap<String, NativeQuantizationOverride>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NativeQuantizationOverride {
    pub group_size: usize,
    pub bits: usize,
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

impl NativeModelConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|err| anyhow!("config.json read failed at {}: {err}", path.display()))?;
        serde_json::from_str(&raw)
            .map_err(|err| anyhow!("config.json parse failed at {}: {err}", path.display()))
    }

    /// Validate that this config belongs to the Qwen3.5 family (either MoE or Dense)
    /// and returns which MLP variant it carries. Production load path uses this for
    /// dispatch; existing strict callers can still call [`Self::validate_qwen3_5_moe`].
    pub fn validate_qwen3_5_family(&self) -> Result<MlpKind> {
        if self.model_type != "qwen3_5_moe" && self.model_type != "qwen3_5" {
            return Err(anyhow!(
                "expected model_type='qwen3_5_moe' or 'qwen3_5', got '{}'",
                self.model_type
            ));
        }
        let kind = self.text_config.validate_family()?;
        if let Some(quant) = &self.quantization_config {
            // 35B-A3B-mxfp4 ships `mxfp4`; 27B Dense MLX-4bit ships `affine`.
            // Both at 4-bit, both with non-zero group_size — the per-tensor dispatch
            // (mxfp4 kernel vs affine kernel) is selected later from `mode`.
            let mode_ok = quant.mode == "mxfp4" || quant.mode == "affine" || quant.mode == "nvfp4";
            // mxfp4/nvfp4 are 4-bit E2M1; affine ships 4/6/8-bit (9B Speed =
            // affine 6-bit, 27B Speed = affine 4-bit). Require 4-bit for the
            // E2M1 formats, allow {4,6,8} for affine.
            let bits_ok = if quant.mode == "affine" {
                matches!(quant.bits, 4 | 6 | 8)
            } else {
                quant.bits == 4
            };
            if !mode_ok || !bits_ok || quant.group_size == 0 {
                return Err(anyhow!(
                    "quantization_config must be (mxfp4|nvfp4)/4-bit or affine/{{4,6,8}}-bit with non-zero group, got mode='{}' bits={} group={}",
                    quant.mode,
                    quant.bits,
                    quant.group_size
                ));
            }
        }
        Ok(kind)
    }

    /// Strict MoE-only contract. Wraps [`Self::validate_qwen3_5_family`] and asserts
    /// the result is MoE — preserves the original error surface that downstream
    /// MoE-only call sites and existing tests depend on.
    pub fn validate_qwen3_5_moe(&self) -> Result<()> {
        if self.model_type != "qwen3_5_moe" {
            return Err(anyhow!(
                "expected model_type='qwen3_5_moe', got '{}'",
                self.model_type
            ));
        }
        self.text_config.validate()?;
        if let Some(quant) = &self.quantization_config {
            if (quant.mode != "mxfp4" && quant.mode != "nvfp4")
                || quant.bits != 4
                || quant.group_size == 0
            {
                return Err(anyhow!(
                    "quantization_config must default to (mxfp4|nvfp4)/4-bit/non-zero group, got mode='{}' bits={} group={}",
                    quant.mode,
                    quant.bits,
                    quant.group_size
                ));
            }
        }
        Ok(())
    }
}

impl NativeTextConfig {
    /// Strict MoE text-config contract (legacy behavior).
    pub fn validate(&self) -> Result<()> {
        if self.model_type != "qwen3_5_moe_text" {
            return Err(anyhow!(
                "expected text_config.model_type='qwen3_5_moe_text', got '{}'",
                self.model_type
            ));
        }
        self.validate_core_dims()?;
        if self.num_experts > 0 {
            if self.num_experts_per_tok == 0 || self.num_experts_per_tok > self.num_experts {
                return Err(anyhow!(
                    "num_experts_per_tok {} invalid against num_experts {}",
                    self.num_experts_per_tok,
                    self.num_experts
                ));
            }
            if self.moe_intermediate_size == 0 {
                return Err(anyhow!("moe_intermediate_size must be non-zero with MoE"));
            }
        }
        Ok(())
    }

    /// Family validator: accepts both MoE (`qwen3_5_moe_text`) and Dense
    /// (`qwen3_5_text`) text configs and returns which MLP variant the
    /// per-layer dispatch should use.
    pub fn validate_family(&self) -> Result<MlpKind> {
        if self.model_type != "qwen3_5_moe_text" && self.model_type != "qwen3_5_text" {
            return Err(anyhow!(
                "expected text_config.model_type='qwen3_5_moe_text' or 'qwen3_5_text', got '{}'",
                self.model_type
            ));
        }
        self.validate_core_dims()?;
        let kind = self.mlp_kind();
        match kind {
            MlpKind::Moe => {
                if self.num_experts_per_tok == 0 || self.num_experts_per_tok > self.num_experts {
                    return Err(anyhow!(
                        "num_experts_per_tok {} invalid against num_experts {}",
                        self.num_experts_per_tok,
                        self.num_experts
                    ));
                }
                if self.moe_intermediate_size == 0 {
                    return Err(anyhow!("moe_intermediate_size must be non-zero with MoE"));
                }
            }
            MlpKind::Dense => {
                if self.intermediate_size == 0 {
                    return Err(anyhow!(
                        "intermediate_size must be non-zero for dense qwen3_5 (e.g. 17408 for 27B)"
                    ));
                }
            }
        }
        Ok(kind)
    }

    /// Infer MLP variant from which family of size fields is populated.
    pub fn mlp_kind(&self) -> MlpKind {
        if self.num_experts > 0 {
            MlpKind::Moe
        } else {
            MlpKind::Dense
        }
    }

    fn validate_core_dims(&self) -> Result<()> {
        if self.hidden_size == 0
            || self.head_dim == 0
            || self.vocab_size == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.num_hidden_layers == 0
        {
            return Err(anyhow!("text_config has zero-sized core dims"));
        }
        if self.layer_types.len() != self.num_hidden_layers {
            return Err(anyhow!(
                "layer_types length {} != num_hidden_layers {}",
                self.layer_types.len(),
                self.num_hidden_layers
            ));
        }
        // RoPE rotary span — partial_rotary_factor == 0.25 → rope_dim = 64 for
        // head_dim=256. Must be even (RoPE rotates pairs).
        let rope_dim = self.rope_dim();
        if rope_dim == 0 || rope_dim % 2 != 0 {
            return Err(anyhow!(
                "rope_dim {} (= head_dim {} × partial_rotary_factor {}) must be a positive even number",
                rope_dim,
                self.head_dim,
                self.partial_rotary_factor
            ));
        }
        Ok(())
    }

    pub fn rope_dim(&self) -> usize {
        ((self.head_dim as f32) * self.partial_rotary_factor) as usize
    }

    /// `(sections, interleaved)` for MRoPE, when the config declares a
    /// usable `mrope_section`.
    ///
    /// `None` means "no MRoPE" and the caller keeps the fused scalar-offset
    /// rope. A section list that does not have three entries summing to
    /// `rope_dim / 2` is also `None` rather than an error: it would only
    /// matter for image input, which refuses to run without this anyway,
    /// and a text-only deploy should not fail to load over it.
    pub fn mrope(&self) -> Option<([usize; 3], bool)> {
        let params = self.rope_parameters.as_ref()?;
        let s = params.mrope_section.as_ref()?;
        if s.len() != 3 || s.iter().sum::<usize>() != self.rope_dim() / 2 {
            return None;
        }
        Some(([s[0], s[1], s[2]], params.mrope_interleaved))
    }

    pub fn is_linear_per_layer(&self) -> Vec<bool> {
        self.layer_types.iter().map(|t| t.is_linear()).collect()
    }

    /// Convenience: index of the first full-attention layer (used by
    /// `Qwen3_5TextModel.fa_idx` for mask creation).
    pub fn first_full_attn_layer(&self) -> Option<usize> {
        self.layer_types
            .iter()
            .position(|t| matches!(t, NativeLayerType::FullAttention))
    }
}

// ─────────────────── vision_config block (Qwen 3.6) ───────────────────

/// `vision_config` block of Qwen 3.6's config.json.
#[derive(Debug, Clone, Deserialize)]
pub struct NativeQwen36VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub num_position_embeddings: usize,
    pub out_hidden_size: usize,
    #[serde(default)]
    pub deepstack_visual_indexes: Vec<usize>,
    #[serde(default = "default_vision_activation")]
    pub hidden_act: String,
}

fn default_vision_activation() -> String {
    "gelu_pytorch_tanh".to_string()
}

impl NativeQwen36VisionConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// Patches folded into one language-model token (`merge²`).
    pub fn merge_unit(&self) -> usize {
        self.spatial_merge_size * self.spatial_merge_size
    }

    /// Side length of the learned position grid (`√num_position_embeddings`).
    pub fn grid_per_side(&self) -> usize {
        (self.num_position_embeddings as f64).sqrt() as usize
    }

    pub fn validate(&self) -> Result<()> {
        if self.depth == 0
            || self.hidden_size == 0
            || self.num_heads == 0
            || self.patch_size == 0
            || self.spatial_merge_size == 0
            || self.temporal_patch_size == 0
        {
            return Err(anyhow!("vision_config has zero-valued core dims"));
        }
        if self.hidden_size % self.num_heads != 0 {
            return Err(anyhow!(
                "vision hidden_size ({}) is not divisible by num_heads ({})",
                self.hidden_size,
                self.num_heads
            ));
        }
        // The rotary table is built over `head_dim / 2` and split in half
        // again for the (h, w) pair, so head_dim must be a multiple of 4.
        if self.head_dim() % 4 != 0 {
            return Err(anyhow!(
                "vision head_dim ({}) must be a multiple of 4 for 2-D RoPE",
                self.head_dim()
            ));
        }
        let side = self.grid_per_side();
        if side * side != self.num_position_embeddings {
            return Err(anyhow!(
                "num_position_embeddings ({}) is not a perfect square",
                self.num_position_embeddings
            ));
        }
        if !self.deepstack_visual_indexes.is_empty() {
            // Deepstack injects intermediate ViT features into the first N
            // decoder layers. The checkpoints we serve ship an empty list;
            // supporting it means plumbing extra tensors into the text
            // stack, which is a separate change.
            return Err(anyhow!(
                "vision_config.deepstack_visual_indexes {:?} is not supported",
                self.deepstack_visual_indexes
            ));
        }
        if self.hidden_act != "gelu_pytorch_tanh" {
            return Err(anyhow!(
                "unsupported vision hidden_act '{}'",
                self.hidden_act
            ));
        }
        Ok(())
    }
}
