//! Native MLX Qwen3.5-MoE model assembly.
//!
//! This module owns the **whole-model glue** between the Phase 3b validated
//! primitives (embedding, RMSNorm, RoPE, SDPA, Conv1d, SSM kernel, router,
//! lm_head) and the Phase 3c per-sequence cache (`NativePromptCache`).
//!
//! ## Landed
//!
//! * `NativeModelConfig` / `NativeTextConfig` — full config.json parser.
//! * `NativeWeights` — multi-shard safetensors loader + `sanitize()` mirror
//!   of `qwen3_5_moe.Model.sanitize` (drop vision tower, rewrite prefixes).
//! * `NativeQwen3_5MoeModel::load / make_cache / eos_tokens / vocab_size /
//!   forward / argmax_last_token`.
//! * **Phase 3d.2** — `layer_full_attn_forward()` body: q/k/v MXFP4 matmul +
//!   q_norm/k_norm RMSNorm + GPT-NeoX RoPE (partial rotary, offset from KV
//!   cache) + KV-cache update + GQA SDPA + sigmoid gate × output + o_proj.
//!   Mirrors `qwen3_next.Qwen3NextAttention.__call__`.
//! * **Phase 3d.3** — `layer_linear_attn_forward()` body: in_proj_qkv /
//!   in_proj_z / in_proj_b / in_proj_a 4-way MXFP4 matmuls + reshape
//!   + Conv1d depthwise (kernel=4) over the persistent conv_state cache slot
//!   + SiLU + per-head q/k RMSNorm with `inv_scale^{1,2}` rescaling +
//!   `gated_delta_step_kernel` Metal SSM update + RMSNormGated output gate +
//!   out_proj MXFP4 matmul. Mirrors `qwen3_next.Qwen3NextGatedDeltaNet.__call__`.
//! * **Phase 3d.4** — `layer_moe_forward()` wiring of `Qwen3NextSparseMoeBlock`
//!   (8-bit affine `gate` → softmax(precise) → top-k argpartition → mxfp4
//!   `switch_mlp` swiglu via `gather_qmm_with_mode` (mirroring `SwitchGLU`'s
//!   `do_sort` branch) → `(y * scores).sum(-2)` + shared expert gated by
//!   `sigmoid(shared_expert_gate(x))`). `forward()` now runs the full
//!   `input_layernorm → attn → +residual → post_attention_layernorm → MoE
//!   → +residual` decoder block per layer, then applies the final
//!   `model.norm` and `lm_head` projection (with `tie_word_embeddings`
//!   support) so it returns real `[B, L, vocab]` logits.
//!
//! ## Pending
//!
//! * **3d.5** — end-to-end parity gate against PyO3 golden transcript.
//!
//! See `.ai/memory/active/mlx-rs-native-port/CONTEXT.md` Session 25+ for the
//! incremental roll-out plan.

// FineTimings + per-layer-kind bumpers moved to `crate::native_runtime` so
// that future model files (Llama, Mistral, …) can share the instrumentation
// primitives without depending on Qwen3.5-MoE assembly.

#[cfg(feature = "mlx-native")]
#[allow(dead_code)] // accessor APIs (config/weights/keys/len/is_empty/first_full_attn_layer) reserved for future inspection / debugging consumers
mod imp {
    use anyhow::{Context, Result, anyhow};
    use mlx_rs::Array;
    use serde::Deserialize;
    use std::collections::{BTreeMap, HashMap};
    use std::ffi::CStr;
    use std::path::Path;
    use std::time::Instant;

    pub(crate) use crate::native_runtime::FineTimings;
    use crate::native_runtime::{
        bump_linear_conv_ms, bump_linear_in_proj_ms, bump_linear_norm_qk_ms,
        bump_linear_out_proj_ms, bump_linear_ssm_ms, drain_layer_sub_timings, fine_timing_active,
        reset_layer_sub_timings, store_fine_timings,
    };

    use crate::native_attention::sdpa;
    use crate::native_cache::{
        NativeArraysCache, NativeKvCache, NativeLayerCache, NativePromptCache,
        NativeRotatingKvCacheTurboQuant,
    };
    use crate::native_conv1d::conv1d;
    use crate::native_embedding::quantized_embedding_lookup_with_mode;
    use crate::native_kernels::{sigmoid_mul_fuse_enabled, sigmoid_mul_fused};
    use crate::native_moe::{
        AffineGateWeights, MoeBlockWeights, SharedMlpWeights, SwitchExpertWeights,
        SwitchGluWeights, moe_block_forward,
    };
    use crate::native_norm::rms_norm;
    use crate::native_quant::{MODE_AFFINE, MODE_MXFP4, MODE_NVFP4, quantized_matmul_with_mode};
    use crate::native_rope::{precompute_rope_freqs, rope, rope_with_freqs};
    use crate::native_snapshot::PromptCacheSnapshot;
    use crate::native_ssm::{compute_g, gated_delta_step_kernel, rms_norm_gated};
    use crate::qwen3_5_mtp::{Qwen35MtpBlock, Qwen35MtpLinear};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};

    pub(crate) fn sigmoid_mul_fuse_enabled_pub() -> bool {
        sigmoid_mul_fuse_enabled()
    }

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
        #[serde(default = "default_rope_theta")]
        pub rope_theta: f32,
        #[serde(default = "default_partial_rotary_factor")]
        pub partial_rotary_factor: f32,
        #[serde(default = "default_max_pos_emb")]
        pub max_position_embeddings: usize,

        // Linear (delta-net) attention dims
        #[serde(default)]
        pub linear_num_value_heads: usize,
        #[serde(default)]
        pub linear_num_key_heads: usize,
        #[serde(default)]
        pub linear_key_head_dim: usize,
        #[serde(default)]
        pub linear_value_head_dim: usize,
        #[serde(default)]
        pub linear_conv_kernel_dim: usize,

        // MoE
        #[serde(default)]
        pub num_experts: usize,
        #[serde(default)]
        pub num_experts_per_tok: usize,
        #[serde(default)]
        pub moe_intermediate_size: usize,
        #[serde(default)]
        pub shared_expert_intermediate_size: usize,
        #[serde(default = "default_norm_topk_prob")]
        pub norm_topk_prob: bool,

        // Dense SwiGLU intermediate dim (Qwen3.6-27B: 17408). Absent from MoE configs
        // (35B-A3B uses moe_intermediate_size + shared_expert_intermediate_size instead).
        #[serde(default)]
        pub intermediate_size: usize,

        // Tied embedding → lm_head reuses embed_tokens. Defaults to false for
        // Qwen3.5-MoE (35B); checked at load time.
        #[serde(default)]
        pub tie_word_embeddings: bool,
        // Other unknown fields (`attn_output_gate`, …) are silently dropped by serde
        // — no allow-list needed.
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
                let mode_ok =
                    quant.mode == "mxfp4" || quant.mode == "affine" || quant.mode == "nvfp4";
                if !mode_ok || quant.bits != 4 || quant.group_size == 0 {
                    return Err(anyhow!(
                        "quantization_config must default to (mxfp4|affine|nvfp4)/4-bit/non-zero group, got mode='{}' bits={} group={}",
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
                    if self.num_experts_per_tok == 0 || self.num_experts_per_tok > self.num_experts
                    {
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

    // ───────────────────────── safetensors weights ─────────────────────────

    /// Multi-shard safetensors weight bag, keyed by tensor name.
    /// Mirrors mlx_lm's `weights = mx.load(path)` dict but covers all shards
    /// in a model directory.
    pub struct NativeWeights {
        tensors: HashMap<String, Array>,
    }

    impl NativeWeights {
        /// Walk all `*.safetensors` files in `dir`, load each, and merge into
        /// a single keyed map. Mirrors mlx_lm's `_get_weights(model_path)`.
        pub fn load_dir(dir: &Path) -> Result<Self> {
            let mut shard_paths = std::fs::read_dir(dir)
                .with_context(|| format!("read_dir({}) failed", dir.display()))?
                .filter_map(|entry| entry.ok().map(|e| e.path()))
                .filter(|p| p.is_file() && p.extension().is_some_and(|ext| ext == "safetensors"))
                .collect::<Vec<_>>();
            shard_paths.sort();
            if shard_paths.is_empty() {
                return Err(anyhow!("no *.safetensors found under {}", dir.display()));
            }
            let mut tensors: HashMap<String, Array> = HashMap::new();
            for shard in shard_paths {
                let map = Array::load_safetensors(&shard).map_err(|err| {
                    anyhow!("load_safetensors({}) failed: {err}", shard.display())
                })?;
                tensors.reserve(map.len());
                for (k, v) in map {
                    if tensors.insert(k.clone(), v).is_some() {
                        return Err(anyhow!(
                            "duplicate weight key `{k}` across safetensors shards in {}",
                            dir.display()
                        ));
                    }
                }
            }
            Ok(Self { tensors })
        }

        /// In-place key normalization mirroring `qwen3_5_moe.Model.sanitize`:
        /// drops vision-tower keys and rewrites prefixes so all surviving
        /// keys are rooted at `language_model.model.*` (or `language_model.lm_head.*`).
        ///
        /// We do **not** split per-expert `experts.gate_up_proj` here — the
        /// production MXFP4 checkpoint stores already-stacked `switch_mlp.*`
        /// tensors. If we ever encounter the legacy layout we'll surface a
        /// targeted error so the caller can run the upstream sanitizer.
        pub fn sanitize(&mut self) -> Result<()> {
            let raw = std::mem::take(&mut self.tensors);
            let mut out: HashMap<String, Array> = HashMap::with_capacity(raw.len());
            for (key, value) in raw {
                if key.starts_with("vision_tower") || key.starts_with("model.visual") {
                    continue;
                }
                let new_key = if let Some(rest) = key.strip_prefix("model.language_model") {
                    format!("language_model.model{rest}")
                } else if key.starts_with("language_model.") {
                    key
                } else {
                    format!("language_model.{key}")
                };
                if out.insert(new_key.clone(), value).is_some() {
                    return Err(anyhow!("sanitize: duplicate normalized key `{new_key}`"));
                }
            }
            // Detect the unsanitized legacy layout and surface a clear error.
            let legacy = out
                .keys()
                .find(|k| k.contains(".mlp.experts.gate_up_proj"))
                .cloned();
            if let Some(k) = legacy {
                return Err(anyhow!(
                    "weights still in legacy MoE layout (found `{k}`); convert with mlx_lm sanitize first"
                ));
            }
            self.tensors = out;
            Ok(())
        }

        pub fn get(&self, name: &str) -> Option<&Array> {
            self.tensors.get(name)
        }

        pub fn require(&self, name: &str) -> Result<&Array> {
            self.tensors
                .get(name)
                .ok_or_else(|| anyhow!("required weight `{name}` not found in safetensors"))
        }

        pub fn len(&self) -> usize {
            self.tensors.len()
        }

        pub fn is_empty(&self) -> bool {
            self.tensors.is_empty()
        }

        pub fn keys(&self) -> impl Iterator<Item = &String> {
            self.tensors.keys()
        }
    }

    // ──────────────────── pre-resolved per-layer weights ─────────────────
    //
    // D1 lever — eliminate per-call HashMap+format!() lookup overhead in the
    // hot `linear()` path. `NativeWeights` is keyed by tensor path string;
    // resolving `{base}.weight` / `{base}.scales` / `{base}.biases` at every
    // forward() pass burns ~3 String allocations + 3 HashMap lookups + a
    // config override traversal per matmul. With 120 in_proj calls and 40
    // routing/o_proj/etc. calls per step on Qwen3.5-MoE, that overhead is the
    // dominant slice of the Native↔PyO3 gap (see notes/native_pyo3_root_cause).
    //
    // We resolve every reachable weight once at `load()` time, clone the
    // `Array` handle (mlx-c refcount-bump, no data copy), and store the
    // tuples in per-layer structs. The hot path becomes a direct field
    // access + a single `quantized_matmul_with_mode` call, mirroring how
    // Python's `nn.Linear.__call__` carries pre-bound module attributes.

    /// One MXFP4/affine quantized linear projection — pre-resolved.
    pub struct ResolvedLinear {
        pub weight: Array,
        pub scales: Array,
        pub biases: Option<Array>,
        pub group_size: i32,
        pub bits: i32,
        pub mode: &'static CStr,
    }

    /// Pre-resolved weights for one linear-attention (GatedDeltaNet) layer.
    pub struct LinearAttnLayerWeights {
        pub input_layernorm: Array,
        pub post_attention_layernorm: Array,
        pub in_proj_qkv: ResolvedLinear,
        pub in_proj_z: ResolvedLinear,
        pub in_proj_b: ResolvedLinear,
        pub in_proj_a: ResolvedLinear,
        pub conv1d_weight: Array,
        pub a_log: Array,
        pub dt_bias: Array,
        pub norm_weight: Array,
        pub out_proj: ResolvedLinear,
    }

    /// Pre-resolved weights for one full-attention layer.
    pub struct FullAttnLayerWeights {
        pub input_layernorm: Array,
        pub post_attention_layernorm: Array,
        pub q_proj: ResolvedLinear,
        pub k_proj: ResolvedLinear,
        pub v_proj: ResolvedLinear,
        pub o_proj: ResolvedLinear,
        pub q_norm_weight: Array,
        pub k_norm_weight: Array,
    }

    pub enum AttnLayerWeights {
        Linear(LinearAttnLayerWeights),
        Full(FullAttnLayerWeights),
    }

    /// Pre-resolved MoE weights — same shape as the per-call
    /// `MoeBlockWeights<'_>` view but owns the `Array` handles.
    /// `moe_view()` returns the borrowed view that `moe_block_forward`
    /// expects.
    pub struct ResolvedMoeWeights {
        // gate (8-bit affine)
        pub gate_w: Array,
        pub gate_s: Array,
        pub gate_b: Array,
        pub gate_gs: i32,
        pub gate_bits: i32,
        // switch_mlp (mxfp4 by default)
        pub sw_gate_w: Array,
        pub sw_gate_s: Array,
        pub sw_up_w: Array,
        pub sw_up_s: Array,
        pub sw_down_w: Array,
        pub sw_down_s: Array,
        pub sw_gs: i32,
        pub sw_bits: i32,
        pub sw_mode: &'static CStr,
        // shared_expert (mxfp4 by default)
        pub se_gate_w: Array,
        pub se_gate_s: Array,
        pub se_up_w: Array,
        pub se_up_s: Array,
        pub se_down_w: Array,
        pub se_down_s: Array,
        pub se_gs: i32,
        pub se_bits: i32,
        pub se_mode: &'static CStr,
        // shared_expert_gate (8-bit affine, output dim = 1)
        pub sg_w: Array,
        pub sg_s: Array,
        pub sg_b: Array,
        pub sg_gs: i32,
        pub sg_bits: i32,
        // routing meta
        pub top_k: i32,
        pub norm_topk_prob: bool,
    }

    impl ResolvedMoeWeights {
        pub fn view(&self) -> MoeBlockWeights<'_> {
            MoeBlockWeights {
                gate: AffineGateWeights {
                    weight: &self.gate_w,
                    scales: &self.gate_s,
                    biases: &self.gate_b,
                    group_size: self.gate_gs,
                    bits: self.gate_bits,
                },
                switch_glu: SwitchGluWeights {
                    gate_proj: SwitchExpertWeights {
                        weight: &self.sw_gate_w,
                        scales: &self.sw_gate_s,
                        biases: None,
                    },
                    up_proj: SwitchExpertWeights {
                        weight: &self.sw_up_w,
                        scales: &self.sw_up_s,
                        biases: None,
                    },
                    down_proj: SwitchExpertWeights {
                        weight: &self.sw_down_w,
                        scales: &self.sw_down_s,
                        biases: None,
                    },
                    group_size: self.sw_gs,
                    bits: self.sw_bits,
                    mode: self.sw_mode,
                },
                shared_expert: SharedMlpWeights {
                    gate_proj_w: &self.se_gate_w,
                    gate_proj_s: &self.se_gate_s,
                    up_proj_w: &self.se_up_w,
                    up_proj_s: &self.se_up_s,
                    down_proj_w: &self.se_down_w,
                    down_proj_s: &self.se_down_s,
                    group_size: self.se_gs,
                    bits: self.se_bits,
                    mode: self.se_mode,
                },
                shared_gate: AffineGateWeights {
                    weight: &self.sg_w,
                    scales: &self.sg_s,
                    biases: &self.sg_b,
                    group_size: self.sg_gs,
                    bits: self.sg_bits,
                },
                top_k: self.top_k,
                norm_topk_prob: self.norm_topk_prob,
            }
        }
    }

    pub struct LayerWeights {
        pub attn: AttnLayerWeights,
        pub moe: ResolvedMoeWeights,
    }

    /// Resolve `(group_size, bits, mode)` for a quantized layer — same logic
    /// as `quant_params_with_mode` but free-standing for use during load().
    fn quant_params_resolve(config: &NativeModelConfig, base: &str) -> (i32, i32, &'static CStr) {
        let cfg = config.quantization_config.as_ref();
        if let Some(q) = cfg
            && let Some(o) = q.overrides.get(base)
        {
            return (o.group_size as i32, o.bits as i32, MODE_AFFINE);
        }
        cfg.map(|q| {
            // Default (non-override) tensors follow the global mode. mxfp4 and
            // nvfp4 are both E2M1 4-bit formats differing only in block-scale
            // precision (E8M0 pow2 group-32 vs E4M3 group-16); the FFI selects
            // the matching Metal kernel from this CStr. affine handled for the
            // 27B-dense MLX-4bit checkpoint that ships a global affine mode.
            let mode = match q.mode.as_str() {
                "nvfp4" => MODE_NVFP4,
                "affine" => MODE_AFFINE,
                _ => MODE_MXFP4,
            };
            (q.group_size as i32, q.bits as i32, mode)
        })
        .unwrap_or((32, 4, MODE_MXFP4))
    }

    /// Build a `ResolvedLinear` by cloning (refcount-bump) the underlying
    /// MXFP4/affine arrays out of the sanitized weight bag.
    fn resolve_linear(
        weights: &NativeWeights,
        config: &NativeModelConfig,
        base: &str,
    ) -> Result<ResolvedLinear> {
        let weight = weights
            .require(&format!("{base}.weight"))
            .with_context(|| format!("resolve_linear({base}): missing .weight"))?
            .clone();
        let scales = weights
            .require(&format!("{base}.scales"))
            .with_context(|| format!("resolve_linear({base}): missing .scales"))?
            .clone();
        let biases = weights.get(&format!("{base}.biases")).cloned();
        let (group_size, bits, mode) = quant_params_resolve(config, base);
        Ok(ResolvedLinear {
            weight,
            scales,
            biases,
            group_size,
            bits,
            mode,
        })
    }

    fn require_clone(weights: &NativeWeights, name: &str) -> Result<Array> {
        Ok(weights.require(name)?.clone())
    }

    fn resolve_full_attn_layer(
        weights: &NativeWeights,
        config: &NativeModelConfig,
        layer_idx: usize,
    ) -> Result<FullAttnLayerWeights> {
        let layer_base = format!("language_model.model.layers.{layer_idx}");
        let prefix = format!("{layer_base}.self_attn");
        Ok(FullAttnLayerWeights {
            input_layernorm: require_clone(
                weights,
                &format!("{layer_base}.input_layernorm.weight"),
            )?,
            post_attention_layernorm: require_clone(
                weights,
                &format!("{layer_base}.post_attention_layernorm.weight"),
            )?,
            q_proj: resolve_linear(weights, config, &format!("{prefix}.q_proj"))?,
            k_proj: resolve_linear(weights, config, &format!("{prefix}.k_proj"))?,
            v_proj: resolve_linear(weights, config, &format!("{prefix}.v_proj"))?,
            o_proj: resolve_linear(weights, config, &format!("{prefix}.o_proj"))?,
            q_norm_weight: require_clone(weights, &format!("{prefix}.q_norm.weight"))?,
            k_norm_weight: require_clone(weights, &format!("{prefix}.k_norm.weight"))?,
        })
    }

    fn resolve_linear_attn_layer(
        weights: &NativeWeights,
        config: &NativeModelConfig,
        layer_idx: usize,
    ) -> Result<LinearAttnLayerWeights> {
        let layer_base = format!("language_model.model.layers.{layer_idx}");
        let prefix = format!("{layer_base}.linear_attn");
        Ok(LinearAttnLayerWeights {
            input_layernorm: require_clone(
                weights,
                &format!("{layer_base}.input_layernorm.weight"),
            )?,
            post_attention_layernorm: require_clone(
                weights,
                &format!("{layer_base}.post_attention_layernorm.weight"),
            )?,
            in_proj_qkv: resolve_linear(weights, config, &format!("{prefix}.in_proj_qkv"))?,
            in_proj_z: resolve_linear(weights, config, &format!("{prefix}.in_proj_z"))?,
            in_proj_b: resolve_linear(weights, config, &format!("{prefix}.in_proj_b"))?,
            in_proj_a: resolve_linear(weights, config, &format!("{prefix}.in_proj_a"))?,
            conv1d_weight: require_clone(weights, &format!("{prefix}.conv1d.weight"))?,
            a_log: require_clone(weights, &format!("{prefix}.A_log"))?,
            dt_bias: require_clone(weights, &format!("{prefix}.dt_bias"))?,
            norm_weight: require_clone(weights, &format!("{prefix}.norm.weight"))?,
            out_proj: resolve_linear(weights, config, &format!("{prefix}.out_proj"))?,
        })
    }

    fn resolve_moe_layer(
        weights: &NativeWeights,
        config: &NativeModelConfig,
        layer_idx: usize,
    ) -> Result<ResolvedMoeWeights> {
        let cfg = &config.text_config;
        let prefix = format!("language_model.model.layers.{layer_idx}.mlp");

        let (gate_gs, gate_bits, _) = quant_params_resolve(config, &format!("{prefix}.gate"));
        let gate_w = require_clone(weights, &format!("{prefix}.gate.weight"))?;
        let gate_s = require_clone(weights, &format!("{prefix}.gate.scales"))?;
        let gate_b = require_clone(weights, &format!("{prefix}.gate.biases"))?;

        let (sg_gs, sg_bits, _) =
            quant_params_resolve(config, &format!("{prefix}.shared_expert_gate"));
        let sg_w = require_clone(weights, &format!("{prefix}.shared_expert_gate.weight"))?;
        let sg_s = require_clone(weights, &format!("{prefix}.shared_expert_gate.scales"))?;
        let sg_b = require_clone(weights, &format!("{prefix}.shared_expert_gate.biases"))?;

        let (sw_gs, sw_bits, sw_mode) =
            quant_params_resolve(config, &format!("{prefix}.switch_mlp.gate_proj"));
        let sw_gate_w = require_clone(weights, &format!("{prefix}.switch_mlp.gate_proj.weight"))?;
        let sw_gate_s = require_clone(weights, &format!("{prefix}.switch_mlp.gate_proj.scales"))?;
        let sw_up_w = require_clone(weights, &format!("{prefix}.switch_mlp.up_proj.weight"))?;
        let sw_up_s = require_clone(weights, &format!("{prefix}.switch_mlp.up_proj.scales"))?;
        let sw_down_w = require_clone(weights, &format!("{prefix}.switch_mlp.down_proj.weight"))?;
        let sw_down_s = require_clone(weights, &format!("{prefix}.switch_mlp.down_proj.scales"))?;

        let (se_gs, se_bits, se_mode) =
            quant_params_resolve(config, &format!("{prefix}.shared_expert.gate_proj"));
        let se_gate_w =
            require_clone(weights, &format!("{prefix}.shared_expert.gate_proj.weight"))?;
        let se_gate_s =
            require_clone(weights, &format!("{prefix}.shared_expert.gate_proj.scales"))?;
        let se_up_w = require_clone(weights, &format!("{prefix}.shared_expert.up_proj.weight"))?;
        let se_up_s = require_clone(weights, &format!("{prefix}.shared_expert.up_proj.scales"))?;
        let se_down_w =
            require_clone(weights, &format!("{prefix}.shared_expert.down_proj.weight"))?;
        let se_down_s =
            require_clone(weights, &format!("{prefix}.shared_expert.down_proj.scales"))?;

        Ok(ResolvedMoeWeights {
            gate_w,
            gate_s,
            gate_b,
            gate_gs,
            gate_bits,
            sw_gate_w,
            sw_gate_s,
            sw_up_w,
            sw_up_s,
            sw_down_w,
            sw_down_s,
            sw_gs,
            sw_bits,
            sw_mode,
            se_gate_w,
            se_gate_s,
            se_up_w,
            se_up_s,
            se_down_w,
            se_down_s,
            se_gs,
            se_bits,
            se_mode,
            sg_w,
            sg_s,
            sg_b,
            sg_gs,
            sg_bits,
            top_k: cfg.num_experts_per_tok as i32,
            norm_topk_prob: cfg.norm_topk_prob,
        })
    }

    // ───────────────────────── model handle ─────────────────────────

    /// Fully loaded Qwen3.5-MoE model handle. Holds the validated config and
    /// the sanitized weight bag. Stateless w.r.t. inference — per-sequence
    /// state lives in [`NativePromptCache`].
    /// Layer-A wrapper-alloc reduction (2026-05-03): per-model cache of
    /// constants that the linear-attn forward path was creating per-call,
    /// per-layer, per-step. With 30 linear-attn layers + ~120 quant calls
    /// per step, the per-step heat budget on `MetalAllocator::malloc` and
    /// `BufferCache::release_cached_buffers` was dominant per the prior
    /// session's symbol diff. These four fields short-circuit:
    ///   * `ones_w_dn`     — rms_norm weight ones([dn]), 30 callsites/step
    ///   * `scale_q`       — Array::from_f32(inv_scale^2), 30 callsites/step
    ///   * `scale_k`       — Array::from_f32(inv_scale), 30 callsites/step
    ///   * `keep_indices_decode` — Array::from_slice for s=1 decode regime,
    ///     30 callsites/step (prefill keeps dynamic path since `s` varies)
    /// Total: ~120 allocs/step → 0 (cached refs).
    ///
    /// **WASH at 35B server long-prompt n=5 A/B (2026-05-03)**: A_off
    /// 169.71 vs B_reuse_on 168.61 tok/s = Δ -0.6% σ -1.35 (within noise).
    /// 32-step PyO3 golden bit-identical PASS both gate states. Apple's
    /// allocator handles ~120 small allocs/step (≈ 2.9 KB) via TLS thread
    /// caches, putting the wallclock cost below noise. The 11.6pp Native
    /// vs PyO3 gap is in LARGER allocations (KV concat, conv_state buffer,
    /// hidden state intermediates) that this lever does not touch.
    /// Infra preserved as **opt-in** for future re-test if the allocator
    /// pattern shifts.
    struct LinearAttnConstants {
        ones_w_dn: Array,
        scale_q: Array,
        scale_k: Array,
        /// Pre-scaled rms_norm weights: `ones([dn]) * scale_q` (resp. scale_k).
        /// Folds the post-rms_norm `multiply(scale_q)` into rms_norm's weight
        /// argument — bit-identical (`(x/rms) * (ones*s) == ((x/rms)*ones) * s`)
        /// — saves 2 multiply ops per linear-attn layer × 21 layers = 42 ops/step.
        /// Identified via op-by-op forward divergence audit 2026-05-11.
        scaled_w_q_dn: Array,
        scaled_w_k_dn: Array,
        keep_indices_decode: Array,
    }

    impl LinearAttnConstants {
        fn from_config(cfg: &NativeTextConfig) -> Result<Self> {
            let dn = cfg.linear_key_head_dim as i32;
            let kernel_size = cfg.linear_conv_kernel_dim as i32;
            let n_keep = kernel_size - 1;
            if dn <= 0 || n_keep <= 0 {
                return Err(anyhow!(
                    "LinearAttnConstants: invalid dims dn={dn}, n_keep={n_keep}"
                ));
            }
            let inv_scale = (dn as f32).powf(-0.5);
            let ones_w_dn = mlx_rs::ops::ones::<f32>(&[dn])
                .context("LinearAttnConstants: ones([dn]) failed")?;
            let scale_q = Array::from_f32(inv_scale * inv_scale);
            let scale_k = Array::from_f32(inv_scale);
            // Pre-multiply ones × scale to absorb the scale into rms_norm weight.
            // Eval immediately so the constants materialize once and the multiply
            // op does NOT count toward decode_step op count.
            let scaled_w_q_dn = mlx_rs::ops::multiply(&ones_w_dn, &scale_q)
                .context("LinearAttnConstants: ones × scale_q failed")?;
            scaled_w_q_dn
                .eval()
                .context("LinearAttnConstants: eval(scaled_w_q) failed")?;
            let scaled_w_k_dn = mlx_rs::ops::multiply(&ones_w_dn, &scale_k)
                .context("LinearAttnConstants: ones × scale_k failed")?;
            scaled_w_k_dn
                .eval()
                .context("LinearAttnConstants: eval(scaled_w_k) failed")?;
            // Decode regime (s=1, the dominant per-step path):
            // keep_indices = (s..s+n_keep).collect() = [1, 2, ..., n_keep].
            // Prefill (s>>1) needs a dynamic path; this cache only covers
            // the decode hot path.
            let keep_indices: Vec<i32> = (1..=n_keep).collect();
            let keep_indices_decode = Array::from_slice(&keep_indices, &[n_keep]);
            Ok(Self {
                ones_w_dn,
                scale_q,
                scale_k,
                scaled_w_q_dn,
                scaled_w_k_dn,
                keep_indices_decode,
            })
        }
    }

    /// Returns true when wrapper-alloc reduction lever is on.
    /// **Default ON 2026-05-11** — required by linear_attn_scale_fuse (default ON);
    /// pairs with that lever to produce the -0.315ms WIN at t=-3.33σ.
    /// Cached LinearAttnConstants (ones_w + scale_q/k + keep_indices + scaled_w_q/k).
    /// Bit-identical to per-call construction (same dtype/shape/value).
    /// Opt out with `LUMEN_NATIVE_ALLOC_REUSE=0`.
    fn alloc_reuse_enabled() -> bool {
        std::env::var("LUMEN_NATIVE_ALLOC_REUSE")
            .map(|v| v != "0")
            .unwrap_or(true)
    }

    /// Linear-attn scale fuse: absorb `scale_q/k` into rms_norm weight.
    /// **Default ON 2026-05-11** — thermal-clean A/B (n=10, STEPS=100) confirmed
    /// Δ=-0.315ms, Welch t=-3.33σ WIN. Closes 51% of remaining decode gap;
    /// Native (14.075ms) vs PyO3 (13.913ms) = 1.16% slower (was 2.4%).
    /// Only active when `alloc_reuse_enabled()` is also true (the fused-weight
    /// constants live in `LinearAttnConstants`). Opt out with
    /// `LUMEN_NATIVE_LINEAR_ATTN_SCALE_FUSE=0`.
    /// Identified via op-by-op forward divergence audit 2026-05-11 — 42 ops/step
    /// removed (highest single divergence). Bit-identical by linearity of rms_norm
    /// weight scaling.
    fn linear_attn_scale_fuse_enabled() -> bool {
        std::env::var("LUMEN_NATIVE_LINEAR_ATTN_SCALE_FUSE")
            .map(|v| v != "0")
            .unwrap_or(true)
    }

    /// Divergence #4: conv-state advance via `slice` (mlx_slice view/copy)
    /// instead of `take_axis` (gather kernel). Mirrors Python
    /// `mx.contiguous(conv_input[:, -n_keep:, :])`. Bit-identical for s==1
    /// decode regime — same indices `[s, s+1, ..., total_len-1]` but cheaper
    /// Metal kernel. Opt out with `LUMEN_NATIVE_CONV_SLICE=0`.
    fn conv_slice_enabled() -> bool {
        std::env::var("LUMEN_NATIVE_CONV_SLICE")
            .map(|v| v != "0")
            .unwrap_or(false)
    }

    /// Output of one MTP speculative step on the Qwen3.5/3.6 trunk.
    ///
    /// `committed`: tokens the caller should append to its generation stream.
    ///   Layout `[next_token, accepted_drafts.., correction_token]`. Length =
    ///   `1 + n_accepted + 1`. The LAST element is the token that must be
    ///   passed as `committed_token` to the next `mtp_step()` call (the
    ///   correction on partial reject or the bonus on full accept — neither
    ///   is in the trunk cache yet).
    /// `n_attempted = n_draft` (drafter proposals).
    /// `n_accepted` = drafts that matched the trunk's verify prediction.
    #[derive(Debug, Clone)]
    pub struct MtpStepOutput {
        pub committed: Vec<u32>,
        pub n_attempted: usize,
        pub n_accepted: usize,
    }

    pub struct NativeQwen3_5MoeModel {
        config: NativeModelConfig,
        weights: NativeWeights,
        is_linear_per_layer: Vec<bool>,
        // D1 — pre-resolved per-layer weight handles. Built once at load()
        // time so the forward path skips the HashMap+format!() chain.
        layer_weights: Vec<LayerWeights>,
        // Layer-A — per-model cache of linear-attn forward-path constants.
        // See LinearAttnConstants doc comment.
        linear_attn_constants: LinearAttnConstants,
        embed_weight: Array,
        embed_scales: Array,
        embed_group: i32,
        embed_bits: i32,
        embed_mode: &'static CStr,
        final_norm_weight: Array,
        // None when `tie_word_embeddings = true` — then we fold the embed
        // tensors into the lm_head matmul via the existing tied path.
        lm_head: Option<ResolvedLinear>,
        // precomputed RoPE inverse-frequency table.
        // Same model shares one (head_dim, rope_theta, partial_rotary_factor)
        // tuple across all full-attention layers, so one Array suffices.
        // Forwarded to MLX's `fast::rope(freqs=...)` when
        // `LUMEN_QWEN35_ROPE_PRECOMPUTE_FREQS=1` is set. Default-off opt-in.
        const_rope_freqs: Array,
        // MTP (Multi-Token Prediction) speculative-decode head. `None` until
        // `try_enable_mtp(block)` is called post-load. The block exposes
        // `forward()` consuming (embeds, h_pre, mtp_kv, ...) — see
        // `qwen3_5_mtp::Qwen35MtpBlock`.
        mtp: Option<Qwen35MtpBlock>,
        // Last-hidden capture slot — written by `forward_impl` at the end of
        // the layer loop (post-MLP residual, pre-final-norm) when
        // `mtp_capture_enabled` is set. Read once per `mtp_step` via
        // `take_captured_h()` and reset to `None`. Shape `[1, L, hidden]`
        // matches the MTP block's `h_pre` input contract.
        mtp_capture_slot: Mutex<Option<Array>>,
        mtp_capture_enabled: AtomicBool,
    }

    /// Post-projection / post-RoPE intermediates shared by the plain and
    /// TurboQuant full-attention cores (`full_attn_qkv_rope` output).
    struct FullAttnQkv {
        q_rope: Array,
        k_rope: Array,
        v_t: Array,
        gate: Array,
        scale: f32,
        b: i32,
        l: i32,
        num_heads: i32,
        head_dim: i32,
    }

    /// Pre-RoPE full-attn projection output (Phase 1b Stage 1). Holds the
    /// post-projection, post-per-head-RMSNorm, post-transpose Q/K/V + gate —
    /// everything `full_attn_qkv_rope` produces *before* applying RoPE. The
    /// batched decode path computes this once over `[N,1,hidden]` (amortizing
    /// the weight-bound q/k/v projections) then applies RoPE + SDPA per seq
    /// (each at its own cache offset). `rope_dim`/`base_theta` are carried so the
    /// per-seq RoPE can be applied downstream.
    struct FullAttnProj {
        queries_t: Array,
        k_t: Array,
        v_t: Array,
        gate: Array,
        scale: f32,
        b: i32,
        l: i32,
        num_heads: i32,
        head_dim: i32,
        rope_dim: i32,
        base_theta: f32,
    }

    impl NativeQwen3_5MoeModel {
        /// Load + validate a checkpoint laid out as a directory containing
        /// `config.json` and one or more `*.safetensors` shards.
        pub fn load(model_dir: &Path) -> Result<Self> {
            if !model_dir.is_dir() {
                return Err(anyhow!(
                    "model dir {} does not exist or is not a directory",
                    model_dir.display()
                ));
            }
            let config_path = model_dir.join("config.json");
            let config = NativeModelConfig::load(&config_path)?;
            config.validate_qwen3_5_moe()?;

            let mut weights = NativeWeights::load_dir(model_dir)?;
            weights.sanitize()?;

            let is_linear_per_layer = config.text_config.is_linear_per_layer();

            // D1 — resolve per-layer weight handles once.
            let num_layers = config.text_config.num_hidden_layers;
            let mut layer_weights = Vec::with_capacity(num_layers);
            for (layer_idx, &is_linear) in is_linear_per_layer.iter().enumerate() {
                let attn = if is_linear {
                    AttnLayerWeights::Linear(resolve_linear_attn_layer(
                        &weights, &config, layer_idx,
                    )?)
                } else {
                    AttnLayerWeights::Full(resolve_full_attn_layer(&weights, &config, layer_idx)?)
                };
                let moe = resolve_moe_layer(&weights, &config, layer_idx)?;
                layer_weights.push(LayerWeights { attn, moe });
            }

            // Embedding (always MXFP4 in production) + final norm + lm_head.
            let embed_base = "language_model.model.embed_tokens";
            let embed_weight = require_clone(&weights, &format!("{embed_base}.weight"))?;
            let embed_scales = require_clone(&weights, &format!("{embed_base}.scales"))?;
            let (embed_group, embed_bits, embed_mode) = quant_params_resolve(&config, embed_base);
            let final_norm_weight = require_clone(&weights, "language_model.model.norm.weight")?;
            let lm_head = if config.text_config.tie_word_embeddings {
                None
            } else {
                Some(resolve_linear(&weights, &config, "language_model.lm_head")?)
            };

            // Layer-A — pre-build linear-attn forward-path constants once.
            let linear_attn_constants = LinearAttnConstants::from_config(&config.text_config)?;

            // RoPE inv-freq table for full-attn
            // layers (linear-attn layers don't use RoPE).
            let rope_dim = config.text_config.rope_dim() as i32;
            let const_rope_freqs = precompute_rope_freqs(rope_dim, config.text_config.rope_theta)
                .context("qwen3_5_moe: precompute rope freqs")?;

            Ok(Self {
                config,
                weights,
                is_linear_per_layer,
                layer_weights,
                linear_attn_constants,
                mtp: None,
                mtp_capture_slot: Mutex::new(None),
                mtp_capture_enabled: AtomicBool::new(false),
                embed_weight,
                embed_scales,
                embed_group,
                embed_bits,
                embed_mode,
                final_norm_weight,
                lm_head,
                const_rope_freqs,
            })
        }

        pub fn config(&self) -> &NativeModelConfig {
            &self.config
        }

        pub fn text_config(&self) -> &NativeTextConfig {
            &self.config.text_config
        }

        pub fn weights(&self) -> &NativeWeights {
            &self.weights
        }

        pub fn eos_tokens(&self) -> &[u32] {
            &self.config.eos_token_ids
        }

        pub fn vocab_size(&self) -> usize {
            self.config.text_config.vocab_size
        }

        pub fn num_layers(&self) -> usize {
            self.config.text_config.num_hidden_layers
        }

        /// Build a fresh per-sequence prompt cache matching this model's
        /// layer layout. SSM slot count for Qwen3.5-MoE is 2 (`[conv_state,
        /// ssm_state]`).
        pub fn make_cache(&self) -> NativePromptCache {
            // Opt-in TurboQuant KV compression for the full-attention layers
            // (`LUMEN_QWEN35_TQ_KV=1`). Bits via `LUMEN_QWEN35_TQ_KV_BITS`
            // (default 8 = near-lossless; 6 = lowest-clean; 4 = aggressive).
            // Linear-attn layers keep their compact SSM state unchanged.
            let tq_enabled = std::env::var("LUMEN_QWEN35_TQ_KV")
                .map(|v| matches!(v.as_str(), "1" | "true" | "on" | "yes"))
                .unwrap_or(false);
            if !tq_enabled {
                return NativePromptCache::new(&self.is_linear_per_layer, /* ssm_slots */ 2);
            }
            let bits: u32 = std::env::var("LUMEN_QWEN35_TQ_KV_BITS")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|&b| (2..=8).contains(&b))
                .unwrap_or(8);
            let max_pos = self.text_config().max_position_embeddings;
            eprintln!(
                "[mlx] Qwen35 TurboQuant KV ENABLED (bits={bits}, max_ctx={max_pos}) — \
                 full-attn KV compressed ~{:.1}× vs bf16",
                16.0 / bits as f32
            );
            NativePromptCache::new_with_tq(
                &self.is_linear_per_layer,
                /* ssm_slots */ 2,
                true,
                max_pos,
                bits,
            )
        }

        /// Resolve `(group_size, bits)` for a quantized linear layer.
        /// Looks up per-tensor overrides in the config (keyed by the tensor
        /// path *without* `.weight`/`.scales` suffix), falling back to the
        /// top-level mxfp4 default. Used by callers that already know the
        /// mode (embedding lookup, etc.).
        fn quant_params(&self, base: &str) -> (i32, i32) {
            let (gs, bits, _mode) = self.quant_params_with_mode(base);
            (gs, bits)
        }

        /// Resolve `(group_size, bits, mode)` for a quantized layer.
        /// Convention (mirrors mlx_lm `_quantize`'s `class_predicate`):
        ///
        /// * Per-tensor override present → mode is **affine** (mlx-lm calls
        ///   `to_quantized(group_size=…, bits=…)` without a `mode=` kwarg, so
        ///   `to_quantized`'s default kicks in, which is `"affine"`).
        /// * No override → top-level `quantization_config.mode`. For Qwen3.5-
        ///   MoE that's always `mxfp4`.
        fn quant_params_with_mode(&self, base: &str) -> (i32, i32, &'static CStr) {
            let cfg = self.config.quantization_config.as_ref();
            if let Some(q) = cfg
                && let Some(o) = q.overrides.get(base)
            {
                return (o.group_size as i32, o.bits as i32, MODE_AFFINE);
            }
            cfg.map(|q| (q.group_size as i32, q.bits as i32, MODE_MXFP4))
                .unwrap_or((32, 4, MODE_MXFP4))
        }

        /// Mode-aware Linear forward: `y = x @ deq(W).T`. The mode is
        /// selected per-tensor by `quant_params_with_mode`; the loader
        /// pulls `{base}.weight`, `{base}.scales`, and (for affine) the
        /// optional `{base}.biases` from the sanitized bag.
        fn linear(&self, base: &str, x: &Array) -> Result<Array> {
            let w = self.weights.require(&format!("{base}.weight"))?;
            let s = self.weights.require(&format!("{base}.scales"))?;
            let biases = self.weights.get(&format!("{base}.biases"));
            let (group_size, bits, mode) = self.quant_params_with_mode(base);
            quantized_matmul_with_mode(x, w, s, biases, true, group_size, bits, mode)
                .with_context(|| format!("linear: {base} matmul failed"))
        }

        /// Backwards-compatible alias retained for the attention call sites
        /// authored before the mode-aware refactor — defers to `linear`.
        fn mxfp4_linear(&self, base: &str, x: &Array) -> Result<Array> {
            self.linear(base, x)
        }

        /// D1 lever — pre-resolved Linear forward. Skips the
        /// HashMap+format!() chain that `linear()` does per call. The
        /// `ResolvedLinear` is built once at `load()` time.
        fn linear_pre(&self, lin: &ResolvedLinear, x: &Array) -> Result<Array> {
            quantized_matmul_with_mode(
                x,
                &lin.weight,
                &lin.scales,
                lin.biases.as_ref(),
                true,
                lin.group_size,
                lin.bits,
                lin.mode,
            )
            .context("linear_pre: matmul failed")
        }

        /// Cast helper — MXFP4 matmul output dtype follows activation dtype,
        /// but `.as_dtype(Float32)` is a no-op when already f32 and a real
        /// cast otherwise. Returning a context'd error makes the call site
        /// readable.
        fn to_f32(arr: Array, what: &str) -> Result<Array> {
            arr.as_dtype(mlx_rs::Dtype::Float32)
                .with_context(|| format!("{what}: cast to f32 failed"))
        }

        /// Full-attn layer body. Mirrors
        /// `qwen3_next.Qwen3NextAttention.__call__` for the production
        /// MXFP4-quantized weights:
        ///
        /// 1. q_proj outputs 2× heads → split into `(queries, gate)`.
        /// 2. k_proj / v_proj produce GQA `[B, n_kv, L, head_dim]` after
        ///    reshape + transpose.
        /// 3. q_norm / k_norm (per-head RMSNorm along head_dim).
        /// 4. Partial-rotary GPT-NeoX RoPE on queries + keys (first
        ///    `partial_rotary_factor * head_dim` features rotated, rest
        ///    pass through).
        /// 5. `cache.update_and_fetch(keys, values)` → full KV history.
        /// 6. `sdpa(queries, keys, values, scale, causal=L>1)` returns
        ///    `[B, num_heads, L, head_dim]`.
        /// 7. Reshape to `[B, L, hidden]`, multiply by `sigmoid(gate)`, run
        ///    `o_proj` MXFP4 matmul.
        ///
        /// `causal=true` for prefill (`L > 1`), `false` for single-token decode.
        pub(crate) fn layer_full_attn_forward(
            &self,
            x: &Array,
            layer_idx: usize,
            causal: bool,
            cache: &mut NativeKvCache,
        ) -> Result<Array> {
            let p = self.full_attn_qkv_rope(x, layer_idx, cache.offset() as i32)?;
            // (5) Append to KV cache and fetch full history.
            let (k_full, v_full) = cache.update_and_fetch(&p.k_rope, &p.v_t)?;
            // (6) GQA SDPA. mlx-rs handles the head broadcast internally.
            let attn_out = sdpa(&p.q_rope, &k_full, &v_full, p.scale, causal)?;
            self.full_attn_finish(
                &attn_out,
                &p.gate,
                p.b,
                p.l,
                p.num_heads,
                p.head_dim,
                layer_idx,
            )
        }

        /// TurboQuant-compressed variant of [`Self::layer_full_attn_forward`].
        /// Projections + RoPE are identical (shared `full_attn_qkv_rope`); only
        /// the KV store + attention core differ: K/Q are rotated by a fixed
        /// Haar-orthogonal R (preserving Q·Kᵀ since R Rᵀ = I), K (rotated) and V
        /// are Lloyd-Max scalar-quantized into the compressed cache, then
        /// dequantized on the fly for a standard SDPA. The cache holds uint8
        /// codes + f32 σ instead of bf16 K/V — ~2-4× smaller for the growing
        /// full-attn history. Stage-1 only (no QJL) for this first cut.
        pub(crate) fn layer_full_attn_forward_tq(
            &self,
            x: &Array,
            layer_idx: usize,
            causal: bool,
            cache: &mut NativeRotatingKvCacheTurboQuant,
        ) -> Result<Array> {
            use crate::turboquant::{
                TURBOQUANT_SEED, lloyd_max_centroids, lloyd_max_dequantize_scaled,
                lloyd_max_quantize_stage1, rotate_last_axis, rotation_matrix_f32,
            };
            let p = self.full_attn_qkv_rope(x, layer_idx, cache.offset() as i32)?;

            let centroids = lloyd_max_centroids(cache.bits())
                .context("layer_full_attn_forward_tq: Lloyd-Max centroids")?;
            let r = rotation_matrix_f32(p.head_dim as usize, TURBOQUANT_SEED)
                .context("layer_full_attn_forward_tq: rotation matrix")?;

            // Rotate K + Q (score-preserving), quantize rotated K. Q stays in
            // the rotated frame to pair with the rotated/dequantized K.
            let k_rot = rotate_last_axis(&p.k_rope, &r)?;
            let q_rot = rotate_last_axis(&p.q_rope, &r)?;
            let (k_codes, k_sigma) = lloyd_max_quantize_stage1(&k_rot, &centroids)
                .context("layer_full_attn_forward_tq: quantize K")?;
            // V is quantized un-rotated: O = S·V is correct with V_dq ≈ V, so no
            // un-rotation step is needed on read.
            let v_bf16 = p
                .v_t
                .as_dtype(mlx_rs::Dtype::Bfloat16)
                .context("layer_full_attn_forward_tq: cast V to bf16")?;
            let (v_codes, v_sigma) = lloyd_max_quantize_stage1(&v_bf16, &centroids)
                .context("layer_full_attn_forward_tq: quantize V")?;

            let ((kc, ks), (vc, vs)) = cache
                .update_and_fetch(&k_codes, &k_sigma, &v_codes, &v_sigma)
                .context("layer_full_attn_forward_tq: cache update_and_fetch")?;

            let k_dq = lloyd_max_dequantize_scaled(&kc, &ks, &centroids)
                .context("layer_full_attn_forward_tq: dequantize K")?;
            let v_dq = lloyd_max_dequantize_scaled(&vc, &vs, &centroids)
                .context("layer_full_attn_forward_tq: dequantize V")?;

            let attn_out = sdpa(&q_rot, &k_dq, &v_dq, p.scale, causal)?;
            self.full_attn_finish(
                &attn_out,
                &p.gate,
                p.b,
                p.l,
                p.num_heads,
                p.head_dim,
                layer_idx,
            )
        }

        /// Shared projections + per-head norm + partial-rotary RoPE for the
        /// full-attention layers. Returns the post-RoPE Q/K, the values, and the
        /// gate — everything the plain and TurboQuant attention cores consume.
        /// `offset` is the cache token offset (RoPE position base).
        fn full_attn_qkv_rope(
            &self,
            x: &Array,
            layer_idx: usize,
            offset: i32,
        ) -> Result<FullAttnQkv> {
            // Projections + per-head norm (batched, weight-bound) then RoPE.
            // Single-seq callers get a thin wrapper over the Stage-1 split, so
            // this path is bit-identical to the pre-split implementation.
            let p = self.full_attn_proj_no_rope(x, layer_idx)?;
            let (q_rope, k_rope) =
                self.apply_full_attn_rope(&p.queries_t, &p.k_t, p.rope_dim, p.base_theta, offset)?;
            Ok(FullAttnQkv {
                q_rope,
                k_rope,
                v_t: p.v_t,
                gate: p.gate,
                scale: p.scale,
                b: p.b,
                l: p.l,
                num_heads: p.num_heads,
                head_dim: p.head_dim,
            })
        }

        /// Apply partial-rotary GPT-NeoX RoPE to the transposed Q/K at `offset`
        /// (the cache token position). Honors `LUMEN_QWEN35_ROPE_PRECOMPUTE_FREQS`.
        /// Factored out of `full_attn_qkv_rope` so the batched decode path can
        /// apply RoPE per seq at each seq's own offset.
        fn apply_full_attn_rope(
            &self,
            queries_t: &Array,
            k_t: &Array,
            rope_dim: i32,
            base_theta: f32,
            offset: i32,
        ) -> Result<(Array, Array)> {
            let use_precomputed_freqs = std::env::var("LUMEN_QWEN35_ROPE_PRECOMPUTE_FREQS")
                .map(|v| v == "1")
                .unwrap_or(false);
            if use_precomputed_freqs {
                let freqs = &self.const_rope_freqs;
                let q = rope_with_freqs(queries_t, rope_dim, false, 1.0, offset, freqs)?;
                let k = rope_with_freqs(k_t, rope_dim, false, 1.0, offset, freqs)?;
                Ok((q, k))
            } else {
                let q = rope(queries_t, rope_dim, false, base_theta, 1.0, offset)?;
                let k = rope(k_t, rope_dim, false, base_theta, 1.0, offset)?;
                Ok((q, k))
            }
        }

        /// Steps (1)-(3) of the full-attn forward: q/k/v projections, gate split,
        /// per-head RMSNorm, transpose — everything *before* RoPE. Batch-generic
        /// (`b` read from `x.shape()[0]`). Returns the pre-RoPE Q/K/V + gate.
        fn full_attn_proj_no_rope(&self, x: &Array, layer_idx: usize) -> Result<FullAttnProj> {
            if x.ndim() != 3 {
                return Err(anyhow!(
                    "layer_full_attn_forward: expected x of rank 3 [B, L, hidden], got ndim={}",
                    x.ndim()
                ));
            }
            let cfg = self.text_config();
            let b = x.shape()[0];
            let l = x.shape()[1];
            let num_heads = cfg.num_attention_heads as i32;
            let num_kv = cfg.num_key_value_heads as i32;
            let head_dim = cfg.head_dim as i32;
            let rope_dim = cfg.rope_dim() as i32;
            let scale = (cfg.head_dim as f32).powf(-0.5);
            let base_theta = cfg.rope_theta;
            let eps = cfg.rms_norm_eps;
            let lw = match &self.layer_weights[layer_idx].attn {
                AttnLayerWeights::Full(f) => f,
                AttnLayerWeights::Linear(_) => {
                    return Err(anyhow!(
                        "layer_full_attn_forward: layer {layer_idx} is linear-attn"
                    ));
                }
            };

            // (1) q/k/v projections. MXFP4 matmul output follows activation
            //     dtype — pass f32 in, get f32 out, but cast unconditionally
            //     to keep the contract resilient if MLX changes defaults.
            let q_proj_raw = self.linear_pre(&lw.q_proj, x)?;
            let q_proj = Self::to_f32(q_proj_raw, &format!("layer{layer_idx}.q_proj"))?;
            let k_proj_raw = self.linear_pre(&lw.k_proj, x)?;
            let k_proj = Self::to_f32(k_proj_raw, &format!("layer{layer_idx}.k_proj"))?;
            let v_proj_raw = self.linear_pre(&lw.v_proj, x)?;
            let v_proj = Self::to_f32(v_proj_raw, &format!("layer{layer_idx}.v_proj"))?;

            // (2a) Reshape q_proj to [B, L, num_heads, head_dim*2] and split
            //      the last axis into two halves (queries, gate).
            let q_4d = mlx_rs::ops::reshape(&q_proj, &[b, l, num_heads, head_dim * 2])
                .context("layer_full_attn_forward: reshape q_proj failed")?;
            let parts = mlx_rs::ops::split(&q_4d, 2, -1)
                .context("layer_full_attn_forward: split q_proj failed")?;
            if parts.len() != 2 {
                return Err(anyhow!(
                    "layer_full_attn_forward: split returned {} parts (want 2)",
                    parts.len()
                ));
            }
            // G6-G-1: avoid 2 .clone() ops/layer × 7 full-attn layers = 14
            // mlx-c calls/step. Use `into_iter` to take owned values without
            // clones (Vec<Array> drops as iter yields).
            let mut parts_iter = parts.into_iter();
            let queries = parts_iter
                .next()
                .expect("layer_full_attn_forward: split parts[0] missing");
            let gate_4d = parts_iter
                .next()
                .expect("layer_full_attn_forward: split parts[1] missing");
            let gate = mlx_rs::ops::reshape(&gate_4d, &[b, l, num_heads * head_dim])
                .context("layer_full_attn_forward: reshape gate failed")?;

            // (3) Per-head RMSNorm. weight shapes: [head_dim] for both
            //     q_norm and k_norm.
            let q_norm_w = &lw.q_norm_weight;
            let k_norm_w = &lw.k_norm_weight;
            let queries_n = rms_norm(&queries, q_norm_w, eps)?;
            let queries_t = mlx_rs::ops::transpose_axes(&queries_n, &[0, 2, 1, 3])
                .context("layer_full_attn_forward: transpose queries failed")?;

            // (2b) Reshape k/v to [B, L, num_kv, head_dim] then RMSNorm/transpose.
            let k_4d = mlx_rs::ops::reshape(&k_proj, &[b, l, num_kv, head_dim])
                .context("layer_full_attn_forward: reshape k_proj failed")?;
            let k_n = rms_norm(&k_4d, k_norm_w, eps)?;
            let k_t = mlx_rs::ops::transpose_axes(&k_n, &[0, 2, 1, 3])
                .context("layer_full_attn_forward: transpose keys failed")?;

            let v_4d = mlx_rs::ops::reshape(&v_proj, &[b, l, num_kv, head_dim])
                .context("layer_full_attn_forward: reshape v_proj failed")?;
            let v_t = mlx_rs::ops::transpose_axes(&v_4d, &[0, 2, 1, 3])
                .context("layer_full_attn_forward: transpose values failed")?;

            // RoPE (step 4) is applied by the caller via `apply_full_attn_rope`
            // — batched single-seq through `full_attn_qkv_rope`, or per-seq at
            // each seq's own offset in the batched decode path.
            Ok(FullAttnProj {
                queries_t,
                k_t,
                v_t,
                gate,
                scale,
                b,
                l,
                num_heads,
                head_dim,
                rope_dim,
                base_theta,
            })
        }

        /// Tail shared by the plain and TurboQuant full-attn cores: gate the
        /// attention output by the sigmoid gate and project back to hidden size
        /// (steps 7-8 of the original `layer_full_attn_forward`).
        fn full_attn_finish(
            &self,
            attn_out: &Array,
            gate: &Array,
            b: i32,
            l: i32,
            num_heads: i32,
            head_dim: i32,
            layer_idx: usize,
        ) -> Result<Array> {
            let lw = match &self.layer_weights[layer_idx].attn {
                AttnLayerWeights::Full(f) => f,
                AttnLayerWeights::Linear(_) => {
                    return Err(anyhow!(
                        "full_attn_finish: layer {layer_idx} is linear-attn"
                    ));
                }
            };
            // (7) Reshape attention output back to [B, L, hidden] and gate.
            let attn_t = mlx_rs::ops::transpose_axes(attn_out, &[0, 2, 1, 3])
                .context("layer_full_attn_forward: transpose attn_out failed")?;
            let attn_flat = mlx_rs::ops::reshape(&attn_t, &[b, l, num_heads * head_dim])
                .context("layer_full_attn_forward: reshape attn_out failed")?;
            let gated = if sigmoid_mul_fuse_enabled_pub() {
                sigmoid_mul_fused(gate, &attn_flat)
                    .context("layer_full_attn_forward: fused sigmoid_mul failed")?
            } else {
                let gate_sig = mlx_rs::ops::sigmoid(gate)
                    .context("layer_full_attn_forward: sigmoid(gate) failed")?;
                mlx_rs::ops::multiply(&attn_flat, &gate_sig)
                    .context("layer_full_attn_forward: gated multiply failed")?
            };

            // (8) o_proj: project back to hidden_size.
            let o_proj_raw = self.linear_pre(&lw.o_proj, &gated)?;
            Self::to_f32(o_proj_raw, &format!("layer{layer_idx}.o_proj"))
        }

        /// Phase 1b Stage 1: batched full-attn over `[N,1,hidden]` with N per-seq
        /// caches. The weight-bound q/k/v/o projections + gate run ONCE over all
        /// N tokens (the throughput lever); only the cheap per-seq RoPE + SDPA
        /// loop (each at its own cache offset / KV history). Bit-identical to the
        /// per-seq single-attn path by construction — same projection math, same
        /// per-seq RoPE/SDPA. Plain (non-TurboQuant) caches only; the caller
        /// routes TurboQuant layers to the per-seq whole-layer fallback.
        fn layer_full_attn_forward_batch(
            &self,
            x: &Array,
            layer_idx: usize,
            caches: &mut [&mut NativeLayerCache],
        ) -> Result<Array> {
            let p = self.full_attn_proj_no_rope(x, layer_idx)?;
            let mut attn_parts: Vec<Array> = Vec::with_capacity(caches.len());
            for (i, c) in caches.iter_mut().enumerate() {
                let idx_i = Array::from_slice(&[i as i32], &[1]);
                let q_i = p
                    .queries_t
                    .take_axis(&idx_i, 0)
                    .context("layer_full_attn_forward_batch: slice queries failed")?;
                let k_i = p
                    .k_t
                    .take_axis(&idx_i, 0)
                    .context("layer_full_attn_forward_batch: slice keys failed")?;
                let v_i = p
                    .v_t
                    .take_axis(&idx_i, 0)
                    .context("layer_full_attn_forward_batch: slice values failed")?;
                let kv = c.as_full_mut()?;
                let offset_i = kv.offset() as i32;
                let (q_rope, k_rope) =
                    self.apply_full_attn_rope(&q_i, &k_i, p.rope_dim, p.base_theta, offset_i)?;
                let (k_full, v_full) = kv.update_and_fetch(&k_rope, &v_i)?;
                let attn_i = sdpa(&q_rope, &k_full, &v_full, p.scale, /* causal */ false)?;
                attn_parts.push(attn_i);
            }
            let refs: Vec<&Array> = attn_parts.iter().collect();
            let attn_stacked = mlx_rs::ops::concatenate_axis(&refs, 0)
                .context("layer_full_attn_forward_batch: stack attn outputs failed")?;
            self.full_attn_finish(
                &attn_stacked,
                &p.gate,
                p.b,
                p.l,
                p.num_heads,
                p.head_dim,
                layer_idx,
            )
        }

        /// Linear-attention (`GatedDeltaNet`) layer body.
        /// Mirrors `qwen3_5.GatedDeltaNet.__call__` for the MXFP4 production
        /// checkpoint (mask path is not used because `create_ssm_mask` returns
        /// `None` whenever `cache.lengths` is unset, which is the only regime
        /// lumen-rs exercises today). The 4-way split (`in_proj_qkv` /
        /// `in_proj_z` / `in_proj_b` / `in_proj_a`) is required for bit-
        /// identical parity — the upstream `mlx_lm/models/qwen3_5_moe.py`
        /// sanitizer does not concat these into the fused `qkvz`/`ba` form
        /// (that lives in `qwen3_next.py` for a different model family), so
        /// the safetensors keep the 4-way layout end-to-end:
        ///
        /// 1. `in_proj_qkv` (output_dim = `key_dim*2 + value_dim`) +
        ///    `in_proj_z` (output_dim = `value_dim`) +
        ///    `in_proj_b` / `in_proj_a` (output_dim = `num_v_heads`) —
        ///    four separate MXFP4 matmuls + cast f32.
        /// 2. `z` reshape to `[B, S, num_v_heads, head_v_dim]`; `b`/`a`
        ///    keep `[B, S, num_v_heads]`.
        /// 3. Pull `conv_state` from `cache[0]` (zero-init on first call).
        /// 4. `conv_input = concat([conv_state, qkv], axis=1)` — `qkv` is
        ///    the direct `in_proj_qkv` output (already `[B, S, conv_dim]`).
        /// 5. Update `cache[0]` to the trailing `kernel_size - 1` rows of
        ///    `conv_input` (the post-step convolution carryover).
        /// 6. `conv_out = silu(conv1d(conv_input))` — depthwise Conv1d with
        ///    `groups == conv_dim`.
        /// 7. Re-split the Conv1d output into `(q, k, v)` heads.
        /// 8. Per-head `fast::rms_norm(weight=ones, eps=1e-6)` on q/k, then
        ///    rescale by `inv_scale^2` (q) and `inv_scale` (k) where
        ///    `inv_scale = head_k_dim^-0.5`.
        /// 9. `gated_delta_step_kernel(q, k, v, g, beta, state)` — Metal SSM
        ///    recurrence (Phase 3b.5.d). `g = compute_g(A_log, a, dt_bias)`,
        ///    `beta = sigmoid(b)`, and `state` comes from `cache[1]`
        ///    (zero-init on first call). Updates `cache[1]` and bumps
        ///    `cache.advance(S)`.
        /// 10. `rms_norm_gated(out, z, norm.weight, eps)` — per-head output
        ///     gate (`silu(z) * rms_norm(out)`).
        /// 11. `out_proj` — MXFP4 matmul from `[B, S, value_dim]` back to
        ///     `[B, S, hidden_size]`.
        pub(crate) fn layer_linear_attn_forward(
            &self,
            x: &Array,
            layer_idx: usize,
            cache: &mut NativeArraysCache,
        ) -> Result<Array> {
            if x.ndim() != 3 {
                return Err(anyhow!(
                    "layer_linear_attn_forward: expected x of rank 3 [B, L, hidden], got ndim={}",
                    x.ndim()
                ));
            }
            let cfg = self.text_config();
            let b = x.shape()[0];
            let s = x.shape()[1];
            let nk = cfg.linear_num_key_heads as i32;
            let nv = cfg.linear_num_value_heads as i32;
            let dn = cfg.linear_key_head_dim as i32;
            let dv = cfg.linear_value_head_dim as i32;
            let kernel_size = cfg.linear_conv_kernel_dim as i32;
            let eps = cfg.rms_norm_eps;
            if nk == 0 || nv == 0 || dn == 0 || dv == 0 || kernel_size <= 1 || nv % nk != 0 {
                return Err(anyhow!(
                    "layer_linear_attn_forward: invalid linear-attn dims (nk={nk}, nv={nv}, dn={dn}, dv={dv}, kernel={kernel_size})"
                ));
            }
            let nv_per_nk = nv / nk;
            let key_dim = nk * dn;
            let value_dim = nv * dv;
            let conv_dim = key_dim * 2 + value_dim;
            let n_keep = kernel_size - 1;
            let lw = match &self.layer_weights[layer_idx].attn {
                AttnLayerWeights::Linear(l) => l,
                AttnLayerWeights::Full(_) => {
                    return Err(anyhow!(
                        "layer_linear_attn_forward: layer {layer_idx} is full-attn"
                    ));
                }
            };
            let timing_on = fine_timing_active();

            // (A: in_proj) Four separate in_proj matmuls + reshape z.
            //     The `qwen3_5_moe` checkpoint stores these split (no
            //     `qkvz`/`ba` fusion in the upstream sanitizer), so MXFP4
            //     parity requires 4 distinct matmuls — fusing in activation
            //     space would change the per-tensor quantization boundaries.
            //     Lever A: dropped 4× redundant `to_f32` casts; `mxfp4_linear`
            //     returns the input dtype (f32 here) so the casts were
            //     identity ops adding only graph nodes.
            let t_in_proj = timing_on.then(Instant::now);
            let mixed_qkv = self.linear_pre(&lw.in_proj_qkv, x)?;
            let z_flat = self.linear_pre(&lw.in_proj_z, x)?;
            let b_pre = self.linear_pre(&lw.in_proj_b, x)?;
            let a_pre = self.linear_pre(&lw.in_proj_a, x)?;
            let z_4d = mlx_rs::ops::reshape(&z_flat, &[b, s, nv, dv])
                .context("layer_linear_attn_forward: reshape z failed")?;
            let _ = nv_per_nk; // retained for future fused-mode toggle
            if let Some(t0) = t_in_proj {
                // Batched eval: matches Python `mx.eval(qkv, z, b, a)` —
                // single GPU sync barrier for all 4 outputs vs 4 sequential
                // syncs. Removes instrumentation artifact from per-layer diff.
                mlx_rs::transforms::eval([&z_4d, &mixed_qkv, &b_pre, &a_pre])
                    .context("layer_linear_attn_forward: batched eval(in_proj) failed")?;
                bump_linear_in_proj_ms(t0.elapsed().as_secs_f64() * 1000.0);
            }

            // (B: conv) conv_state load → concat → cache update via take_axis →
            //           depthwise conv1d → silu → split → reshape into q/k/v.
            let t_conv = timing_on.then(Instant::now);
            // G6-G-1: borrow conv_state from cache instead of cloning. The
            // owned fallback (zeros_dtype) is stashed in `_conv_state_owned`
            // and a `&Array` is used uniformly downstream. Saves 1 mlx-c
            // entry call (mlx_array_set) per linear-attn layer × 21 layers
            // ≈ 21 ops/step on this model.
            let _conv_state_owned;
            let conv_state: &Array = match cache.get(0) {
                Some(arr) => arr,
                None => {
                    _conv_state_owned =
                        mlx_rs::ops::zeros_dtype(&[b, n_keep, conv_dim], mlx_rs::Dtype::Float32)
                            .context("layer_linear_attn_forward: zero-init conv_state failed")?;
                    &_conv_state_owned
                }
            };
            let _ = (key_dim, value_dim); // kept for documentation alignment
            let conv_input = mlx_rs::ops::concatenate_axis(&[conv_state, &mixed_qkv], 1)
                .context("layer_linear_attn_forward: concat conv_input failed")?;
            let total_len = s + n_keep;
            // Layer-A: decode regime (s=1) reuses cached keep_indices_decode.
            // Prefill (s>1) keeps the dynamic path. Bit-identical: same i32
            // values, same shape, same dtype.
            // Divergence #4 fix (2026-05-12): replace `take_axis` (gather kernel)
            // with `try_index` slice (view/copy). Bit-identical to
            // `take_axis(conv_input, [s, s+1, ..., total_len-1], axis=1)` but
            // dispatches to mlx_slice (cheaper than mlx_take_axis on Metal).
            // Mirrors Python's `mx.contiguous(conv_input[:, -n_keep:, :])`.
            let new_conv_state = if conv_slice_enabled() && s == 1 {
                use mlx_rs::ops::indexing::TryIndexOp;
                conv_input
                    .try_index((.., (s as i32)..(total_len as i32), ..))
                    .context("layer_linear_attn_forward: slice(conv_state, decode) failed")?
            } else if alloc_reuse_enabled() && s == 1 {
                mlx_rs::ops::indexing::take_axis(
                    &conv_input,
                    &self.linear_attn_constants.keep_indices_decode,
                    1,
                )
                .context("layer_linear_attn_forward: take_axis(conv_state, cached) failed")?
            } else {
                let keep_indices: Vec<i32> = (s..total_len).collect();
                let keep_idx_arr = Array::from_slice(&keep_indices, &[n_keep]);
                mlx_rs::ops::indexing::take_axis(&conv_input, &keep_idx_arr, 1)
                    .context("layer_linear_attn_forward: take_axis(conv_state) failed")?
            };
            cache.set(0, new_conv_state)?;
            let conv_w = &lw.conv1d_weight;
            let conv_raw = conv1d(
                &conv_input,
                conv_w,
                /* stride */ 1,
                /* padding */ 0,
                conv_dim,
            )?;
            let conv_out = mlx_rs::nn::silu(&conv_raw)
                .context("layer_linear_attn_forward: silu(conv1d) failed")?;
            let conv_parts = mlx_rs::ops::split_sections(&conv_out, &[key_dim, 2 * key_dim], -1)
                .context("layer_linear_attn_forward: split conv_out failed")?;
            if conv_parts.len() != 3 {
                return Err(anyhow!(
                    "layer_linear_attn_forward: conv split returned {} parts (want 3)",
                    conv_parts.len()
                ));
            }
            let q_post = mlx_rs::ops::reshape(&conv_parts[0], &[b, s, nk, dn])
                .context("layer_linear_attn_forward: reshape q_post failed")?;
            let k_post = mlx_rs::ops::reshape(&conv_parts[1], &[b, s, nk, dn])
                .context("layer_linear_attn_forward: reshape k_post failed")?;
            let v_post = mlx_rs::ops::reshape(&conv_parts[2], &[b, s, nv, dv])
                .context("layer_linear_attn_forward: reshape v_post failed")?;
            if let Some(t0) = t_conv {
                mlx_rs::transforms::eval([&q_post, &k_post, &v_post])
                    .context("layer_linear_attn_forward: batched eval(conv) failed")?;
                bump_linear_conv_ms(t0.elapsed().as_secs_f64() * 1000.0);
            }

            // (C: norm_qk) ones-weight rms_norm × 2 + scalar broadcast multiplies.
            let t_norm_qk = timing_on.then(Instant::now);
            // Layer-A: cached ones_w + scale_q/k vs per-call construction.
            // Bit-identical replacement (same dtype/shape/value).
            let (q_n, k_n, q_scaled, k_scaled) = if alloc_reuse_enabled() {
                if linear_attn_scale_fuse_enabled() {
                    // Fused path (LANDED 2026-05-11): absorb scale_q/k into
                    // rms_norm weight (`scaled_w_q = ones × scale_q`). Saves
                    // 2 multiply ops/linear-attn layer × 21 layers = 42 ops/step.
                    // Bit-identical: rms_norm(x, ones*s) = rms_norm(x, ones) * s.
                    let q_scaled = mlx_rs::fast::rms_norm(
                        &q_post,
                        &self.linear_attn_constants.scaled_w_q_dn,
                        1e-6,
                    )
                    .context("layer_linear_attn_forward: rms_norm(q, scaled_w) failed")?;
                    let k_scaled = mlx_rs::fast::rms_norm(
                        &k_post,
                        &self.linear_attn_constants.scaled_w_k_dn,
                        1e-6,
                    )
                    .context("layer_linear_attn_forward: rms_norm(k, scaled_w) failed")?;
                    // q_n/k_n unused in fused path — fabricate dummy refs to
                    // satisfy the tuple return; the `let _ = (q_n, k_n)`
                    // below silences the unused warning.
                    (q_scaled.clone(), k_scaled.clone(), q_scaled, k_scaled)
                } else {
                    let ones_w = &self.linear_attn_constants.ones_w_dn;
                    let q_n = mlx_rs::fast::rms_norm(&q_post, ones_w, 1e-6)
                        .context("layer_linear_attn_forward: rms_norm(q, cached) failed")?;
                    let k_n = mlx_rs::fast::rms_norm(&k_post, ones_w, 1e-6)
                        .context("layer_linear_attn_forward: rms_norm(k, cached) failed")?;
                    let q_scaled = mlx_rs::ops::multiply(&q_n, &self.linear_attn_constants.scale_q)
                        .context("layer_linear_attn_forward: q_scaled multiply (cached) failed")?;
                    let k_scaled = mlx_rs::ops::multiply(&k_n, &self.linear_attn_constants.scale_k)
                        .context("layer_linear_attn_forward: k_scaled multiply (cached) failed")?;
                    (q_n, k_n, q_scaled, k_scaled)
                }
            } else {
                let inv_scale = (dn as f32).powf(-0.5);
                let ones_w = mlx_rs::ops::ones::<f32>(&[dn])
                    .context("layer_linear_attn_forward: ones weight failed")?;
                let q_n = mlx_rs::fast::rms_norm(&q_post, &ones_w, 1e-6)
                    .context("layer_linear_attn_forward: rms_norm(q) failed")?;
                let k_n = mlx_rs::fast::rms_norm(&k_post, &ones_w, 1e-6)
                    .context("layer_linear_attn_forward: rms_norm(k) failed")?;
                let scale_q = Array::from_f32(inv_scale * inv_scale);
                let scale_k = Array::from_f32(inv_scale);
                let q_scaled = mlx_rs::ops::multiply(&q_n, &scale_q)
                    .context("layer_linear_attn_forward: q_scaled multiply failed")?;
                let k_scaled = mlx_rs::ops::multiply(&k_n, &scale_k)
                    .context("layer_linear_attn_forward: k_scaled multiply failed")?;
                (q_n, k_n, q_scaled, k_scaled)
            };
            let _ = (q_n, k_n); // values consumed only via q_scaled/k_scaled
            if let Some(t0) = t_norm_qk {
                mlx_rs::transforms::eval([&q_scaled, &k_scaled])
                    .context("layer_linear_attn_forward: batched eval(norm_qk) failed")?;
                bump_linear_norm_qk_ms(t0.elapsed().as_secs_f64() * 1000.0);
            }

            // (D: ssm) sigmoid(b) + compute_g + Metal SSM step kernel.
            let t_ssm = timing_on.then(Instant::now);
            let beta = mlx_rs::ops::sigmoid(&b_pre)
                .context("layer_linear_attn_forward: sigmoid(b) failed")?;
            let a_log = &lw.a_log;
            let dt_bias = &lw.dt_bias;
            let g = compute_g(a_log, &a_pre, dt_bias)?;
            // G6-G-1: borrow state_in from cache (same pattern as conv_state
            // above). Saves another ~21 mlx-c entry calls/step.
            let _state_in_owned;
            let state_in: &Array = match cache.get(1) {
                Some(arr) => arr,
                None => {
                    _state_in_owned =
                        mlx_rs::ops::zeros_dtype(&[b, nv, dv, dn], mlx_rs::Dtype::Float32)
                            .context("layer_linear_attn_forward: zero-init ssm_state failed")?;
                    &_state_in_owned
                }
            };
            let (out_kernel, new_state) =
                gated_delta_step_kernel(&q_scaled, &k_scaled, &v_post, &g, &beta, state_in)?;
            cache.set(1, new_state)?;
            cache.advance(s as usize);
            if let Some(t0) = t_ssm {
                out_kernel
                    .eval()
                    .context("layer_linear_attn_forward: timing eval(ssm.out_kernel) failed")?;
                bump_linear_ssm_ms(t0.elapsed().as_secs_f64() * 1000.0);
            }

            // (E: out_proj) RMSNormGated → reshape → out_proj matmul → cast.
            let t_out = timing_on.then(Instant::now);
            let norm_w = &lw.norm_weight;
            let out_normed = rms_norm_gated(&out_kernel, &z_4d, norm_w, eps)?;
            let out_flat = mlx_rs::ops::reshape(&out_normed, &[b, s, value_dim])
                .context("layer_linear_attn_forward: reshape out_flat failed")?;
            let out_proj_raw = self.linear_pre(&lw.out_proj, &out_flat)?;
            let out_final = Self::to_f32(out_proj_raw, &format!("layer{layer_idx}.out_proj"))?;
            if let Some(t0) = t_out {
                out_final
                    .eval()
                    .context("layer_linear_attn_forward: timing eval(out.out_final) failed")?;
                bump_linear_out_proj_ms(t0.elapsed().as_secs_f64() * 1000.0);
            }
            Ok(out_final)
        }

        /// Resolve the per-tensor `(group_size, bits, mode)` and weight
        /// references into the structures `native_moe::moe_block_forward`
        /// needs.
        fn moe_weights(&self, layer_idx: usize) -> Result<MoeBlockWeights<'_>> {
            let cfg = self.text_config();
            let prefix = format!("language_model.model.layers.{layer_idx}.mlp");

            // gate (8-bit affine override)
            let (gate_gs, gate_bits, _gate_mode) =
                self.quant_params_with_mode(&format!("{prefix}.gate"));
            let gate_w = self.weights.require(&format!("{prefix}.gate.weight"))?;
            let gate_s = self.weights.require(&format!("{prefix}.gate.scales"))?;
            let gate_b = self.weights.require(&format!("{prefix}.gate.biases"))?;

            // shared_expert_gate (8-bit affine override; output dim = 1)
            let (sg_gs, sg_bits, _sg_mode) =
                self.quant_params_with_mode(&format!("{prefix}.shared_expert_gate"));
            let sg_w = self
                .weights
                .require(&format!("{prefix}.shared_expert_gate.weight"))?;
            let sg_s = self
                .weights
                .require(&format!("{prefix}.shared_expert_gate.scales"))?;
            let sg_b = self
                .weights
                .require(&format!("{prefix}.shared_expert_gate.biases"))?;

            // switch_mlp (mxfp4 default — 4-bit, group_size=32, no biases)
            let (sw_gs, sw_bits, sw_mode) =
                self.quant_params_with_mode(&format!("{prefix}.switch_mlp.gate_proj"));
            let sw_gate_w = self
                .weights
                .require(&format!("{prefix}.switch_mlp.gate_proj.weight"))?;
            let sw_gate_s = self
                .weights
                .require(&format!("{prefix}.switch_mlp.gate_proj.scales"))?;
            let sw_up_w = self
                .weights
                .require(&format!("{prefix}.switch_mlp.up_proj.weight"))?;
            let sw_up_s = self
                .weights
                .require(&format!("{prefix}.switch_mlp.up_proj.scales"))?;
            let sw_down_w = self
                .weights
                .require(&format!("{prefix}.switch_mlp.down_proj.weight"))?;
            let sw_down_s = self
                .weights
                .require(&format!("{prefix}.switch_mlp.down_proj.scales"))?;

            // shared_expert (mxfp4 default)
            let (se_gs, se_bits, se_mode) =
                self.quant_params_with_mode(&format!("{prefix}.shared_expert.gate_proj"));
            let se_gate_w = self
                .weights
                .require(&format!("{prefix}.shared_expert.gate_proj.weight"))?;
            let se_gate_s = self
                .weights
                .require(&format!("{prefix}.shared_expert.gate_proj.scales"))?;
            let se_up_w = self
                .weights
                .require(&format!("{prefix}.shared_expert.up_proj.weight"))?;
            let se_up_s = self
                .weights
                .require(&format!("{prefix}.shared_expert.up_proj.scales"))?;
            let se_down_w = self
                .weights
                .require(&format!("{prefix}.shared_expert.down_proj.weight"))?;
            let se_down_s = self
                .weights
                .require(&format!("{prefix}.shared_expert.down_proj.scales"))?;

            Ok(MoeBlockWeights {
                gate: AffineGateWeights {
                    weight: gate_w,
                    scales: gate_s,
                    biases: gate_b,
                    group_size: gate_gs,
                    bits: gate_bits,
                },
                switch_glu: SwitchGluWeights {
                    gate_proj: SwitchExpertWeights {
                        weight: sw_gate_w,
                        scales: sw_gate_s,
                        biases: None,
                    },
                    up_proj: SwitchExpertWeights {
                        weight: sw_up_w,
                        scales: sw_up_s,
                        biases: None,
                    },
                    down_proj: SwitchExpertWeights {
                        weight: sw_down_w,
                        scales: sw_down_s,
                        biases: None,
                    },
                    group_size: sw_gs,
                    bits: sw_bits,
                    mode: sw_mode,
                },
                shared_expert: SharedMlpWeights {
                    gate_proj_w: se_gate_w,
                    gate_proj_s: se_gate_s,
                    up_proj_w: se_up_w,
                    up_proj_s: se_up_s,
                    down_proj_w: se_down_w,
                    down_proj_s: se_down_s,
                    group_size: se_gs,
                    bits: se_bits,
                    mode: se_mode,
                },
                shared_gate: AffineGateWeights {
                    weight: sg_w,
                    scales: sg_s,
                    biases: sg_b,
                    group_size: sg_gs,
                    bits: sg_bits,
                },
                top_k: cfg.num_experts_per_tok as i32,
                norm_topk_prob: cfg.norm_topk_prob,
            })
        }

        /// One-layer MoE block dispatch. `x` is the
        /// post-attention-layernorm output of shape `[B, L, hidden]`; the
        /// returned tensor is the additive update to the residual stream
        /// (the caller adds `h + mlp_out`).
        pub(crate) fn layer_moe_forward(&self, x: &Array, layer_idx: usize) -> Result<Array> {
            let weights = self.layer_weights[layer_idx].moe.view();
            let raw = moe_block_forward(x, &weights)
                .with_context(|| format!("layer_moe_forward({layer_idx}) failed"))?;
            Self::to_f32(raw, &format!("layer{layer_idx}.mlp"))
        }

        /// Forward pass. Phase 3d.4 closes the loop: every layer runs
        /// `input_layernorm → attn → +residual → post_attention_layernorm →
        /// MoE → +residual`, the final `model.norm` is applied, and either
        /// `lm_head` (untied) or the embedding `as_linear` (tied) projects
        /// the hidden states into vocab logits with shape `[B, L, vocab]`.
        pub fn forward(&self, input_ids: &[u32], cache: &mut NativePromptCache) -> Result<Array> {
            let (logits, _) =
                self.forward_impl(input_ids, cache, /* last_only */ false, &[])?;
            Ok(logits)
        }

        /// Forward with control over whether the lm_head projection runs over
        /// all `L` token positions or only the last one. Internally hot
        /// callers (prefill / extend / decode_step) only need the last
        /// position's logits to pick the next token; computing lm_head over
        /// all `L` rows wastes ~`(L-1) * hidden * vocab` FMAs per request.
        /// For a 2400-token prefill on the 35B production checkpoint
        /// (vocab=248320, hidden=2048) that is ~12% of measured prefill wall
        /// clock. `forward_probe` keeps the full-row contract by passing
        /// `last_only = false`.
        pub fn forward_with_opts(
            &self,
            input_ids: &[u32],
            cache: &mut NativePromptCache,
            last_only: bool,
        ) -> Result<Array> {
            let (logits, _) = self.forward_impl(input_ids, cache, last_only, &[])?;
            Ok(logits)
        }

        /// Forward with optional per-layer hidden-state capture. For each
        /// `layer_idx` in `capture_layer_ids`, the post-MLP residual output of
        /// that layer (i.e., what `_LayerHook` records in the Python
        /// `dflash_runtime.py::patch_model`) is cloned into the returned
        /// `Vec<Array>` in the order the IDs are listed. Each captured tensor
        /// has shape `[1, L, hidden]` matching the layer's running residual.
        ///
        /// Used by D5a (DFlash native port) — the captured hiddens are then
        /// concatenated along the channel axis to form the draft model's
        /// `target_hidden` cross-attn context (matches `fc` weight layout
        /// `[len(target_layer_ids) * hidden, hidden]`).
        ///
        /// `capture_layer_ids` MUST contain valid in-range indices and SHOULD
        /// be small relative to `num_layers()` — duplicate IDs simply produce
        /// duplicate clones; out-of-range IDs are silently ignored (caller
        /// validates).
        pub fn forward_with_capture(
            &self,
            input_ids: &[u32],
            cache: &mut NativePromptCache,
            last_only: bool,
            capture_layer_ids: &[usize],
        ) -> Result<(Array, Vec<Array>)> {
            self.forward_impl(input_ids, cache, last_only, capture_layer_ids)
        }

        /// Inner forward implementation, parameterized by both `last_only`
        /// (lm_head row-trim) and `capture_layer_ids` (per-layer post-MLP
        /// hidden-state capture). Public-facing `forward` /
        /// `forward_with_opts` / `forward_with_capture` are thin wrappers.
        fn forward_impl(
            &self,
            input_ids: &[u32],
            cache: &mut NativePromptCache,
            last_only: bool,
            capture_layer_ids: &[usize],
        ) -> Result<(Array, Vec<Array>)> {
            if input_ids.is_empty() {
                return Err(anyhow!("forward: input_ids is empty"));
            }
            if cache.len() != self.num_layers() {
                return Err(anyhow!(
                    "forward: cache.len()={} mismatches num_layers={}",
                    cache.len(),
                    self.num_layers()
                ));
            }
            let cfg = self.text_config();
            let l = input_ids.len() as i32;

            let timing_on = fine_timing_active();
            let mut fine = FineTimings::default();

            // Embedding lookup. The production checkpoint stores
            // `embed_tokens` as MXFP4-quantized; the lookup returns bf16,
            // which we cast to f32 for downstream determinism.
            let t_embed = timing_on.then(Instant::now);
            let ids_i32: Vec<i32> = input_ids.iter().map(|&u| u as i32).collect();
            let ids_arr = Array::from_slice(&ids_i32, &[l]);
            let embed_raw = quantized_embedding_lookup_with_mode(
                &self.embed_weight,
                &self.embed_scales,
                &ids_arr,
                self.embed_group,
                self.embed_bits,
                self.embed_mode,
            )
            .context("forward: embed_tokens lookup failed")?;
            let embed_f32 = Self::to_f32(embed_raw, "forward.embed_tokens")?;
            let hidden_dim = cfg.hidden_size as i32;
            let hidden_states = mlx_rs::ops::reshape(&embed_f32, &[1, l, hidden_dim])
                .context("forward: reshape embeddings to [1, L, hidden] failed")?;
            if let Some(t0) = t_embed {
                hidden_states
                    .eval()
                    .context("forward: timing eval(embed) failed")?;
                fine.embed_ms = t0.elapsed().as_secs_f64() * 1000.0;
            }

            // Mask rule mirrors `create_attention_mask`: L==1 → no mask;
            // L>1 → causal. (cache.offset > 0 with L==1 is the decode regime
            // where SDPA needs no mask because the new query attends across
            // the full history.)
            let causal = l > 1;

            // Per-target-layer capture buffer. Empty when capture_layer_ids
            // is empty (the existing forward / forward_with_opts call paths).
            // For DFlash, the caller passes `target_layer_ids` (typically 5
            // layers on Qwen3.6-A3B-DFlash: [1, 10, 19, 28, 37]) and consumes
            // the resulting `[1, L, hidden]` clones as cross-attn ctx.
            let mut captured: Vec<Array> = Vec::with_capacity(capture_layer_ids.len());

            let mut hidden_states = hidden_states;
            for layer_idx in 0..self.num_layers() {
                let layer_w = &self.layer_weights[layer_idx];
                let input_norm_w = match &layer_w.attn {
                    AttnLayerWeights::Linear(l) => &l.input_layernorm,
                    AttnLayerWeights::Full(f) => &f.input_layernorm,
                };
                let normed = rms_norm(&hidden_states, input_norm_w, cfg.rms_norm_eps)?;

                let layer_cache = cache
                    .layer_mut(layer_idx)
                    .ok_or_else(|| anyhow!("forward: cache layer {layer_idx} missing"))?;
                let is_linear = self.is_linear_per_layer[layer_idx];
                if timing_on {
                    reset_layer_sub_timings();
                }
                let t_attn = timing_on.then(Instant::now);
                let attn = if is_linear {
                    let lin = layer_cache.as_linear_mut()?;
                    self.layer_linear_attn_forward(&normed, layer_idx, lin)?
                } else if matches!(layer_cache, NativeLayerCache::FullTurboquant(_)) {
                    let tq = layer_cache.as_full_turboquant_mut()?;
                    self.layer_full_attn_forward_tq(&normed, layer_idx, causal, tq)?
                } else {
                    let kv = layer_cache.as_full_mut()?;
                    self.layer_full_attn_forward(&normed, layer_idx, causal, kv)?
                };
                if let Some(t0) = t_attn {
                    attn.eval().context("forward: timing eval(attn) failed")?;
                    let dt = t0.elapsed().as_secs_f64() * 1000.0;
                    if is_linear {
                        fine.linear_attn_ms += dt;
                        fine.linear_attn_count += 1;
                        let sub = drain_layer_sub_timings();
                        fine.linear_in_proj_ms += sub.linear_in_proj_ms;
                        fine.linear_conv_ms += sub.linear_conv_ms;
                        fine.linear_norm_qk_ms += sub.linear_norm_qk_ms;
                        fine.linear_ssm_ms += sub.linear_ssm_ms;
                        fine.linear_out_proj_ms += sub.linear_out_proj_ms;
                    } else {
                        fine.full_attn_ms += dt;
                        fine.full_attn_count += 1;
                    }
                }
                let h = mlx_rs::ops::add(&hidden_states, &attn)
                    .context("forward: residual add (post-attn) failed")?;

                let post_norm_w = match &layer_w.attn {
                    AttnLayerWeights::Linear(l) => &l.post_attention_layernorm,
                    AttnLayerWeights::Full(f) => &f.post_attention_layernorm,
                };
                let h_normed = rms_norm(&h, post_norm_w, cfg.rms_norm_eps)?;
                if timing_on {
                    reset_layer_sub_timings();
                }
                let t_moe = timing_on.then(Instant::now);
                let mlp_out = self.layer_moe_forward(&h_normed, layer_idx)?;
                if let Some(t0) = t_moe {
                    mlp_out.eval().context("forward: timing eval(moe) failed")?;
                    fine.moe_ms += t0.elapsed().as_secs_f64() * 1000.0;
                    fine.moe_count += 1;
                    let sub = drain_layer_sub_timings();
                    fine.moe_routing_ms += sub.moe_routing_ms;
                    fine.moe_switch_glu_ms += sub.moe_switch_glu_ms;
                    fine.moe_shared_expert_ms += sub.moe_shared_expert_ms;
                }
                hidden_states = mlx_rs::ops::add(&h, &mlp_out)
                    .context("forward: residual add (post-mlp) failed")?;

                // DFlash target-hidden capture: record the post-MLP residual
                // (i.e., this layer's output) for any requested target layer.
                // Mirrors `_LayerHook.__call__` in `dflash_runtime.py` which
                // records `out = self._layer(*args, **kwargs)` — that's the
                // DecoderLayer's return, which equals `h + self.mlp(...)` =
                // post-MLP residual. Push order matches `capture_layer_ids`
                // input order so the caller's `concat(axis=-1)` lines up with
                // the draft `fc` weight rows.
                for &cap_id in capture_layer_ids {
                    if cap_id == layer_idx {
                        captured.push(hidden_states.clone());
                    }
                }
            }

            // MTP capture: stash the post-MLP residual (pre-final-norm hidden)
            // when the MTP path is active. Consumed by `mtp_step` as the MTP
            // block's `h_pre` input. Cheap refcount clone — captured even when
            // the slot already holds a stale value (overwritten). The slot is
            // drained by `take_captured_h()` after the trunk forward returns.
            if self.mtp_capture_enabled.load(Ordering::Relaxed) {
                if let Ok(mut slot) = self.mtp_capture_slot.lock() {
                    *slot = Some(hidden_states.clone());
                }
            }

            // Final RMSNorm + lm_head projection. `tie_word_embeddings = false`
            // for the 35B production checkpoint; we still cover the tied path
            // for forward-compat with smaller variants.
            //
            // last_only short-circuit: when the caller only needs the next
            // token's logits (prefill / extend / decode_step), slice
            // hidden_states to position L-1 BEFORE the lm_head matmul. This
            // collapses lm_head from `[1, L, hidden] @ W^T → [1, L, vocab]`
            // to `[1, 1, hidden] @ W^T → [1, 1, vocab]`, saving
            // `(L-1) * hidden * vocab` FMAs. The next-token argmax sees an
            // identical scalar because we slice before the matmul, which is
            // a row-wise linear op — bit-identical with the old path's
            // `take_axis(L-1)` after the matmul.
            let hidden_for_head = if last_only && l > 1 {
                // 1-element index keeps axis 1 (shape [1, 1, hidden]) so the
                // final lm_head + argmax_last_token contract `ndim == 3` is
                // preserved. `Array::from_int` would drop the axis and
                // produce ndim=2, which the next-token argmax rejects.
                let idx_last = Array::from_slice(&[l - 1], &[1]);
                hidden_states
                    .take_axis(&idx_last, 1)
                    .context("forward: slice hidden_states to last position failed")?
            } else {
                hidden_states.clone()
            };
            let t_lm = timing_on.then(Instant::now);
            let normed_out = rms_norm(&hidden_for_head, &self.final_norm_weight, cfg.rms_norm_eps)?;
            let logits = if let Some(lm_head) = &self.lm_head {
                self.linear_pre(lm_head, &normed_out)?
            } else {
                // Tied path — fold the embed tensors directly.
                quantized_matmul_with_mode(
                    &normed_out,
                    &self.embed_weight,
                    &self.embed_scales,
                    None,
                    true,
                    self.embed_group,
                    self.embed_bits,
                    self.embed_mode,
                )
                .context("forward: tied lm_head matmul failed")?
            };
            let logits_f32 = Self::to_f32(logits, "forward.lm_head")?;
            if let Some(t0) = t_lm {
                logits_f32
                    .eval()
                    .context("forward: timing eval(lm_head) failed")?;
                fine.lm_head_ms = t0.elapsed().as_secs_f64() * 1000.0;
                store_fine_timings(fine);
            }
            Ok((logits_f32, captured))
        }

        /// Phase 1b (Stage 0): one batched decode forward over N single-token
        /// sequences, each with its own per-seq cache. Returns one argmax token
        /// per seq in input order.
        ///
        /// The weight-bound work — embed, RMSNorm, residuals, **MoE**, and
        /// **lm_head** — runs batched over `[N, 1, hidden]`, amortizing the
        /// dominant decode cost (expert/projection weight reads) across N tokens.
        /// **Linear-attn (Mamba) layers** are batched by stacking the N seqs'
        /// conv/ssm states along axis 0 into a transient cache and calling the
        /// existing (batch-native `gated_delta_step_kernel`) forward once, then
        /// splitting the updated states back. **Full-attn layers** run the
        /// existing single-seq `layer_full_attn_forward` per seq in a loop (each
        /// using its own KV cache / offset / RoPE) and stack the outputs — this
        /// is bit-identical to single-seq decode by construction (Stage 1 will
        /// batch the full-attn projections; the per-seq SDPA stays).
        ///
        /// `caches[i]` is sequence `i`'s cache; `caches.len() == tokens.len()`.
        /// N==1 is NOT special-cased here (the runner delegates N==1 to
        /// `decode_step` to keep that path byte-identical).
        pub fn forward_decode_batch(
            &self,
            tokens: &[u32],
            caches: &mut [&mut NativePromptCache],
        ) -> Result<Vec<u32>> {
            let n = tokens.len();
            if n == 0 {
                return Ok(Vec::new());
            }
            if caches.len() != n {
                return Err(anyhow!(
                    "forward_decode_batch: caches.len()={} != tokens.len()={n}",
                    caches.len()
                ));
            }
            let num_layers = self.num_layers();
            for (i, c) in caches.iter().enumerate() {
                if c.len() != num_layers {
                    return Err(anyhow!(
                        "forward_decode_batch: cache[{i}].len()={} != num_layers={num_layers}",
                        c.len()
                    ));
                }
            }
            let cfg = self.text_config();
            let hidden_dim = cfg.hidden_size as i32;
            let nn = n as i32;

            // Embedding: one token per seq → [N, hidden] → [N, 1, hidden].
            let ids_i32: Vec<i32> = tokens.iter().map(|&u| u as i32).collect();
            let ids_arr = Array::from_slice(&ids_i32, &[nn]);
            let embed_raw = quantized_embedding_lookup_with_mode(
                &self.embed_weight,
                &self.embed_scales,
                &ids_arr,
                self.embed_group,
                self.embed_bits,
                self.embed_mode,
            )
            .context("forward_decode_batch: embed lookup failed")?;
            let embed_f32 = Self::to_f32(embed_raw, "forward_decode_batch.embed")?;
            let mut hidden_states = mlx_rs::ops::reshape(&embed_f32, &[nn, 1, hidden_dim])
                .context("forward_decode_batch: reshape embed [N,1,hidden] failed")?;

            for layer_idx in 0..num_layers {
                let layer_w = &self.layer_weights[layer_idx];
                let input_norm_w = match &layer_w.attn {
                    AttnLayerWeights::Linear(l) => &l.input_layernorm,
                    AttnLayerWeights::Full(f) => &f.input_layernorm,
                };
                let normed = rms_norm(&hidden_states, input_norm_w, cfg.rms_norm_eps)?;

                let attn = if self.is_linear_per_layer[layer_idx] {
                    // Mamba: stack N per-seq states → transient → batched kernel → split back.
                    let nslots = caches[0]
                        .layer_mut(layer_idx)
                        .ok_or_else(|| anyhow!("forward_decode_batch: layer {layer_idx} missing"))?
                        .as_linear_mut()?
                        .len();
                    let mut transient = NativeArraysCache::new(nslots);
                    for slot in 0..nslots {
                        let mut parts: Vec<Array> = Vec::with_capacity(n);
                        for c in caches.iter_mut() {
                            let lin = c
                                .layer_mut(layer_idx)
                                .ok_or_else(|| {
                                    anyhow!("forward_decode_batch: layer {layer_idx} missing")
                                })?
                                .as_linear_mut()?;
                            let a = lin.get(slot).ok_or_else(|| {
                                anyhow!(
                                    "forward_decode_batch: linear layer {layer_idx} slot {slot} \
                                     empty (seq not prefilled?)"
                                )
                            })?;
                            parts.push(a.clone());
                        }
                        let refs: Vec<&Array> = parts.iter().collect();
                        let stacked = mlx_rs::ops::concatenate_axis(&refs, 0)
                            .context("forward_decode_batch: stack linear state failed")?;
                        transient.set(slot, stacked)?;
                    }
                    let out = self.layer_linear_attn_forward(&normed, layer_idx, &mut transient)?;
                    for slot in 0..nslots {
                        let updated = transient
                            .get(slot)
                            .ok_or_else(|| {
                                anyhow!("forward_decode_batch: transient slot {slot} empty")
                            })?
                            .clone();
                        for (i, c) in caches.iter_mut().enumerate() {
                            let idx_i = Array::from_slice(&[i as i32], &[1]);
                            let part = updated
                                .take_axis(&idx_i, 0)
                                .context("forward_decode_batch: split linear state failed")?;
                            c.layer_mut(layer_idx)
                                .unwrap()
                                .as_linear_mut()?
                                .set(slot, part)?;
                        }
                    }
                    for c in caches.iter_mut() {
                        c.layer_mut(layer_idx).unwrap().as_linear_mut()?.advance(1);
                    }
                    out
                } else {
                    // Full-attn. Plain (bf16) caches batch the q/k/v/o projections
                    // (Stage 1); TurboQuant caches fall back to per-seq whole-layer.
                    let is_tq = matches!(
                        caches[0].layer_mut(layer_idx).ok_or_else(|| anyhow!(
                            "forward_decode_batch: layer {layer_idx} missing"
                        ))?,
                        NativeLayerCache::FullTurboquant(_)
                    );
                    if is_tq {
                        let mut parts: Vec<Array> = Vec::with_capacity(n);
                        for (i, c) in caches.iter_mut().enumerate() {
                            let idx_i = Array::from_slice(&[i as i32], &[1]);
                            let normed_i = normed.take_axis(&idx_i, 0).context(
                                "forward_decode_batch: slice seq for TQ full-attn failed",
                            )?;
                            let tq = c.layer_mut(layer_idx).unwrap().as_full_turboquant_mut()?;
                            parts.push(
                                self.layer_full_attn_forward_tq(&normed_i, layer_idx, false, tq)?,
                            );
                        }
                        let refs: Vec<&Array> = parts.iter().collect();
                        mlx_rs::ops::concatenate_axis(&refs, 0)
                            .context("forward_decode_batch: stack TQ full-attn outputs failed")?
                    } else {
                        let mut layer_caches: Vec<&mut NativeLayerCache> = caches
                            .iter_mut()
                            .map(|c| {
                                c.layer_mut(layer_idx).ok_or_else(|| {
                                    anyhow!("forward_decode_batch: layer {layer_idx} missing")
                                })
                            })
                            .collect::<Result<_>>()?;
                        self.layer_full_attn_forward_batch(&normed, layer_idx, &mut layer_caches)?
                    }
                };

                let h = mlx_rs::ops::add(&hidden_states, &attn)
                    .context("forward_decode_batch: residual add (post-attn) failed")?;
                let post_norm_w = match &layer_w.attn {
                    AttnLayerWeights::Linear(l) => &l.post_attention_layernorm,
                    AttnLayerWeights::Full(f) => &f.post_attention_layernorm,
                };
                let h_normed = rms_norm(&h, post_norm_w, cfg.rms_norm_eps)?;
                let mlp_out = self.layer_moe_forward(&h_normed, layer_idx)?;
                hidden_states = mlx_rs::ops::add(&h, &mlp_out)
                    .context("forward_decode_batch: residual add (post-mlp) failed")?;
            }

            // Final norm + lm_head over [N, 1, hidden] → [N, 1, vocab], per-row argmax.
            let normed_out = rms_norm(&hidden_states, &self.final_norm_weight, cfg.rms_norm_eps)?;
            let logits = if let Some(lm_head) = &self.lm_head {
                self.linear_pre(lm_head, &normed_out)?
            } else {
                quantized_matmul_with_mode(
                    &normed_out,
                    &self.embed_weight,
                    &self.embed_scales,
                    None,
                    true,
                    self.embed_group,
                    self.embed_bits,
                    self.embed_mode,
                )
                .context("forward_decode_batch: tied lm_head matmul failed")?
            };
            let logits_f32 = Self::to_f32(logits, "forward_decode_batch.lm_head")?;
            self.argmax_rows(&logits_f32)
        }

        /// Per-row argmax over `[N, 1, vocab]` logits → `Vec<u32>` of length N.
        /// Single eval via `try_as_slice` (mirrors `argmax_last_token`'s tail).
        fn argmax_rows(&self, logits: &Array) -> Result<Vec<u32>> {
            if logits.ndim() != 3 {
                return Err(anyhow!(
                    "argmax_rows: expected [N, 1, V] logits, got ndim={}",
                    logits.ndim()
                ));
            }
            // Drop the L==1 axis → [N, V].
            let idx0 = Array::from_int(0);
            let rows = logits
                .take_axis(&idx0, 1)
                .context("argmax_rows: take_axis(0, axis=1) failed")?;
            // Argmax over vocab, keeping the batch axis → [N].
            let am = mlx_rs::ops::indexing::argmax_axis(&rows, -1, /* keep_dims */ false)
                .context("argmax_rows: argmax_axis(-1) failed")?;
            let slice = am
                .try_as_slice::<u32>()
                .map_err(|err| anyhow!("argmax_rows: read failed: {err}"))?;
            Ok(slice.to_vec())
        }

        // ─────────────────────────────────────────────────────────────────
        // MTP (Multi-Token Prediction) — speculative decode head wiring.
        // Mirrors `gemma4_moe::mtp_step` Steps A-E, retargeted for Qwen3.6's
        // hybrid (linear-attn + full-attn) trunk. The MTP block lives at
        // `qwen3_5_mtp::Qwen35MtpBlock` and is loaded separately (see
        // `qwen3_5_mtp::load_block_from_hf`).
        // ─────────────────────────────────────────────────────────────────

        /// Install a loaded MTP block onto this trunk. Validates that the
        /// block's static dims line up with the trunk's `text_config`. Idempotent
        /// in spirit — re-enabling replaces the prior block.
        pub fn try_enable_mtp(&mut self, block: Qwen35MtpBlock) -> Result<()> {
            let cfg = self.text_config();
            if block.dims.hidden_size != cfg.hidden_size {
                return Err(anyhow!(
                    "try_enable_mtp: block.hidden_size {} != trunk.hidden_size {}",
                    block.dims.hidden_size,
                    cfg.hidden_size
                ));
            }
            if block.dims.num_attention_heads != cfg.num_attention_heads {
                return Err(anyhow!(
                    "try_enable_mtp: block.num_attention_heads {} != trunk {}",
                    block.dims.num_attention_heads,
                    cfg.num_attention_heads
                ));
            }
            if block.dims.num_key_value_heads != cfg.num_key_value_heads {
                return Err(anyhow!(
                    "try_enable_mtp: block.num_key_value_heads {} != trunk {}",
                    block.dims.num_key_value_heads,
                    cfg.num_key_value_heads
                ));
            }
            if block.dims.head_dim != cfg.head_dim {
                return Err(anyhow!(
                    "try_enable_mtp: block.head_dim {} != trunk {}",
                    block.dims.head_dim,
                    cfg.head_dim
                ));
            }
            self.mtp = Some(block);
            Ok(())
        }

        /// True once `try_enable_mtp` has installed a drafter block.
        pub fn mtp_enabled(&self) -> bool {
            self.mtp.is_some()
        }

        fn set_mtp_capture_enabled(&self, on: bool) {
            self.mtp_capture_enabled.store(on, Ordering::Relaxed);
        }

        fn take_captured_h(&self) -> Option<Array> {
            self.mtp_capture_slot.lock().ok().and_then(|mut s| s.take())
        }

        /// Build a `Qwen35MtpLinear` view of the trunk's lm_head — direct
        /// reference (refcount bump) to the same packed/scales/biases tensors.
        /// Falls back to the tied-embed tensors when `lm_head` is None
        /// (matches the trunk forward path's tied lm_head branch).
        fn trunk_lm_head_view(&self) -> Qwen35MtpLinear {
            if let Some(lm) = &self.lm_head {
                Qwen35MtpLinear {
                    weight: lm.weight.clone(),
                    scales: lm.scales.clone(),
                    biases: lm.biases.clone(),
                    group_size: lm.group_size,
                    bits: lm.bits,
                    mode: lm.mode,
                }
            } else {
                Qwen35MtpLinear {
                    weight: self.embed_weight.clone(),
                    scales: self.embed_scales.clone(),
                    biases: None,
                    group_size: self.embed_group,
                    bits: self.embed_bits,
                    mode: self.embed_mode,
                }
            }
        }

        /// Look up the embedding for a single token id and reshape to
        /// `[1, 1, hidden]`. Mirrors the embed lookup at the head of
        /// `forward_impl` — same quant path, dtype cast, reshape — but as a
        /// standalone helper for the MTP drafter loop (Step B). No embed scale
        /// multiply (Qwen3.6's MTP convention; cf. gemma4 which multiplies by
        /// `sqrt(hidden_size)`).
        fn embed_lookup_single(&self, token_id: u32) -> Result<Array> {
            let ids_arr = Array::from_slice(&[token_id as i32], &[1]);
            let raw = quantized_embedding_lookup_with_mode(
                &self.embed_weight,
                &self.embed_scales,
                &ids_arr,
                self.embed_group,
                self.embed_bits,
                self.embed_mode,
            )
            .context("mtp_step: embed_lookup_single failed")?;
            let f32 = Self::to_f32(raw, "mtp_step.embed")?;
            let hidden = self.text_config().hidden_size as i32;
            mlx_rs::ops::reshape(&f32, &[1, 1, hidden])
                .context("mtp_step: embed reshape [1, 1, hidden] failed")
        }

        /// One MTP speculative step.
        ///
        /// Pre-state contract: `trunk_cache` is positioned at offset `M` —
        /// `committed_token` is NOT yet in the cache. `mtp_kv` is the MTP
        /// block's own KV cache (one full-attention layer); cleared at the
        /// start of every call (fresh K-step extrapolation, no cross-call
        /// drafter context).
        ///
        /// Steps:
        ///   A. Trunk decode `committed_token` (T=1) under `mtp_capture_enabled`.
        ///      Cache offset M → M+1. Captures the trunk's post-MLP residual
        ///      (pre-final-norm hidden) into `mtp_capture_slot`. Argmax of the
        ///      trunk's logits → `next_token`.
        ///   B. Drafter loop k=0..n_draft — `mtp_block.forward(embed(last_tok),
        ///      last_h, mtp_kv, false, trunk_final_norm, trunk_lm_head)`. The
        ///      block's own KV cache advances by 1 per iteration. `last_h`
        ///      threads the post-residual `new_h_pre` into the next iteration.
        ///   C. Trunk verify forward `[next_token, draft_0..draft_{K-1}]` at
        ///      S=K+1. Cache offset M+1 → M+K+2. Returns `[1, K+1, vocab]`
        ///      logits.
        ///   D. Accept-reject: argmax verify_logits per position → `preds`.
        ///      Sequentially accept while `preds[k] == drafts[k]`, break on
        ///      first mismatch.
        ///   E. On partial reject (n_accepted < K): restore the pre-Step-C
        ///      trunk-cache snapshot (back to M+1) and replay
        ///      `[next_token, drafts[..n_accepted]]` at S=1+n_accepted via the
        ///      regular trunk forward to land the cache at
        ///      target = M+1+1+n_accepted. The snapshot+replay path is
        ///      mandatory because the trunk's linear-attention (Mamba) layers
        ///      have non-truncatable SSM state — gemma4's `truncate_to`
        ///      shortcut works only for KV-based caches.
        ///
        /// Returns `MtpStepOutput { committed, n_attempted, n_accepted }`.
        /// The last element of `committed` is `preds[n_accepted]` — the
        /// trunk's prediction at the first non-accepted position, which is
        /// NOT in the cache yet and must be fed as `committed_token` to the
        /// next `mtp_step()` call.
        pub fn mtp_step(
            &self,
            trunk_cache: &mut NativePromptCache,
            mtp_kv: &mut NativeKvCache,
            committed_token: u32,
            n_draft: usize,
        ) -> Result<MtpStepOutput> {
            if n_draft == 0 {
                return Err(anyhow!("mtp_step: n_draft must be >= 1"));
            }
            let block = self
                .mtp
                .as_ref()
                .ok_or_else(|| anyhow!("mtp_step: MTP not enabled (call try_enable_mtp first)"))?;

            let m_pre = trunk_cache.full_attn_offset();

            // Reset the MTP block's own KV cache — every mtp_step is a fresh
            // K-step extrapolation. Cross-call attention context lives in the
            // TRUNK cache; the drafter only needs intra-step (draft-token)
            // attention which is what `mtp_kv` carries.
            mtp_kv.clear();

            // === Step A: trunk decode `committed_token` + capture last hidden ===
            self.set_mtp_capture_enabled(true);
            let logits_a = self
                .forward_with_opts(&[committed_token], trunk_cache, /* last_only */ true)
                .context("mtp_step: Step A trunk forward")?;
            self.set_mtp_capture_enabled(false);
            let trunk_h = self
                .take_captured_h()
                .ok_or_else(|| anyhow!("mtp_step: Step A failed to capture trunk last hidden"))?;
            let next_token = self
                .argmax_last_token(&logits_a)
                .context("mtp_step: Step A argmax next_token")?;
            // trunk_cache offset is now M+1.

            // === Step B: drafter loop ===
            let trunk_final_norm_w = self.final_norm_weight.clone();
            let trunk_lm_head = self.trunk_lm_head_view();
            let mut last_h = trunk_h;
            let mut last_tok = next_token;
            let mut drafts: Vec<u32> = Vec::with_capacity(n_draft);
            for k in 0..n_draft {
                let embeds = self
                    .embed_lookup_single(last_tok)
                    .with_context(|| format!("mtp_step: Step B embed k={k}"))?;
                let (logits_b, new_h) = block
                    .forward(
                        &embeds,
                        &last_h,
                        mtp_kv,
                        /* causal */ false,
                        &trunk_final_norm_w,
                        &trunk_lm_head,
                    )
                    .with_context(|| format!("mtp_step: Step B block.forward k={k}"))?;
                let draft_tok = self
                    .argmax_last_token(&logits_b)
                    .with_context(|| format!("mtp_step: Step B argmax k={k}"))?;
                drafts.push(draft_tok);
                last_h = new_h;
                last_tok = draft_tok;
            }
            // mtp_kv offset is now n_draft (drafter wrote one slot per draft).

            // Snapshot trunk_cache pre-Step-C. Shallow snapshot is cheap
            // (refcount bump per layer) and consumed once on partial reject.
            // Position recorded is `m_pre + 1` — the offset of the cache
            // RIGHT BEFORE the verify forward.
            let snap = PromptCacheSnapshot::capture_shallow(trunk_cache, m_pre + 1)
                .context("mtp_step: snapshot pre-Step-C failed")?;

            // === Step C: trunk verify [next_token, draft_0..draft_{K-1}] ===
            let mut verify_in: Vec<u32> = Vec::with_capacity(1 + n_draft);
            verify_in.push(next_token);
            verify_in.extend_from_slice(&drafts);
            let verify_logits = self
                .forward(&verify_in, trunk_cache)
                .context("mtp_step: Step C trunk verify forward")?;
            // trunk_cache offset is now M + 1 + (K + 1) = M + K + 2.

            // === Step D: accept-reject ===
            // Argmax along the vocab (last) axis → [1, K+1] u32 row of
            // trunk's true predictions per verify position.
            let argmax_per_pos =
                mlx_rs::ops::indexing::argmax_axis(&verify_logits, -1, /* keep_dims */ false)
                    .context("mtp_step: Step D argmax")?
                    .as_dtype(mlx_rs::Dtype::Int32)
                    .context("mtp_step: Step D cast to Int32")?;
            argmax_per_pos
                .eval()
                .context("mtp_step: Step D eval argmax")?;
            let preds_i32: &[i32] = argmax_per_pos.as_slice();
            let preds: Vec<u32> = preds_i32.iter().map(|x| *x as u32).collect();
            if std::env::var("LUMEN_MTP_DEBUG")
                .map(|v| v == "1")
                .unwrap_or(false)
            {
                eprintln!(
                    "[qwen3_5 mtp_step m_pre={m_pre}] next_token={next_token} drafts={drafts:?} preds={preds:?}"
                );
            }
            let mut n_accepted = 0usize;
            for k in 0..n_draft {
                if preds[k] == drafts[k] {
                    n_accepted += 1;
                } else {
                    break;
                }
            }

            // === Step E: rollback trunk_cache on partial reject ===
            //
            // Snapshot+replay (not gemma4's `truncate_to`) because Qwen3.6's
            // hybrid layer stack includes Mamba-style linear-attention layers
            // whose SSM state cannot be naively truncated after K+1 token
            // advances. Restoring the shallow snapshot reattaches the layers
            // to their pre-verify state; the replay forward feeds only the
            // accepted-trajectory tokens so the SSM state lands on the
            // committed history.
            //
            // Full accept (n_accepted == n_draft) skips both — the verify
            // forward already advanced the cache to the correct offset.
            let target_offset = m_pre + 1 + 1 + n_accepted; // committed + next + accepted drafts
            let cur_offset = trunk_cache.full_attn_offset();
            if cur_offset > target_offset {
                snap.restore_into(trunk_cache)
                    .context("mtp_step: Step E snapshot restore failed")?;
                // After restore, trunk_cache.full_attn_offset() == m_pre + 1.
                // Replay `[next_token, accepted_drafts..]` (length 1 + n_accepted)
                // to advance to target_offset = m_pre + 1 + 1 + n_accepted.
                let mut replay: Vec<u32> = Vec::with_capacity(1 + n_accepted);
                replay.push(next_token);
                replay.extend_from_slice(&drafts[..n_accepted]);
                self.forward_with_opts(&replay, trunk_cache, /* last_only */ true)
                    .context("mtp_step: Step E replay forward failed")?;
            }

            // Commit list — last element is `preds[n_accepted]`, the
            // correction/bonus token NOT in the cache. Caller threads it
            // into the next `mtp_step` call as `committed_token`.
            let mut committed = Vec::with_capacity(1 + n_accepted + 1);
            committed.push(next_token);
            committed.extend_from_slice(&drafts[..n_accepted]);
            committed.push(preds[n_accepted]);

            Ok(MtpStepOutput {
                committed,
                n_attempted: n_draft,
                n_accepted,
            })
        }

        /// Argmax of the last token's logits along the vocab axis. Wired for
        /// when `forward` lands; keeps the prefill / decode boundary clean.
        ///
        /// Mirrors `int(mx.argmax(logits[0, -1]).item())` from `mlx_runner.py`
        /// for a `[B=1, L, V]` logits tensor.
        pub fn argmax_last_token(&self, logits: &Array) -> Result<u32> {
            if logits.ndim() != 3 {
                return Err(anyhow!(
                    "argmax_last_token: expected [B, L, V] logits, got ndim={}",
                    logits.ndim()
                ));
            }
            let l = logits.shape()[1];
            // Mirror PyO3 `int(mx.argmax(logits[0, -1]).item())`: 2-op tail.
            // 1) take last row along axis=1 → [B=1, V]
            let idx_last = Array::from_int(l - 1);
            let last_row = logits
                .take_axis(&idx_last, 1)
                .context("argmax_last_token: take_axis(L-1) failed")?;
            // 2) full-reduce argmax over the [B=1, V] view → 0-d scalar (B=1 in runner)
            let scalar = mlx_rs::ops::indexing::argmax(&last_row, /* keep_dims */ false)
                .context("argmax_last_token: argmax (full reduce) failed")?;
            // Read scalar via try_as_slice (single eval). mlx-rs Array::try_item
            // calls eval() twice (mlx-rs-0.25.3/src/array/mod.rs:317-321), which
            // doubles command-buffer submits on every decode step. try_as_slice
            // evals once and returns &[u32] backed by the mapped GPU buffer.
            let slice = scalar
                .try_as_slice::<u32>()
                .map_err(|err| anyhow!("argmax_last_token: scalar read failed: {err}"))?;
            slice
                .first()
                .copied()
                .ok_or_else(|| anyhow!("argmax_last_token: argmax produced empty slice"))
        }
    }
}

#[cfg(feature = "mlx-native")]
#[allow(unused_imports)] // Consumed by runner_native.rs Phase 3d wiring.
pub(crate) use imp::{
    NativeLayerType, NativeModelConfig, NativeQuantizationConfig, NativeQuantizationOverride,
    NativeQwen3_5MoeModel, NativeTextConfig, NativeWeights,
};

// MTP types are part of the public surface (Phase 2 S3) — re-exported from
// the crate root via `lumen_mlx::MtpStepOutput`.
#[cfg(feature = "mlx-native")]
pub use imp::MtpStepOutput;

// ───────────────────────── unit tests ─────────────────────────
//
// Default-feature-build tests exercise pure-Rust paths (config parsing,
// sanitize key rewriting on a fake weight map). MLX FFI tests are gated by
// `feature = "mlx-native"` and `#[ignore]`'d per Issue #5.

#[cfg(all(test, feature = "mlx-native"))]
mod config_tests {
    use super::imp::{MlpKind, NativeLayerType, NativeModelConfig};

    #[test]
    fn parses_checked_in_qwen3_5_moe_config() {
        let raw = include_str!("../../lumen-model/tests/fixtures/qwen3_5_moe_config.json");
        let config: NativeModelConfig =
            serde_json::from_str(raw).expect("checked-in config.json must parse");
        config.validate_qwen3_5_moe().expect("contract must hold");
        let text = &config.text_config;
        assert_eq!(text.hidden_size, 2048);
        assert_eq!(text.head_dim, 256);
        assert_eq!(text.num_attention_heads, 16);
        assert_eq!(text.num_key_value_heads, 2);
        assert_eq!(text.num_hidden_layers, 40);
        assert_eq!(text.vocab_size, 248_320);
        assert_eq!(text.num_experts, 256);
        assert_eq!(text.num_experts_per_tok, 8);
        assert_eq!(text.full_attention_interval, 4);
        // 30 linear + 10 full alternating with interval=4
        let n_full = text
            .layer_types
            .iter()
            .filter(|t| matches!(t, NativeLayerType::FullAttention))
            .count();
        let n_linear = text.layer_types.len() - n_full;
        assert_eq!(n_full, 10);
        assert_eq!(n_linear, 30);
        assert_eq!(text.rope_dim(), 64); // 256 * 0.25
        assert_eq!(text.first_full_attn_layer(), Some(3));
    }

    #[test]
    fn rejects_wrong_model_type() {
        let bad = r#"{
            "model_type": "llama",
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "hidden_size": 16, "head_dim": 8,
                "num_attention_heads": 2, "num_key_value_heads": 1,
                "num_hidden_layers": 1, "vocab_size": 128,
                "rms_norm_eps": 1e-6, "layer_types": ["full_attention"]
            }
        }"#;
        let cfg: NativeModelConfig = serde_json::from_str(bad).unwrap();
        let err = cfg.validate_qwen3_5_moe().unwrap_err();
        assert!(err.to_string().contains("model_type"));
    }

    #[test]
    fn rejects_layer_type_count_mismatch() {
        let bad = r#"{
            "model_type": "qwen3_5_moe",
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "hidden_size": 16, "head_dim": 8,
                "num_attention_heads": 2, "num_key_value_heads": 1,
                "num_hidden_layers": 3, "vocab_size": 128,
                "rms_norm_eps": 1e-6, "layer_types": ["full_attention"]
            }
        }"#;
        let cfg: NativeModelConfig = serde_json::from_str(bad).unwrap();
        let err = cfg.validate_qwen3_5_moe().unwrap_err();
        assert!(err.to_string().contains("layer_types"));
    }

    #[test]
    fn family_validator_accepts_existing_moe_fixture_as_moe() {
        let raw = include_str!("../../lumen-model/tests/fixtures/qwen3_5_moe_config.json");
        let config: NativeModelConfig = serde_json::from_str(raw).unwrap();
        let kind = config
            .validate_qwen3_5_family()
            .expect("MoE fixture should pass family validator");
        assert_eq!(kind, MlpKind::Moe);
    }

    #[test]
    fn family_validator_accepts_dense_27b_style_config() {
        // Mirrors `mlx-community/Qwen3.6-27B-4bit/config.json`: model_type='qwen3_5',
        // text_config.model_type='qwen3_5_text', no MoE fields, dense intermediate_size,
        // affine 4-bit quantization (not mxfp4).
        let raw = r#"{
            "model_type": "qwen3_5",
            "eos_token_id": [248046, 248044],
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 5120,
                "head_dim": 256,
                "num_attention_heads": 24,
                "num_key_value_heads": 4,
                "num_hidden_layers": 4,
                "vocab_size": 248320,
                "rms_norm_eps": 1e-6,
                "full_attention_interval": 4,
                "layer_types": ["linear_attention","linear_attention","linear_attention","full_attention"],
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.25,
                "max_position_embeddings": 262144,
                "linear_num_value_heads": 48,
                "linear_num_key_heads": 16,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "intermediate_size": 17408
            },
            "quantization_config": {"group_size": 64, "bits": 4, "mode": "affine"}
        }"#;
        let config: NativeModelConfig = serde_json::from_str(raw).unwrap();
        let kind = config
            .validate_qwen3_5_family()
            .expect("dense 27B-style config should pass family validator");
        assert_eq!(kind, MlpKind::Dense);

        let text = &config.text_config;
        assert_eq!(text.intermediate_size, 17408);
        assert_eq!(text.num_experts, 0);
        assert_eq!(text.mlp_kind(), MlpKind::Dense);
    }

    #[test]
    fn family_validator_rejects_dense_with_zero_intermediate() {
        let bad = r#"{
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 16, "head_dim": 8,
                "num_attention_heads": 2, "num_key_value_heads": 1,
                "num_hidden_layers": 1, "vocab_size": 128,
                "rms_norm_eps": 1e-6,
                "layer_types": ["full_attention"]
            }
        }"#;
        let cfg: NativeModelConfig = serde_json::from_str(bad).unwrap();
        let err = cfg.validate_qwen3_5_family().unwrap_err();
        assert!(
            err.to_string().contains("intermediate_size"),
            "expected error mentioning intermediate_size, got: {err}"
        );
    }

    #[test]
    fn family_validator_rejects_unknown_model_type() {
        let bad = r#"{
            "model_type": "llama",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 16, "head_dim": 8,
                "num_attention_heads": 2, "num_key_value_heads": 1,
                "num_hidden_layers": 1, "vocab_size": 128,
                "rms_norm_eps": 1e-6,
                "intermediate_size": 32,
                "layer_types": ["full_attention"]
            }
        }"#;
        let cfg: NativeModelConfig = serde_json::from_str(bad).unwrap();
        let err = cfg.validate_qwen3_5_family().unwrap_err();
        assert!(err.to_string().contains("model_type"));
    }

    #[test]
    fn family_validator_accepts_affine_quantization() {
        let raw = r#"{
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 16, "head_dim": 8,
                "num_attention_heads": 2, "num_key_value_heads": 1,
                "num_hidden_layers": 1, "vocab_size": 128,
                "rms_norm_eps": 1e-6,
                "intermediate_size": 32,
                "layer_types": ["full_attention"]
            },
            "quantization_config": {"group_size": 64, "bits": 4, "mode": "affine"}
        }"#;
        let cfg: NativeModelConfig = serde_json::from_str(raw).unwrap();
        cfg.validate_qwen3_5_family()
            .expect("affine 4-bit quant should be accepted");
    }

    #[test]
    fn strict_moe_validator_still_rejects_dense_config() {
        // Regression: the MoE-only validator must still reject Dense even though
        // the family validator accepts it. Existing call sites depend on this.
        let dense = r#"{
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 16, "head_dim": 8,
                "num_attention_heads": 2, "num_key_value_heads": 1,
                "num_hidden_layers": 1, "vocab_size": 128,
                "rms_norm_eps": 1e-6,
                "intermediate_size": 32,
                "layer_types": ["full_attention"]
            }
        }"#;
        let cfg: NativeModelConfig = serde_json::from_str(dense).unwrap();
        let err = cfg.validate_qwen3_5_moe().unwrap_err();
        assert!(err.to_string().contains("qwen3_5_moe"));
    }
}
