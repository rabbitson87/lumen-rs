//! `config.json` of a Qwen3.8-Flash-Next (`qwen4_exp`) checkpoint.
//!
//! Only the text model is read. The fields mirror the reference implementation's
//! `TextArgs`; anything the port does not use (vision, MTP, mrope sections) is
//! ignored rather than validated.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use serde_json::Value;

lumen_flags::flag! {
    /// Keep Flash-Next's attention KV cache (and the sparse indexer's keys) in bf16 rather than
    /// f32: 30 KB per token of context instead of 60, so the memory plan reserves half as much
    /// for the longest prompt (+0.74 GB of expert pool at a 24 GB-class limit) and every retained
    /// conversation state (a session, an agent's system-prompt head, a resume point, a disk
    /// record) is half the size. Attention then runs in bf16 (the cast lands after the norm and
    /// RoPE, as `LUMEN_MLX_KV_BF16` does for Qwen 3.5); the indexer's keys are pooled in f32.
    ///
    /// **`Behavior`: a numerics change.** Teacher-forced over 98K positions of Korean, English and
    /// code (16K-token contexts), against f32: the same top token at 95.9% of positions, KL 0.0093
    /// nats, perplexity -0.05%, flat over depth (for scale, the gs64 and gs32 4-bit builds differ
    /// by KL 0.20). A 16K-token prefill is ~6% faster; decode reads half the KV bytes (~1% of a
    /// token there). States saved under one setting are never restored under the other (the disk
    /// identity includes it). `=0` restores f32.
    pub(crate) kv_bf16 {
        env: "LUMEN_QWEN4_KV_BF16",
        default: true,
        kind: Behavior,
    }
}

/// Bytes of one cached K, V or indexer-key element: 2 under [`kv_bf16`], else 4.
pub fn kv_elem_bytes() -> u64 {
    if kv_bf16::get() { 2 } else { 4 }
}

/// Which attention a decoder layer runs. Every layer also has a MoE block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Gated DeltaNet: a short convolution and a recurrent state, no KV cache.
    LinearAttention,
    /// Gated attention with the QSA block-sparse indexer over a KV cache.
    FullAttention,
}

/// How one tensor's weights are quantized. Every tensor is affine: the
/// checkpoint's `quantization` block carries a default and per-path overrides
/// (the router and the shared-expert gate are 8-bit, group 64).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantSpec {
    pub group_size: i32,
    pub bits: i32,
}

#[derive(Debug, Clone)]
pub struct QuantMap {
    default: QuantSpec,
    overrides: HashMap<String, QuantSpec>,
}

impl QuantMap {
    /// The default spec (what a tensor without an override gets).
    pub fn default_spec(&self) -> QuantSpec {
        self.default
    }

    /// Set the spec of the tensor at `<base>.weight`.
    pub fn set(&mut self, base: &str, spec: QuantSpec) {
        self.overrides.insert(base.to_string(), spec);
    }

    /// The spec for the tensor whose weights live at `<base>.weight`. Builds key
    /// their overrides either with or without the `language_model.` prefix.
    pub fn for_base(&self, base: &str) -> QuantSpec {
        self.overrides
            .get(base)
            .or_else(|| {
                base.strip_prefix("language_model.")
                    .and_then(|b| self.overrides.get(b))
            })
            .copied()
            .unwrap_or(self.default)
    }

    fn from_json(v: &Value) -> Result<Self> {
        let obj = v
            .as_object()
            .ok_or_else(|| anyhow!("config.json: `quantization` is not an object"))?;
        if let Some(mode) = obj.get("mode").and_then(Value::as_str)
            && mode != "affine"
        {
            bail!("qwen4_exp: quantization mode `{mode}` is not supported (only affine)");
        }
        let spec = |v: &Value, what: &str| -> Result<QuantSpec> {
            let get = |k: &str| {
                v.get(k)
                    .and_then(Value::as_i64)
                    .map(|n| n as i32)
                    .ok_or_else(|| anyhow!("config.json: {what} has no integer `{k}`"))
            };
            Ok(QuantSpec {
                group_size: get("group_size")?,
                bits: get("bits")?,
            })
        };
        let default = spec(v, "`quantization`")?;
        let mut overrides = HashMap::new();
        for (k, val) in obj {
            if val.is_object() {
                if let Some(mode) = val.get("mode").and_then(Value::as_str)
                    && mode != "affine"
                {
                    bail!("qwen4_exp: `{k}` uses quantization mode `{mode}` (only affine)");
                }
                overrides.insert(k.clone(), spec(val, k)?);
            }
        }
        Ok(Self { default, overrides })
    }
}

#[derive(Debug, Clone)]
pub struct Qwen4ExpConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub layer_kinds: Vec<LayerKind>,

    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,

    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,

    pub hc_count: usize,
    pub hc_lowrank: usize,

    pub indexer_n_heads: usize,
    pub indexer_kv_heads: usize,
    pub indexer_head_dim: usize,
    pub indexer_budget: usize,
    pub indexer_compress_ratio: usize,

    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    pub ngram_vocab_size_base: usize,
    pub ngram_vocab_divisible_by: usize,
    pub split_ngram_parts: usize,
    pub ple_embed_dim: usize,
    /// 1-based layer numbers after which the n-gram embedding is injected
    /// (`ple_layer_ids = [2]` means before layer index 1).
    pub ple_layer_ids: Vec<usize>,
    pub ple_conv_kernel_size: usize,

    pub partial_rotary_factor: f32,
    pub rope_theta: f32,
    /// The id the n-gram hashes treat as a document boundary (the first of `eos_tokens`).
    pub eos_token_id: u32,
    /// Every id that ends a reply: the config's `eos_token_id` (text and top level),
    /// `generation_config.json`'s, and the chat format's `<|im_end|>`.
    pub eos_tokens: Vec<u32>,
    pub tie_word_embeddings: bool,

    pub quant: QuantMap,

    /// The checkpoint stores RMSNorm weights as `w - 1` (the HF convention, where the
    /// layer multiplies by `1 + w`); the port multiplies by `w`, so these builds get
    /// `+ 1` at load. Decided from the weights, not the config: see the loader.
    pub norms_centered: bool,
}

/// Append the ids in `v` (a number or a list of numbers) that are not there yet.
fn collect_ids(v: Option<&Value>, out: &mut Vec<u32>) {
    let ids: Vec<u32> = match v {
        Some(Value::Number(n)) => n.as_u64().map(|n| n as u32).into_iter().collect(),
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(Value::as_u64)
            .map(|n| n as u32)
            .collect(),
        _ => Vec::new(),
    };
    for id in ids {
        if !out.contains(&id) {
            out.push(id);
        }
    }
}

impl Qwen4ExpConfig {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("config.json");
        let text =
            std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        let json: Value =
            serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
        let mut cfg = Self::from_json(&json)?;
        // The chat model stops on `<|im_end|>`, which only the generation config and the
        // tokenizer name; the text config lists just `<|endoftext|>`.
        let mut more = Vec::new();
        if let Ok(text) = std::fs::read_to_string(model_dir.join("generation_config.json"))
            && let Ok(gen_cfg) = serde_json::from_str::<Value>(&text)
        {
            collect_ids(gen_cfg.get("eos_token_id"), &mut more);
        }
        if let Ok(text) = std::fs::read_to_string(model_dir.join("tokenizer.json"))
            && let Ok(tok) = serde_json::from_str::<Value>(&text)
            && let Some(added) = tok.get("added_tokens").and_then(Value::as_array)
        {
            for t in added {
                if t.get("content").and_then(Value::as_str) == Some("<|im_end|>")
                    && let Some(id) = t.get("id").and_then(Value::as_u64)
                {
                    more.push(id as u32);
                }
            }
        }
        for id in more {
            if !cfg.eos_tokens.contains(&id) {
                cfg.eos_tokens.push(id);
            }
        }
        Ok(cfg)
    }

    pub fn from_json(root: &Value) -> Result<Self> {
        match root.get("model_type").and_then(Value::as_str) {
            Some("qwen4_exp") => {}
            other => bail!("config.json: model_type {other:?} is not `qwen4_exp`"),
        }
        let t = root
            .get("text_config")
            .ok_or_else(|| anyhow!("config.json has no `text_config`"))?;

        let usize_of = |k: &str| -> Result<usize> {
            t.get(k)
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .ok_or_else(|| anyhow!("text_config.{k} is missing or not a non-negative integer"))
        };
        let usize_or = |k: &str, default: usize| -> usize {
            t.get(k)
                .and_then(Value::as_u64)
                .map_or(default, |n| n as usize)
        };

        let num_hidden_layers = usize_of("num_hidden_layers")?;
        let interval = usize_or("full_attention_interval", 4).max(1);
        let layer_kinds: Vec<LayerKind> = match t.get("layer_types").and_then(Value::as_array) {
            Some(a) if !a.is_empty() => a
                .iter()
                .map(|v| match v.as_str() {
                    Some("linear_attention") => Ok(LayerKind::LinearAttention),
                    Some("full_attention") => Ok(LayerKind::FullAttention),
                    other => Err(anyhow!("text_config.layer_types has {other:?}")),
                })
                .collect::<Result<_>>()?,
            _ => (0..num_hidden_layers)
                .map(|i| {
                    if (i + 1) % interval == 0 {
                        LayerKind::FullAttention
                    } else {
                        LayerKind::LinearAttention
                    }
                })
                .collect(),
        };
        if layer_kinds.len() != num_hidden_layers {
            bail!(
                "text_config: {} layer_types for {num_hidden_layers} layers",
                layer_kinds.len()
            );
        }

        let rope = t.get("rope_parameters");
        let rope_f = |k: &str, default: f64| -> f32 {
            rope.and_then(|r| r.get(k))
                .and_then(Value::as_f64)
                .or_else(|| t.get(k).and_then(Value::as_f64))
                .unwrap_or(default) as f32
        };

        let mut eos_tokens = Vec::new();
        collect_ids(t.get("eos_token_id"), &mut eos_tokens);
        collect_ids(root.get("eos_token_id"), &mut eos_tokens);
        collect_ids(
            root.get("generation_config")
                .and_then(|g| g.get("eos_token_id")),
            &mut eos_tokens,
        );
        let eos_token_id = *eos_tokens
            .first()
            .ok_or_else(|| anyhow!("config.json has no usable eos_token_id"))?;

        let quant = QuantMap::from_json(
            root.get("quantization")
                .or_else(|| root.get("quantization_config"))
                .ok_or_else(|| anyhow!("config.json has no `quantization` block (float checkpoints are not supported)"))?,
        )?;

        let cfg = Self {
            hidden_size: usize_of("hidden_size")?,
            num_hidden_layers,
            num_attention_heads: usize_of("num_attention_heads")?,
            num_key_value_heads: usize_of("num_key_value_heads")?,
            head_dim: usize_of("head_dim")?,
            vocab_size: usize_of("vocab_size")?,
            rms_norm_eps: t
                .get("rms_norm_eps")
                .and_then(Value::as_f64)
                .unwrap_or(1e-6) as f32,
            layer_kinds,
            num_experts: usize_of("num_experts")?,
            num_experts_per_tok: usize_of("num_experts_per_tok")?,
            moe_intermediate_size: usize_of("moe_intermediate_size")?,
            shared_expert_intermediate_size: usize_of("shared_expert_intermediate_size")?,
            linear_num_key_heads: usize_of("linear_num_key_heads")?,
            linear_num_value_heads: usize_of("linear_num_value_heads")?,
            linear_key_head_dim: usize_of("linear_key_head_dim")?,
            linear_value_head_dim: usize_of("linear_value_head_dim")?,
            linear_conv_kernel_dim: usize_of("linear_conv_kernel_dim")?,
            hc_count: usize_of("hc_count")?,
            hc_lowrank: usize_of("hc_lowrank")?,
            indexer_n_heads: usize_of("indexer_n_heads")?,
            indexer_kv_heads: usize_of("indexer_kv_heads")?,
            indexer_head_dim: usize_of("indexer_head_dim")?,
            indexer_budget: usize_of("indexer_budget")?,
            indexer_compress_ratio: usize_of("indexer_compress_ratio")?,
            ngram_size: usize_of("ngram_size")?,
            heads_per_ngram: usize_of("heads_per_ngram")?,
            ngram_vocab_size_base: usize_of("ngram_vocab_size_base")?,
            ngram_vocab_divisible_by: usize_or("make_ngram_vocab_size_divisible_by", 128),
            split_ngram_parts: usize_of("split_ngram_parts").unwrap_or(128),
            ple_embed_dim: usize_of("ple_embed_dim")?,
            ple_layer_ids: t
                .get("ple_layer_ids")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(Value::as_u64)
                        .map(|n| n as usize)
                        .collect()
                })
                .unwrap_or_else(|| vec![2]),
            ple_conv_kernel_size: usize_of("ple_conv_kernel_size")?,
            partial_rotary_factor: rope_f("partial_rotary_factor", 0.25),
            rope_theta: rope_f("rope_theta", 10_000_000.0),
            eos_token_id,
            eos_tokens,
            tie_word_embeddings: t
                .get("tie_word_embeddings")
                .or_else(|| root.get("tie_word_embeddings"))
                .and_then(Value::as_bool)
                .unwrap_or(false),
            quant,
            norms_centered: false,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Dimensions the port relies on, checked once so the forward pass can
    /// index without re-deriving them.
    fn validate(&self) -> Result<()> {
        let bad = |what: &str| -> Result<()> { bail!("qwen4_exp config: {what}") };
        if self.hc_count == 0 {
            return bad("hc_count is 0");
        }
        if self.linear_num_key_heads == 0
            || !self
                .linear_num_value_heads
                .is_multiple_of(self.linear_num_key_heads)
        {
            return bad("linear value heads must be a multiple of key heads");
        }
        if self.num_attention_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads.max(1))
        {
            return bad("attention heads must be a multiple of kv heads");
        }
        if self.indexer_kv_heads != 1 {
            return bad("only one indexer kv head is supported");
        }
        if self.indexer_compress_ratio == 0 || self.indexer_budget < self.indexer_compress_ratio {
            return bad("indexer budget / compress ratio");
        }
        if self.rotary_dim() > self.head_dim || self.rotary_dim() > self.indexer_head_dim {
            return bad("rotary dim exceeds a head dim");
        }
        let heads = self.ngram_heads();
        if heads == 0 || !self.ple_embed_dim.is_multiple_of(heads) {
            return bad("ple_embed_dim must be a multiple of the n-gram head count");
        }
        if self.num_experts_per_tok == 0 || self.num_experts_per_tok > self.num_experts {
            return bad("num_experts_per_tok");
        }
        Ok(())
    }

    /// Bytes of cache one token of context takes: K and V of every full-attention layer and the
    /// indexer's key, plus the same for the draft head's one layer, at [`kv_elem_bytes`] each.
    pub fn kv_bytes_per_token(&self) -> u64 {
        let full_layers = self
            .layer_kinds
            .iter()
            .filter(|k| **k == LayerKind::FullAttention)
            .count() as u64
            + 1;
        let per_layer =
            (2 * self.num_key_value_heads * self.head_dim + self.indexer_head_dim) as u64;
        full_layers * per_layer * kv_elem_bytes()
    }

    /// Bytes of state a sequence holds whatever its length: each DeltaNet layer's recurrent state
    /// (f32, a key-dim by value-dim matrix per value head) and its convolution window.
    pub fn state_bytes(&self) -> u64 {
        let layers = self
            .layer_kinds
            .iter()
            .filter(|k| **k == LayerKind::LinearAttention)
            .count() as u64;
        let recurrent =
            self.linear_num_value_heads * self.linear_key_head_dim * self.linear_value_head_dim * 4;
        let channels = 2 * self.linear_num_key_heads * self.linear_key_head_dim
            + self.linear_num_value_heads * self.linear_value_head_dim;
        let window = self.linear_conv_kernel_dim.saturating_sub(1) * channels * 4;
        layers * (recurrent + window) as u64
    }

    /// Bytes of one routed expert as the offload engine reads it: its gate, up and down
    /// projections at the experts' quantization, with a bf16 scale and bias per group (2.76 MB on
    /// Flash-Next: 2,560 x 640, 4-bit, groups of 64).
    pub fn routed_expert_bytes(&self) -> u64 {
        let q = self
            .quant
            .for_base("language_model.model.layers.0.mlp.switch_mlp");
        let weights = (self.hidden_size * self.moe_intermediate_size) as u64;
        let groups = weights / q.group_size.max(1) as u64;
        3 * (weights * q.bits.max(0) as u64 / 8 + groups * 2 * 2)
    }

    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f32 * self.partial_rotary_factor) as usize
    }

    /// Hash heads of the n-gram table: `(ngram_size - 1) * heads_per_ngram`.
    pub fn ngram_heads(&self) -> usize {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    pub fn key_dim(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim
    }

    pub fn value_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    pub fn conv_dim(&self) -> usize {
        self.key_dim() * 2 + self.value_dim()
    }

    /// Width of the hyper-connected residual stream: `hc_count * hidden_size`.
    pub fn stream_dim(&self) -> usize {
        self.hc_count * self.hidden_size
    }

    /// The layer index that follows each 1-based id of `ple_layer_ids`.
    pub fn ple_layers(&self) -> Vec<usize> {
        self.ple_layer_ids
            .iter()
            .filter(|&&id| id >= 1 && id <= self.num_hidden_layers)
            .map(|&id| id - 1)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_json() -> Value {
        serde_json::json!({
            "model_type": "qwen4_exp",
            "quantization": {"group_size": 32, "bits": 4, "mode": "affine",
                "language_model.model.layers.0.mlp.gate": {"group_size": 64, "bits": 8, "mode": "affine"}},
            "text_config": {
                "hidden_size": 64, "num_hidden_layers": 8, "num_attention_heads": 4,
                "num_key_value_heads": 2, "head_dim": 32, "vocab_size": 128,
                "num_experts": 16, "num_experts_per_tok": 4, "moe_intermediate_size": 64,
                "shared_expert_intermediate_size": 64,
                "linear_num_key_heads": 2, "linear_num_value_heads": 4,
                "linear_key_head_dim": 32, "linear_value_head_dim": 32, "linear_conv_kernel_dim": 4,
                "hc_count": 4, "hc_lowrank": 32,
                "indexer_n_heads": 2, "indexer_kv_heads": 1, "indexer_head_dim": 32,
                "indexer_budget": 16, "indexer_compress_ratio": 4,
                "ngram_size": 3, "heads_per_ngram": 2, "ngram_vocab_size_base": 1000,
                "ple_embed_dim": 128, "ple_conv_kernel_size": 4, "ple_layer_ids": [2],
                "eos_token_id": 7,
                "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.25}
            }
        })
    }

    #[test]
    fn a_tiny_config_parses_and_derives_the_dimensions_the_forward_pass_needs() {
        let cfg = Qwen4ExpConfig::from_json(&tiny_json()).unwrap();
        assert_eq!(cfg.num_hidden_layers, 8);
        assert_eq!(cfg.rotary_dim(), 8);
        assert_eq!(cfg.ngram_heads(), 4);
        assert_eq!(cfg.conv_dim(), 2 * 64 + 128);
        assert_eq!(cfg.stream_dim(), 256);
        assert_eq!(cfg.ple_layers(), vec![1]);
        // No `layer_types`: every fourth layer is full attention.
        let kinds: Vec<_> = cfg
            .layer_kinds
            .iter()
            .map(|k| *k == LayerKind::FullAttention)
            .collect();
        assert_eq!(
            kinds,
            [false, false, false, true, false, false, false, true]
        );
        assert_eq!(cfg.eos_token_id, 7);
    }

    #[test]
    fn quantization_overrides_apply_by_base_path() {
        let cfg = Qwen4ExpConfig::from_json(&tiny_json()).unwrap();
        let gate = cfg.quant.for_base("language_model.model.layers.0.mlp.gate");
        assert_eq!((gate.group_size, gate.bits), (64, 8));
        let other = cfg
            .quant
            .for_base("language_model.model.layers.0.mlp.shared_expert.up_proj");
        assert_eq!((other.group_size, other.bits), (32, 4));
    }

    #[test]
    fn a_routed_expert_is_three_quantized_projections_with_a_scale_and_bias_per_group() {
        let mut cfg = Qwen4ExpConfig::from_json(&tiny_json()).unwrap();
        // 64 x 64 at 4 bits in groups of 32: 2,048 bytes of weights and 128 groups a projection.
        assert_eq!(cfg.routed_expert_bytes(), 3 * (2_048 + 128 * 4));
        // Flash-Next: 2,560 x 640, 4-bit, groups of 64, as the expert bank reads it (2.76 MB).
        cfg.hidden_size = 2_560;
        cfg.moe_intermediate_size = 640;
        cfg.quant.set(
            "language_model.model.layers.0.mlp.switch_mlp",
            QuantSpec {
                group_size: 64,
                bits: 4,
            },
        );
        assert_eq!(cfg.routed_expert_bytes(), 2_764_800);
    }

    #[test]
    fn overrides_keyed_without_the_language_model_prefix_apply_too() {
        let mut j = tiny_json();
        j["quantization"]["model.layers.2.mlp.gate"] =
            serde_json::json!({"group_size": 128, "bits": 2, "mode": "affine"});
        let cfg = Qwen4ExpConfig::from_json(&j).unwrap();
        let q = cfg.quant.for_base("language_model.model.layers.2.mlp.gate");
        assert_eq!((q.group_size, q.bits), (128, 2));
    }

    #[test]
    fn a_different_model_type_or_a_float_checkpoint_is_rejected() {
        let mut j = tiny_json();
        j["model_type"] = Value::from("qwen3_5_moe");
        assert!(Qwen4ExpConfig::from_json(&j).is_err());
        let mut j = tiny_json();
        j.as_object_mut().unwrap().remove("quantization");
        assert!(Qwen4ExpConfig::from_json(&j).is_err());
        let mut j = tiny_json();
        j["quantization"]["mode"] = Value::from("mxfp4");
        assert!(Qwen4ExpConfig::from_json(&j).is_err());
    }

    #[test]
    fn inconsistent_dimensions_are_reported_not_trusted() {
        let mut j = tiny_json();
        j["text_config"]["linear_num_value_heads"] = Value::from(5);
        assert!(Qwen4ExpConfig::from_json(&j).is_err());
        let mut j = tiny_json();
        j["text_config"]["ple_embed_dim"] = Value::from(130);
        assert!(Qwen4ExpConfig::from_json(&j).is_err());
    }
}
