//! Weight manifest parsing and semantic classification for Qwen3.5-VL-MoE MXFP4 checkpoints.
//!
//! Responsibilities (no shard I/O yet — that lands with Stage 2-f):
//!   1. Parse `model.safetensors.index.json` into a flat `tensor_name → shard_file` map
//!   2. Group MLX's 2-or-3-tensor-per-weight layout (`.weight/.scales[/.biases]`) into logical
//!      weight groups keyed by their prefix
//!   3. Detect each group's storage kind (MXFP4 / int8-affine / plain BF16) from which suffixes
//!      are present and cross-validate against [`QuantizationConfig::for_weight`]
//!   4. Assign every prefix a [`WeightRole`] and verify the full 40-layer roster is present with
//!      the exact expected sub-weights for each layer's [`LayerType`]
//!
//! Invariants checked by [`Classification::build`] (any violation → typed error):
//!   - Every tensor name reachable from index.json maps to some role; no unknowns tolerated
//!   - Every group's detected storage kind matches what `config.quantization_config.for_weight`
//!     predicts from `.bits` (4 → MXFP4, 8 → int8-affine, no entry → plain)
//!   - Each of the 40 text layers owns exactly the sub-weights expected for its declared
//!     [`LayerType`] (linear-attn vs full-attn have disjoint sub-modules)
//!   - `gate` and `shared_expert_gate` appear in all 40 layers as int8-affine overrides
//!   - `vision_tower.*` tensors are all plain (no scales/biases)
//!
//! This module does **not** yet read tensor payloads; it works against the ~165 KB index
//! descriptor alone so tests run without the 19 GB shard download.

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;

use super::config::{LayerType, Qwen3_5MoeConfig};

// ─────────────────────────────────────────────────────────────────────────────
// Raw index parsing
// ─────────────────────────────────────────────────────────────────────────────

/// Deserialized shape of `model.safetensors.index.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct WeightIndex {
    pub metadata: IndexMetadata,
    pub weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct IndexMetadata {
    pub total_size: u64,
}

impl WeightIndex {
    pub fn from_json_str(s: &str) -> Result<Self, IndexError> {
        serde_json::from_str(s).map_err(IndexError::Parse)
    }

    /// Sorted unique shard filenames referenced by the index.
    pub fn shards(&self) -> Vec<String> {
        let mut v: Vec<String> = self.weight_map.values().cloned().collect();
        v.sort();
        v.dedup();
        v
    }

    /// Group tensor names by their prefix (strip `.weight/.scales/.biases`) and detect storage
    /// kind from which suffixes are present. Every tensor must fall into exactly one group; any
    /// unrecognized suffix or cross-shard split within a single group is an error.
    pub fn groups(&self) -> Result<Vec<WeightGroup>, IndexError> {
        let mut buckets: BTreeMap<String, PartialGroup> = BTreeMap::new();
        for (tensor, shard) in &self.weight_map {
            let (prefix, suffix) = split_suffix(tensor);
            let entry = buckets
                .entry(prefix.to_string())
                .or_insert_with(|| PartialGroup {
                    shard: shard.clone(),
                    weight: false,
                    scales: false,
                    biases: false,
                });
            if entry.shard != *shard {
                return Err(IndexError::GroupSpansShards {
                    prefix: prefix.to_string(),
                    first: entry.shard.clone(),
                    second: shard.clone(),
                });
            }
            match suffix {
                Suffix::Weight => entry.weight = true,
                Suffix::Scales => entry.scales = true,
                Suffix::Biases => entry.biases = true,
            }
        }

        let mut out = Vec::with_capacity(buckets.len());
        for (prefix, p) in buckets {
            let kind = match (p.weight, p.scales, p.biases) {
                (true, false, false) => StorageKind::Plain,
                (true, true, false) => StorageKind::Mxfp4,
                (true, true, true) => StorageKind::Int8Affine,
                (has_w, has_s, has_b) => {
                    return Err(IndexError::UnexpectedSuffixCombination {
                        prefix,
                        weight: has_w,
                        scales: has_s,
                        biases: has_b,
                    });
                }
            };
            out.push(WeightGroup {
                prefix,
                kind,
                shard: p.shard,
            });
        }
        Ok(out)
    }
}

struct PartialGroup {
    shard: String,
    weight: bool,
    scales: bool,
    biases: bool,
}

#[derive(Copy, Clone, Debug)]
enum Suffix {
    Weight,
    Scales,
    Biases,
}

/// Strip the MLX storage suffix. A tensor name like `.linear_attn.A_log` has no suffix;
/// it's treated as a plain weight whose prefix is the full name. Parameters that MLX stores
/// without a `.weight` suffix (SSM state bits: `A_log`, `dt_bias`, `conv1d.weight` which does
/// carry `.weight` literally) end up grouped by their own full name.
fn split_suffix(tensor: &str) -> (&str, Suffix) {
    if let Some(stem) = tensor.strip_suffix(".biases") {
        (stem, Suffix::Biases)
    } else if let Some(stem) = tensor.strip_suffix(".scales") {
        (stem, Suffix::Scales)
    } else if let Some(stem) = tensor.strip_suffix(".weight") {
        (stem, Suffix::Weight)
    } else {
        // No recognized suffix — treat the whole name as its own "weight" group
        // (e.g., `linear_attn.A_log`, `linear_attn.dt_bias`).
        (tensor, Suffix::Weight)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Logical grouping
// ─────────────────────────────────────────────────────────────────────────────

/// A logical weight unit — one prefix, one shard, one storage kind, 1–3 underlying tensors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightGroup {
    pub prefix: String,
    pub kind: StorageKind,
    pub shard: String,
}

/// How a weight group is stored on disk. Derived from the suffix pattern in the index:
///   - `Plain`      : `.weight` only (BF16/FP32, no quantization)
///   - `Mxfp4`      : `.weight + .scales` (4-bit E2M1 + E8M0 scale per 32-element group)
///   - `Int8Affine` : `.weight + .scales + .biases` (8-bit affine with zero-point)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageKind {
    Plain,
    Mxfp4,
    Int8Affine,
}

impl StorageKind {
    pub fn is_quantized(self) -> bool {
        !matches!(self, Self::Plain)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Semantic classification
// ─────────────────────────────────────────────────────────────────────────────

/// Semantic role for a single weight group. Exhaustive over the 1,658 tensors in the
/// MLX MXFP4 checkpoint; any prefix not matching one of these arms is an error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeightRole {
    Embed,
    LmHead,
    FinalNorm,
    InputLayerNorm(usize),
    PostAttentionLayerNorm(usize),
    LinearAttn {
        layer: usize,
        part: LinearAttnPart,
    },
    SelfAttn {
        layer: usize,
        part: SelfAttnPart,
    },
    Mlp {
        layer: usize,
        part: MlpPart,
    },
    /// Vision tower weights (ViT backbone + patch embed + merger). Flat name preserved so the
    /// vision loader can re-parse without re-classifying.
    Vision(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum LinearAttnPart {
    /// SSM state decay parameter (plain fp32).
    ALog,
    /// Causal 1D convolution over the input sequence.
    Conv1d,
    /// Learned bias on the delta-t projection.
    DtBias,
    /// Internal RMSNorm inside the SSM block.
    Norm,
    InProjA,
    InProjB,
    InProjQkv,
    InProjZ,
    OutProj,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SelfAttnPart {
    QNorm,
    KNorm,
    QProj,
    KProj,
    VProj,
    OProj,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MlpPart {
    /// Router logit projection (int8-affine per quantization_config override).
    /// MoE-only.
    Gate,
    /// Shared-expert gate (int8-affine override). MoE-only.
    SharedExpertGate,
    /// Shared-expert SwiGLU projection. MoE-only.
    SharedExpert(ProjKind),
    /// Grouped per-expert SwiGLU projection (`switch_mlp` stores all 256 experts as one tensor
    /// with leading expert axis). MoE-only.
    SwitchMlp(ProjKind),
    /// Standard SwiGLU projection on a Dense MLP layer (Qwen3.6-27B / Qwen3.5-4B). The three
    /// projections (gate/up/down) are stored at `language_model.model.layers.{N}.mlp.{X}_proj`
    /// without any router or shared-expert split.
    Dense(ProjKind),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ProjKind {
    Gate,
    Up,
    Down,
}

impl ProjKind {
    fn from_segment(seg: &str) -> Option<Self> {
        match seg {
            "gate_proj" => Some(Self::Gate),
            "up_proj" => Some(Self::Up),
            "down_proj" => Some(Self::Down),
            _ => None,
        }
    }
}

/// Classify a single prefix. Returns `None` if the prefix doesn't match any known pattern —
/// upstream validation promotes `None` into [`ClassifyError::UnknownPrefix`].
fn classify_prefix(prefix: &str) -> Option<WeightRole> {
    if let Some(rest) = prefix.strip_prefix("vision_tower.") {
        return Some(WeightRole::Vision(rest.to_string()));
    }
    let rest = prefix.strip_prefix("language_model.")?;

    match rest {
        "lm_head" => return Some(WeightRole::LmHead),
        "model.embed_tokens" => return Some(WeightRole::Embed),
        "model.norm" => return Some(WeightRole::FinalNorm),
        _ => {}
    }

    let layer_body = rest.strip_prefix("model.layers.")?;
    let (idx_str, tail) = layer_body.split_once('.')?;
    let layer: usize = idx_str.parse().ok()?;

    match tail {
        "input_layernorm" => return Some(WeightRole::InputLayerNorm(layer)),
        "post_attention_layernorm" => return Some(WeightRole::PostAttentionLayerNorm(layer)),
        _ => {}
    }

    if let Some(rest) = tail.strip_prefix("linear_attn.") {
        let part = match rest {
            "A_log" => LinearAttnPart::ALog,
            "conv1d" => LinearAttnPart::Conv1d,
            "dt_bias" => LinearAttnPart::DtBias,
            "norm" => LinearAttnPart::Norm,
            "in_proj_a" => LinearAttnPart::InProjA,
            "in_proj_b" => LinearAttnPart::InProjB,
            "in_proj_qkv" => LinearAttnPart::InProjQkv,
            "in_proj_z" => LinearAttnPart::InProjZ,
            "out_proj" => LinearAttnPart::OutProj,
            _ => return None,
        };
        return Some(WeightRole::LinearAttn { layer, part });
    }

    if let Some(rest) = tail.strip_prefix("self_attn.") {
        let part = match rest {
            "q_norm" => SelfAttnPart::QNorm,
            "k_norm" => SelfAttnPart::KNorm,
            "q_proj" => SelfAttnPart::QProj,
            "k_proj" => SelfAttnPart::KProj,
            "v_proj" => SelfAttnPart::VProj,
            "o_proj" => SelfAttnPart::OProj,
            _ => return None,
        };
        return Some(WeightRole::SelfAttn { layer, part });
    }

    if let Some(rest) = tail.strip_prefix("mlp.") {
        let part = match rest {
            "gate" => MlpPart::Gate,
            "shared_expert_gate" => MlpPart::SharedExpertGate,
            // Dense SwiGLU triple at `mlp.{gate,up,down}_proj` — distinguishes the
            // 27B/4B Dense layout from the 35B-A3B MoE layout (which nests these
            // inside `shared_expert.` or `switch_mlp.`).
            "gate_proj" => MlpPart::Dense(ProjKind::Gate),
            "up_proj" => MlpPart::Dense(ProjKind::Up),
            "down_proj" => MlpPart::Dense(ProjKind::Down),
            _ => {
                if let Some(proj) = rest.strip_prefix("shared_expert.") {
                    MlpPart::SharedExpert(ProjKind::from_segment(proj)?)
                } else if let Some(proj) = rest.strip_prefix("switch_mlp.") {
                    MlpPart::SwitchMlp(ProjKind::from_segment(proj)?)
                } else {
                    return None;
                }
            }
        };
        return Some(WeightRole::Mlp { layer, part });
    }

    None
}

// ─────────────────────────────────────────────────────────────────────────────
// Full classification + validation
// ─────────────────────────────────────────────────────────────────────────────

/// Fully classified and config-cross-validated weight manifest.
#[derive(Debug, Clone)]
pub struct Classification {
    pub groups: Vec<ClassifiedGroup>,
    /// Layer-indexed inventory: `per_layer[i]` is the set of sub-roles observed for layer `i`.
    pub per_layer: Vec<LayerInventory>,
    pub vision_count: usize,
}

#[derive(Debug, Clone)]
pub struct ClassifiedGroup {
    pub prefix: String,
    pub shard: String,
    pub storage: StorageKind,
    pub role: WeightRole,
}

/// Per-layer sub-role roster. The classifier populates this and then checks it against the
/// layer's declared [`LayerType`]; mismatches become [`ClassifyError`]s.
#[derive(Debug, Clone, Default)]
pub struct LayerInventory {
    pub has_input_ln: bool,
    pub has_post_attn_ln: bool,
    pub linear_parts: BTreeSet<LinearAttnPart>,
    pub self_parts: BTreeSet<SelfAttnPart>,
    pub mlp_parts: BTreeSet<MlpPart>,
}

impl Classification {
    /// Build a classified view from `index` and cross-check against `config`.
    pub fn build(
        index: &WeightIndex,
        config: &Qwen3_5MoeConfig,
    ) -> Result<Self, ClassifyError> {
        let groups = index.groups().map_err(ClassifyError::Index)?;
        let n_layers = config.text_config.num_hidden_layers;

        let mut classified = Vec::with_capacity(groups.len());
        let mut per_layer = vec![LayerInventory::default(); n_layers];
        let mut vision_count = 0usize;

        for g in groups {
            let role = classify_prefix(&g.prefix)
                .ok_or_else(|| ClassifyError::UnknownPrefix(g.prefix.clone()))?;

            // Storage kind must match quantization_config's prediction for this prefix.
            let expected = expected_storage(&g.prefix, config);
            if expected != g.kind {
                return Err(ClassifyError::StorageMismatch {
                    prefix: g.prefix.clone(),
                    expected,
                    found: g.kind,
                });
            }

            match &role {
                WeightRole::Vision(_) => vision_count += 1,
                WeightRole::InputLayerNorm(i) => {
                    check_layer_idx(*i, n_layers)?;
                    per_layer[*i].has_input_ln = true;
                }
                WeightRole::PostAttentionLayerNorm(i) => {
                    check_layer_idx(*i, n_layers)?;
                    per_layer[*i].has_post_attn_ln = true;
                }
                WeightRole::LinearAttn { layer, part } => {
                    check_layer_idx(*layer, n_layers)?;
                    per_layer[*layer].linear_parts.insert(*part);
                }
                WeightRole::SelfAttn { layer, part } => {
                    check_layer_idx(*layer, n_layers)?;
                    per_layer[*layer].self_parts.insert(*part);
                }
                WeightRole::Mlp { layer, part } => {
                    check_layer_idx(*layer, n_layers)?;
                    per_layer[*layer].mlp_parts.insert(*part);
                }
                _ => {}
            }

            classified.push(ClassifiedGroup {
                prefix: g.prefix,
                shard: g.shard,
                storage: g.kind,
                role,
            });
        }

        // Every layer must carry the exact sub-role set expected by its layer_type
        // and the model-wide MLP variant (MoE vs Dense).
        let mlp_kind = config.text_config.mlp_kind();
        for (idx, inv) in per_layer.iter().enumerate() {
            validate_layer(
                idx,
                &config.text_config.layer_types[idx],
                mlp_kind,
                inv,
            )?;
        }

        // Top-level presence checks (one embed, one lm_head, one final norm).
        // 4B-style checkpoints with `tie_word_embeddings: true` reuse `embed_tokens`
        // as the output projection and so ship NO separate `lm_head` weight; allow
        // the lm_head check to pass when zero entries are present in that case.
        let top_roles: Vec<&WeightRole> = classified.iter().map(|c| &c.role).collect();
        ensure_unique(&top_roles, WeightRole::Embed, "embed")?;
        if config.text_config.tie_word_embeddings {
            ensure_at_most_one(&top_roles, WeightRole::LmHead, "lm_head")?;
        } else {
            ensure_unique(&top_roles, WeightRole::LmHead, "lm_head")?;
        }
        ensure_unique(&top_roles, WeightRole::FinalNorm, "final_norm")?;

        Ok(Self {
            groups: classified,
            per_layer,
            vision_count,
        })
    }

    /// Convenience: derive the `layer_type` implied by which attention sub-module is populated.
    /// Returns `None` if both or neither are populated (the validator would have already errored).
    pub fn observed_layer_type(&self, layer: usize) -> Option<LayerType> {
        let inv = self.per_layer.get(layer)?;
        match (!inv.linear_parts.is_empty(), !inv.self_parts.is_empty()) {
            (true, false) => Some(LayerType::LinearAttention),
            (false, true) => Some(LayerType::FullAttention),
            _ => None,
        }
    }
}

fn check_layer_idx(idx: usize, n: usize) -> Result<(), ClassifyError> {
    if idx >= n {
        Err(ClassifyError::LayerOutOfRange { idx, n_layers: n })
    } else {
        Ok(())
    }
}

fn ensure_unique(
    roles: &[&WeightRole],
    target: WeightRole,
    name: &'static str,
) -> Result<(), ClassifyError> {
    let count = roles.iter().filter(|r| ***r == target).count();
    if count != 1 {
        return Err(ClassifyError::TopLevelCount {
            role: name,
            count,
        });
    }
    Ok(())
}

/// Permit zero or one occurrences. Used for `lm_head` in tied-embedding checkpoints
/// (the output projection is shared with `embed_tokens` and ships no separate weight).
fn ensure_at_most_one(
    roles: &[&WeightRole],
    target: WeightRole,
    name: &'static str,
) -> Result<(), ClassifyError> {
    let count = roles.iter().filter(|r| ***r == target).count();
    if count > 1 {
        return Err(ClassifyError::TopLevelCount {
            role: name,
            count,
        });
    }
    Ok(())
}

/// Storage kind we expect a given prefix to have, based purely on `QuantizationConfig` + a
/// handful of architectural rules about which weights skip quantization entirely:
///   - `vision_tower.*` : always plain (the MLX config intentionally excludes vision)
///   - `*.input_layernorm`, `*.post_attention_layernorm`, `model.norm`, `*_norm`, `dt_bias`,
///     `A_log`, `conv1d` : always plain (activations / tiny state)
///   - everything else   : lookup `bits` via `config.quantization_config.for_weight(prefix)`
///                         - bits=4 → MXFP4 (weight + scales)
///                         - bits=8 → Int8Affine (weight + scales + biases)
fn expected_storage(prefix: &str, config: &Qwen3_5MoeConfig) -> StorageKind {
    if prefix.starts_with("vision_tower.") {
        return StorageKind::Plain;
    }
    if is_always_plain(prefix) {
        return StorageKind::Plain;
    }
    // Pure-bf16 checkpoints (e.g. `Qwen3.5-4B-MLX-bf16`) ship with no
    // `quantization_config` block at all — every weight is plain.
    let Some(qcfg) = config.quantization_config.as_ref() else {
        return StorageKind::Plain;
    };
    let params = qcfg.for_weight(prefix);
    // Storage kind is driven by `mode`, not `bits`:
    //   * `mxfp4`  : E2M1 + E8M0 packed; on-disk has `.scales` only.
    //   * `affine` : int affine quant with `.scales` + `.biases` regardless of bit
    //                width (35B-A3B-mxfp4 carries 8-bit affine for `mlp.gate` /
    //                `shared_expert_gate`; 27B-MLX-4bit ships uniform 4-bit affine
    //                for every projection).
    // 4-bit MXFP4 and 4-bit affine occupy the SAME tensor footprint on disk but
    // differ in dequant kernel and tag; misclassifying causes a load-time abort
    // with the mismatch error.
    match params.mode {
        "mxfp4" => match params.bits {
            4 => StorageKind::Mxfp4,
            8 => StorageKind::Int8Affine,
            _ => StorageKind::Plain,
        },
        "affine" => StorageKind::Int8Affine,
        _ => StorageKind::Plain,
    }
}

/// Weights that skip quantization entirely for architectural reasons (norms, SSM state).
fn is_always_plain(prefix: &str) -> bool {
    if prefix.ends_with(".input_layernorm")
        || prefix.ends_with(".post_attention_layernorm")
        || prefix == "language_model.model.norm"
    {
        return true;
    }
    if let Some(tail) = prefix.rsplit_once('.').map(|(_, t)| t) {
        matches!(
            tail,
            "norm" | "q_norm" | "k_norm" | "A_log" | "dt_bias" | "conv1d"
        )
    } else {
        false
    }
}

fn validate_layer(
    idx: usize,
    layer_type: &LayerType,
    mlp_kind: super::config::MlpKind,
    inv: &LayerInventory,
) -> Result<(), ClassifyError> {
    if !inv.has_input_ln || !inv.has_post_attn_ln {
        return Err(ClassifyError::LayerMissingLayerNorm { layer: idx });
    }

    use super::config::MlpKind;
    let expected_mlp: BTreeSet<MlpPart> = match mlp_kind {
        MlpKind::Moe => [
            MlpPart::Gate,
            MlpPart::SharedExpertGate,
            MlpPart::SharedExpert(ProjKind::Gate),
            MlpPart::SharedExpert(ProjKind::Up),
            MlpPart::SharedExpert(ProjKind::Down),
            MlpPart::SwitchMlp(ProjKind::Gate),
            MlpPart::SwitchMlp(ProjKind::Up),
            MlpPart::SwitchMlp(ProjKind::Down),
        ]
        .into_iter()
        .collect(),
        MlpKind::Dense => [
            MlpPart::Dense(ProjKind::Gate),
            MlpPart::Dense(ProjKind::Up),
            MlpPart::Dense(ProjKind::Down),
        ]
        .into_iter()
        .collect(),
    };
    if inv.mlp_parts != expected_mlp {
        let missing: Vec<MlpPart> = expected_mlp.difference(&inv.mlp_parts).copied().collect();
        let extra: Vec<MlpPart> = inv.mlp_parts.difference(&expected_mlp).copied().collect();
        return Err(ClassifyError::LayerMlpMismatch {
            layer: idx,
            missing,
            extra,
        });
    }

    match layer_type {
        LayerType::LinearAttention => {
            if !inv.self_parts.is_empty() {
                return Err(ClassifyError::LayerAttentionKindMismatch {
                    layer: idx,
                    declared: *layer_type,
                    reason: "found self_attn weights in a linear_attention layer",
                });
            }
            let expected: BTreeSet<LinearAttnPart> = [
                LinearAttnPart::ALog,
                LinearAttnPart::Conv1d,
                LinearAttnPart::DtBias,
                LinearAttnPart::Norm,
                LinearAttnPart::InProjA,
                LinearAttnPart::InProjB,
                LinearAttnPart::InProjQkv,
                LinearAttnPart::InProjZ,
                LinearAttnPart::OutProj,
            ]
            .into_iter()
            .collect();
            if inv.linear_parts != expected {
                let missing: Vec<LinearAttnPart> =
                    expected.difference(&inv.linear_parts).copied().collect();
                let extra: Vec<LinearAttnPart> =
                    inv.linear_parts.difference(&expected).copied().collect();
                return Err(ClassifyError::LayerLinearAttnMismatch {
                    layer: idx,
                    missing,
                    extra,
                });
            }
        }
        LayerType::FullAttention => {
            if !inv.linear_parts.is_empty() {
                return Err(ClassifyError::LayerAttentionKindMismatch {
                    layer: idx,
                    declared: *layer_type,
                    reason: "found linear_attn weights in a full_attention layer",
                });
            }
            let expected: BTreeSet<SelfAttnPart> = [
                SelfAttnPart::QNorm,
                SelfAttnPart::KNorm,
                SelfAttnPart::QProj,
                SelfAttnPart::KProj,
                SelfAttnPart::VProj,
                SelfAttnPart::OProj,
            ]
            .into_iter()
            .collect();
            if inv.self_parts != expected {
                let missing: Vec<SelfAttnPart> =
                    expected.difference(&inv.self_parts).copied().collect();
                let extra: Vec<SelfAttnPart> =
                    inv.self_parts.difference(&expected).copied().collect();
                return Err(ClassifyError::LayerSelfAttnMismatch {
                    layer: idx,
                    missing,
                    extra,
                });
            }
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    #[error("failed to deserialize safetensors index.json: {0}")]
    Parse(#[source] serde_json::Error),

    #[error(
        "weight group `{prefix}` straddles shards (first={first}, second={second}); MLX is \
        expected to place `.weight`, `.scales`, and `.biases` for a single weight in one shard"
    )]
    GroupSpansShards {
        prefix: String,
        first: String,
        second: String,
    },

    #[error(
        "weight group `{prefix}` has unexpected suffix combination (weight={weight}, \
        scales={scales}, biases={biases}); expected plain (weight), mxfp4 (weight+scales), or \
        int8-affine (weight+scales+biases)"
    )]
    UnexpectedSuffixCombination {
        prefix: String,
        weight: bool,
        scales: bool,
        biases: bool,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum ClassifyError {
    #[error(transparent)]
    Index(#[from] IndexError),

    #[error("unknown weight prefix `{0}` — not covered by the Qwen3.5-VL-MoE schema")]
    UnknownPrefix(String),

    #[error(
        "storage mismatch for `{prefix}`: quantization_config predicts {expected:?}, but the \
        index carries {found:?}"
    )]
    StorageMismatch {
        prefix: String,
        expected: StorageKind,
        found: StorageKind,
    },

    #[error("layer {idx} out of range (num_hidden_layers={n_layers})")]
    LayerOutOfRange { idx: usize, n_layers: usize },

    #[error("expected exactly one `{role}` weight, found {count}")]
    TopLevelCount { role: &'static str, count: usize },

    #[error("layer {layer} missing input_layernorm or post_attention_layernorm")]
    LayerMissingLayerNorm { layer: usize },

    #[error(
        "layer {layer} attention kind mismatch: declared {declared:?} but {reason}"
    )]
    LayerAttentionKindMismatch {
        layer: usize,
        declared: LayerType,
        reason: &'static str,
    },

    #[error(
        "layer {layer} linear_attn roster mismatch — missing: {missing:?}, extra: {extra:?}"
    )]
    LayerLinearAttnMismatch {
        layer: usize,
        missing: Vec<LinearAttnPart>,
        extra: Vec<LinearAttnPart>,
    },

    #[error(
        "layer {layer} self_attn roster mismatch — missing: {missing:?}, extra: {extra:?}"
    )]
    LayerSelfAttnMismatch {
        layer: usize,
        missing: Vec<SelfAttnPart>,
        extra: Vec<SelfAttnPart>,
    },

    #[error(
        "layer {layer} mlp roster mismatch — missing: {missing:?}, extra: {extra:?}"
    )]
    LayerMlpMismatch {
        layer: usize,
        missing: Vec<MlpPart>,
        extra: Vec<MlpPart>,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const INDEX_JSON: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/qwen3_5_moe_index.json"
    ));
    const CONFIG_JSON: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/qwen3_5_moe_config.json"
    ));

    fn load_config() -> Qwen3_5MoeConfig {
        let mut cfg: Qwen3_5MoeConfig = serde_json::from_str(CONFIG_JSON).unwrap();
        cfg.validate().unwrap();
        cfg
    }

    fn load_index() -> WeightIndex {
        WeightIndex::from_json_str(INDEX_JSON).unwrap()
    }

    #[test]
    fn parses_real_index_with_expected_shape() {
        let idx = load_index();
        assert_eq!(idx.weight_map.len(), 1658);
        assert_eq!(idx.metadata.total_size, 19_319_480_032);
        assert_eq!(
            idx.shards(),
            vec![
                "model-00001-of-00004.safetensors".to_string(),
                "model-00002-of-00004.safetensors".to_string(),
                "model-00003-of-00004.safetensors".to_string(),
                "model-00004-of-00004.safetensors".to_string(),
            ]
        );
    }

    #[test]
    fn groups_partition_all_1658_tensors_cleanly() {
        let idx = load_index();
        let groups = idx.groups().expect("all groups must classify");
        // MLX stores `.biases` only for the 80 int8-affine overrides (gate + shared_expert_gate
        // across 40 layers). MXFP4 groups contribute 2 tensors each (weight+scales), and plain
        // groups (norms, SSM state) contribute 1 each. Verifying the raw partition count is a
        // cheap sanity check without enumerating every role yet.
        let int8_count = groups.iter().filter(|g| g.kind == StorageKind::Int8Affine).count();
        let mxfp4_count = groups.iter().filter(|g| g.kind == StorageKind::Mxfp4).count();
        let plain_count = groups.iter().filter(|g| g.kind == StorageKind::Plain).count();
        assert_eq!(int8_count, 80, "40 layers × 2 gates = 80 int8-affine groups");
        // Reconstruct the total tensor count from group storage kinds and compare.
        let reconstructed = int8_count * 3 + mxfp4_count * 2 + plain_count;
        assert_eq!(reconstructed, idx.weight_map.len());
    }

    #[test]
    fn no_group_straddles_shards() {
        // This is enforced inside `groups()` — if any group's `.weight`, `.scales`, or `.biases`
        // landed in different shards we'd get `GroupSpansShards`. The MLX checkpoint keeps them
        // co-located, so this should succeed.
        load_index().groups().expect("shard coherence");
    }

    #[test]
    fn classifies_full_manifest_without_unknowns() {
        let idx = load_index();
        let cfg = load_config();
        let c = Classification::build(&idx, &cfg).expect("full classification should succeed");
        // Per-role counts:
        //   1 embed + 1 lm_head + 1 final_norm       = 3
        //   + 40 × 2 layernorms                      = 80
        //   + 40 × 8 mlp parts                       = 320
        //   + 30 linear layers × 9 linear_attn parts = 270
        //   + 10 full   layers × 6 self_attn parts   = 60
        //   + 333 vision                             = 333
        //   --------------------------------------------
        //                                          total 1,066 logical groups
        let counts = |pred: fn(&WeightRole) -> bool| {
            c.groups.iter().filter(|g| pred(&g.role)).count()
        };
        assert_eq!(counts(|r| matches!(r, WeightRole::Embed)), 1);
        assert_eq!(counts(|r| matches!(r, WeightRole::LmHead)), 1);
        assert_eq!(counts(|r| matches!(r, WeightRole::FinalNorm)), 1);
        assert_eq!(counts(|r| matches!(r, WeightRole::InputLayerNorm(_))), 40);
        assert_eq!(counts(|r| matches!(r, WeightRole::PostAttentionLayerNorm(_))), 40);
        assert_eq!(counts(|r| matches!(r, WeightRole::LinearAttn { .. })), 30 * 9);
        assert_eq!(counts(|r| matches!(r, WeightRole::SelfAttn { .. })), 10 * 6);
        assert_eq!(counts(|r| matches!(r, WeightRole::Mlp { .. })), 40 * 8);
        assert_eq!(counts(|r| matches!(r, WeightRole::Vision(_))), 333);
        assert_eq!(c.vision_count, 333);
        assert_eq!(c.groups.len(), 3 + 80 + 270 + 60 + 320 + 333);
    }

    #[test]
    fn observed_layer_types_match_config() {
        let idx = load_index();
        let cfg = load_config();
        let c = Classification::build(&idx, &cfg).unwrap();
        for (i, declared) in cfg.text_config.layer_types.iter().enumerate() {
            let observed = c
                .observed_layer_type(i)
                .unwrap_or_else(|| panic!("layer {i} has inconclusive attention inventory"));
            assert_eq!(observed, *declared, "layer {i}");
        }
    }

    #[test]
    fn gate_and_shared_expert_gate_are_int8_affine_in_every_layer() {
        let idx = load_index();
        let cfg = load_config();
        let c = Classification::build(&idx, &cfg).unwrap();
        let mut seen_gate = 0;
        let mut seen_shared = 0;
        for g in &c.groups {
            if let WeightRole::Mlp { part, .. } = &g.role {
                match part {
                    MlpPart::Gate => {
                        assert_eq!(g.storage, StorageKind::Int8Affine);
                        seen_gate += 1;
                    }
                    MlpPart::SharedExpertGate => {
                        assert_eq!(g.storage, StorageKind::Int8Affine);
                        seen_shared += 1;
                    }
                    _ => assert_eq!(
                        g.storage,
                        StorageKind::Mxfp4,
                        "non-gate MLP weight `{}` should be MXFP4",
                        g.prefix
                    ),
                }
            }
        }
        assert_eq!(seen_gate, 40);
        assert_eq!(seen_shared, 40);
    }

    #[test]
    fn vision_tower_weights_are_all_plain() {
        let idx = load_index();
        let cfg = load_config();
        let c = Classification::build(&idx, &cfg).unwrap();
        let vision: Vec<&ClassifiedGroup> = c
            .groups
            .iter()
            .filter(|g| matches!(g.role, WeightRole::Vision(_)))
            .collect();
        assert_eq!(vision.len(), 333);
        for v in vision {
            assert_eq!(
                v.storage,
                StorageKind::Plain,
                "vision weight `{}` must be plain",
                v.prefix
            );
        }
    }

    #[test]
    fn embed_and_lm_head_are_mxfp4() {
        let idx = load_index();
        let cfg = load_config();
        let c = Classification::build(&idx, &cfg).unwrap();
        for g in &c.groups {
            match g.role {
                WeightRole::Embed | WeightRole::LmHead => {
                    assert_eq!(g.storage, StorageKind::Mxfp4, "{}", g.prefix);
                }
                _ => {}
            }
        }
    }

    #[test]
    fn norms_and_ssm_state_are_plain() {
        let idx = load_index();
        let cfg = load_config();
        let c = Classification::build(&idx, &cfg).unwrap();
        for g in &c.groups {
            match &g.role {
                WeightRole::InputLayerNorm(_)
                | WeightRole::PostAttentionLayerNorm(_)
                | WeightRole::FinalNorm => {
                    assert_eq!(g.storage, StorageKind::Plain, "{}", g.prefix);
                }
                WeightRole::LinearAttn { part, .. } => match part {
                    LinearAttnPart::ALog
                    | LinearAttnPart::DtBias
                    | LinearAttnPart::Conv1d
                    | LinearAttnPart::Norm => {
                        assert_eq!(g.storage, StorageKind::Plain, "{}", g.prefix);
                    }
                    _ => assert_eq!(g.storage, StorageKind::Mxfp4, "{}", g.prefix),
                },
                WeightRole::SelfAttn { part, .. } => match part {
                    SelfAttnPart::QNorm | SelfAttnPart::KNorm => {
                        assert_eq!(g.storage, StorageKind::Plain, "{}", g.prefix);
                    }
                    _ => assert_eq!(g.storage, StorageKind::Mxfp4, "{}", g.prefix),
                },
                _ => {}
            }
        }
    }

    #[test]
    fn classify_rejects_unknown_prefix() {
        // Inject a bogus tensor and verify the classifier surfaces it.
        let mut idx = load_index();
        idx.weight_map.insert(
            "language_model.model.layers.0.mystery_module.weight".to_string(),
            "model-00001-of-00004.safetensors".to_string(),
        );
        let cfg = load_config();
        let err = Classification::build(&idx, &cfg).unwrap_err();
        match err {
            ClassifyError::UnknownPrefix(p) => {
                assert_eq!(p, "language_model.model.layers.0.mystery_module");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn classify_rejects_storage_mismatch() {
        // Fabricate an index that gives the `gate` (should be int8-affine) only a .weight/.scales
        // pair — the classifier should complain that the override predicts Int8Affine.
        let mut idx = load_index();
        // Remove the .biases tensor to downgrade gate to Mxfp4 shape.
        idx.weight_map
            .remove("language_model.model.layers.0.mlp.gate.biases")
            .expect("gate.biases should be present in fixture");
        let cfg = load_config();
        let err = Classification::build(&idx, &cfg).unwrap_err();
        match err {
            ClassifyError::StorageMismatch { prefix, expected, found } => {
                assert_eq!(prefix, "language_model.model.layers.0.mlp.gate");
                assert_eq!(expected, StorageKind::Int8Affine);
                assert_eq!(found, StorageKind::Mxfp4);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn detects_missing_linear_attn_part() {
        let mut idx = load_index();
        // Drop A_log from layer 0 (linear_attention) — roster validation should flag it.
        idx.weight_map
            .remove("language_model.model.layers.0.linear_attn.A_log")
            .expect("A_log should be present");
        let cfg = load_config();
        let err = Classification::build(&idx, &cfg).unwrap_err();
        match err {
            ClassifyError::LayerLinearAttnMismatch { layer, missing, .. } => {
                assert_eq!(layer, 0);
                assert!(missing.contains(&LinearAttnPart::ALog));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
