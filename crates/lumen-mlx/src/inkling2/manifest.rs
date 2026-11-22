//! Strict tensor manifest of the `inkling2_loop` checkpoint. Every tensor name
//! maps to exactly one [`Role`], and the placement of roles over layers is
//! checked against `config.json`.
//!
//! Role names are the checkpoint's own keys with the layer index written as
//! `N`. Naming a role does not explain it: nothing here says what a loop, a
//! "query" or a static key computes. That is unverified: needs the reference.
//!
//! Rules, all from names and the config:
//! - a key that maps to no role is an error, and the error prints the key;
//! - each (role, layer) occurs at most once;
//! - a local-only role sits only on local layers, a global-only role only on
//!   global layers, a core-only role only inside the loop's core range;
//! - every role a layer must carry is present on each such layer;
//! - vision and MTP layers each carry their full role set.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, anyhow, bail, ensure};

use super::config::{Inkling2Config, LayerKind};
use super::safetensors_header::TensorInfo;

const MTP_FILE: &str = "mtp.safetensors";
/// Prefixes after which a layer index sits; the index is replaced by `N`.
const LAYERED_PREFIXES: [&str; 3] = ["model.layers.", "vision_tower.backbone.layers.", "layers."];

/// Which part of the checkpoint a tensor belongs to. The vision tower lives in
/// the main shards; MTP lives in `mtp.safetensors`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Part {
    Text,
    Vision,
    Mtp,
}

/// Where a role may sit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// Exactly one tensor in the checkpoint; it has no layer.
    Once,
    /// Required on every local layer, forbidden on global layers.
    Local,
    /// Required on every global layer, forbidden on local layers.
    Global,
    /// Required on every layer of the loop's core range, forbidden outside it.
    Core,
    /// Required on every layer of its part.
    EveryLayer,
    /// Allowed on any layer, required on none.
    AnyLayer,
}

macro_rules! roles {
    ($(
        $part:ident {
            $(
                $place:ident {
                    $( $variant:ident = $key:literal, )*
                }
            )*
        }
    )*) => {
        /// Every tensor role of the checkpoint. A role's key is the checkpoint's
        /// own name with the layer index written as `N`.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub enum Role {
            $($($( $variant, )*)*)*
        }

        impl Role {
            /// Every role, in table order.
            pub const ALL: &'static [Role] = &[$($($( Role::$variant, )*)*)*];

            /// The checkpoint name with the layer index written as `N`.
            pub const fn key(self) -> &'static str {
                match self {
                    $($($( Role::$variant => $key, )*)*)*
                }
            }

            pub const fn part(self) -> Part {
                match self {
                    $($($( Role::$variant => Part::$part, )*)*)*
                }
            }

            pub const fn placement(self) -> Placement {
                match self {
                    $($($( Role::$variant => Placement::$place, )*)*)*
                }
            }
        }
    };
}

roles! {
    Text {
        Once {
            ModelEmbedWeight = "model.embed.weight",
            ModelEmbedScales = "model.embed.scales",
            ModelEmbedBiases = "model.embed.biases",
            ModelEmbedNormWeight = "model.embed_norm.weight",
            ModelFinalInputQuery = "model.final_input.query",
            ModelNormWeight = "model.norm.weight",
            LmHeadWeight = "lm_head.weight",
            LmHeadScales = "lm_head.scales",
            LmHeadBiases = "lm_head.biases",
        }
        Local {
            AttnInProjWeight = "model.layers.N.attn.in_proj.weight",
            AttnInProjScales = "model.layers.N.attn.in_proj.scales",
            AttnInProjBiases = "model.layers.N.attn.in_proj.biases",
            AttnKConv = "model.layers.N.attn.k_conv",
            AttnVConv = "model.layers.N.attn.v_conv",
            AttnNormWeight = "model.layers.N.attn_norm.weight",
        }
        Global {
            AttnQProjWeight = "model.layers.N.attn.q_proj.weight",
            AttnQProjScales = "model.layers.N.attn.q_proj.scales",
            AttnQProjBiases = "model.layers.N.attn.q_proj.biases",
            AttnKProjWeight = "model.layers.N.attn.k_proj.weight",
            AttnKProjScales = "model.layers.N.attn.k_proj.scales",
            AttnKProjBiases = "model.layers.N.attn.k_proj.biases",
            AttnVProjWeight = "model.layers.N.attn.v_proj.weight",
            AttnVProjScales = "model.layers.N.attn.v_proj.scales",
            AttnVProjBiases = "model.layers.N.attn.v_proj.biases",
            AttnKStatic = "model.layers.N.attn.k_static",
            AttnVStatic = "model.layers.N.attn.v_static",
            QInputNormWeight = "model.layers.N.q_input_norm.weight",
            KInputNormWeight = "model.layers.N.k_input_norm.weight",
            VInputNormWeight = "model.layers.N.v_input_norm.weight",
        }
        Core {
            LoopDown = "model.layers.N.loop_down",
            LoopGainForget = "model.layers.N.loop_gain_forget",
            LoopMessageQuery = "model.layers.N.loop_message.query",
        }
        EveryLayer {
            AttnOProjWeight = "model.layers.N.attn.o_proj.weight",
            AttnOProjScales = "model.layers.N.attn.o_proj.scales",
            AttnOProjBiases = "model.layers.N.attn.o_proj.biases",
            AttnQScale = "model.layers.N.attn.q_scale",
            AttnRScale = "model.layers.N.attn.r_scale",
            AttnRelProj = "model.layers.N.attn.rel_proj",
            AttnConv = "model.layers.N.attn_conv",
            MlpConv = "model.layers.N.mlp_conv",
            MlpRouterWeight = "model.layers.N.mlp.router_weight",
            MlpRouterBias = "model.layers.N.mlp.router_bias",
            MlpInputQuery = "model.layers.N.mlp_input.query",
            MlpNormWeight = "model.layers.N.mlp_norm.weight",
            ExpertsW13Weight = "model.layers.N.mlp.experts.w13.weight",
            ExpertsW13Scales = "model.layers.N.mlp.experts.w13.scales",
            ExpertsW2Weight = "model.layers.N.mlp.experts.w2.weight",
            ExpertsW2Scales = "model.layers.N.mlp.experts.w2.scales",
        }
        AnyLayer {
            AttnInputQuery = "model.layers.N.attn_input.query",
            ExpertsW13Biases = "model.layers.N.mlp.experts.w13.biases",
            ExpertsW13GlobalScale = "model.layers.N.mlp.experts.w13.global_scale",
            ExpertsW2Biases = "model.layers.N.mlp.experts.w2.biases",
            ExpertsW2GlobalScale = "model.layers.N.mlp.experts.w2.global_scale",
        }
    }
    Vision {
        Once {
            VisionClassEmbedding = "vision_tower.backbone.embeddings.class_embedding",
            VisionClsPositionEmbedding = "vision_tower.backbone.embeddings.cls_position_embedding",
            VisionPatchEmbeddingWeight = "vision_tower.backbone.embeddings.patch_embedding.weight",
            VisionPositionEmbedding = "vision_tower.backbone.embeddings.position_embedding",
            VisionLnPreWeight = "vision_tower.backbone.ln_pre.weight",
            VisionLnPreBias = "vision_tower.backbone.ln_pre.bias",
            VisionProjInWeight = "vision_tower.proj_in.weight",
            VisionProjOutWeight = "vision_tower.proj_out.weight",
        }
        EveryLayer {
            VisionBlockLn1Weight = "vision_tower.backbone.layers.N.ln_1.weight",
            VisionBlockLn1Bias = "vision_tower.backbone.layers.N.ln_1.bias",
            VisionBlockLn2Weight = "vision_tower.backbone.layers.N.ln_2.weight",
            VisionBlockLn2Bias = "vision_tower.backbone.layers.N.ln_2.bias",
            VisionBlockLs1Gamma = "vision_tower.backbone.layers.N.ls_1.gamma",
            VisionBlockLs2Gamma = "vision_tower.backbone.layers.N.ls_2.gamma",
            VisionBlockMlpCFcWeight = "vision_tower.backbone.layers.N.mlp.c_fc.weight",
            VisionBlockMlpCFcBias = "vision_tower.backbone.layers.N.mlp.c_fc.bias",
            VisionBlockMlpCProjWeight = "vision_tower.backbone.layers.N.mlp.c_proj.weight",
            VisionBlockMlpCProjBias = "vision_tower.backbone.layers.N.mlp.c_proj.bias",
            VisionBlockQProjWeight = "vision_tower.backbone.layers.N.self_attn.q_proj.weight",
            VisionBlockQProjBias = "vision_tower.backbone.layers.N.self_attn.q_proj.bias",
            VisionBlockKProjWeight = "vision_tower.backbone.layers.N.self_attn.k_proj.weight",
            VisionBlockKProjBias = "vision_tower.backbone.layers.N.self_attn.k_proj.bias",
            VisionBlockVProjWeight = "vision_tower.backbone.layers.N.self_attn.v_proj.weight",
            VisionBlockVProjBias = "vision_tower.backbone.layers.N.self_attn.v_proj.bias",
            VisionBlockOutProjWeight = "vision_tower.backbone.layers.N.self_attn.out_proj.weight",
            VisionBlockOutProjBias = "vision_tower.backbone.layers.N.self_attn.out_proj.bias",
        }
    }
    Mtp {
        Once {
            MtpFuseWeight = "fuse_weight",
            MtpMaskEmbed = "mask_embed",
            MtpNormWeight = "norm.weight",
            MtpMarkovHeadDflareNormWeight = "markov_head.dflare_norm.weight",
            MtpMarkovHeadHiddenOutBias = "markov_head.hidden_out.bias",
            MtpMarkovHeadHiddenOutWeight = "markov_head.hidden_out.weight",
            MtpMarkovHeadMarkovW1Weight = "markov_head.markov_w1.weight",
            MtpMarkovHeadRnnProjBias = "markov_head.rnn_proj.bias",
            MtpMarkovHeadRnnProjWeight = "markov_head.rnn_proj.weight",
        }
        EveryLayer {
            MtpLayerDflareNormWeight = "layers.N.dflare_norm.weight",
            MtpLayerInputLayernormWeight = "layers.N.input_layernorm.weight",
            MtpLayerMlpDownProjWeight = "layers.N.mlp.down_proj.weight",
            MtpLayerMlpGateProjWeight = "layers.N.mlp.gate_proj.weight",
            MtpLayerMlpUpProjWeight = "layers.N.mlp.up_proj.weight",
            MtpLayerPostAttnLayernormWeight = "layers.N.post_attention_layernorm.weight",
            MtpLayerSelfAttnKNormWeight = "layers.N.self_attn.k_norm.weight",
            MtpLayerSelfAttnKNormCtxWeight = "layers.N.self_attn.k_norm_ctx.weight",
            MtpLayerSelfAttnKProjWeight = "layers.N.self_attn.k_proj.weight",
            MtpLayerSelfAttnKProjCtxWeight = "layers.N.self_attn.k_proj_ctx.weight",
            MtpLayerSelfAttnOProjWeight = "layers.N.self_attn.o_proj.weight",
            MtpLayerSelfAttnQNormWeight = "layers.N.self_attn.q_norm.weight",
            MtpLayerSelfAttnQProjWeight = "layers.N.self_attn.q_proj.weight",
            MtpLayerSelfAttnVProjWeight = "layers.N.self_attn.v_proj.weight",
            MtpLayerSelfAttnVProjCtxWeight = "layers.N.self_attn.v_proj_ctx.weight",
        }
    }
}

/// One tensor, its role, and its layer (`None` for `Once` roles).
#[derive(Debug, Clone)]
pub struct Entry {
    pub role: Role,
    pub layer: Option<usize>,
    pub tensor: TensorInfo,
}

/// Every tensor of a checkpoint mapped to a role, with the layout checked.
#[derive(Debug, Clone)]
pub struct Manifest {
    entries: Vec<Entry>,
}

impl Manifest {
    /// Maps every tensor and checks the layout. The first violation is returned.
    pub fn build(cfg: &Inkling2Config, tensors: &[TensorInfo]) -> Result<Self> {
        let mut entries = Vec::with_capacity(tensors.len());
        let mut owner: BTreeMap<(Role, Option<usize>), &str> = BTreeMap::new();
        for t in tensors {
            let entry = classify(cfg, t)?;
            if let Some(first) = owner.insert((entry.role, entry.layer), &t.name) {
                bail!(
                    "{}: repeats the role {:?} of {first} (layer {:?})",
                    t.name,
                    entry.role,
                    entry.layer
                );
            }
            entries.push(entry);
        }
        check_layout(cfg, &entries)?;
        Ok(Self { entries })
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// How many tensors of `part` the manifest maps.
    pub fn count(&self, part: Part) -> usize {
        self.entries
            .iter()
            .filter(|e| e.role.part() == part)
            .count()
    }

    /// The layers that carry `role`, ascending.
    pub fn layers_with(&self, role: Role) -> Vec<usize> {
        self.entries
            .iter()
            .filter(|e| e.role == role)
            .filter_map(|e| e.layer)
            .collect::<BTreeSet<usize>>()
            .into_iter()
            .collect()
    }
}

fn classify(cfg: &Inkling2Config, t: &TensorInfo) -> Result<Entry> {
    let unknown = || anyhow!("unknown tensor key {:?} in {}", t.name, t.file);
    let (key, layer) = normalize(&t.name).ok_or_else(unknown)?;
    let role = Role::ALL
        .iter()
        .copied()
        .find(|r| r.key() == key)
        .ok_or_else(unknown)?;
    ensure!(
        (role.part() == Part::Mtp) == (t.file == MTP_FILE),
        "{} is a {:?} tensor but sits in {}",
        t.name,
        role.part(),
        t.file
    );
    if let Some(layer) = layer {
        let count = layer_count(cfg, role.part());
        ensure!(
            layer < count,
            "{}: layer {layer} is not below the {count} {:?} layers",
            t.name,
            role.part()
        );
    }
    Ok(Entry {
        role,
        layer,
        tensor: t.clone(),
    })
}

/// The key with its layer index replaced by `N`, and that index. `None` when a
/// layered prefix is not followed by an index and a tail, or when the index is
/// not written the canonical way (`05` and `+5` are not layer 5).
fn normalize(name: &str) -> Option<(String, Option<usize>)> {
    for prefix in LAYERED_PREFIXES {
        if let Some(rest) = name.strip_prefix(prefix) {
            let (index, tail) = rest.split_once('.')?;
            let layer: usize = index.parse().ok()?;
            if layer.to_string() != index {
                return None;
            }
            return Some((format!("{prefix}N.{tail}"), Some(layer)));
        }
    }
    Some((name.to_string(), None))
}

fn layer_count(cfg: &Inkling2Config, part: Part) -> usize {
    match part {
        Part::Text => cfg.text_config.num_hidden_layers,
        Part::Vision => cfg.vision_config.num_hidden_layers,
        Part::Mtp => cfg.mtp_config.num_hidden_layers,
    }
}

/// Checks the `Once` roles and every layer's placement rules.
fn check_layout(cfg: &Inkling2Config, entries: &[Entry]) -> Result<()> {
    let present: BTreeSet<(Role, Option<usize>)> =
        entries.iter().map(|e| (e.role, e.layer)).collect();
    for &role in Role::ALL {
        if role.placement() == Placement::Once {
            ensure!(
                present.contains(&(role, None)),
                "missing required tensor {}",
                role.key()
            );
        }
    }
    for part in [Part::Text, Part::Vision, Part::Mtp] {
        for layer in 0..layer_count(cfg, part) {
            let (kind, core) = match part {
                Part::Text => (Some(cfg.layer_kind(layer)), cfg.is_core(layer)),
                Part::Vision | Part::Mtp => (None, false),
            };
            let here = describe(part, kind, core);
            for &role in Role::ALL.iter().filter(|r| r.part() == part) {
                let placement = role.placement();
                if placement == Placement::Once {
                    continue;
                }
                let allowed = match placement {
                    Placement::Local => kind == Some(LayerKind::Local),
                    Placement::Global => kind == Some(LayerKind::Global),
                    Placement::Core => core,
                    _ => true,
                };
                let required = allowed && placement != Placement::AnyLayer;
                let has = present.contains(&(role, Some(layer)));
                ensure!(
                    allowed || !has,
                    "{} is on layer {layer} ({here}), which may not carry it",
                    concrete(role, layer)
                );
                ensure!(
                    !required || has,
                    "layer {layer} ({here}) is missing {}",
                    concrete(role, layer)
                );
            }
        }
    }
    Ok(())
}

fn describe(part: Part, kind: Option<LayerKind>, core: bool) -> String {
    match (kind, core) {
        (Some(LayerKind::Local), false) => "local layer".into(),
        (Some(LayerKind::Local), true) => "local core layer".into(),
        (Some(LayerKind::Global), false) => "global layer".into(),
        (Some(LayerKind::Global), true) => "global core layer".into(),
        (None, _) => format!("{part:?} layer"),
    }
}

/// The checkpoint name of `role` at `layer`.
fn concrete(role: Role, layer: usize) -> String {
    role.key().replacen('N', &layer.to_string(), 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_are_unique_and_carry_an_index_exactly_when_placed_per_layer() {
        let keys: BTreeSet<&str> = Role::ALL.iter().map(|r| r.key()).collect();
        assert_eq!(keys.len(), Role::ALL.len());
        for &role in Role::ALL {
            let layered = role.key().contains(".N.") || role.key().starts_with("layers.N.");
            assert_eq!(
                layered,
                role.placement() != Placement::Once,
                "{}",
                role.key()
            );
        }
    }

    #[test]
    fn normalize_writes_the_layer_index_as_n() {
        let layered = normalize("model.layers.12.attn.in_proj.weight");
        assert_eq!(
            layered,
            Some(("model.layers.N.attn.in_proj.weight".into(), Some(12)))
        );
        assert_eq!(
            normalize("model.embed.weight"),
            Some(("model.embed.weight".into(), None))
        );
        assert_eq!(normalize("model.layers.x.attn"), None);
        assert_eq!(normalize("layers.2"), None);
    }

    #[test]
    fn a_padded_or_signed_layer_index_is_not_a_layer() {
        assert_eq!(normalize("model.layers.05.attn.q_proj.weight"), None);
        assert_eq!(normalize("model.layers.+5.attn.q_proj.weight"), None);
        assert_eq!(
            normalize("model.layers.0.attn.q_proj.weight"),
            Some(("model.layers.N.attn.q_proj.weight".into(), Some(0)))
        );
    }
}
