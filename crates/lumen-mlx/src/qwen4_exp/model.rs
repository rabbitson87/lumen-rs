//! Weights, caches and the forward pass of Qwen3.8-Flash-Next.
//!
//! Everything here follows the reference implementation line for line; the
//! comments name the reference function a block mirrors. Activations are f32
//! (the quantized matmuls take f32 and return f32), like the Qwen3.5 port.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use mlx_rs::ops::indexing::{TryIndexOp, put_along_axis, take_along_axis};
use mlx_rs::{Array, Dtype};

use super::config::{LayerKind, Qwen4ExpConfig, kv_bf16};
use super::ngram::NgramHasher;
use super::ngram_table::{DEFAULT_ROW_CACHE, FileTable, NgramTable, ResidentTable, ShardArrays};
use crate::decode_probe::{self as probe, Stage};
use crate::kv_disk::{ArrayRecord, LayerKindTag, LayerMeta, RecordArray, array_at, push_record};
use crate::native_attention::{sdpa, sdpa_with_mask};
use crate::native_cache::NativeKvCache;
use crate::native_embedding::quantized_embedding_lookup_with_mode;
use crate::native_moe::{
    AffineGateWeights, GateProj, MoeBlockWeights, OffloadLayer, SharedMlpWeights,
    SwitchExpertWeights, SwitchGluWeights, moe_after_routing, moe_block_forward, moe_router,
};
use crate::native_quant::MODE_AFFINE;
use crate::native_rope::rope;
use crate::native_ssm::{compute_g, gated_delta_step_kernel};
use crate::offload::{IoMode, IoTuning, MoeOffload};
use crate::qwen3_5_moe::{NativeWeights, ResolvedLinear};

mod mtp;
pub use mtp::MtpMode;
use mtp::{MtpHead, MtpState};

// ---- small helpers -------------------------------------------------------------------------

fn scalar(v: f32) -> Array {
    Array::from_f32(v)
}

fn to_f32(a: Array) -> Result<Array> {
    a.as_dtype(Dtype::Float32).context("cast to f32")
}

/// A copy of `a` that owns its storage, bit for bit.
///
/// A slice is a view: it keeps the whole buffer it was cut from alive. The state a
/// sequence carries to its next chunk (the last rows of a convolution input, the
/// indexer's keys) is cut from a chunk-sized buffer, so storing the slice kept
/// every chunk's full input around until the next chunk replaced it: 36 layers x
/// a 4,096-token input is ~6 GB, which is what ran a 15 GB expert pool out of
/// memory. `x * 1` is exact for every float and cannot be elided into the view.
fn owned(a: &Array) -> Result<Array> {
    Ok(mlx_rs::ops::multiply(
        a,
        &Array::from_f32(1.0).as_dtype(a.dtype())?,
    )?)
}

/// `x / sqrt(sum(x^2) + 1e-6)` over the last axis (the reference's `_l2norm`;
/// not an `rms_norm`, whose epsilon is scaled by the width).
fn l2norm(x: &Array) -> Result<Array> {
    let squares = mlx_rs::ops::multiply(x, x)?;
    let sum = mlx_rs::ops::sum_axis(&squares, -1, true)?;
    let inv = mlx_rs::ops::rsqrt(&mlx_rs::ops::add(&sum, scalar(1e-6))?)?;
    Ok(mlx_rs::ops::multiply(x, &inv)?)
}

/// RMS-normalise each of `groups` consecutive `width`-wide slices of the last
/// axis separately, then scale by `weight` (`groups * width` long): the
/// reference's `RMSNorm(group_size=width)`.
fn group_rms_norm(
    x: &Array,
    weight: &Array,
    ones: &Array,
    groups: i32,
    width: i32,
    eps: f32,
) -> Result<Array> {
    let shape = x.shape().to_vec();
    let mut grouped = shape[..shape.len() - 1].to_vec();
    grouped.push(groups);
    grouped.push(width);
    let normed = mlx_rs::fast::rms_norm(&mlx_rs::ops::reshape(x, &grouped)?, ones, eps)?;
    let flat = mlx_rs::ops::reshape(&normed, &shape)?;
    Ok(mlx_rs::ops::multiply(&flat, weight)?)
}

/// Read every weight the forward pass holds on to into memory, here and now.
///
/// The loader hands back lazy arrays whose read is bound to the loading thread's CPU
/// stream, and the server runs the model on another thread: evaluating such an array
/// there fails with "There is no Stream(cpu, 0) in current thread". A weight the first
/// request happens not to need (the QSA indexer's norms, only used past 2,048 tokens)
/// would surface that on some later request. Arrays *computed* at load (a folded norm, a
/// transposed router) carry the loading thread's GPU stream the same way and are
/// evaluated where they are built. The routed experts and the n-gram table
/// stay lazy on purpose: they are never evaluated, only read from disk by offsets.
fn materialize_trunk(weights: &NativeWeights, with_experts: bool) -> Result<()> {
    let resident: Vec<&Array> = weights
        .tensors()
        .iter()
        .filter(|(name, _)| {
            (with_experts || !name.contains(".switch_mlp.")) && !name.contains(".ngram_embedding.")
        })
        .map(|(_, a)| a)
        .collect();
    for batch in resident.chunks(256) {
        mlx_rs::transforms::eval(batch.iter().copied())?;
    }
    Ok(())
}

fn require(weights: &NativeWeights, name: &str) -> Result<Array> {
    Ok(weights.require(name)?.clone())
}

/// An RMSNorm weight that the HF convention stores as `w - 1`: add the 1 back, in the
/// weight's own dtype like the reference's `sanitize`, then widen. Plain `require`
/// when the build already folded it.
fn centered_norm(weights: &NativeWeights, cfg: &Qwen4ExpConfig, name: &str) -> Result<Array> {
    let w = require(weights, name)?;
    if !cfg.norms_centered {
        return Ok(w);
    }
    let folded = to_f32(mlx_rs::ops::add(
        &w,
        &Array::from_f32(1.0).as_dtype(w.dtype())?,
    )?)?;
    // Computed here, so computed on this thread's stream: see `materialize_trunk`.
    folded
        .eval()
        .with_context(|| format!("{name}: fold the +1"))?;
    Ok(folded)
}

/// The mean of a build's attention q-norm weight says which convention it uses: folded
/// weights sit around 1 (measured 1.28), HF-centered ones around 0 (0.28).
fn norms_are_centered(q_norm_mean: f32) -> bool {
    q_norm_mean < 0.6
}

fn detect_centered_norms(weights: &NativeWeights, cfg: &Qwen4ExpConfig) -> Result<bool> {
    let layer = cfg
        .layer_kinds
        .iter()
        .position(|k| *k == LayerKind::FullAttention)
        .ok_or_else(|| anyhow!("qwen4_exp has no full-attention layer to inspect"))?;
    let name = format!("language_model.model.layers.{layer}.self_attn.q_norm.weight");
    let mean = mlx_rs::ops::mean(&to_f32(require(weights, &name)?)?, None)?;
    mean.eval().context("evaluate the q_norm mean")?;
    Ok(norms_are_centered(mean.item::<f32>()))
}

/// A quantized projection: its bit width comes from the config, its group size
/// from the shapes (`in_features / scales.shape[-1]`), which is what the data
/// says whatever spelling of the config key the build used.
fn quant_linear(
    weights: &NativeWeights,
    cfg: &Qwen4ExpConfig,
    base: &str,
) -> Result<ResolvedLinear> {
    let weight = require(weights, &format!("{base}.weight"))?;
    let scales = require(weights, &format!("{base}.scales"))
        .with_context(|| format!("{base} is not quantized"))?;
    let bits = cfg.quant.for_base(base).bits;
    let packed = *weight.shape().last().unwrap_or(&0) as i64;
    let groups = *scales.shape().last().unwrap_or(&0) as i64;
    let in_features = packed * 32 / i64::from(bits.max(1));
    if groups == 0 || in_features % groups != 0 {
        bail!(
            "{base}: {packed} packed columns / {groups} scale groups do not describe a {bits}-bit tensor"
        );
    }
    let biases = weights.get(&format!("{base}.biases")).cloned();
    // MLX gives `quantized_matmul` the common dtype of the activations and the scales and
    // biases, casting whichever differs on EVERY call. The activations here are f32 and the
    // checkpoint's scales bf16, so each call re-widened the scales and biases (two extra
    // kernels over up to 20 MB for the lm_head, ~12 per layer for the trunk). The widening is
    // exact, so doing it once at load changes no result and costs +0.3 GB for the trunk.
    let (scales, biases) = if widen_scales::get() {
        (widen(scales)?, biases.map(widen).transpose()?)
    } else {
        (scales, biases)
    };
    Ok(ResolvedLinear {
        weight,
        scales,
        biases,
        group_size: (in_features / groups) as i32,
        bits,
        mode: MODE_AFFINE,
    })
}

lumen_flags::flag! {
    /// Keep the resident quantized projections' scales and biases in f32 so the activations'
    /// matmul stops re-casting them on every call. Bit-identical (bf16 -> f32 is exact).
    pub(crate) widen_scales {
        env: "LUMEN_QWEN4_WIDEN_SCALES",
        default: true,
        kind: Optimization,
    }
}

lumen_flags::flag! {
    /// Decode with the offloaded layers pipelined one segment ahead: submit the GPU work that
    /// ends in a layer's routing, then build the rest of that layer and the whole next layer's
    /// graph while the GPU runs it, and only then read the routing back. The graph is the same
    /// one the plain path builds (the gather reads its slot numbers from a persistent array the
    /// CPU fills in after the readback), so the output is bit-identical; what changes is that the
    /// host's graph building no longer sits between two GPU segments. Measured on the real
    /// checkpoint (M3 Max): identical text, but no end-to-end gain (15 GB pool 9.79 -> 9.81 tok/s,
    /// N=10; 6.7 GB +2%, noise); only ~5.5 ms of host build per token is hideable. Kept opt-in.
    pub(crate) build_ahead {
        env: "LUMEN_QWEN4_BUILD_AHEAD",
        default: false,
        kind: Optimization,
    }
}

fn widen(a: Array) -> Result<Array> {
    if a.dtype() == Dtype::Float32 {
        return Ok(a);
    }
    let wide = to_f32(a)?;
    // On the loading thread, like every other array computed at load (see `materialize_trunk`).
    wide.eval().context("widen scales")?;
    Ok(wide)
}

/// A projection that is quantized in some builds and left in float in others
/// (the router, the shared-expert gate, `block_inject_weight`, `in_proj_a/b` and
/// the indexer projection of the gs64 build).
enum Proj {
    Quant(ResolvedLinear),
    /// `[in, out]` float32, transposed and widened at load.
    Dense(Array),
}

impl Proj {
    fn load(weights: &NativeWeights, cfg: &Qwen4ExpConfig, base: &str) -> Result<Self> {
        if weights.get(&format!("{base}.scales")).is_some() {
            Ok(Self::Quant(quant_linear(weights, cfg, base)?))
        } else {
            Ok(Self::Dense(dense_transposed(weights, base)?))
        }
    }

    fn apply(&self, x: &Array) -> Result<Array> {
        match self {
            Self::Quant(l) => l.apply(x),
            Self::Dense(w_t) => Ok(mlx_rs::ops::matmul(x, w_t)?),
        }
    }
}

fn dense_transposed(weights: &NativeWeights, base: &str) -> Result<Array> {
    let w = require(weights, &format!("{base}.weight"))?;
    if w.ndim() != 2 {
        bail!(
            "{base}: a float projection must be 2-D, got {} dims",
            w.ndim()
        );
    }
    let w_t = to_f32(mlx_rs::ops::transpose_axes(&w, &[1, 0])?)?;
    w_t.eval()
        .with_context(|| format!("{base}: widen to f32"))?;
    Ok(w_t)
}

// ---- weights -------------------------------------------------------------------------------

/// `GatedResidual`: mixes the `hc` parallel streams into one input for a block
/// and says how much of the block's output goes back into each stream.
struct HyperConnection {
    norm_weight: Array,
    down: Proj,
    up: Proj,
    /// Absent on the final mixer, which only reads the streams.
    inject: Option<Proj>,
}

impl HyperConnection {
    fn load(
        weights: &NativeWeights,
        cfg: &Qwen4ExpConfig,
        base: &str,
        with_inject: bool,
    ) -> Result<Self> {
        Ok(Self {
            norm_weight: centered_norm(weights, cfg, &format!("{base}.hc_norm.weight"))?,
            down: Proj::load(weights, cfg, &format!("{base}.input_mix_weight_down"))?,
            up: Proj::load(weights, cfg, &format!("{base}.input_mix_weight_up"))?,
            inject: if with_inject {
                Some(Proj::load(
                    weights,
                    cfg,
                    &format!("{base}.block_inject_weight"),
                )?)
            } else {
                None
            },
        })
    }
}

struct Gdn {
    in_proj_qkv: Proj,
    in_proj_z: Proj,
    in_proj_b: Proj,
    in_proj_a: Proj,
    conv1d_weight: Array,
    a_log: Array,
    dt_bias: Array,
    norm_weight: Array,
    out_proj: Proj,
}

struct Indexer {
    index_qk_proj: Proj,
    q_layernorm: Array,
    k_layernorm: Array,
}

struct Qsa {
    q_proj: Proj,
    k_proj: Proj,
    v_proj: Proj,
    o_proj: Proj,
    q_norm: Array,
    k_norm: Array,
    indexer: Indexer,
}

impl Qsa {
    fn load(weights: &NativeWeights, cfg: &Qwen4ExpConfig, b: &str) -> Result<Self> {
        Ok(Self {
            q_proj: Proj::load(weights, cfg, &format!("{b}.q_proj"))?,
            k_proj: Proj::load(weights, cfg, &format!("{b}.k_proj"))?,
            v_proj: Proj::load(weights, cfg, &format!("{b}.v_proj"))?,
            o_proj: Proj::load(weights, cfg, &format!("{b}.o_proj"))?,
            q_norm: centered_norm(weights, cfg, &format!("{b}.q_norm.weight"))?,
            k_norm: centered_norm(weights, cfg, &format!("{b}.k_norm.weight"))?,
            indexer: Indexer {
                index_qk_proj: Proj::load(weights, cfg, &format!("{b}.indexer.index_qk_proj"))?,
                q_layernorm: centered_norm(
                    weights,
                    cfg,
                    &format!("{b}.indexer.q_layernorm.weight"),
                )?,
                k_layernorm: centered_norm(
                    weights,
                    cfg,
                    &format!("{b}.indexer.k_layernorm.weight"),
                )?,
            },
        })
    }
}

enum Mixer {
    Gdn(Gdn),
    Qsa(Qsa),
}

struct Gate8 {
    weight: Array,
    scales: Array,
    biases: Array,
    group_size: i32,
    bits: i32,
}

/// The router and the shared-expert gate: 8-bit affine in the mlx-community
/// build, float in the gs64 build.
enum Gate {
    Affine(Gate8),
    Dense(Array),
}

impl Gate {
    fn load(weights: &NativeWeights, cfg: &Qwen4ExpConfig, base: &str) -> Result<Self> {
        match Proj::load(weights, cfg, base)? {
            Proj::Dense(w_t) => Ok(Self::Dense(w_t)),
            Proj::Quant(l) => Ok(Self::Affine(Gate8 {
                biases: l
                    .biases
                    .ok_or_else(|| anyhow!("{base}: a quantized gate needs biases"))?,
                weight: l.weight,
                scales: l.scales,
                group_size: l.group_size,
                bits: l.bits,
            })),
        }
    }

    fn view(&self) -> GateProj<'_> {
        match self {
            Self::Affine(g) => GateProj::Affine(AffineGateWeights {
                weight: &g.weight,
                scales: &g.scales,
                biases: &g.biases,
                group_size: g.group_size,
                bits: g.bits,
            }),
            Self::Dense(w_t) => GateProj::Dense(w_t),
        }
    }
}

struct Stacked {
    weight: Array,
    scales: Array,
    biases: Option<Array>,
}

/// One layer's MoE block. The stacked routed-expert tensors are loaded lazily
/// and never evaluated when the experts are offloaded.
struct MoeLayer {
    layer: u16,
    gate: Gate,
    shared_gate: Gate,
    switch_gate: Stacked,
    switch_up: Stacked,
    switch_down: Stacked,
    switch_group_size: i32,
    switch_bits: i32,
    shared_gate_proj: ResolvedLinear,
    shared_up_proj: ResolvedLinear,
    shared_down_proj: ResolvedLinear,
    top_k: i32,
}

impl MoeLayer {
    fn load(weights: &NativeWeights, cfg: &Qwen4ExpConfig, layer: usize) -> Result<Self> {
        Self::load_at(
            weights,
            cfg,
            &format!("language_model.model.layers.{layer}.mlp"),
            layer as u16,
        )
    }

    /// The block whose tensors live under `base` (`<layer>.mlp`); `layer` names it to the
    /// offload engine (unused when the experts are resident).
    fn load_at(
        weights: &NativeWeights,
        cfg: &Qwen4ExpConfig,
        base: &str,
        layer: u16,
    ) -> Result<Self> {
        let stacked = |proj: &str| -> Result<Stacked> {
            let b = format!("{base}.switch_mlp.{proj}");
            Ok(Stacked {
                weight: require(weights, &format!("{b}.weight"))?,
                scales: require(weights, &format!("{b}.scales"))?,
                biases: weights.get(&format!("{b}.biases")).cloned(),
            })
        };
        let q = cfg.quant.for_base(&format!("{base}.switch_mlp"));
        Ok(Self {
            layer,
            gate: Gate::load(weights, cfg, &format!("{base}.gate"))?,
            shared_gate: Gate::load(weights, cfg, &format!("{base}.shared_expert_gate"))?,
            switch_gate: stacked("gate_proj")?,
            switch_up: stacked("up_proj")?,
            switch_down: stacked("down_proj")?,
            switch_group_size: q.group_size,
            switch_bits: q.bits,
            shared_gate_proj: quant_linear(
                weights,
                cfg,
                &format!("{base}.shared_expert.gate_proj"),
            )?,
            shared_up_proj: quant_linear(weights, cfg, &format!("{base}.shared_expert.up_proj"))?,
            shared_down_proj: quant_linear(
                weights,
                cfg,
                &format!("{base}.shared_expert.down_proj"),
            )?,
            top_k: cfg.num_experts_per_tok as i32,
        })
    }

    fn view<'a>(&'a self, offload: Option<&'a MoeOffload>) -> MoeBlockWeights<'a> {
        let proj = |s: &'a Stacked| SwitchExpertWeights {
            weight: &s.weight,
            scales: &s.scales,
            biases: s.biases.as_ref(),
        };
        MoeBlockWeights {
            gate: self.gate.view(),
            switch_glu: SwitchGluWeights {
                gate_proj: proj(&self.switch_gate),
                up_proj: proj(&self.switch_up),
                down_proj: proj(&self.switch_down),
                group_size: self.switch_group_size,
                bits: self.switch_bits,
                mode: MODE_AFFINE,
            },
            shared_expert: SharedMlpWeights {
                gate_proj_w: &self.shared_gate_proj.weight,
                gate_proj_s: &self.shared_gate_proj.scales,
                gate_proj_b: self.shared_gate_proj.biases.as_ref(),
                up_proj_w: &self.shared_up_proj.weight,
                up_proj_s: &self.shared_up_proj.scales,
                up_proj_b: self.shared_up_proj.biases.as_ref(),
                down_proj_w: &self.shared_down_proj.weight,
                down_proj_s: &self.shared_down_proj.scales,
                down_proj_b: self.shared_down_proj.biases.as_ref(),
                group_size: self.shared_gate_proj.group_size,
                bits: self.shared_gate_proj.bits,
                mode: MODE_AFFINE,
            },
            shared_gate: self.shared_gate.view(),
            top_k: self.top_k,
            norm_topk_prob: true,
            experts_in_scale_dtype: true,
            offload: offload.map(|offload| OffloadLayer {
                offload,
                layer: self.layer,
            }),
        }
    }
}

/// The n-gram ("PLE") injection of one layer.
struct Ple {
    key_proj: Proj,
    value_proj: Proj,
    norm_key: Array,
    norm_query: Array,
    norm_conv: Array,
    conv1d_weight: Array,
    hasher: NgramHasher,
    table: NgramTable,
    dilation: i32,
}

struct Layer {
    attn_hc: HyperConnection,
    mlp_hc: HyperConnection,
    mixer: Mixer,
    moe: MoeLayer,
    /// Index into [`Qwen4ExpCache::ple_conv`] when this layer injects n-grams.
    ple: Option<(usize, Ple)>,
}

/// What a layer's graph holds when it reaches the MoE block.
struct LayerFront {
    /// The streams after the mixer's write-back (the residual the MoE output is added to).
    h_mid: Array,
    /// The MoE block's input.
    x: Array,
    /// The write-back scale for the MoE block's output.
    inject: Array,
}

struct QuantEmbedding {
    weight: Array,
    scales: Array,
    biases: Option<Array>,
    group_size: i32,
    bits: i32,
}

pub struct Qwen4ExpModel {
    cfg: Qwen4ExpConfig,
    embed: QuantEmbedding,
    layers: Vec<Layer>,
    mixer: HyperConnection,
    lm_head: Option<Proj>,
    offload: Option<MoeOffload>,
    /// The MTP draft head, when its weights are next to the checkpoint and `LoadOptions::mtp`
    /// allows it.
    mtp: Option<MtpHead>,
    /// See [`LoadOptions::scores_budget_bytes`].
    scores_budget: u64,
    /// `ones(hidden)`, the weight of a group norm whose scale is applied afterwards.
    ones_hidden: Array,
    /// The image tower, under `LUMEN_VISION` for a checkpoint that ships one (see
    /// [`Self::forward_image_prompt`]).
    vision: Option<crate::qwen36_vision::NativeQwen36VisionTower>,
}

// ---- caches --------------------------------------------------------------------------------

#[derive(Clone)]
struct GdnState {
    conv: Option<Array>,
    state: Option<Array>,
}

#[derive(Clone)]
struct QsaState {
    kv: NativeKvCache,
    indexer_keys: Option<Array>,
}

#[derive(Clone)]
enum LayerState {
    Gdn(GdnState),
    Qsa(QsaState),
}

impl LayerState {
    fn arrays(&self) -> Vec<&Array> {
        match self {
            Self::Gdn(g) => g.conv.iter().chain(g.state.iter()).collect(),
            Self::Qsa(q) => {
                q.kv.keys()
                    .into_iter()
                    .chain(q.kv.values())
                    .chain(q.indexer_keys.iter())
                    .collect()
            }
        }
    }
}

/// Everything a sequence carries between forward calls. Cloning one is cheap and
/// safe: the arrays are immutable handles, and a clone that later appends makes its
/// own buffers rather than writing into the original's.
#[derive(Clone)]
pub struct Qwen4ExpCache {
    layers: Vec<LayerState>,
    /// Convolution history of each n-gram injection layer.
    ple_conv: Vec<Option<Array>>,
    /// The last `ngram_size - 1` tokens (EOS before the text starts).
    ngram_tail: Vec<u32>,
    tokens: usize,
    /// The draft head's cache, for a model that has one.
    mtp: Option<MtpState>,
    /// The 3-D rotary positions of a sequence that began with an image prompt; `None` for every
    /// text-only one, which keeps the 1-D rope.
    mrope: Option<MropeState>,
}

/// Every token's `[t, h, w]` rotary position so far, and what a text token after the prompt adds
/// to its index (`next_position - prompt length`).
#[derive(Clone)]
struct MropeState {
    positions: Vec<[i32; 3]>,
    delta: i32,
}

impl MropeState {
    /// Append the positions of `n` text tokens following the `fed` tokens this state covers.
    fn extend_text(&mut self, fed: usize, n: usize) {
        let delta = self.delta;
        self.positions.extend((fed..fed + n).map(|i| {
            let p = i as i32 + delta;
            [p, p, p]
        }));
    }
}

/// Where the tokens fed now sit for the rotary embedding.
#[derive(Clone, Copy)]
pub(crate) enum RopeAt<'a> {
    /// At their index: the fused 1-D rope from the cache's offset (every text-only sequence).
    Index,
    /// An image sequence: `new` holds the 3-D positions of the tokens fed now, `all` those of
    /// every token through them (the indexer places each pooled key block at its first token's).
    Mrope {
        new: &'a [[i32; 3]],
        all: &'a [[i32; 3]],
    },
}

/// An image prompt fed through [`Qwen4ExpModel::forward_image_prompt`]: every token's 3-D
/// position and the text delta after it, and each image's placeholder run (prompt-global) with
/// its features.
pub(crate) struct ImageFeed<'a> {
    positions: &'a [[i32; 3]],
    delta: i32,
    runs: &'a [(usize, usize)],
    soft: &'a [Array],
}

impl Qwen4ExpCache {
    /// This state as disk records: one `LayerMeta` per layer (`Linear` for DeltaNet,
    /// holding `[conv, state]`; `Full` for attention, holding `[keys, values,
    /// indexer keys]`) and one `Linear` more per n-gram injection layer (its
    /// convolution history). The n-gram tail is not stored: it is the last tokens of
    /// the prefix, which the manifest already holds.
    pub(crate) fn to_records(&self) -> Result<(Vec<LayerMeta>, Vec<ArrayRecord>)> {
        if self.mrope.is_some() {
            bail!("qwen4_exp: an image sequence's state is not persisted");
        }
        let mut records = Vec::new();
        let mut metas = Vec::with_capacity(self.layers.len() + self.ple_conv.len());
        for layer in &self.layers {
            match layer {
                LayerState::Gdn(g) => {
                    let mut m = LayerMeta::new(LayerKindTag::Linear, self.tokens);
                    m.arrays = vec![
                        push_record(&mut records, g.conv.as_ref())?,
                        push_record(&mut records, g.state.as_ref())?,
                    ];
                    metas.push(m);
                }
                LayerState::Qsa(q) => {
                    let mut m = LayerMeta::new(LayerKindTag::Full, q.kv.offset());
                    m.arrays = vec![
                        push_record(&mut records, q.kv.keys_view()?.as_ref())?,
                        push_record(&mut records, q.kv.values_view()?.as_ref())?,
                        push_record(&mut records, q.indexer_keys.as_ref())?,
                    ];
                    metas.push(m);
                }
            }
        }
        for conv in &self.ple_conv {
            let mut m = LayerMeta::new(LayerKindTag::Linear, self.tokens);
            m.arrays = vec![push_record(&mut records, conv.as_ref())?];
            metas.push(m);
        }
        // The draft head's cache goes last: without it a restored sequence cannot speculate.
        if let Some(h) = &self.mtp {
            let mut m = LayerMeta::new(LayerKindTag::Full, h.qsa.kv.offset());
            m.arrays = vec![
                push_record(&mut records, h.qsa.kv.keys_view()?.as_ref())?,
                push_record(&mut records, h.qsa.kv.values_view()?.as_ref())?,
                push_record(&mut records, h.qsa.indexer_keys.as_ref())?,
                push_record(&mut records, h.last_multi.as_ref())?,
            ];
            metas.push(m);
        }
        Ok((metas, records))
    }

    /// Rebuild a state written by [`Self::to_records`] into `fresh` (a cache just
    /// made by the same model), `tokens` being the prefix it was captured after.
    pub(crate) fn restore_records<T: RecordArray>(
        mut fresh: Self,
        metas: &[LayerMeta],
        records: &[T],
        tokens: &[u32],
        ngram_context: usize,
        eos: u32,
    ) -> Result<Self> {
        let want = fresh.layers.len() + fresh.ple_conv.len();
        // One more record than the trunk's is the draft head's cache.
        let has_head_state = metas.len() == want + 1;
        if metas.len() != want && !has_head_state {
            bail!(
                "persisted state has {} layers, the model has {want}",
                metas.len()
            );
        }
        let slot = |m: &LayerMeta, i: usize| -> Result<Option<Array>> {
            array_at(records, m.arrays.get(i).copied().flatten())
        };
        let n_layers = fresh.layers.len();
        for (i, layer) in fresh.layers.iter_mut().enumerate() {
            let m = &metas[i];
            match (layer, m.kind) {
                (LayerState::Gdn(g), LayerKindTag::Linear) => {
                    g.conv = slot(m, 0)?;
                    g.state = slot(m, 1)?;
                }
                (LayerState::Qsa(q), LayerKindTag::Full) => {
                    q.kv.set_state(slot(m, 0)?, slot(m, 1)?, m.offset);
                    q.indexer_keys = slot(m, 2)?;
                }
                _ => bail!(
                    "persisted layer {i} is a {:?}, not the layer kind the model has",
                    m.kind
                ),
            }
        }
        for (k, conv) in fresh.ple_conv.iter_mut().enumerate() {
            *conv = slot(&metas[n_layers + k], 0)?;
        }
        if let Some(head) = fresh.mtp.as_mut() {
            if has_head_state {
                let m = &metas[want];
                head.qsa.kv.set_state(slot(m, 0)?, slot(m, 1)?, m.offset);
                head.qsa.indexer_keys = slot(m, 2)?;
                head.last_multi = slot(m, 3)?;
            } else {
                // A state saved without the head cannot be drafted from.
                fresh.mtp = None;
            }
        }
        fresh.tokens = tokens.len();
        // The n-gram hashes see the last `ngram_context` tokens, EOS before the start.
        fresh.ngram_tail = if tokens.len() >= ngram_context {
            tokens[tokens.len() - ngram_context..].to_vec()
        } else {
            let mut t = vec![eos; ngram_context - tokens.len()];
            t.extend_from_slice(tokens);
            t
        };
        Ok(fresh)
    }

    /// Every array this state holds, to evaluate together with a forward pass.
    ///
    /// The new state is not an input of the logits, so nothing evaluates it: left lazy
    /// it keeps the graph that computes it, and that graph's inputs (a chunk-sized
    /// convolution input per layer, ~2 MB per prompt token over the model), alive until
    /// the next chunk happens to consume it.
    fn arrays(&self) -> Vec<&Array> {
        let mut out: Vec<&Array> = Vec::new();
        for layer in &self.layers {
            out.extend(layer.arrays());
        }
        out.extend(self.ple_conv.iter().flatten());
        if let Some(m) = &self.mtp {
            out.extend(m.arrays());
        }
        out
    }

    /// Bytes of the buffers this state references. Two states cloned from one share the buffers
    /// neither has appended to, so the sum over several states can count a buffer twice.
    pub(crate) fn nbytes(&self) -> u64 {
        self.arrays().iter().map(|a| a.nbytes() as u64).sum()
    }

    /// Tokens consumed so far.
    pub fn len(&self) -> usize {
        self.tokens
    }

    pub fn is_empty(&self) -> bool {
        self.tokens == 0
    }
}

// ---- load ----------------------------------------------------------------------------------

/// Where the n-gram table's rows are served from.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum NgramSource {
    /// Read from disk when the table is larger than 1 GiB (the shipped model's is
    /// 32 GB), hold it in MLX arrays otherwise.
    #[default]
    Auto,
    Resident,
    File,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LoadOptions {
    /// Serve the routed experts from an SSD-backed slot pool of about this many bytes.
    pub expert_budget_bytes: Option<u64>,
    pub ngram: NgramSource,
    /// Load the MTP draft head (`mtp.safetensors`) when present.
    pub mtp: MtpMode,
    /// Most bytes of attention scores a prefill chunk may hold at once (0 = the default,
    /// [`DEFAULT_SCORES_BUDGET`]): a chunk against a long context is attended in blocks of query
    /// rows so the scores (`heads x rows x context x 4 B`) stay under it.
    pub scores_budget_bytes: u64,
}

/// Fewest query rows attended together: a block this small is all kernel launches.
const MIN_ATTENTION_ROWS: i64 = 64;

/// Scores a chunk may hold at once when nothing says otherwise: an 18K-token context against a
/// 1,024-token chunk is 1.8 GB and runs in one piece.
const DEFAULT_SCORES_BUDGET: u64 = 2_000_000_000;

impl Qwen4ExpModel {
    /// Load a checkpoint directory. `expert_budget_bytes` serves the routed
    /// experts from an SSD-backed slot pool of about that size instead of
    /// holding them in memory.
    pub fn load(model_dir: &Path, expert_budget_bytes: Option<u64>) -> Result<Self> {
        Self::load_with(
            model_dir,
            LoadOptions {
                expert_budget_bytes,
                ngram: NgramSource::Auto,
                mtp: MtpMode::Off,
                scores_budget_bytes: 0,
            },
        )
    }

    pub fn load_with(model_dir: &Path, options: LoadOptions) -> Result<Self> {
        let expert_budget_bytes = options.expert_budget_bytes;
        let mut cfg = Qwen4ExpConfig::load(model_dir)?;
        let mut weights = NativeWeights::load_dir(model_dir)?;
        // The image tower is read from the raw weights: `sanitize` drops `vision_tower.*`.
        let vision = match &cfg.vision {
            Some(vcfg)
                if crate::vision_splice::tower::vision_enabled()
                    && weights.get("vision_tower.pos_embed.weight").is_some() =>
            {
                let tower = crate::qwen36_vision::NativeQwen36VisionTower::load(
                    weights.tensors(),
                    vcfg.clone(),
                )
                .context("qwen4_exp: load the vision tower")?;
                // Settled on the loading thread, as the trunk is (see `materialize_trunk`).
                mlx_rs::transforms::eval(tower.weights())
                    .context("qwen4_exp: read the vision tower")?;
                let bytes: usize = tower.weights().iter().map(|a| a.nbytes()).sum();
                eprintln!(
                    "[qwen4_exp] vision tower loaded ({:.2} GB)",
                    bytes as f64 / 1e9
                );
                Some(tower)
            }
            Some(_) if crate::vision_splice::tower::vision_enabled() => {
                eprintln!(
                    "[qwen4_exp] LUMEN_VISION=1 but the checkpoint ships no vision_tower.* weights: image input stays off"
                );
                None
            }
            _ => None,
        };
        weights.sanitize()?;
        // `mtp.safetensors` sits in the same directory, so the loader picked it up under
        // `language_model.mtp.*`. The head is its own thing: take it out before the trunk is read.
        let head_tensors: HashMap<String, Array> = weights
            .take_prefix("language_model.mtp.")
            .into_iter()
            .filter_map(|(k, v)| {
                k.strip_prefix("language_model.")
                    .map(|k| (k.to_string(), v))
            })
            .collect();
        materialize_trunk(&weights, expert_budget_bytes.is_none())
            .context("qwen4_exp: read the resident weights")?;
        cfg.norms_centered = detect_centered_norms(&weights, &cfg)?;
        if cfg.norms_centered {
            eprintln!(
                "[qwen4_exp] norm weights are zero-centered (HF convention): adding 1 at load"
            );
        }

        let embed_base = "language_model.model.embed_tokens";
        let eq = cfg.quant.for_base(embed_base);
        let embed = QuantEmbedding {
            weight: require(&weights, &format!("{embed_base}.weight"))?,
            scales: require(&weights, &format!("{embed_base}.scales"))?,
            biases: weights.get(&format!("{embed_base}.biases")).cloned(),
            group_size: eq.group_size,
            bits: eq.bits,
        };

        let ple_layers = cfg.ple_layers();
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for (i, kind) in cfg.layer_kinds.iter().enumerate() {
            let base = format!("language_model.model.layers.{i}");
            let mixer = match kind {
                LayerKind::LinearAttention => {
                    let b = format!("{base}.linear_attn");
                    Mixer::Gdn(Gdn {
                        in_proj_qkv: Proj::load(&weights, &cfg, &format!("{b}.in_proj_qkv"))?,
                        in_proj_z: Proj::load(&weights, &cfg, &format!("{b}.in_proj_z"))?,
                        in_proj_b: Proj::load(&weights, &cfg, &format!("{b}.in_proj_b"))?,
                        in_proj_a: Proj::load(&weights, &cfg, &format!("{b}.in_proj_a"))?,
                        conv1d_weight: require(&weights, &format!("{b}.conv1d.weight"))?,
                        a_log: require(&weights, &format!("{b}.A_log"))?,
                        dt_bias: require(&weights, &format!("{b}.dt_bias"))?,
                        norm_weight: require(&weights, &format!("{b}.norm.weight"))?,
                        out_proj: Proj::load(&weights, &cfg, &format!("{b}.out_proj"))?,
                    })
                }
                LayerKind::FullAttention => {
                    Mixer::Qsa(Qsa::load(&weights, &cfg, &format!("{base}.self_attn"))?)
                }
            };
            let ple = match ple_layers.iter().position(|&l| l == i) {
                Some(k) => Some((
                    k,
                    load_ple(&weights, &cfg, &base, model_dir, options.ngram)?,
                )),
                None => None,
            };
            layers.push(Layer {
                attn_hc: HyperConnection::load(
                    &weights,
                    &cfg,
                    &format!("{base}.attn_hyper_connection"),
                    true,
                )?,
                mlp_hc: HyperConnection::load(
                    &weights,
                    &cfg,
                    &format!("{base}.mlp_hyper_connection"),
                    true,
                )?,
                mixer,
                moe: MoeLayer::load(&weights, &cfg, i)?,
                ple,
            });
        }

        let mixer = HyperConnection::load(
            &weights,
            &cfg,
            "language_model.model.hyper_connection_mixer",
            false,
        )?;
        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(Proj::load(&weights, &cfg, "language_model.lm_head")?)
        };
        let want_head = options.mtp != MtpMode::Off && !head_tensors.is_empty();
        let offload = match expert_budget_bytes {
            Some(bytes) => {
                // The draft head's experts go through the same pool as the trunk's: their 1.4 GB
                // would otherwise be resident, taken from it.
                let io = IoTuning {
                    mtp_experts: want_head,
                    ..IoTuning::from_env()?
                };
                Some(
                    MoeOffload::open(model_dir, bytes, cfg.num_experts_per_tok, io)
                        .context("qwen4_exp: expert offload")?,
                )
            }
            None => None,
        };
        let ones_hidden = mlx_rs::ops::ones::<f32>(&[cfg.hidden_size as i32])?;
        ones_hidden.eval()?;
        let mtp = if want_head {
            let head = MtpHead::load(head_tensors, model_dir, &cfg, offload.is_some())
                .context("qwen4_exp: MTP head")?;
            eprintln!(
                "[qwen4_exp] MTP draft head loaded ({:.2} GB resident{})",
                head.resident_bytes as f64 / 1e9,
                if head.streamed {
                    ", its experts in the pool"
                } else {
                    ""
                }
            );
            Some(head)
        } else {
            None
        };
        Ok(Self {
            cfg,
            embed,
            layers,
            mixer,
            lm_head,
            offload,
            mtp,
            scores_budget: if options.scores_budget_bytes == 0 {
                DEFAULT_SCORES_BUDGET
            } else {
                options.scores_budget_bytes
            },
            ones_hidden,
            vision,
        })
    }

    pub fn config(&self) -> &Qwen4ExpConfig {
        &self.cfg
    }

    /// Forget the per-call time split so far (see [`Self::offload_report`]).
    pub fn offload_reset_timing(&self) {
        if let Some(o) = &self.offload {
            o.reset_timing();
        }
    }

    /// Hit rate, per-call time split and read behaviour of the expert bank.
    pub fn offload_report(&self) -> Option<String> {
        self.offload.as_ref().map(|o| match probe::report() {
            Some(profile) => format!("{}\n{profile}", o.report()),
            None => o.report(),
        })
    }

    /// Rows requested from the file-backed n-gram table and how many of them
    /// had to be read from disk; `None` when the table is resident.
    pub fn ngram_stats(&self) -> Option<super::ngram_table::RowStats> {
        self.layers.iter().find_map(|l| match &l.ple {
            Some((
                _,
                Ple {
                    table: NgramTable::File(t),
                    ..
                },
            )) => Some(t.stats()),
            _ => None,
        })
    }

    /// Hit/miss counts of the expert bank when the experts are offloaded.
    pub fn offload_stats(&self) -> Option<(u64, u64)> {
        self.offload.as_ref().map(|o| {
            let s = o.stats();
            (s.hits, s.misses)
        })
    }

    /// A cache holding the state `to_records` wrote after `tokens`.
    pub(crate) fn cache_from_records<T: RecordArray>(
        &self,
        metas: &[LayerMeta],
        records: &[T],
        tokens: &[u32],
    ) -> Result<Qwen4ExpCache> {
        Qwen4ExpCache::restore_records(
            self.make_cache(),
            metas,
            records,
            tokens,
            self.cfg.ngram_size - 1,
            self.cfg.eos_token_id,
        )
    }

    pub fn make_cache(&self) -> Qwen4ExpCache {
        let layers = self
            .cfg
            .layer_kinds
            .iter()
            .map(|k| match k {
                LayerKind::LinearAttention => LayerState::Gdn(GdnState {
                    conv: None,
                    state: None,
                }),
                LayerKind::FullAttention => LayerState::Qsa(QsaState {
                    kv: NativeKvCache::new(),
                    indexer_keys: None,
                }),
            })
            .collect();
        Qwen4ExpCache {
            layers,
            ple_conv: vec![None; self.cfg.ple_layers().len()],
            ngram_tail: vec![self.cfg.eos_token_id; self.cfg.ngram_size - 1],
            tokens: 0,
            mtp: self.mtp.as_ref().map(|_| MtpState::new()),
            mrope: None,
        }
    }
}

fn load_ple(
    weights: &NativeWeights,
    cfg: &Qwen4ExpConfig,
    layer_base: &str,
    model_dir: &Path,
    source: NgramSource,
) -> Result<Ple> {
    let base = format!("{layer_base}.ple");
    let emb = format!("{base}.ple_embedding");
    let ints = |name: &str| -> Result<Vec<u64>> {
        let a = require(weights, name)?;
        a.eval().with_context(|| format!("evaluate {name}"))?;
        Ok(a.try_as_slice::<i64>()
            .map_err(|e| anyhow!("read {name}: {e:?}"))?
            .iter()
            .map(|&v| v as u64)
            .collect())
    };
    let hasher = NgramHasher::new(
        cfg.ngram_size,
        cfg.heads_per_ngram,
        cfg.eos_token_id,
        ints(&format!("{emb}.layer_multipliers"))?,
        ints(&format!("{emb}.ngram_heads_vocab_sizes"))?,
        ints(&format!("{emb}.ngram_heads_offsets"))?,
    )?;
    // The mlx-community build names the shards `shards.<i>`, the gs64 build `shard_<i>`.
    let style = if weights
        .get(&format!("{emb}.ngram_embedding.shards.0.weight"))
        .is_some()
    {
        |s: usize| format!("shards.{s}")
    } else {
        |s: usize| format!("shard_{s}")
    };
    let shard_base = |s: usize| format!("{emb}.ngram_embedding.{}", style(s));
    let first = shard_base(0);
    let first_w = require(weights, &format!("{first}.weight"))?;
    let first_s = require(weights, &format!("{first}.scales"))?;
    // A row is one hash head's slice of the embedding; its bits are the quantization config's
    // (4 in both 4-bit builds, which name the shards, 3 in a uniform 3-bit build, which takes the
    // default), and the shapes have to agree, as the group size is what they say.
    let row_dim = (cfg.ple_embed_dim / cfg.ngram_heads()) as i64;
    let bits = cfg.quant.for_base(&first).bits;
    let packed = *first_w.shape().last().unwrap_or(&0) as i64;
    let groups = *first_s.shape().last().unwrap_or(&0) as i64;
    if packed * 32 != row_dim * i64::from(bits) || groups == 0 || row_dim % groups != 0 {
        bail!(
            "{layer_base}: n-gram shards of {packed} packed columns and {groups} scale groups \
             do not describe {row_dim}-wide {bits}-bit rows"
        );
    }
    let group_size = (row_dim / groups) as i32;
    let rows_per_shard = first_w.shape()[0] as u64;
    let need = hasher.total_rows();
    let have = rows_per_shard * cfg.split_ngram_parts as u64;
    if need > have {
        bail!(
            "layer {layer_base}: the n-gram heads address {need} rows but the table holds {have}"
        );
    }
    let row_bytes = (first_w.shape()[1] as u64) * 4
        + 2 * (first_s.shape()[1] as u64)
            * if first_s.dtype() == Dtype::Float32 {
                4
            } else {
                2
            };
    let table_bytes = have * row_bytes;
    let from_disk = match source {
        NgramSource::File => true,
        NgramSource::Resident => false,
        NgramSource::Auto => table_bytes > (1 << 30),
    };
    let table = if from_disk {
        NgramTable::File(FileTable::open(
            model_dir,
            cfg.split_ngram_parts,
            &shard_base,
            group_size,
            bits,
            DEFAULT_ROW_CACHE,
            IoMode::Buffered,
        )?)
    } else {
        let mut shards = Vec::with_capacity(cfg.split_ngram_parts);
        for s in 0..cfg.split_ngram_parts {
            let b = shard_base(s);
            shards.push(ShardArrays {
                weight: require(weights, &format!("{b}.weight"))?,
                scales: require(weights, &format!("{b}.scales"))?,
                biases: require(weights, &format!("{b}.biases"))?,
            });
        }
        // A resident table is read here, on the loading thread (see `materialize_trunk`).
        let arrays: Vec<&Array> = shards
            .iter()
            .flat_map(|s| [&s.weight, &s.scales, &s.biases])
            .collect();
        mlx_rs::transforms::eval(arrays).context("read the resident n-gram table")?;
        NgramTable::Resident(ResidentTable::new(
            shards,
            rows_per_shard,
            group_size,
            bits,
        )?)
    };
    Ok(Ple {
        key_proj: Proj::load(weights, cfg, &format!("{base}.key_proj"))?,
        value_proj: Proj::load(weights, cfg, &format!("{base}.value_proj"))?,
        norm_key: centered_norm(weights, cfg, &format!("{base}.norm_key.weight"))?,
        norm_query: centered_norm(weights, cfg, &format!("{base}.norm_query.weight"))?,
        norm_conv: centered_norm(weights, cfg, &format!("{base}.norm_conv.weight"))?,
        conv1d_weight: require(weights, &format!("{base}.conv1d.weight"))?,
        hasher,
        table,
        dilation: cfg.ngram_size as i32,
    })
}

/// `sdpa_with_mask`, `rows` query rows at a time. A row's attention does not depend on the
/// other rows, so blocks are the same attention in pieces; each block's scores are freed before
/// the next block's are made, which bounds them by `heads x rows x kv_len x 4` bytes.
fn sdpa_with_mask_in_blocks(
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f32,
    mask: &Array,
    rows: i64,
) -> Result<Array> {
    let s = i64::from(q.shape()[2]);
    if rows >= s {
        return sdpa_with_mask(q, k, v, scale, mask);
    }
    let mut blocks = Vec::with_capacity((s / rows + 1) as usize);
    let mut r0 = 0i64;
    while r0 < s {
        let r1 = (r0 + rows).min(s);
        let range = (r0 as i32)..(r1 as i32);
        blocks.push(sdpa_with_mask(
            &q.try_index((.., .., range.clone(), ..))?,
            k,
            v,
            scale,
            &mask.try_index((.., .., range, ..))?,
        )?);
        r0 = r1;
    }
    let refs: Vec<&Array> = blocks.iter().collect();
    Ok(mlx_rs::ops::concatenate_axis(&refs, 2)?)
}

// ---- forward -------------------------------------------------------------------------------

/// What [`Qwen4ExpModel::forward_verify`] returns.
pub(crate) struct Verified {
    /// `[1, 2, vocab]` f32: row 0 predicts the token after the pending one, row 1 the token
    /// after the draft.
    pub logits: Array,
    /// `[1, 2, hc*hidden]`: the final residual streams of the two rows (the draft head's input).
    pub multi: Array,
    /// The sequence's state after the first row alone: what stays if the draft is rejected.
    pub first: Qwen4ExpCache,
}

/// Tokens a layer-major prefill keeps in flight: each layer's experts are read from the SSD
/// once per group this size, and the group's residual streams (40 KB a token) wait in memory
/// between layers. A group is long enough that a layer's compute outlasts the read of the next.
pub(crate) const PREFILL_GROUP_TOKENS: usize = 8192;

/// A pass of at most this many rows is a decode step or a speculative verify: it keeps nothing
/// alive between layers worth settling.
const DECODE_ROWS: i32 = 4;

/// One sequence of a joint feed ([`Qwen4ExpModel::forward_many`]): `ids` fed into `cache` in
/// the chunks that end at `cuts`, keeping the state at `mark` (one of the cuts) when given.
pub(crate) struct JointFeed<'a> {
    pub(crate) ids: &'a [u32],
    pub(crate) cache: &'a mut Qwen4ExpCache,
    pub(crate) cuts: &'a [usize],
    pub(crate) mark: Option<usize>,
}

/// One feed's part of a layer-major group: what `forward_group` was given for it.
struct GroupFeed<'a> {
    ids: &'a [u32],
    cache: &'a mut Qwen4ExpCache,
    cuts: &'a [usize],
    marks: &'a [usize],
    want_logits: bool,
    /// The image prompt these tokens are part of (the cache started with it).
    image: Option<&'a ImageFeed<'a>>,
}

/// One feed's working state while its group goes through the layers.
struct GroupWork {
    base_tokens: usize,
    /// The n-gram tail the cache had, then the feed's tokens.
    history: Vec<u32>,
    /// Each chunk's `(start, end)` in the feed.
    bounds: Vec<(usize, usize)>,
    /// Each chunk's residual streams, layer by layer.
    streams: Vec<Array>,
    /// The chunk each mark ends (the callers only mark cuts).
    mark_pieces: Vec<Option<usize>>,
    snap_layers: Vec<Vec<LayerState>>,
    snap_ple: Vec<Vec<Option<Array>>>,
    /// An image sequence's 3-D positions through this group's last token, and its text delta.
    mrope: Option<MropeState>,
}

impl GroupWork {
    fn new(model: &Qwen4ExpModel, f: &GroupFeed<'_>) -> Result<Self> {
        let mut history = f.cache.ngram_tail.clone();
        history.extend_from_slice(f.ids);
        let mut start = 0;
        let bounds: Vec<(usize, usize)> = f
            .cuts
            .iter()
            .map(|&end| {
                let b = (start, end);
                start = end;
                b
            })
            .collect();
        let base = f.cache.tokens;
        let streams = match f.image {
            None => bounds
                .iter()
                .map(|&(s, e)| model.embed_streams(&f.ids[s..e]))
                .collect::<Result<Vec<_>>>()?,
            Some(img) => bounds
                .iter()
                .map(|&(s, e)| model.embed_image_streams(&f.ids[s..e], base + s, img))
                .collect::<Result<Vec<_>>>()?,
        };
        let mrope = match (f.image, &f.cache.mrope) {
            (Some(img), _) => {
                let end = base + f.ids.len();
                if base > 0 && f.cache.mrope.is_none() || img.positions.len() < end {
                    bail!(
                        "qwen4_exp: an image prompt starts a fresh sequence and has a position per token"
                    );
                }
                Some(MropeState {
                    positions: img.positions[..end].to_vec(),
                    delta: img.delta,
                })
            }
            (None, Some(m)) => {
                let mut m = m.clone();
                m.extend_text(base, f.ids.len());
                Some(m)
            }
            (None, None) => None,
        };
        Ok(Self {
            base_tokens: f.cache.tokens,
            history,
            bounds,
            streams,
            mark_pieces: f
                .marks
                .iter()
                .map(|m| f.cuts.iter().position(|c| c == m))
                .collect(),
            snap_layers: vec![Vec::new(); f.marks.len()],
            snap_ple: vec![vec![None; f.cache.ple_conv.len()]; f.marks.len()],
            mrope,
        })
    }
}

/// Whether a group feeding several chunks through each layer stages the layer for all of them
/// ([`BulkHint`]): only when one chunk is big enough to take the staging path for its size alone,
/// as every prefill of more than one chunk at the planned chunk sizes has. Staging reads the
/// whole layer, which pays for such a chunk and lets the others gather from it. Chunks that are
/// all small (short feeds fed together) each take their own path instead, for most the pool's,
/// the later ones finding there the experts the first read: forcing a whole-layer read for those
/// made two feeds of 74 and 58 tokens take 11.4 s together against 7.2 s apart (Flash-Next,
/// 2.9 GB pool). Either way each call's output is the same (see `offload::stages_for_size`).
fn stages_every_chunk(
    chunk_rows: impl IntoIterator<Item = usize>,
    top_k: usize,
    experts: usize,
) -> bool {
    chunk_rows
        .into_iter()
        .any(|rows| crate::offload::stages_for_size(rows * top_k, experts))
}

/// Routes every layer call to the staging path while it lives (see `MoeOffload::set_bulk_hint`).
struct BulkHint<'a>(&'a MoeOffload);

impl<'a> BulkHint<'a> {
    fn on(offload: &'a MoeOffload) -> Self {
        offload.set_bulk_hint(true);
        Self(offload)
    }
}

impl Drop for BulkHint<'_> {
    fn drop(&mut self) {
        self.0.set_bulk_hint(false);
    }
}

impl Qwen4ExpModel {
    /// Logits for `ids`, appended to the sequence `cache` holds: `[1, L, vocab]`,
    /// or `[1, 1, vocab]` for the last position when `last_only`.
    pub fn forward(
        &self,
        ids: &[u32],
        cache: &mut Qwen4ExpCache,
        last_only: bool,
    ) -> Result<Array> {
        if ids.is_empty() {
            bail!("qwen4_exp forward: no tokens");
        }
        if cache.layers.len() != self.layers.len() {
            bail!("qwen4_exp forward: cache has {} layers", cache.layers.len());
        }
        let cfg = &self.cfg;
        let l = ids.len() as i32;

        // Tokens the n-gram hashes see: the carried tail, then this call's tokens.
        let mut history = cache.ngram_tail.clone();
        history.extend_from_slice(ids);

        probe::begin_step(ids.len());
        let mut h = self.embed_streams(ids)?;
        probe::stage(Stage::Embed, &[&h])?;
        if let Some(m) = cache.mrope.as_mut() {
            m.extend_text(cache.tokens, ids.len());
        }
        let at = match &cache.mrope {
            Some(m) => RopeAt::Mrope {
                new: &m.positions[cache.tokens..],
                all: &m.positions,
            },
            None => RopeAt::Index,
        };
        let ahead = match &self.offload {
            Some(off) if self.builds_ahead(ids.len()) && cache.mrope.is_none() => Some(off),
            _ => None,
        };
        if let Some(off) = ahead {
            h = self.layers_ahead(off, h, ids.len(), &history, cache)?;
        } else {
            for (i, layer) in self.layers.iter().enumerate() {
                if let Some((k, ple)) = &layer.ple {
                    let injected = self
                        .ple_forward(ple, &h, &history, ids.len(), &mut cache.ple_conv[*k], None)
                        .with_context(|| format!("layer {i}: n-gram injection"))?;
                    h = mlx_rs::ops::add(&h, &injected)?;
                    probe::stage(Stage::Ple, &[&h])?;
                }
                h = self
                    .layer_forward(layer, &h, &mut cache.layers[i], None, at)
                    .with_context(|| format!("layer {i}"))?;
            }
        }

        let first = cache.tokens == 0;
        if let Some(st) = cache.mtp.as_mut() {
            self.mtp_consume(st, ids, &h, first)
                .context("qwen4_exp: feed the draft head")?;
        }
        let h = if last_only && l > 1 {
            h.try_index((.., (l - 1)..l, ..))?
        } else {
            h
        };
        let (mixed, _) = self.hyper_mix(&self.mixer, &h)?;
        let logits = match &self.lm_head {
            Some(head) => head.apply(&mixed)?,
            None => bail!("qwen4_exp: tied embeddings are not supported"),
        };

        cache.tokens += ids.len();
        let keep = cfg.ngram_size - 1;
        cache.ngram_tail = history[history.len() - keep..].to_vec();
        let logits = to_f32(logits)?;
        let mut pending = cache.arrays();
        pending.push(&logits);
        probe::stage(Stage::Tail, &pending)?;
        mlx_rs::transforms::eval(pending).context("qwen4_exp: evaluate the step and its state")?;
        probe::end_step();
        Ok(logits)
    }

    /// Two rows `[pending token, draft]` through the model in one pass, keeping what a
    /// rejected draft needs: the state as it stood after the first row alone. `cache` ends
    /// after both rows. The draft head's cache is untouched here (the caller finishes the
    /// round with [`Self::mtp_finish_round`]).
    pub(crate) fn forward_verify(
        &self,
        ids: &[u32],
        cache: &mut Qwen4ExpCache,
    ) -> Result<Verified> {
        if ids.len() != 2 {
            bail!("qwen4_exp verify: {} rows, expected 2", ids.len());
        }
        if cache.layers.len() != self.layers.len() {
            bail!("qwen4_exp verify: cache has {} layers", cache.layers.len());
        }
        if cache.mrope.is_some() {
            bail!("qwen4_exp verify: an image sequence decodes one token at a time");
        }
        let keep = self.cfg.ngram_size - 1;
        let base_tokens = cache.tokens;
        let mut history = cache.ngram_tail.clone();
        history.extend_from_slice(ids);

        let mut h = self.embed_streams(ids)?;
        let mut gdn_first: Vec<Option<GdnState>> = vec![None; self.layers.len()];
        let mut ple_first: Vec<Option<Array>> = vec![None; cache.ple_conv.len()];
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some((k, ple)) = &layer.ple {
                let injected = self
                    .ple_forward(
                        ple,
                        &h,
                        &history,
                        2,
                        &mut cache.ple_conv[*k],
                        Some(&mut ple_first[*k]),
                    )
                    .with_context(|| format!("layer {i}: n-gram injection"))?;
                h = mlx_rs::ops::add(&h, &injected)?;
            }
            h = self
                .layer_forward(
                    layer,
                    &h,
                    &mut cache.layers[i],
                    Some(&mut gdn_first[i]),
                    RopeAt::Index,
                )
                .with_context(|| format!("layer {i}"))?;
        }
        let multi = h.clone();
        let (mixed, _) = self.hyper_mix(&self.mixer, &h)?;
        let head = self
            .lm_head
            .as_ref()
            .ok_or_else(|| anyhow!("qwen4_exp: tied embeddings are not supported"))?;
        let logits = to_f32(head.apply(&mixed)?)?;
        cache.tokens += 2;
        cache.ngram_tail = history[history.len() - keep..].to_vec();

        // The state after the first row alone.
        let mut layers = Vec::with_capacity(cache.layers.len());
        for (i, st) in cache.layers.iter().enumerate() {
            layers.push(match st {
                LayerState::Gdn(_) => LayerState::Gdn(
                    gdn_first[i]
                        .take()
                        .ok_or_else(|| anyhow!("layer {i}: the first row's state was not kept"))?,
                ),
                LayerState::Qsa(q) => {
                    let n = q.kv.offset();
                    let mut q1 = q.clone();
                    let (k, v) = q.kv.prefix_views(n - 1)?;
                    q1.kv.set_state(k, v, n - 1);
                    q1.indexer_keys = match q.indexer_keys.as_ref() {
                        Some(keys) => Some(owned(&keys.try_index((.., ..(n as i32 - 1), ..))?)?),
                        None => None,
                    };
                    LayerState::Qsa(q1)
                }
            });
        }
        let first = Qwen4ExpCache {
            layers,
            ple_conv: ple_first,
            ngram_tail: history[1..1 + keep].to_vec(),
            tokens: base_tokens + 1,
            mtp: cache.mtp.clone(),
            mrope: None,
        };
        let mut pending = cache.arrays();
        pending.extend(first.arrays());
        pending.push(&logits);
        pending.push(&multi);
        mlx_rs::transforms::eval(pending).context("qwen4_exp: evaluate the verify pass")?;
        Ok(Verified {
            logits,
            multi,
            first,
        })
    }

    /// The residual streams `[1, L, hc * hidden]` a token run starts from: its embeddings,
    /// repeated once per stream.
    fn embed_streams(&self, ids: &[u32]) -> Result<Array> {
        let cfg = &self.cfg;
        let embedded = to_f32(self.embed_untiled(ids)?)?;
        Ok(mlx_rs::ops::tile(&embedded, &[1, 1, cfg.hc_count as i32])?)
    }

    /// The dequantized embedding rows of `ids`, `[1, L, hidden]`.
    fn embed_untiled(&self, ids: &[u32]) -> Result<Array> {
        let cfg = &self.cfg;
        let (l, d) = (ids.len() as i32, cfg.hidden_size as i32);
        if let Some(&bad) = ids.iter().find(|&&t| t as usize >= cfg.vocab_size) {
            bail!(
                "qwen4_exp: token id {bad} is outside the {}-token vocabulary",
                cfg.vocab_size
            );
        }
        let ids_i32: Vec<i32> = ids.iter().map(|&t| t as i32).collect();
        let ids_arr = Array::from_slice(&ids_i32, &[l]);
        let embedded = quantized_embedding_lookup_with_mode(
            &self.embed.weight,
            &self.embed.scales,
            self.embed.biases.as_ref(),
            &ids_arr,
            self.embed.group_size,
            self.embed.bits,
            MODE_AFFINE,
        )
        .context("qwen4_exp: embed_tokens lookup")?;
        Ok(mlx_rs::ops::reshape(&embedded, &[1, l, d])?)
    }

    /// Feed `ids` as several chunks (ending at the offsets `cuts`, the last being `ids.len()`),
    /// layer by layer instead of chunk by chunk: every chunk goes through layer 0, then every
    /// chunk through layer 1, and so on, so a layer's experts are read from the SSD once for
    /// the whole group of chunks rather than once per chunk. The arithmetic of each (layer,
    /// chunk) is the chunk-by-chunk one, with the same chunk boundaries, so the logits and the
    /// state come out bit-for-bit the same as `forward` over the same chunks.
    ///
    /// Returns the last position's logits `[1, 1, vocab]`, and, when `mark` names one of the
    /// cuts, a copy of the state as it stood after the tokens up to that cut: the rollback
    /// point a chunk-by-chunk feed would have taken there.
    ///
    /// The residual streams of a group wait in memory between layers (40 KB a token), so a
    /// longer prompt goes through in groups of `PREFILL_GROUP_TOKENS`.
    pub fn forward_pieces(
        &self,
        ids: &[u32],
        cache: &mut Qwen4ExpCache,
        cuts: &[usize],
        mark: Option<usize>,
    ) -> Result<(Array, Option<Qwen4ExpCache>)> {
        self.forward_pieces_in_groups(ids, cache, cuts, mark, PREFILL_GROUP_TOKENS)
    }

    /// [`Self::forward_pieces`] with a copy of the state at each of several cuts, in the order
    /// of `marks`.
    pub fn forward_pieces_marked(
        &self,
        ids: &[u32],
        cache: &mut Qwen4ExpCache,
        cuts: &[usize],
        marks: &[usize],
    ) -> Result<(Array, Vec<Qwen4ExpCache>)> {
        self.forward_pieces_grouped(ids, cache, cuts, marks, PREFILL_GROUP_TOKENS, None)
    }

    /// [`Self::forward_pieces`] with the group size spelled out (a group always holds at least
    /// one chunk).
    pub(crate) fn forward_pieces_in_groups(
        &self,
        ids: &[u32],
        cache: &mut Qwen4ExpCache,
        cuts: &[usize],
        mark: Option<usize>,
        group_tokens: usize,
    ) -> Result<(Array, Option<Qwen4ExpCache>)> {
        let (logits, mut points) =
            self.forward_pieces_grouped(ids, cache, cuts, mark.as_slice(), group_tokens, None)?;
        Ok((logits, points.pop()))
    }

    /// [`Self::forward_pieces_marked`] with the group size spelled out, and the image prompt the
    /// tokens are, if they are one (see [`Self::forward_image_prompt`]).
    fn forward_pieces_grouped(
        &self,
        ids: &[u32],
        cache: &mut Qwen4ExpCache,
        cuts: &[usize],
        marks: &[usize],
        group_tokens: usize,
        image: Option<&ImageFeed<'_>>,
    ) -> Result<(Array, Vec<Qwen4ExpCache>)> {
        self.check_feed(ids, cache, cuts, marks)?;
        let mut logits = None;
        let mut points: Vec<Option<Qwen4ExpCache>> = vec![None; marks.len()];
        let (mut group_start, mut next_cut) = (0, 0);
        while next_cut < cuts.len() {
            let mut end_cut = next_cut + 1;
            while end_cut < cuts.len() && cuts[end_cut] - group_start <= group_tokens {
                end_cut += 1;
            }
            let group_end = cuts[end_cut - 1];
            let rel_cuts: Vec<usize> = cuts[next_cut..end_cut]
                .iter()
                .map(|c| c - group_start)
                .collect();
            // The marks inside this group, as (index into `marks`, offset in the group).
            let (mark_at, rel_marks): (Vec<usize>, Vec<usize>) = marks
                .iter()
                .enumerate()
                .filter(|(_, m)| **m > group_start && **m <= group_end)
                .map(|(j, m)| (j, m - group_start))
                .unzip();
            let last = end_cut == cuts.len();
            let (group_logits, group_points) = self.forward_group(
                &ids[group_start..group_end],
                cache,
                &rel_cuts,
                &rel_marks,
                last,
                image,
            )?;
            for (j, point) in mark_at.into_iter().zip(group_points) {
                points[j] = Some(point);
            }
            if last {
                logits = group_logits;
            }
            group_start = group_end;
            next_cut = end_cut;
        }
        let logits = logits.ok_or_else(|| anyhow!("qwen4_exp forward: no logits produced"))?;
        let points = points
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| anyhow!("qwen4_exp forward: a mark fell in no group"))?;
        Ok((logits, points))
    }

    /// One layer-major group of [`Self::forward_pieces`]: the logits when `want_logits`, and the
    /// state at each of `marks` (cuts of this group), in their order.
    fn forward_group(
        &self,
        ids: &[u32],
        cache: &mut Qwen4ExpCache,
        cuts: &[usize],
        marks: &[usize],
        want_logits: bool,
        image: Option<&ImageFeed<'_>>,
    ) -> Result<(Option<Array>, Vec<Qwen4ExpCache>)> {
        let mut feeds = [GroupFeed {
            ids,
            cache,
            cuts,
            marks,
            want_logits,
            image,
        }];
        let mut out = self.forward_group_many(&mut feeds)?;
        Ok(out.pop().expect("one feed in, one result out"))
    }

    /// Feed several sequences together: each `JointFeed` exactly as [`Self::forward_pieces`]
    /// would feed it, but every chunk of every feed goes through a layer before any goes through
    /// the next, so a layer's experts are read from the SSD once for all of them. Each feed's
    /// arithmetic is its own (its chunks, in its order, on its own cache; the others only share
    /// the staged experts), so its logits and states come out bit-for-bit as from its own
    /// `forward_pieces`. Each feed has to fit one group (`PREFILL_GROUP_TOKENS`).
    ///
    /// Returns, per feed, the last position's logits and the state at its mark.
    pub(crate) fn forward_many(
        &self,
        feeds: &mut [JointFeed<'_>],
    ) -> Result<Vec<(Array, Option<Qwen4ExpCache>)>> {
        for f in feeds.iter() {
            self.check_feed(f.ids, f.cache, f.cuts, f.mark.as_slice())?;
            if f.ids.len() > PREFILL_GROUP_TOKENS {
                bail!(
                    "qwen4_exp forward: a joint feed of {} tokens is more than one group ({})",
                    f.ids.len(),
                    PREFILL_GROUP_TOKENS
                );
            }
        }
        let marks: Vec<Vec<usize>> = feeds.iter().map(|f| f.mark.into_iter().collect()).collect();
        let mut groups: Vec<GroupFeed<'_>> = feeds
            .iter_mut()
            .zip(&marks)
            .map(|(f, m)| GroupFeed {
                ids: f.ids,
                cache: &mut *f.cache,
                cuts: f.cuts,
                marks: m,
                want_logits: true,
                image: None,
            })
            .collect();
        self.forward_group_many(&mut groups)?
            .into_iter()
            .map(|(logits, mut points)| {
                let logits =
                    logits.ok_or_else(|| anyhow!("qwen4_exp forward: no logits produced"))?;
                Ok((logits, points.pop()))
            })
            .collect()
    }

    /// The checks [`Self::forward_pieces`] makes of a feed before computing any of it.
    fn check_feed(
        &self,
        ids: &[u32],
        cache: &Qwen4ExpCache,
        cuts: &[usize],
        marks: &[usize],
    ) -> Result<()> {
        if ids.is_empty() {
            bail!("qwen4_exp forward: no tokens");
        }
        if cache.layers.len() != self.layers.len() {
            bail!("qwen4_exp forward: cache has {} layers", cache.layers.len());
        }
        let mut prev = 0;
        for &c in cuts {
            if c <= prev || c > ids.len() {
                bail!(
                    "qwen4_exp forward: cuts {cuts:?} do not split {} tokens",
                    ids.len()
                );
            }
            prev = c;
        }
        if prev != ids.len() {
            bail!(
                "qwen4_exp forward: the last cut {prev} is not the end {}",
                ids.len()
            );
        }
        if let Some(m) = marks.iter().find(|m| !cuts.contains(m)) {
            bail!("qwen4_exp forward: mark {m} is not one of the cuts {cuts:?}");
        }
        Ok(())
    }

    /// One layer-major group for each of several feeds (see [`Self::forward_group`]): every
    /// chunk of every feed through layer 0, then every chunk through layer 1, and so on.
    fn forward_group_many(
        &self,
        feeds: &mut [GroupFeed<'_>],
    ) -> Result<Vec<(Option<Array>, Vec<Qwen4ExpCache>)>> {
        let keep = self.cfg.ngram_size - 1;
        let mut work = feeds
            .iter()
            .map(|f| GroupWork::new(self, f))
            .collect::<Result<Vec<_>>>()?;

        let layer_major = work.iter().map(|w| w.bounds.len()).sum::<usize>() > 1;
        let chunk_rows = work
            .iter()
            .flat_map(|w| w.bounds.iter().map(|(s, e)| e - s));
        let _bulk = (layer_major
            && stages_every_chunk(
                chunk_rows,
                self.cfg.num_experts_per_tok,
                self.cfg.num_experts,
            ))
        .then(|| self.offload.as_ref().map(BulkHint::on))
        .flatten();

        for (i, layer) in self.layers.iter().enumerate() {
            for (f, w) in feeds.iter_mut().zip(work.iter_mut()) {
                for (p, h) in w.streams.iter_mut().enumerate() {
                    let (s, e) = w.bounds[p];
                    if let Some((k, ple)) = &layer.ple {
                        let injected = self
                            .ple_forward(
                                ple,
                                h,
                                &w.history[s..keep + e],
                                e - s,
                                &mut f.cache.ple_conv[*k],
                                None,
                            )
                            .with_context(|| format!("layer {i}: n-gram injection"))?;
                        *h = mlx_rs::ops::add(&*h, &injected)?;
                    }
                    let at = match &w.mrope {
                        Some(m) => RopeAt::Mrope {
                            new: &m.positions[w.base_tokens + s..w.base_tokens + e],
                            all: &m.positions[..w.base_tokens + e],
                        },
                        None => RopeAt::Index,
                    };
                    *h = self
                        .layer_forward(layer, h, &mut f.cache.layers[i], None, at)
                        .with_context(|| format!("layer {i}"))?;
                    if layer_major {
                        // Settle the chunk before the next one starts: the staged experts the
                        // gather reads are replaced once the layer is done with, and the pool
                        // slots it reads are released at the next call's routing, which for
                        // another feed's chunk does not wait for this one.
                        h.eval()
                            .with_context(|| format!("layer {i}: evaluate chunk {p}"))?;
                    }
                    for (j, _) in w
                        .mark_pieces
                        .iter()
                        .enumerate()
                        .filter(|(_, mp)| **mp == Some(p))
                    {
                        w.snap_layers[j].push(f.cache.layers[i].clone());
                        if let Some((k, _)) = &layer.ple {
                            w.snap_ple[j][*k] = f.cache.ple_conv[*k].clone();
                        }
                    }
                }
            }
        }

        let mut out = Vec::with_capacity(feeds.len());
        for (f, w) in feeds.iter_mut().zip(work) {
            out.push(self.finish_group(f, w)?);
        }
        Ok(out)
    }

    /// The end of one feed's group: the draft head, the logits, the cache's token count and
    /// n-gram tail, and the states at the marks.
    fn finish_group(
        &self,
        f: &mut GroupFeed<'_>,
        w: GroupWork,
    ) -> Result<(Option<Array>, Vec<Qwen4ExpCache>)> {
        let keep = self.cfg.ngram_size - 1;
        // The draft head reads the final residual streams of every position: feed it the
        // group's (the state at the mark is the head's as of the marked chunk).
        let mut mtp_snapshots = vec![None; f.marks.len()];
        if let Some(st) = f.cache.mtp.as_mut() {
            for (p, h) in w.streams.iter().enumerate() {
                let (s, e) = w.bounds[p];
                self.mtp_consume(st, &f.ids[s..e], h, w.base_tokens + s == 0)
                    .context("qwen4_exp: feed the draft head")?;
                for (j, _) in w
                    .mark_pieces
                    .iter()
                    .enumerate()
                    .filter(|(_, mp)| **mp == Some(p))
                {
                    mtp_snapshots[j] = Some(st.clone());
                }
            }
        }

        let logits = if f.want_logits {
            let (s, e) = *w.bounds.last().expect("a group has a chunk");
            let last = w.streams.last().expect("a group has a chunk");
            let l = (e - s) as i32;
            let h = if l > 1 {
                last.try_index((.., (l - 1)..l, ..))?
            } else {
                last.clone()
            };
            let (mixed, _) = self.hyper_mix(&self.mixer, &h)?;
            let head = self
                .lm_head
                .as_ref()
                .ok_or_else(|| anyhow!("qwen4_exp: tied embeddings are not supported"))?;
            Some(to_f32(head.apply(&mixed)?)?)
        } else {
            None
        };

        f.cache.tokens += f.ids.len();
        f.cache.ngram_tail = w.history[w.history.len() - keep..].to_vec();
        f.cache.mrope = w.mrope.clone();
        let mut pending = f.cache.arrays();
        pending.extend(logits.iter());
        mlx_rs::transforms::eval(pending).context("qwen4_exp: evaluate the group and its state")?;

        let points = f
            .marks
            .iter()
            .zip(w.snap_layers)
            .zip(w.snap_ple)
            .zip(mtp_snapshots)
            .map(|(((&m, layers), ple_conv), mtp)| Qwen4ExpCache {
                layers,
                ple_conv,
                ngram_tail: w.history[m..keep + m].to_vec(),
                tokens: w.base_tokens + m,
                mtp,
                mrope: w.mrope.as_ref().map(|s| MropeState {
                    positions: s.positions[..w.base_tokens + m].to_vec(),
                    delta: s.delta,
                }),
            })
            .collect();
        Ok((logits, points))
    }

    /// `x` `[b, heads, s, d]` rotated at the positions of the `s` tokens fed now: the fused 1-D
    /// rope from `offset` for a text sequence, the 3-D one at each token's `[t, h, w]` for an image
    /// sequence.
    fn rotate(&self, x: &Array, offset: i32, at: RopeAt<'_>) -> Result<Array> {
        match at {
            RopeAt::Index => rope(
                x,
                self.cfg.rotary_dim() as i32,
                false,
                self.cfg.rope_theta,
                1.0,
                offset,
            ),
            RopeAt::Mrope { new, .. } => self.rotate_mrope(x, new),
        }
    }

    /// `x` `[b, heads, s, d]` rotated at the 3-D `positions` of its `s` rows.
    fn rotate_mrope(&self, x: &Array, positions: &[[i32; 3]]) -> Result<Array> {
        let (sections, interleaved) = self
            .cfg
            .mrope
            .ok_or_else(|| anyhow!("qwen4_exp: config.json has no usable mrope_section"))?;
        crate::native_rope::mrope(
            x,
            self.cfg.rotary_dim() as i32,
            positions,
            sections,
            interleaved,
            self.cfg.rope_theta,
        )
    }

    /// [`Self::embed_streams`] for the tokens `ids` of an image prompt starting at `start`: each
    /// image's placeholder rows become its features (before the streams are tiled, so all of
    /// them carry them, as the reference's `masked_scatter` then `repeat` does).
    fn embed_image_streams(&self, ids: &[u32], start: usize, img: &ImageFeed<'_>) -> Result<Array> {
        let embedded = to_f32(self.embed_untiled(ids)?)?;
        let spliced = crate::vision_splice::tower::splice_soft_rows(
            &embedded,
            start as i32,
            ids.len() as i32,
            img.runs,
            img.soft,
            self.cfg.hidden_size as i32,
        )?;
        Ok(mlx_rs::ops::tile(
            &spliced,
            &[1, 1, self.cfg.hc_count as i32],
        )?)
    }

    /// The loaded image tower, if any.
    pub(crate) fn vision(&self) -> Option<&crate::qwen36_vision::NativeQwen36VisionTower> {
        self.vision.as_ref()
    }

    /// Each image's placeholder run in `ids` and its features `[run_len, hidden]`, checked to
    /// agree in number and length.
    pub(crate) fn encode_images(
        &self,
        ids: &[u32],
        images: &[crate::qwen36_vision::PreparedImage],
    ) -> Result<(Vec<(usize, usize)>, Vec<Array>)> {
        let tower = self.vision.as_ref().ok_or_else(|| {
            anyhow!("image input requires LUMEN_VISION=1 and a checkpoint with a vision tower")
        })?;
        let image_token = self
            .cfg
            .image_token_id
            .ok_or_else(|| anyhow!("config.json has no image_token_id"))?;
        let runs = crate::vision_splice::image_token_runs(ids, image_token);
        if runs.len() != images.len() {
            bail!(
                "the prompt has {} image placeholder run(s) for {} image(s)",
                runs.len(),
                images.len()
            );
        }
        let cfg = tower.config();
        let per_patch = cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size * cfg.in_channels;
        let mut soft = Vec::with_capacity(images.len());
        for (i, (img, run)) in images.iter().zip(&runs).enumerate() {
            if img.num_tokens != run.1 {
                bail!(
                    "image {i} encodes to {} tokens but the prompt holds {} placeholders",
                    img.num_tokens,
                    run.1
                );
            }
            let n = (img.grid.0 * img.grid.1) as i32;
            let px = Array::from_slice(&img.patches, &[n, per_patch as i32]);
            let rows = tower
                .forward(&px, img.grid)
                .with_context(|| format!("vision tower, image {i}"))?;
            // Settled now, so the tower's intermediates are gone before the prefill allocates.
            rows.eval()
                .context("vision: evaluate the image's features")?;
            soft.push(rows);
        }
        Ok((runs, soft))
    }

    /// Feed an image prompt into `cache`, a fresh one: `ids` holds each image's placeholder run
    /// (`runs`, prompt-global, with its features `soft`), `positions` every token's 3-D rotary
    /// position and `delta` what a later text token's position adds to its index (see
    /// `qwen36_vision::mrope_positions` / `next_position`). Fed in chunks of `chunk`, layer-major
    /// like [`Self::forward_pieces`]; the sequence then decodes with [`Self::forward`] at
    /// positions shifted by `delta`. Returns the last position's logits.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_image_prompt(
        &self,
        ids: &[u32],
        cache: &mut Qwen4ExpCache,
        chunk: usize,
        positions: &[[i32; 3]],
        delta: i32,
        runs: &[(usize, usize)],
        soft: &[Array],
    ) -> Result<Array> {
        if cache.tokens != 0 {
            bail!("qwen4_exp: an image prompt starts a fresh sequence");
        }
        // The draft head never sees an image prompt; the sequence decodes without it.
        cache.mtp = None;
        if positions.len() != ids.len() {
            bail!(
                "qwen4_exp: {} positions for {} tokens",
                positions.len(),
                ids.len()
            );
        }
        let image = ImageFeed {
            positions,
            delta,
            runs,
            soft,
        };
        let mut cuts: Vec<usize> = (1..)
            .map(|k| k * chunk.max(1))
            .take_while(|&c| c < ids.len())
            .collect();
        cuts.push(ids.len());
        let (logits, _) = self.forward_pieces_grouped(
            ids,
            cache,
            &cuts,
            &[],
            PREFILL_GROUP_TOKENS,
            Some(&image),
        )?;
        Ok(logits)
    }

    /// `GatedResidual.__call__`: the block input and, when the layer has
    /// `block_inject_weight`, the per-stream write-back scale `[1, L, hc]`.
    fn hyper_mix(&self, hc: &HyperConnection, h: &Array) -> Result<(Array, Option<Array>)> {
        let cfg = &self.cfg;
        let (hcn, d) = (cfg.hc_count as i32, cfg.hidden_size as i32);
        let shape = h.shape().to_vec();
        let (b, s) = (shape[0], shape[1]);
        let normed = group_rms_norm(
            h,
            &hc.norm_weight,
            &self.ones_hidden,
            hcn,
            d,
            cfg.rms_norm_eps,
        )?;
        let hcn_f = scalar(hcn as f32);
        let low = mlx_rs::nn::silu(&mlx_rs::ops::divide(&hc.down.apply(&normed)?, &hcn_f)?)?;
        let w = mlx_rs::ops::sigmoid(&hc.up.apply(&low)?)?;
        let w = mlx_rs::ops::reshape(&w, &[b, s, hcn, d])?;
        let streams = mlx_rs::ops::reshape(&normed, &[b, s, hcn, d])?;
        let mixed = mlx_rs::ops::mean_axis(&mlx_rs::ops::multiply(&w, &streams)?, -2, false)?;
        let inject = match &hc.inject {
            None => None,
            Some(inj) => {
                let g = mlx_rs::ops::divide(&inj.apply(&normed)?, &hcn_f)?;
                Some(mlx_rs::ops::multiply(
                    scalar(2.0),
                    &mlx_rs::ops::sigmoid(&g)?,
                )?)
            }
        };
        Ok((mixed, inject))
    }

    /// `hyper + block_output (x) inject`, flattened back to `[1, L, hc*d]`.
    fn write_back(&self, h: &Array, x: &Array, inject: &Array) -> Result<Array> {
        let cfg = &self.cfg;
        let shape = h.shape().to_vec();
        let x4 = mlx_rs::ops::expand_dims(x, -2)?; // [1, L, 1, d]
        let g4 = mlx_rs::ops::expand_dims(inject, -1)?; // [1, L, hc, 1]
        let spread = mlx_rs::ops::multiply(&x4, &g4)?;
        let spread = mlx_rs::ops::reshape(&spread, &[shape[0], shape[1], cfg.stream_dim() as i32])?;
        Ok(mlx_rs::ops::add(h, &spread)?)
    }

    fn layer_forward(
        &self,
        layer: &Layer,
        h: &Array,
        state: &mut LayerState,
        first_row: Option<&mut Option<GdnState>>,
        at: RopeAt<'_>,
    ) -> Result<Array> {
        let front = self.layer_front(layer, h, state, first_row, at)?;
        let view = layer.moe.view(self.offload.as_ref());
        let x = moe_block_forward(&front.x, &view).context("moe block")?;
        self.layer_back(&front.h_mid, &x, &front.inject)
    }

    /// A layer's graph up to the MoE block's input: the mixer with its hyper-connection and
    /// write-back, then the MLP hyper-connection.
    fn layer_front(
        &self,
        layer: &Layer,
        h: &Array,
        state: &mut LayerState,
        first_row: Option<&mut Option<GdnState>>,
        at: RopeAt<'_>,
    ) -> Result<LayerFront> {
        let (x, inject) = self.hyper_mix(&layer.attn_hc, h)?;
        let inject = inject.ok_or_else(|| anyhow!("attn hyper-connection has no inject weight"))?;
        probe::stage(Stage::HcAttn, &[&x, &inject])?;
        let x = match (&layer.mixer, &mut *state) {
            (Mixer::Gdn(w), LayerState::Gdn(st)) => self.gdn_forward(w, &x, st, first_row)?,
            (Mixer::Qsa(w), LayerState::Qsa(st)) => self.qsa_forward(w, &x, st, at)?,
            _ => bail!("cache layer kind does not match the layer"),
        };
        if probe::sync_on() {
            let which = match layer.mixer {
                Mixer::Gdn(_) => Stage::MixerGdn,
                Mixer::Qsa(_) => Stage::MixerQsa,
            };
            let mut pending = state.arrays();
            pending.push(&x);
            probe::stage(which, &pending)?;
        }
        if x.shape()[1] > DECODE_ROWS {
            // A prefill chunk: settle this layer's new state now. Nothing downstream needs
            // it, so left lazy it would keep the layer's chunk-sized convolution input alive
            // until the end of the pass (36 layers x 42 KB per token: 6 GB at 4,096 tokens).
            // One extra sync per layer per chunk, against seconds of chunk. A decode step
            // (one row) or a verify pass (a few) holds next to nothing and skips it.
            let mut pending = state.arrays();
            pending.push(&x);
            mlx_rs::transforms::eval(pending).context("evaluate the layer's new state")?;
        }
        let h_mid = self.write_back(h, &x, &inject)?;
        probe::stage(Stage::Wb1, &[&h_mid])?;

        let (x, inject) = self.hyper_mix(&layer.mlp_hc, &h_mid)?;
        let inject = inject.ok_or_else(|| anyhow!("mlp hyper-connection has no inject weight"))?;
        probe::stage(Stage::HcMlp, &[&x, &inject])?;
        Ok(LayerFront { h_mid, x, inject })
    }

    /// The layer's output: the MoE block's result written back into the streams.
    fn layer_back(&self, h_mid: &Array, moe_out: &Array, inject: &Array) -> Result<Array> {
        let out = self.write_back(h_mid, moe_out, inject)?;
        probe::stage(Stage::Wb2, &[&out])?;
        Ok(out)
    }

    /// Whether this pass takes the pipelined decode path ([`build_ahead`]): a single-row step
    /// with the experts offloaded, and no stage-by-stage profile asking for its own syncs.
    fn builds_ahead(&self, rows: usize) -> bool {
        build_ahead::get() && rows == 1 && self.offload.is_some() && !probe::sync_on()
    }

    /// The layers of a decode step, each one's graph built while the GPU runs the one before.
    ///
    /// Per layer: submit the segment that ends in its routing, build the rest of the layer (the
    /// routed experts over its persistent slot array, the shared expert, the write-back) and the
    /// whole next layer's graph, then wait for the routing, make its experts resident and write
    /// their slots into the array. The plain path does these in the other order, so its host
    /// builds a layer's graph with the GPU idle. Same graph, same kernels, same bits.
    fn layers_ahead(
        &self,
        off: &MoeOffload,
        mut h: Array,
        rows: usize,
        history: &[u32],
        cache: &mut Qwen4ExpCache,
    ) -> Result<Array> {
        let n = self.layers.len();
        let mut front = self.front_at(0, &h, rows, history, cache)?;
        for i in 0..n {
            let layer = &self.layers[i];
            let view = layer.moe.view(Some(off));
            let (inds, scores) =
                moe_router(&front.x, &view).with_context(|| format!("layer {i}"))?;
            let k = inds.shape().last().copied().unwrap_or(1).max(1) as usize;
            let flat = mlx_rs::ops::reshape(&inds, &[-1])?;
            mlx_rs::transforms::async_eval([&flat])
                .with_context(|| format!("layer {i}: submit"))?;
            let (routed, slots) = off.routed_ahead(layer.moe.layer, inds.shape())?;
            let moe_out = moe_after_routing(&front.x, &view, &routed, &scores)
                .with_context(|| format!("layer {i}: moe block"))?;
            h = self.layer_back(&front.h_mid, &moe_out, &front.inject)?;
            let next = if i + 1 < n {
                Some(self.front_at(i + 1, &h, rows, history, cache)?)
            } else {
                None
            };
            off.route_ahead(layer.moe.layer, &flat, k, &slots)
                .with_context(|| format!("layer {i}: expert offload routing failed"))?;
            match next {
                Some(f) => front = f,
                None => break,
            }
        }
        Ok(h)
    }

    /// Layer `i`'s front half on the streams `h`, with its n-gram injection first when it has one.
    fn front_at(
        &self,
        i: usize,
        h: &Array,
        rows: usize,
        history: &[u32],
        cache: &mut Qwen4ExpCache,
    ) -> Result<LayerFront> {
        let layer = &self.layers[i];
        let injected;
        let h = match &layer.ple {
            Some((k, ple)) => {
                let delta = self
                    .ple_forward(ple, h, history, rows, &mut cache.ple_conv[*k], None)
                    .with_context(|| format!("layer {i}: n-gram injection"))?;
                injected = mlx_rs::ops::add(h, &delta)?;
                &injected
            }
            None => h,
        };
        self.layer_front(layer, h, &mut cache.layers[i], None, RopeAt::Index)
            .with_context(|| format!("layer {i}"))
    }

    /// `GatedDeltaNet.__call__`.
    /// `first_row` asks for the state as it stood after the first of two rows (a verify pass
    /// may keep only that row): the convolution window is a slice of the same input and the
    /// recurrence runs one step per kernel call, which is the fused kernel's own arithmetic.
    fn gdn_forward(
        &self,
        w: &Gdn,
        x: &Array,
        st: &mut GdnState,
        first_row: Option<&mut Option<GdnState>>,
    ) -> Result<Array> {
        let cfg = &self.cfg;
        let (b, s) = (x.shape()[0], x.shape()[1]);
        let (nk, nv) = (
            cfg.linear_num_key_heads as i32,
            cfg.linear_num_value_heads as i32,
        );
        let (dk, dv) = (
            cfg.linear_key_head_dim as i32,
            cfg.linear_value_head_dim as i32,
        );
        let kernel = cfg.linear_conv_kernel_dim as i32;
        let (key_dim, value_dim, conv_dim) = (
            cfg.key_dim() as i32,
            cfg.value_dim() as i32,
            cfg.conv_dim() as i32,
        );

        let mixed_qkv = w.in_proj_qkv.apply(x)?;
        let z = mlx_rs::ops::reshape(&w.in_proj_z.apply(x)?, &[b, s, nv, dv])?;
        let b_pre = w.in_proj_b.apply(x)?;
        let a_pre = w.in_proj_a.apply(x)?;

        let conv_state = match &st.conv {
            Some(c) => c.clone(),
            None => mlx_rs::ops::zeros_dtype(&[b, kernel - 1, conv_dim], Dtype::Float32)?,
        };
        let conv_input = mlx_rs::ops::concatenate_axis(&[&conv_state, &mixed_qkv], 1)?;
        let new_conv = owned(&conv_input.try_index((.., s..(s + kernel - 1), ..))?)?;
        let conv_out = mlx_rs::nn::silu(&mlx_rs::ops::conv1d(
            &conv_input,
            &w.conv1d_weight,
            1,
            0,
            1,
            conv_dim,
        )?)?;
        let parts = mlx_rs::ops::split_sections(&conv_out, &[key_dim, 2 * key_dim], -1)?;
        let q = mlx_rs::ops::reshape(&parts[0], &[b, s, nk, dk])?;
        let k = mlx_rs::ops::reshape(&parts[1], &[b, s, nk, dk])?;
        let v = mlx_rs::ops::reshape(&parts[2], &[b, s, nv, dv])?;
        let q = mlx_rs::ops::multiply(&l2norm(&q)?, scalar((dk as f32).powf(-0.5)))?;
        let k = l2norm(&k)?;

        let beta = mlx_rs::ops::sigmoid(&b_pre)?;
        let g = compute_g(&w.a_log, &a_pre, &w.dt_bias)?;
        let state_in = match &st.state {
            Some(s0) => s0.clone(),
            None => mlx_rs::ops::zeros_dtype(&[b, nv, dv, dk], Dtype::Float32)?,
        };
        let (out, new_state) = match first_row {
            Some(first) if s == 2 => {
                let row4 = |a: &Array, r: i32| a.try_index((.., r..(r + 1), .., ..));
                let row3 = |a: &Array, r: i32| a.try_index((.., r..(r + 1), ..));
                let (out0, state0) = gated_delta_step_kernel(
                    &row4(&q, 0)?,
                    &row4(&k, 0)?,
                    &row4(&v, 0)?,
                    &row3(&g, 0)?,
                    &row3(&beta, 0)?,
                    &state_in,
                )?;
                let (out1, state1) = gated_delta_step_kernel(
                    &row4(&q, 1)?,
                    &row4(&k, 1)?,
                    &row4(&v, 1)?,
                    &row3(&g, 1)?,
                    &row3(&beta, 1)?,
                    &state0,
                )?;
                *first = Some(GdnState {
                    conv: Some(owned(&conv_input.try_index((.., 1..kernel, ..))?)?),
                    state: Some(state0),
                });
                (mlx_rs::ops::concatenate_axis(&[&out0, &out1], 1)?, state1)
            }
            _ => gated_delta_step_kernel(&q, &k, &v, &g, &beta, &state_in)?,
        };
        st.conv = Some(new_conv);
        st.state = Some(new_state);

        // RMSNormGated with a sigmoid gate: rms_norm(out) * sigmoid(z).
        let normed = mlx_rs::fast::rms_norm(&out, &w.norm_weight, cfg.rms_norm_eps)?;
        let gated = mlx_rs::ops::multiply(&normed, &mlx_rs::ops::sigmoid(&z)?)?;
        let flat = mlx_rs::ops::reshape(&gated, &[b, s, value_dim])?;
        w.out_proj.apply(&flat)
    }

    /// `Attention.__call__` with the QSA indexer.
    fn qsa_forward(&self, w: &Qsa, x: &Array, st: &mut QsaState, at: RopeAt<'_>) -> Result<Array> {
        let cfg = &self.cfg;
        let (b, s) = (x.shape()[0], x.shape()[1]);
        let (nh, nkv, hd) = (
            cfg.num_attention_heads as i32,
            cfg.num_key_value_heads as i32,
            cfg.head_dim as i32,
        );
        let offset = st.kv.offset() as i32;

        let sparse = self.indexer_mask(&w.indexer, x, offset, &mut st.indexer_keys, at)?;

        let q_full = mlx_rs::ops::reshape(&w.q_proj.apply(x)?, &[b, s, nh, hd * 2])?;
        let halves = mlx_rs::ops::split(&q_full, 2, -1)?;
        let gate = mlx_rs::ops::reshape(&halves[1], &[b, s, nh * hd])?;
        let q = mlx_rs::fast::rms_norm(&halves[0], &w.q_norm, cfg.rms_norm_eps)?;
        let q = mlx_rs::ops::transpose_axes(&q, &[0, 2, 1, 3])?;
        let k = mlx_rs::ops::reshape(&w.k_proj.apply(x)?, &[b, s, nkv, hd])?;
        let k = mlx_rs::fast::rms_norm(&k, &w.k_norm, cfg.rms_norm_eps)?;
        let k = mlx_rs::ops::transpose_axes(&k, &[0, 2, 1, 3])?;
        let v = mlx_rs::ops::reshape(&w.v_proj.apply(x)?, &[b, s, nkv, hd])?;
        let v = mlx_rs::ops::transpose_axes(&v, &[0, 2, 1, 3])?;

        let q = self.rotate(&q, offset, at)?;
        let k = self.rotate(&k, offset, at)?;
        // `LUMEN_QWEN4_KV_BF16`: the cache holds bf16 and attention runs in it.
        let (q, k, v) = if kv_bf16::get() {
            (
                q.as_dtype(Dtype::Bfloat16)?,
                k.as_dtype(Dtype::Bfloat16)?,
                v.as_dtype(Dtype::Bfloat16)?,
            )
        } else {
            (q, k, v)
        };
        let (k_all, v_all) = st.kv.update_and_fetch(&k, &v)?;

        let scale = (hd as f32).powf(-0.5);
        let attn = match &sparse {
            // The mask is boolean and already causal.
            Some(mask) => {
                let kv_len = k_all.shape()[2] as i64;
                // Query rows attended together: the scores are `heads x rows x kv_len` f32.
                let rows = (self.scores_budget as i64 / (i64::from(nh) * kv_len * 4).max(1))
                    .max(MIN_ATTENTION_ROWS);
                sdpa_with_mask_in_blocks(&q, &k_all, &v_all, scale, mask, rows)?
            }
            None => sdpa(&q, &k_all, &v_all, scale, s > 1)?,
        };
        let attn = mlx_rs::ops::transpose_axes(&attn, &[0, 2, 1, 3])?;
        let attn = mlx_rs::ops::reshape(&attn, &[b, s, nh * hd])?;
        let gated = mlx_rs::ops::multiply(&attn, &mlx_rs::ops::sigmoid(&gate)?)?;
        w.o_proj.apply(&gated)
    }

    /// `QSAIndexer.__call__`: which keys each query may attend to once the
    /// context outgrows the budget; `None` while every visible key fits.
    fn indexer_mask(
        &self,
        w: &Indexer,
        x: &Array,
        offset: i32,
        keys: &mut Option<Array>,
        at: RopeAt<'_>,
    ) -> Result<Option<Array>> {
        let cfg = &self.cfg;
        let (b, s) = (x.shape()[0], x.shape()[1]);
        let (nh, hd) = (cfg.indexer_n_heads as i32, cfg.indexer_head_dim as i32);
        let ratio = cfg.indexer_compress_ratio as i32;
        let rot = cfg.rotary_dim() as i32;
        let eps = cfg.rms_norm_eps;

        let qk = w.index_qk_proj.apply(x)?;
        let parts = mlx_rs::ops::split_sections(&qk, &[nh * hd], -1)?;
        let q = mlx_rs::ops::reshape(&parts[0], &[b, s, nh, hd])?;
        let raw_k = mlx_rs::ops::reshape(&parts[1], &[b, s, hd])?;
        // `LUMEN_QWEN4_KV_BF16` stores the keys in bf16 too; they are pooled in f32 below.
        let bf16 = kv_bf16::get();
        let raw_k = if bf16 {
            raw_k.as_dtype(Dtype::Bfloat16)?
        } else {
            raw_k
        };
        let all_k = match keys.as_ref() {
            Some(prev) => mlx_rs::ops::concatenate_axis(&[prev, &raw_k], 1)?,
            None => owned(&raw_k)?,
        };
        *keys = Some(all_k.clone());
        let kv_len = all_k.shape()[1];
        if kv_len as usize <= cfg.indexer_budget {
            return Ok(None);
        }

        let n_blocks = kv_len / ratio;
        // Mean of each block of `ratio` raw keys, normalised, positioned at the block start.
        let blocks = all_k.try_index((.., ..(n_blocks * ratio), ..))?;
        let blocks = if bf16 {
            blocks.as_dtype(Dtype::Float32)?
        } else {
            blocks
        };
        let blocks = mlx_rs::ops::reshape(&blocks, &[b, n_blocks, ratio, hd])?;
        let pooled = mlx_rs::ops::mean_axis(&blocks, 2, false)?;
        let pooled = mlx_rs::fast::rms_norm(&pooled, &w.k_layernorm, eps)?;
        // Each block sits where its first token does: `j * ratio` for a text sequence, that
        // token's 3-D position for an image one.
        let pooled = match at {
            RopeAt::Index => rope(&pooled, rot, false, cfg.rope_theta, ratio as f32, 0)?,
            RopeAt::Mrope { all, .. } => {
                let starts: Vec<[i32; 3]> = (0..n_blocks as usize)
                    .map(|j| all[j * ratio as usize])
                    .collect();
                let x4 = mlx_rs::ops::expand_dims(&pooled, 1)?;
                mlx_rs::ops::squeeze_axes(&self.rotate_mrope(&x4, &starts)?, &[1])?
            }
        };

        let q = mlx_rs::fast::rms_norm(&q, &w.q_layernorm, eps)?;
        let q = mlx_rs::ops::transpose_axes(&q, &[0, 2, 1, 3])?; // [b, nh, s, hd]
        let q = self.rotate(&q, offset, at)?;

        // score[b, s, block] = sum over heads of relu(q . pooled) / sqrt(hd)
        let pooled_t =
            mlx_rs::ops::expand_dims(&mlx_rs::ops::transpose_axes(&pooled, &[0, 2, 1])?, 1)?;
        let dots = mlx_rs::ops::matmul(&q, &pooled_t)?; // [b, nh, s, nb]
        let relu = mlx_rs::ops::maximum(&dots, scalar(0.0))?;
        let scores = mlx_rs::ops::divide(
            &mlx_rs::ops::sum_axis(&relu, 1, false)?,
            scalar((hd as f32).sqrt()),
        )?;

        let q_pos: Vec<i32> = (offset..offset + s).collect();
        let q_pos = Array::from_slice(&q_pos, &[s]);
        let block_end: Vec<i32> = (0..n_blocks).map(|j| j * ratio + ratio - 1).collect();
        let block_end = Array::from_slice(&block_end, &[n_blocks]);
        // A block is a candidate only if it lies entirely in the query's past.
        let visible = mlx_rs::ops::le(
            &mlx_rs::ops::reshape(&block_end, &[1, 1, n_blocks])?,
            &mlx_rs::ops::reshape(&q_pos, &[1, s, 1])?,
        )?;
        let visible = mlx_rs::ops::broadcast_to(&visible, &[b, s, n_blocks])?;
        let neg_inf = scalar(f32::NEG_INFINITY);
        let scores = mlx_rs::ops::r#where(&visible, &scores, &neg_inf)?;

        let top_k = (cfg.indexer_budget as i32 / ratio).min(n_blocks);
        let order =
            mlx_rs::ops::argpartition_axis(&mlx_rs::ops::negative(&scores)?, top_k - 1, -1)?;
        let top = order.try_index((.., .., ..top_k))?.as_dtype(Dtype::Int32)?;
        let top_visible = take_along_axis(&visible, &top, -1)?;
        // A pick that is not visible (fewer past blocks than `top_k`) goes to a sentinel column.
        let top = mlx_rs::ops::r#where(&top_visible, &top, Array::from_int(n_blocks))?;
        let keep_block = mlx_rs::ops::zeros_dtype(&[b, s, n_blocks + 1], Dtype::Bool)?;
        let keep_block = put_along_axis(&keep_block, &top, Array::from_bool(true), -1)?;
        let keep_block = keep_block.try_index((.., .., ..n_blocks))?;

        let mut keep = mlx_rs::ops::repeat_axis::<bool>(keep_block, ratio, -1)?;
        let tail = kv_len - n_blocks * ratio;
        if tail > 0 {
            let pad = mlx_rs::ops::zeros_dtype(&[b, s, tail], Dtype::Bool)?;
            keep = mlx_rs::ops::concatenate_axis(&[&keep, &pad], -1)?;
        }
        // The query's own trailing partial block is always visible, nothing after the query ever is.
        let key_pos: Vec<i32> = (0..kv_len).collect();
        let key_pos =
            mlx_rs::ops::reshape(Array::from_slice(&key_pos, &[kv_len]), &[1, 1, kv_len])?;
        let qp = mlx_rs::ops::reshape(&q_pos, &[1, s, 1])?;
        let own_start = mlx_rs::ops::multiply(
            &mlx_rs::ops::floor_divide(
                &mlx_rs::ops::add(&qp, Array::from_int(1))?,
                Array::from_int(ratio),
            )?,
            Array::from_int(ratio),
        )?;
        let own_tail = mlx_rs::ops::logical_and(
            &mlx_rs::ops::ge(&key_pos, &own_start)?,
            &mlx_rs::ops::le(&key_pos, &qp)?,
        )?;
        let keep = mlx_rs::ops::logical_and(
            &mlx_rs::ops::logical_or(&keep, &own_tail)?,
            &mlx_rs::ops::le(&key_pos, &qp)?,
        )?;
        Ok(Some(mlx_rs::ops::expand_dims(&keep, 1)?))
    }

    /// `PLELayer.__call__`: what the n-gram embedding adds to the streams.
    fn ple_forward(
        &self,
        p: &Ple,
        h: &Array,
        history: &[u32],
        n_new: usize,
        conv: &mut Option<Array>,
        first_row: Option<&mut Option<Array>>,
    ) -> Result<Array> {
        let cfg = &self.cfg;
        let (hcn, d) = (cfg.hc_count as i32, cfg.hidden_size as i32);
        let s = n_new as i32;
        let eps = cfg.rms_norm_eps;

        let rows = p.hasher.row_ids(history, n_new);
        let emb = p.table.gather(&rows)?;
        let emb = mlx_rs::ops::reshape(&to_f32(emb)?, &[1, s, cfg.ple_embed_dim as i32])?;

        let key = group_rms_norm(
            &p.key_proj.apply(&emb)?,
            &p.norm_key,
            &self.ones_hidden,
            hcn,
            d,
            eps,
        )?;
        let key = mlx_rs::ops::reshape(&key, &[1, s, hcn, d])?;
        let value = p.value_proj.apply(&emb)?;
        let query = group_rms_norm(h, &p.norm_query, &self.ones_hidden, hcn, d, eps)?;
        let query = mlx_rs::ops::reshape(&query, &[1, s, hcn, d])?;

        let dots = mlx_rs::ops::sum_axis(&mlx_rs::ops::multiply(&key, &query)?, -1, true)?;
        let gate = mlx_rs::ops::divide(&dots, scalar((d as f32).sqrt()))?;
        let magnitude = mlx_rs::ops::sqrt(&mlx_rs::ops::maximum(
            &mlx_rs::ops::abs(&gate)?,
            scalar(1e-6),
        )?)?;
        let gate = mlx_rs::ops::multiply(&magnitude, &mlx_rs::ops::sign(&gate)?)?;
        let gated = mlx_rs::ops::multiply(
            &mlx_rs::ops::sigmoid(&gate)?,
            &mlx_rs::ops::expand_dims(&value, -2)?,
        )?;
        let gated = mlx_rs::ops::reshape(&gated, &[1, s, hcn * d])?;

        // Dilated causal convolution over the streams, with its history carried between calls.
        let stream = hcn * d;
        let n = (cfg.ple_conv_kernel_size as i32 - 1) * p.dilation;
        let normed = group_rms_norm(&gated, &p.norm_conv, &self.ones_hidden, hcn, d, eps)?;
        let history_rows = match conv.as_ref() {
            Some(c) => c.clone(),
            None => mlx_rs::ops::zeros_dtype(&[1, n, stream], Dtype::Float32)?,
        };
        let full = mlx_rs::ops::concatenate_axis(&[&history_rows, &normed], 1)?;
        if let (Some(first), 2) = (first_row, s) {
            *first = Some(owned(&full.try_index((.., 1..(1 + n), ..))?)?);
        }
        *conv = Some(owned(&full.try_index((.., s..(s + n), ..))?)?);
        let conv_out = mlx_rs::nn::silu(&mlx_rs::ops::conv1d(
            &full,
            &p.conv1d_weight,
            1,
            0,
            p.dilation,
            stream,
        )?)?;
        Ok(mlx_rs::ops::add(&gated, &conv_out)?)
    }
}

#[cfg(test)]
mod convention_tests {
    use super::norms_are_centered;

    #[test]
    fn the_two_shipped_conventions_are_told_apart_by_their_q_norm_mean() {
        assert!(norms_are_centered(0.2833), "HF-centered build");
        assert!(!norms_are_centered(1.2833), "folded build");
        assert!(!norms_are_centered(1.0), "random replica around one");
    }
}

#[cfg(test)]
mod staging_choice_tests {
    use super::stages_every_chunk;

    /// Flash-Next's shape (512 experts, top-10): a chunk of 205 rows takes the staging path for
    /// its size, so a prefill of several chunks, or a short feed fed beside a long one, stages
    /// each layer for all its chunks; short feeds fed together do not.
    #[test]
    fn only_a_chunk_big_enough_to_stage_stages_the_layer_for_all() {
        assert!(stages_every_chunk([1024, 1024, 88], 10, 512));
        assert!(stages_every_chunk([74, 1024, 1024, 88], 10, 512));
        assert!(stages_every_chunk([205, 10], 10, 512));
        assert!(!stages_every_chunk([74, 58], 10, 512));
        assert!(!stages_every_chunk([204, 204, 204], 10, 512));
    }
}

#[cfg(test)]
mod attention_block_tests {
    use super::*;

    /// Attending 200 query rows in blocks gives the attention of all of them at once (to the
    /// rounding of a differently shaped matmul), for a causal-ish boolean mask.
    #[test]
    fn query_blocks_are_the_same_attention_in_pieces() {
        mlx_rs::random::seed(3).unwrap();
        let (h, kvh, s, kv, d) = (4, 2, 200, 300, 32);
        let q = mlx_rs::random::normal::<f32>(&[1, h, s, d], None, None, None).unwrap();
        let k = mlx_rs::random::normal::<f32>(&[1, kvh, kv, d], None, None, None).unwrap();
        let v = mlx_rs::random::normal::<f32>(&[1, kvh, kv, d], None, None, None).unwrap();
        // Query i sees the first 100 + i keys.
        let pos = Array::from_slice(&(0..kv).collect::<Vec<i32>>(), &[1, 1, kv]);
        let lim = Array::from_slice(&(0..s).map(|i| 100 + i).collect::<Vec<i32>>(), &[1, s, 1]);
        let mask = mlx_rs::ops::expand_dims(mlx_rs::ops::lt(&pos, &lim).unwrap(), 1).unwrap();
        let whole = sdpa_with_mask_in_blocks(&q, &k, &v, 0.25, &mask, 1_000).unwrap();
        for rows in [64, 70, 199] {
            let pieces = sdpa_with_mask_in_blocks(&q, &k, &v, 0.25, &mask, rows).unwrap();
            assert_eq!(pieces.shape(), whole.shape());
            let diff = mlx_rs::ops::abs(mlx_rs::ops::subtract(&pieces, &whole).unwrap())
                .unwrap()
                .max(None)
                .unwrap();
            diff.eval().unwrap();
            assert!(
                diff.item::<f32>() < 1e-5,
                "rows {rows}: {}",
                diff.item::<f32>()
            );
        }
    }
}
