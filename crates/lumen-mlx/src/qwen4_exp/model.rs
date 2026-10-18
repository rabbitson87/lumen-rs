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

use super::config::{LayerKind, Qwen4ExpConfig};
use super::ngram::NgramHasher;
use super::ngram_table::{DEFAULT_ROW_CACHE, FileTable, NgramTable, ResidentTable, ShardArrays};
use crate::kv_disk::{ArrayRecord, LayerKindTag, LayerMeta, array_at, push_record};
use crate::native_attention::{sdpa, sdpa_with_mask};
use crate::native_cache::NativeKvCache;
use crate::native_embedding::quantized_embedding_lookup_with_mode;
use crate::native_moe::{
    AffineGateWeights, GateProj, MoeBlockWeights, OffloadLayer, SharedMlpWeights,
    SwitchExpertWeights, SwitchGluWeights, moe_block_forward,
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
    /// `ones(hidden)`, the weight of a group norm whose scale is applied afterwards.
    ones_hidden: Array,
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
}

impl Qwen4ExpCache {
    /// This state as disk records: one `LayerMeta` per layer (`Linear` for DeltaNet,
    /// holding `[conv, state]`; `Full` for attention, holding `[keys, values,
    /// indexer keys]`) and one `Linear` more per n-gram injection layer (its
    /// convolution history). The n-gram tail is not stored: it is the last tokens of
    /// the prefix, which the manifest already holds.
    pub(crate) fn to_records(&self) -> Result<(Vec<LayerMeta>, Vec<ArrayRecord>)> {
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
    pub(crate) fn restore_records(
        mut fresh: Self,
        metas: &[LayerMeta],
        records: &[ArrayRecord],
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
}

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
            },
        )
    }

    pub fn load_with(model_dir: &Path, options: LoadOptions) -> Result<Self> {
        let expert_budget_bytes = options.expert_budget_bytes;
        let mut cfg = Qwen4ExpConfig::load(model_dir)?;
        let mut weights = NativeWeights::load_dir(model_dir)?;
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
            ones_hidden,
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
        self.offload.as_ref().map(MoeOffload::report)
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
    pub(crate) fn cache_from_records(
        &self,
        metas: &[LayerMeta],
        records: &[ArrayRecord],
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
    // 4-bit rows; the group size is what the shapes say (32 in both builds).
    let bits = 4;
    let row_dim = *first_w.shape().last().unwrap_or(&0) as i64 * 32 / bits;
    let groups = *first_s.shape().last().unwrap_or(&0) as i64;
    if groups == 0 || row_dim % groups != 0 {
        bail!("{layer_base}: n-gram shard shapes do not describe 4-bit rows");
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
            bits as i32,
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
            bits as i32,
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
const PREFILL_GROUP_TOKENS: usize = 8192;

/// A pass of at most this many rows is a decode step or a speculative verify: it keeps nothing
/// alive between layers worth settling.
const DECODE_ROWS: i32 = 4;

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

        let mut h = self.embed_streams(ids)?;
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some((k, ple)) = &layer.ple {
                let injected = self
                    .ple_forward(ple, &h, &history, ids.len(), &mut cache.ple_conv[*k], None)
                    .with_context(|| format!("layer {i}: n-gram injection"))?;
                h = mlx_rs::ops::add(&h, &injected)?;
            }
            h = self
                .layer_forward(layer, &h, &mut cache.layers[i], None)
                .with_context(|| format!("layer {i}"))?;
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
        mlx_rs::transforms::eval(pending).context("qwen4_exp: evaluate the step and its state")?;
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
                .layer_forward(layer, &h, &mut cache.layers[i], Some(&mut gdn_first[i]))
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
        if let Some(m) = mark
            && !cuts.contains(&m)
        {
            bail!("qwen4_exp forward: mark {m} is not one of the cuts {cuts:?}");
        }
        let (mut logits, mut snapshot) = (None, None);
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
            let rel_mark = mark
                .filter(|m| *m > group_start && *m <= group_end)
                .map(|m| m - group_start);
            let last = end_cut == cuts.len();
            let (group_logits, group_snapshot) = self.forward_group(
                &ids[group_start..group_end],
                cache,
                &rel_cuts,
                rel_mark,
                last,
            )?;
            snapshot = group_snapshot.or(snapshot);
            if last {
                logits = group_logits;
            }
            group_start = group_end;
            next_cut = end_cut;
        }
        let logits = logits.ok_or_else(|| anyhow!("qwen4_exp forward: no logits produced"))?;
        Ok((logits, snapshot))
    }

    /// One layer-major group of [`Self::forward_pieces`].
    fn forward_group(
        &self,
        ids: &[u32],
        cache: &mut Qwen4ExpCache,
        cuts: &[usize],
        mark: Option<usize>,
        want_logits: bool,
    ) -> Result<(Option<Array>, Option<Qwen4ExpCache>)> {
        let keep = self.cfg.ngram_size - 1;
        let base_tokens = cache.tokens;
        let mut history = cache.ngram_tail.clone();
        history.extend_from_slice(ids);

        let mut start = 0;
        let bounds: Vec<(usize, usize)> = cuts
            .iter()
            .map(|&end| {
                let b = (start, end);
                start = end;
                b
            })
            .collect();
        let mut streams = bounds
            .iter()
            .map(|&(s, e)| self.embed_streams(&ids[s..e]))
            .collect::<Result<Vec<_>>>()?;

        // Several chunks through one layer: its experts are staged once and every chunk's gather
        // reads them from there, whatever share of the experts the chunk routes to.
        let layer_major = bounds.len() > 1;
        let _bulk = layer_major
            .then(|| self.offload.as_ref().map(BulkHint::on))
            .flatten();
        let mark_piece = mark.and_then(|m| cuts.iter().position(|&c| c == m));
        let mut snap_layers: Vec<LayerState> = Vec::new();
        let mut snap_ple: Vec<Option<Array>> = vec![None; cache.ple_conv.len()];

        for (i, layer) in self.layers.iter().enumerate() {
            for (p, h) in streams.iter_mut().enumerate() {
                let (s, e) = bounds[p];
                if let Some((k, ple)) = &layer.ple {
                    let injected = self
                        .ple_forward(
                            ple,
                            h,
                            &history[s..keep + e],
                            e - s,
                            &mut cache.ple_conv[*k],
                            None,
                        )
                        .with_context(|| format!("layer {i}: n-gram injection"))?;
                    *h = mlx_rs::ops::add(&*h, &injected)?;
                }
                *h = self
                    .layer_forward(layer, h, &mut cache.layers[i], None)
                    .with_context(|| format!("layer {i}"))?;
                if layer_major {
                    // Settle the chunk before the next one starts: the staged experts the
                    // gather reads are replaced once the layer is done with.
                    h.eval()
                        .with_context(|| format!("layer {i}: evaluate chunk {p}"))?;
                }
                if mark_piece == Some(p) {
                    snap_layers.push(cache.layers[i].clone());
                    if let Some((k, _)) = &layer.ple {
                        snap_ple[*k] = cache.ple_conv[*k].clone();
                    }
                }
            }
        }

        // The draft head reads the final residual streams of every position: feed it the
        // group's (the state at the mark is the head's as of the marked chunk).
        let mut mtp_snapshot = None;
        if let Some(st) = cache.mtp.as_mut() {
            for (p, h) in streams.iter().enumerate() {
                let (s, e) = bounds[p];
                self.mtp_consume(st, &ids[s..e], h, base_tokens + s == 0)
                    .context("qwen4_exp: feed the draft head")?;
                if mark_piece == Some(p) {
                    mtp_snapshot = Some(st.clone());
                }
            }
        }

        let logits = if want_logits {
            let (s, e) = *bounds.last().expect("a group has a chunk");
            let last = streams.last().expect("a group has a chunk");
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

        cache.tokens += ids.len();
        cache.ngram_tail = history[history.len() - keep..].to_vec();
        let mut pending = cache.arrays();
        pending.extend(logits.iter());
        mlx_rs::transforms::eval(pending).context("qwen4_exp: evaluate the group and its state")?;

        let snapshot = mark.map(|m| Qwen4ExpCache {
            layers: snap_layers,
            ple_conv: snap_ple,
            ngram_tail: history[m..keep + m].to_vec(),
            tokens: base_tokens + m,
            mtp: mtp_snapshot,
        });
        Ok((logits, snapshot))
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
    ) -> Result<Array> {
        let (x, inject) = self.hyper_mix(&layer.attn_hc, h)?;
        let inject = inject.ok_or_else(|| anyhow!("attn hyper-connection has no inject weight"))?;
        let x = match (&layer.mixer, &mut *state) {
            (Mixer::Gdn(w), LayerState::Gdn(st)) => self.gdn_forward(w, &x, st, first_row)?,
            (Mixer::Qsa(w), LayerState::Qsa(st)) => self.qsa_forward(w, &x, st)?,
            _ => bail!("cache layer kind does not match the layer"),
        };
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
        let h = self.write_back(h, &x, &inject)?;

        let (x, inject) = self.hyper_mix(&layer.mlp_hc, &h)?;
        let inject = inject.ok_or_else(|| anyhow!("mlp hyper-connection has no inject weight"))?;
        let view = layer.moe.view(self.offload.as_ref());
        let x = moe_block_forward(&x, &view).context("moe block")?;
        self.write_back(&h, &x, &inject)
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
    fn qsa_forward(&self, w: &Qsa, x: &Array, st: &mut QsaState) -> Result<Array> {
        let cfg = &self.cfg;
        let (b, s) = (x.shape()[0], x.shape()[1]);
        let (nh, nkv, hd) = (
            cfg.num_attention_heads as i32,
            cfg.num_key_value_heads as i32,
            cfg.head_dim as i32,
        );
        let rot = cfg.rotary_dim() as i32;
        let offset = st.kv.offset() as i32;

        let sparse = self.indexer_mask(&w.indexer, x, offset, &mut st.indexer_keys)?;

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

        let q = rope(&q, rot, false, cfg.rope_theta, 1.0, offset)?;
        let k = rope(&k, rot, false, cfg.rope_theta, 1.0, offset)?;
        let (k_all, v_all) = st.kv.update_and_fetch(&k, &v)?;

        let scale = (hd as f32).powf(-0.5);
        let attn = match &sparse {
            // The mask is boolean and already causal.
            Some(mask) => sdpa_with_mask(&q, &k_all, &v_all, scale, mask)?,
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
        let blocks = mlx_rs::ops::reshape(&blocks, &[b, n_blocks, ratio, hd])?;
        let pooled = mlx_rs::ops::mean_axis(&blocks, 2, false)?;
        let pooled = mlx_rs::fast::rms_norm(&pooled, &w.k_layernorm, eps)?;
        let pooled = rope(&pooled, rot, false, cfg.rope_theta, ratio as f32, 0)?;

        let q = mlx_rs::fast::rms_norm(&q, &w.q_layernorm, eps)?;
        let q = mlx_rs::ops::transpose_axes(&q, &[0, 2, 1, 3])?; // [b, nh, s, hd]
        let q = rope(&q, rot, false, cfg.rope_theta, 1.0, offset)?;

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
