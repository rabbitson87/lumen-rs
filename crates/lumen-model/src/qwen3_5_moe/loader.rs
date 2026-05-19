//! Shard loader for the Qwen3.5-VL-MoE MXFP4 checkpoint.
//!
//! This module reads the 4-shard safetensors layout (`model-0000{1..4}-of-00004.safetensors`)
//! plus the `model.safetensors.index.json` manifest, dequantizes each weight group into dense
//! `f32` Candle tensors, and assembles the text-model scaffold defined in [`super::model`].
//!
//! Three storage paths coexist in the checkpoint:
//!   - **MXFP4** (`bits=4, group_size=32`, E2M1 + E8M0): all projection matrices, `lm_head`,
//!     `embed_tokens`, vision tower; the dominant storage type.
//!   - **Int8-affine** (`bits=8, group_size=64`, scale + zero-point both BF16): only the
//!     router `gate` and `shared_expert_gate` projections across all 40 layers (80 groups).
//!   - **Plain BF16** (no quantization): RMS norms, SSM state (`A_log`, `dt_bias`,
//!     `conv1d.weight`), and the self-attention Q/K norms.
//!
//! The dequant kernels live here rather than in `lumen-core` or `lumen-metal` because
//! they are a weight-loading concern — not a KV-cache compression kernel and not a Metal GPU
//! shader. Keeping them local avoids a cross-crate dep shuffle for what is ~60 LOC of pure
//! scalar arithmetic.
//!
//! ## Stage 2-f-b scope (this commit)
//!   - Pure dequant functions covered by unit tests against hand-crafted bit patterns.
//!   - Shard I/O via `std::fs::read` + `safetensors::SafeTensors::deserialize` — simple and
//!     correct for first landing. A future optimization can swap to `memmap2` if the ~20 GB
//!     read pressure ever becomes a bottleneck (decoder-only startup, once per process).
//!   - Single-layer loader (`ShardSet::load_layer`) and full-text-model loader
//!     (`ShardSet::load_text_model`). The latter is memory-hungry (~76 GB f32 dequant for the
//!     full 19 GB MXFP4 payload) — callers should only invoke it on hosts with sufficient RAM.
//!     Fixture tests use `load_layer` to validate correctness without the full blow-up.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
#[cfg(feature = "turboquant-gpu")]
use std::sync::Arc;

use candle_core::{Device, Tensor};
use candle_nn::{Embedding, Linear, RmsNorm};
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensorError, SafeTensors};

#[cfg(feature = "turboquant-gpu")]
use lumen_metal::affine4_gpu::{Affine4Context, Affine4Weight};
#[cfg(feature = "turboquant-gpu")]
use lumen_metal::affine4_linear::Affine4Linear;
#[cfg(feature = "turboquant-gpu")]
use lumen_metal::mxfp4_gpu::{MxFp4Context, Mxfp4Weight};
#[cfg(feature = "turboquant-gpu")]
use lumen_metal::mxfp4_linear::{Mxfp4Linear, Mxfp4SwitchMlp};

use super::config::{LayerType, MlpKind, Qwen3_5MoeConfig};
use super::layer::{AttentionBlock, DecoderLayer};
use super::linear_attn::{GatedDeltaNet, GatedDeltaNetRuntime, conv1d_from_mlx_weight};
use super::model::Qwen3_5MoeTextModel;
use super::moe::{
    DenseMlp, MlpBlock, MoeDims, SharedExpert, SparseMoeBlock, SparseMoeRuntime, SwitchMlp,
    SwitchMlpBackend,
};
use super::proj::ProjLinear;
use super::self_attn::{SelfAttention, SelfAttnRuntime};
use super::weights::{Classification, StorageKind, WeightIndex};

// ─────────────────────────────────────────────────────────────────────────────
// Dequantization primitives
// ─────────────────────────────────────────────────────────────────────────────

/// MXFP4 group size (fixed by OCP MX spec).
pub const MXFP4_GROUP_SIZE: usize = 32;
/// Nibbles (4-bit elements) packed per u32 word in MLX storage, LSB-first.
const NIBBLES_PER_U32: usize = 8;
/// Int8-affine (`bits=8`) bytes packed per u32 word.
const BYTES_PER_U32: usize = 4;

/// E2M1 value lookup indexed by the raw 4-bit nibble `s ee m`.
///
/// Matches `lumen_metal::mxfp4::E2M1_LUT`; duplicated here so this loader stays
/// independent of the Metal crate (which pulls the full Apple `metal` dep chain).
#[rustfmt::skip]
const E2M1_LUT: [f32; 16] = [
    0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
   -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Decode an E8M0 exponent byte into its f32 scale via `2^(byte - 127)`.
///
/// `byte == 0xFF` is the reserved NaN encoding; we map it to `0.0` so a corrupt group
/// dequantizes to zero rather than polluting downstream math. `byte == 0x00` likewise
/// produces `+0.0` (see the metal-crate docs for the precision argument).
#[inline]
fn e8m0_to_f32(byte: u8) -> f32 {
    if byte == 0xFF {
        return 0.0;
    }
    // f32 exponent bias is 127 — the raw biased exponent maps directly into the f32 bit pattern.
    f32::from_bits((byte as u32) << 23)
}

/// Dequantize an MXFP4 weight group into a freshly-allocated `Vec<f32>`.
///
/// `packed` is the raw `.weight` tensor viewed as `u32` words (MLX stores the elements
/// as `u32[.., N/8]`); `scales` is the `.scales` tensor as raw `u8` E8M0 exponents. The
/// returned buffer has `scales.len() * MXFP4_GROUP_SIZE == packed.len() * NIBBLES_PER_U32`
/// elements in row-major order matching the logical shape.
fn dequant_mxfp4(packed: &[u32], scales: &[u8]) -> Result<Vec<f32>, LoadError> {
    let n = packed.len() * NIBBLES_PER_U32;
    if scales.len() * MXFP4_GROUP_SIZE != n {
        return Err(LoadError::ShapeMismatch {
            reason: "mxfp4: scales * 32 != packed * 8".into(),
        });
    }
    let mut out = vec![0f32; n];
    for (group_idx, &sb) in scales.iter().enumerate() {
        let scale = e8m0_to_f32(sb);
        let word_base = group_idx * (MXFP4_GROUP_SIZE / NIBBLES_PER_U32);
        let out_base = group_idx * MXFP4_GROUP_SIZE;
        for w_off in 0..(MXFP4_GROUP_SIZE / NIBBLES_PER_U32) {
            let w = packed[word_base + w_off];
            for i in 0..NIBBLES_PER_U32 {
                let nib = ((w >> (i * 4)) & 0xF) as usize;
                out[out_base + w_off * NIBBLES_PER_U32 + i] = E2M1_LUT[nib] * scale;
            }
        }
    }
    Ok(out)
}

/// Dequantize an int8-affine weight group into a freshly-allocated `Vec<f32>`.
///
/// MLX affine quantization (mode="affine", bits=8) stores:
///   - `packed` : `[.., N/4]` u32, 4 u8 values per word (LSB-first).
///   - `scales` : `[.., N/group_size]` BF16, one per group.
///   - `biases` : `[.., N/group_size]` BF16, zero-point per group.
///
/// Dequantization is `value = u8_value * scale + bias` per MLX's `affine_dequantize` reference.
/// `group_size` comes from `quantization_config.for_weight(prefix).group_size` (64 for all
/// shipped int8-affine overrides).
/// Test-only re-export of the private int8-affine dequantizer so external parity tests
/// can exercise it without duplicating the implementation.
#[doc(hidden)]
/// Test-only public re-export of [`dequant_int4_affine`]. Keeps the production helper
/// private while letting integration tests in `tests/*.rs` and sister crates exercise
/// the exact CPU dequant arithmetic the loader uses.
pub fn debug_dequant_int4_affine(
    packed: &[u32],
    scales_bf16: &[u16],
    biases_bf16: &[u16],
    group_size: usize,
) -> Result<Vec<f32>, LoadError> {
    dequant_int4_affine(packed, scales_bf16, biases_bf16, group_size)
}

pub fn debug_dequant_int8_affine(
    packed: &[u32],
    scales_bf16: &[u16],
    biases_bf16: &[u16],
    group_size: usize,
) -> Result<Vec<f32>, LoadError> {
    dequant_int8_affine(packed, scales_bf16, biases_bf16, group_size)
}

fn dequant_int8_affine(
    packed: &[u32],
    scales_bf16: &[u16],
    biases_bf16: &[u16],
    group_size: usize,
) -> Result<Vec<f32>, LoadError> {
    if !group_size.is_multiple_of(BYTES_PER_U32) {
        return Err(LoadError::ShapeMismatch {
            reason: format!("int8-affine: group_size {group_size} must be a multiple of 4"),
        });
    }
    let n = packed.len() * BYTES_PER_U32;
    if scales_bf16.len() != biases_bf16.len() {
        return Err(LoadError::ShapeMismatch {
            reason: "int8-affine: scales.len() != biases.len()".into(),
        });
    }
    if scales_bf16.len() * group_size != n {
        return Err(LoadError::ShapeMismatch {
            reason: format!(
                "int8-affine: scales * {group_size} != packed * 4 ({} vs {n})",
                scales_bf16.len() * group_size
            ),
        });
    }
    let mut out = vec![0f32; n];
    let words_per_group = group_size / BYTES_PER_U32;
    for (group_idx, (&sb, &bb)) in scales_bf16.iter().zip(biases_bf16).enumerate() {
        let scale = bf16_to_f32(sb);
        let bias = bf16_to_f32(bb);
        let word_base = group_idx * words_per_group;
        let out_base = group_idx * group_size;
        for w_off in 0..words_per_group {
            let w = packed[word_base + w_off];
            for i in 0..BYTES_PER_U32 {
                let v = ((w >> (i * 8)) & 0xFF) as f32;
                out[out_base + w_off * BYTES_PER_U32 + i] = v * scale + bias;
            }
        }
    }
    Ok(out)
}

/// 4-bit affine dequantisation. Eight 4-bit nibbles per `u32` word, with bf16
/// scale + bias per `group_size` elements. Used by `Qwen3.6-27B-MLX-4bit`-style
/// checkpoints whose `quantization_config.mode == "affine"` and `bits == 4`.
fn dequant_int4_affine(
    packed: &[u32],
    scales_bf16: &[u16],
    biases_bf16: &[u16],
    group_size: usize,
) -> Result<Vec<f32>, LoadError> {
    if !group_size.is_multiple_of(NIBBLES_PER_U32) {
        return Err(LoadError::ShapeMismatch {
            reason: format!("int4-affine: group_size {group_size} must be a multiple of 8"),
        });
    }
    let n = packed.len() * NIBBLES_PER_U32;
    if scales_bf16.len() != biases_bf16.len() {
        return Err(LoadError::ShapeMismatch {
            reason: "int4-affine: scales.len() != biases.len()".into(),
        });
    }
    if scales_bf16.len() * group_size != n {
        return Err(LoadError::ShapeMismatch {
            reason: format!(
                "int4-affine: scales * {group_size} != packed * 8 ({} vs {n})",
                scales_bf16.len() * group_size
            ),
        });
    }
    let mut out = vec![0f32; n];
    let words_per_group = group_size / NIBBLES_PER_U32;
    for (group_idx, (&sb, &bb)) in scales_bf16.iter().zip(biases_bf16).enumerate() {
        let scale = bf16_to_f32(sb);
        let bias = bf16_to_f32(bb);
        let word_base = group_idx * words_per_group;
        let out_base = group_idx * group_size;
        for w_off in 0..words_per_group {
            let w = packed[word_base + w_off];
            for i in 0..NIBBLES_PER_U32 {
                // Low nibble first within each byte (matches the same packing
                // convention as MXFP4's `dequant_mxfp4`).
                let v = ((w >> (i * 4)) & 0xF) as f32;
                out[out_base + w_off * NIBBLES_PER_U32 + i] = v * scale + bias;
            }
        }
    }
    Ok(out)
}

/// Widen a BF16 half into f32 by left-shifting into the upper 16 bits.
#[inline]
fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

// ─────────────────────────────────────────────────────────────────────────────
// Shard I/O
// ─────────────────────────────────────────────────────────────────────────────

/// Memory-mapped set of shard files, keyed by their filename as referenced by the index.
///
/// Shards are `memmap2::Mmap` rather than `Vec<u8>` so the ~19 GB payload lives in the kernel
/// page cache — the OS can evict cold pages under memory pressure instead of holding every
/// shard as wired RSS. This is the difference between a ~39 GB peak (owned bytes) and a ~25
/// GB peak (mmap) on a 36 GB Mac during model load.
pub struct ShardSet {
    shards: BTreeMap<String, memmap2::Mmap>,
    index: WeightIndex,
    classification: Classification,
    config: Qwen3_5MoeConfig,
    /// Optional GPU context. When `Some`, MXFP4-stored projections load as GPU-resident
    /// `Mxfp4Linear`/`Mxfp4SwitchMlp` instead of dense f32 tensors. Populated via
    /// [`Self::open_with_gpu`].
    #[cfg(feature = "turboquant-gpu")]
    gpu_ctx: Option<Arc<MxFp4Context>>,
    /// Optional 4-bit affine GPU context. When `Some`, `Int8Affine`-stored projections
    /// with `bits == 4` load as GPU-resident `Affine4Linear` instead of CPU-dequant
    /// to f32. Used by Qwen3.6-27B-MLX-4bit-style checkpoints. Independent from
    /// `gpu_ctx` — a 27B-only loader can leave `gpu_ctx = None` and only set this.
    #[cfg(feature = "turboquant-gpu")]
    affine4_ctx: Option<Arc<Affine4Context>>,
}

impl ShardSet {
    /// Open the 4-shard safetensors directory by memory-mapping every shard and cross-validate
    /// against the classifier. Returns a fully indexed `ShardSet` ready to materialize tensors.
    ///
    /// `shard_dir` must contain `model.safetensors.index.json` plus every shard listed in the
    /// index. `config` is the already-validated [`Qwen3_5MoeConfig`] — pass the exact config
    /// you parsed from `config.json`; this loader does not re-parse it.
    pub fn open(shard_dir: impl AsRef<Path>, config: Qwen3_5MoeConfig) -> Result<Self, LoadError> {
        let dir = shard_dir.as_ref().to_path_buf();
        let index_path = dir.join("model.safetensors.index.json");
        let index_str = std::fs::read_to_string(&index_path).map_err(|e| LoadError::Io {
            path: index_path.clone(),
            source: e,
        })?;
        let index = WeightIndex::from_json_str(&index_str)
            .map_err(|e| LoadError::Classify(format!("index parse: {e}")))?;
        let classification = Classification::build(&index, &config)
            .map_err(|e| LoadError::Classify(format!("{e}")))?;

        let mut shards = BTreeMap::new();
        for shard in index.shards() {
            let path = dir.join(&shard);
            let file = std::fs::File::open(&path).map_err(|e| LoadError::Io {
                path: path.clone(),
                source: e,
            })?;
            // SAFETY: the shard files are read-only and not modified for the lifetime of this
            // `ShardSet`. Standard-practice safetensors loaders (Candle, transformers) do the
            // same.
            let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| LoadError::Io {
                path: path.clone(),
                source: e,
            })?;
            shards.insert(shard, mmap);
        }
        let _ = dir;
        Ok(Self {
            shards,
            index,
            classification,
            config,
            #[cfg(feature = "turboquant-gpu")]
            gpu_ctx: None,
            #[cfg(feature = "turboquant-gpu")]
            affine4_ctx: None,
        })
    }

    /// Open a shard directory with GPU acceleration enabled. MXFP4-stored projections and
    /// routed experts stream directly into Metal buffers instead of dequantizing to f32 — the
    /// 19 GB MXFP4 payload then stays ~19 GB resident instead of blowing up to 76 GB.
    #[cfg(feature = "turboquant-gpu")]
    pub fn open_with_gpu(
        shard_dir: impl AsRef<Path>,
        config: Qwen3_5MoeConfig,
        gpu_ctx: Arc<MxFp4Context>,
    ) -> Result<Self, LoadError> {
        let mut this = Self::open(shard_dir, config)?;
        this.gpu_ctx = Some(gpu_ctx);
        Ok(this)
    }

    /// Attach a 4-bit affine GPU context for Int8Affine-stored projections with `bits=4`.
    /// Chainable with [`Self::open_with_gpu`] for hybrid checkpoints, or callable on a
    /// plain [`Self::open`] result for affine-only checkpoints (e.g. Qwen3.6-27B-MLX-4bit).
    #[cfg(feature = "turboquant-gpu")]
    pub fn with_affine4_gpu(mut self, affine4_ctx: Arc<Affine4Context>) -> Self {
        self.affine4_ctx = Some(affine4_ctx);
        self
    }

    /// Open a shard directory with a 4-bit affine GPU context only. Equivalent to
    /// `Self::open(...)?.with_affine4_gpu(ctx)`. Use when the checkpoint contains
    /// no MXFP4 weights (e.g. Qwen3.6-27B-MLX-4bit).
    #[cfg(feature = "turboquant-gpu")]
    pub fn open_with_affine4_gpu(
        shard_dir: impl AsRef<Path>,
        config: Qwen3_5MoeConfig,
        affine4_ctx: Arc<Affine4Context>,
    ) -> Result<Self, LoadError> {
        Ok(Self::open(shard_dir, config)?.with_affine4_gpu(affine4_ctx))
    }

    /// Access the underlying classification (useful for debugging / downstream validation).
    pub fn classification(&self) -> &Classification {
        &self.classification
    }

    /// Access the parsed config.
    pub fn config(&self) -> &Qwen3_5MoeConfig {
        &self.config
    }

    /// Look up a single tensor by its full safetensors name (e.g.
    /// `"language_model.model.layers.0.linear_attn.A_log"`) and return an owned `TensorView`
    /// into its shard's bytes.
    fn view<'a>(&'a self, name: &str) -> Result<TensorView<'a>, LoadError> {
        let shard_name = self
            .index
            .weight_map
            .get(name)
            .ok_or_else(|| LoadError::MissingTensor(name.to_string()))?;
        let mmap = self
            .shards
            .get(shard_name)
            .ok_or_else(|| LoadError::MissingShard(shard_name.clone()))?;
        let st = SafeTensors::deserialize(&mmap[..]).map_err(LoadError::Safetensors)?;
        st.tensor(name).map_err(LoadError::Safetensors)
    }

    /// Dequantize a weight group (identified by its classifier prefix) into a dense f32
    /// `Vec<f32>` plus its logical shape. The storage kind is looked up from the
    /// classification; the caller doesn't need to branch on it.
    ///
    /// This is the single entry point every higher-level loader uses — keeping the storage
    /// dispatch here means [`Self::load_layer`] and [`Self::load_text_model`] never touch
    /// quantization internals.
    pub fn dequant_group(&self, prefix: &str) -> Result<(Vec<f32>, Vec<usize>), LoadError> {
        let group = self
            .classification
            .groups
            .iter()
            .find(|g| g.prefix == prefix)
            .ok_or_else(|| LoadError::UnknownPrefix(prefix.to_string()))?;

        match group.storage {
            StorageKind::Plain => {
                let name = plain_tensor_name(prefix);
                let view = self.view(&name)?;
                let shape = view.shape().to_vec();
                let data = plain_to_f32(&view)?;
                Ok((data, shape))
            }
            StorageKind::Mxfp4 => {
                let w = self.view(&format!("{prefix}.weight"))?;
                let s = self.view(&format!("{prefix}.scales"))?;
                if w.dtype() != Dtype::U32 {
                    return Err(LoadError::UnexpectedDtype {
                        tensor: format!("{prefix}.weight"),
                        expected: "U32",
                        found: format!("{:?}", w.dtype()),
                    });
                }
                if s.dtype() != Dtype::U8 {
                    return Err(LoadError::UnexpectedDtype {
                        tensor: format!("{prefix}.scales"),
                        expected: "U8",
                        found: format!("{:?}", s.dtype()),
                    });
                }
                let packed = bytes_to_u32_vec(w.data());
                let data = dequant_mxfp4(&packed, s.data())?;
                let shape = unpack_last_dim(w.shape(), NIBBLES_PER_U32);
                Ok((data, shape))
            }
            StorageKind::Int8Affine => {
                let w = self.view(&format!("{prefix}.weight"))?;
                let s = self.view(&format!("{prefix}.scales"))?;
                let b = self.view(&format!("{prefix}.biases"))?;
                if w.dtype() != Dtype::U32 {
                    return Err(LoadError::UnexpectedDtype {
                        tensor: format!("{prefix}.weight"),
                        expected: "U32",
                        found: format!("{:?}", w.dtype()),
                    });
                }
                if s.dtype() != Dtype::BF16 || b.dtype() != Dtype::BF16 {
                    return Err(LoadError::UnexpectedDtype {
                        tensor: format!("{prefix}.scales|biases"),
                        expected: "BF16",
                        found: format!("scales={:?}, biases={:?}", s.dtype(), b.dtype()),
                    });
                }
                // Reaching this branch means `expected_storage` resolved to Int8Affine,
                // which only happens when a `quantization_config` block is present. The
                // unwrap is therefore an internal invariant guard — pure-bf16 checkpoints
                // never enter this arm.
                let qparams = self
                    .config
                    .quantization_config
                    .as_ref()
                    .expect("Int8Affine path requires quantization_config")
                    .for_weight(prefix);
                let group_size = qparams.group_size;
                let packed = bytes_to_u32_vec(w.data());
                let scales = bytes_to_u16_vec(s.data());
                let biases = bytes_to_u16_vec(b.data());
                // 35B-A3B-mxfp4 ships 8-bit affine for the 80 router/shared-gate
                // overrides; 27B-MLX-4bit ships uniform 4-bit affine. Both flow
                // through the same `Int8Affine` storage tag — dispatch by `bits`.
                let (data, elements_per_u32) = match qparams.bits {
                    8 => (
                        dequant_int8_affine(&packed, &scales, &biases, group_size)?,
                        BYTES_PER_U32,
                    ),
                    4 => (
                        dequant_int4_affine(&packed, &scales, &biases, group_size)?,
                        NIBBLES_PER_U32,
                    ),
                    other => {
                        return Err(LoadError::ShapeMismatch {
                            reason: format!(
                                "affine quantization with bits={other} not supported (expect 4 or 8)"
                            ),
                        });
                    }
                };
                let shape = unpack_last_dim(w.shape(), elements_per_u32);
                Ok((data, shape))
            }
        }
    }

    /// Materialize a weight group as a Candle `Tensor` on the target device. Convenience wrapper
    /// over [`Self::dequant_group`].
    pub fn tensor(&self, prefix: &str, device: &Device) -> Result<Tensor, LoadError> {
        let (data, shape) = self.dequant_group(prefix)?;
        Tensor::from_vec(data, shape, device).map_err(LoadError::Candle)
    }

    /// Load a single decoder layer by index. Useful for block-level fixture tests — a 40-layer
    /// full load is memory-hungry enough that exercising one layer at a time is the safer path
    /// until Stage 1-d's streaming MXFP4 QTensor lands.
    pub fn load_layer(&self, layer_idx: usize, device: &Device) -> Result<DecoderLayer, LoadError> {
        let n_layers = self.config.text_config.num_hidden_layers;
        if layer_idx >= n_layers {
            return Err(LoadError::LayerOutOfRange {
                idx: layer_idx,
                n_layers,
            });
        }
        let prefix = format!("language_model.model.layers.{layer_idx}");
        let input_ln = self.load_rms_norm(&format!("{prefix}.input_layernorm"), device)?;
        let post_attn_ln =
            self.load_rms_norm(&format!("{prefix}.post_attention_layernorm"), device)?;

        let attention = match self.config.text_config.layer_types[layer_idx] {
            LayerType::FullAttention => AttentionBlock::Full(self.load_self_attn(&prefix, device)?),
            LayerType::LinearAttention => {
                AttentionBlock::Linear(self.load_linear_attn(&prefix, device)?)
            }
        };
        // Dispatch by `text_config.mlp_kind()`: 35B-A3B-mxfp4 → MoE (router + 256
        // experts + shared expert); Qwen3.6-27B → standard SwiGLU dense MLP.
        let mlp: MlpBlock = match self.config.text_config.mlp_kind() {
            MlpKind::Moe => self.load_moe(&prefix, device)?.into(),
            MlpKind::Dense => self.load_dense_mlp(&prefix, device)?.into(),
        };
        Ok(DecoderLayer::new(input_ln, attention, post_attn_ln, mlp))
    }

    /// Load the full text-only model (embed + 40 decoder layers + final norm + lm_head).
    ///
    /// **Memory warning**: dequantizing the entire MXFP4 checkpoint to f32 takes ~76 GB of
    /// resident RAM. On a 64 GB Mac this will trigger heavy swapping. Use [`Self::load_layer`]
    /// for validation; reserve this method for machines that can actually hold the dense model,
    /// or wait for Stage 1-d's streaming MXFP4 Candle QTensor.
    pub fn load_text_model(&self, device: &Device) -> Result<Qwen3_5MoeTextModel, LoadError> {
        let hidden = self.config.text_config.hidden_size;
        let vocab = self.config.text_config.vocab_size;
        let embed_w = self.tensor("language_model.model.embed_tokens", device)?;
        check_shape("embed_tokens", embed_w.dims(), &[vocab, hidden])?;
        let embed = Embedding::new(embed_w, hidden);

        let n_layers = self.config.text_config.num_hidden_layers;
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            layers.push(self.load_layer(i, device)?);
        }

        let final_norm = self.load_rms_norm("language_model.model.norm", device)?;
        // route lm_head through ProjLinear so the MXFP4 weight stays
        // GPU-resident (~270MB packed) instead of being dequantized to f32 (2.03GB).
        // Saves the 7.25ms/step f32 matmul over [1, 2048] × [248320, 2048] by replacing
        // it with a single MXFP4 dispatch reading 4× less weight bandwidth.
        let lm_head = self.load_proj("language_model.lm_head", &[vocab, hidden], device)?;

        Ok(Qwen3_5MoeTextModel::new(embed, layers, final_norm, lm_head))
    }

    // ── Per-sub-block helpers ──────────────────────────────────────────────

    fn load_rms_norm(&self, prefix: &str, device: &Device) -> Result<RmsNorm, LoadError> {
        let (data, shape) = self.dequant_group(prefix)?;
        if shape.len() != 1 {
            return Err(LoadError::ShapeMismatch {
                reason: format!("rms_norm `{prefix}` expected 1-D, got {shape:?}"),
            });
        }
        let w = Tensor::from_vec(data, shape, device).map_err(LoadError::Candle)?;
        Ok(RmsNorm::new(w, self.config.text_config.rms_norm_eps as f64))
    }

    fn load_self_attn(
        &self,
        layer_prefix: &str,
        device: &Device,
    ) -> Result<SelfAttention, LoadError> {
        let runtime = SelfAttnRuntime::from_text_config(&self.config.text_config)
            .map_err(|e| LoadError::Config(format!("{e}")))?;
        let dims = runtime.dims;
        let sa = format!("{layer_prefix}.self_attn");
        // Option M2 (2026-04-25): q + k + v fused into one [q_out + 2*kv_out, hidden]
        // weight at load time. SelfAttention::forward narrows the combined matmul output
        // back into q_raw / k_raw / v_raw. Saves 2 MXFP4 dispatches per layer × 10
        // full-attention layers.
        let q_p = format!("{sa}.q_proj");
        let k_p = format!("{sa}.k_proj");
        let v_p = format!("{sa}.v_proj");
        let qkv = self.load_proj_fused_axis0(
            &[
                (q_p.as_str(), dims.q_out_dim()),
                (k_p.as_str(), dims.kv_out_dim()),
                (v_p.as_str(), dims.kv_out_dim()),
            ],
            dims.hidden_size,
            device,
        )?;
        let o = self.load_proj(
            &format!("{sa}.o_proj"),
            &[dims.hidden_size, dims.attn_value_dim()],
            device,
        )?;
        let q_norm = self.load_rms_norm(&format!("{sa}.q_norm"), device)?;
        let k_norm = self.load_rms_norm(&format!("{sa}.k_norm"), device)?;
        Ok(SelfAttention::new(runtime, qkv, o, q_norm, k_norm))
    }

    fn load_linear_attn(
        &self,
        layer_prefix: &str,
        device: &Device,
    ) -> Result<GatedDeltaNet, LoadError> {
        let runtime = GatedDeltaNetRuntime::from_text_config(&self.config.text_config)
            .map_err(|e| LoadError::Config(format!("{e}")))?;
        let d = runtime.dims;
        let la = format!("{layer_prefix}.linear_attn");
        // Option M (2026-04-25): qkv + z + b + a all read the same `x` and share `hidden`.
        // Concatenate on axis 0 into a single `[qkv_dim + v_dim + 2*Hv, hidden]` weight so
        // the forward path runs one MXFP4 dispatch per layer instead of four. Saves
        // 3 dispatches × 30 linear_attn layers = 90 dispatches/token.
        let qkv_p = format!("{la}.in_proj_qkv");
        let z_p = format!("{la}.in_proj_z");
        let b_p = format!("{la}.in_proj_b");
        let a_p = format!("{la}.in_proj_a");
        let in_proj_combined = self.load_proj_fused_axis0(
            &[
                (qkv_p.as_str(), d.qkv_dim()),
                (z_p.as_str(), d.v_dim()),
                (b_p.as_str(), d.num_v_heads),
                (a_p.as_str(), d.num_v_heads),
            ],
            d.hidden_size,
            device,
        )?;

        let (conv_data, conv_shape) = self.dequant_group(&format!("{la}.conv1d"))?;
        check_shape("conv1d", &conv_shape, &[d.qkv_dim(), d.conv_kernel, 1])?;
        let conv_w = Tensor::from_vec(conv_data, conv_shape, device).map_err(LoadError::Candle)?;
        let conv = conv1d_from_mlx_weight(conv_w, d.conv_kernel).map_err(LoadError::Candle)?;

        let (a_log_data, a_log_shape) = self.dequant_group(&format!("{la}.A_log"))?;
        check_shape("A_log", &a_log_shape, &[d.num_v_heads])?;
        let a_log = Tensor::from_vec(a_log_data, a_log_shape, device).map_err(LoadError::Candle)?;

        let (dt_bias_data, dt_bias_shape) = self.dequant_group(&format!("{la}.dt_bias"))?;
        check_shape("dt_bias", &dt_bias_shape, &[d.num_v_heads])?;
        let dt_bias =
            Tensor::from_vec(dt_bias_data, dt_bias_shape, device).map_err(LoadError::Candle)?;

        let (norm_data, norm_shape) = self.dequant_group(&format!("{la}.norm"))?;
        check_shape("linear_attn.norm", &norm_shape, &[d.head_dim])?;
        let norm_w = Tensor::from_vec(norm_data, norm_shape, device).map_err(LoadError::Candle)?;

        let out = self.load_proj(
            &format!("{la}.out_proj"),
            &[d.hidden_size, d.v_dim()],
            device,
        )?;

        Ok(GatedDeltaNet::new(
            runtime,
            in_proj_combined,
            conv,
            a_log,
            dt_bias,
            norm_w,
            out,
        ))
    }

    /// Load a Dense SwiGLU MLP (Qwen3.6-27B layout).
    ///
    /// Weight key naming differs from MoE — there is no router, no shared-expert split,
    /// and no per-expert switch_mlp. The full-rank gate/up/down projections live directly
    /// under `model.layers.{N}.mlp`:
    ///
    /// ```text
    ///   model.layers.{N}.mlp.gate_proj.weight  shape [intermediate_size, hidden_size]
    ///   model.layers.{N}.mlp.up_proj.weight    shape [intermediate_size, hidden_size]
    ///   model.layers.{N}.mlp.down_proj.weight  shape [hidden_size, intermediate_size]
    /// ```
    ///
    /// At load time we fuse `gate_proj` + `up_proj` along axis 0 into one
    /// `[2 * intermediate_size, hidden_size]` projection (mirrors the trick already used
    /// for the MoE shared expert in [`Self::load_moe`]) so the forward pass is one combined
    /// matmul + narrow + `silu(gate) * up` instead of two separate matmuls.
    fn load_dense_mlp(&self, layer_prefix: &str, device: &Device) -> Result<DenseMlp, LoadError> {
        let text = &self.config.text_config;
        let hidden = text.hidden_size;
        let intermediate = text.dense_intermediate_size();
        let mlp = format!("{layer_prefix}.mlp");

        let gate_prefix = format!("{mlp}.gate_proj");
        let up_prefix = format!("{mlp}.up_proj");
        let gate_up = self.load_proj_fused_axis0(
            &[
                (gate_prefix.as_str(), intermediate),
                (up_prefix.as_str(), intermediate),
            ],
            hidden,
            device,
        )?;
        let down = self.load_proj(&format!("{mlp}.down_proj"), &[hidden, intermediate], device)?;
        Ok(DenseMlp::new(gate_up, down, intermediate))
    }

    fn load_moe(&self, layer_prefix: &str, device: &Device) -> Result<SparseMoeBlock, LoadError> {
        let runtime = SparseMoeRuntime::from_text_config(&self.config.text_config);
        let d = runtime.dims;
        let mlp = format!("{layer_prefix}.mlp");
        // `gate` and `shared_expert_gate` are int8-affine → always dense (load_proj routes them
        // through the dequant path). The shared-expert projections are MXFP4 → GPU path when
        // a GPU context is installed.
        let gate = self.load_proj(
            &format!("{mlp}.gate"),
            &[d.num_experts, d.hidden_size],
            device,
        )?;
        let shared_gate = self.load_proj(
            &format!("{mlp}.shared_expert_gate"),
            &[1, d.hidden_size],
            device,
        )?;
        // Option J: gate_proj + up_proj fused into one [2*inter, hidden] weight at load
        // time. SharedExpert::forward narrows the combined matmul output back into the
        // gate / up halves. Saves one MXFP4 dispatch per layer × 40 layers.
        let gate_prefix = format!("{mlp}.shared_expert.gate_proj");
        let up_prefix = format!("{mlp}.shared_expert.up_proj");
        let gate_up = self.load_proj_fused_axis0(
            &[
                (gate_prefix.as_str(), d.shared_expert_intermediate_size),
                (up_prefix.as_str(), d.shared_expert_intermediate_size),
            ],
            d.hidden_size,
            device,
        )?;
        let down = self.load_proj(
            &format!("{mlp}.shared_expert.down_proj"),
            &[d.hidden_size, d.shared_expert_intermediate_size],
            device,
        )?;
        let shared = SharedExpert::new(gate_up, down, d.shared_expert_intermediate_size);
        let switch = self.load_switch_mlp_backend(
            &mlp,
            MoeDims::from_config(&self.config.text_config),
            device,
        )?;
        Ok(SparseMoeBlock::new(
            runtime,
            gate,
            shared_gate,
            shared,
            switch,
        ))
    }

    fn load_linear(
        &self,
        prefix: &str,
        expected_shape: &[usize],
        device: &Device,
    ) -> Result<Linear, LoadError> {
        let (data, shape) = self.dequant_group(prefix)?;
        check_shape(prefix, &shape, expected_shape)?;
        let w = Tensor::from_vec(data, shape, device).map_err(LoadError::Candle)?;
        Ok(Linear::new(w, None))
    }

    fn load_3d(
        &self,
        prefix: &str,
        expected_shape: &[usize],
        device: &Device,
    ) -> Result<Tensor, LoadError> {
        let (data, shape) = self.dequant_group(prefix)?;
        check_shape(prefix, &shape, expected_shape)?;
        Tensor::from_vec(data, shape, device).map_err(LoadError::Candle)
    }

    /// Storage-aware 2-D projection loader. When the weight group is MXFP4 *and* this
    /// `ShardSet` was opened with a GPU context, returns `ProjLinear::Mxfp4`. In every other
    /// case (Plain, Int8Affine, or GPU disabled) falls back to the dense f32 path via
    /// [`Self::load_linear`].
    fn load_proj(
        &self,
        prefix: &str,
        expected_shape: &[usize],
        device: &Device,
    ) -> Result<ProjLinear, LoadError> {
        #[cfg(feature = "turboquant-gpu")]
        if let Some(ctx) = self.gpu_ctx.as_ref() {
            if self.storage_kind(prefix)? == StorageKind::Mxfp4 {
                return self.load_proj_mxfp4(prefix, expected_shape, ctx, device);
            }
        }
        // 4-bit affine GPU dispatch — independent from MXFP4 path.
        // Routes Qwen3.6-27B-MLX-4bit-style projections to GPU-resident `Affine4Linear`
        // instead of dequantizing to f32 at load time. The 8-bit affine variant
        // (35B-A3B router/shared_gate, 80 weights total) still flows through the CPU
        // dequant path below.
        #[cfg(feature = "turboquant-gpu")]
        if let Some(ctx) = self.affine4_ctx.as_ref() {
            if self.storage_kind(prefix)? == StorageKind::Int8Affine {
                if let Some(qparams) = self.config.quantization_config.as_ref() {
                    let qp = qparams.for_weight(prefix);
                    if qp.bits == 4 {
                        return self.load_proj_affine4(prefix, expected_shape, ctx, device);
                    }
                }
            }
        }
        let _ = device; // silence unused in non-GPU paths
        Ok(self.load_linear(prefix, expected_shape, device)?.into())
    }

    /// Load a 4-bit affine projection directly into GPU memory. Mirrors
    /// [`Self::load_proj_mxfp4`] but uploads three buffers (packed nibbles + bf16
    /// scales + bf16 biases).
    #[cfg(feature = "turboquant-gpu")]
    fn load_proj_affine4(
        &self,
        prefix: &str,
        expected_shape: &[usize],
        ctx: &Arc<Affine4Context>,
        _device: &Device,
    ) -> Result<ProjLinear, LoadError> {
        if expected_shape.len() != 2 {
            return Err(LoadError::ShapeMismatch {
                reason: format!(
                    "load_proj_affine4 expected 2-D shape, got {expected_shape:?} for {prefix}"
                ),
            });
        }
        let (packed, scales, biases, shape) = self.affine4_raw(prefix)?;
        check_shape(prefix, &shape, expected_shape)?;
        let (out_features, in_features) = (expected_shape[0], expected_shape[1]);
        let weight = Affine4Weight::from_host(
            &ctx.ctx,
            &packed,
            &scales,
            &biases,
            out_features,
            in_features,
        )
        .map_err(|e| LoadError::Config(format!("affine4 upload `{prefix}`: {e}")))?;
        Ok(ProjLinear::from(Affine4Linear::new(
            weight,
            None,
            Arc::clone(ctx),
        )))
    }

    /// Read the raw 4-bit affine packed nibbles + bf16 scales + bf16 biases from the
    /// shard for a given weight prefix. Returns `(packed_u32, scales_u16, biases_u16,
    /// shape_unpacked)` where `shape_unpacked` has the last dim multiplied by 8 to
    /// reflect the logical element count (vs `.weight`'s u32 word count).
    #[cfg(feature = "turboquant-gpu")]
    fn affine4_raw(
        &self,
        prefix: &str,
    ) -> Result<(Vec<u32>, Vec<u16>, Vec<u16>, Vec<usize>), LoadError> {
        let w = self.view(&format!("{prefix}.weight"))?;
        let s = self.view(&format!("{prefix}.scales"))?;
        let b = self.view(&format!("{prefix}.biases"))?;
        if w.dtype() != Dtype::U32 {
            return Err(LoadError::UnexpectedDtype {
                tensor: format!("{prefix}.weight"),
                expected: "U32",
                found: format!("{:?}", w.dtype()),
            });
        }
        if s.dtype() != Dtype::BF16 || b.dtype() != Dtype::BF16 {
            return Err(LoadError::UnexpectedDtype {
                tensor: format!("{prefix}.scales|biases"),
                expected: "BF16",
                found: format!("scales={:?}, biases={:?}", s.dtype(), b.dtype()),
            });
        }
        let packed = bytes_to_u32_vec(w.data());
        let scales = bytes_to_u16_vec(s.data());
        let biases = bytes_to_u16_vec(b.data());
        let shape = unpack_last_dim(w.shape(), NIBBLES_PER_U32);
        Ok((packed, scales, biases, shape))
    }

    #[cfg(feature = "turboquant-gpu")]
    fn load_proj_mxfp4(
        &self,
        prefix: &str,
        expected_shape: &[usize],
        ctx: &Arc<MxFp4Context>,
        _device: &Device,
    ) -> Result<ProjLinear, LoadError> {
        if expected_shape.len() != 2 {
            return Err(LoadError::ShapeMismatch {
                reason: format!(
                    "load_proj_mxfp4 expected 2-D shape, got {expected_shape:?} for {prefix}"
                ),
            });
        }
        let (packed, scales, shape) = self.mxfp4_raw(prefix)?;
        check_shape(prefix, &shape, expected_shape)?;
        let (out_features, in_features) = (expected_shape[0], expected_shape[1]);
        let weight = Mxfp4Weight::from_host(&ctx.ctx, &packed, &scales, out_features, in_features)
            .map_err(|e| LoadError::Config(format!("mxfp4 upload `{prefix}`: {e}")))?;
        Ok(ProjLinear::Mxfp4(Mxfp4Linear::new(
            weight,
            None,
            Arc::clone(ctx),
        )))
    }

    /// Storage-aware fused projection loader. Concatenates N `[out_i, hidden]` weights on
    /// axis 0 into a single `[Σ out_i, hidden]` projection at load time. The forward path
    /// narrows the combined matmul output back into per-part slices, so the runtime sees
    /// exactly one MXFP4 dispatch per layer regardless of how many sub-projections share
    /// the same input.
    ///
    /// All parts must share `hidden`. Used by:
    ///   - Option J: shared_expert gate + up (N=2)
    ///   - Option M: linear_attn in_projs qkv + z + b + a (N=4)
    ///
    /// MXFP4 path concatenates the raw packed nibbles and E8M0 scales (axis-0 concat is a
    /// contiguous host memcpy in row-major). Dense fallback concatenates the dequant f32
    /// weight tensors via `Tensor::cat`.
    fn load_proj_fused_axis0(
        &self,
        parts: &[(&str, usize)], // (prefix, out_features) for each part
        hidden: usize,
        device: &Device,
    ) -> Result<ProjLinear, LoadError> {
        if parts.is_empty() {
            return Err(LoadError::ShapeMismatch {
                reason: "load_proj_fused_axis0 needs at least one part".into(),
            });
        }
        let total_out: usize = parts.iter().map(|(_, o)| *o).sum();

        #[cfg(feature = "turboquant-gpu")]
        if let Some(ctx) = self.gpu_ctx.as_ref() {
            // All-MXFP4 fast path: stitch raw packed + scales without dequant.
            let all_mxfp4 = parts.iter().all(|(p, _)| {
                self.storage_kind(p)
                    .map(|k| k == StorageKind::Mxfp4)
                    .unwrap_or(false)
            });
            if all_mxfp4 {
                let mut combined_packed = Vec::<u32>::new();
                let mut combined_scales = Vec::<u8>::new();
                for (prefix, out_e) in parts {
                    let (packed, scales, shape) = self.mxfp4_raw(prefix)?;
                    check_shape(prefix, &shape, &[*out_e, hidden])?;
                    combined_packed.extend_from_slice(&packed);
                    combined_scales.extend_from_slice(&scales);
                }
                let weight = Mxfp4Weight::from_host(
                    &ctx.ctx,
                    &combined_packed,
                    &combined_scales,
                    total_out,
                    hidden,
                )
                .map_err(|e| {
                    let names: Vec<&str> = parts.iter().map(|(p, _)| *p).collect();
                    LoadError::Config(format!("mxfp4 fused upload `{}`: {e}", names.join(" + ")))
                })?;
                return Ok(ProjLinear::Mxfp4(Mxfp4Linear::new(
                    weight,
                    None,
                    Arc::clone(ctx),
                )));
            }
        }

        // All-Affine4 fast path: same idea as MXFP4 fused upload but with three
        // buffers (packed nibbles + bf16 scales + bf16 biases). Critical for
        // 27B-MLX-4bit where every projection is 4-bit affine — without this the
        // 80-layer gate_up + qkv fused loads dequant to f32 and OOM the host.
        #[cfg(feature = "turboquant-gpu")]
        if let Some(ctx) = self.affine4_ctx.as_ref() {
            let all_affine4 = parts.iter().all(|(p, _)| {
                if self
                    .storage_kind(p)
                    .map(|k| k == StorageKind::Int8Affine)
                    .unwrap_or(false)
                {
                    self.config
                        .quantization_config
                        .as_ref()
                        .map(|q| q.for_weight(p).bits == 4)
                        .unwrap_or(false)
                } else {
                    false
                }
            });
            if all_affine4 {
                let mut combined_packed = Vec::<u32>::new();
                let mut combined_scales = Vec::<u16>::new();
                let mut combined_biases = Vec::<u16>::new();
                for (prefix, out_e) in parts {
                    let (packed, scales, biases, shape) = self.affine4_raw(prefix)?;
                    check_shape(prefix, &shape, &[*out_e, hidden])?;
                    combined_packed.extend_from_slice(&packed);
                    combined_scales.extend_from_slice(&scales);
                    combined_biases.extend_from_slice(&biases);
                }
                let weight = Affine4Weight::from_host(
                    &ctx.ctx,
                    &combined_packed,
                    &combined_scales,
                    &combined_biases,
                    total_out,
                    hidden,
                )
                .map_err(|e| {
                    let names: Vec<&str> = parts.iter().map(|(p, _)| *p).collect();
                    LoadError::Config(format!("affine4 fused upload `{}`: {e}", names.join(" + ")))
                })?;
                return Ok(ProjLinear::from(Affine4Linear::new(
                    weight,
                    None,
                    Arc::clone(ctx),
                )));
            }
        }

        // Dense fallback: dequant each part to f32 Linear, concat on axis 0.
        let mut weights = Vec::with_capacity(parts.len());
        for (prefix, out_e) in parts {
            let l = self.load_linear(prefix, &[*out_e, hidden], device)?;
            weights.push(l.weight().clone());
        }
        let refs: Vec<&Tensor> = weights.iter().collect();
        let combined = Tensor::cat(&refs, 0).map_err(LoadError::Candle)?;
        Ok(Linear::new(combined, None).into())
    }

    /// Fetch the raw packed u32 + scale u8 slices and the *logical* (unpacked) shape for
    /// an MXFP4-stored weight group. Only valid when the group's storage is MXFP4.
    #[cfg(feature = "turboquant-gpu")]
    fn mxfp4_raw(&self, prefix: &str) -> Result<(Vec<u32>, Vec<u8>, Vec<usize>), LoadError> {
        let w = self.view(&format!("{prefix}.weight"))?;
        let s = self.view(&format!("{prefix}.scales"))?;
        if w.dtype() != Dtype::U32 {
            return Err(LoadError::UnexpectedDtype {
                tensor: format!("{prefix}.weight"),
                expected: "U32",
                found: format!("{:?}", w.dtype()),
            });
        }
        if s.dtype() != Dtype::U8 {
            return Err(LoadError::UnexpectedDtype {
                tensor: format!("{prefix}.scales"),
                expected: "U8",
                found: format!("{:?}", s.dtype()),
            });
        }
        let packed = bytes_to_u32_vec(w.data());
        let scales = s.data().to_vec();
        let shape = unpack_last_dim(w.shape(), NIBBLES_PER_U32);
        Ok((packed, scales, shape))
    }

    /// Lookup the storage kind recorded by the classifier for a given group prefix.
    fn storage_kind(&self, prefix: &str) -> Result<StorageKind, LoadError> {
        self.classification
            .groups
            .iter()
            .find(|g| g.prefix == prefix)
            .map(|g| g.storage)
            .ok_or_else(|| LoadError::UnknownPrefix(prefix.to_string()))
    }

    /// Storage-aware `switch_mlp` loader. Returns `SwitchMlpBackend::Mxfp4` when GPU is
    /// active and all three projections are MXFP4; otherwise assembles the existing dense
    /// `SwitchMlp`. Shape check is deferred to the backing constructor.
    fn load_switch_mlp_backend(
        &self,
        mlp_prefix: &str,
        dims: MoeDims,
        device: &Device,
    ) -> Result<SwitchMlpBackend, LoadError> {
        let gate_prefix = format!("{mlp_prefix}.switch_mlp.gate_proj");
        let up_prefix = format!("{mlp_prefix}.switch_mlp.up_proj");
        let down_prefix = format!("{mlp_prefix}.switch_mlp.down_proj");

        #[cfg(feature = "turboquant-gpu")]
        if let Some(ctx) = self.gpu_ctx.as_ref() {
            let all_mxfp4 = self.storage_kind(&gate_prefix)? == StorageKind::Mxfp4
                && self.storage_kind(&up_prefix)? == StorageKind::Mxfp4
                && self.storage_kind(&down_prefix)? == StorageKind::Mxfp4;
            if all_mxfp4 {
                let (gp, gs, gshape) = self.mxfp4_raw(&gate_prefix)?;
                check_shape(
                    &gate_prefix,
                    &gshape,
                    &[
                        dims.num_experts,
                        dims.moe_intermediate_size,
                        dims.hidden_size,
                    ],
                )?;
                let (up, us, ushape) = self.mxfp4_raw(&up_prefix)?;
                check_shape(
                    &up_prefix,
                    &ushape,
                    &[
                        dims.num_experts,
                        dims.moe_intermediate_size,
                        dims.hidden_size,
                    ],
                )?;
                let (dp, ds, dshape) = self.mxfp4_raw(&down_prefix)?;
                check_shape(
                    &down_prefix,
                    &dshape,
                    &[
                        dims.num_experts,
                        dims.hidden_size,
                        dims.moe_intermediate_size,
                    ],
                )?;
                let backend = Mxfp4SwitchMlp::from_host(
                    Arc::clone(ctx),
                    dims.num_experts,
                    dims.hidden_size,
                    dims.moe_intermediate_size,
                    &gp,
                    &gs,
                    &up,
                    &us,
                    &dp,
                    &ds,
                )
                .map_err(|e| LoadError::Config(format!("mxfp4 switch_mlp upload: {e}")))?;
                return Ok(SwitchMlpBackend::Mxfp4(backend));
            }
        }

        // Dense fallback
        let sw = SwitchMlp::new(
            self.load_3d(
                &gate_prefix,
                &[
                    dims.num_experts,
                    dims.moe_intermediate_size,
                    dims.hidden_size,
                ],
                device,
            )?,
            self.load_3d(
                &up_prefix,
                &[
                    dims.num_experts,
                    dims.moe_intermediate_size,
                    dims.hidden_size,
                ],
                device,
            )?,
            self.load_3d(
                &down_prefix,
                &[
                    dims.num_experts,
                    dims.hidden_size,
                    dims.moe_intermediate_size,
                ],
                device,
            )?,
            dims,
        )
        .map_err(|e| LoadError::Config(format!("{e}")))?;
        Ok(SwitchMlpBackend::Dense(sw))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Small helpers
// ─────────────────────────────────────────────────────────────────────────────

/// The name of the safetensors tensor backing a plain (unquantized) weight group. Most plain
/// weights use `<prefix>.weight`, but a handful of SSM state params (`A_log`, `dt_bias`) are
/// stored with their full name as the tensor key — the classifier groups them by full name.
fn plain_tensor_name(prefix: &str) -> String {
    let tail = prefix.rsplit_once('.').map(|(_, t)| t).unwrap_or(prefix);
    if matches!(tail, "A_log" | "dt_bias") {
        prefix.to_string()
    } else {
        format!("{prefix}.weight")
    }
}

/// Widen a plain (non-quantized) tensor into f32. Supports BF16 and F32 source dtypes; anything
/// else in this checkpoint means a loader bug or an upstream format change.
fn plain_to_f32(view: &TensorView<'_>) -> Result<Vec<f32>, LoadError> {
    match view.dtype() {
        Dtype::BF16 => {
            let halves = bytes_to_u16_vec(view.data());
            Ok(halves.into_iter().map(bf16_to_f32).collect())
        }
        Dtype::F32 => Ok(bytes_to_f32_vec(view.data())),
        other => Err(LoadError::UnexpectedDtype {
            tensor: "<plain>".into(),
            expected: "BF16 or F32",
            found: format!("{other:?}"),
        }),
    }
}

/// Expand a packed last-dim into its logical size by multiplying by `factor` (8 for MXFP4,
/// 4 for int8-affine). Returns a new shape vector; leaves all leading dims alone.
fn unpack_last_dim(packed_shape: &[usize], factor: usize) -> Vec<usize> {
    let mut out = packed_shape.to_vec();
    if let Some(last) = out.last_mut() {
        *last *= factor;
    }
    out
}

fn bytes_to_u32_vec(bytes: &[u8]) -> Vec<u32> {
    debug_assert!(bytes.len() % 4 == 0);
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn bytes_to_u16_vec(bytes: &[u8]) -> Vec<u16> {
    debug_assert!(bytes.len() % 2 == 0);
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    debug_assert!(bytes.len() % 4 == 0);
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn check_shape(name: &str, got: &[usize], want: &[usize]) -> Result<(), LoadError> {
    if got != want {
        return Err(LoadError::ShapeMismatch {
            reason: format!("`{name}`: expected {want:?}, got {got:?}"),
        });
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("I/O error reading `{path}`: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("safetensors error: {0}")]
    Safetensors(#[from] SafeTensorError),

    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("classification error: {0}")]
    Classify(String),

    #[error("config error: {0}")]
    Config(String),

    #[error("tensor `{0}` is present in classification but missing from the index")]
    MissingTensor(String),

    #[error("shard `{0}` referenced by the index is missing from the open shard set")]
    MissingShard(String),

    #[error("unknown weight group prefix `{0}` — not in the classified manifest")]
    UnknownPrefix(String),

    #[error("unexpected dtype for `{tensor}`: expected {expected}, found {found}")]
    UnexpectedDtype {
        tensor: String,
        expected: &'static str,
        found: String,
    },

    #[error("shape mismatch: {reason}")]
    ShapeMismatch { reason: String },

    #[error("layer index {idx} out of range (num_hidden_layers={n_layers})")]
    LayerOutOfRange { idx: usize, n_layers: usize },
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Primitive dequant round-trips ──────────────────────────────────────

    #[test]
    fn e2m1_lut_matches_ocp_spec() {
        // Positive side of the table from the OCP MX spec.
        assert_eq!(E2M1_LUT[0b0000], 0.0);
        assert_eq!(E2M1_LUT[0b0001], 0.5);
        assert_eq!(E2M1_LUT[0b0010], 1.0);
        assert_eq!(E2M1_LUT[0b0011], 1.5);
        assert_eq!(E2M1_LUT[0b0100], 2.0);
        assert_eq!(E2M1_LUT[0b0101], 3.0);
        assert_eq!(E2M1_LUT[0b0110], 4.0);
        assert_eq!(E2M1_LUT[0b0111], 6.0);
        for i in 0..8 {
            assert_eq!(E2M1_LUT[i + 8], -E2M1_LUT[i]);
        }
    }

    #[test]
    fn e8m0_scale_matches_powers_of_two() {
        assert_eq!(e8m0_to_f32(127), 1.0);
        assert_eq!(e8m0_to_f32(128), 2.0);
        assert_eq!(e8m0_to_f32(126), 0.5);
        assert_eq!(
            e8m0_to_f32(0xFF),
            0.0,
            "NaN encoding → 0.0 per loader policy"
        );
    }

    #[test]
    fn bf16_widen_matches_bit_shift() {
        // 0x3F80 → f32 1.0 (bf16 1.0 is the upper 16 bits of f32 1.0 = 0x3F80_0000).
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0xBF80), -1.0);
        assert_eq!(bf16_to_f32(0x0000), 0.0);
        assert_eq!(bf16_to_f32(0x4000), 2.0);
    }

    #[test]
    fn dequant_mxfp4_single_group_identity() {
        // Nibble pattern 0..16 with unit scale. Dequant should equal the LUT directly.
        let mut packed = vec![0u32; MXFP4_GROUP_SIZE / NIBBLES_PER_U32];
        for i in 0..MXFP4_GROUP_SIZE {
            let word = i / NIBBLES_PER_U32;
            let slot = i % NIBBLES_PER_U32;
            let nib = (i % 16) as u32;
            packed[word] |= nib << (slot * 4);
        }
        let scales = vec![127u8]; // scale = 1.0
        let out = dequant_mxfp4(&packed, &scales).unwrap();
        assert_eq!(out.len(), MXFP4_GROUP_SIZE);
        for i in 0..MXFP4_GROUP_SIZE {
            assert_eq!(out[i], E2M1_LUT[i % 16], "index {i}");
        }
    }

    #[test]
    fn dequant_mxfp4_scales_per_group() {
        // Two groups, all nibbles = 0b0010 (LUT = 1.0). Scales 1.0 and 4.0.
        let word = 0x2222_2222u32;
        let packed = vec![word; 2 * (MXFP4_GROUP_SIZE / NIBBLES_PER_U32)];
        let scales = vec![127u8, 129u8];
        let out = dequant_mxfp4(&packed, &scales).unwrap();
        assert!(out[..MXFP4_GROUP_SIZE].iter().all(|&v| v == 1.0));
        assert!(out[MXFP4_GROUP_SIZE..].iter().all(|&v| v == 4.0));
    }

    #[test]
    fn dequant_mxfp4_length_mismatch_errors() {
        let packed = vec![0u32; 4]; // 32 nibbles → 1 group
        let scales = vec![127u8; 2]; // claims 2 groups
        let err = dequant_mxfp4(&packed, &scales).unwrap_err();
        assert!(matches!(err, LoadError::ShapeMismatch { .. }));
    }

    #[test]
    fn dequant_int8_affine_identity_scale_and_bias() {
        // group_size=8 (smallest legal multiple of 4), two groups × 8 values = 16 elements.
        // Pack values i mod 256 with scale=1.0 and bias=0 → dequant == original u8 cast to f32.
        let group_size = 8;
        let n = group_size * 2;
        let mut packed = vec![0u32; n / BYTES_PER_U32];
        for i in 0..n {
            let word = i / BYTES_PER_U32;
            let slot = i % BYTES_PER_U32;
            let v = (i as u32) & 0xFF;
            packed[word] |= v << (slot * 8);
        }
        let scales = vec![0x3F80u16; 2]; // bf16 1.0
        let biases = vec![0x0000u16; 2]; // bf16 0.0
        let out = dequant_int8_affine(&packed, &scales, &biases, group_size).unwrap();
        for i in 0..n {
            assert_eq!(out[i], i as f32, "index {i}");
        }
    }

    #[test]
    fn dequant_int8_affine_applies_bias_offset() {
        // Single group, all zero u8 values → dequant should equal the bias.
        let group_size = 8;
        let packed = vec![0u32; group_size / BYTES_PER_U32];
        let scales = vec![0x3F80u16]; // 1.0
        let biases = vec![0xBF80u16]; // -1.0
        let out = dequant_int8_affine(&packed, &scales, &biases, group_size).unwrap();
        assert!(out.iter().all(|&v| v == -1.0));
    }

    #[test]
    fn dequant_int8_affine_applies_scale() {
        // Single group, all 0xFF u8 values (255) → dequant = 255 * scale + 0.
        let group_size = 8;
        let packed = vec![0xFFFF_FFFFu32; group_size / BYTES_PER_U32];
        let scales = vec![0x3E80u16]; // bf16 0.25
        let biases = vec![0x0000u16];
        let out = dequant_int8_affine(&packed, &scales, &biases, group_size).unwrap();
        assert!(out.iter().all(|&v| (v - 255.0 * 0.25).abs() < 1e-4));
    }

    #[test]
    fn dequant_int8_affine_rejects_bad_group_size() {
        let packed = vec![0u32; 2];
        let scales = vec![0x3F80u16];
        let biases = vec![0x0000u16];
        let err = dequant_int8_affine(&packed, &scales, &biases, 7).unwrap_err();
        assert!(matches!(err, LoadError::ShapeMismatch { .. }));
    }

    #[test]
    fn dequant_int8_affine_rejects_scale_bias_length_mismatch() {
        let packed = vec![0u32; 4];
        let scales = vec![0x3F80u16; 2];
        let biases = vec![0x0000u16; 1];
        let err = dequant_int8_affine(&packed, &scales, &biases, 8).unwrap_err();
        assert!(matches!(err, LoadError::ShapeMismatch { .. }));
    }

    #[test]
    fn unpack_last_dim_multiplies_only_trailing_axis() {
        assert_eq!(unpack_last_dim(&[32, 256], NIBBLES_PER_U32), vec![32, 2048]);
        assert_eq!(
            unpack_last_dim(&[256, 2048, 64], NIBBLES_PER_U32),
            vec![256, 2048, 512]
        );
        assert_eq!(unpack_last_dim(&[1, 512], BYTES_PER_U32), vec![1, 2048]);
    }

    #[test]
    fn plain_tensor_name_handles_suffixless_ssm_params() {
        // `A_log` and `dt_bias` are grouped by their full name — no `.weight` suffix.
        assert_eq!(
            plain_tensor_name("language_model.model.layers.0.linear_attn.A_log"),
            "language_model.model.layers.0.linear_attn.A_log"
        );
        assert_eq!(
            plain_tensor_name("language_model.model.layers.0.linear_attn.dt_bias"),
            "language_model.model.layers.0.linear_attn.dt_bias"
        );
        // Everything else appends `.weight`.
        assert_eq!(
            plain_tensor_name("language_model.model.layers.3.input_layernorm"),
            "language_model.model.layers.3.input_layernorm.weight"
        );
    }

    #[test]
    fn bytes_to_u32_roundtrip() {
        let v: Vec<u32> = (0..64).collect();
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        assert_eq!(bytes_to_u32_vec(&bytes), v);
    }

    #[test]
    fn bytes_to_u16_roundtrip() {
        let v: Vec<u16> = (0..32).collect();
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        assert_eq!(bytes_to_u16_vec(&bytes), v);
    }

    // ── Real-shard integration test ────────────────────────────────────────
    //
    // Gated behind `LUMEN_QWEN35_SHARDS` so CI stays fast. Point the env var at a
    // directory containing the 4-shard safetensors + `model.safetensors.index.json`
    // and this test will open the set, load layer 0 (linear_attn), and verify it
    // assembles without errors.
    #[test]
    fn real_shards_layer0_loads_when_available() {
        let Ok(dir) = std::env::var("LUMEN_QWEN35_SHARDS") else {
            eprintln!("skipping real-shard test (set LUMEN_QWEN35_SHARDS to enable)");
            return;
        };
        let cfg_path = PathBuf::from(&dir).join("config.json");
        let cfg_str =
            std::fs::read_to_string(&cfg_path).unwrap_or_else(|e| panic!("read {cfg_path:?}: {e}"));
        let mut cfg: Qwen3_5MoeConfig = serde_json::from_str(&cfg_str).unwrap();
        cfg.validate().unwrap();

        let shards = ShardSet::open(&dir, cfg).expect("open shard set");
        let device = Device::Cpu;
        let layer = shards.load_layer(0, &device).expect("load layer 0");
        assert!(
            layer.is_linear(),
            "layer 0 is linear_attn in the 3:1 pattern"
        );
    }

    /// Env-gated: same as `real_shards_layer0_loads_when_available` but exercises
    /// **layer 3** (the first `full_attention` layer in the 3:1 hybrid pattern).
    /// Validates that `self_attn` (Q/K/V/O + Q/K norms) and the per-MLP-variant
    /// dispatch (`MlpBlock::Dense` for 27B/4B, `MlpBlock::Moe` for 35B-A3B-mxfp4)
    /// load cleanly. Critical because `load_layer(0)` only covers linear-attn.
    #[test]
    fn real_shards_layer3_full_attn_loads_when_available() {
        let Ok(dir) = std::env::var("LUMEN_QWEN35_SHARDS") else {
            eprintln!("skipping full-attn smoke (set LUMEN_QWEN35_SHARDS to enable)");
            return;
        };
        let cfg_path = PathBuf::from(&dir).join("config.json");
        let cfg_str =
            std::fs::read_to_string(&cfg_path).unwrap_or_else(|e| panic!("read {cfg_path:?}: {e}"));
        let mut cfg: Qwen3_5MoeConfig = serde_json::from_str(&cfg_str).unwrap();
        cfg.validate().unwrap();

        // Layer index 3 is full_attn for both the 35B-A3B (`full_attention_interval=4`)
        // and 27B/4B (also 3:1) patterns.
        assert!(
            matches!(cfg.text_config.layer_types[3], LayerType::FullAttention),
            "layer 3 should be full_attention in any 3:1 hybrid checkpoint"
        );
        let shards = ShardSet::open(&dir, cfg).expect("open shard set");
        let device = Device::Cpu;
        let layer = shards
            .load_layer(3, &device)
            .expect("load layer 3 (full_attn)");
        assert!(!layer.is_linear(), "layer 3 must report full_attention");
    }

    /// Env-gated: when real shards are available AND a Metal GPU is present, load layer 0
    /// via the GPU-resident MXFP4 path and confirm it succeeds without the 76 GB f32 blow-up.
    /// This is the primary correctness gate for Stage 1-d.
    #[test]
    #[cfg(feature = "turboquant-gpu")]
    fn real_shards_layer0_loads_on_gpu_when_available() {
        let Ok(dir) = std::env::var("LUMEN_QWEN35_SHARDS") else {
            eprintln!("skipping GPU real-shard test (set LUMEN_QWEN35_SHARDS to enable)");
            return;
        };
        let Ok(gpu_ctx) = MxFp4Context::new() else {
            eprintln!("skipping GPU real-shard test (no Metal device available)");
            return;
        };
        let cfg_path = PathBuf::from(&dir).join("config.json");
        let cfg_str =
            std::fs::read_to_string(&cfg_path).unwrap_or_else(|e| panic!("read {cfg_path:?}: {e}"));
        let mut cfg: Qwen3_5MoeConfig = serde_json::from_str(&cfg_str).unwrap();
        cfg.validate().unwrap();

        let ctx = Arc::new(gpu_ctx);
        let shards = ShardSet::open_with_gpu(&dir, cfg, Arc::clone(&ctx)).expect("open with GPU");
        let device = Device::Cpu;
        let layer = shards.load_layer(0, &device).expect("load layer 0 on GPU");
        assert!(layer.is_linear(), "layer 0 is linear_attn");
    }

    /// Env-gated: load the FULL 27B text model through the GPU affine4 path
    /// (80 decoder layers + embed + lm_head). This is the critical test that
    /// confirms 27B-MLX-4bit fits on a 24 GB Mac mini under the new dispatch.
    /// Without the affine4 fast path in `load_proj_fused_axis0` this would OOM
    /// at ~26 layers (~73 GB f32 fused gate_up dequant).
    ///
    /// Skipped automatically if `LUMEN_QWEN35_SHARDS` is unset, no Metal device
    /// available, or the checkpoint doesn't ship 4-bit affine quantization.
    /// Run with `--release` to keep peak memory bounded.
    #[test]
    #[cfg(feature = "turboquant-gpu")]
    fn real_27b_full_text_model_loads_on_affine4_gpu_when_available() {
        use lumen_metal::affine4_gpu::Affine4Context;

        let Ok(dir) = std::env::var("LUMEN_QWEN35_SHARDS") else {
            eprintln!("skipping full-model affine4 GPU smoke (set LUMEN_QWEN35_SHARDS to enable)");
            return;
        };
        let Ok(mxfp4_ctx) = MxFp4Context::new() else {
            eprintln!("skipping full-model affine4 GPU smoke (no Metal device)");
            return;
        };
        let Ok(affine4_ctx) = Affine4Context::new() else {
            eprintln!("skipping full-model affine4 GPU smoke (no Metal device)");
            return;
        };
        let cfg_path = PathBuf::from(&dir).join("config.json");
        let cfg_str =
            std::fs::read_to_string(&cfg_path).unwrap_or_else(|e| panic!("read {cfg_path:?}: {e}"));
        let mut cfg: Qwen3_5MoeConfig = serde_json::from_str(&cfg_str).unwrap();
        cfg.validate().unwrap();
        let n_layers = cfg.text_config.num_hidden_layers;

        let has_affine4 = cfg
            .quantization_config
            .as_ref()
            .map(|q| {
                q.for_weight("language_model.model.layers.0.self_attn.q_proj")
                    .bits
                    == 4
            })
            .unwrap_or(false);
        if !has_affine4 {
            eprintln!("skipping full-model affine4 smoke: checkpoint not 4-bit affine");
            return;
        }

        let mxfp4 = Arc::new(mxfp4_ctx);
        let affine4 = Arc::new(affine4_ctx);
        let shards = ShardSet::open_with_gpu(&dir, cfg, mxfp4)
            .expect("open with mxfp4 ctx")
            .with_affine4_gpu(affine4);
        let device = Device::Cpu;
        eprintln!("Loading full text model ({n_layers} layers) via affine4 GPU dispatch...");
        let t = std::time::Instant::now();
        let _model = shards
            .load_text_model(&device)
            .expect("full text model load via affine4 GPU");
        let elapsed_s = t.elapsed().as_secs_f64();
        eprintln!("Full 27B text model loaded in {elapsed_s:.1}s ({n_layers} layers)");
    }

    /// Env-gated: load a 27B-style affine-4-bit checkpoint layer through the GPU
    /// `Affine4Linear` path. Verifies that the new Phase 4 dispatch wires up
    /// end-to-end: shard read → `Affine4Weight::from_host` → `ProjLinear::Affine4`.
    ///
    /// Point `LUMEN_QWEN35_SHARDS` at a 4-bit-affine MLX checkpoint (e.g.
    /// `~/models/Qwen3.6-27B-4bit`). On hybrid checkpoints (mixed Mxfp4/Affine4)
    /// projections will dispatch to whichever GPU context fits — both are wired.
    #[test]
    #[cfg(feature = "turboquant-gpu")]
    fn real_shards_layer0_loads_on_affine4_gpu_when_available() {
        use lumen_metal::affine4_gpu::Affine4Context;

        let Ok(dir) = std::env::var("LUMEN_QWEN35_SHARDS") else {
            eprintln!("skipping affine4 GPU real-shard test (set LUMEN_QWEN35_SHARDS to enable)");
            return;
        };
        let Ok(mxfp4_ctx) = MxFp4Context::new() else {
            eprintln!("skipping affine4 GPU real-shard test (no Metal device)");
            return;
        };
        let Ok(affine4_ctx) = Affine4Context::new() else {
            eprintln!("skipping affine4 GPU real-shard test (no Metal device)");
            return;
        };
        let cfg_path = PathBuf::from(&dir).join("config.json");
        let cfg_str =
            std::fs::read_to_string(&cfg_path).unwrap_or_else(|e| panic!("read {cfg_path:?}: {e}"));
        let mut cfg: Qwen3_5MoeConfig = serde_json::from_str(&cfg_str).unwrap();
        cfg.validate().unwrap();

        // Skip when the checkpoint contains no 4-bit affine quantization (e.g.
        // pure-bf16 4B). The test is opt-in to the affine4 path specifically.
        let has_affine4 = cfg
            .quantization_config
            .as_ref()
            .map(|q| {
                let qp = q.for_weight("language_model.model.layers.0.self_attn.q_proj");
                qp.bits == 4
            })
            .unwrap_or(false);
        if !has_affine4 {
            eprintln!(
                "skipping affine4 GPU test: checkpoint not affine 4-bit (has_quant_cfg={})",
                cfg.quantization_config.is_some()
            );
            return;
        }

        let mxfp4 = Arc::new(mxfp4_ctx);
        let affine4 = Arc::new(affine4_ctx);
        let shards = ShardSet::open_with_gpu(&dir, cfg, mxfp4)
            .expect("open with mxfp4 ctx")
            .with_affine4_gpu(affine4);
        let device = Device::Cpu;
        let layer = shards
            .load_layer(0, &device)
            .expect("load layer 0 on affine4 GPU");
        assert!(layer.is_linear(), "layer 0 is linear_attn");
    }
}
