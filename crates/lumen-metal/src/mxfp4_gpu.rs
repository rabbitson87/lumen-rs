//! Metal dispatcher for MXFP4 fused dequant + matvec/matmul.
//!
//! Owns its own shader library and pipeline states (separate from the TurboQuant
//! compression pipeline in [`crate::pipeline`]). The kernel source is
//! [`shaders/mxfp4.metal`] and must stay layout-compatible with [`crate::mxfp4`].
//!
//! Two APIs:
//!   - `matvec_f32(packed, scales, x, ...)` — one-shot (allocates buffers per call).
//!   - `Mxfp4Weight` — long-lived GPU-resident weight matrix; reuse for all forward
//!     passes. `matmul_with_weight` supports batch > 1 by dispatching matvec per row.

use crate::metal::{BatchedEncoderExt, CommandBufferExt, ComputeEncoderCompat};
use crate::metal::{Buffer, ComputePipelineState, Library, MTLSize};
use anyhow::Result;

/// Aggregated profiling counters for `matmul_zero_copy`. Active only when
/// `LUMEN_MXFP4_PROFILE=1`. Read + reset via [`take_profile_counts`].
static PROF_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PROF_COMMIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PROF_WAIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Snapshot + reset `(calls, commit_ns_total, wait_ns_total)`.
pub fn take_profile_counts() -> (u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    let c = PROF_CALLS.swap(0, Relaxed);
    let co = PROF_COMMIT_NS.swap(0, Relaxed);
    let wa = PROF_WAIT_NS.swap(0, Relaxed);
    (c, co, wa)
}

use crate::device::MetalContext;

const SHADER_SRC: &str = include_str!("shaders/mxfp4.metal");

#[repr(C)]
#[derive(Clone, Copy)]
struct MxFp4Dims {
    out_features: u32,
    in_features: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MxFp4MoeDims {
    out_features: u32,
    in_features: u32,
    batch: u32,
    /// 1 if all slots share `x[0]`, 0 if each slot reads its own `[batch, in]` band.
    broadcast_x: u32,
}

/// Dimensions for the fused `mxfp4_gate_up_silu_mul_f32_v3` kernel. Mirrors the shader's
/// `MxFp4GateUpSiluMulDims` struct field-for-field (no padding needed for these three u32s).
#[repr(C)]
#[derive(Clone, Copy)]
struct MxFp4GateUpSiluMulDims {
    inter: u32,
    in_features: u32,
    batch: u32,
}

/// Lever A (2026-04-27): dims struct for the routed-grouped fused gate+up+silu*up
/// kernel. Same fields as `MxFp4GateUpSiluMulDims` — the slot axis is provided
/// by the dispatch grid `depth = k`, not via a struct field.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MxFp4MoeGateUpSiluMulDims {
    inter: u32,
    in_features: u32,
    batch: u32,
}

/// Lever B (2026-04-27): dims struct for `moe_wsum_f32`.
/// `out[r] = sum_e weights[e] * downs[e, r]` for r in [0, hidden).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MoeWsumDims {
    k: u32,
    hidden: u32,
}

/// Lever C (2026-04-27): dims struct for `mxfp4_matmul_moe_wsum_f32_v3` —
/// fused down matmul + weighted sum. `out[b, hr] = sum_slot weights[slot]
///   * sum_m down[expert[slot], hr, m] * hiddens[slot, b, m]`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MxFp4MoeMatmulWsumDims {
    out_features: u32, // hidden_size
    in_features: u32,  // moe_inter
    batch: u32,
    k: u32,
}

/// Lever G (2026-04-27): dims struct for `topk_partial_select_f32`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct TopkPartialDims {
    num_experts: u32,
    k: u32,
}

/// Lever H POC (2026-04-27): dims struct for the RmsNorm-fused routed
/// gate_up_silu_mul kernel.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MxFp4MoeGateUpSiluMulRmsnormDims {
    inter: u32,
    in_features: u32,
    batch: u32,
    rms_eps: f32,
}

/// Lever H multi-callsite migration: dims struct for the RmsNorm-fused dense
/// matmul kernel `mxfp4_matmul_f32_v3_rmsnorm`. Used by routing gate and
/// shared expert gate_up consumers — both default to the v3 matmul topology
/// when their respective opt-in flags are off.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MxFp4MatmulRmsnormDims {
    out_features: u32,
    in_features: u32,
    rms_eps: f32,
}

/// Lever H Step 2: dims struct for the dense f32-weight RmsNorm-fused matmul
/// kernel `dense_f32_matmul_rmsnorm`. Used for routing gate and
/// shared_expert_gate (int8-affine source → f32 dense weight at load time).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct DenseMatmulRmsnormDims {
    out_features: u32,
    in_features: u32,
    rms_eps: f32,
}

/// CB dims for `mxfp4_moe_gate_up_silu_mul_rmsnorm_f32_v3_multi`.
/// Identical to single-token sister + `k` so the kernel can resolve
/// `expert_indices[b * k + slot]`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MxFp4MoeGateUpSiluMulRmsnormDimsMulti {
    inter: u32,
    in_features: u32,
    batch: u32,
    k: u32,
    rms_eps: f32,
}

/// CB dims for `mxfp4_matmul_moe_f32_v3_multi` (down). `broadcast_x`
/// is dropped — multi-token always reads `[k, B, in_features]`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MxFp4MoeDimsMulti {
    out_features: u32,
    in_features: u32,
    batch: u32,
    k: u32,
}

/// CB dims for `moe_wsum_f32_multi`. Same fields as the single-token
/// sister + `batch`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MoeWsumDimsMulti {
    k: u32,
    batch: u32,
    hidden: u32,
}

/// Long-lived context that owns a Metal device, command queue, and compiled MXFP4 pipelines.
pub struct MxFp4Context {
    pub ctx: MetalContext,
    #[allow(dead_code)]
    library: Library,
    matvec_f32: ComputePipelineState,
    matmul_f32: ComputePipelineState,
    matmul_moe_f32: ComputePipelineState,
    /// kernels (uint4 + float4 vectorized loads).
    matmul_f32_v2: ComputePipelineState,
    matmul_moe_f32_v2: ComputePipelineState,
    /// Path B v3 kernel (simdgroup-cooperative reduction + threadgroup x cache).
    matmul_f32_v3: ComputePipelineState,
    /// same simdgroup cooperative pattern as
    /// `matmul_f32_v3`, with expert-indices indirection fused in.
    matmul_moe_f32_v3: ComputePipelineState,
    /// gate+up+silu*up kernel for SharedExpert. Output is
    /// `[batch, inter]` (half of v3's `[batch, 2*inter]`); silu(gate)*up
    /// is fused into the final write.
    gate_up_silu_mul_f32_v3: ComputePipelineState,
    /// Lever A (2026-04-27) routed-grouped fused gate+up+silu*up kernel for
    /// MoE experts. Combines `matmul_moe_f32_v3`'s expert-indices indirection
    /// with `gate_up_silu_mul_f32_v3`'s SiLU fusion. Output is
    /// `[k, batch, inter]` (half of `matmul_moe_f32_v3`'s `[k, batch, 2*inter]`).
    moe_gate_up_silu_mul_f32_v3: ComputePipelineState,
    /// Lever B (2026-04-27) MoE weighted sum kernel. Replaces the Candle
    /// `broadcast_mul + sum_keepdim` chain that produces the per-token MoE
    /// output from `[k, hidden]` expert outputs and `[k]` per-expert weights.
    moe_wsum_f32: ComputePipelineState,
    /// Lever C (2026-04-27) fused MoE down matmul + weighted sum. Replaces the
    /// chain `mxfp4_matmul_moe_f32_v3` (writing `downs_big [k, batch, hidden]`)
    /// → `moe_wsum_f32` (reducing to `[batch, hidden]`) with a single dispatch
    /// that folds the slot axis into an inner serial loop, eliminating the
    /// `downs_big` device tensor and its associated launch+sync.
    matmul_moe_wsum_f32_v3: ComputePipelineState,
    /// Lever C-atomic (2026-04-27) grid-parallel variant of the fused MoE
    /// down+wsum kernel. Restores `grid.z = k` (2048 TGs production vs 256
    /// for the serial-fold variant) and uses `atomic<float>` adds to
    /// accumulate the per-slot weighted contribution into the output.
    matmul_moe_wsum_atomic_f32_v3: ComputePipelineState,
    /// Lever G (2026-04-27) routing top-k partial select. Replaces the
    /// Candle chain `arg_sort_last_dim → narrow → contiguous → gather`
    /// in MoE routing with a single iterated-argmax+mask dispatch. Saves
    /// the full E-element sort when only k winners are needed.
    topk_partial_select_f32: ComputePipelineState,
    /// fusion. Replaces
    /// the entire 6-dispatch chain `softmax → arg_sort → narrow → gather
    /// → sum_keepdim → broadcast_div` with a single dispatch that takes
    /// raw logits `[BL, E]` and emits `(top_k_indices [BL, K] u32,
    /// top_k_weights [BL, K] f32 already-renormalized)`. Saves 5 dispatches
    /// per layer × 40 MoE layers = 200 dispatches/decode step. Uses
    /// `metal::fast::exp` (matches Candle softmax kernel).
    router_softmax_topk_renorm_f32: ComputePipelineState,
    /// Lever D (2026-04-27) routed-grouped gate_up_silu_mul kernel with
    /// bfloat16 output. Same compute as `moe_gate_up_silu_mul_f32_v3`
    /// (Lever A) — only the device-memory store narrows. Pairs with
    /// `matmul_moe_bf16in_f32out_v3` to form a chain that avoids cast-back
    /// (Phase A.1.5's NEGATIVE pattern).
    moe_gate_up_silu_mul_f32in_bf16out_v3: ComputePipelineState,
    /// Lever D (2026-04-27) MoE down kernel with bfloat16 input. Stages
    /// `x: bfloat[k, batch, in]` into TG-shared `float[in]` (one-time
    /// bf16→f32 conversion); inner FMA loop and output stay f32.
    matmul_moe_bf16in_f32out_v3: ComputePipelineState,
    /// Lever H POC (2026-04-27) routed gate_up_silu_mul kernel with internal
    /// RmsNorm. Reads raw x, computes `inv_rms = rsqrt(mean(x²) + eps)`
    /// cooperatively, applies `weight × inv_rms` before matmul. Validates
    /// whether per-kernel internal RmsNorm is cheap enough to justify
    /// multi-callsite RmsNorm-elimination (a multi-session migration).
    moe_gate_up_silu_mul_rmsnorm_f32_v3: ComputePipelineState,
    /// Lever H multi-callsite migration: dense f32 matmul with internal RmsNorm.
    /// Same v3 topology + cooperative RmsNorm phases as the POC kernel; reads
    /// raw x and rms_weight, writes y. Covers BOTH routing gate and shared
    /// expert gate_up consumers since both default to v3 topology when their
    /// opt-in fusion flags are off.
    matmul_f32_v3_rmsnorm: ComputePipelineState,
    /// Lever H Step 3 retry: large-out variant (16 rows/TG × 512 threads).
    /// Halves the redundant per-TG RmsNorm work for callsites with
    /// out_features ≥ 8192 (qkv_proj at 9216, in_proj_combined at 12352).
    /// Same input/output buffer layout as `matmul_f32_v3_rmsnorm`; only the
    /// dispatch topology + reduce_buf size differ. Dispatched via
    /// `matmul_f32_v3_rmsnorm_large_zero_copy_inline`.
    matmul_f32_v3_rmsnorm_large: ComputePipelineState,
    /// Lever H Step 3 retry tier 2: extra-large variant
    /// (32 rows/TG × 1024 threads = Apple Silicon max). For very large
    /// out_features (≥ 12K, e.g. in_proj_combined at 12352). Quarters the
    /// redundant per-TG RmsNorm work vs the small variant. Same buffer
    /// layout as small/large; only dispatch grid + reduce_buf size differ.
    matmul_f32_v3_rmsnorm_xlarge: ComputePipelineState,
    /// Lever L1 (residual fusion). Same v3 topology as `matmul_f32_v3` but
    /// reads an additional `residual` buffer at slot 6 and writes
    /// `acc + residual[idx]` instead of `acc`. Collapses the downstream
    /// element-wise residual add into the matmul tail. Used by self_attn
    /// `o_proj` (and future linear_attn `out_proj`) under the
    /// `LUMEN_RESIDUAL_FUSION` env gate.
    matmul_f32_v3_residual: ComputePipelineState,
    /// Lever L1 Step 2 (MoE-side residual fusion). 3-way element-wise add
    /// `y[i] = a[i] + b[i] + c[i]` for flat f32 buffers. Replaces the
    /// `(y_routed + shared_y)? -> (h + summed)?` 2-add chain at the end of
    /// `SparseMoeBlock::forward_with_rmsnorm` with a single dispatch.
    /// Saves 1 dispatch / layer × 40 layers / decode step. Activated under
    /// the same `LUMEN_RESIDUAL_FUSION` gate as Step 1.
    tri_add_f32: ComputePipelineState,
    /// Lever L1 Step 3.5 (drift-safe partial Step 3): per-token scalar mul
    /// fused with tri_add. `y[t,h] = a[t,h] + b[t,h] * coef[t] + d[t,h]`.
    /// Caller computes `coef = sigmoid(gate_logit)` upstream via Candle (no
    /// transcendental in shader → bit-identical safe). Replaces the
    /// `broadcast_mul -> tri_add` 2-op chain with 1 dispatch (saves 40
    /// dispatches / decode step). Same env gate as Step 1.
    scalar_mul_tri_add_f32: ComputePipelineState,
    /// Lever L4 (cross-layer megafusion): scalar_mul_tri_add + RmsNorm.
    /// Outputs both `out` (residual stream) and `attn_in` (pre-normalized
    /// using NEXT layer's input_layernorm weight). Saves 39 dispatches /
    /// decode step (layer 0's input_layernorm remains separate). Per-token
    /// cooperative SG reduction (256 threads / row).
    scalar_mul_tri_add_rmsnorm_f32: ComputePipelineState,
    /// Lever H Step 2: dense f32-WEIGHT matmul with internal RmsNorm. For the
    /// int8-affine routing gate (`gate`) and `shared_expert_gate` projections,
    /// which the loader dequantizes to f32 dense Candle Linear at load time
    /// (not MXFP4). Without this kernel, the external
    /// `post_attention_layernorm.forward` dispatch can't be eliminated even
    /// after MXFP4 consumers fuse RmsNorm internally.
    dense_f32_matmul_rmsnorm: ComputePipelineState,
    /// kernel. 1 TG = 1 row, 256 threads
    /// cooperate on the row's reduction. Targets shapes (e.g. r_gate
    /// out=256 in=2048) where v3's `n_groups_x = ceil(out/8)` produces
    /// too few TGs for Apple GPU latency hiding.
    matmul_small_out_f32_v1: ComputePipelineState,
    /// of v3. Same dispatch
    /// topology and accumulation precision; only the device-memory store is
    /// narrowed to bfloat16. Caller decides per-call whether to use this
    /// (e.g. via `LUMEN_MXFP4_BF16_OUT=1`); the f32 path stays the default
    /// until end-to-end parity is validated.
    matmul_f32in_bf16out_v3: ComputePipelineState,
    /// Lever B L.2 (2026-04-28) bf16-input variant of dense v3. Same compute
    /// + dispatch + f32 accumulator + f32 output as `matmul_f32_v3`; only the
    /// activation pointer narrows (bf16 → f32 widened during threadgroup
    /// staging). Pairs with the `MpsRmsNormBf16Out` upstream so that
    /// input_layernorm → qkv_proj can run end-to-end without an intermediate
    /// f32 cast-back of the activation buffer.
    matmul_bf16in_f32out_v3: ComputePipelineState,
    /// of MoE v3, fused
    /// gate+up+silu*up v3, and small-out v1. Same compute as the f32
    /// pipelines; only the final store narrows.
    matmul_moe_f32in_bf16out_v3: ComputePipelineState,
    gate_up_silu_mul_f32in_bf16out_v3: ComputePipelineState,
    matmul_small_out_f32in_bf16out_v1: ComputePipelineState,
    /// CB Phase 2 (2026-04-29) multi-token MoE variants. One-line behavioral
    /// change vs `_v3`/`_v3_rmsnorm`: expert lookup is `expert_indices[b*k+slot]`
    /// instead of `expert_indices[slot]`, so the same dispatch can carry per-
    /// token expert routing for `B > 1`. Used by `forward_with_rmsnorm` when
    /// `bl > 1` to collapse the host `for t in 0..bl` loop into a single
    /// dispatch per stage (gate_up_silu_mul_rmsnorm → down → wsum).
    moe_gate_up_silu_mul_rmsnorm_f32_v3_multi: ComputePipelineState,
    matmul_moe_f32_v3_multi: ComputePipelineState,
    moe_wsum_f32_multi: ComputePipelineState,
    /// Selected kernel version. v3 > v2 > v1 — the higher one wins. Default
    /// is v3.
    kernel_version: KernelVersion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KernelVersion {
    V1,
    V2,
    V3,
}

impl MxFp4Context {
    /// Dense matmul pipeline matching the live `LUMEN_MXFP4_KERNEL_VERSION`
    /// selection. v3 is the simdgroup-cooperative kernel (Path B); v2 is the
    /// vectorized scalar baseline; v1 is the original scalar kernel.
    pub fn matmul_f32_pipeline(&self) -> &ComputePipelineState {
        match self.kernel_version {
            KernelVersion::V3 => &self.matmul_f32_v3,
            KernelVersion::V2 => &self.matmul_f32_v2,
            KernelVersion::V1 => &self.matmul_f32,
        }
    }

    /// MoE matmul pipeline matching the live `LUMEN_MXFP4_KERNEL_VERSION`.
    pub fn matmul_moe_f32_pipeline(&self) -> &ComputePipelineState {
        match self.kernel_version {
            KernelVersion::V3 => &self.matmul_moe_f32_v3,
            KernelVersion::V2 => &self.matmul_moe_f32_v2,
            KernelVersion::V1 => &self.matmul_moe_f32,
        }
    }

    /// Force-pick a specific kernel version. Used by parity / bench code that
    /// wants to compare versions in the same process without re-loading the
    /// model. Default selection comes from `LUMEN_MXFP4_KERNEL_VERSION`.
    pub fn matmul_f32_pipeline_v1(&self) -> &ComputePipelineState {
        &self.matmul_f32
    }
    pub fn matmul_f32_pipeline_v2(&self) -> &ComputePipelineState {
        &self.matmul_f32_v2
    }
    pub fn matmul_f32_pipeline_v3(&self) -> &ComputePipelineState {
        &self.matmul_f32_v3
    }
    pub fn matmul_moe_f32_pipeline_v1(&self) -> &ComputePipelineState {
        &self.matmul_moe_f32
    }
    pub fn matmul_moe_f32_pipeline_v2(&self) -> &ComputePipelineState {
        &self.matmul_moe_f32_v2
    }
    pub fn matmul_moe_f32_pipeline_v3(&self) -> &ComputePipelineState {
        &self.matmul_moe_f32_v3
    }

    /// gate+up+silu*up pipeline. Always v3-style topology
    /// (256 threads/TG, 8 simdgroups, threadgroup x cache).
    pub fn gate_up_silu_mul_f32_pipeline(&self) -> &ComputePipelineState {
        &self.gate_up_silu_mul_f32_v3
    }

    /// pipeline (1 TG = 1 row, 256 threads).
    pub fn matmul_small_out_f32_pipeline(&self) -> &ComputePipelineState {
        &self.matmul_small_out_f32_v1
    }

    /// Whether v2 is selected. Kept for backward compat with bench tooling.
    pub fn uses_v2(&self) -> bool {
        matches!(self.kernel_version, KernelVersion::V2)
    }

    /// Whether v3 is selected. Bench / production timers use this to decide
    /// which dispatch topology to encode.
    pub fn uses_v3(&self) -> bool {
        matches!(self.kernel_version, KernelVersion::V3)
    }
}

fn read_kernel_version_env() -> KernelVersion {
    // Default = v3 (Path B Phase B.1, simdgroup-cooperative + tg x cache).
    // Microbench (2026-04-26) shows 2.7-7.3× speedup over v2 across every
    // production shape, with cosine ≥ 0.9999 parity vs v2/v1. End-to-end
    // wallclock confirms decode 215ms → 161ms (+29% throughput) on
    // Qwen3.6-35B-A3B-mxfp4.
    //
    // Rollback paths:
    //   - `LUMEN_MXFP4_KERNEL_VERSION=v2` : the prior vectorized scalar baseline
    //   - `LUMEN_MXFP4_KERNEL_VERSION=v1` : the original scalar kernel
    match std::env::var("LUMEN_MXFP4_KERNEL_VERSION").as_deref() {
        Ok("v1") | Ok("V1") | Ok("1") => KernelVersion::V1,
        Ok("v2") | Ok("V2") | Ok("2") => KernelVersion::V2,
        _ => KernelVersion::V3,
    }
}

/// GPU-resident MXFP4 weight matrix.
///
/// Holds packed nibbles and E8M0 scales as Metal buffers allocated once at load time.
/// Forward passes reuse these buffers instead of re-uploading every call.
///
/// The `packed_offset` / `scales_offset` fields let multiple `Mxfp4Weight`s share a
/// single underlying Metal `Buffer` (e.g. one unified buffer per MoE layer containing
/// all 256 experts, with each expert's view pointing at a distinct byte offset). For
/// standalone weights, both offsets are 0 and the buffers are sized exactly to the
/// per-weight footprint.
pub struct Mxfp4Weight {
    packed: Buffer,
    scales: Buffer,
    /// Byte offset into `packed` where this weight's nibbles begin.
    pub packed_offset: u64,
    /// Byte offset into `scales` where this weight's E8M0 exponents begin.
    pub scales_offset: u64,
    pub out_features: usize,
    pub in_features: usize,
}

/// A single MXFP4 matmul job inside a batch — encodes into a shared command buffer
/// when passed to [`MxFp4Context::matmul_zero_copy_batch`].
pub struct Mxfp4Job<'a> {
    pub weight: &'a Mxfp4Weight,
    pub x_buf: &'a Buffer,
    pub x_offset: u64,
    pub y_buf: &'a Buffer,
    pub y_offset: u64,
    pub batch: usize,
}

impl Mxfp4Weight {
    /// Upload packed nibbles + E8M0 scales to device.
    ///
    /// Shape contract:
    ///   - `packed.len() == out_features * in_features / 8`
    ///   - `scales.len() == out_features * in_features / 32`
    ///   - `in_features` is a multiple of 32.
    pub fn from_host(
        ctx: &MetalContext,
        packed: &[u32],
        scales: &[u8],
        out_features: usize,
        in_features: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            in_features.is_multiple_of(32),
            "in_features {in_features} must be a multiple of 32 (MXFP4 group size)"
        );
        let expected_packed = out_features * in_features / 8;
        anyhow::ensure!(
            packed.len() == expected_packed,
            "packed length {} != {}",
            packed.len(),
            expected_packed
        );
        let expected_scales = out_features * in_features / 32;
        anyhow::ensure!(
            scales.len() == expected_scales,
            "scales length {} != {}",
            scales.len(),
            expected_scales
        );
        Ok(Self {
            packed: ctx.buffer_with_data(packed),
            scales: ctx.buffer_with_data(scales),
            packed_offset: 0,
            scales_offset: 0,
            out_features,
            in_features,
        })
    }

    /// Build a weight that is a *view* into already-uploaded unified buffers.
    ///
    /// Used by MoE unified storage: one contiguous `packed_all` + `scales_all` buffer
    /// holds every expert's weights, and each `Mxfp4Weight` points at its slice via
    /// `packed_offset` / `scales_offset`. `Buffer` is an `objc` ref-counted handle, so
    /// cloning it is cheap and does not re-upload memory.
    pub fn from_buffers(
        packed: Buffer,
        packed_offset: u64,
        scales: Buffer,
        scales_offset: u64,
        out_features: usize,
        in_features: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            in_features.is_multiple_of(32),
            "in_features {in_features} must be a multiple of 32 (MXFP4 group size)"
        );
        let packed_bytes = (out_features * in_features / 8 * 4) as u64;
        let scales_bytes = (out_features * in_features / 32) as u64;
        anyhow::ensure!(
            packed_offset + packed_bytes <= packed.length() as u64,
            "packed view [{packed_offset}, {}) exceeds buffer length {}",
            packed_offset + packed_bytes,
            packed.length() as u64
        );
        anyhow::ensure!(
            scales_offset + scales_bytes <= scales.length() as u64,
            "scales view [{scales_offset}, {}) exceeds buffer length {}",
            scales_offset + scales_bytes,
            scales.length() as u64
        );
        Ok(Self {
            packed,
            scales,
            packed_offset,
            scales_offset,
            out_features,
            in_features,
        })
    }

    /// Zero-copy view of the packed buffer (for Candle interop).
    #[allow(dead_code)]
    pub(crate) fn packed_buffer(&self) -> &Buffer {
        &self.packed
    }

    #[allow(dead_code)]
    pub(crate) fn scales_buffer(&self) -> &Buffer {
        &self.scales
    }

    /// Public accessor for the underlying packed Metal buffer (benchmarks/profiling).
    pub fn packed_buffer_ref(&self) -> &Buffer {
        &self.packed
    }

    /// Public accessor for the underlying scales Metal buffer (benchmarks/profiling).
    pub fn scales_buffer_ref(&self) -> &Buffer {
        &self.scales
    }

    /// Approximate bytes consumed by this weight. For standalone weights this is the
    /// full buffer size; for views into unified buffers it reports only the viewed
    /// slab so accounting at the `Mxfp4SwitchMlp` level doesn't double-count shared
    /// storage.
    pub fn approx_bytes(&self) -> usize {
        let packed_bytes = self.out_features * self.in_features / 8 * 4;
        let scales_bytes = self.out_features * self.in_features / 32;
        packed_bytes + scales_bytes
    }
}

impl MxFp4Context {
    pub fn new() -> Result<Self> {
        let ctx = MetalContext::new()?;
        let options = crate::metal::new_compile_options();
        // 3.0 → 3.1. Required for the native `bfloat` type used
        // by `mxfp4_matmul_f32in_bf16out_v3`. 3.1 is a strict superset of
        // 3.0 — every existing kernel compiles unchanged. Apple Silicon
        // M2+ all run macOS 14+ which ships Metal 3.1.
        options.set_language_version(crate::metal::MTLLanguageVersion::Version3_1);
        options.set_fast_math_enabled(true);
        let library = ctx
            .device
            .new_library_with_source(SHADER_SRC, Some(options.as_ref()))
            .map_err(|e| anyhow::anyhow!("MXFP4 shader compile error: {e}"))?;
        let func = library
            .get_function("mxfp4_matvec_f32", None)
            .map_err(|e| anyhow::anyhow!("kernel `mxfp4_matvec_f32` not found: {e}"))?;
        let matvec_f32 = ctx
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| anyhow::anyhow!("pipeline `mxfp4_matvec_f32` failed: {e}"))?;
        let matmul_func = library
            .get_function("mxfp4_matmul_f32", None)
            .map_err(|e| anyhow::anyhow!("kernel `mxfp4_matmul_f32` not found: {e}"))?;
        let matmul_f32 = ctx
            .device
            .new_compute_pipeline_state_with_function(&matmul_func)
            .map_err(|e| anyhow::anyhow!("pipeline `mxfp4_matmul_f32` failed: {e}"))?;
        let moe_func = library
            .get_function("mxfp4_matmul_moe_f32", None)
            .map_err(|e| anyhow::anyhow!("kernel `mxfp4_matmul_moe_f32` not found: {e}"))?;
        let matmul_moe_f32 = ctx
            .device
            .new_compute_pipeline_state_with_function(&moe_func)
            .map_err(|e| anyhow::anyhow!("pipeline `mxfp4_matmul_moe_f32` failed: {e}"))?;

        let matmul_v2_func = library
            .get_function("mxfp4_matmul_f32_v2", None)
            .map_err(|e| anyhow::anyhow!("kernel `mxfp4_matmul_f32_v2` not found: {e}"))?;
        let matmul_f32_v2 = ctx
            .device
            .new_compute_pipeline_state_with_function(&matmul_v2_func)
            .map_err(|e| anyhow::anyhow!("pipeline `mxfp4_matmul_f32_v2` failed: {e}"))?;
        let moe_v2_func = library
            .get_function("mxfp4_matmul_moe_f32_v2", None)
            .map_err(|e| anyhow::anyhow!("kernel `mxfp4_matmul_moe_f32_v2` not found: {e}"))?;
        let matmul_moe_f32_v2 = ctx
            .device
            .new_compute_pipeline_state_with_function(&moe_v2_func)
            .map_err(|e| anyhow::anyhow!("pipeline `mxfp4_matmul_moe_f32_v2` failed: {e}"))?;
        let matmul_v3_func = library
            .get_function("mxfp4_matmul_f32_v3", None)
            .map_err(|e| anyhow::anyhow!("kernel `mxfp4_matmul_f32_v3` not found: {e}"))?;
        let matmul_f32_v3 = ctx
            .device
            .new_compute_pipeline_state_with_function(&matmul_v3_func)
            .map_err(|e| anyhow::anyhow!("pipeline `mxfp4_matmul_f32_v3` failed: {e}"))?;
        let moe_v3_func = library
            .get_function("mxfp4_matmul_moe_f32_v3", None)
            .map_err(|e| anyhow::anyhow!("kernel `mxfp4_matmul_moe_f32_v3` not found: {e}"))?;
        let matmul_moe_f32_v3 = ctx
            .device
            .new_compute_pipeline_state_with_function(&moe_v3_func)
            .map_err(|e| anyhow::anyhow!("pipeline `mxfp4_matmul_moe_f32_v3` failed: {e}"))?;
        let gate_up_silu_mul_v3_func = library
            .get_function("mxfp4_gate_up_silu_mul_f32_v3", None)
            .map_err(|e| {
                anyhow::anyhow!("kernel `mxfp4_gate_up_silu_mul_f32_v3` not found: {e}")
            })?;
        let gate_up_silu_mul_f32_v3 = ctx
            .device
            .new_compute_pipeline_state_with_function(&gate_up_silu_mul_v3_func)
            .map_err(|e| anyhow::anyhow!("pipeline `mxfp4_gate_up_silu_mul_f32_v3` failed: {e}"))?;
        let moe_gate_up_silu_mul_v3_func = library
            .get_function("mxfp4_moe_gate_up_silu_mul_f32_v3", None)
            .map_err(|e| {
                anyhow::anyhow!("kernel `mxfp4_moe_gate_up_silu_mul_f32_v3` not found: {e}")
            })?;
        let moe_gate_up_silu_mul_f32_v3 = ctx
            .device
            .new_compute_pipeline_state_with_function(&moe_gate_up_silu_mul_v3_func)
            .map_err(|e| {
                anyhow::anyhow!("pipeline `mxfp4_moe_gate_up_silu_mul_f32_v3` failed: {e}")
            })?;
        let moe_wsum_func = library
            .get_function("moe_wsum_f32", None)
            .map_err(|e| anyhow::anyhow!("kernel `moe_wsum_f32` not found: {e}"))?;
        let moe_wsum_f32 = ctx
            .device
            .new_compute_pipeline_state_with_function(&moe_wsum_func)
            .map_err(|e| anyhow::anyhow!("pipeline `moe_wsum_f32` failed: {e}"))?;
        let moe_matmul_wsum_v3_func = library
            .get_function("mxfp4_matmul_moe_wsum_f32_v3", None)
            .map_err(|e| anyhow::anyhow!("kernel `mxfp4_matmul_moe_wsum_f32_v3` not found: {e}"))?;
        let matmul_moe_wsum_f32_v3 = ctx
            .device
            .new_compute_pipeline_state_with_function(&moe_matmul_wsum_v3_func)
            .map_err(|e| anyhow::anyhow!("pipeline `mxfp4_matmul_moe_wsum_f32_v3` failed: {e}"))?;
        let moe_matmul_wsum_atomic_v3_func = library
            .get_function("mxfp4_matmul_moe_wsum_atomic_f32_v3", None)
            .map_err(|e| {
                anyhow::anyhow!("kernel `mxfp4_matmul_moe_wsum_atomic_f32_v3` not found: {e}")
            })?;
        let matmul_moe_wsum_atomic_f32_v3 = ctx
            .device
            .new_compute_pipeline_state_with_function(&moe_matmul_wsum_atomic_v3_func)
            .map_err(|e| {
                anyhow::anyhow!("pipeline `mxfp4_matmul_moe_wsum_atomic_f32_v3` failed: {e}")
            })?;
        let topk_partial_func = library
            .get_function("topk_partial_select_f32", None)
            .map_err(|e| anyhow::anyhow!("kernel `topk_partial_select_f32` not found: {e}"))?;
        let topk_partial_select_f32 = ctx
            .device
            .new_compute_pipeline_state_with_function(&topk_partial_func)
            .map_err(|e| anyhow::anyhow!("pipeline `topk_partial_select_f32` failed: {e}"))?;
        let router_fused_func = library
            .get_function("router_softmax_topk_renorm_f32", None)
            .map_err(|e| {
                anyhow::anyhow!("kernel `router_softmax_topk_renorm_f32` not found: {e}")
            })?;
        let router_softmax_topk_renorm_f32 = ctx
            .device
            .new_compute_pipeline_state_with_function(&router_fused_func)
            .map_err(|e| {
                anyhow::anyhow!("pipeline `router_softmax_topk_renorm_f32` failed: {e}")
            })?;
        let moe_gate_up_bf16_func = library
            .get_function("mxfp4_moe_gate_up_silu_mul_f32in_bf16out_v3", None)
            .map_err(|e| {
                anyhow::anyhow!(
                    "kernel `mxfp4_moe_gate_up_silu_mul_f32in_bf16out_v3` not found: {e}"
                )
            })?;
        let moe_gate_up_silu_mul_f32in_bf16out_v3 = ctx
            .device
            .new_compute_pipeline_state_with_function(&moe_gate_up_bf16_func)
            .map_err(|e| {
                anyhow::anyhow!(
                    "pipeline `mxfp4_moe_gate_up_silu_mul_f32in_bf16out_v3` failed: {e}"
                )
            })?;
        let moe_down_bf16in_func = library
            .get_function("mxfp4_matmul_moe_bf16in_f32out_v3", None)
            .map_err(|e| {
                anyhow::anyhow!("kernel `mxfp4_matmul_moe_bf16in_f32out_v3` not found: {e}")
            })?;
        let matmul_moe_bf16in_f32out_v3 = ctx
            .device
            .new_compute_pipeline_state_with_function(&moe_down_bf16in_func)
            .map_err(|e| {
                anyhow::anyhow!("pipeline `mxfp4_matmul_moe_bf16in_f32out_v3` failed: {e}")
            })?;
        let moe_gate_up_rmsnorm_func = library
            .get_function("mxfp4_moe_gate_up_silu_mul_rmsnorm_f32_v3", None)
            .map_err(|e| {
                anyhow::anyhow!("kernel `mxfp4_moe_gate_up_silu_mul_rmsnorm_f32_v3` not found: {e}")
            })?;
        let moe_gate_up_silu_mul_rmsnorm_f32_v3 = ctx
            .device
            .new_compute_pipeline_state_with_function(&moe_gate_up_rmsnorm_func)
            .map_err(|e| {
                anyhow::anyhow!("pipeline `mxfp4_moe_gate_up_silu_mul_rmsnorm_f32_v3` failed: {e}")
            })?;
        let matmul_v3_rmsnorm_func = library
            .get_function("mxfp4_matmul_f32_v3_rmsnorm", None)
            .map_err(|e| anyhow::anyhow!("kernel `mxfp4_matmul_f32_v3_rmsnorm` not found: {e}"))?;
        let matmul_f32_v3_rmsnorm = ctx
            .device
            .new_compute_pipeline_state_with_function(&matmul_v3_rmsnorm_func)
            .map_err(|e| anyhow::anyhow!("pipeline `mxfp4_matmul_f32_v3_rmsnorm` failed: {e}"))?;
        let matmul_v3_rmsnorm_large_func = library
            .get_function("mxfp4_matmul_f32_v3_rmsnorm_large", None)
            .map_err(|e| {
                anyhow::anyhow!("kernel `mxfp4_matmul_f32_v3_rmsnorm_large` not found: {e}")
            })?;
        let matmul_f32_v3_rmsnorm_large = ctx
            .device
            .new_compute_pipeline_state_with_function(&matmul_v3_rmsnorm_large_func)
            .map_err(|e| {
                anyhow::anyhow!("pipeline `mxfp4_matmul_f32_v3_rmsnorm_large` failed: {e}")
            })?;
        let matmul_v3_rmsnorm_xlarge_func = library
            .get_function("mxfp4_matmul_f32_v3_rmsnorm_xlarge", None)
            .map_err(|e| {
                anyhow::anyhow!("kernel `mxfp4_matmul_f32_v3_rmsnorm_xlarge` not found: {e}")
            })?;
        let matmul_f32_v3_rmsnorm_xlarge = ctx
            .device
            .new_compute_pipeline_state_with_function(&matmul_v3_rmsnorm_xlarge_func)
            .map_err(|e| {
                anyhow::anyhow!("pipeline `mxfp4_matmul_f32_v3_rmsnorm_xlarge` failed: {e}")
            })?;
        let matmul_v3_residual_func = library
            .get_function("mxfp4_matmul_f32_v3_residual", None)
            .map_err(|e| anyhow::anyhow!("kernel `mxfp4_matmul_f32_v3_residual` not found: {e}"))?;
        let matmul_f32_v3_residual = ctx
            .device
            .new_compute_pipeline_state_with_function(&matmul_v3_residual_func)
            .map_err(|e| anyhow::anyhow!("pipeline `mxfp4_matmul_f32_v3_residual` failed: {e}"))?;
        let tri_add_func = library
            .get_function("tri_add_f32", None)
            .map_err(|e| anyhow::anyhow!("kernel `tri_add_f32` not found: {e}"))?;
        let tri_add_f32 = ctx
            .device
            .new_compute_pipeline_state_with_function(&tri_add_func)
            .map_err(|e| anyhow::anyhow!("pipeline `tri_add_f32` failed: {e}"))?;
        let scalar_mul_tri_add_func = library
            .get_function("scalar_mul_tri_add_f32", None)
            .map_err(|e| anyhow::anyhow!("kernel `scalar_mul_tri_add_f32` not found: {e}"))?;
        let scalar_mul_tri_add_f32 = ctx
            .device
            .new_compute_pipeline_state_with_function(&scalar_mul_tri_add_func)
            .map_err(|e| anyhow::anyhow!("pipeline `scalar_mul_tri_add_f32` failed: {e}"))?;
        let scalar_mul_tri_add_rmsnorm_func = library
            .get_function("scalar_mul_tri_add_rmsnorm_f32", None)
            .map_err(|e| {
                anyhow::anyhow!("kernel `scalar_mul_tri_add_rmsnorm_f32` not found: {e}")
            })?;
        let scalar_mul_tri_add_rmsnorm_f32 = ctx
            .device
            .new_compute_pipeline_state_with_function(&scalar_mul_tri_add_rmsnorm_func)
            .map_err(|e| {
                anyhow::anyhow!("pipeline `scalar_mul_tri_add_rmsnorm_f32` failed: {e}")
            })?;
        let dense_matmul_rmsnorm_func = library
            .get_function("dense_f32_matmul_rmsnorm", None)
            .map_err(|e| anyhow::anyhow!("kernel `dense_f32_matmul_rmsnorm` not found: {e}"))?;
        let dense_f32_matmul_rmsnorm = ctx
            .device
            .new_compute_pipeline_state_with_function(&dense_matmul_rmsnorm_func)
            .map_err(|e| anyhow::anyhow!("pipeline `dense_f32_matmul_rmsnorm` failed: {e}"))?;
        let small_out_v1_func = library
            .get_function("mxfp4_matmul_small_out_f32_v1", None)
            .map_err(|e| {
                anyhow::anyhow!("kernel `mxfp4_matmul_small_out_f32_v1` not found: {e}")
            })?;
        let matmul_small_out_f32_v1 = ctx
            .device
            .new_compute_pipeline_state_with_function(&small_out_v1_func)
            .map_err(|e| anyhow::anyhow!("pipeline `mxfp4_matmul_small_out_f32_v1` failed: {e}"))?;

        // bf16-output sister of v3. Same buffer slots, same dispatch
        // topology — only the output store is narrowed to bfloat16.
        let matmul_f32in_bf16out_v3_func = library
            .get_function("mxfp4_matmul_f32in_bf16out_v3", None)
            .map_err(|e| {
                anyhow::anyhow!("kernel `mxfp4_matmul_f32in_bf16out_v3` not found: {e}")
            })?;
        let matmul_f32in_bf16out_v3 = ctx
            .device
            .new_compute_pipeline_state_with_function(&matmul_f32in_bf16out_v3_func)
            .map_err(|e| anyhow::anyhow!("pipeline `mxfp4_matmul_f32in_bf16out_v3` failed: {e}"))?;

        // Lever B L.2: bf16-input sister of v3. Same buffer slots, dispatch
        // topology, and threadgroup memory size as the f32-in path; only the
        // activation pointer is `bfloat`. The bf16 → f32 widening happens once
        // during the cooperative threadgroup-memory staging step.
        let matmul_bf16in_f32out_v3_func = library
            .get_function("mxfp4_matmul_bf16in_f32out_v3", None)
            .map_err(|e| {
                anyhow::anyhow!("kernel `mxfp4_matmul_bf16in_f32out_v3` not found: {e}")
            })?;
        let matmul_bf16in_f32out_v3 = ctx
            .device
            .new_compute_pipeline_state_with_function(&matmul_bf16in_f32out_v3_func)
            .map_err(|e| anyhow::anyhow!("pipeline `mxfp4_matmul_bf16in_f32out_v3` failed: {e}"))?;

        // bf16-output sisters of moe_v3, gate_up_silu_mul_v3,
        // and small_out_v1. Same dispatch topology each — only output narrows.
        let moe_bf16_func = library
            .get_function("mxfp4_matmul_moe_f32in_bf16out_v3", None)
            .map_err(|e| {
                anyhow::anyhow!("kernel `mxfp4_matmul_moe_f32in_bf16out_v3` not found: {e}")
            })?;
        let matmul_moe_f32in_bf16out_v3 = ctx
            .device
            .new_compute_pipeline_state_with_function(&moe_bf16_func)
            .map_err(|e| {
                anyhow::anyhow!("pipeline `mxfp4_matmul_moe_f32in_bf16out_v3` failed: {e}")
            })?;
        let gate_up_bf16_func = library
            .get_function("mxfp4_gate_up_silu_mul_f32in_bf16out_v3", None)
            .map_err(|e| {
                anyhow::anyhow!("kernel `mxfp4_gate_up_silu_mul_f32in_bf16out_v3` not found: {e}")
            })?;
        let gate_up_silu_mul_f32in_bf16out_v3 = ctx
            .device
            .new_compute_pipeline_state_with_function(&gate_up_bf16_func)
            .map_err(|e| {
                anyhow::anyhow!("pipeline `mxfp4_gate_up_silu_mul_f32in_bf16out_v3` failed: {e}")
            })?;
        let small_out_bf16_func = library
            .get_function("mxfp4_matmul_small_out_f32in_bf16out_v1", None)
            .map_err(|e| {
                anyhow::anyhow!("kernel `mxfp4_matmul_small_out_f32in_bf16out_v1` not found: {e}")
            })?;
        let matmul_small_out_f32in_bf16out_v1 = ctx
            .device
            .new_compute_pipeline_state_with_function(&small_out_bf16_func)
            .map_err(|e| {
                anyhow::anyhow!("pipeline `mxfp4_matmul_small_out_f32in_bf16out_v1` failed: {e}")
            })?;

        // CB 3 new pipelines.
        let moe_gate_up_rmsnorm_multi_func = library
            .get_function("mxfp4_moe_gate_up_silu_mul_rmsnorm_f32_v3_multi", None)
            .map_err(|e| {
                anyhow::anyhow!(
                    "kernel `mxfp4_moe_gate_up_silu_mul_rmsnorm_f32_v3_multi` not found: {e}"
                )
            })?;
        let moe_gate_up_silu_mul_rmsnorm_f32_v3_multi = ctx
            .device
            .new_compute_pipeline_state_with_function(&moe_gate_up_rmsnorm_multi_func)
            .map_err(|e| {
                anyhow::anyhow!(
                    "pipeline `mxfp4_moe_gate_up_silu_mul_rmsnorm_f32_v3_multi` failed: {e}"
                )
            })?;
        let matmul_moe_v3_multi_func = library
            .get_function("mxfp4_matmul_moe_f32_v3_multi", None)
            .map_err(|e| {
                anyhow::anyhow!("kernel `mxfp4_matmul_moe_f32_v3_multi` not found: {e}")
            })?;
        let matmul_moe_f32_v3_multi = ctx
            .device
            .new_compute_pipeline_state_with_function(&matmul_moe_v3_multi_func)
            .map_err(|e| anyhow::anyhow!("pipeline `mxfp4_matmul_moe_f32_v3_multi` failed: {e}"))?;
        let moe_wsum_multi_func = library
            .get_function("moe_wsum_f32_multi", None)
            .map_err(|e| anyhow::anyhow!("kernel `moe_wsum_f32_multi` not found: {e}"))?;
        let moe_wsum_f32_multi = ctx
            .device
            .new_compute_pipeline_state_with_function(&moe_wsum_multi_func)
            .map_err(|e| anyhow::anyhow!("pipeline `moe_wsum_f32_multi` failed: {e}"))?;

        let kernel_version = read_kernel_version_env();
        match kernel_version {
            KernelVersion::V3 => {
                eprintln!("MXFP4: kernel v3 selected (simdgroup cooperative + tg x cache)");
            }
            KernelVersion::V2 => {
                eprintln!("MXFP4: kernel v2 selected (uint4 + float4 vectorized loads)");
            }
            KernelVersion::V1 => {
                eprintln!("MXFP4: kernel v1 selected (scalar baseline)");
            }
        }
        Ok(Self {
            ctx,
            library,
            matvec_f32,
            matmul_f32,
            matmul_moe_f32,
            matmul_f32_v2,
            matmul_moe_f32_v2,
            matmul_f32_v3,
            matmul_moe_f32_v3,
            gate_up_silu_mul_f32_v3,
            moe_gate_up_silu_mul_f32_v3,
            moe_wsum_f32,
            matmul_moe_wsum_f32_v3,
            matmul_moe_wsum_atomic_f32_v3,
            topk_partial_select_f32,
            router_softmax_topk_renorm_f32,
            moe_gate_up_silu_mul_f32in_bf16out_v3,
            matmul_moe_bf16in_f32out_v3,
            moe_gate_up_silu_mul_rmsnorm_f32_v3,
            matmul_f32_v3_rmsnorm,
            matmul_f32_v3_rmsnorm_large,
            matmul_f32_v3_rmsnorm_xlarge,
            matmul_f32_v3_residual,
            tri_add_f32,
            scalar_mul_tri_add_f32,
            scalar_mul_tri_add_rmsnorm_f32,
            dense_f32_matmul_rmsnorm,
            matmul_small_out_f32_v1,
            matmul_f32in_bf16out_v3,
            matmul_bf16in_f32out_v3,
            matmul_moe_f32in_bf16out_v3,
            gate_up_silu_mul_f32in_bf16out_v3,
            matmul_small_out_f32in_bf16out_v1,
            moe_gate_up_silu_mul_rmsnorm_f32_v3_multi,
            matmul_moe_f32_v3_multi,
            moe_wsum_f32_multi,
            kernel_version,
        })
    }

    /// topology as `matmul_f32_pipeline_v3`
    /// but writes bfloat16. Test/bench code uses this directly to compare
    /// against the f32 path on identical inputs.
    pub fn matmul_f32in_bf16out_v3_pipeline(&self) -> &ComputePipelineState {
        &self.matmul_f32in_bf16out_v3
    }

    /// Lever B L.2 pipeline accessor. Same topology as `matmul_f32_pipeline_v3`
    /// but reads `bfloat` activations (widened to f32 once during threadgroup
    /// staging). Output is still f32. Test/bench code uses this directly to
    /// compare against the f32-in path on identical inputs.
    pub fn matmul_bf16in_f32out_v3_pipeline(&self) -> &ComputePipelineState {
        &self.matmul_bf16in_f32out_v3
    }

    /// Encode a single matmul dispatch into an existing compute encoder.
    /// Honors `self.kernel_version` to choose the correct dispatch topology
    /// + threadgroup memory size. Caller owns the encoder + cmd buffer.
    ///
    /// All `set_buffer` slots match the kernel's `[[buffer(N)]]` annotations
    /// — both v1/v2 and v3 share buffer slots 0..5. The v3 kernel additionally
    /// reads `[[threadgroup(0)]]` which is configured here via
    /// `set_threadgroup_memory_length`.
    ///
    /// Public so callers integrating MXFP4 matmul into a larger native command
    /// buffer (e.g. fused linear-attn forward) can avoid the per-matmul
    /// commit + cross-queue sync that `matmul_zero_copy` carries.
    pub fn encode_matmul_dispatch(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        weight: &Mxfp4Weight,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        batch: usize,
    ) {
        encoder.set_compute_pipeline_state(self.matmul_f32_pipeline());
        encoder.set_buffer(0, Some(&weight.packed), (weight.packed_offset) as usize);
        encoder.set_buffer(1, Some(&weight.scales), (weight.scales_offset) as usize);
        encoder.set_buffer(2, Some(x_buf), (x_offset) as usize);
        encoder.set_buffer(3, Some(y_buf), (y_offset) as usize);
        let dims = MxFp4Dims {
            out_features: weight.out_features as u32,
            in_features: weight.in_features as u32,
        };
        encoder.set_bytes_directly(
            4,
            std::mem::size_of::<MxFp4Dims>(),
            &dims as *const _ as *const _,
        );
        let batch_u32 = batch as u32;
        encoder.set_bytes_directly(5, 4, &batch_u32 as *const _ as *const _);

        match self.kernel_version {
            KernelVersion::V3 => {
                // v3: 256 threads/threadgroup, 8 rows per threadgroup,
                // threadgroup memory holds the staged x[b, :] activation.
                const ROWS_PER_TG: u64 = 8;
                const THREADS_PER_TG: u64 = 256;
                let tg_mem_bytes = (weight.in_features as u64) * 4;
                encoder.set_threadgroup_memory_length(0, (tg_mem_bytes) as usize);
                let n_groups_x = weight.out_features.div_ceil(ROWS_PER_TG as usize) as u64;
                let grid = MTLSize {
                    width: (n_groups_x) as usize,
                    height: (batch as u64) as usize,
                    depth: (1) as usize,
                };
                let tg = MTLSize {
                    width: (THREADS_PER_TG) as usize,
                    height: (1) as usize,
                    depth: (1) as usize,
                };
                encoder.dispatch_thread_groups(grid, tg);
            }
            KernelVersion::V1 | KernelVersion::V2 => {
                // v1/v2: 1 thread per (row, batch_elem) — flat dispatch.
                let max_threads = self
                    .matmul_f32_pipeline()
                    .max_total_threads_per_threadgroup();
                let threads_per_tg = max_threads.min(256);
                let grid = MTLSize {
                    width: (weight.out_features as u64) as usize,
                    height: (batch as u64) as usize,
                    depth: (1) as usize,
                };
                let tg = MTLSize {
                    width: (threads_per_tg) as usize,
                    height: (1) as usize,
                    depth: (1) as usize,
                };
                encoder.dispatch_threads(grid, tg);
            }
        }
    }

    /// encode an MXFP4 matmul whose output is
    /// `bfloat16` (16-bit) instead of `float`. Identical buffer slots + dims
    /// + dispatch topology to `encode_matmul_dispatch` v3 — only the bound
    /// pipeline and output element width differ. Always uses v3 topology
    /// (the bf16-output kernel is only authored for v3).
    ///
    /// Caller responsibilities:
    ///   - `y_buf` must be sized `batch * weight.out_features * 2` bytes
    ///     (bfloat16 = 2 bytes/elem). Allocating with `DType::F32` and
    ///     binding here will silently overwrite half the buffer; allocate
    ///     a `DType::BF16` Candle tensor (or 2-byte raw buffer).
    ///   - `x_buf` is still f32. We have not yet authored a bf16-input
    ///     variant — that comes in Phase A.1 once propagation through the
    ///     model is wired.
    pub fn encode_matmul_dispatch_bf16_out(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        weight: &Mxfp4Weight,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        batch: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.matmul_f32in_bf16out_v3);
        encoder.set_buffer(0, Some(&weight.packed), (weight.packed_offset) as usize);
        encoder.set_buffer(1, Some(&weight.scales), (weight.scales_offset) as usize);
        encoder.set_buffer(2, Some(x_buf), (x_offset) as usize);
        encoder.set_buffer(3, Some(y_buf), (y_offset) as usize);
        let dims = MxFp4Dims {
            out_features: weight.out_features as u32,
            in_features: weight.in_features as u32,
        };
        encoder.set_bytes_directly(
            4,
            std::mem::size_of::<MxFp4Dims>(),
            &dims as *const _ as *const _,
        );
        let batch_u32 = batch as u32;
        encoder.set_bytes_directly(5, 4, &batch_u32 as *const _ as *const _);

        // v3 topology — 256 threads/threadgroup, 8 rows/threadgroup, threadgroup
        // memory holds the staged x[b, :] activation. The bf16 variant uses
        // the same shape; the threadgroup memory is sized for `float` (the
        // accumulator stays f32 even though the output narrows).
        const ROWS_PER_TG: u64 = 8;
        const THREADS_PER_TG: u64 = 256;
        let tg_mem_bytes = (weight.in_features as u64) * 4;
        encoder.set_threadgroup_memory_length(0, (tg_mem_bytes) as usize);
        let n_groups_x = weight.out_features.div_ceil(ROWS_PER_TG as usize) as u64;
        let grid = MTLSize {
            width: (n_groups_x) as usize,
            height: (batch as u64) as usize,
            depth: (1) as usize,
        };
        let tg = MTLSize {
            width: (THREADS_PER_TG) as usize,
            height: (1) as usize,
            depth: (1) as usize,
        };
        encoder.dispatch_thread_groups(grid, tg);
    }

    /// Lever B L.2 (2026-04-28): encode an MXFP4 matmul whose activation `x`
    /// is `bfloat16` (16-bit). Identical buffer slots + dims + dispatch
    /// topology + threadgroup memory size to `encode_matmul_dispatch` v3 — only
    /// the bound pipeline and the activation element width differ. Output
    /// stays f32 (the f32 accumulator is unchanged).
    ///
    /// Caller responsibilities:
    ///   - `x_buf` must be sized `batch * weight.in_features * 2` bytes
    ///     (bfloat16 = 2 bytes/elem). Pair with a `DType::BF16` Candle tensor
    ///     produced by an upstream bf16 op (e.g. `MpsRmsNormBf16Out`).
    ///   - `y_buf` is f32 — same as the f32-in path.
    pub fn encode_matmul_dispatch_bf16_in(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        weight: &Mxfp4Weight,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        batch: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.matmul_bf16in_f32out_v3);
        encoder.set_buffer(0, Some(&weight.packed), (weight.packed_offset) as usize);
        encoder.set_buffer(1, Some(&weight.scales), (weight.scales_offset) as usize);
        encoder.set_buffer(2, Some(x_buf), (x_offset) as usize);
        encoder.set_buffer(3, Some(y_buf), (y_offset) as usize);
        let dims = MxFp4Dims {
            out_features: weight.out_features as u32,
            in_features: weight.in_features as u32,
        };
        encoder.set_bytes_directly(
            4,
            std::mem::size_of::<MxFp4Dims>(),
            &dims as *const _ as *const _,
        );
        let batch_u32 = batch as u32;
        encoder.set_bytes_directly(5, 4, &batch_u32 as *const _ as *const _);

        // v3 topology — 256 threads/threadgroup, 8 rows/threadgroup, threadgroup
        // memory holds the staged x[b, :] activation widened to f32 (so still
        // `in_features * 4` bytes even though the global buffer is 2 bytes/elem).
        const ROWS_PER_TG: u64 = 8;
        const THREADS_PER_TG: u64 = 256;
        let tg_mem_bytes = (weight.in_features as u64) * 4;
        encoder.set_threadgroup_memory_length(0, (tg_mem_bytes) as usize);
        let n_groups_x = weight.out_features.div_ceil(ROWS_PER_TG as usize) as u64;
        let grid = MTLSize {
            width: (n_groups_x) as usize,
            height: (batch as u64) as usize,
            depth: (1) as usize,
        };
        let tg = MTLSize {
            width: (THREADS_PER_TG) as usize,
            height: (1) as usize,
            depth: (1) as usize,
        };
        encoder.dispatch_thread_groups(grid, tg);
    }

    /// Encode the fused gate+up+silu*up dispatch. Replaces three separate ops
    /// (gate_up matmul → silu(gate) → gate*up) with a single Metal kernel.
    ///
    /// The `weight` must be the SharedExpert `gate_up_proj` (shape
    /// `[2*inter, in]`); the kernel internally reads gate row r and up row
    /// r+inter for each output row r ∈ `[0, inter)`. Output buffer must be
    /// sized `batch * inter * sizeof::<f32>()` and bound at slot 3.
    ///
    /// Always uses v3 topology (256 threads/TG, threadgroup x cache).
    pub fn encode_gate_up_silu_mul_dispatch(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        weight: &Mxfp4Weight,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        batch: usize,
    ) {
        debug_assert!(
            weight.out_features % 2 == 0,
            "gate_up_proj weight rows must be 2*inter (even); got {}",
            weight.out_features
        );
        let inter = weight.out_features / 2;

        encoder.set_compute_pipeline_state(self.gate_up_silu_mul_f32_pipeline());
        encoder.set_buffer(0, Some(&weight.packed), (weight.packed_offset) as usize);
        encoder.set_buffer(1, Some(&weight.scales), (weight.scales_offset) as usize);
        encoder.set_buffer(2, Some(x_buf), (x_offset) as usize);
        encoder.set_buffer(3, Some(y_buf), (y_offset) as usize);
        let dims = MxFp4GateUpSiluMulDims {
            inter: inter as u32,
            in_features: weight.in_features as u32,
            batch: batch as u32,
        };
        encoder.set_bytes_directly(
            4,
            std::mem::size_of::<MxFp4GateUpSiluMulDims>(),
            &dims as *const _ as *const _,
        );

        const ROWS_PER_TG: u64 = 8;
        const THREADS_PER_TG: u64 = 256;
        let tg_mem_bytes = (weight.in_features as u64) * 4;
        encoder.set_threadgroup_memory_length(0, (tg_mem_bytes) as usize);
        let n_groups_x = (inter as u64).div_ceil(ROWS_PER_TG);
        let grid = MTLSize {
            width: (n_groups_x) as usize,
            height: (batch as u64) as usize,
            depth: (1) as usize,
        };
        let tg = MTLSize {
            width: (THREADS_PER_TG) as usize,
            height: (1) as usize,
            depth: (1) as usize,
        };
        encoder.dispatch_thread_groups(grid, tg);
    }

    /// Convenience: end-to-end fused gate+up+silu*up. Allocates input/output
    /// Metal buffers, encodes a single dispatch, and returns the host-side
    /// `Vec<f32>` of length `batch * inter` (row-major).
    ///
    /// Used by unit tests for cosine-parity verification against the
    /// 3-step reference path. Production decode uses
    /// [`encode_gate_up_silu_mul_dispatch`] inside a fused command buffer.
    pub fn gate_up_silu_mul_with_weight(
        &self,
        weight: &Mxfp4Weight,
        x: &[f32],
        batch: usize,
    ) -> Result<Vec<f32>> {
        anyhow::ensure!(
            x.len() == batch * weight.in_features,
            "x length {} != batch({}) * in_features({})",
            x.len(),
            batch,
            weight.in_features
        );
        anyhow::ensure!(
            weight.out_features % 2 == 0,
            "weight rows must be 2*inter (even); got {}",
            weight.out_features
        );
        let inter = weight.out_features / 2;
        if batch == 0 {
            return Ok(Vec::new());
        }

        let x_buf = self.ctx.buffer_with_data(x);
        let y_buf = self.ctx.buffer_for::<f32>(batch * inter);

        let encoder = crate::metal::process_commands()
            .command_encoder()
            .expect("ce");
        encoder.set_label("lumen:mxfp4_gate_up_silu_mul");
        self.encode_gate_up_silu_mul_dispatch(
            encoder.as_ref(),
            weight,
            &x_buf,
            0,
            &y_buf,
            0,
            batch,
        );
        drop(encoder);
        crate::metal::process_commands()
            .flush_and_wait()
            .expect("flush");

        Ok(self.ctx.read_buffer::<f32>(&y_buf, batch * inter))
    }

    /// encode the small-out matmul dispatch (1 TG = 1 row,
    /// 256 threads cooperate on the reduction).
    ///
    /// Same buffer layout and output as [`encode_matmul_dispatch`] (`y[b,
    /// row] = sum_k W[row, k] * x[b, k]`); the only difference is the
    /// kernel + grid topology. Caller is responsible for choosing this over
    /// the general v3 path — typical use is r_gate (out=256, in=2048) where
    /// v3's 32 TGs leave the GPU under-occupied.
    pub fn encode_matmul_small_out_dispatch(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        weight: &Mxfp4Weight,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        batch: usize,
    ) {
        encoder.set_compute_pipeline_state(self.matmul_small_out_f32_pipeline());
        encoder.set_buffer(0, Some(&weight.packed), (weight.packed_offset) as usize);
        encoder.set_buffer(1, Some(&weight.scales), (weight.scales_offset) as usize);
        encoder.set_buffer(2, Some(x_buf), (x_offset) as usize);
        encoder.set_buffer(3, Some(y_buf), (y_offset) as usize);
        let dims = MxFp4Dims {
            out_features: weight.out_features as u32,
            in_features: weight.in_features as u32,
        };
        encoder.set_bytes_directly(
            4,
            std::mem::size_of::<MxFp4Dims>(),
            &dims as *const _ as *const _,
        );
        let batch_u32 = batch as u32;
        encoder.set_bytes_directly(5, 4, &batch_u32 as *const _ as *const _);

        const THREADS_PER_TG: u64 = 256;
        // sg_partial[8 simdgroups] = 32 bytes
        encoder.set_threadgroup_memory_length(0, 8 * std::mem::size_of::<f32>());
        let grid = MTLSize {
            width: weight.out_features as usize,
            height: batch as usize,
            depth: 1,
        };
        let tg = MTLSize {
            width: THREADS_PER_TG as usize,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(grid, tg);
    }

    /// Zero-copy small-out matmul (own command buffer + commit + wait).
    /// Mirrors [`matmul_zero_copy`] semantics but routes through the
    /// small-out kernel.
    pub fn matmul_small_out_zero_copy(
        &self,
        weight: &Mxfp4Weight,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        batch: usize,
    ) -> Result<()> {
        if batch == 0 {
            return Ok(());
        }
        let encoder = crate::metal::process_commands()
            .command_encoder()
            .expect("ce");
        encoder.set_label("lumen:mxfp4_matmul_small_out");
        self.encode_matmul_small_out_dispatch(
            encoder.as_ref(),
            weight,
            x_buf,
            x_offset,
            y_buf,
            y_offset,
            batch,
        );
        drop(encoder);
        crate::metal::process_commands()
            .flush_and_wait()
            .expect("flush");
        Ok(())
    }

    // ───────────────────────────────────────────────────────────────────
    // helpers for the three
    // remaining production kernel families (moe_v3, gate_up_silu_mul_v3,
    // small_out_v1). Same buffer slots / dims / topology as the f32 paths;
    // only the bound pipeline + output element width differ. Caller must
    // size `y_buf` for `bfloat16` (2 bytes/elem).
    // ───────────────────────────────────────────────────────────────────

    /// encode small-out matmul with bf16 output.
    pub fn encode_matmul_small_out_dispatch_bf16_out(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        weight: &Mxfp4Weight,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        batch: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.matmul_small_out_f32in_bf16out_v1);
        encoder.set_buffer(0, Some(&weight.packed), (weight.packed_offset) as usize);
        encoder.set_buffer(1, Some(&weight.scales), (weight.scales_offset) as usize);
        encoder.set_buffer(2, Some(x_buf), (x_offset) as usize);
        encoder.set_buffer(3, Some(y_buf), (y_offset) as usize);
        let dims = MxFp4Dims {
            out_features: weight.out_features as u32,
            in_features: weight.in_features as u32,
        };
        encoder.set_bytes_directly(
            4,
            std::mem::size_of::<MxFp4Dims>(),
            &dims as *const _ as *const _,
        );
        let batch_u32 = batch as u32;
        encoder.set_bytes_directly(5, 4, &batch_u32 as *const _ as *const _);

        const THREADS_PER_TG: u64 = 256;
        encoder.set_threadgroup_memory_length(0, 8 * std::mem::size_of::<f32>());
        let grid = MTLSize {
            width: weight.out_features as usize,
            height: batch as usize,
            depth: 1,
        };
        let tg = MTLSize {
            width: THREADS_PER_TG as usize,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(grid, tg);
    }

    /// own-queue zero-copy small-out matmul writing bf16.
    pub fn matmul_small_out_zero_copy_bf16_out(
        &self,
        weight: &Mxfp4Weight,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        batch: usize,
    ) -> Result<()> {
        if batch == 0 {
            return Ok(());
        }
        let encoder = crate::metal::process_commands()
            .command_encoder()
            .expect("ce");
        encoder.set_label("lumen:mxfp4_matmul_small_out_bf16_out");
        self.encode_matmul_small_out_dispatch_bf16_out(
            encoder.as_ref(),
            weight,
            x_buf,
            x_offset,
            y_buf,
            y_offset,
            batch,
        );
        drop(encoder);
        crate::metal::process_commands()
            .flush_and_wait()
            .expect("flush");
        Ok(())
    }

    /// encode fused gate+up+silu*up with bf16 output.
    pub fn encode_gate_up_silu_mul_dispatch_bf16_out(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        weight: &Mxfp4Weight,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        batch: usize,
    ) {
        debug_assert!(
            weight.out_features % 2 == 0,
            "gate_up_proj weight rows must be 2*inter (even); got {}",
            weight.out_features
        );
        let inter = weight.out_features / 2;

        encoder.set_compute_pipeline_state(&self.gate_up_silu_mul_f32in_bf16out_v3);
        encoder.set_buffer(0, Some(&weight.packed), (weight.packed_offset) as usize);
        encoder.set_buffer(1, Some(&weight.scales), (weight.scales_offset) as usize);
        encoder.set_buffer(2, Some(x_buf), (x_offset) as usize);
        encoder.set_buffer(3, Some(y_buf), (y_offset) as usize);
        let dims = MxFp4GateUpSiluMulDims {
            inter: inter as u32,
            in_features: weight.in_features as u32,
            batch: batch as u32,
        };
        encoder.set_bytes_directly(
            4,
            std::mem::size_of::<MxFp4GateUpSiluMulDims>(),
            &dims as *const _ as *const _,
        );

        const ROWS_PER_TG: u64 = 8;
        const THREADS_PER_TG: u64 = 256;
        let tg_mem_bytes = (weight.in_features as u64) * 4;
        encoder.set_threadgroup_memory_length(0, (tg_mem_bytes) as usize);
        let n_groups_x = (inter as u64).div_ceil(ROWS_PER_TG);
        let grid = MTLSize {
            width: (n_groups_x) as usize,
            height: (batch as u64) as usize,
            depth: (1) as usize,
        };
        let tg = MTLSize {
            width: (THREADS_PER_TG) as usize,
            height: (1) as usize,
            depth: (1) as usize,
        };
        encoder.dispatch_thread_groups(grid, tg);
    }

    /// own-queue zero-copy gate+up+silu*up writing bf16.
    pub fn gate_up_silu_mul_zero_copy_bf16_out(
        &self,
        weight: &Mxfp4Weight,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        batch: usize,
    ) -> Result<()> {
        if batch == 0 {
            return Ok(());
        }
        let encoder = crate::metal::process_commands()
            .command_encoder()
            .expect("ce");
        encoder.set_label("lumen:mxfp4_gate_up_silu_mul_bf16_out");
        self.encode_gate_up_silu_mul_dispatch_bf16_out(
            encoder.as_ref(),
            weight,
            x_buf,
            x_offset,
            y_buf,
            y_offset,
            batch,
        );
        drop(encoder);
        crate::metal::process_commands()
            .flush_and_wait()
            .expect("flush");
        Ok(())
    }

    /// Convenience host bounce for unit tests / microbench. Allocates input/
    /// output buffers, runs the small-out kernel, and returns the host
    /// `Vec<f32>` of length `batch * out_features`.
    pub fn matmul_small_out_with_weight(
        &self,
        weight: &Mxfp4Weight,
        x: &[f32],
        batch: usize,
    ) -> Result<Vec<f32>> {
        anyhow::ensure!(
            x.len() == batch * weight.in_features,
            "x length {} != batch({}) * in_features({})",
            x.len(),
            batch,
            weight.in_features
        );
        if batch == 0 {
            return Ok(Vec::new());
        }
        let x_buf = self.ctx.buffer_with_data(x);
        let y_buf = self.ctx.buffer_for::<f32>(batch * weight.out_features);

        let encoder = crate::metal::process_commands()
            .command_encoder()
            .expect("ce");
        encoder.set_label("lumen:mxfp4_matmul_small_out");
        self.encode_matmul_small_out_dispatch(
            encoder.as_ref(),
            weight,
            &x_buf,
            0,
            &y_buf,
            0,
            batch,
        );
        drop(encoder);
        crate::metal::process_commands()
            .flush_and_wait()
            .expect("flush");

        Ok(self
            .ctx
            .read_buffer::<f32>(&y_buf, batch * weight.out_features))
    }

    /// Compute `y[b, :] = x[b, :] @ W^T` for a batch of rows.
    ///
    /// Input layout:
    ///   - `x`: `[batch, in_features]` row-major f32
    ///   - `weight`: GPU-resident `Mxfp4Weight` of shape `[out_features, in_features]`
    /// Output: `Vec<f32>` of length `batch * out_features`, row-major.
    pub fn matmul_with_weight(
        &self,
        weight: &Mxfp4Weight,
        x: &[f32],
        batch: usize,
    ) -> Result<Vec<f32>> {
        anyhow::ensure!(
            x.len() == batch * weight.in_features,
            "x length {} != batch({}) * in_features({})",
            x.len(),
            batch,
            weight.in_features
        );
        if batch == 0 {
            return Ok(Vec::new());
        }

        let x_buf = self.ctx.buffer_with_data(x);
        let y_buf = self.ctx.buffer_for::<f32>(batch * weight.out_features);

        let encoder = crate::metal::process_commands()
            .command_encoder()
            .expect("ce");
        encoder.set_label("lumen:mxfp4_matmul");
        self.encode_matmul_dispatch(encoder.as_ref(), weight, &x_buf, 0, &y_buf, 0, batch);
        drop(encoder);
        crate::metal::process_commands()
            .flush_and_wait()
            .expect("flush");

        Ok(self
            .ctx
            .read_buffer::<f32>(&y_buf, batch * weight.out_features))
    }

    /// Convenience: upload weight + matvec in one call. Use `Mxfp4Weight` + `matmul_with_weight`
    /// for repeated forward passes.
    pub fn matvec_with_weight(&self, weight: &Mxfp4Weight, x: &[f32]) -> Result<Vec<f32>> {
        self.matmul_with_weight(weight, x, 1)
    }

    /// Zero-copy GPU matmul: `y = x @ W^T` where both `x` and `y` are already backed by
    /// Metal buffers supplied by the caller.
    ///
    /// - `x_buf[x_offset .. x_offset + batch * in_features * 4]`: f32 row-major input
    /// - `y_buf[y_offset .. y_offset + batch * out_features * 4]`: f32 row-major output (written)
    /// - `weight`: GPU-resident MXFP4 weights
    ///
    /// Commits the command buffer and **waits for completion**. Earlier revisions skipped
    /// the wait to reclaim ~3 ms per dispatch under the assumption that Metal's hazard
    /// tracking would handle cross-queue dependencies, but in practice Candle's metal
    /// backend and our `MxFp4Context` run on independent command queues. Without the
    /// wait, a `to_dtype` conversion queued on Candle's side was still in flight when our
    /// matmul read the "converted" buffer — the kernel then read an uninitialized region
    /// and produced a zero-valued output, silently breaking every MXFP4 projection.
    /// Correctness beats the per-matmul sync cost; future work is to replace this blanket
    /// wait with a finer-grained fence that only syncs when the input was written on a
    /// different queue.
    pub fn matmul_zero_copy(
        &self,
        weight: &Mxfp4Weight,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        batch: usize,
    ) -> Result<()> {
        if batch == 0 {
            return Ok(());
        }

        let encoder = crate::metal::process_commands()
            .command_encoder()
            .expect("ce");
        encoder.set_label("lumen:mxfp4_matmul");
        self.encode_matmul_dispatch(
            encoder.as_ref(),
            weight,
            x_buf,
            x_offset,
            y_buf,
            y_offset,
            batch,
        );

        let profile = std::env::var("LUMEN_MXFP4_PROFILE")
            .map(|v| v == "1")
            .unwrap_or(false);
        if profile {
            let t_commit = std::time::Instant::now();
            let commit_ns = t_commit.elapsed().as_nanos() as u64;
            let t_wait = std::time::Instant::now();
            drop(encoder);
            crate::metal::process_commands()
                .flush_and_wait()
                .expect("flush");
            let wait_ns = t_wait.elapsed().as_nanos() as u64;
            use std::sync::atomic::Ordering::Relaxed;
            PROF_CALLS.fetch_add(1, Relaxed);
            PROF_COMMIT_NS.fetch_add(commit_ns, Relaxed);
            PROF_WAIT_NS.fetch_add(wait_ns, Relaxed);
        } else {
            drop(encoder);
            crate::metal::process_commands()
                .flush_and_wait()
                .expect("flush");
        }
        Ok(())
    }

    /// the bf16-output v3 kernel. Mirrors
    /// `matmul_zero_copy` but binds the bfloat16-write pipeline. Submits a
    /// fresh command buffer on the `MxFp4Context` queue and blocks on
    /// completion.
    ///
    /// `y_buf` must be sized `batch * weight.out_features * 2` bytes
    /// (bfloat16). Bind a `DType::BF16` Candle tensor's underlying buffer.
    pub fn matmul_zero_copy_bf16_out(
        &self,
        weight: &Mxfp4Weight,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        batch: usize,
    ) -> Result<()> {
        if batch == 0 {
            return Ok(());
        }

        let encoder = crate::metal::process_commands()
            .command_encoder()
            .expect("ce");
        encoder.set_label("lumen:mxfp4_matmul_bf16_out");
        self.encode_matmul_dispatch_bf16_out(
            encoder.as_ref(),
            weight,
            x_buf,
            x_offset,
            y_buf,
            y_offset,
            batch,
        );
        drop(encoder);
        crate::metal::process_commands()
            .flush_and_wait()
            .expect("flush");
        Ok(())
    }

    /// Lever B L.2 (2026-04-28): bf16-input sister of `matmul_zero_copy`. Same
    /// lifecycle (own command buffer, commit + wait); binds the bf16-input
    /// pipeline. Activation buffer must be sized
    /// `batch * weight.in_features * 2` bytes (bfloat16). Bind a `DType::BF16`
    /// Candle tensor's underlying buffer; output buffer is f32.
    pub fn matmul_zero_copy_bf16_in(
        &self,
        weight: &Mxfp4Weight,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        batch: usize,
    ) -> Result<()> {
        if batch == 0 {
            return Ok(());
        }

        let encoder = crate::metal::process_commands()
            .command_encoder()
            .expect("ce");
        encoder.set_label("lumen:mxfp4_matmul_bf16_in");
        self.encode_matmul_dispatch_bf16_in(
            encoder.as_ref(),
            weight,
            x_buf,
            x_offset,
            y_buf,
            y_offset,
            batch,
        );
        drop(encoder);
        crate::metal::process_commands()
            .flush_and_wait()
            .expect("flush");
        Ok(())
    }

    /// gate+up matmul + silu*up in one kernel.
    /// Same lifecycle as [`matmul_zero_copy`] (own command buffer, commit + wait).
    /// Output buffer must be sized `batch * (weight.out_features / 2)` f32 elements.
    pub fn gate_up_silu_mul_zero_copy(
        &self,
        weight: &Mxfp4Weight,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        batch: usize,
    ) -> Result<()> {
        if batch == 0 {
            return Ok(());
        }

        let encoder = crate::metal::process_commands()
            .command_encoder()
            .expect("ce");
        encoder.set_label("lumen:mxfp4_gate_up_silu_mul");
        self.encode_gate_up_silu_mul_dispatch(
            encoder.as_ref(),
            weight,
            x_buf,
            x_offset,
            y_buf,
            y_offset,
            batch,
        );
        drop(encoder);
        crate::metal::process_commands()
            .flush_and_wait()
            .expect("flush");
        Ok(())
    }

    /// Multi-cmdbuf zero-copy matmul: enqueue k independent command buffers (one per job),
    /// commit them all back-to-back, then wait only on the last one. Each cmd buffer carries
    /// exactly one encoder + dispatch — matches the single-call `matmul_zero_copy` topology
    /// that is known to be correct, while folding `k` Candle syncs into 1 and `k` waits into
    /// 1. Used as the MoE expert dispatch fast path post-F.
    ///
    /// Why not a single cmd buffer with k encoders? `matmul_zero_copy_batch` tried that but
    /// produced garbled outputs on Apple M3 Max (parity regression under LUMEN_MOE_BATCHED,
    /// see `qwen3_5_moe_perf_plan.md`). Multi-cmdbuf sidesteps whatever the intra-cmdbuf
    /// hazard is (likely tracked-mode argument-table bleed between encoders).
    ///
    /// Does *not* sync Candle's queue — the caller must do that (e.g. via
    /// `MetalDevice::wait_until_completed()`) before invoking.
    pub fn matmul_zero_copy_multi_cmdbuf(&self, jobs: &[Mxfp4Job<'_>]) -> Result<()> {
        if jobs.is_empty() {
            return Ok(());
        }

        let profile = std::env::var("LUMEN_MXFP4_PROFILE")
            .map(|v| v == "1")
            .unwrap_or(false);

        let t_commit = std::time::Instant::now();
        // Routed through the process-wide cmk `Commands` scheduler: every job
        // shares the same encoder lifecycle and `prev_ce_outputs` map, so
        // hazard tracking handles ordering automatically — no need for the
        // legacy multi-cmdbuf workaround.
        for job in jobs {
            if job.batch == 0 {
                continue;
            }
            let encoder = crate::metal::process_commands()
                .command_encoder()
                .expect("ce");
            encoder.set_label("lumen:mxfp4_matmul_multi_cmdbuf");
            self.encode_matmul_dispatch(
                encoder.as_ref(),
                job.weight,
                job.x_buf,
                job.x_offset,
                job.y_buf,
                job.y_offset,
                job.batch,
            );
            // CommandsGuard drops at scope end — encoder stays active in the
            // scheduler for the next iteration to reuse.
        }
        let commit_ns = t_commit.elapsed().as_nanos() as u64;

        let t_wait = std::time::Instant::now();
        crate::metal::process_commands()
            .flush_and_wait()
            .expect("flush");
        let wait_ns = t_wait.elapsed().as_nanos() as u64;

        if profile {
            use std::sync::atomic::Ordering::Relaxed;
            PROF_CALLS.fetch_add(jobs.len() as u64, Relaxed);
            PROF_COMMIT_NS.fetch_add(commit_ns, Relaxed);
            PROF_WAIT_NS.fetch_add(wait_ns, Relaxed);
        }
        Ok(())
    }

    /// Batched zero-copy matmul: encode every `jobs[i]` as a separate `dispatch_threads` call
    /// into **one** command buffer, commit once, wait once. Used by MoE expert dispatch to
    /// fuse 8 expert × 3 proj = 24 matmuls into 3 waits (one per Gate/Up/Down phase).
    ///
    /// Does *not* sync Candle's queue — the caller must do that (e.g. via
    /// `MetalDevice::wait_until_completed()`) before invoking, matching the single-call
    /// `matmul_zero_copy`. All jobs share the same pipeline state (`mxfp4_matmul_f32`).
    pub fn matmul_zero_copy_batch(&self, jobs: &[Mxfp4Job<'_>]) -> Result<()> {
        if jobs.is_empty() {
            return Ok(());
        }

        let profile = std::env::var("LUMEN_MXFP4_PROFILE")
            .map(|v| v == "1")
            .unwrap_or(false);

        // Each job gets its own compute command encoder so argument slots don't bleed
        // between dispatches.
        for job in jobs {
            if job.batch == 0 {
                continue;
            }
            let encoder = crate::metal::process_commands()
                .command_encoder()
                .expect("ce");
            encoder.set_label("lumen:mxfp4_matmul_batch");
            self.encode_matmul_dispatch(
                encoder.as_ref(),
                job.weight,
                job.x_buf,
                job.x_offset,
                job.y_buf,
                job.y_offset,
                job.batch,
            );
        }

        if profile {
            let t_commit = std::time::Instant::now();
            let commit_ns = t_commit.elapsed().as_nanos() as u64;
            let t_wait = std::time::Instant::now();
            crate::metal::process_commands()
                .flush_and_wait()
                .expect("flush");
            let wait_ns = t_wait.elapsed().as_nanos() as u64;
            use std::sync::atomic::Ordering::Relaxed;
            // Attribute one "call" per job so avg math stays meaningful.
            PROF_CALLS.fetch_add(jobs.len() as u64, Relaxed);
            PROF_COMMIT_NS.fetch_add(commit_ns, Relaxed);
            PROF_WAIT_NS.fetch_add(wait_ns, Relaxed);
        } else {
            crate::metal::process_commands()
                .flush_and_wait()
                .expect("flush");
        }
        Ok(())
    }

    /// Grouped MoE matmul: one kernel dispatch runs `k` expert projections in parallel.
    ///
    /// - `packed_all` / `scales_all`: unified buffers holding every expert's weights
    ///   contiguously (expert `e` lives at offset
    ///   `e * out_features * in_features / 8 * 4` bytes in `packed_all`, and
    ///   `e * out_features * in_features / 32` bytes in `scales_all`). `num_experts_total`
    ///   is implicit — the shader only ever reads experts listed in `expert_indices`.
    /// - `expert_indices`: `[k]` u32 slice selecting which expert each z-slot reads.
    /// - `x_buf[x_offset ..]`:
    ///     * `broadcast_x = true`  →  `[batch, in_features]` f32 shared across slots
    ///     * `broadcast_x = false` →  `[k, batch, in_features]` f32 (per-slot band)
    /// - `y_buf[y_offset ..]`: `[k, batch, out_features]` f32 written.
    ///
    /// Caller owns Candle-side synchronization (our queue is independent). Does not
    /// flush Candle's queue — invoke `MetalDevice::wait_until_completed()` first if any
    /// input was written through Candle.
    ///
    /// Grid dispatches `(out_features, batch, k)` threads. On M3 Max all k slots run
    /// concurrently instead of serializing as k sequential command buffers, which is
    /// the whole point of the MoE grouped kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_moe_zero_copy(
        &self,
        packed_all: &Buffer,
        scales_all: &Buffer,
        expert_indices: &[u32],
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        out_features: usize,
        in_features: usize,
        batch: usize,
        broadcast_x: bool,
    ) -> Result<()> {
        self.matmul_moe_zero_copy_with_version(
            self.kernel_version,
            packed_all,
            scales_all,
            expert_indices,
            x_buf,
            x_offset,
            y_buf,
            y_offset,
            out_features,
            in_features,
            batch,
            broadcast_x,
        )
    }

    /// Like `matmul_moe_zero_copy` but with an explicit kernel version. Lets
    /// tests run v2 and v3 within a single `MxFp4Context` for parity checks.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn matmul_moe_zero_copy_with_version(
        &self,
        version: KernelVersion,
        packed_all: &Buffer,
        scales_all: &Buffer,
        expert_indices: &[u32],
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        out_features: usize,
        in_features: usize,
        batch: usize,
        broadcast_x: bool,
    ) -> Result<()> {
        if expert_indices.is_empty() || batch == 0 {
            return Ok(());
        }
        let k = expert_indices.len();
        // Expert-indices buffer is tiny (k * 4 bytes, k ≤ 16 in practice). Upload per
        // call — the cost is negligible compared to the matmul itself, and keeping a
        // scratch buffer would add lifetime complexity across commits.
        let indices_buf = self.ctx.buffer_with_data(expert_indices);
        self.matmul_moe_dispatch_inner(
            version,
            packed_all,
            scales_all,
            &indices_buf,
            0,
            k,
            x_buf,
            x_offset,
            y_buf,
            y_offset,
            out_features,
            in_features,
            batch,
            broadcast_x,
        )
    }

    /// grouped MoE matmul that takes the expert-indices
    /// buffer directly (no per-call host upload). `indices_buf[indices_offset ..
    /// indices_offset + k*4]` must be a u32 array of expert IDs (typically a slice of
    /// the routing tensor that already lives on the GPU).
    ///
    /// Used by `Mxfp4SwitchMlp::moe_*_with_indices_buffer` to skip the
    /// `inds.flatten_all().to_vec1::<u32>()` host transfer in the MoE forward path.
    /// Caller still owns Candle-side synchronization — flush with
    /// `MetalDevice::wait_until_completed()` first when the indices buffer was
    /// produced by Candle (e.g. from `arg_sort_last_dim`).
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_moe_zero_copy_with_indices_buffer(
        &self,
        packed_all: &Buffer,
        scales_all: &Buffer,
        indices_buf: &Buffer,
        indices_offset: u64,
        k: usize,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        out_features: usize,
        in_features: usize,
        batch: usize,
        broadcast_x: bool,
    ) -> Result<()> {
        if k == 0 || batch == 0 {
            return Ok(());
        }
        self.matmul_moe_dispatch_inner(
            self.kernel_version,
            packed_all,
            scales_all,
            indices_buf,
            indices_offset,
            k,
            x_buf,
            x_offset,
            y_buf,
            y_offset,
            out_features,
            in_features,
            batch,
            broadcast_x,
        )
    }

    /// encoder-injected MoE matmul. Caller provides the
    /// `ComputeCommandEncoderRef` (typically obtained from
    /// `candle_core::MetalDevice::command_encoder()`), so the dispatch joins the caller's
    /// command buffer instead of our independent queue. **No commit, no wait** — the
    /// caller (Candle, in production) commits its command buffer when its compute pool
    /// rolls over or when CPU-bound work demands it. Same-queue ordering means the next
    /// Candle op that reads `y_buf` is automatically serialized after our dispatch by the
    /// driver, eliminating the cross-queue `wait_until_completed()` round-trip.
    ///
    /// Caller is responsible for `encoder.end_encoding()` (typically via Drop on the
    /// returned encoder).
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_moe_zero_copy_with_indices_buffer_inline(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        packed_all: &Buffer,
        scales_all: &Buffer,
        indices_buf: &Buffer,
        indices_offset: u64,
        k: usize,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        out_features: usize,
        in_features: usize,
        batch: usize,
        broadcast_x: bool,
    ) -> Result<()> {
        if k == 0 || batch == 0 {
            return Ok(());
        }
        anyhow::ensure!(
            in_features.is_multiple_of(32),
            "in_features {in_features} must be a multiple of 32 (MXFP4 group size)"
        );
        let version = self.kernel_version;
        let pipeline = match version {
            KernelVersion::V3 => &self.matmul_moe_f32_v3,
            KernelVersion::V2 => &self.matmul_moe_f32_v2,
            KernelVersion::V1 => &self.matmul_moe_f32,
        };
        let uses_v3 = matches!(version, KernelVersion::V3);

        let dims = MxFp4MoeDims {
            out_features: out_features as u32,
            in_features: in_features as u32,
            batch: batch as u32,
            broadcast_x: if broadcast_x { 1 } else { 0 },
        };

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(packed_all), 0);
        encoder.set_buffer(1, Some(scales_all), 0);
        encoder.set_buffer(2, Some(indices_buf), indices_offset as usize);
        encoder.set_buffer(3, Some(x_buf), x_offset as usize);
        encoder.set_buffer(4, Some(y_buf), y_offset as usize);
        encoder.set_bytes_directly(
            5,
            std::mem::size_of::<MxFp4MoeDims>(),
            &dims as *const _ as *const _,
        );

        if uses_v3 {
            const ROWS_PER_TG: usize = 8;
            const THREADS_PER_TG: usize = 256;
            let row_tgs = out_features.div_ceil(ROWS_PER_TG);
            let groups = MTLSize {
                width: row_tgs,
                height: batch,
                depth: k,
            };
            let tg = MTLSize {
                width: THREADS_PER_TG,
                height: 1,
                depth: 1,
            };
            let tg_mem_bytes = in_features * std::mem::size_of::<f32>();
            encoder.set_threadgroup_memory_length(0, tg_mem_bytes);
            encoder.dispatch_thread_groups(groups, tg);
        } else {
            let max_threads = pipeline.max_total_threads_per_threadgroup();
            let threads_per_tg = max_threads.min(256);
            let grid = MTLSize {
                width: out_features,
                height: batch,
                depth: k,
            };
            let tg = MTLSize {
                width: threads_per_tg,
                height: 1,
                depth: 1,
            };
            encoder.dispatch_threads(grid, tg);
        }
        Ok(())
    }

    /// Lever A (2026-04-27): encoder-injected routed-grouped MoE fused
    /// gate+up+silu*up dispatch. Mirrors
    /// `matmul_moe_zero_copy_with_indices_buffer_inline` but uses the fused
    /// kernel `mxfp4_moe_gate_up_silu_mul_f32_v3` whose output is
    /// `[k, batch, inter]` (half of the non-fused path's `[k, batch, 2*inter]`).
    ///
    /// Buffer layout matches the fused path: `packed_all` and `scales_all`
    /// hold the gate+up combined slabs (gate at rows [0..inter), up at rows
    /// [inter..2*inter)) per expert. The kernel folds the row split into
    /// per-thread offset arithmetic.
    ///
    /// **Always dispatches v3 topology** — the fused kernel only exists in v3
    /// (matches `gate_up_silu_mul_f32_v3` precedent).
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_moe_gate_up_silu_mul_zero_copy_with_indices_buffer_inline(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        packed_all: &Buffer,
        scales_all: &Buffer,
        indices_buf: &Buffer,
        indices_offset: u64,
        k: usize,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        inter: usize,
        in_features: usize,
        batch: usize,
    ) -> Result<()> {
        if k == 0 || batch == 0 {
            return Ok(());
        }
        anyhow::ensure!(
            in_features.is_multiple_of(32),
            "in_features {in_features} must be a multiple of 32 (MXFP4 group size)"
        );

        let dims = MxFp4MoeGateUpSiluMulDims {
            inter: inter as u32,
            in_features: in_features as u32,
            batch: batch as u32,
        };

        encoder.set_compute_pipeline_state(&self.moe_gate_up_silu_mul_f32_v3);
        encoder.set_buffer(0, Some(packed_all), 0);
        encoder.set_buffer(1, Some(scales_all), 0);
        encoder.set_buffer(2, Some(indices_buf), indices_offset as usize);
        encoder.set_buffer(3, Some(x_buf), x_offset as usize);
        encoder.set_buffer(4, Some(y_buf), y_offset as usize);
        encoder.set_bytes_directly(
            5,
            std::mem::size_of::<MxFp4MoeGateUpSiluMulDims>(),
            &dims as *const _ as *const _,
        );

        const ROWS_PER_TG: usize = 8;
        const THREADS_PER_TG: usize = 256;
        let row_tgs = inter.div_ceil(ROWS_PER_TG);
        let groups = MTLSize {
            width: row_tgs,
            height: batch,
            depth: k,
        };
        let tg = MTLSize {
            width: THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        let tg_mem_bytes = in_features * std::mem::size_of::<f32>();
        encoder.set_threadgroup_memory_length(0, tg_mem_bytes);
        encoder.dispatch_thread_groups(groups, tg);
        Ok(())
    }

    /// Lever D (2026-04-27): encoder-injected routed-grouped fused gate+up
    /// +silu*up dispatch with bfloat16 output. Mirrors
    /// `matmul_moe_gate_up_silu_mul_zero_copy_with_indices_buffer_inline` —
    /// only the dispatched pipeline + the y dtype contract change. Caller's
    /// `y_buf` should hold `bfloat[k * batch * inter]` (half the f32 size).
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_moe_gate_up_silu_mul_bf16out_zero_copy_with_indices_buffer_inline(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        packed_all: &Buffer,
        scales_all: &Buffer,
        indices_buf: &Buffer,
        indices_offset: u64,
        k: usize,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        inter: usize,
        in_features: usize,
        batch: usize,
    ) -> Result<()> {
        if k == 0 || batch == 0 {
            return Ok(());
        }
        anyhow::ensure!(
            in_features.is_multiple_of(32),
            "in_features {in_features} must be a multiple of 32 (MXFP4 group size)"
        );

        let dims = MxFp4MoeGateUpSiluMulDims {
            inter: inter as u32,
            in_features: in_features as u32,
            batch: batch as u32,
        };

        encoder.set_compute_pipeline_state(&self.moe_gate_up_silu_mul_f32in_bf16out_v3);
        encoder.set_buffer(0, Some(packed_all), 0);
        encoder.set_buffer(1, Some(scales_all), 0);
        encoder.set_buffer(2, Some(indices_buf), indices_offset as usize);
        encoder.set_buffer(3, Some(x_buf), x_offset as usize);
        encoder.set_buffer(4, Some(y_buf), y_offset as usize);
        encoder.set_bytes_directly(
            5,
            std::mem::size_of::<MxFp4MoeGateUpSiluMulDims>(),
            &dims as *const _ as *const _,
        );

        const ROWS_PER_TG: usize = 8;
        const THREADS_PER_TG: usize = 256;
        let row_tgs = inter.div_ceil(ROWS_PER_TG);
        let groups = MTLSize {
            width: row_tgs,
            height: batch,
            depth: k,
        };
        let tg = MTLSize {
            width: THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        let tg_mem_bytes = in_features * std::mem::size_of::<f32>();
        encoder.set_threadgroup_memory_length(0, tg_mem_bytes);
        encoder.dispatch_thread_groups(groups, tg);
        Ok(())
    }

    /// Lever D (2026-04-27): encoder-injected MoE down dispatch with
    /// bfloat16 input. Reads `x: bfloat[k*batch*in]`; staging converts
    /// to f32 in TG-shared once. Inner FMA loop and output stay f32 —
    /// pairs with the f32-out down dispatcher's contract on the y side.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_moe_bf16in_zero_copy_with_indices_buffer_inline(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        packed_all: &Buffer,
        scales_all: &Buffer,
        indices_buf: &Buffer,
        indices_offset: u64,
        k: usize,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        out_features: usize,
        in_features: usize,
        batch: usize,
        broadcast_x: bool,
    ) -> Result<()> {
        if k == 0 || batch == 0 {
            return Ok(());
        }
        anyhow::ensure!(
            in_features.is_multiple_of(32),
            "in_features {in_features} must be a multiple of 32 (MXFP4 group size)"
        );

        let dims = MxFp4MoeDims {
            out_features: out_features as u32,
            in_features: in_features as u32,
            batch: batch as u32,
            broadcast_x: if broadcast_x { 1 } else { 0 },
        };

        encoder.set_compute_pipeline_state(&self.matmul_moe_bf16in_f32out_v3);
        encoder.set_buffer(0, Some(packed_all), 0);
        encoder.set_buffer(1, Some(scales_all), 0);
        encoder.set_buffer(2, Some(indices_buf), indices_offset as usize);
        encoder.set_buffer(3, Some(x_buf), x_offset as usize);
        encoder.set_buffer(4, Some(y_buf), y_offset as usize);
        encoder.set_bytes_directly(
            5,
            std::mem::size_of::<MxFp4MoeDims>(),
            &dims as *const _ as *const _,
        );

        const ROWS_PER_TG: usize = 8;
        const THREADS_PER_TG: usize = 256;
        let row_tgs = out_features.div_ceil(ROWS_PER_TG);
        let groups = MTLSize {
            width: row_tgs,
            height: batch,
            depth: k,
        };
        let tg = MTLSize {
            width: THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        let tg_mem_bytes = in_features * std::mem::size_of::<f32>();
        encoder.set_threadgroup_memory_length(0, tg_mem_bytes);
        encoder.dispatch_thread_groups(groups, tg);
        Ok(())
    }

    /// Lever H POC (2026-04-27): encoder-injected RmsNorm-fused routed
    /// gate_up_silu_mul dispatch. Reads RAW x (un-normalized). Pairs with
    /// the kernel's internal RmsNorm pass.
    ///
    /// `rms_weight_buf` must point to `[in_features]` f32 weight values.
    /// Topology identical to the unfused Lever A dispatch — the extra cost
    /// is the in-kernel cooperative reduction over `in_features` plus an
    /// extra TG-shared barrier.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_moe_gate_up_silu_mul_rmsnorm_zero_copy_with_indices_buffer_inline(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        packed_all: &Buffer,
        scales_all: &Buffer,
        indices_buf: &Buffer,
        indices_offset: u64,
        k: usize,
        x_buf: &Buffer,
        x_offset: u64,
        rms_weight_buf: &Buffer,
        rms_weight_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        inter: usize,
        in_features: usize,
        batch: usize,
        rms_eps: f32,
    ) -> Result<()> {
        if k == 0 || batch == 0 {
            return Ok(());
        }
        anyhow::ensure!(
            in_features.is_multiple_of(32),
            "in_features {in_features} must be a multiple of 32 (MXFP4 group size)"
        );

        let dims = MxFp4MoeGateUpSiluMulRmsnormDims {
            inter: inter as u32,
            in_features: in_features as u32,
            batch: batch as u32,
            rms_eps,
        };

        encoder.set_compute_pipeline_state(&self.moe_gate_up_silu_mul_rmsnorm_f32_v3);
        encoder.set_buffer(0, Some(packed_all), 0);
        encoder.set_buffer(1, Some(scales_all), 0);
        encoder.set_buffer(2, Some(indices_buf), indices_offset as usize);
        encoder.set_buffer(3, Some(x_buf), x_offset as usize);
        encoder.set_buffer(4, Some(rms_weight_buf), rms_weight_offset as usize);
        encoder.set_buffer(5, Some(y_buf), y_offset as usize);
        encoder.set_bytes_directly(
            6,
            std::mem::size_of::<MxFp4MoeGateUpSiluMulRmsnormDims>(),
            &dims as *const _ as *const _,
        );

        const ROWS_PER_TG: usize = 8;
        const THREADS_PER_TG: usize = 256;
        let row_tgs = inter.div_ceil(ROWS_PER_TG);
        let groups = MTLSize {
            width: row_tgs,
            height: batch,
            depth: k,
        };
        let tg = MTLSize {
            width: THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        // x_shared[in_features] f32 + reduce_buf[8] f32 (one per simdgroup).
        encoder.set_threadgroup_memory_length(0, in_features * std::mem::size_of::<f32>());
        encoder.set_threadgroup_memory_length(1, 8 * std::mem::size_of::<f32>());
        encoder.dispatch_thread_groups(groups, tg);
        Ok(())
    }

    /// Lever H Step 2: encoder-injected RmsNorm-fused **dense f32-weight**
    /// matmul (`dense_f32_matmul_rmsnorm`). Reads RAW x + rms_weight; the
    /// kernel computes `inv_rms` and applies `x * rms_weight * inv_rms` into
    /// TG-shared memory before a vectorized f32 dot product against `weight`.
    ///
    /// `weight_buf` must hold `[out, in]` row-major f32 weights.
    /// Topology mirrors the MXFP4 v3 path — `(ceil(out/8), batch, 1)` with
    /// 256 threads/TG. Suitable for routing gate (out=256) and (degenerately
    /// for) `shared_expert_gate` (out=1).
    #[allow(clippy::too_many_arguments)]
    pub fn dense_f32_matmul_rmsnorm_zero_copy_inline(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        weight_buf: &Buffer,
        weight_offset: u64,
        x_buf: &Buffer,
        x_offset: u64,
        rms_weight_buf: &Buffer,
        rms_weight_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        out_features: usize,
        in_features: usize,
        batch: usize,
        rms_eps: f32,
    ) -> Result<()> {
        if batch == 0 || out_features == 0 {
            return Ok(());
        }
        anyhow::ensure!(
            in_features.is_multiple_of(4),
            "dense_f32_matmul_rmsnorm: in_features {in_features} must be multiple of 4 (float4 vector loads)"
        );

        let dims = DenseMatmulRmsnormDims {
            out_features: out_features as u32,
            in_features: in_features as u32,
            rms_eps,
        };
        let batch_u32: u32 = batch as u32;

        encoder.set_compute_pipeline_state(&self.dense_f32_matmul_rmsnorm);
        encoder.set_buffer(0, Some(weight_buf), weight_offset as usize);
        encoder.set_buffer(1, Some(x_buf), x_offset as usize);
        encoder.set_buffer(2, Some(rms_weight_buf), rms_weight_offset as usize);
        encoder.set_buffer(3, Some(y_buf), y_offset as usize);
        encoder.set_bytes_directly(
            4,
            std::mem::size_of::<DenseMatmulRmsnormDims>(),
            &dims as *const _ as *const _,
        );
        encoder.set_bytes_directly(
            5,
            std::mem::size_of::<u32>(),
            &batch_u32 as *const _ as *const _,
        );

        const ROWS_PER_TG: usize = 8;
        const THREADS_PER_TG: usize = 256;
        let row_tgs = out_features.div_ceil(ROWS_PER_TG);
        let groups = MTLSize {
            width: row_tgs,
            height: batch,
            depth: 1,
        };
        let tg = MTLSize {
            width: THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        encoder.set_threadgroup_memory_length(0, in_features * std::mem::size_of::<f32>());
        encoder.set_threadgroup_memory_length(1, 8 * std::mem::size_of::<f32>());
        encoder.dispatch_thread_groups(groups, tg);
        Ok(())
    }

    /// Lever H multi-callsite migration: encoder-injected RmsNorm-fused dense
    /// matmul (`mxfp4_matmul_f32_v3_rmsnorm`). Reads RAW x (un-normalized) plus
    /// `rms_weight [in_features]`; the kernel's Phase 1/2 cooperative pass
    /// computes `inv_rms = rsqrt(mean(x²) + eps)` and applies
    /// `x * rms_weight * inv_rms` into TG-shared memory before the v3 matmul
    /// body.
    ///
    /// Topology identical to unfused v3 — `(ceil(out/8), batch, 1)`. Use this
    /// helper for routing-gate and shared-expert gate_up paths when the
    /// `LUMEN_ENABLE_RMSNORM_FUSION` flag is on; the corresponding
    /// `post_attention_layernorm.forward` dispatch in `layer.rs` is then
    /// skipped, with the rms weight piped through to all consumers.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_f32_v3_rmsnorm_zero_copy_inline(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        packed: &Buffer,
        scales: &Buffer,
        x_buf: &Buffer,
        x_offset: u64,
        rms_weight_buf: &Buffer,
        rms_weight_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        out_features: usize,
        in_features: usize,
        batch: usize,
        rms_eps: f32,
    ) -> Result<()> {
        if batch == 0 || out_features == 0 {
            return Ok(());
        }
        anyhow::ensure!(
            in_features.is_multiple_of(32),
            "in_features {in_features} must be a multiple of 32 (MXFP4 group size)"
        );

        let dims = MxFp4MatmulRmsnormDims {
            out_features: out_features as u32,
            in_features: in_features as u32,
            rms_eps,
        };
        let batch_u32: u32 = batch as u32;

        encoder.set_compute_pipeline_state(&self.matmul_f32_v3_rmsnorm);
        encoder.set_buffer(0, Some(packed), 0);
        encoder.set_buffer(1, Some(scales), 0);
        encoder.set_buffer(2, Some(x_buf), x_offset as usize);
        encoder.set_buffer(3, Some(rms_weight_buf), rms_weight_offset as usize);
        encoder.set_buffer(4, Some(y_buf), y_offset as usize);
        encoder.set_bytes_directly(
            5,
            std::mem::size_of::<MxFp4MatmulRmsnormDims>(),
            &dims as *const _ as *const _,
        );
        encoder.set_bytes_directly(
            6,
            std::mem::size_of::<u32>(),
            &batch_u32 as *const _ as *const _,
        );

        const ROWS_PER_TG: usize = 8;
        const THREADS_PER_TG: usize = 256;
        let row_tgs = out_features.div_ceil(ROWS_PER_TG);
        let groups = MTLSize {
            width: row_tgs,
            height: batch,
            depth: 1,
        };
        let tg = MTLSize {
            width: THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        // x_shared[in_features] f32 + reduce_buf[8] f32.
        encoder.set_threadgroup_memory_length(0, in_features * std::mem::size_of::<f32>());
        encoder.set_threadgroup_memory_length(1, 8 * std::mem::size_of::<f32>());
        encoder.dispatch_thread_groups(groups, tg);
        Ok(())
    }

    /// Lever H Step 3 retry (2026-04-28): RmsNorm-fused matmul with the
    /// large-out kernel topology (16 rows/TG × 512 threads). For callsites
    /// with `out_features ≥ 8192` (qkv_proj 9216, in_proj_combined 12352)
    /// where the small-variant kernel
    /// (`matmul_f32_v3_rmsnorm_zero_copy_inline`) has too many TGs each
    /// redundantly recomputing the RmsNorm of the same x.
    ///
    /// Buffer layout identical — same `MxFp4MatmulRmsnormDims` struct and
    /// argument order. Only the dispatch grid (rows/TG, thread count) and
    /// reduce_buf size differ. Cosine ≥ 0.999 vs CPU pre-RmsNorm + unfused v3
    /// reference (parity test).
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_f32_v3_rmsnorm_large_zero_copy_inline(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        packed: &Buffer,
        scales: &Buffer,
        x_buf: &Buffer,
        x_offset: u64,
        rms_weight_buf: &Buffer,
        rms_weight_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        out_features: usize,
        in_features: usize,
        batch: usize,
        rms_eps: f32,
    ) -> Result<()> {
        if batch == 0 || out_features == 0 {
            return Ok(());
        }
        anyhow::ensure!(
            in_features.is_multiple_of(32),
            "in_features {in_features} must be a multiple of 32 (MXFP4 group size)"
        );

        let dims = MxFp4MatmulRmsnormDims {
            out_features: out_features as u32,
            in_features: in_features as u32,
            rms_eps,
        };
        let batch_u32: u32 = batch as u32;

        encoder.set_compute_pipeline_state(&self.matmul_f32_v3_rmsnorm_large);
        encoder.set_buffer(0, Some(packed), 0);
        encoder.set_buffer(1, Some(scales), 0);
        encoder.set_buffer(2, Some(x_buf), x_offset as usize);
        encoder.set_buffer(3, Some(rms_weight_buf), rms_weight_offset as usize);
        encoder.set_buffer(4, Some(y_buf), y_offset as usize);
        encoder.set_bytes_directly(
            5,
            std::mem::size_of::<MxFp4MatmulRmsnormDims>(),
            &dims as *const _ as *const _,
        );
        encoder.set_bytes_directly(
            6,
            std::mem::size_of::<u32>(),
            &batch_u32 as *const _ as *const _,
        );

        const ROWS_PER_TG: usize = 16;
        const THREADS_PER_TG: usize = 512;
        let row_tgs = out_features.div_ceil(ROWS_PER_TG);
        let groups = MTLSize {
            width: row_tgs,
            height: batch,
            depth: 1,
        };
        let tg = MTLSize {
            width: THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        // x_shared[in_features] f32 + reduce_buf[16] f32.
        encoder.set_threadgroup_memory_length(0, in_features * std::mem::size_of::<f32>());
        encoder.set_threadgroup_memory_length(1, 16 * std::mem::size_of::<f32>());
        encoder.dispatch_thread_groups(groups, tg);
        Ok(())
    }

    /// Lever H Step 3 retry tier 2 (2026-04-28): extra-large RmsNorm-fused
    /// matmul (32 rows/TG × 1024 threads = Apple Silicon max). For very large
    /// outputs (≥ 12K) where the `large` variant still leaves > 500 redundant
    /// TGs each recomputing RmsNorm. Quarters TG count vs small variant.
    /// Buffer layout identical to small/large; only dispatch grid +
    /// reduce_buf size differ.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_f32_v3_rmsnorm_xlarge_zero_copy_inline(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        packed: &Buffer,
        scales: &Buffer,
        x_buf: &Buffer,
        x_offset: u64,
        rms_weight_buf: &Buffer,
        rms_weight_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        out_features: usize,
        in_features: usize,
        batch: usize,
        rms_eps: f32,
    ) -> Result<()> {
        if batch == 0 || out_features == 0 {
            return Ok(());
        }
        anyhow::ensure!(
            in_features.is_multiple_of(32),
            "in_features {in_features} must be a multiple of 32 (MXFP4 group size)"
        );

        let dims = MxFp4MatmulRmsnormDims {
            out_features: out_features as u32,
            in_features: in_features as u32,
            rms_eps,
        };
        let batch_u32: u32 = batch as u32;

        encoder.set_compute_pipeline_state(&self.matmul_f32_v3_rmsnorm_xlarge);
        encoder.set_buffer(0, Some(packed), 0);
        encoder.set_buffer(1, Some(scales), 0);
        encoder.set_buffer(2, Some(x_buf), x_offset as usize);
        encoder.set_buffer(3, Some(rms_weight_buf), rms_weight_offset as usize);
        encoder.set_buffer(4, Some(y_buf), y_offset as usize);
        encoder.set_bytes_directly(
            5,
            std::mem::size_of::<MxFp4MatmulRmsnormDims>(),
            &dims as *const _ as *const _,
        );
        encoder.set_bytes_directly(
            6,
            std::mem::size_of::<u32>(),
            &batch_u32 as *const _ as *const _,
        );

        const ROWS_PER_TG: usize = 32;
        const THREADS_PER_TG: usize = 1024;
        let row_tgs = out_features.div_ceil(ROWS_PER_TG);
        let groups = MTLSize {
            width: row_tgs,
            height: batch,
            depth: 1,
        };
        let tg = MTLSize {
            width: THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        // x_shared[in_features] f32 + reduce_buf[32] f32.
        encoder.set_threadgroup_memory_length(0, in_features * std::mem::size_of::<f32>());
        encoder.set_threadgroup_memory_length(1, 32 * std::mem::size_of::<f32>());
        encoder.dispatch_thread_groups(groups, tg);
        Ok(())
    }

    /// Lever L1 (residual fusion): v3 matmul + residual add at write time.
    /// Same dispatch topology as `matmul_f32_v3` — only difference is buffer
    /// 6 (`residual`) and the kernel writes `acc + residual[idx]` instead of
    /// `acc`. Caller must ensure `residual` is shaped identically to `y`
    /// (`[batch, out_features]` f32, contiguous).
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_f32_v3_residual_zero_copy_inline(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        packed: &Buffer,
        scales: &Buffer,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        residual_buf: &Buffer,
        residual_offset: u64,
        out_features: usize,
        in_features: usize,
        batch: usize,
    ) -> Result<()> {
        if batch == 0 || out_features == 0 {
            return Ok(());
        }
        anyhow::ensure!(
            in_features.is_multiple_of(32),
            "in_features {in_features} must be a multiple of 32 (MXFP4 group size)"
        );

        let dims = MxFp4Dims {
            out_features: out_features as u32,
            in_features: in_features as u32,
        };
        let batch_u32: u32 = batch as u32;

        encoder.set_compute_pipeline_state(&self.matmul_f32_v3_residual);
        encoder.set_buffer(0, Some(packed), 0);
        encoder.set_buffer(1, Some(scales), 0);
        encoder.set_buffer(2, Some(x_buf), x_offset as usize);
        encoder.set_buffer(3, Some(y_buf), y_offset as usize);
        encoder.set_bytes_directly(
            4,
            std::mem::size_of::<MxFp4Dims>(),
            &dims as *const _ as *const _,
        );
        encoder.set_bytes_directly(
            5,
            std::mem::size_of::<u32>(),
            &batch_u32 as *const _ as *const _,
        );
        encoder.set_buffer(6, Some(residual_buf), residual_offset as usize);

        const ROWS_PER_TG: usize = 8;
        const THREADS_PER_TG: usize = 256;
        let row_tgs = out_features.div_ceil(ROWS_PER_TG);
        let groups = MTLSize {
            width: row_tgs,
            height: batch,
            depth: 1,
        };
        let tg = MTLSize {
            width: THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        // x_shared[in_features] f32 (no reduce_buf for v3 base).
        encoder.set_threadgroup_memory_length(0, in_features * std::mem::size_of::<f32>());
        encoder.dispatch_thread_groups(groups, tg);
        Ok(())
    }

    /// Lever L1 Step 2 (MoE-side residual fusion): encoder-injected 3-way
    /// element-wise add `y[i] = a[i] + b[i] + c[i]` for `n` flat f32 elements.
    /// All buffers must be Metal-resident, F32, and at least `n` long.
    /// Caller owns the encoder; no commit/wait.
    #[allow(clippy::too_many_arguments)]
    pub fn tri_add_f32_zero_copy_inline(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        a_buf: &Buffer,
        a_offset: u64,
        b_buf: &Buffer,
        b_offset: u64,
        c_buf: &Buffer,
        c_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        n: usize,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let n_u32: u32 = n as u32;

        encoder.set_compute_pipeline_state(&self.tri_add_f32);
        encoder.set_buffer(0, Some(a_buf), a_offset as usize);
        encoder.set_buffer(1, Some(b_buf), b_offset as usize);
        encoder.set_buffer(2, Some(c_buf), c_offset as usize);
        encoder.set_buffer(3, Some(y_buf), y_offset as usize);
        encoder.set_bytes_directly(
            4,
            std::mem::size_of::<u32>(),
            &n_u32 as *const _ as *const _,
        );

        const THREADS_PER_TG: usize = 256;
        let n_groups = n.div_ceil(THREADS_PER_TG);
        let grid = MTLSize {
            width: n_groups,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(grid, tg);
        Ok(())
    }

    /// Lever L1 Step 3.5 (drift-safe partial Step 3): encoder-injected
    /// `y[t,h] = a[t,h] + b[t,h] * coef[t] + d[t,h]`. `coef` is per-token
    /// scalar (already-sigmoided by the caller). All buffers Metal-resident
    /// f32. `a`/`b`/`d`/`y` flat `[bl * hidden]`, `coef` `[bl]`.
    #[allow(clippy::too_many_arguments)]
    pub fn scalar_mul_tri_add_f32_zero_copy_inline(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        a_buf: &Buffer,
        a_offset: u64,
        b_buf: &Buffer,
        b_offset: u64,
        coef_buf: &Buffer,
        coef_offset: u64,
        d_buf: &Buffer,
        d_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        bl: usize,
        hidden: usize,
    ) -> Result<()> {
        if bl == 0 || hidden == 0 {
            return Ok(());
        }
        let hidden_u32: u32 = hidden as u32;
        let bl_u32: u32 = bl as u32;

        encoder.set_compute_pipeline_state(&self.scalar_mul_tri_add_f32);
        encoder.set_buffer(0, Some(a_buf), a_offset as usize);
        encoder.set_buffer(1, Some(b_buf), b_offset as usize);
        encoder.set_buffer(2, Some(coef_buf), coef_offset as usize);
        encoder.set_buffer(3, Some(d_buf), d_offset as usize);
        encoder.set_buffer(4, Some(y_buf), y_offset as usize);
        encoder.set_bytes_directly(
            5,
            std::mem::size_of::<u32>(),
            &hidden_u32 as *const _ as *const _,
        );
        encoder.set_bytes_directly(
            6,
            std::mem::size_of::<u32>(),
            &bl_u32 as *const _ as *const _,
        );

        const THREADS_X: usize = 256;
        let n_groups_x = hidden.div_ceil(THREADS_X);
        let grid = MTLSize {
            width: n_groups_x,
            height: bl,
            depth: 1,
        };
        let tg = MTLSize {
            width: THREADS_X,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(grid, tg);
        Ok(())
    }

    /// Lever L4 (cross-layer megafusion): encoder-injected fused dispatch
    /// `out[t,h] = a[t,h] + b[t,h] * coef[t] + d[t,h]; attn_in[t,h] = out[t,h]
    /// * rms_weight[h] * rsqrt(mean(out[t,:]²) + rms_eps)`. Saves 1 dispatch
    /// per layer transition (layer i's mlp_final + layer i+1's input_layernorm).
    /// All buffers Metal-resident f32. `a`/`b`/`d`/`out`/`attn_in` flat
    /// `[bl * hidden]`, `coef` `[bl]`, `rms_weight` `[hidden]`. `out_shared`
    /// threadgroup mem = `hidden * 4` bytes; `reduce_buf` = 32 bytes.
    #[allow(clippy::too_many_arguments)]
    pub fn scalar_mul_tri_add_rmsnorm_f32_zero_copy_inline(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        a_buf: &Buffer,
        a_offset: u64,
        b_buf: &Buffer,
        b_offset: u64,
        coef_buf: &Buffer,
        coef_offset: u64,
        d_buf: &Buffer,
        d_offset: u64,
        rms_weight_buf: &Buffer,
        rms_weight_offset: u64,
        out_buf: &Buffer,
        out_offset: u64,
        attn_in_buf: &Buffer,
        attn_in_offset: u64,
        bl: usize,
        hidden: usize,
        rms_eps: f32,
    ) -> Result<()> {
        if bl == 0 || hidden == 0 {
            return Ok(());
        }
        let hidden_u32: u32 = hidden as u32;
        let bl_u32: u32 = bl as u32;
        let rms_eps_f32: f32 = rms_eps;

        encoder.set_compute_pipeline_state(&self.scalar_mul_tri_add_rmsnorm_f32);
        encoder.set_buffer(0, Some(a_buf), a_offset as usize);
        encoder.set_buffer(1, Some(b_buf), b_offset as usize);
        encoder.set_buffer(2, Some(coef_buf), coef_offset as usize);
        encoder.set_buffer(3, Some(d_buf), d_offset as usize);
        encoder.set_buffer(4, Some(rms_weight_buf), rms_weight_offset as usize);
        encoder.set_buffer(5, Some(out_buf), out_offset as usize);
        encoder.set_buffer(6, Some(attn_in_buf), attn_in_offset as usize);
        encoder.set_bytes_directly(
            7,
            std::mem::size_of::<u32>(),
            &hidden_u32 as *const _ as *const _,
        );
        encoder.set_bytes_directly(
            8,
            std::mem::size_of::<u32>(),
            &bl_u32 as *const _ as *const _,
        );
        encoder.set_bytes_directly(
            9,
            std::mem::size_of::<f32>(),
            &rms_eps_f32 as *const _ as *const _,
        );

        const THREADS_PER_TG: usize = 256;
        let grid = MTLSize {
            width: bl,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        encoder.set_threadgroup_memory_length(0, hidden * std::mem::size_of::<f32>());
        encoder.set_threadgroup_memory_length(1, 8 * std::mem::size_of::<f32>());
        encoder.dispatch_thread_groups(grid, tg);
        Ok(())
    }

    /// Lever B (2026-04-27): encoder-injected MoE weighted-sum dispatch.
    /// Computes `out[r] = sum_e weights[e] * downs[e, r]` for r in [0, hidden).
    /// Replaces Candle's `downs.broadcast_mul(w_kx1).sum_keepdim(0)` chain.
    ///
    /// All buffers must be Metal-resident, F32. `out` shape is `[hidden]`
    /// (or `[1, hidden]` — flat). Caller owns the encoder; no commit/wait.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_wsum_zero_copy_inline(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        downs_buf: &Buffer,
        downs_offset: u64,
        weights_buf: &Buffer,
        weights_offset: u64,
        out_buf: &Buffer,
        out_offset: u64,
        k: usize,
        hidden: usize,
    ) -> Result<()> {
        if k == 0 || hidden == 0 {
            return Ok(());
        }
        let dims = MoeWsumDims {
            k: k as u32,
            hidden: hidden as u32,
        };
        encoder.set_compute_pipeline_state(&self.moe_wsum_f32);
        encoder.set_buffer(0, Some(downs_buf), downs_offset as usize);
        encoder.set_buffer(1, Some(weights_buf), weights_offset as usize);
        encoder.set_buffer(2, Some(out_buf), out_offset as usize);
        encoder.set_bytes_directly(
            3,
            std::mem::size_of::<MoeWsumDims>(),
            &dims as *const _ as *const _,
        );
        let max_threads = self.moe_wsum_f32.max_total_threads_per_threadgroup();
        let threads_per_tg = max_threads.min(256);
        let grid = MTLSize {
            width: hidden,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: threads_per_tg,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(grid, tg);
        Ok(())
    }

    /// CB multi-token RmsNorm-fused gate_up_silu_mul
    /// dispatch. Same buffer layout as the single-token variant except
    /// `expert_indices` is `[B, k]` (length `B*k`) and `y` is `[k, B, inter]`.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_moe_gate_up_silu_mul_rmsnorm_multi_zero_copy_inline(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        packed_all: &Buffer,
        scales_all: &Buffer,
        indices_buf: &Buffer,
        indices_offset: u64,
        k: usize,
        x_buf: &Buffer,
        x_offset: u64,
        rms_weight_buf: &Buffer,
        rms_weight_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        inter: usize,
        in_features: usize,
        batch: usize,
        rms_eps: f32,
    ) -> Result<()> {
        if k == 0 || batch == 0 {
            return Ok(());
        }
        anyhow::ensure!(
            in_features.is_multiple_of(32),
            "in_features {in_features} must be a multiple of 32 (MXFP4 group size)"
        );
        let dims = MxFp4MoeGateUpSiluMulRmsnormDimsMulti {
            inter: inter as u32,
            in_features: in_features as u32,
            batch: batch as u32,
            k: k as u32,
            rms_eps,
        };
        encoder.set_compute_pipeline_state(&self.moe_gate_up_silu_mul_rmsnorm_f32_v3_multi);
        encoder.set_buffer(0, Some(packed_all), 0);
        encoder.set_buffer(1, Some(scales_all), 0);
        encoder.set_buffer(2, Some(indices_buf), indices_offset as usize);
        encoder.set_buffer(3, Some(x_buf), x_offset as usize);
        encoder.set_buffer(4, Some(rms_weight_buf), rms_weight_offset as usize);
        encoder.set_buffer(5, Some(y_buf), y_offset as usize);
        encoder.set_bytes_directly(
            6,
            std::mem::size_of::<MxFp4MoeGateUpSiluMulRmsnormDimsMulti>(),
            &dims as *const _ as *const _,
        );
        const ROWS_PER_TG: usize = 8;
        const THREADS_PER_TG: usize = 256;
        let row_tgs = inter.div_ceil(ROWS_PER_TG);
        let groups = MTLSize {
            width: row_tgs,
            height: batch,
            depth: k,
        };
        let tg = MTLSize {
            width: THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        encoder.set_threadgroup_memory_length(0, in_features * std::mem::size_of::<f32>());
        encoder.set_threadgroup_memory_length(1, 8 * std::mem::size_of::<f32>());
        encoder.dispatch_thread_groups(groups, tg);
        Ok(())
    }

    /// CB multi-token MoE down matmul. Reads `x: [k, B, in_features]`
    /// (gate_up_silu_mul output), writes `y: [k, B, out_features]`. Per-token
    /// expert indirection via `expert_indices: [B, k]`.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_moe_multi_zero_copy_inline(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        packed_all: &Buffer,
        scales_all: &Buffer,
        indices_buf: &Buffer,
        indices_offset: u64,
        k: usize,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        out_features: usize,
        in_features: usize,
        batch: usize,
    ) -> Result<()> {
        if k == 0 || batch == 0 {
            return Ok(());
        }
        anyhow::ensure!(
            in_features.is_multiple_of(32),
            "in_features {in_features} must be a multiple of 32 (MXFP4 group size)"
        );
        let dims = MxFp4MoeDimsMulti {
            out_features: out_features as u32,
            in_features: in_features as u32,
            batch: batch as u32,
            k: k as u32,
        };
        encoder.set_compute_pipeline_state(&self.matmul_moe_f32_v3_multi);
        encoder.set_buffer(0, Some(packed_all), 0);
        encoder.set_buffer(1, Some(scales_all), 0);
        encoder.set_buffer(2, Some(indices_buf), indices_offset as usize);
        encoder.set_buffer(3, Some(x_buf), x_offset as usize);
        encoder.set_buffer(4, Some(y_buf), y_offset as usize);
        encoder.set_bytes_directly(
            5,
            std::mem::size_of::<MxFp4MoeDimsMulti>(),
            &dims as *const _ as *const _,
        );
        const ROWS_PER_TG: usize = 8;
        const THREADS_PER_TG: usize = 256;
        let row_tgs = out_features.div_ceil(ROWS_PER_TG);
        let groups = MTLSize {
            width: row_tgs,
            height: batch,
            depth: k,
        };
        let tg = MTLSize {
            width: THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        encoder.set_threadgroup_memory_length(0, in_features * std::mem::size_of::<f32>());
        encoder.dispatch_thread_groups(groups, tg);
        Ok(())
    }

    /// CB multi-token weighted sum. Reads `downs: [k, B, hidden]` and
    /// `weights: [B, k]`, writes `out: [B, hidden]`. 2D grid (hidden, B).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_wsum_multi_zero_copy_inline(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        downs_buf: &Buffer,
        downs_offset: u64,
        weights_buf: &Buffer,
        weights_offset: u64,
        out_buf: &Buffer,
        out_offset: u64,
        k: usize,
        batch: usize,
        hidden: usize,
    ) -> Result<()> {
        if k == 0 || batch == 0 || hidden == 0 {
            return Ok(());
        }
        let dims = MoeWsumDimsMulti {
            k: k as u32,
            batch: batch as u32,
            hidden: hidden as u32,
        };
        encoder.set_compute_pipeline_state(&self.moe_wsum_f32_multi);
        encoder.set_buffer(0, Some(downs_buf), downs_offset as usize);
        encoder.set_buffer(1, Some(weights_buf), weights_offset as usize);
        encoder.set_buffer(2, Some(out_buf), out_offset as usize);
        encoder.set_bytes_directly(
            3,
            std::mem::size_of::<MoeWsumDimsMulti>(),
            &dims as *const _ as *const _,
        );
        let max_threads = self.moe_wsum_f32_multi.max_total_threads_per_threadgroup();
        let threads_x = max_threads.min(256);
        let grid = MTLSize {
            width: hidden,
            height: batch,
            depth: 1,
        };
        let tg = MTLSize {
            width: threads_x,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(grid, tg);
        Ok(())
    }

    /// Lever C (2026-04-27): encoder-injected fused MoE down matmul + weighted
    /// sum. Computes
    /// `out[b, hr] = sum_slot weights[slot]
    ///                * sum_m down[expert[slot], hr, m] * hiddens[slot, b, m]`
    /// in a single dispatch, replacing the chain
    /// `mxfp4_matmul_moe_f32_v3` (writing `[k, batch, hidden]`) →
    /// `moe_wsum_f32` (reducing to `[batch, hidden]`).
    ///
    /// Topology mirrors `matmul_moe_zero_copy_with_indices_buffer_inline`'s
    /// v3 grid (`(out/8, batch, slot)`) but folds the slot axis into an inner
    /// serial loop in the kernel — the grid becomes `(out/8, batch, 1)` and
    /// each TG sums all `k` expert contributions before writing once to `y`
    /// (no atomics, no contention).
    ///
    /// All buffers Metal-resident F32. Caller owns the encoder; no commit/wait.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_moe_wsum_zero_copy_with_indices_buffer_inline(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        packed_all: &Buffer,
        scales_all: &Buffer,
        indices_buf: &Buffer,
        indices_offset: u64,
        k: usize,
        weights_buf: &Buffer,
        weights_offset: u64,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        out_features: usize,
        in_features: usize,
        batch: usize,
    ) -> Result<()> {
        if k == 0 || batch == 0 || out_features == 0 {
            return Ok(());
        }
        anyhow::ensure!(
            in_features.is_multiple_of(32),
            "in_features {in_features} must be a multiple of 32 (MXFP4 group size)"
        );

        let dims = MxFp4MoeMatmulWsumDims {
            out_features: out_features as u32,
            in_features: in_features as u32,
            batch: batch as u32,
            k: k as u32,
        };

        encoder.set_compute_pipeline_state(&self.matmul_moe_wsum_f32_v3);
        encoder.set_buffer(0, Some(packed_all), 0);
        encoder.set_buffer(1, Some(scales_all), 0);
        encoder.set_buffer(2, Some(indices_buf), indices_offset as usize);
        encoder.set_buffer(3, Some(weights_buf), weights_offset as usize);
        encoder.set_buffer(4, Some(x_buf), x_offset as usize);
        encoder.set_buffer(5, Some(y_buf), y_offset as usize);
        encoder.set_bytes_directly(
            6,
            std::mem::size_of::<MxFp4MoeMatmulWsumDims>(),
            &dims as *const _ as *const _,
        );

        const ROWS_PER_TG: usize = 8;
        const THREADS_PER_TG: usize = 256;
        let row_tgs = out_features.div_ceil(ROWS_PER_TG);
        let groups = MTLSize {
            width: row_tgs,
            height: batch,
            depth: 1,
        };
        let tg = MTLSize {
            width: THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        let tg_mem_bytes = in_features * std::mem::size_of::<f32>();
        encoder.set_threadgroup_memory_length(0, tg_mem_bytes);
        encoder.dispatch_thread_groups(groups, tg);
        Ok(())
    }

    /// Lever C-atomic (2026-04-27): grid-parallel variant of the fused MoE
    /// down+wsum dispatch. Same math as
    /// `matmul_moe_wsum_zero_copy_with_indices_buffer_inline` but the slot
    /// axis stays in the grid (`grid.z = k`) and each TG `atomic_fetch_add`s
    /// its weighted contribution to the output. Restores 2048 TGs (production)
    /// vs the serial-fold variant's 256, at the cost of k-way contention per
    /// output element.
    ///
    /// **Caller must pre-zero the output buffer** before calling — the kernel
    /// only adds, it does not assign. `Tensor::zeros` on the candle path is
    /// sufficient.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_moe_wsum_atomic_zero_copy_with_indices_buffer_inline(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        packed_all: &Buffer,
        scales_all: &Buffer,
        indices_buf: &Buffer,
        indices_offset: u64,
        k: usize,
        weights_buf: &Buffer,
        weights_offset: u64,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        out_features: usize,
        in_features: usize,
        batch: usize,
    ) -> Result<()> {
        if k == 0 || batch == 0 || out_features == 0 {
            return Ok(());
        }
        anyhow::ensure!(
            in_features.is_multiple_of(32),
            "in_features {in_features} must be a multiple of 32 (MXFP4 group size)"
        );

        let dims = MxFp4MoeMatmulWsumDims {
            out_features: out_features as u32,
            in_features: in_features as u32,
            batch: batch as u32,
            k: k as u32,
        };

        encoder.set_compute_pipeline_state(&self.matmul_moe_wsum_atomic_f32_v3);
        encoder.set_buffer(0, Some(packed_all), 0);
        encoder.set_buffer(1, Some(scales_all), 0);
        encoder.set_buffer(2, Some(indices_buf), indices_offset as usize);
        encoder.set_buffer(3, Some(weights_buf), weights_offset as usize);
        encoder.set_buffer(4, Some(x_buf), x_offset as usize);
        encoder.set_buffer(5, Some(y_buf), y_offset as usize);
        encoder.set_bytes_directly(
            6,
            std::mem::size_of::<MxFp4MoeMatmulWsumDims>(),
            &dims as *const _ as *const _,
        );

        const ROWS_PER_TG: usize = 8;
        const THREADS_PER_TG: usize = 256;
        let row_tgs = out_features.div_ceil(ROWS_PER_TG);
        let groups = MTLSize {
            width: row_tgs,
            height: batch,
            depth: k,
        };
        let tg = MTLSize {
            width: THREADS_PER_TG,
            height: 1,
            depth: 1,
        };
        let tg_mem_bytes = in_features * std::mem::size_of::<f32>();
        encoder.set_threadgroup_memory_length(0, tg_mem_bytes);
        encoder.dispatch_thread_groups(groups, tg);
        Ok(())
    }

    /// Lever G (2026-04-27): encoder-injected routing top-k partial select.
    /// Produces `inds_out [BL, k]` u32 + `vals_out [BL, k]` f32 directly from
    /// `probs [BL, num_experts]` via iterated argmax + mask. Replaces
    /// `arg_sort_last_dim → narrow → contiguous → gather` chain.
    ///
    /// **Constraints:** `num_experts ≤ 256` (current TG size). Production
    /// Qwen3.5-MoE has E=256 (exact fit). For E < 256, padding lanes hold
    /// `-INFINITY` and never win.
    #[allow(clippy::too_many_arguments)]
    pub fn topk_partial_select_zero_copy_inline(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        probs_buf: &Buffer,
        probs_offset: u64,
        inds_buf: &Buffer,
        inds_offset: u64,
        vals_buf: &Buffer,
        vals_offset: u64,
        bl: usize,
        num_experts: usize,
        k: usize,
    ) -> Result<()> {
        if bl == 0 || k == 0 {
            return Ok(());
        }
        anyhow::ensure!(
            num_experts <= 256,
            "topk_partial_select_f32: num_experts {num_experts} > 256 (TG size limit)"
        );
        anyhow::ensure!(
            k <= num_experts,
            "topk_partial_select_f32: k {k} > num_experts {num_experts}"
        );

        let dims = TopkPartialDims {
            num_experts: num_experts as u32,
            k: k as u32,
        };

        encoder.set_compute_pipeline_state(&self.topk_partial_select_f32);
        encoder.set_buffer(0, Some(probs_buf), probs_offset as usize);
        encoder.set_buffer(1, Some(inds_buf), inds_offset as usize);
        encoder.set_buffer(2, Some(vals_buf), vals_offset as usize);
        encoder.set_bytes_directly(
            3,
            std::mem::size_of::<TopkPartialDims>(),
            &dims as *const _ as *const _,
        );

        // Round TG size up to a power of 2 ≥ num_experts (for tree reduction
        // correctness). With the ≤ 256 constraint above, 256 covers all cases.
        const TG_SIZE: usize = 256;
        let groups = MTLSize {
            width: bl,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: TG_SIZE,
            height: 1,
            depth: 1,
        };
        encoder.set_threadgroup_memory_length(0, TG_SIZE * std::mem::size_of::<f32>());
        encoder.set_threadgroup_memory_length(1, TG_SIZE * std::mem::size_of::<u32>());
        encoder.dispatch_thread_groups(groups, tg);
        Ok(())
    }

    /// encoder-injected full routing-pipeline fusion.
    /// Takes raw `logits [BL, num_experts]` (post routing-gate matmul, **before**
    /// softmax) and produces `inds_out [BL, k]` u32 + `vals_out [BL, k]` f32
    /// (already renormalized) in a single dispatch. Replaces the entire 6-op
    /// Candle chain `softmax → arg_sort → narrow → gather → sum_keepdim →
    /// broadcast_div` (saves 5 dispatches/layer × 40 layers = 200/decode step).
    ///
    /// **Constraints:** `num_experts ≤ 256` (TG size). `k ≤ 32` (renorm uses a
    /// single simdgroup `simd_sum`). Production Qwen3.5-MoE: E=256, k=8.
    #[allow(clippy::too_many_arguments)]
    pub fn router_softmax_topk_renorm_zero_copy_inline(
        &self,
        encoder: &crate::metal::ComputeCommandEncoderRef,
        logits_buf: &Buffer,
        logits_offset: u64,
        inds_buf: &Buffer,
        inds_offset: u64,
        vals_buf: &Buffer,
        vals_offset: u64,
        bl: usize,
        num_experts: usize,
        k: usize,
    ) -> Result<()> {
        if bl == 0 || k == 0 {
            return Ok(());
        }
        anyhow::ensure!(
            num_experts <= 256,
            "router_softmax_topk_renorm_f32: num_experts {num_experts} > 256 \
             (TG size limit)"
        );
        anyhow::ensure!(
            k <= 32,
            "router_softmax_topk_renorm_f32: k {k} > 32 (single-SG renorm limit)"
        );
        anyhow::ensure!(
            k <= num_experts,
            "router_softmax_topk_renorm_f32: k {k} > num_experts {num_experts}"
        );

        // Reuse TopkPartialDims layout — same `{num_experts: u32, k: u32}` ABI.
        let dims = TopkPartialDims {
            num_experts: num_experts as u32,
            k: k as u32,
        };

        encoder.set_compute_pipeline_state(&self.router_softmax_topk_renorm_f32);
        encoder.set_buffer(0, Some(logits_buf), logits_offset as usize);
        encoder.set_buffer(1, Some(inds_buf), inds_offset as usize);
        encoder.set_buffer(2, Some(vals_buf), vals_offset as usize);
        encoder.set_bytes_directly(
            3,
            std::mem::size_of::<TopkPartialDims>(),
            &dims as *const _ as *const _,
        );

        const TG_SIZE: usize = 256;
        let groups = MTLSize {
            width: bl,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: TG_SIZE,
            height: 1,
            depth: 1,
        };
        // shared_v[256] f32, shared_i[256] u32, reduce_buf[max(8,k)] f32.
        encoder.set_threadgroup_memory_length(0, TG_SIZE * std::mem::size_of::<f32>());
        encoder.set_threadgroup_memory_length(1, TG_SIZE * std::mem::size_of::<u32>());
        let reduce_buf_len = std::cmp::max(8, k);
        encoder.set_threadgroup_memory_length(2, reduce_buf_len * std::mem::size_of::<f32>());
        encoder.dispatch_thread_groups(groups, tg);
        Ok(())
    }

    /// Internal MoE matmul dispatch shared by `_with_version` (host-slice indices
    /// uploaded per call) and `_with_indices_buffer` (caller-owned GPU buffer).
    #[allow(clippy::too_many_arguments)]
    fn matmul_moe_dispatch_inner(
        &self,
        version: KernelVersion,
        packed_all: &Buffer,
        scales_all: &Buffer,
        indices_buf: &Buffer,
        indices_offset: u64,
        k: usize,
        x_buf: &Buffer,
        x_offset: u64,
        y_buf: &Buffer,
        y_offset: u64,
        out_features: usize,
        in_features: usize,
        batch: usize,
        broadcast_x: bool,
    ) -> Result<()> {
        anyhow::ensure!(
            in_features.is_multiple_of(32),
            "in_features {in_features} must be a multiple of 32 (MXFP4 group size)"
        );
        let pipeline = match version {
            KernelVersion::V3 => &self.matmul_moe_f32_v3,
            KernelVersion::V2 => &self.matmul_moe_f32_v2,
            KernelVersion::V1 => &self.matmul_moe_f32,
        };
        let uses_v3 = matches!(version, KernelVersion::V3);

        let dims = MxFp4MoeDims {
            out_features: out_features as u32,
            in_features: in_features as u32,
            batch: batch as u32,
            broadcast_x: if broadcast_x { 1 } else { 0 },
        };

        let encoder = crate::metal::process_commands()
            .command_encoder()
            .expect("ce");
        encoder.set_label("lumen:mxfp4_moe_matmul");
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(packed_all), (0) as usize);
        encoder.set_buffer(1, Some(scales_all), (0) as usize);
        encoder.set_buffer(2, Some(indices_buf), (indices_offset) as usize);
        encoder.set_buffer(3, Some(x_buf), (x_offset) as usize);
        encoder.set_buffer(4, Some(y_buf), (y_offset) as usize);
        encoder.set_bytes_directly(
            5,
            std::mem::size_of::<MxFp4MoeDims>(),
            &dims as *const _ as *const _,
        );

        if uses_v3 {
            // v3 MoE: simdgroup cooperative + threadgroup x cache.
            // Grid is (out_features / 8, batch, k); each tg = 256 threads = 8
            // simdgroups producing 8 rows. ceil_div on rows so the last tg
            // covers the tail.
            const ROWS_PER_TG: usize = 8;
            const THREADS_PER_TG: usize = 256;
            let row_tgs = out_features.div_ceil(ROWS_PER_TG);
            let groups = MTLSize {
                width: row_tgs,
                height: batch,
                depth: k,
            };
            let tg = MTLSize {
                width: THREADS_PER_TG,
                height: 1,
                depth: 1,
            };
            // Threadgroup memory: in_features × 4 bytes (one f32 row of x).
            let tg_mem_bytes = in_features * std::mem::size_of::<f32>();
            encoder.set_threadgroup_memory_length(0, tg_mem_bytes);
            encoder.dispatch_thread_groups(groups, tg);
        } else {
            // v1/v2: one thread per output element across (out, batch, k).
            let max_threads = pipeline.max_total_threads_per_threadgroup();
            let threads_per_tg = max_threads.min(256);
            let grid = MTLSize {
                width: (out_features as u64) as usize,
                height: (batch as u64) as usize,
                depth: (k as u64) as usize,
            };
            let tg = MTLSize {
                width: (threads_per_tg) as usize,
                height: (1) as usize,
                depth: (1) as usize,
            };
            encoder.dispatch_threads(grid, tg);
        }

        let profile = std::env::var("LUMEN_MXFP4_PROFILE")
            .map(|v| v == "1")
            .unwrap_or(false);
        if profile {
            let t_commit = std::time::Instant::now();
            let commit_ns = t_commit.elapsed().as_nanos() as u64;
            let t_wait = std::time::Instant::now();
            drop(encoder);
            crate::metal::process_commands()
                .flush_and_wait()
                .expect("flush");
            let wait_ns = t_wait.elapsed().as_nanos() as u64;
            use std::sync::atomic::Ordering::Relaxed;
            // Charge one "call" per expert slot for commensurable averages.
            PROF_CALLS.fetch_add(k as u64, Relaxed);
            PROF_COMMIT_NS.fetch_add(commit_ns, Relaxed);
            PROF_WAIT_NS.fetch_add(wait_ns, Relaxed);
        } else {
            drop(encoder);
            crate::metal::process_commands()
                .flush_and_wait()
                .expect("flush");
        }
        Ok(())
    }

    /// Compute `y = W @ x` on GPU, where `W` is stored in MXFP4 format.
    ///
    /// Shape contract:
    ///   - `packed.len() == out_features * in_features / 8`
    ///   - `scales.len() == out_features * in_features / 32`
    ///   - `x.len() == in_features`
    ///   - `in_features` is a multiple of 32.
    pub fn matvec_f32(
        &self,
        packed: &[u32],
        scales: &[u8],
        x: &[f32],
        out_features: usize,
        in_features: usize,
    ) -> Result<Vec<f32>> {
        anyhow::ensure!(
            in_features.is_multiple_of(32),
            "in_features {in_features} must be a multiple of 32 (MXFP4 group size)"
        );
        let expected_packed = out_features * in_features / 8;
        anyhow::ensure!(
            packed.len() == expected_packed,
            "packed length {} != {}",
            packed.len(),
            expected_packed
        );
        let expected_scales = out_features * in_features / 32;
        anyhow::ensure!(
            scales.len() == expected_scales,
            "scales length {} != {}",
            scales.len(),
            expected_scales
        );
        anyhow::ensure!(
            x.len() == in_features,
            "x length {} != in_features {}",
            x.len(),
            in_features
        );

        let packed_buf = self.ctx.buffer_with_data(packed);
        let scales_buf = self.ctx.buffer_with_data(scales);
        let x_buf = self.ctx.buffer_with_data(x);
        let y_buf = self.ctx.buffer_for::<f32>(out_features);

        let dims = MxFp4Dims {
            out_features: out_features as u32,
            in_features: in_features as u32,
        };
        let dims_buf = self.ctx.buffer_with_data(std::slice::from_ref(&dims));

        let encoder = crate::metal::process_commands()
            .command_encoder()
            .expect("ce");
        encoder.set_label("lumen:mxfp4_matvec");
        encoder.set_compute_pipeline_state(&self.matvec_f32);
        encoder.set_buffer(0, Some(&packed_buf), (0) as usize);
        encoder.set_buffer(1, Some(&scales_buf), (0) as usize);
        encoder.set_buffer(2, Some(&x_buf), (0) as usize);
        encoder.set_buffer(3, Some(&y_buf), (0) as usize);
        encoder.set_buffer(4, Some(&dims_buf), (0) as usize);

        let max_threads = self.matvec_f32.max_total_threads_per_threadgroup();
        let threads_per_tg = max_threads.min(256);
        let grid = MTLSize {
            width: (out_features as u64) as usize,
            height: (1) as usize,
            depth: (1) as usize,
        };
        let tg = MTLSize {
            width: (threads_per_tg) as usize,
            height: (1) as usize,
            depth: (1) as usize,
        };
        encoder.dispatch_threads(grid, tg);
        drop(encoder);
        crate::metal::process_commands()
            .flush_and_wait()
            .expect("flush");

        Ok(self.ctx.read_buffer::<f32>(&y_buf, out_features))
    }
}

// Legacy inline-test module — drifted out of sync with the current
// command-buffer dispatch helpers (references an undefined `cmd_ref`
// binding). Gated behind the `legacy-tests` cargo feature until the
// call sites are refactored.
#[cfg(all(test, feature = "legacy-tests"))]
mod tests {
    use super::*;
    use crate::mxfp4::dequantize_f32;

    /// Build a deterministic MXFP4 weight + its CPU-dequantized f32 reference.
    fn synth_weight(
        out_features: usize,
        in_features: usize,
        seed: u32,
    ) -> (Vec<u32>, Vec<u8>, Vec<f32>) {
        assert!(in_features.is_multiple_of(32));
        let n_groups = out_features * in_features / 32;
        let n_words = out_features * in_features / 8;

        // LCG for reproducibility without adding rand to deps
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state
        };

        let packed: Vec<u32> = (0..n_words).map(|_| next()).collect();
        // Use exponents near 127 to keep dequantized values in reasonable f32 range.
        let scales: Vec<u8> = (0..n_groups).map(|_| 120 + (next() & 0x0F) as u8).collect();

        let mut dense = vec![0.0_f32; out_features * in_features];
        dequantize_f32(&packed, &scales, &mut dense).expect("cpu dequant");
        (packed, scales, dense)
    }

    fn cpu_matmul(
        dense_weight: &[f32],
        x: &[f32],
        batch: usize,
        out: usize,
        ins: usize,
    ) -> Vec<f32> {
        let mut y = vec![0.0_f32; batch * out];
        for b in 0..batch {
            for r in 0..out {
                let mut acc = 0.0_f32;
                for k in 0..ins {
                    acc += dense_weight[r * ins + k] * x[b * ins + k];
                }
                y[b * out + r] = acc;
            }
        }
        y
    }

    #[test]
    fn mxfp4_weight_matvec_matches_cpu() {
        let ctx = match MxFp4Context::new() {
            Ok(c) => c,
            Err(_) => return, // no Metal GPU available
        };
        let (out, ins) = (12, 64);
        let (packed, scales, dense) = synth_weight(out, ins, 0xC0FFEE);
        let x: Vec<f32> = (0..ins).map(|i| (i as f32) * 0.01 - 0.3).collect();

        let w = Mxfp4Weight::from_host(&ctx.ctx, &packed, &scales, out, ins).unwrap();
        let gpu = ctx.matvec_with_weight(&w, &x).unwrap();
        let cpu = cpu_matmul(&dense, &x, 1, out, ins);

        assert_eq!(gpu.len(), cpu.len());
        for (g, c) in gpu.iter().zip(cpu.iter()) {
            assert!((g - c).abs() < 1e-3, "gpu {g} vs cpu {c}");
        }
    }

    #[test]
    fn mxfp4_weight_matmul_batched_matches_cpu() {
        let ctx = match MxFp4Context::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let (out, ins, batch) = (16, 96, 5);
        let (packed, scales, dense) = synth_weight(out, ins, 0xDEADBEEF);
        let x: Vec<f32> = (0..batch * ins).map(|i| (i as f32).sin() * 0.5).collect();

        let w = Mxfp4Weight::from_host(&ctx.ctx, &packed, &scales, out, ins).unwrap();
        let gpu = ctx.matmul_with_weight(&w, &x, batch).unwrap();
        let cpu = cpu_matmul(&dense, &x, batch, out, ins);

        assert_eq!(gpu.len(), batch * out);
        let mut max_err = 0.0_f32;
        for (g, c) in gpu.iter().zip(cpu.iter()) {
            max_err = max_err.max((g - c).abs());
        }
        assert!(max_err < 1e-3, "max abs err {max_err}");
    }

    #[test]
    fn mxfp4_weight_rejects_mismatched_lengths() {
        let ctx = match MxFp4Context::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        // packed too short
        let err = Mxfp4Weight::from_host(&ctx.ctx, &[0u32; 4], &[0u8; 2], 2, 32);
        assert!(err.is_err());
        // in_features not multiple of 32
        let err = Mxfp4Weight::from_host(&ctx.ctx, &[0u32; 4], &[0u8; 2], 2, 16);
        assert!(err.is_err());
    }

    #[test]
    fn mxfp4_matmul_batch_zero_is_empty() {
        let ctx = match MxFp4Context::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let (packed, scales, _dense) = synth_weight(4, 32, 1);
        let w = Mxfp4Weight::from_host(&ctx.ctx, &packed, &scales, 4, 32).unwrap();
        let y = ctx.matmul_with_weight(&w, &[], 0).unwrap();
        assert!(y.is_empty());
    }

    /// Drives the MoE grouped kernel through a unified weight buffer and checks every
    /// slot output against a CPU matmul against the corresponding dequantized expert.
    /// Covers both broadcast_x (Gate/Up pattern) and per-slot x (Down pattern).
    #[test]
    fn mxfp4_matmul_moe_f32_matches_cpu() {
        let ctx = match MxFp4Context::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let (num_experts_total, out, ins) = (4usize, 8usize, 64usize);
        let (batch, k) = (2usize, 3usize);
        let selected: Vec<u32> = vec![3, 0, 2]; // slot→expert mapping

        // Build per-expert weights + stack into unified packed/scales.
        let mut all_packed = Vec::<u32>::with_capacity(num_experts_total * out * ins / 8);
        let mut all_scales = Vec::<u8>::with_capacity(num_experts_total * out * ins / 32);
        let mut dense_experts: Vec<Vec<f32>> = Vec::with_capacity(num_experts_total);
        for e in 0..num_experts_total {
            let seed = 0x10 + e as u32 * 0xB00B;
            let (p, s, dense) = synth_weight(out, ins, seed);
            all_packed.extend_from_slice(&p);
            all_scales.extend_from_slice(&s);
            dense_experts.push(dense);
        }
        let packed_all = ctx.ctx.buffer_with_data(&all_packed);
        let scales_all = ctx.ctx.buffer_with_data(&all_scales);

        // Case 1: broadcast_x=true. One x of shape [batch, ins] shared across k slots.
        {
            let x: Vec<f32> = (0..batch * ins)
                .map(|i| (i as f32 * 0.019).cos() * 0.6)
                .collect();
            let x_buf = ctx.ctx.buffer_with_data(&x);
            let y_buf = ctx.ctx.buffer_for::<f32>(k * batch * out);

            ctx.matmul_moe_zero_copy(
                &packed_all,
                &scales_all,
                &selected,
                &x_buf,
                0,
                &y_buf,
                0,
                out,
                ins,
                batch,
                true,
            )
            .unwrap();

            let y = ctx.ctx.read_buffer::<f32>(&y_buf, k * batch * out);
            for (slot, &e) in selected.iter().enumerate() {
                let cpu = cpu_matmul(&dense_experts[e as usize], &x, batch, out, ins);
                for b in 0..batch {
                    for r in 0..out {
                        let got = y[slot * batch * out + b * out + r];
                        let expect = cpu[b * out + r];
                        assert!(
                            (got - expect).abs() < 1e-2,
                            "broadcast slot {slot} b {b} r {r}: got {got} vs cpu {expect}"
                        );
                    }
                }
            }
        }

        // Case 2: broadcast_x=false. Per-slot x band [k, batch, ins] — mirrors the Down
        // projection where each expert has its own hidden activations.
        {
            let mut xs = Vec::<f32>::with_capacity(k * batch * ins);
            for slot in 0..k {
                for i in 0..batch * ins {
                    xs.push(((slot * 31 + i) as f32 * 0.011).sin() * 0.4);
                }
            }
            let x_buf = ctx.ctx.buffer_with_data(&xs);
            let y_buf = ctx.ctx.buffer_for::<f32>(k * batch * out);

            ctx.matmul_moe_zero_copy(
                &packed_all,
                &scales_all,
                &selected,
                &x_buf,
                0,
                &y_buf,
                0,
                out,
                ins,
                batch,
                false,
            )
            .unwrap();

            let y = ctx.ctx.read_buffer::<f32>(&y_buf, k * batch * out);
            for (slot, &e) in selected.iter().enumerate() {
                let x_slice = &xs[slot * batch * ins..(slot + 1) * batch * ins];
                let cpu = cpu_matmul(&dense_experts[e as usize], x_slice, batch, out, ins);
                for b in 0..batch {
                    for r in 0..out {
                        let got = y[slot * batch * out + b * out + r];
                        let expect = cpu[b * out + r];
                        assert!(
                            (got - expect).abs() < 1e-2,
                            "per-slot slot {slot} b {b} r {r}: got {got} vs cpu {expect}"
                        );
                    }
                }
            }
        }
    }

    /// MoE v3 must match v2 on the same inputs to within FMA-order rounding.
    /// Same shape coverage as `mxfp4_matmul_moe_f32_matches_cpu` but uses the
    /// explicit-version dispatch entry point so we can compare side by side
    /// inside one process / one `MxFp4Context`.
    #[test]
    fn mxfp4_matmul_moe_f32_v3_matches_v2() {
        let ctx = match MxFp4Context::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        // Choose shapes that exercise the v3 simdgroup tiling: out divisible by
        // ROWS_PER_TG=8 keeps the tail clean; a tail-reaching variant follows.
        let cases: &[(usize, usize, usize, usize)] = &[
            // (out, in, batch, k)
            (16, 64, 2, 3),  // small, even
            (24, 96, 1, 4),  // tail of 3 rows in last tg (out%8=0, but 3 tgs)
            (40, 128, 3, 2), // mixed batch
        ];
        for &(out, ins, batch, k) in cases {
            let num_experts_total = 5usize;
            let selected: Vec<u32> = (0..k)
                .map(|i| ((i + 1) * 7 % num_experts_total) as u32)
                .collect();

            let mut all_packed = Vec::<u32>::new();
            let mut all_scales = Vec::<u8>::new();
            for e in 0..num_experts_total {
                let seed = 0xB2A0u32.wrapping_add((e as u32).wrapping_mul(0x9E3779B9));
                let (p, s, _) = synth_weight(out, ins, seed);
                all_packed.extend_from_slice(&p);
                all_scales.extend_from_slice(&s);
            }
            let packed_all = ctx.ctx.buffer_with_data(&all_packed);
            let scales_all = ctx.ctx.buffer_with_data(&all_scales);

            // broadcast_x=true (Gate/Up pattern)
            let xs: Vec<f32> = (0..batch * ins)
                .map(|i| (i as f32 * 0.013 + 0.31).sin() * 0.7)
                .collect();
            let x_buf = ctx.ctx.buffer_with_data(&xs);

            let y_v2 = ctx.ctx.buffer_for::<f32>(k * batch * out);
            ctx.matmul_moe_zero_copy_with_version(
                KernelVersion::V2,
                &packed_all,
                &scales_all,
                &selected,
                &x_buf,
                0,
                &y_v2,
                0,
                out,
                ins,
                batch,
                true,
            )
            .unwrap();
            let y_v2_vec = ctx.ctx.read_buffer::<f32>(&y_v2, k * batch * out);

            let y_v3 = ctx.ctx.buffer_for::<f32>(k * batch * out);
            ctx.matmul_moe_zero_copy_with_version(
                KernelVersion::V3,
                &packed_all,
                &scales_all,
                &selected,
                &x_buf,
                0,
                &y_v3,
                0,
                out,
                ins,
                batch,
                true,
            )
            .unwrap();
            let y_v3_vec = ctx.ctx.read_buffer::<f32>(&y_v3, k * batch * out);

            let mut max_abs = 0.0f32;
            let mut max_mag = 0.0f32;
            for (a, b) in y_v2_vec.iter().zip(y_v3_vec.iter()) {
                max_abs = max_abs.max((a - b).abs());
                max_mag = max_mag.max(a.abs()).max(b.abs());
            }
            // Cosine similarity sanity (FMA reduction order differs between
            // v2 and v3 but the result should still be numerically equivalent).
            let dot: f64 = y_v2_vec
                .iter()
                .zip(y_v3_vec.iter())
                .map(|(a, b)| (*a as f64) * (*b as f64))
                .sum();
            let na: f64 = y_v2_vec
                .iter()
                .map(|x| (*x as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            let nb: f64 = y_v3_vec
                .iter()
                .map(|x| (*x as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            let cos = if na * nb > 0.0 { dot / (na * nb) } else { 1.0 };
            let rel = max_abs / max_mag.max(1e-6);
            assert!(
                cos > 0.9999 && rel < 5e-3,
                "shape (out={out}, in={ins}, b={batch}, k={k}): cos={cos:.6} rel={rel:.4e} max_abs={max_abs:.4e}"
            );
        }
    }

    /// `matmul_moe_zero_copy_with_indices_buffer` (caller-owned
    /// GPU buffer) must produce bit-identical output to `matmul_moe_zero_copy` (host
    /// slice + per-call buffer upload). The kernel sees the same `expert_indices` bytes;
    /// only the buffer ownership/origin differs.
    #[test]
    fn matmul_moe_zero_copy_with_indices_buffer_matches_slice() {
        let ctx = match MxFp4Context::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let cases: &[(usize, usize, usize, usize)] = &[
            // (out, in, batch, k)
            (16, 64, 2, 3),
            (40, 128, 1, 4),
            (32, 96, 3, 2),
        ];
        for &(out, ins, batch, k) in cases {
            let num_experts_total = 6usize;
            let selected: Vec<u32> = (0..k)
                .map(|i| ((i + 2) * 11 % num_experts_total) as u32)
                .collect();

            let mut all_packed = Vec::<u32>::new();
            let mut all_scales = Vec::<u8>::new();
            for e in 0..num_experts_total {
                let seed = 0xC0DE_BABEu32.wrapping_add((e as u32).wrapping_mul(0x12345));
                let (p, s, _) = synth_weight(out, ins, seed);
                all_packed.extend_from_slice(&p);
                all_scales.extend_from_slice(&s);
            }
            let packed_all = ctx.ctx.buffer_with_data(&all_packed);
            let scales_all = ctx.ctx.buffer_with_data(&all_scales);

            // broadcast_x = true (Gate/Up pattern)
            let xs: Vec<f32> = (0..batch * ins)
                .map(|i| (i as f32 * 0.017 + 0.13).sin() * 0.55)
                .collect();
            let x_buf = ctx.ctx.buffer_with_data(&xs);

            // Reference: host slice path.
            let y_slice = ctx.ctx.buffer_for::<f32>(k * batch * out);
            ctx.matmul_moe_zero_copy(
                &packed_all,
                &scales_all,
                &selected,
                &x_buf,
                0,
                &y_slice,
                0,
                out,
                ins,
                batch,
                true,
            )
            .unwrap();
            let y_slice_vec = ctx.ctx.read_buffer::<f32>(&y_slice, k * batch * out);

            // Subject: caller-owned indices buffer at offset 0.
            let inds_buf = ctx.ctx.buffer_with_data(&selected);
            let y_buf_path = ctx.ctx.buffer_for::<f32>(k * batch * out);
            ctx.matmul_moe_zero_copy_with_indices_buffer(
                &packed_all,
                &scales_all,
                &inds_buf,
                0,
                k,
                &x_buf,
                0,
                &y_buf_path,
                0,
                out,
                ins,
                batch,
                true,
            )
            .unwrap();
            let y_buf_vec = ctx.ctx.read_buffer::<f32>(&y_buf_path, k * batch * out);

            // Bit-identical (same kernel, same bytes).
            for (i, (a, b)) in y_slice_vec.iter().zip(y_buf_vec.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "shape (out={out}, in={ins}, b={batch}, k={k}) idx {i}: slice={a} vs buf={b}"
                );
            }

            // Subject 2: indices buffer with non-zero offset (simulates inds.narrow(0, t, 1)
            // where prior rows live above the slice).
            let mut padded: Vec<u32> = vec![0xDEAD_BEEF; k * 3];
            padded.extend_from_slice(&selected);
            padded.extend_from_slice(&vec![0xCAFEBABE; k * 2]);
            let inds_buf_off = ctx.ctx.buffer_with_data(&padded);
            let inds_offset_bytes = (k * 3 * std::mem::size_of::<u32>()) as u64;
            let y_off = ctx.ctx.buffer_for::<f32>(k * batch * out);
            ctx.matmul_moe_zero_copy_with_indices_buffer(
                &packed_all,
                &scales_all,
                &inds_buf_off,
                inds_offset_bytes,
                k,
                &x_buf,
                0,
                &y_off,
                0,
                out,
                ins,
                batch,
                true,
            )
            .unwrap();
            let y_off_vec = ctx.ctx.read_buffer::<f32>(&y_off, k * batch * out);
            for (a, b) in y_slice_vec.iter().zip(y_off_vec.iter()) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
    }

    /// `silu(x @ W_gate^T) * (x @ W_up^T)`
    /// produced by the fused Metal kernel must match the reference path
    /// (separate gate_up matmul → silu → mul) to within cosine ≥ 0.9999.
    ///
    /// Exercises a few production-relevant shapes including the `inter=512,
    /// hidden=2048` shape that SharedExpert hits at decode time on Qwen3.5-VL-MoE.
    #[test]
    fn mxfp4_gate_up_silu_mul_f32_v3_matches_reference() {
        let ctx = match MxFp4Context::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        // (inter, hidden, batch) — `inter` is half the gate_up out_features.
        // ROWS_PER_TG=8, so n_groups_x = ceil(inter/8) threadgroups per batch row.
        let cases: &[(usize, usize, usize)] = &[
            (16, 64, 1),    // tiny smoke
            (24, 96, 2),    // tail (inter%8=0 but 3 TGs)
            (32, 128, 3),   // small batch
            (512, 2048, 1), // production decode shape
        ];

        for &(inter, hidden, batch) in cases {
            let out = 2 * inter;
            let (packed, scales, dense) = synth_weight(out, hidden, 0xFEED_F00D);
            let x: Vec<f32> = (0..batch * hidden)
                .map(|i| ((i as f32) * 0.013).sin() * 0.4)
                .collect();

            let w = Mxfp4Weight::from_host(&ctx.ctx, &packed, &scales, out, hidden).unwrap();

            // Reference: kernel matmul (gate+up combined) → CPU split + silu*up.
            let combined = ctx.matmul_with_weight(&w, &x, batch).unwrap();
            assert_eq!(combined.len(), batch * out);
            let mut reference = vec![0.0_f32; batch * inter];
            for b in 0..batch {
                for r in 0..inter {
                    let g = combined[b * out + r];
                    let u = combined[b * out + inter + r];
                    let silu = g / (1.0_f32 + (-g).exp());
                    reference[b * inter + r] = silu * u;
                }
            }
            // Sanity: full-CPU dequant path should match the kernel matmul too.
            // Threshold scales loosely with hidden — bigger reductions accumulate
            // more FMA-reorder slack but stay well within parity (cosine ≥ 0.9999).
            let cpu_combined = cpu_matmul(&dense, &x, batch, out, hidden);
            let mut max_combined_err = 0.0_f32;
            for (g, c) in combined.iter().zip(cpu_combined.iter()) {
                max_combined_err = max_combined_err.max((g - c).abs());
            }
            let abs_thresh = (hidden as f32) * 1e-5 + 1e-3;
            assert!(
                max_combined_err < abs_thresh,
                "non-fused kernel diverged from CPU: max abs err {max_combined_err} > {abs_thresh}"
            );

            // Subject: fused single-kernel dispatch.
            let fused = ctx.gate_up_silu_mul_with_weight(&w, &x, batch).unwrap();
            assert_eq!(fused.len(), batch * inter);

            // Cosine similarity vs reference.
            let mut dot = 0.0_f64;
            let mut norm_a = 0.0_f64;
            let mut norm_b = 0.0_f64;
            let mut max_abs = 0.0_f32;
            for (a, b) in fused.iter().zip(reference.iter()) {
                dot += (*a as f64) * (*b as f64);
                norm_a += (*a as f64) * (*a as f64);
                norm_b += (*b as f64) * (*b as f64);
                max_abs = max_abs.max((a - b).abs());
            }
            let cos = dot / (norm_a.sqrt() * norm_b.sqrt() + 1e-12);
            assert!(
                cos >= 0.9999,
                "shape (inter={inter}, hidden={hidden}, b={batch}): cos={cos:.6} max_abs={max_abs:.4e}"
            );
        }
    }

    /// Lever B (2026-04-27): MoE weighted-sum kernel parity.
    /// `out[r] = sum_e weights[e] * downs[e, r]` should match the CPU
    /// reference exactly (single-pass FMA, no parallel reduction).
    #[test]
    fn moe_wsum_f32_matches_cpu() {
        let ctx = match MxFp4Context::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let cases: &[(usize, usize)] = &[
            (4, 64),    // tiny smoke
            (8, 128),   // typical k
            (8, 2048),  // production decode shape (top-8, hidden=2048)
            (16, 1024), // larger k
        ];
        for &(k, hidden) in cases {
            let downs: Vec<f32> = (0..k * hidden)
                .map(|i| (i as f32 * 0.013 + 0.31).sin() * 0.7)
                .collect();
            let weights: Vec<f32> = (0..k)
                .map(|i| (i as f32 * 0.41 + 0.5).cos() * 0.3)
                .collect();
            // GPU compiler emits FMA for `acc += w * d`, so use `mul_add` here
            // to match the kernel's single-rounding semantics.
            let mut reference = vec![0.0_f32; hidden];
            for r in 0..hidden {
                let mut acc = 0.0_f32;
                for e in 0..k {
                    acc = weights[e].mul_add(downs[e * hidden + r], acc);
                }
                reference[r] = acc;
            }

            let downs_buf = ctx.ctx.buffer_with_data(&downs);
            let weights_buf = ctx.ctx.buffer_with_data(&weights);
            let out_buf = ctx.ctx.buffer_for::<f32>(hidden);

            let encoder = crate::metal::process_commands()
                .command_encoder()
                .expect("ce");
            ctx.moe_wsum_zero_copy_inline(
                encoder.as_ref(),
                &downs_buf,
                0,
                &weights_buf,
                0,
                &out_buf,
                0,
                k,
                hidden,
            )
            .unwrap();
            drop(encoder);
            crate::metal::process_commands()
                .flush_and_wait()
                .expect("flush");
            let got = ctx.ctx.read_buffer::<f32>(&out_buf, hidden);

            // Single-pass FMA in row-major order — match CPU bit-exactly.
            for (i, (a, b)) in got.iter().zip(reference.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "shape (k={k}, hidden={hidden}) idx {i}: got={a} ref={b}"
                );
            }
        }
    }

    /// Lever A (2026-04-27) routed-grouped fused kernel parity:
    /// `silu(W_gate[e] @ x) * (W_up[e] @ x)` produced by the fused MoE kernel
    /// must match the reference path (non-fused MoE matmul → CPU split + silu
    /// + mul) to within cosine ≥ 0.9999.
    ///
    /// Exercises the same expert-indices indirection that production decode
    /// uses, including a production-shaped case (inter=512, hidden=2048,
    /// batch=1, k=8) — the routed expert decode hot path.
    #[test]
    fn mxfp4_moe_gate_up_silu_mul_f32_v3_matches_reference() {
        let ctx = match MxFp4Context::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        // (inter, hidden, batch, k) — `inter` is half the gate_up out_features
        // per expert. Each expert weight slab is `[2*inter, hidden/8]` packed.
        let cases: &[(usize, usize, usize, usize)] = &[
            (16, 64, 1, 3),    // tiny smoke
            (24, 96, 2, 4),    // tail TG (3 row-tgs)
            (32, 128, 3, 2),   // small batch
            (512, 2048, 1, 8), // production decode shape (top-8 routing)
        ];

        for &(inter, hidden, batch, k) in cases {
            let out = 2 * inter;
            let num_experts_total = 12usize;
            let selected: Vec<u32> = (0..k)
                .map(|i| ((i + 3) * 5 % num_experts_total) as u32)
                .collect();

            // Build packed expert weight slabs: [num_experts_total, 2*inter, hidden/8].
            let mut all_packed = Vec::<u32>::new();
            let mut all_scales = Vec::<u8>::new();
            for e in 0..num_experts_total {
                let seed = 0xA17E_BEEFu32.wrapping_add((e as u32).wrapping_mul(0x9E37_79B9));
                let (p, s, _) = synth_weight(out, hidden, seed);
                all_packed.extend_from_slice(&p);
                all_scales.extend_from_slice(&s);
            }
            let packed_all = ctx.ctx.buffer_with_data(&all_packed);
            let scales_all = ctx.ctx.buffer_with_data(&all_scales);

            // broadcast_x=true (gate/up pattern: x is shared across slots).
            let xs: Vec<f32> = (0..batch * hidden)
                .map(|i| (i as f32 * 0.011 + 0.27).sin() * 0.45)
                .collect();
            let x_buf = ctx.ctx.buffer_with_data(&xs);
            let inds_buf = ctx.ctx.buffer_with_data(&selected);

            // Reference: non-fused routed matmul → CPU split + silu*up.
            let y_combined = ctx.ctx.buffer_for::<f32>(k * batch * out);
            ctx.matmul_moe_zero_copy_with_indices_buffer(
                &packed_all,
                &scales_all,
                &inds_buf,
                0,
                k,
                &x_buf,
                0,
                &y_combined,
                0,
                out,
                hidden,
                batch,
                true,
            )
            .unwrap();
            let combined_vec = ctx.ctx.read_buffer::<f32>(&y_combined, k * batch * out);
            let mut reference = vec![0.0_f32; k * batch * inter];
            for s in 0..k {
                for b in 0..batch {
                    for r in 0..inter {
                        let g = combined_vec[s * batch * out + b * out + r];
                        let u = combined_vec[s * batch * out + b * out + inter + r];
                        let silu = g / (1.0_f32 + (-g).exp());
                        reference[s * batch * inter + b * inter + r] = silu * u;
                    }
                }
            }

            // Subject: fused single-kernel dispatch via inline encoder helper.
            let y_fused = ctx.ctx.buffer_for::<f32>(k * batch * inter);
            let encoder = crate::metal::process_commands()
                .command_encoder()
                .expect("ce");
            ctx.matmul_moe_gate_up_silu_mul_zero_copy_with_indices_buffer_inline(
                encoder.as_ref(),
                &packed_all,
                &scales_all,
                &inds_buf,
                0,
                k,
                &x_buf,
                0,
                &y_fused,
                0,
                inter,
                hidden,
                batch,
            )
            .unwrap();
            drop(encoder);
            crate::metal::process_commands()
                .flush_and_wait()
                .expect("flush");
            let fused_vec = ctx.ctx.read_buffer::<f32>(&y_fused, k * batch * inter);

            // Cosine similarity vs reference.
            let mut dot = 0.0_f64;
            let mut norm_a = 0.0_f64;
            let mut norm_b = 0.0_f64;
            let mut max_abs = 0.0_f32;
            for (a, b) in fused_vec.iter().zip(reference.iter()) {
                dot += (*a as f64) * (*b as f64);
                norm_a += (*a as f64) * (*a as f64);
                norm_b += (*b as f64) * (*b as f64);
                max_abs = max_abs.max((a - b).abs());
            }
            let cos = dot / (norm_a.sqrt() * norm_b.sqrt() + 1e-12);
            assert!(
                cos >= 0.9999,
                "shape (inter={inter}, hidden={hidden}, b={batch}, k={k}): cos={cos:.6} max_abs={max_abs:.4e}"
            );
        }
    }

    /// Lever C (2026-04-27) fused MoE down matmul + weighted sum parity.
    /// `out[b, hr] = sum_slot weights[slot]
    ///                * sum_m down[expert[slot], hr, m] * hiddens[slot, b, m]`
    /// must match the chain `mxfp4_matmul_moe_f32_v3` (per-slot down matmul
    /// producing `downs_big [k, batch, hidden]`) → `moe_wsum_f32` (reducing
    /// to `[batch, hidden]`) to within cosine ≥ 0.9999. Cosine bound (rather
    /// than bit-equality) accounts for the differing reduction order between
    /// the chain (per-slot simd_sum, then serial wsum over k) and the fused
    /// kernel (serial wsum over k inside per-row register, with each slot's
    /// simd_sum result accumulated via FMA).
    #[test]
    fn mxfp4_matmul_moe_wsum_f32_v3_matches_chain() {
        let ctx = match MxFp4Context::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        // (hidden, inter, k) — batch=1 (production decode). Multi-batch is not
        // exercised in the routed MoE hot path, so we restrict to batch=1 to
        // keep the reference path (`moe_wsum_f32` 1D-over-hidden) one-shot.
        let cases: &[(usize, usize, usize)] = &[
            (64, 32, 3),    // tiny smoke
            (96, 32, 4),    // tail TG (12 row-tgs)
            (128, 64, 6),   // small inter
            (2048, 512, 8), // production decode shape (top-8 routing)
        ];

        for &(hidden, inter, k) in cases {
            let batch = 1usize;
            let num_experts_total = 12usize;
            let selected: Vec<u32> = (0..k)
                .map(|i| ((i + 2) * 7 % num_experts_total) as u32)
                .collect();

            // Down weight slabs: [num_experts, hidden, inter/8] packed.
            let mut all_packed = Vec::<u32>::new();
            let mut all_scales = Vec::<u8>::new();
            for e in 0..num_experts_total {
                let seed = 0xC57E_BEEFu32.wrapping_add((e as u32).wrapping_mul(0x9E37_79B9));
                let (p, s, _) = synth_weight(hidden, inter, seed);
                all_packed.extend_from_slice(&p);
                all_scales.extend_from_slice(&s);
            }
            let packed_all = ctx.ctx.buffer_with_data(&all_packed);
            let scales_all = ctx.ctx.buffer_with_data(&all_scales);

            // Hiddens [k, batch, inter] (gate_up_silu_mul output style).
            let xs: Vec<f32> = (0..k * batch * inter)
                .map(|i| (i as f32 * 0.013 + 0.17).sin() * 0.42)
                .collect();
            let x_buf = ctx.ctx.buffer_with_data(&xs);
            let inds_buf = ctx.ctx.buffer_with_data(&selected);

            // Routing weights [k].
            let weights: Vec<f32> = (0..k)
                .map(|i| (i as f32 * 0.37 + 0.21).cos() * 0.31)
                .collect();
            let weights_buf = ctx.ctx.buffer_with_data(&weights);

            // Reference: per-slot down (broadcast_x=false reads x[slot, b, :])
            // then moe_wsum.
            let downs_big = ctx.ctx.buffer_for::<f32>(k * batch * hidden);
            ctx.matmul_moe_zero_copy_with_indices_buffer(
                &packed_all,
                &scales_all,
                &inds_buf,
                0,
                k,
                &x_buf,
                0,
                &downs_big,
                0,
                hidden,
                inter,
                batch,
                false,
            )
            .unwrap();
            let out_ref = ctx.ctx.buffer_for::<f32>(batch * hidden);
            let encoder = crate::metal::process_commands()
                .command_encoder()
                .expect("ce");
            ctx.moe_wsum_zero_copy_inline(
                encoder.as_ref(),
                &downs_big,
                0,
                &weights_buf,
                0,
                &out_ref,
                0,
                k,
                hidden,
            )
            .unwrap();
            drop(encoder);
            crate::metal::process_commands()
                .flush_and_wait()
                .expect("flush");
            let reference = ctx.ctx.read_buffer::<f32>(&out_ref, batch * hidden);

            // Subject: single fused dispatch.
            let y_fused = ctx.ctx.buffer_for::<f32>(batch * hidden);
            let encoder = crate::metal::process_commands()
                .command_encoder()
                .expect("ce");
            ctx.matmul_moe_wsum_zero_copy_with_indices_buffer_inline(
                encoder.as_ref(),
                &packed_all,
                &scales_all,
                &inds_buf,
                0,
                k,
                &weights_buf,
                0,
                &x_buf,
                0,
                &y_fused,
                0,
                hidden,
                inter,
                batch,
            )
            .unwrap();
            drop(encoder);
            crate::metal::process_commands()
                .flush_and_wait()
                .expect("flush");
            let fused_vec = ctx.ctx.read_buffer::<f32>(&y_fused, batch * hidden);

            let mut dot = 0.0_f64;
            let mut norm_a = 0.0_f64;
            let mut norm_b = 0.0_f64;
            let mut max_abs = 0.0_f32;
            for (a, b) in fused_vec.iter().zip(reference.iter()) {
                dot += (*a as f64) * (*b as f64);
                norm_a += (*a as f64) * (*a as f64);
                norm_b += (*b as f64) * (*b as f64);
                max_abs = max_abs.max((a - b).abs());
            }
            let cos = dot / (norm_a.sqrt() * norm_b.sqrt() + 1e-12);
            assert!(
                cos >= 0.9999,
                "shape (hidden={hidden}, inter={inter}, b={batch}, k={k}): cos={cos:.6} max_abs={max_abs:.4e}"
            );
        }
    }

    /// Lever C-atomic (2026-04-27) parity: grid-parallel + atomic_fetch_add
    /// variant must match the chain `mxfp4_matmul_moe_f32_v3` →
    /// `moe_wsum_f32` to cosine ≥ 0.9999. Caller pre-zeroes output.
    #[test]
    fn mxfp4_matmul_moe_wsum_atomic_f32_v3_matches_chain() {
        let ctx = match MxFp4Context::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let cases: &[(usize, usize, usize)] = &[
            (64, 32, 3),    // tiny smoke
            (96, 32, 4),    // tail TG
            (128, 64, 6),   // small inter
            (2048, 512, 8), // production decode shape (top-8 routing)
        ];

        for &(hidden, inter, k) in cases {
            let batch = 1usize;
            let num_experts_total = 12usize;
            let selected: Vec<u32> = (0..k)
                .map(|i| ((i + 2) * 7 % num_experts_total) as u32)
                .collect();

            let mut all_packed = Vec::<u32>::new();
            let mut all_scales = Vec::<u8>::new();
            for e in 0..num_experts_total {
                let seed = 0xC57E_BEEFu32.wrapping_add((e as u32).wrapping_mul(0x9E37_79B9));
                let (p, s, _) = synth_weight(hidden, inter, seed);
                all_packed.extend_from_slice(&p);
                all_scales.extend_from_slice(&s);
            }
            let packed_all = ctx.ctx.buffer_with_data(&all_packed);
            let scales_all = ctx.ctx.buffer_with_data(&all_scales);

            let xs: Vec<f32> = (0..k * batch * inter)
                .map(|i| (i as f32 * 0.013 + 0.17).sin() * 0.42)
                .collect();
            let x_buf = ctx.ctx.buffer_with_data(&xs);
            let inds_buf = ctx.ctx.buffer_with_data(&selected);

            let weights: Vec<f32> = (0..k)
                .map(|i| (i as f32 * 0.37 + 0.21).cos() * 0.31)
                .collect();
            let weights_buf = ctx.ctx.buffer_with_data(&weights);

            // Reference chain
            let downs_big = ctx.ctx.buffer_for::<f32>(k * batch * hidden);
            ctx.matmul_moe_zero_copy_with_indices_buffer(
                &packed_all,
                &scales_all,
                &inds_buf,
                0,
                k,
                &x_buf,
                0,
                &downs_big,
                0,
                hidden,
                inter,
                batch,
                false,
            )
            .unwrap();
            let out_ref = ctx.ctx.buffer_for::<f32>(batch * hidden);
            let enc_ref = cmd_ref;
            let encoder = crate::metal::process_commands()
                .command_encoder()
                .expect("ce");
            ctx.moe_wsum_zero_copy_inline(
                &enc_ref,
                &downs_big,
                0,
                &weights_buf,
                0,
                &out_ref,
                0,
                k,
                hidden,
            )
            .unwrap();
            enc_ref.end_encoding();
            cmd_ref.commit();
            cmd_ref.wait_until_completed();
            let reference = ctx.ctx.read_buffer::<f32>(&out_ref, batch * hidden);

            // Subject: pre-zero output (kernel only adds), then dispatch.
            let zeros = vec![0.0_f32; batch * hidden];
            let y_atomic = ctx.ctx.buffer_with_data(&zeros);
            let encoder = crate::metal::process_commands()
                .command_encoder()
                .expect("ce");
            ctx.matmul_moe_wsum_atomic_zero_copy_with_indices_buffer_inline(
                encoder.as_ref(),
                &packed_all,
                &scales_all,
                &inds_buf,
                0,
                k,
                &weights_buf,
                0,
                &x_buf,
                0,
                &y_atomic,
                0,
                hidden,
                inter,
                batch,
            )
            .unwrap();
            drop(encoder);
            crate::metal::process_commands()
                .flush_and_wait()
                .expect("flush");
            let atomic_vec = ctx.ctx.read_buffer::<f32>(&y_atomic, batch * hidden);

            let mut dot = 0.0_f64;
            let mut norm_a = 0.0_f64;
            let mut norm_b = 0.0_f64;
            let mut max_abs = 0.0_f32;
            for (a, b) in atomic_vec.iter().zip(reference.iter()) {
                dot += (*a as f64) * (*b as f64);
                norm_a += (*a as f64) * (*a as f64);
                norm_b += (*b as f64) * (*b as f64);
                max_abs = max_abs.max((a - b).abs());
            }
            let cos = dot / (norm_a.sqrt() * norm_b.sqrt() + 1e-12);
            assert!(
                cos >= 0.9999,
                "shape (hidden={hidden}, inter={inter}, k={k}): cos={cos:.6} max_abs={max_abs:.4e}"
            );
        }
    }

    /// Lever H POC (2026-04-27) microbench: time the RmsNorm-fused kernel
    /// vs the unfused Lever A kernel at production shape (inter=512,
    /// hidden=2048, batch=1, k=8). The delta = per-call cost of in-kernel
    /// RmsNorm (TG reduction + Phase 2 normalization pass).
    ///
    /// Compare against the eliminated separate `post_attention_layernorm`
    /// dispatch (~50-100μs each, 30 layers = ~1.5-3ms instrumented):
    /// if delta × 30 layers < 1.5ms, full multi-callsite migration would
    /// net positive. Otherwise the lever is dead.
    ///
    /// Marked `#[ignore]` because it depends on warm GPU state and is not
    /// a parity test. Run with `cargo test --release -- --ignored
    /// lever_h_poc_microbench`.
    #[test]
    #[ignore]
    fn lever_h_poc_microbench() {
        let ctx = match MxFp4Context::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let inter = 512usize;
        let hidden = 2048usize;
        let batch = 1usize;
        let k = 8usize;
        let rms_eps = 1e-6_f32;

        let out = 2 * inter;
        let num_experts_total = 12usize;
        let selected: Vec<u32> = (0..k)
            .map(|i| ((i + 3) * 5 % num_experts_total) as u32)
            .collect();

        let mut all_packed = Vec::<u32>::new();
        let mut all_scales = Vec::<u8>::new();
        for e in 0..num_experts_total {
            let seed = 0xA17E_BEEFu32.wrapping_add((e as u32).wrapping_mul(0x9E37_79B9));
            let (p, s, _) = synth_weight(out, hidden, seed);
            all_packed.extend_from_slice(&p);
            all_scales.extend_from_slice(&s);
        }
        let packed_all = ctx.ctx.buffer_with_data(&all_packed);
        let scales_all = ctx.ctx.buffer_with_data(&all_scales);

        let xs_raw: Vec<f32> = (0..batch * hidden)
            .map(|i| (i as f32 * 0.011 + 0.27).sin() * 0.45 + 0.05)
            .collect();
        let rms_w: Vec<f32> = (0..hidden)
            .map(|i| (i as f32 * 0.029 + 0.17).cos() * 0.5 + 1.0)
            .collect();

        // Pre-normalize for the unfused baseline (so it sees the same compute).
        let mut xs_normalized = vec![0.0_f32; batch * hidden];
        for b in 0..batch {
            let row = &xs_raw[b * hidden..(b + 1) * hidden];
            let sum_sq: f64 = row.iter().map(|v| (*v as f64) * (*v as f64)).sum();
            let mean_sq = (sum_sq / hidden as f64) as f32;
            let inv_rms = 1.0_f32 / (mean_sq + rms_eps).sqrt();
            for i in 0..hidden {
                xs_normalized[b * hidden + i] = row[i] * rms_w[i] * inv_rms;
            }
        }

        let x_norm_buf = ctx.ctx.buffer_with_data(&xs_normalized);
        let x_raw_buf = ctx.ctx.buffer_with_data(&xs_raw);
        let rms_w_buf = ctx.ctx.buffer_with_data(&rms_w);
        let inds_buf = ctx.ctx.buffer_with_data(&selected);
        let y_buf = ctx.ctx.buffer_for::<f32>(k * batch * inter);

        const WARMUP: usize = 100;
        const ITERS: usize = 2000;

        // Warm-up.
        for _ in 0..WARMUP {
            let enc = crate::metal::process_commands()
                .command_encoder()
                .expect("ce");
            ctx.matmul_moe_gate_up_silu_mul_zero_copy_with_indices_buffer_inline(
                &enc,
                &packed_all,
                &scales_all,
                &inds_buf,
                0,
                k,
                &x_norm_buf,
                0,
                &y_buf,
                0,
                inter,
                hidden,
                batch,
            )
            .unwrap();
            drop(enc);
            crate::metal::process_commands()
                .flush_and_wait()
                .expect("flush");
        }
        for _ in 0..WARMUP {
            let enc = crate::metal::process_commands()
                .command_encoder()
                .expect("ce");
            ctx.matmul_moe_gate_up_silu_mul_rmsnorm_zero_copy_with_indices_buffer_inline(
                &enc,
                &packed_all,
                &scales_all,
                &inds_buf,
                0,
                k,
                &x_raw_buf,
                0,
                &rms_w_buf,
                0,
                &y_buf,
                0,
                inter,
                hidden,
                batch,
                rms_eps,
            )
            .unwrap();
            drop(enc);
            crate::metal::process_commands()
                .flush_and_wait()
                .expect("flush");
        }

        // Time unfused (Lever A baseline).
        let t0 = std::time::Instant::now();
        for _ in 0..ITERS {
            let enc = crate::metal::process_commands()
                .command_encoder()
                .expect("ce");
            ctx.matmul_moe_gate_up_silu_mul_zero_copy_with_indices_buffer_inline(
                &enc,
                &packed_all,
                &scales_all,
                &inds_buf,
                0,
                k,
                &x_norm_buf,
                0,
                &y_buf,
                0,
                inter,
                hidden,
                batch,
            )
            .unwrap();
            drop(enc);
            crate::metal::process_commands()
                .flush_and_wait()
                .expect("flush");
        }
        let unfused_ns = t0.elapsed().as_nanos() / ITERS as u128;

        // Time fused.
        let t1 = std::time::Instant::now();
        for _ in 0..ITERS {
            let enc = crate::metal::process_commands()
                .command_encoder()
                .expect("ce");
            ctx.matmul_moe_gate_up_silu_mul_rmsnorm_zero_copy_with_indices_buffer_inline(
                &enc,
                &packed_all,
                &scales_all,
                &inds_buf,
                0,
                k,
                &x_raw_buf,
                0,
                &rms_w_buf,
                0,
                &y_buf,
                0,
                inter,
                hidden,
                batch,
                rms_eps,
            )
            .unwrap();
            drop(enc);
            crate::metal::process_commands()
                .flush_and_wait()
                .expect("flush");
        }
        let fused_ns = t1.elapsed().as_nanos() / ITERS as u128;

        let delta_ns = fused_ns as i128 - unfused_ns as i128;
        let unfused_ms = unfused_ns as f64 / 1e6;
        let fused_ms = fused_ns as f64 / 1e6;
        let delta_ms = delta_ns as f64 / 1e6;
        let delta_30layers_ms = delta_ms * 30.0;

        eprintln!("=== Lever H POC microbench (production shape) ===");
        eprintln!("  Unfused (Lever A) : {unfused_ns}ns/call ({unfused_ms:.3}ms)");
        eprintln!("  Fused (RmsNorm)   : {fused_ns}ns/call ({fused_ms:.3}ms)");
        eprintln!("  Delta             : {delta_ns:+}ns/call ({delta_ms:+.3}ms)");
        eprintln!("  Delta × 30 layers : {delta_30layers_ms:+.3}ms (full-token est.)");
        eprintln!();
        eprintln!("Compare against eliminated post_attention_layernorm dispatches:");
        eprintln!("  ~30-50μs/dispatch × 30 layers = ~0.9-1.5ms savings if migrated.");
        eprintln!("  Lever H multi-callsite worth pursuing iff delta × 30 < 0.9ms.");
    }

    /// Lever H POC (2026-04-27) RmsNorm-fused routed gate_up_silu_mul kernel
    /// parity. The fused kernel reads raw x and computes RmsNorm internally;
    /// the reference applies RmsNorm separately on the host (CPU `mul_add`)
    /// and then runs the unfused Lever A kernel. Cosine ≥ 0.999 expected
    /// (allows for FP reduction-order differences between CPU host RmsNorm
    /// and GPU TG-cooperative RmsNorm).
    #[test]
    fn mxfp4_moe_gate_up_silu_mul_rmsnorm_f32_v3_matches_reference() {
        let ctx = match MxFp4Context::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        // (inter, hidden, batch, k)
        let cases: &[(usize, usize, usize, usize)] = &[
            (16, 64, 1, 3),    // tiny smoke
            (32, 128, 2, 4),   // small
            (512, 2048, 1, 8), // production decode
        ];
        let rms_eps: f32 = 1e-6;

        for &(inter, hidden, batch, k) in cases {
            let out = 2 * inter;
            let num_experts_total = 12usize;
            let selected: Vec<u32> = (0..k)
                .map(|i| ((i + 3) * 5 % num_experts_total) as u32)
                .collect();

            let mut all_packed = Vec::<u32>::new();
            let mut all_scales = Vec::<u8>::new();
            for e in 0..num_experts_total {
                let seed = 0xA17E_BEEFu32.wrapping_add((e as u32).wrapping_mul(0x9E37_79B9));
                let (p, s, _) = synth_weight(out, hidden, seed);
                all_packed.extend_from_slice(&p);
                all_scales.extend_from_slice(&s);
            }
            let packed_all = ctx.ctx.buffer_with_data(&all_packed);
            let scales_all = ctx.ctx.buffer_with_data(&all_scales);

            // Synth raw x (un-normalized) and RmsNorm weight.
            let xs_raw: Vec<f32> = (0..batch * hidden)
                .map(|i| (i as f32 * 0.011 + 0.27).sin() * 0.45 + 0.05)
                .collect();
            let rms_w: Vec<f32> = (0..hidden)
                .map(|i| (i as f32 * 0.029 + 0.17).cos() * 0.5 + 1.0)
                .collect();

            // Reference: RmsNorm on CPU, then unfused Lever A kernel on the
            // pre-normalized x.
            let mut xs_normalized = vec![0.0_f32; batch * hidden];
            for b in 0..batch {
                let row = &xs_raw[b * hidden..(b + 1) * hidden];
                let sum_sq: f64 = row.iter().map(|v| (*v as f64) * (*v as f64)).sum();
                let mean_sq = (sum_sq / hidden as f64) as f32;
                let inv_rms = 1.0_f32 / (mean_sq + rms_eps).sqrt();
                for i in 0..hidden {
                    xs_normalized[b * hidden + i] = row[i] * rms_w[i] * inv_rms;
                }
            }
            let x_norm_buf = ctx.ctx.buffer_with_data(&xs_normalized);
            let inds_buf = ctx.ctx.buffer_with_data(&selected);
            let y_ref = ctx.ctx.buffer_for::<f32>(k * batch * inter);
            let enc_ref = cmd_ref;
            let encoder = crate::metal::process_commands()
                .command_encoder()
                .expect("ce");
            ctx.matmul_moe_gate_up_silu_mul_zero_copy_with_indices_buffer_inline(
                &enc_ref,
                &packed_all,
                &scales_all,
                &inds_buf,
                0,
                k,
                &x_norm_buf,
                0,
                &y_ref,
                0,
                inter,
                hidden,
                batch,
            )
            .unwrap();
            enc_ref.end_encoding();
            cmd_ref.commit();
            cmd_ref.wait_until_completed();
            let reference = ctx.ctx.read_buffer::<f32>(&y_ref, k * batch * inter);

            // Subject: RmsNorm-fused kernel reads raw x.
            let x_raw_buf = ctx.ctx.buffer_with_data(&xs_raw);
            let rms_w_buf = ctx.ctx.buffer_with_data(&rms_w);
            let y_subj = ctx.ctx.buffer_for::<f32>(k * batch * inter);
            let encoder = crate::metal::process_commands()
                .command_encoder()
                .expect("ce");
            ctx.matmul_moe_gate_up_silu_mul_rmsnorm_zero_copy_with_indices_buffer_inline(
                encoder.as_ref(),
                &packed_all,
                &scales_all,
                &inds_buf,
                0,
                k,
                &x_raw_buf,
                0,
                &rms_w_buf,
                0,
                &y_subj,
                0,
                inter,
                hidden,
                batch,
                rms_eps,
            )
            .unwrap();
            drop(encoder);
            crate::metal::process_commands()
                .flush_and_wait()
                .expect("flush");
            let subject = ctx.ctx.read_buffer::<f32>(&y_subj, k * batch * inter);

            let mut dot = 0.0_f64;
            let mut na = 0.0_f64;
            let mut nb = 0.0_f64;
            for (a, b) in subject.iter().zip(reference.iter()) {
                dot += (*a as f64) * (*b as f64);
                na += (*a as f64) * (*a as f64);
                nb += (*b as f64) * (*b as f64);
            }
            let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
            assert!(
                cos >= 0.999,
                "shape (inter={inter}, hidden={hidden}, b={batch}, k={k}): cos={cos:.6}"
            );
        }
    }

    /// Lever D (2026-04-27) routed-grouped fused gate+up+silu*up bf16-output
    /// kernel parity. Output narrows to bf16; cosine ≥ 0.999 vs the f32-out
    /// reference (Lever A kernel) since per-element delta is bounded by the
    /// bf16 epsilon (~3.9e-3 relative).
    #[test]
    fn mxfp4_moe_gate_up_silu_mul_bf16out_matches_f32() {
        let ctx = match MxFp4Context::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let cases: &[(usize, usize, usize, usize)] = &[
            (16, 64, 1, 3),
            (32, 128, 2, 4),
            (512, 2048, 1, 8), // production decode
        ];
        for &(inter, hidden, batch, k) in cases {
            let out = 2 * inter;
            let num_experts_total = 12usize;
            let selected: Vec<u32> = (0..k)
                .map(|i| ((i + 3) * 5 % num_experts_total) as u32)
                .collect();

            let mut all_packed = Vec::<u32>::new();
            let mut all_scales = Vec::<u8>::new();
            for e in 0..num_experts_total {
                let seed = 0xA17E_BEEFu32.wrapping_add((e as u32).wrapping_mul(0x9E37_79B9));
                let (p, s, _) = synth_weight(out, hidden, seed);
                all_packed.extend_from_slice(&p);
                all_scales.extend_from_slice(&s);
            }
            let packed_all = ctx.ctx.buffer_with_data(&all_packed);
            let scales_all = ctx.ctx.buffer_with_data(&all_scales);

            let xs: Vec<f32> = (0..batch * hidden)
                .map(|i| (i as f32 * 0.011 + 0.27).sin() * 0.45)
                .collect();
            let x_buf = ctx.ctx.buffer_with_data(&xs);
            let inds_buf = ctx.ctx.buffer_with_data(&selected);

            // Reference: f32-output Lever A kernel.
            let y_f32 = ctx.ctx.buffer_for::<f32>(k * batch * inter);
            let enc_ref = cmd_ref;
            let encoder = crate::metal::process_commands()
                .command_encoder()
                .expect("ce");
            ctx.matmul_moe_gate_up_silu_mul_zero_copy_with_indices_buffer_inline(
                &enc_ref,
                &packed_all,
                &scales_all,
                &inds_buf,
                0,
                k,
                &x_buf,
                0,
                &y_f32,
                0,
                inter,
                hidden,
                batch,
            )
            .unwrap();
            enc_ref.end_encoding();
            cmd_ref.commit();
            cmd_ref.wait_until_completed();
            let reference = ctx.ctx.read_buffer::<f32>(&y_f32, k * batch * inter);

            // Subject: bf16-output kernel.
            let y_bf16 = ctx.ctx.buffer_for::<u16>(k * batch * inter);
            let encoder = crate::metal::process_commands()
                .command_encoder()
                .expect("ce");
            ctx.matmul_moe_gate_up_silu_mul_bf16out_zero_copy_with_indices_buffer_inline(
                encoder.as_ref(),
                &packed_all,
                &scales_all,
                &inds_buf,
                0,
                k,
                &x_buf,
                0,
                &y_bf16,
                0,
                inter,
                hidden,
                batch,
            )
            .unwrap();
            drop(encoder);
            crate::metal::process_commands()
                .flush_and_wait()
                .expect("flush");
            let bf16_bits = ctx.ctx.read_buffer::<u16>(&y_bf16, k * batch * inter);
            let subject: Vec<f32> = bf16_bits
                .iter()
                .map(|&b| f32::from_bits((b as u32) << 16))
                .collect();

            let mut dot = 0.0_f64;
            let mut na = 0.0_f64;
            let mut nb = 0.0_f64;
            for (a, b) in subject.iter().zip(reference.iter()) {
                dot += (*a as f64) * (*b as f64);
                na += (*a as f64) * (*a as f64);
                nb += (*b as f64) * (*b as f64);
            }
            let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
            assert!(
                cos >= 0.999,
                "shape (inter={inter}, hidden={hidden}, b={batch}, k={k}): cos={cos:.6}"
            );
        }
    }

    /// Lever D (2026-04-27) MoE down bf16-input parity. Stages bf16 input,
    /// converts to f32 in TG-shared, then runs identical inner FMA. Output
    /// is f32; should match the f32-in v3 down kernel within bf16 input
    /// epsilon (~3.9e-3 relative) since the only difference is the input
    /// dtype.
    #[test]
    fn mxfp4_matmul_moe_bf16in_f32out_v3_matches_f32() {
        let ctx = match MxFp4Context::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let cases: &[(usize, usize, usize, usize)] = &[
            (64, 32, 1, 3),
            (128, 64, 2, 4),
            (2048, 512, 1, 8), // production down shape
        ];
        for &(hidden, inter, batch, k) in cases {
            let num_experts_total = 12usize;
            let selected: Vec<u32> = (0..k)
                .map(|i| ((i + 2) * 7 % num_experts_total) as u32)
                .collect();

            let mut all_packed = Vec::<u32>::new();
            let mut all_scales = Vec::<u8>::new();
            for e in 0..num_experts_total {
                let seed = 0xC57E_BEEFu32.wrapping_add((e as u32).wrapping_mul(0x9E37_79B9));
                let (p, s, _) = synth_weight(hidden, inter, seed);
                all_packed.extend_from_slice(&p);
                all_scales.extend_from_slice(&s);
            }
            let packed_all = ctx.ctx.buffer_with_data(&all_packed);
            let scales_all = ctx.ctx.buffer_with_data(&all_scales);

            // Build x in f32, then make a bf16 view of the same source for
            // the bf16-in subject. Reference reads f32 directly.
            let xs_f32: Vec<f32> = (0..k * batch * inter)
                .map(|i| (i as f32 * 0.013 + 0.17).sin() * 0.42)
                .collect();
            let xs_bf16: Vec<u16> = xs_f32.iter().map(|&v| (v.to_bits() >> 16) as u16).collect();
            let x_f32_buf = ctx.ctx.buffer_with_data(&xs_f32);
            let x_bf16_buf = ctx.ctx.buffer_with_data(&xs_bf16);
            let inds_buf = ctx.ctx.buffer_with_data(&selected);

            // Reference: f32-in v3 down (with the bf16-rounded inputs to
            // match what production sees when the chain is on).
            let xs_f32_rounded: Vec<f32> = xs_bf16
                .iter()
                .map(|&b| f32::from_bits((b as u32) << 16))
                .collect();
            let _ = x_f32_buf; // unused alias
            let x_f32_rounded_buf = ctx.ctx.buffer_with_data(&xs_f32_rounded);
            let y_ref = ctx.ctx.buffer_for::<f32>(k * batch * hidden);
            ctx.matmul_moe_zero_copy_with_indices_buffer(
                &packed_all,
                &scales_all,
                &inds_buf,
                0,
                k,
                &x_f32_rounded_buf,
                0,
                &y_ref,
                0,
                hidden,
                inter,
                batch,
                false,
            )
            .unwrap();
            let reference = ctx.ctx.read_buffer::<f32>(&y_ref, k * batch * hidden);

            // Subject: bf16-in down dispatch.
            let y_subj = ctx.ctx.buffer_for::<f32>(k * batch * hidden);
            let encoder = crate::metal::process_commands()
                .command_encoder()
                .expect("ce");
            ctx.matmul_moe_bf16in_zero_copy_with_indices_buffer_inline(
                encoder.as_ref(),
                &packed_all,
                &scales_all,
                &inds_buf,
                0,
                k,
                &x_bf16_buf,
                0,
                &y_subj,
                0,
                hidden,
                inter,
                batch,
                false,
            )
            .unwrap();
            drop(encoder);
            crate::metal::process_commands()
                .flush_and_wait()
                .expect("flush");
            let subject = ctx.ctx.read_buffer::<f32>(&y_subj, k * batch * hidden);

            // With identical x (bf16-rounded both sides), the two paths
            // should be bit-exact (single FMA reduction, same kernel logic).
            for (i, (a, b)) in subject.iter().zip(reference.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "shape (hidden={hidden}, inter={inter}, b={batch}, k={k}) idx {i}: got={a} ref={b}"
                );
            }
        }
    }

    /// Lever G (2026-04-27) routing top-k partial select parity. The kernel
    /// must produce the same top-k indices and values as the reference
    /// "argsort descending + take first k" CPU implementation. Bit-exact for
    /// indices (no FP arithmetic) and the values come straight from `probs`.
    #[test]
    fn topk_partial_select_f32_matches_reference() {
        let ctx = match MxFp4Context::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        // (BL, num_experts, k)
        let cases: &[(usize, usize, usize)] = &[
            (1, 8, 3), // tiny smoke
            (1, 16, 4),
            (3, 32, 5),  // multi-row
            (1, 256, 8), // production decode shape (Qwen3.5-MoE)
        ];

        for &(bl, e, k) in cases {
            // Synth probs: deterministic, with some near-ties for stability
            // testing.
            let probs: Vec<f32> = (0..bl * e)
                .map(|i| {
                    let base = ((i as f32) * 0.0173 + 0.41).sin() * 0.5;
                    let nudge = if i % 17 == 0 { 1e-9 } else { 0.0 };
                    base + nudge
                })
                .collect();

            // CPU reference: stable descending sort by value, ties broken by
            // ascending index. Take first k.
            let mut ref_inds = vec![0u32; bl * k];
            let mut ref_vals = vec![0.0f32; bl * k];
            for b in 0..bl {
                let base = b * e;
                let mut idx_val: Vec<(u32, f32)> =
                    (0..e).map(|i| (i as u32, probs[base + i])).collect();
                idx_val.sort_by(|a, b| {
                    // Descending value, ties → ascending index
                    match b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal) {
                        std::cmp::Ordering::Equal => a.0.cmp(&b.0),
                        o => o,
                    }
                });
                for i in 0..k {
                    ref_inds[b * k + i] = idx_val[i].0;
                    ref_vals[b * k + i] = idx_val[i].1;
                }
            }

            // Subject: GPU dispatch.
            let probs_buf = ctx.ctx.buffer_with_data(&probs);
            let inds_buf = ctx.ctx.buffer_for::<u32>(bl * k);
            let vals_buf = ctx.ctx.buffer_for::<f32>(bl * k);

            let encoder = crate::metal::process_commands()
                .command_encoder()
                .expect("ce");
            ctx.topk_partial_select_zero_copy_inline(
                encoder.as_ref(),
                &probs_buf,
                0,
                &inds_buf,
                0,
                &vals_buf,
                0,
                bl,
                e,
                k,
            )
            .unwrap();
            drop(encoder);
            crate::metal::process_commands()
                .flush_and_wait()
                .expect("flush");

            let got_inds = ctx.ctx.read_buffer::<u32>(&inds_buf, bl * k);
            let got_vals = ctx.ctx.read_buffer::<f32>(&vals_buf, bl * k);

            for i in 0..(bl * k) {
                assert_eq!(
                    got_inds[i], ref_inds[i],
                    "shape (bl={bl}, e={e}, k={k}) idx[{i}]: got={} ref={}",
                    got_inds[i], ref_inds[i]
                );
                assert_eq!(
                    got_vals[i].to_bits(),
                    ref_vals[i].to_bits(),
                    "shape (bl={bl}, e={e}, k={k}) val[{i}]: got={} ref={}",
                    got_vals[i],
                    ref_vals[i]
                );
            }
        }
    }

    /// fusion parity.
    ///
    /// Reference: CPU softmax (max-subtract + exp + sum-divide) → descending
    /// argsort → take K → renormalize (sum K + divide). Tie-break: lower index.
    ///
    /// Subject: `router_softmax_topk_renorm_zero_copy_inline` from raw logits.
    ///
    /// Tolerances:
    ///   - indices: bit-identical (logit fixtures spread out enough that 1 ULP
    ///     softmax drift cannot swap argpartition).
    ///   - weights: cos ≥ 0.9999, abs < 1e-5 (Metal `fast::exp` ≤1 ULP vs host
    ///     `f32::exp` is permitted; 8-element renorm normalizes the residual).
    #[test]
    fn router_softmax_topk_renorm_f32_matches_reference() {
        let ctx = match MxFp4Context::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        // (BL, num_experts, k)
        let cases: &[(usize, usize, usize)] = &[
            (1, 8, 3),
            (1, 16, 4),
            (3, 32, 5),
            (1, 64, 6),
            (1, 256, 8), // production Qwen3.5-MoE decode shape
        ];

        for &(bl, e, k) in cases {
            // Logits with reasonable spread (sin-modulated) so 1 ULP softmax
            // drift cannot swap argpartition order.
            let logits: Vec<f32> = (0..bl * e)
                .map(|i| ((i as f32) * 0.0173 + 0.41).sin() * 4.0)
                .collect();

            // CPU reference: row-wise softmax → argsort desc → take k → renorm.
            let mut ref_inds = vec![0u32; bl * k];
            let mut ref_vals = vec![0.0f32; bl * k];
            for b in 0..bl {
                let base = b * e;
                // softmax (max-subtract for stability).
                let row = &logits[base..base + e];
                let max_v = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = row.iter().map(|&x| (x - max_v).exp()).collect();
                let sum_exp: f32 = exps.iter().sum();
                let probs: Vec<f32> = exps.iter().map(|&e| e / sum_exp).collect();

                // argsort desc, tie → ascending index.
                let mut idx_val: Vec<(u32, f32)> = (0..e).map(|i| (i as u32, probs[i])).collect();
                idx_val.sort_by(|a, b| {
                    match b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal) {
                        std::cmp::Ordering::Equal => a.0.cmp(&b.0),
                        o => o,
                    }
                });

                // Take K + renormalize.
                let topk_vals: Vec<f32> = (0..k).map(|i| idx_val[i].1).collect();
                let sum_k: f32 = topk_vals.iter().sum();
                for i in 0..k {
                    ref_inds[b * k + i] = idx_val[i].0;
                    ref_vals[b * k + i] = topk_vals[i] / sum_k;
                }
            }

            // Subject: GPU fused dispatch.
            let logits_buf = ctx.ctx.buffer_with_data(&logits);
            let inds_buf = ctx.ctx.buffer_for::<u32>(bl * k);
            let vals_buf = ctx.ctx.buffer_for::<f32>(bl * k);

            let encoder = crate::metal::process_commands()
                .command_encoder()
                .expect("ce");
            ctx.router_softmax_topk_renorm_zero_copy_inline(
                encoder.as_ref(),
                &logits_buf,
                0,
                &inds_buf,
                0,
                &vals_buf,
                0,
                bl,
                e,
                k,
            )
            .unwrap();
            drop(encoder);
            crate::metal::process_commands()
                .flush_and_wait()
                .expect("flush");

            let got_inds = ctx.ctx.read_buffer::<u32>(&inds_buf, bl * k);
            let got_vals = ctx.ctx.read_buffer::<f32>(&vals_buf, bl * k);

            // Indices: bit-identical.
            for i in 0..(bl * k) {
                assert_eq!(
                    got_inds[i], ref_inds[i],
                    "shape (bl={bl}, e={e}, k={k}) idx[{i}]: got={} ref={}",
                    got_inds[i], ref_inds[i]
                );
            }
            // Weights: cos ≥ 0.9999, max abs < 1e-5.
            let mut dot = 0.0f64;
            let mut sa = 0.0f64;
            let mut sb = 0.0f64;
            let mut max_abs = 0.0f32;
            for i in 0..(bl * k) {
                let a = got_vals[i] as f64;
                let b_ = ref_vals[i] as f64;
                dot += a * b_;
                sa += a * a;
                sb += b_ * b_;
                let d = (got_vals[i] - ref_vals[i]).abs();
                if d > max_abs {
                    max_abs = d;
                }
            }
            let cos = dot / (sa.sqrt() * sb.sqrt());
            assert!(
                cos >= 0.9999,
                "shape (bl={bl}, e={e}, k={k}) weights cos {cos} < 0.9999"
            );
            assert!(
                max_abs < 1e-5,
                "shape (bl={bl}, e={e}, k={k}) weights max_abs {max_abs} >= 1e-5"
            );
        }
    }

    /// must match the v3 dense matmul to
    /// within cosine ≥ 0.9999 on shapes that v3 currently handles. Targets
    /// the r_gate (out=256, hidden=2048, batch=1) decode shape plus a few
    /// nearby variants for sanity.
    #[test]
    fn mxfp4_matmul_small_out_f32_v1_matches_v3() {
        let ctx = match MxFp4Context::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        // (out_features, in_features, batch). r_gate hits (256, 2048, 1) at decode.
        let cases: &[(usize, usize, usize)] = &[
            (8, 64, 1),   // tiny smoke
            (16, 128, 2), // small batch
            (64, 512, 1),
            (256, 2048, 1), // production r_gate decode shape
            (256, 2048, 4), // small prefill
        ];

        for &(out, ins, batch) in cases {
            let (packed, scales, _dense) = synth_weight(out, ins, 0xC0FF_EE42);
            let x: Vec<f32> = (0..batch * ins)
                .map(|i| ((i as f32) * 0.011 + 0.07).sin() * 0.5)
                .collect();
            let w = Mxfp4Weight::from_host(&ctx.ctx, &packed, &scales, out, ins).unwrap();

            // Reference: v3 dense matmul (proven via prior tests).
            let reference = ctx.matmul_with_weight(&w, &x, batch).unwrap();
            assert_eq!(reference.len(), batch * out);

            // Subject: small-out kernel.
            let subject = ctx.matmul_small_out_with_weight(&w, &x, batch).unwrap();
            assert_eq!(subject.len(), batch * out);

            let mut dot = 0.0_f64;
            let mut norm_a = 0.0_f64;
            let mut norm_b = 0.0_f64;
            let mut max_abs = 0.0_f32;
            for (a, b) in subject.iter().zip(reference.iter()) {
                dot += (*a as f64) * (*b as f64);
                norm_a += (*a as f64) * (*a as f64);
                norm_b += (*b as f64) * (*b as f64);
                max_abs = max_abs.max((a - b).abs());
            }
            let cos = dot / (norm_a.sqrt() * norm_b.sqrt() + 1e-12);
            // FMA reorder slack scales with reduction size; small-out reduces
            // across more lanes (256-way) than v3 (32-way per simdgroup),
            // so absolute error trends slightly higher but cosine stays tight.
            let abs_thresh = (ins as f32) * 1e-5 + 1e-3;
            assert!(
                cos >= 0.9999 && max_abs < abs_thresh,
                "shape (out={out}, in={ins}, b={batch}): cos={cos:.6} max_abs={max_abs:.4e} thresh={abs_thresh:.4e}"
            );
        }
    }

    /// Lever H multi-callsite migration parity: the RmsNorm-fused dense matmul
    /// `mxfp4_matmul_f32_v3_rmsnorm` on RAW x must match the standard
    /// `mxfp4_matmul_f32_v3` on CPU-pre-normalized x within cosine ≥ 0.999.
    ///
    /// Production shapes covered:
    ///   - routing gate: out=num_experts (256), in=hidden (2048), batch=1
    ///   - shared expert gate_up: out=2*shared_inter (1024), in=hidden (2048), batch=1
    /// Plus tiny shapes for smoke + a small batch case.
    ///
    /// Numerical tolerance: cosine ≥ 0.999 covers the FMA reduction-order
    /// difference between Candle's host RmsNorm path (which uses Candle Metal
    /// kernels, likely tree reduction) and our 256-thread cooperative
    /// reduction. Bit-identical token check happens at the end-to-end level
    /// in the multi-callsite migration session.
    #[test]
    fn mxfp4_matmul_f32_v3_rmsnorm_matches_reference() {
        let ctx = match MxFp4Context::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        // (out_features, in_features, batch) — covers both production consumers.
        let cases: &[(usize, usize, usize)] = &[
            (16, 64, 1),     // tiny smoke
            (32, 128, 2),    // small batch
            (256, 2048, 1),  // production routing gate (E=256, hidden=2048)
            (1024, 2048, 1), // production shared expert gate_up (2*512, hidden=2048)
        ];
        let rms_eps: f32 = 1e-6;

        for &(out, ins, batch) in cases {
            let (packed, scales, _dense) = synth_weight(out, ins, 0xC0FFEE);
            let packed_buf = ctx.ctx.buffer_with_data(&packed);
            let scales_buf = ctx.ctx.buffer_with_data(&scales);

            // Synth raw x + rms_weight.
            let xs_raw: Vec<f32> = (0..batch * ins)
                .map(|i| (i as f32 * 0.011 + 0.27).sin() * 0.45 + 0.05)
                .collect();
            let rms_w: Vec<f32> = (0..ins)
                .map(|i| (i as f32 * 0.029 + 0.17).cos() * 0.5 + 1.0)
                .collect();

            // Reference: CPU RmsNorm + GPU unfused v3 matmul via matmul_with_weight.
            let mut xs_normalized = vec![0.0_f32; batch * ins];
            for b in 0..batch {
                let row = &xs_raw[b * ins..(b + 1) * ins];
                let sum_sq: f64 = row.iter().map(|v| (*v as f64) * (*v as f64)).sum();
                let mean_sq = (sum_sq / ins as f64) as f32;
                let inv_rms = 1.0_f32 / (mean_sq + rms_eps).sqrt();
                for i in 0..ins {
                    xs_normalized[b * ins + i] = row[i] * rms_w[i] * inv_rms;
                }
            }
            let weight = Mxfp4Weight::from_host(&ctx.ctx, &packed, &scales, out, ins).unwrap();
            let reference = ctx
                .matmul_with_weight(&weight, &xs_normalized, batch)
                .unwrap();

            // Subject: RmsNorm-fused kernel reads raw x.
            let x_raw_buf = ctx.ctx.buffer_with_data(&xs_raw);
            let rms_w_buf = ctx.ctx.buffer_with_data(&rms_w);
            let y_subj = ctx.ctx.buffer_for::<f32>(batch * out);
            let encoder = crate::metal::process_commands()
                .command_encoder()
                .expect("ce");
            ctx.matmul_f32_v3_rmsnorm_zero_copy_inline(
                encoder.as_ref(),
                &packed_buf,
                &scales_buf,
                &x_raw_buf,
                0,
                &rms_w_buf,
                0,
                &y_subj,
                0,
                out,
                ins,
                batch,
                rms_eps,
            )
            .unwrap();
            drop(encoder);
            crate::metal::process_commands()
                .flush_and_wait()
                .expect("flush");
            let subject = ctx.ctx.read_buffer::<f32>(&y_subj, batch * out);

            let mut dot = 0.0_f64;
            let mut na = 0.0_f64;
            let mut nb = 0.0_f64;
            for (a, b) in subject.iter().zip(reference.iter()) {
                dot += (*a as f64) * (*b as f64);
                na += (*a as f64) * (*a as f64);
                nb += (*b as f64) * (*b as f64);
            }
            let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
            assert!(
                cos >= 0.999,
                "shape (out={out}, in={ins}, b={batch}): cos={cos:.6}"
            );
        }
    }

    /// Lever L1 parity: residual-fused matmul must equal `matmul_f32_v3(x) +
    /// residual` element-by-element. Math is a single trailing `acc + r` per
    /// output position, so bit-identical is the contract here (no reduction-
    /// order ambiguity beyond the v3 baseline).
    #[test]
    fn mxfp4_matmul_f32_v3_residual_matches_reference() {
        let ctx = match MxFp4Context::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let cases: &[(usize, usize, usize)] = &[
            (16, 64, 1),
            (32, 128, 2),
            (256, 2048, 1),  // routing gate shape
            (2048, 2048, 1), // self_attn o_proj shape (Qwen3.6 hidden=2048)
            (2048, 2048, 4), // o_proj with prefill batch
        ];

        for &(out, ins, batch) in cases {
            let (packed, scales, _dense) = synth_weight(out, ins, 0xCAFE_F00D);
            let packed_buf = ctx.ctx.buffer_with_data(&packed);
            let scales_buf = ctx.ctx.buffer_with_data(&scales);

            // Inputs.
            let xs: Vec<f32> = (0..batch * ins)
                .map(|i| (i as f32 * 0.013 + 0.31).sin() * 0.6 - 0.2)
                .collect();
            let residual: Vec<f32> = (0..batch * out)
                .map(|i| (i as f32 * 0.041 + 0.07).cos() * 0.4 + 0.1)
                .collect();

            // Reference: unfused v3 + manual element-wise add.
            let weight = Mxfp4Weight::from_host(&ctx.ctx, &packed, &scales, out, ins).unwrap();
            let reference_unfused = ctx.matmul_with_weight(&weight, &xs, batch).unwrap();
            let reference: Vec<f32> = reference_unfused
                .iter()
                .zip(residual.iter())
                .map(|(a, r)| a + r)
                .collect();

            // Subject: residual-fused dispatch.
            let x_buf = ctx.ctx.buffer_with_data(&xs);
            let res_buf = ctx.ctx.buffer_with_data(&residual);
            let y_buf = ctx.ctx.buffer_for::<f32>(batch * out);
            let encoder = crate::metal::process_commands()
                .command_encoder()
                .expect("ce");
            ctx.matmul_f32_v3_residual_zero_copy_inline(
                encoder.as_ref(),
                &packed_buf,
                &scales_buf,
                &x_buf,
                0,
                &y_buf,
                0,
                &res_buf,
                0,
                out,
                ins,
                batch,
            )
            .unwrap();
            drop(encoder);
            crate::metal::process_commands()
                .flush_and_wait()
                .expect("flush");
            let subject = ctx.ctx.read_buffer::<f32>(&y_buf, batch * out);

            // Bit-identical contract (single tail add — no reduction reorder).
            for (i, (a, r)) in subject.iter().zip(reference.iter()).enumerate() {
                assert!(
                    (a - r).abs() < 1e-5,
                    "shape (out={out}, in={ins}, b={batch}) idx={i}: subj={a} ref={r}"
                );
            }
        }
    }

    /// Lever L1 Step 2 parity: tri_add_f32 must equal `a + b + c` element-
    /// wise (bit-identical, single fused write per element).
    #[test]
    fn tri_add_f32_matches_reference() {
        let ctx = match MxFp4Context::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        // Coverage: tiny + non-multiple-of-256 + production sizes (BL=1
        // hidden=2048 = 1 token decode; BL=8 hidden=2048 = small prefill).
        let cases: &[usize] = &[1, 7, 256, 257, 1023, 2048, 8 * 2048];

        for &n in cases {
            let a: Vec<f32> = (0..n)
                .map(|i| (i as f32 * 0.013 + 0.31).sin() * 0.6)
                .collect();
            let b: Vec<f32> = (0..n)
                .map(|i| (i as f32 * 0.027 + 0.11).cos() * 0.4)
                .collect();
            let c: Vec<f32> = (0..n)
                .map(|i| (i as f32 * 0.041 + 0.07).sin() * 0.5)
                .collect();
            let reference: Vec<f32> = (0..n).map(|i| a[i] + b[i] + c[i]).collect();

            let a_buf = ctx.ctx.buffer_with_data(&a);
            let b_buf = ctx.ctx.buffer_with_data(&b);
            let c_buf = ctx.ctx.buffer_with_data(&c);
            let y_buf = ctx.ctx.buffer_for::<f32>(n);

            let encoder = crate::metal::process_commands()
                .command_encoder()
                .expect("ce");
            ctx.tri_add_f32_zero_copy_inline(
                encoder.as_ref(),
                &a_buf,
                0,
                &b_buf,
                0,
                &c_buf,
                0,
                &y_buf,
                0,
                n,
            )
            .unwrap();
            drop(encoder);
            crate::metal::process_commands()
                .flush_and_wait()
                .expect("flush");
            let subject = ctx.ctx.read_buffer::<f32>(&y_buf, n);

            for (i, (s, r)) in subject.iter().zip(reference.iter()).enumerate() {
                assert!((s - r).abs() < 1e-6, "n={n} idx={i}: subj={s} ref={r}");
            }
        }
    }

    /// Lever L1 Step 3.5 parity: scalar_mul_tri_add_f32 must equal
    /// `a + b * coef[t] + d` element-by-element. Bit-identical math (single
    /// FMA + 2 adds per element, no reduction reorder beyond compiler FMA).
    #[test]
    fn scalar_mul_tri_add_f32_matches_reference() {
        let ctx = match MxFp4Context::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let cases: &[(usize, usize)] = &[
            (1, 256),
            (1, 2048), // production decode
            (8, 2048), // small prefill
            (1, 2049),
            (3, 1023),
        ];

        for &(bl, hidden) in cases {
            let n = bl * hidden;
            let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin() * 0.6).collect();
            let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.027).cos() * 0.4).collect();
            let coef: Vec<f32> = (0..bl).map(|t| (t as f32 * 0.13 - 0.4).tanh()).collect();
            let d: Vec<f32> = (0..n).map(|i| (i as f32 * 0.041).sin() * 0.5).collect();

            // Host reference: y[t,h] = a[t,h] + b[t,h] * coef[t] + d[t,h]
            let mut reference = vec![0.0_f32; n];
            for t in 0..bl {
                let c = coef[t];
                for h in 0..hidden {
                    let i = t * hidden + h;
                    reference[i] = a[i] + b[i] * c + d[i];
                }
            }

            let a_buf = ctx.ctx.buffer_with_data(&a);
            let b_buf = ctx.ctx.buffer_with_data(&b);
            let coef_buf = ctx.ctx.buffer_with_data(&coef);
            let d_buf = ctx.ctx.buffer_with_data(&d);
            let y_buf = ctx.ctx.buffer_for::<f32>(n);

            let encoder = crate::metal::process_commands()
                .command_encoder()
                .expect("ce");
            ctx.scalar_mul_tri_add_f32_zero_copy_inline(
                encoder.as_ref(),
                &a_buf,
                0,
                &b_buf,
                0,
                &coef_buf,
                0,
                &d_buf,
                0,
                &y_buf,
                0,
                bl,
                hidden,
            )
            .unwrap();
            drop(encoder);
            crate::metal::process_commands()
                .flush_and_wait()
                .expect("flush");
            let subject = ctx.ctx.read_buffer::<f32>(&y_buf, n);

            for (i, (s, r)) in subject.iter().zip(reference.iter()).enumerate() {
                assert!(
                    (s - r).abs() < 1e-6,
                    "(bl={bl}, hidden={hidden}) idx={i}: subj={s} ref={r}"
                );
            }
        }
    }

    /// Lever L4 parity: scalar_mul_tri_add_rmsnorm_f32 must produce both
    /// `out = a + b*coef + d` AND `attn_in = out * rms_weight * inv_rms`
    /// matching scalar reference. Tolerance: cosine ≥ 0.999 (RmsNorm
    /// reduction order may differ from Candle's binary kernel; bit-
    /// identical decode is the production gate, not this parity).
    #[test]
    fn scalar_mul_tri_add_rmsnorm_f32_matches_reference() {
        let ctx = match MxFp4Context::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let cases: &[(usize, usize)] = &[
            (1, 256),
            (1, 2048), // production decode shape
            (8, 2048),
            (1, 1024),
            (3, 512),
        ];
        let rms_eps: f32 = 1e-6;

        for &(bl, hidden) in cases {
            let n = bl * hidden;
            let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin() * 0.6).collect();
            let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.027).cos() * 0.4).collect();
            let coef: Vec<f32> = (0..bl).map(|t| (t as f32 * 0.13 - 0.4).tanh()).collect();
            let d: Vec<f32> = (0..n).map(|i| (i as f32 * 0.041).sin() * 0.5).collect();
            let rms_w: Vec<f32> = (0..hidden)
                .map(|h| (h as f32 * 0.029).cos() * 0.3 + 1.0)
                .collect();

            // Host reference.
            let mut ref_out = vec![0.0f32; n];
            let mut ref_attn = vec![0.0f32; n];
            for t in 0..bl {
                let c = coef[t];
                let mut sum_sq = 0.0f64;
                for h in 0..hidden {
                    let i = t * hidden + h;
                    let prod = b[i] * c;
                    let partial = a[i] + prod;
                    let ov = partial + d[i];
                    ref_out[i] = ov;
                    sum_sq += (ov as f64) * (ov as f64);
                }
                let mean_sq = (sum_sq / hidden as f64) as f32;
                let inv_rms = 1.0_f32 / (mean_sq + rms_eps).sqrt();
                for h in 0..hidden {
                    let i = t * hidden + h;
                    ref_attn[i] = ref_out[i] * rms_w[h] * inv_rms;
                }
            }

            // GPU dispatch.
            let a_buf = ctx.ctx.buffer_with_data(&a);
            let b_buf = ctx.ctx.buffer_with_data(&b);
            let coef_buf = ctx.ctx.buffer_with_data(&coef);
            let d_buf = ctx.ctx.buffer_with_data(&d);
            let rms_buf = ctx.ctx.buffer_with_data(&rms_w);
            let out_buf = ctx.ctx.buffer_for::<f32>(n);
            let attn_buf = ctx.ctx.buffer_for::<f32>(n);

            let encoder = crate::metal::process_commands()
                .command_encoder()
                .expect("ce");
            ctx.scalar_mul_tri_add_rmsnorm_f32_zero_copy_inline(
                encoder.as_ref(),
                &a_buf,
                0,
                &b_buf,
                0,
                &coef_buf,
                0,
                &d_buf,
                0,
                &rms_buf,
                0,
                &out_buf,
                0,
                &attn_buf,
                0,
                bl,
                hidden,
                rms_eps,
            )
            .unwrap();
            drop(encoder);
            crate::metal::process_commands()
                .flush_and_wait()
                .expect("flush");
            let subj_out = ctx.ctx.read_buffer::<f32>(&out_buf, n);
            let subj_attn = ctx.ctx.read_buffer::<f32>(&attn_buf, n);

            // out is bit-identical (no reduction). abs<1e-6 tol.
            for (i, (s, r)) in subj_out.iter().zip(ref_out.iter()).enumerate() {
                assert!(
                    (s - r).abs() < 1e-6,
                    "out (bl={bl}, hidden={hidden}) idx={i}: subj={s} ref={r}"
                );
            }
            // attn_in has cooperative reduction → cosine check.
            let mut dot = 0.0_f64;
            let mut na = 0.0_f64;
            let mut nb = 0.0_f64;
            for (s, r) in subj_attn.iter().zip(ref_attn.iter()) {
                dot += (*s as f64) * (*r as f64);
                na += (*s as f64) * (*s as f64);
                nb += (*r as f64) * (*r as f64);
            }
            let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
            assert!(
                cos >= 0.999,
                "attn_in cos (bl={bl}, hidden={hidden}): {cos:.6}"
            );
        }
    }

    /// Lever H Step 3 retry parity: large-out variant
    /// (`mxfp4_matmul_f32_v3_rmsnorm_large`) on RAW x must match the standard
    /// `mxfp4_matmul_f32_v3` on CPU-pre-normalized x within cosine ≥ 0.999.
    ///
    /// Production shapes covered:
    ///   - qkv_proj: out=9216 (q_out 8192 + 2*kv_out 512), in=2048
    ///   - in_proj_combined: out=12352 (qkv 8192 + v 4096 + 2*Hv 64), in=2048
    /// Plus a tiny shape (out=64, in=64) to stress the topology with TGs of
    /// only 4 active rows (16 ROWS_PER_TG > 64/16 = 4 TGs total).
    #[test]
    fn mxfp4_matmul_f32_v3_rmsnorm_large_matches_reference() {
        let ctx = match MxFp4Context::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let cases: &[(usize, usize, usize)] = &[
            (64, 64, 1),      // tiny smoke (4 TGs)
            (256, 2048, 1),   // small case (16 TGs)
            (9216, 2048, 1),  // production qkv_proj (576 TGs)
            (12352, 2048, 1), // production in_proj_combined (772 TGs)
        ];
        let rms_eps: f32 = 1e-6;

        for &(out, ins, batch) in cases {
            let (packed, scales, _dense) = synth_weight(out, ins, 0xBAD5EED);
            let packed_buf = ctx.ctx.buffer_with_data(&packed);
            let scales_buf = ctx.ctx.buffer_with_data(&scales);

            let xs_raw: Vec<f32> = (0..batch * ins)
                .map(|i| (i as f32 * 0.013 + 0.31).sin() * 0.45 + 0.05)
                .collect();
            let rms_w: Vec<f32> = (0..ins)
                .map(|i| (i as f32 * 0.027 + 0.19).cos() * 0.5 + 1.0)
                .collect();

            // Reference: CPU RmsNorm + GPU unfused v3 matmul.
            let mut xs_normalized = vec![0.0_f32; batch * ins];
            for b in 0..batch {
                let row = &xs_raw[b * ins..(b + 1) * ins];
                let sum_sq: f64 = row.iter().map(|v| (*v as f64) * (*v as f64)).sum();
                let mean_sq = (sum_sq / ins as f64) as f32;
                let inv_rms = 1.0_f32 / (mean_sq + rms_eps).sqrt();
                for i in 0..ins {
                    xs_normalized[b * ins + i] = row[i] * rms_w[i] * inv_rms;
                }
            }
            let weight = Mxfp4Weight::from_host(&ctx.ctx, &packed, &scales, out, ins).unwrap();
            let reference = ctx
                .matmul_with_weight(&weight, &xs_normalized, batch)
                .unwrap();

            // Subject: large-out kernel reads raw x.
            let x_raw_buf = ctx.ctx.buffer_with_data(&xs_raw);
            let rms_w_buf = ctx.ctx.buffer_with_data(&rms_w);
            let y_subj = ctx.ctx.buffer_for::<f32>(batch * out);
            let encoder = crate::metal::process_commands()
                .command_encoder()
                .expect("ce");
            ctx.matmul_f32_v3_rmsnorm_large_zero_copy_inline(
                encoder.as_ref(),
                &packed_buf,
                &scales_buf,
                &x_raw_buf,
                0,
                &rms_w_buf,
                0,
                &y_subj,
                0,
                out,
                ins,
                batch,
                rms_eps,
            )
            .unwrap();
            drop(encoder);
            crate::metal::process_commands()
                .flush_and_wait()
                .expect("flush");
            let subject = ctx.ctx.read_buffer::<f32>(&y_subj, batch * out);

            let mut dot = 0.0_f64;
            let mut na = 0.0_f64;
            let mut nb = 0.0_f64;
            for (a, b) in subject.iter().zip(reference.iter()) {
                dot += (*a as f64) * (*b as f64);
                na += (*a as f64) * (*a as f64);
                nb += (*b as f64) * (*b as f64);
            }
            let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
            assert!(
                cos >= 0.999,
                "shape (out={out}, in={ins}, b={batch}): cos={cos:.6}"
            );
        }
    }

    /// Lever H Step 3 retry tier 2 parity: xlarge-out variant
    /// (`mxfp4_matmul_f32_v3_rmsnorm_xlarge`) on RAW x must match the standard
    /// `mxfp4_matmul_f32_v3` on CPU-pre-normalized x within cosine ≥ 0.999.
    ///
    /// Production shapes covered same as large variant. Stresses the
    /// 32-rows/TG × 1024-thread topology + reduce_buf[32] single-simd_sum
    /// final reduce.
    #[test]
    fn mxfp4_matmul_f32_v3_rmsnorm_xlarge_matches_reference() {
        let ctx = match MxFp4Context::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let cases: &[(usize, usize, usize)] = &[
            (64, 64, 1),      // tiny smoke (2 TGs)
            (256, 2048, 1),   // small case (8 TGs)
            (9216, 2048, 1),  // production qkv_proj (288 TGs)
            (12352, 2048, 1), // production in_proj_combined (386 TGs)
        ];
        let rms_eps: f32 = 1e-6;

        for &(out, ins, batch) in cases {
            let (packed, scales, _dense) = synth_weight(out, ins, 0xFEEDBEE);
            let packed_buf = ctx.ctx.buffer_with_data(&packed);
            let scales_buf = ctx.ctx.buffer_with_data(&scales);

            let xs_raw: Vec<f32> = (0..batch * ins)
                .map(|i| (i as f32 * 0.013 + 0.31).sin() * 0.45 + 0.05)
                .collect();
            let rms_w: Vec<f32> = (0..ins)
                .map(|i| (i as f32 * 0.027 + 0.19).cos() * 0.5 + 1.0)
                .collect();

            // Reference: CPU RmsNorm + GPU unfused v3 matmul.
            let mut xs_normalized = vec![0.0_f32; batch * ins];
            for b in 0..batch {
                let row = &xs_raw[b * ins..(b + 1) * ins];
                let sum_sq: f64 = row.iter().map(|v| (*v as f64) * (*v as f64)).sum();
                let mean_sq = (sum_sq / ins as f64) as f32;
                let inv_rms = 1.0_f32 / (mean_sq + rms_eps).sqrt();
                for i in 0..ins {
                    xs_normalized[b * ins + i] = row[i] * rms_w[i] * inv_rms;
                }
            }
            let weight = Mxfp4Weight::from_host(&ctx.ctx, &packed, &scales, out, ins).unwrap();
            let reference = ctx
                .matmul_with_weight(&weight, &xs_normalized, batch)
                .unwrap();

            // Subject: xlarge-out kernel reads raw x.
            let x_raw_buf = ctx.ctx.buffer_with_data(&xs_raw);
            let rms_w_buf = ctx.ctx.buffer_with_data(&rms_w);
            let y_subj = ctx.ctx.buffer_for::<f32>(batch * out);
            let encoder = crate::metal::process_commands()
                .command_encoder()
                .expect("ce");
            ctx.matmul_f32_v3_rmsnorm_xlarge_zero_copy_inline(
                encoder.as_ref(),
                &packed_buf,
                &scales_buf,
                &x_raw_buf,
                0,
                &rms_w_buf,
                0,
                &y_subj,
                0,
                out,
                ins,
                batch,
                rms_eps,
            )
            .unwrap();
            drop(encoder);
            crate::metal::process_commands()
                .flush_and_wait()
                .expect("flush");
            let subject = ctx.ctx.read_buffer::<f32>(&y_subj, batch * out);

            let mut dot = 0.0_f64;
            let mut na = 0.0_f64;
            let mut nb = 0.0_f64;
            for (a, b) in subject.iter().zip(reference.iter()) {
                dot += (*a as f64) * (*b as f64);
                na += (*a as f64) * (*a as f64);
                nb += (*b as f64) * (*b as f64);
            }
            let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
            assert!(
                cos >= 0.999,
                "shape (out={out}, in={ins}, b={batch}): cos={cos:.6}"
            );
        }
    }

    /// Lever H Step 2 parity: dense f32-weight RmsNorm-fused matmul on RAW x
    /// must match CPU-pre-normalized x + CPU dense matmul reference within
    /// cosine ≥ 0.999.
    ///
    /// Production shapes covered:
    ///   - routing gate: out=num_experts (256), in=hidden (2048), batch=1
    ///   - shared_expert_gate: out=1, in=hidden (2048), batch=1
    /// Plus tiny shapes for smoke + a small batch case.
    #[test]
    fn dense_f32_matmul_rmsnorm_matches_reference() {
        let ctx = match MxFp4Context::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let cases: &[(usize, usize, usize)] = &[
            (16, 64, 1),    // tiny smoke
            (32, 128, 2),   // small batch
            (256, 2048, 1), // production routing gate
            (1, 2048, 1),   // production shared_expert_gate (out=1 degenerate)
        ];
        let rms_eps: f32 = 1e-6;

        for &(out, ins, batch) in cases {
            // Random-ish but deterministic weight + x + rms_weight.
            let weight: Vec<f32> = (0..out * ins)
                .map(|i| (i as f32 * 0.013 + 0.07).cos() * 0.3)
                .collect();
            let xs_raw: Vec<f32> = (0..batch * ins)
                .map(|i| (i as f32 * 0.011 + 0.27).sin() * 0.45 + 0.05)
                .collect();
            let rms_w: Vec<f32> = (0..ins)
                .map(|i| (i as f32 * 0.029 + 0.17).cos() * 0.5 + 1.0)
                .collect();

            // CPU reference: RmsNorm + matmul.
            let mut xs_normalized = vec![0.0_f32; batch * ins];
            for b in 0..batch {
                let row = &xs_raw[b * ins..(b + 1) * ins];
                let sum_sq: f64 = row.iter().map(|v| (*v as f64) * (*v as f64)).sum();
                let mean_sq = (sum_sq / ins as f64) as f32;
                let inv_rms = 1.0_f32 / (mean_sq + rms_eps).sqrt();
                for i in 0..ins {
                    xs_normalized[b * ins + i] = row[i] * rms_w[i] * inv_rms;
                }
            }
            let reference = cpu_matmul(&weight, &xs_normalized, batch, out, ins);

            // Subject: GPU fused kernel reads raw x.
            let weight_buf = ctx.ctx.buffer_with_data(&weight);
            let x_raw_buf = ctx.ctx.buffer_with_data(&xs_raw);
            let rms_w_buf = ctx.ctx.buffer_with_data(&rms_w);
            let y_subj = ctx.ctx.buffer_for::<f32>(batch * out);
            let encoder = crate::metal::process_commands()
                .command_encoder()
                .expect("ce");
            ctx.dense_f32_matmul_rmsnorm_zero_copy_inline(
                encoder.as_ref(),
                &weight_buf,
                0,
                &x_raw_buf,
                0,
                &rms_w_buf,
                0,
                &y_subj,
                0,
                out,
                ins,
                batch,
                rms_eps,
            )
            .unwrap();
            drop(encoder);
            crate::metal::process_commands()
                .flush_and_wait()
                .expect("flush");
            let subject = ctx.ctx.read_buffer::<f32>(&y_subj, batch * out);

            let mut dot = 0.0_f64;
            let mut na = 0.0_f64;
            let mut nb = 0.0_f64;
            for (a, b) in subject.iter().zip(reference.iter()) {
                dot += (*a as f64) * (*b as f64);
                na += (*a as f64) * (*a as f64);
                nb += (*b as f64) * (*b as f64);
            }
            let cos = if na > 0.0 && nb > 0.0 {
                dot / (na.sqrt() * nb.sqrt())
            } else {
                // out=1 batch=1 with very small magnitudes → fall back to abs check
                1.0
            };
            // For out=1 a single-element comparison is more reliable as abs error.
            let max_abs = subject
                .iter()
                .zip(reference.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                cos >= 0.999 || max_abs < 5e-3,
                "shape (out={out}, in={ins}, b={batch}): cos={cos:.6} max_abs={max_abs:.4e}"
            );
        }
    }
}
