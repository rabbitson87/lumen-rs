//! Flash Attention 2 — single-dispatch fused Q @ K^T * scale → softmax → @ V.
//!
//! Replaces the three-dispatch SDPA sequence (matmul + softmax + matmul) with
//! a single Metal kernel dispatch that keeps intermediate state in threadgroup
//! SRAM.  Requires Metal-resident f32 tensors with head_dim = 256.
//!
//! Returns `None` when any precondition fails (wrong device, wrong dtype,
//! wrong head_dim, or `LUMEN_DISABLE_FLASH_ATTN=1`) so the caller can fall
//! back to the standard 3-dispatch SDPA.

#[cfg(feature = "model-integration")]
use crate::metal::{BatchedEncoderExt, CommandBufferExt, ComputeEncoderCompat};
use candle_core::{DType, Device, Storage, Tensor};

use std::sync::{
    Once, OnceLock,
    atomic::{AtomicBool, Ordering},
};

// ── Shader source ─────────────────────────────────────────────────────────────
const SHADER_SRC: &str = include_str!("shaders/flash_attn.metal");

// ── Lazy pipeline ─────────────────────────────────────────────────────────────

struct SafePipeline(crate::metal::ComputePipelineState);
// Metal pipeline states are immutable after creation and safe across threads.
unsafe impl Send for SafePipeline {}
unsafe impl Sync for SafePipeline {}

static FA_PIPELINE: OnceLock<SafePipeline> = OnceLock::new();
static FA_FAILED:   AtomicBool             = AtomicBool::new(false);

// bf16 I/O variant — Workstream C, MLX dtype policy alignment. Internal
// accumulation stays f32 (correctness — bf16 mantissa too narrow for
// 256+-element softmax denominators). Selected when all of Q/K/V/output/mask
// are bf16; mixed dtype returns None so caller falls back to 3-dispatch SDPA.
static FA_BF16_PIPELINE: OnceLock<SafePipeline> = OnceLock::new();
static FA_BF16_FAILED:   AtomicBool             = AtomicBool::new(false);

static SDPAV_PIPELINE: OnceLock<SafePipeline> = OnceLock::new();
static SDPAV_FAILED:   AtomicBool             = AtomicBool::new(false);

// SDPA vector port — opt-in until validated. Reads LUMEN_USE_SDPA_VECTOR
// once at first call; can be toggled at runtime via set_sdpa_vector_enabled.
static SDPAV_ENABLED:      AtomicBool = AtomicBool::new(false);
static SDPAV_ENABLED_INIT: Once       = Once::new();

fn init_sdpav_enabled() {
    SDPAV_ENABLED_INIT.call_once(|| {
        if std::env::var("LUMEN_USE_SDPA_VECTOR").as_deref() == Ok("1") {
            SDPAV_ENABLED.store(true, Ordering::Relaxed);
        }
    });
}

fn sdpav_enabled() -> bool {
    init_sdpav_enabled();
    SDPAV_ENABLED.load(Ordering::Relaxed)
}

/// Toggle the MLX-style SDPA vector kernel at runtime (used by A/B benchmarks).
pub fn set_sdpa_vector_enabled(on: bool) {
    init_sdpav_enabled();
    SDPAV_ENABLED.store(on, Ordering::Relaxed);
}

// ── Env-var gate ──────────────────────────────────────────────────────────────
// AtomicBool so benchmarks and tests can override at runtime via set_disabled().
// Reads LUMEN_DISABLE_FLASH_ATTN once at first call, then stays writable.
static FA_DISABLED:      AtomicBool = AtomicBool::new(false);
static FA_DISABLED_INIT: Once       = Once::new();

fn init_disabled() {
    FA_DISABLED_INIT.call_once(|| {
        if std::env::var("LUMEN_DISABLE_FLASH_ATTN").as_deref() == Ok("1") {
            FA_DISABLED.store(true, Ordering::Relaxed);
        }
    });
}

fn is_disabled() -> bool {
    init_disabled();
    FA_DISABLED.load(Ordering::Relaxed)
}

/// Override the disabled state at runtime.  Used by benchmarks and tests so
/// that a single-process A/B can toggle Flash Attention without relying on
/// env-var re-reads (which are cached after the first call).
pub fn set_disabled(disabled: bool) {
    // Ensure env-var has been read at least once before we override it,
    // so that a cold process still respects LUMEN_DISABLE_FLASH_ATTN.
    init_disabled();
    FA_DISABLED.store(disabled, Ordering::Relaxed);
}

// ── Pipeline compilation ──────────────────────────────────────────────────────

fn compile_pipeline() -> anyhow::Result<crate::metal::ComputePipelineState> {
    let device = crate::metal::Device::system_default()
        .ok_or_else(|| anyhow::anyhow!("flash_attn: no Metal GPU"))?;

    let opts = crate::metal::new_compile_options();
    // Version 3.1 required for the `bfloat` type used by `tq_flash_attn_bf16`
    // (Workstream C). The f32 kernel is unaffected by the version bump — Metal
    // 3.1 is a strict superset of 3.0 for our usage.
    opts.set_language_version(crate::metal::MTLLanguageVersion::Version3_1);
    opts.set_fast_math_enabled(true);

    let library = device
        .new_library_with_source(SHADER_SRC, Some(opts.as_ref()))
        .map_err(|e| anyhow::anyhow!("flash_attn Metal compile: {e}"))?;

    let func = library
        .get_function("tq_flash_attn", None)
        .map_err(|e| anyhow::anyhow!("tq_flash_attn not found: {e}"))?;

    device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| anyhow::anyhow!("tq_flash_attn pipeline: {e}"))
}

fn compile_bf16_pipeline() -> anyhow::Result<crate::metal::ComputePipelineState> {
    let device = crate::metal::Device::system_default()
        .ok_or_else(|| anyhow::anyhow!("flash_attn_bf16: no Metal GPU"))?;

    let opts = crate::metal::new_compile_options();
    opts.set_language_version(crate::metal::MTLLanguageVersion::Version3_1);
    opts.set_fast_math_enabled(true);

    let library = device
        .new_library_with_source(SHADER_SRC, Some(opts.as_ref()))
        .map_err(|e| anyhow::anyhow!("flash_attn_bf16 Metal compile: {e}"))?;

    let func = library
        .get_function("tq_flash_attn_bf16", None)
        .map_err(|e| anyhow::anyhow!("tq_flash_attn_bf16 not found: {e}"))?;

    device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| anyhow::anyhow!("tq_flash_attn_bf16 pipeline: {e}"))
}

fn get_bf16_pipeline() -> Option<&'static crate::metal::ComputePipelineState> {
    if FA_BF16_FAILED.load(Ordering::Relaxed) {
        return None;
    }
    if let Some(p) = FA_BF16_PIPELINE.get() {
        return Some(&p.0);
    }
    match compile_bf16_pipeline() {
        Ok(pl) => {
            let _ = FA_BF16_PIPELINE.set(SafePipeline(pl));
            FA_BF16_PIPELINE.get().map(|p| &p.0)
        }
        Err(e) => {
            eprintln!("flash_attn_bf16: pipeline compile failed ({e}); caller falls back");
            FA_BF16_FAILED.store(true, Ordering::Relaxed);
            None
        }
    }
}

fn get_pipeline() -> Option<&'static crate::metal::ComputePipelineState> {
    if FA_FAILED.load(Ordering::Relaxed) {
        return None;
    }
    if let Some(p) = FA_PIPELINE.get() {
        return Some(&p.0);
    }
    match compile_pipeline() {
        Ok(pl) => {
            let _ = FA_PIPELINE.set(SafePipeline(pl));
            FA_PIPELINE.get().map(|p| &p.0)
        }
        Err(e) => {
            eprintln!("flash_attn: pipeline compile failed ({e}); using 3-dispatch SDPA");
            FA_FAILED.store(true, Ordering::Relaxed);
            None
        }
    }
}

fn compile_sdpav_pipeline() -> anyhow::Result<crate::metal::ComputePipelineState> {
    let device = crate::metal::Device::system_default()
        .ok_or_else(|| anyhow::anyhow!("sdpa_vector: no Metal GPU"))?;

    let opts = crate::metal::new_compile_options();
    // Version 3.1 to match the rest of this file (the source includes a
    // `bfloat` kernel — compiling at 3.0 fails even when only the f32 kernel
    // is requested).
    opts.set_language_version(crate::metal::MTLLanguageVersion::Version3_1);
    opts.set_fast_math_enabled(true);

    let library = device
        .new_library_with_source(SHADER_SRC, Some(opts.as_ref()))
        .map_err(|e| anyhow::anyhow!("sdpa_vector Metal compile: {e}"))?;

    let func = library
        .get_function("tq_sdpa_vector", None)
        .map_err(|e| anyhow::anyhow!("tq_sdpa_vector not found: {e}"))?;

    device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| anyhow::anyhow!("tq_sdpa_vector pipeline: {e}"))
}

fn get_sdpav_pipeline() -> Option<&'static crate::metal::ComputePipelineState> {
    if SDPAV_FAILED.load(Ordering::Relaxed) {
        return None;
    }
    if let Some(p) = SDPAV_PIPELINE.get() {
        return Some(&p.0);
    }
    match compile_sdpav_pipeline() {
        Ok(pl) => {
            let _ = SDPAV_PIPELINE.set(SafePipeline(pl));
            SDPAV_PIPELINE.get().map(|p| &p.0)
        }
        Err(e) => {
            eprintln!("sdpa_vector: pipeline compile failed ({e}); falling back to FA2");
            SDPAV_FAILED.store(true, Ordering::Relaxed);
            None
        }
    }
}

// ── Zero-copy Metal buffer extraction ─────────────────────────────────────────

#[cfg(feature = "model-integration")]
fn metal_buf(t: &Tensor) -> Option<(&crate::metal::Buffer, usize)> {
    let (storage, layout) = t.storage_and_layout();
    match &*storage {
        Storage::Metal(ms) => {
            let offset = layout.start_offset() * t.dtype().size_in_bytes();
            // SAFETY: lumen-metal re-exports candle-metal-kernels::metal::Buffer
            // directly, so the types are ABI-identical (same underlying ObjC object).
            let buf: &crate::metal::Buffer =
                unsafe { &*(ms.buffer() as *const _ as *const crate::metal::Buffer) };
            Some((buf, offset))
        }
        _ => None,
    }
}

// ── Inline kernel dispatch ────────────────────────────────────────────────────

#[cfg(feature = "model-integration")]
#[allow(clippy::too_many_arguments)]
fn encode_flash_attn(
    enc:      &crate::metal::ComputeCommandEncoderRef,
    pipeline: &crate::metal::ComputePipelineState,
    q_buf: &crate::metal::Buffer, q_off: usize,
    k_buf: &crate::metal::Buffer, k_off: usize,
    v_buf: &crate::metal::Buffer, v_off: usize,
    o_buf: &crate::metal::Buffer, o_off: usize,
    mask_buf: Option<(&crate::metal::Buffer, usize)>,
    b: u32, h: u32, sq: u32, skv: u32,
    scale: f32,
    group: u32,
) {
    enc.set_compute_pipeline_state(pipeline);

    enc.set_buffer(0, Some(q_buf), q_off);
    enc.set_buffer(1, Some(k_buf), k_off);
    enc.set_buffer(2, Some(v_buf), v_off);
    enc.set_buffer(3, Some(o_buf), o_off);
    if let Some((mb, mo)) = mask_buf {
        enc.set_buffer(4, Some(mb), mo);
    } else {
        enc.set_buffer(4, None, 0);
    }

    let set_u32 = |idx: usize, v: u32| {
        let bytes = v.to_ne_bytes();
        enc.set_bytes_directly(idx, 4, bytes.as_ptr() as *const _);
    };
    let set_f32 = |idx: usize, v: f32| {
        let bytes = v.to_ne_bytes();
        enc.set_bytes_directly(idx, 4, bytes.as_ptr() as *const _);
    };

    set_u32(5, b);
    set_u32(6, h);
    set_u32(7, sq);
    set_u32(8, skv);
    set_f32(9, scale);
    set_u32(10, if mask_buf.is_some() { 1 } else { 0 });
    set_u32(11, group);

    // Grid: one threadgroup per (batch, head, query-row); 256 threads per TG.
    enc.dispatch_thread_groups(
        crate::mtl_size!((b * h * sq) as usize, 1, 1),
        crate::mtl_size!(256_usize, 1, 1),
    );
}

#[cfg(feature = "model-integration")]
#[allow(clippy::too_many_arguments)]
fn encode_sdpa_vector(
    enc:      &crate::metal::ComputeCommandEncoderRef,
    pipeline: &crate::metal::ComputePipelineState,
    q_buf: &crate::metal::Buffer, q_off: usize,
    k_buf: &crate::metal::Buffer, k_off: usize,
    v_buf: &crate::metal::Buffer, v_off: usize,
    o_buf: &crate::metal::Buffer, o_off: usize,
    mask_buf: Option<(&crate::metal::Buffer, usize)>,
    b: u32, h: u32, sq: u32, skv: u32,
    scale: f32,
    group: u32,
) {
    enc.set_compute_pipeline_state(pipeline);

    enc.set_buffer(0, Some(q_buf), q_off);
    enc.set_buffer(1, Some(k_buf), k_off);
    enc.set_buffer(2, Some(v_buf), v_off);
    enc.set_buffer(3, Some(o_buf), o_off);
    if let Some((mb, mo)) = mask_buf {
        enc.set_buffer(4, Some(mb), mo);
    } else {
        enc.set_buffer(4, None, 0);
    }

    let set_u32 = |idx: usize, v: u32| {
        let bytes = v.to_ne_bytes();
        enc.set_bytes_directly(idx, 4, bytes.as_ptr() as *const _);
    };
    let set_f32 = |idx: usize, v: f32| {
        let bytes = v.to_ne_bytes();
        enc.set_bytes_directly(idx, 4, bytes.as_ptr() as *const _);
    };

    set_u32(5, b);
    set_u32(6, h);
    set_u32(7, sq);
    set_u32(8, skv);
    set_f32(9, scale);
    set_u32(10, if mask_buf.is_some() { 1 } else { 0 });
    set_u32(11, group);

    // Grid: (B*H, Sq, 1) threadgroups, 1024 threads per TG (32 sgs × 32 lanes).
    enc.dispatch_thread_groups(
        crate::mtl_size!((b * h) as usize, sq as usize, 1),
        crate::mtl_size!(1024_usize, 1, 1),
    );
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Attempt fused Flash Attention 2 for the given Q / K / V tensors.
///
/// Returns `Some(Ok(output))` on success, `None` if preconditions are not met
/// (caller should fall back to 3-dispatch SDPA), or `Some(Err(_))` on a
/// genuine GPU error.
///
/// Preconditions:
/// - All tensors (Q/K/V/mask) must be Metal-resident and share the same dtype:
///   either all `F32` (uses the existing kernel) or all `BF16` (uses the new
///   bf16-I/O kernel — Workstream C, MLX policy alignment). Mixed dtype
///   returns `None`.
/// - `d == head_dim == 256`.
/// - Q: [B, H,    Sq,  D]
/// - K: [B, H_kv, Skv, D]  (H_kv must divide H; group = H / H_kv ∈ {1, 2, 4, 8, …})
/// - V: [B, H_kv, Skv, D]
/// - `mask` (if present): last two dims must be [Sq, Skv]; dtype must match Q/K/V.
/// - `LUMEN_DISABLE_FLASH_ATTN != "1"`.
///
/// When K/V are passed at full `H` (post-`repeat_kv_heads`), group=1 — kernel
/// behaves identically to the pre-GQA-in-kernel path. When K/V are passed at
/// `H_kv = H / G`, the kernel reads `K[h_kv = h/G]` directly — saves the
/// expand+contiguous materialization upstream.
#[cfg(feature = "model-integration")]
pub fn flash_attn_candle(
    q:     &Tensor,
    k:     &Tensor,
    v:     &Tensor,
    mask:  Option<&Tensor>,
    scale: f32,
) -> Option<candle_core::Result<Tensor>> {
    if is_disabled() { return None; }

    // ── dtype gate ────────────────────────────────────────────────────────────
    // All inputs must share dtype. Production paths: F32 (legacy) and BF16
    // (Workstream C). Mixed dtype = caller bug → fall back to 3-dispatch SDPA.
    let qkv_dtype = q.dtype();
    if k.dtype() != qkv_dtype || v.dtype() != qkv_dtype {
        return None;
    }
    if qkv_dtype != DType::F32 && qkv_dtype != DType::BF16 {
        return None;
    }
    let is_bf16 = qkv_dtype == DType::BF16;

    // SDPA-vector kernel is f32-only for now. Force FA2 path on bf16.
    let use_sdpav = sdpav_enabled() && !is_bf16;
    let pipeline = if is_bf16 {
        get_bf16_pipeline()?
    } else if use_sdpav {
        get_sdpav_pipeline().or_else(get_pipeline)?
    } else {
        get_pipeline()?
    };
    let pipeline_is_sdpav = !is_bf16 && use_sdpav && get_sdpav_pipeline().is_some();

    // ── Shape validation ──────────────────────────────────────────────────────
    let dims_q = q.dims();
    let dims_k = k.dims();
    let dims_v = v.dims();
    if dims_q.len() != 4 || dims_k.len() != 4 || dims_v.len() != 4 { return None; }
    let (b, h, sq, d)  = (dims_q[0] as u32, dims_q[1] as u32, dims_q[2] as u32, dims_q[3] as u32);
    let h_kv = dims_k[1] as u32;
    let skv  = dims_k[2] as u32;
    if d != 256 { return None; }   // kernel hardcoded for head_dim = 256

    // GQA: K/V may be at full H (group=1) or compressed H_kv (group=H/H_kv).
    // Reject non-divisible / KV-not-matching shapes so the caller falls back.
    if h_kv == 0 || h % h_kv != 0 { return None; }
    let group: u32 = h / h_kv;
    if dims_v[1] as u32 != h_kv { return None; }
    if dims_k[0] as u32 != b || dims_v[0] as u32 != b { return None; }
    if dims_v[2] as u32 != skv || dims_k[3] as u32 != d || dims_v[3] as u32 != d { return None; }

    // ── Metal device check ────────────────────────────────────────────────────
    let device = q.device().clone();
    let md = match &device {
        Device::Metal(md) => md,
        _ => return None,
    };

    // ── Force contiguous (copies only when strided/sliced) ────────────────────
    let to_cont = |t: &Tensor| -> candle_core::Result<Tensor> {
        if t.is_contiguous() { Ok(t.clone()) } else { t.contiguous() }
    };
    let q_c = match to_cont(q) { Ok(t) => t, Err(e) => return Some(Err(e)) };
    let k_c = match to_cont(k) { Ok(t) => t, Err(e) => return Some(Err(e)) };
    let v_c = match to_cont(v) { Ok(t) => t, Err(e) => return Some(Err(e)) };

    // ── Extract Metal buffers ─────────────────────────────────────────────────
    let (q_buf, q_off) = metal_buf(&q_c)?;
    let (k_buf, k_off) = metal_buf(&k_c)?;
    let (v_buf, v_off) = metal_buf(&v_c)?;

    // ── Output tensor (matches input dtype) ───────────────────────────────────
    let out = match Tensor::zeros(
        (b as usize, h as usize, sq as usize, d as usize),
        qkv_dtype,
        &device,
    ) {
        Ok(t) => t,
        Err(e) => return Some(Err(e)),
    };
    let (o_buf, o_off) = metal_buf(&out)?;

    // ── Optional mask (must match Q/K/V dtype) ────────────────────────────────
    let mask_c: Option<Tensor> = if let Some(m) = mask {
        let dims_m = m.dims();
        if dims_m.len() < 2 { return None; }
        let m_sq  = dims_m[dims_m.len() - 2] as u32;
        let m_skv = dims_m[dims_m.len() - 1] as u32;
        if m_sq != sq || m_skv != skv { return None; }
        if m.dtype() != qkv_dtype { return None; }
        match to_cont(m) { Ok(t) => Some(t), Err(e) => return Some(Err(e)) }
    } else {
        None
    };
    let mask_arg: Option<(&crate::metal::Buffer, usize)> =
        mask_c.as_ref().and_then(|mc| metal_buf(mc));

    // ── Encode into Candle's command buffer ───────────────────────────────────
    let encoder = match md
        .command_encoder()
        .map_err(|e| candle_core::Error::Msg(format!("flash_attn cmd_encoder: {e}")))
    {
        Ok(enc) => enc,
        Err(e)  => return Some(Err(e)),
    };

    if pipeline_is_sdpav {
        encode_sdpa_vector(
            encoder.as_ref(),
            pipeline,
            q_buf, q_off,
            k_buf, k_off,
            v_buf, v_off,
            o_buf, o_off,
            mask_arg,
            b, h, sq, skv,
            scale,
            group,
        );
    } else {
        // FA2 kernel — same dispatch geometry for f32 and bf16 (only the
        // pipeline differs; buffer layout, threads/TG, and grid are identical).
        encode_flash_attn(
            encoder.as_ref(),
            pipeline,
            q_buf, q_off,
            k_buf, k_off,
            v_buf, v_off,
            o_buf, o_off,
            mask_arg,
            b, h, sq, skv,
            scale,
            group,
        );
    }

    drop(encoder); // schedules on Candle's Metal command queue

    Some(Ok(out))
}
