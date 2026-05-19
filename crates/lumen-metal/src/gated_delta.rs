//! Gated Delta-Net SSM step kernel — port of MLX `gated_delta_step`.
//!
//! Replaces our Candle-based ops loop in `qwen3_5_moe::linear_attn::forward_inner`
//! (8+ Candle dispatches per timestep × 48 linear-attn layers ≈ 380 dispatches
//! per decode step) with a single Metal kernel.
//!
//! Returns `None` when any precondition fails (wrong device, wrong dtype, wrong
//! Dk alignment, or `LUMEN_DISABLE_GDN_KERNEL=1`) so the caller can fall back
//! to the Candle ops path.
//!
//! Algorithm (per timestep, identical to mlx_lm gated_delta_step ops):
//!   state *= g          // decay
//!   kv_mem = state · k  // simdgroup-reduced dot over Dk
//!   delta  = (v - kv_mem) * beta
//!   state += k ⊗ delta  // outer product update
//!   y      = state · q  // simdgroup-reduced dot over Dk

#[cfg(feature = "model-integration")]
use crate::metal::{BatchedEncoderExt, ComputeEncoderCompat};
use candle_core::{Device, Storage, Tensor};

use std::sync::{
    Once, OnceLock,
    atomic::{AtomicBool, Ordering},
};

const SHADER_SRC: &str = include_str!("shaders/gated_delta.metal");

struct SafePipeline(crate::metal::ComputePipelineState);
unsafe impl Send for SafePipeline {}
unsafe impl Sync for SafePipeline {}

static GDN_PIPELINE: OnceLock<SafePipeline> = OnceLock::new();
static GDN_FAILED: AtomicBool = AtomicBool::new(false);

static GDN_BF16_PIPELINE: OnceLock<SafePipeline> = OnceLock::new();
static GDN_BF16_FAILED: AtomicBool = AtomicBool::new(false);

// Default OFF until A/B validated — opt-in via `LUMEN_USE_GDN_KERNEL=1`.
static GDN_ENABLED: AtomicBool = AtomicBool::new(false);
static GDN_ENABLED_INIT: Once = Once::new();

fn init_enabled() {
    GDN_ENABLED_INIT.call_once(|| {
        if std::env::var("LUMEN_USE_GDN_KERNEL").as_deref() == Ok("1") {
            GDN_ENABLED.store(true, Ordering::Relaxed);
        }
    });
}

pub fn is_enabled() -> bool {
    init_enabled();
    GDN_ENABLED.load(Ordering::Relaxed)
}

/// Toggle the GDN kernel at runtime (used by A/B benchmarks).
pub fn set_enabled(on: bool) {
    init_enabled();
    GDN_ENABLED.store(on, Ordering::Relaxed);
}

fn compile_pipeline_named(name: &str) -> anyhow::Result<crate::metal::ComputePipelineState> {
    let device = crate::metal::Device::system_default()
        .ok_or_else(|| anyhow::anyhow!("gated_delta: no Metal GPU"))?;

    let opts = crate::metal::new_compile_options();
    // Bumped from 3.0 to 3.1 for bfloat support in tq_gated_delta_step_bf16.
    // The shared source compiles both kernels; bumping the f32 kernel's
    // language version too has zero functional effect (the f32 kernel uses
    // float only), confirmed against MLX 3.1 baseline 2026-05-08.
    opts.set_language_version(crate::metal::MTLLanguageVersion::Version3_1);
    opts.set_fast_math_enabled(true);

    let library = device
        .new_library_with_source(SHADER_SRC, Some(opts.as_ref()))
        .map_err(|e| anyhow::anyhow!("gated_delta Metal compile: {e}"))?;

    let func = library
        .get_function(name, None)
        .map_err(|e| anyhow::anyhow!("{name} not found: {e}"))?;

    device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| anyhow::anyhow!("{name} pipeline: {e}"))
}

fn compile_pipeline() -> anyhow::Result<crate::metal::ComputePipelineState> {
    compile_pipeline_named("tq_gated_delta_step")
}

fn compile_bf16_pipeline() -> anyhow::Result<crate::metal::ComputePipelineState> {
    compile_pipeline_named("tq_gated_delta_step_bf16")
}

fn get_pipeline() -> Option<&'static crate::metal::ComputePipelineState> {
    if GDN_FAILED.load(Ordering::Relaxed) {
        return None;
    }
    if let Some(p) = GDN_PIPELINE.get() {
        return Some(&p.0);
    }
    match compile_pipeline() {
        Ok(pl) => {
            let _ = GDN_PIPELINE.set(SafePipeline(pl));
            GDN_PIPELINE.get().map(|p| &p.0)
        }
        Err(e) => {
            eprintln!("gated_delta: pipeline compile failed ({e}); using Candle ops loop");
            GDN_FAILED.store(true, Ordering::Relaxed);
            None
        }
    }
}

fn get_bf16_pipeline() -> Option<&'static crate::metal::ComputePipelineState> {
    if GDN_BF16_FAILED.load(Ordering::Relaxed) {
        return None;
    }
    if let Some(p) = GDN_BF16_PIPELINE.get() {
        return Some(&p.0);
    }
    match compile_bf16_pipeline() {
        Ok(pl) => {
            let _ = GDN_BF16_PIPELINE.set(SafePipeline(pl));
            GDN_BF16_PIPELINE.get().map(|p| &p.0)
        }
        Err(e) => {
            eprintln!("gated_delta_bf16: pipeline compile failed ({e}); falling back to f32 path");
            GDN_BF16_FAILED.store(true, Ordering::Relaxed);
            None
        }
    }
}

#[cfg(feature = "model-integration")]
fn metal_buf(t: &Tensor) -> Option<(&crate::metal::Buffer, usize)> {
    let (storage, layout) = t.storage_and_layout();
    match &*storage {
        Storage::Metal(ms) => {
            let offset = layout.start_offset() * t.dtype().size_in_bytes();
            let buf: &crate::metal::Buffer =
                unsafe { &*(ms.buffer() as *const _ as *const crate::metal::Buffer) };
            Some((buf, offset))
        }
        _ => None,
    }
}

/// Inputs:
///   q, k       : [B, T, Hk, Dk]   — f32 OR bf16 (must match v/g/beta dtype)
///   v          : [B, T, Hv, Dv]   — f32 OR bf16
///   g, beta    : [B, T, Hv]       — f32 OR bf16
///   state_in   : [B, Hv, Dv, Dk]  — ALWAYS f32 (Escape #3 in MLX gated_delta.py)
/// Outputs:
///   y          : [B, T, Hv, Dv]   — same dtype as q/k/v/g/beta
///   state_out  : [B, Hv, Dv, Dk]  — ALWAYS f32
///
/// Returns `None` when preconditions fail (caller should fall back to ops loop).
///
/// MLX baseline (gated_delta.py:gated_delta_kernel — verified 2026-05-08):
///   template=[("InT", q.dtype), ("StT", state.dtype), ...]
///   InT ∈ {f32, bf16}, StT = f32 (recurrent state always f32 in MLX).
#[cfg(feature = "model-integration")]
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_step_candle(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state_in: &Tensor,
) -> Option<candle_core::Result<(Tensor, Tensor)>> {
    if !is_enabled() {
        return None;
    }

    // Detect input dtype — q/k/v/g/beta must agree; state_in must be f32.
    let in_dtype = q.dtype();
    if k.dtype() != in_dtype
        || v.dtype() != in_dtype
        || g.dtype() != in_dtype
        || beta.dtype() != in_dtype
    {
        return None;
    }
    if in_dtype != candle_core::DType::F32 && in_dtype != candle_core::DType::BF16 {
        return None;
    }
    if state_in.dtype() != candle_core::DType::F32 {
        return None; // state always f32 — Escape #3
    }
    let is_bf16 = in_dtype == candle_core::DType::BF16;

    // Pick pipeline based on input dtype.
    let pipeline = if is_bf16 {
        get_bf16_pipeline()?
    } else {
        get_pipeline()?
    };

    // Shape extraction
    let qd = q.dims();
    let vd = v.dims();
    if qd.len() != 4 || vd.len() != 4 {
        return None;
    }
    let (b, t, hk, dk) = (qd[0], qd[1], qd[2], qd[3]);
    let (hv, dv) = (vd[2], vd[3]);

    // Preconditions
    if dk % 32 != 0 {
        return None;
    }
    if hv == 0 || hk == 0 || hv % hk != 0 {
        return None;
    }
    let dk_state = state_in.dims();
    if dk_state.len() != 4
        || dk_state[0] != b
        || dk_state[1] != hv
        || dk_state[2] != dv
        || dk_state[3] != dk
    {
        return None;
    }
    if k.dims() != qd {
        return None;
    }
    let gd = g.dims();
    if gd.len() != 3 || gd[0] != b || gd[1] != t || gd[2] != hv {
        return None;
    }
    let bd = beta.dims();
    if bd != gd {
        return None;
    }

    // Device check
    let dev = q.device();
    if !matches!(dev, Device::Metal(_)) {
        return None;
    }

    // Allocate outputs. `y` matches input dtype (bf16 in bf16 path); state stays f32.
    let y = match Tensor::zeros(&[b, t, hv, dv], in_dtype, dev) {
        Ok(t) => t,
        Err(e) => return Some(Err(e)),
    };
    let state_out = match Tensor::zeros(&[b, hv, dv, dk], candle_core::DType::F32, dev) {
        Ok(t) => t,
        Err(e) => return Some(Err(e)),
    };

    let (q_buf, q_off) = metal_buf(q)?;
    let (k_buf, k_off) = metal_buf(k)?;
    let (v_buf, v_off) = metal_buf(v)?;
    let (g_buf, g_off) = metal_buf(g)?;
    let (beta_buf, beta_off) = metal_buf(beta)?;
    let (si_buf, si_off) = metal_buf(state_in)?;
    let (y_buf, y_off) = metal_buf(&y)?;
    let (so_buf, so_off) = metal_buf(&state_out)?;

    let metal_dev = match dev {
        Device::Metal(md) => md,
        _ => unreachable!(),
    };
    let encoder = match metal_dev
        .command_encoder()
        .map_err(|e| candle_core::Error::Msg(format!("gated_delta cmd_encoder: {e}")))
    {
        Ok(enc) => enc,
        Err(e) => return Some(Err(e)),
    };
    encoder.set_compute_pipeline_state(pipeline);

    encoder.set_buffer(0, Some(q_buf), q_off);
    encoder.set_buffer(1, Some(k_buf), k_off);
    encoder.set_buffer(2, Some(v_buf), v_off);
    encoder.set_buffer(3, Some(g_buf), g_off);
    encoder.set_buffer(4, Some(beta_buf), beta_off);
    encoder.set_buffer(5, Some(si_buf), si_off);
    encoder.set_buffer(6, Some(y_buf), y_off);
    encoder.set_buffer(7, Some(so_buf), so_off);

    let set_u32 = |idx: usize, v: u32| {
        let bytes = v.to_ne_bytes();
        encoder.set_bytes_directly(idx, 4, bytes.as_ptr() as *const _);
    };
    set_u32(8, t as u32);
    set_u32(9, dk as u32);
    set_u32(10, dv as u32);
    set_u32(11, hk as u32);
    set_u32(12, hv as u32);

    // Grid: (32 lanes/x, Dv/y, B*Hv/z). TG: (32, 4, 1) → 4 dv per TG along y.
    encoder.dispatch_thread_groups(
        crate::mtl_size!(1usize, dv.div_ceil(4), b * hv),
        crate::mtl_size!(32usize, 4, 1),
    );

    drop(encoder);
    Some(Ok((y, state_out)))
}
