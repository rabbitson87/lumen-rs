//! Native (Candle-bypass) forward path for Qwen3.5-VL-MoE.
//!
//! Path A of the 15+ tok/s roadmap: replace Candle Tensor ops with direct Metal
//! Buffer manipulation on a single command queue. The existing `qwen3_5_moe`
//! module remains the default forward path; this module is gated behind
//! `LUMEN_QWEN35MOE_NATIVE=1` for opt-in routing.
//!
//! See `memory/path_ab_combined_plan.md` for phase breakdown.

#[cfg(feature = "turboquant-gpu")]
pub mod bridge;
#[cfg(feature = "turboquant-gpu")]
pub mod context;
#[cfg(feature = "turboquant-gpu")]
pub mod forward;
#[cfg(feature = "turboquant-gpu")]
pub mod kernels;
#[cfg(feature = "turboquant-gpu")]
pub mod kv_cache;
#[cfg(feature = "turboquant-gpu")]
pub mod linear_attn;
#[cfg(feature = "turboquant-gpu")]
pub mod linear_state;
#[cfg(feature = "turboquant-gpu")]
pub mod self_attn;
#[cfg(feature = "turboquant-gpu")]
pub mod tensor;

#[cfg(feature = "turboquant-gpu")]
pub use bridge::{from_candle_tensor, from_candle_tensor_no_sync, to_candle_tensor};
#[cfg(feature = "turboquant-gpu")]
pub use context::NativeContext;
#[cfg(feature = "turboquant-gpu")]
pub use forward::placeholder_forward;
#[cfg(feature = "turboquant-gpu")]
pub use kernels::{KernelLib, build_rope_tables};
#[cfg(feature = "turboquant-gpu")]
pub use kv_cache::NativeKvCache;
#[cfg(feature = "turboquant-gpu")]
pub use linear_attn::{
    LinearAttnConfig, forward_linear_attn, forward_linear_attn_fused, forward_post_conv_fused,
    forward_post_conv_fused_with_cache, forward_post_conv_fused_with_cache_candle_queue,
    forward_ssm_loop,
};
#[cfg(feature = "turboquant-gpu")]
pub use linear_state::{NativeConvSnapshot, NativeConvState, NativeSsmSnapshot, NativeSsmState};
#[cfg(feature = "turboquant-gpu")]
pub use self_attn::{
    AttnBlockConfig, SelfAttnConfig, forward_attn_block, forward_attn_block_full,
    forward_self_attn, forward_self_attn_full, forward_self_attn_with_native_cache,
};
#[cfg(feature = "turboquant-gpu")]
pub use tensor::{NativeDType, NativeTensor};

/// True iff `LUMEN_QWEN35MOE_NATIVE=1` is set. Triggers the parity probe — runs
/// both Candle and native paths and reports cosine, but the production return value
/// is still Candle. Used for correctness validation, not for performance.
pub fn native_forward_enabled() -> bool {
    std::env::var("LUMEN_QWEN35MOE_NATIVE")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// True iff `LUMEN_NATIVE_OUTPUT=1` is set. **Substitutes** native output for
/// Candle in the production hot path: when conditions are safe (no past KV state,
/// no TurboQuant compressor), the native pipeline runs end-to-end and its result
/// is returned directly, skipping the Candle attention path entirely.
///
/// Unsafe states (`past_kv_len > 0`, `compressed_kv.is_some()`) silently fall back
/// to Candle so cached/compressed state stays consistent. Once decode-with-cache
/// native attention lands, this flag's coverage will widen.
pub fn native_output_enabled() -> bool {
    std::env::var("LUMEN_NATIVE_OUTPUT")
        .map(|v| v == "1")
        .unwrap_or(false)
}

#[cfg(feature = "turboquant-gpu")]
mod shared {
    use super::{KernelLib, NativeContext};
    use anyhow::Result;
    use std::sync::{Mutex, OnceLock};

    /// Lazily-initialized native context + kernel library shared across all native
    /// forward calls in a process. We only initialize once because:
    ///   - `NativeContext::new()` allocates a Metal device + command queue.
    ///   - `KernelLib::new()` compiles 9 Metal shaders (~10–50ms on M3 Max).
    /// Recreating either per-forward would dominate the timing budget.
    pub struct NativeResources {
        pub ctx: NativeContext,
        pub lib: KernelLib,
    }

    static NATIVE_RES: OnceLock<Mutex<NativeResources>> = OnceLock::new();

    /// Acquire the process-wide native context + kernel library, lazily creating
    /// them on first use. Uses `system_default` for the Metal device. Returns a
    /// `Mutex` so concurrent forward calls serialize access.
    pub fn shared_native_resources() -> Result<&'static Mutex<NativeResources>> {
        if let Some(m) = NATIVE_RES.get() {
            return Ok(m);
        }
        let ctx = NativeContext::new()?;
        let lib = KernelLib::new(&ctx)?;
        let _ = NATIVE_RES.set(Mutex::new(NativeResources { ctx, lib }));
        Ok(NATIVE_RES.get().expect("just set"))
    }

    /// Same as [`shared_native_resources`] but binds the underlying `MTLDevice`
    /// to the same one Candle is using on `candle_device`. **This is the path
    /// production wiring should take** — using `system_default()` returns a
    /// different `MTLDevice` handle (even on a single-GPU Mac), and Candle
    /// buffers on one handle are not visible to dispatches scheduled on the
    /// other (silent garbage / SIGSEGV).
    ///
    /// First-call wins: subsequent calls (regardless of `candle_device`) return
    /// the same shared resources.
    pub fn shared_native_resources_for(
        candle_device: &candle_core::Device,
    ) -> Result<&'static Mutex<NativeResources>> {
        if let Some(m) = NATIVE_RES.get() {
            return Ok(m);
        }
        let ctx = NativeContext::from_candle_device(candle_device)?;
        let lib = KernelLib::new(&ctx)?;
        let _ = NATIVE_RES.set(Mutex::new(NativeResources { ctx, lib }));
        Ok(NATIVE_RES.get().expect("just set"))
    }
}

#[cfg(feature = "turboquant-gpu")]
pub use shared::{NativeResources, shared_native_resources, shared_native_resources_for};
