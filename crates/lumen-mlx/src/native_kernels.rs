//! Model-agnostic compile-wrapped MLX kernels.
//!
//! This module holds small, pointwise fusion kernels that any decoder
//! architecture can call. They live separately from the layer-kind
//! modules (`native_ssm`, `native_moe`, `native_attention`, etc.) so a
//! new model bring-up (Llama, Mistral, …) can call into them without
//! depending on a particular layer-kind file.
//!
//! All kernels here are wrapped via [`native_compile_cache::CompiledMultiRefs`]
//! with `shapeless = true` — one mlx-c side compile entry is amortized across
//! the process lifetime and across all callers.
//!
//! Env gates are the canonical defaults from the native-pyo3 gap closure work
//! (see `notes/gap_closure_final.md` and `notes/sigmoid_mul_fuse_regression.md`):
//! - [`sigmoid_mul_fuse_enabled`]: `LUMEN_NATIVE_FUSE_SIGMOID_MUL=1`, default OFF
//!   (anti-pattern #30 calibration site — net REGRESSION at <50 fires/step
//!   without memory-locality bonus).

#[cfg(feature = "mlx-native")]
mod imp {
    use std::sync::{Mutex, OnceLock};

    use anyhow::Context;
    use mlx_rs::Array;
    use mlx_rs::error::Exception;

    use crate::native_compile_cache::{CompiledMultiRefs, invoke_compiled_multi_refs};

    fn sigmoid_mul_compiled_inner(args: &[Array]) -> std::result::Result<Vec<Array>, Exception> {
        let gate = &args[0];
        let other = &args[1];
        let sig = mlx_rs::ops::sigmoid(gate)?;
        let out = sig.multiply(other)?;
        Ok(vec![out])
    }

    static SIGMOID_MUL_SLOT: OnceLock<Mutex<CompiledMultiRefs>> = OnceLock::new();

    /// Whether [`sigmoid_mul_fused`] should be preferred over the explicit
    /// `sigmoid(gate) * other` two-op composition. Driven by
    /// `LUMEN_NATIVE_FUSE_SIGMOID_MUL=1`. Default OFF.
    ///
    /// History: tested as a global fuse for full-attn gate-mul and MoE
    /// shared-expert gate-mul in 2026-05-12. Result was decisive **REGRESSION**
    /// (+6.08σ) — anti-pattern #30 calibration site #3. The op fires ~24
    /// times/step, below the empirical break-even (~50 fires/step) at which
    /// compile dispatch CPU cost is recouped by saved Metal kernel work. Kept
    /// as an opt-in for future shapes where fire count rises above the
    /// break-even (e.g., expanded MoE shared-expert pattern). See
    /// `notes/sigmoid_mul_fuse_regression.md`.
    lumen_flags::flag! {
        /// Compile-wrapped `sigmoid(gate) * other`, bit-identical to the
        /// two-op composition. Default OFF (anti-pattern #30 calibration).
        /// (Parse note: previously only `=1` enabled this; the uniform rule
        /// now accepts any non-`"0"`.)
        pub(crate) sigmoid_mul_fuse {
            env: "LUMEN_NATIVE_FUSE_SIGMOID_MUL",
            default: false,
            kind: Optimization,
        }
    }

    pub fn sigmoid_mul_fuse_enabled() -> bool {
        sigmoid_mul_fuse::get()
    }

    /// Compile-wrapped `sigmoid(gate) * other` (shapeless). One mlx-c compile
    /// entry is shared across all callers in this process. Bit-identical to the
    /// explicit two-op `multiply(sigmoid(gate), other)` composition.
    pub fn sigmoid_mul_fused(gate: &Array, other: &Array) -> anyhow::Result<Array> {
        let args = [gate, other];
        let mut out = invoke_compiled_multi_refs(
            &SIGMOID_MUL_SLOT,
            sigmoid_mul_compiled_inner,
            /* shapeless */ true,
            &args,
        )
        .context("sigmoid_mul_fused: mlx compile dispatch failed")?;
        out.pop().context("sigmoid_mul_fused: empty output vec")
    }
}

#[cfg(feature = "mlx-native")]
pub(crate) use imp::{sigmoid_mul_fuse_enabled, sigmoid_mul_fused};
