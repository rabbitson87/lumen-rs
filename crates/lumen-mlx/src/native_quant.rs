//! Mode-aware MLX quantization wrappers.
//!
//! pmetal-mlx-rs 0.25.8's `dequantize_device` / `quantized_matmul_device` /
//! `quantize_device` hardcode `mode="affine"` (see `DEFAULT_MODE: &CStr =
//! c"affine"` in `pmetal-mlx-rs/src/ops/quantization.rs`). The production
//! Qwen3.6-35B-A3B-mxfp4 weights are MXFP4-quantized, so we need the FFI
//! C ABI's `mode` parameter set to `"mxfp4"`.
//!
//! Strategy:
//!   - Call `mlx_sys::mlx_dequantize/mlx_quantize/mlx_quantized_matmul`
//!     directly with a chosen mode string.
//!   - Use the public `Array::from_ptr(c_array: mlx_array) -> Array`
//!     constructor to wrap the resulting raw FFI handle.
//!   - Use `Array::as_ptr() -> mlx_array` for inputs.
//!
//! When upstream mlx-rs catches up and exposes `mode`, this module becomes
//! a thin alias.
//!
//! See `.ai/memory/active/mlx-rs-native-port/CONTEXT.md` Session 19 for
//! the discovery rationale.

#[cfg(feature = "mlx-native")]
mod imp {
    use anyhow::{Result, anyhow};
    use mlx_rs::Array;
    use std::ffi::CStr;

    /// MLX quantization modes supported by pmetal-mlx-sys's bundled MLX
    /// (v0.30.6). Names map to the C-side `mode` string parameter.
    #[allow(dead_code)]
    pub(crate) const MODE_AFFINE: &CStr = c"affine";
    #[allow(dead_code)]
    pub(crate) const MODE_MXFP4: &CStr = c"mxfp4";
    #[allow(dead_code)]
    pub(crate) const MODE_MXFP8: &CStr = c"mxfp8";
    #[allow(dead_code)]
    pub(crate) const MODE_NVFP4: &CStr = c"nvfp4";

    fn optional_int(value: i32) -> mlx_sys::mlx_optional_int {
        mlx_sys::mlx_optional_int {
            value,
            has_value: true,
        }
    }

    fn optional_dtype_none() -> mlx_sys::mlx_optional_dtype {
        mlx_sys::mlx_optional_dtype {
            value: 0,
            has_value: false,
        }
    }

    /// RAII handle for an `mlx_stream` we own. `mlx_default_gpu_stream_new`
    /// returns a heap-allocated wrapper around the singleton GPU stream;
    /// the FFI op extracts the underlying `mlx::core::Stream` synchronously,
    /// but we still own the wrapper and must free it.
    ///
    /// Note: `mlx_stream_new()` (no `_default_gpu_`) returns an *empty* handle
    /// — passing that to `mlx_dequantize` raises
    /// "expected a non-empty mlx_stream". Always go via the default-GPU/CPU
    /// stream constructors instead.
    #[allow(dead_code)]
    struct StreamGuard(mlx_sys::mlx_stream);

    #[allow(dead_code)]
    impl StreamGuard {
        fn gpu() -> Self {
            Self(unsafe { mlx_sys::mlx_default_gpu_stream_new() })
        }

        fn raw(&self) -> mlx_sys::mlx_stream {
            self.0
        }
    }

    impl Drop for StreamGuard {
        fn drop(&mut self) {
            unsafe {
                let _ = mlx_sys::mlx_stream_free(self.0);
            }
        }
    }

    /// Process-wide singleton GPU stream wrapper. `mlx_default_gpu_stream_new`
    /// returns a heap wrapper around the underlying singleton `mlx::core::Stream`;
    /// each per-call alloc+free amortises to ~1.5μs at single-op microbench
    /// (Step 2 Variant B, `notes/native_pyo3_step2_microbench_falsifies_dispatch_hypothesis.md`)
    /// but the real decode loop does ~120 quantized_matmul calls per step
    /// (in_proj × 4 × 30 layer) plus dequantize/gather_qmm — Lever 2 aggregate
    /// hypothesis is the per-call wrapper churn compounds across this volume.
    ///
    /// Wrapped in a Send+Sync newtype so OnceLock can hold the raw `mlx_stream`
    /// pointer (an opaque struct with an interior `mlx_object*`). The C ABI
    /// makes no thread-safety guarantee for the wrapper handle itself, but the
    /// underlying stream is the singleton and our usage pattern is read-only
    /// (we only pass the raw pointer to `mlx_*` ops; we never mutate the
    /// wrapper). Process-lifetime leak on shutdown is intentional — MLX's
    /// own scheduler holds a refcount on the singleton too.
    ///
    /// **WASH at 35B server long-prompt n=5 A/B (2026-05-03)**: per-call 166.42
    /// vs cached 164.47 tok/s = Δ -1.2% σ -0.96 (within noise floor). 32-step
    /// PyO3 golden parity PASS. The compounding 1.5μs × ~120 calls/step
    /// hypothesis falsified — wrapper alloc/free overlaps with GPU dispatch
    /// and is off the critical path. Singleton path retained as **opt-IN**
    /// (`LUMEN_NATIVE_CACHED_STREAM=1`); per-call remains default.
    struct SharedRaw(mlx_sys::mlx_stream);
    unsafe impl Send for SharedRaw {}
    unsafe impl Sync for SharedRaw {}

    fn cached_gpu_stream_raw() -> mlx_sys::mlx_stream {
        static CACHED: std::sync::OnceLock<SharedRaw> = std::sync::OnceLock::new();
        CACHED
            .get_or_init(|| SharedRaw(unsafe { mlx_sys::mlx_default_gpu_stream_new() }))
            .0
    }

    fn cached_empty_sentinel_raw() -> mlx_sys::mlx_array {
        static CACHED: std::sync::OnceLock<SharedArrayRaw> = std::sync::OnceLock::new();
        struct SharedArrayRaw(mlx_sys::mlx_array);
        unsafe impl Send for SharedArrayRaw {}
        unsafe impl Sync for SharedArrayRaw {}
        CACHED
            .get_or_init(|| SharedArrayRaw(unsafe { mlx_sys::mlx_array_new() }))
            .0
    }

    /// Returns true (default) when each FFI quant op constructs its own
    /// `StreamGuard::gpu()` + `OwnedEmptyArray` sentinels. Set
    /// `LUMEN_NATIVE_CACHED_STREAM=1` to opt in to the OnceLock singleton
    /// path (Lever 2 infrastructure, A/B WASH 2026-05-03).
    fn use_per_call_stream() -> bool {
        std::env::var("LUMEN_NATIVE_CACHED_STREAM")
            .map(|v| v != "1")
            .unwrap_or(true)
    }

    /// RAII guard for the temporary "no biases" sentinel handle. When the
    /// caller passes `biases=None`, the C ABI still requires a value; we hand
    /// it the empty handle from `mlx_array_new()` (ctx=null), which the C
    /// dispatch interprets as `std::nullopt`. The handle still needs freeing
    /// when we synthesised it ourselves.
    struct OwnedEmptyArray(mlx_sys::mlx_array);

    impl OwnedEmptyArray {
        fn new() -> Self {
            Self(unsafe { mlx_sys::mlx_array_new() })
        }

        fn raw(&self) -> mlx_sys::mlx_array {
            self.0
        }
    }

    impl Drop for OwnedEmptyArray {
        fn drop(&mut self) {
            unsafe {
                let _ = mlx_sys::mlx_array_free(self.0);
            }
        }
    }

    fn biases_raw<'a>(
        biases: Option<&'a Array>,
        sentinel: &'a OwnedEmptyArray,
    ) -> mlx_sys::mlx_array {
        match biases {
            Some(b) => b.as_ptr(),
            None => sentinel.raw(),
        }
    }

    /// `Array` ↔ raw FFI bridge guard. Allocates an empty `mlx_array` to
    /// receive the FFI output, hands a `*mut mlx_array` to the caller's
    /// closure, and on success wraps the populated handle into a safe
    /// `Array`. On failure, frees the handle to avoid leaks.
    ///
    /// Mirrors the essence of pmetal-mlx-rs's pub(crate) `Guarded::try_from_op`
    /// — we can't use it directly because the trait + helpers aren't exposed.
    fn array_from_ffi<F>(f: F) -> Result<Array>
    where
        F: FnOnce(*mut mlx_sys::mlx_array) -> i32,
    {
        // G6-G op-count: bump the shared mlx-rs counter so the mxfp4 mlx_sys
        // direct-FFI path is included in `LUMEN_NATIVE_COUNT_OPS=1` totals.
        // Without this, mxfp4 quantized_matmul / gather_qmm / dequantize calls
        // would not be counted, undercounting the true mlx-c entry total.
        mlx_rs::utils::OP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        unsafe {
            let mut raw: mlx_sys::mlx_array = mlx_sys::mlx_array_new();
            let status = f(&mut raw as *mut mlx_sys::mlx_array);
            if status != 0 {
                let _ = mlx_sys::mlx_array_free(raw);
                return Err(anyhow!("mlx FFI quantization op returned status {status}"));
            }
            // Array::from_ptr takes ownership of the raw mlx_array.
            Ok(Array::from_ptr(raw))
        }
    }

    /// Dequantize MXFP4/affine/etc. weights with explicit mode.
    ///
    /// Matches the Python `mx.dequantize(w, scales, biases=None,
    /// group_size, bits, mode="mxfp4")` API.
    ///
    /// MXFP4 weights have no `biases` tensor — pass `None` and we fill an
    /// empty `mlx_array`.
    pub fn dequantize_with_mode(
        w: &Array,
        scales: &Array,
        biases: Option<&Array>,
        group_size: i32,
        bits: i32,
        mode: &CStr,
    ) -> Result<Array> {
        if use_per_call_stream() {
            let stream = StreamGuard::gpu();
            let sentinel = OwnedEmptyArray::new();
            let biases_ptr = biases_raw(biases, &sentinel);
            return array_from_ffi(|res| unsafe {
                mlx_sys::mlx_dequantize(
                    res,
                    w.as_ptr(),
                    scales.as_ptr(),
                    biases_ptr,
                    optional_int(group_size),
                    optional_int(bits),
                    mode.as_ptr(),
                    optional_dtype_none(),
                    stream.raw(),
                )
            });
        }
        let stream_raw = cached_gpu_stream_raw();
        let biases_ptr = match biases {
            Some(b) => b.as_ptr(),
            None => cached_empty_sentinel_raw(),
        };
        array_from_ffi(|res| unsafe {
            mlx_sys::mlx_dequantize(
                res,
                w.as_ptr(),
                scales.as_ptr(),
                biases_ptr,
                optional_int(group_size),
                optional_int(bits),
                mode.as_ptr(),
                optional_dtype_none(),
                stream_raw,
            )
        })
    }

    /// Quantized matmul `y = x @ dequantize(w, scales, biases, group_size, bits, mode)`.
    /// `transpose=true` performs `y = x @ deq(w).T` (the typical Linear layout
    /// for transposed weights).
    pub fn quantized_matmul_with_mode(
        x: &Array,
        w: &Array,
        scales: &Array,
        biases: Option<&Array>,
        transpose: bool,
        group_size: i32,
        bits: i32,
        mode: &CStr,
    ) -> Result<Array> {
        if use_per_call_stream() {
            let stream = StreamGuard::gpu();
            let sentinel = OwnedEmptyArray::new();
            let biases_ptr = biases_raw(biases, &sentinel);
            return array_from_ffi(|res| unsafe {
                mlx_sys::mlx_quantized_matmul(
                    res,
                    x.as_ptr(),
                    w.as_ptr(),
                    scales.as_ptr(),
                    biases_ptr,
                    transpose,
                    optional_int(group_size),
                    optional_int(bits),
                    mode.as_ptr(),
                    stream.raw(),
                )
            });
        }
        let stream_raw = cached_gpu_stream_raw();
        let biases_ptr = match biases {
            Some(b) => b.as_ptr(),
            None => cached_empty_sentinel_raw(),
        };
        array_from_ffi(|res| unsafe {
            mlx_sys::mlx_quantized_matmul(
                res,
                x.as_ptr(),
                w.as_ptr(),
                scales.as_ptr(),
                biases_ptr,
                transpose,
                optional_int(group_size),
                optional_int(bits),
                mode.as_ptr(),
                stream_raw,
            )
        })
    }

    /// `gather_qmm` with explicit mode — quantized matmul with batch-axis
    /// gather indices. Mirrors Python `mx.gather_qmm(x, w, scales, biases,
    /// lhs_indices, rhs_indices, transpose, group_size, bits, mode,
    /// sorted_indices)`.
    ///
    /// Used by the MoE `switch_mlp` path (Phase 3d.4): one batched matmul
    /// per (gate_proj, up_proj, down_proj), where `rhs_indices` selects
    /// the per-token expert. mlx-rs 0.25.8 only exposes a hardcoded-mode
    /// `gather_qmm`, so we drop down to the FFI to pass `mode="mxfp4"`.
    pub fn gather_qmm_with_mode(
        x: &Array,
        w: &Array,
        scales: &Array,
        biases: Option<&Array>,
        lhs_indices: Option<&Array>,
        rhs_indices: Option<&Array>,
        transpose: bool,
        group_size: i32,
        bits: i32,
        mode: &CStr,
        sorted_indices: bool,
    ) -> Result<Array> {
        if use_per_call_stream() {
            let stream = StreamGuard::gpu();
            let biases_sentinel = OwnedEmptyArray::new();
            let lhs_sentinel = OwnedEmptyArray::new();
            let rhs_sentinel = OwnedEmptyArray::new();
            let biases_ptr = biases_raw(biases, &biases_sentinel);
            let lhs_ptr = match lhs_indices {
                Some(idx) => idx.as_ptr(),
                None => lhs_sentinel.raw(),
            };
            let rhs_ptr = match rhs_indices {
                Some(idx) => idx.as_ptr(),
                None => rhs_sentinel.raw(),
            };
            return array_from_ffi(|res| unsafe {
                mlx_sys::mlx_gather_qmm(
                    res,
                    x.as_ptr(),
                    w.as_ptr(),
                    scales.as_ptr(),
                    biases_ptr,
                    lhs_ptr,
                    rhs_ptr,
                    transpose,
                    optional_int(group_size),
                    optional_int(bits),
                    mode.as_ptr(),
                    sorted_indices,
                    stream.raw(),
                )
            });
        }
        let stream_raw = cached_gpu_stream_raw();
        let sentinel = cached_empty_sentinel_raw();
        let biases_ptr = match biases {
            Some(b) => b.as_ptr(),
            None => sentinel,
        };
        let lhs_ptr = match lhs_indices {
            Some(idx) => idx.as_ptr(),
            None => sentinel,
        };
        let rhs_ptr = match rhs_indices {
            Some(idx) => idx.as_ptr(),
            None => sentinel,
        };
        array_from_ffi(|res| unsafe {
            mlx_sys::mlx_gather_qmm(
                res,
                x.as_ptr(),
                w.as_ptr(),
                scales.as_ptr(),
                biases_ptr,
                lhs_ptr,
                rhs_ptr,
                transpose,
                optional_int(group_size),
                optional_int(bits),
                mode.as_ptr(),
                sorted_indices,
                stream_raw,
            )
        })
    }

    /// Quantize a dense matrix `w` with explicit mode. Returns
    /// `(packed_weights, scales, biases_opt)` — `biases_opt` is `Some` for
    /// AFFINE (matches Python `mx.quantize`'s 3-tuple) and `None` for
    /// MXFP4 (block-scaled exponents, no per-group bias term — mlx-c
    /// returns just `[packed, scales]`).
    ///
    /// Used by the Qwen3.5 MTP loader (HF original ships bf16 mtp.* weights;
    /// we quantize at startup) and by dev/test microbenches.
    pub fn quantize_with_mode(
        w: &Array,
        group_size: i32,
        bits: i32,
        mode: &CStr,
    ) -> Result<(Array, Array, Option<Array>)> {
        let stream_raw = cached_gpu_stream_raw();
        unsafe {
            let mut vec_raw: mlx_sys::mlx_vector_array = mlx_sys::mlx_vector_array_new();
            let status = mlx_sys::mlx_quantize(
                &mut vec_raw as *mut mlx_sys::mlx_vector_array,
                w.as_ptr(),
                optional_int(group_size),
                optional_int(bits),
                mode.as_ptr(),
                stream_raw,
            );
            if status != 0 {
                let _ = mlx_sys::mlx_vector_array_free(vec_raw);
                return Err(anyhow!("mlx_quantize returned status {status}"));
            }

            let len = mlx_sys::mlx_vector_array_size(vec_raw);
            if len != 2 && len != 3 {
                let _ = mlx_sys::mlx_vector_array_free(vec_raw);
                return Err(anyhow!(
                    "mlx_quantize produced {len} arrays, expected 2 (MXFP4) or 3 (AFFINE)"
                ));
            }

            let mut packed: mlx_sys::mlx_array = mlx_sys::mlx_array_new();
            let mut scales: mlx_sys::mlx_array = mlx_sys::mlx_array_new();
            let s0 = mlx_sys::mlx_vector_array_get(&mut packed, vec_raw, 0);
            let s1 = mlx_sys::mlx_vector_array_get(&mut scales, vec_raw, 1);
            let biases_opt = if len == 3 {
                let mut biases: mlx_sys::mlx_array = mlx_sys::mlx_array_new();
                let s2 = mlx_sys::mlx_vector_array_get(&mut biases, vec_raw, 2);
                if s2 != 0 {
                    let _ = mlx_sys::mlx_array_free(packed);
                    let _ = mlx_sys::mlx_array_free(scales);
                    let _ = mlx_sys::mlx_array_free(biases);
                    let _ = mlx_sys::mlx_vector_array_free(vec_raw);
                    return Err(anyhow!("mlx_vector_array_get biases failed: status {s2}"));
                }
                Some(Array::from_ptr(biases))
            } else {
                None
            };
            let _ = mlx_sys::mlx_vector_array_free(vec_raw);
            if s0 != 0 || s1 != 0 {
                let _ = mlx_sys::mlx_array_free(packed);
                let _ = mlx_sys::mlx_array_free(scales);
                return Err(anyhow!("mlx_vector_array_get failed: statuses {s0}/{s1}"));
            }
            Ok((Array::from_ptr(packed), Array::from_ptr(scales), biases_opt))
        }
    }
}

#[cfg(feature = "mlx-native")]
#[allow(unused_imports)] // Consumed by Phase 3b wiring in runner_native.rs.
pub(crate) use imp::{
    MODE_AFFINE, MODE_MXFP4, MODE_MXFP8, MODE_NVFP4, dequantize_with_mode, gather_qmm_with_mode,
    quantize_with_mode, quantized_matmul_with_mode,
};

// NVFP4 FFI support smoke test — the decisive check for the gemma4 nvfp4 path.
//
// gemma4 forward dispatches `quantized_matmul_with_mode(..., MODE_NVFP4)` straight to
// `mlx_sys::mlx_quantized_matmul`. Whether nvfp4 works end-to-end therefore hinges entirely
// on whether the linked mlx-c backend accepts `mode="nvfp4"`. If quantize+dequantize succeed
// here, the config wiring (gemma4_moe.rs mode whitelist + quant_params_for) is sufficient and
// no custom Metal kernel is needed for correctness.
//   cargo test -p lumen-mlx --features mlx-native nvfp4_ffi_roundtrip -- --nocapture
#[cfg(all(test, feature = "mlx-native"))]
mod nvfp4_ffi_smoke {
    use super::imp::{
        MODE_NVFP4, dequantize_with_mode, quantize_with_mode, quantized_matmul_with_mode,
    };
    use mlx_rs::Array;

    #[test]
    fn nvfp4_ffi_roundtrip() {
        let rows = 16i32; // out_features
        let cols = 32i32; // in_features (multiple of group 16)
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.137).sin() * 2.0)
            .collect();
        let w = Array::from_slice(&data, &[rows, cols]);

        // (1) quantize(mode=nvfp4) — does the backend accept the mode at all?
        let (packed, scales, biases) = match quantize_with_mode(&w, 16, 4, MODE_NVFP4) {
            Ok(t) => t,
            Err(e) => {
                panic!("mlx FFI quantize(mode=nvfp4) FAILED — backend does not support nvfp4: {e}")
            }
        };
        assert_eq!(packed.shape(), &[rows, cols / 8], "nvfp4 packed shape");
        assert_eq!(
            scales.shape(),
            &[rows, cols / 16],
            "nvfp4 scales (group 16)"
        );
        assert!(biases.is_none(), "nvfp4 is single-level — no biases");

        // (2) dequantize(mode=nvfp4) — reconstruction sanity.
        let deq = dequantize_with_mode(&packed, &scales, None, 16, 4, MODE_NVFP4)
            .expect("mlx FFI dequantize(mode=nvfp4) must succeed");
        let out = deq
            .as_dtype(mlx_rs::Dtype::Float32)
            .expect("nvfp4 dequant → f32 cast");
        out.eval().expect("mlx eval");
        assert_eq!(out.shape(), &[rows, cols], "nvfp4 dequant shape");
        let recon: &[f32] = out.as_slice();
        let (mut num, mut den) = (0.0f32, 0.0f32);
        for (a, b) in recon.iter().zip(&data) {
            num += (a - b).abs();
            den += b.abs();
        }
        let rel = num / den;
        assert!(
            rel < 0.25,
            "nvfp4 reconstruction rel-MAE {rel} too high (~0.1 expected)"
        );

        // (3) quantized_matmul(mode=nvfp4) — the EXACT op gemma4 attention/lm_head dispatches.
        // y[b, o] = sum_k dequant(W)[o, k] * x[b, k]  (transpose=true). batch=2.
        let xb: Vec<f32> = (0..2 * cols).map(|i| (i as f32 * 0.05) - 0.7).collect();
        let x = Array::from_slice(&xb, &[2, cols]);
        let y = quantized_matmul_with_mode(&x, &packed, &scales, None, true, 16, 4, MODE_NVFP4)
            .expect("mlx FFI quantized_matmul(mode=nvfp4) must succeed — gemma4 forward path");
        let yf = y.as_dtype(mlx_rs::Dtype::Float32).expect("y → f32");
        yf.eval().expect("eval y");
        assert_eq!(yf.shape(), &[2, rows], "nvfp4 matmul shape [batch, out]");

        // Cross-check matmul against dequant-then-dot reference.
        let ys: &[f32] = yf.as_slice();
        let mut max_rel = 0.0f32;
        for b in 0..2usize {
            for o in 0..rows as usize {
                let want: f32 = (0..cols as usize)
                    .map(|k| recon[o * cols as usize + k] * xb[b * cols as usize + k])
                    .sum();
                let got = ys[b * rows as usize + o];
                let r = (got - want).abs() / want.abs().max(1e-3);
                max_rel = max_rel.max(r);
            }
        }
        eprintln!("nvfp4 FFI OK — dequant rel-MAE {rel:.4}, matmul max-rel {max_rel:.4}");
        assert!(
            max_rel < 0.05,
            "nvfp4 quantized_matmul diverged from reference: {max_rel}"
        );
    }
}

// NVFP4 kernel-vs-FFI A/B — does a custom fused Metal kernel have any headroom over
// mlx's quantized_matmul(NVFP4) for Gemma4-26B-A4B MoE shapes?
//
// Gate-check before committing to a fused gate_up_silu_mul kernel + forward wiring: if the
// EXISTING custom matvec is already slower per-matmul than the mlx FFI path, fusion (which only
// trims dispatch count while ADDING register pressure) cannot recover the deficit. Both sides
// run device-resident weights in ONE process at the real Gemma4 MoE shape so the comparison is
// within-run (no cross-session thermal drift).
//
//   cargo test -p lumen-mlx --features mlx-native-metal nvfp4_kernel_vs_ffi -- --ignored --nocapture
#[cfg(all(test, feature = "mlx-native"))]
mod nvfp4_kernel_vs_ffi_ab {
    use super::imp::{MODE_NVFP4, quantize_with_mode, quantized_matmul_with_mode};
    use lumen_metal::nvfp4::dequantize_f32;
    use lumen_metal::nvfp4_gpu::Nvfp4Context;
    use mlx_rs::Array;
    use std::time::Instant;

    /// Run one (out, in) shape: quantize once, then time mlx FFI matmul vs the custom kernel,
    /// both with device-resident weights. Returns (mlx_ms, custom_ms).
    fn ab_one(
        ctx: &Nvfp4Context,
        out: usize,
        in_f: usize,
        warmup: usize,
        iters: usize,
    ) -> (f64, f64) {
        assert_eq!(in_f % 16, 0, "in must be group-aligned");
        // Deterministic pseudo-random weight in row-major [out, in].
        let wdata: Vec<f32> = (0..out * in_f)
            .map(|i| ((i as f32) * 0.0007).sin() * 1.5)
            .collect();
        let w = Array::from_slice(&wdata, &[out as i32, in_f as i32]);
        let (packed, scales, biases) =
            quantize_with_mode(&w, 16, 4, MODE_NVFP4).expect("quantize nvfp4");
        assert!(biases.is_none());
        packed.eval().unwrap();
        scales.eval().unwrap();
        let packed_host: Vec<u32> = packed.as_slice::<u32>().to_vec();
        let scales_host: Vec<u8> = scales.as_slice::<u8>().to_vec();

        // Shared decode-time activation: batch=1.
        let xdata: Vec<f32> = (0..in_f)
            .map(|i| ((i as f32) * 0.013).cos() * 0.8)
            .collect();

        // --- mlx FFI path (weights resident as Arrays after first eval) ---
        let x = Array::from_slice(&xdata, &[1, in_f as i32]);
        let mlx_call = || {
            let y = quantized_matmul_with_mode(&x, &packed, &scales, None, true, 16, 4, MODE_NVFP4)
                .expect("mlx qmm nvfp4");
            y.eval().unwrap();
            y
        };
        for _ in 0..warmup {
            mlx_call();
        }
        let t0 = Instant::now();
        let mut last_mlx = None;
        for _ in 0..iters {
            last_mlx = Some(mlx_call());
        }
        let mlx_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        // --- custom Metal kernel (weights resident, dispatch-only timing) ---
        let (custom_ms, custom_y) = ctx
            .bench_matvec(&packed_host, &scales_host, &xdata, out, in_f, warmup, iters)
            .expect("bench_matvec");

        // --- correctness: both vs CPU dequant-then-dot reference ---
        let mut wref = vec![0.0f32; out * in_f];
        dequantize_f32(&packed_host, &scales_host, &mut wref).unwrap();
        let yref: Vec<f32> = (0..out)
            .map(|o| (0..in_f).map(|k| wref[o * in_f + k] * xdata[k]).sum())
            .collect();
        let rel = |got: &[f32]| -> f32 {
            got.iter()
                .zip(&yref)
                .map(|(g, r)| (g - r).abs() / r.abs().max(1e-3))
                .fold(0.0f32, f32::max)
        };
        let mlx_y = last_mlx.unwrap();
        let mlx_yf = mlx_y.as_dtype(mlx_rs::Dtype::Float32).unwrap();
        mlx_yf.eval().unwrap();
        let mlx_rel = rel(mlx_yf.as_slice::<f32>());
        let custom_rel = rel(&custom_y);
        assert!(mlx_rel < 0.05, "mlx diverged: {mlx_rel}");
        assert!(custom_rel < 0.05, "custom kernel diverged: {custom_rel}");

        eprintln!(
            "  shape out={out:>5} in={in_f:>5}  mlx={mlx_ms:.4} ms  custom={custom_ms:.4} ms  \
             custom/mlx={:.2}x  (rel mlx={mlx_rel:.4} custom={custom_rel:.4})",
            custom_ms / mlx_ms
        );
        (mlx_ms, custom_ms)
    }

    #[test]
    #[ignore = "requires Metal device; run manually with --ignored on Apple Silicon"]
    fn nvfp4_kernel_vs_ffi() {
        let ctx = match Nvfp4Context::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("no Metal device, skipping: {e}");
                return;
            }
        };
        let (warmup, iters) = (30usize, 300usize);
        // Gemma4-26B-A4B MoE expert projections (hidden=2816, moe_intermediate=704, nvfp4 g16):
        //   gate/up: out=704  in=2816   |   down: out=2816 in=704
        // Plus the dense MLP (intermediate=4304) and a square baseline for context.
        eprintln!("NVFP4 custom-kernel vs mlx-FFI quantized_matmul, batch=1 decode:");
        let shapes = [
            (704usize, 2816usize),
            (2816, 704),
            (4304, 2816),
            (2816, 4304),
            (2816, 2816),
        ];
        let mut tot_mlx = 0.0;
        let mut tot_custom = 0.0;
        for (o, i) in shapes {
            let (m, c) = ab_one(&ctx, o, i, warmup, iters);
            tot_mlx += m;
            tot_custom += c;
        }
        eprintln!(
            "TOTAL  mlx={tot_mlx:.3} ms  custom={tot_custom:.3} ms  custom/mlx={:.2}x",
            tot_custom / tot_mlx
        );
        eprintln!(
            "VERDICT: a fused gate_up_silu_mul kernel only saves DISPATCH count; if custom/mlx>1 \
             per-matmul, fusion cannot win (it adds register pressure). custom/mlx<1 would justify \
             building the fused kernel."
        );
    }
}

// Single-tensor MXFP4 dequant parity tests against MLX's reference.
//
// These reuse the pre-baked MXFP4 fixtures under
// `crates/lumen-metal/tests/fixtures/` (produced by
// `scripts/generate_mxfp4_fixture.py` via `mx.dequantize(..., mode="mxfp4")`).
// The reference `.ref` payloads come from MLX itself, so a successful
// FFI round-trip through `dequantize_with_mode(..., MODE_MXFP4)` must be
// bit-identical when the underlying mlx-sys/mlx-rs bridge is wired correctly.
//
// They are `#[ignore]`'d by default — MLX FFI requires a host with a Metal
// device (see Issue #5 in active/mlx-rs-native-port/CHECKLIST.md). Run
// manually outside the Codex sandbox:
//   cargo test -p lumen-mlx --features mlx-native -- --ignored
#[cfg(all(test, feature = "mlx-native"))]
mod parity_tests {
    use super::imp::{MODE_MXFP4, dequantize_with_mode, quantized_matmul_with_mode};
    use mlx_rs::Array;
    use std::path::{Path, PathBuf};

    const MAGIC: &[u8; 4] = b"MXFP";
    const HEADER_LEN: usize = 12;
    const GROUP_SIZE: i32 = 32;
    const BITS: i32 = 4;
    const MATMUL_MAGIC: &[u8; 4] = b"TQMM";
    const MATMUL_HEADER_LEN: usize = 4 /* magic */ + 8 * 4 /* 8 u32 fields */;

    fn fixture_dir() -> PathBuf {
        // CARGO_MANIFEST_DIR = .../crates/lumen-mlx
        // fixtures are checked in under lumen-metal so we share them
        // across both the Metal kernel parity test and this FFI parity test.
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("lumen-mlx manifest dir has a parent (crates/)")
            .join("lumen-metal")
            .join("tests")
            .join("fixtures")
    }

    fn read_fixture_payload(path: &Path) -> Option<(usize, usize, Vec<u8>)> {
        let bytes = std::fs::read(path).ok()?;
        if bytes.len() < HEADER_LEN || &bytes[..4] != MAGIC {
            panic!("fixture {path:?} has bad header");
        }
        let rows = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let cols = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        Some((rows, cols, bytes[HEADER_LEN..].to_vec()))
    }

    fn load_fixture(name: &str) -> Option<(usize, usize, Vec<u32>, Vec<u8>, Vec<f32>)> {
        let dir = fixture_dir();
        let (rows, cols, packed_bytes) =
            read_fixture_payload(&dir.join(format!("mxfp4_{name}.weight")))?;
        let (srows, scols, scale_bytes) =
            read_fixture_payload(&dir.join(format!("mxfp4_{name}.scales")))?;
        let (rrows, rcols, ref_bytes) =
            read_fixture_payload(&dir.join(format!("mxfp4_{name}.ref")))?;

        assert_eq!((rows, cols), (srows, scols), "weight/scales shape mismatch");
        assert_eq!((rows, cols), (rrows, rcols), "weight/ref shape mismatch");

        let packed: Vec<u32> = packed_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let reference: Vec<f32> = ref_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        Some((rows, cols, packed, scale_bytes, reference))
    }

    fn run_fixture(name: &str) {
        let Some((rows, cols, packed, scales, reference)) = load_fixture(name) else {
            eprintln!(
                "SKIP {name}: fixtures missing under {}. Run `python scripts/generate_mxfp4_fixture.py` first.",
                fixture_dir().display()
            );
            return;
        };

        let n = rows * cols;
        assert_eq!(packed.len(), n / 8, "{name}: packed length unexpected");
        assert_eq!(scales.len(), n / 32, "{name}: scales length unexpected");
        assert_eq!(reference.len(), n, "{name}: reference length unexpected");

        // mlx_dequantize expects the packed weight as uint32 with shape [rows, cols/8]
        // and scales as uint8 (E8M0 exponents) with shape [rows, cols/32].
        let w = Array::from_slice(&packed, &[rows as i32, (cols / 8) as i32]);
        let s = Array::from_slice(&scales, &[rows as i32, (cols / 32) as i32]);

        // mxfp4 dequant produces bf16 by default. The fixture's `.ref` was
        // generated by `mx.dequantize(...).astype(mx.float32)`, so we follow
        // the same bf16 → f32 cast path before bit-identical comparison.
        let bf16 = dequantize_with_mode(&w, &s, None, GROUP_SIZE, BITS, MODE_MXFP4)
            .expect("mlx_dequantize FFI call must succeed");
        assert_eq!(
            bf16.shape(),
            &[rows as i32, cols as i32],
            "{name}: dequant shape mismatch"
        );
        assert_eq!(
            bf16.dtype(),
            mlx_rs::Dtype::Bfloat16,
            "{name}: mxfp4 dequant default dtype unexpectedly changed"
        );

        let out = bf16
            .as_dtype(mlx_rs::Dtype::Float32)
            .expect("bf16 → f32 cast must succeed");
        out.eval().expect("mlx eval must succeed");

        assert_eq!(
            out.shape(),
            &[rows as i32, cols as i32],
            "{name}: cast shape mismatch"
        );
        assert_eq!(
            out.dtype(),
            mlx_rs::Dtype::Float32,
            "{name}: post-cast dtype not Float32"
        );

        let observed: &[f32] = out.as_slice();
        assert_eq!(
            observed.len(),
            reference.len(),
            "{name}: flat-size mismatch"
        );

        let mut mismatches = 0usize;
        for (i, (&got, &expected)) in observed.iter().zip(reference.iter()).enumerate() {
            if got.to_bits() != expected.to_bits() {
                if mismatches < 5 {
                    eprintln!(
                        "{name}[{i}]: mlx-rs FFI 0x{:08x} ({got}) vs MLX ref 0x{:08x} ({expected})",
                        got.to_bits(),
                        expected.to_bits()
                    );
                }
                mismatches += 1;
            }
        }
        assert_eq!(
            mismatches,
            0,
            "{name}: {mismatches}/{} elements diverged from MLX reference",
            observed.len()
        );
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device; run with `cargo test -p lumen-mlx --features mlx-native -- --ignored`"]
    fn single_group_dequant_matches_mlx() {
        run_fixture("single_group");
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn multi_group_dequant_matches_mlx() {
        run_fixture("multi_group");
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn dynamic_range_dequant_matches_mlx() {
        run_fixture("dynamic_range");
    }

    // ─── Phase 3a.3: quantized_matmul parity ────────────────────────────────
    //
    // The matmul fixture file (`mxfp4_matmul_<name>.bin`) is produced by
    // `scripts/generate_mxfp4_matmul_fixture.py`. Header is 4-byte magic
    // `TQMM` followed by 8 u32 fields (out_features, in_features,
    // group_size, bits, x_len, packed_len, scales_len, y_len), then the
    // payloads in order: x f32, packed u32, scales u8, expected_y f32.

    struct MatmulFixture {
        out_features: usize,
        in_features: usize,
        group_size: i32,
        bits: i32,
        x: Vec<f32>,
        packed: Vec<u32>,
        scales: Vec<u8>,
        expected_y: Vec<f32>,
    }

    fn load_matmul_fixture(name: &str) -> Option<MatmulFixture> {
        let path = fixture_dir().join(format!("mxfp4_matmul_{name}.bin"));
        let bytes = std::fs::read(&path).ok()?;
        if bytes.len() < MATMUL_HEADER_LEN || &bytes[..4] != MATMUL_MAGIC {
            panic!("matmul fixture {path:?} has bad header");
        }
        let read_u32 = |offset: usize| {
            u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize
        };
        let out_features = read_u32(4);
        let in_features = read_u32(8);
        let group_size = read_u32(12) as i32;
        let bits = read_u32(16) as i32;
        let x_len = read_u32(20);
        let packed_len = read_u32(24);
        let scales_len = read_u32(28);
        let y_len = read_u32(32);

        assert_eq!(x_len, in_features, "{name}: x_len mismatch");
        assert_eq!(
            packed_len,
            out_features * in_features / 8,
            "{name}: packed_len mismatch"
        );
        assert_eq!(
            scales_len,
            out_features * in_features / 32,
            "{name}: scales_len mismatch"
        );
        assert_eq!(y_len, out_features, "{name}: y_len mismatch");

        let mut cur = MATMUL_HEADER_LEN;
        let x_bytes = &bytes[cur..cur + x_len * 4];
        cur += x_len * 4;
        let packed_bytes = &bytes[cur..cur + packed_len * 4];
        cur += packed_len * 4;
        let scales_bytes = &bytes[cur..cur + scales_len];
        cur += scales_len;
        let y_bytes = &bytes[cur..cur + y_len * 4];

        let x: Vec<f32> = x_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let packed: Vec<u32> = packed_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let scales = scales_bytes.to_vec();
        let expected_y: Vec<f32> = y_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        Some(MatmulFixture {
            out_features,
            in_features,
            group_size,
            bits,
            x,
            packed,
            scales,
            expected_y,
        })
    }

    fn run_matmul_fixture(name: &str) {
        let Some(fx) = load_matmul_fixture(name) else {
            eprintln!(
                "SKIP matmul/{name}: fixture missing under {}. Run `python scripts/generate_mxfp4_matmul_fixture.py` first.",
                fixture_dir().display()
            );
            return;
        };

        assert_eq!(
            fx.group_size, GROUP_SIZE,
            "{name}: fixture group_size unexpected"
        );
        assert_eq!(fx.bits, BITS, "{name}: fixture bits unexpected");

        let x = Array::from_slice(&fx.x, &[fx.in_features as i32]);
        let w = Array::from_slice(
            &fx.packed,
            &[fx.out_features as i32, (fx.in_features / 8) as i32],
        );
        let s = Array::from_slice(
            &fx.scales,
            &[fx.out_features as i32, (fx.in_features / 32) as i32],
        );

        let y_raw = quantized_matmul_with_mode(
            &x, &w, &s, None, true, /* transpose: y = x @ deq(w).T */
            GROUP_SIZE, BITS, MODE_MXFP4,
        )
        .expect("mlx_quantized_matmul FFI call must succeed");

        assert_eq!(
            y_raw.shape(),
            &[fx.out_features as i32],
            "{name}: matmul output shape mismatch"
        );

        // mxfp4 quantized_matmul follows the dtype of the activation `x`. We
        // pass `x` as Float32, so the result should already be Float32 (no
        // bf16 cast like the dequant path). Cast unconditionally to keep the
        // test resilient if MLX changes the default in a future release.
        let y = y_raw
            .as_dtype(mlx_rs::Dtype::Float32)
            .expect("matmul output → f32 cast must succeed");
        y.eval().expect("mlx eval must succeed");

        let observed: &[f32] = y.as_slice();
        assert_eq!(
            observed.len(),
            fx.expected_y.len(),
            "{name}: matmul size mismatch"
        );

        let mut mismatches = 0usize;
        for (i, (&got, &expected)) in observed.iter().zip(fx.expected_y.iter()).enumerate() {
            if got.to_bits() != expected.to_bits() {
                if mismatches < 5 {
                    eprintln!(
                        "matmul/{name}[{i}]: mlx-rs FFI 0x{:08x} ({got}) vs MLX ref 0x{:08x} ({expected})",
                        got.to_bits(),
                        expected.to_bits()
                    );
                }
                mismatches += 1;
            }
        }
        assert_eq!(
            mismatches,
            0,
            "matmul/{name}: {mismatches}/{} elements diverged from MLX reference",
            observed.len()
        );
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device; run with `cargo test -p lumen-mlx --features mlx-native -- --ignored`"]
    fn tiny_matmul_matches_mlx() {
        run_matmul_fixture("tiny");
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn small_matmul_matches_mlx() {
        run_matmul_fixture("small");
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn wide_matmul_matches_mlx() {
        run_matmul_fixture("wide");
    }
}
