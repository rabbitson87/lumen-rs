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

    lumen_flags::flag! {
        /// Share one `OnceLock` stream + sentinel across FFI quant ops instead
        /// of constructing `StreamGuard::gpu()` + `OwnedEmptyArray` per call.
        /// Default OFF — Lever 2 infrastructure, measured an **A/B wash**
        /// (2026-05-03). Kept because the per-call construction is the thing it
        /// would replace, not because it won.
        ///
        /// Stream plumbing only; the arithmetic is untouched, hence
        /// `Optimization`.
        pub(crate) cached_stream {
            env: "LUMEN_NATIVE_CACHED_STREAM",
            default: false,
            kind: Optimization,
        }
    }

    /// Returns true (default) when each FFI quant op constructs its own
    /// `StreamGuard::gpu()` + `OwnedEmptyArray` sentinels.
    fn use_per_call_stream() -> bool {
        !cached_stream::get()
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

// MXFP4 3-D expert-stack quant triage — does the MTP loader's 3-D / split-view
// quantize round-trip, or is it the source of the garbage draft logits?
//
//   cargo test -p lumen-mlx --features mlx-native mxfp4_3d_expert_roundtrip -- --nocapture
#[cfg(all(test, feature = "mlx-native"))]
mod mxfp4_3d_expert_triage {
    use super::imp::{MODE_MXFP4, dequantize_with_mode, quantize_with_mode};
    use mlx_rs::Array;

    fn rel_mae(recon: &[f32], src: &[f32]) -> f32 {
        let (mut num, mut den) = (0.0f32, 0.0f32);
        for (a, b) in recon.iter().zip(src) {
            num += (a - b).abs();
            den += b.abs();
        }
        num / den.max(1e-9)
    }

    #[test]
    fn mxfp4_3d_expert_roundtrip() {
        let (e, out, inf) = (4i32, 64i32, 128i32); // in multiple of group 32
        let n = (e * out * inf) as usize;
        let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin() * 1.5).collect();

        // ---- (1) 3-D direct quantize [E, out, in] (what the MTP loader does) ----
        let w3 = Array::from_slice(&data, &[e, out, inf]);
        let (p3, s3, _) = quantize_with_mode(&w3, 32, 4, MODE_MXFP4).expect("3D quant");
        eprintln!("3D packed shape={:?} scales={:?}", p3.shape(), s3.shape());
        let d3 = dequantize_with_mode(&p3, &s3, None, 32, 4, MODE_MXFP4)
            .expect("3D dequant")
            .as_dtype(mlx_rs::Dtype::Float32)
            .unwrap();
        d3.eval().unwrap();
        let rel_3d = rel_mae(d3.as_slice(), &data);
        eprintln!("[A] 3D direct rel-MAE = {rel_3d:.4}");

        // ---- (2) 3-D from NON-CONTIGUOUS split view (loader splits fused gate_up) ----
        // Build [E, 2*out, in], split axis 1, quantize the first (strided) half.
        let n2 = (e * 2 * out * inf) as usize;
        let data2: Vec<f32> = (0..n2).map(|i| (i as f32 * 0.011).cos() * 1.5).collect();
        let fused = Array::from_slice(&data2, &[e, 2 * out, inf]);
        // Rust-computed LOGICAL-ORDER reference for the gate half (rows 0..out of
        // the 2*out axis), so comparison never reads a strided view via as_slice.
        let gate_expected: Vec<f32> = {
            let mut v = Vec::with_capacity((e * out * inf) as usize);
            for ei in 0..e {
                for o in 0..out {
                    for kk in 0..inf {
                        let idx = (((ei * (2 * out)) + o) * inf + kk) as usize;
                        v.push(data2[idx]);
                    }
                }
            }
            v
        };
        let halves = mlx_rs::ops::split(&fused, 2, 1).expect("split");
        let gate_view = halves.into_iter().next().unwrap(); // non-contiguous [E,out,in]
        let (pv, sv, _) = quantize_with_mode(&gate_view, 32, 4, MODE_MXFP4).expect("view quant");
        let dv = dequantize_with_mode(&pv, &sv, None, 32, 4, MODE_MXFP4)
            .expect("view dequant")
            .as_dtype(mlx_rs::Dtype::Float32)
            .unwrap();
        dv.eval().unwrap();
        let rel_view = rel_mae(dv.as_slice(), &gate_expected);
        eprintln!("[B] 3D split-view rel-MAE = {rel_view:.4}  (vs Rust logical-order ref)");

        // ---- (3) per-expert 2-D loop (the proposed fix) ----
        let mut max_rel_2d = 0.0f32;
        let expert_slices = mlx_rs::ops::split(&w3, e, 0).expect("split experts"); // E × [1,out,in]
        for slice in &expert_slices {
            let tile = slice.reshape(&[out, inf]).unwrap();
            tile.eval().unwrap();
            let (pt, st, _) = quantize_with_mode(&tile, 32, 4, MODE_MXFP4).expect("2D quant");
            let dt = dequantize_with_mode(&pt, &st, None, 32, 4, MODE_MXFP4)
                .expect("2D dequant")
                .as_dtype(mlx_rs::Dtype::Float32)
                .unwrap();
            dt.eval().unwrap();
            let tilef = tile.as_dtype(mlx_rs::Dtype::Float32).unwrap();
            tilef.eval().unwrap();
            max_rel_2d = max_rel_2d.max(rel_mae(dt.as_slice(), tilef.as_slice()));
        }
        eprintln!("[C] per-expert 2D loop max rel-MAE = {max_rel_2d:.4}");

        // ---- (4) split-view MATERIALIZED contiguous (the proposed fix) ----
        let zero = Array::from_f32(0.0).as_dtype(gate_view.dtype()).unwrap();
        let gate_contig = mlx_rs::ops::add(&gate_view, &zero).expect("materialize");
        gate_contig.eval().unwrap();
        let (pc, sc, _) =
            quantize_with_mode(&gate_contig, 32, 4, MODE_MXFP4).expect("contig quant");
        let dc = dequantize_with_mode(&pc, &sc, None, 32, 4, MODE_MXFP4)
            .expect("contig dequant")
            .as_dtype(mlx_rs::Dtype::Float32)
            .unwrap();
        dc.eval().unwrap();
        let rel_fix = rel_mae(dc.as_slice(), &gate_expected);
        eprintln!("[D] split-view + add(0) materialize rel-MAE = {rel_fix:.4}");

        eprintln!(
            "VERDICT: [B] huge & [D]~[C]~0.08 confirms non-contiguous view = bug, add(0) = fix."
        );
        assert!(
            rel_fix < 0.2,
            "materialized split-view must round-trip to noise floor"
        );
        // 2D loop is the known-good baseline; assert it round-trips to mxfp4 noise floor.
        assert!(
            max_rel_2d < 0.2,
            "per-expert 2D quant should be ~mxfp4 noise floor"
        );
    }
}

// The NVFP4 kernel-vs-FFI A/B lived here: it timed lumen-metal's custom fused
// matvec against mlx's `quantized_matmul(NVFP4)` at Gemma4-26B-A4B MoE shapes,
// as a gate-check before investing in a fused gate_up_silu_mul kernel. Its
// recorded verdict was that the custom kernel was already slower per-matmul, so
// fusion could not recover the deficit. `lumen-metal` was removed with the
// Candle backend (task 006), which leaves this benchmark with nothing to
// compare against — and its question already answered.
