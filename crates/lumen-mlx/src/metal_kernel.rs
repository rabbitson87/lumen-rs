//! Rust FFI wrapper for `mx.fast.metal_kernel`.
//!
//! pmetal-mlx-sys 0.2.4 exposes the full C ABI for runtime-compiled Metal
//! kernels (`mlx_fast_metal_kernel_*`), but pmetal-mlx-rs 0.25.8 does not
//! wrap it. This module provides a focused Rust wrapper modelled on the C
//! example at `pmetal-mlx-sys/src/mlx-c/examples/example-metal-kernel.c`.
//!
//! Used by Phase 3b.5.d to invoke the SSM step kernel from
//! `mlx_lm/models/gated_delta.py`. Generic enough for any
//! `mx.fast.metal_kernel`-style call, so future kernel ports can reuse it.
//!
//! All RAII-guarded; manual `apply` returns owned `Vec<Array>`.

#[cfg(feature = "mlx-native")]
mod imp {
    use anyhow::{Result, anyhow};
    use mlx_rs::{Array, Dtype};
    use std::ffi::CString;

    /// RAII-guarded `mlx_fast_metal_kernel_config`. Drop frees via
    /// `mlx_fast_metal_kernel_config_free`.
    pub struct MetalKernelConfig {
        inner: mlx_sys::mlx_fast_metal_kernel_config,
    }

    impl MetalKernelConfig {
        pub fn new() -> Self {
            Self {
                inner: unsafe { mlx_sys::mlx_fast_metal_kernel_config_new() },
            }
        }

        pub fn raw(&self) -> mlx_sys::mlx_fast_metal_kernel_config {
            self.inner
        }

        pub fn add_output_arg(&self, shape: &[i32], dtype: Dtype) -> Result<()> {
            let status = unsafe {
                mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
                    self.inner,
                    shape.as_ptr(),
                    shape.len(),
                    dtype.into(),
                )
            };
            if status != 0 {
                return Err(anyhow!(
                    "mlx_fast_metal_kernel_config_add_output_arg returned {status}"
                ));
            }
            Ok(())
        }

        pub fn set_grid(&self, g1: i32, g2: i32, g3: i32) -> Result<()> {
            let status =
                unsafe { mlx_sys::mlx_fast_metal_kernel_config_set_grid(self.inner, g1, g2, g3) };
            if status != 0 {
                return Err(anyhow!(
                    "mlx_fast_metal_kernel_config_set_grid returned {status}"
                ));
            }
            Ok(())
        }

        pub fn set_thread_group(&self, t1: i32, t2: i32, t3: i32) -> Result<()> {
            let status = unsafe {
                mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(self.inner, t1, t2, t3)
            };
            if status != 0 {
                return Err(anyhow!(
                    "mlx_fast_metal_kernel_config_set_thread_group returned {status}"
                ));
            }
            Ok(())
        }

        pub fn add_template_arg_dtype(&self, name: &str, dtype: Dtype) -> Result<()> {
            let cname =
                CString::new(name).map_err(|_| anyhow!("template arg name contains nul byte"))?;
            let status = unsafe {
                mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
                    self.inner,
                    cname.as_ptr(),
                    dtype.into(),
                )
            };
            if status != 0 {
                return Err(anyhow!(
                    "mlx_fast_metal_kernel_config_add_template_arg_dtype({name}) returned {status}"
                ));
            }
            Ok(())
        }

        pub fn add_template_arg_int(&self, name: &str, value: i32) -> Result<()> {
            let cname =
                CString::new(name).map_err(|_| anyhow!("template arg name contains nul byte"))?;
            let status = unsafe {
                mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                    self.inner,
                    cname.as_ptr(),
                    value,
                )
            };
            if status != 0 {
                return Err(anyhow!(
                    "mlx_fast_metal_kernel_config_add_template_arg_int({name}) returned {status}"
                ));
            }
            Ok(())
        }

        #[allow(dead_code)]
        pub fn add_template_arg_bool(&self, name: &str, value: bool) -> Result<()> {
            let cname =
                CString::new(name).map_err(|_| anyhow!("template arg name contains nul byte"))?;
            let status = unsafe {
                mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_bool(
                    self.inner,
                    cname.as_ptr(),
                    value,
                )
            };
            if status != 0 {
                return Err(anyhow!(
                    "mlx_fast_metal_kernel_config_add_template_arg_bool({name}) returned {status}"
                ));
            }
            Ok(())
        }
    }

    impl Drop for MetalKernelConfig {
        fn drop(&mut self) {
            unsafe {
                mlx_sys::mlx_fast_metal_kernel_config_free(self.inner);
            }
        }
    }

    /// RAII-guarded `mlx_fast_metal_kernel`. Drop frees via
    /// `mlx_fast_metal_kernel_free`.
    pub struct MetalKernel {
        inner: mlx_sys::mlx_fast_metal_kernel,
    }

    impl MetalKernel {
        /// Construct a kernel with the given source. `input_names` /
        /// `output_names` must match the variables referenced by the source
        /// (e.g. `["q", "k", "v", ...]`).
        pub fn new(
            name: &str,
            input_names: &[&str],
            output_names: &[&str],
            source: &str,
            ensure_row_contiguous: bool,
            atomic_outputs: bool,
        ) -> Result<Self> {
            let cname = CString::new(name).map_err(|_| anyhow!("kernel name contains nul byte"))?;
            let csource =
                CString::new(source).map_err(|_| anyhow!("kernel source contains nul byte"))?;
            // Empty header — we don't use additional code injection.
            let cheader = CString::new("").unwrap();

            // Build the input/output name vectors. RAII guard so they're freed
            // after construction.
            struct VecStringGuard(mlx_sys::mlx_vector_string);
            impl Drop for VecStringGuard {
                fn drop(&mut self) {
                    unsafe {
                        let _ = mlx_sys::mlx_vector_string_free(self.0);
                    }
                }
            }

            let inputs_vec = unsafe { mlx_sys::mlx_vector_string_new() };
            let inputs_guard = VecStringGuard(inputs_vec);
            let mut name_storage: Vec<CString> = Vec::with_capacity(input_names.len());
            for name in input_names {
                let cs = CString::new(*name)
                    .map_err(|_| anyhow!("input name contains nul byte: {name}"))?;
                let status =
                    unsafe { mlx_sys::mlx_vector_string_append_value(inputs_guard.0, cs.as_ptr()) };
                name_storage.push(cs);
                if status != 0 {
                    return Err(anyhow!(
                        "mlx_vector_string_append_value (input) returned {status}"
                    ));
                }
            }

            let outputs_vec = unsafe { mlx_sys::mlx_vector_string_new() };
            let outputs_guard = VecStringGuard(outputs_vec);
            let mut output_name_storage: Vec<CString> = Vec::with_capacity(output_names.len());
            for name in output_names {
                let cs = CString::new(*name)
                    .map_err(|_| anyhow!("output name contains nul byte: {name}"))?;
                let status = unsafe {
                    mlx_sys::mlx_vector_string_append_value(outputs_guard.0, cs.as_ptr())
                };
                output_name_storage.push(cs);
                if status != 0 {
                    return Err(anyhow!(
                        "mlx_vector_string_append_value (output) returned {status}"
                    ));
                }
            }

            let inner = unsafe {
                mlx_sys::mlx_fast_metal_kernel_new(
                    cname.as_ptr(),
                    inputs_guard.0,
                    outputs_guard.0,
                    csource.as_ptr(),
                    cheader.as_ptr(),
                    ensure_row_contiguous,
                    atomic_outputs,
                )
            };
            if inner.ctx.is_null() {
                return Err(anyhow!("mlx_fast_metal_kernel_new returned null handle"));
            }

            Ok(Self { inner })
        }

        /// Apply the kernel with the given inputs and config. Returns the
        /// outputs in the order specified by `output_names` at construction.
        ///
        /// The config must have had `add_output_arg` called once per output.
        pub fn apply(
            &self,
            inputs: &[&Array],
            config: &MetalKernelConfig,
            num_outputs: usize,
        ) -> Result<Vec<Array>> {
            // Build mlx_vector_array of inputs, free on drop.
            struct VecArrayGuard(mlx_sys::mlx_vector_array);
            impl Drop for VecArrayGuard {
                fn drop(&mut self) {
                    unsafe {
                        let _ = mlx_sys::mlx_vector_array_free(self.0);
                    }
                }
            }

            let input_vec = unsafe { mlx_sys::mlx_vector_array_new() };
            let input_guard = VecArrayGuard(input_vec);
            for arr in inputs {
                let status =
                    unsafe { mlx_sys::mlx_vector_array_append_value(input_guard.0, arr.as_ptr()) };
                if status != 0 {
                    return Err(anyhow!(
                        "mlx_vector_array_append_value (input) returned {status}"
                    ));
                }
            }

            let stream = unsafe { mlx_sys::mlx_default_gpu_stream_new() };
            // Stream guard — freed on drop.
            struct StreamGuard(mlx_sys::mlx_stream);
            impl Drop for StreamGuard {
                fn drop(&mut self) {
                    unsafe {
                        let _ = mlx_sys::mlx_stream_free(self.0);
                    }
                }
            }
            let stream_guard = StreamGuard(stream);

            let mut outputs_raw = unsafe { mlx_sys::mlx_vector_array_new() };
            let status = unsafe {
                mlx_sys::mlx_fast_metal_kernel_apply(
                    &mut outputs_raw as *mut mlx_sys::mlx_vector_array,
                    self.inner,
                    input_guard.0,
                    config.raw(),
                    stream_guard.0,
                )
            };
            if status != 0 {
                unsafe {
                    let _ = mlx_sys::mlx_vector_array_free(outputs_raw);
                }
                return Err(anyhow!("mlx_fast_metal_kernel_apply returned {status}"));
            }

            // Pull each output out of the vector_array. Take ownership via
            // `Array::from_ptr`; vector_array is freed afterwards.
            let mut outputs = Vec::with_capacity(num_outputs);
            for i in 0..num_outputs {
                let mut raw: mlx_sys::mlx_array = unsafe { mlx_sys::mlx_array_new() };
                let s = unsafe { mlx_sys::mlx_vector_array_get(&mut raw, outputs_raw, i) };
                if s != 0 {
                    unsafe {
                        let _ = mlx_sys::mlx_array_free(raw);
                        let _ = mlx_sys::mlx_vector_array_free(outputs_raw);
                    }
                    return Err(anyhow!("mlx_vector_array_get(output {i}) returned {s}"));
                }
                outputs.push(unsafe { Array::from_ptr(raw) });
            }

            unsafe {
                let _ = mlx_sys::mlx_vector_array_free(outputs_raw);
            }

            Ok(outputs)
        }
    }

    impl Drop for MetalKernel {
        fn drop(&mut self) {
            unsafe {
                mlx_sys::mlx_fast_metal_kernel_free(self.inner);
            }
        }
    }
}

#[cfg(feature = "mlx-native")]
#[allow(unused_imports)] // Consumed by Phase 3b.5.d SSM kernel + future kernels.
pub(crate) use imp::{MetalKernel, MetalKernelConfig};

// Smoke test — verify the FFI wrapper works with a minimal kernel before
// we use it for the more complex SSM kernel. Mirrors the C example
// `myexp` (`out[elem] = metal::exp(inp[elem])`).
#[cfg(all(test, feature = "mlx-native"))]
mod smoke_tests {
    use super::imp::{MetalKernel, MetalKernelConfig};
    use mlx_rs::{Array, Dtype};

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn myexp_smoke_kernel_works() {
        let source = "uint elem = thread_position_in_grid.x;\n\
                      T tmp = inp[elem];\n\
                      out[elem] = metal::exp(tmp);";
        let kernel = MetalKernel::new(
            "myexp",
            &["inp"],
            &["out"],
            source,
            /* ensure_row_contiguous */ true,
            /* atomic_outputs */ false,
        )
        .expect("MetalKernel::new must succeed");

        let input_data: [f32; 4] = [0.0, 1.0, 2.0, 3.0];
        let input = Array::from_slice(&input_data, &[4]);

        let config = MetalKernelConfig::new();
        config
            .add_template_arg_dtype("T", Dtype::Float32)
            .expect("add_template_arg_dtype must succeed");
        config
            .set_grid(input_data.len() as i32, 1, 1)
            .expect("set_grid must succeed");
        config
            .set_thread_group(input_data.len() as i32, 1, 1)
            .expect("set_thread_group must succeed");
        config
            .add_output_arg(&[input_data.len() as i32], Dtype::Float32)
            .expect("add_output_arg must succeed");

        let outputs = kernel
            .apply(&[&input], &config, 1)
            .expect("metal kernel apply must succeed");
        assert_eq!(outputs.len(), 1);
        let out = &outputs[0];
        assert_eq!(out.shape(), &[4]);
        out.eval().expect("eval must succeed");

        let observed: &[f32] = out.as_slice();
        let expected: Vec<f32> = input_data.iter().map(|x| x.exp()).collect();

        for (i, (&got, &exp)) in observed.iter().zip(expected.iter()).enumerate() {
            // metal::exp is f32-precision; compare close not bit-identical
            // because the algorithm-level precision may differ from f32::exp.
            let abs_diff = (got - exp).abs();
            let rel = abs_diff / exp.abs().max(1e-6);
            assert!(
                rel < 1e-5,
                "myexp[{i}]: got={got} expected={exp} rel_err={rel}"
            );
        }
    }
}
