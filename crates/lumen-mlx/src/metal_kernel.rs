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
        /// `name=dtype` per dtype template argument, as test builds report a
        /// kernel's instantiation (see [`note_kernel_ran`]).
        #[cfg(test)]
        dtype_args: std::cell::RefCell<Vec<String>>,
    }

    impl Default for MetalKernelConfig {
        fn default() -> Self {
            Self::new()
        }
    }

    impl MetalKernelConfig {
        pub fn new() -> Self {
            Self {
                inner: unsafe { mlx_sys::mlx_fast_metal_kernel_config_new() },
                #[cfg(test)]
                dtype_args: Default::default(),
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
            #[cfg(test)]
            self.dtype_args
                .borrow_mut()
                .push(format!("{name}={dtype:?}"));
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
        /// The kernel's name, as test builds report it (see [`note_kernel_ran`]).
        #[cfg(test)]
        name: String,
    }

    /// Test builds only: report each instantiation of a run-time kernel the
    /// first time it is applied — its name, then its dtype template arguments
    /// (`gated_delta_step InT=Bfloat16 StT=Float32`), since MLX compiles each
    /// instantiation separately.
    ///
    /// `cargo xtask test`'s macOS 14 pass reads these lines (it runs with
    /// `--nocapture`) and fails if a kernel declared in source never ran, or
    /// one generic over its float type never ran as bfloat16: a test that
    /// returns early — a missing fixture, say — would otherwise count as a
    /// kernel checked at Metal 3.1 when it was never compiled. MLX compiles at
    /// evaluation, so a kernel test has to evaluate what the kernel returns.
    #[cfg(test)]
    fn note_kernel_ran(name: &str, dtype_args: &[String]) {
        use std::collections::BTreeSet;
        use std::sync::Mutex;
        static SEEN: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());
        let line = std::iter::once(name)
            .chain(dtype_args.iter().map(String::as_str))
            .collect::<Vec<_>>()
            .join(" ");
        if let Ok(mut seen) = SEEN.lock()
            && seen.insert(line.clone())
        {
            eprintln!("[metal-kernel-ran] {line}");
        }
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

            Ok(Self {
                inner,
                #[cfg(test)]
                name: name.to_string(),
            })
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
            #[cfg(test)]
            note_kernel_ran(&self.name, &config.dtype_args.borrow());
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
pub use imp::{MetalKernel, MetalKernelConfig};

// Smoke test — verify the FFI wrapper works with a minimal kernel before
// we use it for the more complex SSM kernel. Mirrors the C example
// `myexp` (`out[elem] = metal::exp(inp[elem])`).
#[cfg(all(test, feature = "mlx-native"))]
mod smoke_tests {
    use super::imp::{MetalKernel, MetalKernelConfig};
    use mlx_rs::{Array, Dtype};

    /// The Metal language version MLX compiles run-time kernels with — the
    /// one the user's macOS supports, not the build machine's.
    ///
    /// `cargo xtask test` runs every kernel's tests a second time with
    /// `MLX_METAL_LANGUAGE_VERSION=3.1`, what macOS 14 compiles with, so a
    /// kernel that needs a newer Metal fails here instead of on a user's Mac.
    /// That pass is only as good as the override, and an MLX build from before
    /// it (rabbitson87/mlx `lumen-rs-patches-v0.32`) silently ignores the
    /// variable — so this kernel reports the version it was compiled with, and
    /// the test holds it to the one requested.
    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn kernels_compile_with_the_requested_metal_version() {
        let kernel = MetalKernel::new(
            "lumen_metal_version_probe",
            &["inp"],
            &["out"],
            "out[0] = __METAL_VERSION__ + int(inp[0]);",
            /* ensure_row_contiguous */ true,
            /* atomic_outputs */ false,
        )
        .expect("probe kernel");
        let input = Array::from_slice(&[0i32], &[1]);
        let config = MetalKernelConfig::new();
        config.set_grid(1, 1, 1).expect("grid");
        config.set_thread_group(1, 1, 1).expect("thread group");
        config.add_output_arg(&[1], Dtype::Int32).expect("output");
        let out = kernel.apply(&[&input], &config, 1).expect("probe apply");
        out[0].eval().expect("probe eval");
        let compiled_with = out[0].as_slice::<i32>()[0];
        eprintln!("run-time kernels compile as __METAL_VERSION__ {compiled_with}");

        match std::env::var("MLX_METAL_LANGUAGE_VERSION").ok().as_deref() {
            Some(requested) => {
                let expected = match requested {
                    "3.1" => 310,
                    "3.2" => 320,
                    "4.0" => 400,
                    other => {
                        panic!("MLX_METAL_LANGUAGE_VERSION={other}: not a version MLX accepts")
                    }
                };
                assert_eq!(
                    compiled_with, expected,
                    "MLX_METAL_LANGUAGE_VERSION={requested} was ignored — this MLX build predates \
                     the override; rebuild it (`cargo clean -p mlx-sys --release`)"
                );
            }
            None => assert!(
                compiled_with >= 310,
                "macOS 14 is the floor: {compiled_with}"
            ),
        }
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn myexp_smoke_kernel_works() {
        myexp_smoke_kernel_works_as(Dtype::Float32, 1e-5);
    }

    /// The bf16 instantiation is a separate compile (see `note_kernel_ran`).
    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn myexp_smoke_kernel_works_in_bf16() {
        myexp_smoke_kernel_works_as(Dtype::Bfloat16, 1e-2);
    }

    fn myexp_smoke_kernel_works_as(dt: Dtype, max_rel: f32) {
        let source = "uint elem = thread_position_in_grid.x;\n\
                      float tmp = static_cast<float>(inp[elem]);\n\
                      out[elem] = static_cast<T>(metal::exp(tmp));";
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
        let input = Array::from_slice(&input_data, &[4]).as_dtype(dt).unwrap();

        let config = MetalKernelConfig::new();
        config
            .add_template_arg_dtype("T", dt)
            .expect("add_template_arg_dtype must succeed");
        config
            .set_grid(input_data.len() as i32, 1, 1)
            .expect("set_grid must succeed");
        config
            .set_thread_group(input_data.len() as i32, 1, 1)
            .expect("set_thread_group must succeed");
        config
            .add_output_arg(&[input_data.len() as i32], dt)
            .expect("add_output_arg must succeed");

        let outputs = kernel
            .apply(&[&input], &config, 1)
            .expect("metal kernel apply must succeed");
        assert_eq!(outputs.len(), 1);
        let out = &outputs[0];
        assert_eq!(out.shape(), &[4]);
        let out = out.as_dtype(Dtype::Float32).unwrap();
        out.eval().expect("eval must succeed");

        let observed: &[f32] = out.as_slice();
        let expected: Vec<f32> = input_data.iter().map(|x| x.exp()).collect();

        for (i, (&got, &exp)) in observed.iter().zip(expected.iter()).enumerate() {
            // metal::exp is f32-precision; compare close not bit-identical
            // because the algorithm-level precision may differ from f32::exp.
            let abs_diff = (got - exp).abs();
            let rel = abs_diff / exp.abs().max(1e-6);
            assert!(
                rel < max_rel,
                "{dt:?} myexp[{i}]: got={got} expected={exp} rel_err={rel}"
            );
        }
    }
}
