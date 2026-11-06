//! Native MLX state-space model (SSM) primitives for the
//! Qwen3.5-MoE delta-net linear-attention block.
//!
//! Sub-pieces validated independently before the full `gated_delta_update`
//! kernel composition lands:
//!
//! * **`compute_g`** (this module, 3b.5.b) — gating activation
//!   `exp(-exp(A_log.astype(f32)) * softplus(a + dt_bias))`. Pure
//!   composition of stock mlx-rs ops; bit-identical against the mlx-lm
//!   reference (`mlx_lm/models/gated_delta.py`).
//! * **`rms_norm_gated`** (3b.5.c, see below) — `RMSNormGated.forward`
//!   from mlx-lm: `weight * (x * sigmoid(z) / rms(x * sigmoid(z)))` with
//!   per-feature weight scaling.
//! * **SSM kernel** (3b.5.d, deferred to a future session) — runtime-
//!   compiled Metal kernel via `mlx_fast_metal_kernel_*` C ABI. The
//!   surrounding pre/post processing (Conv1d, SiLU, RMSNorm on q/k,
//!   `compute_g`, `rms_norm_gated`, out-proj) is all ready by the time
//!   the kernel itself ships.

#[cfg(feature = "mlx-native")]
mod imp {
    use std::sync::{Mutex, OnceLock};

    use anyhow::{Context, Result};
    use mlx_rs::Array;
    use mlx_rs::error::Exception;

    use crate::native_compile_cache::{CompiledMultiRefs, invoke_compiled_multi_refs};

    // Module-level fn pointer for `compute_g` body. Stays as a true `fn`
    // (not a closure) so its `TypeId` is stable across calls — required for
    // `mlx_rs::transforms::compile::compile`'s internal cache to hit instead
    // of re-compiling on every layer.
    //
    // Mirrors `mlx_lm/models/gated_delta.py::compute_g` op-for-op; lives in
    // a separate fn so it can flow into `compile_boxed_slice` via the
    // `&[Array] -> Result<Vec<Array>, Exception>` form. Keep this body
    // equivalent to `compute_g_legacy`, which `LUMEN_NATIVE_COMPILE=0` serves
    // instead.
    fn compute_g_compiled_inner(args: &[Array]) -> std::result::Result<Vec<Array>, Exception> {
        let a_log = &args[0];
        let a = &args[1];
        let dt_bias = &args[2];
        let a_log_f32 = a_log.as_dtype(mlx_rs::Dtype::Float32)?;
        let exp_a_log = a_log_f32.exp()?;
        let a_plus_bias = a.add(dt_bias)?;
        let softplus = mlx_rs::nn::softplus(&a_plus_bias)?;
        let prod = exp_a_log.multiply(&softplus)?;
        let neg = prod.negative()?;
        Ok(vec![neg.exp()?])
    }

    static COMPUTE_G_SLOT: OnceLock<Mutex<CompiledMultiRefs>> = OnceLock::new();

    lumen_flags::flag! {
        /// compute_g persistent compile cache. Default ON (LANDED Phase 3):
        /// wins σ-significantly across a 12-run 4-condition matrix
        /// (Δp50=−0.33 ms, Δtps=+1.96 vs legacy direct dispatch). `=0` falls
        /// back to the legacy direct-ops path.
        ///
        /// **`Behavior`, not `Optimization`:** the legacy ops round some gates
        /// differently from the compiled function, by one unit in the last place
        /// (three of 256 in `linattn_prework_gates_are_the_composed_gates_to_the_bit`
        /// at `=0`), so `=0` can change a near-tie. Gates the fused prework kernel
        /// computes (decode) are the compiled function's either way.
        pub(crate) ssm_compile {
            env: "LUMEN_NATIVE_COMPILE",
            default: true,
            kind: Behavior,
        }
    }

    fn compile_enabled() -> bool {
        ssm_compile::get()
    }

    /// Qwen3.5-MoE delta-net SSM gating activation:
    ///
    /// ```text
    /// g = exp(-exp(A_log.astype(f32)) * softplus(a + dt_bias))
    /// ```
    ///
    /// Shapes: `A_log: [Hv]`, `a: [B, S, Hv]`, `dt_bias: [Hv]`,
    /// returns `[B, S, Hv]` f32 (broadcasting over the head axis).
    ///
    /// When `LUMEN_NATIVE_COMPILE=1` is set the body is dispatched through
    /// the persistent compile cache (`native_compile_cache::invoke_compiled_multi`)
    /// — matches Python's `@partial(mx.compile, shapeless=True)` on the same
    /// function in `mlx_lm/models/gated_delta.py`. The wrapper is built once
    /// per process via `compile_boxed_slice` (mlx-rs fork) and reused for the
    /// process lifetime, avoiding the per-call `Compiled<F, G>` alloc/Drop
    /// measured in Phase 1 microbench (~0.027ms/call × 30 layers = +0.81ms).
    pub fn compute_g(a_log: &Array, a: &Array, dt_bias: &Array) -> Result<Array> {
        if compile_enabled() {
            let args = [a_log, a, dt_bias];
            let mut out = invoke_compiled_multi_refs(
                &COMPUTE_G_SLOT,
                compute_g_compiled_inner,
                /* shapeless */ true,
                &args,
            )
            .context("compute_g (compiled cache): mlx compile dispatch failed")?;
            Ok(out
                .pop()
                .context("compute_g (compiled cache): empty output vec")?)
        } else {
            compute_g_legacy(a_log, a, dt_bias)
        }
    }

    fn compute_g_legacy(a_log: &Array, a: &Array, dt_bias: &Array) -> Result<Array> {
        let a_log_f32 = a_log
            .as_dtype(mlx_rs::Dtype::Float32)
            .context("compute_g: A_log → f32 cast failed")?;
        let exp_a_log = a_log_f32.exp().context("compute_g: exp(A_log) failed")?;
        let a_plus_bias = a
            .add(dt_bias)
            .context("compute_g: a + dt_bias failed (check broadcast)")?;
        let softplus = mlx_rs::nn::softplus(&a_plus_bias)
            .context("compute_g: softplus(a + dt_bias) failed")?;
        let prod = exp_a_log
            .multiply(&softplus)
            .context("compute_g: exp(A_log) * softplus failed (check broadcast)")?;
        let neg = prod.negative().context("compute_g: negate failed")?;
        neg.exp().context("compute_g: outer exp failed")
    }

    // ─── Phase 3b.5.d: gated_delta_step kernel ───────────────────────────
    //
    // Source ported verbatim from `_make_gated_delta_kernel(has_mask=False,
    // vectorized=False)` in `mlx_lm/models/gated_delta.py`. The Python f-string
    // substitutions (`mask_source="true"`, scalar `g` indexing) are baked in.
    //
    // The kernel implements one full SSM unrolled-time recurrence on the GPU:
    //
    //     for t in 0..T:
    //         state *= g                  // decay
    //         kv_mem = state · k          // simdgroup reduction over Dk tile
    //         delta  = (v - kv_mem) * beta
    //         state += k ⊗ delta          // outer-product update
    //         y      = state · q          // simdgroup reduction
    //
    // Inputs (in order, matching Python's `input_names`):
    //     q[B, T, Hk, Dk], k[B, T, Hk, Dk], v[B, T, Hv, Dv],
    //     g[B, T, Hv], beta[B, T, Hv], state_in[B, Hv, Dv, Dk],
    //     T  (0-d int scalar; MLX dereferences scalar arrays at call sites)
    // Outputs:
    //     y[B, T, Hv, Dv], state_out[B, Hv, Dv, Dk]
    // Templates:
    //     InT (input dtype), StT (state dtype),
    //     Dk, Dv, Hk, Hv (compile-time dimensions)
    // Grid: (32, Dv, B * Hv); Threadgroup: (32, 4, 1).
    pub const GATED_DELTA_KERNEL_SOURCE: &str = r#"
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        // g: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          if (true) {
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * g_[hv_idx];
              kv_mem += state[i] * k_[s_idx];
            }
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + k_[s_idx] * delta;
              out += state[i] * q_[s_idx];
            }
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {
              y[dv_idx] = static_cast<InT>(out);
            }
          } else {
            y[dv_idx] = static_cast<InT>(0);
          }
          // Increment data pointers to next time step
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }
    "#;

    thread_local! {
        static GATED_DELTA: std::cell::OnceCell<crate::metal_kernel::MetalKernel> =
            const { std::cell::OnceCell::new() };
        static GATED_DELTA_CAPTURE: std::cell::OnceCell<crate::metal_kernel::MetalKernel> =
            const { std::cell::OnceCell::new() };
    }

    /// One pass of the delta-net SSM recurrence, dispatched as a single
    /// runtime-compiled Metal kernel via `mlx_fast_metal_kernel_*`. Mirrors
    /// `gated_delta_kernel(q, k, v, g, beta, state, mask=None)` from
    /// `mlx_lm/models/gated_delta.py` (no-mask, scalar-gating variant).
    ///
    /// Shapes:
    /// * `q`, `k`: `[B, seq, Hk, Dk]`
    /// * `v`: `[B, seq, Hv, Dv]`
    /// * `g`, `beta`: `[B, seq, Hv]` (scalar gating per head)
    /// * `state_in`: `[B, Hv, Dv, Dk]` (zeros for first call)
    ///
    /// Returns `(y, state_out)` where `y: [B, seq, Hv, Dv]` (dtype = q.dtype)
    /// and `state_out: [B, Hv, Dv, Dk]` (dtype = state_in.dtype).
    pub fn gated_delta_step_kernel(
        q: &Array,
        k: &Array,
        v: &Array,
        g: &Array,
        beta: &Array,
        state_in: &Array,
    ) -> Result<(Array, Array)> {
        let mut outputs = gated_delta_dispatch(q, k, v, g, beta, state_in, false)?.into_iter();
        let y = outputs.next().expect("kernel produced y");
        let state_out = outputs.next().expect("kernel produced state_out");
        Ok((y, state_out))
    }

    /// Shared dispatch of the two gated-delta kernels: one shape check, one
    /// template set, one grid. `capture` picks the capture kernel and its
    /// extra `state_per_pos` output.
    fn gated_delta_dispatch(
        q: &Array,
        k: &Array,
        v: &Array,
        g: &Array,
        beta: &Array,
        state_in: &Array,
        capture: bool,
    ) -> Result<Vec<Array>> {
        use crate::metal_kernel::{MetalKernel, MetalKernelConfig, with_thread_kernel};

        let fn_name = if capture {
            "gated_delta_step_kernel_capture"
        } else {
            "gated_delta_step_kernel"
        };
        // Pull dimensions from input shapes. Indices follow the Python ref.
        let q_shape = q.shape();
        let v_shape = v.shape();
        if q_shape.len() != 4 || v_shape.len() != 4 {
            return Err(anyhow::anyhow!(
                "{fn_name}: q/v must be 4-D, got q={:?} v={:?}",
                q_shape,
                v_shape
            ));
        }
        let b = q_shape[0];
        let seq = q_shape[1];
        let n_k_heads = q_shape[2];
        let head_k_dim = q_shape[3];
        let n_v_heads = v_shape[2];
        let head_v_dim = v_shape[3];

        if head_k_dim % 32 != 0 {
            return Err(anyhow::anyhow!(
                "{fn_name}: head_k_dim must be a multiple of 32 (kernel uses simdgroup reduction over a tile of 32), got {head_k_dim}"
            ));
        }

        let config = MetalKernelConfig::new();
        config.add_template_arg_dtype("InT", q.dtype())?;
        config.add_template_arg_dtype("StT", state_in.dtype())?;
        config.add_template_arg_int("Dk", head_k_dim)?;
        config.add_template_arg_int("Dv", head_v_dim)?;
        config.add_template_arg_int("Hk", n_k_heads)?;
        config.add_template_arg_int("Hv", n_v_heads)?;

        config.set_grid(32, head_v_dim, b * n_v_heads)?;
        // Threadgroup y (Dv tiling) tunes occupancy — x=32 fixed (simd_sum over
        // the Dk tile). y only affects scheduling, not the result. Default 4;
        // `LUMEN_SSM_TG_Y` sweeps {1,2,4,8,16} (must divide head_v_dim).
        let tg_y = std::env::var("LUMEN_SSM_TG_Y")
            .ok()
            .and_then(|s| s.parse::<i32>().ok())
            .filter(|&y| y >= 1 && head_v_dim % y == 0)
            .unwrap_or(4);
        config.set_thread_group(32, tg_y, 1)?;

        config.add_output_arg(&[b, seq, n_v_heads, head_v_dim], q.dtype())?;
        config.add_output_arg(&[b, n_v_heads, head_v_dim, head_k_dim], state_in.dtype())?;
        if capture {
            config.add_output_arg(
                &[b, seq, n_v_heads, head_v_dim, head_k_dim],
                state_in.dtype(),
            )?;
        }

        // T is a 0-d int scalar input. MLX dereferences scalar arrays at
        // call sites, so the kernel sees T as an int value.
        let t_scalar = Array::from_int(seq);
        let inputs = [q, k, v, g, beta, state_in, &t_scalar];
        // Each kernel is constructed with its name as a literal: `cargo xtask
        // test` reads the names off `MetalKernel::new` calls to check that
        // every kernel ran in its Metal 3.1 pass.
        if capture {
            with_thread_kernel(
                &GATED_DELTA_CAPTURE,
                || {
                    MetalKernel::new(
                        "gated_delta_step_capture",
                        &["q", "k", "v", "g", "beta", "state_in", "T"],
                        &["y", "state_out", "state_per_pos"],
                        GATED_DELTA_CAPTURE_KERNEL_SOURCE,
                        /* ensure_row_contiguous */ true,
                        /* atomic_outputs */ false,
                    )
                },
                |kernel| kernel.apply(&inputs, &config, 3),
            )
        } else {
            with_thread_kernel(
                &GATED_DELTA,
                || {
                    MetalKernel::new(
                        "gated_delta_step",
                        &["q", "k", "v", "g", "beta", "state_in", "T"],
                        &["y", "state_out"],
                        GATED_DELTA_KERNEL_SOURCE,
                        /* ensure_row_contiguous */ true,
                        /* atomic_outputs */ false,
                    )
                },
                |kernel| kernel.apply(&inputs, &config, 2),
            )
        }
    }

    /// Capture variant of the gated-delta kernel: identical recurrence, but
    /// ALSO emits the per-position recurrent state into `state_per_pos`
    /// `[B, T, Hv, Dv, Dk]`. Used for O(1) speculative rollback — on a partial
    /// reject, the committed SSM state is `state_per_pos[:, keep-1]` (an
    /// index-select), replacing the O(D) full-trunk replay. The state is f32
    /// (StT), so the last-position slice is bit-identical to `state_out`.
    pub const GATED_DELTA_CAPTURE_KERNEL_SOURCE: &str = r#"
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_[hv_idx];
            kv_mem += state[i] * k_[s_idx];
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_[s_idx] * delta;
            out += state[i] * q_[s_idx];
          }
          out = simd_sum(out);
          if (thread_index_in_simdgroup == 0) {
            y[dv_idx] = static_cast<InT>(out);
          }

          // Per-position state capture: state_per_pos[b, t, hv, dv, :]
          auto sp = state_per_pos + (((((size_t)b_idx * T + t) * Hv + hv_idx) * Dv + dv_idx) * Dk);
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            sp[s_idx] = static_cast<StT>(state[i]);
          }

          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }
    "#;

    /// Capture-variant dispatch. Returns `(y, state_out, state_per_pos)` where
    /// `state_per_pos: [B, seq, Hv, Dv, Dk]` (dtype = state_in.dtype).
    pub fn gated_delta_step_kernel_capture(
        q: &Array,
        k: &Array,
        v: &Array,
        g: &Array,
        beta: &Array,
        state_in: &Array,
    ) -> Result<(Array, Array, Array)> {
        let mut outputs = gated_delta_dispatch(q, k, v, g, beta, state_in, true)?.into_iter();
        let y = outputs.next().expect("kernel produced y");
        let state_out = outputs.next().expect("kernel produced state_out");
        let state_per_pos = outputs.next().expect("kernel produced state_per_pos");
        Ok((y, state_out, state_per_pos))
    }

    /// `Qwen3NextRMSNormGated.__call__` (gate path):
    ///
    /// ```text
    /// x   = mx.fast.rms_norm(hidden_states, weight, eps)
    /// out = silu(gate.astype(f32)) * x.astype(f32)
    /// out = out.astype(hidden_states.dtype)
    /// ```
    ///
    /// All three input shapes match: `hidden_states`, `gate`, both
    /// `[..., hidden]`. `weight` is `[hidden]`.
    ///
    /// When `LUMEN_NATIVE_RMS_NORM_GATED_FUSED=1` is set the
    /// body dispatches to a single runtime-compiled Metal kernel
    /// (`rms_norm_gated_fused`) that fuses rms_norm + silu(gate) + multiply
    /// into one mlx-rs op. Defaults OFF — fused output is within fp32
    /// tolerance of the composed path but is NOT bit-identical to
    /// `mlx.fast.rms_norm`'s reduction order, so PyO3 golden bit-identical
    /// parity is only guaranteed on the default path.
    pub fn rms_norm_gated(
        hidden_states: &Array,
        gate: &Array,
        weight: &Array,
        eps: f32,
    ) -> Result<Array> {
        if rms_norm_gated_fused_enabled() {
            return rms_norm_gated_fused(hidden_states, gate, weight, eps);
        }
        rms_norm_gated_legacy(hidden_states, gate, weight, eps)
    }

    /// Legacy composed-ops path: 6 mlx-rs ops (rms_norm + 2 casts + silu +
    /// multiply + cast). Used by default and on env=0. Kept as the parity
    /// oracle for the fused path.
    pub fn rms_norm_gated_legacy(
        hidden_states: &Array,
        gate: &Array,
        weight: &Array,
        eps: f32,
    ) -> Result<Array> {
        let x = mlx_rs::fast::rms_norm(hidden_states, weight, eps)
            .context("rms_norm_gated: fast::rms_norm failed")?;
        let gate_f32 = gate
            .as_dtype(mlx_rs::Dtype::Float32)
            .context("rms_norm_gated: gate → f32 cast failed")?;
        let gate_silu = mlx_rs::nn::silu(&gate_f32).context("rms_norm_gated: silu(gate) failed")?;
        let x_f32 = x
            .as_dtype(mlx_rs::Dtype::Float32)
            .context("rms_norm_gated: x → f32 cast failed")?;
        let prod = gate_silu
            .multiply(&x_f32)
            .context("rms_norm_gated: silu(gate) * x failed")?;
        prod.as_dtype(hidden_states.dtype())
            .context("rms_norm_gated: output → hidden_states dtype cast failed")
    }

    lumen_flags::flag! {
        /// Fused rms_norm + silu(gate) + multiply Metal kernel. Default OFF —
        /// reserved for super-kernel composition. (Parse note: previously only
        /// `=1` enabled this; the uniform rule now accepts any non-`"0"`.)
        pub(crate) rms_norm_gated_fused {
            env: "LUMEN_NATIVE_RMS_NORM_GATED_FUSED",
            default: false,
            kind: Optimization,
        }
    }

    fn rms_norm_gated_fused_enabled() -> bool {
        rms_norm_gated_fused::get()
    }

    // ─── Phase G5-A: fused rms_norm + silu(gate) + multiply ──────────────
    //
    // Single runtime-compiled Metal kernel that performs the same math as
    // the composed `rms_norm_gated_legacy` but in one mlx-rs op call. The
    // goal is to amortize mlx-rs's per-op `Guarded::try_from_op` overhead
    // and shrink the lazy-graph node count walked by `mlx::core::eval_impl`
    // — both root causes of the residual native vs pyo3 decode gap
    // documented in `.ai/memory/notes/g4_fork_lever_audit_concluded.md`.
    //
    // Math (per row of size D, identical to `rms_norm_gated_legacy`):
    //     sumsq    = Σ x[i]²              (i ∈ 0..D)
    //     inv_rms  = rsqrt(sumsq / D + eps)
    //     y[i]     = silu(z[i]) * x[i] * inv_rms * weight[i]
    //              = (z[i] · σ(z[i])) · x[i] · inv_rms · weight[i]
    //
    // Layout: leading dims (B, S, Hv) are flattened to N rows; each row
    // is `D = hidden` contiguous elements (last axis). Mirrors the
    // semantics of `mlx.fast.rms_norm` which always reduces along the
    // last axis.
    //
    // Grid: (32, 1, N_rows). Threadgroup: (32, 1, 1). One simdgroup per
    // row; 32 lanes cooperate on the D-wide Σx² reduction via `simd_sum`.
    //
    // Templates:
    //   - DT : input/output element dtype (same for x, z, weight, y)
    //   - D  : hidden dimension (must be multiple of 32; checked at the
    //          dispatch site for clarity)
    //
    // Bit-equality vs `mlx.fast.rms_norm` is NOT guaranteed because MLX's
    // built-in rms_norm may use a different reduction tree. Parity gate
    // is therefore tolerance-based against `rms_norm_gated_legacy`.
    pub const RMS_NORM_GATED_KERNEL_SOURCE: &str = r#"
        auto row = thread_position_in_grid.z;
        auto lane = thread_index_in_simdgroup;
        constexpr int per_lane = D / 32;

        auto x_row = x + row * D;
        auto z_row = z + row * D;
        y += row * D;

        float sumsq = 0.0f;
        for (int i = 0; i < per_lane; ++i) {
            int idx = i * 32 + lane;
            float v = static_cast<float>(x_row[idx]);
            sumsq = fma(v, v, sumsq);
        }
        sumsq = simd_sum(sumsq);
        float inv_rms = rsqrt(sumsq / float(D) + eps);

        for (int i = 0; i < per_lane; ++i) {
            int idx = i * 32 + lane;
            float xv = static_cast<float>(x_row[idx]);
            float zv = static_cast<float>(z_row[idx]);
            float silu = zv * (1.0f / (1.0f + metal::exp(-zv)));
            float xn = xv * inv_rms * static_cast<float>(weight[idx]);
            y[idx] = static_cast<DT>(silu * xn);
        }
    "#;

    /// Fused `rms_norm + silu(gate) + multiply` via a single MLX runtime
    /// Metal kernel. See `RMS_NORM_GATED_KERNEL_SOURCE` for the math.
    ///
    /// Output dtype matches `hidden_states.dtype()` (also the kernel's `DT`
    /// template). Leading dims of `hidden_states` and `gate` must match;
    /// `weight` is rank-1 with size = last axis.
    pub fn rms_norm_gated_fused(
        hidden_states: &Array,
        gate: &Array,
        weight: &Array,
        eps: f32,
    ) -> Result<Array> {
        use crate::metal_kernel::{MetalKernel, MetalKernelConfig};

        let h_shape = hidden_states.shape();
        let g_shape = gate.shape();
        let w_shape = weight.shape();
        if h_shape != g_shape {
            return Err(anyhow::anyhow!(
                "rms_norm_gated_fused: shape mismatch hidden_states={:?} vs gate={:?}",
                h_shape,
                g_shape
            ));
        }
        if w_shape.len() != 1 {
            return Err(anyhow::anyhow!(
                "rms_norm_gated_fused: weight must be rank-1, got shape {:?}",
                w_shape
            ));
        }
        let d = *h_shape
            .last()
            .ok_or_else(|| anyhow::anyhow!("rms_norm_gated_fused: hidden_states is scalar"))?;
        if w_shape[0] != d {
            return Err(anyhow::anyhow!(
                "rms_norm_gated_fused: weight len {} does not match hidden_states last dim {}",
                w_shape[0],
                d
            ));
        }
        if d % 32 != 0 {
            return Err(anyhow::anyhow!(
                "rms_norm_gated_fused: hidden (last axis) must be a multiple of 32 (kernel uses simdgroup-wide reduction), got {d}"
            ));
        }
        if hidden_states.dtype() != gate.dtype() {
            return Err(anyhow::anyhow!(
                "rms_norm_gated_fused: hidden_states dtype {:?} != gate dtype {:?}",
                hidden_states.dtype(),
                gate.dtype()
            ));
        }
        let n_rows: i32 = h_shape.iter().take(h_shape.len() - 1).product();

        let kernel = MetalKernel::new(
            "rms_norm_gated",
            &["x", "z", "weight", "eps"],
            &["y"],
            RMS_NORM_GATED_KERNEL_SOURCE,
            /* ensure_row_contiguous */ true,
            /* atomic_outputs */ false,
        )?;

        let config = MetalKernelConfig::new();
        config.add_template_arg_dtype("DT", hidden_states.dtype())?;
        config.add_template_arg_int("D", d)?;

        config.set_grid(32, 1, n_rows)?;
        config.set_thread_group(32, 1, 1)?;

        config.add_output_arg(h_shape, hidden_states.dtype())?;

        let eps_scalar = Array::from_f32(eps);
        let outputs = kernel.apply(&[hidden_states, gate, weight, &eps_scalar], &config, 1)?;
        let mut iter = outputs.into_iter();
        let y = iter
            .next()
            .ok_or_else(|| anyhow::anyhow!("rms_norm_gated_fused: kernel produced no output"))?;
        Ok(y)
    }

    // ─── Fused linear-attn prework for a decode token or a verify ────────
    //
    // Everything a gated-delta layer does between its in-projections and the recurrence, for `T`
    // tokens, in ONE dispatch: the conv window over `[conv_state; mixed_qkv]` (a depthwise conv
    // with `K` taps), SiLU, the q/k/v split, the per-head q/k RMSNorm, the next conv state, and
    // the two gates `beta = sigmoid(b)` and `g = exp(-exp(A_log) * softplus(a + dt_bias))`.
    // The composed path is a concat, a slice, a conv1d, a SiLU, two RMS norms, a sigmoid and a
    // compiled `compute_g`: eight dispatches in a dependent chain, which is what costs (the chain
    // is not bandwidth). `examples/m2c_inproj_tail_fused.rs` is the standalone parity harness of
    // the first version, which left the concat, the slice and the gates outside.
    //
    // Layout: `conv_state` [K-1, conv_dim] (the K-1 inputs before the first token), `mixed_qkv`
    // [T, conv_dim], `conv_w` [conv_dim, K], `wq`/`wk` [dn] (scale folded), `b_pre`/`a_pre`
    // [T, NV]. conv_dim = 2*key_dim + value_dim, key_dim = NK*DN, value_dim = NV*DV. grid (lane,
    // row, token): row 0..NK = q heads, NK..2NK = k heads, 2NK..2NK+NV = v heads; lane = channel
    // within head, so every channel of the conv state is owned by exactly one thread of each
    // token, and the first token's writes the channel's column of the next state.
    //
    // The arithmetic is the composed ops' arithmetic: the window sum accumulates taps in order
    // (the concat's row order), and the gates use the functors MLX's kernels use (`Sigmoid`,
    // `LogAddExp` for softplus, precise `exp`). Token t's sums are the sums one token would do
    // with the window ending at t, so a verify's rows are a decode's rows, bit for bit.
    const LINATTN_PREWORK_SOURCE: &str = r#"
        uint d = thread_position_in_grid.x;
        uint row = thread_position_in_grid.y;
        uint t = thread_position_in_grid.z;
        threadgroup float sh[128];
        uint nk = (uint)NK, nv = (uint)NV;
        uint dn = (uint)DN, dv = (uint)DV;
        uint key_dim = nk * dn;
        uint conv_dim = 2u * key_dim + nv * dv;
        uint region, head, width, ch_base;
        if (row < nk)         { region = 0u; head = row;         width = dn; ch_base = head * dn; }
        else if (row < 2u*nk) { region = 1u; head = row - nk;    width = dn; ch_base = key_dim + head * dn; }
        else                  { region = 2u; head = row - 2u*nk; width = dv; ch_base = 2u*key_dim + head * dv; }
        if (d >= width) return;
        uint c = ch_base + d;
        // Row r of the window [conv_state; mixed]: the state's K-1 rows, then the T tokens.
        #define WINDOW(r) ((r) + 1u < (uint)K ? conv_state[(r) * conv_dim + c] \
                                              : mixed[((r) + 1u - (uint)K) * conv_dim + c])

        float acc = 0.0f;
        for (uint k = 0u; k < (uint)K - 1u; ++k) {
            acc += float(WINDOW(t + k)) * float(conv_w[c * (uint)K + k]);
        }
        acc += float(mixed[t * conv_dim + c]) * float(conv_w[c * (uint)K + (uint)K - 1u]);
        if (t == 0u) {
            for (uint j = 0u; j + 1u < (uint)K; ++j) {
                new_state[j * conv_dim + c] = WINDOW((uint)T + j);
            }
        }
        #undef WINDOW

        if (region == 2u && d == 0u) {
            uint gi = t * nv + head;
            float bv = float(b_pre[gi]);
            float ey = 1.0f / (1.0f + metal::precise::exp(metal::abs(bv)));
            beta[gi] = (bv < 0.0f) ? ey : 1.0f - ey;
            float x = float(a_pre[gi]) + float(dt_bias[head]);
            float hi = metal::max(x, 0.0f);
            float lo = metal::min(x, 0.0f);
            float softplus = (lo == -INFINITY || hi == INFINITY)
                ? hi
                : (hi + log1p(metal::exp(lo - hi)));
            float decay = metal::precise::exp(float(a_log[head])) * softplus;
            g[gi] = metal::precise::exp(-decay);
        }

        // SiLU as MLX composes it: x * sigmoid(x), with its `Sigmoid` functor.
        float sy = 1.0f / (1.0f + metal::precise::exp(metal::abs(acc)));
        float s = acc * ((acc < 0.0f) ? sy : 1.0f - sy);
        if (region == 2u) { out_v[(t * nv + head) * dv + d] = static_cast<InT>(s); return; }

        // The per-head RMS norm as MLX's `rms_single_row` does it for a row of at most 128: the
        // first simdgroup's lane l sums the squares of channels 4l..4l+3 in order, `simd_sum`
        // adds the lanes, and `precise::rsqrt` of the mean. (Same sums, same order, same bits.)
        threadgroup float inv_mean[1];
        sh[d] = s;
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        if (d < 32u) {
            float part = 0.0f;
            if (d * 4u < width) {
                for (uint i = 0u; i < 4u; ++i) {
                    float v = sh[d * 4u + i];
                    part += v * v;
                }
            }
            float total = simd_sum(part);
            if (d == 0u) inv_mean[0] = metal::precise::rsqrt(total / float(width) + float(eps[0]));
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        float wv = (region == 0u) ? float(wq[d]) : float(wk[d]);
        float y = wv * static_cast<InT>(s * inv_mean[0]);
        if (region == 0u) out_q[(t * nk + head) * dn + d] = static_cast<InT>(y);
        else              out_k[(t * nk + head) * dn + d] = static_cast<InT>(y);
    "#;

    /// What [`linattn_prework_fused`] returns for `T` tokens.
    pub struct LinAttnPrework {
        /// `[T, NK, DN]`, scale folded.
        pub q: Array,
        /// `[T, NK, DN]`, scale folded.
        pub k: Array,
        /// `[T, NV, DV]`.
        pub v: Array,
        /// `[K-1, conv_dim]`: the conv state after the last token.
        pub new_state: Array,
        /// `[T, NV]` f32: `sigmoid(b)`.
        pub beta: Array,
        /// `[T, NV]` f32: the decay `exp(-exp(A_log) * softplus(a + dt_bias))`.
        pub g: Array,
    }

    thread_local! {
        static LINATTN_PREWORK: std::cell::OnceCell<crate::metal_kernel::MetalKernel> =
            const { std::cell::OnceCell::new() };
    }

    /// The fused prework of `T` consecutive tokens; see the comment above. `conv_state`:
    /// `[K-1, conv_dim]`, `mixed_qkv`: `[T, conv_dim]`, or `[conv_dim]` for one token (both of one
    /// dtype, which the outputs take), `conv_w`: `[conv_dim, K]`, `wq`/`wk`: `[DN]` (scale
    /// folded), `b_pre`/`a_pre`: `T * NV` values, token-major (the b and a in-projections),
    /// `a_log`/`dt_bias`: `[NV]`.
    #[allow(clippy::too_many_arguments)]
    pub fn linattn_prework_fused(
        conv_state: &Array,
        mixed_qkv: &Array,
        conv_w: &Array,
        wq: &Array,
        wk: &Array,
        eps: f32,
        b_pre: &Array,
        a_pre: &Array,
        a_log: &Array,
        dt_bias: &Array,
        nk: i32,
        nv: i32,
        dn: i32,
        dv: i32,
        k_taps: i32,
    ) -> Result<LinAttnPrework> {
        use crate::metal_kernel::{MetalKernel, MetalKernelConfig, with_thread_kernel};
        if k_taps < 2 {
            return Err(anyhow::anyhow!(
                "linattn_prework_fused: needs at least 2 conv taps, got {k_taps}"
            ));
        }
        // The q/k norm reproduces `rms_single_row`: one simdgroup, four channels per lane.
        if dn > 128 || dn % 4 != 0 || dv > 1024 {
            return Err(anyhow::anyhow!(
                "linattn_prework_fused: head dims dn={dn} dv={dv} outside what the kernel's norm covers"
            ));
        }
        let dtype = mixed_qkv.dtype();
        let conv_dim = 2 * nk * dn + nv * dv;
        let tokens = mixed_qkv.size() as i32 / conv_dim;
        if tokens < 1 || mixed_qkv.size() as i32 != tokens * conv_dim {
            return Err(anyhow::anyhow!(
                "linattn_prework_fused: mixed_qkv {:?} is not whole tokens of {conv_dim}",
                mixed_qkv.shape()
            ));
        }
        for (name, a) in [("b_pre", b_pre), ("a_pre", a_pre)] {
            if a.size() as i32 != tokens * nv {
                return Err(anyhow::anyhow!(
                    "linattn_prework_fused: {name} {:?} is not {tokens} tokens of {nv} heads",
                    a.shape()
                ));
            }
        }
        let config = MetalKernelConfig::new();
        config.add_template_arg_dtype("InT", dtype)?;
        config.add_template_arg_int("NK", nk)?;
        config.add_template_arg_int("NV", nv)?;
        config.add_template_arg_int("DN", dn)?;
        config.add_template_arg_int("DV", dv)?;
        config.add_template_arg_int("K", k_taps)?;
        config.add_template_arg_int("T", tokens)?;
        config.set_grid(dn.max(dv), nk + nk + nv, tokens)?;
        config.set_thread_group(dn.max(dv), 1, 1)?;
        config.add_output_arg(&[tokens, nk, dn], dtype)?;
        config.add_output_arg(&[tokens, nk, dn], dtype)?;
        config.add_output_arg(&[tokens, nv, dv], dtype)?;
        config.add_output_arg(&[k_taps - 1, conv_dim], dtype)?;
        config.add_output_arg(&[tokens, nv], mlx_rs::Dtype::Float32)?;
        config.add_output_arg(&[tokens, nv], mlx_rs::Dtype::Float32)?;
        let eps_arr = Array::from_slice(&[eps], &[1]);
        // Each kernel is constructed with its name as a literal: `cargo xtask test` reads the
        // names off `MetalKernel::new` calls to check that every kernel ran in its Metal 3.1 pass.
        let outs = with_thread_kernel(
            &LINATTN_PREWORK,
            || {
                MetalKernel::new(
                    "linattn_prework_fused",
                    &[
                        "conv_state",
                        "mixed",
                        "conv_w",
                        "wq",
                        "wk",
                        "eps",
                        "b_pre",
                        "a_pre",
                        "a_log",
                        "dt_bias",
                    ],
                    &["out_q", "out_k", "out_v", "new_state", "beta", "g"],
                    LINATTN_PREWORK_SOURCE,
                    /* ensure_row_contiguous */ true,
                    /* atomic_outputs */ false,
                )
            },
            |kernel| {
                kernel.apply(
                    &[
                        conv_state, mixed_qkv, conv_w, wq, wk, &eps_arr, b_pre, a_pre, a_log,
                        dt_bias,
                    ],
                    &config,
                    6,
                )
            },
        )?;
        let mut it = outs.into_iter();
        let mut next = |what: &str| {
            it.next()
                .ok_or_else(|| anyhow::anyhow!("linattn_prework_fused: no {what}"))
        };
        Ok(LinAttnPrework {
            q: next("out_q")?,
            k: next("out_k")?,
            v: next("out_v")?,
            new_state: next("new_state")?,
            beta: next("beta")?,
            g: next("g")?,
        })
    }
}

#[cfg(feature = "mlx-native")]
#[allow(unused_imports)] // Consumed by Phase 3b.5 SSM block assembly.
pub use imp::{
    LinAttnPrework, compute_g, gated_delta_step_kernel, gated_delta_step_kernel_capture,
    linattn_prework_fused, rms_norm_gated, rms_norm_gated_fused, rms_norm_gated_legacy,
};

#[cfg(all(test, feature = "mlx-native"))]
mod capture_kernel_tests {
    use super::imp::{gated_delta_step_kernel, gated_delta_step_kernel_capture};
    use mlx_rs::{Array, Dtype};

    /// Deterministic inputs in [-0.5, 0.5]: `sin((i + 1) * seed) / 2`.
    pub(super) fn mk(n: usize, seed: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32 + 1.0) * seed).sin() * 0.5)
            .collect()
    }
    /// Computed on the GPU, whose `max` propagates NaN, so a NaN output fails
    /// the bit-identity checks below. `fused_kernel_tests` has its own,
    /// host-side one for its tolerance checks.
    fn max_abs_diff(a: &Array, b: &Array) -> f32 {
        let d = mlx_rs::ops::abs(mlx_rs::ops::subtract(a, b).unwrap()).unwrap();
        let m = d.max(None).unwrap().as_dtype(Dtype::Float32).unwrap();
        m.eval().unwrap();
        m.as_slice::<f32>()[0]
    }

    #[test]
    fn capture_matches_base_and_per_position() {
        capture_matches_base_and_per_position_as(Dtype::Float32);
    }

    /// What the models run: bf16 activations over an f32 state — a separate
    /// instantiation of both kernels.
    #[test]
    fn capture_matches_base_and_per_position_in_bf16() {
        capture_matches_base_and_per_position_as(Dtype::Bfloat16);
    }

    fn capture_matches_base_and_per_position_as(dt: Dtype) {
        let (b, t, hk, hv, dk, dv) = (1i32, 4i32, 1i32, 2i32, 32i32, 4i32);
        let act = |n: i32, seed: f32, shape: &[i32]| {
            Array::from_slice(&mk(n as usize, seed), shape)
                .as_dtype(dt)
                .unwrap()
        };
        let q = act(b * t * hk * dk, 0.11, &[b, t, hk, dk]);
        let k = act(b * t * hk * dk, 0.13, &[b, t, hk, dk]);
        let v = act(b * t * hv * dv, 0.17, &[b, t, hv, dv]);
        let g = act(b * t * hv, 0.23, &[b, t, hv]);
        let beta = act(b * t * hv, 0.19, &[b, t, hv]);
        let state = Array::from_slice(&mk((b * hv * dv * dk) as usize, 0.05), &[b, hv, dv, dk]);

        let (y_b, s_b) = gated_delta_step_kernel(&q, &k, &v, &g, &beta, &state).unwrap();
        let (y_c, s_c, sp) =
            gated_delta_step_kernel_capture(&q, &k, &v, &g, &beta, &state).unwrap();
        for a in [&y_b, &s_b, &y_c, &s_c, &sp] {
            a.eval().unwrap();
        }

        // y and final state bit-identical between base and capture.
        assert_eq!(max_abs_diff(&y_b, &y_c), 0.0, "y base vs capture");
        assert_eq!(max_abs_diff(&s_b, &s_c), 0.0, "state_out base vs capture");

        // state_per_pos[:, T-1] == state_out  (last position == final state).
        let parts = mlx_rs::ops::split(&sp, t, 1).unwrap(); // T parts [b,1,hv,dv,dk]
        let last = parts[(t - 1) as usize].reshape(&[b, hv, dv, dk]).unwrap();
        last.eval().unwrap();
        assert_eq!(
            max_abs_diff(&last, &s_b),
            0.0,
            "per-pos[T-1] vs final state"
        );

        // state_per_pos[:, 0] == base run over only token 0 (T=1).
        let q0 = mlx_rs::ops::split(&q, t, 1).unwrap()[0].clone();
        let k0 = mlx_rs::ops::split(&k, t, 1).unwrap()[0].clone();
        let v0 = mlx_rs::ops::split(&v, t, 1).unwrap()[0].clone();
        let g0 = mlx_rs::ops::split(&g, t, 1).unwrap()[0].clone();
        let beta0 = mlx_rs::ops::split(&beta, t, 1).unwrap()[0].clone();
        let (_y0, s0) = gated_delta_step_kernel(&q0, &k0, &v0, &g0, &beta0, &state).unwrap();
        let first = parts[0].reshape(&[b, hv, dv, dk]).unwrap();
        s0.eval().unwrap();
        first.eval().unwrap();
        assert_eq!(
            max_abs_diff(&first, &s0),
            0.0,
            "per-pos[0] vs T=1 final state"
        );
        eprintln!("capture kernel OK: y/state bit-identical, per-pos[0] and [T-1] verified");
    }
}

// Every run-time kernel in this file has to run in a test with no fixture, in
// f32 and in the bf16 the models use: `cargo xtask test`'s macOS 14 pass
// (Metal 3.1) only checks instantiations a test actually runs.
#[cfg(all(test, feature = "mlx-native"))]
mod fused_kernel_tests {
    use super::capture_kernel_tests::mk;
    use super::imp::{
        compute_g, linattn_prework_fused, rms_norm_gated_fused, rms_norm_gated_legacy,
    };
    use mlx_rs::{Array, Dtype};

    fn values(a: &Array) -> Vec<f32> {
        let a = a.as_dtype(Dtype::Float32).unwrap().reshape(&[-1]).unwrap();
        a.eval().unwrap();
        a.as_slice::<f32>().to_vec()
    }

    fn array(xs: &[f32], shape: &[i32], dt: Dtype) -> Array {
        Array::from_slice(xs, shape).as_dtype(dt).unwrap()
    }

    /// bf16 keeps ~3 significant digits; the outputs here stay below ~2.
    fn tolerance(dt: Dtype) -> f32 {
        if dt == Dtype::Float32 { 1e-4 } else { 3e-2 }
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    /// The fused kernel against the composed ops it replaces.
    #[test]
    fn rms_norm_gated_fused_matches_the_composed_path() {
        rms_norm_gated_fused_matches_the_composed_path_as(Dtype::Float32);
    }

    #[test]
    fn rms_norm_gated_fused_matches_the_composed_path_in_bf16() {
        rms_norm_gated_fused_matches_the_composed_path_as(Dtype::Bfloat16);
    }

    fn rms_norm_gated_fused_matches_the_composed_path_as(dt: Dtype) {
        let (rows, d) = (6i32, 64i32);
        let x = array(&mk((rows * d) as usize, 0.13), &[rows, d], dt);
        let gate = array(&mk((rows * d) as usize, 0.07), &[rows, d], dt);
        let w: Vec<f32> = mk(d as usize, 0.31).iter().map(|v| 1.0 + v).collect();
        let w = array(&w, &[d], dt);
        let fused = values(&rms_norm_gated_fused(&x, &gate, &w, 1e-6).unwrap());
        let legacy = values(&rms_norm_gated_legacy(&x, &gate, &w, 1e-6).unwrap());
        let diff = max_abs_diff(&fused, &legacy);
        assert!(
            diff < tolerance(dt),
            "{dt:?} fused vs composed: max |diff| {diff}"
        );
    }

    /// Inputs of the prework kernel for `tokens` tokens.
    struct Prework {
        nk: usize,
        nv: usize,
        dn: usize,
        dv: usize,
        taps: usize,
        tokens: usize,
        conv_state: Array,
        mixed: Array,
        conv_w: Array,
        wq: Array,
        wk: Array,
        b_pre: Array,
        a_pre: Array,
        a_log: Array,
        dt_bias: Array,
    }

    const EPS: f32 = 1e-6;

    impl Prework {
        fn new(nk: usize, nv: usize, dn: usize, dv: usize, seed: f32, dt: Dtype) -> Self {
            Self::tokens(nk, nv, dn, dv, 1, seed, dt)
        }

        fn tokens(
            nk: usize,
            nv: usize,
            dn: usize,
            dv: usize,
            tokens: usize,
            seed: f32,
            dt: Dtype,
        ) -> Self {
            let taps = 4usize;
            let conv_dim = 2 * nk * dn + nv * dv;
            let t = tokens as i32;
            let scaled =
                |n: usize, s: f32| -> Vec<f32> { mk(n, s).iter().map(|v| 1.0 + v).collect() };
            // Spread the gate inputs over the branches of sigmoid and softplus (negative and
            // positive, small and large).
            let wide = |n: usize, s: f32, span: f32| -> Vec<f32> {
                mk(n, s).iter().map(|v| v * span).collect()
            };
            // One token's inputs keep the one-dimensional shapes a decode passes.
            let per_token = |n: usize| -> Vec<i32> {
                if tokens == 1 {
                    vec![n as i32]
                } else {
                    vec![t, n as i32]
                }
            };
            Self {
                nk,
                nv,
                dn,
                dv,
                taps,
                tokens,
                conv_state: array(
                    &mk((taps - 1) * conv_dim, 0.17 + seed),
                    &[(taps - 1) as i32, conv_dim as i32],
                    dt,
                ),
                mixed: array(
                    &mk(tokens * conv_dim, 0.11 + seed),
                    &per_token(conv_dim),
                    dt,
                ),
                conv_w: array(
                    &mk(conv_dim * taps, 0.05 + seed),
                    &[conv_dim as i32, taps as i32],
                    dt,
                ),
                wq: array(&scaled(dn, 0.23 + seed), &[dn as i32], dt),
                wk: array(&scaled(dn, 0.29 + seed), &[dn as i32], dt),
                b_pre: array(
                    &wide(tokens * nv, 0.37 + seed, 24.0),
                    &per_token(nv),
                    Dtype::Float32,
                ),
                a_pre: array(
                    &wide(tokens * nv, 0.41 + seed, 40.0),
                    &per_token(nv),
                    Dtype::Float32,
                ),
                a_log: array(&wide(nv, 0.43 + seed, 3.0), &[nv as i32], Dtype::Float32),
                dt_bias: array(&wide(nv, 0.47 + seed, 2.0), &[nv as i32], dt),
            }
        }

        fn run(&self) -> super::imp::LinAttnPrework {
            self.run_on(&self.conv_state, &self.mixed, &self.b_pre, &self.a_pre)
        }

        /// The kernel on this layer's weights with other token inputs.
        fn run_on(
            &self,
            conv_state: &Array,
            mixed: &Array,
            b_pre: &Array,
            a_pre: &Array,
        ) -> super::imp::LinAttnPrework {
            linattn_prework_fused(
                conv_state,
                mixed,
                &self.conv_w,
                &self.wq,
                &self.wk,
                EPS,
                b_pre,
                a_pre,
                &self.a_log,
                &self.dt_bias,
                self.nk as i32,
                self.nv as i32,
                self.dn as i32,
                self.dv as i32,
                self.taps as i32,
            )
            .unwrap()
        }

        fn conv_dim(&self) -> usize {
            2 * self.nk * self.dn + self.nv * self.dv
        }

        /// `[1, K-1+T, conv_dim]`: the conv state, then the tokens.
        fn window(&self) -> Array {
            let c = self.conv_dim() as i32;
            let state3 = self
                .conv_state
                .reshape(&[1, self.taps as i32 - 1, c])
                .unwrap();
            let mixed3 = self.mixed.reshape(&[1, self.tokens as i32, c]).unwrap();
            mlx_rs::ops::concatenate_axis(&[&state3, &mixed3], 1).unwrap()
        }
    }

    /// The prework kernel against the ops it replaces: concat, slice, conv1d, SiLU, split, two RMS
    /// norms, sigmoid and the compiled `compute_g`.
    #[test]
    fn linattn_prework_matches_the_composed_ops() {
        linattn_prework_matches_the_composed_ops_as(Dtype::Float32);
    }

    #[test]
    fn linattn_prework_matches_the_composed_ops_in_bf16() {
        linattn_prework_matches_the_composed_ops_as(Dtype::Bfloat16);
    }

    /// The Qwen3.6-35B-A3B shape (16 key heads, 32 value heads, 128 channels each): enough norm
    /// rows that a one-ulp difference in the reciprocal square root shows.
    #[test]
    fn linattn_prework_matches_the_composed_ops_at_the_a3b_shape() {
        for seed in [0.0, 0.31, 0.77, 1.9] {
            linattn_prework_matches_the_composed_ops_with(
                Dtype::Float32,
                16,
                32,
                128,
                128,
                1,
                seed,
            );
        }
    }

    /// An MTP verify's widths, at the Qwen3.6/3.8-27B shape (16 key heads, 48 value heads).
    #[test]
    fn linattn_prework_of_a_verify_matches_the_composed_ops() {
        for (tokens, seed) in [(2, 0.0), (3, 0.31), (4, 0.77)] {
            linattn_prework_matches_the_composed_ops_with(
                Dtype::Float32,
                16,
                48,
                128,
                128,
                tokens,
                seed,
            );
        }
        linattn_prework_matches_the_composed_ops_with(Dtype::Bfloat16, 2, 4, 32, 32, 3, 0.0);
    }

    fn linattn_prework_matches_the_composed_ops_as(dt: Dtype) {
        linattn_prework_matches_the_composed_ops_with(dt, 2, 4, 32, 32, 1, 0.0);
    }

    fn linattn_prework_matches_the_composed_ops_with(
        dt: Dtype,
        nk: usize,
        nv: usize,
        dn: usize,
        dv: usize,
        tokens: usize,
        seed: f32,
    ) {
        use mlx_rs::ops::indexing::TryIndexOp;
        let w = Prework::tokens(nk, nv, dn, dv, tokens, seed, dt);
        let got = w.run();

        let (key_dim, conv_dim) = (nk * dn, w.conv_dim());
        let (taps, t) = (w.taps as i32, tokens as i32);
        let conv_input = w.window();
        let conv_w = w.conv_w.reshape(&[conv_dim as i32, taps, 1]).unwrap();
        let conv =
            crate::native_conv1d::conv1d(&conv_input, &conv_w, 1, 0, conv_dim as i32).unwrap();
        let conv = mlx_rs::nn::silu(&conv).unwrap();
        let parts =
            mlx_rs::ops::split_sections(&conv, &[key_dim as i32, 2 * key_dim as i32], -1).unwrap();
        let qk = |part: &Array, weight: &Array| {
            let r = part.reshape(&[1, t, nk as i32, dn as i32]).unwrap();
            mlx_rs::fast::rms_norm(&r, weight, EPS).unwrap()
        };
        let want_q = qk(&parts[0], &w.wq);
        let want_k = qk(&parts[1], &w.wk);
        let want_v = parts[2].clone();
        let want_state = conv_input.try_index((.., t..t + taps - 1, ..)).unwrap();

        for (name, got, want) in [
            ("q", &got.q, &want_q),
            ("k", &got.k, &want_k),
            ("v", &got.v, &want_v),
        ] {
            let (g, w) = (values(got), values(want));
            if dt == Dtype::Float32 {
                // The dtype the model runs this in: the same sums in the same order as MLX's
                // conv, SiLU and `rms_single_row`, so the same bits.
                let differing = g.iter().zip(&w).filter(|(a, b)| a.to_bits() != b.to_bits());
                assert_eq!(
                    differing.count(),
                    0,
                    "{name}: not bit-identical to the composed ops"
                );
            } else {
                // The composed ops round to bf16 between steps; the kernel does not.
                let diff = max_abs_diff(&g, &w);
                assert!(diff < tolerance(dt), "{dt:?} {name}: max |diff| {diff}");
            }
        }
        assert_eq!(
            values(&got.new_state),
            values(&want_state),
            "{dt:?}: the next conv state is a copy of the window's last rows"
        );
    }

    /// Every token of a several-token call is what one token gives with the window ending at it,
    /// to the bit, in f32 and bf16: a verify's rows are the decode's rows, which is what keeps an
    /// MTP verify lossless.
    #[test]
    fn each_token_of_a_verify_is_the_decode_of_that_token() {
        use mlx_rs::ops::indexing::TryIndexOp;
        for (dt, tokens) in [(Dtype::Float32, 4usize), (Dtype::Bfloat16, 3)] {
            let w = Prework::tokens(16, 48, 128, 128, tokens, 0.53, dt);
            let all = w.run();
            let window = w.window();
            let (c, nv, taps) = (w.conv_dim() as i32, w.nv as i32, w.taps as i32);
            let bits = |a: &Array| -> Vec<u32> { values(a).iter().map(|v| v.to_bits()).collect() };
            let row = |a: &Array, t: i32| a.try_index(t).unwrap();
            for t in 0..tokens as i32 {
                let state = window
                    .try_index((0, t..t + taps - 1, ..))
                    .unwrap()
                    .reshape(&[taps - 1, c])
                    .unwrap();
                let one = w.run_on(
                    &state,
                    &row(&w.mixed, t),
                    &row(&w.b_pre, t),
                    &row(&w.a_pre, t),
                );
                for (name, got, want) in [
                    ("q", row(&all.q, t), &one.q),
                    ("k", row(&all.k, t), &one.k),
                    ("v", row(&all.v, t), &one.v),
                    ("beta", row(&all.beta, t), &one.beta),
                    ("g", row(&all.g, t), &one.g),
                ] {
                    assert_eq!(bits(&got), bits(want), "{dt:?} token {t} {name}");
                }
                if t == tokens as i32 - 1 {
                    assert_eq!(bits(&all.new_state), bits(&one.new_state), "{dt:?} state");
                }
            }
            assert_eq!(values(&all.beta).len(), tokens * nv as usize);
        }
    }

    /// The gates are the composed ops' gates to the bit, across the branches of sigmoid and
    /// softplus: the kernel replicates MLX's own functors, so a difference is a defect, not noise.
    #[test]
    fn linattn_prework_gates_are_the_composed_gates_to_the_bit() {
        // The composed gates are the compiled `compute_g` the model runs by default.
        super::imp::ssm_compile::with(true, || {
            // Many heads in one call, so one call covers many gate values.
            for (i, dt) in [Dtype::Float32, Dtype::Bfloat16, Dtype::Float32]
                .into_iter()
                .enumerate()
            {
                let w = Prework::new(1, 256, 32, 32, 0.013 * (i as f32 + 1.0), dt);
                let got = w.run();
                let want_beta = mlx_rs::ops::sigmoid(&w.b_pre).unwrap();
                let want_g = compute_g(&w.a_log, &w.a_pre, &w.dt_bias).unwrap();
                let bits =
                    |a: &Array| -> Vec<u32> { values(a).iter().map(|v| v.to_bits()).collect() };
                assert_eq!(bits(&got.beta), bits(&want_beta), "{dt:?} beta");
                assert_eq!(bits(&got.g), bits(&want_g), "{dt:?} g");
            }
        });
    }
}
