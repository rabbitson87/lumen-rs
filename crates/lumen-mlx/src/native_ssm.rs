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
    // equivalent to `compute_g_legacy` — both are exercised by the parity
    // fixtures.
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

    fn compile_enabled() -> bool {
        // Default ON (LANDED 2026-05-01 Phase 3): compute_g persistent compile
        // cache wins σ-significantly across 12-run 4-condition matrix
        // (Δp50=-0.33ms, Δtps=+1.96 vs legacy direct dispatch). Set
        // `LUMEN_NATIVE_COMPILE=0` to fall back to the legacy direct-ops path.
        std::env::var("LUMEN_NATIVE_COMPILE")
            .map(|v| v != "0")
            .unwrap_or(true)
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
        use crate::metal_kernel::{MetalKernel, MetalKernelConfig};

        // Pull dimensions from input shapes. Indices follow the Python ref.
        let q_shape = q.shape();
        let v_shape = v.shape();
        if q_shape.len() != 4 || v_shape.len() != 4 {
            return Err(anyhow::anyhow!(
                "gated_delta_step_kernel: q/v must be 4-D, got q={:?} v={:?}",
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
                "gated_delta_step_kernel: head_k_dim must be a multiple of 32 (kernel uses simdgroup reduction over a tile of 32), got {head_k_dim}"
            ));
        }

        let kernel = MetalKernel::new(
            "gated_delta_step",
            &["q", "k", "v", "g", "beta", "state_in", "T"],
            &["y", "state_out"],
            GATED_DELTA_KERNEL_SOURCE,
            /* ensure_row_contiguous */ true,
            /* atomic_outputs */ false,
        )?;

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

        // T is a 0-d int scalar input. MLX dereferences scalar arrays at
        // call sites, so the kernel sees T as an int value.
        let t_scalar = Array::from_int(seq);

        let outputs = kernel.apply(&[q, k, v, g, beta, state_in, &t_scalar], &config, 2)?;
        let mut iter = outputs.into_iter();
        let y = iter.next().expect("kernel produced y");
        let state_out = iter.next().expect("kernel produced state_out");
        Ok((y, state_out))
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
        use crate::metal_kernel::{MetalKernel, MetalKernelConfig};

        let q_shape = q.shape();
        let v_shape = v.shape();
        if q_shape.len() != 4 || v_shape.len() != 4 {
            return Err(anyhow::anyhow!(
                "gated_delta_step_kernel_capture: q/v must be 4-D, got q={:?} v={:?}",
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
                "gated_delta_step_kernel_capture: head_k_dim must be a multiple of 32, got {head_k_dim}"
            ));
        }

        let kernel = MetalKernel::new(
            "gated_delta_step_capture",
            &["q", "k", "v", "g", "beta", "state_in", "T"],
            &["y", "state_out", "state_per_pos"],
            GATED_DELTA_CAPTURE_KERNEL_SOURCE,
            /* ensure_row_contiguous */ true,
            /* atomic_outputs */ false,
        )?;

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
        config.add_output_arg(
            &[b, seq, n_v_heads, head_v_dim, head_k_dim],
            state_in.dtype(),
        )?;

        let t_scalar = Array::from_int(seq);
        let outputs = kernel.apply(&[q, k, v, g, beta, state_in, &t_scalar], &config, 3)?;
        let mut iter = outputs.into_iter();
        let y = iter.next().expect("kernel produced y");
        let state_out = iter.next().expect("kernel produced state_out");
        let state_per_pos = iter.next().expect("kernel produced state_per_pos");
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

    fn rms_norm_gated_fused_enabled() -> bool {
        std::env::var("LUMEN_NATIVE_RMS_NORM_GATED_FUSED")
            .map(|v| v == "1")
            .unwrap_or(false)
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

    // ─── Fused linear-attn input-side kernel (M2c) ───────────────────────
    //
    // conv1d(window-K depthwise) + silu + split + per-head q/k RMSNorm in ONE
    // Metal dispatch, producing q_scaled, k_scaled, v_post directly. Replaces
    // ~5 decode (s=1) launches/layer (conv1d, silu, split, rms_norm×2) with 1
    // — the launch-latency lever (gate-0 measured the chain at ~10ms/step).
    // Parity-validated standalone in `examples/m2c_inproj_tail_fused.rs`
    // (rel ~1e-7 vs the op-by-op path).
    //
    // Layout: conv_input [K, conv_dim] (K decode time-taps), conv_w [conv_dim, K],
    // wq/wk [dn] (scale folded). conv_dim = 2*key_dim + value_dim,
    // key_dim = NK*DN, value_dim = NV*DV. grid (lane, row): row 0..NK = q heads,
    // NK..2NK = k heads, 2NK..2NK+NV = v heads; lane = channel within head.
    const INPROJ_TAIL_SOURCE: &str = r#"
        uint d = thread_position_in_grid.x;
        uint row = thread_position_in_grid.y;
        threadgroup float sh[1024];
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
        float acc = 0.0f;
        for (uint k = 0u; k < (uint)K; ++k) {
            acc += float(conv_input[k * conv_dim + c]) * float(conv_w[c * (uint)K + k]);
        }
        float s = acc / (1.0f + metal::exp(-acc));
        if (region == 2u) { out_v[head * dv + d] = static_cast<InT>(s); return; }
        sh[d] = s * s;
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        for (uint stride = width >> 1; stride > 0u; stride >>= 1) {
            if (d < stride) sh[d] += sh[d + stride];
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        }
        float inv = metal::rsqrt(sh[0] / float(width) + float(eps[0]));
        float wv = (region == 0u) ? float(wq[d]) : float(wk[d]);
        float y = s * inv * wv;
        if (region == 0u) out_q[head * dn + d] = static_cast<InT>(y);
        else              out_k[head * dn + d] = static_cast<InT>(y);
    "#;

    /// Fused conv+silu+split+q/k-RMSNorm. Returns (q_scaled[NK,DN],
    /// k_scaled[NK,DN], v_post[NV,DV]). `conv_input`:[K,conv_dim] (contiguous),
    /// `conv_w`:[conv_dim,K], `wq`/`wk`:[DN] (scale folded). Bit-close to the
    /// op-by-op path (see examples/m2c_inproj_tail_fused.rs).
    #[allow(clippy::too_many_arguments)]
    pub fn inproj_tail_fused(
        conv_input: &Array,
        conv_w: &Array,
        wq: &Array,
        wk: &Array,
        eps: f32,
        nk: i32,
        nv: i32,
        dn: i32,
        dv: i32,
        k_taps: i32,
    ) -> Result<(Array, Array, Array)> {
        use crate::metal_kernel::{MetalKernel, MetalKernelConfig};
        let kernel = MetalKernel::new(
            "inproj_tail_fused",
            &["conv_input", "conv_w", "wq", "wk", "eps"],
            &["out_q", "out_k", "out_v"],
            INPROJ_TAIL_SOURCE,
            /* ensure_row_contiguous */ true,
            /* atomic_outputs */ false,
        )?;
        let config = MetalKernelConfig::new();
        config.add_template_arg_dtype("InT", conv_input.dtype())?;
        config.add_template_arg_int("NK", nk)?;
        config.add_template_arg_int("NV", nv)?;
        config.add_template_arg_int("DN", dn)?;
        config.add_template_arg_int("DV", dv)?;
        config.add_template_arg_int("K", k_taps)?;
        let rows = nk + nk + nv;
        let lane = dn.max(dv);
        config.set_grid(lane, rows, 1)?;
        config.set_thread_group(lane, 1, 1)?;
        config.add_output_arg(&[nk, dn], conv_input.dtype())?;
        config.add_output_arg(&[nk, dn], conv_input.dtype())?;
        config.add_output_arg(&[nv, dv], conv_input.dtype())?;
        let eps_arr = Array::from_slice(&[eps], &[1]);
        let outs = kernel.apply(&[conv_input, conv_w, wq, wk, &eps_arr], &config, 3)?;
        let mut it = outs.into_iter();
        let q = it
            .next()
            .ok_or_else(|| anyhow::anyhow!("inproj_tail_fused: no out_q"))?;
        let k = it
            .next()
            .ok_or_else(|| anyhow::anyhow!("inproj_tail_fused: no out_k"))?;
        let v = it
            .next()
            .ok_or_else(|| anyhow::anyhow!("inproj_tail_fused: no out_v"))?;
        Ok((q, k, v))
    }
}

#[cfg(feature = "mlx-native")]
#[allow(unused_imports)] // Consumed by Phase 3b.5 SSM block assembly.
pub use imp::{
    compute_g, gated_delta_step_kernel, gated_delta_step_kernel_capture, inproj_tail_fused,
    rms_norm_gated, rms_norm_gated_fused, rms_norm_gated_legacy,
};

// compute_g bit-identical vs MLX reference.
//
// Fixtures produced by `scripts/generate_compute_g_fixture.py` (magic `TQGG`).
#[cfg(all(test, feature = "mlx-native"))]
mod parity_tests {
    use super::imp::{compute_g, rms_norm_gated, rms_norm_gated_fused, rms_norm_gated_legacy};
    use mlx_rs::Array;
    use std::path::{Path, PathBuf};

    const MAGIC: &[u8; 4] = b"TQGG";
    const HEADER_LEN: usize = 4 /* magic */ + 3 * 4 /* 3 u32 */;
    const GATED_MAGIC: &[u8; 4] = b"TQRG";
    const GATED_HEADER_LEN: usize = 4 /* magic */ + 4 * 4 /* 4 u32 */;

    fn fixture_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("lumen-mlx manifest dir has a parent (crates/)")
            .join("lumen-metal")
            .join("tests")
            .join("fixtures")
    }

    fn read_u32_le(bytes: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
    }
    fn slice_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    struct ComputeGFixture {
        batch: usize,
        seq: usize,
        num_v_heads: usize,
        a_log: Vec<f32>,
        a: Vec<f32>,
        dt_bias: Vec<f32>,
        expected_y: Vec<f32>,
    }

    fn load_fixture(name: &str) -> Option<ComputeGFixture> {
        let path = fixture_dir().join(format!("embed_compute_g_{name}.bin"));
        let bytes = std::fs::read(&path).ok()?;
        if bytes.len() < HEADER_LEN || &bytes[..4] != MAGIC {
            panic!("compute_g fixture {path:?} has bad header");
        }
        let batch = read_u32_le(&bytes, 4) as usize;
        let seq = read_u32_le(&bytes, 8) as usize;
        let num_v_heads = read_u32_le(&bytes, 12) as usize;

        let mut cur = HEADER_LEN;
        let a_log = slice_f32(&bytes[cur..cur + num_v_heads * 4]);
        cur += num_v_heads * 4;
        let a_count = batch * seq * num_v_heads;
        let a = slice_f32(&bytes[cur..cur + a_count * 4]);
        cur += a_count * 4;
        let dt_bias = slice_f32(&bytes[cur..cur + num_v_heads * 4]);
        cur += num_v_heads * 4;
        let expected_y = slice_f32(&bytes[cur..cur + a_count * 4]);

        Some(ComputeGFixture {
            batch,
            seq,
            num_v_heads,
            a_log,
            a,
            dt_bias,
            expected_y,
        })
    }

    fn run_fixture(name: &str) {
        let Some(fx) = load_fixture(name) else {
            eprintln!(
                "SKIP compute_g/{name}: fixture missing under {}. Run `python scripts/generate_compute_g_fixture.py` first.",
                fixture_dir().display()
            );
            return;
        };

        let a_log = Array::from_slice(&fx.a_log, &[fx.num_v_heads as i32]);
        let a = Array::from_slice(
            &fx.a,
            &[fx.batch as i32, fx.seq as i32, fx.num_v_heads as i32],
        );
        let dt_bias = Array::from_slice(&fx.dt_bias, &[fx.num_v_heads as i32]);

        let out = compute_g(&a_log, &a, &dt_bias).expect("compute_g must succeed");
        assert_eq!(
            out.shape(),
            &[fx.batch as i32, fx.seq as i32, fx.num_v_heads as i32],
            "{name}: compute_g output shape mismatch"
        );
        assert_eq!(
            out.dtype(),
            mlx_rs::Dtype::Float32,
            "{name}: compute_g output dtype unexpectedly changed"
        );
        out.eval().expect("mlx eval must succeed");

        let observed: &[f32] = out.as_slice();
        assert_eq!(
            observed.len(),
            fx.expected_y.len(),
            "{name}: compute_g output flat-size mismatch"
        );

        let mut mismatches = 0usize;
        for (i, (&got, &expected)) in observed.iter().zip(fx.expected_y.iter()).enumerate() {
            if got.to_bits() != expected.to_bits() {
                if mismatches < 5 {
                    eprintln!(
                        "compute_g/{name}[{i}]: 0x{:08x} ({got}) vs ref 0x{:08x} ({expected})",
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
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn tiny_compute_g_matches_mlx() {
        run_fixture("tiny");
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn multi_head_compute_g_matches_mlx() {
        run_fixture("multi_head");
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn prod_heads_compute_g_matches_mlx() {
        run_fixture("prod_heads");
    }

    // ─── Phase 3b.5.c: rms_norm_gated parity ────────────────────────────────

    struct GatedFixture {
        batch: usize,
        seq: usize,
        hidden: usize,
        eps: f32,
        h: Vec<f32>,
        z: Vec<f32>,
        w: Vec<f32>,
        expected_y: Vec<f32>,
    }

    fn load_gated_fixture(name: &str) -> Option<GatedFixture> {
        let path = fixture_dir().join(format!("embed_rms_norm_gated_{name}.bin"));
        let bytes = std::fs::read(&path).ok()?;
        if bytes.len() < GATED_HEADER_LEN || &bytes[..4] != GATED_MAGIC {
            panic!("rms_norm_gated fixture {path:?} has bad header");
        }
        let batch = read_u32_le(&bytes, 4) as usize;
        let seq = read_u32_le(&bytes, 8) as usize;
        let hidden = read_u32_le(&bytes, 12) as usize;
        let eps = f32::from_bits(read_u32_le(&bytes, 16));

        let count = batch * seq * hidden;
        let mut cur = GATED_HEADER_LEN;
        let h = slice_f32(&bytes[cur..cur + count * 4]);
        cur += count * 4;
        let z = slice_f32(&bytes[cur..cur + count * 4]);
        cur += count * 4;
        let w = slice_f32(&bytes[cur..cur + hidden * 4]);
        cur += hidden * 4;
        let expected_y = slice_f32(&bytes[cur..cur + count * 4]);

        Some(GatedFixture {
            batch,
            seq,
            hidden,
            eps,
            h,
            z,
            w,
            expected_y,
        })
    }

    fn run_gated_fixture(name: &str) {
        let Some(fx) = load_gated_fixture(name) else {
            eprintln!(
                "SKIP rms_norm_gated/{name}: fixture missing under {}. Run `python scripts/generate_rms_norm_gated_fixture.py` first.",
                fixture_dir().display()
            );
            return;
        };

        let h = Array::from_slice(&fx.h, &[fx.batch as i32, fx.seq as i32, fx.hidden as i32]);
        let z = Array::from_slice(&fx.z, &[fx.batch as i32, fx.seq as i32, fx.hidden as i32]);
        let w = Array::from_slice(&fx.w, &[fx.hidden as i32]);

        let out = rms_norm_gated(&h, &z, &w, fx.eps).expect("rms_norm_gated must succeed");
        assert_eq!(
            out.shape(),
            &[fx.batch as i32, fx.seq as i32, fx.hidden as i32],
            "{name}: rms_norm_gated output shape mismatch"
        );
        // Output should match `hidden_states.dtype()` — fixture is f32 so
        // we expect f32 here. If we ever pass bf16/f16 inputs, the test
        // would need a downcast-then-upcast comparison; not covered yet.
        assert_eq!(
            out.dtype(),
            mlx_rs::Dtype::Float32,
            "{name}: rms_norm_gated output dtype unexpectedly changed"
        );
        out.eval().expect("mlx eval must succeed");

        let observed: &[f32] = out.as_slice();
        assert_eq!(
            observed.len(),
            fx.expected_y.len(),
            "{name}: rms_norm_gated output flat-size mismatch"
        );

        let mut mismatches = 0usize;
        for (i, (&got, &expected)) in observed.iter().zip(fx.expected_y.iter()).enumerate() {
            if got.to_bits() != expected.to_bits() {
                if mismatches < 5 {
                    eprintln!(
                        "rms_norm_gated/{name}[{i}]: 0x{:08x} ({got}) vs ref 0x{:08x} ({expected})",
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
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn tiny_rms_norm_gated_matches_mlx() {
        run_gated_fixture("tiny");
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn multi_rms_norm_gated_matches_mlx() {
        run_gated_fixture("multi");
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn head_v_dim_rms_norm_gated_matches_mlx() {
        run_gated_fixture("head_v_dim");
    }

    // ─── Phase G5-A parity: fused kernel vs legacy composed path ────────────
    //
    // The fused kernel's reduction tree differs from `mlx.fast.rms_norm`, so
    // bit-identical parity is not expected. Tolerance bounds derived from
    // fp32 round-off on a `D=128` simdgroup-wide sum (≤ ~1e-6 relative on
    // inv_rms) compounded with one silu * x * weight multiply.
    fn run_fused_vs_legacy_fixture(name: &str) {
        let Some(fx) = load_gated_fixture(name) else {
            eprintln!(
                "SKIP rms_norm_gated_fused/{name}: fixture missing under {}. Run `python scripts/generate_rms_norm_gated_fixture.py` first.",
                fixture_dir().display()
            );
            return;
        };

        if fx.hidden % 32 != 0 {
            eprintln!(
                "SKIP rms_norm_gated_fused/{name}: fixture hidden={} not multiple of 32 (fused kernel uses simdgroup-wide reduction)",
                fx.hidden
            );
            return;
        }

        let h = Array::from_slice(&fx.h, &[fx.batch as i32, fx.seq as i32, fx.hidden as i32]);
        let z = Array::from_slice(&fx.z, &[fx.batch as i32, fx.seq as i32, fx.hidden as i32]);
        let w = Array::from_slice(&fx.w, &[fx.hidden as i32]);

        let legacy =
            rms_norm_gated_legacy(&h, &z, &w, fx.eps).expect("rms_norm_gated_legacy must succeed");
        let fused =
            rms_norm_gated_fused(&h, &z, &w, fx.eps).expect("rms_norm_gated_fused must succeed");

        assert_eq!(
            legacy.shape(),
            fused.shape(),
            "{name}: fused vs legacy shape mismatch"
        );
        assert_eq!(
            legacy.dtype(),
            fused.dtype(),
            "{name}: fused vs legacy dtype mismatch"
        );
        legacy.eval().expect("legacy eval must succeed");
        fused.eval().expect("fused eval must succeed");

        let legacy_slice: &[f32] = legacy.as_slice();
        let fused_slice: &[f32] = fused.as_slice();
        assert_eq!(
            legacy_slice.len(),
            fused_slice.len(),
            "{name}: fused vs legacy flat-size mismatch"
        );

        // Tolerance: atol 1e-5 + rtol 1e-4 on element-wise. Empirically the
        // fp32 inv_rms agreement between MLX's built-in and our simdgroup
        // sum is ~1 ULP; compounded with silu·x·w·inv_rms the worst-case
        // element diverges by O(eps_machine · |z| · |x|). For fixture
        // values bounded by O(1) this bound is well under 1e-4 rel.
        const ATOL: f32 = 1e-5;
        const RTOL: f32 = 1e-4;
        let mut mismatches = 0usize;
        let mut max_abs_err: f32 = 0.0;
        let mut max_rel_err: f32 = 0.0;
        for (i, (&got, &expected)) in fused_slice.iter().zip(legacy_slice.iter()).enumerate() {
            let abs_err = (got - expected).abs();
            let denom = expected.abs().max(1e-12);
            let rel_err = abs_err / denom;
            max_abs_err = max_abs_err.max(abs_err);
            max_rel_err = max_rel_err.max(rel_err);
            let tol = ATOL + RTOL * expected.abs();
            if abs_err > tol {
                if mismatches < 5 {
                    eprintln!(
                        "rms_norm_gated_fused/{name}[{i}]: got={got} (0x{:08x}) vs legacy={expected} (0x{:08x}) abs_err={abs_err:.3e} rel_err={rel_err:.3e} tol={tol:.3e}",
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
            "{name}: {mismatches}/{} elements outside tolerance (max abs={:.3e} max rel={:.3e})",
            fused_slice.len(),
            max_abs_err,
            max_rel_err
        );
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn tiny_rms_norm_gated_fused_matches_legacy() {
        run_fused_vs_legacy_fixture("tiny");
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn head_v_dim_rms_norm_gated_fused_matches_legacy() {
        run_fused_vs_legacy_fixture("head_v_dim");
    }

    // ─── Phase 3b.5.d: gated_delta_step kernel parity ───────────────────────
    //
    // Fixtures produced by `scripts/generate_gated_delta_fixture.py` (magic
    // `TQGD`). Both Python reference and Rust call the *same* runtime-compiled
    // Metal kernel from the same source string, so the comparison must be
    // bit-identical.

    use super::imp::gated_delta_step_kernel;

    const SSM_KERNEL_MAGIC: &[u8; 4] = b"TQGD";
    const SSM_KERNEL_HEADER_LEN: usize = 4 /* magic */ + 6 * 4 /* 6 u32 */;

    struct GatedDeltaFixture {
        batch: usize,
        seq: usize,
        n_k_heads: usize,
        n_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        q: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
        g: Vec<f32>,
        beta: Vec<f32>,
        state_in: Vec<f32>,
        expected_y: Vec<f32>,
        expected_state: Vec<f32>,
    }

    fn load_gated_delta_fixture(name: &str) -> Option<GatedDeltaFixture> {
        let path = fixture_dir().join(format!("embed_gated_delta_{name}.bin"));
        let bytes = std::fs::read(&path).ok()?;
        if bytes.len() < SSM_KERNEL_HEADER_LEN || &bytes[..4] != SSM_KERNEL_MAGIC {
            panic!("gated_delta fixture {path:?} has bad header");
        }
        let batch = read_u32_le(&bytes, 4) as usize;
        let seq = read_u32_le(&bytes, 8) as usize;
        let n_k_heads = read_u32_le(&bytes, 12) as usize;
        let n_v_heads = read_u32_le(&bytes, 16) as usize;
        let head_k_dim = read_u32_le(&bytes, 20) as usize;
        let head_v_dim = read_u32_le(&bytes, 24) as usize;

        let q_count = batch * seq * n_k_heads * head_k_dim;
        let v_count = batch * seq * n_v_heads * head_v_dim;
        let gbeta_count = batch * seq * n_v_heads;
        let state_count = batch * n_v_heads * head_v_dim * head_k_dim;

        let mut cur = SSM_KERNEL_HEADER_LEN;
        let q = slice_f32(&bytes[cur..cur + q_count * 4]);
        cur += q_count * 4;
        let k = slice_f32(&bytes[cur..cur + q_count * 4]);
        cur += q_count * 4;
        let v = slice_f32(&bytes[cur..cur + v_count * 4]);
        cur += v_count * 4;
        let g = slice_f32(&bytes[cur..cur + gbeta_count * 4]);
        cur += gbeta_count * 4;
        let beta = slice_f32(&bytes[cur..cur + gbeta_count * 4]);
        cur += gbeta_count * 4;
        let state_in = slice_f32(&bytes[cur..cur + state_count * 4]);
        cur += state_count * 4;
        let expected_y = slice_f32(&bytes[cur..cur + v_count * 4]);
        cur += v_count * 4;
        let expected_state = slice_f32(&bytes[cur..cur + state_count * 4]);

        Some(GatedDeltaFixture {
            batch,
            seq,
            n_k_heads,
            n_v_heads,
            head_k_dim,
            head_v_dim,
            q,
            k,
            v,
            g,
            beta,
            state_in,
            expected_y,
            expected_state,
        })
    }

    fn run_gated_delta_fixture(name: &str) {
        let Some(fx) = load_gated_delta_fixture(name) else {
            eprintln!(
                "SKIP gated_delta/{name}: fixture missing under {}. Run `python scripts/generate_gated_delta_fixture.py` first.",
                fixture_dir().display()
            );
            return;
        };

        let q = Array::from_slice(
            &fx.q,
            &[
                fx.batch as i32,
                fx.seq as i32,
                fx.n_k_heads as i32,
                fx.head_k_dim as i32,
            ],
        );
        let k = Array::from_slice(
            &fx.k,
            &[
                fx.batch as i32,
                fx.seq as i32,
                fx.n_k_heads as i32,
                fx.head_k_dim as i32,
            ],
        );
        let v = Array::from_slice(
            &fx.v,
            &[
                fx.batch as i32,
                fx.seq as i32,
                fx.n_v_heads as i32,
                fx.head_v_dim as i32,
            ],
        );
        let g = Array::from_slice(
            &fx.g,
            &[fx.batch as i32, fx.seq as i32, fx.n_v_heads as i32],
        );
        let beta = Array::from_slice(
            &fx.beta,
            &[fx.batch as i32, fx.seq as i32, fx.n_v_heads as i32],
        );
        let state_in = Array::from_slice(
            &fx.state_in,
            &[
                fx.batch as i32,
                fx.n_v_heads as i32,
                fx.head_v_dim as i32,
                fx.head_k_dim as i32,
            ],
        );

        let (y, state_out) = gated_delta_step_kernel(&q, &k, &v, &g, &beta, &state_in)
            .expect("gated_delta_step_kernel must succeed");

        assert_eq!(
            y.shape(),
            &[
                fx.batch as i32,
                fx.seq as i32,
                fx.n_v_heads as i32,
                fx.head_v_dim as i32,
            ],
            "{name}: y shape mismatch"
        );
        assert_eq!(
            state_out.shape(),
            &[
                fx.batch as i32,
                fx.n_v_heads as i32,
                fx.head_v_dim as i32,
                fx.head_k_dim as i32,
            ],
            "{name}: state_out shape mismatch"
        );
        y.eval().expect("eval y must succeed");
        state_out.eval().expect("eval state_out must succeed");

        let observed_y: &[f32] = y.as_slice();
        let observed_state: &[f32] = state_out.as_slice();

        assert_eq!(observed_y.len(), fx.expected_y.len(), "{name}: y flat-size");
        assert_eq!(
            observed_state.len(),
            fx.expected_state.len(),
            "{name}: state flat-size"
        );

        let mut y_mismatches = 0usize;
        for (i, (&got, &expected)) in observed_y.iter().zip(fx.expected_y.iter()).enumerate() {
            if got.to_bits() != expected.to_bits() {
                if y_mismatches < 5 {
                    eprintln!(
                        "gated_delta/{name}.y[{i}]: 0x{:08x} ({got}) vs ref 0x{:08x} ({expected})",
                        got.to_bits(),
                        expected.to_bits()
                    );
                }
                y_mismatches += 1;
            }
        }
        assert_eq!(y_mismatches, 0, "{name}: {y_mismatches} y mismatches");

        let mut state_mismatches = 0usize;
        for (i, (&got, &expected)) in observed_state
            .iter()
            .zip(fx.expected_state.iter())
            .enumerate()
        {
            if got.to_bits() != expected.to_bits() {
                if state_mismatches < 5 {
                    eprintln!(
                        "gated_delta/{name}.state[{i}]: 0x{:08x} ({got}) vs ref 0x{:08x} ({expected})",
                        got.to_bits(),
                        expected.to_bits()
                    );
                }
                state_mismatches += 1;
            }
        }
        assert_eq!(
            state_mismatches, 0,
            "{name}: {state_mismatches} state mismatches"
        );
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn tiny_gated_delta_kernel_matches_mlx() {
        run_gated_delta_fixture("tiny");
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn multi_head_gated_delta_kernel_matches_mlx() {
        run_gated_delta_fixture("multi_head");
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn decode_q1_gated_delta_kernel_matches_mlx() {
        run_gated_delta_fixture("decode_q1");
    }
}

#[cfg(all(test, feature = "mlx-native"))]
mod capture_kernel_tests {
    use super::imp::{gated_delta_step_kernel, gated_delta_step_kernel_capture};
    use mlx_rs::Array;

    fn mk(n: usize, seed: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32 + 1.0) * seed).sin() * 0.5)
            .collect()
    }
    fn max_abs_diff(a: &Array, b: &Array) -> f32 {
        let d = mlx_rs::ops::abs(&mlx_rs::ops::subtract(a, b).unwrap()).unwrap();
        let m = d.max(None).unwrap();
        m.eval().unwrap();
        m.as_slice::<f32>()[0]
    }

    #[test]
    fn capture_matches_base_and_per_position() {
        let (b, t, hk, hv, dk, dv) = (1i32, 4i32, 1i32, 2i32, 32i32, 4i32);
        let q = Array::from_slice(&mk((b * t * hk * dk) as usize, 0.11), &[b, t, hk, dk]);
        let k = Array::from_slice(&mk((b * t * hk * dk) as usize, 0.13), &[b, t, hk, dk]);
        let v = Array::from_slice(&mk((b * t * hv * dv) as usize, 0.17), &[b, t, hv, dv]);
        let g = Array::from_slice(&mk((b * t * hv) as usize, 0.23), &[b, t, hv]);
        let beta = Array::from_slice(&mk((b * t * hv) as usize, 0.19), &[b, t, hv]);
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
