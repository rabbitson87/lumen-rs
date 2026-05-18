//! Native MLX Qwen3.5-MoE sparse MoE block.
//!
//! Mirrors `qwen3_next.Qwen3NextSparseMoeBlock.__call__` 1-for-1 against the
//! production MXFP4 checkpoint:
//!
//! ```text
//! gates  = softmax(gate(x), axis=-1, precise=true)        # 8-bit affine gate
//! inds   = argpartition(gates, kth=-K, axis=-1)[..., -K:] # [B, S, K]
//! scores = take_along_axis(gates, inds, axis=-1)
//! scores = scores / scores.sum(axis=-1, keepdims=true)
//! y      = switch_glu(x, inds)                            # [B, S, K, hidden]
//! y      = (y * scores[..., None]).sum(axis=-2)           # [B, S, hidden]
//! shared = sigmoid(shared_expert_gate(x)) * shared_expert(x)
//! y      = y + shared                                     # [B, S, hidden]
//! ```
//!
//! Per-tensor MXFP4 overrides:
//!   * `mlp.gate`              — 8-bit affine, group_size=64, **with biases**.
//!   * `mlp.shared_expert_gate` — 8-bit affine, group_size=64, **with biases**.
//!
//! Everything else (`switch_mlp.{gate_proj,up_proj,down_proj}` and
//! `shared_expert.{gate_proj,up_proj,down_proj}`) defaults to the top-level
//! mxfp4 4-bit / 32-group quantization (no biases).
//!
//! `switch_glu` mirrors `switch_layers.SwitchGLU.__call__`, including the
//! `do_sort = indices.size >= 64` branch: when there are enough tokens, the
//! routing indices are sorted before the gather_qmm so that consecutive
//! tokens land on consecutive expert weights (lifts arithmetic intensity);
//! the result is unsorted with the inverse permutation. We mirror this
//! exactly so a future Phase 3d.5 PyO3 parity gate sees bit-identical
//! `gather_qmm` dispatches.
//!
//! See `.ai/memory/active/mlx-rs-native-port/CONTEXT.md` Session 28.

#[cfg(feature = "mlx-native")]
#[allow(dead_code)] // `dense_mlp_forward` is a re-exported helper reserved for future shared-MLP-only callsites
mod imp {
    use anyhow::{Context, Result, anyhow};
    use mlx_rs::Array;
    use mlx_rs::error::Exception;
    use mlx_rs::ops::indexing::{Ellipsis, IndexOp};
    use std::ffi::CStr;
    use std::sync::{Mutex, OnceLock};
    use std::time::Instant;

    use crate::native_compile_cache::{
        CompiledMulti, CompiledMultiRefs, invoke_compiled_multi, invoke_compiled_multi_refs,
    };
    use crate::native_runtime::{
        bump_moe_routing_ms, bump_moe_shared_expert_ms, bump_moe_switch_glu_ms, fine_timing_active,
    };
    use crate::native_quant::{
        MODE_AFFINE, gather_qmm_with_mode, quantized_matmul_with_mode,
    };

    // pipeline persistent compile cache.
    //
    // The routing chain `softmax → argpartition → take_along → renorm` is
    // dispatched 30+ times per decode step (one per MoE layer). With
    // `LUMEN_NATIVE_COMPILE_ROUTING=1`, fold the chain into a single
    // compiled graph cached for process lifetime via `compile_boxed_slice`.
    //
    // Hardcoded constants — Qwen3.5-MoE checkpoint family:
    //   * top_k = 8
    //   * norm_topk_prob = true
    //   * num_experts read from gate_logits shape at trace time (constant per
    //     loaded model; cache hits as long as num_experts stays fixed)
    //
    // The matmul (`gate(x)` quantized linear) stays outside the closure —
    // quantized op compile compatibility is a separate question.
    fn routing_compiled_inner(
        args: &[Array],
    ) -> std::result::Result<Vec<Array>, Exception> {
        let gate_logits = &args[0];
        let last_axis_idx = gate_logits.ndim() - 1;
        let last_axis = last_axis_idx as i32;
        let num_experts = gate_logits.shape()[last_axis_idx];
        let top_k: i32 = 8;
        let start = num_experts - top_k;

        let gates = mlx_rs::ops::softmax_axis(gate_logits, last_axis, Some(true))?;
        let part = mlx_rs::ops::argpartition_axis(&gates, -top_k, last_axis)?;
        let inds = part.index((Ellipsis, start..));
        let scores = gates.take_along_axis(&inds, last_axis)?;
        let denom = scores.sum_axis(last_axis, /* keep_dims */ true)?;
        let scores = scores.divide(&denom)?;
        Ok(vec![inds, scores])
    }

    static ROUTING_SLOT: OnceLock<Mutex<CompiledMulti>> = OnceLock::new();

    fn routing_compile_enabled() -> bool {
        std::env::var("LUMEN_NATIVE_COMPILE_ROUTING")
            .map(|v| v == "1")
            .unwrap_or(false)
    }

    // SwiGLU fusion compile cache. Replaces the explicit
    //   silu_gate = silu(gate)        // 1 compile dispatch (compiled_silu)
    //   activated = silu_gate * up    // 1 explicit multiply (separate kernel)
    // pattern with a single compile-wrapped graph:
    //   silu(gate) * up               // 1 compile dispatch, sigmoid+multiply+multiply fused
    // Saves 1 try_from_op + 1 Metal kernel launch per MoE switch_glu / shared_mlp
    // invocation (~32-64 per step depending on layer count and expert routing).
    //
    // Gated by LUMEN_NATIVE_FUSE_SWIGLU=1 (default OFF for A/B).
    // Mirrors Python's `_precise_swiglu` pattern from mlx_lm.
    fn swiglu_compiled_inner(
        args: &[Array],
    ) -> std::result::Result<Vec<Array>, Exception> {
        let gate = &args[0];
        let up = &args[1];
        let sig = mlx_rs::ops::sigmoid(gate)?;
        let silu_gate = gate.multiply(&sig)?;
        let out = silu_gate.multiply(up)?;
        Ok(vec![out])
    }

    static SWIGLU_SLOT: OnceLock<Mutex<CompiledMultiRefs>> = OnceLock::new();

    fn swiglu_fuse_enabled() -> bool {
        // Default ON 2026-05-11 — landed as net WIN -0.671 ms (Welch t=-29σ) at
        // n=10 STEPS=300 PROMPT_LEN=3 on Qwen3.6-35B-A3B-mxfp4. Closes ~62% of
        // the 1.07 ms Native vs PyO3 decode gap. Opt out with
        // LUMEN_NATIVE_FUSE_SWIGLU=0 for parity A/B or rollback debugging.
        std::env::var("LUMEN_NATIVE_FUSE_SWIGLU")
            .map(|v| v != "0")
            .unwrap_or(true)
    }

    fn swiglu_fused(gate: &Array, up: &Array) -> Result<Array> {
        let args = [gate, up];
        let mut out = invoke_compiled_multi_refs(
            &SWIGLU_SLOT,
            swiglu_compiled_inner,
            /* shapeless */ true,
            &args,
        )
        .context("swiglu_fused: mlx compile dispatch failed")?;
        out.pop()
            .context("swiglu_fused: empty output vec")
    }

    // ──────────── GeGLU fusion (Gemma 4) ────────────────────────────
    //
    // Gemma 4 26B-A4B uses `hidden_activation = "gelu_pytorch_tanh"`, which is
    // mathematically `gelu_approximate`:
    //
    //   gelu_approx(x) = 0.5 · x · (1 + tanh(sqrt(2/π) · (x + 0.044715 · x³)))
    //
    // The unfused Gemma 4 hot path is:
    //   gate_act = mlx_rs::nn::gelu_approximate(gate)?     // compile dispatch #1
    //   out      = gate_act.multiply(up)?                  // separate kernel #2
    //
    // → 2 compile dispatches + 2 Metal kernel launches per dense-MLP or per
    // expert tile call. With 30 layers × (1 dense MLP call + N gather_qmm
    // expert tiles) per decode step, the per-step overhead is significant.
    //
    // This fused variant traces the full `gelu_approx(gate) * up` graph into a
    // single compile slot, mirroring the Qwen 3.5/3.6 SwiGLU landing pattern
    // (`swiglu_fused` above). Constants (0.5, 1.0, 0.044715, sqrt(2/π)) are
    // baked into the traced graph on first call and shared for the process
    // lifetime via the `GELU_MUL_SLOT` OnceLock.
    //
    // Empirical expectation from Qwen 3.6 35B-A3B SwiGLU lever (mxfp4 path):
    //   -0.671 ms/step (Welch t=-29σ at n=10 STEPS=300). Closes ~62% of the
    //   1.07 ms Native vs PyO3 decode gap. Gemma 4 GeGLU should land in the
    //   same ballpark because the fuse pattern (binary op + activation) is
    //   structurally identical.
    //
    // Gate: `LUMEN_NATIVE_FUSE_SWIGLU` env var (default ON, shared with
    // SwiGLU so a single off-switch covers all activation fuses for A/B
    // rollback). Set `=0` to disable.
    fn gelu_mul_compiled_inner(
        args: &[Array],
    ) -> std::result::Result<Vec<Array>, Exception> {
        use mlx_rs::ops::tanh;
        let gate = &args[0];
        let up = &args[1];
        // Constants captured into the traced graph on first call.
        // match `gate.dtype()` so the bf16 model
        // doesn't trigger implicit AsType primitives on every multiply.
        // Histogram showed 955 AsType/step on Gemma 4 bf16 path — these
        // f32 scalars × bf16 tensors were the dominant source.
        let dt = gate.dtype();
        let half = Array::from_f32(0.5).as_dtype(dt)?;
        let one = Array::from_f32(1.0).as_dtype(dt)?;
        let c3 = Array::from_f32(0.044715).as_dtype(dt)?;
        let coeff = Array::from_f32(0.7978845608028654_f32).as_dtype(dt)?; // sqrt(2/PI)

        let x_squared = gate.multiply(gate)?;
        let x_cubed = x_squared.multiply(gate)?;
        let inner_add = gate.add(&x_cubed.multiply(&c3)?)?;
        let scaled = coeff.multiply(&inner_add)?;
        let t = tanh(&scaled)?;
        let one_plus_tanh = one.add(&t)?;
        let gelu = half.multiply(gate)?.multiply(&one_plus_tanh)?;
        let out = gelu.multiply(up)?;
        Ok(vec![out])
    }

    static GELU_MUL_SLOT: OnceLock<Mutex<CompiledMultiRefs>> = OnceLock::new();

    /// Fused `gelu_approximate(gate) * up` (a.k.a. GeGLU). Used by Gemma 4's
    /// dense MLP and expert tile activation. Public so `gemma4_moe.rs` can
    /// import it directly without going through the `imp` module barrier.
    pub fn gelu_mul_fused(gate: &Array, up: &Array) -> Result<Array> {
        let args = [gate, up];
        let mut out = invoke_compiled_multi_refs(
            &GELU_MUL_SLOT,
            gelu_mul_compiled_inner,
            /* shapeless */ true,
            &args,
        )
        .context("gelu_mul_fused: mlx compile dispatch failed")?;
        out.pop().context("gelu_mul_fused: empty output vec")
    }

    /// Env-gated check (shared with SwiGLU — single off-switch for A/B).
    pub fn gelu_mul_fuse_enabled() -> bool {
        std::env::var("LUMEN_NATIVE_FUSE_SWIGLU")
            .map(|v| v != "0")
            .unwrap_or(true)
    }

    // Sigmoid-gated multiply fusion is shared with full-attn via
    // `crate::native_kernels::sigmoid_mul_fused` — one compile slot per process.
    use crate::native_kernels::{sigmoid_mul_fuse_enabled, sigmoid_mul_fused};

    /// One-shot `mxfp4 / affine` linear lookup. The caller hands over the
    /// resolved `(weight, scales, biases?)` plus quant params; we dispatch
    /// `quantized_matmul_with_mode(transpose=true)`.
    fn linear_with_mode(
        x: &Array,
        weight: &Array,
        scales: &Array,
        biases: Option<&Array>,
        group_size: i32,
        bits: i32,
        mode: &CStr,
    ) -> Result<Array> {
        quantized_matmul_with_mode(x, weight, scales, biases, true, group_size, bits, mode)
    }

    /// Resolved tensors for one `SwitchLinear` (per-expert MXFP4 weights).
    pub struct SwitchExpertWeights<'a> {
        pub weight: &'a Array,
        pub scales: &'a Array,
        pub biases: Option<&'a Array>,
    }

    /// Resolved tensors for the three SwitchLinear layers + meta.
    pub struct SwitchGluWeights<'a> {
        pub gate_proj: SwitchExpertWeights<'a>,
        pub up_proj: SwitchExpertWeights<'a>,
        pub down_proj: SwitchExpertWeights<'a>,
        pub group_size: i32,
        pub bits: i32,
        /// Mode to use for the three gather_qmm calls — almost always
        /// MXFP4 in the production checkpoint, but parameterised so a
        /// future affine MoE checkpoint can reuse the helper.
        pub mode: &'a CStr,
    }

    /// `SwitchGLU.__call__(x, indices)` — quantized expert dispatch with the
    /// optional sort-by-index optimization.
    ///
    /// `x`: `[B, S, hidden]` (f32). `indices`: `[B, S, K]` (i32 / u32 — both
    /// accepted by mlx_gather_qmm).
    ///
    /// Returns `[B, S, K, hidden]` (per-expert outputs, ready for the
    /// `(y * scores[..., None]).sum(axis=-2)` reduction in the MoE block).
    pub fn switch_glu(
        x: &Array,
        indices: &Array,
        weights: &SwitchGluWeights<'_>,
    ) -> Result<Array> {
        if x.ndim() != 3 {
            return Err(anyhow!(
                "switch_glu: expected x of rank 3 [B, S, hidden], got ndim={}",
                x.ndim()
            ));
        }
        if indices.ndim() != 3 {
            return Err(anyhow!(
                "switch_glu: expected indices of rank 3 [B, S, K], got ndim={}",
                indices.ndim()
            ));
        }
        let ind_shape = indices.shape().to_vec();
        let b = ind_shape[0];
        let s = ind_shape[1];
        let k = ind_shape[2];

        // (a) x = expand_dims(x, (-2, -3)) → [B, S, 1, 1, hidden].
        //     Use sequential single-axis expand_dims to mirror Python's
        //     `expand_dims(x, (-2, -3))` insertion order: the tuple `(-2, -3)`
        //     is processed against the already-shifted shape, yielding
        //     `[B, S, 1, 1, hidden]`.
        let x_5d = mlx_rs::ops::expand_dims_axes(x, &[-2, -3])
            .context("switch_glu: expand_dims(x, (-2,-3)) failed")?;

        // Decide do_sort exactly like Python: total element count of
        // `indices` ≥ 64. For decode (B=1, S=1, K=8) that's 8 < 64 so
        // single-token decode skips the sort; prefill always takes it.
        let do_sort = (b as usize) * (s as usize) * (k as usize) >= 64;

        let (x_for_qmm, idx_for_qmm, inv_order_opt) = if do_sort {
            // _gather_sort: flatten indices, argsort, reorder x/idx.
            let inds_flat = mlx_rs::ops::flatten(indices, /* start */ 0, /* end */ -1)
                .context("switch_glu: flatten(indices) failed")?;
            let order = mlx_rs::ops::argsort(&inds_flat)
                .context("switch_glu: argsort(indices_flat) failed")?;
            let inv_order = mlx_rs::ops::argsort(&order)
                .context("switch_glu: argsort(order) for inv_order failed")?;

            // x.flatten(0, -3): for 5D x = [B, S, 1, 1, H], -3 == axis 2,
            // so flatten axes 0..=2 → [B*S, 1, H].
            let x_flat = mlx_rs::ops::flatten(&x_5d, 0, -3)
                .context("switch_glu: flatten(x, 0, -3) failed")?;

            // x_flat[order // K]: divide order indices by K to map B*S*K
            // index space → B*S row index space.
            // Python `order // K` is floor division. `mlx_rs::ops::divide`
            // returns a true (float) division, which then trips
            // `take_axis`'s integer-indices contract; use `floor_divide` so
            // the result keeps the integer dtype of `order`.
            let k_arr = Array::from_int(k);
            let row_idx = mlx_rs::ops::floor_divide(&order, &k_arr)
                .context("switch_glu: order // K floor_divide failed")?;
            let x_sorted = mlx_rs::ops::indexing::take_axis(&x_flat, &row_idx, 0)
                .context("switch_glu: take_axis(x_flat, order // K) failed")?;
            // indices_flat[order]: idx-sorted permutation.
            let idx_sorted = mlx_rs::ops::indexing::take_axis(&inds_flat, &order, 0)
                .context("switch_glu: take_axis(inds_flat, order) failed")?;
            (x_sorted, idx_sorted, Some(inv_order))
        } else {
            (x_5d, indices.clone(), None)
        };

        // Three gather_qmm calls — MXFP4 with the sorted-flag wired through.
        let x_up = gather_qmm_with_mode(
            &x_for_qmm,
            weights.up_proj.weight,
            weights.up_proj.scales,
            weights.up_proj.biases,
            None,
            Some(&idx_for_qmm),
            true,
            weights.group_size,
            weights.bits,
            weights.mode,
            do_sort,
        )
        .context("switch_glu: gather_qmm(up_proj) failed")?;

        let x_gate = gather_qmm_with_mode(
            &x_for_qmm,
            weights.gate_proj.weight,
            weights.gate_proj.scales,
            weights.gate_proj.biases,
            None,
            Some(&idx_for_qmm),
            true,
            weights.group_size,
            weights.bits,
            weights.mode,
            do_sort,
        )
        .context("switch_glu: gather_qmm(gate_proj) failed")?;

        // swiglu(gate, x_up) = silu(gate) * x_up. LUMEN_NATIVE_FUSE_SWIGLU=1
        // routes through compile-wrapped graph (1 dispatch, 1 fused kernel)
        // instead of explicit silu + multiply (2 dispatches, 2 kernels).
        let activated = if swiglu_fuse_enabled() {
            swiglu_fused(&x_gate, &x_up).context("switch_glu: fused swiglu failed")?
        } else {
            let silu_gate =
                mlx_rs::nn::silu(&x_gate).context("switch_glu: silu(gate_proj) failed")?;
            mlx_rs::ops::multiply(&silu_gate, &x_up)
                .context("switch_glu: silu(gate) * x_up failed")?
        };

        let down = gather_qmm_with_mode(
            &activated,
            weights.down_proj.weight,
            weights.down_proj.scales,
            weights.down_proj.biases,
            None,
            Some(&idx_for_qmm),
            true,
            weights.group_size,
            weights.bits,
            weights.mode,
            do_sort,
        )
        .context("switch_glu: gather_qmm(down_proj) failed")?;

        // _scatter_unsort + unflatten + squeeze(-2).
        let recombined = if let Some(inv_order) = inv_order_opt {
            let reordered = mlx_rs::ops::indexing::take_axis(&down, &inv_order, 0)
                .context("switch_glu: take_axis(down, inv_order) failed")?;
            // x.shape was [B*S*K, 1, hidden] after gather_qmm. Unflatten
            // axis 0 into [B, S, K]: result shape [B, S, K, 1, hidden].
            mlx_rs::ops::unflatten(&reordered, 0, &[b, s, k])
                .context("switch_glu: unflatten(0, [B,S,K]) failed")?
        } else {
            // No-sort path: down shape is [B, S, K, 1, hidden] already
            // (gather_qmm broadcasts x batch dims [B, S, 1] against
            // rhs_indices shape [B, S, K]).
            down
        };

        // squeeze axis -2 (the leftover M=1 from x's expand_dims).
        mlx_rs::ops::squeeze_axes(&recombined, &[-2])
            .context("switch_glu: squeeze(-2) failed")
    }

    /// Resolved tensors for the dense `Qwen3NextMLP` shared expert.
    pub struct SharedMlpWeights<'a> {
        pub gate_proj_w: &'a Array,
        pub gate_proj_s: &'a Array,
        pub up_proj_w: &'a Array,
        pub up_proj_s: &'a Array,
        pub down_proj_w: &'a Array,
        pub down_proj_s: &'a Array,
        pub group_size: i32,
        pub bits: i32,
        pub mode: &'a CStr,
    }

    /// `Qwen3NextMLP.__call__` — `down_proj(silu(gate_proj(x)) * up_proj(x))`.
    pub fn shared_mlp(x: &Array, w: &SharedMlpWeights<'_>) -> Result<Array> {
        let gate = linear_with_mode(
            x,
            w.gate_proj_w,
            w.gate_proj_s,
            None,
            w.group_size,
            w.bits,
            w.mode,
        )
        .context("shared_mlp: gate_proj failed")?;
        let up = linear_with_mode(
            x,
            w.up_proj_w,
            w.up_proj_s,
            None,
            w.group_size,
            w.bits,
            w.mode,
        )
        .context("shared_mlp: up_proj failed")?;
        let activated = if swiglu_fuse_enabled() {
            swiglu_fused(&gate, &up).context("shared_mlp: fused swiglu failed")?
        } else {
            let silu_gate =
                mlx_rs::nn::silu(&gate).context("shared_mlp: silu(gate) failed")?;
            mlx_rs::ops::multiply(&silu_gate, &up)
                .context("shared_mlp: silu(gate) * up failed")?
        };
        linear_with_mode(
            &activated,
            w.down_proj_w,
            w.down_proj_s,
            None,
            w.group_size,
            w.bits,
            w.mode,
        )
        .context("shared_mlp: down_proj failed")
    }

    /// Resolved tensors for the gate / shared_expert_gate (8-bit affine,
    /// group=64, with biases).
    pub struct AffineGateWeights<'a> {
        pub weight: &'a Array,
        pub scales: &'a Array,
        pub biases: &'a Array,
        pub group_size: i32,
        pub bits: i32,
    }

    /// `MoeBlockWeights` bundles the inputs `moe_block_forward` needs. The
    /// caller resolves all tensor handles + per-tensor quant params from the
    /// safetensors map up front so the hot path is purely compute.
    pub struct MoeBlockWeights<'a> {
        pub gate: AffineGateWeights<'a>,
        pub switch_glu: SwitchGluWeights<'a>,
        pub shared_expert: SharedMlpWeights<'a>,
        pub shared_gate: AffineGateWeights<'a>,
        pub top_k: i32,
        pub norm_topk_prob: bool,
    }

    /// Run a full sparse-MoE block on `x` (shape `[B, S, hidden]`).
    pub fn moe_block_forward(x: &Array, w: &MoeBlockWeights<'_>) -> Result<Array> {
        if x.ndim() != 3 {
            return Err(anyhow!(
                "moe_block_forward: expected x of rank 3 [B, S, hidden], got ndim={}",
                x.ndim()
            ));
        }
        let last_axis = (x.ndim() as i32) - 1;
        let timing_on = fine_timing_active();

        // (R: routing) gate matmul + softmax + argpartition + take_along + scores norm.
        // Lever B: dropped explicit `as_dtype(f32)` after gate matmul — the
        // affine quantized linear returns f32 when x is f32, so the cast was
        // a no-op graph node breaking softmax fusion.
        let t_routing = timing_on.then(Instant::now);
        let gate_logits = linear_with_mode(
            x,
            w.gate.weight,
            w.gate.scales,
            Some(w.gate.biases),
            w.gate.group_size,
            w.gate.bits,
            MODE_AFFINE,
        )
        .context("moe_block_forward: gate(x) failed")?;
        let last_logits_axis = (gate_logits.ndim() as i32) - 1;
        let num_experts = gate_logits.shape()[last_logits_axis as usize];

        let (inds, scores) = if routing_compile_enabled() && w.top_k == 8 && w.norm_topk_prob {
            // Cached compile path — Qwen3.5-MoE family (top_k=8, norm=true).
            // Matmul stays outside; only the post-matmul softmax→argpartition→
            // take_along→renorm chain is folded into the compiled graph.
            let mut out = invoke_compiled_multi(
                &ROUTING_SLOT,
                routing_compiled_inner,
                /* shapeless */ false,
                std::slice::from_ref(&gate_logits),
            )
            .context("moe_block_forward: routing cached compile failed")?;
            let scores = out
                .pop()
                .context("moe_block_forward: routing cache returned no scores")?;
            let inds = out
                .pop()
                .context("moe_block_forward: routing cache returned no inds")?;
            (inds, scores)
        } else {
            // Legacy direct-dispatch path (also catches non-Qwen3.5 configs).
            let gates = mlx_rs::ops::softmax_axis(&gate_logits, last_logits_axis, Some(true))
                .context("moe_block_forward: softmax(gate) failed")?;
            let part = mlx_rs::ops::argpartition_axis(&gates, -w.top_k, last_logits_axis)
                .context("moe_block_forward: argpartition(gates) failed")?;
            let start = num_experts - w.top_k;
            let inds = part.index((Ellipsis, start..));
            let scores = gates
                .take_along_axis(&inds, last_logits_axis)
                .context("moe_block_forward: take_along_axis(scores) failed")?;
            let scores = if w.norm_topk_prob {
                let denom = scores
                    .sum_axis(last_logits_axis, /* keep_dims */ true)
                    .context("moe_block_forward: scores.sum_axis failed")?;
                scores
                    .divide(&denom)
                    .context("moe_block_forward: scores renormalize failed")?
            } else {
                scores
            };
            (inds, scores)
        };
        if let Some(t0) = t_routing {
            mlx_rs::transforms::eval([&inds, &scores])
                .context("moe_block_forward: batched eval(routing) failed")?;
            bump_moe_routing_ms(t0.elapsed().as_secs_f64() * 1000.0);
        }

        // (S: switch_glu) per-expert gate_up + silu_mul + down_proj +
        //                 (y * scores).sum(axis=-2) reduction.
        // Lever B: dropped switch_glu output `as_dtype(f32)` — gather_qmm
        // returns f32 when x is f32; the cast was breaking
        // multiply/sum fusion across the wsum reduction.
        let t_glu = timing_on.then(Instant::now);
        let y = switch_glu(x, &inds, &w.switch_glu)
            .context("moe_block_forward: switch_glu failed")?;
        let scores_exp = mlx_rs::ops::expand_dims(&scores, -1)
            .context("moe_block_forward: expand_dims(scores) failed")?;
        let weighted = mlx_rs::ops::multiply(&y, &scores_exp)
            .context("moe_block_forward: y * scores failed")?;
        let combined = weighted
            .sum_axis(-2, /* keep_dims */ false)
            .context("moe_block_forward: sum_axis(-2) failed")?;
        if let Some(t0) = t_glu {
            combined
                .eval()
                .context("moe_block_forward: timing eval(switch_glu.combined) failed")?;
            bump_moe_switch_glu_ms(t0.elapsed().as_secs_f64() * 1000.0);
        }

        // (X: shared_expert) sigmoid(shared_gate(x)) * shared_expert(x).
        // Lever B: dropped 2× redundant `as_dtype(f32)` casts (shared_gate +
        // shared_mlp output) — both quantized matmuls return f32 already.
        let t_shared = timing_on.then(Instant::now);
        let shared_gate_logits = linear_with_mode(
            x,
            w.shared_gate.weight,
            w.shared_gate.scales,
            Some(w.shared_gate.biases),
            w.shared_gate.group_size,
            w.shared_gate.bits,
            MODE_AFFINE,
        )
        .context("moe_block_forward: shared_expert_gate(x) failed")?;
        let shared_y_raw = shared_mlp(x, &w.shared_expert)
            .context("moe_block_forward: shared_expert(x) failed")?;
        let shared_y = if sigmoid_mul_fuse_enabled() {
            sigmoid_mul_fused(&shared_gate_logits, &shared_y_raw)
                .context("moe_block_forward: fused sigmoid_mul(shared_gate, y) failed")?
        } else {
            let shared_gate_sig = mlx_rs::ops::sigmoid(&shared_gate_logits)
                .context("moe_block_forward: sigmoid(shared_gate) failed")?;
            mlx_rs::ops::multiply(&shared_gate_sig, &shared_y_raw)
                .context("moe_block_forward: gate * shared_y failed")?
        };
        if let Some(t0) = t_shared {
            shared_y
                .eval()
                .context("moe_block_forward: timing eval(shared.shared_y) failed")?;
            bump_moe_shared_expert_ms(t0.elapsed().as_secs_f64() * 1000.0);
        }

        // Final residual sum.
        let _ = last_axis; // last_axis was only documentation for axis algebra.
        mlx_rs::ops::add(&combined, &shared_y)
            .context("moe_block_forward: combined + shared_y failed")
    }

    /// Run a dense `Qwen3NextMLP` block as the post-attention MLP — used for
    /// any decoder layer flagged as non-MoE (`mlp_only_layers` membership).
    /// The Qwen3.5-MoE 35B production checkpoint flags every layer as MoE,
    /// but the helper keeps the dispatcher one-line clean.
    pub fn dense_mlp_forward(x: &Array, w: &SharedMlpWeights<'_>) -> Result<Array> {
        let raw = shared_mlp(x, w).context("dense_mlp_forward: shared_mlp failed")?;
        raw.as_dtype(mlx_rs::Dtype::Float32)
            .context("dense_mlp_forward: cast → f32 failed")
    }
}

#[cfg(feature = "mlx-native")]
#[allow(unused_imports)] // Consumed by Phase 3d.4 wiring in native_model.rs.
pub(crate) use imp::{
    AffineGateWeights, MoeBlockWeights, SharedMlpWeights, SwitchExpertWeights,
    SwitchGluWeights, dense_mlp_forward, gelu_mul_fuse_enabled, gelu_mul_fused,
    moe_block_forward, switch_glu,
};
