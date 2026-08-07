//! Native MLX Gemma 4 26B-A4B MoE model: config layer.
//!
//! Mirrors the `qwen3_5_moe` module's config shape so the eventual model
//! assembly can re-use the same NativeWeights/quant dispatch primitives.
//!
//! Reference: `.ai/memory/active/gemma4-26b-a4b-port/GAP.md` and mlx-lm's
//! `mlx_lm/models/gemma4_text.py`.

#[cfg(feature = "mlx-native")]
#[allow(dead_code)] // Phase 1 W1 (a): config-only landing; consumers come in W1 (b)+.
pub(crate) mod imp {
    use anyhow::{Context, Result, anyhow};
    use mlx_rs::Array;
    use serde::Deserialize;
    use std::collections::{BTreeMap, HashMap};
    use std::path::Path;
    use std::time::Instant;

    use std::ffi::CStr;

    use crate::kv_disk::{
        ArrayRecord, LayerKindTag, LayerMeta, record_from_array, record_to_array,
    };
    use crate::native_attention::{build_causal_mask, build_causal_mask_abs, sdpa, sdpa_with_mask};
    use crate::native_cache::{
        NativeKvCache, NativeKvCacheQuantized, NativeRotatingKvCache,
        NativeRotatingKvCacheQuantized, NativeRotatingKvCacheTurboQuant,
    };
    use crate::native_norm::rms_norm;
    // Shared with the Qwen 3.6 tower — see `vision_splice` for why the window
    // arithmetic lives outside both.
    use crate::native_quant::{
        MODE_AFFINE, MODE_MXFP4, MODE_MXFP8, MODE_NVFP4, dequantize_with_mode,
        gather_qmm_with_mode, quantize_with_mode, quantized_matmul_with_mode,
    };
    use crate::native_rope::{rope, rope_with_freqs};
    use crate::vision_splice::{clip_runs_to_window, image_token_runs};
    use mlx_rs::error::Exception;
    use mlx_rs::ops::indexing::{Ellipsis, IndexOp};
    use std::cell::Cell;
    use std::sync::OnceLock;

    // ───────────────────────── Per-step component breakdown ─────────────────
    //
    // env-gated component timing for the MoE decode hot path.
    // Activates when `LUMEN_GEMMA4_BREAKDOWN=1`. Off by default — all eval()
    // barriers and Instant::now() calls compile out to a single OnceLock read.
    //
    // When active, `decoder_layer_forward` inserts an explicit `eval()` after
    // each major section so the per-bucket time reflects GPU work (otherwise
    // MLX's lazy graph would bunch all evaluation into the next sync point and
    // attribute it to the wrong bucket). The added eval barriers serialize
    // dispatch and inflate absolute totals, so callers should read *ratios*
    // rather than treat the numbers as the un-instrumented runtime.

    static GEMMA4_BREAKDOWN_ENABLED: OnceLock<bool> = OnceLock::new();
    static GEMMA4_HONEST_BREAKDOWN_ENABLED: OnceLock<bool> = OnceLock::new();

    fn gemma4_breakdown_active() -> bool {
        *GEMMA4_BREAKDOWN_ENABLED.get_or_init(|| {
            std::env::var("LUMEN_GEMMA4_BREAKDOWN")
                .map(|s| matches!(s.as_str(), "1" | "true" | "TRUE" | "yes"))
                .unwrap_or(false)
        })
    }

    /// Honest CPU-only breakdown — times each stage's Rust-side FFI
    /// dispatch without the per-stage `eval()` barrier. Reveals which
    /// stage's per-op dispatch cost scales asymmetrically with
    /// context length. Legacy `LUMEN_GEMMA4_BREAKDOWN` mode's eval
    /// barriers defeat async-pipelined decode and inflate totals
    /// ~10–17× (see `phase_1_5_complete_70tok_parity_beat.md`
    /// "Honest decomposition" section).
    ///
    /// Numbers from this mode reflect pure Rust dispatch + any forced
    /// syncs MLX itself triggers internally (cmd_buffer commits,
    /// implicit waits on dependent ops). They do NOT attribute GPU
    /// compute time to stages — for that, the inflated mode is still
    /// useful as a *ratio* indicator within a single context length.
    fn gemma4_honest_breakdown_active() -> bool {
        *GEMMA4_HONEST_BREAKDOWN_ENABLED.get_or_init(|| {
            std::env::var("LUMEN_GEMMA4_HONEST_BREAKDOWN")
                .map(|s| matches!(s.as_str(), "1" | "true" | "TRUE" | "yes"))
                .unwrap_or(false)
        })
    }

    fn gemma4_any_breakdown_active() -> bool {
        gemma4_breakdown_active() || gemma4_honest_breakdown_active()
    }

    /// strategic `async_eval` at every K decoder
    /// layers to bound the lazy-graph depth that SDPA's input traversal
    /// pays for.
    ///
    /// Diagnosis (phase_1_5_complete_70tok_parity_beat.md): the bulk of
    /// the long-context regression is in the FFI layer surrounding SDPA
    /// — Rust-side measurement shows 2.35 ms/call at 4K, while MLX C++
    /// SDPA itself takes only 1.265 μs/call. The FFI cost scales with
    /// lazy-graph depth (q_rope/k_full/v_full carry long dependency
    /// chains: cache slice_update → slice → rope at each layer).
    ///
    /// `async_eval` schedules a non-blocking GPU drain, capping the
    /// graph depth without sync-waiting on the CPU side.
    ///
    /// `LUMEN_GEMMA4_ASYNC_EVAL_EVERY_K=N` sets the cadence (N=0 or
    /// unset disables). Small N (e.g. 1) drains every layer — too
    /// frequent, hurts pipelining. Large N (e.g. 30) is the same as
    /// no drain. Sweet spot found via sweep at 4K context.
    fn gemma4_async_eval_every_k() -> usize {
        static FLAG: OnceLock<usize> = OnceLock::new();
        *FLAG.get_or_init(|| {
            std::env::var("LUMEN_GEMMA4_ASYNC_EVAL_EVERY_K")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0)
        })
    }

    /// Hidden-state dump directory for the forward divergence debugger.
    /// When `LUMEN_DUMP_HIDDEN=/some/dir` is set, every position-wise
    /// hidden tensor in `forward_array` is serialized to that dir as a
    /// `TQHD` blob (matches the format `scripts/compare_hidden.py` reads).
    /// Off by default — the OnceLock makes the disabled path a single
    /// branch + Option read with no allocation.
    fn gemma4_dump_dir() -> Option<&'static str> {
        static DUMP_DIR: OnceLock<Option<String>> = OnceLock::new();
        DUMP_DIR
            .get_or_init(|| std::env::var("LUMEN_DUMP_HIDDEN").ok())
            .as_deref()
    }

    /// Step counter used by `dump_hidden` to namespace decode-step blobs.
    /// 0 = prefill (default), 1..N = decode step. Set externally by the
    /// debug example via `set_forward_step` before each `forward_array` call.
    static FORWARD_STEP: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    pub fn set_forward_step(step: usize) {
        FORWARD_STEP.store(step, std::sync::atomic::Ordering::Relaxed);
    }

    fn forward_step() -> usize {
        FORWARD_STEP.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Which decode step to dump. When `LUMEN_DUMP_DECODE_STEP=N` is set,
    /// only that step's hidden states are written (with `stepN_` prefix).
    /// `None` means decode-step dumps are skipped entirely (prefill still
    /// dumps to plain `{name}.bin` when its L>1 guard passes).
    fn gemma4_dump_decode_step() -> Option<usize> {
        static V: OnceLock<Option<usize>> = OnceLock::new();
        *V.get_or_init(|| {
            std::env::var("LUMEN_DUMP_DECODE_STEP")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
        })
    }

    /// Serialize `arr` to `{dump_dir}/{name}.bin` as
    /// `b"TQHD" + rank(u32) + dims(u32 each, LE) + f32 bytes`.
    /// No-op when `LUMEN_DUMP_HIDDEN` is unset.
    fn dump_hidden(arr: &Array, name: &str) -> Result<()> {
        let Some(dir) = gemma4_dump_dir() else {
            return Ok(());
        };
        let step = forward_step();
        let path = if step == 0 {
            // Prefill: dim 1 of [B, L, …] is sequence length. Skip L≤1 so
            // decode-shaped accidental call sites don't clobber the prefill
            // snapshot when the decode-step gate is off.
            if arr.shape().get(1).copied().unwrap_or(0) <= 1 {
                return Ok(());
            }
            format!("{dir}/{name}.bin")
        } else {
            // Decode: only dump when the active step matches the env-gated
            // target (LUMEN_DUMP_DECODE_STEP). Other steps run cache-only.
            match gemma4_dump_decode_step() {
                Some(target) if target == step => format!("{dir}/step{step}_{name}.bin"),
                _ => return Ok(()),
            }
        };
        use std::io::Write;
        std::fs::create_dir_all(dir).with_context(|| format!("dump_hidden: mkdir {dir}"))?;
        let casted = arr
            .as_dtype(mlx_rs::Dtype::Float32)
            .with_context(|| format!("dump_hidden: cast {name} to f32"))?;
        casted
            .eval()
            .with_context(|| format!("dump_hidden: eval {name}"))?;
        let dims = casted.shape().to_vec();
        let data: &[f32] = casted.as_slice::<f32>();
        let mut f =
            std::fs::File::create(&path).with_context(|| format!("dump_hidden: create {path}"))?;
        f.write_all(b"TQHD")?;
        f.write_all(&(dims.len() as u32).to_le_bytes())?;
        for d in &dims {
            f.write_all(&(*d as u32).to_le_bytes())?;
        }
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
        f.write_all(bytes)?;
        Ok(())
    }

    thread_local! {
        static GEMMA4_ATTN_MS: Cell<f64> = const { Cell::new(0.0) };
        static GEMMA4_ATTN_FULL_MS: Cell<f64> = const { Cell::new(0.0) };
        static GEMMA4_ATTN_SLIDING_MS: Cell<f64> = const { Cell::new(0.0) };
        // Attention sub-stage CPU dispatch buckets (honest mode only).
        // Used to localize the per-op cost growth we observe between
        // short and long context: total attn CPU dispatch grew 86× from
        // 8 → 4K tokens while all other stages stayed flat. Sub-buckets
        // pinpoint which op(s) carry the regression.
        static GEMMA4_ATTN_QKVO_MS: Cell<f64> = const { Cell::new(0.0) };
        static GEMMA4_ATTN_NORMS_MS: Cell<f64> = const { Cell::new(0.0) };
        static GEMMA4_ATTN_ROPE_MS: Cell<f64> = const { Cell::new(0.0) };
        static GEMMA4_ATTN_CACHE_MS: Cell<f64> = const { Cell::new(0.0) };
        static GEMMA4_ATTN_SDPA_MS: Cell<f64> = const { Cell::new(0.0) };
        // wraps ONLY the sdpa() call,
        // excluding mask construction / match / outer bookkeeping.
        static GEMMA4_ATTN_SDPA_TIGHT_MS: Cell<f64> = const { Cell::new(0.0) };
        static GEMMA4_DENSE_MS: Cell<f64> = const { Cell::new(0.0) };
        static GEMMA4_ROUTER_MS: Cell<f64> = const { Cell::new(0.0) };
        static GEMMA4_EXPERTS_MS: Cell<f64> = const { Cell::new(0.0) };
    }

    fn bump_gemma4_attn_ms(ms: f64) {
        GEMMA4_ATTN_MS.with(|c| c.set(c.get() + ms));
    }
    pub(crate) fn bump_gemma4_attn_full_ms(ms: f64) {
        GEMMA4_ATTN_FULL_MS.with(|c| c.set(c.get() + ms));
    }
    pub(crate) fn bump_gemma4_attn_sliding_ms(ms: f64) {
        GEMMA4_ATTN_SLIDING_MS.with(|c| c.set(c.get() + ms));
    }
    fn bump_gemma4_attn_qkvo_ms(ms: f64) {
        GEMMA4_ATTN_QKVO_MS.with(|c| c.set(c.get() + ms));
    }
    fn bump_gemma4_attn_norms_ms(ms: f64) {
        GEMMA4_ATTN_NORMS_MS.with(|c| c.set(c.get() + ms));
    }
    fn bump_gemma4_attn_rope_ms(ms: f64) {
        GEMMA4_ATTN_ROPE_MS.with(|c| c.set(c.get() + ms));
    }
    fn bump_gemma4_attn_cache_ms(ms: f64) {
        GEMMA4_ATTN_CACHE_MS.with(|c| c.set(c.get() + ms));
    }
    fn bump_gemma4_attn_sdpa_ms(ms: f64) {
        GEMMA4_ATTN_SDPA_MS.with(|c| c.set(c.get() + ms));
    }
    fn bump_gemma4_attn_sdpa_tight_ms(ms: f64) {
        GEMMA4_ATTN_SDPA_TIGHT_MS.with(|c| c.set(c.get() + ms));
    }
    fn bump_gemma4_dense_ms(ms: f64) {
        GEMMA4_DENSE_MS.with(|c| c.set(c.get() + ms));
    }
    fn bump_gemma4_router_ms(ms: f64) {
        GEMMA4_ROUTER_MS.with(|c| c.set(c.get() + ms));
    }
    fn bump_gemma4_experts_ms(ms: f64) {
        GEMMA4_EXPERTS_MS.with(|c| c.set(c.get() + ms));
    }

    fn reset_gemma4_breakdown() {
        GEMMA4_ATTN_MS.with(|c| c.set(0.0));
        GEMMA4_ATTN_FULL_MS.with(|c| c.set(0.0));
        GEMMA4_ATTN_SLIDING_MS.with(|c| c.set(0.0));
        GEMMA4_ATTN_QKVO_MS.with(|c| c.set(0.0));
        GEMMA4_ATTN_NORMS_MS.with(|c| c.set(0.0));
        GEMMA4_ATTN_ROPE_MS.with(|c| c.set(0.0));
        GEMMA4_ATTN_CACHE_MS.with(|c| c.set(0.0));
        GEMMA4_ATTN_SDPA_MS.with(|c| c.set(0.0));
        GEMMA4_ATTN_SDPA_TIGHT_MS.with(|c| c.set(0.0));
        GEMMA4_DENSE_MS.with(|c| c.set(0.0));
        GEMMA4_ROUTER_MS.with(|c| c.set(0.0));
        GEMMA4_EXPERTS_MS.with(|c| c.set(0.0));
    }

    /// Snapshot of the breakdown counters. Read after `generate()` completes
    /// and divide by `decode_steps` to get per-step averages.
    #[derive(Debug, Clone, Copy, Default)]
    pub struct Gemma4Breakdown {
        pub attn_ms: f64,
        pub attn_full_ms: f64,
        pub attn_sliding_ms: f64,
        pub attn_qkvo_ms: f64,
        pub attn_norms_ms: f64,
        pub attn_rope_ms: f64,
        pub attn_cache_ms: f64,
        pub attn_sdpa_ms: f64,
        pub attn_sdpa_tight_ms: f64,
        pub dense_ms: f64,
        pub router_ms: f64,
        pub experts_ms: f64,
    }

    pub fn take_gemma4_breakdown() -> Gemma4Breakdown {
        let b = Gemma4Breakdown {
            attn_ms: GEMMA4_ATTN_MS.with(|c| c.get()),
            attn_full_ms: GEMMA4_ATTN_FULL_MS.with(|c| c.get()),
            attn_sliding_ms: GEMMA4_ATTN_SLIDING_MS.with(|c| c.get()),
            attn_qkvo_ms: GEMMA4_ATTN_QKVO_MS.with(|c| c.get()),
            attn_norms_ms: GEMMA4_ATTN_NORMS_MS.with(|c| c.get()),
            attn_rope_ms: GEMMA4_ATTN_ROPE_MS.with(|c| c.get()),
            attn_cache_ms: GEMMA4_ATTN_CACHE_MS.with(|c| c.get()),
            attn_sdpa_ms: GEMMA4_ATTN_SDPA_MS.with(|c| c.get()),
            attn_sdpa_tight_ms: GEMMA4_ATTN_SDPA_TIGHT_MS.with(|c| c.get()),
            dense_ms: GEMMA4_DENSE_MS.with(|c| c.get()),
            router_ms: GEMMA4_ROUTER_MS.with(|c| c.get()),
            experts_ms: GEMMA4_EXPERTS_MS.with(|c| c.get()),
        };
        reset_gemma4_breakdown();
        b
    }

    // ───────────────────────── Router compile-fusion (Phase 1.5 P2) ─────
    //
    // Folds the Gemma-4 router post-projection chain
    //
    //   argpartition(scores, -K, -1)[..., -K:]   ← top-K indices
    //   take_along_axis(scores, indices, -1)     ← gather top-K raw logits
    //   softmax(top_logits, -1, precise=true)    ← softmax across the K slice
    //   take_axis(per_expert_scale, indices, 0)  ← per-expert post-multiplier
    //   weights × per_expert                     ← final routing weights
    //
    // into a single compiled mlx graph cached for process lifetime. Without
    // fusion this is 5 mlx-rs FFI dispatches per layer × 30 layers = 150
    // small-op dispatches per decode step.
    //
    // The mlx-lm Python reference effectively compiles this chain through
    // `mx.compile` on the full forward graph; the per-layer breakdown
    // (W6 perf profile) showed routing at ~18.8% of decode wall, which is
    // the dispatch overhead we expect to recover.
    //
    // Env gate: `LUMEN_GEMMA4_FUSE_ROUTER` (default ON 2026-05-12 — we
    // ship the win-direction default per `playbook_perf_lever_discovery`;
    // `=0` falls back to the unfused path for A/B parity sanity).
    //
    // **Pre-projection ops (rms_norm + quantized matmul) stay outside the
    // compile slot.** rms_norm is a single mlx-fast call already, and the
    // quantized matmul carries the `mode` CStr + per-tensor metadata which
    // does not survive `compile_boxed_slice_refs`'s arg list.
    //
    // Note: `top_k` is the value baked into the *first* trace — for Gemma 4
    // 26B-A4B that is always 8 (top_k_experts from config.json), so the
    // single OnceLock slot covers all current deployments. A future model
    // family with a different top_k would need its own slot.
    // Compile slot covers the **post-slice** routing tail:
    //   take_along_axis(scores, indices, -1) → softmax(precise) →
    //   take_axis(per_expert_scale, indices) → ×
    //
    // argpartition + Ellipsis-slice stay outside the compile because mlx
    // compile's `Primitive::output_shapes` cannot infer the slice output
    // shape from a dynamic start (`num_experts - top_k`). Keeping the slice
    // out trims the fuse to 4 ops, which still saves 3 FFI dispatches per
    // layer (×30 layers per decode step).
    fn gemma4_routing_compiled_inner(args: &[Array]) -> std::result::Result<Vec<Array>, Exception> {
        // args[0] = scores  (post-projection logits, [B, L, num_experts])
        // args[1] = indices ([B, L, top_k] — sliced from argpartition outside)
        // args[2] = per_expert_scale ([num_experts])
        let scores = &args[0];
        let indices = &args[1];
        let per_expert_scale = &args[2];
        let last_axis = (scores.ndim() as i32) - 1;

        let top_logits = scores.take_along_axis(indices, last_axis)?;
        let weights = mlx_rs::ops::softmax_axis(&top_logits, last_axis, Some(true))?;
        let per_expert = mlx_rs::ops::indexing::take_axis(per_expert_scale, indices, 0)?;
        let weights = weights.multiply(&per_expert)?;
        Ok(vec![weights])
    }

    static GEMMA4_ROUTING_SLOT: OnceLock<
        std::sync::Mutex<crate::native_compile_cache::CompiledMultiRefs>,
    > = OnceLock::new();

    fn gemma4_router_fuse_enabled() -> bool {
        std::env::var("LUMEN_GEMMA4_FUSE_ROUTER")
            .map(|v| v != "0")
            .unwrap_or(true)
    }

    // ───────────────────────── Dense MLP compile (Phase 1.5 P9) ────────
    //
    // Compile-cache the full Gemma 4 dense MLP chain — 3 quantized matmuls
    // (gate_proj, up_proj, down_proj at 8-bit affine) + the fused GeGLU.
    // Without this, each layer's dense MLP fires 4 separate mlx-c FFI calls
    // (3 qmatmul + 1 compile-traced GeGLU). With the compile slot, the
    // whole chain becomes 1 traced graph reused across all 30 layers.
    //
    // Capture: `MODE_AFFINE` CStr, `group_size=64`, `bits=8`, transpose=true
    // are baked at first-trace time (shared across all dense MLP calls).
    //
    // Why constants survive: `compile_boxed_slice_refs` (mlx-rs fork) traces
    // the inner fn ONCE on first invocation; subsequent calls reuse the
    // compiled graph. The CStr / int values inside the closure body are
    // captured into the traced graph at that single trace time, then
    // reused for every dispatch.
    //
    // Args layout: [x, gate_w, gate_s, gate_b, up_w, up_s, up_b, down_w, down_s, down_b]
    // (Gemma 4 affine MLP always has biases, so 10 arrays — no Option.)
    fn gemma4_dense_mlp_compiled_inner(
        args: &[Array],
    ) -> std::result::Result<Vec<Array>, Exception> {
        use mlx_rs::ops::tanh;
        let x = &args[0];
        let gate_w = &args[1];
        let gate_s = &args[2];
        let gate_b = &args[3];
        let up_w = &args[4];
        let up_s = &args[5];
        let up_b = &args[6];
        let down_w = &args[7];
        let down_s = &args[8];
        let down_b = &args[9];

        // Use mlx-rs public `quantized_matmul` (mode = "affine" by default,
        // matching Gemma 4 dense MLP's 8-bit affine quant). The lazy compile
        // path traces ops during first invocation; the stream is restamped at
        // trace time so we use the default-device variant here.
        let qmm = |xi: &Array,
                   w: &Array,
                   s: &Array,
                   b: &Array|
         -> std::result::Result<Array, Exception> {
            mlx_rs::ops::quantized_matmul(
                xi, w, s, b, /* transpose */ true, /* group_size */ 64, /* bits */ 8,
            )
        };

        let gate = qmm(x, gate_w, gate_s, gate_b)?;
        let up = qmm(x, up_w, up_s, up_b)?;

        // GeGLU: gelu_approx(gate) * up — inlined formula to avoid nested
        // compile (gelu_mul_fused is itself a compile slot, can't nest).
        // match `gate.dtype()` so the bf16 model
        // doesn't trigger implicit AsType primitives on every multiply.
        let dt = gate.dtype();
        let half = Array::from_f32(0.5).as_dtype(dt)?;
        let one = Array::from_f32(1.0).as_dtype(dt)?;
        let c3 = Array::from_f32(0.044715).as_dtype(dt)?;
        let coeff = Array::from_f32(0.7978845608028654_f32).as_dtype(dt)?;
        let x_squared = gate.multiply(&gate)?;
        let x_cubed = x_squared.multiply(&gate)?;
        let inner_add = gate.add(&x_cubed.multiply(&c3)?)?;
        let scaled = coeff.multiply(&inner_add)?;
        let t = tanh(&scaled)?;
        let one_plus_tanh = one.add(&t)?;
        let activated = half
            .multiply(&gate)?
            .multiply(&one_plus_tanh)?
            .multiply(&up)?;

        let down = qmm(&activated, down_w, down_s, down_b)?;
        Ok(vec![down])
    }

    static GEMMA4_DENSE_MLP_SLOT: OnceLock<
        std::sync::Mutex<crate::native_compile_cache::CompiledMultiRefs>,
    > = OnceLock::new();

    fn gemma4_dense_mlp_fuse_enabled() -> bool {
        std::env::var("LUMEN_GEMMA4_FUSE_DENSE_MLP")
            .map(|v| v != "0")
            .unwrap_or(true)
    }

    // Tier 2C (2026-05-16) — pre+post-norm absorbed dense MLP compile slot.
    //
    // Extends `gemma4_dense_mlp_compiled_inner` to bake the
    // `pre_feedforward_layernorm_1` and `post_feedforward_layernorm_1`
    // rms_norm ops into the same traced graph. Per-layer kernel launches
    // for the dense MLP path drop from 3 (pre_norm, fused MLP, post_norm)
    // to 1.
    //
    // Args layout (12 arrays):
    //   [x, pre_norm_w, post_norm_w,
    //    gate_w, gate_s, gate_b, up_w, up_s, up_b, down_w, down_s, down_b]
    //
    // Hardcoded: rms_norm eps = 1e-6 (Gemma 4 text_config.rms_norm_eps).
    fn gemma4_pre_post_norm_dense_mlp_compiled_inner(
        args: &[Array],
    ) -> std::result::Result<Vec<Array>, Exception> {
        use mlx_rs::ops::tanh;
        let x = &args[0];
        let pre_norm_w = &args[1];
        let post_norm_w = &args[2];
        let gate_w = &args[3];
        let gate_s = &args[4];
        let gate_b = &args[5];
        let up_w = &args[6];
        let up_s = &args[7];
        let up_b = &args[8];
        let down_w = &args[9];
        let down_s = &args[10];
        let down_b = &args[11];

        let normed = mlx_rs::fast::rms_norm(x, pre_norm_w, 1e-6_f32)?;

        let qmm = |xi: &Array,
                   w: &Array,
                   s: &Array,
                   b: &Array|
         -> std::result::Result<Array, Exception> {
            mlx_rs::ops::quantized_matmul(
                xi, w, s, b, /* transpose */ true, /* group_size */ 64, /* bits */ 8,
            )
        };

        let gate = qmm(&normed, gate_w, gate_s, gate_b)?;
        let up = qmm(&normed, up_w, up_s, up_b)?;

        let dt = gate.dtype();
        let half = Array::from_f32(0.5).as_dtype(dt)?;
        let one = Array::from_f32(1.0).as_dtype(dt)?;
        let c3 = Array::from_f32(0.044715).as_dtype(dt)?;
        let coeff = Array::from_f32(0.7978845608028654_f32).as_dtype(dt)?;
        let x_squared = gate.multiply(&gate)?;
        let x_cubed = x_squared.multiply(&gate)?;
        let inner_add = gate.add(&x_cubed.multiply(&c3)?)?;
        let scaled = coeff.multiply(&inner_add)?;
        let t = tanh(&scaled)?;
        let one_plus_tanh = one.add(&t)?;
        let activated = half
            .multiply(&gate)?
            .multiply(&one_plus_tanh)?
            .multiply(&up)?;

        let down = qmm(&activated, down_w, down_s, down_b)?;
        let result = mlx_rs::fast::rms_norm(&down, post_norm_w, 1e-6_f32)?;
        Ok(vec![result])
    }

    static GEMMA4_PRE_POST_NORM_DENSE_MLP_SLOT: OnceLock<
        std::sync::Mutex<crate::native_compile_cache::CompiledMultiRefs>,
    > = OnceLock::new();

    fn gemma4_pre_post_norm_dense_mlp_fuse_enabled() -> bool {
        std::env::var("LUMEN_GEMMA4_FUSE_PRE_POST_NORM_DENSE_MLP")
            .map(|v| v == "1")
            .unwrap_or(false)
    }

    fn gemma4_pre_post_norm_dense_mlp_fused(
        x: &Array,
        pre_norm_w: &Array,
        post_norm_w: &Array,
        w: &ResolvedGemma4DenseMlpWeights,
    ) -> Result<Array> {
        let gate_b = w.gate_proj.biases.as_ref().ok_or_else(|| {
            anyhow!("gemma4_pre_post_norm_dense_mlp_fused: gate_proj.biases is None")
        })?;
        let up_b = w.up_proj.biases.as_ref().ok_or_else(|| {
            anyhow!("gemma4_pre_post_norm_dense_mlp_fused: up_proj.biases is None")
        })?;
        let down_b = w.down_proj.biases.as_ref().ok_or_else(|| {
            anyhow!("gemma4_pre_post_norm_dense_mlp_fused: down_proj.biases is None")
        })?;
        let args = [
            x,
            pre_norm_w,
            post_norm_w,
            &w.gate_proj.weight,
            &w.gate_proj.scales,
            gate_b,
            &w.up_proj.weight,
            &w.up_proj.scales,
            up_b,
            &w.down_proj.weight,
            &w.down_proj.scales,
            down_b,
        ];
        let mut out = crate::native_compile_cache::invoke_compiled_multi_refs(
            &GEMMA4_PRE_POST_NORM_DENSE_MLP_SLOT,
            gemma4_pre_post_norm_dense_mlp_compiled_inner,
            /* shapeless */ true,
            &args,
        )
        .context("gemma4_pre_post_norm_dense_mlp_fused: mlx compile dispatch failed")?;
        out.pop()
            .context("gemma4_pre_post_norm_dense_mlp_fused: missing output")
    }

    fn gemma4_dense_mlp_fused(x: &Array, w: &ResolvedGemma4DenseMlpWeights) -> Result<Array> {
        // Dense MLP biases must be present for the compile path (10 args
        // assumed). Fall back to legacy if any bias is None.
        let gate_b = match w.gate_proj.biases.as_ref() {
            Some(b) => b,
            None => {
                return Err(anyhow!(
                    "gemma4_dense_mlp_fused: gate_proj.biases is None — falling back required"
                ));
            }
        };
        let up_b = match w.up_proj.biases.as_ref() {
            Some(b) => b,
            None => return Err(anyhow!("gemma4_dense_mlp_fused: up_proj.biases is None")),
        };
        let down_b = match w.down_proj.biases.as_ref() {
            Some(b) => b,
            None => return Err(anyhow!("gemma4_dense_mlp_fused: down_proj.biases is None")),
        };
        let args = [
            x,
            &w.gate_proj.weight,
            &w.gate_proj.scales,
            gate_b,
            &w.up_proj.weight,
            &w.up_proj.scales,
            up_b,
            &w.down_proj.weight,
            &w.down_proj.scales,
            down_b,
        ];
        let mut out = crate::native_compile_cache::invoke_compiled_multi_refs(
            &GEMMA4_DENSE_MLP_SLOT,
            gemma4_dense_mlp_compiled_inner,
            /* shapeless */ true,
            &args,
        )
        .context("gemma4_dense_mlp_fused: mlx compile dispatch failed")?;
        out.pop().context("gemma4_dense_mlp_fused: missing output")
    }

    /// Run the fused tail. `indices` must already be sliced to shape
    /// `[B, L, top_k]` (argpartition + Ellipsis-slice happens outside the
    /// compile slot — see `gemma4_routing_compiled_inner` doc).
    fn gemma4_routing_fused_tail(
        scores: &Array,
        indices: &Array,
        per_expert_scale: &Array,
    ) -> Result<Array> {
        let args = [scores, indices, per_expert_scale];
        let mut out = crate::native_compile_cache::invoke_compiled_multi_refs(
            &GEMMA4_ROUTING_SLOT,
            gemma4_routing_compiled_inner,
            /* shapeless */ true,
            &args,
        )
        .context("gemma4_routing_fused: mlx compile dispatch failed")?;
        out.pop()
            .context("gemma4_routing_fused: missing weights output")
    }

    // ───────────────────────── Experts compile (Phase 1.5 P10) ────────
    //
    // Fuse the MoE expert path's 3 gather_qmm calls (gate/up/down at 4-bit
    // affine) + the GeGLU activation into a single compile-cached graph.
    //
    // Fused only on the **no-sort decode branch** (`do_sort=false`, i.e.
    // `B*L*K < 64` — every decode step hits this since B=1, L=1, K=8 → 8 < 64).
    // The sort branch (prefill / batched decode) keeps the legacy path because:
    //   1. The sort pre-amble (argsort + take_axis) and post-amble (inv_order
    //      gather + unflatten) stay outside the compile slot anyway.
    //   2. `sorted_indices` is a boolean baked into gather_qmm's traced graph
    //      at first compile — separating the two branches with two slots keeps
    //      each slot homogeneous.
    //
    // Args (11 arrays):
    //   [x, idx,
    //    gate_w, gate_s, gate_b,
    //    up_w,   up_s,   up_b,
    //    down_w, down_s, down_b]
    //
    // Hardcoded params (baked at trace time):
    //   transpose=true, group_size=64, bits=4, mode="affine",
    //   lhs_indices=None, sorted_indices=false
    //
    // GeGLU formula inlined (gelu_approx(gate) * up) — can't nest the existing
    // gelu_mul compile slot inside another compile.
    fn gemma4_experts_compiled_inner(args: &[Array]) -> std::result::Result<Vec<Array>, Exception> {
        use mlx_rs::ops::tanh;
        let x = &args[0];
        let idx = &args[1];
        let gate_w = &args[2];
        let gate_s = &args[3];
        let gate_b = &args[4];
        let up_w = &args[5];
        let up_s = &args[6];
        let up_b = &args[7];
        let down_w = &args[8];
        let down_s = &args[9];
        let down_b = &args[10];

        // mlx-rs's macro-generated `gather_qmm` uses DEFAULT_MODE="affine",
        // matching Gemma 4 experts (4-bit affine). The stream is restamped at
        // trace time so we use the default-device variant.
        let gqmm = |xi: &Array,
                    w: &Array,
                    s: &Array,
                    b: &Array|
         -> std::result::Result<Array, Exception> {
            mlx_rs::ops::gather_qmm(
                xi,
                w,
                s,
                b,
                /* lhs_indices */ None,
                /* rhs_indices */ Some(idx),
                /* transpose */ true,
                /* group_size */ 64,
                /* bits */ 4,
                /* sorted_indices */ false,
            )
        };

        let gate = gqmm(x, gate_w, gate_s, gate_b)?;
        let up = gqmm(x, up_w, up_s, up_b)?;

        // GeGLU: gelu_approx(gate) * up — inlined formula (same constants as
        // P9's gemma4_dense_mlp_compiled_inner; can't nest compile slots).
        // match gate.dtype() to avoid implicit AsType.
        let dt = gate.dtype();
        let half = Array::from_f32(0.5).as_dtype(dt)?;
        let one = Array::from_f32(1.0).as_dtype(dt)?;
        let c3 = Array::from_f32(0.044715).as_dtype(dt)?;
        let coeff = Array::from_f32(0.7978845608028654_f32).as_dtype(dt)?;
        let x_squared = gate.multiply(&gate)?;
        let x_cubed = x_squared.multiply(&gate)?;
        let inner_add = gate.add(&x_cubed.multiply(&c3)?)?;
        let scaled = coeff.multiply(&inner_add)?;
        let t = tanh(&scaled)?;
        let one_plus_tanh = one.add(&t)?;
        let activated = half
            .multiply(&gate)?
            .multiply(&one_plus_tanh)?
            .multiply(&up)?;

        let down = gqmm(&activated, down_w, down_s, down_b)?;
        Ok(vec![down])
    }

    static GEMMA4_EXPERTS_SLOT: OnceLock<
        std::sync::Mutex<crate::native_compile_cache::CompiledMultiRefs>,
    > = OnceLock::new();

    fn gemma4_experts_fuse_enabled() -> bool {
        std::env::var("LUMEN_GEMMA4_FUSE_EXPERTS")
            .map(|v| v != "0")
            .unwrap_or(true)
    }

    // ─────────── Combined routing+experts compile (lever exploration 2026-05-17) ─────
    //
    // Merge `gemma4_routing_fused_tail` + `gemma4_experts_fused` + the
    // outside-the-slot recombine ops (squeeze + expand_dims + multiply + sum)
    // into one compile slot. Eliminates 2 slot-invocation boundaries and 4
    // tiny ops per layer × 25 MoE layers = 150 op dispatches per decode step.
    //
    // Args (14 arrays):
    //   [scores, indices, per_expert_scale, x_for_experts,
    //    gate_w, gate_s, gate_b, up_w, up_s, up_b, down_w, down_s, down_b]
    //
    //   scores: [B, L, num_experts]
    //   indices: [B, L, top_k]      (already sliced from argpartition outside)
    //   per_expert_scale: [num_experts]
    //   x_for_experts: [B, L, hidden]  (un-expanded, plain experts input)
    //
    // Output: [B, L, hidden] (summed over K experts, weighted)
    //
    // Restricted to the no-sort decode branch (top_k=8 < 64), 4-bit affine,
    // group_size=64 — matches `gemma4_experts_fused` constraints.
    fn gemma4_routing_experts_compiled_inner(
        args: &[Array],
    ) -> std::result::Result<Vec<Array>, Exception> {
        use mlx_rs::ops::tanh;

        let scores = &args[0];
        let indices = &args[1];
        let per_expert_scale = &args[2];
        let x = &args[3];
        let gate_w = &args[4];
        let gate_s = &args[5];
        let gate_b = &args[6];
        let up_w = &args[7];
        let up_s = &args[8];
        let up_b = &args[9];
        let down_w = &args[10];
        let down_s = &args[11];
        let down_b = &args[12];

        // 1. Routing tail: take_along + softmax + per_expert_scale gather + multiply.
        let last_axis = (scores.ndim() as i32) - 1;
        let top_logits = scores.take_along_axis(indices, last_axis)?;
        let weights = mlx_rs::ops::softmax_axis(&top_logits, last_axis, Some(true))?;
        let per_expert = mlx_rs::ops::indexing::take_axis(per_expert_scale, indices, 0)?;
        let weights = weights.multiply(&per_expert)?;

        // 2. expand_dims(x, [-2, -3]) → [B, L, 1, 1, hidden] for gather_qmm.
        let x_5d = mlx_rs::ops::expand_dims_axes(x, &[-2, -3])?;

        // 3. Experts: gate / up / GeGLU / down (mirrors gemma4_experts_compiled_inner).
        let gqmm = |xi: &Array,
                    w: &Array,
                    s: &Array,
                    b: &Array|
         -> std::result::Result<Array, Exception> {
            mlx_rs::ops::gather_qmm(
                xi,
                w,
                s,
                b,
                /* lhs_indices */ None,
                /* rhs_indices */ Some(indices),
                /* transpose */ true,
                /* group_size */ 64,
                /* bits */ 4,
                /* sorted_indices */ false,
            )
        };

        let gate = gqmm(&x_5d, gate_w, gate_s, gate_b)?;
        let up = gqmm(&x_5d, up_w, up_s, up_b)?;

        let dt = gate.dtype();
        let half = Array::from_f32(0.5).as_dtype(dt)?;
        let one = Array::from_f32(1.0).as_dtype(dt)?;
        let c3 = Array::from_f32(0.044715).as_dtype(dt)?;
        let coeff = Array::from_f32(0.7978845608028654_f32).as_dtype(dt)?;
        let x_squared = gate.multiply(&gate)?;
        let x_cubed = x_squared.multiply(&gate)?;
        let inner_add = gate.add(&x_cubed.multiply(&c3)?)?;
        let scaled = coeff.multiply(&inner_add)?;
        let t = tanh(&scaled)?;
        let one_plus_tanh = one.add(&t)?;
        let activated = half
            .multiply(&gate)?
            .multiply(&one_plus_tanh)?
            .multiply(&up)?;

        let down = gqmm(&activated, down_w, down_s, down_b)?;

        // 4. Recombine: squeeze(-2) → multiply by weights[..., None] → sum(axis=-2).
        let per_expert_out = mlx_rs::ops::squeeze_axes(&down, &[-2])?;
        let w_expanded = mlx_rs::ops::expand_dims_axes(&weights, &[-1])?;
        let weighted = mlx_rs::ops::multiply(&per_expert_out, &w_expanded)?;
        let summed = mlx_rs::ops::sum_axis(&weighted, -2, /* keep_dims */ false)?;

        Ok(vec![summed])
    }

    static GEMMA4_ROUTING_EXPERTS_SLOT: OnceLock<
        std::sync::Mutex<crate::native_compile_cache::CompiledMultiRefs>,
    > = OnceLock::new();

    fn gemma4_routing_experts_fuse_enabled() -> bool {
        // LANDED 2026-05-17 — bit-identical with baseline (greedy tokens
        // match exactly). 3-pair cool-state A/B at 8K shows +2.2% throughput
        // (53.35 ± 0.5 → 54.5 ± 0.7 tok/s). Default ON; set the env to
        // "0" to revert to the two-slot routing_tail + experts path.
        std::env::var("LUMEN_GEMMA4_FUSE_ROUTING_EXPERTS")
            .map(|v| v != "0")
            .unwrap_or(true)
    }

    // ─ Pre+post-norm absorbed into routing+experts (lever exploration 2026-05-17) ─
    //
    // Extends `gemma4_routing_experts_compiled_inner` to also bake
    // `pre_feedforward_layernorm_2` (applied to h before experts) and
    // `post_feedforward_layernorm_2` (applied to the summed output) into the
    // same compile slot. Analog of `gemma4_pre_post_norm_dense_mlp` (Tier 2C,
    // NEUTRAL on dense path) but riding the +2.2% routing+experts win.
    //
    // Args (15 arrays):
    //   [scores, indices, per_expert_scale, h_raw, pre_norm_w, post_norm_w,
    //    gate_w, gate_s, gate_b, up_w, up_s, up_b, down_w, down_s, down_b]
    //
    //   h_raw: pre-norm-2 input (un-normed)
    //   pre_norm_w / post_norm_w: rms_norm weights (eps = 1e-6 baked in)
    //
    // Output: [B, L, hidden] (final post-norm output)
    fn gemma4_pre_post_norm_routing_experts_compiled_inner(
        args: &[Array],
    ) -> std::result::Result<Vec<Array>, Exception> {
        use mlx_rs::ops::tanh;

        let scores = &args[0];
        let indices = &args[1];
        let per_expert_scale = &args[2];
        let h_raw = &args[3];
        let pre_norm_w = &args[4];
        let post_norm_w = &args[5];
        let gate_w = &args[6];
        let gate_s = &args[7];
        let gate_b = &args[8];
        let up_w = &args[9];
        let up_s = &args[10];
        let up_b = &args[11];
        let down_w = &args[12];
        let down_s = &args[13];
        let down_b = &args[14];

        // 0. Pre-norm.
        let h = mlx_rs::fast::rms_norm(h_raw, pre_norm_w, 1e-6_f32)?;

        // 1. Routing tail.
        let last_axis = (scores.ndim() as i32) - 1;
        let top_logits = scores.take_along_axis(indices, last_axis)?;
        let weights = mlx_rs::ops::softmax_axis(&top_logits, last_axis, Some(true))?;
        let per_expert = mlx_rs::ops::indexing::take_axis(per_expert_scale, indices, 0)?;
        let weights = weights.multiply(&per_expert)?;

        // 2. expand_dims for gather_qmm.
        let x_5d = mlx_rs::ops::expand_dims_axes(&h, &[-2, -3])?;

        // 3. Experts.
        let gqmm = |xi: &Array,
                    w: &Array,
                    s: &Array,
                    b: &Array|
         -> std::result::Result<Array, Exception> {
            mlx_rs::ops::gather_qmm(xi, w, s, b, None, Some(indices), true, 64, 4, false)
        };

        let gate = gqmm(&x_5d, gate_w, gate_s, gate_b)?;
        let up = gqmm(&x_5d, up_w, up_s, up_b)?;

        let dt = gate.dtype();
        let half = Array::from_f32(0.5).as_dtype(dt)?;
        let one = Array::from_f32(1.0).as_dtype(dt)?;
        let c3 = Array::from_f32(0.044715).as_dtype(dt)?;
        let coeff = Array::from_f32(0.7978845608028654_f32).as_dtype(dt)?;
        let x_squared = gate.multiply(&gate)?;
        let x_cubed = x_squared.multiply(&gate)?;
        let inner_add = gate.add(&x_cubed.multiply(&c3)?)?;
        let scaled = coeff.multiply(&inner_add)?;
        let t = tanh(&scaled)?;
        let one_plus_tanh = one.add(&t)?;
        let activated = half
            .multiply(&gate)?
            .multiply(&one_plus_tanh)?
            .multiply(&up)?;

        let down = gqmm(&activated, down_w, down_s, down_b)?;

        // 4. Recombine.
        let per_expert_out = mlx_rs::ops::squeeze_axes(&down, &[-2])?;
        let w_expanded = mlx_rs::ops::expand_dims_axes(&weights, &[-1])?;
        let weighted = mlx_rs::ops::multiply(&per_expert_out, &w_expanded)?;
        let summed = mlx_rs::ops::sum_axis(&weighted, -2, false)?;

        // 5. Post-norm.
        let out = mlx_rs::fast::rms_norm(&summed, post_norm_w, 1e-6_f32)?;

        Ok(vec![out])
    }

    static GEMMA4_PRE_POST_NORM_ROUTING_EXPERTS_SLOT: OnceLock<
        std::sync::Mutex<crate::native_compile_cache::CompiledMultiRefs>,
    > = OnceLock::new();

    fn gemma4_pre_post_norm_routing_experts_fuse_enabled() -> bool {
        // Default OFF — opt-in. Land path: enable, measure, promote if positive.
        std::env::var("LUMEN_GEMMA4_FUSE_NORM_ROUTING_EXPERTS")
            .map(|v| v == "1")
            .unwrap_or(false)
    }

    fn gemma4_pre_post_norm_routing_experts_fused(
        scores: &Array,
        indices: &Array,
        per_expert_scale: &Array,
        h_raw: &Array,
        pre_norm_w: &Array,
        post_norm_w: &Array,
        w: &ResolvedGemma4ExpertsWeights,
    ) -> Result<Array> {
        let gate_b = w.gate_proj.biases.as_ref().ok_or_else(|| {
            anyhow!("gemma4_pre_post_norm_routing_experts_fused: gate_proj.biases is None")
        })?;
        let up_b = w.up_proj.biases.as_ref().ok_or_else(|| {
            anyhow!("gemma4_pre_post_norm_routing_experts_fused: up_proj.biases is None")
        })?;
        let down_b = w.down_proj.biases.as_ref().ok_or_else(|| {
            anyhow!("gemma4_pre_post_norm_routing_experts_fused: down_proj.biases is None")
        })?;
        let args = [
            scores,
            indices,
            per_expert_scale,
            h_raw,
            pre_norm_w,
            post_norm_w,
            &w.gate_proj.weight,
            &w.gate_proj.scales,
            gate_b,
            &w.up_proj.weight,
            &w.up_proj.scales,
            up_b,
            &w.down_proj.weight,
            &w.down_proj.scales,
            down_b,
        ];
        let mut out = crate::native_compile_cache::invoke_compiled_multi_refs(
            &GEMMA4_PRE_POST_NORM_ROUTING_EXPERTS_SLOT,
            gemma4_pre_post_norm_routing_experts_compiled_inner,
            /* shapeless */ true,
            &args,
        )
        .context("gemma4_pre_post_norm_routing_experts_fused: mlx compile dispatch failed")?;
        out.pop()
            .context("gemma4_pre_post_norm_routing_experts_fused: missing output")
    }

    fn gemma4_routing_experts_fused(
        scores: &Array,
        indices: &Array,
        per_expert_scale: &Array,
        x: &Array,
        w: &ResolvedGemma4ExpertsWeights,
    ) -> Result<Array> {
        let gate_b = w
            .gate_proj
            .biases
            .as_ref()
            .ok_or_else(|| anyhow!("gemma4_routing_experts_fused: gate_proj.biases is None"))?;
        let up_b = w
            .up_proj
            .biases
            .as_ref()
            .ok_or_else(|| anyhow!("gemma4_routing_experts_fused: up_proj.biases is None"))?;
        let down_b = w
            .down_proj
            .biases
            .as_ref()
            .ok_or_else(|| anyhow!("gemma4_routing_experts_fused: down_proj.biases is None"))?;
        let args = [
            scores,
            indices,
            per_expert_scale,
            x,
            &w.gate_proj.weight,
            &w.gate_proj.scales,
            gate_b,
            &w.up_proj.weight,
            &w.up_proj.scales,
            up_b,
            &w.down_proj.weight,
            &w.down_proj.scales,
            down_b,
        ];
        let mut out = crate::native_compile_cache::invoke_compiled_multi_refs(
            &GEMMA4_ROUTING_EXPERTS_SLOT,
            gemma4_routing_experts_compiled_inner,
            /* shapeless */ true,
            &args,
        )
        .context("gemma4_routing_experts_fused: mlx compile dispatch failed")?;
        out.pop()
            .context("gemma4_routing_experts_fused: missing output")
    }

    // ───────────────────────── Softcap compile (Tier 3) ────────────────
    //
    // Fuse the Gemma 4 logit softcap chain (multiply + tanh + multiply)
    // into a single mlx compile slot. mlx-lm uses `@partial(mx.compile,
    // shapeless=True) def logit_softcap(softcap, x)` for the same effect.
    //
    // Args (3 arrays): [logits, softcap_inv, softcap]
    // Output: tanh(logits * softcap_inv) * softcap

    fn gemma4_softcap_compiled_inner(
        args: &[Array],
    ) -> Result<Vec<Array>, mlx_rs::error::Exception> {
        let logits = &args[0];
        let softcap_inv = &args[1];
        let softcap = &args[2];
        let scaled = mlx_rs::ops::multiply(logits, softcap_inv)?;
        let t = mlx_rs::ops::tanh(&scaled)?;
        let out = mlx_rs::ops::multiply(&t, softcap)?;
        Ok(vec![out])
    }

    static GEMMA4_SOFTCAP_SLOT: OnceLock<
        std::sync::Mutex<crate::native_compile_cache::CompiledMultiRefs>,
    > = OnceLock::new();

    fn gemma4_softcap_fuse_enabled() -> bool {
        std::env::var("LUMEN_GEMMA4_FUSE_SOFTCAP")
            .map(|v| v != "0")
            .unwrap_or(true)
    }

    fn gemma4_softcap_fused(logits: &Array, softcap_inv: &Array, softcap: &Array) -> Result<Array> {
        let args = [logits, softcap_inv, softcap];
        let mut out = crate::native_compile_cache::invoke_compiled_multi_refs(
            &GEMMA4_SOFTCAP_SLOT,
            gemma4_softcap_compiled_inner,
            /* shapeless */ true,
            &args,
        )
        .context("gemma4_softcap_fused: mlx compile dispatch failed")?;
        out.pop().context("gemma4_softcap_fused: missing output")
    }

    // ───────── Layer epilogue fuse (orthogonal-axes exploration 2026-05-17) ─────
    //
    // Folds the unconditional 4-op tail of `decoder_layer_forward`:
    //   add(h1, h2) → rms_norm(post_ff_w) → add(residual) → multiply(layer_scalar)
    // into a single compile slot. 4 launches → 1 per layer × 48 layers ≈ 192
    // launches/step saved on Gemma 4 26B-A4B.
    //
    // Expected outcome: Tier 2C (async-pipelining absorbed) — dense_mlp norm
    // fuse and pre/post-norm routing+experts fuse both NEUTRAL on the same
    // shape of chain. Run as an evidence point.
    //
    // Args (5 arrays):
    //   [h1, h2, residual, post_ff_w, layer_scalar]
    //   h1, h2, residual: [B, L, hidden]
    //   post_ff_w: [hidden]
    //   layer_scalar: [1]
    fn gemma4_layer_epilogue_compiled_inner(
        args: &[Array],
    ) -> std::result::Result<Vec<Array>, mlx_rs::error::Exception> {
        let h1 = &args[0];
        let h2 = &args[1];
        let residual = &args[2];
        let post_ff_w = &args[3];
        let layer_scalar = &args[4];

        let sum = mlx_rs::ops::add(h1, h2)?;
        let normed = mlx_rs::fast::rms_norm(&sum, post_ff_w, 1e-6_f32)?;
        let with_residual = mlx_rs::ops::add(residual, &normed)?;
        let scaled = mlx_rs::ops::multiply(&with_residual, layer_scalar)?;
        Ok(vec![scaled])
    }

    static GEMMA4_LAYER_EPILOGUE_SLOT: OnceLock<
        std::sync::Mutex<crate::native_compile_cache::CompiledMultiRefs>,
    > = OnceLock::new();

    fn gemma4_layer_epilogue_fuse_enabled() -> bool {
        std::env::var("LUMEN_GEMMA4_FUSE_LAYER_EPILOGUE")
            .map(|v| v == "1")
            .unwrap_or(false)
    }

    fn gemma4_layer_epilogue_fused(
        h1: &Array,
        h2: &Array,
        residual: &Array,
        post_ff_w: &Array,
        layer_scalar: &Array,
    ) -> Result<Array> {
        let args = [h1, h2, residual, post_ff_w, layer_scalar];
        let mut out = crate::native_compile_cache::invoke_compiled_multi_refs(
            &GEMMA4_LAYER_EPILOGUE_SLOT,
            gemma4_layer_epilogue_compiled_inner,
            /* shapeless */ true,
            &args,
        )
        .context("gemma4_layer_epilogue_fused: mlx compile dispatch failed")?;
        out.pop()
            .context("gemma4_layer_epilogue_fused: missing output")
    }

    fn gemma4_experts_fused(
        x: &Array,
        idx: &Array,
        w: &ResolvedGemma4ExpertsWeights,
    ) -> Result<Array> {
        let gate_b = match w.gate_proj.biases.as_ref() {
            Some(b) => b,
            None => return Err(anyhow!("gemma4_experts_fused: gate_proj.biases is None")),
        };
        let up_b = match w.up_proj.biases.as_ref() {
            Some(b) => b,
            None => return Err(anyhow!("gemma4_experts_fused: up_proj.biases is None")),
        };
        let down_b = match w.down_proj.biases.as_ref() {
            Some(b) => b,
            None => return Err(anyhow!("gemma4_experts_fused: down_proj.biases is None")),
        };
        let args = [
            x,
            idx,
            &w.gate_proj.weight,
            &w.gate_proj.scales,
            gate_b,
            &w.up_proj.weight,
            &w.up_proj.scales,
            up_b,
            &w.down_proj.weight,
            &w.down_proj.scales,
            down_b,
        ];
        let mut out = crate::native_compile_cache::invoke_compiled_multi_refs(
            &GEMMA4_EXPERTS_SLOT,
            gemma4_experts_compiled_inner,
            /* shapeless */ true,
            &args,
        )
        .context("gemma4_experts_fused: mlx compile dispatch failed")?;
        out.pop().context("gemma4_experts_fused: missing output")
    }

    // ───────────────────────── config.json parsing ─────────────────────────

    /// Top-level config.json wrapper for `gemma4` (text-only deploy ignores
    /// `vision_config`, `audio_config`, vision/image token ids).
    #[derive(Debug, Clone, Deserialize)]
    pub struct NativeGemma4Config {
        pub model_type: String,
        #[serde(default)]
        pub architectures: Vec<String>,
        #[serde(
            default,
            rename = "eos_token_id",
            deserialize_with = "deserialize_token_ids"
        )]
        pub eos_token_ids: Vec<u32>,
        pub text_config: NativeGemma4TextConfig,
        /// `quantization` and `quantization_config` are duplicates in
        /// lmstudio's MLX shards; we accept either, with `quantization`
        /// taking precedence when both are present.
        #[serde(default)]
        pub quantization: Option<NativeGemma4QuantizationConfig>,
        #[serde(default)]
        pub quantization_config: Option<NativeGemma4QuantizationConfig>,
        #[serde(default)]
        pub tie_word_embeddings: Option<bool>,

        // ── multimodal (image) ──
        // Present on `Gemma4ForConditionalGeneration` checkpoints. All optional
        // so text-only deploys and vision-stripped quantizations still parse.
        #[serde(default)]
        pub vision_config: Option<crate::gemma4_vision::NativeGemma4VisionConfig>,
        /// Placeholder token whose embedding rows get replaced by vision soft
        /// tokens (258880 on 26B-A4B).
        #[serde(default)]
        pub image_token_id: Option<u32>,
        /// `<start_of_image>` / `<end_of_image>` sentinels around each run.
        #[serde(default)]
        pub boi_token_id: Option<u32>,
        #[serde(default)]
        pub eoi_token_id: Option<u32>,
        /// Soft tokens the processor budgets per image (280 on 26B-A4B).
        #[serde(default)]
        pub vision_soft_tokens_per_image: Option<usize>,
    }

    /// `text_config` block — every field the forward path needs.
    ///
    /// Fields absent in 26B-A4B (per-layer input embeddings: 2B/4B-only) are
    /// kept with serde defaults so the same struct handles other Gemma 4
    /// variants once we get there.
    #[derive(Debug, Clone, Deserialize)]
    pub struct NativeGemma4TextConfig {
        pub model_type: String,
        pub hidden_size: usize,
        pub num_hidden_layers: usize,
        pub num_attention_heads: usize,
        pub num_key_value_heads: usize,
        /// Per-layer KV-head count for full attention layers when k_eq_v.
        /// 26B-A4B sets this to 2; absent in smaller variants.
        #[serde(default)]
        pub num_global_key_value_heads: Option<usize>,
        /// Head dim for sliding attention layers (and the default fallback).
        pub head_dim: usize,
        /// Head dim for full attention layers (26B/31B). 0 means "use head_dim".
        #[serde(default)]
        pub global_head_dim: usize,
        pub vocab_size: usize,
        #[serde(default = "default_vocab_size_per_layer_input")]
        pub vocab_size_per_layer_input: usize,
        pub rms_norm_eps: f32,
        pub layer_types: Vec<NativeGemma4LayerType>,
        pub sliding_window: usize,
        #[serde(default = "default_sliding_window_pattern")]
        pub sliding_window_pattern: usize,
        pub max_position_embeddings: usize,

        // RoPE
        pub rope_parameters: NativeGemma4RopeParameters,
        #[serde(default = "default_rope_traditional")]
        pub rope_traditional: bool,
        #[serde(default = "default_partial_rotary_factor")]
        pub partial_rotary_factor: f32,

        // Attention behavior
        #[serde(default)]
        pub attention_bias: bool,
        #[serde(default)]
        pub attention_dropout: f32,
        #[serde(default)]
        pub attention_k_eq_v: bool,

        // MoE
        #[serde(default)]
        pub enable_moe_block: bool,
        #[serde(default)]
        pub num_experts: usize,
        #[serde(default)]
        pub top_k_experts: usize,
        #[serde(default)]
        pub moe_intermediate_size: usize,

        // Dense MLP (always present in 26B; sized via intermediate_size).
        pub intermediate_size: usize,
        #[serde(default)]
        pub use_double_wide_mlp: bool,

        // Per-layer input embedding (2B/4B Gemma 4; 0 for 26B-A4B).
        #[serde(default)]
        pub hidden_size_per_layer_input: usize,
        #[serde(default)]
        pub num_kv_shared_layers: usize,

        // Activation + logit softcap
        #[serde(default = "default_hidden_activation")]
        pub hidden_activation: String,
        pub final_logit_softcapping: f32,

        // Tied embedding → lm_head reuses embed_tokens.
        #[serde(default)]
        pub tie_word_embeddings: bool,

        // Tokens
        #[serde(default)]
        pub pad_token_id: u32,
        #[serde(default = "default_bos_token_id")]
        pub bos_token_id: u32,
        #[serde(
            default,
            rename = "eos_token_id",
            deserialize_with = "deserialize_token_ids"
        )]
        pub eos_token_ids: Vec<u32>,
    }

    /// RoPE parameter block, keyed per attention layer kind. mlx-lm's
    /// `gemma4_text.py` looks up `rope_parameters["sliding_attention"|"full_attention"]`.
    #[derive(Debug, Clone, Deserialize)]
    pub struct NativeGemma4RopeParameters {
        pub full_attention: NativeGemma4RopePerKind,
        pub sliding_attention: NativeGemma4RopePerKind,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct NativeGemma4RopePerKind {
        pub rope_theta: f32,
        #[serde(default = "default_partial_rotary_factor")]
        pub partial_rotary_factor: f32,
        #[serde(default = "default_rope_type")]
        pub rope_type: String,
    }

    /// Layer kind discriminant. mlx-lm config.json string values match
    /// `snake_case` form.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum NativeGemma4LayerType {
        SlidingAttention,
        FullAttention,
    }

    impl NativeGemma4LayerType {
        pub fn is_sliding(self) -> bool {
            matches!(self, NativeGemma4LayerType::SlidingAttention)
        }

        pub fn is_full(self) -> bool {
            matches!(self, NativeGemma4LayerType::FullAttention)
        }
    }

    /// Quantization block — uniform `(group_size, bits, mode)` default plus
    /// per-tensor 8-bit overrides for `mlp.{gate,up,down}_proj` and
    /// `router.proj` (and any future override entries).
    #[derive(Debug, Clone, Deserialize)]
    pub struct NativeGemma4QuantizationConfig {
        pub group_size: usize,
        pub bits: usize,
        pub mode: String,
        #[serde(flatten)]
        pub overrides: BTreeMap<String, NativeGemma4QuantizationOverride>,
    }

    #[derive(Debug, Clone, Deserialize)]
    pub struct NativeGemma4QuantizationOverride {
        pub group_size: usize,
        pub bits: usize,
        /// Optional per-tensor mode override. When absent the override
        /// inherits MODE_AFFINE (Qwen3.6 reference convention: overrides
        /// encode AFFINE exceptions inside an otherwise non-AFFINE model,
        /// e.g. 8-bit AFFINE gate layers inside an MXFP4 model). When
        /// present, must be one of `"affine" | "mxfp4"`.
        #[serde(default)]
        pub mode: Option<String>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(untagged)]
    enum TokenIdValue {
        One(u32),
        Many(Vec<u32>),
    }

    fn deserialize_token_ids<'de, D>(deserializer: D) -> std::result::Result<Vec<u32>, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Option::<TokenIdValue>::deserialize(deserializer)?;
        Ok(match value {
            Some(TokenIdValue::One(id)) => vec![id],
            Some(TokenIdValue::Many(ids)) => ids,
            None => Vec::new(),
        })
    }

    fn default_sliding_window_pattern() -> usize {
        6
    }

    fn default_partial_rotary_factor() -> f32 {
        1.0
    }

    fn default_rope_traditional() -> bool {
        false
    }

    fn default_rope_type() -> String {
        "default".to_string()
    }

    fn default_hidden_activation() -> String {
        "gelu_pytorch_tanh".to_string()
    }

    fn default_bos_token_id() -> u32 {
        2
    }

    fn default_vocab_size_per_layer_input() -> usize {
        0
    }

    impl NativeGemma4Config {
        pub fn load(path: &Path) -> Result<Self> {
            let raw = std::fs::read_to_string(path)
                .map_err(|err| anyhow!("config.json read failed at {}: {err}", path.display()))?;
            let mut cfg: Self = serde_json::from_str(&raw)
                .map_err(|err| anyhow!("config.json parse failed at {}: {err}", path.display()))?;
            // `LUMEN_SLIDING_WINDOW` (desktop CONTEXT card → server) overrides the
            // model's built-in sliding window size. 0 means "no override".
            if let Ok(s) = std::env::var("LUMEN_SLIDING_WINDOW") {
                if let Ok(n) = s.parse::<usize>() {
                    if n > 0 {
                        eprintln!(
                            "[gemma4] sliding_window override via LUMEN_SLIDING_WINDOW: {} → {n}",
                            cfg.text_config.sliding_window
                        );
                        cfg.text_config.sliding_window = n;
                    }
                }
            }
            // `LUMEN_MAX_CTX` caps the maximum position embeddings the model
            // advertises — useful to keep KV cache pool sizing predictable
            // when the model config claims e.g. 128K but the host RAM can't
            // hold it.
            if let Ok(s) = std::env::var("LUMEN_MAX_CTX") {
                if let Ok(n) = s.parse::<usize>() {
                    if n > 0 && n < cfg.text_config.max_position_embeddings {
                        eprintln!(
                            "[gemma4] max_position_embeddings capped via LUMEN_MAX_CTX: {} → {n}",
                            cfg.text_config.max_position_embeddings
                        );
                        cfg.text_config.max_position_embeddings = n;
                    }
                }
            }
            // `LUMEN_GEMMA4_TOP_K` overrides the MoE router's top-k expert
            // count at load time. Quality knob — model was trained at k=8;
            // lowering to k=4 ~halves expert FFN compute per token but may
            // degrade output. Use for A/B measurement; ship only after
            // multi-axis quality eval (HAERAE / KMMLU / GSM8K).
            if let Ok(s) = std::env::var("LUMEN_GEMMA4_TOP_K") {
                if let Ok(n) = s.parse::<usize>() {
                    if n > 0 && n <= cfg.text_config.num_experts {
                        if n != cfg.text_config.top_k_experts {
                            eprintln!(
                                "[gemma4] top_k_experts overridden via LUMEN_GEMMA4_TOP_K: {} → {n}",
                                cfg.text_config.top_k_experts
                            );
                            cfg.text_config.top_k_experts = n;
                        }
                    } else {
                        eprintln!(
                            "[gemma4] LUMEN_GEMMA4_TOP_K={n} ignored (must be 1..={}, got {n})",
                            cfg.text_config.num_experts
                        );
                    }
                }
            }
            Ok(cfg)
        }

        /// Returns whichever of `quantization` / `quantization_config` is
        /// present (preferring the former, which is what lmstudio's MLX
        /// shards reference at runtime).
        pub fn effective_quantization(&self) -> Option<&NativeGemma4QuantizationConfig> {
            self.quantization
                .as_ref()
                .or(self.quantization_config.as_ref())
        }

        /// Validate that this config belongs to the Gemma 4 family with a
        /// reachable text-only forward path.
        pub fn validate_gemma4_family(&self) -> Result<()> {
            if self.model_type != "gemma4" {
                return Err(anyhow!(
                    "expected model_type='gemma4', got '{}'",
                    self.model_type
                ));
            }
            if !self.architectures.is_empty()
                && !self
                    .architectures
                    .iter()
                    .any(|a| a == "Gemma4ForConditionalGeneration")
            {
                return Err(anyhow!(
                    "expected architectures to include 'Gemma4ForConditionalGeneration', got {:?}",
                    self.architectures
                ));
            }
            self.text_config.validate()?;
            if let Some(quant) = self.effective_quantization() {
                if !matches!(quant.mode.as_str(), "affine" | "mxfp4" | "mxfp8" | "nvfp4")
                    || quant.group_size == 0
                {
                    return Err(anyhow!(
                        "quantization default must be mode∈{{affine, mxfp4, mxfp8, nvfp4}} with non-zero group, got mode='{}' bits={} group={}",
                        quant.mode,
                        quant.bits,
                        quant.group_size
                    ));
                }
                if !matches!(quant.bits, 2 | 3 | 4 | 5 | 6 | 8) {
                    return Err(anyhow!(
                        "quantization default bits must be in {{2,3,4,5,6,8}} (mlx-supported), got {}",
                        quant.bits
                    ));
                }
                // (Sanity probe removed: it asserted the specific lmstudio
                // mixed 4/8 packaging, which doesn't hold for in-house
                // conversions via `mlx_lm.convert` — those may keep MLPs at
                // the default bit-width or pick different mixed recipes.
                // Per-key dispatch via `quant_params_for` handles whatever
                // overrides the config actually contains.)
                // Override `group_size` is only required to match the default
                // when the override's mode matches the default mode — when
                // modes differ (e.g. MXFP4 g=32 default with AFFINE g=64 embed
                // override), per-tensor dispatch through `quant_params_for`
                // routes each tensor to a kernel that consumes its own
                // `(group_size, bits, mode)` triple, so cross-mode group_size
                // mismatches are safe. This mirrors Qwen3.5's loader which has
                // no mixed-group-size check at all and ships in production
                // (Qwen3.6 MXFP4 g=32 default + AFFINE g=64 gate overrides).
                let top_mode = quant.mode.as_str();
                for (k, ov) in &quant.overrides {
                    let ov_mode = ov.mode.as_deref().unwrap_or("affine");
                    let modes_match = ov_mode == top_mode;
                    if modes_match && ov.group_size != quant.group_size {
                        return Err(anyhow!(
                            "override '{k}' has group_size={} but default is {} (same mode={top_mode}) — mixed group_size within one mode not supported",
                            ov.group_size,
                            quant.group_size
                        ));
                    }
                    if !matches!(ov.bits, 2 | 3 | 4 | 5 | 6 | 8) {
                        return Err(anyhow!(
                            "override '{k}' bits must be in {{2,3,4,5,6,8}} (mlx-supported), got {}",
                            ov.bits
                        ));
                    }
                }
            }
            Ok(())
        }
    }

    impl NativeGemma4TextConfig {
        pub fn validate(&self) -> Result<()> {
            if self.model_type != "gemma4_text" {
                return Err(anyhow!(
                    "expected text_config.model_type='gemma4_text', got '{}'",
                    self.model_type
                ));
            }
            if self.hidden_size == 0
                || self.num_hidden_layers == 0
                || self.num_attention_heads == 0
                || self.num_key_value_heads == 0
                || self.head_dim == 0
                || self.vocab_size == 0
                || self.intermediate_size == 0
            {
                return Err(anyhow!(
                    "text_config has zero-valued core dims: hidden={} layers={} q_heads={} kv_heads={} head_dim={} vocab={} mlp_inter={}",
                    self.hidden_size,
                    self.num_hidden_layers,
                    self.num_attention_heads,
                    self.num_key_value_heads,
                    self.head_dim,
                    self.vocab_size,
                    self.intermediate_size,
                ));
            }
            if self.layer_types.len() != self.num_hidden_layers {
                return Err(anyhow!(
                    "layer_types length {} != num_hidden_layers {}",
                    self.layer_types.len(),
                    self.num_hidden_layers
                ));
            }
            if self.sliding_window == 0 {
                return Err(anyhow!("sliding_window must be > 0"));
            }
            if self.final_logit_softcapping <= 0.0 {
                return Err(anyhow!(
                    "final_logit_softcapping must be > 0, got {}",
                    self.final_logit_softcapping
                ));
            }
            // 26B-A4B has enable_moe_block=true + num_experts=128 + top_k=8.
            // For now we keep this validator strict so unsupported variants
            // surface early.
            if self.enable_moe_block {
                if self.num_experts == 0 {
                    return Err(anyhow!("enable_moe_block=true but num_experts=0"));
                }
                if self.top_k_experts == 0 || self.top_k_experts > self.num_experts {
                    return Err(anyhow!(
                        "top_k_experts {} invalid against num_experts {}",
                        self.top_k_experts,
                        self.num_experts
                    ));
                }
                if self.moe_intermediate_size == 0 {
                    return Err(anyhow!(
                        "moe_intermediate_size must be > 0 when enable_moe_block=true"
                    ));
                }
            }
            // sliding/full split sanity: every num_hidden_layers/sliding_window_pattern
            // positions should be a full attention layer (5 sliding + 1 full
            // for 26B-A4B's pattern=6).
            let n_full = self.layer_types.iter().filter(|t| t.is_full()).count();
            let expected_full = self.num_hidden_layers / self.sliding_window_pattern;
            if n_full == 0 || n_full > self.num_hidden_layers {
                return Err(anyhow!(
                    "layer_types has {n_full} full_attention entries, expected ~{expected_full} given pattern={}",
                    self.sliding_window_pattern,
                ));
            }
            Ok(())
        }

        /// Resolved head dim for a given layer kind. Sliding layers use
        /// `head_dim`; full layers use `global_head_dim` when set.
        pub fn head_dim_for(&self, kind: NativeGemma4LayerType) -> usize {
            match kind {
                NativeGemma4LayerType::FullAttention if self.global_head_dim != 0 => {
                    self.global_head_dim
                }
                _ => self.head_dim,
            }
        }

        /// Resolved KV-head count for a given layer kind. When
        /// `attention_k_eq_v` is set and the layer is full attention,
        /// `num_global_key_value_heads` (if present) overrides
        /// `num_key_value_heads`.
        pub fn n_kv_heads_for(&self, kind: NativeGemma4LayerType) -> usize {
            match kind {
                NativeGemma4LayerType::FullAttention
                    if self.attention_k_eq_v && self.num_global_key_value_heads.is_some() =>
                {
                    self.num_global_key_value_heads.unwrap()
                }
                _ => self.num_key_value_heads,
            }
        }

        /// Returns true iff this layer should drop the `v_proj` tensor and
        /// reuse `k_proj` as `values` (full attention layers only when
        /// `attention_k_eq_v` is set).
        pub fn use_k_eq_v_for(&self, kind: NativeGemma4LayerType) -> bool {
            self.attention_k_eq_v && kind.is_full()
        }

        /// Resolved RoPE block for a given layer kind.
        pub fn rope_for(&self, kind: NativeGemma4LayerType) -> &NativeGemma4RopePerKind {
            match kind {
                NativeGemma4LayerType::FullAttention => &self.rope_parameters.full_attention,
                NativeGemma4LayerType::SlidingAttention => &self.rope_parameters.sliding_attention,
            }
        }
    }

    // ───────────────────────── safetensors weights ─────────────────────────

    /// Multi-shard safetensors weight bag for Gemma 4 26B-A4B (and family).
    /// Mirrors mlx_lm's `weights = mx.load(path)` dict across all shards in
    /// the model directory.
    ///
    /// Vision/audio keys are stripped at `sanitize()` time so downstream
    /// consumers can assume every surviving key starts with
    /// `language_model.model.*`. (Tied lm_head reuses `embed_tokens` so there
    /// is no separate `lm_head` key.)
    pub struct NativeGemma4Weights {
        tensors: HashMap<String, Array>,
    }

    impl NativeGemma4Weights {
        /// Walk all `*.safetensors` files in `dir`, load each via mlx-rs, and
        /// merge into a single keyed map. Mirrors mlx_lm's
        /// `_get_weights(model_path)`.
        pub fn load_dir(dir: &Path) -> Result<Self> {
            let mut shard_paths = std::fs::read_dir(dir)
                .with_context(|| format!("read_dir({}) failed", dir.display()))?
                .filter_map(|entry| entry.ok().map(|e| e.path()))
                .filter(|p| p.is_file() && p.extension().is_some_and(|ext| ext == "safetensors"))
                .collect::<Vec<_>>();
            shard_paths.sort();
            if shard_paths.is_empty() {
                return Err(anyhow!("no *.safetensors found under {}", dir.display()));
            }
            let mut tensors: HashMap<String, Array> = HashMap::new();
            for shard in shard_paths {
                let map = Array::load_safetensors(&shard).map_err(|err| {
                    anyhow!("load_safetensors({}) failed: {err}", shard.display())
                })?;
                tensors.reserve(map.len());
                for (k, v) in map {
                    if tensors.insert(k.clone(), v).is_some() {
                        return Err(anyhow!(
                            "duplicate weight key `{k}` across safetensors shards in {}",
                            dir.display()
                        ));
                    }
                }
            }
            Ok(Self { tensors })
        }

        /// Drop vision/audio keys. Gemma 4 26B-A4B stores text weights under
        /// `language_model.model.*` already, so no prefix rewriting is needed
        /// — we only remove the multimodal-tower entries that the text-only
        /// forward path will never consume.
        ///
        /// Surfaces a clear error if the checkpoint still ships the legacy
        /// unsplit MoE layout (`.experts.gate_up_proj` / standalone
        /// `.experts.down_proj`) — that needs upstream `Model.sanitize()` to
        /// have been run before saving. The lmstudio-community packaging
        /// already ships the split `experts.switch_glu.{gate,up,down}_proj`
        /// form so this path doesn't fire in production.
        pub fn sanitize(&mut self) -> Result<()> {
            let raw = std::mem::take(&mut self.tensors);
            let mut out: HashMap<String, Array> = HashMap::with_capacity(raw.len());
            let mut dropped = 0usize;
            for (key, value) in raw {
                if Self::is_multimodal_only(&key) {
                    dropped += 1;
                    continue;
                }
                if out.insert(key.clone(), value).is_some() {
                    return Err(anyhow!(
                        "sanitize: duplicate key `{key}` after multimodal strip"
                    ));
                }
            }
            // Surface unsanitized legacy MoE layouts before consumers try to
            // dispatch against the missing split tensors.
            let legacy_unsplit = out
                .keys()
                .find(|k| k.ends_with(".experts.gate_up_proj"))
                .cloned();
            if let Some(k) = legacy_unsplit {
                return Err(anyhow!(
                    "weights still in legacy MoE layout (found `{k}`); convert via mlx-lm's Model.sanitize first"
                ));
            }
            // mlx-lm sanitize also rewrites `.experts.down_proj` →
            // `.experts.switch_glu.down_proj`. Detect the un-rewritten form.
            let legacy_down = out
                .keys()
                .find(|k| k.ends_with(".experts.down_proj.weight") && !k.contains(".switch_glu."))
                .cloned();
            if let Some(k) = legacy_down {
                return Err(anyhow!(
                    "weights still in legacy MoE layout (found `{k}`); convert via mlx-lm's Model.sanitize first"
                ));
            }
            self.tensors = out;
            // dropped is debug-only — surfaced via logs in a future PR.
            let _ = dropped;
            Ok(())
        }

        fn is_multimodal_only(key: &str) -> bool {
            key.starts_with("vision_tower")
                || key.starts_with("embed_vision")
                || key.starts_with("model.visual")
                || key.starts_with("audio_tower")
                || key.starts_with("embed_audio")
        }

        pub fn get(&self, name: &str) -> Option<&Array> {
            self.tensors.get(name)
        }

        /// Raw tensor bag. Read this **before** [`Self::sanitize`] if you need
        /// the `vision_tower.*` / `embed_vision.*` entries it strips.
        pub fn tensors(&self) -> &HashMap<String, Array> {
            &self.tensors
        }

        pub fn require(&self, name: &str) -> Result<&Array> {
            self.tensors
                .get(name)
                .ok_or_else(|| anyhow!("required weight `{name}` not found in safetensors"))
        }

        pub fn len(&self) -> usize {
            self.tensors.len()
        }

        pub fn is_empty(&self) -> bool {
            self.tensors.is_empty()
        }

        pub fn keys(&self) -> impl Iterator<Item = &String> {
            self.tensors.keys()
        }

        /// Verify that every tensor key the forward path will request is
        /// present in the (sanitized) weight bag, given the parsed config.
        ///
        /// Rules:
        ///   - For every decoder layer: 7 norm weights + layer_scalar +
        ///     attention q_proj/k_proj/o_proj (+ v_proj iff not
        ///     `use_k_eq_v_for(kind)`), q_norm, k_norm + dense MLP gate/up/down +
        ///     router proj/scale/per_expert_scale + experts switch_glu
        ///     gate/up/down.
        ///   - Top-level: `embed_tokens.weight` + `model.norm.weight`.
        ///   - Quantized tensors carry `.scales` (and optional `.biases`)
        ///     alongside `.weight`; for the bring-up gate we only require
        ///     `.weight` so the loader works against both quantized and
        ///     unquantized shards.
        pub fn validate_keys_against_config(&self, cfg: &NativeGemma4TextConfig) -> Result<()> {
            let top_level = [
                "language_model.model.embed_tokens.weight",
                "language_model.model.norm.weight",
            ];
            for k in top_level {
                if self.tensors.get(k).is_none() {
                    return Err(anyhow!("missing top-level weight `{k}`"));
                }
            }

            for (layer_idx, layer_kind) in cfg.layer_types.iter().enumerate() {
                let base = format!("language_model.model.layers.{layer_idx}");
                let mut required_keys: Vec<String> = vec![
                    format!("{base}.input_layernorm.weight"),
                    format!("{base}.post_attention_layernorm.weight"),
                    format!("{base}.pre_feedforward_layernorm.weight"),
                    format!("{base}.post_feedforward_layernorm.weight"),
                    format!("{base}.layer_scalar"),
                    // Attention
                    format!("{base}.self_attn.q_proj.weight"),
                    format!("{base}.self_attn.k_proj.weight"),
                    format!("{base}.self_attn.o_proj.weight"),
                    format!("{base}.self_attn.q_norm.weight"),
                    format!("{base}.self_attn.k_norm.weight"),
                    // Dense MLP (every layer)
                    format!("{base}.mlp.gate_proj.weight"),
                    format!("{base}.mlp.up_proj.weight"),
                    format!("{base}.mlp.down_proj.weight"),
                ];
                if !cfg.use_k_eq_v_for(*layer_kind) {
                    required_keys.push(format!("{base}.self_attn.v_proj.weight"));
                }
                if cfg.enable_moe_block {
                    required_keys.extend([
                        format!("{base}.pre_feedforward_layernorm_2.weight"),
                        format!("{base}.post_feedforward_layernorm_1.weight"),
                        format!("{base}.post_feedforward_layernorm_2.weight"),
                        format!("{base}.router.proj.weight"),
                        format!("{base}.router.scale"),
                        format!("{base}.router.per_expert_scale"),
                        format!("{base}.experts.switch_glu.gate_proj.weight"),
                        format!("{base}.experts.switch_glu.up_proj.weight"),
                        format!("{base}.experts.switch_glu.down_proj.weight"),
                    ]);
                }
                for k in &required_keys {
                    if self.tensors.get(k).is_none() {
                        return Err(anyhow!(
                            "missing weight `{k}` (layer {layer_idx}, kind={:?})",
                            layer_kind
                        ));
                    }
                }

                // Spurious v_proj on full-attention layers when k_eq_v is in
                // effect indicates a checkpoint that doesn't match the
                // config's claim — surface early so attention doesn't load a
                // stale tensor.
                if cfg.use_k_eq_v_for(*layer_kind) {
                    let vp = format!("{base}.self_attn.v_proj.weight");
                    if self.tensors.get(&vp).is_some() {
                        return Err(anyhow!(
                            "config declares attention_k_eq_v=true for full attention but `{vp}` is present — checkpoint inconsistent with config"
                        ));
                    }
                }
            }
            Ok(())
        }
    }

    // ───────────────────────── per-layer KV cache ─────────────────────────

    /// Per-layer cache discriminated by the layer's attention kind. Sliding
    /// layers use a `RotatingKvCache(max_size=sliding_window, keep=0)`; full
    /// layers use the existing block-allocated `NativeKvCache`.
    #[derive(Clone)]
    pub enum NativeGemma4LayerCache {
        Sliding(NativeRotatingKvCache),
        Full(NativeKvCache),
        /// Tier 1B (2026-05-16) — 4-bit quantized full-attn cache. Opt-in via
        /// `LUMEN_GEMMA4_QUANT_KV=1`. Stores K/V as quantized 3-tuples; attention
        /// path dispatches to `quantized_matmul` for Q@K^T and scores@V.
        FullQuantized(NativeKvCacheQuantized),
        /// Tier 1B sliding (2026-05-17) — 4-bit quantized rotating-window cache.
        /// Opt-in via `LUMEN_GEMMA4_QUANT_KV_SLIDING=1`. 25 sliding layers × Q4
        /// expected to yield 5× the BW/memory reduction of `FullQuantized`
        /// (which covers only 5 full-attn layers).
        SlidingQuantized(NativeRotatingKvCacheQuantized),
        /// TurboQuant Stage-1 sliding cache (2026-05-17). Lloyd-Max
        /// nearest-centroid against a fixed N(0,1) codebook + Haar rotation
        /// on K (orthogonality preserves inner products). V quantized in
        /// original space so SDPA output → o_proj path is unchanged.
        /// Opt-in via `LUMEN_GEMMA4_QUANT_KV_SLIDING_TURBOQUANT=1`.
        SlidingTurboquant(NativeRotatingKvCacheTurboQuant),
        /// TurboQuant Stage-1 **full-attention** cache (2026-05-26). Same
        /// math as `SlidingTurboquant` but `max_size = max_position_embeddings`
        /// so the rotation never fires under normal operation — behaves as a
        /// step-prealloc append-only buffer. Opt-in via
        /// `LUMEN_GEMMA4_TQ_FULL_ATTN=1` (also requires `LUMEN_GEMMA4_TQ_MODE`
        /// = on or auto). Designed for the long-context BW-bound regime
        /// where full-attn KV grows unbounded — compression's ROI scales
        /// with context length while dispatch overhead stays constant.
        FullTurboquant(NativeRotatingKvCacheTurboQuant),
    }

    impl NativeGemma4LayerCache {
        pub fn offset(&self) -> usize {
            match self {
                NativeGemma4LayerCache::Sliding(c) => c.offset(),
                NativeGemma4LayerCache::Full(c) => c.offset(),
                NativeGemma4LayerCache::FullQuantized(c) => c.offset(),
                NativeGemma4LayerCache::SlidingQuantized(c) => c.offset(),
                NativeGemma4LayerCache::SlidingTurboquant(c) => c.offset(),
                NativeGemma4LayerCache::FullTurboquant(c) => c.offset(),
            }
        }

        pub fn cached_len(&self) -> usize {
            match self {
                NativeGemma4LayerCache::Sliding(c) => c.cached_len(),
                NativeGemma4LayerCache::Full(c) => c.offset(),
                NativeGemma4LayerCache::FullQuantized(c) => c.offset(),
                NativeGemma4LayerCache::SlidingQuantized(c) => c.cached_len(),
                NativeGemma4LayerCache::SlidingTurboquant(c) => c.cached_len(),
                NativeGemma4LayerCache::FullTurboquant(c) => c.cached_len(),
            }
        }

        pub fn empty(&self) -> bool {
            match self {
                NativeGemma4LayerCache::Sliding(c) => c.empty(),
                NativeGemma4LayerCache::Full(c) => c.empty(),
                NativeGemma4LayerCache::FullQuantized(c) => c.empty(),
                NativeGemma4LayerCache::SlidingQuantized(c) => c.empty(),
                NativeGemma4LayerCache::SlidingTurboquant(c) => c.empty(),
                NativeGemma4LayerCache::FullTurboquant(c) => c.empty(),
            }
        }

        pub fn clear(&mut self) {
            match self {
                NativeGemma4LayerCache::Sliding(c) => c.clear(),
                NativeGemma4LayerCache::Full(c) => c.clear(),
                NativeGemma4LayerCache::FullQuantized(c) => c.clear(),
                NativeGemma4LayerCache::SlidingQuantized(c) => c.clear(),
                NativeGemma4LayerCache::SlidingTurboquant(c) => c.clear(),
                NativeGemma4LayerCache::FullTurboquant(c) => c.clear(),
            }
        }

        pub fn as_sliding_mut(&mut self) -> Result<&mut NativeRotatingKvCache> {
            match self {
                NativeGemma4LayerCache::Sliding(c) => Ok(c),
                _ => Err(anyhow!(
                    "NativeGemma4LayerCache: expected Sliding, got non-Sliding"
                )),
            }
        }

        pub fn as_full_mut(&mut self) -> Result<&mut NativeKvCache> {
            match self {
                NativeGemma4LayerCache::Full(c) => Ok(c),
                _ => Err(anyhow!(
                    "NativeGemma4LayerCache: expected Full, got non-Full"
                )),
            }
        }

        pub fn as_full_quantized_mut(&mut self) -> Result<&mut NativeKvCacheQuantized> {
            match self {
                NativeGemma4LayerCache::FullQuantized(c) => Ok(c),
                _ => Err(anyhow!(
                    "NativeGemma4LayerCache: expected FullQuantized, got non-FullQuantized"
                )),
            }
        }

        pub fn as_sliding_quantized_mut(&mut self) -> Result<&mut NativeRotatingKvCacheQuantized> {
            match self {
                NativeGemma4LayerCache::SlidingQuantized(c) => Ok(c),
                _ => Err(anyhow!(
                    "NativeGemma4LayerCache: expected SlidingQuantized, got non-SlidingQuantized"
                )),
            }
        }
    }

    fn gemma4_quant_kv_enabled() -> bool {
        std::env::var("LUMEN_GEMMA4_QUANT_KV")
            .map(|v| v == "1")
            .unwrap_or(false)
    }

    /// Opt-in env gate for the Tier 1B sliding quantized cache. Default OFF
    /// until 8K A/B + 16K A/B + memory delta land.
    fn gemma4_quant_kv_sliding_enabled() -> bool {
        std::env::var("LUMEN_GEMMA4_QUANT_KV_SLIDING")
            .map(|v| v == "1")
            .unwrap_or(false)
    }

    /// TurboQuant K-rotation lever (paper: Google ICLR 2026). Default OFF.
    /// When set, applies a Haar orthogonal R [D,D] to K before quantize and
    /// to Q before quantized_matmul. Orthogonality preserves inner products,
    /// so SDPA scores are mathematically unchanged; the Gaussianized K is
    /// quantized with less per-element error, recovering quality at the same
    /// bit budget OR enabling smaller bit budgets at the same quality.
    /// V stays in original space so the SDPA output → o_proj path doesn't
    /// need an inverse rotation.
    fn gemma4_quant_kv_sliding_rotate_enabled() -> bool {
        std::env::var("LUMEN_GEMMA4_QUANT_KV_SLIDING_ROTATE")
            .map(|v| v == "1")
            .unwrap_or(false)
    }

    /// TurboQuant Stage-1 sliding cache opt-in. Default OFF.
    /// When set, replaces the mlx-affine SlidingQuantized cache with the
    /// TurboQuant Stage-1 path (rotate K + Lloyd-Max nearest-centroid against
    /// fixed N(0,1) codebook + per-vector σ). V uses Lloyd-Max without rotation.
    /// Mutually exclusive with `LUMEN_GEMMA4_QUANT_KV_SLIDING` (TurboQuant
    /// takes precedence when both set).
    ///
    /// Legacy entry point — reads only the static env var (no per-request
    /// adaptive decision). New code should consult [`Gemma4TqMode`] /
    /// [`resolve_tq_for_request`] instead and pass the resolved boolean into
    /// the cache constructor explicitly.
    fn gemma4_quant_kv_sliding_turboquant_enabled() -> bool {
        // For backward compatibility with code paths that don't know the
        // per-request prompt length yet (e.g. `tq_bake_r_enabled` reads at
        // model-load time): treat MODE=on as ON, MODE=auto as ON (so V/Wo
        // bake is still applied — that's harmless when TQ is OFF at runtime
        // because R@R^T = I cancels), and MODE=off as OFF. Legacy env var
        // continues to work.
        match gemma4_tq_mode() {
            Gemma4TqMode::Off => false,
            Gemma4TqMode::On | Gemma4TqMode::Auto => true,
        }
    }

    /// Three-way TurboQuant mode controlled by `LUMEN_GEMMA4_TQ_MODE`.
    ///
    /// * `Off` — never apply TurboQuant; sliding cache stays in bf16.
    /// * `On` — always apply TurboQuant on the sliding cache.
    /// * `Auto` — apply TurboQuant only when the request's `prompt_tokens`
    ///   meets or exceeds `LUMEN_GEMMA4_TQ_AUTO_THRESHOLD_TOKENS` (default 4096).
    ///   Below the threshold, the sliding cache stays in bf16 (full decode
    ///   speed). Above, TQ kicks in to bound memory at long context.
    ///
    /// Backward compatibility: when `LUMEN_GEMMA4_TQ_MODE` is unset, falls
    /// back to the legacy `LUMEN_GEMMA4_QUANT_KV_SLIDING_TURBOQUANT=0/1`
    /// gate so existing deployments keep working unchanged.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Gemma4TqMode {
        Off,
        On,
        Auto,
    }

    pub fn gemma4_tq_mode() -> Gemma4TqMode {
        if let Ok(s) = std::env::var("LUMEN_GEMMA4_TQ_MODE") {
            return match s.trim().to_ascii_lowercase().as_str() {
                "off" | "0" | "false" | "no" => Gemma4TqMode::Off,
                "on" | "1" | "true" | "yes" => Gemma4TqMode::On,
                "auto" => Gemma4TqMode::Auto,
                other => {
                    eprintln!(
                        "[gemma4] WARN unknown LUMEN_GEMMA4_TQ_MODE={other:?}, defaulting to off"
                    );
                    Gemma4TqMode::Off
                }
            };
        }
        // Legacy env var fallback (kept for backward compat).
        if std::env::var("LUMEN_GEMMA4_QUANT_KV_SLIDING_TURBOQUANT")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            Gemma4TqMode::On
        } else {
            Gemma4TqMode::Off
        }
    }

    /// Auto-mode prompt-length threshold (in tokens) at which TQ kicks in.
    /// `LUMEN_GEMMA4_TQ_AUTO_THRESHOLD_TOKENS`, default 4096.
    pub fn gemma4_tq_auto_threshold() -> usize {
        std::env::var("LUMEN_GEMMA4_TQ_AUTO_THRESHOLD_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4096usize)
            .max(1)
    }

    /// Per-request resolution of TQ on/off based on the mode + prompt length.
    /// Returns `true` iff this request should build a `SlidingTurboquant`
    /// cache for its sliding-attention layers.
    pub fn resolve_tq_for_request(prompt_tokens: usize) -> bool {
        match gemma4_tq_mode() {
            Gemma4TqMode::Off => false,
            Gemma4TqMode::On => true,
            Gemma4TqMode::Auto => prompt_tokens >= gemma4_tq_auto_threshold(),
        }
    }

    /// Three-way Simple 4-bit KV-cache mode controlled by
    /// `LUMEN_GEMMA4_QUANT_KV_MODE`.
    ///
    /// * `Off` — never apply Q4; full-attention layers stay in bf16.
    /// * `On` — always apply Q4 on the full-attention KV cache.
    /// * `Auto` — apply Q4 only when the request's `prompt_tokens`
    ///   meets or exceeds `LUMEN_GEMMA4_QUANT_KV_AUTO_THRESHOLD_TOKENS`
    ///   (default 8192). Below the threshold, KV stays in bf16 for full
    ///   decode speed. Above, Q4 compresses 4× to keep memory bounded.
    ///
    /// Backward compatibility: when `LUMEN_GEMMA4_QUANT_KV_MODE` is unset,
    /// falls back to the legacy `LUMEN_GEMMA4_QUANT_KV=0/1` gate.
    ///
    /// Empirical motivation (2026-05-26 sweep): simple Q4 ≈ bf16 in perf
    /// at 8K-32K (within ±5%), so Auto-mode threshold at 8K trades zero
    /// perceived speed for 4× KV memory headroom on long-context requests.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Gemma4QuantKvMode {
        Off,
        On,
        Auto,
    }

    pub fn gemma4_quant_kv_mode() -> Gemma4QuantKvMode {
        if let Ok(s) = std::env::var("LUMEN_GEMMA4_QUANT_KV_MODE") {
            return match s.trim().to_ascii_lowercase().as_str() {
                "off" | "0" | "false" | "no" => Gemma4QuantKvMode::Off,
                "on" | "1" | "true" | "yes" => Gemma4QuantKvMode::On,
                "auto" => Gemma4QuantKvMode::Auto,
                other => {
                    eprintln!(
                        "[gemma4] WARN unknown LUMEN_GEMMA4_QUANT_KV_MODE={other:?}, \
                         defaulting to off"
                    );
                    Gemma4QuantKvMode::Off
                }
            };
        }
        // Legacy env var fallback (kept for backward compat).
        if std::env::var("LUMEN_GEMMA4_QUANT_KV")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            Gemma4QuantKvMode::On
        } else {
            Gemma4QuantKvMode::Off
        }
    }

    /// Auto-mode prompt-length threshold (in tokens) at which Q4 kicks in.
    /// `LUMEN_GEMMA4_QUANT_KV_AUTO_THRESHOLD_TOKENS`, default 16384 (16K).
    ///
    /// Rationale (2026-05-26 3-way sweep on M3 Max): Q4 is ≈ neutral vs bf16
    /// for decode at 8K-32K (within ±5%), so users only see memory benefit
    /// without throughput cost. The default is the 24 GB Mac mini target —
    /// 16K is where bf16 KV pressure starts to bind and quantized sliding-
    /// window wins are verified. (A 128K default made Auto a near no-op.)
    /// This fallback only applies for standalone runs; the desktop app emits
    /// the env var from `kv_auto_threshold_tokens` (default 16384).
    pub fn gemma4_quant_kv_auto_threshold() -> usize {
        std::env::var("LUMEN_GEMMA4_QUANT_KV_AUTO_THRESHOLD_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(16384usize)
            .max(1)
    }

    /// Bit width for the simple Q4 cache (`NativeKvCacheQuantized`).
    /// `LUMEN_GEMMA4_QUANT_KV_BITS`, accepts 3 / 4 / 6 / 8. Default 4.
    /// Trade-off: more bits = less quant noise = closer to bf16 quality at
    /// the cost of smaller memory savings. 4-bit gives 4× compression, 8-bit
    /// gives 2×. Invalid values fall back to 4 with a warning.
    pub fn gemma4_quant_kv_bits() -> u32 {
        let raw = std::env::var("LUMEN_GEMMA4_QUANT_KV_BITS")
            .ok()
            .and_then(|s| s.parse::<u32>().ok());
        match raw {
            Some(b) if b == 3 || b == 4 || b == 6 || b == 8 => b,
            Some(b) => {
                eprintln!(
                    "[gemma4] WARN LUMEN_GEMMA4_QUANT_KV_BITS={b} not in {{3,4,6,8}}, \
                     defaulting to 4"
                );
                4
            }
            None => 4,
        }
    }

    /// Per-request resolution of simple Q4 on/off based on mode + prompt length.
    /// Returns `true` iff this request should build a `NativeKvCacheQuantized`
    /// cache for its full-attention layers.
    pub fn resolve_quant_kv_for_request(prompt_tokens: usize) -> bool {
        match gemma4_quant_kv_mode() {
            Gemma4QuantKvMode::Off => false,
            Gemma4QuantKvMode::On => true,
            Gemma4QuantKvMode::Auto => prompt_tokens >= gemma4_quant_kv_auto_threshold(),
        }
    }

    /// Apply TurboQuant Stage-1 quantization to the **full-attention** layers
    /// (5 of 30 in Gemma 4 26B-A4B) in addition to / instead of the sliding
    /// layers. Default OFF. Independent of [`Gemma4TqMode`] —
    /// `LUMEN_GEMMA4_TQ_FULL_ATTN=1` opts in regardless of whether sliding
    /// TQ is on, because full-attn TQ has a different ROI profile (BW
    /// scales with context, sliding KV is BW-capped at the window size).
    ///
    /// Allocation honors the `tq_mode` gate — if mode is Off, no TQ at
    /// any layer kind. If mode is On / Auto-triggered, this env decides
    /// whether full-attn layers also get the TurboQuant cache.
    pub fn gemma4_tq_full_attn_enabled() -> bool {
        std::env::var("LUMEN_GEMMA4_TQ_FULL_ATTN")
            .map(|v| v == "1")
            .unwrap_or(false)
    }

    /// QJL Stage-2 (1-bit residual correction) on top of the TurboQuant
    /// Stage-1 sliding cache. Requires
    /// `LUMEN_GEMMA4_QUANT_KV_SLIDING_TURBOQUANT=1` to take effect.
    /// Default OFF.
    fn gemma4_quant_kv_sliding_turboquant_qjl_enabled() -> bool {
        std::env::var("LUMEN_GEMMA4_QUANT_KV_SLIDING_TURBOQUANT_QJL")
            .map(|v| v == "1")
            .unwrap_or(false)
    }

    /// QJL projection width m for the Stage-2 estimator.
    ///
    /// Theory (verified empirically — see turboquant.rs unit tests):
    /// the unbiased estimator has per-element K-reconstruction MSE of
    /// `‖r‖² · (π/2)/m`, while Stage-1's per-element MSE is `‖r‖²/D`.
    /// QJL only beats Stage 1 once `m > D · π/2`. For Gemma 4 D=256 the
    /// threshold is ~402; m=128 is variance-dominated and *hurts*
    /// reconstruction. We default to `D·4 = 1024` so Stage 2 is in the
    /// useful regime; users can override via env.
    ///
    /// Storage cost at the current unpacked-bf16-±1 layout: per K vector
    /// 2 B·m. At m=1024 that's 2 KB/K-vector × 1024 tokens × 4 n_kv ×
    /// 25 sliding layers ≈ 200 MB — substantial but fits on a 24 GB Mac.
    /// A future packed-bit kernel would drop this 16×.
    fn gemma4_quant_kv_sliding_turboquant_qjl_m(head_dim: usize) -> usize {
        std::env::var("LUMEN_GEMMA4_QUANT_KV_SLIDING_TURBOQUANT_QJL_M")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(head_dim * 4)
            .min(2048)
            .max(1)
    }

    /// Full-model cache — one slot per decoder layer, allocated based on the
    /// config's `layer_types` (mirrors mlx_lm `Model.make_cache`).
    #[derive(Clone)]
    pub struct NativeGemma4PromptCache {
        layers: Vec<NativeGemma4LayerCache>,
    }

    /// Read a present Array into the disk-record blob, returning its index;
    /// `None` for an absent slot. (P2 Gemma4 disk persistence helper.)
    fn push_record(records: &mut Vec<ArrayRecord>, a: Option<&Array>) -> Result<Option<usize>> {
        match a {
            None => Ok(None),
            Some(arr) => {
                let rec = record_from_array(arr)?;
                let id = records.len();
                records.push(rec);
                Ok(Some(id))
            }
        }
    }

    impl NativeGemma4PromptCache {
        /// Allocate caches for every layer of `cfg`. mlx_lm semantics:
        ///   layer_type == "full_attention"    → KVCache()
        ///   layer_type == "sliding_attention" → RotatingKVCache(max_size=sliding_window, keep=0)
        ///
        /// 26B-A4B sets `num_kv_shared_layers=0`, so every layer gets its own
        /// cache. Variants with shared KVs (2B/4B) would skip the trailing
        /// `num_kv_shared_layers` slots — deferred until we extend support.
        /// Allocate caches reading TurboQuant on/off from the static env var
        /// only (no per-request adaptive). Equivalent to
        /// `for_config_with_tq(cfg, None, None)`. Kept as the legacy entry for
        /// callers that don't yet know the prompt length.
        pub fn for_config(cfg: &NativeGemma4TextConfig) -> Self {
            Self::for_config_with_tq(cfg, None, None)
        }

        /// P2 disk persistence — serialize this whole-cache snapshot into
        /// `(per-layer metadata, flat ArrayRecord blob)` reusing the shared
        /// `kv_disk` LKV1 primitives. Covers all live layer kinds: dense `Full`
        /// / `Sliding`, affine-quantized `FullQuantized` / `SlidingQuantized`
        /// (K/V 3-tuples), and TurboQuant `FullTurboquant` / `SlidingTurboquant`.
        pub fn to_disk_records(&self) -> Result<(Vec<LayerMeta>, Vec<ArrayRecord>)> {
            let mut records: Vec<ArrayRecord> = Vec::new();
            let mut metas: Vec<LayerMeta> = Vec::with_capacity(self.layers.len());

            // Push a quantized `(packed, scales, biases)` 3-tuple, returning the
            // three record slot ids (or `None`s when the tuple is absent).
            let push_q3 = |recs: &mut Vec<ArrayRecord>,
                           t: Option<&(Array, Array, Array)>|
             -> Result<Vec<Option<usize>>> {
                match t {
                    Some((p, s, b)) => Ok(vec![
                        push_record(recs, Some(p))?,
                        push_record(recs, Some(s))?,
                        push_record(recs, Some(b))?,
                    ]),
                    None => Ok(vec![None, None, None]),
                }
            };
            // Serialize a TurboQuant rotating cache (shared by Full/Sliding TQ).
            let tq_records = |recs: &mut Vec<ArrayRecord>,
                              c: &NativeRotatingKvCacheTurboQuant,
                              kind: LayerKindTag|
             -> Result<LayerMeta> {
                let arrays = vec![
                    push_record(recs, c.keys_codes())?,
                    push_record(recs, c.keys_sigma())?,
                    push_record(recs, c.values_codes())?,
                    push_record(recs, c.values_sigma())?,
                    push_record(recs, c.keys_signs())?,
                    push_record(recs, c.keys_residual_norm())?,
                ];
                let mut m = LayerMeta::new(kind, c.offset());
                m.arrays = arrays;
                m.max_size = Some(c.max_size());
                m.keep = Some(c.keep());
                m.idx = Some(c.idx());
                m.bits = Some(c.bits());
                m.qjl_m = c.qjl_m();
                Ok(m)
            };

            for layer in &self.layers {
                match layer {
                    NativeGemma4LayerCache::Full(c) => {
                        // Logical view (`[.., :offset, ..]`) — smaller than the
                        // block-prealloc backing; reconstructed via `set_state`.
                        let kid = push_record(&mut records, c.keys_view()?.as_ref())?;
                        let vid = push_record(&mut records, c.values_view()?.as_ref())?;
                        let mut m = LayerMeta::new(LayerKindTag::Full, c.offset());
                        m.arrays = vec![kid, vid];
                        metas.push(m);
                    }
                    NativeGemma4LayerCache::Sliding(c) => {
                        // Full backing ring (NOT a view) so circular reads
                        // reproduce; `idx`/`offset` restore the ring position.
                        let kid = push_record(&mut records, c.keys())?;
                        let vid = push_record(&mut records, c.values())?;
                        let mut m = LayerMeta::new(LayerKindTag::Sliding, c.offset());
                        m.arrays = vec![kid, vid];
                        m.max_size = Some(c.max_size());
                        m.keep = Some(c.keep());
                        m.idx = Some(c.idx());
                        metas.push(m);
                    }
                    NativeGemma4LayerCache::FullQuantized(c) => {
                        let mut arrays = push_q3(&mut records, c.keys())?;
                        arrays.extend(push_q3(&mut records, c.values())?);
                        let mut m = LayerMeta::new(LayerKindTag::FullQuantized, c.offset());
                        m.arrays = arrays;
                        m.group_size = Some(c.group_size());
                        m.bits = Some(c.bits() as u32);
                        metas.push(m);
                    }
                    NativeGemma4LayerCache::SlidingQuantized(c) => {
                        let mut arrays = push_q3(&mut records, c.keys())?;
                        arrays.extend(push_q3(&mut records, c.values())?);
                        let mut m = LayerMeta::new(LayerKindTag::SlidingQuantized, c.offset());
                        m.arrays = arrays;
                        m.max_size = Some(c.max_size());
                        m.keep = Some(c.keep());
                        m.idx = Some(c.idx());
                        m.group_size = Some(c.group_size());
                        m.bits = Some(c.bits() as u32);
                        metas.push(m);
                    }
                    NativeGemma4LayerCache::SlidingTurboquant(c) => {
                        metas.push(tq_records(
                            &mut records,
                            c,
                            LayerKindTag::SlidingTurboquant,
                        )?);
                    }
                    NativeGemma4LayerCache::FullTurboquant(c) => {
                        metas.push(tq_records(&mut records, c, LayerKindTag::FullTurboquant)?);
                    }
                }
            }
            Ok((metas, records))
        }

        /// Inverse of [`Self::to_disk_records`]. Rebuilds the layer vector from
        /// the persisted metadata + record blob. Layer kinds + geometry come
        /// from `metas` (written by the same model — guarded by the disk-store
        /// fingerprint), so no model config is needed here.
        pub fn from_disk_records(metas: &[LayerMeta], records: &[ArrayRecord]) -> Result<Self> {
            let get = |id: Option<usize>| -> Result<Option<Array>> {
                match id {
                    None => Ok(None),
                    Some(i) => {
                        let rec = records.get(i).ok_or_else(|| {
                            anyhow!(
                                "kv_disk: Gemma4 from_disk_records record index {i} out of range"
                            )
                        })?;
                        Ok(Some(record_to_array(rec)?))
                    }
                }
            };
            // Reconstruct a quantized `(packed, scales, biases)` 3-tuple from
            // three consecutive `m.arrays` slots starting at `base`.
            let get3 = |m: &LayerMeta, base: usize| -> Result<Option<(Array, Array, Array)>> {
                let p = get(m.arrays.get(base).copied().flatten())?;
                let s = get(m.arrays.get(base + 1).copied().flatten())?;
                let b = get(m.arrays.get(base + 2).copied().flatten())?;
                match (p, s, b) {
                    (Some(p), Some(s), Some(b)) => Ok(Some((p, s, b))),
                    (None, None, None) => Ok(None),
                    _ => Err(anyhow!(
                        "kv_disk: Gemma4 quantized 3-tuple partially present"
                    )),
                }
            };
            // Reconstruct a TurboQuant rotating cache (shared Full/Sliding TQ).
            let tq_from = |m: &LayerMeta| -> Result<NativeRotatingKvCacheTurboQuant> {
                let slot = |i: usize| get(m.arrays.get(i).copied().flatten());
                let max_size = m
                    .max_size
                    .ok_or_else(|| anyhow!("kv_disk: Gemma4 TQ layer missing max_size"))?;
                let keep = m
                    .keep
                    .ok_or_else(|| anyhow!("kv_disk: Gemma4 TQ layer missing keep"))?;
                let idx = m
                    .idx
                    .ok_or_else(|| anyhow!("kv_disk: Gemma4 TQ layer missing idx"))?;
                let bits = m
                    .bits
                    .ok_or_else(|| anyhow!("kv_disk: Gemma4 TQ layer missing bits"))?;
                Ok(NativeRotatingKvCacheTurboQuant::from_parts(
                    slot(0)?,
                    slot(1)?,
                    slot(2)?,
                    slot(3)?,
                    slot(4)?,
                    slot(5)?,
                    m.offset,
                    max_size,
                    keep,
                    idx,
                    bits,
                    m.qjl_m,
                ))
            };

            let mut layers = Vec::with_capacity(metas.len());
            for m in metas {
                match m.kind {
                    LayerKindTag::Full => {
                        let mut c = NativeKvCache::new();
                        c.set_state(
                            get(m.arrays.first().copied().flatten())?,
                            get(m.arrays.get(1).copied().flatten())?,
                            m.offset,
                        );
                        layers.push(NativeGemma4LayerCache::Full(c));
                    }
                    LayerKindTag::Sliding => {
                        let keys = get(m.arrays.first().copied().flatten())?;
                        let values = get(m.arrays.get(1).copied().flatten())?;
                        let max_size = m.max_size.ok_or_else(|| {
                            anyhow!("kv_disk: Gemma4 Sliding layer missing max_size")
                        })?;
                        let keep = m
                            .keep
                            .ok_or_else(|| anyhow!("kv_disk: Gemma4 Sliding layer missing keep"))?;
                        let idx = m
                            .idx
                            .ok_or_else(|| anyhow!("kv_disk: Gemma4 Sliding layer missing idx"))?;
                        // The record codec now reads the full ring buffer in
                        // correct logical order (reshape-flat fix), so install it
                        // verbatim with its rotation bookkeeping — handles both
                        // pre-rotation (offset<=max_size) and rotated rings
                        // (offset>max_size, long system prompts).
                        layers.push(NativeGemma4LayerCache::Sliding(
                            NativeRotatingKvCache::from_parts(
                                keys, values, m.offset, max_size, keep, idx,
                            ),
                        ));
                    }
                    LayerKindTag::FullQuantized => {
                        let group_size = m.group_size.ok_or_else(|| {
                            anyhow!("kv_disk: Gemma4 FullQuantized layer missing group_size")
                        })?;
                        let bits = m.bits.ok_or_else(|| {
                            anyhow!("kv_disk: Gemma4 FullQuantized layer missing bits")
                        })? as i32;
                        layers.push(NativeGemma4LayerCache::FullQuantized(
                            NativeKvCacheQuantized::from_parts(
                                get3(m, 0)?,
                                get3(m, 3)?,
                                m.offset,
                                group_size,
                                bits,
                            ),
                        ));
                    }
                    LayerKindTag::SlidingQuantized => {
                        let max_size = m.max_size.ok_or_else(|| {
                            anyhow!("kv_disk: Gemma4 SlidingQuantized layer missing max_size")
                        })?;
                        let keep = m.keep.ok_or_else(|| {
                            anyhow!("kv_disk: Gemma4 SlidingQuantized layer missing keep")
                        })?;
                        let idx = m.idx.ok_or_else(|| {
                            anyhow!("kv_disk: Gemma4 SlidingQuantized layer missing idx")
                        })?;
                        let group_size = m.group_size.ok_or_else(|| {
                            anyhow!("kv_disk: Gemma4 SlidingQuantized layer missing group_size")
                        })?;
                        let bits = m.bits.ok_or_else(|| {
                            anyhow!("kv_disk: Gemma4 SlidingQuantized layer missing bits")
                        })? as i32;
                        layers.push(NativeGemma4LayerCache::SlidingQuantized(
                            NativeRotatingKvCacheQuantized::from_parts(
                                get3(m, 0)?,
                                get3(m, 3)?,
                                m.offset,
                                max_size,
                                keep,
                                idx,
                                group_size,
                                bits,
                            ),
                        ));
                    }
                    LayerKindTag::SlidingTurboquant => {
                        layers.push(NativeGemma4LayerCache::SlidingTurboquant(tq_from(m)?));
                    }
                    LayerKindTag::FullTurboquant => {
                        layers.push(NativeGemma4LayerCache::FullTurboquant(tq_from(m)?));
                    }
                    other => {
                        return Err(anyhow!(
                            "kv_disk: Gemma4 from_disk_records unsupported layer kind {other:?}"
                        ));
                    }
                }
            }
            Ok(Self { layers })
        }

        /// Allocate caches with explicit TurboQuant and simple-Q4 on/off
        /// decisions. `force_tq=Some(true|false)` overrides the TQ env gate;
        /// `force_quant_kv=Some(...)` overrides the simple-Q4 env gate.
        /// `None` for either falls back to the env (legacy behaviour, which
        /// for Auto modes without prompt info defaults to Off — conservative).
        ///
        /// Per-request adaptive callers (see [`resolve_tq_for_request`] /
        /// [`resolve_quant_kv_for_request`]) should always pass `Some(_)`
        /// so the decision reflects this request's prompt length.
        pub fn for_config_with_tq(
            cfg: &NativeGemma4TextConfig,
            force_tq: Option<bool>,
            force_quant_kv: Option<bool>,
        ) -> Self {
            assert_eq!(
                cfg.num_kv_shared_layers, 0,
                "NativeGemma4PromptCache: num_kv_shared_layers > 0 not yet supported (26B-A4B uses 0)"
            );
            // Simple Q4 decision: caller's explicit override wins. Without
            // override, read mode env (Auto without prompt-length info →
            // Off, same conservative fallback as TQ).
            let quant_kv = force_quant_kv.unwrap_or_else(|| match gemma4_quant_kv_mode() {
                Gemma4QuantKvMode::Off => false,
                Gemma4QuantKvMode::On => true,
                Gemma4QuantKvMode::Auto => false,
            });
            let quant_kv_sliding = gemma4_quant_kv_sliding_enabled();
            // TQ decision: caller's explicit override wins. Otherwise read
            // legacy env (treating Auto without prompt info as Off — conservative).
            let quant_kv_sliding_tq = force_tq.unwrap_or_else(|| {
                match gemma4_tq_mode() {
                    Gemma4TqMode::Off => false,
                    Gemma4TqMode::On => true,
                    // Auto without prompt-length info → default OFF (fast path).
                    Gemma4TqMode::Auto => false,
                }
            });
            let tq_full_attn = quant_kv_sliding_tq && gemma4_tq_full_attn_enabled();
            let layers: Vec<NativeGemma4LayerCache> = cfg
                .layer_types
                .iter()
                .map(|kind| match kind {
                    NativeGemma4LayerType::FullAttention => {
                        if tq_full_attn {
                            // Full-attn TurboQuant: same Lloyd-Max + rotation
                            // math as sliding, but with `max_size` set to the
                            // configured context cap so the ring rotation
                            // never fires (acts as a step-prealloc append-
                            // only buffer). The 5 full-attn layers are the
                            // BW-bound target — KV grows linearly with
                            // context, so compression's ROI scales while
                            // dispatch overhead is constant.
                            let bits: u32 =
                                std::env::var("LUMEN_GEMMA4_QUANT_KV_SLIDING_TURBOQUANT_BITS")
                                    .ok()
                                    .and_then(|s| s.parse().ok())
                                    .unwrap_or(4);
                            let max_ctx = cfg.max_position_embeddings.max(1);
                            let cache = NativeRotatingKvCacheTurboQuant::new(max_ctx, 0, bits);
                            let cache = if gemma4_quant_kv_sliding_turboquant_qjl_enabled() {
                                let qjl_m = gemma4_quant_kv_sliding_turboquant_qjl_m(cfg.head_dim);
                                cache.with_qjl(qjl_m)
                            } else {
                                cache
                            };
                            NativeGemma4LayerCache::FullTurboquant(cache)
                        } else if quant_kv {
                            // Bits driven by LUMEN_GEMMA4_QUANT_KV_BITS env
                            // (3 / 4 / 6 / 8, default 4). group_size=64
                            // matches the original FullQuantized layout.
                            let bits = gemma4_quant_kv_bits() as i32;
                            NativeGemma4LayerCache::FullQuantized(NativeKvCacheQuantized::new(
                                64, bits,
                            ))
                        } else {
                            NativeGemma4LayerCache::Full(NativeKvCache::new())
                        }
                    }
                    NativeGemma4LayerType::SlidingAttention => {
                        if quant_kv_sliding_tq {
                            // TurboQuant Stage-1: bits=4 default, tunable via
                            // LUMEN_GEMMA4_QUANT_KV_SLIDING_TURBOQUANT_BITS.
                            let bits: u32 =
                                std::env::var("LUMEN_GEMMA4_QUANT_KV_SLIDING_TURBOQUANT_BITS")
                                    .ok()
                                    .and_then(|s| s.parse().ok())
                                    .unwrap_or(4);
                            let cache =
                                NativeRotatingKvCacheTurboQuant::new(cfg.sliding_window, 0, bits);
                            let cache = if gemma4_quant_kv_sliding_turboquant_qjl_enabled() {
                                let qjl_m = gemma4_quant_kv_sliding_turboquant_qjl_m(cfg.head_dim);
                                cache.with_qjl(qjl_m)
                            } else {
                                cache
                            };
                            NativeGemma4LayerCache::SlidingTurboquant(cache)
                        } else if quant_kv_sliding {
                            // Quality/memory tradeoff knobs (env-tunable):
                            //   GROUP_SIZE: 64 (matches FullQuantized) / 32 / 16
                            //     — smaller groups = finer dequant = less noise.
                            //   BITS: 4 (max memory) / 8 (max quality).
                            //
                            // Default 4-bit + group_size=32. Rationale: 25
                            // sliding layers × 4-bit noise at group_size=64
                            // accumulates and degrades quality (smoke 128 ctx
                            // 2026-05-17: degenerated to 238526×N attractor by
                            // step ~7). Halving group_size halves the per-row
                            // dequant noise floor, preserving 4-bit's memory
                            // win while restoring quality.
                            let bits: i32 = std::env::var("LUMEN_GEMMA4_QUANT_KV_SLIDING_BITS")
                                .ok()
                                .and_then(|s| s.parse().ok())
                                .unwrap_or(4);
                            let group_size: i32 =
                                std::env::var("LUMEN_GEMMA4_QUANT_KV_SLIDING_GROUP_SIZE")
                                    .ok()
                                    .and_then(|s| s.parse().ok())
                                    .unwrap_or(32);
                            NativeGemma4LayerCache::SlidingQuantized(
                                NativeRotatingKvCacheQuantized::new(
                                    cfg.sliding_window,
                                    0,
                                    group_size,
                                    bits,
                                ),
                            )
                        } else {
                            NativeGemma4LayerCache::Sliding(NativeRotatingKvCache::new(
                                cfg.sliding_window,
                                0,
                            ))
                        }
                    }
                })
                .collect();
            Self { layers }
        }

        pub fn len(&self) -> usize {
            self.layers.len()
        }

        /// MTP / prefix-cache rollback hook — truncate this cache's offset
        /// to `target`. Quantized caches dispatch to their own `truncate_to`
        /// (step-prealloc only — see [`NativeKvCacheQuantized::truncate_to`]
        /// for the legacy-concat constraint). Lossy for any rotating cache
        /// if the rolled-back range crossed a sliding-window rotation.
        pub fn truncate_to(&mut self, target: usize) -> Result<()> {
            for (i, lc) in self.layers.iter_mut().enumerate() {
                match lc {
                    NativeGemma4LayerCache::Sliding(c) => c
                        .truncate_to(target)
                        .with_context(|| format!("layer {i} sliding truncate"))?,
                    NativeGemma4LayerCache::Full(c) => c
                        .truncate_to(target)
                        .with_context(|| format!("layer {i} full truncate"))?,
                    NativeGemma4LayerCache::FullQuantized(c) => c
                        .truncate_to(target)
                        .with_context(|| format!("layer {i} full quantized truncate"))?,
                    NativeGemma4LayerCache::SlidingQuantized(c) => c
                        .truncate_to(target)
                        .with_context(|| format!("layer {i} sliding quantized truncate"))?,
                    NativeGemma4LayerCache::SlidingTurboquant(c) => c
                        .truncate_to(target)
                        .with_context(|| format!("layer {i} sliding turboquant truncate"))?,
                    NativeGemma4LayerCache::FullTurboquant(c) => c
                        .truncate_to(target)
                        .with_context(|| format!("layer {i} full turboquant truncate"))?,
                }
            }
            Ok(())
        }

        /// Logical offset of the first layer's cache (same across all
        /// layers in normal operation). Convenience for MTP step accounting.
        pub fn offset(&self) -> usize {
            self.layers.first().map(|lc| lc.offset()).unwrap_or(0)
        }

        /// MTP hook: retrieve K/V from a specific layer's bf16 cache view,
        /// sliced to the logical valid range `[0..offset)` (NOT the full
        /// pre-allocated buffer — the trunk's `update_and_fetch` does the
        /// same logical-length slice for its own SDPA path).
        ///
        /// Both returned arrays have shape `[B, n_kv_heads, T, head_dim]`
        /// where `T = cache.offset()`.
        ///
        /// Errors on quantized caches (Phase 4 dequant fallback).
        /// Errors on rotated sliding caches (`offset > max_size`) — Phase 4
        /// will add proper ring→linear conversion. MTP currently requires
        /// `offset <= sliding_window` (no rotation yet).
        pub fn layer_kv_bf16(&self, layer_idx: usize) -> Result<(Array, Array)> {
            use mlx_rs::ops::indexing::TryIndexOp;
            let lc = self
                .layers
                .get(layer_idx)
                .ok_or_else(|| anyhow!("layer_kv_bf16: layer_idx={layer_idx} out of range"))?;
            match lc {
                NativeGemma4LayerCache::Sliding(c) => {
                    let off = c.offset() as i32;
                    if (off as usize) > c.max_size() {
                        return Err(anyhow!(
                            "layer_kv_bf16: sliding layer {layer_idx} is rotated \
                             (offset={off} > max_size={}); Phase 4 ring rebuild needed",
                            c.max_size()
                        ));
                    }
                    let k_buf = c
                        .keys()
                        .ok_or_else(|| anyhow!("layer_kv_bf16: sliding {layer_idx} no keys"))?;
                    let v_buf = c
                        .values()
                        .ok_or_else(|| anyhow!("layer_kv_bf16: sliding {layer_idx} no values"))?;
                    let k = k_buf
                        .try_index((.., .., 0..off, ..))
                        .context("layer_kv_bf16: slice sliding K")?;
                    let v = v_buf
                        .try_index((.., .., 0..off, ..))
                        .context("layer_kv_bf16: slice sliding V")?;
                    Ok((k, v))
                }
                NativeGemma4LayerCache::Full(c) => {
                    let off = c.offset() as i32;
                    let k_buf = c
                        .keys()
                        .ok_or_else(|| anyhow!("layer_kv_bf16: full {layer_idx} no keys"))?;
                    let v_buf = c
                        .values()
                        .ok_or_else(|| anyhow!("layer_kv_bf16: full {layer_idx} no values"))?;
                    let k = k_buf
                        .try_index((.., .., 0..off, ..))
                        .context("layer_kv_bf16: slice full K")?;
                    let v = v_buf
                        .try_index((.., .., 0..off, ..))
                        .context("layer_kv_bf16: slice full V")?;
                    Ok((k, v))
                }
                NativeGemma4LayerCache::FullQuantized(_)
                | NativeGemma4LayerCache::SlidingQuantized(_)
                | NativeGemma4LayerCache::SlidingTurboquant(_)
                | NativeGemma4LayerCache::FullTurboquant(_) => Err(anyhow!(
                    "layer_kv_bf16: layer {layer_idx} cache is quantized; MTP currently \
                     requires bf16 caches (no quant). Disable LUMEN_GEMMA4_QUANT_KV* or wait \
                     for the Phase 4 dequant fallback."
                )),
            }
        }

        pub fn is_empty(&self) -> bool {
            self.layers.is_empty()
        }

        pub fn layer(&self, idx: usize) -> Option<&NativeGemma4LayerCache> {
            self.layers.get(idx)
        }

        pub fn layer_mut(&mut self, idx: usize) -> Option<&mut NativeGemma4LayerCache> {
            self.layers.get_mut(idx)
        }

        pub fn layers(&self) -> &[NativeGemma4LayerCache] {
            &self.layers
        }

        pub fn layers_mut(&mut self) -> &mut [NativeGemma4LayerCache] {
            &mut self.layers
        }

        pub fn clear(&mut self) {
            for layer in &mut self.layers {
                layer.clear();
            }
        }
    }

    // ───────────────────────── attention-mask routing ─────────────────────────

    /// Build the right additive attention mask for a Gemma 4 layer given its
    /// kind and the current KV cache state.
    ///
    /// Returns `None` when no mask is needed (mlx_lm's `create_attention_mask`
    /// semantics): decode (`query_len == 1`) skips the mask because SDPA naturally
    /// attends to all cached keys.
    ///
    /// For sliding-attention layers, the window cutoff is applied so queries
    /// only see the last `sliding_window` keys.
    pub fn make_attention_mask_for_layer(
        kind: NativeGemma4LayerType,
        cfg: &NativeGemma4TextConfig,
        query_len: usize,
        kv_offset: usize,
    ) -> Result<Option<Array>> {
        if query_len <= 1 {
            // Decode: SDPA against `[B, n_h, 1, D]` query needs no mask.
            return Ok(None);
        }
        let window = match kind {
            NativeGemma4LayerType::SlidingAttention => Some(cfg.sliding_window),
            NativeGemma4LayerType::FullAttention => None,
        };
        build_causal_mask(query_len, kv_offset, window)
    }

    /// Chunked-prefill / rotated-cache aware mask builder.
    ///
    /// Use this when `k_full` was returned from a cache that may have
    /// rotated mid-prefill (sliding window). The K's actual seq length
    /// (`kv_actual`) and absolute starting position (`cache_first_held_pos`)
    /// can both differ from the naive `kv_offset + query_len` /  0 the
    /// non-chunked path assumes.
    ///
    /// Derivation: total absolute positions covered by THIS forward call is
    /// `[kv_offset, kv_offset + query_len)`. Cache holds the most recent
    /// `kv_actual` positions, so its first absolute position is
    /// `(kv_offset + query_len).saturating_sub(kv_actual)`.
    /// Per-(query_len, kv_offset, kv_actual) memoization of the causal mask.
    /// Mirrors mlx_lm's `_make_masks` — build once per layer-type per forward,
    /// share across all 30 layers. Without this, prefill spends ~9.7s/8K-call
    /// rebuilding the same [L, kv] mask 30× (measured 2026-05-15: attn.sdpa
    /// 2K→8K scaling 32×, dominated by mask construction).
    ///
    /// Cache key includes `kv_actual` because sliding-window caches can return
    /// truncated K shapes mid-prefill (rotated). When the key changes (e.g.
    /// next decode step's offset advanced, or new forward call) the slot is
    /// invalidated transparently on miss.
    ///
    /// Thread-local so concurrent forward calls (separate Tokio tasks) don't
    /// collide. Single-tenant decode is the dominant path so a thread_local
    /// is a no-overhead fit.
    struct MaskCacheEntry {
        query_len: usize,
        kv_offset: usize,
        kv_actual: usize,
        mask: Array,
    }

    thread_local! {
        static MASK_CACHE_SLIDING: std::cell::RefCell<Option<MaskCacheEntry>> =
            std::cell::RefCell::new(None);
        static MASK_CACHE_FULL: std::cell::RefCell<Option<MaskCacheEntry>> =
            std::cell::RefCell::new(None);
    }

    pub fn make_attention_mask_for_layer_chunked(
        kind: NativeGemma4LayerType,
        cfg: &NativeGemma4TextConfig,
        query_len: usize,
        kv_offset: usize,
        kv_actual: usize,
    ) -> Result<Option<Array>> {
        if query_len <= 1 {
            // Decode path: invalidate any prefill-time mask still pinned in
            // the cache. Holding a large [L,L] bf16 mask across decode hurts
            // GPU memory pressure (the array is a lazy node retained by
            // RefCell) and was empirically measured to regress decode
            // throughput by ~60% even though decode itself never reads it.
            MASK_CACHE_SLIDING.with(|c| c.borrow_mut().take());
            MASK_CACHE_FULL.with(|c| c.borrow_mut().take());
            return Ok(None);
        }

        let slot: &'static std::thread::LocalKey<std::cell::RefCell<Option<MaskCacheEntry>>> =
            match kind {
                NativeGemma4LayerType::SlidingAttention => &MASK_CACHE_SLIDING,
                NativeGemma4LayerType::FullAttention => &MASK_CACHE_FULL,
            };

        // Cache hit fast path: same shape parameters → reuse the prior mask.
        let hit = slot.with(|c| {
            c.borrow().as_ref().and_then(|e| {
                if e.query_len == query_len && e.kv_offset == kv_offset && e.kv_actual == kv_actual
                {
                    Some(e.mask.clone())
                } else {
                    None
                }
            })
        });
        if let Some(mask) = hit {
            return Ok(Some(mask));
        }

        // Miss: build, then memoize.
        let window = match kind {
            NativeGemma4LayerType::SlidingAttention => Some(cfg.sliding_window),
            NativeGemma4LayerType::FullAttention => None,
        };
        let cache_first_held_pos = (kv_offset + query_len).saturating_sub(kv_actual);
        let built = build_causal_mask_abs(
            kv_offset,
            query_len,
            cache_first_held_pos,
            kv_actual,
            window,
        )?;

        if let Some(ref m) = built {
            slot.with(|c| {
                *c.borrow_mut() = Some(MaskCacheEntry {
                    query_len,
                    kv_offset,
                    kv_actual,
                    mask: m.clone(),
                });
            });
        }
        Ok(built)
    }

    // ───────────────────────── quant param resolver ─────────────────────────

    /// Resolve `(group_size, bits, mode)` for a tensor whose safetensors path
    /// is `base` (e.g. `language_model.model.layers.0.mlp.gate_proj`).
    ///
    /// Overrides are looked up in the parsed `quantization` block by the
    /// **un-suffixed** base path; `.weight`/`.scales`/`.biases` are stripped
    /// here. Falls back to the block default when no override matches.
    /// Mirrors Qwen3.5's convention: per-tensor overrides are always MODE_AFFINE
    /// (typically used to keep a high-precision tensor inside an MXFP4 model),
    /// while the block default dispatches on `quant.mode` ("affine" | "mxfp4").
    pub fn quant_params_for(
        cfg: &NativeGemma4Config,
        base: &str,
    ) -> Result<(i32, i32, &'static CStr)> {
        let quant = cfg
            .effective_quantization()
            .ok_or_else(|| anyhow!("quant_params_for({base}): no quantization block in config"))?;
        let base_key = base
            .trim_end_matches(".weight")
            .trim_end_matches(".scales")
            .trim_end_matches(".biases");
        if let Some(ov) = quant.overrides.get(base_key) {
            // Override mode defaults to AFFINE when absent (Qwen3.6
            // convention). When the override carries an explicit mode,
            // dispatch on it the same way as the top-level default.
            let ov_mode = ov.mode.as_deref().unwrap_or("affine");
            match ov_mode {
                "affine" => {
                    if !matches!(ov.bits, 2 | 3 | 4 | 5 | 6 | 8) {
                        return Err(anyhow!(
                            "quant_params_for({base_key}): affine override bits={} unsupported (mlx supports 2/3/4/5/6/8)",
                            ov.bits
                        ));
                    }
                    return Ok((ov.group_size as i32, ov.bits as i32, MODE_AFFINE));
                }
                "mxfp4" => {
                    if ov.bits != 4 || ov.group_size != 32 {
                        return Err(anyhow!(
                            "quant_params_for({base_key}): mxfp4 override requires bits=4 group_size=32, got bits={} group={}",
                            ov.bits,
                            ov.group_size
                        ));
                    }
                    return Ok((ov.group_size as i32, ov.bits as i32, MODE_MXFP4));
                }
                "mxfp8" => {
                    if ov.bits != 8 || ov.group_size != 32 {
                        return Err(anyhow!(
                            "quant_params_for({base_key}): mxfp8 override requires bits=8 group_size=32, got bits={} group={}",
                            ov.bits,
                            ov.group_size
                        ));
                    }
                    return Ok((ov.group_size as i32, ov.bits as i32, MODE_MXFP8));
                }
                "nvfp4" => {
                    if ov.bits != 4 || ov.group_size != 16 {
                        return Err(anyhow!(
                            "quant_params_for({base_key}): nvfp4 override requires bits=4 group_size=16, got bits={} group={}",
                            ov.bits,
                            ov.group_size
                        ));
                    }
                    return Ok((ov.group_size as i32, ov.bits as i32, MODE_NVFP4));
                }
                other => {
                    return Err(anyhow!(
                        "quant_params_for({base_key}): override has unsupported mode {other:?} (supported: affine, mxfp4, mxfp8, nvfp4)"
                    ));
                }
            }
        }
        let mode_cstr = match quant.mode.as_str() {
            "affine" => {
                if !matches!(quant.bits, 2 | 3 | 4 | 5 | 6 | 8) {
                    return Err(anyhow!(
                        "quant_params_for({base_key}): affine bits={} unsupported (mlx supports 2/3/4/5/6/8)",
                        quant.bits
                    ));
                }
                MODE_AFFINE
            }
            "mxfp4" => {
                if quant.bits != 4 || quant.group_size != 32 {
                    return Err(anyhow!(
                        "quant_params_for({base_key}): mxfp4 requires bits=4 group_size=32, got bits={} group={}",
                        quant.bits,
                        quant.group_size
                    ));
                }
                MODE_MXFP4
            }
            "mxfp8" => {
                if quant.bits != 8 || quant.group_size != 32 {
                    return Err(anyhow!(
                        "quant_params_for({base_key}): mxfp8 requires bits=8 group_size=32, got bits={} group={}",
                        quant.bits,
                        quant.group_size
                    ));
                }
                MODE_MXFP8
            }
            "nvfp4" => {
                if quant.bits != 4 || quant.group_size != 16 {
                    return Err(anyhow!(
                        "quant_params_for({base_key}): nvfp4 requires bits=4 group_size=16, got bits={} group={}",
                        quant.bits,
                        quant.group_size
                    ));
                }
                MODE_NVFP4
            }
            other => {
                return Err(anyhow!(
                    "quant_params_for({base_key}): unsupported quantization mode {other:?} (supported: affine, mxfp4, mxfp8, nvfp4)"
                ));
            }
        };
        Ok((quant.group_size as i32, quant.bits as i32, mode_cstr))
    }

    // ───────────────────────── resolved per-layer weights ─────────────────────────

    /// Quantized linear tensor triple `(weight, scales, biases?)` plus its
    /// `(group_size, bits, mode)` dispatch params. Mirrors Qwen3.5's
    /// `ResolvedLinear`.
    pub struct ResolvedGemma4QuantLinear {
        pub weight: Array,
        pub scales: Array,
        pub biases: Option<Array>,
        pub group_size: i32,
        pub bits: i32,
        pub mode: &'static CStr,
    }

    /// `embed_tokens` is the one weight that may legitimately ship as bf16:
    /// `imatrix_mixed_quant.py` (and mlx-lm's `convert.py` predicate path)
    /// keeps the embedding table at bf16 by default since the gather/lookup
    /// path is bandwidth-bound on the row dimension, not arithmetic-bound,
    /// and the bpw savings are negligible (~0.5% of total) against the
    /// quality hit. mlx-lm's loader infers quant vs bf16 from the presence
    /// of `.scales`; we mirror that here. All other linear layers go
    /// through `ResolvedGemma4QuantLinear` unchanged (they're always
    /// quantized in production catalog builds).
    pub enum EmbedTokensWeights {
        Quantized(ResolvedGemma4QuantLinear),
        Bf16(Array),
    }

    pub struct ResolvedGemma4AttnWeights {
        pub q_proj: ResolvedGemma4QuantLinear,
        pub k_proj: ResolvedGemma4QuantLinear,
        /// Absent on full-attention layers when `attention_k_eq_v=true` —
        /// the forward path reuses `k_proj`'s output for values.
        pub v_proj: Option<ResolvedGemma4QuantLinear>,
        pub o_proj: ResolvedGemma4QuantLinear,
        pub q_norm: Array,
        pub k_norm: Array,
    }

    pub struct ResolvedGemma4DenseMlpWeights {
        pub gate_proj: ResolvedGemma4QuantLinear,
        pub up_proj: ResolvedGemma4QuantLinear,
        pub down_proj: ResolvedGemma4QuantLinear,
    }

    pub struct ResolvedGemma4ExpertsWeights {
        /// Shape `[num_experts, moe_intermediate, hidden]` (quantized).
        pub gate_proj: ResolvedGemma4QuantLinear,
        pub up_proj: ResolvedGemma4QuantLinear,
        /// Shape `[num_experts, hidden, moe_intermediate]`.
        pub down_proj: ResolvedGemma4QuantLinear,
    }

    pub struct ResolvedGemma4RouterWeights {
        /// Shape `[num_experts, hidden]`, 8-bit per the override map.
        pub proj: ResolvedGemma4QuantLinear,
        /// Shape `[hidden]` — pre-router RMSNorm scale (raw, before root_size).
        pub scale: Array,
        /// pre-computed `scale * hidden^-0.5` for the
        /// pre-projection rms_norm weight. Saves 1 Array::from_f32 + 1
        /// multiply on every router call (30 layers × 1 = 30 mlx ops/step
        /// avoided). Mirrors mlx-lm's `Router.__init__` which stores
        /// `self.scale * self._root_size` once.
        pub scaled_weight: Array,
        /// Shape `[num_experts]` — post-routing weight scaling.
        pub per_expert_scale: Array,
    }

    pub struct ResolvedGemma4LayerWeights {
        pub kind: NativeGemma4LayerType,
        pub attn: ResolvedGemma4AttnWeights,
        pub dense_mlp: ResolvedGemma4DenseMlpWeights,
        pub router: ResolvedGemma4RouterWeights,
        pub experts: ResolvedGemma4ExpertsWeights,
        pub input_layernorm: Array,
        pub post_attention_layernorm: Array,
        pub pre_feedforward_layernorm: Array,
        pub pre_feedforward_layernorm_2: Array,
        pub post_feedforward_layernorm: Array,
        pub post_feedforward_layernorm_1: Array,
        pub post_feedforward_layernorm_2: Array,
        pub layer_scalar: Array,
    }

    /// `x[:, start..end, :]` for a `[B, L, H]` array. Expressed as a gather
    /// because mlx-rs exposes no axis-slice op; the cost is prefill-only.
    fn take_span(x: &Array, start: i32, end: i32) -> Result<Array> {
        let idx: Vec<i32> = (start..end).collect();
        let idx = Array::from_slice(&idx, &[idx.len() as i32]);
        mlx_rs::ops::indexing::take_axis(x, &idx, 1).context("take_span: take_axis(axis=1)")
    }

    fn require_clone(weights: &NativeGemma4Weights, name: &str) -> Result<Array> {
        weights
            .require(name)
            .map(|arr| arr.clone())
            .with_context(|| format!("require_clone: missing `{name}`"))
    }

    fn resolve_quant_linear(
        weights: &NativeGemma4Weights,
        cfg: &NativeGemma4Config,
        base: &str,
    ) -> Result<ResolvedGemma4QuantLinear> {
        let weight = require_clone(weights, &format!("{base}.weight"))?;
        let scales = require_clone(weights, &format!("{base}.scales"))?;
        let biases = weights.get(&format!("{base}.biases")).cloned();
        let (group_size, bits, mode) = quant_params_for(cfg, base)?;
        Ok(ResolvedGemma4QuantLinear {
            weight,
            scales,
            biases,
            group_size,
            bits,
            mode,
        })
    }

    /// `embed_tokens`-only resolver — accepts bf16 weights when `.scales` is
    /// absent (skip-list build), delegates to `resolve_quant_linear` otherwise.
    fn resolve_embed_tokens(
        weights: &NativeGemma4Weights,
        cfg: &NativeGemma4Config,
        base: &str,
    ) -> Result<EmbedTokensWeights> {
        if weights.get(&format!("{base}.scales")).is_some() {
            return Ok(EmbedTokensWeights::Quantized(resolve_quant_linear(
                weights, cfg, base,
            )?));
        }
        let w = require_clone(weights, &format!("{base}.weight"))?;
        Ok(EmbedTokensWeights::Bf16(w))
    }

    fn resolve_attn_layer(
        weights: &NativeGemma4Weights,
        cfg: &NativeGemma4Config,
        layer_idx: usize,
        kind: NativeGemma4LayerType,
    ) -> Result<ResolvedGemma4AttnWeights> {
        let base = format!("language_model.model.layers.{layer_idx}.self_attn");
        let q_proj = resolve_quant_linear(weights, cfg, &format!("{base}.q_proj"))?;
        let k_proj = resolve_quant_linear(weights, cfg, &format!("{base}.k_proj"))?;
        let v_proj = if cfg.text_config.use_k_eq_v_for(kind) {
            None
        } else {
            Some(resolve_quant_linear(
                weights,
                cfg,
                &format!("{base}.v_proj"),
            )?)
        };
        let o_proj = resolve_quant_linear(weights, cfg, &format!("{base}.o_proj"))?;
        let q_norm = require_clone(weights, &format!("{base}.q_norm.weight"))?;
        let k_norm = require_clone(weights, &format!("{base}.k_norm.weight"))?;
        Ok(ResolvedGemma4AttnWeights {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
        })
    }

    fn resolve_dense_mlp(
        weights: &NativeGemma4Weights,
        cfg: &NativeGemma4Config,
        layer_idx: usize,
    ) -> Result<ResolvedGemma4DenseMlpWeights> {
        let base = format!("language_model.model.layers.{layer_idx}.mlp");
        Ok(ResolvedGemma4DenseMlpWeights {
            gate_proj: resolve_quant_linear(weights, cfg, &format!("{base}.gate_proj"))?,
            up_proj: resolve_quant_linear(weights, cfg, &format!("{base}.up_proj"))?,
            down_proj: resolve_quant_linear(weights, cfg, &format!("{base}.down_proj"))?,
        })
    }

    fn resolve_experts(
        weights: &NativeGemma4Weights,
        cfg: &NativeGemma4Config,
        layer_idx: usize,
    ) -> Result<ResolvedGemma4ExpertsWeights> {
        let base = format!("language_model.model.layers.{layer_idx}.experts.switch_glu");
        Ok(ResolvedGemma4ExpertsWeights {
            gate_proj: resolve_quant_linear(weights, cfg, &format!("{base}.gate_proj"))?,
            up_proj: resolve_quant_linear(weights, cfg, &format!("{base}.up_proj"))?,
            down_proj: resolve_quant_linear(weights, cfg, &format!("{base}.down_proj"))?,
        })
    }

    fn resolve_router(
        weights: &NativeGemma4Weights,
        cfg: &NativeGemma4Config,
        layer_idx: usize,
    ) -> Result<ResolvedGemma4RouterWeights> {
        let base = format!("language_model.model.layers.{layer_idx}.router");
        let proj = resolve_quant_linear(weights, cfg, &format!("{base}.proj"))?;
        let scale = require_clone(weights, &format!("{base}.scale"))?;
        let per_expert_scale = require_clone(weights, &format!("{base}.per_expert_scale"))?;
        // pre-compute scale × hidden^-0.5 once at load time.
        // cast the root_size scalar to scale.dtype() (bf16)
        // first so the multiply doesn't promote scale to f32. An f32
        // scaled_weight cascades through the router's rms_norm output and
        // poisons the router qmm input.
        let root_size = (cfg.text_config.hidden_size as f32).powf(-0.5);
        let scale_dtype = scale.dtype();
        let root_size_const = Array::from_f32(root_size)
            .as_dtype(scale_dtype)
            .with_context(|| format!("resolve_router({base}): cast root_size to scale dtype"))?;
        let scaled_weight = mlx_rs::ops::multiply(&scale, &root_size_const)
            .with_context(|| format!("resolve_router({base}): scale × root_size"))?;
        Ok(ResolvedGemma4RouterWeights {
            proj,
            scale,
            scaled_weight,
            per_expert_scale,
        })
    }

    fn resolve_layer(
        weights: &NativeGemma4Weights,
        cfg: &NativeGemma4Config,
        layer_idx: usize,
        kind: NativeGemma4LayerType,
    ) -> Result<ResolvedGemma4LayerWeights> {
        let base = format!("language_model.model.layers.{layer_idx}");
        let attn = resolve_attn_layer(weights, cfg, layer_idx, kind)?;
        let dense_mlp = resolve_dense_mlp(weights, cfg, layer_idx)?;
        let router = resolve_router(weights, cfg, layer_idx)?;
        let experts = resolve_experts(weights, cfg, layer_idx)?;
        Ok(ResolvedGemma4LayerWeights {
            kind,
            attn,
            dense_mlp,
            router,
            experts,
            input_layernorm: require_clone(weights, &format!("{base}.input_layernorm.weight"))?,
            post_attention_layernorm: require_clone(
                weights,
                &format!("{base}.post_attention_layernorm.weight"),
            )?,
            pre_feedforward_layernorm: require_clone(
                weights,
                &format!("{base}.pre_feedforward_layernorm.weight"),
            )?,
            pre_feedforward_layernorm_2: require_clone(
                weights,
                &format!("{base}.pre_feedforward_layernorm_2.weight"),
            )?,
            post_feedforward_layernorm: require_clone(
                weights,
                &format!("{base}.post_feedforward_layernorm.weight"),
            )?,
            post_feedforward_layernorm_1: require_clone(
                weights,
                &format!("{base}.post_feedforward_layernorm_1.weight"),
            )?,
            post_feedforward_layernorm_2: require_clone(
                weights,
                &format!("{base}.post_feedforward_layernorm_2.weight"),
            )?,
            layer_scalar: require_clone(weights, &format!("{base}.layer_scalar"))?,
        })
    }

    // ──────────────── TurboQuant rotation bake into Wv/Wo ────────────────
    //
    // The V-side rotation `V_rot = V @ R` and downstream un-rotation
    // `V_dq_unrot = V_dq @ Rᵀ` can be absorbed into the projection
    // weights at load time, eliminating both matmuls at runtime:
    //
    //   V_rot   = X @ W_v @ R          = X @ (W_v @ R)        = X @ W_v_rot
    //   out     = attn @ R^T @ W_o     = attn @ (R^T @ W_o)   = attn @ W_o_rot
    //                                    (block-diagonally per head)
    //
    // Math is exact for the un-quantized weights. With mlx-affine quant,
    // the quant noise pattern differs from the un-baked path (the rotated
    // weight is quantized once instead of `Wv quantized + runtime R
    // matmul`), but the per-element noise magnitude is the same order.
    // K and Q rotations cannot be baked because RoPE sits between
    // projection and rotation and doesn't commute with Haar R.
    //
    // Why this matters: in the SlidingTurboquant branch, V rotation +
    // V_dq un-rotation are the dominant remaining cost beyond mlx's tuned
    // dispatches. The previous Lever 1 (fused rot+encode kernel) regressed
    // -5% because mlx's standalone matmul beats our in-kernel naive
    // matmul. Baking sidesteps the matmul-quality gate entirely.

    /// Bake a Haar rotation `R` into a v_proj quantized weight.
    ///
    /// Math: if `W_v` is viewed as `[H_kv, head_dim, hidden]`, the new
    /// weight is `W_v_rot[h, e, j] = sum_d W_v[h, d, j] * R[d, e]`. At
    /// runtime, `V_rot = X @ W_v_rot^T` produces V already in rotated
    /// space — the explicit `rotate_last_axis(V, R)` call can be skipped.
    fn bake_r_into_v_proj(
        lin: &ResolvedGemma4QuantLinear,
        r_arr: &Array,
        h_kv: i32,
        head_dim: i32,
    ) -> Result<ResolvedGemma4QuantLinear> {
        // Optional re-quant precision override: rotated weights may have a
        // noise distribution that mlx-affine 4-bit can't tolerate well.
        // `LUMEN_GEMMA4_TQ_BAKE_BITS` (default = source bits) lets us pin
        // the requant precision (8-bit recommended for diagnostic A/B).
        let target_bits = std::env::var("LUMEN_GEMMA4_TQ_BAKE_BITS")
            .ok()
            .and_then(|s| s.parse::<i32>().ok())
            .unwrap_or(lin.bits);

        // Dequantize the source weight into f32 for the rotation matmul
        // so R's small Haar entries (~1/√D) don't lose mantissa to bf16.
        let w_bf16 = dequantize_with_mode(
            &lin.weight,
            &lin.scales,
            lin.biases.as_ref(),
            lin.group_size,
            lin.bits,
            lin.mode,
        )
        .context("bake_r_into_v_proj: dequantize")?;
        let w_dense = w_bf16
            .as_dtype(mlx_rs::Dtype::Float32)
            .context("bake_r_into_v_proj: cast W_v to f32")?;

        let shape = w_dense.shape().to_vec();
        if shape.len() != 2 {
            return Err(anyhow!(
                "bake_r_into_v_proj: dequantized W_v must be rank 2, got {:?}",
                shape
            ));
        }
        let out_dim = shape[0];
        let hidden = shape[1];
        if out_dim != h_kv * head_dim {
            return Err(anyhow!(
                "bake_r_into_v_proj: out_dim={out_dim} != H_kv*head_dim={}",
                h_kv * head_dim
            ));
        }

        // [H_kv*head_dim, hidden] → [H_kv, head_dim, hidden] → contract on
        // head_dim via transpose-matmul-transpose.
        let w_3d = mlx_rs::ops::reshape(&w_dense, &[h_kv, head_dim, hidden])
            .context("bake_r_into_v_proj: reshape to 3D")?;
        let w_t = mlx_rs::ops::transpose_axes(&w_3d, &[0, 2, 1])
            .context("bake_r_into_v_proj: transpose to [H_kv, hidden, head_dim]")?;
        // R is f32 by construction; we operate in f32 too.
        let w_t_rot =
            mlx_rs::ops::matmul(&w_t, r_arr).context("bake_r_into_v_proj: matmul W_t @ R")?;
        let w_rot_3d = mlx_rs::ops::transpose_axes(&w_t_rot, &[0, 2, 1])
            .context("bake_r_into_v_proj: transpose back to [H_kv, head_dim, hidden]")?;
        let w_rot_f32 = mlx_rs::ops::reshape(&w_rot_3d, &[h_kv * head_dim, hidden])
            .context("bake_r_into_v_proj: reshape back to 2D")?;
        // Cast to the source's dtype before re-quantizing so scales/biases
        // dtype matches the original (bf16 for Gemma 4 4-bit affine).
        let w_rot_flat = w_rot_f32
            .as_dtype(w_bf16.dtype())
            .context("bake_r_into_v_proj: cast back to source dtype for requant")?;

        let (packed, scales, biases) =
            quantize_with_mode(&w_rot_flat, lin.group_size, target_bits, lin.mode)
                .context("bake_r_into_v_proj: re-quantize")?;

        Ok(ResolvedGemma4QuantLinear {
            weight: packed,
            scales,
            biases,
            group_size: lin.group_size,
            bits: target_bits,
            mode: lin.mode,
        })
    }

    /// Bake `R` into an o_proj quantized weight (block-diagonally per
    /// head). With V un-rotation absorbed here, V_dq stays in rotated
    /// space throughout SDPA and o_proj recovers the original head_dim
    /// space:
    ///
    ///   W_o_rot[hidden, h, e] = sum_d W_o[hidden, h, d] * R[d, e]
    ///
    /// (Equivalent to `R^T` applied along W_o's head_dim block per head.)
    fn bake_r_into_o_proj(
        lin: &ResolvedGemma4QuantLinear,
        r_arr: &Array,
        h: i32,
        head_dim: i32,
    ) -> Result<ResolvedGemma4QuantLinear> {
        let w_bf16 = dequantize_with_mode(
            &lin.weight,
            &lin.scales,
            lin.biases.as_ref(),
            lin.group_size,
            lin.bits,
            lin.mode,
        )
        .context("bake_r_into_o_proj: dequantize")?;
        let w_dense = w_bf16
            .as_dtype(mlx_rs::Dtype::Float32)
            .context("bake_r_into_o_proj: cast W_o to f32")?;

        let shape = w_dense.shape().to_vec();
        if shape.len() != 2 {
            return Err(anyhow!(
                "bake_r_into_o_proj: dequantized W_o must be rank 2, got {:?}",
                shape
            ));
        }
        let hidden = shape[0];
        let in_dim = shape[1];
        if in_dim != h * head_dim {
            return Err(anyhow!(
                "bake_r_into_o_proj: in_dim={in_dim} != H*head_dim={}",
                h * head_dim
            ));
        }

        let w_3d = mlx_rs::ops::reshape(&w_dense, &[hidden, h, head_dim])
            .context("bake_r_into_o_proj: reshape to 3D")?;
        // head_dim is already the last axis — matmul contracts on it.
        let w_rot_3d =
            mlx_rs::ops::matmul(&w_3d, r_arr).context("bake_r_into_o_proj: matmul W_o @ R")?;
        let w_rot_f32 = mlx_rs::ops::reshape(&w_rot_3d, &[hidden, h * head_dim])
            .context("bake_r_into_o_proj: reshape back to 2D")?;
        let w_rot_flat = w_rot_f32
            .as_dtype(w_bf16.dtype())
            .context("bake_r_into_o_proj: cast back to source dtype for requant")?;

        let (packed, scales, biases) =
            quantize_with_mode(&w_rot_flat, lin.group_size, lin.bits, lin.mode)
                .context("bake_r_into_o_proj: re-quantize")?;

        Ok(ResolvedGemma4QuantLinear {
            weight: packed,
            scales,
            biases,
            group_size: lin.group_size,
            bits: lin.bits,
            mode: lin.mode,
        })
    }

    /// Decide whether to bake `R` into V/Wo weights at load time.
    /// Active iff TurboQuant Stage 1 is enabled AND the bake gate is on.
    /// Default ON when TQ is on (the bake fixes a known V rotation perf
    /// cost; opt-out via `LUMEN_GEMMA4_TQ_BAKE_R=0`).
    fn tq_bake_r_enabled() -> bool {
        // Read through the new mode helper so MODE=on AND MODE=auto both
        // trigger the bake. For Auto, bake is a no-op when the runtime TQ
        // path is OFF for a given request (R @ Rᵀ = I cancels the bake in
        // the non-TQ matmul) — so always baking when TQ may fire is
        // strictly better than gating bake on a per-request decision the
        // load-time code can't see.
        let tq_on = !matches!(gemma4_tq_mode(), Gemma4TqMode::Off);
        if !tq_on {
            return false;
        }
        // V rotation skip implies the bake can't apply.
        let skip_v = std::env::var("LUMEN_GEMMA4_TQ_SKIP_V_ROTATE")
            .map(|s| s == "1")
            .unwrap_or(false);
        if skip_v {
            return false;
        }
        std::env::var("LUMEN_GEMMA4_TQ_BAKE_R")
            .map(|s| s != "0")
            .unwrap_or(true)
    }

    // ───────────────────────── top-level model ─────────────────────────

    /// Output of [`NativeGemma4Model::mtp_step`]. See method docs for the
    /// `committed` / `n_attempted` / `n_accepted` contract.
    #[derive(Debug, Clone)]
    pub struct MtpStepOutput {
        pub committed: Vec<u32>,
        pub n_attempted: usize,
        pub n_accepted: usize,
    }

    /// Prompt-Lookup Decoding (Saxena 2023): find the most-recent occurrence
    /// of the last `n_lookup` tokens in `history` and return up to
    /// `max_draft` following tokens as the draft. Returns empty when no
    /// match is found (caller falls back to single-token decode).
    ///
    /// We iterate from the most recent position backward — the earliest
    /// match a long-ctx prompt produces tends to be stale, while the most
    /// recent repetition is usually the active pattern (loop body,
    /// repeated phrase, etc.).
    fn find_lookup_draft(history: &[u32], n_lookup: usize, max_draft: usize) -> Vec<u32> {
        if n_lookup == 0 || max_draft == 0 || history.len() <= n_lookup {
            return Vec::new();
        }
        let pattern_start = history.len() - n_lookup;
        let pattern = &history[pattern_start..];
        // Search positions [0, pattern_start), most-recent first.
        for start in (0..pattern_start).rev() {
            if &history[start..start + n_lookup] == pattern {
                let draft_start = start + n_lookup;
                let draft_end = (draft_start + max_draft).min(pattern_start);
                if draft_end > draft_start {
                    return history[draft_start..draft_end].to_vec();
                }
            }
        }
        Vec::new()
    }

    pub struct NativeGemma4Model {
        config: NativeGemma4Config,
        embed_tokens: EmbedTokensWeights,
        final_norm: Array,
        layers: Vec<ResolvedGemma4LayerWeights>,
        // pre-allocated constants reused on every forward call.
        // Without this, the hot path constructs ~64 fresh mlx Arrays per
        // decode step (30 v_norm `ones`, 30 router root-size scalars, +4
        // misc) — mirrors mlx-lm's `__init__`-time constant pre-bake.
        const_ones_head_dim_sliding: Array,
        const_ones_head_dim_full: Array,
        const_embed_scale: Array,
        const_softcap_inv: Array,
        const_softcap: Array,
        const_last_idx_one: Array, // last_pos index used by argmax_last_token_lazy in decode steps (L=1)
        // True iff Wv/Wo were baked with the Haar rotation R at load
        // time. When set, the SlidingTurboquant forward path skips the
        // runtime V rotation matmul *and* the V_dq un-rotation matmul.
        // Set by `load()` when `LUMEN_GEMMA4_QUANT_KV_SLIDING_TURBOQUANT=1`
        // + `LUMEN_GEMMA4_TQ_BAKE_R != "0"` (default ON when TQ is on).
        tq_bake_active: bool,
        // Independent gates per-leg so we can A/B Wv-only vs Wo-only.
        tq_bake_v_active: bool,
        tq_bake_o_active: bool,
        // Wo bake on full-attention layers (separate from sliding bake_o
        // because full-attn uses global_head_dim=512 — its own R cached
        // separately). Opt-in via LUMEN_GEMMA4_TQ_BAKE_R_O_FULL_ATTN=1.
        // When ON, runtime branch skips V_dq un-rotation for full-attn
        // layers and `sv_inline` becomes eligible (V stays rotated all
        // the way to o_proj).
        tq_bake_o_full_attn_active: bool,
        // MTP (Multi-Token Prediction) drafter — set via `try_enable_mtp()`.
        // None by default; when Some, callers can route through `mtp_step()`.
        // Drafter shares the trunk's KV cache for its last full-attn + last
        // sliding-attn layers, identified by the indices below.
        mtp: Option<crate::gemma4_mtp::imp::ResolvedGemma4MtpDrafter>,
        /// Index of the last full-attention layer (from `layer_types`).
        /// Used by `mtp_step()` to address the trunk KV cache slot the
        /// drafter's full-attn layer shares.
        last_full_attn_idx: Option<usize>,
        /// Index of the last sliding-attention layer.
        last_sliding_attn_idx: Option<usize>,
        /// Capture slot for the final-norm hidden state. When
        /// `mtp_capture_enabled` is true, `forward_array_impl` stashes a
        /// clone of `h` here right before lm_head so `mtp_step()` can seed
        /// the drafter's `last_hidden_state` without re-running the trunk.
        mtp_capture_slot: std::sync::Mutex<Option<Array>>,
        mtp_capture_enabled: std::sync::atomic::AtomicBool,
        /// True while a `mtp_step()` invocation is in progress (Step A
        /// through Step E). Gates `use_custom_flash`: when true, full-attn
        /// layers use mlx::fast::sdpa uniformly so Step A's S=1 and Step C's
        /// S>1 use the same kernel at the same logical positions (avoids
        /// the ~1-ULP drift that flipped argmax on sharp logits). False
        /// when the trunk is called from anywhere outside `mtp_step()` —
        /// e.g. the standard OFF decode path retains the custom-FA-2 win.
        mtp_active: std::sync::atomic::AtomicBool,
        /// Phase B (v0.6.0 tool-calling robustness) — capture slot for the
        /// lm_head input `h_for_lm_head` that the backend's logit-correction
        /// kernel needs to compute `delta_k = h · Δ[k, :]`. Mirrors the MTP
        /// capture mechanism: enabled via a separate atomic so the standard
        /// decode path stays zero-overhead when correction is off. Active
        /// when `LUMEN_GEMMA4_CRITICAL_LOGIT_CORRECTION=1` and the model
        /// directory has a `logit_corrections.bin` sidecar.
        correction_capture_slot: std::sync::Mutex<Option<Array>>,
        correction_capture_enabled: std::sync::atomic::AtomicBool,
        /// Image encoder. `Some` only when `LUMEN_VISION=1` **and** the
        /// checkpoint still carries `vision_tower.*` weights (some AWQ/imatrix
        /// requantizations drop them). `None` leaves the text path byte-for-byte
        /// unchanged, including its memory footprint.
        vision: Option<crate::gemma4_vision::NativeGemma4VisionTower>,
    }

    /// Opt-in gate for loading the ~1.1 GB image tower.
    fn vision_enabled() -> bool {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("LUMEN_VISION")
                .map(|s| matches!(s.as_str(), "1" | "true" | "TRUE" | "yes"))
                .unwrap_or(false)
        })
    }

    impl NativeGemma4Model {
        /// Load a Gemma 4 model from `model_dir` containing `config.json` and
        /// one or more `*.safetensors` shards. Mirrors mlx_lm's
        /// `Gemma4TextModel.from_pretrained` semantics (text-only).
        pub fn load(model_dir: &Path) -> Result<Self> {
            let cfg = NativeGemma4Config::load(&model_dir.join("config.json"))?;
            cfg.validate_gemma4_family()?;

            let mut weights = NativeGemma4Weights::load_dir(model_dir)?;

            // Build the image tower from the raw bag — `sanitize()` strips
            // `vision_tower.*` / `embed_vision.*` right after this.
            //
            // The gate checks the projection, not just the tower: a checkpoint
            // could plausibly keep one and drop the other, and the projection
            // is the piece with no fallback.
            let vision_present = weights.get("vision_tower.std_scale").is_some()
                && weights
                    .get("embed_vision.embedding_projection.weight")
                    .is_some();
            let vision = if vision_enabled() {
                match cfg.vision_config.clone() {
                    Some(vcfg) if vision_present => {
                        // Resolve the projection's quantization through the same
                        // path every other quantized tensor uses, so a per-tensor
                        // override or a non-affine checkpoint (nvfp4 ships one)
                        // is honoured instead of silently mis-dequantized.
                        let (group_size, bits, mode) =
                            quant_params_for(&cfg, "embed_vision.embedding_projection").context(
                                "resolve quantization for embed_vision.embedding_projection",
                            )?;
                        Some(
                            crate::gemma4_vision::NativeGemma4VisionTower::load(
                                weights.tensors(),
                                vcfg,
                                crate::gemma4_vision::VisionProjectionQuant {
                                    group_size,
                                    bits,
                                    mode,
                                },
                            )
                            .context("load Gemma 4 vision tower")?,
                        )
                    }
                    Some(_) => {
                        eprintln!(
                            "[vision] LUMEN_VISION=1 but {} ships no vision_tower.* / \
                             embed_vision.* weights (requantized text-only) — image input \
                             stays disabled",
                            model_dir.display()
                        );
                        None
                    }
                    None => {
                        eprintln!(
                            "[vision] LUMEN_VISION=1 but config.json has no vision_config \
                             — image input stays disabled"
                        );
                        None
                    }
                }
            } else {
                None
            };

            weights.sanitize()?;
            weights.validate_keys_against_config(&cfg.text_config)?;

            let embed_tokens =
                resolve_embed_tokens(&weights, &cfg, "language_model.model.embed_tokens")?;
            let final_norm = require_clone(&weights, "language_model.model.norm.weight")?;

            let mut layers = cfg
                .text_config
                .layer_types
                .iter()
                .enumerate()
                .map(|(idx, kind)| resolve_layer(&weights, &cfg, idx, *kind))
                .collect::<Result<Vec<_>>>()?;

            // TurboQuant rotation bake into Wv/Wo for sliding layers.
            // One-time load-time matmul + requant; eliminates the
            // runtime V rotation + V_dq un-rotation matmul pair on every
            // attention call when TQ Stage 1 is active.
            //
            // Diagnostic gates:
            //   LUMEN_GEMMA4_TQ_BAKE_R_V=1  → opt-IN Wv bake (default OFF
            //     after NEGATIVE 2026-05-17: real-prompt eval shows
            //     garbage tokens even with 8-bit re-quant. Rotation
            //     interacts poorly with mlx-affine quantization of
            //     trained Wv even though dense math is exact and synthetic
            //     quant-noise MSE matches the runtime rotation path.)
            //   LUMEN_GEMMA4_TQ_BAKE_R_O=0  → opt-OUT Wo bake (default
            //     ON: real-prompt eval verified semantically equivalent
            //     output to un-baked Stage 1 path).
            let tq_bake_active = tq_bake_r_enabled();
            let bake_v = tq_bake_active
                && std::env::var("LUMEN_GEMMA4_TQ_BAKE_R_V")
                    .map(|s| s == "1")
                    .unwrap_or(false);
            let bake_o = tq_bake_active
                && std::env::var("LUMEN_GEMMA4_TQ_BAKE_R_O")
                    .map(|s| s != "0")
                    .unwrap_or(true);
            // Full-attn Wo bake: separate opt-in, only valid when sliding
            // bake_o is also on (the runtime forward path that consumes the
            // baked weight is shared logic gated on `tq_bake_o_active`).
            let bake_o_full_attn = bake_o
                && std::env::var("LUMEN_GEMMA4_TQ_BAKE_R_O_FULL_ATTN")
                    .map(|s| s == "1")
                    .unwrap_or(false);
            if tq_bake_active {
                let head_dim_sliding_usize = cfg.text_config.head_dim;
                let r_arr = crate::turboquant::rotation_matrix_f32(
                    head_dim_sliding_usize,
                    crate::turboquant::TURBOQUANT_SEED,
                )
                .context("load: build rotation R for TQ bake")?;
                let h_kv = cfg.text_config.num_key_value_heads as i32;
                let h_q = cfg.text_config.num_attention_heads as i32;
                let head_dim = head_dim_sliding_usize as i32;
                let mut baked_sliding = 0usize;
                for (idx, lw) in layers.iter_mut().enumerate() {
                    if lw.kind != NativeGemma4LayerType::SlidingAttention {
                        continue;
                    }
                    if bake_v {
                        if let Some(ref vp) = lw.attn.v_proj {
                            let vp_rot = bake_r_into_v_proj(vp, &r_arr, h_kv, head_dim)
                                .with_context(|| {
                                    format!("load: bake R into v_proj (layer {idx})")
                                })?;
                            lw.attn.v_proj = Some(vp_rot);
                        }
                    }
                    if bake_o {
                        let op_rot = bake_r_into_o_proj(&lw.attn.o_proj, &r_arr, h_q, head_dim)
                            .with_context(|| format!("load: bake R into o_proj (layer {idx})"))?;
                        lw.attn.o_proj = op_rot;
                    }
                    baked_sliding += 1;
                }

                // Full-attn Wo bake: independent R (D=global_head_dim=512)
                // cached separately by `rotation_matrix_f32` keyed on
                // (dim, seed). `bake_r_into_o_proj` is D-agnostic — takes
                // head_dim as an arg.
                let mut baked_full = 0usize;
                if bake_o_full_attn {
                    let head_dim_full_usize = cfg.text_config.global_head_dim;
                    let r_arr_full = crate::turboquant::rotation_matrix_f32(
                        head_dim_full_usize,
                        crate::turboquant::TURBOQUANT_SEED,
                    )
                    .context("load: build rotation R (D=full) for full-attn Wo bake")?;
                    let head_dim_full = head_dim_full_usize as i32;
                    for (idx, lw) in layers.iter_mut().enumerate() {
                        if lw.kind != NativeGemma4LayerType::FullAttention {
                            continue;
                        }
                        let op_rot =
                            bake_r_into_o_proj(&lw.attn.o_proj, &r_arr_full, h_q, head_dim_full)
                                .with_context(|| {
                                    format!("load: bake R into o_proj full-attn (layer {idx})")
                                })?;
                        lw.attn.o_proj = op_rot;
                        baked_full += 1;
                    }
                }

                eprintln!(
                    "[gemma4-load] TQ rotation bake: sliding={baked_sliding} full_attn={baked_full} \
                     bake_v={bake_v} bake_o={bake_o} bake_o_full_attn={bake_o_full_attn}"
                );
            }
            // The runtime SlidingTurboquant branch needs to know which legs
            // were baked to correctly skip the corresponding runtime matmul.
            let tq_bake_v_active = bake_v;
            let tq_bake_o_active = bake_o;
            let tq_bake_o_full_attn_active = bake_o_full_attn;

            // Pre-bake hot-path constants. Saves ~60 mlx Array
            // allocations per decode step (30 ones + 30 router scalars).
            //
            // store as bf16 (residual-stream dtype). f32
            // 0-d constants poison the residual stream to f32, which causes
            // MLX's affine quantized_matmul to insert 2 internal astype
            // primitives per qmm call (scales bf16->f32 + biases bf16->f32).
            // With ~326 qmm calls/step, that's ~650 AsType/step of pure
            // dtype-cast overhead.
            let head_dim_sliding = cfg.text_config.head_dim as i32;
            let head_dim_full = cfg.text_config.global_head_dim as i32;
            let const_ones_head_dim_sliding = mlx_rs::ops::ones::<f32>(&[head_dim_sliding])
                .and_then(|a| a.as_dtype(mlx_rs::Dtype::Bfloat16))
                .map_err(|e| anyhow!("P8: ones(head_dim_sliding={head_dim_sliding}): {e}"))?;
            let const_ones_head_dim_full = mlx_rs::ops::ones::<f32>(&[head_dim_full])
                .and_then(|a| a.as_dtype(mlx_rs::Dtype::Bfloat16))
                .map_err(|e| anyhow!("P8: ones(head_dim_full={head_dim_full}): {e}"))?;
            let embed_scale = (cfg.text_config.hidden_size as f32).sqrt();
            let const_embed_scale = Array::from_f32(embed_scale)
                .as_dtype(mlx_rs::Dtype::Bfloat16)
                .map_err(|e| anyhow!("P8: const_embed_scale cast to bf16: {e}"))?;
            let softcap = cfg.text_config.final_logit_softcapping;
            let const_softcap_inv = Array::from_f32(1.0f32 / softcap)
                .as_dtype(mlx_rs::Dtype::Bfloat16)
                .map_err(|e| anyhow!("P8: const_softcap_inv cast to bf16: {e}"))?;
            let const_softcap = Array::from_f32(softcap)
                .as_dtype(mlx_rs::Dtype::Bfloat16)
                .map_err(|e| anyhow!("P8: const_softcap cast to bf16: {e}"))?;
            let const_last_idx_one = Array::from_slice(&[0i32], &[1]);

            // Pre-compute last-layer indices for MTP drafter KV sharing.
            // Drafter's full-attn layer reads K/V from the trunk's deepest
            // full-attn layer; same for sliding.
            let last_full_attn_idx = cfg
                .text_config
                .layer_types
                .iter()
                .enumerate()
                .rev()
                .find(|(_, k)| matches!(**k, NativeGemma4LayerType::FullAttention))
                .map(|(i, _)| i);
            let last_sliding_attn_idx = cfg
                .text_config
                .layer_types
                .iter()
                .enumerate()
                .rev()
                .find(|(_, k)| matches!(**k, NativeGemma4LayerType::SlidingAttention))
                .map(|(i, _)| i);

            Ok(Self {
                config: cfg,
                embed_tokens,
                final_norm,
                layers,
                const_ones_head_dim_sliding,
                const_ones_head_dim_full,
                const_embed_scale,
                const_softcap_inv,
                const_softcap,
                const_last_idx_one,
                tq_bake_active,
                tq_bake_v_active,
                tq_bake_o_active,
                tq_bake_o_full_attn_active,
                mtp: None,
                last_full_attn_idx,
                last_sliding_attn_idx,
                mtp_capture_slot: std::sync::Mutex::new(None),
                mtp_capture_enabled: std::sync::atomic::AtomicBool::new(false),
                correction_capture_slot: std::sync::Mutex::new(None),
                correction_capture_enabled: std::sync::atomic::AtomicBool::new(false),
                mtp_active: std::sync::atomic::AtomicBool::new(false),
                vision,
            })
        }

        /// Enable Multi-Token Prediction speculative decoding by loading the
        /// matched drafter checkpoint at `drafter_dir`. Returns true on success.
        ///
        /// For Gemma 4 26B-A4B, the matched drafter is
        /// `mlx-community/gemma-4-26B-A4B-it-assistant-bf16` (4-layer mini,
        /// shares K/V from the trunk's last full-attn + last sliding-attn
        /// layers). See `gemma4_mtp_drafter_architecture.md` for the full spec.
        ///
        /// scaffolds drafter ownership. The `mtp_step()`
        /// orchestration (Step A trunk decode → B drafter loop → C verify →
        /// D accept/reject → E rollback) lands in subsequent phases.
        pub fn try_enable_mtp(&mut self, drafter_dir: &Path) -> Result<bool> {
            let drafter = crate::gemma4_mtp::imp::load_drafter(drafter_dir)
                .with_context(|| format!("try_enable_mtp: {}", drafter_dir.display()))?;
            // Sanity check: drafter's backbone_hidden_size must match trunk's
            // hidden_size; otherwise pre_projection / post_projection won't
            // align with our hidden state shapes.
            let trunk_hidden = self.config.text_config.hidden_size as usize;
            let drafter_backbone = drafter.config.backbone_hidden_size;
            if drafter_backbone != trunk_hidden {
                return Err(anyhow!(
                    "try_enable_mtp: drafter backbone_hidden_size={drafter_backbone} \
                     ≠ trunk hidden_size={trunk_hidden} — drafter is mismatched to this trunk"
                ));
            }
            self.mtp = Some(drafter);
            Ok(true)
        }

        /// Whether MTP is currently enabled (drafter loaded). Phase 3 caller
        /// gate.
        pub fn mtp_enabled(&self) -> bool {
            self.mtp.is_some()
        }

        /// TQ rotation-bake state set at load time. `(bake_v, bake_o)`.
        /// Exposed so the startup log can surface whether the runtime V
        /// rotation matmul + V_dq un-rotation are actually being skipped.
        pub fn tq_bake_state(&self) -> (bool, bool) {
            (self.tq_bake_v_active, self.tq_bake_o_active)
        }

        /// Exposes whether Wo bake is also active on full-attention layers.
        /// Independent flag because full-attn uses a separate R (D=512).
        pub fn tq_bake_o_full_attn_state(&self) -> bool {
            self.tq_bake_o_full_attn_active
        }

        /// Toggle the MTP capture hook in `forward_array_impl`. When set,
        /// the next forward call stashes a clone of the post-final-norm
        /// hidden into `mtp_capture_slot` for `mtp_step()` to consume via
        /// `take_captured_last_h()`. Default off — zero overhead when not
        /// used.
        pub fn set_mtp_capture_enabled(&self, enabled: bool) {
            self.mtp_capture_enabled
                .store(enabled, std::sync::atomic::Ordering::Relaxed);
        }

        /// Take the captured last-layer hidden state set by the prior forward
        /// pass under `set_mtp_capture_enabled(true)`. Returns `None` if no
        /// hidden was captured (capture disabled, or already consumed).
        pub fn take_captured_last_h(&self) -> Option<Array> {
            self.mtp_capture_slot.lock().ok().and_then(|mut s| s.take())
        }

        /// Phase B (v0.6.0 tool-call robustness) — enable capture of the
        /// `h_for_lm_head` (post-final-norm, post-last-slice) hidden vector
        /// that feeds the lm_head matmul. Backend uses this to compute the
        /// per-critical-token logit correction `h · Δ[k, :]`.
        ///
        /// Default off — zero overhead when not engaged. Backend turns this
        /// on only when both:
        ///   - `LUMEN_GEMMA4_CRITICAL_LOGIT_CORRECTION=1` env, and
        ///   - the model directory has a valid `logit_corrections.bin`.
        pub fn set_correction_capture_enabled(&self, enabled: bool) {
            self.correction_capture_enabled
                .store(enabled, std::sync::atomic::Ordering::Relaxed);
        }

        /// Take the captured `h_for_lm_head` from the most recent forward
        /// pass under `set_correction_capture_enabled(true)`. Returns
        /// `None` if no capture occurred (capture disabled, already taken,
        /// or forward not yet called).
        ///
        /// The returned Array still references the lazy MLX graph — callers
        /// must `mx.eval` it (or pull to CPU) before reading values.
        pub fn take_captured_correction_h(&self) -> Option<Array> {
            self.correction_capture_slot
                .lock()
                .ok()
                .and_then(|mut s| s.take())
        }

        /// Index of the trunk's last full-attention layer (drafter's full
        /// layer reads its K/V). `None` if no full-attn layers exist.
        pub fn last_full_attn_idx(&self) -> Option<usize> {
            self.last_full_attn_idx
        }

        /// Index of the trunk's last sliding-attention layer.
        pub fn last_sliding_attn_idx(&self) -> Option<usize> {
            self.last_sliding_attn_idx
        }

        /// MTP decode-loop gate. Caller-side opt-in via env (default OFF).
        /// `LUMEN_GEMMA4_MTP=1` routes `generate()` through `mtp_step()`
        /// when a drafter has been loaded; otherwise the existing single-
        /// token async-pipelined decode loop runs unchanged.
        pub fn mtp_decode_enabled_env() -> bool {
            std::env::var("LUMEN_GEMMA4_MTP")
                .map(|v| v == "1")
                .unwrap_or(false)
        }

        /// `LUMEN_GEMMA4_MTP_BLOCK_SIZE` — number of draft tokens proposed
        /// per `mtp_step()` call. Default 6 per mlx-vlm single-request
        /// recommendation for `gemma-4-26B-A4B-it-assistant-bf16`.
        pub fn mtp_block_size_env() -> usize {
            std::env::var("LUMEN_GEMMA4_MTP_BLOCK_SIZE")
                .ok()
                .and_then(|s| s.parse().ok())
                .filter(|&k: &usize| k > 0)
                .unwrap_or(6)
        }

        /// Output of one MTP step.
        ///
        /// `committed`: full list of tokens the caller should add to its
        ///   generation stream. Always non-empty. The LAST element is the
        ///   token to pass as `committed_token` for the next `mtp_step()`
        ///   call (corresponds to either the bonus token on full-accept or
        ///   the correction token on partial reject — neither is in the
        ///   trunk cache yet).
        /// `n_attempted` = `n_draft` (drafts proposed).
        /// `n_accepted`  = drafts that matched the trunk's prediction.
        pub fn mtp_step(
            &self,
            cache: &mut NativeGemma4PromptCache,
            committed_token: u32,
            n_draft: usize,
        ) -> Result<MtpStepOutput> {
            if n_draft == 0 {
                return Err(anyhow!("mtp_step: n_draft must be >= 1"));
            }
            let drafter = self
                .mtp
                .as_ref()
                .ok_or_else(|| anyhow!("mtp_step: MTP not enabled (call try_enable_mtp first)"))?;
            let last_full_idx = self
                .last_full_attn_idx
                .ok_or_else(|| anyhow!("mtp_step: trunk has no full-attention layers"))?;
            let last_sliding_idx = self
                .last_sliding_attn_idx
                .ok_or_else(|| anyhow!("mtp_step: trunk has no sliding-attention layers"))?;

            // Pre-state: cache offset = M (committed_token NOT yet in cache).
            let m_pre = cache.offset();

            // Mark MTP path active for the duration of this call so the
            // trunk's full-attn SDPA selector falls back to mlx::fast::sdpa
            // (Step A's S=1 and Step C's S>1 must use the SAME kernel at
            // the same logical positions — see use_custom_flash comment).
            // RAII guard: unsets on every exit path (early return, ? bubble).
            struct MtpActiveGuard<'a>(&'a std::sync::atomic::AtomicBool);
            impl<'a> Drop for MtpActiveGuard<'a> {
                fn drop(&mut self) {
                    self.0.store(false, std::sync::atomic::Ordering::Relaxed);
                }
            }
            self.mtp_active
                .store(true, std::sync::atomic::Ordering::Relaxed);
            let _mtp_active_guard = MtpActiveGuard(&self.mtp_active);

            // === Step A: trunk decode committed_token + capture last hidden ===
            self.set_mtp_capture_enabled(true);
            let step_a_input = Array::from_slice(&[committed_token], &[1, 1]);
            let logits_a = self
                .forward_array_last_token(&step_a_input, cache)
                .context("mtp_step: Step A trunk forward")?;
            self.set_mtp_capture_enabled(false);
            let trunk_h_at_m = self
                .take_captured_last_h()
                .ok_or_else(|| anyhow!("mtp_step: Step A failed to capture trunk last hidden"))?;
            let next_token = self
                .argmax_last_token(&logits_a)
                .context("mtp_step: Step A argmax next_token")?;
            // Cache offset is now M+1.

            // === Step B: drafter loop ===
            let mut last_h = trunk_h_at_m;
            let mut last_tok = next_token;
            let mut drafts: Vec<u32> = Vec::with_capacity(n_draft);
            for k in 0..n_draft {
                let tok_arr = Array::from_slice(&[last_tok], &[1, 1]);
                // Trunk embed × embed_scale (= sqrt(hidden_size)) — matches
                // mlx-vlm reference `tok_embed = self._input_embed(tok) *
                // self._input_embed_scale`. Our prior code applied only
                // `embed_lookup_affine` (the lookup) and missed the scale,
                // making the drafter's input ~53x too small (sqrt(2816)).
                let trunk_embed_raw = self
                    .embed_lookup_affine(&tok_arr)
                    .with_context(|| format!("mtp_step: Step B embed lookup k={k}"))?;
                let trunk_embed = mlx_rs::ops::multiply(&trunk_embed_raw, &self.const_embed_scale)
                    .with_context(|| format!("mtp_step: Step B embed × scale k={k}"))?;
                let (k_full, v_full) = cache
                    .layer_kv_bf16(last_full_idx)
                    .with_context(|| format!("mtp_step: Step B KV(full) k={k}"))?;
                let (k_slide, v_slide) = cache
                    .layer_kv_bf16(last_sliding_idx)
                    .with_context(|| format!("mtp_step: Step B KV(sliding) k={k}"))?;
                // Position is HELD CONSTANT across draft steps (mlx-vlm
                // reference comment: "position_ids is held constant across
                // all draft steps"). The drafter's recurrent state
                // `last_hidden_state` encodes the temporal advancement; Q
                // position represents "next position to predict" =
                // m_pre + 1 (= cache.offset after Step A, the slot right
                // after the last committed token). Our prior code used
                // `(m_pre + 1 + k)` which incremented per k — caused the
                // drafter's RoPE to drift forward each step, producing
                // wrong attention against the fixed shared K/V.
                let position = (m_pre + 1) as i32;
                let (h_drafter_space, h_backbone_space) = drafter
                    .draft_step(
                        &trunk_embed,
                        &last_h,
                        &k_full,
                        &v_full,
                        &k_slide,
                        &v_slide,
                        position,
                    )
                    .with_context(|| format!("mtp_step: Step B draft_step k={k}"))?;
                // Drafter's own lm_head (tied embed, no softcap) — matches
                // mlx-vlm reference. Logits in drafter-hidden vocab space.
                let logits_draft = drafter
                    .lm_head(&h_drafter_space)
                    .with_context(|| format!("mtp_step: Step B drafter lm_head k={k}"))?;
                let draft_tok = self
                    .argmax_last_token(&logits_draft)
                    .with_context(|| format!("mtp_step: Step B argmax k={k}"))?;
                drafts.push(draft_tok);
                // Recurrent state for next iteration is the backbone-space
                // post-projection output.
                last_h = h_backbone_space;
                last_tok = draft_tok;
            }

            // === Step C: trunk verify with [next_token, draft_0..draft_{K-1}] ===
            let mut verify_in: Vec<u32> = Vec::with_capacity(1 + n_draft);
            verify_in.push(next_token);
            for d in &drafts {
                verify_in.push(*d);
            }
            let verify_arr = Array::from_slice(&verify_in, &[1, verify_in.len() as i32]);
            let verify_logits = self
                .forward_array(&verify_arr, cache)
                .context("mtp_step: Step C trunk verify forward")?;
            // Cache offset now M + 1 + (K + 1) = M + K + 2.

            // === Step D: accept-reject ===
            // argmax along last (vocab) axis → [1, 1+K] Int32 tensor.
            let argmax_per_pos =
                mlx_rs::ops::indexing::argmax_axis(&verify_logits, -1, /* keep_dims */ false)
                    .context("mtp_step: Step D argmax")?
                    .as_dtype(mlx_rs::Dtype::Int32)
                    .context("mtp_step: Step D cast to Int32")?;
            argmax_per_pos.eval().context("mtp_step: Step D eval")?;
            let preds_i32: &[i32] = argmax_per_pos.as_slice();
            let preds: Vec<u32> = preds_i32.iter().map(|x| *x as u32).collect();
            // DIAGNOSTIC (LUMEN_MTP_DEBUG=1): log drafter vs trunk preds to
            // identify which positions / kinds of tokens the drafter is
            // mis-predicting. Hot path — gated behind env var so production
            // runs aren't slowed by an std::env hit per step (cached via
            // OnceLock would be ideal; this is one-shot debugging so plain
            // env::var is fine).
            if std::env::var("LUMEN_MTP_DEBUG")
                .map(|v| v == "1")
                .unwrap_or(false)
            {
                eprintln!(
                    "[mtp_step m_pre={m_pre}] next_token={next_token} drafts={drafts:?} preds={preds:?}"
                );
            }
            // preds[k] = trunk's true prediction at position M+2+k.
            // drafts[k] = drafter's prediction at position M+2+k.
            let mut n_accepted = 0usize;
            for k in 0..n_draft {
                if preds[k] == drafts[k] {
                    n_accepted += 1;
                } else {
                    break;
                }
            }

            // === Step E: rollback cache on partial reject ===
            let target_offset = m_pre + 1 + 1 + n_accepted; // committed + next + accepted drafts
            let cur_offset = cache.offset();
            if cur_offset > target_offset {
                cache
                    .truncate_to(target_offset)
                    .context("mtp_step: Step E truncate")?;
            }

            // Build commit list. Last element is the next_input (correction
            // on partial reject, bonus on full accept) — NOT in cache yet.
            let mut committed = Vec::with_capacity(1 + n_accepted + 1);
            committed.push(next_token);
            committed.extend_from_slice(&drafts[..n_accepted]);
            committed.push(preds[n_accepted]);

            Ok(MtpStepOutput {
                committed,
                n_attempted: n_draft,
                n_accepted,
            })
        }

        /// Prompt-Lookup Decoding step. Same verify+rollback contract as
        /// [`Self::mtp_step`] but the Step B drafter is replaced by an
        /// n-gram lookup over `history` (CPU only, no GPU). Use when the
        /// model is *not* MTP-enabled or when you want a drafter-free
        /// speculative path.
        ///
        /// Contract: `history` MUST include the prompt plus every token
        /// committed so far (so the last n tokens form a valid prefix
        /// to search against). `committed_token` is the last entry in
        /// `history` and is NOT yet in the cache.
        ///
        /// Returns the same committed/n_attempted/n_accepted tuple as
        /// `mtp_step`. When no lookup match is found, falls back to a
        /// pure single-token decode (commits just `next_token`,
        /// `n_attempted == 0`).
        pub fn lookup_step(
            &self,
            cache: &mut NativeGemma4PromptCache,
            committed_token: u32,
            history: &[u32],
            n_lookup: usize,
            n_draft: usize,
        ) -> Result<MtpStepOutput> {
            if n_draft == 0 {
                return Err(anyhow!("lookup_step: n_draft must be >= 1"));
            }

            // Pre-state: cache offset = M (committed_token NOT yet in cache).
            let m_pre = cache.offset();

            // === Step A: trunk decode committed_token → next_token ===
            let step_a_input = Array::from_slice(&[committed_token], &[1, 1]);
            let logits_a = self
                .forward_array_last_token(&step_a_input, cache)
                .context("lookup_step: Step A trunk forward")?;
            let next_token = self
                .argmax_last_token(&logits_a)
                .context("lookup_step: Step A argmax next_token")?;
            // Cache offset is now M+1.

            // === Step B: n-gram lookup over (history + next_token) ===
            // Build the search corpus once: history (which already ends in
            // `committed_token`) + next_token. The lookup prefix is the
            // last `n_lookup` tokens of this corpus.
            let mut corpus = Vec::with_capacity(history.len() + 1);
            corpus.extend_from_slice(history);
            corpus.push(next_token);
            let drafts = find_lookup_draft(&corpus, n_lookup, n_draft);

            if std::env::var("LUMEN_LOOKUP_DEBUG")
                .map(|v| v == "1")
                .unwrap_or(false)
            {
                eprintln!(
                    "[lookup_step m_pre={m_pre}] next={next_token} drafts={drafts:?} \
                     (n_lookup={n_lookup}, n_draft={n_draft}, corpus_len={})",
                    corpus.len()
                );
            }

            // No match → fall back to single-token decode (cheap path,
            // no extra forward).
            if drafts.is_empty() {
                return Ok(MtpStepOutput {
                    committed: vec![next_token],
                    n_attempted: 0,
                    n_accepted: 0,
                });
            }

            // === Step C: trunk verify [next_token, drafts...] ===
            let mut verify_in: Vec<u32> = Vec::with_capacity(1 + drafts.len());
            verify_in.push(next_token);
            verify_in.extend_from_slice(&drafts);
            let verify_arr = Array::from_slice(&verify_in, &[1, verify_in.len() as i32]);
            let verify_logits = self
                .forward_array(&verify_arr, cache)
                .context("lookup_step: Step C trunk verify forward")?;
            // Cache offset is now M + 1 + (drafts.len() + 1) = M + drafts.len() + 2.

            // === Step D: accept-reject ===
            let argmax_per_pos =
                mlx_rs::ops::indexing::argmax_axis(&verify_logits, -1, /* keep_dims */ false)
                    .context("lookup_step: Step D argmax")?
                    .as_dtype(mlx_rs::Dtype::Int32)
                    .context("lookup_step: Step D cast to Int32")?;
            argmax_per_pos.eval().context("lookup_step: Step D eval")?;
            let preds_i32: &[i32] = argmax_per_pos.as_slice();
            let preds: Vec<u32> = preds_i32.iter().map(|x| *x as u32).collect();
            let mut n_accepted = 0usize;
            for k in 0..drafts.len() {
                if preds[k] == drafts[k] {
                    n_accepted += 1;
                } else {
                    break;
                }
            }

            // === Step E: rollback on partial reject ===
            let target_offset = m_pre + 1 + 1 + n_accepted;
            let cur_offset = cache.offset();
            if cur_offset > target_offset {
                cache
                    .truncate_to(target_offset)
                    .context("lookup_step: Step E truncate")?;
            }

            let mut committed = Vec::with_capacity(1 + n_accepted + 1);
            committed.push(next_token);
            committed.extend_from_slice(&drafts[..n_accepted]);
            committed.push(preds[n_accepted]);

            Ok(MtpStepOutput {
                committed,
                n_attempted: drafts.len(),
                n_accepted,
            })
        }

        pub fn config(&self) -> &NativeGemma4Config {
            &self.config
        }

        pub fn text_config(&self) -> &NativeGemma4TextConfig {
            &self.config.text_config
        }

        pub fn embed_tokens(&self) -> &EmbedTokensWeights {
            &self.embed_tokens
        }

        pub fn final_norm(&self) -> &Array {
            &self.final_norm
        }

        pub fn layers(&self) -> &[ResolvedGemma4LayerWeights] {
            &self.layers
        }

        pub fn num_layers(&self) -> usize {
            self.layers.len()
        }

        pub fn make_cache(&self) -> NativeGemma4PromptCache {
            NativeGemma4PromptCache::for_config(&self.config.text_config)
        }

        /// Allocate a fresh cache with explicit TurboQuant and simple-Q4
        /// on/off decisions. Used by `generate_with_cache` when adaptive
        /// modes (`LUMEN_GEMMA4_TQ_MODE=auto`, `LUMEN_GEMMA4_QUANT_KV_MODE=auto`)
        /// need to react to this request's prompt length. `None` for either
        /// defers to the env (legacy behaviour).
        pub fn make_cache_with_tq(
            &self,
            force_tq: Option<bool>,
            force_quant_kv: Option<bool>,
        ) -> NativeGemma4PromptCache {
            NativeGemma4PromptCache::for_config_with_tq(
                &self.config.text_config,
                force_tq,
                force_quant_kv,
            )
        }

        pub fn eos_tokens(&self) -> &[u32] {
            // For Gemma 4, the chat-aware EOS set lives at the **top level**
            // of config.json (typically `[1, 106, 50]` = <eos>, <turn|>,
            // <|tool_response>) while `text_config.eos_token_id` carries
            // only the bare `<eos>` (1). Prefer the top-level list when
            // it has at least as many entries, otherwise fall back to
            // text_config (covers slim configs that omit the top-level).
            let top = &self.config.eos_token_ids;
            let sub = &self.config.text_config.eos_token_ids;
            if top.len() >= sub.len() && !top.is_empty() {
                top
            } else {
                sub
            }
        }

        pub fn vocab_size(&self) -> usize {
            self.config.text_config.vocab_size
        }

        /// Single quantized-linear application. Wraps
        /// `quantized_matmul_with_mode(transpose=true, …)`.
        ///
        /// bf16-throughout residual stream is canonical
        /// (gate-on parity with mlx-lm at long context). The legacy
        /// `LUMEN_GEMMA4_NO_F32_CAST=0` f32-cast fallback was removed
        /// 2026-05-14 after 6+ months of dormancy. History in
        /// `phase_1_6_regression_resolved_missing_gate.md`.
        fn qmatmul(lin: &ResolvedGemma4QuantLinear, x: &Array) -> Result<Array> {
            quantized_matmul_with_mode(
                x,
                &lin.weight,
                &lin.scales,
                lin.biases.as_ref(),
                /* transpose */ true,
                lin.group_size,
                lin.bits,
                lin.mode,
            )
            .context("qmatmul: quantized_matmul_with_mode failed")
        }

        /// Mirrors mlx-lm's `RMSNormNoScale` (used for Gemma 4's `v_norm`):
        /// `y = x * rsqrt(mean(x*x, axis=-1) + eps)` — no learnable scale.
        ///
        /// uses the pre-allocated `const_ones_head_dim_*`
        /// Array cached at model load. Previous impl allocated a fresh
        /// `ones([head_dim])` on every call — 30 per decode step (one
        /// per attention layer).
        fn rms_norm_no_scale(&self, x: &Array, head_dim: i32, eps: f32) -> Result<Array> {
            let ones = if head_dim == self.config.text_config.global_head_dim as i32 {
                &self.const_ones_head_dim_full
            } else {
                &self.const_ones_head_dim_sliding
            };
            rms_norm(x, ones, eps).context("rms_norm_no_scale: rms_norm failed")
        }

        /// Gemma 4 single-layer attention forward, dispatching on layer kind
        /// (sliding vs full) and the `attention_k_eq_v` short-circuit for
        /// full-attention layers.
        ///
        /// Mirrors mlx_lm's `Gemma4TextModel.Attention.__call__`.
        pub fn layer_attention_forward(
            &self,
            x: &Array,
            layer_idx: usize,
            cache: &mut NativeGemma4LayerCache,
        ) -> Result<Array> {
            if x.ndim() != 3 {
                return Err(anyhow!(
                    "layer_attention_forward: expected x rank 3 [B, L, hidden], got ndim={}",
                    x.ndim()
                ));
            }
            let cfg = &self.config.text_config;
            let lw = &self.layers[layer_idx];
            let kind = lw.kind;

            let b = x.shape()[0];
            let l = x.shape()[1];
            let n_heads = cfg.num_attention_heads as i32;
            let n_kv = cfg.n_kv_heads_for(kind) as i32;
            let head_dim = cfg.head_dim_for(kind) as i32;
            let eps = cfg.rms_norm_eps;

            let rope_params = cfg.rope_for(kind);
            let rope_dim = (head_dim as f32 * rope_params.partial_rotary_factor) as i32;
            let rope_base = rope_params.rope_theta;
            let kv_offset = cache.offset() as i32;

            // Sub-stage CPU dispatch timing (honest mode only). Skipped
            // when no breakdown is active — single Instant::now() per
            // sub-stage is ~50 ns, negligible vs the FFI cost we time.
            let time_substages = gemma4_any_breakdown_active();

            // (1+2+3) Q/K/V projections (3 qmatmul + 3 reshape).
            let qkvo_start = if time_substages {
                Some(Instant::now())
            } else {
                None
            };
            let q_raw = Self::qmatmul(&lw.attn.q_proj, x)?;
            let q = mlx_rs::ops::reshape(&q_raw, &[b, l, n_heads, head_dim])
                .context("attn: reshape Q failed")?;
            // q_norm timed under norms bucket below.
            let k_raw = Self::qmatmul(&lw.attn.k_proj, x)?;
            let k_4d = mlx_rs::ops::reshape(&k_raw, &[b, l, n_kv, head_dim])
                .context("attn: reshape K failed")?;
            let v_4d = match &lw.attn.v_proj {
                Some(v_proj) => {
                    let v_raw = Self::qmatmul(v_proj, x)?;
                    mlx_rs::ops::reshape(&v_raw, &[b, l, n_kv, head_dim])
                        .context("attn: reshape V failed")?
                }
                None => k_4d.clone(),
            };
            if let Some(t0) = qkvo_start {
                bump_gemma4_attn_qkvo_ms(t0.elapsed().as_secs_f64() * 1e3);
            }

            // (4) Norms on Q/K/V + 3× transpose. Bucketed as "norms" since
            // transpose_axes is metadata-only (constant cost vs ctx).
            //
            // Phase 1.8 M2.5 (`LUMEN_GEMMA4_FUSED_QKNORM=1`) fused
            // `rms_norm + transpose_axes` into one Metal dispatch, but
            // bridge-dispatch sync cost overwhelmed the fusion savings
            // (+34 ms regression at PROMPT_LEN=4096). Removed 2026-05-14;
            // future fusion work must use the M4.8 mlx Primitive pattern,
            // not the bridge.
            let norms_start = if time_substages {
                Some(Instant::now())
            } else {
                None
            };

            let q_n = rms_norm(&q, &lw.attn.q_norm, eps).context("attn: q_norm")?;
            let k_n = rms_norm(&k_4d, &lw.attn.k_norm, eps).context("attn: k_norm")?;
            let q_t = mlx_rs::ops::transpose_axes(&q_n, &[0, 2, 1, 3])
                .context("attn: transpose Q failed")?;
            let k_t = mlx_rs::ops::transpose_axes(&k_n, &[0, 2, 1, 3])
                .context("attn: transpose K failed")?;

            let v_n = self.rms_norm_no_scale(&v_4d, head_dim, eps)?;
            let v_t = mlx_rs::ops::transpose_axes(&v_n, &[0, 2, 1, 3])
                .context("attn: transpose V failed")?;
            if let Some(t0) = norms_start {
                bump_gemma4_attn_norms_ms(t0.elapsed().as_secs_f64() * 1e3);
            }

            // (5) RoPE on Q and K (partial-rotary, per-kind theta + dim).
            // documented `LUMEN_GEMMA4_ROPE_PRECOMPUTE_FREQS=1`
            // as WASH (no perf gain vs per-call arange+pow); precompute path
            // removed 2026-05-14.
            let rope_start = if time_substages {
                Some(Instant::now())
            } else {
                None
            };
            // Full attention layers use mlx_lm's `ProportionalRoPE`: freqs are
            // computed against the FULL head_dim (denominator = head_dim, NOT
            // the partial rope_dim), with `(head_dim - rope_dim)/2` trailing
            // entries of `inf` so the un-rotated tail still gets a freq slot
            // but `1/inf=0` makes its rotation a no-op. Sliding layers have
            // `partial_rotary_factor=1.0` so the default freqs-less path
            // remains correct (denominator = rope_dim = head_dim).
            //
            // Pre-fix: we called `rope(..., dimensions=rope_dim, base, ...)`
            // for both kinds. For full attention that put the freq exponent
            // denominator at `rope_dim` (128) instead of `head_dim` (512),
            // producing wrong rotation frequencies. Prefill drift was masked
            // by bf16 noise; decode position accumulation exposed it as a
            // max|Δ|=15 spike at L29 (the only full-attention layer in 26B-A4B).
            let (q_rope, k_rope) = if rope_dim == head_dim {
                let q = rope(&q_t, rope_dim, false, rope_base, 1.0, kv_offset)
                    .context("attn: rope(Q)")?;
                let k = rope(&k_t, rope_dim, false, rope_base, 1.0, kv_offset)
                    .context("attn: rope(K)")?;
                (q, k)
            } else {
                let half_rot = (rope_dim / 2) as usize;
                let half_inf = ((head_dim - rope_dim) / 2) as usize;
                let mut freqs_vals: Vec<f32> = Vec::with_capacity(half_rot + half_inf);
                for i in 0..half_rot {
                    let exp = (2 * i) as f32 / head_dim as f32;
                    freqs_vals.push(rope_base.powf(exp));
                }
                for _ in 0..half_inf {
                    freqs_vals.push(f32::INFINITY);
                }
                let freqs = Array::from_slice(&freqs_vals, &[freqs_vals.len() as i32]);
                let q = rope_with_freqs(&q_t, head_dim, false, 1.0, kv_offset, &freqs)
                    .context("attn: rope_with_freqs(Q)")?;
                let k = rope_with_freqs(&k_t, head_dim, false, 1.0, kv_offset, &freqs)
                    .context("attn: rope_with_freqs(K)")?;
                (q, k)
            };
            if let Some(t0) = rope_start {
                bump_gemma4_attn_rope_ms(t0.elapsed().as_secs_f64() * 1e3);
            }

            // (6) Cache update.
            // Tier 1B (2026-05-16) — quantized full-attn early branch.
            // When cache is FullQuantized, do the entire SDPA inline using
            // mlx::quantized_matmul + softmax + quantized_matmul. Bypasses
            // the bf16 dispatch tree below entirely and jumps straight to
            // the o_proj path with the resulting attn_out.
            let cache_start = if time_substages {
                Some(Instant::now())
            } else {
                None
            };
            if let NativeGemma4LayerCache::FullQuantized(c) = cache {
                let (kt, vt) = c.update_and_fetch(&k_rope, &v_t)?;
                let gs = c.group_size();
                let bits = c.bits();
                // Capture kv_actual before moving kt into reshape branch.
                let kv_actual_q = kt.0.shape()[2] as usize;
                if let Some(t0) = cache_start {
                    bump_gemma4_attn_cache_ms(t0.elapsed().as_secs_f64() * 1e3);
                }
                // Scale Q by 1.0 (Gemma 4 SDPA scale convention — already
                // baked into the post-RoPE Q via the kernel scale arg=1.0).
                // GQA reshape: n_repeats = n_heads / n_kv. Q: [B, n_heads, L, D] →
                // [B, n_kv, n_repeats, L, D]. K/V tuples: each Array gets
                // expand_dims at axis -3 → [B, n_kv, 1, S, ...] for broadcast.
                let b_i = b;
                let l_i = l;
                let n_heads_i = n_heads;
                let n_kv_i = n_kv;
                let head_dim_i = head_dim;
                let n_repeats = n_heads_i / n_kv_i;
                let (q_for_qmm, k_tuple_for_qmm, v_tuple_for_qmm, needs_reshape) = if n_repeats > 1
                {
                    let q_reshaped =
                        mlx_rs::ops::reshape(&q_rope, &[b_i, n_kv_i, n_repeats, l_i, head_dim_i])
                            .context("qkv_quant: reshape Q for GQA")?;
                    let exp = |a: &Array| -> Result<Array> {
                        mlx_rs::ops::expand_dims(a, -3).context("qkv_quant: expand_dims(K/V, -3)")
                    };
                    let kt2 = (exp(&kt.0)?, exp(&kt.1)?, exp(&kt.2)?);
                    let vt2 = (exp(&vt.0)?, exp(&vt.1)?, exp(&vt.2)?);
                    (q_reshaped, kt2, vt2, true)
                } else {
                    (q_rope.clone(), kt, vt, false)
                };
                let sdpa_start = if time_substages {
                    Some(Instant::now())
                } else {
                    None
                };
                // scores = Q @ K^T  (Q is bf16, K is quantized)
                let scores = mlx_rs::ops::quantized_matmul(
                    &q_for_qmm,
                    &k_tuple_for_qmm.0,
                    &k_tuple_for_qmm.1,
                    Some(&k_tuple_for_qmm.2),
                    /* transpose */ true,
                    /* group_size */ gs,
                    /* bits */ bits,
                )
                .context("qkv_quant: quantized_matmul(Q, K)")?;
                // Apply causal mask if multi-token (prefill); decode L=1 needs no mask.
                let scores_masked = if (l_i as usize) > 1 {
                    let mask = make_attention_mask_for_layer_chunked(
                        kind,
                        cfg,
                        l_i as usize,
                        kv_offset as usize,
                        kv_actual_q,
                    )?;
                    match mask {
                        Some(m) => mlx_rs::ops::add(&scores, &m).context("qkv_quant: add mask")?,
                        None => scores,
                    }
                } else {
                    scores
                };
                // Precise softmax along last axis.
                let last_axis = (scores_masked.ndim() as i32) - 1;
                let scores_sm = mlx_rs::ops::softmax_axis(
                    &scores_masked,
                    last_axis,
                    /* precise */ Some(true),
                )
                .context("qkv_quant: softmax")?;
                // out = scores @ V
                let out = mlx_rs::ops::quantized_matmul(
                    &scores_sm,
                    &v_tuple_for_qmm.0,
                    &v_tuple_for_qmm.1,
                    Some(&v_tuple_for_qmm.2),
                    /* transpose */ false,
                    /* group_size */ gs,
                    /* bits */ bits,
                )
                .context("qkv_quant: quantized_matmul(scores, V)")?;
                let attn_out_q = if needs_reshape {
                    mlx_rs::ops::reshape(&out, &[b_i, n_heads_i, l_i, head_dim_i])
                        .context("qkv_quant: reshape output back to [B, n_heads, L, D]")?
                } else {
                    out
                };
                if let Some(t0) = sdpa_start {
                    bump_gemma4_attn_sdpa_ms(t0.elapsed().as_secs_f64() * 1e3);
                }

                // Jump directly to o_proj path (skipping the bf16 dispatch tree).
                let oproj_start = if time_substages {
                    Some(Instant::now())
                } else {
                    None
                };
                let attn_t = mlx_rs::ops::transpose_axes(&attn_out_q, &[0, 2, 1, 3])
                    .context("qkv_quant: transpose output")?;
                let attn_flat = mlx_rs::ops::reshape(&attn_t, &[b_i, l_i, n_heads_i * head_dim_i])
                    .context("qkv_quant: reshape output flat")?;
                let out_final = Self::qmatmul(&lw.attn.o_proj, &attn_flat)?;
                if let Some(t0) = oproj_start {
                    bump_gemma4_attn_qkvo_ms(t0.elapsed().as_secs_f64() * 1e3);
                }
                return Ok(out_final);
            }
            // Tier 1B sliding (2026-05-17) — quantized sliding-window early
            // branch. Mirrors the FullQuantized path above, swapping in the
            // rotating quantized cache. The mask builder
            // `make_attention_mask_for_layer_chunked` already covers the
            // sliding kind; we just hand it the same `kind` arg as the bf16
            // path. Skip the mask Array build when L==1 (decode).
            if let NativeGemma4LayerCache::SlidingQuantized(c) = cache {
                // TurboQuant rotation lever — optionally pre-rotate K (and Q
                // symmetrically) by a Haar orthogonal R [D,D]. Inner products
                // are preserved (R orthogonal) but K's per-coordinate
                // distribution becomes ~N(0,1), which is exactly what mlx's
                // affine quant (and Lloyd-Max) optimize for → less quant error
                // at the same bit budget. V is NOT rotated so the SDPA output
                // remains in original head_dim space for o_proj.
                let rotate_enabled = gemma4_quant_kv_sliding_rotate_enabled();
                let (k_to_cache, q_for_rotation) = if rotate_enabled {
                    let r = crate::turboquant::rotation_matrix_f32(
                        head_dim as usize,
                        crate::turboquant::TURBOQUANT_SEED,
                    )?;
                    let kr = crate::turboquant::rotate_last_axis(&k_rope, &r)?;
                    let qr = crate::turboquant::rotate_last_axis(&q_rope, &r)?;
                    (kr, qr)
                } else {
                    (k_rope.clone(), q_rope.clone())
                };
                let (kt, vt) = c.update_and_fetch(&k_to_cache, &v_t)?;
                let gs = c.group_size();
                let bits = c.bits();
                // Capture kv_actual_q (post-rotation K-length) BEFORE moving
                // the triples into the GQA reshape branch.
                let kv_actual_q = kt.0.shape()[2] as usize;
                if let Some(t0) = cache_start {
                    bump_gemma4_attn_cache_ms(t0.elapsed().as_secs_f64() * 1e3);
                }
                let b_i = b;
                let l_i = l;
                let n_heads_i = n_heads;
                let n_kv_i = n_kv;
                let head_dim_i = head_dim;
                let n_repeats = n_heads_i / n_kv_i;
                let (q_for_qmm, k_tuple_for_qmm, v_tuple_for_qmm, needs_reshape) = if n_repeats > 1
                {
                    let q_reshaped = mlx_rs::ops::reshape(
                        &q_for_rotation,
                        &[b_i, n_kv_i, n_repeats, l_i, head_dim_i],
                    )
                    .context("qkv_quant_sliding: reshape Q for GQA")?;
                    let exp = |a: &Array| -> Result<Array> {
                        mlx_rs::ops::expand_dims(a, -3)
                            .context("qkv_quant_sliding: expand_dims(K/V, -3)")
                    };
                    let kt2 = (exp(&kt.0)?, exp(&kt.1)?, exp(&kt.2)?);
                    let vt2 = (exp(&vt.0)?, exp(&vt.1)?, exp(&vt.2)?);
                    (q_reshaped, kt2, vt2, true)
                } else {
                    (q_for_rotation, kt, vt, false)
                };
                let sdpa_start = if time_substages {
                    Some(Instant::now())
                } else {
                    None
                };
                // scores = Q @ K^T  (Q is bf16, K is quantized)
                let scores = mlx_rs::ops::quantized_matmul(
                    &q_for_qmm,
                    &k_tuple_for_qmm.0,
                    &k_tuple_for_qmm.1,
                    Some(&k_tuple_for_qmm.2),
                    /* transpose */ true,
                    /* group_size */ gs,
                    /* bits */ bits,
                )
                .context("qkv_quant_sliding: quantized_matmul(Q, K)")?;
                // Causal + sliding mask only at prefill (L > 1). Decode L==1
                // sees only the in-window tokens that the cache holds, and
                // SDPA over a permutation-invariant K set produces the same
                // output regardless of ring order — no mask needed.
                let scores_masked = if (l_i as usize) > 1 {
                    let mask = make_attention_mask_for_layer_chunked(
                        kind,
                        cfg,
                        l_i as usize,
                        kv_offset as usize,
                        kv_actual_q,
                    )?;
                    match mask {
                        Some(m) => {
                            mlx_rs::ops::add(&scores, &m).context("qkv_quant_sliding: add mask")?
                        }
                        None => scores,
                    }
                } else {
                    scores
                };
                let last_axis = (scores_masked.ndim() as i32) - 1;
                let scores_sm = mlx_rs::ops::softmax_axis(
                    &scores_masked,
                    last_axis,
                    /* precise */ Some(true),
                )
                .context("qkv_quant_sliding: softmax")?;
                let out = mlx_rs::ops::quantized_matmul(
                    &scores_sm,
                    &v_tuple_for_qmm.0,
                    &v_tuple_for_qmm.1,
                    Some(&v_tuple_for_qmm.2),
                    /* transpose */ false,
                    /* group_size */ gs,
                    /* bits */ bits,
                )
                .context("qkv_quant_sliding: quantized_matmul(scores, V)")?;
                let attn_out_q = if needs_reshape {
                    mlx_rs::ops::reshape(&out, &[b_i, n_heads_i, l_i, head_dim_i])
                        .context("qkv_quant_sliding: reshape output back")?
                } else {
                    out
                };
                if let Some(t0) = sdpa_start {
                    bump_gemma4_attn_sdpa_ms(t0.elapsed().as_secs_f64() * 1e3);
                }
                let oproj_start = if time_substages {
                    Some(Instant::now())
                } else {
                    None
                };
                let attn_t = mlx_rs::ops::transpose_axes(&attn_out_q, &[0, 2, 1, 3])
                    .context("qkv_quant_sliding: transpose output")?;
                let attn_flat = mlx_rs::ops::reshape(&attn_t, &[b_i, l_i, n_heads_i * head_dim_i])
                    .context("qkv_quant_sliding: reshape output flat")?;
                let out_final = Self::qmatmul(&lw.attn.o_proj, &attn_flat)?;
                if let Some(t0) = oproj_start {
                    bump_gemma4_attn_qkvo_ms(t0.elapsed().as_secs_f64() * 1e3);
                }
                return Ok(out_final);
            }
            // TurboQuant Stage-1 sliding cache early branch — rotate K (and Q)
            // by Haar orthogonal R, quantize K via Lloyd-Max nearest-centroid
            // against fixed N(0,1) codebook + per-vector σ. V uses the same
            // Lloyd-Max + σ but WITHOUT rotation (stays in original head_dim
            // space so SDPA output flows straight to o_proj). Decode dispatch
            // uses bf16 SDPA on dequantized K/V (not quantized_matmul, which
            // doesn't fit Lloyd-Max codebook semantics).
            let tq_cache = match cache {
                NativeGemma4LayerCache::SlidingTurboquant(c) => Some(c),
                NativeGemma4LayerCache::FullTurboquant(c) => Some(c),
                _ => None,
            };
            if let Some(c) = tq_cache {
                let centroids = crate::turboquant::lloyd_max_centroids(c.bits())?;

                // ── Diagnostic env gates (Stage 1 root-cause bisection) ──
                //
                // V1 LUMEN_GEMMA4_TQ_DIAG_NO_ROTATE=1
                //     Skip Haar rotation; K/Q stay in RoPE space. RESULT: still
                //     garbage → rotation is NOT the culprit.
                //
                // V2 LUMEN_GEMMA4_TQ_DIAG_V_BF16=1
                //     Skip V quantization — V flows through SDPA as the raw bf16
                //     v_t tensor. Cache still stores quantized V codes/sigma but
                //     they're ignored at fetch time. Isolates V-quant impact.
                //
                // V3 LUMEN_GEMMA4_TQ_DIAG_K_BF16=1
                //     Skip K quantization — K flows through SDPA as raw bf16
                //     k_rot tensor. Symmetric to V2 for K-side isolation.
                //
                // V4 = V2 + V3 simultaneously → full bf16 passthrough through
                //     the SlidingTurboquant branch (still using cache push but
                //     ignoring the quantized fetch). If this still produces
                //     garbage, the bug is in the SDPA call / shape / mask wiring,
                //     not the quant math itself.
                // Haar rotation on Q/K AND V before Lloyd-Max encoding.
                //
                // V rotation is *required* — without it Lloyd-Max (N(0,1)-
                // fitted codebook) is applied to V's raw, non-Gaussian
                // per-coordinate distribution and produces garbage output
                // at every bits level (8/6/4/3) on real prompts. Rotation
                // Gaussianizes per-vector via CLT across D head_dim
                // elements; the codebook then matches the post-rotation
                // distribution and bits=4 reaches bit-identical output vs
                // bf16 baseline (2026-05-17 diagnosis).
                //
                // V is rotated → encoded → dequantized in rotated space →
                // un-rotated (× Rᵀ) before SDPA's softmax·V step so
                // attention output lands back in head_dim space.
                //
                // Escape hatches for diagnostics (default OFF):
                //   LUMEN_GEMMA4_TQ_SKIP_K_ROTATE=1  K + Q skip rotation
                //   LUMEN_GEMMA4_TQ_SKIP_V_ROTATE=1  V skips rotation (
                //                                    reverts to pre-fix
                //                                    broken Stage 1)
                let skip_k_rotate = std::env::var("LUMEN_GEMMA4_TQ_SKIP_K_ROTATE")
                    .map(|s| s == "1")
                    .unwrap_or(false);
                let skip_v_rotate = std::env::var("LUMEN_GEMMA4_TQ_SKIP_V_ROTATE")
                    .map(|s| s == "1")
                    .unwrap_or(false);
                // When R is baked into Wv at load time, V comes out of
                // v_proj already in rotated space → skip the runtime
                // rotation. When R is baked into Wo, V_dq un-rotation is
                // absorbed there → skip the runtime un-rotation. Each
                // leg is independently gated so we can isolate which side
                // of the bake is responsible if quality regresses.
                // Bake state is per-LAYER-KIND. The load-time bake loop
                // applies the Wv/Wo rotation only to sliding-attention layers
                // (see `bake_r_into_v_proj` / `bake_r_into_o_proj` callsites);
                // full-attention layers' Wv/Wo are untouched. So for the
                // FullTurboquant branch, ignore the model-wide bake flags —
                // V must be rotated at runtime AND V_dq must be un-rotated
                // before o_proj (Wo has no R baked in to absorb it).
                let is_sliding_layer = matches!(kind, NativeGemma4LayerType::SlidingAttention);
                let bake_v = self.tq_bake_v_active && is_sliding_layer;
                // bake_o applies to:
                //   sliding layers when tq_bake_o_active (default ON with TQ)
                //   full-attn layers when tq_bake_o_full_attn_active (opt-in,
                //     default OFF). Both require tq_bake_o_active (the runtime
                //     state machine that consumes baked-Wo is shared logic).
                let bake_o =
                    self.tq_bake_o_active && (is_sliding_layer || self.tq_bake_o_full_attn_active);
                let rotate_v = !skip_v_rotate && !bake_v;
                let unrotate_v_dq = !skip_v_rotate && !bake_o;

                let r_arr = crate::turboquant::rotation_matrix_f32(
                    head_dim as usize,
                    crate::turboquant::TURBOQUANT_SEED,
                )?;

                // QJL on/off is needed BEFORE the K encode dispatch because
                // K rot+encode fusion is only safe when QJL Stage-2 is off
                // (Stage-2 reads k_rot to compute the residual K - K_dq, and
                // the fused kernel doesn't materialize k_rot).
                let qjl_m = c.qjl_m();

                // Lloyd-Max quantize K (rotated) and V (un-rotated). Fused
                // kernel collapses σ + normalize + encode into one dispatch;
                // env gate `LUMEN_GEMMA4_TQ_FUSED_ENCODE=0` falls back to the
                // non-fused multi-op chain for A/B comparison.
                //
                // **Super-fusion gates (default OFF — A/B opt-in)**:
                //
                // `LUMEN_GEMMA4_TQ_FUSED_VROT=1` collapses
                //   `(V @ R) → σ → encode` into one Metal kernel. Skips the
                //   bf16 V_rot intermediate plus the matmul + 2 cast
                //   dispatches per layer. NEGATIVE A/B 2026-05-17 at 8K
                //   showed −5% (mlx steel_gemm beat the in-kernel naive
                //   per-thread dot product). Worth retesting at long ctx
                //   where dispatch overhead dominates.
                //
                // `LUMEN_GEMMA4_TQ_FUSED_KROT=1` (added 2026-05-25) does the
                //   same for K — only valid when QJL Stage-2 is OFF (Stage-2
                //   reads k_rot for the residual; fused kernel discards it).
                //   Saves matmul + 2 casts × 25 sliding layers per step
                //   when long-context decode is dispatch-bound.
                //
                // Both retained behind env gates so future iterations
                // (simdgroup_matrix tiles, register-tile GEMM) can re-attempt
                // the trade without re-plumbing the SDPA path.
                let use_fused = std::env::var("LUMEN_GEMMA4_TQ_FUSED_ENCODE")
                    .map(|s| s != "0")
                    .unwrap_or(true);
                let fuse_k_rotate = use_fused
                    && !skip_k_rotate
                    && qjl_m.is_none()
                    && std::env::var("LUMEN_GEMMA4_TQ_FUSED_KROT")
                        .map(|s| s == "1")
                        .unwrap_or(false);
                let fuse_v_rotate = use_fused
                    && rotate_v
                    && std::env::var("LUMEN_GEMMA4_TQ_FUSED_VROT")
                        .map(|s| s == "1")
                        .unwrap_or(false);

                // Q is always rotated separately (its result feeds qk_inline /
                // SDPA, not the encode kernel). When `fuse_k_rotate` is on,
                // k_rot is not materialized — only Q rotation runs through the
                // explicit `rotate_last_axis` path.
                let (k_rot_opt, q_rot) = if skip_k_rotate {
                    (Some(k_rope.clone()), q_rope.clone())
                } else if fuse_k_rotate {
                    // k_rot stays inside the fused encode kernel.
                    let q_rot = crate::turboquant::rotate_last_axis(&q_rope, &r_arr)?;
                    (None, q_rot)
                } else {
                    let k_rot = crate::turboquant::rotate_last_axis(&k_rope, &r_arr)?;
                    let q_rot = crate::turboquant::rotate_last_axis(&q_rope, &r_arr)?;
                    (Some(k_rot), q_rot)
                };

                let (k_codes, k_sigma) = if fuse_k_rotate {
                    // (K @ R) + σ + encode in one Metal kernel.
                    crate::turboquant::rotate_and_lloyd_max_quantize_stage1_fused(
                        &k_rope, &r_arr, &centroids,
                    )?
                } else {
                    let k_rot = k_rot_opt
                        .as_ref()
                        .expect("k_rot must be materialized when fuse_k_rotate=false");
                    if use_fused {
                        crate::turboquant::lloyd_max_quantize_stage1_fused(k_rot, &centroids)?
                    } else {
                        crate::turboquant::lloyd_max_quantize_stage1(k_rot, &centroids)?
                    }
                };

                let (v_codes, v_sigma) = if fuse_v_rotate {
                    // (V @ R) + σ + encode in one Metal kernel.
                    crate::turboquant::rotate_and_lloyd_max_quantize_stage1_fused(
                        &v_t, &r_arr, &centroids,
                    )?
                } else {
                    let v_for_encode = if rotate_v {
                        crate::turboquant::rotate_last_axis(&v_t, &r_arr)?
                    } else {
                        v_t.clone()
                    };
                    if use_fused {
                        crate::turboquant::lloyd_max_quantize_stage1_fused(
                            &v_for_encode,
                            &centroids,
                        )?
                    } else {
                        crate::turboquant::lloyd_max_quantize_stage1(&v_for_encode, &centroids)?
                    }
                };

                // QJL Stage-2 path: K-side only. Compute residual signs +
                // ‖r‖ at encode time (against locally-dequantized K), push
                // into the QJL-extended ring, and add the QJL correction
                // to K_dq before SDPA. The math (see turboquant.rs) shows
                // K_eff = K_dq + ‖r‖·√(π/2)/√m·Φᵀb_k makes
                // <q, K_eff> = <q, K_dq> + <q, r>, so standard SDPA on
                // K_eff yields the unbiased estimator.
                //
                // Stage 2 inline Q@K_codes: when this is a decode step
                // (l==1) AND QJL is off, skip the K_dq materialization
                // entirely and compute scores directly via the custom
                // `lumen_tq_qk_inline` Metal kernel (centroids LUT +
                // per-K-vector σ inline). Eliminates the 32 MB K_dq DRAM
                // write at 8K sliding context. V_dq is still materialized
                // (Stage 3 covers V). **Default ON** (opt-out
                // `LUMEN_GEMMA4_TQ_QK_INLINE=0`) — 3-pair cool A/B at 8K
                // showed +21% decode (37.4 → 45.2 tok/s) vs (TQ Stage 1 +
                // Wo bake) baseline. Inline-correctness gate: unit test
                // `qk_inline_matches_dequant_then_matmul` (rel-MSE < 1e-3).
                let qk_inline_enabled = std::env::var("LUMEN_GEMMA4_TQ_QK_INLINE")
                    .map(|s| s != "0")
                    .unwrap_or(true);
                // Stage 3 (`lumen_tq_sv_inline`) is **default OFF** after a
                // NEGATIVE A/B (2026-05-17): -20% decode at 8K vs Stage-2-only,
                // consistent across 3 cool pairs. The "one thread per D
                // output, serial N loop" pattern undershoots mlx's tile-MMA
                // matmul on the (attn_w @ V_dq) shape — only 16 TGs total at
                // decode (B·H), ~13% GPU occupancy, hot loop fully serial in
                // each thread. mlx's matmul parallelizes both D output and N
                // reduction via register-tile MMA → much better fit on this
                // matmul shape. Net: the V_dq materialization cost saved by
                // the inline kernel is smaller than the parallelism it
                // sacrifices. Future iteration: split N across simdgroups
                // (qvm-style with two-pass reduce) or port mlx's tile MMA
                // pattern. Keep the kernel/plumbing in tree for that retry.
                let sv_inline_enabled = std::env::var("LUMEN_GEMMA4_TQ_SV_INLINE")
                    .map(|s| s == "1")
                    .unwrap_or(false);
                // Full-attn-only variant: sliding D=256 sv_inline is prior
                // NEGATIVE (B*H = 32 → ~13% GPU occupancy underutilizes the
                // SM array vs mlx tile-MMA matmul), but full-attn D=512 with
                // 4× larger V_dq makes materialize cost dominate. This env
                // lets the runtime activate sv_inline only on full-attn
                // layers, sidestepping the sliding regression.
                let sv_inline_full_only = std::env::var("LUMEN_GEMMA4_TQ_SV_INLINE_FULL")
                    .map(|s| s == "1")
                    .unwrap_or(false);
                let sv_inline_for_kind =
                    sv_inline_enabled || (sv_inline_full_only && !is_sliding_layer);
                // `turboquant_qk_inline` D=256 (sliding head_dim) is the
                // shipping inline path.
                //
                // A D=512 variant (`lumen_tq_qk_inline_d512`, VPT=16) is
                // landed in the mlx fork (full-attn global_head_dim=512) but
                // currently crashes at runtime — root cause not yet
                // identified. The full-attn TQ branch falls back to the
                // materialize K_dq + mlx matmul path until that kernel is
                // debugged. Opt-in flag exists for future re-enable:
                //   LUMEN_GEMMA4_TQ_QK_INLINE_D512=1
                let qk_inline_d512_enabled = std::env::var("LUMEN_GEMMA4_TQ_QK_INLINE_D512")
                    .map(|s| s == "1")
                    .unwrap_or(false);
                let inline_kernel_eligible =
                    head_dim == 256 || (head_dim == 512 && qk_inline_d512_enabled);
                let do_inline = qk_inline_enabled
                    && (l as usize) == 1
                    && qjl_m.is_none()
                    && inline_kernel_eligible;
                // Stage 3 (`turboquant_sv_inline`) requires Stage 2 active +
                // V stays in rotated space (Wo bake absorbs the un-rotation).
                // When `unrotate_v_dq=true` the runtime applies Rᵀ to V_dq
                // before SDPA — the inline kernel can't do that, so fall back
                // to materialized V on that path.
                let do_sv_inline = do_inline && sv_inline_for_kind && !unrotate_v_dq;
                let (kv_actual_tq, k_dq_opt, v_dq_opt, kc_inline, ks_inline, vc_inline, vs_inline) =
                    if let Some(m) = qjl_m {
                        // Local Stage-1 K dequant for residual computation.
                        let k_dq_local = crate::turboquant::lloyd_max_dequantize_scaled(
                            &k_codes, &k_sigma, &centroids,
                        )?;
                        let qjl_proj = crate::turboquant::qjl_projection_matrix_f32(
                            head_dim as usize,
                            m,
                            crate::turboquant::TURBOQUANT_SEED,
                        )?;
                        // Packed encode: signs stored as u32 [..., ceil(m/32)]
                        // — 16× smaller than the bf16 ±1 [..., m] equivalent.
                        // `fuse_k_rotate` is forced off when QJL is on (see
                        // gating above), so k_rot_opt is always Some here.
                        let k_rot_ref = k_rot_opt
                            .as_ref()
                            .expect("QJL Stage-2 requires materialized k_rot");
                        let (k_signs_packed, k_rnorm) =
                            crate::turboquant::qjl_encode_stage2_packed(
                                k_rot_ref,
                                &k_dq_local,
                                &qjl_proj,
                            )?;

                        let ((kc, ks, ksg, krn), (vc, vs)) = c.update_and_fetch_qjl(
                            &k_codes,
                            &k_sigma,
                            &k_signs_packed,
                            &k_rnorm,
                            &v_codes,
                            &v_sigma,
                        )?;
                        let kv_actual_tq = kc.shape()[2] as usize;
                        if let Some(t0) = cache_start {
                            bump_gemma4_attn_cache_ms(t0.elapsed().as_secs_f64() * 1e3);
                        }
                        let k_dq =
                            crate::turboquant::lloyd_max_dequantize_scaled(&kc, &ks, &centroids)?;
                        let v_dq =
                            crate::turboquant::lloyd_max_dequantize_scaled(&vc, &vs, &centroids)?;
                        let k_eff = crate::turboquant::qjl_apply_correction_packed(
                            &k_dq, &ksg, &krn, &qjl_proj, m,
                        )?;
                        // If V was rotated for encode, un-rotate V_dq back into
                        // the original head_dim space: V ≈ V_rot · Rᵀ.
                        // Skipped when R is baked into Wo (un-rotation absorbed
                        // there).
                        let v_dq = if unrotate_v_dq {
                            let r_t = mlx_rs::ops::transpose_axes(&r_arr, &[1, 0])
                                .context("tq_sliding: Rᵀ for V un-rotate")?;
                            crate::turboquant::rotate_last_axis(&v_dq, &r_t)?
                        } else {
                            v_dq
                        };
                        (
                            kv_actual_tq,
                            Some(k_eff),
                            Some(v_dq),
                            None,
                            None,
                            None,
                            None,
                        )
                    } else {
                        let ((kc, ks), (vc, vs)) =
                            c.update_and_fetch(&k_codes, &k_sigma, &v_codes, &v_sigma)?;
                        let kv_actual_tq = kc.shape()[2] as usize;
                        if let Some(t0) = cache_start {
                            bump_gemma4_attn_cache_ms(t0.elapsed().as_secs_f64() * 1e3);
                        }
                        // Materialize K_dq only when NOT taking the Stage-2 inline
                        // path. When `do_inline` is true the custom kernel reads
                        // K_codes (uint8) + K_sigma directly — skipping the 32 MB
                        // bf16 K_dq round-trip is the whole point of Stage 2.
                        let k_dq_opt = if do_inline {
                            None
                        } else {
                            Some(crate::turboquant::lloyd_max_dequantize_scaled(
                                &kc, &ks, &centroids,
                            )?)
                        };
                        // Symmetric V-side skip: when Stage 3 (`turboquant_sv_inline`)
                        // is active, V_dq never materializes — the kernel reads
                        // V_codes + V_sigma inline. Requires bake_o=ON so V stays
                        // in rotated space all the way through SDPA → o_proj.
                        let v_dq_opt = if do_sv_inline {
                            None
                        } else {
                            let v_dq = crate::turboquant::lloyd_max_dequantize_scaled(
                                &vc, &vs, &centroids,
                            )?;
                            let v_dq = if unrotate_v_dq {
                                let r_t = mlx_rs::ops::transpose_axes(&r_arr, &[1, 0])
                                    .context("tq_sliding: Rᵀ for V un-rotate (non-QJL)")?;
                                crate::turboquant::rotate_last_axis(&v_dq, &r_t)?
                            } else {
                                v_dq
                            };
                            Some(v_dq)
                        };
                        let (kc_inline, ks_inline) = if do_inline {
                            (Some(kc), Some(ks))
                        } else {
                            (None, None)
                        };
                        let (vc_inline, vs_inline) = if do_sv_inline {
                            (Some(vc), Some(vs))
                        } else {
                            (None, None)
                        };
                        (
                            kv_actual_tq,
                            k_dq_opt,
                            v_dq_opt,
                            kc_inline,
                            ks_inline,
                            vc_inline,
                            vs_inline,
                        )
                    };

                // bf16 SDPA on (Q_rotated, K_dequant_rotated, V_dequant_orig).
                // Score Q_rot · K_rot equals Q · K (orthogonal R), and V is in
                // original space → output is in original space, no inverse R.
                //
                // Stage-2 inline path (decode L=1, env gate, QJL off) replaces
                // (K_dq materialize → fused mlx SDPA) with
                //   (Q @ K_codes inline → softmax → GQA matmul with V_dq).
                let sdpa_start = if time_substages {
                    Some(Instant::now())
                } else {
                    None
                };
                let attn_out = if (l as usize) > 1 {
                    // Prefill: build causal+window mask for sliding kind.
                    let k_dq = k_dq_opt
                        .as_ref()
                        .expect("tq_sliding prefill requires materialized K_dq");
                    let v_dq = v_dq_opt
                        .as_ref()
                        .expect("tq_sliding prefill requires materialized V_dq");
                    let mask = make_attention_mask_for_layer_chunked(
                        kind,
                        cfg,
                        l as usize,
                        kv_offset as usize,
                        kv_actual_tq,
                    )?;
                    match mask {
                        Some(m) => sdpa_with_mask(&q_rot, k_dq, v_dq, 1.0, &m)?,
                        None => sdpa(&q_rot, k_dq, v_dq, 1.0, false)?,
                    }
                } else if let (Some(kc), Some(ks)) = (kc_inline.as_ref(), ks_inline.as_ref()) {
                    // Stage-2 inline path: scores = Q · K_dq^T inline, then
                    // softmax + GQA matmul with V_dq. Skips the K_dq DRAM
                    // round-trip. Mask not needed at decode (the sliding
                    // ring's bounds already define the attention window).
                    //
                    // Debug gate `LUMEN_GEMMA4_TQ_QK_INLINE_REF=1` swaps the
                    // custom kernel for the reference (dequant K_dq → mlx
                    // matmul). Isolates whether end-to-end garbage is due to
                    // the kernel or the post-kernel softmax+GQA-matmul wiring.
                    let use_ref = std::env::var("LUMEN_GEMMA4_TQ_QK_INLINE_REF")
                        .map(|s| s == "1")
                        .unwrap_or(false);

                    // Fused attention path: single kernel does (Q@K_dq^T +
                    // softmax + S@V_dq) inline. Eliminates 2 of 3 dispatches
                    // (qk_inline + softmax + sv_inline → fused_attn). Requires
                    // that V codes are plumbed (`do_sv_inline=true`, i.e.
                    // bake_o is ON so V stays in rotated space all the way
                    // through to o_proj). T=1 decode only.
                    let fused_attn_enabled = std::env::var("LUMEN_GEMMA4_TQ_FUSED_ATTN")
                        .map(|s| s == "1")
                        .unwrap_or(false);
                    if fused_attn_enabled && !use_ref && vc_inline.is_some() && vs_inline.is_some()
                    {
                        // Q in head_dim space already (q_rot is Q after RoPE +
                        // R rotation, same input the qk_inline kernel takes).
                        // Gemma 4 doesn't apply 1/sqrt(D) attention scale (the
                        // q_norm RMSNorm normalizes Q to unit RMS so the raw
                        // inner product against equally-normalized K is
                        // numerically stable without extra scaling — verified
                        // empirically: existing `qk_inline` path also uses
                        // scale=1.0 implicit via score = Q@K_dq^T).
                        let vc = vc_inline.as_ref().expect("fused_attn: vc_inline");
                        let vs = vs_inline.as_ref().expect("fused_attn: vs_inline");
                        // Outer block bumps sdpa_start timing after the match
                        // expression — do not double-bump here.
                        crate::turboquant::turboquant_fused_attn(
                            &q_rot, kc, ks, vc, vs, &centroids, 1.0,
                        )?
                    } else {
                        let scores = if use_ref {
                            let k_dq_dbg =
                                crate::turboquant::lloyd_max_dequantize_scaled(kc, ks, &centroids)?;
                            let k_dq_t = mlx_rs::ops::transpose_axes(&k_dq_dbg, &[0, 1, 3, 2])?;
                            let group_d = n_heads / n_kv;
                            let kv_i = kc.shape()[2];
                            let k_t_r =
                                mlx_rs::ops::reshape(&k_dq_t, &[b, n_kv, 1, head_dim, kv_i])?;
                            let k_t_bcast = mlx_rs::ops::broadcast_to(
                                &k_t_r,
                                &[b, n_kv, group_d, head_dim, kv_i],
                            )?;
                            let k_t_full =
                                mlx_rs::ops::reshape(&k_t_bcast, &[b, n_heads, head_dim, kv_i])?;
                            mlx_rs::ops::matmul(&q_rot, &k_t_full)?
                        } else {
                            crate::turboquant::turboquant_qk_inline(&q_rot, kc, ks, &centroids)?
                        };
                        let last_axis = (scores.ndim() as i32) - 1;
                        let attn_w = mlx_rs::ops::softmax_axis(
                            &scores,
                            last_axis,
                            /* precise */ Some(true),
                        )
                        .context("tq_sliding inline: softmax over scores")?;
                        // Stage 3 inline: when V codes are plumbed
                        // (`do_sv_inline` was true), call the custom
                        // `lumen_tq_sv_inline` kernel — one dispatch, no
                        // V_dq materialization, no GQA reshape dance. Kernel
                        // handles GQA internally via `h_kv = h * H_kv / H`.
                        //
                        // Fallback: when Stage 3 is disabled or bake_o=OFF
                        // (V_dq needed in original head_dim space), do the
                        // GQA reshape matmul against materialized V_dq.
                        if let (Some(vc), Some(vs)) = (vc_inline.as_ref(), vs_inline.as_ref()) {
                            crate::turboquant::turboquant_sv_inline(&attn_w, vc, vs, &centroids)?
                        } else {
                            // GQA matmul: attn_w [B, H, 1, N] @ v_dq
                            // [B, H_kv, N, D]. Reshape to broadcast along the
                            // group axis, matmul, then collapse back.
                            let v_dq = v_dq_opt
                                .as_ref()
                                .expect("tq_sliding inline-fallback requires V_dq");
                            let n_kv_local = n_kv;
                            let kv_actual_i = kv_actual_tq as i32;
                            let group = n_heads / n_kv_local;
                            let attn_w_r = mlx_rs::ops::reshape(
                                &attn_w,
                                &[b, n_kv_local, group, 1, kv_actual_i],
                            )
                            .context("tq_sliding inline: reshape attn_w for GQA")?;
                            let v_dq_r = mlx_rs::ops::reshape(
                                v_dq,
                                &[b, n_kv_local, 1, kv_actual_i, head_dim],
                            )
                            .context("tq_sliding inline: reshape v_dq for GQA")?;
                            let attn_out_r = mlx_rs::ops::matmul(&attn_w_r, &v_dq_r)
                                .context("tq_sliding inline: GQA matmul attn_w @ V")?;
                            mlx_rs::ops::reshape(&attn_out_r, &[b, n_heads, 1, head_dim])
                                .context("tq_sliding inline: collapse GQA matmul output")?
                        }
                    } // end of `else` for fused_attn_enabled branch
                } else {
                    // Decode L=1, bf16 path: no mask needed (sliding window
                    // enforced by ring already; SDPA over the ring's K is
                    // correct).
                    let k_dq = k_dq_opt
                        .as_ref()
                        .expect("tq_sliding decode bf16 path requires K_dq");
                    let v_dq = v_dq_opt
                        .as_ref()
                        .expect("tq_sliding decode bf16 path requires V_dq");
                    sdpa(&q_rot, k_dq, v_dq, 1.0, false)?
                };
                if let Some(t0) = sdpa_start {
                    bump_gemma4_attn_sdpa_ms(t0.elapsed().as_secs_f64() * 1e3);
                }

                let oproj_start = if time_substages {
                    Some(Instant::now())
                } else {
                    None
                };
                let attn_t = mlx_rs::ops::transpose_axes(&attn_out, &[0, 2, 1, 3])
                    .context("tq_sliding: transpose output")?;
                let attn_flat = mlx_rs::ops::reshape(&attn_t, &[b, l, n_heads * head_dim])
                    .context("tq_sliding: reshape output flat")?;
                let out_final = Self::qmatmul(&lw.attn.o_proj, &attn_flat)?;
                if let Some(t0) = oproj_start {
                    bump_gemma4_attn_qkvo_ms(t0.elapsed().as_secs_f64() * 1e3);
                }
                return Ok(out_final);
            }
            let (k_full, v_full) = match cache {
                NativeGemma4LayerCache::Sliding(c) => c.update_and_fetch(&k_rope, &v_t)?,
                NativeGemma4LayerCache::Full(c) => c.update_and_fetch(&k_rope, &v_t)?,
                NativeGemma4LayerCache::FullQuantized(_) => {
                    unreachable!("FullQuantized handled in early return above")
                }
                NativeGemma4LayerCache::SlidingQuantized(_) => {
                    unreachable!("SlidingQuantized handled in early return above")
                }
                NativeGemma4LayerCache::SlidingTurboquant(_) => {
                    unreachable!("SlidingTurboquant handled in early return above")
                }
                NativeGemma4LayerCache::FullTurboquant(_) => {
                    unreachable!("FullTurboquant handled in early return above")
                }
            };
            if let Some(t0) = cache_start {
                bump_gemma4_attn_cache_ms(t0.elapsed().as_secs_f64() * 1e3);
            }

            // (7) SDPA. Gemma 4 uses scale=1.0 (per mlx-lm); SDPA mask is
            //     built per-kind via `make_attention_mask_for_layer`.
            //
            // LUMEN_GEMMA4_SDPA_PRE_EVAL=1 forces
            // q_rope/k_full/v_full to evaluate BEFORE the SDPA construction.
            // Used to test whether the per-call SDPA CPU cost we observe at
            // long context (~1.72 ms/call at 4K vs ~1 μs at 8 tok) is the
            // SDPA primitive itself OR an implicit sync-on-input.
            //
            // If SDPA's bucket time drops sharply when this is set, the
            // input-graph drain is what was being attributed to SDPA. If
            // SDPA's bucket time stays high, SDPA's own construction is
            // doing the work (more interesting MLX-side question).
            //
            // Off by default — pre-evals defeat async pipelining and
            // permanently destroy decode throughput. Only enable for
            // diagnostic runs.
            let sdpa_pre_eval = std::env::var("LUMEN_GEMMA4_SDPA_PRE_EVAL")
                .map(|v| v == "1")
                .unwrap_or(false);
            if sdpa_pre_eval {
                q_rope.eval().context("sdpa_pre_eval: q_rope")?;
                k_full.eval().context("sdpa_pre_eval: k_full")?;
                v_full.eval().context("sdpa_pre_eval: v_full")?;
            }

            let sdpa_start = if time_substages {
                Some(Instant::now())
            } else {
                None
            };
            // Mask builder is chunked-aware: derive kv_actual from the K
            // tensor returned by the cache so a rotated sliding-window
            // cache (chunked prefill chunk N≥2) gets a mask sized to its
            // post-rotation K length rather than the logical kv_offset+l
            // window. Single-pass prefill is a degenerate case where
            // kv_actual == kv_offset + l → identical to the legacy path.
            let kv_actual = k_full.shape()[2] as usize;
            // Flash-Attn-2 prefill kernel
            // eligibility. Q-tiled FA-2 with in-register causal + window
            // mask. Supports head_dim=256 (sliding+causal) and head_dim=512
            // (causal only).
            //
            // Status: **functionally correct but perf-incomplete**.
            // 8K A/B 2026-05-15 showed 5× regression vs mlx::fast::sdpa
            // fallback (570 → 106 tok/s). Root cause: threadgroup memory-
            // bound occupancy starvation — each TG uses 32 KB TGM (K_tile +
            // V_tile), limiting concurrent TGs / GPU core on M3 Max to ~1,
            // so ~40 TGs run concurrently against ~32 k total → ~800x
            // serial. mlx::fast::sdpa fallback wins via well-tiled global
            // matmuls.
            //
            // Opt-IN only via `LUMEN_GEMMA4_PREFILL_KERNEL=1`. Default OFF
            // until occupancy fix lands (smaller KBLOCK / K-then-V tile
            // sharing / larger Q-tile w/ more threads-per-TG). All wiring
            // (kernel, primitive, FFI, dispatch) is in place; the perf
            // delta is purely in kernel-internal tile sizing.
            //
            // Guard: kv_actual == kv_offset + l ensures no cache rotation
            // happened — kernel uses (kv_offset + q_row) and (kv_start + j)
            // as absolute positions, valid only pre-rotation.
            let head_dim_now = q_rope.shape().last().copied().unwrap_or(0);
            let kv_actual_now = k_full.shape()[2] as usize;
            let dtype_bf16 = q_rope.dtype() == mlx_rs::Dtype::Bfloat16
                && k_full.dtype() == mlx_rs::Dtype::Bfloat16
                && v_full.dtype() == mlx_rs::Dtype::Bfloat16;
            let prefill_kernel_enabled = std::env::var("LUMEN_GEMMA4_PREFILL_KERNEL")
                .map(|v| v == "1")
                .unwrap_or(false);
            let prefill_kernel_eligible = prefill_kernel_enabled
                && (l as usize) > 1
                && dtype_bf16
                && (head_dim_now == 256 || head_dim_now == 512)
                && kv_actual_now == (kv_offset as usize + l as usize);
            // sliding-window steel kernel path
            // (mlx-side feature). When eligible, dispatches via
            // `mlx_rs::metal::lumen_sdpa_windowed` which wraps
            // mlx::fast::scaled_dot_product_attention with
            // mask_mode="causal" and explicit window_size > 0. The mlx
            // steel kernel honors `has_window` via kb_start truncation
            // (skips entire K-blocks below the window's lower bound),
            // saving ~(L-W)/L of compute at long context — e.g. 87.5%
            // skipped at L=8192, W=1024.
            //
            // Guards:
            //   - sliding layer (full-attn uses causal sentinel path)
            //   - no rotation (kv_actual == kv_offset + l)
            //   - head_dim ∈ {64, 80, 128, 256} (steel kernel instantiation set)
            //   - dtype bf16
            // Env: LUMEN_GEMMA4_SDPA_WINDOWED=0 opts out.
            let sdpa_windowed_enabled = std::env::var("LUMEN_GEMMA4_SDPA_WINDOWED")
                .map(|v| v != "0")
                .unwrap_or(true);
            // Chunked-prefill rotation support (2026-05-15 follow-up): the
            // steel kernel's window check `row_pos - col_pos >= W` is
            // computed in K-relative coordinates (both row_pos and col_pos
            // are indices into the K tensor returned by the cache). The
            // `qL_off = kL - qL` field in AttnParams gives Q's K-relative
            // start position, which equals
            //   (kv_offset) - cache_first_held_pos
            //   = kv_offset - max(0, kv_offset + qL - kv_actual)
            //   = kL - qL                                  (algebra holds)
            // — so the kernel's per-element causal+window checks are
            // correct whether or not the cache rotated. We therefore drop
            // the prior `kv_actual == kv_offset + l` guard and let chunked
            // prefill's intermediate chunks also use the windowed steel
            // kernel path.
            let use_sdpa_windowed = sdpa_windowed_enabled
                && !prefill_kernel_eligible
                && (l as usize) > 1
                && dtype_bf16
                && matches!(kind, NativeGemma4LayerType::SlidingAttention)
                && (head_dim_now == 64
                    || head_dim_now == 80
                    || head_dim_now == 128
                    || head_dim_now == 256);
            // Skip the mask Array build entirely when an in-kernel mask
            // path will fire (prefill_kernel or sdpa_windowed both encode
            // causal+window themselves).
            let mask = if prefill_kernel_eligible || use_sdpa_windowed {
                None
            } else {
                make_attention_mask_for_layer_chunked(
                    kind,
                    cfg,
                    l as usize,
                    kv_offset as usize,
                    kv_actual,
                )?
            };
            // Tight timer: bracket ONLY the sdpa() / sdpa_with_mask() call,
            // excluding make_attention_mask + match + outer bookkeeping.
            // If this stays in the μs range while attn.sdpa reads ms, the
            // delta is in the outer bookkeeping (timer / cell access /
            // make_mask / drops).
            let tight_start = if time_substages {
                Some(Instant::now())
            } else {
                None
            };
            // strided-cache-aware kernel LANDED 2026-05-14.
            // `NativeKvCache::update_and_fetch` returns a slice view over a
            // step-allocated buffer where axis-1 (head) stride = `capacity*D`
            // (capacity ≥ logical Skv). The kernel previously assumed
            // contiguous `Skv*D` head-stride and read wrong memory for
            // decode positions 1+ (anti-pattern #19: microbench parity ≠ e2e
            // production parity). Two fixes evaluated:
            //   (A) `add(K, 0)` + `add(V, 0)` materialize contiguous on
            //       Rust side — 96.30% e2e parity PASS but +1.23 ms / step
            //       from K+V copy. Net M4.8 → SLOWER than mlx::fast::sdpa.
            //   (B) **LANDED**: kernel reads `k.strides()[1]` /
            //       `v.strides()[1]` from the mlx host side and passes them
            //       as kernel args. Per-row stride (axis-2 = D) stays the
            //       same — only the head-base offset changes. Zero copy.
            //
            // Default ON via `LUMEN_GEMMA4_CUSTOM_FLASH_ATTN`. `=0` opts
            // out (mlx::fast::sdpa fallback).
            // MTP-MODE BYPASS added 2026-05-18 (refined): when an mtp_step
            // is IN PROGRESS, Step A (S=1) and Step C (S=K+1>1) attend at
            // the SAME logical positions but go down different SDPA paths
            // (custom-FA-2 for S=1, mlx::fast::sdpa for S>1). The ~1-ULP
            // numerical drift between these kernels flips argmax on
            // sharply-peaked logits (e.g. "continue text" vs "EOS"). Gate
            // on the `mtp_active` flag — set by `mtp_step()` entry and
            // unset at exit — so the OFF decode path (and any other non-
            // mtp_step caller) retains the custom-FA-2 win even when the
            // drafter is loaded.
            //
            // S=1 GUARD: the custom-FA-2 kernel was tuned only for S=1.
            // Multi-token input already wouldn't run it correctly, so the
            // guard is sound independent of MTP mode.
            let q_is_decode_shape = q_rope.shape().get(2).copied() == Some(1);
            let mtp_active_now = self.mtp_active.load(std::sync::atomic::Ordering::Relaxed);
            let use_custom_flash = !prefill_kernel_eligible
                && !use_sdpa_windowed
                && q_is_decode_shape
                && !mtp_active_now
                && std::env::var("LUMEN_GEMMA4_CUSTOM_FLASH_ATTN")
                    .map(|v| v != "0")
                    .unwrap_or(true)
                && mask.is_none()
                && q_rope.dtype() == mlx_rs::Dtype::Bfloat16
                && k_full.dtype() == mlx_rs::Dtype::Bfloat16
                && v_full.dtype() == mlx_rs::Dtype::Bfloat16
                && q_rope.shape().last().copied() == Some(256);
            // Fix B (2026-05-15): full-attention prefill uses Causal sentinel
            // → mlx::fast::sdpa generates the causal triangle on-the-fly via
            //   arange+greater_equal in its fallback (GPU async, bool). Matches
            //   mlx_lm/models/base.py::create_attention_mask which returns
            //   "causal" STRING for full-attention layers (no sliding window),
            //   bypassing Array mask construction + memory traffic entirely.
            // Guard: only when no rotation (cache_first_held_pos == 0 holds
            // for full-attn caches which never rotate) and kv_actual matches
            // the (kv_offset + l) invariant the sentinel assumes.
            // A/B gate: LUMEN_DISABLE_CAUSAL_SENTINEL=1 reverts to the legacy
            // Array mask path for full-attn prefill (perf comparison only).
            let causal_sentinel_disabled = std::env::var("LUMEN_DISABLE_CAUSAL_SENTINEL")
                .map(|v| v == "1")
                .unwrap_or(false);
            let use_causal_sentinel = !prefill_kernel_eligible
                && !use_custom_flash
                && !causal_sentinel_disabled
                && (l as usize) > 1
                && matches!(kind, NativeGemma4LayerType::FullAttention)
                && kv_actual_now == (kv_offset as usize + l as usize);
            let attn_out = if prefill_kernel_eligible {
                let window: u32 = match kind {
                    NativeGemma4LayerType::SlidingAttention => cfg.sliding_window as u32,
                    NativeGemma4LayerType::FullAttention => 0,
                };
                let stream = mlx_rs::Stream::gpu();
                mlx_rs::metal::lumen_flash_attn_prefill_bf16(
                    &q_rope,
                    &k_full,
                    &v_full,
                    1.0,
                    window,
                    kv_offset as u32,
                    &stream,
                )
                .map_err(|e| anyhow!("attn: lumen_flash_attn_prefill_bf16: {e}"))?
            } else if use_sdpa_windowed {
                // mlx steel kernel with has_window=true: kb_start truncates
                // K iter to [floor((q_min - W + 1)/BK), kb_lim] and the
                // left-edge per-element mask handles partial-block trim.
                let stream = mlx_rs::Stream::gpu();
                mlx_rs::metal::lumen_sdpa_windowed(
                    &q_rope,
                    &k_full,
                    &v_full,
                    1.0,
                    cfg.sliding_window as i32,
                    &stream,
                )
                .map_err(|e| anyhow!("attn: lumen_sdpa_windowed: {e}"))?
            } else if use_custom_flash {
                crate::native_metal_bridge::run_flash_attn_bf16(
                    &q_rope, &k_full, &v_full, 1.0, None,
                )
                .map_err(|e| anyhow!("attn: custom flash_attn_bf16: {e}"))?
            } else if use_causal_sentinel {
                sdpa(&q_rope, &k_full, &v_full, 1.0, true)?
            } else {
                match mask {
                    Some(m) => sdpa_with_mask(&q_rope, &k_full, &v_full, 1.0, &m)?,
                    None => sdpa(&q_rope, &k_full, &v_full, 1.0, false)?,
                }
            };
            if let Some(t0) = tight_start {
                bump_gemma4_attn_sdpa_tight_ms(t0.elapsed().as_secs_f64() * 1e3);
            }
            if let Some(t0) = sdpa_start {
                bump_gemma4_attn_sdpa_ms(t0.elapsed().as_secs_f64() * 1e3);
            }

            // (8) Reshape back to [B, L, n_heads * head_dim] and apply o_proj.
            // Bucketed under qkvo since o_proj is one of the 4 projections.
            let oproj_start = if time_substages {
                Some(Instant::now())
            } else {
                None
            };
            let attn_t = mlx_rs::ops::transpose_axes(&attn_out, &[0, 2, 1, 3])
                .context("attn: transpose output failed")?;
            let attn_flat = mlx_rs::ops::reshape(&attn_t, &[b, l, n_heads * head_dim])
                .context("attn: reshape output failed")?;
            let out = Self::qmatmul(&lw.attn.o_proj, &attn_flat)?;
            if let Some(t0) = oproj_start {
                bump_gemma4_attn_qkvo_ms(t0.elapsed().as_secs_f64() * 1e3);
            }
            Ok(out)
        }

        /// `geglu(gate, up) = gelu_approximate(gate) * up` — Gemma 4's
        /// `hidden_activation = "gelu_pytorch_tanh"`.
        ///
        /// When `LUMEN_NATIVE_FUSE_SWIGLU` is unset or non-zero (default ON)
        /// the gelu + multiply pair is fused into a single compiled mlx graph
        /// via `native_moe::gelu_mul_fused`, saving 1 Metal kernel launch and
        /// 1 compile dispatch per call. This fires for every dense MLP forward
        /// and every expert tile activation — 30 layers × (1 + N tiles) per
        /// decode step.
        ///
        /// Disabled fallback path (`LUMEN_NATIVE_FUSE_SWIGLU=0`) runs the
        /// two ops separately, matching the pre-Phase-1.5 baseline used for
        /// A/B parity sanity.
        fn geglu_apply(gate: &Array, up: &Array) -> Result<Array> {
            if crate::native_moe::gelu_mul_fuse_enabled() {
                crate::native_moe::gelu_mul_fused(gate, up)
                    .context("geglu_apply: fused gelu_mul failed")
            } else {
                let activated = mlx_rs::nn::gelu_approximate(gate)
                    .context("geglu_apply: gelu_approx failed")?;
                mlx_rs::ops::multiply(&activated, up).context("geglu_apply: gelu * up failed")
            }
        }

        /// Dense MLP forward: `down_proj(geglu(gate_proj(x), up_proj(x)))`.
        ///
        /// dispatch to `gemma4_dense_mlp_fused` (compile slot
        /// fusing the 3 quantized matmuls + GeGLU into a single traced graph)
        /// when `LUMEN_GEMMA4_FUSE_DENSE_MLP` is enabled (default ON). Falls
        /// back to the unfused path if (a) env disables, or (b) any of the
        /// three projections is missing biases (the compile slot's args layout
        /// requires 10 arrays — biases must be present). Gemma 4's affine
        /// quantization always emits biases, so case (b) is a defensive guard.
        fn dense_mlp_forward(x: &Array, w: &ResolvedGemma4DenseMlpWeights) -> Result<Array> {
            if gemma4_dense_mlp_fuse_enabled() {
                if let Ok(out) = gemma4_dense_mlp_fused(x, w) {
                    return Ok(out);
                }
                // Fall through to legacy path if compile dispatch fails
                // (e.g. biases missing). The fused helper logs via `.context`,
                // and the legacy path is bit-identical fallback.
            }
            let gate = Self::qmatmul(&w.gate_proj, x)?;
            let up = Self::qmatmul(&w.up_proj, x)?;
            let activated = Self::geglu_apply(&gate, &up)?;
            Self::qmatmul(&w.down_proj, &activated)
        }

        /// Router forward: `rms_norm(x, scale * h^-0.5, eps) → proj → top-k →
        /// softmax → × per_expert_scale[indices]`.
        ///
        /// Returns `(indices, weights)` with shape `[B, L, top_k]` each.
        /// Compute router (scores, indices) without applying the routing tail
        /// weights. Used by the combined routing+experts compile path so the
        /// tail can be folded into the same compile slot as the experts.
        fn router_compute_scores_indices(
            &self,
            x: &Array,
            w: &ResolvedGemma4RouterWeights,
        ) -> Result<(Array, Array)> {
            let cfg = &self.config.text_config;
            let top_k = cfg.top_k_experts as i32;
            let eps = cfg.rms_norm_eps;

            let normed = rms_norm(x, &w.scaled_weight, eps).context("router: rms_norm")?;
            let scores = Self::qmatmul(&w.proj, &normed)?;
            let last_axis = (scores.ndim() as i32) - 1;
            let num_experts = scores.shape()[last_axis as usize];
            let partitioned = mlx_rs::ops::argpartition_axis(&scores, -top_k, last_axis)
                .context("router: argpartition_axis failed")?;
            let start = num_experts - top_k;
            let indices = partitioned.index((Ellipsis, start..));
            Ok((scores, indices))
        }

        fn router_forward(
            &self,
            x: &Array,
            w: &ResolvedGemma4RouterWeights,
        ) -> Result<(Array, Array)> {
            let cfg = &self.config.text_config;
            let hidden = cfg.hidden_size as f32;
            let top_k = cfg.top_k_experts as i32;
            let eps = cfg.rms_norm_eps;

            // 1. rms_norm with weight = scale * hidden^-0.5
            //
            // `scaled_weight` is now pre-computed at model load
            // (see resolve_layer → ResolvedGemma4RouterWeights.scaled_weight).
            // Saves 1 Array::from_f32 + 1 multiply per layer per decode step
            // = 60 mlx ops per step across 30 MoE layers.
            let _ = (hidden,); // hidden no longer needed in the hot path
            let normed = rms_norm(x, &w.scaled_weight, eps).context("router: rms_norm")?;

            // 2. expert_scores = proj(normed)  → [B, L, num_experts]
            let scores = Self::qmatmul(&w.proj, &normed)?;

            // 3. argpartition + Ellipsis-slice stay outside the compile slot
            //    because mlx compile cannot infer the dynamic Slice output
            //    shape. The slice carries no GPU work — it's a stride view.
            let last_axis = (scores.ndim() as i32) - 1;
            let num_experts = scores.shape()[last_axis as usize];
            let partitioned = mlx_rs::ops::argpartition_axis(&scores, -top_k, last_axis)
                .context("router: argpartition_axis failed")?;
            let start = num_experts - top_k;
            let indices = partitioned.index((Ellipsis, start..));

            // 4-6. Fused tail: take_along → softmax(precise) →
            //      take_axis(per_expert_scale, indices) → ×
            //      Default ON via LUMEN_GEMMA4_FUSE_ROUTER. Fallback path
            //      below runs the same 3 ops separately for A/B parity.
            if gemma4_router_fuse_enabled() {
                let weights = gemma4_routing_fused_tail(&scores, &indices, &w.per_expert_scale)?;
                Ok((indices, weights))
            } else {
                let top_logits = scores
                    .take_along_axis(&indices, last_axis)
                    .context("router: take_along_axis(scores, indices) failed")?;
                let weights = mlx_rs::ops::softmax_axis(
                    &top_logits,
                    last_axis,
                    /* precise */ Some(true),
                )
                .context("router: softmax over top-k failed")?;
                let per_expert = mlx_rs::ops::indexing::take_axis(&w.per_expert_scale, &indices, 0)
                    .context("router: take_axis(per_expert_scale, indices) failed")?;
                let weights = mlx_rs::ops::multiply(&weights, &per_expert)
                    .context("router: weights × per_expert_scale failed")?;
                Ok((indices, weights))
            }
        }

        /// Gemma 4 SwitchGLU experts forward — mirrors `Experts.__call__` from
        /// `gemma4_text.py` (geglu activation, gather_mm over `[E, *, *]`
        /// expert weights). Returns the routed expert output `[B, L, hidden]`,
        /// already weighted by `top_k_weights` and summed over the K experts.
        fn experts_forward(
            &self,
            x: &Array,
            indices: &Array,
            weights: &Array,
            w: &ResolvedGemma4ExpertsWeights,
        ) -> Result<Array> {
            if x.ndim() != 3 {
                return Err(anyhow!(
                    "experts_forward: expected x rank 3 [B, L, hidden], got ndim={}",
                    x.ndim()
                ));
            }
            if indices.ndim() != 3 {
                return Err(anyhow!(
                    "experts_forward: expected indices rank 3 [B, L, K], got ndim={}",
                    indices.ndim()
                ));
            }
            let ind_shape = indices.shape().to_vec();
            let b = ind_shape[0];
            let l = ind_shape[1];
            let k = ind_shape[2];

            // expand_dims(x, (-2, -3)) → [B, L, 1, 1, hidden]
            let x_5d = mlx_rs::ops::expand_dims_axes(x, &[-2, -3])
                .context("experts: expand_dims(x, (-2,-3)) failed")?;

            let do_sort = (b as usize) * (l as usize) * (k as usize) >= 64;

            let (x_for_qmm, idx_for_qmm, inv_order_opt) = if do_sort {
                let inds_flat = mlx_rs::ops::flatten(indices, 0, -1)
                    .context("experts: flatten(indices) failed")?;
                let order = mlx_rs::ops::argsort(&inds_flat)
                    .context("experts: argsort(indices_flat) failed")?;
                let inv_order = mlx_rs::ops::argsort(&order)
                    .context("experts: argsort(order) for inv_order failed")?;
                let x_flat = mlx_rs::ops::flatten(&x_5d, 0, -3)
                    .context("experts: flatten(x, 0, -3) failed")?;
                let k_arr = Array::from_int(k);
                let row_idx = mlx_rs::ops::floor_divide(&order, &k_arr)
                    .context("experts: order // K floor_divide failed")?;
                let x_sorted = mlx_rs::ops::indexing::take_axis(&x_flat, &row_idx, 0)
                    .context("experts: take_axis(x_flat, order // K) failed")?;
                let idx_sorted = mlx_rs::ops::indexing::take_axis(&inds_flat, &order, 0)
                    .context("experts: take_axis(inds_flat, order) failed")?;
                (x_sorted, idx_sorted, Some(inv_order))
            } else {
                (x_5d, indices.clone(), None)
            };

            // fused compile slot for the no-sort decode
            // branch. Hits per decode step (B=1, L=1, K=8 → 8 < 64). Bakes
            // group_size=64, bits=4, transpose=true, sorted_indices=false,
            // mode="affine" into the traced graph at first call. Falls back
            // to legacy 3-gather_qmm chain on any error or if env disables.
            let down = if !do_sort
                && gemma4_experts_fuse_enabled()
                && w.gate_proj.bits == 4
                && w.gate_proj.group_size == 64
                && w.gate_proj.mode == MODE_AFFINE
            {
                match gemma4_experts_fused(&x_for_qmm, &idx_for_qmm, w) {
                    Ok(down) => down,
                    Err(_) => {
                        // Legacy fallback if fusion fails (e.g. biases missing).
                        let x_gate = gather_qmm_with_mode(
                            &x_for_qmm,
                            &w.gate_proj.weight,
                            &w.gate_proj.scales,
                            w.gate_proj.biases.as_ref(),
                            None,
                            Some(&idx_for_qmm),
                            true,
                            w.gate_proj.group_size,
                            w.gate_proj.bits,
                            w.gate_proj.mode,
                            do_sort,
                        )
                        .context("experts: gather_qmm(gate_proj) failed")?;
                        let x_up = gather_qmm_with_mode(
                            &x_for_qmm,
                            &w.up_proj.weight,
                            &w.up_proj.scales,
                            w.up_proj.biases.as_ref(),
                            None,
                            Some(&idx_for_qmm),
                            true,
                            w.up_proj.group_size,
                            w.up_proj.bits,
                            w.up_proj.mode,
                            do_sort,
                        )
                        .context("experts: gather_qmm(up_proj) failed")?;
                        let activated = Self::geglu_apply(&x_gate, &x_up)?;
                        gather_qmm_with_mode(
                            &activated,
                            &w.down_proj.weight,
                            &w.down_proj.scales,
                            w.down_proj.biases.as_ref(),
                            None,
                            Some(&idx_for_qmm),
                            true,
                            w.down_proj.group_size,
                            w.down_proj.bits,
                            w.down_proj.mode,
                            do_sort,
                        )
                        .context("experts: gather_qmm(down_proj) failed")?
                    }
                }
            } else {
                let x_gate = gather_qmm_with_mode(
                    &x_for_qmm,
                    &w.gate_proj.weight,
                    &w.gate_proj.scales,
                    w.gate_proj.biases.as_ref(),
                    None,
                    Some(&idx_for_qmm),
                    true,
                    w.gate_proj.group_size,
                    w.gate_proj.bits,
                    w.gate_proj.mode,
                    do_sort,
                )
                .context("experts: gather_qmm(gate_proj) failed")?;
                let x_up = gather_qmm_with_mode(
                    &x_for_qmm,
                    &w.up_proj.weight,
                    &w.up_proj.scales,
                    w.up_proj.biases.as_ref(),
                    None,
                    Some(&idx_for_qmm),
                    true,
                    w.up_proj.group_size,
                    w.up_proj.bits,
                    w.up_proj.mode,
                    do_sort,
                )
                .context("experts: gather_qmm(up_proj) failed")?;
                let activated = Self::geglu_apply(&x_gate, &x_up)?;
                gather_qmm_with_mode(
                    &activated,
                    &w.down_proj.weight,
                    &w.down_proj.scales,
                    w.down_proj.biases.as_ref(),
                    None,
                    Some(&idx_for_qmm),
                    true,
                    w.down_proj.group_size,
                    w.down_proj.bits,
                    w.down_proj.mode,
                    do_sort,
                )
                .context("experts: gather_qmm(down_proj) failed")?
            };

            // Unsort + reshape to [B, L, K, 1, hidden], squeeze axis -2.
            let recombined = if let Some(inv_order) = inv_order_opt {
                let reordered = mlx_rs::ops::indexing::take_axis(&down, &inv_order, 0)
                    .context("experts: take_axis(down, inv_order) failed")?;
                mlx_rs::ops::unflatten(&reordered, 0, &[b, l, k])
                    .context("experts: unflatten(0, [B,L,K]) failed")?
            } else {
                down
            };
            let per_expert_out = mlx_rs::ops::squeeze_axes(&recombined, &[-2])
                .context("experts: squeeze(-2) failed")?;
            // per_expert_out shape: [B, L, K, hidden]

            // bf16-throughout — keep per-expert output
            // in bf16 (legacy f32-cast opt-out removed 2026-05-14).

            // y = (per_expert_out * weights[..., None]).sum(axis=-2)
            let w_expanded = mlx_rs::ops::expand_dims_axes(weights, &[-1])
                .context("experts: expand_dims(weights, -1) failed")?;
            let weighted = mlx_rs::ops::multiply(&per_expert_out, &w_expanded)
                .context("experts: weighted multiply failed")?;
            let summed = mlx_rs::ops::sum_axis(&weighted, -2, /* keep_dims */ false)
                .context("experts: sum(axis=-2) failed")?;
            Ok(summed)
        }

        /// Full Gemma 4 decoder layer forward: input_norm → attention →
        /// post_attention_norm → +residual → dual feed-forward (Dense MLP +
        /// MoE in parallel, summed) → post_feedforward_norm → +residual →
        /// × layer_scalar. Mirrors `Gemma4TextModel.DecoderLayer.__call__`.
        pub fn decoder_layer_forward(
            &self,
            x: &Array,
            layer_idx: usize,
            cache: &mut NativeGemma4LayerCache,
        ) -> Result<Array> {
            let lw = &self.layers[layer_idx];
            let eps = self.config.text_config.rms_norm_eps;
            // `time_stages` controls whether per-stage timers fire.
            // `skip_eval_barriers` distinguishes the two breakdown modes:
            //   - legacy `LUMEN_GEMMA4_BREAKDOWN=1`: time_stages=true,
            //     skip_eval_barriers=false → eval() drains GPU into bucket,
            //     numbers inflated ~10–17× by serialization but ratios
            //     within a single context length remain informative.
            //   - `LUMEN_GEMMA4_HONEST_BREAKDOWN=1`: time_stages=true,
            //     skip_eval_barriers=true → pure Rust-side FFI dispatch
            //     time per stage. Reveals which stage's per-op cost
            //     scales asymmetrically with context length.
            let time_stages = gemma4_any_breakdown_active();
            let skip_eval_barriers = gemma4_honest_breakdown_active();

            // 1. residual = x; h = input_layernorm(x); attn(h)
            // borrow `x` directly instead of `x.clone()` to
            // skip the per-layer `mlx_array_set` refcount-bump FFI call
            // (~325 ns/layer × 48 layers/step ≈ 15.6 μs/step saved).
            let residual: &Array = x;
            let h = rms_norm(x, &lw.input_layernorm, eps).context("layer: input_layernorm")?;

            let attn_start = if time_stages {
                Some(Instant::now())
            } else {
                None
            };
            let h = self.layer_attention_forward(&h, layer_idx, cache)?;
            if let Some(t0) = attn_start {
                if !skip_eval_barriers {
                    // Drain GPU into this bucket. Defeats async pipelining;
                    // numbers inflate but stage attribution is accurate.
                    h.eval().context("breakdown: eval after attn")?;
                }
                let ms = t0.elapsed().as_secs_f64() * 1e3;
                bump_gemma4_attn_ms(ms);
                match lw.kind {
                    NativeGemma4LayerType::FullAttention => bump_gemma4_attn_full_ms(ms),
                    NativeGemma4LayerType::SlidingAttention => bump_gemma4_attn_sliding_ms(ms),
                }
            }

            let h = rms_norm(&h, &lw.post_attention_layernorm, eps)
                .context("layer: post_attention_layernorm")?;
            let h = mlx_rs::ops::add(residual, &h).context("layer: +residual (attn)")?;

            // 2. Dual feed-forward (Dense MLP + MoE).
            // borrow current `h` for the post-FF residual add
            // instead of `h.clone()`. Old `h` stays alive in its lexical
            // scope (function body) even after subsequent `let h = ...`
            // shadowing — `residual` keeps a valid &Array to it. Saves 1
            // `mlx_array_set` FFI/layer.
            let residual: &Array = &h;

            // Dense MLP path: pre_ff_norm → mlp → post_ff_1
            // Tier 2C (2026-05-16) — opt-in fuse path absorbs both norms
            // into the dense_mlp compile slot (3 launches → 1).
            let dense_start = if time_stages {
                Some(Instant::now())
            } else {
                None
            };
            let h1 = if gemma4_pre_post_norm_dense_mlp_fuse_enabled() {
                match gemma4_pre_post_norm_dense_mlp_fused(
                    &h,
                    &lw.pre_feedforward_layernorm,
                    &lw.post_feedforward_layernorm_1,
                    &lw.dense_mlp,
                ) {
                    Ok(out) => out,
                    Err(_) => {
                        let h1 = rms_norm(&h, &lw.pre_feedforward_layernorm, eps)
                            .context("layer: pre_feedforward_layernorm")?;
                        let h1 = Self::dense_mlp_forward(&h1, &lw.dense_mlp)?;
                        rms_norm(&h1, &lw.post_feedforward_layernorm_1, eps)
                            .context("layer: post_feedforward_layernorm_1")?
                    }
                }
            } else {
                let h1 = rms_norm(&h, &lw.pre_feedforward_layernorm, eps)
                    .context("layer: pre_feedforward_layernorm")?;
                let h1 = Self::dense_mlp_forward(&h1, &lw.dense_mlp)?;
                rms_norm(&h1, &lw.post_feedforward_layernorm_1, eps)
                    .context("layer: post_feedforward_layernorm_1")?
            };
            if let Some(t0) = dense_start {
                if !skip_eval_barriers {
                    h1.eval().context("breakdown: eval after dense_mlp")?;
                }
                bump_gemma4_dense_ms(t0.elapsed().as_secs_f64() * 1e3);
            }

            // MoE path: 3 nested compile-slot expansion levels.
            //   1. LUMEN_GEMMA4_FUSE_NORM_ROUTING_EXPERTS=1 (Tier 2C-MoE):
            //      pre-norm + routing tail + experts + recombine + post-norm
            //      all in a single compile slot. Default OFF (exploration).
            //   2. LUMEN_GEMMA4_FUSE_ROUTING_EXPERTS=1 (LANDED, default ON):
            //      routing tail + experts + recombine in a single compile slot.
            //      Pre/post norms stay outside.
            //   3. Legacy: router_forward (routing_fused_tail slot) +
            //      experts_forward (experts_fused slot) chain.
            let needs_post_norm_outside;
            let h2 = if gemma4_pre_post_norm_routing_experts_fuse_enabled()
                && lw.experts.gate_proj.bits == 4
                && lw.experts.gate_proj.group_size == 64
                && lw.experts.gate_proj.mode == MODE_AFFINE
            {
                let router_start = if time_stages {
                    Some(Instant::now())
                } else {
                    None
                };
                let (scores, indices) = self.router_compute_scores_indices(&h, &lw.router)?;
                if let Some(t0) = router_start {
                    if !skip_eval_barriers {
                        scores
                            .eval()
                            .context("breakdown: eval after router (scores)")?;
                        indices
                            .eval()
                            .context("breakdown: eval after router (indices)")?;
                    }
                    bump_gemma4_router_ms(t0.elapsed().as_secs_f64() * 1e3);
                }
                let experts_start = if time_stages {
                    Some(Instant::now())
                } else {
                    None
                };
                let h2 = match gemma4_pre_post_norm_routing_experts_fused(
                    &scores,
                    &indices,
                    &lw.router.per_expert_scale,
                    &h,
                    &lw.pre_feedforward_layernorm_2,
                    &lw.post_feedforward_layernorm_2,
                    &lw.experts,
                ) {
                    Ok(out) => out,
                    Err(_) => {
                        // Fallback: do pre-norm here and continue without
                        // the slot. Caller will apply post-norm outside.
                        let weights = gemma4_routing_fused_tail(
                            &scores,
                            &indices,
                            &lw.router.per_expert_scale,
                        )?;
                        let h2_pre = rms_norm(&h, &lw.pre_feedforward_layernorm_2, eps)?;
                        let h2_inner =
                            self.experts_forward(&h2_pre, &indices, &weights, &lw.experts)?;
                        rms_norm(&h2_inner, &lw.post_feedforward_layernorm_2, eps)?
                    }
                };
                if let Some(t0) = experts_start {
                    if !skip_eval_barriers {
                        h2.eval().context("breakdown: eval after experts")?;
                    }
                    bump_gemma4_experts_ms(t0.elapsed().as_secs_f64() * 1e3);
                }
                needs_post_norm_outside = false;
                h2
            } else if gemma4_routing_experts_fuse_enabled()
                && lw.experts.gate_proj.bits == 4
                && lw.experts.gate_proj.group_size == 64
                && lw.experts.gate_proj.mode == MODE_AFFINE
            {
                let router_start = if time_stages {
                    Some(Instant::now())
                } else {
                    None
                };
                let (scores, indices) = self.router_compute_scores_indices(&h, &lw.router)?;
                if let Some(t0) = router_start {
                    if !skip_eval_barriers {
                        scores
                            .eval()
                            .context("breakdown: eval after router (scores)")?;
                        indices
                            .eval()
                            .context("breakdown: eval after router (indices)")?;
                    }
                    bump_gemma4_router_ms(t0.elapsed().as_secs_f64() * 1e3);
                }
                let h2 = rms_norm(&h, &lw.pre_feedforward_layernorm_2, eps)
                    .context("layer: pre_feedforward_layernorm_2")?;
                let experts_start = if time_stages {
                    Some(Instant::now())
                } else {
                    None
                };
                let h2 = match gemma4_routing_experts_fused(
                    &scores,
                    &indices,
                    &lw.router.per_expert_scale,
                    &h2,
                    &lw.experts,
                ) {
                    Ok(out) => out,
                    Err(_) => {
                        // Fallback to the legacy two-slot path.
                        let weights = gemma4_routing_fused_tail(
                            &scores,
                            &indices,
                            &lw.router.per_expert_scale,
                        )?;
                        self.experts_forward(&h2, &indices, &weights, &lw.experts)?
                    }
                };
                if let Some(t0) = experts_start {
                    if !skip_eval_barriers {
                        h2.eval().context("breakdown: eval after experts")?;
                    }
                    bump_gemma4_experts_ms(t0.elapsed().as_secs_f64() * 1e3);
                }
                needs_post_norm_outside = true;
                h2
            } else {
                let router_start = if time_stages {
                    Some(Instant::now())
                } else {
                    None
                };
                let (indices, weights) = self.router_forward(&h, &lw.router)?;
                if let Some(t0) = router_start {
                    if !skip_eval_barriers {
                        indices
                            .eval()
                            .context("breakdown: eval after router (indices)")?;
                        weights
                            .eval()
                            .context("breakdown: eval after router (weights)")?;
                    }
                    bump_gemma4_router_ms(t0.elapsed().as_secs_f64() * 1e3);
                }

                let h2 = rms_norm(&h, &lw.pre_feedforward_layernorm_2, eps)
                    .context("layer: pre_feedforward_layernorm_2")?;
                let experts_start = if time_stages {
                    Some(Instant::now())
                } else {
                    None
                };
                let h2 = self.experts_forward(&h2, &indices, &weights, &lw.experts)?;
                if let Some(t0) = experts_start {
                    if !skip_eval_barriers {
                        h2.eval().context("breakdown: eval after experts")?;
                    }
                    bump_gemma4_experts_ms(t0.elapsed().as_secs_f64() * 1e3);
                }
                needs_post_norm_outside = true;
                h2
            };
            let h2 = if needs_post_norm_outside {
                rms_norm(&h2, &lw.post_feedforward_layernorm_2, eps)
                    .context("layer: post_feedforward_layernorm_2")?
            } else {
                h2
            };

            // Epilogue: add(h1, h2) → rms_norm(post_ff) → +residual → ×layer_scalar.
            // Opt-in fuse path (LUMEN_GEMMA4_FUSE_LAYER_EPILOGUE=1) collapses
            // the 4-op tail into one compile slot (~192 launches/step saved
            // across 48 layers). Default OFF — Tier 2C risk: dense_mlp norm
            // fuse and pre/post-norm routing+experts both NEUTRAL on similar
            // elementwise+rms_norm chains (async pipelining absorbs them).
            let h = if gemma4_layer_epilogue_fuse_enabled() {
                match gemma4_layer_epilogue_fused(
                    &h1,
                    &h2,
                    residual,
                    &lw.post_feedforward_layernorm,
                    &lw.layer_scalar,
                ) {
                    Ok(out) => out,
                    Err(_) => {
                        // Unfused fallback (bit-identical).
                        let h = mlx_rs::ops::add(&h1, &h2).context("layer: h1 + h2")?;
                        let h = rms_norm(&h, &lw.post_feedforward_layernorm, eps)
                            .context("layer: post_feedforward_layernorm")?;
                        let h = mlx_rs::ops::add(residual, &h).context("layer: +residual (ff)")?;
                        mlx_rs::ops::multiply(&h, &lw.layer_scalar)
                            .context("layer: × layer_scalar")?
                    }
                }
            } else {
                let h = mlx_rs::ops::add(&h1, &h2).context("layer: h1 + h2")?;
                let h = rms_norm(&h, &lw.post_feedforward_layernorm, eps)
                    .context("layer: post_feedforward_layernorm")?;
                let h = mlx_rs::ops::add(residual, &h).context("layer: +residual (ff)")?;
                mlx_rs::ops::multiply(&h, &lw.layer_scalar).context("layer: × layer_scalar")?
            };
            Ok(h)
        }

        /// Phase 3: post-attention body of a decoder layer (post_attn_norm +
        /// residual, dense MLP, MoE, epilogue), as a standalone helper so the
        /// batched decode path can run it ONCE over `[N,1,H]` after looping the
        /// attention per seq. This uses the UNFUSED (legacy) compute path — which
        /// is exactly what the single-seq `decoder_layer_forward` runs by default
        /// for NVFP4 experts (the fused routing+experts slot requires
        /// `mode == MODE_AFFINE`, group 64; NVFP4 falls to the legacy branch) and
        /// when the opt-in dense-MLP / epilogue fuses are off (both default OFF).
        /// So for the production nvfp4 Gemma 4 it is bit-identical to the
        /// single-seq layer; parity is asserted live. `residual_in` is the layer
        /// input x; `attn_out` is the attention output.
        fn decoder_layer_post_attn(
            &self,
            residual_in: &Array,
            attn_out: &Array,
            layer_idx: usize,
        ) -> Result<Array> {
            let lw = &self.layers[layer_idx];
            let eps = self.config.text_config.rms_norm_eps;

            let h = rms_norm(attn_out, &lw.post_attention_layernorm, eps)
                .context("post_attn: post_attention_layernorm")?;
            let h = mlx_rs::ops::add(residual_in, &h).context("post_attn: +residual (attn)")?;
            let residual: &Array = &h;

            // Dense MLP (unfused): pre_ff_norm → mlp → post_ff_1.
            let h1 = rms_norm(&h, &lw.pre_feedforward_layernorm, eps)
                .context("post_attn: pre_feedforward_layernorm")?;
            let h1 = Self::dense_mlp_forward(&h1, &lw.dense_mlp)?;
            let h1 = rms_norm(&h1, &lw.post_feedforward_layernorm_1, eps)
                .context("post_attn: post_feedforward_layernorm_1")?;

            // MoE (unfused legacy): router → pre_norm → experts → post_norm.
            let (indices, weights) = self.router_forward(&h, &lw.router)?;
            let h2 = rms_norm(&h, &lw.pre_feedforward_layernorm_2, eps)
                .context("post_attn: pre_feedforward_layernorm_2")?;
            let h2 = self.experts_forward(&h2, &indices, &weights, &lw.experts)?;
            let h2 = rms_norm(&h2, &lw.post_feedforward_layernorm_2, eps)
                .context("post_attn: post_feedforward_layernorm_2")?;

            // Epilogue (unfused): add → post_ff_norm → +residual → ×layer_scalar.
            let h = mlx_rs::ops::add(&h1, &h2).context("post_attn: h1 + h2")?;
            let h = rms_norm(&h, &lw.post_feedforward_layernorm, eps)
                .context("post_attn: post_feedforward_layernorm")?;
            let h = mlx_rs::ops::add(residual, &h).context("post_attn: +residual (ff)")?;
            mlx_rs::ops::multiply(&h, &lw.layer_scalar).context("post_attn: × layer_scalar")
        }

        /// Phase 3: one batched decode forward over N single-token sequences,
        /// each with its own per-seq Gemma 4 cache. Mirrors Qwen's
        /// `forward_decode_batch`: the weight-bound embed/dense-MLP/MoE/lm_head
        /// run batched over `[N,1,H]`, while the attention loops per seq (each at
        /// its OWN rotating sliding-window cache offset — so the windows rotate
        /// independently, no ragged-window mask needed). Returns one argmax token
        /// per seq. N==1 is handled by the caller via the single-seq decode path.
        pub fn forward_decode_batch(
            &self,
            tokens: &[u32],
            caches: &mut [&mut NativeGemma4PromptCache],
        ) -> Result<Vec<u32>> {
            let cfg = &self.config.text_config;
            assert_eq!(
                cfg.hidden_size_per_layer_input, 0,
                "forward_decode_batch: hidden_size_per_layer_input > 0 not supported"
            );
            let n = tokens.len();
            if n == 0 {
                return Ok(Vec::new());
            }
            if caches.len() != n {
                return Err(anyhow!(
                    "forward_decode_batch: caches.len()={} != tokens.len()={n}",
                    caches.len()
                ));
            }
            let n_layers = cfg.num_hidden_layers;
            for (i, c) in caches.iter().enumerate() {
                if c.len() != n_layers {
                    return Err(anyhow!(
                        "forward_decode_batch: cache[{i}].len()={} != num_layers={n_layers}",
                        c.len()
                    ));
                }
            }
            let nn = n as i32;
            let hidden = cfg.hidden_size as i32;

            // Embed: one token per seq → [N, hidden] → [N, 1, hidden] × scale.
            let ids_i32: Vec<i32> = tokens.iter().map(|&u| u as i32).collect();
            let ids_arr = Array::from_slice(&ids_i32, &[nn]);
            let embed_rows = self.embed_lookup_affine(&ids_arr)?;
            let h = mlx_rs::ops::reshape(&embed_rows, &[nn, 1, hidden])
                .context("forward_decode_batch: reshape embed [N,1,H]")?;
            let mut h = mlx_rs::ops::multiply(&h, &self.const_embed_scale)
                .context("forward_decode_batch: × sqrt(hidden)")?;

            for idx in 0..n_layers {
                let lw = &self.layers[idx];
                let normed = rms_norm(&h, &lw.input_layernorm, cfg.rms_norm_eps)
                    .context("forward_decode_batch: input_layernorm")?;
                // Per-seq attention (sliding + global), each on its own cache.
                let mut attn_parts: Vec<Array> = Vec::with_capacity(n);
                for (i, c) in caches.iter_mut().enumerate() {
                    let idx_i = Array::from_slice(&[i as i32], &[1]);
                    let normed_i = mlx_rs::ops::indexing::take_axis(&normed, &idx_i, 0)
                        .context("forward_decode_batch: slice seq for attention")?;
                    let lc = c.layer_mut(idx).ok_or_else(|| {
                        anyhow!("forward_decode_batch: cache layer {idx} missing")
                    })?;
                    let attn_i = self.layer_attention_forward(&normed_i, idx, lc)?;
                    attn_parts.push(attn_i);
                }
                let refs: Vec<&Array> = attn_parts.iter().collect();
                let attn_batch = mlx_rs::ops::concatenate_axis(&refs, 0)
                    .context("forward_decode_batch: stack attention outputs")?;
                h = self.decoder_layer_post_attn(&h, &attn_batch, idx)?;
            }

            let h = rms_norm(&h, &self.final_norm, cfg.rms_norm_eps)
                .context("forward_decode_batch: final_norm")?;
            let logits = self.tied_lm_head_plus_softcap(&h)?;
            self.argmax_rows(&logits)
        }

        /// Per-row argmax over `[N, 1, vocab]` logits → `Vec<u32>` of length N
        /// (single eval via `try_as_slice`). Phase 3 batched-decode counterpart
        /// of `argmax_last_token` (which is `[1,1]`-only).
        fn argmax_rows(&self, logits: &Array) -> Result<Vec<u32>> {
            if logits.ndim() != 3 {
                return Err(anyhow!(
                    "argmax_rows: expected [N, 1, V] logits, got ndim={}",
                    logits.ndim()
                ));
            }
            let idx0 = Array::from_int(0);
            let rows = mlx_rs::ops::indexing::take_axis(logits, &idx0, 1)
                .context("argmax_rows: take_axis(0, axis=1)")?;
            let am = mlx_rs::ops::indexing::argmax_axis(&rows, -1, /* keep_dims */ false)
                .context("argmax_rows: argmax_axis(-1)")?;
            let slice = am
                .try_as_slice::<u32>()
                .map_err(|err| anyhow!("argmax_rows: read failed: {err}"))?;
            Ok(slice.to_vec())
        }

        /// Embedding lookup with affine-quant rows (4-bit) + biases support.
        /// `quantized_embedding_lookup_with_mode` in `native_embedding` is
        /// hard-coded to `biases=None`, which is fine for MXFP4 but misses
        /// Gemma 4's affine `embed_tokens.biases`. We replicate the take_axis
        /// + dequantize_with_mode pattern with the optional biases plumbed
        /// through.
        fn embed_lookup_affine(&self, token_ids: &Array) -> Result<Array> {
            let embed = match &self.embed_tokens {
                EmbedTokensWeights::Bf16(w) => {
                    return w
                        .take_axis(token_ids, 0)
                        .context("embed_lookup(bf16): take_axis failed");
                }
                EmbedTokensWeights::Quantized(q) => q,
            };
            let selected_packed = embed
                .weight
                .take_axis(token_ids, 0)
                .context("embed_lookup: take_axis(packed) failed")?;
            let selected_scales = embed
                .scales
                .take_axis(token_ids, 0)
                .context("embed_lookup: take_axis(scales) failed")?;
            let selected_biases = embed
                .biases
                .as_ref()
                .map(|b| b.take_axis(token_ids, 0))
                .transpose()
                .context("embed_lookup: take_axis(biases) failed")?;
            dequantize_with_mode(
                &selected_packed,
                &selected_scales,
                selected_biases.as_ref(),
                embed.group_size,
                embed.bits,
                embed.mode,
            )
            .context("embed_lookup: dequantize_with_mode failed")
        }

        /// Replace the embedding rows covered by each image's placeholder run
        /// with that image's soft tokens, returning the rebuilt
        /// `[1, l, hidden]` stream.
        ///
        /// `runs` are **prompt-global** `(start, len)` spans and `soft[i]` is
        /// the `[len_i, hidden]` encoding of `runs[i]`; `h` covers the prompt
        /// window `[span_start, span_start + l)`. Single-pass prefill passes
        /// `span_start = 0` with the whole prompt. Chunked prefill hands over
        /// one window at a time, so a run may fall entirely outside the window
        /// or straddle either edge — only the overlapping rows are spliced
        /// here and the remainder arrives with the neighbouring chunk.
        fn splice_soft_tokens_for_span(
            &self,
            h: &Array,
            span_start: i32,
            l: i32,
            runs: &[(usize, usize)],
            soft: &[Array],
        ) -> Result<Array> {
            let hidden = self.config.text_config.hidden_size as i32;
            let dt = h.dtype();
            let slices = clip_runs_to_window(span_start, l, runs)?;
            // No image touches this window — hand back the embeddings untouched
            // rather than round-tripping them through a gather + concat.
            if slices.is_empty() {
                return Ok(h.clone());
            }

            let mut segments: Vec<Array> = Vec::with_capacity(slices.len() * 2 + 1);
            let mut cursor: i32 = 0;
            for s in &slices {
                if s.local_start > cursor {
                    segments.push(
                        take_span(h, cursor, s.local_start)
                            .context("splice: text span before image")?,
                    );
                }
                let soft_i = soft
                    .get(s.image)
                    .ok_or_else(|| anyhow!("splice: no soft tokens for image {}", s.image))?;
                let run_len = runs[s.image].1 as i32;
                let soft_3d = mlx_rs::ops::reshape(soft_i, &[1, run_len, hidden])
                    .context("splice: reshape soft tokens")?;
                let soft_3d = take_span(&soft_3d, s.row_start, s.row_end)
                    .context("splice: clip soft tokens to window")?;
                segments.push(soft_3d.as_dtype(dt).context("splice: cast soft tokens")?);
                cursor = s.local_end;
            }
            if cursor < l {
                segments.push(take_span(h, cursor, l).context("splice: trailing text span")?);
            }

            let refs: Vec<&Array> = segments.iter().collect();
            let out =
                mlx_rs::ops::concatenate_axis(&refs, 1).context("splice: concatenate segments")?;
            debug_assert_eq!(out.shape(), [1, l, hidden]);
            Ok(out)
        }

        /// Apply Gemma 4's `final_logit_softcapping`:
        ///   `out = tanh(out / softcap) * softcap`
        /// with `softcap = 30.0` per 26B-A4B's config.
        fn apply_logit_softcap(&self, logits: &Array) -> Result<Array> {
            // Tier 3 (2026-05-15): compile-slot fuse mirrors mlx-lm's
            // `@partial(mx.compile, shapeless=True) def logit_softcap(...)`.
            // Three ops (multiply, tanh, multiply) collapse to a single
            // traced graph, eliminating two kernel launches + intermediate
            // tensor materialization. Most impact comes at decode (per-step
            // [1, 1, V] softcap) since lm_head slice already trimmed
            // prefill softcap input shape. Env-gated A/B
            // (`LUMEN_GEMMA4_FUSE_SOFTCAP=0`) falls back to the unfused
            // path; default ON.
            if gemma4_softcap_fuse_enabled() {
                if let Ok(out) =
                    gemma4_softcap_fused(logits, &self.const_softcap_inv, &self.const_softcap)
                {
                    return Ok(out);
                }
                // Trace dispatch failed for some reason — fall through to
                // unfused legacy path (bit-identical numerically).
            }
            let scaled = mlx_rs::ops::multiply(logits, &self.const_softcap_inv)
                .context("softcap: logits × (1/cap) failed")?;
            let tanh = mlx_rs::ops::tanh(&scaled).context("softcap: tanh failed")?;
            mlx_rs::ops::multiply(&tanh, &self.const_softcap).context("softcap: tanh × cap failed")
        }

        /// Full model forward: `[B, L]` token-id input → `[B, L, vocab]` logits.
        ///
        /// Mirrors mlx_lm's `Gemma4TextModel.__call__` + `Model.__call__`:
        ///   1. embedding lookup (affine 4-bit)
        ///   2. h = embed × sqrt(hidden_size)
        ///   3. 30 decoder layers
        ///   4. final RMSNorm
        ///   5. tied lm_head = embed_tokens.as_linear(h) — i.e.
        ///      `quantized_matmul(h, embed.weight, transpose=true, …)`
        ///   6. final_logit_softcapping
        ///
        /// 26B-A4B has `hidden_size_per_layer_input = 0` so the per-layer
        /// input embedding branch is skipped.
        pub fn forward(
            &self,
            input_ids: &[u32],
            cache: &mut NativeGemma4PromptCache,
        ) -> Result<Array> {
            if input_ids.is_empty() {
                return Err(anyhow!("forward: empty input_ids"));
            }
            let l = input_ids.len() as i32;
            let ids = Array::from_slice(input_ids, &[1, l]);
            self.forward_array(&ids, cache)
        }

        /// Is the image tower loaded and usable?
        pub fn vision_available(&self) -> bool {
            self.vision.is_some()
        }

        /// Token id whose embedding rows vision soft tokens replace.
        pub fn image_token_id(&self) -> Option<u32> {
            self.config.image_token_id
        }

        /// Prompt tokens this image will occupy: its soft-token run plus the
        /// `<|image>` / `<image|>` sentinels and the newline the renderer emits
        /// after the block.
        ///
        /// Header-only — no pixel decode — so the server can size a prompt for
        /// the context guard and usage accounting before deciding whether to
        /// run anything at all.
        pub fn image_prompt_tokens(&self, encoded: &[u8]) -> Result<usize> {
            let vision = self.vision.as_ref().ok_or_else(|| {
                anyhow!("image input requires LUMEN_VISION=1 and a vision-capable checkpoint")
            })?;
            let vcfg = vision.config();
            let budget = crate::gemma4_vision::imp::soft_token_budget_override()
                .or(self.config.vision_soft_tokens_per_image)
                .unwrap_or(280);
            let soft = crate::gemma4_vision::soft_token_count(
                encoded,
                vcfg.patch_size,
                budget,
                vcfg.pooling_kernel_size,
            )
            .map_err(|e| anyhow!("image sizing failed: {e}"))?;
            // + `<|image>`, `<image|>`, and the trailing newline.
            Ok(soft + 3)
        }

        /// Decode + resize one image onto the patch grid, without running the
        /// tower.
        ///
        /// Callers hold the result: `num_soft_tokens` sizes the prompt's
        /// placeholder run, and the same [`PreparedImage`] then goes straight
        /// into [`Self::encode_prepared_image`]. Re-deriving it from the bytes
        /// at encode time would decode and bicubic-resize every image twice.
        ///
        /// [`PreparedImage`]: crate::gemma4_vision::PreparedImage
        pub fn prepare_image(&self, encoded: &[u8]) -> Result<crate::gemma4_vision::PreparedImage> {
            let vision = self.vision.as_ref().ok_or_else(|| {
                anyhow!("image input requires LUMEN_VISION=1 and a vision-capable checkpoint")
            })?;
            let vcfg = vision.config();
            let budget = crate::gemma4_vision::imp::soft_token_budget_override()
                .or(self.config.vision_soft_tokens_per_image)
                .unwrap_or(280);
            crate::gemma4_vision::prepare_image(
                encoded,
                vcfg.patch_size,
                budget,
                vcfg.pooling_kernel_size,
            )
            .map_err(|e| anyhow!("image preprocessing failed: {e}"))
        }

        /// Run the tower on an already-prepared image, returning
        /// `[num_soft_tokens, hidden]` language-model embeddings.
        pub fn encode_prepared_image(
            &self,
            prepared: &crate::gemma4_vision::PreparedImage,
        ) -> Result<Array> {
            let vision = self.vision.as_ref().ok_or_else(|| {
                anyhow!("image input requires LUMEN_VISION=1 and a vision-capable checkpoint")
            })?;

            let n = (prepared.grid.0 * prepared.grid.1) as i32;
            let width = (3 * vision.config().patch_size * vision.config().patch_size) as i32;
            let px = Array::from_slice(&prepared.pixel_values, &[n, width]);
            let soft = vision
                .forward(&px, prepared.grid)
                .context("vision tower forward")?;
            // Materialize now so the tower's intermediates are freed before the
            // 30-layer text prefill allocates its own working set.
            soft.eval().context("vision: eval soft tokens")?;
            Ok(soft)
        }

        /// Forward with image inputs spliced in.
        ///
        /// `input_ids` must already contain one run of `image_token_id` per
        /// image, of exactly `prepared[i].num_soft_tokens` length (the caller
        /// builds this while rendering the chat template). Each run's embedding
        /// rows are replaced wholesale by that image's soft tokens.
        ///
        /// Ordering matters: upstream scales the *text* embeddings by
        /// `sqrt(hidden_size)` and then `masked_scatter`s the **unscaled** image
        /// features over them, so the splice happens after the scale, not
        /// before.
        pub fn forward_with_images(
            &self,
            input_ids: &[u32],
            images: &[crate::gemma4_vision::PreparedImage],
            cache: &mut NativeGemma4PromptCache,
            slice_last_token: bool,
        ) -> Result<Array> {
            if images.is_empty() {
                let ids = Array::from_slice(input_ids, &[1, input_ids.len() as i32]);
                return self.forward_array_impl(&ids, cache, slice_last_token);
            }
            let (runs, soft_tokens) = self.encode_images_for_prompt(input_ids, images)?;
            let ids = Array::from_slice(input_ids, &[1, input_ids.len() as i32]);
            self.forward_array_impl_with_soft(&ids, cache, slice_last_token, 0, &runs, &soft_tokens)
        }

        /// Locate each image's placeholder run in `prompt_ids` and encode the
        /// matching image, checking the two agree on length.
        ///
        /// Split out from [`Self::forward_with_images`] because chunked prefill
        /// has to encode every image **once**, before the chunk loop starts,
        /// and then splice slices of the result into whichever chunks the runs
        /// land in. Returns prompt-global runs paired with `[len_i, hidden]`
        /// soft tokens, ready for [`Self::forward_last_token_with_soft`].
        pub fn encode_images_for_prompt(
            &self,
            prompt_ids: &[u32],
            images: &[crate::gemma4_vision::PreparedImage],
        ) -> Result<(Vec<(usize, usize)>, Vec<Array>)> {
            let image_token = self
                .config
                .image_token_id
                .ok_or_else(|| anyhow!("config.json has no image_token_id"))?;

            let runs = image_token_runs(prompt_ids, image_token);
            if runs.len() != images.len() {
                return Err(anyhow!(
                    "prompt has {} image-token run(s) but {} image(s) were supplied",
                    runs.len(),
                    images.len()
                ));
            }

            let mut soft_tokens = Vec::with_capacity(images.len());
            for (idx, (prepared, run)) in images.iter().zip(runs.iter()).enumerate() {
                if prepared.num_soft_tokens != run.1 {
                    return Err(anyhow!(
                        "image {idx} encodes to {} soft tokens but the prompt reserved {} \
                         image_token placeholders",
                        prepared.num_soft_tokens,
                        run.1
                    ));
                }
                soft_tokens.push(
                    self.encode_prepared_image(prepared)
                        .with_context(|| format!("encoding image {idx}"))?,
                );
            }
            Ok((runs, soft_tokens))
        }

        /// [`Self::forward_last_token`] for one chunk of an image-bearing
        /// prompt.
        ///
        /// `span_start` is the chunk's offset in the full prompt; `runs` /
        /// `soft` come from [`Self::encode_images_for_prompt`] and stay
        /// prompt-global across the whole chunk loop. Chunks that contain no
        /// placeholder rows fall through to the plain text path.
        pub fn forward_last_token_with_soft(
            &self,
            input_ids: &[u32],
            cache: &mut NativeGemma4PromptCache,
            span_start: usize,
            runs: &[(usize, usize)],
            soft: &[Array],
        ) -> Result<Array> {
            if input_ids.is_empty() {
                return Err(anyhow!("forward_last_token_with_soft: empty input_ids"));
            }
            let ids = Array::from_slice(input_ids, &[1, input_ids.len() as i32]);
            self.forward_array_impl_with_soft(
                &ids,
                cache,
                /* slice_last_token */ true,
                span_start as i32,
                runs,
                soft,
            )
        }

        /// Prefill-optimized variant of `forward()` — returns only the LAST
        /// position's logits `[B, 1, V]` instead of the full `[B, L, V]`.
        /// At long prefill (e.g. L=8192) this skips the dominant `8192 ×
        /// hidden × vocab` quantized matmul that gets immediately discarded
        /// by `argmax_last_token_lazy`. mlx-lm doesn't do this optimization —
        /// it computes full prefill logits and slices afterwards. Saves
        /// ~2 s at 8K prefill on Gemma 4 26B-A4B vs full logits.
        ///
        /// Safe ONLY when the caller will reduce to last position anyway
        /// (greedy / generate / decode loop). For full-position logits
        /// (forward_probe, dump_hidden output divergence checks, training
        /// gradients) use `forward()` instead.
        pub fn forward_last_token(
            &self,
            input_ids: &[u32],
            cache: &mut NativeGemma4PromptCache,
        ) -> Result<Array> {
            if input_ids.is_empty() {
                return Err(anyhow!("forward_last_token: empty input_ids"));
            }
            let l = input_ids.len() as i32;
            let ids = Array::from_slice(input_ids, &[1, l]);
            self.forward_array_last_token(&ids, cache)
        }

        /// Same as `forward()` but accepts the token-id input as an already-
        /// shaped `[1, L]` mlx Array. Used by the async-pipelined decode loop
        /// in `generate()` (Phase 1.5 P4) so the previous step's argmax can be
        /// fed back without round-tripping through the CPU — the Array stays
        /// lazy until the next step's logits are scheduled, enabling the
        /// GPU to overlap step N+1's prefill with step N's argmax eval.
        ///
        /// Mirrors mlx_lm's pattern: `_step(y)` where `y` is the previous
        /// step's sampled token (still lazy at this point).
        pub fn forward_array(
            &self,
            input_ids: &Array,
            cache: &mut NativeGemma4PromptCache,
        ) -> Result<Array> {
            self.forward_array_impl(input_ids, cache, /* slice_last_token */ false)
        }

        /// Prefill-optimized variant of `forward_array()` — slices the
        /// post-final-norm hidden state to the last position before the
        /// tied lm_head quantized matmul. Returns `[B, 1, V]` logits.
        ///
        /// See `forward_last_token` for the rationale + safety constraints.
        pub fn forward_array_last_token(
            &self,
            input_ids: &Array,
            cache: &mut NativeGemma4PromptCache,
        ) -> Result<Array> {
            self.forward_array_impl(input_ids, cache, /* slice_last_token */ true)
        }

        /// Shared implementation of `forward_array` and
        /// `forward_array_last_token`. The `slice_last_token` flag controls
        /// whether the post-final-norm hidden state is reduced to its last
        /// position before applying the tied lm_head + softcap (saves the
        /// dominant prefill matmul cost when callers only need next-token
        /// logits).
        fn forward_array_impl(
            &self,
            input_ids: &Array,
            cache: &mut NativeGemma4PromptCache,
            slice_last_token: bool,
        ) -> Result<Array> {
            self.forward_array_impl_with_soft(input_ids, cache, slice_last_token, 0, &[], &[])
        }

        /// [`Self::forward_array_impl`] with optional vision splicing.
        ///
        /// `runs` are prompt-global `(start, len)` spans of `image_token_id`,
        /// ascending and non-overlapping; `soft[i]` is `[len_i, hidden]`.
        /// `span_start` is where `input_ids` begins in that prompt — `0` for a
        /// single-pass prefill, the running offset for a chunked one.
        fn forward_array_impl_with_soft(
            &self,
            input_ids: &Array,
            cache: &mut NativeGemma4PromptCache,
            slice_last_token: bool,
            span_start: i32,
            runs: &[(usize, usize)],
            soft: &[Array],
        ) -> Result<Array> {
            let cfg = &self.config.text_config;
            assert_eq!(
                cfg.hidden_size_per_layer_input, 0,
                "forward: hidden_size_per_layer_input > 0 (Gemma 4 2B/4B path) not yet supported"
            );
            let shape = input_ids.shape();
            if shape.len() != 2 || shape[0] != 1 {
                return Err(anyhow!(
                    "forward_array: expected input_ids shape [1, L], got {shape:?}"
                ));
            }
            let l = shape[1];

            // (1) Token ids → quantized embedding rows (dequantized to bf16).
            // residual stream stays bf16 (legacy f32-cast
            // opt-out removed 2026-05-14).
            let ids_flat =
                mlx_rs::ops::reshape(input_ids, &[l]).context("forward: flatten input_ids")?;
            let embed_rows = self.embed_lookup_affine(&ids_flat)?; // [L, hidden] in bf16
            let h = mlx_rs::ops::reshape(&embed_rows, &[1, l, cfg.hidden_size as i32])
                .context("forward: reshape embed [B, L, H]")?;

            // (2) h *= sqrt(hidden_size) — Phase 1.5 P8: cached const_embed_scale.
            let h = mlx_rs::ops::multiply(&h, &self.const_embed_scale)
                .context("forward: × sqrt(hidden_size)")?;

            // (2b) Vision splice. Upstream applies the embed scale to the text
            // embeddings and then `masked_scatter`s the image features over
            // them, so the soft tokens must land *after* the multiply and stay
            // unscaled. Rebuilding h by concatenating the untouched spans is
            // cheaper than a scatter and keeps everything on the lazy graph.
            let h = if runs.is_empty() {
                h
            } else {
                self.splice_soft_tokens_for_span(&h, span_start, l, runs, soft)?
            };
            dump_hidden(&h, "embed")?;

            // (3) Decoder layers.
            //
            // every K layers, schedule a non-blocking
            // `async_eval` on the running residual `h` to bound the lazy
            // graph depth. SDPA's input traversal cost (the dominant
            // long-context overhead per phase_1_5 diagnosis) scales with
            // the depth of pending ops on q/k/v — capping it via periodic
            // drain trades a small CPU sync for substantially cheaper FFI
            // per SDPA call. K is tunable via env (default 0 = disabled).
            let mut h = h;
            let n_layers = cfg.num_hidden_layers;
            let async_eval_every_k = gemma4_async_eval_every_k();
            for idx in 0..n_layers {
                let layer_cache = cache
                    .layer_mut(idx)
                    .ok_or_else(|| anyhow!("forward: missing cache slot {idx}"))?;
                h = self.decoder_layer_forward(&h, idx, layer_cache)?;
                dump_hidden(&h, &format!("L{idx:02}"))?;
                if async_eval_every_k > 0 && (idx + 1) % async_eval_every_k == 0 {
                    // Non-blocking GPU drain. Does NOT sync the CPU; just
                    // schedules pending ops so subsequent layers' SDPA
                    // input traversal hits a shallower lazy graph.
                    mlx_rs::transforms::async_eval([&h])
                        .context("forward: per-K-layer async_eval")?;
                }
            }

            // (4) Final norm.
            let h =
                rms_norm(&h, &self.final_norm, cfg.rms_norm_eps).context("forward: final_norm")?;
            dump_hidden(&h, "final_norm")?;

            // MTP capture hook — when the drafter is active and the MTP
            // step requested last-layer h capture, stash a clone here.
            // Plain `&Array::clone` is a refcount bump on the underlying
            // lazy graph, not a deep copy. No cost when capture is off.
            if self
                .mtp_capture_enabled
                .load(std::sync::atomic::Ordering::Relaxed)
            {
                if let Ok(mut slot) = self.mtp_capture_slot.lock() {
                    *slot = Some(h.clone());
                }
            }

            // Optional: reduce h to last position before lm_head when the
            // caller only needs next-token logits. Skips the dominant
            // `L × hidden × vocab` quantized matmul (e.g. 8192 × 2880 ×
            // 262144 / 2 ≈ 3 TFLOPs at 8K, ~8000× more than slice-then-mm).
            // L=1 path is a no-op so the guard simply avoids the extra
            // take_axis dispatch at decode.
            let h_for_lm_head = if slice_last_token && l > 1 {
                let last_pos = (l - 1) as i32;
                let last_idx = Array::from_slice(&[last_pos], &[1]);
                mlx_rs::ops::indexing::take_axis(&h, &last_idx, 1)
                    .context("forward: slice h to last position for lm_head")?
            } else {
                h
            };

            // Phase B (v0.6.0 tool-call robustness) — stash a clone of
            // `h_for_lm_head` for the backend's logit-correction kernel.
            // Same lazy-graph refcount-bump semantics as the MTP capture
            // above; no overhead when capture is off.
            if self
                .correction_capture_enabled
                .load(std::sync::atomic::Ordering::Relaxed)
            {
                if let Ok(mut slot) = self.correction_capture_slot.lock() {
                    *slot = Some(h_for_lm_head.clone());
                }
            }

            // (5+6) Tied lm_head + softcap.
            let logits = self.tied_lm_head_plus_softcap(&h_for_lm_head)?;

            dump_hidden(&logits, "logits")?;
            Ok(logits)
        }

        /// Tied lm_head (`h @ embed.weight.T` via quantized matmul) followed
        /// by Gemma 4's final logit softcap. Extracted from
        /// `forward_array_impl` so the MTP drafter path can reuse the same
        /// trunk lm_head when projecting drafter hidden → vocab logits.
        pub fn tied_lm_head_plus_softcap(&self, h: &Array) -> Result<Array> {
            let logits = match &self.embed_tokens {
                EmbedTokensWeights::Bf16(w) => {
                    let w_t = mlx_rs::ops::transpose_axes(w, &[1, 0])
                        .context("tied_lm_head(bf16): transpose embed weights")?;
                    mlx_rs::ops::matmul(h, &w_t).context("tied_lm_head(bf16): matmul h @ W.T")?
                }
                EmbedTokensWeights::Quantized(embed) => quantized_matmul_with_mode(
                    h,
                    &embed.weight,
                    &embed.scales,
                    embed.biases.as_ref(),
                    /* transpose */ true,
                    embed.group_size,
                    embed.bits,
                    embed.mode,
                )
                .context("tied_lm_head_plus_softcap: quantized_matmul failed")?,
            };
            self.apply_logit_softcap(&logits)
        }

        /// Argmax of logits at the last sequence position. Convenience for
        /// greedy decoding.
        pub fn argmax_last_token(&self, logits: &Array) -> Result<u32> {
            let argmax = self.argmax_last_token_lazy(logits)?;
            self.read_token_u32(&argmax)
        }

        /// Lazy variant of `argmax_last_token` — returns the argmax as a
        /// lazy `[1, 1]` Int32 Array WITHOUT forcing GPU evaluation. The
        /// caller can `mlx_rs::transforms::async_eval(&[&out])` to schedule
        /// async eval, then build the next step's forward graph using `out`
        /// as input (passed to `forward_array`). The GPU overlaps step N+1's
        /// prefill with step N's argmax. Sync via `read_token_u32(&out)` once
        /// the next step has been scheduled.
        ///
        /// Output shape: `[1, 1]` Int32 — the inner dim is the token id
        /// position (kept so it can be fed back to `forward_array` directly).
        pub fn argmax_last_token_lazy(&self, logits: &Array) -> Result<Array> {
            // logits shape: [B, L, V]. Take last position along axis 1.
            // cache the L=1 case (decode hot path) and only
            // build a fresh index when L>1 (prefill, fires once).
            let l = logits.shape()[1];
            let last_pos = l - 1;
            let last_idx_owned;
            let last_idx_ref: &Array = if last_pos == 0 {
                &self.const_last_idx_one
            } else {
                last_idx_owned = Array::from_slice(&[last_pos], &[1]);
                &last_idx_owned
            };
            let last_logits = mlx_rs::ops::indexing::take_axis(logits, last_idx_ref, 1)
                .context("argmax_last_token_lazy: take_axis(L-1)")?;
            // last_logits shape [B, 1, V] → argmax over axis -1, KEEPING the
            // squeezed axis so the result is [B, 1] = [1, 1], directly
            // feedable to forward_array.
            let argmax =
                mlx_rs::ops::indexing::argmax_axis(&last_logits, -1, /* keep_dims */ false)
                    .context("argmax_last_token_lazy: argmax_axis")?;
            // Cast to Int32 for token-id semantics; result is lazy.
            argmax
                .as_dtype(mlx_rs::Dtype::Int32)
                .context("argmax_last_token_lazy: cast to Int32")
        }

        /// Force-evaluate a single-element `[1, 1]` (or `[1]`) Int32 Array and
        /// read out the token id as a u32.
        pub fn read_token_u32(&self, argmax: &Array) -> Result<u32> {
            argmax.eval().context("read_token_u32: eval")?;
            let v: &[i32] = argmax.as_slice();
            Ok(v[0] as u32)
        }

        /// Convenience helper for greedy multi-step generation.
        ///
        /// Steps:
        ///   1. Allocate a fresh cache from `make_cache()`.
        ///   2. Prefill the whole `prompt_ids` in one `forward()` call.
        ///   3. Up to `cfg.max_new_tokens - 1` decode steps, each feeding the
        ///      previously argmaxed token id back through the model.
        ///   4. Optionally early-exits when `eos_tokens()` produces a match.
        ///
        /// Returns prefill / decode wall-clock so callers can compute a
        /// decode tokens-per-second baseline (e.g. compare against
        /// `mlx_lm.server`'s warm baseline of ~37 tok/s on M4 Pro).
        pub fn generate(&self, prompt_ids: &[u32], cfg: &GenerateConfig) -> Result<GenerateStats> {
            self.generate_with_cache(prompt_ids, cfg, None)
        }

        /// Prefix-cache-aware variant of [`Self::generate`]. When `cache_in`
        /// is `Some`, the supplied cache is reused (skip make_cache).
        /// Crucially, the cache may already contain a prefix of `prompt_ids`
        /// — `forward_last_token`/`forward_array` advance from `cache.offset()`,
        /// so this naturally extends a previously cached prefix.
        ///
        /// Contract: caller is responsible for ensuring `cache.offset() <=
        /// prompt_ids.len()` AND `prompt_ids[..cache.offset()]` matches the
        /// tokens already in the cache. Use [`NativeGemma4PromptCache::clone`]
        /// + [`NativeGemma4PromptCache::truncate_to`] to manage prefix-cache
        /// fork/extend semantics from the outside (see `Gemma4Backend`).
        pub fn generate_with_cache(
            &self,
            prompt_ids: &[u32],
            cfg: &GenerateConfig,
            cache_in: Option<&mut NativeGemma4PromptCache>,
        ) -> Result<GenerateStats> {
            self.generate_with_cache_and_images(prompt_ids, &[], cfg, cache_in)
        }

        /// [`Self::generate_with_cache`] with image inputs.
        ///
        /// `prompt_ids` must carry one run of `image_token_id` per image, sized
        /// to that image's `num_soft_tokens`. Only the prefill differs — decode
        /// steps are pure text and take the unchanged path, so MTP /
        /// lookup-spec / sampling all behave identically.
        pub fn generate_with_cache_and_images(
            &self,
            prompt_ids: &[u32],
            images: &[crate::gemma4_vision::PreparedImage],
            cfg: &GenerateConfig,
            cache_in: Option<&mut NativeGemma4PromptCache>,
        ) -> Result<GenerateStats> {
            if prompt_ids.is_empty() {
                return Err(anyhow!("generate: empty prompt"));
            }
            if cfg.max_new_tokens == 0 {
                return Err(anyhow!("generate: max_new_tokens must be >= 1"));
            }

            let mut owned_cache;
            let cache_ref: &mut NativeGemma4PromptCache = match cache_in {
                Some(c) => c,
                None => {
                    // Per-request adaptive TQ + simple Q4: when MODE=auto,
                    // the threshold is compared against this prompt's length.
                    // MODE=on/off bypass the threshold. Caller-supplied
                    // caches keep whatever choice they were built with
                    // (we don't override mid-flight).
                    let force_tq = resolve_tq_for_request(prompt_ids.len());
                    let force_quant_kv = resolve_quant_kv_for_request(prompt_ids.len());
                    if gemma4_tq_mode() == Gemma4TqMode::Auto {
                        eprintln!(
                            "[gemma4] tq_auto: prompt_tokens={} threshold={} → tq={}",
                            prompt_ids.len(),
                            gemma4_tq_auto_threshold(),
                            if force_tq { "ON" } else { "OFF" }
                        );
                    }
                    if gemma4_quant_kv_mode() == Gemma4QuantKvMode::Auto {
                        eprintln!(
                            "[gemma4] quant_kv_auto: prompt_tokens={} threshold={} → q4={}",
                            prompt_ids.len(),
                            gemma4_quant_kv_auto_threshold(),
                            if force_quant_kv { "ON" } else { "OFF" }
                        );
                    }
                    owned_cache = self.make_cache_with_tq(Some(force_tq), Some(force_quant_kv));
                    &mut owned_cache
                }
            };
            // Rebind so the body can use `cache` (which it does heavily)
            // and `&mut cache` works (reborrows the underlying cache).
            let mut cache: &mut NativeGemma4PromptCache = cache_ref;
            // Silence the "unused mut" lint when caller passes their own
            // cache (we still need `mut` because the body mutates via this
            // binding).
            let _ = &mut cache;
            let mut generated: Vec<u32> = Vec::with_capacity(cfg.max_new_tokens);
            let eos = self.eos_tokens().to_vec();

            // ── Prefill ──────────────────────────────────────────────────
            // Use forward_last_token: greedy generate only needs the
            // next-token logits, so the post-final-norm hidden state is
            // sliced to position L-1 before the tied lm_head. At long
            // prefill (e.g. L=8192, hidden=2880, vocab=262144) this skips
            // the dominant ~3 TFLOP qmatmul that would be discarded by
            // `argmax_last_token_lazy` anyway. forward() retains full-
            // logits semantics for forward_probe / debug callers.
            let prefill_start = Instant::now();
            let logits = if images.is_empty() {
                self.forward_last_token(prompt_ids, &mut cache)
                    .context("generate: prefill forward_last_token")?
            } else {
                // Image prefill still slices to the last token; the only
                // difference is that the placeholder rows carry vision
                // features instead of `<|image|>` embeddings.
                self.forward_with_images(prompt_ids, images, &mut cache, true)
                    .context("generate: prefill forward_with_images")?
            };
            // Lazy argmax — schedule async eval so the GPU starts work while
            // we queue up the first decode step's graph.
            let mut current = self
                .argmax_last_token_lazy(&logits)
                .context("generate: prefill argmax_lazy")?;
            mlx_rs::transforms::async_eval([&current]).context("generate: prefill async_eval")?;
            // mirror mlx-lm Python's `mx.eval(current)` after
            // `mx.async_eval` so prefill GPU work is attributed to prefill_ms,
            // NOT to decode step[0]. Without this block prefill_ms is just
            // lazy-build time and decode step[0] sync-waits for the prefill
            // drain (was the source of the "25 ms gap" measurement artifact —
            // see phase_1_8_post_m4_8_real_gap_25ms.md). Default ON; set
            // `LUMEN_GEMMA4_PREFILL_SYNC=0` to opt out for diagnostics.
            let prefill_sync = std::env::var("LUMEN_GEMMA4_PREFILL_SYNC")
                .map(|v| v != "0")
                .unwrap_or(true);
            if prefill_sync {
                current.eval().context("generate: prefill blocking eval")?;
            }
            let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1e3;
            // Snapshot substage timings accumulated during prefill ONLY.
            // Counters reset to 0 here so decode timing is isolated downstream.
            // Activates only when one of the breakdown modes is enabled (otherwise
            // counters are all 0). Use `LUMEN_GEMMA4_BREAKDOWN=1` for true GPU
            // attribution (eval barriers between substages, drained by `current.eval()`
            // above); `LUMEN_GEMMA4_HONEST_BREAKDOWN=1` for Rust-dispatch-only.
            let prefill_breakdown = if gemma4_any_breakdown_active() {
                Some(take_gemma4_breakdown())
            } else {
                None
            };

            // ── Sampled decode path (temperature / top-p / repeat-penalty) ──
            //
            // Routes here only when `cfg.sampling` is `Some` AND the
            // config is non-greedy (temperature > 0 OR repeat_penalty != 1).
            // The greedy fast path (with async pipelining + MTP + lookup
            // spec) stays bit-identical when this branch is not taken.
            //
            // The prefill argmax `current` is already lazy from above; we
            // discard it and re-sample the prefill logits with the same
            // sampler so the very first decoded token also respects
            // temperature / top-p.
            if cfg
                .sampling
                .as_ref()
                .map(|s| !s.is_greedy())
                .unwrap_or(false)
            {
                use crate::gemma4_sampling::imp::{Xorshift64, sample_next_token_with_eos_guard};
                let sampling = cfg.sampling.as_ref().unwrap();
                // Discard the lazy prefill argmax; we'll sample from
                // `logits` (the [1, 1, V] prefill output already in scope)
                // directly. No extra prefill pass.
                let _ = current;
                let mut rng = Xorshift64::new(sampling.seed);
                let runaway = lumen_core::runaway::RunawayDetector::from_env();
                let mut thinking_budget = crate::gemma4_thinking::ChannelBudget::from_env();
                let decode_start = Instant::now();

                // Soft-EOS suppression (see gemma4_sampling docs).
                let min_tokens_before_eos: usize = std::env::var("LUMEN_MIN_TOKENS_BEFORE_EOS")
                    .ok()
                    .and_then(|s| s.trim().parse::<usize>().ok())
                    .unwrap_or(0);
                let eos_top_k_guard: usize = std::env::var("LUMEN_EOS_TOP_K_GUARD")
                    .ok()
                    .and_then(|s| s.trim().parse::<usize>().ok())
                    .unwrap_or(0);
                let eos_min_logit_margin: f32 = std::env::var("LUMEN_EOS_MIN_LOGIT_MARGIN")
                    .ok()
                    .and_then(|s| s.trim().parse::<f32>().ok())
                    .unwrap_or(0.0);

                let first_tok = sample_next_token_with_eos_guard(
                    &logits,
                    &generated,
                    sampling,
                    &mut rng,
                    min_tokens_before_eos,
                    eos_top_k_guard,
                    eos_min_logit_margin,
                    &eos,
                    None,
                )
                .context("generate(sampled): sample prefill token")?;
                generated.push(first_tok);
                let mut hit_eos_s = cfg.stop_on_eos && eos.contains(&first_tok);

                let mut current_u32 = first_tok;
                let mut decode_steps_s: usize = 0;
                while generated.len() < cfg.max_new_tokens && !hit_eos_s {
                    let input = Array::from_slice(&[current_u32 as i32], &[1, 1])
                        .as_dtype(mlx_rs::Dtype::Int32)
                        .context("generate(sampled): build input array")?;
                    let step_logits = self
                        .forward_array_last_token(&input, &mut cache)
                        .context("generate(sampled): decode forward")?;
                    let sampled = sample_next_token_with_eos_guard(
                        &step_logits,
                        &generated,
                        sampling,
                        &mut rng,
                        min_tokens_before_eos,
                        eos_top_k_guard,
                        eos_min_logit_margin,
                        &eos,
                        None,
                    )
                    .context("generate(sampled): sample step")?;
                    thinking_budget.observe(sampled);
                    let next_tok = if let Some(forced) = thinking_budget.try_force_close() {
                        eprintln!(
                            "[thinking-budget] forcing channel close at step {decode_steps_s} ({} tokens)",
                            generated.len()
                        );
                        forced
                    } else if thinking_budget.should_block_channel_open()
                        && sampled == crate::gemma4_thinking::TOK_CHANNEL_OPEN
                    {
                        eprintln!(
                            "[thinking-budget] blocking channel re-open at step {decode_steps_s} ({} tokens); emitting <turn|>",
                            generated.len()
                        );
                        crate::gemma4_thinking::TOK_TURN_CLOSE
                    } else {
                        sampled
                    };
                    generated.push(next_tok);
                    decode_steps_s += 1;
                    if cfg.stop_on_eos && eos.contains(&next_tok) {
                        hit_eos_s = true;
                    }
                    if let Some(reason) = runaway.check(&generated) {
                        eprintln!(
                            "[runaway] sampled decode aborted at step {decode_steps_s} ({} tokens): {reason}",
                            generated.len()
                        );
                        hit_eos_s = true;
                    }
                    if thinking_budget.should_hard_break() {
                        eprintln!(
                            "[thinking-budget] hard break at step {decode_steps_s} ({} tokens) — force-close did not help",
                            generated.len()
                        );
                        hit_eos_s = true;
                    }
                    current_u32 = next_tok;
                }
                let decode_ms = decode_start.elapsed().as_secs_f64() * 1e3;
                let decode_tok_per_sec = if decode_ms > 0.0 && decode_steps_s > 0 {
                    decode_steps_s as f64 * 1e3 / decode_ms
                } else {
                    0.0
                };
                return Ok(GenerateStats {
                    prompt_tokens: prompt_ids.len(),
                    generated_tokens: generated,
                    prefill_ms,
                    prefill_breakdown,
                    decode_ms,
                    decode_steps: decode_steps_s,
                    decode_tok_per_sec,
                    stopped_on_eos: hit_eos_s,
                });
            }

            // ── MTP decode path (opt-in: LUMEN_GEMMA4_MTP=1 + try_enable_mtp()) ──
            //
            // Routes through `mtp_step()` (Step A→E speculative decoding).
            // Each call yields up to `n_draft + 2` tokens. Falls back to
            // the standard async-pipelined single-token loop when the env
            // gate is off OR no drafter is loaded — bit-identical behavior
            // in the default-OFF case (zero regression).
            if self.mtp_enabled() && Self::mtp_decode_enabled_env() {
                let n_draft = Self::mtp_block_size_env();
                let decode_start = Instant::now();
                // Prefill produced `current` = argmax of position L-1's
                // logits = first decoded token at position L. The OFF path
                // pushes this directly to `generated` then loops; MTP path
                // must do the same BEFORE calling mtp_step (which advances
                // past this token).
                let mut current_u32 = self
                    .read_token_u32(&current)
                    .context("generate(MTP): read prefill first token")?;
                generated.push(current_u32);
                let mut decode_steps_mtp: usize = 0;
                let mut hit_eos_mtp = cfg.stop_on_eos && eos.contains(&current_u32);
                let runaway_mtp = lumen_core::runaway::RunawayDetector::from_env();
                let mut budget_mtp = crate::gemma4_thinking::ChannelBudget::from_env();
                budget_mtp.observe(current_u32);
                while generated.len() < cfg.max_new_tokens && !hit_eos_mtp {
                    let out = self
                        .mtp_step(&mut cache, current_u32, n_draft)
                        .context("generate(MTP): mtp_step")?;
                    decode_steps_mtp += 1;
                    // The last element of `committed` is the next-call input
                    // (correction on partial reject / bonus on full accept).
                    let next_input = *out.committed.last().unwrap();
                    for t in &out.committed {
                        if generated.len() >= cfg.max_new_tokens {
                            break;
                        }
                        generated.push(*t);
                        if cfg.stop_on_eos && eos.contains(t) {
                            hit_eos_mtp = true;
                            break;
                        }
                    }
                    if !hit_eos_mtp {
                        if let Some(reason) = runaway_mtp.check(&generated) {
                            eprintln!(
                                "[runaway] MTP decode aborted at step {decode_steps_mtp} ({} tokens): {reason}",
                                generated.len()
                            );
                            hit_eos_mtp = true;
                        }
                        for t in &out.committed {
                            budget_mtp.observe(*t);
                        }
                        if budget_mtp.exceeded() {
                            eprintln!(
                                "[thinking-budget] MTP decode aborted at step {decode_steps_mtp} ({} tokens, count={})",
                                generated.len(),
                                budget_mtp.thought_count(),
                            );
                            hit_eos_mtp = true;
                        }
                    }
                    current_u32 = next_input;
                }
                let decode_ms = decode_start.elapsed().as_secs_f64() * 1e3;
                let decode_tok_per_sec = if decode_ms > 0.0 && !generated.is_empty() {
                    generated.len() as f64 * 1e3 / decode_ms
                } else {
                    0.0
                };
                return Ok(GenerateStats {
                    prompt_tokens: prompt_ids.len(),
                    generated_tokens: generated,
                    prefill_ms,
                    prefill_breakdown,
                    decode_ms,
                    decode_steps: decode_steps_mtp,
                    decode_tok_per_sec,
                    stopped_on_eos: hit_eos_mtp,
                });
            }

            // ── Prompt-Lookup Decoding (opt-in: LUMEN_GEMMA4_LOOKUP_SPEC=1) ──
            //
            // Drafter-free speculative decoding via n-gram match on the
            // generated context. Each step: trunk decode → lookup match
            // (CPU) → trunk verify [1, K+1] → accept-reject → rollback.
            // No drafter weights, no GPU draft forwards. Falls back to a
            // single-token decode when no match is found (zero overhead).
            //
            // Tuning:
            //   LUMEN_GEMMA4_LOOKUP_N (default 3) — prefix length
            //   LUMEN_GEMMA4_LOOKUP_K (default 10) — max draft length
            if std::env::var("LUMEN_GEMMA4_LOOKUP_SPEC")
                .map(|v| v == "1")
                .unwrap_or(false)
            {
                let n_lookup: usize = std::env::var("LUMEN_GEMMA4_LOOKUP_N")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(3);
                let n_draft: usize = std::env::var("LUMEN_GEMMA4_LOOKUP_K")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(10);
                let decode_start = Instant::now();
                let mut current_u32 = self
                    .read_token_u32(&current)
                    .context("generate(LOOKUP): read prefill first token")?;
                generated.push(current_u32);
                let mut decode_steps_lk: usize = 0;
                let mut hit_eos_lk = cfg.stop_on_eos && eos.contains(&current_u32);
                // History = prompt_ids ++ generated. The lookup_step contract
                // expects `history` to end with the current uncommitted
                // committed_token (= current_u32). After push above, that's
                // exactly the case.
                let mut history: Vec<u32> =
                    Vec::with_capacity(prompt_ids.len() + cfg.max_new_tokens);
                history.extend_from_slice(prompt_ids);
                history.push(current_u32);
                let runaway_lk = lumen_core::runaway::RunawayDetector::from_env();
                let mut budget_lk = crate::gemma4_thinking::ChannelBudget::from_env();
                budget_lk.observe(current_u32);
                while generated.len() < cfg.max_new_tokens && !hit_eos_lk {
                    let out = self
                        .lookup_step(&mut cache, current_u32, &history, n_lookup, n_draft)
                        .context("generate(LOOKUP): lookup_step")?;
                    decode_steps_lk += 1;
                    let next_input = *out.committed.last().unwrap();
                    for t in &out.committed {
                        if generated.len() >= cfg.max_new_tokens {
                            break;
                        }
                        generated.push(*t);
                        history.push(*t);
                        if cfg.stop_on_eos && eos.contains(t) {
                            hit_eos_lk = true;
                            break;
                        }
                    }
                    if !hit_eos_lk {
                        if let Some(reason) = runaway_lk.check(&generated) {
                            eprintln!(
                                "[runaway] LOOKUP decode aborted at step {decode_steps_lk} ({} tokens): {reason}",
                                generated.len()
                            );
                            hit_eos_lk = true;
                        }
                        for t in &out.committed {
                            budget_lk.observe(*t);
                        }
                        if budget_lk.exceeded() {
                            eprintln!(
                                "[thinking-budget] LOOKUP decode aborted at step {decode_steps_lk} ({} tokens, count={})",
                                generated.len(),
                                budget_lk.thought_count(),
                            );
                            hit_eos_lk = true;
                        }
                    }
                    current_u32 = next_input;
                }
                let decode_ms = decode_start.elapsed().as_secs_f64() * 1e3;
                let decode_tok_per_sec = if decode_ms > 0.0 && !generated.is_empty() {
                    generated.len() as f64 * 1e3 / decode_ms
                } else {
                    0.0
                };
                return Ok(GenerateStats {
                    prompt_tokens: prompt_ids.len(),
                    generated_tokens: generated,
                    prefill_ms,
                    prefill_breakdown,
                    decode_ms,
                    decode_steps: decode_steps_lk,
                    decode_tok_per_sec,
                    stopped_on_eos: hit_eos_lk,
                });
            }

            // ── Decode loop — async pipelining (mirrors mlx-lm's generate.py:455-470) ──
            //
            // At each iteration we have TWO tokens in flight:
            //   - `current`  — the argmax we're about to read back as u32
            //   - `next_lazy`— the next-step argmax we just scheduled
            //
            // GPU pipeline:
            //   T0:  prefill eval running   ──┐
            //   T1:  schedule step-1 forward + argmax (uses lazy `current`)
            //   T1:  async_eval(next_lazy)  ──┐
            //   T1:  sync read `current` (→ u32)
            //   T2:  step-1 eval running    ──┘ (already in GPU queue)
            //   T2:  schedule step-2 forward + argmax (uses lazy `next_lazy`)
            //   ...
            //
            // This overlaps the Rust-side EOS check + graph-building work
            // with the GPU's compute on the previous step.
            let decode_start = Instant::now();
            let mut decode_steps: usize = 0;
            let mut hit_eos = false;
            let runaway_greedy = lumen_core::runaway::RunawayDetector::from_env();
            let mut budget_greedy = crate::gemma4_thinking::ChannelBudget::from_env();

            // gated by `LUMEN_GEMMA4_PER_STEP_LATENCY=1`.
            // Collects per-step wall-clock + 3-substage breakdown so step[0]
            // outliers can be attributed to (a) CPU lazy-graph build, (b)
            // mlx async_eval scheduling, or (c) read_token GPU drain.
            // Supersedes the earlier `LUMEN_GEMMA4_DECODE_LOOP_TIMING`
            // aggregate-only mode (deleted 2026-05-14 — per-step strictly
            // dominates aggregate for warmup-residual localization).
            let per_step_latency = std::env::var("LUMEN_GEMMA4_PER_STEP_LATENCY")
                .map(|v| v == "1")
                .unwrap_or(false);
            let mut fwd_per_step: Vec<f64> = if per_step_latency {
                Vec::with_capacity(cfg.max_new_tokens)
            } else {
                Vec::new()
            };
            let mut ae_per_step: Vec<f64> = if per_step_latency {
                Vec::with_capacity(cfg.max_new_tokens)
            } else {
                Vec::new()
            };
            let mut rd_per_step: Vec<f64> = if per_step_latency {
                Vec::with_capacity(cfg.max_new_tokens)
            } else {
                Vec::new()
            };
            let mut step_latencies_ms: Vec<f64> = if per_step_latency {
                Vec::with_capacity(cfg.max_new_tokens)
            } else {
                Vec::new()
            };
            let mut last_step_mark = if per_step_latency {
                Some(Instant::now())
            } else {
                None
            };

            // Metal frame capture (.gputrace) for
            // apples-to-apples comparison with mlx-lm. Mirrors the warmup +
            // bounded window pattern in `scripts/mlxlm_metal_capture.py`.
            //   LUMEN_METAL_CAPTURE=<path.gputrace>   enable + output path
            //   LUMEN_METAL_CAPTURE_WARMUP=N         decode steps before
            //                                         start (default 5)
            //   LUMEN_METAL_CAPTURE_STEPS=M          decode steps captured
            //                                         (default 10)
            //   LUMEN_METAL_CAPTURE_EXIT=1           stop generate() once
            //                                         capture finishes
            let capture_path = std::env::var("LUMEN_METAL_CAPTURE").ok();
            let capture_warmup: usize = std::env::var("LUMEN_METAL_CAPTURE_WARMUP")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(5);
            let capture_steps_window: usize = std::env::var("LUMEN_METAL_CAPTURE_STEPS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(10);
            let capture_exit = std::env::var("LUMEN_METAL_CAPTURE_EXIT")
                .map(|v| v == "1")
                .unwrap_or(false);
            let mut capture_started = false;
            let mut capture_stopped = false;

            while generated.len() < cfg.max_new_tokens {
                if generated.len() + 1 == cfg.max_new_tokens {
                    // Last token — no need to schedule a next step.
                    let token = self
                        .read_token_u32(&current)
                        .context("generate: read final token")?;
                    generated.push(token);
                    if cfg.stop_on_eos && eos.contains(&token) {
                        hit_eos = true;
                    }
                    break;
                }

                // Reshape current [1, 1] argmax to [1, 1] (already is) and
                // feed into forward_array — graph builds without forcing
                // `current` to evaluate.
                let t_fwd_start = if per_step_latency {
                    Some(Instant::now())
                } else {
                    None
                };
                let next_logits = self
                    .forward_array_last_token(&current, &mut cache)
                    .context("generate: decode forward_array_last_token")?;
                let next_lazy = self
                    .argmax_last_token_lazy(&next_logits)
                    .context("generate: decode argmax_lazy")?;
                let fwd_ms = t_fwd_start
                    .map(|t| t.elapsed().as_secs_f64() * 1e3)
                    .unwrap_or(0.0);
                // Schedule async eval — GPU starts working on this while we
                // sync-read `current` below.
                let t_ae_start = if per_step_latency {
                    Some(Instant::now())
                } else {
                    None
                };
                mlx_rs::transforms::async_eval([&next_lazy])
                    .context("generate: decode async_eval")?;
                let ae_ms = t_ae_start
                    .map(|t| t.elapsed().as_secs_f64() * 1e3)
                    .unwrap_or(0.0);

                // Now sync-read the *current* token (which the GPU should
                // have already completed during the previous iteration's
                // graph-build work).
                let t_rd_start = if per_step_latency {
                    Some(Instant::now())
                } else {
                    None
                };
                let token = self
                    .read_token_u32(&current)
                    .context("generate: read current token")?;
                let rd_ms = t_rd_start
                    .map(|t| t.elapsed().as_secs_f64() * 1e3)
                    .unwrap_or(0.0);
                if per_step_latency {
                    fwd_per_step.push(fwd_ms);
                    ae_per_step.push(ae_ms);
                    rd_per_step.push(rd_ms);
                }
                generated.push(token);
                decode_steps += 1;

                // record wall-clock for
                // this step (from end-of-previous-step's read_token to end
                // of this step's read_token).
                if let Some(ref mut mark) = last_step_mark {
                    let now = Instant::now();
                    let ms = now.duration_since(*mark).as_secs_f64() * 1e3;
                    step_latencies_ms.push(ms);
                    *mark = now;
                }

                // Metal capture window: drain → start, drain → stop. Drain
                // forces in-flight async work to complete so the .gputrace
                // bundle is bounded to the [start, stop) decode range.
                if let Some(ref p) = capture_path {
                    if !capture_started && decode_steps >= capture_warmup {
                        mlx_rs::transforms::eval([&current, &next_lazy])
                            .context("generate: pre-capture eval drain")?;
                        mlx_rs::metal::start_capture(p)
                            .context("generate: metal::start_capture")?;
                        eprintln!(
                            "[metal-capture] started at decode_step={decode_steps} \
                             path={p} window={capture_steps_window}"
                        );
                        capture_started = true;
                    } else if capture_started
                        && !capture_stopped
                        && decode_steps >= capture_warmup + capture_steps_window
                    {
                        mlx_rs::transforms::eval([&current, &next_lazy])
                            .context("generate: pre-stop eval drain")?;
                        mlx_rs::metal::stop_capture().context("generate: metal::stop_capture")?;
                        eprintln!(
                            "[metal-capture] stopped after {capture_steps_window} \
                             captured decode_steps; open .gputrace in Xcode"
                        );
                        capture_stopped = true;
                        if capture_exit {
                            hit_eos = false;
                            break;
                        }
                    }
                }

                if cfg.stop_on_eos && eos.contains(&token) {
                    hit_eos = true;
                    // Even though we already scheduled `next_lazy`, we throw
                    // it away — caller's contract is to stop on EOS.
                    break;
                }
                if let Some(reason) = runaway_greedy.check(&generated) {
                    eprintln!(
                        "[runaway] greedy decode aborted at step {decode_steps} ({} tokens): {reason}",
                        generated.len()
                    );
                    hit_eos = true;
                    break;
                }
                budget_greedy.observe(token);
                if budget_greedy.exceeded() {
                    eprintln!(
                        "[thinking-budget] greedy decode aborted at step {decode_steps} ({} tokens, count={})",
                        generated.len(),
                        budget_greedy.thought_count(),
                    );
                    hit_eos = true;
                    break;
                }
                current = next_lazy;
            }
            // Defensive: ensure capture is stopped even if loop ends before
            // reaching capture_warmup + capture_steps_window (e.g. EOS).
            if capture_path.is_some() && capture_started && !capture_stopped {
                mlx_rs::metal::stop_capture()
                    .context("generate: metal::stop_capture (end-of-loop)")?;
                eprintln!("[metal-capture] stopped at end-of-loop (decode_steps={decode_steps})");
            }
            let decode_ms = decode_start.elapsed().as_secs_f64() * 1e3;

            // per-step wall-clock + substage breakdown.
            if per_step_latency && !step_latencies_ms.is_empty() {
                let n = step_latencies_ms.len();
                let have_sub =
                    fwd_per_step.len() == n && ae_per_step.len() == n && rd_per_step.len() == n;
                eprintln!("[per-step-latency] {} steps:", n);
                if have_sub {
                    eprintln!(
                        "  {:>4} | {:>10} | {:>8} {:>8} {:>8} | {:>8}",
                        "step", "wall_ms", "fwd_ms", "ae_ms", "rd_ms", "sum_ms"
                    );
                }
                for (i, ms) in step_latencies_ms.iter().enumerate() {
                    if have_sub {
                        let sum = fwd_per_step[i] + ae_per_step[i] + rd_per_step[i];
                        eprintln!(
                            "  [{:>3}] | {:>10.3} | {:>8.3} {:>8.3} {:>8.3} | {:>8.3}",
                            i, ms, fwd_per_step[i], ae_per_step[i], rd_per_step[i], sum
                        );
                    } else {
                        eprintln!("  step[{:>3}] {:8.3} ms", i, ms);
                    }
                }
                // Buckets: first 1, first 4, first 8, last quarter (steady).
                let mean_first = |k: usize| -> f64 {
                    let k = k.min(n);
                    if k == 0 {
                        0.0
                    } else {
                        step_latencies_ms[..k].iter().sum::<f64>() / k as f64
                    }
                };
                let last_q_start = n.saturating_sub(n / 4).max(1);
                let mean_last_quarter: f64 = if n - last_q_start > 0 {
                    step_latencies_ms[last_q_start..].iter().sum::<f64>()
                        / (n - last_q_start) as f64
                } else {
                    0.0
                };
                eprintln!(
                    "[per-step-latency] mean: first1={:.2} first4={:.2} first8={:.2} last-quarter={:.2} ms",
                    mean_first(1),
                    mean_first(4),
                    mean_first(8),
                    mean_last_quarter,
                );
            }
            let decode_tok_per_sec = if decode_ms > 0.0 && decode_steps > 0 {
                decode_steps as f64 * 1e3 / decode_ms
            } else {
                0.0
            };

            Ok(GenerateStats {
                prompt_tokens: prompt_ids.len(),
                generated_tokens: generated,
                prefill_ms,
                prefill_breakdown,
                decode_ms,
                decode_steps,
                decode_tok_per_sec,
                stopped_on_eos: hit_eos,
            })
        }
    }

    // ───────────────────────── generate() helpers ─────────────────────────

    /// Config for `NativeGemma4Model::generate()`.
    #[derive(Debug, Clone)]
    pub struct GenerateConfig {
        /// Total number of new tokens (including the very first argmax from
        /// the prefill output). Must be >= 1.
        pub max_new_tokens: usize,
        /// If true, stop as soon as any of `eos_tokens()` is produced.
        pub stop_on_eos: bool,
        /// Optional sampling config. `None` (or any config whose
        /// `is_greedy()` returns true) takes the existing GPU-pipelined
        /// argmax path — bit-identical to pre-sampling behavior. Anything
        /// else routes through the CPU sampler in `gemma4_sampling`
        /// (temperature / top-p / repeat-penalty).
        pub sampling: Option<crate::gemma4_sampling::imp::SamplingConfig>,
    }

    impl Default for GenerateConfig {
        fn default() -> Self {
            Self {
                max_new_tokens: 32,
                stop_on_eos: true,
                sampling: None,
            }
        }
    }

    /// Output of a single `generate()` call, including timing breakdown.
    #[derive(Debug, Clone)]
    pub struct GenerateStats {
        pub prompt_tokens: usize,
        pub generated_tokens: Vec<u32>,
        /// Wall-clock for the single prefill `forward()` pass (ms).
        pub prefill_ms: f64,
        /// Per-substage breakdown for the prefill forward only (decode excluded).
        /// `Some` only when `LUMEN_GEMMA4_BREAKDOWN=1` or
        /// `LUMEN_GEMMA4_HONEST_BREAKDOWN=1` is set.
        pub prefill_breakdown: Option<Gemma4Breakdown>,
        /// Wall-clock for all decode steps combined (ms). Excludes the first
        /// token (it comes from the prefill argmax).
        pub decode_ms: f64,
        /// Number of decode `forward()` calls actually executed.
        pub decode_steps: usize,
        /// `decode_steps * 1000 / decode_ms`. Zero when no decode steps ran.
        pub decode_tok_per_sec: f64,
        pub stopped_on_eos: bool,
    }

    // ───────────────────────── tests ─────────────────────────

    #[cfg(test)]
    mod tests {
        use super::*;

        const LMSTUDIO_CONFIG_PATH: &str = "/path/to/models/gemma-4-26b-a4b-mlx-4bit/config.json";

        fn minimal_config_json(layer_types: &str, with_quant: bool) -> String {
            let quant_block = if with_quant {
                r#","quantization":{"group_size":64,"bits":4,"mode":"affine","language_model.model.layers.0.mlp.gate_proj":{"group_size":64,"bits":8}}"#
            } else {
                ""
            };
            format!(
                r#"{{
                    "architectures": ["Gemma4ForConditionalGeneration"],
                    "model_type": "gemma4",
                    "eos_token_id": [1, 106, 50]
                    {quant_block},
                    "text_config": {{
                        "model_type": "gemma4_text",
                        "hidden_size": 2816,
                        "num_hidden_layers": 6,
                        "num_attention_heads": 16,
                        "num_key_value_heads": 8,
                        "num_global_key_value_heads": 2,
                        "head_dim": 256,
                        "global_head_dim": 512,
                        "vocab_size": 262144,
                        "rms_norm_eps": 1e-6,
                        "layer_types": [{layer_types}],
                        "sliding_window": 1024,
                        "sliding_window_pattern": 6,
                        "max_position_embeddings": 262144,
                        "rope_parameters": {{
                            "full_attention": {{
                                "partial_rotary_factor": 0.25,
                                "rope_theta": 1000000.0,
                                "rope_type": "proportional"
                            }},
                            "sliding_attention": {{
                                "rope_theta": 10000.0,
                                "rope_type": "default"
                            }}
                        }},
                        "attention_k_eq_v": true,
                        "enable_moe_block": true,
                        "num_experts": 128,
                        "top_k_experts": 8,
                        "moe_intermediate_size": 704,
                        "intermediate_size": 2112,
                        "final_logit_softcapping": 30.0,
                        "tie_word_embeddings": true
                    }}
                }}"#
            )
        }

        #[test]
        fn parses_minimal_gemma4_config_with_quant() {
            let json = minimal_config_json(
                r#""sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention""#,
                true,
            );
            let cfg: NativeGemma4Config = serde_json::from_str(&json).expect("parse");
            cfg.validate_gemma4_family().expect("validate");
            assert_eq!(cfg.model_type, "gemma4");
            assert_eq!(cfg.eos_token_ids, vec![1, 106, 50]);
            assert_eq!(cfg.text_config.num_hidden_layers, 6);
            assert!(cfg.text_config.layer_types[5].is_full());
            assert!(cfg.text_config.layer_types[0].is_sliding());
            assert!(cfg.text_config.attention_k_eq_v);
            let quant = cfg.effective_quantization().expect("quant present");
            assert_eq!(quant.bits, 4);
            assert_eq!(quant.mode, "affine");
            assert!(
                quant
                    .overrides
                    .contains_key("language_model.model.layers.0.mlp.gate_proj")
            );
        }

        /// MXFP4 ship-recipe config (group_size=32, bits=4, mode="mxfp4")
        /// validates clean, and `quant_params_for` dispatches MODE_MXFP4 for
        /// default tensors while a per-tensor AFFINE override (e.g. embed_tokens
        /// kept at higher precision inside an MXFP4 model — mirrors Qwen3.6's
        /// gate-layers-at-8-bit pattern) dispatches MODE_AFFINE.
        #[test]
        fn quant_params_for_dispatches_mxfp4_with_affine_override() {
            let json = minimal_config_json(
                r#""sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention""#,
                true,
            )
            .replace(
                r#","quantization":{"group_size":64,"bits":4,"mode":"affine","language_model.model.layers.0.mlp.gate_proj":{"group_size":64,"bits":8}}"#,
                r#","quantization":{"group_size":32,"bits":4,"mode":"mxfp4","language_model.model.embed_tokens":{"group_size":32,"bits":8}}"#,
            );
            let cfg: NativeGemma4Config = serde_json::from_str(&json).expect("parse");
            cfg.validate_gemma4_family().expect("mxfp4 validate");
            let quant = cfg.effective_quantization().expect("quant present");
            assert_eq!(quant.mode, "mxfp4");
            assert_eq!(quant.group_size, 32);
            // Default tensor → MXFP4 dispatch.
            let (gs, bits, mode) =
                quant_params_for(&cfg, "language_model.model.layers.0.mlp.gate_proj.weight")
                    .expect("default dispatch");
            assert_eq!(gs, 32);
            assert_eq!(bits, 4);
            assert_eq!(mode, MODE_MXFP4);
            // Override path → AFFINE dispatch at override's bit-width.
            let (gs, bits, mode) =
                quant_params_for(&cfg, "language_model.model.embed_tokens.weight")
                    .expect("override dispatch");
            assert_eq!(gs, 32);
            assert_eq!(bits, 8);
            assert_eq!(mode, MODE_AFFINE);
        }

        /// Mode strings outside the supported set {affine, mxfp4, mxfp8, nvfp4}
        /// must be rejected at validation time, with the offending mode echoed
        /// in the error message so `mlx_lm.convert` builds with an unknown
        /// `--q-mode` fail fast at load rather than producing silent garbage
        /// during decode. (nvfp4 / mxfp8 are now supported — use a deliberately
        /// bogus mode so this stays meaningful as the supported set grows.)
        #[test]
        fn rejects_unsupported_quant_mode() {
            let json = minimal_config_json(
                r#""sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention""#,
                true,
            )
            .replace(r#""mode":"affine""#, r#""mode":"notarealmode""#);
            let cfg: NativeGemma4Config = serde_json::from_str(&json).expect("parse");
            let err = cfg.validate_gemma4_family().unwrap_err().to_string();
            assert!(
                err.contains("notarealmode"),
                "error should echo bad mode: {err}"
            );
        }

        /// Overrides with an explicit `"mode": "mxfp4"` dispatch MODE_MXFP4
        /// instead of the default MODE_AFFINE. This matters when mlx-lm.convert
        /// emits per-tensor overrides during build (some quantize_model code
        /// paths inline the predicate's dict into the config). The post-build
        /// strip script removes redundant mode-matching overrides, but the
        /// loader must still handle the explicit-mode case for forward-compat
        /// with future mixed-mode configs (e.g. MXFP4 default + mxfp4-with-
        /// different-group-size override, or future MXFP8 in either slot).
        #[test]
        fn quant_params_for_honors_mxfp4_override_mode() {
            let json = minimal_config_json(
                r#""sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention""#,
                true,
            )
            .replace(
                r#","quantization":{"group_size":64,"bits":4,"mode":"affine","language_model.model.layers.0.mlp.gate_proj":{"group_size":64,"bits":8}}"#,
                r#","quantization":{"group_size":64,"bits":4,"mode":"affine","language_model.model.layers.0.mlp.gate_proj":{"group_size":32,"bits":4,"mode":"mxfp4"}}"#,
            );
            let cfg: NativeGemma4Config = serde_json::from_str(&json).expect("parse");
            cfg.validate_gemma4_family().expect(
                "cross-mode group_size mismatch is allowed (each tensor dispatches with its own params)",
            );
            // Cross-mode mismatch is OK because per-tensor dispatch routes each
            // tensor to a kernel that consumes its own (group_size, bits, mode).
            let (gs, bits, mode) =
                quant_params_for(&cfg, "language_model.model.layers.0.mlp.gate_proj.weight")
                    .expect("override dispatch");
            assert_eq!(gs, 32);
            assert_eq!(bits, 4);
            assert_eq!(mode, MODE_MXFP4);
        }

        /// MXFP4 has fixed format requirements (bits=4, group_size=32). Configs
        /// that pass validation but slip past with non-MXFP4-shaped weights are
        /// caught at the `quant_params_for` dispatch site.
        #[test]
        fn mxfp4_rejects_wrong_group_size_at_dispatch() {
            let json = minimal_config_json(
                r#""sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention""#,
                true,
            )
            .replace(
                r#","quantization":{"group_size":64,"bits":4,"mode":"affine","language_model.model.layers.0.mlp.gate_proj":{"group_size":64,"bits":8}}"#,
                r#","quantization":{"group_size":64,"bits":4,"mode":"mxfp4"}"#,
            );
            let cfg: NativeGemma4Config = serde_json::from_str(&json).expect("parse");
            cfg.validate_gemma4_family()
                .expect("validate (group check is at dispatch)");
            let err = quant_params_for(&cfg, "language_model.model.layers.0.mlp.gate_proj.weight")
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("mxfp4") && err.contains("group_size=32"),
                "error should explain mxfp4 shape: {err}"
            );
        }

        #[test]
        fn rejects_wrong_top_level_model_type() {
            let json = minimal_config_json(
                r#""sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention""#,
                true,
            )
            .replace(r#""model_type": "gemma4""#, r#""model_type": "qwen3_5_moe""#);
            let cfg: NativeGemma4Config = serde_json::from_str(&json).expect("parse");
            assert!(cfg.validate_gemma4_family().is_err());
        }

        #[test]
        fn rejects_layer_type_count_mismatch() {
            // num_hidden_layers=6 but only 5 layer_types
            let json = minimal_config_json(
                r#""sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention""#,
                true,
            );
            let cfg: NativeGemma4Config = serde_json::from_str(&json).expect("parse");
            assert!(cfg.validate_gemma4_family().is_err());
        }

        #[test]
        fn rejects_zero_softcap() {
            let json = minimal_config_json(
                r#""sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention""#,
                true,
            )
            .replace(r#""final_logit_softcapping": 30.0"#, r#""final_logit_softcapping": 0"#);
            let cfg: NativeGemma4Config = serde_json::from_str(&json).expect("parse");
            assert!(cfg.validate_gemma4_family().is_err());
        }

        /// A config WITHOUT the lmstudio-style per-tensor MLP override must
        /// still validate: in-house `mlx_lm.convert` builds keep MLPs at the
        /// default bit-width, and per-key dispatch via `quant_params_for`
        /// handles whatever overrides (if any) the config actually contains.
        /// (Was `rejects_missing_mlp_override`; the sanity probe that required
        /// the override was removed — see `validate_gemma4_family`.)
        #[test]
        fn accepts_missing_mlp_override() {
            let json = minimal_config_json(
                r#""sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention""#,
                true,
            )
            .replace(
                r#","language_model.model.layers.0.mlp.gate_proj":{"group_size":64,"bits":8}"#,
                "",
            );
            let cfg: NativeGemma4Config = serde_json::from_str(&json).expect("parse");
            cfg.validate_gemma4_family()
                .expect("missing MLP override is allowed (per-key dispatch handles defaults)");
        }

        #[test]
        fn head_dim_and_kv_dispatch_per_layer_kind() {
            let json = minimal_config_json(
                r#""sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention""#,
                true,
            );
            let cfg: NativeGemma4Config = serde_json::from_str(&json).expect("parse");
            let tc = &cfg.text_config;
            assert_eq!(
                tc.head_dim_for(NativeGemma4LayerType::SlidingAttention),
                256
            );
            assert_eq!(tc.head_dim_for(NativeGemma4LayerType::FullAttention), 512);
            assert_eq!(
                tc.n_kv_heads_for(NativeGemma4LayerType::SlidingAttention),
                8
            );
            assert_eq!(tc.n_kv_heads_for(NativeGemma4LayerType::FullAttention), 2);
            assert!(tc.use_k_eq_v_for(NativeGemma4LayerType::FullAttention));
            assert!(!tc.use_k_eq_v_for(NativeGemma4LayerType::SlidingAttention));
            assert_eq!(
                tc.rope_for(NativeGemma4LayerType::FullAttention).rope_theta,
                1_000_000.0
            );
            assert_eq!(
                tc.rope_for(NativeGemma4LayerType::SlidingAttention)
                    .rope_theta,
                10_000.0
            );
        }

        // ──────────────────── Gemma4PromptCache + mask routing ────────────────────

        fn minimal_text_config_for_cache_tests(sliding_window: usize) -> NativeGemma4TextConfig {
            let json = format!(
                r#"{{
                    "model_type": "gemma4_text",
                    "hidden_size": 2816,
                    "num_hidden_layers": 6,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 8,
                    "num_global_key_value_heads": 2,
                    "head_dim": 256,
                    "global_head_dim": 512,
                    "vocab_size": 262144,
                    "rms_norm_eps": 1e-6,
                    "layer_types": [
                        "sliding_attention","sliding_attention","sliding_attention",
                        "sliding_attention","sliding_attention","full_attention"
                    ],
                    "sliding_window": {sliding_window},
                    "sliding_window_pattern": 6,
                    "max_position_embeddings": 262144,
                    "rope_parameters": {{
                        "full_attention": {{
                            "partial_rotary_factor": 0.25,
                            "rope_theta": 1000000.0,
                            "rope_type": "proportional"
                        }},
                        "sliding_attention": {{
                            "rope_theta": 10000.0,
                            "rope_type": "default"
                        }}
                    }},
                    "attention_k_eq_v": true,
                    "enable_moe_block": true,
                    "num_experts": 128,
                    "top_k_experts": 8,
                    "moe_intermediate_size": 704,
                    "intermediate_size": 2112,
                    "final_logit_softcapping": 30.0,
                    "tie_word_embeddings": true
                }}"#
            );
            serde_json::from_str(&json).expect("parse minimal text config")
        }

        #[test]
        fn prompt_cache_for_config_matches_layer_types() {
            let cfg = minimal_text_config_for_cache_tests(1024);
            let cache = NativeGemma4PromptCache::for_config(&cfg);
            assert_eq!(cache.len(), 6);
            // 0..4 sliding, 5 full
            for (i, layer) in cache.layers().iter().enumerate() {
                let expect_sliding = i < 5;
                match layer {
                    NativeGemma4LayerCache::Sliding(c) => {
                        assert!(expect_sliding, "layer {i} should not be sliding");
                        assert_eq!(c.max_size(), 1024);
                        assert_eq!(c.keep(), 0);
                        assert_eq!(c.offset(), 0);
                    }
                    NativeGemma4LayerCache::Full(c) => {
                        assert!(!expect_sliding, "layer {i} should not be full");
                        assert_eq!(c.offset(), 0);
                    }
                    // Quantized variants only appear when LUMEN_GEMMA4_QUANT_KV*
                    // env vars are set; default-env test should never see them.
                    NativeGemma4LayerCache::FullQuantized(_)
                    | NativeGemma4LayerCache::SlidingQuantized(_)
                    | NativeGemma4LayerCache::SlidingTurboquant(_)
                    | NativeGemma4LayerCache::FullTurboquant(_) => {
                        panic!("layer {i}: unexpected quantized variant under default env")
                    }
                }
            }
        }

        #[test]
        fn layer_cache_kind_dispatch_errors_on_mismatch() {
            let cfg = minimal_text_config_for_cache_tests(8);
            let mut cache = NativeGemma4PromptCache::for_config(&cfg);
            // layer 0 is sliding
            let sliding = cache.layer_mut(0).unwrap();
            assert!(sliding.as_sliding_mut().is_ok());
            assert!(cache.layer_mut(0).unwrap().as_full_mut().is_err());
            // layer 5 is full
            let full = cache.layer_mut(5).unwrap();
            assert!(full.as_full_mut().is_ok());
            assert!(cache.layer_mut(5).unwrap().as_sliding_mut().is_err());
        }

        #[test]
        fn prompt_cache_clear_resets_layers() {
            let cfg = minimal_text_config_for_cache_tests(8);
            let mut cache = NativeGemma4PromptCache::for_config(&cfg);
            cache.clear();
            for layer in cache.layers() {
                assert!(layer.empty());
                assert_eq!(layer.offset(), 0);
            }
        }

        #[test]
        fn mask_routing_decode_returns_none() {
            let cfg = minimal_text_config_for_cache_tests(8);
            for kind in [
                NativeGemma4LayerType::SlidingAttention,
                NativeGemma4LayerType::FullAttention,
            ] {
                let m = make_attention_mask_for_layer(kind, &cfg, 1, 5).expect("mask routing");
                assert!(
                    m.is_none(),
                    "decode (query_len=1) must return None mask, got Some for kind={kind:?}"
                );
            }
        }

        #[test]
        #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
        fn mask_routing_prefill_full_returns_causal() {
            let cfg = minimal_text_config_for_cache_tests(8);
            let m = make_attention_mask_for_layer(NativeGemma4LayerType::FullAttention, &cfg, 4, 0)
                .expect("mask routing")
                .expect("non-empty mask for prefill");
            assert_eq!(m.shape(), &[4, 4]);
        }

        #[test]
        #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
        fn mask_routing_prefill_sliding_applies_window() {
            let cfg = minimal_text_config_for_cache_tests(2);
            let m =
                make_attention_mask_for_layer(NativeGemma4LayerType::SlidingAttention, &cfg, 4, 0)
                    .expect("mask routing")
                    .expect("non-empty mask");
            assert_eq!(m.shape(), &[4, 4]);
            m.eval().expect("eval");
            // Inspect last row: query at position 3 with window 2 attends to keys 2..=3 only.
            let v = m.as_slice::<f32>();
            let last_row = &v[3 * 4..4 * 4];
            assert!(!last_row[0].is_finite(), "key 0 masked");
            assert!(!last_row[1].is_finite(), "key 1 masked");
            assert!(last_row[2].is_finite(), "key 2 attended");
            assert!(last_row[3].is_finite(), "key 3 attended");
        }

        #[test]
        fn weights_is_multimodal_only_classification() {
            assert!(NativeGemma4Weights::is_multimodal_only(
                "vision_tower.encoder.layers.0.input_layernorm.weight"
            ));
            assert!(NativeGemma4Weights::is_multimodal_only(
                "embed_vision.embedding_projection.weight"
            ));
            assert!(NativeGemma4Weights::is_multimodal_only(
                "model.visual.proj.weight"
            ));
            assert!(NativeGemma4Weights::is_multimodal_only(
                "audio_tower.layers.0.weight"
            ));
            assert!(NativeGemma4Weights::is_multimodal_only(
                "embed_audio.proj.weight"
            ));
            assert!(!NativeGemma4Weights::is_multimodal_only(
                "language_model.model.layers.0.self_attn.q_proj.weight"
            ));
            assert!(!NativeGemma4Weights::is_multimodal_only(
                "language_model.model.embed_tokens.weight"
            ));
        }

        #[test]
        fn loads_lmstudio_4bit_weights_when_present() {
            let dir = Path::new("/path/to/models/gemma-4-26b-a4b-mlx-4bit");
            if !dir.exists() {
                eprintln!("skip: lmstudio gemma-4-26b-a4b-mlx-4bit not present on this host");
                return;
            }
            let cfg = NativeGemma4Config::load(&dir.join("config.json")).expect("load config.json");
            cfg.validate_gemma4_family()
                .expect("validate lmstudio config");

            let mut weights = NativeGemma4Weights::load_dir(dir).expect("load_dir lmstudio shards");
            weights.sanitize().expect("sanitize lmstudio weights");

            // After sanitize, no multimodal-only keys should remain.
            for k in weights.keys() {
                assert!(
                    !NativeGemma4Weights::is_multimodal_only(k),
                    "multimodal key `{k}` survived sanitize"
                );
            }

            // Full forward-path key reachability against the actual config.
            weights
                .validate_keys_against_config(&cfg.text_config)
                .expect("validate keys against lmstudio config");

            // Spot-check that full-attention layers indeed lack v_proj.
            for layer_idx in [5usize, 11, 17, 23, 29] {
                let vp = format!("language_model.model.layers.{layer_idx}.self_attn.v_proj.weight");
                assert!(
                    weights.get(&vp).is_none(),
                    "full attention layer {layer_idx} unexpectedly has {vp}"
                );
            }
            // Sliding-attention layers must carry v_proj.
            for layer_idx in [0usize, 14, 28] {
                let vp = format!("language_model.model.layers.{layer_idx}.self_attn.v_proj.weight");
                assert!(
                    weights.get(&vp).is_some(),
                    "sliding attention layer {layer_idx} missing {vp}"
                );
            }
        }

        // ──────────────────── quant_params_for / model load ────────────────────

        #[test]
        fn quant_params_for_uses_default_when_no_override() {
            let json = minimal_config_json(
                r#""sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention""#,
                true,
            );
            let cfg: NativeGemma4Config = serde_json::from_str(&json).expect("parse");
            // attention q_proj has no override → default 4-bit.
            let (gs, bits, _mode) =
                quant_params_for(&cfg, "language_model.model.layers.0.self_attn.q_proj")
                    .expect("lookup");
            assert_eq!(gs, 64);
            assert_eq!(bits, 4);
        }

        #[test]
        fn quant_params_for_picks_up_override() {
            let json = minimal_config_json(
                r#""sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention""#,
                true,
            );
            let cfg: NativeGemma4Config = serde_json::from_str(&json).expect("parse");
            // mlp.gate_proj has 8-bit override in the synthetic fixture.
            let (gs, bits, _mode) =
                quant_params_for(&cfg, "language_model.model.layers.0.mlp.gate_proj.weight")
                    .expect("lookup");
            assert_eq!(gs, 64);
            assert_eq!(bits, 8);
        }

        /// Chunked prefill functional smoke for the windowed steel kernel —
        /// runs the same prompt through (a) single-pass forward and (b)
        /// chunked forward with chunk_size=1024 (forces sliding cache
        /// rotation on chunks 2+) and verifies both complete without
        /// crash, produce valid tokens in [0, vocab), and reach the
        /// correct cache offsets. Greedy argmax bit-identity is NOT
        /// asserted because bf16 attention accumulation order differs
        /// between (a) and (b) — both are mathematically valid sliding
        /// attention but pick different numerical paths, and top-K
        /// candidates near the argmax may flip on tiny logit
        /// perturbations. See memory note
        /// `gemma4_sliding_window_steel_kernel_landed.md`.
        #[test]
        #[ignore = "MLX FFI requires non-sandbox host with Metal device + full lmstudio shards (~16 GB)"]
        fn chunked_prefill_windowed_kernel_smoke() {
            let dir = Path::new("/path/to/models/gemma-4-26b-a4b-mlx-4bit-ours");
            if !dir.exists() {
                eprintln!("skip: model not present");
                return;
            }
            let model = NativeGemma4Model::load(dir).expect("load model");
            let vocab = model.vocab_size() as i32;
            let span = std::cmp::max(200, vocab - 20) as u32;
            let prompt: Vec<u32> = (0..4096u32).map(|i| 10 + (i * 7) % span).collect();

            // (a) Single-pass forward.
            let mut cache_single = model.make_cache();
            let logits_single = model
                .forward(&prompt, &mut cache_single)
                .expect("single-pass forward");
            let next_single = model
                .argmax_last_token(&logits_single)
                .expect("argmax single");
            assert!((next_single as i32) < vocab, "single-pass argmax in vocab");

            // (b) Chunked forward (4 × 1024-token chunks).
            let mut cache_chunked = model.make_cache();
            let chunk_size = 1024usize;
            let mut last_logits = None;
            for chunk in prompt.chunks(chunk_size) {
                let l = model
                    .forward_last_token(chunk, &mut cache_chunked)
                    .expect("chunked forward");
                last_logits = Some(l);
            }
            let logits_chunked = last_logits.expect("chunked produced no logits");
            let next_chunked = model
                .argmax_last_token(&logits_chunked)
                .expect("argmax chunked");
            assert!((next_chunked as i32) < vocab, "chunked argmax in vocab");

            // Cache offset must match in both modes regardless of internal
            // rotation. This pins the per-layer cache state invariant.
            for idx in 0..model.num_layers() {
                let s = cache_single.layer(idx).unwrap().offset();
                let c = cache_chunked.layer(idx).unwrap().offset();
                assert_eq!(s, prompt.len(), "single-pass cache offset L{idx}");
                assert_eq!(c, prompt.len(), "chunked cache offset L{idx}");
            }

            eprintln!(
                "[chunked-smoke] single-pass argmax={}  chunked argmax={}  (bf16 paths diverge OK)",
                next_single, next_chunked
            );

            // Re-measure chunked perf in isolation (warm cache, no model
            // reload, no single-pass warmup interleaved).
            let warmup_iters = 2;
            let trials = 3;
            for _ in 0..warmup_iters {
                let mut c = model.make_cache();
                for chunk in prompt.chunks(chunk_size) {
                    let l = model
                        .forward_last_token(chunk, &mut c)
                        .expect("warmup chunked");
                    l.eval().expect("warmup eval");
                }
            }
            let mut total_ms = 0.0_f64;
            for _ in 0..trials {
                let mut c = model.make_cache();
                let t0 = Instant::now();
                for chunk in prompt.chunks(chunk_size) {
                    let l = model
                        .forward_last_token(chunk, &mut c)
                        .expect("timed chunked");
                    l.eval().expect("timed eval");
                }
                total_ms += t0.elapsed().as_secs_f64() * 1e3;
            }
            let mean_ms = total_ms / trials as f64;
            let tps = prompt.len() as f64 / (mean_ms / 1e3);
            eprintln!(
                "[chunked-smoke] {} trials, chunk_size={}, prompt_len={}, mean={:.0}ms ({:.1} tok/s)",
                trials,
                chunk_size,
                prompt.len(),
                mean_ms,
                tps
            );
        }

        #[test]
        #[ignore = "MLX FFI requires non-sandbox host with Metal device + full lmstudio shards (~16 GB)"]
        fn forward_smoke_lmstudio_prefill_and_decode() {
            let dir = Path::new("/path/to/models/gemma-4-26b-a4b-mlx-4bit");
            if !dir.exists() {
                eprintln!("skip: lmstudio model not present");
                return;
            }
            let model = NativeGemma4Model::load(dir).expect("load model");
            let vocab = model.vocab_size() as i32;
            let mut cache = model.make_cache();

            // Prefill 4 deterministic token ids (avoid BOS=2 / EOS markers).
            let prefill_ids = vec![100u32, 200, 300, 400];
            let logits = model
                .forward(&prefill_ids, &mut cache)
                .expect("forward prefill");
            assert_eq!(logits.shape(), &[1, prefill_ids.len() as i32, vocab]);
            // Cache offsets: every layer should reflect the prefill length.
            for idx in 0..model.num_layers() {
                assert_eq!(
                    cache.layer(idx).unwrap().offset(),
                    prefill_ids.len(),
                    "cache offset for layer {idx} after prefill"
                );
            }

            // Greedy step from prefill output.
            let next = model.argmax_last_token(&logits).expect("argmax last");
            assert!((next as i32) < vocab, "argmax in vocab");

            // Decode step: feed the predicted token.
            let logits2 = model.forward(&[next], &mut cache).expect("forward decode");
            assert_eq!(logits2.shape(), &[1, 1, vocab]);
            for idx in 0..model.num_layers() {
                assert_eq!(
                    cache.layer(idx).unwrap().offset(),
                    prefill_ids.len() + 1,
                    "cache offset for layer {idx} after decode step"
                );
            }
        }

        /// Multi-step greedy generation smoke test + decode tok/s benchmark.
        ///
        /// Mirrors the `forward_smoke_lmstudio_prefill_and_decode` setup but
        /// drives the full `generate()` helper for 16 new tokens. Prints
        /// timing so it can be diffed against `mlx_lm.server`'s warm
        /// baseline (~37 tok/s on M4 Pro, ~55 tok/s on M3 Max).
        #[test]
        #[ignore = "MLX FFI requires non-sandbox host with Metal device + full lmstudio shards (~16 GB)"]
        fn generate_smoke_and_benchmark_lmstudio() {
            let dir = Path::new("/path/to/models/gemma-4-26b-a4b-mlx-4bit");
            if !dir.exists() {
                eprintln!("skip: lmstudio model not present");
                return;
            }
            let load_start = std::time::Instant::now();
            let model = NativeGemma4Model::load(dir).expect("load model");
            let load_ms = load_start.elapsed().as_secs_f64() * 1e3;
            let vocab = model.vocab_size() as i32;

            // Deterministic prompt that avoids BOS / EOS / turn markers.
            let prompt = vec![100u32, 200, 300, 400];

            // Warmup pass: drives all Metal kernel JIT for the prefill +
            // decode shapes we're about to measure. Numbers reported below
            // are the *warm* run (steady-state).
            let warmup_cfg = GenerateConfig {
                max_new_tokens: 4,
                stop_on_eos: false,
                sampling: None,
            };
            let _ = model
                .generate(&prompt, &warmup_cfg)
                .expect("warmup generate ok");

            let gen_cfg = GenerateConfig {
                max_new_tokens: 32,
                // Disable EOS early-exit so the benchmark gets a full
                // 31-step decode loop regardless of which token the model
                // happens to argmax first.
                stop_on_eos: false,
                sampling: None,
            };

            let stats = model.generate(&prompt, &gen_cfg).expect("generate ok");

            // Shape / correctness asserts.
            assert_eq!(stats.prompt_tokens, prompt.len());
            assert_eq!(stats.generated_tokens.len(), gen_cfg.max_new_tokens);
            assert_eq!(stats.decode_steps, gen_cfg.max_new_tokens - 1);
            assert!(!stats.stopped_on_eos, "stop_on_eos=false respected");
            for t in &stats.generated_tokens {
                assert!((*t as i32) < vocab, "generated token in vocab");
            }
            // Decode must have made forward progress within a sane budget.
            assert!(
                stats.decode_tok_per_sec > 0.0,
                "decode tok/s must be positive (got {:.3})",
                stats.decode_tok_per_sec
            );

            eprintln!(
                "[gen-bench] load={:.0}ms  prompt={}tok  new={}tok  prefill={:.1}ms  decode={:.1}ms  decode_tok/s={:.1}",
                load_ms,
                stats.prompt_tokens,
                stats.generated_tokens.len(),
                stats.prefill_ms,
                stats.decode_ms,
                stats.decode_tok_per_sec,
            );
            eprintln!("[gen-bench] tokens = {:?}", stats.generated_tokens);
        }

        /// Per-step decode component breakdown.
        ///
        /// Requires `LUMEN_GEMMA4_BREAKDOWN=1` env to enable the eval-barrier
        /// instrumentation. Without it the bucket counters stay zero and the
        /// test prints the trivial "no instrumentation" message.
        ///
        /// Reports per-step ms for: attention, dense MLP, router, experts.
        /// (All other ops — norms, residuals, lm_head, argmax, embed — are
        /// lumped into "other" derived from total decode wall - sum of buckets.)
        ///
        /// Run:
        ///   LUMEN_GEMMA4_BREAKDOWN=1 cargo test -p lumen-mlx \
        ///     --features mlx-native --release --lib \
        ///     gemma4_decode_component_breakdown_lmstudio -- --ignored --nocapture
        #[test]
        #[ignore = "MLX FFI requires non-sandbox host + lmstudio shards (~16 GB); set LUMEN_GEMMA4_BREAKDOWN=1"]
        fn gemma4_decode_component_breakdown_lmstudio() {
            let dir = Path::new("/path/to/models/gemma-4-26b-a4b-mlx-4bit");
            if !dir.exists() {
                eprintln!("skip: lmstudio model not present");
                return;
            }
            if std::env::var("LUMEN_GEMMA4_BREAKDOWN").unwrap_or_default() != "1" {
                eprintln!(
                    "skip: LUMEN_GEMMA4_BREAKDOWN env not set; re-run with \
                     LUMEN_GEMMA4_BREAKDOWN=1 to enable the eval-barrier path"
                );
                return;
            }
            let model = NativeGemma4Model::load(dir).expect("load model");

            // Deterministic synthetic prompt — same as the gen-bench smoke
            // test so the prefill shape compile cache hits warm.
            let prompt = vec![100u32, 200, 300, 400];

            // Warmup so per-shape compile cost is amortized out of the
            // measurement (it would otherwise dwarf attention/MoE wall-time).
            let _ = model
                .generate(
                    &prompt,
                    &GenerateConfig {
                        max_new_tokens: 4,
                        stop_on_eos: false,
                        sampling: None,
                    },
                )
                .expect("warmup");
            // Drain warmup counters before the measured run.
            let _ = take_gemma4_breakdown();

            // Measured run: N=32 decode steps. Larger N tightens the per-step
            // average; 32 is enough to surface the dominant bucket without
            // dragging the test past a few seconds.
            let cfg = GenerateConfig {
                max_new_tokens: 32,
                stop_on_eos: false,
                sampling: None,
            };
            let stats = model.generate(&prompt, &cfg).expect("generate");
            let decode_steps = stats.decode_steps as f64;
            assert!(decode_steps > 0.0, "must have decode steps");

            let b = take_gemma4_breakdown();
            let attn_per = b.attn_ms / decode_steps;
            let dense_per = b.dense_ms / decode_steps;
            let router_per = b.router_ms / decode_steps;
            let experts_per = b.experts_ms / decode_steps;
            let summed = attn_per + dense_per + router_per + experts_per;
            let total_step = stats.decode_ms / decode_steps;
            let other_per = (total_step - summed).max(0.0);

            eprintln!(
                "[breakdown] decode_steps={} total_decode={:.0}ms total/step={:.2}ms tok/s={:.1}",
                stats.decode_steps, stats.decode_ms, total_step, stats.decode_tok_per_sec
            );
            eprintln!(
                "[breakdown]   attn   = {:6.2} ms/step ({:5.1}%)",
                attn_per,
                100.0 * attn_per / total_step
            );
            eprintln!(
                "[breakdown]   dense  = {:6.2} ms/step ({:5.1}%)",
                dense_per,
                100.0 * dense_per / total_step
            );
            eprintln!(
                "[breakdown]   router = {:6.2} ms/step ({:5.1}%)",
                router_per,
                100.0 * router_per / total_step
            );
            eprintln!(
                "[breakdown]   exprts = {:6.2} ms/step ({:5.1}%)",
                experts_per,
                100.0 * experts_per / total_step
            );
            eprintln!(
                "[breakdown]   other  = {:6.2} ms/step ({:5.1}%)  (norms/residuals/embed/lm_head/argmax)",
                other_per,
                100.0 * other_per / total_step
            );
        }

        /// E2E pipeline smoke: chat template → generate() → decode().
        ///
        /// Drives the full request path end-to-end: a 2-message conversation
        /// (system + user) is rendered into token ids by the Gemma 4 chat
        /// template, fed through `generate()`, then the resulting tokens
        /// are decoded back to a string. Asserts that:
        ///   • The render produces a sane prompt with BOS / `<|turn>` etc.
        ///   • Generation runs without panicking on real semantic content.
        ///   • Decoding emits a non-empty UTF-8 string (special tokens
        ///     stripped, so the visible reply only).
        ///
        /// This is the first end-to-end smoke test that proves the W4 (a)
        /// tokenizer + chat template are correctly wired to W3 (d)'s
        /// `generate()` helper.
        #[test]
        #[ignore = "MLX FFI requires non-sandbox host with Metal device + full lmstudio shards (~16 GB)"]
        fn chat_template_generate_e2e_lmstudio() {
            use crate::gemma4_chat::imp::{
                ChatMessage, ChatRole, Gemma4ChatTemplate, RenderOptions,
            };

            let dir = Path::new("/path/to/models/gemma-4-26b-a4b-mlx-4bit");
            if !dir.exists() {
                eprintln!("skip: lmstudio model not present");
                return;
            }
            let model = NativeGemma4Model::load(dir).expect("load model");
            let tpl = Gemma4ChatTemplate::from_dir(dir).expect("load chat template");

            let msgs = [
                ChatMessage {
                    role: ChatRole::System,
                    content: "Be concise.",
                },
                ChatMessage {
                    role: ChatRole::User,
                    content: "Say hi in one word.",
                },
            ];
            let prompt_ids = tpl
                .render_to_ids(&msgs, &RenderOptions::default())
                .expect("render chat template");
            assert!(prompt_ids.len() > 5, "prompt should be non-trivial");
            assert_eq!(prompt_ids[0], 2, "prompt starts with <bos>");

            let gen_cfg = GenerateConfig {
                max_new_tokens: 8,
                stop_on_eos: true,
                sampling: None,
            };
            let stats = model
                .generate(&prompt_ids, &gen_cfg)
                .expect("generate from chat template");
            assert_eq!(stats.prompt_tokens, prompt_ids.len());
            assert!(!stats.generated_tokens.is_empty());

            let decoded = tpl
                .decode(&stats.generated_tokens, /* skip_special */ true)
                .expect("decode generated tokens");
            // Visible text should be a valid (possibly empty after stripping
            // specials) string; just assert it's a string we can read.
            eprintln!(
                "[chat-e2e] tokens={:?}  decoded={:?}  stop_eos={}",
                stats.generated_tokens, decoded, stats.stopped_on_eos
            );
        }

        /// EOS termination explicit coverage.
        ///
        /// Two-pass test on a single model instance:
        ///   1. **POSITIVE path** — chat-templated "Hi" prompt with
        ///      `stop_on_eos=true` + max=64 budget. Asserts the model
        ///      terminates *before* the budget cap and the last generated
        ///      token is in `eos_tokens() = [1, 106, 50]`. This covers
        ///      the chat-aware EOS set (turn close = 106, raw eos = 1,
        ///      tool response = 50) regardless of which one fires.
        ///   2. **NEGATIVE path** — same prompt with `stop_on_eos=false` +
        ///      max=24. Asserts the model runs the full 24 tokens (no
        ///      early exit) and that an EOS token *does* appear in the
        ///      output (since the natural reply is short — if it didn't,
        ///      the prompt isn't exercising the EOS path).
        ///
        /// Why this matters: the EOS set was wrong in W3 (d) before the
        /// W4 (a) fix (`eos_tokens()` preferred `text_config.eos_token_id=1`
        /// over top-level `[1, 106, 50]`, causing chat turns to overshoot
        /// past `<turn|>=106`). This test guards against that class of
        /// regression.
        #[test]
        #[ignore = "MLX FFI requires non-sandbox host with Metal device + full lmstudio shards (~16 GB)"]
        fn generate_chat_eos_termination_paths_lmstudio() {
            use crate::gemma4_chat::imp::{
                ChatMessage, ChatRole, Gemma4ChatTemplate, RenderOptions,
            };

            let dir = Path::new("/path/to/models/gemma-4-26b-a4b-mlx-4bit");
            if !dir.exists() {
                eprintln!("skip: lmstudio model not present");
                return;
            }
            let model = NativeGemma4Model::load(dir).expect("load model");
            let tpl = Gemma4ChatTemplate::from_dir(dir).expect("load tpl");
            let eos: Vec<u32> = model.eos_tokens().to_vec();
            assert_eq!(eos, vec![1u32, 106, 50], "EOS set must be chat-aware");

            let msgs = [ChatMessage {
                role: ChatRole::User,
                content: "Hi",
            }];
            let prompt = tpl
                .render_to_ids(&msgs, &RenderOptions::default())
                .expect("render");

            // Warmup so JIT is amortized (kernel compile noise out of the
            // EOS-stop signal we're measuring).
            let _ = model
                .generate(
                    &prompt,
                    &GenerateConfig {
                        max_new_tokens: 4,
                        stop_on_eos: false,
                        sampling: None,
                    },
                )
                .expect("warmup");

            // ── POSITIVE: stop_on_eos=true must terminate early ──────────
            let pos = model
                .generate(
                    &prompt,
                    &GenerateConfig {
                        max_new_tokens: 64,
                        stop_on_eos: true,
                        sampling: None,
                    },
                )
                .expect("positive generate");
            assert!(
                pos.stopped_on_eos,
                "expected stopped_on_eos=true (got false; tokens={:?})",
                pos.generated_tokens
            );
            assert!(
                pos.generated_tokens.len() < 64,
                "must stop before max=64, got {} tokens",
                pos.generated_tokens.len()
            );
            let last = *pos.generated_tokens.last().expect("non-empty");
            assert!(
                eos.contains(&last),
                "last token {last} not in EOS set {eos:?}; full={:?}",
                pos.generated_tokens
            );
            eprintln!(
                "[w6-u positive] generated {} tokens, last={} ∈ {:?}",
                pos.generated_tokens.len(),
                last,
                eos
            );

            // ── NEGATIVE: stop_on_eos=false must run to budget ───────────
            let neg = model
                .generate(
                    &prompt,
                    &GenerateConfig {
                        max_new_tokens: 24,
                        stop_on_eos: false,
                        sampling: None,
                    },
                )
                .expect("negative generate");
            assert!(
                !neg.stopped_on_eos,
                "stop_on_eos=false respected: stopped_on_eos must be false"
            );
            assert_eq!(
                neg.generated_tokens.len(),
                24,
                "must produce exactly 24 tokens when EOS check off"
            );
            // Sanity: the natural reply is short, so EOS should appear in
            // the forced 24-token continuation. If it didn't, the prompt
            // isn't exercising the EOS path and the positive arm is
            // testing nothing useful.
            let has_eos = neg.generated_tokens.iter().any(|t| eos.contains(t));
            assert!(
                has_eos,
                "forced 24-token continuation must contain at least one EOS; got {:?}",
                neg.generated_tokens
            );
            eprintln!(
                "[w6-u negative] generated {} tokens (forced), eos_present={}, tokens={:?}",
                neg.generated_tokens.len(),
                has_eos,
                neg.generated_tokens
            );
        }

        /// Long-context correctness past `sliding_window=1024`.
        ///
        /// Drives a 1280-token prefill (256 tokens past the rotating cache
        /// max_size) through the model and asserts:
        ///   1. No crash / panic.
        ///   2. Sliding-attention layers' `cached_len()` equals the full
        ///      prompt length after prefill — `NativeRotatingKvCache`
        ///      mirrors mlx-lm semantics: prefill never trims (S=prompt_len
        ///      passes through wholesale), trimming only kicks in on
        ///      incremental decode (S=1) updates.
        ///   3. Full-attention layers' `cached_len()` grows linearly with
        ///      the input (=1280) — no wrap.
        ///   4. The post-prefill argmax produces a token in vocab range.
        ///   5. A short 4-step decode after the long prefill triggers the
        ///      sliding rotation: cached_len drops to `sliding_window=1024`
        ///      on the *first* decode step (trim = 1280+1 - 1024 = 257
        ///      tokens evicted from the middle) and stays at 1024
        ///      thereafter. Full layers grow to 1284.
        ///
        /// Synthetic token IDs (not chat-templated) are used here because
        /// the goal is structural correctness of the cache + attention
        /// shape pipeline, not output quality. Output text from random
        /// token sequences is meaningless — we only check no NaN / no
        /// out-of-vocab id.
        #[test]
        #[ignore = "MLX FFI requires non-sandbox host with Metal device + full lmstudio shards (~16 GB)"]
        fn long_context_sliding_window_correctness_lmstudio() {
            let dir = Path::new("/path/to/models/gemma-4-26b-a4b-mlx-4bit");
            if !dir.exists() {
                eprintln!("skip: lmstudio model not present");
                return;
            }
            let model = NativeGemma4Model::load(dir).expect("load model");
            let cfg = model.text_config();
            let sliding_window = cfg.sliding_window; // 1024 for Gemma 4
            assert_eq!(sliding_window, 1024, "expected sliding_window=1024");
            let vocab = cfg.vocab_size as u32;

            // 1280-token synthetic prompt = sliding_window + 256.
            //
            // Token id formula picks a deterministic value above the
            // special-token range (avoid 0..16 = pad/bos/eos/control) and
            // below vocab. We don't care that this isn't valid Korean —
            // we're testing cache rotation, not semantics.
            let prompt_len = sliding_window + 256;
            let prompt: Vec<u32> = (0..prompt_len)
                .map(|i| 100u32 + ((i as u32).wrapping_mul(37)) % (vocab - 200))
                .collect();

            // Warmup so per-shape JIT compile is amortized.
            let _ = model
                .generate(
                    &[100u32, 200, 300, 400],
                    &GenerateConfig {
                        max_new_tokens: 2,
                        stop_on_eos: false,
                        sampling: None,
                    },
                )
                .expect("warmup");

            // Long-context prefill + short decode.
            let stats = model
                .generate(
                    &prompt,
                    &GenerateConfig {
                        max_new_tokens: 4,
                        stop_on_eos: false,
                        sampling: None,
                    },
                )
                .expect("long-context generate");

            assert_eq!(stats.prompt_tokens, prompt_len, "prompt length stored");
            assert_eq!(stats.generated_tokens.len(), 4, "all 4 decode steps ran");
            for t in &stats.generated_tokens {
                assert!(
                    (*t as u32) < vocab,
                    "generated token {t} out of vocab {vocab}"
                );
            }

            // Cache invariants: build a fresh cache and replay the prefill
            // so we can inspect per-layer cached_len. (We can't re-use the
            // generate's cache — it's freed inside the call.)
            let mut cache = model.make_cache();
            let logits = model.forward(&prompt, &mut cache).expect("prefill forward");
            let _ = model.argmax_last_token(&logits).expect("argmax");

            // Layer 0 = sliding → after prefill, cached_len == prompt_len
            // (NativeRotatingKvCache only trims on decode-step updates;
            // prefill passes through wholesale, matching mlx-lm semantics).
            let l0 = cache.layer(0).expect("layer 0");
            assert!(
                matches!(l0, NativeGemma4LayerCache::Sliding(_)),
                "layer 0 expected to be sliding"
            );
            assert_eq!(
                l0.cached_len(),
                prompt_len,
                "after prefill sliding cached_len = prompt_len (no trim on prefill); got {}",
                l0.cached_len()
            );

            // Layer 5 = full → cached_len equals prompt_len.
            let l5 = cache.layer(5).expect("layer 5");
            assert!(
                matches!(l5, NativeGemma4LayerCache::Full(_)),
                "layer 5 expected to be full attention"
            );
            assert_eq!(
                l5.cached_len(),
                prompt_len,
                "full cache must grow to prompt_len={prompt_len}, got {}",
                l5.cached_len()
            );

            // Drive 4 decode steps and re-check.
            let mut next = model.argmax_last_token(&logits).expect("argmax2");
            for _ in 0..4 {
                let lg = model.forward(&[next], &mut cache).expect("decode forward");
                next = model.argmax_last_token(&lg).expect("decode argmax");
            }
            let l0_after = cache.layer(0).expect("l0 after");
            let l5_after = cache.layer(5).expect("l5 after");
            assert_eq!(
                l0_after.cached_len(),
                sliding_window,
                "sliding stays pinned at sliding_window after decode"
            );
            assert_eq!(
                l5_after.cached_len(),
                prompt_len + 4,
                "full grows to prompt_len + 4 = {} after 4 decode steps",
                prompt_len + 4
            );

            eprintln!(
                "[w6-t long-ctx] prompt={} new={} prefill={:.0}ms decode={:.0}ms decode_tok/s={:.2}",
                stats.prompt_tokens,
                stats.generated_tokens.len(),
                stats.prefill_ms,
                stats.decode_ms,
                stats.decode_tok_per_sec
            );
            eprintln!(
                "[w6-t long-ctx] sliding(l0).cached_len={} full(l5).cached_len={} (after 4 decode)",
                l0_after.cached_len(),
                l5_after.cached_len()
            );
        }

        #[test]
        #[ignore = "MLX FFI requires non-sandbox host with Metal device + full lmstudio shards (~16 GB)"]
        fn attention_forward_smoke_lmstudio_sliding_and_full() {
            let dir = Path::new("/path/to/models/gemma-4-26b-a4b-mlx-4bit");
            if !dir.exists() {
                eprintln!("skip: lmstudio model not present");
                return;
            }
            let model = NativeGemma4Model::load(dir).expect("load model");
            let hidden = model.text_config().hidden_size as i32;
            let l = 4i32;
            let b = 1i32;
            let n = (b * l * hidden) as usize;
            // Synthetic small-magnitude input — keep values bounded so RMSNorm
            // and softmax stay numerically well-behaved.
            let data: Vec<f32> = (0..n).map(|i| 0.001 * ((i % 17) as f32 - 8.0)).collect();
            let x = Array::from_slice(&data, &[b, l, hidden]);
            let x = x
                .as_dtype(mlx_rs::Dtype::Float32)
                .expect("cast input to f32");

            let mut cache = model.make_cache();

            // Layer 0 — sliding attention.
            let layer0 = cache.layer_mut(0).expect("layer 0 cache");
            let out0 = model
                .layer_attention_forward(&x, 0, layer0)
                .expect("sliding attention forward");
            assert_eq!(out0.shape(), &[b, l, hidden]);
            assert_eq!(cache.layer(0).unwrap().offset(), l as usize);

            // Layer 5 — full attention (k_eq_v branch).
            let layer5 = cache.layer_mut(5).expect("layer 5 cache");
            let out5 = model
                .layer_attention_forward(&x, 5, layer5)
                .expect("full attention forward");
            assert_eq!(out5.shape(), &[b, l, hidden]);
            assert_eq!(cache.layer(5).unwrap().offset(), l as usize);

            // Decode-step sanity: feed a 1-token follow-up to layer 0 and
            // make sure RotatingKvCache grows correctly.
            let n_decode = (b * 1 * hidden) as usize;
            let data1: Vec<f32> = (0..n_decode)
                .map(|i| 0.001 * ((i % 13) as f32 - 6.0))
                .collect();
            let x1 = Array::from_slice(&data1, &[b, 1i32, hidden])
                .as_dtype(mlx_rs::Dtype::Float32)
                .expect("cast decode input");
            let layer0 = cache.layer_mut(0).expect("layer 0 cache");
            let out_decode = model
                .layer_attention_forward(&x1, 0, layer0)
                .expect("decode attention forward");
            assert_eq!(out_decode.shape(), &[b, 1i32, hidden]);
            assert_eq!(cache.layer(0).unwrap().offset(), (l + 1) as usize);
        }

        #[test]
        #[ignore = "MLX FFI requires non-sandbox host with Metal device + full lmstudio shards (~16 GB)"]
        fn loads_lmstudio_4bit_full_model_when_present() {
            let dir = Path::new("/path/to/models/gemma-4-26b-a4b-mlx-4bit");
            if !dir.exists() {
                eprintln!("skip: lmstudio gemma-4-26b-a4b-mlx-4bit not present");
                return;
            }
            let model = NativeGemma4Model::load(dir).expect("load full Gemma 4 model");
            assert_eq!(model.num_layers(), 30);
            assert_eq!(model.vocab_size(), 262144);
            // lmstudio config has top-level eos_token_id=[1, 106, 50]
            // and text_config.eos_token_id=1. We prefer the longer top-level
            // (chat-aware) list, mirroring HF generation_config.json.
            assert_eq!(model.eos_tokens(), &[1u32, 106, 50]);

            // Spot-check per-layer resolution.
            for (idx, layer) in model.layers().iter().enumerate() {
                // q_norm head_dim matches layer kind.
                let expected_head_dim = model.text_config().head_dim_for(layer.kind) as i32;
                assert_eq!(
                    layer.attn.q_norm.shape(),
                    &[expected_head_dim],
                    "layer {idx} ({:?}): q_norm shape",
                    layer.kind
                );
                // Quant params:
                // attention q_proj is 4-bit default; mlp.gate_proj is 8-bit override.
                assert_eq!(layer.attn.q_proj.bits, 4);
                assert_eq!(layer.dense_mlp.gate_proj.bits, 8);
                assert_eq!(layer.router.proj.bits, 8);
                assert_eq!(layer.experts.gate_proj.bits, 4);
                // v_proj presence per layer kind.
                match layer.kind {
                    NativeGemma4LayerType::FullAttention => {
                        assert!(
                            layer.attn.v_proj.is_none(),
                            "full attention layer {idx} must not have v_proj"
                        );
                    }
                    NativeGemma4LayerType::SlidingAttention => {
                        assert!(
                            layer.attn.v_proj.is_some(),
                            "sliding attention layer {idx} must have v_proj"
                        );
                    }
                }
            }

            // make_cache shape sanity.
            let cache = model.make_cache();
            assert_eq!(cache.len(), 30);
            for (idx, layer_cache) in cache.layers().iter().enumerate() {
                let expected_sliding = model.text_config().layer_types[idx].is_sliding();
                let is_sliding = matches!(layer_cache, NativeGemma4LayerCache::Sliding(_));
                assert_eq!(is_sliding, expected_sliding, "layer {idx} cache kind");
            }
        }

        #[test]
        fn parses_lmstudio_4bit_config_when_present() {
            let path = Path::new(LMSTUDIO_CONFIG_PATH);
            if !path.exists() {
                eprintln!("skip: {LMSTUDIO_CONFIG_PATH} not present on this host");
                return;
            }
            let cfg = NativeGemma4Config::load(path).expect("load lmstudio config");
            cfg.validate_gemma4_family()
                .expect("validate lmstudio 26B-A4B config");
            let tc = &cfg.text_config;
            assert_eq!(tc.hidden_size, 2816);
            assert_eq!(tc.num_hidden_layers, 30);
            assert_eq!(tc.num_attention_heads, 16);
            assert_eq!(tc.num_key_value_heads, 8);
            assert_eq!(tc.num_global_key_value_heads, Some(2));
            assert_eq!(tc.head_dim, 256);
            assert_eq!(tc.global_head_dim, 512);
            assert_eq!(tc.vocab_size, 262144);
            assert_eq!(tc.intermediate_size, 2112);
            assert_eq!(tc.moe_intermediate_size, 704);
            assert_eq!(tc.num_experts, 128);
            assert_eq!(tc.top_k_experts, 8);
            assert_eq!(tc.sliding_window, 1024);
            assert!(tc.attention_k_eq_v);
            assert!(tc.enable_moe_block);
            assert!(tc.tie_word_embeddings);
            assert!((tc.final_logit_softcapping - 30.0).abs() < 1e-6);
            // Full attention at layers 5/11/17/23/29.
            for idx in [5usize, 11, 17, 23, 29] {
                assert!(
                    tc.layer_types[idx].is_full(),
                    "expected layer {idx} to be full attention"
                );
            }
            // All other layers sliding.
            let n_full = tc.layer_types.iter().filter(|t| t.is_full()).count();
            assert_eq!(n_full, 5);
            let quant = cfg.effective_quantization().expect("quant present");
            assert_eq!(quant.bits, 4);
            assert_eq!(quant.group_size, 64);
            assert_eq!(quant.mode, "affine");
            // Spot-check a handful of 8-bit override entries spanning the layer stack.
            for layer_idx in [0usize, 14, 29] {
                for tensor in [
                    "mlp.gate_proj",
                    "mlp.up_proj",
                    "mlp.down_proj",
                    "router.proj",
                ] {
                    let key = format!("language_model.model.layers.{layer_idx}.{tensor}");
                    let ov = quant
                        .overrides
                        .get(&key)
                        .unwrap_or_else(|| panic!("missing override for {key}"));
                    assert_eq!(ov.bits, 8, "override at {key} should be 8-bit");
                    assert_eq!(ov.group_size, 64);
                }
            }
        }
    }
}
