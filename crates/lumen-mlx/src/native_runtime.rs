//! Shared MLX-native decode-runtime infrastructure used by all model
//! implementations (Qwen3.5-MoE today; Llama / Mistral / etc. parallel files
//! later).
//!
//! Owns instrumentation primitives that are not specific to any single
//! architecture:
//! - [`FineTimings`] — per-layer-kind wall-clock breakdown. The current shape
//!   matches Qwen3.5-MoE (full-attn / linear-attn / MoE), but new model files
//!   should reuse this struct and only populate the fields that apply.
//!   Sub-buckets (`linear_*`, `moe_*`) can be extended via the bump helpers.
//! - [`fine_timing_active`] — env gate `LUMEN_NATIVE_TIMING=1`.
//! - [`take_fine_timings`] — thread-local drain consumed by the runner.
//! - `bump_*_ms` helpers — per-sub-bucket accumulators called from layer-kind
//!   modules (`native_moe`, `native_ssm`, etc.). Each model file is
//!   responsible for assembling sub-buckets into its forward path.

/// Per-`forward()` layer-kind timing breakdown. Populated only when
/// `LUMEN_NATIVE_TIMING=1` is set and the `mlx-native` feature is built;
/// otherwise [`take_fine_timings`] returns `None`.
///
/// Each `*_ms` field is the wall-clock time spent in that section **after**
/// an explicit `eval()` barrier. Inserting evals serializes the MLX dispatch
/// graph, so the totals here over-count vs. the un-instrumented path —
/// consumers should look at the *ratio* of buckets, not absolute totals.
///
/// Sub-buckets (`linear_*`, `moe_*`) are aggregated across all layers of the
/// matching kind; together they sum to (≈) the parent bucket modulo per-layer
/// outer-barrier overhead.
#[derive(Default, Clone, Copy, Debug)]
pub struct FineTimings {
    pub embed_ms: f64,
    pub full_attn_ms: f64,
    pub linear_attn_ms: f64,
    pub moe_ms: f64,
    pub lm_head_ms: f64,
    pub full_attn_count: usize,
    pub linear_attn_count: usize,
    pub moe_count: usize,

    // linear_attn sub-buckets (summed across all linear-attn layers).
    pub linear_in_proj_ms: f64,
    pub linear_conv_ms: f64,
    pub linear_norm_qk_ms: f64,
    pub linear_ssm_ms: f64,
    pub linear_out_proj_ms: f64,

    // MoE sub-buckets (summed across all MoE layers).
    pub moe_routing_ms: f64,
    pub moe_switch_glu_ms: f64,
    pub moe_shared_expert_ms: f64,
}

#[cfg(feature = "mlx-native")]
mod imp {
    use std::cell::RefCell;
    use std::sync::OnceLock;

    use super::FineTimings;

    static FINE_TIMING_ENABLED: OnceLock<bool> = OnceLock::new();

    fn fine_timing_enabled() -> bool {
        *FINE_TIMING_ENABLED.get_or_init(|| {
            std::env::var("LUMEN_NATIVE_TIMING")
                .map(|s| matches!(s.as_str(), "1" | "true" | "TRUE" | "yes"))
                .unwrap_or(false)
        })
    }

    thread_local! {
        static LAST_FINE_TIMINGS: RefCell<Option<FineTimings>> =
            const { RefCell::new(None) };
        /// Per-layer sub-bucket scratch slot. Model forward implementations
        /// reset before each linear-attn / MoE layer call, then drain +
        /// aggregate after.
        static LAYER_SUB_TIMINGS: RefCell<LayerSubTimings> =
            const { RefCell::new(LayerSubTimings::ZERO) };
    }

    #[derive(Default, Clone, Copy)]
    pub(crate) struct LayerSubTimings {
        pub linear_in_proj_ms: f64,
        pub linear_conv_ms: f64,
        pub linear_norm_qk_ms: f64,
        pub linear_ssm_ms: f64,
        pub linear_out_proj_ms: f64,
        pub moe_routing_ms: f64,
        pub moe_switch_glu_ms: f64,
        pub moe_shared_expert_ms: f64,
    }

    impl LayerSubTimings {
        pub(crate) const ZERO: Self = Self {
            linear_in_proj_ms: 0.0,
            linear_conv_ms: 0.0,
            linear_norm_qk_ms: 0.0,
            linear_ssm_ms: 0.0,
            linear_out_proj_ms: 0.0,
            moe_routing_ms: 0.0,
            moe_switch_glu_ms: 0.0,
            moe_shared_expert_ms: 0.0,
        };
    }

    pub(crate) fn take_fine_timings() -> Option<FineTimings> {
        LAST_FINE_TIMINGS.with(|cell| cell.borrow_mut().take())
    }

    pub(crate) fn store_fine_timings(t: FineTimings) {
        LAST_FINE_TIMINGS.with(|cell| *cell.borrow_mut() = Some(t));
    }

    pub(crate) fn reset_layer_sub_timings() {
        LAYER_SUB_TIMINGS.with(|c| *c.borrow_mut() = LayerSubTimings::ZERO);
    }

    pub(crate) fn drain_layer_sub_timings() -> LayerSubTimings {
        LAYER_SUB_TIMINGS.with(|c| {
            let v = *c.borrow();
            *c.borrow_mut() = LayerSubTimings::ZERO;
            v
        })
    }

    pub(crate) fn fine_timing_active() -> bool {
        fine_timing_enabled()
    }

    pub(crate) fn bump_linear_in_proj_ms(ms: f64) {
        LAYER_SUB_TIMINGS.with(|c| c.borrow_mut().linear_in_proj_ms += ms);
    }
    pub(crate) fn bump_linear_conv_ms(ms: f64) {
        LAYER_SUB_TIMINGS.with(|c| c.borrow_mut().linear_conv_ms += ms);
    }
    pub(crate) fn bump_linear_norm_qk_ms(ms: f64) {
        LAYER_SUB_TIMINGS.with(|c| c.borrow_mut().linear_norm_qk_ms += ms);
    }
    pub(crate) fn bump_linear_ssm_ms(ms: f64) {
        LAYER_SUB_TIMINGS.with(|c| c.borrow_mut().linear_ssm_ms += ms);
    }
    pub(crate) fn bump_linear_out_proj_ms(ms: f64) {
        LAYER_SUB_TIMINGS.with(|c| c.borrow_mut().linear_out_proj_ms += ms);
    }
    pub(crate) fn bump_moe_routing_ms(ms: f64) {
        LAYER_SUB_TIMINGS.with(|c| c.borrow_mut().moe_routing_ms += ms);
    }
    pub(crate) fn bump_moe_switch_glu_ms(ms: f64) {
        LAYER_SUB_TIMINGS.with(|c| c.borrow_mut().moe_switch_glu_ms += ms);
    }
    pub(crate) fn bump_moe_shared_expert_ms(ms: f64) {
        LAYER_SUB_TIMINGS.with(|c| c.borrow_mut().moe_shared_expert_ms += ms);
    }
}

#[cfg(feature = "mlx-native")]
#[allow(unused_imports)]
pub(crate) use imp::{
    LayerSubTimings, bump_linear_conv_ms, bump_linear_in_proj_ms, bump_linear_norm_qk_ms,
    bump_linear_out_proj_ms, bump_linear_ssm_ms, bump_moe_routing_ms, bump_moe_shared_expert_ms,
    bump_moe_switch_glu_ms, drain_layer_sub_timings, fine_timing_active, reset_layer_sub_timings,
    store_fine_timings, take_fine_timings,
};

// Stub layer for non-mlx-native builds — callers must still compile.
#[cfg(not(feature = "mlx-native"))]
pub(crate) fn take_fine_timings() -> Option<FineTimings> {
    None
}

#[cfg(not(feature = "mlx-native"))]
#[allow(dead_code)]
pub(crate) fn fine_timing_active() -> bool {
    false
}

#[cfg(not(feature = "mlx-native"))]
#[allow(dead_code)]
pub(crate) fn bump_moe_routing_ms(_ms: f64) {}

#[cfg(not(feature = "mlx-native"))]
#[allow(dead_code)]
pub(crate) fn bump_moe_switch_glu_ms(_ms: f64) {}

#[cfg(not(feature = "mlx-native"))]
#[allow(dead_code)]
pub(crate) fn bump_moe_shared_expert_ms(_ms: f64) {}
