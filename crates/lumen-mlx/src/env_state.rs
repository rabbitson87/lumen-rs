//! Bench env-state logger.
//!
//! Anti-pattern #36 mitigation: every perf-measurement bench should log the
//! state of the env gates it depends on before timing begins, so a later
//! reader can tell whether the measurement was on the canonical path or a
//! legacy / experimental regime.
//!
//! Usage:
//! ```ignore
//! use lumen_mlx::env_state::log_env_state;
//! log_env_state("mlx-e2e", &[
//!     "LUMEN_MLX_BACKEND",
//!     "LUMEN_NATIVE_KV_STEP_PREALLOC",
//!     "LUMEN_NATIVE_ALLOC_REUSE",
//! ]);
//! ```
//!
//! Output (each var on its own line to keep the log greppable):
//! ```text
//! [mlx-e2e env] LUMEN_MLX_BACKEND=native (explicit)
//! [mlx-e2e env] LUMEN_NATIVE_KV_STEP_PREALLOC=(unset → library default = 1)
//! [mlx-e2e env] LUMEN_NATIVE_ALLOC_REUSE=(unset → library default = 1)
//! ```
//!
//! The "library default" annotation is informational — derived from
//! `KNOWN_DEFAULTS` below. Update that table when a library default flips.

/// Library-default state for known perf-impacting env gates. The bench logger
/// surfaces these annotations alongside the env value so the reader can spot
/// "unset" entries that nonetheless represent an active fix.
///
/// Each entry is `(var_name, default_state)`. `default_state` is a short
/// human-readable string — typically `"1 (...)"` for default-on gates and
/// `"0 (...)"` for default-off opt-ins, with a brief context tag in the
/// parens.
const KNOWN_DEFAULTS: &[(&str, &str)] = &[
    // Gemma 4
    ("LUMEN_GEMMA4_FUSE_DENSE_MLP", "1 (compile-slot fuse)"),
    ("LUMEN_GEMMA4_FUSE_EXPERTS", "1 (compile-slot fuse)"),
    ("LUMEN_GEMMA4_FUSE_ROUTER", "1 (compile-slot fuse)"),
    (
        "LUMEN_GEMMA4_CUSTOM_FLASH_ATTN",
        "1 (M4.8 LANDED 2026-05-14, stride-aware kernel, zero-copy strided K/V)",
    ),
    (
        "LUMEN_GEMMA4_PREFILL_SYNC",
        "1 (Phase 1.8 RESOLVED 2026-05-14, drain prefill GPU before decode)",
    ),
    // Native MLX runner (Qwen3.5-MoE / Qwen3.6-27B / Qwen3.6-35B)
    ("LUMEN_NATIVE_KV_STEP_PREALLOC", "1 (G2 LANDED)"),
    ("LUMEN_NATIVE_ALLOC_REUSE", "1 (LANDED 2026-05-11)"),
    (
        "LUMEN_NATIVE_LINEAR_ATTN_SCALE_FUSE",
        "1 (LANDED 2026-05-11, scale_fuse −3.33σ win)",
    ),
    ("LUMEN_NATIVE_DEFER_CLEAR_CACHE", "1 (Phase C #5 LANDED)"),
    ("LUMEN_NATIVE_FUSE_SWIGLU", "1 (LANDED)"),
    ("LUMEN_NATIVE_COMPILE", "1 (compute_g compile cache LANDED)"),
    ("LUMEN_NATIVE_COMPILE_ROUTING", "0 (A/B WASH)"),
    ("LUMEN_NATIVE_CACHED_STREAM", "0 (A/B WASH)"),
    ("LUMEN_NATIVE_CONV_SLICE", "0 (A/B WASH)"),
    (
        "LUMEN_NATIVE_RMS_NORM_GATED_FUSED",
        "0 (reserved for super-kernel composition)",
    ),
    (
        "LUMEN_NATIVE_FUSE_SIGMOID_MUL",
        "0 (anti-pattern #30 calibration)",
    ),
    (
        "LUMEN_NATIVE_STREAM_INCR",
        "0 (FALSIFIED, regression at warmup)",
    ),
    ("LUMEN_NATIVE_NO_CLEAR_CACHE", "0 (clear_cache fix opt-out)"),
    (
        "LUMEN_MLX_KV_BF16",
        "0 (halves full-attn KV: -33 KB/slot, +2-4% decode, but output can \
         change — see examples/kv_bf16_ab.rs)",
    ),
    (
        "LUMEN_QWEN35_ROPE_PRECOMPUTE_FREQS",
        "0 (opt-in; mirrors Gemma 4 — A/B WASH expected per MLX lazy dedup)",
    ),
    // Compute-g fusion (Qwen3.6-27B)
    (
        "LUMEN_FUSED_COMPUTE_G",
        "0 (Phase 19.A.4.1 production +3% NEGATIVE, anti-pattern #27)",
    ),
    // Backend / harness selection
    ("LUMEN_MLX_BACKEND", "native (default)"),
];

/// Resolve the library-default annotation for `var`. Returns the documented
/// state string if we know it, or a generic "(unset)" otherwise so the
/// reader knows we didn't track its default centrally.
fn default_annotation(var: &str) -> &'static str {
    for (name, ann) in KNOWN_DEFAULTS {
        if *name == var {
            return ann;
        }
    }
    "(unset; default unknown — add to env_state::KNOWN_DEFAULTS if relevant)"
}

/// Log the resolved state of every env var in `vars`, prefixed with
/// `tag`. Writes to stderr so it doesn't pollute structured stdout output.
pub fn log_env_state(tag: &str, vars: &[&str]) {
    for var in vars {
        match std::env::var(var) {
            Ok(v) => eprintln!("[{tag} env] {var}={v} (explicit)"),
            Err(_) => {
                let ann = default_annotation(var);
                eprintln!("[{tag} env] {var}=(unset → {ann})");
            }
        }
    }
}
