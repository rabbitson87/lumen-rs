//! Persistent `mlx_rs::compile()` wrapper cache.
//!
//! the per-call pattern
//!
//! ```ignore
//! let mut compiled = mlx_rs::transforms::compile::compile(my_fn, true);
//! compiled(args)?
//! ```
//!
//! creates a fresh `Compiled<F, G>` struct and runs its `Drop`
//! (`mlx_detail_compile_erase`) on every call. The Phase 1 microbench
//! (`crates/lumen-mlx/examples/bench_compile_overhead.rs`) measured this
//! at ~0.0278ms/call on Apple Silicon — at 30 layers per decode step that is
//! ~0.83ms wasted on wrapper alloc/Drop.
//!
//! This module provides a `OnceLock<Mutex<CompiledMulti>>`-backed cache that
//! constructs the wrapper exactly once per call site (via the
//! `compile_boxed_slice` helper added to our path-based mlx-rs fork) and
//! reuses it for the lifetime of the process. The mlx-c side compile cache
//! entry is preserved as long as the box is alive.

#[cfg(feature = "mlx-native")]
mod imp {
    use std::process;
    use std::sync::{Mutex, OnceLock};

    use mlx_rs::Array;
    use mlx_rs::error::Exception;
    use mlx_rs::transforms::compile::{compile_boxed_slice, compile_boxed_slice_refs};

    pub type CompiledMulti = Box<dyn FnMut(&[Array]) -> Result<Vec<Array>, Exception> + Send>;

    /// Ref-variant slice box — argument is `&[&Array]` so callers can dispatch
    /// without cloning each input Array (~0.3μs FFI per refcount-bump avoided).
    pub type CompiledMultiRefs =
        Box<dyn for<'a> FnMut(&[&'a Array]) -> Result<Vec<Array>, Exception> + Send>;

    /// Invoke a persistently-cached `mlx_rs::compile()` wrapper.
    ///
    /// On first call, `inner_fn` is wrapped via `compile_boxed_slice(inner_fn,
    /// shapeless)` and stored in `slot`. Subsequent calls reuse the stored
    /// box, avoiding the per-call wrapper alloc/Drop measured in Phase 1.
    ///
    /// `shapeless` follows Python's `@partial(mx.compile, shapeless=...)`
    /// convention: `true` lets the compiled graph accept varying input
    /// shapes (lighter — but ops with shape-dependent output like
    /// `slice`/`split` may fail trace), `false` bakes the input shape into
    /// the graph (heavier — re-compile per distinct shape, but supports any
    /// op).
    ///
    /// `inner_fn` must be a stable identifier (fn pointer or top-level fn) so
    /// its `TypeId` — used as the mlx-c side cache key — is consistent across
    /// callers. After the first init, the `inner_fn` value passed in
    /// subsequent calls is discarded (only the cached box is used).
    ///
    /// On `Mutex` poisoning, this aborts the process: a panic during a prior
    /// `Compiled` call leaves raw mlx-c handles in an indeterminate state
    /// that is not safe to recover from. Fail-fast is the only correct option.
    pub fn invoke_compiled_multi<F>(
        slot: &'static OnceLock<Mutex<CompiledMulti>>,
        inner_fn: F,
        shapeless: bool,
        args: &[Array],
    ) -> Result<Vec<Array>, Exception>
    where
        F: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'static + Send,
    {
        let mutex = slot.get_or_init(|| Mutex::new(compile_boxed_slice(inner_fn, shapeless)));
        let mut guard = match mutex.lock() {
            Ok(g) => g,
            Err(_) => {
                eprintln!(
                    "native_compile_cache: Mutex poisoned (a prior compiled call panicked); \
                     aborting to avoid using corrupt mlx-c compile state"
                );
                process::abort();
            }
        };
        guard(args)
    }

    /// Ref-variant of [`invoke_compiled_multi`]. Accepts `args: &[&Array]` so
    /// callers can avoid the `[a.clone(), b.clone()]` refcount-bump pattern
    /// at dispatch sites. Saves ~0.3μs FFI (`mlx_array_set`) per input slot.
    ///
    /// See `compile_boxed_slice_refs` in the mlx-rs fork for the inner closure
    /// contract: `f` still takes `&[Array]` because that's what mlx-c hands
    /// back during graph resolution — only the outer dispatch slice's element
    /// type differs.
    pub fn invoke_compiled_multi_refs<F>(
        slot: &'static OnceLock<Mutex<CompiledMultiRefs>>,
        inner_fn: F,
        shapeless: bool,
        args: &[&Array],
    ) -> Result<Vec<Array>, Exception>
    where
        F: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'static + Send,
    {
        let mutex = slot.get_or_init(|| Mutex::new(compile_boxed_slice_refs(inner_fn, shapeless)));
        let mut guard = match mutex.lock() {
            Ok(g) => g,
            Err(_) => {
                eprintln!(
                    "native_compile_cache: Mutex poisoned (a prior compiled call panicked); \
                     aborting to avoid using corrupt mlx-c compile state"
                );
                process::abort();
            }
        };
        guard(args)
    }
}

#[cfg(feature = "mlx-native")]
pub use imp::*;
