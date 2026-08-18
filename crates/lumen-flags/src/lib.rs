//! One macro for every `LUMEN_*` boolean gate (005 Phase 2).
//!
//! The audit that started task 005 counted **370 `LUMEN_*` env reads, 27 of
//! them documented, and exactly one testable on both paths**. Two recorded
//! defects (`rotating-cache-both-paths`, `causal-mask-builders-agree`) were
//! precisely "the alternate path was unreachable from any test". Every
//! hand-rolled flag makes four decisions independently — parse rule, caching,
//! overridability, documentation — and the drift between those decisions is
//! where the defects lived.
//!
//! [`flag!`] makes them once:
//!
//! ```rust
//! lumen_flags::flag! {
//!     /// An example gate. (Deliberately NOT a production env name: the
//!     /// equivalence matrix flips those in the real environment, and this
//!     /// doctest asserting the default was its very first casualty.)
//!     doctest_example {
//!         env: "LUMEN_FLAGS_DOCTEST_EXAMPLE",
//!         default: true,
//!         kind: Optimization,
//!     }
//! }
//! assert!(doctest_example::get());
//! doctest_example::with(false, || assert!(!doctest_example::get()));
//! ```
//!
//! Each invocation expands to a module with:
//!
//!   * `get()` — thread-local override, else process override, else the
//!     env-seeded `OnceLock`. The layering exists because this codebase needed
//!     all three at different times: production reads want once-per-process
//!     semantics; unit tests need a scoped, panic-safe, per-thread pin
//!     (`native_cache::test_support` was born because a `OnceLock` alone left
//!     the legacy path unreachable from any test); A/B harnesses need an
//!     unscoped process-wide setter (`set_kv_store_bf16` was born because the
//!     `OnceLock` could not be flipped between two in-process conditions).
//!   * `with(v, f)` — the scoped pin, restored on unwind.
//!   * `set(v)` / `clear()` — the harness setter.
//!   * a [`FlagDesc`] in [`REGISTRY`], collected at link time, from which
//!     `cargo xtask flags` generates `docs/env-flags.md` and decides which
//!     flags the equivalence matrix may flip.
//!
//! Parse rule, fixed everywhere: unset → default; `"0"` → false; any other
//! value → true. Flags needing fallback envs or fancier parsing stay
//! hand-rolled and out of the registry — better a visible exception than a
//! macro with four escape hatches.

use std::sync::atomic::{AtomicU8, Ordering};

pub use linkme;

/// What flipping the flag is allowed to do to model output. This field is the
/// load-bearing one: `cargo xtask flags` flips every `Optimization` flag and
/// requires the suite to pass identically, so a miscategorized flag either
/// breaks the matrix (Behavior marked Optimization) or silently escapes it
/// (the reverse).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlagKind {
    /// Must not change output — only speed or memory. The equivalence matrix
    /// flips these and requires green.
    Optimization,
    /// Intentionally changes output (e.g. `LUMEN_MLX_KV_BF16`, whose entire
    /// point is a different-but-equivalent numeric regime). Never flipped by
    /// the matrix.
    Behavior,
    /// Logging, dumps, timing — no effect on results, may cost speed.
    Diagnostic,
}

impl FlagKind {
    pub fn as_str(self) -> &'static str {
        match self {
            FlagKind::Optimization => "Optimization",
            FlagKind::Behavior => "Behavior",
            FlagKind::Diagnostic => "Diagnostic",
        }
    }
}

/// One registered flag. `doc` is the same text rustdoc shows on the module, so
/// the generated `docs/env-flags.md` cannot drift from the in-source docs.
#[derive(Debug)]
pub struct FlagDesc {
    pub env: &'static str,
    pub default: bool,
    pub kind: FlagKind,
    pub doc: &'static str,
    /// `module_path!()` of the declaring crate/module, for the docs table.
    pub declared_in: &'static str,
}

/// Link-time registry of every [`flag!`] in every linked crate.
#[linkme::distributed_slice]
pub static REGISTRY: [FlagDesc];

/// The registry, sorted by env name — the stable order the docs generator and
/// the equivalence matrix both use.
pub fn registry_sorted() -> Vec<&'static FlagDesc> {
    let mut v: Vec<&'static FlagDesc> = REGISTRY.iter().collect();
    v.sort_by_key(|d| d.env);
    v
}

/// The one parse rule. Public so the expansion can call it; not meant for
/// direct use.
#[doc(hidden)]
pub fn read_env(env: &str, default: bool) -> bool {
    match std::env::var(env) {
        Ok(v) => v != "0",
        Err(_) => default,
    }
}

/// Tri-state process override backing `set()`/`clear()`. Public for the
/// expansion; the `u8` states are 0 = forced off, 1 = forced on, 2 = unset.
#[doc(hidden)]
pub struct ProcessOverride(pub AtomicU8);

#[doc(hidden)]
impl Default for ProcessOverride {
    fn default() -> Self {
        Self::new()
    }
}

impl ProcessOverride {
    pub const fn new() -> Self {
        Self(AtomicU8::new(2))
    }
    pub fn get(&self) -> Option<bool> {
        match self.0.load(Ordering::Relaxed) {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }
    pub fn set(&self, v: Option<bool>) {
        self.0.store(v.map_or(2, u8::from), Ordering::Relaxed);
    }
}

/// Declare a boolean env flag. See the crate docs for the expansion contract.
#[macro_export]
macro_rules! flag {
    (
        $(#[doc = $doc:expr])+
        $vis:vis $name:ident {
            env: $env:literal,
            default: $default:literal,
            kind: $kind:ident $(,)?
        }
    ) => {
        $(#[doc = $doc])+
        ///
        /// Declared through [`lumen_flags::flag!`]; see `docs/env-flags.md`.
        $vis mod $name {
            /// Registry entry — link-time, so declaring the flag is
            /// registering it.
            #[$crate::linkme::distributed_slice($crate::REGISTRY)]
            #[linkme(crate = $crate::linkme)]
            static DESC: $crate::FlagDesc = $crate::FlagDesc {
                env: $env,
                default: $default,
                kind: $crate::FlagKind::$kind,
                doc: concat!($($doc, "\n"),+),
                declared_in: module_path!(),
            };

            static PROCESS: $crate::ProcessOverride = $crate::ProcessOverride::new();

            ::std::thread_local! {
                static THREAD: ::std::cell::Cell<Option<bool>> =
                    const { ::std::cell::Cell::new(None) };
            }

            /// Current value: thread pin → process override → env-seeded cache.
            pub fn get() -> bool {
                if let Some(v) = THREAD.with(::std::cell::Cell::get) {
                    return v;
                }
                if let Some(v) = PROCESS.get() {
                    return v;
                }
                static CACHED: ::std::sync::OnceLock<bool> = ::std::sync::OnceLock::new();
                *CACHED.get_or_init(|| $crate::read_env($env, $default))
            }

            /// Run `f` with the flag pinned for this thread, restoring the
            /// previous pin even on panic. The unit-test override: scoped, so
            /// parallel test threads cannot interfere.
            pub fn with<R>(value: bool, f: impl FnOnce() -> R) -> R {
                struct Reset(Option<bool>);
                impl Drop for Reset {
                    fn drop(&mut self) {
                        THREAD.with(|c| c.set(self.0));
                    }
                }
                let _reset = Reset(THREAD.with(::std::cell::Cell::get));
                THREAD.with(|c| c.set(Some(value)));
                f()
            }

            // The three below are generated for every flag, and no single flag
            // uses all of them — `set`/`clear` matter to an in-process A/B
            // harness, `describe` to whoever walks the registry. The macro
            // cannot know which, so the allow lives here rather than being
            // pasted at each call site that happens not to need one.
            /// Process-wide override for A/B harnesses that flip a flag
            /// between in-process conditions — the thing a bare `OnceLock`
            /// cannot do. Overridden per-thread by [`with`].
            #[allow(dead_code)]
            pub fn set(value: bool) {
                PROCESS.set(Some(value));
            }

            /// Drop the process override; `get()` falls back to the env-seeded
            /// value.
            #[allow(dead_code)]
            pub fn clear() {
                PROCESS.set(None);
            }

            /// This flag's registry entry.
            #[allow(dead_code)]
            pub fn describe() -> &'static $crate::FlagDesc {
                &DESC
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    flag! {
        /// Test flag: default-on optimization.
        on_by_default {
            env: "LUMEN_FLAGS_TEST_ON",
            default: true,
            kind: Optimization,
        }
    }

    flag! {
        /// Test flag: default-off behavior.
        off_by_default {
            env: "LUMEN_FLAGS_TEST_OFF",
            default: false,
            kind: Behavior,
        }
    }

    #[test]
    fn defaults_apply_when_env_is_unset() {
        assert!(on_by_default::get());
        assert!(!off_by_default::get());
    }

    #[test]
    fn with_pins_and_restores_even_on_panic() {
        assert!(on_by_default::get());
        on_by_default::with(false, || assert!(!on_by_default::get()));
        assert!(on_by_default::get(), "pin must be restored");

        let r = std::panic::catch_unwind(|| {
            on_by_default::with(false, || panic!("unwind through the pin"))
        });
        assert!(r.is_err());
        assert!(on_by_default::get(), "pin must be restored across unwind");
    }

    #[test]
    fn with_nests() {
        on_by_default::with(false, || {
            on_by_default::with(true, || assert!(on_by_default::get()));
            assert!(!on_by_default::get(), "inner pin must restore to outer");
        });
    }

    #[test]
    fn set_overrides_process_wide_and_with_wins_over_set() {
        off_by_default::set(true);
        assert!(off_by_default::get());
        off_by_default::with(false, || assert!(!off_by_default::get()));
        assert!(off_by_default::get());
        off_by_default::clear();
        assert!(!off_by_default::get());
    }

    /// The registry is the whole point — a flag that does not appear there is
    /// invisible to the docs generator and the equivalence matrix.
    #[test]
    fn declared_flags_appear_in_the_registry() {
        let sorted = registry_sorted();
        let on = sorted
            .iter()
            .find(|d| d.env == "LUMEN_FLAGS_TEST_ON")
            .expect("on_by_default registered");
        assert!(on.default);
        assert_eq!(on.kind, FlagKind::Optimization);
        assert!(on.doc.contains("default-on optimization"));
        let off = sorted
            .iter()
            .find(|d| d.env == "LUMEN_FLAGS_TEST_OFF")
            .expect("off_by_default registered");
        assert!(!off.default);
        assert_eq!(off.kind, FlagKind::Behavior);
    }

    /// The parse rule in one place: `"0"` is the only falsy string.
    #[test]
    fn parse_rule_is_zero_means_off() {
        assert!(!read_env("LUMEN_FLAGS_TEST_UNSET_X", false));
        assert!(read_env("LUMEN_FLAGS_TEST_UNSET_X", true));
        // Values come via the real environment in integration use; the unit
        // check exercises the code path through a set variable.
        // SAFETY: test-only var nothing else reads, single test binary.
        unsafe { std::env::set_var("LUMEN_FLAGS_TEST_PARSE", "0") };
        assert!(!read_env("LUMEN_FLAGS_TEST_PARSE", true));
        unsafe { std::env::set_var("LUMEN_FLAGS_TEST_PARSE", "1") };
        assert!(read_env("LUMEN_FLAGS_TEST_PARSE", false));
        unsafe { std::env::set_var("LUMEN_FLAGS_TEST_PARSE", "yes") };
        assert!(
            read_env("LUMEN_FLAGS_TEST_PARSE", false),
            "any non-\"0\" value is truthy"
        );
    }
}
