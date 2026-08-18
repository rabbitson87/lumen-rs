//! `cargo xtask faults` — every fault sweep, in one command (005 Phase 3).
//!
//! The sweeps are ordinary tests and run inside `cargo xtask test` along with
//! everything else. This exists because they are also a **category** that gets
//! asked about as a unit — "does a corrupt checkpoint still fail cleanly?" —
//! and because the answer to that question should not require knowing which six
//! test targets happen to implement it.
//!
//! `--cancel` is the interrupt-integrity half, and it is separate for a reason
//! rather than for tidiness: the byte-level sweeps ask what happens to bad
//! *input*, while cancellation asks what happens to *resources* when a good
//! request dies partway. SQLite states the second one as "committed or
//! completely rolled back"; applied to a cache rather than a database, it reads
//! — after an interruption, every KV snapshot is either owned by a live entry
//! or released, and never both or neither.
//!
//! What is **not** here: the tier-2 half of interrupt integrity (active memory
//! returning to baseline needs a loaded model). `docs/release-checklist.md`
//! carries it.

use std::process::{Command, ExitCode};

/// A fault sweep: the test target that implements it, and what it establishes.
struct Sweep {
    /// `cargo test --test <target>`, or a filter into a crate's lib tests.
    target: &'static str,
    package: &'static str,
    /// Empty runs the whole target; otherwise a libtest filter.
    filter: &'static str,
    /// `--features` value; empty selects the crate default.
    features: &'static str,
    what: &'static str,
}

const SWEEPS: &[Sweep] = &[
    Sweep {
        target: "kv_disk_faults",
        package: "lumen-mlx",
        filter: "",
        features: "",
        what: "every truncation offset and every single-byte flip of an LKV1 \
               snapshot; found the 280 TB allocation bomb that SIGABRTs",
    },
    Sweep {
        target: "kv_disk_write_faults",
        package: "lumen-mlx",
        filter: "",
        features: "",
        what: "every partial write, each fed back to the reader — the writer \
               cannot emit anything outside the set the reader rejects",
    },
    Sweep {
        target: "kv_disk_store_policy",
        package: "lumen-mlx",
        filter: "",
        features: "",
        what: "TTL, LRU eviction, key sanitization (a key cannot escape the \
               cache directory) and the fingerprint check",
    },
    Sweep {
        target: "config_faults",
        package: "lumen-mlx",
        filter: "",
        features: "",
        what: "corrupt / truncated / null-bearing config.json against the Qwen \
               parser; found the JGOS-31B null shape",
    },
    Sweep {
        target: "gemma4_config_faults",
        package: "lumen-mlx",
        filter: "",
        features: "",
        what: "the same sweep against Gemma 4, where the null shape was still \
               live despite being remembered as fixed",
    },
    Sweep {
        target: "weights_faults",
        package: "lumen-mlx",
        filter: "",
        features: "mlx-native",
        what: "truncated safetensors shards; found the silent-corruption path \
               where a partial download loads and returns wrong values",
    },
    Sweep {
        target: "prefill_budget_faults",
        package: "lumen-mlx",
        filter: "",
        features: "",
        what: "the allocation-budget guard swept over every magnitude — this \
               project's malloc-fail-at-N analogue, since MLX allocation is \
               not fallible from Rust",
    },
];

/// Interrupt integrity. These live in `lumen-mlx`'s lib tests because
/// `PrefixCacheStore` is private; the seam that makes them tier 0 is that it
/// takes its runner as a trait.
const CANCEL: &[Sweep] = &[Sweep {
    target: "",
    package: "lumen-mlx",
    filter: "prefix_cache::tests::",
    features: "",
    what: "a runner that fails on its Nth call, swept across four scenarios: \
           every live snapshot stays owned or released, no entry references a \
           released snapshot, nothing is left in_use, and the next request \
           still succeeds",
}];

pub fn main(args: Vec<String>) -> ExitCode {
    let cancel = args.iter().any(|a| a == "--cancel");
    let sweeps: &[Sweep] = if cancel { CANCEL } else { SWEEPS };

    if cancel {
        println!("interrupt integrity — resources, not durable state\n");
    } else {
        println!("fault sweeps — corrupt input at every offset\n");
    }

    let mut failed = Vec::new();
    for s in sweeps {
        let label = if s.target.is_empty() {
            s.filter
        } else {
            s.target
        };
        println!("── {label}");
        println!("     {}", s.what);

        let mut cmd = Command::new("cargo");
        cmd.args(["test", "-p", s.package]);
        if !s.features.is_empty() {
            cmd.args(["--features", s.features]);
        }
        if s.target.is_empty() {
            cmd.arg("--lib");
        } else {
            cmd.args(["--test", s.target]);
        }
        cmd.arg("--");
        if !s.filter.is_empty() {
            cmd.arg(s.filter);
        }
        cmd.arg("--test-threads=1");

        match cmd.status() {
            Ok(st) if st.success() => {}
            Ok(_) => failed.push(label),
            Err(e) => {
                eprintln!("   could not run: {e}");
                failed.push(label);
            }
        }
    }

    println!();
    if failed.is_empty() {
        println!("all {} sweeps green", sweeps.len());
        if !cancel {
            println!("  interrupt integrity is a separate question: `--cancel`");
        } else {
            println!(
                "  the tier-2 half — active memory returning to its pre-request\n\
                 \x20 baseline — needs a loaded model; see docs/release-checklist.md"
            );
        }
        ExitCode::SUCCESS
    } else {
        println!("FAILED: {}", failed.join(", "));
        ExitCode::FAILURE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every sweep says what it establishes. A list of test-target names is a
    /// thing you have to go read the tests to understand, which defeats the
    /// point of grouping them.
    #[test]
    fn every_sweep_states_what_it_establishes() {
        for s in SWEEPS.iter().chain(CANCEL) {
            assert!(
                s.what.len() > 40,
                "sweep {:?} needs a real description",
                s.target
            );
            assert!(
                !s.target.is_empty() || !s.filter.is_empty(),
                "a sweep must name either a test target or a filter"
            );
        }
    }

    /// The sweep list must not drift from the test targets that exist. A stale
    /// entry makes the command fail for a reason that has nothing to do with
    /// the code under test; a missing one makes the command quietly narrower
    /// than it claims.
    #[test]
    fn every_named_target_exists() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("workspace root")
            .to_path_buf();
        for s in SWEEPS {
            if s.target.is_empty() {
                continue;
            }
            let p = root
                .join("crates")
                .join(s.package)
                .join("tests")
                .join(format!("{}.rs", s.target));
            assert!(p.exists(), "no such test target: {}", p.display());
        }
    }
}
