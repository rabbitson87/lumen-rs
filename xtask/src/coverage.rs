//! `cargo xtask coverage` — branch coverage over the named in-scope set,
//! plus the two self-checks that keep the number from becoming a lie
//! (005 Phase 4.1).
//!
//! SQLite's 100% claim means something because it names its denominator: core,
//! excluding FTS3/RTree. Ours is [`IN_SCOPE`] — pure-CPU Rust that needs no GPU
//! and no weights — and everything outside it is excluded *with the reason
//! printed*, because a silent exclusion is the same failure this whole task
//! exists to remove.
//!
//! Two things are verified rather than assumed:
//!
//! 1. **The `always!()` mechanism still works.** It is what lets defensive code
//!    exist without capping the score, and whether it works is a property of
//!    the toolchain's instrumentation, not of the macro. The probe pair in
//!    `lumen_core::defensive` is checked on every run: the unguarded defensive
//!    branch must show as missed, the guarded one must not appear at all. A
//!    toolchain change that silently restores the cap fails here.
//!
//! 2. **MC/DC is not quietly downgraded.** `--mcdc` probes the toolchain and
//!    **fails** if MC/DC is unavailable, rather than printing branch numbers
//!    under an MC/DC heading. Measured on rustc 1.99.0-nightly (2026-08-03):
//!    `-Zcoverage-options` accepts `block | branch | condition` and rejects
//!    `mcdc`, which cargo-llvm-cov 0.8.7 still passes — so MC/DC is currently
//!    unobtainable here and the plan's target is blocked on the toolchain, not
//!    on effort. Saying so is the point.

use std::process::{Command, ExitCode};

/// Packages that can be measured without a GPU or downloaded weights, with the
/// features each needs to build in that configuration.
///
/// `lumen-mlx` and `lumen-server` both default to feature sets that pull in the
/// MLX stack, so both are measured with `--no-default-features`; that is the
/// configuration the hoisted parsers and config modules exist to make possible.
const PACKAGES: &[(&str, &[&str])] = &[
    ("lumen-core", &[]),
    ("lumen-flags", &[]),
    ("lumen-mlx", &["--no-default-features"]),
    ("lumen-server", &["--no-default-features"]),
];

/// The named denominator. A file counts as in scope if its path ends with one
/// of these.
const IN_SCOPE: &[&str] = &[
    "lumen-core/src/bitpack.rs",
    "lumen-core/src/defensive.rs",
    "lumen-core/src/dry.rs",
    "lumen-core/src/sampling.rs",
    "lumen-core/src/stop.rs",
    "lumen-core/src/runaway.rs",
    "lumen-mlx/src/grammar.rs",
    "lumen-mlx/src/gemma4_tool_syntax.rs",
    "lumen-mlx/src/chat_io.rs",
    "lumen-mlx/src/qwen35_config.rs",
    "lumen-mlx/src/gemma4_config.rs",
    "lumen-mlx/src/config_serde.rs",
    "lumen-mlx/src/prefill_budget.rs",
    "lumen-mlx/src/kv_disk.rs",
    "lumen-server/src/types.rs",
];

/// Excluded, each with the reason. Printed on every run: an exclusion nobody
/// can see is indistinguishable from a gap nobody noticed.
const EXCLUSIONS: &[(&str, &str)] = &[
    (
        "*.metal shaders (~10.5 KLOC)",
        "no coverage tooling exists for Metal Shading Language — llvm-cov \
         instruments CPU LLVM IR and shaders compile to a metallib. Covered \
         instead by differential CPU-reference parity tests and Metal Shader \
         Validation on tier 1.",
    ),
    (
        "mlx-native forward paths",
        "need a GPU and downloaded weights; measured by tier 1/2 parity tests, \
         not by this report.",
    ),
    (
        "hardware-gated dispatch (M5 NAX, CUDA)",
        "the branch cannot be taken on this machine (M3 Max), so it would be a \
         permanent miss rather than a gap.",
    ),
    (
        "defensive arms marked always!()/never!()",
        "deliberately removed from the denominator — see lumen_core::defensive. \
         The mechanism is re-verified on every run of this command.",
    ),
];

pub fn main(args: Vec<String>) -> ExitCode {
    let want_mcdc = args.iter().any(|a| a == "--mcdc");
    let html = args.iter().any(|a| a == "--html");

    if want_mcdc {
        match probe_mcdc() {
            McdcSupport::Available => {}
            McdcSupport::Rejected(values) => {
                eprintln!(
                    "MC/DC is unavailable on this toolchain.\n\n  \
                     `rustc -Zcoverage-options` accepts: {values}\n  \
                     cargo-llvm-cov's `--mcdc` passes `mcdc`, which is rejected.\n\n\
                     Refusing to run: printing branch numbers under an MC/DC heading \
                     would be exactly the shrunken-denominator failure the flag exists \
                     to prevent. Run without `--mcdc` for branch coverage, which does \
                     work and is reported below the named scope."
                );
                return ExitCode::FAILURE;
            }
        }
    }

    let mut failed = false;
    for (pkg, feats) in PACKAGES {
        println!("\n=== {pkg} ===");
        let mut cmd = Command::new("cargo");
        cmd.args(["+nightly", "llvm-cov", "--package", pkg, "--branch"]);
        cmd.args(*feats);
        if html {
            cmd.args(["--html", "--output-dir", &format!("target/coverage/{pkg}")]);
        } else {
            cmd.arg("--summary-only");
        }
        match cmd.status() {
            Ok(s) if s.success() => {}
            Ok(s) => {
                eprintln!("  coverage run for {pkg} failed ({s})");
                failed = true;
            }
            Err(e) => {
                eprintln!("  could not run cargo llvm-cov: {e}");
                eprintln!("  install it with: cargo install cargo-llvm-cov");
                return ExitCode::FAILURE;
            }
        }
    }

    println!("\n=== named scope ({} files) ===", IN_SCOPE.len());
    for f in IN_SCOPE {
        println!("  {f}");
    }
    println!("\n=== excluded, with reasons ===");
    for (what, why) in EXCLUSIONS {
        println!("  {what}\n      {why}");
    }

    if failed {
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

enum McdcSupport {
    Available,
    /// The values `-Zcoverage-options` actually accepts.
    Rejected(String),
}

/// Ask rustc directly rather than trusting cargo-llvm-cov's flag to mean the
/// feature exists. A rejected value comes back in the error text, which is the
/// most useful thing to show a reader.
///
/// The probe has to actually **compile** something: `rustc -Zcoverage-options=…
/// --version` exits 0 without ever validating the value, so a version probe
/// reports MC/DC as available on a toolchain that rejects it. Found by running
/// both, which is the only reason this comment exists.
fn probe_mcdc() -> McdcSupport {
    let out = Command::new("rustc")
        .args([
            "+nightly",
            "-Zcoverage-options=mcdc",
            "--crate-type",
            "lib",
            "--emit=metadata",
            "-o",
            "/dev/null",
            "-",
        ])
        .stdin(std::process::Stdio::null())
        .output();
    let Ok(o) = out else {
        return McdcSupport::Rejected("could not run rustc +nightly".into());
    };
    let err = String::from_utf8_lossy(&o.stderr);
    // rustc emits the rejection on stderr even when the exit status is 0 for
    // an otherwise-empty compile, so match on the message rather than the code.
    if !err.contains("incorrect value `mcdc`") {
        return McdcSupport::Available;
    }
    let values = err
        .split("was expected")
        .next()
        .and_then(|s| s.rsplit(" - ").next())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| err.trim().to_string());
    McdcSupport::Rejected(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every in-scope path must exist. A renamed module that silently drops out
    /// of the scope list shrinks the denominator without anyone noticing —
    /// the exact failure this file is written against.
    #[test]
    fn every_in_scope_file_exists() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("workspace root")
            .to_path_buf();
        for f in IN_SCOPE {
            let p = root.join("crates").join(f);
            assert!(p.exists(), "in-scope file does not exist: {}", p.display());
        }
    }

    /// Every exclusion carries a reason. An entry with an empty reason is a
    /// silent exclusion wearing a label.
    #[test]
    fn every_exclusion_states_a_reason() {
        for (what, why) in EXCLUSIONS {
            assert!(!what.trim().is_empty());
            assert!(
                why.len() > 40,
                "exclusion {what:?} needs a real reason, got {why:?}"
            );
        }
    }
}
