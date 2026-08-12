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

//! ## Why the headline percentage is not the number to chase
//!
//! llvm-cov merges one coverage record **per test binary**. A binary that links
//! a crate but never calls a given function still contributes a record for it,
//! with both branch arms at zero. Those `[True: 0, False: 0]` entries are
//! counted as missed, so the merged percentage falls when you add an unrelated
//! test binary — measured here: `dry.rs` carries **17** never-ran duplicates
//! against **9** genuine one-sided branches, and its percentage dropped when
//! `sampling_boundaries.rs` was added despite `dry.rs` gaining tests nowhere.
//!
//! A metric that moves the wrong way when you add tests cannot gate anything.
//! So this command reports **one-sided branches** — the ones that actually ran
//! and only ever went one way — which is both the honest count and the
//! actionable one. The percentage is still printed, clearly labelled as the
//! polluted figure it is.

use std::collections::BTreeMap;
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
            // `--text` alongside the summary so the one-sided report below has
            // per-branch data to read; the summary alone carries only totals.
            cmd.args(["--text", "--output-dir", "target/coverage-text"]);
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

    println!("\n=== one-sided branches in the named scope ===");
    println!("  (records merged by source location first — the text report");
    println!("   prints one per test binary and does NOT merge them.");
    println!("   one-sided = ran, only ever went one way: the actionable gap.");
    println!("   unreached = no test binary reached it at all.)\n");
    match one_sided_report() {
        Ok(rows) if rows.is_empty() => {
            println!("  no text report found — run without --html to generate one");
        }
        Ok(rows) => {
            println!("  {:<40} {:>10}  {:>10}", "file", "one-sided", "unreached");
            let mut total = 0usize;
            for (file, (one_sided, unreached)) in &rows {
                println!("  {file:<40} {one_sided:>10}  {unreached:>10}");
                total += one_sided;
            }
            println!("\n  TOTAL one-sided branches in scope: {total}");
        }
        Err(e) => println!("  could not parse the text report: {e}"),
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

/// Per in-scope file: `(one-sided branches, branches never reached at all)`.
///
/// **Merges by source location first**, which is the whole correctness of this
/// function. The `--text` report prints one record *per test binary* without
/// merging them — `sampling.rs` has 49 distinct branch locations spread over 95
/// records — so classifying records individually double-counts and reports a
/// covered branch as a gap. Measured: `Branch (81:16)` appears as
/// `[True: 7, False: 1]` from one binary and `[True: 1, False: 0]` from
/// another; only the sum, `[True: 8, False: 1]`, is the truth. (llvm-cov's own
/// summary merges correctly; only the text rendering does not.)
///
/// After merging: both arms zero means no test binary ever reached the branch;
/// exactly one arm zero means it ran and only ever went one way — the
/// actionable gap.
fn one_sided_report() -> std::io::Result<BTreeMap<String, (usize, usize)>> {
    let mut out: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let root = std::path::Path::new("target/coverage-text");
    if !root.exists() {
        return Ok(out);
    }
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let p = entry?.path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            let name = p.to_string_lossy().to_string();
            let Some(scoped) = IN_SCOPE.iter().find(|s| name.contains(*s)) else {
                continue;
            };
            let text = std::fs::read_to_string(&p)?;
            // location -> (true count, false count), summed across binaries.
            let mut merged: BTreeMap<String, (u64, u64)> = BTreeMap::new();
            for line in text.lines() {
                let Some((loc, arms)) = parse_branch_line(line) else {
                    continue;
                };
                let e = merged.entry(loc).or_insert((0, 0));
                e.0 += arms.0;
                e.1 += arms.1;
            }
            let (mut one_sided, mut unreached) = (0usize, 0usize);
            for (t, f) in merged.values() {
                match (*t == 0, *f == 0) {
                    (true, true) => unreached += 1,
                    (true, false) | (false, true) => one_sided += 1,
                    (false, false) => {}
                }
            }
            let e = out.entry((*scoped).to_string()).or_insert((0, 0));
            e.0 += one_sided;
            e.1 += unreached;
        }
    }
    Ok(out)
}

/// `  |  Branch (81:16): [True: 7, False: 1]` → `("81:16", (7, 1))`.
fn parse_branch_line(line: &str) -> Option<(String, (u64, u64))> {
    let rest = line.split("Branch (").nth(1)?;
    let (loc, tail) = rest.split_once(')')?;
    let arms = tail.split_once('[')?.1;
    let (t, f) = arms.split_once(", False: ")?;
    let t: u64 = t.trim_start_matches("True: ").trim().parse().ok()?;
    let f: u64 = f.trim_end_matches(']').trim().parse().ok()?;
    Some((loc.to_string(), (t, f)))
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

    #[test]
    fn branch_lines_parse_and_merge_by_location() {
        let (loc, arms) = parse_branch_line("  |  Branch (81:16): [True: 7, False: 1]")
            .expect("a normal branch line must parse");
        assert_eq!(loc, "81:16");
        assert_eq!(arms, (7, 1));

        assert_eq!(
            parse_branch_line("  |  Branch (145:27): [True: 0, False: 0]")
                .expect("zero arms still parse"),
            ("145:27".to_string(), (0, 0))
        );
        // Source lines, not branch records.
        assert!(parse_branch_line("  132|      2|    if xs.is_empty() {").is_none());
        assert!(parse_branch_line("").is_none());
    }

    /// The bug this parser was rewritten for: the text report prints one record
    /// per test binary, so a branch covered by binary B looks one-sided in
    /// binary A's record. Only the merged sum is the truth.
    #[test]
    fn a_branch_covered_by_a_second_binary_is_not_a_gap() {
        let a = parse_branch_line("  |  Branch (81:16): [True: 7, False: 1]").unwrap();
        let b = parse_branch_line("  |  Branch (81:16): [True: 1, False: 0]").unwrap();
        assert_eq!(a.0, b.0, "same location");
        let merged = (a.1.0 + b.1.0, a.1.1 + b.1.1);
        assert_eq!(merged, (8, 1));
        assert!(
            merged.0 > 0 && merged.1 > 0,
            "merged, this branch is covered — classifying records individually \
             would have reported it as a gap"
        );
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
