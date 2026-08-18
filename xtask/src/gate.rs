//! `cargo xtask gate` — the single pre-push command (005 Phase 5).
//!
//! Everything this task built is only useful if someone runs it, and a list of
//! seven commands in a document is a list nobody runs in full. This is that
//! list, executed in the order that fails cheapest first: a feature-combination
//! check before clippy, clippy before tests, tier-0 tests before the GPU suite,
//! and the hygiene greps last because they are instant and their failure is a
//! one-line fix.
//!
//! **Feature-combination checks come first for a specific reason.** The
//! `default = []` configuration of `lumen-mlx` once accumulated 72 compile
//! errors without anyone noticing, because nothing ever built it — the same
//! disease as an untested branch, one level up. A configuration nobody compiles
//! rots exactly as quietly as a branch nobody executes.
//!
//! Soak-scale work is deliberately **not** here: fuzzing, the full flag matrix
//! and coverage all take minutes to hours. Those belong to a release, and
//! `docs/release-checklist.md` names them. A pre-push gate that takes an hour
//! is a pre-push gate people skip.

use std::process::{Command, ExitCode};
use std::time::Instant;

/// One gate step. `optional` steps report and keep going — used for checks that
/// depend on a tool the machine may not have, where failing the whole gate
/// would punish the wrong person.
struct Step {
    name: &'static str,
    /// Why this step exists, printed on failure so the reader does not have to
    /// go looking for it.
    rationale: &'static str,
    program: &'static str,
    args: &'static [&'static str],
    optional: bool,
}

const STEPS: &[Step] = &[
    Step {
        name: "check: default features",
        rationale: "the `default = []` build once carried 72 errors because nothing \
                    compiled it; a configuration nobody builds rots as quietly as a \
                    branch nobody runs",
        program: "cargo",
        args: &["check", "--workspace", "--all-targets"],
        optional: false,
    },
    Step {
        name: "check: mlx-native",
        rationale: "the configuration that actually ships; the default build \
                    passing says nothing about it, because most of the crate is \
                    behind this gate",
        program: "cargo",
        args: &[
            "check",
            "--workspace",
            "--all-targets",
            "--features",
            "lumen-mlx/mlx-native",
        ],
        optional: false,
    },
    Step {
        name: "fmt",
        rationale: "a formatting diff in a review hides the change being reviewed",
        program: "cargo",
        args: &["fmt", "--all", "--check"],
        optional: false,
    },
    Step {
        // `--all` above means "all workspace members", and the fuzz crate is
        // not one — same blind spot that let two of its targets stop compiling.
        // Stable, unlike the compile check below, because rustfmt does not need
        // to resolve `libfuzzer-sys`.
        name: "fmt: fuzz targets",
        rationale: "`cargo fmt --all` covers workspace members, and the fuzz crate is \
                    excluded from the workspace",
        program: "cargo",
        args: &["fmt", "--manifest-path", "fuzz/Cargo.toml", "--check"],
        optional: false,
    },
    Step {
        name: "clippy",
        // BLOCKING as of the backlog reaching zero. It was report-only while
        // 349 warnings stood, because a gate that fails on day one is a gate
        // people skip.
        //
        // Two things the cleanup found that report-only had been hiding:
        //
        // * a **deny-by-default error** (`approx_constant` on a `3.14` in a test
        //   fixture) had been failing `cargo clippy` outright for as long as the
        //   step was optional. Not a warning among 349 — an error, invisible
        //   because nothing read the exit code.
        // * two `drop(x)` calls on borrows, which free nothing and were written
        //   believing they did.
        //
        // Four lints are allowed workspace-wide in the root `Cargo.toml`, each
        // with its reason next to it. Everything else is expected to be zero:
        // one lint left warning forever re-teaches the skimming this step was
        // reinstated to stop.
        rationale: "lints that catch real bugs, not just style — blocking now \
                    that the backlog is zero; report-only was hiding a \
                    deny-by-default error, not just warnings",
        program: "cargo",
        args: &[
            "clippy",
            "--workspace",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ],
        optional: false,
    },
    Step {
        name: "frontend check",
        rationale: "the gate was Rust-only, so `svelte-check` had been reporting two \
                    errors for as long as nobody ran it — one of them a dead \
                    comparison against a `LifecycleState` variant that does not \
                    exist, which made a restart-after-crash spin its full 30s \
                    timeout. Node is not a new dependency: `tauri.conf.json` \
                    already runs `npm run build` before every app build",
        program: "npm",
        args: &["--prefix", "crates/lumen-app/frontend", "run", "check"],
        optional: false,
    },
    Step {
        name: "check: fuzz targets",
        // Found by running this for the first time: `grammar_build` and
        // `grammar_x_output` had not compiled since `build_qwen35_tool_grammar_lark`
        // grew its `ToolCalls` argument. Two things hid it. The fuzz crate is
        // excluded from the workspace (its binaries need nightly plus sanitizer
        // flags), so every `--workspace` step above skips it; and `cargo fuzz
        // run <name>` passes `--bin <name>`, so soaking one target never builds
        // the other four. `cargo xtask fuzz --all` would have caught it, and
        // nothing runs that outside a release.
        //
        // Blocking, but skipped with a printed reason when nightly is absent —
        // see `nightly_available`. Reporting it as an ordinary optional FAILURE
        // would make "you have no nightly toolchain" look exactly like "the
        // targets are broken again", which is the signal this step exists to
        // give.
        rationale: "the fuzz crate is outside the workspace, so no --workspace step \
                    compiles it and soaking one target does not build the rest; two \
                    targets had been uncompilable for three commits",
        program: "cargo",
        args: &[
            "+nightly",
            "check",
            "--manifest-path",
            "fuzz/Cargo.toml",
            "--all-targets",
        ],
        optional: false,
    },
    Step {
        name: "test (representative)",
        rationale: "`cargo test --workspace` alone is green by omission — the \
                    interesting harnesses are feature-gated, and parallel threads \
                    share one Metal command buffer",
        program: "cargo",
        args: &["run", "-q", "-p", "xtask", "--", "test"],
        optional: false,
    },
    Step {
        name: "red-green",
        rationale: "every regression guard is re-proved against its own defect; a \
                    guard that passes with the fix reverted is worth nothing",
        program: "cargo",
        args: &["run", "-q", "-p", "xtask", "--", "red-green"],
        optional: false,
    },
    Step {
        name: "tiers",
        rationale: "an `#[ignore]` with no reason cannot be classified, scheduled \
                    or decided about — it is the shape that quietly becomes \
                    permanent, and one already had (a microbenchmark nobody could \
                    tell from a skipped assertion)",
        program: "cargo",
        args: &["run", "-q", "-p", "xtask", "--", "tiers"],
        optional: false,
    },
    Step {
        name: "flags --check",
        rationale: "docs/env-flags.md is generated; a stale copy is a lie about what \
                    the binary reads",
        program: "cargo",
        args: &["run", "-q", "-p", "xtask", "--", "flags", "--check"],
        optional: true,
    },
];

/// Grep-based hygiene rules. Each is `(what, pattern, allowed-path-substring)`;
/// a hit outside the allowance fails the gate.
///
/// These are policy from `docs/maintainer-workflow.md` §5 rather than
/// correctness, but they are the kind of policy that is only ever violated by
/// accident — and once pushed, a personal path in a committed file is in the
/// history for good.
/// Files that **define** the hygiene rules, and therefore have to contain the
/// patterns they forbid. Exempting them individually per rule went wrong twice
/// — first the docs that document the grep, then the gate source that runs it —
/// so it is one list with one reason.
///
/// A rule that fails on its own definition is a rule people learn to route
/// around, which is worse than not having it.
const RULE_DEFINITIONS: &[&str] = &[
    "docs/maintainer-workflow.md",
    "docs/release-checklist.md",
    "xtask/src/gate.rs",
    "LICENSE",
    ".ai/",
    "target/",
];

const HYGIENE: &[(&str, &str)] = &[
    (
        // §5 names this exactly: no `/Users/sonheesung` paths. Grepping for
        // `/Users/` generally would also flag `/Users/dev` in a deploy example
        // and `/Users/foo` in a UI string — placeholders that are the
        // DOCUMENTED replacement, so a rule that fails on them teaches people
        // to ignore the rule.
        "personal home paths",
        "/Users/sonheesung",
    ),
    ("personal email", "hsng95@gmail.com"),
];

pub fn main(args: Vec<String>) -> ExitCode {
    let quick = args.iter().any(|a| a == "--quick");
    let started = Instant::now();
    let mut failures: Vec<&str> = Vec::new();

    for step in STEPS {
        if quick && matches!(step.name, "test (representative)" | "red-green") {
            println!("── {} … SKIPPED (--quick)", step.name);
            continue;
        }
        // Derived from the args rather than a struct field: a step whose first
        // argument is `+nightly` needs nightly, and there is no way for the two
        // to disagree.
        if step.args.first() == Some(&"+nightly") && !nightly_available() {
            println!(
                "── {} … SKIPPED (no nightly toolchain — `rustup toolchain install nightly`)",
                step.name
            );
            continue;
        }
        print!("── {} … ", step.name);
        use std::io::Write;
        let _ = std::io::stdout().flush();

        let t = Instant::now();
        let status = Command::new(step.program).args(step.args).status();
        let elapsed = t.elapsed().as_secs_f64();

        match status {
            Ok(s) if s.success() => println!("ok ({elapsed:.1}s)"),
            Ok(_) if step.optional => {
                println!("FAILED ({elapsed:.1}s) — optional, continuing");
                eprintln!("   why it exists: {}", step.rationale);
            }
            Ok(_) => {
                println!("FAILED ({elapsed:.1}s)");
                eprintln!("   why it exists: {}", step.rationale);
                failures.push(step.name);
            }
            Err(e) => {
                println!("could not run: {e}");
                if !step.optional {
                    failures.push(step.name);
                }
            }
        }
    }

    print!("── hygiene … ");
    match hygiene_violations() {
        Ok(v) if v.is_empty() => println!("ok"),
        Ok(v) => {
            println!("FAILED");
            for line in &v {
                eprintln!("   {line}");
            }
            eprintln!(
                "   why it exists: docs/maintainer-workflow.md §5 — once pushed, a \
                 personal path is in the history for good"
            );
            failures.push("hygiene");
        }
        Err(e) => println!("could not run git grep: {e}"),
    }

    let total = started.elapsed().as_secs_f64();
    println!();
    if failures.is_empty() {
        println!("gate PASSED in {total:.1}s");
        if quick {
            println!("  (--quick skipped the test and red-green steps)");
        }
        println!(
            "  not covered here — these belong to a release, see docs/release-checklist.md:\n\
             \x20   fuzz soak, the full Optimization-flag matrix, coverage, Metal validation"
        );
        ExitCode::SUCCESS
    } else {
        println!("gate FAILED in {total:.1}s: {}", failures.join(", "));
        ExitCode::FAILURE
    }
}

/// Whether a nightly toolchain is installed.
///
/// The one gate step that needs it (`check: fuzz targets`) is blocking when it
/// can run and skipped-with-a-reason when it cannot, rather than optional. An
/// optional step reports its failure and continues, which would print the same
/// FAILED line for "the fuzz targets no longer compile" and for "this machine
/// has no nightly" — and the first of those is the reason the step exists.
fn nightly_available() -> bool {
    Command::new("cargo")
        .args(["+nightly", "--version"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// Run the hygiene greps over tracked files only — `git grep` rather than a
/// filesystem walk, so build output and gitignored personal state cannot
/// produce a false positive.
fn hygiene_violations() -> std::io::Result<Vec<String>> {
    let mut out = Vec::new();
    for (what, pattern) in HYGIENE {
        let res = Command::new("git")
            .args(["grep", "-nI", "--", pattern])
            .output()?;
        // `git grep` exits 1 with no matches, which is the good case.
        let text = String::from_utf8_lossy(&res.stdout);
        for line in text.lines() {
            let path = line.split(':').next().unwrap_or(line);
            if RULE_DEFINITIONS.iter().any(|a| path.contains(a)) {
                continue;
            }
            out.push(format!("{what}: {line}"));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every step must explain itself. A gate step whose failure message is
    /// just its name sends the reader hunting, and the usual outcome of that is
    /// the step getting deleted rather than the failure getting fixed.
    #[test]
    fn every_step_states_why_it_exists() {
        for s in STEPS {
            assert!(!s.name.trim().is_empty());
            assert!(
                s.rationale.len() > 30,
                "step {:?} needs a real rationale, got {:?}",
                s.name,
                s.rationale
            );
        }
        for (what, pattern) in HYGIENE {
            assert!(!what.trim().is_empty() && !pattern.trim().is_empty());
        }
        // The exemption list must cover this file: it necessarily contains
        // every pattern it forbids.
        assert!(
            RULE_DEFINITIONS.contains(&"xtask/src/gate.rs"),
            "the gate source defines the patterns, so it must be exempt or the \
             gate can never pass"
        );
    }

    /// Cheap steps must come before expensive ones, or the gate spends minutes
    /// before telling you about a formatting error.
    #[test]
    fn the_cheap_checks_run_first() {
        let names: Vec<&str> = STEPS.iter().map(|s| s.name).collect();
        let fmt = names.iter().position(|n| *n == "fmt").expect("fmt step");
        let test = names
            .iter()
            .position(|n| *n == "test (representative)")
            .expect("test step");
        let rg = names
            .iter()
            .position(|n| *n == "red-green")
            .expect("red-green step");
        assert!(fmt < test, "fmt is seconds; the suite is minutes");
        assert!(
            test < rg,
            "red-green re-runs guards, so it costs more than one pass"
        );
    }
}
