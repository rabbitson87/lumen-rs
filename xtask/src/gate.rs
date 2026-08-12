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
        name: "clippy",
        // REPORT-ONLY, deliberately. At `-D warnings` this workspace fails with
        // **330** pre-existing warnings (69 collapsible-if, 27 doc-indent, 18
        // very-complex-type, 13 dead-code, …) — it has never been clippy-clean.
        // Making the gate fail on day one would mean the gate gets skipped,
        // which is the same failure as a pre-push check that takes an hour.
        // Flip `optional` to false once the count reaches zero; the number is
        // the debt, and `cargo clippy --workspace --all-targets` prints it.
        rationale: "lints that catch real bugs, not just style — report-only \
                    until the 330-warning backlog is cleared, then make it \
                    blocking",
        program: "cargo",
        args: &[
            "clippy",
            "--workspace",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ],
        optional: true,
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
const HYGIENE: &[(&str, &str, &[&str])] = &[
    (
        // `docs/maintainer-workflow.md` §5 names this exactly: no
        // `/Users/sonheesung` paths. Grepping for `/Users/` generally would
        // also flag `/Users/dev` in a deploy example and `/Users/foo` in a UI
        // string — placeholders that are the DOCUMENTED replacement, so a rule
        // that fails on them teaches people to ignore the rule.
        "personal home paths",
        "/Users/sonheesung",
        // The workflow doc quotes the pattern in order to define it, and the
        // grep for it lives there too.
        &[
            "docs/maintainer-workflow.md",
            ".ai/",
            "target/",
            "xtask/src/gate.rs",
        ],
    ),
    (
        "personal email",
        "hsng95@gmail.com",
        &["LICENSE", "docs/maintainer-workflow.md", ".ai/"],
    ),
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

/// Run the hygiene greps over tracked files only — `git grep` rather than a
/// filesystem walk, so build output and gitignored personal state cannot
/// produce a false positive.
fn hygiene_violations() -> std::io::Result<Vec<String>> {
    let mut out = Vec::new();
    for (what, pattern, allowed) in HYGIENE {
        let res = Command::new("git")
            .args(["grep", "-nI", "--", pattern])
            .output()?;
        // `git grep` exits 1 with no matches, which is the good case.
        let text = String::from_utf8_lossy(&res.stdout);
        for line in text.lines() {
            let path = line.split(':').next().unwrap_or(line);
            if allowed.iter().any(|a| path.contains(a)) {
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
        for (what, pattern, _) in HYGIENE {
            assert!(!what.trim().is_empty() && !pattern.trim().is_empty());
        }
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
