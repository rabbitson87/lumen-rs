//! `cargo xtask fuzz` — drive the libFuzzer soak targets.
//!
//! ```text
//! cargo xtask fuzz --list                 # targets + what each one probes
//! cargo xtask fuzz tool_body_parse        # 60-second smoke soak
//! cargo xtask fuzz grammar_x_output --minutes 30
//! cargo xtask fuzz --all --minutes 2      # every target in sequence
//! ```
//!
//! Thin on purpose: the real work is `cargo +nightly fuzz run`, and this
//! wrapper exists so nobody has to remember the nightly toolchain, the
//! `-max_total_time` spelling, or which directory the corpus lives in. What it
//! adds beyond dispatch:
//!
//!   * **Preflight with instructions.** A missing `cargo-fuzz` or nightly
//!     toolchain fails with the install command, not a cargo error three
//!     layers deep.
//!   * **Crash triage that names the follow-up.** When a run ends with
//!     artifacts, the summary prints each new file and the two steps the
//!     methodology requires: promote the input into `fuzz/seeds/<target>/` and
//!     commit it so tier-0 replays it forever, and record the fix as a
//!     `red-green` defect. A crasher that stays in `fuzz/artifacts/` on one
//!     laptop is a bug report nobody filed.
//!
//! Directory split: `fuzz/seeds/` is the committed, permanent input set;
//! `fuzz/corpus/` is libFuzzer's local working state (gitignored — thousands
//! of coverage-minimized blobs, regrown from the seeds anywhere). Runs get
//! both directories, so seeds feed every soak without polluting the repo with
//! generated state.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

/// Every target under `fuzz/fuzz_targets/`, with the one-line purpose shown by
/// `--list`. Kept in sync with the directory by
/// `target_table_matches_the_fuzz_targets_directory` — a typo'd name prints
/// this table rather than a bare cargo error.
const TARGETS: &[(&str, &str)] = &[
    (
        "tool_body_parse",
        "raw bytes → parse_tool_call_body; opener-boundary + JSON round-trip invariants",
    ),
    (
        "args_to_json",
        "raw bytes → gemma4_args_to_json; Ok must be genuinely valid JSON",
    ),
    (
        "grammar_build",
        "ToolSet → build_qwen35_tool_grammar_lark; hostile schema alphabet, no panics",
    ),
    (
        "grammar_x_output",
        "ToolSet + ModelOutput mutated together; the dbsqlfuzz cross-invariant",
    ),
    (
        "request_parse",
        "raw bytes → the five HTTP request deserializers; a panic is a remote crash",
    ),
];

pub fn main(args: Vec<String>) -> ExitCode {
    let mut minutes: f64 = 1.0;
    let mut all = false;
    let mut names: Vec<String> = Vec::new();

    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--list" => {
                println!("fuzz targets (committed seeds under fuzz/seeds/<name>/):\n");
                for (name, what) in TARGETS {
                    println!("  {name:<18} {what}");
                }
                return ExitCode::SUCCESS;
            }
            "--all" => all = true,
            "--minutes" => {
                let Some(v) = it.next().and_then(|v| v.parse::<f64>().ok()) else {
                    eprintln!("--minutes needs a number");
                    return ExitCode::from(2);
                };
                if v <= 0.0 {
                    eprintln!("--minutes must be positive");
                    return ExitCode::from(2);
                }
                minutes = v;
            }
            other if other.starts_with('-') => {
                eprintln!("unknown flag {other:?}");
                return ExitCode::from(2);
            }
            other => names.push(other.to_string()),
        }
    }

    if all {
        names = TARGETS.iter().map(|(n, _)| (*n).to_string()).collect();
    }
    if names.is_empty() {
        eprintln!("usage: cargo xtask fuzz <TARGET…|--all> [--minutes N] | --list");
        return ExitCode::from(2);
    }
    for n in &names {
        if !TARGETS.iter().any(|(t, _)| t == n) {
            eprintln!("unknown fuzz target {n:?}; known targets:\n");
            for (name, what) in TARGETS {
                eprintln!("  {name:<18} {what}");
            }
            return ExitCode::from(2);
        }
    }

    if let Err(msg) = preflight() {
        eprintln!("{msg}");
        return ExitCode::from(1);
    }

    let root = repo_root();
    let secs = (minutes * 60.0).round() as u64;
    let mut failed: Vec<String> = Vec::new();

    for name in &names {
        println!("\n=== fuzz {name} ({minutes} min) ===");
        let before = artifact_files(&root, name);

        // `cargo fuzz run` exits nonzero on a crash — that is the tool working,
        // so it feeds triage rather than aborting the sweep.
        let status = Command::new("cargo")
            .current_dir(&root)
            .args([
                "+nightly",
                "fuzz",
                "run",
                name,
                // Working corpus first (where libFuzzer writes new coverage),
                // committed seeds second (read-only input to every run).
                &format!("fuzz/corpus/{name}"),
                &format!("fuzz/seeds/{name}"),
                "--",
                &format!("-max_total_time={secs}"),
            ])
            .status();

        let ok = matches!(status, Ok(s) if s.success());
        let after = artifact_files(&root, name);
        let new: Vec<_> = after.iter().filter(|f| !before.contains(*f)).collect();

        if !new.is_empty() {
            println!("\n{name}: {} new crash artifact(s):", new.len());
            for f in &new {
                println!("  {}", f.display());
            }
            println!(
                "  next: 1) cp the input into fuzz/seeds/{name}/ and commit it — tier 0 then \
                 replays it forever;\n        2) fix, and record the defect in \
                 xtask/src/red_green.rs so the fix is proven red→green."
            );
            failed.push(name.clone());
        } else if !ok {
            println!("{name}: run failed without producing an artifact (see cargo output above)");
            failed.push(name.clone());
        } else {
            println!("{name}: clean for {minutes} min");
        }
    }

    println!("\n==============================================================================");
    if failed.is_empty() {
        println!("all {} target(s) clean", names.len());
        ExitCode::SUCCESS
    } else {
        println!(
            "{} of {} target(s) need attention: {failed:?}",
            failed.len(),
            names.len()
        );
        ExitCode::FAILURE
    }
}

/// Fail fast, with the command that fixes it — not a cargo error three layers
/// deep into a toolchain the caller didn't know was required.
fn preflight() -> Result<(), String> {
    let nightly = Command::new("rustup")
        .args(["run", "nightly", "rustc", "--version"])
        .output();
    if !matches!(nightly, Ok(o) if o.status.success()) {
        return Err(
            "nightly toolchain missing — install with: rustup toolchain install nightly"
                .to_string(),
        );
    }
    let fuzz = Command::new("cargo").args(["fuzz", "--version"]).output();
    if !matches!(fuzz, Ok(o) if o.status.success()) {
        return Err("cargo-fuzz missing — install with: cargo install cargo-fuzz".to_string());
    }
    Ok(())
}

fn repo_root() -> PathBuf {
    // xtask always runs from a checkout; CARGO_MANIFEST_DIR is xtask/.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask sits one level below the repo root")
        .to_path_buf()
}

fn artifact_files(root: &Path, target: &str) -> Vec<PathBuf> {
    let dir = root.join("fuzz").join("artifacts").join(target);
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    entries.filter_map(|e| e.ok().map(|e| e.path())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `--list` table is the interface; a target added to `fuzz/Cargo.toml`
    /// but not here would be invisible, and one removed there but not here
    /// would dispatch into a cargo error. The filesystem is the referee.
    #[test]
    fn target_table_matches_the_fuzz_targets_directory() {
        let dir = repo_root().join("fuzz").join("fuzz_targets");
        let mut on_disk: Vec<String> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
            .filter_map(|e| {
                let name = e.ok()?.file_name().into_string().ok()?;
                name.strip_suffix(".rs").map(str::to_owned)
            })
            .collect();
        on_disk.sort();
        let mut listed: Vec<String> = TARGETS.iter().map(|(n, _)| (*n).to_string()).collect();
        listed.sort();
        assert_eq!(
            listed, on_disk,
            "xtask::fuzz::TARGETS and fuzz/fuzz_targets/ disagree — update whichever is stale"
        );
    }

    /// Every target ships a non-empty committed seed set: a soak that starts
    /// from zero bytes spends its first hour rediscovering the input format,
    /// and the tier-0 replay would silently cover nothing.
    #[test]
    fn every_target_has_a_seed_corpus() {
        for (name, _) in TARGETS {
            let dir = repo_root().join("fuzz").join("seeds").join(name);
            let n = std::fs::read_dir(&dir)
                .map(|d| d.filter_map(Result::ok).count())
                .unwrap_or(0);
            assert!(
                n > 0,
                "fuzz/seeds/{name}/ is empty — commit at least one seed input"
            );
        }
    }
}
