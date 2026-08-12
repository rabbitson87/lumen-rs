//! `cargo xtask flags` — the env-flag inventory and the disabled-optimization
//! equivalence matrix (005 Phase 2).
//!
//! ```text
//! cargo xtask flags --list            # print the registry
//! cargo xtask flags --docs           # regenerate docs/env-flags.md
//! cargo xtask flags --check          # fail if docs/env-flags.md is stale, or
//!                                    #   if the registry disagrees with source
//! cargo xtask flags                  # equivalence: suite once with EVERY
//!                                    #   Optimization flag flipped
//! cargo xtask flags --one-at-a-time  # bisect: one suite run per flag
//! ```
//!
//! The equivalence matrix is SQLite's disabled-optimization testing: run the
//! suite with an optimization off (or a default-off one on) and require the
//! same results, because an `Optimization`-kind flag promises output identity
//! and the promise is only worth something while both sides stay green.
//! `Behavior` flags (e.g. `LUMEN_MLX_KV_BF16`, which intentionally changes
//! numerics) and `Diagnostic` flags are never flipped.
//!
//! The default mode flips everything at once — one extra suite run, and any
//! failure proves *some* flag's alternate path is broken. `--one-at-a-time`
//! then names it. This mirrors how the plan intended bisection to work rather
//! than paying N suite runs on every invocation.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const DOCS_PATH: &str = "docs/env-flags.md";

#[derive(Debug)]
struct Flag {
    env: String,
    default: bool,
    kind: String,
    declared_in: String,
    doc: String,
}

pub fn main(args: Vec<String>) -> ExitCode {
    let mut list = false;
    let mut docs = false;
    let mut check = false;
    let mut one_at_a_time = false;
    for a in &args {
        match a.as_str() {
            "--list" => list = true,
            "--docs" => docs = true,
            "--check" => check = true,
            "--one-at-a-time" => one_at_a_time = true,
            other => {
                eprintln!(
                    "unknown flag {other:?}\nusage: cargo xtask flags [--list|--docs|--check|--one-at-a-time]"
                );
                return ExitCode::from(2);
            }
        }
    }

    let flags = match dump_registry() {
        Ok(f) => f,
        Err(e) => {
            eprintln!("failed to dump the flag registry: {e}");
            return ExitCode::FAILURE;
        }
    };

    if list {
        println!("{} registered flags:\n", flags.len());
        for f in &flags {
            println!(
                "  {:<40} default={:<5} {:<12} ({})",
                f.env, f.default, f.kind, f.declared_in
            );
        }
        return ExitCode::SUCCESS;
    }

    if docs {
        let rendered = render_docs(&flags);
        if let Err(e) = std::fs::write(repo_root().join(DOCS_PATH), rendered) {
            eprintln!("write {DOCS_PATH}: {e}");
            return ExitCode::FAILURE;
        }
        println!("wrote {DOCS_PATH} ({} flags)", flags.len());
        return ExitCode::SUCCESS;
    }

    if check {
        return run_check(&flags);
    }

    run_equivalence(&flags, one_at_a_time)
}

/// Registry via the dump example. Debug build — the registry is static data
/// and a release build of lumen-mlx takes minutes for no benefit here.
fn dump_registry() -> Result<Vec<Flag>, String> {
    let out = Command::new("cargo")
        .current_dir(repo_root())
        .args([
            "run",
            "-p",
            "lumen-mlx",
            "--features",
            "mlx-native",
            "--example",
            "dump_flags",
        ])
        .output()
        .map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(format!(
            "dump_flags failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let mut flags = Vec::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let v: serde_json::Value =
            serde_json::from_str(line).map_err(|e| format!("bad dump line {line:?}: {e}"))?;
        flags.push(Flag {
            env: v["env"].as_str().unwrap_or_default().to_string(),
            default: v["default"].as_bool().unwrap_or_default(),
            kind: v["kind"].as_str().unwrap_or_default().to_string(),
            declared_in: v["declared_in"].as_str().unwrap_or_default().to_string(),
            doc: v["doc"].as_str().unwrap_or_default().to_string(),
        });
    }
    if flags.is_empty() {
        return Err(
            "registry dumped empty — the dump example's link anchor is broken (see \
             examples/dump_flags.rs)"
                .into(),
        );
    }
    Ok(flags)
}

/// Two staleness checks, both of which exist because of an observed failure
/// mode rather than caution:
///  1. docs/env-flags.md must match a fresh render (the 370-vs-27 audit gap).
///  2. every `env: "LUMEN_…"` string in source must appear in the linked
///     registry — the dump depends on the linker pulling the declaring rlib,
///     which an anchor forces today but no contract guarantees.
fn run_check(flags: &[Flag]) -> ExitCode {
    let mut failed = false;

    let committed = std::fs::read_to_string(repo_root().join(DOCS_PATH)).unwrap_or_default();
    if committed != render_docs(flags) {
        eprintln!("{DOCS_PATH} is stale — regenerate with: cargo xtask flags --docs");
        failed = true;
    }

    match grep_declared_envs() {
        Ok(declared) => {
            let registered: std::collections::HashSet<&str> =
                flags.iter().map(|f| f.env.as_str()).collect();
            for env in &declared {
                if !registered.contains(env.as_str()) {
                    eprintln!(
                        "flag {env} is declared in source but missing from the linked registry — \
                         linker dropped it, or it is feature-gated off in the dump build"
                    );
                    failed = true;
                }
            }
            println!(
                "registry {} flags / source {} declarations",
                flags.len(),
                declared.len()
            );
        }
        Err(e) => {
            eprintln!("source grep failed: {e}");
            failed = true;
        }
    }

    if failed {
        ExitCode::FAILURE
    } else {
        println!("flags check clean");
        ExitCode::SUCCESS
    }
}

/// `env: "…"` occurrences inside `flag!` invocations, straight from source.
fn grep_declared_envs() -> Result<Vec<String>, String> {
    // `-n` and the filename, because a declaration inside a `#[cfg(test)] mod`
    // is not production surface: the dump binary is built without `cfg(test)`,
    // so those flags legitimately never reach the registry. Counting them made
    // `--check` fail on lumen-flags' own two test flags — the same mistake the
    // coverage tool made with assertions in inline test modules, which is worth
    // noting as a pattern: test-only code keeps getting counted as API.
    let out = Command::new("git")
        .current_dir(repo_root())
        .args(["grep", "-n", "env: \"", "--", "crates/*/src/*.rs"])
        .output()
        .map_err(|e| e.to_string())?;
    let cfg_test = Command::new("git")
        .current_dir(repo_root())
        .args(["grep", "-n", "#\\[cfg(test)\\]", "--", "crates/*/src/*.rs"])
        .output()
        .map_err(|e| e.to_string())?;

    // file -> first `#[cfg(test)]` line, if any.
    let mut test_mod_at: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    for line in String::from_utf8_lossy(&cfg_test.stdout).lines() {
        let mut parts = line.splitn(3, ':');
        let (Some(file), Some(no)) = (parts.next(), parts.next()) else {
            continue;
        };
        let Ok(no) = no.parse::<usize>() else {
            continue;
        };
        test_mod_at
            .entry(file.to_string())
            .and_modify(|e| *e = (*e).min(no))
            .or_insert(no);
    }

    let mut envs = Vec::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let mut parts = line.splitn(3, ':');
        let (Some(file), Some(no), Some(text)) = (parts.next(), parts.next(), parts.next()) else {
            continue;
        };
        let Ok(no) = no.parse::<usize>() else {
            continue;
        };
        if test_mod_at.get(file).is_some_and(|cut| no >= *cut) {
            continue;
        }
        if let Some(rest) = text.trim().strip_prefix("env: \"")
            && let Some(env) = rest.split('"').next()
        {
            envs.push(env.to_string());
        }
    }
    envs.sort();
    envs.dedup();
    Ok(envs)
}

fn render_docs(flags: &[Flag]) -> String {
    let mut s = String::new();
    s.push_str(
        "# Env flags\n\n\
         GENERATED — do not edit. Regenerate with `cargo xtask flags --docs`;\n\
         `cargo xtask flags --check` fails when this file is stale.\n\n\
         Source of truth is the `lumen_flags::flag!` declaration next to the code\n\
         each flag gates; this file is its projection. Parse rule for every flag:\n\
         unset → default, `\"0\"` → off, any other value → on.\n\n\
         | Env | Default | Kind | Declared in |\n|---|---|---|---|\n",
    );
    for f in flags {
        s.push_str(&format!(
            "| `{}` | {} | {} | `{}` |\n",
            f.env,
            if f.default { "on" } else { "off" },
            f.kind,
            f.declared_in
        ));
    }
    s.push_str("\n## Details\n");
    for f in flags {
        s.push_str(&format!(
            "\n### `{}`\n\n*{}, default {}.*\n\n{}\n",
            f.env,
            f.kind,
            if f.default { "on" } else { "off" },
            f.doc.trim()
        ));
    }
    s
}

/// The matrix itself. Flips only `Optimization` flags; a `Behavior` flag in
/// the flip set would fail the suite by design and teach people to ignore the
/// matrix.
fn run_equivalence(flags: &[Flag], one_at_a_time: bool) -> ExitCode {
    let opt: Vec<&Flag> = flags.iter().filter(|f| f.kind == "Optimization").collect();
    if opt.is_empty() {
        eprintln!("no Optimization flags registered — nothing to flip");
        return ExitCode::FAILURE;
    }

    let flipped = |f: &Flag| if f.default { "0" } else { "1" };

    if !one_at_a_time {
        println!(
            "equivalence: one suite run with all {} Optimization flags flipped:",
            opt.len()
        );
        for f in &opt {
            println!("  {}={}", f.env, flipped(f));
        }
        let mut cmd = Command::new("cargo");
        cmd.current_dir(repo_root()).args(["xtask", "test"]);
        for f in &opt {
            cmd.env(&f.env, flipped(f));
        }
        let ok = matches!(cmd.status(), Ok(s) if s.success());
        return if ok {
            println!("\nequivalence PASS: the suite is green with every Optimization flag flipped");
            ExitCode::SUCCESS
        } else {
            eprintln!(
                "\nequivalence FAIL: some flag's alternate path breaks the suite.\n\
                 Bisect with: cargo xtask flags --one-at-a-time"
            );
            ExitCode::FAILURE
        };
    }

    let mut broken: Vec<&str> = Vec::new();
    for f in &opt {
        println!("\n=== {}={} (default {}) ===", f.env, flipped(f), f.default);
        let ok = matches!(
            Command::new("cargo")
                .current_dir(repo_root())
                .args(["xtask", "test"])
                .env(&f.env, flipped(f))
                .status(),
            Ok(s) if s.success()
        );
        println!("{}: {}", f.env, if ok { "PASS" } else { "FAIL" });
        if !ok {
            broken.push(&f.env);
        }
    }
    println!("\n==============================================================================");
    if broken.is_empty() {
        println!("all {} Optimization flags pass individually", opt.len());
        ExitCode::SUCCESS
    } else {
        eprintln!("broken alternate paths: {broken:?}");
        ExitCode::FAILURE
    }
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask sits one level below the repo root")
        .to_path_buf()
}
