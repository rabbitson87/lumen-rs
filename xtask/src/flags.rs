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

/// Three staleness checks, each of which exists because of an observed failure
/// mode rather than caution:
///  1. docs/env-flags.md must match a fresh render (the 370-vs-27 audit gap).
///  2. every `env: "LUMEN_…"` string in source must appear in the linked
///     registry — the dump depends on the linker pulling the declaring rlib,
///     which an anchor forces today but no contract guarantees.
///  3. every env var **named in any committed doc** must exist in the source.
///     The project was renamed and `KESTREL_*` became `LUMEN_*`, but README.md
///     and getting-started.md kept telling readers to set the old names — three
///     variables, in the two most-read files, that had done nothing for as long
///     as the rename was old. Checks 1 and 2 could not see it: both start from
///     the `flag!` registry, and these are hand-rolled `env::var` reads. A doc
///     is the only place a dead variable can hide in plain sight, because
///     setting it fails silently by construction.
fn run_check(flags: &[Flag]) -> ExitCode {
    let mut failed = false;

    match check_documented_envs_exist() {
        Ok(dead) if dead.is_empty() => {}
        Ok(dead) => {
            for (file, line, var) in &dead {
                eprintln!(
                    "{file}:{line} documents {var}, which appears nowhere in the source — \
                     a reader who sets it gets silence"
                );
            }
            failed = true;
        }
        Err(e) => {
            eprintln!("documented-env scan failed: {e}");
            failed = true;
        }
    }

    let committed = std::fs::read_to_string(repo_root().join(DOCS_PATH)).unwrap_or_default();
    if committed != render_docs(flags) {
        eprintln!("{DOCS_PATH} is stale — regenerate with: cargo xtask flags --docs");
        failed = true;
    }

    match audit_env_vars(flags) {
        Ok((all, unmanaged)) => {
            println!(
                "env vars: {} read in source / {} registered / {} unmanaged (baseline {UNMANAGED_BASELINE})",
                all.len(),
                flags.len(),
                unmanaged.len()
            );
            if unmanaged.len() > UNMANAGED_BASELINE {
                eprintln!(
                    "unmanaged flag count rose to {} (baseline {UNMANAGED_BASELINE}). A new \
                     hand-rolled env::var is invisible to docs, to `--one-at-a-time`, and — if \
                     it gates an optimization — to the equivalence matrix. Register it, or \
                     raise the baseline deliberately.",
                    unmanaged.len()
                );
                for v in unmanaged.iter().skip(UNMANAGED_BASELINE) {
                    eprintln!("    {v}");
                }
                failed = true;
            }
        }
        Err(e) => {
            eprintln!("env audit failed: {e}");
            failed = true;
        }
    }

    match audit_env_vars(flags) {
        Ok((all, _)) => {
            let registered: std::collections::HashSet<&str> =
                flags.iter().map(|f| f.env.as_str()).collect();
            // A registry flag read ALSO via raw `env::var` is worse than an
            // unregistered one. The raw site ignores `with()` / `set()`, so a
            // harness that flips the flag registry-side flips it everywhere
            // except there — and the two sites can drift apart on parsing, which
            // is exactly what happened: `LUMEN_NATIVE_TIMING` kept the old
            // `1|true|TRUE|yes` rule at its raw site after the registry moved to
            // "any non-`0`", so `=on` meant true one place and false the other.
            for v in all.iter().filter(|v| registered.contains(v.as_str())) {
                eprintln!(
                    "{v} is a registry flag AND is read directly with env::var. The raw \
                     read bypasses with()/set() and can drift on parsing — call the \
                     flag's accessor instead."
                );
                failed = true;
            }
        }
        Err(e) => {
            eprintln!("env audit failed: {e}");
            failed = true;
        }
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

/// Env vars named in committed docs that do not exist anywhere in the source.
///
/// Returns `(file, line, var)`. The match is deliberately broad — any
/// `SCREAMING_SNAKE` token with a known project prefix — because the failure
/// this catches is a *renamed* prefix, and a scanner that only knew the current
/// prefix would have been blind to exactly the case that motivated it.
fn check_documented_envs_exist() -> Result<Vec<(String, usize, String)>, String> {
    // The file that defines the rule has to be allowed to name a dead variable,
    // same reasoning as `RULE_DEFINITIONS` in gate.rs.
    const RULE_DEFINITIONS: &[&str] = &["xtask/src/flags.rs", "docs/env-flags.md"];
    // Prefixes this project has shipped under. `KESTREL_` is dead and stays in
    // the list on purpose: dropping it is how the check would stop catching the
    // next rename.
    const PREFIXES: &[&str] = &["LUMEN_", "KESTREL_"];

    // `*.md` AND the app's env schema + locale strings. Scanning only markdown
    // missed a dead variable that mattered more than any of the doc ones:
    // `LUMEN_BATCHED_PREFILL_CHUNK` was a **number slider in the app's Advanced
    // settings**, min 128 / max 8192 / default 512, whose reader went with the
    // Candle backend in `7eacd3a`. A doc that lies costs a reader a minute; a
    // UI control that lies is a switch the user watches do nothing.
    let docs = Command::new("git")
        .current_dir(repo_root())
        .args([
            "ls-files",
            "*.md",
            "crates/lumen-app/frontend/src/lib/env-schema.ts",
            "crates/lumen-app/frontend/src/messages/*.ts",
        ])
        .output()
        .map_err(|e| e.to_string())?;

    let mut dead = Vec::new();
    for file in String::from_utf8_lossy(&docs.stdout).lines() {
        if RULE_DEFINITIONS.iter().any(|d| file.ends_with(d)) {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(repo_root().join(file)) else {
            continue;
        };
        for (i, line) in text.lines().enumerate() {
            for var in env_tokens(line, PREFIXES) {
                if !source_mentions(&var)? {
                    dead.push((file.to_string(), i + 1, var));
                }
            }
        }
    }
    Ok(dead)
}

/// `SCREAMING_SNAKE` tokens on `line` carrying one of `prefixes`.
fn env_tokens(line: &str, prefixes: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    let bytes: Vec<char> = line.chars().collect();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_uppercase() {
            let start = i;
            while i < bytes.len()
                && (bytes[i].is_ascii_uppercase() || bytes[i].is_ascii_digit() || bytes[i] == '_')
            {
                i += 1;
            }
            let tok: String = bytes[start..i].iter().collect();
            if prefixes.iter().any(|p| tok.starts_with(p)) && tok.len() > 8 && !out.contains(&tok) {
                out.push(tok);
            }
        } else {
            i += 1;
        }
    }
    out
}

/// Does `var` appear anywhere under `crates/` or `xtask/`?
fn source_mentions(var: &str) -> Result<bool, String> {
    let out = Command::new("git")
        .current_dir(repo_root())
        .args([
            "grep",
            "-l",
            "--fixed-strings",
            var,
            "--",
            // Backend crates + xtask only. NOT `crates/` wholesale: that would
            // include the frontend files this check now scans, so a variable
            // named only in `env-schema.ts` would find itself and pass — a
            // check that reports clean because it is looking in a mirror.
            "crates/lumen-core",
            "crates/lumen-mlx",
            "crates/lumen-server",
            "crates/lumen-diffusion",
            "crates/lumen-app/src",
            "crates/turboquant-cache",
            "xtask/",
        ])
        .output()
        .map_err(|e| e.to_string())?;
    Ok(!out.stdout.is_empty())
}

/// The unmanaged-flag ratchet.
///
/// Hand-rolled `env::var` reads are:
/// undocumented in the generated `docs/env-flags.md`, invisible to
/// `--one-at-a-time`, and — for any that gate an optimization — never
/// both-path tested. `causal-mask-builders-agree` and
/// `rotating-cache-both-paths` in `red_green.rs` are both that defect: an
/// alternate path no test could reach.
///
/// Migrating 190 at once is not the answer; most are one-shot diagnostics that
/// would only add noise to the registry. So this is a **ratchet**, the same
/// shape as the one-sided branch count in `cargo xtask coverage`: the number is
/// allowed to fall and not to rise. Lower it when you migrate one.
///
/// Raising it is a decision, not a fix. If a new flag gates an optimization,
/// it belongs in the registry so the equivalence matrix covers it.
/// 202 distinct `LUMEN_*` names are read via `env::var` in library source and
/// 12 live in the registry, but the two sets are **nearly disjoint** rather
/// than nested: a registry flag is read through the `flag!`-generated `get()`,
/// so it does not appear as a literal `env::var("LUMEN_X")`. 201 of the 202 raw
/// reads are unregistered. (I first wrote 190 here by subtracting one from the
/// other, which is the arithmetic of a subset and these are not.)
///
/// 201 -> 193 when the eight Gemma 4 fusion flags moved into the registry,
/// 193 -> 192 with `LUMEN_MLX_NO_OVERLAP`, 192 -> 190 with
/// `LUMEN_NATIVE_FUSE_GATE_UP` and `LUMEN_NATIVE_CACHED_STREAM`.
const UNMANAGED_BASELINE: usize = 190;

/// Every `LUMEN_*` read via `env::var` in library source, and whether the
/// registry knows about it.
fn audit_env_vars(flags: &[Flag]) -> Result<(Vec<String>, Vec<String>), String> {
    let out = Command::new("git")
        .current_dir(repo_root())
        .args([
            "grep",
            "-oh",
            "-E",
            r#"env::var(_os)?\("LUMEN_[A-Z0-9_]*"\)"#,
            "--",
            "crates/lumen-core/src",
            "crates/lumen-mlx/src",
            "crates/lumen-server/src",
            "crates/lumen-diffusion/src",
            "crates/lumen-app/src",
        ])
        .output()
        .map_err(|e| e.to_string())?;
    let registered: std::collections::HashSet<&str> =
        flags.iter().map(|f| f.env.as_str()).collect();
    let mut all: Vec<String> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.split('"').nth(1).map(str::to_string))
        .collect();
    all.sort();
    all.dedup();
    let unmanaged: Vec<String> = all
        .iter()
        .filter(|v| !registered.contains(v.as_str()))
        .cloned()
        .collect();
    Ok((all, unmanaged))
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The token scanner has to find a variable wherever a doc happens to spell
    /// it — bare, in backticks, in a table cell, with a value attached.
    #[test]
    fn env_tokens_finds_variables_in_the_shapes_docs_actually_use() {
        let cases = [
            (
                "| `LUMEN_GEMMA4_PREFILL_SYNC=0` | on | skip |",
                "LUMEN_GEMMA4_PREFILL_SYNC",
            ),
            (
                "set KESTREL_GEMMA4_CUSTOM_FLASH_ATTN before running",
                "KESTREL_GEMMA4_CUSTOM_FLASH_ATTN",
            ),
            ("unset `LUMEN_AFFINE8_NAIVE`.", "LUMEN_AFFINE8_NAIVE"),
        ];
        for (line, want) in cases {
            let got = env_tokens(line, &["LUMEN_", "KESTREL_"]);
            assert!(got.iter().any(|t| t == want), "{line:?} -> {got:?}");
        }
    }

    /// It must not fire on ordinary prose. A scanner that flags `MODEL_ID` or a
    /// shouted word costs more attention than it saves, and a check people
    /// learn to ignore is the failure mode this whole command exists to avoid.
    #[test]
    fn env_tokens_ignores_everything_that_is_not_a_project_variable() {
        for line in [
            "Set MODEL_ID to a local directory.",
            "This is IMPORTANT and MUST be read.",
            "HTTP GET /v1/models returns JSON.",
            "See LUMEN_ for the prefix convention.", // bare prefix names nothing
        ] {
            assert!(
                env_tokens(line, &["LUMEN_", "KESTREL_"]).is_empty(),
                "false positive on {line:?}"
            );
        }
    }

    /// `KESTREL_` is dead, and stays in the prefix list on purpose. Dropping a
    /// retired prefix is exactly how the check would go quiet on the next
    /// rename — the case that motivated it in the first place.
    #[test]
    fn the_retired_prefix_is_still_scanned() {
        assert!(
            !env_tokens(
                "`KESTREL_GEMMA4_PER_STEP_LATENCY=1`",
                &["LUMEN_", "KESTREL_"]
            )
            .is_empty(),
            "a retired prefix must still be recognised, or renames stop being caught"
        );
    }
}
