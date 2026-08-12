//! Runs the workspace test suite the only way it is reliable on this machine.
//!
//! Two things make a plain `cargo test --workspace` unrepresentative:
//!
//! 1. **Feature gates.** The interesting harnesses live behind `mlx-native`.
//!    Without it they compile to zero tests and the run is green by omission.
//! 2. **Parallel test threads share one Metal command buffer.** libtest's
//!    default is one thread per core. Two tests encoding at once trips
//!    `A command encoder is already encoding to this command buffer` and takes
//!    the whole process down with SIGABRT — intermittently, so a green run
//!    proves nothing about the next one. `--test-threads=1` removes the
//!    contention; the GPU work here is dominated by compilation anyway.
//!
//! Release is not an optimization preference: several GPU harnesses take
//! minutes unoptimized and seconds otherwise.
//!
//! `lumen-metal/model-integration` used to be in `FEATURES` alongside
//! `mlx-native`; that crate went with the Candle backend.
//!
//! ## `--validate`
//!
//! This project's answer to Valgrind, and the reason it exists is that the
//! usual answer does not apply: `llvm-cov` and Miri both stop at the FFI
//! boundary, and there are ~300 `unsafe` sites past it handing raw buffers to
//! Metal. An out-of-bounds binding there is not a crash — it reads whatever is
//! adjacent in a shared heap and produces plausible, wrong numbers, which is
//! the failure class this whole task exists to make visible.
//!
//! Metal's own validation layers do catch it, so `--validate` turns them on:
//!
//! * `MTL_DEBUG_LAYER=1` — API-level checks (encoder state, resource lifetime,
//!   argument types).
//! * `MTL_SHADER_VALIDATION=1` — in-shader bounds checking on buffer and
//!   texture access, which is the half that catches a wrong offset.
//!
//! Both cost real time, so they are opt-in rather than always-on. A run with
//! them enabled is slower by roughly an order of magnitude on the GPU
//! harnesses; a run without them says nothing about buffer bounds.

use std::process::{Command, ExitCode};

const FEATURES: &str = "lumen-mlx/mlx-native";

/// Metal's validation layers, and what each one buys.
const VALIDATION_ENV: &[(&str, &str)] = &[
    ("MTL_DEBUG_LAYER", "1"),
    ("MTL_SHADER_VALIDATION", "1"),
    // Report rather than only abort, so a run surfaces every finding instead of
    // the first one.
    ("MTL_DEBUG_LAYER_ERROR_MODE", "assert"),
];

/// Where `lumen_testkit::cases` accumulates. Truncated before the run so the
/// count reflects this run and not every run since the last `cargo clean`.
fn cases_file() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|r| r.join("target").join("effective-cases.jsonl"))
        .unwrap_or_else(|| std::path::PathBuf::from("target/effective-cases.jsonl"))
}

/// Sum of the `n` fields written during the run, with the largest contributors.
///
/// This is the metric that replaces SQLite's 590:1 test-LOC ratio. libtest
/// reports a sweep over 3,400 budgets as `1 passed`, which is exactly as
/// informative as counting test files; the number below is what was actually
/// executed. Only the parameterized sweeps report, so it is a floor rather than
/// a total — a floor that can only be raised by writing generators, which is
/// the mechanism worth copying.
fn report_effective_cases() {
    let Ok(text) = std::fs::read_to_string(cases_file()) else {
        return;
    };
    let mut total = 0usize;
    let mut rows: Vec<(usize, String)> = Vec::new();
    for line in text.lines() {
        let Some(n) = line
            .split("\"n\":")
            .nth(1)
            .and_then(|r| r.split(&[',', '}'][..]).next())
            .and_then(|n| n.trim().parse::<usize>().ok())
        else {
            continue;
        };
        let what = line
            .split("\"what\":\"")
            .nth(1)
            .and_then(|r| r.split('"').next())
            .unwrap_or("?")
            .to_string();
        total += n;
        rows.push((n, what));
    }
    if rows.is_empty() {
        return;
    }
    rows.sort_by(|a, b| b.0.cmp(&a.0));
    rows.dedup_by(|a, b| a.1 == b.1);
    eprintln!("\neffective cases executed (parameterized sweeps only): {total}");
    for (n, what) in rows.iter().take(6) {
        eprintln!("  {n:>7}  {what}");
    }
}

pub fn main(args: Vec<String>) -> ExitCode {
    let validate = args.iter().any(|a| a == "--validate");
    let _ = std::fs::remove_file(cases_file());
    let args: Vec<String> = args.into_iter().filter(|a| a != "--validate").collect();

    let mut cmd = Command::new("cargo");
    cmd.args(["test", "--workspace", "--features", FEATURES, "--release"]);
    // Anything before `--` is a cargo argument (`-p`, `--test`, a filter);
    // libtest options go after it, alongside the thread pin.
    let (cargo_args, test_args) = match args.iter().position(|a| a == "--") {
        Some(i) => (&args[..i], &args[i + 1..]),
        None => (&args[..], &args[args.len()..]),
    };
    cmd.args(cargo_args);
    cmd.arg("--").arg("--test-threads=1").args(test_args);

    if validate {
        for (k, v) in VALIDATION_ENV {
            cmd.env(k, v);
        }
        eprintln!(
            "Metal validation ON: {}",
            VALIDATION_ENV
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
        eprintln!(
            "  expect this to be substantially slower — shader validation bounds-checks \
             every buffer access."
        );
    }

    eprintln!(
        "running: cargo test --workspace --features {FEATURES} --release{} -- \
         --test-threads=1{}",
        cargo_args
            .iter()
            .map(|a| format!(" {a}"))
            .collect::<String>(),
        test_args
            .iter()
            .map(|a| format!(" {a}"))
            .collect::<String>(),
    );

    match cmd.status() {
        Ok(s) if s.success() => {
            report_effective_cases();
            ExitCode::SUCCESS
        }
        Ok(_) => {
            report_effective_cases();
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("could not spawn cargo: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The validation layers are only useful if they are actually the ones
    /// Metal reads. A typo here produces a run that looks validated and is not,
    /// which is worse than not offering the flag.
    #[test]
    fn the_validation_variables_are_the_ones_metal_reads() {
        let names: Vec<&str> = VALIDATION_ENV.iter().map(|(k, _)| *k).collect();
        assert!(names.contains(&"MTL_DEBUG_LAYER"));
        assert!(
            names.contains(&"MTL_SHADER_VALIDATION"),
            "shader validation is the half that bounds-checks buffer access — \
             without it `--validate` only checks API usage"
        );
        for (_, v) in VALIDATION_ENV {
            assert!(!v.is_empty(), "an empty value disables the layer");
        }
    }
}
