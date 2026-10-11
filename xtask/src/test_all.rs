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
    workspace_root()
        .join("target")
        .join("effective-cases.jsonl")
}

/// The repository root, whatever directory `cargo xtask` was run from.
fn workspace_root() -> std::path::PathBuf {
    let xtask = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    xtask.parent().unwrap_or(xtask).to_path_buf()
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
    rows.sort_by_key(|r| std::cmp::Reverse(r.0));
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
            // A filtered run is someone looking at one thing; the kernel pass
            // is a property of the whole tree, so it belongs to the full run.
            if !args.is_empty() {
                eprintln!(
                    "\n(skipping the Metal {OLDEST_METAL} kernel pass: it runs on an unfiltered \
                     `cargo xtask test`)"
                );
                return ExitCode::SUCCESS;
            }
            if oldest_metal_kernel_pass(validate) {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
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

/// The Metal language version the oldest supported macOS (14) compiles
/// run-time kernels as. The release targets macOS 14, and MLX compiles
/// run-time kernels for the version the *user's* macOS supports — which this
/// machine never does, so a kernel that needs a newer Metal would pass every
/// local test and fail only on the user's Mac.
const OLDEST_METAL: &str = "3.1";

/// Every test in every source file that builds a run-time kernel
/// (`MetalKernel::new`), run again with `MLX_METAL_LANGUAGE_VERSION` set so MLX
/// compiles each kernel for the Metal language version macOS 14 supports —
/// with this machine's compiler, and for the instantiations the tests use.
///
/// The files are found, not listed, so a new kernel is covered by the test
/// written for it; a kernel file with no test at all fails the pass, because
/// then nothing would check it. Ignored tests that only need a Metal device
/// (tier 1) are included — most kernel tests are — and the ones that need
/// weights are not. What counts is what ran (see [`kernel_coverage_gaps`]).
/// `metal_kernel`'s probe test checks that the variable took effect; an MLX
/// build from before the override would otherwise pass this vacuously.
fn oldest_metal_kernel_pass(validate: bool) -> bool {
    eprintln!(
        "\n── run-time kernels as macOS 14 compiles them \
         (MLX_METAL_LANGUAGE_VERSION={OLDEST_METAL}) ──"
    );
    let files = match kernel_files() {
        Ok(f) if !f.is_empty() => f,
        Ok(_) => {
            eprintln!("no file builds a run-time kernel (`MetalKernel::new`) — nothing to check");
            return false;
        }
        Err(e) => {
            eprintln!("could not find the kernel files: {e}");
            return false;
        }
    };
    let listed = match list_lib_tests() {
        Ok(l) => l,
        Err(e) => {
            eprintln!("could not list the tests: {e}");
            return false;
        }
    };
    let root = workspace_root();
    let mut names: Vec<String> = Vec::new();
    let mut untested: Vec<String> = Vec::new();
    let mut declared: Vec<String> = Vec::new();
    for file in &files {
        let Ok(src) = std::fs::read_to_string(root.join(file)) else {
            eprintln!("could not read {file}");
            return false;
        };
        match declared_kernels(&src) {
            Ok(k) => declared.extend(k),
            Err(e) => {
                eprintln!("{file}: {e}");
                return false;
            }
        }
        let prefix = module_prefix(file);
        let mut found = false;
        for (test, reason) in test_fns(&src) {
            if reason
                .as_deref()
                .and_then(crate::tiers::tier_id)
                .is_some_and(|t| t >= 2)
            {
                continue; // needs weights or a fixture this pass cannot assume
            }
            for full in listed
                .iter()
                .filter(|n| n.starts_with(&prefix) && n.ends_with(&format!("::{test}")))
            {
                found = true;
                names.push(full.clone());
            }
        }
        if !found {
            untested.push(file.clone());
        }
    }
    if !untested.is_empty() {
        for f in &untested {
            eprintln!("{f} builds a run-time kernel but has no test this pass can run");
        }
        return false;
    }

    let mut cmd = Command::new("cargo");
    cmd.args([
        "test",
        "--workspace",
        "--features",
        FEATURES,
        "--release",
        "--lib",
    ])
    .args([
        "--",
        "--test-threads=1",
        "--include-ignored",
        "--nocapture",
        "--exact",
    ])
    .args(&names)
    .env("MLX_METAL_LANGUAGE_VERSION", OLDEST_METAL);
    if validate {
        for (k, v) in VALIDATION_ENV {
            cmd.env(k, v);
        }
    }
    eprintln!(
        "{} tests from {} kernel files, {} kernels declared",
        names.len(),
        files.len(),
        declared.len()
    );
    let out = match cmd.output() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("could not spawn cargo: {e}");
            return false;
        }
    };
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    // Results and failures, without the per-test chatter `--nocapture` adds.
    for line in text
        .lines()
        .filter(|l| l.starts_with("test ") || l.contains("panicked") || l.starts_with("error"))
    {
        eprintln!("{line}");
    }
    if !out.status.success() {
        return false;
    }
    let gaps = kernel_coverage_gaps(&declared, &text);
    if !gaps.is_empty() {
        for gap in &gaps {
            eprintln!("{gap}");
        }
        return false;
    }
    eprintln!(
        "all {} kernels compiled and ran as Metal {OLDEST_METAL}",
        declared.len()
    );
    true
}

/// What the pass's output says was left unchecked, one message per gap.
///
/// A test that returned early (a missing fixture) passes without compiling
/// anything, so only a kernel that actually ran was checked at Metal 3.1 —
/// read off the `[metal-kernel-ran] <name> <dtype args…>` lines `MetalKernel`
/// prints in test builds. And MLX compiles each instantiation separately: a
/// kernel generic over its float type that only ran in f32 left the bfloat16
/// build the models use unchecked.
fn kernel_coverage_gaps(declared: &[String], output: &str) -> Vec<String> {
    let mut ran: std::collections::BTreeMap<&str, Vec<&str>> = Default::default();
    for line in output.lines() {
        if let Some(rest) = line.trim().strip_prefix("[metal-kernel-ran] ") {
            let (name, args) = rest.split_once(' ').unwrap_or((rest, ""));
            ran.entry(name).or_default().push(args);
        }
    }
    let mut gaps = Vec::new();
    for k in declared {
        match ran.get(k.as_str()) {
            None => gaps.push(format!(
                "kernel `{k}` never ran in this pass — its tests skipped or do not call it; \
                 add a test that runs it without fixtures"
            )),
            Some(runs)
                if runs.iter().any(|a| !a.is_empty())
                    && !runs.iter().any(|a| a.contains("=Bfloat16")) =>
            {
                gaps.push(format!(
                    "kernel `{k}` only ran as [{}] — the models run it in bfloat16, a \
                     separate compile; add a bf16 case",
                    runs.join("], [")
                ))
            }
            Some(_) => {}
        }
    }
    gaps
}

/// The kernel names a source file declares: the string literal each
/// `MetalKernel::new(` call opens with. A name that is not a literal is an
/// error — the pass could not hold that kernel to running.
fn declared_kernels(src: &str) -> Result<Vec<String>, String> {
    let code: String = src
        .lines()
        .map(|l| l.split_once("//").map_or(l, |(code, _)| code))
        .collect::<Vec<_>>()
        .join("\n");
    code.split("MetalKernel::new(")
        .skip(1)
        .map(|rest| {
            let rest = rest.trim_start();
            rest.strip_prefix('"')
                .and_then(|r| r.split('"').next())
                .map(str::to_string)
                .ok_or_else(|| {
                    let call: String = rest.chars().take(40).collect();
                    format!(
                        "`MetalKernel::new({call}…` does not name its kernel with a string \
                         literal, so the Metal {OLDEST_METAL} pass cannot check that it ran"
                    )
                })
        })
        .collect()
}

/// Source files that construct a run-time kernel, relative to the root —
/// untracked ones included, so a new kernel is checked before it is staged.
fn kernel_files() -> Result<Vec<String>, String> {
    let out = Command::new("git")
        .current_dir(workspace_root())
        .args([
            "grep",
            "-l",
            "--untracked",
            "MetalKernel::new(",
            "--",
            "crates/*/src/*.rs",
        ])
        .output()
        .map_err(|e| e.to_string())?;
    // `git grep` exits 1 for "no match"; anything else is a failure.
    if !matches!(out.status.code(), Some(0 | 1)) {
        return Err(String::from_utf8_lossy(&out.stderr).into_owned());
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::to_string)
        .collect())
}

/// Every lib test's full path, ignored ones included, from the same build the
/// main run used (so listing compiles nothing new).
fn list_lib_tests() -> Result<Vec<String>, String> {
    let out = Command::new("cargo")
        .args([
            "test",
            "--workspace",
            "--features",
            FEATURES,
            "--release",
            "--lib",
        ])
        .args(["--", "--list", "--include-ignored"])
        .output()
        .map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into_owned());
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.strip_suffix(": test"))
        .map(str::to_string)
        .collect())
}

/// `crates/lumen-mlx/src/native_ssm.rs` → `native_ssm::`; a `mod.rs` or
/// `lib.rs` names its directory (or the crate root, which is every test).
fn module_prefix(file: &str) -> String {
    let rel = file.split_once("/src/").map(|(_, r)| r).unwrap_or(file);
    let rel = rel.strip_suffix(".rs").unwrap_or(rel);
    let parts: Vec<&str> = rel
        .split('/')
        .filter(|p| *p != "mod" && *p != "lib")
        .collect();
    if parts.is_empty() {
        String::new()
    } else {
        format!("{}::", parts.join("::"))
    }
}

/// `(name, ignore reason)` for every `#[test]` fn in `src`: attributes in any
/// order, wrapped reasons joined, blank lines and comments between them skipped.
fn test_fns(src: &str) -> Vec<(String, Option<String>)> {
    let mut out = Vec::new();
    let (mut is_test, mut reason) = (false, None::<String>);
    for (_, line) in crate::tiers::logical_lines(src) {
        if line.is_empty() || line.starts_with("//") {
            continue;
        }
        if line.starts_with("#[") {
            if line.starts_with("#[test]") {
                is_test = true;
            } else if line.starts_with("#[ignore") {
                reason = Some(
                    line.split_once('"')
                        .and_then(|(_, r)| r.split('"').next())
                        .unwrap_or("")
                        .to_string(),
                );
            }
            continue;
        }
        if is_test
            && let Some(rest) = line.split("fn ").nth(1)
            && let Some(name) = rest.split('(').next()
        {
            out.push((name.trim().to_string(), reason.take()));
        }
        (is_test, reason) = (false, None);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pass finds kernel tests by file and name; both halves have to read
    /// the shapes this repo writes.
    #[test]
    fn kernel_tests_are_found_by_file_and_name() {
        assert_eq!(
            module_prefix("crates/lumen-mlx/src/native_ssm.rs"),
            "native_ssm::"
        );
        assert_eq!(module_prefix("crates/x/src/gemma4/mod.rs"), "gemma4::");
        assert_eq!(
            module_prefix("crates/x/src/gemma4/attn.rs"),
            "gemma4::attn::"
        );
        assert_eq!(module_prefix("crates/x/src/lib.rs"), "");
        let src = "#[test]\n#[ignore = \"MLX FFI requires non-sandbox host with Metal device\"]\n\
                   fn gpu_one() {}\n#[test]\nfn plain() {}\n#[test]\n#[ignore = \"needs a checkpoint\"]\n\
                   fn weights() {}\n\
                   #[ignore = \"needs a \\\n    checkpoint\"]\n#[test]\n\n/// doc\nfn wrapped_first() {}\n\
                   fn not_a_test() {}\n";
        let fns = test_fns(src);
        assert_eq!(
            fns.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
            ["gpu_one", "plain", "weights", "wrapped_first"]
        );
        assert_eq!(fns[1].1, None);
        let tier = |i: usize| fns[i].1.as_deref().and_then(crate::tiers::tier_id);
        assert_eq!(tier(0), Some(1), "a Metal-only test is in the pass");
        assert!(tier(2).is_some_and(|t| t >= 2), "a weights test is not");
        assert!(
            tier(3).is_some_and(|t| t >= 2),
            "a wrapped reason written above `#[test]` still keeps it out: {:?}",
            fns[3].1
        );
    }

    /// The names the pass holds every kernel to, read off the constructor
    /// calls however they are wrapped.
    #[test]
    fn declared_kernels_are_read_from_the_constructor_calls() {
        let src = "let k = MetalKernel::new(\n            \"gated_delta_step\",\n            &[\"q\"],\n\
                   ); let j = MetalKernel::new(\"myexp\", &[], &[]); // MetalKernel::new(name)";
        assert_eq!(
            declared_kernels(src).unwrap(),
            ["gated_delta_step", "myexp"]
        );
        let err = declared_kernels("let k = MetalKernel::new(NAME, &[], &[]);").unwrap_err();
        assert!(err.contains("NAME"), "{err}");
    }

    /// A kernel counts as checked only when it ran, and a kernel generic over
    /// its float type only when it also ran as the bfloat16 the models use.
    #[test]
    fn coverage_counts_instantiations_that_ran() {
        let declared = ["a", "b", "c", "d"].map(String::from);
        let out = "running 4 tests\n[metal-kernel-ran] a InT=Float32 StT=Float32\n\
                   [metal-kernel-ran] a InT=Bfloat16 StT=Float32\n[metal-kernel-ran] b\n\
                   [metal-kernel-ran] c DT=Float32\ntest x ... ok\n";
        let gaps = kernel_coverage_gaps(&declared, out);
        assert_eq!(gaps.len(), 2, "{gaps:?}");
        assert!(
            gaps[0].contains("`c` only ran as [DT=Float32]"),
            "{}",
            gaps[0]
        );
        assert!(gaps[1].contains("`d` never ran"), "{}", gaps[1]);
    }

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
