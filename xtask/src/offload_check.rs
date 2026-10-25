//! `cargo xtask offload-check` — does serving the experts from the SSD change a single token?
//!
//! The offload engine is only worth having if it is exact. The proof set is ordinary tests and
//! runs inside `cargo xtask test`; this runs it as one question, and with `--real <model dir>`
//! asks it of a real checkpoint too:
//!
//! 1. the replica tests (`qwen4_exp`, `offload`): logits bit-identical to the resident model at
//!    every pool size and region count, through the staging path, a layer-major prefill, a
//!    warmed pool and the expert pack, and for prefill calls of every size (MLX's gather kernel
//!    changes with the call size, so the path a call takes must not depend on the pool);
//! 2. with `--real`: the same greedy continuation generated at several pool sizes and with the
//!    pack on and off must give the same token ids (each configuration is one process, so no
//!    run inherits another's cache).
//!
//! Speculative decoding is not part of the claim: a two-row verify pass re-associates sums, so
//! `LUMEN_QWEN4_MTP` stays off here.

use std::process::{Command, ExitCode};

const USAGE: &str = "usage: cargo xtask offload-check [--real <model dir>]";

/// The pool sizes (GB) and pack modes the real checkpoint is run at.
const REAL_RUNS: &[(&str, &str)] = &[("6", "off"), ("6", "auto"), ("12", "off"), ("20", "auto")];

pub fn main(args: Vec<String>) -> ExitCode {
    let real = match args.as_slice() {
        [] => None,
        [flag, dir] if flag == "--real" => Some(dir.clone()),
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    println!("── replica tests ──");
    let status = Command::new("cargo")
        .current_dir(crate::repo_root())
        .args([
            "test",
            "-p",
            "lumen-mlx",
            "--features",
            "mlx-native",
            "--lib",
            "--",
            "qwen4_exp",
            "offload",
            "--test-threads=1",
        ])
        .status();
    if !matches!(status, Ok(s) if s.success()) {
        eprintln!("offload-check: the replica tests failed");
        return ExitCode::FAILURE;
    }
    let Some(dir) = real else {
        println!(
            "offload-check: replica tests passed (add --real <model dir> to check a checkpoint)"
        );
        return ExitCode::SUCCESS;
    };
    println!("── {dir} ──");
    let mut reference: Option<(String, String)> = None;
    for (pool, pack) in REAL_RUNS {
        let out = Command::new("cargo")
            .current_dir(crate::repo_root())
            .env("LUMEN_MOE_PACK", pack)
            .env("LUMEN_MOE_IO_MODE", "direct")
            .env("LUMEN_QWEN4_MTP", "off")
            // Each run must compute from scratch, not restore a state an earlier run saved.
            .env("LUMEN_KV_DISK", "0")
            .args([
                "run",
                "--release",
                "-q",
                "-p",
                "lumen-mlx",
                "--features",
                "mlx-native",
                "--example",
                "qwen4_exp_generate",
                "--",
                &dir,
                "Explain in a few sentences how a hash table handles collisions.",
                "48",
                pool,
                "--chat",
            ])
            .output();
        let Ok(out) = out else {
            eprintln!("offload-check: could not run the example");
            return ExitCode::FAILURE;
        };
        let stderr = String::from_utf8_lossy(&out.stderr);
        let Some(tokens) = stderr.lines().find_map(|l| l.strip_prefix("tokens: ")) else {
            eprintln!("offload-check: pool {pool} GB pack {pack}: no token line\n{stderr}");
            return ExitCode::FAILURE;
        };
        let label = format!("pool {pool} GB, pack {pack}");
        println!("{label}: {} tokens", tokens.matches(',').count() + 1);
        match &reference {
            None => reference = Some((label, tokens.to_string())),
            Some((first, want)) if want != tokens => {
                eprintln!("offload-check: {label} differs from {first}\n  {want}\n  {tokens}");
                return ExitCode::FAILURE;
            }
            Some(_) => {}
        }
    }
    println!("offload-check: every configuration gave the same tokens");
    ExitCode::SUCCESS
}
