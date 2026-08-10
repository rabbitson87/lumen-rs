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

use std::process::{Command, ExitCode};

const FEATURES: &str = "lumen-mlx/mlx-native";

pub fn main(args: Vec<String>) -> ExitCode {
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
        Ok(s) if s.success() => ExitCode::SUCCESS,
        Ok(_) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("could not spawn cargo: {e}");
            ExitCode::FAILURE
        }
    }
}
