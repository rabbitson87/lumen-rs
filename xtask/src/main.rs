//! Workspace task runner.
//!
//! ```text
//! cargo xtask test                 # the workspace suite, features + serial
//! cargo xtask test -p lumen-mlx    # extra args pass through to cargo test
//! cargo xtask red-green            # verify every regression guard
//! cargo xtask red-green --list     # names + the symptom each defect caused
//! cargo xtask red-green lark-opener
//! cargo xtask fuzz --list          # libFuzzer soak targets + what each probes
//! cargo xtask fuzz tool_body_parse --minutes 10
//! cargo xtask flags --list         # env-flag registry (kind, default, docs)
//! cargo xtask flags                # suite with every Optimization flag flipped
//! cargo xtask offload-check        # is SSD expert offload bit-identical to resident?
//! cargo xtask offload-check --real <model dir>
//! ```

mod coverage;
mod faults;
mod flags;
mod fuzz;
mod gate;
mod offload_check;
mod red_green;
mod test_all;
mod tiers;

use std::path::{Path, PathBuf};
use std::process::Command;

const USAGE: &str = "usage: cargo xtask <test [--validate] [CARGO ARGS…] | red-green [--list] [NAME…] | fuzz <TARGET…|--all|--list> [--minutes N] | flags [--list|--docs|--check|--one-at-a-time] | coverage [--mcdc] [--html] | gate [--quick] | tiers | faults [--cancel] | offload-check [--real DIR]>";

fn main() -> std::process::ExitCode {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("test") => test_all::main(args.collect()),
        Some("red-green") => red_green::main(args.collect()),
        Some("fuzz") => fuzz::main(args.collect()),
        Some("flags") => flags::main(args.collect()),
        Some("coverage") => coverage::main(args.collect()),
        Some("gate") => gate::main(args.collect()),
        Some("tiers") => tiers::main(args.collect()),
        Some("faults") => faults::main(args.collect()),
        Some("offload-check") => offload_check::main(args.collect()),
        Some(other) => {
            eprintln!("unknown task {other:?}\n\n{USAGE}");
            std::process::ExitCode::from(2)
        }
        None => {
            eprintln!("{USAGE}");
            std::process::ExitCode::from(2)
        }
    }
}

/// The repository root, whatever directory `cargo xtask` was run from.
fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is `<workspace>/xtask`.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask sits one level below the repo root")
        .to_path_buf()
}

/// `git ARGS` run at the repository root, returning its stdout.
///
/// The exit status is deliberately not checked: `git grep` exits 1 when
/// nothing matches, which is the good case for most callers. A caller that has
/// to tell "no match" from a failure runs git itself.
fn git(args: &[&str]) -> Result<String, String> {
    let out = Command::new("git")
        .current_dir(repo_root())
        .args(args)
        .output()
        .map_err(|e| e.to_string())?;
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}
