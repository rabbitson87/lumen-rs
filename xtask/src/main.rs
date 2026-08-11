//! Workspace task runner.
//!
//! ```text
//! cargo xtask test                 # the workspace suite, features + serial
//! cargo xtask test -p lumen-metal  # extra args pass through to cargo test
//! cargo xtask red-green            # verify every regression guard
//! cargo xtask red-green --list     # names + the symptom each defect caused
//! cargo xtask red-green lark-opener
//! cargo xtask fuzz --list          # libFuzzer soak targets + what each probes
//! cargo xtask fuzz tool_body_parse --minutes 10
//! ```

mod fuzz;
mod red_green;
mod test_all;

const USAGE: &str = "usage: cargo xtask <test [CARGO ARGS…] | red-green [--list] [NAME…] | fuzz <TARGET…|--all|--list> [--minutes N]>";

fn main() -> std::process::ExitCode {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("test") => test_all::main(args.collect()),
        Some("red-green") => red_green::main(args.collect()),
        Some("fuzz") => fuzz::main(args.collect()),
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
