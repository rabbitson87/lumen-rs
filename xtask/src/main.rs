//! Workspace task runner.
//!
//! ```text
//! cargo xtask red-green            # verify every regression guard
//! cargo xtask red-green --list     # names + the symptom each defect caused
//! cargo xtask red-green lark-opener
//! ```

mod red_green;

fn main() -> std::process::ExitCode {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("red-green") => red_green::main(args.collect()),
        Some(other) => {
            eprintln!("unknown task {other:?}\n\nusage: cargo xtask red-green [--list] [NAME…]");
            std::process::ExitCode::from(2)
        }
        None => {
            eprintln!("usage: cargo xtask red-green [--list] [NAME…]");
            std::process::ExitCode::from(2)
        }
    }
}
