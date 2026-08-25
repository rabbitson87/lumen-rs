//! What the MODELS card's **Use** + **Start** buttons actually do, without the
//! buttons.
//!
//! Starting the server from the desktop app is a materially different launch
//! from `MODEL_ID=… lumen-server`: it resolves the model id against the local
//! scan (so a flat directory name becomes an absolute path) and sets two dozen
//! environment variables, several of which change server decisions —
//! `LUMEN_PREFILL_CHUNK` doubles as the prompt-size reject cap, and an emitted
//! `LUMEN_SPEC` would disable the server's own MTP auto-enable.
//!
//! None of that is reachable from a test that only drives the HTTP API. This
//! reproduces the launch exactly — same config source, same id resolution as
//! `commands::start_server`, same `apply_env` — and either prints it or runs
//! it, so the surface can be inspected without a display.
//!
//! It is not a substitute for pressing the button, which is drivable: posting a
//! `CGEvent` to the HID tap clicks a Tauri webview with no Accessibility grant
//! (AppleScript's `System Events … click at` is the one that needs it and fails
//! with `-25204`). Do that when the question is whether the UI works; use this
//! when the question is what the UI hands the server.
//!
//! ```text
//! cargo run -p lumen-app --example ui_launch -- --model Qwen3.8      # print
//! cargo run -p lumen-app --example ui_launch -- --model Qwen3.8 --run
//! ```
//!
//! `--model` is matched as a case-insensitive substring of the catalog id, so
//! the full `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed` is not needed.

use anyhow::{Context, Result, anyhow};
use lumen_app::{catalog, config::PersistentConfig, models, server};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let want = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .ok_or_else(|| anyhow!("usage: --model <substring-of-catalog-id> [--run]"))?;
    let run = args.iter().any(|a| a == "--run");

    // Same three sources the Tauri command reads: persisted config, the
    // server's own catalog, and the local model scan.
    let cfg = PersistentConfig::load_or_default().context("load app config")?;
    let binary = server::resolve_binary_public(cfg.server_binary_path.as_deref())
        .context("locate lumen-server the way the app does")?;
    let cat = catalog::fetch(&binary).context("lumen-server --catalog")?;
    let entries = models::scan_local(&cfg.models_dir, &cat).context("scan local models")?;

    let entry = entries
        .iter()
        .find(|m| m.id.to_lowercase().contains(&want.to_lowercase()))
        .ok_or_else(|| {
            anyhow!(
                "no local model matching {want:?}; the app sees: {:?}",
                entries.iter().map(|m| &m.id).collect::<Vec<_>>()
            )
        })?;

    // The MODELS card refuses to select an entry the catalog does not know —
    // `disabled={!m.supported}`. Reproduce that rather than launching something
    // the UI would not let a user launch.
    if !entry.supported {
        return Err(anyhow!(
            "`{}` is on disk but NOT in the catalog, so the UI's Use button is \
             disabled for it and this launch is not one a user could perform. \
             Local directories must be named `<org>--<repo>` matching a \
             recommended id.",
            entry.id
        ));
    }
    if !entry.ready {
        return Err(anyhow!("`{}` is incomplete on disk", entry.id));
    }

    // `commands::start_server` passes the resolved absolute path, not the id,
    // so the loader skips HF Hub entirely.
    let model_arg = entry.path.to_string_lossy().into_owned();
    let env = server::launch_env(
        &cfg,
        &model_arg,
        Some(entry.size_bytes),
        None,
        server::ServeKind::Chat,
    );

    eprintln!(
        "model : {} ({})",
        entry.id,
        entry.label.as_deref().unwrap_or("—")
    );
    eprintln!("binary: {}", binary.display());
    eprintln!("env   : {} variables", env.len());
    for (k, v) in &env {
        eprintln!("  {k}={v}");
    }
    if !run {
        eprintln!("\n(dry run — pass --run to actually start the server)");
        return Ok(());
    }

    let status = std::process::Command::new(&binary)
        .envs(&env)
        .status()
        .context("spawn lumen-server")?;
    std::process::exit(status.code().unwrap_or(1));
}
