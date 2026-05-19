//! Smoke test for the model scanner. Runs the same `scan_local` the desktop
//! UI uses against the directory passed in (defaults to `~/models`), prints
//! one row per detected model.
//!
//! Usage:
//!   cargo run -p lumen-app --example scan_models
//!   cargo run -p lumen-app --example scan_models -- /some/other/path

use std::path::PathBuf;

use lumen_app::catalog::Catalog;
use lumen_app::models;

fn main() -> anyhow::Result<()> {
    let dir: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            directories::UserDirs::new()
                .map(|u| u.home_dir().join("models"))
                .unwrap_or_else(|| PathBuf::from("./models"))
        });

    // Fetch the catalog so we can flag supported entries.
    let cat = match lumen_app::server::resolve_binary_public(None) {
        Ok(bin) => lumen_app::catalog::fetch(&bin).unwrap_or_default(),
        Err(_) => Catalog::default(),
    };
    eprintln!(
        "scanning {} (catalog: {} recommended models)",
        dir.display(),
        cat.recommended.len()
    );

    let entries = models::scan_local(&dir, &cat)?;
    println!(
        "{:<3} {:<55} {:>10} {:<8} {:<32}",
        "✓", "id", "size", "status", "label"
    );
    for e in &entries {
        let size_mb = e.size_bytes / (1024 * 1024);
        let size = if size_mb >= 1024 {
            format!("{:.1}G", size_mb as f64 / 1024.0)
        } else {
            format!("{}M", size_mb)
        };
        let status = if e.ready { "ready" } else { "partial" };
        let mark = if e.supported { "✓" } else { " " };
        let label = e.label.as_deref().unwrap_or("—");
        println!(
            "{:<3} {:<55} {:>10} {:<8} {:<32}",
            mark, e.id, size, status, label
        );
    }
    eprintln!(
        "\n{} models total · {} supported · {} ready",
        entries.len(),
        entries.iter().filter(|e| e.supported).count(),
        entries.iter().filter(|e| e.ready).count()
    );
    Ok(())
}
