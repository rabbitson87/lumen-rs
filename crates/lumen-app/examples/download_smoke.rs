//! End-to-end smoke test for `lumen_app::models::download()`.
//!
//! Validates that the raw-HTTPS downloader (no `hf-hub` crate, no external
//! CLI) actually fetches all weight + metadata files for both:
//!   1. Single-file repos (e.g. `Qwen/Qwen2.5-0.5B` ships one `model.safetensors`)
//!   2. Multi-shard repos (e.g. `hsng95/gemma-4-26b-a4b-mlx-imatrix3plus-awq`
//!      ships 3 `model-NNNNN-of-00003.safetensors` + index.json).
//!
//! Run:
//!   cargo run --release -p lumen-app --example download_smoke -- <repo_id> [models_dir]
//!
//! Defaults to `Qwen/Qwen2.5-0.5B` into `/tmp/lumen-models-smoke/` because
//! it's the smallest catalog entry (~1 GB) — good for a fast e2e check.

use std::path::PathBuf;

use anyhow::Result;
use lumen_app::models::{DownloadProgress, download};
use tokio::sync::mpsc;

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let repo_id = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "Qwen/Qwen2.5-0.5B".to_string());
    let models_dir = PathBuf::from(
        args.get(2)
            .cloned()
            .unwrap_or_else(|| "/tmp/lumen-models-smoke".to_string()),
    );

    eprintln!("[smoke] repo_id   = {repo_id}");
    eprintln!("[smoke] models_dir = {}", models_dir.display());
    std::fs::create_dir_all(&models_dir).ok();

    let (tx, mut rx) = mpsc::channel::<DownloadProgress>(256);

    // Background progress reporter — prints one line per file completion +
    // a percentage tick every ~10% during large shard streams.
    let progress = tokio::spawn(async move {
        let mut last_pct_by_file: std::collections::HashMap<String, u8> =
            std::collections::HashMap::new();
        while let Some(p) = rx.recv().await {
            if p.done {
                let mb = p.downloaded_bytes as f64 / 1e6;
                eprintln!(
                    "  ✓ {:<40} {:>9.2} MB",
                    p.file, mb
                );
            } else if let Some(total) = p.total_bytes {
                if total == 0 {
                    continue;
                }
                let pct = ((p.downloaded_bytes * 100) / total) as u8;
                let last = last_pct_by_file.get(&p.file).copied().unwrap_or(0);
                if pct >= last + 10 {
                    eprintln!(
                        "    {:<40} {:>3}%  ({:.1} / {:.1} MB)",
                        p.file,
                        pct,
                        p.downloaded_bytes as f64 / 1e6,
                        total as f64 / 1e6,
                    );
                    last_pct_by_file.insert(p.file.clone(), pct);
                }
            }
        }
    });

    let t0 = std::time::Instant::now();
    let target = download(&models_dir, &repo_id, None, tx).await?;
    progress.await.ok();
    let elapsed = t0.elapsed();

    eprintln!();
    eprintln!("[smoke] target dir: {}", target.display());

    // Validate: config.json + at least one .safetensors must exist.
    let mut has_config = false;
    let mut has_tokenizer = false;
    let mut safetensors_count = 0usize;
    let mut total_bytes = 0u64;
    for entry in std::fs::read_dir(&target)? {
        let e = entry?;
        let name = e.file_name().to_string_lossy().to_string();
        let meta = e.metadata()?;
        if name == "config.json" {
            has_config = true;
        }
        if name == "tokenizer.json" || name == "tokenizer.model" {
            has_tokenizer = true;
        }
        if name.ends_with(".safetensors") {
            safetensors_count += 1;
            total_bytes += meta.len();
        }
    }

    eprintln!("[smoke] elapsed:        {:.1}s", elapsed.as_secs_f64());
    eprintln!("[smoke] has_config:     {has_config}");
    eprintln!("[smoke] has_tokenizer:  {has_tokenizer}");
    eprintln!("[smoke] safetensors:    {safetensors_count} file(s)");
    eprintln!("[smoke] weight bytes:   {:.2} GB", total_bytes as f64 / 1e9);

    if !has_config || !has_tokenizer || safetensors_count == 0 {
        anyhow::bail!(
            "incomplete download: config={has_config} tokenizer={has_tokenizer} \
             safetensors={safetensors_count}"
        );
    }
    eprintln!("[smoke] ✓ PASS");
    Ok(())
}
