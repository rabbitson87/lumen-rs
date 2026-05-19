use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

use crate::catalog::Catalog;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    /// HF-style id (`Qwen/Qwen2.5-1.5B`) or local-only short name.
    pub id: String,
    /// Absolute path to the model directory.
    pub path: PathBuf,
    /// Total size of weight files in bytes.
    pub size_bytes: u64,
    /// `true` if this directory looks like a complete HF snapshot
    /// (`config.json` + at least one `.safetensors`).
    pub ready: bool,
    /// `true` if the model id matches a curated entry in the server-side
    /// catalog. The MODELS card hides unsupported entries by default — the
    /// user can flip a debug toggle to see everything.
    #[serde(default)]
    pub supported: bool,
    /// Catalog label if matched (e.g., "Qwen 2.5 — 1.5B Instruct"). `None`
    /// means unsupported / unknown.
    #[serde(default)]
    pub label: Option<String>,
}

/// Scan `models_dir` and return one entry per model-bearing subdirectory.
/// Handles two on-disk layouts:
///
/// 1. **Flat** (README convention `huggingface-cli download --local-dir
///    ~/models/<name>`): the directory `<name>/` contains `config.json` and
///    weight files directly. `id` = `<name>` as-is.
///
/// 2. **HF cache** (default `huggingface-cli download` output, also what
///    hf-hub crate produces): directory is `models--<org>--<repo>/`, with
///    actual files under `snapshots/<commit_sha>/` (symlinks to `blobs/`).
///    `id` = `<org>/<repo>`.
///
/// `catalog` is consulted to set the `supported` / `label` flags.
pub fn scan_local(models_dir: &Path, catalog: &Catalog) -> Result<Vec<ModelEntry>> {
    if !models_dir.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in std::fs::read_dir(models_dir)
        .with_context(|| format!("read_dir {}", models_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        // Skip dotfiles + hf-hub internal dirs.
        if name.starts_with('.') {
            continue;
        }

        let (id, scan_dir) = if let Some(rest) = name.strip_prefix("models--") {
            // HF cache layout. Strip prefix, restore `<org>/<repo>`.
            let id = rest.replacen("--", "/", 1);
            let snapshot = resolve_hf_snapshot(&path);
            match snapshot {
                Some(snap) => (id, snap),
                None => continue, // empty / partial cache entry
            }
        } else {
            // Flat layout (README convention) — scan dir directly.
            (name, path.clone())
        };

        let config_present = scan_dir.join("config.json").exists();
        let weight_present = WalkDir::new(&scan_dir)
            .max_depth(2)
            .into_iter()
            .filter_map(|e| e.ok())
            .any(|e| {
                e.path()
                    .extension()
                    .and_then(|x| x.to_str())
                    .map(|ext| ext == "safetensors" || ext == "gguf")
                    .unwrap_or(false)
            });
        // Size is summed from the original entry path (covers both HF blobs/
        // and flat layout). `follow_links=false` (the default) so symlinks
        // in HF snapshots aren't double-counted.
        let size_bytes: u64 = WalkDir::new(&path)
            .max_depth(4)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .filter_map(|e| e.metadata().ok())
            .map(|m| m.len())
            .sum();
        let rec = catalog.find_recommended(&id);
        out.push(ModelEntry {
            id,
            path,
            size_bytes,
            ready: config_present && weight_present,
            supported: rec.is_some(),
            label: rec.map(|r| r.label.clone()),
        });
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(out)
}

/// Locate the active snapshot directory inside an `models--<org>--<repo>/`
/// HF cache entry. Tries `refs/main` first (the canonical pointer), then
/// falls back to the first available `snapshots/*` directory.
fn resolve_hf_snapshot(hf_dir: &Path) -> Option<PathBuf> {
    let refs_main = hf_dir.join("refs").join("main");
    if let Ok(sha) = std::fs::read_to_string(&refs_main) {
        let sha = sha.trim();
        if !sha.is_empty() {
            let snap = hf_dir.join("snapshots").join(sha);
            if snap.exists() {
                return Some(snap);
            }
        }
    }
    // Fallback: pick any snapshot subdir.
    std::fs::read_dir(hf_dir.join("snapshots"))
        .ok()?
        .filter_map(|e| e.ok())
        .find(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .map(|e| e.path())
}

/// Resolve a repo id to a local directory under `models_dir`. Mirrors the
/// hf-hub on-disk layout so existing caches are reused.
pub fn local_path_for(models_dir: &Path, repo_id: &str) -> PathBuf {
    let flat = repo_id.replacen('/', "--", 1);
    models_dir.join(flat)
}

#[derive(Debug, Clone, Serialize)]
pub struct DownloadProgress {
    pub repo_id: String,
    pub file: String,
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
    pub done: bool,
}

/// Download a HF repo via hf-hub into `models_dir/<repo_id>`. Files are
/// pulled one-by-one because hf-hub's streaming API is per-file; the
/// caller can stream `DownloadProgress` events through the channel.
pub async fn download(
    models_dir: &Path,
    repo_id: &str,
    files: Option<Vec<String>>,
    tx: tokio::sync::mpsc::Sender<DownloadProgress>,
) -> Result<PathBuf> {
    use hf_hub::api::tokio::ApiBuilder;

    let target = local_path_for(models_dir, repo_id);
    std::fs::create_dir_all(&target).ok();

    let api = ApiBuilder::new()
        .with_cache_dir(models_dir.to_path_buf())
        .build()
        .context("init hf-hub api")?;
    let repo = api.model(repo_id.to_string());

    // Default file list — everything hf-hub considers essential for a
    // standard transformer repo. Caller can override.
    let files = files.unwrap_or_else(|| {
        vec![
            "config.json".into(),
            "tokenizer.json".into(),
            "tokenizer_config.json".into(),
            "generation_config.json".into(),
            "model.safetensors".into(),
        ]
    });

    for file in files {
        let _ = tx
            .send(DownloadProgress {
                repo_id: repo_id.into(),
                file: file.clone(),
                downloaded_bytes: 0,
                total_bytes: None,
                done: false,
            })
            .await;
        // hf-hub handles partial-file caching internally; we just await
        // the final path.
        match repo.get(&file).await {
            Ok(p) => {
                let size = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
                let _ = tx
                    .send(DownloadProgress {
                        repo_id: repo_id.into(),
                        file: file.clone(),
                        downloaded_bytes: size,
                        total_bytes: Some(size),
                        done: true,
                    })
                    .await;
            }
            Err(e) => {
                // Skip files that don't exist in this repo (e.g. some repos
                // ship `model.safetensors.index.json` + shards instead of a
                // single `model.safetensors`). Real network errors propagate.
                let msg = format!("{e}");
                if msg.contains("404") || msg.contains("not found") {
                    continue;
                }
                return Err(anyhow::anyhow!("download {file}: {e}"));
            }
        }
    }

    Ok(target)
}

pub async fn delete(models_dir: &Path, repo_id: &str) -> Result<()> {
    let target = local_path_for(models_dir, repo_id);
    if target.exists() {
        std::fs::remove_dir_all(&target)
            .with_context(|| format!("remove_dir_all {}", target.display()))?;
    }
    Ok(())
}
