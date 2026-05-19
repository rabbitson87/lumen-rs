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
            // hf-hub cache layout: `models--<org>--<repo>/snapshots/<sha>/<files>`
            let id = rest.replacen("--", "/", 1);
            let snapshot = resolve_hf_snapshot(&path);
            match snapshot {
                Some(snap) => (id, snap),
                None => continue, // empty / partial cache entry
            }
        } else if name.contains("--") {
            // Lumen flat hf layout: `<org>--<repo>/<files>` — what the
            // in-app downloader (`models::download`) writes. Restore the
            // canonical `<org>/<repo>` id so catalog matching works the
            // same as for hf-hub-style caches.
            let id = name.replacen("--", "/", 1);
            (id, path.clone())
        } else {
            // Plain flat (README convention, no org prefix in dir name).
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

/// Download a HF repo via plain HTTPS (no `hf-hub` crate, no external CLI).
/// Files land under `models_dir/<org>--<repo>/<file>` — the flat layout
/// `scan_local()` already understands.
///
/// Streams one file at a time, emitting `DownloadProgress` per chunk so the
/// frontend can render a progress bar. Multi-shard repos (Gemma-4, MoE,
/// 3B+ Qwen) are handled by probing `model.safetensors.index.json` and
/// enumerating every shard listed in `weight_map`. Single-file repos
/// (Qwen 0.5B/1.5B) fall through to plain `model.safetensors`.
///
/// HF resolve URL pattern:
///   `https://huggingface.co/<repo_id>/resolve/main/<file>`
/// 302-redirects to the LFS storage for large files. `reqwest` follows
/// redirects by default (up to 10 hops).
pub async fn download(
    models_dir: &Path,
    repo_id: &str,
    files: Option<Vec<String>>,
    tx: tokio::sync::mpsc::Sender<DownloadProgress>,
) -> Result<PathBuf> {
    let client = reqwest::Client::builder()
        .user_agent(concat!("lumen-app/", env!("CARGO_PKG_VERSION")))
        .build()
        .context("init http client")?;

    let target = local_path_for(models_dir, repo_id);
    std::fs::create_dir_all(&target)
        .with_context(|| format!("create_dir_all {}", target.display()))?;

    let resolve_url = |file: &str| {
        format!("https://huggingface.co/{}/resolve/main/{}", repo_id, file)
    };

    // Resolve file list: caller-supplied OR probe repo for standard layout.
    let files = match files {
        Some(f) => f,
        None => {
            let mut list: Vec<String> = vec![
                "config.json".into(),
                "tokenizer.json".into(),
                "tokenizer.model".into(),
                "tokenizer_config.json".into(),
                "special_tokens_map.json".into(),
                "added_tokens.json".into(),
                "generation_config.json".into(),
                "chat_template.jinja".into(),
                "preprocessor_config.json".into(),
            ];

            // Probe for sharded weights.
            let idx_url = resolve_url("model.safetensors.index.json");
            let idx_resp = client
                .get(&idx_url)
                .send()
                .await
                .with_context(|| format!("probe index for {repo_id}"))?;
            if idx_resp.status().is_success() {
                list.push("model.safetensors.index.json".into());
                let idx_text = idx_resp.text().await.context("read index body")?;
                let v: serde_json::Value = serde_json::from_str(&idx_text)
                    .context("parse safetensors index")?;
                if let Some(wm) = v.get("weight_map").and_then(|w| w.as_object()) {
                    let mut shards = std::collections::BTreeSet::new();
                    for (_k, vv) in wm {
                        if let Some(s) = vv.as_str() {
                            shards.insert(s.to_string());
                        }
                    }
                    list.extend(shards);
                }
            } else if idx_resp.status() == reqwest::StatusCode::NOT_FOUND {
                // Single-file repo — fall through to plain `model.safetensors`.
                list.push("model.safetensors".into());
            } else {
                return Err(anyhow::anyhow!(
                    "probe safetensors index for {repo_id}: HTTP {}",
                    idx_resp.status()
                ));
            }
            list
        }
    };

    for file in &files {
        let url = resolve_url(file);
        let dest = target.join(file);

        // Skip if a non-empty file already exists at the destination (resume
        // semantics — coarse, but matches user expectation when re-clicking
        // download on an already-complete model).
        if let Ok(meta) = std::fs::metadata(&dest) {
            if meta.len() > 0 {
                let _ = tx
                    .send(DownloadProgress {
                        repo_id: repo_id.into(),
                        file: file.clone(),
                        downloaded_bytes: meta.len(),
                        total_bytes: Some(meta.len()),
                        done: true,
                    })
                    .await;
                continue;
            }
        }

        let _ = tx
            .send(DownloadProgress {
                repo_id: repo_id.into(),
                file: file.clone(),
                downloaded_bytes: 0,
                total_bytes: None,
                done: false,
            })
            .await;

        let resp = client
            .get(&url)
            .send()
            .await
            .with_context(|| format!("GET {url}"))?;
        let status = resp.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            // File not present in this repo — skip silently. Common for the
            // optional metadata files in the default list.
            continue;
        }
        let mut resp = resp
            .error_for_status()
            .with_context(|| format!("GET {url}"))?;
        let total = resp.content_length();

        let mut file_w = tokio::fs::File::create(&dest)
            .await
            .with_context(|| format!("create {}", dest.display()))?;
        let mut downloaded: u64 = 0;
        let mut last_emit: u64 = 0;
        while let Some(chunk) = resp
            .chunk()
            .await
            .with_context(|| format!("read chunk from {url}"))?
        {
            use tokio::io::AsyncWriteExt;
            file_w
                .write_all(&chunk)
                .await
                .with_context(|| format!("write {}", dest.display()))?;
            downloaded += chunk.len() as u64;
            // Throttle progress events to every ~512 KB so the frontend isn't
            // flooded on large shards (5+ GB).
            if downloaded - last_emit >= 512 * 1024 {
                let _ = tx
                    .send(DownloadProgress {
                        repo_id: repo_id.into(),
                        file: file.clone(),
                        downloaded_bytes: downloaded,
                        total_bytes: total,
                        done: false,
                    })
                    .await;
                last_emit = downloaded;
            }
        }
        use tokio::io::AsyncWriteExt;
        file_w
            .flush()
            .await
            .with_context(|| format!("flush {}", dest.display()))?;

        let _ = tx
            .send(DownloadProgress {
                repo_id: repo_id.into(),
                file: file.clone(),
                downloaded_bytes: downloaded,
                total_bytes: total.or(Some(downloaded)),
                done: true,
            })
            .await;
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
