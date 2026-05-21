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
    /// HF Hub commit SHA recorded when this snapshot was downloaded. `None`
    /// for legacy installs from before v0.1.3 (no SHA was tracked at the time)
    /// and for local-only models without an HF id. Used by `check_model_updates`
    /// to detect when the Hub-side repo has been re-uploaded (same id, new
    /// weights — e.g. `hsng95/gemma-4-26b-a4b-mlx-imatrix3plus-awq` after the imatrix
    /// rebuild). Persisted in a small `.lumen_hub_sha` text file inside the
    /// model directory.
    #[serde(default)]
    pub local_sha: Option<String>,
}

/// File where we persist the HF Hub commit SHA at download time. Sits next
/// to `config.json` etc. — small (~40 bytes), purely informational, ignored
/// by every other consumer of the model directory.
pub const SHA_MARKER_FILE: &str = ".lumen_hub_sha";

/// Write the SHA marker. Called by `download()` after all files transfer.
/// Best-effort: a failure here is logged but doesn't abort the install,
/// because the model files are already on disk and usable — we just can't
/// detect future Hub updates.
pub fn write_sha_marker(model_dir: &Path, sha: &str) {
    let marker = model_dir.join(SHA_MARKER_FILE);
    if let Err(e) = std::fs::write(&marker, sha.trim()) {
        eprintln!(
            "lumen-app: could not write {}: {e} (model still usable, just won't detect future updates)",
            marker.display()
        );
    }
}

/// Read the SHA marker if present. Trims whitespace so future formats can
/// add trailing newlines / metadata lines without breaking older readers.
fn read_sha_marker(model_dir: &Path) -> Option<String> {
    std::fs::read_to_string(model_dir.join(SHA_MARKER_FILE))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
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
        // Multi-shard models advertise their shard set in
        // `model.safetensors.index.json`. A snapshot is only "ready" if every
        // listed shard actually exists on disk — otherwise the loader will
        // crash with a missing-tensor error mid-decode. This catches the case
        // where the in-app downloader is killed between shards or the user
        // closes the app before all shards finish.
        let shards_complete = verify_shards_complete(&scan_dir);
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
        let local_sha = read_sha_marker(&scan_dir);
        out.push(ModelEntry {
            id,
            path,
            size_bytes,
            ready: config_present && weight_present && shards_complete,
            supported: rec.is_some(),
            label: rec.map(|r| r.label.clone()),
            local_sha,
        });
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(out)
}

/// Verify every shard referenced by `model.safetensors.index.json` is
/// present on disk. Returns `true` when:
///   - no index file is present (single-shard model — the surrounding
///     `weight_present` check already covers this), OR
///   - the index parses cleanly AND every unique shard filename it lists
///     exists as a regular file in `scan_dir`.
///
/// The in-app downloader (see [`download`]) streams shards sequentially.
/// If it's killed between shards, the index.json (cheap, written first)
/// still claims every shard, but the later shards are missing on disk.
/// The MLX loader then crashes with a misleading "missing top-level
/// weight `<name>`" error mid-load — the actual root cause is the missing
/// `model-NN-of-NN.safetensors`. This check surfaces the truncation at
/// scan time so the desktop UI can flag the model as not-ready.
///
/// Does NOT validate individual file sizes — a shard that's present but
/// truncated will still slip through. Sufficient for the common
/// "downloader killed cleanly between shards" failure mode.
fn verify_shards_complete(scan_dir: &Path) -> bool {
    let index_path = scan_dir.join("model.safetensors.index.json");
    if !index_path.exists() {
        return true;
    }
    let Ok(text) = std::fs::read_to_string(&index_path) else {
        return false;
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) else {
        return false;
    };
    let Some(weight_map) = json.get("weight_map").and_then(|v| v.as_object()) else {
        return false;
    };
    let mut shards: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for v in weight_map.values() {
        if let Some(s) = v.as_str() {
            shards.insert(s);
        }
    }
    shards.iter().all(|shard| scan_dir.join(shard).is_file())
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

/// Latest commit SHA for the `main` branch of an HF Hub repo. Used to detect
/// "same repo id, new weights" the way `hsng95/gemma-4-26b-a4b-mlx-imatrix3plus-awq` was
/// rebuilt in v0.1.3 — the old broken-3bit and the new imatrix mixed-precision
/// build live at the same id, so byte-for-byte comparison would require
/// re-downloading.  The Hub commit SHA is the single 40-char string that
/// changes on every weight rebuild and is cheap to query (one HTTPS call).
///
/// API: `GET https://huggingface.co/api/models/<repo_id>` → JSON with `sha`.
/// Falls back to `None` on any error (offline, repo gone, etc.) — the caller
/// treats unknown remote SHA as "can't determine update status, allow use".
pub async fn fetch_hub_sha(
    client: &reqwest::Client,
    repo_id: &str,
) -> Result<String> {
    let url = format!("https://huggingface.co/api/models/{repo_id}");
    let resp = client
        .get(&url)
        .send()
        .await
        .with_context(|| format!("GET {url}"))?;
    let status = resp.status();
    if !status.is_success() {
        return Err(anyhow::anyhow!("HF API for {repo_id}: HTTP {status}"));
    }
    let body: serde_json::Value = resp
        .json()
        .await
        .with_context(|| format!("parse HF API json for {repo_id}"))?;
    body.get("sha")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow::anyhow!("HF API for {repo_id} missing sha field"))
}

/// Result of an update check for a single model. `needs_update` is true iff
/// both the local SHA and the remote SHA are known AND they differ.  When
/// either is unknown (legacy install / offline) we report `needs_update: false`
/// — silent, don't block the user.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateStatus {
    pub repo_id: String,
    pub local_sha: Option<String>,
    pub remote_sha: Option<String>,
    pub needs_update: bool,
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

    // Stamp the HF Hub commit SHA so future revision checks can detect when
    // the same repo id has been re-uploaded with different weights.  Best-
    // effort — `fetch_hub_sha` failure (offline finish, transient 5xx) just
    // skips the marker; the model is still usable, the user just won't see
    // an Update notification if the repo changes later.
    match fetch_hub_sha(&client, repo_id).await {
        Ok(sha) => write_sha_marker(&target, &sha),
        Err(e) => eprintln!(
            "lumen-app: could not fetch HF Hub SHA for {repo_id} post-download: {e} \
             (download succeeded; future update detection disabled for this install)"
        ),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("lumen-models-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn write_index(dir: &Path, shards: &[&str]) {
        let map: serde_json::Map<String, serde_json::Value> = shards
            .iter()
            .enumerate()
            .map(|(i, s)| (format!("tensor.{i}"), serde_json::Value::String((*s).to_string())))
            .collect();
        let json = serde_json::json!({ "metadata": {}, "weight_map": map });
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_string(&json).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn shards_complete_returns_true_when_no_index() {
        let dir = tmpdir("no-index");
        assert!(verify_shards_complete(&dir));
    }

    #[test]
    fn shards_complete_returns_true_when_all_shards_present() {
        let dir = tmpdir("all-present");
        write_index(&dir, &["shard-a.safetensors", "shard-b.safetensors"]);
        std::fs::write(dir.join("shard-a.safetensors"), b"x").unwrap();
        std::fs::write(dir.join("shard-b.safetensors"), b"y").unwrap();
        assert!(verify_shards_complete(&dir));
    }

    #[test]
    fn shards_complete_returns_false_when_shard_missing() {
        let dir = tmpdir("missing");
        write_index(&dir, &["s1.safetensors", "s2.safetensors", "s3.safetensors"]);
        std::fs::write(dir.join("s1.safetensors"), b"x").unwrap();
        // s2 + s3 missing — mirrors the killed-mid-download case
        assert!(!verify_shards_complete(&dir));
    }
}
