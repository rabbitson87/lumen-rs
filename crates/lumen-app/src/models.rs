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
/// HuggingFace Hub repo id grammar: `<org>/<repo>` where each segment is
/// `[A-Za-z0-9._-]+`. Rejects empty parts, missing slash, multiple slashes,
/// and any character HF wouldn't accept in a repo URL — notably whitespace
/// (Finder's "Item 2" duplicate convention) which would 401 every update
/// check and never resolve to a real Hub URL.
fn is_valid_hf_repo_id(id: &str) -> bool {
    let mut parts = id.split('/');
    let (Some(org), Some(repo), None) = (parts.next(), parts.next(), parts.next()) else {
        return false;
    };
    !org.is_empty() && !repo.is_empty() && is_hf_segment(org) && is_hf_segment(repo)
}

/// Plain flat (no-org) local-only model directory name. Same character class
/// as an HF segment — no whitespace, no slash. Skips Finder duplicates and
/// scratch dirs that would otherwise surface as bogus model entries.
fn is_valid_flat_id(name: &str) -> bool {
    !name.is_empty() && is_hf_segment(name)
}

fn is_hf_segment(s: &str) -> bool {
    s.chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-')
}

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
            if !is_valid_hf_repo_id(&id) {
                continue;
            }
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
            // Strict HF naming check filters Finder duplicates (`… 2`),
            // .DS_Store siblings, and anything else that wouldn't round-trip
            // to a valid Hub URL. Avoids spamming HF API with 401s on the
            // periodic update check.
            if !is_valid_hf_repo_id(&id) {
                continue;
            }
            (id, path.clone())
        } else {
            // Plain flat (README convention, no org prefix in dir name).
            // Skip names with whitespace or other non-HF-id characters —
            // these are almost always Finder duplicates or scratch dirs,
            // never real local-only models.
            if !is_valid_flat_id(&name) {
                continue;
            }
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

/// Parse a `.safetensors` file's header and return the EXPECTED total file
/// size in bytes (8-byte length prefix + JSON header + tensor body). Reads
/// ONLY the prefix + header (typically < 1 MB) — the body is never touched.
///
/// safetensors layout:
///   [u64 LE: header_len][JSON header of header_len bytes][raw tensor body]
///
/// The JSON header maps each tensor name to its `data_offsets: [start, end]`
/// **relative to the body start** (i.e. byte 8 + header_len of the file).
/// The body's total length is therefore `max(end over all tensors)`.
///
/// Returns `None` on any I/O / parse error so the caller can fall back to
/// presence-only checks.
fn expected_safetensors_size(path: &Path) -> Option<u64> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut header_len_buf = [0u8; 8];
    f.read_exact(&mut header_len_buf).ok()?;
    let header_len = u64::from_le_bytes(header_len_buf);
    // Sanity bounds: real safetensors headers are KB-MB range. Reject
    // anything < 2 (invalid JSON) or > 100 MB (likely truncated /
    // corrupted file producing a bogus length).
    if header_len < 2 || header_len > 100 * 1024 * 1024 {
        return None;
    }
    let mut header_buf = vec![0u8; header_len as usize];
    f.read_exact(&mut header_buf).ok()?;
    let json: serde_json::Value = serde_json::from_slice(&header_buf).ok()?;
    let obj = json.as_object()?;
    let mut max_end: u64 = 0;
    for (key, v) in obj {
        // `__metadata__` is an opaque string-keyed dict, not a tensor.
        if key == "__metadata__" {
            continue;
        }
        let offsets = v.get("data_offsets")?.as_array()?;
        if offsets.len() != 2 {
            return None;
        }
        let end = offsets[1].as_u64()?;
        if end > max_end {
            max_end = end;
        }
    }
    Some(8 + header_len + max_end)
}

/// Verify every shard referenced by `model.safetensors.index.json` is
/// present on disk AND has the byte length its safetensors header
/// promises. Also rejects any directory that still contains in-progress
/// `*.part` files (the new downloader's atomic-rename sentinel).
///
/// Returns `true` when:
///   - the index exists, parses cleanly, every shard exists, every
///     shard's actual size matches `expected_safetensors_size`, AND no
///     `.part` files linger, OR
///   - no index file is present (single-shard / non-safetensors model
///     — covered by `verify_no_part_files` + the surrounding
///     `weight_present` check), AND no `.part` files linger
///
/// The in-app downloader (see [`download`]) streams shards sequentially.
/// Failure modes this catches:
///   1. Killed BETWEEN shards: later shard files are missing → not
///      ready (also caught by the legacy check this replaces).
///   2. Killed MID shard: previously the truncated 1.8 GB shard was
///      treated as "ready" because the filename existed. Now the
///      header-vs-actual size check rejects it, AND the new
///      `<shard>.part` sidecar makes the in-flight state explicit so
///      a clean re-launch re-downloads.
fn verify_shards_complete(scan_dir: &Path) -> bool {
    if !verify_no_part_files(scan_dir) {
        return false;
    }
    // Filename-pattern check FIRST — covers the case where
    // `model.safetensors.index.json` itself never finished downloading
    // but some shards already landed. Returns Some(false) if shards
    // exist but the `MMMMM` total claims more than are present.
    if let Some(filename_complete) = verify_shard_filename_completeness(scan_dir) {
        if !filename_complete {
            return false;
        }
    }
    let index_path = scan_dir.join("model.safetensors.index.json");
    if !index_path.exists() {
        // Non-sharded: still validate any standalone `model.safetensors`
        // against its own header so a single-file truncation is caught.
        let single = scan_dir.join("model.safetensors");
        if single.exists() {
            return safetensors_size_matches(&single);
        }
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
    shards.iter().all(|shard| {
        let path = scan_dir.join(shard);
        path.is_file() && safetensors_size_matches(&path)
    })
}

/// Compare a `.safetensors` file's on-disk size with the size its header
/// promises. Returns `true` when they match exactly, OR when the header
/// can't be parsed (we don't want to false-positive on legitimately
/// unusual / future formats — the loader will catch real corruption).
fn safetensors_size_matches(path: &Path) -> bool {
    let Ok(meta) = std::fs::metadata(path) else {
        return false;
    };
    let actual = meta.len();
    match expected_safetensors_size(path) {
        Some(expected) => actual == expected,
        // Header unreadable. Could be a truncation so severe it can't
        // even hold an 8-byte length prefix, or a non-safetensors file
        // that happens to be named `.safetensors`. Conservative: only
        // reject when the file is implausibly small (< 1 KB — every
        // real safetensors has at least an 8-byte prefix + a non-empty
        // JSON header that easily exceeds 1 KB for multi-tensor files).
        None => actual >= 1024,
    }
}

/// Reject directories containing any `*.part` file (the new download
/// sentinel — see [`download`]). A `.part` file means a previous
/// download was killed mid-write and the corresponding final file is
/// missing or incomplete.
fn verify_no_part_files(scan_dir: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(scan_dir) else {
        return true; // Don't reject on missing dir — surrounding code handles that.
    };
    for e in entries.flatten() {
        if let Some(name) = e.file_name().to_str() {
            if name.ends_with(".part") {
                return false;
            }
        }
    }
    true
}

/// Catch the failure mode where the downloader was killed BEFORE writing
/// `model.safetensors.index.json` but some shards already landed. Without
/// the index file, `verify_shards_complete` can't tell that shards are
/// missing — but the surviving shards' filenames advertise the total
/// shard count via the canonical HF pattern `model-NNNNN-of-MMMMM.safetensors`.
///
/// This pulls `MMMMM` out of any shard's filename and verifies that all
/// `NNNNN` from 1 to `MMMMM` exist on disk. Returns:
///   - `Some(true)` if a complete shard set is present.
///   - `Some(false)` if shards exist but are missing some sequence
///     numbers (the truncation case).
///   - `None` if NO sharded filenames are present (caller should fall
///     back to single-file checks).
fn verify_shard_filename_completeness(scan_dir: &Path) -> Option<bool> {
    use std::collections::BTreeSet;
    let entries = std::fs::read_dir(scan_dir).ok()?;
    let mut seen_total: Option<u32> = None;
    let mut present_indices: BTreeSet<u32> = BTreeSet::new();
    for e in entries.flatten() {
        let Some(name) = e.file_name().to_str().map(|s| s.to_string()) else {
            continue;
        };
        // Match `model-NNNNN-of-MMMMM.safetensors` (also tolerate
        // 1-6 digit widths since some HF repos pad differently).
        let Some(rest) = name
            .strip_prefix("model-")
            .and_then(|s| s.strip_suffix(".safetensors"))
        else {
            continue;
        };
        let parts: Vec<&str> = rest.splitn(2, "-of-").collect();
        if parts.len() != 2 {
            continue;
        }
        let (Ok(n), Ok(m)) = (parts[0].parse::<u32>(), parts[1].parse::<u32>()) else {
            continue;
        };
        if m == 0 {
            continue;
        }
        match seen_total {
            Some(prev) if prev != m => {
                // Inconsistent `-of-MMMMM` across shards — likely a
                // mix-up. Treat as broken.
                return Some(false);
            }
            _ => seen_total = Some(m),
        }
        present_indices.insert(n);
    }
    let total = seen_total?;
    // HF shard numbering is 1-based, contiguous.
    for i in 1..=total {
        if !present_indices.contains(&i) {
            return Some(false);
        }
    }
    Some(true)
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

/// Build a `reqwest::Client` configured for HuggingFace Hub calls.
///
/// Attaches `Authorization: Bearer <token>` automatically when one of the
/// well-known HF token env vars is set:
///   - `HF_TOKEN` (preferred; matches `huggingface_hub` Python lib + the
///     `huggingface-cli login` convention)
///   - `HUGGING_FACE_HUB_TOKEN` (legacy fallback)
///
/// Without a token the client behaves identically to before — anonymous
/// access works for any non-gated repo.
///
/// Single source of truth so `fetch_hub_sha`, `download`, and the
/// periodic `check_model_updates` path all auth-up consistently. Adding a
/// new HTTP call site? Use this builder, don't roll your own.
pub fn hf_client(timeout: Option<std::time::Duration>) -> Result<reqwest::Client> {
    let token = std::env::var("HF_TOKEN")
        .ok()
        .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok())
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty());
    let mut builder =
        reqwest::Client::builder().user_agent(concat!("lumen-app/", env!("CARGO_PKG_VERSION")));
    if let Some(d) = timeout {
        builder = builder.timeout(d);
    }
    if let Some(t) = &token {
        use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
        let mut headers = HeaderMap::new();
        let mut val = HeaderValue::try_from(format!("Bearer {t}"))
            .context("invalid HF_TOKEN — must be ASCII (no newline / non-printable bytes)")?;
        val.set_sensitive(true);
        headers.insert(AUTHORIZATION, val);
        builder = builder.default_headers(headers);
    }
    builder.build().context("init hf http client")
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
pub async fn fetch_hub_sha(client: &reqwest::Client, repo_id: &str) -> Result<String> {
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
    let client = hf_client(None).context("init hf http client for download")?;

    let target = local_path_for(models_dir, repo_id);
    std::fs::create_dir_all(&target)
        .with_context(|| format!("create_dir_all {}", target.display()))?;

    let resolve_url =
        |file: &str| format!("https://huggingface.co/{}/resolve/main/{}", repo_id, file);

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
                let v: serde_json::Value =
                    serde_json::from_str(&idx_text).context("parse safetensors index")?;
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
        let part = target.join(format!("{file}.part"));

        // Skip the network round-trip only when the destination file
        // ALREADY exists at its full advertised size. Truncated files
        // (downloader killed mid-write before .part rename existed)
        // are detected here by HEAD-ing the remote size and comparing
        // — mismatched local size triggers re-download.
        if let Ok(meta) = std::fs::metadata(&dest) {
            if meta.len() > 0 {
                let head = client
                    .head(&url)
                    .send()
                    .await
                    .with_context(|| format!("HEAD {url}"))?;
                let expected = head.content_length();
                let local_ok = match expected {
                    Some(n) => meta.len() == n,
                    // HEAD didn't report size (rare on HF) — fall back to
                    // safetensors header self-check for known shard files.
                    None => {
                        if file.ends_with(".safetensors") {
                            safetensors_size_matches(&dest)
                        } else {
                            true
                        }
                    }
                };
                if local_ok {
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
                // Truncated — drop the bogus file so the loop below
                // can re-fetch from scratch.
                eprintln!(
                    "[lumen-app] {} is {} bytes but remote reports {} — re-downloading",
                    dest.display(),
                    meta.len(),
                    expected
                        .map(|n| n.to_string())
                        .unwrap_or_else(|| "?".into()),
                );
                let _ = std::fs::remove_file(&dest);
            }
        }

        // Clean up any stale `.part` from a prior killed run so we
        // restart from byte 0. (Real Range/resume support is a later
        // enhancement — for now, the safer simpler thing is to throw
        // away the partial bytes and re-download cleanly.)
        let _ = std::fs::remove_file(&part);

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

        let mut file_w = tokio::fs::File::create(&part)
            .await
            .with_context(|| format!("create {}", part.display()))?;
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
                .with_context(|| format!("write {}", part.display()))?;
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
            .with_context(|| format!("flush {}", part.display()))?;
        // Drop the handle BEFORE the rename so Windows-style locks
        // (no-op on macOS / Linux but cheap insurance) don't bite.
        drop(file_w);

        // Post-write sanity: if the server told us the content length,
        // verify the bytes on disk match before promoting to the final
        // name. A short read (server dropped connection silently) gets
        // surfaced here instead of leaking a truncated file under the
        // canonical name.
        if let Some(expected) = total {
            if downloaded != expected {
                let _ = std::fs::remove_file(&part);
                return Err(anyhow::anyhow!(
                    "short read for {file}: got {downloaded} bytes, expected {expected}"
                ));
            }
        }

        // Atomic promotion: `.part` → final filename. On any sane FS
        // (HFS+ / APFS / ext4) this is a single inode rename, so a
        // crash between flush and rename leaves the `.part` for the
        // scan to detect — never a half-named final file.
        std::fs::rename(&part, &dest)
            .with_context(|| format!("rename {} -> {}", part.display(), dest.display()))?;

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
    use std::io::Write;

    fn tmpdir(name: &str) -> PathBuf {
        let p =
            std::env::temp_dir().join(format!("lumen-models-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn write_index(dir: &Path, shards: &[&str]) {
        let map: serde_json::Map<String, serde_json::Value> = shards
            .iter()
            .enumerate()
            .map(|(i, s)| {
                (
                    format!("tensor.{i}"),
                    serde_json::Value::String((*s).to_string()),
                )
            })
            .collect();
        let json = serde_json::json!({ "metadata": {}, "weight_map": map });
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_string(&json).unwrap(),
        )
        .unwrap();
    }

    /// Write a syntactically-valid (but tiny) safetensors file whose
    /// header advertises a `body_len`-byte body. Used to fabricate
    /// both "well-formed" and "truncated" shards in tests without
    /// pulling in the real `safetensors` crate.
    fn write_safetensors(path: &Path, body_len: u64) {
        let header_json = format!(
            "{{\"t\":{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[0,{}]}}}}",
            body_len / 4,
            body_len,
        );
        let header_bytes = header_json.as_bytes();
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        f.write_all(header_bytes).unwrap();
        f.write_all(&vec![0u8; body_len as usize]).unwrap();
    }

    fn write_truncated_safetensors(path: &Path, advertised_body: u64, actual_body: u64) {
        let header_json = format!(
            "{{\"t\":{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[0,{}]}}}}",
            advertised_body / 4,
            advertised_body,
        );
        let header_bytes = header_json.as_bytes();
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        f.write_all(header_bytes).unwrap();
        f.write_all(&vec![0u8; actual_body as usize]).unwrap();
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
        write_safetensors(&dir.join("shard-a.safetensors"), 4096);
        write_safetensors(&dir.join("shard-b.safetensors"), 4096);
        assert!(verify_shards_complete(&dir));
    }

    #[test]
    fn shards_complete_returns_false_when_shard_missing() {
        let dir = tmpdir("missing");
        write_index(
            &dir,
            &["s1.safetensors", "s2.safetensors", "s3.safetensors"],
        );
        write_safetensors(&dir.join("s1.safetensors"), 4096);
        // s2 + s3 missing — mirrors the killed-between-shards case
        assert!(!verify_shards_complete(&dir));
    }

    #[test]
    fn shards_complete_returns_false_when_shard_truncated() {
        // The exact scenario the user reported: shard file exists on
        // disk, has substantial bytes, but is short of what its own
        // safetensors header claims (e.g. 1.8 GB of a 5 GB shard).
        let dir = tmpdir("truncated");
        write_index(&dir, &["only.safetensors"]);
        write_truncated_safetensors(&dir.join("only.safetensors"), 4096, 1024);
        assert!(!verify_shards_complete(&dir));
    }

    #[test]
    fn shards_complete_returns_false_when_part_file_present() {
        let dir = tmpdir("part-sentinel");
        write_index(&dir, &["a.safetensors"]);
        write_safetensors(&dir.join("a.safetensors"), 4096);
        // Stale `.part` from an interrupted run.
        std::fs::write(dir.join("a.safetensors.part"), b"in-progress").unwrap();
        assert!(!verify_shards_complete(&dir));
    }

    #[test]
    fn shards_complete_catches_missing_index_with_partial_shards() {
        // index.json never finished downloading, but the model dir has
        // 2/8 advertised shards. Filename pattern reveals MMMMM=8.
        let dir = tmpdir("no-index-partial");
        write_safetensors(&dir.join("model-00001-of-00008.safetensors"), 4096);
        write_safetensors(&dir.join("model-00002-of-00008.safetensors"), 4096);
        assert!(!verify_shards_complete(&dir));
    }

    #[test]
    fn shards_complete_passes_complete_shard_set_without_index() {
        // All 3 shards present, no index file (legacy / partial download
        // of optional metadata). Filename pattern says MMMMM=3 → ready.
        let dir = tmpdir("no-index-complete");
        write_safetensors(&dir.join("model-00001-of-00003.safetensors"), 4096);
        write_safetensors(&dir.join("model-00002-of-00003.safetensors"), 4096);
        write_safetensors(&dir.join("model-00003-of-00003.safetensors"), 4096);
        assert!(verify_shards_complete(&dir));
    }

    #[test]
    fn expected_safetensors_size_matches_actual_for_valid_file() {
        let dir = tmpdir("expected-size");
        let path = dir.join("t.safetensors");
        write_safetensors(&path, 8192);
        let expected = expected_safetensors_size(&path).expect("header parse");
        let actual = std::fs::metadata(&path).unwrap().len();
        assert_eq!(expected, actual);
    }

    #[test]
    fn expected_safetensors_size_detects_truncation() {
        let dir = tmpdir("expected-size-trunc");
        let path = dir.join("t.safetensors");
        write_truncated_safetensors(&path, 8192, 2048);
        let expected = expected_safetensors_size(&path).expect("header parse");
        let actual = std::fs::metadata(&path).unwrap().len();
        assert_ne!(expected, actual);
        assert!(!safetensors_size_matches(&path));
    }
}
