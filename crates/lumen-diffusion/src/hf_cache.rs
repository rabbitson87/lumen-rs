//! Resolve HuggingFace Hub snapshot directories from the local cache.
//!
//! The FLUX.2 pipeline assembles its components from several HF repos. Rather
//! than hardcode machine-specific snapshot paths (`~/.cache/huggingface/hub/
//! models--org--repo/snapshots/<hash>/…`, which pin a commit hash and leak the
//! developer's home dir), we resolve them at runtime from the standard
//! `huggingface_hub` cache layout. Callers fall back to a clear load error if a
//! repo isn't downloaded, and every path is overridable via env in the pipeline.
//!
//! Pure Rust (no MLX) — compiled unconditionally.

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// HF cache hub root: `$HF_HOME/hub` if set, else `~/.cache/huggingface/hub`.
fn hub_root() -> Option<PathBuf> {
    if let Some(h) = std::env::var_os("HF_HOME") {
        return Some(PathBuf::from(h).join("hub"));
    }
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/huggingface/hub"))
}

/// Cache directory name for a repo id: `org/repo` → `models--org--repo`.
fn repo_cache_name(repo_id: &str) -> String {
    format!("models--{}", repo_id.replace('/', "--"))
}

/// Newest local snapshot directory for `repo_id`, if the repo is downloaded.
///
/// Scans `models--<org>--<repo>/snapshots/<hash>/` and returns the most
/// recently modified snapshot (handles re-downloads / multiple revisions).
pub fn snapshot_dir(repo_id: &str) -> Option<PathBuf> {
    let snaps = hub_root()?.join(repo_cache_name(repo_id)).join("snapshots");
    let mut best: Option<(SystemTime, PathBuf)> = None;
    for entry in std::fs::read_dir(&snaps).ok()?.flatten() {
        let p = entry.path();
        if !p.is_dir() {
            continue;
        }
        let mtime = entry
            .metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .unwrap_or(UNIX_EPOCH);
        if best.as_ref().is_none_or(|(t, _)| mtime >= *t) {
            best = Some((mtime, p));
        }
    }
    best.map(|(_, p)| p)
}

/// Resolve `<repo snapshot>/<rel>` if the repo is in the local cache.
pub fn snapshot_path(repo_id: &str, rel: &str) -> Option<PathBuf> {
    snapshot_dir(repo_id).map(|d| d.join(rel))
}

/// Resolve a repo's snapshot dir, or — if not downloaded — return the repo id
/// as a `PathBuf` so the subsequent load fails with a clear "no such directory:
/// <repo id>" error instead of a leaked absolute path.
pub fn snapshot_dir_or_id(repo_id: &str) -> PathBuf {
    snapshot_dir(repo_id).unwrap_or_else(|| PathBuf::from(repo_id))
}

/// Like [`snapshot_dir_or_id`] but for a file within the snapshot.
pub fn snapshot_path_or_rel(repo_id: &str, rel: &str) -> PathBuf {
    snapshot_path(repo_id, rel).unwrap_or_else(|| PathBuf::from(repo_id).join(rel))
}
