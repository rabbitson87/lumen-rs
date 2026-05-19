//! Catalog mirror types — deserialize the JSON emitted by
//! `lumen-server --catalog` so the desktop app can filter the MODELS card
//! to "actually serveable" entries instead of letting users download
//! arbitrary HF repos.
//!
//! The lumen-server side (`crates/lumen-server/src/catalog.rs`) is the
//! single source of truth. These types mirror its `Serialize` output with
//! `Deserialize` impls on the desktop side.

use std::process::Command;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ModelFamily {
    Qwen25,
    Qwen35Dense,
    Qwen35Moe,
    Gemma4,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FamilyInfo {
    pub family: ModelFamily,
    pub label: String,
    pub backend: String,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendedModel {
    pub id: String,
    pub family: ModelFamily,
    pub label: String,
    pub approx_size_gb: u32,
    pub min_ram_gb: u32,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendedEmbedding {
    pub id: String,
    pub label: String,
    pub approx_size_gb: u32,
    pub min_ram_gb: u32,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Catalog {
    pub families: Vec<FamilyInfo>,
    pub recommended: Vec<RecommendedModel>,
    #[serde(default)]
    pub embeddings: Vec<RecommendedEmbedding>,
}

impl Catalog {
    /// Find the catalog entry matching this id, if any.
    ///
    /// Match strategy:
    /// 1. Exact id match (HF cache format: `mlx-community/gemma-4-26b-a4b-mlx-4bit`)
    /// 2. Trailing path component match (flat-dir convention: README's
    ///    `--local-dir ~/models/gemma-4-26b-a4b-mlx-4bit` strips the org prefix)
    /// 3. Case-insensitive variants of either of the above
    pub fn find_recommended(&self, id: &str) -> Option<&RecommendedModel> {
        if let Some(r) = self.recommended.iter().find(|r| r.id == id) {
            return Some(r);
        }
        let suffix = id.rsplit('/').next().unwrap_or(id);
        if let Some(r) = self
            .recommended
            .iter()
            .find(|r| r.id.rsplit('/').next().unwrap_or(&r.id) == suffix)
        {
            return Some(r);
        }
        let id_low = id.to_lowercase();
        let suf_low = suffix.to_lowercase();
        self.recommended.iter().find(|r| {
            let rid_low = r.id.to_lowercase();
            let rsuf_low = r.id.rsplit('/').next().unwrap_or(&r.id).to_lowercase();
            rid_low == id_low || rsuf_low == suf_low
        })
    }
}

/// Spawn `lumen-server --catalog` and parse the JSON it prints. Fast — the
/// server early-exits before any model load, env probing, or HTTP listener.
/// Total time on cold disk: ~50ms.
pub fn fetch(binary: &std::path::Path) -> Result<Catalog> {
    let out = Command::new(binary)
        .arg("--catalog")
        .output()
        .with_context(|| format!("spawn {} --catalog", binary.display()))?;
    if !out.status.success() {
        anyhow::bail!(
            "{} --catalog exited {}: {}",
            binary.display(),
            out.status,
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let cat: Catalog = serde_json::from_slice(&out.stdout)
        .context("parse catalog JSON from lumen-server --catalog")?;
    Ok(cat)
}
