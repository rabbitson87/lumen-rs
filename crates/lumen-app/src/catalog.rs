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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendedImageModel {
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
    /// Text-to-image (diffusion) models. Served in `LUMEN_SERVE=image` mode via
    /// `POST /v1/images/generations`. `#[serde(default)]` keeps older servers
    /// (no `image_models` field) deserializing cleanly.
    #[serde(default)]
    pub image_models: Vec<RecommendedImageModel>,
}

impl Catalog {
    /// Find the catalog entry matching this id, if any.
    ///
    /// Match strategy (most specific → most permissive):
    /// 1. Exact id match
    /// 2. Trailing path component match
    /// 3. Case-insensitive variants of (1) or (2)
    /// 4. **Canonical match** — `-mlx-` / `mlx-` / `-mlx` tokens stripped on
    ///    both sides. Catches naming drift: local cache
    ///    `gemma-4-26b-a4b-mlx-4bit` ↔ canonical HF id
    ///    `mlx-community/gemma-4-26b-a4b-4bit` (org prefix already says "mlx",
    ///    so the infix is redundant and the canonical repo omits it).
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
        if let Some(r) = self.recommended.iter().find(|r| {
            let rid_low = r.id.to_lowercase();
            let rsuf_low = r.id.rsplit('/').next().unwrap_or(&r.id).to_lowercase();
            rid_low == id_low || rsuf_low == suf_low
        }) {
            return Some(r);
        }
        let id_canon = canonical_name(&id_low);
        let suf_canon = canonical_name(&suf_low);
        self.recommended.iter().find(|r| {
            let rid_canon = canonical_name(&r.id.to_lowercase());
            let rsuf_canon =
                canonical_name(&r.id.rsplit('/').next().unwrap_or(&r.id).to_lowercase());
            rid_canon == id_canon || rsuf_canon == suf_canon
        })
    }

    /// Find a text-to-image model entry by id (exact or trailing-component).
    pub fn find_image_model(&self, id: &str) -> Option<&RecommendedImageModel> {
        let suffix = id.rsplit('/').next().unwrap_or(id);
        self.image_models
            .iter()
            .find(|m| m.id == id || m.id.rsplit('/').next().unwrap_or(&m.id) == suffix)
    }

    /// Whether the given active-model id refers to a diffusion image model
    /// (→ launch the server in `LUMEN_SERVE=image` mode).
    pub fn is_image_model(&self, id: &str) -> bool {
        self.find_image_model(id).is_some()
    }
}

/// Normalize away the optional `-mlx-` / `mlx-` / `-mlx` tokens that some
/// repos sprinkle into MLX-quantized names. Lets us match a local cache
/// like `gemma-4-26b-a4b-mlx-4bit` against catalog id
/// `mlx-community/gemma-4-26b-a4b-4bit` (and the inverse) without listing
/// every spelling in the catalog.
fn canonical_name(s: &str) -> String {
    s.replace("-mlx-", "-")
        .replace("mlx-", "")
        .replace("-mlx", "")
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
