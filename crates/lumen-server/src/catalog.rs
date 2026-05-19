//! Supported-model catalog.
//!
//! Single source of truth for "what can lumen-server actually serve" — both
//! the family enum and the curated recommended-model list. The lumen-app
//! desktop control plane fetches this via the `--catalog` CLI flag so the
//! MODELS card only surfaces serveable entries instead of letting the user
//! download arbitrary HF repos and hit obscure load errors later.
//!
//! Adding a new family:
//! 1. Add a variant to `ModelFamily`
//! 2. Append a `FamilyInfo` entry to `FAMILIES` (with backend + notes)
//! 3. Append `RecommendedModel` entries to `RECOMMENDED`
//! 4. Extend `family_of()` to detect repo-id patterns for that family
//! 5. Hook the actual loader into `engine::load()` (separate concern)

use serde::Serialize;

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ModelFamily {
    /// Qwen 2 / 2.5 dense via the Candle backend. Default fallback.
    Qwen25,
    /// Qwen 3.5 / 3.6 Dense (27B). mlx-native backend, requires `qwen3_5_moe`
    /// feature flag (the dense path shares the MoE loader).
    Qwen35Dense,
    /// Qwen 3.5 / 3.6 Mixture-of-Experts (A3B-MXFP4). mlx-native backend.
    Qwen35Moe,
    /// Gemma 4 (26B-A4B / E4B). mlx-native backend.
    Gemma4,
}

#[derive(Debug, Clone, Serialize)]
pub struct FamilyInfo {
    pub family: ModelFamily,
    pub label: &'static str,
    pub backend: &'static str,
    pub notes: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct RecommendedModel {
    pub id: &'static str,
    pub family: ModelFamily,
    pub label: &'static str,
    pub approx_size_gb: u32,
    pub min_ram_gb: u32,
    pub notes: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct RecommendedEmbedding {
    pub id: &'static str,
    pub label: &'static str,
    pub approx_size_gb: u32,
    pub min_ram_gb: u32,
    pub notes: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct Catalog {
    pub families: Vec<FamilyInfo>,
    pub recommended: Vec<RecommendedModel>,
    pub embeddings: Vec<RecommendedEmbedding>,
}

pub const FAMILIES: &[FamilyInfo] = &[
    FamilyInfo {
        family: ModelFamily::Qwen25,
        label: "Qwen 2.5",
        backend: "candle",
        notes: "Dense transformer family. Solid baseline; smallest sizes available.",
    },
    FamilyInfo {
        family: ModelFamily::Qwen35Dense,
        label: "Qwen 3.5/3.6 Dense",
        backend: "mlx-native",
        notes: "Dense variant of the Qwen 3 family (27B). Higher quality, more RAM.",
    },
    FamilyInfo {
        family: ModelFamily::Qwen35Moe,
        label: "Qwen 3.5/3.6 MoE",
        backend: "mlx-native",
        notes: "Mixture-of-Experts with ~A3B activation. Best quality-per-token on M-series.",
    },
    FamilyInfo {
        family: ModelFamily::Gemma4,
        label: "Gemma 4",
        backend: "mlx-native",
        notes: "Google Gemma 4 MoE family (26B-A4B / E4B).",
    },
];

pub const RECOMMENDED: &[RecommendedModel] = &[
    RecommendedModel {
        id: "Qwen/Qwen2.5-0.5B",
        family: ModelFamily::Qwen25,
        label: "Qwen 2.5 — 0.5B (base)",
        approx_size_gb: 1,
        min_ram_gb: 4,
        notes: "Smallest model — useful for smoke tests + low-RAM machines.",
    },
    RecommendedModel {
        id: "Qwen/Qwen2.5-1.5B-Instruct",
        family: ModelFamily::Qwen25,
        label: "Qwen 2.5 — 1.5B Instruct",
        approx_size_gb: 3,
        min_ram_gb: 8,
        notes: "Good quality-to-size ratio. Chat-tuned.",
    },
    RecommendedModel {
        id: "Qwen/Qwen2.5-3B-Instruct",
        family: ModelFamily::Qwen25,
        label: "Qwen 2.5 — 3B Instruct",
        approx_size_gb: 6,
        min_ram_gb: 12,
        notes: "Strong general-purpose baseline.",
    },
    RecommendedModel {
        id: "mlx-community/Qwen3.6-35B-A3B-mxfp4",
        family: ModelFamily::Qwen35Moe,
        label: "Qwen 3.6 — 35B A3B (MXFP4)",
        approx_size_gb: 19,
        min_ram_gb: 24,
        notes: "MoE flagship. 35B params, ~3B active per token. Best speed/quality on M3 Max.",
    },
    RecommendedModel {
        id: "mlx-community/gemma-4-26b-a4b-mlx-4bit",
        family: ModelFamily::Gemma4,
        label: "Gemma 4 — 26B A4B (4-bit)",
        approx_size_gb: 14,
        min_ram_gb: 20,
        notes: "Gemma 4 26B MoE, ~4B active per token. 4-bit quant — balanced quality.",
    },
    RecommendedModel {
        id: "mlx-community/gemma-4-26b-a4b-mlx-3bit",
        family: ModelFamily::Gemma4,
        label: "Gemma 4 — 26B A4B (3-bit)",
        approx_size_gb: 11,
        min_ram_gb: 16,
        notes: "Same Gemma 4 26B but 3-bit — fits comfortably in 16 GB Mac mini / MacBook Air.",
    },
];

/// Recommended embedding models. Currently a small curated list — selected
/// in the OpenAI tab of the desktop app's API card, sets `EMBEDDING_MODEL_ID`
/// at subprocess spawn. When empty the `/v1/embeddings` endpoint returns 503.
pub const EMBEDDINGS: &[RecommendedEmbedding] = &[RecommendedEmbedding {
    id: "mlx-community/Qwen3-Embedding-0.6B-8bit",
    label: "Qwen 3 Embedding — 0.6B (8-bit)",
    approx_size_gb: 1,
    min_ram_gb: 4,
    notes: "Small, fast embedding model. Good default for retrieval / semantic search.",
}];

/// Detect which family a repo id / local dir name belongs to. Mirrors the
/// detection logic in `engine::detect_architecture` but maps to the catalog
/// enum (one variant per loader path) — the engine's `"qwen2"` fallback is
/// excluded because we don't want arbitrary repo ids slipping through as
/// "supported".
pub fn family_of(model_id: &str) -> Option<ModelFamily> {
    let lower = model_id.to_lowercase();

    if is_qwen3_5_dense(&lower) {
        return Some(ModelFamily::Qwen35Dense);
    }
    if lower.contains("qwen3.6")
        || lower.contains("qwen3_5")
        || lower.contains("qwen3-next")
        || lower.contains("a3b-mxfp4")
    {
        return Some(ModelFamily::Qwen35Moe);
    }
    if is_gemma4(&lower) {
        return Some(ModelFamily::Gemma4);
    }
    // Strict match — don't accept arbitrary IDs as "Qwen 2.5". Must contain
    // the explicit version marker.
    if lower.contains("qwen2.5") || lower.contains("qwen2_5") {
        return Some(ModelFamily::Qwen25);
    }
    None
}

fn is_gemma4(lower_id: &str) -> bool {
    lower_id.contains("gemma-4")
        || lower_id.contains("gemma4-")
        || lower_id.contains("gemma_4")
        || lower_id.contains("gemma4_")
}

fn is_qwen3_5_dense(lower_id: &str) -> bool {
    let qwen35_family = lower_id.contains("qwen3.6") || lower_id.contains("qwen3_5");
    if !qwen35_family {
        return false;
    }
    if lower_id.contains("-dense") || lower_id.contains("_dense") {
        return true;
    }
    if lower_id.contains("a3b") || lower_id.contains("moe") {
        return false;
    }
    lower_id.contains("27b") || lower_id.contains("-27-")
}

pub fn catalog() -> Catalog {
    Catalog {
        families: FAMILIES.to_vec(),
        recommended: RECOMMENDED.to_vec(),
        embeddings: EMBEDDINGS.to_vec(),
    }
}
