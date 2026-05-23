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
    // Qwen 2.5 omitted: mlx-native runner doesn't parse Qwen 2.5's flat
    // config.json shape (expects `text_config` nesting). Re-add when the
    // native loader gains a Qwen 2.5 path.
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
    // Qwen 2.5 entries are intentionally excluded from the recommended catalog:
    // the mlx-native runner (now the default backend) expects a nested
    // `text_config` object in `config.json` which only Gemma 4 and other
    // multimodal models ship.  Qwen 2.5's flat config fails parsing at load
    // time, so surfacing them as "recommended" misleads users into a broken
    // download.  Re-add once the native runner gains a Qwen 2.5 config path.
    RecommendedModel {
        id: "mlx-community/Qwen3.6-35B-A3B-mxfp4",
        family: ModelFamily::Qwen35Moe,
        label: "Qwen 3.6 — 35B A3B (MXFP4)",
        approx_size_gb: 19,
        min_ram_gb: 24,
        notes: "MoE flagship. 35B params, ~3B active per token. Best speed/quality on M3 Max.",
    },
    RecommendedModel {
        id: "mlx-community/gemma-4-26b-a4b-4bit",
        family: ModelFamily::Gemma4,
        label: "Gemma 4 — 26B A4B (4-bit, LM Studio community)",
        approx_size_gb: 14,
        min_ram_gb: 20,
        notes: "Gemma 4 26B MoE, ~4B active per token. 4-bit quant — balanced quality. Community build by LM Studio.",
    },
    // Lumen Gemma 4 26B-A4B 3-tier family — pick the tier matching the host RAM and
    // workload. Multi-angle eval (PPL × 4 corpora + 7 downstream tasks: MMLU, ARC,
    // HellaSwag, TruthfulQA, GSM8K, KMMLU, HAERAE) confirms each tier has a distinct
    // specialty: Standard wins English narrative + factual + CoT-math, Quality wins
    // broad knowledge + balanced, Flagship-KR wins Korean chat + lowest tulu PPL.
    RecommendedModel {
        id: "hsng95/gemma-4-26b-a4b-mlx-imatrix3plus-awq",
        family: ModelFamily::Gemma4,
        label: "Gemma 4 — 26B A4B Standard (HIGH=4, hsng95)",
        approx_size_gb: 12,
        min_ram_gb: 16,
        notes: "Standard tier — 12 GB, 3.916 bpw. Mixed 4/3-bit AFFINE imatrix + AWQ Option B (mean_sq), mlp_down groups dropped, embed_tokens at AFFINE 8-bit. Best for 24 GB Macs. Wins multi-angle eval on wikitext / TruthfulQA / GSM8K chain-of-thought. 11K Korean long-context CLEAN.",
    },
    RecommendedModel {
        id: "hsng95/gemma-4-26b-a4b-mlx-imatrix3plus-awq-high6",
        family: ModelFamily::Gemma4,
        label: "Gemma 4 — 26B A4B Quality (HIGH=6, hsng95)",
        approx_size_gb: 14,
        min_ram_gb: 24,
        notes: "Quality tier — 14 GB, 4.674 bpw. Top sensitivity tier elevated to 6-bit AFFINE on top of Standard recipe (Standard 3.916 bpw → +0.76 bpw for top-35% tensors). 3-seed mean Tulu PPL 62.68 (vs Standard 66.86, Δ −4.18). Wins multi-angle eval on MMLU / ARC / KMMLU — most balanced knowledge model. Recommended default for 32 GB+ Macs.",
    },
    RecommendedModel {
        id: "hsng95/gemma-4-26b-a4b-mlx-imatrix3plus-awq-high6-top40",
        family: ModelFamily::Gemma4,
        label: "Gemma 4 — 26B A4B Flagship-KR (HIGH=6 + top4=0.40, hsng95)",
        approx_size_gb: 15,
        min_ram_gb: 32,
        notes: "Flagship Korean tier — 15 GB, 5.057 bpw. HIGH=6 with top4_fraction=0.40 (40% tensors at 6-bit). 3-seed mean Tulu PPL 57.85 (vs Standard 66.86, Δ −9.01) — lowest-PPL MLX-loadable Gemma 4 26B-A4B build available. Wins HAERAE Korean knowledge (0.752) + tulu PPL. Korean chat flagship. Requires 36 GB+ for ≥8K context.",
    },
];

/// Recommended embedding models. Curated list of Qwen3-Embedding family —
/// shown in the API card of the desktop app; user downloads explicitly,
/// then selects from the downloaded set. Selection writes `EMBEDDING_MODEL_ID`
/// into the spawned server's env; when unset the `/v1/embeddings` endpoint
/// returns 503.
pub const EMBEDDINGS: &[RecommendedEmbedding] = &[
    RecommendedEmbedding {
        id: "mlx-community/Qwen3-Embedding-0.6B-8bit",
        label: "Qwen 3 Embedding — 0.6B (8-bit)",
        approx_size_gb: 1,
        min_ram_gb: 4,
        notes: "Small, fast embedding model. Good default for retrieval / semantic search. Runs on the fused 8-bit Metal kernel — packed weights stay GPU-resident (~600 MB).",
    },
    RecommendedEmbedding {
        id: "mlx-community/Qwen3-Embedding-4B-4bit-DWQ",
        label: "Qwen 3 Embedding — 4B (4-bit DWQ)",
        approx_size_gb: 3,
        min_ram_gb: 6,
        notes: "Mid-size, distilled-weight quant (DWQ). Noticeably better recall on long passages and multilingual queries vs 0.6B. Runs on the fused 4-bit Metal kernel (`affine4_qmv_fast_bf16in_bf16out`) — packed weights stay GPU-resident (~2.3 GB), no eager dequant.",
    },
    RecommendedEmbedding {
        id: "mlx-community/Qwen3-Embedding-8B-4bit-DWQ",
        label: "Qwen 3 Embedding — 8B (4-bit DWQ)",
        approx_size_gb: 5,
        min_ram_gb: 10,
        notes: "Top-of-family accuracy. Use when retrieval quality matters more than per-request latency. Fused 4-bit Metal kernel — packed weights stay GPU-resident (~4.3 GB), no eager dequant.",
    },
];

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
