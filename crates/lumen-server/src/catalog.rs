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
    // Gemma 4 lineup — two-tier, workload-anchored:
    //   • Chat (mlx-community 4-bit) — general-purpose Korean chat / direct Q&A;
    //     weak channel-token weights, can't reliably enter thinking mode.
    //   • Agent Lite (hsng95 imatrix3plus-awq HIGH=4) — Korean agentic loops
    //     (tool calls, multi-step reasoning); imatrix-AWQ amplifies channel
    //     tokens so client must opt into thinking explicitly.
    // Both ship are Korean-first specialty — English fact / math / code are
    // weak across the family; recommend Qwen 3 family for those workloads.
    RecommendedModel {
        // CRITICAL: `-it-` (instruction-tuned) variant. The non-`it` repo
        // `mlx-community/gemma-4-26b-a4b-4bit` is the BASE pretrain model —
        // it has no instruction-following ability and produces self-completion
        // loops on chat-template prompts (next-token LM behavior, not assistant
        // behavior). Always pick the `-it-` repo for chat use.
        id: "mlx-community/gemma-4-26b-a4b-it-4bit",
        family: ModelFamily::Gemma4,
        label: "Gemma 4 — Chat (Community IT 4-bit)",
        approx_size_gb: 14,
        min_ram_gb: 20,
        notes: "**Chat tier** — uniform 4-bit community build of Gemma 4's instruction-tuned (`it`) variant. Best for: Korean general chat, FAQ, simple Q&A. Not for: tool calling, multi-step reasoning, long agentic loops (channel-token weights too weak to enter thinking mode reliably). English / math / code: weak across all Gemma 4 ships — consider Qwen 3.6 instead.",
    },
    RecommendedModel {
        id: "hsng95/gemma-4-26b-a4b-mlx-imatrix3plus-awq",
        family: ModelFamily::Gemma4,
        label: "Gemma 4 — Agent Lite (12 GB)",
        approx_size_gb: 12,
        min_ram_gb: 16,
        notes: "**Agent tier** — 12 GB, 3.916 bpw mixed 4/3-bit AFFINE imatrix + AWQ. Best for: Korean agentic workloads (tool calls, multi-step reasoning, structured outputs). Always defaults to thinking-mode generation — client must send `chat_template_kwargs.enable_thinking=true` (or `reasoning_effort` / `thinking: true`) to activate. Not for: casual chat (overkill), English fact-checking, math CoT — Korean-first specialty. Best for 24 GB Macs.",
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
