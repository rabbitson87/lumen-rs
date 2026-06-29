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
        // Dense 27B with a built-in MTP self-speculative head (mtp/weights.safetensors
        // sidecar in the repo). 4-bit affine (group-64), ~16 GB. The dense trunk
        // gives higher MTP draft-accept than the 35B MoE, so self-spec decode nets
        // ~17.5 tok/s on M3 Max (≈19 with 3-bit requant via LUMEN_NATIVE_REQUANT_BITS=3).
        // The plain `Qwen/Qwen3.6-27B` base has NO mtp head → no speculative speedup,
        // so prefer this MTPLX build. Serve with MTP: native runner + LUMEN_QWEN35_MTP=1
        // + LUMEN_SPEC=mtp + LUMEN_QWEN35_HF_ORIGINAL=<this model dir>.
        id: "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed",
        family: ModelFamily::Qwen35Dense,
        label: "Qwen 3.6 — 27B Dense (MTPLX)",
        approx_size_gb: 16,
        min_ram_gb: 24,
        notes: "Dense 27B, 4-bit affine, with a built-in MTP self-speculative head. Stronger English / math / code than the 35B MoE; ~17.5 tok/s decode on M3 Max with MTP (≈19 at 3-bit). ~16 GB, needs ~24 GB RAM. Enable speculative decode: native runner + `LUMEN_QWEN35_MTP=1` + `LUMEN_SPEC=mtp` (point `LUMEN_QWEN35_HF_ORIGINAL` at the model dir, which holds the `mtp/` head).",
    },
    // Gemma 4 lineup — the NVFP4 base + two derived hybrids that share its fast
    // NVFP4 expert core. Built + measured 2026-06-02 (see the hsng95
    // `…-nvfp4-{smin,qmax}-mlx` model cards). NVFP4's finer per-group scales
    // preserve the channel-close token (id 101) so the whole trio reliably
    // enters AND exits thinking mode — unlike uniform affine 4-bit. Korean-first
    // specialty; for English fact / math / heavy code prefer the Qwen 3.6 entry.
    RecommendedModel {
        // The shared NVFP4 base (4-bit, group-16, E4M3 per-group scales),
        // instruction-tuned. The smin / qmax variants below derive from this by
        // re-quantizing select tensors. Fastest decode + best clean/Korean
        // perplexity of the family; same weights Ollama's `gemma4:26b-mlx` serves.
        id: "mlx-community/gemma-4-26b-a4b-it-nvfp4",
        family: ModelFamily::Gemma4,
        label: "Gemma 4 — Basic (Community NVFP4)",
        approx_size_gb: 16,
        min_ram_gb: 22,
        notes: "**Basic tier — base of the trio.** NVFP4 (4-bit, group-16, E4M3 scales) instruction-tuned build. Finer per-group scales preserve the thinking channel-close token, so it reliably enters *and exits* thinking mode (no affine-4bit infinite-reasoning failure). Fastest decode + best clean/Korean perplexity of the Gemma 4 family. Same weights Ollama serves for `gemma4:26b-mlx`. Pick this for a balanced default; pick Quality-Max or Size-Min below to trade toward output quality or footprint. English / math / heavy code remain weak across Gemma 4 — consider Qwen 3.6.",
    },
    RecommendedModel {
        // Quality-max derivation of the NVFP4 base: tied embed_tokens + the last
        // 13 layers' attention (q/k/v/o) re-quantized to 8-bit affine. Restores
        // input+output precision on the tied embed and late-layer hidden state.
        // Measured: perplexity 53.9 (vs base 56.5 — better) and the closest token
        // agreement of any variant (KL 0.96 / top-1 72.7% vs base). Decode 59.7
        // tok/s — far faster than the retired full-affine hybrid route (52.9).
        id: "hsng95/gemma-4-26b-a4b-nvfp4-qmax-mlx",
        family: ModelFamily::Gemma4,
        label: "Gemma 4 — Quality-Max (NVFP4 hybrid)",
        approx_size_gb: 16,
        min_ram_gb: 22,
        notes: "**Quality tier** — NVFP4 base + 8-bit tied embed_tokens + 8-bit attention on the last 13 layers. Lower perplexity than the base (53.9 vs 56.5) and the closest token agreement of any variant — a refined NVFP4. Best for: output quality, tool-calling / agentic robustness, thinking-mode tasks. GSM8K 84% (base 88%, within noise). ~16 GB, ~60 tok/s decode. Pair with `LUMEN_GEMMA4_GRAMMAR_LARK=1` for structural tool-call enforcement.",
    },
    RecommendedModel {
        // Size-min derivation: the MoE experts (the size bulk) re-quantized to
        // 3-bit affine + dense MLP to 4-bit. 12.5 GB (−20% vs base). Decode 77.9
        // tok/s — FASTER than the base (3-bit affine has no NVFP4 per-group
        // fp-scale matmul overhead). GSM8K 84% ≈ base 88%: the MoE top-8 routing
        // average absorbs the 3-bit expert reconstruction noise. Closer to the
        // base (KL 1.42) than stock affine it-4bit (1.95) despite being smaller.
        id: "hsng95/gemma-4-26b-a4b-nvfp4-smin-mlx",
        family: ModelFamily::Gemma4,
        label: "Gemma 4 — Size-Min (NVFP4 hybrid)",
        approx_size_gb: 12,
        min_ram_gb: 16,
        notes: "**Size / speed tier** — NVFP4 base with 3-bit MoE experts + 4-bit dense MLP. 12.5 GB and the fastest decode of the family (~78 tok/s, even faster than the base). GSM8K 84% (base 88%, within noise) — 3-bit expert noise absorbed by MoE top-8 routing; closer to the base than stock it-4bit despite 3 GB smaller. Best for: 16 GB Macs, max throughput, or running alongside other models. Korean + thinking-mode retained from the NVFP4 base.",
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
