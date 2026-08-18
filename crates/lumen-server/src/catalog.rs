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

/// Recommended text-to-image (diffusion) model. Unlike chat/embedding models
/// these are served via `POST /v1/images/generations` with the server launched
/// in `LUMEN_SERVE=image` mode (the diffusion pipeline cannot co-reside with an
/// LLM on a 36 GB Mac). The desktop app surfaces these in their own card and
/// launches the server in image mode when one is selected.
#[derive(Debug, Clone, Serialize)]
pub struct RecommendedImageModel {
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
    pub image_models: Vec<RecommendedImageModel>,
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
    RecommendedModel {
        // Qwen3.8's trunk is the SAME architecture as the 3.6-27B above — its
        // `text_config` is byte-identical (64 layers, hidden 5120,
        // full_attention_interval 4, same linear-attn dims, same vision tower),
        // and the MLX build carries the same tensor names. It is a newer, more
        // capable generation on that backbone, not a new model to port.
        //
        // Heavier on disk than the 3.6 row despite the same shape: this build is
        // 4-bit group-32 with per-tensor 8-bit group-64 on lm_head, embed_tokens,
        // every linear_attn.out_proj and the last layers' MLP — so 21 GB, not 16,
        // and it wants ~32 GB of RAM rather than 24.
        //
        // MTP measured on this checkpoint (M3 Max, K=2, greedy, 320 tokens):
        // accept 0.494, output bit-exact against the non-MTP baseline (320/320),
        // roughly 2x the baseline decode rate. Needs no env vars — the head
        // auto-enables from the repo's own `mtp.safetensors`.
        id: "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed",
        family: ModelFamily::Qwen35Dense,
        label: "Qwen 3.8 — 27B Dense (MTPLX)",
        approx_size_gb: 21,
        min_ram_gb: 32,
        notes: "**Newest Qwen dense.** Same 27B hybrid backbone as the 3.6 entry (Gated DeltaNet linear-attn + 1-in-4 full attention, 262k context, vision tower) with a stronger 3.8-generation checkpoint — Qwen reports large agentic/coding gains over 3.6-27B. Ships a built-in MTP self-speculative head that auto-enables with **no env vars**: measured accept 0.494 at K=2 and ~2× baseline decode on M3 Max, with output bit-exact against the non-MTP path (320/320 tokens). Mixed precision (4-bit group-32 + 8-bit group-64 on lm_head / embeddings / attention out-projections / late MLP), so it is larger than the 3.6 build: ~21 GB, wants ~32 GB RAM. Prefer the 3.6-27B row on a 24 GB machine.",
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

/// Recommended text-to-image models. Served via `POST /v1/images/generations`
/// (OpenAI-compatible) with the server launched in `LUMEN_SERVE=image` mode.
/// FLUX.2-dev is assembled from three 4-bit MLX repos (DiT+VAE, text encoder);
/// the desktop app downloads them and launches the diffusion backend.
pub const IMAGE_MODELS: &[RecommendedImageModel] = &[
    RecommendedImageModel {
        id: "flux2-dev",
        label: "FLUX.2 dev — text-to-image (4-bit)",
        approx_size_gb: 31,
        min_ram_gb: 36,
        notes: "**Black Forest Labs FLUX.2 [dev]** — 32B rectified-flow image model, native MLX 4-bit. Generates images from text via `POST /v1/images/generations` (OpenAI-compatible). Components: DiT+VAE (`AITRADER/FLUX2-dev-mlx-4bit`, ~18 GB) + Mistral-Small-3.2 text encoder (`mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit`, ~13 GB). Served in dedicated image mode (cannot co-reside with an LLM on 36 GB). ~140 s / 256² image on M3 Max; 512²/28-step for full quality. Knobs: size, steps, seed, guidance.",
    },
    RecommendedImageModel {
        // Full-precision (bf16) sibling of `flux2-dev`. The DiT auto-detects
        // dense vs quantized per linear; the text encoder does the same via the
        // `Linear` enum (`.scales`-presence). When this id is the active image
        // model the diffusion engine auto-resolves the downloaded repo's
        // `transformer/` `text_encoder/` `vae/` subdirs (or set the 3
        // `LUMEN_FLUX2_*` env vars explicitly).
        id: "black-forest-labs/FLUX.2-dev",
        label: "FLUX.2 dev — full precision (bf16)",
        approx_size_gb: 113,
        min_ram_gb: 128,
        notes: "**Black Forest Labs FLUX.2 [dev] — full precision (bf16).** The official non-quantized repo (`black-forest-labs/FLUX.2-dev`, gated — auto-accept on download). Identical 32B rectified-flow architecture as the 4-bit `flux2-dev` entry but every weight is dense BF16: maximum image quality, no quantization loss. ~113 GB on disk, needs ≥128 GB RAM (≈64 GB peak just for the encoder is not enough — the full pipeline is large). Components are the repo's own `transformer/` (bf16 DiT), `text_encoder/` (bf16 Mistral-Small-3.2) and `vae/` subdirs — the engine resolves them automatically once downloaded, or point at them with `LUMEN_FLUX2_DIT_DIR` / `LUMEN_FLUX2_ENCODER_DIR` / `LUMEN_FLUX2_VAE_PATH`. Served in dedicated image mode. Use the 4-bit entry on 36–64 GB machines.",
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
    if is_qwen35_family(&lower) || lower.contains("qwen3-next") || lower.contains("a3b-mxfp4") {
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

/// Repo-id spellings that mean "this is the `qwen3_5` architecture family".
///
/// One list, because it has already been copied enough times to go wrong: the
/// MTP loader used to select its head shape from this same set of substrings and
/// silently mis-shaped any release outside it. Version markers accumulate here
/// (3.5, 3.6, 3.8 …) since Qwen keeps the `model_type: "qwen3_5"` backbone
/// across generations.
///
/// This is a *display* heuristic — it decides what the app's MODELS card offers.
/// Load-time behaviour reads the checkpoint's own config.json, never this.
/// NOTE the absent spelling: `"qwen3.5"` with a dot is deliberately NOT here.
/// The dot-spelled 3.5 repos in the wild are dense 9B builds whose ids carry no
/// size marker, so `is_qwen3_5_dense` would reject them and `family_of` would
/// label them MoE — a wrong answer where the current one is `None` ("not a
/// catalog model"). Add it only together with a size rule that gets 9B right.
fn is_qwen35_family(lower_id: &str) -> bool {
    lower_id.contains("qwen3_5") || lower_id.contains("qwen3.6") || lower_id.contains("qwen3.8")
}

fn is_qwen3_5_dense(lower_id: &str) -> bool {
    if !is_qwen35_family(lower_id) {
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
        image_models: IMAGE_MODELS.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::{Catalog, ModelFamily, catalog, family_of};

    #[test]
    fn the_qwen38_row_is_serveable_and_downloadable() {
        let Catalog { recommended, .. } = catalog();
        let row = recommended
            .iter()
            .find(|r| r.id == "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed")
            .expect(
                "Qwen3.8-27B must be in the recommended list — the app's MODELS card and its \
                     hf-cache scan are both filtered by these ids, so an absent row means the \
                     model cannot be downloaded or selected",
            );
        assert!(matches!(row.family, ModelFamily::Qwen35Dense));
        // Sized from the shipped repo (19.5 GB trunk + 0.92 GB vision + the
        // head), not copied from the 3.6 row — this build is mixed 4/8-bit and
        // genuinely larger.
        assert_eq!(row.approx_size_gb, 21);
        assert_eq!(row.min_ram_gb, 32);
    }

    #[test]
    fn the_36_row_survives_the_38_row() {
        // The 3.8 entry was added ALONGSIDE 3.6, not in place of it: the hf-cache
        // scan only surfaces ids present in this list, so dropping 3.6 would make
        // an already-downloaded 3.6 vanish from the app.
        let ids: Vec<&str> = catalog().recommended.iter().map(|r| r.id).collect();
        assert!(ids.contains(&"Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed"));
        assert!(ids.contains(&"Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed"));
    }

    #[test]
    fn a_38_id_resolves_to_the_dense_family() {
        // RED before the `is_qwen35_family` hoist: `family_of` tested only for
        // "qwen3.6" / "qwen3_5", so every 3.8 id returned None.
        assert_eq!(
            family_of("Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed"),
            Some(ModelFamily::Qwen35Dense)
        );
        assert_eq!(
            family_of("mlx-community/Qwen3.8-27B-4bit"),
            Some(ModelFamily::Qwen35Dense)
        );
    }

    #[test]
    fn the_previously_known_families_are_unchanged() {
        assert_eq!(
            family_of("Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed"),
            Some(ModelFamily::Qwen35Dense)
        );
        assert_eq!(
            family_of("mlx-community/Qwen3.6-35B-A3B-mxfp4"),
            Some(ModelFamily::Qwen35Moe)
        );
        assert_eq!(
            family_of("mlx-community/gemma-4-26b-a4b-it-nvfp4"),
            Some(ModelFamily::Gemma4)
        );
        assert_eq!(
            family_of("Qwen/Qwen2.5-7B-Instruct"),
            Some(ModelFamily::Qwen25)
        );
        assert_eq!(family_of("meta-llama/Llama-3-8B"), None);
    }

    #[test]
    fn a_dot_spelled_35_id_stays_unknown_rather_than_being_called_moe() {
        // The dot-spelled 3.5 repos are dense 9B builds carrying no size marker,
        // so admitting them to `is_qwen35_family` without a size rule would label
        // them MoE. `None` ("not a catalog model") is the honest answer until the
        // dense heuristic can tell 9B apart.
        assert_eq!(
            family_of("Youssofal/Qwen3.5-9B-MTPLX-Optimized-Speed"),
            None
        );
    }
}
