use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use lumen_model::gemma::GemmaModel;
use lumen_model::gemma_gguf::GemmaGgufModel;
use lumen_model::qwen::QwenModel;

#[cfg(feature = "qwen3_5_moe")]
use lumen_model::qwen3_5_moe::backend::Qwen35MoeBackend;

/// Gemma 4 26B-A4B native MLX backend wrapper. See
/// `crates/lumen-mlx/src/gemma4_backend.rs` for the trait-shape API
/// this dispatches into.
#[cfg(feature = "mlx-native")]
use lumen_mlx::gemma4::Gemma4Backend;

use crate::types::*;

/// Adaptive routing mode (selected at startup; cannot switch at runtime
/// without cold-load due to ~22.79 GB active memory per backend on 36 GB
/// unified-memory Mac → simultaneous hot-load OOMs).
///
/// See `notes/adaptive_backend_routing_plan.md` and
/// `notes/phase_a_profile_gap_analysis.md` for the deployment guide.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackendMode {
    /// Single-tenant: MLX standalone 71-72 tok/s @ 35B-A3B mxfp4 (1.51× Candle).
    Mlx,
    /// Multi-tenant: Candle CB N=8 1.89× wallclock aggregate (~89 tok/s).
    Candle,
}

/// Resolve the adaptive routing mode from env. Precedence:
///   1. `LUMEN_MODE=mlx|candle|auto`   — explicit selection
///   2. `USE_MLX=1`                       — legacy alias for `LUMEN_MODE=mlx`
///   3. default                           — Candle (safest for unknown workload)
///
/// `auto` resolves to Candle. We prefer Candle as the unknown-workload default
/// because (a) MLX 60-72 tok/s win is single-tenant only and (b) Candle's CB
/// safely degrades to single-tenant 47.6 tok/s without OOM risk.
fn resolve_backend_mode() -> BackendMode {
    resolve_backend_mode_from(|name| std::env::var(name).ok())
}

/// Pure-function variant for unit testing — env access is injected so tests
/// don't race on the global env.
fn resolve_backend_mode_from<F>(get: F) -> BackendMode
where
    F: Fn(&str) -> Option<String>,
{
    if let Some(raw) = get("LUMEN_MODE") {
        match raw.trim().to_ascii_lowercase().as_str() {
            "mlx" => return BackendMode::Mlx,
            "candle" | "auto" => return BackendMode::Candle,
            other => {
                eprintln!(
                    "warn: LUMEN_MODE={other:?} unrecognized; falling back to candle. \
                     Valid: mlx | candle | auto."
                );
                return BackendMode::Candle;
            }
        }
    }
    if matches!(get("USE_MLX").as_deref(), Some("1")) {
        return BackendMode::Mlx;
    }
    BackendMode::Candle
}

/// Model backend — supports multiple architectures.
enum ModelBackend {
    Qwen(QwenModel),
    Gemma(GemmaModel),
    GemmaGguf(GemmaGgufModel),
    #[cfg(feature = "qwen3_5_moe")]
    Qwen35Moe(Qwen35MoeBackend),
    /// Track B Phase 1: MLX backend via Python subprocess + JSON-RPC. Greedy
    /// only at B1; sampling lands in B2.
    Mlx(lumen_mlx::MlxBackend),
    /// native-Rust Gemma 4 26B-A4B MoE on MLX. Avoids the
    /// PyO3 subprocess hop entirely — `NativeGemma4Model` + the chat /
    /// response parser live in `lumen-mlx::gemma4::*`.
    #[cfg(feature = "mlx-native")]
    Gemma4Native(Gemma4Backend),
}

impl ModelBackend {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        match self {
            Self::Qwen(m) => m.encode(text),
            Self::Gemma(m) => m.encode(text),
            Self::GemmaGguf(m) => m.encode(text),
            #[cfg(feature = "qwen3_5_moe")]
            Self::Qwen35Moe(m) => m.encode(text),
            Self::Mlx(m) => m.encode(text),
            #[cfg(feature = "mlx-native")]
            Self::Gemma4Native(m) => m.encode(text),
        }
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        match self {
            Self::Qwen(m) => m.decode(tokens),
            Self::Gemma(m) => m.decode(tokens),
            Self::GemmaGguf(m) => m.decode(tokens),
            #[cfg(feature = "qwen3_5_moe")]
            Self::Qwen35Moe(m) => m.decode(tokens),
            Self::Mlx(m) => m.decode(tokens),
            #[cfg(feature = "mlx-native")]
            Self::Gemma4Native(m) => m.decode(tokens),
        }
    }

    /// Tokenize the chat-templated prompt for accurate `prompt_tokens`. Falls
    /// back to a `len/4` heuristic only if the backend errors during encode.
    fn count_chat_prompt_tokens(&self, messages: &[(String, String)], thinking: bool) -> u32 {
        let res: Result<Vec<u32>> = match self {
            Self::Qwen(m) => m.build_chat_input(messages),
            Self::Gemma(m) => m.build_chat_input(messages),
            Self::GemmaGguf(m) => m.build_chat_input(messages, thinking),
            #[cfg(feature = "qwen3_5_moe")]
            Self::Qwen35Moe(m) => m.build_chat_input(messages, thinking),
            Self::Mlx(m) => m.build_chat_input(messages, thinking),
            #[cfg(feature = "mlx-native")]
            Self::Gemma4Native(m) => m.build_chat_input(messages, thinking),
        };
        match res {
            Ok(ids) => ids.len() as u32,
            Err(_) => {
                let chars: usize = messages.iter().map(|(_, c)| c.len()).sum();
                ((chars as u32) / 4).max(1)
            }
        }
    }

    fn generate(
        &mut self,
        input_ids: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        session_id: Option<&str>,
    ) -> Result<Vec<u32>> {
        // Speculative decoding for GGUF models when SPEC_DRAFT_LAYERS is set
        if let Self::GemmaGguf(m) = self {
            static SPEC_CONFIG: std::sync::OnceLock<Option<(usize, usize)>> =
                std::sync::OnceLock::new();
            let spec = SPEC_CONFIG.get_or_init(|| {
                std::env::var("SPEC_DRAFT_LAYERS").ok().and_then(|s| {
                    let layers: usize = s.parse().ok()?;
                    let tokens: usize = std::env::var("SPEC_DRAFT_TOKENS")
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(4);
                    Some((layers, tokens))
                })
            });
            if let Some((draft_layers, draft_tokens)) = spec {
                return m.generate_speculative(
                    input_ids,
                    max_new_tokens,
                    temperature,
                    top_p,
                    *draft_layers,
                    *draft_tokens,
                );
            }
        }
        match self {
            Self::Qwen(m) => m.generate(input_ids, max_new_tokens, temperature, top_p),
            Self::Gemma(m) => m.generate(input_ids, max_new_tokens, temperature, top_p),
            Self::GemmaGguf(m) => m.generate(input_ids, max_new_tokens, temperature, top_p),
            #[cfg(feature = "qwen3_5_moe")]
            Self::Qwen35Moe(m) => m.generate(input_ids, max_new_tokens, temperature, top_p),
            Self::Mlx(m) => {
                // MLX backend B1: greedy-only via chat_streaming-style loop.
                // Generate path used by `/v1/completions` — drive prefill +
                // decode_step with no message wrapping. When `session_id` is
                // provided, route through `completion_session` so the cached
                // token prefix from prior turns is reused.
                let _ = (temperature, top_p);
                if let Some(sid) = session_id {
                    return m.completion_session(input_ids, max_new_tokens, sid);
                }
                let seq_id = m.alloc_seq_id();
                let (mut last, mut pos) = m.prefill(seq_id, input_ids)?;
                let mut out: Vec<u32> = vec![last];
                let eos = m.eos_tokens().to_vec();
                if !eos.contains(&last) {
                    for _ in 1..max_new_tokens {
                        let (n, p) = m.decode_step(seq_id, last, pos)?;
                        last = n;
                        pos = p;
                        out.push(n);
                        if eos.contains(&n) {
                            break;
                        }
                    }
                }
                m.remove_seq(seq_id).ok();
                Ok(out)
            }
            #[cfg(feature = "mlx-native")]
            Self::Gemma4Native(m) => {
                // greedy only; sampling lands in W5.
                let _ = session_id;
                m.generate(input_ids, max_new_tokens, temperature, top_p)
            }
        }
    }

    /// Drop a per-session prompt cache. Returns true if the session existed.
    /// Only the MLX backend tracks sessions today; other backends report
    /// "not found" silently.
    fn drop_session(&mut self, session_id: &str) -> bool {
        match self {
            Self::Mlx(m) => m.drop_session(session_id),
            _ => false,
        }
    }

    /// Drop an A1 prefix-cache entry by its auto-generated key. Returns true
    /// if the entry existed. MLX-only feature (Mlx + Gemma4Native back-ends).
    fn drop_prefix_cache(&mut self, key: &str) -> bool {
        match self {
            Self::Mlx(m) => m.drop_prefix_cache(key),
            #[cfg(feature = "mlx-native")]
            Self::Gemma4Native(m) => m.drop_prefix_cache(key),
            _ => false,
        }
    }

    /// Clear all A1 prefix-cache entries. Returns the number released.
    fn clear_prefix_cache(&mut self) -> usize {
        match self {
            Self::Mlx(m) => m.clear_prefix_cache(),
            #[cfg(feature = "mlx-native")]
            Self::Gemma4Native(m) => m.clear_prefix_cache(),
            _ => 0,
        }
    }

    fn chat(
        &mut self,
        messages: &[(String, String)],
        max_new_tokens: usize,
        temperature: f32,
        thinking: bool,
        session_id: Option<&str>,
    ) -> Result<String> {
        match self {
            Self::Qwen(m) => m.chat(messages, max_new_tokens, temperature),
            Self::Gemma(m) => m.chat(messages, max_new_tokens, temperature),
            Self::GemmaGguf(m) => {
                m.chat_with_options(messages, max_new_tokens, temperature, thinking)
            }
            #[cfg(feature = "qwen3_5_moe")]
            Self::Qwen35Moe(m) => m.chat(messages, max_new_tokens, temperature, thinking),
            Self::Mlx(m) => {
                let _ = temperature; // greedy at B1
                if let Some(sid) = session_id {
                    m.chat_streaming_session(messages, max_new_tokens, thinking, sid, |_| {})
                } else {
                    let seq_id = m.alloc_seq_id();
                    m.chat_streaming(messages, max_new_tokens, thinking, seq_id, |_| {})
                }
            }
            #[cfg(feature = "mlx-native")]
            Self::Gemma4Native(m) => {
                // W4 (c): greedy only — temperature/session_id ignored. The
                // parsed response carries reasoning + tool_calls separately,
                // but the wire-level engine contract is a single visible
                // string; HTTP layer can re-render structured fields once
                // the OpenAI shapes are extended.
                //
                // `thinking=true` empty-content fix (2026-05-14): when the
                // model is asked to think and the budget runs out before it
                // closes the `<|channel|>thought\n…<|channel|>` block, all
                // tokens end up in `resp.reasoning` and `resp.visible` is
                // empty. Returning the empty string to the HTTP layer
                // surfaces as `content: ""` — which client SDKs treat as a
                // failed turn. Fall back to reasoning so the user at least
                // sees the chain of thought rather than nothing.
                let _ = temperature;
                // Prefix-cache integration: when caller supplies `session_id`,
                // route through `chat_with_prefix_cache` so the system prompt
                // prefill is shared across batch requests with the same id.
                // Without `session_id`, fall back to the stateless `chat()`
                // path so single-shot requests stay zero-overhead.
                let resp = if let Some(sid) = session_id {
                    m.chat_with_prefix_cache(
                        messages,
                        max_new_tokens,
                        temperature,
                        thinking,
                        sid,
                    )?
                } else {
                    m.chat(messages, max_new_tokens, temperature, thinking)?
                };
                if resp.visible.is_empty() && !resp.reasoning.is_empty() {
                    Ok(resp.reasoning)
                } else {
                    Ok(resp.visible)
                }
            }
        }
    }

    fn chat_streaming<F>(
        &mut self,
        messages: &[(String, String)],
        max_new_tokens: usize,
        temperature: f32,
        thinking: bool,
        session_id: Option<&str>,
        on_token: F,
    ) -> Result<String>
    where
        F: FnMut(&str),
    {
        match self {
            Self::GemmaGguf(m) => {
                m.chat_streaming(messages, max_new_tokens, temperature, thinking, on_token)
            }
            // Fallback: generate all, send as one chunk
            Self::Qwen(m) => {
                let text = m.chat(messages, max_new_tokens, temperature)?;
                let mut on_token = on_token;
                on_token(&text);
                Ok(text)
            }
            Self::Gemma(m) => {
                let text = m.chat(messages, max_new_tokens, temperature)?;
                let mut on_token = on_token;
                on_token(&text);
                Ok(text)
            }
            #[cfg(feature = "qwen3_5_moe")]
            Self::Qwen35Moe(m) => {
                m.chat_streaming(messages, max_new_tokens, temperature, thinking, on_token)
            }
            Self::Mlx(m) => {
                let _ = temperature; // greedy at B1
                if let Some(sid) = session_id {
                    m.chat_streaming_session(messages, max_new_tokens, thinking, sid, on_token)
                } else {
                    let seq_id = m.alloc_seq_id();
                    m.chat_streaming(messages, max_new_tokens, thinking, seq_id, on_token)
                }
            }
            #[cfg(feature = "mlx-native")]
            Self::Gemma4Native(m) => {
                // W4 (c): same greedy-only constraint as `chat()`. Adapts
                // the FnMut(&str) -> Result<()> shape that Gemma4Backend
                // uses to the engine's FnMut(&str) (return type `()`) by
                // swallowing the result — token-flush errors are best-effort.
                //
                // Note: streaming on_token still only fires for *visible*
                // tokens (parser-side filter), so when thinking=true and the
                // thought channel never closes, the client sees no SSE
                // chunks. The post-stream fallback below makes the final
                // `Ok(...)` carry the reasoning text so non-streaming
                // callers (e.g. final-message accumulation) still get
                // something useful.
                let _ = (temperature, session_id);
                let mut on_token = on_token;
                let resp = m.chat_streaming(messages, max_new_tokens, thinking, |chunk| {
                    on_token(chunk);
                    Ok(())
                })?;
                if resp.visible.is_empty() && !resp.reasoning.is_empty() {
                    Ok(resp.reasoning)
                } else {
                    Ok(resp.visible)
                }
            }
        }
    }
}

/// Inference engine wrapping a model backend and tokenizer.
pub struct InferenceEngine {
    backend: ModelBackend,
    model_id: String,
}

/// Detect model architecture from model_id string.
///
/// The Qwen3.5 family (Qwen3.6-35B-A3B-mxfp4, Qwen3.6-27B Dense, Qwen3-Next, …) shares a
/// common hybrid linear+full-attention backbone but splits on the per-layer MLP variant:
///
///   * `qwen3_5_moe`   — 256-expert routed MoE (35B-A3B-mxfp4). Production path.
///   * `qwen3_5_dense` — Standard SwiGLU MLP (27B). Same backbone, dense MLP.
///
/// Both share KV cache, snapshot/restore, TurboQuant hooks. Routing is name-based here;
/// the loader subsequently confirms via `text_config.mlp_kind()` against the config.json.
fn detect_architecture(model_id: &str) -> &'static str {
    let lower = model_id.to_lowercase();
    if is_qwen3_5_dense(&lower) {
        "qwen3_5_dense"
    } else if lower.contains("qwen3.6") || lower.contains("qwen3_5") || lower.contains("qwen3-next")
        || lower.contains("a3b-mxfp4")
    {
        "qwen3_5_moe"
    } else if is_gemma4_native(&lower) {
        // Gemma 4 26B-A4B MoE via the native MLX backend.
        // Distinct from `"gemma4"` below (which historically routes Gemma
        // 1/2 through Candle).
        "gemma4_native"
    } else if lower.contains("gemma") {
        "gemma4"
    } else if lower.contains("qwen") {
        "qwen2"
    } else {
        // Default: try to load config.json and detect model_type
        "qwen2"
    }
}

/// Match Gemma 4 26B-A4B MoE checkpoints by repo / dir name. The pattern
/// `"gemma-4"` (or `gemma4-26b` / `gemma_4` / `gemma4_text`) is intentional
/// — the legacy `"gemma"` substring also matches Gemma 1/2 paths
/// (`google/gemma-2b` etc.) which route to the Candle `GemmaModel` instead.
fn is_gemma4_native(lower_id: &str) -> bool {
    lower_id.contains("gemma-4")
        || lower_id.contains("gemma4-")
        || lower_id.contains("gemma_4")
        || lower_id.contains("gemma4_")
}

/// Match Qwen3.5/3.6 dense (non-MoE) variants by repo-name conventions.
/// Currently covers Qwen3.6-27B (the only published dense variant of the family).
/// Future dense releases (e.g. hypothetical Qwen3.6-14B Dense) extend this list.
fn is_qwen3_5_dense(lower_id: &str) -> bool {
    let qwen35_family = lower_id.contains("qwen3.6") || lower_id.contains("qwen3_5");
    if !qwen35_family {
        return false;
    }
    // Explicit "-dense" tag wins.
    if lower_id.contains("-dense") || lower_id.contains("_dense") {
        return true;
    }
    // 27B is the published dense variant. A3B / MoE markers are MoE — bail.
    if lower_id.contains("a3b") || lower_id.contains("moe") {
        return false;
    }
    lower_id.contains("27b") || lower_id.contains("-27-")
}

impl InferenceEngine {
    /// Load a model, auto-detecting architecture from model_id.
    ///
    /// If model_id ends with `.gguf`, loads as GGUF quantized model.
    /// Set `TQ_BITS` env var (e.g. "4") to enable TurboQuant compressed KV cache.
    /// Set `TOKENIZER_ID` for GGUF models (default: "google/gemma-4-31B-it").
    pub fn load(model_id: &str) -> Result<Self> {
        // Backend selection (precedence: LUMEN_MODE > USE_MLX legacy):
        //
        //   LUMEN_MODE=mlx     — N=1 single-tenant. MLX 60-72 tok/s standalone.
        //   LUMEN_MODE=candle  — N≥2 multi-tenant. Continuous batching (N=8 1.89× wallclock).
        //   LUMEN_MODE=auto    — default = candle (safest for unknown workload).
        //   USE_MLX=1            — legacy. Equivalent to LUMEN_MODE=mlx.
        //
        // 36 GB Mac unified memory cap (~22.79 GB active per backend) prevents
        // simultaneous hot-load of both — pick at startup. See
        // `notes/adaptive_backend_routing_plan.md` for deployment guide.
        let mode = resolve_backend_mode();
        if mode == BackendMode::Mlx {
            eprintln!("Loading MLX backend (mode={mode:?}): {model_id}");
            let backend = lumen_mlx::MlxBackend::load(model_id)?;
            return Ok(Self {
                backend: ModelBackend::Mlx(backend),
                model_id: model_id.to_string(),
            });
        }
        eprintln!("Loading Candle backend (mode={mode:?}): {model_id}");

        // GGUF path detection
        if model_id.ends_with(".gguf") {
            let tokenizer_id = std::env::var("TOKENIZER_ID")
                .unwrap_or_else(|_| "google/gemma-4-E4B-it".to_string());
            eprintln!("Loading GGUF model: {model_id}");
            let mut model = GemmaGgufModel::load(model_id, &tokenizer_id)?;

            // Enable TurboQuant GPU compressed KV cache if TQ_BITS is set.
            // Recommended for 27B+ models with long context (8K+).
            // NOT recommended for small models (E4B) with short context — use SDPA instead.
            //
            // Environment variables:
            //   TQ_BITS=3       — quantization bits (2, 3, or 4). 3-bit recommended.
            //   TQ_LAYERS=42    — number of transformer layers (auto: model-dependent)
            //   TQ_KV_HEADS=4   — number of KV attention heads (auto: model-dependent)
            //   TQ_HEAD_DIM=256 — attention head dimension (auto: model-dependent)
            //   TQ_MAX_SEQ=8192 — max sequence length for KV cache pool
            #[cfg(feature = "turboquant-gpu")]
            if let Ok(bits_str) = std::env::var("TQ_BITS") {
                let bits: u32 = bits_str.parse().unwrap_or(3);
                let n_layers = std::env::var("TQ_LAYERS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(42);
                let n_kv_heads = std::env::var("TQ_KV_HEADS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(4);
                let head_dim = std::env::var("TQ_HEAD_DIM")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(256);
                model.enable_compressed_kv(bits, n_layers, n_kv_heads, head_dim)?;
            }

            // Optional: vLLM-style PagedAttention (single-sequence for now).
            //   PAGED_KV=1                    — enable
            //   PAGED_KV_MB=2048              — pool budget in MB (default 2048)
            //   PAGED_BLOCK_SIZE=16           — tokens per block
            //   PAGED_LAYERS=48               — transformer layers
            //   PAGED_KV_HEADS=8              — KV heads per layer
            //   PAGED_HEAD_DIM_SLIDING=256    — sliding window head dim
            //   PAGED_HEAD_DIM_GLOBAL=512     — global layer head dim
            //   PAGED_GLOBAL_EVERY=6          — global layer pattern
            if std::env::var("PAGED_KV")
                .map(|v| v == "1" || v == "true")
                .unwrap_or(false)
            {
                let n_layers: u32 = std::env::var("PAGED_LAYERS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(48);
                let n_kv_heads: u32 = std::env::var("PAGED_KV_HEADS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(8);
                let hd_sliding: u32 = std::env::var("PAGED_HEAD_DIM_SLIDING")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(256);
                let hd_global: u32 = std::env::var("PAGED_HEAD_DIM_GLOBAL")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(512);
                let global_every: u32 = std::env::var("PAGED_GLOBAL_EVERY")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(6);
                model.enable_paged_kv(n_layers, n_kv_heads, hd_sliding, hd_global, global_every)?;
            }

            return Ok(Self {
                backend: ModelBackend::GemmaGguf(model),
                model_id: model_id.to_string(),
            });
        }

        let arch = detect_architecture(model_id);
        eprintln!("Detected architecture: {arch}");

        // Both `qwen3_5_moe` (35B-A3B-mxfp4) and `qwen3_5_dense` (Qwen3.6-27B) share
        // the same Candle backend path. The loader inspects `text_config.mlp_kind()`
        // and constructs either a `SparseMoeBlock` or a `DenseMlp`, packed into the
        // common `MlpBlock` enum that `DecoderLayer.forward` dispatches on.
        #[cfg(feature = "qwen3_5_moe")]
        if arch == "qwen3_5_moe" || arch == "qwen3_5_dense" {
            use std::path::PathBuf;
            use std::sync::Arc;
            use lumen_metal::affine4_gpu::Affine4Context;
            use lumen_metal::mxfp4_gpu::MxFp4Context;
            use lumen_model::qwen3_5_moe::backend::Qwen35MoeBackend;

            let shard_dir = std::env::var("LUMEN_QWEN35_SHARDS")
                .map(PathBuf::from)
                .map_err(|_| anyhow::anyhow!(
                    "{arch} requires LUMEN_QWEN35_SHARDS=<dir> pointing to the \
                     shard directory (config.json + model.safetensors.index.json + shards)"
                ))?;
            let gpu_ctx = Arc::new(MxFp4Context::new()?);
            let mut backend = if arch == "qwen3_5_dense" {
                // 27B dense ships uniform 4-bit affine quantization → also wire up
                // the GPU-resident Affine4Context so projections stay on device.
                let affine4_ctx = Arc::new(Affine4Context::new()?);
                Qwen35MoeBackend::load_with_affine4(model_id, &shard_dir, gpu_ctx, affine4_ctx)?
            } else {
                Qwen35MoeBackend::load(model_id, &shard_dir, gpu_ctx)?
            };

            // Optional MTP speculative-draft head (llama.cpp PR #22673 port).
            // Active only when:
            //   1. `LUMEN_SPEC` env contains "mtp" (case-insensitive), AND
            //   2. `LUMEN_QWEN35_HF_ORIGINAL` env points at the HF original snapshot
            //      directory (which carries the `mtp.*` weights), AND
            //   3. the model arch is Qwen3.5/3.6 (config carries an MTP head).
            // Any one missing → backend stays in non-MTP mode silently (the
            // existing decode path still works fine).
            let spec_kind = std::env::var("LUMEN_SPEC")
                .ok()
                .map(|s| s.to_ascii_lowercase())
                .unwrap_or_default();
            if spec_kind.contains("mtp") {
                match backend.try_enable_mtp(&shard_dir) {
                    Ok(true) => eprintln!("  MTP speculative draft enabled (LUMEN_SPEC=mtp)"),
                    Ok(false) => eprintln!(
                        "  MTP requested via LUMEN_SPEC=mtp but no `mtp.*` weights resolved \
                         — continuing without speculation. Set LUMEN_QWEN35_HF_ORIGINAL to \
                         the HuggingFace original snapshot dir (e.g. \
                         ~/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/<hash>/)",
                    ),
                    Err(e) => eprintln!(
                        "  MTP enable failed: {e:#} — continuing without speculation"
                    ),
                }
            }
            return Ok(Self {
                backend: ModelBackend::Qwen35Moe(backend),
                model_id: model_id.to_string(),
            });
        }

        // native Gemma 4 26B-A4B MoE on MLX. Source the
        // model directory from either:
        //   1. `model_id` itself, if it's an existing local path, OR
        //   2. `LUMEN_GEMMA4_DIR` env var.
        // The directory must contain `config.json`, `tokenizer.json`, and
        // the safetensors shards (`model-XXXXX-of-YYYYY.safetensors` +
        // `model.safetensors.index.json`).
        #[cfg(feature = "mlx-native")]
        if arch == "gemma4_native" {
            use std::path::PathBuf;
            let dir = if std::path::Path::new(model_id).is_dir() {
                PathBuf::from(model_id)
            } else if let Ok(d) = std::env::var("LUMEN_GEMMA4_DIR") {
                PathBuf::from(d)
            } else {
                return Err(anyhow::anyhow!(
                    "gemma4_native requires either MODEL_ID pointing at an existing \
                     local directory or LUMEN_GEMMA4_DIR=<dir> with config.json + \
                     tokenizer.json + safetensors shards"
                ));
            };
            let backend = Gemma4Backend::from_dir(model_id, &dir)?;
            return Ok(Self {
                backend: ModelBackend::Gemma4Native(backend),
                model_id: model_id.to_string(),
            });
        }

        let backend = match arch {
            "gemma4" => {
                let mut model = GemmaModel::load(model_id)?;

                // Enable TurboQuant if TQ_BITS is set. The hook lives on the
                // candle-side Gemma 4 E4B path and only compiles when the
                // optional `turboquant` feature on `lumen-model` is enabled
                // (it pulls in the candle-transformers turboquant cfg
                // block and reactivates the circular workspace deps via
                // the workspace `[patch]` override).
                #[cfg(feature = "turboquant")]
                if let Ok(bits_str) = std::env::var("TQ_BITS") {
                    let bits: u32 = bits_str.parse().unwrap_or(4);
                    let text_cfg = model.text_config();
                    let head_dim = text_cfg.head_dim;
                    let n_layers = text_cfg.num_hidden_layers;
                    let n_kv_heads = text_cfg.num_key_value_heads;
                    model.enable_turboquant(bits, n_layers, n_kv_heads, head_dim);
                }

                ModelBackend::Gemma(model)
            }
            _ => ModelBackend::Qwen(QwenModel::load(model_id)?),
        };

        Ok(Self {
            backend,
            model_id: model_id.to_string(),
        })
    }

    /// Run warmup forward passes to compile all Metal shaders and stabilize GPU power state.
    pub fn warmup(&mut self) -> Result<()> {
        let skip = std::env::var("SKIP_WARMUP").is_ok();
        if skip {
            eprintln!("Skipping warmup (SKIP_WARMUP set)");
            return Ok(());
        }

        eprintln!("Warming up GPU...");
        let t = std::time::Instant::now();

        // Single short pass — just compiles Metal shaders + stabilizes GPU
        let messages = vec![("user".to_string(), "Hi".to_string())];
        let _ = self.backend.chat(&messages, 3, 0.0, false, None)?;
        eprintln!(
            "  pass 1 done ({:.0}ms)",
            t.elapsed().as_secs_f64() * 1000.0
        );

        // One more decode step to ensure pipeline is warm
        let messages = vec![("user".to_string(), "Hi".to_string())];
        let _ = self.backend.chat(&messages, 2, 0.0, false, None)?;

        eprintln!(
            "  warmup complete in {:.0}ms",
            t.elapsed().as_secs_f64() * 1000.0
        );
        Ok(())
    }

    /// Handle a chat completion request.
    pub fn chat_completion(
        &mut self,
        req: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        let messages: Vec<(String, String)> = req
            .messages
            .iter()
            .map(|m| (m.role.clone(), m.content.clone()))
            .collect();

        // log per-request prompt size
        // so we can diagnose OOMs that originate from oversized requests
        // (e.g. an external agent stuffing its skill catalog into a system
        // prompt). Bytes ≠ tokens but is a fast proxy: anything past ~64 KB
        // is a strong signal to investigate before suspecting a kernel bug.
        let prompt_bytes: usize = messages.iter().map(|(_, c)| c.len()).sum();
        eprintln!(
            "[chat] msgs={} prompt_bytes={} max_tokens={} thinking={} stream={}",
            messages.len(),
            prompt_bytes,
            req.max_tokens,
            req.thinking,
            req.stream,
        );

        let content = self.backend.chat(
            &messages,
            req.max_tokens,
            req.temperature,
            req.thinking,
            req.session_id.as_deref(),
        )?;

        let prompt_tokens = self
            .backend
            .count_chat_prompt_tokens(&messages, req.thinking);
        let completion_tokens = count_tokens(&self.backend, &content);

        Ok(ChatCompletionResponse {
            id: format!("chatcmpl-{}", gen_id()),
            object: "chat.completion".into(),
            created: unix_timestamp(),
            model: req.model.clone(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessageResponse {
                    role: "assistant".into(),
                    content,
                },
                finish_reason: "stop".into(),
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        })
    }

    /// Handle a completion request.
    pub fn completion(&mut self, req: &CompletionRequest) -> Result<CompletionResponse> {
        let input_ids = self.backend.encode(&req.prompt)?;
        let prompt_tokens = input_ids.len() as u32;

        let output_ids = self.backend.generate(
            &input_ids,
            req.max_tokens,
            req.temperature,
            req.top_p,
            req.session_id.as_deref(),
        )?;
        let completion_tokens = output_ids.len() as u32;
        let text = self.backend.decode(&output_ids)?;

        Ok(CompletionResponse {
            id: format!("cmpl-{}", gen_id()),
            object: "text_completion".into(),
            created: unix_timestamp(),
            model: req.model.clone(),
            choices: vec![CompletionChoice {
                index: 0,
                text,
                finish_reason: "stop".into(),
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        })
    }

    /// Handle an Anthropic Messages API request.
    pub fn anthropic_messages(&mut self, req: &AnthropicRequest) -> Result<AnthropicResponse> {
        let mut messages: Vec<(String, String)> = Vec::new();

        // System message
        if let Some(ref system) = req.system {
            let system_text = match system {
                AnthropicSystem::Text(s) => s.clone(),
                AnthropicSystem::Blocks(blocks) => blocks
                    .iter()
                    .filter_map(|b| b.text.clone())
                    .collect::<Vec<_>>()
                    .join("\n"),
            };
            if !system_text.is_empty() {
                messages.push(("system".into(), system_text));
            }
        }

        // User/assistant messages
        for msg in &req.messages {
            messages.push((msg.role.clone(), msg.content.as_text()));
        }

        let content = self.backend.chat(
            &messages,
            req.max_tokens,
            req.temperature,
            req.thinking,
            req.session_id.as_deref(),
        )?;

        let prompt_tokens = self
            .backend
            .count_chat_prompt_tokens(&messages, req.thinking);
        let output_tokens = count_tokens(&self.backend, &content);

        Ok(AnthropicResponse {
            id: format!("msg_{}", gen_id()),
            r#type: "message".into(),
            role: "assistant".into(),
            model: req.model.clone(),
            content: vec![AnthropicResponseBlock {
                r#type: "text".into(),
                text: content,
            }],
            stop_reason: "end_turn".into(),
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: prompt_tokens,
                output_tokens,
            },
        })
    }

    /// Handle a streaming chat completion — sends tokens via channel as they're generated.
    fn chat_completion_streaming(
        &mut self,
        req: &ChatCompletionRequest,
        token_tx: &mpsc::Sender<StreamEvent>,
    ) {
        let messages: Vec<(String, String)> = req
            .messages
            .iter()
            .map(|m| (m.role.clone(), m.content.clone()))
            .collect();

        let prompt_tokens = self
            .backend
            .count_chat_prompt_tokens(&messages, req.thinking);

        // mirror chat_completion log
        // on the streaming path. Moltis defaults to stream=true.
        let prompt_bytes: usize = messages.iter().map(|(_, c)| c.len()).sum();
        eprintln!(
            "[chat-stream] msgs={} prompt_bytes={} prompt_tokens={} max_tokens={} thinking={}",
            messages.len(),
            prompt_bytes,
            prompt_tokens,
            req.max_tokens,
            req.thinking,
        );

        // chunked prefill
        // is back with sliding-window-aware mask construction
        // (`make_attention_mask_for_layer_chunked`). Cap protects against
        // runaway KV growth from misbehaving clients while letting normal
        // long-context (≤32K) requests flow.
        const PREFILL_TOKEN_CAP: u32 = 32_768;
        if prompt_tokens > PREFILL_TOKEN_CAP {
            let msg = format!(
                "prompt too large: {prompt_tokens} tokens > server cap {PREFILL_TOKEN_CAP}. \
                 Reduce system prompt / message history, or wait for chunked prefill support."
            );
            eprintln!("[chat-stream] REJECTED ({msg})");
            let _ = token_tx.try_send(StreamEvent::Error(msg));
            return;
        }

        let result = self.backend.chat_streaming(
            &messages,
            req.max_tokens,
            req.temperature,
            req.thinking,
            req.session_id.as_deref(),
            |text| {
                let _ = token_tx.try_send(StreamEvent::Delta(text.to_string()));
            },
        );

        match result {
            Ok(full_text) => {
                let completion_tokens = count_tokens(&self.backend, &full_text);
                let _ = token_tx.try_send(StreamEvent::Done {
                    prompt_tokens,
                    completion_tokens,
                });
            }
            Err(e) => {
                eprintln!("[chat-stream] ERR: {e:#}");
                let _ = token_tx.try_send(StreamEvent::Error(e.to_string()));
            }
        }
    }

    /// Handle a streaming Anthropic messages request.
    fn anthropic_messages_streaming(
        &mut self,
        req: &AnthropicRequest,
        token_tx: &mpsc::Sender<StreamEvent>,
    ) {
        let mut messages: Vec<(String, String)> = Vec::new();

        if let Some(ref system) = req.system {
            let system_text = match system {
                AnthropicSystem::Text(s) => s.clone(),
                AnthropicSystem::Blocks(blocks) => blocks
                    .iter()
                    .filter_map(|b| b.text.clone())
                    .collect::<Vec<_>>()
                    .join("\n"),
            };
            if !system_text.is_empty() {
                messages.push(("system".into(), system_text));
            }
        }

        for msg in &req.messages {
            messages.push((msg.role.clone(), msg.content.as_text()));
        }

        let prompt_tokens = self
            .backend
            .count_chat_prompt_tokens(&messages, req.thinking);

        let result = self.backend.chat_streaming(
            &messages,
            req.max_tokens,
            req.temperature,
            req.thinking,
            req.session_id.as_deref(),
            |text| {
                let _ = token_tx.try_send(StreamEvent::Delta(text.to_string()));
            },
        );

        match result {
            Ok(full_text) => {
                let completion_tokens = count_tokens(&self.backend, &full_text);
                let _ = token_tx.try_send(StreamEvent::Done {
                    prompt_tokens,
                    completion_tokens,
                });
            }
            Err(e) => {
                let _ = token_tx.try_send(StreamEvent::Error(e.to_string()));
            }
        }
    }

    /// Drop a per-session prompt cache. Returns true if the session existed.
    pub fn drop_session(&mut self, session_id: &str) -> bool {
        self.backend.drop_session(session_id)
    }

    /// Drop an A1 prefix-cache entry by its auto-generated key. Returns true
    /// if the entry existed.
    pub fn drop_prefix_cache(&mut self, key: &str) -> bool {
        self.backend.drop_prefix_cache(key)
    }

    /// Clear all A1 prefix-cache entries. Returns the number released.
    pub fn clear_prefix_cache(&mut self) -> usize {
        self.backend.clear_prefix_cache()
    }

    /// List available models.
    pub fn list_models(&self) -> ModelListResponse {
        ModelListResponse {
            object: "list".into(),
            data: vec![ModelObject {
                id: self.model_id.clone(),
                object: "model".into(),
                created: unix_timestamp(),
                owned_by: "turboquant".into(),
            }],
        }
    }
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Exact token count via the backend tokenizer. Falls back to a char/4
/// heuristic if encoding fails (and bottoms at 1 for any non-empty text so
/// `usage.completion_tokens` is never silently zero on short responses).
fn count_tokens(backend: &ModelBackend, text: &str) -> u32 {
    if text.is_empty() {
        return 0;
    }
    match backend.encode(text) {
        Ok(ids) => ids.len() as u32,
        Err(_) => ((text.len() as u32) / 4).max(1),
    }
}

fn gen_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    // Append a process-local monotonic counter so concurrent requests within
    // the same second don't collide (unix_timestamp has 1s resolution).
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{:x}-{:x}", unix_timestamp(), n)
}

// ── Channel-based Engine Handle ──────────────────────────────────────────

use tokio::sync::{mpsc, oneshot};

/// Events emitted during streaming generation.
pub enum StreamEvent {
    /// New text fragment (delta).
    Delta(String),
    /// Generation complete with token counts.
    Done {
        prompt_tokens: u32,
        completion_tokens: u32,
    },
    /// Error during generation.
    Error(String),
}

/// Request sent through the channel to the engine thread.
pub enum EngineRequest {
    ChatCompletion {
        req: ChatCompletionRequest,
        reply: oneshot::Sender<Result<ChatCompletionResponse>>,
    },
    StreamingChatCompletion {
        req: ChatCompletionRequest,
        token_tx: mpsc::Sender<StreamEvent>,
    },
    Completion {
        req: CompletionRequest,
        reply: oneshot::Sender<Result<CompletionResponse>>,
    },
    AnthropicMessages {
        req: AnthropicRequest,
        reply: oneshot::Sender<Result<AnthropicResponse>>,
    },
    StreamingAnthropicMessages {
        req: AnthropicRequest,
        token_tx: mpsc::Sender<StreamEvent>,
    },
    ListModels {
        reply: oneshot::Sender<ModelListResponse>,
    },
    DropSession {
        session_id: String,
        reply: oneshot::Sender<bool>,
    },
    DropPrefixCache {
        key: String,
        reply: oneshot::Sender<bool>,
    },
    ClearPrefixCache {
        reply: oneshot::Sender<usize>,
    },
}

/// Client-side handle to the engine (clone-friendly, no Mutex).
#[derive(Clone)]
pub struct EngineHandle {
    tx: mpsc::Sender<EngineRequest>,
}

impl EngineHandle {
    pub fn new(tx: mpsc::Sender<EngineRequest>) -> Self {
        Self { tx }
    }

    pub async fn chat_completion(
        &self,
        req: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(EngineRequest::ChatCompletion {
                req,
                reply: reply_tx,
            })
            .await
            .map_err(|_| anyhow::anyhow!("engine channel closed"))?;
        reply_rx
            .await
            .map_err(|_| anyhow::anyhow!("engine dropped reply"))?
    }

    pub async fn completion(&self, req: CompletionRequest) -> Result<CompletionResponse> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(EngineRequest::Completion {
                req,
                reply: reply_tx,
            })
            .await
            .map_err(|_| anyhow::anyhow!("engine channel closed"))?;
        reply_rx
            .await
            .map_err(|_| anyhow::anyhow!("engine dropped reply"))?
    }

    pub async fn anthropic_messages(&self, req: AnthropicRequest) -> Result<AnthropicResponse> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(EngineRequest::AnthropicMessages {
                req,
                reply: reply_tx,
            })
            .await
            .map_err(|_| anyhow::anyhow!("engine channel closed"))?;
        reply_rx
            .await
            .map_err(|_| anyhow::anyhow!("engine dropped reply"))?
    }

    pub async fn chat_completion_streaming(
        &self,
        req: ChatCompletionRequest,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        let (token_tx, token_rx) = mpsc::channel(256);
        self.tx
            .send(EngineRequest::StreamingChatCompletion { req, token_tx })
            .await
            .map_err(|_| anyhow::anyhow!("engine channel closed"))?;
        Ok(token_rx)
    }

    pub async fn anthropic_messages_streaming(
        &self,
        req: AnthropicRequest,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        let (token_tx, token_rx) = mpsc::channel(256);
        self.tx
            .send(EngineRequest::StreamingAnthropicMessages { req, token_tx })
            .await
            .map_err(|_| anyhow::anyhow!("engine channel closed"))?;
        Ok(token_rx)
    }

    pub async fn list_models(&self) -> ModelListResponse {
        let (reply_tx, reply_rx) = oneshot::channel();
        let _ = self
            .tx
            .send(EngineRequest::ListModels { reply: reply_tx })
            .await;
        reply_rx.await.unwrap_or_else(|_| ModelListResponse {
            object: "list".into(),
            data: vec![],
        })
    }

    pub async fn drop_session(&self, session_id: String) -> bool {
        let (reply_tx, reply_rx) = oneshot::channel();
        if self
            .tx
            .send(EngineRequest::DropSession {
                session_id,
                reply: reply_tx,
            })
            .await
            .is_err()
        {
            return false;
        }
        reply_rx.await.unwrap_or(false)
    }

    pub async fn drop_prefix_cache(&self, key: String) -> bool {
        let (reply_tx, reply_rx) = oneshot::channel();
        if self
            .tx
            .send(EngineRequest::DropPrefixCache {
                key,
                reply: reply_tx,
            })
            .await
            .is_err()
        {
            return false;
        }
        reply_rx.await.unwrap_or(false)
    }

    pub async fn clear_prefix_cache(&self) -> usize {
        let (reply_tx, reply_rx) = oneshot::channel();
        if self
            .tx
            .send(EngineRequest::ClearPrefixCache { reply: reply_tx })
            .await
            .is_err()
        {
            return 0;
        }
        reply_rx.await.unwrap_or(0)
    }
}

impl InferenceEngine {
    /// Run the engine loop. When `BATCHED_ENGINE=1` and backend is GemmaGguf or Qwen35Moe,
    /// streaming chat/anthropic requests go through a continuous-batching
    /// scheduler. All other paths remain sequential.
    pub async fn run(mut self, mut rx: mpsc::Receiver<EngineRequest>) {
        let batched = std::env::var("BATCHED_ENGINE").ok().as_deref() == Some("1");
        match &self.backend {
            ModelBackend::GemmaGguf(_) if batched => self.run_batched(&mut rx).await,
            #[cfg(feature = "qwen3_5_moe")]
            ModelBackend::Qwen35Moe(_) if batched => self.run_batched_qwen35(&mut rx).await,
            _ => self.run_sequential(&mut rx).await,
        }
    }

    async fn run_sequential(&mut self, rx: &mut mpsc::Receiver<EngineRequest>) {
        while let Some(req) = rx.recv().await {
            self.dispatch_request_sequential(req);
        }
    }

    fn dispatch_request_sequential(&mut self, req: EngineRequest) {
        match req {
            EngineRequest::ChatCompletion { req, reply } => {
                let _ = reply.send(self.chat_completion(&req));
            }
            EngineRequest::StreamingChatCompletion { req, token_tx } => {
                self.chat_completion_streaming(&req, &token_tx);
            }
            EngineRequest::Completion { req, reply } => {
                let _ = reply.send(self.completion(&req));
            }
            EngineRequest::AnthropicMessages { req, reply } => {
                let _ = reply.send(self.anthropic_messages(&req));
            }
            EngineRequest::StreamingAnthropicMessages { req, token_tx } => {
                self.anthropic_messages_streaming(&req, &token_tx);
            }
            EngineRequest::ListModels { reply } => {
                let _ = reply.send(self.list_models());
            }
            EngineRequest::DropSession { session_id, reply } => {
                let _ = reply.send(self.drop_session(&session_id));
            }
            EngineRequest::DropPrefixCache { key, reply } => {
                let _ = reply.send(self.drop_prefix_cache(&key));
            }
            EngineRequest::ClearPrefixCache { reply } => {
                let _ = reply.send(self.clear_prefix_cache());
            }
        }
    }

    /// Continuous-batching scheduler for streaming chat / anthropic requests.
    /// One decode step processes up to `PAGED_MAX_BATCH` active seqs at once
    /// via `forward_batched_decode_v2`. Non-streaming requests are serviced
    /// between decode steps (they temporarily pause the batch).
    async fn run_batched(&mut self, rx: &mut mpsc::Receiver<EngineRequest>) {
        use candle_core::{DType, Tensor};
        use std::collections::HashMap;


        let max_batch: usize = std::env::var("PAGED_MAX_BATCH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);

        let mut active: HashMap<u64, ActiveSeqState> = HashMap::new();
        let mut next_seq_id: u64 = 1;
        // Seqs whose local kv_cache has advanced past their paged state
        // (i.e. they ran a step with paged_store_enabled = false). Must be
        // migrated into paged before joining a batched (N>=2) decode step.
        let mut dirty_paged: std::collections::HashSet<u64> = std::collections::HashSet::new();

        eprintln!("[batched engine] active scheduler (max_batch={max_batch})");

        loop {
            // 1. Admit new streaming requests non-blockingly, up to max_batch.
            while active.len() < max_batch {
                match rx.try_recv() {
                    Ok(EngineRequest::StreamingChatCompletion { req, token_tx }) => {
                        match self.start_streaming_seq(
                            next_seq_id,
                            &req.messages,
                            req.max_tokens,
                            req.temperature,
                            req.top_p,
                            req.thinking,
                            token_tx.clone(),
                        ) {
                            Ok(seq) => {
                                active.insert(next_seq_id, seq);
                                next_seq_id += 1;
                            }
                            Err(e) => {
                                let _ = token_tx.try_send(StreamEvent::Error(e.to_string()));
                            }
                        }
                    }
                    Ok(EngineRequest::StreamingAnthropicMessages { req, token_tx }) => {
                        let mut messages: Vec<ChatMessage> = Vec::new();
                        if let Some(ref system) = req.system {
                            let system_text = match system {
                                AnthropicSystem::Text(s) => s.clone(),
                                AnthropicSystem::Blocks(blocks) => blocks
                                    .iter()
                                    .filter_map(|b| b.text.clone())
                                    .collect::<Vec<_>>()
                                    .join("\n"),
                            };
                            if !system_text.is_empty() {
                                messages.push(ChatMessage {
                                    role: "system".into(),
                                    content: system_text,
                                });
                            }
                        }
                        for msg in &req.messages {
                            messages.push(ChatMessage {
                                role: msg.role.clone(),
                                content: msg.content.as_text(),
                            });
                        }
                        match self.start_streaming_seq(
                            next_seq_id,
                            &messages,
                            req.max_tokens,
                            req.temperature,
                            0.9, // AnthropicRequest has no top_p; use default
                            req.thinking,
                            token_tx.clone(),
                        ) {
                            Ok(seq) => {
                                active.insert(next_seq_id, seq);
                                next_seq_id += 1;
                            }
                            Err(e) => {
                                let _ = token_tx.try_send(StreamEvent::Error(e.to_string()));
                            }
                        }
                    }
                    Ok(other) => {
                        // Non-streaming: service sequentially (pauses batch).
                        self.dispatch_request_sequential(other);
                    }
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => return,
                }
            }

            if active.is_empty() {
                // No active seqs — block until a request arrives.
                match rx.recv().await {
                    Some(req) => {
                        if matches!(
                            req,
                            EngineRequest::StreamingChatCompletion { .. }
                                | EngineRequest::StreamingAnthropicMessages { .. }
                        ) {
                            // Re-insert into rx via self-send isn't possible; handle inline.
                            match req {
                                EngineRequest::StreamingChatCompletion { req, token_tx } => {
                                    match self.start_streaming_seq(
                                        next_seq_id,
                                        &req.messages,
                                        req.max_tokens,
                                        req.temperature,
                                        req.top_p,
                                        req.thinking,
                                        token_tx.clone(),
                                    ) {
                                        Ok(seq) => {
                                            active.insert(next_seq_id, seq);
                                            next_seq_id += 1;
                                        }
                                        Err(e) => {
                                            let _ = token_tx
                                                .try_send(StreamEvent::Error(e.to_string()));
                                        }
                                    }
                                }
                                EngineRequest::StreamingAnthropicMessages { req, token_tx } => {
                                    let mut messages: Vec<ChatMessage> = Vec::new();
                                    if let Some(ref system) = req.system {
                                        let system_text = match system {
                                            AnthropicSystem::Text(s) => s.clone(),
                                            AnthropicSystem::Blocks(blocks) => blocks
                                                .iter()
                                                .filter_map(|b| b.text.clone())
                                                .collect::<Vec<_>>()
                                                .join("\n"),
                                        };
                                        if !system_text.is_empty() {
                                            messages.push(ChatMessage {
                                                role: "system".into(),
                                                content: system_text,
                                            });
                                        }
                                    }
                                    for msg in &req.messages {
                                        messages.push(ChatMessage {
                                            role: msg.role.clone(),
                                            content: msg.content.as_text(),
                                        });
                                    }
                                    match self.start_streaming_seq(
                                        next_seq_id,
                                        &messages,
                                        req.max_tokens,
                                        req.temperature,
                                        0.9,
                                        req.thinking,
                                        token_tx.clone(),
                                    ) {
                                        Ok(seq) => {
                                            active.insert(next_seq_id, seq);
                                            next_seq_id += 1;
                                        }
                                        Err(e) => {
                                            let _ = token_tx
                                                .try_send(StreamEvent::Error(e.to_string()));
                                        }
                                    }
                                }
                                _ => unreachable!(),
                            }
                        } else {
                            self.dispatch_request_sequential(req);
                        }
                    }
                    None => return,
                }
                continue;
            }

            // 2. Decode step. N=1 → single-seq SDPA path (faster at short ctx).
            //    N≥2 → batched paged kernel.
            let ids: Vec<u64> = active.keys().copied().collect();
            let last_tokens: Vec<u32> = ids.iter().map(|id| active[id].last_token).collect();
            let positions: Vec<usize> = ids.iter().map(|id| active[id].position).collect();

            let gem = match &mut self.backend {
                ModelBackend::GemmaGguf(g) => g,
                _ => {
                    eprintln!("[batched engine] non-Gemma backend unreachable");
                    return;
                }
            };
            let device = gem.device().clone();

            let t_step = std::time::Instant::now();
            let debug_timing = std::env::var("BATCHED_TIMING").is_ok();
            // N=1 GPU fast path: keep logits on device, sample on GPU, emit a
            // single u32 — avoids the 1 MB F16→F32 materialization that
            // dominates CPU sampling. Returns Some(next_tok) on success.
            let n1_gpu_tok: Option<u32> = if ids.len() == 1 {
                let t_flags = std::time::Instant::now();
                gem.model_mut().set_current_seq_id(ids[0]);
                gem.model_mut().set_use_compressed_for_attn(false);
                gem.model_mut().set_paged_store_enabled(false);
                dirty_paged.insert(ids[0]);
                let seq = active.get(&ids[0]).unwrap();
                // GPU sampler skips top-p and n-gram penalty. Default behavior
                // is to use it whenever top_p>=1 (no nucleus filter needed).
                // `FORCE_GPU_SAMPLE=1` forces it on even when top_p<1 (slight
                // quality trade-off: nucleus filter skipped).
                let force_gpu = std::env::var("FORCE_GPU_SAMPLE").is_ok();
                let gpu_ok = force_gpu || seq.top_p >= 1.0;
                let t_tok = std::time::Instant::now();
                let tok = match Tensor::new(&[last_tokens[0]], &device).and_then(|t| t.unsqueeze(0))
                {
                    Ok(t) => t,
                    Err(e) => {
                        eprintln!("[batched engine] tok tensor err: {e}");
                        continue;
                    }
                };
                let t_fwd = std::time::Instant::now();
                let logits = match gem.model_mut().forward(&tok, positions[0]) {
                    Ok(l) => l,
                    Err(e) => {
                        eprintln!("[batched engine] single-seq forward failed: {e}");
                        for (_id, seq) in active.drain() {
                            let _ = seq.token_tx.try_send(StreamEvent::Error(e.to_string()));
                        }
                        continue;
                    }
                };
                if gpu_ok {
                    let t_sample = std::time::Instant::now();
                    let logits1d = match logits.squeeze(0) {
                        Ok(t) => t,
                        Err(e) => {
                            eprintln!("[batched engine] logits squeeze err: {e}");
                            continue;
                        }
                    };
                    let sampled = lumen_model::sampling::sample_token_gpu(
                        &logits1d,
                        seq.temperature,
                        seq.repeat_penalty,
                        &seq.generated,
                    );
                    match sampled {
                        Ok(t) => {
                            if debug_timing {
                                eprintln!(
                                    "  N=1 GPU: flags={:.2}ms tok={:.2}ms fwd={:.2}ms gpu_sample={:.2}ms",
                                    (t_tok - t_flags).as_secs_f64() * 1000.0,
                                    (t_fwd - t_tok).as_secs_f64() * 1000.0,
                                    (t_sample - t_fwd).as_secs_f64() * 1000.0,
                                    t_sample.elapsed().as_secs_f64() * 1000.0,
                                );
                            }
                            Some(t)
                        }
                        Err(e) => {
                            eprintln!("[batched engine] gpu sample err seq {}: {e}", ids[0]);
                            continue;
                        }
                    }
                } else {
                    // CPU fallback: materialize full vocab.
                    let t_cpu = std::time::Instant::now();
                    let _out = match logits
                        .squeeze(0)
                        .and_then(|t| t.to_dtype(DType::F32))
                        .and_then(|t| t.to_vec1::<f32>())
                    {
                        Ok(v) => v,
                        Err(e) => {
                            eprintln!("[batched engine] single logits cpu err: {e}");
                            continue;
                        }
                    };
                    if debug_timing {
                        eprintln!(
                            "  N=1 CPU: flags={:.2}ms tok={:.2}ms fwd={:.2}ms cpu={:.2}ms",
                            (t_tok - t_flags).as_secs_f64() * 1000.0,
                            (t_fwd - t_tok).as_secs_f64() * 1000.0,
                            (t_cpu - t_fwd).as_secs_f64() * 1000.0,
                            t_cpu.elapsed().as_secs_f64() * 1000.0,
                        );
                    }
                    // Hand the flat logits through the shared path below by
                    // stashing into a one-element Vec — re-enter via flat.
                    // To keep control-flow simple, just emit via the legacy
                    // sampler inline here and skip the shared path.
                    let next_tok = match lumen_model::sampling::sample_token_cpu(
                        &_out,
                        seq.temperature,
                        seq.top_p,
                        seq.repeat_penalty,
                        &seq.generated,
                    ) {
                        Ok(t) => t,
                        Err(e) => {
                            eprintln!("[batched engine] cpu sample err seq {}: {e}", ids[0]);
                            continue;
                        }
                    };
                    Some(next_tok)
                }
            } else {
                None
            };

            let flat: Vec<f32> = if let Some(_tok) = n1_gpu_tok {
                // Single-seq path already sampled; skip shared flat/sample.
                Vec::new()
            } else if ids.len() == 1 {
                // unreachable (kept for type exhaustiveness)
                Vec::new()
            } else {
                // Promotion: any seq that was running under SDPA-only with
                // paged_store_enabled=false has stale paged state. Bulk-copy
                // its kv_cache into paged before the batched dispatch.
                for id in &ids {
                    if dirty_paged.remove(id) {
                        if let Err(e) = gem.model_mut().migrate_seq_to_paged(*id) {
                            eprintln!("[batched engine] migrate_seq_to_paged({id}) failed: {e}");
                        }
                    }
                }
                gem.model_mut().set_use_compressed_for_attn(true);
                gem.model_mut().set_paged_store_enabled(true);
                let toks = match Tensor::new(last_tokens.as_slice(), &device)
                    .and_then(|t| t.reshape((ids.len(), 1)))
                {
                    Ok(t) => t,
                    Err(e) => {
                        eprintln!("[batched engine] input tensor err: {e}");
                        continue;
                    }
                };
                let logits = match gem
                    .model_mut()
                    .forward_batched_decode_v2(&toks, &ids, &positions)
                {
                    Ok(l) => l,
                    Err(e) => {
                        eprintln!("[batched engine] forward_batched_decode_v2 failed: {e}");
                        for (_id, seq) in active.drain() {
                            let _ = seq.token_tx.try_send(StreamEvent::Error(e.to_string()));
                        }
                        continue;
                    }
                };
                match logits
                    .to_dtype(DType::F32)
                    .and_then(|t| t.flatten_all())
                    .and_then(|t| t.to_vec1())
                {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("[batched engine] logits cpu err: {e}");
                        continue;
                    }
                }
            };
            let vocab = if flat.is_empty() || ids.is_empty() {
                0
            } else {
                flat.len() / ids.len()
            };

            // 3. Per-seq sampling: N=1 fast path already has a token in
            //    `n1_gpu_tok`; N>=2 materialized logits into `flat` for CPU
            //    sampling. Emit Delta and check termination per seq.
            let mut to_remove: Vec<u64> = Vec::new();
            for (row, &id) in ids.iter().enumerate() {
                let seq = active.get_mut(&id).unwrap();
                let next_tok = if let Some(tok) = n1_gpu_tok {
                    tok
                } else {
                    let row_slice = &flat[row * vocab..(row + 1) * vocab];
                    match lumen_model::sampling::sample_token_cpu(
                        row_slice,
                        seq.temperature,
                        seq.top_p,
                        seq.repeat_penalty,
                        &seq.generated,
                    ) {
                        Ok(t) => t,
                        Err(e) => {
                            eprintln!("[batched engine] sample err seq {id}: {e}");
                            continue;
                        }
                    }
                };
                seq.generated.push(next_tok);
                seq.last_token = next_tok;
                seq.position += 1;

                // Detokenize + emit incremental text delta.
                if let Ok(text) = gem.decode(&seq.generated) {
                    if !text.contains('\u{FFFD}') && text.len() > seq.prev_text.len() {
                        let delta = text[seq.prev_text.len()..].to_string();
                        if !delta.is_empty() {
                            let _ = seq.token_tx.try_send(StreamEvent::Delta(delta));
                            seq.prev_text = text;
                        }
                    }
                }

                if seq.eos_tokens.contains(&next_tok) || seq.generated.len() >= seq.max_new {
                    let decode_ms = seq.decode_start.elapsed().as_secs_f64() * 1000.0;
                    let n_gen = seq.generated.len();
                    let per_seq_tps = n_gen as f64 / (decode_ms / 1000.0);
                    eprintln!(
                        "[batched engine] seq {id} done: {n_gen} tokens in {decode_ms:.0}ms ({per_seq_tps:.1} tok/s)",
                    );
                    let _ = seq.token_tx.try_send(StreamEvent::Done {
                        prompt_tokens: seq.prompt_tokens,
                        completion_tokens: n_gen as u32,
                    });
                    to_remove.push(id);
                }
            }
            for id in to_remove {
                active.remove(&id);
                dirty_paged.remove(&id);
            }

            let step_ms = t_step.elapsed().as_secs_f64() * 1000.0;
            let agg_tps = ids.len() as f64 / (step_ms / 1000.0);
            eprintln!(
                "[batched engine] step: N={} latency={:.1}ms aggregate={:.1} tok/s",
                ids.len(),
                step_ms,
                agg_tps,
            );
        }
    }

    // ── Qwen35Moe continuous-batching scheduler ──────────────────────────────

    /// Continuous-batching scheduler for Qwen35Moe streaming requests.
    /// N=1 → single-seq SDPA path via `decode_step_batch([1, …])`;
    /// N≥2 → sequential-per-seq batch (Phase 1 — SSM state clobbered, known limitation).
    #[cfg(feature = "qwen3_5_moe")]
    async fn run_batched_qwen35(&mut self, rx: &mut mpsc::Receiver<EngineRequest>) {
        use candle_core::DType;
        use std::collections::HashMap;

        let max_batch: usize = std::env::var("PAGED_MAX_BATCH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);

        let mut active: HashMap<u64, ActiveSeqState> = HashMap::new();
        let mut next_seq_id: u64 = 1;

        eprintln!("[qwen35 batched] active scheduler (max_batch={max_batch})");

        loop {
            // 1. Admit new streaming requests, up to max_batch.
            while active.len() < max_batch {
                match rx.try_recv() {
                    Ok(EngineRequest::StreamingChatCompletion { req, token_tx }) => {
                        match self.start_streaming_seq_qwen35(
                            next_seq_id,
                            &req.messages,
                            req.max_tokens,
                            req.temperature,
                            req.top_p,
                            req.thinking,
                            token_tx.clone(),
                        ) {
                            Ok(seq) => { active.insert(next_seq_id, seq); next_seq_id += 1; }
                            Err(e) => { let _ = token_tx.try_send(StreamEvent::Error(e.to_string())); }
                        }
                    }
                    Ok(EngineRequest::StreamingAnthropicMessages { req, token_tx }) => {
                        let mut messages: Vec<ChatMessage> = Vec::new();
                        if let Some(ref system) = req.system {
                            let system_text = match system {
                                AnthropicSystem::Text(s) => s.clone(),
                                AnthropicSystem::Blocks(blocks) => blocks
                                    .iter()
                                    .filter_map(|b| b.text.clone())
                                    .collect::<Vec<_>>()
                                    .join("\n"),
                            };
                            if !system_text.is_empty() {
                                messages.push(ChatMessage { role: "system".into(), content: system_text });
                            }
                        }
                        for msg in &req.messages {
                            messages.push(ChatMessage { role: msg.role.clone(), content: msg.content.as_text() });
                        }
                        match self.start_streaming_seq_qwen35(
                            next_seq_id, &messages, req.max_tokens, req.temperature, 0.9, req.thinking, token_tx.clone(),
                        ) {
                            Ok(seq) => { active.insert(next_seq_id, seq); next_seq_id += 1; }
                            Err(e) => { let _ = token_tx.try_send(StreamEvent::Error(e.to_string())); }
                        }
                    }
                    Ok(other) => { self.dispatch_request_sequential(other); }
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => return,
                }
            }

            if active.is_empty() {
                match rx.recv().await {
                    Some(req) => {
                        if matches!(req, EngineRequest::StreamingChatCompletion { .. } | EngineRequest::StreamingAnthropicMessages { .. }) {
                            match req {
                                EngineRequest::StreamingChatCompletion { req, token_tx } => {
                                    match self.start_streaming_seq_qwen35(
                                        next_seq_id, &req.messages, req.max_tokens, req.temperature, req.top_p, req.thinking, token_tx.clone(),
                                    ) {
                                        Ok(seq) => { active.insert(next_seq_id, seq); next_seq_id += 1; }
                                        Err(e) => { let _ = token_tx.try_send(StreamEvent::Error(e.to_string())); }
                                    }
                                }
                                EngineRequest::StreamingAnthropicMessages { req, token_tx } => {
                                    let mut messages: Vec<ChatMessage> = Vec::new();
                                    if let Some(ref system) = req.system {
                                        let system_text = match system {
                                            AnthropicSystem::Text(s) => s.clone(),
                                            AnthropicSystem::Blocks(blocks) => blocks.iter().filter_map(|b| b.text.clone()).collect::<Vec<_>>().join("\n"),
                                        };
                                        if !system_text.is_empty() {
                                            messages.push(ChatMessage { role: "system".into(), content: system_text });
                                        }
                                    }
                                    for msg in &req.messages { messages.push(ChatMessage { role: msg.role.clone(), content: msg.content.as_text() }); }
                                    match self.start_streaming_seq_qwen35(
                                        next_seq_id, &messages, req.max_tokens, req.temperature, 0.9, req.thinking, token_tx.clone(),
                                    ) {
                                        Ok(seq) => { active.insert(next_seq_id, seq); next_seq_id += 1; }
                                        Err(e) => { let _ = token_tx.try_send(StreamEvent::Error(e.to_string())); }
                                    }
                                }
                                _ => unreachable!(),
                            }
                        } else {
                            self.dispatch_request_sequential(req);
                        }
                    }
                    None => return,
                }
                continue;
            }

            // 2. Decode step.
            let ids: Vec<u64> = active.keys().copied().collect();
            let last_tokens: Vec<u32> = ids.iter().map(|id| active[id].last_token).collect();
            let positions: Vec<usize> = ids.iter().map(|id| active[id].position).collect();

            let q = match &mut self.backend {
                #[cfg(feature = "turboquant-gpu")]
                ModelBackend::Qwen35Moe(q) => q,
                _ => { eprintln!("[qwen35 batched] non-Qwen35 backend unreachable"); return; }
            };

            let t_step = std::time::Instant::now();
            let breakdown = std::env::var("LUMEN_HTTP_BREAKDOWN").map(|v| v == "1").unwrap_or(false);

            // ── MTP fast path: single-seq + greedy + MTP enabled ─────────────
            // Multi-seq batching and non-greedy sampling stay on the existing
            // decode_step_batch path (no semantic change). The MTP step emits
            // 1 + n_accepted tokens in one engine step, so we update seq state
            // and continue without touching the normal sampling/emit code below.
            let mtp_eligible = ids.len() == 1
                && q.has_mtp()
                && {
                    let s = &active[&ids[0]];
                    s.temperature == 0.0 && (s.repeat_penalty - 1.0).abs() < 1e-9
                };
            if std::env::var("LUMEN_SPEC_DEBUG").is_ok() {
                let s = active.get(&ids[0]);
                eprintln!(
                    "[mtp_debug] ids.len={} has_mtp={} temp={:?} rp={:?} → eligible={}",
                    ids.len(),
                    q.has_mtp(),
                    s.map(|s| s.temperature),
                    s.map(|s| s.repeat_penalty),
                    mtp_eligible,
                );
            }
            if mtp_eligible {
                let id = ids[0];
                let n_max: usize = std::env::var("LUMEN_SPEC_DRAFT_N_MAX")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(3);
                match q.mtp_step(id, last_tokens[0], positions[0], n_max) {
                    Ok((committed, n_drafted, n_accepted)) => {
                        let seq = active.get_mut(&id).unwrap();
                        // Commit each token: append, advance position, stream, check EOS.
                        let mut hit_terminator = false;
                        for tok in &committed {
                            seq.generated.push(*tok);
                            seq.last_token = *tok;
                            seq.position += 1;
                            // Stream decode + emit (text delta).
                            let q2 = match &self.backend {
                                #[cfg(feature = "turboquant-gpu")]
                                ModelBackend::Qwen35Moe(qq) => qq,
                                _ => unreachable!(),
                            };
                            if let Ok(text) = q2.decode(&seq.generated) {
                                if !text.contains('\u{FFFD}') && text.len() > seq.prev_text.len() {
                                    let delta = text[seq.prev_text.len()..].to_string();
                                    if !delta.is_empty() {
                                        let _ = seq.token_tx.try_send(StreamEvent::Delta(delta));
                                        seq.prev_text = text;
                                    }
                                }
                            }
                            if seq.eos_tokens.contains(tok) || seq.generated.len() >= seq.max_new {
                                hit_terminator = true;
                                break;
                            }
                        }
                        let step_ms = t_step.elapsed().as_secs_f64() * 1000.0;
                        eprintln!(
                            "[qwen35 batched] mtp_step seq {id}: drafted={n_drafted} accepted={n_accepted} \
                             emitted={} latency={:.1}ms ({:.1} tok/s)",
                            committed.len(),
                            step_ms,
                            committed.len() as f64 / (step_ms / 1000.0),
                        );
                        if hit_terminator {
                            let decode_ms = seq.decode_start.elapsed().as_secs_f64() * 1000.0;
                            let n_gen = seq.generated.len();
                            eprintln!(
                                "[qwen35 batched] seq {id} done: {n_gen} tokens in {decode_ms:.0}ms ({:.1} tok/s)",
                                n_gen as f64 / (decode_ms / 1000.0),
                            );
                            let _ = seq.token_tx.try_send(StreamEvent::Done {
                                prompt_tokens: seq.prompt_tokens,
                                completion_tokens: n_gen as u32,
                            });
                            active.remove(&id);
                            if let ModelBackend::Qwen35Moe(q) = &mut self.backend {
                                q.remove_sequence(id);
                            }
                        }
                        continue;
                    }
                    Err(e) => {
                        eprintln!(
                            "[qwen35 batched] mtp_step seq {id} failed: {e:#} \
                             — falling through to decode_step_batch for this iteration"
                        );
                        // Fall through to the existing single-seq path. The
                        // failure may have left MTP state inconsistent; future
                        // iterations may continue to fail and the dispatch
                        // will keep falling through — acceptable degrade.
                    }
                }
            }

            // Both N=1 and N≥2 go through decode_step_batch → [B, vocab] for CPU sampling.
            // N=1 keeps the single-seq KV path (no SSM pollution); N≥2 is sequential-per-seq.
            let logits = match q.decode_step_batch(&ids, &last_tokens, &positions) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("[qwen35 batched] decode_step_batch failed: {e}");
                    for (_id, seq) in active.drain() {
                        let _ = seq.token_tx.try_send(StreamEvent::Error(e.to_string()));
                    }
                    continue;
                }
            };
            let t_decode = if breakdown { Some(std::time::Instant::now()) } else { None };

            // Greedy fast path: when ALL seqs in this step are temperature==0 +
            // repeat_penalty==1.0 (no penalty), do GPU argmax over `[B, vocab]`
            // and transfer just `B*4` bytes instead of `B*vocab*4` bytes (~1MB
            // per seq). This was the dominant 60% of step latency at HTTP path.
            let all_greedy = ids.iter().all(|id| {
                let s = &active[id];
                s.temperature == 0.0 && (s.repeat_penalty - 1.0).abs() < 1e-9
            });

            // For mixed/non-greedy seqs we still need full logits on CPU.
            // For all-greedy: skip the full transfer entirely.
            let (flat, argmax_idx): (Vec<f32>, Option<Vec<u32>>) = if all_greedy {
                let am = match logits.argmax(candle_core::D::Minus1).and_then(|t| t.flatten_all()).and_then(|t| t.to_vec1::<u32>()) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("[qwen35 batched] argmax err: {e}");
                        continue;
                    }
                };
                if std::env::var("LUMEN_DECODE_TRACE").is_ok() {
                    eprintln!(
                        "[batch_decode_trace] positions={positions:?} last_tokens={last_tokens:?} argmax={am:?}"
                    );
                }
                (Vec::new(), Some(am))
            } else {
                let v = match logits.to_dtype(DType::F32).and_then(|t| t.flatten_all()).and_then(|t| t.to_vec1()) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("[qwen35 batched] logits cpu err: {e}");
                        continue;
                    }
                };
                (v, None)
            };
            let t_xfer = if breakdown { Some(std::time::Instant::now()) } else { None };
            let vocab = if all_greedy {
                logits.dims().last().copied().unwrap_or(0)
            } else if flat.is_empty() || ids.is_empty() {
                0
            } else {
                flat.len() / ids.len()
            };

            // 3. Per-seq CPU sampling + emit + EOS check.
            let mut sample_ms = 0.0f64;
            let mut decode_text_ms = 0.0f64;
            let mut send_ms = 0.0f64;
            let mut to_remove: Vec<u64> = Vec::new();
            for (row, &id) in ids.iter().enumerate() {
                let seq = active.get_mut(&id).unwrap();
                let _ = vocab; // suppress unused warning when all_greedy path skips slicing
                let ts0 = if breakdown { Some(std::time::Instant::now()) } else { None };
                let next_tok = if let Some(ref am) = argmax_idx {
                    am[row]
                } else {
                    let row_slice = &flat[row * vocab..(row + 1) * vocab];
                    match lumen_model::sampling::sample_token_cpu(
                        row_slice, seq.temperature, seq.top_p, seq.repeat_penalty, &seq.generated,
                    ) {
                        Ok(t) => t,
                        Err(e) => {
                            eprintln!("[qwen35 batched] sample err seq {id}: {e}");
                            continue;
                        }
                    }
                };
                let ts1 = if breakdown { Some(std::time::Instant::now()) } else { None };
                seq.generated.push(next_tok);
                seq.last_token = next_tok;
                seq.position += 1;

                let q2 = match &self.backend {
                    #[cfg(feature = "turboquant-gpu")]
                    ModelBackend::Qwen35Moe(q) => q,
                    _ => unreachable!(),
                };
                if let Ok(text) = q2.decode(&seq.generated) {
                    if !text.contains('\u{FFFD}') && text.len() > seq.prev_text.len() {
                        let delta = text[seq.prev_text.len()..].to_string();
                        if !delta.is_empty() {
                            let ts2 = if breakdown { Some(std::time::Instant::now()) } else { None };
                            let _ = seq.token_tx.try_send(StreamEvent::Delta(delta));
                            seq.prev_text = text;
                            if let (Some(a), Some(b)) = (ts1, ts2) {
                                decode_text_ms += b.duration_since(a).as_secs_f64() * 1000.0;
                                send_ms += ts2.unwrap().elapsed().as_secs_f64() * 1000.0;
                            }
                        }
                    }
                }
                if let (Some(a), Some(b)) = (ts0, ts1) {
                    sample_ms += b.duration_since(a).as_secs_f64() * 1000.0;
                }

                if seq.eos_tokens.contains(&next_tok) || seq.generated.len() >= seq.max_new {
                    let decode_ms = seq.decode_start.elapsed().as_secs_f64() * 1000.0;
                    let n_gen = seq.generated.len();
                    eprintln!(
                        "[qwen35 batched] seq {id} done: {n_gen} tokens in {decode_ms:.0}ms ({:.1} tok/s)",
                        n_gen as f64 / (decode_ms / 1000.0),
                    );
                    let _ = seq.token_tx.try_send(StreamEvent::Done {
                        prompt_tokens: seq.prompt_tokens,
                        completion_tokens: n_gen as u32,
                    });
                    to_remove.push(id);
                }
            }
            let step_ms = t_step.elapsed().as_secs_f64() * 1000.0;
            if breakdown {
                let decode_ms = t_decode.unwrap().duration_since(t_step).as_secs_f64() * 1000.0;
                let xfer_ms = t_xfer.unwrap().duration_since(t_decode.unwrap()).as_secs_f64() * 1000.0;
                eprintln!(
                    "[qwen35 batched] step: N={} total={:.1}ms (decode={:.1} xfer={:.1} sample={:.1} decode_text={:.1} send={:.1})",
                    ids.len(), step_ms, decode_ms, xfer_ms, sample_ms, decode_text_ms, send_ms,
                );
            } else {
                eprintln!(
                    "[qwen35 batched] step: N={} latency={:.1}ms agg={:.1} tok/s",
                    ids.len(), step_ms, ids.len() as f64 / (step_ms / 1000.0),
                );
            }
            for id in to_remove {
                active.remove(&id);
                if let ModelBackend::Qwen35Moe(q) = &mut self.backend {
                    q.remove_sequence(id);
                }
            }
        }
    }

    /// Prefill a new Qwen35Moe streaming sequence under `seq_id`.
    #[cfg(feature = "qwen3_5_moe")]
    fn start_streaming_seq_qwen35(
        &mut self,
        seq_id: u64,
        messages: &[ChatMessage],
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        thinking: bool,
        token_tx: mpsc::Sender<StreamEvent>,
    ) -> Result<ActiveSeqState> {
        use candle_core::Tensor;

        let q = match &mut self.backend {
            ModelBackend::Qwen35Moe(q) => q,
            _ => return Err(anyhow::anyhow!("start_streaming_seq_qwen35: wrong backend")),
        };
        let device = q.device().clone();
        let eos_tokens = q.eos_tokens().to_vec();

        let msg_pairs: Vec<(String, String)> = messages
            .iter()
            .map(|m| (m.role.clone(), m.content.clone()))
            .collect();
        let prompt_ids = q.build_chat_input(&msg_pairs, thinking)?;
        let prompt_tokens = prompt_ids.len() as u32;
        let max_new = if max_tokens == 0 { 256 } else { max_tokens };
        let repeat_penalty: f32 = std::env::var("REPEAT_PENALTY")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);

        q.init_sequence(seq_id);
        q.model_mut().set_current_seq_id(seq_id);

        // Prefill: feed prompt[..len-1] for KV warmup; last token enters the decode loop.
        // On any error, clean up the allocated sequence state before propagating.
        let t_prefill = std::time::Instant::now();
        let prefix = &prompt_ids[..prompt_ids.len().saturating_sub(1)];
        if !prefix.is_empty() {
            let result = Tensor::new(prefix, &device)
                .and_then(|t| t.unsqueeze(0))
                .and_then(|t| q.model_mut().forward_with_offset(&t, 0));
            if let Err(e) = result {
                q.remove_sequence(seq_id);
                return Err(anyhow::anyhow!("prefill seq {seq_id}: {e}"));
            }
        }
        // MTP: mirror the prefill into the MTP block's KV (no-op when MTP disabled).
        // Must run BEFORE any other forward call clears the trunk's captured
        // h_pre_norm. `mirror_prefill_into_mtp` consumes it via `take_h_pre_norm`.
        if q.has_mtp() && !prefix.is_empty() {
            if let Err(e) = q.mirror_prefill_into_mtp(seq_id, prefix) {
                eprintln!("[qwen35 batched] seq {seq_id} MTP prefill mirror failed: {e:#}");
            }
        }
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
        let position = prefix.len();
        let last_token = *prompt_ids.last().unwrap();

        eprintln!(
            "[qwen35 batched] seq {seq_id} prefill: {} tokens in {prefill_ms:.0}ms ({:.1} tok/s)",
            prompt_ids.len(),
            prompt_ids.len() as f64 / (prefill_ms / 1000.0),
        );

        Ok(ActiveSeqState {
            seq_id,
            token_tx,
            prompt_len: prompt_ids.len(),
            generated: Vec::new(),
            max_new,
            last_token,
            position,
            prev_text: String::new(),
            prompt_tokens,
            decode_start: std::time::Instant::now(),
            temperature,
            top_p: if top_p <= 0.0 { 0.9 } else { top_p },
            repeat_penalty,
            eos_tokens,
        })
    }

    /// Prefill a new streaming seq under `seq_id`; returns the active-state
    /// struct ready for the batched decode loop (caller inserts into the map).
    fn start_streaming_seq(
        &mut self,
        seq_id: u64,
        messages: &[ChatMessage],
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        thinking: bool,
        token_tx: mpsc::Sender<StreamEvent>,
    ) -> Result<ActiveSeqState> {
        use candle_core::Tensor;

        let gem = match &mut self.backend {
            ModelBackend::GemmaGguf(g) => g,
            _ => {
                return Err(anyhow::anyhow!(
                    "batched scheduler requires GemmaGguf backend"
                ));
            }
        };
        let device = gem.device().clone();

        let msg_pairs: Vec<(String, String)> = messages
            .iter()
            .map(|m| (m.role.clone(), m.content.clone()))
            .collect();
        let prompt_ids = gem.build_chat_input(&msg_pairs, thinking)?;
        let prompt_tokens = prompt_ids.len() as u32;
        let max_new = if max_tokens == 0 { 256 } else { max_tokens };
        let repeat_penalty: f32 = std::env::var("REPEAT_PENALTY")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);

        gem.model_mut().set_current_seq_id(seq_id);
        // Prefill always seeds paged state for this seq, regardless of the
        // batched-loop's current store-gating. Re-enable explicitly in case
        // a previous N=1 decode step turned it off.
        gem.model_mut().set_paged_store_enabled(true);

        // Prefill: feed prompt[..len-1], sample first token from logits[len-1].
        let t_prefill = std::time::Instant::now();
        let prefix = &prompt_ids[..prompt_ids.len() - 1];
        if !prefix.is_empty() {
            let t = Tensor::new(prefix, &device)?.unsqueeze(0)?;
            let _ = gem.model_mut().forward(&t, 0)?;
        }
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
        let position = prefix.len();
        let last_token = *prompt_ids.last().unwrap();

        eprintln!(
            "[batched engine] seq {seq_id} prefill: {} tokens in {prefill_ms:.0}ms ({:.1} tok/s)",
            prompt_ids.len(),
            prompt_ids.len() as f64 / (prefill_ms / 1000.0),
        );

        Ok(ActiveSeqState {
            seq_id,
            token_tx,
            prompt_len: prompt_ids.len(),
            generated: Vec::new(),
            max_new,
            last_token,
            position,
            prev_text: String::new(),
            prompt_tokens,
            decode_start: std::time::Instant::now(),
            temperature,
            top_p: if top_p <= 0.0 { 0.9 } else { top_p },
            repeat_penalty,
            eos_tokens: vec![1, 106], // Gemma: <eos>=1, <end_of_turn>=106
        })
    }
}

#[cfg(test)]
mod backend_mode_tests {
    use super::{resolve_backend_mode_from, BackendMode};
    use std::collections::HashMap;

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> =
            pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        move |name: &str| map.get(name).cloned()
    }

    #[test]
    fn default_is_candle() {
        assert_eq!(resolve_backend_mode_from(env(&[])), BackendMode::Candle);
    }

    #[test]
    fn explicit_mlx() {
        assert_eq!(
            resolve_backend_mode_from(env(&[("LUMEN_MODE", "mlx")])),
            BackendMode::Mlx
        );
    }

    #[test]
    fn explicit_candle() {
        assert_eq!(
            resolve_backend_mode_from(env(&[("LUMEN_MODE", "candle")])),
            BackendMode::Candle
        );
    }

    #[test]
    fn auto_resolves_to_candle() {
        assert_eq!(
            resolve_backend_mode_from(env(&[("LUMEN_MODE", "auto")])),
            BackendMode::Candle
        );
    }

    #[test]
    fn case_and_whitespace_insensitive() {
        assert_eq!(
            resolve_backend_mode_from(env(&[("LUMEN_MODE", "  MLX  ")])),
            BackendMode::Mlx
        );
        assert_eq!(
            resolve_backend_mode_from(env(&[("LUMEN_MODE", "Candle")])),
            BackendMode::Candle
        );
    }

    #[test]
    fn invalid_value_falls_back_to_candle() {
        assert_eq!(
            resolve_backend_mode_from(env(&[("LUMEN_MODE", "gpt-5")])),
            BackendMode::Candle
        );
    }

    #[test]
    fn legacy_use_mlx_one() {
        assert_eq!(
            resolve_backend_mode_from(env(&[("USE_MLX", "1")])),
            BackendMode::Mlx
        );
    }

    #[test]
    fn legacy_use_mlx_other_values_ignored() {
        // Only USE_MLX=1 historically meant "on". Other values fall through.
        assert_eq!(
            resolve_backend_mode_from(env(&[("USE_MLX", "true")])),
            BackendMode::Candle
        );
        assert_eq!(
            resolve_backend_mode_from(env(&[("USE_MLX", "0")])),
            BackendMode::Candle
        );
    }

    #[test]
    fn lumen_mode_takes_precedence_over_use_mlx() {
        // Explicit candle overrides legacy USE_MLX=1.
        assert_eq!(
            resolve_backend_mode_from(env(&[
                ("LUMEN_MODE", "candle"),
                ("USE_MLX", "1"),
            ])),
            BackendMode::Candle
        );
        // And the inverse — LUMEN_MODE=mlx without USE_MLX.
        assert_eq!(
            resolve_backend_mode_from(env(&[("LUMEN_MODE", "mlx")])),
            BackendMode::Mlx
        );
    }
}

#[cfg(test)]
mod arch_detection_tests {
    use super::detect_architecture;

    #[test]
    fn moe_checkpoints_route_to_qwen3_5_moe() {
        assert_eq!(
            detect_architecture("mlx-community/Qwen3.6-35B-A3B-mxfp4"),
            "qwen3_5_moe"
        );
        assert_eq!(
            detect_architecture("Qwen/Qwen3.6-35B-A3B"),
            "qwen3_5_moe"
        );
        assert_eq!(
            detect_architecture("Qwen/Qwen3-Next-80B-A3B"),
            "qwen3_5_moe"
        );
    }

    #[test]
    fn dense_27b_routes_to_qwen3_5_dense() {
        assert_eq!(
            detect_architecture("mlx-community/Qwen3.6-27B-4bit"),
            "qwen3_5_dense"
        );
        assert_eq!(detect_architecture("Qwen/Qwen3.6-27B"), "qwen3_5_dense");
        assert_eq!(
            detect_architecture("mlx-community/Qwen3.6-27B-bf16"),
            "qwen3_5_dense"
        );
    }

    #[test]
    fn explicit_dense_tag_wins_for_qwen3_5_family() {
        assert_eq!(
            detect_architecture("user/Qwen3.6-Some-Dense-variant"),
            "qwen3_5_dense"
        );
        assert_eq!(
            detect_architecture("user/Qwen3_5_dense_experimental"),
            "qwen3_5_dense"
        );
    }

    #[test]
    fn moe_marker_overrides_27b_substring() {
        // Defensive: if a MoE checkpoint happens to contain "27b" elsewhere in the
        // name (unlikely but possible), the explicit moe/a3b marker should still win.
        assert_eq!(
            detect_architecture("user/Qwen3.6-A3B-MoE-variant-rev27b"),
            "qwen3_5_moe"
        );
    }

    #[test]
    fn non_qwen35_models_keep_existing_routing() {
        // `gemma-4` strings now route to the new
        // native MLX path; the legacy Candle Gemma 1/2 path still wins
        // for `gemma-2b` / `gemma-7b` / `gemma-2-9b` etc.
        assert_eq!(
            detect_architecture("google/gemma-4-26B-A4B-it"),
            "gemma4_native"
        );
        assert_eq!(detect_architecture("google/gemma-2b"), "gemma4");
        assert_eq!(detect_architecture("google/gemma-2-9b-it"), "gemma4");
        assert_eq!(
            detect_architecture("Qwen/Qwen2.5-1.5B-Instruct"),
            "qwen2"
        );
        assert_eq!(
            detect_architecture("meta-llama/Llama-3-8B-Instruct"),
            "qwen2"
        );
    }

    #[test]
    fn gemma4_native_routing_variants() {
        // All four canonical Gemma 4 26B-A4B name patterns route to the
        // native MLX backend.
        assert_eq!(
            detect_architecture("/Users/me/models/gemma-4-26b-a4b-mlx-4bit"),
            "gemma4_native"
        );
        assert_eq!(
            detect_architecture("lmstudio-community/gemma4-26b-a4b"),
            "gemma4_native"
        );
        assert_eq!(detect_architecture("Gemma_4_text"), "gemma4_native");
        assert_eq!(detect_architecture("gemma4_26b"), "gemma4_native");
    }
}

/// Mirror of the in-loop struct so `start_streaming_seq` can return it.
pub(crate) struct ActiveSeqState {
    pub seq_id: u64,
    pub token_tx: mpsc::Sender<StreamEvent>,
    pub prompt_len: usize,
    pub generated: Vec<u32>,
    pub max_new: usize,
    pub last_token: u32,
    pub position: usize,
    pub prev_text: String,
    pub prompt_tokens: u32,
    pub decode_start: std::time::Instant,
    pub temperature: f32,
    pub top_p: f32,
    pub repeat_penalty: f32,
    /// EOS token IDs used by the batched decode loop for this sequence's backend.
    pub eos_tokens: Vec<u32>,
}
