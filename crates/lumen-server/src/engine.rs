use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use lumen_mlx::SamplingOverrides;
use lumen_mlx::chat_io::{
    AssistantToolCall, BackendStreamEvent, ChatTurn, ParsedResponse, ParsedToolCall,
    ResolvedToolChoice, ToolDef,
};
use lumen_model::gemma::GemmaModel;
use lumen_model::gemma_gguf::GemmaGgufModel;
use lumen_model::qwen::QwenModel;

#[cfg(feature = "qwen3_5_moe")]
use lumen_model::qwen3_5_moe::backend::Qwen35MoeBackend;

/// Gemma 4 26B-A4B native MLX backend wrapper. See
/// `crates/lumen-mlx/src/gemma4_backend.rs` for the trait-shape API
/// this dispatches into.
use crate::load_stats::ServerLoadStats;
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
///   1. `LUMEN_MODE=mlx|candle|auto`  — explicit selection
///   2. `USE_MLX=1`                     — legacy alias for `LUMEN_MODE=mlx`
///   3. default                         — Mlx when the `mlx-native` feature is
///                                        compiled in (mlx-rs gather_qmm path
///                                        is +57% tok/s vs Candle at short
///                                        prompts and 33× vs Candle at
///                                        PROMPT_LEN=2048); otherwise Candle.
///
/// `auto` follows the same rule as the unset case.
fn resolve_backend_mode() -> BackendMode {
    resolve_backend_mode_from(|name| std::env::var(name).ok())
}

/// Token-chunk size for the experimental batched engine's eager prefill.
///
/// Only consulted on the `BATCHED_ENGINE=1` + GemmaGguf path. Prompts whose
/// prefix is `<= chunk` keep the single-forward path unchanged; longer prompts
/// are forwarded in chunks of this size so one sequence's long prefill does not
/// monopolize a single giant forward and stall other batched sequences.
/// `0` or an unparseable value falls back to the default. Default: 512.
fn batched_prefill_chunk() -> usize {
    std::env::var("LUMEN_BATCHED_PREFILL_CHUNK")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(512)
}

/// Iteration-level chunked prefill gate (WS-E Lever 1 — head-of-line removal).
///
/// When set (`LUMEN_PAGED_BATCH_DECODE=1`), `start_streaming_seq` defers a
/// long prompt's prefill into the main batched loop: one chunk per iteration,
/// so already-decoding sequences keep advancing between chunks instead of
/// stalling for the full prefill. Default OFF → prefill runs synchronously
/// inside `start_streaming_seq` exactly as before (decode loop byte-identical).
#[inline]
fn iter_level_prefill_enabled() -> bool {
    std::env::var("LUMEN_PAGED_BATCH_DECODE").ok().as_deref() == Some("1")
}

/// Default backend when no env is set. mlx-native build → Mlx, otherwise Candle.
#[inline]
fn default_backend_mode() -> BackendMode {
    #[cfg(feature = "mlx-native")]
    {
        BackendMode::Mlx
    }
    #[cfg(not(feature = "mlx-native"))]
    {
        BackendMode::Candle
    }
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
            "candle" => return BackendMode::Candle,
            "auto" => return default_backend_mode(),
            other => {
                eprintln!(
                    "warn: LUMEN_MODE={other:?} unrecognized; falling back to default. \
                     Valid: mlx | candle | auto."
                );
                return default_backend_mode();
            }
        }
    }
    if matches!(get("USE_MLX").as_deref(), Some("1")) {
        return BackendMode::Mlx;
    }
    default_backend_mode()
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
    /// Unified mlx-native backend — covers Qwen 2.5 / 3.5 / 3.6 dense + MoE
    /// AND Gemma 4 26B-A4B. Family-specific dispatch happens inside
    /// `MlxBackend`; the engine sees one variant.
    Mlx(lumen_mlx::MlxBackend),
}

impl ModelBackend {
    /// Returns true when the backend should default `enable_thinking=true`
    /// for OpenAI-compat clients that don't send a thinking signal.
    ///
    /// Default is `false`. Previously hardcoded `true` for Gemma 4
    /// destabilized both imatrix-AWQ builds (channel-open over-amplified →
    /// infinite reasoning) and mlx-community uniform 4bit (channel weights
    /// too weak → degenerate system-prompt cycling). The opt-in env
    /// `LUMEN_BACKEND_THINKING_DEFAULT={on,1,true}` re-enables the default
    /// for operators running newer hybrid variants (e.g. v0.6.0
    /// `embed8-last13attn8`) whose channel weights are strong enough to
    /// tolerate it — and where the tool_call rank improves significantly
    /// when sampled from the thought-channel-open position rather than the
    /// empty-channel-close one. Clients can still override per-request via
    /// `chat_template_kwargs.enable_thinking`, `reasoning_effort`, or
    /// `thinking: true`.
    fn is_reasoning_first_family(&self) -> bool {
        static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *FLAG.get_or_init(|| {
            std::env::var("LUMEN_BACKEND_THINKING_DEFAULT")
                .map(|s| {
                    matches!(
                        s.trim().to_ascii_lowercase().as_str(),
                        "1" | "on" | "true" | "yes"
                    )
                })
                .unwrap_or(false)
        })
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        match self {
            Self::Qwen(m) => m.encode(text),
            Self::Gemma(m) => m.encode(text),
            Self::GemmaGguf(m) => m.encode(text),
            #[cfg(feature = "qwen3_5_moe")]
            Self::Qwen35Moe(m) => m.encode(text),
            Self::Mlx(m) => m.encode(text),
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
        ov: &SamplingOverrides,
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
            Self::Mlx(m) => m.generate(
                input_ids,
                max_new_tokens,
                temperature,
                top_p,
                ov,
                session_id,
            ),
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
    /// if the entry existed. MLX-only feature.
    fn drop_prefix_cache(&mut self, key: &str) -> bool {
        match self {
            Self::Mlx(m) => m.drop_prefix_cache(key),
            _ => false,
        }
    }

    /// Clear all A1 prefix-cache entries. Returns the number released.
    fn clear_prefix_cache(&mut self) -> usize {
        match self {
            Self::Mlx(m) => m.clear_prefix_cache(),
            _ => 0,
        }
    }

    fn chat(
        &mut self,
        messages: &[(String, String)],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        ov: &SamplingOverrides,
        thinking: bool,
        session_id: Option<&str>,
        tools: &[ToolDef<'_>],
        tool_choice: &ResolvedToolChoice<'_>,
        response_schema: Option<&serde_json::Value>,
    ) -> Result<ParsedResponse> {
        match self {
            Self::Qwen(m) => {
                let _ = (top_p, tools, tool_choice, response_schema);
                let visible = m.chat(messages, max_new_tokens, temperature)?;
                Ok(text_only_response(visible))
            }
            Self::Gemma(m) => {
                let _ = (top_p, tools, tool_choice, response_schema);
                let visible = m.chat(messages, max_new_tokens, temperature)?;
                Ok(text_only_response(visible))
            }
            Self::GemmaGguf(m) => {
                let _ = (top_p, tools, tool_choice, response_schema);
                let visible =
                    m.chat_with_options(messages, max_new_tokens, temperature, thinking)?;
                Ok(text_only_response(visible))
            }
            #[cfg(feature = "qwen3_5_moe")]
            Self::Qwen35Moe(m) => {
                let _ = (top_p, tools, tool_choice, response_schema);
                let visible = m.chat(messages, max_new_tokens, temperature, thinking)?;
                Ok(text_only_response(visible))
            }
            Self::Mlx(m) => m.chat(
                messages,
                max_new_tokens,
                temperature,
                top_p,
                ov,
                thinking,
                session_id,
                tools,
                tool_choice,
                response_schema,
            ),
        }
    }

    fn chat_streaming<F>(
        &mut self,
        messages: &[(String, String)],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        ov: &SamplingOverrides,
        thinking: bool,
        session_id: Option<&str>,
        tools: &[ToolDef<'_>],
        tool_choice: &ResolvedToolChoice<'_>,
        response_schema: Option<&serde_json::Value>,
        mut on_event: F,
    ) -> Result<ParsedResponse>
    where
        F: FnMut(BackendStreamEvent<'_>) -> Result<()>,
    {
        match self {
            Self::GemmaGguf(m) => {
                let _ = (top_p, tools, tool_choice, response_schema);
                let mut adapter = |chunk: &str| {
                    let _ = on_event(BackendStreamEvent::Text(chunk));
                };
                let visible = m.chat_streaming(
                    messages,
                    max_new_tokens,
                    temperature,
                    thinking,
                    &mut adapter,
                )?;
                Ok(text_only_response(visible))
            }
            // Fallback: generate all, send as one chunk
            Self::Qwen(m) => {
                let _ = (top_p, tools, tool_choice, response_schema);
                let text = m.chat(messages, max_new_tokens, temperature)?;
                on_event(BackendStreamEvent::Text(&text))?;
                Ok(text_only_response(text))
            }
            Self::Gemma(m) => {
                let _ = (top_p, tools, tool_choice, response_schema);
                let text = m.chat(messages, max_new_tokens, temperature)?;
                on_event(BackendStreamEvent::Text(&text))?;
                Ok(text_only_response(text))
            }
            #[cfg(feature = "qwen3_5_moe")]
            Self::Qwen35Moe(m) => {
                let _ = (top_p, tools, tool_choice, response_schema);
                let mut adapter = |chunk: &str| {
                    let _ = on_event(BackendStreamEvent::Text(chunk));
                };
                let visible = m.chat_streaming(
                    messages,
                    max_new_tokens,
                    temperature,
                    thinking,
                    &mut adapter,
                )?;
                Ok(text_only_response(visible))
            }
            Self::Mlx(m) => m.chat_streaming(
                messages,
                max_new_tokens,
                temperature,
                top_p,
                ov,
                thinking,
                session_id,
                tools,
                tool_choice,
                response_schema,
                on_event,
            ),
        }
    }
}

/// Wrap a plain visible-text response for backends that don't yet emit
/// structured tool_calls. Phase 1.3 keeps the API uniform; Phase 2 wires
/// Qwen 3.6 tool-call parsing to populate `tool_calls` for that family.
fn text_only_response(visible: String) -> ParsedResponse {
    ParsedResponse {
        visible,
        reasoning: String::new(),
        tool_calls: Vec::new(),
    }
}

impl ModelBackend {
    /// Structured-history dispatch — used when the request contains
    /// `assistant.tool_calls` or `role:"tool"` entries. Only the Mlx
    /// (Gemma 4) path gets the full structured renderer; legacy backends
    /// (Candle Qwen / Gemma / GemmaGguf / Qwen35Moe) flatten the history
    /// to plain `(role, content)` (dropping tool metadata) and call the
    /// existing chat path. Tool turns are surfaced as user-role
    /// `"[tool result] ..."` text so legacy paths don't reject them
    /// outright — quality may suffer until each backend gains a real
    /// tool-aware renderer.
    fn chat_from_history(
        &mut self,
        turns: &[ChatTurn<'_>],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        ov: &SamplingOverrides,
        thinking: bool,
        session_id: Option<&str>,
        tools: &[ToolDef<'_>],
        tool_choice: &ResolvedToolChoice<'_>,
        response_schema: Option<&serde_json::Value>,
    ) -> Result<ParsedResponse> {
        if let Self::Mlx(m) = self {
            return m.chat_from_history(
                turns,
                max_new_tokens,
                temperature,
                top_p,
                ov,
                thinking,
                session_id,
                tools,
                tool_choice,
                response_schema,
            );
        }
        // Legacy backends: flatten + delegate to plain chat.
        let plain = flatten_turns_for_plain_backend(turns);
        self.chat(
            &plain,
            max_new_tokens,
            temperature,
            top_p,
            ov,
            thinking,
            session_id,
            tools,
            tool_choice,
            response_schema,
        )
    }

    /// Phase 1.5: streaming variant of `chat_from_history`. Mlx path
    /// (Gemma 4) routes through the structured renderer; legacy
    /// backends flatten + delegate to `chat_streaming` so the request
    /// doesn't error out.
    fn chat_streaming_from_history<F>(
        &mut self,
        turns: &[ChatTurn<'_>],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        ov: &SamplingOverrides,
        thinking: bool,
        session_id: Option<&str>,
        tools: &[ToolDef<'_>],
        tool_choice: &ResolvedToolChoice<'_>,
        response_schema: Option<&serde_json::Value>,
        on_event: F,
    ) -> Result<ParsedResponse>
    where
        F: FnMut(BackendStreamEvent<'_>) -> Result<()>,
    {
        if let Self::Mlx(m) = self {
            return m.chat_streaming_from_history(
                turns,
                max_new_tokens,
                temperature,
                top_p,
                ov,
                thinking,
                session_id,
                tools,
                tool_choice,
                response_schema,
                on_event,
            );
        }
        let plain = flatten_turns_for_plain_backend(turns);
        self.chat_streaming(
            &plain,
            max_new_tokens,
            temperature,
            top_p,
            ov,
            thinking,
            session_id,
            tools,
            tool_choice,
            response_schema,
            on_event,
        )
    }
}

fn flatten_turns_for_plain_backend(turns: &[ChatTurn<'_>]) -> Vec<(String, String)> {
    turns
        .iter()
        .map(|t| match t {
            ChatTurn::System(s) => ("system".to_string(), (*s).to_string()),
            ChatTurn::User(s) => ("user".to_string(), (*s).to_string()),
            ChatTurn::Assistant { text, .. } => ("assistant".to_string(), (*text).to_string()),
            ChatTurn::Tool { content, .. } => {
                ("user".to_string(), format!("[tool result] {}", *content))
            }
        })
        .collect()
}

/// Inference engine wrapping a model backend and tokenizer.
pub struct InferenceEngine {
    backend: ModelBackend,
    model_id: String,
    /// Shared process-lifetime serving counters for `GET /v1/loads`. Cloned
    /// into `EngineHandle` at startup so the route reads the same atomics the
    /// engine bumps at each chat completion.
    load_stats: Arc<ServerLoadStats>,
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
    } else if lower.contains("qwen3.6")
        || lower.contains("qwen3_5")
        || lower.contains("qwen3-next")
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
            // MlxBackend::load now handles arch detection internally —
            // Qwen3.5 family vs Gemma 4 vs (future) any other mlx-native
            // model. The engine only sees one variant.
            eprintln!("Loading MLX backend (mode={mode:?}): {model_id}");
            let backend = lumen_mlx::MlxBackend::load(model_id)?;
            eprintln!("[mlx] family={:?}", backend.kind());
            let cfg_summary = backend.runtime_config_summary();
            if !cfg_summary.is_empty() {
                eprintln!("[mlx-config] {cfg_summary}");
            }
            return Ok(Self {
                backend: ModelBackend::Mlx(backend),
                model_id: model_id.to_string(),
                load_stats: ServerLoadStats::new_arc(model_id),
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
                load_stats: ServerLoadStats::new_arc(model_id),
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
            use lumen_metal::affine4_gpu::Affine4Context;
            use lumen_metal::mxfp4_gpu::MxFp4Context;
            use lumen_model::qwen3_5_moe::backend::Qwen35MoeBackend;
            use std::path::PathBuf;

            let shard_dir = std::env::var("LUMEN_QWEN35_SHARDS")
                .map(PathBuf::from)
                .map_err(|_| {
                    anyhow::anyhow!(
                        "{arch} requires LUMEN_QWEN35_SHARDS=<dir> pointing to the \
                     shard directory (config.json + model.safetensors.index.json + shards)"
                    )
                })?;
            let gpu_ctx = Arc::new(MxFp4Context::new()?);
            let mut backend = if arch == "qwen3_5_dense" {
                // 27B dense ships uniform 4-bit affine quantization → also wire up
                // the GPU-resident Affine4Context so projections stay on device.
                let affine4_ctx = Arc::new(Affine4Context::new()?);
                Qwen35MoeBackend::load_with_affine4(model_id, &shard_dir, gpu_ctx, affine4_ctx)?
            } else {
                Qwen35MoeBackend::load(model_id, &shard_dir, gpu_ctx)?
            };

            return Ok(Self {
                backend: ModelBackend::Qwen35Moe(backend),
                model_id: model_id.to_string(),
                load_stats: ServerLoadStats::new_arc(model_id),
            });
        }

        // Note: `arch == "gemma4_native"` is now handled by `MlxBackend::load`
        // upstream (see the `if mode == BackendMode::Mlx` early-return). The
        // Candle fallback path below only sees non-mlx arches.

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
            load_stats: ServerLoadStats::new_arc(model_id),
        })
    }

    /// Clone the shared lifetime-stats accumulator. Used at startup to hand
    /// the same `Arc` to `EngineHandle` so `GET /v1/loads` reads the counters
    /// the engine bumps.
    pub fn load_stats(&self) -> Arc<ServerLoadStats> {
        Arc::clone(&self.load_stats)
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
        let _ = self.backend.chat(
            &messages,
            3,
            0.0,
            1.0,
            &SamplingOverrides::default(),
            false,
            None,
            &[],
            &ResolvedToolChoice::Auto,
            None,
        )?;
        eprintln!(
            "  pass 1 done ({:.0}ms)",
            t.elapsed().as_secs_f64() * 1000.0
        );

        // One more decode step to ensure pipeline is warm
        let messages = vec![("user".to_string(), "Hi".to_string())];
        let _ = self.backend.chat(
            &messages,
            2,
            0.0,
            1.0,
            &SamplingOverrides::default(),
            false,
            None,
            &[],
            &ResolvedToolChoice::Auto,
            None,
        )?;

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
        // Detect turn-2+ shape: any message carries tool_calls (assistant
        // replay) or role=="tool" (tool result). When present we route
        // through the structured `chat_from_history` path which can stitch
        // these into the canonical Gemma 4 model turn. Otherwise the
        // simple flat `(role, content)` path is fastest and avoids any
        // chat-template behavior change for non-tool flows.
        let needs_structured = needs_structured_history(&req.messages);

        let prompt_bytes: usize = req.messages.iter().map(|m| m.content.len()).sum();
        let tools_owned = openai_tools_to_defs(req.tools.as_deref());
        eprintln!(
            "[chat] msgs={} prompt_bytes={} max_tokens={} thinking={} stream={} tools={} structured={}",
            req.messages.len(),
            prompt_bytes,
            req.max_tokens,
            req.enable_thinking_with_backend_default(self.backend.is_reasoning_first_family()),
            req.stream,
            tools_owned.len(),
            needs_structured,
        );

        // Build the prompt-token-count input regardless of path. For the
        // plain path this is the same `(role, content)` vector the
        // backend receives; for the structured path it's a flattened
        // view used only for billing token estimates (the actual prompt
        // tokenization happens inside Gemma4Backend with the rich layout).
        let mut messages: Vec<(String, String)> = req
            .messages
            .iter()
            .map(|m| (m.role.clone(), m.content.clone()))
            .collect();
        lumen_mlx::chat_io::strip_client_meta_wrappers_flat(&mut messages);

        // Owning storage for `ChatTurn` borrows when routing structured.
        let arg_values: Vec<serde_json::Value> = if needs_structured {
            req.messages
                .iter()
                .flat_map(|m| {
                    m.tool_calls
                        .iter()
                        .flat_map(|calls| calls.iter())
                        .map(|c| {
                            // OpenAI ships arguments as a JSON-encoded
                            // string. Parse to a Value so the renderer can
                            // walk the structure; fall back to {} on
                            // malformed input rather than failing the whole
                            // request.
                            serde_json::from_str(&c.function.arguments)
                                .unwrap_or_else(|_| serde_json::json!({}))
                        })
                        .collect::<Vec<_>>()
                })
                .collect()
        } else {
            Vec::new()
        };
        let assistant_tc_buf: Vec<Vec<AssistantToolCall<'_>>> = if needs_structured {
            let mut next_arg = 0;
            req.messages
                .iter()
                .map(|m| {
                    m.tool_calls
                        .as_ref()
                        .map(|calls| {
                            calls
                                .iter()
                                .map(|c| {
                                    let av = &arg_values[next_arg];
                                    next_arg += 1;
                                    AssistantToolCall {
                                        id: c.id.as_str(),
                                        name: c.function.name.as_str(),
                                        arguments: av,
                                    }
                                })
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default()
                })
                .collect()
        } else {
            Vec::new()
        };

        let tool_choice =
            resolve_openai_tool_choice(req.tool_choice.as_ref(), !tools_owned.is_empty());
        // Resolve enable_thinking once — the backend hint covers the
        // common case where OpenAI-compat clients send placeholder model
        // ids ("gpt-3.5-turbo") that hide the actual loaded family.
        let thinking_on =
            req.enable_thinking_with_backend_default(self.backend.is_reasoning_first_family());
        let ov = req.sampling_overrides();
        // Wall-clock around the full generation for the `/v1/loads` last
        // tok/s gauge. Backend `GenerateStats` carries a finer decode-only
        // rate, but it is not threaded back here; this end-to-end rate is the
        // honest, cheap-to-measure figure for observability.
        let gen_started = Instant::now();
        let mut parsed = if needs_structured {
            let mut turns: Vec<ChatTurn<'_>> = req
                .messages
                .iter()
                .enumerate()
                .map(|(i, m)| match m.role.as_str() {
                    "system" | "System" | "SYSTEM" => ChatTurn::System(m.content.as_str()),
                    "user" | "User" | "USER" => ChatTurn::User(m.content.as_str()),
                    "tool" => ChatTurn::Tool {
                        tool_call_id: m.tool_call_id.as_deref().unwrap_or(""),
                        name: m.name.as_deref(),
                        content: m.content.as_str(),
                    },
                    _ => ChatTurn::Assistant {
                        text: m.content.as_str(),
                        tool_calls: assistant_tc_buf.get(i).map(Vec::as_slice).unwrap_or(&[]),
                    },
                })
                .collect();
            lumen_mlx::chat_io::strip_client_meta_wrappers(&mut turns);
            self.backend.chat_from_history(
                &turns,
                req.max_tokens,
                req.temperature,
                req.top_p,
                &ov,
                thinking_on,
                req.session_id.as_deref(),
                &tools_owned,
                &tool_choice,
                req.response_json_schema().as_ref(),
            )?
        } else {
            self.backend.chat(
                &messages,
                req.max_tokens,
                req.temperature,
                req.top_p,
                &ov,
                thinking_on,
                req.session_id.as_deref(),
                &tools_owned,
                &tool_choice,
                req.response_json_schema().as_ref(),
            )?
        };

        let prompt_tokens = self
            .backend
            .count_chat_prompt_tokens(&messages, thinking_on);
        // Bug A: resolve abbreviated tool names by unique suffix match.
        remap_tool_call_names(&mut parsed.tool_calls, &tools_owned);
        // Stop sequences: truncate the visible text at the earliest match so
        // token counts and the returned content both reflect the trim. No-op
        // when no stops were requested. finish_reason stays "stop" (the
        // no-tool-call default below).
        if !ov.stop.is_empty() {
            let mut earliest = None;
            for s in &ov.stop {
                if let Some(i) = parsed.visible.find(s.as_str()) {
                    earliest = Some(earliest.map_or(i, |e: usize| e.min(i)));
                }
            }
            if let Some(i) = earliest {
                parsed.visible.truncate(i);
            }
        }
        // response_format: llguidance's `from_json_schema` grammar shapes the
        // JSON correctly but never reports is_stopped/is_accepting at the
        // closing brace (confirmed: both stay false through completion), so
        // it doesn't force EOS and the model free-runs trailing prose after
        // the `}`. Deterministically truncate the non-streaming answer to the
        // first complete balanced JSON value so response_format yields clean
        // JSON. No-op (keeps full text) if no complete value is present.
        if req.response_json_schema().is_some()
            && let Some(end) = first_json_value_end(&parsed.visible)
        {
            parsed.visible.truncate(end);
        }
        let completion_tokens =
            completion_tokens_with_tools(&self.backend, &parsed.visible, &parsed.tool_calls);

        // Decide finish_reason + message shape from whether the model
        // emitted any tool_calls. OpenAI spec: when tool_calls present,
        // content may be null; finish_reason="tool_calls".
        let has_tool_calls = !parsed.tool_calls.is_empty();
        // Strip Gemma 4 chat-template role label (`thought\n` / `thought `) from
        // the start of the reasoning channel — it's a template artifact, not
        // user-visible reasoning content. Matches vLLM's reasoning parser.
        let reasoning_trimmed = {
            let r = parsed.reasoning.trim();
            r.strip_prefix("thought\n")
                .or_else(|| r.strip_prefix("thought "))
                .unwrap_or(r)
                .trim()
                .to_string()
        };
        let has_reasoning = !reasoning_trimmed.is_empty();
        let (content, tool_calls) = if has_tool_calls {
            let visible = parsed.visible.trim();
            let content = if visible.is_empty() {
                None
            } else {
                Some(visible.to_string())
            };
            (
                content,
                Some(parsed_to_openai_tool_calls(&parsed.tool_calls)),
            )
        } else {
            (Some(parsed.visible.clone()), None)
        };
        // Reasoning placement. Default mirrors Ollama's OpenAI-compat layer
        // (`ollama/openai/openai.go`): thinking goes ONLY into the `reasoning`
        // field and `content` carries the visible answer alone — no
        // `<think>…</think>` envelope. This keeps Lumen byte-compatible with
        // Ollama so clients (Ayla) render identically against either backend.
        // Set `LUMEN_REASONING_IN_CONTENT=1` to restore the legacy dual
        // emission (also prepend a `<think>…</think>` envelope to content for
        // text-tag-only clients).
        let content = if has_reasoning && reasoning_in_content() {
            let envelope = format!("<think>\n{reasoning_trimmed}\n</think>\n\n");
            Some(match content {
                Some(c) if !c.is_empty() => format!("{envelope}{c}"),
                _ => envelope,
            })
        } else {
            content
        };
        let reasoning = if has_reasoning {
            Some(reasoning_trimmed)
        } else {
            None
        };
        let finish_reason = if has_tool_calls { "tool_calls" } else { "stop" };

        // Observability: bump lifetime counters for `GET /v1/loads`.
        let elapsed_s = gen_started.elapsed().as_secs_f64();
        let tok_per_sec = if elapsed_s > 0.0 {
            completion_tokens as f64 / elapsed_s
        } else {
            0.0
        };
        self.load_stats
            .record(prompt_tokens as u64, completion_tokens as u64, tok_per_sec);

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
                    tool_calls,
                    reasoning,
                },
                finish_reason: finish_reason.into(),
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

        let ov = req.sampling_overrides();
        let output_ids = self.backend.generate(
            &input_ids,
            req.max_tokens,
            req.temperature,
            req.top_p,
            &ov,
            req.session_id.as_deref(),
        )?;
        let completion_tokens = output_ids.len() as u32;
        let mut text = self.backend.decode(&output_ids)?;
        // Stop sequences: truncate the decoded text at the earliest match.
        // No-op when no stops were requested. finish_reason stays "stop".
        if !ov.stop.is_empty() {
            let mut earliest = None;
            for s in &ov.stop {
                if let Some(i) = text.find(s.as_str()) {
                    earliest = Some(earliest.map_or(i, |e: usize| e.min(i)));
                }
            }
            if let Some(i) = earliest {
                text.truncate(i);
            }
        }

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
        let tools_owned = anthropic_tools_to_defs(req.tools.as_deref());
        let needs_structured = anthropic_needs_structured_history(&req.messages);
        let ov = req.sampling_overrides();
        let tool_choice =
            resolve_anthropic_tool_choice(req.tool_choice.as_ref(), !tools_owned.is_empty());

        let system_text = req
            .system
            .as_ref()
            .map(|sys| match sys {
                AnthropicSystem::Text(s) => s.clone(),
                AnthropicSystem::Blocks(blocks) => blocks
                    .iter()
                    .filter_map(|b| b.text.clone())
                    .collect::<Vec<_>>()
                    .join("\n"),
            })
            .filter(|s| !s.is_empty());

        // Flat `(role, content)` view — used as the plain-path input AND
        // by `count_chat_prompt_tokens` regardless of which dispatch path
        // we take below. For the structured path it's a token-budget
        // approximation only; the actual prompt may include more tokens
        // (tool definitions + tool_response wrapping).
        let mut messages: Vec<(String, String)> = Vec::new();
        if let Some(ref s) = system_text {
            messages.push(("system".into(), s.clone()));
        }
        for msg in &req.messages {
            messages.push((msg.role.clone(), msg.content.as_text()));
        }
        lumen_mlx::chat_io::strip_client_meta_wrappers_flat(&mut messages);

        let mut parsed = if needs_structured {
            // Owning storage for ChatTurn::Assistant.tool_calls borrows.
            // Anthropic ships `tool_use.input` as a real JSON Value, so we
            // don't need to parse strings (vs OpenAI's JSON-encoded
            // arguments). Per-message expansion: one assistant message
            // with text + N tool_use blocks becomes one ChatTurn::Assistant;
            // one user message with tool_result blocks expands to N
            // ChatTurn::Tool entries plus any remaining text becomes a
            // ChatTurn::User.
            // Per-message extraction with direct match on the content
            // variant — avoids the `Cow::synthesize` path through
            // `.blocks()` which would return a temporary slice and cause
            // borrow-checker grief on subsequent ChatTurn references.
            let assistant_text_buf: Vec<String> = req
                .messages
                .iter()
                .map(|m| {
                    if m.role != "assistant" {
                        return String::new();
                    }
                    match &m.content {
                        AnthropicContent::Text(s) => s.clone(),
                        AnthropicContent::Blocks(blocks) => blocks
                            .iter()
                            .filter_map(|b| match b {
                                AnthropicContentBlock::Text { text } => Some(text.as_str()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n"),
                    }
                })
                .collect();
            let assistant_tc_buf: Vec<Vec<AssistantToolCall<'_>>> = req
                .messages
                .iter()
                .map(|m| {
                    if m.role != "assistant" {
                        return Vec::new();
                    }
                    match &m.content {
                        AnthropicContent::Text(_) => Vec::new(),
                        AnthropicContent::Blocks(blocks) => blocks
                            .iter()
                            .filter_map(|b| match b {
                                AnthropicContentBlock::ToolUse { id, name, input } => {
                                    Some(AssistantToolCall {
                                        id: id.as_str(),
                                        name: name.as_str(),
                                        arguments: input,
                                    })
                                }
                                _ => None,
                            })
                            .collect(),
                    }
                })
                .collect();
            // tool_result content can be string or array-of-text-blocks;
            // pre-flatten to owned strings so the ChatTurn::Tool borrows
            // are stable for the request lifetime.
            let tool_result_buf: Vec<Vec<(String, String)>> = req
                .messages
                .iter()
                .map(|m| {
                    if m.role != "user" {
                        return Vec::new();
                    }
                    match &m.content {
                        AnthropicContent::Text(_) => Vec::new(),
                        AnthropicContent::Blocks(blocks) => blocks
                            .iter()
                            .filter_map(|b| match b {
                                AnthropicContentBlock::ToolResult {
                                    tool_use_id,
                                    content,
                                    is_error,
                                } => {
                                    // Phase 1.6: when is_error:true, prefix
                                    // the content with "[ERROR] " so the
                                    // model recognizes it as a failure and
                                    // can recover gracefully (apologize,
                                    // suggest alternatives, retry with
                                    // different args, etc.). Mirrors how
                                    // Claude interprets is_error blocks.
                                    let body = if *is_error {
                                        format!("[ERROR] {}", content.as_text())
                                    } else {
                                        content.as_text()
                                    };
                                    Some((tool_use_id.clone(), body))
                                }
                                _ => None,
                            })
                            .collect(),
                    }
                })
                .collect();
            // Extra `user` text in a tool-result message (Anthropic allows
            // mixing text + tool_result blocks in one user message). We
            // emit that text as a User turn AFTER all tool_result turns.
            let user_text_buf: Vec<String> = req
                .messages
                .iter()
                .map(|m| {
                    if m.role != "user" {
                        return String::new();
                    }
                    match &m.content {
                        AnthropicContent::Text(s) => s.clone(),
                        AnthropicContent::Blocks(blocks) => blocks
                            .iter()
                            .filter_map(|b| match b {
                                AnthropicContentBlock::Text { text } => Some(text.as_str()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n"),
                    }
                })
                .collect();

            let mut turns: Vec<ChatTurn<'_>> = Vec::with_capacity(req.messages.len() + 4);
            if let Some(ref s) = system_text {
                turns.push(ChatTurn::System(s.as_str()));
            }
            for (i, msg) in req.messages.iter().enumerate() {
                match msg.role.as_str() {
                    "assistant" => {
                        turns.push(ChatTurn::Assistant {
                            text: assistant_text_buf[i].as_str(),
                            tool_calls: assistant_tc_buf[i].as_slice(),
                        });
                    }
                    "user" => {
                        // Emit any tool_result blocks first (they belong
                        // inside the preceding model turn per Gemma 4's
                        // layout, which `render_chat_history` handles by
                        // consuming consecutive Tool turns into the prior
                        // Assistant turn).
                        for (tcid, content) in &tool_result_buf[i] {
                            turns.push(ChatTurn::Tool {
                                tool_call_id: tcid.as_str(),
                                name: None,
                                content: content.as_str(),
                            });
                        }
                        // Then any free-text portion of the user message.
                        // (`user_text_buf` already covers both
                        // `AnthropicContent::Text(s)` — via the synthesized
                        // single-block view — and `AnthropicContent::Blocks`
                        // shapes, so a missing text means the message had
                        // no text content at all.)
                        let utext = user_text_buf[i].as_str();
                        if !utext.is_empty() {
                            turns.push(ChatTurn::User(utext));
                        }
                    }
                    _ => {
                        return Err(anyhow::anyhow!(
                            "anthropic_messages: unsupported role {:?}",
                            msg.role
                        ));
                    }
                }
            }
            lumen_mlx::chat_io::strip_client_meta_wrappers(&mut turns);
            self.backend.chat_from_history(
                &turns,
                req.max_tokens,
                req.temperature,
                req.top_p,
                &ov,
                req.enable_thinking_with_backend_default(self.backend.is_reasoning_first_family()),
                req.session_id.as_deref(),
                &tools_owned,
                &tool_choice,
                // Anthropic /v1/messages has no `response_format`.
                None,
            )?
        } else {
            // Plain path — uses the flat messages built above. Matches
            // the pre-Phase-1.4 behavior bit-for-bit.
            self.backend.chat(
                &messages,
                req.max_tokens,
                req.temperature,
                req.top_p,
                &ov,
                req.enable_thinking_with_backend_default(self.backend.is_reasoning_first_family()),
                req.session_id.as_deref(),
                &tools_owned,
                &tool_choice,
                None,
            )?
        };

        let prompt_tokens = self.backend.count_chat_prompt_tokens(
            &messages,
            req.enable_thinking_with_backend_default(self.backend.is_reasoning_first_family()),
        );
        // Bug A: resolve abbreviated tool names by unique suffix match.
        remap_tool_call_names(&mut parsed.tool_calls, &tools_owned);
        // Stop sequences: truncate the visible text at the earliest match and
        // record which sequence fired (Anthropic reports it in `stop_sequence`
        // with stop_reason="stop_sequence"). No-op when no stops were requested.
        let mut matched_stop: Option<String> = None;
        if !ov.stop.is_empty() {
            let mut earliest: Option<usize> = None;
            for s in &ov.stop {
                if let Some(i) = parsed.visible.find(s.as_str()) {
                    if earliest.map_or(true, |e| i < e) {
                        earliest = Some(i);
                        matched_stop = Some(s.clone());
                    }
                }
            }
            if let Some(i) = earliest {
                parsed.visible.truncate(i);
            }
        }
        let output_tokens =
            completion_tokens_with_tools(&self.backend, &parsed.visible, &parsed.tool_calls);

        // Assemble content[] per Anthropic spec: any leading visible text
        // as a `text` block, then one `tool_use` block per parsed tool call.
        // stop_reason="tool_use" when tool calls present, else "end_turn".
        let has_tool_calls = !parsed.tool_calls.is_empty();
        let mut content: Vec<AnthropicResponseBlock> = Vec::new();
        let visible = parsed.visible.trim();
        if !visible.is_empty() {
            content.push(AnthropicResponseBlock::Text {
                text: visible.to_string(),
            });
        }
        if has_tool_calls {
            for call in &parsed.tool_calls {
                content.push(AnthropicResponseBlock::ToolUse {
                    id: format!("toolu_{}", gen_id()),
                    name: call.name.clone(),
                    input: call.arguments.clone(),
                });
            }
        }
        // If the model produced nothing visible AND no tool calls, fall
        // back to an empty text block so the response still conforms to
        // the spec (content[] cannot be empty).
        if content.is_empty() {
            content.push(AnthropicResponseBlock::Text {
                text: String::new(),
            });
        }
        // A matched stop sequence (only meaningful without tool calls) maps to
        // Anthropic's stop_reason="stop_sequence" + the matched string.
        let stop_seq_hit = if has_tool_calls { None } else { matched_stop };
        let stop_reason = if has_tool_calls {
            "tool_use"
        } else if stop_seq_hit.is_some() {
            "stop_sequence"
        } else {
            "end_turn"
        };

        Ok(AnthropicResponse {
            id: format!("msg_{}", gen_id()),
            r#type: "message".into(),
            role: "assistant".into(),
            model: req.model.clone(),
            content,
            stop_reason: stop_reason.into(),
            stop_sequence: stop_seq_hit,
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
        let ov = req.sampling_overrides();
        let mut messages: Vec<(String, String)> = req
            .messages
            .iter()
            .map(|m| (m.role.clone(), m.content.clone()))
            .collect();
        lumen_mlx::chat_io::strip_client_meta_wrappers_flat(&mut messages);

        let tool_names: Vec<&str> = req
            .tools
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .map(|t| match t {
                crate::types::Tool::Function { function } => function.name.as_str(),
            })
            .collect();
        let inferred_mode = lumen_mlx::chat_io::classify_request_mode(&messages, &tool_names);
        eprintln!(
            "[chat-io] inferred_mode={} (tools={}, system_len={})",
            inferred_mode.as_str(),
            tool_names.len(),
            messages
                .iter()
                .find(|(r, _)| r.eq_ignore_ascii_case("system"))
                .map(|(_, c)| c.len())
                .unwrap_or(0)
        );

        let prompt_tokens = self.backend.count_chat_prompt_tokens(
            &messages,
            req.enable_thinking_with_backend_default(self.backend.is_reasoning_first_family()),
        );

        let needs_structured = needs_structured_history(&req.messages);
        let prompt_bytes: usize = messages.iter().map(|(_, c)| c.len()).sum();
        eprintln!(
            "[chat-stream] msgs={} prompt_bytes={} prompt_tokens={} max_tokens={} thinking={} structured={}",
            messages.len(),
            prompt_bytes,
            prompt_tokens,
            req.max_tokens,
            req.enable_thinking_with_backend_default(self.backend.is_reasoning_first_family()),
            needs_structured,
        );

        // chunked prefill
        // is back with sliding-window-aware mask construction
        // (`make_attention_mask_for_layer_chunked`). Cap protects against
        // runaway KV growth from misbehaving clients while letting normal
        // long-context (≤32K) requests flow.
        //
        // Prompt-size REJECTION cap (distinct from the per-step *chunk size*).
        // Chunked prefill (`forward_chunked`) bounds activation memory, so this
        // is a safety rail against runaway prompts, NOT a hard chunking limit —
        // raise it to serve longer contexts. Precedence:
        //   `LUMEN_MAX_PROMPT_TOKENS` (clear name) → `LUMEN_PREFILL_CHUNK`
        //   (legacy; the desktop CONTEXT card still emits it) → 32K default.
        // NOTE: this is a *reject* cap, not a sliding context window. Dropping
        // old turns to fit (true context-shift) needs token access inside the
        // backend (the engine only has a token *count* here) — tracked
        // separately; for now an over-cap prompt is rejected with guidance.
        let prefill_token_cap: u32 = std::env::var("LUMEN_MAX_PROMPT_TOKENS")
            .or_else(|_| std::env::var("LUMEN_PREFILL_CHUNK"))
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(32_768);
        if prompt_tokens > prefill_token_cap {
            let msg = format!(
                "prompt too large: {prompt_tokens} tokens > server cap {prefill_token_cap}. \
                 Raise LUMEN_MAX_PROMPT_TOKENS (chunked prefill handles the memory — \
                 check the Tuning memory calculator first) or trim the history."
            );
            eprintln!("[chat-stream] REJECTED ({msg})");
            let _ = token_tx.try_send(StreamEvent::Error(msg));
            return;
        }

        let tools_owned = openai_tools_to_defs(req.tools.as_deref());
        let tool_choice =
            resolve_openai_tool_choice(req.tool_choice.as_ref(), !tools_owned.is_empty());
        // OpenAI `response_format` → JSON-schema constrained decoding
        // (Gemma 4 only). `None` when absent / `text` → exact existing path.
        let response_schema = req.response_json_schema();

        // Phase 1.5: when prior assistant tool_calls or role:"tool" are
        // in the request, dispatch through chat_streaming_from_history
        // so the structured renderer handles them. Mirrors the
        // chat_completion non-stream path. Owning buffers
        // (`arg_values`, `assistant_tc_buf`) keep ChatTurn borrows
        // stable until the call returns.
        // Phase 1.6c: tracks tool_calls the backend's parser announced
        // mid-decode via `BackendStreamEvent::ToolCallStart`. Lives
        // outside both branches so the post-decode emission loop can
        // skip Start chunks already on the wire.
        // Kept (always empty) so the reconciliation loop's `early_tc.get(i)`
        // falls through to a fresh, sequential per-call emission. Early
        // per-call Starts are intentionally NOT emitted (see closures below).
        let early_tc: Vec<(String, u32, String)> = Vec::new();
        // Stop-sequence filtering shared by both stream closures. The matcher
        // holds back text that might be the prefix of a stop string, emits only
        // safe-to-stream text, and trims at a full match. `stopped_by_seq`
        // distinguishes the early decode-loop break below from a real error.
        let stop_matcher =
            std::cell::RefCell::new(lumen_core::stop::StopMatcher::new(ov.stop.clone()));
        let stopped_by_seq = std::cell::Cell::new(false);
        // response_format streaming: when a JSON schema constrains the output,
        // end the stream at the first complete JSON value (llguidance shapes
        // the value but doesn't force EOS at its close, so the model would
        // otherwise free-run trailing prose). `None` for non-response_format
        // requests, leaving the normal stop-matcher path untouched.
        let json_stop = response_schema
            .as_ref()
            .map(|_| std::cell::RefCell::new(JsonValueStop::default()));
        // Wall-clock around the streaming generation for the `/v1/loads`
        // last tok/s gauge (recorded at the `Done` terminal below).
        let gen_started = Instant::now();
        let result = if needs_structured {
            let arg_values: Vec<serde_json::Value> = req
                .messages
                .iter()
                .flat_map(|m| {
                    m.tool_calls
                        .iter()
                        .flat_map(|calls| calls.iter())
                        .map(|c| {
                            serde_json::from_str(&c.function.arguments)
                                .unwrap_or_else(|_| serde_json::json!({}))
                        })
                        .collect::<Vec<_>>()
                })
                .collect();
            let assistant_tc_buf: Vec<Vec<AssistantToolCall<'_>>> = {
                let mut next_arg = 0;
                req.messages
                    .iter()
                    .map(|m| {
                        m.tool_calls
                            .as_ref()
                            .map(|calls| {
                                calls
                                    .iter()
                                    .map(|c| {
                                        let av = &arg_values[next_arg];
                                        next_arg += 1;
                                        AssistantToolCall {
                                            id: c.id.as_str(),
                                            name: c.function.name.as_str(),
                                            arguments: av,
                                        }
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_default()
                    })
                    .collect()
            };
            let mut turns: Vec<ChatTurn<'_>> = req
                .messages
                .iter()
                .enumerate()
                .map(|(i, m)| match m.role.as_str() {
                    "system" | "System" | "SYSTEM" => ChatTurn::System(m.content.as_str()),
                    "user" | "User" | "USER" => ChatTurn::User(m.content.as_str()),
                    "tool" => ChatTurn::Tool {
                        tool_call_id: m.tool_call_id.as_deref().unwrap_or(""),
                        name: m.name.as_deref(),
                        content: m.content.as_str(),
                    },
                    _ => ChatTurn::Assistant {
                        text: m.content.as_str(),
                        tool_calls: assistant_tc_buf.get(i).map(Vec::as_slice).unwrap_or(&[]),
                    },
                })
                .collect();
            lumen_mlx::chat_io::strip_client_meta_wrappers(&mut turns);
            self.backend.chat_streaming_from_history(
                &turns,
                req.max_tokens,
                req.temperature,
                req.top_p,
                &ov,
                req.enable_thinking_with_backend_default(self.backend.is_reasoning_first_family()),
                req.session_id.as_deref(),
                &tools_owned,
                &tool_choice,
                response_schema.as_ref(),
                |ev: BackendStreamEvent<'_>| -> Result<()> {
                    match ev {
                        BackendStreamEvent::Text(t) => {
                            if let Some(js) = json_stop.as_ref() {
                                // response_format: stream up to the first
                                // complete JSON value, then end the decode loop
                                // (reuses the stopped_by_seq early-break path).
                                let (emit, stopped) = js.borrow_mut().push(t);
                                if !emit.is_empty() {
                                    let _ = token_tx.try_send(StreamEvent::Delta(emit));
                                }
                                if stopped {
                                    stopped_by_seq.set(true);
                                    return Err(anyhow::anyhow!("__lumen_stop_sequence__"));
                                }
                            } else {
                                let mut sm = stop_matcher.borrow_mut();
                                if sm.is_inert() {
                                    let _ = token_tx.try_send(StreamEvent::Delta(t.to_string()));
                                } else {
                                    let step = sm.push(t);
                                    if !step.emit.is_empty() {
                                        let _ = token_tx.try_send(StreamEvent::Delta(step.emit));
                                    }
                                    if step.stopped {
                                        stopped_by_seq.set(true);
                                        drop(sm);
                                        // Break the backend decode loop early. The
                                        // `stopped_by_seq` flag distinguishes this
                                        // from a real error in the match below.
                                        return Err(anyhow::anyhow!("__lumen_stop_sequence__"));
                                    }
                                }
                            }
                        }
                        BackendStreamEvent::Reasoning(t) => {
                            let _ = token_tx.try_send(StreamEvent::ReasoningDelta(t.to_string()));
                        }
                        BackendStreamEvent::ToolCallStart { name } => {
                            // Early per-call Start suppressed. Args are not known
                            // until a call closes, so emitting Start0,Start1,Start2
                            // up front (then Args0,Args1,Args2 in a batch) made
                            // clients fold every later parallel call's arguments
                            // onto index 0 — only the first tool call kept its args.
                            // The reconciliation loop below now emits each call as a
                            // sequential Start_i → Args_i → Stop_i unit.
                            let _ = name;
                        }
                    }
                    Ok(())
                },
            )
        } else {
            self.backend.chat_streaming(
                &messages,
                req.max_tokens,
                req.temperature,
                req.top_p,
                &ov,
                req.enable_thinking_with_backend_default(self.backend.is_reasoning_first_family()),
                req.session_id.as_deref(),
                &tools_owned,
                &tool_choice,
                response_schema.as_ref(),
                |ev: BackendStreamEvent<'_>| -> Result<()> {
                    match ev {
                        BackendStreamEvent::Text(t) => {
                            if let Some(js) = json_stop.as_ref() {
                                // response_format: stream up to the first
                                // complete JSON value, then end the decode loop
                                // (reuses the stopped_by_seq early-break path).
                                let (emit, stopped) = js.borrow_mut().push(t);
                                if !emit.is_empty() {
                                    let _ = token_tx.try_send(StreamEvent::Delta(emit));
                                }
                                if stopped {
                                    stopped_by_seq.set(true);
                                    return Err(anyhow::anyhow!("__lumen_stop_sequence__"));
                                }
                            } else {
                                let mut sm = stop_matcher.borrow_mut();
                                if sm.is_inert() {
                                    let _ = token_tx.try_send(StreamEvent::Delta(t.to_string()));
                                } else {
                                    let step = sm.push(t);
                                    if !step.emit.is_empty() {
                                        let _ = token_tx.try_send(StreamEvent::Delta(step.emit));
                                    }
                                    if step.stopped {
                                        stopped_by_seq.set(true);
                                        drop(sm);
                                        // Break the backend decode loop early. The
                                        // `stopped_by_seq` flag distinguishes this
                                        // from a real error in the match below.
                                        return Err(anyhow::anyhow!("__lumen_stop_sequence__"));
                                    }
                                }
                            }
                        }
                        BackendStreamEvent::Reasoning(t) => {
                            let _ = token_tx.try_send(StreamEvent::ReasoningDelta(t.to_string()));
                        }
                        BackendStreamEvent::ToolCallStart { name } => {
                            // Early per-call Start suppressed. Args are not known
                            // until a call closes, so emitting Start0,Start1,Start2
                            // up front (then Args0,Args1,Args2 in a batch) made
                            // clients fold every later parallel call's arguments
                            // onto index 0 — only the first tool call kept its args.
                            // The reconciliation loop below now emits each call as a
                            // sequential Start_i → Args_i → Stop_i unit.
                            let _ = name;
                        }
                    }
                    Ok(())
                },
            )
        };

        match result {
            Ok(mut parsed) => {
                // Release any partial tail the stop matcher was holding back
                // (a fragment that could have been the prefix of a stop string
                // but never completed) so it still streams to the client.
                let tail = stop_matcher.borrow_mut().flush();
                if !tail.is_empty() {
                    let _ = token_tx.try_send(StreamEvent::Delta(tail));
                }
                // Bug A: resolve abbreviated tool names (e.g. the model dropped
                // an MCP namespace) by unique suffix match before emitting.
                remap_tool_call_names(&mut parsed.tool_calls, &tools_owned);
                // Phase 1.5: emit each parsed tool_call as the
                // start / arguments / stop trio so the SSE layer can
                // synthesize OpenAI `delta.tool_calls` chunks. Phase
                // 1.6c: if the backend already announced this call
                // via `BackendStreamEvent::ToolCallStart`, skip the
                // redundant Start (we just emit ArgumentsDelta + Stop
                // with the matching index).
                //
                // NOTE: `early_tc` lives inside the `if/else` branch
                // above so it's not visible here — we re-derive
                // index/id from the call order. The early Start chunks
                // already went out on the wire; this loop only owns
                // ArgumentsDelta + Stop.
                let has_tool_calls = !parsed.tool_calls.is_empty();
                for (i, call) in parsed.tool_calls.iter().enumerate() {
                    // Phase 1.6c: if the backend already announced
                    // this call mid-decode (early_tc[i] matches by
                    // name) we reuse its (index, id). Otherwise emit
                    // a fresh Start now — covers legacy backends that
                    // don't fire `BackendStreamEvent::ToolCallStart`.
                    let (index, already_started) = match early_tc.get(i) {
                        Some((name, idx, _)) if name == &call.name => (*idx, true),
                        _ => (i as u32, false),
                    };
                    if !already_started {
                        let id = format!("call_{}", gen_id());
                        let _ = token_tx.try_send(StreamEvent::ToolCallStart {
                            index,
                            id,
                            name: call.name.clone(),
                        });
                    }
                    let args =
                        serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".into());
                    let _ = token_tx.try_send(StreamEvent::ToolCallArgumentsDelta {
                        index,
                        partial_json: args,
                    });
                    let _ = token_tx.try_send(StreamEvent::ToolCallStop { index });
                }
                let completion_tokens = completion_tokens_with_tools(
                    &self.backend,
                    &parsed.visible,
                    &parsed.tool_calls,
                );
                let finish_reason = if has_tool_calls {
                    FinishReason::ToolCalls
                } else {
                    FinishReason::Stop
                };
                // Observability: bump lifetime counters for `GET /v1/loads`.
                let elapsed_s = gen_started.elapsed().as_secs_f64();
                let tok_per_sec = if elapsed_s > 0.0 {
                    completion_tokens as f64 / elapsed_s
                } else {
                    0.0
                };
                self.load_stats
                    .record(prompt_tokens as u64, completion_tokens as u64, tok_per_sec);
                let _ = token_tx.try_send(StreamEvent::Done {
                    prompt_tokens,
                    completion_tokens,
                    finish_reason,
                });
            }
            Err(_) if stopped_by_seq.get() => {
                // Stopped by a client stop sequence. The visible text up to the
                // stop was already streamed (already trimmed by the matcher),
                // and any held tail beyond the stop is intentionally dropped —
                // do NOT flush here. No tool calls are possible mid-text, so we
                // finish exactly like the Ok arm's no-tool-call terminal chunk:
                // a `Done` with `FinishReason::Stop`. `completion_tokens` is a
                // best-effort 0 since the parsed visible text is unavailable on
                // this early-break path.
                // Observability: still count the request (gen tokens unknown →
                // 0, tok/s left unchanged via the 0.0 guard in `record`).
                self.load_stats.record(prompt_tokens as u64, 0, 0.0);
                let _ = token_tx.try_send(StreamEvent::Done {
                    prompt_tokens,
                    completion_tokens: 0,
                    finish_reason: FinishReason::Stop,
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
        let needs_structured = anthropic_needs_structured_history(&req.messages);
        let ov = req.sampling_overrides();

        let system_text: Option<String> = req.system.as_ref().map(|sys| match sys {
            AnthropicSystem::Text(s) => s.clone(),
            AnthropicSystem::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| b.text.clone())
                .collect::<Vec<_>>()
                .join("\n"),
        });

        let mut messages: Vec<(String, String)> = Vec::new();
        if let Some(ref s) = system_text {
            if !s.is_empty() {
                messages.push(("system".into(), s.clone()));
            }
        }
        for msg in &req.messages {
            messages.push((msg.role.clone(), msg.content.as_text()));
        }
        lumen_mlx::chat_io::strip_client_meta_wrappers_flat(&mut messages);

        let prompt_tokens = self.backend.count_chat_prompt_tokens(
            &messages,
            req.enable_thinking_with_backend_default(self.backend.is_reasoning_first_family()),
        );

        let tools_owned = anthropic_tools_to_defs(req.tools.as_deref());
        let tool_choice =
            resolve_anthropic_tool_choice(req.tool_choice.as_ref(), !tools_owned.is_empty());

        // Phase 1.5: structured-history dispatch for Anthropic streaming.
        // Mirrors `anthropic_messages` non-stream: build owning buffers
        // (`assistant_text_buf`, `assistant_tc_buf`, `tool_result_buf`,
        // `user_text_buf`) so ChatTurn borrows stay alive across the
        // backend call.
        //
        // Phase 1.6c: tracks tool_calls the backend announced mid-decode
        // so the post-decode loop skips redundant Start chunks.
        // Kept (always empty) so the reconciliation loop's `early_tc.get(i)`
        // falls through to a fresh, sequential per-call emission. Early
        // per-call Starts are intentionally NOT emitted (see closures below).
        let early_tc: Vec<(String, u32, String)> = Vec::new();
        let result = if needs_structured {
            let assistant_text_buf: Vec<String> = req
                .messages
                .iter()
                .map(|m| {
                    if m.role != "assistant" {
                        return String::new();
                    }
                    match &m.content {
                        AnthropicContent::Text(s) => s.clone(),
                        AnthropicContent::Blocks(blocks) => blocks
                            .iter()
                            .filter_map(|b| match b {
                                AnthropicContentBlock::Text { text } => Some(text.as_str()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n"),
                    }
                })
                .collect();
            let assistant_tc_buf: Vec<Vec<AssistantToolCall<'_>>> = req
                .messages
                .iter()
                .map(|m| {
                    if m.role != "assistant" {
                        return Vec::new();
                    }
                    match &m.content {
                        AnthropicContent::Text(_) => Vec::new(),
                        AnthropicContent::Blocks(blocks) => blocks
                            .iter()
                            .filter_map(|b| match b {
                                AnthropicContentBlock::ToolUse { id, name, input } => {
                                    Some(AssistantToolCall {
                                        id: id.as_str(),
                                        name: name.as_str(),
                                        arguments: input,
                                    })
                                }
                                _ => None,
                            })
                            .collect(),
                    }
                })
                .collect();
            let tool_result_buf: Vec<Vec<(String, String)>> = req
                .messages
                .iter()
                .map(|m| {
                    if m.role != "user" {
                        return Vec::new();
                    }
                    match &m.content {
                        AnthropicContent::Text(_) => Vec::new(),
                        AnthropicContent::Blocks(blocks) => blocks
                            .iter()
                            .filter_map(|b| match b {
                                AnthropicContentBlock::ToolResult {
                                    tool_use_id,
                                    content,
                                    is_error,
                                } => {
                                    let body = if *is_error {
                                        format!("[ERROR] {}", content.as_text())
                                    } else {
                                        content.as_text()
                                    };
                                    Some((tool_use_id.clone(), body))
                                }
                                _ => None,
                            })
                            .collect(),
                    }
                })
                .collect();
            let user_text_buf: Vec<String> = req
                .messages
                .iter()
                .map(|m| {
                    if m.role != "user" {
                        return String::new();
                    }
                    match &m.content {
                        AnthropicContent::Text(s) => s.clone(),
                        AnthropicContent::Blocks(blocks) => blocks
                            .iter()
                            .filter_map(|b| match b {
                                AnthropicContentBlock::Text { text } => Some(text.as_str()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n"),
                    }
                })
                .collect();

            let mut turns: Vec<ChatTurn<'_>> = Vec::with_capacity(req.messages.len() + 4);
            if let Some(ref s) = system_text {
                if !s.is_empty() {
                    turns.push(ChatTurn::System(s.as_str()));
                }
            }
            for (i, msg) in req.messages.iter().enumerate() {
                match msg.role.as_str() {
                    "assistant" => {
                        turns.push(ChatTurn::Assistant {
                            text: assistant_text_buf[i].as_str(),
                            tool_calls: assistant_tc_buf[i].as_slice(),
                        });
                    }
                    "user" => {
                        for (tid, body) in &tool_result_buf[i] {
                            turns.push(ChatTurn::Tool {
                                tool_call_id: tid.as_str(),
                                name: None,
                                content: body.as_str(),
                            });
                        }
                        if !user_text_buf[i].is_empty() {
                            turns.push(ChatTurn::User(user_text_buf[i].as_str()));
                        }
                    }
                    _ => {}
                }
            }
            lumen_mlx::chat_io::strip_client_meta_wrappers(&mut turns);
            self.backend.chat_streaming_from_history(
                &turns,
                req.max_tokens,
                req.temperature,
                req.top_p,
                &ov,
                req.enable_thinking_with_backend_default(self.backend.is_reasoning_first_family()),
                req.session_id.as_deref(),
                &tools_owned,
                &tool_choice,
                // Anthropic Messages API has no `response_format` field.
                None,
                |ev: BackendStreamEvent<'_>| -> Result<()> {
                    match ev {
                        BackendStreamEvent::Text(t) => {
                            let _ = token_tx.try_send(StreamEvent::Delta(t.to_string()));
                        }
                        BackendStreamEvent::Reasoning(t) => {
                            let _ = token_tx.try_send(StreamEvent::ReasoningDelta(t.to_string()));
                        }
                        BackendStreamEvent::ToolCallStart { name } => {
                            // Early per-call Start suppressed — see the call-1
                            // closure above and the reconciliation loop below.
                            // Sequential Start_i → Args_i → Stop_i per call is the
                            // standard order; batched up-front Starts dropped args
                            // for parallel calls after index 0.
                            let _ = name;
                        }
                    }
                    Ok(())
                },
            )
        } else {
            self.backend.chat_streaming(
                &messages,
                req.max_tokens,
                req.temperature,
                req.top_p,
                &ov,
                req.enable_thinking_with_backend_default(self.backend.is_reasoning_first_family()),
                req.session_id.as_deref(),
                &tools_owned,
                &tool_choice,
                // Anthropic Messages API has no `response_format` field.
                None,
                |ev: BackendStreamEvent<'_>| -> Result<()> {
                    match ev {
                        BackendStreamEvent::Text(t) => {
                            let _ = token_tx.try_send(StreamEvent::Delta(t.to_string()));
                        }
                        BackendStreamEvent::Reasoning(t) => {
                            let _ = token_tx.try_send(StreamEvent::ReasoningDelta(t.to_string()));
                        }
                        BackendStreamEvent::ToolCallStart { name } => {
                            // Early per-call Start suppressed — see the call-1
                            // closure above and the reconciliation loop below.
                            // Sequential Start_i → Args_i → Stop_i per call is the
                            // standard order; batched up-front Starts dropped args
                            // for parallel calls after index 0.
                            let _ = name;
                        }
                    }
                    Ok(())
                },
            )
        };

        match result {
            Ok(mut parsed) => {
                // Bug A: resolve abbreviated tool names (e.g. the model dropped
                // an MCP namespace) by unique suffix match before emitting.
                remap_tool_call_names(&mut parsed.tool_calls, &tools_owned);
                let has_tool_calls = !parsed.tool_calls.is_empty();
                for (i, call) in parsed.tool_calls.iter().enumerate() {
                    let (index, already_started) = match early_tc.get(i) {
                        Some((name, idx, _)) if name == &call.name => (*idx, true),
                        _ => (i as u32, false),
                    };
                    if !already_started {
                        let id = format!("toolu_{}", gen_id());
                        let _ = token_tx.try_send(StreamEvent::ToolCallStart {
                            index,
                            id,
                            name: call.name.clone(),
                        });
                    }
                    let args =
                        serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".into());
                    let _ = token_tx.try_send(StreamEvent::ToolCallArgumentsDelta {
                        index,
                        partial_json: args,
                    });
                    let _ = token_tx.try_send(StreamEvent::ToolCallStop { index });
                }
                let completion_tokens = completion_tokens_with_tools(
                    &self.backend,
                    &parsed.visible,
                    &parsed.tool_calls,
                );
                let finish_reason = if has_tool_calls {
                    FinishReason::ToolCalls
                } else {
                    FinishReason::Stop
                };
                let _ = token_tx.try_send(StreamEvent::Done {
                    prompt_tokens,
                    completion_tokens,
                    finish_reason,
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

/// Whether reasoning/thinking should also be mirrored into `content` wrapped
/// in a `<think>…</think>` envelope (legacy dual emission). Default `false`,
/// matching Ollama's OpenAI-compat layer (thinking lives only in the
/// `reasoning` field; `content` is the visible answer alone). Opt back in with
/// `LUMEN_REASONING_IN_CONTENT=1` for text-tag-only clients.
pub(crate) fn reasoning_in_content() -> bool {
    std::env::var("LUMEN_REASONING_IN_CONTENT")
        .map(|v| {
            let v = v.trim();
            v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("on")
        })
        .unwrap_or(false)
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

/// Completion tokens that account for tool-call body output. The visible
/// channel is the assistant's natural-language text; tool-call markup
/// (id + name + JSON-serialized arguments) is emitted by the model but
/// the response parser strips it from `visible`, so a tool-only turn
/// would otherwise report `0` — diverging from real OpenAI / Anthropic
/// APIs which always count the serialized tool_use body. We approximate
/// the missing cost by re-tokenizing `name + JSON(arguments)` per call.
/// Byte index just past the first complete balanced JSON value (object or
/// array) in `s`, or `None` if no complete value is present. String- and
/// escape-aware so braces/brackets inside string literals don't affect the
/// depth count. Used to trim trailing prose from `response_format` output
/// (the JSON-schema grammar shapes the value but doesn't force EOS at its
/// close). The returned index lands on a `}`/`]` (ASCII) so it is always a
/// valid UTF-8 char boundary for `String::truncate`.
/// Streaming analogue of [`first_json_value_end`]: a stateful tracker fed the
/// decoded text chunks of a `response_format` stream. It emits text up to and
/// including the first complete balanced JSON value, then signals `stopped` so
/// the decode loop ends — trimming the trailing prose that llguidance's
/// JSON-schema grammar permits after the closing `}` (it never forces EOS at
/// completion). String- and escape-aware so braces inside strings don't count.
#[derive(Default)]
struct JsonValueStop {
    started: bool,
    depth: i32,
    in_str: bool,
    escaped: bool,
    done: bool,
}

impl JsonValueStop {
    /// Feed one decoded text chunk. Returns `(emit, stopped)` — `emit` is the
    /// portion of `t` to stream (truncated at the JSON close on completion),
    /// `stopped` is true once the first complete value has closed.
    fn push(&mut self, t: &str) -> (String, bool) {
        if self.done {
            return (String::new(), true);
        }
        for (i, &c) in t.as_bytes().iter().enumerate() {
            if self.in_str {
                if self.escaped {
                    self.escaped = false;
                } else if c == b'\\' {
                    self.escaped = true;
                } else if c == b'"' {
                    self.in_str = false;
                }
                continue;
            }
            match c {
                b'"' => self.in_str = true,
                b'{' | b'[' => {
                    self.started = true;
                    self.depth += 1;
                }
                b'}' | b']' => {
                    self.depth -= 1;
                    if self.started && self.depth == 0 {
                        self.done = true;
                        // `i` lands on a `}`/`]` (ASCII) so `i + 1` is a char
                        // boundary.
                        return (t[..i + 1].to_string(), true);
                    }
                }
                _ => {}
            }
        }
        (t.to_string(), false)
    }
}

fn first_json_value_end(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let start = bytes.iter().position(|&b| b == b'{' || b == b'[')?;
    let mut depth = 0i32;
    let mut in_str = false;
    let mut escaped = false;
    for (i, &c) in bytes.iter().enumerate().skip(start) {
        if in_str {
            if escaped {
                escaped = false;
            } else if c == b'\\' {
                escaped = true;
            } else if c == b'"' {
                in_str = false;
            }
            continue;
        }
        match c {
            b'"' => in_str = true,
            b'{' | b'[' => depth += 1,
            b'}' | b']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
    }
    None
}

fn completion_tokens_with_tools(
    backend: &ModelBackend,
    visible: &str,
    tool_calls: &[ParsedToolCall],
) -> u32 {
    let mut total = count_tokens(backend, visible);
    for call in tool_calls {
        let args = serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".into());
        total = total.saturating_add(count_tokens(backend, &call.name));
        total = total.saturating_add(count_tokens(backend, &args));
    }
    total
}

fn gen_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    // Append a process-local monotonic counter so concurrent requests within
    // the same second don't collide (unix_timestamp has 1s resolution).
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{:x}-{:x}", unix_timestamp(), n)
}

// ── Tool-calling bridges ─────────────────────────────────────────────────
//
// Convert the OpenAI / Anthropic request types into the backend-facing
// `ToolDef<'_>` shape, and the backend's `ParsedToolCall` output back into
// each API's wire format. Lifetimes flow from the request that owns the
// underlying strings, so the returned `Vec<ToolDef>` borrows from `req.tools`.

/// Decide whether the request needs the structured-history chat path. We
/// route there only when at least one message carries tool metadata that
/// the flat `(role, content)` shape can't preserve (assistant.tool_calls
/// on a prior turn, or a `role:"tool"` response). Vanilla single-turn
/// requests with `tools[]` defined but no history fall through to the
/// plain path — which still renders tool *definitions* via Phase 1.3's
/// `render_to_ids_with_tools`.
fn needs_structured_history(messages: &[ChatMessage]) -> bool {
    messages.iter().any(|m| {
        m.role == "tool"
            || m.tool_calls
                .as_ref()
                .map(|c| !c.is_empty())
                .unwrap_or(false)
    })
}

/// Anthropic variant of `needs_structured_history`. We scan content blocks
/// for `tool_use` (assistant emitted) or `tool_result` (user replying)
/// since Anthropic encodes tool turns inside the message body rather than
/// via a separate `role:"tool"`.
fn anthropic_needs_structured_history(messages: &[AnthropicMessage]) -> bool {
    messages.iter().any(|m| match &m.content {
        AnthropicContent::Text(_) => false,
        AnthropicContent::Blocks(blocks) => blocks.iter().any(|b| {
            matches!(
                b,
                AnthropicContentBlock::ToolUse { .. } | AnthropicContentBlock::ToolResult { .. }
            )
        }),
    })
}

/// Phase 1.6: resolve the OpenAI `tool_choice` into our canonical
/// `ResolvedToolChoice`. Maps `"auto"` / `"required"` / `"none"` and
/// `{type:"function", function:{name}}`. `Required` collapses to
/// `Auto` when no tools are defined (no point in forcing a tool call
/// with nothing to choose from).
fn resolve_openai_tool_choice<'a>(
    tool_choice: Option<&'a ToolChoice>,
    has_tools: bool,
) -> ResolvedToolChoice<'a> {
    let resolved = match tool_choice {
        None => ResolvedToolChoice::Auto,
        Some(ToolChoice::Mode(m)) => match m.as_str() {
            "none" => ResolvedToolChoice::None,
            "required" if has_tools => ResolvedToolChoice::Required,
            _ => ResolvedToolChoice::Auto,
        },
        Some(ToolChoice::Named { function, .. }) if has_tools => {
            ResolvedToolChoice::Tool(function.name.as_str())
        }
        Some(ToolChoice::Named { .. }) => ResolvedToolChoice::Auto,
    };
    auto_to_required_when_env_set(resolved, has_tools)
}

/// When `LUMEN_TOOL_CHOICE_AUTO_AS_REQUIRED` is truthy AND the request
/// supplies tools, upgrade `Auto` → `Required` so the chat-template
/// prefill kicks in (Phase 2 grammar then sees the prefilled
/// `<|tool_call>` and the model emits `call:NAME{…}` natively from
/// training). Used by deployments where the client (e.g. Ayla) emits
/// `tool_choice="auto"` but the model has weak self-emission of
/// `<|tool_call>` (id 48) under uniform-4bit quants
/// (`mlx-community/gemma-4-26b-a4b-it-4bit` etc.). Tools-bearing
/// requests in those deployments effectively always want a tool call —
/// the "final answer" tool's `summary` field carries the user-visible
/// text.
///
/// Conservative default OFF preserves OpenAI-spec semantics for mixed
/// chat/agent clients (Claude Code, Cursor) that depend on `auto`
/// meaning "model decides".
///
/// No-op when the request explicitly chose `None` / `Required` / `Tool`
/// — only the implicit `Auto` is touched.
fn auto_to_required_when_env_set<'a>(
    choice: ResolvedToolChoice<'a>,
    has_tools: bool,
) -> ResolvedToolChoice<'a> {
    if !has_tools || !matches!(choice, ResolvedToolChoice::Auto) {
        return choice;
    }
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let enabled = *CACHED.get_or_init(|| {
        std::env::var("LUMEN_TOOL_CHOICE_AUTO_AS_REQUIRED")
            .map(|v| !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(false)
    });
    if enabled {
        ResolvedToolChoice::Required
    } else {
        choice
    }
}

/// Phase 1.6: Anthropic equivalent — `Auto` / `Any` (≈ OpenAI's
/// `required`) / `Tool{name}`. Anthropic has no explicit `None`;
/// callers omit `tools[]` instead.
fn resolve_anthropic_tool_choice<'a>(
    tool_choice: Option<&'a AnthropicToolChoice>,
    has_tools: bool,
) -> ResolvedToolChoice<'a> {
    let resolved = match tool_choice {
        None => ResolvedToolChoice::Auto,
        Some(AnthropicToolChoice::Auto) => ResolvedToolChoice::Auto,
        Some(AnthropicToolChoice::Any) if has_tools => ResolvedToolChoice::Required,
        Some(AnthropicToolChoice::Any) => ResolvedToolChoice::Auto,
        Some(AnthropicToolChoice::Tool { name }) if has_tools => {
            ResolvedToolChoice::Tool(name.as_str())
        }
        Some(AnthropicToolChoice::Tool { .. }) => ResolvedToolChoice::Auto,
    };
    auto_to_required_when_env_set(resolved, has_tools)
}

fn openai_tools_to_defs(tools: Option<&[Tool]>) -> Vec<ToolDef<'_>> {
    let Some(tools) = tools else {
        return Vec::new();
    };
    tools
        .iter()
        .map(|t| {
            let Tool::Function { function } = t;
            ToolDef {
                name: function.name.as_str(),
                description: function.description.as_deref(),
                parameters: function.parameters.as_ref(),
                response: None,
            }
        })
        .collect()
}

fn anthropic_tools_to_defs(tools: Option<&[AnthropicTool]>) -> Vec<ToolDef<'_>> {
    let Some(tools) = tools else {
        return Vec::new();
    };
    tools
        .iter()
        .map(|t| ToolDef {
            name: t.name.as_str(),
            description: t.description.as_deref(),
            parameters: Some(&t.input_schema),
            response: None,
        })
        .collect()
}

/// Convert backend-parsed tool calls into OpenAI-shaped `ToolCall`s. The
/// `id` is generated server-side; clients echo it on the next turn as
/// `tool_call_id`. `function.arguments` is JSON-encoded into a string per
/// OpenAI spec (the client `JSON.parse`'s it themselves).
/// Bug A mitigation: resolve a tool name the model abbreviated.
///
/// Weak/quantized models sometimes emit a tool's *documented* short name
/// (`ctx_read`) instead of its *callable* namespaced name
/// (`mcp__lean_ctx_ctx_read`) — the MCP `mcp__<server>_<tool>` prefix appears
/// in the schema but the prose docs use the short form, and the model follows
/// the prose. When an emitted name matches NO tool exactly but UNIQUELY matches
/// one tool by suffix at a separator boundary (the char before the suffix is
/// non-alphanumeric, so `read` can't grab `thread`), remap it to the full name.
/// Ambiguous or absent matches are left untouched. Keeps stock omp working
/// without renaming tools.
fn remap_tool_call_names(calls: &mut [ParsedToolCall], tools: &[ToolDef<'_>]) {
    if calls.is_empty() || tools.is_empty() {
        return;
    }
    for c in calls.iter_mut() {
        if tools.iter().any(|t| t.name == c.name) {
            continue; // already a valid, exact name
        }
        let emitted = c.name.as_str();
        let mut hit: Option<&str> = None;
        let mut ambiguous = false;
        for t in tools {
            let n = t.name;
            if n.len() <= emitted.len() || !n.ends_with(emitted) {
                continue;
            }
            let before = n.as_bytes()[n.len() - emitted.len() - 1];
            if before.is_ascii_alphanumeric() {
                continue; // suffix must start at a separator boundary
            }
            if hit.is_some() {
                ambiguous = true;
                break;
            }
            hit = Some(n);
        }
        if ambiguous {
            continue;
        }
        if let Some(full) = hit {
            eprintln!(
                "[mlx] tool-name resolved {:?} -> {:?} (suffix match)",
                c.name, full
            );
            c.name = full.to_string();
        }
    }
}

fn parsed_to_openai_tool_calls(calls: &[ParsedToolCall]) -> Vec<ToolCall> {
    calls
        .iter()
        .map(|c| ToolCall {
            id: format!("call_{}", gen_id()),
            kind: "function".into(),
            function: FunctionCall {
                name: c.name.clone(),
                arguments: serde_json::to_string(&c.arguments).unwrap_or_else(|_| "{}".into()),
            },
        })
        .collect()
}

// ── Channel-based Engine Handle ──────────────────────────────────────────

use tokio::sync::{mpsc, oneshot};

/// Events emitted during streaming generation.
pub enum StreamEvent {
    /// New visible-text fragment (assistant content).
    Delta(String),
    /// Reasoning-channel fragment (Gemma 4 `<|channel>thought\n…<channel|>`
    /// block content). The SSE handler emits this dual-channel: into the
    /// OpenAI `delta.reasoning` field for spec-compliant clients AND
    /// wrapped in `<think>…</think>` inside `delta.content` for clients
    /// (Ayla UI ChatWindow.tsx) that parse text-tag thinking.
    ReasoningDelta(String),
    /// A tool call has been parsed; emit the OpenAI-style envelope
    /// (id + name once, then incremental `arguments` deltas) or the
    /// Anthropic-style envelope (content_block_start tool_use, then
    /// `input_json_delta` chunks). `index` is monotonically increasing
    /// across all tool calls in the response, NOT counting any
    /// preceding text block.
    ToolCallStart {
        index: u32,
        id: String,
        name: String,
    },
    /// Incremental JSON-string fragment of the tool call's arguments.
    /// Clients accumulate these into a single JSON object. Phase 1.5
    /// MVP emits the full serialized argument string in one chunk per
    /// call; future token-aware backends may emit multiple chunks.
    ToolCallArgumentsDelta { index: u32, partial_json: String },
    /// Closes the tool-call envelope. OpenAI clients ignore this;
    /// Anthropic clients use it to emit `content_block_stop`.
    ToolCallStop { index: u32 },
    /// Generation complete with token counts and a finish-reason hint.
    Done {
        prompt_tokens: u32,
        completion_tokens: u32,
        finish_reason: FinishReason,
    },
    /// Error during generation.
    Error(String),
}

/// Why the model stopped generating. Maps to OpenAI `finish_reason`
/// (`"stop"` / `"tool_calls"` / `"length"`) and Anthropic
/// `stop_reason` (`"end_turn"` / `"tool_use"` / `"max_tokens"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    ToolCalls,
    Length,
}

impl FinishReason {
    pub fn openai_str(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::ToolCalls => "tool_calls",
            FinishReason::Length => "length",
        }
    }
    pub fn anthropic_str(self) -> &'static str {
        match self {
            FinishReason::Stop => "end_turn",
            FinishReason::ToolCalls => "tool_use",
            FinishReason::Length => "max_tokens",
        }
    }
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
    /// Shared lifetime serving counters for `GET /v1/loads`. The same `Arc`
    /// the engine bumps at each chat completion (handed over at startup).
    load_stats: Arc<ServerLoadStats>,
}

impl EngineHandle {
    pub fn new(tx: mpsc::Sender<EngineRequest>, load_stats: Arc<ServerLoadStats>) -> Self {
        Self { tx, load_stats }
    }

    /// Read-side handle to the shared serving counters for the
    /// `/v1/loads` route.
    pub fn load_stats(&self) -> &Arc<ServerLoadStats> {
        &self.load_stats
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

        // Overlap scheduling (default ON): defer each token's detokenize +
        // channel-send by one decode step so the main loop can issue the next
        // forward (enqueue GPU work) before paying the CPU post-processing
        // cost. `LUMEN_BATCHED_NO_OVERLAP=1` falls back to the original fully
        // synchronous detok-in-place behavior — a field kill-switch for
        // regressions, no rebuild required.
        let overlap_enabled = std::env::var("LUMEN_BATCHED_NO_OVERLAP").is_err();

        eprintln!(
            "[batched engine] active scheduler (max_batch={max_batch}, overlap={})",
            if overlap_enabled { "on" } else { "off" },
        );

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
                            req.enable_thinking_with_backend_default(
                                self.backend.is_reasoning_first_family(),
                            ),
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
                                messages.push(ChatMessage::new_text("system", system_text));
                            }
                        }
                        for msg in &req.messages {
                            messages.push(ChatMessage::new_text(
                                msg.role.clone(),
                                msg.content.as_text(),
                            ));
                        }
                        match self.start_streaming_seq(
                            next_seq_id,
                            &messages,
                            req.max_tokens,
                            req.temperature,
                            0.95, // AnthropicRequest has no top_p; use Gemma 4 default (matches default_top_p)
                            req.enable_thinking_with_backend_default(
                                self.backend.is_reasoning_first_family(),
                            ),
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
                                        req.enable_thinking_with_backend_default(
                                            self.backend.is_reasoning_first_family(),
                                        ),
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
                                            messages
                                                .push(ChatMessage::new_text("system", system_text));
                                        }
                                    }
                                    for msg in &req.messages {
                                        messages.push(ChatMessage::new_text(
                                            msg.role.clone(),
                                            msg.content.as_text(),
                                        ));
                                    }
                                    match self.start_streaming_seq(
                                        next_seq_id,
                                        &messages,
                                        req.max_tokens,
                                        req.temperature,
                                        0.95, // Gemma 4 default top_p (matches default_top_p)
                                        req.enable_thinking_with_backend_default(
                                            self.backend.is_reasoning_first_family(),
                                        ),
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

            // 1b. WS-E Lever 1: advance iteration-level prefill by ONE chunk for
            //     ONE prefilling seq, then FALL THROUGH to decode the ready
            //     seqs. This interleaves a new (possibly very long) prompt's
            //     prefill with the ongoing decode of established sequences:
            //     each loop turn does (≤1 prefill chunk) + (decode of ready
            //     seqs) instead of monopolizing the engine for an entire
            //     multi-second prefill. A seq whose prefix completes this turn
            //     is seeded and joins the decode batch from the next turn (its
            //     first decode token comes from the standard decode path, so
            //     seeding semantics match the synchronous-prefill case).
            //
            //     Only reachable when LUMEN_PAGED_BATCH_DECODE=1 (else
            //     `prefill_remaining` is always None → this block is skipped and
            //     the decode path below is byte-identical to today).
            if iter_level_prefill_enabled() {
                // Round-robin: pick the lowest-id prefilling seq this turn so
                // multiple concurrent prefills make steady, fair progress.
                let next_prefill: Option<u64> = active
                    .iter()
                    .filter(|(_, s)| s.prefill_remaining.is_some())
                    .map(|(id, _)| *id)
                    .min();
                if let Some(id) = next_prefill {
                    let chunk = batched_prefill_chunk();
                    let device = match &self.backend {
                        ModelBackend::GemmaGguf(g) => g.device().clone(),
                        _ => {
                            eprintln!("[batched engine] non-Gemma backend unreachable (prefill)");
                            return;
                        }
                    };
                    // Take cursor out so no borrow of `active` is held across
                    // the model forward.
                    let (prompt_ids, cursor) = active
                        .get_mut(&id)
                        .and_then(|s| s.prefill_remaining.take())
                        .expect("prefilling seq has cursor");
                    let prefix_len = prompt_ids.len() - 1;
                    let start = cursor;
                    let end = (start + chunk).min(prefix_len);

                    let gem = match &mut self.backend {
                        ModelBackend::GemmaGguf(g) => g,
                        _ => return,
                    };
                    gem.model_mut().set_current_seq_id(id);
                    gem.model_mut().set_paged_store_enabled(true);
                    let chunk_ok = match Tensor::new(&prompt_ids[start..end], &device)
                        .and_then(|t| t.unsqueeze(0))
                    {
                        Ok(t) => gem.model_mut().forward(&t, start).map(|_| ()),
                        Err(e) => Err(e),
                    };
                    match chunk_ok {
                        Ok(()) => {
                            if let Some(s) = active.get_mut(&id) {
                                if end >= prefix_len {
                                    s.position = prefix_len;
                                    s.last_token = *prompt_ids.last().unwrap();
                                    s.prefill_remaining = None;
                                    eprintln!(
                                        "[batched engine] seq {id} prefill complete: {} tokens (iter-chunked)",
                                        prompt_ids.len(),
                                    );
                                } else {
                                    s.prefill_remaining = Some((prompt_ids, end));
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("[batched engine] prefill chunk err seq {id}: {e}");
                            if let Some(s) = active.remove(&id) {
                                let _ = s.token_tx.try_send(StreamEvent::Error(e.to_string()));
                            }
                        }
                    }
                    // Fall through to decode the ready seqs this same turn.
                }
            }

            // 2. Decode step. N=1 → single-seq SDPA path (faster at short ctx).
            //    N≥2 → batched paged kernel.
            //    Prefilling seqs are excluded (handled in 1b above); when the
            //    flag is off, no seq is ever prefilling so `ids` == all active.
            let ids: Vec<u64> = active
                .iter()
                .filter(|(_, s)| s.prefill_remaining.is_none())
                .map(|(id, _)| *id)
                .collect();
            if ids.is_empty() {
                // Every active seq is still prefilling — loop back to 1b.
                continue;
            }
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
                // Snapshot sampling params before the forward so we don't hold
                // an immutable borrow of `active` across the forward + the
                // overlap flush (which needs a mutable borrow). `generated` is
                // cloned for the GPU repeat-penalty / CPU sampler input.
                let (s_top_p, s_temperature, s_repeat_penalty, s_generated) = {
                    let seq = active.get(&ids[0]).unwrap();
                    (
                        seq.top_p,
                        seq.temperature,
                        seq.repeat_penalty,
                        seq.generated.clone(),
                    )
                };
                // GPU sampler skips top-p and n-gram penalty. Default behavior
                // is to use it whenever top_p>=1 (no nucleus filter needed).
                // `FORCE_GPU_SAMPLE=1` forces it on even when top_p<1 (slight
                // quality trade-off: nucleus filter skipped).
                let force_gpu = std::env::var("FORCE_GPU_SAMPLE").is_ok();
                let gpu_ok = force_gpu || s_top_p >= 1.0;
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
                // Overlap: `forward` has enqueued this step's GPU kernels but the
                // device→host sync only happens at sampling time below. Detok +
                // emit the *previous* step's deferred token now so that CPU work
                // hides behind the in-flight GPU forward.
                if overlap_enabled {
                    if let Some(seq) = active.get_mut(&ids[0]) {
                        flush_pending_emit(gem, seq);
                    }
                }
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
                        s_temperature,
                        s_repeat_penalty,
                        &s_generated,
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
                        s_temperature,
                        s_top_p,
                        s_repeat_penalty,
                        &s_generated,
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
                // Overlap: the batched forward has enqueued this step's GPU
                // kernels; the device→host sync happens at `to_vec1` below.
                // Detok + emit every seq's previous-step deferred token now so
                // the CPU work overlaps the in-flight GPU forward. FIFO per seq
                // (size-1 pending) preserves exact token order.
                if overlap_enabled {
                    for id in &ids {
                        if let Some(seq) = active.get_mut(id) {
                            flush_pending_emit(gem, seq);
                        }
                    }
                }
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

                // Sampling + state advance + stop-check stay fully synchronous.
                // Only the expensive detokenize + channel-send is deferred:
                //   - overlap ON: stash `next_tok` in `pending_emit`; it is
                //     flushed at the top of the NEXT decode step (after that
                //     step's forward is issued) so the detok hides behind the
                //     in-flight GPU forward. On completion we flush it
                //     synchronously below so the final text precedes `Done`.
                //   - overlap OFF: detok + emit inline, exactly as before.
                if overlap_enabled {
                    seq.pending_emit = Some(next_tok);
                } else if let Ok(text) = gem.decode(&seq.generated) {
                    if !text.contains('\u{FFFD}') && text.len() > seq.prev_text.len() {
                        let delta = text[seq.prev_text.len()..].to_string();
                        if !delta.is_empty() {
                            let _ = seq.token_tx.try_send(StreamEvent::Delta(delta));
                            seq.prev_text = text;
                        }
                    }
                }

                if seq.eos_tokens.contains(&next_tok) || seq.generated.len() >= seq.max_new {
                    // Flush any deferred text before `Done` so the stream is
                    // complete and correctly truncated at the stop token (no
                    // token sampled after the stop is ever emitted — we stop
                    // sampling here). No-op when overlap is off.
                    flush_pending_emit(gem, seq);
                    let decode_ms = seq.decode_start.elapsed().as_secs_f64() * 1000.0;
                    let n_gen = seq.generated.len();
                    let per_seq_tps = n_gen as f64 / (decode_ms / 1000.0);
                    eprintln!(
                        "[batched engine] seq {id} done: {n_gen} tokens in {decode_ms:.0}ms ({per_seq_tps:.1} tok/s)",
                    );
                    let _ = seq.token_tx.try_send(StreamEvent::Done {
                        prompt_tokens: seq.prompt_tokens,
                        completion_tokens: n_gen as u32,
                        finish_reason: FinishReason::Stop,
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
                            req.enable_thinking_with_backend_default(
                                self.backend.is_reasoning_first_family(),
                            ),
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
                                messages.push(ChatMessage::new_text("system", system_text));
                            }
                        }
                        for msg in &req.messages {
                            messages.push(ChatMessage::new_text(
                                msg.role.clone(),
                                msg.content.as_text(),
                            ));
                        }
                        match self.start_streaming_seq_qwen35(
                            next_seq_id,
                            &messages,
                            req.max_tokens,
                            req.temperature,
                            0.95, // Gemma 4 default top_p (matches default_top_p)
                            req.enable_thinking_with_backend_default(
                                self.backend.is_reasoning_first_family(),
                            ),
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
                        self.dispatch_request_sequential(other);
                    }
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => return,
                }
            }

            if active.is_empty() {
                match rx.recv().await {
                    Some(req) => {
                        if matches!(
                            req,
                            EngineRequest::StreamingChatCompletion { .. }
                                | EngineRequest::StreamingAnthropicMessages { .. }
                        ) {
                            match req {
                                EngineRequest::StreamingChatCompletion { req, token_tx } => {
                                    match self.start_streaming_seq_qwen35(
                                        next_seq_id,
                                        &req.messages,
                                        req.max_tokens,
                                        req.temperature,
                                        req.top_p,
                                        req.enable_thinking_with_backend_default(
                                            self.backend.is_reasoning_first_family(),
                                        ),
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
                                            messages
                                                .push(ChatMessage::new_text("system", system_text));
                                        }
                                    }
                                    for msg in &req.messages {
                                        messages.push(ChatMessage::new_text(
                                            msg.role.clone(),
                                            msg.content.as_text(),
                                        ));
                                    }
                                    match self.start_streaming_seq_qwen35(
                                        next_seq_id,
                                        &messages,
                                        req.max_tokens,
                                        req.temperature,
                                        0.95, // Gemma 4 default top_p (matches default_top_p)
                                        req.enable_thinking_with_backend_default(
                                            self.backend.is_reasoning_first_family(),
                                        ),
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

            // 2. Decode step.
            let ids: Vec<u64> = active.keys().copied().collect();
            let last_tokens: Vec<u32> = ids.iter().map(|id| active[id].last_token).collect();
            let positions: Vec<usize> = ids.iter().map(|id| active[id].position).collect();

            let q = match &mut self.backend {
                #[cfg(feature = "turboquant-gpu")]
                ModelBackend::Qwen35Moe(q) => q,
                _ => {
                    eprintln!("[qwen35 batched] non-Qwen35 backend unreachable");
                    return;
                }
            };

            let t_step = std::time::Instant::now();
            let breakdown = std::env::var("LUMEN_HTTP_BREAKDOWN")
                .map(|v| v == "1")
                .unwrap_or(false);

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
            let t_decode = if breakdown {
                Some(std::time::Instant::now())
            } else {
                None
            };

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
                let am = match logits
                    .argmax(candle_core::D::Minus1)
                    .and_then(|t| t.flatten_all())
                    .and_then(|t| t.to_vec1::<u32>())
                {
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
                let v = match logits
                    .to_dtype(DType::F32)
                    .and_then(|t| t.flatten_all())
                    .and_then(|t| t.to_vec1())
                {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("[qwen35 batched] logits cpu err: {e}");
                        continue;
                    }
                };
                (v, None)
            };
            let t_xfer = if breakdown {
                Some(std::time::Instant::now())
            } else {
                None
            };
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
                let ts0 = if breakdown {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                let next_tok = if let Some(ref am) = argmax_idx {
                    am[row]
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
                            eprintln!("[qwen35 batched] sample err seq {id}: {e}");
                            continue;
                        }
                    }
                };
                let ts1 = if breakdown {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
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
                            let ts2 = if breakdown {
                                Some(std::time::Instant::now())
                            } else {
                                None
                            };
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
                        finish_reason: FinishReason::Stop,
                    });
                    to_remove.push(id);
                }
            }
            let step_ms = t_step.elapsed().as_secs_f64() * 1000.0;
            if breakdown {
                let decode_ms = t_decode.unwrap().duration_since(t_step).as_secs_f64() * 1000.0;
                let xfer_ms = t_xfer
                    .unwrap()
                    .duration_since(t_decode.unwrap())
                    .as_secs_f64()
                    * 1000.0;
                eprintln!(
                    "[qwen35 batched] step: N={} total={:.1}ms (decode={:.1} xfer={:.1} sample={:.1} decode_text={:.1} send={:.1})",
                    ids.len(),
                    step_ms,
                    decode_ms,
                    xfer_ms,
                    sample_ms,
                    decode_text_ms,
                    send_ms,
                );
            } else {
                eprintln!(
                    "[qwen35 batched] step: N={} latency={:.1}ms agg={:.1} tok/s",
                    ids.len(),
                    step_ms,
                    ids.len() as f64 / (step_ms / 1000.0),
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
            top_p: if top_p <= 0.0 { 0.95 } else { top_p },
            repeat_penalty,
            eos_tokens,
            pending_emit: None,
            // Qwen35 has no paged backend; prefill is always synchronous here.
            prefill_remaining: None,
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
        //
        // OPT-IN chunked prefill (BATCHED_ENGINE=1 only): rather than one giant
        // forward over the whole prompt — which monopolizes a single forward and
        // stalls every other batched sequence at head-of-line — split the prefix
        // into fixed token chunks and forward each at its running position offset.
        // The GGUF model is stateful: forwarding prefix[start..end] at index_pos
        // = start grows the KV cache exactly as a single forward would, so the
        // final KV state (and thus the seeded first token) is bit-identical.
        // Earlier chunks' logits are discarded; the prefill logits are unused
        // here regardless (last_token comes from prompt_ids.last()), so no
        // logits need to be returned from the loop.
        let prefix_len = prompt_ids.len() - 1;

        // WS-E Lever 1: when iteration-level prefill is enabled AND the prompt
        // is long enough to actually stall the batch, defer prefill into the
        // main loop. The seq is admitted with a `prefill_remaining` cursor; the
        // loop forwards one chunk per iteration so other active seqs keep
        // decoding. The KV cache is grown identically chunk-by-chunk (the GGUF
        // model is stateful: forwarding prefix[start..end] at index_pos=start
        // produces a bit-identical final KV state vs a single forward), so the
        // seeded first token is unchanged. Short prompts and the flag-off case
        // fall through to the original synchronous path below (byte-identical).
        let chunk = batched_prefill_chunk();
        let defer_prefill = iter_level_prefill_enabled() && prefix_len > chunk;

        let (position, last_token, prefill_remaining) = if defer_prefill {
            // Admit immediately; the loop advances the cursor. position is the
            // *committed* prefill length (0 until the first chunk runs).
            (0usize, prompt_ids[0], Some((prompt_ids.clone(), 0usize)))
        } else {
            let t_prefill = std::time::Instant::now();
            let prefix = &prompt_ids[..prefix_len];
            if !prefix.is_empty() {
                if prefix.len() > chunk {
                    // Long prompt: chunk to bound per-forward attention cost.
                    let mut start = 0usize;
                    while start < prefix.len() {
                        let end = (start + chunk).min(prefix.len());
                        let t = Tensor::new(&prefix[start..end], &device)?.unsqueeze(0)?;
                        let _ = gem.model_mut().forward(&t, start)?;
                        start = end;
                    }
                } else {
                    // Short prompt: EXACT current single-forward path (byte-identical).
                    let t = Tensor::new(prefix, &device)?.unsqueeze(0)?;
                    let _ = gem.model_mut().forward(&t, 0)?;
                }
            }
            let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "[batched engine] seq {seq_id} prefill: {} tokens in {prefill_ms:.0}ms ({:.1} tok/s)",
                prompt_ids.len(),
                prompt_ids.len() as f64 / (prefill_ms / 1000.0),
            );
            (prefix.len(), *prompt_ids.last().unwrap(), None)
        };

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
            top_p: if top_p <= 0.0 { 0.95 } else { top_p },
            repeat_penalty,
            eos_tokens: vec![1, 106], // Gemma: <eos>=1, <end_of_turn>=106
            pending_emit: None,
            prefill_remaining,
        })
    }
}

#[cfg(test)]
mod tool_name_resolve_tests {
    use super::remap_tool_call_names;
    use lumen_mlx::chat_io::{ParsedToolCall, ToolDef};
    use serde_json::Value as JsonValue;

    fn td(name: &str) -> ToolDef<'_> {
        ToolDef {
            name,
            description: None,
            parameters: None,
            response: None,
        }
    }
    fn tc(name: &str) -> ParsedToolCall {
        ParsedToolCall {
            name: name.to_string(),
            arguments: JsonValue::Null,
        }
    }

    #[test]
    fn resolves_dropped_mcp_namespace_by_unique_suffix() {
        let tools = [
            td("read"),
            td("mcp__lean_ctx_ctx_read"),
            td("mcp__lean_ctx_ctx_tree"),
            td("bash"),
        ];
        // Exact native name is kept (exact match wins over suffix).
        let mut a = [tc("read")];
        remap_tool_call_names(&mut a, &tools);
        assert_eq!(a[0].name, "read");
        // Abbreviated MCP names → full namespaced names.
        let mut b = [tc("ctx_read")];
        remap_tool_call_names(&mut b, &tools);
        assert_eq!(b[0].name, "mcp__lean_ctx_ctx_read");
        let mut c = [tc("ctx_tree")];
        remap_tool_call_names(&mut c, &tools);
        assert_eq!(c[0].name, "mcp__lean_ctx_ctx_tree");
    }

    #[test]
    fn leaves_unknown_and_ambiguous_untouched() {
        let tools = [td("mcp__a_get"), td("mcp__b_get")];
        // Ambiguous suffix `get` matches two tools → unchanged.
        let mut a = [tc("get")];
        remap_tool_call_names(&mut a, &tools);
        assert_eq!(a[0].name, "get");
        // No suffix match at all → unchanged.
        let mut b = [tc("nonexistent")];
        remap_tool_call_names(&mut b, &tools);
        assert_eq!(b[0].name, "nonexistent");
    }

    #[test]
    fn requires_separator_boundary() {
        // `read` must NOT be rewritten to `thread` — the suffix has no
        // separator before it.
        let tools = [td("thread")];
        let mut a = [tc("read")];
        remap_tool_call_names(&mut a, &tools);
        assert_eq!(a[0].name, "read");
    }
}

#[cfg(test)]
mod backend_mode_tests {
    use super::{BackendMode, resolve_backend_mode_from};
    use std::collections::HashMap;

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |name: &str| map.get(name).cloned()
    }

    #[cfg(feature = "mlx-native")]
    const DEFAULT: BackendMode = BackendMode::Mlx;
    #[cfg(not(feature = "mlx-native"))]
    const DEFAULT: BackendMode = BackendMode::Candle;

    #[test]
    fn default_matches_compiled_feature() {
        assert_eq!(resolve_backend_mode_from(env(&[])), DEFAULT);
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
    fn auto_resolves_to_default() {
        assert_eq!(
            resolve_backend_mode_from(env(&[("LUMEN_MODE", "auto")])),
            DEFAULT
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
    fn invalid_value_falls_back_to_default() {
        assert_eq!(
            resolve_backend_mode_from(env(&[("LUMEN_MODE", "gpt-5")])),
            DEFAULT
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
    fn legacy_use_mlx_other_values_fall_through_to_default() {
        // Only USE_MLX=1 historically meant "on". Other values fall through.
        assert_eq!(
            resolve_backend_mode_from(env(&[("USE_MLX", "true")])),
            DEFAULT
        );
        assert_eq!(resolve_backend_mode_from(env(&[("USE_MLX", "0")])), DEFAULT);
    }

    #[test]
    fn lumen_mode_takes_precedence_over_use_mlx() {
        // Explicit candle overrides legacy USE_MLX=1.
        assert_eq!(
            resolve_backend_mode_from(env(&[("LUMEN_MODE", "candle"), ("USE_MLX", "1"),])),
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
        assert_eq!(detect_architecture("Qwen/Qwen3.6-35B-A3B"), "qwen3_5_moe");
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
        assert_eq!(detect_architecture("Qwen/Qwen2.5-1.5B-Instruct"), "qwen2");
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
    /// Overlap scheduling: a token that has been sampled, appended to
    /// `generated`, and stop-checked in a *previous* decode step, but whose
    /// detokenize + channel-send was deferred by one iteration so the main
    /// loop could issue the next forward before paying the CPU post-processing
    /// cost. `None` until the seq has produced at least one token (or after a
    /// flush). Per-seq FIFO of size 1 → strict token order is preserved.
    pub pending_emit: Option<u32>,
    /// Iteration-level chunked prefill cursor (Lever 1 — head-of-line removal).
    ///
    /// When `Some((prompt_ids, cursor))`, this sequence has NOT finished its
    /// prefill: the main loop forwards one chunk (`min(remaining, chunk)`
    /// tokens of `prompt_ids[..len-1]`) per iteration at running position
    /// `cursor`, so already-decoding sequences keep advancing between this
    /// seq's prefill chunks instead of stalling behind a single monolithic
    /// prefill forward. When the cursor reaches `len-1`, `last_token`/`position`
    /// are seeded and this field is set to `None` (seq becomes decode-ready).
    ///
    /// Only populated when `LUMEN_PAGED_BATCH_DECODE=1` (default OFF). With the
    /// flag off, prefill completes synchronously in `start_streaming_seq` and
    /// this stays `None`, so the decode loop is byte-identical to today.
    pub prefill_remaining: Option<(Vec<u32>, usize)>,
}

/// Overlap-scheduling helper: detokenize the seq's full `generated` prefix and
/// emit the incremental text delta vs `prev_text`, then clear `pending_emit`.
///
/// This is intentionally *cumulative* (decode the whole prefix, diff against
/// `prev_text`) rather than per-token: the diff `text[prev_text.len()..]`
/// guarantees that emitting "one decode step late" produces a byte-for-byte
/// identical stream to emitting eagerly — no byte is ever dropped or
/// duplicated, regardless of how many tokens accumulated since the last flush.
/// Multi-byte UTF-8 boundaries are handled by the replacement-char guard
/// exactly as the original synchronous path did.
#[inline]
fn flush_pending_emit(gem: &GemmaGgufModel, seq: &mut ActiveSeqState) {
    if seq.pending_emit.is_none() {
        return;
    }
    if let Ok(text) = gem.decode(&seq.generated) {
        if !text.contains('\u{FFFD}') && text.len() > seq.prev_text.len() {
            let delta = text[seq.prev_text.len()..].to_string();
            if !delta.is_empty() {
                let _ = seq.token_tx.try_send(StreamEvent::Delta(delta));
                seq.prev_text = text;
            }
        }
    }
    seq.pending_emit = None;
}

// ─────────────────────────────────────────────────────────────────────────
// Phase 1.4 helpers — unit tests
//
// Pure functions only; full integration (request → backend dispatch →
// response) is exercised via manual smoke tests against a loaded model.
// ─────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod phase_1_4_dispatch_tests {
    use super::*;

    fn mk_msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage::new_text(role, content)
    }

    #[test]
    fn needs_structured_history_plain_chat_returns_false() {
        let msgs = vec![mk_msg("user", "weather?")];
        assert!(!needs_structured_history(&msgs));
    }

    #[test]
    fn needs_structured_history_with_tool_role_returns_true() {
        let msgs = vec![
            mk_msg("user", "weather?"),
            mk_msg("assistant", ""),
            ChatMessage {
                role: "tool".into(),
                content: "20C".into(),
                tool_call_id: Some("call_abc".into()),
                ..Default::default()
            },
        ];
        assert!(needs_structured_history(&msgs));
    }

    #[test]
    fn needs_structured_history_with_assistant_tool_calls_returns_true() {
        let msgs = vec![
            mk_msg("user", "weather?"),
            ChatMessage {
                role: "assistant".into(),
                content: "".into(),
                tool_calls: Some(vec![ToolCall {
                    id: "call_abc".into(),
                    kind: "function".into(),
                    function: FunctionCall {
                        name: "get_weather".into(),
                        arguments: "{}".into(),
                    },
                }]),
                ..Default::default()
            },
        ];
        assert!(needs_structured_history(&msgs));
    }

    #[test]
    fn needs_structured_history_empty_tool_calls_array_returns_false() {
        let msgs = vec![
            mk_msg("user", "weather?"),
            ChatMessage {
                role: "assistant".into(),
                content: "Seoul is sunny".into(),
                tool_calls: Some(vec![]),
                ..Default::default()
            },
        ];
        // An assistant message with an explicit-but-empty tool_calls array
        // is just a normal text turn — the structured path would add
        // no value here.
        assert!(!needs_structured_history(&msgs));
    }

    #[test]
    fn flatten_turns_preserves_text_roles() {
        let turns = vec![
            ChatTurn::System("be brief"),
            ChatTurn::User("hi"),
            ChatTurn::Assistant {
                text: "hello",
                tool_calls: &[],
            },
        ];
        let flat = flatten_turns_for_plain_backend(&turns);
        assert_eq!(
            flat,
            vec![
                ("system".to_string(), "be brief".to_string()),
                ("user".to_string(), "hi".to_string()),
                ("assistant".to_string(), "hello".to_string()),
            ]
        );
    }

    #[test]
    fn flatten_turns_maps_tool_to_user_marker() {
        // Tool turns must NOT propagate as role="tool" to legacy backends
        // (they reject unknown roles). They get wrapped as user-role text
        // with a marker so the chat-template still accepts them.
        let turns = vec![
            ChatTurn::User("weather"),
            ChatTurn::Assistant {
                text: "",
                tool_calls: &[],
            },
            ChatTurn::Tool {
                tool_call_id: "call_abc",
                name: Some("get_weather"),
                content: "20C sunny",
            },
        ];
        let flat = flatten_turns_for_plain_backend(&turns);
        assert_eq!(flat[2].0, "user");
        assert!(flat[2].1.contains("20C sunny"));
        assert!(flat[2].1.contains("[tool result]"));
    }
}
