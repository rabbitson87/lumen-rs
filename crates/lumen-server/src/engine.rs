use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::load_stats::ServerLoadStats;
use crate::types::*;
use anyhow::Result;
use lumen_mlx::SamplingOverrides;
use lumen_mlx::chat_io::{
    AssistantToolCall, BackendStreamEvent, ChatTurn, ParsedResponse, ParsedToolCall,
    ResolvedToolChoice, ToolDef,
};

/// Model backend — supports multiple architectures.
/// The engine's model backend.
///
/// A single variant since the Candle backends were removed (task 006). It is
/// kept as an enum rather than collapsed into the inner type because this is
/// the seam a second backend would plug into, and because collapsing it would
/// churn every call site for no behavior change.
enum ModelBackend {
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
            Self::Mlx(m) => m.encode(text),
        }
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        match self {
            Self::Mlx(m) => m.decode(tokens),
        }
    }

    /// Tokenize the chat-templated prompt for accurate `prompt_tokens`. Falls
    /// back to a `len/4` heuristic only if the backend errors during encode.
    ///
    /// Takes the whole `SamplingOverrides` rather than a pre-extracted effort
    /// **so that it cannot be handed a different one than the renderer uses**.
    /// It was, and the earlier version of this comment named the hazard exactly
    /// ("`effort` must match what the real request will render with") while all
    /// six call sites passed the raw `ov.reasoning_effort`. On a checkpoint that
    /// does not declare `reasoning_effort` the renderer correctly drops it and
    /// the counter did not: measured on Qwen3.5-9B, a `thinking: true` request
    /// prefilled 12 tokens and reported 54. Wrong usage, and the same figure
    /// feeds the context guard, so a request near the limit could be rejected
    /// for tokens it never had.
    ///
    /// Now there is one source: `resolved_effort`, the same call the decode
    /// paths make.
    fn count_chat_prompt_tokens(
        &self,
        messages: &[(String, String)],
        thinking: bool,
        ov: &lumen_mlx::SamplingOverrides,
    ) -> u32 {
        let res: Result<Vec<u32>> = match self {
            Self::Mlx(m) => m.build_chat_input(messages, thinking, m.resolved_effort(ov)),
        };
        match res {
            Ok(ids) => ids.len() as u32,
            Err(_) => {
                let chars: usize = messages.iter().map(|(_, c)| c.len()).sum();
                ((chars as u32) / 4).max(1)
            }
        }
    }

    /// Extra prompt tokens contributed by inline images.
    ///
    /// `count_chat_prompt_tokens` renders text only, so an image request's real
    /// prompt is hundreds of tokens longer than it reports — enough to matter
    /// both for the context guard and for the usage figures we hand back.
    /// Header-only, so this stays cheap even for a request we then reject.
    fn image_prompt_tokens(&self, images: &[Vec<Vec<u8>>]) -> u32 {
        match self {
            Self::Mlx(m) => m.image_prompt_tokens(images) as u32,
        }
    }

    /// Hard model context ceiling (tokens), when the backend exposes one.
    /// Used by the prompt-size guard as an absolute reject limit: a prompt
    /// over this can't fit the KV and OOM-aborts MLX, so it must be rejected
    /// regardless of `LUMEN_MAX_PROMPT_TOKENS`.
    fn max_context(&self) -> Option<u32> {
        match self {
            Self::Mlx(m) => m.max_context().map(|c| c as u32),
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
        match self {
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
        }
    }

    /// Drop an A1 prefix-cache entry by its auto-generated key. Returns true
    /// if the entry existed. MLX-only feature.
    fn drop_prefix_cache(&mut self, key: &str) -> bool {
        match self {
            Self::Mlx(m) => m.drop_prefix_cache(key),
        }
    }

    /// Clear all A1 prefix-cache entries. Returns the number released.
    fn clear_prefix_cache(&mut self) -> usize {
        match self {
            Self::Mlx(m) => m.clear_prefix_cache(),
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

    /// [`Self::chat`] with inline images (`images[i]` belongs to `messages[i]`).
    /// Requires a vision-capable model (Gemma 4 or Qwen 3.6, with
    /// `LUMEN_VISION=1`); the backend rejects the request rather than silently
    /// answering without the image.
    #[allow(clippy::too_many_arguments)]
    fn chat_with_images(
        &mut self,
        messages: &[(String, String)],
        images: &[Vec<Vec<u8>>],
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
            Self::Mlx(m) => m.chat_with_images(
                messages,
                images,
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
        on_event: F,
    ) -> Result<ParsedResponse>
    where
        F: FnMut(BackendStreamEvent<'_>) -> Result<()>,
    {
        match self {
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

    /// [`Self::chat_from_history`] with images attached to `User` turns.
    /// Only a vision-capable MLX backend (Gemma 4 or Qwen 3.6) can consume them.
    #[allow(clippy::too_many_arguments)]
    fn chat_from_history_with_images(
        &mut self,
        turns: &[ChatTurn<'_>],
        images: &[Vec<Vec<u8>>],
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
            Self::Mlx(m) => m.chat_from_history_with_images(
                turns,
                images,
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

    /// [`Self::chat_streaming_from_history`] with images on `User` turns.
    #[allow(clippy::too_many_arguments)]
    fn chat_streaming_from_history_with_images<F>(
        &mut self,
        turns: &[ChatTurn<'_>],
        images: &[Vec<Vec<u8>>],
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
        match self {
            Self::Mlx(m) => m.chat_streaming_from_history_with_images(
                turns,
                images,
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

    /// [`Self::chat_streaming`] with inline images. Only the MLX Gemma 4
    /// backend can consume them; every other backend rejects the request
    /// rather than streaming an answer that never saw the image.
    #[allow(clippy::too_many_arguments)]
    fn chat_streaming_with_images<F>(
        &mut self,
        messages: &[(String, String)],
        images: &[Vec<Vec<u8>>],
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
        match self {
            Self::Mlx(m) => m.chat_streaming_with_images(
                messages,
                images,
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
        let Self::Mlx(m) = self;
        m.chat_from_history(
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
        let Self::Mlx(m) = self;
        m.chat_streaming_from_history(
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
        )
    }
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

impl InferenceEngine {
    /// Load a model, auto-detecting architecture from `model_id`.
    ///
    /// `MlxBackend::load` does the detection internally — Qwen 2.5 / 3.5 / 3.6
    /// dense and MoE, Gemma 4, and whatever is added next — so the engine only
    /// ever sees one backend variant.
    pub fn load(model_id: &str) -> Result<Self> {
        eprintln!("Loading MLX backend: {model_id}");
        let backend = lumen_mlx::MlxBackend::load(model_id)?;
        eprintln!("[mlx] family={:?}", backend.kind());
        let cfg_summary = backend.runtime_config_summary();
        if !cfg_summary.is_empty() {
            eprintln!("[mlx-config] {cfg_summary}");
        }
        Ok(Self {
            backend: ModelBackend::Mlx(backend),
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
        // `kept[i]` is the index in `req.messages` that survived as
        // `messages[i]`. Any per-message side table has to be re-indexed
        // through it — the strip *removes* turns, so a naive `req.messages`
        // walk would bind image `i` to the wrong message (or to none).
        let kept = lumen_mlx::chat_io::strip_client_meta_wrappers_flat_indexed(&mut messages);
        let images = images_aligned_to_kept(&req.messages, &kept);

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
        let tools_owned = tools_visible_to_model(tools_owned, &tool_choice);
        // Resolve enable_thinking once — the backend hint covers the
        // common case where OpenAI-compat clients send placeholder model
        // ids ("gpt-3.5-turbo") that hide the actual loaded family.
        let thinking_on =
            req.enable_thinking_with_backend_default(self.backend.is_reasoning_first_family());
        let ov = req.sampling_overrides();
        // Prompt-size reject cap — guard the MLX prefill from OOM-crashing on
        // oversized prompts. The streaming path already had this guard; the
        // non-streaming path did NOT, so a large prompt reached prefill and
        // aborted the whole process via an uncaught Metal OOM. Reject early
        // with a clean error instead. (count_chat_prompt_tokens is a cheap
        // tokenize, no forward; the same count is reused for usage below.)
        // Images add their soft-token runs on top of the rendered text; count
        // them here too or an image request slips past the guard by hundreds of
        // tokens per image.
        let image_tokens = images
            .as_deref()
            .map(|i| self.backend.image_prompt_tokens(i))
            .unwrap_or(0);
        let prompt_tokens_guard = self.backend.count_chat_prompt_tokens(
            &messages,
            thinking_on,
            &req.sampling_overrides(),
        ) + image_tokens;
        guard_prompt_fits(&self.backend, prompt_tokens_guard)?;
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
            // `turns` is 1:1 with `req.messages` and the turn strip applies the
            // same predicate as the flat one, so the surviving turns line up
            // with `images` — which is already indexed by the survivors.
            match images.as_deref() {
                Some(imgs) => self.backend.chat_from_history_with_images(
                    &turns,
                    imgs,
                    req.max_tokens,
                    req.temperature,
                    req.top_p,
                    &ov,
                    thinking_on,
                    req.session_id.as_deref(),
                    &tools_owned,
                    &tool_choice,
                    req.response_json_schema().as_ref(),
                )?,
                None => self.backend.chat_from_history(
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
                )?,
            }
        } else if let Some(images) = images.as_deref() {
            self.backend.chat_with_images(
                &messages,
                images,
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

        // Same text count as the guard above, plus the image runs the model
        // actually prefilled — reporting the text-only figure would under-count
        // an image request by hundreds of tokens.
        let prompt_tokens = self.backend.count_chat_prompt_tokens(
            &messages,
            thinking_on,
            &req.sampling_overrides(),
        ) + image_tokens;
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
        let tools_owned = tools_visible_to_model(tools_owned, &tool_choice);

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
        // Images travel beside the flattened text, so they have to survive the
        // same strip — see `images_aligned_to_kept` on the OpenAI path.
        let all_images = anthropic_images_flat(&req.messages, system_text.is_some())?;
        let kept = lumen_mlx::chat_io::strip_client_meta_wrappers_flat_indexed(&mut messages);
        let images: Option<Vec<Vec<Vec<u8>>>> = if kept.iter().any(|&i| !all_images[i].is_empty()) {
            Some(kept.iter().map(|&i| all_images[i].clone()).collect())
        } else {
            None
        };

        // Prompt-size reject cap (Anthropic /v1/messages) — same OOM guard as
        // the OpenAI path: reject oversized prompts before prefill rather than
        // letting them reach the backend and crash the server via an uncaught
        // Metal OOM. Uses the flat `messages` count (the same approximation
        // already used for usage below).
        let anthropic_thinking =
            req.enable_thinking_with_backend_default(self.backend.is_reasoning_first_family());
        let prompt_tokens_guard = self.backend.count_chat_prompt_tokens(
            &messages,
            anthropic_thinking,
            &req.sampling_overrides(),
        );
        guard_prompt_fits(&self.backend, prompt_tokens_guard)?;

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

            // Message-indexed view of the already-decoded images (the flat
            // vector carries a leading slot for the synthesized system entry).
            let msg_images: &[Vec<Vec<u8>>] = &all_images[usize::from(system_text.is_some())..];
            let tool_result_counts: Vec<usize> = tool_result_buf.iter().map(Vec::len).collect();
            let user_has_text: Vec<bool> = user_text_buf.iter().map(|s| !s.is_empty()).collect();

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
                        // no text content at all.) An image-only message still
                        // gets a turn — that is where its placeholder run goes.
                        let utext = user_text_buf[i].as_str();
                        if !utext.is_empty() || !msg_images[i].is_empty() {
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
            // Built by replaying the expansion above; the length check catches
            // the two drifting apart rather than letting an image bind to the
            // wrong turn.
            let turn_images = anthropic_turn_images(
                &req.messages,
                system_text.is_some(),
                msg_images,
                &tool_result_counts,
                &user_has_text,
            )?;
            anyhow::ensure!(
                turn_images.len() == turns.len(),
                "anthropic turn/image expansion disagree ({} vs {} turns)",
                turn_images.len(),
                turns.len()
            );
            let kept = lumen_mlx::chat_io::strip_client_meta_wrappers_indexed(&mut turns);
            let turn_images: Vec<Vec<Vec<u8>>> =
                kept.iter().map(|&i| turn_images[i].clone()).collect();
            if turn_images.iter().any(|v| !v.is_empty()) {
                self.backend.chat_from_history_with_images(
                    &turns,
                    &turn_images,
                    req.max_tokens,
                    req.temperature,
                    req.top_p,
                    &ov,
                    req.enable_thinking_with_backend_default(
                        self.backend.is_reasoning_first_family(),
                    ),
                    req.session_id.as_deref(),
                    &tools_owned,
                    &tool_choice,
                    None,
                )?
            } else {
                self.backend.chat_from_history(
                    &turns,
                    req.max_tokens,
                    req.temperature,
                    req.top_p,
                    &ov,
                    req.enable_thinking_with_backend_default(
                        self.backend.is_reasoning_first_family(),
                    ),
                    req.session_id.as_deref(),
                    &tools_owned,
                    &tool_choice,
                    // Anthropic /v1/messages has no `response_format`.
                    None,
                )?
            }
        } else if let Some(images) = images.as_deref() {
            self.backend.chat_with_images(
                &messages,
                images,
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

        // Images add their placeholder runs on top of the rendered text.
        let prompt_tokens = self.backend.count_chat_prompt_tokens(
            &messages,
            req.enable_thinking_with_backend_default(self.backend.is_reasoning_first_family()),
            &req.sampling_overrides(),
        ) + images
            .as_deref()
            .map(|i| self.backend.image_prompt_tokens(i))
            .unwrap_or(0);
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
                    if earliest.is_none_or(|e| i < e) {
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
        // See `chat_completion`: the strip removes turns, so per-message image
        // attachments have to be re-indexed through the surviving positions.
        let kept = lumen_mlx::chat_io::strip_client_meta_wrappers_flat_indexed(&mut messages);
        let images = images_aligned_to_kept(&req.messages, &kept);

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

        // Includes the image soft-token runs; this figure feeds both the
        // context guard below and the `usage` block at the end of the stream.
        let prompt_tokens = self.backend.count_chat_prompt_tokens(
            &messages,
            req.enable_thinking_with_backend_default(self.backend.is_reasoning_first_family()),
            &req.sampling_overrides(),
        ) + images
            .as_deref()
            .map(|i| self.backend.image_prompt_tokens(i))
            .unwrap_or(0);

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
        if let Err(e) = guard_prompt_fits(&self.backend, prompt_tokens) {
            let _ = token_tx.try_send(StreamEvent::Error(e.to_string()));
            return;
        }

        let tools_owned = openai_tools_to_defs(req.tools.as_deref());
        let tool_choice =
            resolve_openai_tool_choice(req.tool_choice.as_ref(), !tools_owned.is_empty());
        let tools_owned = tools_visible_to_model(tools_owned, &tool_choice);
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
            // See the non-streaming path: post-strip turns line up with
            // `images`, which is already indexed by the survivors.
            let on_event = |ev: BackendStreamEvent<'_>| -> Result<()> {
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
            };
            let thinking =
                req.enable_thinking_with_backend_default(self.backend.is_reasoning_first_family());
            match images.as_deref() {
                Some(imgs) => self.backend.chat_streaming_from_history_with_images(
                    &turns,
                    imgs,
                    req.max_tokens,
                    req.temperature,
                    req.top_p,
                    &ov,
                    thinking,
                    req.session_id.as_deref(),
                    &tools_owned,
                    &tool_choice,
                    response_schema.as_ref(),
                    on_event,
                ),
                None => self.backend.chat_streaming_from_history(
                    &turns,
                    req.max_tokens,
                    req.temperature,
                    req.top_p,
                    &ov,
                    thinking,
                    req.session_id.as_deref(),
                    &tools_owned,
                    &tool_choice,
                    response_schema.as_ref(),
                    on_event,
                ),
            }
        } else {
            // Bound first so both the text and the image dispatch below can
            // take it — only one of them runs.
            let on_event = |ev: BackendStreamEvent<'_>| -> Result<()> {
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
            };
            let thinking =
                req.enable_thinking_with_backend_default(self.backend.is_reasoning_first_family());
            match images.as_ref() {
                Some(imgs) => self.backend.chat_streaming_with_images(
                    &messages,
                    imgs,
                    req.max_tokens,
                    req.temperature,
                    req.top_p,
                    &ov,
                    thinking,
                    req.session_id.as_deref(),
                    &tools_owned,
                    &tool_choice,
                    response_schema.as_ref(),
                    on_event,
                ),
                None => self.backend.chat_streaming(
                    &messages,
                    req.max_tokens,
                    req.temperature,
                    req.top_p,
                    &ov,
                    thinking,
                    req.session_id.as_deref(),
                    &tools_owned,
                    &tool_choice,
                    response_schema.as_ref(),
                    on_event,
                ),
            }
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
        // Images ride beside the flattened text and must survive the same
        // strip — see `images_aligned_to_kept` on the OpenAI path.
        let all_images = match anthropic_images_flat(
            &req.messages,
            system_text.as_ref().is_some_and(|s| !s.is_empty()),
        ) {
            Ok(v) => v,
            Err(e) => {
                let _ = token_tx.try_send(StreamEvent::Error(e.to_string()));
                return;
            }
        };
        let kept = lumen_mlx::chat_io::strip_client_meta_wrappers_flat_indexed(&mut messages);
        let images: Option<Vec<Vec<Vec<u8>>>> = if kept.iter().any(|&i| !all_images[i].is_empty()) {
            Some(kept.iter().map(|&i| all_images[i].clone()).collect())
        } else {
            None
        };

        let prompt_tokens = self.backend.count_chat_prompt_tokens(
            &messages,
            req.enable_thinking_with_backend_default(self.backend.is_reasoning_first_family()),
            &req.sampling_overrides(),
        ) + images
            .as_deref()
            .map(|i| self.backend.image_prompt_tokens(i))
            .unwrap_or(0);
        // Prompt-size reject cap (Anthropic streaming) — guard the prefill from
        // an uncaught Metal OOM that would crash the server process.
        if let Err(e) = guard_prompt_fits(&self.backend, prompt_tokens) {
            let _ = token_tx.try_send(StreamEvent::Error(e.to_string()));
            return;
        }

        let tools_owned = anthropic_tools_to_defs(req.tools.as_deref());
        let tool_choice =
            resolve_anthropic_tool_choice(req.tool_choice.as_ref(), !tools_owned.is_empty());
        let tools_owned = tools_visible_to_model(tools_owned, &tool_choice);

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

            // Message-indexed view of the already-decoded images (the flat
            // vector carries a leading slot for the synthesized system entry).
            let msg_images: &[Vec<Vec<u8>>] =
                &all_images[usize::from(system_text.as_ref().is_some_and(|s| !s.is_empty()))..];
            let tool_result_counts: Vec<usize> = tool_result_buf.iter().map(Vec::len).collect();
            let user_has_text: Vec<bool> = user_text_buf.iter().map(|s| !s.is_empty()).collect();

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
                        // An image-only message still gets a turn — that is
                        // where its placeholder run goes.
                        if !user_text_buf[i].is_empty() || !msg_images[i].is_empty() {
                            turns.push(ChatTurn::User(user_text_buf[i].as_str()));
                        }
                    }
                    _ => {}
                }
            }
            // Built by replaying the expansion above; the length check catches
            // the two drifting apart rather than letting an image bind to the
            // wrong turn.
            let turn_images = match anthropic_turn_images(
                &req.messages,
                system_text.as_ref().is_some_and(|s| !s.is_empty()),
                msg_images,
                &tool_result_counts,
                &user_has_text,
            ) {
                Ok(v) if v.len() == turns.len() => v,
                Ok(v) => {
                    let _ = token_tx.try_send(StreamEvent::Error(format!(
                        "anthropic turn/image expansion disagree ({} vs {} turns)",
                        v.len(),
                        turns.len()
                    )));
                    return;
                }
                Err(e) => {
                    let _ = token_tx.try_send(StreamEvent::Error(e.to_string()));
                    return;
                }
            };
            let kept = lumen_mlx::chat_io::strip_client_meta_wrappers_indexed(&mut turns);
            let turn_images: Vec<Vec<Vec<u8>>> =
                kept.iter().map(|&i| turn_images[i].clone()).collect();
            let on_event = |ev: BackendStreamEvent<'_>| -> Result<()> {
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
            };
            if turn_images.iter().any(|v| !v.is_empty()) {
                self.backend.chat_streaming_from_history_with_images(
                    &turns,
                    &turn_images,
                    req.max_tokens,
                    req.temperature,
                    req.top_p,
                    &ov,
                    req.enable_thinking_with_backend_default(
                        self.backend.is_reasoning_first_family(),
                    ),
                    req.session_id.as_deref(),
                    &tools_owned,
                    &tool_choice,
                    None,
                    on_event,
                )
            } else {
                self.backend.chat_streaming_from_history(
                    &turns,
                    req.max_tokens,
                    req.temperature,
                    req.top_p,
                    &ov,
                    req.enable_thinking_with_backend_default(
                        self.backend.is_reasoning_first_family(),
                    ),
                    req.session_id.as_deref(),
                    &tools_owned,
                    &tool_choice,
                    // Anthropic Messages API has no `response_format` field.
                    None,
                    on_event,
                )
            }
        } else {
            // Bound first so both dispatches below can take it — only one runs.
            let on_event = |ev: BackendStreamEvent<'_>| -> Result<()> {
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
            };
            let thinking =
                req.enable_thinking_with_backend_default(self.backend.is_reasoning_first_family());
            match images.as_deref() {
                Some(imgs) => self.backend.chat_streaming_with_images(
                    &messages,
                    imgs,
                    req.max_tokens,
                    req.temperature,
                    req.top_p,
                    &ov,
                    thinking,
                    req.session_id.as_deref(),
                    &tools_owned,
                    &tool_choice,
                    // Anthropic Messages API has no `response_format` field.
                    None,
                    on_event,
                ),
                None => self.backend.chat_streaming(
                    &messages,
                    req.max_tokens,
                    req.temperature,
                    req.top_p,
                    &ov,
                    thinking,
                    req.session_id.as_deref(),
                    &tools_owned,
                    &tool_choice,
                    None,
                    on_event,
                ),
            }
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
/// Resolve the prompt-token REJECT cap shared by the streaming and
/// non-streaming chat paths. Guards the MLX prefill from OOM-crashing on
/// oversized prompts: a prompt of ~20k (Gemma-4-26B) to ~32k (Qwen3.6-35B)
/// tokens triggers an uncaught Metal OOM that aborts the whole server
/// process — so an over-cap prompt must be rejected with a clean error
/// BEFORE prefill, not allowed through. Default 16384 — empirically safe on
/// a 36 GB Mac for both families; raise via `LUMEN_MAX_PROMPT_TOKENS` when
/// more RAM is available (longer prefill needs proportionally more memory).
/// Precedence: `LUMEN_MAX_PROMPT_TOKENS` → legacy `LUMEN_PREFILL_CHUNK` → 16K.
fn resolve_prompt_cap() -> u32 {
    std::env::var("LUMEN_MAX_PROMPT_TOKENS")
        .or_else(|_| std::env::var("LUMEN_PREFILL_CHUNK"))
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16_384)
}

/// Effective prompt-token reject limit = min(operator cap, model max_ctx).
/// The model context is a HARD ceiling: a prompt longer than it can't fit the
/// KV cache, and on MLX the over-ctx prefill aborts the whole process with an
/// uncaught Metal OOM. So even when the operator raises `LUMEN_MAX_PROMPT_TOKENS`,
/// `max_ctx` still bounds the accepted prompt. Returns
/// `(effective_limit, operator_cap, max_ctx_opt)` for messaging/logging.
fn effective_prompt_cap_for(backend: &ModelBackend) -> (u32, u32, Option<u32>) {
    let operator_cap = resolve_prompt_cap();
    let max_ctx = backend.max_context();
    let effective = match max_ctx {
        Some(mc) => operator_cap.min(mc),
        None => operator_cap,
    };
    (effective, operator_cap, max_ctx)
}

/// Pre-flight prompt-size guard shared by every chat entry (OpenAI + Anthropic,
/// streaming + non-streaming). Rejects an oversized prompt BEFORE prefill so it
/// never reaches Metal (an over-ctx prefill OOM-aborts the process — a C++
/// exception Rust cannot catch, so rejection is the only defense). On reject it
/// emits a single parseable `[mlx-guard] PROMPT_REJECTED ...` stderr line that
/// the desktop app watches for to pop the "prompt too large" modal, and returns
/// an `Err` whose message explains the cause + the fix.
fn guard_prompt_fits(backend: &ModelBackend, prompt_tokens: u32) -> Result<()> {
    let (effective, operator_cap, max_ctx) = effective_prompt_cap_for(backend);
    if prompt_tokens <= effective {
        return Ok(());
    }
    let max_ctx_field = max_ctx
        .map(|m| m.to_string())
        .unwrap_or_else(|| "none".to_string());
    // Space-delimited key=val so the app's log watcher can parse it without a
    // structured side-channel (the app tails server stderr).
    eprintln!(
        "[mlx-guard] PROMPT_REJECTED prompt_tokens={prompt_tokens} effective_cap={effective} \
         max_ctx={max_ctx_field} prompt_cap={operator_cap}"
    );
    // Bound by max_ctx (not the operator cap) → the real fix is to shrink the
    // prompt or, only if memory allows, raise LUMEN_MAX_CTX. Bound by the
    // operator cap → raising LUMEN_MAX_PROMPT_TOKENS (memory permitting) works.
    let hint = match max_ctx {
        Some(mc) if mc <= operator_cap => format!(
            " The model context is capped at {mc} tokens (LUMEN_MAX_CTX). \
             Trim the prompt/conversation, or raise LUMEN_MAX_CTX only if there is \
             enough memory to prefill it (raising it past what the GPU can hold just \
             re-triggers the out-of-memory crash)."
        ),
        _ => format!(
            " Server prompt cap is {operator_cap} tokens (LUMEN_MAX_PROMPT_TOKENS). \
             Trim the prompt, or raise the cap only if there is enough RAM for the \
             longer prefill."
        ),
    };
    Err(anyhow::anyhow!(
        "prompt too large: {prompt_tokens} tokens > limit {effective}.{hint}"
    ))
}

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

/// Per-message image attachments, re-indexed onto the post-strip message
/// vector: `kept[i]` is the index in `req.messages` that survived as
/// `messages[i]`, so the returned vector lines up index-for-index with the
/// `(role, content)` pairs the backend receives.
///
/// Returns `None` when nothing survived with an image, which is also the
/// "route to the plain text path" signal — a request whose only image rode on
/// a stripped meta-wrapper turn has no image left to encode.
fn images_aligned_to_kept(messages: &[ChatMessage], kept: &[usize]) -> Option<Vec<Vec<Vec<u8>>>> {
    if !ChatMessage::any_images_at(messages, kept) {
        return None;
    }
    Some(kept.iter().map(|&i| messages[i].images.clone()).collect())
}

/// Per-message image attachments from an Anthropic request, aligned with the
/// flattened `(role, content)` vector.
///
/// That vector may carry a synthesized leading `system` entry which has no
/// counterpart in `req.messages`, so the offset has to be reproduced here or
/// every image binds one turn early.
fn anthropic_images_flat(
    messages: &[AnthropicMessage],
    has_system: bool,
) -> Result<Vec<Vec<Vec<u8>>>> {
    let mut out: Vec<Vec<Vec<u8>>> = Vec::with_capacity(messages.len() + 1);
    if has_system {
        // A system block carries no images (Anthropic's system field is text
        // or text blocks only), but the slot must exist to keep the indices
        // lined up with `messages`.
        out.push(Vec::new());
    }
    for msg in messages {
        let mut row = Vec::new();
        if let AnthropicContent::Blocks(blocks) = &msg.content {
            for b in blocks {
                if let AnthropicContentBlock::Image { source } = b {
                    row.push(
                        source
                            .decode()
                            .map_err(|e| anyhow::anyhow!("image block: {e}"))?,
                    );
                }
            }
        }
        out.push(row);
    }
    Ok(out)
}

/// Per-turn image attachments for the Anthropic structured path.
///
/// The message-indexed vector cannot be reused here. One Anthropic message
/// expands into several turns — a user message with N `tool_result` blocks
/// becomes N `Tool` turns followed by its `User` turn — so from the first tool
/// result onward, message index and turn index diverge and every later image
/// would bind to the wrong turn. This replays that same expansion and emits
/// exactly one row per turn.
///
/// `has_system_turn` mirrors whatever condition the caller used to push its
/// `ChatTurn::System`; the two Anthropic call sites disagree about the empty
/// system string, so it is passed rather than re-derived.
///
/// Images on an assistant turn are refused. There is nowhere to put them: the
/// renderers place a placeholder run at the head of a user turn, and an
/// assistant turn carrying tool calls may render no text at all.
fn anthropic_turn_images(
    messages: &[AnthropicMessage],
    has_system_turn: bool,
    msg_images: &[Vec<Vec<u8>>],
    tool_result_counts: &[usize],
    user_has_text: &[bool],
) -> Result<Vec<Vec<Vec<u8>>>> {
    let mut out: Vec<Vec<Vec<u8>>> = Vec::with_capacity(messages.len() + 4);
    if has_system_turn {
        out.push(Vec::new());
    }
    for (i, msg) in messages.iter().enumerate() {
        let attached = msg_images.get(i).cloned().unwrap_or_default();
        match msg.role.as_str() {
            "assistant" => {
                if !attached.is_empty() {
                    return Err(anyhow::anyhow!(
                        "images can only be attached to a user message; message {i} is an \
                         assistant turn"
                    ));
                }
                out.push(Vec::new());
            }
            "user" => {
                // Tool turns come first and never carry images.
                for _ in 0..tool_result_counts.get(i).copied().unwrap_or(0) {
                    out.push(Vec::new());
                }
                // The User turn is emitted when the message has text *or*
                // images — an image-only message still needs a turn to hang
                // its placeholder run on.
                if user_has_text.get(i).copied().unwrap_or(false) || !attached.is_empty() {
                    out.push(attached);
                }
            }
            _ => {}
        }
    }
    Ok(out)
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

/// Drop the tool definitions when `tool_choice` resolved to `"none"`.
///
/// `"none"` promises the model generates a message instead of calling a tool.
/// Until this existed the promise was not kept: the resolver produced
/// `ResolvedToolChoice::None`, the backends used it to *skip building a
/// grammar*, and nothing else looked at it — so the tool block still went into
/// the prompt and Qwen 3.6 answered `"What is the weather in Busan?"` with a
/// `get_weather` call. Gemma 4 happened to comply, which is luck, not a
/// guarantee.
///
/// Withholding the definitions makes it structural: a model cannot call what it
/// was never shown. Filtering the parsed call afterwards would be the
/// alternative and is worse — the model would have spent its budget on a call
/// that gets thrown away, leaving an empty reply.
fn tools_visible_to_model<'a>(
    tools: Vec<ToolDef<'a>>,
    tool_choice: &ResolvedToolChoice<'_>,
) -> Vec<ToolDef<'a>> {
    if matches!(tool_choice, ResolvedToolChoice::None) {
        Vec::new()
    } else {
        tools
    }
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
        // `disable_parallel_tool_use` rides on each variant but says nothing
        // about *whether* a tool is called, so it is deliberately ignored here
        // and read separately via `AnthropicToolChoice::parallel_tool_calls`.
        // Conflating the two is the mistake this whole change undoes.
        Some(AnthropicToolChoice::Auto { .. }) => ResolvedToolChoice::Auto,
        Some(AnthropicToolChoice::Any { .. }) if has_tools => ResolvedToolChoice::Required,
        Some(AnthropicToolChoice::Any { .. }) => ResolvedToolChoice::Auto,
        Some(AnthropicToolChoice::Tool { name, .. }) if has_tools => {
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
fn remap_tool_call_names(calls: &mut Vec<ParsedToolCall>, tools: &[ToolDef<'_>]) {
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
            continue;
        }
        // Bug A's mirror image: the emitted name is LONGER than a declared one
        // and ends with it, because the model stuttered its opening bytes.
        // Measured on Qwen3.8-27B: `<function=geget_weather>` for a declared
        // `get_weather`, on the second call of a turn — after the one-call
        // grammar released and the tail decoded unconstrained.
        //
        // Repaired only when the extra prefix is itself a prefix of the
        // declared name, which is what makes it a stutter rather than a
        // different word. Without that guard a declared `list` would swallow an
        // emitted `blacklist`; with it, `black` is not a prefix of `list` and
        // nothing happens.
        if let Some(full) = unique_stutter_match(emitted, tools) {
            eprintln!(
                "[mlx] tool-name resolved {:?} -> {:?} (stutter)",
                c.name, full
            );
            c.name = full.to_string();
        }
    }

    // Whatever is still unmatched is a name the client never declared, and
    // handing one over is a contract violation: the client looks up a function
    // it does not have. Dropping is the conservative half; saying so is the
    // other half, because "accepted and then silently not applied" is the
    // failure mode this file keeps paying for.
    let declared = |c: &ParsedToolCall| tools.iter().any(|t| t.name == c.name);
    if !calls.iter().all(declared) {
        for c in calls.iter().filter(|c| !declared(c)) {
            eprintln!(
                "[mlx] tool-call DROPPED: {:?} is not among the {} declared tools",
                c.name,
                tools.len()
            );
        }
        calls.retain(declared);
    }
}

/// The one declared tool that `emitted` is a stutter of, if exactly one is.
///
/// `emitted` must end with the declared name and the leading remainder must be
/// a prefix of it — `ge` + `get_weather`. Ambiguity yields `None`; so does a
/// remainder that is merely extra text.
fn unique_stutter_match<'a>(emitted: &str, tools: &'a [ToolDef<'a>]) -> Option<&'a str> {
    let mut hit: Option<&str> = None;
    for t in tools {
        let n = t.name;
        if n.len() >= emitted.len() || !emitted.ends_with(n) {
            continue;
        }
        let extra = &emitted[..emitted.len() - n.len()];
        if extra.is_empty() || !n.starts_with(extra) {
            continue;
        }
        if hit.is_some() {
            return None; // ambiguous
        }
        hit = Some(n);
    }
    hit
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

/// Resolve an MLX multi-request feature flag. An explicit per-feature env var
/// always wins (truthy = `1`/`true`/`on`/`yes`, anything else = off); when the
/// per-feature var is UNSET, the flag inherits from `LUMEN_MLX_SERVER_MODE` —
/// the single "multi-request serving" switch that turns on the whole concurrent
/// path (batched decode + shared-prefix KV dedup) for live multi-user serving
/// OR bulk batch jobs. Most MLX users run solo and leave this off, so the
/// default desktop path stays byte-identical.
#[cfg(feature = "mlx-native")]
pub(crate) fn mlx_feature_on(specific: &str) -> bool {
    fn truthy(v: &str) -> bool {
        matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "on" | "yes"
        )
    }
    match std::env::var(specific) {
        Ok(v) => truthy(&v),
        Err(_) => std::env::var("LUMEN_MLX_SERVER_MODE")
            .map(|v| truthy(&v))
            .unwrap_or(false),
    }
}

impl InferenceEngine {
    /// Run the engine loop. Streaming requests go through the MLX
    /// continuous-batching scheduler when `LUMEN_MLX_BATCH_DECODE` is on;
    /// everything else is sequential.
    ///
    /// `BATCHED_ENGINE=1` used to select a Candle continuous-batching
    /// scheduler for the GGUF and Qwen3.5-MoE backends. Those backends are
    /// gone, and so is the variable — MLX batching has its own switch.
    pub async fn run(mut self, mut rx: mpsc::Receiver<EngineRequest>) {
        // Wraps the per-feature gate below — `mlx_feature_on` lets a single
        // `LUMEN_MLX_SERVER_MODE=1` switch enable the whole multi-request path
        // (batched decode + shared-prefix dedup), while a per-feature var still
        // overrides it (e.g. `LUMEN_MLX_BATCH_DECODE=0` opts one part back out).
        // Phase 2: opt-in multi-seq scheduler for the MLX-native Qwen3.6 path.
        // Distinct from BATCHED_ENGINE (which gates the Candle paths). Default
        // OFF — the sequential path stays byte-identical until enabled. Computed
        // as a plain bool BEFORE the match so no `&MlxBackend` (which is not
        // `Send`, holding raw Metal pointers) is held across the `.await`.
        // Phase 3 follow-up: both MLX families route here. Gemma 4 emits a
        // reasoning channel even at thinking=false; the batched scheduler now
        // splits visible vs reasoning via `stream_channels` (same
        // `ResponseParser` the sequential path uses), so `delta.content` no
        // longer leaks `thought` content and Gemma 4 batched serving matches
        // the sequential stream.
        #[cfg(feature = "mlx-native")]
        let mlx_batched = mlx_feature_on("LUMEN_MLX_BATCH_DECODE")
            && matches!(&self.backend, ModelBackend::Mlx(_));
        #[cfg(feature = "mlx-native")]
        if mlx_batched {
            self.run_batched_mlx(&mut rx).await;
            return;
        }
        self.run_sequential(&mut rx).await;
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
    /// One decode step processes up to `LUMEN_MLX_BATCH_MAX` active seqs at
    /// once via `forward_batched_decode_v2`. Non-streaming requests are
    /// serviced between decode steps (they temporarily pause the batch).
    #[cfg(feature = "mlx-native")]
    fn mlx_batched_driver(&mut self) -> Option<&mut dyn lumen_mlx::MlxBatchedSeqDriver> {
        match &mut self.backend {
            ModelBackend::Mlx(b) => Some(b.batched_seq_driver_mut()),
        }
    }

    /// Admit one streaming chat request into the MLX batched scheduler, or serve
    /// it inline when it isn't batch-eligible. Shared by both admission sites in
    /// `run_batched_mlx` (the non-blocking `try_recv` loop AND the blocking
    /// empty-batch `recv`) so the FIRST request of a concurrent burst seeds the
    /// batch instead of being dispatched sequentially — which would block the
    /// loop and serialize the whole burst onto the sequential path.
    ///
    /// On success the seq is inserted into `active` (or finalized in place if its
    /// first token already hit EOS / the cap) and `next_seq_id` is bumped.
    /// Ineligible requests (tools, response_format, stop, non-zero temperature,
    /// or thinking enabled) fall through to `chat_completion_streaming`.
    #[cfg(feature = "mlx-native")]
    fn admit_streaming_mlx(
        &mut self,
        req: ChatCompletionRequest,
        token_tx: mpsc::Sender<StreamEvent>,
        active: &mut std::collections::HashMap<u64, ActiveSeqState>,
        next_seq_id: &mut u64,
    ) {
        let thinking =
            req.enable_thinking_with_backend_default(self.backend.is_reasoning_first_family());
        if !Self::mlx_batch_eligible(&req) || thinking {
            self.chat_completion_streaming(&req, &token_tx);
            return;
        }
        let sid = *next_seq_id;
        match self.start_streaming_seq_mlx(
            sid,
            &req.messages,
            req.max_tokens,
            thinking,
            token_tx.clone(),
        ) {
            Ok((seq, done)) => {
                if done {
                    let _ = seq.token_tx.try_send(StreamEvent::Done {
                        prompt_tokens: seq.prompt_tokens,
                        completion_tokens: seq.generated.len() as u32,
                        finish_reason: FinishReason::Stop,
                    });
                    if let Some(qb) = self.mlx_batched_driver() {
                        let _ = qb.remove_seq(sid);
                    }
                } else {
                    active.insert(sid, seq);
                }
                *next_seq_id += 1;
            }
            Err(e) => {
                let _ = token_tx.try_send(StreamEvent::Error(e.to_string()));
            }
        }
    }

    #[cfg(feature = "mlx-native")]
    async fn run_batched_mlx(&mut self, rx: &mut mpsc::Receiver<EngineRequest>) {
        use std::collections::HashMap;

        // `PAGED_MAX_BATCH` is the pre-v9 spelling, kept as a fallback so
        // existing launch scripts keep working. It never had anything to do
        // with PagedAttention — that crate was deleted without ever having read
        // it — so the name it is read under now says what it configures.
        let max_batch: usize = std::env::var("LUMEN_MLX_BATCH_MAX")
            .or_else(|_| std::env::var("PAGED_MAX_BATCH"))
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);

        let mut active: HashMap<u64, ActiveSeqState> = HashMap::new();
        let mut next_seq_id: u64 = 1;

        eprintln!("[mlx batched] active scheduler (max_batch={max_batch})");

        loop {
            // 1. Admit eligible streaming chat requests; route everything else
            //    (non-streaming, tools, response_format, sampled, thinking,
            //    Anthropic) to the sequential path so behavior is unchanged.
            while active.len() < max_batch {
                match rx.try_recv() {
                    Ok(EngineRequest::StreamingChatCompletion { req, token_tx }) => {
                        self.admit_streaming_mlx(req, token_tx, &mut active, &mut next_seq_id);
                    }
                    Ok(other) => {
                        self.dispatch_request_sequential(other);
                    }
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => return,
                }
            }

            // Nothing active: block for the next request rather than spin. A
            // streaming chat request is ADMITTED into the batch here (not
            // dispatched sequentially) so the first request of a concurrent
            // burst seeds the batch — the loop then re-enters the non-blocking
            // admission above and picks up the rest of the burst.
            if active.is_empty() {
                match rx.recv().await {
                    Some(EngineRequest::StreamingChatCompletion { req, token_tx }) => {
                        self.admit_streaming_mlx(req, token_tx, &mut active, &mut next_seq_id);
                        continue;
                    }
                    Some(other) => {
                        self.dispatch_request_sequential(other);
                        continue;
                    }
                    None => return,
                }
            }

            // 2. One greedy decode step across all active seqs.
            let ids: Vec<u64> = active.keys().copied().collect();
            let last_tokens: Vec<u32> = ids.iter().map(|id| active[id].last_token).collect();
            let positions: Vec<usize> = ids.iter().map(|id| active[id].position).collect();

            let t_step = std::time::Instant::now();
            let results = {
                let qb = match self.mlx_batched_driver() {
                    Some(q) => q,
                    None => {
                        eprintln!("[mlx batched] non-Qwen35 backend unreachable");
                        return;
                    }
                };
                qb.decode_step_batch(&ids, &last_tokens, &positions)
            };
            let results = match results {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("[mlx batched] decode_step_batch failed: {e}");
                    for (_id, seq) in active.drain() {
                        let _ = seq.token_tx.try_send(StreamEvent::Error(e.to_string()));
                    }
                    continue;
                }
            };

            // 3. Per-seq emit + EOS / max-tokens check.
            let mut to_remove: Vec<u64> = Vec::new();
            for (row, &id) in ids.iter().enumerate() {
                let (next_tok, new_pos) = results[row];
                let channels = self.mlx_batched_driver().and_then(|qb| {
                    let seq = active.get(&id)?;
                    let mut g = seq.generated.clone();
                    g.push(next_tok);
                    qb.stream_channels(&g).ok()
                });
                let seq = active.get_mut(&id).unwrap();
                seq.generated.push(next_tok);
                seq.last_token = next_tok;
                seq.position = new_pos;
                if let Some((visible, reasoning)) = channels {
                    emit_channel_delta(seq, visible, reasoning);
                }
                if seq.eos_tokens.contains(&next_tok) || seq.generated.len() >= seq.max_new {
                    let n_gen = seq.generated.len();
                    let _ = seq.token_tx.try_send(StreamEvent::Done {
                        prompt_tokens: seq.prompt_tokens,
                        completion_tokens: n_gen as u32,
                        finish_reason: FinishReason::Stop,
                    });
                    to_remove.push(id);
                }
            }

            let step_ms = t_step.elapsed().as_secs_f64() * 1000.0;
            let active_gb = lumen_mlx::metal_memory::get_active_memory().unwrap_or(0) as f64 / 1e9;
            eprintln!(
                "[mlx batched] step: N={} latency={:.1}ms agg={:.1} tok/s active={:.2}GB",
                ids.len(),
                step_ms,
                ids.len() as f64 / (step_ms / 1000.0),
                active_gb,
            );

            for id in to_remove {
                active.remove(&id);
                if let Some(qb) = self.mlx_batched_driver() {
                    let _ = qb.remove_seq(id);
                }
            }
        }
    }

    /// Eligibility for the greedy MLX batched scheduler: plain chat only.
    ///
    /// Anything ineligible falls back to the sequential path, which is the
    /// full-featured one. In particular **image requests are ineligible**: the
    /// batched driver prefills through `build_chat_input` + `prefill`, neither
    /// of which carries images, so admitting one would answer from the text
    /// alone — the silent drop this whole path is guarded against elsewhere.
    #[cfg(feature = "mlx-native")]
    fn mlx_batch_eligible(req: &ChatCompletionRequest) -> bool {
        req.temperature == 0.0
            && req.tools.as_ref().map(|t| t.is_empty()).unwrap_or(true)
            && req.response_format.is_none()
            && req.stop.is_none()
            && !req.messages.iter().any(|m| !m.images.is_empty())
    }

    /// Prefill a new MLX-native streaming sequence under `seq_id`. Unlike the
    /// Candle `start_streaming_seq_qwen35` (which warms `prompt[..len-1]` and
    /// enters the loop on the last prompt token), the MLX `prefill` consumes the
    /// FULL prompt and returns the FIRST GENERATED token — so that token is
    /// emitted here and seeded into `generated`; the decode loop continues from
    /// it. Returns `(state, done)` where `done` is true when the first token is
    /// already EOS / hits the cap (the caller finalizes without inserting).
    #[cfg(feature = "mlx-native")]
    fn start_streaming_seq_mlx(
        &mut self,
        seq_id: u64,
        messages: &[ChatMessage],
        max_tokens: usize,
        thinking: bool,
        token_tx: mpsc::Sender<StreamEvent>,
    ) -> Result<(ActiveSeqState, bool)> {
        let qb = self
            .mlx_batched_driver()
            .ok_or_else(|| anyhow::anyhow!("start_streaming_seq_mlx: not an MLX backend"))?;

        let msg_pairs: Vec<(String, String)> = messages
            .iter()
            .map(|m| (m.role.clone(), m.content.clone()))
            .collect();
        // `None`, and provably so rather than by omission: `admit_streaming_mlx`
        // routes to the non-batched path whenever `thinking` is true (see its
        // `|| thinking` guard), and Qwen 3.8's template emits the effort block
        // only when thinking is on. A batched sequence therefore never has an
        // effort to carry.
        let prompt_ids = qb.build_chat_input(&msg_pairs, thinking, None)?;
        let prompt_tokens = prompt_ids.len() as u32;
        let eos_tokens = qb.eos_tokens().to_vec();
        let max_new = if max_tokens == 0 { 256 } else { max_tokens };

        let t_prefill = std::time::Instant::now();
        let (first_tok, position) = qb
            .prefill(seq_id, &prompt_ids)
            .map_err(|e| anyhow::anyhow!("prefill seq {seq_id}: {e}"))?;
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[mlx batched] seq {seq_id} prefill: {} tokens in {prefill_ms:.0}ms ({:.1} tok/s)",
            prompt_ids.len(),
            prompt_ids.len() as f64 / (prefill_ms / 1000.0),
        );

        // first_tok is the first generated token: emit it (channel-aware) and
        // seed `generated`. Using the same `stream_channels` split as the decode
        // loop keeps Gemma 4's reasoning channel out of `delta.content` from
        // token 0 (e.g. when the very first token opens the `<|channel>` span).
        let generated = vec![first_tok];
        let (vis0, rea0) = qb.stream_channels(&generated).unwrap_or_default();
        let mut prev_text = String::new();
        let mut prev_reasoning = String::new();
        if !vis0.contains('\u{FFFD}') && !vis0.is_empty() {
            let _ = token_tx.try_send(StreamEvent::Delta(vis0.clone()));
            prev_text = vis0;
        }
        if !rea0.contains('\u{FFFD}') && !rea0.is_empty() {
            let _ = token_tx.try_send(StreamEvent::ReasoningDelta(rea0.clone()));
            prev_reasoning = rea0;
        }
        let done = eos_tokens.contains(&first_tok) || generated.len() >= max_new;

        let state = ActiveSeqState {
            token_tx,
            generated,
            max_new,
            last_token: first_tok,
            position,
            prev_text,
            prev_reasoning,
            prompt_tokens,
            eos_tokens,
        };
        Ok((state, done))
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
        let mut a = vec![tc("read")];
        remap_tool_call_names(&mut a, &tools);
        assert_eq!(a[0].name, "read");
        // Abbreviated MCP names → full namespaced names.
        let mut b = vec![tc("ctx_read")];
        remap_tool_call_names(&mut b, &tools);
        assert_eq!(b[0].name, "mcp__lean_ctx_ctx_read");
        let mut c = vec![tc("ctx_tree")];
        remap_tool_call_names(&mut c, &tools);
        assert_eq!(c[0].name, "mcp__lean_ctx_ctx_tree");
    }

    /// A name that resolves to nothing is DROPPED, not forwarded.
    ///
    /// Forwarding it is a contract violation — the client looks up a function
    /// it never declared. Measured on Qwen3.8-27B: `geget_weather` reached the
    /// response for a client that had declared only `get_weather`.
    #[test]
    fn an_unresolvable_name_is_dropped_not_forwarded() {
        let tools = [td("mcp__a_get"), td("mcp__b_get")];
        // Ambiguous suffix `get` matches two tools → cannot be resolved.
        let mut a = vec![tc("get")];
        remap_tool_call_names(&mut a, &tools);
        assert!(a.is_empty(), "ambiguous name must not reach the client");
        // No match at all.
        let mut b = vec![tc("nonexistent")];
        remap_tool_call_names(&mut b, &tools);
        assert!(b.is_empty(), "undeclared name must not reach the client");
        // Dropping is per call, not per turn: the valid ones survive, in order.
        let mut c = vec![tc("mcp__a_get"), tc("nonexistent"), tc("mcp__b_get")];
        remap_tool_call_names(&mut c, &tools);
        let kept: Vec<&str> = c.iter().map(|x| x.name.as_str()).collect();
        assert_eq!(kept, ["mcp__a_get", "mcp__b_get"]);
    }

    /// The `geget_weather` case: the model stuttered the opening bytes of a
    /// declared name once the one-call grammar released and the tail decoded
    /// unconstrained.
    #[test]
    fn resolves_a_stuttered_name() {
        let tools = [td("get_weather"), td("bash")];
        for emitted in ["geget_weather", "gget_weather", "get_get_weather"] {
            let mut a = vec![tc(emitted)];
            remap_tool_call_names(&mut a, &tools);
            assert_eq!(
                a.first().map(|c| c.name.as_str()),
                Some("get_weather"),
                "{emitted}"
            );
        }
    }

    /// The stutter repair must not turn a different word into a declared tool.
    ///
    /// The extra prefix has to be a prefix of the declared name — that is what
    /// makes it a stutter. `black` is not a prefix of `list`, so `blacklist`
    /// stays unresolved (and is therefore dropped, per the rule above).
    #[test]
    fn a_longer_word_is_not_a_stutter() {
        let tools = [td("list")];
        let mut a = vec![tc("blacklist")];
        remap_tool_call_names(&mut a, &tools);
        assert!(a.is_empty(), "blacklist is not a stutter of list");

        // `rrrun` against `run` + `rrun` is NOT ambiguous and must resolve:
        // `run` is rejected (`rr` is not a prefix of `run`), leaving `rrun`
        // alone. Pinned because it reads like a two-candidate case and is not —
        // the prefix guard already eliminated one.
        let tools = [td("run"), td("rrun")];
        let mut b = vec![tc("rrrun")];
        remap_tool_call_names(&mut b, &tools);
        assert_eq!(b.first().map(|c| c.name.as_str()), Some("rrun"));

        // Genuine ambiguity needs self-similar names, and then nothing is
        // guessed: `aaaa` is a one-`a` stutter of `aaa` and a two-`a` stutter
        // of `aa`, so it resolves to neither and is dropped.
        let tools = [td("aa"), td("aaa")];
        let mut c = vec![tc("aaaa")];
        remap_tool_call_names(&mut c, &tools);
        assert!(c.is_empty(), "two candidates must resolve to neither");
    }

    #[test]
    fn requires_separator_boundary() {
        // `read` must NOT be rewritten to `thread` — the suffix has no
        // separator before it. It is now dropped rather than forwarded, which
        // keeps the original point (never `thread`) and adds the new one.
        let tools = [td("thread")];
        let mut a = vec![tc("read")];
        remap_tool_call_names(&mut a, &tools);
        assert!(
            a.is_empty(),
            "read must not become thread, and must not reach the client either"
        );
    }
}

/// Per-sequence state carried by the MLX batched decode loop.
///
/// Several fields (`seq_id`, `prompt_len`, `decode_start`, `temperature`,
/// `top_p`, `repeat_penalty`, `pending_emit`, `prefill_remaining`) were read
/// only by the Candle continuous-batching scheduler and went with it. The MLX
/// scheduler admits greedy requests only — anything with a non-zero
/// temperature, tools, `response_format` or stop strings is routed to the
/// sequential path — so the sampling fields were constants it never consulted.
pub(crate) struct ActiveSeqState {
    pub token_tx: mpsc::Sender<StreamEvent>,
    pub generated: Vec<u32>,
    pub max_new: usize,
    pub last_token: u32,
    pub position: usize,
    pub prev_text: String,
    /// Channel-aware streaming: cumulative reasoning-channel text already
    /// emitted as `ReasoningDelta`. Mirrors `prev_text` for the visible
    /// channel.
    pub prev_reasoning: String,
    pub prompt_tokens: u32,
    /// EOS token IDs used by the batched decode loop for this sequence.
    pub eos_tokens: Vec<u32>,
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
/// Channel-aware incremental emit for the MLX batched scheduler. Diffs the
/// freshly-decoded `(visible, reasoning)` channel strings against the seq's
/// cumulative `prev_text` / `prev_reasoning` and sends only the new tail of
/// each channel (`Delta` / `ReasoningDelta`). The replacement-char guard skips
/// emitting across an incomplete multi-byte UTF-8 boundary exactly like the
/// flat-decode path: the next token completes the char and the cumulative diff
/// recovers the held bytes, so no byte is dropped or duplicated.
#[cfg(feature = "mlx-native")]
fn emit_channel_delta(seq: &mut ActiveSeqState, visible: String, reasoning: String) {
    if !visible.contains('\u{FFFD}') && visible.len() > seq.prev_text.len() {
        let delta = visible[seq.prev_text.len()..].to_string();
        if !delta.is_empty() {
            let _ = seq.token_tx.try_send(StreamEvent::Delta(delta));
            seq.prev_text = visible;
        }
    }
    if !reasoning.contains('\u{FFFD}') && reasoning.len() > seq.prev_reasoning.len() {
        let delta = reasoning[seq.prev_reasoning.len()..].to_string();
        if !delta.is_empty() {
            let _ = seq.token_tx.try_send(StreamEvent::ReasoningDelta(delta));
            seq.prev_reasoning = reasoning;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Phase 1.4 helpers — unit tests
//
// Pure functions only; full integration (request → backend dispatch →
// response) is exercised via manual smoke tests against a loaded model.
// ─────────────────────────────────────────────────────────────────────────

/// Images and a tool-calling history need different renderers, and the
/// structured one wins the branch. Without an explicit refusal a request
/// carrying both would answer from the text alone — the silent-drop failure
/// this whole change set exists to remove.
/// The batched scheduler's driver prefills through `build_chat_input` +
/// `prefill`, neither of which carries images. Admitting an image request there
/// would answer from the text alone, so eligibility has to exclude it — the
/// sequential fallback is the path that knows about images.
#[cfg(all(test, feature = "mlx-native"))]
mod batch_eligibility {
    use super::*;
    use serde_json::json;

    const TINY_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

    fn req(content: serde_json::Value) -> ChatCompletionRequest {
        serde_json::from_value(json!({
            "model": "m",
            "messages": [{ "role": "user", "content": content }],
            "temperature": 0.0,
        }))
        .expect("parse request")
    }

    #[test]
    fn plain_greedy_chat_is_eligible() {
        assert!(InferenceEngine::mlx_batch_eligible(&req(json!("hello"))));
    }

    #[test]
    fn image_requests_are_not_eligible() {
        let url = format!("data:image/png;base64,{TINY_PNG_B64}");
        let r = req(json!([
            { "type": "text", "text": "what is this?" },
            { "type": "image_url", "image_url": { "url": url } },
        ]));
        assert!(
            !InferenceEngine::mlx_batch_eligible(&r),
            "an image request must fall back to the sequential path"
        );
    }
}

/// Alignment of the Anthropic structured path's per-turn image vector.
///
/// This is the one place in the image plumbing with no runtime error signal to
/// fall back on: a misaligned row does not fail, it splices one image's pixels
/// onto another turn's placeholder rows and the model answers confidently about
/// the wrong picture. So the expansion is pinned directly.
#[cfg(test)]
mod tool_choice_none_withholds_tools {
    use super::{ResolvedToolChoice, ToolDef, tools_visible_to_model};

    fn one_tool() -> Vec<ToolDef<'static>> {
        vec![ToolDef {
            name: "get_weather",
            description: None,
            parameters: None,
            response: None,
        }]
    }

    #[test]
    fn none_hides_them() {
        assert!(
            tools_visible_to_model(one_tool(), &ResolvedToolChoice::None).is_empty(),
            "a model cannot call what it was never shown"
        );
    }

    #[test]
    fn every_other_choice_keeps_them() {
        for choice in [
            ResolvedToolChoice::Auto,
            ResolvedToolChoice::Required,
            ResolvedToolChoice::Tool("get_weather"),
        ] {
            assert_eq!(
                tools_visible_to_model(one_tool(), &choice).len(),
                1,
                "{choice:?} must still see the tools"
            );
        }
    }
}

#[cfg(test)]
mod anthropic_turn_image_alignment {
    use super::anthropic_turn_images;
    use crate::types::{AnthropicContent, AnthropicMessage};

    fn msg(role: &str) -> AnthropicMessage {
        AnthropicMessage {
            role: role.to_string(),
            content: AnthropicContent::Text(String::new()),
        }
    }

    fn img(tag: u8) -> Vec<Vec<u8>> {
        vec![vec![tag]]
    }

    #[test]
    fn tool_results_expand_one_message_into_several_turns() {
        // user(2 tool_results + text + image) → Tool, Tool, User(image).
        // A message-indexed vector would have put the image on the first Tool
        // turn, two rows early.
        let messages = vec![msg("assistant"), msg("user")];
        let rows = anthropic_turn_images(
            &messages,
            false,
            &[Vec::new(), img(7)],
            &[0, 2],
            &[false, true],
        )
        .expect("build rows");
        assert_eq!(rows.len(), 4, "1 assistant + 2 tool + 1 user turn");
        assert!(rows[0].is_empty(), "assistant turn");
        assert!(rows[1].is_empty(), "first tool_result turn");
        assert!(rows[2].is_empty(), "second tool_result turn");
        assert_eq!(rows[3], img(7), "the image belongs to the user turn");
    }

    #[test]
    fn a_system_turn_shifts_every_row() {
        let messages = vec![msg("user")];
        let rows =
            anthropic_turn_images(&messages, true, &[img(1)], &[0], &[true]).expect("build rows");
        assert_eq!(rows.len(), 2);
        assert!(rows[0].is_empty(), "system turn carries no image");
        assert_eq!(rows[1], img(1));
    }

    #[test]
    fn an_image_only_message_still_gets_a_turn() {
        // No text, so the turn builder would previously have emitted nothing —
        // and the image would have had no turn to attach to.
        let messages = vec![msg("user")];
        let rows =
            anthropic_turn_images(&messages, false, &[img(3)], &[0], &[false]).expect("build rows");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0], img(3));
    }

    #[test]
    fn a_textless_imageless_message_emits_no_turn() {
        // A user message that is nothing but tool_results contributes its Tool
        // turns and no User turn.
        let messages = vec![msg("user")];
        let rows = anthropic_turn_images(&messages, false, &[Vec::new()], &[1], &[false])
            .expect("build rows");
        assert_eq!(rows.len(), 1, "the tool turn only");
        assert!(rows[0].is_empty());
    }

    #[test]
    fn an_image_on_an_assistant_turn_is_refused() {
        let messages = vec![msg("assistant")];
        let err = anthropic_turn_images(&messages, false, &[img(9)], &[0], &[false])
            .expect_err("an assistant turn has nowhere to put an image");
        assert!(err.to_string().contains("user message"), "{err}");
    }
}

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
}
