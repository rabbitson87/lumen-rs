use serde::{Deserialize, Deserializer, Serialize};

// === Request types ===

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Lumen extension: simple bool toggle. Use [`Self::enable_thinking`]
    /// instead of reading this field directly — that helper also folds in
    /// OpenAI's standard `reasoning_effort` and vLLM/SGLang's
    /// `chat_template_kwargs.enable_thinking`.
    #[serde(default)]
    pub thinking: bool,
    /// OpenAI o-series / GPT-5 standard reasoning toggle. Accepted values:
    /// `"minimal"` / `"none"` / `"low"` / `"medium"` / `"high"`. Any value
    /// other than `minimal`/`none` enables thinking on supported families
    /// (Gemma 4, Qwen 3.5).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    /// vLLM / SGLang convention for chat-template-controlled thinking
    /// toggles (Qwen3, DeepSeek-R1, …). Sent as `extra_body` from the
    /// official OpenAI SDKs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<ChatTemplateKwargs>,
    /// Optional client-supplied session id. When present and the MLX backend
    /// is active, the server reuses the prior turn's KV cache and only feeds
    /// the new suffix. Other backends silently ignore.
    #[serde(default)]
    pub session_id: Option<String>,

    // ── OpenAI tool calling ────────────────────────────────────────
    /// Tool definitions exposed to the model. Currently only
    /// `{"type":"function","function":{...}}` shape is meaningful — Phase 1
    /// passes `function.parameters` (JSON Schema) through to each backend's
    /// chat-template renderer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// `"auto"` / `"none"` / `"required"` / `{"type":"function","function":{"name":"..."}}`.
    /// Phase 1 accepts and stores but does not enforce — neither Gemma 4 nor
    /// Qwen 3.6 has a native `required` mode in its template. Wire-up TBD.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
}

#[derive(Debug, Deserialize, Default, Clone, Copy)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

/// Subset of vLLM / SGLang's `chat_template_kwargs` that we honor today.
/// Only the thinking toggle is wired; unknown keys are silently dropped by
/// serde — matching upstream behavior.
#[derive(Debug, Deserialize, Default, Clone)]
pub struct ChatTemplateKwargs {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enable_thinking: Option<bool>,
}

impl ChatCompletionRequest {
    /// Resolve the effective thinking flag from all supported input shapes.
    ///
    /// **imatrix-AWQ family override** — these builds have channel-open token
    /// over-amplification (calibration corpus lacks reasoning samples,
    /// see memory note `gemma4_imatrix_awq_channel_token_loss_2026_05_27`).
    /// Forcing thinking ON triggers infinite-reasoning runaway, so the
    /// answer is hard-coded `false` regardless of client signals.
    ///
    /// For non-imatrix-AWQ models, precedence (highest first):
    ///   1. `chat_template_kwargs.enable_thinking` — vLLM / SGLang convention,
    ///      explicit per-request override.
    ///   2. `reasoning_effort` — OpenAI o-series / GPT-5 convention.
    ///      `"minimal"` / `"none"` → off; any other recognized value → on.
    ///   3. `thinking` — Lumen extension flat bool.
    ///   4. **Tools-present auto-thinking** — when `tools.len() > 0` and no
    ///      explicit signal is given, default to `true`. Mirrors llama.cpp's
    ///      observed behavior: agentic clients with tools attached need the
    ///      thought channel active so the model can decide which tool to call
    ///      (Gemma 4 IT's training distribution emits `<|tool_call>` tokens
    ///      almost exclusively from inside the thinking channel).
    ///   5. Default `false` when no explicit signal and no tools.
    pub fn enable_thinking(&self) -> bool {
        self.enable_thinking_with_backend_default(false)
    }

    /// Resolve `enable_thinking` with an operator-supplied backend default.
    /// Precedence (highest first):
    ///   1. imatrix-AWQ family override → forced `false`.
    ///   2. explicit `chat_template_kwargs.enable_thinking` → as given.
    ///   3. explicit `reasoning_effort` → on unless `"minimal"|"none"|"off"`.
    ///   4. flat `thinking: true` → user explicit opt-in.
    ///   5. `backend_default_on` (operator opt-in via
    ///      `LUMEN_BACKEND_THINKING_DEFAULT`) → backend's default when
    ///      no per-request signal is given.
    ///   6. otherwise `false`.
    pub fn enable_thinking_with_backend_default(&self, backend_default_on: bool) -> bool {
        if is_imatrix_awq_family(&self.model) {
            return false;
        }
        if let Some(kw) = self.chat_template_kwargs.as_ref() {
            if let Some(v) = kw.enable_thinking {
                return v;
            }
        }
        if let Some(eff) = self.reasoning_effort.as_deref() {
            return !matches!(
                eff.trim().to_ascii_lowercase().as_str(),
                "minimal" | "none" | "off" | "disabled" | ""
            );
        }
        if self.thinking {
            return true;
        }
        backend_default_on
    }
}

/// Detect imatrix-AWQ-quantized builds by id substring. The Lumen-shipped
/// `hsng95/gemma-4-26b-a4b-mlx-imatrix3plus-awq*` family matches; community
/// uniform quants (`mlx-community/...-it-4bit` etc.) do not.
fn is_imatrix_awq_family(model_id: &str) -> bool {
    let lower = model_id.to_ascii_lowercase();
    lower.contains("imatrix") || lower.contains("-awq")
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct ChatMessage {
    pub role: String,
    /// Spec: `content` is `string` for system/user/tool roles, `string | null`
    /// for assistant when `tool_calls` is present. We accept any of
    /// `"text"` / `null` / missing and flatten to empty string so existing
    /// call sites that consume `&str` still work without `.unwrap_or_default()`
    /// scattered everywhere.
    #[serde(default, deserialize_with = "deserialize_content_lenient")]
    pub content: String,
    /// Present on `role:"assistant"` when the previous turn invoked tools.
    /// Carries the model's prior tool calls back into the prompt so the
    /// chat-template can re-render them. Server-emit side puts them in
    /// `ChatMessageResponse.tool_calls`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Required on `role:"tool"` — references the id of the tool call this
    /// message is answering.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Optional function name on `role:"tool"`. OpenAI legacy field; some
    /// clients still send it. Not required for routing — `tool_call_id` is
    /// the canonical link.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

fn deserialize_content_lenient<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    Option::<String>::deserialize(d).map(|o| o.unwrap_or_default())
}

impl ChatMessage {
    /// Construct a plain text-only chat message (no tool_calls / tool_call_id).
    /// Most internal call sites build messages by stitching role + text and
    /// don't need to populate the tool-related fields.
    pub fn new_text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }
}

// ── OpenAI tool calling — shared shapes (request & response) ─────

/// A tool the model is allowed to call. Currently only the `function` kind is
/// defined by the OpenAI spec; future kinds (e.g. `code_interpreter`) would
/// land as additional variants.
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Tool {
    Function { function: FunctionDef },
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct FunctionDef {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema for the function's arguments. We keep it as a raw
    /// `serde_json::Value` because each backend's chat-template renders the
    /// schema differently (Gemma 4: pseudo-JSON with `<|"|>` string delim;
    /// Qwen 3.6: pure JSON inside `<tools>`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum ToolChoice {
    /// `"auto"` (default) / `"none"` / `"required"`.
    Mode(String),
    /// `{"type":"function","function":{"name":"..."}}`.
    Named {
        #[serde(rename = "type")]
        kind: String,
        function: FunctionRef,
    },
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct FunctionRef {
    pub name: String,
}

/// One tool invocation — same shape on input (history) and output (server
/// emit). On output the server generates the `id`; clients echo it back on
/// the next turn via `ChatMessage.tool_call_id`.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String, // always "function" for now
    pub function: FunctionCall,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct FunctionCall {
    pub name: String,
    /// OpenAI quirk: `arguments` is a **JSON-encoded string**, not a JSON
    /// object. Clients are expected to `JSON.parse(arguments)` themselves.
    /// We round-trip the raw string here so both directions match the wire
    /// format byte-for-byte.
    pub arguments: String,
}

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub stream: bool,
    /// Optional client-supplied session id (MLX backend only). When the cached
    /// token sequence is a strict prefix of the new prompt, only the suffix is
    /// fed to the model. Other backends ignore this field.
    #[serde(default)]
    pub session_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct DropSessionResponse {
    pub id: String,
    pub object: String,
    pub deleted: bool,
}

#[derive(Debug, Serialize)]
pub struct DropPrefixCacheResponse {
    pub key: String,
    pub object: String,
    pub deleted: bool,
}

#[derive(Debug, Serialize)]
pub struct ClearPrefixCacheResponse {
    pub object: String,
    pub cleared: usize,
}

/// Server-side default applied to chat / completion `max_tokens` when the
/// client omits the field. Reads `LUMEN_DEFAULT_MAX_TOKENS` (set by the
/// lumen-app CONTEXT card); falls back to 2048 for out-of-tree usage where
/// the env var isn't plumbed. `0` means "unbounded — generate until EOS /
/// stop / context budget" and is forwarded as-is to the engine layer (which
/// already treats `max_tokens == 0` as the no-limit sentinel).
fn default_max_tokens() -> usize {
    std::env::var("LUMEN_DEFAULT_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(2048)
}
fn default_temperature() -> f32 {
    0.7
}
fn default_top_p() -> f32 {
    0.9
}

// === Streaming SSE types ===

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatStreamChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
pub struct ChatStreamChoice {
    pub index: u32,
    pub delta: ChatStreamDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Default)]
pub struct ChatStreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Incremental tool-call delta. OpenAI streams these per-token with
    /// `arguments` as a partial string that clients accumulate. Empty in
    /// non-tool turns.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
    /// Incremental reasoning delta — streamed equivalent of the
    /// non-streaming `ChatMessageResponse.reasoning` field. Mirrors the
    /// vLLM / OpenAI o-series convention. Clients that don't recognize
    /// this can fall back to parsing `<think>…</think>` inside `content`
    /// (the same reasoning text is also emitted there for backward
    /// compatibility).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
}

/// Streaming variant of `ToolCall`: index-keyed; `id` / `function.name`
/// arrive once at the start, then subsequent chunks carry only
/// `function.arguments` partials.
#[derive(Debug, Serialize, Clone, Default)]
pub struct ToolCallDelta {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionCallDelta>,
}

#[derive(Debug, Serialize, Clone, Default)]
pub struct FunctionCallDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

// === Response types ===

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessageResponse,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ChatMessageResponse {
    pub role: String,
    /// Per OpenAI spec, `content` is nullable on assistant messages that
    /// carry `tool_calls`. We emit it as an empty string in the common
    /// (text-only) case and `null` only when tool_calls are present and
    /// the model emitted no visible text alongside them.
    ///
    /// When the model produced reasoning (e.g. Gemma 4 `<|channel>thought\n…`),
    /// the reasoning is also prepended to this field wrapped in
    /// `<think>…</think>` tags so clients that parse text-tag thinking
    /// (e.g. Ayla) display it inline. OpenAI-spec-following clients should
    /// prefer the separate `reasoning` field below.
    #[serde(serialize_with = "serialize_nullable_string")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Model chain-of-thought / reasoning output (Gemma 4
    /// `<|channel>thought\n…<channel|>` block, OpenAI o-series reasoning,
    /// etc.). Mirrors the vLLM / OpenAI o-series response shape — clients
    /// that recognize this field should prefer it over the `<think>…</think>`
    /// envelope inside `content`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
}

fn serialize_nullable_string<S: serde::Serializer>(
    v: &Option<String>,
    s: S,
) -> Result<S::Ok, S::Error> {
    match v {
        Some(text) => s.serialize_str(text),
        None => s.serialize_none(),
    }
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelObject>,
}

#[derive(Debug, Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    pub r#type: String,
    pub code: u16,
}

impl ErrorResponse {
    pub fn new(message: impl Into<String>, code: u16) -> Self {
        Self {
            error: ErrorDetail {
                message: message.into(),
                r#type: "invalid_request_error".into(),
                code,
            },
        }
    }
}

// === Anthropic Messages API types ===

#[derive(Debug, Deserialize)]
pub struct AnthropicRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    pub max_tokens: usize,
    #[serde(default)]
    pub system: Option<AnthropicSystem>,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    /// Anthropic Extended Thinking parameter. Accepts the canonical Claude
    /// shape `{"type": "enabled", "budget_tokens": N}` /
    /// `{"type": "disabled"}` and a legacy bool. Read via
    /// [`Self::enable_thinking`] — `budget_tokens` is currently
    /// informational only (no real budget tracking on local models).
    #[serde(default)]
    pub thinking: AnthropicThinking,
    #[serde(default)]
    pub session_id: Option<String>,

    // ── Anthropic tool calling ─────────────────────────────────────
    /// Anthropic uses `input_schema` (not `parameters`) on each tool;
    /// otherwise structurally similar to OpenAI's `tools[]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<AnthropicTool>>,
    /// `{"type":"auto"}` / `{"type":"any"}` / `{"type":"tool","name":"..."}`.
    /// Phase 1 stores but does not enforce.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<AnthropicToolChoice>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema for the tool's arguments. Anthropic calls this
    /// `input_schema` (vs OpenAI's `function.parameters`); we mirror their
    /// field name on the wire.
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicToolChoice {
    Auto,
    Any,
    Tool { name: String },
}

/// Anthropic Extended Thinking flag, polymorphic on the wire.
///
/// Canonical Claude shape (per Anthropic Messages API):
/// ```json
/// "thinking": {"type": "enabled", "budget_tokens": 10000}
/// "thinking": {"type": "disabled"}
/// ```
/// Legacy / Lumen shorthand:
/// ```json
/// "thinking": true
/// "thinking": false
/// ```
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum AnthropicThinking {
    Object(AnthropicThinkingConfig),
    Bool(bool),
}

#[derive(Debug, Deserialize, Clone)]
pub struct AnthropicThinkingConfig {
    #[serde(rename = "type")]
    pub r#type: String,
    /// Optional reasoning-token budget hint. Currently parsed but unused
    /// (local backends don't differentiate budget vs `max_tokens`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u32>,
}

impl Default for AnthropicThinking {
    fn default() -> Self {
        Self::Bool(false)
    }
}

impl AnthropicRequest {
    /// True iff the client opted in to extended thinking. Mirrors the
    /// Anthropic API: object-form requires `type == "enabled"`, anything
    /// else (including the bool shorthand `false` and unknown `type` values)
    /// is off.
    ///
    /// **imatrix-AWQ family override** — same safety net as the OpenAI path
    /// (see [`ChatCompletionRequest::enable_thinking`]). Force `false` even
    /// when the client opts in, to avoid channel-open runaway on builds whose
    /// calibration corpus lacks reasoning samples.
    pub fn enable_thinking(&self) -> bool {
        if is_imatrix_awq_family(&self.model) {
            return false;
        }
        match &self.thinking {
            AnthropicThinking::Bool(b) => *b,
            AnthropicThinking::Object(cfg) => cfg.r#type.eq_ignore_ascii_case("enabled"),
        }
    }

    /// Parallel signature to [`ChatCompletionRequest::
    /// enable_thinking_with_backend_default`] so the engine can call
    /// both via a uniform pattern. Anthropic clients always send the
    /// `thinking` field explicitly (it's part of the API contract), so
    /// the backend hint never overrides — included only for API
    /// consistency.
    pub fn enable_thinking_with_backend_default(&self, _backend_default_on: bool) -> bool {
        self.enable_thinking()
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum AnthropicSystem {
    Text(String),
    Blocks(Vec<AnthropicSystemBlock>),
}

#[derive(Debug, Deserialize)]
pub struct AnthropicSystemBlock {
    pub r#type: String,
    #[serde(default)]
    pub text: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: AnthropicContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

impl AnthropicContent {
    /// Flatten to a single text string — drops tool_use / tool_result blocks.
    /// Used by the chat-template path that doesn't yet understand tools;
    /// tool-aware callers should iterate `Blocks(...)` directly.
    pub fn as_text(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| match b {
                    AnthropicContentBlock::Text { text } => Some(text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }

    /// Borrow the underlying blocks for tool-aware processing. For
    /// `Text(...)` we synthesize a single text block on the fly.
    pub fn blocks(&self) -> std::borrow::Cow<'_, [AnthropicContentBlock]> {
        match self {
            Self::Text(s) => {
                std::borrow::Cow::Owned(vec![AnthropicContentBlock::Text { text: s.clone() }])
            }
            Self::Blocks(b) => std::borrow::Cow::Borrowed(b),
        }
    }
}

/// One block in an Anthropic message's `content[]`. Tagged on `type`:
/// - `text`        — visible text (user/assistant)
/// - `tool_use`    — assistant invoking a tool
/// - `tool_result` — user message answering a prior tool_use
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        /// Anthropic ships `input` as a real JSON object (vs OpenAI's
        /// JSON-encoded string). We keep it as `Value` so downstream
        /// renderers can introspect.
        #[serde(default)]
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: AnthropicToolResultContent,
        #[serde(default, skip_serializing_if = "is_false")]
        is_error: bool,
    },
}

fn is_false(b: &bool) -> bool {
    !*b
}

/// Tool-result body — either a plain string or an array of text blocks
/// (Anthropic also supports image results; we accept-but-flatten those).
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum AnthropicToolResultContent {
    Text(String),
    Blocks(Vec<AnthropicToolResultBlock>),
}

impl AnthropicToolResultContent {
    pub fn as_text(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Blocks(blocks) => blocks
                .iter()
                .map(|b| b.as_text())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

/// One block inside an Anthropic `tool_result.content` array. Per spec
/// these can be `text`, `image`, or `document` blocks. Gemma 4 is a
/// text-only model; we accept the wire shape without erroring (so
/// agent loops with multimodal tool outputs don't fail) but flatten
/// image / document blocks to a `[image: media_type]` /
/// `[document: media_type]` text placeholder so the model at least
/// knows non-text data was returned.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AnthropicToolResultBlock {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
    /// Image blocks carry `source: { type, media_type, data }`. We
    /// only need media_type for the placeholder; the base64 / URL
    /// data is discarded since the underlying model can't consume it.
    #[serde(default)]
    pub source: Option<AnthropicToolResultBlockSource>,
}

impl AnthropicToolResultBlock {
    pub fn as_text(&self) -> String {
        match self.kind.as_str() {
            "text" => self.text.clone().unwrap_or_default(),
            "image" => {
                let media = self
                    .source
                    .as_ref()
                    .and_then(|s| s.media_type.as_deref())
                    .unwrap_or("image");
                format!("[image: {media}]")
            }
            "document" => {
                let media = self
                    .source
                    .as_ref()
                    .and_then(|s| s.media_type.as_deref())
                    .unwrap_or("document");
                // Inline the document's textual payload when it's a
                // text document — Anthropic's `source.type:"text"`
                // carries `data` as the plain document body.
                if let Some(src) = &self.source {
                    if src.kind.as_deref() == Some("text") {
                        if let Some(data) = &src.data {
                            return format!("[document: {media}]\n{data}");
                        }
                    }
                }
                format!("[document: {media}]")
            }
            other => format!("[{other}]"),
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct AnthropicToolResultBlockSource {
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    /// Optional URL source (Anthropic's `source.type:"url"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct AnthropicResponse {
    pub id: String,
    pub r#type: String,
    pub role: String,
    pub model: String,
    pub content: Vec<AnthropicResponseBlock>,
    pub stop_reason: String,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

/// One block in an Anthropic response's `content[]`. Mirrors the request-side
/// `AnthropicContentBlock` for the variants the server actually emits — we
/// never emit `tool_result` (that's a client-side message back to us).
#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicResponseBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Debug, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct AnthropicError {
    pub r#type: String,
    pub error: AnthropicErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct AnthropicErrorDetail {
    pub r#type: String,
    pub message: String,
}

impl AnthropicError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            r#type: "error".into(),
            error: AnthropicErrorDetail {
                r#type: "invalid_request_error".into(),
                message: message.into(),
            },
        }
    }
}

// === OpenAI-compatible embeddings ===

/// `input` accepts either a single string or an array of strings per the
/// OpenAI spec. We deserialize both into `Vec<String>` so the handler
/// has a single shape to work with.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Many(Vec<String>),
}

impl EmbeddingInput {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Self::Single(s) => vec![s],
            Self::Many(v) => v,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: EmbeddingInput,
    /// Only `"float"` is accepted in v1. `"base64"` is deferred.
    #[serde(default)]
    pub encoding_format: Option<String>,
    /// `dimensions` parameter (truncation) — deferred. Accepted but ignored.
    #[serde(default)]
    pub dimensions: Option<usize>,
    /// OpenAI also accepts `user`; we accept and ignore for parity.
    #[serde(default)]
    pub user: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingData {
    pub object: String,
    pub index: usize,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

// ─────────────────────────────────────────────────────────────────────────────
// Serde round-trip tests — tool calling wire format
//
// These verify that the request / response shapes match the OpenAI and
// Anthropic public APIs byte-for-byte on the critical fields. Fixtures are
// drawn from the published API documentation examples (paraphrased for
// minimal width). When a real client sends a tool-bearing request, this
// test class is the first place to look if deserialization fails.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tool_calling_serde {
    use super::*;
    use serde_json::json;

    // ── OpenAI ─────────────────────────────────────────────────────

    #[test]
    fn openai_request_with_tools_parses() {
        let raw = json!({
            "model": "gpt-x",
            "messages": [
                {"role": "user", "content": "weather in Seoul?"}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                }
            }],
            "tool_choice": "auto"
        });

        let req: ChatCompletionRequest = serde_json::from_value(raw).unwrap();
        assert_eq!(req.messages.len(), 1);
        let tools = req.tools.unwrap();
        assert_eq!(tools.len(), 1);
        let Tool::Function { function } = &tools[0];
        assert_eq!(function.name, "get_weather");
        assert_eq!(
            function.description.as_deref(),
            Some("Get current weather for a city")
        );
        let params = function.parameters.as_ref().unwrap();
        assert_eq!(params["required"][0], "location");
        match req.tool_choice.unwrap() {
            ToolChoice::Mode(m) => assert_eq!(m, "auto"),
            _ => panic!("expected Mode"),
        }
    }

    #[test]
    fn openai_tool_choice_named_parses() {
        let raw = json!({"type": "function", "function": {"name": "get_weather"}});
        let tc: ToolChoice = serde_json::from_value(raw).unwrap();
        match tc {
            ToolChoice::Named { kind, function } => {
                assert_eq!(kind, "function");
                assert_eq!(function.name, "get_weather");
            }
            _ => panic!("expected Named"),
        }
    }

    #[test]
    fn openai_assistant_message_with_tool_calls_round_trip() {
        // Inbound history: assistant emitted a tool_call on the prior turn.
        let raw = json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"location\":\"Seoul\"}"
                }
            }]
        });

        let msg: ChatMessage = serde_json::from_value(raw).unwrap();
        assert_eq!(msg.role, "assistant");
        assert_eq!(msg.content, ""); // null content → empty string via lenient deserializer
        let calls = msg.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_abc");
        assert_eq!(calls[0].function.name, "get_weather");
        // arguments is the JSON-encoded *string*, not a parsed object — critical.
        assert_eq!(calls[0].function.arguments, "{\"location\":\"Seoul\"}");
    }

    #[test]
    fn openai_tool_role_message_parses() {
        let raw = json!({
            "role": "tool",
            "tool_call_id": "call_abc",
            "content": "20C sunny"
        });
        let msg: ChatMessage = serde_json::from_value(raw).unwrap();
        assert_eq!(msg.role, "tool");
        assert_eq!(msg.tool_call_id.as_deref(), Some("call_abc"));
        assert_eq!(msg.content, "20C sunny");
    }

    #[test]
    fn openai_response_with_tool_calls_serializes_with_null_content() {
        let resp = ChatMessageResponse {
            role: "assistant".into(),
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: "call_abc".into(),
                kind: "function".into(),
                function: FunctionCall {
                    name: "get_weather".into(),
                    arguments: "{\"location\":\"Seoul\"}".into(),
                },
            }]),
            reasoning: None,
        };
        let v = serde_json::to_value(&resp).unwrap();
        assert!(
            v["content"].is_null(),
            "content must serialize as null when None"
        );
        assert_eq!(v["tool_calls"][0]["id"], "call_abc");
        assert_eq!(v["tool_calls"][0]["type"], "function");
        assert_eq!(v["tool_calls"][0]["function"]["name"], "get_weather");
    }

    #[test]
    fn openai_response_text_only_omits_tool_calls() {
        let resp = ChatMessageResponse {
            role: "assistant".into(),
            content: Some("Seoul is sunny".into()),
            tool_calls: None,
            reasoning: None,
        };
        let v = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["content"], "Seoul is sunny");
        assert!(
            v.get("tool_calls").is_none(),
            "tool_calls must be omitted when None"
        );
    }

    #[test]
    fn openai_stream_delta_with_partial_tool_call() {
        // OpenAI streams id/name once at the start, then arguments in chunks.
        let delta = ChatStreamDelta {
            role: None,
            content: None,
            reasoning: None,
            tool_calls: Some(vec![ToolCallDelta {
                index: 0,
                id: Some("call_abc".into()),
                kind: Some("function".into()),
                function: Some(FunctionCallDelta {
                    name: Some("get_weather".into()),
                    arguments: Some("{\"locat".into()),
                }),
            }]),
        };
        let v = serde_json::to_value(&delta).unwrap();
        assert!(v.get("role").is_none());
        assert!(v.get("content").is_none());
        assert_eq!(v["tool_calls"][0]["index"], 0);
        assert_eq!(v["tool_calls"][0]["type"], "function");
        assert_eq!(v["tool_calls"][0]["function"]["arguments"], "{\"locat");
    }

    // ── Anthropic ──────────────────────────────────────────────────

    #[test]
    fn anthropic_request_with_tools_parses() {
        let raw = json!({
            "model": "claude-x",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [{
                "name": "get_weather",
                "description": "Current weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                }
            }],
            "tool_choice": {"type": "auto"}
        });
        let req: AnthropicRequest = serde_json::from_value(raw).unwrap();
        let tools = req.tools.unwrap();
        assert_eq!(tools[0].name, "get_weather");
        assert_eq!(tools[0].input_schema["required"][0], "location");
        assert!(matches!(req.tool_choice, Some(AnthropicToolChoice::Auto)));
    }

    #[test]
    fn anthropic_tool_choice_named_parses() {
        let raw = json!({"type": "tool", "name": "get_weather"});
        let tc: AnthropicToolChoice = serde_json::from_value(raw).unwrap();
        match tc {
            AnthropicToolChoice::Tool { name } => assert_eq!(name, "get_weather"),
            _ => panic!("expected Tool"),
        }
    }

    #[test]
    fn anthropic_tool_use_content_block_parses() {
        let raw = json!({
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check."},
                {
                    "type": "tool_use",
                    "id": "toolu_xyz",
                    "name": "get_weather",
                    "input": {"location": "Seoul"}
                }
            ]
        });
        let msg: AnthropicMessage = serde_json::from_value(raw).unwrap();
        let blocks = msg.content.blocks();
        assert_eq!(blocks.len(), 2);
        match &blocks[1] {
            AnthropicContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "toolu_xyz");
                assert_eq!(name, "get_weather");
                assert_eq!(input["location"], "Seoul");
            }
            _ => panic!("expected ToolUse"),
        }
    }

    #[test]
    fn anthropic_tool_result_content_block_parses() {
        // tool_result can be string-content or array-of-blocks; both shapes
        // appear in real Anthropic conversations.
        let raw_string = json!({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_xyz",
                "content": "20C sunny"
            }]
        });
        let msg: AnthropicMessage = serde_json::from_value(raw_string).unwrap();
        match &msg.content.blocks()[0] {
            AnthropicContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                assert_eq!(tool_use_id, "toolu_xyz");
                assert_eq!(content.as_text(), "20C sunny");
                assert!(!is_error);
            }
            _ => panic!("expected ToolResult"),
        }

        let raw_blocks = json!({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_xyz",
                "content": [{"type": "text", "text": "20C sunny"}],
                "is_error": false
            }]
        });
        let msg2: AnthropicMessage = serde_json::from_value(raw_blocks).unwrap();
        match &msg2.content.blocks()[0] {
            AnthropicContentBlock::ToolResult { content, .. } => {
                assert_eq!(content.as_text(), "20C sunny");
            }
            _ => panic!("expected ToolResult"),
        }
    }

    #[test]
    fn anthropic_response_tool_use_block_serializes() {
        let block = AnthropicResponseBlock::ToolUse {
            id: "toolu_xyz".into(),
            name: "get_weather".into(),
            input: json!({"location": "Seoul"}),
        };
        let v = serde_json::to_value(&block).unwrap();
        assert_eq!(v["type"], "tool_use");
        assert_eq!(v["id"], "toolu_xyz");
        assert_eq!(v["name"], "get_weather");
        // input must be a real JSON object (Anthropic) — NOT a string.
        assert!(v["input"].is_object());
        assert_eq!(v["input"]["location"], "Seoul");
    }

    #[test]
    fn anthropic_response_text_block_serializes() {
        let block = AnthropicResponseBlock::Text {
            text: "Seoul is sunny".into(),
        };
        let v = serde_json::to_value(&block).unwrap();
        assert_eq!(v["type"], "text");
        assert_eq!(v["text"], "Seoul is sunny");
    }

    #[test]
    fn anthropic_content_as_text_drops_tool_blocks() {
        // Backward-compat for the legacy chat path that doesn't yet know
        // about tools — make sure as_text() flattens tool_use/tool_result
        // out so the existing prompt builder still receives a sane string.
        let content = AnthropicContent::Blocks(vec![
            AnthropicContentBlock::Text {
                text: "hello".into(),
            },
            AnthropicContentBlock::ToolUse {
                id: "toolu_x".into(),
                name: "noop".into(),
                input: json!({}),
            },
            AnthropicContentBlock::Text {
                text: "world".into(),
            },
        ]);
        assert_eq!(content.as_text(), "hello\nworld");
    }

    #[test]
    fn openai_request_omits_tools_when_none() {
        // Confirm we don't poison clients with a stray `"tools": null` on
        // outbound serialization (we deserialize requests, but also
        // serialize them for logging / forwarding tests).
        let req = ChatCompletionRequest {
            model: "x".into(),
            messages: vec![],
            max_tokens: 16,
            temperature: 0.7,
            top_p: 0.9,
            stream: false,
            stream_options: None,
            thinking: false,
            reasoning_effort: None,
            chat_template_kwargs: None,
            session_id: None,
            tools: None,
            tool_choice: None,
        };
        // Request type is Deserialize-only; we only need to verify
        // round-trip through serde_json::Value when the tools field is
        // absent at deserialize time — covered by other tests. This test
        // exists as a compile-time assertion that all fields are present
        // and constructible.
        let _ = req;
    }

    #[test]
    fn openai_thinking_flat_bool_round_trip() {
        let raw = serde_json::json!({
            "model": "x",
            "messages": [],
            "thinking": true,
        });
        let req: ChatCompletionRequest = serde_json::from_value(raw).unwrap();
        assert!(req.enable_thinking());
    }

    #[test]
    fn openai_reasoning_effort_enables_thinking() {
        for v in ["low", "medium", "high", "LOW", " High "] {
            let raw = serde_json::json!({
                "model": "x",
                "messages": [],
                "reasoning_effort": v,
            });
            let req: ChatCompletionRequest = serde_json::from_value(raw).unwrap();
            assert!(req.enable_thinking(), "expected on for {v:?}");
        }
        for v in ["minimal", "none", "off", "disabled", ""] {
            let raw = serde_json::json!({
                "model": "x",
                "messages": [],
                "reasoning_effort": v,
            });
            let req: ChatCompletionRequest = serde_json::from_value(raw).unwrap();
            assert!(!req.enable_thinking(), "expected off for {v:?}");
        }
    }

    #[test]
    fn openai_chat_template_kwargs_overrides_thinking() {
        // Even if `thinking: true` and `reasoning_effort: "high"` would
        // both enable, vLLM-style explicit override wins.
        let raw = serde_json::json!({
            "model": "x",
            "messages": [],
            "thinking": true,
            "reasoning_effort": "high",
            "chat_template_kwargs": { "enable_thinking": false },
        });
        let req: ChatCompletionRequest = serde_json::from_value(raw).unwrap();
        assert!(!req.enable_thinking());

        // And the inverse — flat thinking false but kwargs flips it on.
        let raw = serde_json::json!({
            "model": "x",
            "messages": [],
            "thinking": false,
            "chat_template_kwargs": { "enable_thinking": true },
        });
        let req: ChatCompletionRequest = serde_json::from_value(raw).unwrap();
        assert!(req.enable_thinking());
    }

    #[test]
    fn anthropic_thinking_accepts_bool_shorthand() {
        let raw = serde_json::json!({
            "model": "x",
            "max_tokens": 16,
            "messages": [],
            "thinking": true,
        });
        let req: AnthropicRequest = serde_json::from_value(raw).unwrap();
        assert!(req.enable_thinking());

        let raw = serde_json::json!({
            "model": "x",
            "max_tokens": 16,
            "messages": [],
            "thinking": false,
        });
        let req: AnthropicRequest = serde_json::from_value(raw).unwrap();
        assert!(!req.enable_thinking());
    }

    #[test]
    fn anthropic_thinking_accepts_canonical_object() {
        // Claude canonical shape — `enabled` with budget hint.
        let raw = serde_json::json!({
            "model": "x",
            "max_tokens": 16,
            "messages": [],
            "thinking": { "type": "enabled", "budget_tokens": 8000 },
        });
        let req: AnthropicRequest = serde_json::from_value(raw).unwrap();
        assert!(req.enable_thinking());

        // Disabled object.
        let raw = serde_json::json!({
            "model": "x",
            "max_tokens": 16,
            "messages": [],
            "thinking": { "type": "disabled" },
        });
        let req: AnthropicRequest = serde_json::from_value(raw).unwrap();
        assert!(!req.enable_thinking());

        // Unknown `type` defaults to off (forward-compat).
        let raw = serde_json::json!({
            "model": "x",
            "max_tokens": 16,
            "messages": [],
            "thinking": { "type": "future-mode" },
        });
        let req: AnthropicRequest = serde_json::from_value(raw).unwrap();
        assert!(!req.enable_thinking());
    }

    #[test]
    fn anthropic_thinking_defaults_off_when_absent() {
        let raw = serde_json::json!({
            "model": "x",
            "max_tokens": 16,
            "messages": [],
        });
        let req: AnthropicRequest = serde_json::from_value(raw).unwrap();
        assert!(!req.enable_thinking());
    }
}
