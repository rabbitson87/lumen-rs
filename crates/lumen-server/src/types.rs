use serde::{Deserialize, Deserializer, Serialize};

use lumen_mlx::SamplingOverrides;
use lumen_mlx::chat_io::ReasoningEffort;

/// OpenAI `stop`: accepts either a single string or an array of strings.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StopField {
    One(String),
    Many(Vec<String>),
}

impl StopField {
    /// Flatten to a list, dropping empties (an empty stop never matches).
    pub fn into_vec(self) -> Vec<String> {
        match self {
            StopField::One(s) => {
                if s.is_empty() {
                    Vec::new()
                } else {
                    vec![s]
                }
            }
            StopField::Many(v) => v.into_iter().filter(|s| !s.is_empty()).collect(),
        }
    }
}

fn stop_field_vec(stop: &Option<StopField>) -> Vec<String> {
    match stop {
        Some(StopField::One(s)) if !s.is_empty() => vec![s.clone()],
        Some(StopField::Many(v)) => v.iter().filter(|s| !s.is_empty()).cloned().collect(),
        _ => Vec::new(),
    }
}

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
    /// Per-request sampling overrides (Gemma 4 sampler only; other families
    /// ignore). When omitted, the backend falls back to env / family default.
    /// OpenAI chat doesn't standardize `top_k`/`seed`/`repeat_penalty`, but we
    /// accept them leniently so clients can tune the local sampler.
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    /// OpenAI `stop`: a single string or an array of strings. Generation
    /// halts when any is produced (the stop text is trimmed from the output).
    #[serde(default)]
    pub stop: Option<StopField>,
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
    /// OpenAI `parallel_tool_calls`. Absent means `true`, which is the
    /// documented default.
    ///
    /// `tool_choice` decides *whether* a tool is called; this decides *how
    /// many*. Until it was added the field was accepted and silently dropped —
    /// no `deny_unknown_fields` here — so a client asking for one call got
    /// however many the model produced, with a 200 and nothing to indicate the
    /// parameter had not been honoured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    // ── OpenAI structured outputs ──────────────────────────────────
    /// OpenAI `response_format`. When `json_object` / `json_schema`, the
    /// Gemma 4 backend builds an **Eager** grammar (active from token 0)
    /// that constrains the visible output to valid JSON matching the
    /// schema. Absent / `text` → no constraint (current behavior).
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
}

/// OpenAI `response_format` discriminated union. `text` is the implicit
/// default (no constraint); `json_object` forces any valid JSON object;
/// `json_schema` forces JSON matching the supplied schema.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: JsonSchemaSpec },
}

/// `response_format.json_schema` payload. Only `schema` is load-bearing
/// for grammar construction; `name` / `strict` are accepted for OpenAI
/// wire compatibility and currently advisory.
#[derive(Debug, Deserialize)]
pub struct JsonSchemaSpec {
    #[serde(default)]
    pub name: Option<String>,
    pub schema: serde_json::Value,
    #[serde(default)]
    pub strict: Option<bool>,
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
    /// Per-request sampling overrides for the backend (Gemma 4 sampler).
    pub fn sampling_overrides(&self) -> SamplingOverrides {
        SamplingOverrides {
            top_k: self.top_k,
            seed: self.seed,
            repeat_penalty: self.repeat_penalty,
            min_p: self.min_p,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            stop: stop_field_vec(&self.stop),
            parallel_tool_calls: self.parallel_tool_calls,
            // Only meaningful when thinking is actually on — the upstream
            // template gates its whole effort block on `enable_thinking`. The
            // `thinking` resolution here is the same one the request already
            // goes through, so a client that turns thinking off never gets an
            // effort instruction, matching the template.
            reasoning_effort: if self.enable_thinking() {
                // `reasoning_effort|default('xhigh')` — upstream injects the
                // xhigh sentence whenever thinking is on, INCLUDING when the
                // client named no level. Omitting it here would prompt a 3.8
                // checkpoint differently from every other 3.8 deployment.
                //
                // Known divergence: `LUMEN_BACKEND_THINKING_DEFAULT=1` turns
                // thinking on without a per-request signal, and this resolution
                // does not see that operator default, so such a request gets no
                // effort sentence. Explicit per-request signals are unaffected.
                match self.reasoning_effort.as_deref() {
                    Some(raw) => ReasoningEffort::from_request(raw),
                    None => Some(ReasoningEffort::default()),
                }
            } else {
                None
            },
        }
    }

    /// Resolve the JSON schema to constrain decoding against, derived from
    /// `response_format`:
    ///   - `json_schema` → the user-supplied `.schema`;
    ///   - `json_object` → a permissive any-object schema (`{"type":"object"}`);
    ///   - `text` / absent → `None` (no constraint — current behavior).
    ///
    /// Opt-in: returning `None` preserves the exact existing decode path.
    pub fn response_json_schema(&self) -> Option<serde_json::Value> {
        match self.response_format.as_ref()? {
            ResponseFormat::Text => None,
            ResponseFormat::JsonObject => Some(serde_json::json!({ "type": "object" })),
            ResponseFormat::JsonSchema { json_schema } => Some(json_schema.schema.clone()),
        }
    }

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

#[derive(Debug, Clone, Default)]
pub struct ChatMessage {
    pub role: String,
    /// Spec: `content` is `string` for system/user/tool roles, `string | null`
    /// for assistant when `tool_calls` is present. The OpenAI spec *also*
    /// allows the structured array form
    /// `[{ "type":"text", "text":"…" }, …]` — clients like Oh My Pi (omp)
    /// always send the user turn that way. We accept `string` / `null` /
    /// missing / array-of-parts and flatten to a single string so existing
    /// call sites that consume `&str` still work without `.unwrap_or_default()`
    /// scattered everywhere.
    ///
    /// Deserialization lives in the hand-written `Deserialize` impl below (and
    /// its `RawChatMessage` mirror), because one input field — `content` — has
    /// to populate both this and `images`, which serde's derive can't express.
    pub content: String,
    /// Present on `role:"assistant"` when the previous turn invoked tools.
    /// Carries the model's prior tool calls back into the prompt so the
    /// chat-template can re-render them. Server-emit side puts them in
    /// `ChatMessageResponse.tool_calls`.
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Required on `role:"tool"` — references the id of the tool call this
    /// message is answering.
    pub tool_call_id: Option<String>,
    /// Optional function name on `role:"tool"`. OpenAI legacy field; some
    /// clients still send it. Not required for routing — `tool_call_id` is
    /// the canonical link.
    pub name: Option<String>,
    /// Decoded bytes of every `image_url` content part on this message, in the
    /// order they appeared. Empty for text-only messages.
    ///
    /// Only `data:` URLs are accepted — the server never fetches remote URLs,
    /// because that would turn a chat request into an outbound HTTP fetch on
    /// the caller's behalf (SSRF). Clients must inline the image.
    pub images: Vec<Vec<u8>>,
}

/// Deserialization mirror of [`ChatMessage`]. `content` stays a raw `Value` so
/// one pass can split it into flattened text plus decoded images — serde's
/// derive can't route a single input field into two struct fields.
#[derive(Deserialize)]
struct RawChatMessage {
    #[serde(default)]
    role: String,
    #[serde(default)]
    content: Option<serde_json::Value>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCall>>,
    #[serde(default)]
    tool_call_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    /// The model's reasoning trace, returned by the client on a replayed
    /// assistant turn. `reasoning_content` is the DeepSeek / vLLM / SGLang
    /// spelling — and the name Qwen's own `chat_template.jinja` reads.
    #[serde(default)]
    reasoning_content: Option<String>,
    /// The spelling **Lumen itself emits** (`ChatMessageResponse.reasoning`,
    /// `ChatStreamDelta.reasoning`, matching Ollama's OpenAI-compat layer). A
    /// client that hands our own response object back has to work, so the field
    /// we write is a field we read.
    #[serde(default)]
    reasoning: Option<String>,
}

impl<'de> Deserialize<'de> for ChatMessage {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::Error as _;
        let raw = RawChatMessage::deserialize(d)?;
        let (content, images) = match raw.content {
            Some(v) => split_content(v).map_err(D::Error::custom)?,
            None => (String::new(), Vec::new()),
        };
        let content = fold_reasoning_into_content(
            &raw.role,
            content,
            raw.reasoning_content.as_deref(),
            raw.reasoning.as_deref(),
        );
        Ok(ChatMessage {
            role: raw.role,
            content,
            tool_calls: raw.tool_calls,
            tool_call_id: raw.tool_call_id,
            name: raw.name,
            images,
        })
    }
}

/// Normalize however the client returned the reasoning trace into the one
/// representation the renderers read: the `<think>…</think>` envelope at the
/// head of the assistant turn's `content`.
///
/// Three spellings reach us and all three mean the same thing —
/// `reasoning_content` (DeepSeek/vLLM, and Qwen's template's own field name),
/// `reasoning` (what we emit), and the envelope already inside `content` (what
/// we emit under `LUMEN_REASONING_IN_CONTENT=1`). Canonicalizing here, in the
/// one place every request passes through, is what keeps the renderers, the
/// prefix-cache key and the token count from each deciding separately.
///
/// Non-assistant roles are left alone: only the assistant has a trace, and a
/// user turn that happens to open with `<think>` is the user's text.
fn fold_reasoning_into_content(
    role: &str,
    content: String,
    reasoning_content: Option<&str>,
    reasoning: Option<&str>,
) -> String {
    if !role.eq_ignore_ascii_case("assistant") {
        return content;
    }
    // An envelope already in `content` wins — it is the trace in situ, and
    // wrapping it again would nest two blocks.
    if lumen_mlx::chat_io::has_reasoning_envelope(&content) {
        return content;
    }
    let trace = reasoning_content
        .or(reasoning)
        .map(str::trim)
        .filter(|t| !t.is_empty());
    match trace {
        Some(t) => lumen_mlx::chat_io::join_reasoning_envelope(t, &content),
        None => content,
    }
}

/// Split an OpenAI `content` value into flattened text and decoded images.
///
/// Accepts the four shapes a spec-conformant client may send:
/// `"text"` | `null` | missing | `[{ "type":"text", … }, { "type":"image_url", … }]`.
fn split_content(v: serde_json::Value) -> Result<(String, Vec<Vec<u8>>), String> {
    match v {
        serde_json::Value::Null => Ok((String::new(), Vec::new())),
        serde_json::Value::String(s) => Ok((s, Vec::new())),
        serde_json::Value::Array(parts) => {
            let mut text = String::new();
            let mut images = Vec::new();
            for p in &parts {
                if let Some(t) = p.get("text").and_then(serde_json::Value::as_str) {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(t);
                    continue;
                }
                let url = p
                    .get("image_url")
                    .and_then(|iu| iu.get("url"))
                    .and_then(serde_json::Value::as_str);
                if let Some(url) = url {
                    images.push(decode_data_url(url)?);
                }
            }
            Ok((text, images))
        }
        other => Err(format!(
            "message content must be a string, null, or array of content parts, got {other}"
        )),
    }
}

/// Decode a `data:<mime>;base64,<payload>` URL into raw bytes.
fn decode_data_url(url: &str) -> Result<Vec<u8>, String> {
    use base64::Engine as _;
    let rest = url.strip_prefix("data:").ok_or_else(|| {
        format!(
            "image_url must be a data: URL; remote fetching is not supported (got {}…)",
            url.chars().take(32).collect::<String>()
        )
    })?;
    let (meta, payload) = rest
        .split_once(',')
        .ok_or_else(|| "malformed data: URL — missing ','".to_string())?;
    if !meta.contains("base64") {
        return Err("image data: URL must be base64-encoded".to_string());
    }
    base64::engine::general_purpose::STANDARD
        .decode(payload.trim())
        .map_err(|e| format!("image_url base64 decode failed: {e}"))
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
            images: Vec::new(),
        }
    }

    /// True when any of `indices` points at a message carrying an inline image.
    ///
    /// Takes indices rather than a slice because the routing decision has to be
    /// made against the *post-strip* message set: a request whose only image
    /// rode on a meta-wrapper turn that `strip_client_meta_wrappers_flat`
    /// removed has no image left to encode, and must take the text path.
    pub fn any_images_at(messages: &[ChatMessage], indices: &[usize]) -> bool {
        indices.iter().any(|&i| !messages[i].images.is_empty())
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
    /// Per-request sampling overrides (Gemma 4 sampler only). See
    /// `ChatCompletionRequest` for semantics.
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub stop: Option<StopField>,
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
    // 0.95 matches the published Gemma 4 model params (Ollama
    // `gemma4:26b-mlx` Modelfile sets `top_p 0.95`). Applied only when the
    // client omits `top_p`; an explicit request value is always honored.
    // Override via `LUMEN_TOP_P`.
    std::env::var("LUMEN_TOP_P")
        .ok()
        .and_then(|s| s.trim().parse::<f32>().ok())
        .filter(|v| *v > 0.0 && *v <= 1.0)
        .unwrap_or(0.95)
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
    /// Anthropic natively supports `top_k`. `seed` / `repeat_penalty` are
    /// Lumen extensions for the local Gemma 4 sampler (other families ignore).
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
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
    Auto {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    Any {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    Tool {
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
}

impl AnthropicToolChoice {
    /// Anthropic hangs `disable_parallel_tool_use` off `tool_choice` itself
    /// rather than off the request, and it appears on every variant. Reported
    /// as OpenAI's `parallel_tool_calls` so one representation reaches the
    /// grammar — see [`lumen_mlx::grammar::ToolCalls`].
    pub fn parallel_tool_calls(&self) -> Option<bool> {
        let disabled = match self {
            Self::Auto {
                disable_parallel_tool_use,
            }
            | Self::Any {
                disable_parallel_tool_use,
            }
            | Self::Tool {
                disable_parallel_tool_use,
                ..
            } => *disable_parallel_tool_use,
        };
        disabled.map(|d| !d)
    }
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

impl CompletionRequest {
    /// Per-request sampling overrides for the backend (Gemma 4 sampler).
    pub fn sampling_overrides(&self) -> SamplingOverrides {
        SamplingOverrides {
            top_k: self.top_k,
            seed: self.seed,
            repeat_penalty: self.repeat_penalty,
            min_p: self.min_p,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            stop: stop_field_vec(&self.stop),
            // /v1/completions has no tools, so no tool-call count to cap.
            parallel_tool_calls: None,
            // Raw-prompt completions bypass the chat template entirely, so
            // there is no system block to prepend an effort instruction to.
            reasoning_effort: None,
        }
    }
}

impl AnthropicRequest {
    /// Per-request sampling overrides for the backend (Gemma 4 sampler).
    /// Anthropic carries `stop_sequences`; min_p/penalties aren't part of its
    /// API, so they fall back to env/family default.
    pub fn sampling_overrides(&self) -> SamplingOverrides {
        SamplingOverrides {
            top_k: self.top_k,
            seed: self.seed,
            repeat_penalty: self.repeat_penalty,
            min_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            stop: self.stop_sequences.clone().unwrap_or_default(),
            // Anthropic hangs the knob off `tool_choice`, inverted.
            parallel_tool_calls: self
                .tool_choice
                .as_ref()
                .and_then(|c| c.parallel_tool_calls()),
            // Anthropic has no `reasoning_effort` — its thinking control is a
            // token budget, which says nothing about which of Qwen's three
            // levels was meant. Left at the template's own default rather than
            // inventing a mapping from `budget_tokens`.
            reasoning_effort: self.enable_thinking().then(ReasoningEffort::default),
        }
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
    ///
    /// A `thinking` block becomes the leading `<think>…</think>` envelope, the
    /// same shape the OpenAI path folds `reasoning_content` into, so a replayed
    /// assistant turn renders with the trace the model actually produced —
    /// which is what lets it be a prefix of its own KV. `redacted_thinking`
    /// carries no readable trace and contributes nothing.
    pub fn as_text(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Blocks(blocks) => {
                let visible = blocks
                    .iter()
                    .filter_map(|b| match b {
                        AnthropicContentBlock::Text { text } => Some(text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                let trace = blocks
                    .iter()
                    .filter_map(|b| match b {
                        AnthropicContentBlock::Thinking { thinking, .. } => Some(thinking.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                if trace.trim().is_empty() {
                    visible
                } else {
                    lumen_mlx::chat_io::join_reasoning_envelope(&trace, &visible)
                }
            }
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
/// - `text`              — visible text (user/assistant)
/// - `image`             — inline image (base64 source only)
/// - `tool_use`          — assistant invoking a tool
/// - `tool_result`       — user message answering a prior tool_use
/// - `thinking`          — the assistant's extended-thinking trace, replayed
/// - `redacted_thinking` — the same, encrypted server-side
///
/// The tag is exhaustive: an unrecognized `type` fails the request rather than
/// being dropped, which is what surfaces a genuinely unsupported block instead
/// of answering as though it were not there.
///
/// That exhaustiveness is why `thinking` has to be listed. The Anthropic
/// Messages API *requires* a client to return the thinking blocks of any
/// assistant turn it replays once extended thinking is on, so a client doing
/// exactly what the spec demands was getting `unknown variant "thinking"` —
/// the whole request rejected over a block we only needed to read.
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentBlock {
    Text {
        text: String,
    },
    /// Extended-thinking trace. `signature` is Anthropic's integrity token; we
    /// have nothing to verify it against and nothing that would accept it back,
    /// so it is accepted and ignored rather than made to fail the request.
    Thinking {
        #[serde(default)]
        thinking: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    /// Encrypted thinking. The payload is opaque by construction — there is no
    /// trace to recover from it, so it contributes nothing to the prompt. Still
    /// accepted, because rejecting it would fail a conversation for carrying a
    /// block the client had no choice about.
    RedactedThinking {
        #[serde(default)]
        data: String,
    },
    Image {
        source: AnthropicImageSource,
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

/// `image.source` — Anthropic's inline-image envelope.
///
/// Only `type: "base64"` is accepted. Anthropic also defines a `url` source,
/// but fetching one would make the server issue outbound requests on a
/// caller's behalf (SSRF), which is the same reason the OpenAI path takes
/// `data:` URLs only.
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicImageSource {
    Base64 {
        #[serde(default)]
        media_type: String,
        data: String,
    },
}

impl AnthropicImageSource {
    /// Decode to raw image bytes.
    pub fn decode(&self) -> Result<Vec<u8>, String> {
        use base64::Engine as _;
        match self {
            Self::Base64 { data, .. } => base64::engine::general_purpose::STANDARD
                .decode(data.trim())
                .map_err(|e| format!("image source base64 decode failed: {e}")),
        }
    }
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

/// What we put in a `thinking` block's `signature`.
///
/// Anthropic signs its thinking blocks so the trace can be verified when the
/// client replays it. We have no key and nothing to verify against, but the
/// field is not optional in the SDK types (`ThinkingBlock.signature: str`), so
/// omitting it would make a strict client fail to parse a response it otherwise
/// understands. A short, obviously-not-base64 constant says what it is and
/// cannot be mistaken for a real signature — and if someone replays one of our
/// blocks to the actual Anthropic API, being rejected is the correct outcome.
///
/// The input side accepts any signature and ignores it, so this round-trips.
pub const LUMEN_THINKING_SIGNATURE: &str = "lumen-unsigned";

/// One block in an Anthropic response's `content[]`. Mirrors the request-side
/// `AnthropicContentBlock` for the variants the server actually emits — we
/// never emit `tool_result` (that's a client-side message back to us).
#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicResponseBlock {
    /// Extended-thinking trace. Per the Messages API this comes **first** in
    /// `content[]`, ahead of any text or tool_use, and is only present when the
    /// request enabled thinking.
    Thinking {
        thinking: String,
        signature: String,
    },
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

impl AnthropicResponseBlock {
    /// A thinking block carrying our own (unverifiable) signature.
    pub fn thinking(trace: impl Into<String>) -> Self {
        Self::Thinking {
            thinking: trace.into(),
            signature: LUMEN_THINKING_SIGNATURE.to_string(),
        }
    }
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

// === OpenAI Images API types (POST /v1/images/generations) ===

/// Request body for `POST /v1/images/generations`. OpenAI-compatible plus a
/// few diffusion extensions (`steps`, `seed`, `guidance`).
#[derive(Debug, Deserialize)]
pub struct ImageGenerationRequest {
    pub prompt: String,
    #[serde(default = "default_image_n")]
    pub n: usize,
    /// One of "256x256", "512x512", "1024x1024". Default "1024x1024".
    #[serde(default)]
    pub size: Option<String>,
    /// "b64_json" (supported) or "url" (rejected — local server has no URL host).
    #[serde(default)]
    pub response_format: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    // ── diffusion extensions ──
    /// Denoise steps. Default 28.
    #[serde(default = "default_image_steps")]
    pub steps: usize,
    /// Latent RNG seed. Default 0. Varied per image when `n > 1`.
    #[serde(default)]
    pub seed: u64,
    /// Classifier-free guidance scalar. Default 4.0.
    #[serde(default = "default_image_guidance")]
    pub guidance: f32,
}

fn default_image_n() -> usize {
    1
}
fn default_image_steps() -> usize {
    28
}
fn default_image_guidance() -> f32 {
    4.0
}

impl ImageGenerationRequest {
    /// Parse `(width, height)` from the `size` field. Accepts "WxH" with
    /// W,H multiples of 16. Defaults to 1024x1024 when absent.
    pub fn dimensions(&self) -> Result<(i32, i32), String> {
        let s = self.size.as_deref().unwrap_or("1024x1024");
        let (w, h) = s
            .split_once('x')
            .ok_or_else(|| format!("invalid size {s:?}; expected WxH e.g. \"512x512\""))?;
        let w: i32 = w
            .trim()
            .parse()
            .map_err(|_| format!("invalid width in size {s:?}"))?;
        let h: i32 = h
            .trim()
            .parse()
            .map_err(|_| format!("invalid height in size {s:?}"))?;
        if w <= 0 || h <= 0 || w % 16 != 0 || h % 16 != 0 {
            return Err(format!(
                "size {s:?}: width/height must be positive multiples of 16"
            ));
        }
        Ok((w, h))
    }
}

#[derive(Debug, Serialize)]
pub struct ImageGenerationResponse {
    pub created: u64,
    pub data: Vec<ImageData>,
}

#[derive(Debug, Serialize)]
pub struct ImageData {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b64_json: Option<String>,
}

#[cfg(test)]
mod image_content_serde {
    use super::*;
    use serde_json::json;

    /// 1×1 red PNG, base64. Content is irrelevant here — only that the bytes
    /// survive the data-URL round trip.
    const TINY_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

    fn msg(v: serde_json::Value) -> ChatMessage {
        serde_json::from_value(v).expect("parse ChatMessage")
    }

    #[test]
    fn plain_string_content_has_no_images() {
        let m = msg(json!({ "role": "user", "content": "hello" }));
        assert_eq!(m.content, "hello");
        assert!(m.images.is_empty());
    }

    #[test]
    fn null_and_missing_content_stay_empty() {
        assert_eq!(
            msg(json!({ "role": "assistant", "content": null })).content,
            ""
        );
        assert_eq!(msg(json!({ "role": "assistant" })).content, "");
    }

    #[test]
    fn text_parts_still_flatten_with_newlines() {
        let m = msg(json!({
            "role": "user",
            "content": [
                { "type": "text", "text": "line one" },
                { "type": "text", "text": "line two" },
            ]
        }));
        assert_eq!(m.content, "line one\nline two");
        assert!(m.images.is_empty());
    }

    /// The behavior this whole change exists to fix: `image_url` used to be
    /// dropped on the floor.
    #[test]
    fn image_url_part_is_decoded_not_dropped() {
        let m = msg(json!({
            "role": "user",
            "content": [
                { "type": "text", "text": "what is this?" },
                { "type": "image_url", "image_url": { "url": format!("data:image/png;base64,{TINY_PNG_B64}") } },
            ]
        }));
        assert_eq!(m.content, "what is this?");
        assert_eq!(m.images.len(), 1);
        assert_eq!(
            &m.images[0][..8],
            b"\x89PNG\r\n\x1a\n",
            "PNG magic survived"
        );
    }

    #[test]
    fn multiple_images_keep_their_order() {
        let url = format!("data:image/png;base64,{TINY_PNG_B64}");
        let m = msg(json!({
            "role": "user",
            "content": [
                { "type": "image_url", "image_url": { "url": url } },
                { "type": "image_url", "image_url": { "url": url } },
                { "type": "text", "text": "compare" },
            ]
        }));
        assert_eq!(m.images.len(), 2);
        assert_eq!(m.content, "compare");
    }

    /// Fetching remote URLs would make the server issue outbound requests on a
    /// caller's behalf; reject rather than silently answering without the image.
    #[test]
    fn remote_image_urls_are_rejected() {
        let err = serde_json::from_value::<ChatMessage>(json!({
            "role": "user",
            "content": [{ "type": "image_url", "image_url": { "url": "https://example.com/a.png" } }]
        }))
        .unwrap_err()
        .to_string();
        assert!(err.contains("data:"), "unexpected error: {err}");
    }

    #[test]
    fn non_base64_data_url_is_rejected() {
        assert!(serde_json::from_value::<ChatMessage>(json!({
            "role": "user",
            "content": [{ "type": "image_url", "image_url": { "url": "data:image/png,rawbytes" } }]
        }))
        .is_err());
    }

    /// Anthropic ships images as a tagged `image` block with a base64
    /// `source`, not as an `image_url` part. Before this variant existed the
    /// whole request failed to deserialize on the unknown tag, so
    /// Anthropic-format clients could not send images at all.
    #[test]
    fn anthropic_image_block_decodes() {
        let msg: AnthropicMessage = serde_json::from_value(json!({
            "role": "user",
            "content": [
                { "type": "text", "text": "what is this?" },
                { "type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": TINY_PNG_B64 } },
            ]
        }))
        .expect("parse anthropic message");
        let blocks = match &msg.content {
            AnthropicContent::Blocks(b) => b.clone(),
            other => panic!("expected blocks, got {other:?}"),
        };
        assert_eq!(blocks.len(), 2);
        let bytes = match &blocks[1] {
            AnthropicContentBlock::Image { source } => source.decode().expect("decode"),
            other => panic!("expected an image block, got {other:?}"),
        };
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n", "PNG magic survived");
        // Text flattening still ignores the image block — images travel on
        // their own channel, not through `as_text`.
        assert_eq!(msg.content.as_text(), "what is this?");
    }

    /// Anthropic also defines a `url` image source. Fetching it would make the
    /// server issue outbound requests for a caller, so it is refused the same
    /// way remote `image_url`s are on the OpenAI path.
    #[test]
    fn anthropic_url_image_source_is_rejected() {
        let err = serde_json::from_value::<AnthropicMessage>(json!({
            "role": "user",
            "content": [{ "type": "image", "source": {
                "type": "url", "url": "https://example.com/a.png" } }]
        }))
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("url") || err.contains("variant"),
            "unexpected: {err}"
        );
    }

    /// An unrecognized block type fails the request rather than being dropped —
    /// otherwise an unsupported modality would answer as if it were absent.
    #[test]
    fn unknown_anthropic_block_type_is_an_error() {
        assert!(
            serde_json::from_value::<AnthropicMessage>(json!({
                "role": "user",
                "content": [{ "type": "video", "source": {} }]
            }))
            .is_err()
        );
    }

    #[test]
    fn any_images_at_only_sees_the_indices_it_is_given() {
        let url = format!("data:image/png;base64,{TINY_PNG_B64}");
        let msgs = vec![
            msg(json!({ "role": "system", "content": "be brief" })),
            msg(json!({ "role": "user", "content": "ignore me" })),
            msg(json!({
                "role": "user",
                "content": [{ "type": "image_url", "image_url": { "url": url } }]
            })),
        ];
        assert!(ChatMessage::any_images_at(&msgs, &[0, 1, 2]));
        assert!(ChatMessage::any_images_at(&msgs, &[2]));
        // The image-bearing turn was stripped → nothing left to encode, so the
        // request must fall back to the text path rather than the vision one.
        assert!(!ChatMessage::any_images_at(&msgs, &[0, 1]));
        assert!(!ChatMessage::any_images_at(&msgs, &[]));
    }
}

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
    fn openai_array_content_flattens_to_text() {
        // omp (Oh My Pi) and other spec-conformant clients send the user turn
        // as a structured content array, not a bare string.
        let raw = json!({
            "role": "user",
            "content": [
                { "type": "text", "text": "현재 프로젝트를 파악해줄래?" }
            ]
        });
        let msg: ChatMessage = serde_json::from_value(raw).unwrap();
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "현재 프로젝트를 파악해줄래?");
    }

    #[test]
    fn openai_array_content_multi_part_text_is_joined() {
        // Image parts used to be silently discarded here. They now decode into
        // `ChatMessage::images` (see the `image_content_serde` module); this
        // test keeps pinning the text-flattening half of the contract.
        let raw = json!({
            "role": "user",
            "content": [
                { "type": "text", "text": "line one" },
                { "type": "text", "text": "line two" }
            ]
        });
        let msg: ChatMessage = serde_json::from_value(raw).unwrap();
        assert_eq!(msg.content, "line one\nline two");
        assert!(msg.images.is_empty());
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
        assert!(matches!(
            req.tool_choice,
            Some(AnthropicToolChoice::Auto { .. })
        ));
    }

    #[test]
    fn anthropic_tool_choice_named_parses() {
        let raw = json!({"type": "tool", "name": "get_weather"});
        let tc: AnthropicToolChoice = serde_json::from_value(raw).unwrap();
        match tc {
            AnthropicToolChoice::Tool { name, .. } => assert_eq!(name, "get_weather"),
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
            parallel_tool_calls: None,
            temperature: 0.7,
            top_p: 0.9,
            top_k: None,
            seed: None,
            repeat_penalty: None,
            min_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            stop: None,
            stream: false,
            stream_options: None,
            thinking: false,
            reasoning_effort: None,
            chat_template_kwargs: None,
            session_id: None,
            tools: None,
            tool_choice: None,
            response_format: None,
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

    #[test]
    fn image_request_defaults_and_dimensions() {
        let raw = serde_json::json!({ "prompt": "a red fox in snow" });
        let req: ImageGenerationRequest = serde_json::from_value(raw).unwrap();
        assert_eq!(req.n, 1);
        assert_eq!(req.steps, 28);
        assert_eq!(req.seed, 0);
        assert!((req.guidance - 4.0).abs() < 1e-6);
        // Default size is 1024x1024 when `size` is absent.
        assert_eq!(req.dimensions().unwrap(), (1024, 1024));
    }

    #[test]
    fn image_request_parses_explicit_size_and_extensions() {
        let raw = serde_json::json!({
            "prompt": "p",
            "n": 2,
            "size": "512x512",
            "steps": 10,
            "seed": 7,
            "guidance": 3.5,
            "response_format": "b64_json",
        });
        let req: ImageGenerationRequest = serde_json::from_value(raw).unwrap();
        assert_eq!(req.n, 2);
        assert_eq!(req.steps, 10);
        assert_eq!(req.seed, 7);
        assert!((req.guidance - 3.5).abs() < 1e-6);
        assert_eq!(req.dimensions().unwrap(), (512, 512));
    }

    #[test]
    fn image_request_rejects_non_multiple_of_16_size() {
        let raw = serde_json::json!({ "prompt": "p", "size": "513x512" });
        let req: ImageGenerationRequest = serde_json::from_value(raw).unwrap();
        assert!(req.dimensions().is_err());
    }
}

#[cfg(test)]
mod reasoning_effort_wiring_tests {
    use super::*;

    fn req(json: serde_json::Value) -> ChatCompletionRequest {
        serde_json::from_value(json).expect("request parses")
    }

    #[test]
    fn a_requested_effort_reaches_the_backend_overrides() {
        let r = req(serde_json::json!({
            "model": "q",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "high",
        }));
        assert!(r.enable_thinking(), "`high` turns thinking on");
        assert_eq!(
            r.sampling_overrides().reasoning_effort,
            Some(ReasoningEffort::Xhigh),
            "the effort must survive into SamplingOverrides — it is the only \
             channel that carries it to the prompt renderer"
        );
    }

    #[test]
    fn low_and_xhigh_are_carried_distinctly() {
        for (sent, want) in [
            ("low", ReasoningEffort::Low),
            ("medium", ReasoningEffort::Medium),
            ("xhigh", ReasoningEffort::Xhigh),
        ] {
            let r = req(serde_json::json!({
                "model": "q",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": sent,
            }));
            assert_eq!(
                r.sampling_overrides().reasoning_effort,
                Some(want),
                "sent {sent}"
            );
        }
    }

    #[test]
    fn an_effort_that_disables_thinking_carries_none() {
        let r = req(serde_json::json!({
            "model": "q",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "minimal",
        }));
        assert!(!r.enable_thinking());
        assert_eq!(r.sampling_overrides().reasoning_effort, None);
    }

    #[test]
    fn no_field_means_no_effort() {
        let r = req(serde_json::json!({
            "model": "q",
            "messages": [{"role": "user", "content": "hi"}],
        }));
        assert_eq!(r.sampling_overrides().reasoning_effort, None);
    }
}

/// The reasoning trace's round trip: what a client sends back, and whether the
/// renderer can find it.
///
/// A `thinking:true` generation prompt stops at an open `<think>` block and the
/// model writes the trace itself, so the KV holds tokens no empty block can
/// reproduce. Until the request type had somewhere to put the trace, a session
/// re-prefilled the whole conversation on every turn — correct output, and the
/// entire point of `session_id` silently lost.
#[cfg(test)]
mod reasoning_round_trip {
    use super::*;

    fn msg(json: serde_json::Value) -> ChatMessage {
        serde_json::from_value(json).expect("message parses")
    }

    #[test]
    fn the_deepseek_field_name_is_accepted() {
        // `reasoning_content` is vLLM's and SGLang's spelling — and the field
        // Qwen's own chat_template.jinja reads.
        let m = msg(serde_json::json!({
            "role": "assistant",
            "content": "42",
            "reasoning_content": "six times seven",
        }));
        assert_eq!(m.content, "<think>\nsix times seven\n</think>\n\n42");
    }

    #[test]
    fn the_field_lumen_emits_is_a_field_lumen_reads() {
        // `ChatMessageResponse.reasoning` is what we put on the wire, so a
        // client that hands our own response object straight back has to work.
        // Failing this would mean the server could not consume its own output.
        let m = msg(serde_json::json!({
            "role": "assistant",
            "content": "42",
            "reasoning": "six times seven",
        }));
        assert_eq!(m.content, "<think>\nsix times seven\n</think>\n\n42");
    }

    #[test]
    fn an_envelope_already_in_content_is_not_wrapped_twice() {
        // `LUMEN_REASONING_IN_CONTENT=1` puts the trace inside `content`. A
        // client echoing that back must not end up with nested blocks, whether
        // or not it also sets the dedicated field.
        let echoed = "<think>\nsix times seven\n</think>\n\n42";
        for extra in [
            serde_json::json!({}),
            serde_json::json!({"reasoning": "six times seven"}),
        ] {
            let mut body = serde_json::json!({"role": "assistant", "content": echoed});
            if let Some(o) = extra.as_object() {
                for (k, v) in o {
                    body[k] = v.clone();
                }
            }
            assert_eq!(msg(body).content, echoed);
        }
    }

    #[test]
    fn a_turn_without_a_trace_is_untouched() {
        // The no-regression case: every request that does not carry reasoning
        // must render exactly the bytes it did before this field existed.
        let m = msg(serde_json::json!({"role": "assistant", "content": "42"}));
        assert_eq!(m.content, "42");
        // An empty or whitespace-only trace is no trace.
        for blank in ["", "   ", "\n"] {
            let m = msg(serde_json::json!({
                "role": "assistant", "content": "42", "reasoning_content": blank,
            }));
            assert_eq!(m.content, "42", "blank {blank:?} must not open a block");
        }
    }

    #[test]
    fn only_the_assistant_has_a_trace() {
        // A user turn opening with `<think>` is the user's text, and a stray
        // field on a user message is not a reason to rewrite what they typed.
        let m = msg(serde_json::json!({
            "role": "user",
            "content": "<think> is a tag I use",
            "reasoning_content": "not mine",
        }));
        assert_eq!(m.content, "<think> is a tag I use");
    }

    #[test]
    fn an_anthropic_thinking_block_is_accepted_and_kept() {
        // The Messages API *requires* a client to replay the thinking blocks of
        // any assistant turn it sends back once extended thinking is on. The
        // block tag is exhaustive, so before `thinking` was listed, a client
        // doing exactly what the spec demands had its whole request rejected
        // with `unknown variant`.
        let m: AnthropicMessage = serde_json::from_value(serde_json::json!({
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "six times seven", "signature": "sig"},
                {"type": "text", "text": "42"},
            ],
        }))
        .expect("a spec-conformant replay must parse");
        assert_eq!(
            m.content.as_text(),
            "<think>\nsix times seven\n</think>\n\n42",
            "the trace has to reach the renderer, not just survive parsing"
        );
    }

    #[test]
    fn redacted_thinking_parses_and_contributes_nothing() {
        // The payload is encrypted — there is no trace to recover. Accepting it
        // is still right: the client had no choice about receiving it.
        let m: AnthropicMessage = serde_json::from_value(serde_json::json!({
            "role": "assistant",
            "content": [
                {"type": "redacted_thinking", "data": "EncryptedBlob=="},
                {"type": "text", "text": "42"},
            ],
        }))
        .expect("redacted blocks must not fail the request");
        assert_eq!(m.content.as_text(), "42");
    }

    /// The loop, closed: our own response block, parsed back as a request.
    ///
    /// The two sides are separate types (`AnthropicResponseBlock` is emit-only),
    /// so nothing but a test makes them agree. This is the property the whole
    /// round-trip rests on — a client that echoes our `content[]` back has to
    /// produce a prompt carrying the trace, or the KV cannot be extended.
    #[test]
    fn an_anthropic_thinking_block_we_emitted_parses_back_as_one_we_accept() {
        let emitted = AnthropicResponseBlock::thinking("six times seven");
        let wire = serde_json::to_value(&emitted).expect("serializes");
        assert_eq!(wire["type"], "thinking");

        let replayed: AnthropicMessage = serde_json::from_value(serde_json::json!({
            "role": "assistant",
            "content": [wire, {"type": "text", "text": "42"}],
        }))
        .expect("our own output must be valid input");
        assert_eq!(
            replayed.content.as_text(),
            "<think>\nsix times seven\n</think>\n\n42",
            "the trace has to reach the renderer, not merely survive the parse"
        );
    }

    #[test]
    fn an_unknown_block_type_still_fails() {
        // Widening the enum must not turn it into a shrug. An unrecognized
        // block is a genuinely unsupported feature and has to say so rather
        // than be answered as though it were not there.
        let r: Result<AnthropicMessage, _> = serde_json::from_value(serde_json::json!({
            "role": "assistant",
            "content": [{"type": "server_tool_use", "id": "x", "name": "y", "input": {}}],
        }));
        assert!(r.is_err(), "unknown block types must still be rejected");
    }
}
