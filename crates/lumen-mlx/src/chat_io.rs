//! Non-feature-gated chat I/O data types.
//!
//! Lives outside `gemma4_response` / `gemma4_tools` (both `#[cfg(feature =
//! "mlx-native")]`) so the lumen-server engine can use these structs as the
//! universal `Backend::chat` return / argument shape regardless of which
//! features the binary was compiled with.
//!
//! The pure-data structs have no Metal / MLX dependency — only the
//! tokenizer-driven *parsers* and *renderers* that produce / consume them
//! sit behind feature gates.

use serde_json::Value as JsonValue;

/// One parsed function call extracted from a backend's output (e.g. Gemma 4
/// `<|tool_call>call:NAME{...}<tool_call|>` block).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedToolCall {
    pub name: String,
    /// Canonicalized JSON arguments. HTTP layer re-encodes this to the
    /// wire format each spec demands:
    /// - OpenAI: `function.arguments` is a JSON-encoded **string**
    /// - Anthropic: `input` is the raw JSON **object**
    pub arguments: JsonValue,
}

/// Structured chat output — visible reply text, optional reasoning block,
/// and any tool calls the model emitted. Backends without tool-call demux
/// (legacy Candle, Qwen 3.5 family at Phase 1.3) return empty `tool_calls`.
#[derive(Debug, Clone, Default)]
pub struct ParsedResponse {
    pub visible: String,
    pub reasoning: String,
    pub tool_calls: Vec<ParsedToolCall>,
}

/// One tool surface the model is allowed to invoke. Used by chat-template
/// renderers (e.g. Gemma 4 `format_function_declaration` → tokenized
/// `<|tool>declaration:NAME{...}<tool|>` blocks). Intentionally independent
/// of `lumen-server`'s OpenAI / Anthropic `Tool` enums so this crate stays
/// API-agnostic.
#[derive(Debug, Clone)]
pub struct ToolDef<'a> {
    pub name: &'a str,
    pub description: Option<&'a str>,
    /// JSON Schema describing the function arguments. Backend-specific
    /// renderers walk this Value to produce the wire format.
    pub parameters: Option<&'a JsonValue>,
    /// Optional response schema — Gemma 4's template supports a
    /// `response:{...}` block when present. Rare in practice (the OpenAI
    /// / Anthropic specs don't expose this directly).
    pub response: Option<&'a JsonValue>,
}

/// Structured chat turn used by the tool-aware backend path. Carries the
/// extra fields (`tool_calls`, `tool_call_id`) that the legacy `(role,
/// content)` shape can't represent. The chat-template renderer groups a
/// `Assistant{ tool_calls }` followed by N `Tool{ tool_call_id, ... }`
/// turns into a single model turn per the canonical Gemma 4 layout.
#[derive(Debug, Clone)]
pub enum ChatTurn<'a> {
    System(&'a str),
    User(&'a str),
    Assistant {
        /// Visible text. May be empty when `tool_calls` is non-empty.
        text: &'a str,
        /// Tool calls the assistant emitted on this turn (in order).
        tool_calls: &'a [AssistantToolCall<'a>],
    },
    /// Tool execution result coming back from the client. Always follows
    /// (in input order) the Assistant turn that issued the matching
    /// `tool_call_id`.
    Tool {
        /// Matches `AssistantToolCall::id` from the prior assistant turn.
        tool_call_id: &'a str,
        /// Optional function name — clients may or may not send it.
        /// Renderers that need the name resolve via `tool_call_id` lookup
        /// against the preceding assistant's `tool_calls`.
        name: Option<&'a str>,
        /// Raw tool output. The renderer is responsible for any
        /// model-specific wrapping (Gemma 4 puts this inside
        /// `<|tool_response>response:NAME{value:<|"|>...<|"|>}<tool_response|>`).
        content: &'a str,
    },
}

/// One historical tool call attached to an `Assistant` turn — used when
/// the client is replaying a prior turn whose assistant message issued
/// tool calls. Mirrors `ParsedToolCall` but borrows its fields rather than
/// owning them (the surrounding request keeps them alive).
#[derive(Debug, Clone)]
pub struct AssistantToolCall<'a> {
    pub id: &'a str,
    pub name: &'a str,
    /// Parsed JSON arguments. The OpenAI wire format ships these as a
    /// JSON-encoded string; engine.rs deserializes before constructing
    /// the borrow.
    pub arguments: &'a JsonValue,
}

/// Phase 1.6c: streaming events emitted from the backend's decode loop
/// up to the engine. Replaces the old `Fn(&str)` text-only callback so
/// backends can also signal "I just parsed the name of a tool call"
/// before the full tool-call body is produced, shaving 200-400ms off
/// the time-to-first-tool-call-chunk for clients (Claude Code, Cursor,
/// Aider, etc.) that render a "🔧 calling <name>..." indicator.
///
/// The id is NOT emitted by the backend — the HTTP layer assigns
/// wire ids (`call_…` / `toolu_…`) when relaying to SSE.
#[derive(Debug, Clone)]
pub enum BackendStreamEvent<'a> {
    /// Visible-text delta (the assistant's natural-language reply).
    Text(&'a str),
    /// Backend's parser has identified the start of a tool call —
    /// it's seen `call:NAME{` and accumulated `NAME`. Args body is
    /// not yet known; the engine emits the SSE start chunk now and
    /// fills in args via a separate `ArgumentsDelta` event after
    /// the backend finishes parsing the body.
    ToolCallStart { name: &'a str },
}

/// Phase 1.6: normalized tool-choice intent passed from the HTTP layer
/// to the backend renderer. OpenAI's `ToolChoice` and Anthropic's
/// `AnthropicToolChoice` both map onto this — `Auto` is the default
/// (model decides), `None` strips tool definitions from the prompt
/// entirely, `Required` and `Tool(name)` prefill `<|tool_call>` (and
/// optionally `call:NAME{`) at the end of the generation prompt so
/// the model MUST start generating a tool call.
#[derive(Debug, Clone, Copy)]
pub enum ResolvedToolChoice<'a> {
    Auto,
    None,
    Required,
    Tool(&'a str),
}

impl Default for ResolvedToolChoice<'_> {
    fn default() -> Self {
        ResolvedToolChoice::Auto
    }
}
