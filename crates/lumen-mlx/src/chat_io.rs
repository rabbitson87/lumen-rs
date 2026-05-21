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
