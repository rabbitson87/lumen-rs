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
    /// Reasoning-channel delta (Gemma 4 `<|channel>thought\n…<channel|>`
    /// block content, etc.). Emitted separately from `Text` so the HTTP
    /// layer can wrap with `<think>…</think>` (Ayla-style text tag) AND
    /// populate the OpenAI `delta.reasoning` field — keeping the two
    /// channels disambiguated for downstream clients.
    Reasoning(&'a str),
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

/// Heuristic: does this user-role message look like a client-injected
/// meta-instruction wrapper (e.g. "If you have fully completed the task,
/// call the task_complete tool now. Otherwise return your final answer.")?
///
/// Why: such wrappers consistently push chat-template-strict models (Gemma 4)
/// off-distribution — user-role meta-instructions are uncommon in training,
/// referencing tool names in user prose is uncommon, and per-turn
/// accumulation compounds the drift. Empirical failure mode: repetition loops
/// + max_tokens runaway (decoder never emits EOS).
///
/// To avoid false positives on legitimate user prose, the heuristic requires
/// BOTH a meta-instruction *opener* AND a tool/completion *signal*. Generic
/// conditionals ("if you have time, please...") do not match because they
/// lack the tool/completion signal.
pub fn is_client_meta_wrapper(content: &str) -> bool {
    let t = content.trim();
    if t.len() < 30 || t.len() > 500 {
        return false;
    }
    let lower = t.to_lowercase();
    let opener = lower.starts_with("if you have")
        || lower.starts_with("if the task")
        || lower.starts_with("once you have")
        || lower.starts_with("when the task");
    let signal = lower.contains("task_complete")
        || lower.contains("fully completed")
        || lower.contains("final answer")
        || (lower.contains("call the") && lower.contains("tool"));
    opener && signal
}

/// Strip client-injected meta-instruction user wrappers from a built
/// `Vec<ChatTurn>`, in place. Default ON; disable with
/// `LUMEN_STRIP_CLIENT_META_WRAPPERS=0`. Logs each strip to stderr.
///
/// Only User turns are inspected; Tool/Assistant/System turns are never
/// touched. Wrapper user turns are removed entirely — their intent
/// (forcing structured outputs) is better expressed via the request-level
/// `tool_choice="required"` (OpenAI) or `tool_choice={type:"any"}`
/// (Anthropic) parameters, both first-class.
pub fn strip_client_meta_wrappers(turns: &mut Vec<ChatTurn<'_>>) {
    let _ = strip_client_meta_wrappers_indexed(turns);
}

/// [`strip_client_meta_wrappers`] that also reports, for each surviving turn,
/// its index in the input.
///
/// The turn counterpart of [`strip_client_meta_wrappers_flat_indexed`], and it
/// exists for the same reason: a caller carrying a per-turn side table — image
/// attachments — has to filter it with the *same* predicate, or entry `i` ends
/// up describing a different turn than it did before the strip.
pub fn strip_client_meta_wrappers_indexed(turns: &mut Vec<ChatTurn<'_>>) -> Vec<usize> {
    if std::env::var("LUMEN_STRIP_CLIENT_META_WRAPPERS")
        .map(|s| s == "0" || s.eq_ignore_ascii_case("false") || s.eq_ignore_ascii_case("off"))
        .unwrap_or(false)
    {
        return (0..turns.len()).collect();
    }
    let before = turns.len();
    let mut kept: Vec<usize> = Vec::with_capacity(before);
    let mut idx = 0usize;
    turns.retain(|t| {
        let i = idx;
        idx += 1;
        match t {
            ChatTurn::User(s) if is_client_meta_wrapper(s) => {
                let preview: String = s.trim().chars().take(80).collect();
                eprintln!(
                    "[chat-io] stripped client meta-wrapper user turn (len={}): {preview:?}...",
                    s.len()
                );
                false
            }
            _ => {
                kept.push(i);
                true
            }
        }
    });
    if turns.len() < before {
        eprintln!(
            "[chat-io] stripped {} client meta-wrapper(s); use tool_choice=\"required\" \
             to force structured outputs instead",
            before - turns.len()
        );
    }
    kept
}

/// Heuristic classification of the chat request into "agent" vs "chat"
/// mode, used by the server to log the inferred operating mode and (in
/// downstream callers) tune sampling guards.
///
/// **Agent mode signals** (any one is enough — non-trivial agentic loops
/// always carry at least one):
///   - A tool named `task_complete` is offered (Ayla/Moltis convention)
///   - System prompt contains an "Agentic Loop" / "Task Completion" /
///     "Thinking Discipline" section header
///   - More than one tool is offered AND the system prompt mentions
///     `task_complete` by name
///
/// Anything else is treated as **chat mode** (plain conversation, simple
/// tool use, no agentic discipline rules). The returned label is purely
/// observational — current callers log it; a future change may use it to
/// auto-tune `min_tokens_before_eos` to protect chat-mode responses from
/// the EOS-sampling outlier failure mode without disrupting agent-mode
/// fast tool calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferredMode {
    Chat,
    Agent,
}

impl InferredMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            InferredMode::Chat => "chat",
            InferredMode::Agent => "agent",
        }
    }
}

pub fn classify_request_mode<S1, S2>(messages: &[(S1, S2)], tool_names: &[&str]) -> InferredMode
where
    S1: AsRef<str>,
    S2: AsRef<str>,
{
    // Strongest signal: explicit `task_complete` tool.
    if tool_names.iter().any(|n| *n == "task_complete") {
        return InferredMode::Agent;
    }

    // Look at the first system message for agentic markers.
    let system_text = messages
        .iter()
        .find(|(r, _)| {
            let r = r.as_ref();
            r.eq_ignore_ascii_case("system")
        })
        .map(|(_, c)| c.as_ref());

    if let Some(sys) = system_text {
        // Header / section markers used by all popular agent frameworks.
        const AGENTIC_MARKERS: &[&str] = &[
            "Agentic Loop",
            "Task Completion",
            "Thinking Discipline",
            "Tool Usage Discipline",
            "Scope Discipline",
        ];
        if AGENTIC_MARKERS.iter().any(|m| sys.contains(m)) {
            return InferredMode::Agent;
        }
        // `task_complete` namedrop + multiple tools = agentic.
        if tool_names.len() > 1 && sys.contains("task_complete") {
            return InferredMode::Agent;
        }
    }

    InferredMode::Chat
}

/// Plain-path variant of [`strip_client_meta_wrappers`] operating on
/// `Vec<(String, String)>` (role, content) — used by the non-structured
/// chat path where `messages` never gets converted to `ChatTurn`. Same
/// semantics and env override (`LUMEN_STRIP_CLIENT_META_WRAPPERS=0`)
/// as the structured variant.
pub fn strip_client_meta_wrappers_flat(messages: &mut Vec<(String, String)>) {
    let _ = strip_client_meta_wrappers_flat_indexed(messages);
}

/// [`strip_client_meta_wrappers_flat`] that also reports, for each surviving
/// message, its index in the input.
///
/// Callers that carry a per-message side table — image attachments, in
/// particular — must filter it with the *same* predicate, or entry `i` ends up
/// describing a different turn than it did before the strip. Returning the
/// surviving indices keeps that predicate in one place instead of inviting a
/// second, drifting copy at the call site.
pub fn strip_client_meta_wrappers_flat_indexed(messages: &mut Vec<(String, String)>) -> Vec<usize> {
    if std::env::var("LUMEN_STRIP_CLIENT_META_WRAPPERS")
        .map(|s| s == "0" || s.eq_ignore_ascii_case("false") || s.eq_ignore_ascii_case("off"))
        .unwrap_or(false)
    {
        return (0..messages.len()).collect();
    }
    let before = messages.len();
    let mut kept: Vec<usize> = Vec::with_capacity(before);
    let mut idx = 0usize;
    messages.retain(|(role, content)| {
        let i = idx;
        idx += 1;
        let is_user = matches!(role.as_str(), "user" | "User" | "USER");
        if is_user && is_client_meta_wrapper(content) {
            let preview: String = content.trim().chars().take(80).collect();
            eprintln!(
                "[chat-io] stripped client meta-wrapper user turn (len={}): {preview:?}...",
                content.len()
            );
            false
        } else {
            kept.push(i);
            true
        }
    });
    if messages.len() < before {
        eprintln!(
            "[chat-io] stripped {} client meta-wrapper(s); use tool_choice=\"required\" \
             to force structured outputs instead",
            before - messages.len()
        );
    }
    kept
}

#[cfg(test)]
mod meta_wrapper_tests {
    use super::*;

    #[test]
    fn detects_moltis_wrapper() {
        let s = "If you have fully completed the task, call the task_complete tool now. \
                 Otherwise return your final answer.";
        assert!(is_client_meta_wrapper(s));
    }

    #[test]
    fn does_not_match_legit_user_prose() {
        assert!(!is_client_meta_wrapper(
            "If you have time, please review my PR"
        ));
        assert!(!is_client_meta_wrapper(
            "Once you have finished, let me know"
        ));
        assert!(!is_client_meta_wrapper("How do I call the API?"));
        assert!(!is_client_meta_wrapper("What is 2+2?"));
        // Too short to be a wrapper (under 30 chars).
        assert!(!is_client_meta_wrapper("If you have fully completed"));
    }

    #[test]
    fn strips_only_user_wrappers() {
        let mut turns = vec![
            ChatTurn::System("be helpful"),
            ChatTurn::User("What's the weather?"),
            ChatTurn::Assistant {
                text: "",
                tool_calls: &[],
            },
            ChatTurn::Tool {
                tool_call_id: "c1",
                name: Some("get_weather"),
                content: "sunny",
            },
            ChatTurn::User(
                "If you have fully completed the task, call the task_complete tool now. \
                 Otherwise return your final answer.",
            ),
        ];
        strip_client_meta_wrappers(&mut turns);
        assert_eq!(turns.len(), 4);
        // Real user turn preserved
        assert!(matches!(turns[1], ChatTurn::User(s) if s == "What's the weather?"));
    }

    const WRAPPER: &str = "If you have fully completed the task, call the task_complete \
                           tool now. Otherwise return your final answer.";

    fn flat(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(r, c)| (r.to_string(), c.to_string()))
            .collect()
    }

    /// The surviving indices are what lets a caller filter a parallel
    /// side-table (image attachments) in lockstep with the messages.
    #[test]
    fn indexed_strip_reports_surviving_positions() {
        let mut msgs = flat(&[
            ("system", "be helpful"),
            ("user", WRAPPER),
            ("user", "what is in this image?"),
        ]);
        let kept = strip_client_meta_wrappers_flat_indexed(&mut msgs);
        assert_eq!(kept, vec![0, 2]);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[1].1, "what is in this image?");
    }

    #[test]
    fn indexed_strip_is_identity_when_nothing_matches() {
        let mut msgs = flat(&[("system", "be helpful"), ("user", "hello")]);
        assert_eq!(
            strip_client_meta_wrappers_flat_indexed(&mut msgs),
            vec![0, 1]
        );
        assert_eq!(msgs.len(), 2);
    }
}
