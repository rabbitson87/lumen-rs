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
#[derive(Debug, Clone, Copy, Default)]
pub enum ResolvedToolChoice<'a> {
    #[default]
    Auto,
    None,
    Required,
    Tool(&'a str),
}

/// Reasoning-effort level, as Qwen 3.8's chat template understands it.
///
/// 3.8 added a block that prepends an instruction sentence to the system
/// prompt when thinking is on. The model was tuned with that sentence present,
/// so omitting it prompts a 3.8 checkpoint the way a 3.6 one expects. Only
/// three levels exist upstream, and an unrecognized one makes the template
/// `raise_exception`.
///
/// `Medium` deliberately carries no text — that is the upstream template's own
/// behaviour (it sets the instruction string only for `xhigh` and `low`), not
/// an omission here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    Xhigh,
    Medium,
    Low,
}

impl Default for ReasoningEffort {
    /// The template's own default (`reasoning_effort|default('xhigh')`).
    fn default() -> Self {
        Self::Xhigh
    }
}

impl ReasoningEffort {
    /// Map an OpenAI-style `reasoning_effort` string onto Qwen's vocabulary.
    ///
    /// The two scales do not agree: OpenAI ships `minimal|low|medium|high`,
    /// Qwen expects `low|medium|xhigh`. `"high"` is therefore both the most
    /// likely input and one the upstream template would refuse, so it maps to
    /// `Xhigh` (each is the top of its own scale).
    ///
    /// Returns `None` for the values that mean "do not think at all"
    /// (`minimal`, `none`, `off`, `disabled`, empty) — those already turn
    /// thinking off one layer up in `enable_thinking_with_backend_default`, and
    /// the template emits no instruction when thinking is off.
    ///
    /// An unrecognized value falls back to the default rather than erroring: a
    /// server should not fail a request over a spelling it does not know, and
    /// the alternative is propagating the template's `raise_exception`.
    pub fn from_request(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "minimal" | "none" | "off" | "disabled" | "" => None,
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" | "xhigh" => Some(Self::Xhigh),
            _ => Some(Self::default()),
        }
    }

    /// The instruction sentence to prepend to the system block, verbatim from
    /// Qwen 3.8's `chat_template.jinja` so the model sees the exact phrasing it
    /// was trained on. `None` for `Medium`, which upstream leaves empty.
    pub fn instructions(self) -> Option<&'static str> {
        match self {
            Self::Xhigh => Some(
                "Reasoning effort is set to xhigh. Please think carefully through the task, \
                 validate key assumptions, consider plausible alternatives, and prioritize \
                 correctness, consistency, and clarity in the final answer.",
            ),
            Self::Low => Some(
                "Reasoning effort is set to low. Keep your thinking brief and focused, moving \
                 directly to the conclusion without unnecessary elaboration.",
            ),
            Self::Medium => None,
        }
    }

    /// Short, stable tag for cache keys. MUST distinguish every level that can
    /// change the rendered prompt — see `auto_prefix_key`, where a shared key
    /// across levels would serve an `xhigh` prefix to a `low` request.
    pub fn cache_tag(self) -> &'static str {
        match self {
            Self::Xhigh => "xh",
            Self::Medium => "md",
            Self::Low => "lo",
        }
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
    if tool_names.contains(&"task_complete") {
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

// ─────────────────────────────────────────────────────────────────────────────
// Reasoning-trace round-trip
// ─────────────────────────────────────────────────────────────────────────────

/// Opening tag of the reasoning envelope.
pub const REASONING_OPEN: &str = "<think>";
/// Closing tag of the reasoning envelope.
pub const REASONING_CLOSE: &str = "</think>";

/// Wrap a reasoning trace and the visible reply into the single string an
/// assistant turn's `content` carries through the renderers.
///
/// This is deliberately the *same* envelope `LUMEN_REASONING_IN_CONTENT=1`
/// already emits on the response side, so a client that echoes our `content`
/// back verbatim and a client that returns the trace in a dedicated
/// `reasoning_content` field converge on one representation before anything
/// downstream has to care which it was.
///
/// The alternative — a third field on `ChatTurn::Assistant` and on the flat
/// `(role, content)` pair — would have to be threaded through 55 flat and 16
/// structured call sites to change the same rendered bytes. The envelope
/// reaches both paths, both APIs, the prefix-cache key and the token count at
/// once, because all of them already agree that the turn's text is its content.
pub fn join_reasoning_envelope(reasoning: &str, visible: &str) -> String {
    let reasoning = reasoning.trim();
    format!("{REASONING_OPEN}\n{reasoning}\n{REASONING_CLOSE}\n\n{visible}")
}

/// Split an assistant turn's text back into `(reasoning, visible)`.
///
/// Returns `("", content)` unchanged when there is no envelope — which is what
/// makes every existing caller byte-identical, since a turn without a trace
/// renders exactly as it did before.
///
/// Only a *leading* envelope counts. A `<think>` further into the reply is the
/// model talking about the tag, not a trace, and rewriting it would corrupt the
/// turn.
pub fn split_reasoning_envelope(content: &str) -> (&str, &str) {
    let Some(rest) = content.strip_prefix(REASONING_OPEN) else {
        return ("", content);
    };
    let Some(end) = rest.find(REASONING_CLOSE) else {
        // An unterminated `<think>` is a truncated turn, not an envelope.
        return ("", content);
    };
    let reasoning = rest[..end].trim();
    let visible = rest[end + REASONING_CLOSE.len()..].trim_start();
    (reasoning, visible)
}

/// Whether `content` already carries a reasoning envelope. Used to keep a
/// client that sends *both* the envelope and the dedicated field from getting
/// two nested blocks.
pub fn has_reasoning_envelope(content: &str) -> bool {
    !split_reasoning_envelope(content).0.is_empty()
}

#[cfg(test)]
mod reasoning_envelope_tests {
    use super::*;

    #[test]
    fn a_turn_without_a_trace_is_returned_untouched() {
        // The property every existing call site depends on: no envelope, no
        // change. This is what makes the split safe to apply unconditionally.
        for s in [
            "",
            "hello",
            "a <think> mid-sentence",
            "<think> unterminated",
        ] {
            assert_eq!(split_reasoning_envelope(s), ("", s), "input {s:?}");
        }
    }

    #[test]
    fn the_split_recovers_exactly_what_the_join_wrote() {
        let joined = join_reasoning_envelope("  let me think  ", "the answer");
        assert_eq!(joined, "<think>\nlet me think\n</think>\n\nthe answer");
        assert_eq!(
            split_reasoning_envelope(&joined),
            ("let me think", "the answer")
        );
    }

    #[test]
    fn an_empty_trace_round_trips_as_the_empty_block() {
        // `<think>\n\n</think>\n\n` is what a `thinking:false` generation
        // prompt ends with, so this case has to stay byte-exact or the
        // non-thinking replay stops matching the KV it is trying to extend.
        let joined = join_reasoning_envelope("", "hi");
        assert_eq!(joined, "<think>\n\n</think>\n\nhi");
        assert_eq!(split_reasoning_envelope(&joined), ("", "hi"));
    }

    #[test]
    fn a_multi_line_trace_survives_the_round_trip() {
        let trace = "step one\nstep two\n\nstep three";
        let joined = join_reasoning_envelope(trace, "done");
        assert_eq!(split_reasoning_envelope(&joined), (trace, "done"));
    }
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

#[cfg(test)]
mod reasoning_effort_mapping_tests {
    use super::ReasoningEffort;

    #[test]
    fn openai_high_maps_onto_qwens_xhigh() {
        // The scales disagree: OpenAI ships minimal|low|medium|high, Qwen wants
        // low|medium|xhigh. "high" is the likeliest input AND one the upstream
        // template would `raise_exception` on, so it must land on the top of
        // Qwen's scale rather than pass through.
        assert_eq!(
            ReasoningEffort::from_request("high"),
            Some(ReasoningEffort::Xhigh)
        );
        assert_eq!(
            ReasoningEffort::from_request("xhigh"),
            Some(ReasoningEffort::Xhigh)
        );
    }

    #[test]
    fn the_thinking_disabling_values_carry_no_effort() {
        for v in ["minimal", "none", "off", "disabled", "", "  "] {
            assert_eq!(
                ReasoningEffort::from_request(v),
                None,
                "{v:?} should carry no effort"
            );
        }
    }

    #[test]
    fn an_unknown_value_falls_back_instead_of_failing_the_request() {
        // The upstream template raises on an unrecognized effort. A server
        // should not turn a spelling it does not know into a 500.
        assert_eq!(
            ReasoningEffort::from_request("turbo"),
            Some(ReasoningEffort::Xhigh)
        );
        assert_eq!(ReasoningEffort::default(), ReasoningEffort::Xhigh);
    }

    #[test]
    fn case_and_padding_are_tolerated() {
        assert_eq!(
            ReasoningEffort::from_request("  LOW "),
            Some(ReasoningEffort::Low)
        );
        assert_eq!(
            ReasoningEffort::from_request("Medium"),
            Some(ReasoningEffort::Medium)
        );
    }

    #[test]
    fn only_medium_has_no_sentence_and_every_level_keys_apart() {
        assert!(ReasoningEffort::Medium.instructions().is_none());
        assert!(ReasoningEffort::Xhigh.instructions().is_some());
        assert!(ReasoningEffort::Low.instructions().is_some());
        let tags = [
            ReasoningEffort::Xhigh.cache_tag(),
            ReasoningEffort::Medium.cache_tag(),
            ReasoningEffort::Low.cache_tag(),
        ];
        let uniq: std::collections::HashSet<_> = tags.iter().collect();
        assert_eq!(uniq.len(), 3, "cache tags must distinguish every level");
    }
}
