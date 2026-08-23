//! Qwen 3.5 / 3.6 tool-calling renderer + response parser (Phase 2).
//!
//! Qwen 3.6 uses a **nested-XML** tool format trained into the model — NOT
//! Hermes JSON, NOT Gemma 4's special-token bracket form. Reference shape
//! (from the model's chat_template.jinja):
//!
//! ```text
//! <|im_start|>system
//! # Tools
//!
//! You have access to the following functions:
//!
//! <tools>
//! {"name":"get_weather","description":"...","parameters":{...}}
//! </tools>
//!
//! If you choose to call a function ONLY reply in the following format with
//! NO suffix:
//!
//! <tool_call>
//! <function=example_function_name>
//! <parameter=example_parameter_1>
//! value_1
//! </parameter>
//! </function>
//! </tool_call>
//!
//! <IMPORTANT>
//! Reminder:
//! - Function calls MUST follow the specified format ...
//! </IMPORTANT>
//! {existing system content, optional}<|im_end|>
//! ```
//!
//! Assistant turn with tool_calls:
//!
//! ```text
//! <|im_start|>assistant
//! <think>
//!
//! </think>
//!
//! {optional natural-language reasoning}
//!
//! <tool_call>
//! <function=NAME>
//! <parameter=KEY1>
//! VALUE1
//! </parameter>
//! <parameter=KEY2>
//! {"nested":"json"}
//! </parameter>
//! </function>
//! </tool_call><|im_end|>
//! ```
//!
//! Tool response turn (role:"tool" — wrapped in user role per the template):
//!
//! ```text
//! <|im_start|>user
//! <tool_response>
//! {raw tool result text}
//! </tool_response><|im_end|>
//! ```
//!
//! Adjacent tool responses are batched into ONE user turn (separated by
//! `\n` between `<tool_response>` blocks).

use anyhow::{Result, anyhow};
use serde_json::{Map, Value as JsonValue};

use crate::chat_io::{
    AssistantToolCall, ChatTurn, ParsedResponse, ParsedToolCall, REASONING_CLOSE, REASONING_OPEN,
    ReasoningEffort, ResolvedToolChoice, ToolDef, split_reasoning_envelope,
};

// ─────────────────────────────────────────────────────────────────────────────
// Chat template renderer
// ─────────────────────────────────────────────────────────────────────────────

/// The long instruction block that follows the `<tools>...</tools>` section
/// in the system prompt. Copied verbatim from Qwen 3.6's chat_template.jinja
/// so the model recognizes the exact phrasing it was trained on.
const TOOL_INSTRUCTION_BLOCK: &str = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

/// Render the system prefix containing the `<tools>` block + instruction.
/// Optional existing system content is appended after the IMPORTANT block,
/// separated by `\n\n` — matches Qwen's template behavior.
///
/// `pub` (re-exported `#[doc(hidden)]` from the crate root) so the
/// `chat_render` fuzz target can reach it. The enclosing module stays private,
/// so this widens the API by exactly one function.
pub fn render_tools_system_block(
    tools: &[ToolDef<'_>],
    extra_system: Option<&str>,
    effort: Option<ReasoningEffort>,
) -> String {
    let mut s = String::new();
    s.push_str("<|im_start|>system\n");
    // Qwen 3.8 puts the reasoning-effort sentence at the very top of the system
    // block — BEFORE `# Tools`, while the user's own system text still goes
    // after the IMPORTANT block. That asymmetry is why the effort cannot simply
    // be prepended to `extra_system` by the caller.
    if let Some(text) = effort.and_then(ReasoningEffort::instructions) {
        s.push_str(text);
        s.push_str("\n\n");
    }
    s.push_str("# Tools\n\nYou have access to the following functions:\n\n<tools>");
    for tool in tools {
        s.push('\n');
        let obj = tool_def_to_json(tool);
        s.push_str(&serde_json::to_string(&obj).unwrap_or_else(|_| "{}".into()));
    }
    s.push_str("\n</tools>");
    s.push_str(TOOL_INSTRUCTION_BLOCK);
    if let Some(extra) = extra_system {
        let trimmed = extra.trim();
        if !trimmed.is_empty() {
            s.push_str("\n\n");
            s.push_str(trimmed);
        }
    }
    s.push_str("<|im_end|>\n");
    s
}

/// Render the no-tools system block: the effort sentence (if any), then the
/// client's system text (if any), in that order.
///
/// Returns the empty string when both are absent — Qwen 3.8 emits a system
/// block for the effort alone, but emits nothing at all when there is neither,
/// which is what every pre-3.8 checkpoint did unconditionally.
fn render_system_block(system: Option<&str>, effort: Option<ReasoningEffort>) -> String {
    let instructions = effort.and_then(ReasoningEffort::instructions);
    let system = system.filter(|s| !s.is_empty());
    if instructions.is_none() && system.is_none() {
        return String::new();
    }
    let mut s = String::from("<|im_start|>system\n");
    if let Some(text) = instructions {
        s.push_str(text);
        if system.is_some() {
            s.push_str("\n\n");
        }
    }
    if let Some(text) = system {
        s.push_str(text);
    }
    s.push_str("<|im_end|>\n");
    s
}

/// Convert a `ToolDef` to the JSON shape Qwen expects inside `<tools>` —
/// matches OpenAI's function-tool wire format (object with `name`,
/// `description`, `parameters`).
fn tool_def_to_json(tool: &ToolDef<'_>) -> JsonValue {
    let mut obj = Map::new();
    obj.insert("type".into(), JsonValue::String("function".into()));
    let mut func = Map::new();
    func.insert("name".into(), JsonValue::String(tool.name.to_string()));
    if let Some(desc) = tool.description {
        func.insert("description".into(), JsonValue::String(desc.to_string()));
    }
    if let Some(params) = tool.parameters {
        func.insert("parameters".into(), params.clone());
    } else {
        func.insert("parameters".into(), JsonValue::Object(Map::new()));
    }
    obj.insert("function".into(), JsonValue::Object(func));
    JsonValue::Object(obj)
}

/// Remove any `</think>` the model closed without having opened one itself.
///
/// A reasoning-first checkpoint sometimes writes a short trace and closes it
/// even when the prompt handed it an already-closed block — measured on
/// Qwen3.8-27B, `'Blue\n</think>\n\nBlue'` for a thinking-**off** request. Never
/// at temperature 0 (0/16) and rarely when sampling (1/32 at 0.8, 2/6 at the
/// 0.7 default), so it is the model's doing and not a prompt defect: the
/// rendered generation prompt ends with a closed `<think>\n\n</think>\n\n`.
///
/// Dropping it rather than re-reading it as a delimiter is deliberate. The
/// tool-aware parser has always dropped an unbalanced close, so this is what
/// makes the two paths agree — the same reply currently comes back differently
/// depending only on whether the request carried tools. And the text before the
/// tag cannot be moved to `reasoning` on the streaming surface, because it has
/// already gone out as content deltas; splitting non-streaming alone would make
/// the two surfaces disagree instead.
///
/// A *balanced* pair is left alone: a model quoting `<think>…</think>` in its
/// answer is showing the syntax, not delimiting anything.
pub fn strip_unbalanced_think_close(text: &str) -> std::borrow::Cow<'_, str> {
    let mut depth = 0usize;
    strip_think_close_tracking(text, &mut depth)
}

/// [`strip_unbalanced_think_close`] with the open-tag depth owned by the
/// caller, so a streaming splitter can carry it across deltas.
///
/// Chunk-local depth would be useless here: a balanced `<think>…</think>` the
/// model quotes almost always spans several decode steps, so every close would
/// look unbalanced and the protection would never fire.
fn strip_think_close_tracking<'a>(text: &'a str, depth: &mut usize) -> std::borrow::Cow<'a, str> {
    if !text.contains(REASONING_CLOSE) {
        *depth += text.matches(REASONING_OPEN).count();
        return std::borrow::Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    loop {
        let open = rest.find(REASONING_OPEN);
        let close = rest.find(REASONING_CLOSE);
        match (open, close) {
            (Some(o), c) if c.is_none_or(|c| o < c) => {
                *depth += 1;
                out.push_str(&rest[..o + REASONING_OPEN.len()]);
                rest = &rest[o + REASONING_OPEN.len()..];
            }
            (_, Some(c)) => {
                out.push_str(&rest[..c]);
                if *depth > 0 {
                    *depth -= 1;
                    out.push_str(REASONING_CLOSE);
                }
                rest = &rest[c + REASONING_CLOSE.len()..];
            }
            // No close left — nothing further to strip.
            (Some(_), None) | (None, None) => {
                out.push_str(rest);
                break;
            }
        }
    }
    std::borrow::Cow::Owned(out)
}

/// Split a plain (no-tools) turn generated with thinking ON into
/// `(reasoning, visible)`.
///
/// With thinking on the *prompt* ends at `<think>\n`, so the model's output
/// begins mid-trace and closes with a bare `</think>` — there is no opener in
/// the generated text to key on. Everything up to that close tag is the trace;
/// the `\n\n` the template pairs with the tag is part of the delimiter.
///
/// No close tag means the model spent its whole budget thinking and never
/// reached an answer. Reporting that as reasoning with empty visible text is
/// the truthful shape, and it is what the tool path's parser already does.
pub fn split_open_think(text: &str) -> (&str, &str) {
    let Some(idx) = text.find("</think>") else {
        return (text, "");
    };
    let mut rest = &text[idx + "</think>".len()..];
    for sep in ["\n\n", "\n"] {
        if let Some(stripped) = rest.strip_prefix(sep) {
            rest = stripped;
            break;
        }
    }
    (&text[..idx], rest)
}

/// Streaming counterpart of [`split_open_think`] for the plain path's
/// `on_token` callback, which sees deltas and cannot wait for the whole turn.
///
/// Holds back any tail that could still grow into `</think>` so a tag split
/// across two decode steps is not emitted as visible text.
pub struct ThinkingSplitter {
    phase: Phase,
    held: String,
    /// Open `<think>` tags the model has emitted itself, carried across deltas.
    open_depth: usize,
}

#[derive(PartialEq)]
enum Phase {
    /// Inside the block the prompt opened — bytes are the trace.
    Trace,
    /// The close tag has arrived but its trailing separator may not have. The
    /// tag and the `\n\n` can land in different decode steps, so the separator
    /// has to be eaten lazily or it reaches the client as a leading blank line.
    AfterTag,
    /// Everything from here is the answer.
    Visible,
}

impl ThinkingSplitter {
    /// `thinking_open` mirrors [`qwen3_generation_header`]: true when the
    /// prompt left the block open and the model is writing its trace now.
    pub fn new(thinking_open: bool) -> Self {
        Self {
            phase: if thinking_open {
                Phase::Trace
            } else {
                Phase::Visible
            },
            held: String::new(),
            open_depth: 0,
        }
    }

    /// Feed one decoded delta; returns `(reasoning_delta, visible_delta)`.
    pub fn feed(&mut self, delta: &str) -> (String, String) {
        if self.phase == Phase::Visible {
            // A reasoning-first checkpoint sometimes closes a block it never
            // opened, even on a thinking-off turn. Hold back only what could
            // still grow into the tag (≤7 bytes, so no measurable latency) and
            // drop it when it completes — see `strip_unbalanced_think_close`
            // for why dropping rather than re-splitting.
            self.held.push_str(delta);
            let keep = partial_tail_len(&self.held, REASONING_CLOSE);
            let flushable: String = self.held.drain(..self.held.len() - keep).collect();
            let visible = strip_think_close_tracking(&flushable, &mut self.open_depth);
            return (String::new(), visible.into_owned());
        }
        self.held.push_str(delta);
        let mut reasoning = String::new();
        if self.phase == Phase::Trace {
            match self.held.find("</think>") {
                Some(idx) => {
                    reasoning = self.held[..idx].to_string();
                    self.held.drain(..idx + "</think>".len());
                    self.phase = Phase::AfterTag;
                }
                None => {
                    // Keep back only what could still become the close tag.
                    let keep = partial_tail_len(&self.held, "</think>");
                    let flush: String = self.held.drain(..self.held.len() - keep).collect();
                    return (flush, String::new());
                }
            }
        }
        // `Phase::AfterTag`: eat at most one separator, but only once enough
        // bytes are in hand to tell `\n` from `\n\n` — and the tag itself may
        // have ended the chunk, so an empty hand means "not yet", not "none".
        if self.held.is_empty() || self.held == "\n" {
            return (reasoning, String::new());
        }
        if self.held.starts_with("\n\n") {
            self.held.drain(.."\n\n".len());
        } else if self.held.starts_with('\n') {
            self.held.drain(.."\n".len());
        }
        self.phase = Phase::Visible;
        (reasoning, std::mem::take(&mut self.held))
    }

    /// Tails still held when decoding stops, as `(reasoning, visible)`.
    ///
    /// In `Trace` the held bytes were never closed, so they were all trace. In
    /// `Visible` they are a partial close tag that never completed and belong
    /// to the answer. In `AfterTag` they can only be the delimiter's own
    /// separator, which is dropped.
    pub fn finish(self) -> (String, String) {
        match self.phase {
            Phase::Trace => (self.held, String::new()),
            Phase::Visible => (String::new(), self.held),
            Phase::AfterTag => (String::new(), String::new()),
        }
    }
}

/// Length of the longest suffix of `s` that is a proper prefix of `needle`.
fn partial_tail_len(s: &str, needle: &str) -> usize {
    (1..needle.len())
        .rev()
        .find(|&n| {
            s.len() >= n && s.is_char_boundary(s.len() - n) && needle.starts_with(&s[s.len() - n..])
        })
        .unwrap_or(0)
}

/// Render the assistant-turn opening (without tool_calls):
///   `<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{visible}`
///
/// `reasoning` comes out of the turn's own text — see
/// [`split_reasoning_envelope`]. A turn that carries no trace splits to an
/// empty one and renders `<think>\n\n</think>\n\n{content}`, byte for byte what
/// this emitted before the trace could be returned at all.
///
/// Splitting is also what stops a client from getting *two* blocks: with
/// `LUMEN_REASONING_IN_CONTENT=1` the trace is already inside `content`, and
/// prefixing an empty block in front of it produced
/// `<think>\n\n</think>\n\n<think>\n…` — a shape no Qwen template ever emits.
///
/// The block is emitted unconditionally, not gated on `preserve_thinking`.
/// Every checkpoint's *generation* prompt ends with one, so this is the
/// rendering that reproduces the tokens the model was actually fed; 3.6's
/// template drops it from history instead, which is self-consistent for
/// upstream but would cost this path the KV it can otherwise extend.
fn render_assistant_open(content: &str, _thinking: bool) -> String {
    let (reasoning, visible) = split_reasoning_envelope(content);
    let mut s = String::new();
    s.push_str("<|im_start|>assistant\n");
    s.push_str("<think>\n");
    s.push_str(reasoning);
    s.push_str("\n</think>\n\n");
    s.push_str(visible);
    s
}

/// Render one assistant tool_call as the nested-XML form. `is_first`
/// controls whether a leading `\n\n` separator (after content) is needed.
fn render_tool_call(call: &AssistantToolCall<'_>, is_first: bool, has_content: bool) -> String {
    let mut s = String::new();
    if is_first {
        if has_content {
            s.push_str("\n\n");
        }
    } else {
        s.push('\n');
    }
    s.push_str("<tool_call>\n<function=");
    s.push_str(call.name);
    s.push_str(">\n");
    if let JsonValue::Object(map) = call.arguments {
        for (key, val) in map {
            s.push_str("<parameter=");
            s.push_str(key);
            s.push_str(">\n");
            s.push_str(&value_to_param_str(val));
            s.push_str("\n</parameter>\n");
        }
    }
    s.push_str("</function>\n</tool_call>");
    s
}

/// Render a parameter value the way Qwen's template does: strings are raw,
/// everything else is `tojson`-serialized (objects, arrays, numbers, bools,
/// null).
fn value_to_param_str(val: &JsonValue) -> String {
    match val {
        JsonValue::String(s) => s.clone(),
        other => serde_json::to_string(other).unwrap_or_else(|_| "null".into()),
    }
}

/// Build the full chat prompt from flat `(role, content)` messages + tool
/// definitions. This is the legacy entry point — no assistant tool_calls
/// or tool-response history. Used by the non-structured codepath.
pub fn format_qwen3_chat_with_tools(
    messages: &[(String, String)],
    thinking: bool,
    tools: &[ToolDef<'_>],
    effort: Option<ReasoningEffort>,
) -> String {
    let mut s = String::new();
    let mut idx = 0;

    // System-prefix handling: if tools present, emit the `<tools>` block;
    // pull any leading system message's content into the tail of that block.
    if !tools.is_empty() {
        let leading_system = if messages
            .first()
            .map(|(role, _)| role.eq_ignore_ascii_case("system"))
            .unwrap_or(false)
        {
            idx = 1;
            Some(messages[0].1.as_str())
        } else {
            None
        };
        s.push_str(&render_tools_system_block(tools, leading_system, effort));
    } else if let Some((role, content)) = messages.first()
        && role.eq_ignore_ascii_case("system")
    {
        s.push_str(&render_system_block(Some(content), effort));
        idx = 1;
    } else {
        // No system message: 3.8 still emits a system block holding just the
        // effort sentence. Renders to nothing when there is no effort, which is
        // the pre-3.8 behaviour byte for byte.
        s.push_str(&render_system_block(None, effort));
    }

    while idx < messages.len() {
        let (role, content) = &messages[idx];
        let role_lower = role.to_ascii_lowercase();
        match role_lower.as_str() {
            "user" => {
                s.push_str("<|im_start|>user\n");
                s.push_str(content);
                s.push_str("<|im_end|>\n");
            }
            "assistant" => {
                s.push_str(&render_assistant_open(content, thinking));
                s.push_str("<|im_end|>\n");
            }
            _ => {
                // Unknown roles are dropped — caller is responsible for
                // sending structured `ChatTurn::Tool` via the
                // `_from_history` entry point.
            }
        }
        idx += 1;
    }

    // Generation prompt
    s.push_str(qwen3_generation_header(thinking));
    s
}

/// The trailing **generation header** that `format_qwen3_chat_with_tools`
/// (and the `_from_history` variant) append after the last rendered message.
///
/// This is the part the NEXT turn replaces with the assistant's actual
/// response, so a conversation-boundary prefix-cache snapshot must stop
/// BEFORE it. Exposing it lets the generic prefix cache compute the reusable
/// boundary as `full_ids.len() - tokenize(header).len()` (verified by
/// `full_ids.ends_with(header_ids)`) — no model-specific token counting.
///
/// Reusable pattern: every chat family exposes its own `*_generation_header`
/// so the shared conversation-boundary prefix-cache works for any model that
/// renders `<stable conversation>` + `<generation header>`.
pub fn qwen3_generation_header(thinking: bool) -> &'static str {
    if thinking {
        "<|im_start|>assistant\n<think>\n"
    } else {
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    }
}

/// Build the full chat prompt from structured `ChatTurn[]` history with
/// tool_calls + tool_response turns. Mirrors the Jinja template behavior:
///
///  - leading System turn (if any) merges into the `<tools>` system block
///    when tools are present
///  - Assistant turns may carry `tool_calls`; each gets rendered as a
///    nested `<tool_call><function=...>...</function></tool_call>` block
///    appended to the assistant's visible text
///  - Adjacent Tool turns are batched into one `<|im_start|>user`
///    containing back-to-back `<tool_response>` blocks (no separator
///    `<|im_end|>` until the run ends)
pub fn format_qwen3_chat_with_tools_from_history(
    turns: &[ChatTurn<'_>],
    thinking: bool,
    tools: &[ToolDef<'_>],
    effort: Option<ReasoningEffort>,
) -> String {
    let mut s = String::new();
    let mut idx = 0;

    // System prefix
    let leading_system = if matches!(turns.first(), Some(ChatTurn::System(_))) {
        idx = 1;
        match turns[0] {
            ChatTurn::System(t) => Some(t),
            _ => None,
        }
    } else {
        None
    };
    if !tools.is_empty() {
        s.push_str(&render_tools_system_block(tools, leading_system, effort));
    } else if leading_system.is_some() || effort.is_some() {
        s.push_str(&render_system_block(leading_system, effort));
    }

    // Body
    let mut in_tool_run = false;
    while idx < turns.len() {
        let turn = &turns[idx];
        // Close an open tool-response run when we hit a non-Tool turn.
        if !matches!(turn, ChatTurn::Tool { .. }) && in_tool_run {
            s.push_str("<|im_end|>\n");
            in_tool_run = false;
        }
        match turn {
            ChatTurn::System(_) => {
                // Mid-history system messages are not supported by Qwen's
                // template; drop them.
            }
            ChatTurn::User(text) => {
                s.push_str("<|im_start|>user\n");
                s.push_str(text);
                s.push_str("<|im_end|>\n");
            }
            ChatTurn::Assistant { text, tool_calls } => {
                let content = *text;
                // The separator before the first `<tool_call>` follows the
                // *visible* reply, so it has to be decided after the reasoning
                // envelope is off — a turn that is pure trace plus tool calls
                // has no visible text to separate from.
                let has_content = !split_reasoning_envelope(content).1.trim().is_empty();
                s.push_str(&render_assistant_open(content, thinking));
                if !tool_calls.is_empty() {
                    for (i, call) in tool_calls.iter().enumerate() {
                        s.push_str(&render_tool_call(call, i == 0, has_content));
                    }
                }
                s.push_str("<|im_end|>\n");
            }
            ChatTurn::Tool { content, .. } => {
                if !in_tool_run {
                    s.push_str("<|im_start|>user");
                    in_tool_run = true;
                }
                s.push_str("\n<tool_response>\n");
                s.push_str(content);
                s.push_str("\n</tool_response>");
            }
        }
        idx += 1;
    }
    if in_tool_run {
        s.push_str("<|im_end|>\n");
    }

    // Generation prompt
    s.push_str(qwen3_generation_header(thinking));
    s
}

/// Phase 2: tool_choice prefill string. Auto/None → empty; Required →
/// `<tool_call>\n<function=`; Tool(name) → fully-resolved opener that
/// forces the model to emit a body for that specific function. Engine
/// must append this AFTER the generation prompt (`<|im_start|>assistant\n
/// <think>...</think>\n\n`).
pub fn qwen35_tool_choice_prefill_str(choice: &ResolvedToolChoice<'_>) -> String {
    match choice {
        ResolvedToolChoice::Auto | ResolvedToolChoice::None => String::new(),
        ResolvedToolChoice::Required => "<tool_call>\n<function=".to_string(),
        ResolvedToolChoice::Tool(name) => {
            format!("<tool_call>\n<function={name}>\n")
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Response parser
// ─────────────────────────────────────────────────────────────────────────────

/// Events emitted by `Qwen35ResponseParser` during incremental decode.
/// Mirrors the family-agnostic event shape used by the Gemma 4 path so
/// the backend can forward them to `BackendStreamEvent::ToolCallStart` /
/// `BackendStreamEvent::Text` without re-deriving meaning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Qwen35ParseEvent {
    /// User-visible text delta (outside any `<tool_call>` span and
    /// outside any `<think>` reasoning span — those are stripped).
    Text(String),
    /// Reasoning delta — bytes from inside the `<think>` span.
    ///
    /// The trace was always accumulated for `finish()`, but never surfaced as
    /// it arrived, so a *streaming* tool-aware turn dropped it entirely: the
    /// Anthropic route had no `thinking` block to open and the OpenAI route no
    /// `delta.reasoning` to send. Measured against the running server — a
    /// `thinking`-enabled request with tools attached streamed
    /// `text@0, tool_use@1` and no trace at all, while the same request without
    /// tools streamed `thinking@0, text@1`.
    Reasoning(String),
    /// The opening of a tool_call has been observed and the function
    /// name has been fully accumulated. Fires exactly once per
    /// `<tool_call>...</tool_call>` block, between the `<function=NAME>`
    /// line and the first `<parameter=>` block.
    ToolCallStart { name: String },
}

/// Incremental parser for the assistant decode stream. Accumulates text,
/// detects `<tool_call>` blocks, extracts arguments as a JSON object, and
/// surfaces them through `finish() -> ParsedResponse`. Streaming events
/// (Text deltas + ToolCallStart) are emitted by `feed()` so the backend
/// can forward them upstream as `BackendStreamEvent`.
///
/// Parsing strategy: scan the running buffer on each `feed()` for the
/// next state transition. The buffer keeps only the unprocessed suffix
/// (everything before the current scan cursor is finalized either as a
/// Text emission or absorbed into a tool_call accumulator).
pub struct Qwen35ResponseParser {
    state: State,
    buffer: String,
    reasoning: String,
    visible: String,
    in_think: bool,
    parsed_calls: Vec<ParsedToolCall>,
    pending: Option<PendingCall>,
}

#[derive(Debug)]
enum State {
    /// Outside any tool_call / think block — text goes to `visible`.
    Visible,
    /// Inside a `<think>...</think>` block — text goes to `reasoning`.
    Thinking,
    /// Inside a `<tool_call>...</tool_call>` block but BEFORE the
    /// `<function=NAME>\n` line is fully parsed. `ToolCallStart` will
    /// fire on transition out of this state.
    InToolCallHeader,
    /// Inside a tool_call body — between parameters / waiting for
    /// next `<parameter=>` or `</function>`.
    InToolCallBody,
    /// Inside a `<parameter=KEY>\nVALUE...` block — accumulating
    /// value bytes until `</parameter>` arrives. `key` is the
    /// already-parsed parameter name that will be inserted into the
    /// pending call's args once the value closes.
    InParameterValue { key: String },
}

#[derive(Debug)]
struct PendingCall {
    name: String,
    args: Map<String, JsonValue>,
}

impl Qwen35ResponseParser {
    pub fn new() -> Self {
        Self {
            state: State::Visible,
            buffer: String::new(),
            reasoning: String::new(),
            visible: String::new(),
            in_think: false,
            parsed_calls: Vec::new(),
            pending: None,
        }
    }

    /// A parser for a turn whose **prompt** already opened the thinking block.
    ///
    /// `qwen3_generation_header(true)` ends at `<think>\n`, so with thinking on
    /// the model writes its trace with no opener of its own and closes with a
    /// bare `</think>`. Starting in [`State::Visible`] therefore never entered
    /// the thinking state at all: the whole chain-of-thought was accumulated as
    /// visible text, `reasoning` came back empty, and the stray close tag was
    /// handed to the client in the middle of the answer.
    ///
    /// The state cannot be recovered later from the token stream. Visible text
    /// is emitted as it arrives, so by the time the `</think>` shows up the
    /// trace has already been streamed out as content deltas — which is why
    /// this is a constructor and not a fixup in `try_advance_visible`.
    pub fn with_thinking_open() -> Self {
        Self {
            state: State::Thinking,
            in_think: true,
            ..Self::new()
        }
    }

    /// Feed a decoded text delta. Returns events to surface upstream.
    /// Visible text deltas are returned VERBATIM (already split at any
    /// `<tool_call>` / `<think>` boundary) so the backend can pipe them
    /// straight to `BackendStreamEvent::Text` without further parsing.
    pub fn feed(&mut self, delta: &str) -> Vec<Qwen35ParseEvent> {
        self.buffer.push_str(delta);
        let mut events = Vec::new();
        loop {
            let progressed = match &self.state {
                State::Visible => self.try_advance_visible(&mut events),
                State::Thinking => self.try_advance_thinking(&mut events),
                State::InToolCallHeader => self.try_advance_header(&mut events),
                State::InToolCallBody => self.try_advance_body(),
                State::InParameterValue { .. } => self.try_advance_value(),
            };
            if !progressed {
                break;
            }
        }
        events
    }

    /// Consume the parser and return the final structured response.
    pub fn finish(self) -> ParsedResponse {
        ParsedResponse {
            visible: self.visible,
            reasoning: self.reasoning,
            tool_calls: self.parsed_calls,
        }
    }

    #[allow(dead_code)]
    // no caller. NOTE: this never ran, so the `defense-in-depth` it claims does
    // not exist — the engine's downgrade-to-Auto is the only check in the path.
    /// Total accumulated visible bytes so far (live view — for
    /// `completion_tokens_with_tools` post-decode accounting).
    pub fn visible_so_far(&self) -> &str {
        &self.visible
    }

    /// How many tool calls have been **fully** parsed so far — a live view of
    /// what `finish()` would return.
    ///
    /// Increments only when `</tool_call>` is consumed, so a call that is
    /// still mid-body does not count. This is the decode loop's stop signal
    /// for `parallel_tool_calls: false`
    /// ([`ToolCalls::must_stop_after_completed_calls`]); Qwen's closer is
    /// literal text rather than a special token, so there is no id to watch
    /// the way the Gemma 4 loop watches `<tool_call|>`.
    ///
    /// [`ToolCalls::must_stop_after_completed_calls`]: crate::grammar::ToolCalls::must_stop_after_completed_calls
    pub fn completed_calls(&self) -> usize {
        self.parsed_calls.len()
    }

    /// Force-required-params hook (opt-in via the decode loop).
    ///
    /// When the parser is sitting at a CLEAN decision point inside a
    /// tool-call body — i.e. right after the `<function=NAME>` opener or a
    /// just-closed `</parameter>`, with no partial delimiter buffered — and
    /// the current function `NAME` has a REQUIRED parameter that has not yet
    /// been emitted, return the first such missing key (schema order). The
    /// decode loop injects `<parameter=KEY>\n`, forcing the model to supply
    /// the value rather than close the call with missing required args
    /// (`<function=read></function>` → "path is required").
    ///
    /// Returns `None` when not in a body, when a partial token is buffered
    /// (the model is mid-emitting and must not be pre-empted), or when every
    /// required param for the current call is already present.
    pub fn next_required_param_to_force(
        &self,
        required: &std::collections::HashMap<String, Vec<String>>,
    ) -> Option<String> {
        if !matches!(self.state, State::InToolCallBody) {
            return None;
        }
        if !self.buffer.trim().is_empty() {
            return None;
        }
        let pending = self.pending.as_ref()?;
        let reqs = required.get(&pending.name)?;
        reqs.iter()
            .find(|k| !pending.args.contains_key(k.as_str()))
            .cloned()
    }

    fn try_advance_visible(&mut self, events: &mut Vec<Qwen35ParseEvent>) -> bool {
        // Look for the next interesting delimiter: <tool_call> or <think>
        // or </think>. Whatever comes first wins.
        let candidates: &[(&'static str, DelimKind)] = &[
            ("<tool_call>", DelimKind::ToolCallOpen),
            ("<think>", DelimKind::ThinkOpen),
            ("</think>", DelimKind::ThinkClose),
        ];
        let (delim_idx, delim_kind, delim_str) = self.next_delim(candidates);
        if let (Some(idx), Some(kind), Some(needle)) = (delim_idx, delim_kind, delim_str) {
            // Emit the prefix as visible text (if non-empty).
            let prefix = self.buffer[..idx].to_string();
            if !prefix.is_empty() {
                self.visible.push_str(&prefix);
                events.push(Qwen35ParseEvent::Text(prefix));
            }
            let consumed = idx + needle.len();
            self.buffer.drain(..consumed);
            match kind {
                DelimKind::ToolCallOpen => {
                    self.state = State::InToolCallHeader;
                }
                DelimKind::ThinkOpen => {
                    self.state = State::Thinking;
                    self.in_think = true;
                }
                DelimKind::ThinkClose => {
                    // Unbalanced close — model emitted </think> without
                    // a prior <think>; treat as no-op (drop the tag).
                }
            }
            true
        } else {
            // No full delimiter found. Emit safely as visible text
            // anything that cannot possibly be the start of a partial
            // delimiter we're tracking. Keep the suffix that COULD be a
            // partial `<…` so future feed() may complete it.
            let safe_len = self.safe_visible_len();
            if safe_len > 0 {
                let chunk: String = self.buffer.drain(..safe_len).collect();
                self.visible.push_str(&chunk);
                events.push(Qwen35ParseEvent::Text(chunk));
                // Loop again in case the remaining buffer now starts
                // with a full delimiter we previously missed.
                true
            } else {
                false
            }
        }
    }

    fn try_advance_thinking(&mut self, events: &mut Vec<Qwen35ParseEvent>) -> bool {
        if let Some(idx) = self.buffer.find("</think>") {
            let prefix = self.buffer[..idx].to_string();
            self.reasoning.push_str(&prefix);
            if !prefix.is_empty() {
                events.push(Qwen35ParseEvent::Reasoning(prefix.clone()));
            }
            let mut consumed = idx + "</think>".len();
            // The template writes `'</think>\n\n' + content`, so the blank line
            // is part of the delimiter and not the start of the answer. Taking
            // it here is what stops a thinking-on reply reaching the client as
            // `"\n\nRed"`. At most one separator, so an answer that genuinely
            // opens with a blank line keeps it.
            for sep in ["\n\n", "\n"] {
                if self.buffer[consumed..].starts_with(sep) {
                    consumed += sep.len();
                    break;
                }
            }
            self.buffer.drain(..consumed);
            self.state = State::Visible;
            true
        } else {
            // Buffer might be split mid-delimiter. Drain everything
            // except a trailing `<` (which could be the start of
            // `</think>`).
            let keep = self.partial_suffix_len("</think>");
            if self.buffer.len() > keep {
                let chunk: String = self.buffer.drain(..self.buffer.len() - keep).collect();
                self.reasoning.push_str(&chunk);
                events.push(Qwen35ParseEvent::Reasoning(chunk));
                true
            } else {
                false
            }
        }
    }

    fn try_advance_header(&mut self, events: &mut Vec<Qwen35ParseEvent>) -> bool {
        // We're between `<tool_call>` and the function name + body.
        // Expected next: a `<function=NAME>\n` block. Strip leading
        // whitespace/newlines, then look for `<function=NAME>` followed
        // by `\n`.
        // Skip leading whitespace
        let leading_ws = self
            .buffer
            .chars()
            .take_while(|c| c.is_whitespace())
            .count();
        if leading_ws > 0 {
            // Drain the whitespace bytes (char count == byte count for ASCII).
            self.buffer.drain(..leading_ws);
        }
        let fn_open = "<function=";
        if let Some(idx) = self.buffer.find(fn_open) {
            // Anything before the `<function=` should be dropped — Qwen
            // never emits content between `<tool_call>` and `<function=`.
            if idx > 0 {
                self.buffer.drain(..idx);
            }
            // Now buffer starts with `<function=`. Look for closing `>`.
            if let Some(gt) = self.buffer.find('>') {
                let name = self.buffer[fn_open.len()..gt].trim().to_string();
                // Drain `<function=NAME>` itself.
                self.buffer.drain(..gt + 1);
                // Skip a single leading `\n` if present.
                if self.buffer.starts_with('\n') {
                    self.buffer.drain(..1);
                }
                self.pending = Some(PendingCall {
                    name: name.clone(),
                    args: Map::new(),
                });
                events.push(Qwen35ParseEvent::ToolCallStart { name });
                self.state = State::InToolCallBody;
                true
            } else {
                // `<function=` opened but closing `>` not in buffer yet.
                false
            }
        } else {
            // `<function=` not yet present. If buffer is large and has
            // no partial prefix of `<function=`, drop to avoid runaway
            // memory — but in practice this state is brief.
            let keep = self.partial_suffix_len(fn_open).max(self.buffer.len());
            // Keep everything for now; the model's next tokens should
            // produce `<function=`. If buffer grows pathologically, the
            // engine's max_tokens will cap it.
            let _ = keep;
            false
        }
    }

    fn try_advance_body(&mut self) -> bool {
        // Expected: a sequence of `<parameter=KEY>` openers (which
        // transition into `InParameterValue { key }`) interleaved
        // with `</function>\n</tool_call>` that closes the call.
        let param_open = "<parameter=";
        let fn_close = "</function>";
        let tool_close = "</tool_call>";
        // Skip leading whitespace
        let leading = self
            .buffer
            .chars()
            .take_while(|c| c.is_whitespace())
            .count();
        if leading > 0 {
            self.buffer.drain(..leading);
        }
        if self.buffer.starts_with(param_open) {
            if let Some(gt) = self.buffer.find('>') {
                let key = self.buffer[param_open.len()..gt].trim().to_string();
                self.buffer.drain(..gt + 1);
                // Strip one leading newline before VALUE
                if self.buffer.starts_with('\n') {
                    self.buffer.drain(..1);
                }
                self.state = State::InParameterValue { key };
                true
            } else {
                // Need more tokens for the `>`.
                false
            }
        } else if self.buffer.starts_with(fn_close) {
            self.buffer.drain(..fn_close.len());
            // Don't try to consume `</tool_call>` here — chunked feed
            // means it may not be in the buffer yet. The next loop
            // iteration's `tool_close` branch handles it.
            true
        } else if self.buffer.starts_with(tool_close) {
            self.buffer.drain(..tool_close.len());
            if let Some(pending) = self.pending.take() {
                self.parsed_calls.push(ParsedToolCall {
                    name: pending.name,
                    arguments: JsonValue::Object(pending.args),
                });
            }
            self.state = State::Visible;
            true
        } else if self.buffer.is_empty() {
            false
        } else {
            // Unrecognized prefix — could be partial `<parameter=` /
            // `</function>` / `</tool_call>` waiting for more tokens.
            // Keep the maximum partial suffix; otherwise drop one byte
            // to make progress (handles stray whitespace / noise).
            let keep = self.partial_suffix_len(param_open).max(
                self.partial_suffix_len(fn_close)
                    .max(self.partial_suffix_len(tool_close)),
            );
            if self.buffer.len() > keep {
                self.buffer.drain(..1);
                true
            } else {
                false
            }
        }
    }

    /// Inside `<parameter=KEY>\n...` — accumulate value bytes until
    /// `</parameter>` arrives, then move back to `InToolCallBody`.
    /// The value's framing newlines (one leading, one trailing) are
    /// stripped to match Qwen's template:
    ///   `<parameter=KEY>\nVALUE\n</parameter>`
    /// → captured VALUE excludes the framing `\n`s but preserves any
    /// internal newlines in multi-line values.
    fn try_advance_value(&mut self) -> bool {
        let param_close = "</parameter>";
        if let Some(end) = self.buffer.find(param_close) {
            let mut raw = self.buffer[..end].to_string();
            // Strip exactly one leading and one trailing `\n` framing.
            if raw.starts_with('\n') {
                raw.remove(0);
            }
            if raw.ends_with('\n') {
                raw.pop();
            }
            self.buffer.drain(..end + param_close.len());
            let key = match std::mem::replace(&mut self.state, State::InToolCallBody) {
                State::InParameterValue { key } => key,
                other => {
                    self.state = other;
                    return false;
                }
            };
            if let Some(pending) = self.pending.as_mut() {
                pending.args.insert(key, parse_param_value(&raw));
            }
            true
        } else {
            // Wait for the close tag to arrive. We do NOT drain — the
            // accumulated value bytes are needed at close time.
            false
        }
    }

    /// Length of the largest suffix of `self.buffer` that is a prefix of
    /// `needle`. Used to know how much of the buffer to retain while
    /// waiting for the rest of a delimiter to arrive.
    fn partial_suffix_len(&self, needle: &str) -> usize {
        let buf = self.buffer.as_bytes();
        let needle = needle.as_bytes();
        let max = buf.len().min(needle.len().saturating_sub(1));
        for k in (1..=max).rev() {
            if buf[buf.len() - k..] == needle[..k] {
                return k;
            }
        }
        0
    }

    /// Bytes safe to emit as visible text without risking that they're
    /// the start of a tracked delimiter. Equal to `buffer.len() -
    /// max(partial suffix of any tracked open delimiter)`.
    fn safe_visible_len(&self) -> usize {
        let max_partial = [
            self.partial_suffix_len("<tool_call>"),
            self.partial_suffix_len("<think>"),
            self.partial_suffix_len("</think>"),
        ]
        .into_iter()
        .max()
        .unwrap_or(0);
        self.buffer.len().saturating_sub(max_partial)
    }

    /// Find the next occurrence (smallest index) of any delimiter in
    /// `candidates`. Returns `(index, kind, needle)` or all-`None`.
    fn next_delim(
        &self,
        candidates: &[(&'static str, DelimKind)],
    ) -> (Option<usize>, Option<DelimKind>, Option<&'static str>) {
        let mut best: Option<(usize, DelimKind, &'static str)> = None;
        for (needle, kind) in candidates {
            if let Some(i) = self.buffer.find(needle) {
                match best {
                    Some((bi, _, _)) if bi <= i => {}
                    _ => best = Some((i, *kind, *needle)),
                }
            }
        }
        match best {
            Some((i, k, n)) => (Some(i), Some(k), Some(n)),
            None => (None, None, None),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum DelimKind {
    ToolCallOpen,
    ThinkOpen,
    ThinkClose,
}

/// Parse a parameter value the way the model emits it: strings are raw,
/// JSON-shaped values (objects/arrays/numbers/bools/null) are
/// `serde_json::from_str`-able. We attempt JSON first; on failure we
/// fall back to the raw string.
fn parse_param_value(raw: &str) -> JsonValue {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return JsonValue::String(String::new());
    }
    // Only try JSON for values that LOOK like JSON to avoid swallowing
    // bare words ("hello") that serde_json::from_str would reject.
    let first = trimmed.as_bytes()[0];
    let looks_json = matches!(first, b'{' | b'[' | b'"' | b't' | b'f' | b'n')
        || first.is_ascii_digit()
        || first == b'-';
    if looks_json && let Ok(v) = serde_json::from_str::<JsonValue>(trimmed) {
        return v;
    }
    JsonValue::String(raw.to_string())
}

#[allow(dead_code)]
// no caller. NOTE: this never ran, so the `defense-in-depth` it claims does
// not exist — the engine's downgrade-to-Auto is the only check in the path.
/// Reject obviously-malformed tool_choice prefill for `Tool(name)` where
/// the requested name doesn't appear in the tool defs. Engine layer
/// already downgrades to Auto in that case, but defense-in-depth.
pub fn validate_tool_choice_against_defs<'a>(
    choice: &ResolvedToolChoice<'a>,
    tools: &[ToolDef<'_>],
) -> Result<()> {
    if let ResolvedToolChoice::Tool(name) = choice
        && !tools.iter().any(|t| t.name == *name)
    {
        return Err(anyhow!("tool_choice references unknown function: {name}"));
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn weather_tool() -> JsonValue {
        json!({
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["c", "f"]}
            },
            "required": ["city"]
        })
    }

    #[test]
    fn render_system_tools_block_when_no_existing_system() {
        let params = weather_tool();
        let tools = vec![ToolDef {
            name: "get_weather",
            description: Some("Get weather"),
            parameters: Some(&params),
            response: None,
        }];
        let messages = vec![("user".into(), "what is the weather?".into())];
        let s = format_qwen3_chat_with_tools(&messages, false, &tools, None);
        assert!(s.contains("<|im_start|>system\n# Tools"));
        assert!(s.contains("<tools>"));
        assert!(s.contains("\"name\":\"get_weather\""));
        assert!(s.contains("</tools>"));
        assert!(s.contains("<IMPORTANT>"));
        assert!(s.contains("<|im_start|>user\nwhat is the weather?<|im_end|>"));
        assert!(s.ends_with("<think>\n\n</think>\n\n"));
    }

    #[test]
    fn force_required_param_detects_missing_then_clears() {
        let mut required = std::collections::HashMap::new();
        required.insert("read".to_string(), vec!["path".to_string()]);

        // Model opens the function but emits NO parameter → force `path`.
        let mut p = Qwen35ResponseParser::new();
        let _ = p.feed("<tool_call>\n<function=read>\n");
        assert_eq!(
            p.next_required_param_to_force(&required).as_deref(),
            Some("path")
        );

        // Once the param is supplied, nothing left to force.
        let _ = p.feed("<parameter=path>\nCargo.toml\n</parameter>\n");
        assert_eq!(p.next_required_param_to_force(&required), None);

        // Mid-emitting a partial `<parameter=` → do NOT pre-empt the model.
        let mut p2 = Qwen35ResponseParser::new();
        let _ = p2.feed("<tool_call>\n<function=read>\n<param");
        assert_eq!(p2.next_required_param_to_force(&required), None);

        // A tool with no required entry is never forced.
        let mut p3 = Qwen35ResponseParser::new();
        let _ = p3.feed("<tool_call>\n<function=todo_write>\n");
        assert_eq!(p3.next_required_param_to_force(&required), None);
    }

    #[test]
    fn render_system_tools_merges_existing_system_content() {
        let params = weather_tool();
        let tools = vec![ToolDef {
            name: "get_weather",
            description: None,
            parameters: Some(&params),
            response: None,
        }];
        let messages = vec![
            ("system".into(), "You are a helpful assistant.".into()),
            ("user".into(), "weather?".into()),
        ];
        let s = format_qwen3_chat_with_tools(&messages, false, &tools, None);
        // System block should contain both the tools block AND the
        // existing system content appended after </IMPORTANT>.
        assert!(s.contains("</IMPORTANT>\n\nYou are a helpful assistant.<|im_end|>"));
    }

    #[test]
    fn render_without_tools_is_plain_im_format() {
        let messages = vec![
            ("system".into(), "You are X.".into()),
            ("user".into(), "hi".into()),
        ];
        let s = format_qwen3_chat_with_tools(&messages, false, &[], None);
        assert!(s.starts_with("<|im_start|>system\nYou are X.<|im_end|>"));
        assert!(s.contains("<|im_start|>user\nhi<|im_end|>"));
        assert!(!s.contains("<tools>"));
    }

    #[test]
    fn render_assistant_tool_call_from_history() {
        let params = weather_tool();
        let tools = vec![ToolDef {
            name: "get_weather",
            description: None,
            parameters: Some(&params),
            response: None,
        }];
        let args = json!({"city": "Seoul", "unit": "c"});
        let tc = vec![AssistantToolCall {
            id: "call_1",
            name: "get_weather",
            arguments: &args,
        }];
        let turns = vec![
            ChatTurn::User("weather?"),
            ChatTurn::Assistant {
                text: "",
                tool_calls: tc.as_slice(),
            },
            ChatTurn::Tool {
                tool_call_id: "call_1",
                name: Some("get_weather"),
                content: "{\"temp\":15}",
            },
            ChatTurn::User("anything else?"),
        ];
        let s = format_qwen3_chat_with_tools_from_history(&turns, false, &tools, None);
        assert!(s.contains("<tool_call>\n<function=get_weather>"));
        assert!(s.contains("<parameter=city>\nSeoul\n</parameter>"));
        assert!(s.contains("<parameter=unit>\nc\n</parameter>"));
        assert!(s.contains("</function>\n</tool_call>"));
        // Tool response wrapped in user turn
        assert!(s.contains("<|im_start|>user\n<tool_response>\n{\"temp\":15}\n</tool_response>"));
        // Final user turn closes the tool batch with <|im_end|>
        assert!(s.contains("</tool_response><|im_end|>\n<|im_start|>user\nanything else?"));
    }

    /// The agentic path's half of the reasoning round-trip.
    ///
    /// Two things are asserted together because they are the same line of code:
    /// a turn with no trace must render exactly as it always did (the empty
    /// block), and a turn that carries one must render *that* trace rather than
    /// an empty block with the envelope stranded in the visible text — which is
    /// the doubled `<think>\n\n</think>\n\n<think>\n…` this used to emit.
    #[test]
    fn a_replayed_tool_turn_carries_the_trace_it_was_given() {
        use crate::chat_io::join_reasoning_envelope;

        let plain = vec![ChatTurn::Assistant {
            text: "hello",
            tool_calls: &[],
        }];
        let s = format_qwen3_chat_with_tools_from_history(&plain, true, &[], None);
        assert!(
            s.contains("<|im_start|>assistant\n<think>\n\n</think>\n\nhello<|im_end|>"),
            "a turn without a trace must be unchanged:\n{s}",
        );

        let with_trace = join_reasoning_envelope("checking the map", "hello");
        let turns = vec![ChatTurn::Assistant {
            text: &with_trace,
            tool_calls: &[],
        }];
        let s = format_qwen3_chat_with_tools_from_history(&turns, true, &[], None);
        assert!(
            s.contains(
                "<|im_start|>assistant\n<think>\nchecking the map\n</think>\n\nhello<|im_end|>"
            ),
            "the returned trace belongs inside the block:\n{s}",
        );
        assert!(
            !s.contains("</think>\n\n<think>"),
            "two blocks is a shape no Qwen template emits:\n{s}",
        );
    }

    /// A turn that is pure reasoning plus tool calls has no visible text, so it
    /// must not get the `\n\n` separator that follows visible text. Before the
    /// split, the envelope counted as content and the separator went in.
    #[test]
    fn a_trace_only_tool_turn_gets_no_visible_text_separator() {
        use crate::chat_io::join_reasoning_envelope;

        let args = json!({"city": "Seoul"});
        let tc = vec![AssistantToolCall {
            id: "call_1",
            name: "get_weather",
            arguments: &args,
        }];
        let text = join_reasoning_envelope("need the weather", "");
        let turns = vec![ChatTurn::Assistant {
            text: &text,
            tool_calls: tc.as_slice(),
        }];
        let s = format_qwen3_chat_with_tools_from_history(&turns, true, &[], None);
        assert!(
            s.contains("</think>\n\n<tool_call>\n<function=get_weather>"),
            "no visible text means no extra separator before the call:\n{s}",
        );
    }

    #[test]
    fn render_batched_tool_responses() {
        let turns = vec![
            ChatTurn::User("q"),
            ChatTurn::Assistant {
                text: "",
                tool_calls: &[],
            },
            ChatTurn::Tool {
                tool_call_id: "1",
                name: None,
                content: "r1",
            },
            ChatTurn::Tool {
                tool_call_id: "2",
                name: None,
                content: "r2",
            },
        ];
        let s = format_qwen3_chat_with_tools_from_history(&turns, false, &[], None);
        // Both tool_responses in ONE user turn.
        assert!(s.contains("<|im_start|>user\n<tool_response>\nr1\n</tool_response>\n<tool_response>\nr2\n</tool_response><|im_end|>"));
    }

    #[test]
    fn tools_template_carries_an_image_block_ahead_of_its_text() {
        // Images and tools compose only because they touch different parts of
        // the prompt: the image block rides inside a user message body, the
        // tool template wraps around it. Verify the template passes the block
        // through untouched and in front of the question — the placeholder run
        // is spliced positionally, so a template that reordered or escaped it
        // would put the pixels on the wrong rows.
        let params = weather_tool();
        let tools = vec![ToolDef {
            name: "get_weather",
            description: Some("Get weather"),
            parameters: Some(&params),
            response: None,
        }];
        let body = format!("{}what city is this?", crate::qwen36_vision::IMAGE_BLOCK);
        let messages = vec![("user".into(), body)];
        let s = format_qwen3_chat_with_tools(&messages, false, &tools, None);
        assert!(
            s.contains(&format!(
                "<|im_start|>user\n{}what city is this?<|im_end|>",
                crate::qwen36_vision::IMAGE_BLOCK
            )),
            "image block must survive the tools template, ahead of the text:\n{s}"
        );
        assert_eq!(
            s.matches("<|image_pad|>").count(),
            1,
            "exactly one placeholder per image before id-level expansion"
        );
    }

    #[test]
    fn history_template_folds_a_system_turn_into_the_tools_block() {
        // This is why the structured-history image path refuses anything but a
        // User turn: a System turn's text does not render where it was written,
        // it is absorbed into the `<tools>` block. A placeholder attached there
        // would move relative to the other images, and the k-th placeholder run
        // would then be spliced with the wrong one.
        let params = weather_tool();
        let tools = vec![ToolDef {
            name: "get_weather",
            description: Some("Get weather"),
            parameters: Some(&params),
            response: None,
        }];
        let turns = vec![ChatTurn::System("SYSTEM_MARKER"), ChatTurn::User("hi")];
        let s = format_qwen3_chat_with_tools_from_history(&turns, false, &tools, None);
        assert!(s.contains("SYSTEM_MARKER"), "system text is still present");
        assert!(
            !s.contains("<|im_start|>system\nSYSTEM_MARKER"),
            "system turn is folded into the tools block, not emitted verbatim:\n{s}"
        );
    }

    #[test]
    fn render_param_value_preserves_strings_serializes_objects() {
        let v_str = JsonValue::String("hello".into());
        assert_eq!(value_to_param_str(&v_str), "hello");
        let v_obj = json!({"a": 1, "b": [2, 3]});
        let rendered = value_to_param_str(&v_obj);
        // Order from serde_json is insertion-order; assertions check
        // that both keys are present in the JSON shape.
        assert!(rendered.contains("\"a\":1"));
        assert!(rendered.contains("\"b\":[2,3]"));
    }

    #[test]
    fn parser_visible_text_only() {
        let mut p = Qwen35ResponseParser::new();
        let evs = p.feed("Hello, world!");
        assert_eq!(evs, vec![Qwen35ParseEvent::Text("Hello, world!".into())]);
        let resp = p.finish();
        assert_eq!(resp.visible, "Hello, world!");
        assert_eq!(resp.reasoning, "");
        assert!(resp.tool_calls.is_empty());
    }

    #[test]
    fn parser_thinking_stripped_from_visible() {
        let mut p = Qwen35ResponseParser::new();
        let evs = p.feed("pre <think>secret reasoning</think>post");
        // No Text event for "secret reasoning"
        let text_evs: Vec<_> = evs
            .iter()
            .filter_map(|e| match e {
                Qwen35ParseEvent::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert!(text_evs.iter().any(|t| t.contains("pre ")));
        assert!(text_evs.iter().any(|t| t.contains("post")));
        assert!(!text_evs.iter().any(|t| t.contains("secret")));
        let resp = p.finish();
        assert_eq!(resp.reasoning, "secret reasoning");
    }

    /// With thinking ON the prompt — not the model — emits the `<think>`
    /// opener, so the output starts mid-trace and closes with a bare
    /// `</think>`.
    ///
    /// Measured on Qwen3.8-27B before this existed: a `thinking:true` request
    /// came back with no `reasoning` field at all, `content` holding the raw
    /// chain-of-thought, and a stray `</think>` sitting in the middle of the
    /// visible answer:
    ///
    /// ```text
    /// content: "We need answer user's simple request: … red.\n</think>\n\nRed"
    /// ```
    #[test]
    fn a_prompt_opened_think_block_is_reasoning_not_visible_text() {
        let mut p = Qwen35ResponseParser::with_thinking_open();
        let evs = p.feed("weighing it up\n</think>\n\nRed");
        let text: String = evs
            .iter()
            .filter_map(|e| match e {
                Qwen35ParseEvent::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            text, "Red",
            "the trace must not reach the client as content"
        );
        let resp = p.finish();
        assert_eq!(resp.reasoning.trim(), "weighing it up");
        assert_eq!(resp.visible.trim(), "Red");
        assert!(
            !resp.visible.contains("</think>"),
            "the close tag is a delimiter, not part of the answer: {:?}",
            resp.visible
        );
    }

    /// The plain (no-tools) path has no parser at all — it returned the whole
    /// decode as `visible` with `reasoning: String::new()` hardcoded. Measured
    /// on Qwen3.8-27B, a `thinking:true` request came back with no `reasoning`
    /// key and this as its content:
    ///
    /// ```text
    /// "We need answer user's simple request: … red.\n</think>\n\nRed"
    /// ```
    #[test]
    fn the_plain_path_splits_a_prompt_opened_trace() {
        assert_eq!(
            split_open_think("weighing it up\n</think>\n\nRed"),
            ("weighing it up\n", "Red")
        );
        // One separator only — an answer that really starts with a blank line
        // keeps it.
        assert_eq!(split_open_think("t\n</think>\n\n\nRed"), ("t\n", "\nRed"));
        // Never closed: the model spent the budget thinking and reached no
        // answer. Saying so beats presenting the trace as the reply.
        assert_eq!(split_open_think("still thinking"), ("still thinking", ""));
    }

    /// …which is exactly why the callers must gate it on the thinking flag.
    ///
    /// A thinking-OFF reply carries no `</think>` — the prompt closed the block
    /// — so it is indistinguishable from a trace that ran out of budget. Caught
    /// by hand against the running server, not by the suite: an ungated split
    /// answered `content: ""`, `reasoning: "Red"` for a plain request.
    #[test]
    fn a_thinking_off_reply_looks_exactly_like_an_unclosed_trace() {
        assert_eq!(split_open_think("Red"), ("Red", ""));
        // The distinction is not in the bytes, so it cannot be recovered here.
        // It lives in the prompt, which is why `thinking` is threaded to the
        // call sites rather than sniffed from the reply.
        let mut off = ThinkingSplitter::new(false);
        assert_eq!(off.feed("Red"), (String::new(), "Red".to_string()));
    }

    /// A `</think>` split across two decode steps must not leak as visible
    /// text — once a byte is streamed as content the client has it.
    #[test]
    fn the_streaming_splitter_holds_a_tag_split_across_chunks() {
        for cut in 1.."</think>".len() {
            let (a, b) = "</think>".split_at(cut);
            let mut s = ThinkingSplitter::new(true);
            let mut reasoning = String::new();
            let mut visible = String::new();
            for chunk in ["trace", a, b, "\n\nRed"] {
                let (r, v) = s.feed(chunk);
                reasoning.push_str(&r);
                visible.push_str(&v);
            }
            reasoning.push_str(&s.finish().0);
            assert_eq!(
                visible, "Red",
                "cut after {cut} leaked the tag or the trace"
            );
            assert_eq!(reasoning, "trace", "cut after {cut}");
        }
    }

    #[test]
    fn the_streaming_splitter_is_a_pass_through_when_thinking_is_off() {
        let mut s = ThinkingSplitter::new(false);
        assert_eq!(s.feed("Red"), (String::new(), "Red".to_string()));
        assert_eq!(s.finish(), (String::new(), String::new()));
    }

    /// A reasoning-first checkpoint sometimes closes a block it never opened,
    /// even on a thinking-OFF turn where the prompt handed it a closed one.
    ///
    /// Measured on Qwen3.8-27B: `'Blue\n</think>\n\nBlue'`. Never at
    /// temperature 0 (0/16), rarely when sampling (1/32 at 0.8, 2/6 at the 0.7
    /// default) — so it is the model's doing, not a prompt defect: the rendered
    /// generation prompt ends with a closed `<think>\n\n</think>\n\n`.
    ///
    /// The tool-aware parser has always dropped an unbalanced close, so the
    /// same reply used to come back differently depending only on whether the
    /// request carried tools.
    #[test]
    fn a_stray_close_tag_never_reaches_the_answer() {
        assert_eq!(
            strip_unbalanced_think_close("Blue\n</think>\n\nBlue"),
            "Blue\n\n\nBlue"
        );
        // Nothing to strip: borrowed, not rebuilt.
        assert!(matches!(
            strip_unbalanced_think_close("just an answer"),
            std::borrow::Cow::Borrowed(_)
        ));
        // A model showing the syntax is not delimiting anything.
        assert_eq!(
            strip_unbalanced_think_close("use <think>x</think> like so"),
            "use <think>x</think> like so"
        );
        // …but a second, unmatched close after a balanced pair still goes.
        assert_eq!(
            strip_unbalanced_think_close("<think>a</think>b</think>c"),
            "<think>a</think>bc"
        );
    }

    /// The streaming surface has to reach the same answer as the batch one,
    /// including when the tag is split across decode steps and when a balanced
    /// pair spans them — chunk-local depth would call every close unbalanced.
    #[test]
    fn the_streaming_splitter_drops_a_stray_close_the_same_way() {
        for chunks in [
            vec!["Blue\n", "</think>", "\n\nBlue"],
            vec!["Blue\n<", "/think", ">\n\nBlue"],
            vec!["Blue\n</thi", "nk>\n\nBlue"],
        ] {
            let mut s = ThinkingSplitter::new(false);
            let mut visible = String::new();
            for c in &chunks {
                visible.push_str(&s.feed(c).1);
            }
            let (r, v) = s.finish();
            visible.push_str(&v);
            assert!(r.is_empty());
            assert_eq!(visible, "Blue\n\n\nBlue", "chunks {chunks:?}");
        }
        // Balanced across chunks: nothing is dropped.
        let mut s = ThinkingSplitter::new(false);
        let mut visible = String::new();
        for c in ["use <think>", "x", "</think> like so"] {
            visible.push_str(&s.feed(c).1);
        }
        visible.push_str(&s.finish().1);
        assert_eq!(visible, "use <think>x</think> like so");
    }

    /// The trace has to be emitted as it arrives, not only accumulated.
    ///
    /// `finish()` always carried it, so the non-streaming answer was right
    /// while the streaming one silently had no trace at all — measured against
    /// the running server, a `thinking`-enabled Anthropic request *with tools*
    /// streamed `text@0, tool_use@1` and nothing else, where the same request
    /// without tools streamed `thinking@0, text@1`. The tool-aware path is the
    /// agentic one, so that was the case where it mattered most.
    #[test]
    fn the_tool_parser_streams_the_trace_instead_of_only_keeping_it() {
        let mut p = Qwen35ResponseParser::with_thinking_open();
        let mut reasoning = String::new();
        let mut text = String::new();
        // Fed one chunk at a time, the way decode delivers it.
        for chunk in ["weigh", "ing it up", "\n</thi", "nk>\n\nRed"] {
            for ev in p.feed(chunk) {
                match ev {
                    Qwen35ParseEvent::Reasoning(t) => reasoning.push_str(&t),
                    Qwen35ParseEvent::Text(t) => text.push_str(&t),
                    Qwen35ParseEvent::ToolCallStart { .. } => unreachable!("no tools here"),
                }
            }
        }
        assert_eq!(reasoning.trim(), "weighing it up");
        assert_eq!(text.trim(), "Red");
        // …and the accumulated view still agrees with the streamed one.
        let resp = p.finish();
        assert_eq!(resp.reasoning.trim(), "weighing it up");
        assert_eq!(resp.visible.trim(), "Red");
    }

    /// The thinking-OFF turn must be untouched: there the prompt closes the
    /// block itself, so the model's output is visible text from the first byte
    /// and starting in the thinking state would swallow the whole answer.
    #[test]
    fn a_prompt_closed_think_block_leaves_the_answer_visible() {
        let mut p = Qwen35ResponseParser::new();
        p.feed("Red");
        let resp = p.finish();
        assert_eq!(resp.visible, "Red");
        assert_eq!(resp.reasoning, "");
    }

    #[test]
    fn parser_full_tool_call() {
        let mut p = Qwen35ResponseParser::new();
        let stream = "I'll call the function.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nSeoul\n</parameter>\n<parameter=unit>\nc\n</parameter>\n</function>\n</tool_call>";
        let evs = p.feed(stream);
        // Must contain a ToolCallStart event with the right name.
        let has_start = evs.iter().any(
            |e| matches!(e, Qwen35ParseEvent::ToolCallStart { name } if name == "get_weather"),
        );
        assert!(has_start, "expected ToolCallStart event");
        let resp = p.finish();
        assert_eq!(resp.tool_calls.len(), 1);
        let call = &resp.tool_calls[0];
        assert_eq!(call.name, "get_weather");
        let args = call.arguments.as_object().unwrap();
        assert_eq!(args.get("city").unwrap().as_str(), Some("Seoul"));
        assert_eq!(args.get("unit").unwrap().as_str(), Some("c"));
    }

    #[test]
    fn parser_streaming_chunked_emits_start_early() {
        let mut p = Qwen35ResponseParser::new();
        let chunks = [
            "I'll ",
            "call.",
            "\n<tool_call>\n<function=",
            "search>\n",
            "<parameter=q>\nrust\n</parameter>\n",
            "</function>\n</tool_call>",
        ];
        let mut early_start_seen = false;
        let mut chunks_before_start = 0;
        for (i, c) in chunks.iter().enumerate() {
            let evs = p.feed(c);
            if !early_start_seen {
                for e in &evs {
                    if let Qwen35ParseEvent::ToolCallStart { name } = e {
                        assert_eq!(name, "search");
                        early_start_seen = true;
                        chunks_before_start = i + 1;
                    }
                }
            }
        }
        assert!(early_start_seen);
        // Must fire BEFORE the body chunk (parameter=q) is fed (i.e.
        // at chunk index 3 when "search>\n" closes the function tag).
        assert!(chunks_before_start <= 4, "got {chunks_before_start}");
        let resp = p.finish();
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].name, "search");
        assert_eq!(
            resp.tool_calls[0]
                .arguments
                .as_object()
                .unwrap()
                .get("q")
                .unwrap()
                .as_str(),
            Some("rust")
        );
    }

    #[test]
    fn parser_json_param_value_decoded() {
        let mut p = Qwen35ResponseParser::new();
        let stream = "<tool_call>\n<function=f>\n<parameter=opts>\n{\"k\": 1}\n</parameter>\n</function>\n</tool_call>";
        let _ = p.feed(stream);
        let resp = p.finish();
        let opts = resp.tool_calls[0].arguments.get("opts").unwrap();
        assert_eq!(opts, &json!({"k": 1}));
    }

    #[test]
    fn parser_array_param_value_decoded() {
        let mut p = Qwen35ResponseParser::new();
        let stream = "<tool_call>\n<function=f>\n<parameter=arr>\n[1,2,3]\n</parameter>\n</function>\n</tool_call>";
        let _ = p.feed(stream);
        let resp = p.finish();
        assert_eq!(
            resp.tool_calls[0].arguments.get("arr").unwrap(),
            &json!([1, 2, 3])
        );
    }

    #[test]
    fn parser_two_sequential_tool_calls() {
        let mut p = Qwen35ResponseParser::new();
        let stream = "<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=b>\n<parameter=y>\nhi\n</parameter>\n</function>\n</tool_call>";
        let _ = p.feed(stream);
        let resp = p.finish();
        assert_eq!(resp.tool_calls.len(), 2);
        assert_eq!(resp.tool_calls[0].name, "a");
        assert_eq!(resp.tool_calls[1].name, "b");
    }

    // ───────────────── parallel_tool_calls on the Qwen path ─────────────────

    /// `completed_calls()` is the decode loop's stop signal, so it must count
    /// only calls whose `</tool_call>` has actually arrived. Counting at
    /// `</function>` — or at the point the grammar reports finished — would cut
    /// the framing `finish()` needs, which is the mistake the Gemma 4 rule
    /// documents having already made once.
    #[test]
    fn completed_calls_counts_only_fully_closed_calls() {
        let mut p = Qwen35ResponseParser::new();
        for (chunk, expected) in [
            ("<tool_call>\n", 0),
            ("<function=a>\n", 0),
            ("<parameter=x>\n1\n</parameter>\n", 0),
            ("</function>\n", 0),
            ("</tool_call>", 1),
            ("\n<tool_call>\n<function=b>\n</function>\n", 1),
            ("</tool_call>", 2),
        ] {
            let _ = p.feed(chunk);
            assert_eq!(
                p.completed_calls(),
                expected,
                "after feeding {chunk:?} the count should be {expected}"
            );
        }
    }

    /// The defect this replaces, without a model: the whole Qwen decode path
    /// built `ToolCalls::ExactlyOne` correctly and then never consulted it, so
    /// `parallel_tool_calls: false` was accepted and silently ignored.
    /// Measured on Qwen3.8-27B, `tool_choice=required` + `parallel_tool_calls
    /// =false` returned SEVEN identical calls.
    ///
    /// Simulates `chat_with_tools_impl`'s loop: feed a chunk, then consult the
    /// policy at the top of the next iteration. Both settings share one stream
    /// so the difference is the policy and nothing else.
    #[test]
    fn exactly_one_cuts_the_turn_where_one_or_more_keeps_decoding() {
        let chunks = [
            "<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n",
            "</tool_call>",
            "\n<tool_call>\n<function=b>\n<parameter=y>\nhi\n</parameter>\n</function>\n",
            "</tool_call>",
        ];
        for (calls, want_calls, want_fed) in [
            (crate::grammar::ToolCalls::ExactlyOne, 1, 2),
            (crate::grammar::ToolCalls::OneOrMore, 2, 4),
        ] {
            let mut p = Qwen35ResponseParser::new();
            let mut fed = 0;
            for chunk in chunks {
                if calls.must_stop_after_completed_calls(p.completed_calls()) {
                    break;
                }
                let _ = p.feed(chunk);
                fed += 1;
            }
            assert_eq!(fed, want_fed, "{calls:?}: wrong number of decode steps");
            let resp = p.finish();
            assert_eq!(
                resp.tool_calls.len(),
                want_calls,
                "{calls:?}: wrong number of calls returned"
            );
            assert_eq!(resp.tool_calls[0].name, "a");
        }
    }

    #[test]
    fn parser_token_by_token_real_world_chunks() {
        // Mirrors the exact chunk pattern the live MLX backend feeds
        // when Qwen 3.6-35B-A3B-mxfp4 emits a single tool call. Pinned
        // here so the parser's state-machine never regresses on tiny
        // BPE token chunks (the issue Phase 2 Stage 4 caught live).
        let mut p = Qwen35ResponseParser::new();
        let chunks = [
            "<tool_call>",
            "\n",
            "<",
            "function",
            "=get",
            "_weather",
            ">",
            "\n",
            "<",
            "parameter",
            "=",
            "city",
            ">",
            "\n",
            "Se",
            "oul",
            "\n",
            "</",
            "parameter",
            ">",
            "\n",
            "</",
            "function",
            ">",
            "\n",
            "</tool_call>",
        ];
        let mut saw_start = false;
        for c in chunks {
            for e in p.feed(c) {
                if let Qwen35ParseEvent::ToolCallStart { name } = e {
                    assert_eq!(name, "get_weather");
                    saw_start = true;
                }
            }
        }
        assert!(saw_start, "ToolCallStart must fire across chunked tokens");
        let resp = p.finish();
        assert_eq!(resp.tool_calls.len(), 1);
        let call = &resp.tool_calls[0];
        assert_eq!(call.name, "get_weather");
        let args = call.arguments.as_object().unwrap();
        assert_eq!(args.get("city").and_then(|v| v.as_str()), Some("Seoul"));
    }

    #[test]
    fn prefill_str_auto_none_empty() {
        assert_eq!(
            qwen35_tool_choice_prefill_str(&ResolvedToolChoice::Auto),
            ""
        );
        assert_eq!(
            qwen35_tool_choice_prefill_str(&ResolvedToolChoice::None),
            ""
        );
    }

    #[test]
    fn prefill_str_required_opens_function_tag() {
        assert_eq!(
            qwen35_tool_choice_prefill_str(&ResolvedToolChoice::Required),
            "<tool_call>\n<function="
        );
    }

    #[test]
    fn prefill_str_named_tool_full_opener() {
        assert_eq!(
            qwen35_tool_choice_prefill_str(&ResolvedToolChoice::Tool("get_weather")),
            "<tool_call>\n<function=get_weather>\n"
        );
    }
}

/// Qwen 3.8's `reasoning_effort` block, checked against the shipped
/// `chat_template.jinja` read as spec.
///
/// The load-bearing test is the LAST one: every pre-3.8 checkpoint must render
/// exactly as it did before this feature existed. The effort is opt-in per
/// checkpoint precisely so that stays true.
#[cfg(test)]
mod reasoning_effort_tests {
    use super::{format_qwen3_chat_with_tools, render_system_block, render_tools_system_block};
    use crate::chat_io::{ReasoningEffort, ToolDef};
    use serde_json::json;

    const XHIGH_TEXT: &str = "Reasoning effort is set to xhigh.";
    const LOW_TEXT: &str = "Reasoning effort is set to low.";

    fn tools() -> Vec<ToolDef<'static>> {
        vec![ToolDef {
            name: "get_weather",
            description: Some("Get weather"),
            parameters: None,
            response: None,
        }]
    }

    #[test]
    fn with_tools_the_effort_goes_before_the_tools_header_not_after_it() {
        let s = render_tools_system_block(
            &tools(),
            Some("You are terse."),
            Some(ReasoningEffort::Xhigh),
        );
        let effort_at = s.find(XHIGH_TEXT).expect("effort sentence present");
        let tools_at = s.find("# Tools").expect("tools header present");
        let system_at = s.find("You are terse.").expect("system content present");
        assert!(
            effort_at < tools_at,
            "template puts the effort sentence ahead of `# Tools`"
        );
        assert!(
            tools_at < system_at,
            "the client's system text still trails the IMPORTANT block"
        );
    }

    #[test]
    fn medium_renders_no_sentence_because_upstream_leaves_it_empty() {
        let with = render_tools_system_block(&tools(), None, Some(ReasoningEffort::Medium));
        let without = render_tools_system_block(&tools(), None, None);
        assert_eq!(
            with, without,
            "upstream sets an instruction string only for xhigh and low"
        );
    }

    #[test]
    fn low_and_xhigh_render_different_prompts() {
        let lo = render_tools_system_block(&tools(), None, Some(ReasoningEffort::Low));
        let hi = render_tools_system_block(&tools(), None, Some(ReasoningEffort::Xhigh));
        assert!(lo.contains(LOW_TEXT) && !lo.contains(XHIGH_TEXT));
        assert!(hi.contains(XHIGH_TEXT) && !hi.contains(LOW_TEXT));
    }

    #[test]
    fn a_system_block_is_created_when_the_request_has_none() {
        // 3.8 emits a system block holding just the effort sentence.
        let s = render_system_block(None, Some(ReasoningEffort::Xhigh));
        assert!(s.starts_with("<|im_start|>system\n"));
        assert!(s.contains(XHIGH_TEXT));
        assert!(s.ends_with("<|im_end|>\n"));
    }

    #[test]
    fn nothing_at_all_is_emitted_without_system_or_effort() {
        assert_eq!(render_system_block(None, None), "");
    }

    #[test]
    fn the_effort_precedes_the_system_text_in_the_no_tools_branch() {
        let s = render_system_block(Some("You are terse."), Some(ReasoningEffort::Low));
        assert_eq!(
            s,
            format!(
                "<|im_start|>system\n{}\n\nYou are terse.<|im_end|>\n",
                ReasoningEffort::Low.instructions().unwrap()
            )
        );
    }

    // ── The regression this whole design exists to prevent ────────────────
    #[test]
    fn a_pre_38_checkpoint_renders_byte_identically() {
        // `None` is what a 3.5/3.6 checkpoint always resolves to, because
        // `checkpoint_declares_reasoning_effort` reads the shipped template and
        // theirs has no such block. Whatever a client sends, these two must be
        // the same bytes.
        let msgs = vec![
            ("system".to_string(), "You are terse.".to_string()),
            ("user".to_string(), "hi".to_string()),
        ];
        let before = format_qwen3_chat_with_tools(&msgs, false, &tools(), None);
        assert!(!before.contains("Reasoning effort"));
        assert!(before.contains("You are terse."));

        // …and with no tools either.
        let plain = format_qwen3_chat_with_tools(&msgs, false, &[], None);
        assert_eq!(
            plain,
            "<|im_start|>system\nYou are terse.<|im_end|>\n\
             <|im_start|>user\nhi<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n",
            "the no-tools, no-effort rendering is the pre-3.8 one, exactly"
        );
    }

    #[test]
    fn tool_json_is_untouched_by_the_effort_block() {
        let params = json!({"type":"object","properties":{}});
        let t = vec![ToolDef {
            name: "f",
            description: None,
            parameters: Some(&params),
            response: None,
        }];
        let hi = render_tools_system_block(&t, None, Some(ReasoningEffort::Xhigh));
        let none = render_tools_system_block(&t, None, None);
        // The effort contribution is exactly `sentence + "\n\n"` at the head of
        // the block; deleting it must leave the pre-3.8 rendering byte for byte.
        let injected = format!("{}\n\n", ReasoningEffort::Xhigh.instructions().unwrap());
        assert_eq!(
            hi.replace(&injected, ""),
            none,
            "removing the sentence must leave the 3.6 block behind"
        );
    }
}
