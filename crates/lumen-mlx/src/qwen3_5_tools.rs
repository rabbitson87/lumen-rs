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
    AssistantToolCall, ChatTurn, ParsedResponse, ParsedToolCall, ReasoningEffort,
    ResolvedToolChoice, ToolDef,
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

/// Render the assistant-turn opening (without tool_calls):
///   `<|im_start|>assistant\n<think>\n\n</think>\n\n{content}`
/// or with thinking enabled:
///   `<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{content}`
fn render_assistant_open(content: &str, thinking: bool) -> String {
    let mut s = String::new();
    s.push_str("<|im_start|>assistant\n");
    if thinking {
        // Replay-mode: we don't have the original reasoning trace from the
        // client; emit empty thinking block to satisfy template.
        s.push_str("<think>\n\n</think>\n\n");
    } else {
        s.push_str("<think>\n\n</think>\n\n");
    }
    s.push_str(content);
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
                let has_content = !content.trim().is_empty();
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
                State::Thinking => self.try_advance_thinking(),
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

    fn try_advance_thinking(&mut self) -> bool {
        if let Some(idx) = self.buffer.find("</think>") {
            let prefix = self.buffer[..idx].to_string();
            self.reasoning.push_str(&prefix);
            self.buffer.drain(..idx + "</think>".len());
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
