//! Grammar-constrained tool calling for Gemma 4 via llguidance.
//!
//! Mirrors llama.cpp's `peg-gemma4` lazy GBNF mechanism, adapted to
//! Lumen's CPU sampling pipeline (logits already on host after
//! [`super::gemma4_sampling::imp::last_logits_to_cpu_f32`]).
//!
//! State machine per request:
//!
//! ```text
//!   tool_choice="auto"           tool_choice="required"
//!   ─────────────────────        ──────────────────────
//!   FREE (matcher=None)          CONSTRAINED (matcher=Some)
//!     │                              │
//!     │ sampled token == 48          │ every sampled token
//!     │ (<|tool_call>)               │
//!     ▼                              ▼
//!   CONSTRAINED                    CONSTRAINED ... stop_reason()
//!     │
//!     │ every subsequent sampled token
//!     ▼
//!   CONSTRAINED ... stop_reason() == Eos / NoExtension
//! ```
//!
//! After the Matcher reports termination, the grammar releases — subsequent
//! tokens are unconstrained until the model emits `<turn|>` and the decode
//! loop ends naturally.

#![allow(dead_code)] // Phase 2.3 skeleton — wired up in Phase 2.4+.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use llguidance::{
    Matcher, ParserFactory,
    api::TopLevelGrammar,
    token_bytes_from_tokenizer_json,
    toktrie::{ApproximateTokEnv, TokEnv, TokRxInfo, TokTrie},
};
use serde_json::{Value, json};

// Gemma 4 special-token IDs needed by the grammar layer. Defined locally
// to keep this module feature-gate-free; `gemma4_chat::imp` is only
// reachable behind `mlx-native`, but grammar construction itself is pure
// llguidance + serde_json. These are model-family invariants (same across
// every Gemma 4 size) sourced from `tokenizer.json`.
const TOK_EOS: u32 = 1;
const TOK_TURN_CLOSE: u32 = 106; // `<turn|>` — end-of-assistant-turn delimiter.

/// Gemma 4 native tool-call opener token id (`<|tool_call>`).
pub const TOK_TOOL_CALL_OPEN: u32 = 48;

/// Gemma 4 native tool-call closer token id (`<tool_call|>`).
pub const TOK_TOOL_CALL_CLOSE: u32 = 49;

/// Gemma 4 custom string delimiter token id (`<|"|>`), used inside
/// `call:NAME{key:value}` bodies to wrap string-typed argument values.
/// Lark grammar references this via `<[52]>`.
pub const TOK_QUOTE_DELIM: u32 = 52;

/// How aggressively to enforce the tool-call grammar on a given request.
///
/// Derived from the OpenAI `tool_choice` request field:
///   - `"auto"` (default) → [`GrammarMode::Lazy`] — model samples freely until
///     it emits `<|tool_call>`, then the grammar activates and forces a valid
///     `call:NAME{args}<tool_call|>` completion.
///   - `"required"` → [`GrammarMode::Eager`] — grammar is active from the
///     first decode step, so the very first token is forced to be one of the
///     allowed tool-call openers.
///   - `"none"` or no tools → caller passes `None` for the grammar state,
///     skipping this module entirely.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrammarMode {
    Lazy,
    Eager,
}

/// Per-request grammar guard. Built once at the start of a chat completion
/// (after the prompt is rendered) and stepped per sampled token.
pub struct Gemma4GrammarState {
    factory: Arc<ParserFactory>,
    schema: TopLevelGrammar,
    matcher: Option<Matcher>,
    mode: GrammarMode,
    /// True once `stop_reason()` reports termination — the matcher releases
    /// and subsequent tokens are unconstrained.
    finished: bool,
    /// Token id that flips a [`GrammarMode::Lazy`] matcher on when first
    /// sampled. Defaults to Gemma 4's `<|tool_call>` (id 48), but is
    /// parameterized so other families can reuse this state machine with
    /// their own opener (e.g. Qwen 3.6's `<tool_call>` — though Qwen's
    /// required/named path runs Eager and never relies on this trigger).
    lazy_trigger_token: u32,
}

impl Gemma4GrammarState {
    /// Build a new request-scoped grammar guard.
    ///
    /// `tools` is the OpenAI-spec `tools` array (each entry has the shape
    /// `{"type":"function","function":{"name":...,"parameters":...}}`).
    pub fn new(factory: Arc<ParserFactory>, tools: &[Value], mode: GrammarMode) -> Result<Self> {
        if tools.is_empty() {
            return Err(anyhow!(
                "Gemma4GrammarState::new called with empty tools — caller should skip grammar"
            ));
        }
        let schema = build_tool_grammar(tools)?;
        let mut state = Self {
            factory,
            schema,
            matcher: None,
            mode,
            finished: false,
            lazy_trigger_token: TOK_TOOL_CALL_OPEN,
        };
        if matches!(mode, GrammarMode::Eager) {
            state.activate()?;
        }
        Ok(state)
    }

    /// Response-format variant — constrains the **visible** output to
    /// valid JSON matching an arbitrary JSON Schema, from token 0.
    ///
    /// Unlike [`new`] / [`new_lark`] (which target the tool-call body and
    /// derive an internal schema from the OpenAI `tools` array), this
    /// builds the top-level grammar directly from the caller's schema via
    /// [`TopLevelGrammar::from_json_schema`]. Intended for OpenAI
    /// `response_format` (`json_object` / `json_schema`), where the entire
    /// assistant message must be a single JSON value. Always built in the
    /// requested `mode`; pass [`GrammarMode::Eager`] so the constraint is
    /// live from the first decode step (there is no lazy trigger token for
    /// free-form JSON output).
    pub fn new_json_schema(
        factory: Arc<ParserFactory>,
        schema: &Value,
        mode: GrammarMode,
    ) -> Result<Self> {
        let schema = TopLevelGrammar::from_json_schema(schema.clone());
        let mut state = Self {
            factory,
            schema,
            matcher: None,
            mode,
            finished: false,
            lazy_trigger_token: TOK_TOOL_CALL_OPEN,
        };
        if matches!(mode, GrammarMode::Eager) {
            state.activate()?;
        }
        Ok(state)
    }

    /// Lark-syntax variant — emits the Gemma 4 native `call:NAME{…}`
    /// format so the response parser ([`crate::gemma4_response`])
    /// accepts the output directly. Prefer this over [`new`] for
    /// production wiring; [`new`] builds the JSON-shape grammar used
    /// for schema-only smoke tests.
    pub fn new_lark(
        factory: Arc<ParserFactory>,
        tools: &[Value],
        mode: GrammarMode,
    ) -> Result<Self> {
        Self::new_lark_inner(factory, tools, mode, false)
    }

    /// Strict, dup-free Lark variant — same native `call:NAME{…}` output as
    /// [`new_lark`] (so [`crate::gemma4_response`] parses it unchanged) but
    /// the body grammar enforces `required` fields and forbids
    /// duplicate/repeated fields. Pair with [`GrammarMode::Eager`] for
    /// `tool_choice=required`/named so a prefill-forced `<|tool_call>` opener
    /// can't yield an empty-param body.
    ///
    /// When Eager, the prefilled opener tokens (`<|tool_call>` and, for a
    /// named choice, `call:NAME{`) were injected into the model's context via
    /// `parser.push()` and never *sampled*, so the matcher has not seen them.
    /// The caller MUST replay them through [`observe_prefill`] before the
    /// first decode step so the Eager matcher's state lines up with what the
    /// model already has in context; otherwise the mask would re-force the
    /// opener and corrupt the body.
    pub fn new_lark_strict(
        factory: Arc<ParserFactory>,
        tools: &[Value],
        mode: GrammarMode,
    ) -> Result<Self> {
        Self::new_lark_inner(factory, tools, mode, true)
    }

    fn new_lark_inner(
        factory: Arc<ParserFactory>,
        tools: &[Value],
        mode: GrammarMode,
        strict: bool,
    ) -> Result<Self> {
        if tools.is_empty() {
            return Err(anyhow!(
                "Gemma4GrammarState::new_lark called with empty tools"
            ));
        }
        let schema = if strict {
            build_tool_grammar_lark_strict(tools)?
        } else {
            build_tool_grammar_lark(tools)?
        };
        let mut state = Self {
            factory,
            schema,
            matcher: None,
            mode,
            finished: false,
            lazy_trigger_token: TOK_TOOL_CALL_OPEN,
        };
        if matches!(mode, GrammarMode::Eager) {
            state.activate()?;
        }
        Ok(state)
    }

    /// Qwen 3.6 variant — constrains the **nested-XML** tool-call body the
    /// Qwen 3.6 model is trained on and that [`crate::qwen3_5_tools::
    /// Qwen35ResponseParser`] reads:
    ///
    /// ```text
    /// <tool_call>
    /// <function=NAME>
    /// <parameter=KEY>
    /// VALUE
    /// </parameter>
    /// </function>
    /// </tool_call>
    /// ```
    ///
    /// Unlike Gemma 4's native `call:NAME{…}` form (or the JSON-union the
    /// SGLang reference uses), Qwen reads this XML verbatim, so the grammar is
    /// built directly over those tags ([`build_qwen35_tool_grammar_lark`]).
    /// The body is **dup-free + required-enforcing** in the same spirit as
    /// [`new_lark_strict`]: each `<parameter=KEY>` appears at most once and
    /// every required key is mandatory, so a quantized 35B can't emit
    /// `<function=read></function>` with empty params (the empty-param defect).
    ///
    /// `opener_token` is the id flipped on in [`GrammarMode::Lazy`] when first
    /// sampled. Qwen's `<tool_call>` opener can tokenize to multiple ids, so
    /// the Lazy/auto path should NOT rely on a single trigger; prefer
    /// [`GrammarMode::Eager`] with [`observe_prefill`] for the
    /// `tool_choice=required`/named path where the opener is prefilled. The
    /// argument is kept for completeness (and for an exact single-token opener
    /// if the tokenizer provides one); pass `None` to leave the Gemma default
    /// (harmless — it just never matches a Qwen sample).
    pub fn new_qwen35_xml(
        factory: Arc<ParserFactory>,
        tools: &[Value],
        mode: GrammarMode,
        opener_token: Option<u32>,
    ) -> Result<Self> {
        if tools.is_empty() {
            return Err(anyhow!(
                "Gemma4GrammarState::new_qwen35_xml called with empty tools"
            ));
        }
        let schema = build_qwen35_tool_grammar_lark(tools)?;
        let mut state = Self {
            factory,
            schema,
            matcher: None,
            mode,
            finished: false,
            lazy_trigger_token: opener_token.unwrap_or(TOK_TOOL_CALL_OPEN),
        };
        if matches!(mode, GrammarMode::Eager) {
            state.activate()?;
        }
        Ok(state)
    }

    /// True iff the grammar is currently constraining sampling. Used by the
    /// sampler to decide whether to compute the mask at all.
    pub fn is_active(&self) -> bool {
        !self.finished && self.matcher.is_some()
    }

    /// Mask raw f32 logits so positions outside the current grammar are
    /// set to `-inf`. No-op when the grammar isn't constraining (lazy
    /// pre-trigger or already finished) — caller can blind-apply this
    /// every step.
    ///
    /// Returns the number of positions masked, useful for sanity logging
    /// (0 in the no-op case; for an active matcher, expect vocab − allowed).
    pub fn apply_mask_to_logits(&mut self, logits: &mut [f32]) -> Result<usize> {
        if self.finished {
            return Ok(0);
        }
        let Some(m) = self.matcher.as_mut() else {
            return Ok(0);
        };
        let mask = m
            .compute_mask()
            .map_err(|e| anyhow!("Matcher::compute_mask: {e}"))?;
        // SimpleVob.len() reports the underlying vocab size used when the
        // ParserFactory was built (from tokenizer.json). The model's lm_head
        // can emit a LARGER, padded vocab (e.g. Qwen 3.6: tokenizer 248077 vs
        // lm_head 248320 — the tail are reserved/padding ids the tokenizer
        // never decodes). Treat those extra logits as disallowed (-inf): they
        // are never legal grammar outputs anyway. A logits buffer SMALLER than
        // the mask, however, means the tokenizer env and the model are
        // genuinely mismatched and downstream sampling is meaningless — error.
        if logits.len() < mask.len() {
            return Err(anyhow!(
                "grammar mask vocab mismatch: mask.len()={} > logits.len()={}",
                mask.len(),
                logits.len()
            ));
        }
        let mut masked = 0usize;
        let mask_len = mask.len();
        for (i, l) in logits.iter_mut().enumerate() {
            // Indices past the tokenizer vocab (lm_head padding) are always
            // disallowed; within range, defer to the grammar matcher.
            if i >= mask_len || !mask.is_allowed(i as u32) {
                *l = f32::NEG_INFINITY;
                masked += 1;
            }
        }
        Ok(masked)
    }

    /// Observe a sampled token and advance internal state.
    ///
    /// In lazy mode, this also handles the activation transition:
    /// when the model emits `<|tool_call>` (id 48) for the first time,
    /// the matcher is created and `consume_token` is called so the
    /// grammar tracks the opener as having been emitted.
    pub fn observe(&mut self, token: u32) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        if self.matcher.is_none()
            && matches!(self.mode, GrammarMode::Lazy)
            && token == self.lazy_trigger_token
        {
            self.activate()?;
        }
        if let Some(m) = &mut self.matcher {
            m.consume_token(token)
                .map_err(|e| anyhow!("Matcher::consume_token({token}): {e}"))?;
            if m.is_stopped() {
                self.finished = true;
            }
        }
        Ok(())
    }

    /// Replay tokens the model received via prompt **prefill** (not via
    /// sampling) into the matcher so its parse position matches the model's
    /// context. Used for the Eager required/named path, where the chat
    /// template prefills the `<|tool_call>` opener (and, for a named choice,
    /// `call:NAME{`) — those tokens were pushed into the prompt, never
    /// sampled, so the matcher never saw them.
    ///
    /// Unlike [`observe`], this does NOT run the lazy-activation transition
    /// (the matcher is already active in Eager mode) and is a no-op in Lazy
    /// mode or once finished. Each token must be one the grammar accepts at
    /// the current position (the prefilled `call:NAME{` prefix is exactly the
    /// grammar's deterministic opener), so `consume_token` should not error;
    /// if it does, the error is surfaced so the caller can fall back to free
    /// sampling rather than decode against a desynced matcher.
    pub fn observe_prefill(&mut self, token: u32) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        let Some(m) = self.matcher.as_mut() else {
            return Ok(());
        };
        m.consume_token(token)
            .map_err(|e| anyhow!("Matcher::consume_token(prefill {token}): {e}"))?;
        if m.is_stopped() {
            self.finished = true;
        }
        Ok(())
    }

    fn activate(&mut self) -> Result<()> {
        // `Matcher::new` accepts the `Result` directly so internal errors
        // (e.g. grammar compile failures) are kept on the matcher and
        // surfaced via `is_error()` / `get_error()` rather than panicking.
        let parser_result = self.factory.create_parser(self.schema.clone());
        let matcher = Matcher::new(parser_result);
        if matcher.is_error() {
            let err = matcher
                .get_error()
                .unwrap_or_else(|| "unknown grammar build error".to_string());
            return Err(anyhow!("llguidance Matcher build failed: {err}"));
        }
        self.matcher = Some(matcher);
        Ok(())
    }
}

/// Construct the JSON Schema enforced after the model emits `<|tool_call>`.
///
/// Gemma 4's native tool-call body is `call:NAME{key:value,…}` followed by
/// the `<tool_call|>` closer (which we let the tokenizer handle outside the
/// grammar). The schema below constrains the body to a discriminated union
/// over the tools the client offered:
///
/// ```json
/// {
///   "oneOf": [
///     { "type":"object", "properties":{ "name":{ "const":"task_complete" },
///                                       "arguments":{...task_complete schema...} },
///       "required":["name","arguments"] },
///     { "type":"object", "properties":{ "name":{ "const":"ask_to_user" },
///                                       "arguments":{...ask_to_user schema...} },
///       "required":["name","arguments"] }
///   ]
/// }
/// ```
///
/// The downstream Lumen response parser already understands
/// `call:NAME{args}<tool_call|>` ([`crate::gemma4_response`]), so this schema
/// just needs to produce the JSON-shaped args body; the surrounding tokens
/// are emitted by the model freely.
fn build_tool_grammar(tools: &[Value]) -> Result<TopLevelGrammar> {
    let mut variants: Vec<Value> = Vec::with_capacity(tools.len());
    for t in tools {
        let function = t
            .get("function")
            .ok_or_else(|| anyhow!("tool entry missing `function` field: {t}"))?;
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("tool function missing `name` string"))?;
        let parameters = function.get("parameters").cloned().unwrap_or(json!({
            "type": "object",
            "properties": {},
        }));
        variants.push(json!({
            "type": "object",
            "properties": {
                "name": { "const": name },
                "arguments": parameters,
            },
            "required": ["name", "arguments"],
            "additionalProperties": false,
        }));
    }
    let schema = json!({ "oneOf": variants });
    Ok(TopLevelGrammar::from_json_schema(schema))
}

/// Build a Lark-syntax grammar that matches Gemma 4's **native**
/// tool-call body format `call:NAME{key:value,…}` (not the JSON-style
/// `{"name":"NAME","arguments":{…}}` shape produced by
/// [`build_tool_grammar`]). Used after the model emits `<|tool_call>`
/// (id 48) so the matcher constrains every subsequent token until the
/// args body is complete; the model then emits `<tool_call|>` (id 49)
/// freely outside the grammar.
///
/// The grammar's shape, given two tools `task_complete(summary)` and
/// `python_interpreter(code)`:
///
/// ```text
/// start: tool_call
/// tool_call: "call:" tool_name "{" tool_body "}"
/// tool_name: "task_complete" | "python_interpreter"
/// tool_body: tool_task_complete_body | tool_python_interpreter_body
/// tool_task_complete_body: tc_field ("," tc_field)*
/// tc_field: "summary:" string_val | "context_keys:" string_array | ...
/// tool_python_interpreter_body: "code:" string_val
/// string_val: <[52]> <[^52]>* <[52]>
/// string_array: "[" string_val ("," string_val)* "]"
/// number_val: /-?[0-9]+(\.[0-9]+)?/
/// bool_val: "true" | "false"
/// ```
///
/// Returns a `TopLevelGrammar` ready to feed to `ParserFactory::
/// create_parser`. Limitations: this builder accepts the common JSON
/// Schema subset (`string` / `number` / `integer` / `boolean` / `array`
/// / `object` / `enum` / `const`); exotic features (`oneOf` /
/// `patternProperties` / `$ref` / format constraints) collapse to "any
/// of the supported types" with the schema's required-field constraint
/// dropped. Schemas not in the supported subset fall back to an
/// unconstrained body `<[^125]>*` (anything except `}` id 125) so the
/// matcher still terminates correctly.
fn build_tool_grammar_lark(tools: &[Value]) -> Result<TopLevelGrammar> {
    build_tool_grammar_lark_with_mode(tools, false)
}

/// Strict, dup-free variant of [`build_tool_grammar_lark`]: emits the same
/// native `call:NAME{…}` format (so [`crate::gemma4_response`] parses it
/// unchanged) but the per-tool body enforces `required` fields and forbids
/// duplicate/repeated fields. Used for `tool_choice=required`/named in
/// **Eager** mode, where a prefill-forced `<|tool_call>` would otherwise
/// leave the args body unconstrained (the empty-param defect). See
/// [`lark_body_for_object_schema`] for the `strict` body shape.
fn build_tool_grammar_lark_strict(tools: &[Value]) -> Result<TopLevelGrammar> {
    build_tool_grammar_lark_with_mode(tools, true)
}

fn build_tool_grammar_lark_with_mode(tools: &[Value], strict: bool) -> Result<TopLevelGrammar> {
    Ok(TopLevelGrammar::from_lark(lark_grammar_string(
        tools, strict,
    )?))
}

/// Build a Lark grammar matching Qwen 3.6's **nested-XML** tool-call form
/// (`<tool_call><function=NAME><parameter=KEY>\nVALUE\n</parameter>…
/// </function></tool_call>`). Always dup-free + required-enforcing: each
/// `<parameter=KEY>` appears at most once, required keys are mandatory, so the
/// model cannot emit `<function=read></function>` with no params.
///
/// Only the **tag structure** and **key set** are constrained — VALUE bytes
/// are left free (`/(.|\n)*?/`-style, terminating at the `</parameter>` the
/// grammar requires next) so the model freely writes string / JSON values that
/// [`crate::qwen3_5_tools::parse_param_value`] decodes. Constraining the value
/// shape would risk diverging from what the Qwen parser accepts; the
/// load-bearing win is "every required key present, each at most once".
pub fn build_qwen35_tool_grammar_lark(tools: &[Value]) -> Result<TopLevelGrammar> {
    Ok(TopLevelGrammar::from_lark(qwen35_lark_grammar_string(
        tools,
    )?))
}

/// Render the raw Lark text for [`build_qwen35_tool_grammar_lark`]. Split out
/// so tests can substring-match the unescaped grammar.
fn qwen35_lark_grammar_string(tools: &[Value]) -> Result<String> {
    if tools.is_empty() {
        return Err(anyhow!("build_qwen35_tool_grammar_lark: empty tools"));
    }
    let mut call_alts: Vec<String> = Vec::with_capacity(tools.len());
    let mut body_rules: Vec<(String, String)> = Vec::with_capacity(tools.len());

    for t in tools {
        let function = t
            .get("function")
            .ok_or_else(|| anyhow!("tool entry missing `function` field: {t}"))?;
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("tool function missing `name` string"))?;
        if !is_safe_ident(name) {
            return Err(anyhow!(
                "tool name {name:?} contains non-identifier characters; \
                 Qwen35 XML Lark generation refuses to escape"
            ));
        }
        let parameters = function.get("parameters").cloned().unwrap_or(json!({
            "type": "object",
            "properties": {},
        }));
        let body_rule_name = format!("q_{name}_body");
        let body_rule_rhs = qwen35_body_for_object_schema(&parameters)?;
        body_rules.push((body_rule_name.clone(), body_rule_rhs));
        // One alternative per tool so the `<function=NAME>` literal binds to
        // THAT tool's parameter set, not any tool's.
        call_alts.push(format!(
            "(\"<function={name}>\\n\" {body_rule_name} \"</function>\\n\")"
        ));
    }

    let call_alt = call_alts.join("\n          | ");
    let mut grammar = String::new();
    grammar.push_str("start: tool_call\n");
    grammar.push_str("tool_call: \"<tool_call>\\n\" function_block \"</tool_call>\"\n");
    grammar.push_str(&format!("function_block: {call_alt}\n"));
    for (rule_name, rule_rhs) in &body_rules {
        grammar.push_str(&format!("{rule_name}: {rule_rhs}\n"));
    }
    // A parameter value: any bytes (incl. newlines) up to the framing newline
    // that precedes `</parameter>`. Non-greedy so the FIRST `\n</parameter>`
    // closes the value. The framing `\n` on each side matches the renderer's
    // `<parameter=KEY>\nVALUE\n</parameter>`.
    grammar.push_str("param_value: /(.|\\n)*?/\n");
    // Generic well-formed parameter block — used by the permissive fallback
    // for tools whose schema isn't in the supported subset (unknown
    // `properties` shape or non-identifier keys). The key name is left free
    // (`/[^>\\n]*/`) but the surrounding tags are still enforced, so even the
    // fallback can't degenerate into unstructured output.
    grammar.push_str(
        "param_block: \"<parameter=\" /[^>\\n]*/ \">\\n\" param_value \"\\n</parameter>\\n\"\n",
    );
    Ok(grammar)
}

/// Render the per-tool body RHS for the Qwen35 XML grammar: a dup-free,
/// required-enforcing sequence of `<parameter=KEY>\nVALUE\n</parameter>\n`
/// blocks. Required keys (canonical/schema order) are mandatory and appear
/// first; optional keys follow, each `?`-gated. No Kleene star over the key
/// alternation ⇒ no duplicate-parameter n-gram cycle.
fn qwen35_body_for_object_schema(schema: &Value) -> Result<String> {
    let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
        // Unknown shape — allow any sequence of well-formed parameter blocks
        // (still structurally constrained: each is `<parameter=…>…</parameter>`).
        return Ok("(param_block)*".to_string());
    };
    if properties.is_empty() {
        return Ok("\"\"".to_string());
    }
    // Per-key rendered block, keyed by name for canonical ordering.
    let mut rendered: std::collections::BTreeMap<String, String> =
        std::collections::BTreeMap::new();
    for (key, _schema) in properties {
        if !is_safe_ident(key) {
            // Unsupported key shape — fall back to any well-formed blocks.
            return Ok("(param_block)*".to_string());
        }
        rendered.insert(
            key.clone(),
            format!("(\"<parameter={key}>\\n\" param_value \"\\n</parameter>\\n\")"),
        );
    }
    let required: std::collections::BTreeSet<String> = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let req_keys: Vec<&String> = rendered.keys().filter(|k| required.contains(*k)).collect();
    let opt_keys: Vec<&String> = rendered.keys().filter(|k| !required.contains(*k)).collect();

    let mut segments: Vec<String> = Vec::with_capacity(rendered.len());
    // Required first, each mandatory (no `?`). Optionals follow, each `?`.
    // Unlike the Gemma `call:NAME{…}` body there is no `,` separator between
    // parameter blocks — each block is fully self-delimiting (`</parameter>\n`)
    // — so required and optional blocks compose without comma bookkeeping.
    for k in &req_keys {
        segments.push(rendered[*k].clone());
    }
    for k in &opt_keys {
        segments.push(format!("({})?", rendered[*k]));
    }
    if segments.is_empty() {
        return Ok("(param_block)*".to_string());
    }
    Ok(segments.join(" "))
}

/// Render the raw Lark grammar text for `tools` in the given mode. Split out
/// of [`build_tool_grammar_lark_with_mode`] so tests can assert on the
/// unescaped grammar string (the `TopLevelGrammar` wrapper serializes the
/// Lark text with JSON escaping, which is awkward to substring-match).
fn lark_grammar_string(tools: &[Value], strict: bool) -> Result<String> {
    if tools.is_empty() {
        return Err(anyhow!("build_tool_grammar_lark: empty tools"));
    }
    let mut tool_names: Vec<String> = Vec::with_capacity(tools.len());
    let mut tool_body_rules: Vec<(String, String)> = Vec::with_capacity(tools.len());
    let mut extra_rules: Vec<String> = Vec::new();

    for t in tools {
        let function = t
            .get("function")
            .ok_or_else(|| anyhow!("tool entry missing `function` field: {t}"))?;
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("tool function missing `name` string"))?;
        if !is_safe_ident(name) {
            return Err(anyhow!(
                "tool name {name:?} contains non-identifier characters; \
                 Lark rule generation refuses to escape"
            ));
        }
        let parameters = function.get("parameters").cloned().unwrap_or(json!({
            "type": "object",
            "properties": {},
        }));
        let body_rule_name = format!("tool_{}_body", name);
        let body_rule_body = lark_body_for_object_schema(&parameters, &mut extra_rules, strict)?;
        tool_names.push(name.to_string());
        tool_body_rules.push((body_rule_name, body_rule_body));
    }

    let tool_name_alt = tool_names
        .iter()
        .map(|n| format!("\"{n}\""))
        .collect::<Vec<_>>()
        .join(" | ");
    let tool_body_alt = tool_body_rules
        .iter()
        .map(|(rule, _)| rule.clone())
        .collect::<Vec<_>>()
        .join(" | ");
    let tool_call_lhs = tool_names
        .iter()
        .zip(tool_body_rules.iter())
        .map(|(n, (rule, _))| format!("(\"{n}\" \"{{\" {rule} \"}}\")"))
        .collect::<Vec<_>>()
        .join("\n          | ");

    // The shape `call:NAME{body}` is encoded as one alternative per tool
    // so each tool's `{...}` body is bound to ITS schema, not any tool's
    // schema. That's stricter than `tool_name "{" tool_body "}"` (which
    // would let task_complete's keys appear inside python_interpreter's
    // braces).
    let _ = (tool_name_alt, tool_body_alt); // diagnostics: also reachable as fallback

    let mut grammar = String::new();
    grammar.push_str("start: tool_call\n");
    grammar.push_str(&format!("tool_call: \"call:\" ({tool_call_lhs})\n"));
    for (rule_name, rule_body) in &tool_body_rules {
        grammar.push_str(&format!("{rule_name}: {rule_body}\n"));
    }
    for rule in &extra_rules {
        grammar.push_str(rule);
        grammar.push('\n');
    }
    // Standard primitive rules — declared once at the bottom even if
    // some tools don't reference them. Cheap; llguidance compiles
    // unused rules away.
    grammar.push_str(&format!(
        "string_val: <[{TOK_QUOTE_DELIM}]> <[^{TOK_QUOTE_DELIM}]>* <[{TOK_QUOTE_DELIM}]>\n"
    ));
    grammar.push_str("number_val: /-?[0-9]+(\\.[0-9]+)?/\n");
    grammar.push_str("bool_val: \"true\" | \"false\"\n");

    Ok(grammar)
}

/// True when the string is a safe `[a-zA-Z_][a-zA-Z0-9_]*` identifier —
/// used to refuse tool / property names that would need escaping when
/// emitted as Lark literals.
fn is_safe_ident(s: &str) -> bool {
    let mut chars = s.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Render a Lark RHS for an object-shaped JSON Schema's body — used both
/// as a tool's top-level args envelope and recursively for nested
/// `{"type":"object"}` fields. Pushes any helper rules into
/// `extra_rules`.
///
/// `strict` selects the body shape:
///   - **`false` (permissive, Lazy/auto default)** → `field_a ("," field_b)*`:
///     ANY subset of fields in ANY order, with repetition allowed. Proven
///     for the `auto` path where the model self-emits `<|tool_call>` and
///     reliably closes the body from its training distribution.
///   - **`true` (dup-free, Eager required/named)** → required fields in a
///     fixed canonical order each emitted EXACTLY once, followed by each
///     optional field at most once in canonical order. No Kleene-star over
///     fields ⇒ no duplicate-field n-gram cycle (the failure mode that kept
///     Eager disabled — see [`Gemma4Backend::build_grammar_state`]). This is
///     the native-`call:NAME{…}` analogue of SGLang's JSON `required` +
///     `minItems:1` enforcement: it forces at-least the required keys, so a
///     quantized model can no longer emit `call:read{}` with empty params.
///
/// In both modes, the body collapses to `<[^125]>*` (any token except `}`)
/// when the schema isn't in the supported subset, and to `""` when there
/// are no properties.
fn lark_body_for_object_schema(
    schema: &Value,
    extra_rules: &mut Vec<String>,
    strict: bool,
) -> Result<String> {
    let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
        return Ok("<[^125]>*".to_string());
    };
    if properties.is_empty() {
        return Ok("\"\"".to_string());
    }
    // Per-field rendered alternative `("prop:" value_rule)`, keyed by name so
    // the strict path can pick required vs optional in canonical order.
    let mut rendered: std::collections::BTreeMap<String, String> =
        std::collections::BTreeMap::new();
    for (prop_name, prop_schema) in properties {
        if !is_safe_ident(prop_name) {
            // Unsupported key shape — fall back to permissive body.
            return Ok("<[^125]>*".to_string());
        }
        let value_rule = lark_value_for_schema(prop_schema, extra_rules, strict)?;
        rendered.insert(
            prop_name.clone(),
            format!("(\"{prop_name}:\" {value_rule})"),
        );
    }
    if !strict {
        // Permissive: any subset in any order, repetition allowed.
        let field_rule = rendered.values().cloned().collect::<Vec<_>>().join(" | ");
        return Ok(format!("({field_rule}) (\",\" ({field_rule}))*"));
    }

    // Strict (dup-free): required fields first (canonical order, each once),
    // then optional fields (canonical order, each at most once via `?`).
    // `,` separators are folded into each segment so the body never has a
    // dangling/leading comma regardless of which optionals appear.
    let required: std::collections::BTreeSet<String> = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    // Only required fields we actually know how to render (subset of
    // `properties`); unknown required keys can't be enforced.
    let req_fields: Vec<&String> = rendered.keys().filter(|k| required.contains(*k)).collect();
    let opt_fields: Vec<&String> = rendered.keys().filter(|k| !required.contains(*k)).collect();

    let mut segments: Vec<String> = Vec::with_capacity(rendered.len());
    if req_fields.is_empty() {
        // No (renderable) required fields. Keep dup-free but allow an empty
        // body: first optional standalone-or-absent, each later optional
        // comma-prefixed and `?`-gated. `(a)? (","b)? (","c)?` would permit a
        // leading comma if `a` is skipped, so make the first optional carry
        // no comma and gate the *rest* on a comma.
        let mut iter = opt_fields.iter();
        if let Some(first) = iter.next() {
            segments.push(format!("({})?", rendered[*first]));
            for f in iter {
                segments.push(format!("(\",\" {})?", rendered[*f]));
            }
        }
    } else {
        // Required fields: first standalone, the rest comma-prefixed —
        // all mandatory (no `?`). Optionals follow, each comma-prefixed
        // and `?`-gated (a comma is always legal here because at least one
        // required field precedes them).
        let mut req_iter = req_fields.iter();
        let first_req = req_iter.next().expect("req_fields non-empty");
        segments.push(rendered[*first_req].clone());
        for f in req_iter {
            segments.push(format!("\",\" {}", rendered[*f]));
        }
        for f in &opt_fields {
            segments.push(format!("(\",\" {})?", rendered[*f]));
        }
    }
    if segments.is_empty() {
        // Defensive: properties non-empty but nothing rendered (shouldn't
        // happen). Permissive fallback keeps the matcher terminating.
        return Ok("<[^125]>*".to_string());
    }
    Ok(segments.join(" "))
}

/// Render a Lark RHS for a single JSON Schema value (string / number /
/// boolean / array / object / enum / const). Falls back to
/// `<[^44,125]>*` (anything except `,` or `}`) for unsupported shapes
/// so the rest of the body still validates.
fn lark_value_for_schema(
    schema: &Value,
    extra_rules: &mut Vec<String>,
    strict: bool,
) -> Result<String> {
    if let Some(c) = schema.get("const") {
        return Ok(lark_const_literal(c));
    }
    if let Some(en) = schema.get("enum").and_then(Value::as_array) {
        let alts: Vec<String> = en.iter().map(lark_const_literal).collect();
        return Ok(format!("({})", alts.join(" | ")));
    }
    let type_str = schema.get("type").and_then(Value::as_str).unwrap_or("");
    match type_str {
        "string" => Ok("string_val".to_string()),
        "number" | "integer" => Ok("number_val".to_string()),
        "boolean" => Ok("bool_val".to_string()),
        "array" => {
            let items = schema.get("items").cloned().unwrap_or(json!({}));
            let item_rule = lark_value_for_schema(&items, extra_rules, strict)?;
            Ok(format!("(\"[\" ({item_rule} (\",\" {item_rule})*)? \"]\")"))
        }
        "object" => {
            let body = lark_body_for_object_schema(schema, extra_rules, strict)?;
            Ok(format!("(\"{{\" {body} \"}}\")"))
        }
        // Unknown / mixed type → permissive non-greedy "anything until
        // field boundary (`,` id 44 or `}` id 125)".
        _ => Ok("<[^44,125]>*".to_string()),
    }
}

/// Encode a JSON `const` value as a Lark literal — string in
/// Gemma-quote form, scalar verbatim. Unsupported (e.g. nested object
/// const) falls back to permissive content.
fn lark_const_literal(c: &Value) -> String {
    match c {
        Value::String(s) => {
            // Strings appear inside `<|"|>...<|"|>`. Body is bytes.
            // Escape `\` and `"` for Lark literal — Lark's literal
            // syntax follows Python's.
            let escaped = s.replace('\\', "\\\\").replace('"', "\\\"");
            format!("(<[{TOK_QUOTE_DELIM}]> \"{escaped}\" <[{TOK_QUOTE_DELIM}]>)")
        }
        Value::Number(n) => format!("\"{n}\""),
        Value::Bool(true) => "\"true\"".to_string(),
        Value::Bool(false) => "\"false\"".to_string(),
        Value::Null => "\"null\"".to_string(),
        _ => "<[^44,125]>*".to_string(),
    }
}

/// Build a real Gemma 4 vocab-aware [`TokEnv`] from `tokenizer.json` so
/// llguidance masks line up with the model's actual token ids (vocab
/// 262144). EOS is set to `<eos>` (id 1) and the chat-style turn delimiter
/// `<turn|>` (id 106) is registered as `tok_end_of_turn` so the grammar
/// stop machinery recognises both.
pub fn build_tok_env_from_tokenizer_json<P: AsRef<Path>>(path: P) -> Result<TokEnv> {
    let raw = std::fs::read_to_string(path.as_ref())
        .with_context(|| format!("read tokenizer.json: {:?}", path.as_ref()))?;
    let tj: Value = serde_json::from_str(&raw).context("parse tokenizer.json")?;
    let token_bytes =
        token_bytes_from_tokenizer_json(&tj).context("token_bytes_from_tokenizer_json")?;
    if token_bytes.is_empty() {
        return Err(anyhow!("tokenizer.json produced empty token table"));
    }
    let mut info = TokRxInfo::new(token_bytes.len() as u32, TOK_EOS);
    info.tok_end_of_turn = Some(TOK_TURN_CLOSE);
    let trie = TokTrie::from(&info, &token_bytes);
    Ok(Arc::new(ApproximateTokEnv::new(trie)) as TokEnv)
}

/// Build a request-shareable [`ParserFactory`] backed by the real Gemma 4
/// tokenizer. Construction is the expensive part (~10–50 ms — slicer
/// precomputes general regex masks once); per-request `create_parser`
/// calls are cheap. Cache the returned `Arc<ParserFactory>` for the
/// lifetime of the backend.
pub fn shared_factory_from_tokenizer<P: AsRef<Path>>(path: P) -> Result<Arc<ParserFactory>> {
    let tok_env = build_tok_env_from_tokenizer_json(path)?;
    let factory = ParserFactory::new_simple(&tok_env)
        .map_err(|e| anyhow!("ParserFactory::new_simple: {e}"))?;
    Ok(Arc::new(factory))
}

/// Test-only placeholder using `ApproximateTokEnv::single_byte_env`. Not
/// suitable for live sampling — masks would not align with the Gemma 4
/// vocabulary. Reserved for `#[cfg(test)]` smoke checks that only need a
/// working factory shape.
#[cfg(test)]
fn shared_factory_placeholder() -> Arc<ParserFactory> {
    let tok_env = ApproximateTokEnv::single_byte_env();
    let factory = ParserFactory::new_simple(&tok_env)
        .expect("ParserFactory::new_simple(single_byte_env) must succeed");
    Arc::new(factory)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tools() -> Vec<Value> {
        vec![
            json!({
                "type": "function",
                "function": {
                    "name": "task_complete",
                    "description": "Signal task completion",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary": { "type": "string" }
                        },
                        "required": ["summary"],
                    },
                },
            }),
            json!({
                "type": "function",
                "function": {
                    "name": "ask_to_user",
                    "description": "Ask the user a clarifying question",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": { "type": "string" },
                            "options": {
                                "type": "array",
                                "items": { "type": "string" },
                            },
                        },
                        "required": ["question"],
                    },
                },
            }),
        ]
    }

    #[test]
    fn grammar_builds_for_two_tools() {
        let g = build_tool_grammar(&sample_tools()).expect("build grammar");
        // Smoke: roundtrip the grammar to JSON so we can eyeball it if the
        // test fails. `TopLevelGrammar` is Serialize.
        let s = serde_json::to_string(&g).expect("serialise grammar");
        assert!(s.contains("task_complete"));
        assert!(s.contains("ask_to_user"));
    }

    #[test]
    fn grammar_builds_with_minimal_function_def() {
        let tools = vec![json!({
            "type": "function",
            "function": { "name": "noop" }
        })];
        let g = build_tool_grammar(&tools).expect("build grammar with empty params");
        let s = serde_json::to_string(&g).expect("serialise grammar");
        assert!(s.contains("noop"));
    }

    #[test]
    fn empty_tools_rejected() {
        let factory = shared_factory_placeholder();
        let r = Gemma4GrammarState::new(factory, &[], GrammarMode::Lazy);
        assert!(r.is_err());
    }

    #[test]
    fn lazy_state_starts_inactive() {
        let factory = shared_factory_placeholder();
        let state = Gemma4GrammarState::new(factory, &sample_tools(), GrammarMode::Lazy)
            .expect("build lazy state");
        assert!(!state.is_active(), "lazy mode is inactive until trigger");
    }

    #[test]
    fn eager_state_starts_active() {
        let factory = shared_factory_placeholder();
        let state = Gemma4GrammarState::new(factory, &sample_tools(), GrammarMode::Eager)
            .expect("build eager state");
        assert!(
            state.is_active(),
            "eager mode is constraining from the first decode step"
        );
    }

    #[test]
    fn lazy_observe_non_trigger_stays_inactive() {
        let factory = shared_factory_placeholder();
        let mut state = Gemma4GrammarState::new(factory, &sample_tools(), GrammarMode::Lazy)
            .expect("build lazy state");
        // Any token other than `<|tool_call>` (id 48) is free-text and
        // must not flip the matcher on. Using a small id keeps the test
        // independent of the placeholder vocab's specific symbols.
        state.observe(42).expect("observe(42) free-text");
        assert!(!state.is_active(), "free text does not activate lazy mode");
    }

    #[test]
    fn observe_when_inactive_apply_mask_is_noop() {
        let factory = shared_factory_placeholder();
        let mut state = Gemma4GrammarState::new(factory, &sample_tools(), GrammarMode::Lazy)
            .expect("build lazy state");
        // Synthetic logits buffer sized to the placeholder vocab.
        let mut logits = vec![1.0_f32; 262];
        let masked = state
            .apply_mask_to_logits(&mut logits)
            .expect("apply_mask noop");
        assert_eq!(masked, 0, "inactive grammar should not mask any logit");
        assert!(
            logits
                .iter()
                .all(|v| v.is_finite() && (*v - 1.0).abs() < 1e-6),
            "logits buffer must be untouched when grammar is inactive"
        );
    }

    #[test]
    fn lark_grammar_builds_for_two_tools() {
        let g = build_tool_grammar_lark(&sample_tools()).expect("build lark grammar");
        let s = serde_json::to_string(&g).expect("serialise lark grammar");
        assert!(s.contains("task_complete"), "task_complete name in grammar");
        assert!(s.contains("ask_to_user"), "ask_to_user name in grammar");
        assert!(s.contains("call:"), "native opener present");
        assert!(
            s.contains("string_val"),
            "primitive string_val rule present"
        );
        assert!(
            s.contains("<[52]>") || s.contains("[52]"),
            "quote token reference"
        );
    }

    #[test]
    fn lark_grammar_handles_ayla_tools() {
        // Snapshot of the two tools Ayla ships in production: task_complete
        // (string + array of strings) and python_interpreter (string code).
        // Catches regressions in the schema → Lark mapping for the array
        // type, which the JSON-Schema path handled via `"items":{...}`
        // and the Lark path must translate to `"[" item ("," item)* "]"`.
        let tools = vec![
            json!({
                "type": "function",
                "function": {
                    "name": "task_complete",
                    "description": "Signal task done",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary": { "type": "string" },
                            "context_keys": {
                                "type": "array",
                                "items": { "type": "string" }
                            },
                            "next_gate": { "type": "string" },
                            "reason": { "type": "string" }
                        },
                        "required": ["summary"]
                    }
                }
            }),
            json!({
                "type": "function",
                "function": {
                    "name": "python_interpreter",
                    "parameters": {
                        "type": "object",
                        "properties": { "code": { "type": "string" } },
                        "required": ["code"]
                    }
                }
            }),
        ];
        let g = build_tool_grammar_lark(&tools).expect("build ayla lark grammar");
        let s = serde_json::to_string(&g).expect("serialise grammar");
        assert!(s.contains("summary"));
        assert!(s.contains("context_keys"));
        assert!(s.contains("code"));
        assert!(s.contains("python_interpreter"));
    }

    #[test]
    fn lark_grammar_state_starts_active_in_eager() {
        let factory = shared_factory_placeholder();
        let state = Gemma4GrammarState::new_lark(factory, &sample_tools(), GrammarMode::Eager)
            .expect("build eager lark state");
        assert!(
            state.is_active(),
            "eager Lark grammar constrains from step 0"
        );
    }

    #[test]
    fn lark_grammar_rejects_unsafe_tool_name() {
        // Tool names like `foo.bar` or `foo-bar` would need Lark
        // escaping; the builder refuses early instead of generating a
        // grammar that won't parse.
        let tools = vec![json!({
            "type": "function",
            "function": { "name": "foo-bar", "parameters": { "type":"object","properties":{} } }
        })];
        let r = build_tool_grammar_lark(&tools);
        assert!(r.is_err(), "unsafe tool name must be rejected");
    }

    #[test]
    fn apply_mask_on_eager_state_with_mismatched_vocab_errors() {
        // Sanity that vocab-size mismatch surfaces as an error instead of
        // silently sampling against a stale mask. Eager mode activates the
        // matcher immediately so compute_mask runs.
        let factory = shared_factory_placeholder();
        let mut state = Gemma4GrammarState::new(factory, &sample_tools(), GrammarMode::Eager)
            .expect("build eager state");
        let mut logits = vec![1.0_f32; 8]; // far smaller than placeholder vocab
        let r = state.apply_mask_to_logits(&mut logits);
        assert!(r.is_err(), "vocab mismatch must error");
    }

    // ── WS-C #1: strict (dup-free, required-enforcing) Lark path ──

    #[test]
    fn strict_lark_builds_same_native_format() {
        // Strict still emits the native `call:NAME{…}` format the Gemma
        // response parser understands — NOT JSON. This is the load-bearing
        // invariant: switching to JSON-union would break the parser.
        let s = lark_grammar_string(&sample_tools(), true).expect("build strict lark");
        assert!(
            s.contains("call:"),
            "native opener preserved in strict mode"
        );
        assert!(s.contains("task_complete"));
        assert!(s.contains("ask_to_user"));
        // `summary` is required on task_complete; `question` required on
        // ask_to_user — both must appear in their body rules.
        assert!(s.contains("summary:"));
        assert!(s.contains("question:"));
    }

    #[test]
    fn strict_lark_body_is_dup_free() {
        // The n-gram cycle that kept Eager disabled came from the permissive
        // body `(field) ("," (field))*` — a Kleene star over the field
        // alternation, which lets the model repeat `summary:…,summary:…`
        // forever. The strict body must NOT contain that field-repeat.
        let permissive = lark_grammar_string(&sample_tools(), false).unwrap();
        let strict = lark_grammar_string(&sample_tools(), true).unwrap();
        // Permissive: body is `(...) ("," (...))*` — the `))*` field-repeat.
        assert!(
            permissive.contains("))*"),
            "permissive body uses a Kleene star over the field alternation"
        );
        // Strict: required fields are a fixed sequence, optionals are
        // `(\",\" field)?` — never a `*`-repeat of the field alternation. No
        // body line may contain the `))*` field-repeat marker.
        for line in strict.lines() {
            if line.starts_with("tool_") && line.contains("_body:") {
                assert!(
                    !line.contains("))*"),
                    "strict body line must not repeat fields with a Kleene star: {line}"
                );
            }
        }
    }

    #[test]
    fn strict_lark_enforces_required_field_no_optional_gate() {
        // A tool whose only property is required must emit that field
        // unconditionally (no `?` gate around it) so an empty `call:read{}`
        // is grammatically impossible — the empty-param defect fix.
        let tools = vec![json!({
            "type": "function",
            "function": {
                "name": "read",
                "parameters": {
                    "type": "object",
                    "properties": { "path": { "type": "string" } },
                    "required": ["path"]
                }
            }
        })];
        let s = lark_grammar_string(&tools, true).unwrap();
        // The sole required field is emitted as a bare (non-`?`) body, so the
        // body cannot be empty. Locate the body rule line and assert it is
        // exactly the required field with no `?` gate.
        let body_line = s
            .lines()
            .find(|l| l.starts_with("tool_read_body:"))
            .expect("read body rule present");
        assert!(
            body_line.contains("\"path:\" string_val"),
            "required field present: {body_line}"
        );
        assert!(
            !body_line.contains(")?"),
            "required-only field must not be optional-gated: {body_line}"
        );
    }

    #[test]
    fn permissive_lark_unchanged_for_auto_path() {
        // Guard: the auto/Lazy path must be byte-identical to the pre-fix
        // grammar. The permissive body keeps the `(field) ("," (field))*`
        // shape with the field alternation and the Kleene star.
        let s = lark_grammar_string(&sample_tools(), false).unwrap();
        assert!(s.contains("call:"));
        // task_complete has a single property `summary`, so its body is
        // `(("summary:" string_val)) ("," (("summary:" string_val)))*`.
        let body_line = s
            .lines()
            .find(|l| l.starts_with("tool_task_complete_body:"))
            .expect("task_complete body present");
        assert!(
            body_line.contains("))*"),
            "permissive body retains the field Kleene star: {body_line}"
        );
    }

    #[test]
    fn strict_lark_state_starts_active_in_eager() {
        let factory = shared_factory_placeholder();
        let state =
            Gemma4GrammarState::new_lark_strict(factory, &sample_tools(), GrammarMode::Eager)
                .expect("build eager strict lark state");
        assert!(
            state.is_active(),
            "eager strict Lark grammar constrains from step 0 (required tool_choice path)"
        );
    }

    // ── WS-C #2: Qwen 3.6 nested-XML tool grammar ──

    #[test]
    fn qwen35_grammar_builds_native_xml_format() {
        // Must emit the nested-XML tags the Qwen35 parser reads — NOT
        // `call:NAME{…}` (Gemma) nor JSON-union. This is the load-bearing
        // invariant: a wrong format would make the matcher reject every
        // model token.
        let s = qwen35_lark_grammar_string(&sample_tools()).expect("build qwen35 grammar");
        assert!(s.contains("<tool_call>"), "tool_call opener tag");
        assert!(s.contains("</tool_call>"), "tool_call closer tag");
        assert!(
            s.contains("<function=task_complete>"),
            "function tag per tool"
        );
        assert!(s.contains("<function=ask_to_user>"));
        assert!(s.contains("<parameter="), "parameter tag");
        assert!(s.contains("</parameter>"));
        assert!(s.contains("</function>"));
        // No Gemma native form leaked in. The Gemma body opens with the
        // QUOTED literal `"call:"` (an emitted terminal), distinct from this
        // grammar's `tool_call:` rule NAME — assert the quoted literal is
        // absent rather than the bare substring.
        assert!(
            !s.contains("\"call:\""),
            "must not emit Gemma quoted call: literal"
        );
    }

    #[test]
    fn qwen35_grammar_enforces_required_param() {
        // A tool whose only property is required must emit that parameter
        // block unconditionally (no `?` gate) so `<function=read></function>`
        // (empty params) is grammatically impossible — the empty-param defect.
        let tools = vec![json!({
            "type": "function",
            "function": {
                "name": "read",
                "parameters": {
                    "type": "object",
                    "properties": { "path": { "type": "string" } },
                    "required": ["path"]
                }
            }
        })];
        let s = qwen35_lark_grammar_string(&tools).unwrap();
        let body_line = s
            .lines()
            .find(|l| l.starts_with("q_read_body:"))
            .expect("read body rule present");
        assert!(
            body_line.contains("<parameter=path>"),
            "required param present: {body_line}"
        );
        assert!(
            !body_line.contains(")?"),
            "required-only param must not be optional-gated: {body_line}"
        );
    }

    #[test]
    fn qwen35_grammar_body_is_dup_free() {
        // No Kleene star over the parameter-block alternation in a body rule
        // ⇒ no duplicate-parameter n-gram cycle (the failure mode that kept
        // Gemma's Eager Lark disabled).
        let s = qwen35_lark_grammar_string(&sample_tools()).unwrap();
        for line in s.lines() {
            if line.starts_with("q_") && line.contains("_body:") {
                assert!(
                    !line.contains(")*"),
                    "qwen35 body line must not repeat params with a Kleene star: {line}"
                );
            }
        }
    }

    #[test]
    fn qwen35_grammar_optional_params_are_gated() {
        // ask_to_user: `question` required, `options` optional → options must
        // be `?`-gated, question must not be.
        let s = qwen35_lark_grammar_string(&sample_tools()).unwrap();
        let body_line = s
            .lines()
            .find(|l| l.starts_with("q_ask_to_user_body:"))
            .expect("ask_to_user body present");
        assert!(body_line.contains("<parameter=question>"));
        assert!(body_line.contains("<parameter=options>"));
        assert!(
            body_line.contains(")?"),
            "optional param must be `?`-gated: {body_line}"
        );
    }

    #[test]
    fn qwen35_grammar_state_active_in_eager() {
        let factory = shared_factory_placeholder();
        let state =
            Gemma4GrammarState::new_qwen35_xml(factory, &sample_tools(), GrammarMode::Eager, None)
                .expect("build eager qwen35 xml state");
        assert!(
            state.is_active(),
            "eager qwen35 grammar constrains from step 0 (required tool_choice path)"
        );
    }

    #[test]
    fn qwen35_grammar_eager_matcher_compiles() {
        // The Eager constructor calls `activate()` → `create_parser` →
        // `Matcher::new`, which surfaces a Lark COMPILE error as
        // `matcher.is_error()` → our `Err`. A successful build here proves the
        // generated Lark (incl. the `param_value` / `param_block` regexes)
        // actually compiles under llguidance, not just that the string looks
        // right. Uses the placeholder single-byte factory.
        let factory = shared_factory_placeholder();
        let state =
            Gemma4GrammarState::new_qwen35_xml(factory, &sample_tools(), GrammarMode::Eager, None)
                .expect("eager qwen35 grammar must compile under llguidance");
        assert!(state.is_active());
    }

    #[test]
    fn qwen35_grammar_fallback_param_block_compiles() {
        // A tool whose parameters aren't a supported object schema falls back
        // to `(param_block)*`. Verify that path also compiles (the `param_block`
        // rule + its `/[^>\n]*/` key regex).
        let tools = vec![json!({
            "type": "function",
            "function": { "name": "raw", "parameters": { "type": "string" } }
        })];
        let s = qwen35_lark_grammar_string(&tools).unwrap();
        assert!(s.contains("param_block"), "fallback uses param_block: {s}");
        let factory = shared_factory_placeholder();
        let state = Gemma4GrammarState::new_qwen35_xml(factory, &tools, GrammarMode::Eager, None)
            .expect("fallback qwen35 grammar must compile under llguidance");
        assert!(state.is_active());
    }

    #[test]
    fn qwen35_grammar_rejects_unsafe_tool_name() {
        let tools = vec![json!({
            "type": "function",
            "function": { "name": "foo-bar", "parameters": { "type":"object","properties":{} } }
        })];
        let r = build_qwen35_tool_grammar_lark(&tools);
        assert!(r.is_err(), "unsafe tool name must be rejected");
    }

    #[test]
    fn strict_lark_handles_no_required_fields() {
        // A schema with only optional properties must still build (dup-free,
        // every field `?`-gated, empty body allowed) — no panic, no Kleene
        // star over fields.
        let tools = vec![json!({
            "type": "function",
            "function": {
                "name": "search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" },
                        "limit": { "type": "integer" }
                    }
                    // no "required"
                }
            }
        })];
        let s = lark_grammar_string(&tools, true).expect("build no-required strict lark");
        assert!(s.contains("query:"));
        assert!(s.contains("limit:"));
        let body_line = s
            .lines()
            .find(|l| l.starts_with("tool_search_body:"))
            .expect("search body present");
        assert!(
            !body_line.contains("))*"),
            "no-required strict body must still be dup-free: {body_line}"
        );
    }
}
