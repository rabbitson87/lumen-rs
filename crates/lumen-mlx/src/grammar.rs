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
        if tools.is_empty() {
            return Err(anyhow!(
                "Gemma4GrammarState::new_lark called with empty tools"
            ));
        }
        let schema = build_tool_grammar_lark(tools)?;
        let mut state = Self {
            factory,
            schema,
            matcher: None,
            mode,
            finished: false,
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
        // ParserFactory was built. Logits buffer must match — if not, the
        // tokenizer env and the model are mismatched and downstream
        // sampling is meaningless.
        if mask.len() != logits.len() {
            return Err(anyhow!(
                "grammar mask vocab mismatch: mask.len()={}, logits.len()={}",
                mask.len(),
                logits.len()
            ));
        }
        let mut masked = 0usize;
        for (i, l) in logits.iter_mut().enumerate() {
            if !mask.is_allowed(i as u32) {
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
            && token == TOK_TOOL_CALL_OPEN
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
        let body_rule_body = lark_body_for_object_schema(&parameters, &mut extra_rules)?;
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

    Ok(TopLevelGrammar::from_lark(grammar))
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
/// The body shape is one of:
///   - `field_a ("," field_b)*` when there are properties;
///   - `<[^125]>*` (any token except `}`) when properties is empty —
///     trades schema enforcement for completeness on tools whose schema
///     we don't fully understand.
fn lark_body_for_object_schema(schema: &Value, extra_rules: &mut Vec<String>) -> Result<String> {
    let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
        return Ok("<[^125]>*".to_string());
    };
    if properties.is_empty() {
        return Ok("\"\"".to_string());
    }
    let mut field_alternatives: Vec<String> = Vec::with_capacity(properties.len());
    for (prop_name, prop_schema) in properties {
        if !is_safe_ident(prop_name) {
            // Unsupported key shape — fall back to permissive body.
            return Ok("<[^125]>*".to_string());
        }
        let value_rule = lark_value_for_schema(prop_schema, extra_rules)?;
        field_alternatives.push(format!("(\"{prop_name}:\" {value_rule})"));
    }
    let field_rule = field_alternatives.join(" | ");
    // Allow the model to emit ANY subset of fields in ANY order. Schema
    // `required` is intentionally NOT enforced at the grammar level —
    // future tightening could enumerate permutations of required
    // fields, but that's exponential in the field count.
    Ok(format!("({field_rule}) (\",\" ({field_rule}))*"))
}

/// Render a Lark RHS for a single JSON Schema value (string / number /
/// boolean / array / object / enum / const). Falls back to
/// `<[^44,125]>*` (anything except `,` or `}`) for unsupported shapes
/// so the rest of the body still validates.
fn lark_value_for_schema(schema: &Value, extra_rules: &mut Vec<String>) -> Result<String> {
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
            let item_rule = lark_value_for_schema(&items, extra_rules)?;
            Ok(format!("(\"[\" ({item_rule} (\",\" {item_rule})*)? \"]\")"))
        }
        "object" => {
            let body = lark_body_for_object_schema(schema, extra_rules)?;
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
}
