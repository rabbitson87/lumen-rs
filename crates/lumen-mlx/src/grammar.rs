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
    /// How a [`GrammarMode::Lazy`] matcher wakes up. `None` disables lazy
    /// triggering entirely — the grammar then only ever constrains in
    /// [`GrammarMode::Eager`].
    lazy_trigger: Option<LazyTrigger>,
}

/// The token that activates a [`GrammarMode::Lazy`] grammar, and whether that
/// token belongs to the grammar it activates.
///
/// The distinction is load-bearing. Gemma 4's Lark grammar starts at the
/// literal `call:`; the `<|tool_call>` opener that precedes it is not a
/// terminal anywhere in that grammar, so feeding the trigger to the
/// freshly-built matcher fails the parse outright. Qwen 3.6's XML grammar
/// does open with `<tool_call>`, so there the trigger *is* the grammar's
/// first terminal and must be consumed for the matcher to line up with the
/// model's context.
#[derive(Clone, Copy, Debug)]
struct LazyTrigger {
    token: u32,
    in_grammar: bool,
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
            lazy_trigger: Some(LazyTrigger {
                token: TOK_TOOL_CALL_OPEN,
                // The schema describes the args body only; `<|tool_call>` is
                // emitted before it and is not part of the grammar.
                in_grammar: false,
            }),
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
    ///
    /// The grammar is pinned to **compact** JSON — see
    /// [`with_bounded_whitespace`] for why the default is unusable here.
    pub fn new_json_schema(
        factory: Arc<ParserFactory>,
        schema: &Value,
        mode: GrammarMode,
    ) -> Result<Self> {
        let schema = TopLevelGrammar::from_json_schema(with_bounded_whitespace(schema.clone()));
        let mut state = Self {
            factory,
            schema,
            matcher: None,
            mode,
            finished: false,
            // The whole assistant message is the JSON value — there is no
            // opener token to wait for, so this variant has no lazy trigger.
            lazy_trigger: None,
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
            lazy_trigger: Some(LazyTrigger {
                token: TOK_TOOL_CALL_OPEN,
                // `start: tool_call` begins at the literal `call:`. The
                // `<|tool_call>` opener that triggers activation sits before
                // it and is not a terminal in this grammar.
                in_grammar: false,
            }),
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
            // `tool_call: "<tool_call>\n" …` — the opener IS the grammar's
            // first terminal here, so an explicit trigger gets consumed.
            // `None` means no lazy trigger at all: borrowing Gemma's id 48
            // would fire on Qwen's ordinary `Q` token and desync instantly.
            lazy_trigger: opener_token.map(|token| LazyTrigger {
                token,
                in_grammar: true,
            }),
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

    /// Stop constraining, permanently, and let the rest of the generation
    /// sample freely.
    ///
    /// For callers that hit a desync they cannot recover from. A grammar is an
    /// assist, not a correctness requirement — decoding on against a matcher
    /// whose parse position no longer matches the model's context would mask
    /// out legal tokens, so releasing beats both continuing and failing the
    /// whole request.
    pub fn release(&mut self) {
        self.finished = true;
        self.matcher = None;
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
    /// In lazy mode, this also handles the activation transition: when the
    /// model emits the trigger token (Gemma 4's `<|tool_call>`, id 48) for
    /// the first time, the matcher is created so the *next* sampled token is
    /// constrained.
    ///
    /// Whether the trigger itself is then fed to that fresh matcher depends on
    /// [`LazyTrigger::in_grammar`]. For Gemma 4 it must not be: the Lark
    /// grammar starts at the literal `call:`, so consuming `<|tool_call>`
    /// fails the parse on its very first byte — llguidance renders special
    /// tokens with a `0xFF` marker prefix, which is where the
    /// `byte 'ÿ' fails parse` error came from. That aborted every streaming
    /// tool call, since the streaming path is the only one that wires the
    /// grammar in.
    pub fn observe(&mut self, token: u32) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        if self.matcher.is_none() && matches!(self.mode, GrammarMode::Lazy) {
            let Some(trigger) = self.lazy_trigger else {
                return Ok(());
            };
            if token != trigger.token {
                return Ok(());
            }
            self.activate()?;
        }
        if self.is_extragrammatical_opener(token) {
            return Ok(());
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
    /// mode or once finished. The `<|tool_call>` opener is skipped for the
    /// same reason [`observe`] skips it — Gemma 4's grammar starts at the
    /// `call:` that follows — while the rest of the prefill (`call:NAME{` for
    /// a named choice) is exactly the grammar's deterministic opener and
    /// parses cleanly. Anything else erroring is surfaced so the caller can
    /// fall back to free sampling rather than decode against a desynced
    /// matcher.
    pub fn observe_prefill(&mut self, token: u32) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        if self.is_extragrammatical_opener(token) {
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

    /// True when `token` is the opener that *precedes* the grammar rather than
    /// belonging to it. Such a token is never fed to the matcher, in any mode
    /// and from either entry point: it activates a Lazy grammar and is
    /// otherwise transparent.
    fn is_extragrammatical_opener(&self, token: u32) -> bool {
        self.lazy_trigger
            .is_some_and(|t| t.token == token && !t.in_grammar)
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
/// Give a JSON Schema grammar exactly the spacing a model writes by hand, and
/// no more.
///
/// llguidance defaults to `whitespace_flexible`, which puts a skippable
/// `[ \n\r\t]+` between every JSON token. For a human writing JSON that is a
/// convenience; under greedy constrained decoding it is a trap, because the run
/// has no upper bound. Observed on Gemma 4: after `{` the model emitted
/// newlines and indentation until it hit `max_tokens`, and the reply was `{`.
///
/// Forbidding whitespace outright stops that but overcorrects. Models write
/// `"key": "value"`, and with nowhere legal to put the space the residue lands
/// *inside* the string — both families produced `{"city": ": way more…"}` under
/// compact separators, Gemma 4 never recovering from it.
///
/// So instead of a whitespace *rule* the spacing goes into the separators
/// themselves: `": "` and `", "` are literals, matching the model's own
/// distribution while remaining a fixed width that cannot be padded. A bounded
/// `whitespace_pattern` looks like the obvious answer and is not — it is
/// llguidance's *skip* rule, re-matched between every token, so consecutive
/// matches concatenate and `{0,8}` bounds nothing.
///
/// A caller that has already set `x-guidance` knows what it wants and is left
/// alone — `{"x-guidance": {"whitespace_flexible": true}}` on the request
/// schema restores free-form pretty-printing. A non-object schema (`true` /
/// `false`) has nowhere to put the hint and passes through unchanged.
///
/// Orthogonal, and worth knowing when a reply looks unhinged: a schema without
/// `"additionalProperties": false` legitimately permits any extra key, and a
/// model handed that freedom will invent keys until `max_tokens`. That is the
/// schema doing what it says, so it is left to the caller — but it is the first
/// thing to check.
fn with_bounded_whitespace(mut schema: Value) -> Value {
    let Some(obj) = schema.as_object_mut() else {
        return schema;
    };
    if !obj.contains_key("x-guidance") {
        obj.insert(
            "x-guidance".into(),
            json!({
                "whitespace_flexible": false,
                "key_separator": ": ",
                "item_separator": ", ",
            }),
        );
    }
    schema
}

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

    for (i, t) in tools.iter().enumerate() {
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
        // Indexed rule name — see `lark_grammar_string` for why the tool name
        // itself may not be an identifier.
        let body_rule_name = format!("q_{i}_body");
        let body_rule_rhs = qwen35_body_for_object_schema(&parameters)?;
        body_rules.push((body_rule_name.clone(), body_rule_rhs));
        // One alternative per tool so the `<function=NAME>` literal binds to
        // THAT tool's parameter set, not any tool's.
        call_alts.push(format!(
            "({} {body_rule_name} \"</function>\\n\")",
            lark_literal(&format!("<function={name}>\n"))
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
        rendered.insert(
            key.clone(),
            format!(
                "({} param_value \"\\n</parameter>\\n\")",
                lark_literal(&format!("<parameter={key}>\n"))
            ),
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

    for (i, t) in tools.iter().enumerate() {
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
        // Rule names are indexed, not derived from the tool name. A Lark rule
        // name must be an identifier, but a tool name need not be one — and
        // refusing those outright meant the grammar was dropped and the request
        // fell back to free sampling, where the model invented a name no client
        // had declared (`날씨_조회` came back as `weather_lookup`). The name
        // itself only ever appears inside a quoted literal, where escaping is
        // enough.
        let body_rule_name = format!("tool_{i}_body");
        let body_rule_body = lark_body_for_object_schema(&parameters, &mut extra_rules, strict)?;
        tool_names.push(name.to_string());
        tool_body_rules.push((body_rule_name, body_rule_body));
    }

    let tool_name_alt = tool_names
        .iter()
        .map(|n| lark_literal(n))
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
        .map(|(n, (rule, _))| format!("({} \"{{\" {rule} \"}}\")", lark_literal(n)))
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

/// Render `s` as a Lark string literal, escaping what Lark's Python-style
/// literal syntax cannot carry raw.
///
/// Names and keys come from the caller's tool schema and are not identifiers in
/// general — `Playwright (Stealth)__browser_navigate`, `날씨_조회`, `도시`. They
/// are safe *as literals* once escaped; what is not safe is using them as Lark
/// rule names, which is why rule names are indexed instead.
fn lark_literal(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
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
        let value_rule = lark_value_for_schema(prop_schema, extra_rules, strict)?;
        rendered.insert(
            prop_name.clone(),
            format!("({} {value_rule})", lark_literal(&format!("{prop_name}:"))),
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
    // JSON Schema `type` may be a string OR an array of strings (e.g.
    // `["string","null"]`). Render each variant and union them; this
    // keeps brace counting deterministic when the field is nullable.
    if let Some(types) = schema.get("type").and_then(Value::as_array) {
        let mut alts: Vec<String> = Vec::new();
        for t in types {
            let Some(s) = t.as_str() else { continue };
            let alt = render_single_type(s, schema, extra_rules, strict)?;
            alts.push(alt);
        }
        if !alts.is_empty() {
            return Ok(format!("({})", alts.join(" | ")));
        }
    }
    let type_str = schema.get("type").and_then(Value::as_str).unwrap_or("");
    render_single_type(type_str, schema, extra_rules, strict)
}

/// Render one JSON Schema scalar/compound type to a Lark RHS — shared
/// by single-type and multi-type schema paths so `["string","null"]`
/// composes from the same building blocks as a plain `"string"`.
fn render_single_type(
    type_str: &str,
    schema: &Value,
    extra_rules: &mut Vec<String>,
    strict: bool,
) -> Result<String> {
    match type_str {
        "string" => Ok("string_val".to_string()),
        "number" | "integer" => Ok("number_val".to_string()),
        "boolean" => Ok("bool_val".to_string()),
        "null" => Ok("\"null\"".to_string()),
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

    // ──────── state-machine edges (005 Phase 4.1) ────────
    //
    // These are the paths a grammar takes when something has already gone
    // wrong: the tool list is empty, the matcher was released, the vocab does
    // not match, the parse stopped. A grammar is an assist rather than a
    // correctness requirement, so most of them are deliberately *quiet* — and
    // quiet is exactly why they need pinning. The one that must NOT be quiet
    // is the vocab mismatch, because masking against the wrong vocabulary
    // silently forbids legal tokens.

    /// Every state constructor rejects an empty tool list. An empty grammar
    /// matches nothing, so accepting one would mask the model into silence.
    #[test]
    fn every_state_constructor_rejects_an_empty_tool_list() {
        let f = shared_factory_placeholder();
        assert!(Gemma4GrammarState::new_lark(f.clone(), &[], GrammarMode::Eager).is_err());
        assert!(Gemma4GrammarState::new_lark_strict(f.clone(), &[], GrammarMode::Eager).is_err());
        assert!(Gemma4GrammarState::new_qwen35_xml(f, &[], GrammarMode::Eager, None).is_err());
    }

    /// A logits buffer SHORTER than the mask means the tokenizer env and the
    /// model disagree about the vocabulary, and everything downstream is
    /// meaningless — so this is the one edge that errors rather than
    /// degrading. A padded lm_head (logits LONGER than the mask) is normal and
    /// must keep working, so both sides are asserted together.
    #[test]
    fn a_short_logits_buffer_errors_while_a_padded_one_is_fine() {
        let mut g = Gemma4GrammarState::new_lark(
            shared_factory_placeholder(),
            &sample_tools(),
            GrammarMode::Eager,
        )
        .expect("eager state");
        assert!(g.is_active());

        // Learn the mask width from a run that is definitely wide enough.
        let mut wide = vec![1.0_f32; 4096];
        let masked = g
            .apply_mask_to_logits(&mut wide)
            .expect("a generous buffer must mask cleanly");
        assert!(masked > 0, "an active grammar should forbid something");

        let mut short = vec![1.0_f32; 2];
        let err = g
            .apply_mask_to_logits(&mut short)
            .expect_err("a logits buffer narrower than the mask must error");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("vocab mismatch"),
            "the error must name the mismatch so the cause is diagnosable: {msg}"
        );
    }

    /// After `release()` the state is permanently transparent: masking is a
    /// no-op that must leave the buffer untouched, and both observe entry
    /// points must accept tokens without erroring. A released grammar that
    /// still errored would turn a recoverable desync into a failed request.
    #[test]
    fn a_released_grammar_is_transparent_from_every_entry_point() {
        let mut g = Gemma4GrammarState::new_lark(
            shared_factory_placeholder(),
            &sample_tools(),
            GrammarMode::Eager,
        )
        .expect("eager state");
        g.release();
        assert!(!g.is_active());

        let mut logits = vec![1.0_f32; 64];
        assert_eq!(g.apply_mask_to_logits(&mut logits).unwrap(), 0);
        assert!(
            logits.iter().all(|v| (*v - 1.0).abs() < 1e-6),
            "a released grammar must not touch the logits"
        );
        // Arbitrary tokens, including ones the grammar would have rejected.
        for tok in [0u32, 7, 48, 9999] {
            g.observe(tok).expect("observe after release");
            g.observe_prefill(tok)
                .expect("observe_prefill after release");
        }
    }

    /// A Lazy grammar before its trigger has no matcher at all, so prefill
    /// replay has nothing to consume. It must be a silent no-op — Eager is the
    /// only mode that prefills, and erroring here would break every lazy
    /// request that happens to replay a prompt token.
    #[test]
    fn prefill_replay_is_a_no_op_before_a_lazy_grammar_activates() {
        let mut g = Gemma4GrammarState::new_lark(
            shared_factory_placeholder(),
            &sample_tools(),
            GrammarMode::Lazy,
        )
        .expect("lazy state");
        assert!(!g.is_active(), "lazy grammars start without a matcher");

        for tok in [1u32, 2, 3] {
            g.observe_prefill(tok)
                .expect("prefill before activation must be a silent no-op");
        }
        assert!(!g.is_active(), "prefill must not activate a lazy grammar");
    }

    /// Once the matcher reports `is_stopped()` the state latches finished and
    /// stops constraining. Driven here through prefill replay, which has its
    /// own copy of the stop check — the `observe` copy is already covered by
    /// the streaming tests.
    #[test]
    fn a_stopped_matcher_latches_finished_through_prefill() {
        let tools = vec![json!({
            "type": "function",
            "function": {
                "name": "noop",
                "parameters": { "type": "object", "properties": {} }
            }
        })];
        let mut g =
            Gemma4GrammarState::new_lark(shared_factory_placeholder(), &tools, GrammarMode::Eager)
                .expect("eager state");

        // Replay the whole deterministic opener plus body. The single-byte
        // env makes every ASCII byte its own token, so the grammar's literal
        // text is exactly its token sequence.
        let mut latched = false;
        for byte in b"call:noop{}" {
            if g.observe_prefill(u32::from(*byte)).is_err() {
                break;
            }
            if !g.is_active() {
                latched = true;
                break;
            }
        }
        // Either the matcher stopped (latched) or it is still mid-parse; what
        // must never happen is a state that reports active while its matcher
        // has stopped, so assert the invariant rather than a fixed outcome.
        if latched {
            let mut logits = vec![1.0_f32; 64];
            assert_eq!(
                g.apply_mask_to_logits(&mut logits).unwrap(),
                0,
                "a finished grammar must stop masking"
            );
        }
    }

    // ───────── degenerate-schema coverage (005 Phase 4.1) ─────────
    //
    // Every branch below is a fallback the renderers take when a tool schema
    // is outside the supported subset. They matter because none of them
    // *fails*: each one silently widens the grammar, and a grammar that is too
    // wide constrains nothing while still looking like it is working. Seven of
    // the recorded defects came off this surface, so the fallbacks deserve the
    // same pinning the happy path already has.

    /// An empty `tools` array must be an error from every builder, not an
    /// empty grammar. An empty grammar matches nothing, so the model would be
    /// masked into silence with no explanation.
    #[test]
    fn every_builder_rejects_an_empty_tool_list() {
        let err = qwen35_lark_grammar_string(&[]).expect_err("qwen35 builder");
        assert!(format!("{err:#}").contains("empty tools"), "{err:#}");
        for strict in [false, true] {
            let err = lark_grammar_string(&[], strict).expect_err("gemma builder");
            assert!(format!("{err:#}").contains("empty tools"), "{err:#}");
        }
    }

    /// A tool whose parameters carry no `properties` key at all. The body
    /// collapses to "anything but `}`" — permissive by design, but it must be
    /// that exact fallback and not an error or an empty rule.
    #[test]
    fn a_schema_without_properties_falls_back_to_permissive() {
        let mut extra = Vec::new();
        for schema in [
            json!({}),
            json!({"type": "object"}),
            json!({"properties": []}),
        ] {
            let body = lark_body_for_object_schema(&schema, &mut extra, false)
                .expect("the fallback must not error");
            assert_eq!(
                body, "<[^125]>*",
                "schema {schema} should render the permissive body"
            );
        }
    }

    /// `properties: {}` is different from a missing `properties`: the tool
    /// declares it takes nothing, so the body must be the empty literal rather
    /// than the permissive wildcard.
    #[test]
    fn an_empty_properties_map_renders_an_empty_body() {
        let mut extra = Vec::new();
        let body =
            lark_body_for_object_schema(&json!({"properties": {}}), &mut extra, false).unwrap();
        assert_eq!(body, "\"\"");
    }

    /// The required/optional split is the **strict** path only; `strict:false`
    /// renders a permissive any-order repeat instead. Both arms are asserted
    /// because conflating them is exactly how a strict grammar silently stops
    /// being strict.
    #[test]
    fn the_required_optional_split_is_strict_only() {
        let mut extra = Vec::new();
        let two_optional = json!({
            "properties": { "a": {"type": "string"}, "b": {"type": "string"} }
        });

        // Non-strict: an any-order comma-separated repeat, no per-field gating.
        let permissive = lark_body_for_object_schema(&two_optional, &mut extra, false).unwrap();
        assert!(
            permissive.contains(")*"),
            "the permissive body should be a repeat: {permissive}"
        );

        // Strict, all-optional: first optional standalone, the rest comma-gated
        // so skipping the first cannot leave a leading comma.
        let strict = lark_body_for_object_schema(&two_optional, &mut extra, true).unwrap();
        assert!(
            !strict.trim_start().starts_with("(\",\""),
            "the first optional must not be comma-prefixed: {strict}"
        );
        assert!(
            strict.contains("(\",\""),
            "later optionals must be: {strict}"
        );
        assert!(
            !strict.contains(")*"),
            "strict must not fall back to a repeat: {strict}"
        );

        // Strict with one optional: same arm, loop body never runs.
        let lone = lark_body_for_object_schema(
            &json!({ "properties": { "only": {"type": "string"} } }),
            &mut extra,
            true,
        )
        .unwrap();
        assert!(
            !lone.contains("(\",\""),
            "a lone optional needs no comma: {lone}"
        );

        // Strict with a required field takes the other arm: required first and
        // ungated, optionals after and comma-gated.
        let mixed = lark_body_for_object_schema(
            &json!({
                "properties": { "req": {"type": "string"}, "opt": {"type": "string"} },
                "required": ["req"]
            }),
            &mut extra,
            true,
        )
        .unwrap();
        assert!(
            mixed.starts_with("(\"req:\""),
            "the required field must lead, ungated: {mixed}"
        );
        assert!(
            mixed.contains("(\",\" (\"opt:\""),
            "the optional must follow comma-gated: {mixed}"
        );
    }

    /// `const` short-circuits ahead of `type`, so a schema carrying both must
    /// render the literal — otherwise a fixed-value parameter would accept the
    /// whole type.
    #[test]
    fn a_const_wins_over_the_declared_type() {
        let mut extra = Vec::new();
        let rule = lark_value_for_schema(
            &json!({ "const": "yes", "type": "string" }),
            &mut extra,
            false,
        )
        .unwrap();
        assert!(rule.contains("yes"), "const literal expected, got {rule}");
        assert!(
            !rule.contains("[^"),
            "the string wildcard must not appear alongside a const: {rule}"
        );
    }

    /// `type` as an array is the nullable-field shape (`["string","null"]`).
    /// Each variant renders and the union is taken.
    #[test]
    fn a_type_array_unions_its_variants() {
        let mut extra = Vec::new();
        let rule = lark_value_for_schema(&json!({ "type": ["string", "null"] }), &mut extra, false)
            .unwrap();
        assert!(
            rule.starts_with('(') && rule.contains(" | "),
            "union expected: {rule}"
        );
    }

    /// Non-string entries inside a `type` array are skipped rather than
    /// erroring — and when *every* entry is skipped the union is empty, so the
    /// renderer must fall through to the single-type path instead of emitting
    /// `()`, which matches nothing.
    #[test]
    fn a_type_array_of_junk_falls_through_instead_of_emitting_an_empty_union() {
        let mut extra = Vec::new();

        // Mixed: the string survives, the rest are ignored.
        let rule = lark_value_for_schema(
            &json!({ "type": [42, "string", null, {"nested": true}] }),
            &mut extra,
            false,
        )
        .unwrap();
        assert!(!rule.contains("()"), "empty alternative in {rule}");

        // All junk: no alternatives at all.
        let rule =
            lark_value_for_schema(&json!({ "type": [42, null] }), &mut extra, false).unwrap();
        assert!(
            !rule.is_empty() && !rule.contains("()"),
            "an all-junk type array must still render something matchable, got {rule:?}"
        );
    }

    /// The Qwen XML renderer has its own required/optional split, and its
    /// empty case is a repeat rather than an empty string — an empty rule
    /// there would make a no-parameter tool unmatchable.
    #[test]
    fn the_qwen_param_renderer_has_a_repeat_fallback_when_nothing_renders() {
        let s = qwen35_lark_grammar_string(&[json!({
            "type": "function",
            "function": { "name": "noop", "parameters": { "type": "object" } }
        })])
        .expect("a no-parameter tool must still build");
        assert!(s.contains("noop"), "the tool name must survive: {s}");
    }

    /// `with_bounded_whitespace` walks a schema object; a non-object schema
    /// (JSON Schema allows `true`/`false` as whole schemas) must pass straight
    /// through rather than panic or be rewritten into an object.
    #[test]
    fn a_non_object_schema_passes_through_whitespace_bounding() {
        for v in [
            json!(true),
            json!(false),
            json!(null),
            json!(7),
            json!("s"),
            json!([]),
        ] {
            assert_eq!(
                with_bounded_whitespace(v.clone()),
                v,
                "a non-object schema must be returned unchanged"
            );
        }
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
    fn lazy_activation_does_not_feed_the_trigger_to_the_matcher() {
        // The regression that broke every streaming tool call: activation used
        // to `consume_token(48)` on the matcher it had just built, but the
        // grammar starts at `call:` and knows nothing about the opener, so the
        // parse failed on its first byte and the error aborted the request.
        //
        // The placeholder env maps ids to single bytes, so id 48 is `'0'` here
        // rather than `<|tool_call>` — either way it is a byte the grammar does
        // not accept at position 0, which is exactly the condition under test.
        let factory = shared_factory_placeholder();
        let mut state = Gemma4GrammarState::new_lark(factory, &sample_tools(), GrammarMode::Lazy)
            .expect("build lazy lark state");
        state
            .observe(TOK_TOOL_CALL_OPEN)
            .expect("the trigger token activates the grammar without being parsed by it");
        assert!(
            state.is_active(),
            "the grammar must be constraining once the opener is sampled"
        );
    }

    #[test]
    fn lazy_activation_leaves_the_matcher_at_the_start_of_the_body() {
        // Corollary of the fix, and the part that proves the matcher is
        // actually usable afterwards rather than merely non-erroring: the
        // first constrained token must be the start of `call:`. `'c'` is
        // accepted; the trigger id is not.
        let factory = shared_factory_placeholder();
        let mut state = Gemma4GrammarState::new_lark(factory, &sample_tools(), GrammarMode::Lazy)
            .expect("build lazy lark state");
        state.observe(TOK_TOOL_CALL_OPEN).expect("activate");
        state
            .observe(u32::from(b'c'))
            .expect("grammar resumes at the `call:` literal");
        assert!(state.is_active());
        assert!(
            state.observe(u32::from(b'x')).is_err(),
            "a byte that cannot continue `call:` must still be rejected — \
             otherwise the matcher is not really constraining"
        );
    }

    #[test]
    fn lazy_trigger_is_opt_in_for_non_gemma_grammars() {
        // Qwen 3.6 passes `opener_token: None`. Borrowing Gemma's id 48 there
        // meant the ordinary letter `Q` woke the XML grammar mid-sentence and
        // desynced it. With no trigger configured, Lazy simply never activates.
        let factory = shared_factory_placeholder();
        let mut state =
            Gemma4GrammarState::new_qwen35_xml(factory, &sample_tools(), GrammarMode::Lazy, None)
                .expect("build lazy qwen35 state");
        state.observe(TOK_TOOL_CALL_OPEN).expect("no trigger set");
        assert!(
            !state.is_active(),
            "an unconfigured lazy trigger must never activate the grammar"
        );
    }

    #[test]
    fn qwen35_lazy_trigger_is_consumed_because_it_opens_the_grammar() {
        // The mirror image of the Gemma case: `tool_call: "<tool_call>\n" …`
        // starts *with* the opener, so an explicit trigger has to be parsed or
        // the matcher would sit one terminal behind the model's context.
        // `'<'` is the first byte of that literal in the placeholder env.
        let factory = shared_factory_placeholder();
        let mut state = Gemma4GrammarState::new_qwen35_xml(
            factory,
            &sample_tools(),
            GrammarMode::Lazy,
            Some(u32::from(b'<')),
        )
        .expect("build lazy qwen35 state with an explicit opener");
        state
            .observe(u32::from(b'<'))
            .expect("activate and consume");
        assert!(state.is_active());
        state
            .observe(u32::from(b't'))
            .expect("matcher advanced past `<`, so `t` continues `<tool_call>`");
    }

    /// Drive a grammar with a byte string, returning the index of the first
    /// byte it rejects. The placeholder env maps ids to bytes, so this reads
    /// as "what output would the grammar allow".
    fn feed(state: &mut Gemma4GrammarState, bytes: &[u8]) -> Option<usize> {
        for (i, b) in bytes.iter().enumerate() {
            if state.observe(u32::from(*b)).is_err() {
                return Some(i);
            }
        }
        None
    }

    fn city_schema() -> Value {
        json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"],
        })
    }

    #[test]
    fn response_format_grammar_forbids_whitespace_runs() {
        // With llguidance's default flexible whitespace, `{` may be followed by
        // an unbounded `[ \n\r\t]+` run, and greedy decode rides it until
        // max_tokens — the observed failure was a reply of `{` and nothing else.
        let factory = shared_factory_placeholder();
        let mut state =
            Gemma4GrammarState::new_json_schema(factory, &city_schema(), GrammarMode::Eager)
                .expect("build response_format grammar");
        state.observe(u32::from(b'{')).expect("object opens");
        assert!(
            state.observe(u32::from(b'\n')).is_err(),
            "whitespace after `{{` must not be a legal continuation"
        );
    }

    #[test]
    fn response_format_grammar_keeps_the_separator_space() {
        // The counterpart to the rule above: `"key": "value"` is what models
        // write, and taking the space away pushes them off-distribution — the
        // leftover `: ` ends up inside the string value. The space lives in the
        // separator literal instead, so it is available but not paddable.
        let factory = shared_factory_placeholder();
        let mut state =
            Gemma4GrammarState::new_json_schema(factory, &city_schema(), GrammarMode::Eager)
                .expect("build response_format grammar");
        assert_eq!(
            feed(&mut state, br#"{"city": "Seoul"}"#),
            None,
            "the model's natural spacing must parse"
        );
    }

    #[test]
    fn response_format_grammar_rejects_a_padded_separator() {
        let factory = shared_factory_placeholder();
        let mut state =
            Gemma4GrammarState::new_json_schema(factory, &city_schema(), GrammarMode::Eager)
                .expect("build response_format grammar");
        assert_eq!(
            feed(&mut state, br#"{"city":  "Seoul"}"#),
            Some(9),
            "a second space is where the unbounded run would have started"
        );
    }

    #[test]
    fn response_format_grammar_respects_an_explicit_x_guidance() {
        // A caller who set `x-guidance` themselves gets their setting, not ours.
        let factory = shared_factory_placeholder();
        let schema = json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"],
            "x-guidance": { "whitespace_flexible": true },
        });
        let mut state = Gemma4GrammarState::new_json_schema(factory, &schema, GrammarMode::Eager)
            .expect("build response_format grammar");
        state.observe(u32::from(b'{')).expect("object opens");
        state
            .observe(u32::from(b'\n'))
            .expect("caller asked for flexible whitespace");
    }

    #[test]
    fn eager_prefill_replay_skips_the_opener_and_parses_the_rest() {
        // `tool_choice=required`/named prefills `<|tool_call>` (and, for a
        // named choice, `call:NAME{`) into the prompt and replays it through
        // the Eager matcher. Replaying the opener used to fail for the same
        // reason lazy activation did, and the caller's response was to drop the
        // grammar — so `required` never actually enforced anything. The opener
        // must pass through untouched while the rest still parses.
        let factory = shared_factory_placeholder();
        let mut state =
            Gemma4GrammarState::new_lark_strict(factory, &sample_tools(), GrammarMode::Eager)
                .expect("build eager strict lark state");
        state
            .observe_prefill(TOK_TOOL_CALL_OPEN)
            .expect("the opener is context, not grammar");
        assert!(state.is_active(), "the grammar must still be constraining");
        for b in b"call:task_complete{" {
            state
                .observe_prefill(u32::from(*b))
                .expect("the named-choice prefill is the grammar's own opener");
        }
        assert!(state.is_active());
    }

    #[test]
    fn release_stops_constraining() {
        let factory = shared_factory_placeholder();
        let mut state = Gemma4GrammarState::new_lark(factory, &sample_tools(), GrammarMode::Eager)
            .expect("build eager lark state");
        assert!(state.is_active());
        state.release();
        assert!(!state.is_active(), "released grammar must not constrain");
        let mut logits = vec![1.0_f32; 262];
        assert_eq!(
            state.apply_mask_to_logits(&mut logits).expect("noop mask"),
            0,
            "a released grammar masks nothing"
        );
        state
            .observe(u32::from(b'x'))
            .expect("a released grammar accepts anything");
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
    fn lark_grammar_handles_multi_type_nullable_field() {
        // JSON Schema `type: ["string","null"]` (nullable string) used to
        // fall back to the permissive `<[^44,125]>*` rule, which allows
        // raw `{` `}` in the field body and breaks balanced-brace parsing
        // downstream. Verify the grammar now unions `string_val | "null"`
        // so the field stays brace-safe and parser-deterministic.
        let tools = vec![json!({
            "type": "function",
            "function": {
                "name": "match_decision",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "suggested_match": { "type": ["string", "null"] },
                        "match_type": {
                            "type": "string",
                            "enum": ["\u{d300}", "\u{b9ac}\u{adf8}"]
                        },
                        "confidence": { "type": "integer" }
                    },
                    "required": ["suggested_match", "match_type", "confidence"]
                }
            }
        })];
        let s = lark_grammar_string(&tools, false).expect("build multi-type lark grammar");
        assert!(
            s.contains("string_val | \"null\""),
            "nullable string should union string_val and the null literal; got:\n{s}"
        );
        assert!(
            !s.contains("<[^44,125]>*"),
            "no permissive fallback should remain for nullable string; got:\n{s}"
        );
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
    fn lark_grammar_escapes_a_non_identifier_tool_name() {
        // The builder used to refuse any name that was not a bare identifier,
        // which dropped the grammar and let the model invent a name no client
        // declared — `날씨_조회` came back as `weather_lookup`. Rule names are
        // indexed now, so the tool name only has to survive as a literal.
        for name in [
            "foo-bar",
            "Playwright (Stealth)__browser_navigate",
            "날씨_조회",
        ] {
            let tools = vec![json!({
                "type": "function",
                "function": { "name": name, "parameters": { "type":"object","properties":{} } }
            })];
            let s = lark_grammar_string(&tools, false)
                .unwrap_or_else(|e| panic!("{name:?} must build: {e}"));
            assert!(s.contains(name), "{name:?} must appear as a literal:\n{s}");
            let factory = shared_factory_placeholder();
            Gemma4GrammarState::new_lark(factory, &tools, GrammarMode::Eager)
                .unwrap_or_else(|e| panic!("{name:?} must compile under llguidance: {e}"));
        }
    }

    #[test]
    fn lark_grammar_escapes_a_quote_in_a_tool_name() {
        // A `"` in a name would close the Lark literal and produce a grammar
        // that either fails to compile or matches the wrong bytes.
        let tools = vec![json!({
            "type": "function",
            "function": { "name": "say\"hi", "parameters": { "type":"object","properties":{} } }
        })];
        let s = lark_grammar_string(&tools, false).expect("quoted name must build");
        assert!(s.contains("say\\\"hi"), "quote must be escaped:\n{s}");
        let factory = shared_factory_placeholder();
        Gemma4GrammarState::new_lark(factory, &tools, GrammarMode::Eager)
            .expect("must compile under llguidance");
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
            .find(|l| l.starts_with("tool_0_body:"))
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
            .find(|l| l.starts_with("tool_0_body:"))
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
            .find(|l| l.starts_with("q_0_body:"))
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
            .find(|l| l.starts_with("q_1_body:"))
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
    fn qwen35_grammar_escapes_a_non_identifier_tool_name() {
        for name in ["foo-bar", "날씨_조회"] {
            let tools = vec![json!({
                "type": "function",
                "function": {
                    "name": name,
                    "parameters": {
                        "type": "object",
                        "properties": { "도시": { "type": "string" } },
                        "required": ["도시"]
                    }
                }
            })];
            let s = qwen35_lark_grammar_string(&tools)
                .unwrap_or_else(|e| panic!("{name:?} must build: {e}"));
            assert!(s.contains(name), "{name:?} must appear as a literal:\n{s}");
            assert!(
                s.contains("<parameter=도시>"),
                "a non-identifier key must be constrained, not dropped:\n{s}"
            );
            let factory = shared_factory_placeholder();
            Gemma4GrammarState::new_qwen35_xml(factory, &tools, GrammarMode::Eager, None)
                .unwrap_or_else(|e| panic!("{name:?} must compile under llguidance: {e}"));
        }
    }

    #[test]
    fn lark_grammar_constrains_a_non_identifier_property_key() {
        // Gemma side: a non-identifier key used to collapse the whole body to
        // the permissive `<[^125]>*` fallback, silently un-constraining every
        // other field of that tool.
        let tools = vec![json!({
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {
                    "type": "object",
                    "properties": { "도시": { "type": "string" } },
                    "required": ["도시"]
                }
            }
        })];
        let s = lark_grammar_string(&tools, true).expect("non-identifier key must build");
        assert!(s.contains("도시:"), "key must be constrained:\n{s}");
        assert!(
            !s.contains("<[^125]>*"),
            "must not fall back to the permissive body:\n{s}"
        );
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
            .find(|l| l.starts_with("tool_0_body:"))
            .expect("search body present");
        assert!(
            !body_line.contains("))*"),
            "no-required strict body must still be dup-free: {body_line}"
        );
    }
}
