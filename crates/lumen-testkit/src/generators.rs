//! Structure-aware input generators for the tool-calling surface.
//!
//! This is the dbsqlfuzz half of the plan in
//! `.ai/memory/active/005-sqlite-testing-strategy`: SQLite's fuzzer does not
//! emit random bytes, it emits *plausible SQL with hostile structure*, and it
//! mutates the database file alongside the query so the two can disagree. The
//! analogue here is a tool schema set and a model output stream generated
//! together, so a fuzz target can assert the cross-invariant that matters —
//! parsing a stream produced against a grammar never yields a tool nobody
//! declared.
//!
//! ## Why the generators live here and emit plain types
//!
//! `lumen-testkit` is a dev-dependency of the crates under test, so it cannot
//! depend on them back. Everything below therefore emits `String` and
//! `serde_json::Value`; each fuzz target converts into whatever typed form its
//! entry point wants. That constraint turns out to be the right shape anyway —
//! the same `ToolSet` feeds the Gemma 4 pseudo-JSON parser, the Lark grammar
//! builder and the OpenAI request deserializer without any of them leaking into
//! this crate.
//!
//! ## What "hostile" means concretely
//!
//! Not random noise. Every generator draws from a small alphabet of inputs that
//! have *actually broken this codebase*, mixed with well-formed material so the
//! parser gets far enough in to reach the interesting code. Of the defects
//! recorded in `xtask/src/red_green.rs`, seven live in this surface, and the
//! choices below are taken from them:
//!
//!   * `tool-name-scanner` — `call:bad call:good{x:1}` parsed as one tool named
//!     `"bad call:good"`. Hence names containing spaces and embedded `call:`.
//!   * `args-unicode-keys` — `{도시:…}` arriving as mojibake. Hence non-ASCII
//!     keys and names, in more than one script.
//!   * `grammar-literal-escaping` — a literal escaping its own quotes. Hence
//!     names carrying `"`, `\` and newlines.
//!   * `json-whitespace` / `json-separator-space` — significant whitespace
//!     around separators. Hence the whitespace mutations.
//!   * `lark-opener` — a truncated or doubled opener token.
//!
//! A generator that only produced well-formed input would exercise none of
//! them; one that only produced garbage would be rejected at the first byte and
//! exercise none of them either.

use arbitrary::{Arbitrary, Result, Unstructured};
use serde_json::{Value, json};

/// Tool names that are legal for a client to send and awkward for us to
/// handle. The empty string and the 1 KB name are boundary cases; the rest are
/// shapes that have produced real defects.
const HOSTILE_NAMES: &[&str] = &[
    "get_weather",
    "날씨_조회",
    "obtenir_météo",
    "получить_погоду",
    "get weather",
    "call:nested",
    "bad call:good",
    "with\"quote",
    "with\\backslash",
    "with\nnewline",
    "with{brace}",
    "with,comma",
    "",
    "a",
    "_",
    "0leading_digit",
    "emoji_🛠_tool",
    "trailing_space ",
    " leading_space",
    "very.dotted.name",
];

/// Argument keys, same idea. `<|\"|>` is Gemma 4's own string delimiter, so a
/// key containing it probes the parser's delimiter scanning.
const HOSTILE_KEYS: &[&str] = &[
    "city",
    "도시",
    "ville",
    "key with space",
    "key\"quote",
    "key\\slash",
    "",
    "nested",
    "unit",
    "<|\"|>",
    "0",
];

/// JSON Schema fragments a real client might send, including the shapes that
/// have no `type` at all.
fn arbitrary_schema(u: &mut Unstructured<'_>, depth: u8) -> Result<Value> {
    // Bounded recursion: a schema deep enough to blow the stack is a valid
    // thing to send, but generating one every time would drown the shallow
    // cases that reach more code.
    if depth >= 3 {
        return Ok(json!({"type": "string"}));
    }
    Ok(match u.int_in_range(0u8..=9)? {
        0 => json!({"type": "string"}),
        1 => json!({"type": "integer", "minimum": -1, "maximum": 1_000_000}),
        2 => json!({"type": "boolean"}),
        3 => json!({"type": "string", "enum": ["celsius", "섭씨", "", "a\"b"]}),
        4 => json!({"const": 42}),
        // No `type` key at all — legal JSON Schema, and a shape the builder has
        // to survive rather than index blindly.
        5 => json!({"description": "untyped"}),
        6 => json!({
            "type": "array",
            "items": arbitrary_schema(u, depth + 1)?,
        }),
        7 => json!({
            "type": "object",
            "properties": {
                "inner": arbitrary_schema(u, depth + 1)?,
            },
            "additionalProperties": false,
        }),
        8 => json!({
            "type": "object",
            "properties": {},
            "additionalProperties": true,
        }),
        _ => json!({"type": ["string", "null"]}),
    })
}

/// A set of tool declarations in OpenAI function-calling shape.
#[derive(Debug, Clone)]
pub struct ToolSet {
    /// Ready to hand to a grammar builder or a request body.
    pub tools: Vec<Value>,
}

impl ToolSet {
    /// Declared names, in declaration order. The invariant a fuzz target checks
    /// is that nothing downstream ever produces a name outside this set.
    pub fn names(&self) -> Vec<String> {
        self.tools
            .iter()
            .filter_map(|t| {
                t.get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
            .collect()
    }
}

impl<'a> Arbitrary<'a> for ToolSet {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        // Zero tools is a real request shape (tool_choice="none" with an empty
        // array) and has its own code path, so the range starts at 0.
        let n = u.int_in_range(0usize..=8)?;
        let mut tools = Vec::with_capacity(n);
        for _ in 0..n {
            let name = if u.ratio(1, 8)? {
                // Occasionally a name long enough to matter for buffer and
                // grammar-size handling.
                "n".repeat(1024)
            } else {
                (*u.choose(HOSTILE_NAMES)?).to_string()
            };
            let n_props = u.int_in_range(0usize..=3)?;
            let mut props = serde_json::Map::new();
            let mut required = Vec::new();
            for _ in 0..n_props {
                let key = (*u.choose(HOSTILE_KEYS)?).to_string();
                props.insert(key.clone(), arbitrary_schema(u, 0)?);
                if u.ratio(1, 2)? {
                    required.push(Value::String(key));
                }
            }
            tools.push(json!({
                "type": "function",
                "function": {
                    "name": name,
                    "description": "generated",
                    "parameters": {
                        "type": "object",
                        "properties": Value::Object(props),
                        "required": Value::Array(required),
                    }
                }
            }));
        }
        Ok(ToolSet { tools })
    }
}

/// Structured corruptions of a Gemma 4 tool-call stream. Each one names a way
/// the stream can be malformed that a plain byte mutation would essentially
/// never produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Mutation {
    /// Well-formed. The control case — a generator that never emits valid input
    /// tests only the reject path.
    None,
    /// `call` without its `:`.
    TruncatedOpener,
    /// `call:call:name{...}`.
    DoubledOpener,
    /// One `{` too few.
    UnbalancedOpen,
    /// One `}` too many.
    UnbalancedClose,
    /// A whole `call:...{...}` inside an argument value.
    NestedCall,
    /// Value delimiters opened and never closed.
    UnterminatedString,
    /// Whitespace inserted around `:` and `,`.
    SeparatorSpace,
    /// Two calls with no separator between them.
    Adjacent,
    /// Empty body between the braces.
    EmptyArgs,
}

const ALL_MUTATIONS: &[Mutation] = &[
    Mutation::None,
    Mutation::TruncatedOpener,
    Mutation::DoubledOpener,
    Mutation::UnbalancedOpen,
    Mutation::UnbalancedClose,
    Mutation::NestedCall,
    Mutation::UnterminatedString,
    Mutation::SeparatorSpace,
    Mutation::Adjacent,
    Mutation::EmptyArgs,
];

/// A decoded `<|tool_call>…<tool_call|>` body, as the parser sees it.
#[derive(Debug, Clone)]
pub struct ModelOutput {
    pub text: String,
    pub mutation: Mutation,
    /// Names deliberately written into the stream. Under
    /// [`ModelOutput::for_tools`] these are drawn from the declared set, which
    /// is what makes the "never yields an undeclared tool" invariant a real
    /// test rather than a tautology.
    pub intended_names: Vec<String>,
}

impl ModelOutput {
    /// Build a stream that references `names`, so a target can assert that
    /// parsing never invents a name outside them. With an empty `names` the
    /// generator falls back to the hostile alphabet.
    pub fn for_tools(u: &mut Unstructured<'_>, names: &[String]) -> Result<Self> {
        let mutation = *u.choose(ALL_MUTATIONS)?;
        let n_calls = u.int_in_range(1usize..=3)?;
        let mut text = String::new();
        let mut intended = Vec::new();

        for c in 0..n_calls {
            let name = if names.is_empty() {
                (*u.choose(HOSTILE_NAMES)?).to_string()
            } else {
                names[u.int_in_range(0..=names.len() - 1)?].clone()
            };
            intended.push(name.clone());

            let opener = match mutation {
                Mutation::TruncatedOpener => "call",
                Mutation::DoubledOpener => "call:call:",
                _ => "call:",
            };
            text.push_str(opener);
            text.push_str(&name);
            text.push('{');

            if mutation != Mutation::EmptyArgs {
                let n_args = u.int_in_range(1usize..=3)?;
                for a in 0..n_args {
                    if a > 0 {
                        text.push_str(if mutation == Mutation::SeparatorSpace {
                            " , "
                        } else {
                            ","
                        });
                    }
                    let key = *u.choose(HOSTILE_KEYS)?;
                    text.push_str(key);
                    text.push_str(if mutation == Mutation::SeparatorSpace {
                        " : "
                    } else {
                        ":"
                    });
                    match mutation {
                        Mutation::NestedCall => {
                            text.push_str("<|\"|>call:inner{x:1}<|\"|>");
                        }
                        Mutation::UnterminatedString => {
                            text.push_str("<|\"|>never closed");
                        }
                        _ => {
                            // Mix quoted strings with bare scalars; the parser
                            // has to tell them apart.
                            match u.int_in_range(0u8..=4)? {
                                0 => text.push_str("<|\"|>서울<|\"|>"),
                                1 => text.push_str("<|\"|>a\"b<|\"|>"),
                                2 => text.push_str("42"),
                                3 => text.push_str("true"),
                                _ => text.push_str("<|\"|><|\"|>"),
                            }
                        }
                    }
                }
            }

            match mutation {
                Mutation::UnbalancedOpen => {} // brace deliberately omitted
                Mutation::UnbalancedClose => text.push_str("}}"),
                _ => text.push('}'),
            }
            if c + 1 < n_calls && mutation != Mutation::Adjacent {
                text.push(' ');
            }
        }

        Ok(ModelOutput {
            text,
            mutation,
            intended_names: intended,
        })
    }
}

impl<'a> Arbitrary<'a> for ModelOutput {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        Self::for_tools(u, &[])
    }
}

/// A `ToolSet` and a `ModelOutput` drawn from one `Unstructured`, with the
/// output referencing the set's declared names.
///
/// This pairing is the point of the whole module — mutating *both* sides at
/// once is what turns `tool-name-scanner` and `grammar-rule-names` from
/// "found in production" into "found in a fuzz run".
#[derive(Debug, Clone)]
pub struct GrammarAndOutput {
    pub tools: ToolSet,
    pub output: ModelOutput,
}

impl<'a> Arbitrary<'a> for GrammarAndOutput {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let tools = ToolSet::arbitrary(u)?;
        let output = ModelOutput::for_tools(u, &tools.names())?;
        Ok(GrammarAndOutput { tools, output })
    }
}

/// An OpenAI or Anthropic chat request body, as JSON.
///
/// Emitted untyped so the fuzz target owns the `serde` step — deserialization
/// is itself part of what is being tested, and a typed generator would skip it.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub json: Value,
    /// `true` for the Anthropic `/v1/messages` shape, `false` for OpenAI.
    pub anthropic: bool,
}

impl<'a> Arbitrary<'a> for ChatRequest {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let tools = ToolSet::arbitrary(u)?;
        let anthropic = u.ratio(1, 3)?;

        // Every `tool_choice` variant, including the object form that names a
        // specific function — the shape behind the `tool-choice-none` defect.
        let tool_choice = match u.int_in_range(0u8..=4)? {
            0 => json!("auto"),
            1 => json!("none"),
            2 => json!("required"),
            3 => {
                let names = tools.names();
                let name = names.first().cloned().unwrap_or_default();
                json!({"type": "function", "function": {"name": name}})
            }
            _ => Value::Null,
        };

        let n_msgs = u.int_in_range(0usize..=4)?;
        let mut messages = Vec::with_capacity(n_msgs);
        for _ in 0..n_msgs {
            let role = *u.choose(&["system", "user", "assistant", "tool"])?;
            let content = match u.int_in_range(0u8..=4)? {
                0 => json!("plain text"),
                1 => json!(""),
                2 => json!([{"type": "text", "text": "블록 콘텐츠"}]),
                3 => json!([{
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,aGVsbG8="}
                }]),
                _ => json!([{"type": "tool_result", "tool_use_id": "id", "content": "ok"}]),
            };
            messages.push(json!({"role": role, "content": content}));
        }

        // 0 and a value past any real cap are both boundary cases the server
        // has to clamp rather than trust.
        let max_tokens = match u.int_in_range(0u8..=3)? {
            0 => json!(0),
            1 => json!(1),
            2 => json!(u32::MAX),
            _ => Value::Null,
        };

        let mut body = serde_json::Map::new();
        body.insert("model".into(), json!("generated"));
        body.insert("messages".into(), Value::Array(messages));
        if !tools.tools.is_empty() || u.ratio(1, 2)? {
            body.insert("tools".into(), Value::Array(tools.tools.clone()));
        }
        if !tool_choice.is_null() {
            body.insert("tool_choice".into(), tool_choice);
        }
        if !max_tokens.is_null() {
            body.insert("max_tokens".into(), max_tokens);
        }
        if u.ratio(1, 4)? {
            body.insert("stream".into(), json!(true));
        }
        if u.ratio(1, 4)? {
            body.insert("temperature".into(), json!(0.0));
        }

        Ok(ChatRequest {
            json: Value::Object(body),
            anthropic,
        })
    }
}

/// Deterministic driver — SQLite's fuzzcheck half.
///
/// Walks a seeded byte stream so the same seed always yields the same inputs,
/// which is what lets a plain `#[test]` exercise the generators in tier 0
/// without nightly or `cargo-fuzz`. The libFuzzer targets consume the identical
/// `Arbitrary` impls; only the byte source differs.
pub fn seeded_inputs<T, F>(seed: u64, iters: usize, mut f: F)
where
    T: for<'a> Arbitrary<'a>,
    F: FnMut(T),
{
    // xorshift64* — deterministic, no dependency, and the entropy quality does
    // not matter here because the structure comes from the generators.
    let mut state = seed | 1;
    let mut bytes = vec![0u8; 512];
    for _ in 0..iters {
        for chunk in bytes.chunks_mut(8) {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let v = state.wrapping_mul(0x2545_F491_4F6C_DD1D).to_le_bytes();
            chunk.copy_from_slice(&v[..chunk.len()]);
        }
        let mut u = Unstructured::new(&bytes);
        if let Ok(value) = T::arbitrary(&mut u) {
            f(value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The generators are only useful if a fixed seed reproduces a crash, so
    /// determinism is the first thing to hold them to.
    #[test]
    fn seeded_inputs_are_reproducible() {
        let mut first = Vec::new();
        seeded_inputs::<GrammarAndOutput, _>(42, 32, |g| first.push(g.output.text));
        let mut second = Vec::new();
        seeded_inputs::<GrammarAndOutput, _>(42, 32, |g| second.push(g.output.text));
        assert_eq!(first, second, "same seed must produce the same inputs");
        assert!(!first.is_empty(), "generator produced nothing at seed 42");
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = Vec::new();
        seeded_inputs::<GrammarAndOutput, _>(1, 32, |g| a.push(g.output.text));
        let mut b = Vec::new();
        seeded_inputs::<GrammarAndOutput, _>(2, 32, |g| b.push(g.output.text));
        assert_ne!(a, b, "distinct seeds must explore distinct inputs");
    }

    /// A generator that only emits well-formed input tests only the happy path,
    /// and one that never does gets rejected before reaching the interesting
    /// code. Both halves have to show up.
    #[test]
    fn generator_covers_valid_and_malformed() {
        let mut seen = std::collections::HashSet::new();
        seeded_inputs::<GrammarAndOutput, _>(7, 400, |g| {
            seen.insert(g.output.mutation);
        });
        assert!(
            seen.contains(&Mutation::None),
            "never generated a well-formed stream: {seen:?}"
        );
        assert!(
            seen.len() >= 6,
            "expected most mutation kinds within 400 draws, saw {}: {seen:?}",
            seen.len()
        );
    }

    /// The cross-invariant target needs the stream to reference declared names;
    /// if the pairing silently produced unrelated names the invariant would
    /// pass vacuously.
    #[test]
    fn paired_output_references_declared_tools() {
        let mut paired = 0usize;
        let mut checked = 0usize;
        seeded_inputs::<GrammarAndOutput, _>(11, 200, |g| {
            let declared = g.tools.names();
            if declared.is_empty() {
                return;
            }
            checked += 1;
            if g.output.intended_names.iter().all(|n| declared.contains(n)) {
                paired += 1;
            }
        });
        assert!(checked > 0, "no draw produced a non-empty tool set");
        assert_eq!(
            paired, checked,
            "every intended name must come from the declared set"
        );
    }

    #[test]
    fn tool_sets_reach_the_hostile_alphabet() {
        let mut names = std::collections::HashSet::new();
        seeded_inputs::<ToolSet, _>(3, 400, |t| names.extend(t.names()));
        assert!(
            names.iter().any(|n| !n.is_ascii()),
            "expected a non-ASCII tool name among {} draws",
            names.len()
        );
        assert!(
            names.iter().any(|n| n.contains('"') || n.contains('\\')),
            "expected a quote/backslash tool name"
        );
        assert!(names.contains(""), "expected the empty tool name");
    }

    #[test]
    fn chat_requests_cover_every_tool_choice_variant() {
        let mut variants = std::collections::HashSet::new();
        seeded_inputs::<ChatRequest, _>(5, 400, |r| {
            let v = r
                .json
                .get("tool_choice")
                .map(|c| {
                    c.as_str()
                        .map(str::to_owned)
                        .unwrap_or_else(|| "object".into())
                })
                .unwrap_or_else(|| "absent".into());
            variants.insert(v);
        });
        for want in ["auto", "none", "required", "object", "absent"] {
            assert!(
                variants.contains(want),
                "tool_choice variant {want:?} never generated; saw {variants:?}"
            );
        }
    }
}
