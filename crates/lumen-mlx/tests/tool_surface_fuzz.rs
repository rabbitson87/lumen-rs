//! Deterministic fuzz driver over the tool-calling text surface.
//!
//! This is the *fuzzcheck* half of the two-driver split in
//! `.ai/memory/active/005-sqlite-testing-strategy`: the same
//! `lumen_testkit::generators` impls that the libFuzzer targets consume, walked
//! here from a fixed seed inside an ordinary `#[test]`. It needs no nightly, no
//! `cargo-fuzz` and no GPU, so it runs on every `cargo test` rather than only
//! when someone remembers to soak — which, with no CI running tests, is the
//! difference between a gate and a good intention.
//!
//! Seeds are fixed, so a failure here is reproducible from the test name alone.
//!
//! ## What is asserted, and what deliberately is not
//!
//! **Never panics** applies to every input. These entry points take strings off
//! the wire (a model's output, a client's schema) and a panic in either is a
//! remote crash, so it is the one invariant that holds regardless of how
//! malformed the input is.
//!
//! **Names must be declared** applies only to well-formed streams. The parser
//! is handed text, not a tool set — it cannot know what was declared, so on a
//! stream with a deliberately mangled opener a mangled name is the correct
//! answer, not a bug. Asserting otherwise would be asserting something false
//! and the test would be deleted the first time it fired. On a *well-formed*
//! stream built against a known tool set, though, every parsed name has to come
//! back from that set — which is exactly what `tool-name-scanner` violated
//! (`call:bad call:good{x:1}` parsed as one tool named `"bad call:good"`).
//!
//! **JSON out must be JSON** — `gemma4_args_to_json` returning `Ok` with a
//! value that cannot round-trip through `serde_json` would be a silent
//! corruption rather than a rejection.

use lumen_mlx::gemma4_tool_syntax::{gemma4_args_to_json, parse_tool_call_body};
use lumen_mlx::grammar::{ToolCalls, build_qwen35_tool_grammar_lark};
use lumen_testkit::generators::{ChatRequest, GrammarAndOutput, Mutation, ToolSet, seeded_inputs};

/// Every entry point below is reached with both well-formed and malformed
/// input; anything that panics fails the test by unwinding out of it.
#[test]
fn parser_survives_generated_tool_call_streams() {
    let mut parsed_ok = 0usize;
    let mut total = 0usize;
    let mut asserted = 0usize;
    let mut boundary_checked = 0usize;

    seeded_inputs::<GrammarAndOutput, _>(0xF022, 600, |g| {
        total += 1;
        let declared = g.tools.names();
        let Ok(calls) = parse_tool_call_body(&g.output.text) else {
            // Rejecting a malformed stream is a correct outcome; only a panic
            // or a bad `Ok` is a defect.
            return;
        };
        parsed_ok += 1;

        for call in &calls {
            // Arguments must survive a round trip. A value that serializes but
            // does not parse back is corruption dressed as success.
            let encoded =
                serde_json::to_string(&call.arguments).expect("parsed arguments must serialize");
            serde_json::from_str::<serde_json::Value>(&encoded)
                .expect("parsed arguments must round-trip through serde_json");

            // The sharp one, and it holds for *every* input regardless of how
            // mangled: a returned name may never contain the opener token. If
            // it does, the scanner ran past a call boundary and swallowed the
            // next call into this one's name — which is `tool-name-scanner`
            // exactly (`call:bad call:good{x:1}` → one tool named
            // "bad call:good"). Unlike the declared-set check below this needs
            // no precondition, so it is what makes the driver non-vacuous.
            assert!(
                !call.name.contains("call:"),
                "parsed name swallowed a call boundary: {:?}\n  stream: {:?}",
                call.name,
                g.output.text,
            );
            boundary_checked += 1;

            // Only names that survive the wire format can be asserted on. A
            // declared name containing `call:` makes the stream genuinely
            // ambiguous — `call:bad call:good{x:1}` cannot be told apart from
            // two calls, and the parser answering "good" is the
            // `tool-name-scanner` fix working, not failing. The generator emits
            // such names deliberately (the grammar builder must handle them);
            // the round-trip invariant just does not apply to them.
            if g.output.mutation == Mutation::None
                && !declared.is_empty()
                && declared.iter().all(|n| stream_safe(n))
            {
                assert!(
                    declared.contains(&call.name),
                    "well-formed stream yielded undeclared tool {:?}\n  declared: {declared:?}\n  \
                     stream: {:?}",
                    call.name,
                    g.output.text,
                );
                asserted += 1;
            }
        }
    });

    assert!(total > 0, "generator produced no inputs");
    assert!(
        parsed_ok > 0,
        "no generated stream parsed successfully — the generator is producing only garbage, so \
         nothing past the reject path was exercised"
    );
    assert!(
        boundary_checked > 0,
        "no parsed call was ever checked for a swallowed boundary"
    );
    assert!(
        asserted > 0,
        "the declared-name invariant was never reachable: every well-formed draw had an \
         unencodable name, so the sharpest assertion here ran zero times"
    );
}

/// Can this name be written into `call:NAME{` and read back unambiguously?
///
/// `call:` is the killer — a name containing it is indistinguishable from a
/// second call. Braces and commas terminate the name or the argument list.
fn stream_safe(name: &str) -> bool {
    !name.is_empty()
        && !name.contains("call:")
        && !name.contains(['{', '}', ',', '\n', '\r'])
        && name.trim() == name
}

/// `gemma4_args_to_json` is reached directly as well as through the body
/// parser, so it gets its own pass over raw argument text.
#[test]
fn args_to_json_survives_generated_bodies() {
    let mut ok = 0usize;
    seeded_inputs::<GrammarAndOutput, _>(0xA265, 600, |g| {
        // Feed the whole stream and each brace-delimited slice of it; the inner
        // slices are what the function sees in production.
        // The braces are part of the input: `gemma4_args_to_json` recognizes a
        // bare key by the `{` or `,` in front of it, and parses the result as a
        // JSON object. Stripping them leaves keys unquoted and no object at
        // all, which fails for a reason that has nothing to do with the code
        // under test.
        let mut candidates = vec![g.output.text.clone()];
        if let (Some(a), Some(b)) = (g.output.text.find('{'), g.output.text.rfind('}'))
            && a < b
        {
            candidates.push(g.output.text[a..=b].to_string());
        }
        for c in candidates {
            if let Ok(v) = gemma4_args_to_json(&c) {
                ok += 1;
                let encoded = serde_json::to_string(&v).expect("value must serialize");
                serde_json::from_str::<serde_json::Value>(&encoded).expect("value must round-trip");
            }
        }
    });
    assert!(ok > 0, "no generated argument body parsed successfully");
}

/// The grammar builder takes client-supplied JSON Schema, so it is fed the
/// hostile schema alphabet directly. Errors are fine; panics are not.
#[test]
fn grammar_builder_survives_generated_tool_sets() {
    let mut built = 0usize;
    let mut rejected = 0usize;
    seeded_inputs::<ToolSet, _>(0x6244, 300, |t| {
        if t.tools.is_empty() {
            return;
        }
        match build_qwen35_tool_grammar_lark(&t.tools, ToolCalls::OneOrMore) {
            Ok(_) => built += 1,
            Err(_) => rejected += 1,
        }
    });
    assert!(
        built + rejected > 0,
        "no non-empty tool set was generated in 300 draws"
    );
    assert!(
        built > 0,
        "every generated tool set was rejected ({rejected} of them) — the generator is not \
         producing anything the builder accepts, so the build path went untested"
    );
}

/// Request bodies are deserialized by the HTTP layer before anything else runs,
/// so the shapes a client can send are worth walking even though the typed
/// structs live in `lumen-server`: this pass covers the JSON that reaches them.
#[test]
fn generated_chat_requests_are_well_formed_json() {
    let mut with_tools = 0usize;
    seeded_inputs::<ChatRequest, _>(0xC4A7, 300, |r| {
        let encoded = serde_json::to_string(&r.json).expect("request must serialize");
        let back: serde_json::Value =
            serde_json::from_str(&encoded).expect("request must round-trip");
        assert_eq!(back, r.json, "request round-trip changed the body");
        if r.json
            .get("tools")
            .is_some_and(|t| !t.as_array().unwrap_or(&vec![]).is_empty())
        {
            with_tools += 1;
        }
    });
    assert!(
        with_tools > 0,
        "no generated request carried tools — the tool-bearing path went untested"
    );
}
