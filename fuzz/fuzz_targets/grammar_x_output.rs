//! The dbsqlfuzz differentiator: schema set and model output mutated together.
//!
//! `tool_body_parse` explores the text surface alone; this target draws a
//! `ToolSet` *and* a `ModelOutput` referencing its declared names from one
//! `Unstructured`, so the mutator can drive the two sides apart — the
//! configuration that turned `tool-name-scanner` and `grammar-rule-names` from
//! "found in production" into "findable in a fuzz run".
//!
//! Invariants mirror the deterministic driver in
//! `crates/lumen-mlx/tests/tool_surface_fuzz.rs`, including its precondition
//! discipline: the unconditional assertions (no swallowed opener, JSON
//! round-trip) hold for every input, while the declared-set check applies only
//! to well-formed streams over encodable names — asserting it on an ambiguous
//! stream would fault the parser for refusing to read minds.
#![no_main]

use libfuzzer_sys::fuzz_target;
use lumen_mlx::gemma4_tool_syntax::parse_tool_call_body;
use lumen_mlx::grammar::build_qwen35_tool_grammar_lark;
use lumen_testkit::generators::{GrammarAndOutput, Mutation};

fn stream_safe(name: &str) -> bool {
    !name.is_empty()
        && !name.contains("call:")
        && !name.contains(['{', '}', ',', '\n', '\r'])
        && name.trim() == name
}

fuzz_target!(|g: GrammarAndOutput| {
    // Grammar side: must survive whatever schema the output was paired with.
    if !g.tools.tools.is_empty() {
        let _ = build_qwen35_tool_grammar_lark(&g.tools.tools);
    }

    // Parser side.
    let Ok(calls) = parse_tool_call_body(&g.output.text) else {
        return;
    };
    let declared = g.tools.names();
    for call in &calls {
        assert!(
            !call.name.contains("call:"),
            "parsed name swallowed a call boundary: {:?} from {:?}",
            call.name,
            g.output.text
        );
        let encoded = serde_json::to_string(&call.arguments).expect("Ok arguments must serialize");
        serde_json::from_str::<serde_json::Value>(&encoded).expect("Ok arguments must round-trip");

        if g.output.mutation == Mutation::None
            && !declared.is_empty()
            && declared.iter().all(|n| stream_safe(n))
        {
            assert!(
                declared.contains(&call.name),
                "well-formed stream yielded undeclared tool {:?}\n  declared: {declared:?}\n  \
                 stream: {:?}",
                call.name,
                g.output.text
            );
        }
    }
});
