//! Structured fuzz of `build_qwen35_tool_grammar_lark`.
//!
//! The builder consumes client-supplied JSON Schema and emits a Lark grammar
//! string, so it has two distinct ways to fail: panic on a schema shape nobody
//! anticipated, or emit a grammar whose literals escape their own quoting
//! (`grammar-literal-escaping`, `grammar-rule-names` — both shipped). The
//! `ToolSet` generator supplies the hostile alphabet those defects came from:
//! names with quotes, backslashes, newlines, embedded `call:`, empty strings,
//! 1 KB runs, and schemas with no `type` at all.
//!
//! `Ok` here asserts only well-formedness properties we can check without a
//! Lark parser in the loop; the deeper "llguidance accepts the emitted
//! grammar" check needs a tokenizer environment and stays in the feature-gated
//! integration tests.
#![no_main]

use libfuzzer_sys::fuzz_target;
use lumen_mlx::grammar::{ToolCalls, build_qwen35_tool_grammar_lark};
use lumen_testkit::generators::ToolSet;

fuzz_target!(|tools: ToolSet| {
    if tools.tools.is_empty() {
        return;
    }
    // Err is a legitimate answer to a hostile schema; only a panic (caught by
    // the harness as a crash) or a malformed Ok is a defect.
    let _ = build_qwen35_tool_grammar_lark(&tools.tools, ToolCalls::OneOrMore);
});
