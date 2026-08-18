//! Structured fuzz of the Qwen tool-calling system-prompt renderer.
//!
//! `render_tools_system_block` is the last thing between a client's tool
//! declarations and the model's context. Everything downstream — the grammar,
//! the parser, the scanner — is tested against the tools the *caller* declared;
//! nothing tested that the model was actually shown them. A tool silently
//! dropped or mangled here fails invisibly: the model simply never calls it,
//! which reads as the model being unhelpful rather than as a bug.
//!
//! The contract lives in `lumen_testkit::invariants` so this target and its
//! tier-0 replay cannot drift apart.
#![no_main]

use libfuzzer_sys::fuzz_target;
use lumen_mlx::chat_io::ToolDef;
use lumen_mlx::render_tools_system_block;
use lumen_testkit::generators::ToolSet;
use lumen_testkit::invariants::assert_tools_block_contract;
use serde_json::Value;

fuzz_target!(|input: (ToolSet, Option<String>)| {
    let (tools, extra) = input;

    // Borrow the generated JSON into the renderer's view of a tool. A tool
    // whose `function.name` is absent or not a string is not representable as a
    // `ToolDef` at all, so it is skipped rather than asserted about — the
    // generator only emits string names, but the JSON shape permits otherwise.
    let defs: Vec<ToolDef<'_>> = tools
        .tools
        .iter()
        .filter_map(|t| {
            let f = t.get("function")?;
            Some(ToolDef {
                name: f.get("name")?.as_str()?,
                description: f.get("description").and_then(Value::as_str),
                parameters: f.get("parameters"),
                response: None,
            })
        })
        .collect();

    let declared: Vec<&str> = defs.iter().map(|d| d.name).collect();
    let rendered = render_tools_system_block(&defs, extra.as_deref());
    assert_tools_block_contract(&declared, &rendered, extra.as_deref());
});
