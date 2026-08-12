//! Tier-0 replay of the committed fuzz inputs (005 Phase 1.4).
//!
//! This is the part of the fuzzing setup that matters most in a repo with no
//! CI running tests: soaks are occasional and machine-local, but every input
//! they ever surfaced — the hand-written seeds, and each crasher promoted into
//! `fuzz/seeds/` — replays here in milliseconds on every `cargo test`,
//! forever. SQLite runs its historical crashers on every build for the same
//! reason: a fuzzer finding is only a regression test if something replays it.
//!
//! Replays `fuzz/seeds/<target>/` and, when present, `fuzz/artifacts/<target>/`
//! (raw crash inputs cargo-fuzz wrote and someone committed mid-triage). The
//! gitignored working corpus is deliberately not replayed — thousands of
//! coverage-minimized blobs that regrow on any machine, meaningful only to
//! libFuzzer's own coverage map.
//!
//! Each replay body mirrors its fuzz target exactly. If a target's invariants
//! change, change them here too — drift between the two means a crasher can
//! pass replay while still crashing the soak, which un-fixes the bug the
//! corpus exists to pin. The `request_parse` twin lives in
//! `lumen-server/tests/fuzz_corpus_replay.rs` because its types live there.

use std::path::PathBuf;

use lumen_mlx::gemma4_tool_syntax::{gemma4_args_to_json, parse_tool_call_body};
use lumen_mlx::grammar::{ToolCalls, build_qwen35_tool_grammar_lark};
use lumen_testkit::arbitrary;
use lumen_testkit::generators::{GrammarAndOutput, ToolSet};

/// Committed inputs for `target`: seeds always, artifacts when any exist.
/// Panics on a missing/empty seed directory — an empty replay set silently
/// covering nothing is the green-by-omission failure this file exists to
/// prevent.
fn committed_inputs(target: &str) -> Vec<(PathBuf, Vec<u8>)> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fuzz");
    let mut out = Vec::new();
    for dir in [
        root.join("seeds").join(target),
        root.join("artifacts").join(target),
    ] {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue; // artifacts/ legitimately absent until a crash is found
        };
        for e in entries.filter_map(Result::ok) {
            let path = e.path();
            if path.is_file() {
                let bytes = std::fs::read(&path)
                    .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
                out.push((path, bytes));
            }
        }
    }
    assert!(
        !out.is_empty(),
        "no committed inputs for {target:?} — fuzz/seeds/{target}/ must hold at least one file"
    );
    out
}

#[test]
fn replay_tool_body_parse() {
    for (path, bytes) in committed_inputs("tool_body_parse") {
        let Ok(text) = std::str::from_utf8(&bytes) else {
            continue;
        };
        let Ok(calls) = parse_tool_call_body(text) else {
            continue;
        };
        for call in &calls {
            assert!(
                !call.name.contains("call:"),
                "{}: parsed name swallowed a call boundary: {:?}",
                path.display(),
                call.name
            );
            let encoded = serde_json::to_string(&call.arguments)
                .unwrap_or_else(|e| panic!("{}: serialize: {e}", path.display()));
            serde_json::from_str::<serde_json::Value>(&encoded)
                .unwrap_or_else(|e| panic!("{}: round-trip: {e}", path.display()));
        }
    }
}

#[test]
fn replay_args_to_json() {
    for (path, bytes) in committed_inputs("args_to_json") {
        let Ok(text) = std::str::from_utf8(&bytes) else {
            continue;
        };
        if let Ok(v) = gemma4_args_to_json(text) {
            let encoded = serde_json::to_string(&v)
                .unwrap_or_else(|e| panic!("{}: serialize: {e}", path.display()));
            serde_json::from_str::<serde_json::Value>(&encoded)
                .unwrap_or_else(|e| panic!("{}: round-trip: {e}", path.display()));
        }
    }
}

#[test]
fn replay_grammar_build() {
    for (_path, bytes) in committed_inputs("grammar_build") {
        let mut u = arbitrary::Unstructured::new(&bytes);
        let Ok(tools) = <ToolSet as arbitrary::Arbitrary>::arbitrary(&mut u) else {
            continue;
        };
        if !tools.tools.is_empty() {
            let _ = build_qwen35_tool_grammar_lark(&tools.tools, ToolCalls::OneOrMore);
        }
    }
}

#[test]
fn replay_grammar_x_output() {
    for (path, bytes) in committed_inputs("grammar_x_output") {
        let mut u = arbitrary::Unstructured::new(&bytes);
        let Ok(g) = <GrammarAndOutput as arbitrary::Arbitrary>::arbitrary(&mut u) else {
            continue;
        };
        if !g.tools.tools.is_empty() {
            let _ = build_qwen35_tool_grammar_lark(&g.tools.tools, ToolCalls::OneOrMore);
        }
        let Ok(calls) = parse_tool_call_body(&g.output.text) else {
            continue;
        };
        for call in &calls {
            assert!(
                !call.name.contains("call:"),
                "{}: parsed name swallowed a call boundary: {:?}",
                path.display(),
                call.name
            );
        }
    }
}

/// The known-defect reproducers are also asserted *positively* — not just "no
/// panic" but "the parser gets the answer right". A refactor that regressed
/// `tool-name-scanner` would pass a pure no-panic replay while shipping the
/// bug; this pins the corrected behavior to the exact input that shipped it.
#[test]
fn seed_reproducers_get_the_historically_correct_answer() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fuzz/seeds");

    // tool-name-scanner: `call:bad call:good{x:1}` must parse as ONE call named
    // "good" — the first opener has no brace and is skipped, not merged.
    let scanner = std::fs::read_to_string(root.join("tool_body_parse/seed_tool_name_scanner"))
        .expect("seed_tool_name_scanner must exist");
    let calls = parse_tool_call_body(&scanner).expect("reproducer must parse");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "good");

    // args-unicode-keys: a non-ASCII bare key must survive as itself.
    let unicode = std::fs::read_to_string(root.join("args_to_json/seed_unicode_key"))
        .expect("seed_unicode_key must exist");
    let v = gemma4_args_to_json(&unicode).expect("unicode-key reproducer must parse");
    assert_eq!(v.get("도시").and_then(|s| s.as_str()), Some("서울"));
}
