//! Raw-bytes fuzz of `parse_tool_call_body`.
//!
//! Deliberately takes `&[u8]` rather than a generated structure: this parser's
//! whole job is surviving arbitrary decoded model text, and a byte-level
//! mutator explores delimiter-boundary bugs (a `call:` split across a UTF-8
//! boundary, a `<|"|>` with one byte flipped) that a structure-aware generator
//! rounds away. The structured exploration of the same entry point lives in
//! `grammar_x_output`.
//!
//! Invariants — the same three the deterministic driver asserts, so a crasher
//! found here is replayable there once its input is committed to the corpus:
//!   1. never panics;
//!   2. every parsed name is opener-free (`tool-name-scanner`);
//!   3. `Ok` arguments round-trip through serde_json.
#![no_main]

use libfuzzer_sys::fuzz_target;
use lumen_mlx::gemma4_tool_syntax::parse_tool_call_body;

fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };
    let Ok(calls) = parse_tool_call_body(text) else {
        return;
    };
    for call in &calls {
        assert!(
            !call.name.contains("call:"),
            "parsed name swallowed a call boundary: {:?} from {text:?}",
            call.name
        );
        let encoded = serde_json::to_string(&call.arguments).expect("Ok arguments must serialize");
        serde_json::from_str::<serde_json::Value>(&encoded)
            .expect("Ok arguments must round-trip through serde_json");
    }
});
