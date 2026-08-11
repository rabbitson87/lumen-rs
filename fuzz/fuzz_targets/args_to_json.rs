//! Raw-bytes fuzz of `gemma4_args_to_json`.
//!
//! The pseudo-JSON argument converter does placeholder substitution, bare-key
//! quoting and a final serde parse — three passes over the same string, which
//! is exactly the shape where an index from pass one goes stale in pass three.
//! `args-unicode-keys` was that bug (byte indexing re-encoded multi-byte keys
//! as Latin-1 mojibake), so the mutator gets raw bytes and the UTF-8 gate up
//! front mirrors production, where the text arrives already decoded.
//!
//! Invariants: never panics; `Ok` means genuinely valid JSON, since a value
//! that serializes but does not parse back is corruption dressed as success.
#![no_main]

use libfuzzer_sys::fuzz_target;
use lumen_mlx::gemma4_tool_syntax::gemma4_args_to_json;

fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };
    if let Ok(v) = gemma4_args_to_json(text) {
        let encoded = serde_json::to_string(&v).expect("Ok value must serialize");
        serde_json::from_str::<serde_json::Value>(&encoded)
            .expect("Ok value must round-trip through serde_json");
    }
});
