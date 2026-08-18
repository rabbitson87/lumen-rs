//! Structured fuzz of the Lark literal-escaping pair.
//!
//! `grammar_build` reaches `lark_literal` and `is_safe_ident` through the whole
//! builder, but it asserts only that the builder does not panic. That leaves
//! the property those two functions exist to guarantee untested: a tool name
//! arriving from a client must not be able to **escape its own quoting**. If it
//! can, everything after it in the emitted grammar is caller-controlled — the
//! name stops being data and becomes grammar. `grammar-literal-escaping`
//! shipped exactly there.
//!
//! The contract itself lives in `lumen_testkit::invariants` so this target and
//! its tier-0 replay assert the same thing by construction rather than by
//! someone remembering to edit both.
//!
//! ## Deliberately not asserted
//!
//! Control characters outside `\n\r\t` (`\x00`, `\x0b`, `\x1b`, …) are passed
//! through raw by `lark_literal`, and this target does not call that a defect.
//! Whether Lark accepts a raw `\x0b` inside a string literal is a question
//! about llguidance's parser, not about this function, and asserting a guess
//! would make the target fire on its first run for a reason nobody could act
//! on. That end-to-end question is answered where it can be: the
//! feature-gated grammar tests build a real matcher.
#![no_main]

use libfuzzer_sys::fuzz_target;
use lumen_mlx::grammar::{is_safe_ident, lark_literal};
use lumen_testkit::invariants::assert_lark_literal_contract;

// Raw bytes rather than an `Arbitrary` type, so a seed file *is* the input
// string and stays readable in review — same choice as `tool_body_parse`.
fuzz_target!(|data: &[u8]| {
    let Ok(s) = std::str::from_utf8(data) else {
        return;
    };
    assert_lark_literal_contract(s, &lark_literal(s), is_safe_ident(s));
});
