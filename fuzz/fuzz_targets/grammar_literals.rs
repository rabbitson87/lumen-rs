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
//! ## The gap this target used to document, and how it closed
//!
//! The first version of this file declined to assert anything about control
//! characters outside `\n\r\t`, on the grounds that whether Lark accepts a raw
//! `\x0b` is a question about llguidance rather than about `lark_literal`, and
//! that guessing would make the target fire for a reason nobody could act on.
//!
//! That was the right call about *guessing* and the wrong call about stopping
//! there. The question was answerable in a unit test — `ApproximateTokEnv::
//! single_byte_env` builds a real matcher with no model — and the answer was
//! that every one of them is a `lexer error`, which drops the grammar and lets
//! the model invent a tool nobody declared. See `grammar-control-chars`.
//!
//! So the contract now asserts no raw ASCII control survives escaping, and the
//! end-to-end half lives in `grammar::tests::
//! lark_grammar_escapes_control_chars_in_a_tool_name`. A documented gap is
//! better than a silent one, but it is not the same as a closed one.
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
