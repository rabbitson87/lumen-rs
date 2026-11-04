//! Raw-bytes fuzz of the Qwen tool-call stream parser, fed in chunks.
//!
//! The parser's input is whatever a model decoded, delivered a chunk at a time,
//! so the interesting bugs sit where a chunk boundary meets a delimiter or a
//! multibyte character (a non-breaking space after `<tool_call>` once panicked
//! the engine thread). The first byte picks how finely the text is cut; the
//! rest is the text. The invariants live in
//! `lumen_mlx::parser_contract::assert_qwen35_parser_contract`, which the tier-0
//! replay in `crates/lumen-mlx/tests/fuzz_corpus_replay.rs` calls too.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    lumen_mlx::parser_contract::assert_qwen35_parser_contract(data);
});
