//! What the Qwen tool-call stream parser promises over arbitrary text.
//!
//! The parser reads what a model decoded, a chunk at a time, so it has to
//! survive any UTF-8 cut anywhere. [`assert_qwen35_parser_contract`] states
//! that once, for the `qwen_parser_feed` fuzz target and its tier-0 replay in
//! `tests/fuzz_corpus_replay.rs` alike: both call it, so a committed crasher
//! cannot pass replay while still crashing the soak. Hidden from the docs; it
//! exists for those two callers and the parser's own tests.

use crate::qwen3_5_tools::{Qwen35ParseEvent, Qwen35ResponseParser};

/// Everything one run of the parser yields.
#[derive(Debug)]
pub(crate) struct Outcome {
    /// What the events add up to: `(text, reasoning, tool-call starts)`.
    pub(crate) events: (String, String, usize),
    pub(crate) visible: String,
    pub(crate) reasoning: String,
    pub(crate) calls: Vec<(String, serde_json::Value)>,
}

/// Feed `chunks` to a fresh parser, in order, and finish it.
pub(crate) fn run(chunks: &[&str]) -> Outcome {
    let mut p = Qwen35ResponseParser::new();
    let (mut text, mut reasoning, mut starts) = (String::new(), String::new(), 0);
    for chunk in chunks {
        for event in p.feed(chunk) {
            match event {
                Qwen35ParseEvent::Text(t) => text.push_str(&t),
                Qwen35ParseEvent::Reasoning(r) => reasoning.push_str(&r),
                Qwen35ParseEvent::ToolCallStart { .. } => starts += 1,
            }
        }
    }
    let done = p.finish();
    Outcome {
        events: (text, reasoning, starts),
        visible: done.visible,
        reasoning: done.reasoning,
        calls: done
            .tool_calls
            .into_iter()
            .map(|c| (c.name, c.arguments))
            .collect(),
    }
}

/// `data` is one byte that picks how finely the text is cut, then the text
/// (invalid UTF-8 is read lossily, which keeps the fuzzer's inputs useful).
///
/// Whatever the text and the cuts, the parser does not panic and the events add
/// up to the response it finishes with — fed whole and fed in pieces alike.
///
/// What is deliberately not asserted is that the two runs agree. They do not,
/// always: the template's `\n\n` after `</think>` is swallowed only when it
/// arrives in the chunk with the tag, and a parameter value's leading newline
/// is stripped at the opener when the opener's chunk holds it and again at the
/// closer (task 025, Issues 3 and 4). Asserting agreement would have this fail
/// on those two until they are fixed, and hide anything new behind them.
///
/// # Panics
/// On any violation; this is assertion machinery for tests and fuzz targets.
pub fn assert_qwen35_parser_contract(data: &[u8]) {
    let Some((&density, rest)) = data.split_first() else {
        return;
    };
    let text = String::from_utf8_lossy(rest);
    let mut chunks = Vec::new();
    let mut from = 0;
    if !rest.is_empty() {
        let mask = density % 4 + 1;
        for (at, _) in text.char_indices().skip(1) {
            if rest[at % rest.len()] % mask == 0 {
                chunks.push(&text[from..at]);
                from = at;
            }
        }
    }
    chunks.push(&text[from..]);
    for (how, fed) in [("whole", vec![&*text]), ("in pieces", chunks)] {
        let run = run(&fed);
        assert_eq!(run.events.0, run.visible, "{how}: {text:?}");
        assert_eq!(run.events.1, run.reasoning, "{how}: {text:?}");
        assert!(run.events.2 >= run.calls.len(), "{how}: {text:?}");
    }
}
