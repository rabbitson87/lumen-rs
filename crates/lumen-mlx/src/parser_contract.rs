//! What the Qwen tool-call stream parser promises over arbitrary text.
//!
//! The parser reads what a model decoded, a chunk at a time, so it has to
//! survive any UTF-8 cut anywhere. [`assert_qwen35_parser_contract`] states
//! that once, for the `qwen_parser_feed` fuzz target and its tier-0 replay in
//! `tests/fuzz_corpus_replay.rs` alike: both call it, so a committed crasher
//! cannot pass replay while still crashing the soak. Hidden from the docs; it
//! exists for those two callers and the parser's own tests.

use crate::qwen3_5_tools::{Qwen35ParseEvent, Qwen35ResponseParser};

/// Everything one run of the parser yields, to compare runs by.
#[derive(Debug, PartialEq)]
pub(crate) struct Outcome {
    /// What the events add up to: `(text, reasoning, tool-call starts)`.
    pub(crate) events: (String, String, usize),
    /// Each call's streamed `arguments` pieces, joined; a call still open at
    /// the end is the last entry.
    pub(crate) args: Vec<String>,
    pub(crate) visible: String,
    pub(crate) reasoning: String,
    pub(crate) calls: Vec<(String, serde_json::Value)>,
}

/// Feed `chunks` to a fresh parser, in order, and finish it.
pub(crate) fn run(chunks: &[&str]) -> Outcome {
    run_with(Qwen35ResponseParser::new(), chunks)
}

/// [`run`] on `p`, which may start anywhere (a thinking-open parser).
pub(crate) fn run_with(mut p: Qwen35ResponseParser, chunks: &[&str]) -> Outcome {
    let (mut text, mut reasoning, mut starts) = (String::new(), String::new(), 0);
    let (mut args, mut open) = (Vec::new(), None::<String>);
    let mut take = |event| match event {
        Qwen35ParseEvent::Text(t) => text.push_str(&t),
        Qwen35ParseEvent::Reasoning(r) => reasoning.push_str(&r),
        Qwen35ParseEvent::ToolCallStart { .. } => starts += 1,
        Qwen35ParseEvent::ToolCallArgs(piece) => open.get_or_insert_default().push_str(&piece),
        Qwen35ParseEvent::ToolCallEnd => args.extend(open.take()),
    };
    for chunk in chunks {
        p.feed(chunk).into_iter().for_each(&mut take);
    }
    p.flush().into_iter().for_each(&mut take);
    args.extend(open);
    let done = p.finish();
    Outcome {
        events: (text, reasoning, starts),
        args,
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
/// Whatever the text and the cuts: the parser does not panic, the events add up
/// to the response it finishes with, and the response is the one the whole text
/// gives — which tokens a model happens to write must not change what a client
/// is told.
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
    let whole = run(&[&text]);
    assert_eq!(whole.events.0, whole.visible, "{text:?}");
    assert_eq!(whole.events.1, whole.reasoning, "{text:?}");
    assert!(whole.events.2 >= whole.calls.len(), "{text:?}");
    assert_streamed_args_are_the_calls(&whole, &text);
    assert_eq!(run(&chunks), whole, "{text:?} as {chunks:?}");
}

/// Each finished call's streamed pieces read back as exactly its arguments, in
/// its key order. They are its serialization byte for byte while no key
/// repeats; a repeated key leaves both values in the text, and reading it back
/// keeps the first position and the last value, as the parser does.
pub(crate) fn assert_streamed_args_are_the_calls(outcome: &Outcome, text: &str) {
    assert!(outcome.args.len() >= outcome.calls.len(), "{text:?}");
    for (streamed, (_, arguments)) in outcome.args.iter().zip(&outcome.calls) {
        let read: serde_json::Value = serde_json::from_str(streamed)
            .unwrap_or_else(|e| panic!("streamed {streamed:?} is not JSON ({e}) for {text:?}"));
        assert_eq!(
            serde_json::to_string(&read).expect("serializes"),
            serde_json::to_string(arguments).expect("serializes"),
            "streamed {streamed:?} for {text:?}"
        );
    }
}
