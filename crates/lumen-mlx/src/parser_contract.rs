//! What the Qwen tool-call stream parser promises over arbitrary text.
//!
//! The parser reads what a model decoded, a chunk at a time, so it has to
//! survive any UTF-8 cut anywhere. [`assert_qwen35_parser_contract`] states
//! that once, for the `qwen_parser_feed` fuzz target and its tier-0 replay in
//! `tests/fuzz_corpus_replay.rs` alike: both call it, so a committed crasher
//! cannot pass replay while still crashing the soak. Hidden from the docs; it
//! exists for those two callers and the parser's own tests.

use crate::qwen3_5_tools::{
    JsonTypes, ParamType, ParamTypes, Qwen35ParseEvent, Qwen35ResponseParser,
};

/// Each way a tool schema can have the parser read a value, every parameter
/// typed alike: the contract holds for each.
const PARAM_TYPES: [ParamType; 7] = [
    ParamType::Untyped,
    ParamType::String,
    ParamType::Boolean,
    ParamType::Integer,
    ParamType::Number,
    ParamType::Union(JsonTypes::BOOLEAN.or(JsonTypes::NULL)),
    ParamType::Union(JsonTypes::STRING.or(JsonTypes::INTEGER).or(JsonTypes::NULL)),
];

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
#[cfg(test)]
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
    for ty in PARAM_TYPES {
        let run = |chunks: &[&str]| {
            run_with(
                Qwen35ResponseParser::new().with_param_types(ParamTypes::all(ty)),
                chunks,
            )
        };
        let whole = run(&[&text]);
        assert_eq!(whole.events.0, whole.visible, "{text:?} {ty:?}");
        assert_eq!(whole.events.1, whole.reasoning, "{text:?} {ty:?}");
        assert!(whole.events.2 >= whole.calls.len(), "{text:?} {ty:?}");
        assert_streamed_args_are_the_calls(&whole, &text);
        assert_eq!(run(&chunks), whole, "{text:?} as {chunks:?} {ty:?}");
    }
}

/// Each finished call's streamed pieces read back as exactly its arguments, in
/// its key order. They are its serialization byte for byte while no key
/// repeats; a repeated key leaves both values in the text, and reading it back
/// keeps the first position and the last value, as the parser does.
///
/// Both sides are read back by the same parser, as a client reads the stream
/// and the whole body: serde_json's default float parsing is not correctly
/// rounded, so a 22-digit number read back once differs by an ulp from the
/// value the parser took with a correctly rounded one.
pub(crate) fn assert_streamed_args_are_the_calls(outcome: &Outcome, text: &str) {
    assert!(outcome.args.len() >= outcome.calls.len(), "{text:?}");
    let read_back = |json: &str| {
        let value: serde_json::Value = serde_json::from_str(json)
            .unwrap_or_else(|e| panic!("{json:?} is not JSON ({e}) for {text:?}"));
        serde_json::to_string(&value).expect("serializes")
    };
    for (streamed, (_, arguments)) in outcome.args.iter().zip(&outcome.calls) {
        let whole = serde_json::to_string(arguments).expect("serializes");
        assert_eq!(
            read_back(streamed),
            read_back(&whole),
            "streamed {streamed:?} for {text:?}"
        );
    }
}
