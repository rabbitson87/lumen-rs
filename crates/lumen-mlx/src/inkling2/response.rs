//! Reads what Inkling 2 generates: reasoning, the visible answer and tool calls.
//!
//! The checkpoint's `tokenizer_config.json` describes the format as a `response_template`; this is
//! that description as a state machine over token ids, which is sturdier than patterns over text
//! because every marker is one control token. A turn is a run of messages and ends at
//! `<|content_model_end_sampling|>`:
//!
//! ```text
//! <|message_model|> [NAME] <|content_thinking|>   REASONING   <|end_message|>
//! <|message_model|> [NAME] <|content_text|>       ANSWER      <|end_message|>
//! <|message_model|> NAME   <|content_invoke_tool_json|> {"name":…,"args":{…}} <|end_message|>
//! <|content_model_end_sampling|>
//! ```
//!
//! The header (`<|message_model|>` and the name) may be missing before a content marker, as the
//! template's patterns allow. Several reasoning or answer blocks join with nothing between them.

use anyhow::Result;
use serde_json::Value;

use crate::chat_io::{BackendStreamEvent, ParsedResponse, ParsedToolCall};
use crate::stream_detok::StreamDetok;

/// The ids of the markers, looked up by their spelling in the checkpoint's tokenizer.
#[derive(Debug, Clone, Copy)]
pub struct Markers {
    pub message_model: u32,
    pub content_text: u32,
    pub content_thinking: u32,
    pub content_tool_error: u32,
    pub content_invoke_tool_json: u32,
    pub content_invoke_tool_text: u32,
    pub end_message: u32,
    pub end_sampling: u32,
}

impl Markers {
    /// Looks every marker up with `id_of` (a tokenizer's single-token lookup).
    pub fn lookup(id_of: impl Fn(&str) -> Option<u32>) -> Result<Self> {
        let get = |name: &str| {
            id_of(name).ok_or_else(|| anyhow::anyhow!("the tokenizer has no {name} token"))
        };
        Ok(Self {
            message_model: get("<|message_model|>")?,
            content_text: get("<|content_text|>")?,
            content_thinking: get("<|content_thinking|>")?,
            content_tool_error: get("<|content_tool_error|>")?,
            content_invoke_tool_json: get("<|content_invoke_tool_json|>")?,
            content_invoke_tool_text: get("<|content_invoke_tool_text|>")?,
            end_message: get("<|end_message|>")?,
            end_sampling: get("<|content_model_end_sampling|>")?,
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Reasoning,
    Answer,
    ToolCall,
}

enum State {
    /// Outside a message.
    Between,
    /// After `<|message_model|>`: the name, up to the content marker.
    Header,
    /// Inside a message body, up to `<|end_message|>`.
    Body(Kind),
    /// `<|content_model_end_sampling|>` came.
    Done,
}

/// Incremental parser: feed the generated token ids one at a time.
pub struct ResponseParser {
    markers: Markers,
    state: State,
    /// The tokens of the name in the header being read.
    name: Vec<u32>,
    /// The tokens of the open body.
    body: Vec<u32>,
    /// The name a tool call's header gave.
    header_name: String,
    detok: StreamDetok,
    response: ParsedResponse,
}

impl ResponseParser {
    pub fn new(markers: Markers) -> Self {
        Self {
            markers,
            state: State::Between,
            name: Vec::new(),
            body: Vec::new(),
            header_name: String::new(),
            detok: StreamDetok::tail(),
            response: ParsedResponse::default(),
        }
    }

    /// Whether the model ended its turn (`<|content_model_end_sampling|>` was read).
    pub fn is_done(&self) -> bool {
        matches!(self.state, State::Done)
    }

    /// Feeds one generated token. `decode` turns ids into text; `emit` gets the events a stream
    /// carries: text and reasoning deltas as they become ready, and the start of a tool call.
    pub fn push(
        &mut self,
        token: u32,
        decode: &dyn Fn(&[u32]) -> Result<String>,
        emit: &mut dyn FnMut(BackendStreamEvent<'_>) -> Result<()>,
    ) -> Result<()> {
        let m = self.markers;
        match self.state {
            State::Done => {}
            State::Between => {
                if token == m.message_model {
                    self.name.clear();
                    self.state = State::Header;
                } else if token == m.end_sampling {
                    self.state = State::Done;
                } else if token == m.end_message {
                    // A stray terminator: nothing is open.
                } else if let Some(kind) = self.content_kind(token) {
                    self.open_body(kind, decode, emit)?;
                } else {
                    // Text with no header. Keep it rather than lose it.
                    self.open_body(Kind::Answer, decode, emit)?;
                    self.push(token, decode, emit)?;
                }
            }
            State::Header => {
                if let Some(kind) = self.content_kind(token) {
                    self.open_body(kind, decode, emit)?;
                } else if token == m.end_sampling {
                    self.state = State::Done;
                } else if token == m.end_message {
                    self.state = State::Between;
                } else {
                    self.name.push(token);
                }
            }
            State::Body(kind) => {
                if token == m.end_message || token == m.end_sampling {
                    self.close_body(kind, decode, emit)?;
                    self.state = if token == m.end_sampling {
                        State::Done
                    } else {
                        State::Between
                    };
                } else {
                    self.body.push(token);
                    if kind != Kind::ToolCall {
                        self.send_ready(kind, decode, emit)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// The generation stopped (the model's end, or the budget). Closes what is open and returns
    /// the response. A tool call still open is not in `tool_calls` and sets `cut_call`.
    pub fn finish(
        mut self,
        decode: &dyn Fn(&[u32]) -> Result<String>,
        emit: &mut dyn FnMut(BackendStreamEvent<'_>) -> Result<()>,
    ) -> Result<ParsedResponse> {
        if let State::Body(kind) = self.state {
            if kind == Kind::ToolCall {
                self.response.cut_call = true;
            } else {
                self.close_body(kind, decode, emit)?;
            }
        }
        Ok(self.response)
    }

    fn content_kind(&self, token: u32) -> Option<Kind> {
        let m = &self.markers;
        if token == m.content_thinking {
            Some(Kind::Reasoning)
        } else if token == m.content_text
            || token == m.content_tool_error
            || token == m.content_invoke_tool_text
        {
            Some(Kind::Answer)
        } else if token == m.content_invoke_tool_json {
            Some(Kind::ToolCall)
        } else {
            None
        }
    }

    fn open_body(
        &mut self,
        kind: Kind,
        decode: &dyn Fn(&[u32]) -> Result<String>,
        emit: &mut dyn FnMut(BackendStreamEvent<'_>) -> Result<()>,
    ) -> Result<()> {
        self.body.clear();
        self.detok = StreamDetok::tail();
        self.header_name = if kind == Kind::ToolCall && !self.name.is_empty() {
            decode(&self.name)?
        } else {
            String::new()
        };
        self.name.clear();
        if kind == Kind::ToolCall && !self.header_name.is_empty() {
            emit(BackendStreamEvent::ToolCallStart {
                name: &self.header_name,
            })?;
        }
        self.state = State::Body(kind);
        Ok(())
    }

    /// Sends the text of the open body that is ready.
    fn send_ready(
        &mut self,
        kind: Kind,
        decode: &dyn Fn(&[u32]) -> Result<String>,
        emit: &mut dyn FnMut(BackendStreamEvent<'_>) -> Result<()>,
    ) -> Result<()> {
        if let Some(piece) = self.detok.next(&self.body, |t| decode(t)) {
            self.deliver(kind, &piece, emit)?;
        }
        Ok(())
    }

    fn deliver(
        &mut self,
        kind: Kind,
        piece: &str,
        emit: &mut dyn FnMut(BackendStreamEvent<'_>) -> Result<()>,
    ) -> Result<()> {
        match kind {
            Kind::Reasoning => {
                self.response.reasoning.push_str(piece);
                emit(BackendStreamEvent::Reasoning(piece))
            }
            _ => {
                self.response.visible.push_str(piece);
                emit(BackendStreamEvent::Text(piece))
            }
        }
    }

    fn close_body(
        &mut self,
        kind: Kind,
        decode: &dyn Fn(&[u32]) -> Result<String>,
        emit: &mut dyn FnMut(BackendStreamEvent<'_>) -> Result<()>,
    ) -> Result<()> {
        if kind != Kind::ToolCall {
            if let Some(rest) = self.detok.finish(&self.body, |t| decode(t)) {
                self.deliver(kind, &rest, emit)?;
            }
            return Ok(());
        }
        let text = decode(&self.body)?;
        match parse_tool_call(&text, &self.header_name) {
            Some(call) => self.response.tool_calls.push(call),
            None => {
                // A call the model wrote that is not JSON of the promised shape: the client
                // should see it, not lose it.
                self.deliver(Kind::Answer, &text, emit)?;
            }
        }
        Ok(())
    }
}

/// `{"name":NAME,"args":{…}}`, the name falling back to the one in the header.
fn parse_tool_call(text: &str, header_name: &str) -> Option<ParsedToolCall> {
    let value: Value = serde_json::from_str(text.trim()).ok()?;
    let object = value.as_object()?;
    let name = object
        .get("name")
        .and_then(Value::as_str)
        .filter(|n| !n.is_empty())
        .unwrap_or(header_name);
    if name.is_empty() {
        return None;
    }
    let arguments = match object.get("args") {
        Some(args @ Value::Object(_)) => args.clone(),
        None | Some(Value::Null) => Value::Object(Default::default()),
        Some(_) => return None,
    };
    Some(ParsedToolCall {
        name: name.to_string(),
        arguments,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: Markers = Markers {
        message_model: 1,
        content_text: 2,
        content_thinking: 3,
        content_tool_error: 4,
        content_invoke_tool_json: 5,
        content_invoke_tool_text: 6,
        end_message: 7,
        end_sampling: 8,
    };

    /// Tokens 100.. are text pieces: a table from id to string.
    const PIECES: &[&str] = &[
        "Hel",
        "lo",
        " world",
        "{\"name\":\"get_weather\",",
        "\"args\":{\"city\":\"Paris\"}}",
        "get_weather",
        "I think",
        " so",
        "{\"na",
        "me\":\"f\",\"args\":[1]}",
        "\u{fffd}",
    ];

    fn decode(ids: &[u32]) -> Result<String> {
        Ok(ids
            .iter()
            .map(|&i| PIECES[(i - 100) as usize])
            .collect::<String>())
    }

    #[derive(Debug, PartialEq)]
    enum Ev {
        Text(String),
        Reasoning(String),
        Start(String),
    }

    fn run(tokens: &[u32]) -> (ParsedResponse, Vec<Ev>, bool) {
        let mut parser = ResponseParser::new(M);
        let mut events = Vec::new();
        let mut emit = |e: BackendStreamEvent<'_>| -> Result<()> {
            match e {
                BackendStreamEvent::Text(t) => events.push(Ev::Text(t.to_string())),
                BackendStreamEvent::Reasoning(t) => events.push(Ev::Reasoning(t.to_string())),
                BackendStreamEvent::ToolCallStart { name } => events.push(Ev::Start(name.into())),
                _ => {}
            }
            Ok(())
        };
        for &t in tokens {
            parser.push(t, &decode, &mut emit).unwrap();
        }
        let done = parser.is_done();
        let response = parser.finish(&decode, &mut emit).unwrap();
        (response, events, done)
    }

    #[test]
    fn reasoning_then_an_answer_then_the_end() {
        let (r, events, done) = run(&[1, 3, 106, 107, 7, 1, 2, 100, 101, 102, 7, 8]);
        assert_eq!(r.reasoning, "I think so");
        assert_eq!(r.visible, "Hello world");
        assert!(r.tool_calls.is_empty() && !r.cut_call && done);
        assert_eq!(events[0], Ev::Reasoning("I think".into()));
        assert!(events.contains(&Ev::Text("Hel".into())));
    }

    #[test]
    fn an_answer_with_no_header_is_still_an_answer() {
        let (r, _, done) = run(&[2, 100, 101, 7, 8]);
        assert_eq!(r.visible, "Hello");
        assert!(done);
    }

    #[test]
    fn a_tool_call_takes_its_name_from_the_header_and_its_arguments_from_the_json() {
        let (r, events, done) = run(&[1, 105, 5, 103, 104, 7, 8]);
        assert!(done);
        assert_eq!(events, vec![Ev::Start("get_weather".into())]);
        assert_eq!(r.tool_calls.len(), 1);
        assert_eq!(r.tool_calls[0].name, "get_weather");
        assert_eq!(
            r.tool_calls[0].arguments,
            serde_json::json!({"city": "Paris"})
        );
        assert!(r.visible.is_empty());
    }

    #[test]
    fn an_answer_and_two_calls_in_one_turn() {
        let (r, _, _) = run(&[
            1, 2, 100, 7, 1, 105, 5, 103, 104, 7, 1, 105, 5, 103, 104, 7, 8,
        ]);
        assert_eq!(r.visible, "Hel");
        assert_eq!(r.tool_calls.len(), 2);
    }

    #[test]
    fn a_call_cut_by_the_budget_is_not_a_call() {
        let (r, _, done) = run(&[1, 105, 5, 103]);
        assert!(r.tool_calls.is_empty() && r.cut_call && !done);
    }

    #[test]
    fn a_call_that_is_not_the_promised_json_is_shown_not_lost() {
        let (r, events, _) = run(&[1, 5, 108, 109, 7, 8]);
        assert!(r.tool_calls.is_empty());
        assert_eq!(r.visible, "{\"name\":\"f\",\"args\":[1]}");
        assert_eq!(
            events,
            vec![Ev::Text("{\"name\":\"f\",\"args\":[1]}".into())]
        );
    }

    #[test]
    fn a_budget_cut_in_the_middle_of_an_answer_keeps_what_was_written() {
        let (r, _, done) = run(&[1, 2, 100, 101]);
        assert_eq!(r.visible, "Hello");
        assert!(!done);
    }

    #[test]
    fn the_name_in_the_json_wins_and_a_missing_one_falls_back_to_the_header() {
        let call = parse_tool_call(r#"{"name":"a","args":{}}"#, "b").unwrap();
        assert_eq!(call.name, "a");
        let call = parse_tool_call(r#"{"args":{"x":1}}"#, "b").unwrap();
        assert_eq!(call.name, "b");
        assert!(parse_tool_call(r#"{"args":{}}"#, "").is_none());
        assert!(parse_tool_call(r#"{"name":"a","args":3}"#, "").is_none());
        assert_eq!(
            parse_tool_call(r#"{"name":"a"}"#, "").unwrap().arguments,
            serde_json::json!({})
        );
    }

    #[test]
    fn markers_are_looked_up_by_spelling() {
        let ids = |s: &str| {
            Some(match s {
                "<|message_model|>" => 1,
                "<|content_text|>" => 2,
                "<|content_thinking|>" => 3,
                "<|content_tool_error|>" => 4,
                "<|content_invoke_tool_json|>" => 5,
                "<|content_invoke_tool_text|>" => 6,
                "<|end_message|>" => 7,
                "<|content_model_end_sampling|>" => 8,
                _ => return None,
            })
        };
        let m = Markers::lookup(ids).unwrap();
        assert_eq!((m.message_model, m.end_sampling), (1, 8));
        let err = Markers::lookup(|s| (s != "<|end_message|>").then_some(1)).unwrap_err();
        assert!(err.to_string().contains("<|end_message|>"), "{err}");
    }
}
