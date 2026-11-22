//! The Inkling 2 chat prompt: what the checkpoint's `chat_template.jinja` renders, in Rust.
//!
//! The template has no generation prompt: the prompt ends after the last message (or the thinking
//! effort), and the model writes its own `<|message_model|>` header. A reply is
//! `<|message_model|><|content_thinking|>…<|end_message|>` (reasoning), then
//! `<|message_model|><|content_text|>…<|end_message|>` (the answer) and/or
//! `<|message_model|>NAME<|content_invoke_tool_json|>{…}<|end_message|>` (a tool call), then
//! `<|content_model_end_sampling|>`.
//!
//! It is a port of the template, not a use of it: the template carries a `{% generation %}` tag
//! and `raise_exception`, which this crate's template engine does not have, and its text has to
//! be built as a [`PromptText`] so what the client sent is encoded as ordinary text. The golden
//! prompts in `tests/fixtures/inkling2_chat/golden.json` (rendered by Hugging Face from the real
//! template, `scripts/golden/inkling2_chat_golden.py`) pin the two together.

use anyhow::{Result, bail};
use serde_json::Value;

use crate::chat_io::{ChatTurn, ToolDef, split_reasoning_envelope};
use crate::prompt_text::PromptText;

pub(crate) const MESSAGE_USER: &str = "<|message_user|>";
pub(crate) const MESSAGE_MODEL: &str = "<|message_model|>";
pub(crate) const MESSAGE_SYSTEM: &str = "<|message_system|>";
pub(crate) const MESSAGE_TOOL: &str = "<|message_tool|>";
pub(crate) const CONTENT_TEXT: &str = "<|content_text|>";
pub(crate) const CONTENT_THINKING: &str = "<|content_thinking|>";
pub(crate) const CONTENT_XML: &str = "<|content_xml|>";
pub(crate) const CONTENT_IMAGE: &str = "<|content_image|>";
/// The token an image block is: the template writes one per image, the processor expands it to
/// one per block.
pub(crate) const IMAGE_TOKEN: &str = "<|unused_200054|>";
pub(crate) const CONTENT_INVOKE_TOOL_JSON: &str = "<|content_invoke_tool_json|>";
pub(crate) const END_MESSAGE: &str = "<|end_message|>";
pub(crate) const END_SAMPLING: &str = "<|content_model_end_sampling|>";

/// The highest thinking effort the template accepts.
const MAX_EFFORT: f64 = 0.99;
/// What the template uses when the request names none: the `high` level.
pub const DEFAULT_EFFORT: Effort = Effort(0.9);

/// A thinking effort level in `[0, 0.99]`, the number the template prints:
/// `Thinking effort level: 0.9`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Effort(f64);

impl Effort {
    /// A level, which must be in `[0, 0.99]`.
    pub fn new(level: f64) -> Result<Self> {
        if !(0.0..=MAX_EFFORT).contains(&level) {
            bail!("reasoning_effort must be in [0.0, {MAX_EFFORT}], got {level}");
        }
        Ok(Self(level))
    }

    pub fn level(self) -> f64 {
        self.0
    }

    /// The level in ten-thousandths, which is how [`crate::chat_io::ReasoningEffort::Level`]
    /// carries it (a level with more than four decimals is rounded to four).
    pub fn basis_points(self) -> u16 {
        (self.0 * 10_000.0).round() as u16
    }

    /// The level a `ReasoningEffort::Level` carries.
    pub fn from_basis_points(basis_points: u16) -> Self {
        Self(f64::from(basis_points) / 10_000.0)
    }

    /// The template's vocabulary: `none` 0, `minimal` 0.1, `low` 0.2, `medium` 0.7, `high` 0.9,
    /// `xhigh` and `max` 0.99. A number in the same range is taken as a level too (the template
    /// itself only takes numbers that arrive as numbers, but the request carries a string).
    /// `None` for a spelling that is neither.
    pub fn parse(raw: &str) -> Option<Self> {
        let key = raw.trim().to_ascii_lowercase();
        let level = match key.as_str() {
            "none" | "off" | "disabled" | "" => 0.0,
            "minimal" => 0.1,
            "low" => 0.2,
            "medium" => 0.7,
            "high" => 0.9,
            "xhigh" | "max" => 0.99,
            other => other.parse::<f64>().ok()?,
        };
        Self::new(level).ok()
    }

    /// The effort of a request. `thinking` is the server's resolved switch and `raw` the client's
    /// `reasoning_effort`. With thinking on, the named level (an unknown spelling falls back to
    /// the default, as a server should not fail a request over a spelling) or the default. With
    /// thinking off, 0 — except that a level the client named at or below `minimal` is kept, so
    /// `minimal` is the template's 0.1 and not 0.
    pub fn resolve(thinking: bool, raw: Option<&str>) -> Self {
        Self::resolve_named(thinking, raw.and_then(Self::parse))
    }

    /// [`Self::resolve`] for a level already read from the request.
    pub fn resolve_named(thinking: bool, named: Option<Self>) -> Self {
        match (thinking, named) {
            (true, Some(e)) => e,
            (true, None) => DEFAULT_EFFORT,
            (false, Some(e)) if e.0 <= 0.1 => e,
            (false, _) => Self(0.0),
        }
    }

    /// The text the template prints for the level (`0` for zero, else Python's `str(float)`).
    pub fn text(self) -> String {
        if self.0 == 0.0 {
            "0".to_string()
        } else {
            python_float_repr(self.0)
        }
    }

    /// A tag for cache keys: every level that renders differently has its own.
    pub fn cache_tag(self) -> String {
        self.text()
    }
}

/// Renders `turns` and the declared `tools` the way the checkpoint's template does.
///
/// The thinking effort is emitted before the first message that is not a system message (so after
/// the leading system messages), or at the end when there are only system messages. Client text
/// (messages, tool definitions, call names and arguments) is marked as such in the result.
pub fn render_prompt(
    turns: &[ChatTurn<'_>],
    tools: &[ToolDef<'_>],
    effort: Effort,
) -> Result<PromptText> {
    render_prompt_with_images(turns, tools, effort, &[])
}

/// [`render_prompt`] for turns that carry images. `images[i]` lists the image tokens of each
/// image attached to `turns[i]` (its blocks; a turn past the end of `images` has none). Each image
/// is a message of its own before the turn's text, as the template renders image parts and as the
/// oMLX registration orders them (images first), with its token written once per block as the
/// processor expands it. Only user and tool turns can carry images.
pub fn render_prompt_with_images(
    turns: &[ChatTurn<'_>],
    tools: &[ToolDef<'_>],
    effort: Effort,
    images: &[Vec<usize>],
) -> Result<PromptText> {
    let mut out = PromptText::new();
    if !tools.is_empty() {
        out.push_str(MESSAGE_SYSTEM);
        out.push_str("tool_declare");
        out.push_str(CONTENT_XML);
        out.push_content(&declared_tools_json(tools));
        out.push_str(END_MESSAGE);
    }
    let mut effort_emitted = false;
    for (index, turn) in turns.iter().enumerate() {
        let attached: &[usize] = images.get(index).map_or(&[], Vec::as_slice);
        if !attached.is_empty() && !matches!(turn, ChatTurn::User(_) | ChatTurn::Tool { .. }) {
            bail!("image content is only supported for user and tool messages");
        }
        if !effort_emitted && !matches!(turn, ChatTurn::System(_)) {
            push_effort(&mut out, effort);
            effort_emitted = true;
        }
        match turn {
            ChatTurn::System(text) => push_text_message(&mut out, MESSAGE_SYSTEM, text),
            ChatTurn::User(text) => {
                for &blocks in attached {
                    push_image_message(&mut out, MESSAGE_USER, blocks);
                }
                if !text.is_empty() || attached.is_empty() {
                    push_text_message(&mut out, MESSAGE_USER, text);
                }
            }
            ChatTurn::Assistant { text, tool_calls } => {
                let (reasoning, visible) = split_reasoning_envelope(text);
                if !reasoning.is_empty() {
                    out.push_str(MESSAGE_MODEL);
                    out.push_str(CONTENT_THINKING);
                    out.push_content(reasoning);
                    out.push_str(END_MESSAGE);
                }
                // Empty text is dropped only when the turn says something else (a reasoning
                // block or a call); a bare empty reply still renders its empty text message.
                if !visible.is_empty() || (tool_calls.is_empty() && reasoning.is_empty()) {
                    push_text_message(&mut out, MESSAGE_MODEL, visible);
                }
                for call in tool_calls.iter() {
                    let Value::Object(args) = call.arguments else {
                        bail!(
                            "tool call arguments of {:?} must be an object, not a JSON string or \
                             another value",
                            call.name
                        );
                    };
                    out.push_str(MESSAGE_MODEL);
                    out.push_content(call.name);
                    out.push_str(CONTENT_INVOKE_TOOL_JSON);
                    let mut body = String::from("{\"name\":");
                    write_json_string(&mut body, call.name);
                    body.push_str(",\"args\":{");
                    // The arguments' own keys come out sorted; what is inside a value keeps the
                    // order the client sent.
                    let mut entries: Vec<_> = args.iter().collect();
                    entries.sort_by(|a, b| a.0.cmp(b.0));
                    for (i, (key, value)) in entries.into_iter().enumerate() {
                        if i > 0 {
                            body.push(',');
                        }
                        write_json_string(&mut body, key);
                        body.push(':');
                        write_json(&mut body, value, false);
                    }
                    body.push_str("}}");
                    out.push_content(&body);
                    out.push_str(END_MESSAGE);
                }
                out.push_str(END_SAMPLING);
            }
            ChatTurn::Tool {
                tool_call_id,
                name,
                content,
            } => {
                let push_header = |out: &mut PromptText| {
                    out.push_str(MESSAGE_TOOL);
                    match name.filter(|n| !n.is_empty()) {
                        Some(name) => out.push_content(name),
                        None if !tool_call_id.is_empty() => {
                            // The name of the call this result answers, found among all the
                            // assistant turns' calls (a repeated id names it once per match).
                            for call in turns
                                .iter()
                                .filter_map(|t| match t {
                                    ChatTurn::Assistant { tool_calls, .. } => {
                                        Some(tool_calls.iter())
                                    }
                                    _ => None,
                                })
                                .flatten()
                            {
                                if call.id == *tool_call_id {
                                    out.push_content(call.name);
                                }
                            }
                        }
                        None => {}
                    }
                };
                for &blocks in attached {
                    push_header(&mut out);
                    push_image_body(&mut out, blocks);
                }
                if !content.is_empty() || attached.is_empty() {
                    push_header(&mut out);
                    out.push_str(CONTENT_TEXT);
                    out.push_content(content);
                    out.push_str(END_MESSAGE);
                }
            }
        }
    }
    if !effort_emitted {
        push_effort(&mut out, effort);
    }
    Ok(out)
}

/// `<role><|content_image|>` + one image token per block + `<|end_message|>`.
/// Tokens an image adds to a prompt besides its blocks: the role marker, `<|content_image|>` and
/// `<|end_message|>` of the message the image is.
pub const IMAGE_MESSAGE_FRAME_TOKENS: usize = 3;

/// Tokens an encoded image adds to a prompt, from its header alone.
pub fn image_message_tokens(encoded: &[u8]) -> Result<usize> {
    Ok(super::pixels::token_count(encoded)? + IMAGE_MESSAGE_FRAME_TOKENS)
}

fn push_image_message(out: &mut PromptText, role: &str, blocks: usize) {
    out.push_str(role);
    push_image_body(out, blocks);
}

fn push_image_body(out: &mut PromptText, blocks: usize) {
    out.push_str(CONTENT_IMAGE);
    for _ in 0..blocks {
        out.push_str(IMAGE_TOKEN);
    }
    out.push_str(END_MESSAGE);
}

fn push_text_message(out: &mut PromptText, role: &str, text: &str) {
    out.push_str(role);
    out.push_str(CONTENT_TEXT);
    out.push_content(text);
    out.push_str(END_MESSAGE);
}

fn push_effort(out: &mut PromptText, effort: Effort) {
    out.push_str(MESSAGE_SYSTEM);
    out.push_str(CONTENT_TEXT);
    out.push_str("Thinking effort level: ");
    out.push_str(&effort.text());
    out.push_str(END_MESSAGE);
}

/// The `tool_declare` payload: each tool as `{description, name, parameters, type}`, the whole
/// array with its keys sorted at every depth.
fn declared_tools_json(tools: &[ToolDef<'_>]) -> String {
    let specs: Vec<Value> = tools
        .iter()
        .map(|tool| {
            let parameters = match tool.parameters {
                Some(Value::Object(o)) if !o.is_empty() => Value::Object(o.clone()),
                Some(Value::Null) | None => Value::Object(Default::default()),
                Some(other) => other.clone(),
            };
            serde_json::json!({
                "description": tool.description.unwrap_or(""),
                "name": tool.name,
                "parameters": parameters,
                "type": "function",
            })
        })
        .collect();
    let mut out = String::new();
    write_json(&mut out, &Value::Array(specs), true);
    out
}

// ---- JSON as Python's `json.dumps(ensure_ascii=False, separators=(",", ":"))` -----------------

/// Compact JSON the way Python writes it, which is what the template's `tojson` filter does:
/// non-ASCII text kept, control characters as `\u00xx`, floats as `repr(float)`. With
/// `sort_keys`, the keys of every object are sorted.
pub(crate) fn write_json(out: &mut String, value: &Value, sort_keys: bool) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.push_str(&i.to_string());
            } else if let Some(u) = n.as_u64() {
                out.push_str(&u.to_string());
            } else if let Some(f) = n.as_f64() {
                out.push_str(&python_float_repr(f));
            }
        }
        Value::String(s) => write_json_string(out, s),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_json(out, item, sort_keys);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            let mut entries: Vec<_> = map.iter().collect();
            if sort_keys {
                entries.sort_by(|a, b| a.0.cmp(b.0));
            }
            for (i, (key, item)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_json_string(out, key);
                out.push(':');
                write_json(out, item, sort_keys);
            }
            out.push('}');
        }
    }
}

fn write_json_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python's `repr(float)` for a finite number: the shortest digits that read back, in fixed
/// notation for decimal exponents from -4 to 15 and otherwise as `d.ddde±XX`.
pub(crate) fn python_float_repr(x: f64) -> String {
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.to_string();
    }
    // Rust's `{:e}` prints the shortest round-trip digits: "1.2345e-7", "1e16", "-5e-1".
    let sci = format!("{:e}", x.abs());
    let (mantissa, exponent) = sci.split_once('e').expect("LowerExp has an exponent");
    let exp10: i32 = exponent.parse().expect("exponent is an integer");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let sign = if x < 0.0 { "-" } else { "" };
    if (-4..16).contains(&exp10) {
        let decimal_point = exp10 + 1; // digits before the point
        let body = if decimal_point <= 0 {
            format!("0.{}{}", "0".repeat((-decimal_point) as usize), digits)
        } else if decimal_point as usize >= digits.len() {
            format!(
                "{}{}.0",
                digits,
                "0".repeat(decimal_point as usize - digits.len())
            )
        } else {
            format!(
                "{}.{}",
                &digits[..decimal_point as usize],
                &digits[decimal_point as usize..]
            )
        };
        format!("{sign}{body}")
    } else {
        let mantissa = if digits.len() > 1 {
            format!("{}.{}", &digits[..1], &digits[1..])
        } else {
            digits
        };
        let exp_sign = if exp10 < 0 { '-' } else { '+' };
        format!("{sign}{mantissa}e{exp_sign}{:02}", exp10.abs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_io::AssistantToolCall;
    use serde_json::json;

    const GOLDEN: &str = include_str!("../../tests/fixtures/inkling2_chat/golden.json");

    /// A golden conversation, owned so the borrowed turns can point into it.
    struct Conversation {
        messages: Vec<Value>,
        calls: Vec<Vec<(String, String, Value)>>,
        tools: Vec<Value>,
        effort: Effort,
        /// Per message, one entry (a single block) per image part of its content.
        images: Vec<Vec<usize>>,
    }

    fn is_image(part: &Value) -> bool {
        matches!(
            part["type"].as_str(),
            Some("image" | "input_image" | "image_url")
        )
    }

    /// The text of a message: its string content, or its one text part when the content is a
    /// list (the server flattens a list to the text and the images the same way).
    fn text_of(msg: &Value) -> String {
        match &msg["content"] {
            Value::String(s) => s.clone(),
            Value::Array(parts) => {
                let texts: Vec<&str> = parts
                    .iter()
                    .filter(|p| !is_image(p))
                    .filter_map(|p| p["text"].as_str())
                    .collect();
                assert!(texts.len() <= 1, "golden cases carry at most one text part");
                texts.concat()
            }
            _ => String::new(),
        }
    }

    impl Conversation {
        fn from_case(case: &Value) -> Self {
            let messages: Vec<Value> = case["messages"].as_array().unwrap().clone();
            let calls = messages
                .iter()
                .map(|m| {
                    m["tool_calls"]
                        .as_array()
                        .map(|calls| {
                            calls
                                .iter()
                                .map(|c| {
                                    (
                                        c["id"].as_str().unwrap().to_string(),
                                        c["function"]["name"].as_str().unwrap().to_string(),
                                        c["function"]["arguments"].clone(),
                                    )
                                })
                                .collect()
                        })
                        .unwrap_or_default()
                })
                .collect();
            let tools = case["tools"].as_array().cloned().unwrap_or_default();
            let effort = match &case["effort"] {
                Value::String(s) => Effort::parse(s).expect("a vocabulary effort"),
                Value::Number(n) => Effort::new(n.as_f64().unwrap()).unwrap(),
                _ => DEFAULT_EFFORT,
            };
            let images = messages
                .iter()
                .map(|m| match &m["content"] {
                    Value::Array(parts) => vec![1; parts.iter().filter(|p| is_image(p)).count()],
                    _ => Vec::new(),
                })
                .collect();
            Self {
                messages,
                calls,
                tools,
                effort,
                images,
            }
        }

        /// The envelope a returned `reasoning_content` is folded into before rendering.
        fn assistant_texts(&self) -> Vec<String> {
            self.messages
                .iter()
                .map(|m| match m["reasoning_content"].as_str() {
                    Some(r) => crate::chat_io::join_reasoning_envelope(r, &text_of(m)),
                    None => text_of(m),
                })
                .collect()
        }

        fn render(&self) -> Result<PromptText> {
            self.render_with_blocks(&self.images)
        }

        fn render_with_blocks(&self, images: &[Vec<usize>]) -> Result<PromptText> {
            let texts = self.assistant_texts();
            let call_refs: Vec<Vec<AssistantToolCall<'_>>> = self
                .calls
                .iter()
                .map(|cs| {
                    cs.iter()
                        .map(|(id, name, args)| AssistantToolCall {
                            id,
                            name,
                            arguments: args,
                        })
                        .collect()
                })
                .collect();
            let turns: Vec<ChatTurn<'_>> = self
                .messages
                .iter()
                .enumerate()
                .map(|(i, m)| match m["role"].as_str().unwrap() {
                    "system" => ChatTurn::System(&texts[i]),
                    "user" => ChatTurn::User(&texts[i]),
                    "assistant" => ChatTurn::Assistant {
                        text: &texts[i],
                        tool_calls: &call_refs[i],
                    },
                    "tool" => ChatTurn::Tool {
                        tool_call_id: m["tool_call_id"].as_str().unwrap(),
                        name: m["name"].as_str(),
                        content: &texts[i],
                    },
                    other => panic!("role {other}"),
                })
                .collect();
            let tools: Vec<ToolDef<'_>> = self
                .tools
                .iter()
                .map(|t| {
                    let f = t.get("function").unwrap_or(t);
                    ToolDef {
                        name: f["name"].as_str().unwrap(),
                        description: f["description"].as_str(),
                        parameters: f.get("parameters"),
                        response: None,
                    }
                })
                .collect();
            render_prompt_with_images(&turns, &tools, self.effort, images)
        }
    }

    fn cases() -> Vec<Value> {
        let golden: Value = serde_json::from_str(GOLDEN).expect("golden.json");
        golden["cases"].as_array().expect("cases").clone()
    }

    #[test]
    fn every_golden_prompt_is_what_the_template_renders() {
        let cases = cases();
        assert!(cases.len() >= 30, "{} golden cases", cases.len());
        for case in &cases {
            let name = case["name"].as_str().unwrap();
            let got = Conversation::from_case(case)
                .render()
                .unwrap_or_else(|e| panic!("{name}: {e:#}"));
            assert_eq!(
                got.as_str(),
                case["text"].as_str().unwrap(),
                "case {name:?}"
            );
        }
    }

    /// The same prompts as token ids, against the real tokenizer: what HF's `apply_chat_template`
    /// encodes is what [`TextTokenizer::encode_prompt`] encodes, for every case whose client text
    /// spells no control token.
    #[test]
    #[ignore = "needs the real tokenizer.json; set LUMEN_INKLING2_MODEL_DIR to the checkpoint directory"]
    fn every_golden_prompt_encodes_to_the_ids_hf_gives() {
        let dir = std::env::var("LUMEN_INKLING2_MODEL_DIR").expect("LUMEN_INKLING2_MODEL_DIR");
        let tok = crate::text_tokenizer::TextTokenizer::from_file(
            std::path::Path::new(&dir).join("tokenizer.json"),
        )
        .expect("tokenizer.json");
        let mut checked = 0;
        for case in &cases() {
            let name = case["name"].as_str().unwrap();
            let Some(want) = case["token_ids"].as_array() else {
                continue;
            };
            let prompt = Conversation::from_case(case).render().unwrap();
            let got = tok.encode_prompt(&prompt).unwrap();
            let want: Vec<u32> = want.iter().map(|v| v.as_u64().unwrap() as u32).collect();
            assert_eq!(got, want, "case {name:?}");
            checked += 1;
        }
        assert!(checked >= 28, "{checked} cases had ids");
    }

    /// The processor writes each image's token once per block: the golden text (one token per
    /// image) with the k-th token expanded to `blocks[k]` copies is the renderer's own prompt.
    #[test]
    fn an_image_is_one_token_per_block_as_the_processor_expands_it() {
        for name in [
            "two images then text",
            "an image in a tool result",
            "an image in a later turn",
        ] {
            let case = cases().into_iter().find(|c| c["name"] == name).unwrap();
            let conversation = Conversation::from_case(&case);
            let wanted: Vec<usize> = (0..conversation.images.iter().map(Vec::len).sum::<usize>())
                .map(|k| 3 + 2 * k)
                .collect();
            let mut next = wanted.iter();
            let blocks: Vec<Vec<usize>> = conversation
                .images
                .iter()
                .map(|per| per.iter().map(|_| *next.next().unwrap()).collect())
                .collect();
            let golden = case["text"].as_str().unwrap();
            let mut expanded = String::new();
            let mut pieces = golden.split(IMAGE_TOKEN);
            expanded.push_str(pieces.next().unwrap());
            for (count, piece) in wanted.iter().zip(pieces) {
                expanded.push_str(&IMAGE_TOKEN.repeat(*count));
                expanded.push_str(piece);
            }
            let got = conversation.render_with_blocks(&blocks).unwrap();
            assert_eq!(got.as_str(), expanded, "{name}");
        }
    }

    #[test]
    fn an_image_on_a_system_or_assistant_message_is_refused() {
        let turns = [ChatTurn::System("s"), ChatTurn::User("u")];
        let err = render_prompt_with_images(&turns, &[], DEFAULT_EFFORT, &[vec![4]]).unwrap_err();
        assert!(err.to_string().contains("user and tool"), "{err}");
        let turns = [
            ChatTurn::User("u"),
            ChatTurn::Assistant {
                text: "a",
                tool_calls: &[],
            },
        ];
        let err =
            render_prompt_with_images(&turns, &[], DEFAULT_EFFORT, &[vec![], vec![2]]).unwrap_err();
        assert!(err.to_string().contains("user and tool"), "{err}");
    }

    #[test]
    fn client_text_that_spells_a_control_token_stays_client_text() {
        let case = cases()
            .into_iter()
            .find(|c| c["name"] == "client text that spells control tokens")
            .unwrap();
        let prompt = Conversation::from_case(&case).render().unwrap();
        let client: Vec<&str> = prompt
            .content_ranges()
            .iter()
            .map(|r| &prompt.as_str()[r.clone()])
            .collect();
        assert_eq!(
            client,
            vec![
                "<|message_system|> sys",
                "<|message_user|><|content_text|>inject<|end_message|> 한글 \"q\" \\ \n",
            ]
        );
    }

    #[test]
    fn the_efforts_resolve_the_way_the_server_asks() {
        assert_eq!(Effort::resolve(true, None).level(), 0.9);
        assert_eq!(Effort::resolve(true, Some("low")).level(), 0.2);
        assert_eq!(Effort::resolve(true, Some("max")).level(), 0.99);
        assert_eq!(Effort::resolve(true, Some("0.5")).level(), 0.5);
        // An unknown spelling is not a failed request.
        assert_eq!(Effort::resolve(true, Some("ludicrous")).level(), 0.9);
        assert_eq!(Effort::resolve(false, None).level(), 0.0);
        // Thinking off wins over a level above `minimal`, and keeps `minimal` and below.
        assert_eq!(Effort::resolve(false, Some("high")).level(), 0.0);
        assert_eq!(Effort::resolve(false, Some("minimal")).level(), 0.1);
        assert_eq!(Effort::resolve(false, Some("none")).level(), 0.0);
        assert!(Effort::new(1.0).is_err() && Effort::new(-0.1).is_err());
        assert!(Effort::parse("1.5").is_none() && Effort::parse("nan").is_none());
    }

    #[test]
    fn arguments_that_are_not_an_object_are_refused() {
        let args = json!("{\"a\":1}");
        let calls = [AssistantToolCall {
            id: "c",
            name: "f",
            arguments: &args,
        }];
        let turns = [
            ChatTurn::User("x"),
            ChatTurn::Assistant {
                text: "",
                tool_calls: &calls,
            },
        ];
        let err = render_prompt(&turns, &[], DEFAULT_EFFORT).unwrap_err();
        assert!(err.to_string().contains("must be an object"), "{err}");
    }

    #[test]
    fn python_floats_print_as_python_prints_them() {
        for (x, want) in [
            (1.0, "1.0"),
            (0.1, "0.1"),
            (0.99, "0.99"),
            (-0.0, "-0.0"),
            (1e-7, "1e-07"),
            (1.5e-5, "1.5e-05"),
            (0.0001, "0.0001"),
            (0.00012345, "0.00012345"),
            (1e16, "1e+16"),
            (1.2345e20, "1.2345e+20"),
            (1e15, "1000000000000000.0"),
            (12345678.5, "12345678.5"),
            (123456789012.0, "123456789012.0"),
            (-2.5e-10, "-2.5e-10"),
        ] {
            assert_eq!(python_float_repr(x), want, "{x:e}");
        }
    }

    #[test]
    fn json_strings_escape_as_python_escapes_them() {
        let mut out = String::new();
        write_json(
            &mut out,
            &json!({"s": "a<b>&'\"\\\n\r\t\u{8}\u{c}\u{1}\u{7f}é한"}),
            false,
        );
        assert_eq!(
            out,
            "{\"s\":\"a<b>&'\\\"\\\\\\n\\r\\t\\b\\f\\u0001\u{7f}é한\"}"
        );
    }
}
