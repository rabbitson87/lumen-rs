use atomic_http::external::http::{Request, Response, StatusCode};
use atomic_http::*;
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc::error::TryRecvError;

use crate::engine::{EngineHandle, StreamEvent};
use crate::types::{AnthropicError, AnthropicRequest};

pub async fn handle(
    request: Request<ArenaBody>,
    mut response: Response<ArenaWriter>,
    handle: EngineHandle,
) -> Result<(), SendableError> {
    let req: AnthropicRequest = match request.get_json_arena() {
        Ok(r) => r,
        Err(e) => {
            let err = AnthropicError::new(format!("invalid request: {e}"));
            response.body_mut().set_arena_json(&err)?;
            *response.status_mut() = StatusCode::from_u16(400)?;
            response.responser_arena().await?;
            return Ok(());
        }
    };

    if req.stream {
        return handle_streaming(req, response, handle).await;
    }

    match handle.anthropic_messages(req).await {
        Ok(resp) => {
            response.body_mut().set_arena_json(&resp)?;
            *response.status_mut() = StatusCode::from_u16(200)?;
        }
        Err(e) => {
            let err = AnthropicError::new(crate::types::inference_error_message(&e));
            response.body_mut().set_arena_json(&err)?;
            *response.status_mut() = StatusCode::from_u16(500)?;
        }
    }

    response.responser_arena().await?;
    Ok(())
}

/// What `message_start` should report as `input_tokens`, and what is left over.
///
/// Anthropic names the prompt size exactly once, in the *first* event of the
/// stream — before a single token has been decoded. Our engine only learns the
/// figure at the same moment, so it sends it ahead of prefill as
/// [`StreamEvent::Start`]; everything else arrives later and cannot be used
/// here. The returned event is whatever showed up instead and must be handled
/// by the caller's loop rather than dropped.
///
/// Falling back to `0` is deliberate: a stream that never sends `Start` (an
/// error before decode, or a future backend that skips it) still gets a
/// well-formed `message_start` instead of hanging on an event that may never
/// come. `0` is what the field was hardcoded to for the whole life of this
/// route, so the fallback is exactly the old behaviour and nothing worse.
fn message_start_input_tokens(first: Option<StreamEvent>) -> (u32, Option<StreamEvent>) {
    match first {
        Some(StreamEvent::Start { prompt_tokens }) => (prompt_tokens, None),
        other => (0, other),
    }
}

/// Anthropic SSE streaming format:
/// event: message_start → event: content_block_start → event: content_block_delta (×N)
/// → event: content_block_stop → event: message_delta → event: message_stop
async fn handle_streaming(
    req: AnthropicRequest,
    response: Response<ArenaWriter>,
    handle: EngineHandle,
) -> Result<(), SendableError> {
    let model = req.model.clone();
    // Read before the request moves into the engine. `AnthropicRequest`'s
    // thinking flag is the client's own — unlike the OpenAI one it does not
    // fall through to a backend default — so this is exactly "did the caller
    // ask for extended thinking?", which is the spec's condition for a
    // `thinking` block appearing in the response at all.
    let emit_thinking = req.enable_thinking();
    let mut tcp = response.into_body().stream;

    tcp.write_all(
        b"HTTP/1.1 200 OK\r\n\
          Content-Type: text/event-stream\r\n\
          Cache-Control: no-cache\r\n\
          Connection: keep-alive\r\n\
          \r\n",
    )
    .await?;

    let mut token_rx = match handle.anthropic_messages_streaming(req).await {
        Ok(rx) => rx,
        Err(e) => {
            let msg = format!(
                "event: error\ndata: {{\"type\":\"error\",\"error\":{{\"type\":\"server_error\",\"message\":\"{e}\"}}}}\n\n"
            );
            tcp.write_all(msg.as_bytes()).await?;
            tcp.flush().await?;
            return Ok(());
        }
    };

    let msg_id = format!(
        "msg_{:x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    );

    // message_start — the one place Anthropic's format has for `input_tokens`,
    // and it goes out before the first token. The engine sends `Start` as soon
    // as it has counted the prompt (ahead of prefill), so waiting for it here
    // costs a channel hop, not a generation.
    //
    // It used to be hardcoded `0` with a comment claiming the real figure was
    // "surfaced in message_start above" — it was not, and an Anthropic SDK
    // accumulating usage across the stream reported 0 input tokens for a
    // 289-token tool prompt.
    //
    // Anything other than `Start` first (an immediate `Error`, or a backend
    // that never learned to send it) is carried into the loop below and
    // `message_start` goes out with `0` as before, rather than blocking the
    // stream on an event that may never come.
    let (input_tokens, mut carry) = message_start_input_tokens(token_rx.recv().await);
    let start = format!(
        "event: message_start\ndata: {{\"type\":\"message_start\",\"message\":{{\"id\":\"{msg_id}\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"{model}\",\"content\":[],\"stop_reason\":null,\"usage\":{{\"input_tokens\":{input_tokens},\"output_tokens\":1}}}}}}\n\n"
    );
    tcp.write_all(start.as_bytes()).await?;

    // Phase 1.5: content_block_start is LAZY — we don't pre-emit it because
    // tool-only responses have no leading text. `BlockStream` owns which block
    // is open and what index it got; the delta prefix is rebuilt whenever that
    // index changes, because a `thinking` block ahead of the text moves every
    // later index by one and a delta pinned to `index:0` would land on the
    // wrong block while still parsing as valid JSON.
    let delta_suffix: &[u8] = b"}}\n\n";
    let mut delta_prefix: Vec<u8> = Vec::with_capacity(128);
    let mut buf: Vec<u8> = Vec::with_capacity(384);
    let mut blocks = BlockStream::new();

    // Phase 1.6: Anthropic SSE `ping` keepalive. Matches real Claude
    // wire — when prefill is long-running (≥1s with no decoded token
    // yet), some proxies / clients drop the SSE connection. Periodic
    // `event: ping` keeps the channel alive. Real Claude emits these
    // occasionally during long generations as well; we mirror the
    // pattern. Static bytes — no allocation per send.
    let ping_event: &[u8] = b"event: ping\ndata: {\"type\":\"ping\"}\n\n";

    // `carry` is already live from the `message_start` peek above — anything
    // that arrived instead of `Start` is still unhandled and enters here.
    loop {
        let event_opt: Option<StreamEvent> = match carry.take() {
            Some(e) => Some(e),
            None => loop {
                tokio::select! {
                    res = token_rx.recv() => break res,
                    _ = tokio::time::sleep(std::time::Duration::from_millis(1000)) => {
                        tcp.write_all(ping_event).await?;
                        tcp.flush().await?;
                    }
                }
            },
        };
        let event = match event_opt {
            Some(e) => e,
            None => break,
        };
        match event {
            // Already consumed above; `message_start` is on the wire, so a
            // second one has nowhere to go.
            StreamEvent::Start { .. } => {}
            StreamEvent::Delta(text) => {
                buf.clear();
                let idx = blocks.open_text(&mut buf);
                set_text_delta_prefix(&mut delta_prefix, idx);
                append_anthropic_delta(&mut buf, &delta_prefix, delta_suffix, &text);
                loop {
                    match token_rx.try_recv() {
                        Ok(StreamEvent::Delta(more)) => {
                            append_anthropic_delta(&mut buf, &delta_prefix, delta_suffix, &more);
                        }
                        Ok(other) => {
                            carry = Some(other);
                            break;
                        }
                        Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => break,
                    }
                }
                tcp.write_all(&buf).await?;
            }
            StreamEvent::ReasoningDelta(text) => {
                // The Messages API's own channel for the trace. This used to be
                // discarded outright, so an Anthropic client could not see the
                // reasoning and — worse — could not hand it back on the next
                // turn, which is what a `thinking`-enabled conversation needs to
                // extend its KV instead of re-prefilling.
                //
                // Gated on the request's own flag: a client that did not enable
                // thinking must not start receiving a block type it never asked
                // for.
                if !emit_thinking {
                    continue;
                }
                buf.clear();
                let idx = blocks.open_thinking(&mut buf);
                append_thinking_delta(&mut buf, idx, &text);
                loop {
                    match token_rx.try_recv() {
                        Ok(StreamEvent::ReasoningDelta(more)) => {
                            append_thinking_delta(&mut buf, idx, &more);
                        }
                        Ok(other) => {
                            carry = Some(other);
                            break;
                        }
                        Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => break,
                    }
                }
                tcp.write_all(&buf).await?;
            }
            StreamEvent::ToolCallStart { id, name, .. } => {
                buf.clear();
                blocks.open_tool(&mut buf, &id, &name);
                tcp.write_all(&buf).await?;
            }
            StreamEvent::ToolCallArgumentsDelta { partial_json, .. } => {
                let idx = blocks.current_tool_index();
                // partial_json is the raw JSON object string; double-encode it
                // into the wire envelope's `partial_json` string field.
                let pj = serde_json::to_string(&partial_json)
                    .map_err(|e| SendableError::from(e.to_string()))?;
                let evt = format!(
                    "event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"index\":{idx},\"delta\":{{\"type\":\"input_json_delta\",\"partial_json\":{pj}}}}}\n\n"
                );
                tcp.write_all(evt.as_bytes()).await?;
            }
            StreamEvent::ToolCallStop { .. } => {
                buf.clear();
                blocks.close_open(&mut buf);
                tcp.write_all(&buf).await?;
            }
            StreamEvent::Done {
                prompt_tokens,
                completion_tokens: output_tokens,
                finish_reason,
            } => {
                // Close whatever is still open, and — if the turn produced no
                // text and no tool calls — add the pro-forma empty text block
                // so downstream parsers never see a `content[]` that is
                // effectively empty. A trace on its own does not count as an
                // answer, so a thinking-only turn still gets one.
                buf.clear();
                blocks.finish(&mut buf);
                tcp.write_all(&buf).await?;

                let stop_reason = finish_reason.anthropic_str();
                let delta = format!(
                    "event: message_delta\ndata: {{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":\"{stop_reason}\",\"stop_sequence\":null}},\"usage\":{{\"output_tokens\":{output_tokens}}}}}\n\n"
                );
                tcp.write_all(delta.as_bytes()).await?;

                tcp.write_all(b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
                    .await?;
                // Anthropic's format reports `input_tokens` once, in
                // `message_start`, and that is where it now goes — from the
                // `Start` event, not from here. `Done` carries the same figure
                // for the OpenAI route's benefit; on this path it is redundant.
                debug_assert!(
                    carry.is_some() || input_tokens == prompt_tokens,
                    "message_start reported {input_tokens} input tokens, Done says {prompt_tokens}",
                );
                break;
            }
            // Relay it as an Anthropic `error` event rather than dropping it —
            // a silently truncated stream gives the client no way to tell a
            // rejected request from an empty answer.
            StreamEvent::Error(msg) => {
                let mut buf = b"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":".to_vec();
                serde_json::to_writer(&mut buf, &msg).expect("write to Vec cannot fail");
                buf.extend_from_slice(b"}}\n\n");
                tcp.write_all(&buf).await?;
                break;
            }
        }
    }

    tcp.flush().await?;
    Ok(())
}

fn append_anthropic_delta(buf: &mut Vec<u8>, prefix: &[u8], suffix: &[u8], text: &str) {
    buf.extend_from_slice(prefix);
    serde_json::to_writer(&mut *buf, text).expect("write to Vec cannot fail");
    buf.extend_from_slice(suffix);
}

/// Rebuild the `text_delta` envelope for `index`. Cheap, and only the index
/// varies — but it does vary now, so it cannot stay a `const`.
fn set_text_delta_prefix(prefix: &mut Vec<u8>, index: u32) {
    prefix.clear();
    prefix.extend_from_slice(
        format!(
            "event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"index\":{index},\"delta\":{{\"type\":\"text_delta\",\"text\":"
        )
        .as_bytes(),
    );
}

fn append_thinking_delta(buf: &mut Vec<u8>, index: u32, text: &str) {
    buf.extend_from_slice(
        format!(
            "event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"index\":{index},\"delta\":{{\"type\":\"thinking_delta\",\"thinking\":"
        )
        .as_bytes(),
    );
    serde_json::to_writer(&mut *buf, text).expect("write to Vec cannot fail");
    buf.extend_from_slice(b"}}\n\n");
}

/// Which block, if any, is currently open.
#[derive(Debug, PartialEq)]
enum OpenBlock {
    Thinking(u32),
    Text(u32),
    Tool(u32),
}

/// Assigns `content[]` indices and emits the block-lifecycle SSE frames.
///
/// Anthropic's streaming format identifies every delta by an index into the
/// final `content[]` array, and the order is fixed: `thinking` first, then
/// `text`, then `tool_use`. Before thinking blocks existed the loop could
/// hardcode `index:0` for text and start tools at 1; with a block that may or
/// may not precede them, every index becomes conditional — which is exactly the
/// kind of bookkeeping that breaks a client silently, since a misindexed delta
/// is still well-formed JSON.
///
/// So it lives here, writing frames into a caller-owned buffer (which keeps the
/// loop's delta batching) and answerable by a test that reads the frames back.
struct BlockStream {
    next_index: u32,
    open: Option<OpenBlock>,
    /// Whether any block other than `thinking` was ever opened. A trace on its
    /// own is not an answer, so this decides the pro-forma empty text block.
    emitted_answer: bool,
}

impl BlockStream {
    fn new() -> Self {
        Self {
            next_index: 0,
            open: None,
            emitted_answer: false,
        }
    }

    fn close_open(&mut self, out: &mut Vec<u8>) {
        let Some(open) = self.open.take() else { return };
        let idx = match open {
            OpenBlock::Thinking(i) => {
                // Anthropic closes a thinking block with the signature it will
                // expect back on a replay. Ours is a constant — see
                // `LUMEN_THINKING_SIGNATURE` — but the frame has to be there or
                // a client assembling the block finds no signature at all.
                let sig = serde_json::to_string(crate::types::LUMEN_THINKING_SIGNATURE)
                    .expect("a str always serializes");
                out.extend_from_slice(
                    format!(
                        "event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"index\":{i},\"delta\":{{\"type\":\"signature_delta\",\"signature\":{sig}}}}}\n\n"
                    )
                    .as_bytes(),
                );
                i
            }
            OpenBlock::Text(i) | OpenBlock::Tool(i) => i,
        };
        out.extend_from_slice(
            format!(
                "event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":{idx}}}\n\n"
            )
            .as_bytes(),
        );
        self.next_index = idx.saturating_add(1);
    }

    /// Open a thinking block if one is not already open. Returns its index.
    fn open_thinking(&mut self, out: &mut Vec<u8>) -> u32 {
        if let Some(OpenBlock::Thinking(i)) = self.open {
            return i;
        }
        self.close_open(out);
        let idx = self.next_index;
        out.extend_from_slice(
            format!(
                "event: content_block_start\ndata: {{\"type\":\"content_block_start\",\"index\":{idx},\"content_block\":{{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}}}\n\n"
            )
            .as_bytes(),
        );
        self.open = Some(OpenBlock::Thinking(idx));
        idx
    }

    /// Open a text block if one is not already open. Returns its index.
    fn open_text(&mut self, out: &mut Vec<u8>) -> u32 {
        if let Some(OpenBlock::Text(i)) = self.open {
            return i;
        }
        self.close_open(out);
        let idx = self.next_index;
        out.extend_from_slice(
            format!(
                "event: content_block_start\ndata: {{\"type\":\"content_block_start\",\"index\":{idx},\"content_block\":{{\"type\":\"text\",\"text\":\"\"}}}}\n\n"
            )
            .as_bytes(),
        );
        self.open = Some(OpenBlock::Text(idx));
        self.emitted_answer = true;
        idx
    }

    fn open_tool(&mut self, out: &mut Vec<u8>, id: &str, name: &str) -> u32 {
        self.close_open(out);
        let idx = self.next_index;
        let id_json = serde_json::to_string(id).expect("a str always serializes");
        let name_json = serde_json::to_string(name).expect("a str always serializes");
        out.extend_from_slice(
            format!(
                "event: content_block_start\ndata: {{\"type\":\"content_block_start\",\"index\":{idx},\"content_block\":{{\"type\":\"tool_use\",\"id\":{id_json},\"name\":{name_json},\"input\":{{}}}}}}\n\n"
            )
            .as_bytes(),
        );
        self.open = Some(OpenBlock::Tool(idx));
        self.emitted_answer = true;
        idx
    }

    /// Index a tool-argument delta belongs to. Falls back the way the previous
    /// code did — to whatever index is next — when no tool block is open.
    fn current_tool_index(&self) -> u32 {
        match self.open {
            Some(OpenBlock::Tool(i)) => i,
            _ => self.next_index,
        }
    }

    /// Close whatever is open and, if nothing but a thinking block was ever
    /// emitted, add the pro-forma empty text block so a client reading
    /// `content[].text` still finds one.
    fn finish(&mut self, out: &mut Vec<u8>) {
        let needs_placeholder = !self.emitted_answer;
        self.close_open(out);
        if needs_placeholder {
            self.open_text(out);
            self.close_open(out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PREFIX: &[u8] = b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":";
    const SUFFIX: &[u8] = b"}}\n\n";

    #[test]
    fn anthropic_delta_byte_equivalent_to_legacy_format() {
        let text = "Hello \"world\" with \\backslash and 한글";
        let mut buf = Vec::new();
        append_anthropic_delta(&mut buf, PREFIX, SUFFIX, text);

        let escaped = serde_json::to_string(text).unwrap();
        let legacy = format!(
            "event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{{\"type\":\"text_delta\",\"text\":{escaped}}}}}\n\n"
        );
        assert_eq!(std::str::from_utf8(&buf).unwrap(), legacy);
    }

    #[test]
    fn anthropic_delta_parses_as_json() {
        let mut buf = Vec::new();
        append_anthropic_delta(&mut buf, PREFIX, SUFFIX, "tab:\tnewline:\n\"quoted\"");
        let s = std::str::from_utf8(&buf).unwrap();
        let json_part = s
            .strip_prefix("event: content_block_delta\ndata: ")
            .unwrap()
            .trim_end_matches("\n\n");
        let v: serde_json::Value = serde_json::from_str(json_part).unwrap();
        assert_eq!(v["delta"]["text"], "tab:\tnewline:\n\"quoted\"");
    }

    /// `message_start` must carry the real prompt size.
    ///
    /// It was hardcoded `0`, under a comment in the `Done` arm claiming the
    /// figure was "surfaced in message_start above" — it never was. Measured on
    /// Qwen3.8-27B: a 289-token tool prompt streamed to an Anthropic client
    /// that accumulates usage across the stream, and the client's final message
    /// said 0 input tokens. Unlike OpenAI, there is no later event to correct
    /// it: the format states `input_tokens` once, first.
    #[test]
    fn message_start_reports_the_prompt_size_the_engine_measured() {
        let (n, carry) =
            message_start_input_tokens(Some(StreamEvent::Start { prompt_tokens: 289 }));
        assert_eq!(n, 289, "the count the engine sent has to reach the wire");
        assert!(carry.is_none(), "Start is consumed, not replayed");
    }

    /// Parse the SSE frames a `BlockStream` wrote into `(event, json)` pairs,
    /// so a test can assert on what a client would actually see rather than on
    /// a byte string.
    fn frames(buf: &[u8]) -> Vec<(String, serde_json::Value)> {
        std::str::from_utf8(buf)
            .expect("frames are utf-8")
            .split("\n\n")
            .filter(|f| !f.trim().is_empty())
            .map(|f| {
                let (ev, data) = f.split_once('\n').expect("event line then data line");
                let ev = ev
                    .strip_prefix("event: ")
                    .expect("event: prefix")
                    .to_string();
                let data = data.strip_prefix("data: ").expect("data: prefix");
                (ev, serde_json::from_str(data).expect("data is json"))
            })
            .collect()
    }

    /// Shape of each frame, as `type[:index]`, for comparing whole sequences.
    fn shape(buf: &[u8]) -> Vec<String> {
        frames(buf)
            .into_iter()
            .map(|(_, v)| {
                let kind = match v["type"].as_str().expect("typed frame") {
                    "content_block_start" => format!("start:{}", v["content_block"]["type"]),
                    "content_block_delta" => format!("delta:{}", v["delta"]["type"]),
                    other => other.to_string(),
                };
                format!("{kind}@{}", v["index"])
            })
            .collect()
    }

    /// The trace gets its own block at index 0, and everything after it shifts.
    ///
    /// Anthropic identifies every delta by its index into the final `content[]`,
    /// so a misindexed delta is still well-formed JSON that lands on the wrong
    /// block. Before this, text was pinned to `index:0` by a `const` prefix and
    /// tools started at 1 — correct only while nothing could precede them.
    #[test]
    fn a_thinking_block_takes_index_zero_and_pushes_the_rest_along() {
        let mut b = BlockStream::new();
        let mut out = Vec::new();
        let ti = b.open_thinking(&mut out);
        append_thinking_delta(&mut out, ti, "weighing it up");
        let xi = b.open_text(&mut out);
        let mut prefix = Vec::new();
        set_text_delta_prefix(&mut prefix, xi);
        append_anthropic_delta(&mut out, &prefix, SUFFIX, "Red");
        b.open_tool(&mut out, "toolu_1", "get_weather");
        b.finish(&mut out);

        assert_eq!(
            shape(&out),
            [
                "start:\"thinking\"@0",
                "delta:\"thinking_delta\"@0",
                // the signature closes the thinking block, per the API
                "delta:\"signature_delta\"@0",
                "content_block_stop@0",
                "start:\"text\"@1",
                "delta:\"text_delta\"@1",
                "content_block_stop@1",
                "start:\"tool_use\"@2",
                "content_block_stop@2",
            ]
        );
        let f = frames(&out);
        assert_eq!(f[1].1["delta"]["thinking"], "weighing it up");
        assert_eq!(
            f[2].1["delta"]["signature"],
            crate::types::LUMEN_THINKING_SIGNATURE
        );
        assert_eq!(f[5].1["delta"]["text"], "Red");
    }

    /// Without a trace the indices must be what they always were — this is the
    /// no-regression half, and it covers every request that does not enable
    /// thinking, which is most of them.
    #[test]
    fn without_a_trace_text_is_still_index_zero_and_tools_start_at_one() {
        let mut b = BlockStream::new();
        let mut out = Vec::new();
        let xi = b.open_text(&mut out);
        assert_eq!(xi, 0);
        let mut prefix = Vec::new();
        set_text_delta_prefix(&mut prefix, xi);
        append_anthropic_delta(&mut out, &prefix, SUFFIX, "hi");
        assert_eq!(b.open_tool(&mut out, "toolu_1", "f"), 1);
        b.close_open(&mut out);
        assert_eq!(b.open_tool(&mut out, "toolu_2", "g"), 2);
        b.finish(&mut out);

        assert_eq!(
            shape(&out),
            [
                "start:\"text\"@0",
                "delta:\"text_delta\"@0",
                "content_block_stop@0",
                "start:\"tool_use\"@1",
                "content_block_stop@1",
                "start:\"tool_use\"@2",
                "content_block_stop@2",
            ]
        );
        // And the text delta envelope is byte-identical to the old `const`.
        assert!(
            std::str::from_utf8(&out).unwrap().contains(
                "{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}"
            ),
            "the index-0 text delta must not have changed shape",
        );
    }

    /// A tool-only turn still puts the first tool at index 0.
    #[test]
    fn a_tool_only_turn_starts_at_index_zero() {
        let mut b = BlockStream::new();
        let mut out = Vec::new();
        assert_eq!(b.open_tool(&mut out, "toolu_1", "f"), 0);
        b.finish(&mut out);
        assert_eq!(
            shape(&out),
            ["start:\"tool_use\"@0", "content_block_stop@0"]
        );
    }

    /// A turn that produced only a trace is not an answer, so the pro-forma
    /// empty text block still goes out — after the thinking block, at index 1.
    #[test]
    fn a_thinking_only_turn_still_gets_its_placeholder_text_block() {
        let mut b = BlockStream::new();
        let mut out = Vec::new();
        let ti = b.open_thinking(&mut out);
        append_thinking_delta(&mut out, ti, "…");
        b.finish(&mut out);
        assert_eq!(
            shape(&out),
            [
                "start:\"thinking\"@0",
                "delta:\"thinking_delta\"@0",
                "delta:\"signature_delta\"@0",
                "content_block_stop@0",
                "start:\"text\"@1",
                "content_block_stop@1",
            ]
        );
    }

    /// An empty turn is unchanged: one empty text block at index 0.
    #[test]
    fn an_empty_turn_still_emits_one_empty_text_block() {
        let mut b = BlockStream::new();
        let mut out = Vec::new();
        b.finish(&mut out);
        assert_eq!(shape(&out), ["start:\"text\"@0", "content_block_stop@0"]);
    }

    /// A stream that opens with anything else still gets a well-formed
    /// `message_start`, and the event is handed back rather than swallowed —
    /// dropping it would lose an `Error` and truncate the stream in silence.
    #[test]
    fn a_stream_that_never_sends_start_still_opens_and_keeps_its_first_event() {
        let (n, carry) = message_start_input_tokens(Some(StreamEvent::Delta("hello".to_string())));
        assert_eq!(n, 0);
        assert!(
            matches!(carry, Some(StreamEvent::Delta(ref t)) if t == "hello"),
            "the first event must survive into the main loop",
        );

        let (n, carry) = message_start_input_tokens(None);
        assert_eq!(n, 0);
        assert!(carry.is_none());
    }
}
