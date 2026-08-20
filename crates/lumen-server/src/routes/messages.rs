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
            let err = AnthropicError::new(format!("inference error: {e}"));
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

    // Phase 1.5: content_block_start is now LAZY — we don't pre-emit it
    // because tool-only responses have no leading text. The first text
    // Delta opens a text block (index 0); the first ToolCallStart opens
    // a tool_use block at whichever wire index is next. The static
    // delta_prefix / delta_suffix below stays pinned to `index:0` since
    // visible text Deltas always precede tool_calls in our backends.
    let delta_prefix: &[u8] = b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":";
    let delta_suffix: &[u8] = b"}}\n\n";
    let mut buf: Vec<u8> = Vec::with_capacity(delta_prefix.len() + delta_suffix.len() + 256);

    let mut wire_index: u32 = 0;
    let mut text_block_open: bool = false;
    let mut current_tool_wire_index: Option<u32> = None;

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
                if !text_block_open {
                    tcp.write_all(b"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n").await?;
                    text_block_open = true;
                    // text block always claims index 0; tool blocks start at 1
                    wire_index = 1;
                }
                buf.clear();
                append_anthropic_delta(&mut buf, delta_prefix, delta_suffix, &text);
                loop {
                    match token_rx.try_recv() {
                        Ok(StreamEvent::Delta(more)) => {
                            append_anthropic_delta(&mut buf, delta_prefix, delta_suffix, &more);
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
            StreamEvent::ReasoningDelta(_text) => {
                // Anthropic Messages API has a dedicated `thinking`
                // content block type for extended reasoning, distinct
                // from the OpenAI `delta.reasoning` field. Proper
                // wiring (start `content_block_start` with
                // `{"type":"thinking"}`, emit `thinking_delta` chunks,
                // close with `content_block_stop`) is deferred — for
                // now we silently discard reasoning tokens on the
                // Anthropic SSE path so the model output still
                // terminates cleanly without leaking reasoning into
                // the visible text block. OpenAI clients still get
                // the full reasoning via `/v1/chat/completions`.
            }
            StreamEvent::ToolCallStart { id, name, .. } => {
                if text_block_open {
                    tcp.write_all(b"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n").await?;
                    text_block_open = false;
                }
                let idx = wire_index;
                let id_json =
                    serde_json::to_string(&id).map_err(|e| SendableError::from(e.to_string()))?;
                let name_json =
                    serde_json::to_string(&name).map_err(|e| SendableError::from(e.to_string()))?;
                let evt = format!(
                    "event: content_block_start\ndata: {{\"type\":\"content_block_start\",\"index\":{idx},\"content_block\":{{\"type\":\"tool_use\",\"id\":{id_json},\"name\":{name_json},\"input\":{{}}}}}}\n\n"
                );
                tcp.write_all(evt.as_bytes()).await?;
                current_tool_wire_index = Some(idx);
            }
            StreamEvent::ToolCallArgumentsDelta { partial_json, .. } => {
                let idx = current_tool_wire_index.unwrap_or(wire_index);
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
                if let Some(idx) = current_tool_wire_index.take() {
                    let evt = format!(
                        "event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":{idx}}}\n\n"
                    );
                    tcp.write_all(evt.as_bytes()).await?;
                    wire_index = idx.saturating_add(1);
                }
            }
            StreamEvent::Done {
                prompt_tokens,
                completion_tokens: output_tokens,
                finish_reason,
            } => {
                // Close any lingering text block.
                if text_block_open {
                    tcp.write_all(b"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n").await?;
                }
                // If the model produced nothing AND no tool calls, emit a
                // pro-forma empty text block so downstream parsers never see
                // a content[] array that's effectively empty.
                if wire_index == 0 && current_tool_wire_index.is_none() {
                    tcp.write_all(b"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n").await?;
                    tcp.write_all(b"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n").await?;
                }

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
