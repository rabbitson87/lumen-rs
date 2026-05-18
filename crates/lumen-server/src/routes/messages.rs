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

    // message_start
    let start = format!(
        "event: message_start\ndata: {{\"type\":\"message_start\",\"message\":{{\"id\":\"{msg_id}\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"{model}\",\"content\":[],\"stop_reason\":null,\"usage\":{{\"input_tokens\":0,\"output_tokens\":0}}}}}}\n\n"
    );
    tcp.write_all(start.as_bytes()).await?;

    // content_block_start
    tcp.write_all(b"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n").await?;

    // Hot-path delta envelope (Anthropic content_block_delta). Static prefix/suffix
    // bytes — only the content text is escaped and appended per token.
    let delta_prefix: &[u8] = b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":";
    let delta_suffix: &[u8] = b"}}\n\n";
    let mut buf: Vec<u8> = Vec::with_capacity(delta_prefix.len() + delta_suffix.len() + 256);

    // content_block_delta events
    let mut carry: Option<StreamEvent> = None;
    loop {
        let event = match carry.take() {
            Some(e) => e,
            None => match token_rx.recv().await {
                Some(e) => e,
                None => break,
            },
        };
        match event {
            StreamEvent::Delta(text) => {
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
            StreamEvent::Done {
                prompt_tokens,
                completion_tokens: output_tokens,
            } => {
                // content_block_stop
                tcp.write_all(b"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n").await?;

                // message_delta
                let delta = format!(
                    "event: message_delta\ndata: {{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":\"end_turn\"}},\"usage\":{{\"output_tokens\":{output_tokens}}}}}\n\n"
                );
                tcp.write_all(delta.as_bytes()).await?;

                // message_stop
                tcp.write_all(b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
                    .await?;
                let _ = prompt_tokens; // used in message_start above (approximated)
                break;
            }
            StreamEvent::Error(_) => break,
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
}
