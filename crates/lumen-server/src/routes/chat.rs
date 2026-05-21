use atomic_http::external::http::{Request, Response, StatusCode};
use atomic_http::*;
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc::error::TryRecvError;

use crate::engine::{EngineHandle, StreamEvent};
#[allow(unused_imports)]
use crate::engine::FinishReason;
use crate::types::{
    ChatCompletionChunk, ChatCompletionRequest, ChatStreamChoice, ChatStreamDelta, ErrorResponse,
    FunctionCallDelta, ToolCallDelta, Usage,
};

pub async fn handle(
    request: Request<ArenaBody>,
    mut response: Response<ArenaWriter>,
    handle: EngineHandle,
) -> Result<(), SendableError> {
    let req: ChatCompletionRequest = match request.get_json_arena() {
        Ok(r) => r,
        Err(e) => {
            let err = ErrorResponse::new(format!("invalid request: {e}"), 400);
            response.body_mut().set_arena_json(&err)?;
            *response.status_mut() = StatusCode::from_u16(400)?;
            response.responser_arena().await?;
            return Ok(());
        }
    };

    if req.stream {
        return handle_streaming(req, response, handle).await;
    }

    match handle.chat_completion(req).await {
        Ok(resp) => {
            response.body_mut().set_arena_json(&resp)?;
            *response.status_mut() = StatusCode::from_u16(200)?;
        }
        Err(e) => {
            let err = ErrorResponse::new(format!("inference error: {e}"), 500);
            response.body_mut().set_arena_json(&err)?;
            *response.status_mut() = StatusCode::from_u16(500)?;
        }
    }

    response.responser_arena().await?;
    Ok(())
}

async fn handle_streaming(
    req: ChatCompletionRequest,
    response: Response<ArenaWriter>,
    handle: EngineHandle,
) -> Result<(), SendableError> {
    let model = req.model.clone();
    let include_usage = req
        .stream_options
        .as_ref()
        .map(|o| o.include_usage)
        .unwrap_or(false);
    let mut tcp = response.into_body().stream;

    // Write HTTP headers for SSE
    tcp.write_all(
        b"HTTP/1.1 200 OK\r\n\
          Content-Type: text/event-stream\r\n\
          Cache-Control: no-cache\r\n\
          Connection: keep-alive\r\n\
          \r\n",
    )
    .await?;

    let mut token_rx = match handle.chat_completion_streaming(req).await {
        Ok(rx) => rx,
        Err(e) => {
            let msg = format!("data: {{\"error\":\"{e}\"}}\n\n");
            tcp.write_all(msg.as_bytes()).await?;
            tcp.flush().await?;
            return Ok(());
        }
    };

    let id = format!(
        "chatcmpl-{:x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    );
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Send initial chunk with role
    let initial = ChatCompletionChunk {
        id: id.clone(),
        object: "chat.completion.chunk".into(),
        created,
        model: model.clone(),
        choices: vec![ChatStreamChoice {
            index: 0,
            delta: ChatStreamDelta {
                role: Some("assistant".into()),
                content: None,
                tool_calls: None,
            },
            finish_reason: None,
        }],
        usage: None,
    };
    write_sse(&mut tcp, &initial).await?;

    // Hot-path delta envelope: precompute the static prefix/suffix once. The only
    // per-token cost is JSON-escaping the content text and a single TCP send.
    // Eliminates 5+ allocations per token (id/model clones, format!, double serde).
    let id_json = serde_json::to_string(&id).map_err(|e| SendableError::from(e.to_string()))?;
    let model_json =
        serde_json::to_string(&model).map_err(|e| SendableError::from(e.to_string()))?;
    let delta_prefix = format!(
        "data: {{\"id\":{id_json},\"object\":\"chat.completion.chunk\",\"created\":{created},\"model\":{model_json},\"choices\":[{{\"index\":0,\"delta\":{{\"content\":"
    );
    // Match legacy serde output: `usage: Option<Usage>` is serialized with
    // `skip_serializing_if = "Option::is_none"`, so it is omitted entirely on
    // delta chunks. Keeping byte-equivalence preserves wire-format compatibility.
    let delta_suffix: &[u8] = b"},\"finish_reason\":null}]}\n\n";
    let mut buf: Vec<u8> = Vec::with_capacity(delta_prefix.len() + delta_suffix.len() + 256);

    // Per-delta wallclock instrumentation (Phase C Option B).
    // LUMEN_STREAM_TIMING=1 → logs first/skip/last delta-recv + last
    // write_all + post-DONE-flush wallclock so the SSE pipeline's
    // contribution to the ~6% native-specific HTTP envelope loss can be
    // localized vs lib.rs::chat_streaming inner emit rate and the bench
    // client steady rate. LUMEN_STREAM_SKIP=N (default 20) matches
    // `bench_b4_decode_ab.py --skip-first`.
    let stream_timing = std::env::var("LUMEN_STREAM_TIMING").is_ok();
    let stream_skip: usize = std::env::var("LUMEN_STREAM_SKIP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let mut t_first_recv: Option<std::time::Instant> = None;
    let mut t_skip_recv: Option<std::time::Instant> = None;
    let mut t_last_recv: Option<std::time::Instant> = None;
    let mut t_last_write: Option<std::time::Instant> = None;
    let mut n_deltas: usize = 0;

    // Stream content deltas. `carry` lets the inner try_recv drain push a non-Delta
    // event back into the outer loop's processing path, so finalization stays in one place.
    //
    // chunked prefill can hold the
    // engine task busy for several seconds on long prompts (10K+ tokens)
    // before the first decode token is produced. SSE-comment heartbeats
    // (`: text\n\n`) are spec-compliant but some OpenAI clients filter
    // them out before signalling activity to the application layer, so
    // the upstream still surfaces "empty response" on timeout. Emit an
    // empty `data:` delta chunk instead — well-formed OpenAI streaming
    // event with no content, which every compliant client recognizes as
    // live traffic but ignores at the consumer level.
    let keepalive_chunk: Vec<u8> = {
        let mut s = String::with_capacity(delta_prefix.len() + 32);
        s.push_str("data: {\"id\":");
        s.push_str(&id_json);
        s.push_str(",\"object\":\"chat.completion.chunk\",\"created\":");
        s.push_str(&created.to_string());
        s.push_str(",\"model\":");
        s.push_str(&model_json);
        s.push_str(",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":null}]}\n\n");
        s.into_bytes()
    };
    let mut carry: Option<StreamEvent> = None;
    loop {
        let event_opt: Option<StreamEvent> = match carry.take() {
            Some(e) => Some(e),
            None => loop {
                tokio::select! {
                    res = token_rx.recv() => break res,
                    _ = tokio::time::sleep(std::time::Duration::from_millis(1000)) => {
                        tcp.write_all(&keepalive_chunk).await?;
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
            StreamEvent::Delta(text) => {
                if stream_timing {
                    let now = std::time::Instant::now();
                    if t_first_recv.is_none() {
                        t_first_recv = Some(now);
                    }
                    if n_deltas == stream_skip {
                        t_skip_recv = Some(now);
                    }
                    t_last_recv = Some(now);
                    n_deltas += 1;
                }
                buf.clear();
                append_delta(&mut buf, &delta_prefix, delta_suffix, &text);
                // Coalesce: drain any additional ready deltas into the same buffer
                // so a burst from the model becomes one TCP send instead of N.
                loop {
                    match token_rx.try_recv() {
                        Ok(StreamEvent::Delta(more)) => {
                            if stream_timing {
                                let now = std::time::Instant::now();
                                if n_deltas == stream_skip {
                                    t_skip_recv = Some(now);
                                }
                                t_last_recv = Some(now);
                                n_deltas += 1;
                            }
                            append_delta(&mut buf, &delta_prefix, delta_suffix, &more);
                        }
                        Ok(other) => {
                            carry = Some(other);
                            break;
                        }
                        Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => break,
                    }
                }
                tcp.write_all(&buf).await?;
                if stream_timing {
                    t_last_write = Some(std::time::Instant::now());
                }
            }
            StreamEvent::ToolCallStart { index, id: call_id, name } => {
                // Phase 1.5: OpenAI tool_calls start chunk. Carries
                // `index`, `id`, `type:"function"`, and `function.name`
                // exactly once; subsequent argument chunks reference
                // the same `index` without re-sending id/name.
                let chunk = ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk".into(),
                    created,
                    model: model.clone(),
                    choices: vec![ChatStreamChoice {
                        index: 0,
                        delta: ChatStreamDelta {
                            role: None,
                            content: None,
                            tool_calls: Some(vec![ToolCallDelta {
                                index,
                                id: Some(call_id),
                                kind: Some("function".into()),
                                function: Some(FunctionCallDelta {
                                    name: Some(name),
                                    arguments: Some(String::new()),
                                }),
                            }]),
                        },
                        finish_reason: None,
                    }],
                    usage: None,
                };
                write_sse(&mut tcp, &chunk).await?;
            }
            StreamEvent::ToolCallArgumentsDelta {
                index,
                partial_json,
            } => {
                let chunk = ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk".into(),
                    created,
                    model: model.clone(),
                    choices: vec![ChatStreamChoice {
                        index: 0,
                        delta: ChatStreamDelta {
                            role: None,
                            content: None,
                            tool_calls: Some(vec![ToolCallDelta {
                                index,
                                id: None,
                                kind: None,
                                function: Some(FunctionCallDelta {
                                    name: None,
                                    arguments: Some(partial_json),
                                }),
                            }]),
                        },
                        finish_reason: None,
                    }],
                    usage: None,
                };
                write_sse(&mut tcp, &chunk).await?;
            }
            StreamEvent::ToolCallStop { .. } => {
                // OpenAI SSE has no per-tool-call stop event; the
                // final `finish_reason: "tool_calls"` chunk closes the
                // implicit envelope. Suppress.
            }
            StreamEvent::Done {
                prompt_tokens,
                completion_tokens,
                finish_reason,
            } => {
                // Final content chunk with finish_reason. Per OpenAI spec, this
                // chunk's `usage` is null even when include_usage=true; the
                // usage payload arrives in a follow-up chunk with empty choices.
                let chunk = ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk".into(),
                    created,
                    model: model.clone(),
                    choices: vec![ChatStreamChoice {
                        index: 0,
                        delta: ChatStreamDelta {
                            role: None,
                            content: None,
                            tool_calls: None,
                        },
                        finish_reason: Some(finish_reason.openai_str().into()),
                    }],
                    usage: None,
                };
                write_sse(&mut tcp, &chunk).await?;

                if include_usage {
                    let usage_chunk = ChatCompletionChunk {
                        id: id.clone(),
                        object: "chat.completion.chunk".into(),
                        created,
                        model: model.clone(),
                        choices: vec![],
                        usage: Some(Usage {
                            prompt_tokens,
                            completion_tokens,
                            total_tokens: prompt_tokens + completion_tokens,
                        }),
                    };
                    write_sse(&mut tcp, &usage_chunk).await?;
                }
                break;
            }
            StreamEvent::Error(_) => break,
        }
    }

    tcp.write_all(b"data: [DONE]\n\n").await?;
    tcp.flush().await?;
    if stream_timing {
        let t_done = std::time::Instant::now();
        if let (Some(tf), Some(tl)) = (t_first_recv, t_last_recv) {
            let recv_span_ms = (tl - tf).as_secs_f64() * 1000.0;
            let steady_span_ms = t_skip_recv
                .map(|ts| (tl - ts).as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            let post_done_ms = t_last_write
                .map(|tw| (t_done - tw).as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            let steady_n = n_deltas.saturating_sub(stream_skip + 1);
            let steady_rate = if steady_span_ms > 0.0 {
                steady_n as f64 / (steady_span_ms / 1000.0)
            } else {
                0.0
            };
            eprintln!(
                "[stream-timing] sse: n_deltas={n_deltas} first->last_recv={recv_span_ms:.1}ms skip{stream_skip}->last_recv={steady_span_ms:.1}ms steady_rate_recv={steady_rate:.2}tok/s last_write->DONE_flush={post_done_ms:.1}ms"
            );
        }
    }
    Ok(())
}

/// Append one SSE delta envelope to `buf`. The text is JSON-string-escaped via
/// `serde_json::to_writer` directly into the buffer — no intermediate String alloc.
fn append_delta(buf: &mut Vec<u8>, prefix: &str, suffix: &[u8], text: &str) {
    buf.extend_from_slice(prefix.as_bytes());
    // Infallible: writing into Vec<u8> never fails, and serde_json on a &str
    // can only fail on Write error.
    serde_json::to_writer(&mut *buf, text).expect("write to Vec cannot fail");
    buf.extend_from_slice(suffix);
}

async fn write_sse<T: serde::Serialize>(
    tcp: &mut tokio::net::TcpStream,
    data: &T,
) -> Result<(), SendableError> {
    let json = serde_json::to_string(data).map_err(|e| SendableError::from(e.to_string()))?;
    let msg = format!("data: {json}\n\n");
    tcp.write_all(msg.as_bytes()).await?;
    tcp.flush().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_prefix(id: &str, model: &str, created: u64) -> String {
        let id_json = serde_json::to_string(id).unwrap();
        let model_json = serde_json::to_string(model).unwrap();
        format!(
            "data: {{\"id\":{id_json},\"object\":\"chat.completion.chunk\",\"created\":{created},\"model\":{model_json},\"choices\":[{{\"index\":0,\"delta\":{{\"content\":"
        )
    }

    const SUFFIX: &[u8] = b"},\"finish_reason\":null}]}\n\n";

    #[test]
    fn append_delta_produces_valid_sse_chunk() {
        let prefix = build_prefix("chatcmpl-abc", "qwen-test", 1234);
        let mut buf = Vec::new();
        append_delta(&mut buf, &prefix, SUFFIX, "hello");
        let s = std::str::from_utf8(&buf).unwrap();
        assert!(s.starts_with("data: {"));
        assert!(s.ends_with("\n\n"));
        // The full envelope after the `data: ` prefix must be valid JSON.
        let json_part = s.strip_prefix("data: ").unwrap().trim_end_matches("\n\n");
        let v: serde_json::Value = serde_json::from_str(json_part).unwrap();
        assert_eq!(v["choices"][0]["delta"]["content"], "hello");
        assert_eq!(v["model"], "qwen-test");
        assert_eq!(v["id"], "chatcmpl-abc");
    }

    #[test]
    fn append_delta_escapes_special_characters() {
        let prefix = build_prefix("id", "m", 0);
        let mut buf = Vec::new();
        // Content with quotes, backslashes, newlines, control chars must be JSON-escaped.
        append_delta(&mut buf, &prefix, SUFFIX, "a\"b\\c\nd\te");
        let s = std::str::from_utf8(&buf).unwrap();
        let json_part = s.strip_prefix("data: ").unwrap().trim_end_matches("\n\n");
        let v: serde_json::Value = serde_json::from_str(json_part).unwrap();
        assert_eq!(v["choices"][0]["delta"]["content"], "a\"b\\c\nd\te");
    }

    #[test]
    fn append_delta_handles_unicode() {
        let prefix = build_prefix("id", "m", 0);
        let mut buf = Vec::new();
        append_delta(&mut buf, &prefix, SUFFIX, "한국어 🚀");
        let s = std::str::from_utf8(&buf).unwrap();
        let json_part = s.strip_prefix("data: ").unwrap().trim_end_matches("\n\n");
        let v: serde_json::Value = serde_json::from_str(json_part).unwrap();
        assert_eq!(v["choices"][0]["delta"]["content"], "한국어 🚀");
    }

    #[test]
    fn append_delta_coalesces_into_shared_buffer() {
        let prefix = build_prefix("id", "m", 0);
        let mut buf = Vec::new();
        append_delta(&mut buf, &prefix, SUFFIX, "first");
        append_delta(&mut buf, &prefix, SUFFIX, "second");
        let s = std::str::from_utf8(&buf).unwrap();
        // Two complete SSE events back-to-back, separated by the inner blank line.
        assert_eq!(s.matches("data: ").count(), 2);
        assert!(s.contains("\"first\""));
        assert!(s.contains("\"second\""));
    }

    #[test]
    fn append_delta_byte_equivalent_to_legacy_path() {
        // Validates that the optimized path produces the same bytes as the previous
        // structured-serialize path would have produced.
        let id = "chatcmpl-deadbeef".to_string();
        let model = "qwen3-test".to_string();
        let created: u64 = 1735689600;
        let text = "Hello, world! \"quoted\" \\backslash\\";

        // New path
        let prefix = build_prefix(&id, &model, created);
        let mut new_buf = Vec::new();
        append_delta(&mut new_buf, &prefix, SUFFIX, text);

        // Legacy path (what the old code did)
        let chunk = ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk".into(),
            created,
            model: model.clone(),
            choices: vec![ChatStreamChoice {
                index: 0,
                delta: ChatStreamDelta {
                    role: None,
                    content: Some(text.to_string()),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
        };
        let legacy = format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap());

        assert_eq!(std::str::from_utf8(&new_buf).unwrap(), legacy);
    }
}
