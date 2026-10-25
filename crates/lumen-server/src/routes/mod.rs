pub mod chat;
pub mod completions;
pub mod embeddings;
pub mod health;
pub mod images;
pub mod loads;
pub mod messages;
pub mod models;
pub mod prefix_cache;
pub mod sessions;

use atomic_http::external::http::{HeaderMap, Response, StatusCode};
use atomic_http::*;

use crate::types::ErrorResponse;

/// Send `body` as the JSON response, with `status`.
pub async fn respond<T: serde::Serialize>(
    mut response: Response<ArenaWriter>,
    status: u16,
    body: &T,
) -> Result<(), SendableError> {
    response.body_mut().set_arena_json(body)?;
    *response.status_mut() = StatusCode::from_u16(status)?;
    response.responser_arena().await?;
    Ok(())
}

/// [`respond`] with the OpenAI error shape, `status` doubling as its `code`.
pub async fn reply_err(
    response: Response<ArenaWriter>,
    message: impl Into<String>,
    status: u16,
) -> Result<(), SendableError> {
    respond(response, status, &ErrorResponse::new(message, status)).await
}

/// The `{id}` of a `{prefix}{id}` path: a single, non-empty segment.
pub fn path_id<'a>(path: &'a str, prefix: &str) -> Option<&'a str> {
    path.strip_prefix(prefix)
        .filter(|id| !id.is_empty() && !id.contains('/'))
}

/// The head of a server-sent-events response.
///
/// The streaming routes write their head by hand rather than through
/// `responser_arena`, so headers the connection handler set on the response
/// (CORS) would otherwise never reach the client.
pub fn sse_head(headers: &HeaderMap) -> String {
    let mut head = String::from(
        "HTTP/1.1 200 OK\r\n\
         Content-Type: text/event-stream\r\n\
         Cache-Control: no-cache\r\n\
         Connection: keep-alive\r\n",
    );
    for (name, value) in headers {
        if let Ok(value) = value.to_str() {
            head.push_str(name.as_str());
            head.push_str(": ");
            head.push_str(value);
            head.push_str("\r\n");
        }
    }
    head.push_str("\r\n");
    head
}

/// How long a streaming route waits for the engine to admit or refuse a
/// request before it commits to a `200` and an event stream. An idle engine
/// answers in milliseconds (it counts the prompt and checks it against the
/// context cap before prefill); one busy with another request does not answer
/// until that request is done, and a client must not wait for headers that
/// long. One second is also when the first keep-alive chunk would go out.
const ADMISSION_WAIT: std::time::Duration = std::time::Duration::from_secs(1);

/// What a streaming route learned about its request before writing the head.
pub enum Admission {
    /// The engine refused the request before decoding (the context guard): it
    /// goes out as an HTTP error, as the non-streaming route sends it.
    Refused(String),
    /// The engine answered in time; its first event (or `None` if it closed
    /// the stream) is the stream's to handle.
    Answered(Option<crate::engine::StreamEvent>),
    /// No answer within the wait: the stream opens now and a later refusal
    /// is relayed in-band, as before.
    Pending,
}

/// Wait for the engine's first event on a stream, up to [`ADMISSION_WAIT`].
///
/// The engine sends `Start` once a request is admitted, so an `Error` ahead of
/// it is a refusal. Without this wait a refused stream went out as `200` plus
/// an error chunk, which an SDK reads as a broken stream (and may retry)
/// rather than as the `400` the same request gets without `stream`.
pub async fn await_admission(
    rx: &mut tokio::sync::mpsc::Receiver<crate::engine::StreamEvent>,
) -> Admission {
    await_admission_within(rx, ADMISSION_WAIT).await
}

async fn await_admission_within(
    rx: &mut tokio::sync::mpsc::Receiver<crate::engine::StreamEvent>,
    wait: std::time::Duration,
) -> Admission {
    match tokio::time::timeout(wait, rx.recv()).await {
        Ok(Some(crate::engine::StreamEvent::Error(msg))) => Admission::Refused(msg),
        Ok(first) => Admission::Answered(first),
        Err(_) => Admission::Pending,
    }
}

#[cfg(test)]
mod admission_tests {
    use super::{Admission, await_admission_within};
    use crate::engine::StreamEvent;
    use std::time::Duration;

    const WAIT: Duration = Duration::from_millis(50);

    #[tokio::test]
    async fn an_error_before_the_start_is_a_refusal() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        tx.send(StreamEvent::Error("prompt too large".into()))
            .await
            .unwrap();
        assert!(matches!(
            await_admission_within(&mut rx, WAIT).await,
            Admission::Refused(m) if m == "prompt too large"
        ));
    }

    #[tokio::test]
    async fn an_admitted_stream_keeps_its_first_event() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        tx.send(StreamEvent::Start { prompt_tokens: 7 })
            .await
            .unwrap();
        assert!(matches!(
            await_admission_within(&mut rx, WAIT).await,
            Admission::Answered(Some(StreamEvent::Start { prompt_tokens: 7 }))
        ));
        drop(tx);
        assert!(matches!(
            await_admission_within(&mut rx, WAIT).await,
            Admission::Answered(None)
        ));
    }

    /// A busy engine has not looked at the request yet: the route must not
    /// hold the head back until it does.
    #[tokio::test]
    async fn a_busy_engine_does_not_hold_the_head_back() {
        let (_tx, mut rx) = tokio::sync::mpsc::channel::<StreamEvent>(4);
        let t0 = std::time::Instant::now();
        assert!(matches!(
            await_admission_within(&mut rx, WAIT).await,
            Admission::Pending
        ));
        assert!(t0.elapsed() < Duration::from_secs(1));
    }
}

#[cfg(test)]
mod sse_head_tests {
    use super::sse_head;
    use atomic_http::external::http::{HeaderMap, HeaderValue};

    #[test]
    fn without_extra_headers_the_head_is_unchanged() {
        assert_eq!(
            sse_head(&HeaderMap::new()),
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
             Cache-Control: no-cache\r\nConnection: keep-alive\r\n\r\n"
        );
    }

    #[test]
    fn headers_set_by_the_connection_handler_reach_the_stream() {
        let mut h = HeaderMap::new();
        h.insert("access-control-allow-origin", HeaderValue::from_static("*"));
        let head = sse_head(&h);
        assert!(
            head.contains("\r\naccess-control-allow-origin: *\r\n"),
            "{head:?}"
        );
        assert!(head.ends_with("\r\n\r\n"));
    }
}
