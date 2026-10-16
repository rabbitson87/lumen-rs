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
