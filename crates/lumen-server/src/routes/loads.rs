use atomic_http::external::http::{Request, Response};
use atomic_http::*;

use super::respond;
use crate::engine::EngineHandle;

/// `GET /v1/loads` — pure-JSON serving stats snapshot (WS-F #2).
///
/// Reads the process-lifetime atomic counters the engine bumps at each chat
/// completion directly off the shared `Arc<ServerLoadStats>` carried by the
/// `EngineHandle`. No channel round-trip — the counters are lock-free, so
/// this stays cheap and never queues behind an in-flight generation.
pub async fn handle(
    _request: Request<ArenaBody>,
    response: Response<ArenaWriter>,
    handle: EngineHandle,
) -> Result<(), SendableError> {
    let snapshot = handle.load_stats().snapshot();
    respond(response, 200, &snapshot).await
}
