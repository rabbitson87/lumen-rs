use atomic_http::external::http::{Request, Response, StatusCode};
use atomic_http::*;

use crate::engine::EngineHandle;

/// `GET /v1/loads` — pure-JSON serving stats snapshot (WS-F #2).
///
/// Reads the process-lifetime atomic counters the engine bumps at each chat
/// completion directly off the shared `Arc<ServerLoadStats>` carried by the
/// `EngineHandle`. No channel round-trip — the counters are lock-free, so
/// this stays cheap and never queues behind an in-flight generation.
pub async fn handle(
    _request: Request<ArenaBody>,
    mut response: Response<ArenaWriter>,
    handle: EngineHandle,
) -> Result<(), SendableError> {
    let snapshot = handle.load_stats().snapshot();
    response.body_mut().set_arena_json(&snapshot)?;
    *response.status_mut() = StatusCode::from_u16(200)?;
    response.responser_arena().await?;
    Ok(())
}
