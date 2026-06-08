use atomic_http::external::http::{Response, StatusCode};
use atomic_http::*;
use serde::Serialize;

#[derive(Serialize)]
struct Health {
    status: &'static str,
}

/// Liveness probe. The accept loop only starts after the model is loaded, so
/// any reachable `/health` already implies a ready server — a static 200 is
/// correct (no need for llama.cpp's 503-while-loading middleware here).
pub async fn handle(mut response: Response<ArenaWriter>) -> Result<(), SendableError> {
    response
        .body_mut()
        .set_arena_json(&Health { status: "ok" })?;
    *response.status_mut() = StatusCode::from_u16(200)?;
    response.responser_arena().await?;
    Ok(())
}
