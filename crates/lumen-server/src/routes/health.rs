use atomic_http::external::http::Response;
use atomic_http::*;
use serde::Serialize;

use super::respond;

#[derive(Serialize)]
struct Health {
    status: &'static str,
}

/// Liveness probe. The accept loop only starts after the model is loaded, so
/// any reachable `/health` already implies a ready server — a static 200 is
/// correct (no need for llama.cpp's 503-while-loading middleware here).
pub async fn handle(response: Response<ArenaWriter>) -> Result<(), SendableError> {
    respond(response, 200, &Health { status: "ok" }).await
}
