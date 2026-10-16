use atomic_http::external::http::{Request, Response};
use atomic_http::*;

use super::{path_id, reply_err, respond};
use crate::engine::EngineHandle;
use crate::types::DropSessionResponse;

/// `DELETE /v1/sessions/{id}` — drop the MLX prompt cache for `id`. Returns
/// `{"id":..., "object":"session", "deleted": bool}` (200 on hit, 404 on miss).
pub async fn handle(
    request: Request<ArenaBody>,
    response: Response<ArenaWriter>,
    handle: EngineHandle,
) -> Result<(), SendableError> {
    let Some(id) = path_id(request.uri().path(), "/v1/sessions/") else {
        return reply_err(response, "invalid session id", 400).await;
    };
    let id = id.to_string();

    let deleted = handle.drop_session(id.clone()).await;
    let body = DropSessionResponse {
        id,
        object: "session".into(),
        deleted,
    };
    respond(response, if deleted { 200 } else { 404 }, &body).await
}
