use atomic_http::external::http::{Request, Response, StatusCode};
use atomic_http::*;

use crate::engine::EngineHandle;
use crate::types::{DropSessionResponse, ErrorResponse};

/// `DELETE /v1/sessions/{id}` — drop the MLX prompt cache for `id`. Returns
/// `{"id":..., "object":"session", "deleted": bool}` (200 on hit, 404 on miss).
pub async fn handle(
    request: Request<ArenaBody>,
    mut response: Response<ArenaWriter>,
    handle: EngineHandle,
) -> Result<(), SendableError> {
    let path = request.uri().path();
    let id = match path.strip_prefix("/v1/sessions/") {
        Some(id) if !id.is_empty() && !id.contains('/') => id.to_string(),
        _ => {
            let err = ErrorResponse::new("invalid session id", 400);
            response.body_mut().set_arena_json(&err)?;
            *response.status_mut() = StatusCode::from_u16(400)?;
            response.responser_arena().await?;
            return Ok(());
        }
    };

    let deleted = handle.drop_session(id.clone()).await;
    let body = DropSessionResponse {
        id,
        object: "session".into(),
        deleted,
    };
    response.body_mut().set_arena_json(&body)?;
    *response.status_mut() = if deleted {
        StatusCode::from_u16(200)?
    } else {
        StatusCode::from_u16(404)?
    };
    response.responser_arena().await?;
    Ok(())
}
