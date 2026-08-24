use atomic_http::external::http::{Request, Response, StatusCode};
use atomic_http::*;

use crate::engine::EngineHandle;
use crate::types::{CompletionRequest, ErrorResponse};

pub async fn handle(
    request: Request<ArenaBody>,
    mut response: Response<ArenaWriter>,
    handle: EngineHandle,
) -> Result<(), SendableError> {
    let req: CompletionRequest = match request.get_json_arena() {
        Ok(r) => r,
        Err(e) => {
            let err = ErrorResponse::new(format!("invalid request: {e}"), 400);
            response.body_mut().set_arena_json(&err)?;
            *response.status_mut() = StatusCode::from_u16(400)?;
            response.responser_arena().await?;
            return Ok(());
        }
    };

    match handle.completion(req).await {
        Ok(resp) => {
            response.body_mut().set_arena_json(&resp)?;
            *response.status_mut() = StatusCode::from_u16(200)?;
        }
        Err(e) => {
            let err = ErrorResponse::new(crate::types::inference_error_message(&e), 500);
            response.body_mut().set_arena_json(&err)?;
            *response.status_mut() = StatusCode::from_u16(500)?;
        }
    }

    response.responser_arena().await?;
    Ok(())
}
