use atomic_http::external::http::{Request, Response, StatusCode};
use atomic_http::*;

use crate::engine::EngineHandle;

pub async fn handle(
    _request: Request<ArenaBody>,
    mut response: Response<ArenaWriter>,
    handle: EngineHandle,
) -> Result<(), SendableError> {
    let models = handle.list_models().await;
    response.body_mut().set_arena_json(&models)?;
    *response.status_mut() = StatusCode::from_u16(200)?;
    response.responser_arena().await?;
    Ok(())
}
