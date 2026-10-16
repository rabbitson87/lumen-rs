use atomic_http::external::http::{Request, Response};
use atomic_http::*;

use super::respond;
use crate::engine::EngineHandle;

pub async fn handle(
    _request: Request<ArenaBody>,
    response: Response<ArenaWriter>,
    handle: EngineHandle,
) -> Result<(), SendableError> {
    let models = handle.list_models().await;
    respond(response, 200, &models).await
}
