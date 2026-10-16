use atomic_http::external::http::{Request, Response};
use atomic_http::*;

use super::{reply_err, respond};
use crate::engine::EngineHandle;
use crate::types::CompletionRequest;

pub async fn handle(
    request: Request<ArenaBody>,
    response: Response<ArenaWriter>,
    handle: EngineHandle,
) -> Result<(), SendableError> {
    let req: CompletionRequest = match request.get_json_arena() {
        Ok(r) => r,
        Err(e) => return reply_err(response, format!("invalid request: {e}"), 400).await,
    };

    match handle.completion(req).await {
        Ok(resp) => respond(response, 200, &resp).await,
        Err(e) => {
            let status = crate::types::inference_error_status(&e);
            reply_err(response, crate::types::inference_error_message(&e), status).await
        }
    }
}
