use atomic_http::external::http::{Request, Response};
use atomic_http::*;

use super::{path_id, reply_err, respond};
use crate::engine::EngineHandle;
use crate::types::{ClearPrefixCacheResponse, DropPrefixCacheResponse};

/// `DELETE /v1/prefix-cache/{key}` — drop the A1 prefix-cache entry for `key`.
/// On hit: 200 with `{key, object:"prefix_cache", deleted: true}`. On miss:
/// 404 with `deleted: false`. The key is the auto-generated
/// `auto-<16-hex>` value (hash of system message text); see eprintln logs
/// from the MLX backend for the exact value at request time.
pub async fn handle_delete_one(
    request: Request<ArenaBody>,
    response: Response<ArenaWriter>,
    handle: EngineHandle,
) -> Result<(), SendableError> {
    let Some(key) = path_id(request.uri().path(), "/v1/prefix-cache/") else {
        return reply_err(response, "invalid prefix-cache key", 400).await;
    };
    let key = key.to_string();

    let deleted = handle.drop_prefix_cache(key.clone()).await;
    let body = DropPrefixCacheResponse {
        key,
        object: "prefix_cache".into(),
        deleted,
    };
    respond(response, if deleted { 200 } else { 404 }, &body).await
}

/// `DELETE /v1/prefix-cache` — drop ALL entries. Returns
/// `{object:"prefix_cache.clear", cleared: N}`. Always 200.
pub async fn handle_delete_all(
    _request: Request<ArenaBody>,
    response: Response<ArenaWriter>,
    handle: EngineHandle,
) -> Result<(), SendableError> {
    let cleared = handle.clear_prefix_cache().await;
    let body = ClearPrefixCacheResponse {
        object: "prefix_cache.clear".into(),
        cleared,
    };
    respond(response, 200, &body).await
}
