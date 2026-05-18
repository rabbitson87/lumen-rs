use atomic_http::external::http::{Request, Response, StatusCode};
use atomic_http::*;

use crate::engine::EngineHandle;
use crate::types::{ClearPrefixCacheResponse, DropPrefixCacheResponse, ErrorResponse};

/// `DELETE /v1/prefix-cache/{key}` — drop the A1 prefix-cache entry for `key`.
/// On hit: 200 with `{key, object:"prefix_cache", deleted: true}`. On miss:
/// 404 with `deleted: false`. The key is the auto-generated
/// `auto-<16-hex>` value (hash of system message text); see eprintln logs
/// from the MLX backend for the exact value at request time.
pub async fn handle_delete_one(
    request: Request<ArenaBody>,
    mut response: Response<ArenaWriter>,
    handle: EngineHandle,
) -> Result<(), SendableError> {
    let path = request.uri().path();
    let key = match path.strip_prefix("/v1/prefix-cache/") {
        Some(k) if !k.is_empty() && !k.contains('/') => k.to_string(),
        _ => {
            let err = ErrorResponse::new("invalid prefix-cache key", 400);
            response.body_mut().set_arena_json(&err)?;
            *response.status_mut() = StatusCode::from_u16(400)?;
            response.responser_arena().await?;
            return Ok(());
        }
    };

    let deleted = handle.drop_prefix_cache(key.clone()).await;
    let body = DropPrefixCacheResponse {
        key,
        object: "prefix_cache".into(),
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

/// `DELETE /v1/prefix-cache` — drop ALL entries. Returns
/// `{object:"prefix_cache.clear", cleared: N}`. Always 200.
pub async fn handle_delete_all(
    _request: Request<ArenaBody>,
    mut response: Response<ArenaWriter>,
    handle: EngineHandle,
) -> Result<(), SendableError> {
    let cleared = handle.clear_prefix_cache().await;
    let body = ClearPrefixCacheResponse {
        object: "prefix_cache.clear".into(),
        cleared,
    };
    response.body_mut().set_arena_json(&body)?;
    *response.status_mut() = StatusCode::from_u16(200)?;
    response.responser_arena().await?;
    Ok(())
}
