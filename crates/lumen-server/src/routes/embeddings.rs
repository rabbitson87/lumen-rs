use atomic_http::external::http::{Request, Response, StatusCode};
use atomic_http::*;

use crate::embedding::EmbeddingHandle;
use crate::types::{
    EmbeddingData, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage, ErrorResponse,
};

pub async fn handle(
    request: Request<ArenaBody>,
    mut response: Response<ArenaWriter>,
    embedding: Option<EmbeddingHandle>,
) -> Result<(), SendableError> {
    let handle = match embedding {
        Some(h) => h,
        None => {
            let err = ErrorResponse::new(
                "embedding model not loaded — set EMBEDDING_MODEL_ID and restart the server",
                503,
            );
            response.body_mut().set_arena_json(&err)?;
            *response.status_mut() = StatusCode::from_u16(503)?;
            response.responser_arena().await?;
            return Ok(());
        }
    };

    let req: EmbeddingRequest = match request.get_json_arena() {
        Ok(r) => r,
        Err(e) => {
            let err = ErrorResponse::new(format!("invalid request: {e}"), 400);
            response.body_mut().set_arena_json(&err)?;
            *response.status_mut() = StatusCode::from_u16(400)?;
            response.responser_arena().await?;
            return Ok(());
        }
    };

    if let Some(fmt) = req.encoding_format.as_deref() {
        if fmt != "float" {
            let err = ErrorResponse::new(
                format!("encoding_format={fmt:?} not supported — only \"float\" in v1"),
                400,
            );
            response.body_mut().set_arena_json(&err)?;
            *response.status_mut() = StatusCode::from_u16(400)?;
            response.responser_arena().await?;
            return Ok(());
        }
    }

    let texts = req.input.into_vec();
    if texts.is_empty() {
        let err = ErrorResponse::new("input must contain at least one string", 400);
        response.body_mut().set_arena_json(&err)?;
        *response.status_mut() = StatusCode::from_u16(400)?;
        response.responser_arena().await?;
        return Ok(());
    }

    match handle.embed(texts).await {
        Ok(batch) => {
            let data: Vec<EmbeddingData> = batch
                .embeddings
                .into_iter()
                .enumerate()
                .map(|(index, embedding)| EmbeddingData {
                    object: "embedding".into(),
                    index,
                    embedding,
                })
                .collect();
            let resp = EmbeddingResponse {
                object: "list".into(),
                data,
                model: req.model,
                usage: EmbeddingUsage {
                    prompt_tokens: batch.prompt_tokens,
                    total_tokens: batch.prompt_tokens,
                },
            };
            response.body_mut().set_arena_json(&resp)?;
            *response.status_mut() = StatusCode::from_u16(200)?;
        }
        Err(e) => {
            let err = ErrorResponse::new(format!("embedding error: {e}"), 500);
            response.body_mut().set_arena_json(&err)?;
            *response.status_mut() = StatusCode::from_u16(500)?;
        }
    }

    response.responser_arena().await?;
    Ok(())
}
