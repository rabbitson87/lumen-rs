use atomic_http::external::http::{Request, Response};
use atomic_http::*;

use super::{reply_err, respond};
use crate::embedding::EmbeddingHandle;
use crate::types::{EmbeddingData, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage};

pub async fn handle(
    request: Request<ArenaBody>,
    response: Response<ArenaWriter>,
    embedding: Option<EmbeddingHandle>,
) -> Result<(), SendableError> {
    let Some(handle) = embedding else {
        return reply_err(
            response,
            "embedding model not loaded — set EMBEDDING_MODEL_ID and restart the server",
            503,
        )
        .await;
    };

    let req: EmbeddingRequest = match request.get_json_arena() {
        Ok(r) => r,
        Err(e) => return reply_err(response, format!("invalid request: {e}"), 400).await,
    };

    if let Some(fmt) = req.encoding_format.as_deref() {
        if fmt != "float" {
            return reply_err(
                response,
                format!("encoding_format={fmt:?} not supported — only \"float\" in v1"),
                400,
            )
            .await;
        }
    }

    let texts = req.input.into_vec();
    if texts.is_empty() {
        return reply_err(response, "input must contain at least one string", 400).await;
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
            respond(response, 200, &resp).await
        }
        Err(e) => reply_err(response, format!("embedding error: {e}"), 500).await,
    }
}
