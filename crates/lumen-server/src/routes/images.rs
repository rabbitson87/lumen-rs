//! `POST /v1/images/generations` — OpenAI-compatible text-to-image.
//!
//! Backed by the native FLUX.2-dev pipeline in `lumen-diffusion` via a
//! channel-actor (`DiffusionHandle`). When no diffusion backend is loaded
//! (i.e. the server is in `chat` mode), this returns 503.

use atomic_http::external::http::{Request, Response, StatusCode};
use atomic_http::*;

use crate::diffusion_engine::{DiffusionHandle, GenerateParams};
use crate::types::{ErrorResponse, ImageData, ImageGenerationRequest, ImageGenerationResponse};

/// Upper bound on `n` to avoid a single request monopolizing the (single,
/// blocking) diffusion worker for an unbounded time.
const MAX_IMAGES_PER_REQUEST: usize = 4;

pub async fn handle(
    request: Request<ArenaBody>,
    mut response: Response<ArenaWriter>,
    diffusion: Option<DiffusionHandle>,
) -> Result<(), SendableError> {
    // 503 when the image backend is not loaded (chat-only mode).
    let handle = match diffusion {
        Some(h) => h,
        None => {
            return reply_err(
                response,
                "image backend not loaded — start the server with LUMEN_SERVE=image or \
                 LUMEN_SERVE=hybrid to enable /v1/images/generations",
                503,
            )
            .await;
        }
    };

    let req: ImageGenerationRequest = match request.get_json_arena() {
        Ok(r) => r,
        Err(e) => return reply_err(response, format!("invalid request: {e}"), 400).await,
    };

    // response_format: only b64_json is supported (no URL host on a local server).
    if let Some(fmt) = req.response_format.as_deref() {
        if fmt != "b64_json" {
            return reply_err(
                response,
                format!("response_format={fmt:?} not supported — only \"b64_json\" (local server has no URL host)"),
                400,
            )
            .await;
        }
    }

    let (width, height) = match req.dimensions() {
        Ok(wh) => wh,
        Err(msg) => return reply_err(response, msg, 400).await,
    };

    let n = req.n.clamp(1, MAX_IMAGES_PER_REQUEST);
    if req.prompt.trim().is_empty() {
        return reply_err(response, "prompt must not be empty", 400).await;
    }

    let mut data: Vec<ImageData> = Vec::with_capacity(n);
    for i in 0..n {
        // Vary the seed per image so `n > 1` yields distinct results.
        let seed = req.seed.wrapping_add(i as u64);
        let params = GenerateParams {
            prompt: req.prompt.clone(),
            width,
            height,
            steps: req.steps.max(1),
            seed,
            guidance: req.guidance,
        };
        match handle.generate(params).await {
            Ok(img) => {
                use base64::Engine;
                let b64 = base64::engine::general_purpose::STANDARD.encode(&img.png);
                data.push(ImageData {
                    b64_json: Some(b64),
                });
            }
            Err(e) => {
                return reply_err(response, format!("image generation error: {e:#}"), 500).await;
            }
        }
    }

    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let resp = ImageGenerationResponse { created, data };
    response.body_mut().set_arena_json(&resp)?;
    *response.status_mut() = StatusCode::from_u16(200)?;
    response.responser_arena().await?;
    Ok(())
}

async fn reply_err(
    mut response: Response<ArenaWriter>,
    message: impl Into<String>,
    code: u16,
) -> Result<(), SendableError> {
    let err = ErrorResponse::new(message, code);
    response.body_mut().set_arena_json(&err)?;
    *response.status_mut() = StatusCode::from_u16(code)?;
    response.responser_arena().await?;
    Ok(())
}
