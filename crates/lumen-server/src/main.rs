mod catalog;
mod embedding;
mod engine;
mod load_stats;
mod routes;
mod types;

use atomic_http::external::http::{Response, StatusCode};
use atomic_http::*;

use embedding::{EmbeddingHandle, EmbeddingService};
use engine::{EngineHandle, InferenceEngine};
use types::ErrorResponse;

const DEFAULT_MODEL: &str = "Qwen/Qwen2.5-0.5B";
// 41110 picked because: (1) far from common services (8080 collides with web
// dev servers, 11434 collides with Ollama), (2) phonetic "LU-MEN" via leetspeak
// L=1, U=4(rotated), M=1, EN=10. Override with PORT= env var as before.
const DEFAULT_PORT: u16 = 41110;

/// Embedded mlx Metal kernel library. Populated by build.rs from
/// `target/.../out/build/lib/mlx.metallib`. Under non-mlx-native builds
/// this is an empty placeholder (the unpack path below is feature-gated).
#[cfg(feature = "mlx-native")]
static EMBEDDED_MLX_METALLIB: &[u8] = include_bytes!(env!("LUMEN_MLX_METALLIB_PATH"));

/// Drop the embedded metallib next to the running binary so mlx's
/// `load_colocated_library("mlx")` finds it on startup. Idempotent —
/// skips if a matching-size file already exists. Errors surface as
/// stderr warnings; mlx will then fail with its native error message,
/// which is more actionable than a panic here.
#[cfg(feature = "mlx-native")]
fn ensure_metallib_colocated() {
    if EMBEDDED_MLX_METALLIB.is_empty() {
        return; // Built without an actual metallib — nothing to unpack.
    }
    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[metallib] current_exe failed: {e} — skipping unpack");
            return;
        }
    };
    let Some(dir) = exe.parent() else {
        return;
    };
    let target = dir.join("mlx.metallib");
    // Skip if already present with the same size (cheap idempotency check —
    // avoids rewriting on every restart).
    if let Ok(meta) = std::fs::metadata(&target) {
        if meta.len() as usize == EMBEDDED_MLX_METALLIB.len() {
            return;
        }
    }
    match std::fs::write(&target, EMBEDDED_MLX_METALLIB) {
        Ok(_) => eprintln!(
            "[metallib] unpacked {} ({} MB)",
            target.display(),
            EMBEDDED_MLX_METALLIB.len() / (1024 * 1024)
        ),
        Err(e) => eprintln!(
            "[metallib] WARN failed to write {}: {e} — mlx will fall back \
             to its built-in search paths",
            target.display()
        ),
    }
}

#[tokio::main]
async fn main() -> Result<(), SendableError> {
    // ── Early-exit flags ───────────────────────────────────────────
    // Handle before any heavy initialization (env loading, model loading,
    // metallib unpack). These flags exist so external tooling (the lumen-app
    // desktop control plane in particular) can introspect the binary without
    // paying the startup cost.
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--catalog") {
        let cat = catalog::catalog();
        let json = serde_json::to_string_pretty(&cat)
            .map_err(|e| SendableError::from(format!("serialize catalog: {e}")))?;
        println!("{json}");
        return Ok(());
    }
    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("lumen-server {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }

    // Auto-load .env from CWD or binary-adjacent directory so deployments
    // can drop config without exporting every var manually. Errors are
    // silently ignored (missing .env is normal — env vars still take
    // precedence via export / launchd).
    if let Ok(path) = dotenvy::dotenv() {
        eprintln!("[env] loaded {}", path.display());
    } else if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let near = dir.join(".env");
            if dotenvy::from_path(&near).is_ok() {
                eprintln!("[env] loaded {}", near.display());
            }
        }
    }

    // Unpack the embedded mlx.metallib next to the binary BEFORE any mlx
    // code path runs (load happens lazily on first MTL::Device usage).
    #[cfg(feature = "mlx-native")]
    ensure_metallib_colocated();

    // R2: Candle's command buffer dispatch batching defaults to 50 ops/buffer.
    // Empirically on Apple Silicon (M-series) for Qwen3.6-27B Dense decode,
    // a smaller value (~10) yields ~+7% tok/s — smaller buffers commit sooner,
    // letting the GPU pipeline through the 5-buffer pool with less idle gap.
    // Keep as a soft default: respect user override.
    if std::env::var_os("CANDLE_METAL_COMPUTE_PER_BUFFER").is_none() {
        // SAFETY: set before any threads are spawned.
        unsafe {
            std::env::set_var("CANDLE_METAL_COMPUTE_PER_BUFFER", "10");
        }
    }

    let model_id = std::env::var("MODEL_ID").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(DEFAULT_PORT);

    eprintln!("Loading model: {model_id}");
    let mut engine = InferenceEngine::load(&model_id)
        .map_err(|e| SendableError::from(format!("failed to load model: {e:#}")))?;

    // Metal memory limits (wired / cache / soft).
    //
    // Three caps work together to keep MLX from running away on macOS:
    //
    //   wired_limit  — OS page-locked ceiling. Prevents resident weights
    //                  from being paged out mid-decode (2-5× decode regression
    //                  on M-series otherwise). Default 28 GB (safe for both
    //                  36 GB M3 Max and 24 GB M4 Pro; mlx caps to the device's
    //                  actual `max_recommended_working_set_size`).
    //   cache_limit  — buffer pool cap. Without this, MLX retains transient
    //                  attention QK^T peaks across requests and the heap
    //                  grows monotonically until the OS kills the process.
    //   memory_limit — soft total cap. MLX starts evicting cache more
    //                  aggressively above this. Acts as a guard rail before
    //                  the hard wired ceiling.
    //
    // Defaults assume 36 GB M3 Max dev box. For 24 GB Mac Mini, set
    // `LUMEN_WIRED_LIMIT_GB=12 LUMEN_CACHE_LIMIT_GB=4 LUMEN_MEMORY_LIMIT_GB=14`.
    // `LUMEN_DISABLE_WIRED_LIMIT=1` skips all three (mlx defaults — for A/B).
    #[cfg(feature = "mlx-native")]
    if std::env::var("LUMEN_DISABLE_WIRED_LIMIT").unwrap_or_default() != "1" {
        fn env_gb(name: &str, default_gb: usize) -> usize {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(default_gb)
        }
        let wired_gb = env_gb("LUMEN_WIRED_LIMIT_GB", 28);
        let cache_gb = env_gb("LUMEN_CACHE_LIMIT_GB", 8);
        let memory_gb = env_gb("LUMEN_MEMORY_LIMIT_GB", 32);
        let gb_to_bytes = |gb: usize| gb * 1024 * 1024 * 1024;

        // Byte-precision override for wired_limit — set by the desktop app to
        // exactly match the active model's safetensors size. Takes precedence
        // over the GB-rounded knob so 13.45 GB weights don't get evicted by a
        // 13 GB ceiling.
        let wired_bytes = std::env::var("LUMEN_WIRED_LIMIT_BYTES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or_else(|| gb_to_bytes(wired_gb));

        match lumen_mlx::metal_memory::set_wired_limit(wired_bytes) {
            Ok(prev) => eprintln!(
                "[mlx-mem] wired_limit raised to {} MB (prev {} MB)",
                wired_bytes / (1024 * 1024),
                prev / (1024 * 1024),
            ),
            Err(e) => eprintln!(
                "[mlx-mem] WARN set_wired_limit={} MB: {e}",
                wired_bytes / (1024 * 1024),
            ),
        }
        match lumen_mlx::metal_memory::set_memory_limit(gb_to_bytes(memory_gb)) {
            Ok(prev) => eprintln!(
                "[mlx-mem] memory_limit set to {memory_gb} GB (prev {} GB)",
                prev / (1024 * 1024 * 1024)
            ),
            Err(e) => eprintln!("[mlx-mem] WARN set_memory_limit={memory_gb} GB: {e}"),
        }
        match lumen_mlx::metal_memory::set_cache_limit(gb_to_bytes(cache_gb)) {
            Ok(prev) => eprintln!(
                "[mlx-mem] cache_limit set to {cache_gb} GB (prev {} GB)",
                prev / (1024 * 1024 * 1024)
            ),
            Err(e) => eprintln!("[mlx-mem] WARN set_cache_limit={cache_gb} GB: {e}"),
        }
    }

    engine
        .warmup()
        .map_err(|e| SendableError::from(format!("warmup failed: {e}")))?;

    // Channel-based engine: no Mutex, requests queue through channel.
    // Hand the shared lifetime-stats accumulator to the handle BEFORE moving
    // the engine into its task, so `GET /v1/loads` reads the same atomics the
    // engine bumps at each chat completion.
    let load_stats = engine.load_stats();
    let (tx, rx) = tokio::sync::mpsc::channel(32);
    let handle = EngineHandle::new(tx, load_stats);
    tokio::spawn(async move { engine.run(rx).await });

    // Optional embedding model. Only spawn the subprocess when
    // `EMBEDDING_MODEL_ID` is set — otherwise `/v1/embeddings` returns
    // 503 with an actionable message. The service runs on a dedicated
    // blocking thread because the underlying subprocess pipe is
    // synchronous (blocking_recv inside the loop).
    let embedding_handle: Option<EmbeddingHandle> = match std::env::var("EMBEDDING_MODEL_ID") {
        Ok(model_id) if !model_id.is_empty() => {
            eprintln!("Loading embedding model: {model_id}");
            match EmbeddingService::load(&model_id) {
                Ok(service) => {
                    let (etx, erx) = tokio::sync::mpsc::channel(64);
                    std::thread::Builder::new()
                        .name("embedding-service".into())
                        .spawn(move || service.run(erx))
                        .map_err(|e| {
                            SendableError::from(format!("failed to spawn embedding thread: {e}"))
                        })?;
                    Some(EmbeddingHandle::new(etx))
                }
                Err(e) => {
                    eprintln!(
                        "[embedding] WARN failed to load {model_id}: {e:#} — /v1/embeddings will return 503"
                    );
                    None
                }
            }
        }
        _ => None,
    };

    let addr = format!("0.0.0.0:{port}");
    let mut server = Server::new(&addr).await?;
    eprintln!("TurboQuant serving on :{port}");
    eprintln!("  POST   /v1/messages          (Anthropic Messages API)");
    eprintln!("  POST   /v1/chat/completions  (OpenAI)");
    eprintln!("  POST   /v1/completions       (OpenAI)");
    eprintln!(
        "  POST   /v1/embeddings        (OpenAI{})",
        if embedding_handle.is_some() {
            ""
        } else {
            " — disabled, set EMBEDDING_MODEL_ID"
        }
    );
    eprintln!("  GET    /v1/models");
    eprintln!("  GET    /v1/loads             (live serving stats, JSON)");
    eprintln!("  DELETE /v1/sessions/{{id}}     (drop MLX prompt cache)");
    eprintln!("  DELETE /v1/prefix-cache/{{key}} (drop one A1 master snapshot)");
    eprintln!("  DELETE /v1/prefix-cache        (clear all A1 master snapshots)");

    loop {
        let accept = server.accept().await?;
        let handle = handle.clone();
        let embedding = embedding_handle.clone();

        tokio::spawn(async move {
            if let Err(e) = handle_connection(accept, handle, embedding).await {
                eprintln!("connection error: {e}");
            }
        });
    }
}

async fn handle_connection(
    accept: Accept,
    handle: EngineHandle,
    embedding: Option<EmbeddingHandle>,
) -> Result<(), SendableError> {
    let (request, response) = accept.parse_request_arena_writer().await?;

    let method = request.method().as_str();
    let path = request.uri().path();

    eprintln!("{method} {path}");

    match (method, path) {
        ("POST", "/v1/messages") => routes::messages::handle(request, response, handle).await,
        ("POST", "/v1/chat/completions") => routes::chat::handle(request, response, handle).await,
        ("POST", "/v1/completions") => routes::completions::handle(request, response, handle).await,
        ("POST", "/v1/embeddings") => {
            routes::embeddings::handle(request, response, embedding).await
        }
        ("GET", "/health") => routes::health::handle(response).await,
        ("GET", "/v1/loads") => routes::loads::handle(request, response, handle).await,
        ("GET", "/v1/models") => routes::models::handle(request, response, handle).await,
        ("DELETE", p) if p.starts_with("/v1/sessions/") => {
            routes::sessions::handle(request, response, handle).await
        }
        ("DELETE", "/v1/prefix-cache") => {
            routes::prefix_cache::handle_delete_all(request, response, handle).await
        }
        ("DELETE", p) if p.starts_with("/v1/prefix-cache/") => {
            routes::prefix_cache::handle_delete_one(request, response, handle).await
        }
        _ => handle_not_found(response).await,
    }
}

async fn handle_not_found(mut response: Response<ArenaWriter>) -> Result<(), SendableError> {
    let err = ErrorResponse::new("not found", 404);
    response.body_mut().set_arena_json(&err)?;
    *response.status_mut() = StatusCode::from_u16(404)?;
    response.responser_arena().await?;
    Ok(())
}
