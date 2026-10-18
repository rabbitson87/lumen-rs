use serde::Serialize;
use tauri::{AppHandle, Emitter, State};

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::catalog::Catalog;
use crate::config::{
    AdvancedConfig, ContextConfig, FlashNextConfig, PersistentConfig, QuantConfig, ServerConfig,
    config_dir,
};
use crate::doctor::{self, CheckResult, DoctorReport};
use crate::flash_next;
use crate::hardware::{self, MeasureProgress, MeasureStage, SsdOutcome};
use crate::models::{self, DownloadProgress, ModelEntry};
use crate::pack_build::{self, EVENT_PACK, FlashNextStatus};
use crate::server::{self, ServerStatus, TYPED_ENV_KEYS};
use crate::state::AppState;
use crate::sysinfo::{self, MemoryUsage, SystemInfo};

/// Wrap anyhow::Error → String so Tauri can serialize it for the frontend.
type CmdResult<T> = Result<T, String>;

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

// ── Config ─────────────────────────────────────────────────────────────

#[tauri::command]
pub async fn get_config(state: State<'_, AppState>) -> CmdResult<PersistentConfig> {
    let g = state.config.lock().await;
    Ok(g.clone())
}

/// Apply `edit` to the config, persist it, and return what was saved.
async fn update_with(
    state: &AppState,
    edit: impl FnOnce(&mut PersistentConfig),
) -> CmdResult<PersistentConfig> {
    let mut g = state.config.lock().await;
    edit(&mut g);
    g.save().map_err(err)?;
    Ok(g.clone())
}

#[tauri::command]
pub async fn update_server_config(
    state: State<'_, AppState>,
    server: ServerConfig,
) -> CmdResult<PersistentConfig> {
    update_with(&state, |c| c.server = server).await
}

#[tauri::command]
pub async fn update_quant_config(
    state: State<'_, AppState>,
    quant: QuantConfig,
) -> CmdResult<PersistentConfig> {
    update_with(&state, |c| c.quant = quant).await
}

#[tauri::command]
pub async fn update_context_config(
    state: State<'_, AppState>,
    context: ContextConfig,
) -> CmdResult<PersistentConfig> {
    update_with(&state, |c| c.context = context).await
}

#[tauri::command]
pub async fn update_advanced_config(
    state: State<'_, AppState>,
    advanced: AdvancedConfig,
) -> CmdResult<PersistentConfig> {
    update_with(&state, |c| c.advanced = advanced).await
}

/// Saves the Flash-Next card's settings. The values are only ever sent to a Flash-Next server
/// (`server.rs`), so storing them while another model is selected changes nothing there.
#[tauri::command]
pub async fn update_flash_next_config(
    state: State<'_, AppState>,
    flash_next: FlashNextConfig,
) -> CmdResult<PersistentConfig> {
    update_with(&state, |c| c.flash_next = flash_next).await
}

#[tauri::command]
pub async fn update_env_overrides(
    state: State<'_, AppState>,
    env_overrides: BTreeMap<String, String>,
) -> CmdResult<PersistentConfig> {
    update_with(&state, |c| c.env_overrides = env_overrides).await
}

/// Names of env vars that are also editable via dedicated UI fields. The
/// frontend uses this to flag `env_overrides` keys that "shadow" a typed
/// field.
#[tauri::command]
pub async fn typed_env_keys() -> CmdResult<Vec<String>> {
    Ok(TYPED_ENV_KEYS.iter().map(|s| s.to_string()).collect())
}

#[tauri::command]
pub async fn open_config_dir() -> CmdResult<String> {
    let dir = config_dir().map_err(err)?;
    Ok(dir.to_string_lossy().into_owned())
}

// ── System info ────────────────────────────────────────────────────────

#[tauri::command]
pub async fn get_system_info() -> CmdResult<SystemInfo> {
    Ok(sysinfo::probe())
}

/// Live system memory snapshot for the topbar monitor. Returns `None` only
/// if the `vm_stat` probe fails — the UI hides the indicator in that case
/// rather than rendering nonsense.
#[tauri::command]
pub async fn get_memory_usage() -> CmdResult<Option<MemoryUsage>> {
    Ok(sysinfo::current_memory_usage())
}

/// Reset the Metal memory caps in `ServerConfig` to RAM-aware defaults
/// (70/20/85% of total installed RAM). Useful after the user manually
/// fiddles with the caps and wants to restart from a known-good point.
#[tauri::command]
pub async fn reset_memory_caps(state: State<'_, AppState>) -> CmdResult<PersistentConfig> {
    let info = sysinfo::probe();
    let mut g = state.config.lock().await;
    g.server.wired_limit_gb = Some(info.recommended.wired_limit_gb);
    g.server.cache_limit_gb = Some(info.recommended.cache_limit_gb);
    g.server.memory_limit_gb = Some(info.recommended.memory_limit_gb);
    g.server.disable_wired_limit = false;
    g.save().map_err(err)?;
    Ok(g.clone())
}

// ── Models ─────────────────────────────────────────────────────────────

#[tauri::command]
pub async fn list_models(state: State<'_, AppState>) -> CmdResult<Vec<ModelEntry>> {
    let g = state.config.lock().await;
    let dir = g.models_dir.clone();
    drop(g);
    let cat = state.catalog.lock().await;
    let entries = models::scan_local(&dir, &cat).map_err(err)?;
    Ok(entries)
}

#[tauri::command]
pub async fn get_catalog(state: State<'_, AppState>) -> CmdResult<Catalog> {
    Ok(state.catalog.lock().await.clone())
}

/// Re-fetch the catalog by re-spawning `lumen-server --catalog`. Useful after
/// the user installs a new server binary or changes `server_binary_path`.
#[tauri::command]
pub async fn refresh_catalog(state: State<'_, AppState>) -> CmdResult<Catalog> {
    let g = state.config.lock().await;
    let explicit = g.server_binary_path.clone();
    drop(g);
    let bin = server::resolve_binary_public(explicit.as_deref()).map_err(err)?;
    let cat = crate::catalog::fetch(&bin).map_err(err)?;
    let mut slot = state.catalog.lock().await;
    *slot = cat.clone();
    Ok(cat)
}

#[tauri::command]
pub async fn set_active_model(
    state: State<'_, AppState>,
    model_id: String,
) -> CmdResult<PersistentConfig> {
    // Diffusion image models (`flux2-dev`) are synthetic catalog ids — they are
    // NOT a single on-disk directory (the diffusion backend assembles them from
    // several component repos), so they never appear in `scan_local`. Promote
    // them to active directly, bypassing the on-disk readiness check below.
    {
        let cat = state.catalog.lock().await;
        if cat.is_image_model(&model_id) {
            drop(cat);
            let mut g = state.config.lock().await;
            // Image models occupy their own slot (independent of the chat model)
            // so a chat + image pair can be active at once → hybrid serve.
            // Clicking the already-active image model toggles it back off.
            g.active_image_model = if g.active_image_model.as_deref() == Some(model_id.as_str()) {
                None
            } else {
                Some(model_id)
            };
            g.save().map_err(err)?;
            return Ok(g.clone());
        }
    }
    // Validate the model is on disk AND its download completed cleanly
    // before promoting it to active. Without this, a truncated shard
    // can be selected as the active model and the server crashes
    // mid-load with a misleading "missing weight" error.
    {
        let g = state.config.lock().await;
        let models_dir = g.models_dir.clone();
        drop(g);
        let cat = state.catalog.lock().await;
        let entries = models::scan_local(&models_dir, &cat).map_err(err)?;
        match entries.iter().find(|e| e.id == model_id) {
            Some(e) if !e.ready => {
                return Err(format!(
                    "model '{model_id}' download is incomplete — re-download required before use",
                ));
            }
            None => {
                return Err(format!(
                    "model '{model_id}' not found in {}",
                    models_dir.display()
                ));
            }
            _ => {}
        }
    }
    let mut g = state.config.lock().await;
    // Clicking the already-active chat model toggles it off (lets the user run
    // image-only by deselecting the LLM).
    g.active_model = if g.active_model.as_deref() == Some(model_id.as_str()) {
        None
    } else {
        Some(model_id)
    };
    g.save().map_err(err)?;
    Ok(g.clone())
}

#[tauri::command]
pub async fn add_local_model(
    state: State<'_, AppState>,
    src_path: String,
    id: String,
) -> CmdResult<Vec<ModelEntry>> {
    use std::path::PathBuf;
    let g = state.config.lock().await;
    let dst_root = g.models_dir.clone();
    drop(g);
    let src = PathBuf::from(&src_path);
    let dst = models::local_path_for(&dst_root, &id);
    if !src.exists() {
        return Err(format!("src not found: {}", src.display()));
    }
    std::fs::create_dir_all(&dst_root).ok();
    if dst.exists() {
        return Err(format!("destination already exists: {}", dst.display()));
    }
    // Symlink the existing directory so we don't double-spend disk; users on
    // SMB / external drives can always remove + re-add.
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(&src, &dst).map_err(err)?;
        let cat = state.catalog.lock().await;
        models::scan_local(&dst_root, &cat).map_err(err)
    }
    #[cfg(not(unix))]
    {
        let _ = (src, dst);
        Err("symlink import not supported on this platform".into())
    }
}

#[tauri::command]
pub async fn delete_model(state: State<'_, AppState>, model_id: String) -> CmdResult<()> {
    let g = state.config.lock().await;
    let dir = g.models_dir.clone();
    drop(g);
    models::delete(&dir, &model_id).await.map_err(err)
}

#[tauri::command]
pub async fn download_model(
    app: AppHandle,
    state: State<'_, AppState>,
    repo_id: String,
    files: Option<Vec<String>>,
) -> CmdResult<()> {
    let g = state.config.lock().await;
    let dir = g.models_dir.clone();
    drop(g);
    let (tx, mut rx) = tokio::sync::mpsc::channel::<DownloadProgress>(32);
    let app_emit = app.clone();
    let pump = tokio::spawn(async move {
        while let Some(p) = rx.recv().await {
            let _ = app_emit.emit("lumen://download", p);
        }
    });
    let res = models::download(&dir, &repo_id, files, tx).await;
    let _ = pump.await;
    // Successful (re-)download clears the outdated flag — the SHA marker we
    // just wrote inside the download is the new ground truth.
    if res.is_ok() {
        state.outdated_models.lock().await.remove(&repo_id);
    }
    res.map(|_| ()).map_err(err)
}

/// Check each provided installed model against its HF Hub `main` commit SHA.
/// Returns a status entry per model.  Side-effect: updates
/// `AppState.outdated_models` so `start_server` can refuse to launch with a
/// stale active model.
///
/// Behaviour:
/// - `local_sha` known + `remote_sha` known + differ → `needs_update = true`
/// - `local_sha` missing (legacy install) OR `remote_sha` fetch failed
///   (offline / repo gone) → `needs_update = false` (don't block the user)
///
/// Network: one parallel HTTPS call per repo, ~50 ms each on a warm connection,
/// so checking a dozen installed models takes well under a second.
#[tauri::command]
pub async fn check_model_updates(
    state: State<'_, AppState>,
    repo_ids: Vec<String>,
) -> CmdResult<Vec<models::UpdateStatus>> {
    let g = state.config.lock().await;
    let dir = g.models_dir.clone();
    drop(g);

    let cat = state.catalog.lock().await;
    let entries = models::scan_local(&dir, &cat).map_err(err)?;
    drop(cat);

    // 8s timeout — anonymous HF API replies in <200ms; if we hit the cap
    // something is wrong with the network and we'd rather skip the update
    // check than wedge the UI. `hf_client` attaches HF_TOKEN automatically
    // when set, so private/gated repos work for users who configured one.
    let client =
        models::hf_client(Some(std::time::Duration::from_secs(8))).map_err(|e| e.to_string())?;

    let mut results = Vec::with_capacity(repo_ids.len());
    let mut new_outdated = std::collections::HashSet::new();
    for repo_id in &repo_ids {
        // Only HF-style ids (`org/repo`) can be checked against the Hub.
        // Local-only model names are skipped silently.
        let is_hf_id = repo_id.contains('/');
        let local_sha = entries
            .iter()
            .find(|m| &m.id == repo_id)
            .and_then(|m| m.local_sha.clone());

        let remote_sha = if is_hf_id {
            match models::fetch_hub_sha(&client, repo_id).await {
                Ok(s) => Some(s),
                Err(e) => {
                    eprintln!("lumen-app: hub SHA fetch for {repo_id} failed: {e}");
                    None
                }
            }
        } else {
            None
        };

        let needs_update = matches!(
            (local_sha.as_deref(), remote_sha.as_deref()),
            (Some(l), Some(r)) if l != r
        );
        if needs_update {
            new_outdated.insert(repo_id.clone());
        }
        results.push(models::UpdateStatus {
            repo_id: repo_id.clone(),
            local_sha,
            remote_sha,
            needs_update,
        });
    }

    *state.outdated_models.lock().await = new_outdated;
    Ok(results)
}

// ── Server lifecycle ───────────────────────────────────────────────────

#[tauri::command]
pub async fn start_server(app: AppHandle, state: State<'_, AppState>) -> CmdResult<ServerStatus> {
    let g = state.config.lock().await;
    let cfg = g.clone();
    // Two independent slots: a chat/LLM model and an image/diffusion model.
    // Either or both may be set → chat, image, or hybrid serve.
    let chat_id = g.active_model.clone();
    let image_id = g.active_image_model.clone();
    let models_dir = g.models_dir.clone();
    let sup = state.supervisor.clone();
    drop(g);

    if chat_id.is_none() && image_id.is_none() {
        return Err(
            "no active model — pick a chat and/or image model in the MODELS card first".to_string(),
        );
    }

    // Hard-gate the launch when the most-recent revision check flagged the
    // active chat model as out of date.  The frontend already greys out the
    // Start button in that state — this is the belt-and-suspenders guard for
    // direct RPC bypass (CLI testing, future plugin, etc.) so the engine never
    // loads weights against a tokenizer/config that has since been re-uploaded.
    if let Some(ref cid) = chat_id {
        if state.outdated_models.lock().await.contains(cid) {
            return Err(format!(
                "active model `{cid}` is out of date — open the MODELS card \
                 and click Update first, then start the server"
            ));
        }
    }

    // Resolve the on-disk path of the active chat model. Flat-layout dirs use
    // the dir name (e.g. `gemma-4-26b-a4b-mlx-imatrix3plus-awq`) which won't
    // match the HF Hub id (`hsng95/gemma-4-26b-a4b-mlx-imatrix3plus-awq`). When
    // the model is found locally we pass the absolute path as MODEL_ID — the MLX
    // native runner and tokenizer loader both detect `is_dir()` and skip HF Hub
    // entirely. The diffusion backend resolves its own component repos from
    // IMAGE_MODEL_ID, so the image slot needs no path resolution here.
    let cat = state.catalog.lock().await;
    let (model_arg, active_bytes) = if let Some(ref cid) = chat_id {
        let entries = models::scan_local(&models_dir, &cat).map_err(err)?;
        match entries.iter().find(|m| m.id == *cid) {
            // MODEL_ID is the local path; every loader takes the weights
            // from it.
            Some(entry) => (
                entry.path.to_string_lossy().into_owned(),
                Some(entry.size_bytes),
            ),
            None => {
                // Not on disk — resolve to the canonical HF id so the server
                // can fetch.
                let id = cat
                    .find_recommended(cid)
                    .map(|r| r.id.clone())
                    .unwrap_or_else(|| cid.clone());
                (id, None)
            }
        }
    } else {
        // Image-only: no LLM to resolve. Empty MODEL_ID → server omits it.
        (String::new(), None)
    };
    drop(cat);

    let serve = match (chat_id.is_some(), image_id.is_some()) {
        (true, true) => server::ServeKind::Hybrid,
        (false, true) => server::ServeKind::Image,
        (true, false) => server::ServeKind::Chat,
        // Guarded by the early `is_none() && is_none()` return above.
        (false, false) => unreachable!(),
    };

    sup.start(
        app,
        &cfg,
        &model_arg,
        active_bytes,
        image_id.as_deref(),
        serve,
    )
    .await
    .map_err(err)
}

#[tauri::command]
pub async fn stop_server(app: AppHandle, state: State<'_, AppState>) -> CmdResult<ServerStatus> {
    let sup = state.supervisor.clone();
    sup.stop(app).await.map_err(err)
}

#[tauri::command]
pub async fn server_status(state: State<'_, AppState>) -> CmdResult<ServerStatus> {
    Ok(state.supervisor.status().await)
}

#[derive(Debug, Serialize)]
pub struct ServerMetrics {
    pub tokens_per_sec: Option<f64>,
    pub ms_per_step: Option<f64>,
    pub kv_cache_mb: Option<u64>,
    pub requests_per_min: Option<u32>,
}

/// Live decode metrics — EMA-smoothed tok/s + ms-per-step + requests-per-min
/// aggregated from lumen-server stderr by `ServerSupervisor`. Values stay
/// `None` until the first chat request finishes. `kv_cache_mb` is reserved
/// for a future `/v1/stats` endpoint and is always `None` for now.
#[tauri::command]
pub async fn server_metrics(state: State<'_, AppState>) -> CmdResult<ServerMetrics> {
    let snap = state.supervisor.metrics().await;
    Ok(ServerMetrics {
        tokens_per_sec: snap.tokens_per_sec,
        ms_per_step: snap.ms_per_step,
        kv_cache_mb: None,
        requests_per_min: snap.requests_per_min,
    })
}

// ── Text-to-image generation (diffusion proxy) ──────────────────────

/// One generated image, returned as a base-64 PNG (OpenAI `images` shape).
#[derive(Debug, Serialize)]
pub struct GeneratedImage {
    /// Base-64-encoded PNG bytes (no `data:` prefix). The frontend wraps this
    /// in `data:image/png;base64,<…>` for `<img src>` + download.
    pub b64_json: String,
}

/// Proxy a text-to-image request to the locally-running server's
/// `POST /v1/images/generations` endpoint (image mode). Runs the HTTP call
/// from Rust so the Tauri webview never has to fetch a custom localhost port
/// itself (avoids any CSP / mixed-content surprises across platforms).
///
/// The server must already be Running in image mode — `start_server` does that
/// when the active model is a diffusion catalog id. Generation is slow
/// (minutes on 36 GB), so the per-request timeout is generous.
#[tauri::command]
pub async fn generate_image(
    state: State<'_, AppState>,
    prompt: String,
    size: String,
    steps: u32,
    seed: i64,
    guidance: f32,
) -> CmdResult<GeneratedImage> {
    let status = state.supervisor.status().await;
    if status.state != server::LifecycleState::Running {
        return Err("server is not running — start it in image mode first".to_string());
    }
    // Loopback host for the request even if the server binds 0.0.0.0 / ::.
    let host = match status.host.as_str() {
        "0.0.0.0" | "::" | "" => "127.0.0.1",
        h => h,
    };
    let url = format!("http://{host}:{}/v1/images/generations", status.port);

    let body = serde_json::json!({
        "prompt": prompt,
        "size": size,
        "steps": steps,
        "seed": seed,
        "guidance": guidance,
    });

    let client = reqwest::Client::builder()
        // Image generation can take several minutes; don't cut it short.
        .timeout(std::time::Duration::from_secs(900))
        .build()
        .map_err(err)?;

    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;

    if !resp.status().is_success() {
        let code = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("image generation failed ({code}): {text}"));
    }

    let parsed: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("invalid response JSON: {e}"))?;
    let b64 = parsed
        .get("data")
        .and_then(|d| d.get(0))
        .and_then(|d| d.get("b64_json"))
        .and_then(|b| b.as_str())
        .ok_or_else(|| "response missing data[0].b64_json".to_string())?;

    Ok(GeneratedImage {
        b64_json: b64.to_string(),
    })
}

// ── Doctor (preflight diagnostics) ──────────────────────────────────

#[tauri::command]
pub async fn doctor_run(state: State<'_, AppState>) -> CmdResult<DoctorReport> {
    let g = state.config.lock().await;
    let cfg = g.clone();
    drop(g);
    Ok(doctor::run_all(&cfg).await)
}

#[tauri::command]
pub async fn doctor_fix(state: State<'_, AppState>, action: String) -> CmdResult<String> {
    let g = state.config.lock().await;
    let cfg = g.clone();
    drop(g);
    doctor::try_fix(&action, &cfg).await
}

/// How long the SSD read test may take before the app stops waiting for it. The
/// test's own budget is shorter (`ProbePlan::DEFAULT`); this is for a volume
/// that stalls inside a single read, which no between-reads check can interrupt.
const SSD_PROBE_HARD_LIMIT: Duration = Duration::from_secs(9);

/// Runs the SSD read test on a worker thread and stops waiting for it after
/// [`SSD_PROBE_HARD_LIMIT`]. `None`: cancelled. Besides the row it returns what was measured, for
/// the caller to store.
async fn run_ssd_probe(
    cfg: PersistentConfig,
    cancel: Arc<AtomicBool>,
) -> CmdResult<Option<(CheckResult, Option<doctor::SsdReading>)>> {
    let worker = {
        let cancel = Arc::clone(&cancel);
        tokio::task::spawn_blocking(move || doctor::probe_flash_next_ssd(&cfg, &cancel))
    };
    match tokio::time::timeout(SSD_PROBE_HARD_LIMIT, worker).await {
        Ok(joined) => joined.map_err(err),
        Err(_) => {
            // The read cannot be interrupted, but the thread stops at its next check.
            cancel.store(true, Ordering::Relaxed);
            Ok(Some((doctor::flash_next_ssd_stalled(), None)))
        }
    }
}

/// Takes the slot that says an SSD test (the Doctor's or the Measure action) is running, and
/// hands back its cancel flag. A second test while one runs is refused.
fn begin_ssd_test(state: &AppState) -> CmdResult<Arc<AtomicBool>> {
    let cancel = Arc::new(AtomicBool::new(false));
    let mut slot = state.ssd_probe.lock().unwrap_or_else(|p| p.into_inner());
    if slot.is_some() {
        return Err("the SSD speed test is already running".to_string());
    }
    *slot = Some(Arc::clone(&cancel));
    Ok(cancel)
}

fn end_ssd_test(state: &AppState) {
    *state.ssd_probe.lock().unwrap_or_else(|p| p.into_inner()) = None;
}

/// Stores what the SSD test measured (and when), next to the folder it read.
async fn store_ssd_reading(state: &AppState, reading: &doctor::SsdReading) -> CmdResult<()> {
    let mut g = state.config.lock().await;
    g.measured_hardware.record_ssd(
        reading.random_gbps,
        reading.sequential_gbps,
        &reading.volume,
        hardware::now_secs(),
    );
    g.save().map_err(err)
}

/// Flash-Next's SSD read test (~1 GB). Separate from `doctor_run` because it is
/// slow and heavy on the disk: it runs only when the user asks. `None` means it
/// was cancelled; the panel then leaves the report alone. What it measured is stored,
/// as the Flash-Next card's Measure action does, for the decode-speed estimate.
#[tauri::command]
pub async fn doctor_probe_ssd(state: State<'_, AppState>) -> CmdResult<Option<CheckResult>> {
    let cfg = state.config.lock().await.clone();
    let cancel = begin_ssd_test(&state)?;
    let out = run_ssd_probe(cfg, cancel).await;
    end_ssd_test(&state);
    match out? {
        Some((row, reading)) => {
            if let Some(reading) = reading {
                store_ssd_reading(&state, &reading).await?;
            }
            Ok(Some(row))
        }
        None => Ok(None),
    }
}

#[tauri::command]
pub async fn doctor_probe_cancel(state: State<'_, AppState>) -> CmdResult<()> {
    cancel_ssd_test(&state);
    Ok(())
}

fn cancel_ssd_test(state: &AppState) {
    if let Some(cancel) = state
        .ssd_probe
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .as_ref()
    {
        cancel.store(true, Ordering::Relaxed);
    }
}

// ── Measuring this Mac ─────────────────────────────────────────────────

fn emit_measure(app: &AppHandle, progress: &MeasureProgress) {
    let _ = app.emit(hardware::EVENT_MEASURE, progress);
}

/// Measures this Mac for the decode-speed estimate: its memory bandwidth (`lumen-server
/// --measure-memory-bandwidth`, a second or two), then the SSD's random-read speed on the
/// Flash-Next model's own folder (the Doctor's test; skipped when the model is not installed).
/// Each number is stored as soon as it is known, so a failure or a cancel in the second part
/// keeps the first, and a failing memory test keeps the numbers measured before. Progress goes out
/// as `lumen://measure` events and the final report is the return value.
#[tauri::command]
pub async fn measure_hardware(
    app: AppHandle,
    state: State<'_, AppState>,
) -> CmdResult<MeasureProgress> {
    let cancel = begin_ssd_test(&state)?;
    let report = measure(&app, &state, &cancel).await;
    end_ssd_test(&state);
    emit_measure(&app, &report);
    Ok(report)
}

async fn measure(app: &AppHandle, state: &AppState, cancel: &Arc<AtomicBool>) -> MeasureProgress {
    let cfg = state.config.lock().await.clone();

    emit_measure(app, &MeasureProgress::running(MeasureStage::Memory));
    let explicit = cfg.server_binary_path.clone();
    let memory: Result<f64, String> = match server::resolve_binary_public(explicit.as_deref()) {
        Ok(bin) => hardware::run_memory_probe(&bin, hardware::MEMORY_PROBE_TIMEOUT)
            .await
            .map(|p| p.gpu_read_gbps)
            .map_err(|e| format!("{e:#}")),
        Err(e) => Err(format!("{e:#}")),
    };
    let memory = match memory {
        Ok(gbps) => {
            let mut g = state.config.lock().await;
            g.measured_hardware
                .record_memory(gbps, hardware::now_secs());
            g.save()
                .map(|()| gbps)
                .map_err(|e| format!("could not save the result: {e:#}"))
        }
        Err(e) => Err(e),
    };
    if cancel.load(Ordering::Relaxed) {
        return hardware::finish(memory, SsdOutcome::Cancelled, true);
    }

    let ssd = if flash_next::find_model_dir(&cfg).is_none() {
        SsdOutcome::NotInstalled
    } else {
        emit_measure(app, &MeasureProgress::running(MeasureStage::Ssd));
        match run_ssd_probe(cfg, Arc::clone(cancel)).await {
            Ok(Some((_, Some(reading)))) => match store_ssd_reading(state, &reading).await {
                Ok(()) => SsdOutcome::Measured(reading.random_gbps),
                Err(e) => SsdOutcome::Failed(e),
            },
            Ok(Some((row, None))) => SsdOutcome::Failed(row.message),
            Ok(None) => SsdOutcome::Cancelled,
            Err(e) => SsdOutcome::Failed(e),
        }
    };
    hardware::finish(memory, ssd, cancel.load(Ordering::Relaxed))
}

/// Stops the Measure action: the SSD part ends at its next read, and the memory part (a second or
/// two) is let finish.
#[tauri::command]
pub async fn cancel_measure(state: State<'_, AppState>) -> CmdResult<()> {
    cancel_ssd_test(&state);
    Ok(())
}

// ── Flash-Next expert pack ─────────────────────────────────────────────

/// Whether the server is up (or coming up): a pack build reads the whole model from the same
/// disk and is refused then.
async fn server_busy(state: &AppState) -> bool {
    matches!(
        state.supervisor.status().await.state,
        server::LifecycleState::Running | server::LifecycleState::Starting
    )
}

/// What the Flash-Next settings show about the model on disk: the pack, the draft head, the
/// space the pack needs, and whether a build may start.
#[tauri::command]
pub async fn flash_next_status(state: State<'_, AppState>) -> CmdResult<FlashNextStatus> {
    let cfg = state.config.lock().await.clone();
    let busy = server_busy(&state).await;
    let building = state.pack_builder.is_running().await;
    tokio::task::spawn_blocking(move || pack_build::status(&cfg, busy, building))
        .await
        .map_err(err)
}

/// Start building the expert pack in the background (`lumen-server --pack-experts <model dir>`,
/// the same binary `start_server` resolves). Progress and the outcome arrive as `lumen://pack`
/// events; the refusals (server running, under 70 GB free, already built) come back as errors.
#[tauri::command]
pub async fn start_pack_build(app: AppHandle, state: State<'_, AppState>) -> CmdResult<()> {
    let cfg = state.config.lock().await.clone();
    let busy = server_busy(&state).await;
    let building = state.pack_builder.is_running().await;
    let (status, bin) = tokio::task::spawn_blocking(move || {
        let status = pack_build::status(&cfg, busy, building);
        let bin = server::resolve_binary_public(cfg.server_binary_path.as_deref());
        (status, bin)
    })
    .await
    .map_err(err)?;
    if let Some(block) = status.build_block {
        return Err(pack_build::block_message(block, status.free_bytes));
    }
    let dir = status
        .dir
        .ok_or_else(|| "Flash-Next is not installed".to_string())?;
    let bin = bin.map_err(err)?;
    let emitter = app.clone();
    let emit: pack_build::Emit = Arc::new(move |progress| {
        let _ = emitter.emit(EVENT_PACK, progress);
    });
    state
        .pack_builder
        .start(bin, std::path::PathBuf::from(dir), emit)
        .await
        .map_err(err)
}

/// Stop the running pack build; it removes its partial file. `false` when none runs.
#[tauri::command]
pub async fn cancel_pack_build(state: State<'_, AppState>) -> CmdResult<bool> {
    Ok(state.pack_builder.cancel().await)
}
