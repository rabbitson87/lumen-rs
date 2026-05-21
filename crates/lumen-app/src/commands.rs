use serde::Serialize;
use tauri::{AppHandle, Emitter, State};

use std::collections::BTreeMap;

use crate::catalog::Catalog;
use crate::config::{
    AdvancedConfig, ContextConfig, PersistentConfig, QuantConfig, ServerConfig, config_dir,
};
use crate::doctor::{self, DoctorReport};
use crate::models::{self, DownloadProgress, ModelEntry};
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

#[tauri::command]
pub async fn update_server_config(
    state: State<'_, AppState>,
    server: ServerConfig,
) -> CmdResult<PersistentConfig> {
    let mut g = state.config.lock().await;
    g.server = server;
    g.save().map_err(err)?;
    Ok(g.clone())
}

#[tauri::command]
pub async fn update_quant_config(
    state: State<'_, AppState>,
    quant: QuantConfig,
) -> CmdResult<PersistentConfig> {
    let mut g = state.config.lock().await;
    g.quant = quant;
    g.save().map_err(err)?;
    Ok(g.clone())
}

#[tauri::command]
pub async fn update_context_config(
    state: State<'_, AppState>,
    context: ContextConfig,
) -> CmdResult<PersistentConfig> {
    let mut g = state.config.lock().await;
    g.context = context;
    g.save().map_err(err)?;
    Ok(g.clone())
}

#[tauri::command]
pub async fn update_advanced_config(
    state: State<'_, AppState>,
    advanced: AdvancedConfig,
) -> CmdResult<PersistentConfig> {
    let mut g = state.config.lock().await;
    g.advanced = advanced;
    g.save().map_err(err)?;
    Ok(g.clone())
}

#[tauri::command]
pub async fn update_env_overrides(
    state: State<'_, AppState>,
    env_overrides: BTreeMap<String, String>,
) -> CmdResult<PersistentConfig> {
    let mut g = state.config.lock().await;
    g.env_overrides = env_overrides;
    g.save().map_err(err)?;
    Ok(g.clone())
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
    g.active_model = Some(model_id);
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
        return models::scan_local(&dst_root, &cat).map_err(err);
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

    let client = reqwest::Client::builder()
        .user_agent(concat!("lumen-app/", env!("CARGO_PKG_VERSION")))
        .timeout(std::time::Duration::from_secs(8))
        .build()
        .map_err(|e| e.to_string())?;

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
    let mut cfg = g.clone();
    let active_id = g
        .active_model
        .clone()
        .ok_or_else(|| "no active model — pick one in the MODELS card first".to_string())?;
    let models_dir = g.models_dir.clone();
    let sup = state.supervisor.clone();
    drop(g);

    // Hard-gate the launch when the most-recent revision check flagged the
    // active model as out of date.  The frontend already greys out the Start
    // button in that state — this is the belt-and-suspenders guard for direct
    // RPC bypass (CLI testing, future plugin, etc.) so the engine never loads
    // weights against a tokenizer/config that has since been re-uploaded.
    if state.outdated_models.lock().await.contains(&active_id) {
        return Err(format!(
            "active model `{active_id}` is out of date — open the MODELS card \
             and click Update first, then start the server"
        ));
    }

    // Resolve the on-disk path of the active model. Flat-layout dirs use the
    // dir name (e.g. `gemma-4-26b-a4b-mlx-imatrix3plus-awq`) which won't match the HF Hub
    // id (`hsng95/gemma-4-26b-a4b-mlx-imatrix3plus-awq`). When the model is found
    // locally we pass the absolute path as MODEL_ID — the MLX native runner
    // and tokenizer loader both detect `is_dir()` and skip HF Hub entirely.
    let cat = state.catalog.lock().await;
    let entries = models::scan_local(&models_dir, &cat).map_err(err)?;
    let active_entry = entries.iter().find(|m| m.id == active_id);

    let (model_arg, active_bytes) = if let Some(entry) = active_entry {
        // Mirror into local_model_dir too — engine.rs reads LUMEN_GEMMA4_DIR /
        // LUMEN_QWEN35_SHARDS for the non-MLX (Candle) Gemma4Native /
        // Qwen35Moe paths.
        if cfg.server.local_model_dir.is_none() {
            cfg.server.local_model_dir = Some(entry.path.clone());
        }
        (
            entry.path.to_string_lossy().into_owned(),
            Some(entry.size_bytes),
        )
    } else {
        // Not on disk — resolve to the canonical HF id so the server can fetch.
        let id = cat
            .find_recommended(&active_id)
            .map(|r| r.id.clone())
            .unwrap_or_else(|| active_id.clone());
        (id, None)
    };
    drop(cat);

    sup.start(app, &cfg, &model_arg, active_bytes)
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
