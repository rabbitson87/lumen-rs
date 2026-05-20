use std::collections::HashSet;
use std::sync::Arc;

use anyhow::Result;
use tauri::AppHandle;
use tokio::sync::Mutex;

use crate::catalog::Catalog;
use crate::config::PersistentConfig;
use crate::server::{self, ServerSupervisor};

/// Tauri-managed shared state. Mutex-guarded config; the supervisor owns its
/// own internal lock so two concurrent commands (e.g. status poll +
/// start_server) don't serialize on it.
pub struct AppState {
    pub config: Mutex<PersistentConfig>,
    pub supervisor: Arc<ServerSupervisor>,
    /// Cached catalog from `lumen-server --catalog`. Fetched at startup,
    /// refreshable via the `refresh_catalog` Tauri command. Defaults to an
    /// empty catalog if the binary can't be located yet — UI handles that.
    pub catalog: Mutex<Catalog>,
    /// Repo ids whose local SHA differs from the HF Hub `main` SHA. Populated
    /// by the `check_model_updates` command (and refreshed on demand from the
    /// frontend). `start_server` consults this set and refuses to launch the
    /// engine when the active model is in it — forcing the user through the
    /// "Update" flow so the engine never loads stale weights against the new
    /// tokenizer / config / catalog label.
    pub outdated_models: Mutex<HashSet<String>>,
}

impl AppState {
    pub fn load(_app: &AppHandle) -> Result<Self> {
        let cfg = PersistentConfig::load_or_default()?;
        // Best-effort initial fetch — the server binary may not be on disk
        // yet on a fresh install, and the Doctor panel surfaces that case.
        let catalog = match server::resolve_binary_public(cfg.server_binary_path.as_deref()) {
            Ok(bin) => crate::catalog::fetch(&bin).unwrap_or_default(),
            Err(_) => Catalog::default(),
        };
        Ok(Self {
            config: Mutex::new(cfg),
            supervisor: ServerSupervisor::new(),
            catalog: Mutex::new(catalog),
            outdated_models: Mutex::new(HashSet::new()),
        })
    }
}
