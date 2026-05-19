//! Manual update flow — wraps `tauri-plugin-updater` so the frontend can
//! drive the UX (no built-in dialog popups). Three commands:
//!
//! - `current_version` — what's running now (for "Lumen v0.1.0" display)
//! - `check_for_updates` — hits the configured endpoint, returns metadata
//! - `install_update` — downloads + applies + asks the process plugin to
//!   restart. Emits `lumen://update-progress` events during the download.
//!
//! Sidecar lumen-server ships inside the same .app bundle, so a Tauri update
//! atomically refreshes both binaries (no version-skew window).

use serde::Serialize;
use tauri::{AppHandle, Emitter};
use tauri_plugin_updater::UpdaterExt;

pub const EVENT_PROGRESS: &str = "lumen://update-progress";

type CmdResult<T> = Result<T, String>;

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

#[derive(Debug, Clone, Serialize)]
pub struct UpdateInfo {
    pub available: bool,
    pub current_version: String,
    pub latest_version: Option<String>,
    pub release_notes: Option<String>,
    pub published_at: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct UpdateProgress {
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
    pub done: bool,
}

#[tauri::command]
pub async fn current_version(app: AppHandle) -> CmdResult<String> {
    Ok(app.package_info().version.to_string())
}

#[tauri::command]
pub async fn check_for_updates(app: AppHandle) -> CmdResult<UpdateInfo> {
    let updater = app.updater().map_err(err)?;
    let current = app.package_info().version.to_string();
    match updater.check().await.map_err(err)? {
        Some(u) => Ok(UpdateInfo {
            available: true,
            current_version: current,
            latest_version: Some(u.version.clone()),
            release_notes: u.body.clone(),
            published_at: u.date.map(|d| d.to_string()),
        }),
        None => Ok(UpdateInfo {
            available: false,
            current_version: current,
            latest_version: None,
            release_notes: None,
            published_at: None,
        }),
    }
}

#[tauri::command]
pub async fn install_update(app: AppHandle) -> CmdResult<()> {
    let updater = app.updater().map_err(err)?;
    let update = updater
        .check()
        .await
        .map_err(err)?
        .ok_or_else(|| "no update available".to_string())?;

    let app_for_progress = app.clone();
    let mut downloaded: u64 = 0;
    update
        .download_and_install(
            move |chunk, total| {
                downloaded += chunk as u64;
                let _ = app_for_progress.emit(
                    EVENT_PROGRESS,
                    UpdateProgress {
                        downloaded_bytes: downloaded,
                        total_bytes: total,
                        done: false,
                    },
                );
            },
            move || {
                // No-op: the plugin will restart the process via the process
                // plugin below, and the frontend can listen for the `done`
                // progress event before triggering its own restart prompt.
            },
        )
        .await
        .map_err(err)?;

    let _ = app.emit(
        EVENT_PROGRESS,
        UpdateProgress {
            downloaded_bytes: downloaded,
            total_bytes: Some(downloaded),
            done: true,
        },
    );

    // Restart so the new bundle (including the sidecar lumen-server) takes
    // over. The frontend should warn the user before invoking this so they
    // can stop the server first.
    app.restart();
}
