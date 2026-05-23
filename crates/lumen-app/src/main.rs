#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

mod catalog;
mod commands;
mod config;
mod doctor;
mod models;
mod server;
mod state;
mod sysinfo;
mod updater;

use state::AppState;
use tauri::Manager;

fn main() {
    // Surface HF_TOKEN detection at startup so users who configured one
    // can see it took effect (no token value is logged — only its
    // presence + source env var). Token unlocks gated HF repos and
    // raises the anonymous rate limit ceiling for periodic update checks.
    let hf_token_src = if std::env::var("HF_TOKEN")
        .ok()
        .filter(|t| !t.trim().is_empty())
        .is_some()
    {
        Some("HF_TOKEN")
    } else if std::env::var("HUGGING_FACE_HUB_TOKEN")
        .ok()
        .filter(|t| !t.trim().is_empty())
        .is_some()
    {
        Some("HUGGING_FACE_HUB_TOKEN")
    } else {
        None
    };
    match hf_token_src {
        Some(name) => {
            eprintln!("[lumen-app] HuggingFace token detected via ${name} — gated repos accessible");
        }
        None => {
            eprintln!(
                "[lumen-app] no HuggingFace token configured (HF_TOKEN / HUGGING_FACE_HUB_TOKEN); anonymous HF access only — gated repos will 401"
            );
        }
    }

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_process::init())
        .setup(|app| {
            let state = AppState::load(&app.handle())?;
            app.manage(state);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_config,
            commands::update_server_config,
            commands::update_quant_config,
            commands::update_context_config,
            commands::update_advanced_config,
            commands::update_env_overrides,
            commands::typed_env_keys,
            commands::list_models,
            commands::get_catalog,
            commands::refresh_catalog,
            commands::set_active_model,
            commands::add_local_model,
            commands::delete_model,
            commands::download_model,
            commands::check_model_updates,
            commands::start_server,
            commands::stop_server,
            commands::server_status,
            commands::server_metrics,
            commands::open_config_dir,
            commands::get_system_info,
            commands::get_memory_usage,
            commands::reset_memory_caps,
            commands::doctor_run,
            commands::doctor_fix,
            updater::check_for_updates,
            updater::install_update,
            updater::current_version,
        ])
        .build(tauri::generate_context!())
        .expect("error while building Lumen")
        .run(|app_handle, event| {
            // Tauri may std::process::exit() after this returns, which skips
            // tokio runtime drop and therefore skips Child::kill_on_drop on
            // the sidecar. Send SIGTERM → SIGKILL synchronously here so the
            // server's port + RAM are reclaimed before the app process dies.
            if let tauri::RunEvent::ExitRequested { .. } = event {
                let state: tauri::State<AppState> = app_handle.state();
                state.supervisor.shutdown_blocking();
            }
        });
}
