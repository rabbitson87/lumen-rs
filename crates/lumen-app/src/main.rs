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
            commands::start_server,
            commands::stop_server,
            commands::server_status,
            commands::server_metrics,
            commands::open_config_dir,
            commands::get_system_info,
            commands::reset_memory_caps,
            commands::doctor_run,
            commands::doctor_fix,
            updater::check_for_updates,
            updater::install_update,
            updater::current_version,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Lumen");
}
