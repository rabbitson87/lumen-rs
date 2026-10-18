//! Library entry — the modules `src/main.rs` builds the app from, public so
//! they're also reachable from `examples/*` (which can only reference the
//! crate's public surface, not bin-private modules). The bin imports them
//! rather than declaring them again, so each is compiled, and tested, once.

pub mod catalog;
pub mod config;
pub mod doctor;
pub mod flash_next;
pub mod hardware;
pub mod models;
pub mod pack_build;
pub mod server;
pub mod sysinfo;
