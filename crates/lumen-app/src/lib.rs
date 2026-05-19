//! Library entry — re-exports the modules used by `src/main.rs` so they're
//! also reachable from `examples/*` (which can only reference the crate's
//! public surface, not bin-private modules).

pub mod catalog;
pub mod config;
pub mod doctor;
pub mod models;
pub mod server;
pub mod sysinfo;
