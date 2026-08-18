//! The server's library half.
//!
//! `main.rs` is the binary: argument/env resolution, the tokio accept loop and
//! connection dispatch. Everything it drives lives here.
//!
//! ## Why this exists
//!
//! The crate used to be binary-only, which meant nothing outside it could
//! reach [`types`] — 36 `Deserialize` derives covering the entire OpenAI and
//! Anthropic request surface, i.e. the exact bytes an untrusted client sends.
//! An integration test or a fuzz target cannot link a `[[bin]]`, so that
//! surface had no way to be tested from outside the crate at all, and the
//! `tool-choice-none` and `anthropic-turn-images` defects recorded in
//! `xtask/src/red_green.rs` both landed in it.
//!
//! Splitting costs nothing at runtime — the binary is a thin wrapper over the
//! same code — and makes the request types reachable from
//! `tests/` and from `fuzz/`.

pub mod catalog;
pub mod diffusion_engine;
pub mod embedding;
pub mod engine;
pub mod load_stats;
pub mod routes;
pub mod types;
