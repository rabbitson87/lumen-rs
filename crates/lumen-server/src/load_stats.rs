//! Process-lifetime serving counters for the `GET /v1/loads` observability
//! endpoint (WS-F #2). Pure JSON, no Prometheus.
//!
//! All counters are lock-free `AtomicU64`s bumped at the chat
//! generation-completion points in `engine.rs` (both the streaming and
//! non-streaming paths). The route handler reads them via a shared
//! `Arc<ServerLoadStats>` that is created once at startup and cloned into
//! both the `InferenceEngine` (writer) and the `EngineHandle` (reader).
//!
//! `last_tok_per_sec` is stored as `tok/s * 1000` in an integer so the
//! whole struct stays lock-free; the route divides by 1000.0 on read.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use serde::Serialize;

/// Lock-free lifetime accumulators shared between the engine (writer) and
/// the `/v1/loads` route (reader).
pub struct ServerLoadStats {
    /// Process start; `uptime_s` is derived from this on read.
    start_instant: Instant,
    /// Active model id (set once at startup).
    model_id: String,
    requests_total: AtomicU64,
    lifetime_prompt_tokens: AtomicU64,
    lifetime_gen_tokens: AtomicU64,
    /// Most recent decode throughput, stored as `tok/s * 1000` to keep the
    /// field a plain integer atomic.
    last_tok_per_sec_milli: AtomicU64,
}

impl ServerLoadStats {
    /// Build a fresh shared accumulator anchored at "now".
    pub fn new_arc(model_id: impl Into<String>) -> Arc<Self> {
        Arc::new(Self {
            start_instant: Instant::now(),
            model_id: model_id.into(),
            requests_total: AtomicU64::new(0),
            lifetime_prompt_tokens: AtomicU64::new(0),
            lifetime_gen_tokens: AtomicU64::new(0),
            last_tok_per_sec_milli: AtomicU64::new(0),
        })
    }

    /// Record one completed chat generation. Cheap (relaxed atomics) so it
    /// is safe to call on the streaming done path. `tok_per_sec` may be 0.0
    /// when throughput is not measurable (e.g. a stop-sequence early break).
    pub fn record(&self, prompt_tokens: u64, gen_tokens: u64, tok_per_sec: f64) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.lifetime_prompt_tokens
            .fetch_add(prompt_tokens, Ordering::Relaxed);
        self.lifetime_gen_tokens
            .fetch_add(gen_tokens, Ordering::Relaxed);
        if tok_per_sec > 0.0 {
            let milli = (tok_per_sec * 1000.0).round() as u64;
            self.last_tok_per_sec_milli.store(milli, Ordering::Relaxed);
        }
    }

    /// Serializable point-in-time view for the `/v1/loads` route.
    pub fn snapshot(&self) -> LoadStatsSnapshot {
        LoadStatsSnapshot {
            model: self.model_id.clone(),
            uptime_s: self.start_instant.elapsed().as_secs_f64(),
            requests_total: self.requests_total.load(Ordering::Relaxed),
            lifetime_prompt_tokens: self.lifetime_prompt_tokens.load(Ordering::Relaxed),
            lifetime_gen_tokens: self.lifetime_gen_tokens.load(Ordering::Relaxed),
            last_tok_per_sec: self.last_tok_per_sec_milli.load(Ordering::Relaxed) as f64 / 1000.0,
            // Single-tenant sequential engine: at most one generation runs at
            // a time, and counters are read between requests. `0` is the
            // honest steady-state value; live concurrency tracking would need
            // an in-flight gauge the sequential engine does not maintain.
            active_seqs: Some(0),
            // Placeholder (WS4): KV-cache byte accounting is not surfaced by
            // the backends yet. Emitted as JSON null.
            kv_cache_mb: None,
        }
    }
}

/// JSON body of `GET /v1/loads`.
#[derive(Serialize)]
pub struct LoadStatsSnapshot {
    pub model: String,
    pub uptime_s: f64,
    pub requests_total: u64,
    pub lifetime_prompt_tokens: u64,
    pub lifetime_gen_tokens: u64,
    pub last_tok_per_sec: f64,
    pub active_seqs: Option<u64>,
    pub kv_cache_mb: Option<f64>,
}
