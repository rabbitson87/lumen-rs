//! MLX backend with three runner implementations selected via
//! `LUMEN_MLX_BACKEND`:
//!
//! - **PyO3 in-process (default, `pyo3`)**: embeds a CPython interpreter via
//!   PyO3 and imports `mlx_runner.MlxRunner` directly. No subprocess, no IPC
//!   pipe — each `prefill` / `decode_step` is a Python method call inside the
//!   Rust process, serialized through the GIL. Closes the ~21 % IPC overhead
//!   vs Phase 1. Also the only runner that supports Track A1 prefix caching
//!   (snapshot/restore/fork).
//!
//! - **Subprocess JSON-RPC (`subprocess`)**: `LUMEN_MLX_BACKEND=subprocess`
//!   or the legacy `LUMEN_MLX_SUBPROCESS=1` falls back to spawning `python
//!   mlx_runner.py` and pipes newline-delimited JSON. Kept as a debugging aid
//!   + as the supported path for environments where embedding Python
//!   (libpython linkage) is impractical.
//!
//! - **Native Rust mlx-rs (`native`)**: `LUMEN_MLX_BACKEND=native` selects
//!   `runner_native::NativeMlxRunner`, a pure-Rust port over `mlx-rs`. No
//!   Python in the process. Built only when the `mlx-native` Cargo feature is
//!   enabled. Validated against the PyO3 runner via the parity harness
//!   (`LUMEN_MLX_GOLDEN_IN`); see `.ai/memory/active/mlx-rs-native-port/` for
//!   the staged port plan and current gate status. Native runner accepts both
//!   local model directories and HF Hub repo ids — same UX as PyO3.
//!
//! The public `MlxBackend` API is identical for all three — `MlxBackend::load`
//! picks the runner based on the env var, and dispatches each method to the
//! chosen variant.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use hf_hub::api::sync::ApiBuilder;
use tokenizers::Tokenizer;

mod gemma4_backend;
mod gemma4_chat;
mod gemma4_moe;
mod gemma4_mtp;
mod gemma4_response;

/// Metal memory configuration re-exports. Used by `lumen-server` to
/// raise the wired-memory cap (mirrors mlx-lm's `wired_limit()` context).
#[cfg(feature = "mlx-native")]
pub mod metal_memory {
    pub use mlx_rs::memory::{
        clear_cache, get_active_memory, get_cache_memory, get_peak_memory, set_cache_limit,
        set_memory_limit, set_wired_limit,
    };
}

/// Public surface for the Gemma 4 26B-A4B MoE port (Phase 1 W4 (c) onwards).
///
/// Re-exports the model, chat template, and response parser from the
/// individual `gemma4_*` modules into a single namespace so dependent
/// crates (e.g. `lumen-server`) can `use lumen_mlx::gemma4::*`.
///
/// Only available with `--features mlx-native`. With the feature off the
/// re-exports vanish along with the underlying types, mirroring the
/// existing native-vs-pyo3 split.
#[cfg(feature = "mlx-native")]
pub mod gemma4 {
    pub use crate::gemma4_backend::imp::Gemma4Backend;
    pub use crate::gemma4_chat::imp::{
        ChatMessage, ChatRole, Gemma4ChatTemplate, RenderOptions, TOK_BOS, TOK_CHANNEL_CLOSE,
        TOK_CHANNEL_OPEN, TOK_EOS, TOK_PAD, TOK_QUOTE_DELIM, TOK_THINK, TOK_TOOL_RESPONSE_CLOSE,
        TOK_TOOL_RESPONSE_OPEN, TOK_TURN_CLOSE, TOK_TURN_OPEN,
    };
    pub use crate::gemma4_moe::imp::{
        Gemma4Breakdown, GenerateConfig, GenerateStats, MtpStepOutput, NativeGemma4Config,
        NativeGemma4Model, NativeGemma4PromptCache, set_forward_step, take_gemma4_breakdown,
    };
    pub use crate::gemma4_response::imp::{
        ParseState, ParsedResponse, ParsedToolCall, ResponseParser, TOK_TOOL_CALL_CLOSE,
        TOK_TOOL_CALL_OPEN,
    };

    /// MTP (Multi-Token Prediction) drafter for Gemma 4. Phase 1: types +
    /// weight loader only — forward pass and decode-loop integration are
    /// in subsequent phases. See memory `gemma4_mtp_drafter_architecture.md`.
    pub mod mtp {
        pub use crate::gemma4_mtp::imp::{
            NativeGemma4MtpConfig, NativeGemma4MtpLayerType, NativeGemma4MtpRopeKind,
            NativeGemma4MtpRopeParams, NativeGemma4MtpTextConfig, ResolvedGemma4MtpAttnWeights,
            ResolvedGemma4MtpDrafter, ResolvedGemma4MtpLayerWeights, ResolvedGemma4MtpMlpWeights,
            load_drafter,
        };
    }
}
pub mod env_state;
mod golden;
mod metal_kernel;
pub mod native_attention;
mod native_cache;
mod native_compile_cache;
mod native_conv1d;
mod native_embedding;
mod native_kernels;
mod native_lm_head;
#[cfg(feature = "mlx-native")]
pub mod native_metal_bridge;
mod native_moe;
mod native_norm;
mod native_quant;
mod native_rope;
mod native_router;
mod native_runtime;
mod native_snapshot;
pub mod native_ssm;
mod qwen3_5_moe;
mod runner_native;
#[cfg(feature = "mlx-pyo3")]
mod runner_pyo3;
mod runner_subprocess;
mod spec_decode;
#[cfg(feature = "mlx-native")]
mod turboquant;

use runner_native::NativeMlxRunner;

pub use native_runtime::FineTimings;
#[cfg(feature = "mlx-pyo3")]
use runner_pyo3::Pyo3Runner;
use runner_subprocess::SubprocessRunner;

/// Output of `Runner::forward_probe`: per-row argmaxes + max-abs logit + new
/// position. Used by Track A2 drift baseline + spec-decode verify-loop.
#[derive(Clone, Debug)]
pub struct ProbeRows {
    pub row_argmaxes: Vec<u32>,
    pub row_max_abs: Vec<f32>,
    pub position: usize,
}

trait Runner {
    fn name(&self) -> &'static str;
    fn prefill(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)>;
    fn decode_step(
        &mut self,
        seq_id: u64,
        last_token: u32,
        position: usize,
    ) -> Result<(u32, usize)>;
    fn remove_seq(&mut self, seq_id: u64) -> Result<()>;
    fn extend(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)>;
    fn forward_probe(&mut self, seq_id: u64, tokens: &[u32]) -> Result<ProbeRows>;
    fn snapshot_state(&mut self, seq_id: u64) -> Result<u64>;
    fn restore_state(&mut self, seq_id: u64, snapshot_id: u64) -> Result<usize>;
    fn release_snapshot(&mut self, snapshot_id: u64) -> Result<()>;
    /// Deep-copy snapshot for fork-to-new-seq (Track A1 prefix caching).
    /// Materializes independent buffers so the snapshot can seed a different
    /// seq's cache without aliasing the source. Returns (snapshot_id, position).
    fn snapshot_state_deep(&mut self, seq_id: u64) -> Result<(u64, usize)>;
    /// Initialize a fresh seq `dst_seq_id` with cache cloned from a deep
    /// snapshot. Snapshot is *not* consumed — multi-fork supported. Returns
    /// the position of the new seq.
    fn fork_from_snapshot(&mut self, snapshot_id: u64, dst_seq_id: u64) -> Result<usize>;
}

impl Runner for SubprocessRunner {
    fn name(&self) -> &'static str {
        "subprocess"
    }

    fn prefill(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)> {
        SubprocessRunner::prefill(self, seq_id, tokens)
    }

    fn decode_step(
        &mut self,
        seq_id: u64,
        last_token: u32,
        position: usize,
    ) -> Result<(u32, usize)> {
        SubprocessRunner::decode_step(self, seq_id, last_token, position)
    }

    fn remove_seq(&mut self, seq_id: u64) -> Result<()> {
        SubprocessRunner::remove_seq(self, seq_id)
    }

    fn extend(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)> {
        SubprocessRunner::extend(self, seq_id, tokens)
    }

    fn forward_probe(&mut self, seq_id: u64, tokens: &[u32]) -> Result<ProbeRows> {
        SubprocessRunner::forward_probe(self, seq_id, tokens)
    }

    fn snapshot_state(&mut self, seq_id: u64) -> Result<u64> {
        SubprocessRunner::snapshot_state(self, seq_id)
    }

    fn restore_state(&mut self, seq_id: u64, snapshot_id: u64) -> Result<usize> {
        SubprocessRunner::restore_state(self, seq_id, snapshot_id)
    }

    fn release_snapshot(&mut self, snapshot_id: u64) -> Result<()> {
        SubprocessRunner::release_snapshot(self, snapshot_id)
    }

    fn snapshot_state_deep(&mut self, seq_id: u64) -> Result<(u64, usize)> {
        SubprocessRunner::snapshot_state_deep(self, seq_id)
    }

    fn fork_from_snapshot(&mut self, snapshot_id: u64, dst_seq_id: u64) -> Result<usize> {
        SubprocessRunner::fork_from_snapshot(self, snapshot_id, dst_seq_id)
    }
}

#[cfg(feature = "mlx-pyo3")]
impl Runner for Pyo3Runner {
    fn name(&self) -> &'static str {
        "pyo3"
    }

    fn prefill(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)> {
        Pyo3Runner::prefill(self, seq_id, tokens)
    }

    fn decode_step(
        &mut self,
        seq_id: u64,
        last_token: u32,
        position: usize,
    ) -> Result<(u32, usize)> {
        Pyo3Runner::decode_step(self, seq_id, last_token, position)
    }

    fn remove_seq(&mut self, seq_id: u64) -> Result<()> {
        Pyo3Runner::remove_seq(self, seq_id)
    }

    fn extend(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)> {
        Pyo3Runner::extend(self, seq_id, tokens)
    }

    fn forward_probe(&mut self, seq_id: u64, tokens: &[u32]) -> Result<ProbeRows> {
        Pyo3Runner::forward_probe(self, seq_id, tokens)
    }

    fn snapshot_state(&mut self, seq_id: u64) -> Result<u64> {
        Pyo3Runner::snapshot_state(self, seq_id)
    }

    fn restore_state(&mut self, seq_id: u64, snapshot_id: u64) -> Result<usize> {
        Pyo3Runner::restore_state(self, seq_id, snapshot_id)
    }

    fn release_snapshot(&mut self, snapshot_id: u64) -> Result<()> {
        Pyo3Runner::release_snapshot(self, snapshot_id)
    }

    fn snapshot_state_deep(&mut self, seq_id: u64) -> Result<(u64, usize)> {
        Pyo3Runner::snapshot_state_deep(self, seq_id)
    }

    fn fork_from_snapshot(&mut self, snapshot_id: u64, dst_seq_id: u64) -> Result<usize> {
        Pyo3Runner::fork_from_snapshot(self, snapshot_id, dst_seq_id)
    }
}

impl Runner for NativeMlxRunner {
    fn name(&self) -> &'static str {
        "native"
    }

    fn prefill(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)> {
        NativeMlxRunner::prefill(self, seq_id, tokens)
    }

    fn decode_step(
        &mut self,
        seq_id: u64,
        last_token: u32,
        position: usize,
    ) -> Result<(u32, usize)> {
        NativeMlxRunner::decode_step(self, seq_id, last_token, position)
    }

    fn remove_seq(&mut self, seq_id: u64) -> Result<()> {
        NativeMlxRunner::remove_seq(self, seq_id)
    }

    fn extend(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)> {
        NativeMlxRunner::extend(self, seq_id, tokens)
    }

    fn forward_probe(&mut self, seq_id: u64, tokens: &[u32]) -> Result<ProbeRows> {
        NativeMlxRunner::forward_probe(self, seq_id, tokens)
    }

    fn snapshot_state(&mut self, seq_id: u64) -> Result<u64> {
        NativeMlxRunner::snapshot_state(self, seq_id)
    }

    fn restore_state(&mut self, seq_id: u64, snapshot_id: u64) -> Result<usize> {
        NativeMlxRunner::restore_state(self, seq_id, snapshot_id)
    }

    fn release_snapshot(&mut self, snapshot_id: u64) -> Result<()> {
        NativeMlxRunner::release_snapshot(self, snapshot_id)
    }

    fn snapshot_state_deep(&mut self, seq_id: u64) -> Result<(u64, usize)> {
        NativeMlxRunner::snapshot_state_deep(self, seq_id)
    }

    fn fork_from_snapshot(&mut self, snapshot_id: u64, dst_seq_id: u64) -> Result<usize> {
        NativeMlxRunner::fork_from_snapshot(self, snapshot_id, dst_seq_id)
    }
}

/// Loads the HF tokenizer that mirrors what mlx_lm uses internally. We keep a
/// Rust copy so encode/decode happen without crossing the Python boundary.
fn load_tokenizer_via_hub(model_id: &str) -> Result<Tokenizer> {
    // If `model_id` is itself a local directory (the desktop control plane
    // passes absolute paths for models already on disk), try `tokenizer.json`
    // from that directory before reaching out to HF Hub. Avoids 404s for repos
    // whose canonical org prefix the caller didn't know.
    let local = std::path::Path::new(model_id);
    if local.is_dir() {
        let tj = local.join("tokenizer.json");
        if tj.is_file() {
            return Tokenizer::from_file(&tj).map_err(|e| anyhow!("tokenizer from_file: {e}"));
        }
    }
    let api = ApiBuilder::new().build().context("hf_hub api init")?;
    let repo = api.model(model_id.to_string());
    let path = repo
        .get("tokenizer.json")
        .context("download tokenizer.json")?;
    Tokenizer::from_file(&path).map_err(|e| anyhow!("tokenizer from_file: {e}"))
}

/// Crate-level resource directory (where `python/mlx_runner.py` lives). PyO3
/// adds this to `sys.path`; the subprocess runner uses the file path directly.
fn crate_python_dir() -> PathBuf {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_dir.join("python")
}

enum RunnerImpl {
    Subprocess(SubprocessRunner),
    #[cfg(feature = "mlx-pyo3")]
    Pyo3(Pyo3Runner),
    Native(NativeMlxRunner),
}

impl RunnerImpl {
    fn as_runner(&self) -> &dyn Runner {
        match self {
            Self::Subprocess(r) => r,
            #[cfg(feature = "mlx-pyo3")]
            Self::Pyo3(r) => r,
            Self::Native(r) => r,
        }
    }

    fn as_runner_mut(&mut self) -> &mut dyn Runner {
        match self {
            Self::Subprocess(r) => r,
            #[cfg(feature = "mlx-pyo3")]
            Self::Pyo3(r) => r,
            Self::Native(r) => r,
        }
    }

    fn name(&self) -> &'static str {
        self.as_runner().name()
    }

    fn prefill(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)> {
        self.as_runner_mut().prefill(seq_id, tokens)
    }

    fn decode_step(
        &mut self,
        seq_id: u64,
        last_token: u32,
        position: usize,
    ) -> Result<(u32, usize)> {
        self.as_runner_mut()
            .decode_step(seq_id, last_token, position)
    }

    fn remove_seq(&mut self, seq_id: u64) -> Result<()> {
        self.as_runner_mut().remove_seq(seq_id)
    }

    fn extend(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)> {
        self.as_runner_mut().extend(seq_id, tokens)
    }

    fn forward_probe(&mut self, seq_id: u64, tokens: &[u32]) -> Result<ProbeRows> {
        self.as_runner_mut().forward_probe(seq_id, tokens)
    }

    fn snapshot_state(&mut self, seq_id: u64) -> Result<u64> {
        self.as_runner_mut().snapshot_state(seq_id)
    }

    fn restore_state(&mut self, seq_id: u64, snapshot_id: u64) -> Result<usize> {
        self.as_runner_mut().restore_state(seq_id, snapshot_id)
    }

    fn release_snapshot(&mut self, snapshot_id: u64) -> Result<()> {
        self.as_runner_mut().release_snapshot(snapshot_id)
    }

    fn snapshot_state_deep(&mut self, seq_id: u64) -> Result<(u64, usize)> {
        self.as_runner_mut().snapshot_state_deep(seq_id)
    }

    fn fork_from_snapshot(&mut self, snapshot_id: u64, dst_seq_id: u64) -> Result<usize> {
        self.as_runner_mut()
            .fork_from_snapshot(snapshot_id, dst_seq_id)
    }

    fn take_native_decode_timing_log(&mut self) -> Option<Vec<(f64, f64)>> {
        match self {
            Self::Native(r) => r.take_decode_timing_log(),
            _ => None,
        }
    }

    fn take_native_decode_fine_timing_log(&mut self) -> Option<Vec<FineTimings>> {
        match self {
            Self::Native(r) => r.take_decode_fine_timing_log(),
            _ => None,
        }
    }

    fn take_pyo3_decode_stage_timings(&mut self) -> Result<Vec<(u64, u64, u64, u64)>> {
        match self {
            #[cfg(feature = "mlx-pyo3")]
            Self::Pyo3(r) => r.take_decode_stage_timings(),
            _ => Ok(Vec::new()),
        }
    }

    /// Native-only DFlash-style prefill that captures per-target-layer
    /// post-MLP residual hiddens. Errors on non-native backends; the Pyo3 /
    /// Subprocess paths already have their own DFlash entry points wired
    /// through `mlx_runner.py::dflash_prefill`. Returns `(next_token,
    /// position, captured_hiddens)` where each captured `mlx_rs::Array` has
    /// shape `[1, prompt_len, hidden]` in the order of `capture_layer_ids`.
    #[cfg(feature = "mlx-native")]
    fn prefill_with_capture(
        &mut self,
        seq_id: u64,
        tokens: &[u32],
        capture_layer_ids: &[usize],
    ) -> Result<(u32, usize, Vec<mlx_rs::Array>)> {
        match self {
            Self::Native(r) => r.prefill_with_capture(seq_id, tokens, capture_layer_ids),
            _ => Err(anyhow!(
                "prefill_with_capture is only supported on the native (mlx-rs) backend; \
                 set LUMEN_MLX_BACKEND=native"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RunnerKind {
    Pyo3,
    Subprocess,
    Native,
}

fn truthy_env(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim),
        Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
    )
}

fn runner_kind_from_env(
    backend: Option<&str>,
    legacy_subprocess: Option<&str>,
) -> Result<RunnerKind> {
    if let Some(raw) = backend {
        let normalized = raw.trim().to_ascii_lowercase();
        return match normalized.as_str() {
            "" => Ok(default_runner_kind()),
            "pyo3" | "python" | "in-process" | "in_process" => Ok(RunnerKind::Pyo3),
            "subprocess" | "process" | "json-rpc" | "json_rpc" => Ok(RunnerKind::Subprocess),
            "native" | "mlx-rs" | "mlx_rs" => Ok(RunnerKind::Native),
            _ => Err(anyhow!(
                "unsupported LUMEN_MLX_BACKEND={raw:?}; expected pyo3, subprocess, or native"
            )),
        };
    }

    if truthy_env(legacy_subprocess) {
        Ok(RunnerKind::Subprocess)
    } else {
        Ok(default_runner_kind())
    }
}

/// Built-in default runner. When the `mlx-native` feature is compiled in we
/// prefer the native mlx-rs path because it lands ~+57% throughput vs PyO3
/// on Qwen3.6-35B-A3B-mxfp4 (and 33× vs Candle at PROMPT_LEN=2048). PyO3
/// remains the fallback when the native feature isn't compiled in.
#[inline]
fn default_runner_kind() -> RunnerKind {
    #[cfg(feature = "mlx-native")]
    {
        RunnerKind::Native
    }
    #[cfg(not(feature = "mlx-native"))]
    {
        RunnerKind::Pyo3
    }
}

fn selected_runner_kind() -> Result<RunnerKind> {
    let backend = std::env::var("LUMEN_MLX_BACKEND").ok();
    let legacy_subprocess = std::env::var("LUMEN_MLX_SUBPROCESS").ok();
    runner_kind_from_env(backend.as_deref(), legacy_subprocess.as_deref())
}

/// Per-session prompt-cache state. Holds the running token sequence (prompt +
/// generated) so that the next request on the same session can skip prefilling
/// the longest common prefix. `last_access` is bumped on every prefill / extend
/// touch so TTL + LRU eviction can run from a single timestamp.
struct SessionState {
    seq_id: u64,
    tokens: Vec<u32>,
    last_access: Instant,
}

/// A1 prefix-cache master snapshot entry. Holds a deep snapshot of the per-
/// layer cache state taken at the end of a *shared* prefix (typically the
/// system-prompt block). Multiple incoming requests whose prompt starts with
/// `prefix_tokens` reuse this snapshot via `fork_from_snapshot`, paying only
/// the suffix prefill instead of the full prefix prefill.
///
/// Master is reusable; `release_snapshot` is only called on cache eviction
/// (LRU / TTL / explicit drop).
struct PrefixCacheEntry {
    master_snapshot_id: u64,
    prefix_tokens: Vec<u32>,
    last_access: Instant,
    hits: u64,
}

/// Read session-eviction limits from env. Both default to "no limit".
///
/// - `LUMEN_MLX_SESSION_TTL_SECS=N` — drop sessions idle > N seconds.
/// - `LUMEN_MLX_SESSION_MAX=N`      — keep at most N sessions, evicting
///   least-recently-used first.
fn read_session_limits() -> (Option<Duration>, Option<usize>) {
    let ttl = std::env::var("LUMEN_MLX_SESSION_TTL_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&n| n > 0)
        .map(Duration::from_secs);
    let max = std::env::var("LUMEN_MLX_SESSION_MAX")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0);
    (ttl, max)
}

/// Read A1 prefix-cache config from env.
///
/// - `LUMEN_MLX_PREFIX_CACHE=1`         — enable feature (default OFF).
/// - `LUMEN_MLX_PREFIX_CACHE_TTL_SECS=N` — drop entries idle > N seconds.
/// - `LUMEN_MLX_PREFIX_CACHE_MAX=N`      — keep at most N entries, LRU-evict.
fn read_prefix_cache_limits() -> (bool, Option<Duration>, Option<usize>) {
    let enabled = truthy_env(std::env::var("LUMEN_MLX_PREFIX_CACHE").ok().as_deref());
    let ttl = std::env::var("LUMEN_MLX_PREFIX_CACHE_TTL_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&n| n > 0)
        .map(Duration::from_secs);
    let max = std::env::var("LUMEN_MLX_PREFIX_CACHE_MAX")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0);
    (enabled, ttl, max)
}

/// Compute a process-stable cache key for a chat request based on the system
/// message content. Returns `None` if there is no system message to share —
/// the prefix cache only ever shares the system-prompt block.
fn auto_prefix_key(messages: &[(String, String)]) -> Option<String> {
    let (role, content) = messages.first()?;
    if role != "system" || content.is_empty() {
        return None;
    }
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    content.hash(&mut h);
    Some(format!("auto-{:016x}", h.finish()))
}

/// Render the chat-template system-prompt block alone (no trailing assistant
/// header). Tokenizing this produces the prefix tokens we expect to find at
/// the start of the full chat-input tokenization.
fn format_system_prefix(message: &(String, String)) -> String {
    let (role, content) = message;
    let mut s = String::new();
    s.push_str("<|im_start|>");
    s.push_str(role);
    s.push('\n');
    s.push_str(content);
    s.push_str("<|im_end|>\n");
    s
}

pub struct MlxBackend {
    runner: RunnerImpl,
    pub model_id: String,
    pub eos_tokens: Vec<u32>,
    pub vocab_size: usize,
    tokenizer: Option<Tokenizer>,
    next_seq_id: AtomicU64,
    sessions: std::collections::HashMap<String, SessionState>,
    session_ttl: Option<Duration>,
    session_max: Option<usize>,
    prefix_caches: std::collections::HashMap<String, PrefixCacheEntry>,
    prefix_cache_enabled: bool,
    prefix_cache_ttl: Option<Duration>,
    prefix_cache_max: Option<usize>,
}

impl MlxBackend {
    /// Load the model. Default path is PyO3 in-process. `LUMEN_MLX_BACKEND`
    /// selects the runner (`pyo3`, `subprocess`, or `native`); legacy
    /// `LUMEN_MLX_SUBPROCESS=1` still switches to the subprocess JSON-RPC
    /// fallback. The `native` runner (built with the `mlx-native` feature) is
    /// gated by the parity harness — see `LUMEN_MLX_GOLDEN_IN`. The native
    /// runner accepts both a local model directory (containing `config.json`
    /// plus `*.safetensors` shards) and a HuggingFace Hub repo id (e.g.
    /// `mlx-community/Qwen3.6-35B-A3B-mxfp4`); the latter is downloaded into
    /// the standard HF cache on first use.
    pub fn load(model_id: &str) -> Result<Self> {
        let runner_kind = selected_runner_kind()?;
        let (runner, eos_tokens, vocab_size) = match runner_kind {
            RunnerKind::Subprocess => {
                eprintln!("[mlx] runner=subprocess (LUMEN_MLX_BACKEND=subprocess)");
                let mut r = SubprocessRunner::spawn()?;
                let info = r.load(model_id)?;
                (RunnerImpl::Subprocess(r), info.eos_tokens, info.vocab_size)
            }
            #[cfg(feature = "mlx-pyo3")]
            RunnerKind::Pyo3 => {
                eprintln!(
                    "[mlx] runner=pyo3 (in-process, set LUMEN_MLX_BACKEND=subprocess for fallback)"
                );
                let mut r = Pyo3Runner::new()?;
                let info = r.load(model_id)?;
                (RunnerImpl::Pyo3(r), info.eos_tokens, info.vocab_size)
            }
            #[cfg(not(feature = "mlx-pyo3"))]
            RunnerKind::Pyo3 => {
                return Err(anyhow!(
                    "RunnerKind::Pyo3 requested but lumen-mlx was built without \
                     the `mlx-pyo3` feature. Rebuild with `--features mlx-pyo3` or \
                     use a different backend (native / subprocess)."
                ));
            }
            RunnerKind::Native => {
                eprintln!(
                    "[mlx] runner=native (mlx-rs Phase 3d.5; parity gate via LUMEN_MLX_GOLDEN_IN)"
                );
                let mut r = NativeMlxRunner::new()?;
                let info = r.load(model_id)?;
                (RunnerImpl::Native(r), info.eos_tokens, info.vocab_size)
            }
        };

        let tokenizer = match load_tokenizer_via_hub(model_id) {
            Ok(tok) => Some(tok),
            Err(e) => {
                eprintln!("[mlx] tokenizer load failed (encode disabled): {e}");
                None
            }
        };

        eprintln!(
            "[mlx] loaded: runner={} vocab={vocab_size} eos={eos_tokens:?}",
            runner.name(),
        );

        let (session_ttl, session_max) = read_session_limits();
        if let Some(ttl) = session_ttl {
            eprintln!("[mlx] session TTL = {}s", ttl.as_secs());
        }
        if let Some(max) = session_max {
            eprintln!("[mlx] session LRU cap = {max}");
        }

        let (prefix_cache_enabled, prefix_cache_ttl, prefix_cache_max) = read_prefix_cache_limits();
        if prefix_cache_enabled {
            eprintln!("[mlx] prefix cache ENABLED (auto-keyed by system message hash)");
            if let Some(ttl) = prefix_cache_ttl {
                eprintln!("[mlx] prefix cache TTL = {}s", ttl.as_secs());
            }
            if let Some(max) = prefix_cache_max {
                eprintln!("[mlx] prefix cache LRU cap = {max}");
            }
        }

        Ok(Self {
            runner,
            model_id: model_id.to_string(),
            eos_tokens,
            vocab_size,
            tokenizer,
            next_seq_id: AtomicU64::new(1),
            sessions: std::collections::HashMap::new(),
            session_ttl,
            session_max,
            prefix_caches: std::collections::HashMap::new(),
            prefix_cache_enabled,
            prefix_cache_ttl,
            prefix_cache_max,
        })
    }

    pub fn prefill(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)> {
        self.runner.prefill(seq_id, tokens)
    }

    pub fn decode_step(
        &mut self,
        seq_id: u64,
        last_token: u32,
        position: usize,
    ) -> Result<(u32, usize)> {
        self.runner.decode_step(seq_id, last_token, position)
    }

    /// Drains the per-step `(forward_ms, tail_ms)` timing log captured by the
    /// native runner when `LUMEN_NATIVE_TIMING=1` was set at backend
    /// construction. Returns `None` for non-native runners or when the env was
    /// not set. Subsequent calls return an empty `Vec` until the next decode.
    pub fn take_native_decode_timing_log(&mut self) -> Option<Vec<(f64, f64)>> {
        self.runner.take_native_decode_timing_log()
    }

    /// Drains the per-step layer-kind breakdown (`embed`, `full_attn`,
    /// `linear_attn`, `moe`, `lm_head`) captured by the native runner when
    /// `LUMEN_NATIVE_TIMING=1` was set. Aligned 1:1 with
    /// `take_native_decode_timing_log()`. Returns `None` for non-native
    /// runners or when the env was not set.
    pub fn take_native_decode_fine_timing_log(&mut self) -> Option<Vec<FineTimings>> {
        self.runner.take_native_decode_fine_timing_log()
    }

    /// Drains the per-step PyO3 decode_step stage timings when
    /// `LUMEN_PYO3_DECODE_STAGE_TIMING=1` was set. Returns
    /// `(arr_ns, forward_ns, sync_ns, tail_ns)` per step where:
    ///   - arr_ns:     `mx.array([[last_token]])` creation
    ///   - forward_ns: `self.model(arr, cache=cache)` (lazy graph build)
    ///   - sync_ns:    `mx.argmax(...).item()` (forces GPU sync)
    ///   - tail_ns:    `_apply_kv_quant` + state update
    /// Returns empty Vec for non-PyO3 backends or when env was not set.
    pub fn take_pyo3_decode_stage_timings(&mut self) -> Result<Vec<(u64, u64, u64, u64)>> {
        self.runner.take_pyo3_decode_stage_timings()
    }

    pub fn remove_seq(&mut self, seq_id: u64) -> Result<()> {
        self.runner.remove_seq(seq_id)
    }

    pub fn runner_extend(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)> {
        self.runner.extend(seq_id, tokens)
    }

    /// One-shot batched forward of `tokens` at the seq's current cache state.
    /// Returns per-row argmax + max-abs logit. State advances by `tokens.len()`
    /// — caller is responsible for cleanup. Used by Track A2 drift baseline.
    pub fn forward_probe(&mut self, seq_id: u64, tokens: &[u32]) -> Result<ProbeRows> {
        self.runner.forward_probe(seq_id, tokens)
    }

    /// Native-only prefill that captures the post-MLP residual at each target
    /// layer for DFlash cross-attn ctx (D5a). Returns `(next_token, position,
    /// hiddens)` where `hiddens[i]` has shape `[1, prompt_len, hidden]` for
    /// layer `capture_layer_ids[i]`. The caller typically follows with
    /// `mlx_rs::ops::concatenate_axis(&hiddens, -1)` to build the
    /// `target_hidden` tensor expected by the DFlash draft (matches the
    /// `fc` weight layout `[len(target_layer_ids) * hidden, hidden]`).
    ///
    /// Errors when invoked against the Pyo3 / Subprocess backends. Use
    /// `LUMEN_MLX_BACKEND=native` to select the supported runner.
    #[cfg(feature = "mlx-native")]
    pub fn prefill_with_capture(
        &mut self,
        seq_id: u64,
        tokens: &[u32],
        capture_layer_ids: &[usize],
    ) -> Result<(u32, usize, Vec<mlx_rs::Array>)> {
        self.runner
            .prefill_with_capture(seq_id, tokens, capture_layer_ids)
    }

    /// Capture the seq's per-layer cache state. Returns an opaque snapshot id.
    /// Snapshot is consumed by `restore_state` (one-shot) or freed by
    /// `release_snapshot`. Used by spec-decode partial-accept rollback.
    pub fn snapshot_state(&mut self, seq_id: u64) -> Result<u64> {
        self.runner.snapshot_state(seq_id)
    }

    /// Roll back the seq's cache to a previously captured snapshot. Returns
    /// the position the seq is at after restoration. Snapshot is consumed.
    pub fn restore_state(&mut self, seq_id: u64, snapshot_id: u64) -> Result<usize> {
        self.runner.restore_state(seq_id, snapshot_id)
    }

    /// Free a snapshot without restoring (e.g., when spec verify fully accepts
    /// and rollback isn't needed).
    pub fn release_snapshot(&mut self, snapshot_id: u64) -> Result<()> {
        self.runner.release_snapshot(snapshot_id)
    }

    /// Capture a deep-copy snapshot of the seq's cache state. Unlike
    /// `snapshot_state`, the snapshot is independent (does not alias the
    /// source seq's state) so it can seed a fresh seq via
    /// `fork_from_snapshot`. The master snapshot is *reusable* across many
    /// forks until released. Used by Track A1 prefix caching.
    /// Returns `(snapshot_id, position)`.
    pub fn snapshot_state_deep(&mut self, seq_id: u64) -> Result<(u64, usize)> {
        self.runner.snapshot_state_deep(seq_id)
    }

    /// Initialize a fresh seq `dst_seq_id` whose cache is cloned from a deep
    /// snapshot. Snapshot is *not* consumed; the caller can fork the same
    /// master snapshot into many destination seqs (each independent of the
    /// others and of the source). Returns the dst seq's starting position.
    pub fn fork_from_snapshot(&mut self, snapshot_id: u64, dst_seq_id: u64) -> Result<usize> {
        self.runner.fork_from_snapshot(snapshot_id, dst_seq_id)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let tok = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| anyhow!("tokenizer not loaded"))?;
        let enc = tok
            .encode(text, true)
            .map_err(|e| anyhow!("tokenizer encode: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        let tok = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| anyhow!("tokenizer not loaded"))?;
        tok.decode(tokens, true)
            .map_err(|e| anyhow!("tokenizer decode: {e}"))
    }

    pub fn build_chat_input(
        &self,
        messages: &[(String, String)],
        thinking: bool,
    ) -> Result<Vec<u32>> {
        let prompt = format_qwen3_chat(messages, thinking);
        self.encode(&prompt)
    }

    pub fn alloc_seq_id(&self) -> u64 {
        self.next_seq_id.fetch_add(1, Ordering::Relaxed)
    }

    pub fn eos_tokens(&self) -> &[u32] {
        &self.eos_tokens
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Streaming chat. Calls `on_token` with each new decoded text fragment.
    /// Greedy-only at B1+B2 phase 1.
    ///
    /// When `LUMEN_MLX_SPEC=ngram` is set, dispatches to the N-gram K=2
    /// speculative-decode path (`chat_streaming_spec_ngram`). Default OFF.
    pub fn chat_streaming<F>(
        &mut self,
        messages: &[(String, String)],
        max_new_tokens: usize,
        thinking: bool,
        seq_id: u64,
        mut on_token: F,
    ) -> Result<String>
    where
        F: FnMut(&str),
    {
        if let Some(cfg) = spec_decode::read_spec_config() {
            return self.chat_streaming_spec_ngram(
                messages,
                max_new_tokens,
                thinking,
                seq_id,
                cfg,
                on_token,
            );
        }

        if self.prefix_cache_enabled {
            if let Some(key) = auto_prefix_key(messages) {
                return self.chat_streaming_prefix_cache(
                    messages,
                    max_new_tokens,
                    thinking,
                    seq_id,
                    &key,
                    on_token,
                );
            }
        }

        let prompt_ids = self.build_chat_input(messages, thinking)?;
        if prompt_ids.is_empty() {
            return Err(anyhow!("empty prompt after tokenization"));
        }
        let t_prefill = std::time::Instant::now();
        let (mut last, mut pos) = self.prefill(seq_id, &prompt_ids)?;
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[mlx] seq {seq_id} prefill: {} tokens in {prefill_ms:.0}ms ({:.1} tok/s) -> tok={last}",
            prompt_ids.len(),
            prompt_ids.len() as f64 / (prefill_ms / 1000.0)
        );

        // Per-emit wallclock instrumentation (Phase C Option B).
        // LUMEN_STREAM_TIMING=1 → logs first/skip/last on_token wallclock so
        // the inner emit rate can be compared with the SSE-write rate (chat.rs)
        // and the bench client steady rate to localize ~6% native-specific
        // HTTP envelope loss. LUMEN_STREAM_SKIP=N (default 20) matches
        // `bench_b4_decode_ab.py --skip-first`.
        let stream_timing = std::env::var("LUMEN_STREAM_TIMING").is_ok();
        let stream_skip: usize = std::env::var("LUMEN_STREAM_SKIP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(20);
        // H1 incremental detokenize: decode only the unemitted token tail per
        // step instead of the full accumulated `generated` vector. Eliminates
        // the O(n²) re-detokenize cost in the chat_streaming decode loop. Gate
        // default OFF; flip via `LUMEN_NATIVE_STREAM_INCR=1`.
        let incr_detok = std::env::var("LUMEN_NATIVE_STREAM_INCR")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let mut t_first_emit: Option<std::time::Instant> = None;
        let mut t_skip_emit: Option<std::time::Instant> = None;
        let mut t_last_emit: Option<std::time::Instant> = None;
        let mut n_emits: usize = 0;

        let mut generated: Vec<u32> = vec![last];
        let mut prev_text = String::new();
        let mut emitted_idx: usize = 0;
        if let Ok(text) = self.decode(&generated) {
            if !text.is_empty() && !text.contains('\u{FFFD}') {
                if stream_timing {
                    let now = std::time::Instant::now();
                    if t_first_emit.is_none() {
                        t_first_emit = Some(now);
                    }
                    if n_emits == stream_skip {
                        t_skip_emit = Some(now);
                    }
                    t_last_emit = Some(now);
                    n_emits += 1;
                }
                on_token(&text);
                prev_text = text;
                emitted_idx = generated.len();
            }
        }
        if self.eos_tokens.contains(&last) {
            let out = self.decode(&generated).unwrap_or_default();
            self.remove_seq(seq_id).ok();
            return Ok(out);
        }

        let t_decode = std::time::Instant::now();
        for step in 1..max_new_tokens {
            let (next, new_pos) = self.decode_step(seq_id, last, pos)?;
            last = next;
            pos = new_pos;
            generated.push(next);
            if incr_detok {
                let tail_start = emitted_idx;
                if tail_start < generated.len() {
                    if let Ok(text) = self.decode(&generated[tail_start..]) {
                        if !text.is_empty() && !text.contains('\u{FFFD}') {
                            if stream_timing {
                                let now = std::time::Instant::now();
                                if t_first_emit.is_none() {
                                    t_first_emit = Some(now);
                                }
                                if n_emits == stream_skip {
                                    t_skip_emit = Some(now);
                                }
                                t_last_emit = Some(now);
                                n_emits += 1;
                            }
                            on_token(&text);
                            emitted_idx = generated.len();
                        }
                    }
                }
            } else if let Ok(text) = self.decode(&generated) {
                if text.len() > prev_text.len() && !text.contains('\u{FFFD}') {
                    if stream_timing {
                        let now = std::time::Instant::now();
                        if t_first_emit.is_none() {
                            t_first_emit = Some(now);
                        }
                        if n_emits == stream_skip {
                            t_skip_emit = Some(now);
                        }
                        t_last_emit = Some(now);
                        n_emits += 1;
                    }
                    on_token(&text[prev_text.len()..]);
                    prev_text = text;
                }
            }
            if self.eos_tokens.contains(&next) {
                eprintln!(
                    "[mlx] seq {seq_id} EOS at step {step} ({:.1} tok/s)",
                    step as f64 / (t_decode.elapsed().as_secs_f64())
                );
                break;
            }
        }
        let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
        let n_gen = generated.len();
        eprintln!(
            "[mlx] seq {seq_id} done: {n_gen} tokens in {decode_ms:.0}ms ({:.1} tok/s)",
            n_gen as f64 / (decode_ms / 1000.0)
        );
        if stream_timing {
            if let (Some(tf), Some(tl)) = (t_first_emit, t_last_emit) {
                let emit_span_ms = (tl - tf).as_secs_f64() * 1000.0;
                let steady_span_ms = t_skip_emit
                    .map(|ts| (tl - ts).as_secs_f64() * 1000.0)
                    .unwrap_or(0.0);
                let steady_n = n_emits.saturating_sub(stream_skip + 1);
                let steady_rate = if steady_span_ms > 0.0 {
                    steady_n as f64 / (steady_span_ms / 1000.0)
                } else {
                    0.0
                };
                eprintln!(
                    "[stream-timing] seq {seq_id} emit: n_emits={n_emits} first->last={emit_span_ms:.1}ms skip{stream_skip}->last={steady_span_ms:.1}ms steady_rate_emit={steady_rate:.2}tok/s"
                );
            }
        }
        let out = self.decode(&generated).unwrap_or_default();
        self.remove_seq(seq_id).ok();
        Ok(out)
    }

    /// Prefix-cache-aware variant of `chat_streaming` (Track A1).
    ///
    /// On cache **hit** for `prefix_cache_key` *and* the new prompt strictly
    /// extends the cached prefix tokens: forks `seq_id` from the master
    /// snapshot and only feeds the suffix — saving the system-prompt prefill
    /// that all hits share.
    ///
    /// On cache **miss** (or divergent prompt): falls back to a two-stage
    /// prefill — prefill(system_prefix) → snapshot_state_deep → extend(suffix)
    /// → decode. The fresh master is stored under `prefix_cache_key` for
    /// subsequent hits. If the system block can't be isolated (no system
    /// message, or starts_with check fails), this degenerates to a single
    /// cold prefill with no caching side-effect.
    fn chat_streaming_prefix_cache<F>(
        &mut self,
        messages: &[(String, String)],
        max_new_tokens: usize,
        thinking: bool,
        seq_id: u64,
        prefix_cache_key: &str,
        mut on_token: F,
    ) -> Result<String>
    where
        F: FnMut(&str),
    {
        let prompt_ids = self.build_chat_input(messages, thinking)?;
        if prompt_ids.is_empty() {
            return Err(anyhow!("empty prompt after tokenization"));
        }

        self.evict_stale_prefix_caches();

        // ── Cache lookup ──
        let cached = self
            .prefix_caches
            .get(prefix_cache_key)
            .map(|e| (e.master_snapshot_id, e.prefix_tokens.clone(), e.last_access));

        let (mut last, mut pos) = match cached {
            Some((master, ref prefix, _))
                if !prefix.is_empty()
                    && prompt_ids.len() > prefix.len()
                    && prompt_ids.starts_with(prefix) =>
            {
                // ── Cache HIT: fork master → seq, extend suffix ──
                let suffix = &prompt_ids[prefix.len()..];
                let t = std::time::Instant::now();
                let fork_pos = self.runner.fork_from_snapshot(master, seq_id)?;
                let (last, pos) = self.runner.extend(seq_id, suffix)?;
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                if let Some(entry) = self.prefix_caches.get_mut(prefix_cache_key) {
                    entry.last_access = Instant::now();
                    entry.hits += 1;
                }
                eprintln!(
                    "[mlx] prefix-cache HIT key={prefix_cache_key:?} \
                     prefix={} suffix={} fork+extend={ms:.0}ms (fork_pos={fork_pos})",
                    prefix.len(),
                    suffix.len()
                );
                (last, pos)
            }
            _ => {
                // ── Cache MISS: cold prefill (with optional snapshot capture) ──
                let prefix_len = self.detect_system_prefix_len(messages)?;
                let usable_prefix = prefix_len > 0
                    && prefix_len < prompt_ids.len()
                    && prompt_ids[..prefix_len] == prompt_ids[..prefix_len]
                    && prompt_ids.starts_with(&prompt_ids[..prefix_len]);

                if usable_prefix {
                    // Two-stage prefill: prefix → snapshot → suffix.
                    let prefix = prompt_ids[..prefix_len].to_vec();
                    let suffix = &prompt_ids[prefix_len..];

                    let t = std::time::Instant::now();
                    let _ = self.runner.prefill(seq_id, &prefix)?;
                    let prefix_ms = t.elapsed().as_secs_f64() * 1000.0;

                    let snap_result = self.runner.snapshot_state_deep(seq_id);
                    match snap_result {
                        Ok((snap_id, _snap_pos)) => {
                            // Drop any prior entry for this key (free its snapshot).
                            if let Some(old) = self.prefix_caches.remove(prefix_cache_key) {
                                let _ = self.runner.release_snapshot(old.master_snapshot_id);
                            }
                            self.prefix_caches.insert(
                                prefix_cache_key.to_string(),
                                PrefixCacheEntry {
                                    master_snapshot_id: snap_id,
                                    prefix_tokens: prefix.clone(),
                                    last_access: Instant::now(),
                                    hits: 0,
                                },
                            );
                            self.evict_stale_prefix_caches();
                            eprintln!(
                                "[mlx] prefix-cache MISS key={prefix_cache_key:?} \
                                 prefilled prefix={} in {prefix_ms:.0}ms, \
                                 stored snapshot={snap_id}",
                                prefix.len()
                            );
                        }
                        Err(e) => {
                            eprintln!(
                                "[mlx] prefix-cache MISS key={prefix_cache_key:?} \
                                 snapshot_state_deep failed ({e}); falling through \
                                 without caching"
                            );
                        }
                    }

                    let t = std::time::Instant::now();
                    let (last, pos) = self.runner.extend(seq_id, suffix)?;
                    let suffix_ms = t.elapsed().as_secs_f64() * 1000.0;
                    eprintln!(
                        "[mlx] seq {seq_id} suffix extend: {} tokens in {suffix_ms:.0}ms",
                        suffix.len()
                    );
                    (last, pos)
                } else {
                    // No usable system prefix — plain cold prefill, no cache write.
                    let t = std::time::Instant::now();
                    let (last, pos) = self.runner.prefill(seq_id, &prompt_ids)?;
                    let ms = t.elapsed().as_secs_f64() * 1000.0;
                    eprintln!(
                        "[mlx] prefix-cache MISS key={prefix_cache_key:?} \
                         no isolatable prefix; cold prefill {} tokens in {ms:.0}ms",
                        prompt_ids.len()
                    );
                    (last, pos)
                }
            }
        };

        // ── Decode loop (parallel to chat_streaming) ──
        let mut generated: Vec<u32> = vec![last];
        let mut prev_text = String::new();
        if let Ok(text) = self.decode(&generated) {
            if !text.is_empty() && !text.contains('\u{FFFD}') {
                on_token(&text);
                prev_text = text;
            }
        }
        if self.eos_tokens.contains(&last) {
            let out = self.decode(&generated).unwrap_or_default();
            self.remove_seq(seq_id).ok();
            return Ok(out);
        }

        let t_decode = std::time::Instant::now();
        for step in 1..max_new_tokens {
            let (next, new_pos) = self.decode_step(seq_id, last, pos)?;
            last = next;
            pos = new_pos;
            generated.push(next);
            if let Ok(text) = self.decode(&generated) {
                if text.len() > prev_text.len() && !text.contains('\u{FFFD}') {
                    on_token(&text[prev_text.len()..]);
                    prev_text = text;
                }
            }
            if self.eos_tokens.contains(&next) {
                eprintln!(
                    "[mlx] seq {seq_id} EOS at step {step} ({:.1} tok/s)",
                    step as f64 / (t_decode.elapsed().as_secs_f64())
                );
                break;
            }
        }
        let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
        let n_gen = generated.len();
        eprintln!(
            "[mlx] seq {seq_id} done: {n_gen} tokens in {decode_ms:.0}ms ({:.1} tok/s)",
            n_gen as f64 / (decode_ms / 1000.0)
        );
        let out = self.decode(&generated).unwrap_or_default();
        self.remove_seq(seq_id).ok();
        Ok(out)
    }

    /// Returns the length, in tokens, of the system-prompt block at the start
    /// of a chat-templated prompt — i.e. how many of the first prompt tokens
    /// are shareable. Returns 0 if there's no system message or if tokenizing
    /// the system block alone doesn't produce a strict prefix of the full
    /// chat-input (rare but possible due to tokenizer boundary effects).
    fn detect_system_prefix_len(&self, messages: &[(String, String)]) -> Result<usize> {
        let Some(first) = messages.first() else {
            return Ok(0);
        };
        if first.0 != "system" || first.1.is_empty() {
            return Ok(0);
        }
        let block = format_system_prefix(first);
        let sys_ids = self.encode(&block)?;
        Ok(sys_ids.len())
    }

    /// Drop a prefix-cache entry by key, releasing its master snapshot.
    /// Returns true if the entry existed.
    pub fn drop_prefix_cache(&mut self, key: &str) -> bool {
        if let Some(entry) = self.prefix_caches.remove(key) {
            let _ = self.runner.release_snapshot(entry.master_snapshot_id);
            eprintln!(
                "[mlx] prefix-cache key={key:?} dropped (hits={})",
                entry.hits
            );
            true
        } else {
            false
        }
    }

    /// Drop all prefix-cache entries; returns the number released.
    pub fn clear_prefix_cache(&mut self) -> usize {
        let n = self.prefix_caches.len();
        let entries: Vec<_> = self.prefix_caches.drain().collect();
        for (_, entry) in entries {
            let _ = self.runner.release_snapshot(entry.master_snapshot_id);
        }
        if n > 0 {
            eprintln!("[mlx] prefix-cache cleared ({n} entries)");
        }
        n
    }

    pub fn prefix_cache_count(&self) -> usize {
        self.prefix_caches.len()
    }

    /// Drop entries idle past TTL, then enforce LRU cap. No-op when both are
    /// unset.
    fn evict_stale_prefix_caches(&mut self) {
        let now = Instant::now();
        let victims = pick_prefix_eviction_victims(
            &self.prefix_caches,
            now,
            self.prefix_cache_ttl,
            self.prefix_cache_max,
        );
        for (key, reason) in victims {
            if let Some(entry) = self.prefix_caches.remove(&key) {
                let _ = self.runner.release_snapshot(entry.master_snapshot_id);
                eprintln!("[mlx] prefix-cache key={key:?} evicted ({reason})");
            }
        }
    }

    /// N-gram K=2 speculative decode variant of `chat_streaming`. Drift-aware:
    /// commits target's verify-row argmax, not the strict S=1 baseline. Sequence
    /// will not be bit-identical to baseline — measure with cosine + top-1 match.
    fn chat_streaming_spec_ngram<F>(
        &mut self,
        messages: &[(String, String)],
        max_new_tokens: usize,
        thinking: bool,
        seq_id: u64,
        cfg: spec_decode::SpecConfig,
        mut on_token: F,
    ) -> Result<String>
    where
        F: FnMut(&str),
    {
        let prompt_ids = self.build_chat_input(messages, thinking)?;
        if prompt_ids.is_empty() {
            return Err(anyhow!("empty prompt after tokenization"));
        }
        let t_prefill = std::time::Instant::now();
        let (mut t_pred, mut pos) = self.prefill(seq_id, &prompt_ids)?;
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[mlx-spec] seq {seq_id} prefill: {} tokens in {prefill_ms:.0}ms (n={} k={}) -> tok={t_pred}",
            prompt_ids.len(),
            cfg.n,
            cfg.k,
        );

        // History for n-gram lookup includes prompt + everything we've committed
        // (initialise with the prompt; we extend as we commit).
        let mut history: Vec<u32> = prompt_ids.clone();
        let mut ngram = spec_decode::NgramTable::new(cfg.n);
        // Pre-populate from prompt so first committed token can match a prompt
        // pattern (e.g., repeated phrases).
        if history.len() >= cfg.n {
            for w in history.windows(cfg.n) {
                ngram.observe(&w[..cfg.n - 1], w[cfg.n - 1]);
            }
        }

        let mut generated: Vec<u32> = Vec::new();
        let mut prev_text = String::new();
        let mut stats = spec_decode::SpecStats::default();

        let t_decode = std::time::Instant::now();
        let mut step: usize = 0;
        while step < max_new_tokens {
            // Quick path: try a spec attempt if we have enough history.
            let proposal = if history.len() >= cfg.n - 1 {
                ngram.propose(&history, cfg.k)
            } else {
                Vec::new()
            };

            if proposal.is_empty() || proposal[0] != t_pred {
                // No usable proposal, or draft d_0 disagrees with target's known
                // prediction → just commit t_pred via decode_step.
                if proposal.is_empty() {
                    stats.fallthrough_no_proposal += 1;
                } else {
                    stats.fallthrough_d0_mismatch += 1;
                }
                // Commit t_pred.
                generated.push(t_pred);
                ngram.observe(&history, t_pred);
                history.push(t_pred);
                stats.committed_via_baseline += 1;
                if let Ok(text) = self.decode(&generated) {
                    if text.len() > prev_text.len() && !text.contains('\u{FFFD}') {
                        on_token(&text[prev_text.len()..]);
                        prev_text = text;
                    }
                }
                if self.eos_tokens.contains(&t_pred) {
                    break;
                }
                step += 1;
                if step >= max_new_tokens {
                    break;
                }
                let (next, new_pos) = self.decode_step(seq_id, t_pred, pos)?;
                t_pred = next;
                pos = new_pos;
                continue;
            }

            // proposal.len() >= 1, proposal[0] == t_pred.
            stats.attempts += 1;

            if proposal.len() < 2 {
                // K=1 fallback: just commit d_0 = t_pred (already known correct)
                // and ask target for the next prediction via decode_step.
                generated.push(t_pred);
                ngram.observe(&history, t_pred);
                history.push(t_pred);
                stats.committed_via_baseline += 1;
                if let Ok(text) = self.decode(&generated) {
                    if text.len() > prev_text.len() && !text.contains('\u{FFFD}') {
                        on_token(&text[prev_text.len()..]);
                        prev_text = text;
                    }
                }
                if self.eos_tokens.contains(&t_pred) {
                    break;
                }
                step += 1;
                if step >= max_new_tokens {
                    break;
                }
                let (next, new_pos) = self.decode_step(seq_id, t_pred, pos)?;
                t_pred = next;
                pos = new_pos;
                continue;
            }

            let d0 = proposal[0];
            let d1 = proposal[1];

            // Snapshot before verify forward.
            let snap = self.snapshot_state(seq_id)?;
            let probe = self.forward_probe(seq_id, &[d0, d1])?;

            if probe.row_argmaxes.len() < 2 {
                // Defensive: should not happen for K=2 verify.
                self.restore_state(seq_id, snap).ok();
                return Err(anyhow!("verify forward returned <2 rows"));
            }
            let row0 = probe.row_argmaxes[0]; // pred conditioned on cache + d0
            let row1 = probe.row_argmaxes[1]; // pred conditioned on cache + d0 + d1

            if row0 == d1 {
                // Full accept: commit d0, d1, row1 (corrective).
                stats.full_accept += 1;
                self.release_snapshot(snap).ok();
                // State already at offset M+2 (cache has d0, d1).
                generated.push(d0);
                ngram.observe(&history, d0);
                history.push(d0);
                generated.push(d1);
                ngram.observe(&history, d1);
                history.push(d1);
                stats.committed_via_spec += 2;
                if let Ok(text) = self.decode(&generated) {
                    if text.len() > prev_text.len() && !text.contains('\u{FFFD}') {
                        on_token(&text[prev_text.len()..]);
                        prev_text = text;
                    }
                }
                if self.eos_tokens.contains(&d0) || self.eos_tokens.contains(&d1) {
                    break;
                }
                // Now commit row1 as the corrective + 3rd token.
                step += 2;
                if step >= max_new_tokens {
                    break;
                }
                generated.push(row1);
                ngram.observe(&history, row1);
                history.push(row1);
                stats.committed_via_spec += 1;
                if let Ok(text) = self.decode(&generated) {
                    if text.len() > prev_text.len() && !text.contains('\u{FFFD}') {
                        on_token(&text[prev_text.len()..]);
                        prev_text = text;
                    }
                }
                if self.eos_tokens.contains(&row1) {
                    break;
                }
                step += 1;
                if step >= max_new_tokens {
                    break;
                }
                // Advance state past row1 + get next t_pred.
                pos = probe.position; // == M+2 (verified by the runner)
                let (next, new_pos) = self.decode_step(seq_id, row1, pos)?;
                t_pred = next;
                pos = new_pos;
            } else {
                // Partial accept (j=0): commit d0, then row0 (corrective).
                stats.partial_accept += 1;
                // Roll back to before verify forward.
                self.restore_state(seq_id, snap)?;
                generated.push(d0);
                ngram.observe(&history, d0);
                history.push(d0);
                stats.committed_via_spec += 1;
                if let Ok(text) = self.decode(&generated) {
                    if text.len() > prev_text.len() && !text.contains('\u{FFFD}') {
                        on_token(&text[prev_text.len()..]);
                        prev_text = text;
                    }
                }
                if self.eos_tokens.contains(&d0) {
                    break;
                }
                // Re-feed d0 via decode_step (state was rolled back to M).
                step += 1;
                if step >= max_new_tokens {
                    break;
                }
                let (after_d0, new_pos) = self.decode_step(seq_id, d0, pos)?;
                pos = new_pos;
                // after_d0 is target's S=1 prediction at offset M+1. Commit it.
                generated.push(after_d0);
                ngram.observe(&history, after_d0);
                history.push(after_d0);
                stats.committed_via_spec += 1;
                if let Ok(text) = self.decode(&generated) {
                    if text.len() > prev_text.len() && !text.contains('\u{FFFD}') {
                        on_token(&text[prev_text.len()..]);
                        prev_text = text;
                    }
                }
                if self.eos_tokens.contains(&after_d0) {
                    break;
                }
                step += 1;
                if step >= max_new_tokens {
                    break;
                }
                // Get next t_pred.
                let (next, new_pos) = self.decode_step(seq_id, after_d0, pos)?;
                t_pred = next;
                pos = new_pos;
            }
        }

        let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
        let n_gen = generated.len();
        stats.log_summary(&format!("seq {seq_id}"));
        eprintln!(
            "[mlx-spec] seq {seq_id} done: {n_gen} tokens in {decode_ms:.0}ms ({:.1} tok/s)",
            n_gen as f64 / (decode_ms / 1000.0)
        );
        let out = self.decode(&generated).unwrap_or_default();
        self.remove_seq(seq_id).ok();
        Ok(out)
    }

    /// Streaming chat with prompt-cache reuse keyed by `session_id`.
    ///
    /// Behavior:
    /// 1. Tokenize the full chat-templated prompt for this turn.
    /// 2. Look up `session_id`. If found and the cached prefix is a strict
    ///    prefix of the new prompt, only feed the suffix (`extend`) — the cache
    ///    state from prior turns is reused. If divergent, drop the session
    ///    (hybrid SSM/linear-attn caches cannot roll back).
    /// 3. Decode greedily until EOS or `max_new_tokens`.
    /// 4. Persist `prompt + generated` as the session's new token sequence.
    pub fn chat_streaming_session<F>(
        &mut self,
        messages: &[(String, String)],
        max_new_tokens: usize,
        thinking: bool,
        session_id: &str,
        mut on_token: F,
    ) -> Result<String>
    where
        F: FnMut(&str),
    {
        let prompt_ids = self.build_chat_input(messages, thinking)?;
        if prompt_ids.is_empty() {
            return Err(anyhow!("empty prompt after tokenization"));
        }

        self.evict_stale_sessions();

        // Decide: reuse existing session, or start fresh.
        let (seq_id, mut last, mut pos, fresh) = match self.sessions.get(session_id) {
            Some(state)
                if !state.tokens.is_empty()
                    && prompt_ids.len() > state.tokens.len()
                    && prompt_ids.starts_with(&state.tokens) =>
            {
                let suffix = prompt_ids[state.tokens.len()..].to_vec();
                let seq_id = state.seq_id;
                let cached_len = state.tokens.len();
                let t = std::time::Instant::now();
                let (last, pos) = self.runner.extend(seq_id, &suffix)?;
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                eprintln!(
                    "[mlx] seq {seq_id} session={session_id:?} reuse: cached={cached_len} \
                     suffix={} in {ms:.0}ms",
                    suffix.len()
                );
                (seq_id, last, pos, false)
            }
            _ => {
                // No usable session → drop any old one, alloc new.
                if let Some(old) = self.sessions.remove(session_id) {
                    let _ = self.runner.remove_seq(old.seq_id);
                    eprintln!(
                        "[mlx] session={session_id:?} divergent prompt; dropping old seq {} ({} tokens)",
                        old.seq_id,
                        old.tokens.len()
                    );
                }
                let seq_id = self.alloc_seq_id();
                let t = std::time::Instant::now();
                let (last, pos) = self.prefill(seq_id, &prompt_ids)?;
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                eprintln!(
                    "[mlx] seq {seq_id} session={session_id:?} fresh prefill: {} tokens in {ms:.0}ms",
                    prompt_ids.len()
                );
                (seq_id, last, pos, true)
            }
        };
        let _ = fresh;

        let mut generated: Vec<u32> = vec![last];
        let mut prev_text = String::new();
        if let Ok(text) = self.decode(&generated) {
            if !text.is_empty() && !text.contains('\u{FFFD}') {
                on_token(&text);
                prev_text = text;
            }
        }

        if !self.eos_tokens.contains(&last) {
            for _ in 1..max_new_tokens {
                let (next, new_pos) = self.decode_step(seq_id, last, pos)?;
                last = next;
                pos = new_pos;
                generated.push(next);
                if let Ok(text) = self.decode(&generated) {
                    if text.len() > prev_text.len() && !text.contains('\u{FFFD}') {
                        on_token(&text[prev_text.len()..]);
                        prev_text = text;
                    }
                }
                if self.eos_tokens.contains(&next) {
                    break;
                }
            }
        }

        let out = self.decode(&generated).unwrap_or_default();

        // Persist session: prompt + all generated tokens. Don't remove_seq —
        // the cache state is needed for the next turn.
        let mut new_tokens = prompt_ids;
        new_tokens.extend_from_slice(&generated);
        self.sessions.insert(
            session_id.to_string(),
            SessionState {
                seq_id,
                tokens: new_tokens,
                last_access: Instant::now(),
            },
        );

        Ok(out)
    }

    /// Non-streaming completion (raw text path used by `/v1/completions`) with
    /// prompt-cache reuse keyed by `session_id`. Same strict-prefix policy as
    /// `chat_streaming_session` — divergent prompts drop the session and
    /// re-prefill. Returns the generated token IDs (caller decodes).
    pub fn completion_session(
        &mut self,
        input_ids: &[u32],
        max_new_tokens: usize,
        session_id: &str,
    ) -> Result<Vec<u32>> {
        if input_ids.is_empty() {
            return Err(anyhow!("empty prompt"));
        }

        self.evict_stale_sessions();

        let prompt_ids: Vec<u32> = input_ids.to_vec();

        let (seq_id, mut last, mut pos) = match self.sessions.get(session_id) {
            Some(state)
                if !state.tokens.is_empty()
                    && prompt_ids.len() > state.tokens.len()
                    && prompt_ids.starts_with(&state.tokens) =>
            {
                let suffix = prompt_ids[state.tokens.len()..].to_vec();
                let seq_id = state.seq_id;
                let cached_len = state.tokens.len();
                let t = std::time::Instant::now();
                let (last, pos) = self.runner.extend(seq_id, &suffix)?;
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                eprintln!(
                    "[mlx] seq {seq_id} session={session_id:?} (completion) reuse: cached={cached_len} \
                     suffix={} in {ms:.0}ms",
                    suffix.len()
                );
                (seq_id, last, pos)
            }
            _ => {
                if let Some(old) = self.sessions.remove(session_id) {
                    let _ = self.runner.remove_seq(old.seq_id);
                    eprintln!(
                        "[mlx] session={session_id:?} (completion) divergent prompt; dropping old seq {} ({} tokens)",
                        old.seq_id,
                        old.tokens.len()
                    );
                }
                let seq_id = self.alloc_seq_id();
                let t = std::time::Instant::now();
                let (last, pos) = self.prefill(seq_id, &prompt_ids)?;
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                eprintln!(
                    "[mlx] seq {seq_id} session={session_id:?} (completion) fresh prefill: {} tokens in {ms:.0}ms",
                    prompt_ids.len()
                );
                (seq_id, last, pos)
            }
        };

        let mut generated: Vec<u32> = vec![last];
        if !self.eos_tokens.contains(&last) {
            for _ in 1..max_new_tokens {
                let (next, new_pos) = self.decode_step(seq_id, last, pos)?;
                last = next;
                pos = new_pos;
                generated.push(next);
                if self.eos_tokens.contains(&next) {
                    break;
                }
            }
        }

        let mut new_tokens = prompt_ids;
        new_tokens.extend_from_slice(&generated);
        self.sessions.insert(
            session_id.to_string(),
            SessionState {
                seq_id,
                tokens: new_tokens,
                last_access: Instant::now(),
            },
        );

        Ok(generated)
    }

    /// Drop a session's prompt cache. Returns true if the session existed.
    pub fn drop_session(&mut self, session_id: &str) -> bool {
        if let Some(state) = self.sessions.remove(session_id) {
            let _ = self.runner.remove_seq(state.seq_id);
            true
        } else {
            false
        }
    }

    /// Number of live sessions. Used by tests + diagnostics.
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Drop sessions idle past TTL, then enforce LRU cap if set. No-op when
    /// neither limit is configured. Called at the entry of every session-aware
    /// generation request.
    fn evict_stale_sessions(&mut self) {
        let now = Instant::now();
        let evicted =
            pick_eviction_victims(&self.sessions, now, self.session_ttl, self.session_max);
        for (key, reason) in evicted {
            if let Some(state) = self.sessions.remove(&key) {
                let _ = self.runner.remove_seq(state.seq_id);
                eprintln!("[mlx] session={key:?} evicted ({reason})");
            }
        }
    }
}

/// Pure helper for `evict_stale_sessions`: picks (key, human_reason) pairs
/// for every session that should be dropped at this moment. TTL victims first,
/// then LRU until `session_max` is satisfied. Pulled out so the policy is unit-
/// testable without spinning up a real runner.
fn pick_eviction_victims(
    sessions: &std::collections::HashMap<String, SessionState>,
    now: Instant,
    session_ttl: Option<Duration>,
    session_max: Option<usize>,
) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut alive: std::collections::HashMap<String, Instant> = sessions
        .iter()
        .map(|(k, s)| (k.clone(), s.last_access))
        .collect();

    if let Some(ttl) = session_ttl {
        let stale: Vec<String> = alive
            .iter()
            .filter(|&(_, t)| now.duration_since(*t) > ttl)
            .map(|(k, _)| k.clone())
            .collect();
        for k in stale {
            alive.remove(&k);
            out.push((k, format!("TTL > {}s", ttl.as_secs())));
        }
    }

    if let Some(max) = session_max {
        while alive.len() > max {
            let victim = alive
                .iter()
                .min_by_key(|&(_, t)| *t)
                .map(|(k, _)| k.clone());
            match victim {
                Some(k) => {
                    alive.remove(&k);
                    out.push((k, format!("LRU cap {max}")));
                }
                None => break,
            }
        }
    }

    out
}

/// Pure helper for `evict_stale_prefix_caches`: picks (key, human_reason)
/// pairs for every prefix-cache entry that should be dropped at this moment.
/// TTL victims first, then LRU until `max` is satisfied. Pulled out so the
/// policy is unit-testable without spinning up a real runner.
fn pick_prefix_eviction_victims(
    entries: &std::collections::HashMap<String, PrefixCacheEntry>,
    now: Instant,
    ttl: Option<Duration>,
    max: Option<usize>,
) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut alive: std::collections::HashMap<String, Instant> = entries
        .iter()
        .map(|(k, e)| (k.clone(), e.last_access))
        .collect();

    if let Some(ttl) = ttl {
        let stale: Vec<String> = alive
            .iter()
            .filter(|&(_, t)| now.duration_since(*t) > ttl)
            .map(|(k, _)| k.clone())
            .collect();
        for k in stale {
            alive.remove(&k);
            out.push((k, format!("TTL > {}s", ttl.as_secs())));
        }
    }

    if let Some(max) = max {
        while alive.len() > max {
            let victim = alive
                .iter()
                .min_by_key(|&(_, t)| *t)
                .map(|(k, _)| k.clone());
            match victim {
                Some(k) => {
                    alive.remove(&k);
                    out.push((k, format!("LRU cap {max}")));
                }
                None => break,
            }
        }
    }

    out
}

/// Qwen3 chat template — same as `format_qwen3_chat` in lumen-model.
fn format_qwen3_chat(messages: &[(String, String)], thinking: bool) -> String {
    let mut s = String::new();
    for (role, content) in messages {
        s.push_str("<|im_start|>");
        s.push_str(role);
        s.push('\n');
        s.push_str(content);
        s.push_str("<|im_end|>\n");
    }
    s.push_str("<|im_start|>assistant\n");
    if thinking {
        s.push_str("<think>\n");
    } else {
        s.push_str("<think>\n\n</think>\n\n");
    }
    s
}

/// Info returned by both runners after `load()`.
pub(crate) struct LoadInfo {
    pub eos_tokens: Vec<u32>,
    pub vocab_size: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Smoke test
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::golden::{
        DecodeRecord, PrefillRecord, RunnerTranscript, capture_runner_transcript,
        compare_runner_to_golden_transcript, load_runner_transcript,
    };

    #[cfg(feature = "mlx-native")]
    const DEFAULT_RUNNER: RunnerKind = RunnerKind::Native;
    #[cfg(not(feature = "mlx-native"))]
    const DEFAULT_RUNNER: RunnerKind = RunnerKind::Pyo3;

    #[test]
    fn runner_kind_default_matches_compiled_feature() {
        assert_eq!(runner_kind_from_env(None, None).unwrap(), DEFAULT_RUNNER);
    }

    #[test]
    fn runner_kind_accepts_explicit_backend() {
        assert_eq!(
            runner_kind_from_env(Some("subprocess"), None).unwrap(),
            RunnerKind::Subprocess
        );
        assert_eq!(
            runner_kind_from_env(Some("pyo3"), Some("1")).unwrap(),
            RunnerKind::Pyo3
        );
    }

    #[test]
    fn runner_kind_preserves_legacy_subprocess_env() {
        assert_eq!(
            runner_kind_from_env(None, Some("1")).unwrap(),
            RunnerKind::Subprocess
        );
        assert_eq!(
            runner_kind_from_env(None, Some("0")).unwrap(),
            DEFAULT_RUNNER
        );
    }

    #[test]
    fn runner_kind_accepts_native_backend() {
        assert_eq!(
            runner_kind_from_env(Some("native"), None).unwrap(),
            RunnerKind::Native
        );
    }

    #[cfg(not(feature = "mlx-native"))]
    #[test]
    fn native_runner_reports_missing_feature() {
        let err = match NativeMlxRunner::new() {
            Ok(_) => panic!("native runner should require the mlx-native feature"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("mlx-native"));
    }

    #[cfg(feature = "mlx-native")]
    #[test]
    fn native_runner_load_rejects_empty_model_id() {
        // The Phase 3e.b resolver short-circuits on empty model_id before
        // any network call, so this stays deterministic in the sandbox.
        let mut runner = NativeMlxRunner::new().unwrap();
        let err = match runner.load("") {
            Ok(_) => panic!("native runner load should fail on empty model_id"),
            Err(err) => err,
        };
        let s = err.to_string();
        assert!(
            s.contains("model_id is empty"),
            "unexpected error message: {s}"
        );
    }

    #[cfg(feature = "mlx-native")]
    #[test]
    fn native_runner_load_uses_local_directory_first() {
        // The Phase 3e.b resolver should prefer an existing local directory
        // over a HF Hub fallback — proves we don't call out to the network
        // when the model is already on disk. We point at a tempdir without
        // the required config.json so the call fails *locally* with a
        // typed error from NativeQwen3_5MoeModel::load (no HF Hub wrapper).
        let dir =
            std::env::temp_dir().join(format!("lumen_native_resolve_local_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);

        let mut runner = NativeMlxRunner::new().unwrap();
        let err = match runner.load(dir.to_str().unwrap()) {
            Ok(_) => panic!("native runner load should fail without config.json"),
            Err(err) => err,
        };
        let s = err.to_string();
        assert!(
            !s.contains("HF Hub") && !s.contains("hf_hub"),
            "resolver should not have hit HF Hub for local directory: {s}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn runner_transcript_roundtrips_json() {
        let transcript = sample_transcript();
        let json = serde_json::to_string_pretty(&transcript).unwrap();
        let decoded: RunnerTranscript = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, transcript);
    }

    #[test]
    fn runner_transcript_reports_decode_mismatch() {
        let golden = sample_transcript();
        let mut actual = golden.clone();
        actual.decode_steps[0].next_token += 1;

        let err = actual.compare_to_golden(&golden).unwrap_err();
        assert!(err.to_string().contains("decode step 0 mismatch"));
    }

    #[test]
    fn capture_runner_transcript_records_prefill_decode_and_removes_seq() {
        let mut runner = ScriptedRunner::default();
        let transcript =
            capture_runner_transcript(&mut runner, "test/model", 7, &[11, 22, 33], 2).unwrap();

        assert!(runner.removed);
        assert_eq!(transcript.runner, "scripted");
        assert_eq!(transcript.seq_id, 7);
        assert_eq!(transcript.prompt_tokens, vec![11, 22, 33]);
        assert_eq!(
            transcript.prefill,
            PrefillRecord {
                next_token: 101,
                position: 3
            }
        );
        assert_eq!(
            transcript.decode_steps,
            vec![
                DecodeRecord {
                    input_token: 101,
                    input_position: 3,
                    next_token: 102,
                    position: 4,
                },
                DecodeRecord {
                    input_token: 102,
                    input_position: 4,
                    next_token: 103,
                    position: 5,
                },
            ]
        );
    }

    #[test]
    fn parity_harness_compares_runner_to_saved_transcript() {
        let golden = scripted_runner_transcript();
        let root = std::env::temp_dir().join(format!(
            "lumen_mlx_parity_harness_{}.json",
            std::process::id()
        ));
        std::fs::write(&root, serde_json::to_string_pretty(&golden).unwrap()).unwrap();

        let loaded = load_runner_transcript(&root).unwrap();
        let mut runner = ScriptedRunner::default();
        compare_runner_to_golden_transcript(&mut runner, &loaded).unwrap();

        std::fs::remove_file(&root).unwrap();
    }

    #[cfg(feature = "mlx-native")]
    #[test]
    #[ignore = "loads full MLX model; set LUMEN_MLX_GOLDEN_IN to a PyO3 transcript JSON after native prefill/decode are implemented"]
    fn native_runner_matches_pyo3_golden_transcript() {
        let path = std::env::var("LUMEN_MLX_GOLDEN_IN")
            .expect("set LUMEN_MLX_GOLDEN_IN to a saved PyO3 runner transcript JSON");
        let golden = load_runner_transcript(path).unwrap();

        let mut runner = NativeMlxRunner::new().unwrap();
        runner.load(&golden.model_id).unwrap();
        compare_runner_to_golden_transcript(&mut runner, &golden).unwrap();
    }

    #[test]
    #[ignore = "spawns Python + downloads/loads ~19GB model"]
    fn smoke_load_prefill_decode() {
        let mut b = MlxBackend::load("mlx-community/Qwen3.6-35B-A3B-mxfp4").unwrap();
        let (first, pos) = b.prefill(1, &[12, 34, 56]).unwrap();
        assert!(pos == 3);
        let (_second, pos2) = b.decode_step(1, first, pos).unwrap();
        assert!(pos2 == 4);
        b.remove_seq(1).unwrap();
    }

    #[cfg(feature = "mlx-pyo3")]
    #[test]
    #[ignore = "spawns Python + downloads/loads ~19GB model; set LUMEN_MLX_GOLDEN_OUT to write JSON"]
    fn record_pyo3_golden_transcript() {
        let model_id = std::env::var("MODEL_ID")
            .unwrap_or_else(|_| "mlx-community/Qwen3.6-35B-A3B-mxfp4".to_string());
        let prompt_tokens = parse_prompt_tokens_env().unwrap_or_else(|| vec![12, 34, 56]);
        let decode_steps = std::env::var("STEPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);

        let mut runner = Pyo3Runner::new().unwrap();
        runner.load(&model_id).unwrap();
        let transcript =
            capture_runner_transcript(&mut runner, &model_id, 1, &prompt_tokens, decode_steps)
                .unwrap();
        let json = serde_json::to_string_pretty(&transcript).unwrap();

        if let Ok(path) = std::env::var("LUMEN_MLX_GOLDEN_OUT") {
            std::fs::write(&path, json).unwrap();
            eprintln!("wrote PyO3 MLX golden transcript to {path}");
        } else {
            println!("{json}");
        }
    }

    fn sample_transcript() -> RunnerTranscript {
        RunnerTranscript {
            schema_version: golden::TRANSCRIPT_SCHEMA_VERSION,
            runner: "pyo3".to_string(),
            model_id: "test/model".to_string(),
            seq_id: 1,
            prompt_tokens: vec![12, 34, 56],
            prefill: PrefillRecord {
                next_token: 100,
                position: 3,
            },
            decode_steps: vec![DecodeRecord {
                input_token: 100,
                input_position: 3,
                next_token: 101,
                position: 4,
            }],
        }
    }

    fn scripted_runner_transcript() -> RunnerTranscript {
        RunnerTranscript {
            schema_version: golden::TRANSCRIPT_SCHEMA_VERSION,
            runner: "pyo3".to_string(),
            model_id: "test/model".to_string(),
            seq_id: 1,
            prompt_tokens: vec![12, 34, 56],
            prefill: PrefillRecord {
                next_token: 101,
                position: 3,
            },
            decode_steps: vec![DecodeRecord {
                input_token: 101,
                input_position: 3,
                next_token: 102,
                position: 4,
            }],
        }
    }

    fn fake_session(seq_id: u64, last_access: Instant) -> SessionState {
        SessionState {
            seq_id,
            tokens: vec![1, 2, 3],
            last_access,
        }
    }

    #[test]
    fn pick_eviction_victims_no_limits_keeps_all() {
        let mut map = std::collections::HashMap::new();
        let now = Instant::now();
        map.insert("a".to_string(), fake_session(1, now));
        map.insert("b".to_string(), fake_session(2, now));
        let victims = pick_eviction_victims(&map, now, None, None);
        assert!(victims.is_empty());
    }

    #[test]
    fn pick_eviction_victims_ttl_drops_stale_only() {
        let mut map = std::collections::HashMap::new();
        let now = Instant::now();
        let stale = now - Duration::from_secs(60);
        map.insert("fresh".to_string(), fake_session(1, now));
        map.insert("stale".to_string(), fake_session(2, stale));

        let victims = pick_eviction_victims(&map, now, Some(Duration::from_secs(10)), None);
        let keys: Vec<&str> = victims.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, vec!["stale"]);
        assert!(victims[0].1.contains("TTL"));
    }

    #[test]
    fn pick_eviction_victims_lru_drops_oldest_until_cap() {
        let mut map = std::collections::HashMap::new();
        let now = Instant::now();
        map.insert(
            "oldest".to_string(),
            fake_session(1, now - Duration::from_secs(30)),
        );
        map.insert(
            "middle".to_string(),
            fake_session(2, now - Duration::from_secs(10)),
        );
        map.insert("newest".to_string(), fake_session(3, now));

        let victims = pick_eviction_victims(&map, now, None, Some(1));
        // Expect 2 victims (3 - 1 cap), oldest first.
        assert_eq!(victims.len(), 2);
        assert_eq!(victims[0].0, "oldest");
        assert_eq!(victims[1].0, "middle");
        assert!(victims[0].1.contains("LRU"));
    }

    #[test]
    fn pick_eviction_victims_ttl_then_lru() {
        // TTL drops one, LRU drops another from the survivors.
        let mut map = std::collections::HashMap::new();
        let now = Instant::now();
        map.insert(
            "expired".to_string(),
            fake_session(1, now - Duration::from_secs(120)),
        );
        map.insert(
            "oldish".to_string(),
            fake_session(2, now - Duration::from_secs(20)),
        );
        map.insert("fresh".to_string(), fake_session(3, now));

        let victims = pick_eviction_victims(&map, now, Some(Duration::from_secs(60)), Some(1));
        // expired: TTL. oldish: LRU (fresh survives since cap=1).
        assert_eq!(victims.len(), 2);
        let reasons: Vec<&str> = victims.iter().map(|(_, r)| r.as_str()).collect();
        assert!(reasons.iter().any(|r| r.contains("TTL")));
        assert!(reasons.iter().any(|r| r.contains("LRU")));
    }

    fn fake_prefix_entry(snapshot_id: u64, last_access: Instant) -> PrefixCacheEntry {
        PrefixCacheEntry {
            master_snapshot_id: snapshot_id,
            prefix_tokens: vec![1, 2, 3],
            last_access,
            hits: 0,
        }
    }

    #[test]
    fn pick_prefix_eviction_no_limits_keeps_all() {
        let mut map = std::collections::HashMap::new();
        let now = Instant::now();
        map.insert("a".to_string(), fake_prefix_entry(11, now));
        map.insert("b".to_string(), fake_prefix_entry(12, now));
        let victims = pick_prefix_eviction_victims(&map, now, None, None);
        assert!(victims.is_empty());
    }

    #[test]
    fn pick_prefix_eviction_lru_drops_oldest_until_cap() {
        let mut map = std::collections::HashMap::new();
        let now = Instant::now();
        map.insert(
            "old".to_string(),
            fake_prefix_entry(1, now - Duration::from_secs(30)),
        );
        map.insert("new".to_string(), fake_prefix_entry(2, now));
        let victims = pick_prefix_eviction_victims(&map, now, None, Some(1));
        let keys: Vec<&str> = victims.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, vec!["old"]);
        assert!(victims[0].1.contains("LRU"));
    }

    #[test]
    fn pick_prefix_eviction_ttl_drops_stale_only() {
        let mut map = std::collections::HashMap::new();
        let now = Instant::now();
        map.insert("fresh".to_string(), fake_prefix_entry(1, now));
        map.insert(
            "stale".to_string(),
            fake_prefix_entry(2, now - Duration::from_secs(120)),
        );
        let victims = pick_prefix_eviction_victims(&map, now, Some(Duration::from_secs(60)), None);
        let keys: Vec<&str> = victims.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, vec!["stale"]);
        assert!(victims[0].1.contains("TTL"));
    }

    #[test]
    fn auto_prefix_key_is_stable_within_process() {
        let m1 = vec![
            ("system".to_string(), "You are a helpful assistant.".into()),
            ("user".to_string(), "Hi".into()),
        ];
        let m2 = vec![
            ("system".to_string(), "You are a helpful assistant.".into()),
            ("user".to_string(), "Different user message.".into()),
        ];
        let m3 = vec![
            ("system".to_string(), "Different system prompt.".into()),
            ("user".to_string(), "Hi".into()),
        ];
        let k1 = auto_prefix_key(&m1).unwrap();
        let k2 = auto_prefix_key(&m2).unwrap();
        let k3 = auto_prefix_key(&m3).unwrap();
        assert_eq!(k1, k2, "same system prompt should produce same key");
        assert_ne!(
            k1, k3,
            "different system prompt should produce different key"
        );
    }

    #[test]
    fn auto_prefix_key_returns_none_without_system() {
        let m = vec![("user".to_string(), "Hi".into())];
        assert!(auto_prefix_key(&m).is_none());
        let empty: Vec<(String, String)> = vec![];
        assert!(auto_prefix_key(&empty).is_none());
        let empty_sys = vec![("system".to_string(), "".into())];
        assert!(auto_prefix_key(&empty_sys).is_none());
    }

    fn parse_prompt_tokens_env() -> Option<Vec<u32>> {
        let raw = std::env::var("PROMPT_TOKENS").ok()?;
        let tokens: Vec<u32> = raw
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::parse)
            .collect::<std::result::Result<_, _>>()
            .ok()?;
        (!tokens.is_empty()).then_some(tokens)
    }

    #[derive(Default)]
    struct ScriptedRunner {
        removed: bool,
    }

    impl Runner for ScriptedRunner {
        fn name(&self) -> &'static str {
            "scripted"
        }

        fn prefill(&mut self, _seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)> {
            Ok((101, tokens.len()))
        }

        fn decode_step(
            &mut self,
            _seq_id: u64,
            last_token: u32,
            position: usize,
        ) -> Result<(u32, usize)> {
            Ok((last_token + 1, position + 1))
        }

        fn remove_seq(&mut self, _seq_id: u64) -> Result<()> {
            self.removed = true;
            Ok(())
        }

        fn extend(&mut self, _seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)> {
            Ok((tokens.last().copied().unwrap_or(0) + 1, tokens.len()))
        }

        fn forward_probe(&mut self, _seq_id: u64, tokens: &[u32]) -> Result<ProbeRows> {
            let row_argmaxes: Vec<u32> = tokens.iter().map(|t| t + 1).collect();
            let row_max_abs = vec![1.0_f32; tokens.len()];
            Ok(ProbeRows {
                row_argmaxes,
                row_max_abs,
                position: tokens.len(),
            })
        }

        fn snapshot_state(&mut self, _seq_id: u64) -> Result<u64> {
            Ok(1)
        }

        fn restore_state(&mut self, _seq_id: u64, _snapshot_id: u64) -> Result<usize> {
            Ok(0)
        }

        fn release_snapshot(&mut self, _snapshot_id: u64) -> Result<()> {
            Ok(())
        }

        fn snapshot_state_deep(&mut self, _seq_id: u64) -> Result<(u64, usize)> {
            Ok((1, 0))
        }

        fn fork_from_snapshot(&mut self, _snapshot_id: u64, _dst_seq_id: u64) -> Result<usize> {
            Ok(0)
        }
    }
}
