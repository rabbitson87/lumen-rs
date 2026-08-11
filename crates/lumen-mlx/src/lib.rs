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
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use hf_hub::api::sync::ApiBuilder;
use tokenizers::Tokenizer;

pub mod chat_io;
mod gemma4_backend;
mod gemma4_chat;
mod gemma4_critical_correction;
mod gemma4_moe;
mod gemma4_mtp;
mod gemma4_response;
mod gemma4_sampling;
mod gemma4_thinking;
/// Gemma 4's tool-call body grammar. Ungated on purpose — see the module docs.
pub mod gemma4_tool_syntax;
mod gemma4_tools;
pub mod gemma4_vision;
pub mod grammar;
mod jinja_chat;
pub mod qwen36_vision;
/// Resource-bounded image decoding shared by both image towers.
mod vision_image;
/// Placeholder-run bookkeeping shared by both image towers.
mod vision_splice;
// On-disk KV-cache persistence primitives (P0). Pure format/store layer
// compiles under default features; Array<->record conversion is gated behind
// `mlx-native`. Phase 1 wires this into the prefix-cache disk tier.
pub mod kv_disk;

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
        NativeGemma4Model, NativeGemma4PromptCache, quant_params_for, set_forward_step,
        take_gemma4_breakdown,
    };
    pub use crate::gemma4_response::imp::{
        ParseState, ParsedResponse, ParsedToolCall, ResponseParser, TOK_TOOL_CALL_CLOSE,
        TOK_TOOL_CALL_OPEN, gemma4_args_to_json, parse_tool_call_body,
    };
    pub use crate::gemma4_tools::imp::{
        ToolDef, format_tool_call_body, render_tool_definitions, render_tool_definitions_text,
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
/// Qwen3-Embedding on mlx-rs — the `/v1/embeddings` backend.
pub mod embedding;
pub mod env_state;
mod golden;
pub mod metal_kernel;
pub mod native_attention;
mod native_cache;
/// Block granularity the per-sequence full-attention KV cache grows in
/// (`mlx_lm.cache.KVCache(step=256)` semantics). Exposed for harnesses that
/// reason about allocation rounding — see `examples/kv_concurrency_ab.rs`.
#[cfg(feature = "mlx-native")]
pub use native_cache::KV_CACHE_STEP;
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
mod prefix_cache;
mod qwen3_5_moe;
// Unlike its `qwen3_5_moe` sibling — which keeps a `#[cfg]`'d `mod imp` inside
// an ungated file — every item here takes or returns `mlx_rs::Array`, so the
// whole module is gated at the declaration (same shape as
// `native_metal_bridge` above). Without this the crate does not build under
// its own `default = []`, which is the configuration the pure modules
// (`grammar`, `chat_io`, `gemma4_tool_syntax`) exist to be testable in.
#[cfg(feature = "mlx-native")]
mod qwen3_5_mtp;
mod qwen3_5_tools;
// Phase 2 Step B microbench — synthetic-weight latency probe at
// Qwen3.6-35B-A3B-mxfp4 shapes. Internal API used by
// `examples/bench_qwen35_mtp_step_b.rs` to validate the K=2 vs K=3 cycle
// math before investing in the HF-native loader + runner wiring.
#[cfg(feature = "mlx-native")]
pub use qwen3_5_moe::MtpStepOutput;
#[cfg(feature = "mlx-native")]
pub use qwen3_5_moe::set_kv_store_bf16;

/// Qwen 3.5/3.6 `config.json` parsing. Public so the Phase 3 fault sweep can
/// drive the loader that every downloaded checkpoint hits first.
#[cfg(feature = "mlx-native")]
pub mod qwen3_5_moe_config {
    pub use crate::qwen3_5_moe::{
        NativeLayerType, NativeModelConfig, NativeTextConfig, NativeWeights,
    };
}

/// Link anchor for `examples/dump_flags.rs`. The flag registry is collected by
/// linkme at link time, but an rlib's objects are only linked when referenced —
/// a binary that touches nothing in this crate sees an empty registry. Calling
/// this (a no-op) forces the linkage.
#[cfg(feature = "mlx-native")]
pub fn flags_link_anchor() {}
#[cfg(feature = "mlx-native")]
pub use qwen3_5_mtp::{
    HiTrainCfg, MtpLoadQuant, MtpLoraPos, MtpMlpConfig, MtpMoeConfig, Qwen35MtpBlock,
    Qwen35MtpDims, StepBBenchPoint, load_block_from_hf, run_step_b_synthetic_bench,
    smoke_forward_with_synth_trunk,
};
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

/// Per-request sampling overrides. `None` = fall back to env/family default.
/// Extend with new fields (min_p, penalties, stop, logit_bias) without
/// re-threading — only `build_sampling_config` reads them.
#[derive(Clone, Default, Debug)]
pub struct SamplingOverrides {
    pub top_k: Option<usize>,
    pub seed: Option<u64>,
    pub repeat_penalty: Option<f32>,
    pub min_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    /// Stop strings (OpenAI `stop` / Anthropic `stop_sequences`). Matched
    /// incrementally in the streaming loop, not in `build_sampling_config`.
    pub stop: Vec<String>,
}

/// Output of `Runner::forward_probe`: per-row argmaxes + max-abs logit + new
/// position. Used by Track A2 drift baseline + spec-decode verify-loop.
#[derive(Clone, Debug)]
pub struct ProbeRows {
    pub row_argmaxes: Vec<u32>,
    pub row_max_abs: Vec<f32>,
    /// Per-row `top1 - top2` logit gap — how decisively the argmax won.
    ///
    /// Needed to interpret an argmax disagreement between two numeric
    /// configurations: a flip where the gap was ~0 is a tie broken differently,
    /// which is what any perturbation is expected to do and says nothing about
    /// quality; a flip at a large gap is a genuinely changed prediction. Used
    /// by `examples/kv_bf16_quality.rs` to tell those apart.
    ///
    /// Empty when the runner does not compute it (the PyO3 path); check the
    /// length before indexing.
    pub row_top2_gap: Vec<f32>,
    pub position: usize,
}

trait Runner {
    fn name(&self) -> &'static str;
    fn prefill(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)>;
    /// [`Runner::prefill`] with encoded images spliced over their placeholder
    /// runs.
    ///
    /// Only the native runner implements this. The others reject rather than
    /// prefilling a prompt whose image placeholders would never be filled —
    /// which would answer confidently about an image the model never saw.
    #[cfg(feature = "mlx-native")]
    fn prefill_with_images(
        &mut self,
        _seq_id: u64,
        _tokens: &[u32],
        _images: &[crate::qwen36_vision::PreparedImage],
    ) -> Result<(u32, usize)> {
        Err(anyhow!(
            "image input requires the native MLX runner (LUMEN_MLX_BACKEND=native)"
        ))
    }
    fn decode_step(
        &mut self,
        seq_id: u64,
        last_token: u32,
        position: usize,
    ) -> Result<(u32, usize)>;
    /// Phase 1: advance N sequences by one token each, returning
    /// `(next_token, position)` per input index. The default implementation
    /// loops over [`Runner::decode_step`] and is therefore **bit-identical** to
    /// N independent single-seq decodes — it lets a multi-sequence scheduler
    /// drive all active seqs in one tick. The native runner overrides this with
    /// a single batched forward (Phase 1b: the model trunk + SSM kernel are
    /// already batch-shaped), keeping this signature stable across that swap.
    /// `seq_ids`, `last_tokens`, `positions` must be equal length, aligned by
    /// index.
    fn decode_step_batch(
        &mut self,
        seq_ids: &[u64],
        last_tokens: &[u32],
        positions: &[usize],
    ) -> Result<Vec<(u32, usize)>> {
        if seq_ids.len() != last_tokens.len() || seq_ids.len() != positions.len() {
            return Err(anyhow!(
                "decode_step_batch: mismatched input lengths \
                 (seqs={}, tokens={}, positions={})",
                seq_ids.len(),
                last_tokens.len(),
                positions.len()
            ));
        }
        let mut out = Vec::with_capacity(seq_ids.len());
        for i in 0..seq_ids.len() {
            out.push(self.decode_step(seq_ids[i], last_tokens[i], positions[i])?);
        }
        Ok(out)
    }
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

    // Gated to match the trait declaration — `Runner::prefill_with_images`
    // only exists under `mlx-native`, so an ungated impl is not overriding
    // anything. The `MlxRunnerEnum` impl below already carries this gate.
    #[cfg(feature = "mlx-native")]
    fn prefill_with_images(
        &mut self,
        seq_id: u64,
        tokens: &[u32],
        images: &[crate::qwen36_vision::PreparedImage],
    ) -> Result<(u32, usize)> {
        NativeMlxRunner::prefill_with_images(self, seq_id, tokens, images)
    }

    fn decode_step(
        &mut self,
        seq_id: u64,
        last_token: u32,
        position: usize,
    ) -> Result<(u32, usize)> {
        NativeMlxRunner::decode_step(self, seq_id, last_token, position)
    }

    fn decode_step_batch(
        &mut self,
        seq_ids: &[u64],
        last_tokens: &[u32],
        positions: &[usize],
    ) -> Result<Vec<(u32, usize)>> {
        // Phase 1b override: true batched forward (vs the trait default loop).
        NativeMlxRunner::decode_step_batch(self, seq_ids, last_tokens, positions)
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

/// Resolve the on-disk `tokenizer.json` path for `model_id`, mirroring
/// [`load_tokenizer_via_hub`]'s lookup: a local directory's `tokenizer.json`
/// first, else the HF-Hub-cached download. Used to build the llguidance parser
/// factory for the Qwen 3.6 tool grammar (WS-C #2) — llguidance needs the raw
/// JSON path (not the in-memory `Tokenizer`) to recover the byte-level token
/// table its masks index into.
fn resolve_tokenizer_json_path(model_id: &str) -> Result<PathBuf> {
    let local = std::path::Path::new(model_id);
    if local.is_dir() {
        let tj = local.join("tokenizer.json");
        if tj.is_file() {
            return Ok(tj);
        }
    }
    let api = ApiBuilder::new().build().context("hf_hub api init")?;
    let repo = api.model(model_id.to_string());
    repo.get("tokenizer.json")
        .context("download tokenizer.json for grammar factory")
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

    #[cfg(feature = "mlx-native")]
    fn prefill_with_images(
        &mut self,
        seq_id: u64,
        tokens: &[u32],
        images: &[crate::qwen36_vision::PreparedImage],
    ) -> Result<(u32, usize)> {
        self.as_runner_mut()
            .prefill_with_images(seq_id, tokens, images)
    }

    /// Decode + resize one image onto the tower's patch grid. Native only —
    /// the other runners have no tower to size against.
    #[cfg(feature = "mlx-native")]
    fn prepare_image(&self, encoded: &[u8]) -> Result<crate::qwen36_vision::PreparedImage> {
        match self {
            Self::Native(r) => r.prepare_image(encoded),
            _ => Err(anyhow!(
                "image input requires the native MLX runner (LUMEN_MLX_BACKEND=native)"
            )),
        }
    }

    /// Token id whose embedding rows the vision features replace.
    #[cfg(feature = "mlx-native")]
    fn image_token_id(&self) -> Option<u32> {
        match self {
            Self::Native(r) => r.image_token_id(),
            _ => None,
        }
    }

    /// The loaded tower's config, for header-only prompt sizing.
    #[cfg(feature = "mlx-native")]
    fn vision_config(&self) -> Option<crate::qwen36_vision::NativeQwen36VisionConfig> {
        match self {
            Self::Native(r) => r.vision_config(),
            _ => None,
        }
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

    fn decode_step_batch(
        &mut self,
        seq_ids: &[u64],
        last_tokens: &[u32],
        positions: &[usize],
    ) -> Result<Vec<(u32, usize)>> {
        // Forward to the active runner so the native runner's batched override
        // (Phase 1b) is used; the default loop only applies to runners that
        // don't override it.
        self.as_runner_mut()
            .decode_step_batch(seq_ids, last_tokens, positions)
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

    /// L2 disk tier — persist a snapshot durably. Native-only; other backends
    /// are no-ops (their snapshot state is not serializable here).
    fn persist_snapshot_disk(
        &mut self,
        snapshot_id: u64,
        key: &str,
        prefix_tokens: &[u32],
        last_token: Option<u32>,
    ) -> Result<()> {
        match self {
            Self::Native(r) => r.persist_snapshot_disk(snapshot_id, key, prefix_tokens, last_token),
            _ => Ok(()),
        }
    }

    /// L2 disk tier — rehydrate a persisted snapshot. Native-only; other
    /// backends always miss.
    fn load_persisted_disk(
        &mut self,
        key: &str,
        prompt_ids: &[u32],
    ) -> Result<Option<(u64, Vec<u32>, Option<u32>)>> {
        match self {
            Self::Native(r) => r.load_persisted_disk(key, prompt_ids),
            _ => Ok(None),
        }
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

    /// Grammar-aware decode step (WS-C #2). Native-only. Runs the normal
    /// decode forward (KV + linear-attn conv/SSM state advance identically),
    /// then masks the host logits via `mask` before argmaxing — so the
    /// grammar only changes WHICH token is sampled, never the recurrent state.
    /// Errors on non-native backends (grammar is only wired on the native MLX
    /// path).
    #[cfg(feature = "mlx-native")]
    fn decode_step_masked(
        &mut self,
        seq_id: u64,
        last_token: u32,
        mask: &mut dyn FnMut(&mut [f32]) -> Result<()>,
    ) -> Result<(u32, usize)> {
        match self {
            Self::Native(r) => r.decode_step_masked(seq_id, last_token, mask),
            _ => Err(anyhow!(
                "decode_step_masked (grammar) is only supported on the native (mlx-rs) backend; \
                 set LUMEN_MLX_BACKEND=native"
            )),
        }
    }

    /// Install a Qwen3.5/3.6 MTP block onto the active runner. Native-only.
    /// Phase 2 S3 wiring — see `qwen3_5_moe::mtp_step` for cycle semantics.
    #[cfg(feature = "mlx-native")]
    fn enable_qwen35_mtp(&mut self, block: crate::qwen3_5_mtp::Qwen35MtpBlock) -> Result<()> {
        match self {
            Self::Native(r) => r.enable_mtp(block),
            _ => Err(anyhow!(
                "enable_qwen35_mtp is only supported on the native (mlx-rs) backend; \
                 set LUMEN_MLX_BACKEND=native"
            )),
        }
    }

    #[cfg(feature = "mlx-native")]
    fn qwen35_mtp_enabled(&self) -> bool {
        match self {
            Self::Native(r) => r.mtp_enabled(),
            _ => false,
        }
    }

    #[cfg(feature = "mlx-native")]
    fn enable_mtp_calib(&self, depth_count: usize) -> Result<()> {
        match self {
            Self::Native(r) => r.enable_mtp_calib(depth_count),
            _ => Err(anyhow!("enable_mtp_calib requires the native backend")),
        }
    }

    #[cfg(feature = "mlx-native")]
    fn finalize_mtp_calib(&self, eps: f32) -> Result<Vec<u8>> {
        match self {
            Self::Native(r) => r.finalize_mtp_calib(eps),
            _ => Err(anyhow!("finalize_mtp_calib requires the native backend")),
        }
    }

    #[cfg(feature = "mlx-native")]
    fn set_mtp_corrector_bytes(&self, bytes: &[u8]) -> Result<()> {
        match self {
            Self::Native(r) => r.set_mtp_corrector_bytes(bytes),
            _ => Err(anyhow!("set_mtp_corrector requires the native backend")),
        }
    }

    #[cfg(feature = "mlx-native")]
    fn enable_mtp_proc_calib(&self, depth_count: usize) -> Result<()> {
        match self {
            Self::Native(r) => r.enable_mtp_proc_calib(depth_count),
            _ => Err(anyhow!("enable_mtp_proc_calib requires the native backend")),
        }
    }

    #[cfg(feature = "mlx-native")]
    fn finalize_mtp_proc_calib(&self) -> Result<Vec<u8>> {
        match self {
            Self::Native(r) => r.finalize_mtp_proc_calib(),
            _ => Err(anyhow!(
                "finalize_mtp_proc_calib requires the native backend"
            )),
        }
    }

    #[cfg(feature = "mlx-native")]
    fn set_mtp_procrustes_bytes(&self, bytes: &[u8]) -> Result<()> {
        match self {
            Self::Native(r) => r.set_mtp_procrustes_bytes(bytes),
            _ => Err(anyhow!("set_mtp_procrustes requires the native backend")),
        }
    }

    #[cfg(feature = "mlx-native")]
    fn enable_mtp_c4_calib(&self) -> Result<()> {
        match self {
            Self::Native(r) => r.enable_mtp_c4_calib(),
            _ => Err(anyhow!("enable_mtp_c4_calib requires the native backend")),
        }
    }

    #[cfg(feature = "mlx-native")]
    fn train_mtp_c4_lmhead(&self, rank: usize, steps: usize, lr: f32) -> Result<()> {
        match self {
            Self::Native(r) => r.train_mtp_c4_lmhead(rank, steps, lr),
            _ => Err(anyhow!("train_mtp_c4_lmhead requires the native backend")),
        }
    }

    #[cfg(feature = "mlx-native")]
    fn enable_mtp_hi_calib(&self) -> Result<()> {
        match self {
            Self::Native(r) => r.enable_mtp_hi_calib(),
            _ => Err(anyhow!("enable_mtp_hi_calib requires the native backend")),
        }
    }

    #[cfg(feature = "mlx-native")]
    fn train_mtp_headinternal(&self, cfg: crate::qwen3_5_mtp::HiTrainCfg) -> Result<(f32, f32)> {
        match self {
            Self::Native(r) => r.train_mtp_headinternal(cfg),
            _ => Err(anyhow!(
                "train_mtp_headinternal requires the native backend"
            )),
        }
    }

    #[cfg(feature = "mlx-native")]
    fn qwen35_mtp_step(
        &mut self,
        seq_id: u64,
        committed_token: u32,
        n_draft: usize,
        temperature: f32,
        top_p: f32,
    ) -> Result<crate::qwen3_5_moe::MtpStepOutput> {
        match self {
            Self::Native(r) => r.mtp_step(seq_id, committed_token, n_draft, temperature, top_p),
            _ => Err(anyhow!(
                "qwen35_mtp_step is only supported on the native (mlx-rs) backend; \
                 set LUMEN_MLX_BACKEND=native"
            )),
        }
    }
}

/// `RunnerImpl` exposes the narrow `SnapshotRunner` surface the prefix cache
/// needs, delegating to the active variant's `Runner` impl. Keeps `prefix_cache`
/// decoupled from the runner enum — the store works against these five methods.
impl prefix_cache::SnapshotRunner for RunnerImpl {
    fn prefill(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)> {
        self.as_runner_mut().prefill(seq_id, tokens)
    }
    fn extend(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)> {
        self.as_runner_mut().extend(seq_id, tokens)
    }
    fn snapshot_state_deep(&mut self, seq_id: u64) -> Result<(u64, usize)> {
        self.as_runner_mut().snapshot_state_deep(seq_id)
    }
    fn fork_from_snapshot(&mut self, snapshot_id: u64, dst_seq_id: u64) -> Result<usize> {
        self.as_runner_mut()
            .fork_from_snapshot(snapshot_id, dst_seq_id)
    }
    fn release_snapshot(&mut self, snapshot_id: u64) -> Result<()> {
        self.as_runner_mut().release_snapshot(snapshot_id)
    }
    // L2 disk tier hooks (tools path). Forward to the inherent methods, which
    // are no-ops on non-native backends / when the disk tier is off.
    fn persist_snapshot(
        &mut self,
        snapshot_id: u64,
        key: &str,
        prefix_tokens: &[u32],
        last_token: Option<u32>,
    ) -> Result<()> {
        RunnerImpl::persist_snapshot_disk(self, snapshot_id, key, prefix_tokens, last_token)
    }
    fn load_persisted_snapshot(
        &mut self,
        key: &str,
        prompt_ids: &[u32],
    ) -> Result<Option<(u64, Vec<u32>, Option<u32>)>> {
        RunnerImpl::load_persisted_disk(self, key, prompt_ids)
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

/// Compute a process-stable cache key for a chat request based on the system
/// message content. Returns `None` if there is no system message to share —
/// the prefix cache only ever shares the system-prompt block.
/// True when `dir` carries a self-contained MTP head — the `mtp/weights.safetensors`
/// sidecar that MTPLX repos (e.g. Qwen3.6-27B-MTPLX) ship, or a flat
/// `mtp.safetensors`. Used to AUTO-enable MTP at load without env vars.
#[cfg(feature = "mlx-native")]
fn model_has_mtp_head(dir: &std::path::Path) -> bool {
    dir.join("mtp").join("weights.safetensors").is_file() || dir.join("mtp.safetensors").is_file()
}

/// Resolve the effective MTP draft count for the streaming decode, assuming the
/// caller has already confirmed the head is installed (`qwen35_mtp_enabled()`).
/// `LUMEN_SPEC=mtp` → explicit on (K = `LUMEN_SPEC_K`, default 2). `LUMEN_SPEC`
/// set to anything else → `None` (defer to that mode). UNSET → AUTO-route
/// through MTP (K = `LUMEN_SPEC_K`, default 1 — the dense-27B accept sweet spot),
/// so a self-contained MTP model serves at spec speed with no env vars.
#[cfg(feature = "mlx-native")]
fn effective_qwen35_mtp_k() -> Option<usize> {
    let k_env = || {
        std::env::var("LUMEN_SPEC_K")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&v| v >= 1)
    };
    match std::env::var("LUMEN_SPEC") {
        Ok(s) if s.eq_ignore_ascii_case("mtp") => Some(k_env().unwrap_or(2)),
        Ok(_) => None,
        Err(_) => Some(k_env().unwrap_or(1)),
    }
}

/// Should prefill stop one token short so the first *generated* token can be
/// masked?
///
/// Prefill argmaxes its final position with no grammar mask, so with an active
/// grammar the first generated token is the one token the matcher never
/// constrains. Holding the last prompt token back and feeding it through the
/// masked decode step fixes that — but only when a grammar is actually active,
/// so the unconstrained path stays byte-identical, and only when the held-back
/// token is not an image placeholder, which would tear a placeholder run out of
/// the span the images are spliced over.
#[cfg(feature = "mlx-native")]
fn hold_back_last_prompt_token(
    grammar_active: bool,
    prompt_ids: &[u32],
    image_token: Option<u32>,
) -> bool {
    grammar_active && prompt_ids.len() > 1 && image_token != prompt_ids.last().copied()
}

/// Would a Gemma 4 request get a grammar if it went through the streaming
/// decode?
///
/// The non-streaming path runs a batched `generate()` that applies no mask, so
/// this is what decides whether a request must be re-routed through streaming
/// to get the constraint it asked for. It mirrors the conditions
/// `Gemma4Backend::select_grammar_state` uses — response_format unconditionally,
/// tools only when the Lark grammar is enabled and `tool_choice` is not
/// `"none"` — because a mismatch either loses the grammar again or pays a cold
/// prefill for nothing.
#[cfg(feature = "mlx-native")]
fn gemma4_grammar_would_constrain(
    response_schema: Option<&serde_json::Value>,
    tools: &[crate::chat_io::ToolDef<'_>],
    tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
) -> bool {
    if response_schema.is_some() {
        return true;
    }
    !tools.is_empty()
        && !matches!(tool_choice, crate::chat_io::ResolvedToolChoice::None)
        && crate::gemma4_backend::imp::gemma4_grammar_lark_enabled()
}

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

/// `auto_prefix_key` for the structured-history shape. Hashes the first
/// turn's content iff it's a `System` turn. Returns `None` when the chat
/// starts with `User` (no shared prefix worth caching) or when the system
/// content is empty.
fn auto_prefix_key_from_turns(turns: &[crate::chat_io::ChatTurn<'_>]) -> Option<String> {
    use crate::chat_io::ChatTurn;
    let content = match turns.first()? {
        ChatTurn::System(s) if !s.is_empty() => *s,
        _ => return None,
    };
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    content.hash(&mut h);
    Some(format!("auto-{:016x}", h.finish()))
}

/// Whether force-required-params injection is enabled (default off). When on,
/// the Qwen3.6 tool decode loop injects a `<parameter=KEY>\n` opener whenever
/// the model is about to close a `<function=NAME>` block with a REQUIRED param
/// still missing — turning the model's empty `<function=read></function>`
/// (→ "path is required") into a structurally valid call whose value the model
/// still writes itself. See [`MlxBackend::chat_with_tools_impl`].
fn force_required_params_enabled() -> bool {
    std::env::var("LUMEN_QWEN35_FORCE_REQUIRED_PARAMS")
        .map(|v| v == "1" || v == "on" || v == "true")
        .unwrap_or(false)
}

/// Build the force-required map (tool name → required param keys, schema order)
/// from the request's tool defs. Only the `required` array of each tool's JSON
/// Schema is used; tools with no required params are omitted.
fn force_required_params_map(
    tools: &[crate::chat_io::ToolDef<'_>],
) -> std::collections::HashMap<String, Vec<String>> {
    let mut map = std::collections::HashMap::new();
    for t in tools {
        let Some(params) = t.parameters else { continue };
        let Some(req) = params.get("required").and_then(|v| v.as_array()) else {
            continue;
        };
        let keys: Vec<String> = req
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        if !keys.is_empty() {
            map.insert(t.name.to_string(), keys);
        }
    }
    map
}

/// Whether the Qwen 3.6 tool-call grammar (WS-C #2) is enabled. **Default ON**
/// for `tool_choice=required`/named so the quantized-35B empty-parameter defect
/// is structurally blocked. Set `LUMEN_QWEN35_TOOL_GRAMMAR=0` to disable (A/B
/// or if a grammar build issue is suspected). Independent of
/// `LUMEN_QWEN35_FORCE_REQUIRED_PARAMS` (the older heuristic injector) — the
/// grammar supersedes it but both can run; the grammar's required-key
/// enforcement is the stronger guarantee.
fn qwen35_tool_grammar_enabled() -> bool {
    std::env::var("LUMEN_QWEN35_TOOL_GRAMMAR")
        .map(|v| !(v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false")))
        .unwrap_or(true)
}

/// Whether the Qwen 3.6 `response_format` (JSON-schema) grammar is enabled.
/// **Default ON** when a request carries a JSON schema — `response_format` is an
/// explicit per-request client signal, so the constraint is opt-in by the
/// caller. Set `LUMEN_QWEN35_RESPONSE_FORMAT=0` to disable as a safety
/// kill-switch (A/B or if a grammar build issue is suspected); the request then
/// decodes free text (the server still trims to the first JSON value). Mirrors
/// the `LUMEN_QWEN35_TOOL_GRAMMAR` style.
fn qwen35_response_format_enabled() -> bool {
    std::env::var("LUMEN_QWEN35_RESPONSE_FORMAT")
        .map(|v| !(v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false")))
        .unwrap_or(true)
}

/// Opt-in: also constrain the `tool_choice=auto` path with an **Eager**
/// grammar (active from token 0). Default OFF — auto requests should be free to
/// emit plain text and only fall into tool-call structure when the model
/// chooses to. Mirrors `LUMEN_GEMMA4_TOOL_GRAMMAR_EAGER`. Required/named are
/// always Eager regardless of this flag.
fn qwen35_tool_grammar_eager_auto() -> bool {
    std::env::var("LUMEN_QWEN35_TOOL_GRAMMAR_EAGER")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("on") || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
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

/// Unified mlx-native backend. Wraps either the Qwen3.5/3.6 family or the
/// Gemma 4 family — both run on the same mlx-rs FFI + Metal kernels +
/// shared `native_*` infrastructure (cache, attention, RoPE, RMSNorm). The
/// only difference is the model-assembly module on top, which has its own
/// config schema + tokenizer + chat template per family.
///
/// Adding a new family:
/// 1. Drop a `<family>_moe.rs` (or equivalent) next to `qwen3_5_moe.rs`
///    using the shared `native_*` primitives
/// 2. Add a variant to this enum + `MlxBackendKind`
/// 3. Wire arch detection in `MlxBackend::load`
/// Phase 3: the minimal per-seq driver surface the batched MLX scheduler
/// (`lumen-server::engine::run_batched_mlx`) needs, so one scheduler works for
/// any MLX family (Qwen 3.6 + Gemma 4). Both backends implement it by forwarding
/// to their inherent methods — the trait only exists so the scheduler can hold
/// one `&mut dyn` regardless of family.
pub trait MlxBatchedSeqDriver {
    fn build_chat_input(&self, messages: &[(String, String)], thinking: bool) -> Result<Vec<u32>>;
    fn eos_tokens(&self) -> &[u32];
    fn prefill(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)>;
    fn decode_step_batch(
        &mut self,
        seq_ids: &[u64],
        last_tokens: &[u32],
        positions: &[usize],
    ) -> Result<Vec<(u32, usize)>>;
    fn decode(&self, tokens: &[u32]) -> Result<String>;
    fn remove_seq(&mut self, seq_id: u64) -> Result<()>;
    /// Decode generated tokens into `(visible, reasoning)` channel strings for
    /// channel-aware batched streaming. Default: everything is visible (plain
    /// models like Qwen 3.6 greedy carry no special reasoning-channel tokens).
    /// Gemma 4 overrides this to split on its `<|channel>…<channel|>` reasoning
    /// markers so the batched streamer doesn't leak `thought` content into the
    /// visible delta the way a flat `decode()` diff would.
    fn stream_channels(&self, generated: &[u32]) -> Result<(String, String)> {
        Ok((self.decode(generated)?, String::new()))
    }
}

impl MlxBatchedSeqDriver for MlxQwen35Backend {
    fn build_chat_input(&self, m: &[(String, String)], t: bool) -> Result<Vec<u32>> {
        MlxQwen35Backend::build_chat_input(self, m, t)
    }
    fn eos_tokens(&self) -> &[u32] {
        MlxQwen35Backend::eos_tokens(self)
    }
    fn prefill(&mut self, s: u64, t: &[u32]) -> Result<(u32, usize)> {
        MlxQwen35Backend::prefill(self, s, t)
    }
    fn decode_step_batch(
        &mut self,
        s: &[u64],
        l: &[u32],
        p: &[usize],
    ) -> Result<Vec<(u32, usize)>> {
        MlxQwen35Backend::decode_step_batch(self, s, l, p)
    }
    fn decode(&self, t: &[u32]) -> Result<String> {
        MlxQwen35Backend::decode(self, t)
    }
    fn remove_seq(&mut self, s: u64) -> Result<()> {
        MlxQwen35Backend::remove_seq(self, s)
    }
}

#[cfg(feature = "mlx-native")]
impl MlxBatchedSeqDriver for crate::gemma4::Gemma4Backend {
    fn build_chat_input(&self, m: &[(String, String)], t: bool) -> Result<Vec<u32>> {
        crate::gemma4::Gemma4Backend::build_chat_input(self, m, t)
    }
    fn eos_tokens(&self) -> &[u32] {
        crate::gemma4::Gemma4Backend::eos_tokens(self)
    }
    fn prefill(&mut self, s: u64, t: &[u32]) -> Result<(u32, usize)> {
        crate::gemma4::Gemma4Backend::prefill(self, s, t)
    }
    fn decode_step_batch(
        &mut self,
        s: &[u64],
        l: &[u32],
        p: &[usize],
    ) -> Result<Vec<(u32, usize)>> {
        crate::gemma4::Gemma4Backend::decode_step_batch(self, s, l, p)
    }
    fn decode(&self, t: &[u32]) -> Result<String> {
        crate::gemma4::Gemma4Backend::decode(self, t)
    }
    fn remove_seq(&mut self, s: u64) -> Result<()> {
        crate::gemma4::Gemma4Backend::remove_seq(self, s)
    }
    fn stream_channels(&self, g: &[u32]) -> Result<(String, String)> {
        crate::gemma4::Gemma4Backend::stream_channels(self, g)
    }
}

pub enum MlxBackend {
    /// Qwen 2.5 / 3.5 / 3.6 (dense + MoE). Loaded via `NativeMlxRunner`
    /// (or the legacy pyo3 / subprocess runners for fallback).
    Qwen35Family(MlxQwen35Backend),
    /// Gemma 4 26B-A4B MoE (3-bit / 4-bit MLX shards).
    #[cfg(feature = "mlx-native")]
    Gemma4(crate::gemma4::Gemma4Backend),
}

impl MlxBackend {
    /// Whether the loaded backend is a Gemma 4 family model. Used by the
    /// HTTP layer to enable thinking-mode by default — Gemma 4's chat
    /// template needs `<|think|>` injection to activate the reasoning
    /// block, and OpenAI-compat clients (Ayla, Moltis) routinely
    /// hard-code `model: "gpt-3.5-turbo"` so we can't infer the family
    /// from the request payload.
    pub fn is_gemma4(&self) -> bool {
        #[cfg(feature = "mlx-native")]
        {
            matches!(self, Self::Gemma4(_))
        }
        #[cfg(not(feature = "mlx-native"))]
        {
            false
        }
    }

    /// Load a model, picking the correct family path from `model_id` (a
    /// local directory or an HF Hub repo id). Family detection is
    /// substring-based; explicit override via `LUMEN_MLX_FAMILY=gemma4|qwen`
    /// is accepted as an escape hatch.
    pub fn load(model_id: &str) -> Result<Self> {
        let family = detect_mlx_family(model_id);
        match family {
            #[cfg(feature = "mlx-native")]
            MlxFamily::Gemma4 => {
                use std::path::PathBuf;
                let dir = if std::path::Path::new(model_id).is_dir() {
                    PathBuf::from(model_id)
                } else if let Ok(d) = std::env::var("LUMEN_GEMMA4_DIR") {
                    PathBuf::from(d)
                } else {
                    return Err(anyhow!(
                        "Gemma 4 native MLX requires either MODEL_ID pointing at an \
                         existing local directory or LUMEN_GEMMA4_DIR=<dir>"
                    ));
                };
                let inner = crate::gemma4::Gemma4Backend::from_dir(model_id, &dir)?;
                Ok(Self::Gemma4(inner))
            }
            MlxFamily::Qwen35 => {
                let inner = MlxQwen35Backend::load(model_id)?;
                Ok(Self::Qwen35Family(inner))
            }
        }
    }

    /// Escape hatch: return a mutable reference to the inner Qwen3.5
    /// backend if this is one. Examples + benchmarks that drive the
    /// low-level `prefill` / `decode_step` / `snapshot_state` API directly
    /// use this; production code paths in the engine should go through the
    /// unified `chat` / `chat_streaming` / `generate` methods instead.
    pub fn as_qwen35_mut(&mut self) -> Option<&mut MlxQwen35Backend> {
        match self {
            Self::Qwen35Family(m) => Some(m),
            #[cfg(feature = "mlx-native")]
            Self::Gemma4(_) => None,
        }
    }

    /// Same as [`as_qwen35_mut`] but borrows immutably.
    pub fn as_qwen35(&self) -> Option<&MlxQwen35Backend> {
        match self {
            Self::Qwen35Family(m) => Some(m),
            #[cfg(feature = "mlx-native")]
            Self::Gemma4(_) => None,
        }
    }

    /// Mutable accessor to the inner Gemma 4 backend (Phase 3 batched scheduler).
    #[cfg(feature = "mlx-native")]
    pub fn as_gemma4_mut(&mut self) -> Option<&mut crate::gemma4::Gemma4Backend> {
        match self {
            Self::Gemma4(m) => Some(m),
            Self::Qwen35Family(_) => None,
        }
    }

    /// Immutable counterpart of [`as_gemma4_mut`].
    #[cfg(feature = "mlx-native")]
    pub fn as_gemma4(&self) -> Option<&crate::gemma4::Gemma4Backend> {
        match self {
            Self::Gemma4(m) => Some(m),
            Self::Qwen35Family(_) => None,
        }
    }

    /// Phase 3: family-agnostic `&mut dyn` driver for the batched scheduler.
    #[cfg(feature = "mlx-native")]
    pub fn batched_seq_driver_mut(&mut self) -> &mut dyn MlxBatchedSeqDriver {
        match self {
            Self::Qwen35Family(m) => m,
            Self::Gemma4(m) => m,
        }
    }

    /// Public discriminant for callers that need to branch on family
    /// (typically only the engine layer, e.g. for picking session vs
    /// prefix-cache semantics).
    pub fn kind(&self) -> MlxBackendKind {
        match self {
            Self::Qwen35Family(_) => MlxBackendKind::Qwen35Family,
            #[cfg(feature = "mlx-native")]
            Self::Gemma4(_) => MlxBackendKind::Gemma4,
        }
    }

    /// One-line summary of the effective runtime config (post env override)
    /// for startup logging. Returns empty string for backends that don't
    /// implement this yet — caller should skip the log line in that case.
    pub fn runtime_config_summary(&self) -> String {
        match self {
            #[cfg(feature = "mlx-native")]
            Self::Gemma4(m) => m.runtime_config_summary(),
            _ => String::new(),
        }
    }

    pub fn model_id(&self) -> &str {
        match self {
            Self::Qwen35Family(m) => m.model_id(),
            #[cfg(feature = "mlx-native")]
            Self::Gemma4(m) => m.model_id(),
        }
    }

    /// Effective model max context, when the backend exposes a hard ceiling.
    /// `Some(n)` means a prompt longer than `n` tokens can never be served
    /// (the KV can't hold it; on MLX an over-ctx prefill OOM-aborts the
    /// process), so the engine rejects such prompts up front. Gemma 4 reports
    /// its `LUMEN_MAX_CTX`-capped value; Qwen3.5 returns `None` (its context is
    /// not env-capped here, so only the operator prompt cap applies).
    pub fn max_context(&self) -> Option<usize> {
        match self {
            Self::Qwen35Family(_) => None,
            #[cfg(feature = "mlx-native")]
            Self::Gemma4(m) => Some(m.max_context()),
        }
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        match self {
            Self::Qwen35Family(m) => m.encode(text),
            #[cfg(feature = "mlx-native")]
            Self::Gemma4(m) => m.encode(text),
        }
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        match self {
            Self::Qwen35Family(m) => m.decode(tokens),
            #[cfg(feature = "mlx-native")]
            Self::Gemma4(m) => m.decode(tokens),
        }
    }

    pub fn build_chat_input(
        &self,
        messages: &[(String, String)],
        thinking: bool,
    ) -> Result<Vec<u32>> {
        match self {
            Self::Qwen35Family(m) => m.build_chat_input(messages, thinking),
            #[cfg(feature = "mlx-native")]
            Self::Gemma4(m) => m.build_chat_input(messages, thinking),
        }
    }

    /// Prompt tokens the attached images will add on top of the rendered text.
    ///
    /// `build_chat_input` only sees `(role, content)` strings, so an image
    /// request's real prompt is hundreds of tokens longer than what it reports
    /// — enough to matter for the context guard and for `usage.prompt_tokens`.
    /// Reads image headers only; never decodes pixels. Backends without image
    /// support contribute nothing.
    pub fn image_prompt_tokens(&self, images: &[Vec<Vec<u8>>]) -> usize {
        match self {
            #[cfg(feature = "mlx-native")]
            Self::Gemma4(m) => images
                .iter()
                .flatten()
                .map(|bytes| m.image_prompt_tokens(bytes).unwrap_or(0))
                .sum(),
            #[cfg(feature = "mlx-native")]
            Self::Qwen35Family(m) => images
                .iter()
                .flatten()
                .map(|bytes| m.image_prompt_tokens(bytes).unwrap_or(0))
                .sum(),
            #[allow(unreachable_patterns)]
            _ => 0,
        }
    }

    pub fn drop_prefix_cache(&mut self, key: &str) -> bool {
        match self {
            Self::Qwen35Family(m) => m.drop_prefix_cache(key),
            #[cfg(feature = "mlx-native")]
            Self::Gemma4(m) => m.drop_prefix_cache(key),
        }
    }

    pub fn clear_prefix_cache(&mut self) -> usize {
        match self {
            Self::Qwen35Family(m) => m.clear_prefix_cache(),
            #[cfg(feature = "mlx-native")]
            Self::Gemma4(m) => m.clear_prefix_cache(),
        }
    }

    /// Drop a Qwen3.5 session entry. No-op on Gemma 4 (different session
    /// model — Gemma 4 uses prefix caches keyed by session id rather than
    /// per-seq KV state).
    pub fn drop_session(&mut self, session_id: &str) -> bool {
        match self {
            Self::Qwen35Family(m) => m.drop_session(session_id),
            #[cfg(feature = "mlx-native")]
            Self::Gemma4(_) => false,
        }
    }

    /// Unified chat — handles family-specific call shapes internally.
    /// `top_p` is used by the Gemma 4 path's sampler; Qwen35Family ignores
    /// it (its sampling is configured server-side via REPEAT_PENALTY env
    /// and request-level temperature).
    pub fn chat(
        &mut self,
        messages: &[(String, String)],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        ov: &crate::SamplingOverrides,
        thinking: bool,
        session_id: Option<&str>,
        tools: &[crate::chat_io::ToolDef<'_>],
        tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
        response_schema: Option<&serde_json::Value>,
    ) -> Result<crate::chat_io::ParsedResponse> {
        use crate::chat_io::ParsedResponse;
        match self {
            Self::Qwen35Family(m) => {
                let _ = (top_p, temperature, ov);
                // response_format (WS-F #1) takes precedence over tools — when a
                // JSON schema is present, the whole assistant message is
                // constrained to that schema (matching Gemma4's
                // `select_grammar_state` policy: response_format > tool grammar).
                if let Some(schema) = response_schema {
                    let _ = (tools, tool_choice, session_id);
                    let seq_id = m.alloc_seq_id();
                    return m.chat_response_format(
                        messages,
                        &[],
                        max_new_tokens,
                        thinking,
                        seq_id,
                        schema,
                    );
                }
                // Phase 2: when tools are provided, route through the
                // Qwen 3.6 nested-XML tool template + parser. No-tools
                // path keeps the existing fast lane (MTP / spec /
                // prefix-cache friendly).
                if !tools.is_empty() {
                    let _ = session_id; // tool-aware path doesn't yet use prefix-cache
                    let seq_id = m.alloc_seq_id();
                    return m.chat_with_tools(
                        messages,
                        &[],
                        max_new_tokens,
                        thinking,
                        seq_id,
                        tools,
                        tool_choice,
                    );
                }
                let _ = tool_choice;
                let visible = if let Some(sid) = session_id {
                    m.chat_streaming_session(messages, max_new_tokens, thinking, sid, |_| {})?
                } else {
                    let seq_id = m.alloc_seq_id();
                    m.chat_streaming(messages, max_new_tokens, thinking, seq_id, |_| {})?
                };
                Ok(ParsedResponse {
                    visible,
                    reasoning: String::new(),
                    tool_calls: Vec::new(),
                })
            }
            #[cfg(feature = "mlx-native")]
            Self::Gemma4(m) => {
                // The batched `generate()` path below applies no grammar at
                // all, so any request that would have one runs through the
                // streaming decode with a no-op sink and returns its
                // ParsedResponse. It forgoes the prefix cache — a cold prefill
                // in exchange for the constraint the caller asked for, rather
                // than silently unconstrained output.
                //
                // This covers `response_format` and tool calls alike. Leaving
                // tools out is what let a non-streaming `tool_choice=required`
                // return `迎get_weather`: no matcher was ever built, so the
                // name was whatever the model felt like, and only the response
                // parser's fuzzy repair stood between that and the client.
                if gemma4_grammar_would_constrain(response_schema, tools, tool_choice) {
                    return m.chat_streaming(
                        messages,
                        max_new_tokens,
                        temperature,
                        top_p,
                        ov,
                        thinking,
                        tools,
                        tool_choice,
                        response_schema,
                        |_| Ok(()),
                    );
                }
                // Resolve prefix-cache key. Priority:
                //   1. Explicit `session_id` from the request (deterministic
                //      across clients that opt in).
                //   2. Auto-key from the system prompt hash (works for any
                //      OpenAI-style client that doesn't know about
                //      `session_id` but does include a stable system turn —
                //      the common case for chat UIs).
                let key = session_id
                    .map(String::from)
                    .or_else(|| auto_prefix_key(messages));
                if let Some(k) = key {
                    m.chat_with_prefix_cache(
                        messages,
                        max_new_tokens,
                        temperature,
                        top_p,
                        ov,
                        thinking,
                        &k,
                        tools,
                        tool_choice,
                    )
                } else {
                    m.chat(
                        messages,
                        max_new_tokens,
                        temperature,
                        top_p,
                        ov,
                        thinking,
                        tools,
                        tool_choice,
                    )
                }
            }
        }
    }

    /// [`Self::chat`] with inline images (`images[i]` belongs to `messages[i]`).
    ///
    /// Falls through to [`Self::chat`] when nothing is attached, so callers can
    /// route every request here. Image requests skip the prefix cache: the
    /// cached prefix is keyed on text alone, and a vision prompt's placeholder
    /// rows are only meaningful together with the image they were spliced from.
    #[allow(clippy::too_many_arguments)]
    pub fn chat_with_images(
        &mut self,
        messages: &[(String, String)],
        images: &[Vec<Vec<u8>>],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        ov: &crate::SamplingOverrides,
        thinking: bool,
        session_id: Option<&str>,
        tools: &[crate::chat_io::ToolDef<'_>],
        tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
        response_schema: Option<&serde_json::Value>,
    ) -> Result<crate::chat_io::ParsedResponse> {
        if !images.iter().any(|v| !v.is_empty()) {
            return self.chat(
                messages,
                max_new_tokens,
                temperature,
                top_p,
                ov,
                thinking,
                session_id,
                tools,
                tool_choice,
                response_schema,
            );
        }
        match self {
            // The variant itself is `#[cfg]`'d on `mlx-native`, so the arm has
            // to be too — as does everything it reaches
            // (`gemma4_grammar_would_constrain`).
            #[cfg(feature = "mlx-native")]
            Self::Gemma4(m) => {
                let _ = session_id;
                // Same trick as the text path: only the streaming decode
                // applies a grammar, so a non-streaming request that wants one
                // runs it with a no-op sink.
                if gemma4_grammar_would_constrain(response_schema, tools, tool_choice) {
                    return m.chat_streaming_with_images(
                        messages,
                        images,
                        max_new_tokens,
                        temperature,
                        top_p,
                        ov,
                        thinking,
                        tools,
                        tool_choice,
                        response_schema,
                        |_| Ok(()),
                    );
                }
                m.chat_with_images(
                    messages,
                    images,
                    max_new_tokens,
                    temperature,
                    top_p,
                    ov,
                    thinking,
                    tools,
                    tool_choice,
                )
            }
            Self::Qwen35Family(m) => {
                // The Qwen3.6 image path is greedy-only: the family's sampled
                // branches key on text.
                let _ = (temperature, top_p, ov, session_id);
                let seq_id = m.alloc_seq_id();
                // response_format wins over tools, matching the text path.
                if let Some(schema) = response_schema {
                    let _ = (tools, tool_choice);
                    return m.chat_response_format(
                        messages,
                        images,
                        max_new_tokens,
                        thinking,
                        seq_id,
                        schema,
                    );
                }
                if !tools.is_empty() {
                    return m.chat_with_tools(
                        messages,
                        images,
                        max_new_tokens,
                        thinking,
                        seq_id,
                        tools,
                        tool_choice,
                    );
                }
                let _ = tool_choice;
                // Non-streaming: drive the same loop and discard the deltas.
                let visible = m.chat_streaming_with_images(
                    messages,
                    images,
                    max_new_tokens,
                    thinking,
                    seq_id,
                    |_| {},
                )?;
                Ok(crate::chat_io::ParsedResponse {
                    visible,
                    ..Default::default()
                })
            }
            #[allow(unreachable_patterns)]
            _ => Err(anyhow::anyhow!(
                "image input requires a vision-capable MLX backend (LUMEN_VISION=1)"
            )),
        }
    }

    /// [`Self::chat_streaming`] with inline images.
    ///
    /// Same contract as [`Self::chat_with_images`]: falls through to the text
    /// path when nothing is attached, and rejects rather than silently
    /// answering without the image on a backend that cannot see it.
    #[allow(clippy::too_many_arguments)]
    pub fn chat_streaming_with_images<F>(
        &mut self,
        messages: &[(String, String)],
        images: &[Vec<Vec<u8>>],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        ov: &crate::SamplingOverrides,
        thinking: bool,
        session_id: Option<&str>,
        tools: &[crate::chat_io::ToolDef<'_>],
        tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
        response_schema: Option<&serde_json::Value>,
        on_event: F,
    ) -> Result<crate::chat_io::ParsedResponse>
    where
        F: FnMut(crate::chat_io::BackendStreamEvent<'_>) -> Result<()>,
    {
        if !images.iter().any(|v| !v.is_empty()) {
            return self.chat_streaming(
                messages,
                max_new_tokens,
                temperature,
                top_p,
                ov,
                thinking,
                session_id,
                tools,
                tool_choice,
                response_schema,
                on_event,
            );
        }
        match self {
            #[cfg(feature = "mlx-native")]
            Self::Gemma4(m) => {
                let _ = session_id;
                m.chat_streaming_with_images(
                    messages,
                    images,
                    max_new_tokens,
                    temperature,
                    top_p,
                    ov,
                    thinking,
                    tools,
                    tool_choice,
                    response_schema,
                    on_event,
                )
            }
            Self::Qwen35Family(m) => {
                let _ = (temperature, top_p, ov, session_id);
                let seq_id = m.alloc_seq_id();
                let mut on_event = on_event;
                // response_format wins over tools, matching the text path.
                if let Some(schema) = response_schema {
                    let _ = (tools, tool_choice);
                    let mut sink = |chunk: &str| {
                        let _ = on_event(crate::chat_io::BackendStreamEvent::Text(chunk));
                    };
                    return m.chat_streaming_response_format(
                        messages,
                        images,
                        max_new_tokens,
                        thinking,
                        seq_id,
                        schema,
                        &mut sink,
                    );
                }
                if !tools.is_empty() {
                    return m.chat_streaming_with_tools(
                        messages,
                        images,
                        max_new_tokens,
                        thinking,
                        seq_id,
                        tools,
                        tool_choice,
                        on_event,
                    );
                }
                let _ = tool_choice;
                let mut sink = |chunk: &str| {
                    let _ = on_event(crate::chat_io::BackendStreamEvent::Text(chunk));
                };
                let visible = m.chat_streaming_with_images(
                    messages,
                    images,
                    max_new_tokens,
                    thinking,
                    seq_id,
                    &mut sink,
                )?;
                Ok(crate::chat_io::ParsedResponse {
                    visible,
                    ..Default::default()
                })
            }
            #[allow(unreachable_patterns)]
            _ => Err(anyhow::anyhow!(
                "image input requires a vision-capable MLX backend (LUMEN_VISION=1)"
            )),
        }
    }

    /// Structured-history variant of `chat`. Used by the turn-2+ path
    /// where the request carries assistant.tool_calls / role:tool entries
    /// that the legacy `(role, content)` shape can't represent. Falls
    /// through to plain-text rendering for Qwen 3.5 (Phase 2 will wire its
    /// own structured renderer).
    /// [`Self::chat_from_history`] with images attached to `User` turns.
    ///
    /// Falls through to the text path when nothing is attached. Image requests
    /// bypass the prefix cache (keyed on text alone) and reject
    /// `response_format`, which routes through a grammar path that has no
    /// image wiring.
    #[allow(clippy::too_many_arguments)]
    pub fn chat_from_history_with_images(
        &mut self,
        turns: &[crate::chat_io::ChatTurn<'_>],
        images: &[Vec<Vec<u8>>],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        ov: &crate::SamplingOverrides,
        thinking: bool,
        session_id: Option<&str>,
        tools: &[crate::chat_io::ToolDef<'_>],
        tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
        response_schema: Option<&serde_json::Value>,
    ) -> Result<crate::chat_io::ParsedResponse> {
        if !images.iter().any(|v| !v.is_empty()) {
            return self.chat_from_history(
                turns,
                max_new_tokens,
                temperature,
                top_p,
                ov,
                thinking,
                session_id,
                tools,
                tool_choice,
                response_schema,
            );
        }
        match self {
            #[cfg(feature = "mlx-native")]
            Self::Gemma4(m) => {
                let _ = session_id;
                // Only the streaming decode applies a grammar; run it with a
                // no-op sink for the non-streaming call.
                if gemma4_grammar_would_constrain(response_schema, tools, tool_choice) {
                    return m.chat_streaming_from_history_with_images(
                        turns,
                        images,
                        max_new_tokens,
                        temperature,
                        top_p,
                        ov,
                        thinking,
                        tools,
                        tool_choice,
                        response_schema,
                        |_| Ok(()),
                    );
                }
                m.chat_from_history_with_images(
                    turns,
                    images,
                    max_new_tokens,
                    temperature,
                    top_p,
                    ov,
                    thinking,
                    tools,
                    tool_choice,
                )
            }
            Self::Qwen35Family(m) => {
                let _ = (temperature, top_p, ov, session_id);
                let seq_id = m.alloc_seq_id();
                if let Some(schema) = response_schema {
                    let _ = (tools, tool_choice);
                    return m.chat_response_format_from_history(
                        turns,
                        images,
                        max_new_tokens,
                        thinking,
                        seq_id,
                        schema,
                    );
                }
                m.chat_with_tools_from_history(
                    turns,
                    images,
                    max_new_tokens,
                    thinking,
                    seq_id,
                    tools,
                    tool_choice,
                )
            }
            #[allow(unreachable_patterns)]
            _ => Err(anyhow::anyhow!(
                "image input with a tool-calling history requires a vision-capable MLX \
                 backend (LUMEN_VISION=1)"
            )),
        }
    }

    pub fn chat_from_history(
        &mut self,
        turns: &[crate::chat_io::ChatTurn<'_>],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        ov: &crate::SamplingOverrides,
        thinking: bool,
        session_id: Option<&str>,
        tools: &[crate::chat_io::ToolDef<'_>],
        tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
        response_schema: Option<&serde_json::Value>,
    ) -> Result<crate::chat_io::ParsedResponse> {
        match self {
            Self::Qwen35Family(m) => {
                let _ = (top_p, temperature, ov, session_id);
                // response_format (WS-F #1) precedence over tools — see flat
                // `chat` path. The structured-history renderer is reused (no
                // tools) so assistant.tool_calls / role:tool turns still render.
                if let Some(schema) = response_schema {
                    let _ = (tools, tool_choice);
                    let seq_id = m.alloc_seq_id();
                    return m.chat_response_format_from_history(
                        turns,
                        &[],
                        max_new_tokens,
                        thinking,
                        seq_id,
                        schema,
                    );
                }
                // Phase 2: structured-history paths ALWAYS go through
                // the tool-aware renderer — the legacy IM-only template
                // cannot represent `<tool_call>` blocks or
                // `<tool_response>` turns. Tools may be empty (rare —
                // history-only replay without re-declaring tools); in
                // that case the renderer omits the system `<tools>`
                // block but still emits the assistant tool_calls and
                // tool_response blocks correctly.
                let seq_id = m.alloc_seq_id();
                m.chat_with_tools_from_history(
                    turns,
                    &[],
                    max_new_tokens,
                    thinking,
                    seq_id,
                    tools,
                    tool_choice,
                )
            }
            #[cfg(feature = "mlx-native")]
            Self::Gemma4(m) => {
                // Prefix-cache wiring (was Phase 1.4 TODO; landed v0.4.7).
                // Same key-resolution policy as the flat-message `chat()`
                // path above — explicit session_id > auto-hash of System
                // turn content. Falls through to the no-prefix-cache path
                // when neither yields a key (e.g. user-first chat with no
                // system turn).
                // Mirror the flat `chat` path — a request that would get a
                // grammar routes through the grammar-aware streaming decode
                // with a no-op sink, because the cache/`generate` path applies
                // none. Forgoes the prefix cache for those requests.
                if gemma4_grammar_would_constrain(response_schema, tools, tool_choice) {
                    return m.chat_streaming_from_history(
                        turns,
                        max_new_tokens,
                        temperature,
                        top_p,
                        ov,
                        thinking,
                        tools,
                        tool_choice,
                        response_schema,
                        |_| Ok(()),
                    );
                }
                let key = session_id
                    .map(String::from)
                    .or_else(|| auto_prefix_key_from_turns(turns));
                if let Some(k) = key {
                    m.chat_from_history_with_prefix_cache(
                        turns,
                        max_new_tokens,
                        temperature,
                        top_p,
                        ov,
                        thinking,
                        &k,
                        tools,
                        tool_choice,
                    )
                } else {
                    m.chat_from_history(
                        turns,
                        max_new_tokens,
                        temperature,
                        top_p,
                        ov,
                        thinking,
                        tools,
                        tool_choice,
                    )
                }
            }
        }
    }

    /// Unified streaming chat — same shape as `chat` but with an event callback.
    /// `on_event` receives `BackendStreamEvent::Text` for visible-text deltas
    /// and `BackendStreamEvent::ToolCallStart { name }` when the backend's
    /// parser identifies the start of a tool call (Gemma 4 only at Phase 1.6c;
    /// Qwen35 family emits Text only). The final `ParsedResponse` carries
    /// the full structured result including any tool_calls.
    pub fn chat_streaming<F>(
        &mut self,
        messages: &[(String, String)],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        ov: &crate::SamplingOverrides,
        thinking: bool,
        session_id: Option<&str>,
        tools: &[crate::chat_io::ToolDef<'_>],
        tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
        response_schema: Option<&serde_json::Value>,
        mut on_event: F,
    ) -> Result<crate::chat_io::ParsedResponse>
    where
        F: FnMut(crate::chat_io::BackendStreamEvent<'_>) -> Result<()>,
    {
        use crate::chat_io::{BackendStreamEvent, ParsedResponse};
        match self {
            Self::Qwen35Family(m) => {
                let _ = (top_p, temperature, ov);
                // response_format (WS-F #1) precedence over tools — the whole
                // assistant message is JSON-schema constrained and streamed as
                // `BackendStreamEvent::Text` deltas (no tool demux).
                if let Some(schema) = response_schema {
                    let _ = (tools, tool_choice, session_id);
                    let seq_id = m.alloc_seq_id();
                    let mut text_adapter = |chunk: &str| {
                        let _ = on_event(BackendStreamEvent::Text(chunk));
                    };
                    return m.chat_streaming_response_format(
                        messages,
                        &[],
                        max_new_tokens,
                        thinking,
                        seq_id,
                        schema,
                        &mut text_adapter,
                    );
                }
                // Phase 2: with tools provided, route through tool-aware
                // streaming path so `<tool_call>` blocks demux into
                // `BackendStreamEvent::ToolCallStart` + `parsed.tool_calls`.
                // No-tools path keeps the existing fast lane.
                if !tools.is_empty() {
                    let _ = session_id;
                    let seq_id = m.alloc_seq_id();
                    return m.chat_streaming_with_tools(
                        messages,
                        &[],
                        max_new_tokens,
                        thinking,
                        seq_id,
                        tools,
                        tool_choice,
                        on_event,
                    );
                }
                let _ = tool_choice;
                let mut text_adapter = |chunk: &str| {
                    let _ = on_event(BackendStreamEvent::Text(chunk));
                };
                let visible = if let Some(sid) = session_id {
                    m.chat_streaming_session(
                        messages,
                        max_new_tokens,
                        thinking,
                        sid,
                        &mut text_adapter,
                    )?
                } else {
                    let seq_id = m.alloc_seq_id();
                    m.chat_streaming(
                        messages,
                        max_new_tokens,
                        thinking,
                        seq_id,
                        &mut text_adapter,
                    )?
                };
                Ok(ParsedResponse {
                    visible,
                    reasoning: String::new(),
                    tool_calls: Vec::new(),
                })
            }
            #[cfg(feature = "mlx-native")]
            Self::Gemma4(m) => {
                // Prefix-cache wiring (landed v0.4.7) — this is the path
                // GUI / OpenAI streaming clients hit. Previously
                // `let _ = session_id;` discarded the chance to fork from
                // a shared system-prompt cache; turn-2+ requests would
                // cold-prefill the entire 5K-token chat history every
                // time. With auto-key from the system turn, the chat
                // history grows by ~70 tokens per turn and we prefill
                // only the new suffix → ~5s → ~100ms per turn.
                let key = session_id
                    .map(String::from)
                    .or_else(|| auto_prefix_key(messages));
                if let Some(k) = key {
                    m.chat_streaming_with_prefix_cache(
                        messages,
                        max_new_tokens,
                        temperature,
                        top_p,
                        ov,
                        thinking,
                        &k,
                        tools,
                        tool_choice,
                        response_schema,
                        on_event,
                    )
                } else {
                    m.chat_streaming(
                        messages,
                        max_new_tokens,
                        temperature,
                        top_p,
                        ov,
                        thinking,
                        tools,
                        tool_choice,
                        response_schema,
                        on_event,
                    )
                }
            }
        }
    }

    /// Phase 1.5: structured-history streaming entry point. Routes
    /// Gemma 4 through `chat_streaming_from_history` so turn-2 agent
    /// loops can stream natural-language continuations. Qwen 3.5
    /// family currently flattens to plain text (loses tool metadata);
    /// future Hermes-style structured streaming lands in Phase 2.
    /// [`Self::chat_streaming_from_history`] with images on `User` turns.
    #[allow(clippy::too_many_arguments)]
    pub fn chat_streaming_from_history_with_images<F>(
        &mut self,
        turns: &[crate::chat_io::ChatTurn<'_>],
        images: &[Vec<Vec<u8>>],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        ov: &crate::SamplingOverrides,
        thinking: bool,
        session_id: Option<&str>,
        tools: &[crate::chat_io::ToolDef<'_>],
        tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
        response_schema: Option<&serde_json::Value>,
        on_event: F,
    ) -> Result<crate::chat_io::ParsedResponse>
    where
        F: FnMut(crate::chat_io::BackendStreamEvent<'_>) -> Result<()>,
    {
        if !images.iter().any(|v| !v.is_empty()) {
            return self.chat_streaming_from_history(
                turns,
                max_new_tokens,
                temperature,
                top_p,
                ov,
                thinking,
                session_id,
                tools,
                tool_choice,
                response_schema,
                on_event,
            );
        }
        match self {
            #[cfg(feature = "mlx-native")]
            Self::Gemma4(m) => {
                let _ = session_id;
                m.chat_streaming_from_history_with_images(
                    turns,
                    images,
                    max_new_tokens,
                    temperature,
                    top_p,
                    ov,
                    thinking,
                    tools,
                    tool_choice,
                    response_schema,
                    on_event,
                )
            }
            Self::Qwen35Family(m) => {
                let _ = (temperature, top_p, ov, session_id);
                let seq_id = m.alloc_seq_id();
                let mut on_event = on_event;
                if let Some(schema) = response_schema {
                    let _ = (tools, tool_choice);
                    let mut sink = |chunk: &str| {
                        let _ = on_event(crate::chat_io::BackendStreamEvent::Text(chunk));
                    };
                    return m.chat_streaming_response_format_from_history(
                        turns,
                        images,
                        max_new_tokens,
                        thinking,
                        seq_id,
                        schema,
                        &mut sink,
                    );
                }
                m.chat_streaming_with_tools_from_history(
                    turns,
                    images,
                    max_new_tokens,
                    thinking,
                    seq_id,
                    tools,
                    tool_choice,
                    on_event,
                )
            }
            #[allow(unreachable_patterns)]
            _ => Err(anyhow::anyhow!(
                "image input with a tool-calling history requires a vision-capable MLX \
                 backend (LUMEN_VISION=1)"
            )),
        }
    }

    pub fn chat_streaming_from_history<F>(
        &mut self,
        turns: &[crate::chat_io::ChatTurn<'_>],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        ov: &crate::SamplingOverrides,
        thinking: bool,
        session_id: Option<&str>,
        tools: &[crate::chat_io::ToolDef<'_>],
        tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
        response_schema: Option<&serde_json::Value>,
        on_event: F,
    ) -> Result<crate::chat_io::ParsedResponse>
    where
        F: FnMut(crate::chat_io::BackendStreamEvent<'_>) -> Result<()>,
    {
        use crate::chat_io::ParsedResponse;
        match self {
            Self::Qwen35Family(m) => {
                let _ = (top_p, temperature, ov, session_id);
                // response_format (WS-F #1) precedence over tools — streamed as
                // `BackendStreamEvent::Text` deltas; structured-history renderer
                // reused with no tools.
                if let Some(schema) = response_schema {
                    let _ = (tools, tool_choice);
                    let seq_id = m.alloc_seq_id();
                    let mut on_event = on_event;
                    let mut text_adapter = |chunk: &str| {
                        let _ = on_event(crate::chat_io::BackendStreamEvent::Text(chunk));
                    };
                    return m.chat_streaming_response_format_from_history(
                        turns,
                        &[],
                        max_new_tokens,
                        thinking,
                        seq_id,
                        schema,
                        &mut text_adapter,
                    );
                }
                // Phase 2: structured-history streaming ALWAYS routes
                // through the tool-aware path — same rationale as the
                // non-streaming `chat_from_history` branch above.
                let seq_id = m.alloc_seq_id();
                m.chat_streaming_with_tools_from_history(
                    turns,
                    &[],
                    max_new_tokens,
                    thinking,
                    seq_id,
                    tools,
                    tool_choice,
                    on_event,
                )
            }
            #[cfg(feature = "mlx-native")]
            Self::Gemma4(m) => {
                // Prefix-cache wiring (landed v0.4.7). Same key resolution
                // as the flat-message streaming path; the structured-turns
                // shape is used by turn-2+ tool-call loops where the saved
                // common prefix is even larger (system + tool defs +
                // multiple prior assistant/tool exchanges).
                let key = session_id
                    .map(String::from)
                    .or_else(|| auto_prefix_key_from_turns(turns));
                if let Some(k) = key {
                    m.chat_streaming_from_history_with_prefix_cache(
                        turns,
                        max_new_tokens,
                        temperature,
                        top_p,
                        ov,
                        thinking,
                        &k,
                        tools,
                        tool_choice,
                        response_schema,
                        on_event,
                    )
                } else {
                    m.chat_streaming_from_history(
                        turns,
                        max_new_tokens,
                        temperature,
                        top_p,
                        ov,
                        thinking,
                        tools,
                        tool_choice,
                        response_schema,
                        on_event,
                    )
                }
            }
        }
        .map(|p: ParsedResponse| p)
    }

    /// Unified completion (`/v1/completions`) — greedy raw token generation.
    pub fn generate(
        &mut self,
        input_ids: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        ov: &crate::SamplingOverrides,
        session_id: Option<&str>,
    ) -> Result<Vec<u32>> {
        match self {
            Self::Qwen35Family(m) => {
                let _ = (temperature, top_p, ov);
                if let Some(sid) = session_id {
                    return m.completion_session(input_ids, max_new_tokens, sid);
                }
                let seq_id = m.alloc_seq_id();
                let (mut last, mut pos) = m.prefill(seq_id, input_ids)?;
                let mut out: Vec<u32> = vec![last];
                let eos = m.eos_tokens().to_vec();
                if !eos.contains(&last) {
                    for _ in 1..max_new_tokens {
                        let (n, p) = m.decode_step(seq_id, last, pos)?;
                        last = n;
                        pos = p;
                        out.push(n);
                        if eos.contains(&n) {
                            break;
                        }
                    }
                }
                m.remove_seq(seq_id).ok();
                Ok(out)
            }
            #[cfg(feature = "mlx-native")]
            Self::Gemma4(m) => {
                let _ = session_id;
                m.generate(input_ids, max_new_tokens, temperature, top_p, ov)
            }
        }
    }
}

/// Family discriminant — exposed for engine-layer logging / metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlxBackendKind {
    Qwen35Family,
    #[cfg(feature = "mlx-native")]
    Gemma4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MlxFamily {
    Qwen35,
    #[cfg(feature = "mlx-native")]
    Gemma4,
}

fn detect_mlx_family(model_id: &str) -> MlxFamily {
    // Manual override takes priority — useful when the heuristics misfire
    // for a custom repo name.
    if let Ok(v) = std::env::var("LUMEN_MLX_FAMILY") {
        match v.trim().to_lowercase().as_str() {
            #[cfg(feature = "mlx-native")]
            "gemma4" | "gemma" | "gemma-4" => return MlxFamily::Gemma4,
            "qwen" | "qwen35" | "qwen3_5" | "qwen3.6" => return MlxFamily::Qwen35,
            _ => {}
        }
    }
    let lower = model_id.to_lowercase();
    #[cfg(feature = "mlx-native")]
    if lower.contains("gemma-4")
        || lower.contains("gemma4-")
        || lower.contains("gemma_4")
        || lower.contains("gemma4_")
    {
        return MlxFamily::Gemma4;
    }
    MlxFamily::Qwen35
}

pub struct MlxQwen35Backend {
    runner: RunnerImpl,
    pub model_id: String,
    pub eos_tokens: Vec<u32>,
    pub vocab_size: usize,
    tokenizer: Option<Tokenizer>,
    next_seq_id: AtomicU64,
    sessions: std::collections::HashMap<String, SessionState>,
    session_ttl: Option<Duration>,
    session_max: Option<usize>,
    prefix_store: prefix_cache::PrefixCacheStore,
    /// Lazily-built llguidance parser factory for the Qwen 3.6 tool-call
    /// grammar (WS-C #2), backed by this model's `tokenizer.json`. `None`
    /// inside the `Option` means construction failed (no tokenizer.json, or
    /// the vocab table was empty) — such requests degrade gracefully to
    /// unconstrained sampling rather than erroring. Built once on first
    /// grammar-eligible request; the `Arc` is cloned per request.
    grammar_factory: OnceLock<Option<std::sync::Arc<llguidance::ParserFactory>>>,
    /// OPT-IN draft-model speculative decode (default OFF). `Some` only when
    /// `LUMEN_MLX_DRAFT_MODEL` is set AND the draft loaded with a vocab size
    /// matching the target. `None` → the entire draft path is dormant and the
    /// baseline decode runs byte-identically. Held as its own isolated
    /// `NativeMlxRunner` (second model + its own seq/KV state).
    #[cfg(feature = "mlx-native")]
    draft: Option<DraftRunner>,
}

/// Isolated draft model + its resolved config for greedy spec decode.
#[cfg(feature = "mlx-native")]
struct DraftRunner {
    runner: NativeMlxRunner,
    cfg: spec_decode::DraftConfig,
    eos_tokens: Vec<u32>,
}

/// Messages rewritten to carry their images, plus everything prefill needs to
/// pair each placeholder run with the right image. See
/// [`MlxQwen35Backend::attach_image_blocks`].
#[cfg(feature = "mlx-native")]
struct AttachedImages {
    /// `(role, content)` with an `<|image_pad|>` block prepended to the body of
    /// every message that had an image attached.
    messages: Vec<(String, String)>,
    /// Decoded and resized images, in prompt order.
    prepared: Vec<crate::qwen36_vision::PreparedImage>,
    /// `counts[k]` is how many placeholder rows `prepared[k]` occupies.
    counts: Vec<usize>,
}

impl MlxQwen35Backend {
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

        let prefix_store = prefix_cache::PrefixCacheStore::from_env();
        if prefix_store.enabled() {
            eprintln!("[mlx] prefix cache ENABLED (auto-keyed by system message hash)");
            if let Some(ttl) = prefix_store.ttl() {
                eprintln!("[mlx] prefix cache TTL = {}s", ttl.as_secs());
            }
            if let Some(max) = prefix_store.max() {
                eprintln!("[mlx] prefix cache LRU cap = {max}");
            }
        }

        let mut me = Self {
            runner,
            model_id: model_id.to_string(),
            eos_tokens,
            vocab_size,
            tokenizer,
            next_seq_id: AtomicU64::new(1),
            sessions: std::collections::HashMap::new(),
            session_ttl,
            session_max,
            prefix_store,
            grammar_factory: OnceLock::new(),
            #[cfg(feature = "mlx-native")]
            draft: None,
        };
        // Phase 2 S4 — opt-in MTP auto-enable. Honored only when running on
        // the native runner with `LUMEN_QWEN35_MTP=1`. Loader failure is
        // non-fatal: we log + continue with the baseline decode path.
        #[cfg(feature = "mlx-native")]
        if let Err(err) = me.try_enable_qwen35_mtp_from_env() {
            eprintln!("[mlx] qwen3.5 MTP auto-enable skipped: {err}");
        }
        // OPT-IN draft-model speculative decode auto-load. Honored only on the
        // native runner with `LUMEN_MLX_DRAFT_MODEL` set. Any failure (load /
        // vocab-mismatch) is non-fatal: we log + leave `draft = None` so the
        // baseline decode path runs unchanged.
        #[cfg(feature = "mlx-native")]
        {
            let target_vocab = me.vocab_size;
            me.try_load_draft_model_from_env(target_vocab);
        }
        Ok(me)
    }

    pub fn prefill(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)> {
        self.runner.prefill(seq_id, tokens)
    }

    /// Prompt tokens one attached image adds — its merged-token run plus the
    /// vision sentinels. Header-only; no pixel decode.
    #[cfg(feature = "mlx-native")]
    pub fn image_prompt_tokens(&self, encoded: &[u8]) -> Result<usize> {
        let cfg = self
            .runner
            .vision_config()
            .ok_or_else(|| anyhow!("vision tower not loaded"))?;
        crate::qwen36_vision::image_prompt_tokens(encoded, &cfg)
            .map_err(|e| anyhow!("image sizing failed: {e}"))
    }

    /// [`Self::prefill`] with encoded images spliced over their placeholder
    /// runs. An empty slice is byte-identical to [`Self::prefill`].
    #[cfg(feature = "mlx-native")]
    pub fn prefill_with_images(
        &mut self,
        seq_id: u64,
        tokens: &[u32],
        images: &[crate::qwen36_vision::PreparedImage],
    ) -> Result<(u32, usize)> {
        if images.is_empty() {
            return self.runner.prefill(seq_id, tokens);
        }
        self.runner.prefill_with_images(seq_id, tokens, images)
    }

    pub fn decode_step(
        &mut self,
        seq_id: u64,
        last_token: u32,
        position: usize,
    ) -> Result<(u32, usize)> {
        self.runner.decode_step(seq_id, last_token, position)
    }

    /// Phase 1: advance N already-prefilled sequences by one token each in a
    /// single call, returning `(next_token, position)` per input index. Used by
    /// the multi-sequence batched scheduler (Phase 2). Currently a bit-identical
    /// loop over per-seq decode; swaps to a single batched forward in Phase 1b.
    pub fn decode_step_batch(
        &mut self,
        seq_ids: &[u64],
        last_tokens: &[u32],
        positions: &[usize],
    ) -> Result<Vec<(u32, usize)>> {
        self.runner
            .decode_step_batch(seq_ids, last_tokens, positions)
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

    /// Install a loaded Qwen3.5/3.6 MTP drafter block. Native-only (errors on
    /// PyO3 / Subprocess runners). After enable, callers may use
    /// [`Self::qwen35_mtp_step`] to advance one speculative cycle. Phase 2
    /// S3 surface — gating + opt-in env lives in the upper engine.
    #[cfg(feature = "mlx-native")]
    pub fn enable_qwen35_mtp(&mut self, block: crate::qwen3_5_mtp::Qwen35MtpBlock) -> Result<()> {
        self.runner.enable_qwen35_mtp(block)
    }

    /// True once an MTP drafter has been installed on this backend.
    #[cfg(feature = "mlx-native")]
    pub fn qwen35_mtp_enabled(&self) -> bool {
        self.runner.qwen35_mtp_enabled()
    }

    /// Begin MTP drift-corrector calibration: subsequent `qwen35_mtp_step`
    /// calls accumulate (recursive draft hidden, trunk true hidden) pairs up to
    /// `depth_count` (= max draft K used during calibration).
    #[cfg(feature = "mlx-native")]
    pub fn qwen35_enable_mtp_calib(&self, depth_count: usize) -> Result<()> {
        self.runner.enable_mtp_calib(depth_count)
    }

    /// Finish calibration: fit the corrector and return its serialized bytes
    /// (write to a file, then load via `qwen35_load_mtp_corrector`).
    #[cfg(feature = "mlx-native")]
    pub fn qwen35_finalize_mtp_calib(&self, eps: f32) -> Result<Vec<u8>> {
        self.runner.finalize_mtp_calib(eps)
    }

    /// Install a fitted corrector (serialized bytes from
    /// `qwen35_finalize_mtp_calib`). Applied in the drafter loop thereafter.
    #[cfg(feature = "mlx-native")]
    pub fn qwen35_load_mtp_corrector(&self, bytes: &[u8]) -> Result<()> {
        self.runner.set_mtp_corrector_bytes(bytes)
    }

    /// Start orthogonal-Procrustes calibration (the rotation shot). Heavy:
    /// allocates `depth_count · hidden²` f64.
    #[cfg(feature = "mlx-native")]
    pub fn qwen35_enable_mtp_proc_calib(&self, depth_count: usize) -> Result<()> {
        self.runner.enable_mtp_proc_calib(depth_count)
    }

    /// Finish Procrustes calibration: prints the decisive residual report
    /// (rel_raw vs rel_proc per depth) and returns the fitted corrector bytes.
    #[cfg(feature = "mlx-native")]
    pub fn qwen35_finalize_mtp_proc_calib(&self) -> Result<Vec<u8>> {
        self.runner.finalize_mtp_proc_calib()
    }

    /// Install a fitted Procrustes corrector (serialized bytes).
    #[cfg(feature = "mlx-native")]
    pub fn qwen35_load_mtp_procrustes(&self, bytes: &[u8]) -> Result<()> {
        self.runner.set_mtp_procrustes_bytes(bytes)
    }

    /// Begin C4 calibration (collect lm_head-input / trunk-argmax pairs).
    #[cfg(feature = "mlx-native")]
    pub fn qwen35_enable_mtp_c4_calib(&self) -> Result<()> {
        self.runner.enable_mtp_c4_calib()
    }

    /// Train + install the C4 lm_head LoRA from collected calib pairs.
    #[cfg(feature = "mlx-native")]
    pub fn qwen35_train_mtp_c4_lmhead(&self, rank: usize, steps: usize, lr: f32) -> Result<()> {
        self.runner.train_mtp_c4_lmhead(rank, steps, lr)
    }

    /// Begin head-internal PoC calibration (collect k=0 embed/h_pre/target).
    #[cfg(feature = "mlx-native")]
    pub fn qwen35_enable_mtp_hi_calib(&self) -> Result<()> {
        self.runner.enable_mtp_hi_calib()
    }

    /// Train the head-internal (eh_proj) LoRA on collected calib; returns
    /// `(in_sample_accept_before, in_sample_accept_after)` — the GO/fold gate.
    #[cfg(feature = "mlx-native")]
    pub fn qwen35_train_mtp_headinternal(
        &self,
        cfg: crate::qwen3_5_mtp::HiTrainCfg,
    ) -> Result<(f32, f32)> {
        self.runner.train_mtp_headinternal(cfg)
    }

    /// Advance one MTP speculative cycle. See `qwen3_5_moe::mtp_step` for
    /// the Step A-E contract. Returns the committed token list (length
    /// `1 + n_accepted + 1`); the LAST element must be fed as
    /// `committed_token` to the next call.
    #[cfg(feature = "mlx-native")]
    pub fn qwen35_mtp_step(
        &mut self,
        seq_id: u64,
        committed_token: u32,
        n_draft: usize,
        temperature: f32,
        top_p: f32,
    ) -> Result<crate::qwen3_5_moe::MtpStepOutput> {
        self.runner
            .qwen35_mtp_step(seq_id, committed_token, n_draft, temperature, top_p)
    }

    /// Env-driven MTP auto-enable hook called at the end of `load()`.
    /// Triggered by `LUMEN_QWEN35_MTP=1`. Resolves the HF-original snapshot
    /// directory holding the bf16 `mtp.*` tensors from
    /// `LUMEN_QWEN35_HF_ORIGINAL` (required when MTP is enabled), detects the
    /// trunk variant (27B Dense vs 35B-A3B MoE) from the model_id substring,
    /// loads the block via `load_block_from_hf` (AFFINE4 group_size=64), and
    /// installs it through `enable_qwen35_mtp`. Returns `Err` only when the
    /// user-facing config is broken (env present but path missing / invalid
    /// dims / dtype) — propagated by the caller as a non-fatal warning so
    /// the server boots even if MTP weights are absent.
    #[cfg(feature = "mlx-native")]
    fn try_enable_qwen35_mtp_from_env(&mut self) -> Result<()> {
        // `LUMEN_QWEN35_MTP`: "0"/"false" hard-disables; "1"/"true" forces on
        // (errors loudly if prerequisites missing); UNSET = AUTO — enable when
        // the loaded model ships a self-contained MTP head and we're on the
        // native runner. This makes self-contained MTPLX repos "just work"
        // (17.5 tok/s) without any env vars; the escape hatch is MTP=0.
        let env_val = std::env::var("LUMEN_QWEN35_MTP").ok();
        let explicit_off = env_val
            .as_deref()
            .map(|v| v == "0" || v.eq_ignore_ascii_case("false"))
            .unwrap_or(false);
        if explicit_off {
            return Ok(());
        }
        let explicit_on = env_val
            .as_deref()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        // MTP runs only on the native runner. Auto mode skips silently on other
        // runners; explicit on fails loudly so misconfig is visible.
        if !matches!(self.runner, RunnerImpl::Native(_)) {
            if explicit_on {
                return Err(anyhow!(
                    "LUMEN_QWEN35_MTP=1 requires the native runner (set LUMEN_MLX_BACKEND=native)"
                ));
            }
            return Ok(());
        }

        // Resolve the dir holding the MTP head.
        // * Explicit `LUMEN_QWEN35_HF_ORIGINAL` is honored ONLY under explicit
        //   opt-in (`MTP=1`). A bare HF_ORIGINAL without opt-in (e.g. a bench
        //   harness that manages MTP itself) must NOT auto-trigger enable.
        // * Otherwise (auto mode) default to the loaded model dir and require a
        //   self-contained `mtp/weights.safetensors` head.
        let hf_path: std::path::PathBuf = match std::env::var("LUMEN_QWEN35_HF_ORIGINAL") {
            Ok(d) => {
                if !explicit_on {
                    return Ok(());
                }
                let p = std::path::PathBuf::from(&d);
                if !p.is_dir() {
                    return Err(anyhow!("LUMEN_QWEN35_HF_ORIGINAL `{d}` is not a directory"));
                }
                p
            }
            Err(_) => match crate::runner_native::resolve_model_dir(&self.model_id) {
                Ok(dir) if model_has_mtp_head(&dir) => dir,
                _ => {
                    if explicit_on {
                        return Err(anyhow!(
                            "LUMEN_QWEN35_MTP=1 requires LUMEN_QWEN35_HF_ORIGINAL to point at a \
                             snapshot holding `mtp.*` tensors, or a model dir carrying \
                             `mtp/weights.safetensors`"
                        ));
                    }
                    // Auto mode: model has no self-contained MTP head → run the
                    // baseline decode path unchanged.
                    return Ok(());
                }
            },
        };
        // Dim detection from model_id — same substring rules as the server's
        // `is_qwen3_5_dense`. 27B = Dense; everything else under the Qwen3.6
        // family with mtp.* = MoE (35B-A3B currently).
        let lower = self.model_id.to_lowercase();
        let is_27b_dense = (lower.contains("qwen3.6") || lower.contains("qwen3_5"))
            && !lower.contains("a3b")
            && !lower.contains("moe")
            && (lower.contains("27b") || lower.contains("-27-") || lower.contains("-dense"));
        let (dims, mlp_cfg, label) = if is_27b_dense {
            (
                crate::qwen3_5_mtp::Qwen35MtpDims {
                    hidden_size: 5120,
                    num_attention_heads: 24,
                    num_key_value_heads: 4,
                    head_dim: 256,
                    rope_theta: 10_000_000.0,
                    rope_dim: 64,
                    rms_norm_eps: 1e-6,
                    attn_output_gate: true,
                },
                crate::qwen3_5_mtp::MtpMlpConfig::Dense {
                    intermediate_size: 17_408,
                },
                "Qwen3.6-27B (Dense)",
            )
        } else {
            // Default to the 35B-A3B MoE shape — matches the published
            // checkpoint and the bench_qwen35_mtp_loader_smoke.rs 35b branch.
            (
                crate::qwen3_5_mtp::Qwen35MtpDims {
                    hidden_size: 2048,
                    num_attention_heads: 16,
                    num_key_value_heads: 2,
                    head_dim: 256,
                    rope_theta: 10_000_000.0,
                    rope_dim: 64,
                    rms_norm_eps: 1e-6,
                    attn_output_gate: true,
                },
                crate::qwen3_5_mtp::MtpMlpConfig::Moe(crate::qwen3_5_mtp::MtpMoeConfig {
                    num_experts: 256,
                    num_experts_per_tok: 8,
                    moe_intermediate_size: 512,
                    shared_expert_intermediate_size: 512,
                    norm_topk_prob: true,
                }),
                "Qwen3.6-35B-A3B (MoE)",
            )
        };
        let quant = crate::qwen3_5_mtp::MtpLoadQuant::Affine4 { group_size: 64 };
        let t0 = std::time::Instant::now();
        let block = crate::qwen3_5_mtp::load_block_from_hf(&hf_path, dims, mlp_cfg, quant)
            .with_context(|| format!("load_block_from_hf({})", hf_path.display()))?;
        self.enable_qwen35_mtp(block)?;
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!("[mlx] qwen3.5 MTP ENABLED ({label}, AFFINE4 gs=64) in {dt:.0}ms");
        Ok(())
    }

    /// OPT-IN draft-model speculative decode auto-load. No-op unless
    /// `LUMEN_MLX_DRAFT_MODEL` is set. Loads a *separate* `NativeMlxRunner`
    /// holding the draft model + its own seq/KV state, then verifies the draft
    /// tokenizer vocab matches the target. On any failure or mismatch we log
    /// and leave `self.draft = None` (baseline decode runs unchanged — never
    /// crash). Honored only when the target itself runs on the native runner;
    /// the draft loop reuses native-only primitives (`forward_probe`,
    /// `snapshot_state`, `restore_state`).
    ///
    /// `target_vocab` is the already-loaded target's vocab size.
    #[cfg(feature = "mlx-native")]
    fn try_load_draft_model_from_env(&mut self, target_vocab: usize) {
        let Some(cfg) = spec_decode::read_draft_config() else {
            return; // LUMEN_MLX_DRAFT_MODEL unset → feature OFF.
        };
        if !matches!(self.runner, RunnerImpl::Native(_)) {
            eprintln!(
                "[mlx-draft] LUMEN_MLX_DRAFT_MODEL set but target runner is not native — \
                 draft spec-decode DISABLED (set LUMEN_MLX_BACKEND=native to use it)"
            );
            return;
        }
        let t0 = std::time::Instant::now();
        let mut runner = match NativeMlxRunner::new() {
            Ok(r) => r,
            Err(err) => {
                eprintln!("[mlx-draft] draft runner init failed: {err} — draft DISABLED");
                return;
            }
        };
        let info = match runner.load(&cfg.model) {
            Ok(info) => info,
            Err(err) => {
                eprintln!(
                    "[mlx-draft] draft model `{}` load failed: {err} — draft DISABLED",
                    cfg.model
                );
                return;
            }
        };
        // VERIFY vocab compatibility: argmax-only verify compares token ids
        // across the two models, so their vocabularies MUST line up.
        if info.vocab_size != target_vocab {
            eprintln!(
                "[mlx-draft] draft vocab {} != target vocab {} — draft DISABLED \
                 (incompatible tokenizers)",
                info.vocab_size, target_vocab
            );
            return;
        }
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[mlx-draft] draft model `{}` loaded (vocab={} n_max={} p_min={:.3}) in {dt:.0}ms — \
             greedy spec-decode ENABLED",
            cfg.model, info.vocab_size, cfg.n_max, cfg.p_min
        );
        self.draft = Some(DraftRunner {
            runner,
            cfg,
            eos_tokens: info.eos_tokens,
        });
    }

    /// True when the OPT-IN draft-model spec path is loaded and usable.
    #[cfg(feature = "mlx-native")]
    fn draft_enabled(&self) -> bool {
        self.draft.is_some()
    }

    /// OPT-IN greedy draft-model speculative decode (mirrors the llama.cpp
    /// greedy draft-model spec). ARGMAX-ONLY verify — engaged only for greedy
    /// requests (the Qwen3.5 family streaming path is greedy by construction;
    /// the dispatcher discards `temperature`/`top_p` before reaching here, so
    /// every call on this path is greedy). If a sampled request ever reaches
    /// this function it would still be greedy here, but the wiring gates on the
    /// greedy precondition explicitly.
    ///
    /// Per step:
    ///   PROPOSE: greedy-roll the draft `n_max` tokens via `draft.decode_step`
    ///            (autoregressive). `p_min` gating is a documented no-op for
    ///            now (see `DraftConfig::p_min`).
    ///   VERIFY:  `target.forward_probe(seq, &draft_tokens)` → K argmax rows in
    ///            ONE target forward.
    ///   ACCEPT:  longest prefix where `probe.row_argmaxes[i] == draft[i]`.
    ///            On full match, append the bonus `row_argmaxes[n]`. On partial,
    ///            `restore_state` the target to before the verify forward and
    ///            commit only the accepted prefix + the corrective
    ///            `row_argmaxes[n_acc]`. The draft KV is kept in lockstep by
    ///            re-anchoring it to the accepted tokens via `extend`.
    #[cfg(feature = "mlx-native")]
    fn chat_streaming_spec_draft<F>(
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
        let (n_max, draft_eos) = {
            let d = self
                .draft
                .as_ref()
                .ok_or_else(|| anyhow!("chat_streaming_spec_draft: draft not loaded"))?;
            (d.cfg.n_max, d.eos_tokens.clone())
        };

        let prompt_ids = self.build_chat_input(messages, thinking)?;
        if prompt_ids.is_empty() {
            return Err(anyhow!("empty prompt after tokenization"));
        }

        // Prefill BOTH models with the same prompt. Use a dedicated seq id for
        // the draft so its KV state is fully isolated from the target's.
        let draft_seq_id: u64 = seq_id;
        let t_prefill = std::time::Instant::now();
        let (mut t_pred, mut pos) = self.prefill(seq_id, &prompt_ids)?;
        // Draft prefill — its predicted next token is ignored; the draft is
        // re-driven from the committed tokens each step.
        {
            let d = self.draft.as_mut().expect("draft present");
            d.runner.prefill(draft_seq_id, &prompt_ids)?;
        }
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[mlx-draft] seq {seq_id} prefill: {} tokens in {prefill_ms:.0}ms (n_max={n_max}) -> tok={t_pred}",
            prompt_ids.len(),
        );

        let mut generated: Vec<u32> = Vec::new();
        let mut prev_text = String::new();
        let mut n_attempts: u64 = 0;
        let mut n_accept_tokens: u64 = 0;
        let mut n_verify_tokens: u64 = 0;

        let t_decode = std::time::Instant::now();
        // `step` counts emitted tokens against `max_new_tokens`. Helper macro
        // emits a committed token (decode + on_token) and checks EOS / budget.
        // Returns true when the loop should terminate.
        let mut step: usize = 0;
        let mut should_stop = false;

        // Commit-and-emit closure-free helper kept inline to satisfy the
        // borrow checker (mutably borrows self for `decode`).
        macro_rules! emit_committed {
            ($tok:expr) => {{
                let tok = $tok;
                generated.push(tok);
                if let Ok(text) = self.decode(&generated) {
                    if text.len() > prev_text.len() && !text.contains('\u{FFFD}') {
                        on_token(&text[prev_text.len()..]);
                        prev_text = text;
                    }
                }
                step += 1;
                if self.eos_tokens.contains(&tok) || step >= max_new_tokens {
                    should_stop = true;
                }
            }};
        }

        while step < max_new_tokens && !should_stop {
            // ── PROPOSE: roll the draft n_max tokens. The draft must FIRST
            // consume the target's committed token (`t_pred`) so its KV is at
            // the same logical offset as the target before it predicts. We feed
            // `t_pred` then take the draft's own argmax chain.
            n_attempts += 1;
            let mut draft_tokens: Vec<u32> = Vec::with_capacity(n_max);
            {
                let d = self.draft.as_mut().expect("draft present");
                // Step the draft on the just-committed target token to align
                // its KV, getting the draft's first proposal d_0.
                let (mut d_next, _dpos) = d.runner.decode_step(draft_seq_id, t_pred, 0)?;
                draft_tokens.push(d_next);
                // Continue greedily for the remaining n_max-1 tokens.
                for _ in 1..n_max {
                    // p_min gating (LUMEN_MLX_DRAFT_PMIN): decode_step returns
                    // argmax-only (no probability), so probability-based early
                    // stop is a TODO(hardware-verify). With the default
                    // p_min=0.0 the full n_max chain is proposed.
                    if draft_eos.contains(&d_next) {
                        break; // draft hit its own EOS — stop proposing.
                    }
                    let (n, _p) = d.runner.decode_step(draft_seq_id, d_next, 0)?;
                    d_next = n;
                    draft_tokens.push(d_next);
                }
            }

            if draft_tokens.is_empty() {
                // Degenerate: no proposal. Fall back to a single target step.
                emit_committed!(t_pred);
                if should_stop {
                    break;
                }
                let (next, new_pos) = self.decode_step(seq_id, t_pred, pos)?;
                t_pred = next;
                pos = new_pos;
                continue;
            }

            // ── VERIFY: snapshot the target, then run ONE batched forward over
            // the draft tokens. row_argmaxes[i] = the target's greedy next
            // token *conditioned on the cache + draft[0..=i]*.
            let snap = self.snapshot_state(seq_id)?;
            let probe = self.forward_probe(seq_id, &draft_tokens)?;
            n_verify_tokens += draft_tokens.len() as u64;
            let rows = &probe.row_argmaxes;
            if rows.len() != draft_tokens.len() {
                self.restore_state(seq_id, snap).ok();
                return Err(anyhow!(
                    "draft verify returned {} rows for {} draft tokens",
                    rows.len(),
                    draft_tokens.len()
                ));
            }

            // ── ACCEPT: the target's true next token at offset M is `t_pred`
            // (known from the prior step). Accept draft[i] iff it equals the
            // target's prediction at that position. Position 0's target pred is
            // `t_pred`; position i>0's is rows[i-1].
            let mut n_acc = 0usize;
            while n_acc < draft_tokens.len() {
                let target_pred = if n_acc == 0 { t_pred } else { rows[n_acc - 1] };
                if draft_tokens[n_acc] == target_pred {
                    n_acc += 1;
                } else {
                    break;
                }
            }

            // Corrective token = the target's prediction at the first rejected
            // position (= t_pred when n_acc==0, else rows[n_acc-1]). On full
            // accept this is rows[len-1] = the bonus token.
            let corrective = if n_acc == 0 { t_pred } else { rows[n_acc - 1] };
            n_accept_tokens += n_acc as u64;

            // Roll the TARGET cache to exactly `n_acc` accepted draft tokens.
            // forward_probe advanced the cache by the full draft length; restore
            // then re-feed only the accepted prefix.
            pos = self.restore_state(seq_id, snap)?;
            if n_acc > 0 {
                // extend re-consumes the accepted draft tokens into the cache;
                // its returned next-token prediction is the target's pred after
                // the accepted prefix == `corrective` (rows[n_acc-1]) and we
                // commit that as the corrective.
                let (_n, new_pos) = self.runner.extend(seq_id, &draft_tokens[..n_acc])?;
                pos = new_pos;
            }

            // Commit accepted draft prefix + the corrective token.
            for i in 0..n_acc {
                emit_committed!(draft_tokens[i]);
                if should_stop {
                    break;
                }
            }
            if should_stop {
                break;
            }
            emit_committed!(corrective);
            if should_stop {
                break;
            }

            // Re-anchor the DRAFT KV to the committed tokens. The draft already
            // consumed `t_pred` + its own proposed chain during PROPOSE; that
            // chain diverged past `n_acc`. Simplest correct fix: roll the draft
            // forward over the corrective token so both models share the same
            // logical suffix for the next round. Because the draft consumed the
            // *full* proposed chain (not just the accepted prefix), we re-prime
            // it from the corrective token; the next PROPOSE will feed it the
            // next committed `t_pred`. Mild KV drift on the draft is acceptable
            // (the draft is only a speculator; the target verify is the source
            // of truth). TODO(hardware-verify): tighten draft KV rollback to
            // exactly the accepted prefix if accept-rate suffers.
            {
                let d = self.draft.as_mut().expect("draft present");
                let _ = d.runner.decode_step(draft_seq_id, corrective, 0)?;
            }

            // Advance the target to get the next `t_pred` after the corrective.
            // The target cache is currently at M + n_acc (extend) and has NOT
            // consumed `corrective`. Feed it via decode_step.
            let (next, new_pos) = self.decode_step(seq_id, corrective, pos)?;
            t_pred = next;
            pos = new_pos;
        }

        let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
        let n_gen = generated.len();
        let accept_rate = if n_verify_tokens > 0 {
            n_accept_tokens as f64 / n_verify_tokens as f64
        } else {
            0.0
        };
        eprintln!(
            "[mlx-draft] seq {seq_id} done: {n_gen} tokens in {decode_ms:.0}ms ({:.1} tok/s) \
             attempts={n_attempts} accepted={n_accept_tokens}/{n_verify_tokens} (rate={accept_rate:.2})",
            n_gen as f64 / (decode_ms / 1000.0)
        );
        let out = self.decode(&generated).unwrap_or_default();
        self.remove_seq(seq_id).ok();
        // Free the draft's seq/KV state too.
        if let Some(d) = self.draft.as_mut() {
            d.runner.remove_seq(draft_seq_id).ok();
        }
        Ok(out)
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

    /// Encode WITHOUT special tokens — for injecting literal control text
    /// (e.g. forced `<parameter=KEY>` openers) into the decode stream, where
    /// an auto-prepended BOS/EOS would corrupt the running KV sequence.
    fn encode_raw(&self, text: &str) -> Result<Vec<u32>> {
        let tok = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| anyhow!("tokenizer not loaded"))?;
        let enc = tok
            .encode(text, false)
            .map_err(|e| anyhow!("tokenizer encode_raw: {e}"))?;
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

    /// [`Self::build_chat_input`] with inline images.
    ///
    /// Returns the prompt ids alongside the prepared images in prompt order —
    /// the pairing `prefill_with_images` relies on.
    ///
    /// The rendered text carries one `<|image_pad|>` per image; the expansion
    /// to one placeholder per merged token happens after tokenization, so the
    /// counts stay exact regardless of how the tokenizer would have handled a
    /// long repeated literal.
    ///
    /// Images are placed at the **start** of their turn, before that message's
    /// text: the flattened `(role, content)` shape has already lost where each
    /// content part sat, and image-then-question is what essentially every
    /// vision client sends.
    #[cfg(feature = "mlx-native")]
    pub fn build_chat_input_with_images(
        &self,
        messages: &[(String, String)],
        images: &[Vec<Vec<u8>>],
        thinking: bool,
    ) -> Result<(Vec<u32>, Vec<crate::qwen36_vision::PreparedImage>)> {
        let attached = self.attach_image_blocks(messages, images)?;
        let ids = self.encode(&format_qwen3_chat(&attached.messages, thinking))?;
        let ids = self.expand_image_placeholders(&ids, &attached.counts)?;
        Ok((ids, attached.prepared))
    }

    /// Decode every attached image and splice an `<|image_pad|>` block into the
    /// head of its message, returning the rewritten messages together with the
    /// prepared images and their token counts **in prompt order**.
    ///
    /// Prompt order is the contract: `prefill_with_images` pairs the k-th
    /// placeholder run with the k-th prepared image, and
    /// [`expand_image_placeholders`](Self::expand_image_placeholders) sizes the
    /// k-th run from `counts[k]`. Iterating messages in order and prepending to
    /// each body is what keeps all three in step.
    #[cfg(feature = "mlx-native")]
    fn attach_image_blocks(
        &self,
        messages: &[(String, String)],
        images: &[Vec<Vec<u8>>],
    ) -> Result<AttachedImages> {
        let mut out = AttachedImages {
            messages: Vec::with_capacity(messages.len()),
            prepared: Vec::new(),
            counts: Vec::new(),
        };
        for (i, (role, content)) in messages.iter().enumerate() {
            let attached = images.get(i).map(Vec::as_slice).unwrap_or(&[]);
            let mut body = String::new();
            for bytes in attached {
                let p = self.runner.prepare_image(bytes)?;
                out.counts.push(p.num_tokens);
                out.prepared.push(p);
                body.push_str(crate::qwen36_vision::IMAGE_BLOCK);
            }
            body.push_str(content);
            out.messages.push((role.clone(), body));
        }
        Ok(out)
    }

    /// Blow each single `<|image_pad|>` token up to the run of placeholder rows
    /// its image will occupy.
    ///
    /// Done on token ids rather than on the rendered text so the counts stay
    /// exact no matter how the tokenizer would have merged a long repeated
    /// literal.
    #[cfg(feature = "mlx-native")]
    fn expand_image_placeholders(&self, ids: &[u32], counts: &[usize]) -> Result<Vec<u32>> {
        let image_token = self
            .runner
            .image_token_id()
            .ok_or_else(|| anyhow!("config.json has no image_token_id"))?;
        crate::qwen36_vision::expand_image_placeholders(ids, image_token, counts)
            .map_err(|e| anyhow!("expand image placeholders: {e}"))
    }

    pub fn build_chat_input(
        &self,
        messages: &[(String, String)],
        thinking: bool,
    ) -> Result<Vec<u32>> {
        let prompt = format_qwen3_chat(messages, thinking);
        self.encode(&prompt)
    }

    /// Phase 2: tool-aware variant of `build_chat_input` that uses Qwen
    /// 3.6's nested-XML tool template. Returns the token ids AND the
    /// raw prefill string (which the caller must feed into the
    /// `Qwen35ResponseParser` before decoding so it starts in the
    /// correct state for Required/Tool(name) prefills).
    pub fn build_chat_input_with_tools(
        &self,
        messages: &[(String, String)],
        thinking: bool,
        tools: &[crate::chat_io::ToolDef<'_>],
        tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
    ) -> Result<(Vec<u32>, String)> {
        let (ids, prefill, _prefill_tokens) =
            self.build_chat_input_with_tools_split(messages, thinking, tools, tool_choice)?;
        Ok((ids, prefill))
    }

    /// Like [`build_chat_input_with_tools`] but also returns the exact trailing
    /// token ids that correspond to the appended `tool_choice` prefill suffix
    /// (computed as `full.len() - prompt_only.len()`). The Eager grammar path
    /// replays these through [`crate::grammar::Gemma4GrammarState::
    /// observe_prefill`] so the matcher's parse position lines up with what the
    /// model already has in context. Empty for Auto/None (no prefill).
    fn build_chat_input_with_tools_split(
        &self,
        messages: &[(String, String)],
        thinking: bool,
        tools: &[crate::chat_io::ToolDef<'_>],
        tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
    ) -> Result<(Vec<u32>, String, Vec<u32>)> {
        use crate::qwen3_5_tools::{format_qwen3_chat_with_tools, qwen35_tool_choice_prefill_str};
        let prompt_only = format_qwen3_chat_with_tools(messages, thinking, tools);
        let prefill = qwen35_tool_choice_prefill_str(tool_choice);
        let (ids, prefill_tokens) = self.encode_with_prefill_split(&prompt_only, &prefill)?;
        Ok((ids, prefill, prefill_tokens))
    }

    /// [`build_chat_input_with_tools_split`](Self::build_chat_input_with_tools_split)
    /// with inline images.
    ///
    /// The two features compose because they touch different parts of the
    /// prompt: images add placeholder runs inside the message bodies, the tool
    /// template wraps everything and appends its `tool_choice` prefill at the
    /// very end. Ordering matters in one place — the prefill split is taken
    /// *before* the placeholders expand. Expansion only ever grows runs that
    /// sit ahead of the prefill, so the prefill's token ids are the same either
    /// way, but taking the split afterwards would compare a grown `ids` against
    /// an ungrown `prompt_ids` and report a spurious BPE desync.
    #[cfg(feature = "mlx-native")]
    fn build_chat_input_with_tools_and_images_split(
        &self,
        messages: &[(String, String)],
        images: &[Vec<Vec<u8>>],
        thinking: bool,
        tools: &[crate::chat_io::ToolDef<'_>],
        tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
    ) -> Result<(
        Vec<u32>,
        String,
        Vec<u32>,
        Vec<crate::qwen36_vision::PreparedImage>,
    )> {
        use crate::qwen3_5_tools::{format_qwen3_chat_with_tools, qwen35_tool_choice_prefill_str};
        let attached = self.attach_image_blocks(messages, images)?;
        let prompt_only = format_qwen3_chat_with_tools(&attached.messages, thinking, tools);
        let prefill = qwen35_tool_choice_prefill_str(tool_choice);
        let (ids, prefill_tokens) = self.encode_with_prefill_split(&prompt_only, &prefill)?;
        let ids = self.expand_image_placeholders(&ids, &attached.counts)?;
        Ok((ids, prefill, prefill_tokens, attached.prepared))
    }

    /// Encode `prompt_only + prefill` and isolate the prefill's trailing token
    /// ids by tokenizing `prompt_only` alone and taking the suffix of the
    /// combined ids. If the combined encoding doesn't extend the prompt-only
    /// encoding as a clean prefix (BPE re-merged across the join), the prefill
    /// token split is reported empty — the caller then skips `observe_prefill`
    /// and, for the Eager path, drops the grammar (a desynced matcher is worse
    /// than free sampling). The full `ids` are always exact.
    fn encode_with_prefill_split(
        &self,
        prompt_only: &str,
        prefill: &str,
    ) -> Result<(Vec<u32>, Vec<u32>)> {
        let mut full = String::with_capacity(prompt_only.len() + prefill.len());
        full.push_str(prompt_only);
        full.push_str(prefill);
        let ids = self.encode(&full)?;
        if prefill.is_empty() {
            return Ok((ids, Vec::new()));
        }
        let prompt_ids = self.encode(prompt_only)?;
        // The split is valid only if the full encoding starts with the
        // prompt-only encoding (no cross-join BPE merge). Otherwise report no
        // clean prefill split.
        let prefill_tokens =
            if prompt_ids.len() <= ids.len() && ids[..prompt_ids.len()] == prompt_ids[..] {
                ids[prompt_ids.len()..].to_vec()
            } else {
                Vec::new()
            };
        Ok((ids, prefill_tokens))
    }

    /// Structured-history variant — used when the request carries
    /// prior assistant tool_calls or role:"tool" turns.
    pub fn build_chat_input_with_tools_from_history(
        &self,
        turns: &[crate::chat_io::ChatTurn<'_>],
        thinking: bool,
        tools: &[crate::chat_io::ToolDef<'_>],
        tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
    ) -> Result<(Vec<u32>, String)> {
        let (ids, prefill, _prefill_tokens) = self.build_chat_input_with_tools_from_history_split(
            turns,
            thinking,
            tools,
            tool_choice,
        )?;
        Ok((ids, prefill))
    }

    /// `_split` variant of [`build_chat_input_with_tools_from_history`] —
    /// see [`build_chat_input_with_tools_split`] for the prefill-token contract.
    fn build_chat_input_with_tools_from_history_split(
        &self,
        turns: &[crate::chat_io::ChatTurn<'_>],
        thinking: bool,
        tools: &[crate::chat_io::ToolDef<'_>],
        tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
    ) -> Result<(Vec<u32>, String, Vec<u32>)> {
        use crate::qwen3_5_tools::{
            format_qwen3_chat_with_tools_from_history, qwen35_tool_choice_prefill_str,
        };
        let prompt_only = format_qwen3_chat_with_tools_from_history(turns, thinking, tools);
        let prefill = qwen35_tool_choice_prefill_str(tool_choice);
        let (ids, prefill_tokens) = self.encode_with_prefill_split(&prompt_only, &prefill)?;
        Ok((ids, prefill, prefill_tokens))
    }

    /// [`build_chat_input_with_tools_from_history_split`](Self::build_chat_input_with_tools_from_history_split)
    /// with inline images, `images[i]` belonging to `turns[i]`.
    ///
    /// Only `User` turns can carry an image. That is a real restriction rather
    /// than a simplification: this renderer does not emit every turn's text
    /// verbatim — a leading `System` turn is folded into the `<tools>` block and
    /// an `Assistant` turn with tool calls may render no text at all — so a
    /// placeholder attached elsewhere could vanish or move, and the k-th
    /// placeholder run would then be spliced with the wrong image. Refusing is
    /// the only honest option; silently dropping the image is the failure mode
    /// this whole feature exists to avoid.
    #[cfg(feature = "mlx-native")]
    fn build_chat_input_with_tools_and_images_from_history_split(
        &self,
        turns: &[crate::chat_io::ChatTurn<'_>],
        images: &[Vec<Vec<u8>>],
        thinking: bool,
        tools: &[crate::chat_io::ToolDef<'_>],
        tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
    ) -> Result<(
        Vec<u32>,
        String,
        Vec<u32>,
        Vec<crate::qwen36_vision::PreparedImage>,
    )> {
        use crate::chat_io::ChatTurn;
        use crate::qwen3_5_tools::{
            format_qwen3_chat_with_tools_from_history, qwen35_tool_choice_prefill_str,
        };

        // Rewrite the User turns that carry images. The rewritten bodies are
        // owned here so the borrowed `ChatTurn`s built below can point at them.
        let mut bodies: Vec<Option<String>> = Vec::with_capacity(turns.len());
        let mut prepared = Vec::new();
        let mut counts = Vec::new();
        for (i, turn) in turns.iter().enumerate() {
            let attached = images.get(i).map(Vec::as_slice).unwrap_or(&[]);
            if attached.is_empty() {
                bodies.push(None);
                continue;
            }
            let ChatTurn::User(text) = turn else {
                return Err(anyhow!(
                    "images can only be attached to a user message on the Qwen 3.6 \
                     tool-calling path"
                ));
            };
            let mut body = String::new();
            for bytes in attached {
                let p = self.runner.prepare_image(bytes)?;
                counts.push(p.num_tokens);
                prepared.push(p);
                body.push_str(crate::qwen36_vision::IMAGE_BLOCK);
            }
            body.push_str(text);
            bodies.push(Some(body));
        }
        let rewritten: Vec<ChatTurn<'_>> = turns
            .iter()
            .zip(bodies.iter())
            .map(|(turn, body)| match body {
                Some(b) => ChatTurn::User(b.as_str()),
                None => turn.clone(),
            })
            .collect();

        let prompt_only = format_qwen3_chat_with_tools_from_history(&rewritten, thinking, tools);
        let prefill = qwen35_tool_choice_prefill_str(tool_choice);
        let (ids, prefill_tokens) = self.encode_with_prefill_split(&prompt_only, &prefill)?;
        let ids = self.expand_image_placeholders(&ids, &counts)?;
        Ok((ids, prefill, prefill_tokens, prepared))
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

    /// Lazily build (and cache) the llguidance parser factory for this model's
    /// tokenizer. Returns `None` if the factory can't be built (no
    /// `tokenizer.json`, empty vocab table) — callers then skip grammar and
    /// sample unconstrained. The expensive construction (~10–50 ms) happens
    /// once; the `Arc` is cloned per request. WS-C #2.
    fn grammar_factory(&self) -> Option<std::sync::Arc<llguidance::ParserFactory>> {
        self.grammar_factory
            .get_or_init(|| match resolve_tokenizer_json_path(&self.model_id) {
                Ok(path) => match crate::grammar::shared_factory_from_tokenizer(&path) {
                    Ok(f) => Some(f),
                    Err(e) => {
                        eprintln!(
                            "[qwen35-backend] grammar factory unavailable \
                             (tools sample without grammar mask): {e:#}"
                        );
                        None
                    }
                },
                Err(e) => {
                    eprintln!(
                        "[qwen35-backend] grammar factory: tokenizer.json unresolved \
                         (tools sample without grammar mask): {e:#}"
                    );
                    None
                }
            })
            .clone()
    }

    /// Build a per-request Qwen 3.6 tool-call grammar state (WS-C #2) from the
    /// request's tool defs + `tool_choice`. Returns `None` (sample
    /// unconstrained) when: the grammar is disabled, there are no tools,
    /// `tool_choice=None`, the factory is unavailable, a named choice refers to
    /// an unknown tool, or the grammar fails to compile. Required/named build
    /// an **Eager** grammar (active from token 0) so a prefill-forced opener
    /// can't produce an empty-param body; `auto` builds a **Lazy** grammar
    /// (off until the model starts a tool call) unless
    /// `LUMEN_QWEN35_TOOL_GRAMMAR_EAGER=1`.
    fn build_qwen35_tool_grammar(
        &self,
        tools: &[crate::chat_io::ToolDef<'_>],
        tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
    ) -> Option<crate::grammar::Gemma4GrammarState> {
        use crate::chat_io::ResolvedToolChoice;
        use crate::grammar::{Gemma4GrammarState, GrammarMode};

        // Grammar masking is only wired on the native MLX runner
        // (`decode_step_masked`); other backends can't apply the mask, so don't
        // build a grammar that would be inert.
        #[cfg(not(feature = "mlx-native"))]
        {
            let _ = (tools, tool_choice);
            return None;
        }
        #[cfg(feature = "mlx-native")]
        {
            if !qwen35_tool_grammar_enabled() || tools.is_empty() {
                return None;
            }
            // tool_choice=None means "must not call a tool" — no grammar.
            let mode = match tool_choice {
                ResolvedToolChoice::None => return None,
                ResolvedToolChoice::Required | ResolvedToolChoice::Tool(_) => GrammarMode::Eager,
                ResolvedToolChoice::Auto => {
                    if qwen35_tool_grammar_eager_auto() {
                        GrammarMode::Eager
                    } else {
                        GrammarMode::Lazy
                    }
                }
            };
            // For a named choice, the grammar must include the named tool; if the
            // engine didn't already downgrade an unknown name to Auto, skip rather
            // than build a grammar that can never match.
            if let ResolvedToolChoice::Tool(name) = tool_choice {
                if !tools.iter().any(|t| t.name == *name) {
                    eprintln!(
                        "[qwen35-backend] tool grammar skipped: tool_choice names unknown tool {name:?}"
                    );
                    return None;
                }
            }
            let factory = self.grammar_factory()?;
            // Convert the borrowed ToolDefs into the OpenAI-style `tools` JSON the
            // grammar builder consumes (`{"type":"function","function":{...}}`).
            let tools_json: Vec<serde_json::Value> = tools
                .iter()
                .map(|t| {
                    let mut func = serde_json::Map::new();
                    func.insert("name".into(), serde_json::Value::String(t.name.to_string()));
                    if let Some(p) = t.parameters {
                        func.insert("parameters".into(), p.clone());
                    }
                    serde_json::json!({ "type": "function", "function": func })
                })
                .collect();
            match Gemma4GrammarState::new_qwen35_xml(factory, &tools_json, mode, None) {
                Ok(state) => {
                    eprintln!(
                        "[qwen35-backend] tool grammar active for {} tool(s) (mode={mode:?})",
                        tools.len()
                    );
                    Some(state)
                }
                Err(e) => {
                    eprintln!("[qwen35-backend] tool grammar build failed (sampling free): {e:#}");
                    None
                }
            }
        }
    }

    /// Build a per-request `response_format` (JSON-schema) grammar for Qwen 3.6
    /// (WS-F #1). Returns `None` (decode free text — the server still trims to
    /// the first JSON value) when the response-format grammar is disabled, the
    /// factory is unavailable, or the schema fails to compile into an
    /// llguidance grammar.
    ///
    /// The grammar is model-agnostic ([`Gemma4GrammarState::new_json_schema`] —
    /// the factory is built from *this* model's `tokenizer.json`) and **Eager**:
    /// unlike a tool call there is no opener token to lazily trigger on, so the
    /// JSON constraint must be live from the very first decode step (the whole
    /// assistant message must be the JSON value).
    fn build_qwen35_response_format_grammar(
        &self,
        schema: &serde_json::Value,
    ) -> Option<crate::grammar::Gemma4GrammarState> {
        use crate::grammar::{Gemma4GrammarState, GrammarMode};

        // Grammar masking is only wired on the native MLX runner
        // (`decode_step_masked`); other backends can't apply the mask.
        #[cfg(not(feature = "mlx-native"))]
        {
            let _ = schema;
            return None;
        }
        #[cfg(feature = "mlx-native")]
        {
            if !qwen35_response_format_enabled() {
                return None;
            }
            let factory = self.grammar_factory()?;
            match Gemma4GrammarState::new_json_schema(factory, schema, GrammarMode::Eager) {
                Ok(state) => {
                    eprintln!("[qwen35-backend] response_format grammar active (json_schema)");
                    Some(state)
                }
                Err(e) => {
                    eprintln!(
                        "[qwen35-backend] response_format grammar build failed \
                         (sampling free): {e:#}"
                    );
                    None
                }
            }
        }
    }

    /// Shared decode loop for the `response_format` (JSON-schema) entry points.
    /// Drives an Eager JSON-schema grammar over the **visible output channel**:
    /// the entire assistant message is constrained to be a single schema-shaped
    /// JSON value. Used by the flat-message and structured-history chat paths
    /// (streaming + non-streaming) — `on_token` receives each visible text
    /// fragment so streaming clients see the JSON build up incrementally.
    ///
    /// SSM safety: every sampled token still flows through a normal forward;
    /// the grammar only masks the **host logits** AFTER the forward (via
    /// `decode_step_masked`), changing WHICH token is argmaxed, never skipping a
    /// forward or jumping tokens — so the conv/SSM recurrent state of Qwen 3.6's
    /// ~75% linear-attn layers advances identically to an unconstrained decode.
    ///
    /// On grammar build failure (`grammar == None`) or a mid-decode matcher
    /// desync, this degrades to a plain greedy decode — the server-side
    /// first-JSON-value trim still yields a parseable object in the common case.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn chat_response_format_impl<F>(
        &mut self,
        prompt_ids: Vec<u32>,
        // Images to splice over the prompt's placeholder runs, in prompt order.
        // Empty for a text-only request, which then prefills byte-identically
        // to the pre-vision path.
        prepared_images: Vec<crate::qwen36_vision::PreparedImage>,
        grammar: Option<crate::grammar::Gemma4GrammarState>,
        max_new_tokens: usize,
        seq_id: u64,
        mut on_token: F,
    ) -> Result<crate::chat_io::ParsedResponse>
    where
        F: FnMut(&str),
    {
        use crate::chat_io::ParsedResponse;

        if prompt_ids.is_empty() {
            return Err(anyhow!("empty prompt after tokenization"));
        }

        let mut grammar = grammar;
        #[cfg_attr(not(feature = "mlx-native"), allow(unused_variables))]
        let grammar_active_at_start = grammar.as_ref().is_some_and(|g| g.is_active());

        // ── First-token masking ──
        // For a free-form JSON `response_format` there is NO opener token to
        // prefill (unlike a tool call's `<tool_call>` block), so the VERY FIRST
        // generated token must already be grammar-masked (e.g. forced to `{`).
        // `prefill`'s `argmax_last_token` is unmasked, so when a grammar is
        // active we prefill `prompt_ids[..n-1]` and produce the first token via
        // a *masked* decode step on the final prompt token. The cache advance is
        // identical (N-1 prefill + 1 decode == N-token forward); the only
        // difference is the first token's logits get the JSON mask. When no
        // grammar is active we keep the plain single-shot prefill.
        let t_prefill = std::time::Instant::now();
        let (mut last, mut pos);
        #[cfg(feature = "mlx-native")]
        let mut first_masked = false;
        // Vision prompts prefill through the image-splicing path; the held-back
        // token is the tail of the assistant header, never a placeholder, so the
        // runs stay wholly inside the prefilled span either way.
        #[cfg(feature = "mlx-native")]
        macro_rules! prefill_span {
            ($ids:expr) => {
                if prepared_images.is_empty() {
                    self.prefill(seq_id, $ids)
                } else {
                    self.prefill_with_images(seq_id, $ids, &prepared_images)
                }
            };
        }
        #[cfg(feature = "mlx-native")]
        if grammar_active_at_start && prompt_ids.len() >= 2 {
            let split = prompt_ids.len() - 1;
            let (_pf, _pos0) = prefill_span!(&prompt_ids[..split])?;
            let final_tok = prompt_ids[split];
            let stepped = {
                let g = grammar.as_mut().expect("grammar active implies Some");
                let mut mask =
                    |buf: &mut [f32]| -> Result<()> { g.apply_mask_to_logits(buf).map(|_| ()) };
                self.runner.decode_step_masked(seq_id, final_tok, &mut mask)
            };
            let (n, p) = stepped?;
            last = n;
            pos = p;
            first_masked = true;
        } else {
            let (l, p) = prefill_span!(&prompt_ids)?;
            last = l;
            pos = p;
        }
        #[cfg(not(feature = "mlx-native"))]
        {
            if !prepared_images.is_empty() {
                return Err(anyhow!(
                    "image input requires a build with the `mlx-native` feature"
                ));
            }
            let (l, p) = self.prefill(seq_id, &prompt_ids)?;
            last = l;
            pos = p;
        }
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[mlx] seq {seq_id} prefill-rf: {} tokens in {prefill_ms:.0}ms ({:.1} tok/s) -> tok={last}",
            prompt_ids.len(),
            prompt_ids.len() as f64 / (prefill_ms / 1000.0)
        );

        // Advance the matcher with the first token (`decode_step_masked` masks
        // but does NOT observe). When the token came from the masked decode
        // (`first_masked`) it's grammar-legal by construction so `observe`
        // succeeds; in the unmasked fallback path `observe` may reject it — drop
        // the grammar then (the server's first-JSON-value trim still recovers an
        // object).
        #[cfg(feature = "mlx-native")]
        let _ = first_masked;
        if let Some(g) = grammar.as_mut() {
            if g.is_active() {
                if let Err(e) = g.observe(last) {
                    eprintln!(
                        "[qwen35-backend] response_format grammar first-token observe \
                         desynced (dropping grammar, sampling free): {e:#}"
                    );
                    grammar = None;
                }
            }
        }

        let mut generated: Vec<u32> = vec![last];
        let mut emitted_idx: usize = 0;
        if let Ok(text) = self.decode(&generated) {
            if !text.is_empty() && !text.contains('\u{FFFD}') {
                on_token(&text);
                emitted_idx = generated.len();
            }
        }
        if self.eos_tokens.contains(&last) {
            let out = self.decode(&generated).unwrap_or_default();
            self.remove_seq(seq_id).ok();
            return Ok(ParsedResponse {
                visible: out,
                reasoning: String::new(),
                tool_calls: Vec::new(),
            });
        }

        let t_decode = std::time::Instant::now();
        for step in 1..max_new_tokens {
            // Grammar-masked decode when active: same forward (KV + SSM state
            // advance identically), host logits masked before argmax, then the
            // sampled token fed back to the matcher.
            let grammar_active = grammar.as_ref().is_some_and(|g| g.is_active());
            let (next, new_pos) = if grammar_active {
                #[cfg(feature = "mlx-native")]
                {
                    let stepped = {
                        let g = grammar.as_mut().expect("grammar_active implies Some");
                        let mut mask = |buf: &mut [f32]| -> Result<()> {
                            g.apply_mask_to_logits(buf).map(|_| ())
                        };
                        self.runner.decode_step_masked(seq_id, last, &mut mask)
                    };
                    let (next, new_pos) = stepped?;
                    let observe_err = grammar.as_mut().and_then(|g| g.observe(next).err());
                    if let Some(e) = observe_err {
                        eprintln!(
                            "[qwen35-backend] response_format grammar observe(sampled) \
                             desynced (dropping grammar): {e:#}"
                        );
                        grammar = None;
                    }
                    (next, new_pos)
                }
                #[cfg(not(feature = "mlx-native"))]
                {
                    self.decode_step(seq_id, last, pos)?
                }
            } else {
                self.decode_step(seq_id, last, pos)?
            };
            last = next;
            pos = new_pos;
            generated.push(next);
            let tail_start = emitted_idx;
            if tail_start < generated.len() {
                if let Ok(text) = self.decode(&generated[tail_start..]) {
                    if !text.is_empty() && !text.contains('\u{FFFD}') {
                        on_token(&text);
                        emitted_idx = generated.len();
                    }
                }
            }
            if self.eos_tokens.contains(&next) {
                eprintln!(
                    "[mlx] seq {seq_id} EOS-rf at step {step} ({:.1} tok/s)",
                    step as f64 / (t_decode.elapsed().as_secs_f64())
                );
                break;
            }
        }
        let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
        let n_gen = generated.len();
        eprintln!(
            "[mlx] seq {seq_id} done-rf: {n_gen} tokens in {decode_ms:.0}ms ({:.1} tok/s)",
            n_gen as f64 / (decode_ms / 1000.0)
        );
        let out = self.decode(&generated).unwrap_or_default();
        self.remove_seq(seq_id).ok();
        Ok(ParsedResponse {
            visible: out,
            reasoning: String::new(),
            tool_calls: Vec::new(),
        })
    }

    /// `response_format` non-streaming chat (flat messages). Builds a
    /// JSON-schema grammar from `schema` and runs the masked decode loop.
    pub fn chat_response_format(
        &mut self,
        messages: &[(String, String)],
        images: &[Vec<Vec<u8>>],
        max_new_tokens: usize,
        thinking: bool,
        seq_id: u64,
        schema: &serde_json::Value,
    ) -> Result<crate::chat_io::ParsedResponse> {
        let (prompt_ids, prepared) =
            self.build_response_format_input(messages, images, thinking)?;
        let grammar = self.build_qwen35_response_format_grammar(schema);
        self.chat_response_format_impl(
            prompt_ids,
            prepared,
            grammar,
            max_new_tokens,
            seq_id,
            |_| {},
        )
    }

    /// Prompt for a flat-message `response_format` request, with or without
    /// images. Split out so the streaming and non-streaming entry points cannot
    /// drift on how the placeholder runs are built.
    fn build_response_format_input(
        &self,
        messages: &[(String, String)],
        images: &[Vec<Vec<u8>>],
        thinking: bool,
    ) -> Result<(Vec<u32>, Vec<crate::qwen36_vision::PreparedImage>)> {
        if !images.iter().any(|v| !v.is_empty()) {
            return Ok((self.build_chat_input(messages, thinking)?, Vec::new()));
        }
        #[cfg(feature = "mlx-native")]
        {
            self.build_chat_input_with_images(messages, images, thinking)
        }
        #[cfg(not(feature = "mlx-native"))]
        {
            Err(anyhow!(
                "image input requires a build with the `mlx-native` feature"
            ))
        }
    }

    /// Structured-history counterpart of
    /// [`build_response_format_input`](Self::build_response_format_input).
    ///
    /// Goes through the tool-aware renderer with no tools — it is the only one
    /// that can represent assistant `tool_calls` / `role:tool` turns — so the
    /// same `User`-turn-only image rule applies.
    fn build_response_format_input_from_history(
        &self,
        turns: &[crate::chat_io::ChatTurn<'_>],
        images: &[Vec<Vec<u8>>],
        thinking: bool,
    ) -> Result<(Vec<u32>, Vec<crate::qwen36_vision::PreparedImage>)> {
        use crate::chat_io::ResolvedToolChoice;
        if !images.iter().any(|v| !v.is_empty()) {
            let (ids, _prefill, _prefill_tokens) = self
                .build_chat_input_with_tools_from_history_split(
                    turns,
                    thinking,
                    &[],
                    &ResolvedToolChoice::None,
                )?;
            return Ok((ids, Vec::new()));
        }
        #[cfg(feature = "mlx-native")]
        {
            let (ids, _prefill, _prefill_tokens, prepared) = self
                .build_chat_input_with_tools_and_images_from_history_split(
                    turns,
                    images,
                    thinking,
                    &[],
                    &ResolvedToolChoice::None,
                )?;
            Ok((ids, prepared))
        }
        #[cfg(not(feature = "mlx-native"))]
        {
            Err(anyhow!(
                "image input requires a build with the `mlx-native` feature"
            ))
        }
    }

    /// `response_format` streaming chat (flat messages). Same as
    /// [`chat_response_format`] but forwards each visible JSON fragment to
    /// `on_token`.
    #[allow(clippy::too_many_arguments)]
    pub fn chat_streaming_response_format<F>(
        &mut self,
        messages: &[(String, String)],
        images: &[Vec<Vec<u8>>],
        max_new_tokens: usize,
        thinking: bool,
        seq_id: u64,
        schema: &serde_json::Value,
        on_token: F,
    ) -> Result<crate::chat_io::ParsedResponse>
    where
        F: FnMut(&str),
    {
        let (prompt_ids, prepared) =
            self.build_response_format_input(messages, images, thinking)?;
        let grammar = self.build_qwen35_response_format_grammar(schema);
        self.chat_response_format_impl(
            prompt_ids,
            prepared,
            grammar,
            max_new_tokens,
            seq_id,
            on_token,
        )
    }

    /// `response_format` non-streaming chat (structured history). Routes the
    /// turns through the tool-aware renderer (it is the only renderer that
    /// represents assistant.tool_calls / role:tool turns) with no tools, then
    /// runs the JSON-schema masked decode loop.
    pub fn chat_response_format_from_history(
        &mut self,
        turns: &[crate::chat_io::ChatTurn<'_>],
        images: &[Vec<Vec<u8>>],
        max_new_tokens: usize,
        thinking: bool,
        seq_id: u64,
        schema: &serde_json::Value,
    ) -> Result<crate::chat_io::ParsedResponse> {
        let (prompt_ids, prepared) =
            self.build_response_format_input_from_history(turns, images, thinking)?;
        let grammar = self.build_qwen35_response_format_grammar(schema);
        self.chat_response_format_impl(
            prompt_ids,
            prepared,
            grammar,
            max_new_tokens,
            seq_id,
            |_| {},
        )
    }

    /// `response_format` streaming chat (structured history).
    #[allow(clippy::too_many_arguments)]
    pub fn chat_streaming_response_format_from_history<F>(
        &mut self,
        turns: &[crate::chat_io::ChatTurn<'_>],
        images: &[Vec<Vec<u8>>],
        max_new_tokens: usize,
        thinking: bool,
        seq_id: u64,
        schema: &serde_json::Value,
        on_token: F,
    ) -> Result<crate::chat_io::ParsedResponse>
    where
        F: FnMut(&str),
    {
        let (prompt_ids, prepared) =
            self.build_response_format_input_from_history(turns, images, thinking)?;
        let grammar = self.build_qwen35_response_format_grammar(schema);
        self.chat_response_format_impl(
            prompt_ids,
            prepared,
            grammar,
            max_new_tokens,
            seq_id,
            on_token,
        )
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
        on_token: F,
    ) -> Result<String>
    where
        F: FnMut(&str),
    {
        self.chat_streaming_with_images(messages, &[], max_new_tokens, thinking, seq_id, on_token)
    }

    /// [`Self::chat_streaming`] with inline images.
    ///
    /// `images[i]` holds the encoded byte streams attached to `messages[i]`.
    /// An empty slice is byte-identical to [`Self::chat_streaming`].
    ///
    /// Image requests skip MTP, draft/n-gram speculation and the prefix cache:
    /// all four are keyed on text alone, and a vision prompt's placeholder rows
    /// only mean anything together with the image spliced over them.
    #[allow(clippy::too_many_arguments)]
    pub fn chat_streaming_with_images<F>(
        &mut self,
        messages: &[(String, String)],
        images: &[Vec<Vec<u8>>],
        max_new_tokens: usize,
        thinking: bool,
        seq_id: u64,
        mut on_token: F,
    ) -> Result<String>
    where
        F: FnMut(&str),
    {
        let has_images = images.iter().any(|v| !v.is_empty());
        // Phase 2 S4 — MTP routing. `LUMEN_SPEC=mtp` activates the
        // qwen3_5_mtp speculative path when (1) the native runner installed
        // a drafter (via LUMEN_QWEN35_MTP=1 at load), and (2) the request
        // hasn't opted out. Falls back to baseline decode_step otherwise so
        // unmtp-loaded deployments behave identically.
        #[cfg(feature = "mlx-native")]
        if !has_images && self.qwen35_mtp_enabled() {
            if let Some(k) = effective_qwen35_mtp_k() {
                return self.chat_streaming_qwen35_mtp(
                    messages,
                    max_new_tokens,
                    thinking,
                    seq_id,
                    k,
                    on_token,
                );
            }
        }

        // OPT-IN draft-model speculative decode. Engaged only when (1) a draft
        // model was loaded from `LUMEN_MLX_DRAFT_MODEL` AND (2) the request is
        // greedy. This inner Qwen3.5-family path is greedy by construction (the
        // dispatcher in `MlxBackend::chat_streaming` discards temperature/top_p
        // before calling here), so reaching this branch implies greedy. Default
        // OFF: when no draft is loaded, this is a no-op and the existing path
        // below runs byte-identically.
        #[cfg(feature = "mlx-native")]
        if !has_images && self.draft_enabled() {
            return self.chat_streaming_spec_draft(
                messages,
                max_new_tokens,
                thinking,
                seq_id,
                on_token,
            );
        }

        if !has_images && let Some(cfg) = spec_decode::read_spec_config() {
            return self.chat_streaming_spec_ngram(
                messages,
                max_new_tokens,
                thinking,
                seq_id,
                cfg,
                on_token,
            );
        }

        if self.prefix_store.enabled() {
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

        // Images change both halves: the prompt gains a placeholder run per
        // image, and prefill has to splice the encoded features over it.
        #[cfg(feature = "mlx-native")]
        let (prompt_ids, prepared) = if has_images {
            self.build_chat_input_with_images(messages, images, thinking)?
        } else {
            (self.build_chat_input(messages, thinking)?, Vec::new())
        };
        #[cfg(not(feature = "mlx-native"))]
        let prompt_ids = {
            if has_images {
                return Err(anyhow!(
                    "image input requires a build with the `mlx-native` feature"
                ));
            }
            self.build_chat_input(messages, thinking)?
        };
        if prompt_ids.is_empty() {
            return Err(anyhow!("empty prompt after tokenization"));
        }
        let t_prefill = std::time::Instant::now();
        #[cfg(feature = "mlx-native")]
        let (mut last, mut pos) = self.prefill_with_images(seq_id, &prompt_ids, &prepared)?;
        #[cfg(not(feature = "mlx-native"))]
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

    /// Phase 2: tool-aware non-streaming chat. Routes through the
    /// Qwen 3.6 nested-XML chat template and parses any
    /// `<tool_call>...</tool_call>` blocks from the model's output.
    /// Bypasses MTP / spec-decode / prefix-cache for now — tool-calling
    /// is a feature gate and the fast paths can be re-added once the
    /// happy path is verified on real hardware.
    pub fn chat_with_tools(
        &mut self,
        messages: &[(String, String)],
        images: &[Vec<Vec<u8>>],
        max_new_tokens: usize,
        thinking: bool,
        seq_id: u64,
        tools: &[crate::chat_io::ToolDef<'_>],
        tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
    ) -> Result<crate::chat_io::ParsedResponse> {
        self.chat_with_tools_flat(
            messages,
            images,
            max_new_tokens,
            thinking,
            seq_id,
            tools,
            tool_choice,
            |_ev| Ok(()),
        )
    }

    /// Phase 2: tool-aware streaming chat. Emits
    /// `BackendStreamEvent::Text` for visible-text deltas and
    /// `BackendStreamEvent::ToolCallStart { name }` the moment the
    /// parser sees `<function=NAME>`.
    #[allow(clippy::too_many_arguments)]
    pub fn chat_streaming_with_tools<F>(
        &mut self,
        messages: &[(String, String)],
        images: &[Vec<Vec<u8>>],
        max_new_tokens: usize,
        thinking: bool,
        seq_id: u64,
        tools: &[crate::chat_io::ToolDef<'_>],
        tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
        on_event: F,
    ) -> Result<crate::chat_io::ParsedResponse>
    where
        F: FnMut(crate::chat_io::BackendStreamEvent<'_>) -> Result<()>,
    {
        self.chat_with_tools_flat(
            messages,
            images,
            max_new_tokens,
            thinking,
            seq_id,
            tools,
            tool_choice,
            on_event,
        )
    }

    /// Structured-history non-streaming variant.
    pub fn chat_with_tools_from_history(
        &mut self,
        turns: &[crate::chat_io::ChatTurn<'_>],
        images: &[Vec<Vec<u8>>],
        max_new_tokens: usize,
        thinking: bool,
        seq_id: u64,
        tools: &[crate::chat_io::ToolDef<'_>],
        tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
    ) -> Result<crate::chat_io::ParsedResponse> {
        self.chat_with_tools_history(
            turns,
            images,
            max_new_tokens,
            thinking,
            seq_id,
            tools,
            tool_choice,
            |_ev| Ok(()),
        )
    }

    /// Structured-history streaming variant.
    #[allow(clippy::too_many_arguments)]
    pub fn chat_streaming_with_tools_from_history<F>(
        &mut self,
        turns: &[crate::chat_io::ChatTurn<'_>],
        images: &[Vec<Vec<u8>>],
        max_new_tokens: usize,
        thinking: bool,
        seq_id: u64,
        tools: &[crate::chat_io::ToolDef<'_>],
        tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
        on_event: F,
    ) -> Result<crate::chat_io::ParsedResponse>
    where
        F: FnMut(crate::chat_io::BackendStreamEvent<'_>) -> Result<()>,
    {
        self.chat_with_tools_history(
            turns,
            images,
            max_new_tokens,
            thinking,
            seq_id,
            tools,
            tool_choice,
            on_event,
        )
    }

    /// Prompt-building half of the flat-message tool path, shared by the
    /// streaming and non-streaming entry points (they differ only in the sink).
    ///
    /// Images turn off both prefix-cache reuse and the incremental boundary:
    /// each is keyed on text alone, so a hit would hand this request KV rows
    /// built from another request's pixels.
    #[allow(clippy::too_many_arguments)]
    fn chat_with_tools_flat<F>(
        &mut self,
        messages: &[(String, String)],
        images: &[Vec<Vec<u8>>],
        max_new_tokens: usize,
        thinking: bool,
        seq_id: u64,
        tools: &[crate::chat_io::ToolDef<'_>],
        tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
        on_event: F,
    ) -> Result<crate::chat_io::ParsedResponse>
    where
        F: FnMut(crate::chat_io::BackendStreamEvent<'_>) -> Result<()>,
    {
        let has_images = images.iter().any(|v| !v.is_empty());
        #[cfg(feature = "mlx-native")]
        let (prompt_ids, prefill, prefill_tokens, prepared) = if has_images {
            self.build_chat_input_with_tools_and_images_split(
                messages,
                images,
                thinking,
                tools,
                tool_choice,
            )?
        } else {
            let (ids, prefill, prefill_tokens) =
                self.build_chat_input_with_tools_split(messages, thinking, tools, tool_choice)?;
            (ids, prefill, prefill_tokens, Vec::new())
        };
        #[cfg(not(feature = "mlx-native"))]
        let (prompt_ids, prefill, prefill_tokens, prepared) = {
            if has_images {
                return Err(anyhow!(
                    "image input requires a build with the `mlx-native` feature"
                ));
            }
            let (ids, prefill, prefill_tokens) =
                self.build_chat_input_with_tools_split(messages, thinking, tools, tool_choice)?;
            (ids, prefill, prefill_tokens, Vec::new())
        };
        let grammar = self.build_qwen35_tool_grammar(tools, tool_choice);
        let prefix_key = if has_images {
            None
        } else {
            auto_prefix_key(messages)
        };
        let incremental_boundary = if has_images {
            None
        } else {
            self.detect_system_tools_prefix_len(messages, tools)
                .ok()
                .filter(|&b| b > 0 && b < prompt_ids.len())
        };
        self.chat_with_tools_impl(
            prompt_ids,
            prefill,
            prefill_tokens,
            grammar,
            max_new_tokens,
            seq_id,
            prefix_key.as_deref(),
            incremental_boundary,
            if force_required_params_enabled() {
                force_required_params_map(tools)
            } else {
                Default::default()
            },
            prepared,
            on_event,
        )
    }

    /// Structured-history counterpart of [`chat_with_tools_flat`](Self::chat_with_tools_flat).
    #[allow(clippy::too_many_arguments)]
    fn chat_with_tools_history<F>(
        &mut self,
        turns: &[crate::chat_io::ChatTurn<'_>],
        images: &[Vec<Vec<u8>>],
        max_new_tokens: usize,
        thinking: bool,
        seq_id: u64,
        tools: &[crate::chat_io::ToolDef<'_>],
        tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
        on_event: F,
    ) -> Result<crate::chat_io::ParsedResponse>
    where
        F: FnMut(crate::chat_io::BackendStreamEvent<'_>) -> Result<()>,
    {
        let has_images = images.iter().any(|v| !v.is_empty());
        #[cfg(feature = "mlx-native")]
        let (prompt_ids, prefill, prefill_tokens, prepared) = if has_images {
            self.build_chat_input_with_tools_and_images_from_history_split(
                turns,
                images,
                thinking,
                tools,
                tool_choice,
            )?
        } else {
            let (ids, prefill, prefill_tokens) = self
                .build_chat_input_with_tools_from_history_split(
                    turns,
                    thinking,
                    tools,
                    tool_choice,
                )?;
            (ids, prefill, prefill_tokens, Vec::new())
        };
        #[cfg(not(feature = "mlx-native"))]
        let (prompt_ids, prefill, prefill_tokens, prepared) = {
            if has_images {
                return Err(anyhow!(
                    "image input requires a build with the `mlx-native` feature"
                ));
            }
            let (ids, prefill, prefill_tokens) = self
                .build_chat_input_with_tools_from_history_split(
                    turns,
                    thinking,
                    tools,
                    tool_choice,
                )?;
            (ids, prefill, prefill_tokens, Vec::new())
        };
        let grammar = self.build_qwen35_tool_grammar(tools, tool_choice);
        let prefix_key = if has_images {
            None
        } else {
            auto_prefix_key_from_turns(turns)
        };
        let incremental_boundary = if has_images {
            None
        } else {
            self.detect_system_tools_prefix_len_from_turns(turns, tools)
                .ok()
                .filter(|&b| b > 0 && b < prompt_ids.len())
        };
        self.chat_with_tools_impl(
            prompt_ids,
            prefill,
            prefill_tokens,
            grammar,
            max_new_tokens,
            seq_id,
            prefix_key.as_deref(),
            incremental_boundary,
            if force_required_params_enabled() {
                force_required_params_map(tools)
            } else {
                Default::default()
            },
            prepared,
            on_event,
        )
    }

    /// Shared decode loop for the four tool-aware entry points. Prefill
    /// the prompt (including any tool_choice prefill suffix), feed the
    /// prefill string into the parser so its state machine starts at
    /// the right place (`InToolCallHeader` for Required, `InToolCallBody`
    /// for Tool(name)), then drive a simple greedy decode loop calling
    /// the parser on each new text delta and surfacing events upstream.
    fn chat_with_tools_impl<F>(
        &mut self,
        prompt_ids: Vec<u32>,
        prefill_str: String,
        // Exact trailing token ids of `prefill_str` as they appear in
        // `prompt_ids` (from the `_split` builders). Replayed into an Eager
        // grammar via `observe_prefill` so the matcher's parse position lines
        // up with the model's prefilled `<tool_call>\n<function=…` context.
        // Empty when there's no prefill or the BPE split wasn't clean.
        prefill_tokens: Vec<u32>,
        // Per-request Qwen 3.6 tool-call grammar (WS-C #2). `Some` only for
        // `tool_choice=required`/named (Eager) or auto with the eager opt-in;
        // `None` → unconstrained decode (byte-identical to the pre-grammar
        // path). When active, each decode step routes through
        // `decode_step_masked` so disallowed tokens are masked before argmax.
        grammar: Option<crate::grammar::Gemma4GrammarState>,
        max_new_tokens: usize,
        seq_id: u64,
        // Auto-derived key (from system message hash) or explicit session_id
        // passed by the public callers. `None` disables prefix caching for
        // this request even when the feature is enabled — useful for ad-hoc
        // benchmarks that want clean cold-prefill timing.
        prefix_cache_key: Option<&str>,
        // Phase 0 incremental-prefix boundary: token length of the shared
        // system-prompt head, when known and a strict interior of the prompt.
        // `Some(b)` lets the cold-MISS path snapshot `[..b]` as a reusable
        // boundary (only acts when `LUMEN_MLX_PREFIX_INCREMENTAL=1`); `None`
        // keeps the original single-prefill MISS.
        incremental_boundary: Option<usize>,
        // Tool name → required param keys. Empty unless
        // `LUMEN_QWEN35_FORCE_REQUIRED_PARAMS` is on; when non-empty the decode
        // loop injects a `<parameter=KEY>\n` opener before the model can close
        // a `<function=NAME>` block with a required param still missing.
        force_required: std::collections::HashMap<String, Vec<String>>,
        // Images to splice over the prompt's placeholder runs, in prompt order.
        // Empty for a text-only request, which then prefills byte-identically
        // to the pre-vision path.
        prepared_images: Vec<crate::qwen36_vision::PreparedImage>,
        mut on_event: F,
    ) -> Result<crate::chat_io::ParsedResponse>
    where
        F: FnMut(crate::chat_io::BackendStreamEvent<'_>) -> Result<()>,
    {
        use crate::qwen3_5_tools::Qwen35ResponseParser;

        if prompt_ids.is_empty() {
            return Err(anyhow!("empty prompt after tokenization"));
        }

        let mut parser = Qwen35ResponseParser::new();
        // Prime the parser with the prefill so its state machine
        // transitions OUT of Visible BEFORE the model starts generating.
        // Events from the prefill are SUPPRESSED — the engine assigns
        // the wire id only when the MODEL produces the call, not when
        // the prompt does.
        if !prefill_str.is_empty() {
            let _ = parser.feed(&prefill_str);
        }

        let debug_qwen_tools = std::env::var("LUMEN_QWEN35_TOOL_DEBUG").is_ok();
        let t_prefill = std::time::Instant::now();
        // A vision prompt cannot reuse a cached prefix: the cache is keyed on
        // text alone, and the placeholder rows only mean anything once this
        // request's images have been spliced over them. The callers already
        // pass `None` for the key; this asserts the invariant at the one place
        // that would silently corrupt the prompt if it were violated.
        //
        // ── First-token masking ──
        // Prefill argmaxes its final position with no grammar mask, so with an
        // active grammar the very first generated token was the one token the
        // matcher never got to constrain. Checking it afterwards is too late:
        // the model's unmasked pick disagreeing with the grammar is precisely
        // when the grammar is doing its job, yet the old code responded by
        // throwing the grammar away. `tool_choice=required` therefore never
        // enforced anything — asked for `get_weather`, the model would open
        // `<function=` and continue `weather`, the matcher would reject it, and
        // the request would finish on free sampling. Holding the last prompt
        // token back and feeding it through `decode_step_masked` puts that
        // first choice under the mask like every later one.
        //
        // Only the last token moves, and it is always the tail of the assistant
        // header or the `tool_choice` prefill — never an image placeholder — so
        // the image runs stay wholly inside the prefilled span. The `is_none`
        // guard makes that explicit rather than relying on it.
        let mut grammar = grammar;
        #[cfg(feature = "mlx-native")]
        let hold_back_last = hold_back_last_prompt_token(
            grammar.as_ref().is_some_and(|g| g.is_active()),
            &prompt_ids,
            self.runner.image_token_id(),
        );
        #[cfg(not(feature = "mlx-native"))]
        let hold_back_last = false;
        let prefill_ids = if hold_back_last {
            &prompt_ids[..prompt_ids.len() - 1]
        } else {
            &prompt_ids[..]
        };

        #[cfg(feature = "mlx-native")]
        let (mut last, mut pos) = if prepared_images.is_empty() {
            self.prefix_store.prefill_optionally_cached(
                &mut self.runner,
                seq_id,
                prefill_ids,
                prefix_cache_key,
                incremental_boundary,
            )?
        } else {
            self.prefill_with_images(seq_id, prefill_ids, &prepared_images)?
        };
        #[cfg(not(feature = "mlx-native"))]
        let (mut last, mut pos) = {
            if !prepared_images.is_empty() {
                return Err(anyhow!(
                    "image input requires a build with the `mlx-native` feature"
                ));
            }
            self.prefix_store.prefill_optionally_cached(
                &mut self.runner,
                seq_id,
                prefill_ids,
                prefix_cache_key,
                incremental_boundary,
            )?
        };
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[mlx] seq {seq_id} prefill-tools: {} tokens in {prefill_ms:.0}ms ({:.1} tok/s) -> tok={last}",
            prefill_ids.len(),
            prefill_ids.len() as f64 / (prefill_ms / 1000.0)
        );

        // ── WS-C #2: grammar prefill replay ──
        // For an active (Eager) grammar, the chat template prefilled the
        // `<tool_call>\n<function=…` opener via `prompt_ids` — those tokens
        // went into the model's context but were never *sampled*, so the
        // matcher hasn't seen them. Replay the exact prefill tokens through
        // `observe_prefill` so the matcher's parse position matches the model.
        // A desync here (empty split despite a prefill, or a token the grammar
        // rejects) drops the grammar → free sampling, never a corrupt masked
        // decode. Unlike the first *generated* token above, these are tokens the
        // prompt already committed to, so there is nothing left to constrain.
        if let Some(g) = grammar.as_mut() {
            if g.is_active() {
                let mut desync = !prefill_str.is_empty() && prefill_tokens.is_empty();
                if !desync {
                    for tok in &prefill_tokens {
                        if let Err(e) = g.observe_prefill(*tok) {
                            eprintln!(
                                "[qwen35-backend] grammar prefill replay desynced \
                                 (dropping grammar, sampling free): {e:#}"
                            );
                            desync = true;
                            break;
                        }
                    }
                }
                if desync {
                    grammar = None;
                }
            }
        }

        // Now that the matcher is aligned, produce the first generated token
        // under the mask by feeding it the prompt token we held back.
        #[cfg(feature = "mlx-native")]
        if hold_back_last {
            let held = *prompt_ids.last().expect("hold_back_last implies non-empty");
            let stepped = {
                let g = grammar
                    .as_mut()
                    .expect("hold_back_last implies an active grammar");
                let mut mask =
                    |buf: &mut [f32]| -> Result<()> { g.apply_mask_to_logits(buf).map(|_| ()) };
                self.runner.decode_step_masked(seq_id, held, &mut mask)
            };
            match stepped {
                Ok((next, new_pos)) => {
                    last = next;
                    pos = new_pos;
                }
                Err(e) => {
                    // The mask itself failed (vocab mismatch, matcher error).
                    // Take the step unmasked so the request still answers.
                    eprintln!(
                        "[qwen35-backend] masked first step failed (dropping grammar): {e:#}"
                    );
                    grammar = None;
                    let (next, new_pos) = self.decode_step(seq_id, held, pos)?;
                    last = next;
                    pos = new_pos;
                }
            }
        }
        if let Some(g) = grammar.as_mut() {
            if g.is_active() {
                if let Err(e) = g.observe(last) {
                    eprintln!(
                        "[qwen35-backend] grammar first-token observe desynced \
                         (dropping grammar, sampling free): {e:#}"
                    );
                    grammar = None;
                }
            }
        }

        // The grammar (WS-C #2) enforces required params structurally and
        // dup-free, superseding the heuristic `force_required` injector (whose
        // `extend`-based forcing bypasses the matcher). When the grammar is
        // active, disable force-injection so the two don't fight. Computed
        // AFTER the prefill replay so a desync-dropped grammar correctly
        // re-enables the injector fallback.
        let grammar_will_constrain = grammar.as_ref().is_some_and(|g| g.is_active());
        let force_active = !force_required.is_empty() && !grammar_will_constrain;

        // Detokenize + parser-feed for the first decoded token.
        let mut generated: Vec<u32> = vec![last];
        let mut emitted_idx: usize = 0;
        if let Ok(text) = self.decode(&generated) {
            if debug_qwen_tools {
                eprintln!("[qwen35-tools] first decoded text: {text:?}");
            }
            if !text.is_empty() && !text.contains('\u{FFFD}') {
                for ev in parser.feed(&text) {
                    forward_parse_event(ev, &mut on_event)?;
                }
                emitted_idx = generated.len();
            }
        }
        if self.eos_tokens.contains(&last) {
            self.remove_seq(seq_id).ok();
            return Ok(parser.finish());
        }

        let t_decode = std::time::Instant::now();
        // WS-B Lever 1 (overlap scheduling) is NOT applied to this Qwen3.6
        // loop — intentionally left synchronous. The Gemma4 sampled path
        // (`gemma4_backend::decode_streaming_with_prompt`) gets the overlap;
        // here the structure does not cleanly admit a "issue next forward
        // before CPU work" deferral:
        //   1. `decode_step` / `decode_step_masked` are a FUSED black box
        //      across the `MlxRunner` trait (Subprocess / PyO3 / Native):
        //      they forward, host-sync, argmax, and return only the *token*
        //      (`(u32, usize)`), not the lazy logits array. Overlapping the
        //      next forward would require returning lazy logits and splitting
        //      sampling out across all four backends — an invasive trait
        //      refactor out of scope here.
        //   2. The grammar-masked path applies its mask BETWEEN forward and
        //      argmax on a host copy, so the chosen token isn't known until
        //      after the host sync — there is nothing to pipeline ahead of it.
        //   3. The force-required-param injection below issues a variable-
        //      length mid-loop `extend`, breaking any fixed one-forward-per-
        //      step pipeline.
        //   4. ~75% of Qwen3.6 layers are linear-attn: `decode_step` advances
        //      conv/SSM recurrent state in place per call, so a speculative
        //      next forward issued before the current token is committed would
        //      have to be rolled back on the injection/grammar-desync paths.
        // Correctness first: Qwen3.6 stays on the exact synchronous path.
        for step in 1..max_new_tokens {
            // Force-required-params injection (opt-in). If the previous feed
            // left the parser at a clean tool-call-body boundary with a
            // REQUIRED param still missing, inject `<parameter=KEY>\n` so the
            // model writes the value instead of closing an empty call
            // (`<function=read></function>` → "path is required"). The value
            // is still model-generated — only the opener is forced.
            if force_active {
                if let Some(key) = parser.next_required_param_to_force(&force_required) {
                    let inj = format!("<parameter={key}>\n");
                    let inj_tokens = self.encode_raw(&inj)?;
                    if !inj_tokens.is_empty() {
                        // Forward the still-unforwarded `last` plus the opener
                        // in one extend; `last` is consumed (already pushed +
                        // fed in the prior step).
                        let mut combined = Vec::with_capacity(1 + inj_tokens.len());
                        combined.push(last);
                        combined.extend_from_slice(&inj_tokens);
                        let (next2, pos2) = self.runner.extend(seq_id, &combined)?;
                        generated.extend_from_slice(&inj_tokens);
                        emitted_idx = generated.len();
                        for ev in parser.feed(&inj) {
                            forward_parse_event(ev, &mut on_event)?;
                        }
                        if debug_qwen_tools {
                            eprintln!("[qwen35-tools] forced required param {key:?}");
                        }
                        // `next2` is produced-but-unforwarded — it becomes the
                        // new `last`; push + feed it to restore the loop's
                        // top-of-iteration invariant, then re-evaluate.
                        last = next2;
                        pos = pos2;
                        generated.push(last);
                        if let Ok(text) = self.decode(&generated[emitted_idx..]) {
                            if !text.is_empty() && !text.contains('\u{FFFD}') {
                                for ev in parser.feed(&text) {
                                    forward_parse_event(ev, &mut on_event)?;
                                }
                                emitted_idx = generated.len();
                            }
                        }
                        if self.eos_tokens.contains(&last) {
                            break;
                        }
                        continue;
                    }
                }
            }
            // Grammar-masked decode when active: same forward (KV + SSM state
            // advance identically), but the host logits are grammar-masked
            // before argmax, then the sampled token is fed back to the matcher.
            // SSM safety: masking happens AFTER the forward on a host copy, so
            // it only changes WHICH token is chosen, never the recurrence; the
            // chosen token is then fed through the next normal forward.
            let grammar_active = grammar.as_ref().is_some_and(|g| g.is_active());
            let (next, new_pos) = if grammar_active {
                #[cfg(feature = "mlx-native")]
                {
                    // Scope the `&mut grammar` borrow to the masked forward so
                    // it ends before the post-step `g.observe` / `grammar =
                    // None` reassignment.
                    let stepped = {
                        let g = grammar.as_mut().expect("grammar_active implies Some");
                        let mut mask = |buf: &mut [f32]| -> Result<()> {
                            g.apply_mask_to_logits(buf).map(|_| ())
                        };
                        self.runner.decode_step_masked(seq_id, last, &mut mask)
                    };
                    let (next, new_pos) = stepped?;
                    // Advance the matcher with the sampled token. A desync here
                    // (shouldn't happen — the token was masked-in) drops the
                    // grammar so the rest of the call samples freely.
                    let observe_err = grammar.as_mut().and_then(|g| g.observe(next).err());
                    if let Some(e) = observe_err {
                        eprintln!(
                            "[qwen35-backend] grammar observe(sampled) desynced \
                             (dropping grammar): {e:#}"
                        );
                        grammar = None;
                    }
                    (next, new_pos)
                }
                #[cfg(not(feature = "mlx-native"))]
                {
                    // Grammar masking requires the native runner; without it,
                    // fall back to an unconstrained step (grammar is never
                    // built on non-native backends, so this is unreachable).
                    self.decode_step(seq_id, last, pos)?
                }
            } else {
                self.decode_step(seq_id, last, pos)?
            };
            last = next;
            pos = new_pos;
            generated.push(next);
            let tail_start = emitted_idx;
            if tail_start < generated.len() {
                if let Ok(text) = self.decode(&generated[tail_start..]) {
                    if !text.is_empty() && !text.contains('\u{FFFD}') {
                        if debug_qwen_tools {
                            eprintln!("[qwen35-tools] step={step} text={text:?}");
                        }
                        for ev in parser.feed(&text) {
                            forward_parse_event(ev, &mut on_event)?;
                        }
                        emitted_idx = generated.len();
                    }
                }
            }
            if self.eos_tokens.contains(&next) {
                eprintln!(
                    "[mlx] seq {seq_id} EOS-tools at step {step} ({:.1} tok/s)",
                    step as f64 / (t_decode.elapsed().as_secs_f64())
                );
                break;
            }
        }
        let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
        let n_gen = generated.len();
        eprintln!(
            "[mlx] seq {seq_id} done-tools: {n_gen} tokens in {decode_ms:.0}ms ({:.1} tok/s)",
            n_gen as f64 / (decode_ms / 1000.0)
        );
        if debug_qwen_tools {
            let full = self.decode(&generated).unwrap_or_default();
            eprintln!(
                "[qwen35-tools] full output ({} tokens):\n---\n{full}\n---",
                n_gen
            );
        }
        self.remove_seq(seq_id).ok();
        Ok(parser.finish())
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

        // ── Conversation-boundary snapshot point (model-agnostic) ──
        // The reusable boundary is the full prompt MINUS the trailing
        // generation header (`<|im_start|>assistant\n…`) — the part the NEXT
        // turn replaces with the assistant's actual response. Snapshotting
        // here lets turn N+1 reuse turn N's WHOLE conversation and prefill
        // only its new tokens (warm-turn cost ≈ new-message length, not the
        // whole growing conversation). If the header doesn't tokenize to a
        // clean suffix, fall back to the old system-message boundary; if even
        // that isn't isolatable, cold-prefill without caching.
        let header_ids = self
            .encode(crate::qwen3_5_tools::qwen3_generation_header(thinking))
            .unwrap_or_default();
        let boundary = if !header_ids.is_empty()
            && prompt_ids.len() > header_ids.len()
            && prompt_ids.ends_with(&header_ids)
        {
            prompt_ids.len() - header_ids.len()
        } else {
            let sys = self.detect_system_prefix_len(messages).unwrap_or(0);
            if sys > 0 && sys < prompt_ids.len() {
                sys
            } else {
                0
            }
        };

        self.prefix_store.evict_stale(&mut self.runner);

        // Boundary snapshot to durably persist to the L2 disk tier, captured in
        // the cache-write branch and flushed AFTER the decode loop so the write
        // never delays token delivery. `(snapshot_id, boundary_prefix_tokens)`.
        let mut persist_after: Option<(u64, Vec<u32>)> = None;

        let (mut last, mut pos);
        if boundary == 0 {
            // No isolatable prefix — plain cold prefill, no cache write.
            let t = std::time::Instant::now();
            let (l, p) = self.runner.prefill(seq_id, &prompt_ids)?;
            last = l;
            pos = p;
            eprintln!(
                "[mlx] prefix-cache MISS key={prefix_cache_key:?} \
                 no isolatable prefix; cold prefill {} tokens in {:.0}ms",
                prompt_ids.len(),
                t.elapsed().as_secs_f64() * 1000.0
            );
        } else {
            // Stage 1: position seq_id's cache exactly at `boundary` — fork a
            // cached prefix and extend up to the boundary when one is a strict
            // prefix, else cold-prefill up to the boundary.
            let cached = match self
                .prefix_store
                .get_master_for(prefix_cache_key, &prompt_ids)
            {
                Some(c) => Some(c),
                // L2 disk tier: in-memory miss → try to rehydrate a persisted
                // boundary snapshot for this key (survives restart / eviction).
                // No-op (None) when the disk tier is off, so this stays
                // byte-identical to the cold-prefill path. On a disk HIT we
                // register it in the in-memory store so subsequent turns in this
                // process hit memory directly.
                None => match self
                    .runner
                    .load_persisted_disk(prefix_cache_key, &prompt_ids)
                {
                    Ok(Some((snap_id, prefix_tokens, last_token))) => {
                        eprintln!(
                            "[mlx] prefix-cache: disk rehydrate key={prefix_cache_key:?} \
                             prefix={}",
                            prefix_tokens.len()
                        );
                        self.prefix_store.store_master(
                            &mut self.runner,
                            prefix_cache_key,
                            snap_id,
                            prefix_tokens.clone(),
                            last_token,
                        );
                        Some((snap_id, prefix_tokens))
                    }
                    Ok(None) => None,
                    Err(e) => {
                        eprintln!(
                            "[mlx] prefix-cache: disk rehydrate failed (key={prefix_cache_key:?}): {e:#}"
                        );
                        None
                    }
                },
            };
            let t = std::time::Instant::now();
            let reused = match cached {
                Some((master, prefix))
                    if !prefix.is_empty()
                        && prefix.len() <= boundary
                        && prompt_ids.starts_with(&prefix) =>
                {
                    let _ = self.runner.fork_from_snapshot(master, seq_id)?;
                    if boundary > prefix.len() {
                        let _ = self
                            .runner
                            .extend(seq_id, &prompt_ids[prefix.len()..boundary])?;
                    }
                    Some(prefix.len())
                }
                _ => {
                    let _ = self.runner.prefill(seq_id, &prompt_ids[..boundary])?;
                    None
                }
            };

            // Snapshot at the conversation boundary and (re)store it for this
            // key so the NEXT turn forks from here. Replaces any prior entry.
            match self.runner.snapshot_state_deep(seq_id) {
                Ok((snap_id, _)) => {
                    // Boundary snapshot — the trailing header is always re-extended
                    // below for fresh logits, so no stored argmax is needed (and an
                    // exact-prefix HIT would re-extend the header rather than reuse
                    // this), hence `last_token = None`.
                    self.prefix_store.store_master(
                        &mut self.runner,
                        prefix_cache_key,
                        snap_id,
                        prompt_ids[..boundary].to_vec(),
                        None,
                    );
                    // L2 disk tier: DEFER the durable persist until after the
                    // decode loop (see `persist_after` use below). Capturing the
                    // boundary snapshot id + tokens here, but writing after
                    // generation, keeps the (synchronous, durable) disk write off
                    // the token-delivery path — TTFT / inter-token latency are
                    // unaffected for streaming. No-op when the disk tier is off.
                    persist_after = Some((snap_id, prompt_ids[..boundary].to_vec()));
                }
                Err(e) => eprintln!(
                    "[mlx] prefix-cache: boundary snapshot failed ({e}); not cached this turn"
                ),
            }

            // Stage 2: extend the trailing generation header, then decode.
            let (l, p) = self.runner.extend(seq_id, &prompt_ids[boundary..])?;
            last = l;
            pos = p;
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            let header_len = prompt_ids.len() - boundary;
            match reused {
                Some(reused_len) => eprintln!(
                    "[mlx] prefix-cache HIT key={prefix_cache_key:?} reused={reused_len} \
                     boundary={boundary} header={header_len} fork+extend={ms:.0}ms"
                ),
                None => eprintln!(
                    "[mlx] prefix-cache MISS key={prefix_cache_key:?} \
                     cold-to-boundary={boundary} header={header_len} prefill={ms:.0}ms"
                ),
            }
        }

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
            self.persist_boundary_snapshot(persist_after.take(), prefix_cache_key);
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
        self.persist_boundary_snapshot(persist_after.take(), prefix_cache_key);
        let out = self.decode(&generated).unwrap_or_default();
        self.remove_seq(seq_id).ok();
        Ok(out)
    }

    /// Flush a deferred boundary snapshot to the L2 disk tier (synchronous,
    /// durable). Called after generation so the disk write never delays token
    /// delivery. No-op when `persist` is `None` or the disk tier is off;
    /// non-fatal on error (the request already succeeded).
    fn persist_boundary_snapshot(&mut self, persist: Option<(u64, Vec<u32>)>, key: &str) {
        if let Some((snap_id, prefix)) = persist {
            if let Err(e) = self
                .runner
                .persist_snapshot_disk(snap_id, key, &prefix, None)
            {
                eprintln!("[mlx] prefix-cache: disk persist skipped (key={key:?}): {e:#}");
            }
        }
        // L2 spill: under memory pressure, drop cold in-memory snapshots (now
        // durably on disk) to free RAM; they rehydrate on next access. Off
        // unless `LUMEN_KV_SPILL_MEM_GB` is set. Gated because the pressure
        // probe reads MLX active memory, which has no meaning — and no
        // symbol — without `mlx-native`.
        #[cfg(feature = "mlx-native")]
        if crate::kv_disk::under_memory_pressure() {
            let keep = crate::kv_disk::spill_keep_floor();
            let before = self.prefix_store.len();
            self.prefix_store.spill(&mut self.runner, keep);
            let after = self.prefix_store.len();
            if after < before {
                eprintln!(
                    "[mlx] prefix-cache: spilled {} cold snapshot(s) to disk (kept {after})",
                    before - after
                );
            }
        }
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

    /// `detect_system_prefix_len` for the structured-history shape. Returns the
    /// token length of the leading `System` turn's rendered block (0 if the
    /// history doesn't start with a non-empty system turn). Used to supply the
    /// Phase 0 incremental-prefix boundary for the tool-history entry points.
    fn detect_system_prefix_len_from_turns(
        &self,
        turns: &[crate::chat_io::ChatTurn<'_>],
    ) -> Result<usize> {
        use crate::chat_io::ChatTurn;
        let content = match turns.first() {
            Some(ChatTurn::System(s)) if !s.is_empty() => *s,
            _ => return Ok(0),
        };
        let block = format_system_prefix(&("system".to_string(), content.to_string()));
        let sys_ids = self.encode(&block)?;
        Ok(sys_ids.len())
    }

    /// Tools-aware variant of [`Self::detect_system_prefix_len`]: returns the
    /// token length of the **system + rendered-tools** head — the stable prefix
    /// every same-system/same-tools request shares, which is what the agentic
    /// chat path actually re-uses across turns and client compactions. The
    /// plain system-only boundary leaves the ~25K-token tool-schema block out
    /// of the snapshot, so it gets cold-prefilled every divergent turn; this
    /// captures it. Mirrors exactly what `format_qwen3_chat_with_tools_*` emits
    /// before the first body turn (`render_tools_system_block`), so
    /// `prompt_ids[..len]` is a strict prefix. Falls back to the system-only
    /// boundary when there are no tools.
    fn detect_system_tools_prefix_len(
        &self,
        messages: &[(String, String)],
        tools: &[crate::chat_io::ToolDef<'_>],
    ) -> Result<usize> {
        if tools.is_empty() {
            return self.detect_system_prefix_len(messages);
        }
        let leading_system = match messages.first() {
            Some((role, text)) if role == "system" && !text.is_empty() => Some(text.as_str()),
            _ => None,
        };
        let block = crate::qwen3_5_tools::render_tools_system_block(tools, leading_system);
        let ids = self.encode(&block)?;
        Ok(ids.len())
    }

    /// `detect_system_tools_prefix_len` for the structured-history shape.
    fn detect_system_tools_prefix_len_from_turns(
        &self,
        turns: &[crate::chat_io::ChatTurn<'_>],
        tools: &[crate::chat_io::ToolDef<'_>],
    ) -> Result<usize> {
        use crate::chat_io::ChatTurn;
        if tools.is_empty() {
            return self.detect_system_prefix_len_from_turns(turns);
        }
        let leading_system = match turns.first() {
            Some(ChatTurn::System(s)) if !s.is_empty() => Some(*s),
            _ => None,
        };
        let block = crate::qwen3_5_tools::render_tools_system_block(tools, leading_system);
        let ids = self.encode(&block)?;
        Ok(ids.len())
    }

    /// Drop a prefix-cache entry by key, releasing its master snapshot.
    /// Returns true if the entry existed.
    pub fn drop_prefix_cache(&mut self, key: &str) -> bool {
        match self.prefix_store.drop_entry(&mut self.runner, key) {
            Some(hits) => {
                eprintln!("[mlx] prefix-cache key={key:?} dropped (hits={hits})");
                true
            }
            None => false,
        }
    }

    /// Drop all prefix-cache entries; returns the number released.
    pub fn clear_prefix_cache(&mut self) -> usize {
        let n = self.prefix_store.clear(&mut self.runner);
        if n > 0 {
            eprintln!("[mlx] prefix-cache cleared ({n} entries)");
        }
        n
    }

    pub fn prefix_cache_count(&self) -> usize {
        self.prefix_store.len()
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

    /// Phase 2 S4 — Qwen3.5 MTP streaming chat. Drop-in replacement for
    /// `chat_streaming` that routes the decode loop through
    /// `qwen35_mtp_step` (Step A-E orchestration) instead of `decode_step`.
    /// Each cycle commits 1 + n_accepted + 1 tokens at once; EOS detection
    /// and `max_new_tokens` clamp apply per-emitted-token within the cycle.
    /// Stats logged: per-cycle wallclock, accept_rate, emit_mean.
    #[cfg(feature = "mlx-native")]
    fn chat_streaming_qwen35_mtp<F>(
        &mut self,
        messages: &[(String, String)],
        max_new_tokens: usize,
        thinking: bool,
        seq_id: u64,
        k: usize,
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
        let (mut last, _pos) = self.prefill(seq_id, &prompt_ids)?;
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[mlx-mtp] seq {seq_id} prefill: {} tokens in {prefill_ms:.0}ms (K={k}) -> tok={last}",
            prompt_ids.len(),
        );

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

        // MTP acceptance regime. Default greedy (temp 0 = exact argmax-match,
        // lossless). `LUMEN_MTP_TEMP>0` switches to Leviathan rejection
        // sampling (accept min(1,p/q) + residual) so accept rate rises toward
        // the sampling regime; `LUMEN_MTP_TOPP` sets the nucleus.
        let mtp_temp: f32 = std::env::var("LUMEN_MTP_TEMP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        let mtp_topp: f32 = std::env::var("LUMEN_MTP_TOPP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);
        let t_decode = std::time::Instant::now();
        let mut cycles: usize = 0;
        let mut accepted_total: usize = 0;
        let mut attempted_total: usize = 0;
        let mut stop = false;
        while generated.len() < max_new_tokens && !stop {
            let t_cycle = std::time::Instant::now();
            let out = self
                .qwen35_mtp_step(seq_id, last, k, mtp_temp, mtp_topp)
                .with_context(|| {
                    format!("[mlx-mtp] seq {seq_id} mtp_step cycle {cycles} failed")
                })?;
            let cycle_ms = t_cycle.elapsed().as_secs_f64() * 1000.0;
            cycles += 1;
            accepted_total += out.n_accepted;
            attempted_total += out.n_attempted;
            // Emit committed tokens one-by-one, honoring EOS + max_new_tokens
            // mid-cycle. We always keep the LAST committed token as `last` for
            // the next mtp_step call (it's the correction/bonus not yet in
            // cache), even if EOS was emitted earlier in the cycle.
            for tok in &out.committed {
                if generated.len() >= max_new_tokens {
                    stop = true;
                    break;
                }
                generated.push(*tok);
                if let Ok(text) = self.decode(&generated) {
                    if text.len() > prev_text.len() && !text.contains('\u{FFFD}') {
                        on_token(&text[prev_text.len()..]);
                        prev_text = text;
                    }
                }
                if self.eos_tokens.contains(tok) {
                    stop = true;
                    break;
                }
            }
            last = *out.committed.last().expect("mtp_step committed non-empty");
            if std::env::var("LUMEN_MTP_DEBUG")
                .map(|v| v == "1")
                .unwrap_or(false)
            {
                eprintln!(
                    "[mlx-mtp-cycle] seq {seq_id} {cycle_ms:.2} ms  n_acc={}/{}  emitted={}",
                    out.n_accepted,
                    out.n_attempted,
                    out.committed.len(),
                );
            }
        }
        let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
        let n_gen = generated.len();
        let accept_rate = if attempted_total > 0 {
            (accepted_total as f64) / (attempted_total as f64)
        } else {
            0.0
        };
        let emit_per_cycle = if cycles > 0 {
            ((n_gen - 1) as f64) / (cycles as f64)
        } else {
            0.0
        };
        eprintln!(
            "[mlx-mtp] seq {seq_id} done: {n_gen} tokens in {decode_ms:.0}ms ({:.1} tok/s); \
             cycles={cycles} accept={accept_rate:.3} emit/cycle={emit_per_cycle:.2}",
            n_gen as f64 / (decode_ms / 1000.0),
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

/// Phase 2: bridges Qwen35ResponseParser events to the generic
/// `BackendStreamEvent` shape the engine consumes. Mirrors the
/// Gemma 4 `emit_token_event` adapter.
fn forward_parse_event<F>(
    ev: crate::qwen3_5_tools::Qwen35ParseEvent,
    on_event: &mut F,
) -> Result<()>
where
    F: FnMut(crate::chat_io::BackendStreamEvent<'_>) -> Result<()>,
{
    use crate::chat_io::BackendStreamEvent;
    use crate::qwen3_5_tools::Qwen35ParseEvent;
    match ev {
        Qwen35ParseEvent::Text(text) => on_event(BackendStreamEvent::Text(&text)),
        Qwen35ParseEvent::ToolCallStart { name } => {
            on_event(BackendStreamEvent::ToolCallStart { name: &name })
        }
    }
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

/// Regression guards for the grammar-routing defects.
///
/// Each test names the behaviour it restores and the symptom that behaviour
/// had before. `tools/red_green.py` re-derives the red state mechanically by
/// reverting the fix and asserting these fail — a claim that "this was broken"
/// is only worth as much as a way to reproduce it.
#[cfg(all(test, feature = "mlx-native"))]
mod grammar_routing_regressions {
    use super::{gemma4_grammar_would_constrain, hold_back_last_prompt_token};
    use crate::chat_io::{ResolvedToolChoice, ToolDef};

    fn one_tool() -> Vec<ToolDef<'static>> {
        vec![ToolDef {
            name: "get_weather",
            description: None,
            parameters: None,
            response: None,
        }]
    }

    // ── Gemma 4 non-streaming ran with no grammar at all ──────────────────
    // RED: a non-streaming `tool_choice=required` returned `迎get_weather`.
    // Only `response_format` was re-routed to the streaming decode, so tool
    // requests kept the batched `generate()` path, which applies no mask.

    #[test]
    fn tools_alone_must_route_through_the_grammar_aware_decode() {
        assert!(
            gemma4_grammar_would_constrain(None, &one_tool(), &ResolvedToolChoice::Auto),
            "a tool request gets a grammar, so it cannot use the unmasked batched path"
        );
        assert!(gemma4_grammar_would_constrain(
            None,
            &one_tool(),
            &ResolvedToolChoice::Required
        ));
    }

    #[test]
    fn requests_that_get_no_grammar_keep_the_fast_path() {
        // The re-route costs a cold prefill, so it must not fire for requests
        // that would not be constrained anyway.
        assert!(
            !gemma4_grammar_would_constrain(None, &[], &ResolvedToolChoice::Auto),
            "no tools, no schema"
        );
        assert!(
            !gemma4_grammar_would_constrain(None, &one_tool(), &ResolvedToolChoice::None),
            "tool_choice=none builds no grammar"
        );
    }

    #[test]
    fn a_schema_routes_even_without_tools() {
        let schema = serde_json::json!({ "type": "object" });
        assert!(gemma4_grammar_would_constrain(
            Some(&schema),
            &[],
            &ResolvedToolChoice::Auto
        ));
    }

    // ── Qwen 3.6: the first generated token escaped the mask ──────────────
    // RED: `tool_choice=required` never enforced anything. Prefill argmaxed
    // its last position unmasked, and the caller responded to the resulting
    // mismatch by dropping the grammar.

    #[test]
    fn an_active_grammar_holds_the_last_prompt_token_back() {
        assert!(
            hold_back_last_prompt_token(true, &[1, 2, 3], None),
            "the first generated token must be produced under the mask"
        );
    }

    #[test]
    fn no_grammar_means_the_prefill_is_untouched() {
        assert!(
            !hold_back_last_prompt_token(false, &[1, 2, 3], None),
            "the unconstrained path must stay byte-identical"
        );
    }

    #[test]
    fn an_image_placeholder_is_never_held_back() {
        // Splitting a placeholder off the prompt tail would tear a run out of
        // the span the images are spliced over.
        assert!(
            !hold_back_last_prompt_token(true, &[1, 2, 77], Some(77)),
            "the tail is an image placeholder"
        );
        assert!(
            hold_back_last_prompt_token(true, &[77, 2, 3], Some(77)),
            "a placeholder elsewhere in the prompt is fine"
        );
    }

    #[test]
    fn a_single_token_prompt_is_left_alone() {
        assert!(
            !hold_back_last_prompt_token(true, &[1], None),
            "there is nothing to prefill if the only token is held back"
        );
    }
}

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
        let backend = MlxBackend::load("mlx-community/Qwen3.6-35B-A3B-mxfp4").unwrap();
        // Unwrap to the Qwen35 inner for the low-level prefill / decode_step
        // round-trip (the unified MlxBackend façade only exposes the
        // high-level `chat` / `generate` methods).
        let MlxBackend::Qwen35Family(mut b) = backend else {
            panic!("expected Qwen35Family for this smoke test");
        };
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
            // A wide, uniform gap: this fake never produces a near-tie, so a
            // consumer that treats small gaps specially takes its ordinary path.
            let row_top2_gap = vec![1.0_f32; tokens.len()];
            Ok(ProbeRows {
                row_argmaxes,
                row_max_abs,
                row_top2_gap,
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

    #[test]
    fn decode_step_batch_default_loops_per_seq() {
        // The default `decode_step_batch` must equal N independent `decode_step`
        // calls. ScriptedRunner::decode_step returns (last_token+1, position+1),
        // so a 3-seq batch maps each (last, pos) -> (last+1, pos+1) in order.
        let mut r = ScriptedRunner::default();
        let got = r
            .decode_step_batch(&[1, 2, 3], &[10, 20, 30], &[5, 6, 7])
            .unwrap();
        assert_eq!(got, vec![(11, 6), (21, 7), (31, 8)]);

        // Equivalence: the batch result equals calling decode_step per seq.
        let mut r2 = ScriptedRunner::default();
        let seq_ids = [7u64, 8, 9];
        let last = [100u32, 200, 300];
        let pos = [1usize, 2, 3];
        let per_seq: Vec<(u32, usize)> = seq_ids
            .iter()
            .zip(last.iter())
            .zip(pos.iter())
            .map(|((&s, &l), &p)| r2.decode_step(s, l, p).unwrap())
            .collect();
        let mut r3 = ScriptedRunner::default();
        let batched = r3.decode_step_batch(&seq_ids, &last, &pos).unwrap();
        assert_eq!(batched, per_seq);
    }

    #[test]
    fn decode_step_batch_rejects_mismatched_lengths() {
        let mut r = ScriptedRunner::default();
        assert!(
            r.decode_step_batch(&[1, 2], &[10], &[5, 6]).is_err(),
            "mismatched token length must error"
        );
        assert!(
            r.decode_step_batch(&[1, 2], &[10, 20], &[5]).is_err(),
            "mismatched position length must error"
        );
        // Empty batch is a valid no-op.
        assert_eq!(r.decode_step_batch(&[], &[], &[]).unwrap(), vec![]);
    }
}
