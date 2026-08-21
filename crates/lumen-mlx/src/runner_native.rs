use anyhow::{Result, anyhow};

use crate::{LoadInfo, ProbeRows};

#[cfg(feature = "mlx-native")]
mod imp {
    use super::*;
    use anyhow::Context;
    use hf_hub::api::sync::ApiBuilder;
    use mlx_rs::Array;
    use serde::Deserialize;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    #[cfg(test)]
    const MLX_SMOKE_CHILD_MODE_ENV: &str = "TQ_MLX_SMOKE_CHILD_MODE";
    #[cfg(test)]
    const MLX_SMOKE_METAL_AVAILABLE_MODE: &str = "metal_available";
    #[cfg(test)]
    const MLX_SMOKE_METADATA_MODE: &str = "metadata";
    #[cfg(test)]
    const MLX_SMOKE_RAW_ARRAY_EMPTY_MODE: &str = "raw_array_empty";
    #[cfg(test)]
    const MLX_SMOKE_RAW_ARRAY_NEW_INT_MODE: &str = "raw_array_new_int";
    #[cfg(test)]
    const MLX_SMOKE_RAW_ARRAY_NEW_BOOL_MODE: &str = "raw_array_new_bool";
    #[cfg(test)]
    const MLX_SMOKE_RAW_ARRAY_NEW_FLOAT32_MODE: &str = "raw_array_new_float32";
    #[cfg(test)]
    const MLX_SMOKE_RAW_ARRAY_NEW_FLOAT64_MODE: &str = "raw_array_new_float64";
    #[cfg(test)]
    const MLX_SMOKE_RAW_ARRAY_NEW_DATA_INT_MODE: &str = "raw_array_new_data_int";
    #[cfg(test)]
    const MLX_SMOKE_RAW_ARRAY_NEW_DATA_EMPTY_INT_MODE: &str = "raw_array_new_data_empty_int";
    #[cfg(test)]
    const MLX_SMOKE_SCALAR_ITEM_MODE: &str = "scalar_item";
    #[cfg(test)]
    const MLX_SMOKE_ARGMAX_EVAL_MODE: &str = "argmax_eval";
    #[cfg(test)]
    const MLX_SMOKE_SAFETENSORS_MODE: &str = "safetensors";

    #[cfg(test)]
    fn log_mlx_smoke_phase(mode: &str, phase: &str) -> Result<()> {
        use std::io::{self, Write};

        let mut stderr = io::stderr().lock();
        writeln!(stderr, "native mlx-rs smoke [{mode}]: {phase}")
            .map_err(|err| anyhow!("failed to write native mlx-rs smoke marker: {err}"))?;
        stderr
            .flush()
            .map_err(|err| anyhow!("failed to flush native mlx-rs smoke marker: {err}"))?;
        Ok(())
    }

    #[cfg(test)]
    unsafe extern "C" fn log_mlx_error_handler(
        msg: *const std::os::raw::c_char,
        _data: *mut std::os::raw::c_void,
    ) {
        use std::ffi::CStr;
        use std::io::{self, Write};

        let message = if msg.is_null() {
            "<null mlx error>"
        } else {
            unsafe { CStr::from_ptr(msg) }
                .to_str()
                .unwrap_or("<non-utf8 mlx error>")
        };

        let _ = writeln!(
            io::stderr().lock(),
            "native mlx-rs smoke [mlx_error]: {message}"
        );
    }

    #[cfg(test)]
    fn install_mlx_smoke_error_handler() {
        unsafe {
            mlx_sys::mlx_set_error_handler(Some(log_mlx_error_handler), std::ptr::null_mut(), None);
        }
    }

    #[cfg(test)]
    fn mlx_metal_available_smoke() -> Result<()> {
        log_mlx_smoke_phase(
            MLX_SMOKE_METAL_AVAILABLE_MODE,
            "before mlx_metal_is_available",
        )?;
        let mut is_available = false;
        let status = unsafe { mlx_sys::mlx_metal_is_available(&mut is_available as *mut bool) };
        log_mlx_smoke_phase(
            MLX_SMOKE_METAL_AVAILABLE_MODE,
            "after mlx_metal_is_available",
        )?;
        if status != 0 {
            return Err(anyhow!(
                "native mlx-sys metal availability smoke returned status {status}"
            ));
        }
        log_mlx_smoke_phase(
            MLX_SMOKE_METAL_AVAILABLE_MODE,
            if is_available {
                "metal available"
            } else {
                "metal unavailable"
            },
        )?;

        Ok(())
    }

    use crate::kv_disk::{DiskKvStore, KvManifest};
    use crate::native_cache::{NativeKvCache, NativePromptCache, SharedPrefixKv};
    use crate::native_runtime::{FineTimings, take_fine_timings};
    use crate::native_snapshot::PromptCacheSnapshot;
    use crate::qwen3_5_moe::{MtpStepOutput, NativeQwen3_5MoeModel};
    use crate::qwen3_5_mtp::Qwen35MtpBlock;
    use std::collections::HashMap;

    /// Per-sequence runtime state held by the native runner.
    struct NativeSeqState {
        cache: NativePromptCache,
        position: usize,
        /// MTP block's own KV cache (one full-attention layer; see
        /// `qwen3_5_mtp::Qwen35MtpBlock`). Cleared at the start of every
        /// `mtp_step` call (fresh K-step extrapolation). Idle / unused when
        /// MTP is not enabled.
        mtp_kv: NativeKvCache,
        /// 1-forward MTP carry (`LUMEN_MTP_ONE_FORWARD`): the trunk hidden that
        /// predecessor-seeds the next cycle's drafter. `None` at seq creation
        /// (forces a bootstrap Step-A on the first `mtp_step`); thereafter holds
        /// `verify_hidden[n_accepted]` from the prior cycle. Idle on the
        /// 2-forward path (which always returns `None`).
        mtp_carried: Option<mlx_rs::Array>,
        /// Prompt token ids this seq was prefilled with. Used by the batched
        /// scheduler's shared-prefix dedup (Phase 4) to compute the longest
        /// common prompt prefix across the active batch. Empty for seqs created
        /// via snapshot fork (those don't participate in dedup).
        prompt: Vec<u32>,
        /// MRoPE position minus token count for this seq's prompt — nonzero
        /// only when an image shortened the position run. Decode adds it to
        /// `cache.offset()` so the fused rope keeps counting from the right
        /// place. `0` on every text-only seq, which is the byte-identical path.
        mrope_delta: i32,
    }

    /// process-wide background worker that runs
    /// `mlx_clear_cache()` off the request thread. The clear takes ~45 ms
    /// for long-prompt workloads (per `notes/native_stream_timing_concluded.md`)
    /// and currently blocks the post-DONE flush, adding ~3 % to total request
    /// latency. Deferring it to a background thread lets the HTTP layer flush
    /// the trailing `[DONE]` frame immediately and only pays the clear cost
    /// in idle time before the next request. The worker coalesces bursts
    /// (drain extras before running clear) so back-to-back requests don't
    /// queue up multiple redundant clears.
    ///
    /// Lifecycle: `OnceLock` lazy init on first request_clear; the spawned
    /// thread lives for the process lifetime and exits when the channel is
    /// dropped (which only happens at process teardown).
    struct CacheCleanerWorker {
        tx: std::sync::mpsc::Sender<()>,
    }

    impl CacheCleanerWorker {
        fn global() -> &'static Self {
            static WORKER: std::sync::OnceLock<CacheCleanerWorker> = std::sync::OnceLock::new();
            WORKER.get_or_init(|| {
                let (tx, rx) = std::sync::mpsc::channel::<()>();
                std::thread::Builder::new()
                    .name("mlx-cache-cleaner".into())
                    .spawn(move || {
                        while rx.recv().is_ok() {
                            // Coalesce: if multiple clears were requested
                            // back-to-back, only run one (clear is idempotent
                            // on a settled cache pool).
                            while rx.try_recv().is_ok() {}
                            unsafe {
                                mlx_sys::mlx_clear_cache();
                            }
                        }
                    })
                    .expect("spawn mlx-cache-cleaner thread");
                CacheCleanerWorker { tx }
            })
        }

        fn request_clear(&self) {
            // Send is fire-and-forget. Channel error means the worker thread
            // is gone (only at process teardown); silently ignore to keep
            // remove_seq infallible.
            let _ = self.tx.send(());
        }
    }

    /// Phase 4 dedup switch. An explicit `LUMEN_MLX_SHARED_PREFIX` wins;
    /// otherwise it inherits from `LUMEN_MLX_SERVER_MODE` (the bundled
    /// multi-request serving / bulk-batch switch — dedup pays off most in bulk
    /// jobs where many requests share one system prompt). Default OFF (solo
    /// desktop path unchanged). Cached on first read; truthy = `1`/`true`/`on`/`yes`.
    fn shared_prefix_enabled() -> bool {
        fn truthy(v: &str) -> bool {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "on" | "yes"
            )
        }
        static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *C.get_or_init(|| match std::env::var("LUMEN_MLX_SHARED_PREFIX") {
            Ok(v) => truthy(&v),
            Err(_) => std::env::var("LUMEN_MLX_SERVER_MODE")
                .map(|v| truthy(&v))
                .unwrap_or(false),
        })
    }

    /// Minimum shared prefix length (tokens) worth deduping
    /// (`LUMEN_MLX_SHARED_PREFIX_MIN`, default 64). Short shared prefixes save
    /// little RAM but still pay the per-step `sdpa_split` merge overhead.
    fn shared_prefix_min_tokens() -> usize {
        std::env::var("LUMEN_MLX_SHARED_PREFIX_MIN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(64)
    }

    /// Longest common prefix length across a set of token sequences.
    fn longest_common_prefix_len(seqs: &[&[u32]]) -> usize {
        let Some(min_len) = seqs.iter().map(|s| s.len()).min() else {
            return 0;
        };
        if seqs.is_empty() {
            return 0;
        }
        let first = seqs[0];
        for i in 0..min_len {
            let c = first[i];
            if seqs.iter().any(|s| s[i] != c) {
                return i;
            }
        }
        min_len
    }

    pub(crate) struct NativeMlxRunner {
        model: Option<NativeQwen3_5MoeModel>,
        seqs: HashMap<u64, NativeSeqState>,
        /// `Some` when `LUMEN_NATIVE_TIMING=1`. Each entry is
        /// `(forward_ms, tail_ms)` for one `decode_step` call. Adds an extra
        /// `eval()` after `forward()` to split the two halves cleanly, so the
        /// totals here are slightly higher than the un-instrumented path.
        timing_log: Option<Vec<(f64, f64)>>,
        /// `Some` when `LUMEN_NATIVE_TIMING=1`. Drained from the per-thread
        /// `take_fine_timings()` slot populated inside `forward()`. Aligned
        /// 1:1 with `timing_log` entries (same flag drives both).
        fine_timing_log: Option<Vec<FineTimings>>,
        /// Track A1 / A2 — captured prompt-cache states. Mirrors
        /// `mlx_runner.py::MlxRunner._snapshots`. Shallow snapshots are
        /// 1-shot (consumed by `restore_state`); deep snapshots are reusable
        /// across multiple `fork_from_snapshot` calls until explicitly
        /// released.
        snapshots: HashMap<u64, PromptCacheSnapshot>,
        next_snapshot_id: u64,
        /// Phase 4 in-batch shared-prefix dedup. `Some` once the batched
        /// scheduler has detected a common prompt prefix across ≥2 active seqs
        /// and truncated their full-attn caches to suffix-only; holds the single
        /// shared prefix KV those seqs attend against. `None` when dedup is off
        /// (`LUMEN_MLX_SHARED_PREFIX` unset) or no common prefix has formed.
        shared_prefix: Option<SharedPrefixKv>,
        /// L2 disk persistence tier (opt-in via `LUMEN_KV_DISK`). `Some` once a
        /// model is loaded with the disk tier enabled — prefix-cache boundary
        /// snapshots are serialized here so they survive process restart /
        /// in-memory eviction (avoids re-paying cold prefill). `None` keeps the
        /// path byte-identical to the in-memory-only behaviour.
        disk: Option<DiskKvStore>,
        /// Model identity used to namespace `disk` (sanitized `model_id`). A
        /// mismatch means a persisted snapshot was made by a different model and
        /// is discarded on load.
        fingerprint: String,
    }

    #[allow(dead_code)]
    #[derive(Debug)]
    struct NativeModelAssets {
        root: PathBuf,
        config_json: PathBuf,
        tokenizer_json: Option<PathBuf>,
        safetensors: Vec<PathBuf>,
    }

    #[allow(dead_code)]
    impl NativeModelAssets {
        fn discover_local(model_id: &str) -> Result<Self> {
            let root = PathBuf::from(model_id);
            if !root.is_dir() {
                return Err(anyhow!(
                    "native mlx-rs loader currently accepts local model directories only: {model_id}"
                ));
            }

            let config_json = root.join("config.json");
            if !config_json.is_file() {
                return Err(anyhow!(
                    "native mlx-rs loader could not find config.json in {}",
                    root.display()
                ));
            }

            let tokenizer_json = root.join("tokenizer.json");
            let tokenizer_json = tokenizer_json.is_file().then_some(tokenizer_json);

            let mut safetensors = std::fs::read_dir(&root)
                .map_err(|err| {
                    anyhow!(
                        "native mlx-rs loader could not list model directory {}: {err}",
                        root.display()
                    )
                })?
                .map(|entry| {
                    let entry = entry.map_err(|err| {
                        anyhow!(
                            "native mlx-rs loader could not read model directory entry in {}: {err}",
                            root.display()
                        )
                    })?;
                    Ok(entry.path())
                })
                .collect::<Result<Vec<_>>>()?;

            safetensors.retain(|path| {
                path.is_file()
                    && path
                        .extension()
                        .is_some_and(|extension| extension == "safetensors")
            });
            safetensors.sort();

            if safetensors.is_empty() {
                return Err(anyhow!(
                    "native mlx-rs loader could not find safetensors weights in {}",
                    root.display()
                ));
            }

            Ok(Self {
                root,
                config_json,
                tokenizer_json,
                safetensors,
            })
        }
    }

    #[allow(dead_code)]
    #[derive(Debug, Clone, Deserialize)]
    struct NativeModelConfig {
        model_type: String,
        #[serde(
            default,
            rename = "eos_token_id",
            deserialize_with = "deserialize_token_ids"
        )]
        eos_token_ids: Vec<u32>,
        text_config: Option<NativeTextConfig>,
        quantization_config: Option<NativeQuantizationConfig>,
    }

    #[allow(dead_code)]
    impl NativeModelConfig {
        fn load(path: &Path) -> Result<Self> {
            let config = std::fs::read_to_string(path).map_err(|err| {
                anyhow!(
                    "native mlx-rs loader could not read config {}: {err}",
                    path.display()
                )
            })?;
            serde_json::from_str(&config).map_err(|err| {
                anyhow!(
                    "native mlx-rs loader could not parse config {}: {err}",
                    path.display()
                )
            })
        }

        fn validate_qwen3_5_moe_contract(&self) -> Result<&NativeTextConfig> {
            if self.model_type != "qwen3_5_moe" {
                return Err(anyhow!(
                    "native mlx-rs loader expected model_type `qwen3_5_moe`, got `{}`",
                    self.model_type
                ));
            }

            let text = self.text_config.as_ref().ok_or_else(|| {
                anyhow!("native mlx-rs loader expected `text_config` for qwen3_5_moe config")
            })?;
            text.validate_qwen3_5_moe_text_contract()?;

            if let Some(quantization) = &self.quantization_config {
                quantization.validate_mxfp4_default()?;
            }

            Ok(text)
        }
    }

    #[allow(dead_code)]
    #[derive(Debug, Clone, Deserialize)]
    struct NativeTextConfig {
        model_type: String,
        hidden_size: usize,
        head_dim: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        num_hidden_layers: usize,
        #[serde(default)]
        layer_types: Vec<NativeLayerType>,
        vocab_size: usize,
    }

    #[allow(dead_code)]
    impl NativeTextConfig {
        fn validate_qwen3_5_moe_text_contract(&self) -> Result<()> {
            if self.model_type != "qwen3_5_moe_text" {
                return Err(anyhow!(
                    "native mlx-rs loader expected text_config.model_type `qwen3_5_moe_text`, got `{}`",
                    self.model_type
                ));
            }
            if self.hidden_size == 0 || self.head_dim == 0 || self.vocab_size == 0 {
                return Err(anyhow!(
                    "native mlx-rs loader requires non-zero hidden_size, head_dim, and vocab_size"
                ));
            }
            if self.num_attention_heads == 0
                || self.num_key_value_heads == 0
                || self.num_hidden_layers == 0
            {
                return Err(anyhow!(
                    "native mlx-rs loader requires non-zero attention heads, KV heads, and layers"
                ));
            }
            if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
                return Err(anyhow!(
                    "native mlx-rs loader expected hidden_size to be divisible by num_attention_heads, got {} % {}",
                    self.hidden_size,
                    self.num_attention_heads
                ));
            }
            if self.layer_types.len() != self.num_hidden_layers {
                return Err(anyhow!(
                    "native mlx-rs loader expected layer_types length {} to equal num_hidden_layers {}",
                    self.layer_types.len(),
                    self.num_hidden_layers
                ));
            }

            Ok(())
        }
    }

    #[allow(dead_code)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
    #[serde(rename_all = "snake_case")]
    enum NativeLayerType {
        LinearAttention,
        FullAttention,
    }

    #[allow(dead_code)]
    #[derive(Debug, Clone, Deserialize)]
    struct NativeQuantizationConfig {
        group_size: usize,
        bits: usize,
        mode: String,
        #[serde(flatten)]
        overrides: std::collections::BTreeMap<String, NativeQuantizationOverride>,
    }

    #[allow(dead_code)]
    impl NativeQuantizationConfig {
        fn validate_mxfp4_default(&self) -> Result<()> {
            if self.mode != "mxfp4" || self.bits != 4 || self.group_size == 0 {
                return Err(anyhow!(
                    "native mlx-rs loader expected default MXFP4 quantization with non-zero group_size, got mode=`{}` bits={} group_size={}",
                    self.mode,
                    self.bits,
                    self.group_size
                ));
            }
            if let Some((name, override_config)) =
                self.overrides.iter().find(|(_, override_config)| {
                    override_config.group_size == 0 || override_config.bits == 0
                })
            {
                return Err(anyhow!(
                    "native mlx-rs loader expected non-zero quantization override for `{name}`, got bits={} group_size={}",
                    override_config.bits,
                    override_config.group_size
                ));
            }
            Ok(())
        }
    }

    #[allow(dead_code)]
    #[derive(Debug, Clone, Deserialize)]
    struct NativeQuantizationOverride {
        group_size: usize,
        bits: usize,
    }

    #[allow(dead_code)]
    #[derive(Debug, Deserialize)]
    #[serde(untagged)]
    enum TokenIdValue {
        One(u32),
        Many(Vec<u32>),
    }

    fn deserialize_token_ids<'de, D>(deserializer: D) -> std::result::Result<Vec<u32>, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Option::<TokenIdValue>::deserialize(deserializer)?;
        Ok(match value {
            Some(TokenIdValue::One(id)) => vec![id],
            Some(TokenIdValue::Many(ids)) => ids,
            None => Vec::new(),
        })
    }

    #[allow(dead_code)]
    struct NativeSafetensorWeights {
        tensors: std::collections::HashMap<String, Array>,
    }

    #[allow(dead_code)]
    impl NativeSafetensorWeights {
        fn load_file(path: &Path) -> Result<Self> {
            let tensors = Array::load_safetensors(path).map_err(|err| {
                anyhow!(
                    "native mlx-rs loader could not load safetensors {}: {err}",
                    path.display()
                )
            })?;
            Ok(Self { tensors })
        }

        fn get_required(&self, name: &str) -> Result<&Array> {
            self.tensors.get(name).ok_or_else(|| {
                anyhow!("native mlx-rs loader missing required tensor `{name}` in safetensors map")
            })
        }
    }

    #[cfg(test)]
    fn mlx_raw_array_empty_smoke() -> Result<()> {
        install_mlx_smoke_error_handler();
        log_mlx_smoke_phase(
            MLX_SMOKE_RAW_ARRAY_EMPTY_MODE,
            "installed mlx error handler",
        )?;
        log_mlx_smoke_phase(MLX_SMOKE_RAW_ARRAY_EMPTY_MODE, "before mlx_array_new")?;
        let value = unsafe { mlx_sys::mlx_array_new() };
        log_mlx_smoke_phase(MLX_SMOKE_RAW_ARRAY_EMPTY_MODE, "after mlx_array_new")?;
        if value.ctx.is_null() {
            log_mlx_smoke_phase(MLX_SMOKE_RAW_ARRAY_EMPTY_MODE, "empty handle")?;
            return Ok(());
        }

        let status = unsafe { mlx_sys::mlx_array_free(value) };
        log_mlx_smoke_phase(MLX_SMOKE_RAW_ARRAY_EMPTY_MODE, "after mlx_array_free")?;
        if status != 0 {
            return Err(anyhow!(
                "native mlx-sys empty array free returned status {status}"
            ));
        }

        Ok(())
    }

    #[cfg(test)]
    fn mlx_raw_array_new_int_smoke() -> Result<()> {
        install_mlx_smoke_error_handler();
        log_mlx_smoke_phase(
            MLX_SMOKE_RAW_ARRAY_NEW_INT_MODE,
            "installed mlx error handler",
        )?;
        log_mlx_smoke_phase(MLX_SMOKE_RAW_ARRAY_NEW_INT_MODE, "before mlx_array_new_int")?;
        let value = unsafe { mlx_sys::mlx_array_new_int(7) };
        log_mlx_smoke_phase(MLX_SMOKE_RAW_ARRAY_NEW_INT_MODE, "after mlx_array_new_int")?;
        if value.ctx.is_null() {
            return Err(anyhow!(
                "native mlx-sys raw array constructor returned an empty handle"
            ));
        }

        let status = unsafe { mlx_sys::mlx_array_free(value) };
        log_mlx_smoke_phase(MLX_SMOKE_RAW_ARRAY_NEW_INT_MODE, "after mlx_array_free")?;
        if status != 0 {
            return Err(anyhow!(
                "native mlx-sys raw array free returned status {status}"
            ));
        }

        Ok(())
    }

    #[cfg(test)]
    fn mlx_raw_array_new_bool_smoke() -> Result<()> {
        install_mlx_smoke_error_handler();
        log_mlx_smoke_phase(
            MLX_SMOKE_RAW_ARRAY_NEW_BOOL_MODE,
            "installed mlx error handler",
        )?;
        log_mlx_smoke_phase(
            MLX_SMOKE_RAW_ARRAY_NEW_BOOL_MODE,
            "before mlx_array_new_bool",
        )?;
        let value = unsafe { mlx_sys::mlx_array_new_bool(true) };
        log_mlx_smoke_phase(
            MLX_SMOKE_RAW_ARRAY_NEW_BOOL_MODE,
            "after mlx_array_new_bool",
        )?;
        if value.ctx.is_null() {
            return Err(anyhow!(
                "native mlx-sys raw bool array constructor returned an empty handle"
            ));
        }

        let status = unsafe { mlx_sys::mlx_array_free(value) };
        log_mlx_smoke_phase(MLX_SMOKE_RAW_ARRAY_NEW_BOOL_MODE, "after mlx_array_free")?;
        if status != 0 {
            return Err(anyhow!(
                "native mlx-sys raw bool array free returned status {status}"
            ));
        }

        Ok(())
    }

    #[cfg(test)]
    fn mlx_raw_array_new_float32_smoke() -> Result<()> {
        install_mlx_smoke_error_handler();
        log_mlx_smoke_phase(
            MLX_SMOKE_RAW_ARRAY_NEW_FLOAT32_MODE,
            "installed mlx error handler",
        )?;
        log_mlx_smoke_phase(
            MLX_SMOKE_RAW_ARRAY_NEW_FLOAT32_MODE,
            "before mlx_array_new_float32",
        )?;
        let value = unsafe { mlx_sys::mlx_array_new_float32(7.0) };
        log_mlx_smoke_phase(
            MLX_SMOKE_RAW_ARRAY_NEW_FLOAT32_MODE,
            "after mlx_array_new_float32",
        )?;
        if value.ctx.is_null() {
            return Err(anyhow!(
                "native mlx-sys raw float32 array constructor returned an empty handle"
            ));
        }

        let status = unsafe { mlx_sys::mlx_array_free(value) };
        log_mlx_smoke_phase(MLX_SMOKE_RAW_ARRAY_NEW_FLOAT32_MODE, "after mlx_array_free")?;
        if status != 0 {
            return Err(anyhow!(
                "native mlx-sys raw float32 array free returned status {status}"
            ));
        }

        Ok(())
    }

    #[cfg(test)]
    fn mlx_raw_array_new_float64_smoke() -> Result<()> {
        install_mlx_smoke_error_handler();
        log_mlx_smoke_phase(
            MLX_SMOKE_RAW_ARRAY_NEW_FLOAT64_MODE,
            "installed mlx error handler",
        )?;
        log_mlx_smoke_phase(
            MLX_SMOKE_RAW_ARRAY_NEW_FLOAT64_MODE,
            "before mlx_array_new_float64",
        )?;
        let value = unsafe { mlx_sys::mlx_array_new_float64(7.0) };
        log_mlx_smoke_phase(
            MLX_SMOKE_RAW_ARRAY_NEW_FLOAT64_MODE,
            "after mlx_array_new_float64",
        )?;
        if value.ctx.is_null() {
            return Err(anyhow!(
                "native mlx-sys raw float64 array constructor returned an empty handle"
            ));
        }

        let status = unsafe { mlx_sys::mlx_array_free(value) };
        log_mlx_smoke_phase(MLX_SMOKE_RAW_ARRAY_NEW_FLOAT64_MODE, "after mlx_array_free")?;
        if status != 0 {
            return Err(anyhow!(
                "native mlx-sys raw float64 array free returned status {status}"
            ));
        }

        Ok(())
    }

    #[cfg(test)]
    fn mlx_raw_array_new_data_int_smoke() -> Result<()> {
        install_mlx_smoke_error_handler();
        log_mlx_smoke_phase(
            MLX_SMOKE_RAW_ARRAY_NEW_DATA_INT_MODE,
            "installed mlx error handler",
        )?;
        let data = [7_i32];
        let shape = [1_i32];
        log_mlx_smoke_phase(
            MLX_SMOKE_RAW_ARRAY_NEW_DATA_INT_MODE,
            "before mlx_array_new_data",
        )?;
        let value = unsafe {
            mlx_sys::mlx_array_new_data(
                data.as_ptr().cast(),
                shape.as_ptr(),
                shape.len() as i32,
                mlx_sys::mlx_dtype__MLX_INT32,
            )
        };
        log_mlx_smoke_phase(
            MLX_SMOKE_RAW_ARRAY_NEW_DATA_INT_MODE,
            "after mlx_array_new_data",
        )?;
        if value.ctx.is_null() {
            return Err(anyhow!(
                "native mlx-sys raw int data array constructor returned an empty handle"
            ));
        }

        let status = unsafe { mlx_sys::mlx_array_free(value) };
        log_mlx_smoke_phase(
            MLX_SMOKE_RAW_ARRAY_NEW_DATA_INT_MODE,
            "after mlx_array_free",
        )?;
        if status != 0 {
            return Err(anyhow!(
                "native mlx-sys raw int data array free returned status {status}"
            ));
        }

        Ok(())
    }

    #[cfg(test)]
    fn mlx_raw_array_new_data_empty_int_smoke() -> Result<()> {
        install_mlx_smoke_error_handler();
        log_mlx_smoke_phase(
            MLX_SMOKE_RAW_ARRAY_NEW_DATA_EMPTY_INT_MODE,
            "installed mlx error handler",
        )?;
        let data: [i32; 0] = [];
        let shape = [0_i32];
        log_mlx_smoke_phase(
            MLX_SMOKE_RAW_ARRAY_NEW_DATA_EMPTY_INT_MODE,
            "before mlx_array_new_data",
        )?;
        let value = unsafe {
            mlx_sys::mlx_array_new_data(
                data.as_ptr().cast(),
                shape.as_ptr(),
                shape.len() as i32,
                mlx_sys::mlx_dtype__MLX_INT32,
            )
        };
        log_mlx_smoke_phase(
            MLX_SMOKE_RAW_ARRAY_NEW_DATA_EMPTY_INT_MODE,
            "after mlx_array_new_data",
        )?;
        if value.ctx.is_null() {
            return Err(anyhow!(
                "native mlx-sys raw empty int data array constructor returned an empty handle"
            ));
        }

        let status = unsafe { mlx_sys::mlx_array_free(value) };
        log_mlx_smoke_phase(
            MLX_SMOKE_RAW_ARRAY_NEW_DATA_EMPTY_INT_MODE,
            "after mlx_array_free",
        )?;
        if status != 0 {
            return Err(anyhow!(
                "native mlx-sys raw empty int data array free returned status {status}"
            ));
        }

        Ok(())
    }

    #[cfg(test)]
    fn mlx_tensor_metadata_smoke() -> Result<()> {
        log_mlx_smoke_phase(MLX_SMOKE_METADATA_MODE, "before Array::from_int")?;
        let value = Array::from_int(7);
        log_mlx_smoke_phase(MLX_SMOKE_METADATA_MODE, "after Array::from_int")?;

        log_mlx_smoke_phase(MLX_SMOKE_METADATA_MODE, "before ndim")?;
        if value.ndim() != 0 {
            return Err(anyhow!(
                "native mlx-rs tensor metadata smoke returned ndim {}, expected scalar ndim 0",
                value.ndim()
            ));
        }
        log_mlx_smoke_phase(MLX_SMOKE_METADATA_MODE, "after ndim")?;
        log_mlx_smoke_phase(MLX_SMOKE_METADATA_MODE, "before size")?;
        if value.size() != 1 {
            return Err(anyhow!(
                "native mlx-rs tensor metadata smoke returned size {}, expected 1",
                value.size()
            ));
        }
        log_mlx_smoke_phase(MLX_SMOKE_METADATA_MODE, "after size")?;
        log_mlx_smoke_phase(MLX_SMOKE_METADATA_MODE, "before shape")?;
        if !value.shape().is_empty() {
            return Err(anyhow!(
                "native mlx-rs tensor metadata smoke returned shape {:?}, expected scalar shape []",
                value.shape()
            ));
        }
        log_mlx_smoke_phase(MLX_SMOKE_METADATA_MODE, "after shape")?;

        Ok(())
    }

    #[cfg(test)]
    fn mlx_tensor_scalar_item_smoke() -> Result<()> {
        log_mlx_smoke_phase(MLX_SMOKE_SCALAR_ITEM_MODE, "before Array::from_int")?;
        let value = Array::from_int(7);
        log_mlx_smoke_phase(MLX_SMOKE_SCALAR_ITEM_MODE, "after Array::from_int")?;
        log_mlx_smoke_phase(MLX_SMOKE_SCALAR_ITEM_MODE, "before try_item")?;
        let observed = value.try_item::<i32>().map_err(|err| {
            anyhow!("native mlx-rs tensor smoke failed during scalar read: {err}")
        })?;
        log_mlx_smoke_phase(MLX_SMOKE_SCALAR_ITEM_MODE, "after try_item")?;
        if observed != 7 {
            return Err(anyhow!(
                "native mlx-rs tensor smoke returned scalar {observed}, expected 7"
            ));
        }

        Ok(())
    }

    #[cfg(test)]
    fn mlx_tensor_argmax_eval_smoke() -> Result<()> {
        use mlx_rs::ops::indexing;

        log_mlx_smoke_phase(MLX_SMOKE_ARGMAX_EVAL_MODE, "before Array::from_slice")?;
        let values = Array::from_slice(&[0.25_f32, 3.5, -1.0, 2.0], &[4]);
        log_mlx_smoke_phase(MLX_SMOKE_ARGMAX_EVAL_MODE, "after Array::from_slice")?;

        log_mlx_smoke_phase(MLX_SMOKE_ARGMAX_EVAL_MODE, "before argmax")?;
        let index = indexing::argmax(&values, false)?;
        log_mlx_smoke_phase(MLX_SMOKE_ARGMAX_EVAL_MODE, "after argmax")?;

        log_mlx_smoke_phase(MLX_SMOKE_ARGMAX_EVAL_MODE, "before eval")?;
        index
            .eval()
            .map_err(|err| anyhow!("native mlx-rs argmax smoke failed during eval: {err}"))?;
        log_mlx_smoke_phase(MLX_SMOKE_ARGMAX_EVAL_MODE, "after eval")?;

        log_mlx_smoke_phase(MLX_SMOKE_ARGMAX_EVAL_MODE, "before try_item")?;
        let observed = index.try_item::<u32>().map_err(|err| {
            anyhow!("native mlx-rs argmax smoke failed during scalar read: {err}")
        })?;
        log_mlx_smoke_phase(MLX_SMOKE_ARGMAX_EVAL_MODE, "after try_item")?;
        if observed != 1 {
            return Err(anyhow!(
                "native mlx-rs argmax smoke returned index {observed}, expected 1"
            ));
        }

        Ok(())
    }

    #[cfg(test)]
    fn mlx_tensor_safetensors_smoke() -> Result<()> {
        use mlx_rs::ops::indexing;
        use std::collections::HashMap;

        log_mlx_smoke_phase(MLX_SMOKE_SAFETENSORS_MODE, "before Array::from_slice")?;
        let values = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
        log_mlx_smoke_phase(MLX_SMOKE_SAFETENSORS_MODE, "after Array::from_slice")?;

        let path = std::env::temp_dir().join(format!(
            "lumen_mlx_native_smoke_{}.safetensors",
            std::process::id()
        ));

        let mut arrays = HashMap::new();
        arrays.insert("weights".to_string(), values);

        log_mlx_smoke_phase(MLX_SMOKE_SAFETENSORS_MODE, "before save_safetensors")?;
        Array::save_safetensors(&arrays, None, &path)
            .map_err(|err| anyhow!("native mlx-rs safetensors smoke failed during save: {err}"))?;
        log_mlx_smoke_phase(MLX_SMOKE_SAFETENSORS_MODE, "after save_safetensors")?;

        log_mlx_smoke_phase(MLX_SMOKE_SAFETENSORS_MODE, "before load_safetensors")?;
        let loaded = NativeSafetensorWeights::load_file(&path)?;
        log_mlx_smoke_phase(MLX_SMOKE_SAFETENSORS_MODE, "after load_safetensors")?;

        let _ = std::fs::remove_file(&path);

        let loaded_values = loaded.get_required("weights")?;
        if loaded_values.shape() != [2, 2] {
            return Err(anyhow!(
                "native mlx-rs safetensors smoke loaded shape {:?}, expected [2, 2]",
                loaded_values.shape()
            ));
        }

        log_mlx_smoke_phase(MLX_SMOKE_SAFETENSORS_MODE, "before argmax")?;
        let index = indexing::argmax(loaded_values, false)?;
        log_mlx_smoke_phase(MLX_SMOKE_SAFETENSORS_MODE, "after argmax")?;

        log_mlx_smoke_phase(MLX_SMOKE_SAFETENSORS_MODE, "before eval")?;
        index
            .eval()
            .map_err(|err| anyhow!("native mlx-rs safetensors smoke failed during eval: {err}"))?;
        log_mlx_smoke_phase(MLX_SMOKE_SAFETENSORS_MODE, "after eval")?;

        log_mlx_smoke_phase(MLX_SMOKE_SAFETENSORS_MODE, "before try_item")?;
        let observed = index.try_item::<u32>().map_err(|err| {
            anyhow!("native mlx-rs safetensors smoke failed during scalar read: {err}")
        })?;
        log_mlx_smoke_phase(MLX_SMOKE_SAFETENSORS_MODE, "after try_item")?;
        if observed != 3 {
            return Err(anyhow!(
                "native mlx-rs safetensors smoke returned argmax {observed}, expected 3"
            ));
        }

        Ok(())
    }

    /// Resolve `model_id` to a local directory containing `config.json` plus
    /// `*.safetensors` shards. If `model_id` already names a local directory,
    /// returns it as-is. Otherwise treats `model_id` as a HuggingFace Hub repo
    /// id (e.g. `mlx-community/Qwen3.6-35B-A3B-mxfp4`) and downloads
    /// `config.json` + every `*.safetensors` sibling into the HF cache, then
    /// returns the snapshot directory path.
    ///
    /// Network failures, gated repos, or repos without safetensors shards
    /// surface as a typed error pointing the user back to either a local
    /// directory or `LUMEN_MLX_BACKEND=pyo3`.
    pub(crate) fn resolve_model_dir(model_id: &str) -> Result<PathBuf> {
        if model_id.is_empty() {
            return Err(anyhow!(
                "native mlx-rs runner: model_id is empty (pass a local directory or HF Hub repo id)"
            ));
        }
        let local = PathBuf::from(model_id);
        if local.is_dir() {
            return Ok(local);
        }

        let api = ApiBuilder::new()
            .build()
            .context("native mlx-rs runner: hf_hub api init")?;
        let repo = api.model(model_id.to_string());

        let info = repo.info().with_context(|| {
            format!(
                "native mlx-rs runner: list HF Hub files for `{model_id}` (set LUMEN_MLX_BACKEND=pyo3 or pass a local model directory if offline)"
            )
        })?;

        let safetensors: Vec<String> = info
            .siblings
            .iter()
            .map(|s| s.rfilename.clone())
            .filter(|name| name.ends_with(".safetensors"))
            .collect();
        if safetensors.is_empty() {
            return Err(anyhow!(
                "native mlx-rs runner: HF Hub repo `{model_id}` exposes no .safetensors shards (the native loader needs MXFP4 safetensors weights)"
            ));
        }

        let config_path = repo.get("config.json").with_context(|| {
            format!("native mlx-rs runner: download config.json from `{model_id}`")
        })?;
        let snapshot_dir = config_path
            .parent()
            .ok_or_else(|| {
                anyhow!(
                    "native mlx-rs runner: hf_hub returned config.json without a parent directory: {}",
                    config_path.display()
                )
            })?
            .to_path_buf();

        for fname in &safetensors {
            repo.get(fname).with_context(|| {
                format!("native mlx-rs runner: download `{fname}` from `{model_id}`")
            })?;
        }

        // Optional companions; ignore failures so offline-friendly partial
        // snapshots still work.
        let _ = repo.get("model.safetensors.index.json");
        let _ = repo.get("tokenizer.json");

        Ok(snapshot_dir)
    }

    /// Chunk size (in tokens) for the long-prompt prefill/extend loop.
    /// `LUMEN_QWEN35_PREFILL_CHUNK` overrides; default 2048 mirrors the
    /// Gemma 4 backend. A zero / unparseable value falls back to the default.
    /// NOTE: `forward_chunked` clamps this DOWN to keep one chunk's full-attn
    /// scores under a byte budget, so a value larger than the prompt no longer
    /// forces single-pass on its own — for the legacy single-pass A/B path also
    /// raise `LUMEN_QWEN35_PREFILL_SCORES_GB` to a huge value.
    fn qwen35_prefill_chunk() -> usize {
        std::env::var("LUMEN_QWEN35_PREFILL_CHUNK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(2048)
    }

    /// Run `tokens` through the trunk in query-windowed chunks, accumulating
    /// KV / linear-attn recurrent state in `cache`, and return the FINAL
    /// chunk's logits (`[1, 1, vocab]` when `last_only`).
    ///
    /// Why chunk: a single `forward` over a very long prompt (agentic system
    /// prompts can be 20k+ tokens) makes the per-position attention/MoE
    /// intermediates scale with the full sequence length, overflowing Metal's
    /// single-buffer limit on unified-memory Macs (observed: a ~130 GB
    /// allocation attempt for a 21k-token prompt → `argmax_last_token`
    /// scalar-read failure). Bounding the query window to `chunk` caps those
    /// intermediates regardless of total prompt length.
    ///
    /// Correctness — chunking is equivalent to single-pass here because:
    ///  * RoPE positions and the causal-sentinel SDPA both key off
    ///    `cache.offset()`, which advances by each chunk's length, so chunk N's
    ///    queries land at the right absolute positions and attend bottom-right
    ///    across the full accumulated KV. Full-attn caches never rotate, so the
    ///    sentinel's `kv_actual == kv_offset + l` invariant (what makes it
    ///    valid for q_len &lt; k_len) always holds.
    ///  * The linear-attn layers carry their conv/SSM recurrent state through
    ///    the cache (`state_in` ← `cache.get(1)`, `cache.set(1, new_state)`),
    ///    exactly as decode's incremental recurrence does — a gated-delta scan
    ///    over a chunk starting from the prior chunk's state is identical to
    ///    scanning the concatenation.
    ///
    /// Each chunk is `eval`'d before the next so MLX materializes the chunk's
    /// KV/state and frees its transient graph; skipping the per-chunk eval
    /// would collapse the loop into one giant lazy graph evaluated at once —
    /// i.e. the exact OOM we are avoiding.
    /// Argmax over a host `f32` logit buffer (already grammar-masked).
    /// Returns the index of the max, or `None` for an empty buffer. NaN
    /// entries never win (`>` is false for NaN), and `-inf` masked positions
    /// are correctly never selected unless every position is `-inf` — in which
    /// case the first index is returned (a degenerate grammar, surfaced as a
    /// valid-but-arbitrary token rather than a panic).
    fn argmax_host_f32(buf: &[f32]) -> Option<u32> {
        let mut best_idx: usize = 0;
        let mut best_val = f32::NEG_INFINITY;
        let mut seen = false;
        for (i, &v) in buf.iter().enumerate() {
            if !seen || v > best_val {
                best_idx = i;
                best_val = v;
                seen = true;
            }
        }
        seen.then_some(best_idx as u32)
    }

    /// `vision` carries an image-bearing prompt's extras, all addressed
    /// against the **whole** prompt: per-token MRoPE positions, the placeholder
    /// runs, and each image's encoded block. Each chunk gets the slice of
    /// positions it covers and splices whichever image rows fall inside it.
    /// `None` is the text path, unchanged.
    #[allow(clippy::type_complexity)]
    fn forward_chunked(
        model: &NativeQwen3_5MoeModel,
        tokens: &[u32],
        cache: &mut NativePromptCache,
        last_only: bool,
        label: &str,
        vision: Option<(&[[i32; 3]], &[(usize, usize)], &[Array])>,
    ) -> Result<Array> {
        let requested_chunk = qwen35_prefill_chunk();
        let n = tokens.len();
        // ── Always-chunk invariant: bound full-attn scores per chunk ──
        // Qwen3.x is hybrid: linear-attn layers are O(seq) (cheap), but the
        // full-attention layers (every `full_attention_interval`) form a
        // [heads, q_len, kv_len] scores buffer just like Gemma. A single
        // un-chunked pass over a long prompt makes kv_len == q_len == prompt
        // length, and that buffer can exceed Metal's per-buffer cap (observed:
        // a ~130 GB allocation attempt for a 21k-token single pass). Clamp the
        // chunk DOWN so one chunk's full-attn scores stay under a byte budget
        // given the worst-case kv_len (= prompt length). Short prompts are
        // unaffected (the cap is large); an over-large
        // `LUMEN_QWEN35_PREFILL_CHUNK` is clamped, never honored — chunking is
        // mandatory. The legacy single-pass A/B path now also requires raising
        // `LUMEN_QWEN35_PREFILL_SCORES_GB` (e.g. a huge value) alongside the
        // chunk override.
        let chunk = {
            // The arithmetic lives in `prefill_budget` so it can be swept at
            // tier 0 (it is this crate's `malloc`-fail-at-N analogue: MLX
            // allocation cannot fail into an `Err`, so the guard is to not
            // allocate). Behaviour here is unchanged by the hoist.
            let budget =
                crate::prefill_budget::scores_budget_from_env("LUMEN_QWEN35_PREFILL_SCORES_GB");
            let d = crate::prefill_budget::clamp_chunk(
                requested_chunk,
                budget,
                model.config().text_config.num_attention_heads,
                n,
            );
            if d.clamped() {
                eprintln!(
                    "[prefill] qwen chunk clamped {requested_chunk} → {} \
                     (heads={} kv_upper={} budget={:.1}GB) — \
                     keeps single-chunk full-attn scores under the Metal buffer cap",
                    d.chunk,
                    d.heads.max(1),
                    d.kv_upper.max(1),
                    budget as f64 / 1e9
                );
            }
            d.chunk
        };
        if n <= chunk {
            return match vision {
                Some((positions, runs, soft)) => model
                    .forward_chunk_with_images(tokens, cache, last_only, positions, 0, runs, soft),
                None => model.forward_with_opts(tokens, cache, last_only),
            }
            .with_context(|| format!("{label}: forward ({n} tokens)"));
        }
        let n_chunks = n.div_ceil(chunk);
        let report = std::env::var("LUMEN_QWEN35_PREFILL_CHUNK_LOG")
            .map(|v| v == "1")
            .unwrap_or(false);
        let mut logits_opt: Option<Array> = None;
        let mut start = 0usize;
        let mut idx = 0usize;
        while start < n {
            let end = (start + chunk).min(n);
            let is_last = end == n;
            // Intermediate chunks only need to grow the cache; force `last_only`
            // so their lm_head stays `[1, 1, vocab]` (cheap) — their logits are
            // discarded. The final chunk honours the caller's `last_only`.
            let lo = if is_last { last_only } else { true };
            let t0 = report.then(std::time::Instant::now);
            let out = match vision {
                Some((positions, runs, soft)) => model.forward_chunk_with_images(
                    &tokens[start..end],
                    cache,
                    lo,
                    &positions[start..end],
                    start,
                    runs,
                    soft,
                ),
                None => model.forward_with_opts(&tokens[start..end], cache, lo),
            }
            .with_context(|| {
                format!(
                    "{label}: chunk {}/{} ({} tokens)",
                    idx + 1,
                    n_chunks,
                    end - start
                )
            })?;
            // CRITICAL: force evaluation so the chunk's transient graph is freed
            // before the next chunk (see fn doc). Bit-identical to single-pass.
            out.eval()
                .with_context(|| format!("{label}: chunk {}/{} eval", idx + 1, n_chunks))?;
            if let Some(t0) = t0 {
                let active = crate::metal_memory::get_active_memory().unwrap_or(0);
                let peak = crate::metal_memory::get_peak_memory().unwrap_or(0);
                eprintln!(
                    "[{label}] chunk {}/{} ({} tok) {:.0}ms  mlx-mem active={:.1}GB peak={:.1}GB",
                    idx + 1,
                    n_chunks,
                    end - start,
                    t0.elapsed().as_secs_f64() * 1000.0,
                    active as f64 / 1e9,
                    peak as f64 / 1e9,
                );
            }
            if is_last {
                logits_opt = Some(out);
            }
            start = end;
            idx += 1;
        }
        logits_opt.ok_or_else(|| anyhow!("{label}: chunked forward produced no logits"))
    }

    impl NativeMlxRunner {
        pub(crate) fn new() -> Result<Self> {
            // Through the registry, not `env::var`. This site used to read the
            // variable directly, which meant two things at once: it ignored the
            // flag's `with()` / `set()` overrides — so a harness enabling
            // timing registry-side got it everywhere except here — and it kept
            // the OLD parse rule (`1|true|TRUE|yes`) after the registry moved
            // to "any non-`0`". `LUMEN_NATIVE_TIMING=on` was true for the flag
            // and false for this line.
            let timing_enabled = crate::native_runtime::fine_timing_active();
            // Lightweight stage timing (forward+eval / tail) without the
            // layer-kind fine probes that force intermediate evals. Used for
            // apples-to-apples Native vs PyO3 stage comparison.
            let stage_only = std::env::var("LUMEN_NATIVE_STAGE_TIMING")
                .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes"))
                .unwrap_or(false);
            if timing_enabled {
                eprintln!(
                    "[mlx-native] timing instrumentation ENABLED (forward/tail + layer-kind breakdown)"
                );
                eprintln!(
                    "[mlx-native] ⚠️  LUMEN_NATIVE_TIMING inserts per-layer GPU sync barriers — \
                     profiling ONLY, cuts throughput ~8× (~70→~8 tok/s). Unset for normal serving."
                );
            } else if stage_only {
                eprintln!(
                    "[mlx-native] stage timing ENABLED (forward+eval/tail only; no layer probes)"
                );
            }
            let any_timing = timing_enabled || stage_only;
            Ok(Self {
                model: None,
                seqs: HashMap::new(),
                timing_log: if any_timing { Some(Vec::new()) } else { None },
                fine_timing_log: if timing_enabled {
                    Some(Vec::new())
                } else {
                    None
                },
                snapshots: HashMap::new(),
                next_snapshot_id: 1,
                shared_prefix: None,
                disk: None,
                fingerprint: String::new(),
            })
        }

        pub(crate) fn take_decode_timing_log(&mut self) -> Option<Vec<(f64, f64)>> {
            self.timing_log.as_mut().map(std::mem::take)
        }

        pub(crate) fn take_decode_fine_timing_log(&mut self) -> Option<Vec<FineTimings>> {
            self.fine_timing_log.as_mut().map(std::mem::take)
        }

        fn require_model(&self) -> Result<&NativeQwen3_5MoeModel> {
            self.model
                .as_ref()
                .ok_or_else(|| anyhow!("native mlx-rs runner: model not loaded"))
        }

        pub(crate) fn load(&mut self, model_id: &str) -> Result<LoadInfo> {
            let resolved = resolve_model_dir(model_id)?;
            let model = NativeQwen3_5MoeModel::load(&resolved)
                .with_context(|| format!("native mlx-rs runner: load `{}`", resolved.display()))?;
            let info = LoadInfo {
                eos_tokens: model.eos_tokens().to_vec(),
                vocab_size: model.vocab_size(),
            };
            self.model = Some(model);
            self.seqs.clear();
            self.snapshots.clear();
            self.next_snapshot_id = 1;
            self.init_disk_tier(model_id);
            Ok(info)
        }

        /// Construct the opt-in L2 disk persistence tier from env. Off unless
        /// `LUMEN_KV_DISK` is truthy. `LUMEN_KV_DISK_MAX_GB` bounds the on-disk
        /// LRU budget (0 / unset = unbounded). `fingerprint` namespaces the
        /// cache directory by model identity so a different model can't load a
        /// stale snapshot. Failures degrade to "no disk tier" — never fatal.
        fn init_disk_tier(&mut self, model_id: &str) {
            self.fingerprint = model_id
                .chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                        c
                    } else {
                        '_'
                    }
                })
                .collect();
            let enabled = matches!(
                std::env::var("LUMEN_KV_DISK")
                    .ok()
                    .map(|s| s.trim().to_ascii_lowercase())
                    .as_deref(),
                Some("1" | "true" | "yes" | "on")
            );
            if !enabled {
                self.disk = None;
                return;
            }
            let max_bytes = std::env::var("LUMEN_KV_DISK_MAX_GB")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .map(|gb| gb * 1024 * 1024 * 1024)
                .unwrap_or(0);
            // Last-access TTL, default 1 day; 0 disables expiry.
            let ttl_secs = std::env::var("LUMEN_KV_DISK_TTL_SECS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(86_400);
            match DiskKvStore::open(
                DiskKvStore::default_root(),
                &self.fingerprint,
                max_bytes,
                ttl_secs,
            ) {
                Ok(store) => {
                    eprintln!(
                        "[mlx] kv-disk tier ON: fingerprint={:?} entries={} max_gb={}",
                        self.fingerprint,
                        store.len(),
                        max_bytes / (1024 * 1024 * 1024),
                    );
                    self.disk = Some(store);
                }
                Err(e) => {
                    eprintln!("[mlx] kv-disk tier disabled (open failed): {e:#}");
                    self.disk = None;
                }
            }
        }

        pub(crate) fn prefill(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)> {
            self.prefill_with_images(seq_id, tokens, &[])
        }

        /// Decode + resize one image onto the tower's patch grid.
        pub(crate) fn prepare_image(
            &self,
            encoded: &[u8],
        ) -> Result<crate::qwen36_vision::PreparedImage> {
            self.require_model()?.prepare_image(encoded)
        }

        /// Token id whose embedding rows the vision features replace.
        pub(crate) fn image_token_id(&self) -> Option<u32> {
            self.model.as_ref().and_then(|m| m.image_token_id())
        }

        /// The loaded tower's config, for header-only prompt sizing.
        pub(crate) fn vision_config(
            &self,
        ) -> Option<crate::qwen36_vision::NativeQwen36VisionConfig> {
            self.model.as_ref().and_then(|m| m.vision_config())
        }

        /// [`Self::prefill`] with encoded images spliced over their
        /// placeholder runs.
        ///
        /// `images` are already decoded and resized; this encodes each once
        /// before the chunk loop, derives the prompt's MRoPE positions, and
        /// records the position/token-count delta so decode can keep using the
        /// fused rope. An empty slice is byte-identical to [`Self::prefill`].
        pub(crate) fn prefill_with_images(
            &mut self,
            seq_id: u64,
            tokens: &[u32],
            images: &[crate::qwen36_vision::PreparedImage],
        ) -> Result<(u32, usize)> {
            if tokens.is_empty() {
                return Err(anyhow!("native mlx-rs runner: prefill given empty tokens"));
            }
            if self.seqs.contains_key(&seq_id) {
                return Err(anyhow!(
                    "native mlx-rs runner: seq_id {seq_id} already initialized"
                ));
            }
            let model = self.require_model()?;
            let mut cache = model.make_cache();
            let timing_on = self.fine_timing_log.is_some();
            // LUMEN_NATIVE_LM_HEAD_FULL=1 disables the last-only slice
            // optimization for prefill (forces lm_head to compute over all
            // L positions, matching the pre-patch path). Used for A/B
            // bench. Default: optimization ON.
            let last_only = std::env::var("LUMEN_NATIVE_LM_HEAD_FULL").is_err();
            let t_start = timing_on.then(Instant::now);

            // Encode every image up front: the placeholder runs are
            // prompt-global, so any chunk may need any of them.
            let vision = if images.is_empty() {
                None
            } else {
                let image_token = model
                    .image_token_id()
                    .ok_or_else(|| anyhow!("config.json has no image_token_id"))?;
                let merge = model
                    .vision_merge_size()
                    .ok_or_else(|| anyhow!("vision tower not loaded"))?;
                let grids: Vec<(usize, usize)> =
                    images.iter().map(|p| p.merged_grid(merge)).collect();
                let positions = crate::qwen36_vision::mrope_positions(tokens, image_token, &grids)
                    .map_err(|e| anyhow!("native mlx-rs runner: mrope positions: {e}"))?;
                let (runs, soft) = model.encode_images_for_prompt(tokens, images)?;
                Some((positions, runs, soft))
            };
            // Positions run shorter than tokens once an image is present; the
            // difference is what decode has to add to `cache.offset()`.
            let mrope_delta = vision
                .as_ref()
                .map(|(pos, _, _)| crate::qwen36_vision::next_position(pos) - tokens.len() as i32)
                .unwrap_or(0);
            let vision_ref = vision
                .as_ref()
                .map(|(p, r, s)| (p.as_slice(), r.as_slice(), s.as_slice()));

            let logits = forward_chunked(
                model,
                tokens,
                &mut cache,
                last_only,
                "native-prefill",
                vision_ref,
            )
            .with_context(|| format!("native mlx-rs runner: prefill forward (seq_id={seq_id})"))?;
            if timing_on {
                logits
                    .eval()
                    .context("native mlx-rs runner: prefill timing eval failed")?;
            }
            let next_tok = model.argmax_last_token(&logits)?;
            if let (Some(t0), Some(fine)) = (t_start, take_fine_timings()) {
                let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let prompt_len = tokens.len();
                let throughput = if total_ms > 0.0 {
                    (prompt_len as f64) / (total_ms / 1000.0)
                } else {
                    0.0
                };
                eprintln!(
                    "[native-prefill] seq_id={seq_id} prompt_len={prompt_len} total_ms={total_ms:.2} throughput={throughput:.1}tok/s embed_ms={:.2} full_attn_ms={:.2}/{} linear_attn_ms={:.2}/{} moe_ms={:.2}/{} lm_head_ms={:.2}",
                    fine.embed_ms,
                    fine.full_attn_ms,
                    fine.full_attn_count,
                    fine.linear_attn_ms,
                    fine.linear_attn_count,
                    fine.moe_ms,
                    fine.moe_count,
                    fine.lm_head_ms,
                );
                eprintln!(
                    "[native-prefill]   linear_sub: in_proj={:.2} conv={:.2} norm_qk={:.2} ssm={:.2} out_proj={:.2}",
                    fine.linear_in_proj_ms,
                    fine.linear_conv_ms,
                    fine.linear_norm_qk_ms,
                    fine.linear_ssm_ms,
                    fine.linear_out_proj_ms,
                );
                eprintln!(
                    "[native-prefill]   moe_sub: routing={:.2} switch_glu={:.2} shared_expert={:.2}",
                    fine.moe_routing_ms, fine.moe_switch_glu_ms, fine.moe_shared_expert_ms,
                );
            }
            let position = tokens.len();
            self.seqs.insert(
                seq_id,
                NativeSeqState {
                    cache,
                    position,
                    mtp_kv: NativeKvCache::new(),
                    mtp_carried: None,
                    prompt: tokens.to_vec(),
                    mrope_delta,
                },
            );
            Ok((next_tok, position))
        }

        /// DFlash prefill variant: full-prompt forward that ALSO captures the
        /// post-MLP residual at each layer in `capture_layer_ids`. Returns
        /// `(next_token, position, captured_hiddens)` where each captured
        /// `Array` has shape `[1, prompt_len, hidden]` matching the layer's
        /// running residual. Mirrors `mlx_runner.py::dflash_prefill` —
        /// the captured hiddens are concatenated along the channel axis (in
        /// the order of `capture_layer_ids`) to form the draft model's
        /// `target_hidden` cross-attn ctx for the first block.
        ///
        /// Differs from baseline `prefill` in:
        ///  * `last_only = false` so lm_head sees the full row trace (some
        ///    callers want full logits; for next-token-only use the
        ///    returned `next_token` directly).
        ///  * Captures `hidden_states` snapshots inside the layer loop.
        ///
        /// Cache state advances by `tokens.len()` exactly as baseline prefill.
        pub(crate) fn prefill_with_capture(
            &mut self,
            seq_id: u64,
            tokens: &[u32],
            capture_layer_ids: &[usize],
        ) -> Result<(u32, usize, Vec<mlx_rs::Array>)> {
            if tokens.is_empty() {
                return Err(anyhow!(
                    "native mlx-rs runner: prefill_with_capture given empty tokens"
                ));
            }
            if self.seqs.contains_key(&seq_id) {
                return Err(anyhow!(
                    "native mlx-rs runner: prefill_with_capture seq_id {seq_id} already exists"
                ));
            }
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| anyhow!("native mlx-rs runner: model not loaded"))?;
            let n_layers = model.num_layers();
            for &id in capture_layer_ids {
                if id >= n_layers {
                    return Err(anyhow!(
                        "native mlx-rs runner: capture_layer_id {id} >= num_layers {n_layers}"
                    ));
                }
            }
            let mut cache = model.make_cache();
            // last_only=false: DFlash dflash_prefill keeps the full-row lm_head
            // (matches `self.model(prompt[None], cache=target_cache)` in
            // mlx_runner.py). The next-token-only optimization used in baseline
            // prefill would clip the captured trace's tail-row in lm_head, but
            // the captures themselves live earlier in the layer loop and are
            // unaffected. Keeping full lm_head here for parity with the
            // Python reference path.
            let (logits, captured) = model
                .forward_with_capture(
                    tokens,
                    &mut cache,
                    /* last_only */ false,
                    capture_layer_ids,
                )
                .with_context(|| {
                    format!("native mlx-rs runner: prefill_with_capture forward (seq_id={seq_id})")
                })?;
            let next_tok = model.argmax_last_token(&logits)?;
            let position = tokens.len();
            self.seqs.insert(
                seq_id,
                NativeSeqState {
                    cache,
                    position,
                    mtp_kv: NativeKvCache::new(),
                    mtp_carried: None,
                    prompt: tokens.to_vec(),
                    mrope_delta: 0,
                },
            );
            Ok((next_tok, position, captured))
        }

        pub(crate) fn decode_step(
            &mut self,
            seq_id: u64,
            last_token: u32,
            _position: usize,
        ) -> Result<(u32, usize)> {
            // G6-G op-count instrumentation. When `LUMEN_NATIVE_COUNT_OPS=1`,
            // reset the mlx-rs `Guarded::try_from_op` counter at the start of
            // each decode_step and report the delta after `argmax_last_token`
            // sync. This directly measures the per-step mlx-c entry call
            // count on the native path — the decisive test for the residual
            // native vs PyO3 gap. See CHECKLIST §G6-G.
            //
            // `LUMEN_NATIVE_COUNT_OPS_BREAKDOWN=1` additionally collects
            // per-callsite histogram (Mutex-guarded — only enable for short
            // bench runs, adds non-trivial per-op overhead).
            let count_ops = std::env::var("LUMEN_NATIVE_COUNT_OPS")
                .map(|v| v == "1")
                .unwrap_or(false);
            let count_breakdown = std::env::var("LUMEN_NATIVE_COUNT_OPS_BREAKDOWN")
                .map(|v| v == "1")
                .unwrap_or(false);
            // G6-G+ wallclock timing per callsite. Wraps each FFI call with
            // Instant::now() deltas — adds ~30-50 ns/op overhead so off by
            // default. Use to separate cheap-per-call vs expensive-per-call
            // levers (op count alone can't distinguish).
            let time_breakdown = std::env::var("LUMEN_NATIVE_TIME_OPS_BREAKDOWN")
                .map(|v| v == "1")
                .unwrap_or(false);
            if count_ops {
                mlx_rs::utils::reset_op_counter();
                if count_breakdown {
                    mlx_rs::utils::enable_op_breakdown();
                }
                if time_breakdown {
                    mlx_rs::utils::enable_op_timing();
                }
            }
            // Split-borrow: take &model and &mut state from disjoint fields
            // so the borrow checker sees them as independent.
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| anyhow!("native mlx-rs runner: model not loaded"))?;
            let state = self.seqs.get_mut(&seq_id).ok_or_else(|| {
                anyhow!("native mlx-rs runner: decode_step on unknown seq_id {seq_id}")
            })?;
            let next_tok = if self.timing_log.is_some() {
                let t0 = Instant::now();
                let logits = model
                    .forward_decode_shifted(
                        &[last_token],
                        &mut state.cache,
                        /* last_only */ true,
                        state.mrope_delta,
                    )
                    .with_context(|| {
                        format!("native mlx-rs runner: decode forward (seq_id={seq_id})")
                    })?;
                logits
                    .eval()
                    .context("native mlx-rs runner: timing eval(forward) failed")?;
                let t1 = Instant::now();
                let next_tok = model.argmax_last_token(&logits)?;
                let t2 = Instant::now();
                let forward_ms = t1.duration_since(t0).as_secs_f64() * 1000.0;
                let tail_ms = t2.duration_since(t1).as_secs_f64() * 1000.0;
                if let Some(log) = self.timing_log.as_mut() {
                    log.push((forward_ms, tail_ms));
                }
                if let (Some(fine_log), Some(fine)) =
                    (self.fine_timing_log.as_mut(), take_fine_timings())
                {
                    fine_log.push(fine);
                }
                next_tok
            } else {
                let logits = model
                    .forward_decode_shifted(
                        &[last_token],
                        &mut state.cache,
                        /* last_only */ true,
                        state.mrope_delta,
                    )
                    .with_context(|| {
                        format!("native mlx-rs runner: decode forward (seq_id={seq_id})")
                    })?;
                model.argmax_last_token(&logits)?
            };
            if count_ops {
                let n = mlx_rs::utils::read_op_counter();
                eprintln!("[G6-G] decode_step ops={n}");
                if count_breakdown {
                    let breakdown = mlx_rs::utils::take_op_breakdown();
                    let total: usize = breakdown.iter().map(|(_, c)| c).sum();
                    eprintln!("[G6-G-BREAKDOWN] total={total} top-25 callsites:");
                    for (loc, count) in breakdown.iter().take(25) {
                        let pct = (*count as f64) / (total as f64) * 100.0;
                        eprintln!("  {count:>6}  ({pct:>5.1}%)  {loc}");
                    }
                }
                if time_breakdown {
                    let timing = mlx_rs::utils::take_op_timing();
                    let total_ns: u128 = timing.iter().map(|(_, _, t)| *t).sum();
                    let total_ms = (total_ns as f64) / 1_000_000.0;
                    let n_unique = timing.len();
                    eprintln!(
                        "[G6-G-TIMING] aggregate_ffi_wall={total_ms:.3}ms unique_callsites={n_unique} top-50:"
                    );
                    for (loc, count, ns) in timing.iter().take(50) {
                        let total_us = (*ns as f64) / 1_000.0;
                        let avg_ns = if *count > 0 {
                            (*ns as f64) / (*count as f64)
                        } else {
                            0.0
                        };
                        let pct = if total_ns > 0 {
                            (*ns as f64) / (total_ns as f64) * 100.0
                        } else {
                            0.0
                        };
                        eprintln!(
                            "  {total_us:>9.2}us ({pct:>5.1}%) n={count:>5} avg={avg_ns:>7.0}ns  {loc}"
                        );
                    }
                }
            }
            state.position += 1;
            Ok((next_tok, state.position))
        }

        /// Phase 1b: batched decode of N sequences in ONE forward. N==1 delegates
        /// to [`Self::decode_step`] (byte-identical single-seq path). N>=2 borrows
        /// the N per-seq caches disjointly out of `self.seqs` and runs
        /// `forward_decode_batch` (batched trunk/MoE/lm_head + batched Mamba via
        /// state-stacking + per-seq full-attn). Each seq's `position` is bumped by
        /// one. Result is aligned to `seq_ids` order.
        /// Phase 4 in-batch shared-prefix dedup. When `LUMEN_MLX_SHARED_PREFIX`
        /// is set and ≥2 batched seqs share a prompt prefix of at least
        /// `LUMEN_MLX_SHARED_PREFIX_MIN` tokens (default 64), capture that prefix
        /// ONCE from a donor seq and truncate every matching seq's full-attn
        /// cache to suffix-only — so the shared prefix KV is stored a single
        /// time instead of once per sequence. Idempotent and incremental: a
        /// later-joining seq that matches the established prefix is attached;
        /// the shared buffer is dropped once no active seq references it. No-op
        /// (returns immediately) when the flag is off, leaving every cache full
        /// and the decode path byte-identical to the pre-dedup behavior.
        fn maybe_setup_shared_prefix(&mut self, seq_ids: &[u64]) -> Result<()> {
            if !shared_prefix_enabled() {
                return Ok(());
            }
            // Drop a stale shared prefix once the group that used it is gone.
            if let Some(sp) = &self.shared_prefix {
                let p = sp.prefix_len;
                let any_attached = self.seqs.values().any(|s| s.cache.shared_prefix_len() == p);
                if !any_attached {
                    self.shared_prefix = None;
                }
            }
            let min_tokens = shared_prefix_min_tokens();
            match &self.shared_prefix {
                None => {
                    let prompts: Vec<&[u32]> = seq_ids
                        .iter()
                        .filter_map(|id| self.seqs.get(id).map(|s| s.prompt.as_slice()))
                        .collect();
                    if prompts.len() < 2 {
                        return Ok(());
                    }
                    let p = longest_common_prefix_len(&prompts);
                    if p < min_tokens {
                        return Ok(());
                    }
                    let donor_id = seq_ids[0];
                    let shared = {
                        let donor = self.seqs.get(&donor_id).ok_or_else(|| {
                            anyhow!("shared-prefix: donor seq {donor_id} missing")
                        })?;
                        if donor.cache.shared_prefix_len() != 0 {
                            return Ok(());
                        }
                        donor.cache.extract_shared_prefix(p)?
                    };
                    // Every seq shares prompt[..p] by construction of the LCP.
                    for id in seq_ids {
                        if let Some(s) = self.seqs.get_mut(id)
                            && s.cache.shared_prefix_len() == 0
                        {
                            s.cache.keep_suffix_from(p)?;
                        }
                    }
                    eprintln!(
                        "[mlx shared-prefix] built: prefix_len={p} seqs={} (single-copy KV)",
                        seq_ids.len()
                    );
                    self.shared_prefix = Some(shared);
                }
                Some(sp) => {
                    let p = sp.prefix_len;
                    // Attach late-joiners whose prompt matches the established
                    // prefix, using an already-attached seq as the reference.
                    let reference: Option<Vec<u32>> = self
                        .seqs
                        .values()
                        .find(|s| s.cache.shared_prefix_len() == p && s.prompt.len() >= p)
                        .map(|s| s.prompt[..p].to_vec());
                    if let Some(refpx) = reference {
                        for id in seq_ids {
                            if let Some(s) = self.seqs.get_mut(id)
                                && s.cache.shared_prefix_len() == 0
                                && s.prompt.len() >= p
                                && s.prompt[..p] == refpx[..]
                            {
                                s.cache.keep_suffix_from(p)?;
                                eprintln!(
                                    "[mlx shared-prefix] attached late seq {id} at prefix_len={p}"
                                );
                            }
                        }
                    }
                }
            }
            Ok(())
        }

        pub(crate) fn decode_step_batch(
            &mut self,
            seq_ids: &[u64],
            last_tokens: &[u32],
            positions: &[usize],
        ) -> Result<Vec<(u32, usize)>> {
            let n = seq_ids.len();
            if last_tokens.len() != n || positions.len() != n {
                return Err(anyhow!(
                    "decode_step_batch: mismatched lengths (seqs={n}, tokens={}, positions={})",
                    last_tokens.len(),
                    positions.len()
                ));
            }
            if n == 0 {
                return Ok(Vec::new());
            }
            if n == 1 {
                return Ok(vec![self.decode_step(
                    seq_ids[0],
                    last_tokens[0],
                    positions[0],
                )?]);
            }
            // Distinct seq_ids are required for the disjoint HashMap borrow.
            let id_set: std::collections::HashSet<u64> = seq_ids.iter().copied().collect();
            if id_set.len() != n {
                return Err(anyhow!("decode_step_batch: duplicate seq_id in batch"));
            }
            // Phase 4: (re)establish the shared prefix before borrowing seqs.
            // Mutates seq caches (truncate) + self.shared_prefix; full &mut self.
            self.maybe_setup_shared_prefix(seq_ids)?;
            // Split-borrow: &model, &shared_prefix and &mut seqs are disjoint fields.
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| anyhow!("native mlx-rs runner: model not loaded"))?;
            let shared = self.shared_prefix.as_ref();
            // Borrow each requested seq's state exactly once (disjoint by key).
            let mut by_id: std::collections::HashMap<u64, &mut NativeSeqState> = self
                .seqs
                .iter_mut()
                .filter(|(k, _)| id_set.contains(k))
                .map(|(k, v)| (*k, v))
                .collect();
            if by_id.len() != n {
                return Err(anyhow!(
                    "decode_step_batch: {} of {n} seq_ids found in runner",
                    by_id.len()
                ));
            }
            // Re-order into caller's seq order.
            let mut states: Vec<&mut NativeSeqState> = Vec::with_capacity(n);
            for sid in seq_ids {
                states.push(by_id.remove(sid).expect("checked present above"));
            }
            // Run the batched forward; the cache-slice borrow ends at block exit.
            let next = {
                let mut caches: Vec<&mut NativePromptCache> =
                    states.iter_mut().map(|s| &mut s.cache).collect();
                model
                    .forward_decode_batch(last_tokens, &mut caches, shared)
                    .context("native mlx-rs runner: forward_decode_batch failed")?
            };
            if next.len() != n {
                return Err(anyhow!(
                    "decode_step_batch: forward returned {} tokens for {n} seqs",
                    next.len()
                ));
            }
            // Bump each seq's position and assemble the result.
            let mut out = Vec::with_capacity(n);
            for (i, s) in states.iter_mut().enumerate() {
                s.position += 1;
                out.push((next[i], s.position));
            }
            Ok(out)
        }

        /// Grammar-aware decode step. Runs the **same forward** as
        /// [`Self::decode_step`] — so the KV cache and, critically for Qwen
        /// 3.6's ~75% linear-attn layers, the conv/SSM recurrent state advance
        /// IDENTICALLY — but instead of an on-GPU argmax it pulls the
        /// last-position logits to the host, hands them to `mask` (which sets
        /// disallowed positions to `-inf` via the grammar matcher), then
        /// argmaxes the masked buffer on the host.
        ///
        /// SSM safety: the forward is byte-for-byte the decode forward; masking
        /// happens entirely AFTER the forward, on a host copy of the logits, so
        /// it only changes WHICH token is selected, never the state recurrence.
        /// The selected token is returned for the caller to feed back through a
        /// normal `decode_step`/`decode_step_masked` next iteration — no token
        /// is jumped and no forward is skipped.
        pub(crate) fn decode_step_masked(
            &mut self,
            seq_id: u64,
            last_token: u32,
            mask: &mut dyn FnMut(&mut [f32]) -> Result<()>,
        ) -> Result<(u32, usize)> {
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| anyhow!("native mlx-rs runner: model not loaded"))?;
            let state = self.seqs.get_mut(&seq_id).ok_or_else(|| {
                anyhow!("native mlx-rs runner: decode_step_masked on unknown seq_id {seq_id}")
            })?;
            let logits = model
                .forward_with_opts(&[last_token], &mut state.cache, /* last_only */ true)
                .with_context(|| {
                    format!("native mlx-rs runner: masked decode forward (seq_id={seq_id})")
                })?;
            // Pull last-position logits to host (single eval), mask, argmax.
            let mut buf = crate::gemma4_sampling::imp::last_logits_to_cpu_f32(&logits)
                .context("native mlx-rs runner: masked decode last-logits-to-cpu failed")?;
            mask(&mut buf).context("native mlx-rs runner: grammar mask callback failed")?;
            let next_tok = argmax_host_f32(&buf)
                .ok_or_else(|| anyhow!("native mlx-rs runner: masked decode empty logit buffer"))?;
            state.position += 1;
            Ok((next_tok, state.position))
        }

        /// [`Self::decode_step_masked`] with the final pick handed to the
        /// caller instead of hardcoded to argmax.
        ///
        /// The forward, the KV advance and the linear-attn conv/SSM state
        /// advance are identical to both siblings — only the choice of token
        /// out of the host logit buffer differs, which is exactly the property
        /// that lets sampling and a grammar mask compose in one callback
        /// without either one touching the recurrent state.
        pub(crate) fn decode_step_pick(
            &mut self,
            seq_id: u64,
            last_token: u32,
            pick: &mut dyn FnMut(&mut [f32]) -> Result<u32>,
        ) -> Result<(u32, usize)> {
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| anyhow!("native mlx-rs runner: model not loaded"))?;
            let state = self.seqs.get_mut(&seq_id).ok_or_else(|| {
                anyhow!("native mlx-rs runner: decode_step_pick on unknown seq_id {seq_id}")
            })?;
            let logits = model
                .forward_with_opts(&[last_token], &mut state.cache, /* last_only */ true)
                .with_context(|| {
                    format!("native mlx-rs runner: sampled decode forward (seq_id={seq_id})")
                })?;
            let mut buf = crate::gemma4_sampling::imp::last_logits_to_cpu_f32(&logits)
                .context("native mlx-rs runner: sampled decode last-logits-to-cpu failed")?;
            let next_tok =
                pick(&mut buf).context("native mlx-rs runner: sampler callback failed")?;
            state.position += 1;
            Ok((next_tok, state.position))
        }

        pub(crate) fn remove_seq(&mut self, seq_id: u64) -> Result<()> {
            self.seqs.remove(&seq_id);
            // Long-prompt repeated-request degradation fix: drop MLX's
            // reusable-buffer cache after each completion. Without this, MLX
            // retains freed buffers for reuse but the cache grows with KV-cache
            // sized allocations, causing fragmentation + swap pressure on
            // subsequent prefills (mlx-c/memory.h::mlx_clear_cache exposes
            // this; mlx-rs has no safe wrapper yet so we call sys directly).
            // Default ON; opt-out via LUMEN_NATIVE_NO_CLEAR_CACHE=1 to A/B.
            // Diagnostic: LUMEN_NATIVE_MEM_PROBE=1 prints active/cache MB.
            //
            // the synchronous clear call can take
            // up to ~45 ms once the buffer cache has grown to ~9 GB across
            // repeated long-prompt requests (cache-saturated regime, see
            // `notes/native_stream_timing_concluded.md`), blocking the
            // post-DONE flush. **Default ON** — defers the clear to a
            // background thread so the request returns immediately; the
            // worker coalesces bursts and runs the clear before the next
            // request typically arrives. Set `LUMEN_NATIVE_DEFER_CLEAR_CACHE=0`
            // to force synchronous (matches prior behavior for parity A/B
            // or to keep `pre/post-clear` mem probes interleaved). Race-safe:
            // mlx_clear_cache only frees pool buffers (already-released by
            // MLX), not in-flight tensors. Smoke 2026-05-02 fresh server:
            // both paths show 0.2-0.3 ms post-DONE flush; the win
            // materializes at cache-saturated steady state.
            let probe = std::env::var("LUMEN_NATIVE_MEM_PROBE").is_ok();
            let no_clear = std::env::var("LUMEN_NATIVE_NO_CLEAR_CACHE").is_ok();
            let defer_clear = std::env::var("LUMEN_NATIVE_DEFER_CLEAR_CACHE")
                .map(|v| v != "0")
                .unwrap_or(true);
            if probe {
                let mut active: usize = 0;
                let mut cache: usize = 0;
                unsafe {
                    mlx_sys::mlx_get_active_memory(&mut active as *mut _);
                    mlx_sys::mlx_get_cache_memory(&mut cache as *mut _);
                }
                eprintln!(
                    "[mlx-native] seq {seq_id} pre-clear: active={:.0} MB cache={:.0} MB",
                    active as f64 / 1024.0 / 1024.0,
                    cache as f64 / 1024.0 / 1024.0,
                );
            }
            if !no_clear {
                if defer_clear {
                    // Hand off to background worker; returns immediately.
                    // Post-clear probe is skipped in the deferred path because
                    // the clear hasn't run yet by the time we'd read it.
                    CacheCleanerWorker::global().request_clear();
                } else {
                    unsafe {
                        mlx_sys::mlx_clear_cache();
                    }
                }
            }
            if probe && !defer_clear {
                let mut active: usize = 0;
                let mut cache: usize = 0;
                unsafe {
                    mlx_sys::mlx_get_active_memory(&mut active as *mut _);
                    mlx_sys::mlx_get_cache_memory(&mut cache as *mut _);
                }
                eprintln!(
                    "[mlx-native] seq {seq_id} post-clear: active={:.0} MB cache={:.0} MB",
                    active as f64 / 1024.0 / 1024.0,
                    cache as f64 / 1024.0 / 1024.0,
                );
            }
            Ok(())
        }

        pub(crate) fn extend(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)> {
            if tokens.is_empty() {
                return Err(anyhow!("native mlx-rs runner: extend given empty tokens"));
            }
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| anyhow!("native mlx-rs runner: model not loaded"))?;
            let state = self.seqs.get_mut(&seq_id).ok_or_else(|| {
                anyhow!("native mlx-rs runner: extend on unknown seq_id {seq_id}")
            })?;
            let logits =
                forward_chunked(model, tokens, &mut state.cache, true, "native-extend", None)
                    .with_context(|| {
                        format!("native mlx-rs runner: extend forward (seq_id={seq_id})")
                    })?;
            let next_tok = model.argmax_last_token(&logits)?;
            state.position += tokens.len();
            Ok((next_tok, state.position))
        }

        // ─────────────────────────────────────────────────────────────────
        // MTP (Multi-Token Prediction) speculative-decode wiring.
        // `enable_mtp` installs a pre-loaded `Qwen35MtpBlock` onto the trunk
        // model. `mtp_step` advances one MTP cycle (Steps A-E in
        // `qwen3_5_moe::mtp_step`) and returns the committed token list
        // (1 + n_accepted + 1 entries). Caller drives the decode loop with
        // `mtp_step` instead of `decode_step` while the MTP path is active.
        // Gating + opt-in lives in the upper engine (Phase 2 S4).
        // ─────────────────────────────────────────────────────────────────

        pub(crate) fn enable_mtp(&mut self, block: Qwen35MtpBlock) -> Result<()> {
            let model = self
                .model
                .as_mut()
                .ok_or_else(|| anyhow!("native mlx-rs runner: enable_mtp before model loaded"))?;
            model
                .try_enable_mtp(block)
                .context("native mlx-rs runner: try_enable_mtp failed")
        }

        pub(crate) fn mtp_enabled(&self) -> bool {
            self.model.as_ref().is_some_and(|m| m.mtp_enabled())
        }

        /// Start MTP corrector calibration (collect drift pairs up to depth K).
        pub(crate) fn enable_mtp_calib(&self, depth_count: usize) -> Result<()> {
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| anyhow!("native runner: enable_mtp_calib before model loaded"))?;
            model.enable_mtp_calib(depth_count);
            Ok(())
        }

        /// Drain calibration stats, fit a corrector, return its serialized bytes.
        pub(crate) fn finalize_mtp_calib(&self, eps: f32) -> Result<Vec<u8>> {
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| anyhow!("native runner: finalize_mtp_calib before model loaded"))?;
            let stats = model
                .take_mtp_calib()
                .ok_or_else(|| anyhow!("native runner: no calibration in progress"))?;
            for (d, (n, pre, post)) in stats.drift_report(eps).into_iter().enumerate() {
                eprintln!(
                    "[mtp-calib] depth {} : n={n} rel-MAE pre={pre:.4} post(affine)={post:.4}",
                    d + 1
                );
            }
            Ok(stats.finalize(eps).to_bytes())
        }

        /// Install a corrector from its serialized bytes (or clear with empty).
        pub(crate) fn set_mtp_corrector_bytes(&self, bytes: &[u8]) -> Result<()> {
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| anyhow!("native runner: set_mtp_corrector before model loaded"))?;
            let corr = lumen_core::mtp_corrector::MtpCorrector::from_bytes(bytes)
                .ok_or_else(|| anyhow!("native runner: corrector bytes failed to parse"))?;
            model.set_mtp_corrector(Some(corr));
            Ok(())
        }

        // ── Orthogonal-Procrustes corrector (the rotation shot) ──────────
        pub(crate) fn enable_mtp_proc_calib(&self, depth_count: usize) -> Result<()> {
            let model = self.model.as_ref().ok_or_else(|| {
                anyhow!("native runner: enable_mtp_proc_calib before model loaded")
            })?;
            model.enable_mtp_proc_calib(depth_count);
            Ok(())
        }

        /// Drain Procrustes stats, print the decisive residual report
        /// (`rel_raw` vs `rel_proc` per depth), return the fitted corrector bytes.
        pub(crate) fn finalize_mtp_proc_calib(&self) -> Result<Vec<u8>> {
            let model = self.model.as_ref().ok_or_else(|| {
                anyhow!("native runner: finalize_mtp_proc_calib before model loaded")
            })?;
            let stats = model
                .take_mtp_proc_calib()
                .ok_or_else(|| anyhow!("native runner: no Procrustes calibration in progress"))?;
            for (d, (n, raw, proc)) in stats.residual_report().into_iter().enumerate() {
                eprintln!(
                    "[mtp-proc-calib] depth {} : n={n} rel-MAE raw={raw:.4} post(rotation)={proc:.4}",
                    d + 1
                );
            }
            Ok(stats.finalize().to_bytes())
        }

        pub(crate) fn set_mtp_procrustes_bytes(&self, bytes: &[u8]) -> Result<()> {
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| anyhow!("native runner: set_mtp_procrustes before model loaded"))?;
            let corr = lumen_core::mtp_procrustes::ProcrustesCorrector::from_bytes(bytes)
                .ok_or_else(|| anyhow!("native runner: procrustes bytes failed to parse"))?;
            model.set_mtp_procrustes(Some(corr));
            Ok(())
        }

        // ── C4 trained lm_head LoRA ──────────────────────────────────────
        pub(crate) fn enable_mtp_c4_calib(&self) -> Result<()> {
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| anyhow!("native runner: enable_mtp_c4_calib before model loaded"))?;
            model.enable_mtp_c4_calib();
            Ok(())
        }

        pub(crate) fn train_mtp_c4_lmhead(&self, rank: usize, steps: usize, lr: f32) -> Result<()> {
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| anyhow!("native runner: train_mtp_c4_lmhead before model loaded"))?;
            model.train_mtp_c4_lmhead(rank, steps, lr)
        }

        pub(crate) fn enable_mtp_hi_calib(&self) -> Result<()> {
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| anyhow!("native runner: enable_mtp_hi_calib before model loaded"))?;
            model.enable_mtp_hi_calib();
            Ok(())
        }

        pub(crate) fn train_mtp_headinternal(
            &self,
            cfg: crate::qwen3_5_mtp::HiTrainCfg,
        ) -> Result<(f32, f32)> {
            let model = self.model.as_ref().ok_or_else(|| {
                anyhow!("native runner: train_mtp_headinternal before model loaded")
            })?;
            model.train_mtp_headinternal(cfg)
        }

        pub(crate) fn mtp_step(
            &mut self,
            seq_id: u64,
            committed_token: u32,
            n_draft: usize,
            sampling: &lumen_core::sampling::SamplingConfig,
        ) -> Result<MtpStepOutput> {
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| anyhow!("native mlx-rs runner: model not loaded"))?;
            if !model.mtp_enabled() {
                return Err(anyhow!(
                    "native mlx-rs runner: mtp_step called but MTP not enabled (call enable_mtp first)"
                ));
            }
            let state = self.seqs.get_mut(&seq_id).ok_or_else(|| {
                anyhow!("native mlx-rs runner: mtp_step on unknown seq_id {seq_id}")
            })?;
            let mut out = model
                .mtp_step(
                    &mut state.cache,
                    &mut state.mtp_kv,
                    committed_token,
                    n_draft,
                    state.mtp_carried.take(),
                    sampling,
                )
                .with_context(|| {
                    format!("native mlx-rs runner: mtp_step orchestration (seq_id={seq_id})")
                })?;
            // Position advances by the number of committed tokens — same
            // semantics as a series of `decode_step` calls.
            state.position += out.committed.len();
            // Stash the 1-forward carry for the next cycle (None on 2-forward).
            // Moved out of the returned struct — the upper decode loop ignores it.
            state.mtp_carried = out.carried_hidden.take();
            Ok(out)
        }

        pub(crate) fn forward_probe(&mut self, seq_id: u64, tokens: &[u32]) -> Result<ProbeRows> {
            if tokens.is_empty() {
                return Err(anyhow!(
                    "native mlx-rs runner: forward_probe given empty tokens"
                ));
            }
            // Mirrors `mlx_runner.py::MlxRunner.forward_probe`: batch-1 forward
            // over `tokens` (length K), then per-row argmax + max(|logit|) over
            // the vocab axis. Cache state advances by K so spec-decode and the
            // Track A2 drift baseline see identical bookkeeping vs the PyO3
            // path.
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| anyhow!("native mlx-rs runner: model not loaded"))?;
            let state = self.seqs.get_mut(&seq_id).ok_or_else(|| {
                anyhow!("native mlx-rs runner: forward_probe on unknown seq_id {seq_id}")
            })?;

            let logits = model.forward(tokens, &mut state.cache).with_context(|| {
                format!("native mlx-rs runner: forward_probe forward (seq_id={seq_id})")
            })?;
            if logits.ndim() != 3 {
                return Err(anyhow!(
                    "native mlx-rs runner: forward_probe expected [B, K, V] logits, got ndim={}",
                    logits.ndim()
                ));
            }
            let k = logits.shape()[1] as usize;

            // Drop the batch axis: [1, K, V] → [K, V].
            let idx_zero = Array::from_int(0);
            let kv = logits
                .take_axis(&idx_zero, 0)
                .context("native mlx-rs runner: forward_probe take_axis(B=0) failed")?;

            // Per-row argmax over the vocab axis → [K] u32.
            let argmaxes = mlx_rs::ops::indexing::argmax_axis(&kv, -1, /* keep_dims */ false)
                .context("native mlx-rs runner: forward_probe argmax_axis failed")?;
            // Per-row max(|logit|) over the vocab axis → [K] f32.
            let abs_kv =
                mlx_rs::ops::abs(&kv).context("native mlx-rs runner: forward_probe abs failed")?;
            let max_abs = mlx_rs::ops::max_axis(&abs_kv, -1, /* keep_dims */ false)
                .context("native mlx-rs runner: forward_probe max_axis failed")?;
            // Per-row `top1 - top2` gap → [K] f32. `topk_axis` does not sort
            // its output, so the gap is the spread of the two values rather
            // than a subtraction in index order.
            let top2 = mlx_rs::ops::indexing::topk_axis(&kv, 2, -1)
                .context("native mlx-rs runner: forward_probe topk_axis failed")?;
            let top2_hi = mlx_rs::ops::max_axis(&top2, -1, /* keep_dims */ false)
                .context("native mlx-rs runner: forward_probe top2 max failed")?;
            let top2_lo = mlx_rs::ops::min_axis(&top2, -1, /* keep_dims */ false)
                .context("native mlx-rs runner: forward_probe top2 min failed")?;
            let top2_gap = mlx_rs::ops::subtract(&top2_hi, &top2_lo)
                .context("native mlx-rs runner: forward_probe top2 gap failed")?;

            argmaxes
                .eval()
                .context("native mlx-rs runner: forward_probe eval(argmaxes) failed")?;
            max_abs
                .eval()
                .context("native mlx-rs runner: forward_probe eval(max_abs) failed")?;
            top2_gap
                .eval()
                .context("native mlx-rs runner: forward_probe eval(top2_gap) failed")?;

            let row_argmaxes: Vec<u32> = argmaxes.as_slice::<u32>().to_vec();
            let row_max_abs: Vec<f32> = max_abs.as_slice::<f32>().to_vec();
            let row_top2_gap: Vec<f32> = top2_gap.as_slice::<f32>().to_vec();

            state.position += k;
            Ok(ProbeRows {
                row_argmaxes,
                row_max_abs,
                row_top2_gap,
                position: state.position,
            })
        }

        // Snapshot / fork APIs power Track A1 prefix caching (deep snapshot
        // + fork) and Track A2 spec-decode rollback (shallow snapshot +
        // restore). Mirrors `mlx_runner.py::MlxRunner.snapshot_state` /
        // `restore_state` / `snapshot_state_deep` / `fork_from_snapshot` /
        // `release_snapshot`. See `native_snapshot.rs` for capture/restore
        // semantics.

        pub(crate) fn snapshot_state(&mut self, seq_id: u64) -> Result<u64> {
            let seq = self.seqs.get(&seq_id).ok_or_else(|| {
                anyhow!("native mlx-rs runner: snapshot_state — unknown seq_id {seq_id}")
            })?;
            let snap = PromptCacheSnapshot::capture_shallow(&seq.cache, seq.position)
                .context("native mlx-rs runner: snapshot_state — capture_shallow failed")?;
            let sid = self.next_snapshot_id;
            self.next_snapshot_id = self
                .next_snapshot_id
                .checked_add(1)
                .ok_or_else(|| anyhow!("native mlx-rs runner: snapshot_id counter exhausted"))?;
            self.snapshots.insert(sid, snap);
            Ok(sid)
        }

        pub(crate) fn restore_state(&mut self, seq_id: u64, snapshot_id: u64) -> Result<usize> {
            // Mirror PyO3 `restore_state`: snapshot is consumed (1-shot).
            let snap = self.snapshots.remove(&snapshot_id).ok_or_else(|| {
                anyhow!("native mlx-rs runner: restore_state — unknown snapshot_id {snapshot_id}")
            })?;
            let seq = self.seqs.get_mut(&seq_id).ok_or_else(|| {
                anyhow!("native mlx-rs runner: restore_state — unknown seq_id {seq_id}")
            })?;
            snap.restore_into(&mut seq.cache)
                .context("native mlx-rs runner: restore_state — restore_into failed")?;
            seq.position = snap.position();
            Ok(snap.position())
        }

        pub(crate) fn release_snapshot(&mut self, snapshot_id: u64) -> Result<()> {
            // Idempotent — silent no-op if id is unknown (matches PyO3).
            self.snapshots.remove(&snapshot_id);
            Ok(())
        }

        /// L2 disk tier: serialize the deep snapshot `snapshot_id` to disk under
        /// `key` so a future process can fork it instead of cold-prefilling.
        /// No-op when the disk tier is off. Dense `Full` + `Linear` layers only
        /// (TurboQuant-compressed layers cause `to_records` to error, surfaced
        /// here as a non-fatal `Err` the caller logs and ignores).
        pub(crate) fn persist_snapshot_disk(
            &mut self,
            snapshot_id: u64,
            key: &str,
            prefix_tokens: &[u32],
            last_token: Option<u32>,
        ) -> Result<()> {
            if self.disk.is_none() {
                return Ok(());
            }
            // Read the snapshot to host bytes BEFORE borrowing `disk` mutably so
            // the two field borrows don't overlap.
            let (layers, records, position, is_deep) = {
                let snap = self.snapshots.get(&snapshot_id).ok_or_else(|| {
                    anyhow!("native mlx-rs runner: persist_snapshot_disk — unknown snapshot {snapshot_id}")
                })?;
                let (layers, records) = snap.to_records()?;
                (layers, records, snap.position(), snap.is_deep())
            };
            let created_at_unix = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let manifest = KvManifest {
                model_fingerprint: self.fingerprint.clone(),
                created_at_unix,
                position,
                prefix_tokens: prefix_tokens.to_vec(),
                last_token,
                is_deep,
                layers,
            };
            let disk = self
                .disk
                .as_mut()
                .expect("disk tier present (checked above)");
            disk.put(key, &manifest, &records)?;
            Ok(())
        }

        /// L2 disk tier: rehydrate a persisted snapshot for `key` whose token
        /// prefix is a prefix of `prompt_ids`. On success, materializes it into
        /// the in-memory snapshot table (so it can be forked) and returns
        /// `(snapshot_id, prefix_tokens, last_token)`. `None` on miss / disk off
        /// / corrupt / non-prefix / fingerprint mismatch (the store itself
        /// discards mismatched entries).
        pub(crate) fn load_persisted_disk(
            &mut self,
            key: &str,
            prompt_ids: &[u32],
        ) -> Result<Option<(u64, Vec<u32>, Option<u32>)>> {
            if self.disk.is_none() {
                return Ok(None);
            }
            let loaded = {
                let disk = self
                    .disk
                    .as_mut()
                    .expect("disk tier present (checked above)");
                disk.get(key)?
            };
            let Some((manifest, records)) = loaded else {
                return Ok(None);
            };
            // The persisted prefix must be a genuine prefix of this prompt, else
            // forking it would produce a wrong KV state.
            if manifest.prefix_tokens.is_empty() || !prompt_ids.starts_with(&manifest.prefix_tokens)
            {
                return Ok(None);
            }
            let snap = PromptCacheSnapshot::from_records(
                &manifest.layers,
                &records,
                manifest.position,
                /* is_deep = */ true,
            )
            .context("native mlx-rs runner: load_persisted_disk — from_records failed")?;
            let sid = self.next_snapshot_id;
            self.next_snapshot_id = self
                .next_snapshot_id
                .checked_add(1)
                .ok_or_else(|| anyhow!("native mlx-rs runner: snapshot_id counter exhausted"))?;
            self.snapshots.insert(sid, snap);
            Ok(Some((sid, manifest.prefix_tokens, manifest.last_token)))
        }

        pub(crate) fn snapshot_state_deep(&mut self, seq_id: u64) -> Result<(u64, usize)> {
            let seq = self.seqs.get(&seq_id).ok_or_else(|| {
                anyhow!("native mlx-rs runner: snapshot_state_deep — unknown seq_id {seq_id}")
            })?;
            let snap = PromptCacheSnapshot::capture_deep(&seq.cache, seq.position)
                .context("native mlx-rs runner: snapshot_state_deep — capture_deep failed")?;
            let position = snap.position();
            let sid = self.next_snapshot_id;
            self.next_snapshot_id = self
                .next_snapshot_id
                .checked_add(1)
                .ok_or_else(|| anyhow!("native mlx-rs runner: snapshot_id counter exhausted"))?;
            self.snapshots.insert(sid, snap);
            Ok((sid, position))
        }

        pub(crate) fn fork_from_snapshot(
            &mut self,
            snapshot_id: u64,
            dst_seq_id: u64,
        ) -> Result<usize> {
            if self.seqs.contains_key(&dst_seq_id) {
                return Err(anyhow!(
                    "native mlx-rs runner: fork_from_snapshot — dst_seq_id {dst_seq_id} already exists"
                ));
            }
            // Snapshot is NOT consumed — multiple forks from the same master
            // remain mutually independent (each install_into_fresh re-deep-clones).
            let snap = self.snapshots.get(&snapshot_id).ok_or_else(|| {
                anyhow!(
                    "native mlx-rs runner: fork_from_snapshot — unknown snapshot_id {snapshot_id}"
                )
            })?;
            // Build a fresh empty cache mirroring the model's layer kinds. We
            // borrow the existing seqs[*].cache as a structural template — any
            // existing seq has the same `is_linear_per_layer` profile because
            // they were all built from the same model. If no seqs exist yet,
            // fall back to constructing from the model's layer types.
            let mut fresh = self.require_model()?.make_cache();
            snap.install_into_fresh(&mut fresh)
                .context("native mlx-rs runner: fork_from_snapshot — install_into_fresh failed")?;
            let position = snap.position();
            self.seqs.insert(
                dst_seq_id,
                NativeSeqState {
                    cache: fresh,
                    position,
                    mtp_kv: NativeKvCache::new(),
                    mtp_carried: None,
                    prompt: Vec::new(),
                    // Snapshots are only taken on text prefixes (image requests
                    // bypass the prefix cache), so there is no shift to carry.
                    mrope_delta: 0,
                },
            );
            Ok(position)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::process::{Command, Output};

        #[test]
        fn native_model_assets_discovers_local_directory() -> Result<()> {
            let root = test_model_dir("complete")?;
            std::fs::write(root.join("config.json"), "{}")?;
            std::fs::write(root.join("tokenizer.json"), "{}")?;
            std::fs::write(root.join("model-00002-of-00002.safetensors"), b"")?;
            std::fs::write(root.join("model-00001-of-00002.safetensors"), b"")?;

            let assets = NativeModelAssets::discover_local(root.to_str().ok_or_else(|| {
                anyhow!(
                    "test model directory path is not valid utf-8: {}",
                    root.display()
                )
            })?)?;

            assert_eq!(assets.root, root);
            assert_eq!(assets.config_json, assets.root.join("config.json"));
            assert_eq!(
                assets.tokenizer_json,
                Some(assets.root.join("tokenizer.json"))
            );
            assert_eq!(
                assets.safetensors,
                vec![
                    assets.root.join("model-00001-of-00002.safetensors"),
                    assets.root.join("model-00002-of-00002.safetensors"),
                ]
            );

            std::fs::remove_dir_all(&assets.root)?;
            Ok(())
        }

        #[test]
        fn native_model_assets_requires_config_and_weights() -> Result<()> {
            let missing_config = test_model_dir("missing-config")?;
            std::fs::write(missing_config.join("model.safetensors"), b"")?;
            let err =
                NativeModelAssets::discover_local(missing_config.to_str().ok_or_else(|| {
                    anyhow!(
                        "test model directory path is not valid utf-8: {}",
                        missing_config.display()
                    )
                })?)
                .unwrap_err();
            assert!(err.to_string().contains("config.json"));
            std::fs::remove_dir_all(&missing_config)?;

            let missing_weights = test_model_dir("missing-weights")?;
            std::fs::write(missing_weights.join("config.json"), "{}")?;
            let err =
                NativeModelAssets::discover_local(missing_weights.to_str().ok_or_else(|| {
                    anyhow!(
                        "test model directory path is not valid utf-8: {}",
                        missing_weights.display()
                    )
                })?)
                .unwrap_err();
            assert!(err.to_string().contains("safetensors weights"));
            std::fs::remove_dir_all(&missing_weights)?;

            Ok(())
        }

        #[test]
        fn native_model_config_validates_qwen3_5_moe_contract() -> Result<()> {
            let root = test_model_dir("valid-config")?;
            let config_path = root.join("config.json");
            std::fs::write(
                &config_path,
                r#"{
                    "model_type": "qwen3_5_moe",
                    "eos_token_id": [248046, 248044],
                    "quantization_config": {
                        "group_size": 32,
                        "bits": 4,
                        "mode": "mxfp4"
                    },
                    "text_config": {
                        "model_type": "qwen3_5_moe_text",
                        "hidden_size": 16,
                        "head_dim": 4,
                        "num_attention_heads": 4,
                        "num_key_value_heads": 2,
                        "num_hidden_layers": 3,
                        "layer_types": [
                            "linear_attention",
                            "linear_attention",
                            "full_attention"
                        ],
                        "vocab_size": 1024
                    }
                }"#,
            )?;

            let config = NativeModelConfig::load(&config_path)?;
            let text = config.validate_qwen3_5_moe_contract()?;

            assert_eq!(config.eos_token_ids, vec![248046, 248044]);
            assert_eq!(text.hidden_size, 16);
            assert_eq!(
                text.layer_types,
                vec![
                    NativeLayerType::LinearAttention,
                    NativeLayerType::LinearAttention,
                    NativeLayerType::FullAttention,
                ]
            );

            std::fs::remove_dir_all(root)?;
            Ok(())
        }

        #[test]
        fn native_model_config_accepts_checked_in_qwen3_5_moe_fixture() -> Result<()> {
            let fixture = include_str!("../tests/fixtures/qwen3_5_moe_config.json");
            let config: NativeModelConfig = serde_json::from_str(fixture)?;
            let text = config.validate_qwen3_5_moe_contract()?;

            assert_eq!(text.model_type, "qwen3_5_moe_text");
            assert_eq!(text.num_hidden_layers, text.layer_types.len());
            assert!(
                config
                    .quantization_config
                    .as_ref()
                    .is_some_and(|quantization| !quantization.overrides.is_empty())
            );

            Ok(())
        }

        #[test]
        fn native_model_config_accepts_scalar_eos_token_id() -> Result<()> {
            let root = test_model_dir("scalar-eos-config")?;
            let config_path = root.join("config.json");
            std::fs::write(
                &config_path,
                r#"{
                    "model_type": "qwen3_5_moe",
                    "eos_token_id": 42,
                    "text_config": {
                        "model_type": "qwen3_5_moe_text",
                        "hidden_size": 8,
                        "head_dim": 4,
                        "num_attention_heads": 2,
                        "num_key_value_heads": 1,
                        "num_hidden_layers": 1,
                        "layer_types": ["full_attention"],
                        "vocab_size": 128
                    }
                }"#,
            )?;

            let config = NativeModelConfig::load(&config_path)?;
            config.validate_qwen3_5_moe_contract()?;
            assert_eq!(config.eos_token_ids, vec![42]);

            std::fs::remove_dir_all(root)?;
            Ok(())
        }

        #[test]
        fn native_model_config_rejects_contract_mismatches() -> Result<()> {
            let wrong_model_type = serde_json::from_str::<NativeModelConfig>(
                r#"{
                    "model_type": "other",
                    "text_config": {
                        "model_type": "qwen3_5_moe_text",
                        "hidden_size": 8,
                        "head_dim": 4,
                        "num_attention_heads": 2,
                        "num_key_value_heads": 1,
                        "num_hidden_layers": 1,
                        "layer_types": ["full_attention"],
                        "vocab_size": 128
                    }
                }"#,
            )?;
            let err = wrong_model_type
                .validate_qwen3_5_moe_contract()
                .unwrap_err();
            assert!(err.to_string().contains("model_type"));

            let mismatched_layers = serde_json::from_str::<NativeModelConfig>(
                r#"{
                    "model_type": "qwen3_5_moe",
                    "text_config": {
                        "model_type": "qwen3_5_moe_text",
                        "hidden_size": 8,
                        "head_dim": 4,
                        "num_attention_heads": 2,
                        "num_key_value_heads": 1,
                        "num_hidden_layers": 2,
                        "layer_types": ["full_attention"],
                        "vocab_size": 128
                    }
                }"#,
            )?;
            let err = mismatched_layers
                .validate_qwen3_5_moe_contract()
                .unwrap_err();
            assert!(err.to_string().contains("layer_types"));

            let invalid_quantization = serde_json::from_str::<NativeModelConfig>(
                r#"{
                    "model_type": "qwen3_5_moe",
                    "quantization_config": {
                        "group_size": 32,
                        "bits": 8,
                        "mode": "mxfp4"
                    },
                    "text_config": {
                        "model_type": "qwen3_5_moe_text",
                        "hidden_size": 8,
                        "head_dim": 4,
                        "num_attention_heads": 2,
                        "num_key_value_heads": 1,
                        "num_hidden_layers": 1,
                        "layer_types": ["full_attention"],
                        "vocab_size": 128
                    }
                }"#,
            )?;
            let err = invalid_quantization
                .validate_qwen3_5_moe_contract()
                .unwrap_err();
            assert!(err.to_string().contains("MXFP4"));

            Ok(())
        }

        #[test]
        fn mlx_smoke_child_entry() -> Result<()> {
            match std::env::var(MLX_SMOKE_CHILD_MODE_ENV).as_deref() {
                Ok(MLX_SMOKE_METAL_AVAILABLE_MODE) => mlx_metal_available_smoke(),
                Ok(MLX_SMOKE_METADATA_MODE) => mlx_tensor_metadata_smoke(),
                Ok(MLX_SMOKE_RAW_ARRAY_EMPTY_MODE) => mlx_raw_array_empty_smoke(),
                Ok(MLX_SMOKE_RAW_ARRAY_NEW_INT_MODE) => mlx_raw_array_new_int_smoke(),
                Ok(MLX_SMOKE_RAW_ARRAY_NEW_BOOL_MODE) => mlx_raw_array_new_bool_smoke(),
                Ok(MLX_SMOKE_RAW_ARRAY_NEW_FLOAT32_MODE) => mlx_raw_array_new_float32_smoke(),
                Ok(MLX_SMOKE_RAW_ARRAY_NEW_FLOAT64_MODE) => mlx_raw_array_new_float64_smoke(),
                Ok(MLX_SMOKE_RAW_ARRAY_NEW_DATA_INT_MODE) => mlx_raw_array_new_data_int_smoke(),
                Ok(MLX_SMOKE_RAW_ARRAY_NEW_DATA_EMPTY_INT_MODE) => {
                    mlx_raw_array_new_data_empty_int_smoke()
                }
                Ok(MLX_SMOKE_SCALAR_ITEM_MODE) => mlx_tensor_scalar_item_smoke(),
                Ok(MLX_SMOKE_ARGMAX_EVAL_MODE) => mlx_tensor_argmax_eval_smoke(),
                Ok(MLX_SMOKE_SAFETENSORS_MODE) => mlx_tensor_safetensors_smoke(),
                Ok(mode) => Err(anyhow!("unknown native mlx-rs child smoke mode: {mode}")),
                Err(_) => Ok(()),
            }
        }

        #[test]
        #[ignore = "diagnostic smoke for raw mlx-sys metal availability; runs MLX FFI in a child process"]
        fn mlx_metal_available_smoke_isolated() -> Result<()> {
            let output = run_mlx_smoke_child(MLX_SMOKE_METAL_AVAILABLE_MODE)?;
            if !output.status.success() {
                return Err(anyhow!(
                    "native mlx-sys metal availability child smoke failed with status {}\nstdout:\n{}\nstderr:\n{}",
                    output.status,
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr),
                ));
            }

            let stderr = String::from_utf8_lossy(&output.stderr);
            if !stderr
                .contains("native mlx-rs smoke [metal_available]: after mlx_metal_is_available")
            {
                return Err(anyhow!(
                    "native mlx-sys metal availability child smoke did not reach completion marker with status {}\nstdout:\n{}\nstderr:\n{}",
                    output.status,
                    String::from_utf8_lossy(&output.stdout),
                    stderr,
                ));
            }

            Ok(())
        }

        #[test]
        #[ignore = "diagnostic smoke for the raw mlx-sys empty array constructor; runs MLX FFI in a child process"]
        fn mlx_raw_array_empty_smoke_isolated() -> Result<()> {
            let output = run_mlx_smoke_child(MLX_SMOKE_RAW_ARRAY_EMPTY_MODE)?;
            if !output.status.success() {
                return Err(anyhow!(
                    "native mlx-sys empty array child smoke failed with status {}\nstdout:\n{}\nstderr:\n{}",
                    output.status,
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr),
                ));
            }

            let stderr = String::from_utf8_lossy(&output.stderr);
            if !stderr.contains("native mlx-rs smoke [raw_array_empty]: after mlx_array_new") {
                return Err(anyhow!(
                    "native mlx-sys empty array child smoke did not reach completion marker with status {}\nstdout:\n{}\nstderr:\n{}",
                    output.status,
                    String::from_utf8_lossy(&output.stdout),
                    stderr,
                ));
            }

            Ok(())
        }

        #[test]
        #[ignore = "diagnostic smoke for the raw mlx-sys int allocation path; runs the unsafe path in a child process"]
        fn mlx_raw_array_new_int_diagnostic_isolated() -> Result<()> {
            assert_raw_array_constructor_diagnostic(
                MLX_SMOKE_RAW_ARRAY_NEW_INT_MODE,
                "native mlx-rs smoke [raw_array_new_int]: before mlx_array_new_int",
                "native mlx-rs smoke [raw_array_new_int]: after mlx_array_free",
            )
        }

        #[test]
        #[ignore = "diagnostic smoke for the raw mlx-sys bool allocation path; runs the unsafe path in a child process"]
        fn mlx_raw_array_new_bool_diagnostic_isolated() -> Result<()> {
            assert_raw_array_constructor_diagnostic(
                MLX_SMOKE_RAW_ARRAY_NEW_BOOL_MODE,
                "native mlx-rs smoke [raw_array_new_bool]: before mlx_array_new_bool",
                "native mlx-rs smoke [raw_array_new_bool]: after mlx_array_free",
            )
        }

        #[test]
        #[ignore = "diagnostic smoke for the raw mlx-sys float32 allocation path; runs the unsafe path in a child process"]
        fn mlx_raw_array_new_float32_diagnostic_isolated() -> Result<()> {
            assert_raw_array_constructor_diagnostic(
                MLX_SMOKE_RAW_ARRAY_NEW_FLOAT32_MODE,
                "native mlx-rs smoke [raw_array_new_float32]: before mlx_array_new_float32",
                "native mlx-rs smoke [raw_array_new_float32]: after mlx_array_free",
            )
        }

        #[test]
        #[ignore = "diagnostic smoke for the raw mlx-sys float64 allocation path; runs the unsafe path in a child process"]
        fn mlx_raw_array_new_float64_diagnostic_isolated() -> Result<()> {
            assert_raw_array_constructor_diagnostic(
                MLX_SMOKE_RAW_ARRAY_NEW_FLOAT64_MODE,
                "native mlx-rs smoke [raw_array_new_float64]: before mlx_array_new_float64",
                "native mlx-rs smoke [raw_array_new_float64]: after mlx_array_free",
            )
        }

        #[test]
        #[ignore = "diagnostic smoke for the raw mlx-sys int data array constructor path; runs the unsafe path in a child process"]
        fn mlx_raw_array_new_data_int_diagnostic_isolated() -> Result<()> {
            assert_raw_array_constructor_diagnostic(
                MLX_SMOKE_RAW_ARRAY_NEW_DATA_INT_MODE,
                "native mlx-rs smoke [raw_array_new_data_int]: before mlx_array_new_data",
                "native mlx-rs smoke [raw_array_new_data_int]: after mlx_array_free",
            )
        }

        #[test]
        #[ignore = "diagnostic smoke for the raw mlx-sys empty int data array constructor path; runs the unsafe path in a child process"]
        fn mlx_raw_array_new_data_empty_int_diagnostic_isolated() -> Result<()> {
            assert_raw_array_constructor_diagnostic(
                MLX_SMOKE_RAW_ARRAY_NEW_DATA_EMPTY_INT_MODE,
                "native mlx-rs smoke [raw_array_new_data_empty_int]: before mlx_array_new_data",
                "native mlx-rs smoke [raw_array_new_data_empty_int]: after mlx_array_free",
            )
        }

        #[test]
        #[ignore = "diagnostic smoke for mlx-rs allocation/metadata; runs MLX FFI in a child process"]
        fn mlx_metadata_smoke_isolated() -> Result<()> {
            let output = run_mlx_smoke_child(MLX_SMOKE_METADATA_MODE)?;
            if !output.status.success() {
                return Err(anyhow!(
                    "native mlx-rs metadata child smoke failed with status {}\nstdout:\n{}\nstderr:\n{}",
                    output.status,
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr),
                ));
            }

            let stderr = String::from_utf8_lossy(&output.stderr);
            if !stderr.contains("native mlx-rs smoke [metadata]: after shape") {
                return Err(anyhow!(
                    "native mlx-rs metadata child smoke did not reach completion marker with status {}\nstdout:\n{}\nstderr:\n{}",
                    output.status,
                    String::from_utf8_lossy(&output.stdout),
                    stderr,
                ));
            }

            Ok(())
        }

        #[test]
        #[ignore = "diagnostic smoke for the mlx-rs scalar read path; runs the unsafe path in a child process"]
        fn mlx_scalar_item_diagnostic_isolated() -> Result<()> {
            let output = run_mlx_smoke_child(MLX_SMOKE_SCALAR_ITEM_MODE)?;
            let stderr = String::from_utf8_lossy(&output.stderr);

            if output.status.success() {
                if stderr.contains("native mlx-rs smoke [scalar_item]: after try_item") {
                    return Ok(());
                }
                return Err(anyhow!(
                    "native mlx-rs scalar child smoke exited successfully without reaching completion marker\nstdout:\n{}\nstderr:\n{}",
                    String::from_utf8_lossy(&output.stdout),
                    stderr,
                ));
            }

            if !stderr.contains("native mlx-rs smoke [scalar_item]: before Array::from_int") {
                return Err(anyhow!(
                    "native mlx-rs scalar child smoke failed before reaching the allocation marker with status {}\nstdout:\n{}\nstderr:\n{}",
                    output.status,
                    String::from_utf8_lossy(&output.stdout),
                    stderr,
                ));
            }

            Ok(())
        }

        #[test]
        #[ignore = "diagnostic smoke for mlx-rs argmax/eval; runs MLX FFI in a child process"]
        fn mlx_argmax_eval_diagnostic_isolated() -> Result<()> {
            let output = run_mlx_smoke_child(MLX_SMOKE_ARGMAX_EVAL_MODE)?;
            let stderr = String::from_utf8_lossy(&output.stderr);

            if output.status.success() {
                if stderr.contains("native mlx-rs smoke [argmax_eval]: after try_item") {
                    return Ok(());
                }
                return Err(anyhow!(
                    "native mlx-rs argmax/eval child smoke exited successfully without reaching completion marker\nstdout:\n{}\nstderr:\n{}",
                    String::from_utf8_lossy(&output.stdout),
                    stderr,
                ));
            }

            if !stderr.contains("native mlx-rs smoke [argmax_eval]: before Array::from_slice") {
                return Err(anyhow!(
                    "native mlx-rs argmax/eval child smoke failed before reaching the allocation marker with status {}\nstdout:\n{}\nstderr:\n{}",
                    output.status,
                    String::from_utf8_lossy(&output.stdout),
                    stderr,
                ));
            }

            Ok(())
        }

        #[test]
        #[ignore = "diagnostic smoke for mlx-rs safetensors save/load; runs MLX FFI in a child process"]
        fn mlx_safetensors_diagnostic_isolated() -> Result<()> {
            let output = run_mlx_smoke_child(MLX_SMOKE_SAFETENSORS_MODE)?;
            let stderr = String::from_utf8_lossy(&output.stderr);

            if output.status.success() {
                if stderr.contains("native mlx-rs smoke [safetensors]: after try_item") {
                    return Ok(());
                }
                return Err(anyhow!(
                    "native mlx-rs safetensors child smoke exited successfully without reaching completion marker\nstdout:\n{}\nstderr:\n{}",
                    String::from_utf8_lossy(&output.stdout),
                    stderr,
                ));
            }

            if !stderr.contains("native mlx-rs smoke [safetensors]: before Array::from_slice") {
                return Err(anyhow!(
                    "native mlx-rs safetensors child smoke failed before reaching the allocation marker with status {}\nstdout:\n{}\nstderr:\n{}",
                    output.status,
                    String::from_utf8_lossy(&output.stdout),
                    stderr,
                ));
            }

            Ok(())
        }

        fn run_mlx_smoke_child(mode: &str) -> Result<Output> {
            let output = Command::new(std::env::current_exe()?)
                .arg("runner_native::imp::tests::mlx_smoke_child_entry")
                .arg("--exact")
                .arg("--nocapture")
                .env(MLX_SMOKE_CHILD_MODE_ENV, mode)
                .output()
                .map_err(|err| anyhow!("failed to run native mlx-rs child smoke: {err}"))?;
            Ok(output)
        }

        fn assert_raw_array_constructor_diagnostic(
            mode: &str,
            before_marker: &str,
            after_marker: &str,
        ) -> Result<()> {
            let output = run_mlx_smoke_child(mode)?;
            let stderr = String::from_utf8_lossy(&output.stderr);

            if output.status.success() {
                if stderr.contains(after_marker) {
                    return Ok(());
                }
                return Err(anyhow!(
                    "native mlx-sys raw array child smoke exited successfully without reaching completion marker\nstdout:\n{}\nstderr:\n{}",
                    String::from_utf8_lossy(&output.stdout),
                    stderr,
                ));
            }

            if !stderr.contains(before_marker) {
                return Err(anyhow!(
                    "native mlx-sys raw array child smoke failed before reaching the allocation marker with status {}\nstdout:\n{}\nstderr:\n{}",
                    output.status,
                    String::from_utf8_lossy(&output.stdout),
                    stderr,
                ));
            }

            Ok(())
        }

        fn test_model_dir(name: &str) -> Result<PathBuf> {
            let root = std::env::temp_dir().join(format!(
                "lumen_mlx_native_assets_{}_{}",
                std::process::id(),
                name
            ));
            if root.exists() {
                std::fs::remove_dir_all(&root)?;
            }
            std::fs::create_dir_all(&root)?;
            Ok(root)
        }
    }
}

#[cfg(not(feature = "mlx-native"))]
mod imp {
    use super::*;

    pub(crate) struct NativeMlxRunner;

    impl NativeMlxRunner {
        pub(crate) fn new() -> Result<Self> {
            Err(anyhow!(
                "native mlx-rs runner requested, but lumen-mlx was built without the `mlx-native` feature"
            ))
        }

        pub(crate) fn load(&mut self, _model_id: &str) -> Result<LoadInfo> {
            Err(anyhow!(
                "native mlx-rs runner requested, but lumen-mlx was built without the `mlx-native` feature"
            ))
        }

        pub(crate) fn prefill(&mut self, _seq_id: u64, _tokens: &[u32]) -> Result<(u32, usize)> {
            Err(anyhow!(
                "native mlx-rs runner requested, but lumen-mlx was built without the `mlx-native` feature"
            ))
        }

        pub(crate) fn decode_step(
            &mut self,
            _seq_id: u64,
            _last_token: u32,
            _position: usize,
        ) -> Result<(u32, usize)> {
            Err(anyhow!(
                "native mlx-rs runner requested, but lumen-mlx was built without the `mlx-native` feature"
            ))
        }

        pub(crate) fn remove_seq(&mut self, _seq_id: u64) -> Result<()> {
            Ok(())
        }

        pub(crate) fn extend(&mut self, _seq_id: u64, _tokens: &[u32]) -> Result<(u32, usize)> {
            Err(anyhow!(
                "native mlx-rs runner requested, but lumen-mlx was built without the `mlx-native` feature"
            ))
        }

        pub(crate) fn forward_probe(&mut self, _seq_id: u64, _tokens: &[u32]) -> Result<ProbeRows> {
            Err(anyhow!(
                "native mlx-rs runner requested, but lumen-mlx was built without the `mlx-native` feature"
            ))
        }

        pub(crate) fn snapshot_state(&mut self, _seq_id: u64) -> Result<u64> {
            Err(anyhow!(
                "native mlx-rs runner requested, but lumen-mlx was built without the `mlx-native` feature"
            ))
        }

        pub(crate) fn restore_state(&mut self, _seq_id: u64, _snapshot_id: u64) -> Result<usize> {
            Err(anyhow!(
                "native mlx-rs runner requested, but lumen-mlx was built without the `mlx-native` feature"
            ))
        }

        pub(crate) fn release_snapshot(&mut self, _snapshot_id: u64) -> Result<()> {
            Err(anyhow!(
                "native mlx-rs runner requested, but lumen-mlx was built without the `mlx-native` feature"
            ))
        }

        pub(crate) fn snapshot_state_deep(&mut self, _seq_id: u64) -> Result<(u64, usize)> {
            Err(anyhow!(
                "native mlx-rs runner requested, but lumen-mlx was built without the `mlx-native` feature"
            ))
        }

        pub(crate) fn fork_from_snapshot(
            &mut self,
            _snapshot_id: u64,
            _dst_seq_id: u64,
        ) -> Result<usize> {
            Err(anyhow!(
                "native mlx-rs runner requested, but lumen-mlx was built without the `mlx-native` feature"
            ))
        }

        // L2 disk tier. Both mirror what the real runner does when no disk
        // tier is configured (`self.disk.is_none()`): persisting is a no-op
        // and a lookup is a miss. Erroring instead would be louder but wrong —
        // the caller treats a failure as "the disk write broke", and nothing
        // broke here; there is simply no state. Unreachable in practice, since
        // `load()` above fails long before anything holds a snapshot id.
        pub(crate) fn persist_snapshot_disk(
            &mut self,
            _snapshot_id: u64,
            _key: &str,
            _prefix_tokens: &[u32],
            _last_token: Option<u32>,
        ) -> Result<()> {
            Ok(())
        }

        pub(crate) fn load_persisted_disk(
            &mut self,
            _key: &str,
            _prompt_ids: &[u32],
        ) -> Result<Option<(u64, Vec<u32>, Option<u32>)>> {
            Ok(None)
        }

        pub(crate) fn take_decode_timing_log(&mut self) -> Option<Vec<(f64, f64)>> {
            None
        }

        pub(crate) fn take_decode_fine_timing_log(
            &mut self,
        ) -> Option<Vec<crate::native_runtime::FineTimings>> {
            None
        }
    }
}

pub(crate) use imp::NativeMlxRunner;
#[cfg(feature = "mlx-native")]
pub(crate) use imp::resolve_model_dir;
