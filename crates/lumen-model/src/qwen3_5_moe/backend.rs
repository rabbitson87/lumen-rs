//! High-level wrapper that turns a loaded [`Qwen3_5MoeTextModel`] into something the
//! server engine can actually call — tokenizer, encode/decode, autoregressive generate,
//! and a Qwen3-style chat template. Mirrors the API surface of [`crate::qwen::QwenModel`]
//! so the server's [`ModelBackend`](../../lumen_server/src/engine.rs) can dispatch
//! the same way.
//!
//! ## Current limitations (Stage 6-a scope)
//! - **No KV cache.** `generate` is prefill-only — each new token re-runs the full 40-layer
//!   forward over the whole sequence so far. Correctness is fine; throughput is awful.
//!   Real decoding lands alongside Stage 5 (KV cache + SSM state passthrough).
//! - **Hardcoded chat template.** Qwen3 follows the same `<|im_start|>role\ncontent<|im_end|>`
//!   Qwen2 inherited. A proper `minijinja`-driven loader (picking up `chat_template.jinja`)
//!   is a Stage 3 follow-up.
//! - **Lm-head stays dense.** Loader dequantizes `language_model.lm_head` to f32 (~2 GB).
//!   Wrapping it as `Mxfp4Linear` is a trivial next PR but not required for correctness.

#![cfg(feature = "turboquant-gpu")]

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Storage, Tensor, D};
use tokenizers::Tokenizer;
use lumen_metal::metal::{Buffer, CommandQueue, Device as MtlDevice};
use lumen_metal::affine4_gpu::Affine4Context;
use lumen_metal::mxfp4_gpu::MxFp4Context;
use lumen_metal::sampling::SamplingKernels;

use crate::sampling::sample_token_cpu_full_owned;

/// Same ABI-transmute trick used by `mxfp4_linear` and the native bridge: pull a
/// borrowed `metal::Buffer` ref + byte offset out of a Metal-resident Candle
/// tensor. Returns `None` for CPU-resident tensors. The caller is responsible
/// for ensuring the tensor outlives the borrow.
fn metal_buffer_of(t: &Tensor) -> Option<(&Buffer, u64)> {
    let (storage_guard, layout) = t.storage_and_layout();
    match &*storage_guard {
        Storage::Metal(ms) => {
            let offset_bytes = (layout.start_offset() * t.dtype().size_in_bytes()) as u64;
            let candle_buf = ms.buffer();
            let buf_ptr = candle_buf as *const _ as *const Buffer;
            Some((unsafe { &*buf_ptr }, offset_bytes))
        }
        _ => None,
    }
}

/// GPU sampler resources: the compiled argmax pipeline, a private command
/// queue (kept separate from Candle's), and a 1-element u32 buffer that holds
/// the argmax output. `None` when the backend is running on CPU.
struct GpuSampler {
    kernels: SamplingKernels,
    queue: CommandQueue,
    out_buf: Buffer,
    /// Held to allocate scratch buffers up front (and as a clone target if
    /// new buffers are ever needed beyond `MAX_PENALTY_ENTRIES`).
    device: MtlDevice,
    /// pre-allocated scratch buffers for the GPU
    /// penalty pass. Sized for the worst-case window (256 recent tokens
    /// → up to 256 unique repeat-penalty + 256 ngram match entries =
    /// 512 entries with margin). Per-step we only `memcpy` into the
    /// shared-storage buffer, no per-token Metal allocator round-trip.
    /// This removed the 1.2–1.8 ms outliers seen with per-call
    /// `new_buffer_with_data` (Metal allocator latency).
    scratch_idx_buf: Buffer,
    scratch_mul_buf: Buffer,
    scratch_capacity: usize,
}

const MAX_PENALTY_ENTRIES: usize = 512;

impl GpuSampler {
    fn new(device: &MtlDevice) -> Result<Self> {
        let kernels = SamplingKernels::new(device)?;
        let queue = device
            .new_command_queue()
            .map_err(|e| anyhow::anyhow!("new_command_queue: {e}"))?;
        let out_buf = SamplingKernels::alloc_u32_scalar(device);
        let scratch_idx_buf = device
            .new_buffer(
                MAX_PENALTY_ENTRIES * 4,
                lumen_metal::metal::MTLResourceOptions::StorageModeShared,
            )
            .map_err(|e| anyhow::anyhow!("scratch_idx_buf alloc: {e}"))?;
        let scratch_mul_buf = device
            .new_buffer(
                MAX_PENALTY_ENTRIES * 4,
                lumen_metal::metal::MTLResourceOptions::StorageModeShared,
            )
            .map_err(|e| anyhow::anyhow!("scratch_mul_buf alloc: {e}"))?;
        Ok(Self {
            kernels,
            queue,
            out_buf,
            device: device.clone(),
            scratch_idx_buf,
            scratch_mul_buf,
            scratch_capacity: MAX_PENALTY_ENTRIES,
        })
    }
}

use super::config::Qwen3_5MoeConfig;
use super::loader::ShardSet;
use super::model::Qwen3_5MoeTextModel;
use super::mtp_drafter::MtpDrafter;

/// Lookahead-decode summary captured from the most recent `generate_with_opts`
/// call. Populated only when `LUMEN_LOOKAHEAD_DECODE=1` was active for that
/// call; `None` otherwise. External A/B benches (e.g. `bench_lookahead_decode_ab`)
/// read this to compute avg_committed/step and observe pool saturation.
#[derive(Debug, Clone, Copy, Default)]
pub struct LookaheadStats {
    pub attempts: usize,
    pub full_accepts: usize,
    pub jacobi_accepted_total: usize,
    pub committed_total: usize,
    pub pool_len: usize,
    pub w: usize,
    pub g: usize,
    pub n: usize,
}

/// Qwen3.5-VL-MoE text model with tokenizer and generation loop.
pub struct Qwen35MoeBackend {
    model: Qwen3_5MoeTextModel,
    tokenizer: Tokenizer,
    device: Device,
    eos_tokens: Vec<u32>,
    model_id: String,
    /// GPU argmax sampler — `Some` when running on Metal, used by the greedy
    /// fast path in [`generate_with_opts`] (`temperature == 0` etc).
    sampler: Option<GpuSampler>,
    /// Per-decode-step wallclock ms from the most recent
    /// `generate_with_opts` call. Cleared at the start of each call.
    /// Excludes prefill cost (only the decode-loop steps are recorded).
    /// Used by external A/B benches (e.g. `bench_lever_h_ab`) to pool
    /// decode-only timings without amortizing prefill.
    last_decode_step_ms: Vec<f64>,
    /// Lookahead-decode summary from the most recent `generate_with_opts` call,
    /// or `None` if lookahead was disabled. Reset every call.
    last_lookahead_stats: Option<LookaheadStats>,
    /// MTP speculative drafter (llama.cpp PR #22673 port). `Some` after
    /// [`Self::try_enable_mtp`] succeeds — drives draft-then-verify decode for
    /// greedy generations. `None` for non-Qwen3.6 models or when
    /// `LUMEN_QWEN35_HF_ORIGINAL` isn't set to a snapshot carrying `mtp.*`.
    mtp: Option<MtpDrafter>,
}

impl Qwen35MoeBackend {
    /// Load the full text model from a local shard directory + fetch the tokenizer from HF.
    ///
    /// `model_id` is used for HF tokenizer lookup (e.g. `"mlx-community/Qwen3.6-35B-A3B-mxfp4"`).
    /// `shard_dir` must contain `config.json`, `model.safetensors.index.json`, and the shard
    /// files. On hosts with 64 GB RAM the MXFP4 weights stay resident on the Metal device
    /// (~19 GB); `lm_head` + `embed_tokens` dequantize to f32 (~4 GB combined).
    /// Load with both MXFP4 and 4-bit affine GPU contexts. The shard set will
    /// dispatch each projection to the appropriate GPU-resident wrapper based on
    /// its on-disk storage kind. Used by Qwen3.6-27B-MLX-4bit-style checkpoints
    /// which carry a uniform 4-bit affine quantization.
    ///
    /// `gpu_ctx` may be a freshly-constructed (unused) `MxFp4Context` when the
    /// checkpoint contains no MXFP4 weights — the shader compile cost (~50 ms)
    /// is amortized over the model lifetime. Future cleanup can elide it.
    pub fn load_with_affine4(
        model_id: &str,
        shard_dir: &Path,
        gpu_ctx: Arc<MxFp4Context>,
        affine4_ctx: Arc<Affine4Context>,
    ) -> Result<Self> {
        Self::load_inner(model_id, shard_dir, gpu_ctx, Some(affine4_ctx))
    }

    pub fn load(model_id: &str, shard_dir: &Path, gpu_ctx: Arc<MxFp4Context>) -> Result<Self> {
        Self::load_inner(model_id, shard_dir, gpu_ctx, None)
    }

    fn load_inner(
        model_id: &str,
        shard_dir: &Path,
        gpu_ctx: Arc<MxFp4Context>,
        affine4_ctx: Option<Arc<Affine4Context>>,
    ) -> Result<Self> {
        eprintln!(
            "Loading {model_id} (arch=qwen3_5_moe) from shards: {}",
            shard_dir.display()
        );

        // Resolve device: Metal (default) or CPU via DEVICE=cpu.
        let device = crate::loader::get_device()?;

        // Load config.json from the shard directory.
        let cfg_path = shard_dir.join("config.json");
        let cfg_str = std::fs::read_to_string(&cfg_path)
            .with_context(|| format!("reading {}", cfg_path.display()))?;
        let mut config: Qwen3_5MoeConfig = serde_json::from_str(&cfg_str)?;
        config.validate().map_err(|e| anyhow::anyhow!("{e}"))?;

        let n_layers = config.text_config.num_hidden_layers;
        let vocab = config.text_config.vocab_size;
        let head_dim = config.text_config.head_dim as usize;
        let n_kv_heads = config.text_config.num_key_value_heads;
        eprintln!(
            "  layers={n_layers}, hidden={}, heads={}, vocab={vocab}",
            config.text_config.hidden_size, config.text_config.num_attention_heads,
        );

        // Load weights through the GPU-aware ShardSet — MXFP4 stays on device.
        // When affine4_ctx is supplied, 4-bit affine projections also stay GPU-resident.
        let mut shards = ShardSet::open_with_gpu(shard_dir, config, gpu_ctx)
            .context("open shard set with GPU")?;
        if let Some(a4) = affine4_ctx {
            shards = shards.with_affine4_gpu(a4);
        }
        eprintln!("  shards open, loading {n_layers} layers into Metal...");
        let t_load = std::time::Instant::now();
        let mut model = shards
            .load_text_model(&device)
            .context("load_text_model failed")?;
        let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;
        eprintln!("  model loaded in {load_ms:.0}ms ({n_layers} layers)");

        // Enable vanilla per-layer KV caches on the 10 full-attention layers. `LUMEN_MAX_SEQ`
        // controls the initial capacity; the underlying `KvCache` reallocates on overflow.
        let max_seq: usize = std::env::var("LUMEN_MAX_SEQ")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8192);
        model.enable_kv_cache(max_seq);
        let n_full_attn = model.assign_tq_slots();
        eprintln!(
            "  KV cache enabled (max_seq={max_seq}), {n_full_attn} full-attn slots",
        );

        // TurboQuant compressed-KV backend, **default-OFF** for qwen3_5_moe.
        //
        // Reverted from default-ON after B4.1 production validation (2026-04-30):
        // single-step microbench (cos>0.98 at 4-bit) is **not** a quality
        // predictor for autoregressive decode. In production, accumulated drift
        // across 10 full-attn layers + 20+ decode steps collapses output to
        // garbage (token 0 = "!" spam). Verified via curl on long prompt:
        //   - TQ 4-bit: garbage from step 1
        //   - TQ 8-bit: ~22 coherent tokens, then garbage
        //   - TQ off:   coherent throughout
        // Speed parity with TQ off (100ms/step at long context) means TQ
        // brings no decode win to offset the quality loss for this arch.
        //
        // Opt-in: `TQ_ENABLE=1` (or any non-empty value). Override bits via
        // `TQ_BITS=N` (2/3/4/8). The legacy `TQ_DISABLE=1` flag is honoured
        // for callers that previously opted out.
        let legacy_disable = std::env::var("TQ_DISABLE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let tq_enable = std::env::var("TQ_ENABLE")
            .map(|v| !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(false);
        let tq_active = tq_enable && !legacy_disable;
        if tq_active {
            let bits: u32 = std::env::var("TQ_BITS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(4);
            let tq_cfg = lumen_core::config::TurboQuantConfig {
                bits,
                qjl_m: head_dim / 2,
                seed: 42,
                head_dim,
                lloyd_max_iter: 1000,
            };
            match lumen_metal::GpuCompressor::new(tq_cfg, n_full_attn, n_kv_heads, max_seq) {
                Ok(compressor) => {
                    model.set_compressed_kv(Box::new(compressor));
                    eprintln!(
                        "  TurboQuant enabled (bits={bits}, head_dim={head_dim}, \
                         kv_heads={n_kv_heads}, layers={n_full_attn}, max_seq={max_seq})",
                    );
                }
                Err(e) => eprintln!("  TurboQuant init failed, staying on vanilla KV: {e}"),
            }
        } else if legacy_disable {
            eprintln!("  TurboQuant disabled via TQ_DISABLE=1 (already default-off)");
        } else {
            eprintln!("  TurboQuant default-off (set TQ_ENABLE=1 to opt in)");
        }

        // Tokenizer — fetch from HF Hub (auto-cached).
        let tokenizer = crate::loader::load_tokenizer(model_id)
            .with_context(|| format!("tokenizer from {model_id}"))?;
        eprintln!(
            "  tokenizer loaded (vocab={})",
            tokenizer.get_vocab_size(false)
        );

        // Qwen3 EOS set. `<|im_end|>` = 151645, `<|endoftext|>` = 151643, plus common tool
        // stop tokens the shipped chat template uses. Any one of them terminates generation.
        let eos_tokens = resolve_eos_tokens(&tokenizer);

        // GPU argmax sampler — only when device is Metal. CPU fallback path stays the
        // CPU `sample_token_cpu_full` route.
        let sampler = if device.is_metal() {
            match MtlDevice::system_default() {
                Some(d) => match GpuSampler::new(&d) {
                    Ok(s) => Some(s),
                    Err(e) => {
                        eprintln!("  GPU sampler init failed: {e} — falling back to CPU sampling");
                        None
                    }
                },
                None => None,
            }
        } else {
            None
        };

        Ok(Self {
            model,
            tokenizer,
            device,
            eos_tokens,
            model_id: model_id.to_string(),
            sampler,
            last_decode_step_ms: Vec::new(),
            last_lookahead_stats: None,
            mtp: None,
        })
    }

    /// Try to enable MTP speculative decoding by loading the `mtp.*` head from the HF
    /// original snapshot at `LUMEN_QWEN35_HF_ORIGINAL`. Returns `Ok(false)` when the env
    /// var is unset or the config has no MTP head — callers should treat that as
    /// "not enabled" rather than an error. `Ok(true)` on success (drafter constructed,
    /// trunk `capture_h_pre_norm` enabled). Hard error if the env var resolves to a
    /// malformed checkpoint.
    ///
    /// `shard_dir` is still required because we re-open the trunk shard set to inherit
    /// its config + classification; the MTP weights themselves come from the HF
    /// original path, not from this directory.
    ///
    /// Idempotent — calling twice with the same shard_dir is a no-op on the
    /// second call.
    pub fn try_enable_mtp(&mut self, shard_dir: &Path) -> Result<bool> {
        use crate::qwen3_5_moe::mtp_drafter::MtpDrafter;
        if self.mtp.is_some() {
            return Ok(true);
        }

        // Re-open a ShardSet purely to access the sidecar. The mmaps are cheap
        // (kernel page cache), classification re-runs on the index (~ms range).
        // The trunk model is already loaded — we just need MTP weights.
        let cfg_path = shard_dir.join("config.json");
        let cfg_str = std::fs::read_to_string(&cfg_path)
            .with_context(|| format!("reading {} for MTP enable", cfg_path.display()))?;
        let mut config: Qwen3_5MoeConfig = serde_json::from_str(&cfg_str)?;
        config.validate().map_err(|e| anyhow::anyhow!("{e}"))?;

        // Guard: MTP only makes sense when the config declares an MTP head AND
        // the MLP variant is one MtpBlock supports (Dense for Qwen3.6-27B).
        if config.text_config.mtp_num_hidden_layers == 0 {
            return Ok(false);
        }

        let shards = ShardSet::open(shard_dir, config)
            .context("open shard set for MTP HF-native load")?;
        let Some(block) = shards.load_mtp_block(&self.device)? else {
            return Ok(false);
        };

        let n_embd = self.model.embed_tokens().embeddings().dims()[1];
        self.mtp = Some(MtpDrafter::new(block, n_embd));
        // Trunk must emit pre-norm hidden so process_after_trunk / verify can
        // mirror the trajectory into MTP's KV.
        self.model.set_capture_h_pre_norm(true);
        Ok(true)
    }

    /// Read-only access to the MTP drafter (if enabled).
    pub fn mtp_drafter(&self) -> Option<&crate::qwen3_5_moe::mtp_drafter::MtpDrafter> {
        self.mtp.as_ref()
    }

    /// Mutable access to the MTP drafter (for the engine's draft / verify loop).
    pub fn mtp_drafter_mut(
        &mut self,
    ) -> Option<&mut crate::qwen3_5_moe::mtp_drafter::MtpDrafter> {
        self.mtp.as_mut()
    }

    /// Whether MTP speculative decoding is currently enabled on this backend.
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Per-decode-step wallclock ms recorded by the most recent
    /// `generate_with_opts` call. Excludes prefill (only the decode-loop
    /// steps are recorded). For pooled A/B benchmarks.
    pub fn last_decode_step_ms(&self) -> &[f64] {
        &self.last_decode_step_ms
    }

    /// Lookahead-decode summary from the most recent `generate_with_opts`
    /// call. `Some(stats)` only when `LUMEN_LOOKAHEAD_DECODE=1` was active;
    /// otherwise `None`. Reset on every call.
    pub fn last_lookahead_stats(&self) -> Option<&LookaheadStats> {
        self.last_lookahead_stats.as_ref()
    }

    /// Mutable access to the underlying text model. Used by the hidden-state dump
    /// example to call `forward_with_offset` directly.
    pub fn model_mut(&mut self) -> &mut Qwen3_5MoeTextModel {
        &mut self.model
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// EOS token IDs for this model (used by the batched engine for termination checks).
    pub fn eos_tokens(&self) -> &[u32] {
        &self.eos_tokens
    }

    /// Build tokenized prompt from a sequence of `(role, content)` message pairs.
    /// Mirrors `GemmaGgufModel::build_chat_input` so the engine can dispatch uniformly.
    pub fn build_chat_input(&self, messages: &[(String, String)], thinking: bool) -> Result<Vec<u32>> {
        let prompt = format_qwen3_chat(messages, thinking);
        self.encode(&prompt)
    }

    // ── Per-sequence lifecycle (for concurrent streaming) ────────────────────────

    /// Allocate KV cache slots for a new streaming sequence.
    /// Call before the first prefill. Must NOT call `reset_cache()` after this.
    /// If MTP is enabled (see [`Self::try_enable_mtp`]) the MTP block's KV
    /// cache for `seq_id` is initialised in lockstep — required so the very
    /// first `mtp_step` call has a populated cache to advance.
    pub fn init_sequence(&mut self, seq_id: u64) {
        self.model.init_sequence(seq_id);
        if let Some(mtp) = self.mtp.as_mut() {
            // Match the trunk's max_seq_len. We resolve it the same way the
            // load path does — single source of truth for the cache capacity.
            let max_seq: usize = std::env::var("LUMEN_MAX_SEQ")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(8192);
            mtp.init_sequence(seq_id, max_seq, &self.device);
        }
    }

    /// Free KV cache memory for a completed sequence (both trunk + MTP).
    pub fn remove_sequence(&mut self, seq_id: u64) {
        self.model.remove_sequence(seq_id);
        if let Some(mtp) = self.mtp.as_mut() {
            mtp.remove_sequence(seq_id);
        }
    }

    /// Mirror a completed trunk prefill into the MTP block's KV cache.
    ///
    /// Must be called exactly once per sequence, **after** the trunk's prefill
    /// `forward_with_offset` returns and **before** the first `mtp_step`. The
    /// trunk's `last_h_pre_norm` must still hold the prefill hidden trajectory
    /// (i.e. no other forward calls between prefill and this).
    ///
    /// `prefill_tokens` are the token IDs fed to the trunk during prefill
    /// (typically `prompt[..len-1]`). On exit MTP's KV is at position
    /// `prefill_tokens.len()` — same as the trunk's — and the drafter's
    /// `pending_h` is seeded with the final prefill hidden.
    ///
    /// No-op when MTP is not enabled.
    pub fn mirror_prefill_into_mtp(
        &mut self,
        seq_id: u64,
        prefill_tokens: &[u32],
    ) -> Result<()> {
        let Self { model, mtp, .. } = self;
        let Some(drafter) = mtp.as_mut() else {
            return Ok(()); // MTP not enabled — silently skip.
        };
        if prefill_tokens.is_empty() {
            return Ok(());
        }
        let h_pre = model
            .take_h_pre_norm()
            .ok_or_else(|| anyhow::anyhow!("mirror_prefill_into_mtp: trunk had no h_pre_norm — \
                ensure prefill ran with capture_h_pre_norm enabled"))?;
        drafter
            .process_after_trunk(
                seq_id,
                prefill_tokens,
                &h_pre,
                model.embed_tokens(),
                model.final_norm(),
                model.lm_head(),
            )
            .map_err(|e| anyhow::anyhow!("mirror_prefill_into_mtp: {e}"))?;
        Ok(())
    }

    /// One MTP-accelerated decode step: draft `n_max` tokens, verify against
    /// the trunk, accept the longest greedy-matching prefix, and return the
    /// committed tokens.
    ///
    /// Returns `(committed, n_drafted, n_accepted)`:
    /// * `committed` — `[trunk_sample, accepted_drafts...]`, length `1 + n_accepted`.
    /// * `n_drafted` — always equals `n_max` on the happy path (used for stats).
    /// * `n_accepted` — ∈ `0..=n_max`.
    ///
    /// Errors propagate from any inner forward / sampler / drafter call. On
    /// error the backend's KV may be in a partial state — the caller should
    /// drop the sequence rather than retry.
    ///
    /// Requires:
    /// - MTP enabled ([`Self::has_mtp`] is true)
    /// - Sequence initialised via [`Self::init_sequence`] (+ optional
    ///   [`Self::mirror_prefill_into_mtp`] for the first step)
    /// - Trunk `capture_h_pre_norm` enabled (auto-on after `try_enable_mtp`)
    ///
    /// Greedy-only — caller must NOT invoke this when temperature > 0 or
    /// any sampling penalty is active. Verification uses argmax matching
    /// which presupposes the trunk would have sampled the same token.
    pub fn mtp_step(
        &mut self,
        seq_id: u64,
        last_token: u32,
        position: usize,
        n_max: usize,
    ) -> Result<(Vec<u32>, usize, usize)> {
        use candle_core::D;

        if n_max == 0 {
            // Degenerate: just do a regular single-token decode and return it.
            let next = self.decode_step_single(seq_id, last_token, position)?;
            return Ok((vec![next], 0, 0));
        }

        // ── Step A: trunk decode 1 token at `position`, capture h_pre ────────
        self.model.set_current_seq_id(seq_id);
        let input = Tensor::new(&[last_token], &self.device)?.unsqueeze(0)?; // [1, 1]
        let logits = self
            .model
            .forward_with_offset(&input, position)
            .map_err(|e| anyhow::anyhow!("mtp_step trunk decode: {e}"))?; // [1, 1, vocab]
        let next_tok = logits
            .squeeze(0)?
            .squeeze(0)?
            .argmax(D::Minus1)?
            .to_scalar::<u32>()?;
        if std::env::var("LUMEN_DECODE_TRACE").is_ok() {
            eprintln!(
                "[mtp_stepA_trace] pos={position} last_tok={last_token} next_tok={next_tok}"
            );
        }
        // Trunk top-5 at this position — for comparison against MTP block top-5.
        if std::env::var("LUMEN_MTP_BLOCK_TRACE").is_ok() {
            let row = logits.squeeze(0)?.squeeze(0)?.to_dtype(candle_core::DType::F32)?;
            let v: Vec<f32> = row.to_vec1()?;
            let mut idx: Vec<usize> = (0..v.len()).collect();
            idx.sort_by(|a, b| v[*b].partial_cmp(&v[*a]).unwrap_or(std::cmp::Ordering::Equal));
            let top5: Vec<(usize, f32)> = idx.iter().take(5).map(|&i| (i, v[i])).collect();
            eprintln!("[trunk_top5_stepA] pos={position}: {top5:?}");
        }
        let h_pre_decode = self.model.take_h_pre_norm().ok_or_else(|| {
            anyhow::anyhow!("mtp_step: trunk forward produced no h_pre_norm (capture flag off?)")
        })?; // [1, 1, hidden]

        // ── Step B: mirror the decode step into MTP KV + draft n_max tokens ──
        // Split borrow: mtp drafter is &mut, trunk model components are &.
        let Self { model, mtp, .. } = self;
        let drafter = mtp
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("mtp_step called without MTP enabled"))?;
        drafter
            .process_after_trunk(
                seq_id,
                &[last_token],
                &h_pre_decode,
                model.embed_tokens(),
                model.final_norm(),
                model.lm_head(),
            )
            .map_err(|e| anyhow::anyhow!("mtp_step process_after_trunk: {e}"))?;
        let drafts = drafter
            .draft(
                seq_id,
                next_tok,
                n_max,
                model.embed_tokens(),
                model.final_norm(),
                model.lm_head(),
            )
            .map_err(|e| anyhow::anyhow!("mtp_step draft: {e}"))?;
        let n_drafted = drafts.len();
        // NLL drops the `drafter` (&mut Mtp) + `model` (&Embedding/&RmsNorm/
        // &ProjLinear) borrows here as their last use was above. We can now
        // mut-borrow `self.model` for the verify forward.

        // ── Step C: trunk verify on [next_tok, ...drafts] at position+1 ──────
        // Snapshot before verify so we can rebuild trunk KV if any draft is
        // rejected (Mamba SSM state isn't simply truncatable — must re-evolve).
        let snapshot = self
            .model
            .snapshot_state()
            .map_err(|e| anyhow::anyhow!("mtp_step snapshot: {e}"))?;

        let mut verify_seq = Vec::with_capacity(1 + n_drafted);
        verify_seq.push(next_tok);
        verify_seq.extend_from_slice(&drafts);
        let verify_input = Tensor::new(verify_seq.as_slice(), &self.device)?.unsqueeze(0)?; // [1, 1+N]
        let verify_logits = self
            .model
            .forward_with_offset(&verify_input, position + 1)
            .map_err(|e| anyhow::anyhow!("mtp_step verify forward: {e}"))?; // [1, 1+N, vocab]
        let verify_h_pre = self.model.take_h_pre_norm().ok_or_else(|| {
            anyhow::anyhow!("mtp_step: trunk verify produced no h_pre_norm")
        })?; // [1, 1+N, hidden]

        // ── Step D: greedy match → n_accepted ────────────────────────────────
        // Target argmax per verify row: index i predicts the (committed +i+1)-th token.
        // verify_logits[0, i, :] argmax should equal drafts[i] to accept draft #i.
        let target_argmax = verify_logits
            .argmax(D::Minus1)
            .map_err(|e| anyhow::anyhow!("mtp_step verify argmax: {e}"))?
            .squeeze(0)
            .map_err(|e| anyhow::anyhow!("mtp_step squeeze: {e}"))?
            .to_vec1::<u32>()
            .map_err(|e| anyhow::anyhow!("mtp_step to_vec1: {e}"))?; // length 1+N
        let n_accepted = target_argmax
            .iter()
            .take(n_drafted)
            .zip(drafts.iter())
            .take_while(|(t, d)| **t == **d)
            .count();

        if std::env::var("LUMEN_MTP_TRACE").is_ok() {
            eprintln!(
                "[mtp_trace] pos={position} next_tok={next_tok} drafts={drafts:?} \
                 target_argmax={target_argmax:?} n_accepted={n_accepted}",
            );
        }

        // ── Step E: rebuild trunk KV to position + 1 + n_accepted ────────────
        // Contract: after mtp_step, trunk KV holds positions [0, position+n_accepted]
        // (size = position+1+n_accepted). The LAST committed token
        // (committed[n_accepted] = drafts[n_accepted-1] for n_acc≥1, or
        // next_tok for n_acc=0) is intentionally NOT written here — the next
        // mtp_step's Step A will consume it via `forward([last_token])` at
        // offset position+1+n_accepted, which matches the engine's
        // `seq.position += committed_len` increment.
        //
        // **Why skip the last write**: Mamba SSM layers in the hybrid
        // architecture advance their recurrent state on every input. If
        // rebuild writes 1+n_accepted tokens AND the next Step A re-consumes
        // the last one at the same offset, the SSM evolves twice for that
        // token → trunk argmax diverges (manifests as MTP-on vs MTP-off
        // greedy output differing). GQA attention is idempotent on
        // overwrite, but SSM is not — so we MUST write exactly n_accepted
        // tokens here, leaving one for Step A.
        //
        // Always restore + rebuild — even for n_accepted == n_drafted —
        // because the verify forward wrote 1+n_drafted tokens, one too
        // many under the new contract.
        self.model
            .restore_state(&snapshot)
            .map_err(|e| anyhow::anyhow!("mtp_step restore_state: {e}"))?;
        if n_accepted > 0 {
            // Write [next_tok, drafts[0..n_accepted-1]] (length n_accepted)
            // at offset position+1. The last committed token
            // drafts[n_accepted-1] is reserved for the next Step A.
            let kept = &verify_seq[..n_accepted];
            let kept_input = Tensor::new(kept, &self.device)?.unsqueeze(0)?;
            let _ = self
                .model
                .forward_with_offset(&kept_input, position + 1)
                .map_err(|e| anyhow::anyhow!("mtp_step rebuild forward: {e}"))?;
            let _ = self.model.take_h_pre_norm();
        }
        // n_accepted == 0: nothing to rebuild — restore already left trunk KV
        // at state-after-Step-A. Next Step A will write `next_tok` at
        // offset position+1.
        let _ = &verify_h_pre; // sliced below in the accept-side mirror

        // ── Step F: mirror the COMMITTED suffix into MTP, then accept ────────
        // The committed sequence is `[next_tok, drafts[..n_accepted]]` of
        // length `1 + n_accepted`. We need MTP KV to reflect those positions.
        // Currently MTP KV (after step B's process + step's draft) sits at
        // position + 1 + n_drafted with rejected-tail entries. `accept`
        // truncates it to position + 1 + n_accepted via the drafter's own
        // bookkeeping.
        //
        // We also need MTP to ingest the verify-batch's h_pre for the
        // committed suffix so its `pending_h` after this step reflects the
        // trunk's last committed hidden — not MTP's own draft hidden.
        let committed_len = 1 + n_accepted;
        let n_committed_after = position + 1 + committed_len; // total trunk pos including this step
        let Self { model, mtp, .. } = self;
        let drafter = mtp
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("mtp_step: MTP vanished mid-step"))?;
        // First: rewind MTP KV + reset pending_h from verify_h (drafter knows
        // verify_h from its OWN process_after_trunk call earlier — wait, that
        // one only ingested the single decode-step hidden. We need to also
        // ingest the verify suffix.)
        //
        // Simpler model: re-mirror the committed suffix from scratch.
        //   1. mtp.accept(seq, n_accepted=0, n_committed_after=position+1) → rewind to position+1 and reset pending_h
        //   2. process_after_trunk(seq, &[next_tok, ...drafts[..n_accepted]], verify_h_pre_committed_slice)
        // Net: MTP KV ends at position+1+committed_len = n_committed_after, pending_h = verify_h's last committed row.
        drafter
            .accept(seq_id, 0, position + 1)
            .map_err(|e| anyhow::anyhow!("mtp_step pre-mirror accept: {e}"))?;
        if committed_len > 0 {
            // Slice verify_h_pre to the committed rows: [1, committed_len, hidden].
            let h_committed = verify_h_pre
                .narrow(1, 0, committed_len)
                .map_err(|e| anyhow::anyhow!("mtp_step h slice: {e}"))?;
            let committed_tokens = &verify_seq[..committed_len];
            drafter
                .process_after_trunk(
                    seq_id,
                    committed_tokens,
                    &h_committed,
                    model.embed_tokens(),
                    model.final_norm(),
                    model.lm_head(),
                )
                .map_err(|e| anyhow::anyhow!("mtp_step suffix mirror: {e}"))?;
        }
        let _ = n_committed_after;

        Ok((verify_seq[..committed_len].to_vec(), n_drafted, n_accepted))
    }

    /// Prefill under `seq_id` and return `(first_generated_token, position)`.
    /// `position` = `input_ids.len()` (next decode step's `seqlen_offset`).
    /// Does NOT call `reset_cache()` globally — safe for concurrent sequences.
    pub fn prefill_sequence(&mut self, seq_id: u64, input_ids: &[u32]) -> Result<(u32, usize)> {
        use candle_core::D;
        if input_ids.is_empty() {
            anyhow::bail!("prefill_sequence: empty input_ids for seq {seq_id}");
        }
        self.model.set_current_seq_id(seq_id);
        let prompt = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;
        let logits_all = self.model.forward_with_offset(&prompt, 0)
            .map_err(|e| anyhow::anyhow!("prefill_sequence forward: {e}"))?; // [1, S, vocab]
        let last_logits = logits_all
            .narrow(D::Minus2, input_ids.len() - 1, 1)
            .and_then(|t| t.squeeze(0))
            .and_then(|t| t.squeeze(0))
            .map_err(|e| anyhow::anyhow!("prefill_sequence narrow: {e}"))?; // [vocab]
        let first_token = last_logits
            .argmax(D::Minus1)
            .and_then(|t| t.to_scalar::<u32>())
            .map_err(|e| anyhow::anyhow!("prefill_sequence argmax: {e}"))?;
        Ok((first_token, input_ids.len()))
    }

    /// Pre-flight helper: raw logits from a forward at the given offset.
    /// Used by the speculative-decode N-step equivalence test (R3 pre-flight).
    /// Returns `[1, S, vocab]` where `S = input_ids.len()`.
    /// Mutates the seq's KV cache (advances by `S` positions).
    pub fn forward_logits_at_offset(
        &mut self,
        seq_id: u64,
        input_ids: &[u32],
        offset: usize,
    ) -> Result<Tensor> {
        if input_ids.is_empty() {
            anyhow::bail!("forward_logits_at_offset: empty input_ids");
        }
        self.model.set_current_seq_id(seq_id);
        let input = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;
        self.model
            .forward_with_offset(&input, offset)
            .map_err(|e| anyhow::anyhow!("forward_logits_at_offset: {e}"))
    }

    /// Single-seq decode step (N=1 fast path). Returns next token.
    /// Caller must pass `seq_id`, `last_token`, and `position` = past KV length.
    pub fn decode_step_single(
        &mut self,
        seq_id: u64,
        last_token: u32,
        position: usize,
    ) -> Result<u32> {
        use candle_core::D;
        self.model.set_current_seq_id(seq_id);
        let input = Tensor::new(&[last_token], &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward_with_offset(&input, position)
            .map_err(|e| anyhow::anyhow!("decode_step_single: {e}"))?; // [1, 1, vocab]
        let next = logits.squeeze(0)?.squeeze(0)?
            .argmax(D::Minus1)?
            .to_scalar::<u32>()
            .map_err(|e| anyhow::anyhow!("decode_step_single argmax: {e}"))?;
        if std::env::var("LUMEN_DECODE_TRACE").is_ok() {
            eprintln!(
                "[decode_single_trace] pos={position} last_tok={last_token} next_tok={next}"
            );
        }
        Ok(next)
    }

    /// Multi-seq decode (N≥2 path). Returns `[B, vocab]` logit tensor for CPU sampling.
    ///
    /// Default: Phase 1 sequential per-seq (`forward_batch_decode_seqs`).
    /// `BATCHED_MOE=1` → Phase 2 batched MoE (`_v2`, currently a regression).
    /// `CB_PHASE0=1` → Phase 0 diagnostic (`_phase0`): per-seq layer forward
    /// like v1, but batched final_norm + lm_head. Used to isolate the
    /// per-layer `Tensor::cat` overhead from the MoE bl=2 inefficiency.
    /// `BATCHED_MOE=1` takes precedence over `CB_PHASE0=1`.
    pub fn decode_step_batch(
        &mut self,
        seq_ids: &[u64],
        last_tokens: &[u32],
        positions: &[usize],
    ) -> Result<Tensor> {
        let use_v2 = std::env::var("BATCHED_MOE").map(|v| v == "1").unwrap_or(false);
        let use_phase0 = std::env::var("CB_PHASE0").map(|v| v == "1").unwrap_or(false);
        if use_v2 {
            self.model
                .forward_batch_decode_seqs_v2(last_tokens, seq_ids, positions)
                .map_err(|e| anyhow::anyhow!("decode_step_batch v2: {e}"))
        } else if use_phase0 {
            self.model
                .forward_batch_decode_seqs_phase0(last_tokens, seq_ids, positions)
                .map_err(|e| anyhow::anyhow!("decode_step_batch phase0: {e}"))
        } else {
            self.model
                .forward_batch_decode_seqs(last_tokens, seq_ids, positions)
                .map_err(|e| anyhow::anyhow!("decode_step_batch: {e}"))
        }
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenizer encode: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|e| anyhow::anyhow!("tokenizer decode: {e}"))
    }

    /// Autoregressive generation using the per-layer KV append cache.
    ///
    /// - Prefill: one forward on the full prompt, `seqlen_offset=0`. Every full-attn layer's
    ///   `KvCache` accumulates the rotated K/V.
    /// - Decode: one forward per token on `[next]`, `seqlen_offset=past_len`. The cache appends
    ///   the new K/V to the running concatenation, so attention runs over O(S) keys without
    ///   re-projecting the prompt.
    ///
    /// Returns **only** the generated tokens (not prompt). Stops on any EOS token or when
    /// `max_new_tokens` is reached.
    ///
    /// Note: the 30 `GatedDeltaNet` (SSM) layers still run stateless per call — their
    /// recurrent-state passthrough is the follow-up.
    pub fn generate(
        &mut self,
        input_ids: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
    ) -> Result<Vec<u32>> {
        self.generate_with_opts(input_ids, max_new_tokens, temperature, top_p, 20, 1.1)
    }

    /// Full-knobs autoregressive generation.
    ///
    /// - `top_k`: keep the K largest logits (others → -inf). `0` disables. Qwen3 ships
    ///   `top_k=20` in `generation_config.json`.
    /// - `repeat_penalty`: divides recently-seen tokens' logits. 1.0 disables.
    ///
    /// Sampling runs on CPU since the logits vector is already f32 after narrow. The
    /// forward itself stays on Metal.
    pub fn generate_with_opts(
        &mut self,
        input_ids: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        repeat_penalty: f32,
    ) -> Result<Vec<u32>> {
        if input_ids.is_empty() {
            anyhow::bail!("empty input_ids");
        }
        self.last_decode_step_ms.clear();
        self.last_lookahead_stats = None;
        self.model.reset_cache();

        // native-path smoke test. When `LUMEN_QWEN35MOE_NATIVE=1`,
        // round-trip the input_ids tensor through the native bridge to verify
        // the new context + buffer plumbing is wired before we replace any
        // real layers. The Candle forward below is unaffected.
        if crate::qwen3_5_moe_native::native_forward_enabled() {
            let nctx = crate::qwen3_5_moe_native::NativeContext::new()?;
            let probe = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;
            let _ = crate::qwen3_5_moe_native::placeholder_forward(&nctx, &probe)?;
            eprintln!(
                "  [native] Phase A.0 routing OK (probe shape=[1, {}])",
                input_ids.len()
            );
        }

        // ── Prefill ───────────────────────────────────────────────────────
        let _ = lumen_metal::mxfp4_linear::take_trace_counts();
        let t_prefill = std::time::Instant::now();
        let prompt_tensor = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;
        let logits_all = self.model.forward_with_offset(&prompt_tensor, 0)?; // [1, S, vocab]
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
        let past_len = input_ids.len();
        let (zc, cp) = lumen_metal::mxfp4_linear::take_trace_counts();

        // Full generation history (prompt + generated). Used for repeat penalty + n-gram.
        let mut history: Vec<u32> = Vec::with_capacity(input_ids.len() + max_new_tokens);
        history.extend_from_slice(input_ids);

        // Greedy fast path: at `temperature == 0` the sampler degenerates to
        // argmax over the (penalized) logits row. We dispatch GPU penalty +
        // argmax kernels and pull a single u32 (4 B) back, avoiding the 248K-
        // vocab host transfer entirely.
        //
        // the prior gate also required `top_k == 0` and
        // `repeat_penalty == 1.0`; with the GPU penalty kernel and the fact
        // that `top_k`/`top_p` are irrelevant at temp=0 (argmax wins after
        // any monotonic top-k filter), we relax the gate to just `temp == 0`.
        // `sample_last_token` now carries `repeat_penalty` and `history` into
        // the GPU pre-pass so rep+ngram penalties apply identically to the
        // CPU path.
        let greedy_fast = temperature == 0.0 && self.sampler.is_some();
        let _ = top_p; // `top_p` is irrelevant when temperature == 0 → silence "unused".

        // Sample first generated token from the last prompt position.
        let last = logits_all.narrow(D::Minus2, past_len - 1, 1)?;
        let mut next = self.sample_last_token(
            &last,
            greedy_fast,
            temperature,
            top_p,
            top_k,
            repeat_penalty,
            &history,
        )?;
        history.push(next);
        let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
        generated.push(next);
        eprintln!(
            "  prefill: {past_len} tokens in {prefill_ms:.0}ms -> tok={next} \
             [mxfp4 matmul: zero-copy={zc} cpu_fallback={cp}]",
        );

        if self.eos_tokens.contains(&next) {
            return Ok(generated);
        }

        // ── Decode ─────────────────────────────────────────────────────────
        // Fine-grained per-step timing gated by `LUMEN_DECODE_TIMING=1`. Each marker syncs
        // the device first so the reported ms reflect actual GPU completion. Used to size the
        // ROI of A.6 (GPU sampler + lm_head dispatch fusion); off by default since each sync
        // costs an extra commit/wait pair.
        let decode_timing = std::env::var("LUMEN_DECODE_TIMING")
            .map(|v| v == "1")
            .unwrap_or(false);

        // Lookahead Decoding (Phase L.2): Jacobi window + n-gram pool. Gated by
        // `LUMEN_LOOKAHEAD_DECODE=1` and only in the greedy fast path. Mutually
        // exclusive with `LUMEN_NGRAM_SPEC`: when both are set, lookahead wins
        // and the n-gram path is skipped. Algorithm: at each step we forward
        // `[next, j_0..j_{W-1}, guess_packs]` and greedily accept the longest
        // matching Jacobi prefix; the +1 free token (post-accept) is committed
        // unconditionally. Snapshot/restore lifecycle reuses the same pattern
        // as the n-gram path.
        let lookahead_enabled = std::env::var("LUMEN_LOOKAHEAD_DECODE")
            .map(|v| v == "1")
            .unwrap_or(false)
            && greedy_fast;
        let lookahead_w: usize = std::env::var("LUMEN_LOOKAHEAD_W")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4usize)
            .max(1);
        let lookahead_g: usize = std::env::var("LUMEN_LOOKAHEAD_G")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0usize); // L.2 default: G=0 (Jacobi only). G>0 wired in L.3.
        let lookahead_n: usize = std::env::var("LUMEN_LOOKAHEAD_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3usize)
            .max(2);
        let mut lookahead_proposer: Option<crate::lookahead::LookaheadProposer> =
            if lookahead_enabled {
                let cfg = crate::lookahead::LookaheadConfig {
                    window: lookahead_w,
                    guesses: lookahead_g,
                    ngram: lookahead_n,
                    pool_max: 4096,
                    stable_threshold: 3,
                };
                let mut p = crate::lookahead::LookaheadProposer::new(cfg);
                p.seed(next);
                Some(p)
            } else {
                None
            };
        let mut lookahead_attempts = 0usize;
        let mut lookahead_jacobi_accepted_total = 0usize;
        let mut lookahead_committed_total = 0usize;
        let mut lookahead_full_accepts = 0usize;

        // Spec-decode (Phase S.1): N-gram-driven draft + greedy verify. Gated by
        // `LUMEN_NGRAM_SPEC=1` and only active in the greedy fast path (sampling
        // verify is Phase S.2). On reject we restore the layer state captured
        // before the verify forward and replay the j+1 accepted tokens.
        // Suppressed when lookahead is enabled (mutual exclusion).
        let spec_enabled = std::env::var("LUMEN_NGRAM_SPEC")
            .map(|v| v == "1")
            .unwrap_or(false)
            && greedy_fast
            && !lookahead_enabled;
        let spec_k = std::env::var("LUMEN_SPEC_K")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4usize)
            .max(1);
        let spec_n = std::env::var("LUMEN_SPEC_NGRAM")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3usize)
            .max(2);
        // Min suffix length for ngram_lookup. Raised to 3 (was 2) since length-2
        // matches were the dominant noise source in 35B measurements: lots of
        // attempts with avg j < 1, dragging net throughput below baseline.
        let spec_min_n = std::env::var("LUMEN_SPEC_MIN_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3usize)
            .max(2)
            .min(spec_n);
        // Cost-aware gating: track recent j values; if rolling average drops
        // below threshold, enter cooldown (skip ngram_lookup entirely). Verify
        // forward costs ~2.5x single-token decode, so break-even requires avg
        // j >= ~2.5; we use threshold 2.0 to keep some slack.
        let spec_gate = std::env::var("LUMEN_SPEC_GATE")
            .map(|v| v != "0")
            .unwrap_or(true);
        let gate_window: usize = 8;
        let gate_threshold: f64 = 2.0;
        let gate_cooldown: usize = 30; // tokens to skip after a bad streak
        let mut recent_j: std::collections::VecDeque<u8> =
            std::collections::VecDeque::with_capacity(gate_window);
        let mut cooldown_until: usize = 0;

        let mut spec_attempts = 0usize;
        let mut spec_accepted_total = 0usize;
        let mut spec_drafts_total = 0usize;
        let mut spec_full_accepts = 0usize;
        let mut spec_skipped_by_gate = 0usize;

        let mut step = 1usize;
        while step < max_new_tokens {
            let _ = lumen_metal::mxfp4_linear::take_trace_counts();
            let t_step = std::time::Instant::now();
            let offset = past_len + step - 1;

            // Lookahead Decoding path (Phase L.2). Mutually exclusive with the
            // n-gram path. Builds `[next, jacobi_W, guess_packs]` of length
            // `1 + W + G·W` and greedy-accepts the longest matching Jacobi
            // prefix. Pool gathering runs after each commit so the n-gram pool
            // grows over the trajectory; G>0 paths consume those entries via
            // `LookaheadProposer::propose`.
            let lookahead_proposal = if lookahead_enabled {
                lookahead_proposer.as_ref().and_then(|p| p.propose(&history))
            } else {
                None
            };

            if let Some(proposal) = lookahead_proposal {
                lookahead_attempts += 1;
                let w = proposal.jacobi.len();
                let g_count = proposal.guesses.len();
                debug_assert!(w >= 1);

                // Snapshot every layer's recurrent / append-cache state so a
                // partial accept (and any G>0 path) can roll back.
                let snap = self
                    .model
                    .snapshot_state()
                    .map_err(|e| anyhow::anyhow!("snapshot_state: {e}"))?;

                // verify_in = [next, jacobi_W, guess_packs flat] of length
                // `1 + W + g_count·W`.
                let mut verify_in: Vec<u32> = Vec::with_capacity(1 + w + g_count * w);
                verify_in.push(next);
                verify_in.extend_from_slice(&proposal.jacobi);
                for g in &proposal.guesses {
                    verify_in.extend_from_slice(g);
                }
                let v_tensor =
                    Tensor::new(verify_in.as_slice(), &self.device)?.unsqueeze(0)?;
                let logits = self.model.forward_with_offset(&v_tensor, offset)?;
                let argmax = logits.argmax(D::Minus1)?.flatten_all()?.to_vec1::<u32>()?;
                debug_assert_eq!(argmax.len(), verify_in.len());

                // Standard next-token-shift convention: argmax[i] is the
                // model's prediction for the input position i+1's slot in
                // the sequence. With input[1+i] = jacobi[i], argmax[i] is
                // comparable to jacobi[i] for greedy acceptance, and
                // argmax[W] is the +1 free post-Jacobi prediction.
                let jacobi_accept = lookahead_proposer
                    .as_ref()
                    .map(|p| p.greedy_jacobi_accept(&argmax))
                    .unwrap_or(0);

                // L.2 G=0 default path: ignore guess packs (no acceptance
                // attempted). L.3 will add per-pack acceptance and select the
                // best Jacobi+pack combination.

                let new_next: u32;
                let committed_count = jacobi_accept + 1;
                let restore_needed = jacobi_accept < w || g_count > 0;

                if jacobi_accept == w {
                    lookahead_full_accepts += 1;
                    for j in &proposal.jacobi {
                        history.push(*j);
                        generated.push(*j);
                    }
                    new_next = argmax[w]; // post-Jacobi free token
                    history.push(new_next);
                    generated.push(new_next);
                } else {
                    for j in &proposal.jacobi[..jacobi_accept] {
                        history.push(*j);
                        generated.push(*j);
                    }
                    new_next = argmax[jacobi_accept];
                    history.push(new_next);
                    generated.push(new_next);
                }

                if restore_needed {
                    self.model
                        .restore_state(&snap)
                        .map_err(|e| anyhow::anyhow!("restore_state: {e}"))?;
                    let replay_len = jacobi_accept + 1;
                    let mut fixup_in: Vec<u32> = Vec::with_capacity(replay_len);
                    fixup_in.push(next);
                    fixup_in.extend_from_slice(&proposal.jacobi[..jacobi_accept]);
                    let f_tensor =
                        Tensor::new(fixup_in.as_slice(), &self.device)?.unsqueeze(0)?;
                    let _ = self.model.forward_with_offset(&f_tensor, offset)?;
                }

                lookahead_jacobi_accepted_total += jacobi_accept;
                lookahead_committed_total += committed_count;

                if let Some(p) = lookahead_proposer.as_mut() {
                    p.update(&argmax, committed_count);
                    p.observe_committed(&history);
                }

                next = new_next;
                step += committed_count;

                let step_ms = t_step.elapsed().as_secs_f64() * 1000.0;
                self.last_decode_step_ms.push(step_ms);
                let kind = if jacobi_accept == w { "FULL" } else { "PART" };
                let baseline_ms_per_tok: f64 = 175.0;
                let real_gain =
                    (committed_count as f64 * baseline_ms_per_tok) / step_ms.max(1.0);
                eprintln!(
                    "  decode lookahead[{kind}] step ~{step}: W={w} G={g_count} \
                     jacobi_acc={jacobi_accept} committed={committed_count} \
                     ({step_ms:.0}ms real_gain={real_gain:.2}x) tok={next}"
                );

                if self.eos_tokens.contains(&new_next) {
                    break;
                }
                continue;
            }

            // Try N-gram speculation. Falls back to single-token decode on no
            // match, or on cost-aware gate cooldown.
            let in_cooldown = spec_gate && step < cooldown_until;
            let proposed = if spec_enabled && !in_cooldown {
                crate::spec_decode::ngram_lookup(&history, spec_n, spec_min_n, spec_k)
            } else {
                if in_cooldown && spec_enabled {
                    spec_skipped_by_gate += 1;
                }
                None
            };

            if let Some(draft) = proposed {
                spec_attempts += 1;
                spec_drafts_total += draft.len();

                // Snapshot every layer's recurrent / append-cache state so a partial
                // accept can roll back to this point.
                let snap = self
                    .model
                    .snapshot_state()
                    .map_err(|e| anyhow::anyhow!("snapshot_state: {e}"))?;

                // Verify batch: [next, d_0, ..., d_{K-1}] of length K+1.
                let mut verify_in: Vec<u32> = Vec::with_capacity(1 + draft.len());
                verify_in.push(next);
                verify_in.extend_from_slice(&draft);
                let v_tensor = Tensor::new(verify_in.as_slice(), &self.device)?.unsqueeze(0)?;
                let logits = self.model.forward_with_offset(&v_tensor, offset)?; // [1, K+1, vocab]

                // Per-position argmax → [1, K+1] u32 tensor → flatten to vec.
                let argmax = logits
                    .argmax(D::Minus1)?
                    .flatten_all()?
                    .to_vec1::<u32>()?;
                debug_assert_eq!(argmax.len(), verify_in.len());

                let j = crate::spec_decode::greedy_accept_count(
                    &argmax[..draft.len()],
                    &draft,
                );
                spec_accepted_total += j;
                let new_next: u32;
                if j == draft.len() {
                    // Full accept: K drafts + argmax[K] = K+1 tokens.
                    spec_full_accepts += 1;
                    for d in &draft {
                        history.push(*d);
                        generated.push(*d);
                    }
                    new_next = argmax[verify_in.len() - 1];
                    history.push(new_next);
                    generated.push(new_next);
                    next = new_next;
                    step += verify_in.len();
                } else {
                    // Partial / full reject. Restore + fix-up forward of j+1 tokens
                    // so the recurrent state and KV append cache match "we just ran
                    // [next, d_0..d_{j-1}] from the snapshot point" before continuing.
                    self.model
                        .restore_state(&snap)
                        .map_err(|e| anyhow::anyhow!("restore_state: {e}"))?;
                    let mut fixup_in: Vec<u32> = Vec::with_capacity(j + 1);
                    fixup_in.push(next);
                    fixup_in.extend_from_slice(&draft[..j]);
                    let f_tensor =
                        Tensor::new(fixup_in.as_slice(), &self.device)?.unsqueeze(0)?;
                    let _ = self.model.forward_with_offset(&f_tensor, offset)?;
                    for i in 0..j {
                        history.push(draft[i]);
                        generated.push(draft[i]);
                    }
                    new_next = argmax[j];
                    history.push(new_next);
                    generated.push(new_next);
                    next = new_next;
                    step += j + 1;
                }

                // Cost-aware gate: maintain rolling window of last `gate_window`
                // j values; when full and avg < threshold, cool down for
                // `gate_cooldown` tokens to dodge guaranteed-loss regions.
                if spec_gate {
                    if recent_j.len() == gate_window {
                        recent_j.pop_front();
                    }
                    recent_j.push_back(j as u8);
                    if recent_j.len() == gate_window {
                        let sum: usize = recent_j.iter().map(|&v| v as usize).sum();
                        let avg = sum as f64 / gate_window as f64;
                        if avg < gate_threshold {
                            cooldown_until = step + gate_cooldown;
                            recent_j.clear();
                        }
                    }
                }

                let step_ms = t_step.elapsed().as_secs_f64() * 1000.0;
                self.last_decode_step_ms.push(step_ms);
                let kind = if j == draft.len() { "FULL" } else { "PART" };
                // Cost-aware gain: (j+1) tokens vs verify-cost / baseline 175ms.
                // Replaces the misleading raw-count "gain" used previously.
                let baseline_ms_per_tok: f64 = 175.0;
                let real_gain = ((j as f64 + 1.0) * baseline_ms_per_tok) / step_ms.max(1.0);
                eprintln!(
                    "  decode spec[{kind}] step ~{step}: K={} accepted={j} \
                     ({step_ms:.0}ms real_gain={real_gain:.2}x) tok={next}",
                    draft.len(),
                );
                if self.eos_tokens.contains(&new_next) {
                    break;
                }
                continue;
            }

            // Standard single-token decode.
            let input = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;
            let t_fwd = if decode_timing {
                Some(std::time::Instant::now())
            } else {
                None
            };
            let logits = self.model.forward_with_offset(&input, offset)?; // [1, 1, vocab]
            let fwd_ms = if let Some(t) = t_fwd {
                let _ = self.device.synchronize();
                Some(t.elapsed().as_secs_f64() * 1000.0)
            } else {
                None
            };

            let t_samp = if decode_timing {
                Some(std::time::Instant::now())
            } else {
                None
            };
            next = self.sample_last_token(
                &logits,
                greedy_fast,
                temperature,
                top_p,
                top_k,
                repeat_penalty,
                &history,
            )?;
            let samp_ms = t_samp.map(|t| t.elapsed().as_secs_f64() * 1000.0);
            history.push(next);
            generated.push(next);

            let step_ms = t_step.elapsed().as_secs_f64() * 1000.0;
            self.last_decode_step_ms.push(step_ms);
            if let (Some(f), Some(s)) = (fwd_ms, samp_ms) {
                let path = if greedy_fast { "gpu_argmax" } else { "cpu_full" };
                eprintln!(
                    "    decode-split step {step}: fwd={f:.1} sample[{path}]={s:.1} \
                     other={:.1} (vocab={})",
                    step_ms - f - s,
                    logits.dims().last().copied().unwrap_or(0),
                );
            }
            let (zc, cp) = lumen_metal::mxfp4_linear::take_trace_counts();
            let (pc, pco, pwa) = lumen_metal::mxfp4_gpu::take_profile_counts();
            if pc > 0 {
                eprintln!(
                    "  decode step {step}: offset={offset} tok={next} ({step_ms:.0}ms) \
                     [zc={zc} cpu={cp}] prof: calls={pc} commit_avg={:.1}us wait_avg={:.1}us \
                     wait_total={:.1}ms",
                    pco as f64 / pc as f64 / 1000.0,
                    pwa as f64 / pc as f64 / 1000.0,
                    pwa as f64 / 1_000_000.0,
                );
            } else {
                eprintln!(
                    "  decode step {step}: offset={offset} tok={next} ({step_ms:.0}ms) \
                     [zc={zc} cpu={cp}]"
                );
            }

            step += 1;

            if self.eos_tokens.contains(&next) {
                break;
            }
        }

        if spec_enabled {
            let total_drafted = spec_drafts_total.max(1);
            let acc_rate = (spec_accepted_total as f64) / (total_drafted as f64);
            let avg_j = if spec_attempts > 0 {
                spec_accepted_total as f64 / spec_attempts as f64
            } else {
                0.0
            };
            eprintln!(
                "  spec-decode summary: attempts={spec_attempts} full_accepts={spec_full_accepts} \
                 accepted={spec_accepted_total}/{spec_drafts_total} (acc_rate={acc_rate:.3}) \
                 avg_j={avg_j:.2} gate_skipped={spec_skipped_by_gate} \
                 (min_n={spec_min_n} K={spec_k} gate={})",
                if spec_gate { "on" } else { "off" },
            );
        }
        if lookahead_enabled {
            let avg_jacobi = if lookahead_attempts > 0 {
                lookahead_jacobi_accepted_total as f64 / lookahead_attempts as f64
            } else {
                0.0
            };
            let avg_committed = if lookahead_attempts > 0 {
                lookahead_committed_total as f64 / lookahead_attempts as f64
            } else {
                0.0
            };
            let pool_len = lookahead_proposer
                .as_ref()
                .map(|p| p.pool_len())
                .unwrap_or(0);
            eprintln!(
                "  lookahead-decode summary: attempts={lookahead_attempts} \
                 full_accepts={lookahead_full_accepts} \
                 jacobi_accepted_total={lookahead_jacobi_accepted_total} \
                 committed_total={lookahead_committed_total} \
                 avg_jacobi={avg_jacobi:.2} avg_committed={avg_committed:.2} \
                 pool_len={pool_len} \
                 (W={lookahead_w} G={lookahead_g} N={lookahead_n})",
            );
            self.last_lookahead_stats = Some(LookaheadStats {
                attempts: lookahead_attempts,
                full_accepts: lookahead_full_accepts,
                jacobi_accepted_total: lookahead_jacobi_accepted_total,
                committed_total: lookahead_committed_total,
                pool_len,
                w: lookahead_w,
                g: lookahead_g,
                n: lookahead_n,
            });
        }
        Ok(generated)
    }

    /// Sample the next token from a `[..., vocab]` logits view of the last position.
    ///
    /// When `greedy_fast` is set (and the GPU sampler is available + the tensor lives on
    /// Metal), dispatches an `argmax_f32` kernel directly against the existing logits
    /// buffer and pulls back a single u32 — bypassing the 248K-vocab host transfer and
    /// the CPU softmax/penalty pipeline entirely.
    ///
    /// Otherwise routes through the existing CPU sampler (`sample_token_cpu_full`).
    ///
    /// A4 (2026-04-30) — extends the GPU fast path to `temperature > 0` via
    /// Gumbel-max (`argmax(logits/T + G)`, `G_i = -ln(-ln(U_i))`). Reuses the
    /// existing Phase B GPU penalty pre-pass (in-place on the logits buffer).
    /// Activates when `temperature > 0 && top_p >= 1.0 && sampler exists`
    /// AND the logits tensor is Metal-resident.
    ///
    /// extends the stochastic GPU path to honor
    /// `top_p < 1.0` and / or `top_k > 0` via the fused
    /// `topk_topp_gumbel_argmax_f32` kernel. One threadgroup performs a
    /// per-thread top-K partial select, tree-reduces, runs softmax + top-p
    /// truncation + Gumbel-max sampling, and writes a single u32 token —
    /// removing the 248K-elem host transfer that the CPU fallback used to pay
    /// on every decode step. Gated by env `LUMEN_GPU_SAMPLE_TOPP_TOPK`
    /// (**default OFF** until 35B A/B confirms σ ≥ +2 long-Skv win; set to
    /// `1` to opt in). The plain Candle Gumbel path is preserved for
    /// `top_p >= 1.0 && top_k == 0` regardless of this flag.
    #[allow(clippy::too_many_arguments)]
    fn sample_last_token(
        &self,
        last: &Tensor,
        greedy_fast: bool,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        repeat_penalty: f32,
        history: &[u32],
    ) -> Result<u32> {
        // default OFF until 35B A/B confirms σ ≥ +2 win at long
        // Skv. Smoke test (Skv=13 short prompt) showed both paths produce
        // coherent output but the new kernel may regress short-prompt decode
        // by ~10%; the +1-3% win is hypothesized to materialize only at full
        // production Skv where the 1 MB host-transfer dominates. Opt-in
        // explicitly with `LUMEN_GPU_SAMPLE_TOPP_TOPK=1`.
        let topp_topk_opt_in = std::env::var("LUMEN_GPU_SAMPLE_TOPP_TOPK")
            .map(|v| v == "1")
            .unwrap_or(false);

        // Both GPU paths (greedy_fast + GPU temp) need the sampler + a
        // Metal-resident contiguous tensor. Bundle the entry conditions so
        // either path can dispatch from one code site.
        //
        // The relaxed `top_p >= 1.0 || topp_topk_opt_in` clause is what
        // routes nucleus / top-k decode steps to the new fused kernel
        // (instead of falling back to the CPU sampler). Top-k is always
        // satisfied by the new kernel (clamped to [1, K_MAX=32]).
        let gpu_temp_eligible = !greedy_fast
            && temperature > 0.0
            && self.sampler.is_some()
            && (top_p >= 1.0 || topp_topk_opt_in);

        if (greedy_fast || gpu_temp_eligible) && self.sampler.is_some() {
            let sampler = self.sampler.as_ref().unwrap();
            let last_c = last.contiguous()?;
            if let Some((buf, off)) = metal_buffer_of(&last_c) {
                let n = *last_c
                    .dims()
                    .last()
                    .ok_or_else(|| anyhow::anyhow!("logits tensor has rank 0"))?;
                // Forward-pass writes use Candle's command queue; our sampler queue is
                // separate. Sync once so subsequent dispatches see the fully-written logits.
                let _ = self.device.synchronize();

                // GPU repeat+ngram penalty pre-pass.
                // `aggregate_penalty_multipliers` collapses both penalty
                // mechanisms into one `(token_idx, multiplier)` list per
                // unique idx; the GPU kernel applies them in place to the
                // logits buffer. Empty list (e.g. rep_pen=1 + no ngram match)
                // skips the kernel entirely.
                let (idx_data, mul_data) = aggregate_penalty_multipliers(
                    history,
                    repeat_penalty,
                    4,    // ngram window
                    5.0,  // ngram penalty multiplier
                    n,
                );

                if greedy_fast {
                    // A.1 — fused penalty+argmax in one command buffer. Saves the
                    // commit/wait round-trip the two-call sequence used to pay,
                    // and Metal's intra-cmd-buffer hazard tracking inserts the
                    // implicit barrier between the two encoders.
                    sampler.kernels.apply_penalties_then_argmax(
                        &sampler.queue,
                        buf,
                        off,
                        n as u32,
                        &sampler.scratch_idx_buf,
                        &sampler.scratch_mul_buf,
                        sampler.scratch_capacity,
                        &idx_data,
                        &mul_data,
                        &sampler.out_buf,
                        0,
                    )?;
                    return Ok(SamplingKernels::read_u32_scalar(&sampler.out_buf));
                }

                // Stochastic path: penalty pre-pass first (in place on the
                // logits buffer), then either:
                //   - top_p >= 1.0 && top_k == 0 → Candle Gumbel argmax (A4)
                //   - else                       → fused topk_topp_gumbel kernel (Phase C #4)
                sampler.kernels.apply_token_penalties_f32_into(
                    &sampler.queue,
                    buf,
                    off,
                    &sampler.scratch_idx_buf,
                    &sampler.scratch_mul_buf,
                    sampler.scratch_capacity,
                    &idx_data,
                    &mul_data,
                )?;

                let needs_filter = top_p < 1.0 || top_k > 0;
                if !needs_filter {
                    // A4 — GPU stochastic path via Gumbel-max. After the
                    // penalty pre-pass mutated the buffer in place, view the
                    // same backing storage through Candle: scale by 1/T, add
                    // Gumbel noise, argmax. Avoids materializing the 248K-elem
                    // F32 vector on the host.
                    let logits_1d = last_c.flatten_all()?;
                    debug_assert_eq!(logits_1d.elem_count(), n);
                    let scaled = (logits_1d / temperature as f64)?;
                    let device = last_c.device();
                    // Uniform on (eps, 1] — eps avoids `log(0) = -inf`.
                    let u = Tensor::rand(1e-7f32, 1.0f32, scaled.shape(), device)?;
                    // Gumbel: G = -ln(-ln(U)).
                    let gumbel = u.log()?.neg()?.log()?.neg()?;
                    let perturbed = (scaled + gumbel)?;
                    let tok = perturbed
                        .argmax(0)?
                        .to_scalar::<u32>()
                        .context("gpu gumbel-argmax failed")?;
                    return Ok(tok);
                }

                // fused top-k + top-p + Gumbel-max. Single
                // dispatch (1 threadgroup × 64 threads) does:
                //   - per-thread local top-K (sorted descending)
                //   - tree reduction merge to global top-K
                //   - softmax + top-p truncation + Gumbel-max sample
                // Writes one u32 token to `sampler.out_buf`.
                //
                // K_MAX is 32 in the kernel; if the caller asked for >32,
                // clamp; if 0 (no top-k cap), use the cap. Effective top-p
                // also clamps to (0, 1] (kernel rejects values outside).
                let effective_k: u32 = {
                    let k = if top_k == 0 { 32 } else { top_k as u32 };
                    k.clamp(1, lumen_metal::sampling::TOPK_K_MAX)
                };
                let effective_p: f32 = top_p.clamp(f32::MIN_POSITIVE, 1.0);
                let inv_temp: f32 = 1.0_f32 / temperature;

                // RNG seed: monotone per-call counter mixed with a process-
                // start nanosecond stamp. Non-cryptographic; just needs to
                // change every step so consecutive Gumbel draws differ.
                static SAMPLE_COUNTER: std::sync::atomic::AtomicU64 =
                    std::sync::atomic::AtomicU64::new(0);
                static SAMPLE_SEED_HI: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
                let seed_hi: u32 = *SAMPLE_SEED_HI.get_or_init(|| {
                    let nanos = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.subsec_nanos())
                        .unwrap_or(0);
                    nanos ^ (std::process::id() as u32)
                });
                let counter =
                    SAMPLE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let seed_lo: u32 =
                    (counter as u32) ^ ((counter >> 32) as u32) ^ seed_hi.rotate_left(13);

                sampler.kernels.topk_topp_gumbel_argmax_f32_into(
                    &sampler.queue,
                    buf,
                    off,
                    n as u32,
                    effective_k,
                    effective_p,
                    inv_temp,
                    seed_lo,
                    seed_hi,
                    &sampler.out_buf,
                    0,
                )?;
                return Ok(SamplingKernels::read_u32_scalar(&sampler.out_buf));
            }
            // Tensor wasn't Metal-resident (CPU device or non-contiguous after copy) →
            // fall through to the CPU path.
        }

        let last_vec = tensor_to_vec_f32(last)?;
        // pass owned Vec<f32> through to skip the internal
        // 248K-elem clone the borrow-API variant pays.
        sample_token_cpu_full_owned(
            last_vec,
            temperature,
            top_p,
            top_k,
            repeat_penalty,
            history,
        )
    }

    /// Apply the Qwen3 chat template + generate a response. `thinking=true` leaves the
    /// `<think>` block open so the model emits reasoning (shipped default); `false`
    /// closes it immediately with `<think>\n\n</think>\n\n` to short-circuit reasoning.
    pub fn chat(
        &mut self,
        messages: &[(String, String)],
        max_new_tokens: usize,
        temperature: f32,
        thinking: bool,
    ) -> Result<String> {
        let prompt = format_qwen3_chat(messages, thinking);
        eprintln!("chat prompt ({} chars, thinking={thinking}): {prompt:?}", prompt.len());
        let ids = self.encode(&prompt)?;
        eprintln!("chat encoded {} tokens: {:?}", ids.len(), &ids[..ids.len().min(20)]);
        let greedy = std::env::var("LUMEN_GREEDY").map(|v| v == "1").unwrap_or(false);
        let (eff_temp, top_k, rep_pen) = if greedy {
            (0.0_f32, 0_usize, 1.0_f32)
        } else {
            (temperature, 20_usize, 1.1_f32)
        };
        let out = self.generate_with_opts(&ids, max_new_tokens, eff_temp, 0.95, top_k, rep_pen)?;
        self.decode(&out)
    }

    /// Streaming variant of [`Self::chat`]: calls `on_token` with each decoded text fragment
    /// as tokens are generated so the server can emit SSE events in real time.
    ///
    /// Runs a simple single-token decode loop (no spec/lookahead — those are throughput
    /// optimisations irrelevant for streaming UX). Emits the same
    /// `decode step N: offset=X tok=Y (Zms)` log lines as `generate_with_opts` so
    /// `bench_window_logparse.py` and other log-based benchmarks continue to work.
    pub fn chat_streaming<F>(
        &mut self,
        messages: &[(String, String)],
        max_new_tokens: usize,
        temperature: f32,
        thinking: bool,
        mut on_token: F,
    ) -> Result<String>
    where
        F: FnMut(&str),
    {
        let prompt = format_qwen3_chat(messages, thinking);
        eprintln!("chat_streaming prompt ({} chars, thinking={thinking})", prompt.len());
        let ids = self.encode(&prompt)?;
        eprintln!("chat_streaming encoded {} tokens", ids.len());

        if ids.is_empty() {
            anyhow::bail!("empty input_ids");
        }

        let greedy = std::env::var("LUMEN_GREEDY").map(|v| v == "1").unwrap_or(false);
        let (eff_temp, top_k, rep_pen) = if greedy {
            (0.0_f32, 0_usize, 1.0_f32)
        } else {
            (temperature, 20_usize, 1.1_f32)
        };
        let greedy_fast = eff_temp == 0.0 && self.sampler.is_some();

        self.last_decode_step_ms.clear();
        self.last_lookahead_stats = None;
        self.model.reset_cache();

        // ── Prefill ───────────────────────────────────────────────────────
        let t_prefill = std::time::Instant::now();
        let prompt_tensor = Tensor::new(&ids[..], &self.device)?.unsqueeze(0)?;
        let logits_all = self.model.forward_with_offset(&prompt_tensor, 0)?; // [1, S, vocab]
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
        let past_len = ids.len();

        let mut history: Vec<u32> = Vec::with_capacity(ids.len() + max_new_tokens);
        history.extend_from_slice(&ids);

        let last = logits_all.narrow(D::Minus2, past_len - 1, 1)?;
        let mut next =
            self.sample_last_token(&last, greedy_fast, eff_temp, 0.95, top_k, rep_pen, &history)?;
        history.push(next);
        let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
        generated.push(next);
        eprintln!("  prefill: {past_len} tokens in {prefill_ms:.0}ms -> tok={next}");

        // Incremental decode state — buffer incomplete multi-byte characters.
        let mut prev_text = String::new();
        if let Ok(text) = self.decode(&generated) {
            if !text.is_empty() && !text.contains('\u{FFFD}') {
                on_token(&text);
                prev_text = text;
            }
        }

        if self.eos_tokens.contains(&next) {
            return self.decode(&generated);
        }

        // ── Decode ────────────────────────────────────────────────────────
        let mut step = 1usize;
        while step < max_new_tokens {
            let t_step = std::time::Instant::now();
            let offset = past_len + step - 1;

            let input = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward_with_offset(&input, offset)?; // [1, 1, vocab]
            next =
                self.sample_last_token(&logits, greedy_fast, eff_temp, 0.95, top_k, rep_pen, &history)?;
            history.push(next);
            generated.push(next);

            let step_ms = t_step.elapsed().as_secs_f64() * 1000.0;
            self.last_decode_step_ms.push(step_ms);
            let (zc, cp) = lumen_metal::mxfp4_linear::take_trace_counts();
            eprintln!(
                "  decode step {step}: offset={offset} tok={next} ({step_ms:.0}ms) \
                 [zc={zc} cpu={cp}]"
            );

            // Incremental decode — skip fragments with incomplete UTF-8.
            if let Ok(text) = self.decode(&generated) {
                if text.len() > prev_text.len() && !text.contains('\u{FFFD}') {
                    on_token(&text[prev_text.len()..]);
                    prev_text = text;
                }
            }

            step += 1;
            if self.eos_tokens.contains(&next) {
                break;
            }
        }

        // Flush any remaining buffered text (e.g. trailing multi-byte sequence).
        if let Ok(text) = self.decode(&generated) {
            if text.len() > prev_text.len() {
                let delta = &text[prev_text.len()..];
                let clean: String = delta.replace('\u{FFFD}', "");
                if !clean.is_empty() {
                    on_token(&clean);
                }
            }
        }

        self.decode(&generated)
    }
}

/// Read a logit tensor (any shape with vocab on the last dim) into `Vec<f32>` on CPU.
fn tensor_to_vec_f32(t: &Tensor) -> Result<Vec<f32>> {
    let flat = t.flatten_all()?.to_dtype(DType::F32)?;
    Ok(flat.to_vec1::<f32>()?)
}

/// build the GPU penalty input for the greedy
/// sample fast path. Combines `repeat_penalty` (frequency over a 256-token
/// window) and an n-gram repetition penalty (n=4, mul=5.0) into a single
/// per-token-index multiplier so the GPU kernel can apply them in one
/// dispatch without write contention.
///
/// Mirrors the CPU sampler's penalty composition exactly:
///   repeat_penalty: `mul *= rp.powi(count)` for each token in the freq map
///   ngram_penalty:  `mul *= 5.0` for each match position's `next_tok`
/// Same token may appear in both — multipliers are multiplied together,
/// preserving the `if logit > 0 then logit /= mul else logit *= mul` rule
/// (since divide-by-product == divide-once-per-component).
///
/// Returns parallel `(token_idx, multiplier)` arrays of equal length. Only
/// entries with `multiplier > 1` are emitted; if neither penalty fires the
/// arrays are empty and the GPU kernel is skipped.
fn aggregate_penalty_multipliers(
    history: &[u32],
    repeat_penalty: f32,
    ngram_n: usize,
    ngram_penalty: f32,
    vocab: usize,
) -> (Vec<u32>, Vec<f32>) {
    use std::collections::HashMap;
    let mut combined: HashMap<u32, f32> = HashMap::new();

    if (repeat_penalty - 1.0).abs() > 1e-9 && !history.is_empty() {
        let window = history.len().min(256);
        let recent = &history[history.len() - window..];
        let mut freq: HashMap<u32, u32> = HashMap::new();
        for &tok in recent {
            *freq.entry(tok).or_insert(0) += 1;
        }
        for (&tok, &count) in &freq {
            if (tok as usize) < vocab {
                let mul = repeat_penalty.powi(count as i32);
                if mul > 1.0 {
                    *combined.entry(tok).or_insert(1.0) *= mul;
                }
            }
        }
    }

    if history.len() >= ngram_n && ngram_penalty > 1.0 {
        let prefix = &history[history.len() - (ngram_n - 1)..];
        let window = history.len().min(256);
        let start = history.len() - window;
        let end = history.len() - (ngram_n - 1);
        for i in start..end {
            if i + ngram_n > history.len() {
                break;
            }
            if &history[i..i + ngram_n - 1] == prefix {
                let next_tok = history[i + ngram_n - 1];
                if (next_tok as usize) < vocab {
                    *combined.entry(next_tok).or_insert(1.0) *= ngram_penalty;
                }
            }
        }
    }

    if combined.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut idx: Vec<u32> = Vec::with_capacity(combined.len());
    let mut mul: Vec<f32> = Vec::with_capacity(combined.len());
    for (k, v) in combined {
        idx.push(k);
        mul.push(v);
    }
    (idx, mul)
}

/// Qwen3 chat template — matches the `chat_template.jinja` shipped with
/// `mlx-community/Qwen3.6-35B-A3B-mxfp4` for the plain text-only, no-tools path:
///
/// - `<|im_start|>{role}\n{content}<|im_end|>\n` per message
/// - `<|im_start|>assistant\n<think>\n` generation prompt (thinking on, default)
/// - `<|im_start|>assistant\n<think>\n\n</think>\n\n` (thinking off)
///
/// Tool calls, vision content, and assistant reasoning history aren't handled here —
/// they require the full Jinja loop (minijinja). Text chat completions go through
/// this path.
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

fn resolve_eos_tokens(tok: &Tokenizer) -> Vec<u32> {
    let candidates = ["<|im_end|>", "<|endoftext|>"];
    let mut out = Vec::new();
    for name in candidates {
        if let Some(id) = tok.token_to_id(name) {
            out.push(id);
        }
    }
    // Fallback: Qwen3.6 vocab ships `<|im_end|>` at 248046 and `<|endoftext|>` at 248044.
    // Older Qwen2/3 builds used 151645 / 151643 — we stay correct for the current model.
    if out.is_empty() {
        out.extend_from_slice(&[248046, 248044]);
    }
    out
}
