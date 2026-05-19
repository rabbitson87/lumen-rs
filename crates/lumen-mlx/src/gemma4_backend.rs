//! Gemma 4 26B-A4B native MLX backend wrapper.
//!
//! Bundles `NativeGemma4Model` + `Gemma4ChatTemplate` + per-request
//! `NativeGemma4PromptCache` behind a synchronous, blocking API that
//! mirrors the `ModelBackend` trait shape used by `lumen-server`.
//!
//! Why a separate wrapper layer (rather than calling the primitives
//! directly from `engine.rs`):
//!   • Encapsulates the cache-allocation-per-request invariant; callers
//!     can't accidentally reuse a dirty cache.
//!   • Lets the existing engine treat Gemma 4 like any other backend
//!     (Qwen, Gemma, GGUF) via the same method shape.
//!   • Provides a single place to add session/prefix-cache reuse later
//!     without touching engine.rs.
//!
//! Threading: this struct is `Send` (NativeGemma4Model + Gemma4ChatTemplate
//! both are) but **not** `Sync` — callers must wrap it in an external
//! synchronization primitive (Mutex / channel) for concurrent use, same
//! as the other backends.

#[cfg(feature = "mlx-native")]
#[allow(dead_code)] // re-exported via `pub mod gemma4`
pub(crate) mod imp {
    use anyhow::{Context, Result, anyhow};
    use std::collections::HashMap;
    use std::path::Path;
    use std::time::Instant;

    use crate::gemma4_chat::imp::{ChatMessage, ChatRole, Gemma4ChatTemplate, RenderOptions};
    use crate::gemma4_moe::imp::{GenerateConfig, NativeGemma4Model, NativeGemma4PromptCache};
    use crate::gemma4_response::imp::{ParsedResponse, ResponseParser};

    /// Prefix-cache entry: snapshot of a prompt-prefilled cache that can be
    /// cloned + truncated to serve subsequent requests sharing a common
    /// prefix (e.g. the same system message across a 1000-item batch).
    pub struct Gemma4PrefixCacheEntry {
        /// The KV cache snapshot. `master.offset()` == `prefix_tokens.len()`.
        master: NativeGemma4PromptCache,
        /// Token sequence that was prefilled into `master`. Used for LCP
        /// detection against subsequent prompts.
        prefix_tokens: Vec<u32>,
        last_access: Instant,
        hits: u64,
    }

    /// Synchronous, single-tenant Gemma 4 backend.
    ///
    /// Each `generate*` call allocates a fresh `NativeGemma4PromptCache`
    /// **unless** the prefix-cache path (`chat_with_prefix_cache`) is used.
    pub struct Gemma4Backend {
        model: NativeGemma4Model,
        chat: Gemma4ChatTemplate,
        model_id: String,
        /// Per-key prefix caches. Keyed by caller-supplied string (e.g. the
        /// system message hash from the Moltis side, or a batch id).
        prefix_caches: HashMap<String, Gemma4PrefixCacheEntry>,
    }

    impl Gemma4Backend {
        /// Load model + tokenizer from a local directory containing
        /// `config.json`, `tokenizer.json`, and the safetensors shards.
        ///
        /// `model_id` is the OpenAI-style label echoed back in
        /// `/v1/chat/completions` responses; it's not parsed for routing.
        pub fn from_dir<P: AsRef<Path>>(model_id: impl Into<String>, dir: P) -> Result<Self> {
            let dir = dir.as_ref();
            let model = NativeGemma4Model::load(dir)
                .with_context(|| format!("Gemma4Backend::from_dir({dir:?}): model load"))?;
            let chat = Gemma4ChatTemplate::from_dir(dir)
                .with_context(|| format!("Gemma4Backend::from_dir({dir:?}): tokenizer load"))?;
            Ok(Self {
                model,
                chat,
                model_id: model_id.into(),
                prefix_caches: HashMap::new(),
            })
        }

        pub fn model_id(&self) -> &str {
            &self.model_id
        }

        pub fn model(&self) -> &NativeGemma4Model {
            &self.model
        }

        pub fn chat_template(&self) -> &Gemma4ChatTemplate {
            &self.chat
        }

        // ── Trait-shape API used by `lumen-server` ─────────────────

        /// Tokenize raw text without applying the chat template. Used by
        /// the `/v1/completions` (legacy) endpoint.
        pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
            self.chat.encode_plain(text)
        }

        /// Decode a token-id sequence into a UTF-8 string. Special tokens
        /// (turn markers, channel markers, etc.) are stripped so the
        /// caller gets the visible text only — matches the contract of
        /// `Qwen3.5MoeBackend::decode`.
        pub fn decode(&self, tokens: &[u32]) -> Result<String> {
            self.chat.decode(tokens, /* skip_special */ true)
        }

        /// Build a chat-templated prompt token list.
        ///
        /// `messages` is the same `(role, content)` shape as the rest of
        /// the engine; the role string is matched case-insensitively
        /// against `system` / `user` / `assistant`.
        ///
        /// `thinking` toggles `enable_thinking` on the chat template — set
        /// true to allow the model to emit a `<|channel>thought…<channel|>`
        /// reasoning block before its visible reply.
        pub fn build_chat_input(
            &self,
            messages: &[(String, String)],
            thinking: bool,
        ) -> Result<Vec<u32>> {
            let parsed: Vec<ChatMessage<'_>> = messages
                .iter()
                .map(|(role, content)| {
                    let role = match role.as_str() {
                        "system" | "System" | "SYSTEM" => ChatRole::System,
                        "user" | "User" | "USER" => ChatRole::User,
                        "assistant" | "Assistant" | "ASSISTANT" | "model" => ChatRole::Assistant,
                        other => {
                            return Err(anyhow!(
                                "Gemma4Backend::build_chat_input: unknown role {other:?}"
                            ));
                        }
                    };
                    Ok(ChatMessage {
                        role,
                        content: content.as_str(),
                    })
                })
                .collect::<Result<_>>()?;
            self.chat.render_to_ids(
                &parsed,
                &RenderOptions {
                    enable_thinking: thinking,
                    add_generation_prompt: true,
                },
            )
        }

        /// `/v1/completions` path: raw greedy generation from token ids.
        /// `temperature` / `top_p` are accepted but ignored at this phase —
        /// W5 (sampling) lands them.
        pub fn generate(
            &mut self,
            input_ids: &[u32],
            max_new_tokens: usize,
            _temperature: f32,
            _top_p: f32,
        ) -> Result<Vec<u32>> {
            let cfg = GenerateConfig {
                max_new_tokens,
                stop_on_eos: true,
            };
            let stats = self.model.generate(input_ids, &cfg)?;
            Ok(stats.generated_tokens)
        }

        /// `/v1/chat/completions` path: render → generate → parse.
        ///
        /// Returns the parsed response (visible text, reasoning, tool calls)
        /// so the HTTP layer can ship structured fields per the OpenAI spec.
        pub fn chat(
            &mut self,
            messages: &[(String, String)],
            max_new_tokens: usize,
            _temperature: f32,
            thinking: bool,
        ) -> Result<ParsedResponse> {
            let prompt = self.build_chat_input(messages, thinking)?;
            let cfg = GenerateConfig {
                max_new_tokens,
                stop_on_eos: true,
            };
            let stats = self.model.generate(&prompt, &cfg)?;

            let mut parser = ResponseParser::new(&self.chat);
            for token in &stats.generated_tokens {
                parser.push(*token)?;
            }
            parser.finalize()
        }

        /// Prefix-cache-aware `chat()` variant for the Moltis batch workload.
        ///
        /// **When to use**: caller has a batch of requests that share a
        /// large common prefix (e.g. same system prompt) and want to amortize
        /// the prefill cost.
        ///
        /// **Semantics**:
        /// 1. Build prompt from messages.
        /// 2. If `prefix_cache_key` has an entry with a token prefix matching
        ///    the new prompt (longest common prefix > 0), clone the master
        ///    cache, truncate to LCP, forward only the suffix — saves the
        ///    common-prefix prefill cost.
        /// 3. Otherwise, full cold prefill from a fresh cache.
        /// 4. After generation, snapshot the post-prompt (pre-decode-advance)
        ///    cache state as the new master under `prefix_cache_key`.
        ///
        /// **Limitations**: sliding-window cache rotations crossed during
        /// decode make `truncate_to` lossy. For the Moltis sports-matching
        /// workload with ≤4K context and ≤150 output tokens, the sliding
        /// window (1024) won't rotate, so snapshots are exact.
        pub fn chat_with_prefix_cache(
            &mut self,
            messages: &[(String, String)],
            max_new_tokens: usize,
            _temperature: f32,
            thinking: bool,
            prefix_cache_key: &str,
        ) -> Result<ParsedResponse> {
            let prompt = self.build_chat_input(messages, thinking)?;
            if prompt.is_empty() {
                return Err(anyhow!("chat_with_prefix_cache: empty prompt"));
            }

            // ── Lookup + cache fork ──
            let (mut cache, hit_kind, lcp) = match self.prefix_caches.get(prefix_cache_key) {
                Some(entry) => {
                    let lcp = entry
                        .prefix_tokens
                        .iter()
                        .zip(prompt.iter())
                        .take_while(|(a, b)| a == b)
                        .count();
                    if lcp > 0 && lcp <= entry.prefix_tokens.len() {
                        let mut cache = entry.master.clone();
                        if lcp < cache.offset() {
                            cache
                                .truncate_to(lcp)
                                .context("chat_with_prefix_cache: truncate cloned master to LCP")?;
                        }
                        (cache, "hit", lcp)
                    } else {
                        (self.model.make_cache(), "miss-no-overlap", 0)
                    }
                }
                None => (self.model.make_cache(), "miss-no-entry", 0),
            };

            // Update hit stats
            if hit_kind == "hit" {
                if let Some(entry) = self.prefix_caches.get_mut(prefix_cache_key) {
                    entry.last_access = Instant::now();
                    entry.hits += 1;
                }
            }

            let suffix_len = prompt.len().saturating_sub(cache.offset());
            eprintln!(
                "[gemma4-backend] prefix-cache key={prefix_cache_key:?} \
                 result={hit_kind} lcp={lcp} suffix_len={suffix_len}"
            );

            // ── Generate (suffix prefill + decode) ──
            let suffix = &prompt[cache.offset()..];
            let cfg = GenerateConfig {
                max_new_tokens,
                stop_on_eos: true,
            };
            let stats = self
                .model
                .generate_with_cache(suffix, &cfg, Some(&mut cache))
                .context("chat_with_prefix_cache: generate_with_cache")?;

            // ── Snapshot post-prompt state as the new master ──
            // After generate, `cache.offset()` is well past `prompt.len()`
            // (advanced by decoded tokens). Truncate back to prompt.len()
            // so the master snapshot represents only the prompt prefix —
            // this is what subsequent requests with the same system prompt
            // want to fork from.
            let target_offset = prompt.len();
            let mut master_snapshot = cache.clone();
            if master_snapshot.offset() > target_offset {
                master_snapshot
                    .truncate_to(target_offset)
                    .context("chat_with_prefix_cache: truncate master to prompt end")?;
            }
            // Drop any prior entry for this key (cheap; just frees the old
            // master arrays via refcount).
            self.prefix_caches.insert(
                prefix_cache_key.to_string(),
                Gemma4PrefixCacheEntry {
                    master: master_snapshot,
                    prefix_tokens: prompt.clone(),
                    last_access: Instant::now(),
                    hits: 0,
                },
            );

            // ── Parse decoded tokens into ParsedResponse ──
            let mut parser = ResponseParser::new(&self.chat);
            for token in &stats.generated_tokens {
                parser.push(*token)?;
            }
            parser.finalize()
        }

        /// Drop a prefix-cache entry by key. Returns true if it existed.
        pub fn drop_prefix_cache(&mut self, key: &str) -> bool {
            self.prefix_caches.remove(key).is_some()
        }

        /// Clear all prefix-cache entries; returns the number released.
        pub fn clear_prefix_cache(&mut self) -> usize {
            let n = self.prefix_caches.len();
            self.prefix_caches.clear();
            n
        }

        /// Number of live prefix-cache entries.
        pub fn prefix_cache_count(&self) -> usize {
            self.prefix_caches.len()
        }

        /// Streaming variant of `chat()`. Calls `on_token` once per
        /// generated token with the *decoded* text fragment (special
        /// tokens stripped) so the caller can flush SSE events as they
        /// arrive.
        ///
        /// Returns the same `ParsedResponse` as `chat()` after the loop
        /// completes — caller may use it to compute final token counts
        /// or to inspect tool calls / reasoning content separately.
        pub fn chat_streaming(
            &mut self,
            messages: &[(String, String)],
            max_new_tokens: usize,
            thinking: bool,
            mut on_token: impl FnMut(&str) -> Result<()>,
        ) -> Result<ParsedResponse> {
            let prompt = self.build_chat_input(messages, thinking)?;
            // ── Manual prefill + decode loop so we can inject the
            //    on_token callback between steps.
            //
            // async pipelining mirroring mlx-lm's generate_step.
            // Schedule next-step graph + async_eval BEFORE syncing current
            // token's argmax. GPU overlaps step N's eval with step N+1's
            // graph build, recovering decode tok/s parity with mlx-lm.
            //
            // dedicated generation stream (mirrors
            // mlx-lm's `mx.new_thread_local_stream(mx.default_device())` +
            // `with mx.stream(generation_stream)`). Isolating the forward
            // work onto its own stream prevents contention with any other
            // mlx ops happening concurrently (e.g. tokenizer / preprocess)
            // and lets the MLX backend keep a tighter Metal command queue
            // for the decode pipeline.
            let gen_stream = mlx_rs::Stream::gpu();
            mlx_rs::with_new_default_stream(gen_stream, || -> Result<ParsedResponse> {
                let mut cache = self.model.make_cache();

                // DISABLED 2026-05-14 (debt #X):
                // The mlx_lm-style chunked prefill broke for Gemma 4's
                // sliding-window cache. After chunk N the sliding cache
                // rotates so `k_full.shape[2] != kv_offset + chunk_len`, and
                // the current `make_attention_mask_for_layer` builds the mask
                // as `(query_len, kv_offset + query_len)` which then fails to
                // broadcast against the rotated K (observed:
                // `(4096,8192) vs (1,16,4096,5119)`). Proper fix needs the
                // cache to expose its absolute-position window so the mask
                // construction can size against `k_full.shape[2]`. Reverting
                // to single-pass forward for now; chunked path will return
                // when the sliding-aware mask lands.
                //
                // the single-call forward
                // path overflows the Metal command-buffer memory budget around
                // ~6-10K tokens on a 36 GB unified-memory box (anti-pattern: one
                // graph holds every layer's intermediate Q/K/V/attn-out for the
                // full prompt length). Split into fixed-size chunks and eval
                // between chunks so peak in-flight work is bounded to one
                // chunk's footprint. `cache.offset()` advances naturally so
                // RoPE / mask / KV cache writes pick up the correct absolute
                // position for chunk 2+ — mirrors mlx-lm's `prefill_step_size`.
                //
                // Default chunk size = 4096. Bench at PROMPT_LEN=4096 is known
                // to fit inside a single Metal command buffer on a 36 GB box,
                // so 4096 is the largest empirically-safe size. Larger chunks
                // mean fewer chunk-boundary evals → lower total prefill
                // wall-time, important because the HTTP layer can't stream
                // back to the client until the first decode token is produced.
                // Override with `LUMEN_GEMMA4_PREFILL_CHUNK` (env, in tokens).
                // Default chunk reduced 4096 → 2048 after observing post-eval
                // Metal CB OOMs at chunk N where the full-attention cache had
                // grown to ~10K KV. The attention QxK^T graph for a 4096-token
                // chunk over a 10K KV is too large for one command buffer;
                // halving the chunk keeps per-CB work bounded.
                let chunk_size: usize = std::env::var("LUMEN_GEMMA4_PREFILL_CHUNK")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .filter(|&n: &usize| n > 0)
                    .unwrap_or(2048);
                // Chunked prefill re-enabled (2026-05-14, take 2): the mask
                // builder now reads kv_actual from k_full.shape, so rotated
                // sliding caches no longer trigger broadcast mismatches.
                let chunks: Vec<&[u32]> = prompt.chunks(chunk_size).collect();
                let n_chunks = chunks.len();
                eprintln!(
                    "[prefill] start {} chunks (size {}, total {} tokens)",
                    n_chunks,
                    chunk_size,
                    prompt.len()
                );
                let mut logits_opt: Option<mlx_rs::Array> = None;
                for (i, chunk) in chunks.into_iter().enumerate() {
                    let t0 = std::time::Instant::now();
                    // Use forward_last_token: only the final chunk's logits feed
                    // into `argmax_last_token_lazy` below (intermediate chunks'
                    // logits are discarded), and decode only needs the last-
                    // position next-token logits. Slicing h to last position
                    // before the tied lm_head skips ~3 TFLOPs per long chunk
                    // (8K × hidden × vocab quantized matmul) whose output is
                    // immediately reduced to a single argmax. Bit-identical
                    // tokens; see playbook_lm_head_last_token_slice.md.
                    let chunk_logits = self
                        .model
                        .forward_last_token(chunk, &mut cache)
                        .with_context(|| {
                            format!(
                                "chat_streaming: prefill chunk {}/{} ({} tokens)",
                                i + 1,
                                n_chunks,
                                chunk.len()
                            )
                        })?;
                    // EVEN the final chunk gets eval'd. Keeping it lazy in the
                    // multi-chunk regime piles chunk-N's forward graph onto
                    // the subsequent argmax + async_eval call, which then
                    // re-issues the whole forward as part of one Metal
                    // command buffer → CB-budget OOM. Single-pass prefill
                    // (n_chunks=1) hits this branch too: cost of the extra
                    // eval is bounded by chunk_size; for normal short
                    // prompts it's a no-op vs the pre-existing eval inside
                    // argmax_last_token_lazy + async_eval downstream.
                    chunk_logits.eval().with_context(|| {
                        format!("chat_streaming: prefill chunk {}/{} eval", i + 1, n_chunks)
                    })?;
                    let active = crate::metal_memory::get_active_memory().unwrap_or(0);
                    let peak = crate::metal_memory::get_peak_memory().unwrap_or(0);
                    let cache = crate::metal_memory::get_cache_memory().unwrap_or(0);
                    eprintln!(
                        "[prefill] chunk {}/{} done in {:.0}ms  mlx-mem active={:.1}GB peak={:.1}GB cache={:.1}GB",
                        i + 1,
                        n_chunks,
                        t0.elapsed().as_secs_f64() * 1000.0,
                        active as f64 / 1e9,
                        peak as f64 / 1e9,
                        cache as f64 / 1e9
                    );
                    if i + 1 == n_chunks {
                        logits_opt = Some(chunk_logits);
                    }
                }
                let logits = logits_opt
                    .ok_or_else(|| anyhow!("chat_streaming: empty prompt has no chunks"))?;
                let mut current = self
                    .model
                    .argmax_last_token_lazy(&logits)
                    .context("chat_streaming: prefill argmax_lazy")?;
                mlx_rs::transforms::async_eval([&current])
                    .context("chat_streaming: prefill async_eval")?;

                // tested mlx-lm's `mx.clear_cache()`
                // post-prefill pattern (generate.py:451) → NEGATIVE -4.4σ
                // (29.54 → 29.30 tok/s). On our path the prefill cache holds
                // intermediate buffers reused immediately by decode steps;
                // clearing forces re-alloc and net costs more than it saves.
                // No clear_cache().

                let mut parser = ResponseParser::new(&self.chat);
                let eos = self.model.eos_tokens().to_vec();

                // env-gated op-counter instrumentation.
                // When `LUMEN_GEMMA4_COUNT_OPS=1`, resets mlx-rs `OP_COUNTER`
                // before each decode step's forward+argmax, reads the delta
                // after sync, and logs per-step ops. Aggregates total ops + per-
                // step wallclock over the decode loop, prints summary at end.
                //
                // Hypothesis: ~570 ops/step × ~5-10 μs FFI overhead per op =
                // ~3-6 ms/step lower bound just from per-op Rust↔mlx-c crossing.
                // If actual FFI cost is much lower than that, then the 2.20× gap
                // vs mlx-lm is elsewhere (compile coverage, scheduling).
                let count_ops_enabled = std::env::var("LUMEN_GEMMA4_COUNT_OPS")
                    .map(|v| v == "1")
                    .unwrap_or(false);
                let breakdown_enabled = std::env::var("LUMEN_GEMMA4_COUNT_OPS_BREAKDOWN")
                    .map(|v| v == "1")
                    .unwrap_or(false);
                let timing_enabled = std::env::var("LUMEN_GEMMA4_TIME_OPS_BREAKDOWN")
                    .map(|v| v == "1")
                    .unwrap_or(false);
                if count_ops_enabled {
                    if breakdown_enabled {
                        mlx_rs::utils::enable_op_breakdown();
                    }
                    if timing_enabled {
                        mlx_rs::utils::enable_op_timing();
                    }
                }
                let mut total_ops: usize = 0;
                let mut total_step_ms: f64 = 0.0;
                let mut step_samples: usize = 0;

                // kernel-cache hit/miss counters
                // (MLX patched — see mlx::core::metal::Device::kernel_cache_*).
                // LUMEN_KERNEL_CACHE_STATS=1 resets counters before the decode
                // loop, prints per-step deltas, and emits a summary at the end.
                // Lets us test the "shape-specialized kernel cache miss"
                // hypothesis: if our misses/step is much higher than mlx-lm's,
                // we have direct evidence of H1.
                let kcache_stats_enabled = std::env::var("LUMEN_KERNEL_CACHE_STATS")
                    .map(|v| v == "1")
                    .unwrap_or(false);
                if kcache_stats_enabled {
                    mlx_rs::metal::reset_kernel_cache_stats()
                        .context("chat_streaming: reset_kernel_cache_stats")?;
                    mlx_rs::metal::reset_cmd_buffer_stats()
                        .context("chat_streaming: reset_cmd_buffer_stats")?;
                    mlx_rs::metal::reset_scheduler_stats()
                        .context("chat_streaming: reset_scheduler_stats")?;
                    mlx_rs::metal::reset_eval_gpu_stats()
                        .context("chat_streaming: reset_eval_gpu_stats")?;
                    mlx_rs::metal::reset_prim_histogram()
                        .context("chat_streaming: reset_prim_histogram")?;
                    mlx_rs::metal::reset_prim_histogram_dynamic()
                        .context("chat_streaming: reset_prim_histogram_dynamic")?;
                    mlx_rs::metal::reset_astype_pair_stats()
                        .context("chat_streaming: reset_astype_pair_stats")?;
                }
                let mut prev_hits: u64 = 0;
                let mut prev_misses: u64 = 0;
                let mut prev_commits: u64 = 0;
                let mut prev_cbops: u64 = 0;
                let mut prev_sched_nt: u64 = 0;
                let mut prev_sched_cc: u64 = 0;
                let mut prev_sched_wait: u64 = 0;

                // Metal GPU trace capture.
                // LUMEN_METAL_CAPTURE=<path.gputrace> → start capture before
                // the first decode step. LUMEN_METAL_CAPTURE_STEPS=<N>
                // controls how many decode steps to record (default 10) before
                // stop_capture() fires. The .gputrace bundle can then be opened
                // in Xcode for frame-by-frame comparison against mlx-lm.
                let capture_path = std::env::var("LUMEN_METAL_CAPTURE").ok();
                let capture_steps: usize = std::env::var("LUMEN_METAL_CAPTURE_STEPS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(10);
                if let Some(ref p) = capture_path {
                    mlx_rs::metal::start_capture(p)
                        .context("chat_streaming: metal::start_capture")?;
                    eprintln!("[metal-capture] started: path={p} capture_steps={capture_steps}");
                }
                let mut capture_stopped = false;

                let mut count = 0usize;
                loop {
                    if count + 1 == max_new_tokens {
                        // Last token: read sync, emit, done.
                        let token = self
                            .model
                            .read_token_u32(&current)
                            .context("chat_streaming: read final token")?;
                        parser.push(token)?;
                        let chunk = self.chat.decode(&[token], true)?;
                        if !chunk.is_empty() {
                            on_token(&chunk)?;
                        }
                        break;
                    }

                    if count_ops_enabled {
                        mlx_rs::utils::reset_op_counter();
                    }
                    let step_start = std::time::Instant::now();

                    // Schedule step N+1's forward + argmax (uses lazy `current`).
                    // forward_array_last_token: L=1 path is a no-op slice, so this
                    // is identical to forward_array() at decode. Wired through for
                    // API consistency with the prefill path.
                    let next_logits = self
                        .model
                        .forward_array_last_token(&current, &mut cache)
                        .context("chat_streaming: decode forward_array_last_token")?;
                    let next_lazy = self
                        .model
                        .argmax_last_token_lazy(&next_logits)
                        .context("chat_streaming: decode argmax_lazy")?;
                    mlx_rs::transforms::async_eval([&next_lazy])
                        .context("chat_streaming: decode async_eval")?;

                    // Sync read step N's token while GPU is computing step N+1.
                    let token = self
                        .model
                        .read_token_u32(&current)
                        .context("chat_streaming: read token")?;

                    if count_ops_enabled {
                        let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;
                        let n = mlx_rs::utils::read_op_counter();
                        total_ops += n;
                        total_step_ms += step_ms;
                        step_samples += 1;
                        eprintln!(
                            "[gemma4-ffi] step={count} ops={n} step_ms={step_ms:.2} \
                         (forward+argmax+sync_read)"
                        );
                    }

                    if kcache_stats_enabled {
                        let (hits, misses) = mlx_rs::metal::kernel_cache_stats().unwrap_or((0, 0));
                        let (commits, cbops) = mlx_rs::metal::cmd_buffer_stats().unwrap_or((0, 0));
                        let (sched_nt, sched_cc, sched_wait, sched_max_act) =
                            mlx_rs::metal::scheduler_stats().unwrap_or((0, 0, 0, 0));
                        let d_hits = hits.saturating_sub(prev_hits);
                        let d_misses = misses.saturating_sub(prev_misses);
                        let d_commits = commits.saturating_sub(prev_commits);
                        let d_cbops = cbops.saturating_sub(prev_cbops);
                        let d_sched_nt = sched_nt.saturating_sub(prev_sched_nt);
                        let d_sched_cc = sched_cc.saturating_sub(prev_sched_cc);
                        let d_sched_wait = sched_wait.saturating_sub(prev_sched_wait);
                        let total = d_hits + d_misses;
                        let miss_rate = if total > 0 {
                            100.0 * d_misses as f64 / total as f64
                        } else {
                            0.0
                        };
                        let ops_per_buf = if d_commits > 0 {
                            d_cbops as f64 / d_commits as f64
                        } else {
                            0.0
                        };
                        let d_sched_wait_us = d_sched_wait as f64 / 1000.0;
                        eprintln!(
                            "[gemma4-kcache] step={count} hits={d_hits} misses={d_misses} \
                         miss_rate={miss_rate:.2}% | cmd_bufs={d_commits} \
                         ops_per_buf={ops_per_buf:.1} | sched_nt={d_sched_nt} \
                         sched_cc={d_sched_cc} sched_wait_us={d_sched_wait_us:.2} \
                         sched_max_act={sched_max_act}"
                        );
                        prev_hits = hits;
                        prev_misses = misses;
                        prev_commits = commits;
                        prev_cbops = cbops;
                        prev_sched_nt = sched_nt;
                        prev_sched_cc = sched_cc;
                        prev_sched_wait = sched_wait;
                    }

                    parser.push(token)?;
                    count += 1;
                    let chunk = self.chat.decode(&[token], true)?;
                    if !chunk.is_empty() {
                        on_token(&chunk)?;
                    }

                    // Stop Metal capture after `capture_steps` decode steps.
                    // We do this AFTER reading the token so the synchronously
                    // completed work is included in the trace, then stop before
                    // the next forward to avoid flooding the .gputrace bundle.
                    if capture_path.is_some() && !capture_stopped && count >= capture_steps {
                        mlx_rs::metal::stop_capture()
                            .context("chat_streaming: metal::stop_capture")?;
                        eprintln!(
                            "[metal-capture] stopped after {count} decode steps; \
                         open in Xcode to inspect"
                        );
                        capture_stopped = true;
                    }

                    if eos.contains(&token) {
                        break;
                    }
                    current = next_lazy;
                }

                // Defensive: ensure capture is stopped even if loop ends early
                // (e.g. EOS before capture_steps reached).
                if capture_path.is_some() && !capture_stopped {
                    mlx_rs::metal::stop_capture()
                        .context("chat_streaming: metal::stop_capture (end-of-loop)")?;
                    eprintln!("[metal-capture] stopped at end-of-loop ({count} steps recorded)");
                }

                if kcache_stats_enabled && count > 0 {
                    let (hits, misses) = mlx_rs::metal::kernel_cache_stats().unwrap_or((0, 0));
                    let (commits, cbops) = mlx_rs::metal::cmd_buffer_stats().unwrap_or((0, 0));
                    let (sched_nt, sched_cc, sched_wait, sched_max_act) =
                        mlx_rs::metal::scheduler_stats().unwrap_or((0, 0, 0, 0));
                    let (eval_calls, eval_ns) = mlx_rs::metal::eval_gpu_stats().unwrap_or((0, 0));
                    let total = hits + misses;
                    let miss_rate = if total > 0 {
                        100.0 * misses as f64 / total as f64
                    } else {
                        0.0
                    };
                    let h_per_step = hits as f64 / count as f64;
                    let m_per_step = misses as f64 / count as f64;
                    let bufs_per_step = commits as f64 / count as f64;
                    let ops_per_buf = if commits > 0 {
                        cbops as f64 / commits as f64
                    } else {
                        0.0
                    };
                    let sched_wait_us = sched_wait as f64 / 1000.0;
                    let sched_wait_us_per_step = sched_wait_us / count as f64;
                    let sched_lock_calls = sched_nt + sched_cc;
                    let sched_ns_per_acquire = if sched_lock_calls > 0 {
                        sched_wait as f64 / sched_lock_calls as f64
                    } else {
                        0.0
                    };
                    let eval_us_total = eval_ns as f64 / 1000.0;
                    let eval_us_per_step = eval_us_total / count as f64;
                    let eval_us_per_call = if eval_calls > 0 {
                        eval_ns as f64 / 1000.0 / eval_calls as f64
                    } else {
                        0.0
                    };
                    let eval_calls_per_step = eval_calls as f64 / count as f64;
                    let (
                        h_rms,
                        h_qmm,
                        h_reshape,
                        h_broadcast,
                        h_multiply,
                        h_transpose,
                        h_compiled,
                        h_other,
                    ) = mlx_rs::metal::prim_histogram().unwrap_or((0, 0, 0, 0, 0, 0, 0, 0));
                    let c = count as f64;
                    eprintln!(
                        "[gemma4-primhist] /step: RMSNorm={:.1} QuantizedMatmul={:.1} \
                     Reshape={:.1} Broadcast={:.1} Multiply={:.1} \
                     Transpose={:.1} Compiled={:.1} Other={:.1}",
                        h_rms as f64 / c,
                        h_qmm as f64 / c,
                        h_reshape as f64 / c,
                        h_broadcast as f64 / c,
                        h_multiply as f64 / c,
                        h_transpose as f64 / c,
                        h_compiled as f64 / c,
                        h_other as f64 / c,
                    );
                    // F3: AsType dtype-pair breakdown.
                    if let Ok((b2f, f2b, noop, other)) = mlx_rs::metal::astype_pair_stats() {
                        eprintln!(
                            "[gemma4-astype-pairs] /step: bf16->f32={:.1} \
                         f32->bf16={:.1} noop={:.1} other={:.1}",
                            b2f as f64 / c,
                            f2b as f64 / c,
                            noop as f64 / c,
                            other as f64 / c,
                        );
                    }
                    // Dynamic primitive-type breakdown — every distinct
                    // primitive name with its per-step rate. Lets us pinpoint
                    // which specific ops account for the "Other" 1.8k/step.
                    if let Ok(dump) = mlx_rs::metal::prim_histogram_dynamic() {
                        let mut entries: Vec<(String, u64)> = dump
                            .lines()
                            .filter_map(|line| {
                                let mut parts = line.splitn(2, '=');
                                let name = parts.next()?.to_string();
                                let cnt: u64 = parts.next()?.parse().ok()?;
                                Some((name, cnt))
                            })
                            .collect();
                        entries.sort_by(|a, b| b.1.cmp(&a.1));
                        let max_show = 25.min(entries.len());
                        eprintln!("[gemma4-primhist-dyn] top-{max_show} per-step:");
                        for (name, cnt) in entries.iter().take(max_show) {
                            let per_step = *cnt as f64 / c;
                            eprintln!("    {name:<30} total={cnt:>7} per_step={per_step:>7.1}");
                        }
                    }
                    eprintln!(
                        "[gemma4-kcache-summary] decode_steps={count} \
                     total_hits={hits} total_misses={misses} \
                     hits/step={h_per_step:.1} misses/step={m_per_step:.1} \
                     overall_miss_rate={miss_rate:.2}% | \
                     total_cmd_bufs={commits} bufs/step={bufs_per_step:.2} \
                     ops_per_buf={ops_per_buf:.1} | \
                     sched_lock_acquires={sched_lock_calls} \
                     sched_wait_us_total={sched_wait_us:.1} \
                     sched_wait_us/step={sched_wait_us_per_step:.2} \
                     sched_ns/acquire={sched_ns_per_acquire:.1} \
                     sched_max_active={sched_max_act} | \
                     eval_gpu_calls={eval_calls} \
                     eval_calls/step={eval_calls_per_step:.1} \
                     eval_us/step={eval_us_per_step:.1} \
                     eval_us/call={eval_us_per_call:.2}"
                    );
                }

                if count_ops_enabled && step_samples > 0 {
                    let avg_ops = (total_ops as f64) / (step_samples as f64);
                    let avg_step_ms = total_step_ms / (step_samples as f64);
                    let avg_ffi_us_per_op = if avg_ops > 0.0 {
                        (avg_step_ms * 1000.0) / avg_ops
                    } else {
                        0.0
                    };
                    eprintln!(
                        "[gemma4-ffi-summary] steps={step_samples} avg_ops={avg_ops:.1} \
                     avg_step_ms={avg_step_ms:.2} \
                     avg_us_per_op={avg_ffi_us_per_op:.2} \
                     (upper bound — includes GPU compute, not pure FFI cost)"
                    );
                    if breakdown_enabled {
                        let breakdown = mlx_rs::utils::take_op_breakdown();
                        let total: usize = breakdown.iter().map(|(_, c)| c).sum();
                        let cumulative_avg = (total as f64) / (step_samples as f64);
                        eprintln!(
                            "[gemma4-ffi-breakdown] total_ops_recorded={total} (avg/step={cumulative_avg:.1}) top-30 callsites:"
                        );
                        for (loc, count) in breakdown.iter().take(30) {
                            let pct = (*count as f64) / (total as f64) * 100.0;
                            let per_step = (*count as f64) / (step_samples as f64);
                            eprintln!(
                                "  {count:>7}  ({pct:>5.1}%)  per_step={per_step:>6.1}  {loc}"
                            );
                        }
                    }
                    if timing_enabled {
                        let timing = mlx_rs::utils::take_op_timing();
                        let total_ns: u128 = timing.iter().map(|(_, _, t)| *t).sum();
                        let total_ms = (total_ns as f64) / 1_000_000.0;
                        eprintln!(
                            "[gemma4-ffi-timing] total_ffi_ms={total_ms:.2} (avg/step={:.2}) top-30 by wall:",
                            total_ms / step_samples as f64,
                        );
                        for (loc, count, ns) in timing.iter().take(30) {
                            let ms = (*ns as f64) / 1_000_000.0;
                            let per_op_us = (*ns as f64) / (*count as f64) / 1000.0;
                            let pct = ms / total_ms * 100.0;
                            eprintln!(
                                "  {ms:>7.2}ms  ({pct:>5.1}%)  count={count:>6}  per_op={per_op_us:>6.2}μs  {loc}"
                            );
                        }
                    }
                }

                parser.finalize()
            }) // close with_new_default_stream
        }
    }

    // ───────────────────────── tests ─────────────────────────

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::path::Path;

        const LMSTUDIO_DIR: &str = "/path/to/models/gemma-4-26b-a4b-mlx-4bit";

        fn dir_present() -> bool {
            Path::new(LMSTUDIO_DIR).exists()
        }

        #[test]
        #[ignore = "requires lmstudio shards (~16 GB) + Metal"]
        fn backend_load_and_basic_chat() {
            if !dir_present() {
                eprintln!("skip: lmstudio model not present");
                return;
            }
            let mut backend =
                Gemma4Backend::from_dir("gemma-4-26b-a4b", LMSTUDIO_DIR).expect("load");
            assert_eq!(backend.model_id(), "gemma-4-26b-a4b");

            let msgs = vec![("user".to_string(), "Say hi in one word.".to_string())];
            let resp = backend.chat(&msgs, 8, 0.0, false).expect("chat");
            eprintln!(
                "[backend-chat] visible={:?} reasoning={:?} tools={}",
                resp.visible,
                resp.reasoning,
                resp.tool_calls.len()
            );
            assert!(
                !resp.visible.is_empty() || !resp.reasoning.is_empty(),
                "model should produce *some* content"
            );
        }

        #[test]
        #[ignore = "requires lmstudio shards (~16 GB) + Metal"]
        fn backend_chat_streaming_delivers_chunks() {
            if !dir_present() {
                eprintln!("skip: lmstudio model not present");
                return;
            }
            let mut backend =
                Gemma4Backend::from_dir("gemma-4-26b-a4b", LMSTUDIO_DIR).expect("load");

            let msgs = vec![("user".to_string(), "Say hi in one word.".to_string())];
            let mut chunks: Vec<String> = Vec::new();
            let resp = backend
                .chat_streaming(&msgs, 8, false, |chunk| {
                    chunks.push(chunk.to_string());
                    Ok(())
                })
                .expect("chat_streaming");

            eprintln!(
                "[backend-stream] chunks={:?} final={:?}",
                chunks, resp.visible
            );
            assert!(
                !chunks.is_empty(),
                "streaming should deliver at least one chunk"
            );
        }

        #[test]
        #[ignore = "requires lmstudio shards (~16 GB) + Metal"]
        fn backend_build_chat_input_starts_with_bos() {
            if !dir_present() {
                eprintln!("skip: lmstudio model not present");
                return;
            }
            let backend = Gemma4Backend::from_dir("gemma-4-26b-a4b", LMSTUDIO_DIR).expect("load");
            let msgs = vec![("user".to_string(), "hi".to_string())];
            let ids = backend.build_chat_input(&msgs, false).expect("build");
            assert_eq!(ids[0], 2, "BOS=2 first");
        }
    }
}
