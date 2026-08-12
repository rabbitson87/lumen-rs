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
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, OnceLock};
    use std::time::Instant;

    use crate::chat_io::BackendStreamEvent;
    use crate::gemma4_chat::imp::{ChatMessage, ChatRole, Gemma4ChatTemplate, RenderOptions};
    use crate::gemma4_critical_correction::CorrectionTable;
    use crate::grammar::{Gemma4GrammarState, GrammarMode, shared_factory_from_tokenizer};
    use crate::jinja_chat::imp::{JinjaChatTemplate, JinjaRenderOptions};
    use llguidance::ParserFactory;

    /// Phase B (v0.6.0) — env gate for the runtime logit-correction kernel.
    ///
    /// Off by default. When set to a truthy value AND the model directory
    /// has a valid `logit_corrections.bin` sidecar, the backend captures
    /// `h_for_lm_head` at each decode step and applies per-critical-token
    /// corrections immediately after the CPU pull, before any masks
    /// (grammar, EOS guard, DRY) modify the logit buffer.
    ///
    /// Falsy values (`0`, `false`, empty): no capture, no correction —
    /// bit-identical to v0.5.x decode.
    fn gemma4_critical_logit_correction_enabled() -> bool {
        match std::env::var("LUMEN_GEMMA4_CRITICAL_LOGIT_CORRECTION").ok() {
            Some(s) => !s.is_empty() && s != "0" && !s.eq_ignore_ascii_case("false"),
            None => false,
        }
    }

    /// Background worker that drops MLX's reusable Metal-buffer pool after each
    /// chat completion. MLX keeps freed buffers in a reuse cache that grows with
    /// every request's KV-sized allocations; without a periodic clear, process
    /// RAM climbs monotonically across independent requests until unified memory
    /// swaps and decode collapses (observed: 35 → 2.5 tok/s on a 24 GB Mac with
    /// gemma-4-26b-a4b, restored only by a server restart). The gemma4 streaming
    /// path uses a stack-local cache and never reaches `remove_seq`, so the clear
    /// is funnelled through `log_chat_done` instead — the one point every chat
    /// path (streaming + non-streaming) passes through.
    ///
    /// Mirrors the proven Qwen3.5 native-path fix (`runner_native.rs`'s
    /// `CacheCleanerWorker`); kept self-contained here to avoid coupling across
    /// the two feature-gated `imp` modules. Deferred to this thread so the
    /// post-DONE flush isn't blocked by the ~45 ms clear at cache-saturated
    /// steady state. Race-safe: `mlx_clear_cache` only frees already-released
    /// pool buffers, never in-flight tensors.
    struct Gemma4CacheCleaner {
        tx: std::sync::mpsc::Sender<()>,
    }

    impl Gemma4CacheCleaner {
        fn global() -> &'static Self {
            static WORKER: OnceLock<Gemma4CacheCleaner> = OnceLock::new();
            WORKER.get_or_init(|| {
                let (tx, rx) = std::sync::mpsc::channel::<()>();
                std::thread::Builder::new()
                    .name("gemma4-cache-cleaner".into())
                    .spawn(move || {
                        while rx.recv().is_ok() {
                            // Coalesce bursts: one clear settles the pool, so
                            // drain any queued requests before running it.
                            while rx.try_recv().is_ok() {}
                            unsafe {
                                mlx_sys::mlx_clear_cache();
                            }
                        }
                    })
                    .expect("spawn gemma4-cache-cleaner thread");
                Gemma4CacheCleaner { tx }
            })
        }

        fn request_clear(&self) {
            // Fire-and-forget; channel error only at process teardown.
            let _ = self.tx.send(());
        }
    }

    /// Reclaim MLX's buffer pool after a chat completion (see
    /// [`Gemma4CacheCleaner`]). Default ON; shares the Qwen3.5 path's toggles so
    /// one env var controls both native backends: opt out with
    /// `LUMEN_NATIVE_NO_CLEAR_CACHE=1`, force synchronous (block the flush, for
    /// A/B or mem-probe interleaving) with `LUMEN_NATIVE_DEFER_CLEAR_CACHE=0`.
    fn reclaim_mlx_cache_after_chat() {
        if std::env::var("LUMEN_NATIVE_NO_CLEAR_CACHE").is_ok() {
            return;
        }
        let defer = std::env::var("LUMEN_NATIVE_DEFER_CLEAR_CACHE")
            .map(|v| v != "0")
            .unwrap_or(true);
        if defer {
            Gemma4CacheCleaner::global().request_clear();
        } else {
            unsafe {
                mlx_sys::mlx_clear_cache();
            }
        }
    }

    /// Dump rendered prompt to stderr when `LUMEN_DUMP_PROMPT` is set.
    /// Set to `1` / `true` for compact dump (decoded text + length), set to
    /// `full` for verbose dump (token IDs + decoded text, no truncation).
    ///
    /// Useful for diagnosing chat-template rendering bugs without rebuilding
    /// the model — the dump shows exactly what the model sees, including all
    /// special tokens (`<|turn>`, `<|tool_call>`, `<|tool_response>`, etc.)
    /// in their human-readable form.
    fn maybe_dump_prompt(chat: &Gemma4ChatTemplate, ids: &[u32], origin: &'static str) {
        let mode = match std::env::var("LUMEN_DUMP_PROMPT").ok() {
            Some(s) if !s.is_empty() && s != "0" && !s.eq_ignore_ascii_case("false") => s,
            _ => return,
        };
        let full = mode.eq_ignore_ascii_case("full");
        let decoded = chat
            .decode(ids, /* skip_special_tokens = */ false)
            .unwrap_or_else(|e| format!("<decode error: {e:#}>"));
        eprintln!(
            "[dump-prompt:{origin}] n_tokens={} text_bytes={}{}",
            ids.len(),
            decoded.len(),
            if full || decoded.len() <= 4096 {
                ""
            } else {
                " (truncated — set LUMEN_DUMP_PROMPT=full for verbose)"
            }
        );
        eprintln!("[dump-prompt:{origin}] ---BEGIN---");
        if full || decoded.len() <= 4096 {
            eprintln!("{decoded}");
        } else {
            // UTF-8 safe truncation: snap byte offsets down/up to the
            // nearest `char_boundary` so we never slice mid-character
            // (Korean / Japanese / emoji are multi-byte). Without this
            // the worker panics with "byte index N is not a char
            // boundary" and the request fails with empty response.
            let mut head_end = 2048usize.min(decoded.len());
            while head_end > 0 && !decoded.is_char_boundary(head_end) {
                head_end -= 1;
            }
            let mut tail_start = decoded.len().saturating_sub(2048);
            while tail_start < decoded.len() && !decoded.is_char_boundary(tail_start) {
                tail_start += 1;
            }
            eprintln!("{}", &decoded[..head_end]);
            eprintln!(
                "...<{} bytes elided>...",
                tail_start.saturating_sub(head_end)
            );
            eprintln!("{}", &decoded[tail_start..]);
        }
        eprintln!("[dump-prompt:{origin}] ---END---");
        if full {
            eprintln!("[dump-prompt:{origin}] token_ids = {ids:?}");
        }
    }

    use crate::gemma4_moe::imp::{GenerateConfig, NativeGemma4Model, NativeGemma4PromptCache};
    use crate::gemma4_response::imp::{ParseState, ParsedResponse, ResponseParser};
    use crate::gemma4_sampling::imp::SamplingConfig;
    use crate::kv_disk::{DiskKvStore, KvManifest};

    /// Builds a `SamplingConfig` from request-supplied `temperature`/`top_p`
    /// plus operator-supplied env (`REPEAT_PENALTY`, `LUMEN_REPEAT_LAST_N`,
    /// `LUMEN_TOP_K`, `LUMEN_SAMPLE_SEED`). Returns `None` when the result
    /// is greedy
    /// (temperature ≤ 0 AND repeat_penalty == 1) so the decode loop can
    /// take the existing fast path.
    ///
    /// Defaults reflect the OpenAI request defaults (`0.7` / `0.9`) when
    /// the request didn't set them; `REPEAT_PENALTY` defaults to `1.1`
    /// (matching Ollama's gemma4 default) to suppress degenerate
    /// repetition loops on 4-bit quantized weights. Operators can set
    /// `REPEAT_PENALTY=1.0` in the app's SERVER card to restore the
    /// no-penalty behavior. NOTE: a non-1.0 penalty makes `is_greedy()`
    /// false, so even `temperature=0` requests route through the CPU
    /// sampling pipeline (near-greedy, but with the penalty applied).
    /// Emit the canonical `[gemma4] chat done:` log split into prefill
    /// (prompt-processing), decode (per-step generation), and the
    /// composite end-to-end rate.
    ///
    /// The user-perceived "answer is slow" complaint usually has a
    /// specific bottleneck — short prompt + long answer = decode-bound;
    /// long prompt + short answer = prefill-bound. Reporting only the
    /// decode rate (the previous behavior) hid which one was the
    /// problem and tempted users to chase the wrong knob (`bits`,
    /// `sliding`, quant level — none of which help if prefill is the
    /// limiter).
    ///
    /// Output shape (the order matters — `lumen-app`'s
    /// `parse_tok_per_sec` picks the LAST `(N.N tok/s)` group, so
    /// putting e2e last surfaces it in the UI):
    ///
    ///   `[gemma4] chat done: prefill P tok in Pms ({P_tps} tok/s) | decode D tok in Dms ({D_tps} tok/s) | e2e E tok in Ems ({E_tps} tok/s)`
    ///
    /// Where E_tps uses `decode_tokens / (prefill_ms + decode_ms)` —
    /// the e2e rate from the user's perspective is determined by
    /// answer tokens produced over total wall-clock, not (prompt +
    /// answer) / wall-clock.
    fn log_chat_done(prefill_tokens: usize, prefill_ms: f64, decode_tokens: usize, decode_ms: f64) {
        let p_tps = if prefill_ms > 0.0 && prefill_tokens > 0 {
            prefill_tokens as f64 / (prefill_ms / 1000.0)
        } else {
            0.0
        };
        let d_tps = if decode_ms > 0.0 && decode_tokens > 0 {
            decode_tokens as f64 / (decode_ms / 1000.0)
        } else {
            0.0
        };
        let e2e_ms = prefill_ms + decode_ms;
        let e_tps = if e2e_ms > 0.0 && decode_tokens > 0 {
            decode_tokens as f64 / (e2e_ms / 1000.0)
        } else {
            0.0
        };
        eprintln!(
            "[gemma4] chat done: prefill {prefill_tokens} tok in {prefill_ms:.0}ms ({p_tps:.1} tok/s) | decode {decode_tokens} tok in {decode_ms:.0}ms ({d_tps:.1} tok/s) | e2e {decode_tokens} tok in {e2e_ms:.0}ms ({e_tps:.1} tok/s)"
        );
        // Every gemma4 chat path funnels through here, so this is the single
        // point to reclaim MLX's buffer pool and stop the per-request RAM climb
        // that swaps the box and collapses decode (35 → 2.5 tok/s) until restart.
        reclaim_mlx_cache_after_chat();
    }

    fn build_sampling_config(
        temperature: f32,
        top_p: f32,
        ov: &crate::SamplingOverrides,
    ) -> Option<SamplingConfig> {
        // Temperature parity with Ollama `gemma4:26b-mlx` (Modelfile sets
        // `temperature 1`). The server's serde default is 0.7 (see
        // `default_temperature()` in lumen-server). When a request arrives with
        // *exactly* that value the client almost certainly omitted the field,
        // so we substitute the Gemma 4 default (1.0). An explicit non-default
        // temperature — including `0.0` for greedy/grammar/structured paths — is
        // always honored. temp=0.7 was too peaky: on hard reasoning prompts the
        // model failed to escape an n-gram cycle / never sampled `<channel|>`
        // (101) to close thinking, where Ollama (temp=1.0) converged. Override
        // the Gemma 4 default via `LUMEN_TEMPERATURE`. Gemma 4-scoped only —
        // 1.0 is too hot for Qwen, so the global serde default stays 0.7.
        let gemma4_default_temp: f32 = std::env::var("LUMEN_TEMPERATURE")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .filter(|v: &f32| *v >= 0.0)
            .unwrap_or(1.0);
        let temperature = if (temperature - 0.7).abs() < 1e-6 {
            gemma4_default_temp
        } else {
            temperature
        };
        let repeat_penalty: f32 = ov.repeat_penalty.unwrap_or_else(|| {
            std::env::var("REPEAT_PENALTY")
                .ok()
                .and_then(|s| s.parse().ok())
                .filter(|v: &f32| *v > 0.0)
                .unwrap_or(1.1)
        });
        let repeat_penalty_last_n: usize = std::env::var("LUMEN_REPEAT_LAST_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(64);
        // Top-k sampling clamp — default 64 to match the published Gemma 4
        // model params (Ollama `gemma4:26b-mlx` Modelfile sets `top_k 64`;
        // 40 is only Ollama's *global* default, which that Modelfile
        // overrides). top_k=40 was too tight: on hard prompts it left too few
        // candidates to escape an n-gram repetition cycle, so agentic+thinking
        // generations degenerated where Ollama (top_k=64) converged. Verified
        // by reading Ollama's renderer (identical prompt) + `ollama show
        // gemma4:26b-mlx`. Set `LUMEN_TOP_K=0` to disable, or override freely.
        let top_k: usize = ov.top_k.unwrap_or_else(|| {
            std::env::var("LUMEN_TOP_K")
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(64)
        });
        let seed: u64 = ov.seed.unwrap_or_else(|| {
            std::env::var("LUMEN_SAMPLE_SEED")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(|| {
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_nanos() as u64)
                        .unwrap_or(0x9E3779B97F4A7C15)
                })
        });
        let cfg = SamplingConfig {
            temperature,
            top_p,
            top_k,
            repeat_penalty,
            repeat_penalty_last_n,
            presence_penalty: ov.presence_penalty.unwrap_or(0.0),
            frequency_penalty: ov.frequency_penalty.unwrap_or(0.0),
            min_p: ov.min_p.unwrap_or(0.0),
            seed,
            dry: lumen_core::dry::dry_config_from_env(),
        };
        if cfg.is_greedy() { None } else { Some(cfg) }
    }

    /// Phase 1.5: predicate guarding visible-text emission in
    /// `chat_streaming`. Returns `true` if the just-pushed token is the
    /// `<|tool_call>` open sentinel, lives inside a tool-call body, or
    /// closes one with `<tool_call|>`. We MUST NOT fire the visible-text
    /// callback in any of those cases — the tool-call body is surfaced
    /// separately via `ParsedResponse.tool_calls` after generation
    /// finishes and (in streaming mode) via the `ToolCall*` envelope.
    /// Otherwise SSE clients would see the raw `call:NAME{...}` body
    /// streamed as visible content AND the tool_calls envelope, double-
    /// rendering the same data.
    #[inline]
    fn is_tool_call_boundary(before: ParseState, after: ParseState) -> bool {
        matches!(before, ParseState::ToolCall) || matches!(after, ParseState::ToolCall)
    }

    /// Phase 1.6c: shared per-token event dispatch. Replaces the
    /// previous inline visible-token suppression at each callback
    /// site. Three cases:
    ///   1. Visible→Visible token (regular text). Decode with
    ///      `skip_special=true` and emit `BackendStreamEvent::Text`.
    ///   2. Inside `<|tool_call>...<tool_call|>` body. Decode with
    ///      `skip_special=false` so the body chars reach the parser,
    ///      feed `ResponseParser::observe_tool_text_fragment`, and
    ///      fire `BackendStreamEvent::ToolCallStart{name}` the moment
    ///      `call:NAME{` becomes parseable. (Args body keeps
    ///      buffering inside the parser; the engine emits one
    ///      `ArgumentsDelta` chunk at `<tool_call|>` close.)
    ///   3. Boundary tokens (`<|tool_call>` open / `<tool_call|>`
    ///      close). Suppressed — no event fires.
    ///      Per-token stderr trace. Enabled when `LUMEN_GEMMA4_TOKEN_TRACE` is
    ///      set to a non-empty / non-`0` / non-`false` value. Prints one line
    ///      per sampled token with id + decoded text (special tokens visible)
    /// + parser state transition. Use for debugging reasoning runaway,
    ///   tool-call structure, or channel close failures.
    ///
    /// Cheap when off (single env lookup per call, cached on first hit
    /// via a `OnceLock`). Bounded cost when on (one decode per token,
    /// ~50 µs at vocab 262144).
    fn gemma4_token_trace_enabled() -> bool {
        static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *CACHED.get_or_init(|| {
            std::env::var("LUMEN_GEMMA4_TOKEN_TRACE")
                .map(|v| !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false"))
                .unwrap_or(false)
        })
    }

    fn emit_token_event(
        chat: &Gemma4ChatTemplate,
        parser: &mut ResponseParser<'_>,
        token: u32,
        state_before: ParseState,
        on_event: &mut impl FnMut(BackendStreamEvent<'_>) -> Result<()>,
    ) -> Result<()> {
        let state_after = parser.state();
        if gemma4_token_trace_enabled() {
            // Decode WITH specials so `<|channel>` / `<turn|>` / `<|tool_call>`
            // boundary markers stay visible — that's the whole point of a
            // trace dump, you want to see exactly when the model switched
            // channels.
            let text = chat
                .decode(&[token], /* skip_special */ false)
                .unwrap_or_else(|_| String::from("<decode-err>"));
            eprintln!("[token-trace] id={token:>6} {state_before:?}→{state_after:?} text={text:?}");
        }
        let in_tool_call_span = matches!(state_before, ParseState::ToolCall)
            || matches!(state_after, ParseState::ToolCall);
        let in_reasoning_span = matches!(state_before, ParseState::Reasoning)
            || matches!(state_after, ParseState::Reasoning);
        if !in_tool_call_span && !in_reasoning_span {
            // Pure visible-channel token.
            let chunk = chat.decode(&[token], true)?;
            if !chunk.is_empty() {
                on_event(BackendStreamEvent::Text(&chunk))?;
            }
            return Ok(());
        }
        // Reasoning-channel token. Emit as `Reasoning` so the HTTP layer
        // can wrap with `<think>…</think>` envelope AND populate the
        // OpenAI `delta.reasoning` field — clients (Ayla UI text-tag
        // parser, vLLM-spec OpenAI clients) see thinking progress in
        // real time instead of an apparently-stuck stream.
        if in_reasoning_span && !in_tool_call_span {
            // Skip the boundary tokens themselves (channel open/close)
            // — they decode to the empty string under skip_special=true
            // but emitting an empty Reasoning event would be noise.
            if matches!(state_before, ParseState::Reasoning)
                && matches!(state_after, ParseState::Reasoning)
            {
                let chunk = chat.decode(&[token], /* skip_special */ true)?;
                if !chunk.is_empty() {
                    on_event(BackendStreamEvent::Reasoning(&chunk))?;
                }
            }
            return Ok(());
        }
        // We're inside a tool_call span. Suppress the visible-channel
        // emit (would leak the body into delta.content), but feed the
        // body chars to the parser so it can detect `call:NAME{` and
        // fire the early ToolCallStart event.
        if matches!(state_after, ParseState::ToolCall) {
            let body_chunk = chat.decode(&[token], /* skip_special */ false)?;
            if !body_chunk.is_empty()
                && let Some(name) = parser.observe_tool_text_fragment(&body_chunk)
            {
                on_event(BackendStreamEvent::ToolCallStart {
                    name: name.as_str(),
                })?;
            }
        }
        Ok(())
    }

    impl Gemma4Backend {
        /// Channel-aware decode for the batched streaming scheduler.
        ///
        /// The batched `run_batched_mlx` loop emits incremental deltas by
        /// decoding the full `generated` prefix and diffing against the prior
        /// text. A flat `decode()` strips the special `<|channel>`/`<channel|>`
        /// boundary markers but leaves the *reasoning content between them* in
        /// the visible text — so naive diffing leaks `thought` into
        /// `delta.content`. This reuses the exact same `ResponseParser` state
        /// machine the sequential path uses, partitioning the prefix into the
        /// visible and reasoning channels and decoding each independently.
        /// Returns `(visible, reasoning)` so the scheduler can diff each against
        /// its own prior length and route to `Delta` / `ReasoningDelta`.
        ///
        /// Re-parsing the whole prefix each step is `O(n)`; for the typical
        /// 256-token batched chat output the cost is negligible versus the
        /// ~10 ms GPU step, and the cumulative-diff guarantees the same
        /// byte-for-byte stream the sequential path produces.
        pub fn stream_channels(&self, generated: &[u32]) -> Result<(String, String)> {
            let mut parser = ResponseParser::new(&self.chat);
            for &tok in generated {
                parser.push(tok)?;
            }
            let parsed = parser.finalize()?;
            Ok((parsed.visible, parsed.reasoning))
        }
    }

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
    /// Flag for the Lark-format Gemma 4 tool-call grammar. **Default ON**
    /// (v0.7.0+). When active (env unset or truthy),
    /// [`Gemma4Backend::build_grammar_state`] constructs a matcher that
    /// constrains tool-call bodies to the native
    /// `call:NAME{key:value,…}` format with per-tool schema enforcement.
    /// Disable with `LUMEN_GEMMA4_GRAMMAR_LARK=0` (or `false`).
    ///
    /// Rationale for ON-by-default: the grammar adds negligible CPU per
    /// sampled token but eliminates the most common tool-call format
    /// failure mode on the imatrix-AWQ family (channel boundary logit
    /// suppression produces structurally invalid `call:…{…}` bodies
    /// downstream clients reject). For agentic clients (Moltis-style
    /// matchers, OpenAI-spec tool callers) ON is the safer production
    /// default. Cached on first read.
    pub(crate) fn gemma4_grammar_lark_enabled() -> bool {
        static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *CACHED.get_or_init(|| {
            std::env::var("LUMEN_GEMMA4_GRAMMAR_LARK")
                .map(|v| !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false"))
                .unwrap_or(true)
        })
    }

    /// Opt-in: enforce `tool_choice=required`/named with an **Eager** grammar
    /// (active from token 0) instead of the default Lazy grammar (which only
    /// activates after the model self-emits `<|tool_call>` and thus does not
    /// constrain a prefill-forced call). Default OFF: Eager + agentic loops
    /// were observed to drive some quantized builds into an n-gram cycle on
    /// later turns (see `build_grammar_state`), and that interaction is not
    /// yet re-verified — so strict enforcement stays operator-gated until
    /// A/B'd on the target model. Does not affect `tool_choice=auto`.
    fn gemma4_tool_grammar_eager() -> bool {
        static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *CACHED.get_or_init(|| {
            std::env::var("LUMEN_GEMMA4_TOOL_GRAMMAR_EAGER")
                .map(|v| !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false"))
                .unwrap_or(false)
        })
    }

    /// True for model ids the operator has flagged as instruction-tuned
    /// via imatrix-AWQ calibration. These builds produce degenerate
    /// tool-call outputs (channel logits suppressed by quantization) and
    /// should sample freely regardless of `tool_choice`. Mirrors the
    /// server-side check in `lumen-server::types::is_imatrix_awq_family`
    /// so backend + server agree on the gate.
    fn is_imatrix_awq_family(model_id: &str) -> bool {
        let lower = model_id.to_ascii_lowercase();
        lower.contains("imatrix") || lower.contains("-awq")
    }

    /// Read the `LUMEN_USE_JINJA_RENDERER` env once at backend creation.
    /// Truthy values: `1`, `true`, `on`, `yes` (case-insensitive).
    fn env_jinja_renderer_on() -> bool {
        match std::env::var("LUMEN_USE_JINJA_RENDERER") {
            Ok(v) => matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "on" | "yes"
            ),
            Err(_) => false,
        }
    }

    /// Convert the engine's `(role, content)` pair shape into ChatTurns
    /// so the jinja renderer can consume them. Mirrors the role-string
    /// matching of `Gemma4Backend::parse_role_pairs`.
    fn pairs_to_turns(messages: &[(String, String)]) -> Result<Vec<crate::chat_io::ChatTurn<'_>>> {
        use crate::chat_io::ChatTurn;
        messages
            .iter()
            .map(|(role, content)| {
                let r = role.as_str();
                match r {
                    "system" | "System" | "SYSTEM" => Ok(ChatTurn::System(content.as_str())),
                    "user" | "User" | "USER" => Ok(ChatTurn::User(content.as_str())),
                    "assistant" | "Assistant" | "ASSISTANT" | "model" => Ok(ChatTurn::Assistant {
                        text: content.as_str(),
                        tool_calls: &[],
                    }),
                    other => Err(anyhow!(
                        "Gemma4Backend::pairs_to_turns: unknown role {other:?}"
                    )),
                }
            })
            .collect()
    }

    /// Phase 3: live per-sequence decode state for the batched scheduler.
    /// Holds one independent rotating Gemma 4 cache + its token position. Mirrors
    /// the Qwen runner's `NativeSeqState`; populated at `prefill`, advanced at
    /// `decode_step`/`decode_step_batch`, dropped at `remove_seq`.
    struct NativeGemma4SeqState {
        cache: crate::gemma4_moe::imp::NativeGemma4PromptCache,
        position: usize,
    }

    pub struct Gemma4Backend {
        model: NativeGemma4Model,
        /// Phase 3: per-seq live decode caches keyed by seq_id, for the batched
        /// scheduler. Separate from `prefix_caches` (snapshots, not live decode).
        seqs: HashMap<u64, NativeGemma4SeqState>,
        chat: Gemma4ChatTemplate,
        /// Minijinja-based renderer that evaluates the model's upstream
        /// `chat_template.jinja` directly. Initialized only when
        /// `LUMEN_USE_JINJA_RENDERER` is on at backend-creation time;
        /// otherwise the hand-port in `chat` is the sole render path.
        /// Default OFF in v0.7.0 — opt-in production validation; the
        /// hand-port and jinja paths are byte-identical for the cases
        /// covered by `jinja_chat::imp::tests` (parity vs
        /// `transformers.AutoTokenizer.apply_chat_template` golden
        /// vectors), so flipping this on should be safe for most
        /// agentic workloads.
        jinja_chat: Option<JinjaChatTemplate>,
        model_id: String,
        /// Per-key prefix caches. Keyed by caller-supplied string (e.g. the
        /// system message hash from the Moltis side, or a batch id).
        prefix_caches: HashMap<String, Gemma4PrefixCacheEntry>,
        /// Directory the model was loaded from (holds `tokenizer.json`).
        /// Retained so the llguidance `ParserFactory` can be lazily built
        /// the first time a tools-bearing request arrives.
        model_dir: PathBuf,
        /// Cached llguidance factory keyed by this backend's tokenizer.
        /// Built once per backend instance (~10–50 ms slicer init); per-
        /// request `create_parser` calls are cheap. Stays `Err` if the
        /// tokenizer.json failed to parse so subsequent tool-bearing
        /// requests degrade gracefully (sample without grammar) rather
        /// than fail outright.
        grammar_factory: OnceLock<Option<Arc<ParserFactory>>>,
        /// Phase B (v0.6.0) — cached per-critical-token logit correction
        /// table loaded from `<model_dir>/logit_corrections.bin`. Lazy-
        /// initialized on first decode step; stays `None` when the
        /// sidecar is missing or the env gate is off.
        correction_table: OnceLock<Option<Arc<CorrectionTable>>>,
        /// L2 disk persistence tier (opt-in via `LUMEN_KV_DISK`). `Some` once a
        /// model is loaded with the disk tier enabled — boundary prefix
        /// snapshots are serialized here so they survive process restart /
        /// in-memory eviction. `None` keeps behaviour byte-identical to the
        /// in-memory-only path. Fingerprint = sanitized `model_id`.
        disk: Option<DiskKvStore>,
    }

    /// Open the opt-in L2 disk persistence tier from env (shared `LUMEN_KV_DISK`
    /// gate + `LUMEN_KV_DISK_MAX_GB` budget), namespaced by `model_id`. Returns
    /// `None` when off or on open failure (never fatal).
    fn open_kv_disk(model_id: &str) -> Option<DiskKvStore> {
        let enabled = matches!(
            std::env::var("LUMEN_KV_DISK")
                .ok()
                .map(|s| s.trim().to_ascii_lowercase())
                .as_deref(),
            Some("1" | "true" | "yes" | "on")
        );
        if !enabled {
            return None;
        }
        let fp: String = model_id
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
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
        match DiskKvStore::open(DiskKvStore::default_root(), &fp, max_bytes, ttl_secs) {
            Ok(s) => {
                eprintln!(
                    "[gemma4] kv-disk tier ON: fingerprint={fp:?} entries={} max_gb={}",
                    s.len(),
                    max_bytes / (1024 * 1024 * 1024),
                );
                Some(s)
            }
            Err(e) => {
                eprintln!("[gemma4] kv-disk tier disabled (open failed): {e:#}");
                None
            }
        }
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
            let jinja_chat = if env_jinja_renderer_on() {
                match JinjaChatTemplate::from_dir(dir) {
                    Ok(j) => {
                        eprintln!(
                            "[gemma4] minijinja renderer ACTIVE (LUMEN_USE_JINJA_RENDERER=1)"
                        );
                        Some(j)
                    }
                    Err(e) => {
                        eprintln!(
                            "[gemma4] jinja renderer init failed ({e:?}); falling back to hand-port"
                        );
                        None
                    }
                }
            } else {
                None
            };
            let model_id: String = model_id.into();
            let disk = open_kv_disk(&model_id);
            Ok(Self {
                model,
                seqs: HashMap::new(),
                chat,
                jinja_chat,
                model_id,
                prefix_caches: HashMap::new(),
                model_dir: dir.to_path_buf(),
                grammar_factory: OnceLock::new(),
                correction_table: OnceLock::new(),
                disk,
            })
        }

        /// Lazily load the per-critical-token logit correction sidecar from
        /// `<model_dir>/logit_corrections.bin`. Returns `None` (graceful
        /// degrade) when:
        ///   - `LUMEN_GEMMA4_CRITICAL_LOGIT_CORRECTION=0` (default), OR
        ///   - the sidecar file is absent, OR
        ///   - the sidecar fails to parse (logged once at first call).
        ///
        /// Also flips the model's correction-capture flag on the first
        /// successful load so the very next `forward_array_*` stashes
        /// `h_for_lm_head` for the correction step.
        fn correction_table(&self) -> Option<Arc<CorrectionTable>> {
            if !gemma4_critical_logit_correction_enabled() {
                return None;
            }
            let cached = self
                .correction_table
                .get_or_init(
                    || match CorrectionTable::load_from_model_dir(&self.model_dir) {
                        Ok(Some(t)) => {
                            eprintln!(
                                "[gemma4-backend] logit-correction sidecar loaded: \
                                 {} critical ids, hidden={}",
                                t.critical_ids.len(),
                                t.hidden
                            );
                            Some(Arc::new(t))
                        }
                        Ok(None) => {
                            eprintln!(
                                "[gemma4-backend] LUMEN_GEMMA4_CRITICAL_LOGIT_CORRECTION=1 \
                                 but `{}/logit_corrections.bin` missing — running uncorrected",
                                self.model_dir.display()
                            );
                            None
                        }
                        Err(e) => {
                            eprintln!(
                                "[gemma4-backend] logit-correction sidecar failed to load \
                                 (decoding uncorrected): {e:#}"
                            );
                            None
                        }
                    },
                )
                .clone();
            if cached.is_some() {
                self.model.set_correction_capture_enabled(true);
            }
            cached
        }

        /// Phase B helper — pull the model's captured `h_for_lm_head` to a
        /// CPU f32 buffer for the logit-correction kernel. Returns `None`
        /// when nothing was captured (capture disabled, already consumed,
        /// or forward not yet called). Failures during the MLX→CPU eval
        /// are logged and swallowed so a sidecar bug never breaks decode.
        fn take_captured_correction_h_as_f32(&self) -> Option<Vec<f32>> {
            let h = self.model.take_captured_correction_h()?;
            let h_f32 = match h.as_dtype(mlx_rs::Dtype::Float32) {
                Ok(a) => a,
                Err(e) => {
                    eprintln!("[gemma4-backend] correction h dtype cast failed: {e}");
                    return None;
                }
            };
            if let Err(e) = h_f32.eval() {
                eprintln!("[gemma4-backend] correction h eval failed: {e}");
                return None;
            }
            // h shape is [1, 1, hidden]. as_slice() flattens; that's
            // exactly what the correction kernel expects.
            Some(h_f32.as_slice::<f32>().to_vec())
        }

        /// Lazily build (or return the cached) llguidance `ParserFactory`
        /// bound to this backend's `tokenizer.json`. Returns `None` when
        /// either:
        ///   - tokenizer.json parsing failed (logged once at first call);
        ///   - the backend's `model_id` belongs to the imatrix-AWQ family,
        ///     which the operator runs with tool-calling disabled.
        ///
        /// Callers that get `None` should skip grammar wiring and sample
        /// freely; that's the graceful-degrade path for misconfiguration.
        fn grammar_factory(&self) -> Option<Arc<ParserFactory>> {
            if is_imatrix_awq_family(&self.model_id) {
                return None;
            }
            self.grammar_factory
                .get_or_init(|| {
                    let path = self.model_dir.join("tokenizer.json");
                    match shared_factory_from_tokenizer(&path) {
                        Ok(f) => Some(f),
                        Err(e) => {
                            eprintln!(
                                "[gemma4-backend] grammar factory unavailable \
                                 (tools will sample without grammar mask): {e:#}"
                            );
                            None
                        }
                    }
                })
                .clone()
        }

        /// Build a per-request grammar state from the parsed tool defs +
        /// resolved tool_choice. Returns `None` unless the operator opts
        /// in via `LUMEN_GEMMA4_GRAMMAR_LARK=1` AND the request actually
        /// expresses a tool intent.
        ///
        /// Background: the JSON Schema path produces JSON output
        /// (`{"name":"X","arguments":{...}}`) which Gemma 4's response
        /// parser ([`crate::gemma4_response::imp::parse_tool_call_body`])
        /// can't read — it expects the native pseudo-JSON format
        /// `call:NAME{key:value,…}` emitted from training. The Lark
        /// generator in [`crate::grammar::build_tool_grammar_lark`] (gated
        /// via `Gemma4GrammarState::new_lark`) emits the right format.
        ///
        /// Default OFF preserves the proven native-only path (no schema
        /// validation, model emits canonical format from training). Opt
        /// in when:
        ///   - operating on models that drift off-format under quant;
        ///   - serving clients that demand strict schema enforcement;
        ///   - debugging grammar issues against the curl smoke fixture.
        ///
        /// `imatrix-AWQ` family is still skipped via
        /// [`grammar_factory`] regardless of env (those builds force
        /// thinking off and don't exercise tool calling).
        fn build_grammar_state(
            &self,
            tools: &[crate::chat_io::ToolDef<'_>],
            choice: &crate::chat_io::ResolvedToolChoice<'_>,
        ) -> Option<Gemma4GrammarState> {
            use crate::chat_io::ResolvedToolChoice;
            if !gemma4_grammar_lark_enabled() {
                return None;
            }
            // env is ON — any further skip is unexpected during agentic flows,
            // so log the reason so the operator can diagnose.
            if tools.is_empty() {
                eprintln!("[gemma4-backend] Lark grammar skipped: tools empty");
                return None;
            }
            // Mode + grammar shape per `tool_choice` (WS-C #1 fix):
            //
            //   - `Auto` → **Lazy + permissive** Lark over all tools. The
            //     proven default: the matcher activates only when the model
            //     self-emits `<|tool_call>` (id 48), then schema-constrains
            //     the body. Byte-identical to the pre-fix path. The
            //     `LUMEN_GEMMA4_TOOL_GRAMMAR_EAGER` opt-in additionally flips
            //     `Auto` to Eager+strict for operators who want every turn
            //     forced.
            //
            //   - `Required` / `Tool(name)` → **Eager + STRICT (dup-free)**
            //     Lark, UNCONDITIONALLY. This is the actual bug: required/named
            //     prefill the `<|tool_call>` opener via `parser.push()` (never
            //     sampled), so a *Lazy* matcher never activates and the args
            //     body is generated UNCONSTRAINED — which lets quantized
            //     35B/26B builds emit `call:read{}` with empty params and spin
            //     in an 845+-message loop. The Eager path was previously gated
            //     off because the *permissive* Lark body
            //     (`field ("," field)*`) over-permits duplicate fields and
            //     drove an n-gram cycle. The strict body
            //     ([`build_tool_grammar_lark_strict`]) emits required fields
            //     once each in fixed order then optional fields at most once —
            //     no Kleene-star over fields, so there is no duplicate-field
            //     cycle. This is the native-format analogue of SGLang's JSON
            //     `required` + `minItems:1` enforcement. The prefilled opener
            //     is replayed into the Eager matcher via
            //     [`Gemma4GrammarState::observe_prefill`] in
            //     `decode_streaming_with_prompt` so the matcher's parse
            //     position matches the model's context.
            //
            // `tool_choice=None` => no grammar (model must not call a tool).
            let eager_auto = gemma4_tool_grammar_eager();
            let (mode, strict, name_filter): (GrammarMode, bool, Option<&str>) = match choice {
                ResolvedToolChoice::Auto => {
                    if eager_auto {
                        (GrammarMode::Eager, true, None)
                    } else {
                        (GrammarMode::Lazy, false, None)
                    }
                }
                ResolvedToolChoice::Required => (GrammarMode::Eager, true, None),
                ResolvedToolChoice::Tool(name) => (GrammarMode::Eager, true, Some(*name)),
                ResolvedToolChoice::None => {
                    eprintln!("[gemma4-backend] Lark grammar skipped: tool_choice=None");
                    return None;
                }
            };
            let Some(factory) = self.grammar_factory() else {
                eprintln!(
                    "[gemma4-backend] Lark grammar skipped: factory unavailable \
                     (imatrix-AWQ family or tokenizer.json load error)"
                );
                return None;
            };
            let tools_json: Vec<serde_json::Value> = tools
                .iter()
                .filter(|t| name_filter.is_none_or(|n| t.name == n))
                .map(|t| {
                    let mut function = serde_json::json!({ "name": t.name });
                    if let Some(d) = t.description {
                        function["description"] = serde_json::Value::String(d.to_string());
                    }
                    if let Some(p) = t.parameters {
                        function["parameters"] = p.clone();
                    }
                    serde_json::json!({ "type": "function", "function": function })
                })
                .collect();
            if tools_json.is_empty() {
                eprintln!(
                    "[gemma4-backend] Lark grammar skipped: tool_choice names an unknown tool"
                );
                return None;
            }
            let built = if strict {
                Gemma4GrammarState::new_lark_strict(factory, &tools_json, mode)
            } else {
                Gemma4GrammarState::new_lark(factory, &tools_json, mode)
            };
            match built {
                Ok(s) => {
                    eprintln!(
                        "[gemma4-backend] Lark grammar active for {} tool(s) (mode={mode:?}, strict={strict})",
                        tools.len()
                    );
                    Some(s)
                }
                Err(e) => {
                    eprintln!(
                        "[gemma4-backend] Lark grammar state build failed \
                         (falling back to free sampling): {e:#}"
                    );
                    None
                }
            }
        }

        /// Build a per-request grammar state that constrains the **visible**
        /// output to valid JSON matching `schema` (OpenAI `response_format`
        /// → `json_object` / `json_schema`). Returns `None` — sampling
        /// proceeds unconstrained — when the grammar factory is unavailable
        /// (imatrix-AWQ family or a tokenizer.json load error), mirroring
        /// [`Gemma4Backend::build_grammar_state`]'s factory check.
        ///
        /// Always **Eager** ([`GrammarMode::Eager`]): there is no lazy
        /// trigger token for free-form JSON, so the constraint must be live
        /// from token 0. Unlike the tool grammar, this is independent of
        /// `tool_choice` / `LUMEN_GEMMA4_GRAMMAR_LARK` — it activates
        /// solely from the presence of `response_format` in the request.
        fn build_response_format_grammar(
            &self,
            schema: &serde_json::Value,
        ) -> Option<Gemma4GrammarState> {
            let Some(factory) = self.grammar_factory() else {
                eprintln!(
                    "[gemma4-backend] response_format grammar skipped: factory \
                     unavailable (imatrix-AWQ family or tokenizer.json load error)"
                );
                return None;
            };
            match Gemma4GrammarState::new_json_schema(factory, schema, GrammarMode::Eager) {
                Ok(s) => {
                    eprintln!(
                        "[gemma4-backend] response_format grammar active (Eager JSON schema)"
                    );
                    Some(s)
                }
                Err(e) => {
                    eprintln!(
                        "[gemma4-backend] response_format grammar build failed \
                         (falling back to free sampling): {e:#}"
                    );
                    None
                }
            }
        }

        /// Select the per-request grammar: when `response_schema` is
        /// `Some`, build the response-format grammar and **skip the tool
        /// grammar** (response_format + tools together is unusual; OpenAI
        /// semantics give the response_format precedence). Otherwise fall
        /// back to the existing tool-grammar path. Centralizes the
        /// precedence so every streaming entrypoint behaves identically.
        ///
        /// Absent `response_schema` ⇒ byte-for-byte the prior
        /// `build_grammar_state(tools, choice)` behavior.
        fn select_grammar_state(
            &self,
            tools: &[crate::chat_io::ToolDef<'_>],
            choice: &crate::chat_io::ResolvedToolChoice<'_>,
            response_schema: Option<&serde_json::Value>,
        ) -> Option<Gemma4GrammarState> {
            if let Some(schema) = response_schema {
                if !tools.is_empty() {
                    eprintln!(
                        "[gemma4-backend] response_format present with tools — \
                         preferring response_format grammar, skipping tool grammar"
                    );
                }
                return self.build_response_format_grammar(schema);
            }
            self.build_grammar_state(tools, choice)
        }

        pub fn model_id(&self) -> &str {
            &self.model_id
        }

        pub fn model(&self) -> &NativeGemma4Model {
            &self.model
        }

        /// Effective max context (already capped by `LUMEN_MAX_CTX` at config
        /// load, see `gemma4_moe.rs`). A prompt can never exceed this — the KV
        /// cache physically can't hold more, and on MLX an over-max_ctx prefill
        /// OOM-aborts the whole process. The engine uses this as a hard prompt
        /// reject ceiling.
        pub fn max_context(&self) -> usize {
            self.model.config().text_config.max_position_embeddings
        }

        /// Phase 3: EOS token id set for the loaded Gemma 4 model.
        pub fn eos_tokens(&self) -> &[u32] {
            self.model.eos_tokens()
        }

        /// Phase 3: prefill `tokens` as a fresh live decode sequence `seq_id`,
        /// returning `(first_generated_token, position)`. Mirrors the MLX prefill
        /// convention (full prompt consumed; returns the first generated token).
        pub fn prefill(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)> {
            let mut cache = self.make_cache_for_prompt_len(tokens.len());
            let logits = self
                .model
                .forward_last_token(tokens, &mut cache)
                .with_context(|| format!("gemma4 prefill seq {seq_id}"))?;
            let first_tok = self.model.argmax_last_token(&logits)?;
            self.seqs.insert(
                seq_id,
                NativeGemma4SeqState {
                    cache,
                    position: tokens.len(),
                },
            );
            Ok((first_tok, tokens.len()))
        }

        /// Phase 3: advance one live sequence by a single token (greedy).
        pub fn decode_step(
            &mut self,
            seq_id: u64,
            last_token: u32,
            _position: usize,
        ) -> Result<(u32, usize)> {
            let model = &self.model;
            let state = self
                .seqs
                .get_mut(&seq_id)
                .ok_or_else(|| anyhow!("gemma4 decode_step: unknown seq_id {seq_id}"))?;
            let logits = model
                .forward_last_token(&[last_token], &mut state.cache)
                .with_context(|| format!("gemma4 decode_step seq {seq_id}"))?;
            let tok = model.argmax_last_token(&logits)?;
            state.position += 1;
            Ok((tok, state.position))
        }

        /// Phase 3: batched decode of N live sequences in ONE forward. N==1
        /// delegates to `decode_step` (byte-identical). N>=2 borrows the N per-seq
        /// caches disjointly out of `self.seqs` and runs the model's
        /// `forward_decode_batch` (batched trunk/MoE/lm_head + per-seq sliding /
        /// global attention, each rotating its own cache independently). Result is
        /// aligned to `seq_ids` order; each seq's position is bumped by one.
        pub fn decode_step_batch(
            &mut self,
            seq_ids: &[u64],
            last_tokens: &[u32],
            positions: &[usize],
        ) -> Result<Vec<(u32, usize)>> {
            let n = seq_ids.len();
            if last_tokens.len() != n || positions.len() != n {
                return Err(anyhow!(
                    "gemma4 decode_step_batch: mismatched lengths (seqs={n}, tokens={}, positions={})",
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
            let id_set: std::collections::HashSet<u64> = seq_ids.iter().copied().collect();
            if id_set.len() != n {
                return Err(anyhow!(
                    "gemma4 decode_step_batch: duplicate seq_id in batch"
                ));
            }
            let model = &self.model;
            let mut by_id: HashMap<u64, &mut NativeGemma4SeqState> = self
                .seqs
                .iter_mut()
                .filter(|(k, _)| id_set.contains(k))
                .map(|(k, v)| (*k, v))
                .collect();
            if by_id.len() != n {
                return Err(anyhow!(
                    "gemma4 decode_step_batch: {} of {n} seq_ids found",
                    by_id.len()
                ));
            }
            let mut states: Vec<&mut NativeGemma4SeqState> = Vec::with_capacity(n);
            for sid in seq_ids {
                states.push(by_id.remove(sid).expect("checked present above"));
            }
            let next = {
                let mut caches: Vec<&mut crate::gemma4_moe::imp::NativeGemma4PromptCache> =
                    states.iter_mut().map(|s| &mut s.cache).collect();
                model
                    .forward_decode_batch(last_tokens, &mut caches)
                    .context("gemma4 decode_step_batch: forward_decode_batch")?
            };
            if next.len() != n {
                return Err(anyhow!(
                    "gemma4 decode_step_batch: forward returned {} tokens for {n} seqs",
                    next.len()
                ));
            }
            let mut out = Vec::with_capacity(n);
            for (i, s) in states.iter_mut().enumerate() {
                s.position += 1;
                out.push((next[i], s.position));
            }
            Ok(out)
        }

        /// Phase 3: drop a live decode sequence.
        pub fn remove_seq(&mut self, seq_id: u64) -> Result<()> {
            self.seqs.remove(&seq_id);
            Ok(())
        }

        /// Allocate a fresh cache with the per-request adaptive decision
        /// for both TQ and simple Q4. `prompt_len` is this request's
        /// prompt-token count; auto-mode env vars compare it against
        /// their respective thresholds. `On` / `Off` modes ignore the
        /// length (binary decision). Use this everywhere a cache is
        /// built without prefix-cache reuse so the adaptive mode actually
        /// fires on long-context requests.
        fn make_cache_for_prompt_len(
            &self,
            prompt_len: usize,
        ) -> crate::gemma4_moe::imp::NativeGemma4PromptCache {
            use crate::gemma4_moe::imp::{
                Gemma4QuantKvMode, Gemma4TqMode, gemma4_quant_kv_auto_threshold,
                gemma4_quant_kv_mode, gemma4_tq_auto_threshold, gemma4_tq_mode,
                resolve_quant_kv_for_request, resolve_tq_for_request,
            };
            let force_tq = resolve_tq_for_request(prompt_len);
            let force_quant_kv = resolve_quant_kv_for_request(prompt_len);
            if gemma4_tq_mode() == Gemma4TqMode::Auto {
                eprintln!(
                    "[gemma4-backend] tq_auto: prompt_tokens={prompt_len} \
                     threshold={} → tq={}",
                    gemma4_tq_auto_threshold(),
                    if force_tq { "ON" } else { "OFF" }
                );
            }
            if gemma4_quant_kv_mode() == Gemma4QuantKvMode::Auto {
                eprintln!(
                    "[gemma4-backend] quant_kv_auto: prompt_tokens={prompt_len} \
                     threshold={} → q4={}",
                    gemma4_quant_kv_auto_threshold(),
                    if force_quant_kv { "ON" } else { "OFF" }
                );
            }
            self.model
                .make_cache_with_tq(Some(force_tq), Some(force_quant_kv))
        }

        /// One-line runtime config summary for startup logging. Captures
        /// the effective values AFTER all env overrides (LUMEN_MAX_CTX,
        /// LUMEN_SLIDING_WINDOW, LUMEN_GEMMA4_TOP_K, etc.) have been
        /// applied — what the model actually runs with, not what the on-
        /// disk config.json claims.
        pub fn runtime_config_summary(&self) -> String {
            // TurboQuant infra remains in-tree (env-gated default OFF) for
            // future CUDA / PagedAttention work — see CLAUDE.md Phase 2/3.
            // It is not surfaced here because empirical sweeps showed TQ is
            // net-negative on Apple Silicon batch=1; users should not need
            // to reason about it. Q4 simple quantization is the supported
            // memory-saving lever and is exposed below.
            let cfg = &self.model.config().text_config;
            let q4_mode = match crate::gemma4_moe::imp::gemma4_quant_kv_mode() {
                crate::gemma4_moe::imp::Gemma4QuantKvMode::Off => "off",
                crate::gemma4_moe::imp::Gemma4QuantKvMode::On => "on",
                crate::gemma4_moe::imp::Gemma4QuantKvMode::Auto => "auto",
            };
            let q4_threshold = crate::gemma4_moe::imp::gemma4_quant_kv_auto_threshold();
            let q4_bits = crate::gemma4_moe::imp::gemma4_quant_kv_bits();
            format!(
                "max_ctx={} sliding_window={} top_k_experts={}/{} layers={} vocab={} mtp={} \
                 quant_kv={q4_mode} quant_kv_threshold={q4_threshold} quant_kv_bits={q4_bits}",
                cfg.max_position_embeddings,
                cfg.sliding_window,
                cfg.top_k_experts,
                cfg.num_experts,
                cfg.num_hidden_layers,
                cfg.vocab_size,
                if self.model.mtp_enabled() {
                    "loaded"
                } else {
                    "off"
                },
            )
        }

        /// Load the Gemma 4 assistant drafter from `drafter_dir` and enable
        /// MTP speculative decoding. Returns `Ok(true)` when the drafter
        /// passes the trunk/drafter `backbone_hidden_size` compatibility
        /// check and is wired in. After this call, requests routed through
        /// `chat_streaming` (greedy) will use `mtp_step()` whenever
        /// `LUMEN_GEMMA4_MTP=1` is set.
        pub fn try_enable_mtp(&mut self, drafter_dir: &std::path::Path) -> Result<bool> {
            self.model.try_enable_mtp(drafter_dir)
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
            self.build_chat_input_with_tools(messages, thinking, &[], false)
        }

        /// Prompt tokens one attached image adds — its soft-token run plus the
        /// sentinels and trailing newline. Header-only; no pixel decode.
        pub fn image_prompt_tokens(&self, encoded: &[u8]) -> Result<usize> {
            self.model.image_prompt_tokens(encoded)
        }

        /// Structured-history variant — accepts `ChatTurn`s carrying
        /// `tool_calls` / tool result data the `(role, content)` shape
        /// can't represent. Used by the turn-2 continuation path
        /// (`chat_with_history`).
        /// [`Self::build_chat_input_from_history`] with per-turn image
        /// soft-token counts.
        ///
        /// Rejects the jinja path for the same reason the flat renderer does:
        /// the template emits a single `<|image|>` and expanding it into the
        /// per-image run is the processor's job, which is not reimplemented
        /// inside the template engine.
        pub fn build_chat_input_from_history_with_images(
            &self,
            turns: &[crate::chat_io::ChatTurn<'_>],
            thinking: bool,
            tools: &[crate::chat_io::ToolDef<'_>],
            image_counts: &[Vec<usize>],
            close_thought_channel: bool,
        ) -> Result<Vec<u32>> {
            if image_counts.iter().all(|c| c.is_empty()) {
                return self.build_chat_input_from_history(
                    turns,
                    thinking,
                    tools,
                    close_thought_channel,
                );
            }
            if self.jinja_chat.is_some() {
                return Err(anyhow!(
                    "image input is not supported with a jinja chat template; \
                     unset the template to use the built-in Gemma 4 renderer"
                ));
            }
            let ids = self.chat.render_chat_history_with_images(
                turns,
                &RenderOptions {
                    enable_thinking: thinking,
                    add_generation_prompt: true,
                    close_thought_channel,
                },
                tools,
                image_counts,
            )?;
            maybe_dump_prompt(&self.chat, &ids, "from_history_with_images");
            Ok(ids)
        }

        pub fn build_chat_input_from_history(
            &self,
            turns: &[crate::chat_io::ChatTurn<'_>],
            thinking: bool,
            tools: &[crate::chat_io::ToolDef<'_>],
            close_thought_channel: bool,
        ) -> Result<Vec<u32>> {
            let ids = if let Some(j) = &self.jinja_chat {
                j.render_to_ids(
                    turns,
                    &JinjaRenderOptions {
                        enable_thinking: thinking,
                        add_generation_prompt: true,
                    },
                    if tools.is_empty() { None } else { Some(tools) },
                )?
            } else {
                self.chat.render_chat_history(
                    turns,
                    &RenderOptions {
                        enable_thinking: thinking,
                        add_generation_prompt: true,
                        close_thought_channel,
                    },
                    tools,
                )?
            };
            maybe_dump_prompt(&self.chat, &ids, "from_history");
            Ok(ids)
        }

        fn build_chat_input_from_history_no_gen(
            &self,
            turns: &[crate::chat_io::ChatTurn<'_>],
            thinking: bool,
            tools: &[crate::chat_io::ToolDef<'_>],
        ) -> Result<Vec<u32>> {
            let ids = if let Some(j) = &self.jinja_chat {
                j.render_to_ids(
                    turns,
                    &JinjaRenderOptions {
                        enable_thinking: thinking,
                        add_generation_prompt: false,
                    },
                    if tools.is_empty() { None } else { Some(tools) },
                )?
            } else {
                self.chat.render_chat_history(
                    turns,
                    &RenderOptions {
                        enable_thinking: thinking,
                        add_generation_prompt: false,
                        // No generation prompt is emitted, so the thought
                        // channel has nowhere to be prefilled.
                        close_thought_channel: false,
                    },
                    tools,
                )?
            };
            maybe_dump_prompt(&self.chat, &ids, "from_history_no_gen");
            Ok(ids)
        }

        /// Tool-aware variant of `build_chat_input`. Empty `tools` slice
        /// produces the exact same token sequence as `build_chat_input`;
        /// otherwise tool definitions get injected into the system turn
        /// per the canonical `chat_template.jinja`.
        pub fn build_chat_input_with_tools(
            &self,
            messages: &[(String, String)],
            thinking: bool,
            tools: &[crate::gemma4_tools::imp::ToolDef<'_>],
            close_thought_channel: bool,
        ) -> Result<Vec<u32>> {
            let ids = if let Some(j) = &self.jinja_chat {
                let turns = pairs_to_turns(messages)?;
                j.render_to_ids(
                    &turns,
                    &JinjaRenderOptions {
                        enable_thinking: thinking,
                        add_generation_prompt: true,
                    },
                    if tools.is_empty() { None } else { Some(tools) },
                )?
            } else {
                let parsed = Self::parse_role_pairs(messages)?;
                self.chat.render_to_ids_with_tools(
                    &parsed,
                    &RenderOptions {
                        enable_thinking: thinking,
                        add_generation_prompt: true,
                        close_thought_channel,
                    },
                    tools,
                )?
            };
            maybe_dump_prompt(&self.chat, &ids, "with_tools");
            Ok(ids)
        }

        /// Like `build_chat_input_with_tools` but without the trailing
        /// `<start_of_turn>model\n` generation prompt (and the empty thought
        /// channel that gets appended when `thinking=false`). Used by
        /// prefix-cache callers to compute the trailing header token count
        /// so the cache snapshot can stop just before it — the trailing
        /// header is the only part that diverges between turn N (where it
        /// sits at the prompt tail) and turn N+1 (where it sits in the
        /// middle, followed by the actual assistant response).
        fn build_chat_input_no_gen(
            &self,
            messages: &[(String, String)],
            thinking: bool,
            tools: &[crate::gemma4_tools::imp::ToolDef<'_>],
        ) -> Result<Vec<u32>> {
            if let Some(j) = &self.jinja_chat {
                let turns = pairs_to_turns(messages)?;
                j.render_to_ids(
                    &turns,
                    &JinjaRenderOptions {
                        enable_thinking: thinking,
                        add_generation_prompt: false,
                    },
                    if tools.is_empty() { None } else { Some(tools) },
                )
            } else {
                let parsed = Self::parse_role_pairs(messages)?;
                self.chat.render_to_ids_with_tools(
                    &parsed,
                    &RenderOptions {
                        enable_thinking: thinking,
                        add_generation_prompt: false,
                        // No generation prompt is emitted, so the thought
                        // channel has nowhere to be prefilled.
                        close_thought_channel: false,
                    },
                    tools,
                )
            }
        }

        fn parse_role_pairs(messages: &[(String, String)]) -> Result<Vec<ChatMessage<'_>>> {
            messages
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
                .collect()
        }

        /// `/v1/completions` path. When `temperature > 0` (or
        /// `REPEAT_PENALTY` env is set), routes through CPU sampling;
        /// otherwise the existing GPU-pipelined greedy path runs.
        pub fn generate(
            &mut self,
            input_ids: &[u32],
            max_new_tokens: usize,
            temperature: f32,
            top_p: f32,
            ov: &crate::SamplingOverrides,
        ) -> Result<Vec<u32>> {
            let cfg = GenerateConfig {
                max_new_tokens,
                stop_on_eos: true,
                sampling: build_sampling_config(temperature, top_p, ov),
            };
            let stats = self.model.generate(input_ids, &cfg)?;
            eprintln!(
                "[gemma4] completion done: {} tokens in {:.0}ms ({:.1} tok/s)",
                stats.decode_steps, stats.decode_ms, stats.decode_tok_per_sec
            );
            Ok(stats.generated_tokens)
        }

        /// `/v1/chat/completions` path: render → generate → parse.
        ///
        /// Returns the parsed response (visible text, reasoning, tool calls)
        /// so the HTTP layer can ship structured fields per the OpenAI spec.
        /// Structured-history variant of `chat`. Used when the request
        /// includes assistant messages with `tool_calls` or `role:"tool"`
        /// messages (i.e. turn-2+ of an agent loop). Renders the Gemma 4
        /// model turn with interleaved tool_call / tool_response blocks
        /// then runs the same generate + parse path as `chat`.
        pub fn chat_from_history(
            &mut self,
            turns: &[crate::chat_io::ChatTurn<'_>],
            max_new_tokens: usize,
            temperature: f32,
            top_p: f32,
            ov: &crate::SamplingOverrides,
            thinking: bool,
            tools: &[crate::chat_io::ToolDef<'_>],
            tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
        ) -> Result<ParsedResponse> {
            self.chat_from_history_with_images(
                turns,
                &[],
                max_new_tokens,
                temperature,
                top_p,
                ov,
                thinking,
                tools,
                tool_choice,
            )
        }

        /// [`Self::chat_from_history`] with images attached to `User` turns.
        ///
        /// `images[i]` holds the encoded byte streams on `turns[i]` — indexed
        /// by **turn**, since the caller expands one request message into
        /// several turns. Empty is byte-identical to [`Self::chat_from_history`].
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
            tools: &[crate::chat_io::ToolDef<'_>],
            tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
        ) -> Result<ParsedResponse> {
            let (counts, prepared) = self.measure_turn_images(images)?;
            let (prompt, prefill_tokens) = self.build_prompt_and_prefill_from_history_with_images(
                turns,
                &counts,
                thinking,
                tools,
                tool_choice,
                false,
            )?;
            let cfg = GenerateConfig {
                max_new_tokens,
                stop_on_eos: true,
                sampling: build_sampling_config(temperature, top_p, ov),
            };
            let stats = if prepared.is_empty() {
                self.model.generate(&prompt, &cfg)?
            } else {
                self.model
                    .generate_with_cache_and_images(&prompt, &prepared, &cfg, None)?
            };
            eprintln!(
                "[gemma4] chat_from_history done: {} tokens in {:.0}ms ({:.1} tok/s)",
                stats.decode_steps, stats.decode_ms, stats.decode_tok_per_sec
            );

            let mut parser = ResponseParser::new(&self.chat);
            for tok in &prefill_tokens {
                parser.push(*tok)?;
            }
            for token in &stats.generated_tokens {
                parser.push(*token)?;
            }
            parser.finalize()
        }

        pub fn chat(
            &mut self,
            messages: &[(String, String)],
            max_new_tokens: usize,
            temperature: f32,
            top_p: f32,
            ov: &crate::SamplingOverrides,
            thinking: bool,
            tools: &[crate::gemma4_tools::imp::ToolDef<'_>],
            tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
        ) -> Result<ParsedResponse> {
            self.chat_with_images(
                messages,
                &[],
                max_new_tokens,
                temperature,
                top_p,
                ov,
                thinking,
                tools,
                tool_choice,
            )
        }

        /// Render an image-bearing prompt and return it alongside the images in
        /// prompt order.
        ///
        /// Every image is measured first — the prompt must reserve exactly as
        /// many `<|image|>` placeholders as the tower will emit soft tokens,
        /// and that count depends on the image's aspect ratio. Measuring runs
        /// decode + resize only, not the tower.
        ///
        /// The returned `PreparedImage`s are flattened in prompt order, matching
        /// the placeholder runs the renderer just emitted, which is the pairing
        /// `encode_images_for_prompt` relies on. They are carried rather than
        /// re-derived so each image is decoded and resized exactly once.
        #[allow(clippy::too_many_arguments)]
        fn build_image_prompt(
            &self,
            messages: &[(String, String)],
            images: &[Vec<Vec<u8>>],
            thinking: bool,
            tools: &[crate::gemma4_tools::imp::ToolDef<'_>],
            tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
            close_thought_channel: bool,
        ) -> Result<(Vec<u32>, Vec<u32>, Vec<crate::gemma4_vision::PreparedImage>)> {
            let mut counts: Vec<Vec<usize>> = Vec::with_capacity(images.len());
            let mut flat: Vec<crate::gemma4_vision::PreparedImage> = Vec::new();
            for per_msg in images {
                let mut row = Vec::with_capacity(per_msg.len());
                for bytes in per_msg {
                    let prepared = self.model.prepare_image(bytes)?;
                    row.push(prepared.num_soft_tokens);
                    flat.push(prepared);
                }
                counts.push(row);
            }
            let (prompt, prefill_tokens) = self.build_prompt_and_prefill_with_images(
                messages,
                &counts,
                thinking,
                tools,
                tool_choice,
                close_thought_channel,
            )?;
            Ok((prompt, prefill_tokens, flat))
        }

        /// [`Self::chat`] with inline images.
        ///
        /// `images[i]` holds the encoded image byte streams attached to
        /// `messages[i]`. An empty slice is byte-identical to [`Self::chat`].
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
            tools: &[crate::gemma4_tools::imp::ToolDef<'_>],
            tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
        ) -> Result<ParsedResponse> {
            let (prompt, prefill_tokens, flat) =
                self.build_image_prompt(messages, images, thinking, tools, tool_choice, false)?;
            let cfg = GenerateConfig {
                max_new_tokens,
                stop_on_eos: true,
                sampling: build_sampling_config(temperature, top_p, ov),
            };
            let stats = if flat.is_empty() {
                self.model.generate(&prompt, &cfg)?
            } else {
                self.model
                    .generate_with_cache_and_images(&prompt, &flat, &cfg, None)?
            };
            log_chat_done(
                stats.prompt_tokens,
                stats.prefill_ms,
                stats.decode_steps,
                stats.decode_ms,
            );

            let mut parser = ResponseParser::new(&self.chat);
            // Phase 1.6: feed tool_choice prefill tokens to the parser
            // first so its state machine matches the prompt the model
            // saw. Required: parser enters ToolCall on token 48.
            // Tool(name): parser enters ToolCall on token 48 and
            // accumulates "call:NAME{" before the model's first
            // generated args token arrives.
            for tok in &prefill_tokens {
                parser.push(*tok)?;
            }
            for token in &stats.generated_tokens {
                parser.push(*token)?;
            }
            parser.finalize()
        }

        /// Phase 1.6: assemble prompt + tool_choice prefill in one call.
        /// `None` choice means strip tool defs entirely (we hand an
        /// empty `tools` slice to the renderer so the system turn never
        /// mentions tools); `Required` / `Tool(name)` append the
        /// `<|tool_call>` (+ `call:NAME{`) prefill to the generation
        /// prompt. Returns the full prompt token vector and the prefill
        /// slice the caller must replay through the response parser.
        fn build_prompt_and_prefill(
            &self,
            messages: &[(String, String)],
            thinking: bool,
            tools: &[crate::gemma4_tools::imp::ToolDef<'_>],
            tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
            close_thought_channel: bool,
        ) -> Result<(Vec<u32>, Vec<u32>)> {
            self.build_prompt_and_prefill_with_images(
                messages,
                &[],
                thinking,
                tools,
                tool_choice,
                close_thought_channel,
            )
        }

        /// [`Self::build_prompt_and_prefill`] with per-message image
        /// soft-token counts. Empty `image_counts` is byte-identical.
        #[allow(clippy::too_many_arguments)]
        fn build_prompt_and_prefill_with_images(
            &self,
            messages: &[(String, String)],
            image_counts: &[Vec<usize>],
            thinking: bool,
            tools: &[crate::gemma4_tools::imp::ToolDef<'_>],
            tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
            close_thought_channel: bool,
        ) -> Result<(Vec<u32>, Vec<u32>)> {
            use crate::chat_io::ResolvedToolChoice;
            let effective_tools: &[crate::gemma4_tools::imp::ToolDef<'_>] = match tool_choice {
                ResolvedToolChoice::None => &[],
                _ => tools,
            };
            let has_images = image_counts.iter().any(|c| !c.is_empty());

            let mut prompt = if !has_images {
                // No images → keep the untouched path, including the
                // model-supplied jinja template when one is present.
                self.build_chat_input_with_tools(
                    messages,
                    thinking,
                    effective_tools,
                    close_thought_channel,
                )?
            } else {
                if self.jinja_chat.is_some() {
                    // The jinja renderer emits a single `<|image|>`; expanding it
                    // to the per-image soft-token run is the processor's job and
                    // we don't reimplement that inside the template engine.
                    return Err(anyhow!(
                        "image input is not supported with a jinja chat template; \
                         unset the template to use the built-in Gemma 4 renderer"
                    ));
                }
                let parsed = Self::parse_role_pairs(messages)?;
                let ids = self.chat.render_to_ids_with_tools_and_images(
                    &parsed,
                    &RenderOptions {
                        enable_thinking: thinking,
                        add_generation_prompt: true,
                        close_thought_channel,
                    },
                    effective_tools,
                    image_counts,
                )?;
                maybe_dump_prompt(&self.chat, &ids, "with_images");
                ids
            };
            let prefill = self.chat.tool_choice_prefill_tokens(tool_choice)?;
            prompt.extend(prefill.iter().copied());
            Ok((prompt, prefill))
        }

        /// History variant of `build_prompt_and_prefill`.
        fn build_prompt_and_prefill_from_history(
            &self,
            turns: &[crate::chat_io::ChatTurn<'_>],
            thinking: bool,
            tools: &[crate::gemma4_tools::imp::ToolDef<'_>],
            tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
            close_thought_channel: bool,
        ) -> Result<(Vec<u32>, Vec<u32>)> {
            self.build_prompt_and_prefill_from_history_with_images(
                turns,
                &[],
                thinking,
                tools,
                tool_choice,
                close_thought_channel,
            )
        }

        /// [`Self::build_prompt_and_prefill_from_history`] with per-turn image
        /// soft-token counts. Empty counts are byte-identical.
        #[allow(clippy::too_many_arguments)]
        fn build_prompt_and_prefill_from_history_with_images(
            &self,
            turns: &[crate::chat_io::ChatTurn<'_>],
            image_counts: &[Vec<usize>],
            thinking: bool,
            tools: &[crate::gemma4_tools::imp::ToolDef<'_>],
            tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
            close_thought_channel: bool,
        ) -> Result<(Vec<u32>, Vec<u32>)> {
            use crate::chat_io::ResolvedToolChoice;
            let effective_tools: &[crate::chat_io::ToolDef<'_>] = match tool_choice {
                ResolvedToolChoice::None => &[],
                _ => tools,
            };
            let mut prompt = self.build_chat_input_from_history_with_images(
                turns,
                thinking,
                effective_tools,
                image_counts,
                close_thought_channel,
            )?;
            let prefill = self.chat.tool_choice_prefill_tokens(tool_choice)?;
            prompt.extend(prefill.iter().copied());
            Ok((prompt, prefill))
        }

        /// Longest shared-prefix token length for **batch fan-out** caching:
        /// the render of every message except the final (per-item, varying)
        /// turn, taken as a strict prefix of `full_prompt`. This is the
        /// boundary to snapshot so sibling requests sharing the leading
        /// messages (e.g. a stable system prompt) fork from it and prefill
        /// only their own suffix.
        ///
        /// Unlike the legacy "snapshot the whole prompt + `truncate_to(lcp)` on
        /// reuse" path, forking from this boundary never rolls the rotating
        /// sliding cache back, so it works even when the shared prefix exceeds
        /// the sliding window (`> 1024` tok) — exactly the production
        /// sports-matching case where every item shares a >1k-token system
        /// prompt. Returns 0 (no caching) when there's no worthwhile shared
        /// prefix (single message, or the head render isn't a clean prefix).
        fn batch_fanout_boundary(
            &self,
            messages: &[(String, String)],
            thinking: bool,
            tools: &[crate::gemma4_tools::imp::ToolDef<'_>],
            full_prompt: &[u32],
        ) -> usize {
            if messages.len() < 2 {
                return 0;
            }
            let head = &messages[..messages.len() - 1];
            let head_ids = match self.build_chat_input_no_gen(head, thinking, tools) {
                Ok(v) => v,
                Err(_) => return 0,
            };
            full_prompt
                .iter()
                .zip(head_ids.iter())
                .take_while(|(a, b)| a == b)
                .count()
                .min(full_prompt.len().saturating_sub(1))
        }

        /// `batch_fanout_boundary` for the structured-`ChatTurn` entry points
        /// (tool-aware follow-ups). Same semantics: render all turns except the
        /// final one as the shared-prefix boundary.
        fn batch_fanout_boundary_from_history(
            &self,
            turns: &[crate::chat_io::ChatTurn<'_>],
            thinking: bool,
            tools: &[crate::chat_io::ToolDef<'_>],
            full_prompt: &[u32],
        ) -> usize {
            if turns.len() < 2 {
                return 0;
            }
            let head = &turns[..turns.len() - 1];
            let head_ids = match self.build_chat_input_from_history_no_gen(head, thinking, tools) {
                Ok(v) => v,
                Err(_) => return 0,
            };
            full_prompt
                .iter()
                .zip(head_ids.iter())
                .take_while(|(a, b)| a == b)
                .count()
                .min(full_prompt.len().saturating_sub(1))
        }

        /// Sub-key under which the **system-boundary** (batch fan-out) snapshot
        /// lives, distinct from `key` which holds the **full-prompt**
        /// (multi-turn extend) snapshot. Two snapshots per logical key let one
        /// cache serve both reuse patterns optimally — see `prefix_fork`. The
        /// NUL separator can't collide with a caller-supplied key.
        fn sys_key(key: &str) -> String {
            format!("{key}\u{0}sys")
        }

        /// Fork a cache for `prompt` from the best snapshot under `key`: the
        /// full-prompt snapshot (`key`, for multi-turn extends) or the
        /// system-boundary snapshot (`sys_key(key)`, for batch fan-out
        /// siblings). A snapshot is reusable only when its saved tokens are a
        /// **strict prefix** of `prompt` (`lcp == saved_len < prompt.len`), so
        /// the clone is forked at `offset == lcp` with NO `truncate_to`
        /// rollback — correct even for >sliding_window prefixes. Picks the
        /// longer match. No match → fresh cache + `"miss"`.
        fn prefix_fork(
            &mut self,
            prompt: &[u32],
            key: &str,
        ) -> (NativeGemma4PromptCache, &'static str) {
            let strict = |e: &Gemma4PrefixCacheEntry| -> Option<usize> {
                let lcp = e
                    .prefix_tokens
                    .iter()
                    .zip(prompt.iter())
                    .take_while(|(a, b)| a == b)
                    .count();
                (lcp == e.prefix_tokens.len() && lcp < prompt.len()).then_some(lcp)
            };
            let sys_k = Self::sys_key(key);
            let full_lcp = self.prefix_caches.get(key).and_then(&strict);
            let sys_lcp = self.prefix_caches.get(&sys_k).and_then(&strict);
            let pick = match (full_lcp, sys_lcp) {
                (Some(f), Some(s)) if s > f => Some((sys_k, s, "hit-sys")),
                (Some(f), _) => Some((key.to_string(), f, "hit-full")),
                (None, Some(s)) => Some((sys_k, s, "hit-sys")),
                (None, None) => None,
            };
            match pick {
                Some((k, lcp, kind)) => {
                    let entry = self.prefix_caches.get_mut(&k).unwrap();
                    entry.last_access = Instant::now();
                    entry.hits += 1;
                    let cache = entry.master.clone();
                    debug_assert_eq!(cache.offset(), lcp);
                    (cache, kind)
                }
                None => {
                    // L2 disk tier: rehydrate a persisted snapshot (survives
                    // restart / eviction) before paying a cold prefill. Register
                    // the hit in-mem so later forks this process hit memory.
                    if let Some((k, cache, prefix_tokens)) = self.load_prefix_from_disk(prompt, key)
                    {
                        eprintln!(
                            "[gemma4] prefix-cache: disk rehydrate key={k:?} prefix={}",
                            prefix_tokens.len()
                        );
                        self.prefix_caches.insert(
                            k,
                            Gemma4PrefixCacheEntry {
                                master: cache.clone(),
                                prefix_tokens,
                                last_access: Instant::now(),
                                hits: 1,
                            },
                        );
                        return (cache, "hit-disk");
                    }
                    (self.make_cache_for_prompt_len(prompt.len()), "miss")
                }
            }
        }

        /// On a `prefix_fork` miss, prime the system-boundary snapshot for
        /// `key` (under `sys_key`) so sibling requests fork from it. Prefills
        /// `prompt[..boundary]` only (no decode → `offset == boundary`), clones
        /// it as the snapshot, and leaves `cache` advanced to `boundary` so the
        /// caller continues with the suffix on the same cache. No-op when
        /// `boundary == 0` (nothing worth caching).
        fn prime_system_snapshot(
            &mut self,
            cache: &mut NativeGemma4PromptCache,
            prompt: &[u32],
            key: &str,
            boundary: usize,
            ctx: &'static str,
        ) -> Result<()> {
            if boundary == 0 {
                return Ok(());
            }
            self.model
                .forward_last_token(&prompt[..boundary], cache)
                .context(ctx)?;
            self.save_prefix_snapshot(&Self::sys_key(key), cache, &prompt[..boundary]);
            Ok(())
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
            temperature: f32,
            top_p: f32,
            ov: &crate::SamplingOverrides,
            thinking: bool,
            prefix_cache_key: &str,
            tools: &[crate::gemma4_tools::imp::ToolDef<'_>],
            tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
        ) -> Result<ParsedResponse> {
            let (prompt, prefill_tokens) =
                self.build_prompt_and_prefill(messages, thinking, tools, tool_choice, false)?;
            if prompt.is_empty() {
                return Err(anyhow!("chat_with_prefix_cache: empty prompt"));
            }

            // ── Lookup + cache fork (dual-snapshot strategy) ──
            // `prefix_fork` reuses the full-prompt or system-boundary snapshot
            // (whichever is the longer strict prefix) with NO `truncate_to`
            // rollback. On a miss we prime the system-boundary snapshot so
            // sibling requests (same system prompt, varying query) fork from
            // it — works even for >sliding_window prefixes.
            use crate::chat_io::ResolvedToolChoice;
            let effective_tools_for_boundary: &[crate::gemma4_tools::imp::ToolDef<'_>] =
                match tool_choice {
                    ResolvedToolChoice::None => &[],
                    _ => tools,
                };
            let (mut cache, hit_kind) = self.prefix_fork(&prompt, prefix_cache_key);
            if hit_kind == "miss" {
                let boundary = self.batch_fanout_boundary(
                    messages,
                    thinking,
                    effective_tools_for_boundary,
                    &prompt,
                );
                self.prime_system_snapshot(
                    &mut cache,
                    &prompt,
                    prefix_cache_key,
                    boundary,
                    "chat_with_prefix_cache: prime boundary prefill",
                )?;
            }

            let prefilled = cache.offset();
            let suffix_len = prompt.len().saturating_sub(prefilled);
            eprintln!(
                "[gemma4-backend] prefix-cache key={prefix_cache_key:?} \
                 result={hit_kind} prefilled={prefilled} suffix_len={suffix_len}"
            );

            // ── Generate (suffix prefill + decode) ──
            // No post-decode snapshot: the boundary snapshot saved on the miss
            // path is stable across sibling requests and needs no rollback to
            // reuse, so we keep it as-is.
            let suffix = &prompt[prefilled..];
            let cfg = GenerateConfig {
                max_new_tokens,
                stop_on_eos: true,
                sampling: build_sampling_config(temperature, top_p, ov),
            };
            let stats = self
                .model
                .generate_with_cache(suffix, &cfg, Some(&mut cache))
                .context("chat_with_prefix_cache: generate_with_cache")?;
            log_chat_done(
                stats.prompt_tokens,
                stats.prefill_ms,
                stats.decode_steps,
                stats.decode_ms,
            );

            // ── Parse decoded tokens into ParsedResponse ──
            let mut parser = ResponseParser::new(&self.chat);
            for tok in &prefill_tokens {
                parser.push(*tok)?;
            }
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

        /// Snapshot the post-prefill cache under `key` for future requests
        /// to fork from. MUST be called immediately after prefill, when
        /// `cache.offset() == prompt.len()` exactly — that's the only point
        /// where the snapshot represents just the prompt prefix without
        /// needing a `truncate_to` step. Post-decode snapshots used to call
        /// `truncate_to(prompt.len())` to roll back the offset advance from
        /// decoded tokens, but rotating sliding caches reject post-rotation
        /// rollback (offset > max_size). Snapshotting pre-decode avoids the
        /// rollback entirely and works regardless of which decode branch
        /// (greedy/sampled/MTP) runs next.
        fn save_prefix_snapshot(
            &mut self,
            key: &str,
            cache: &NativeGemma4PromptCache,
            prompt: &[u32],
        ) {
            debug_assert_eq!(
                cache.offset(),
                prompt.len(),
                "save_prefix_snapshot must be called at post-prefill (cache.offset == prompt.len)"
            );
            self.prefix_caches.insert(
                key.to_string(),
                Gemma4PrefixCacheEntry {
                    master: cache.clone(),
                    prefix_tokens: prompt.to_vec(),
                    last_access: Instant::now(),
                    hits: 0,
                },
            );
            // Every snapshot insert funnels through here, so bounding the map
            // here keeps `prefix_caches` from growing without limit across a
            // long-lived server (each entry pins a full KV snapshot — hundreds
            // of MB for a multi-k-token prefix).
            self.evict_prefix_cache();

            // L2 disk tier: durably persist this boundary snapshot so a future
            // process forks it instead of cold-prefilling. This funnel is the
            // only safe point (`offset == prompt.len()`, pre-decode, no rotation
            // rollback). Synchronous (durable across restart); no-op when the
            // disk tier is off; non-fatal + skipped for quantized/TQ layers
            // (`to_disk_records` errors → dense-only until those are wired).
            if self.disk.is_some() {
                match cache.to_disk_records() {
                    Ok((layers, records)) => {
                        let created_at_unix = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_secs())
                            .unwrap_or(0);
                        let fp = self.disk.as_ref().unwrap().fingerprint().to_string();
                        let manifest = KvManifest {
                            model_fingerprint: fp,
                            created_at_unix,
                            position: prompt.len(),
                            prefix_tokens: prompt.to_vec(),
                            last_token: None,
                            is_deep: true,
                            layers,
                        };
                        if let Err(e) = self.disk.as_mut().unwrap().put(key, &manifest, &records) {
                            eprintln!("[gemma4] kv-disk persist skipped (key={key:?}): {e:#}");
                        }
                    }
                    Err(e) => {
                        eprintln!("[gemma4] kv-disk persist skipped (key={key:?}): {e:#}");
                    }
                }
                // L2 spill: under memory pressure, drop cold in-memory prefix
                // snapshots (now durably on disk) to free RAM; they rehydrate via
                // `prefix_fork`'s disk path on next access. Off unless
                // `LUMEN_KV_SPILL_MEM_GB` is set.
                if crate::kv_disk::under_memory_pressure() {
                    self.spill_prefix_cache(crate::kv_disk::spill_keep_floor());
                }
            }
        }

        /// Drop the coldest in-memory prefix snapshots until at most `keep_n`
        /// remain (LRU by `last_access`). Spilled entries stay on the disk tier
        /// and rehydrate via `prefix_fork`'s disk lookup.
        fn spill_prefix_cache(&mut self, keep_n: usize) {
            let before = self.prefix_caches.len();
            while self.prefix_caches.len() > keep_n {
                let Some(lru) = self
                    .prefix_caches
                    .iter()
                    .min_by_key(|(_, e)| e.last_access)
                    .map(|(k, _)| k.clone())
                else {
                    break;
                };
                self.prefix_caches.remove(&lru);
            }
            let after = self.prefix_caches.len();
            if after < before {
                eprintln!(
                    "[gemma4] prefix-cache: spilled {} cold snapshot(s) to disk (kept {after})",
                    before - after
                );
            }
        }

        /// L2 disk tier: try to rehydrate a persisted boundary snapshot for
        /// `key` (or its `sys_key`) whose tokens are a STRICT prefix of `prompt`
        /// (so the forked cache's `offset == prefix_len` with no rotation
        /// rollback). Picks the longer match. Returns the loaded cache + its
        /// disk key + prefix tokens, or `None` (disk off / miss / non-prefix /
        /// corrupt / unsupported layers).
        fn load_prefix_from_disk(
            &mut self,
            prompt: &[u32],
            key: &str,
        ) -> Option<(String, NativeGemma4PromptCache, Vec<u32>)> {
            self.disk.as_ref()?;
            let candidates = [key.to_string(), Self::sys_key(key)];
            let mut best: Option<(String, NativeGemma4PromptCache, Vec<u32>)> = None;
            for k in candidates {
                let loaded = match self.disk.as_mut().unwrap().get(&k) {
                    Ok(Some(v)) => v,
                    _ => continue,
                };
                let (manifest, records) = loaded;
                let plen = manifest.prefix_tokens.len();
                let is_strict =
                    plen > 0 && plen < prompt.len() && prompt.starts_with(&manifest.prefix_tokens);
                if !is_strict {
                    continue;
                }
                if best
                    .as_ref()
                    .map(|(_, _, pt)| plen > pt.len())
                    .unwrap_or(true)
                {
                    match NativeGemma4PromptCache::from_disk_records(&manifest.layers, &records) {
                        Ok(cache) => best = Some((k, cache, manifest.prefix_tokens)),
                        Err(e) => {
                            eprintln!("[gemma4] kv-disk rehydrate decode failed (key={k:?}): {e:#}")
                        }
                    }
                }
            }
            best
        }

        /// Max live prefix-cache entries before LRU eviction (`LUMEN_PREFIX_CACHE_MAX_ENTRIES`,
        /// default 16). Note dual-snapshot uses up to 2 entries per logical key
        /// (`key` + `sys_key`), so 16 ≈ 8 distinct system prompts. Cached on first read.
        fn prefix_cache_max_entries() -> usize {
            static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
            *CACHED.get_or_init(|| {
                std::env::var("LUMEN_PREFIX_CACHE_MAX_ENTRIES")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .filter(|&n: &usize| n > 0)
                    .unwrap_or(16)
            })
        }

        /// Time-to-live for prefix-cache entries in seconds
        /// (`LUMEN_PREFIX_CACHE_TTL_SECS`, default 0 = disabled — rely on LRU).
        /// Entries not accessed within the TTL are dropped on the next insert.
        /// Cached on first read.
        fn prefix_cache_ttl_secs() -> u64 {
            static CACHED: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
            *CACHED.get_or_init(|| {
                std::env::var("LUMEN_PREFIX_CACHE_TTL_SECS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0)
            })
        }

        /// Enforce the prefix-cache bounds: drop TTL-stale entries, then evict
        /// least-recently-used entries until at most `max_entries` remain.
        /// `last_access` (updated on every `prefix_fork` hit) is the recency
        /// key. Cheap: O(n) per call, n bounded by `max_entries`.
        fn evict_prefix_cache(&mut self) {
            let ttl = Self::prefix_cache_ttl_secs();
            if ttl > 0 {
                let now = Instant::now();
                self.prefix_caches
                    .retain(|_, e| now.duration_since(e.last_access).as_secs() < ttl);
            }
            let max = Self::prefix_cache_max_entries();
            while self.prefix_caches.len() > max {
                let Some(lru) = self
                    .prefix_caches
                    .iter()
                    .min_by_key(|(_, e)| e.last_access)
                    .map(|(k, _)| k.clone())
                else {
                    break;
                };
                self.prefix_caches.remove(&lru);
            }
        }

        /// Streaming variant of `chat_with_prefix_cache`. Mirrors the same
        /// lookup → fork → suffix-prefill → snapshot flow but routes through
        /// `decode_streaming_with_prompt` so per-token `BackendStreamEvent`s
        /// reach the caller's SSE writer. Used by the chat-stream HTTP path
        /// (the GUI / OpenAI clients) when an auto-key from the system
        /// prompt or an explicit session_id resolves to a string.
        pub fn chat_streaming_with_prefix_cache(
            &mut self,
            messages: &[(String, String)],
            max_new_tokens: usize,
            temperature: f32,
            top_p: f32,
            ov: &crate::SamplingOverrides,
            thinking: bool,
            prefix_cache_key: &str,
            tools: &[crate::gemma4_tools::imp::ToolDef<'_>],
            tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
            response_schema: Option<&serde_json::Value>,
            on_event: impl FnMut(BackendStreamEvent<'_>) -> Result<()>,
        ) -> Result<ParsedResponse> {
            let (prompt, prefill_tokens) =
                self.build_prompt_and_prefill(messages, thinking, tools, tool_choice, false)?;
            if prompt.is_empty() {
                return Err(anyhow!("chat_streaming_with_prefix_cache: empty prompt"));
            }
            // Compute the trailing `<start_of_turn>model\n` (+ optional empty
            // thought channel + optional tool_choice prefill) length so the
            // snapshot can stop just before it. This is the only segment
            // that diverges between turn N (where it sits at the prompt
            // tail with no following content) and turn N+1 (where the same
            // template segment sits mid-prompt followed by the actual
            // assistant response from turn N). Critically: use the *same*
            // tools / tool_choice as the full prompt build, otherwise tool
            // definitions in the system block create a spurious diff that
            // inflates trailing_header_len by hundreds-to-thousands of
            // tokens (and we'd snapshot way too early, cache becomes a
            // misleading subset of the actual prompt prefix).
            use crate::chat_io::ResolvedToolChoice;
            let effective_tools_for_no_gen: &[crate::gemma4_tools::imp::ToolDef<'_>] =
                match tool_choice {
                    ResolvedToolChoice::None => &[],
                    _ => tools,
                };
            // `prompt` is `render_to_ids_with_tools(add_gen=true) +
            // tool_choice_prefill`. `no_gen` is
            // `render_to_ids_with_tools(add_gen=false)`. Diff =
            // generation prompt block + tool_choice prefill. All three of
            // those segments are PROMPT-TAIL-only and diverge in the next
            // turn the same way (mid-prompt model header tokens differ from
            // prompt-tail).
            let trailing_header_len = self
                .build_chat_input_no_gen(messages, thinking, effective_tools_for_no_gen)
                .map(|no_gen| prompt.len().saturating_sub(no_gen.len()))
                .unwrap_or(0);
            // Dual-snapshot: fork the longest strict-prefix snapshot (full or
            // system) with no rollback; on a miss prime the system-boundary
            // snapshot. `decode_streaming_with_prompt` separately records the
            // full-prompt snapshot under `key` (pre-decode, at
            // `prompt.len() - trailing_header_len`) for multi-turn extends — so
            // one streaming call maintains BOTH snapshots.
            let (mut cache, hit_kind) = self.prefix_fork(&prompt, prefix_cache_key);
            if hit_kind == "miss" {
                let boundary = self.batch_fanout_boundary(
                    messages,
                    thinking,
                    effective_tools_for_no_gen,
                    &prompt,
                );
                self.prime_system_snapshot(
                    &mut cache,
                    &prompt,
                    prefix_cache_key,
                    boundary,
                    "chat_streaming_with_prefix_cache: prime boundary prefill",
                )?;
            }
            let suffix_len = prompt.len().saturating_sub(cache.offset());
            eprintln!(
                "[gemma4-backend] prefix-cache key={prefix_cache_key:?} \
                 result={hit_kind} prefilled={} suffix_len={suffix_len} header_tail={trailing_header_len}",
                cache.offset()
            );
            let grammar = self.select_grammar_state(tools, tool_choice, response_schema);
            self.decode_streaming_with_prompt(
                prompt,
                prefill_tokens,
                max_new_tokens,
                temperature,
                top_p,
                ov,
                grammar,
                Some(cache),
                Some((prefix_cache_key.to_string(), trailing_header_len)),
                None, /* vision */
                on_event,
            )
        }

        /// History variant of `chat_streaming_with_prefix_cache` — same
        /// caching behavior but builds the prompt from structured
        /// `ChatTurn`s (turn-2+ requests that include `tool_calls` or
        /// `role:"tool"` entries the flat `(role, content)` tuple shape
        /// can't represent).
        pub fn chat_streaming_from_history_with_prefix_cache(
            &mut self,
            turns: &[crate::chat_io::ChatTurn<'_>],
            max_new_tokens: usize,
            temperature: f32,
            top_p: f32,
            ov: &crate::SamplingOverrides,
            thinking: bool,
            prefix_cache_key: &str,
            tools: &[crate::gemma4_tools::imp::ToolDef<'_>],
            tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
            response_schema: Option<&serde_json::Value>,
            on_event: impl FnMut(BackendStreamEvent<'_>) -> Result<()>,
        ) -> Result<ParsedResponse> {
            let (prompt, prefill_tokens) = self.build_prompt_and_prefill_from_history(
                turns,
                thinking,
                tools,
                tool_choice,
                false,
            )?;
            if prompt.is_empty() {
                return Err(anyhow!(
                    "chat_streaming_from_history_with_prefix_cache: empty prompt"
                ));
            }
            use crate::chat_io::ResolvedToolChoice;
            let effective_tools_for_no_gen: &[crate::chat_io::ToolDef<'_>] = match tool_choice {
                ResolvedToolChoice::None => &[],
                _ => tools,
            };
            // Pass the same `effective_tools` as the full prompt build so the
            // diff captures ONLY the generation prompt (+ optional tool_choice
            // prefill) — not tool definitions in the system block (those are
            // shared across turns and must NOT be excluded from the snapshot).
            let trailing_header_len = self
                .build_chat_input_from_history_no_gen(turns, thinking, effective_tools_for_no_gen)
                .map(|no_gen| prompt.len().saturating_sub(no_gen.len()))
                .unwrap_or(0);
            let (mut cache, hit_kind) = self.prefix_fork(&prompt, prefix_cache_key);
            if hit_kind == "miss" {
                let boundary = self.batch_fanout_boundary_from_history(
                    turns,
                    thinking,
                    effective_tools_for_no_gen,
                    &prompt,
                );
                self.prime_system_snapshot(
                    &mut cache,
                    &prompt,
                    prefix_cache_key,
                    boundary,
                    "chat_streaming_from_history_with_prefix_cache: prime boundary prefill",
                )?;
            }
            let suffix_len = prompt.len().saturating_sub(cache.offset());
            eprintln!(
                "[gemma4-backend] prefix-cache key={prefix_cache_key:?} \
                 result={hit_kind} prefilled={} suffix_len={suffix_len} header_tail={trailing_header_len} (from-history)",
                cache.offset()
            );
            let grammar = self.select_grammar_state(tools, tool_choice, response_schema);
            self.decode_streaming_with_prompt(
                prompt,
                prefill_tokens,
                max_new_tokens,
                temperature,
                top_p,
                ov,
                grammar,
                Some(cache),
                Some((prefix_cache_key.to_string(), trailing_header_len)),
                None, /* vision */
                on_event,
            )
        }

        /// Non-streaming history variant — used by tool-aware completion
        /// requests that issue follow-up calls after a tool result arrives.
        /// Same lookup → fork → snapshot pattern as `chat_with_prefix_cache`,
        /// but the prompt comes from `ChatTurn`s instead of `(role,
        /// content)` tuples.
        pub fn chat_from_history_with_prefix_cache(
            &mut self,
            turns: &[crate::chat_io::ChatTurn<'_>],
            max_new_tokens: usize,
            temperature: f32,
            top_p: f32,
            ov: &crate::SamplingOverrides,
            thinking: bool,
            prefix_cache_key: &str,
            tools: &[crate::chat_io::ToolDef<'_>],
            tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
        ) -> Result<ParsedResponse> {
            let (prompt, prefill_tokens) = self.build_prompt_and_prefill_from_history(
                turns,
                thinking,
                tools,
                tool_choice,
                false,
            )?;
            if prompt.is_empty() {
                return Err(anyhow!("chat_from_history_with_prefix_cache: empty prompt"));
            }
            // Dual-snapshot strategy (see `chat_with_prefix_cache`): fork the
            // longest strict-prefix snapshot (no rollback), prime the
            // system-boundary snapshot on a miss.
            use crate::chat_io::ResolvedToolChoice;
            let effective_tools_for_boundary: &[crate::chat_io::ToolDef<'_>] = match tool_choice {
                ResolvedToolChoice::None => &[],
                _ => tools,
            };
            let (mut cache, hit_kind) = self.prefix_fork(&prompt, prefix_cache_key);
            if hit_kind == "miss" {
                let boundary = self.batch_fanout_boundary_from_history(
                    turns,
                    thinking,
                    effective_tools_for_boundary,
                    &prompt,
                );
                self.prime_system_snapshot(
                    &mut cache,
                    &prompt,
                    prefix_cache_key,
                    boundary,
                    "chat_from_history_with_prefix_cache: prime boundary prefill",
                )?;
            }

            let prefilled = cache.offset();
            let suffix_len = prompt.len().saturating_sub(prefilled);
            eprintln!(
                "[gemma4-backend] prefix-cache key={prefix_cache_key:?} \
                 result={hit_kind} prefilled={prefilled} suffix_len={suffix_len} (from-history non-streaming)"
            );

            // No post-decode snapshot — the boundary snapshot above is reused
            // without rollback by subsequent requests.
            let suffix = &prompt[prefilled..];
            let cfg = GenerateConfig {
                max_new_tokens,
                stop_on_eos: true,
                sampling: build_sampling_config(temperature, top_p, ov),
            };
            let stats = self
                .model
                .generate_with_cache(suffix, &cfg, Some(&mut cache))
                .context("chat_from_history_with_prefix_cache: generate_with_cache")?;
            log_chat_done(
                stats.prompt_tokens,
                stats.prefill_ms,
                stats.decode_steps,
                stats.decode_ms,
            );

            let mut parser = ResponseParser::new(&self.chat);
            for tok in &prefill_tokens {
                parser.push(*tok)?;
            }
            for token in &stats.generated_tokens {
                parser.push(*token)?;
            }
            parser.finalize()
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
            temperature: f32,
            top_p: f32,
            ov: &crate::SamplingOverrides,
            thinking: bool,
            tools: &[crate::gemma4_tools::imp::ToolDef<'_>],
            tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
            response_schema: Option<&serde_json::Value>,
            on_event: impl FnMut(BackendStreamEvent<'_>) -> Result<()>,
        ) -> Result<ParsedResponse> {
            let (prompt, prefill_tokens) = self.build_prompt_and_prefill(
                messages,
                thinking,
                tools,
                tool_choice,
                response_schema.is_some(),
            )?;
            let grammar = self.select_grammar_state(tools, tool_choice, response_schema);
            self.decode_streaming_with_prompt(
                prompt,
                prefill_tokens,
                max_new_tokens,
                temperature,
                top_p,
                ov,
                grammar,
                None, /* pre_built_cache */
                None, /* snapshot_prefix_key */
                None, /* vision */
                on_event,
            )
        }

        /// [`Self::chat_streaming`] with inline images.
        ///
        /// `images[i]` holds the encoded image byte streams attached to
        /// `messages[i]`. With nothing attached this is byte-identical to
        /// [`Self::chat_streaming`], so callers can route every request here.
        ///
        /// Images are encoded once, up front, and handed to the prefill loop;
        /// the chunked prefill then splices each image's rows into whichever
        /// chunk covers them. Decode is pure text and takes the unchanged
        /// path. Like the non-streaming variant this skips the prefix cache —
        /// the cached prefix is keyed on text alone, and a vision prompt's
        /// placeholder rows only mean anything together with the image they
        /// were spliced from.
        #[allow(clippy::too_many_arguments)]
        pub fn chat_streaming_with_images(
            &mut self,
            messages: &[(String, String)],
            images: &[Vec<Vec<u8>>],
            max_new_tokens: usize,
            temperature: f32,
            top_p: f32,
            ov: &crate::SamplingOverrides,
            thinking: bool,
            tools: &[crate::gemma4_tools::imp::ToolDef<'_>],
            tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
            response_schema: Option<&serde_json::Value>,
            on_event: impl FnMut(BackendStreamEvent<'_>) -> Result<()>,
        ) -> Result<ParsedResponse> {
            if !images.iter().any(|v| !v.is_empty()) {
                return self.chat_streaming(
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
                );
            }

            let (prompt, prefill_tokens, flat) = self.build_image_prompt(
                messages,
                images,
                thinking,
                tools,
                tool_choice,
                response_schema.is_some(),
            )?;
            // Encode before the prefill loop starts: the runs are prompt-global
            // and every chunk needs to be able to look up any of them.
            let vision = self
                .model
                .encode_images_for_prompt(&prompt, &flat)
                .context("chat_streaming: encode images")?;
            let grammar = self.select_grammar_state(tools, tool_choice, response_schema);
            self.decode_streaming_with_prompt(
                prompt,
                prefill_tokens,
                max_new_tokens,
                temperature,
                top_p,
                ov,
                grammar,
                None, /* pre_built_cache — images bypass the prefix cache */
                None, /* snapshot_prefix_key */
                Some(vision),
                on_event,
            )
        }

        /// Phase 1.5: structured-history variant of `chat_streaming` for
        /// turn-2+ agent loops. The flat-history streaming path can't
        /// represent `role:"tool"` messages — this one routes through
        /// the structured renderer (`build_chat_input_from_history`)
        /// and shares the same decode loop. Used when the request's
        /// message stream includes an assistant `tool_calls` or
        /// `role:"tool"` entry.
        pub fn chat_streaming_from_history(
            &mut self,
            turns: &[crate::chat_io::ChatTurn<'_>],
            max_new_tokens: usize,
            temperature: f32,
            top_p: f32,
            ov: &crate::SamplingOverrides,
            thinking: bool,
            tools: &[crate::gemma4_tools::imp::ToolDef<'_>],
            tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
            response_schema: Option<&serde_json::Value>,
            on_event: impl FnMut(BackendStreamEvent<'_>) -> Result<()>,
        ) -> Result<ParsedResponse> {
            self.chat_streaming_from_history_with_images(
                turns,
                &[],
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

        /// [`Self::chat_streaming_from_history`] with images on `User` turns.
        #[allow(clippy::too_many_arguments)]
        pub fn chat_streaming_from_history_with_images(
            &mut self,
            turns: &[crate::chat_io::ChatTurn<'_>],
            images: &[Vec<Vec<u8>>],
            max_new_tokens: usize,
            temperature: f32,
            top_p: f32,
            ov: &crate::SamplingOverrides,
            thinking: bool,
            tools: &[crate::gemma4_tools::imp::ToolDef<'_>],
            tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
            response_schema: Option<&serde_json::Value>,
            on_event: impl FnMut(BackendStreamEvent<'_>) -> Result<()>,
        ) -> Result<ParsedResponse> {
            let (counts, prepared) = self.measure_turn_images(images)?;
            let (prompt, prefill_tokens) = self.build_prompt_and_prefill_from_history_with_images(
                turns,
                &counts,
                thinking,
                tools,
                tool_choice,
                response_schema.is_some(),
            )?;
            let vision = if prepared.is_empty() {
                None
            } else {
                Some(
                    self.model
                        .encode_images_for_prompt(&prompt, &prepared)
                        .context("chat_streaming_from_history: encode images")?,
                )
            };
            let grammar = self.select_grammar_state(tools, tool_choice, response_schema);
            self.decode_streaming_with_prompt(
                prompt,
                prefill_tokens,
                max_new_tokens,
                temperature,
                top_p,
                ov,
                grammar,
                None, /* pre_built_cache — images bypass the prefix cache */
                None, /* snapshot_prefix_key */
                vision,
                on_event,
            )
        }

        /// Decode + resize every image once, returning per-turn counts (for the
        /// renderer) alongside the flattened prepared images in prompt order
        /// (for the tower).
        fn measure_turn_images(
            &self,
            images: &[Vec<Vec<u8>>],
        ) -> Result<(Vec<Vec<usize>>, Vec<crate::gemma4_vision::PreparedImage>)> {
            let mut counts = Vec::with_capacity(images.len());
            let mut flat = Vec::new();
            for per_turn in images {
                let mut row = Vec::with_capacity(per_turn.len());
                for bytes in per_turn {
                    let prepared = self.model.prepare_image(bytes)?;
                    row.push(prepared.num_soft_tokens);
                    flat.push(prepared);
                }
                counts.push(row);
            }
            Ok((counts, flat))
        }

        fn decode_streaming_with_prompt(
            &mut self,
            prompt: Vec<u32>,
            prefill_tokens: Vec<u32>,
            max_new_tokens: usize,
            temperature: f32,
            top_p: f32,
            ov: &crate::SamplingOverrides,
            // Phase 2.5: grammar-constrained tool calling. `None` for the
            // default text path; `Some(state)` engages the llguidance mask
            // during the sampled decode branch (no-op on greedy / MTP — those
            // paths don't sample multinomially and would need different
            // wiring). The state is observed every sampled step so the
            // lazy trigger fires on `<|tool_call>` even when grammar
            // started inactive.
            mut grammar: Option<Gemma4GrammarState>,
            // Pre-built cache from a prefix-cache lookup. When `Some`, its
            // `offset()` tells the chunked prefill where the new suffix
            // begins — only `&prompt[offset..]` gets prefilled. When `None`,
            // the function allocates a fresh empty cache (legacy behavior).
            pre_built_cache: Option<NativeGemma4PromptCache>,
            // When `Some((key, trailing_header_len))`, the cache state is
            // snapshotted under `key` AFTER prefilling `prompt[..prompt.len()
            // - trailing_header_len]` but BEFORE prefilling the trailing
            // header. The trailing header is the `<start_of_turn>model\n`
            // (+ optional empty thought channel) segment that the chat
            // template appends when add_generation_prompt=true; it sits at
            // the tail of turn N but mid-prompt of turn N+1, so excluding it
            // from the snapshot lets turn N+1 hit with exact LCP match
            // instead of falling 4-6 tokens short and triggering a truncate
            // on the cloned master (which fails for rotating sliding caches
            // post-rotation). `None` means "do not record" (one-off / debug
            // requests). `Some((key, 0))` snapshots the full prompt — used
            // when trailing header detection isn't available.
            snapshot_prefix_key: Option<(String, usize)>,
            // Encoded images for an image-bearing prompt: prompt-global
            // `(start, len)` placeholder runs paired with their `[len_i,
            // hidden]` soft tokens, already produced by
            // `encode_images_for_prompt`. Encoding happens once here, before
            // the chunk loop, and each chunk splices whichever rows fall in
            // its window. `None` is the text path, unchanged.
            vision: Option<(Vec<(usize, usize)>, Vec<mlx_rs::Array>)>,
            mut on_event: impl FnMut(BackendStreamEvent<'_>) -> Result<()>,
        ) -> Result<ParsedResponse> {
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
            // Per-request adaptive cache: when MODE=auto for TQ or simple Q4,
            // the resolution depends on this prompt's length, not the global
            // env at server launch. Pre-built caches (from prefix-cache hits)
            // keep whatever choice they were built with.
            let prompt_len = prompt.len();
            mlx_rs::with_new_default_stream(gen_stream, || -> Result<ParsedResponse> {
                let mut cache =
                    pre_built_cache.unwrap_or_else(|| self.make_cache_for_prompt_len(prompt_len));

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
                let requested_chunk: usize = std::env::var("LUMEN_GEMMA4_PREFILL_CHUNK")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .filter(|&n: &usize| n > 0)
                    .unwrap_or(2048);
                // ── Always-chunk invariant: single-pass OOM guard ──
                // The quantized-KV / TurboQuant attention path materializes the
                // full [heads, q_len, kv_len] scores array — fused flash-SDPA
                // (`mlx::fast::sdpa`, O(block) memory) is unavailable when K/V are
                // stored quantized, so Q@K^T is an explicit `quantized_matmul`
                // whose output lives in one Metal buffer. A single un-chunked pass
                // over a long prompt then allocates a scores array that exceeds
                // Metal's per-buffer cap (observed: 72 GB > 38.9 GB at q=k≈47K).
                // Bound q_len so one chunk's scores stay under a safe byte budget
                // given the worst-case kv_len (== full prompt length). This makes
                // chunking mandatory: an over-large env/config chunk is clamped
                // DOWN, never up. Default 8 GB fits any box that can host a 26B
                // model (Metal maxBufferLength ≥ ~16 GB there); kv-quant off uses
                // flash-SDPA and is unaffected by the materialization, but the
                // clamp also keeps the per-command-buffer intermediate graph
                // bounded, so it is applied uniformly. Override the budget with
                // `LUMEN_GEMMA4_PREFILL_SCORES_GB`.
                // The arithmetic lives in `prefill_budget` so it can be swept at
                // tier 0; the two backends had it duplicated. Behaviour here is
                // unchanged by the hoist.
                let scores_budget_bytes =
                    crate::prefill_budget::scores_budget_from_env("LUMEN_GEMMA4_PREFILL_SCORES_GB");
                let decision = crate::prefill_budget::clamp_chunk(
                    requested_chunk,
                    scores_budget_bytes,
                    self.model.config().text_config.num_attention_heads,
                    prompt.len(),
                );
                let chunk_size = decision.chunk;
                if decision.clamped() {
                    eprintln!(
                        "[prefill] chunk clamped {requested_chunk} → {chunk_size} \
                         (heads={} kv_upper={} budget={:.1}GB) — \
                         keeps single-chunk scores under the Metal buffer cap",
                        decision.heads.max(1),
                        decision.kv_upper.max(1),
                        scores_budget_bytes as f64 / 1e9
                    );
                }
                // Chunked prefill re-enabled (2026-05-14, take 2): the mask
                // builder now reads kv_actual from k_full.shape, so rotated
                // sliding caches no longer trigger broadcast mismatches.
                //
                // Prefix-cache hit path: when `pre_built_cache` already covers
                // tokens [0..cache.offset()] of the prompt, only the suffix
                // [cache.offset()..] needs to be prefilled. For a turn-2+ chat
                // where the system prompt is unchanged, this collapses the
                // common ~5K-token system prefix into a no-op and prefills
                // only the new user message (typically <100 tokens).
                let prefill_start = cache.offset();
                // Snapshot-split offset: when set, prefill is split into two
                // stages — chunks for `[prefill_start..snapshot_split]` run
                // first, then the snapshot is taken (cache.offset ==
                // snapshot_split exactly), then the trailing chunk for
                // `[snapshot_split..prompt.len()]` runs. The trailing chunk
                // is the assistant turn header (3-6 tokens). Excluding it
                // from the snapshot prefix lets the next turn's request hit
                // with exact LCP match instead of falling short and
                // requiring a truncate on the cloned master.
                let snapshot_split = match snapshot_prefix_key.as_ref() {
                    Some((_, h)) if *h > 0 && prefill_start + *h <= prompt.len() => {
                        Some(prompt.len() - *h)
                    }
                    _ => None,
                };
                let stage1_end = snapshot_split.unwrap_or(prompt.len());
                let stage1_data = &prompt[prefill_start..stage1_end];
                let stage1_chunks: Vec<&[u32]> = stage1_data.chunks(chunk_size).collect();
                let n_chunks = stage1_chunks.len();
                if prefill_start > 0 {
                    eprintln!(
                        "[prefill] prefix-cache fork: cached {} tokens, prefilling suffix {} tokens in {} chunks (size {})",
                        prefill_start,
                        stage1_data.len() + (prompt.len() - stage1_end),
                        n_chunks + if snapshot_split.is_some() { 1 } else { 0 },
                        chunk_size
                    );
                } else {
                    eprintln!(
                        "[prefill] start {} chunks (size {}, total {} tokens){}",
                        n_chunks + if snapshot_split.is_some() { 1 } else { 0 },
                        chunk_size,
                        prompt.len(),
                        if let Some(s) = snapshot_split {
                            format!(
                                " [split at {} for prefix-cache snapshot, trailing {} tokens]",
                                s,
                                prompt.len() - s
                            )
                        } else {
                            String::new()
                        }
                    );
                }
                // Wall-clock spanning all chunks + the final async_eval —
                // surfaces prefill cost separately in the chat done log
                // so users can see whether a slow response was prompt
                // processing or generation (the two costs are very
                // different per-token).
                let t_prefill_total = std::time::Instant::now();
                // Report the number of tokens we actually prefilled in this
                // call (== suffix length on a cache fork, == full prompt
                // length on a cold start). The downstream
                // `log_chat_done(prefill_prompt_tokens, prefill_total_ms, ...)`
                // tok/s reading should reflect the real work done, not the
                // cached prefix the GPU never touched.
                let prefill_prompt_tokens = prompt.len() - prefill_start;
                let mut logits_opt: Option<mlx_rs::Array> = None;
                for (i, chunk) in stage1_chunks.into_iter().enumerate() {
                    let t0 = std::time::Instant::now();
                    // Use forward_last_token: only the final chunk's logits feed
                    // into `argmax_last_token_lazy` below (intermediate chunks'
                    // logits are discarded), and decode only needs the last-
                    // position next-token logits. Slicing h to last position
                    // before the tied lm_head skips ~3 TFLOPs per long chunk
                    // (8K × hidden × vocab quantized matmul) whose output is
                    // immediately reduced to a single argmax. Bit-identical
                    // tokens; see playbook_lm_head_last_token_slice.md.
                    //
                    // With images attached the chunk goes through the
                    // soft-token variant instead: the runs are prompt-global,
                    // so the chunk's absolute start tells the splice which
                    // rows (if any) land in this window.
                    let chunk_start = prefill_start + i * chunk_size;
                    let chunk_logits = match vision.as_ref() {
                        Some((runs, soft)) => self.model.forward_last_token_with_soft(
                            chunk,
                            &mut cache,
                            chunk_start,
                            runs,
                            soft,
                        ),
                        None => self.model.forward_last_token(chunk, &mut cache),
                    }
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

                // ── Snapshot AFTER stage 1, BEFORE stage 2 ──
                // At this point `cache.offset() == stage1_end ==
                // prompt.len() - trailing_header_len`. The cache state
                // represents the conversation up through the last user
                // turn's `<end_of_turn>` — exactly the boundary that
                // matches across turn N and turn N+1. Snapshotting before
                // the trailing `<start_of_turn>model\n` (+ thought channel)
                // is what makes the next turn's LCP check land on the full
                // prefix length, no truncate needed on the cloned master.
                if let Some((key, _)) = snapshot_prefix_key.as_ref() {
                    if snapshot_split.is_some() {
                        self.save_prefix_snapshot(key, &cache, &prompt[..stage1_end]);
                    } else {
                        // No split (header_len == 0 or stage1 was the whole
                        // prompt): snapshot at post-prefill, full prompt.
                        // Future turns will fall through to the truncate
                        // path which may fail for rotating caches with long
                        // prompts (logged + skipped, non-fatal).
                        self.save_prefix_snapshot(key, &cache, &prompt);
                    }
                }

                // ── Stage 2: prefill the trailing header ──
                // The trailing `<start_of_turn>model\n` (+ optional thought
                // channel) is short (3-6 tokens) so a single forward call
                // is enough. Its logits become the prefill's last-token
                // logits feeding into the decode argmax.
                if let Some(split) = snapshot_split {
                    let trailing = &prompt[split..];
                    let trailing_logits = match vision.as_ref() {
                        Some((runs, soft)) => self
                            .model
                            .forward_last_token_with_soft(trailing, &mut cache, split, runs, soft),
                        None => self.model.forward_last_token(trailing, &mut cache),
                    }
                    .context("chat_streaming: prefill trailing header")?;
                    trailing_logits
                        .eval()
                        .context("chat_streaming: prefill trailing header eval")?;
                    logits_opt = Some(trailing_logits);
                }

                let logits = logits_opt
                    .ok_or_else(|| anyhow!("chat_streaming: empty prompt has no chunks"))?;
                let mut current = self
                    .model
                    .argmax_last_token_lazy(&logits)
                    .context("chat_streaming: prefill argmax_lazy")?;
                mlx_rs::transforms::async_eval([&current])
                    .context("chat_streaming: prefill async_eval")?;
                let prefill_total_ms = t_prefill_total.elapsed().as_secs_f64() * 1000.0;

                // tested mlx-lm's `mx.clear_cache()`
                // post-prefill pattern (generate.py:451) → NEGATIVE -4.4σ
                // (29.54 → 29.30 tok/s). On our path the prefill cache holds
                // intermediate buffers reused immediately by decode steps;
                // clearing forces re-alloc and net costs more than it saves.
                // No clear_cache().

                let mut parser = ResponseParser::new(&self.chat);
                // Phase 1.6: replay tool_choice prefill tokens so the
                // parser's state machine matches the prompt the model
                // saw (e.g. enters ToolCall on token 48 for Required).
                for tok in &prefill_tokens {
                    parser.push(*tok)?;
                }
                // WS-C #1: for the Eager required/named grammar, the
                // `<|tool_call>` opener (+ `call:NAME{` for a named choice) was
                // prefilled into the prompt via the tokens above — it was never
                // *sampled*, so an Eager matcher has not advanced past it. Replay
                // those same tokens through the matcher so its parse position
                // matches the model's context; otherwise the very first mask
                // would re-force the opener and corrupt the body. No-op for the
                // Lazy `auto` path (matcher inactive ⇒ `observe_prefill` returns
                // early) and when there are no prefill tokens. If replay desyncs
                // (a prefilled token the grammar doesn't accept at that
                // position), drop the grammar and sample freely rather than
                // decode against a corrupt matcher.
                if grammar.as_ref().is_some_and(|g| g.is_active()) && !prefill_tokens.is_empty() {
                    let mut desync: Option<String> = None;
                    if let Some(g) = grammar.as_mut() {
                        for tok in &prefill_tokens {
                            if let Err(e) = g.observe_prefill(*tok) {
                                desync = Some(format!("{e:#}"));
                                break;
                            }
                        }
                    }
                    if let Some(e) = desync {
                        eprintln!(
                            "[gemma4-backend] grammar prefill replay desynced \
                             (falling back to free sampling): {e}"
                        );
                        grammar = None;
                    }
                }
                let eos = self.model.eos_tokens().to_vec();

                let sampling_cfg = build_sampling_config(temperature, top_p, ov);

                // ── MTP decode branch (DEFAULT OFF — opt-in only) ──
                //
                // ⚠️  PERF WARNING: measured NET LOSS −51% on Apple Silicon
                // M3 Max batch=1 with Korean conversational workload
                // (77.8 → 37.4 tok/s decode, accept 19.8% at n_draft=6,
                // 256-token output). See memory
                // `gemma4_mtp_chat_path_default_off_2026_05_24.md` for the
                // full measurement + root-cause analysis.
                //
                // ⚠️  GREEDY NON-IDENTICAL by design: OFF path uses the
                // custom-FA-2 attention kernel on full-attn layers (5-10%
                // faster) while mtp_step internally forces mlx::fast::sdpa
                // (so Step A's S=1 and Step C's S=K+1 stay on the same
                // kernel for accept-rate sanity). The ~1-ULP kernel drift
                // means OFF and ON produce different greedy continuations
                // even at temperature=0. See gemma4_moe.rs:4820 +
                // `mtp_active` flag.
                //
                // STRUCTURAL LIMIT (external evidence): Apple Silicon
                // batch=1 + 26B-A4B MoE + spec decode rarely net-positive.
                // Best published M-series batch=1 number is +13% (M1 Max
                // code prompt, accept ~60%, lilting.ch). Korean
                // conversational n_draft=6 falls far below that. Likely
                // useful regimes: (a) batch ≥ 2 server scenarios, (b)
                // dense model class, (c) extremely repetitive workloads
                // (code completion). For Mac mini single-user agent
                // deploy, leave default OFF.
                //
                // Mirrors `NativeGemma4Model::generate`'s MTP path
                // (gemma4_moe.rs:6076). Routes through `mtp_step()`
                // (Step A→E speculative decoding); each call yields up
                // to `n_draft + 2` tokens.
                //
                // Gate: greedy only (`sampling_cfg.is_none()`), drafter
                // loaded via `try_enable_mtp()` at startup, AND
                // `LUMEN_GEMMA4_MTP=1` explicit opt-in. Falls through to
                // the existing sampled / greedy branches otherwise —
                // bit-identical to v0.4.3 when env is unset.
                if sampling_cfg.is_none()
                    && self.model.mtp_enabled()
                    && NativeGemma4Model::mtp_decode_enabled_env()
                {
                    let n_draft = NativeGemma4Model::mtp_block_size_env();
                    let t_decode = std::time::Instant::now();
                    // Sync-read the prefill argmax (already async_eval'd).
                    let mut current_u32 = self
                        .model
                        .read_token_u32(&current)
                        .context("chat_streaming(mtp): read prefill first token")?;
                    let mut all_tokens: Vec<u32> = Vec::with_capacity(max_new_tokens);
                    all_tokens.push(current_u32);
                    let state_before = parser.state();
                    parser.push(current_u32)?;
                    emit_token_event(
                        &self.chat,
                        &mut parser,
                        current_u32,
                        state_before,
                        &mut on_event,
                    )?;
                    let mut hit_eos = eos.contains(&current_u32);
                    let mut n_cycles: usize = 0;
                    let mut accepted_total: usize = 0;
                    let mut attempted_total: usize = 0;
                    while all_tokens.len() < max_new_tokens && !hit_eos {
                        let out = self
                            .model
                            .mtp_step(&mut cache, current_u32, n_draft)
                            .with_context(|| {
                                format!("chat_streaming(mtp): mtp_step cycle {n_cycles}")
                            })?;
                        n_cycles += 1;
                        accepted_total += out.n_accepted;
                        attempted_total += out.n_attempted;
                        for tok in &out.committed {
                            if all_tokens.len() >= max_new_tokens {
                                break;
                            }
                            all_tokens.push(*tok);
                            let state_before = parser.state();
                            parser.push(*tok)?;
                            emit_token_event(
                                &self.chat,
                                &mut parser,
                                *tok,
                                state_before,
                                &mut on_event,
                            )?;
                            if eos.contains(tok) {
                                hit_eos = true;
                                break;
                            }
                        }
                        // The last committed entry is the next call's input
                        // (correction on partial reject / bonus on full
                        // accept). Mirrors generate(MTP)'s loop.
                        current_u32 = *out
                            .committed
                            .last()
                            .expect("chat_streaming(mtp): mtp_step committed non-empty");
                    }
                    let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
                    let count = all_tokens.len();
                    let accept_rate = if attempted_total > 0 {
                        100.0 * accepted_total as f64 / attempted_total as f64
                    } else {
                        0.0
                    };
                    eprintln!(
                        "[gemma4-mtp] chat decode: {count} tokens in {n_cycles} cycles \
                         accept={accepted_total}/{attempted_total} ({accept_rate:.1}%) n_draft={n_draft}"
                    );
                    log_chat_done(prefill_prompt_tokens, prefill_total_ms, count, decode_ms);
                    return parser.finalize();
                }

                // ── Sampled decode branch ──
                //
                // When request triggers non-greedy sampling
                // (temperature > 0 OR REPEAT_PENALTY env != 1), use a
                // simple per-step loop that pulls last-position logits
                // to CPU, applies penalty + temperature + top-p, and
                // multinomial-samples the next token. No async
                // pipelining; the CPU sampling cost is ~1-2 ms / step
                // at vocab 262144 vs ~30 ms GPU step time, so the net
                // impact is < 5%. The greedy path below stays bit-
                // identical when sampling is disabled.
                if let Some(sampling) = sampling_cfg {
                    use crate::gemma4_sampling::imp::{
                        LogitCorrectionCtx, Xorshift64,
                        sample_next_token_with_eos_guard_and_grammar,
                    };
                    let mut rng = Xorshift64::new(sampling.seed);
                    let mut all_tokens: Vec<u32> = Vec::with_capacity(max_new_tokens);
                    let t_decode = std::time::Instant::now();
                    // Phase 2.5: grammar wired into the sampled-decode path.
                    // `None` here = no tools → bit-identical to pre-grammar
                    // behaviour. Caller passes `Some(state)` when the
                    // request offered tools AND `tool_choice` is non-`None`.
                    let mut grammar = grammar;

                    // Phase B (v0.6.0) — runtime logit correction for
                    // critical tool/channel/turn tokens on quantized
                    // variants. First call to `correction_table()` also
                    // flips the model's correction-capture flag, so every
                    // subsequent `forward_array_*` stashes `h_for_lm_head`.
                    // The prefill above already ran without capture (we
                    // discard its h) — first_tok is sampled uncorrected,
                    // streaming steps below get the correction. Gate B
                    // verification shows this is sufficient (first_tok is
                    // usually a turn/role boundary, not the tool-decision
                    // step).
                    let correction_table = self.correction_table();

                    // Env-gated soft-EOS suppression — see
                    // `sample_next_token_with_eos_guard`. Two guards:
                    //   LUMEN_MIN_TOKENS_BEFORE_EOS=N — hard mask of
                    //     `<turn|>` (id 106) for the first N tokens.
                    //   LUMEN_EOS_TOP_K_GUARD=K — at every step, soft
                    //     mask `<turn|>` when its rank in raw logits is
                    //     below top-K (outlier rejection). Active also
                    //     after the min_tokens window has elapsed.
                    // Both default 0 (off). Ollama has NO separate EOS
                    // guard — its implicit EOS clamp IS top_k (64 for this
                    // model), which we now apply globally in
                    // `sample_from_logits` (see LUMEN_TOP_K). Keeping these off avoids double-
                    // suppressing `<turn|>` and the non-termination runaway
                    // that causes (model can't end its turn). Set either
                    // env >0 to re-enable as a targeted guard.
                    let min_tokens_before_eos: usize = std::env::var("LUMEN_MIN_TOKENS_BEFORE_EOS")
                        .ok()
                        .and_then(|s| s.trim().parse::<usize>().ok())
                        .unwrap_or(0);
                    let eos_top_k_guard: usize = std::env::var("LUMEN_EOS_TOP_K_GUARD")
                        .ok()
                        .and_then(|s| s.trim().parse::<usize>().ok())
                        .unwrap_or(0);
                    let eos_min_logit_margin: f32 = std::env::var("LUMEN_EOS_MIN_LOGIT_MARGIN")
                        .ok()
                        .and_then(|s| s.trim().parse::<f32>().ok())
                        .unwrap_or(0.0);

                    let first_tok = sample_next_token_with_eos_guard_and_grammar(
                        &logits,
                        &all_tokens,
                        &sampling,
                        &mut rng,
                        min_tokens_before_eos,
                        eos_top_k_guard,
                        eos_min_logit_margin,
                        &eos,
                        grammar.as_mut(),
                        None,
                    )
                    .context("chat_streaming(sampled): sample prefill token")?;
                    all_tokens.push(first_tok);
                    let state_before = parser.state();
                    parser.push(first_tok)?;
                    emit_token_event(
                        &self.chat,
                        &mut parser,
                        first_tok,
                        state_before,
                        &mut on_event,
                    )?;

                    if !eos.contains(&first_tok) {
                        let runaway = lumen_core::runaway::RunawayDetector::from_env();
                        let mut thinking_budget = crate::gemma4_thinking::ChannelBudget::from_env();
                        thinking_budget.observe(first_tok);

                        // ── WS-B Lever 1: overlap scheduling (SGLang MLX
                        // `event_loop_overlap_mlx` 1:1 port for the sampled
                        // decode path) ────────────────────────────────────
                        //
                        // Default ON; `LUMEN_MLX_NO_OVERLAP=1` restores the
                        // exact original synchronous path.
                        //
                        // Per step we (a) sample token N (blocks on this
                        // step's logits via `last_logits_to_cpu_f32`'s eval),
                        // (b) build the next input + issue forward(N+1) and
                        // `async_eval` its logits to kick the GPU NOW, then
                        // (c) run the deferred `parser.push + emit_token_event`
                        // of token N-1 on the CPU while the GPU computes the
                        // N+1 logits. A size-1 FIFO (`pending`) holds exactly
                        // one (token, state_before) per step, so emitting one
                        // step late yields a byte-identical token stream: the
                        // parser still receives tokens in strict order, and
                        // sampling/stop/grammar/thinking-budget all read
                        // `all_tokens` (never the parser), so deferring the
                        // parser advance does not affect any control decision.
                        //
                        // Everything that affects correctness — sampling,
                        // `all_tokens.push`, thinking-budget force-close /
                        // channel-block, eos check, runaway check, hard-break
                        // — stays FULLY SYNCHRONOUS and in the original order.
                        // Only the parser advance + detok + SSE send is
                        // deferred (the parser is consumed solely by
                        // `emit_token_event` and `finalize`).
                        let overlap_enabled = std::env::var("LUMEN_MLX_NO_OVERLAP").is_err();
                        // size-1 FIFO: the previous step's (token, state_before)
                        // whose parser.push + emit was deferred.
                        let mut pending: Option<(u32, ParseState)> = None;

                        // Flush the deferred (token, state_before): advance the
                        // parser and run the detok + SSE emit. Kept here as a
                        // closure so prefill-handoff, the loop body, and the
                        // post-loop flush share one definition.
                        macro_rules! flush_pending {
                            () => {
                                if let Some((tok, st_before)) = pending.take() {
                                    parser.push(tok)?;
                                    emit_token_event(
                                        &self.chat,
                                        &mut parser,
                                        tok,
                                        st_before,
                                        &mut on_event,
                                    )?;
                                }
                            };
                        }

                        // `step_logits` always holds the logits for the NEXT
                        // token to sample. Seed it with the forward over
                        // `first_tok` (the prefill already produced
                        // `first_tok`; the cache offset points just past it).
                        //
                        // INVARIANT: exactly ONE `forward_array_last_token`
                        // per generated token — the cache is mutated in place,
                        // so a second forward over the same token would
                        // double-advance KV. The overlap path therefore does
                        // NOT re-forward; it carries the lazy logits array it
                        // already issued into the next iteration.
                        let mut current_u32 = first_tok;
                        let issue_forward = |model: &NativeGemma4Model,
                                             current: u32,
                                             cache: &mut NativeGemma4PromptCache|
                         -> Result<mlx_rs::Array> {
                            let input = mlx_rs::Array::from_slice(&[current as i32], &[1, 1])
                                .as_dtype(mlx_rs::Dtype::Int32)
                                .context("chat_streaming(sampled): build input array")?;
                            model
                                .forward_array_last_token(&input, cache)
                                .context("chat_streaming(sampled): decode forward")
                        };
                        let mut step_logits = issue_forward(&self.model, current_u32, &mut cache)?;
                        if overlap_enabled {
                            // Kick the GPU for the very first decode step so the
                            // first sample's host pull does not stall on a cold
                            // GPU.
                            mlx_rs::transforms::async_eval([&step_logits])
                                .context("chat_streaming(sampled): seed async_eval")?;
                        }

                        while all_tokens.len() < max_new_tokens {
                            // Phase B: build per-step correction context
                            // from the just-captured `h_for_lm_head`. h
                            // ownership transfers via take_*, so a forward
                            // without a paired sample-step would leak the
                            // captured clone (acceptable — never happens
                            // in the streaming loop).
                            let h_buf = if correction_table.is_some() {
                                self.take_captured_correction_h_as_f32()
                            } else {
                                None
                            };
                            let correction_ctx = match (&correction_table, &h_buf) {
                                (Some(tbl), Some(h)) => Some(LogitCorrectionCtx {
                                    table: tbl,
                                    hidden_f32: h,
                                    softcap: 30.0,
                                }),
                                _ => None,
                            };
                            // Sample token N. `sample_next_token_with_eos_guard_and_grammar`
                            // pulls last-position logits to host (forces eval),
                            // so the forward issued in the *previous* iteration
                            // (already async_eval'd) has completed by the time
                            // this returns.
                            let sampled = sample_next_token_with_eos_guard_and_grammar(
                                &step_logits,
                                &all_tokens,
                                &sampling,
                                &mut rng,
                                min_tokens_before_eos,
                                eos_top_k_guard,
                                eos_min_logit_margin,
                                &eos,
                                grammar.as_mut(),
                                correction_ctx.as_ref(),
                            )
                            .context("chat_streaming(sampled): sample step")?;
                            thinking_budget.observe(sampled);
                            let next_tok = if let Some(forced) = thinking_budget.try_force_close() {
                                eprintln!(
                                    "[thinking-budget] forcing channel close at {} tokens (count={})",
                                    all_tokens.len(),
                                    thinking_budget.max_thinking_tokens,
                                );
                                forced
                            } else if thinking_budget.should_block_channel_open()
                                && sampled == crate::gemma4_thinking::TOK_CHANNEL_OPEN
                            {
                                eprintln!(
                                    "[thinking-budget] blocking channel re-open at {} tokens; emitting <turn|>",
                                    all_tokens.len()
                                );
                                crate::gemma4_thinking::TOK_TURN_CLOSE
                            } else {
                                sampled
                            };
                            all_tokens.push(next_tok);
                            current_u32 = next_tok;

                            // Stop conditions are evaluated synchronously from
                            // `all_tokens`; if we stop, no further forward is
                            // issued.
                            let stop_eos = eos.contains(&next_tok);
                            let stop_runaway = runaway.check(&all_tokens);
                            let stop_hard_break = thinking_budget.should_hard_break();
                            let stopping = stop_eos || stop_runaway.is_some() || stop_hard_break;
                            let at_budget = all_tokens.len() >= max_new_tokens;

                            if overlap_enabled && !stopping && !at_budget {
                                // OVERLAP PATH:
                                //   1. issue forward(N+1) (the ONLY forward for
                                //      token N+1) + async_eval to kick the GPU;
                                //   2. flush the deferred emit of token N-1
                                //      (parser advance + detok + SSE) while the
                                //      GPU computes the N+1 logits;
                                //   3. stash token N for next iteration's flush.
                                step_logits = issue_forward(&self.model, current_u32, &mut cache)?;
                                mlx_rs::transforms::async_eval([&step_logits])
                                    .context("chat_streaming(sampled): overlap async_eval")?;

                                // CPU work overlaps the in-flight GPU forward.
                                flush_pending!();
                                let state_before = parser.state();
                                pending = Some((next_tok, state_before));
                            } else {
                                // SYNCHRONOUS PATH (overlap off, stopping, or at
                                // budget): flush any deferred token first to
                                // preserve order, emit token N inline, then —
                                // only if we will iterate again — issue the next
                                // forward synchronously.
                                flush_pending!();
                                let state_before = parser.state();
                                parser.push(next_tok)?;
                                emit_token_event(
                                    &self.chat,
                                    &mut parser,
                                    next_tok,
                                    state_before,
                                    &mut on_event,
                                )?;
                                if !stopping && !at_budget {
                                    step_logits =
                                        issue_forward(&self.model, current_u32, &mut cache)?;
                                }
                            }

                            if stop_eos {
                                break;
                            }
                            if let Some(reason) = stop_runaway {
                                eprintln!(
                                    "[runaway] chat_streaming sampled decode aborted at {} tokens: {reason}",
                                    all_tokens.len()
                                );
                                break;
                            }
                            if stop_hard_break {
                                eprintln!(
                                    "[thinking-budget] hard break at {} tokens — force-close did not help",
                                    all_tokens.len()
                                );
                                break;
                            }
                        }
                        // Flush the final deferred token before finalize so the
                        // streamed text is complete and in order.
                        flush_pending!();
                    }
                    let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
                    let count = all_tokens.len();
                    log_chat_done(prefill_prompt_tokens, prefill_total_ms, count, decode_ms);
                    return parser.finalize();
                }

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
                let t_decode = std::time::Instant::now();
                loop {
                    if count + 1 == max_new_tokens {
                        // Last token: read sync, emit, done.
                        let token = self
                            .model
                            .read_token_u32(&current)
                            .context("chat_streaming: read final token")?;
                        let state_before = parser.state();
                        parser.push(token)?;
                        emit_token_event(
                            &self.chat,
                            &mut parser,
                            token,
                            state_before,
                            &mut on_event,
                        )?;
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

                    let state_before = parser.state();
                    parser.push(token)?;
                    count += 1;
                    emit_token_event(&self.chat, &mut parser, token, state_before, &mut on_event)?;

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
                let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
                log_chat_done(prefill_prompt_tokens, prefill_total_ms, count, decode_ms);

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
                        entries.sort_by_key(|e| std::cmp::Reverse(e.1));
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
            let resp = backend
                .chat(
                    &msgs,
                    8,
                    0.0,
                    1.0,
                    &crate::SamplingOverrides::default(),
                    false,
                    &[],
                    &crate::chat_io::ResolvedToolChoice::Auto,
                )
                .expect("chat");
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
                .chat_streaming(
                    &msgs,
                    8,
                    0.0,
                    1.0,
                    &crate::SamplingOverrides::default(),
                    false,
                    &[],
                    &crate::chat_io::ResolvedToolChoice::Auto,
                    None,
                    |ev| {
                        if let crate::chat_io::BackendStreamEvent::Text(t) = ev {
                            chunks.push(t.to_string());
                        }
                        Ok(())
                    },
                )
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
