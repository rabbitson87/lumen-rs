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
    use crate::jinja_chat::imp::{JinjaChatTemplate, JinjaRenderOptions};
    use crate::gemma4_critical_correction::CorrectionTable;
    use crate::grammar::{Gemma4GrammarState, GrammarMode, shared_factory_from_tokenizer};
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

    /// Builds a `SamplingConfig` from request-supplied `temperature`/`top_p`
    /// plus operator-supplied env (`REPEAT_PENALTY`, `LUMEN_REPEAT_LAST_N`,
    /// `LUMEN_SAMPLE_SEED`). Returns `None` when the result is greedy
    /// (temperature ≤ 0 AND repeat_penalty == 1) so the decode loop can
    /// take the existing fast path.
    ///
    /// Defaults reflect the OpenAI request defaults (`0.7` / `0.9`) when
    /// the request didn't set them; `REPEAT_PENALTY=1.0` (no penalty)
    /// preserves existing behavior. Operators flip `REPEAT_PENALTY` to
    /// ~1.1 in the app's SERVER card to suppress degenerate loops on
    /// aggressively quantized models (3-bit MXFP4 etc.).
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
    }

    fn build_sampling_config(temperature: f32, top_p: f32) -> Option<SamplingConfig> {
        let repeat_penalty: f32 = std::env::var("REPEAT_PENALTY")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|v: &f32| *v > 0.0)
            .unwrap_or(1.0);
        let repeat_penalty_last_n: usize = std::env::var("LUMEN_REPEAT_LAST_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(64);
        let seed: u64 = std::env::var("LUMEN_SAMPLE_SEED")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| {
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(0x9E3779B97F4A7C15)
            });
        let cfg = SamplingConfig {
            temperature,
            top_p,
            repeat_penalty,
            repeat_penalty_last_n,
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
    /// Per-token stderr trace. Enabled when `LUMEN_GEMMA4_TOKEN_TRACE` is
    /// set to a non-empty / non-`0` / non-`false` value. Prints one line
    /// per sampled token with id + decoded text (special tokens visible)
    /// + parser state transition. Use for debugging reasoning runaway,
    /// tool-call structure, or channel close failures.
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
            eprintln!(
                "[token-trace] id={token:>6} {state_before:?}→{state_after:?} text={text:?}"
            );
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
            if !body_chunk.is_empty() {
                if let Some(name) = parser.observe_tool_text_fragment(&body_chunk) {
                    on_event(BackendStreamEvent::ToolCallStart {
                        name: name.as_str(),
                    })?;
                }
            }
        }
        Ok(())
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
    /// Opt-in flag for the Lark-format Gemma 4 tool-call grammar. Default
    /// OFF. When set (`1` / truthy), [`Gemma4Backend::build_grammar_state`]
    /// constructs an active matcher that constrains tool-call bodies to
    /// the native `call:NAME{key:value,…}` format with per-tool schema
    /// enforcement. Cached on first read.
    fn gemma4_grammar_lark_enabled() -> bool {
        static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *CACHED.get_or_init(|| {
            std::env::var("LUMEN_GEMMA4_GRAMMAR_LARK")
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

    pub struct Gemma4Backend {
        model: NativeGemma4Model,
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
            Ok(Self {
                model,
                chat,
                jinja_chat,
                model_id: model_id.into(),
                prefix_caches: HashMap::new(),
                model_dir: dir.to_path_buf(),
                grammar_factory: OnceLock::new(),
                correction_table: OnceLock::new(),
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
                .get_or_init(|| {
                    match CorrectionTable::load_from_model_dir(&self.model_dir) {
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
                    }
                })
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
            // Always Lazy (never Eager) regardless of `tool_choice`.
            // Reasoning: empirically the Eager + agentic-loop combination
            // sends the model into an n-gram cycle on the 3rd-or-later
            // turn — the grammar over-permits duplicate fields
            // (`tc_field ("," tc_field)*`) so the model can't commit to
            // closing the body. Lazy mode dodges this entirely: grammar
            // only activates AFTER the model self-emits `<|tool_call>`
            // (id 48), which means
            //   - `Required` / `Tool(_)` runs use the chat-template
            //     prefill to inject id 48 — prefill goes through
            //     `parser.push()`, NOT through `state.observe()`, so the
            //     Lazy matcher stays inactive and the model emits the
            //     body freely from training distribution (the pre-Lark
            //     behaviour, proven to work end-to-end).
            //   - `Auto` runs still get the schema safety net when the
            //     model decides on its own to emit `<|tool_call>` —
            //     observe(48) flips the matcher on and the args body is
            //     schema-constrained.
            // Future Eager re-enable should pair with a permutation-
            // ordered, no-duplicate body grammar (see
            // `build_tool_grammar_lark` doc).
            if matches!(choice, ResolvedToolChoice::None) {
                eprintln!("[gemma4-backend] Lark grammar skipped: tool_choice=None");
                return None;
            }
            let mode = GrammarMode::Lazy;
            let Some(factory) = self.grammar_factory() else {
                eprintln!(
                    "[gemma4-backend] Lark grammar skipped: factory unavailable \
                     (imatrix-AWQ family or tokenizer.json load error)"
                );
                return None;
            };
            let tools_json: Vec<serde_json::Value> = tools
                .iter()
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
            match Gemma4GrammarState::new_lark(factory, &tools_json, mode) {
                Ok(s) => {
                    eprintln!(
                        "[gemma4-backend] Lark grammar active for {} tool(s) (mode={mode:?})",
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

        pub fn model_id(&self) -> &str {
            &self.model_id
        }

        pub fn model(&self) -> &NativeGemma4Model {
            &self.model
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
            self.build_chat_input_with_tools(messages, thinking, &[])
        }

        /// Structured-history variant — accepts `ChatTurn`s carrying
        /// `tool_calls` / tool result data the `(role, content)` shape
        /// can't represent. Used by the turn-2 continuation path
        /// (`chat_with_history`).
        pub fn build_chat_input_from_history(
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
        ) -> Result<Vec<u32>> {
            let cfg = GenerateConfig {
                max_new_tokens,
                stop_on_eos: true,
                sampling: build_sampling_config(temperature, top_p),
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
            thinking: bool,
            tools: &[crate::chat_io::ToolDef<'_>],
            tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
        ) -> Result<ParsedResponse> {
            let (prompt, prefill_tokens) =
                self.build_prompt_and_prefill_from_history(turns, thinking, tools, tool_choice)?;
            let cfg = GenerateConfig {
                max_new_tokens,
                stop_on_eos: true,
                sampling: build_sampling_config(temperature, top_p),
            };
            let stats = self.model.generate(&prompt, &cfg)?;
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
            thinking: bool,
            tools: &[crate::gemma4_tools::imp::ToolDef<'_>],
            tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
        ) -> Result<ParsedResponse> {
            let (prompt, prefill_tokens) =
                self.build_prompt_and_prefill(messages, thinking, tools, tool_choice)?;
            let cfg = GenerateConfig {
                max_new_tokens,
                stop_on_eos: true,
                sampling: build_sampling_config(temperature, top_p),
            };
            let stats = self.model.generate(&prompt, &cfg)?;
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
        ) -> Result<(Vec<u32>, Vec<u32>)> {
            use crate::chat_io::ResolvedToolChoice;
            let effective_tools: &[crate::gemma4_tools::imp::ToolDef<'_>] = match tool_choice {
                ResolvedToolChoice::None => &[],
                _ => tools,
            };
            let mut prompt =
                self.build_chat_input_with_tools(messages, thinking, effective_tools)?;
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
        ) -> Result<(Vec<u32>, Vec<u32>)> {
            use crate::chat_io::ResolvedToolChoice;
            let effective_tools: &[crate::chat_io::ToolDef<'_>] = match tool_choice {
                ResolvedToolChoice::None => &[],
                _ => tools,
            };
            let mut prompt =
                self.build_chat_input_from_history(turns, thinking, effective_tools)?;
            let prefill = self.chat.tool_choice_prefill_tokens(tool_choice)?;
            prompt.extend(prefill.iter().copied());
            Ok((prompt, prefill))
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
            thinking: bool,
            prefix_cache_key: &str,
            tools: &[crate::gemma4_tools::imp::ToolDef<'_>],
            tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
        ) -> Result<ParsedResponse> {
            let (prompt, prefill_tokens) =
                self.build_prompt_and_prefill(messages, thinking, tools, tool_choice)?;
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
                        (self.make_cache_for_prompt_len(prompt.len()), "miss-no-overlap", 0)
                    }
                }
                None => (self.make_cache_for_prompt_len(prompt.len()), "miss-no-entry", 0),
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
                sampling: build_sampling_config(temperature, top_p),
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

            // ── Snapshot post-prompt state as the new master ──
            // After generate, `cache.offset()` is well past `prompt.len()`
            // (advanced by decoded tokens). Truncate back to prompt.len()
            // so the master snapshot represents only the prompt prefix —
            // this is what subsequent requests with the same system prompt
            // want to fork from.
            //
            // NOTE: this non-streaming path uses `generate_with_cache`
            // (bundled prefill+decode), so we can't snapshot pre-decode
            // like the streaming path does. truncate_to MAY fail for
            // rotating sliding caches when prompt > sliding_window (KV
            // for early positions has been overwritten by post-rotation
            // ring writes). Treat the failure as non-fatal: skip the
            // cache write and let the next request cold-prefill.
            // Long-prompt callers should prefer the streaming path which
            // snapshots pre-decode and avoids this failure mode entirely.
            let target_offset = prompt.len();
            let mut master_snapshot = cache.clone();
            let truncate_ok = if master_snapshot.offset() > target_offset {
                match master_snapshot.truncate_to(target_offset) {
                    Ok(()) => true,
                    Err(e) => {
                        eprintln!(
                            "[gemma4-backend] chat_with_prefix_cache snapshot skipped \
                             (truncate failed, key={prefix_cache_key:?}): {e:#}"
                        );
                        false
                    }
                }
            } else {
                true
            };
            if truncate_ok {
                self.prefix_caches.insert(
                    prefix_cache_key.to_string(),
                    Gemma4PrefixCacheEntry {
                        master: master_snapshot,
                        prefix_tokens: prompt.clone(),
                        last_access: Instant::now(),
                        hits: 0,
                    },
                );
            }

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

        /// Lookup-or-make helper shared by all `*_with_prefix_cache` entry
        /// points. Returns the cache to feed to the chunked prefill (already
        /// truncated to the longest common prefix with `prompt`), the human-
        /// readable hit kind for logging, and the LCP length.
        ///
        /// Edge case: an exact match (LCP == prompt.len()) would leave an
        /// empty suffix and skip the prefill loop entirely, but the decode
        /// loop still needs fresh next-token logits to start. We clamp
        /// `lcp <= prompt.len() - 1` so there's always at least one suffix
        /// token to forward through the model.
        fn lookup_or_make_cache_for_prompt(
            &mut self,
            prompt: &[u32],
            key: &str,
        ) -> (NativeGemma4PromptCache, &'static str, usize) {
            let prompt_max = prompt.len().saturating_sub(1);
            match self.prefix_caches.get_mut(key) {
                Some(entry) => {
                    let raw_lcp = entry
                        .prefix_tokens
                        .iter()
                        .zip(prompt.iter())
                        .take_while(|(a, b)| a == b)
                        .count();
                    let lcp = raw_lcp.min(prompt_max);
                    if lcp > 0 && lcp <= entry.prefix_tokens.len() {
                        entry.last_access = Instant::now();
                        entry.hits += 1;
                        let mut cache = entry.master.clone();
                        if lcp < cache.offset() {
                            // Truncate failures are recoverable: drop the
                            // cache and fall through to a cold start so the
                            // request still succeeds.
                            if let Err(e) = cache.truncate_to(lcp) {
                                eprintln!(
                                    "[gemma4-backend] prefix-cache truncate failed (lcp={lcp}): {e:#}, falling back to cold prefill"
                                );
                                return (self.make_cache_for_prompt_len(prompt.len()), "miss-truncate-fail", 0);
                            }
                        }
                        (cache, "hit", lcp)
                    } else {
                        (self.make_cache_for_prompt_len(prompt.len()), "miss-no-overlap", 0)
                    }
                }
                None => (self.make_cache_for_prompt_len(prompt.len()), "miss-no-entry", 0),
            }
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
            thinking: bool,
            prefix_cache_key: &str,
            tools: &[crate::gemma4_tools::imp::ToolDef<'_>],
            tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
            on_event: impl FnMut(BackendStreamEvent<'_>) -> Result<()>,
        ) -> Result<ParsedResponse> {
            let (prompt, prefill_tokens) =
                self.build_prompt_and_prefill(messages, thinking, tools, tool_choice)?;
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
            let (cache, hit_kind, lcp) =
                self.lookup_or_make_cache_for_prompt(&prompt, prefix_cache_key);
            let suffix_len = prompt.len().saturating_sub(cache.offset());
            eprintln!(
                "[gemma4-backend] prefix-cache key={prefix_cache_key:?} \
                 result={hit_kind} lcp={lcp} suffix_len={suffix_len} header_tail={trailing_header_len}"
            );
            let grammar = self.build_grammar_state(tools, tool_choice);
            self.decode_streaming_with_prompt(
                prompt,
                prefill_tokens,
                max_new_tokens,
                temperature,
                top_p,
                grammar,
                Some(cache),
                Some((prefix_cache_key.to_string(), trailing_header_len)),
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
            thinking: bool,
            prefix_cache_key: &str,
            tools: &[crate::gemma4_tools::imp::ToolDef<'_>],
            tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
            on_event: impl FnMut(BackendStreamEvent<'_>) -> Result<()>,
        ) -> Result<ParsedResponse> {
            let (prompt, prefill_tokens) =
                self.build_prompt_and_prefill_from_history(turns, thinking, tools, tool_choice)?;
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
            let (cache, hit_kind, lcp) =
                self.lookup_or_make_cache_for_prompt(&prompt, prefix_cache_key);
            let suffix_len = prompt.len().saturating_sub(cache.offset());
            eprintln!(
                "[gemma4-backend] prefix-cache key={prefix_cache_key:?} \
                 result={hit_kind} lcp={lcp} suffix_len={suffix_len} header_tail={trailing_header_len} (from-history)"
            );
            let grammar = self.build_grammar_state(tools, tool_choice);
            self.decode_streaming_with_prompt(
                prompt,
                prefill_tokens,
                max_new_tokens,
                temperature,
                top_p,
                grammar,
                Some(cache),
                Some((prefix_cache_key.to_string(), trailing_header_len)),
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
            thinking: bool,
            prefix_cache_key: &str,
            tools: &[crate::chat_io::ToolDef<'_>],
            tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
        ) -> Result<ParsedResponse> {
            let (prompt, prefill_tokens) =
                self.build_prompt_and_prefill_from_history(turns, thinking, tools, tool_choice)?;
            if prompt.is_empty() {
                return Err(anyhow!(
                    "chat_from_history_with_prefix_cache: empty prompt"
                ));
            }
            let (mut cache, hit_kind, lcp) =
                self.lookup_or_make_cache_for_prompt(&prompt, prefix_cache_key);
            let suffix_len = prompt.len().saturating_sub(cache.offset());
            eprintln!(
                "[gemma4-backend] prefix-cache key={prefix_cache_key:?} \
                 result={hit_kind} lcp={lcp} suffix_len={suffix_len} (from-history non-streaming)"
            );

            let suffix = &prompt[cache.offset()..];
            let cfg = GenerateConfig {
                max_new_tokens,
                stop_on_eos: true,
                sampling: build_sampling_config(temperature, top_p),
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

            // Snapshot post-prompt master for subsequent forks.
            let target_offset = prompt.len();
            let mut master_snapshot = cache.clone();
            if master_snapshot.offset() > target_offset {
                if let Err(e) = master_snapshot.truncate_to(target_offset) {
                    eprintln!(
                        "[gemma4-backend] prefix-cache snapshot skipped (truncate failed): {e:#}"
                    );
                } else {
                    self.prefix_caches.insert(
                        prefix_cache_key.to_string(),
                        Gemma4PrefixCacheEntry {
                            master: master_snapshot,
                            prefix_tokens: prompt.clone(),
                            last_access: Instant::now(),
                            hits: 0,
                        },
                    );
                }
            } else {
                self.prefix_caches.insert(
                    prefix_cache_key.to_string(),
                    Gemma4PrefixCacheEntry {
                        master: master_snapshot,
                        prefix_tokens: prompt.clone(),
                        last_access: Instant::now(),
                        hits: 0,
                    },
                );
            }

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
            thinking: bool,
            tools: &[crate::gemma4_tools::imp::ToolDef<'_>],
            tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
            on_event: impl FnMut(BackendStreamEvent<'_>) -> Result<()>,
        ) -> Result<ParsedResponse> {
            let (prompt, prefill_tokens) =
                self.build_prompt_and_prefill(messages, thinking, tools, tool_choice)?;
            let grammar = self.build_grammar_state(tools, tool_choice);
            self.decode_streaming_with_prompt(
                prompt,
                prefill_tokens,
                max_new_tokens,
                temperature,
                top_p,
                grammar,
                None,
                None,
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
            thinking: bool,
            tools: &[crate::gemma4_tools::imp::ToolDef<'_>],
            tool_choice: &crate::chat_io::ResolvedToolChoice<'_>,
            on_event: impl FnMut(BackendStreamEvent<'_>) -> Result<()>,
        ) -> Result<ParsedResponse> {
            let (prompt, prefill_tokens) =
                self.build_prompt_and_prefill_from_history(turns, thinking, tools, tool_choice)?;
            let grammar = self.build_grammar_state(tools, tool_choice);
            self.decode_streaming_with_prompt(
                prompt,
                prefill_tokens,
                max_new_tokens,
                temperature,
                top_p,
                grammar,
                None,
                None,
                on_event,
            )
        }

        fn decode_streaming_with_prompt(
            &mut self,
            prompt: Vec<u32>,
            prefill_tokens: Vec<u32>,
            max_new_tokens: usize,
            temperature: f32,
            top_p: f32,
            // Phase 2.5: grammar-constrained tool calling. `None` for the
            // default text path; `Some(state)` engages the llguidance mask
            // during the sampled decode branch (no-op on greedy / MTP — those
            // paths don't sample multinomially and would need different
            // wiring). The state is observed every sampled step so the
            // lazy trigger fires on `<|tool_call>` even when grammar
            // started inactive.
            grammar: Option<Gemma4GrammarState>,
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
                let mut cache = pre_built_cache.unwrap_or_else(|| {
                    self.make_cache_for_prompt_len(prompt_len)
                });

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
                            format!(" [split at {} for prefix-cache snapshot, trailing {} tokens]", s, prompt.len() - s)
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
                    let trailing_logits = self
                        .model
                        .forward_last_token(trailing, &mut cache)
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
                let eos = self.model.eos_tokens().to_vec();

                let sampling_cfg = build_sampling_config(temperature, top_p);

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
                    // Both default 0 (off) — bit-identical to no-guard
                    // sampling when unset.
                    let min_tokens_before_eos: usize =
                        std::env::var("LUMEN_MIN_TOKENS_BEFORE_EOS")
                            .ok()
                            .and_then(|s| s.trim().parse::<usize>().ok())
                            .unwrap_or(0);
                    let eos_top_k_guard: usize = std::env::var("LUMEN_EOS_TOP_K_GUARD")
                        .ok()
                        .and_then(|s| s.trim().parse::<usize>().ok())
                        .unwrap_or(0);
                    let eos_min_logit_margin: f32 =
                        std::env::var("LUMEN_EOS_MIN_LOGIT_MARGIN")
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
                        let mut thinking_budget =
                            crate::gemma4_thinking::ChannelBudget::from_env();
                        thinking_budget.observe(first_tok);
                        let mut current_u32 = first_tok;
                        while all_tokens.len() < max_new_tokens {
                            let input = mlx_rs::Array::from_slice(&[current_u32 as i32], &[1, 1])
                                .as_dtype(mlx_rs::Dtype::Int32)
                                .context("chat_streaming(sampled): build input array")?;
                            let step_logits = self
                                .model
                                .forward_array_last_token(&input, &mut cache)
                                .context("chat_streaming(sampled): decode forward")?;
                            // Phase B: build per-step correction context
                            // from the just-captured `h_for_lm_head`. h
                            // ownership transfers via take_*, so a forward
                            // without a paired sample-step would leak the
                            // captured clone (acceptable — never happens
                            // in the streaming loop).
                            let h_buf =
                                if correction_table.is_some() {
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
                            let next_tok = if let Some(forced) =
                                thinking_budget.try_force_close()
                            {
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
                            let state_before = parser.state();
                            parser.push(next_tok)?;
                            emit_token_event(
                                &self.chat,
                                &mut parser,
                                next_tok,
                                state_before,
                                &mut on_event,
                            )?;
                            if eos.contains(&next_tok) {
                                break;
                            }
                            if let Some(reason) = runaway.check(&all_tokens) {
                                eprintln!(
                                    "[runaway] chat_streaming sampled decode aborted at {} tokens: {reason}",
                                    all_tokens.len()
                                );
                                break;
                            }
                            if thinking_budget.should_hard_break() {
                                eprintln!(
                                    "[thinking-budget] hard break at {} tokens — force-close did not help",
                                    all_tokens.len()
                                );
                                break;
                            }
                            current_u32 = next_tok;
                        }
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
            let resp = backend
                .chat(
                    &msgs,
                    8,
                    0.0,
                    1.0,
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
                    false,
                    &[],
                    &crate::chat_io::ResolvedToolChoice::Auto,
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
