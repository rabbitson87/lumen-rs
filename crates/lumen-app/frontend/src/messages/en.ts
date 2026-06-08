/**
 * English message catalogue. Keys are dot-namespaced by surface
 * (`tabs.*`, `header.*`, `cards.<card>.*`, `language.*`).
 *
 * Coverage policy: top-level navigation, card headings, action buttons,
 * status banners, and the language picker itself are fully translated.
 * Long-form descriptive hints (cache quantization explainers, etc.) remain in
 * the templates and can be migrated incrementally as needed.
 */
export const en: Record<string, string> = {
  // ── Tab bar ─────────────────────────────────────────────────────
  "tabs.main": "Models & Server",
  "tabs.tuning": "Tuning",
  "tabs.api": "API",
  "tabs.debug": "Debug",
  "tabs.language": "Language",

  // ── Header ──────────────────────────────────────────────────────
  "header.start": "Start",
  "header.stop": "Stop",
  "header.starting": "Starting…",
  "header.stopping": "Stopping…",
  "header.logs": "Logs",
  "header.env": "Env",
  "header.doctor": "Doctor",
  "header.update": "Update",
  "header.title.brokenActive":
    "Active model's download is incomplete. Re-download it first (MODELS card → Re-download).",
  "header.title.outdatedActive":
    "Active model has a newer version on Hub. Update it first (MODELS card → Update).",

  // ── Status indicators ───────────────────────────────────────────
  "status.stopped": "stopped",
  "status.starting": "starting",
  "status.running": "running",
  "status.stopping": "stopping",
  "status.crashed": "crashed",

  // ── Common actions ──────────────────────────────────────────────
  "action.download": "Download",
  "action.delete": "Delete",
  "action.use": "Use",
  "action.update": "Update",
  "action.redownload": "Re-download",
  "action.downloading": "Downloading…",
  "action.reset": "Reset",
  "action.cancel": "Cancel",
  "action.confirm": "Confirm",
  "action.close": "Close",
  "action.openConfig": "Open config folder",

  // ── MODELS card ─────────────────────────────────────────────────
  "models.title": "MODELS",
  "models.empty.unsupported":
    "No supported models on disk. Download one from the curated list below.",
  "models.empty.none": "No local models. Download one from the curated list below.",
  "models.thisMac": "This Mac:",
  "models.thisMac.ramSuffix": "GB RAM",
  "models.thisMac.overflow": "— models over this size are marked.",
  "models.picker.placeholder": "— pick a recommended model —",
  "models.picker.allDownloaded": "all recommended already downloaded",
  "models.broken.label": "⚠ Incomplete download — re-download required before use",
  "models.downloading.label": "⬇ Downloading…",
  "models.outdated.label": "⚠ Newer weights available on Hub — update required before use",
  "models.unsupported.label": "not in supported catalog",

  // ── SERVER card ─────────────────────────────────────────────────
  "server.title": "SERVER",
  "server.cors": "CORS",
  "server.cors.off": "off (specific IP)",
  "server.cors.localhost": "localhost (127.0.0.1)",
  "server.cors.all": "all / 0.0.0.0 (risky)",
  "server.host": "Host",
  "server.port": "Port",
  "server.apiKey": "API key",
  "server.apiKey.hint": "→ set in the API card",
  "server.memory.title": "Metal memory",
  "server.memory.titleHint": "(mlx-native)",
  "server.memory.tunedFor": "tuned for",
  "server.memory.systemDefault": "system default for",
  "server.memory.wired": "Wired GB",
  "server.memory.cache": "Cache GB",
  "server.memory.memory": "Memory GB",
  "server.memory.explainer": "What do these mean?",

  // ── METRICS card ────────────────────────────────────────────────
  "metrics.title": "METRICS",
  "metrics.tokensPerSec": "tok/s",
  "metrics.msPerStep": "ms / step",
  "metrics.kvCache": "KV cache",
  "metrics.requestsPerMin": "req/min",

  // ── CONTEXT card ────────────────────────────────────────────────
  "context.title": "CONTEXT",
  "context.titleHint": "(driven by QUANT state)",
  "context.max": "Max",
  "context.sliding": "Sliding",
  "context.prefill": "Prefill",
  "context.defaultMaxTokens": "Default max_tokens",
  "context.kvQuant.label": "Cache mode:",
  "context.kvQuant.offHint": "· baseline KV memory (no compression)",
  "context.recommended": "Recommended max on this Mac",
  "context.recommended.suffix": "tokens",
  "context.warn.turnOnKvQuant":
    "— turn cache quantization ON to handle longer contexts safely",

  // ── CACHE / KV-quant (Tuning tab) ───────────────────────────────
  "quant.title": "CACHE",
  "quant.titleHint": "(KV cache quantization)",
  "quant.mode": "Cache mode",
  "quant.mode.off": "Off",
  "quant.mode.on": "On",
  "quant.mode.auto": "Auto",
  "quant.autoThreshold": "Auto threshold (tokens)",
  "quant.bits": "Bits",
  "quant.on": "ON",
  "quant.off": "OFF",

  // ── DEBUG card ──────────────────────────────────────────────────
  "debug.title": "DEBUG",
  "debug.titleHint": "(emergency-check switches)",
  "debug.intro":
    "Escape hatches for when normal operation fails. Leave everything blank/off for regular use — the values that actually matter (model, memory caps, backend) live in the Models & Server tab.",
  "debug.memoryBypass": "Memory bypass",
  "debug.memoryBypass.label": "Bypass all caps",
  "debug.memoryBypass.hint": "skip wired+cache+memory; let MLX/macOS manage",
  "debug.loader": "Loader overrides",
  "debug.tokenizer": "Tokenizer",
  "debug.tokenizer.placeholder": "HF repo id (override)",
  "debug.weightsDir": "Weights dir",
  "debug.weightsDir.placeholder": "auto-set from active model",
  "debug.skipWarmup": "Skip warmup",
  "debug.skipWarmup.hint": "faster start, first request slower",

  // ── LANGUAGE tab ────────────────────────────────────────────────
  "language.title": "LANGUAGE",
  "language.choose": "Interface language",
  "language.note":
    "Language preference is stored locally in your browser and applies immediately — no restart needed.",

  // ── Confirm modal ───────────────────────────────────────────────
  "confirm.delete.chat.title": "Delete model?",
  "confirm.delete.embedding.title": "Delete embedding model?",
  "confirm.delete.warning":
    "Weights will be removed from disk. You can re-download from the catalog later.",
  "confirm.delete.embedding.activeNote":
    " The active embedding will be cleared.",
  "confirm.delete.busy": "Deleting…",

  // ── Footer tabs (bottom dock) ───────────────────────────────────
  "footer.logs": "Logs",
  "footer.env": "Env overrides",
  "footer.doctor": "Doctor",
  "footer.update": "Update",
  "footer.logs.empty":
    "No log output yet. Start the server to see decode/encode traces.",

  // ── CACHE / KV-quant tooltips ───────────────────────────────────
  "quant.tooltip.mode":
    "Off: never compress KV (fastest decode at any context, largest memory). On: always compress (4–5× KV memory savings, decode ≈ same ±5%). Auto: compress only when this request's prompt is at or above the threshold below — short chats stay at full speed, only long-context requests pay the quant trade-off. Per-request decision logged as `[gemma4-backend] quant_kv_auto: ...`.",
  "quant.tooltip.autoThreshold":
    "Prompt-token count at which Auto mode flips KV quantization ON for the request. Default 16384 (16K) — tuned for the 24 GB Mac mini, where bf16 KV pressure starts to bind and quantized sliding-window wins are verified from ~16K up; below it full-speed bf16 decode wins. Big-memory machines (64 GB+) can raise this or use Off. (A 128K default made Auto a near no-op.)",
  "quant.tooltip.bits":
    "Quantization bits per KV channel. 8: highest quality, 2× smaller than bf16. 6: balanced ≈ 2.7× smaller. 4: recommended default, 4× smaller. 3: max compression ≈ 5.3×, small quality drop. Uses mlx affine quantization (group_size=64) — no rotation or residual correction stage.",

  // ── CONTEXT card hints ──────────────────────────────────────────
  "context.hint.max.prefix":
    "Max sequence length (tokens). Caps the model's max_position_embeddings when host RAM can't hold the model's native limit (Gemma 4 claims 128K).",
  "context.hint.max.kvOn":
    "Current cache quantization gives roughly the listed KV compression",
  "context.hint.max.kvOnRealistic": "— realistic on this Mac:",
  "context.hint.max.kvOff":
    "Cache quantization OFF — KV stays bf16, so practical limit on this Mac",
  "context.hint.max.kvOffFallback": "is much lower than the model's native max",
  "context.hint.max.env": "Env:",
  "context.hint.sliding":
    "Sliding-window attention size. Some layers (Gemma 4: 25 of 30) only attend to the last N tokens instead of the full sequence → bounded KV memory for long contexts. 0 = use the model's built-in default; N>0 overrides it (smaller = less KV, weaker long-range recall).",
  "context.hint.sliding.kvStacks":
    "Stacks with cache quantization — sliding bounds which tokens are kept, quantization controls how they're stored.",
  "context.hint.prefill":
    "Prompt-processing chunk cap. Server rejects prompts longer than this with a \"prompt too large\" error. Larger = accepts long prompts but more peak memory during prefill (attention QK·T = chunk × KV",
  "context.hint.defaultMaxTokens":
    "Generation budget applied when an OpenAI-compatible chat/completion request omits `max_tokens` + ceiling for clients that send an explicit `max_tokens`. E.g. set to 8192 and a client sending 204800 will be capped to 8192. `0` disables the cap (unbounded until EOS / context — beware of runaway CoT). Emitted as both `LUMEN_DEFAULT_MAX_TOKENS` and `LUMEN_MAX_TOKENS_CAP`.",

  // ── SERVER memory explainer ─────────────────────────────────────
  "server.memory.explainer.intro":
    "Apple Silicon shares one pool of RAM between CPU and GPU. These three caps tell MLX how much of that pool it may use:",
  "server.memory.explainer.wired":
    "Wired GB — RAM that stays pinned for the GPU and can never be paged out. Auto-set to the exact safetensors byte size of the active model (via LUMEN_WIRED_LIMIT_BYTES), so a 14.45 GB model isn't truncated to a 14 GB ceiling. Override the input if you want extra headroom for KV cache.",
  "server.memory.explainer.cache":
    "Cache GB — MLX's transient buffer reuse pool (activations, scratch). A small fixed budget (2 GB) is enough; scaling it with system RAM just reserves memory you'd rather give back to the OS.",
  "server.memory.explainer.memory":
    "Memory GB — Soft total ceiling for Metal allocations. Hitting it triggers cache eviction before the hard wired limit. Set to model size + 2 GB + KV cache budget (≈ ctx ÷ 8K).",
  "server.memory.wired.titleHint":
    "LUMEN_WIRED_LIMIT_BYTES — exact safetensors size",
  "server.host.title.pin": "Pin to a specific IP",
  "server.host.title.auto": "Auto-set by CORS scope",

  // ── MODELS card extras ──────────────────────────────────────────
  "models.action.title.redownload":
    "Verify and re-download missing or truncated files",
  "models.action.title.update": "Re-download with the latest Hub weights",
  "models.action.title.unsupported":
    "Not in the server-side supported catalog",

  // ── Header memory bar ───────────────────────────────────────────
  "header.memory.title":
    "System memory — wired + active + compressor",
  "header.statusError": "· error",
  "header.doctor.title.idle": "Run preflight checks",

  // ── API tab (ApiTabs.svelte) ────────────────────────────────────
  "api.title": "API",
  "api.style.openai": "OpenAI-style",
  "api.style.claude": "Claude-style",
  "api.serverNotRunning":
    "server is {state} — values shown are what clients will use once you Start",
  "api.baseUrl": "Base URL",
  "api.apiKey": "API key",
  "api.apiKey.placeholder": "(none — auth disabled)",
  "api.copy": "Copy",
  "api.copied": "copied",
  "api.copyFailed": "copy failed:",
  "api.endpoints": "Endpoints",
  "api.curlExample": "curl example",
  "api.anthropicVersion": "anthropic-version",
  "api.embedding.title": "Embedding model",
  "api.embedding.empty":
    "No embedding models downloaded yet — pick one below to download first.",
  "api.embedding.disable": "Disable embedding",
  "api.embedding.disable.title":
    "Stop using any embedding model (disables /v1/embeddings)",
  "api.embedding.activeMissing.prefix": "Active embedding",
  "api.embedding.activeMissing.suffix": "is not in the local catalog.",
  "api.embedding.download.title": "Download embedding",
  "api.embedding.download.placeholder": "— pick to download —",
  "api.embedding.endpointRequiresEmbedding": "(requires Embedding model)",

  // ── CACHE bits comparison hints ─────────────────────────────────
  "quant.hint.kvOff":
    "Cache quantization OFF — KV stays bf16 (max speed, max memory)",
  "quant.hint.smallerVsFp16": "× smaller vs bf16",
  "quant.hint.lowestMemory": "lowest memory, small quality drop",
  "quant.hint.balancedQuality": "balanced memory / quality",
  "quant.hint.highestQuality": "closest to bf16 quality",
  "quant.hint.baseline": "recommended default (4× KV savings)",

  // ── CONTEXT banner ──────────────────────────────────────────────
  "context.banner.smallerThanBf16": "× smaller than bf16",
  "context.banner.kvCache": "KV cache ~",

  // ── MODELS card embedding mini-row ─────────────────────────────
  "models.embedding.label": "Embedding:",
  "models.embedding.disabled": "(disabled)",
  "models.embedding.autoFetched": "(server auto-fetched from cache)",
  "models.embedding.pickHint": "Pick / change in the API tab",

  // ── EnvOverrides panel ──────────────────────────────────────────
  "env.title": "Environment overrides",
  "env.intro":
    "Each row maps to one env var the server reads at startup. Strings only — booleans should be \"1\" or \"0\".",
  "env.placeholder.key": "key",
  "env.placeholder.value": "value",
  "env.empty": "No overrides set. Add a row below.",
  "env.add": "Add row",
  "env.save": "Save",
  "env.savedHint":
    "Saved. Restart the server to apply (Stop → Start).",
  "env.knownKeysTitle": "Typed keys (auto-completed):",
  "env.intro2":
    "Typed knobs passed to the lumen-server subprocess. Toggle, pick a value, or clear (×) to revert to default. Keys that shadow a typed UI field are highlighted.",
  "env.row.remove": "Remove",
  "env.row.shadowsPrefix": "⚠ shadows the UI field for",
  "env.add2": "+ Add",
  "env.revert": "Revert",
  "env.applyHint": "Changes apply on next server start.",
  "env.empty2": "No overrides set.",

  // ── Section groupings ────────────────────────────────────────────
  "env.section.thinking": "Reasoning",
  "env.section.sampling": "Sampling",
  "env.section.safety": "Safety net",
  "env.section.advanced": "Advanced",
  "env.section.debug": "Debug / triage",

  // ── Per-entry labels (key = env var name) ──────────────────────
  "env.entry.LUMEN_BACKEND_THINKING_DEFAULT.label": "Backend thinking default",
  "env.entry.LUMEN_BACKEND_THINKING_DEFAULT.help":
    "When ON, OpenAI-compat clients with no per-request thinking signal get thinking enabled by default. Per-request signals still override.",

  "env.entry.LUMEN_TEMPERATURE.label": "Temperature",
  "env.entry.LUMEN_TEMPERATURE.help":
    "Gemma 4 sampling temperature. Substituted when the client omits it (server default 0.7); an explicit request value is honored. Ollama gemma4 = 1.0. Gemma 4 only.",

  "env.entry.LUMEN_TOP_P.label": "Top-p (nucleus)",
  "env.entry.LUMEN_TOP_P.help":
    "Cumulative-probability cutoff. Applied when the client omits top_p. Ollama gemma4 = 0.95.",

  "env.entry.LUMEN_TOP_K.label": "Top-k",
  "env.entry.LUMEN_TOP_K.help":
    "Restrict candidates to the top k tokens. 0 disables. Ollama gemma4 = 64. Too low fails to escape repetition cycles.",

  "env.entry.REPEAT_PENALTY.label": "Repeat penalty",
  "env.entry.REPEAT_PENALTY.help":
    "Penalize recently emitted tokens. 1.0 = off. Ollama = 1.1.",

  "env.entry.LUMEN_DRY_MULTIPLIER.label": "DRY repetition suppression",
  "env.entry.LUMEN_DRY_MULTIPLIER.help":
    "DRY (Don't Repeat Yourself) strength. 0 = off. 0.8 recommended — directly suppresses mixed degenerate runaways like }}}}/~~~~ (multi-turn stability).",

  "env.entry.LUMEN_MAX_THINKING_TOKENS.label": "Max thinking tokens (hard cap)",
  "env.entry.LUMEN_MAX_THINKING_TOKENS.help":
    "Force-emit channel-close after N reasoning tokens. 0 disables. Recommended 600 for Gemma 4.",

  "env.entry.LUMEN_MAX_FORCE_CLOSE_ATTEMPTS.label": "Force-close attempts",
  "env.entry.LUMEN_MAX_FORCE_CLOSE_ATTEMPTS.help":
    "Number of times to force the channel-close token before terminating the turn. Default 1.",

  "env.entry.LUMEN_RUNAWAY_DETECT.label": "N-gram runaway detector",
  "env.entry.LUMEN_RUNAWAY_DETECT.help":
    "Automatically truncate the response when an n-gram cycle is detected. Default ON.",

  "env.entry.LUMEN_RUNAWAY_NGRAM.label": "Runaway n-gram size",
  "env.entry.LUMEN_RUNAWAY_NGRAM.help":
    "N-gram length the detector watches for cyclic repetition. Default 4.",

  "env.entry.LUMEN_RUNAWAY_NGRAM_MAX_REPEATS.label": "Runaway max repeats",
  "env.entry.LUMEN_RUNAWAY_NGRAM_MAX_REPEATS.help":
    "Tolerated repeats of the same n-gram before truncation. Default 8.",

  "env.entry.LUMEN_GEMMA4_CRITICAL_LOGIT_CORRECTION.label": "Phase B logit correction (Gemma 4)",
  "env.entry.LUMEN_GEMMA4_CRITICAL_LOGIT_CORRECTION.help":
    "Apply the sidecar Δ to 7 critical token ids (channel/turn/tool boundaries). Default ON.",

  "env.entry.LUMEN_GEMMA4_GRAMMAR_LARK.label": "Lark grammar (Gemma 4 tool calls)",
  "env.entry.LUMEN_GEMMA4_GRAMMAR_LARK.help":
    "Constrain tool-call output with a Lark grammar -- guarantees structurally valid call:NAME{...} bodies. Default ON; disable only if you need free-form tool-call emission.",

  "env.entry.LUMEN_TOOL_CHOICE_AUTO_AS_REQUIRED.label": "Promote tool_choice auto -> required",
  "env.entry.LUMEN_TOOL_CHOICE_AUTO_AS_REQUIRED.help":
    "When the client sends tool_choice=\"auto\", upgrade it to required so every turn forces a tool call. Turn ON for agentic loops (e.g. Ayla) where the model emits task_complete weakly and the loop repeats across multiple responses. The answer text rides in the tool's summary field. Default OFF.",
  "env.entry.LUMEN_GEMMA4_EMPTY_THOUGHT_ON_NOTHINK.label": "Inject empty thought channel on nothink",
  "env.entry.LUMEN_GEMMA4_EMPTY_THOUGHT_ON_NOTHINK.help":
    "Whether to pre-fill an empty <|channel>thought<channel|> block on the generation prompt when thinking is OFF. The jinja template injects it, but Ollama's native gemma4 renderer does not (emptyBlockOnNothink=false). Injecting it makes the quantized model bail mid-sentence with <turn|> and never call task_complete. Default OFF (= Ollama behavior, recommended). Turn ON to restore the old jinja-faithful behavior.",

  "env.entry.LUMEN_USE_JINJA_RENDERER.label": "Use minijinja renderer (Gemma 4)",
  "env.entry.LUMEN_USE_JINJA_RENDERER.help":
    "Render the chat template via minijinja against the model's chat_template.jinja, instead of the Rust hand-port. Byte-identical on golden vectors; flip ON to opt into the upstream-authoritative path.",

  "env.entry.LUMEN_DUMP_PROMPT.label": "Dump prompt",
  "env.entry.LUMEN_DUMP_PROMPT.help":
    "Print the chat-templated prompt sent to the model. off / preview / full.",

  "env.entry.LUMEN_LOG_REQUEST_BODY.label": "Log request body",
  "env.entry.LUMEN_LOG_REQUEST_BODY.help":
    "Print a one-line [diag] summary of each /v1/chat/completions request.",

  "env.entry.LUMEN_GEMMA4_TOKEN_TRACE.label": "Per-token trace (Gemma 4)",
  "env.entry.LUMEN_GEMMA4_TOKEN_TRACE.help":
    "Print one [token-trace] line per sampled token. High volume — debug only.",

  "env.entry.LUMEN_EOS_GUARD_VERBOSE.label": "EOS guard verbose log",
  "env.entry.LUMEN_EOS_GUARD_VERBOSE.help":
    "Log every EOS-guard suppression event in the sampling pipeline.",

  "env.entry.LUMEN_QWEN35_PREFILL_CHUNK.label": "Qwen prefill chunk size",
  "env.entry.LUMEN_QWEN35_PREFILL_CHUNK.help":
    "Tokens per prefill chunk for Qwen 3.6 long prompts. Larger = fewer GPU syncs (faster cold prefill) but more peak memory. Raise on Macs with more RAM; lower if a long prompt OOMs. Default 2048.",
  "env.entry.LUMEN_QWEN35_PREFILL_CHUNK_LOG.label": "Qwen prefill chunk log",
  "env.entry.LUMEN_QWEN35_PREFILL_CHUNK_LOG.help":
    "Print per-chunk prefill timing and peak Metal memory. Debug only.",
  "env.entry.LUMEN_NATIVE_TIMING.label": "Native stage timing",
  "env.entry.LUMEN_NATIVE_TIMING.help":
    "Log per-stage forward timing (embed / attention / linear-attn / MoE / lm_head ms) for the native MLX runner. Use to find prefill/decode bottlenecks. ⚠️ Profiling only — inserts GPU sync barriers per layer that break MLX pipelining and cut throughput ~8× (e.g. ~70→~8 tok/s). Turn OFF for normal use.",
  "env.entry.LUMEN_QWEN35_TOOL_DEBUG.label": "Qwen tool-call raw dump",
  "env.entry.LUMEN_QWEN35_TOOL_DEBUG.help":
    "Log the raw decoded model output for Qwen3.6 tool-call turns (full <tool_call>…</tool_call> text). Use to diagnose missing/dropped tool arguments. Verbose — debug only.",
  "env.entry.LUMEN_QWEN35_FORCE_REQUIRED_PARAMS.label": "Force required tool params",
  "env.entry.LUMEN_QWEN35_FORCE_REQUIRED_PARAMS.help":
    "For Qwen3.6 tool calls, inject a <parameter=KEY> opener before the model can close a function with a required parameter missing — preventing empty calls like read() with no path. The model still writes the value. Helps weak/quantized models with many tools; off by default.",
  "env.entry.LUMEN_QWEN35_TQ_KV.label": "TurboQuant KV cache",
  "env.entry.LUMEN_QWEN35_TQ_KV.help":
    "Compress the Qwen3.6 full-attention KV cache with TurboQuant (rotation + Lloyd-Max scalar quant), cutting the growing KV memory ~2-4× so longer contexts fit before OOM. Trades a small dequant-on-read cost; at long context the memory saved can outweigh it. Linear-attention layers are unaffected. Experimental — measure quality (cosine/top-1) before relying on it. Off by default.",
  "env.entry.LUMEN_QWEN35_TQ_KV_BITS.label": "TurboQuant KV bits",
  "env.entry.LUMEN_QWEN35_TQ_KV_BITS.help":
    "Lloyd-Max bit width for TurboQuant KV (2-8). 8 = near-lossless (start here), 6 = lowest clean, 4 = aggressive/lossy. Only used when TurboQuant KV cache is enabled. Lower bits = more memory saved but more quality loss.",

  // Memory calculator (predict peak memory vs context / chunk / KV mode).
  "memcalc.title": "Memory calculator",
  "memcalc.noGeometry":
    "No memory profile for model '{model}'. The calculator supports the catalog's native MLX models (e.g. Qwen3.6-35B).",
  "memcalc.budget": "Budget",
  "memcalc.budget.hint":
    "MLX memory budget ≈ machine RAM minus OS headroom. Check the startup log line [mlx-mem] memory_limit set to N GB for the exact value.",
  "memcalc.kv": "KV",
  "memcalc.bits": "bits",
  "memcalc.bits.hint":
    "bits = quality only (LUMEN_QWEN35_TQ_KV_BITS). Storage is ~2× for ANY bits because codes are stored unpacked (uint8). 8 = near-lossless, 6 = lowest clean, 4 = lossy. Tick 'uint4 packed' at 4-bit to preview true ~4× (packing not yet wired).",
  "memcalc.packed": "uint4 packed*",
  "memcalc.peak": "peak / budget",
  "memcalc.chunk": "Chunk",
  "memcalc.prefix": "prefix cache",
  "memcalc.context": "Context",
  "memcalc.over": "over budget",
  "memcalc.maxAtConfig": "Max context at this config:",
  "memcalc.table.title": "Max context (tokens) — budget {budget} GB",
  "memcalc.table.note":
    "Shrinking the chunk is the biggest lever (attention scores scale with chunk × context). TQ trims persistent KV. *TQ4 = uint4 packing, not yet wired (preview). Estimates — calibrate against your peak= logs.",
  "memcalc.apply": "Apply to tuning",
  "memcalc.applied": "Applied ✓ — Stop → Start to take effect",
  "memcalc.apply.hint":
    "Updates the QUANT card (KV mode / bits) + env overrides (chunk / prefix). Restart to apply.",

  // Shared toast for QUANT / CONTEXT / SERVER card saves. The variant shown
  // depends on whether the inference server is currently running — running
  // server needs a restart for env-derived knobs (cache mode/bits, ctx
  // caps, etc.) to take effect.
  "config.savedRestartHint": "Saved. Restart the server to apply (Stop → Start).",
  "config.saved": "Saved.",
  // Inline action button on the savedToast — single-click "Stop → wait →
  // Start" so users don't have to find the toggle in the header.
  "config.restartNow": "Restart now",
  "config.restarting": "Restarting server…",
  "config.restarted": "Server restarted.",

  // ── DoctorPanel ─────────────────────────────────────────────────
  "doctor.title": "Doctor",
  "doctor.run": "Run checks",
  "doctor.running": "Running…",
  "doctor.empty": "No report yet. Click Run checks to start.",
  "doctor.overall.healthy": "Healthy",
  "doctor.overall.degraded": "Degraded",
  "doctor.overall.blocked": "Blocked",
  "doctor.overall.unknown": "Unknown",
  "doctor.status.pass": "PASS",
  "doctor.status.warn": "WARN",
  "doctor.status.fail": "FAIL",
  "doctor.fix": "Fix",
  "doctor.fixing": "Fixing…",
  "doctor.fixHint": "Suggested fix:",
  "doctor.fixCommand": "Command:",
  "doctor.recheck": "Re-check",
  "doctor.checking": "Checking…",
  "doctor.intro":
    "Diagnostics run on app start and on demand. Each row links to a fix.",
  "doctor.idle": "Click Re-check to run diagnostics.",
  "doctor.working": "Working…",
  "doctor.fixIt": "Fix it",
  "doctor.failed": "failed:",

  // ── UpdatePanel ─────────────────────────────────────────────────
  "update.title": "Lumen update",
  "update.checking": "Checking for updates…",
  "update.check": "Check for updates",
  "update.current": "Current version:",
  "update.latest": "Latest version:",
  "update.upToDate": "You're up to date.",
  "update.available": "Update available",
  "update.install": "Install update",
  "update.installing": "Installing…",
  "update.serverWarn":
    "Stop the server before installing — the app must restart to apply the update.",
  "update.releaseNotes": "Release notes",
  "update.published": "Published",
  "update.error": "Update error:",
  "update.installRestart": "Install & restart",
  "update.applying": "— applying…",
  "update.confirm.running":
    "The inference server is running. Installing the update will stop it and restart the app. Continue?",
  "update.availableSuffix": "available",
  "update.onLatest": "You're on the latest version.",

  // ── Doctor check NAMES (keyed by backend `check.id`) ────────────
  "doctor.check.os_version.name": "macOS version",
  "doctor.check.architecture.name": "CPU architecture",
  "doctor.check.ram.name": "Total RAM",
  "doctor.check.disk_free.name": "Free disk space",
  "doctor.check.models_dir.name": "Models directory",
  "doctor.check.server_binary.name": "lumen-server binary",
  "doctor.check.port_free.name": "Server port",
  "doctor.check.active_model.name": "Active model",
  "doctor.check.huggingface.name": "Hugging Face network",

  // ── Doctor message templates (id + status hint) ─────────────────
  "doctor.msg.os_version.ok": "macOS",
  "doctor.msg.os_version.warn.suffix": "— supported, but 14+ recommended",
  "doctor.msg.os_version.fail.suffix": "— unsupported",
  "doctor.msg.os_version.unknown": "macOS version unknown",
  "doctor.msg.architecture.silicon": "Apple Silicon (arm64)",
  "doctor.msg.architecture.intel": "Intel Mac (x86_64)",
  "doctor.msg.architecture.other": "unsupported architecture:",
  "doctor.msg.ram.gb": "GB",
  "doctor.msg.disk_free.template": "{gb} GB free at {path}",
  "doctor.msg.models_dir.writable": "writable:",
  "doctor.msg.models_dir.notWritable": "not writable:",
  "doctor.msg.models_dir.missing": "missing:",
  "doctor.msg.server_binary.found": "found:",
  "doctor.msg.server_binary.notExecutable": "not executable:",
  "doctor.msg.server_binary.notFound": "not found",
  "doctor.msg.port_free.available": "port {port} available",
  "doctor.msg.port_free.inUse": "port {port} in use",
  "doctor.msg.active_model.none": "no model selected",
  "doctor.msg.active_model.ready": "{id} ready",
  "doctor.msg.active_model.incomplete": "{id} on disk but incomplete",
  "doctor.msg.active_model.missing": "{id} not on disk",
  "doctor.msg.huggingface.reachable": "reachable ({code})",
  "doctor.msg.huggingface.unexpected": "unexpected status {code}",
  "doctor.msg.huggingface.unreachable": "unreachable",
  "doctor.msg.huggingface.clientInitFailed": "http client init failed",

  // ── Doctor fix hints (matched against backend English text) ─────
  "doctor.hint.os_version.warn":
    "Apple Silicon MPS performance improvements landed in macOS 14 (Sonoma). Earlier versions work but leave a few % on the table.",
  "doctor.hint.os_version.fail":
    "lumen requires macOS 11 (Big Sur) or newer for the Metal stack. Update via System Settings → General → Software Update.",
  "doctor.hint.os_version.unknown":
    "Could not run `sw_vers`. If you're on macOS this is unusual — please report the issue.",
  "doctor.hint.architecture.intel":
    "Metal works on Intel Macs but Apple Silicon is 5-20× faster for inference and is the supported development target. Consider running smaller (<2B) models, or build/run on an M-series machine.",
  "doctor.hint.architecture.unsupported":
    "lumen targets macOS Apple Silicon (and limited Intel Mac). Other platforms are not yet supported.",
  "doctor.hint.ram.warnLow":
    "16 GB is enough for 1.5-7B models. For 13B+ or Mixture-of-Experts models, 24 GB+ is recommended.",
  "doctor.hint.ram.tight":
    "8-16 GB is tight. Stick to <2B parameter models. Turn on 3-bit cache quantization and disable the wired memory caps (Server card → Disable caps) if you OOM.",
  "doctor.hint.ram.fail":
    "Less than 8 GB RAM — lumen will OOM on almost any model. Free RAM or use a larger machine.",
  "doctor.hint.disk_free.warn":
    "Most modern weight sets are 5-30 GB each. Keep an eye on free space — the MODELS card shows per-model sizes.",
  "doctor.hint.disk_free.fail":
    "Less than 20 GB free. Model downloads will fail mid-stream. Free up disk (or change the weights path in SERVER → Weights dir).",
  "doctor.hint.models_dir.notWritable":
    "Fix permissions on the models directory, or change it in SERVER → Weights dir.",
  "doctor.hint.models_dir.missing": "Click Fix to create the directory.",
  "doctor.hint.server_binary.notExecutable":
    "Set the executable bit on the binary, or rebuild it.",
  "doctor.hint.server_binary.notFound":
    "Build the inference server from source, or set SERVER → server_binary_path in config.toml.",
  "doctor.hint.port_free.inUse":
    "Another process is bound to this port. Change PORT in the SERVER card, or stop the other process.",
  "doctor.hint.active_model.none":
    "Pick a model in the ACTIVE MODEL card, or download one from HF Hub via the MODELS card.",
  "doctor.hint.active_model.incomplete":
    "Re-download the model from the MODELS card, or remove + re-add.",
  "doctor.hint.active_model.missing":
    "Download the model from the MODELS card, or pick a different active model.",
  "doctor.hint.huggingface.unexpected":
    "huggingface.co responded but not with 2xx/3xx. Service may be degraded — downloads may fail.",
  "doctor.hint.huggingface.unreachable":
    "Model downloads via HF Hub will fail. Check your internet connection, VPN, or proxy.",

  // ── Doctor detail (incomplete-shard explainer) ─────────────────
  "doctor.detail.active_model.incomplete":
    "Required files (config.json + at least one safetensors/gguf shard) are missing.",
};
