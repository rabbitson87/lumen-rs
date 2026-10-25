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
  "header.title.noModel":
    "Select a chat and/or image model in the MODELS card first.",

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
  "action.deselect": "Deselect",
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
  "server.launchAtLogin": "Launch at login",
  "server.launchAtLogin.hint": "open Lumen after a reboot or login",
  "server.autoStart": "Auto-start server",
  "server.autoStart.hint": "start the server whenever Lumen opens",
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
  "spec.title": "SPECULATIVE DECODING",
  "spec.titleHint": "(draft tokens, then verify)",
  "spec.kind": "Strategy",
  "spec.off": "Off",
  "spec.lookup": "Lookup",
  "spec.mtp": "MTP",
  "spec.hint.kind":
    "Off is the safe default. Lookup drafts from n-grams already in the prompt (no extra weights, helps on repetitive text). MTP uses the model's own multi-token-prediction head and needs a checkpoint that ships one. Output is unchanged either way \u2014 rejected drafts are discarded. Sets",
  "spec.draftK": "Draft tokens per step",
  "spec.hint.draftK":
    "How many tokens to draft before verifying. Higher drafts more per step but wastes more work when the draft is rejected; 1\u20132 is usually best, and past ~3 acceptance falls faster than the saving grows. Sets",
  "batch.title": "BATCHING",
  "batch.titleHint": "(serve several requests per step)",
  "batch.enable": "Continuous batching",
  "batch.off": "Off",
  "batch.on": "On",
  "batch.hint.enable":
    "Decode several sequences in one forward pass. Raises total throughput and lowers per-sequence speed. Only greedy requests are admitted \u2014 anything with temperature or top-p falls back to single-sequence decode. Sets",
  "batch.max": "Max sequences per step",
  "batch.hint.max":
    "Upper bound on sequences decoded together. Each concurrent sequence holds its own KV cache, so this is a memory knob as much as a throughput one. Sets",
  "diskkv.title": "DISK KV CACHE",
  "diskkv.titleHint": "(persist prefix KV across restarts)",
  "diskkv.enable": "Enable",
  "diskkv.off": "Off",
  "diskkv.on": "On",
  "diskkv.maxGb": "Max size (GB)",
  "diskkv.ttl": "TTL (seconds)",
  "diskkv.hint.enable":
    "Persist the prompt-prefix KV cache to disk so a server restart reuses it instead of cold-prefilling. Sets",
  "diskkv.hint.maxGb": "On-disk budget; oldest entries evicted past this. 0 = unlimited. Sets",
  "diskkv.hint.ttl": "Drop disk entries idle longer than this. Default 86400 (1 day); 0 = never expire. Sets",
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
  "debug.weightsDir": "Gemma 4 weights dir",
  "debug.weightsDir.placeholder": "only for a Gemma 4 model not downloaded here",
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
  "models.action.title.deselect":
    "Click to deselect — run image-only, or pick a different LLM",

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

  // ── Image models card (text-to-image / diffusion) ───────────────
  "imageModels.title": "IMAGE MODELS",
  "imageModels.titleHint": "text-to-image",
  "imageModels.badge": "IMAGE",
  "imageModels.active": "Active",
  "imageModels.overBudget": "Larger than this Mac's RAM — generation may swap or fail.",
  "imageModels.action.title.overBudget": "Exceeds this Mac's RAM budget",
  "imageModels.action.title.deselect": "Click to deselect this image model",
  "imageModels.startHint": "Press Start to launch the server in image mode, then a generation panel appears below. Selecting an LLM as well launches hybrid mode (both serve at once).",

  // ── Image generation panel ──────────────────────────────────────
  "image.title": "IMAGE GENERATION",
  "image.prompt": "Prompt",
  "image.prompt.placeholder": "A cinematic photo of a red fox in a snowy forest, golden hour…",
  "image.size": "Size",
  "image.steps": "Steps",
  "image.steps.hint": "Default 28. Use 4–8 for a fast draft.",
  "image.seed": "Seed",
  "image.guidance": "Guidance",
  "image.generate": "Generate",
  "image.generating": "Generating…",
  "image.generating.note": "This can take a few minutes on 36 GB.",
  "image.download": "Download PNG",

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
    "Tokens per prefill chunk for Qwen 3.6 long prompts. Larger = fewer GPU syncs (faster cold prefill) but more peak memory. Raise on Macs with more RAM; lower if a long prompt OOMs. Default 2048. For Flash-Next use the Flash-Next card (Tuning tab): its default is 1024, and its chunk comes out of the expert pool.",
  "env.entry.LUMEN_MAX_PROMPT_TOKENS.label": "Max prompt tokens (reject cap)",
  "env.entry.LUMEN_MAX_PROMPT_TOKENS.help":
    "Hard ceiling on prompt length: requests longer than this are rejected (clean error) BEFORE prefill — this guards the server from an uncaught Metal OOM that otherwise crashes the whole process on very long prompts (~20k tok for Gemma-4-26B, ~32k for Qwen3.6-35B on a 36 GB Mac). Default 16384, empirically safe for both families. Falls back to LUMEN_PREFILL_CHUNK. Raise ONLY if you have more RAM — a longer prefill needs proportionally more memory, and an over-large value re-exposes the OOM crash. For Flash-Next set it in the Flash-Next card (Tuning tab): the model plans its expert pool around it, since the KV cache for the longest prompt (60 KB per token) comes out of the pool. Without a choice there the app sends this Mac's default (16,384; 8,192 under 11 GB of GPU memory), and an entry here replaces it.",
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
  "env.entry.LUMEN_QWEN35_TOOL_GRAMMAR.label": "Tool-call grammar",
  "env.entry.LUMEN_QWEN35_TOOL_GRAMMAR.help":
    "Constrain Qwen3.6 tool-call output with a grammar so required tool_choice / named tool_choice can only emit well-formed calls with all required parameters present (a stronger, structural version of 'Force required tool params'). Masks logits after each forward — sampling only, no token skipping, so SSM/conv state is unaffected. On by default; set 0 to disable.",
  "env.entry.LUMEN_QWEN35_TOOL_GRAMMAR_EAGER.label": "Tool grammar on auto",
  "env.entry.LUMEN_QWEN35_TOOL_GRAMMAR_EAGER.help":
    "Also apply the tool-call grammar when tool_choice is 'auto' (not just required/named). Off by default — auto lets the model decide whether to call a tool at all, so constraining it early can over-trigger calls. Enable only if a model over-emits malformed tool calls under auto.",
  "env.entry.LUMEN_QWEN35_TQ_KV.label": "TurboQuant KV cache",
  "env.entry.LUMEN_QWEN35_TQ_KV.help":
    "Compress the Qwen3.6 full-attention KV cache with TurboQuant (rotation + Lloyd-Max scalar quant), cutting the growing KV memory ~2-4× so longer contexts fit before OOM. Trades a small dequant-on-read cost; at long context the memory saved can outweigh it. Linear-attention layers are unaffected. Experimental — measure quality (cosine/top-1) before relying on it. Off by default.",
  "env.entry.LUMEN_QWEN35_TQ_KV_BITS.label": "TurboQuant KV bits",
  "env.entry.LUMEN_QWEN35_TQ_KV_BITS.help":
    "Lloyd-Max bit width for TurboQuant KV (2-8). 8 = near-lossless (start here), 6 = lowest clean, 4 = aggressive/lossy. Only used when TurboQuant KV cache is enabled. Lower bits = more memory saved but more quality loss.",
  "env.entry.LUMEN_GEMMA4_TOOL_GRAMMAR_EAGER.label": "Enforce tool_choice (Gemma 4)",
  "env.entry.LUMEN_GEMMA4_TOOL_GRAMMAR_EAGER.help":
    "When tool_choice is `required` or names a function, constrain decoding with an Eager grammar (active from the first token) so the call's arguments are schema-valid. Off by default: the default Lazy grammar only activates after the model emits the tool-call opener, so a forced call's body is currently unconstrained. Eager + long agentic loops drove some quantized builds into a repetition cycle — verify on your model before relying on it. Does not affect tool_choice=auto.",
  "env.entry.LUMEN_MLX_PREFIX_INCREMENTAL.label": "Incremental prefix cache (shared system prompt)",
  "env.entry.LUMEN_MLX_PREFIX_INCREMENTAL.help":
    "On a cold prefill, also snapshot the shared system-prompt head so a later request with the SAME system prompt but a different user turn forks the cached head instead of re-prefilling it from scratch. Speeds up multi-user / branching workloads that share one big system prompt. Off by default; each extra snapshot pins KV memory (bounded by the per-key branch cap), so enable it on machines with RAM headroom (high-end MacBook / Mac Studio). Qwen 3.6 native path only.",
  "env.entry.LUMEN_MLX_SERVER_MODE.label": "Multi-request serving mode (multi-user + bulk batch)",
  "env.entry.LUMEN_MLX_SERVER_MODE.help":
    "ONE switch for concurrent serving on a high-RAM Mac (high-end MacBook / Mac Studio): turns on the whole multi-request path at once — batched decode (several sequences advance per GPU step) AND single-copy shared-prefix KV dedup. Covers both live multi-user serving and bulk/offline batch jobs (where many requests share one system prompt, so dedup pays off most). Off by default: most MLX users run a single conversation, and the solo path is byte-identical when off. Each part can still be overridden individually (e.g. set LUMEN_MLX_BATCH_DECODE=0 to keep decode batched off while server mode is on). MLX (Qwen 3.6 / Gemma 4) only; non-greedy / tool / thinking requests always fall back to the sequential path.",
  "env.entry.LUMEN_MLX_BATCH_DECODE.label": "Batched decode (override)",
  "env.entry.LUMEN_MLX_BATCH_DECODE.help":
    "Per-feature override for the batched multi-sequence decode scheduler. Normally inherited from LUMEN_MLX_SERVER_MODE; set explicitly to force on (1) or off (0) regardless of server mode. When on, concurrent greedy chat requests advance together (one GPU step amortized across N sequences), giving higher aggregate throughput for multiple users; a single request (N=1) is byte-identical to the sequential path. MLX native path only.",
  "env.entry.LUMEN_MLX_SHARED_PREFIX.label": "Single-copy shared-prefix KV (in-batch dedup)",
  "env.entry.LUMEN_MLX_SHARED_PREFIX.help":
    "When several requests decode concurrently in the same batch and share a common prompt prefix (e.g. many users hitting one big system prompt), store that prefix's attention KV ONCE and let every sequence attend it via a flash-style log-sum-exp merge, instead of every sequence carrying its own copy. Frees roughly (N-1) x prefix_tokens of full-attention KV — meaningful for many-user / long-system-prompt serving on high-end Macs. Off by default (decode is byte-identical when off); the merge is numerically equal to full attention to ~1e-4 (not bit-identical). Requires batched decode (LUMEN_MLX_BATCH_DECODE) and the Qwen 3.6 native path.",
  "env.entry.LUMEN_MLX_SHARED_PREFIX_MIN.label": "Shared-prefix min tokens",
  "env.entry.LUMEN_MLX_SHARED_PREFIX_MIN.help":
    "Minimum shared prefix length (in tokens) before single-copy dedup activates. Short shared prefixes save little KV but still pay the per-step split-attention merge cost, so dedup only kicks in once the common prefix is at least this long. Default 64. Only used when LUMEN_MLX_SHARED_PREFIX is on.",
  "env.entry.LUMEN_MLX_DRAFT_MODEL.label": "Draft model (spec decode) — experimental",
  "env.entry.LUMEN_MLX_DRAFT_MODEL.help":
    "EXPERIMENTAL. Path or HF id of a small draft model for greedy draft-model speculative decoding on the Qwen3.5/3.6 native path. The draft proposes tokens that the target verifies in one batched forward, accepting the longest matching prefix — lossless vs greedy (target's argmax is always the source of truth). Requires the native runner and a draft whose vocab matches the target (auto-disabled on mismatch). Engaged for greedy requests only. Empty = OFF (default; existing decode runs unchanged).",
  "env.entry.LUMEN_MLX_DRAFT_NMAX.label": "Draft tokens per step",
  "env.entry.LUMEN_MLX_DRAFT_NMAX.help":
    "How many tokens the draft model proposes per verify attempt (1-16, default 4). Higher = more potential speedup when the draft agrees with the target, but more wasted draft work on rejection. Only used when a draft model is set.",
  "env.entry.LUMEN_MLX_DRAFT_PMIN.label": "Draft min top-1 prob",
  "env.entry.LUMEN_MLX_DRAFT_PMIN.help":
    "Minimum draft top-1 probability to keep extending a proposal (0-1, default 0 = disabled). Currently a no-op placeholder: the draft proposes via argmax-only steps with no probability readout, so probability gating is pending hardware verification. Leave at 0.",


  // Memory calculator (predict peak memory vs context / chunk / KV mode).
  "memcalc.title": "Memory calculator",
  "memcalc.noGeometry":
    "No memory profile for model '{model}'. The calculator supports the catalog's native MLX models (e.g. Qwen3.6-35B, Gemma 4 26B-A4B).",
  "memcalc.flashNext":
    "Flash-Next streams its experts from the SSD, so its memory plan and settings are in the Flash-Next card above.",
  "memcalc.active": "Weights / active",
  "memcalc.active.hint":
    "Resident weight memory at a small prefill — the estimate's anchor. Pre-filled from the model's measured value; correct it to your own startup log line [mlx-mem] ... active=N GB for an exact, machine-specific estimate.",
  "memcalc.budget": "Budget",
  "memcalc.budget.available": "📊 {n} usable",
  "memcalc.budget.availableHint":
    "Snap the budget to the memory the model can realistically claim and re-optimize. Reclaim-aware: only WIRED memory is truly unavailable — macOS swap-compresses/evicts other apps' inactive + compressed pages when the model allocates, so this is far higher than the top-right 'used' gauge (which counts that reclaimable memory as used). Updates live.",
  "memcalc.budget.overRam":
    "Budget {budget} GB exceeds this machine's {ram} GB RAM — a limit above physical RAM just thrashes (swap). Lower it to ≤ {ram}.",
  "memcalc.budget.hint":
    "MLX memory budget ≈ machine RAM minus OS headroom. Check the startup log line [mlx-mem] memory_limit set to N GB for the exact value.",
  "memcalc.optimize": "✨ Optimize",
  "memcalc.optimize.hint":
    "Auto-fill every knob with the best config that fits this RAM budget (maximizes prompt + output context, conceding KV quality / prefill speed only as the budget tightens). Review, then Apply.",
  "memcalc.reco.ok":
    "Recommended (agentic): full quality (bf16) + prefix cache, ample context fits {budget} GB with headroom.",
  "memcalc.reco.aggressive":
    "Recommended: {budget} GB is tight, so quality/prefix were traded down to keep a usable context. Raise the budget for bf16 + prefix cache.",
  "memcalc.reco.tooSmall":
    "{budget} GB is below this model's load floor (weights + overhead). Raise the budget or pick a smaller model.",
  "memcalc.kv": "KV",
  "memcalc.bits": "bits",
  "memcalc.bits.hint":
    "bits = quality only (LUMEN_QWEN35_TQ_KV_BITS). Storage is ~2× for ANY bits because codes are stored unpacked (uint8). 8 = near-lossless, 6 = lowest clean, 4 = lossy. Tick 'uint4 packed' at 4-bit to preview true ~4× (packing not yet wired).",
  "memcalc.packed": "uint4 packed*",
  "memcalc.peak": "peak / budget",
  "memcalc.chunk": "Chunk",
  "memcalc.prefix": "prefix cache",
  "memcalc.maxPrompt": "Max prompt",
  "memcalc.outMax": "Output max",
  "memcalc.workingSet":
    "Working set: prompt {prompt} + out {out} = {total} tok → peak {peak} GB",
  "memcalc.context": "Context",
  "memcalc.over": "over budget",
  "memcalc.maxAtConfig": "Max context at this config:",
  "memcalc.table.title": "Max context (tokens) — budget {budget} GB",
  "memcalc.table.note":
    "Shrinking the chunk is the biggest lever (attention scores scale with chunk × context). TQ trims persistent KV. *TQ4 = uint4 packing, not yet wired (preview). Estimates — calibrate against your peak= logs.",
  "memcalc.apply": "Apply to tuning",
  "memcalc.applied": "Applied ✓ — Stop → Start to take effect",
  "memcalc.apply.hint":
    "Updates the QUANT card (KV mode / bits), the CONTEXT card (max ctx / max prompt / output) + env overrides (chunk / prefix), model-aware. Restart to apply.",
  "memcalc.applyBudget": "set as memory limit",
  "memcalc.applyBudget.hint":
    "Also write this budget to the server's MLX memory_limit_gb (the hard cap). Off = budget is a planning input only.",

  // Flash-Next memory calculator (expert pool vs the longest prompt; the experts stream from the SSD).
  "flashnext.geometry":
    "Qwen 3.8 Flash-Next · 48 layers × 512 experts · {kb} KB of f32 KV per token of context",
  "flashnext.thisMac": "This Mac: {ram} GB RAM, so the GPU may fill ~{usable} GB.",
  "flashnext.thisMac.limit": "(Capped by the Memory GB setting at {n} GB.)",
  "flashnext.freeNow":
    "Other apps hold memory right now: a server started now plans within {now} GB ({below} GB less), so its expert pool gets at most {pool} GB. The pool keeps that size until the server restarts: close apps before starting it for a bigger one.",
  "flashnext.freeNow.floor":
    "Other apps hold memory right now: a server started now has {now} GB to plan within ({below} GB less), and not even the smallest pool ({floor} GB) fits beside them, so expect swapping. Close apps before starting the server.",
  "flashnext.tight":
    "That is a tight machine: the plan takes {chunk}-token chunks and {ctx}-token prompts unless told otherwise.",
  "flashnext.fits":
    "It fits: the trunk, the smallest pool and the reserve for a {ctx}-token prompt need {need} GB.",
  "flashnext.noFit":
    "Does not fit: the trunk, the smallest pool and the reserve for a {ctx}-token prompt need {need} GB. Up to {max} tokens fit at a {chunk}-token chunk; lower the longest prompt or the chunk.",
  "flashnext.noFitAny":
    "Does not fit even with an empty prompt: this Mac needs at least {min} GB of RAM.",
  "flashnext.pool": "Expert pool",
  "flashnext.pool.auto": "auto",
  "flashnext.pool.hint":
    "Experts kept in memory; the rest come from the SSD, so a bigger pool decodes faster. auto is what is left after the trunk and the reserve for the longest prompt: a longer prompt shrinks it.",
  "flashnext.pool.floor": "Raised to the {floor} GB minimum the server runs with.",
  "flashnext.pool.oneRegion":
    "Under {gb} GB the pool lends one staging region only, so prefill cannot read the next layer while it computes the current one (slower prefill, nothing fails).",
  "flashnext.chunk": "Chunk",
  "flashnext.chunk.default": "default",
  "flashnext.card.title":
    "FLASH-NEXT",
  "flashnext.card.titleHint":
    "(experts streamed from the SSD)",
  "flashnext.disabled":
    "Select the Flash-Next model to change these settings.",
  "flashnext.shadowed":
    "An env override replaces the setting here: {keys}. It is applied last, so it wins.",
  "flashnext.shadowed.clear":
    "Remove the override",
  "flashnext.chunk.hint":
    "Tokens per prefill step. Its activations (about 1.1 MB per token) come out of the pool; the default follows this Mac.",
  "flashnext.plan.title":
    "Memory plan",
  "flashnext.context.hint":
    "The server's prompt cap (LUMEN_MAX_PROMPT_TOKENS): its KV cache (60 KB per token) comes out of the pool. Picking a chip saves it; this Mac's default is what the app sends when none is set.",
  "flashnext.peak": "resident / usable",
  "flashnext.row.trunk": "resident weights",
  "flashnext.row.pool": "expert pool",
  "flashnext.row.buffers": "prefill buffers",
  "flashnext.row.kv": "KV @ prompt",
  "flashnext.row.scores": "attention scores",
  "flashnext.overBy":
    "{over} GB over what the GPU may fill: lower the pool to {pool} GB or less (that is what auto gives), or the longest prompt.",
  "flashnext.autoPool": "Auto pool at this prompt:",
  "flashnext.maxCtx.auto": "Longest prompt before auto reaches the {floor} GB floor:",
  "flashnext.maxCtx.manual": "Longest prompt that fits beside this pool:",
  "flashnext.speed.title":
    "Decode speed",
  "flashnext.speed.withPack":
    "With the expert pack: about {tps} tok/s",
  "flashnext.speed.col.pool": "pool",
  "flashnext.speed.col.shards":
    "shipped shards (tok/s)",
  "flashnext.speed.col.pack":
    "expert pack (tok/s)",
  "flashnext.speed.intro":
    "Decode speed depends mostly on this Mac's memory bandwidth and SSD read speed. Measure both to get an estimate for this Mac.",
  "flashnext.speed.missing.both":
    "Not measured yet: memory bandwidth and SSD read speed.",
  "flashnext.speed.missing.memory":
    "Not measured yet: memory bandwidth.",
  "flashnext.speed.missing.ssd":
    "Not measured yet: SSD read speed.",
  "flashnext.speed.missing.ssdStale":
    "The SSD speed was measured with the model in another folder. Measure again.",
  "flashnext.speed.reference":
    "Reference: measured on an M3 Max with 36 GB (memory {mem} GB/s, SSD {ssd} GB/s), expert pool set explicitly, direct reads (the page cache is not counted). Not this Mac.",
  "flashnext.speed.estimated":
    "Estimated on this Mac: {tps} tok/s at a {pool} GB pool",
  "flashnext.speed.accuracy":
    "An estimate, about ±10%.",
  "flashnext.speed.usingPack":
    "using the expert pack",
  "flashnext.speed.inputs":
    "memory {mem} GB/s · SSD {ssd} GB/s · measured {date}",
  "flashnext.speed.slowDisk":
    "This SSD reads slowly ({ssd} GB/s random), and decode speed depends on it. Keep the model on the Mac's internal SSD.",
  "flashnext.speed.note":
    "A real 16 / 24 GB Mac's own memory pressure is not measured here. Prefill (~230 tok/s) does not depend on the pool.",
  "flashnext.disk": "Disk needed",
  "flashnext.disk.detail": "{model} GB weights",
  "flashnext.disk.pack": "GB expert pack",
  "flashnext.disk.mtp": "GB draft head",
  "flashnext.active": "Active:",
  "flashnext.default": "default",
  "flashnext.on": "on",
  "flashnext.off": "off",
  "flashnext.choice.auto": "Auto",
  "flashnext.choice.off": "Off",
  "flashnext.choice.on": "On",
  "flashnext.choice.build": "Build",
  "flashnext.choice.direct": "Direct",
  "flashnext.choice.buffered": "Buffered",
  "flashnext.opt.pack": "Expert pack",
  "flashnext.opt.mtp": "Draft head",
  "flashnext.opt.io": "Expert reads",
  "flashnext.opt.lanes": "Reader threads",
  "flashnext.prompt": "Longest prompt",
  "flashnext.summary":
    "Runs with: pool {pool} GB ({how}), prompts up to {prompt} tokens, expert pack {pack}, draft head {head}",
  "flashnext.summary.speed":
    "about {tps} tok/s on this Mac",
  "flashnext.summary.speedUnknown":
    "speed not measured yet",
  "flashnext.summary.auto": "auto",
  "flashnext.summary.set": "set by you",
  "flashnext.summary.noFit": "this does not fit the GPU's memory",
  "flashnext.reset": "Reset to recommended",
  "flashnext.reset.hint":
    "Puts every option in this card back to the server's own default (nothing is sent for it). Env overrides are left alone.",
  "flashnext.pool.custom": "Custom…",
  "flashnext.pool.custom.hint": "GB: 8, 8g, 8GB, 12.5g",
  "flashnext.pool.custom.invalid": "Not a size: try 8, 8g or 12.5g.",
  "flashnext.pool.chip.auto":
    "What is left after the trunk and the reserve for the longest prompt: the server sizes it itself.",
  "flashnext.pool.chip.noFit":
    "A {gb} GB pool does not fit beside prompts up to {ctx} tokens on this Mac: up to {max} GB does. A shorter longest prompt makes room (otherwise the server lowers the pool and logs it).",
  "flashnext.pool.active.auto": "auto = {pool} GB on this Mac",
  "flashnext.pool.active.set": "{pool} GB",
  "flashnext.pool.lowered":
    "Does not fit with prompts up to {ctx} tokens: the server lowers it to about {max} GB and logs it. A shorter longest prompt makes room.",
  "flashnext.pool.noFitAuto": "Not even the smallest pool fits with this prompt length.",
  "flashnext.pool.legend":
    "~ estimated tok/s on this Mac (an estimate, about ±10%).",
  "flashnext.prompt.custom.hint": "tokens or K: 48k, 50000",
  "flashnext.prompt.chip.default":
    "{n} tokens: this Mac's default, which the app sends unless you set one.",
  "flashnext.prompt.chip.pool": "Prompts up to {n} tokens leave an auto pool of {gb} GB.",
  "flashnext.prompt.chip.noFit":
    "Prompts up to {n} tokens leave no room for the pool on this Mac: it would drop below its {floor} GB floor.",
  "flashnext.prompt.active": "{n} tokens",
  "flashnext.prompt.set": "set by you",
  "flashnext.prompt.default": "this Mac's default",
  "flashnext.prompt.pool": "auto pool {gb} GB",
  "flashnext.prompt.noFitActive": "Does not fit.",
  "flashnext.prompt.maxCtx": "Max context ({n}) caps it; a longer choice raises Max context.",
  "flashnext.pack.mode.auto": "Use the pack when it sits next to the weights.",
  "flashnext.pack.mode.off": "Never use the pack: read the shipped shards.",
  "flashnext.pack.mode.build": "Build the pack when the server starts, if there is none.",
  "flashnext.pack.status.complete":
    "Pack present: experts.lpack ({gb} GB), decode about 7-15% faster.",
  "flashnext.pack.inUse": "in use",
  "flashnext.pack.unused": "not used while Off",
  "flashnext.pack.status.incomplete":
    "An unfinished pack ({gb} GB) is on disk; the server ignores it. Building makes a new one.",
  "flashnext.pack.status.absent": "No pack yet. It needs {need} GB of disk; {free} GB is free.",
  "flashnext.pack.status.absentNoFree": "No pack yet. It needs {need} GB of disk.",
  "flashnext.pack.status.notInstalled": "Flash-Next is not installed.",
  "flashnext.pack.build": "Build expert pack",
  "flashnext.pack.build.hint":
    "Runs lumen-server --pack-experts on the model in the background; about a minute on an internal SSD.",
  "flashnext.pack.starting": "Starting the build…",
  "flashnext.pack.building": "Building the pack: layer {n} of {total}",
  "flashnext.pack.cancel": "Cancel",
  "flashnext.pack.cancelled": "Build cancelled. Nothing was left on disk.",
  "flashnext.pack.failed": "The build failed:",
  "flashnext.pack.built": "The pack is built.",
  "flashnext.pack.block.not_installed": "Flash-Next is not installed.",
  "flashnext.pack.block.building": "A build is already running.",
  "flashnext.pack.block.already_built": "The pack is already built.",
  "flashnext.pack.block.server_running":
    "Stop the server first: the build reads the whole model from the same disk.",
  "flashnext.pack.block.not_enough_disk": "Needs {need} GB free; this volume has {free} GB.",
  "flashnext.mtp.mode.auto":
    "Use the draft head when mtp.safetensors is installed, under the server's speed governor.",
  "flashnext.mtp.mode.on":
    "Like Auto, but the server refuses to start without the head.",
  "flashnext.mtp.mode.off":
    "Never use the draft head (the default): the text is the same as plain decoding.",
  "flashnext.mtp.status.present": "mtp.safetensors is present.",
  "flashnext.mtp.status.presentOff":
    "Installed, not used (off by default).",
  "flashnext.mtp.status.missingOff":
    "Not installed. Not needed while the draft head is off.",
  "flashnext.mtp.status.missing": "mtp.safetensors is missing.",
  "flashnext.mtp.status.missingOn":
    "mtp.safetensors is missing, and On makes the server refuse to start.",
  "flashnext.mtp.fetch": "Downloads the head once (1.5 GB) into the model directory.",
  "flashnext.mtp.note":
    "Speculative decoding. Measured on the reference Mac (M3 Max, 36 GB), it gained at most 4% and often lost 1-4%, and its text is not bit-identical to plain decoding, so it is off unless you choose it. Needs the 1.5 GB head: python3 scripts/qwen4_mtp_fetch.py <model dir>",
  "flashnext.io.mode.auto": "Auto: uses the page cache while the Mac has more than ~12 GB to spare.",
  "flashnext.io.mode.direct": "Direct: bypasses the page cache; every miss reads the SSD.",
  "flashnext.io.mode.buffered": "Buffered: reads go through the page cache.",
  "flashnext.lanes.hint": "Threads that read experts from the SSD; 32 is the default.",
  "flashnext.settings.hint":
    "These are saved as you pick them; a restart applies them.",
  "flashnext.measure.button":
    "Measure this Mac",
  "flashnext.measure.again":
    "Measure again",
  "flashnext.measure.hint":
    "Reads memory for about 2 s, then ~1 GB of the model from the SSD for a few seconds. Best with the server stopped.",
  "flashnext.measure.needsModel":
    "The SSD part needs the Flash-Next model on disk; only memory will be measured.",
  "flashnext.measure.memory":
    "Measuring memory bandwidth…",
  "flashnext.measure.ssd":
    "Measuring SSD read speed…",
  "flashnext.measure.cancel":
    "Cancel",
  "flashnext.measure.cancelled":
    "Measurement cancelled. What was already measured is kept.",
  "flashnext.measure.failed":
    "Nothing could be measured.",
  "flashnext.measure.done":
    "Measured.",
  "flashnext.measure.memoryError":
    "Memory test failed: {error}",
  "flashnext.measure.ssdError":
    "SSD test failed: {error}",
  "flashnext.measure.ssdSkipped":
    "The SSD part was skipped: it needs the Flash-Next model on disk.",

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
  "doctor.ssd.run": "Test SSD speed",
  "doctor.ssd.cancel": "Cancel",
  "doctor.ssd.cancelled": "SSD test cancelled.",
  "doctor.ssd.note":
    "Flash-Next reads its experts from the SSD all the time. 'Test SSD speed' reads ~1 GB of its weights for a few seconds the way the server does. Stop the server first: it uses the same disk.",

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
  "doctor.check.flash_next_ram.name": "Flash-Next memory",
  "doctor.check.flash_next_disk.name": "Flash-Next disk",
  "doctor.check.flash_next_pack.name": "Flash-Next expert pack",
  "doctor.check.flash_next_mtp.name": "Flash-Next draft head",
  "doctor.check.flash_next_ssd.name": "Flash-Next SSD",

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
  "doctor.msg.flash_next_ram.ok": "{ram} GB RAM: expert pool ~{pool} GB at {ctx} tokens",
  "doctor.msg.flash_next_ram.context":
    "{ram} GB RAM: a {ctx}-token context does not fit, up to {max} do",
  "doctor.msg.flash_next_ram.short": "{ram} GB RAM, Flash-Next needs ~{need} GB",
  "doctor.msg.flash_next_disk.built": "{free} GB free, expert pack already built",
  "doctor.msg.flash_next_disk.room": "{free} GB free, room for the {pack} GB expert pack",
  "doctor.msg.flash_next_disk.noRoom": "{free} GB free, not enough for the {pack} GB expert pack",
  "doctor.msg.flash_next_disk.unknown": "free space unknown",
  "doctor.msg.flash_next_pack.present": "experts.lpack present ({gb} GB)",
  "doctor.msg.flash_next_pack.absent": "no expert pack (optional)",
  "doctor.msg.flash_next_pack.incomplete": "experts.lpack incomplete ({gb} GB of {full} GB)",
  "doctor.msg.flash_next_mtp.present": "draft head present (mtp.safetensors)",
  "doctor.msg.flash_next_mtp.off":
    "draft head installed, not used (off by default)",
  "doctor.msg.flash_next_mtp.absent": "no draft head (optional)",
  "doctor.msg.flash_next_mtp.required": "LUMEN_QWEN4_MTP=on but mtp.safetensors is missing",
  "doctor.msg.flash_next_ssd.fast":
    "random {rand} GB/s (sequential {seq} GB/s): fast enough for ~{tps} tok/s",
  "doctor.msg.flash_next_ssd.slow": "random {rand} GB/s (sequential {seq} GB/s): slow, ~{tps} tok/s",
  "doctor.msg.flash_next_ssd.fastNoEstimate":
    "random {rand} GB/s (sequential {seq} GB/s): fast enough",
  "doctor.msg.flash_next_ssd.slowNoEstimate":
    "random {rand} GB/s (sequential {seq} GB/s): slow",
  "doctor.msg.flash_next_ssd.notInstalled": "Flash-Next is not installed",
  "doctor.msg.flash_next_ssd.noShard": "no weight shard to read",
  "doctor.msg.flash_next_ssd.readFailed": "read failed",
  "doctor.msg.flash_next_ssd.noRandom": "could not measure random reads",
  "doctor.msg.flash_next_ssd.stalled": "the volume did not answer in time",

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
  "doctor.hint.flash_next_ram.short":
    "Flash-Next needs its 3.6 GB of resident weights, an expert pool of at least 2.8 GB and a prefill reserve inside the GPU's share of RAM (two thirds up to 32 GB, three quarters above; the memory limit lowers it). Below that the model cannot load.",
  "doctor.hint.flash_next_ram.context":
    "The KV cache of the longest prompt (60 KB per token) comes out of the GPU's share of RAM. The app sends LUMEN_MAX_PROMPT_TOKENS (16,384 tokens, 8,192 on a tight Mac) unless you set it, lowered to Max context. Lower it (the Flash-Next memory calculator's Apply writes it) or the prefill chunk, or raise the memory limit if it is below your RAM.",
  "doctor.hint.flash_next_disk.room":
    "An expert pack makes decode about 7-15% faster. Build it once with the command below, or set LUMEN_MOE_PACK=build and the server builds it on its next start.",
  "doctor.hint.flash_next_disk.noRoom":
    "The expert pack is optional. To use it, free about 70 GB on this volume, then build it with the command below.",
  "doctor.hint.flash_next_pack.incomplete":
    "The server ignores an unfinished pack and reads the shards instead. Delete it and build it again.",
  "doctor.hint.flash_next_mtp.required":
    "The server refuses to start with LUMEN_QWEN4_MTP=on and no draft head. Download the head (1.5 GB) with the command below, or remove the override.",
  "doctor.hint.flash_next_mtp.absent":
    "The draft head (1.5 GB, downloaded once) is off by default: measured on the reference Mac it gained at most 4% and often lost 1-4%, and its text is not bit-identical to plain decoding. To try it, download the head with the command below and set the draft head to Auto in the Flash-Next card.",
  "doctor.hint.flash_next_ssd.slow":
    "Decode speed is set by how fast the SSD serves expert reads. Keep the model on the Mac's internal SSD (external and network drives are usually below 2 GB/s), or give the expert pool more memory.",
  "doctor.hint.flash_next_ssd.stalled":
    "A read took longer than the test's time limit. The weights may sit on a network or sleeping external drive; decoding reads from them constantly, so keep the model on the Mac's internal SSD.",

  // ── Doctor detail (incomplete-shard explainer) ─────────────────
  "doctor.detail.active_model.incomplete":
    "Required files (config.json + at least one safetensors/gguf shard) are missing.",
  "doctor.detail.flash_next_ram":
    "The expert pool is what the GPU's share of RAM leaves after the 3.6 GB of resident weights and a prefill's reserve: buffers, the chunk's activations, the KV cache of the longest prompt (60 KB per token) and attention scores. A longer prompt cap shrinks it; a bigger pool means fewer SSD reads per token.",
  "doctor.detail.flash_next_ram.oneRegion":
    "With less than 4.2 GB the pool has room for one staging region only, so prefill cannot read the next layer while it computes the current one.",
  "doctor.detail.flash_next_pack.absent":
    "Without the pack the server reads each expert as nine scattered ranges of the weight shards. The pack stores the same bytes as one record per expert.",
  "doctor.detail.flash_next_ssd":
    "Read {total} MB of {shard} with the page cache bypassed: {seq} MB front to back, then {rand} MB as {reads} random {rec} MB reads on 4 threads, the way the server fetches experts.",
  "doctor.detail.flash_next_ssd.pack":
    "The estimate uses the expert pack's read efficiency.",
  "doctor.detail.flash_next_ssd.estimate":
    "Decode speed is estimated from this volume's random reads and this Mac's memory bandwidth ({mem} GB/s, measured), for an expert pool of ~{pool} GB, what the memory plan leaves on this Mac.",
  "doctor.detail.flash_next_ssd.measure":
    "Measure this Mac's memory bandwidth in the Flash-Next card (Tuning tab) to get a decode-speed estimate.",
  "doctor.detail.flash_next_ssd.early":
    "The test hit its time limit and stopped early; the speeds come from what it had read.",

  // ── Prompt-too-large / OOM-guard modal ─────────────────────────
  "oom.title": "Prompt too large for this model",
  "oom.body":
    "An incoming request's prompt was larger than this model can hold in memory on this machine. The server rejected it before it could crash the process with an out-of-memory error. Values below are token counts.",
  "oom.label.prompt": "Prompt",
  "oom.label.limit": "Accepted limit",
  "oom.label.context": "Model context (LUMEN_MAX_CTX)",
  "oom.recommend.title": "How to fix",
  "oom.recommend.trim":
    "Reduce what the client sends — shorter prompt or trimmed conversation history. Most reliable.",
  "oom.recommend.ctx":
    "Keep Context max at a memory-safe value. Raising LUMEN_MAX_CTX past what the GPU can actually hold just re-triggers this crash — it does not add memory.",
  "oom.recommend.kv":
    "Turn on KV cache quantization (4-bit) to fit more context in the same memory.",
  "oom.recommend.calc":
    "Use the memory calculator in the Tuning tab to find the safe context for your RAM.",
  "oom.action.tuning": "Open Tuning",
  "oom.action.dismiss": "Dismiss",
};
