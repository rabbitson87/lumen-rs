/**
 * English message catalogue. Keys are dot-namespaced by surface
 * (`tabs.*`, `header.*`, `cards.<card>.*`, `language.*`).
 *
 * Coverage policy: top-level navigation, card headings, action buttons,
 * status banners, and the language picker itself are fully translated.
 * Long-form descriptive hints (TurboQuant explainers, etc.) remain in
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
  "context.turboquant.on": "TurboQuant:",
  "context.turboquant.off": "· baseline KV memory (no compression)",
  "context.recommended": "Recommended max on this Mac",
  "context.recommended.suffix": "tokens",
  "context.warn.turnOnTurboquant":
    "— turn TurboQuant ON to handle longer contexts safely",

  // ── QUANT (Tuning tab) ──────────────────────────────────────────
  "quant.title": "QUANT",
  "quant.titleHint": "(TurboQuant KV cache)",
  "quant.master": "TurboQuant",
  "quant.mode": "TurboQuant mode",
  "quant.mode.off": "Off",
  "quant.mode.on": "On",
  "quant.mode.auto": "Auto",
  "quant.autoThreshold": "Auto threshold (tokens)",
  "quant.qjl": "QJL residual (Stage 2)",
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

  // ── QUANT card tooltips ─────────────────────────────────────────
  "quant.tooltip.master":
    "Master switch for KV-cache quantization (Lloyd-Max + Haar rotation, Stage 1). ON saves ~4–8× KV memory at small accuracy cost. OFF keeps KV in bf16 — recommended only if you see quality issues at long context.",
  "quant.tooltip.mode":
    "Off: never compress KV (fastest short-prompt decode). On: always compress (max long-context memory savings, ~20–56% slower decode). Auto: compress only when this request's prompt is at or above the threshold below — short chats stay fast, long context still saves memory. Per-request decision logged as `[gemma4] tq_auto: ...`.",
  "quant.tooltip.autoThreshold":
    "Prompt-token count at which Auto mode flips TurboQuant ON for the request. Default 4096 — below this, the bf16 sliding cache fits comfortably and full-speed decode wins; above it, TQ's per-step overhead is amortised by the KV bandwidth savings.",
  "quant.tooltip.qjl":
    "Stage-2 unbiased 1-bit correction for the Stage-1 residual: projects (original − reconstructed) into m-dim Gaussian space and packs only the sign. Recovers ~2–3% Top-5 / +0.003 cosine at small extra cost (~m/8 bytes per K/V vector; ~25 MB for Gemma 4 sliding window at m=1024). Requires Stage 1 ON.",
  "quant.tooltip.bits":
    "Lloyd-Max bits per KV channel. 4: highest quality, ~4× smaller than FP16. 3: balanced — recommended default. 2: max compression (~8× smaller), small quality drop. Applies to the sliding-window KV on Gemma 4.",

  // ── CONTEXT card hints ──────────────────────────────────────────
  "context.hint.max.prefix":
    "Max sequence length (tokens). Caps the model's max_position_embeddings when host RAM can't hold the model's native limit (Gemma 4 claims 128K).",
  "context.hint.max.tqOn":
    "Current TurboQuant gives roughly the listed KV compression",
  "context.hint.max.tqOnRealistic": "— realistic on this Mac:",
  "context.hint.max.tqOff":
    "TurboQuant OFF — KV stays bf16, so practical limit on this Mac",
  "context.hint.max.tqOffRealistic": "is",
  "context.hint.max.tqOffFallback": "is much lower than the model's native max",
  "context.hint.max.env": "Env:",
  "context.hint.sliding":
    "Sliding-window attention size. Some layers (Gemma 4: 25 of 30) only attend to the last N tokens instead of the full sequence → bounded KV memory for long contexts. 0 = use the model's built-in default; N>0 overrides it (smaller = less KV, weaker long-range recall).",
  "context.hint.sliding.stacks":
    "Stacks with TurboQuant — sliding bounds which tokens are kept, TurboQuant compresses how they're stored.",
  "context.hint.prefill":
    "Prompt-processing chunk cap. Server rejects prompts longer than this with a \"prompt too large\" error. Larger = accepts long prompts but more peak memory during prefill (attention QK·T = chunk × KV",

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

  // ── QUANT bits comparison hints ─────────────────────────────────
  "quant.hint.tqOff":
    "TurboQuant OFF — KV cache stays bf16 (~5 GB at 11K Korean context on Gemma 4)",
  "quant.hint.smallerVsFp16": "× smaller vs FP16",
  "quant.hint.cosine": "cosine",
  "quant.hint.top5": "Top-5",
  "quant.hint.vs4bit": "vs 4-bit baseline:",
  "quant.hint.kvMemory": "KV memory",
  "quant.hint.baseline": "baseline (highest quality)",

  // ── CONTEXT banner ──────────────────────────────────────────────
  "context.banner.smallerThanBf16": "× smaller than bf16",
  "context.banner.kvCache": "KV cache ~",

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
    "Raw env-var overrides passed to the lumen-server subprocess. Useful for one-off knobs not surfaced in the UI (e.g. LUMEN_GEMMA4_FUSE_EXPERTS, LUMEN_AFFINE4_FORCE_CPU). Keys that shadow a typed UI field are highlighted.",
  "env.row.remove": "Remove",
  "env.row.shadowsPrefix": "⚠ shadows the UI field for",
  "env.add2": "+ Add",
  "env.revert": "Revert",
  "env.applyHint": "Changes apply on next server start.",
  "env.empty2": "No overrides set.",

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
    "8-16 GB is tight. Stick to <2B parameter models. Use 3-bit TurboQuant and disable the wired memory caps (Server card → Disable caps) if you OOM.",
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
