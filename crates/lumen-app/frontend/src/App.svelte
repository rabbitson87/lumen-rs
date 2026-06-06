<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { SvelteMap } from "svelte/reactivity";
  import {
    api,
    onLog,
    onStatus,
    onDownload,
    type PersistentConfig,
    type ModelEntry,
    type ServerStatus,
    type ServerMetrics,
    type LogLine,
    type DownloadProgress,
    type DoctorReport,
    type Catalog,
    type RecommendedModel,
    type SystemInfo,
    type MemoryUsage,
    type CorsMode,
  } from "./lib/api";
  import EnvOverrides from "./lib/EnvOverrides.svelte";
  import MemoryCalculator from "./lib/MemoryCalculator.svelte";
  import DoctorPanel from "./lib/DoctorPanel.svelte";
  import UpdatePanel from "./lib/UpdatePanel.svelte";
  import ApiTabs from "./lib/ApiTabs.svelte";
  import { bytes, duration } from "./lib/format";
  import { i18n, t, LOCALE_LABELS, type Locale } from "./lib/i18n.svelte";

  // Tooltip action: portals tooltip into <body> on hover so it escapes any
  // ancestor overflow / stacking-context clipping (cards near the window edge
  // were truncating absolute-positioned tooltips). Reads `data-tooltip` from
  // the bound element, clamps position into the viewport with an 8 px margin,
  // and flips below the trigger if there's no room above.
  function tooltip(node: HTMLElement) {
    let tip: HTMLElement | null = null;

    function show() {
      const text = node.getAttribute("data-tooltip");
      if (!text) return;
      tip = document.createElement("div");
      tip.className = "tooltip-portal";
      tip.textContent = text;
      document.body.appendChild(tip);

      const margin = 8;
      const rect = node.getBoundingClientRect();
      const tipRect = tip.getBoundingClientRect();

      let top = rect.top - tipRect.height - 8;
      if (top < margin) top = rect.bottom + 8;

      let left = rect.left + rect.width / 2 - tipRect.width / 2;
      const maxLeft = window.innerWidth - tipRect.width - margin;
      left = Math.max(margin, Math.min(maxLeft, left));

      tip.style.top = `${top}px`;
      tip.style.left = `${left}px`;
    }

    function hide() {
      tip?.remove();
      tip = null;
    }

    node.addEventListener("mouseenter", show);
    node.addEventListener("mouseleave", hide);
    node.addEventListener("blur", hide);

    return {
      destroy() {
        hide();
        node.removeEventListener("mouseenter", show);
        node.removeEventListener("mouseleave", hide);
        node.removeEventListener("blur", hide);
      },
    };
  }

  // ── Reactive state ────────────────────────────────────────────────
  let config = $state<PersistentConfig | null>(null);
  let models = $state<ModelEntry[]>([]);
  let status = $state<ServerStatus>({
    state: "stopped",
    pid: null,
    port: 8080,
    host: "127.0.0.1",
    model_id: null,
    uptime_secs: null,
    last_error: null,
  });
  let metrics = $state<ServerMetrics>({
    tokens_per_sec: null,
    ms_per_step: null,
    kv_cache_mb: null,
    requests_per_min: null,
  });
  let logs = $state<LogLine[]>([]);
  let logsOpen = $state(false);
  // Container ref + sticky-bottom state for the log viewer.
  // Auto-scrolls to the latest line unless the user has scrolled up to
  // read older entries (then we leave the scroll alone until they reach
  // the bottom again). Threshold = 30 px below the bottom counts as "at
  // the bottom".
  let logsContainer = $state<HTMLDivElement | undefined>(undefined);
  let logsStickToBottom = $state(true);
  const LOG_STICK_THRESHOLD_PX = 30;
  const LOG_MAX_LINES = 2000;
  let envOpen = $state(false);
  let doctorOpen = $state(false);
  let doctorReport = $state<DoctorReport | null>(null);
  let updateOpen = $state(false);

  type TopTab = "main" | "tuning" | "api" | "debug" | "language";
  let activeTab = $state<TopTab>("main");
  // Memory calculator card (Tuning tab) — collapsed by default.
  let calcOpen = $state(false);
  // SvelteMap (not plain Map) so `.set()`/`.delete()` mutations trigger UI
  // updates without manual reassignment — fixes the case where completed
  // download lines wouldn't disappear after the 3s auto-dismiss timer.
  let downloads = new SvelteMap<string, DownloadProgress>();
  let statusMessage = $state<string | null>(null);
  // Optional inline action attached to the current statusMessage. Used by
  // `savedToast()` to surface a "Restart now" button on the same toast when
  // a config save would otherwise require manual Stop → Start. `null` =
  // plain informational toast (legacy behavior).
  let statusAction = $state<{ label: string; onClick: () => Promise<void> } | null>(null);
  let typedEnvKeys = $state<Set<string>>(new Set());
  let catalog = $state<Catalog>({ families: [], recommended: [], embeddings: [] });
  let systemInfo = $state<SystemInfo | null>(null);
  let memoryUsage = $state<MemoryUsage | null>(null);
  let selectedRecommend = $state<string>("");

  // Set of repo ids whose local SHA differs from HF Hub `main` — driven by
  // `api.checkModelUpdates(...)` after every model-list refresh. Drives the
  // "Update available" badge + Start-button gating.  Cleared/repopulated on
  // every refresh; never persisted to disk (backend owns the cache).
  let outdatedModels = $state<Set<string>>(new Set());

  let visibleModels = $derived(models.filter((m) => m.supported));

  let activeOutdated = $derived(
    config?.active_model != null && outdatedModels.has(config.active_model)
  );

  // True when the configured active model exists on disk but its
  // download is incomplete (truncated shard, missing index, stray
  // .part file). Block the Start button so the server doesn't crash
  // mid-load with a missing-weight error — the user must Re-download
  // from the MODELS card first.
  let activeBroken = $derived.by(() => {
    if (!config?.active_model) return false;
    const m = models.find((x) => x.id === config!.active_model);
    return m != null && !m.ready;
  });

  /// Refresh the outdated-set against HF Hub. Best-effort: any network failure
  /// just leaves the previous state in place (we never auto-flag-as-outdated
  /// on a transient offline check, otherwise users would be blocked from
  /// starting the server on flaky wifi). Called from `onMount`, after every
  /// `setActive`, and after each download completes.
  async function refreshOutdated() {
    if (models.length === 0) {
      outdatedModels = new Set();
      return;
    }
    try {
      const statuses = await api.checkModelUpdates(models.map((m) => m.id));
      const next = new Set<string>();
      for (const s of statuses) {
        if (s.needs_update) next.add(s.repo_id);
      }
      outdatedModels = next;
    } catch (e) {
      console.warn("check_model_updates failed:", e);
    }
  }

  // Active model + derived hints (bits, recommended memory caps).
  let activeModel = $derived.by(() => {
    if (!config?.active_model) return null;
    return models.find((m) => m.id === config!.active_model) ?? null;
  });

  // Bit-width detected from the active model id. The base model determines its
  // own quantization — TurboQuant's `bits` knob is a KV-cache concept but in
  // practice it shadows the model's native quant for display purposes.
  let modelBitsLabel = $derived.by(() => {
    const id = config?.active_model?.toLowerCase() ?? "";
    if (!id) return null;
    if (id.includes("mxfp4")) return "4 (MXFP4)";
    const m = id.match(/(\d)\s*-?\s*bit/);
    if (m) return m[1];
    return "16 (fp16)";
  });

  // Conservative KV-cache estimate: ~1 GB per 8K tokens (most decoder-only LLMs
  // with GQA + f16 cache fall in that range).
  function kvEstimateGb(ctxMax: number): number {
    return Math.max(0, ctxMax / 8192);
  }

  // KV compression ratio under the current simple-Q4 config. bf16=16 bits,
  // quantized = `bits` bits/scalar (group_size=64 has negligible overhead
  // for this ratio estimate). Off → 1.0 (no compression).
  let kvCompressRatio = $derived.by(() => {
    if (!config || config.quant.kv_mode === "off") return 1.0;
    const bits = config.quant.bits;
    return bits > 0 ? 16 / bits : 1.0;
  });

  let kvQuantStateLabel = $derived.by(() => {
    if (!config) return "—";
    const mode = config.quant.kv_mode;
    if (mode === "off") return "OFF (bf16 KV)";
    const stage = `${config.quant.bits}-bit`;
    const suffix = mode === "auto"
      ? ` (auto ≥${config.quant.kv_auto_threshold_tokens}t)`
      : "";
    return `${stage}${suffix}`;
  });

  // Realistic max-context budget. Assumes ~2 GB free for KV after model +
  // overhead on the host Mac, scaled by the TurboQuant compression ratio.
  // Returns rounded K-tokens (e.g. 32 for 32 768 tokens).
  let realisticMaxCtxK = $derived.by(() => {
    if (!systemInfo || !activeModel) return null;
    const modelGb = activeModel.size_bytes / 1024 ** 3;
    const overheadGb = 3; // server + tokenizer + activations
    const freeForKv = Math.max(0.5, systemInfo.ram_gb - modelGb - overheadGb);
    const bf16TokensPerGb = 8192; // baseline: ~1 GB / 8K tokens at bf16
    const realisticTokens = freeForKv * bf16TokensPerGb * kvCompressRatio;
    return Math.round(realisticTokens / 1024);
  });
  // Precise float used for display (3 decimal places); the persisted
  // `memory_limit_gb` ServerConfig field is still usize so we ceil() at save.
  let recommendedMemoryGbExact = $derived.by(() => {
    if (!activeModel || !config) return null;
    const modelGb = activeModel.size_bytes / 1024 ** 3;
    const ctx = config.context.max ?? 81920;
    return modelGb + 2 + kvEstimateGb(ctx);
  });
  let recommendedMemoryGb = $derived.by(() =>
    recommendedMemoryGbExact != null ? Math.ceil(recommendedMemoryGbExact) : null,
  );
  // Wired = exact model size (byte precision via LUMEN_WIRED_LIMIT_BYTES on
  // the server side; the UI surfaces the float-GB label).
  let recommendedWiredGb = $derived.by(() => {
    if (!activeModel) return null;
    return activeModel.size_bytes / 1024 ** 3;
  });
  let recommendedWiredLabel = $derived.by(() => {
    if (!activeModel) return null;
    return `${(activeModel.size_bytes / 1024 ** 3).toFixed(3)} GB (exact)`;
  });
  // Cache pool is for transient activations — a flat 2 GB is enough on every
  // Apple Silicon machine we've measured.
  let recommendedCacheGb = $derived.by(() => (recommendedMemoryGb ? 2 : null));

  let availableRecommended = $derived.by(() => {
    const have = new Set(models.map((m) => m.id));
    const ram = systemInfo?.ram_gb ?? Infinity;
    // Sort: fits-in-RAM first (ascending min_ram_gb), then over-budget last.
    return catalog.recommended
      .filter((r) => !have.has(r.id))
      .slice()
      .sort((a, b) => {
        const aOk = a.min_ram_gb <= ram ? 0 : 1;
        const bOk = b.min_ram_gb <= ram ? 0 : 1;
        if (aOk !== bOk) return aOk - bOk;
        return a.min_ram_gb - b.min_ram_gb;
      });
  });

  let doctorCounts = $derived.by(() => {
    if (!doctorReport) return { pass: 0, warn: 0, fail: 0 };
    let pass = 0, warn = 0, fail = 0;
    for (const c of doctorReport.checks) {
      if (c.status === "pass") pass++;
      else if (c.status === "warn") warn++;
      else fail++;
    }
    return { pass, warn, fail };
  });

  // ── Lifecycle ────────────────────────────────────────────────────
  let unlistenLog: (() => void) | undefined;
  let unlistenStatus: (() => void) | undefined;
  let unlistenDownload: (() => void) | undefined;
  let pollHandle: ReturnType<typeof setInterval> | undefined;

  onMount(async () => {
    config = await api.getConfig();
    models = await api.listModels();
    // Background revision check — fire-and-forget so app boot isn't blocked
    // on HF Hub round-trip. Result drives the "Update available" UI state.
    refreshOutdated();
    status = await api.serverStatus();
    metrics = await api.serverMetrics();
    typedEnvKeys = new Set(await api.typedEnvKeys());
    try {
      catalog = await api.getCatalog();
    } catch (e) {
      console.error("get_catalog failed:", e);
    }
    try {
      systemInfo = await api.getSystemInfo();
    } catch (e) {
      console.error("get_system_info failed:", e);
    }
    // Auto-run preflight diagnostics. Open the Doctor tab automatically if
    // anything is blocking or degraded so the issue is visible without a click.
    try {
      const r = await api.doctorRun();
      doctorReport = r;
      if (r.overall === "blocked" || r.overall === "degraded") {
        doctorOpen = true;
        logsOpen = false;
        envOpen = false;
      }
    } catch (e) {
      console.error("doctor run failed:", e);
    }

    // Heal any stale memory caps left over from a previous session that
    // pre-dates the auto-tune feature (e.g. system-default 16/2/20 saved
    // before an active model existed). No-op if caps already match the
    // tuned recommendation or no model is active.
    await syncTunedMemoryCaps();

    unlistenLog = await onLog((lines) => {
      // One reactive update per batch (not per line): trim once after appending.
      logs = [...logs, ...lines].slice(-LOG_MAX_LINES);
    });
    unlistenStatus = await onStatus((s) => {
      status = s;
    });
    unlistenDownload = await onDownload((p) => {
      const key = `${p.repo_id}/${p.file}`;
      downloads.set(key, p);
      if (p.done) {
        api.listModels().then((m) => {
          models = m;
          refreshOutdated();
        });
        // Auto-clear completed lines after 3s so the progress panel
        // doesn't crowd up during back-to-back downloads. Re-check `done`
        // first so an in-flight resume on the same key (rare but possible
        // with retry logic) doesn't drop a now-active entry.
        setTimeout(() => {
          if (downloads.get(key)?.done) {
            downloads.delete(key);
          }
        }, 3000);
      } else if (p.downloaded_bytes === 0) {
        // The backend emits a `0 bytes / done:false` placeholder right before
        // each HTTP GET, then `continue`s silently on 404 (common for the
        // optional metadata files in the default probe list: tokenizer.model,
        // special_tokens_map.json, added_tokens.json, preprocessor_config.json
        // — most modern repos ship a subset). Those entries never get a
        // follow-up event and would sit in the UI forever. Auto-remove after
        // 10s IFF still at 0 bytes and not done — any real file is throttled
        // to emit chunks every 512 KB, so a true active download will bump
        // downloaded_bytes well before the timeout fires.
        setTimeout(() => {
          const curr = downloads.get(key);
          if (curr && !curr.done && curr.downloaded_bytes === 0) {
            downloads.delete(key);
          }
        }, 10000);
      }
    });

    try {
      memoryUsage = await api.getMemoryUsage();
    } catch (e) {
      console.error("get_memory_usage failed:", e);
    }

    pollHandle = setInterval(async () => {
      if (status.state === "running") {
        metrics = await api.serverMetrics();
      }
      try {
        memoryUsage = await api.getMemoryUsage();
      } catch {}
    }, 2000);
  });

  onDestroy(() => {
    unlistenLog?.();
    unlistenStatus?.();
    unlistenDownload?.();
    if (pollHandle) clearInterval(pollHandle);
  });

  // Track scroll position so auto-scroll only fires when the user is
  // already at (or near) the bottom. Detached from the DOM until the
  // logs panel is opened — `logsContainer` is `undefined` while
  // `logsOpen=false`.
  function onLogsScroll() {
    if (!logsContainer) return;
    const el = logsContainer;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    logsStickToBottom = distanceFromBottom < LOG_STICK_THRESHOLD_PX;
  }

  // Auto-scroll to bottom when new logs arrive, but only if the user
  // hasn't scrolled up. `requestAnimationFrame` defers until after the
  // DOM has been updated with the new line. Also fires when the panel
  // re-opens so the user lands at the latest entry rather than wherever
  // the scrollTop was last frame.
  $effect(() => {
    // Reactive deps: re-run whenever a log is appended OR the panel
    // opens.
    logs.length;
    logsOpen;
    if (!logsOpen || !logsStickToBottom || !logsContainer) return;
    const el = logsContainer;
    requestAnimationFrame(() => {
      el.scrollTop = el.scrollHeight;
    });
  });

  // ── Mutators ─────────────────────────────────────────────────────
  async function toggleServer() {
    statusMessage = null;
    try {
      if (status.state === "running" || status.state === "starting") {
        status = await api.stopServer();
      } else {
        status = await api.startServer();
      }
    } catch (e) {
      statusMessage = String(e);
    }
  }

  async function setActive(id: string) {
    if (!config) return;
    config = await api.setActiveModel(id);
    // Without this, switching models leaves the previous model's caps in
    // place (e.g. 16 GB wired saved for a 16 GB model is wildly oversized
    // for an 11 GB one). Re-tune to track the new model + current ctx.
    await syncTunedMemoryCaps();
  }

  /// Sync the persisted Metal memory caps to the tuned recommendation
  /// derived from the active model + current context. No-op when no active
  /// model is set (system defaults stay in place) or values already match.
  /// `wired_limit_gb = null` so the backend emits byte-exact
  /// `LUMEN_WIRED_LIMIT_BYTES` from `active_model.size_bytes` instead of a
  /// GB-rounded ceiling that could truncate a 14.45 GB model to 14 GB.
  async function syncTunedMemoryCaps() {
    if (!config || !activeModel) return;
    if (recommendedMemoryGb == null || recommendedCacheGb == null) return;
    const needsUpdate =
      config.server.wired_limit_gb !== null ||
      config.server.cache_limit_gb !== recommendedCacheGb ||
      config.server.memory_limit_gb !== recommendedMemoryGb ||
      config.server.disable_wired_limit;
    if (!needsUpdate) return;
    config.server.wired_limit_gb = null;
    config.server.cache_limit_gb = recommendedCacheGb;
    config.server.memory_limit_gb = recommendedMemoryGb;
    config.server.disable_wired_limit = false;
    config = await api.updateServerConfig(config.server);
  }

  // One-click "Stop → wait → Start" — used by the inline action button on
  // the savedToast when the server is running. Polls `serverStatus` until
  // it drains to `stopped` (or `errored`) before issuing `startServer` so
  // the new env vars from `spawn_server_command` are actually used. Without
  // the drain wait, `startServer` would race the still-tearing-down
  // supervisor and surface a stale state.
  async function restartServer() {
    statusMessage = t("config.restarting");
    statusAction = null;
    try {
      if (status.state === "running" || status.state === "starting") {
        status = await api.stopServer();
        let waited = 0;
        const POLL_MS = 200;
        const TIMEOUT_MS = 30_000;
        while (
          status.state !== "stopped" &&
          status.state !== "errored" &&
          waited < TIMEOUT_MS
        ) {
          await new Promise((r) => setTimeout(r, POLL_MS));
          status = await api.serverStatus();
          waited += POLL_MS;
        }
      }
      status = await api.startServer();
      statusMessage = t("config.restarted");
      setTimeout(() => (statusMessage = null), 2000);
    } catch (e) {
      statusMessage = String(e);
      setTimeout(() => (statusMessage = null), 4000);
    }
  }

  // Shared post-save toast. Two modes:
  //   * server running   → "Saved. Restart to apply" + inline [Restart now]
  //                        button (calls `restartServer`).
  //   * server stopped   → plain "Saved." (next start picks up the change
  //                        naturally — no restart needed).
  // Without this, users who change TurboQuant / KV-quant / ctx caps while
  // the server is running see the UI update but the running process keeps
  // the OLD env vars until manual Stop → Start. The QUANT / CONTEXT /
  // SERVER cards previously had no hint at all (only env_overrides did).
  function savedToast() {
    if (status.state === "running" || status.state === "starting") {
      statusMessage = t("config.savedRestartHint");
      statusAction = { label: t("config.restartNow"), onClick: restartServer };
      setTimeout(() => {
        if (statusMessage === t("config.savedRestartHint")) {
          statusMessage = null;
          statusAction = null;
        }
      }, 6000);
    } else {
      statusMessage = t("config.saved");
      statusAction = null;
      setTimeout(() => {
        if (statusMessage === t("config.saved")) {
          statusMessage = null;
        }
      }, 2000);
    }
  }

  async function saveServer() {
    if (!config) return;
    config = await api.updateServerConfig(config.server);
    savedToast();
  }

  async function saveQuant() {
    if (!config) return;
    config = await api.updateQuantConfig(config.quant);
    savedToast();
  }

  async function saveContext() {
    if (!config) return;
    config = await api.updateContextConfig(config.context);
    // ctx affects KV-cache headroom in the tuned memory recommendation
    // (~1 GB per 8K tokens). Re-sync so saved caps follow.
    await syncTunedMemoryCaps();
    savedToast();
  }

  async function resetMemoryCaps() {
    if (!config) return;
    if (activeModel && recommendedMemoryGb) {
      // Wired = exact model size — clear `wired_limit_gb` so the backend
      // emits `LUMEN_WIRED_LIMIT_BYTES=<active_model.size_bytes>` automatically
      // (see commands.rs::start_server). Cache + memory stay GB-rounded.
      config.server.wired_limit_gb = null;
      config.server.cache_limit_gb = recommendedCacheGb;
      config.server.memory_limit_gb = recommendedMemoryGb;
      config.server.disable_wired_limit = false;
      config = await api.updateServerConfig(config.server);
      statusMessage = `Memory tuned for ${activeModel.id.split("/").pop()} + ctx ${config.context.max}`;
    } else {
      config = await api.resetMemoryCaps();
      statusMessage = `Memory caps reset to ${systemInfo?.ram_gb ?? "?"} GB profile`;
    }
    setTimeout(() => (statusMessage = null), 2000);
  }

  // Apply the memory calculator's previewed config to the real tuning state.
  // KV mode + bits land on the *structured* quant config so the QUANT card
  // reflects them; chunk / prefix / the Qwen3.6 TQ-KV activation live only as
  // env overrides (no dedicated card). Both persist via their own command and a
  // single restart toast — env is read once at spawn, so Stop → Start applies.
  async function applyCalcSelection(sel: {
    kvOn: boolean;
    bits: number;
    chunk: number;
    prefix: boolean;
  }) {
    if (!config) return;
    let next = await api.updateQuantConfig({
      ...config.quant,
      kv_mode: sel.kvOn ? "on" : "off",
      bits: sel.bits,
    });
    const envPatch: Record<string, string> = {
      LUMEN_QWEN35_TQ_KV: sel.kvOn ? "1" : "0",
      LUMEN_QWEN35_PREFILL_CHUNK: String(sel.chunk),
      LUMEN_MLX_PREFIX_CACHE: sel.prefix ? "1" : "0",
    };
    if (sel.kvOn) envPatch.LUMEN_QWEN35_TQ_KV_BITS = String(sel.bits);
    next = await api.updateEnvOverrides({ ...next.env_overrides, ...envPatch });
    config = next;
    savedToast();
  }

  async function saveEnvOverrides(next: Record<string, string>) {
    if (!config) return;
    config = await api.updateEnvOverrides(next);
    // Same restart semantics as the QUANT / CONTEXT / SERVER cards: env
    // overrides are read once at server spawn, so a running process keeps the
    // OLD values until Stop → Start. Surface the actionable "Restart now"
    // toast instead of saving silently (which reads as "nothing happened").
    savedToast();
  }

  async function startDownload() {
    const id = selectedRecommend;
    if (!id) return;
    try {
      await api.downloadModel(id);
      selectedRecommend = "";
    } catch (e) {
      statusMessage = String(e);
    }
  }

  // Confirmation modal state. Used by both chat-model row Delete buttons
  // (kind="chat") and the embedding picker's Delete button (kind="embedding").
  // The kind controls labelling + post-delete cleanup (clearing the active
  // embedding_model_id from config when its weights are removed).
  let confirmDel = $state<{ id: string; kind: "chat" | "embedding" } | null>(null);
  let confirmDelBusy = $state(false);

  function removeModel(id: string) {
    confirmDel = { id, kind: "chat" };
  }

  function deleteEmbedding(id: string) {
    confirmDel = { id, kind: "embedding" };
  }

  async function performConfirmedDelete() {
    if (!confirmDel || confirmDelBusy) return;
    const { id, kind } = confirmDel;
    confirmDelBusy = true;
    try {
      await api.deleteModel(id);
      models = await api.listModels();
      refreshOutdated();
      // If we just deleted the currently-selected embedding, clear it from
      // config so /v1/embeddings doesn't keep pointing at a missing dir.
      if (
        kind === "embedding" &&
        config &&
        config.server.embedding_model_id === id
      ) {
        config.server.embedding_model_id = null;
        await saveServer();
      }
      confirmDel = null;
    } catch (e) {
      statusMessage = `Delete failed: ${e}`;
    } finally {
      confirmDelBusy = false;
    }
  }

  function cancelConfirmedDelete() {
    if (confirmDelBusy) return;
    confirmDel = null;
  }

  /// "Update" button handler for an out-of-date installed model.  Deletes
  /// the local directory + triggers a fresh download against the current
  /// Hub `main` SHA.  Re-uses the existing download path so the user sees
  /// the same progress UI; SHA marker is rewritten at the end.
  async function updateOutdated(id: string) {
    if (!confirm(
      `Update ${id}? Old weights will be removed and the latest version downloaded.`
    )) return;
    try {
      await api.deleteModel(id);
      models = await api.listModels();
      await api.downloadModel(id, null);
      // listModels + refreshOutdated happen on download completion (see the
      // onDownload listener above) — no explicit refresh needed here.
    } catch (e) {
      statusMessage = `Update failed: ${e}`;
    }
  }

  // Re-download for a model whose previous download was interrupted
  // (truncated shard, missing index, etc.). Wipes any stale .part / partial
  // files and starts the download fresh. The downloader itself will skip
  // any file that already exists at its full advertised size — so resumed
  // model dirs only re-transfer the broken / missing pieces.
  async function repairModel(id: string) {
    if (!confirm(
      `Re-download ${id}? Existing files in the model directory will be wiped first to ensure a clean state.`
    )) return;
    try {
      await api.deleteModel(id);
      models = await api.listModels();
      await api.downloadModel(id, null);
    } catch (e) {
      statusMessage = `Re-download failed: ${e}`;
    }
  }

  // ── Shared utility-class chains (Tailwind 4) ─────────────────────
  // Declared as plain string consts because Svelte's `{@const}` can only
  // live at the top of `{#if}`/`{#each}`/etc. — not at the top of `<main>`.
  // Hoisting here keeps the template DRY while letting the same chain be
  // reused across cards in different conditional branches.
  const cardBase = "bg-panel border border-border rounded-[10px] px-5 py-4.5 min-h-0";
  const cardH2 = "m-0 mb-3.5 text-xs font-semibold tracking-[0.08em] text-text-dim uppercase";
  const kvRow = "grid grid-cols-[120px_1fr] items-center gap-2.5 py-1.5";
  const helpIcon = "inline-flex items-center justify-center w-3.5 h-3.5 ml-1 rounded-full border border-border text-text-dim text-[10px] leading-none cursor-help select-none transition-[color,border-color] duration-120 hover:text-text hover:border-text";
  const toggleLabel = "flex items-center gap-2 cursor-pointer";
  const tabBase = "bg-transparent border-0 border-b-2 px-4 py-2 text-[13px] rounded-none transition-[color,border-color] duration-120";
  const tabActive = "text-text border-b-accent font-medium";
  const tabIdle = "text-text-dim border-b-transparent hover:text-text";
  const ctxHint = "dim -mt-0.5 mb-2 ml-32.5 text-[11px] leading-[1.55]";
  const inlineCode = "px-1 bg-bg border border-border rounded-[3px] text-[10.5px]";
  const cardSection = "mt-4 mb-1.5 text-[11px] font-semibold tracking-[0.06em] text-text-dim uppercase border-t border-border pt-2.5";
  const sectionGrid = "grid grid-cols-3 gap-x-6 gap-y-0";
  const colSpanFull = "col-span-full";
  const ramTunedHint = "dim text-[10px] font-normal normal-case tracking-normal ml-2";
  const resetMemBtn = "ml-2 px-2 py-0.5 text-[10px] font-normal normal-case tracking-normal";
  const debugHint = "dim col-span-full -mt-0.5 mb-1.5 ml-32.5 text-[11px] leading-[1.55]";
  const panelScroll = "max-h-[40vh] overflow-y-auto bg-panel";
  // Vertical-divider tabs: each footer tab carries only a right border
  // (`border-r border-border`) so they line up cleanly with the container's
  // own `border-t` and don't show any bottom/top notches against the nav
  // row above. `last:border-r-0` keeps the rightmost tab's edge clean.
  // `!rounded-none` (Tailwind 4 important prefix) overrides the unlayered
  // `button { border-radius: 6px }` rule in app.css — without it the
  // hover/active background's rounded corners pushed past the divider
  // lines and produced visible notches.
  //
  // Active state is signalled inline (text color → accent) rather than via
  // an outline-style background tint or underline. Hover keeps `bg-panel-2`
  // as a separate cue so hovering an active tab still gives feedback
  // without two signals competing for the same channel.
  const footerTabBase = "text-left !border-0 !border-r !border-border last:!border-r-0 bg-transparent px-3.5 py-1 text-xs !rounded-none hover:bg-panel-2";
  const footerTabActive = "text-accent";
</script>

<!-- ── Top bar ──────────────────────────────────────────────────── -->
<header class="flex items-center gap-4 px-4 py-2.5 bg-panel border-b border-border shrink-0 sticky top-0 z-10">
  <div class="font-semibold text-sm text-accent">● Lumen</div>
  <div class="flex items-center gap-1.5">
    <span class="dot {status.state}"></span>
    <span class="mono">{t(`status.${status.state}`)}</span>
    {#if status.state === "running"}
      <span class="dim mono">:{status.port}</span>
      {#if status.uptime_secs != null}
        <span class="dim">·</span>
        <span class="dim mono">{duration(status.uptime_secs)}</span>
      {/if}
    {/if}
    {#if status.last_error}
      <span class="text-err" title={status.last_error}>{t("header.statusError")}</span>
    {/if}
  </div>
  <div class="ml-auto flex items-center gap-2.5">
    {#if statusMessage}
      <span class="dim">{statusMessage}</span>
      {#if statusAction}
        <button
          class="text-[11px] px-2 py-0.5 border border-accent text-accent rounded-md hover:bg-accent/6 transition-colors"
          onclick={statusAction.onClick}
        >{statusAction.label}</button>
      {/if}
    {/if}
    {#if memoryUsage}
      {@const usedGb = memoryUsage.used_bytes / 1024 ** 3}
      {@const totalGb = memoryUsage.total_bytes / 1024 ** 3}
      {@const pct = (usedGb / totalGb) * 100}
      {@const memHot = pct >= 92}
      {@const memWarn = pct >= 80 && pct < 92}
      <span
        class={`mono inline-flex items-center gap-1.5 text-[11px] px-2 py-0.5 border rounded-md bg-panel ${
          memHot ? "text-err border-err" :
          memWarn ? "text-warn border-warn" :
          "text-text-dim border-border"
        }`}
        title="System memory: {usedGb.toFixed(1)} / {totalGb.toFixed(0)} GB ({pct.toFixed(0)}%) — wired + active + compressor"
      >
        <span class="inline-block w-12 h-1.5 bg-border rounded-[3px] overflow-hidden">
          <span
            class={`block h-full transition-[width] duration-400 ease-out ${
              memHot ? "bg-err" : memWarn ? "bg-warn" : "bg-text-dim"
            }`}
            style="width: {Math.min(100, pct).toFixed(1)}%"
          ></span>
        </span>
        {usedGb.toFixed(1)}/{totalGb.toFixed(0)} GB
      </span>
    {/if}
    <button
      class="inline-flex items-center gap-1.5 text-xs"
      onclick={() => {
        doctorOpen = !doctorOpen;
        if (doctorOpen) {
          logsOpen = false;
          envOpen = false;
        }
      }}
      title={doctorReport ? `${doctorCounts.pass} pass · ${doctorCounts.warn} warn · ${doctorCounts.fail} fail` : "Run preflight checks"}
    >
      <span class={`w-2 h-2 rounded-full ${
        doctorReport?.overall === "healthy" ? "bg-ok shadow-[0_0_6px_var(--color-ok)]" :
        doctorReport?.overall === "degraded" ? "bg-warn" :
        doctorReport?.overall === "blocked" ? "bg-err shadow-[0_0_6px_var(--color-err)]" :
        "bg-text-dim"
      }`}></span>
      {t("header.doctor")}
      {#if doctorReport && (doctorCounts.warn > 0 || doctorCounts.fail > 0)}
        <span class="text-text-dim text-[11px] ml-0.5 mono">
          {#if doctorCounts.fail > 0}{doctorCounts.fail}✗{/if}
          {#if doctorCounts.warn > 0}{doctorCounts.warn}!{/if}
        </span>
      {/if}
    </button>
    <button
      class={status.state === "running" || status.state === "starting" ? "danger" : "primary"}
      onclick={toggleServer}
      disabled={status.state === "starting" || status.state === "stopping" ||
        (status.state !== "running" && (activeOutdated || activeBroken))}
      title={status.state === "running"
        ? ""
        : activeBroken
          ? t("header.title.brokenActive")
          : activeOutdated
            ? t("header.title.outdatedActive")
            : ""}
    >
      {status.state === "running" || status.state === "starting" ? t("header.stop") : t("header.start")}
    </button>
  </div>
</header>

<!-- ── Tab bar (top-level grouping) ─────────────────────────────── -->
<nav class="flex gap-1 px-4 py-1.5 bg-bg border-b border-border sticky top-13.25 z-9">
  <button
    class={`${tabBase} ${activeTab === "main" ? tabActive : tabIdle}`}
    onclick={() => (activeTab = "main")}
  >{t("tabs.main")}</button>
  <button
    class={`${tabBase} ${activeTab === "tuning" ? tabActive : tabIdle}`}
    onclick={() => (activeTab = "tuning")}
  >{t("tabs.tuning")}</button>
  <button
    class={`${tabBase} ${activeTab === "api" ? tabActive : tabIdle}`}
    onclick={() => (activeTab = "api")}
  >{t("tabs.api")}</button>
  <button
    class={`${tabBase} ${activeTab === "debug" ? tabActive : tabIdle}`}
    onclick={() => (activeTab = "debug")}
  >{t("tabs.debug")}</button>
  <button
    class={`${tabBase} ${activeTab === "language" ? tabActive : tabIdle}`}
    onclick={() => (activeTab = "language")}
  >{t("tabs.language")}</button>
</nav>

<!-- ── Card grid ───────────────────────────────────────────────── -->
<main class={`grid gap-4 p-4.5 ${activeTab === "tuning" ? "grid-cols-6" : "grid-cols-3"}`}>
  {#if activeTab === "tuning"}
  <!-- QUANT -->
  <section class="{cardBase} col-span-3">
    <h2 class={cardH2}>{t("quant.title")} <span class="dim">{t("quant.titleHint")}</span></h2>
    {#if config}
      <div class={kvRow}>
        <span class="dim">
          {t("quant.mode")}
          <span
            class={helpIcon}
            use:tooltip
            data-tooltip={t("quant.tooltip.mode")}
          >?</span>
        </span>
        <div class="flex gap-1">
          {#each [["off","quant.mode.off"],["on","quant.mode.on"],["auto","quant.mode.auto"]] as [v, key]}
            <button
              class={`px-2.5 py-1 min-w-12 ${config.quant.kv_mode === v ? "primary" : ""}`}
              onclick={() => {
                if (!config) return;
                config.quant.kv_mode = v as "off" | "on" | "auto";
                saveQuant();
              }}
            >{t(key)}</button>
          {/each}
        </div>
      </div>
      {#if config.quant.kv_mode === "auto"}
        <div class={kvRow}>
          <span class="dim">
            {t("quant.autoThreshold")}
            <span
              class={helpIcon}
              use:tooltip
              data-tooltip={t("quant.tooltip.autoThreshold")}
            >?</span>
          </span>
          <input
            class="mono w-24 text-right"
            type="number"
            min="1"
            max="1000000"
            step="1024"
            value={config.quant.kv_auto_threshold_tokens}
            onchange={(e) => {
              if (!config) return;
              const v = parseInt((e.currentTarget as HTMLInputElement).value, 10);
              if (!isNaN(v) && v >= 1) {
                config.quant.kv_auto_threshold_tokens = v;
                saveQuant();
              }
            }}
          />
        </div>
      {/if}
      <div class={kvRow}>
        <span class="dim">
          {t("quant.bits")}
          <span
            class={helpIcon}
            use:tooltip
            data-tooltip={t("quant.tooltip.bits")}
          >?</span>
        </span>
        <div class="flex gap-1">
          {#each [3, 4, 6, 8] as b}
            <button
              class={`px-2.5 py-1 min-w-9 ${config.quant.bits === b ? "primary" : ""}`}
              disabled={config.quant.kv_mode === "off"}
              onclick={() => {
                if (!config) return;
                config.quant.bits = b;
                saveQuant();
              }}
            >{b}</button>
          {/each}
        </div>
      </div>
      <div class="dim flex flex-col gap-0.5 mt-1 ml-32.5 text-[11px] leading-normal">
        {#if config.quant.kv_mode === "off"}
          <span class="text-warn">{t("quant.hint.kvOff")}</span>
        {:else if config.quant.bits === 3}
          <span><b class="text-text">3-bit</b> · ~5.3{t("quant.hint.smallerVsFp16")}</span>
          <span class="text-text-dim">{t("quant.hint.lowestMemory")}</span>
        {:else if config.quant.bits === 4}
          <span><b class="text-text">4-bit</b> · ~4{t("quant.hint.smallerVsFp16")}</span>
          <span class="text-text-dim">{t("quant.hint.baseline")}</span>
        {:else if config.quant.bits === 6}
          <span><b class="text-text">6-bit</b> · ~2.7{t("quant.hint.smallerVsFp16")}</span>
          <span class="text-text-dim">{t("quant.hint.balancedQuality")}</span>
        {:else if config.quant.bits === 8}
          <span><b class="text-text">8-bit</b> · 2{t("quant.hint.smallerVsFp16")}</span>
          <span class="text-text-dim">{t("quant.hint.highestQuality")}</span>
        {/if}
      </div>
    {/if}
  </section>

  <!-- METRICS -->
  <section class="{cardBase} col-span-3">
    <h2 class={cardH2}>{t("metrics.title")}</h2>
    <div class={kvRow}>
      <span class="dim">{t("metrics.tokensPerSec")}</span>
      <span class="mono">{metrics.tokens_per_sec?.toFixed(1) ?? "—"}</span>
    </div>
    <div class={kvRow}>
      <span class="dim">{t("metrics.msPerStep")}</span>
      <span class="mono">{metrics.ms_per_step?.toFixed(2) ?? "—"}</span>
    </div>
    <div class={kvRow}>
      <span class="dim">{t("metrics.kvCache")}</span>
      <span class="mono">{metrics.kv_cache_mb != null ? `${metrics.kv_cache_mb} MB` : "—"}</span>
    </div>
    <div class={kvRow}>
      <span class="dim">{t("metrics.requestsPerMin")}</span>
      <span class="mono">{metrics.requests_per_min ?? "—"}</span>
    </div>
  </section>

  <!-- CONTEXT -->
  <section class="{cardBase} col-span-6">
    <h2 class={cardH2}>{t("context.title")} <span class="dim">{t("context.titleHint")}</span></h2>

    <!-- Memory calculator: borderless disclosure at the top of the context card -->
    <button
      class="flex items-center gap-2 w-full text-left bg-transparent p-0 m-0 mb-2 pb-2 border-b border-border hover:text-text"
      onclick={() => (calcOpen = !calcOpen)}
    >
      <span class="dim text-[11px] font-semibold tracking-[0.06em] uppercase">{t("memcalc.title")}</span>
      <span class="dim ml-auto text-[11px]">{calcOpen ? "▾" : "▸"}</span>
    </button>
    {#if calcOpen}
      <div class="mb-3.5">
        <MemoryCalculator
          modelId={config?.active_model ?? null}
          env={config?.env_overrides ?? {}}
          onApply={applyCalcSelection}
        />
      </div>
    {/if}

    {#if config}
      <div class={`-mt-1 mb-3 px-2.5 py-2 bg-panel-2 border border-border border-l-[3px] rounded text-xs leading-[1.55] flex flex-col gap-0.5 ${
        config.quant.kv_mode !== "off" ? "border-l-accent" : "border-l-warn"
      }`}>
        <div>
          <span class="dim">{t("context.kvQuant.label")}</span>
          <b class="text-text">{kvQuantStateLabel}</b>
          {#if config.quant.kv_mode !== "off"}
            <span class="dim">· {t("context.banner.kvCache")}{kvCompressRatio.toFixed(1)}{t("context.banner.smallerThanBf16")}</span>
          {:else}
            <span class="dim">{t("context.kvQuant.offHint")}</span>
          {/if}
        </div>
        {#if realisticMaxCtxK != null}
          <div class="dim">
            {t("context.recommended")} ({systemInfo?.ram_gb} GB):
            <b class="text-text">~{realisticMaxCtxK}K {t("context.recommended.suffix")}</b>
            {#if config.quant.kv_mode === "off"}
              <span class="text-warn">{t("context.warn.turnOnKvQuant")}</span>
            {/if}
          </div>
        {/if}
      </div>

      <div class={kvRow}>
        <span class="dim">{t("context.max")}</span>
        <input
          type="number"
          min="512"
          step="512"
          bind:value={config.context.max}
          onchange={saveContext}
        />
      </div>
      <div class={ctxHint}>
        {t("context.hint.max.prefix")}
        {#if config.quant.kv_mode !== "off"}
          {t("context.hint.max.kvOn")}
          {#if realisticMaxCtxK != null}
            {t("context.hint.max.kvOnRealistic")} <b class="text-text">~{realisticMaxCtxK}K</b>.
          {:else}.{/if}
        {:else}
          {t("context.hint.max.kvOff")}
          {#if realisticMaxCtxK != null}<b class="text-text">~{realisticMaxCtxK}K</b>{:else}{t("context.hint.max.kvOffFallback")}{/if}.
        {/if}
        {t("context.hint.max.env")} <code class={inlineCode}>LUMEN_MAX_CTX</code>.
      </div>

      <div class={kvRow}>
        <span class="dim">{t("context.sliding")}</span>
        <input
          type="number"
          min="0"
          step="256"
          bind:value={config.context.sliding}
          onchange={saveContext}
        />
      </div>
      <div class={ctxHint}>
        {t("context.hint.sliding")}
        {#if config.quant.kv_mode !== "off"}
          {t("context.hint.sliding.kvStacks")}
        {/if}
        {t("context.hint.max.env")} <code class={inlineCode}>LUMEN_SLIDING_WINDOW</code>.
      </div>

      <div class={kvRow}>
        <span class="dim">{t("context.prefill")}</span>
        <input
          type="number"
          min="512"
          step="512"
          bind:value={config.context.prefill}
          onchange={saveContext}
        />
      </div>
      <div class={ctxHint}>
        {t("context.hint.prefill")}{#if config.quant.kv_mode !== "off"}, ~{kvCompressRatio.toFixed(1)}×{/if}).
        {t("context.hint.max.env")} <code class={inlineCode}>LUMEN_PREFILL_CHUNK</code>.
      </div>

      <div class={kvRow}>
        <span class="dim">{t("context.defaultMaxTokens")}</span>
        <input
          type="number"
          min="1"
          step="256"
          bind:value={config.context.default_max_tokens}
          onchange={saveContext}
        />
      </div>
      <div class={ctxHint}>
        {t("context.hint.defaultMaxTokens")}
        {t("context.hint.max.env")} <code class={inlineCode}>LUMEN_DEFAULT_MAX_TOKENS</code>.
      </div>
    {/if}
  </section>

  {/if}

  {#if activeTab === "main"}
  <!-- MODELS -->
  <section class="{cardBase} col-span-3">
    <h2 class={cardH2}>{t("models.title")}</h2>
    <div class="grid grid-cols-2 gap-x-3 gap-y-1.5 mb-3.5">
      {#each visibleModels as m}
        {@const needsUpdate = outdatedModels.has(m.id)}
        {@const isActive = config?.active_model === m.id}
        {@const isBroken = !m.ready}
        <div
          class={`grid grid-cols-[18px_minmax(0,1fr)_90px_auto_auto] items-center gap-3 px-2.5 py-2 rounded-md bg-panel-2 border hover:border-border ${
            !m.supported ? "opacity-55" : ""
          } ${
            isBroken ? "border-err shadow-[0_0_0_1px_var(--color-err)_inset] bg-err/10" :
            needsUpdate && isActive ? "border-warn shadow-[0_0_0_1px_var(--color-warn)_inset] bg-warn/15" :
            needsUpdate ? "border-warn bg-warn/10" :
            isActive ? "border-accent shadow-[0_0_0_1px_var(--color-accent)_inset] bg-accent/[0.06]" :
            "border-transparent"
          }`}
        >
          <span class="mono text-ok font-bold">{isActive ? "✓" : ""}</span>
          <div class="min-w-0 flex flex-col gap-0.5">
            <div class="mono overflow-hidden text-ellipsis whitespace-nowrap text-[13px]">{m.id}</div>
            {#if isBroken}
              <div class="text-[11px] overflow-hidden text-ellipsis whitespace-nowrap text-err">{t("models.broken.label")}</div>
            {:else if needsUpdate}
              <div class="text-[11px] overflow-hidden text-ellipsis whitespace-nowrap text-warn">{t("models.outdated.label")}</div>
            {:else if m.label}
              <div class="text-[11px] overflow-hidden text-ellipsis whitespace-nowrap text-text-dim">{m.label}</div>
            {:else if !m.supported}
              <div class="text-[11px] overflow-hidden text-ellipsis whitespace-nowrap text-warn">{t("models.unsupported.label")}</div>
            {/if}
          </div>
          <span class="mono text-text-dim text-right text-xs">{bytes(m.size_bytes)}</span>
          {#if isBroken}
            <button
              class="primary"
              onclick={() => repairModel(m.id)}
              title={t("models.action.title.redownload")}
            >{t("action.redownload")}</button>
          {:else if needsUpdate}
            <button
              class="primary"
              onclick={() => updateOutdated(m.id)}
              title={t("models.action.title.update")}
            >{t("action.update")}</button>
          {:else}
            <button
              onclick={() => setActive(m.id)}
              disabled={isActive || !m.supported}
              title={!m.supported ? t("models.action.title.unsupported") : ""}
            >{t("action.use")}</button>
          {/if}
          <button class="danger" onclick={() => removeModel(m.id)}>{t("action.delete")}</button>
        </div>
      {/each}
      {#if visibleModels.length === 0 && models.length > 0}
        <p class="dim col-span-full m-0">{t("models.empty.unsupported")}</p>
      {:else if models.length === 0}
        <p class="dim col-span-full m-0">{t("models.empty.none")}</p>
      {/if}
    </div>
    {#if systemInfo}
      <div class="dim mono text-[11px] mb-1.5">
        {t("models.thisMac")} <b class="text-text">{systemInfo.ram_gb} {t("models.thisMac.ramSuffix")}</b> ({systemInfo.arch}) {t("models.thisMac.overflow")}
      </div>
    {/if}
    <div class="grid grid-cols-[minmax(0,1fr)_auto] gap-2.5">
      <select bind:value={selectedRecommend}>
        <option value="" disabled>{t("models.picker.placeholder")}</option>
        {#each availableRecommended as r}
          {@const fits = !systemInfo || r.min_ram_gb <= systemInfo.ram_gb}
          <option value={r.id}>
            {fits ? "" : "⚠ "}{r.label} · {r.approx_size_gb}GB · ≥{r.min_ram_gb}GB RAM
          </option>
        {/each}
        {#if availableRecommended.length === 0}
          <option value="" disabled>{t("models.picker.allDownloaded")}</option>
        {/if}
      </select>
      <button
        class="primary"
        onclick={startDownload}
        disabled={!selectedRecommend}
      >{t("action.download")}</button>
    </div>
    {#if selectedRecommend}
      {@const r = catalog.recommended.find((x) => x.id === selectedRecommend)}
      {#if r}
        <div class="dim mt-1.5 text-[11px] leading-normal">{r.notes}</div>
      {/if}
    {/if}
    {#if downloads.size > 0}
      <div class="mt-2 flex flex-col gap-0.5">
        {#each [...downloads.entries()] as [key, p]}
          <div class="mono grid grid-cols-[16px_1fr_80px] text-xs gap-1.5">
            <span class={p.done ? "text-ok" : "text-text-dim"}>{p.done ? "✓" : "…"}</span>
            <span>{key}</span>
            <span class="dim">{bytes(p.downloaded_bytes)}</span>
          </div>
        {/each}
      </div>
    {/if}
    {#if catalog && catalog.embeddings.length > 0}
      {@const activeEmb = config?.server.embedding_model_id ?? null}
      {@const embDownloaded = activeEmb ? models.some((m) => m.id === activeEmb) : false}
      <div class="mt-3 pt-2.5 border-t border-border flex flex-wrap items-center gap-x-2 gap-y-1 text-xs">
        <span class="dim">{t("models.embedding.label")}</span>
        {#if !activeEmb}
          <span class="dim">{t("models.embedding.disabled")}</span>
        {:else if embDownloaded}
          <span class="mono text-ok">✓</span>
          <span class="mono">{activeEmb}</span>
        {:else}
          <span class="mono text-warn">⚠</span>
          <span class="mono">{activeEmb}</span>
          <span class="text-warn text-[11px]">{t("models.embedding.autoFetched")}</span>
        {/if}
        <span class="dim ml-auto text-[11px]">{t("models.embedding.pickHint")}</span>
      </div>
    {/if}
  </section>

  <!-- SERVER -->
  <section class="{cardBase} col-span-3">
    <h2 class={cardH2}>{t("server.title")}</h2>
    {#if config}
      <div class={sectionGrid}>
      <div class={kvRow}>
        <span class="dim">{t("server.cors")}</span>
        <select
          value={config.server.cors}
          onchange={(e) => {
            if (!config) return;
            const v = (e.target as HTMLSelectElement).value as CorsMode;
            config.server.cors = v;
            // Host binding follows CORS scope by default — most users don't
            // need them independent. `off` lets the user pin a specific IP.
            if (v === "localhost") config.server.host = "127.0.0.1";
            else if (v === "all") config.server.host = "0.0.0.0";
            saveServer();
          }}
        >
          <option value="off">{t("server.cors.off")}</option>
          <option value="localhost">{t("server.cors.localhost")}</option>
          <option value="all">{t("server.cors.all")}</option>
        </select>
      </div>
      <div class={kvRow}>
        <span class="dim">{t("server.host")}</span>
        <input
          type="text"
          bind:value={config.server.host}
          disabled={config.server.cors !== "off"}
          onchange={saveServer}
          title={config.server.cors === "off" ? t("server.host.title.pin") : t("server.host.title.auto")}
        />
      </div>
      <div class={kvRow}>
        <span class="dim">{t("server.port")}</span>
        <input type="number" min="1" max="65535" bind:value={config.server.port} onchange={saveServer} />
      </div>
      <div class={kvRow}>
        <span class="dim">{t("server.apiKey")}</span>
        <span class="dim text-[11px] italic">{t("server.apiKey.hint")}</span>
      </div>

      <h3 class="{cardSection} {colSpanFull}">
        {t("server.memory.title")} <span class="dim">{t("server.memory.titleHint")}</span>
        {#if activeModel && recommendedMemoryGbExact != null && recommendedWiredGb != null && recommendedCacheGb != null}
          <span class={ramTunedHint}>
            · {t("server.memory.tunedFor")} <b class="text-text">{activeModel.id.split("/").pop()}</b>
            + ctx {config.context.max}
            ({recommendedWiredGb.toFixed(3)}/{recommendedCacheGb.toFixed(3)}/{recommendedMemoryGbExact.toFixed(3)} GB)
          </span>
          <button class={resetMemBtn} onclick={resetMemoryCaps}>{t("action.reset")}</button>
        {:else if systemInfo}
          <span class={ramTunedHint}>
            · {t("server.memory.systemDefault")} {systemInfo.ram_gb} GB
            ({systemInfo.recommended.wired_limit_gb.toFixed(3)}/{systemInfo.recommended.cache_limit_gb.toFixed(3)}/{systemInfo.recommended.memory_limit_gb.toFixed(3)} GB)
          </span>
          <button class={resetMemBtn} onclick={resetMemoryCaps}>{t("action.reset")}</button>
        {/if}
      </h3>
      <details class="mem-explainer {colSpanFull} my-0.5 mb-1.5 text-xs">
        <summary class="dim cursor-pointer list-none py-1">{t("server.memory.explainer")}</summary>
        <div class="dim py-1.5 pb-1 leading-relaxed">
          {t("server.memory.explainer.intro")}
          <ul class="mt-1.5 pl-4.5 list-disc">
            <li class="mb-1">{t("server.memory.explainer.wired")}</li>
            <li class="mb-1">{t("server.memory.explainer.cache")}</li>
            <li class="mb-1">{t("server.memory.explainer.memory")}</li>
          </ul>
        </div>
      </details>
      <div class={kvRow}>
        <span class="dim">{t("server.memory.wired")}</span>
        <div class="flex items-center gap-2">
          <input
            class="flex-1"
            type="number"
            placeholder={recommendedWiredLabel ?? String(systemInfo?.recommended.wired_limit_gb ?? 28)}
            value={config.server.wired_limit_gb ?? ""}
            oninput={(e) => {
              if (!config) return;
              const v = (e.target as HTMLInputElement).value;
              config.server.wired_limit_gb = v === "" ? null : Number(v);
            }}
            onchange={saveServer}
          />
          {#if activeModel && config.server.wired_limit_gb == null}
            <span class="dim mono text-[11px] whitespace-nowrap" title={t("server.memory.wired.titleHint")}>
              = {Math.round(activeModel.size_bytes / 1024 ** 2).toLocaleString()} MB
            </span>
          {/if}
        </div>
      </div>
      <div class={kvRow}>
        <span class="dim">{t("server.memory.cache")}</span>
        <input
          type="number"
          placeholder={String(recommendedCacheGb ?? systemInfo?.recommended.cache_limit_gb ?? 8)}
          value={config.server.cache_limit_gb ?? ""}
          oninput={(e) => {
            if (!config) return;
            const v = (e.target as HTMLInputElement).value;
            config.server.cache_limit_gb = v === "" ? null : Number(v);
          }}
          onchange={saveServer}
        />
      </div>
      <div class={kvRow}>
        <span class="dim">{t("server.memory.memory")}</span>
        <input
          type="number"
          placeholder={String(recommendedMemoryGb ?? systemInfo?.recommended.memory_limit_gb ?? 32)}
          value={config.server.memory_limit_gb ?? ""}
          oninput={(e) => {
            if (!config) return;
            const v = (e.target as HTMLInputElement).value;
            config.server.memory_limit_gb = v === "" ? null : Number(v);
          }}
          onchange={saveServer}
        />
      </div>
      </div>
    {/if}
  </section>

  {/if}

  {#if activeTab === "debug"}
  <!-- DEBUG / power-user knobs (A/B testing, loader overrides) -->
  <section class="{cardBase} col-span-3">
    <h2 class={cardH2}>{t("debug.title")} <span class="dim">{t("debug.titleHint")}</span></h2>
    {#if config}
      <p class="dim m-0 mb-3 px-3 py-2 bg-panel-2 rounded-[5px] text-xs leading-normal">
        {t("debug.intro")}
      </p>
      <div class={sectionGrid}>
        <h3 class="{cardSection} {colSpanFull}">{t("debug.memoryBypass")}</h3>
        <div class={kvRow}>
          <span class="dim">{t("debug.memoryBypass.label")}</span>
          <label class={toggleLabel}>
            <input class="w-3.5 h-3.5 m-0 accent-accent" type="checkbox" bind:checked={config.server.disable_wired_limit} onchange={saveServer} />
            <span class="dim">{t("debug.memoryBypass.hint")}</span>
          </label>
        </div>

        <h3 class="{cardSection} {colSpanFull}">{t("debug.loader")}</h3>
        <div class={kvRow}>
          <span class="dim">{t("debug.tokenizer")}</span>
          <input
            type="text"
            placeholder={t("debug.tokenizer.placeholder")}
            value={config.server.tokenizer_id ?? ""}
            oninput={(e) => {
              if (!config) return;
              const v = (e.target as HTMLInputElement).value;
              config.server.tokenizer_id = v === "" ? null : v;
            }}
            onchange={saveServer}
          />
        </div>
        <div class={kvRow}>
          <span class="dim">{t("debug.weightsDir")}</span>
          <input
            type="text"
            placeholder={t("debug.weightsDir.placeholder")}
            value={config.server.local_model_dir ?? ""}
            oninput={(e) => {
              if (!config) return;
              const v = (e.target as HTMLInputElement).value;
              config.server.local_model_dir = v === "" ? null : v;
            }}
            onchange={saveServer}
          />
        </div>
        <div class={kvRow}>
          <span class="dim">{t("debug.skipWarmup")}</span>
          <label class={toggleLabel}>
            <input class="w-3.5 h-3.5 m-0 accent-accent" type="checkbox" bind:checked={config.server.skip_warmup} onchange={saveServer} />
            <span class="dim">{t("debug.skipWarmup.hint")}</span>
          </label>
        </div>
      </div>
    {/if}
  </section>
  {/if}

  {#if activeTab === "api"}
  <!-- API (OpenAI / Claude tabs) -->
  <section class="{cardBase} col-span-3">
    {#if config}
      <ApiTabs
        {config}
        {status}
        {catalog}
        {models}
        {systemInfo}
        {downloads}
        onEmbeddingChange={(v) => {
          if (!config) return;
          config.server.embedding_model_id = v;
          saveServer();
        }}
        onApiKeyChange={(v) => {
          if (!config) return;
          config.server.api_key = v;
          saveServer();
        }}
        onDownloadEmbedding={async (id) => {
          try {
            await api.downloadModel(id, null);
          } catch (e) {
            statusMessage = `Embedding download failed: ${e}`;
          }
        }}
        onDeleteEmbedding={deleteEmbedding}
      />
    {/if}
  </section>

  {/if}

  {#if activeTab === "language"}
  <!-- LANGUAGE picker. Persisted to localStorage by `i18n.svelte.ts`; switch
       takes effect immediately because every translated string reads
       `i18n.locale` reactively. -->
  <section class="{cardBase} col-span-3">
    <h2 class={cardH2}>{t("language.title")}</h2>
    <p class="dim m-0 mb-3 px-3 py-2 bg-panel-2 rounded-[5px] text-xs leading-normal">
      {t("language.note")}
    </p>
    <h3 class={cardSection}>{t("language.choose")}</h3>
    <div class="flex flex-col gap-1 mt-1">
      {#each Object.entries(LOCALE_LABELS) as [code, label]}
        <label class="flex items-center gap-2 px-2.5 py-2 rounded-md bg-panel-2 cursor-pointer hover:border-border border border-transparent">
          <input
            type="radio"
            name="locale"
            value={code}
            checked={i18n.locale === code}
            onchange={() => i18n.set(code as Locale)}
            class="w-3.5 h-3.5 m-0 accent-accent"
          />
          <span>{label}</span>
          <span class="dim mono text-[11px] ml-auto">{code}</span>
        </label>
      {/each}
    </div>
  </section>
  {/if}

</main>

<!-- ── Delete confirmation modal ─────────────────────────────────
     Used by both the chat-model row Delete buttons and the embedding
     picker's Delete button. Backdrop click + Esc dismiss. -->
{#if confirmDel}
  <div
    class="fixed inset-0 z-100 flex items-center justify-center bg-black/55 backdrop-blur-sm"
    onclick={cancelConfirmedDelete}
    onkeydown={(e) => { if (e.key === "Escape") cancelConfirmedDelete(); }}
    role="presentation"
  >
    <div
      class="bg-panel border border-border rounded-xl shadow-[0_8px_32px_rgba(0,0,0,0.5)] p-5 min-w-105 max-w-135"
      onclick={(e) => e.stopPropagation()}
      onkeydown={(e) => e.stopPropagation()}
      role="dialog"
      aria-modal="true"
      aria-labelledby="confirm-del-title"
      tabindex="-1"
    >
      <h3 id="confirm-del-title" class="m-0 mb-2.5 text-sm font-semibold text-text">
        {confirmDel.kind === "embedding" ? t("confirm.delete.embedding.title") : t("confirm.delete.chat.title")}
      </h3>
      <p class="mb-1.5 text-[13px] text-text leading-normal break-all">
        <span class="mono text-accent">{confirmDel.id}</span>
      </p>
      <p class="mb-4 text-xs text-text-dim leading-[1.55]">
        {t("confirm.delete.warning")}{confirmDel.kind === "embedding" && config && config.server.embedding_model_id === confirmDel.id ? t("confirm.delete.embedding.activeNote") : ""}
      </p>
      <div class="flex justify-end gap-2">
        <button onclick={cancelConfirmedDelete} disabled={confirmDelBusy}>{t("action.cancel")}</button>
        <button class="danger" onclick={performConfirmedDelete} disabled={confirmDelBusy}>
          {confirmDelBusy ? t("confirm.delete.busy") : t("action.delete")}
        </button>
      </div>
    </div>
  </div>
{/if}

<!-- ── Footer panel: logs + env overrides ──────────────────────── -->
<footer class="border-t border-border bg-panel flex flex-col fixed left-0 right-0 bottom-0 z-8">
  {#if logsOpen}
    <div
      bind:this={logsContainer}
      onscroll={onLogsScroll}
      class="{panelScroll} mono px-4 py-1.5 pb-2.5 text-xs"
    >
      <!-- Virtualization via `content-visibility: auto`: the browser
           skips layout + paint for off-screen entries even though all
           ${LOG_MAX_LINES} live in the DOM. `contain-intrinsic-size`
           gives a per-line estimate so scrollbar height stays stable
           before items are first measured. Combined with the 2000-line
           cap this keeps the panel responsive on multi-hour sessions. -->
      {#each logs as l}
        <div
          class={`whitespace-pre-wrap break-all ${l.stream === "stderr" ? "text-[#d6b9ff]" : "text-text-dim"}`}
          style="content-visibility: auto; contain-intrinsic-size: auto 18px;"
        >{l.line}</div>
      {/each}
      {#if logs.length === 0}
        <div class="dim">{t("footer.logs.empty")}</div>
      {/if}
    </div>
  {/if}
  {#if envOpen && config}
    <div class={panelScroll}>
      <EnvOverrides
        value={config.env_overrides}
        typedKeys={typedEnvKeys}
        onSave={saveEnvOverrides}
      />
    </div>
  {/if}
  {#if doctorOpen}
    <div class={panelScroll}>
      <DoctorPanel
        report={doctorReport}
        onReport={(r) => (doctorReport = r)}
      />
    </div>
  {/if}
  {#if updateOpen}
    <div class={panelScroll}>
      <UpdatePanel serverRunning={status.state === "running" || status.state === "starting"} />
    </div>
  {/if}
  <div class="flex items-center h-9 shrink-0 border-t border-border bg-panel">
    <button
      class={`${footerTabBase} ${logsOpen ? footerTabActive : ""}`}
      onclick={() => {
        logsOpen = !logsOpen;
        if (logsOpen) {
          envOpen = false;
          doctorOpen = false;
          updateOpen = false;
        }
      }}
    >
      {t("footer.logs")} {logsOpen ? "▾" : "▸"} <span class="dim mono">({logs.length})</span>
    </button>
    <button
      class={`${footerTabBase} ${envOpen ? footerTabActive : ""}`}
      onclick={() => {
        envOpen = !envOpen;
        if (envOpen) {
          logsOpen = false;
          doctorOpen = false;
          updateOpen = false;
        }
      }}
    >
      {t("footer.env")} {envOpen ? "▾" : "▸"}
      <span class="dim mono">({config ? Object.keys(config.env_overrides).length : 0})</span>
    </button>
    <button
      class={`${footerTabBase} ${doctorOpen ? footerTabActive : ""}`}
      onclick={() => {
        doctorOpen = !doctorOpen;
        if (doctorOpen) {
          logsOpen = false;
          envOpen = false;
          updateOpen = false;
        }
      }}
    >
      {t("footer.doctor")} {doctorOpen ? "▾" : "▸"}
      {#if doctorReport}
        <span class="dim mono">
          ({doctorCounts.pass}✓{doctorCounts.warn > 0 ? ` ${doctorCounts.warn}!` : ""}{doctorCounts.fail > 0 ? ` ${doctorCounts.fail}✗` : ""})
        </span>
      {/if}
    </button>
    <button
      class={`${footerTabBase} ${updateOpen ? footerTabActive : ""}`}
      onclick={() => {
        updateOpen = !updateOpen;
        if (updateOpen) {
          logsOpen = false;
          envOpen = false;
          doctorOpen = false;
        }
      }}
    >
      {t("footer.update")} {updateOpen ? "▾" : "▸"}
    </button>
  </div>
</footer>

<style>
  /* Tooltip portal: appended to <body> by the `tooltip` action so it
     escapes ancestor overflow / stacking clipping. Must be :global() since
     the element lives outside this component's scoped CSS. */
  :global(.tooltip-portal) {
    position: fixed;
    max-width: 280px;
    padding: 8px 10px;
    background: var(--panel-2);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text);
    font-size: 11px;
    line-height: 1.45;
    text-align: left;
    white-space: normal;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    pointer-events: none;
    z-index: 10000;
  }

  /* Custom `<details>` marker — pseudo-elements (`::before` and the
     webkit-specific `::-webkit-details-marker`) aren't reachable via Tailwind
     utility classes, so the "What do these mean?" Metal-memory explainer in
     the SERVER card keeps a small CSS rule here. */
  details.mem-explainer > summary::-webkit-details-marker {
    display: none;
  }
  details.mem-explainer > summary::before {
    content: "▸ ";
    color: var(--text-dim);
  }
  details.mem-explainer[open] > summary::before {
    content: "▾ ";
  }
</style>
