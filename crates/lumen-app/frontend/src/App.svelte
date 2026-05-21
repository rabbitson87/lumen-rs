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
  import DoctorPanel from "./lib/DoctorPanel.svelte";
  import UpdatePanel from "./lib/UpdatePanel.svelte";
  import ApiTabs from "./lib/ApiTabs.svelte";
  import { bytes, duration } from "./lib/format";

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
  let envOpen = $state(false);
  let doctorOpen = $state(false);
  let doctorReport = $state<DoctorReport | null>(null);
  let updateOpen = $state(false);

  type TopTab = "main" | "tuning" | "api" | "debug";
  let activeTab = $state<TopTab>("main");
  // SvelteMap (not plain Map) so `.set()`/`.delete()` mutations trigger UI
  // updates without manual reassignment — fixes the case where completed
  // download lines wouldn't disappear after the 3s auto-dismiss timer.
  let downloads = new SvelteMap<string, DownloadProgress>();
  let statusMessage = $state<string | null>(null);
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

  // KV compression ratio under the current TurboQuant config.
  // Stage-1 ratio at head_dim=256 (Gemma 4 sliding):
  //   4-bit: ~4.0x, 3-bit: ~5.3x, 2-bit: ~8.0x
  // QJL Stage-2 at default m=1024 adds ~128 B/vector, which dilutes Stage-1
  // ratio roughly: total = bf16/(stage1_bytes + qjl_bytes).
  let turboquantKvRatio = $derived.by(() => {
    if (!config?.quant.turboquant_enabled) return 1.0;
    const bits = config.quant.bits;
    // bf16 = 16 bits/scalar; head_dim=256 → 4096 bits/vector
    const stage1Bits = bits * 256;
    const qjlBits = config.quant.turboquant_qjl_enabled ? config.quant.qjl_m : 0;
    const totalBits = stage1Bits + qjlBits;
    return totalBits > 0 ? 4096 / totalBits : 1.0;
  });

  let turboquantStateLabel = $derived.by(() => {
    if (!config?.quant.turboquant_enabled) return "OFF (bf16 KV)";
    const stage = `${config.quant.bits}-bit`;
    const qjl = config.quant.turboquant_qjl_enabled ? " + QJL" : "";
    return `${stage}${qjl}`;
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
    const realisticTokens = freeForKv * bf16TokensPerGb * turboquantKvRatio;
    return Math.round(realisticTokens / 1024);
  });
  // Precise float used for display (3 decimal places); the persisted
  // `memory_limit_gb` ServerConfig field is still usize so we ceil() at save.
  let recommendedMemoryGbExact = $derived.by(() => {
    if (!activeModel || !config) return null;
    const modelGb = activeModel.size_bytes / 1024 ** 3;
    const ctx = config.context.max ?? 8192;
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

    unlistenLog = await onLog((l) => {
      logs = [...logs.slice(-499), l];
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

  async function saveServer() {
    if (!config) return;
    config = await api.updateServerConfig(config.server);
    statusMessage = "Server config saved";
    setTimeout(() => (statusMessage = null), 1500);
  }

  async function saveQuant() {
    if (!config) return;
    config = await api.updateQuantConfig(config.quant);
  }

  async function saveContext() {
    if (!config) return;
    config = await api.updateContextConfig(config.context);
    // ctx affects KV-cache headroom in the tuned memory recommendation
    // (~1 GB per 8K tokens). Re-sync so saved caps follow.
    await syncTunedMemoryCaps();
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

  async function saveEnvOverrides(next: Record<string, string>) {
    if (!config) return;
    config = await api.updateEnvOverrides(next);
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
  const footerTabBase = "text-left border-0 bg-transparent px-3.5 py-1 text-xs rounded-none border-b-2 border-b-transparent hover:bg-panel-2";
  const footerTabActive = "bg-panel-2 border-b-accent";
</script>

<!-- ── Top bar ──────────────────────────────────────────────────── -->
<header class="flex items-center gap-4 px-4 py-2.5 bg-panel border-b border-border shrink-0 sticky top-0 z-10">
  <div class="font-semibold text-sm text-accent">● Lumen</div>
  <div class="flex items-center gap-1.5">
    <span class="dot {status.state}"></span>
    <span class="mono">{status.state}</span>
    {#if status.state === "running"}
      <span class="dim mono">:{status.port}</span>
      {#if status.uptime_secs != null}
        <span class="dim">·</span>
        <span class="dim mono">{duration(status.uptime_secs)}</span>
      {/if}
    {/if}
    {#if status.last_error}
      <span class="text-err" title={status.last_error}>· error</span>
    {/if}
  </div>
  <div class="ml-auto flex items-center gap-2.5">
    {#if statusMessage}<span class="dim">{statusMessage}</span>{/if}
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
      Doctor
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
        (status.state !== "running" && activeOutdated)}
      title={activeOutdated && status.state !== "running"
        ? "Active model has a newer version on Hub. Update it first (MODELS card → Update)."
        : ""}
    >
      {status.state === "running" || status.state === "starting" ? "Stop" : "Start"}
    </button>
  </div>
</header>

<!-- ── Tab bar (top-level grouping) ─────────────────────────────── -->
<nav class="flex gap-1 px-4 py-1.5 bg-bg border-b border-border sticky top-13.25 z-9">
  <button
    class={`${tabBase} ${activeTab === "main" ? tabActive : tabIdle}`}
    onclick={() => (activeTab = "main")}
  >Models &amp; Server</button>
  <button
    class={`${tabBase} ${activeTab === "tuning" ? tabActive : tabIdle}`}
    onclick={() => (activeTab = "tuning")}
  >Tuning</button>
  <button
    class={`${tabBase} ${activeTab === "api" ? tabActive : tabIdle}`}
    onclick={() => (activeTab = "api")}
  >API</button>
  <button
    class={`${tabBase} ${activeTab === "debug" ? tabActive : tabIdle}`}
    onclick={() => (activeTab = "debug")}
  >Debug</button>
</nav>

<!-- ── Card grid ───────────────────────────────────────────────── -->
<main class={`grid gap-4 p-4.5 ${activeTab === "tuning" ? "grid-cols-6" : "grid-cols-3"}`}>
  {#if activeTab === "tuning"}
  <!-- QUANT -->
  <section class="{cardBase} col-span-3">
    <h2 class={cardH2}>QUANT <span class="dim">(TurboQuant KV cache)</span></h2>
    {#if config}
      <div class={kvRow}>
        <span class="dim">
          TurboQuant
          <span
            class={helpIcon}
            use:tooltip
            data-tooltip="Master switch for KV-cache quantization (Lloyd-Max + Haar rotation, Stage 1). ON saves ~4–8× KV memory at small accuracy cost. OFF keeps KV in bf16 — recommended only if you see quality issues at long context."
          >?</span>
        </span>
        <label class={toggleLabel}>
          <input
            class="w-3.5 h-3.5 m-0 accent-accent"
            type="checkbox"
            checked={config.quant.turboquant_enabled}
            onchange={(e) => {
              if (!config) return;
              config.quant.turboquant_enabled = (e.currentTarget as HTMLInputElement).checked;
              saveQuant();
            }}
          />
          <span>{config.quant.turboquant_enabled ? "ON" : "OFF"}</span>
        </label>
      </div>
      <div class={kvRow}>
        <span class="dim">
          QJL residual (Stage 2)
          <span
            class={helpIcon}
            use:tooltip
            data-tooltip="Stage-2 unbiased 1-bit correction for the Stage-1 residual: projects (original − reconstructed) into m-dim Gaussian space and packs only the sign. Recovers ~2–3% Top-5 / +0.003 cosine at small extra cost (~m/8 bytes per K/V vector; ~25 MB for Gemma 4 sliding window at m=1024). Requires Stage 1 ON."
          >?</span>
        </span>
        <label class={toggleLabel}>
          <input
            class="w-3.5 h-3.5 m-0 accent-accent"
            type="checkbox"
            checked={config.quant.turboquant_qjl_enabled}
            disabled={!config.quant.turboquant_enabled}
            onchange={(e) => {
              if (!config) return;
              config.quant.turboquant_qjl_enabled = (e.currentTarget as HTMLInputElement).checked;
              saveQuant();
            }}
          />
          <span>{config.quant.turboquant_qjl_enabled ? "ON" : "OFF"}</span>
        </label>
      </div>
      <div class={kvRow}>
        <span class="dim">
          Bits
          <span
            class={helpIcon}
            use:tooltip
            data-tooltip="Lloyd-Max bits per KV channel. 4: highest quality, ~4× smaller than FP16. 3: balanced — recommended default. 2: max compression (~8× smaller), small quality drop. Applies to the sliding-window KV on Gemma 4."
          >?</span>
        </span>
        <div class="flex gap-1">
          {#each [2, 3, 4] as b}
            <button
              class={`px-2.5 py-1 min-w-9 ${config.quant.bits === b ? "primary" : ""}`}
              disabled={!config.quant.turboquant_enabled}
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
        {#if !config.quant.turboquant_enabled}
          <span class="text-warn">TurboQuant OFF — KV cache stays bf16 (~5 GB at 11K Korean context on Gemma 4)</span>
        {:else if config.quant.bits === 2}
          <span><b class="text-text">2-bit</b> · ~8× smaller vs FP16 · cosine <b class="text-text">0.9851</b> · Top-5 <b class="text-text">89%</b></span>
          <span class="text-text-dim">vs 4-bit baseline: <b class="text-ok">−50% KV memory</b> · cosine <b class="text-warn">−1.3%</b></span>
        {:else if config.quant.bits === 3}
          <span><b class="text-text">3-bit</b> · ~5× smaller vs FP16 · cosine <b class="text-text">0.9945</b> · Top-5 <b class="text-text">94%</b></span>
          <span class="text-text-dim">vs 4-bit baseline: <b class="text-ok">−25% KV memory</b> · cosine <b class="text-text">−0.4%</b></span>
        {:else if config.quant.bits === 4}
          <span><b class="text-text">4-bit</b> · ~4× smaller vs FP16 · cosine <b class="text-text">0.9983</b> · Top-5 <b class="text-text">96%</b></span>
          <span class="text-text-dim">baseline (highest quality)</span>
        {/if}
        <span class="text-text-dim">
          QJL m + seed (TurboQuant internals) → Debug tab
        </span>
      </div>
    {/if}
  </section>

  <!-- METRICS -->
  <section class="{cardBase} col-span-3">
    <h2 class={cardH2}>METRICS</h2>
    <div class={kvRow}>
      <span class="dim">tokens/sec</span>
      <span class="mono">{metrics.tokens_per_sec?.toFixed(1) ?? "—"}</span>
    </div>
    <div class={kvRow}>
      <span class="dim">ms / step</span>
      <span class="mono">{metrics.ms_per_step?.toFixed(2) ?? "—"}</span>
    </div>
    <div class={kvRow}>
      <span class="dim">KV cache</span>
      <span class="mono">{metrics.kv_cache_mb != null ? `${metrics.kv_cache_mb} MB` : "—"}</span>
    </div>
    <div class={kvRow}>
      <span class="dim">req/min</span>
      <span class="mono">{metrics.requests_per_min ?? "—"}</span>
    </div>
  </section>

  <!-- CONTEXT -->
  <section class="{cardBase} col-span-6">
    <h2 class={cardH2}>CONTEXT <span class="dim">(driven by QUANT state)</span></h2>
    {#if config}
      <div class={`-mt-1 mb-3 px-2.5 py-2 bg-panel-2 border border-border border-l-[3px] rounded text-xs leading-[1.55] flex flex-col gap-0.5 ${
        config.quant.turboquant_enabled ? "border-l-accent" : "border-l-warn"
      }`}>
        <div>
          <span class="dim">TurboQuant:</span>
          <b class="text-text">{turboquantStateLabel}</b>
          {#if config.quant.turboquant_enabled}
            <span class="dim">· KV cache ~{turboquantKvRatio.toFixed(1)}× smaller than bf16</span>
          {:else}
            <span class="dim">· baseline KV memory (no compression)</span>
          {/if}
        </div>
        {#if realisticMaxCtxK != null}
          <div class="dim">
            Recommended max on this Mac ({systemInfo?.ram_gb} GB):
            <b class="text-text">~{realisticMaxCtxK}K tokens</b>
            {#if !config.quant.turboquant_enabled}
              <span class="text-warn">— turn TurboQuant ON to handle longer contexts safely</span>
            {/if}
          </div>
        {/if}
      </div>

      <div class={kvRow}>
        <span class="dim">Max</span>
        <input
          type="number"
          min="512"
          step="512"
          bind:value={config.context.max}
          onchange={saveContext}
        />
      </div>
      <div class={ctxHint}>
        Max sequence length (tokens). Caps the model's <code class={inlineCode}>max_position_embeddings</code>
        when host RAM can't hold the model's native limit (Gemma 4 claims 128K).
        {#if config.quant.turboquant_enabled}
          Current TurboQuant <b class="text-text">{turboquantStateLabel}</b> gives ~{turboquantKvRatio.toFixed(1)}× KV compression
          {#if realisticMaxCtxK != null}
            — realistic on this Mac: <b class="text-text">~{realisticMaxCtxK}K</b>.
          {:else}.{/if}
        {:else}
          <span class="text-warn">TurboQuant OFF</span> — KV stays bf16, so practical limit on this Mac
          {#if realisticMaxCtxK != null}is <b class="text-text">~{realisticMaxCtxK}K</b>{:else}is much lower than the model's native max{/if}.
        {/if}
        Env: <code class={inlineCode}>LUMEN_MAX_CTX</code>.
      </div>

      <div class={kvRow}>
        <span class="dim">Sliding</span>
        <input
          type="number"
          min="0"
          step="256"
          bind:value={config.context.sliding}
          onchange={saveContext}
        />
      </div>
      <div class={ctxHint}>
        Sliding-window attention size. Some layers (Gemma 4: 25 of 30) only attend
        to the last N tokens instead of the full sequence → bounded KV memory for
        long contexts. <b class="text-text">0</b> = use the model's built-in default; <b class="text-text">N&gt;0</b> overrides it
        (smaller = less KV, weaker long-range recall).
        {#if config.quant.turboquant_enabled}
          Stacks with TurboQuant — sliding bounds <i>which</i> tokens are kept, TurboQuant compresses <i>how</i> they're stored.
        {/if}
        Env: <code class={inlineCode}>LUMEN_SLIDING_WINDOW</code>.
      </div>

      <div class={kvRow}>
        <span class="dim">Prefill</span>
        <input
          type="number"
          min="512"
          step="512"
          bind:value={config.context.prefill}
          onchange={saveContext}
        />
      </div>
      <div class={ctxHint}>
        Prompt-processing chunk cap. Server rejects prompts longer than this with
        a "prompt too large" error. Larger = accepts long prompts but more peak
        memory during prefill (attention QK<sup>T</sup> = chunk × KV
        {#if config.quant.turboquant_enabled}, shrunk ~{turboquantKvRatio.toFixed(1)}× by TurboQuant{/if}).
        Env: <code class={inlineCode}>LUMEN_PREFILL_CHUNK</code>.
      </div>
    {/if}
  </section>

  {/if}

  {#if activeTab === "main"}
  <!-- MODELS -->
  <section class="{cardBase} col-span-3">
    <h2 class={cardH2}>MODELS</h2>
    <div class="grid grid-cols-2 gap-x-3 gap-y-1.5 mb-3.5">
      {#each visibleModels as m}
        {@const needsUpdate = outdatedModels.has(m.id)}
        {@const isActive = config?.active_model === m.id}
        <div
          class={`grid grid-cols-[18px_minmax(0,1fr)_90px_auto_auto] items-center gap-3 px-2.5 py-2 rounded-md bg-panel-2 border hover:border-border ${
            !m.supported ? "opacity-55" : ""
          } ${
            needsUpdate && isActive ? "border-warn shadow-[0_0_0_1px_var(--color-warn)_inset] bg-warn/15" :
            needsUpdate ? "border-warn bg-warn/10" :
            isActive ? "border-accent shadow-[0_0_0_1px_var(--color-accent)_inset] bg-accent/[0.06]" :
            "border-transparent"
          }`}
        >
          <span class="mono text-ok font-bold">{isActive ? "✓" : ""}</span>
          <div class="min-w-0 flex flex-col gap-0.5">
            <div class="mono overflow-hidden text-ellipsis whitespace-nowrap text-[13px]">{m.id}</div>
            {#if needsUpdate}
              <div class="text-[11px] overflow-hidden text-ellipsis whitespace-nowrap text-warn">⚠ Newer weights available on Hub — update required before use</div>
            {:else if m.label}
              <div class="text-[11px] overflow-hidden text-ellipsis whitespace-nowrap text-text-dim">{m.label}</div>
            {:else if !m.supported}
              <div class="text-[11px] overflow-hidden text-ellipsis whitespace-nowrap text-warn">not in supported catalog</div>
            {/if}
          </div>
          <span class="mono text-text-dim text-right text-xs">{bytes(m.size_bytes)}</span>
          {#if needsUpdate}
            <button
              class="primary"
              onclick={() => updateOutdated(m.id)}
              title="Re-download with the latest Hub weights"
            >Update</button>
          {:else}
            <button
              onclick={() => setActive(m.id)}
              disabled={isActive || !m.supported}
              title={!m.supported ? "Not in the server-side supported catalog" : ""}
            >Use</button>
          {/if}
          <button class="danger" onclick={() => removeModel(m.id)}>Delete</button>
        </div>
      {/each}
      {#if visibleModels.length === 0 && models.length > 0}
        <p class="dim col-span-full m-0">No supported models on disk. Download one from the curated list below.</p>
      {:else if models.length === 0}
        <p class="dim col-span-full m-0">No local models. Download one from the curated list below.</p>
      {/if}
    </div>
    {#if systemInfo}
      <div class="dim mono text-[11px] mb-1.5">
        This Mac: <b class="text-text">{systemInfo.ram_gb} GB RAM</b> ({systemInfo.arch}) — models over this size are marked.
      </div>
    {/if}
    <div class="grid grid-cols-[minmax(0,1fr)_auto] gap-2.5">
      <select bind:value={selectedRecommend}>
        <option value="" disabled>— pick a recommended model —</option>
        {#each availableRecommended as r}
          {@const fits = !systemInfo || r.min_ram_gb <= systemInfo.ram_gb}
          <option value={r.id}>
            {fits ? "" : "⚠ "}{r.label} · {r.approx_size_gb}GB · ≥{r.min_ram_gb}GB RAM
          </option>
        {/each}
        {#if availableRecommended.length === 0}
          <option value="" disabled>all recommended already downloaded</option>
        {/if}
      </select>
      <button
        class="primary"
        onclick={startDownload}
        disabled={!selectedRecommend}
      >Download</button>
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
  </section>

  <!-- SERVER -->
  <section class="{cardBase} col-span-3">
    <h2 class={cardH2}>SERVER</h2>
    {#if config}
      <div class={sectionGrid}>
      <div class={kvRow}>
        <span class="dim">CORS</span>
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
          }}
        >
          <option value="off">off (specific IP)</option>
          <option value="localhost">localhost (127.0.0.1)</option>
          <option value="all">all / 0.0.0.0 (risky)</option>
        </select>
      </div>
      <div class={kvRow}>
        <span class="dim">Host</span>
        <input
          type="text"
          bind:value={config.server.host}
          disabled={config.server.cors !== "off"}
          title={config.server.cors === "off" ? "Pin to a specific IP" : "Auto-set by CORS scope"}
        />
      </div>
      <div class={kvRow}>
        <span class="dim">Port</span>
        <input type="number" min="1" max="65535" bind:value={config.server.port} />
      </div>
      <div class={kvRow}>
        <span class="dim">API key</span>
        <span class="dim text-[11px] italic">→ set in the API card</span>
      </div>

      <h3 class="{cardSection} {colSpanFull}">
        Metal memory <span class="dim">(mlx-native)</span>
        {#if activeModel && recommendedMemoryGbExact != null && recommendedWiredGb != null && recommendedCacheGb != null}
          <span class={ramTunedHint}>
            · tuned for <b class="text-text">{activeModel.id.split("/").pop()}</b>
            + ctx {config.context.max}
            ({recommendedWiredGb.toFixed(3)}/{recommendedCacheGb.toFixed(3)}/{recommendedMemoryGbExact.toFixed(3)} GB)
          </span>
          <button class={resetMemBtn} onclick={resetMemoryCaps}>Reset</button>
        {:else if systemInfo}
          <span class={ramTunedHint}>
            · system default for {systemInfo.ram_gb} GB
            ({systemInfo.recommended.wired_limit_gb.toFixed(3)}/{systemInfo.recommended.cache_limit_gb.toFixed(3)}/{systemInfo.recommended.memory_limit_gb.toFixed(3)} GB)
          </span>
          <button class={resetMemBtn} onclick={resetMemoryCaps}>Reset</button>
        {/if}
      </h3>
      <details class="mem-explainer {colSpanFull} my-0.5 mb-1.5 text-xs">
        <summary class="dim cursor-pointer list-none py-1">What do these mean?</summary>
        <div class="dim py-1.5 pb-1 leading-relaxed">
          Apple Silicon shares one pool of RAM between CPU and GPU. These three caps
          tell MLX how much of that pool it may use:
          <ul class="mt-1.5 pl-4.5 list-disc">
            <li class="mb-1">
              <b class="text-text">Wired GB</b> — RAM that stays pinned for the GPU and can never be
              paged out. Auto-set to the <i>exact</i> safetensors byte size of the
              active model (via <code class={inlineCode}>LUMEN_WIRED_LIMIT_BYTES</code>), so a
              14.45 GB model isn't truncated to a 14 GB ceiling. Override the
              input if you want extra headroom for KV cache.
            </li>
            <li class="mb-1">
              <b class="text-text">Cache GB</b> — MLX's transient buffer reuse pool (activations,
              scratch). A small fixed budget (2 GB) is enough; scaling it with
              system RAM just reserves memory you'd rather give back to the OS.
            </li>
            <li class="mb-1">
              <b class="text-text">Memory GB</b> — Soft total ceiling for Metal allocations.
              Hitting it triggers cache eviction before the hard wired limit.
              Set to model size + 2 GB + KV cache budget (≈ ctx ÷ 8K).
            </li>
          </ul>
        </div>
      </details>
      <div class={kvRow}>
        <span class="dim">Wired GB</span>
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
          />
          {#if activeModel && config.server.wired_limit_gb == null}
            <span class="dim mono text-[11px] whitespace-nowrap" title="LUMEN_WIRED_LIMIT_BYTES — exact safetensors size">
              = {Math.round(activeModel.size_bytes / 1024 ** 2).toLocaleString()} MB
            </span>
          {/if}
        </div>
      </div>
      <div class={kvRow}>
        <span class="dim">Cache GB</span>
        <input
          type="number"
          placeholder={String(recommendedCacheGb ?? systemInfo?.recommended.cache_limit_gb ?? 8)}
          value={config.server.cache_limit_gb ?? ""}
          oninput={(e) => {
            if (!config) return;
            const v = (e.target as HTMLInputElement).value;
            config.server.cache_limit_gb = v === "" ? null : Number(v);
          }}
        />
      </div>
      <div class={kvRow}>
        <span class="dim">Memory GB</span>
        <input
          type="number"
          placeholder={String(recommendedMemoryGb ?? systemInfo?.recommended.memory_limit_gb ?? 32)}
          value={config.server.memory_limit_gb ?? ""}
          oninput={(e) => {
            if (!config) return;
            const v = (e.target as HTMLInputElement).value;
            config.server.memory_limit_gb = v === "" ? null : Number(v);
          }}
        />
      </div>
      </div>
      <div class="mt-2.5 flex justify-end">
        <button class="primary" onclick={saveServer}>Save</button>
      </div>
    {/if}
  </section>

  {/if}

  {#if activeTab === "debug"}
  <!-- DEBUG / power-user knobs (A/B testing, loader overrides) -->
  <section class="{cardBase} col-span-3">
    <h2 class={cardH2}>DEBUG <span class="dim">(A/B + loader overrides)</span></h2>
    {#if config}
      <p class="dim m-0 mb-3 px-3 py-2 bg-panel-2 rounded-[5px] text-xs leading-normal">
        These knobs are for benchmarking + troubleshooting. Leave everything blank/off
        for normal use — the values that actually matter (model, memory caps, backend)
        live in the Models &amp; Server tab.
      </p>
      <div class={sectionGrid}>
        <h3 class="{cardSection} {colSpanFull}">Memory bypass</h3>
        <div class={kvRow}>
          <span class="dim">Bypass all caps</span>
          <label class={toggleLabel}>
            <input class="w-3.5 h-3.5 m-0 accent-accent" type="checkbox" bind:checked={config.server.disable_wired_limit} />
            <span class="dim">skip wired+cache+memory; let MLX/macOS manage</span>
          </label>
        </div>

        <h3 class="{cardSection} {colSpanFull}">Loader overrides</h3>
        <div class={kvRow}>
          <span class="dim">Tokenizer</span>
          <input
            type="text"
            placeholder="HF repo id (override)"
            value={config.server.tokenizer_id ?? ""}
            oninput={(e) => {
              if (!config) return;
              const v = (e.target as HTMLInputElement).value;
              config.server.tokenizer_id = v === "" ? null : v;
            }}
          />
        </div>
        <div class={kvRow}>
          <span class="dim">Weights dir</span>
          <input
            type="text"
            placeholder="auto-set from active model"
            value={config.server.local_model_dir ?? ""}
            oninput={(e) => {
              if (!config) return;
              const v = (e.target as HTMLInputElement).value;
              config.server.local_model_dir = v === "" ? null : v;
            }}
          />
        </div>
        <div class={kvRow}>
          <span class="dim">Compute /buf</span>
          <input
            type="number"
            placeholder="10 (candle only)"
            min="1"
            value={config.server.candle_compute_per_buffer ?? ""}
            oninput={(e) => {
              if (!config) return;
              const v = (e.target as HTMLInputElement).value;
              config.server.candle_compute_per_buffer = v === "" ? null : Number(v);
            }}
          />
        </div>
        <div class={kvRow}>
          <span class="dim">Repeat pen.</span>
          <input
            type="number"
            step="0.05"
            placeholder="1.0"
            value={config.server.repeat_penalty ?? ""}
            oninput={(e) => {
              if (!config) return;
              const v = (e.target as HTMLInputElement).value;
              config.server.repeat_penalty = v === "" ? null : Number(v);
            }}
          />
        </div>
        <div class={kvRow}>
          <span class="dim">Skip warmup</span>
          <label class={toggleLabel}>
            <input class="w-3.5 h-3.5 m-0 accent-accent" type="checkbox" bind:checked={config.server.skip_warmup} />
            <span class="dim">faster start, first request slower</span>
          </label>
        </div>

        <h3 class="{cardSection} {colSpanFull}">TurboQuant internals</h3>
        <p class="dim {colSpanFull} my-1 mb-2 text-[11px] leading-normal italic">
          Stage-2 residual correction knobs. Leave at defaults — these only matter when
          tuning the compression algorithm itself or reproducing a benchmark run.
        </p>
        <div class={kvRow}>
          <span class="dim">QJL m</span>
          <input
            type="number"
            min="16"
            max="256"
            step="16"
            bind:value={config.quant.qjl_m}
            onchange={saveQuant}
          />
        </div>
        <div class={debugHint}>
          QJL projection dimension for residual sign-bit correction. Recommended:
          <b class="text-text">head_dim / 2</b> (typically 64). Higher = more accurate inner-product
          estimate but more KV memory; lower = noisier attention scores. <b class="text-text">When to
          change:</b> only when running quality vs memory ablation studies.
        </div>

        <div class={kvRow}>
          <span class="dim">Seed</span>
          <input
            type="number"
            bind:value={config.quant.seed}
            onchange={saveQuant}
          />
        </div>
        <div class={debugHint}>
          Random seed for the orthogonal rotation matrix + Gaussian projection
          matrix. Same seed → bit-identical compression output for the same input.
          <b class="text-text">When to change:</b> reproducing a specific benchmark, or A/B-testing
          whether a particular seed got lucky/unlucky on a corner case. Different
          seeds are statistically equivalent — don't expect quality differences.
        </div>
      </div>
      <div class="mt-2.5 flex justify-end">
        <button class="primary" onclick={saveServer}>Save</button>
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
        Delete {confirmDel.kind === "embedding" ? "embedding model" : "model"}?
      </h3>
      <p class="mb-1.5 text-[13px] text-text leading-normal break-all">
        <span class="mono text-accent">{confirmDel.id}</span>
      </p>
      <p class="mb-4 text-xs text-text-dim leading-[1.55]">
        Weights will be removed from disk. You can re-download from the
        catalog later.{confirmDel.kind === "embedding" && config && config.server.embedding_model_id === confirmDel.id ? " The active embedding will be cleared." : ""}
      </p>
      <div class="flex justify-end gap-2">
        <button onclick={cancelConfirmedDelete} disabled={confirmDelBusy}>Cancel</button>
        <button class="danger" onclick={performConfirmedDelete} disabled={confirmDelBusy}>
          {confirmDelBusy ? "Deleting…" : "Delete"}
        </button>
      </div>
    </div>
  </div>
{/if}

<!-- ── Footer panel: logs + env overrides ──────────────────────── -->
<footer class="border-t border-border bg-panel flex flex-col fixed left-0 right-0 bottom-0 z-8">
  {#if logsOpen}
    <div class="{panelScroll} mono px-4 py-1.5 pb-2.5 text-xs">
      {#each logs as l}
        <div class={`whitespace-pre-wrap break-all ${l.stream === "stderr" ? "text-[#d6b9ff]" : "text-text-dim"}`}>{l.line}</div>
      {/each}
      {#if logs.length === 0}
        <div class="dim">No log output yet. Start the server to see decode/encode traces.</div>
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
      Logs {logsOpen ? "▾" : "▸"} <span class="dim mono">({logs.length})</span>
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
      Env overrides {envOpen ? "▾" : "▸"}
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
      Doctor {doctorOpen ? "▾" : "▸"}
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
      Update {updateOpen ? "▾" : "▸"}
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
