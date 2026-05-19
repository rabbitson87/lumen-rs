<script lang="ts">
  import { onMount, onDestroy } from "svelte";
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
    type CorsMode,
  } from "./lib/api";
  import EnvOverrides from "./lib/EnvOverrides.svelte";
  import DoctorPanel from "./lib/DoctorPanel.svelte";
  import UpdatePanel from "./lib/UpdatePanel.svelte";
  import ApiTabs from "./lib/ApiTabs.svelte";
  import { bytes, duration } from "./lib/format";

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
  let downloads = $state<Map<string, DownloadProgress>>(new Map());
  let statusMessage = $state<string | null>(null);
  let typedEnvKeys = $state<Set<string>>(new Set());
  let catalog = $state<Catalog>({ families: [], recommended: [], embeddings: [] });
  let systemInfo = $state<SystemInfo | null>(null);
  let selectedRecommend = $state<string>("");

  let visibleModels = $derived(models.filter((m) => m.supported));

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

    unlistenLog = await onLog((l) => {
      logs = [...logs.slice(-499), l];
    });
    unlistenStatus = await onStatus((s) => {
      status = s;
    });
    unlistenDownload = await onDownload((p) => {
      const next = new Map(downloads);
      const key = `${p.repo_id}/${p.file}`;
      next.set(key, p);
      downloads = next;
      if (p.done) {
        api.listModels().then((m) => (models = m));
        // Auto-clear completed lines after 3s so the progress panel
        // doesn't crowd up during back-to-back downloads. Re-check
        // `done` before removing so an in-flight resume doesn't drop
        // a now-active entry under the same key.
        setTimeout(() => {
          const cur = new Map(downloads);
          if (cur.get(key)?.done) {
            cur.delete(key);
            downloads = cur;
          }
        }, 3000);
      }
    });

    pollHandle = setInterval(async () => {
      if (status.state === "running") {
        metrics = await api.serverMetrics();
      }
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

  async function removeModel(id: string) {
    if (!confirm(`Delete ${id}? Weights will be removed from disk.`)) return;
    await api.deleteModel(id);
    models = await api.listModels();
  }
</script>

<!-- ── Top bar ──────────────────────────────────────────────────── -->
<header class="topbar">
  <div class="brand">● Lumen</div>
  <div class="status">
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
      <span class="err" title={status.last_error}>· error</span>
    {/if}
  </div>
  <div class="actions">
    {#if statusMessage}<span class="dim">{statusMessage}</span>{/if}
    <button
      class="health"
      class:healthy={doctorReport?.overall === "healthy"}
      class:degraded={doctorReport?.overall === "degraded"}
      class:blocked={doctorReport?.overall === "blocked"}
      onclick={() => {
        doctorOpen = !doctorOpen;
        if (doctorOpen) {
          logsOpen = false;
          envOpen = false;
        }
      }}
      title={doctorReport ? `${doctorCounts.pass} pass · ${doctorCounts.warn} warn · ${doctorCounts.fail} fail` : "Run preflight checks"}
    >
      <span class="health-dot"></span>
      Doctor
      {#if doctorReport && (doctorCounts.warn > 0 || doctorCounts.fail > 0)}
        <span class="health-badge mono">
          {#if doctorCounts.fail > 0}{doctorCounts.fail}✗{/if}
          {#if doctorCounts.warn > 0}{doctorCounts.warn}!{/if}
        </span>
      {/if}
    </button>
    <button
      class={status.state === "running" || status.state === "starting" ? "danger" : "primary"}
      onclick={toggleServer}
      disabled={status.state === "starting" || status.state === "stopping"}
    >
      {status.state === "running" || status.state === "starting" ? "Stop" : "Start"}
    </button>
  </div>
</header>

<!-- ── Tab bar (top-level grouping) ─────────────────────────────── -->
<nav class="tab-bar">
  <button
    class="top-tab"
    class:active={activeTab === "main"}
    onclick={() => (activeTab = "main")}
  >Models &amp; Server</button>
  <button
    class="top-tab"
    class:active={activeTab === "tuning"}
    onclick={() => (activeTab = "tuning")}
  >Tuning</button>
  <button
    class="top-tab"
    class:active={activeTab === "api"}
    onclick={() => (activeTab = "api")}
  >API</button>
  <button
    class="top-tab"
    class:active={activeTab === "debug"}
    onclick={() => (activeTab = "debug")}
  >Debug</button>
</nav>

<!-- ── Card grid ───────────────────────────────────────────────── -->
<main class="grid" class:tuning-mode={activeTab === "tuning"}>
  {#if activeTab === "tuning"}
  <!-- QUANT -->
  <section class="card span-3">
    <h2>QUANT <span class="dim">(TurboQuant KV cache)</span></h2>
    {#if config}
      <div class="kv">
        <span class="dim">Bits</span>
        <div class="seg">
          {#each [2, 3, 4] as b}
            <button
              class={config.quant.bits === b ? "primary" : ""}
              onclick={() => {
                if (!config) return;
                config.quant.bits = b;
                saveQuant();
              }}
            >{b}</button>
          {/each}
        </div>
      </div>
      <div class="quant-tradeoff dim">
        {#if config.quant.bits === 2}
          <span class="qt-row"><b>2-bit</b> · ~8× smaller vs FP16 · cosine <b>0.9851</b> · Top-5 <b>89%</b></span>
          <span class="qt-row qt-delta">vs 4-bit baseline: <b class="ok">−50% KV memory</b> · cosine <b class="warn">−1.3%</b></span>
        {:else if config.quant.bits === 3}
          <span class="qt-row"><b>3-bit</b> · ~5× smaller vs FP16 · cosine <b>0.9945</b> · Top-5 <b>94%</b></span>
          <span class="qt-row qt-delta">vs 4-bit baseline: <b class="ok">−25% KV memory</b> · cosine <b>−0.4%</b></span>
        {:else if config.quant.bits === 4}
          <span class="qt-row"><b>4-bit</b> · ~4× smaller vs FP16 · cosine <b>0.9983</b> · Top-5 <b>96%</b></span>
          <span class="qt-row qt-delta">baseline (highest quality)</span>
        {/if}
        <span class="qt-row qt-delta">
          QJL m + seed (TurboQuant internals) → Debug tab
        </span>
      </div>
    {/if}
  </section>

  <!-- METRICS -->
  <section class="card span-3">
    <h2>METRICS</h2>
    <div class="kv">
      <span class="dim">tokens/sec</span>
      <span class="mono">{metrics.tokens_per_sec?.toFixed(1) ?? "—"}</span>
    </div>
    <div class="kv">
      <span class="dim">ms / step</span>
      <span class="mono">{metrics.ms_per_step?.toFixed(2) ?? "—"}</span>
    </div>
    <div class="kv">
      <span class="dim">KV cache</span>
      <span class="mono">{metrics.kv_cache_mb != null ? `${metrics.kv_cache_mb} MB` : "—"}</span>
    </div>
    <div class="kv">
      <span class="dim">req/min</span>
      <span class="mono">{metrics.requests_per_min ?? "—"}</span>
    </div>
  </section>

  <!-- CONTEXT -->
  <section class="card span-6">
    <h2>CONTEXT</h2>
    {#if config}
      <div class="kv">
        <span class="dim">Max</span>
        <input
          type="number"
          min="512"
          step="512"
          bind:value={config.context.max}
          onchange={saveContext}
        />
      </div>
      <div class="ctx-hint dim">
        Max sequence length (tokens). Caps the model's <code>max_position_embeddings</code>
        when host RAM can't hold the model's native limit (Gemma 4 claims 128K
        but a 16 GB Mac realistically handles ~16K). Env: <code>LUMEN_MAX_CTX</code>.
      </div>

      <div class="kv">
        <span class="dim">Sliding</span>
        <input
          type="number"
          min="0"
          step="256"
          bind:value={config.context.sliding}
          onchange={saveContext}
        />
      </div>
      <div class="ctx-hint dim">
        Sliding-window attention size. Some layers (Gemma 4: 25 of 30) only attend
        to the last N tokens instead of the full sequence → bounded KV memory for
        long contexts. <b>0</b> = use the model's built-in default; <b>N&gt;0</b> overrides it
        (smaller = less KV, weaker long-range recall). Env: <code>LUMEN_SLIDING_WINDOW</code>.
      </div>

      <div class="kv">
        <span class="dim">Prefill</span>
        <input
          type="number"
          min="512"
          step="512"
          bind:value={config.context.prefill}
          onchange={saveContext}
        />
      </div>
      <div class="ctx-hint dim">
        Prompt-processing chunk cap. Server rejects prompts longer than this with
        a "prompt too large" error. Larger = accepts long prompts but more peak
        memory during prefill (attention QK<sup>T</sup> = chunk × KV). Env:
        <code>LUMEN_PREFILL_CHUNK</code>.
      </div>
    {/if}
  </section>

  {/if}

  {#if activeTab === "main"}
  <!-- MODELS -->
  <section class="card span-3">
    <h2>MODELS</h2>
    <div class="models">
      {#each visibleModels as m}
        <div
          class="model-row"
          class:dimmed={!m.supported}
          class:row-active={config?.active_model === m.id}
        >
          <span class="mono mark">{config?.active_model === m.id ? "✓" : ""}</span>
          <div class="model-cell">
            <div class="model-id mono">{m.id}</div>
            {#if m.label}
              <div class="model-label dim">{m.label}</div>
            {:else if !m.supported}
              <div class="model-label warn">not in supported catalog</div>
            {/if}
          </div>
          <span class="dim mono model-size">{bytes(m.size_bytes)}</span>
          <button
            onclick={() => setActive(m.id)}
            disabled={config?.active_model === m.id || !m.supported}
            title={!m.supported ? "Not in the server-side supported catalog" : ""}
          >Use</button>
          <button class="danger" onclick={() => removeModel(m.id)}>Delete</button>
        </div>
      {/each}
      {#if visibleModels.length === 0 && models.length > 0}
        <p class="dim models-empty">No supported models on disk. Download one from the curated list below.</p>
      {:else if models.length === 0}
        <p class="dim models-empty">No local models. Download one from the curated list below.</p>
      {/if}
    </div>
    {#if systemInfo}
      <div class="dim ram-hint mono">
        This Mac: <b>{systemInfo.ram_gb} GB RAM</b> ({systemInfo.arch}) — models over this size are marked.
      </div>
    {/if}
    <div class="dl-row">
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
        <div class="dl-hint dim">{r.notes}</div>
      {/if}
    {/if}
    {#if downloads.size > 0}
      <div class="downloads">
        {#each [...downloads.entries()] as [key, p]}
          <div class="dl-line mono">
            <span class={p.done ? "ok" : "dim"}>{p.done ? "✓" : "…"}</span>
            <span>{key}</span>
            <span class="dim">{bytes(p.downloaded_bytes)}</span>
          </div>
        {/each}
      </div>
    {/if}
  </section>

  <!-- SERVER -->
  <section class="card span-3">
    <h2>SERVER</h2>
    {#if config}
      <div class="srv-grid">
      <div class="kv">
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
      <div class="kv">
        <span class="dim">Host</span>
        <input
          type="text"
          bind:value={config.server.host}
          disabled={config.server.cors !== "off"}
          title={config.server.cors === "off" ? "Pin to a specific IP" : "Auto-set by CORS scope"}
        />
      </div>
      <div class="kv">
        <span class="dim">Port</span>
        <input type="number" min="1" max="65535" bind:value={config.server.port} />
      </div>
      <div class="kv">
        <span class="dim">API key</span>
        <span class="dim hint-inline">→ set in the API card</span>
      </div>

      <h3 class="card-section">
        Metal memory <span class="dim">(mlx-native)</span>
        {#if activeModel && recommendedMemoryGbExact != null && recommendedWiredGb != null && recommendedCacheGb != null}
          <span class="ram-tuned-hint dim">
            · tuned for <b>{activeModel.id.split("/").pop()}</b>
            + ctx {config.context.max}
            ({recommendedWiredGb.toFixed(3)}/{recommendedCacheGb.toFixed(3)}/{recommendedMemoryGbExact.toFixed(3)} GB)
          </span>
          <button class="reset-mem" onclick={resetMemoryCaps}>Reset</button>
        {:else if systemInfo}
          <span class="ram-tuned-hint dim">
            · system default for {systemInfo.ram_gb} GB
            ({systemInfo.recommended.wired_limit_gb.toFixed(3)}/{systemInfo.recommended.cache_limit_gb.toFixed(3)}/{systemInfo.recommended.memory_limit_gb.toFixed(3)} GB)
          </span>
          <button class="reset-mem" onclick={resetMemoryCaps}>Reset</button>
        {/if}
      </h3>
      <details class="mem-explainer">
        <summary class="dim">What do these mean?</summary>
        <div class="mem-explainer-body dim">
          Apple Silicon shares one pool of RAM between CPU and GPU. These three caps
          tell MLX how much of that pool it may use:
          <ul>
            <li>
              <b>Wired GB</b> — RAM that stays pinned for the GPU and can never be
              paged out. Auto-set to the <i>exact</i> safetensors byte size of the
              active model (via <code>LUMEN_WIRED_LIMIT_BYTES</code>), so a
              14.45 GB model isn't truncated to a 14 GB ceiling. Override the
              input if you want extra headroom for KV cache.
            </li>
            <li>
              <b>Cache GB</b> — MLX's transient buffer reuse pool (activations,
              scratch). A small fixed budget (2 GB) is enough; scaling it with
              system RAM just reserves memory you'd rather give back to the OS.
            </li>
            <li>
              <b>Memory GB</b> — Soft total ceiling for Metal allocations.
              Hitting it triggers cache eviction before the hard wired limit.
              Set to model size + 2 GB + KV cache budget (≈ ctx ÷ 8K).
            </li>
          </ul>
        </div>
      </details>
      <div class="kv">
        <span class="dim">Wired GB</span>
        <div class="wired-row">
          <input
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
            <span class="dim mono wired-hint" title="LUMEN_WIRED_LIMIT_BYTES — exact safetensors size">
              = {Math.round(activeModel.size_bytes / 1024 ** 2).toLocaleString()} MB
            </span>
          {/if}
        </div>
      </div>
      <div class="kv">
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
      <div class="kv">
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
      <div class="actions-row">
        <button class="primary" onclick={saveServer}>Save</button>
      </div>
    {/if}
  </section>

  {/if}

  {#if activeTab === "debug"}
  <!-- DEBUG / power-user knobs (A/B testing, loader overrides) -->
  <section class="card span-3">
    <h2>DEBUG <span class="dim">(A/B + loader overrides)</span></h2>
    {#if config}
      <p class="dim debug-intro">
        These knobs are for benchmarking + troubleshooting. Leave everything blank/off
        for normal use — the values that actually matter (model, memory caps, backend)
        live in the Models &amp; Server tab.
      </p>
      <div class="adv-grid">
        <h3 class="card-section">Memory bypass</h3>
        <div class="kv">
          <span class="dim">Bypass all caps</span>
          <label class="toggle">
            <input type="checkbox" bind:checked={config.server.disable_wired_limit} />
            <span class="dim">skip wired+cache+memory; let MLX/macOS manage</span>
          </label>
        </div>

        <h3 class="card-section">Loader overrides</h3>
        <div class="kv">
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
        <div class="kv">
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
        <div class="kv">
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
        <div class="kv">
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
        <div class="kv">
          <span class="dim">Skip warmup</span>
          <label class="toggle">
            <input type="checkbox" bind:checked={config.server.skip_warmup} />
            <span class="dim">faster start, first request slower</span>
          </label>
        </div>

        <h3 class="card-section">TurboQuant internals</h3>
        <p class="debug-note dim">
          Stage-2 residual correction knobs. Leave at defaults — these only matter when
          tuning the compression algorithm itself or reproducing a benchmark run.
        </p>
        <div class="kv">
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
        <div class="debug-hint dim">
          QJL projection dimension for residual sign-bit correction. Recommended:
          <b>head_dim / 2</b> (typically 64). Higher = more accurate inner-product
          estimate but more KV memory; lower = noisier attention scores. <b>When to
          change:</b> only when running quality vs memory ablation studies.
        </div>

        <div class="kv">
          <span class="dim">Seed</span>
          <input
            type="number"
            bind:value={config.quant.seed}
            onchange={saveQuant}
          />
        </div>
        <div class="debug-hint dim">
          Random seed for the orthogonal rotation matrix + Gaussian projection
          matrix. Same seed → bit-identical compression output for the same input.
          <b>When to change:</b> reproducing a specific benchmark, or A/B-testing
          whether a particular seed got lucky/unlucky on a corner case. Different
          seeds are statistically equivalent — don't expect quality differences.
        </div>
      </div>
      <div class="actions-row">
        <button class="primary" onclick={saveServer}>Save</button>
      </div>
    {/if}
  </section>
  {/if}

  {#if activeTab === "api"}
  <!-- API (OpenAI / Claude tabs) -->
  <section class="card span-3">
    {#if config}
      <ApiTabs
        {config}
        {status}
        {catalog}
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
      />
    {/if}
  </section>

  {/if}

</main>

<!-- ── Footer panel: logs + env overrides ──────────────────────── -->
<footer class="footer">
  {#if logsOpen}
    <div class="panel-scroll logs-body mono">
      {#each logs as l}
        <div class="log-line {l.stream}">{l.line}</div>
      {/each}
      {#if logs.length === 0}
        <div class="dim">No log output yet. Start the server to see decode/encode traces.</div>
      {/if}
    </div>
  {/if}
  {#if envOpen && config}
    <div class="panel-scroll">
      <EnvOverrides
        value={config.env_overrides}
        typedKeys={typedEnvKeys}
        onSave={saveEnvOverrides}
      />
    </div>
  {/if}
  {#if doctorOpen}
    <div class="panel-scroll">
      <DoctorPanel
        report={doctorReport}
        onReport={(r) => (doctorReport = r)}
      />
    </div>
  {/if}
  {#if updateOpen}
    <div class="panel-scroll">
      <UpdatePanel serverRunning={status.state === "running" || status.state === "starting"} />
    </div>
  {/if}
  <div class="footer-tabs">
    <button
      class="footer-tab"
      class:active={logsOpen}
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
      class="footer-tab"
      class:active={envOpen}
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
      class="footer-tab"
      class:active={doctorOpen}
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
      class="footer-tab"
      class:active={updateOpen}
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
  .topbar {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 10px 16px;
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
    position: sticky;
    top: 0;
    z-index: 10;
  }

  .tab-bar {
    display: flex;
    gap: 4px;
    padding: 6px 16px;
    background: var(--bg);
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 53px;
    z-index: 9;
  }
  .top-tab {
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 8px 16px;
    font-size: 13px;
    color: var(--text-dim);
    border-radius: 0;
    transition: color 120ms ease, border-color 120ms ease;
  }
  .top-tab:hover {
    color: var(--text);
    background: transparent;
  }
  .top-tab.active {
    color: var(--text);
    border-bottom-color: var(--accent);
    font-weight: 500;
  }
  .brand {
    font-weight: 600;
    font-size: 14px;
    color: var(--accent);
  }
  .status {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .actions {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .err {
    color: var(--err);
  }
  .ok {
    color: var(--ok);
  }
  .warn {
    color: var(--warn);
  }

  .health {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
  }
  .health .health-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--text-dim);
  }
  .health.healthy .health-dot {
    background: var(--ok);
    box-shadow: 0 0 6px var(--ok);
  }
  .health.degraded .health-dot {
    background: var(--warn);
  }
  .health.blocked .health-dot {
    background: var(--err);
    box-shadow: 0 0 6px var(--err);
  }
  .health-badge {
    color: var(--text-dim);
    font-size: 11px;
    margin-left: 2px;
  }

  .grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 16px;
    padding: 18px;
  }
  /* Tuning tab uses a finer 6-column grid so QUANT (span-3) + METRICS (span-3)
     fill row 1 as halves, and CONTEXT (span-3) lands as a half on row 2. */
  .grid.tuning-mode {
    grid-template-columns: repeat(6, minmax(0, 1fr));
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 20px;
    min-height: 0;
  }
  .card.span-2 {
    grid-column: span 2;
  }
  .card.span-3 {
    grid-column: span 3;
  }
  .card.span-6 {
    grid-column: span 6;
  }
  .card h2 {
    margin: 0 0 14px 0;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: var(--text-dim);
    text-transform: uppercase;
  }
  .card-h-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 4px;
  }
  .card-h-row h2 {
    margin: 0 0 14px 0;
  }
  .ram-tuned-hint {
    font-size: 10px;
    font-weight: 400;
    text-transform: none;
    letter-spacing: 0;
    margin-left: 8px;
  }
  .reset-mem {
    margin-left: 8px;
    padding: 2px 8px;
    font-size: 10px;
    font-weight: 400;
    text-transform: none;
    letter-spacing: 0;
  }
  .ram-hint {
    font-size: 11px;
    margin-bottom: 6px;
  }

  .kv {
    display: grid;
    grid-template-columns: 120px 1fr;
    align-items: center;
    gap: 10px;
    padding: 5px 0;
  }
  .kv .path {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .seg {
    display: flex;
    gap: 4px;
  }
  .seg button {
    padding: 4px 10px;
    min-width: 36px;
  }

  .ctx-hint {
    margin: -2px 0 8px 130px;
    font-size: 11px;
    line-height: 1.55;
  }
  .ctx-hint code {
    padding: 1px 5px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 3px;
    font-size: 10.5px;
  }
  .ctx-hint b {
    color: var(--text);
  }

  .quant-tradeoff {
    display: flex;
    flex-direction: column;
    gap: 2px;
    margin: 4px 0 0 130px;
    font-size: 11px;
    line-height: 1.5;
  }
  .qt-row b {
    color: var(--text);
  }
  .qt-row .ok {
    color: var(--ok);
  }
  .qt-row .warn {
    color: var(--warn);
  }
  .qt-delta {
    color: var(--text-dim);
  }

  .debug-intro {
    margin: 0 0 12px;
    padding: 8px 12px;
    background: var(--panel-2);
    border-radius: 5px;
    font-size: 12px;
    line-height: 1.5;
  }
  .debug-note {
    grid-column: 1 / -1;
    margin: 4px 0 8px;
    font-size: 11px;
    line-height: 1.5;
    font-style: italic;
  }
  .debug-hint {
    grid-column: 1 / -1;
    margin: -2px 0 6px 130px;
    font-size: 11px;
    line-height: 1.55;
  }
  .debug-hint b {
    color: var(--text);
  }

  .wired-row {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .wired-row input {
    flex: 1;
  }
  .wired-hint {
    font-size: 11px;
    white-space: nowrap;
  }

  .ro-display {
    padding: 6px 10px;
    background: var(--panel-2);
    border: 1px solid var(--border);
    border-radius: 4px;
    font-size: 13px;
  }
  .ro-note {
    margin-left: 8px;
    font-size: 11px;
  }

  .mem-explainer {
    grid-column: 1 / -1;
    margin: 2px 0 6px;
    font-size: 12px;
  }
  .mem-explainer > summary {
    cursor: pointer;
    list-style: none;
    padding: 4px 0;
  }
  .mem-explainer > summary::-webkit-details-marker {
    display: none;
  }
  .mem-explainer > summary::before {
    content: "▸ ";
    color: var(--text-dim);
  }
  .mem-explainer[open] > summary::before {
    content: "▾ ";
  }
  .mem-explainer-body {
    padding: 6px 0 4px;
    line-height: 1.6;
  }
  .mem-explainer-body ul {
    margin: 6px 0 0;
    padding-left: 18px;
  }
  .mem-explainer-body li {
    margin-bottom: 4px;
  }
  .mem-explainer-body b {
    color: var(--text);
  }

  .actions-row {
    margin-top: 10px;
    display: flex;
    justify-content: flex-end;
  }

  .models {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 6px 12px;
    margin-bottom: 14px;
  }
  .model-row {
    display: grid;
    grid-template-columns: 18px minmax(0, 1fr) 90px auto auto;
    align-items: center;
    gap: 12px;
    padding: 8px 10px;
    border-radius: 6px;
    background: var(--panel-2);
    border: 1px solid transparent;
  }
  .model-row:hover {
    border-color: var(--border);
  }
  .model-row.dimmed {
    opacity: 0.55;
  }
  .model-row.row-active {
    border-color: var(--accent);
    box-shadow: 0 0 0 1px var(--accent) inset;
    background: rgba(127, 179, 255, 0.06);
  }
  .mark {
    color: var(--ok);
    font-weight: 700;
  }
  .model-cell {
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .model-id {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 13px;
  }
  .model-label {
    font-size: 11px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .model-size {
    text-align: right;
    font-size: 12px;
  }
  .models-empty {
    grid-column: 1 / -1;
    margin: 0;
  }
  .hint-inline {
    font-size: 11px;
    font-style: italic;
  }

  .dl-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 10px;
  }
  .dl-hint {
    margin-top: 6px;
    font-size: 11px;
    line-height: 1.5;
  }
  .downloads {
    margin-top: 8px;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .dl-line {
    display: grid;
    grid-template-columns: 16px 1fr 80px;
    font-size: 12px;
    gap: 6px;
  }

  .card-section {
    margin: 16px 0 6px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.06em;
    color: var(--text-dim);
    text-transform: uppercase;
    border-top: 1px solid var(--border);
    padding-top: 10px;
  }
  .srv-grid,
  .adv-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    column-gap: 24px;
    row-gap: 0;
  }
  .srv-grid .card-section,
  .adv-grid .card-section {
    grid-column: 1 / -1;
  }
  .toggle {
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: pointer;
  }
  .toggle input[type="checkbox"] {
    accent-color: var(--accent);
    width: 14px;
    height: 14px;
    margin: 0;
  }

  .footer {
    border-top: 1px solid var(--border);
    background: var(--panel);
    display: flex;
    flex-direction: column;
    position: fixed;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 8;
  }
  .footer-tabs {
    display: flex;
    align-items: center;
    height: 36px;
    flex-shrink: 0;
    border-top: 1px solid var(--border);
    background: var(--panel);
  }
  .panel-scroll {
    max-height: 40vh;
    overflow-y: auto;
    background: var(--panel);
  }
  .footer-tab {
    text-align: left;
    border: none;
    background: transparent;
    padding: 4px 14px;
    font-size: 12px;
    border-radius: 0;
    border-bottom: 2px solid transparent;
  }
  .footer-tab:hover {
    background: var(--panel-2);
  }
  .footer-tab.active {
    background: var(--panel-2);
    border-bottom-color: var(--accent);
  }
  .logs-body {
    padding: 6px 16px 10px;
    font-size: 12px;
  }
  .log-line {
    white-space: pre-wrap;
    word-break: break-all;
    color: var(--text-dim);
  }
  .log-line.stderr {
    color: #d6b9ff;
  }
</style>
