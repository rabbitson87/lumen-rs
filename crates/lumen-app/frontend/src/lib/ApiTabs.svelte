<script lang="ts">
  import type {
    PersistentConfig,
    ServerStatus,
    Catalog,
    ModelEntry,
    SystemInfo,
  } from "./api";

  interface Props {
    config: PersistentConfig;
    status: ServerStatus;
    catalog: Catalog;
    models: ModelEntry[];
    systemInfo: SystemInfo | null;
    onEmbeddingChange: (value: string | null) => void;
    onApiKeyChange: (value: string | null) => void;
    onDownloadEmbedding: (repoId: string) => void;
  }
  let {
    config,
    status,
    catalog,
    models,
    systemInfo,
    onEmbeddingChange,
    onApiKeyChange,
    onDownloadEmbedding,
  }: Props = $props();

  // Embeddings already downloaded locally — selectable in the dropdown.
  // Matches `catalog.embeddings` against `models` (which is the unified
  // list of every model dir on disk, chat or embedding).
  let downloadedEmbeddings = $derived.by(() => {
    const have = new Set(models.map((m) => m.id));
    return catalog.embeddings.filter((e) => have.has(e.id));
  });

  // Catalog entries not yet on disk — listed in the "Download embedding"
  // picker. Sorted with the ones that fit this Mac's RAM first.
  let availableEmbeddings = $derived.by(() => {
    const have = new Set(models.map((m) => m.id));
    const ram = systemInfo?.ram_gb ?? Infinity;
    return catalog.embeddings
      .filter((e) => !have.has(e.id))
      .slice()
      .sort((a, b) => {
        const aOk = a.min_ram_gb <= ram ? 0 : 1;
        const bOk = b.min_ram_gb <= ram ? 0 : 1;
        if (aOk !== bOk) return aOk - bOk;
        return a.min_ram_gb - b.min_ram_gb;
      });
  });

  let selectedEmbDl = $state<string>("");
  function triggerEmbeddingDownload() {
    if (!selectedEmbDl) return;
    onDownloadEmbedding(selectedEmbDl);
    selectedEmbDl = "";
  }

  type Style = "openai" | "claude";
  let activeStyle = $state<Style>("openai");

  let baseUrl = $derived.by(() => {
    const host =
      config.server.host === "0.0.0.0" || config.server.host === "::"
        ? "127.0.0.1"
        : config.server.host;
    return `http://${host}:${config.server.port}`;
  });

  let openaiBase = $derived(`${baseUrl}/v1`);
  let claudeBase = $derived(baseUrl); // `/v1/messages` is the actual endpoint

  let modelId = $derived(config.active_model ?? "MODEL_ID");
  let hasApiKey = $derived(!!config.server.api_key && config.server.api_key.length > 0);
  let apiKeyPreview = $derived(config.server.api_key ?? "");

  let openaiCurl = $derived(
    hasApiKey
      ? `curl ${openaiBase}/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer ${apiKeyPreview}" \\
  -d '{
    "model": "${modelId}",
    "messages": [{"role":"user","content":"hi"}]
  }'`
      : `curl ${openaiBase}/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${modelId}",
    "messages": [{"role":"user","content":"hi"}]
  }'`,
  );

  let claudeCurl = $derived(
    hasApiKey
      ? `curl ${claudeBase}/v1/messages \\
  -H "Content-Type: application/json" \\
  -H "x-api-key: ${apiKeyPreview}" \\
  -H "anthropic-version: 2023-06-01" \\
  -d '{
    "model": "${modelId}",
    "max_tokens": 1024,
    "messages": [{"role":"user","content":"hi"}]
  }'`
      : `curl ${claudeBase}/v1/messages \\
  -H "Content-Type: application/json" \\
  -H "anthropic-version: 2023-06-01" \\
  -d '{
    "model": "${modelId}",
    "max_tokens": 1024,
    "messages": [{"role":"user","content":"hi"}]
  }'`,
  );

  let copyHint = $state<string | null>(null);
  async function copy(text: string) {
    try {
      await navigator.clipboard.writeText(text);
      copyHint = "copied";
      setTimeout(() => (copyHint = null), 1200);
    } catch (e) {
      copyHint = "copy failed: " + e;
    }
  }

  // Tab button classes — Tailwind doesn't have a clean `hover:not(.active)`,
  // so we toggle the whole set via Svelte. Keeps the original behavior:
  // active → accent fill, inactive → dim text that brightens on hover.
  const tabBase = "bg-transparent border-0 px-3.5 py-1 text-xs rounded";
  const tabActive = "bg-accent text-bg font-medium";
  const tabIdle = "text-text-dim hover:text-text";
</script>

<div class="flex items-center gap-4 mb-3">
  <h2 class="m-0 text-xs font-semibold tracking-[0.08em] text-text-dim uppercase">API</h2>
  <div class="flex gap-0.5 bg-panel-2 border border-border rounded-md p-0.5">
    <button
      class={`${tabBase} ${activeStyle === "openai" ? tabActive : tabIdle}`}
      onclick={() => (activeStyle = "openai")}
    >
      OpenAI-style
    </button>
    <button
      class={`${tabBase} ${activeStyle === "claude" ? tabActive : tabIdle}`}
      onclick={() => (activeStyle = "claude")}
    >
      Claude-style
    </button>
  </div>
  {#if status.state !== "running"}
    <span class="dim text-[11px] ml-auto">
      server is <span class="text-warn">{status.state}</span> — values shown are what clients will use once you Start
    </span>
  {/if}
</div>

<div class="grid grid-cols-2 gap-6">
  {#if activeStyle === "openai"}
    <div class="flex flex-col gap-1.5 min-w-0">
      <div class="flex flex-col gap-1">
        <span class="dim">Base URL</span>
        <div class="flex items-center gap-2">
          <code class="mono flex-1 px-2.5 py-[7px] bg-bg border border-border rounded text-accent overflow-x-auto whitespace-nowrap">{openaiBase}</code>
          <button onclick={() => copy(openaiBase)}>Copy</button>
        </div>
      </div>
      <div class="flex flex-col gap-1">
        <span class="dim">API key</span>
        <input
          type="password"
          placeholder="(none — auth disabled)"
          value={config.server.api_key ?? ""}
          oninput={(e) => {
            const v = (e.target as HTMLInputElement).value;
            onApiKeyChange(v === "" ? null : v);
          }}
        />
      </div>
      <div class="flex flex-col gap-1">
        <span class="dim">Embedding model</span>
        <select
          value={config.server.embedding_model_id ?? ""}
          onchange={(e) => {
            const v = (e.target as HTMLSelectElement).value;
            onEmbeddingChange(v === "" ? null : v);
          }}
        >
          <option value="">(none — /v1/embeddings disabled)</option>
          {#each downloadedEmbeddings as emb}
            <option value={emb.id}>
              {emb.label} · {emb.approx_size_gb}GB
            </option>
          {/each}
          {#if config.server.embedding_model_id && !downloadedEmbeddings.find((e) => e.id === config.server.embedding_model_id)}
            <option value={config.server.embedding_model_id}>
              {config.server.embedding_model_id} (custom)
            </option>
          {/if}
        </select>
      </div>
      {#if downloadedEmbeddings.length === 0}
        <div class="mt-0.5 text-[11px] leading-normal text-warn">
          No embedding models downloaded yet — pick one below to download first.
        </div>
      {/if}
      {#if availableEmbeddings.length > 0}
        <div class="flex flex-col gap-1">
          <span class="dim">Download embedding</span>
          <div class="flex items-center gap-1.5">
            <select class="flex-1" bind:value={selectedEmbDl}>
              <option value="" disabled>— pick to download —</option>
              {#each availableEmbeddings as emb}
                {@const fits = !systemInfo || emb.min_ram_gb <= systemInfo.ram_gb}
                <option value={emb.id}>
                  {fits ? "" : "⚠ "}{emb.label} · {emb.approx_size_gb}GB · ≥{emb.min_ram_gb}GB RAM
                </option>
              {/each}
            </select>
            <button onclick={triggerEmbeddingDownload} disabled={!selectedEmbDl}>
              Download
            </button>
          </div>
        </div>
      {/if}
      <div class="mt-2 pt-2.5 border-t border-border">
        <div class="dim text-[11px] uppercase tracking-[0.06em] mb-1">Endpoints</div>
        <div class="mono text-xs py-0.5 text-text-dim">POST /chat/completions</div>
        <div class="mono text-xs py-0.5 text-text-dim">POST /completions</div>
        <div class="mono text-xs py-0.5 text-text-dim">
          POST /embeddings <span class="dim">(requires Embedding model)</span>
        </div>
        <div class="mono text-xs py-0.5 text-text-dim">GET  /models</div>
      </div>
    </div>
    <div class="flex flex-col gap-1.5 min-w-0">
      <div class="flex items-center justify-between text-[11px] uppercase tracking-[0.06em] mb-1">
        <span class="dim">curl example</span>
        <button onclick={() => copy(openaiCurl)}>Copy</button>
      </div>
      <pre class="mono m-0 px-3 py-2.5 bg-bg border border-border rounded-md text-[11px] leading-[1.55] text-text overflow-x-auto whitespace-pre">{openaiCurl}</pre>
    </div>
  {:else}
    <div class="flex flex-col gap-1.5 min-w-0">
      <div class="flex flex-col gap-1">
        <span class="dim">Base URL</span>
        <div class="flex items-center gap-2">
          <code class="mono flex-1 px-2.5 py-[7px] bg-bg border border-border rounded text-accent overflow-x-auto whitespace-nowrap">{claudeBase}</code>
          <button onclick={() => copy(claudeBase)}>Copy</button>
        </div>
      </div>
      <div class="flex flex-col gap-1">
        <span class="dim">API key</span>
        <input
          type="password"
          placeholder="(none — auth disabled)"
          value={config.server.api_key ?? ""}
          oninput={(e) => {
            const v = (e.target as HTMLInputElement).value;
            onApiKeyChange(v === "" ? null : v);
          }}
        />
      </div>
      <div class="flex flex-col gap-1">
        <span class="dim">anthropic-version</span>
        <code class="mono inline-block px-2 py-1 bg-bg border border-border rounded">2023-06-01</code>
      </div>
      <div class="mt-2 pt-2.5 border-t border-border">
        <div class="dim text-[11px] uppercase tracking-[0.06em] mb-1">Endpoints</div>
        <div class="mono text-xs py-0.5 text-text-dim">POST /v1/messages</div>
      </div>
    </div>
    <div class="flex flex-col gap-1.5 min-w-0">
      <div class="flex items-center justify-between text-[11px] uppercase tracking-[0.06em] mb-1">
        <span class="dim">curl example</span>
        <button onclick={() => copy(claudeCurl)}>Copy</button>
      </div>
      <pre class="mono m-0 px-3 py-2.5 bg-bg border border-border rounded-md text-[11px] leading-[1.55] text-text overflow-x-auto whitespace-pre">{claudeCurl}</pre>
    </div>
  {/if}
</div>

{#if copyHint}
  <div class="dim mt-1.5 text-[11px] text-right">{copyHint}</div>
{/if}
