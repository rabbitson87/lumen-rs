<script lang="ts">
  import type { PersistentConfig, ServerStatus, Catalog } from "./api";

  interface Props {
    config: PersistentConfig;
    status: ServerStatus;
    catalog: Catalog;
    onEmbeddingChange: (value: string | null) => void;
    onApiKeyChange: (value: string | null) => void;
  }
  let { config, status, catalog, onEmbeddingChange, onApiKeyChange }: Props =
    $props();

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
</script>

<div class="api-card-h">
  <h2 class="api-h">API</h2>
  <div class="tabs">
    <button class="tab" class:active={activeStyle === "openai"} onclick={() => (activeStyle = "openai")}>
      OpenAI-style
    </button>
    <button class="tab" class:active={activeStyle === "claude"} onclick={() => (activeStyle = "claude")}>
      Claude-style
    </button>
  </div>
  {#if status.state !== "running"}
    <span class="dim status-note">server is <span class="warn">{status.state}</span> — values shown are what clients will use once you Start</span>
  {/if}
</div>

<div class="api-body">
  {#if activeStyle === "openai"}
    <div class="api-col">
      <div class="kv">
        <span class="dim">Base URL</span>
        <div class="copyable">
          <code class="url mono">{openaiBase}</code>
          <button onclick={() => copy(openaiBase)}>Copy</button>
        </div>
      </div>
      <div class="kv">
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
      <div class="kv">
        <span class="dim">Embedding model</span>
        <select
          value={config.server.embedding_model_id ?? ""}
          onchange={(e) => {
            const v = (e.target as HTMLSelectElement).value;
            onEmbeddingChange(v === "" ? null : v);
          }}
        >
          <option value="">(none — /v1/embeddings disabled)</option>
          {#each catalog.embeddings as emb}
            <option value={emb.id}>
              {emb.label} · {emb.approx_size_gb}GB · ≥{emb.min_ram_gb}GB RAM
            </option>
          {/each}
          {#if config.server.embedding_model_id && !catalog.embeddings.find((e) => e.id === config.server.embedding_model_id)}
            <option value={config.server.embedding_model_id}>
              {config.server.embedding_model_id} (custom)
            </option>
          {/if}
        </select>
      </div>
      {#if config.server.embedding_model_id}
        {@const emb = catalog.embeddings.find((e) => e.id === config.server.embedding_model_id)}
        {#if emb}
          <div class="dl-hint dim">{emb.notes}</div>
        {/if}
      {/if}
      <div class="endpoints">
        <div class="ep-h dim">Endpoints</div>
        <div class="ep mono">POST /chat/completions</div>
        <div class="ep mono">POST /completions</div>
        <div class="ep mono">POST /embeddings <span class="dim">(requires Embedding model)</span></div>
        <div class="ep mono">GET  /models</div>
      </div>
    </div>
    <div class="api-col">
      <div class="curl-h">
        <span class="dim">curl example</span>
        <button onclick={() => copy(openaiCurl)}>Copy</button>
      </div>
      <pre class="curl mono">{openaiCurl}</pre>
    </div>
  {:else}
    <div class="api-col">
      <div class="kv">
        <span class="dim">Base URL</span>
        <div class="copyable">
          <code class="url mono">{claudeBase}</code>
          <button onclick={() => copy(claudeBase)}>Copy</button>
        </div>
      </div>
      <div class="kv">
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
      <div class="kv">
        <span class="dim">anthropic-version</span>
        <code class="mono inline-code">2023-06-01</code>
      </div>
      <div class="endpoints">
        <div class="ep-h dim">Endpoints</div>
        <div class="ep mono">POST /v1/messages</div>
      </div>
    </div>
    <div class="api-col">
      <div class="curl-h">
        <span class="dim">curl example</span>
        <button onclick={() => copy(claudeCurl)}>Copy</button>
      </div>
      <pre class="curl mono">{claudeCurl}</pre>
    </div>
  {/if}
</div>

{#if copyHint}
  <div class="copy-hint dim">{copyHint}</div>
{/if}

<style>
  .api-card-h {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 12px;
  }
  .api-h {
    margin: 0;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: var(--text-dim);
    text-transform: uppercase;
  }
  .tabs {
    display: flex;
    gap: 2px;
    background: var(--panel-2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 2px;
  }
  .tab {
    background: transparent;
    border: none;
    padding: 4px 14px;
    font-size: 12px;
    border-radius: 4px;
    color: var(--text-dim);
  }
  .tab.active {
    background: var(--accent);
    color: var(--bg);
    font-weight: 500;
  }
  .tab:hover:not(.active) {
    color: var(--text);
    background: transparent;
  }
  .status-note {
    font-size: 11px;
    margin-left: auto;
  }
  .warn {
    color: var(--warn);
  }

  .api-body {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
  }
  .api-col {
    display: flex;
    flex-direction: column;
    gap: 6px;
    min-width: 0;
  }

  .copyable {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .url {
    flex: 1;
    padding: 7px 10px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 5px;
    color: var(--accent);
    overflow-x: auto;
    white-space: nowrap;
  }

  .endpoints {
    margin-top: 8px;
    padding-top: 10px;
    border-top: 1px solid var(--border);
  }
  .ep-h {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
  }
  .ep {
    font-size: 12px;
    padding: 2px 0;
    color: var(--text-dim);
  }

  .curl-h {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
  }
  .curl {
    margin: 0;
    padding: 10px 12px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 11px;
    line-height: 1.55;
    color: var(--text);
    overflow-x: auto;
    white-space: pre;
  }
  .inline-code {
    padding: 4px 8px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    display: inline-block;
  }

  .copy-hint {
    margin-top: 6px;
    font-size: 11px;
    text-align: right;
  }
  .dl-hint {
    margin-top: 2px;
    margin-left: 130px;
    font-size: 11px;
    line-height: 1.5;
  }
</style>
