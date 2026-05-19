<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import {
    api,
    onUpdateProgress,
    type UpdateInfo,
    type UpdateProgress,
  } from "./api";
  import { bytes } from "./format";

  interface Props {
    /** Set when the parent has a running server — we surface a confirm before
     *  install so the user can stop it. */
    serverRunning: boolean;
  }
  let { serverRunning }: Props = $props();

  let info = $state<UpdateInfo | null>(null);
  let checking = $state(false);
  let installing = $state(false);
  let error = $state<string | null>(null);
  let progress = $state<UpdateProgress | null>(null);

  let unlistenProgress: (() => void) | undefined;

  onMount(async () => {
    unlistenProgress = await onUpdateProgress((p) => (progress = p));
    // Also fetch current version so the panel shows something useful before
    // the first check.
    try {
      const ver = await api.currentVersion();
      info = {
        available: false,
        current_version: ver,
        latest_version: null,
        release_notes: null,
        published_at: null,
      };
    } catch (e) {
      error = String(e);
    }
  });

  onDestroy(() => unlistenProgress?.());

  async function check() {
    checking = true;
    error = null;
    try {
      info = await api.checkForUpdates();
    } catch (e) {
      error = String(e);
    } finally {
      checking = false;
    }
  }

  async function install() {
    if (!info?.available) return;
    if (serverRunning) {
      const ok = confirm(
        "The inference server is running. Installing the update will stop it and restart the app. Continue?",
      );
      if (!ok) return;
    }
    installing = true;
    error = null;
    progress = null;
    try {
      await api.installUpdate();
      // app.restart() never returns from the frontend's POV.
    } catch (e) {
      error = String(e);
      installing = false;
    }
  }
</script>

<div class="upd-body">
  <div class="upd-header">
    <div>
      <div class="upd-version mono">
        Lumen <span class="dim">v</span>{info?.current_version ?? "?"}
      </div>
      {#if info?.available && info.latest_version}
        <div class="upd-latest">
          <span class="ok">v{info.latest_version}</span>
          <span class="dim">available</span>
          {#if info.published_at}
            <span class="dim mono">· {info.published_at}</span>
          {/if}
        </div>
      {:else if info && !info.available}
        <div class="dim">You're on the latest version.</div>
      {/if}
    </div>
    <div class="upd-actions">
      <button onclick={check} disabled={checking || installing}>
        {checking ? "Checking…" : "Check for updates"}
      </button>
      {#if info?.available}
        <button class="primary" onclick={install} disabled={installing}>
          {installing ? "Installing…" : "Install & restart"}
        </button>
      {/if}
    </div>
  </div>

  {#if error}
    <div class="upd-error mono">{error}</div>
  {/if}

  {#if progress}
    <div class="upd-progress">
      <div class="upd-bar">
        <div
          class="upd-fill"
          style="width: {progress.total_bytes
            ? (progress.downloaded_bytes / progress.total_bytes) * 100
            : 50}%"
        ></div>
      </div>
      <div class="dim mono upd-pct">
        {bytes(progress.downloaded_bytes)}
        {#if progress.total_bytes} / {bytes(progress.total_bytes)}{/if}
        {progress.done ? "— applying…" : ""}
      </div>
    </div>
  {/if}

  {#if info?.available && info.release_notes}
    <div class="upd-notes">
      <div class="upd-notes-h dim">Release notes</div>
      <pre class="mono">{info.release_notes}</pre>
    </div>
  {/if}
</div>

<style>
  .upd-body {
    padding: 12px 16px 16px;
    font-size: 12px;
  }
  .upd-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 16px;
    margin-bottom: 10px;
  }
  .upd-version {
    font-size: 14px;
    font-weight: 500;
  }
  .upd-latest {
    margin-top: 2px;
  }
  .ok {
    color: var(--ok);
    font-weight: 500;
  }
  .upd-actions {
    display: flex;
    gap: 8px;
  }
  .upd-error {
    color: var(--err);
    padding: 6px 10px;
    background: rgba(255, 122, 122, 0.08);
    border-radius: 4px;
    margin-bottom: 8px;
  }
  .upd-progress {
    margin: 8px 0;
  }
  .upd-bar {
    width: 100%;
    height: 6px;
    background: var(--panel-2);
    border-radius: 3px;
    overflow: hidden;
  }
  .upd-fill {
    height: 100%;
    background: var(--accent);
    transition: width 200ms ease;
  }
  .upd-pct {
    margin-top: 4px;
    font-size: 11px;
  }
  .upd-notes {
    margin-top: 12px;
    border-top: 1px solid var(--border);
    padding-top: 8px;
  }
  .upd-notes-h {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
  }
  .upd-notes pre {
    margin: 0;
    white-space: pre-wrap;
    word-break: break-word;
    font-size: 11px;
    line-height: 1.5;
    color: var(--text);
    max-height: 200px;
    overflow-y: auto;
  }
</style>
