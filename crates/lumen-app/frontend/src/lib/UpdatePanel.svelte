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

<div class="px-4 pt-3 pb-4 text-xs">
  <div class="flex items-start justify-between gap-4 mb-2.5">
    <div>
      <div class="mono text-sm font-medium">
        Lumen <span class="dim">v</span>{info?.current_version ?? "?"}
      </div>
      {#if info?.available && info.latest_version}
        <div class="mt-0.5">
          <span class="text-ok font-medium">v{info.latest_version}</span>
          <span class="dim">available</span>
          {#if info.published_at}
            <span class="dim mono">· {info.published_at}</span>
          {/if}
        </div>
      {:else if info && !info.available}
        <div class="dim">You're on the latest version.</div>
      {/if}
    </div>
    <div class="flex gap-2">
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
    <div class="mono text-err px-2.5 py-1.5 bg-err/10 rounded mb-2">{error}</div>
  {/if}

  {#if progress}
    <div class="my-2">
      <div class="w-full h-1.5 bg-panel-2 rounded-[3px] overflow-hidden">
        <div
          class="h-full bg-accent transition-[width] duration-200 ease-out"
          style="width: {progress.total_bytes
            ? (progress.downloaded_bytes / progress.total_bytes) * 100
            : 50}%"
        ></div>
      </div>
      <div class="dim mono mt-1 text-[11px]">
        {bytes(progress.downloaded_bytes)}
        {#if progress.total_bytes} / {bytes(progress.total_bytes)}{/if}
        {progress.done ? "— applying…" : ""}
      </div>
    </div>
  {/if}

  {#if info?.available && info.release_notes}
    <div class="mt-3 border-t border-border pt-2">
      <div class="dim text-[11px] uppercase tracking-[0.06em] mb-1.5">Release notes</div>
      <pre class="mono m-0 whitespace-pre-wrap wrap-break-word text-[11px] leading-normal text-text max-h-50 overflow-y-auto">{info.release_notes}</pre>
    </div>
  {/if}
</div>
