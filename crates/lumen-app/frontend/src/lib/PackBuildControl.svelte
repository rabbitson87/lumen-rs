<script lang="ts">
  // Builds the Flash-Next expert pack in the background: a button while it can start (with the
  // reason it cannot), a progress bar and Cancel while it runs, and how the last build ended.
  // Used under the pack option and in the doctor's pack row.
  import { t } from "./i18n.svelte";
  import { fill } from "./format";
  import type { FlashNextStatus, PackProgress } from "./api";

  interface Props {
    status: FlashNextStatus | null;
    /** The latest `lumen://pack` report of this session, if any. */
    build: PackProgress | null;
    onBuild: () => Promise<void>;
    onCancel: () => Promise<void>;
    /** Starting a build is not possible here (the Flash-Next settings are read-only while another
     *  model is selected). Cancelling one that is already running stays possible. */
    disabled?: boolean;
  }
  let { status, build, onBuild, onCancel, disabled = false }: Props = $props();

  let starting = $state(false);
  let startError = $state<string | null>(null);

  const running = $derived(build?.state === "running" || status?.building === true);
  const pct = $derived(
    build && build.layers > 0 ? Math.min(100, (build.layer / build.layers) * 100) : null,
  );
  const gb = (b: number | null | undefined) => (b == null ? "?" : (b / 1e9).toFixed(0));
  const block = $derived(status?.build_block ?? null);
  // Only the reasons worth telling: an installed model whose pack is not there yet.
  const canOffer = $derived(status?.installed === true && status.pack !== "complete");

  async function start() {
    starting = true;
    startError = null;
    try {
      await onBuild();
    } catch (e) {
      startError = String(e);
    } finally {
      starting = false;
    }
  }
</script>

{#if canOffer || running}
  <div class="flex flex-col gap-1">
    {#if running}
      <div class="flex items-center gap-2">
        <div class="h-1.5 flex-1 bg-panel-2 rounded-sm overflow-hidden">
          {#if pct != null}
            <div class="h-full bg-accent transition-[width] duration-300" style={`width: ${pct}%`}></div>
          {:else}
            <div class="h-full w-1/4 bg-accent animate-pulse"></div>
          {/if}
        </div>
        <button class="px-2 py-0.5 text-[10.5px] danger" onclick={() => onCancel()}>
          {t("flashnext.pack.cancel")}
        </button>
      </div>
      <div class="dim text-[10.5px]">
        {build && build.layers > 0
          ? fill(t("flashnext.pack.building"), { n: build.layer, total: build.layers })
          : t("flashnext.pack.starting")}
      </div>
    {:else}
      <div class="flex items-center gap-2 flex-wrap">
        <button
          class="px-2.5 py-1 text-[11px] primary disabled:opacity-50 disabled:cursor-not-allowed"
          disabled={disabled || starting || block != null}
          aria-disabled={disabled || starting || block != null || undefined}
          title={block ? fill(t(`flashnext.pack.block.${block}`), { free: gb(status?.free_bytes), need: gb(status?.need_bytes) }) : t("flashnext.pack.build.hint")}
          onclick={start}
        >{t("flashnext.pack.build")}</button>
        {#if block}
          <span class="text-warn text-[10.5px]">
            {fill(t(`flashnext.pack.block.${block}`), { free: gb(status?.free_bytes), need: gb(status?.need_bytes) })}
          </span>
        {/if}
      </div>
    {/if}
    {#if startError}<div class="text-err text-[10.5px] break-words">{startError}</div>{/if}
    {#if !running && build?.state === "failed"}
      <div class="text-err text-[10.5px] break-words">
        {t("flashnext.pack.failed")}{build.error ? ` ${build.error}` : ""}
      </div>
    {:else if !running && build?.state === "cancelled"}
      <div class="dim text-[10.5px]">{t("flashnext.pack.cancelled")}</div>
    {:else if !running && build?.state === "done"}
      <div class="text-ok text-[10.5px]">{t("flashnext.pack.built")}</div>
    {/if}
  </div>
{/if}
