<script lang="ts">
  // Builds one of Flash-Next's derived files in the background: the expert pack (`kind` "pack")
  // or the 3-bit copy of the experts ("copy"). A button while it can start (with the reason it
  // cannot), a progress bar and Cancel while it runs, and how the last build ended. Used under the
  // card's pack and copy options and in the doctor's rows for them. Its text is the
  // `flashnext.<kind>.*` family.
  import { t } from "./i18n.svelte";
  import { fill } from "./format";
  import type { BuildKind, FlashNextStatus, PackProgress } from "./api";

  interface Props {
    /** Which file this control builds. */
    kind?: BuildKind;
    status: FlashNextStatus | null;
    /** The latest `lumen://pack` report of this session for this kind, if any. */
    build: PackProgress | null;
    onBuild: () => Promise<void>;
    onCancel: () => Promise<void>;
    /** Starting a build is not possible here (the Flash-Next settings are read-only while another
     *  model is selected). Cancelling one that is already running stays possible. */
    disabled?: boolean;
  }
  let { kind = "pack", status, build, onBuild, onCancel, disabled = false }: Props = $props();

  let starting = $state(false);
  let startError = $state<string | null>(null);

  const key = (k: string) => t(`flashnext.${kind}.${k}`);
  // One build runs at a time: the other kind's shows here as the `building` refusal.
  const running = $derived(build?.state === "running" || status?.building_kind === kind);
  const pct = $derived(
    build && build.layers > 0 ? Math.min(100, (build.layer / build.layers) * 100) : null,
  );
  const gb = (b: number | null | undefined) => (b == null ? "?" : (b / 1e9).toFixed(0));
  const block = $derived((kind === "copy" ? status?.copy_block : status?.build_block) ?? null);
  const fileState = $derived(kind === "copy" ? status?.copy : status?.pack);
  const needBytes = $derived(kind === "copy" ? status?.copy_need_bytes : status?.need_bytes);
  const blockText = $derived(
    block ? fill(key(`block.${block}`), { free: gb(status?.free_bytes), need: gb(needBytes) }) : null,
  );
  // Only the reasons worth telling: an installed model whose file is not there yet.
  const canOffer = $derived(status?.installed === true && fileState !== "complete");

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
          {key("cancel")}
        </button>
      </div>
      <div class="dim text-[10.5px]">
        {build && build.layers > 0
          ? fill(key("building"), { n: build.layer, total: build.layers })
          : key("starting")}
      </div>
    {:else}
      <div class="flex items-center gap-2 flex-wrap">
        <button
          class="px-2.5 py-1 text-[11px] primary disabled:opacity-50 disabled:cursor-not-allowed"
          disabled={disabled || starting || block != null}
          aria-disabled={disabled || starting || block != null || undefined}
          title={blockText ?? key("build.hint")}
          onclick={start}
        >{key("build")}</button>
        {#if blockText}
          <span class="text-warn text-[10.5px]">{blockText}</span>
        {/if}
      </div>
    {/if}
    {#if startError}<div class="text-err text-[10.5px] break-words">{startError}</div>{/if}
    {#if !running && build?.state === "failed"}
      <div class="text-err text-[10.5px] break-words">
        {key("failed")}{build.error ? ` ${build.error}` : ""}
      </div>
    {:else if !running && build?.state === "cancelled"}
      <div class="dim text-[10.5px]">{key("cancelled")}</div>
    {:else if !running && build?.state === "done"}
      <div class="text-ok text-[10.5px]">{key("built")}</div>
    {/if}
  </div>
{/if}
