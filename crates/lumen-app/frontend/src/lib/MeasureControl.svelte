<script lang="ts">
  // "Measure this Mac": the memory-bandwidth test and then the SSD read test, as one action. A
  // button while it can start, a progress line and Cancel while it runs, and how the last run
  // ended (what failed, what was skipped). Decode speed rests on both numbers, so this is how
  // the card gets from the reference table to an estimate for this Mac.
  import { t } from "./i18n.svelte";
  import { fill } from "./format";
  import type { FlashNextStatus, MeasureProgress } from "./api";

  interface Props {
    /** Both numbers are measured: the button then offers to measure again. */
    measured: boolean;
    status: FlashNextStatus | null;
    /** The latest `lumen://measure` report of this session, if any. */
    measure: MeasureProgress | null;
    /** The Flash-Next model is the selected one; otherwise starting is disabled. */
    enabled: boolean;
    onMeasure: () => Promise<void>;
    onCancel: () => Promise<void>;
  }
  let { measured, status, measure, enabled, onMeasure, onCancel }: Props = $props();

  let starting = $state(false);
  let startError = $state<string | null>(null);

  const running = $derived(measure?.state === "running");
  const installed = $derived(status?.installed === true);

  async function start() {
    starting = true;
    startError = null;
    try {
      await onMeasure();
    } catch (e) {
      startError = String(e);
    } finally {
      starting = false;
    }
  }
</script>

<div class="flex flex-col gap-1">
  {#if running}
    <div class="flex items-center gap-2">
      <div class="h-1.5 flex-1 bg-panel-2 rounded-sm overflow-hidden">
        <div class="h-full w-1/4 bg-accent animate-pulse"></div>
      </div>
      <button class="px-2 py-0.5 text-[10.5px] danger" onclick={() => onCancel()}>
        {t("flashnext.measure.cancel")}
      </button>
    </div>
    <div class="dim text-[10.5px]">
      {measure?.stage === "ssd" ? t("flashnext.measure.ssd") : t("flashnext.measure.memory")}
    </div>
  {:else}
    <div class="flex items-center gap-2 flex-wrap">
      <button
        class="px-2.5 py-1 text-[11px] primary disabled:opacity-50 disabled:cursor-not-allowed"
        disabled={!enabled || starting}
        aria-disabled={!enabled || starting || undefined}
        title={t("flashnext.measure.hint")}
        onclick={start}
      >{measured ? t("flashnext.measure.again") : t("flashnext.measure.button")}</button>
      {#if status && !installed}
        <span class="dim text-[10.5px]">{t("flashnext.measure.needsModel")}</span>
      {/if}
    </div>
    <div class="dim text-[10px] leading-snug">{t("flashnext.measure.hint")}</div>
  {/if}
  {#if startError}<div class="text-err text-[10.5px] break-words">{startError}</div>{/if}

  {#if !running && measure}
    {#if measure.state === "cancelled"}
      <div class="dim text-[10.5px]">{t("flashnext.measure.cancelled")}</div>
    {:else if measure.state === "failed"}
      <div class="text-err text-[10.5px]">{t("flashnext.measure.failed")}</div>
    {:else if measure.state === "done" && !measure.memory_error && !measure.ssd_error}
      <div class="text-ok text-[10.5px]">{t("flashnext.measure.done")}</div>
    {/if}
    {#if measure.memory_error}
      <div class="text-err text-[10.5px] break-words">
        {fill(t("flashnext.measure.memoryError"), { error: measure.memory_error })}
      </div>
    {/if}
    {#if measure.ssd_error}
      <div class="text-err text-[10.5px] break-words">
        {fill(t("flashnext.measure.ssdError"), { error: measure.ssd_error })}
      </div>
    {/if}
    {#if measure.ssd_skipped && measure.state !== "cancelled"}
      <div class="dim text-[10.5px]">{t("flashnext.measure.ssdSkipped")}</div>
    {/if}
  {/if}
</div>
