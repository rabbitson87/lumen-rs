<script lang="ts">
  // The memory plan of Qwen 3.8 Flash-Next, read-only. The other families fit their weights in
  // memory and trade context against KV; this one streams its experts from the SSD, and the
  // server sizes the expert pool (= decode speed) from what the GPU's working set leaves after the
  // trunk and a prefill's reserve, KV of the longest prompt included. See `memory-estimate.ts`.
  //
  // It shows what a launch carries, from the Flash-Next card's settings (and any env override of
  // theirs, passed as `env`); the choices themselves are made in that card, above it.
  import { t } from "./i18n.svelte";
  import type { FlashNextStatus } from "./api";
  import { fill, fmtTps } from "./format";
  import {
    FLASH_NEXT,
    FLASH_NEXT_DECODE_TPS,
    flashNextDecodeTps,
    flashNextDiskBytes,
    flashNextEffective,
    flashNextMaxContext,
    flashNextMinRamGb,
    flashNextPlan,
    flashNextPlanFits,
    flashNextUsableBytes,
  } from "./memory-estimate";

  interface Props {
    /** Installed RAM (GB, from the system probe). */
    ramGb?: number;
    /** The cap on MLX memory in GiB the launch will carry (`LUMEN_MEMORY_LIMIT_GB`); `null` = none. */
    memoryLimitGb?: number | null;
    /** What a launch carries: the Flash-Next settings, replaced by env overrides of the same name. */
    env: Record<string, string>;
    /** Current structured `context.max` (reaches the server as `LUMEN_MAX_CTX`). */
    contextMax?: number;
    status?: FlashNextStatus | null;
  }
  let { ramGb, memoryLimitGb = null, env, contextMax, status = null }: Props = $props();

  // The system probe falls back to 16 GB when it cannot read the RAM; so does this.
  const ram = $derived(ramGb && ramGb > 0 ? ramGb : 16);
  const usable = $derived(flashNextUsableBytes(ram, memoryLimitGb));
  // What the server does on this Mac when nothing is overridden.
  const defaults = $derived(flashNextPlan(usable));
  const eff = $derived(flashNextEffective({ env, contextMax, usableBytes: usable, status }));

  const chunk = $derived(eff.chunk);

  const est = $derived(eff.estimate);
  // Whether the model runs at this chunk and prompt at all (the smallest pool fits).
  const loads = $derived(flashNextPlanFits(est.plan));
  const loadNeed = $derived(
    FLASH_NEXT.trunkBytes + FLASH_NEXT.poolFloorBytes + est.plan.reserveBytes,
  );
  const longestAtFloor = $derived(flashNextMaxContext(usable, chunk, null));
  // Longest prompt: beside the chosen pool, or with `auto` before the pool is squeezed to its floor.
  const maxCtx = $derived(flashNextMaxContext(usable, chunk, eff.poolExplicit ? est.poolBytes : null));
  const minRam = $derived(flashNextMinRamGb());
  const tps = $derived(flashNextDecodeTps(est.poolBytes));
  const tpsPack = $derived(flashNextDecodeTps(est.poolBytes, true));
  const buildsPack = $derived(
    env.LUMEN_MOE_PACK?.trim().toLowerCase() === "build" && status?.pack !== "complete",
  );
  const disk = $derived(
    flashNextDiskBytes({
      pack: eff.pack || buildsPack,
      mtp: status?.mtp_present === true,
    }),
  );
  const overBy = $derived(Math.max(0, est.residentBytes - est.usableBytes));
  const capped = $derived(
    memoryLimitGb != null &&
      memoryLimitGb > 0 &&
      flashNextUsableBytes(ram) > flashNextUsableBytes(ram, memoryLimitGb),
  );

  const gb = (b: number, d = 1) => (b / 1e9).toFixed(d);
  const fmtTok = (n: number) => n.toLocaleString("en-US");
</script>

<div class="text-xs">
  <div class="dim text-[10.5px] mb-1.5">
    {fill(t("flashnext.geometry"), { kb: (FLASH_NEXT.kvBytesPerToken / 1000).toFixed(1) })}
  </div>

  <!-- Can this Mac run it at this context -->
  <div class="text-[10.5px] mb-1 leading-snug dim">
    {fill(t("flashnext.thisMac"), { ram, usable: gb(usable) })}
    {#if capped}
      {fill(t("flashnext.thisMac.limit"), { n: memoryLimitGb ?? 0 })}
    {/if}
    {#if defaults.tight}
      {fill(t("flashnext.tight"), { chunk: defaults.chunk, ctx: fmtTok(defaults.ctxTokens) })}
    {/if}
  </div>
  <div class="text-[10.5px] mb-2.5 leading-snug" class:text-ok={loads} class:text-err={!loads}>
    {#if loads}
      {fill(t("flashnext.fits"), { ctx: fmtTok(est.plan.ctxTokens), need: gb(loadNeed) })}
    {:else if longestAtFloor > 0}
      {fill(t("flashnext.noFit"), {
        ctx: fmtTok(est.plan.ctxTokens),
        need: gb(loadNeed),
        max: fmtTok(longestAtFloor),
        chunk,
      })}
    {:else}
      {fill(t("flashnext.noFitAny"), { min: minRam })}
    {/if}
  </div>

  <!-- The pool notes -->
  <div class="dim text-[10px] mb-2.5 leading-snug">
    {t("flashnext.pool.hint")}
    {#if est.poolRaisedToFloor}
      <span class="text-warn">{fill(t("flashnext.pool.floor"), { floor: gb(FLASH_NEXT.poolFloorBytes, 2) })}</span>
    {:else if est.oneStagingRegion}
      <span>{fill(t("flashnext.pool.oneRegion"), { gb: gb(FLASH_NEXT.poolPrefetchBytes) })}</span>
    {/if}
  </div>

  <!-- Resident memory against what the GPU may fill -->
  <div class="h-2.5 w-full bg-panel-2 rounded-sm overflow-hidden mt-2 mb-1">
    <div
      class={`h-full ${est.fits ? "bg-ok" : "bg-err"}`}
      style={`width: ${Math.min(100, est.utilization * 100).toFixed(1)}%`}
    ></div>
  </div>
  <div class="flex items-baseline justify-between gap-2 mb-1.5">
    <span class="dim text-[10px] shrink-0">{t("flashnext.peak")}</span>
    <span class="mono text-[11px] whitespace-nowrap {est.fits ? '' : 'text-err'}">
      {gb(est.residentBytes)} / {gb(est.usableBytes)} GB · {(est.utilization * 100).toFixed(0)}%{est.fits ? "" : " ⚠"}
    </span>
  </div>
  <div class="grid grid-cols-[1fr_auto] gap-x-3 gap-y-0.5 text-[10.5px] mono mb-2">
    <span class="dim">{t("flashnext.row.trunk")}</span><span class="text-right">{gb(FLASH_NEXT.trunkBytes, 2)} GB</span>
    <span class="dim">{t("flashnext.row.pool")}</span><span class="text-right">{gb(est.poolBytes, 2)} GB</span>
    <span class="dim">{t("flashnext.row.buffers")}</span><span class="text-right">{gb(est.plan.buffersBytes, 2)} GB</span>
    <span class="dim">{t("flashnext.row.kv")}</span><span class="text-right">{gb(est.plan.kvBytes, 2)} GB</span>
    <span class="dim">{t("flashnext.row.scores")}</span><span class="text-right">{gb(est.plan.scoresBytes, 2)} GB</span>
  </div>
  {#if loads && !est.fits}
    <div class="text-err text-[10px] mb-2 leading-snug">
      {fill(t("flashnext.overBy"), { over: gb(overBy), pool: gb(est.plan.poolBytes) })}
    </div>
  {/if}

  <div class="flex items-baseline justify-between gap-2 mb-0.5 pt-1.5 border-t border-text-dim/15">
    <span class="text-[11px] shrink-0">{t("flashnext.autoPool")}</span>
    <span class="mono whitespace-nowrap {loads ? '' : 'text-err'}">{gb(est.plan.poolBytes)} GB</span>
  </div>
  <div class="flex items-baseline justify-between gap-2 mb-2">
    <span class="text-[11px] shrink-0">
      {!eff.poolExplicit
        ? fill(t("flashnext.maxCtx.auto"), { floor: gb(FLASH_NEXT.poolFloorBytes, 1) })
        : t("flashnext.maxCtx.manual")}
    </span>
    <span class="mono font-semibold whitespace-nowrap {maxCtx > 0 ? 'text-ok' : 'text-err'}">{fmtTok(maxCtx)} tok</span>
  </div>

  <!-- Decode speed -->
  <div class="text-[10px] text-text-dim uppercase tracking-wider mb-1">{t("flashnext.speed.title")}</div>
  {#if tps}
    <div class="text-[11px] mb-1">
      {fill(t("flashnext.speed.estimate"), { pool: gb(est.poolBytes), tps: fmtTps(tps.tps) })}
      {#if tpsPack}
        <span class="dim">· {fill(t("flashnext.speed.withPack"), { tps: fmtTps(tpsPack.tps) })}</span>
      {/if}
      {#if tps.basis === "below"}
        <span class="text-warn">{t("flashnext.speed.below")}</span>
      {:else if tps.basis === "above"}
        <span class="dim">{t("flashnext.speed.above")}</span>
      {/if}
    </div>
  {/if}
  <div class="overflow-x-auto">
    <table class="text-[10.5px] mono border-collapse">
      <thead>
        <tr class="text-text-dim">
          <th class="text-left font-normal pr-3 pb-0.5">{t("flashnext.speed.col.pool")}</th>
          <th class="text-right font-normal px-2 pb-0.5">{t("flashnext.speed.col.shards")}</th>
          <th class="text-right font-normal px-2 pb-0.5">{t("flashnext.speed.col.pack")}</th>
        </tr>
      </thead>
      <tbody>
        {#each FLASH_NEXT_DECODE_TPS as p}
          <tr>
            <td class="text-left pr-3 text-text-dim">{p.poolGb} GB{p.emulated ? " *" : ""}</td>
            <td class="text-right px-2">{p.shards ?? "—"}</td>
            <td class="text-right px-2">{p.pack ?? "—"}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
  <div class="dim text-[10px] mt-1 mb-2 leading-snug">{t("flashnext.speed.note")}</div>

  <!-- Disk -->
  <div class="flex items-baseline justify-between gap-2 mb-0.5">
    <span class="text-[11px] shrink-0">{t("flashnext.disk")}</span>
    <span class="mono whitespace-nowrap">{gb(disk, 1)} GB</span>
  </div>
  <div class="dim text-[10px] mb-1 leading-snug">
    {fill(t("flashnext.disk.detail"), { model: gb(FLASH_NEXT.diskModelBytes, 0) })}{eff.pack || buildsPack
      ? ` + ${gb(FLASH_NEXT.diskPackBytes, 0)} ${t("flashnext.disk.pack")}`
      : ""}{status?.mtp_present ? ` + ${gb(FLASH_NEXT.diskMtpBytes, 1)} ${t("flashnext.disk.mtp")}` : ""}
  </div>
  <div class="dim text-[10px] leading-snug">{t("flashnext.settings.hint")}</div>
</div>
