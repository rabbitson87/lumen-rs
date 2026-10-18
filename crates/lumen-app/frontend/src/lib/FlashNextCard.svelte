<script lang="ts">
  // The Flash-Next card of the Tuning tab: the options as pickers with preset chips, a status
  // line each, a live summary of what the server will run with, and (beside them) the memory plan
  // that follows from the choices. Choosing a chip stores one typed setting through `onChange`
  // (`[flash_next]` in the app's config); an option's default chip stores "unset", so an untouched
  // option stays the server's own default.
  //
  // The settings are only operable while the Flash-Next model is the selected one (`enabled`).
  // Otherwise every control is disabled and the stored values are shown read-only.
  import { t } from "./i18n.svelte";
  import ChoiceChips from "./ChoiceChips.svelte";
  import FlashNextCalculator from "./FlashNextCalculator.svelte";
  import PackBuildControl from "./PackBuildControl.svelte";
  import PoolPicker from "./PoolPicker.svelte";
  import PromptPicker from "./PromptPicker.svelte";
  import type { FlashNextSettings, FlashNextStatus, MeasureProgress, PackProgress } from "./api";
  import type { Chip } from "./chips";
  import { fill } from "./format";
  import {
    FLASH_NEXT_LANE_PRESETS,
    FLASH_NEXT_LANES_DEFAULT,
    envChoice,
    flashNextEffective,
    flashNextEnv,
    flashNextHardware,
    flashNextIsDefault,
    flashNextPlan,
    flashNextShadowedKeys,
    flashNextUsableBytes,
    lanesValue,
    mtpMode,
    parseGbInput,
  } from "./memory-estimate";

  interface Props {
    /** The stored settings. */
    settings: FlashNextSettings;
    /** The env overrides: the escape hatch, applied after the settings, so they win. */
    overrides: Record<string, string>;
    /** The Flash-Next model is the selected one: the controls work. */
    enabled: boolean;
    ramGb?: number;
    /** The memory cap (GiB) the launch carries; `null` = none. */
    memoryLimitGb?: number | null;
    /** The Max context card. */
    contextMax?: number;
    status: FlashNextStatus | null;
    /** The latest expert-pack build report of this session. */
    build: PackProgress | null;
    /** The latest Measure report of this session. */
    measure: MeasureProgress | null;
    /** Store some settings (`null` = back to the server's default). */
    onChange: (patch: Partial<FlashNextSettings>) => void;
    onReset: () => void;
    /** Remove env overrides that replace the card's settings. */
    onClearOverrides: (keys: string[]) => void;
    onBuild: () => Promise<void>;
    onCancelBuild: () => Promise<void>;
    onMeasure: () => Promise<void>;
    onCancelMeasure: () => Promise<void>;
  }
  let {
    settings,
    overrides,
    enabled,
    ramGb,
    memoryLimitGb = null,
    contextMax,
    status,
    build,
    measure,
    onChange,
    onReset,
    onClearOverrides,
    onBuild,
    onCancelBuild,
    onMeasure,
    onCancelMeasure,
  }: Props = $props();

  const ram = $derived(ramGb && ramGb > 0 ? ramGb : 16);
  const usable = $derived(flashNextUsableBytes(ram, memoryLimitGb));
  // What a launch carries: the settings, replaced by overrides of the same name.
  const env = $derived(flashNextEnv(settings, overrides));
  // This Mac's measured memory and SSD speeds; the card shows no speed for it until both exist.
  const hw = $derived(flashNextHardware(status?.hardware).hw);
  const eff = $derived(flashNextEffective({ env, contextMax, usableBytes: usable, status, hw }));
  const defaults = $derived(flashNextPlan(usable));
  const shadowed = $derived(flashNextShadowedKeys(overrides));
  const off = $derived(!enabled);

  const packMode = $derived(envChoice(env.LUMEN_MOE_PACK, ["auto", "off", "build"] as const, "auto"));
  const head = $derived(mtpMode(env.LUMEN_QWEN4_MTP));
  const ioMode = $derived(envChoice(env.LUMEN_MOE_IO_MODE, ["auto", "direct", "buffered"] as const, "auto"));
  const lanes = $derived(lanesValue(env.LUMEN_MOE_IO_LANES));
  const poolGb = $derived(
    eff.poolExplicit ? (parseGbInput(env.LUMEN_MOE_EXPERT_CACHE_GB ?? "") ?? null) : null,
  );
  const isDefault = $derived(flashNextIsDefault(settings));
  const maxCtxLowers = $derived(
    eff.launch.maxCtx != null && eff.launch.maxCtx < eff.launch.promptCap ? eff.launch.maxCtx : null,
  );

  const CHUNKS = [256, 512, 1024, 2048];
  const chunkOptions = $derived(
    CHUNKS.includes(eff.chunk) ? CHUNKS : [...CHUNKS, eff.chunk].sort((a, b) => a - b),
  );

  const gb = (b: number | null | undefined, d = 0) => (b == null ? "?" : (b / 1e9).toFixed(d));

  const packChips = $derived<Chip[]>([
    { value: "auto", label: t("flashnext.choice.auto"), title: t("flashnext.pack.mode.auto") },
    { value: "off", label: t("flashnext.choice.off"), title: t("flashnext.pack.mode.off") },
    { value: "build", label: t("flashnext.choice.build"), title: t("flashnext.pack.mode.build") },
  ]);
  // Off is the default (nothing is sent); Auto and On are the opt-ins.
  const headChips = $derived<Chip[]>([
    { value: "off", label: t("flashnext.choice.off"), sub: t("flashnext.default"), title: t("flashnext.mtp.mode.off") },
    { value: "auto", label: t("flashnext.choice.auto"), title: t("flashnext.mtp.mode.auto") },
    { value: "on", label: t("flashnext.choice.on"), title: t("flashnext.mtp.mode.on") },
  ]);
  const ioChips = $derived<Chip[]>([
    { value: "auto", label: t("flashnext.choice.auto"), title: t("flashnext.io.mode.auto") },
    { value: "direct", label: t("flashnext.choice.direct"), title: t("flashnext.io.mode.direct") },
    { value: "buffered", label: t("flashnext.choice.buffered"), title: t("flashnext.io.mode.buffered") },
  ]);
  const laneChips = $derived<Chip[]>(
    [...new Set<number>([...FLASH_NEXT_LANE_PRESETS, lanes])]
      .sort((a, b) => a - b)
      .map((n) => ({
        value: String(n),
        label: String(n),
        sub: n === FLASH_NEXT_LANES_DEFAULT ? t("flashnext.default") : undefined,
      })),
  );

  const summary = $derived(
    fill(t("flashnext.summary"), {
      pool: (eff.estimate.poolBytes / 1e9).toFixed(1),
      how: eff.poolExplicit ? t("flashnext.summary.set") : t("flashnext.summary.auto"),
      prompt: eff.promptTokens.toLocaleString("en-US"),
      pack: eff.pack ? t("flashnext.on") : t("flashnext.off"),
      head: eff.head ? t("flashnext.on") : t("flashnext.off"),
    }) +
      ", " +
      // A speed only for this Mac: with both numbers measured, never a reference machine's.
      (eff.tps
        ? fill(t("flashnext.summary.speed"), { tps: eff.tps.tps.toFixed(1) })
        : t("flashnext.summary.speedUnknown")),
  );
  const fits = $derived(eff.estimate.fits);
</script>

<div class="flex flex-col gap-3" data-flash-next-enabled={enabled}>
  {#if off}
    <div class="text-warn text-[11px] leading-snug" role="status">{t("flashnext.disabled")}</div>
  {/if}
  {#if shadowed.length > 0}
    <div class="flex items-center gap-3 flex-wrap text-warn text-[10.5px] leading-snug">
      <span>{fill(t("flashnext.shadowed"), { keys: shadowed.join(", ") })}</span>
      <button
        class="px-2 py-0.5 text-[10.5px] whitespace-nowrap"
        disabled={off}
        onclick={() => onClearOverrides(shadowed)}
      >{t("flashnext.shadowed.clear")}</button>
    </div>
  {/if}

  <div class="grid gap-x-8 gap-y-4 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
    <div
      class="flex flex-col gap-3 min-w-0"
      role="group"
      aria-label={t("flashnext.card.title")}
      aria-disabled={off || undefined}
    >
      <!-- What the server will run with, live -->
      <div class="flex items-start gap-2 px-2.5 py-2 bg-panel-2 border border-border rounded border-l-[3px] {fits ? 'border-l-accent' : 'border-l-err'}">
        <div class="flex-1 text-[11.5px] leading-snug">
          <b class="text-text">{summary}</b>
          {#if !fits}<span class="text-err"> · {t("flashnext.summary.noFit")}</span>{/if}
        </div>
        <button
          class="px-2 py-0.5 text-[10.5px] whitespace-nowrap"
          disabled={off || isDefault}
          title={t("flashnext.reset.hint")}
          onclick={onReset}
        >{t("flashnext.reset")}</button>
      </div>

      <!-- Expert cache size -->
      <div class="grid grid-cols-[165px_1fr] gap-x-3 gap-y-1">
        <div>
          <div class="font-medium">{t("flashnext.pool")}</div>
          <div class="mono text-[9.5px] text-text-dim">LUMEN_MOE_EXPERT_CACHE_GB</div>
        </div>
        <PoolPicker
          gb={poolGb}
          usableBytes={usable}
          chunk={eff.chunk}
          ctxTokens={eff.promptTokens}
          pack={eff.pack}
          {hw}
          disabled={off}
          onSelect={(v) => onChange({ expert_cache_gb: v })}
        />
      </div>

      <!-- Longest prompt -->
      <div class="grid grid-cols-[165px_1fr] gap-x-3 gap-y-1">
        <div>
          <div class="font-medium">{t("flashnext.prompt")}</div>
          <div class="mono text-[9.5px] text-text-dim">LUMEN_MAX_PROMPT_TOKENS</div>
        </div>
        <div class="flex flex-col gap-1">
          <PromptPicker
            tokens={eff.promptTokens}
            explicit={eff.promptExplicit}
            machineDefault={defaults.ctxTokens}
            {maxCtxLowers}
            usableBytes={usable}
            chunk={eff.chunk}
            disabled={off}
            onSelect={(n) => onChange({ max_prompt_tokens: n })}
          />
          <div class="dim text-[10px] leading-snug">{t("flashnext.context.hint")}</div>
        </div>
      </div>

      <!-- Prefill chunk -->
      <div class="grid grid-cols-[165px_1fr] gap-x-3 gap-y-1">
        <div>
          <div class="font-medium">{t("flashnext.chunk")}</div>
          <div class="mono text-[9.5px] text-text-dim">LUMEN_QWEN35_PREFILL_CHUNK</div>
        </div>
        <div class="flex flex-col gap-1">
          <select
            class="mono !w-48 disabled:opacity-60 disabled:cursor-not-allowed"
            value={eff.chunk}
            disabled={off}
            aria-label={t("flashnext.chunk")}
            onchange={(e) => {
              const v = Number((e.currentTarget as HTMLSelectElement).value);
              onChange({ prefill_chunk: v === defaults.chunk ? null : v });
            }}
          >
            {#each chunkOptions as c}
              <option value={c}>{c}{c === defaults.chunk ? ` (${t("flashnext.chunk.default")})` : ""}</option>
            {/each}
          </select>
          <div class="dim text-[10px] leading-snug">{t("flashnext.chunk.hint")}</div>
        </div>
      </div>

      <!-- Expert pack -->
      <div class="grid grid-cols-[165px_1fr] gap-x-3 gap-y-1">
        <div>
          <div class="font-medium">{t("flashnext.opt.pack")}</div>
          <div class="mono text-[9.5px] text-text-dim">LUMEN_MOE_PACK</div>
        </div>
        <div class="flex flex-col gap-1.5">
          <ChoiceChips
            options={packChips}
            value={packMode}
            label={t("flashnext.opt.pack")}
            disabled={off}
            onSelect={(v) => onChange({ pack: v as FlashNextSettings["pack"] })}
          />
          <div class="text-[10.5px] leading-snug">
            {#if !status || !status.installed}
              <span class="dim">{t("flashnext.pack.status.notInstalled")}</span>
            {:else if status.pack === "complete"}
              <span class="text-ok">
                {fill(t("flashnext.pack.status.complete"), { gb: gb(status.pack_bytes) })}
              </span>
              <span class="dim">· {packMode === "off" ? t("flashnext.pack.unused") : t("flashnext.pack.inUse")}</span>
            {:else if status.pack === "incomplete"}
              <span class="text-warn">
                {fill(t("flashnext.pack.status.incomplete"), { gb: gb(status.pack_bytes) })}
              </span>
            {:else}
              <span class="dim">
                {status.free_bytes == null
                  ? fill(t("flashnext.pack.status.absentNoFree"), { need: gb(status.need_bytes) })
                  : fill(t("flashnext.pack.status.absent"), { need: gb(status.need_bytes), free: gb(status.free_bytes) })}
              </span>
            {/if}
          </div>
          <PackBuildControl {status} {build} disabled={off} onBuild={onBuild} onCancel={onCancelBuild} />
        </div>
      </div>

      <!-- Draft head -->
      <div class="grid grid-cols-[165px_1fr] gap-x-3 gap-y-1">
        <div>
          <div class="font-medium">{t("flashnext.opt.mtp")}</div>
          <div class="mono text-[9.5px] text-text-dim">LUMEN_QWEN4_MTP</div>
        </div>
        <div class="flex flex-col gap-1.5">
          <ChoiceChips
            options={headChips}
            value={head}
            label={t("flashnext.opt.mtp")}
            disabled={off}
            onSelect={(v) => onChange({ draft_head: v as FlashNextSettings["draft_head"] })}
          />
          <div class="text-[10.5px] leading-snug">
            {#if !status || !status.installed}
              <span class="dim">{t("flashnext.pack.status.notInstalled")}</span>
            {:else if head === "off"}
              <!-- Off is the default: an installed head that is not used is not a problem. -->
              <span class="dim">
                {status.mtp_present ? t("flashnext.mtp.status.presentOff") : t("flashnext.mtp.status.missingOff")}
              </span>
            {:else if status.mtp_present}
              <span class="text-ok">{t("flashnext.mtp.status.present")}</span>
            {:else}
              <span class={head === "on" ? "text-err" : "text-warn"}>
                {head === "on" ? t("flashnext.mtp.status.missingOn") : t("flashnext.mtp.status.missing")}
              </span>
            {/if}
          </div>
          {#if status?.installed && !status.mtp_present && head !== "off"}
            <div class="mono text-[10px] px-2 py-1 bg-bg rounded text-accent select-text overflow-x-auto">
              $ python3 scripts/qwen4_mtp_fetch.py {status.dir ?? "<model dir>"}
            </div>
            <div class="dim text-[10px] leading-snug">{t("flashnext.mtp.fetch")}</div>
          {/if}
          <div class="dim text-[10px] leading-snug">{t("flashnext.mtp.note")}</div>
        </div>
      </div>

      <!-- Expert reads -->
      <div class="grid grid-cols-[165px_1fr] gap-x-3 gap-y-1">
        <div>
          <div class="font-medium">{t("flashnext.opt.io")}</div>
          <div class="mono text-[9.5px] text-text-dim">LUMEN_MOE_IO_MODE</div>
        </div>
        <div class="flex flex-col gap-1">
          <ChoiceChips
            options={ioChips}
            value={ioMode}
            label={t("flashnext.opt.io")}
            disabled={off}
            onSelect={(v) => onChange({ io_mode: v as FlashNextSettings["io_mode"] })}
          />
          <div class="dim text-[10.5px] leading-snug">{t(`flashnext.io.mode.${ioMode}`)}</div>
        </div>
      </div>

      <!-- Reader threads -->
      <div class="grid grid-cols-[165px_1fr] gap-x-3 gap-y-1">
        <div>
          <div class="font-medium">{t("flashnext.opt.lanes")}</div>
          <div class="mono text-[9.5px] text-text-dim">LUMEN_MOE_IO_LANES</div>
        </div>
        <div class="flex flex-col gap-1">
          <ChoiceChips
            options={laneChips}
            value={String(lanes)}
            label={t("flashnext.opt.lanes")}
            disabled={off}
            onSelect={(v) => onChange({ io_lanes: Number(v) === FLASH_NEXT_LANES_DEFAULT ? null : Number(v) })}
          />
          <div class="dim text-[10.5px] leading-snug">{t("flashnext.lanes.hint")}</div>
        </div>
      </div>
    </div>

    <!-- What the choices come to: the memory plan, the speed table, the disk -->
    <div class="min-w-0 lg:border-l lg:border-border lg:pl-6">
      <div class="text-[10px] text-text-dim uppercase tracking-wider mb-1.5">{t("flashnext.plan.title")}</div>
      <FlashNextCalculator
        {ramGb}
        {memoryLimitGb}
        {env}
        {contextMax}
        {status}
        {measure}
        enabled={enabled}
        onMeasure={onMeasure}
        onCancelMeasure={onCancelMeasure}
      />
    </div>
  </div>
</div>
