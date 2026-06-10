<script lang="ts">
  import { t } from "./i18n.svelte";
  import {
    geometryForModel,
    estimateAt,
    maxContextTokens,
    recommendForBudget,
    bytesToGB,
    envKeysForFamily,
    type KvMode,
    type MemoryConfig,
    type ModelFamily,
  } from "./memory-estimate";

  interface Props {
    modelId: string | null;
    /** Current env overrides — used to seed KV mode / chunk / prefix defaults. */
    env: Record<string, string>;
    /** Current structured `context.max` — seeds the context slider. */
    contextMax?: number;
    /** Current `context.prefill` (prompt-reject cap) — seeds "max prompt". */
    contextPrefill?: number;
    /** Current `context.default_max_tokens` — seeds "output max". */
    defaultMaxTokens?: number;
    /** Machine RAM (GB) — caps the budget and warns on over-allocation. */
    ramGb?: number;
    /** Current server `memory_limit_gb` — seeds the budget (clamped to RAM). */
    budgetDefault?: number;
    /** Live memory free for the model = RAM − other processes (from the poll). */
    availableGb?: number;
    /** Active model on-disk size (bytes) — sanity-anchors the weights estimate. */
    activeModelBytes?: number;
    /**
     * Push the previewed config into the live tuning state. Receives the
     * semantic knobs (incl. the resolved model `family` so the parent writes
     * the right `LUMEN_<FAMILY>_*` env namespace) plus the chosen context cap,
     * the prompt-reject cap, and the output-token budget.
     */
    onApply?: (sel: {
      family: ModelFamily;
      kvOn: boolean;
      bits: number;
      chunk: number;
      prefix: boolean;
      ctx: number;
      /** Prompt-size reject cap → context.prefill (LUMEN_MAX_PROMPT_TOKENS). */
      prefillLimit: number;
      /** Output budget → context.default_max_tokens. */
      outMaxTokens: number;
      /** When non-null, also write this as the server's MLX memory limit. */
      budgetGB: number | null;
    }) => void;
  }
  let {
    modelId,
    env,
    contextMax,
    contextPrefill,
    defaultMaxTokens,
    ramGb,
    budgetDefault,
    availableGb,
    activeModelBytes,
    onApply,
  }: Props = $props();

  const GB = 1024 * 1024 * 1024;

  const geom = $derived(geometryForModel(modelId ?? ""));

  let weightsGb = $state(0); // calibratable resident weights (0 ⇒ model anchor)

  // Resident weights ("active" at small prefill) used by every estimate. The
  // model anchor is a real measurement, but the operator can override it with
  // their own startup `[mlx-mem] ... active=` reading for an exact, machine- and
  // quant-specific figure (different nvfp4/mxfp4 variants differ ±2 GiB).
  const effGeom = $derived(
    geom
      ? {
          ...geom,
          weightsBytes: weightsGb > 0 ? weightsGb * GB : geom.weightsBytes,
        }
      : null,
  );

  // Seed controls from the live env so the calculator opens on the *current*
  // serving config, then let the user explore alternatives locally.
  const truthy = (v: string | undefined) =>
    v != null && ["1", "true", "on", "yes"].includes(v.trim().toLowerCase());

  let budgetGB = $state(31);
  let kvMode = $state<KvMode>("bf16");
  let bits = $state(8); // LUMEN_QWEN35_TQ_KV_BITS — quality (storage unaffected)
  let packed = $state(false); // preview: uint4 packing (not yet wired)
  let chunk = $state(2048);
  let prefixCache = $state(true);
  let ctxK = $state(50); // context in thousands of tokens (slider)
  let prefillLimit = $state(40960); // max prompt tokens (reject cap) → context.prefill
  let outMaxTokens = $state(2048); // output budget → context.default_max_tokens
  let applyBudget = $state(false); // also write budget → server memory_limit_gb

  // One-shot seed from env when the model/env first resolves. Reads the env
  // namespace for the *resolved* family so the calculator opens on whatever the
  // currently-loaded backend is actually serving (Qwen3.6 vs Gemma 4).
  let seeded = false;
  $effect(() => {
    if (seeded || !geom) return;
    seeded = true;
    const k = envKeysForFamily(geom.family);
    const kvRaw = env[k.kv]?.trim().toLowerCase();
    kvMode =
      k.kvKind === "bool"
        ? truthy(env[k.kv])
          ? "tq"
          : "bf16"
        : kvRaw && kvRaw !== "off"
          ? "tq"
          : "bf16";
    const bb = Number(env[k.kvBits]);
    if (Number.isFinite(bb) && bb >= 2 && bb <= 8) bits = bb;
    const c = Number(env[k.prefillChunk]);
    if (Number.isFinite(c) && c > 0) chunk = c;
    prefixCache = env["LUMEN_MLX_PREFIX_CACHE"]?.trim() !== "0";
    // Calibrate weights to the model's measured anchor (operator-overridable).
    if (weightsGb <= 0)
      weightsGb = Math.round((geom.weightsBytes / GB) * 10) / 10;
    // Budget: prefer the LIVE memory actually free for the model (RAM − other
    // processes); the machine rarely has all its RAM spare. Fall back to the
    // configured MLX limit, then ~85% of RAM. Never seed above physical RAM (a
    // limit over RAM just thrashes — the user's 79 GB limit on a 36 GB Mac).
    const ram = ramGb && ramGb > 0 ? ramGb : undefined;
    let b =
      availableGb && availableGb > 0
        ? availableGb
        : budgetDefault && budgetDefault > 0
          ? budgetDefault
          : ram
            ? Math.round(ram * 0.85)
            : 31;
    if (ram && b > ram) b = Math.round(ram * 0.85);
    budgetGB = b;
    // Seed the prompt-reject cap + output budget from the structured context.
    if (contextPrefill && contextPrefill > 0) prefillLimit = contextPrefill;
    if (defaultMaxTokens && defaultMaxTokens > 0) outMaxTokens = defaultMaxTokens;
    // Seed the slider at the worst-case working set (prompt cap + output) so
    // the breakdown opens on the memory the server must actually hold, falling
    // back to context.max. Clamped to the model's hard context ceiling.
    const ws = prefillLimit + outMaxTokens;
    const seedCtx = ws > 0 ? ws : (contextMax ?? 50_000);
    ctxK = Math.min(
      Math.round(geom.maxContext / 1000),
      Math.max(1, Math.round(seedCtx / 1000)),
    );
  });

  // Effective storage mode: TurboQuant stores uint8 codes UNPACKED, so memory
  // is ~2× for ANY bits (bits = quality only). The optional uint4 packing
  // preview (bits=4) is the only thing that changes storage (~4×).
  const effMode = $derived<KvMode>(
    kvMode === "tq" && packed && bits === 4 ? "tq_packed4" : kvMode,
  );

  const BUDGET_PRESETS = [16, 24, 31, 36, 48, 64, 96, 128];
  const CHUNK_PRESETS = [2048, 1024, 512, 256];

  const cfg = $derived<MemoryConfig | null>(
    effGeom
      ? {
          geometry: effGeom,
          budgetBytes: budgetGB * GB,
          kvMode: effMode,
          chunkTokens: chunk,
          prefixCache,
        }
      : null,
  );

  const ctxTokens = $derived(Math.round(ctxK * 1000));
  const breakdown = $derived(cfg ? estimateAt(cfg, ctxTokens) : null);
  const maxCtx = $derived(cfg ? maxContextTokens(cfg) : 0);

  // Worst-case working set the server must hold = prompt-reject cap + output
  // budget. This is the number that determines whether a big-prompt client
  // (e.g. omp) fits; surfaced as its own peak so raising the prompt cap shows
  // its memory cost immediately.
  const wsTokens = $derived(prefillLimit + outMaxTokens);
  const wsBreakdown = $derived(cfg ? estimateAt(cfg, wsTokens) : null);
  const wsOverContext = $derived(wsTokens > (geom?.maxContext ?? Infinity));

  const fmt = (b: number) => bytesToGB(b).toFixed(2);
  const fmtTok = (n: number) => n.toLocaleString("en-US");

  // Comparison table: max context per (KV mode × chunk) at the chosen budget.
  const TABLE_MODES: { key: KvMode; label: string }[] = [
    { key: "bf16", label: "bf16" },
    { key: "tq", label: "TQ ~2×" },
    { key: "tq_packed4", label: "TQ4*" },
  ];
  const tableRows = $derived.by(() => {
    if (!cfg) return [];
    return TABLE_MODES.map((m) => ({
      label: m.label,
      cells: CHUNK_PRESETS.map((ch) =>
        maxContextTokens({ ...cfg, kvMode: m.key, chunkTokens: ch }),
      ),
    }));
  });

  const barColor = $derived(
    !breakdown
      ? "bg-text-dim"
      : breakdown.utilization > 1
        ? "bg-err"
        : breakdown.utilization > 0.9
          ? "bg-warn"
          : "bg-ok",
  );

  // "Optimize for this machine": fill every knob with a budget-fitted config
  // derived from the chosen RAM budget. The user reviews the result, then
  // Apply → Stop → Start. Maximizes the working set (prompt cap + output) under
  // budget, conceding KV quality / prefill speed only as the budget tightens.
  let recoNote = $state<"" | "ok" | "aggressive" | "tooSmall">("");
  function optimize() {
    if (!effGeom) return;
    const r = recommendForBudget(effGeom, budgetGB * GB);
    kvMode = r.kvMode;
    bits = r.bits;
    packed = false;
    chunk = r.chunk;
    prefixCache = r.prefix;
    prefillLimit = r.prefillLimit;
    outMaxTokens = r.outMaxTokens;
    ctxK = Math.max(1, Math.round(r.ctxMax / 1000));
    recoNote = r.note;
  }

  // Push the previewed knobs into the live tuning state. The parent maps them
  // onto the structured quant config (QUANT card) + the family's env overrides +
  // the structured `context.max`, and a Stop → Start applies them. `packed` is a
  // memory *preview* (uint4 not yet wired), so it is intentionally not exported.
  let applied = $state(false);
  function apply() {
    if (!geom) return;
    // context.max must cover both the explored slider context AND the
    // worst-case working set (prompt cap + output); clamp to the model ceiling.
    const ctx = Math.min(
      geom.maxContext,
      Math.max(ctxTokens, prefillLimit + outMaxTokens),
    );
    onApply?.({
      family: geom.family,
      kvOn: kvMode === "tq",
      bits,
      chunk,
      prefix: prefixCache,
      ctx,
      prefillLimit,
      outMaxTokens,
      budgetGB: applyBudget ? budgetGB : null,
    });
    applied = true;
    setTimeout(() => (applied = false), 1800);
  }
</script>

<div class="text-xs">
  {#if !geom}
    <div class="dim text-[11px] leading-normal">
      {t("memcalc.noGeometry").replace("{model}", modelId ?? "—")}
    </div>
  {:else}
    <div class="dim text-[10.5px] mb-1.5">
      {geom.label} · {geom.fullAttnLayers} full-attn · {geom.nKvHeads} KV-head · head_dim {geom.headDim}
    </div>

    <!-- Weights calibration: grounds the whole estimate in the operator's own
         startup measurement instead of the catalog anchor. -->
    <div class="flex items-center gap-1.5 mb-2" title={t("memcalc.active.hint")}>
      <span class="dim text-[10px]">{t("memcalc.active")}</span>
      <input
        class="mono text-right w-14 text-[11px]"
        type="number"
        min="1"
        max="512"
        step="0.1"
        bind:value={weightsGb}
      />
      <span class="dim text-[10px]">GB</span>
      {#if activeModelBytes}
        <span class="dim text-[10px]"
          >· disk {(activeModelBytes / GB).toFixed(1)}</span
        >
      {/if}
    </div>

    <!-- Budget -->
    <div class="flex items-center gap-2 mb-1.5">
      <span class="dim w-20 shrink-0">{t("memcalc.budget")}</span>
      <input
        class="mono text-right w-16"
        type="number"
        min="4"
        max="512"
        step="1"
        bind:value={budgetGB}
      />
      <span class="dim">GB</span>
      <div class="flex gap-1 flex-wrap ml-1">
        {#if availableGb && availableGb > 0}
          <button
            class={`px-1.5 py-0.5 text-[10px] mono border ${budgetGB === availableGb ? "bg-panel-2 text-text border-border" : "bg-transparent text-ok border-ok/40 hover:bg-panel-2"}`}
            onclick={() => {
              budgetGB = availableGb ?? budgetGB;
              optimize();
            }}
            title={t("memcalc.budget.availableHint")}
          >{t("memcalc.budget.available").replace("{n}", String(availableGb))}</button>
        {/if}
        {#each BUDGET_PRESETS as p}
          <button
            class={`px-1.5 py-0.5 text-[10px] mono ${budgetGB === p ? "bg-panel-2 text-text" : "bg-transparent text-text-dim hover:bg-panel-2"}`}
            onclick={() => {
              budgetGB = p;
              optimize();
            }}
          >{p}</button>
        {/each}
      </div>
      <button
        class="px-2 py-0.5 text-[10px] mono bg-accent/10 text-accent border border-accent/40 rounded hover:bg-accent/20 ml-auto whitespace-nowrap"
        onclick={optimize}
        title={t("memcalc.optimize.hint")}
      >{t("memcalc.optimize")}</button>
    </div>
    <div class="dim text-[10px] mb-2 leading-snug">{t("memcalc.budget.hint")}</div>
    {#if ramGb && budgetGB > ramGb}
      <div class="text-err text-[10px] mb-2 leading-snug">
        {t("memcalc.budget.overRam")
          .replace("{budget}", String(budgetGB))
          .replace("{ram}", String(ramGb))}
      </div>
    {/if}
    {#if recoNote}
      <div
        class="text-[10px] mb-2 leading-snug"
        class:text-ok={recoNote === "ok"}
        class:text-warn={recoNote === "aggressive"}
        class:text-err={recoNote === "tooSmall"}
      >
        {t(`memcalc.reco.${recoNote}`).replace("{budget}", String(budgetGB))}
      </div>
    {/if}

    <!-- KV mode + bits + chunk + prefix -->
    <div class="flex items-center gap-x-3 gap-y-1.5 mb-1 flex-wrap">
      <div class="flex items-center gap-1.5">
        <span class="dim">{t("memcalc.kv")}</span>
        <select class="mono" bind:value={kvMode}>
          <option value="bf16">bf16</option>
          <option value="tq">TurboQuant</option>
        </select>
      </div>
      {#if kvMode === "tq"}
        <div class="flex items-center gap-1.5">
          <span class="dim">{t("memcalc.bits")}</span>
          <select class="mono" bind:value={bits}>
            <option value={8}>8</option>
            <option value={6}>6</option>
            <option value={4}>4</option>
          </select>
        </div>
        {#if bits === 4}
          <label class="inline-flex items-center gap-1.5 cursor-pointer">
            <input type="checkbox" bind:checked={packed} />
            <span class="mono text-[11px]">{t("memcalc.packed")}</span>
          </label>
        {/if}
      {/if}
      <div class="flex items-center gap-1.5">
        <span class="dim">{t("memcalc.chunk")}</span>
        <select class="mono" bind:value={chunk}>
          {#each CHUNK_PRESETS as c}
            <option value={c}>{c}</option>
          {/each}
        </select>
      </div>
      <label class="inline-flex items-center gap-1.5 cursor-pointer">
        <input type="checkbox" bind:checked={prefixCache} />
        <span class="mono text-[11px]">{t("memcalc.prefix")}</span>
      </label>
    </div>
    {#if kvMode === "tq"}
      <div class="dim text-[10px] mb-2 leading-snug">{t("memcalc.bits.hint")}</div>
    {:else}
      <div class="mb-1"></div>
    {/if}

    <!-- Prompt-reject cap + output budget (the omp "prefill limit" lives here) -->
    <div class="flex items-center gap-x-3 gap-y-1.5 mb-1 flex-wrap">
      <div class="flex items-center gap-1.5">
        <span class="dim">{t("memcalc.maxPrompt")}</span>
        <input
          class="mono text-right w-20"
          type="number"
          min="256"
          max={geom.maxContext}
          step="1024"
          bind:value={prefillLimit}
        />
      </div>
      <div class="flex items-center gap-1.5">
        <span class="dim">{t("memcalc.outMax")}</span>
        <input
          class="mono text-right w-16"
          type="number"
          min="0"
          max={geom.maxContext}
          step="256"
          bind:value={outMaxTokens}
        />
      </div>
    </div>
    {#if wsBreakdown}
      <div
        class="dim text-[10px] mb-2 leading-snug"
        class:text-err={!wsBreakdown.fits || wsOverContext}
      >
        {t("memcalc.workingSet")
          .replace("{prompt}", fmtTok(prefillLimit))
          .replace("{out}", fmtTok(outMaxTokens))
          .replace("{total}", fmtTok(wsTokens))
          .replace("{peak}", fmt(wsBreakdown.peakBytes))}{wsOverContext
          ? ` ⚠ > ${fmtTok(geom.maxContext)}`
          : wsBreakdown.fits
            ? ""
            : " ⚠"}
      </div>
    {/if}

    <!-- Context slider -->
    <div class="flex items-center gap-2 mb-1">
      <span class="dim w-20 shrink-0">{t("memcalc.context")}</span>
      <input
        type="range"
        min="1"
        max={Math.round(geom.maxContext / 1000)}
        step="1"
        bind:value={ctxK}
        class="flex-1"
      />
      <span class="mono text-right w-20">{fmtTok(ctxTokens)}</span>
    </div>

    {#if breakdown}
      <!-- Utilization bar -->
      <div class="h-2.5 w-full bg-panel-2 rounded-sm overflow-hidden mt-2 mb-1">
        <div
          class={`h-full ${barColor}`}
          style={`width: ${Math.min(100, breakdown.utilization * 100).toFixed(1)}%`}
        ></div>
      </div>
      <div class="flex items-baseline justify-between gap-2 mb-1.5">
        <span class="dim text-[10px] shrink-0">{t("memcalc.peak")}</span>
        <span class="mono text-[11px] whitespace-nowrap {breakdown.fits ? '' : 'text-err'}">
          {fmt(breakdown.peakBytes)} / {budgetGB.toFixed(0)} GB · {(breakdown.utilization * 100).toFixed(0)}%{breakdown.fits ? "" : " ⚠"}
        </span>
      </div>
      <!-- Component breakdown -->
      <div class="grid grid-cols-[1fr_auto] gap-x-3 gap-y-0.5 text-[10.5px] mono mb-2">
        <span class="dim">weights</span><span class="text-right">{fmt(breakdown.weightsBytes)} GB</span>
        <span class="dim">KV @ ctx</span><span class="text-right">{fmt(breakdown.kvBytes)} GB</span>
        {#if prefixCache}
          <span class="dim">snapshot</span><span class="text-right">{fmt(breakdown.snapshotBytes)} GB</span>
        {/if}
        <span class="dim">attn peak</span><span class="text-right">{fmt(breakdown.attnScoreBytes)} GB</span>
        <span class="dim">overhead</span><span class="text-right">{fmt(breakdown.overheadBytes)} GB</span>
      </div>

      <div class="flex items-baseline justify-between gap-2 mb-2 pt-1.5 border-t border-text-dim/15">
        <span class="text-[11px] shrink-0">{t("memcalc.maxAtConfig")}</span>
        <span class="mono font-semibold text-ok whitespace-nowrap">{fmtTok(maxCtx)} tok</span>
      </div>
    {/if}

    <!-- Comparison table -->
    <div class="text-[10px] text-text-dim uppercase tracking-wider mb-1">
      {t("memcalc.table.title").replace("{budget}", String(budgetGB))}
    </div>
    <div class="overflow-x-auto">
      <table class="text-[10.5px] mono border-collapse">
        <thead>
          <tr class="text-text-dim">
            <th class="text-left font-normal pr-3 pb-0.5">KV \ chunk</th>
            {#each CHUNK_PRESETS as c}
              <th class="text-right font-normal px-2 pb-0.5">{c}</th>
            {/each}
          </tr>
        </thead>
        <tbody>
          {#each tableRows as row}
            <tr>
              <td class="text-left pr-3 text-text-dim">{row.label}</td>
              {#each row.cells as cell}
                <td class="text-right px-2">{fmtTok(cell)}</td>
              {/each}
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
    <div class="dim text-[10px] mt-1 leading-snug">{t("memcalc.table.note")}</div>

    {#if onApply}
      <div class="flex items-center gap-2 mt-3 pt-2.5 border-t border-border">
        <button
          class="px-2.5 py-1 text-[11px] mono bg-panel-2 text-text border border-border rounded hover:border-accent disabled:opacity-50"
          onclick={apply}
        >{t("memcalc.apply")}</button>
        {#if applied}
          <span class="text-[10.5px] text-ok">{t("memcalc.applied")}</span>
        {:else}
          <span class="dim text-[10px] leading-snug">{t("memcalc.apply.hint")}</span>
        {/if}
      </div>
      <label
        class="inline-flex items-center gap-1.5 mt-1.5 cursor-pointer"
        title={t("memcalc.applyBudget.hint")}
      >
        <input type="checkbox" bind:checked={applyBudget} />
        <span class="dim text-[10px]">{t("memcalc.applyBudget")} ({budgetGB} GB)</span>
      </label>
    {/if}
  {/if}
</div>
