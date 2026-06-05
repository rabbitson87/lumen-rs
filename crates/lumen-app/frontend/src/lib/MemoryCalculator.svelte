<script lang="ts">
  import { t } from "./i18n.svelte";
  import {
    geometryForModel,
    estimateAt,
    maxContextTokens,
    bytesToGB,
    type KvMode,
    type MemoryConfig,
  } from "./memory-estimate";

  interface Props {
    modelId: string | null;
    /** Current env overrides — used to seed KV mode / chunk / prefix defaults. */
    env: Record<string, string>;
  }
  let { modelId, env }: Props = $props();

  const geom = $derived(geometryForModel(modelId ?? ""));

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

  // One-shot seed from env when the model/env first resolves.
  let seeded = false;
  $effect(() => {
    if (seeded || !geom) return;
    seeded = true;
    kvMode = truthy(env["LUMEN_QWEN35_TQ_KV"]) ? "tq" : "bf16";
    const bb = Number(env["LUMEN_QWEN35_TQ_KV_BITS"]);
    if (Number.isFinite(bb) && bb >= 2 && bb <= 8) bits = bb;
    const c = Number(env["LUMEN_QWEN35_PREFILL_CHUNK"]);
    if (Number.isFinite(c) && c > 0) chunk = c;
    prefixCache = env["LUMEN_MLX_PREFIX_CACHE"]?.trim() !== "0";
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
    geom
      ? {
          geometry: geom,
          budgetBytes: budgetGB * 1024 * 1024 * 1024,
          kvMode: effMode,
          chunkTokens: chunk,
          prefixCache,
        }
      : null,
  );

  const ctxTokens = $derived(Math.round(ctxK * 1000));
  const breakdown = $derived(cfg ? estimateAt(cfg, ctxTokens) : null);
  const maxCtx = $derived(cfg ? maxContextTokens(cfg) : 0);

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
</script>

<div class="text-xs">
  {#if !geom}
    <div class="dim text-[11px] leading-normal">
      {t("memcalc.noGeometry").replace("{model}", modelId ?? "—")}
    </div>
  {:else}
    <div class="dim text-[10.5px] mb-2">
      {geom.label} · {geom.fullAttnLayers} full-attn · {geom.nKvHeads} KV-head · head_dim {geom.headDim}
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
        {#each BUDGET_PRESETS as p}
          <button
            class={`px-1.5 py-0.5 text-[10px] mono ${budgetGB === p ? "bg-panel-2 text-text" : "bg-transparent text-text-dim hover:bg-panel-2"}`}
            onclick={() => (budgetGB = p)}
          >{p}</button>
        {/each}
      </div>
    </div>
    <div class="dim text-[10px] mb-2 leading-snug">{t("memcalc.budget.hint")}</div>

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
  {/if}
</div>
