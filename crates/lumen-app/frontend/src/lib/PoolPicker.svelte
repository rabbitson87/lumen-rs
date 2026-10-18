<script lang="ts">
  // The expert cache size: `Auto | 4 GB | … | 20 GB | Custom`. Each chip shows the decode speed
  // that pool buys on this Mac (once its memory and SSD are measured; no speed before) and is
  // disabled when it does not fit beside the current longest prompt.
  // Writes a plain number of GB (`8`), or null for Auto: the parent stores it as
  // `flash_next.expert_cache_gb` (→ `LUMEN_MOE_EXPERT_CACHE_GB`).
  import { t } from "./i18n.svelte";
  import ChoiceChips from "./ChoiceChips.svelte";
  import type { Chip } from "./chips";
  import { fill, fmtTps } from "./format";
  import {
    FLASH_NEXT,
    FLASH_NEXT_POOL_MIN_GB,
    FLASH_NEXT_POOL_PRESETS_GB,
    flashNextPoolChip,
    type FlashNextHardware,
    formatGbValue,
    parseGbInput,
  } from "./memory-estimate";

  interface Props {
    /** The pool the user set, in GB; `null` = auto. */
    gb: number | null;
    usableBytes: number;
    chunk: number;
    /** The longest prompt the plan holds KV for. */
    ctxTokens: number;
    /** Whether the expert pack is in use (the speed estimate follows its efficiency). */
    pack: boolean;
    /** This Mac's measured memory and SSD speeds; `null` until both are known. */
    hw: FlashNextHardware | null;
    onSelect: (gb: number | null) => void;
    /** Read-only: the chips and the custom box cannot be used. */
    disabled?: boolean;
  }
  let { gb, usableBytes, chunk, ctxTokens, pack, hw, onSelect, disabled = false }: Props = $props();

  const presets: readonly number[] = FLASH_NEXT_POOL_PRESETS_GB;
  const fmtTok = (n: number) => n.toLocaleString("en-US");
  const gbText = (bytes: number) => (bytes / 1e9).toFixed(1);

  let customOpen = $state(false);
  let text = $state("");
  let invalid = $state(false);

  const selected = $derived(
    gb == null ? "auto" : customOpen || !presets.includes(gb) ? "custom" : String(gb),
  );
  const auto = $derived(flashNextPoolChip({ usableBytes, chunk, ctxTokens, gb: null, pack, hw }));

  const speed = (c: ReturnType<typeof flashNextPoolChip>) =>
    c.tps ? `~${fmtTps(c.tps.tps)}` : undefined;

  const chips = $derived.by((): Chip[] => [
    {
      value: "auto",
      label: t("flashnext.pool.auto"),
      sub: `${gbText(auto.poolBytes)} GB${speed(auto) ? ` · ${speed(auto)}` : ""}`,
      title: t("flashnext.pool.chip.auto"),
      warn: !auto.fits,
    },
    ...presets.map((p) => {
      const c = flashNextPoolChip({ usableBytes, chunk, ctxTokens, gb: p, pack, hw });
      return {
        value: String(p),
        label: `${p} GB`,
        sub: speed(c),
        disabled: !c.fits,
        title: c.fits
          ? undefined
          : fill(t("flashnext.pool.chip.noFit"), {
              gb: p,
              ctx: fmtTok(ctxTokens),
              max: auto.fits ? auto.maxFitGb.toFixed(1) : "0",
            }),
      };
    }),
    { value: "custom", label: t("flashnext.pool.custom") },
  ]);

  const active = $derived(flashNextPoolChip({ usableBytes, chunk, ctxTokens, gb, pack, hw }));
  function choose(value: string) {
    invalid = false;
    if (value === "auto") {
      customOpen = false;
      onSelect(null);
    } else if (value === "custom") {
      customOpen = true;
      text = gb != null ? formatGbValue(gb) : "";
    } else {
      customOpen = false;
      onSelect(Number(value));
    }
  }

  function commit() {
    const v = parseGbInput(text);
    if (v == null) {
      invalid = text.trim() !== "";
      return;
    }
    invalid = false;
    const rounded = Math.round(v * 100) / 100;
    if (presets.includes(rounded)) customOpen = false;
    if (rounded !== gb) onSelect(rounded);
  }
</script>

<div class="flex flex-col gap-1">
  <ChoiceChips options={chips} value={selected} onSelect={choose} label={t("flashnext.pool")} {disabled} />
  {#if selected === "custom"}
    <div class="flex items-center gap-2">
      <input
        class="mono !w-28 text-right"
        type="text"
        inputmode="decimal"
        placeholder="12.5g"
        aria-label={t("flashnext.pool.custom")}
        {disabled}
        bind:value={text}
        onkeydown={(e) => e.key === "Enter" && commit()}
        onblur={commit}
      />
      <span class="dim text-[10px]">{t("flashnext.pool.custom.hint")}</span>
      {#if invalid}<span class="text-err text-[10px]">{t("flashnext.pool.custom.invalid")}</span>{/if}
    </div>
  {/if}
  <div class="text-[10.5px] leading-snug">
    {#if gb == null}
      <span class="dim">{t("flashnext.active")}</span>
      <b class="text-text">{fill(t("flashnext.pool.active.auto"), { pool: gbText(active.poolBytes) })}</b>
    {:else}
      <span class="dim">{t("flashnext.active")}</span>
      <b class="text-text">{fill(t("flashnext.pool.active.set"), { pool: formatGbValue(gb) })}</b>
    {/if}
    {#if active.tps}
      <span class="dim">· ≈ {fmtTps(active.tps.tps)} tok/s</span>
    {/if}
    {#if gb != null && gb < FLASH_NEXT_POOL_MIN_GB}
      <span class="text-warn">{fill(t("flashnext.pool.floor"), { floor: (FLASH_NEXT.poolFloorBytes / 1e9).toFixed(2) })}</span>
    {:else if gb != null && !active.fits}
      <span class="text-warn">
        {fill(t("flashnext.pool.lowered"), {
          ctx: fmtTok(ctxTokens),
          max: auto.fits ? auto.maxFitGb.toFixed(1) : "0",
        })}
      </span>
    {:else if !active.fits}
      <span class="text-err">{t("flashnext.pool.noFitAuto")}</span>
    {/if}
  </div>
  {#if hw}
    <div class="dim text-[9.5px] leading-snug">{t("flashnext.pool.legend")}</div>
  {/if}
</div>
