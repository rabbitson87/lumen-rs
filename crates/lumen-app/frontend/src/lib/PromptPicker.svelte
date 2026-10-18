<script lang="ts">
  // The longest prompt the server plans for: `8K | 16K | 32K | 64K | 128K | Custom`. Each chip
  // shows the pool Auto would leave with it and is disabled when that drops below the floor.
  // Writes `flash_next.max_prompt_tokens` (→ `LUMEN_MAX_PROMPT_TOKENS`); this Mac's default stores
  // null (the app sends the default itself). Raising it also raises the Max context card, if that
  // is below.
  import { t } from "./i18n.svelte";
  import ChoiceChips from "./ChoiceChips.svelte";
  import type { Chip } from "./chips";
  import { fill } from "./format";
  import {
    FLASH_NEXT,
    FLASH_NEXT_PROMPT_PRESETS,
    flashNextPromptChip,
    formatPromptShort,
    parsePromptInput,
  } from "./memory-estimate";

  interface Props {
    /** The prompt the plan holds KV for now (the override, else this Mac's default). */
    tokens: number;
    /** The user set the longest prompt (`LUMEN_MAX_PROMPT_TOKENS`). */
    explicit: boolean;
    /** What the app sends when it is not set. */
    machineDefault: number;
    /** Max context as the server reads it, when it is lower than the prompt cap. */
    maxCtxLowers?: number | null;
    usableBytes: number;
    chunk: number;
    /** `null` = this Mac's default (removes the override). */
    onSelect: (tokens: number | null) => void;
    /** Read-only: the chips and the custom box cannot be used. */
    disabled?: boolean;
  }
  let {
    tokens,
    explicit,
    machineDefault,
    maxCtxLowers = null,
    usableBytes,
    chunk,
    onSelect,
    disabled = false,
  }: Props = $props();

  const presets: readonly number[] = FLASH_NEXT_PROMPT_PRESETS;
  const fmtTok = (n: number) => n.toLocaleString("en-US");
  const gbText = (bytes: number) => (bytes / 1e9).toFixed(1);

  let customOpen = $state(false);
  let text = $state("");
  let invalid = $state(false);

  const selected = $derived(customOpen || !presets.includes(tokens) ? "custom" : String(tokens));

  const chips = $derived.by((): Chip[] => [
    ...presets.map((p) => {
      const c = flashNextPromptChip({ usableBytes, chunk, tokens: p });
      return {
        value: String(p),
        label: formatPromptShort(p) + (p === machineDefault ? " •" : ""),
        sub: c.fits ? `${gbText(c.poolBytes)} GB` : "—",
        disabled: !c.fits,
        title: c.fits
          ? p === machineDefault
            ? fill(t("flashnext.prompt.chip.default"), { n: fmtTok(p) })
            : fill(t("flashnext.prompt.chip.pool"), { n: fmtTok(p), gb: gbText(c.poolBytes) })
          : fill(t("flashnext.prompt.chip.noFit"), {
              n: fmtTok(p),
              floor: (FLASH_NEXT.poolFloorBytes / 1e9).toFixed(1),
            }),
      };
    }),
    { value: "custom", label: t("flashnext.pool.custom") },
  ]);

  const active = $derived(flashNextPromptChip({ usableBytes, chunk, tokens }));

  function choose(value: string) {
    invalid = false;
    if (value === "custom") {
      customOpen = true;
      text = formatPromptShort(tokens);
      return;
    }
    customOpen = false;
    const n = Number(value);
    onSelect(n === machineDefault ? null : n);
  }

  function commit() {
    const v = parsePromptInput(text);
    if (v == null) {
      invalid = text.trim() !== "";
      return;
    }
    invalid = false;
    const n = Math.min(FLASH_NEXT.maxContext, Math.max(1024, v));
    if (presets.includes(n)) customOpen = false;
    if (n !== tokens) onSelect(n === machineDefault ? null : n);
  }
</script>

<div class="flex flex-col gap-1">
  <ChoiceChips options={chips} value={selected} onSelect={choose} label={t("flashnext.prompt")} {disabled} />
  {#if selected === "custom"}
    <div class="flex items-center gap-2">
      <input
        class="mono !w-28 text-right"
        type="text"
        inputmode="numeric"
        placeholder="48k"
        aria-label={t("flashnext.pool.custom")}
        {disabled}
        bind:value={text}
        onkeydown={(e) => e.key === "Enter" && commit()}
        onblur={commit}
      />
      <span class="dim text-[10px]">{t("flashnext.prompt.custom.hint")}</span>
      {#if invalid}<span class="text-err text-[10px]">{t("flashnext.pool.custom.invalid")}</span>{/if}
    </div>
  {/if}
  <div class="text-[10.5px] leading-snug">
    <span class="dim">{t("flashnext.active")}</span>
    <b class="text-text">{fill(t("flashnext.prompt.active"), { n: fmtTok(tokens) })}</b>
    <span class="dim">
      · {explicit ? t("flashnext.prompt.set") : t("flashnext.prompt.default")}
      · {fill(t("flashnext.prompt.pool"), { gb: gbText(active.poolBytes) })}
    </span>
    {#if !active.fits}
      <span class="text-err">{t("flashnext.prompt.noFitActive")}</span>
    {/if}
    {#if maxCtxLowers != null}
      <span class="text-warn">{fill(t("flashnext.prompt.maxCtx"), { n: fmtTok(maxCtxLowers) })}</span>
    {/if}
  </div>
</div>
