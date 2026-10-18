<script lang="ts">
  // A segmented control of preset chips: one is selected, a disabled one says why in its tooltip.
  // Shared by every Flash-Next picker, so they look and behave alike. With `disabled` the whole
  // group is read-only: every chip is disabled and dimmed, and the selected one still shows.
  import type { Chip } from "./chips";

  interface Props {
    options: Chip[];
    value: string;
    onSelect: (value: string) => void;
    /** Names the group for assistive tech. */
    label?: string;
    /** The whole group is read-only. */
    disabled?: boolean;
  }
  let { options, value, onSelect, label, disabled = false }: Props = $props();
</script>

<div
  class={`flex flex-wrap gap-1 ${disabled ? "opacity-55 cursor-not-allowed" : ""}`}
  role="radiogroup"
  aria-label={label}
  aria-disabled={disabled || undefined}
>
  {#each options as o (o.value)}
    <button
      type="button"
      role="radio"
      aria-checked={o.value === value}
      aria-disabled={disabled || o.disabled || undefined}
      disabled={disabled || o.disabled}
      title={o.title}
      class={`flex flex-col items-center leading-tight !px-2 !py-1 min-w-11 text-[10.5px] mono !rounded border transition-colors ${
        o.value === value
          ? "!border-accent !bg-accent/10 !text-text hover:!bg-accent/20"
          : o.disabled
            ? "!border-border/50 !bg-transparent !text-text-dim"
            : "!border-border !bg-transparent !text-text-dim hover:!bg-panel-2 hover:!text-text"
      }`}
      onclick={() => !disabled && !o.disabled && onSelect(o.value)}
    >
      <span class={o.warn ? "text-warn" : ""}>{o.label}</span>
      {#if o.sub}<span class="text-[9.5px] opacity-70">{o.sub}</span>{/if}
    </button>
  {/each}
</div>
