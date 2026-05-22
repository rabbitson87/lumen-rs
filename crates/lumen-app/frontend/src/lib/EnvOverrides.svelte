<script lang="ts">
  import { t } from "./i18n.svelte";
  interface Props {
    value: Record<string, string>;
    typedKeys: Set<string>;
    onSave: (next: Record<string, string>) => void;
  }
  let { value, typedKeys, onSave }: Props = $props();

  // Local editable copy so the user can mutate freely before committing.
  // The $effect below re-syncs from `value` when the parent prop changes; the
  // initial snapshot here is intentional.
  // svelte-ignore state_referenced_locally
  let rows = $state<Array<{ key: string; value: string }>>(
    Object.entries(value).map(([key, v]) => ({ key, value: v })),
  );

  let dirty = $derived.by(() => {
    const cur = Object.fromEntries(
      rows.filter((r) => r.key.trim() !== "").map((r) => [r.key, r.value]),
    );
    const a = JSON.stringify(cur);
    const b = JSON.stringify(value);
    return a !== b;
  });

  function add() {
    rows = [...rows, { key: "", value: "" }];
  }
  function remove(i: number) {
    rows = rows.filter((_, j) => j !== i);
  }
  function commit() {
    const out: Record<string, string> = {};
    for (const r of rows) {
      const k = r.key.trim();
      if (k === "") continue;
      out[k] = r.value;
    }
    onSave(out);
  }
  function resetFromProp() {
    rows = Object.entries(value).map(([key, v]) => ({ key, value: v }));
  }

  // Re-sync when parent prop changes (e.g. config reload).
  $effect(() => {
    const incoming = JSON.stringify(value);
    const current = JSON.stringify(
      Object.fromEntries(
        rows.filter((r) => r.key.trim() !== "").map((r) => [r.key, r.value]),
      ),
    );
    if (incoming !== current && !dirty) {
      resetFromProp();
    }
  });

  // Inline `<code>` styling — formerly `.env-help code { ... }`. Lifting the
  // utility chain to a const so the markup stays readable when the same
  // pattern repeats across multiple help strings.
  const inlineCode = "font-mono bg-panel-2 px-1 rounded-[3px]";
</script>

<div class="px-4 pt-2 pb-3 text-xs">
  <div class="dim mb-2 text-[11px] leading-normal">
    {t("env.intro2")}
  </div>
  <div class="flex flex-col gap-1 mb-2">
    {#each rows as r, i}
      {@const shadowed = typedKeys.has(r.key.trim())}
      <div class="grid grid-cols-[280px_1fr_28px] gap-1.5 items-center">
        <input
          class={`mono uppercase ${shadowed ? "border-warn!" : ""}`}
          type="text"
          placeholder="LUMEN_..."
          bind:value={r.key}
        />
        <input class="mono" type="text" placeholder={t("env.placeholder.value")} bind:value={r.value} />
        <button
          class="min-w-0 px-2 py-0.5 bg-transparent text-text-dim hover:bg-panel-2 hover:text-err"
          onclick={() => remove(i)}
          title={t("env.row.remove")}
        >×</button>
      </div>
      {#if shadowed}
        <div class="text-[11px] text-warn pl-1">
          {t("env.row.shadowsPrefix")} <code class="mono">{r.key.trim()}</code>
        </div>
      {/if}
    {/each}
    {#if rows.length === 0}
      <div class="dim">{t("env.empty2")}</div>
    {/if}
  </div>
  <div class="flex items-center gap-2">
    <button onclick={add}>{t("env.add2")}</button>
    <button class="primary" disabled={!dirty} onclick={commit}>{t("env.save")}</button>
    {#if dirty}
      <button onclick={resetFromProp}>{t("env.revert")}</button>
    {/if}
    <span class="dim ml-auto text-[11px]">
      {t("env.applyHint")}
    </span>
  </div>
</div>
