<script lang="ts">
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
</script>

<div class="env-body">
  <div class="env-help dim">
    Raw env-var overrides passed to the <code>lumen-server</code> subprocess. Useful for
    one-off knobs not surfaced in the UI (e.g. <code>LUMEN_GEMMA4_FUSE_EXPERTS</code>,
    <code>LUMEN_AFFINE4_FORCE_CPU</code>). Keys that shadow a typed UI field are highlighted.
  </div>
  <div class="env-rows">
    {#each rows as r, i}
      {@const shadowed = typedKeys.has(r.key.trim())}
      <div class="env-row" class:shadowed>
        <input
          class="env-key mono"
          type="text"
          placeholder="LUMEN_..."
          bind:value={r.key}
        />
        <input class="env-val mono" type="text" placeholder="value" bind:value={r.value} />
        <button class="env-del" onclick={() => remove(i)} title="Remove">×</button>
      </div>
      {#if shadowed}
        <div class="env-warn">⚠ shadows the UI field for <code class="mono">{r.key.trim()}</code></div>
      {/if}
    {/each}
    {#if rows.length === 0}
      <div class="dim">No overrides set.</div>
    {/if}
  </div>
  <div class="env-actions">
    <button onclick={add}>+ Add</button>
    <button class="primary" disabled={!dirty} onclick={commit}>Save</button>
    {#if dirty}
      <button onclick={resetFromProp}>Revert</button>
    {/if}
    <span class="dim env-hint">
      Changes apply on next server start.
    </span>
  </div>
</div>

<style>
  .env-body {
    padding: 8px 16px 12px;
    font-size: 12px;
  }
  .env-help {
    margin-bottom: 8px;
    font-size: 11px;
    line-height: 1.5;
  }
  .env-help code {
    font-family: var(--mono);
    background: var(--panel-2);
    padding: 0 4px;
    border-radius: 3px;
  }
  .env-rows {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-bottom: 8px;
  }
  .env-row {
    display: grid;
    grid-template-columns: 280px 1fr 28px;
    gap: 6px;
    align-items: center;
  }
  .env-row.shadowed .env-key {
    border-color: var(--warn);
  }
  .env-key {
    text-transform: uppercase;
  }
  .env-del {
    padding: 2px 8px;
    min-width: 0;
    background: transparent;
    color: var(--text-dim);
  }
  .env-del:hover {
    background: var(--panel-2);
    color: var(--err);
  }
  .env-warn {
    font-size: 11px;
    color: var(--warn);
    padding-left: 4px;
  }
  .env-warn code {
    font-family: var(--mono);
  }
  .env-actions {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .env-hint {
    margin-left: auto;
    font-size: 11px;
  }
</style>
