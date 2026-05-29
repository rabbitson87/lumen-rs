<script lang="ts">
  import { t } from "./i18n.svelte";
  import {
    ENV_SCHEMA,
    ENV_SECTIONS,
    envBoolOn,
    encodeBool,
    type EnvEntry,
    type EnvSection,
  } from "./env-schema";

  interface Props {
    value: Record<string, string>;
    typedKeys: Set<string>;
    onSave: (next: Record<string, string>) => void;
  }
  let { value, typedKeys, onSave }: Props = $props();

  // Edited copy: stores only the entries the user has actively set. Entries
  // that match `defaultValue` or are absent from `value` render as their
  // default placeholder but do not get persisted until the user touches them.
  // svelte-ignore state_referenced_locally
  let edits = $state<Record<string, string | null>>({});

  function reset() {
    edits = {};
  }

  // Re-sync edits from `value` when the parent prop changes (config reload).
  $effect(() => {
    // touch `value` so the effect retriggers on prop change
    const _ = JSON.stringify(value);
    edits = {};
  });

  // Compose final `Record<string, string>` to compare against `value` for
  // dirty detection and to send back on save.
  function compose(): Record<string, string> {
    const out: Record<string, string> = { ...value };
    for (const [k, v] of Object.entries(edits)) {
      if (v == null) {
        delete out[k];
      } else {
        out[k] = v;
      }
    }
    return out;
  }

  let dirty = $derived.by(() => {
    const cur = compose();
    return JSON.stringify(cur) !== JSON.stringify(value);
  });

  // Resolve the displayed value for an entry: edit > stored > "" (unset).
  function currentValue(entry: EnvEntry): string {
    if (entry.key in edits) {
      return edits[entry.key] ?? "";
    }
    return value[entry.key] ?? "";
  }

  function isSet(entry: EnvEntry): boolean {
    if (entry.key in edits) return edits[entry.key] != null;
    return entry.key in value;
  }

  function setValue(entry: EnvEntry, next: string) {
    edits = { ...edits, [entry.key]: next };
  }

  function clearValue(entry: EnvEntry) {
    edits = { ...edits, [entry.key]: null };
  }

  function commit() {
    onSave(compose());
  }

  function entriesBySection(section: EnvSection): EnvEntry[] {
    return ENV_SCHEMA.filter((e) => e.section === section);
  }

  // Coerce a number input value, clamped to the entry's range.
  function clampNumber(entry: EnvEntry, raw: string): string {
    if (raw.trim() === "") return "";
    const n = Number(raw);
    if (!Number.isFinite(n)) return raw;
    const min = entry.min ?? Number.NEGATIVE_INFINITY;
    const max = entry.max ?? Number.POSITIVE_INFINITY;
    const clamped = Math.max(min, Math.min(max, n));
    return String(clamped);
  }
</script>

<div class="px-4 pt-2 pb-3 text-xs">
  <div class="dim mb-3 text-[11px] leading-normal">
    {t("env.intro2")}
  </div>

  {#each ENV_SECTIONS as sec}
    {@const sectionEntries = entriesBySection(sec.id)}
    {#if sectionEntries.length > 0}
      <div class="mb-3">
        <div class="text-[11px] font-semibold text-text-dim uppercase tracking-wider mb-1.5">
          {t(sec.labelKey)}
        </div>
        <div class="flex flex-col gap-1.5">
          {#each sectionEntries as entry (entry.key)}
            {@const shadowed = typedKeys.has(entry.key)}
            {@const setNow = isSet(entry)}
            <div
              class={`grid grid-cols-[1fr_180px_28px] gap-2 items-center py-0.5 ${
                setNow ? "" : "opacity-60"
              }`}
            >
              <div class="flex flex-col min-w-0">
                <div class="flex items-center gap-1.5">
                  <span class="mono text-[10.5px] truncate">{entry.key}</span>
                  {#if shadowed}
                    <span class="text-warn text-[10px] shrink-0">⚠</span>
                  {/if}
                </div>
                <div class="text-[10.5px] text-text-dim leading-snug">
                  {t(entry.labelKey)}
                </div>
                {#if entry.helpKey}
                  <div class="text-[10px] text-text-dim/70 leading-snug">
                    {t(entry.helpKey)}
                  </div>
                {/if}
              </div>

              <div class="flex items-center justify-end">
                {#if entry.kind === "bool"}
                  <label class="inline-flex items-center gap-1.5 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={envBoolOn(currentValue(entry))}
                      onchange={(e) =>
                        setValue(entry, encodeBool((e.currentTarget as HTMLInputElement).checked))}
                    />
                    <span class="mono text-[11px]">
                      {envBoolOn(currentValue(entry)) ? "on" : "off"}
                    </span>
                  </label>
                {:else if entry.kind === "number"}
                  <input
                    class="mono text-right"
                    type="number"
                    min={entry.min}
                    max={entry.max}
                    step={entry.step ?? 1}
                    placeholder={entry.defaultValue}
                    value={currentValue(entry)}
                    oninput={(e) =>
                      setValue(entry, (e.currentTarget as HTMLInputElement).value)}
                    onblur={(e) =>
                      setValue(
                        entry,
                        clampNumber(entry, (e.currentTarget as HTMLInputElement).value),
                      )}
                  />
                {:else if entry.kind === "enum"}
                  <select
                    class="mono"
                    value={currentValue(entry) || entry.defaultValue}
                    onchange={(e) =>
                      setValue(entry, (e.currentTarget as HTMLSelectElement).value)}
                  >
                    {#each entry.options ?? [] as opt}
                      <option value={opt}>{opt}</option>
                    {/each}
                  </select>
                {:else}
                  <input
                    class="mono"
                    type="text"
                    placeholder={entry.defaultValue}
                    value={currentValue(entry)}
                    oninput={(e) =>
                      setValue(entry, (e.currentTarget as HTMLInputElement).value)}
                  />
                {/if}
              </div>

              <button
                class="min-w-0 px-1.5 py-0.5 bg-transparent text-text-dim hover:bg-panel-2 hover:text-err"
                disabled={!setNow}
                onclick={() => clearValue(entry)}
                title={t("env.row.remove")}
              >×</button>
            </div>
          {/each}
        </div>
      </div>
    {/if}
  {/each}

  <div class="flex items-center gap-2 pt-2 border-t border-text-dim/20">
    <button class="primary" disabled={!dirty} onclick={commit}>{t("env.save")}</button>
    {#if dirty}
      <button onclick={reset}>{t("env.revert")}</button>
    {/if}
    <span class="dim ml-auto text-[11px]">
      {t("env.applyHint")}
    </span>
  </div>
</div>
