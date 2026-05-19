<script lang="ts">
  import { api, type DoctorReport, type CheckResult } from "./api";

  interface Props {
    report: DoctorReport | null;
    onReport: (r: DoctorReport) => void;
  }
  let { report, onReport }: Props = $props();

  let running = $state(false);
  let busyAction = $state<string | null>(null);
  let expanded = $state<Set<string>>(new Set());
  let fixMessage = $state<string | null>(null);

  async function recheck() {
    running = true;
    fixMessage = null;
    try {
      onReport(await api.doctorRun());
    } finally {
      running = false;
    }
  }

  async function applyFix(c: CheckResult) {
    if (!c.fix_action) return;
    busyAction = c.id;
    fixMessage = null;
    try {
      const msg = await api.doctorFix(c.fix_action);
      fixMessage = `[${c.id}] ${msg}`;
      onReport(await api.doctorRun());
    } catch (e) {
      fixMessage = `[${c.id}] failed: ${e}`;
    } finally {
      busyAction = null;
    }
  }

  function toggle(id: string) {
    const next = new Set(expanded);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    expanded = next;
  }

  function icon(status: string) {
    if (status === "pass") return "✓";
    if (status === "warn") return "!";
    return "✗";
  }
</script>

<div class="doctor-body">
  <div class="doctor-header">
    <button onclick={recheck} disabled={running} class="primary">
      {running ? "Checking…" : "Re-check"}
    </button>
    {#if fixMessage}
      <span class="dim doctor-msg mono">{fixMessage}</span>
    {/if}
    <span class="dim doctor-hint">
      Diagnostics run on app start and on demand. Each row links to a fix.
    </span>
  </div>

  {#if report}
    <div class="doctor-rows">
      {#each report.checks as c}
        <div class="doctor-row {c.status}" class:expanded={expanded.has(c.id)}>
          <button
            class="row-head"
            onclick={() => toggle(c.id)}
            aria-expanded={expanded.has(c.id)}
          >
            <span class="row-icon {c.status}">{icon(c.status)}</span>
            <span class="row-name">{c.name}</span>
            <span class="row-msg mono">{c.message}</span>
            <span class="row-chev">{expanded.has(c.id) ? "▾" : "▸"}</span>
          </button>
          {#if expanded.has(c.id)}
            <div class="row-body">
              {#if c.detail}
                <div class="row-detail dim">{c.detail}</div>
              {/if}
              {#if c.fix_hint}
                <div class="row-hint">→ {c.fix_hint}</div>
              {/if}
              {#if c.fix_command}
                <div class="row-cmd mono">$ {c.fix_command}</div>
              {/if}
              {#if c.fix_action}
                <div class="row-actions">
                  <button
                    onclick={() => applyFix(c)}
                    disabled={busyAction !== null}
                    class="primary"
                  >
                    {busyAction === c.id ? "Working…" : "Fix it"}
                  </button>
                </div>
              {/if}
            </div>
          {/if}
        </div>
      {/each}
    </div>
  {:else}
    <div class="dim">Click <span class="mono">Re-check</span> to run diagnostics.</div>
  {/if}
</div>

<style>
  .doctor-body {
    padding: 8px 16px 12px;
    font-size: 12px;
  }
  .doctor-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
  }
  .doctor-msg {
    font-size: 11px;
  }
  .doctor-hint {
    margin-left: auto;
    font-size: 11px;
  }

  .doctor-rows {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .doctor-row {
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    background: var(--panel-2);
  }
  .doctor-row.pass {
    border-color: rgba(91, 214, 163, 0.25);
  }
  .doctor-row.warn {
    border-color: rgba(240, 182, 92, 0.45);
  }
  .doctor-row.fail {
    border-color: rgba(255, 122, 122, 0.5);
  }

  .row-head {
    display: grid;
    grid-template-columns: 22px 200px 1fr 16px;
    align-items: center;
    gap: 10px;
    width: 100%;
    text-align: left;
    border: none;
    background: transparent;
    padding: 6px 10px;
    border-radius: 0;
  }
  .row-head:hover {
    background: rgba(255, 255, 255, 0.03);
  }
  .row-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    font-size: 11px;
    font-weight: 700;
    color: var(--bg);
  }
  .row-icon.pass {
    background: var(--ok);
  }
  .row-icon.warn {
    background: var(--warn);
  }
  .row-icon.fail {
    background: var(--err);
  }
  .row-name {
    font-weight: 500;
  }
  .row-msg {
    color: var(--text-dim);
    font-size: 12px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .row-chev {
    color: var(--text-dim);
    font-size: 10px;
  }

  .row-body {
    padding: 6px 16px 10px 42px;
    border-top: 1px solid var(--border);
    background: var(--panel);
  }
  .row-detail {
    margin-bottom: 6px;
    font-size: 11px;
    line-height: 1.5;
  }
  .row-hint {
    margin-bottom: 6px;
    line-height: 1.5;
  }
  .row-cmd {
    padding: 4px 8px;
    background: var(--bg);
    border-radius: 4px;
    color: var(--accent);
    margin-bottom: 8px;
    user-select: text;
    overflow-x: auto;
  }
  .row-actions {
    display: flex;
    gap: 6px;
  }
</style>
