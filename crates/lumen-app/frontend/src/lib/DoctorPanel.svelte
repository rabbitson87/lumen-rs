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

  // Status-driven border/bg tokens. Tailwind 4 alpha syntax (`border-ok/25`)
  // resolves the `--color-ok` theme var with a percentage opacity via
  // `color-mix`, giving a near-identical render to the prior `rgba(...)`.
  function rowBorder(s: string) {
    if (s === "pass") return "border-ok/25";
    if (s === "warn") return "border-warn/45";
    return "border-err/50";
  }
  function iconBg(s: string) {
    if (s === "pass") return "bg-ok";
    if (s === "warn") return "bg-warn";
    return "bg-err";
  }
</script>

<div class="px-4 pt-2 pb-3 text-xs">
  <div class="flex items-center gap-2.5 mb-2.5">
    <button onclick={recheck} disabled={running} class="primary">
      {running ? "Checking…" : "Re-check"}
    </button>
    {#if fixMessage}
      <span class="dim mono text-[11px]">{fixMessage}</span>
    {/if}
    <span class="dim ml-auto text-[11px]">
      Diagnostics run on app start and on demand. Each row links to a fix.
    </span>
  </div>

  {#if report}
    <div class="flex flex-col gap-1">
      {#each report.checks as c}
        <div class={`border ${rowBorder(c.status)} rounded-md overflow-hidden bg-panel-2`}>
          <button
            class="grid grid-cols-[22px_200px_1fr_16px] items-center gap-2.5 w-full text-left bg-transparent border-0 px-2.5 py-1.5 rounded-none hover:bg-white/[0.03]"
            onclick={() => toggle(c.id)}
            aria-expanded={expanded.has(c.id)}
          >
            <span class={`inline-flex items-center justify-center w-[18px] h-[18px] rounded-full text-[11px] font-bold text-bg ${iconBg(c.status)}`}>
              {icon(c.status)}
            </span>
            <span class="font-medium">{c.name}</span>
            <span class="mono text-text-dim text-xs overflow-hidden text-ellipsis whitespace-nowrap">{c.message}</span>
            <span class="text-text-dim text-[10px]">{expanded.has(c.id) ? "▾" : "▸"}</span>
          </button>
          {#if expanded.has(c.id)}
            <div class="px-4 pt-1.5 pb-2.5 pl-[42px] border-t border-border bg-panel">
              {#if c.detail}
                <div class="dim mb-1.5 text-[11px] leading-normal">{c.detail}</div>
              {/if}
              {#if c.fix_hint}
                <div class="mb-1.5 leading-normal">→ {c.fix_hint}</div>
              {/if}
              {#if c.fix_command}
                <div class="mono px-2 py-1 bg-bg rounded text-accent mb-2 select-text overflow-x-auto">$ {c.fix_command}</div>
              {/if}
              {#if c.fix_action}
                <div class="flex gap-1.5">
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
