<script lang="ts">
  import { api, type DoctorReport, type CheckResult } from "./api";
  import { t } from "./i18n.svelte";

  /**
   * Translate a doctor check's user-facing strings. The backend ships
   * stable English text so it stays grep-friendly in logs / bug reports;
   * we map it to a locale key here on the frontend boundary.
   *
   * Strategy: dispatch on `check.id` + a small classification of the
   * English `message` (status / contained substring). Interpolated values
   * (paths, sizes, version numbers, port numbers) are preserved verbatim.
   */
  function translatedName(c: CheckResult): string {
    return t(`doctor.check.${c.id}.name`);
  }

  function translatedMessage(c: CheckResult): string {
    const m = c.message;
    switch (c.id) {
      case "os_version": {
        // raw shapes: "macOS 14.0" / "macOS 11.5 — supported, but 14+ recommended"
        // / "macOS 10.15 — unsupported" / "macOS version unknown"
        if (m === "macOS version unknown") return t("doctor.msg.os_version.unknown");
        const match = m.match(/^macOS (\S+)(.*)$/);
        if (!match) return m;
        const ver = match[1];
        const tail = match[2];
        let suffix = "";
        if (tail.includes("supported, but 14")) suffix = " " + t("doctor.msg.os_version.warn.suffix");
        else if (tail.includes("unsupported")) suffix = " " + t("doctor.msg.os_version.fail.suffix");
        return `${t("doctor.msg.os_version.ok")} ${ver}${suffix}`;
      }
      case "architecture":
        if (m === "Apple Silicon (arm64)") return t("doctor.msg.architecture.silicon");
        if (m === "Intel Mac (x86_64)") return t("doctor.msg.architecture.intel");
        if (m.startsWith("unsupported architecture:"))
          return m.replace("unsupported architecture:", t("doctor.msg.architecture.other"));
        return m;
      case "ram": {
        const match = m.match(/^(\d+) GB$/);
        if (!match) return m;
        return `${match[1]} ${t("doctor.msg.ram.gb")}`;
      }
      case "disk_free": {
        // "120 GB free at /Users/foo/models"
        const match = m.match(/^(\d+) GB free at (.+)$/);
        if (!match) return m;
        return t("doctor.msg.disk_free.template")
          .replace("{gb}", match[1])
          .replace("{path}", match[2]);
      }
      case "models_dir":
        if (m.startsWith("writable:")) return m.replace("writable:", t("doctor.msg.models_dir.writable"));
        if (m.startsWith("not writable:")) return m.replace("not writable:", t("doctor.msg.models_dir.notWritable"));
        if (m.startsWith("missing:")) return m.replace("missing:", t("doctor.msg.models_dir.missing"));
        return m;
      case "server_binary":
        if (m === "not found") return t("doctor.msg.server_binary.notFound");
        if (m.startsWith("found:")) return m.replace("found:", t("doctor.msg.server_binary.found"));
        if (m.startsWith("not executable:")) return m.replace("not executable:", t("doctor.msg.server_binary.notExecutable"));
        return m;
      case "port_free": {
        const avail = m.match(/^port (\d+) available$/);
        if (avail) return t("doctor.msg.port_free.available").replace("{port}", avail[1]);
        const inUse = m.match(/^port (\d+) in use$/);
        if (inUse) return t("doctor.msg.port_free.inUse").replace("{port}", inUse[1]);
        return m;
      }
      case "active_model": {
        if (m === "no model selected") return t("doctor.msg.active_model.none");
        const ready = m.match(/^(.+) ready$/);
        if (ready) return t("doctor.msg.active_model.ready").replace("{id}", ready[1]);
        const incomplete = m.match(/^(.+) on disk but incomplete$/);
        if (incomplete) return t("doctor.msg.active_model.incomplete").replace("{id}", incomplete[1]);
        const missing = m.match(/^(.+) not on disk$/);
        if (missing) return t("doctor.msg.active_model.missing").replace("{id}", missing[1]);
        return m;
      }
      case "huggingface": {
        if (m === "unreachable") return t("doctor.msg.huggingface.unreachable");
        if (m === "http client init failed") return t("doctor.msg.huggingface.clientInitFailed");
        const reach = m.match(/^reachable \((\d+)\)$/);
        if (reach) return t("doctor.msg.huggingface.reachable").replace("{code}", reach[1]);
        const unexpected = m.match(/^unexpected status (\d+)$/);
        if (unexpected) return t("doctor.msg.huggingface.unexpected").replace("{code}", unexpected[1]);
        return m;
      }
    }
    return m;
  }

  /**
   * Map the backend's English `fix_hint` (a stable, immutable string in
   * `doctor.rs`) onto a translation key. The map is keyed on identifying
   * substring — exact-match would be fragile against future whitespace
   * tweaks in the Rust source.
   */
  function translatedFixHint(c: CheckResult): string | null {
    if (!c.fix_hint) return null;
    const h = c.fix_hint;
    if (h.includes("MPS performance improvements")) return t("doctor.hint.os_version.warn");
    if (h.includes("requires macOS 11")) return t("doctor.hint.os_version.fail");
    if (h.includes("Could not run `sw_vers`")) return t("doctor.hint.os_version.unknown");
    if (h.includes("5-20× faster")) return t("doctor.hint.architecture.intel");
    if (h.includes("targets macOS Apple Silicon")) return t("doctor.hint.architecture.unsupported");
    if (h.includes("1.5-7B models")) return t("doctor.hint.ram.warnLow");
    if (h.includes("8-16 GB is tight")) return t("doctor.hint.ram.tight");
    if (h.includes("Less than 8 GB RAM")) return t("doctor.hint.ram.fail");
    if (h.includes("Most modern weight sets")) return t("doctor.hint.disk_free.warn");
    if (h.includes("Less than 20 GB free")) return t("doctor.hint.disk_free.fail");
    if (h.includes("Fix permissions on the models")) return t("doctor.hint.models_dir.notWritable");
    if (h.includes("Click Fix to create")) return t("doctor.hint.models_dir.missing");
    if (h.includes("Set the executable bit")) return t("doctor.hint.server_binary.notExecutable");
    if (h.includes("Build the inference server")) return t("doctor.hint.server_binary.notFound");
    if (h.includes("Another process is bound")) return t("doctor.hint.port_free.inUse");
    if (h.includes("Pick a model in the ACTIVE MODEL")) return t("doctor.hint.active_model.none");
    if (h.includes("Re-download the model from the MODELS")) return t("doctor.hint.active_model.incomplete");
    if (h.includes("Download the model from the MODELS")) return t("doctor.hint.active_model.missing");
    if (h.includes("not with 2xx/3xx")) return t("doctor.hint.huggingface.unexpected");
    if (h.includes("Check your internet connection")) return t("doctor.hint.huggingface.unreachable");
    return h;
  }

  function translatedDetail(c: CheckResult): string | null {
    if (!c.detail) return null;
    if (c.id === "active_model" && c.detail.includes("Required files (config.json")) {
      return t("doctor.detail.active_model.incomplete");
    }
    return c.detail;
  }

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
      fixMessage = `[${c.id}] ${t("doctor.failed")} ${e}`;
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
      {running ? t("doctor.checking") : t("doctor.recheck")}
    </button>
    {#if fixMessage}
      <span class="dim mono text-[11px]">{fixMessage}</span>
    {/if}
    <span class="dim ml-auto text-[11px]">
      {t("doctor.intro")}
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
            <span class="font-medium">{translatedName(c)}</span>
            <span class="mono text-text-dim text-xs overflow-hidden text-ellipsis whitespace-nowrap">{translatedMessage(c)}</span>
            <span class="text-text-dim text-[10px]">{expanded.has(c.id) ? "▾" : "▸"}</span>
          </button>
          {#if expanded.has(c.id)}
            <div class="px-4 pt-1.5 pb-2.5 pl-[42px] border-t border-border bg-panel">
              {#if translatedDetail(c)}
                <div class="dim mb-1.5 text-[11px] leading-normal">{translatedDetail(c)}</div>
              {/if}
              {#if translatedFixHint(c)}
                <div class="mb-1.5 leading-normal">→ {translatedFixHint(c)}</div>
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
                    {busyAction === c.id ? t("doctor.working") : t("doctor.fixIt")}
                  </button>
                </div>
              {/if}
            </div>
          {/if}
        </div>
      {/each}
    </div>
  {:else}
    <div class="dim">{t("doctor.idle")}</div>
  {/if}
</div>
