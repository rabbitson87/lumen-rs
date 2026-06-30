<script lang="ts">
  import { api } from "./api";
  import { t } from "./i18n.svelte";

  // The active image model id (e.g. "flux2-dev") — shown as the panel subtitle.
  let { modelId }: { modelId: string } = $props();

  // ── Generation controls ──────────────────────────────────────────
  let prompt = $state("");
  let size = $state<"256x256" | "512x512" | "1024x1024">("512x512");
  let steps = $state(28);
  let seed = $state(0);
  let guidance = $state(4.0);

  // ── Result / status ──────────────────────────────────────────────
  let generating = $state(false);
  let error = $state<string | null>(null);
  // Base-64 PNG (no data: prefix) of the most-recent generation.
  let resultB64 = $state<string | null>(null);
  // Wall-clock seconds of the in-flight generation (live counter).
  let elapsed = $state(0);
  let timer: ReturnType<typeof setInterval> | undefined;

  let dataUrl = $derived(resultB64 ? `data:image/png;base64,${resultB64}` : null);

  // A filesystem-friendly download name derived from the prompt + seed.
  let downloadName = $derived.by(() => {
    const slug =
      prompt
        .trim()
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-+|-+$/g, "")
        .slice(0, 40) || "image";
    return `lumen-${slug}-${seed}.png`;
  });

  async function generate() {
    if (generating || !prompt.trim()) return;
    error = null;
    generating = true;
    elapsed = 0;
    timer = setInterval(() => (elapsed += 1), 1000);
    try {
      const img = await api.generateImage({
        prompt: prompt.trim(),
        size,
        steps,
        seed,
        guidance,
      });
      resultB64 = img.b64_json;
    } catch (e) {
      error = String(e);
    } finally {
      generating = false;
      if (timer) clearInterval(timer);
    }
  }

  const fieldRow =
    "grid grid-cols-[120px_1fr] items-center gap-2.5 py-1.5";
</script>

<section class="bg-panel border border-border rounded-[10px] px-5 py-4.5 min-h-0 col-span-3">
  <h2 class="m-0 mb-3.5 text-xs font-semibold tracking-[0.08em] text-text-dim uppercase flex items-center gap-2">
    <span class="image-badge">{t("imageModels.badge")}</span>
    {t("image.title")}
    <span class="dim mono normal-case tracking-normal ml-1">{modelId}</span>
  </h2>

  <!-- Prompt -->
  <div class="flex flex-col gap-1 mb-2">
    <span class="dim text-xs">{t("image.prompt")}</span>
    <textarea
      class="w-full min-h-20 resize-y mono text-[13px]"
      placeholder={t("image.prompt.placeholder")}
      bind:value={prompt}
      disabled={generating}
    ></textarea>
  </div>

  <!-- Controls -->
  <div class={fieldRow}>
    <span class="dim">{t("image.size")}</span>
    <select bind:value={size} disabled={generating}>
      <option value="256x256">256 × 256</option>
      <option value="512x512">512 × 512</option>
      <option value="1024x1024">1024 × 1024</option>
    </select>
  </div>
  <div class={fieldRow}>
    <span class="dim">{t("image.steps")}</span>
    <input
      type="number"
      min="1"
      max="100"
      step="1"
      bind:value={steps}
      disabled={generating}
    />
  </div>
  <div class="dim -mt-0.5 mb-1 ml-32.5 text-[11px] leading-[1.55]">
    {t("image.steps.hint")}
  </div>
  <div class={fieldRow}>
    <span class="dim">{t("image.seed")}</span>
    <input
      type="number"
      min="0"
      step="1"
      bind:value={seed}
      disabled={generating}
    />
  </div>
  <div class={fieldRow}>
    <span class="dim">{t("image.guidance")}</span>
    <input
      type="number"
      min="0"
      max="20"
      step="0.5"
      bind:value={guidance}
      disabled={generating}
    />
  </div>

  <!-- Generate -->
  <div class="flex items-center gap-3 mt-3">
    <button
      class="primary"
      onclick={generate}
      disabled={generating || !prompt.trim()}
    >
      {#if generating}
        <span class="spinner" aria-hidden="true"></span>
        {t("image.generating")} ({elapsed}s)
      {:else}
        {t("image.generate")}
      {/if}
    </button>
    {#if generating}
      <span class="dim text-[11px] leading-[1.4]">{t("image.generating.note")}</span>
    {/if}
  </div>

  {#if error}
    <p class="mt-3 px-3 py-2 bg-err/10 border border-err rounded text-xs text-err leading-normal break-words whitespace-pre-wrap">
      {error}
    </p>
  {/if}

  {#if dataUrl}
    <div class="mt-4 flex flex-col gap-2">
      <img
        src={dataUrl}
        alt={prompt || "generated image"}
        class="max-w-full rounded-md border border-border bg-bg"
      />
      <a
        href={dataUrl}
        download={downloadName}
        class="self-start text-[11px] px-2.5 py-1 border border-accent text-accent rounded-md hover:bg-accent/6 transition-colors no-underline"
      >
        ↓ {t("image.download")}
      </a>
    </div>
  {/if}
</section>

<style>
  /* Inline button spinner — small ring that doesn't depend on any global
     keyframes already in app.css. */
  .spinner {
    display: inline-block;
    width: 12px;
    height: 12px;
    margin-right: 2px;
    vertical-align: -1px;
    border: 2px solid currentColor;
    border-top-color: transparent;
    border-radius: 50%;
    animation: img-spin 0.7s linear infinite;
  }
  @keyframes img-spin {
    to {
      transform: rotate(360deg);
    }
  }

  /* "IMAGE" badge chip — marks the active model in this panel header as an
     image / diffusion model, mirroring the corner ribbon on the model cards.
     Uses the accent token (no hardcoded brand colors). */
  .image-badge {
    display: inline-block;
    padding: 1px 6px;
    border-radius: 4px;
    background: var(--accent);
    color: var(--bg);
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.1em;
    line-height: 1.4;
    text-transform: uppercase;
  }
</style>
