// Memory estimator for the native MLX serving path.
//
// Predicts the *peak* working-set so the operator can pick a context / prompt
// cap (`LUMEN_PREFILL_CHUNK`) and a KV mode that fit under the machine's MLX
// memory budget — without trial-and-error OOMs.
//
// The model is calibrated against real `[native-prefill] ... peak=` logs on
// Qwen3.6-35B-A3B-mxfp4 (M3 Max). Three terms scale the footprint:
//
//   active(C)  = weights + KV(C) [+ snapshot(C) when prefix cache is on]
//   peak(C,H)  = active(C) + attnScores(C,H) + OVERHEAD
//
// where C = context tokens, H = prefill/extend chunk tokens.
//
// KV(C)        — grows with context; tiny for GQA models (few KV heads).
// attnScores   — the transient Q·Kᵀ score matrix materialized one full-attn
//                layer at a time: n_heads · H · C · 4 bytes (f32). This is the
//                DOMINANT context-scaling term (≈6× the KV slope at H=2048),
//                which is why shrinking the chunk is the #1 lever for fitting
//                long context — TQ KV mostly trims the persistent `active`.
//
// Validation (Qwen3.6-35B, observed vs predicted transient peak−active):
//   C≈46638, H=2048 → 16·2048·46638·4 = 6.10 GB  (observed ≈ 6.9 GB)
//   C≈50135, H=2048 → 16·2048·50135·4 = 6.56 GB  (observed ≈ 8.4 GB, +snapshot/MoE)
// The fixed OVERHEAD term absorbs MoE intermediates / softmax / mask (~1.5 GB).

const GB = 1024 * 1024 * 1024;

/**
 * Model family — selects which env-var namespace the "Apply" action writes
 * (each backend reads its own `LUMEN_<FAMILY>_*` knobs) and how the KV is
 * stored. Extend alongside `MODEL_GEOMETRY` when a new backend lands.
 */
export type ModelFamily = "qwen35" | "gemma4";

/** Per-model geometry needed to size the KV cache and attention transients. */
export type ModelGeometry = {
  /** Backend family — drives the env namespace written on Apply. */
  family: ModelFamily;
  /** Human label for the picker. */
  label: string;
  /** In-memory weight bytes ≈ on-disk safetensors total. */
  weightsBytes: number;
  /** Full-attention layers — only these grow KV. Linear/SSM layers are flat. */
  fullAttnLayers: number;
  /** KV heads (GQA: often « attention heads → small KV). */
  nKvHeads: number;
  /** Attention heads — drives the transient score-matrix size. */
  nHeads: number;
  headDim: number;
  /** Hard upper bound on context (`max_position_embeddings`). */
  maxContext: number;
};

/**
 * Known catalog geometries, keyed by a substring of the model id. Values are
 * read from each model's `config.json` + safetensors sizes (see the Rust
 * `text_config`). Extend as models are added; `weightsBytes` is the quant
 * checkpoint's actual size.
 */
export const MODEL_GEOMETRY: Record<string, ModelGeometry> = {
  "qwen3.6-35b": {
    family: "qwen35",
    label: "Qwen3.6-35B-A3B mxfp4",
    // MLX-RESIDENT footprint, not the 19.32 GB on-disk safetensors size:
    // observed `active=20.7GB` at a 2k-token prefill (weights + runtime). Anchor
    // to the operator's own startup `active=` log for exactness — the disk size
    // under-counts the resident set by ~2.4 GiB (GPU-packed format + runtime).
    weightsBytes: 21_900_000_000, // ~20.4 GiB resident
    fullAttnLayers: 10, // 40 layers, full_attention_interval=4
    nKvHeads: 2,
    nHeads: 16,
    headDim: 256,
    maxContext: 262_144,
  },
  // ── Qwen 27B dense (3.6 and 3.8) ──────────────────────────────────────
  // Same geometry for both: `text_config` is byte-identical between the two
  // generations (64 layers, full_attention_interval=4 → 16 full-attn layers,
  // 24 heads / 4 KV heads, head_dim 256). Only `weightsBytes` differs, because
  // the 3.8 build is quantized differently. Both keys existed as a gap before —
  // the calculator fell through to "No memory profile for model" for every 27B.
  "qwen3.8-27b": {
    family: "qwen35",
    label: "Qwen3.8-27B Dense (MTPLX 4/8-bit)",
    // Sum of the shipped repo: 19.5 GB trunk + 0.92 GB vision tower + 0.85 GB
    // MTP head = 21.3 GB on disk, plus the ~2.4 GiB runtime gap the 35B entry
    // documents. UNLIKE the other two entries this is NOT yet anchored to an
    // observed startup `active=` — treat it as an upper-ish estimate and
    // re-anchor from the operator's own log.
    weightsBytes: 23_900_000_000, // ~22.3 GiB resident (estimated, not observed)
    fullAttnLayers: 16, // 64 layers, full_attention_interval=4
    nKvHeads: 4,
    nHeads: 24,
    headDim: 256,
    maxContext: 262_144,
  },
  "qwen3.6-27b": {
    family: "qwen35",
    label: "Qwen3.6-27B Dense (MTPLX 4-bit)",
    // Flat affine 4-bit group-64: ~16 GB on disk + the runtime gap.
    weightsBytes: 18_400_000_000, // ~17.1 GiB resident (estimated, not observed)
    fullAttnLayers: 16,
    nKvHeads: 4,
    nHeads: 24,
    headDim: 256,
    maxContext: 262_144,
  },
  "gemma-4-26b": {
    family: "gemma4",
    label: "Gemma 4 26B-A4B (4-bit)",
    // Resident ≈ 4-bit checkpoint + runtime. Anchor to the operator's own
    // startup `active=` log for exactness (nvfp4/mxfp4 variants differ ±2 GiB).
    weightsBytes: 15_800_000_000, // ~14.7 GiB resident (it-4bit class)
    // 30 layers, layer_types has 5 full-attention; the other 25 are
    // sliding-window (1024) → their KV is BOUNDED and does not scale with
    // context, so only the 5 full-attn layers drive the context-growth term.
    fullAttnLayers: 5,
    nKvHeads: 8,
    nHeads: 16,
    headDim: 256,
    maxContext: 262_144,
  },
};

/** How the full-attention KV is stored. `bytesPerElem` is per K/V scalar. */
export type KvMode = "bf16" | "tq" | "tq_packed4";

/**
 * Bytes per stored K/V element for each mode.
 * - bf16: 2 B.
 * - tq:  uint8 code (1 B) + per-vector f32 σ amortized over head_dim.
 *        ≈ 1 + 4/headDim. NOTE: identical for any `bits` (2-8) because the
 *        current cache stores codes UNPACKED — bits changes quality, not size.
 * - tq_packed4: future uint4 packing (0.5 B/code + σ). Not yet wired into the
 *        cache; shown so the calculator can preview the win.
 */
export function kvBytesPerElem(mode: KvMode, headDim: number): number {
  switch (mode) {
    case "bf16":
      return 2;
    case "tq":
      return 1 + 4 / headDim;
    case "tq_packed4":
      return 0.5 + 4 / headDim;
  }
}

export type MemoryConfig = {
  geometry: ModelGeometry;
  /** MLX memory budget in bytes (machine-dependent — e.g. memory_limit). */
  budgetBytes: number;
  kvMode: KvMode;
  /** Prefill / extend chunk size (`LUMEN_PREFILL_CHUNK`). */
  chunkTokens: number;
  /** Whether the prefix cache holds a snapshot (~1 extra KV copy). */
  prefixCache: boolean;
  /** Fixed transient overhead (MoE/softmax/mask), bytes. */
  overheadBytes?: number;
};

// Transient extras not in the score-matrix term: MoE expert intermediates,
// softmax/mask buffers, and the prefix snapshot of the 30 linear-attn SSM/conv
// states. Grows mildly with context; this constant is tuned to be accurate in
// the high-context regime where the budget actually binds (slightly
// conservative at short context — the safe direction for limit-setting).
// Calibrate against your own `peak=` minus `active=` logs.
const DEFAULT_OVERHEAD = Math.round(2.5 * GB);

/** KV cache bytes for `contextTokens` under the given mode. */
export function kvBytes(
  g: ModelGeometry,
  mode: KvMode,
  contextTokens: number,
): number {
  const perElem = kvBytesPerElem(mode, g.headDim);
  // K and V, per full-attn layer, per KV head, per head_dim element, per token.
  return g.fullAttnLayers * 2 * g.nKvHeads * g.headDim * perElem * contextTokens;
}

/** Transient attention score-matrix bytes at `contextTokens` with `chunk`. */
export function attnScoreBytes(
  g: ModelGeometry,
  contextTokens: number,
  chunkTokens: number,
): number {
  return g.nHeads * Math.min(chunkTokens, contextTokens) * contextTokens * 4;
}

export type MemoryBreakdown = {
  weightsBytes: number;
  kvBytes: number;
  snapshotBytes: number;
  /** Persistent working set (weights + KV [+ snapshot]). */
  activeBytes: number;
  attnScoreBytes: number;
  overheadBytes: number;
  /** active + transient — the value that must stay under budget. */
  peakBytes: number;
  budgetBytes: number;
  /** peakBytes / budgetBytes. > 1 ⇒ OOM / thrash. */
  utilization: number;
  fits: boolean;
};

/** Full breakdown at a specific context length. */
export function estimateAt(
  cfg: MemoryConfig,
  contextTokens: number,
): MemoryBreakdown {
  const g = cfg.geometry;
  const overhead = cfg.overheadBytes ?? DEFAULT_OVERHEAD;
  const kv = kvBytes(g, cfg.kvMode, contextTokens);
  const snapshot = cfg.prefixCache ? kv : 0;
  const active = g.weightsBytes + kv + snapshot;
  const attn = attnScoreBytes(g, contextTokens, cfg.chunkTokens);
  const peak = active + attn + overhead;
  return {
    weightsBytes: g.weightsBytes,
    kvBytes: kv,
    snapshotBytes: snapshot,
    activeBytes: active,
    attnScoreBytes: attn,
    overheadBytes: overhead,
    peakBytes: peak,
    budgetBytes: cfg.budgetBytes,
    utilization: peak / cfg.budgetBytes,
    fits: peak <= cfg.budgetBytes,
  };
}

/**
 * Largest context (tokens) whose peak still fits the budget, for this config.
 * Closed form: once C ≥ chunk, peak(C) is linear in C
 *   peak = weights + overhead + C·(kvSlope·(1+snap) + nHeads·chunk·4)
 * so solve for C. Clamped to [0, maxContext]. Returns 0 if even an empty
 * context's fixed cost (weights + overhead) already exceeds the budget.
 */
export function maxContextTokens(cfg: MemoryConfig): number {
  const g = cfg.geometry;
  const overhead = cfg.overheadBytes ?? DEFAULT_OVERHEAD;
  const fixed = g.weightsBytes + overhead;
  if (fixed >= cfg.budgetBytes) return 0;

  const perElem = kvBytesPerElem(cfg.kvMode, g.headDim);
  const kvSlope = g.fullAttnLayers * 2 * g.nKvHeads * g.headDim * perElem;
  const snapMul = cfg.prefixCache ? 2 : 1;
  // attnScore slope assumes C ≥ chunk (the long-context regime we size for).
  const attnSlope = g.nHeads * cfg.chunkTokens * 4;
  const slope = kvSlope * snapMul + attnSlope;

  const c = Math.floor((cfg.budgetBytes - fixed) / slope);
  return Math.max(0, Math.min(c, g.maxContext));
}

/**
 * A budget-optimized config: the knob set the calculator fills when the
 * operator picks a machine RAM budget. Maximizes the usable working set
 * (prompt cap + output) under the budget, conceding KV quality / prefill speed
 * only as the budget tightens.
 */
export type Recommendation = {
  kvMode: KvMode; // "bf16" | "tq" (packed not auto-recommended — not wired)
  bits: number; // 8 | 6 | 4 — meaningful only when kvMode = "tq"
  chunk: number; // per-step prefill chunk (OOM lever)
  prefix: boolean; // prefix-cache snapshot (+1 KV copy; agentic speed)
  prefillLimit: number; // max prompt tokens (reject cap) → context.prefill
  outMaxTokens: number; // output budget → context.default_max_tokens
  ctxMax: number; // context.max to set (= working set)
  /** prefillLimit + outMaxTokens — the peak-sizing sequence length. */
  workingSet: number;
  /**
   * "ok"          — fits the model's full context window with headroom.
   * "aggressive"  — budget-bound; this is the most context it can hold.
   * "tooSmall"    — budget below the model's load floor (weights+overhead).
   */
  note: "ok" | "aggressive" | "tooSmall";
};

const RECO_SAFETY = 0.9; // leave 10% headroom over the predicted peak
const RECO_OUT_RESERVE = 8192; // default output-token budget carved from WS
const RECO_USABLE_CTX = 8192; // min context to call a prefix-on config "usable"
const RECO_AMPLE_CTX = 32768; // context above which the config is "ample" (ok)
const RECO_BAL_CHUNK = 512; // balanced prefill chunk (speed vs attn-score peak)
// KV ladder: least-aggressive (best quality/decode) → most. bf16 first so we
// only quantize when the budget can't otherwise hold a usable context.
const RECO_KV_LADDER: { kvMode: KvMode; bits: number }[] = [
  { kvMode: "bf16", bits: 8 },
  { kvMode: "tq", bits: 8 },
  { kvMode: "tq", bits: 6 },
  { kvMode: "tq", bits: 4 },
];
// Chunk ladder: largest (fastest prefill) → smallest (lowest attn-score peak).
const RECO_CHUNKS = [2048, 1024, 512, 256];

/**
 * Recommend a memory-fitted config for `budgetBytes` on this model — **balanced
 * for agentic/omp use**: keep the prefix cache ON (the dominant warm-turn lever)
 * and full bf16 quality whenever the budget can still hold a usable context;
 * trade those down only when it can't. Strategy:
 *  1. Phase A (preferred): least-aggressive KV whose prefix-ON context at the
 *     balanced chunk is ≥ usable. bf16 first → only quantize if bf16 prefix-on
 *     can't reach a usable context.
 *  2. Phase B (tight budget): if no KV gives a usable prefix-on context, drop
 *     the prefix cache (frees one KV copy) to salvage context.
 *  3. Pick the largest chunk that still fits the chosen context (prefill speed).
 *  4. Reserve an output budget; the rest is the prompt-reject cap.
 */
export function recommendForBudget(
  g: ModelGeometry,
  budgetBytes: number,
): Recommendation {
  const E = budgetBytes * RECO_SAFETY;
  const fixed = g.weightsBytes + DEFAULT_OVERHEAD;
  if (fixed >= E) {
    return {
      kvMode: "tq",
      bits: 4,
      chunk: 256,
      prefix: false,
      prefillLimit: 0,
      outMaxTokens: RECO_OUT_RESERVE,
      ctxMax: 0,
      workingSet: 0,
      note: "tooSmall",
    };
  }

  const peakAt = (
    kvMode: KvMode,
    chunk: number,
    prefix: boolean,
    ctx: number,
  ): number =>
    estimateAt(
      { geometry: g, budgetBytes, kvMode, chunkTokens: chunk, prefixCache: prefix },
      ctx,
    ).peakBytes;
  const maxCtxOf = (kvMode: KvMode, chunk: number, prefix: boolean): number =>
    Math.min(
      g.maxContext,
      maxContextTokens({
        geometry: g,
        budgetBytes: E,
        kvMode,
        chunkTokens: chunk,
        prefixCache: prefix,
      }),
    );
  const largestChunk = (kvMode: KvMode, prefix: boolean, ctx: number): number => {
    for (const ch of RECO_CHUNKS) if (peakAt(kvMode, ch, prefix, ctx) <= E) return ch;
    return 256;
  };
  const build = (
    kv: { kvMode: KvMode; bits: number },
    prefix: boolean,
    ws: number,
  ): Recommendation => {
    const chunk = largestChunk(kv.kvMode, prefix, ws);
    const out = Math.min(RECO_OUT_RESERVE, Math.max(256, Math.floor(ws * 0.1)));
    const prefillLimit = Math.max(256, ws - out);
    const note =
      kv.kvMode === "bf16" && ws >= RECO_AMPLE_CTX ? "ok" : "aggressive";
    return {
      kvMode: kv.kvMode,
      bits: kv.bits,
      chunk,
      prefix,
      prefillLimit,
      outMaxTokens: out,
      ctxMax: ws,
      workingSet: prefillLimit + out,
      note,
    };
  };

  // Phase A — prefix cache ON, least-aggressive KV that stays usable.
  for (const kv of RECO_KV_LADDER) {
    const ws = maxCtxOf(kv.kvMode, RECO_BAL_CHUNK, true);
    if (ws >= RECO_USABLE_CTX) return build(kv, true, ws);
  }
  // Phase B — budget too tight for a prefix-on context; drop prefix to salvage.
  for (const kv of RECO_KV_LADDER) {
    const ws = maxCtxOf(kv.kvMode, 256, false);
    if (ws >= 1024) return build(kv, false, ws);
  }
  return {
    kvMode: "tq",
    bits: 4,
    chunk: 256,
    prefix: false,
    prefillLimit: 256,
    outMaxTokens: RECO_OUT_RESERVE,
    ctxMax: 0,
    workingSet: 256,
    note: "tooSmall",
  };
}

/**
 * Per-family env-var namespace for the calculator-controlled knobs. Each
 * backend reads its OWN `LUMEN_<FAMILY>_*` keys, so Apply must write the right
 * set (and the seed must read it back from the same keys). `prefix` is shared
 * across families (the MLX prefix cache is backend-agnostic).
 */
export type FamilyEnvKeys = {
  /** KV quant on/off. Qwen35: "1"/"0"; Gemma4: a mode string ("on"/"off"). */
  kv: string;
  /** Whether `kv` is a boolean ("1"/"0") or a mode enum ("on"/"off"). */
  kvKind: "bool" | "mode";
  kvBits: string;
  prefillChunk: string;
};

export function envKeysForFamily(family: ModelFamily): FamilyEnvKeys {
  switch (family) {
    case "qwen35":
      return {
        kv: "LUMEN_QWEN35_TQ_KV",
        kvKind: "bool",
        kvBits: "LUMEN_QWEN35_TQ_KV_BITS",
        prefillChunk: "LUMEN_QWEN35_PREFILL_CHUNK",
      };
    case "gemma4":
      return {
        kv: "LUMEN_GEMMA4_QUANT_KV_MODE",
        kvKind: "mode",
        kvBits: "LUMEN_GEMMA4_QUANT_KV_BITS",
        prefillChunk: "LUMEN_GEMMA4_PREFILL_CHUNK",
      };
  }
}

/** Resolve geometry from a model id by substring match (case-insensitive). */
export function geometryForModel(modelId: string): ModelGeometry | null {
  const id = modelId.toLowerCase();
  for (const [key, geom] of Object.entries(MODEL_GEOMETRY)) {
    if (id.includes(key)) return geom;
  }
  return null;
}

export const bytesToGB = (b: number): number => b / GB;
