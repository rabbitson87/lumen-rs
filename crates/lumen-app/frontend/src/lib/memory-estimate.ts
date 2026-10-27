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

import type { FlashNextSettings, HardwareView } from "./api";

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

// ── Flash-Next (qwen4-exp): routed experts streamed from the SSD ─────────────
//
// A different shape from the families above. The 97 GB checkpoint is NOT resident: ~3.6 GB of
// trunk stays in memory, the 68 GB of routed experts are served from the SSD through a pool of
// memory slots, and the server sizes that pool from what the GPU's working set leaves after a
// prefill's reserve (`plan_memory` in `qwen4_exp/runner.rs`):
//
//   reserve = 0.6 GB + 1.1 MB x chunk + KV(longest prompt) + half the attention scores
//   pool    = usable - trunk - reserve                      (`auto`; never below the floor)
//
// so the longest prompt the user allows and the pool trade against each other. That prompt is
// the server's prompt cap (`LUMEN_MAX_PROMPT_TOKENS`, else the legacy `LUMEN_PREFILL_CHUNK`, which
// the app sets from the Prefill card) lowered to `LUMEN_MAX_CTX` (the Max context card) when that
// is smaller; `LUMEN_MAX_CTX` alone never raises it. For Flash-Next the app sends
// `LUMEN_MAX_PROMPT_TOKENS` itself, the machine's default (16,384 tokens, 8,192 on a tight Mac),
// unless the user set it, so the Prefill card (the other families' setting) is not the cap.
// The pool is what decides decode speed.
//
// Sizes are decimal bytes, as the server reads them (`LUMEN_MOE_EXPERT_CACHE_GB=10` is 10e9);
// RAM comes from `SystemInfo.ram_gb`, which is GiB. The constants mirror `src/flash_next.rs`
// (the test in `memory-estimate.test.ts` reads that file) — change them together.

export const FLASH_NEXT = {
  trunkBytes: 3.6e9,
  expertBytes: 2.76e6,
  /** The smallest pool the server runs with: a staging region plus a layer, 2 x 512 slots. */
  poolFloorBytes: 2_826_240_000,
  /** Room for three layers (1,536 slots): the pool lends a second staging region and prefill
   *  reads the next layer while it computes the current one. Below it prefill is slower. */
  poolPrefetchBytes: 4_239_360_000,
  /** A "tight" machine (the GPU may use less than `tightBelowBytes`) takes the smaller values. */
  defaultChunk: 1024,
  tightChunk: 512,
  defaultContext: 16_384,
  tightContext: 8_192,
  maxContext: 262_144,
  tightBelowBytes: 11e9,
  scoresBudgetBytes: 2e9,
  tightScoresBudgetBytes: 0.8e9,
  reserveBaseBytes: 0.6e9,
  reservePerTokenBytes: 1.1e6,
  /** 13 layers (12 full-attention + the draft head's) x (K,V of 2 heads x 256 dims + the
   *  indexer's 128-dim key), bf16: the server's default (`LUMEN_QWEN4_KV_BF16`). */
  kvBytesPerToken: 29_952,
  attentionHeads: 24,
  /** GPU working set as a share of RAM (Apple reports ~2/3 up to 32 GiB, ~3/4 above) and the
   *  share of it the server lets itself fill (`WORKING_SET_SHARE` in lumen-mlx). */
  workingSetShareSmall: 2 / 3,
  workingSetShareLarge: 0.75,
  smallRamGib: 32,
  fillableShare: 0.88,
  /** What the server leaves the rest of the machine when it plans within the memory free at its
   *  load (`LUMEN_MEMORY_PLAN_RECLAIMABLE`, on by default): 2 GB, or 5% of RAM if more. */
  liveHeadroomBytes: 2e9,
  liveHeadroomRamShare: 0.05,
  /** What it takes on disk. */
  diskModelBytes: 97e9,
  diskPackBytes: 68e9,
  diskMtpBytes: 1.5e9,
} as const;

/** Same rule as `family_of` in `lumen-server/src/catalog.rs`. */
export function isFlashNextModelId(id: string | null | undefined): boolean {
  const s = (id ?? "").toLowerCase();
  return s.includes("flash-next") || s.includes("qwen4_exp") || s.includes("qwen4-exp");
}

/** The part of `ramGb` GiB of RAM the server may fill, lowered to `memoryLimitGb` (GiB, what
 *  `LUMEN_MEMORY_LIMIT_GB` allows) when there is one. */
export function flashNextUsableBytes(ramGb: number, memoryLimitGb?: number | null): number {
  const F = FLASH_NEXT;
  const ram = ramGb * GB;
  const share = ram <= F.smallRamGib * GB ? F.workingSetShareSmall : F.workingSetShareLarge;
  const usable = ram * share * F.fillableShare;
  return memoryLimitGb != null && memoryLimitGb > 0 ? Math.min(usable, memoryLimitGb * GB) : usable;
}

/** What the server leaves the rest of the machine when it plans within what is free at its load
 *  (`live_headroom` in `flash_next.rs`). */
export function flashNextLiveHeadroom(ramBytes: number): number {
  const F = FLASH_NEXT;
  return Math.max(F.liveHeadroomBytes, Math.floor(ramBytes * F.liveHeadroomRamShare));
}

/** The memory a server started now plans within (`usable_now` in `flash_next.rs`): `usableBytes`
 *  lowered to what the rest of the machine gives up without being compressed or swapped
 *  (`reclaimableBytes`: free, file-backed and purgeable pages) plus the running server's footprint
 *  (a new one replaces it), less the headroom. The plan's chunk and context stay `usableBytes`';
 *  only the pool shrinks. */
export function flashNextUsableNow(
  usableBytes: number,
  ramBytes: number,
  reclaimableBytes: number,
  serverFootprintBytes: number,
): number {
  const free = reclaimableBytes + serverFootprintBytes - flashNextLiveHeadroom(ramBytes);
  return Math.min(usableBytes, Math.max(0, free));
}

/** What the memory is spent on for a GPU that may use `usableBytes`. */
export type FlashNextPlan = {
  chunk: number;
  /** The longest prompt the plan holds KV and attention scores for. */
  ctxTokens: number;
  tight: boolean;
  /** 0.6 GB of buffers plus the chunk's activations. */
  buffersBytes: number;
  kvBytes: number;
  /** Half the attention scores of a chunk against the whole context (capped by a budget). */
  scoresBytes: number;
  reserveBytes: number;
  /** What `auto` gives the pool: the rest. Below the floor the model does not fit. */
  poolBytes: number;
};

/** The server's plan. `chunk` and `ctxTokens` are what `LUMEN_QWEN35_PREFILL_CHUNK` and
 *  `LUMEN_MAX_PROMPT_TOKENS` / `LUMEN_MAX_CTX` set; null/undefined take the machine's default. */
export function flashNextPlan(
  usableBytes: number,
  chunk?: number | null,
  ctxTokens?: number | null,
): FlashNextPlan {
  const F = FLASH_NEXT;
  const tight = usableBytes < F.tightBelowBytes;
  const c = chunk ?? (tight ? F.tightChunk : F.defaultChunk);
  const x = ctxTokens ?? (tight ? F.tightContext : F.defaultContext);
  const budget = tight ? F.tightScoresBudgetBytes : F.scoresBudgetBytes;
  const scores = Math.min(F.attentionHeads * c * x * 4, budget);
  const buffers = F.reserveBaseBytes + c * F.reservePerTokenBytes;
  const kv = x * F.kvBytesPerToken;
  const reserve = buffers + kv + Math.floor(scores / 2);
  return {
    chunk: c,
    ctxTokens: x,
    tight,
    buffersBytes: buffers,
    kvBytes: kv,
    scoresBytes: Math.floor(scores / 2),
    reserveBytes: reserve,
    poolBytes: Math.max(0, usableBytes - F.trunkBytes - reserve),
  };
}

/** The trunk, the smallest pool and the reserve fit the usable memory. */
export function flashNextPlanFits(plan: FlashNextPlan): boolean {
  return plan.poolBytes >= FLASH_NEXT.poolFloorBytes;
}

/** The longest prompt the server plans memory for, from its prompt cap and `LUMEN_MAX_CTX`
 *  (`context_override` in the server's `qwen4_exp/runner.rs`; mirrored in `flash_next.rs`).
 *  `LUMEN_MAX_CTX` alone is held to the default context. `null`: the machine's default. */
export function flashNextPlannedContext(
  promptCap: number | null,
  maxCtx: number | null,
): number | null {
  if (promptCap != null) return maxCtx != null ? Math.min(promptCap, maxCtx) : promptCap;
  return maxCtx != null ? Math.min(maxCtx, FLASH_NEXT.defaultContext) : null;
}

const positiveInt = (v: string | undefined): number | null => {
  const t = v?.trim();
  return t && /^\+?\d+$/.test(t) && Number(t) > 0 ? Number(t) : null;
};

/** The prompt length the server plans for on a GPU that may use `usableBytes` when nothing sets
 *  it (16,384 tokens, 8,192 on a tight machine); the value the app sends Flash-Next as
 *  `LUMEN_MAX_PROMPT_TOKENS`. */
export function flashNextDefaultContext(usableBytes: number): number {
  return flashNextPlan(usableBytes).ctxTokens;
}

/** What a launch plans for, from the settings the app holds. `LUMEN_MAX_PROMPT_TOKENS` is the
 *  user's override when there is one, else the machine's default (what the app sends); the Max
 *  context card (or a `LUMEN_MAX_CTX` override) only lowers it. */
export function flashNextLaunchContext(
  env: Record<string, string>,
  contextMax: number | undefined,
  usableBytes: number,
): {
  /** The prompt length the plan holds KV for. */
  planned: number;
  /** The server's prompt cap as it will read it. */
  promptCap: number;
  /** Max context as the server will read it. */
  maxCtx: number | null;
  /** The user set `LUMEN_MAX_PROMPT_TOKENS`, rather than the app sending the machine's default. */
  explicit: boolean;
} {
  const explicit = positiveInt(env.LUMEN_MAX_PROMPT_TOKENS);
  const promptCap = explicit ?? flashNextDefaultContext(usableBytes);
  const maxCtx =
    positiveInt(env.LUMEN_MAX_CTX) ?? (contextMax && contextMax > 0 ? contextMax : null);
  return {
    planned: flashNextPlannedContext(promptCap, maxCtx) ?? promptCap,
    promptCap,
    maxCtx,
    explicit: explicit != null,
  };
}

/** Whether a Mac with `ramGb` GiB runs the model at its default plan. */
export function flashNextFits(ramGb: number, memoryLimitGb?: number | null): boolean {
  return flashNextPlanFits(flashNextPlan(flashNextUsableBytes(ramGb, memoryLimitGb)));
}

/** RAM (GiB) below which the model cannot run at its default plan. */
export function flashNextMinRamGb(): number {
  for (let gib = 1; gib <= 1024; gib++) if (flashNextFits(gib)) return gib;
  return 1024;
}

/** The longest prompt (tokens) the plan can hold beside `poolBytes` (`null` = the smallest
 *  pool, i.e. the most this Mac can hold at all). 0 when not even an empty prompt fits. */
export function flashNextMaxContext(
  usableBytes: number,
  chunk?: number | null,
  poolBytes?: number | null,
): number {
  const F = FLASH_NEXT;
  const pool = Math.max(poolBytes ?? F.poolFloorBytes, F.poolFloorBytes);
  const fits = (ctx: number) =>
    F.trunkBytes + pool + flashNextPlan(usableBytes, chunk, ctx).reserveBytes <= usableBytes;
  if (!fits(0)) return 0;
  let lo = 0;
  let hi = F.maxContext;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (fits(mid)) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

/** Upper bound for a pool typed by hand: RAM at the app's usual 85% share, less the trunk. */
export function flashNextMaxPoolGb(ramGb: number): number {
  const gb = Math.floor((ramGb * GB * 0.85 - FLASH_NEXT.trunkBytes) / 1e9);
  return Math.max(gb, FLASH_NEXT.poolFloorBytes / 1e9);
}

export type FlashNextConfig = {
  usableBytes: number;
  /** Expert pool in bytes; `null` = `auto` (the plan's). Raised to the floor if smaller. */
  poolBytes: number | null;
  chunk: number;
  ctxTokens: number;
};

export type FlashNextEstimate = {
  usableBytes: number;
  plan: FlashNextPlan;
  /** The pool actually used: the request, or the plan's, never below the floor. */
  poolBytes: number;
  poolRaisedToFloor: boolean;
  /** trunk + pool + reserve */
  residentBytes: number;
  /** residentBytes / usableBytes. > 1 ⇒ a long prompt can fail with `Insufficient Memory`. */
  utilization: number;
  /** Everything fits the usable memory. With `auto` it does unless the pool hit its floor. */
  fits: boolean;
  /** Under the size that lends a second staging region: slower prefill, nothing fails. */
  oneStagingRegion: boolean;
};

export function flashNextEstimate(cfg: FlashNextConfig): FlashNextEstimate {
  const F = FLASH_NEXT;
  const plan = flashNextPlan(cfg.usableBytes, cfg.chunk, cfg.ctxTokens);
  const wanted = cfg.poolBytes ?? plan.poolBytes;
  const pool = Math.max(wanted, F.poolFloorBytes);
  const resident = F.trunkBytes + pool + plan.reserveBytes;
  return {
    usableBytes: cfg.usableBytes,
    plan,
    poolBytes: pool,
    poolRaisedToFloor: wanted < F.poolFloorBytes,
    residentBytes: resident,
    utilization: resident / cfg.usableBytes,
    // A byte of float slack: with `auto` the sum is `usable` by construction.
    fits: resident <= cfg.usableBytes + 1,
    oneStagingRegion: pool < F.poolPrefetchBytes,
  };
}

// ── Decode speed: what this Mac's memory and SSD buy ─────────────────────────
//
// Decode runs a token at a time. For each, the GPU streams the resident trunk and the ~1.3 GB of
// experts the router picked through memory, and every expert that is not in the pool is read from
// the SSD first. So the speed is set by two numbers of the Mac itself, its memory bandwidth and
// its SSD's random-read speed, and by how often the pool already holds the expert:
//
//   time_per_token_ms = fixed_ms + miss_gb / ssd_effective_gbps * 1000
//   fixed_ms          = streamGb / memory_gbps * 1000 + otherMs
//   miss_gb           = (1 - hitRate(pool_gb)) * expertGbPerToken
//   ssd_effective     = ssd_random_gbps * ssdEfficiency (shards, or the expert pack)
//   tokens/s          = 1000 / time_per_token_ms
//
// The constants mirror `flash_next.rs` (`memory-estimate.test.ts` reads that file), and were
// calibrated on one machine, the reference below. Until this Mac is measured the card shows that
// machine's table as reference material and promises nothing for this Mac.

export const FLASH_NEXT_SPEED_MODEL = {
  /** GB streamed through memory per token: the 1.325 GB of experts a token routes to + the trunk. */
  streamGb: 4.925,
  /** Per-token ms that do not scale with memory bandwidth (dispatch, synchronisation). */
  otherMs: 23.7,
  expertGbPerToken: 1.325,
  /** [pool GB, hit rate] as the engine measured it (300-token window, direct reads); 2.83 GB is
   *  the engine's floor. Linear between points, the first slope continues below the first (never
   *  under 0), held above the last. */
  hitRatePoints: [
    [2.83, 0.477],
    [4.2, 0.554],
    [6.7, 0.664],
    [10.0, 0.75],
    [15.0, 0.826],
    [19.8, 0.867],
  ],
  /** Share of the measured random-read speed decode gets, reading the shards / the pack. */
  ssdEfficiencyShards: 0.63,
  ssdEfficiencyPack: 0.72,
  /** The machine the constants were calibrated on: an M3 Max with 36 GB. */
  referenceMemoryGbps: 281,
  referenceSsdGbps: 6.57,
  /** Below this random-read speed the SSD is the problem worth warning about. */
  slowSsdGbps: 2,
} as const;

/**
 * Decode speed (tokens/s) by expert-pool size on the reference Mac (36 GB M3 Max, memory 281 GB/s,
 * SSD 6.57 GB/s): the pool set explicitly, direct reads (the page cache is not counted), the
 * engine's default eviction, no draft head, 300 tokens of prose, medians of three runs. `shards`
 * reads the shipped weights, `pack` the contiguous expert pack. Shown as reference material until
 * this Mac is measured, and the anchors the model is tested against. A real 16 or 24 GB Mac's own
 * memory pressure is not measured here. Prefill (~230 tokens/s) does not depend on the pool.
 */
export const FLASH_NEXT_DECODE_TPS: readonly {
  poolGb: number;
  shards: number;
  pack: number;
}[] = [
  { poolGb: 2.83, shards: 4.72, pack: 5.44 },
  { poolGb: 4.2, shards: 5.58, pack: 6.14 },
  { poolGb: 6.7, shards: 6.69, pack: 7.15 },
  { poolGb: 10, shards: 8.06, pack: 8.96 },
  { poolGb: 15, shards: 10.39, pack: 11.15 },
  { poolGb: 19.8, shards: 11.96, pack: 12.95 },
];

/** Share of a token's experts a pool of `poolGb` already holds. */
export function flashNextHitRate(poolGb: number): number {
  const pts = FLASH_NEXT_SPEED_MODEL.hitRatePoints;
  const last = pts[pts.length - 1];
  if (poolGb >= last[0]) return last[1];
  let i = pts.findIndex((p, k) => k > 0 && poolGb < p[0]);
  if (i < 1) i = 1;
  const [x0, y0] = pts[i - 1];
  const [x1, y1] = pts[i];
  return Math.min(1, Math.max(0, y0 + ((poolGb - x0) * (y1 - y0)) / (x1 - x0)));
}

const positive = (x: number | null | undefined): x is number =>
  x != null && Number.isFinite(x) && x > 0;

/**
 * Decode speed (tokens/s) for a pool of `poolGb` on a Mac whose memory streams at `memoryGbps` and
 * whose SSD serves random expert-sized reads at `ssdRandomGbps`; `pack` reads the expert pack
 * instead of the shipped shards. A pool under the engine's floor counts as the floor. A faster Mac
 * is not clamped to the reference one. `null` when an input is missing, zero or negative, never a
 * 0 or an infinity.
 */
export function flashNextDecodeTps(
  poolGb: number,
  memoryGbps: number | null | undefined,
  ssdRandomGbps: number | null | undefined,
  pack: boolean,
): number | null {
  if (!positive(poolGb) || !positive(memoryGbps) || !positive(ssdRandomGbps)) return null;
  const M = FLASH_NEXT_SPEED_MODEL;
  // The engine runs no pool under its floor (1,024 slots): a smaller one is that floor.
  const pool = Math.max(poolGb, FLASH_NEXT.poolFloorBytes / 1e9);
  const efficiency = pack ? M.ssdEfficiencyPack : M.ssdEfficiencyShards;
  const fixedMs = (M.streamGb / memoryGbps) * 1000 + M.otherMs;
  const missGb = (1 - flashNextHitRate(pool)) * M.expertGbPerToken;
  const ms = fixedMs + (missGb / (ssdRandomGbps * efficiency)) * 1000;
  return Number.isFinite(ms) && ms > 0 ? 1000 / ms : null;
}

/** The two numbers an estimate for this Mac needs. */
export type FlashNextHardware = { memoryGbps: number; ssdRandomGbps: number };

/** What is measured: both numbers when the estimate can be made, else which are missing. An SSD
 *  number measured on another model folder (`ssd_stale`) counts as missing. */
export function flashNextHardware(hw: HardwareView | null | undefined): {
  hw: FlashNextHardware | null;
  missing: ("memory" | "ssd")[];
  /** The SSD was measured, but on another model folder. */
  ssdStale: boolean;
  /** The older of the two measurement times (unix seconds), when both exist. */
  measuredAt: number | null;
} {
  const memory = positive(hw?.memory_gbps) ? hw.memory_gbps : null;
  const ssd = positive(hw?.ssd_random_gbps) && !hw?.ssd_stale ? hw.ssd_random_gbps : null;
  const missing: ("memory" | "ssd")[] = [];
  if (memory == null) missing.push("memory");
  if (ssd == null) missing.push("ssd");
  const times = [hw?.memory_measured_at, hw?.ssd_measured_at].filter(
    (x): x is number => typeof x === "number",
  );
  return {
    hw: memory != null && ssd != null ? { memoryGbps: memory, ssdRandomGbps: ssd } : null,
    missing,
    ssdStale: hw?.ssd_stale === true,
    measuredAt: memory != null && ssd != null && times.length === 2 ? Math.min(...times) : null,
  };
}

/** Whether the measured SSD is slow enough to warn about. */
export function flashNextSlowSsd(hw: FlashNextHardware | null): boolean {
  return hw != null && hw.ssdRandomGbps < FLASH_NEXT_SPEED_MODEL.slowSsdGbps;
}

/** Disk the model needs, bytes: the shipped weights plus the optional pack and draft head. */
export function flashNextDiskBytes(opts: { pack: boolean; mtp: boolean }): number {
  return (
    FLASH_NEXT.diskModelBytes +
    (opts.pack ? FLASH_NEXT.diskPackBytes : 0) +
    (opts.mtp ? FLASH_NEXT.diskMtpBytes : 0)
  );
}

// ── The Flash-Next option pickers ────────────────────────────────────────────
//
// What the preset chips of the settings show and write. Every option's default equals the
// server's, and choosing the default removes the override rather than writing it down.

export const FLASH_NEXT_POOL_PRESETS_GB = [4, 6, 8, 12, 16, 20] as const;
export const FLASH_NEXT_PROMPT_PRESETS = [8_192, 16_384, 32_768, 65_536, 131_072] as const;
export const FLASH_NEXT_LANE_PRESETS = [8, 16, 32, 64] as const;
export const FLASH_NEXT_LANES_DEFAULT = 32;
/** The smallest pool a hand-typed value may name: just above the server's floor. */
export const FLASH_NEXT_POOL_MIN_GB = 2.9;

/** A pool size in GB as typed: `8`, `8g`, `8GB`, `12.5g` (any case, spaces allowed). `null` when
 *  it is not a positive number. The server reads the same forms. */
export function parseGbInput(text: string): number | null {
  const m = text.trim().match(/^(\d+(?:\.\d+)?)\s*(?:g|gb)?$/i);
  const n = m ? Number(m[1]) : NaN;
  return Number.isFinite(n) && n > 0 ? n : null;
}

/** The plain number of GB the app writes into `LUMEN_MOE_EXPERT_CACHE_GB`: `8`, `12.5`. */
export function formatGbValue(gb: number): string {
  return String(Math.round(gb * 100) / 100);
}

/** A prompt length as typed: tokens (`32768`) or K of 1,024 (`32k`, `1.5K`). `null` when it is
 *  not a positive whole number of tokens. */
export function parsePromptInput(text: string): number | null {
  const m = text.trim().match(/^(\d+(?:\.\d+)?)\s*(k?)$/i);
  if (!m) return null;
  const n = Math.round(Number(m[1]) * (m[2] ? 1024 : 1));
  return Number.isFinite(n) && n > 0 ? n : null;
}

/** `16384` → `16K` when it is a whole number of K, else with thousands separators. */
export function formatPromptShort(tokens: number): string {
  return tokens % 1024 === 0 ? `${tokens / 1024}K` : tokens.toLocaleString("en-US");
}

/** Estimated decode speed of a pool on this Mac (`null` until both numbers are measured), from the
 *  pack's efficiency when the pack is in use, else the shards'. */
export function flashNextChipTps(
  poolBytes: number,
  pack: boolean,
  hw: FlashNextHardware | null,
): { tps: number; withPack: boolean } | null {
  if (hw == null) return null;
  const tps = flashNextDecodeTps(poolBytes / 1e9, hw.memoryGbps, hw.ssdRandomGbps, pack);
  return tps == null ? null : { tps, withPack: pack };
}

export type FlashNextPoolChip = {
  /** The pool the chip means in bytes (`auto`: what the plan leaves, never below the floor). */
  poolBytes: number;
  tps: { tps: number; withPack: boolean } | null;
  /** The pool fits beside the trunk and the reserve for the current longest prompt. */
  fits: boolean;
  /** The largest pool (GB, to 0.1) that fits at this prompt. */
  maxFitGb: number;
};

/** What a pool chip shows: `gb` null is `auto`. A preset larger than the pool that fits at the
 *  current prompt is not offered (the server would lower it and log it). */
export function flashNextPoolChip(a: {
  usableBytes: number;
  chunk: number;
  ctxTokens: number;
  gb: number | null;
  pack: boolean;
  /** This Mac's measured numbers; without them a chip shows no speed. */
  hw: FlashNextHardware | null;
}): FlashNextPoolChip {
  const plan = flashNextPlan(a.usableBytes, a.chunk, a.ctxTokens);
  const planFits = flashNextPlanFits(plan);
  const wanted = a.gb == null ? plan.poolBytes : a.gb * 1e9;
  const poolBytes = Math.max(wanted, FLASH_NEXT.poolFloorBytes);
  return {
    poolBytes,
    tps: flashNextChipTps(poolBytes, a.pack, a.hw),
    fits: planFits && (a.gb == null || wanted <= plan.poolBytes + 1),
    maxFitGb: Math.floor(plan.poolBytes / 1e8) / 10,
  };
}

/** What a prompt chip shows: the pool `auto` leaves with that longest prompt. Under the floor the
 *  model does not fit with it. */
export function flashNextPromptChip(a: { usableBytes: number; chunk: number; tokens: number }): {
  poolBytes: number;
  fits: boolean;
} {
  const plan = flashNextPlan(a.usableBytes, a.chunk, a.tokens);
  return { poolBytes: plan.poolBytes, fits: flashNextPlanFits(plan) };
}

/** A choice read from an override: trimmed, case-insensitive, the default when unset or not one of
 *  `allowed`. */
export function envChoice<T extends string>(
  raw: string | undefined,
  allowed: readonly T[],
  fallback: T,
): T {
  const v = raw?.trim().toLowerCase();
  return (allowed as readonly string[]).includes(v ?? "") ? (v as T) : fallback;
}

/** `LUMEN_QWEN4_MTP` as the server reads it: unset is `off`; `auto` uses the draft head when it is
 *  installed; `1`/`true`/`on` do too but refuse to start without it. */
export function mtpMode(raw: string | undefined): "off" | "auto" | "on" {
  const v = raw?.trim().toLowerCase();
  if (v === "auto") return "auto";
  if (v === "on" || v === "1" || v === "true") return "on";
  return "off";
}

/** Reader threads from `LUMEN_MOE_IO_LANES`: the server's default (32) when unset or invalid. */
export function lanesValue(raw: string | undefined): number {
  const t = raw?.trim();
  const n = t && /^\d+$/.test(t) ? Number(t) : NaN;
  return n >= 1 && n <= 64 ? n : FLASH_NEXT_LANES_DEFAULT;
}

export type FlashNextModelStatus = {
  installed: boolean;
  /** `absent`, `incomplete` or `complete` */
  pack: string;
  mtp_present: boolean;
} | null;

/** Whether decode will read the expert pack: `auto` uses it when it is complete, `build` makes it
 *  first, `off` never does. */
export function packInUse(
  mode: "auto" | "off" | "build",
  status: FlashNextModelStatus,
): boolean {
  if (mode === "off") return false;
  if (mode === "build") return true;
  return status?.pack === "complete";
}

/** Whether the draft head will be loaded: only when chosen (`auto` / `on`), and needs the file. */
export function mtpInUse(mode: "off" | "auto" | "on", status: FlashNextModelStatus): boolean {
  return mode !== "off" && status?.mtp_present === true;
}

/** What the settings run with, for the summary line: the plan's pool and prompt, whether the
 *  pack and the draft head are in use, and the decode speed that estimates. */
export function flashNextEffective(a: {
  env: Record<string, string>;
  contextMax?: number;
  usableBytes: number;
  status: FlashNextModelStatus;
  /** This Mac's measured numbers; without them there is no speed. */
  hw?: FlashNextHardware | null;
}) {
  const poolGb = parseGbInput(a.env.LUMEN_MOE_EXPERT_CACHE_GB ?? "");
  const poolExplicit = poolGb != null && a.env.LUMEN_MOE_EXPERT_CACHE_GB?.trim().toLowerCase() !== "auto";
  const chunkEnv = positiveInt(a.env.LUMEN_QWEN35_PREFILL_CHUNK);
  const defaults = flashNextPlan(a.usableBytes);
  const chunk = chunkEnv ?? defaults.chunk;
  const launch = flashNextLaunchContext(a.env, a.contextMax, a.usableBytes);
  const pack = packInUse(envChoice(a.env.LUMEN_MOE_PACK, ["auto", "off", "build"] as const, "auto"), a.status);
  const head = mtpInUse(mtpMode(a.env.LUMEN_QWEN4_MTP), a.status);
  const estimate = flashNextEstimate({
    usableBytes: a.usableBytes,
    poolBytes: poolExplicit ? poolGb! * 1e9 : null,
    chunk,
    ctxTokens: launch.planned,
  });
  return {
    poolExplicit,
    chunk,
    promptTokens: launch.planned,
    promptExplicit: launch.explicit,
    estimate,
    pack,
    head,
    tps: flashNextChipTps(estimate.poolBytes, pack, a.hw ?? null),
    machineDefaultPrompt: defaults.ctxTokens,
    launch,
  };
}

// ── The settings the Flash-Next card stores ────────────────────────────────
//
// They live in the app's config (`[flash_next]`, `FlashNextConfig` in `src/config.rs`) as typed
// fields and reach a Flash-Next server as the variables below, only when set away from the
// server's default (`FlashNextConfig::env`, which this mirrors; `memory-estimate.test.ts` checks
// the two agree). The env overrides stay as the escape hatch: applied last, they win.

/** The variables the settings stand for. */
export const FLASH_NEXT_ENV_KEYS = [
  "LUMEN_MOE_EXPERT_CACHE_GB",
  "LUMEN_MAX_PROMPT_TOKENS",
  "LUMEN_QWEN35_PREFILL_CHUNK",
  "LUMEN_MOE_PACK",
  "LUMEN_QWEN4_MTP",
  "LUMEN_MOE_IO_MODE",
  "LUMEN_MOE_IO_LANES",
] as const;

/** The settings of a config that never saw the card: every option left to the server. */
export const FLASH_NEXT_DEFAULTS: Readonly<FlashNextSettings> = {
  expert_cache_gb: null,
  max_prompt_tokens: null,
  prefill_chunk: null,
  pack: "auto",
  draft_head: "off",
  io_mode: "auto",
  io_lanes: null,
};

/** The settings are operable only while the Flash-Next model is the selected one. */
export function flashNextSettingsEnabled(selectedModelId: string | null | undefined): boolean {
  return isFlashNextModelId(selectedModelId);
}

const positiveNumber = (n: number | null | undefined): n is number =>
  n != null && Number.isFinite(n) && n > 0;

/** The variables a Flash-Next launch carries for `s`: only options chosen away from the server's
 *  default, as plain values (`8`, not `8g`). A value that could not be one counts as unset. */
export function flashNextSettingsEnv(s: FlashNextSettings): Record<string, string> {
  const env: Record<string, string> = {};
  if (positiveNumber(s.expert_cache_gb)) env.LUMEN_MOE_EXPERT_CACHE_GB = formatGbValue(s.expert_cache_gb);
  if (positiveNumber(s.max_prompt_tokens)) env.LUMEN_MAX_PROMPT_TOKENS = String(s.max_prompt_tokens);
  if (positiveNumber(s.prefill_chunk)) env.LUMEN_QWEN35_PREFILL_CHUNK = String(s.prefill_chunk);
  if (s.pack !== "auto") env.LUMEN_MOE_PACK = s.pack;
  // Off is the server's own default; `auto` and `on` are the opt-ins.
  if (s.draft_head !== "off") env.LUMEN_QWEN4_MTP = s.draft_head;
  if (s.io_mode !== "auto") env.LUMEN_MOE_IO_MODE = s.io_mode;
  if (
    s.io_lanes != null &&
    Number.isInteger(s.io_lanes) &&
    s.io_lanes >= 1 &&
    s.io_lanes <= 64 &&
    s.io_lanes !== FLASH_NEXT_LANES_DEFAULT
  ) {
    env.LUMEN_MOE_IO_LANES = String(s.io_lanes);
  }
  return env;
}

/** What a launch carries: the settings, replaced by an env override of the same name. */
export function flashNextEnv(
  s: FlashNextSettings,
  overrides: Record<string, string>,
): Record<string, string> {
  return { ...flashNextSettingsEnv(s), ...overrides };
}

/** The env overrides that replace what the card says (applied last, they win). */
export function flashNextShadowedKeys(overrides: Record<string, string>): string[] {
  return FLASH_NEXT_ENV_KEYS.filter((k) => k in overrides);
}

/** Every option is at the server's default: nothing to reset. */
export function flashNextIsDefault(s: FlashNextSettings): boolean {
  return Object.keys(flashNextSettingsEnv(s)).length === 0;
}

/** Whether an env-panel entry of this name shadows a typed setting. The Flash-Next names are
 *  typed only while Flash-Next is the selected model; for any other model they are plain entries
 *  (`LUMEN_MAX_PROMPT_TOKENS` and the prefill chunk are rows of the panel for every family). */
export function envShadowsTypedKey(
  key: string,
  typedKeys: ReadonlySet<string>,
  flashNextActive: boolean,
): boolean {
  if (!typedKeys.has(key)) return false;
  return flashNextActive || !(FLASH_NEXT_ENV_KEYS as readonly string[]).includes(key);
}
