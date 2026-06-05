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

/** Per-model geometry needed to size the KV cache and attention transients. */
export type ModelGeometry = {
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

/** Resolve geometry from a model id by substring match (case-insensitive). */
export function geometryForModel(modelId: string): ModelGeometry | null {
  const id = modelId.toLowerCase();
  for (const [key, geom] of Object.entries(MODEL_GEOMETRY)) {
    if (id.includes(key)) return geom;
  }
  return null;
}

export const bytesToGB = (b: number): number => b / GB;
