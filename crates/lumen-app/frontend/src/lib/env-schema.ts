// Schema-driven env overrides.
//
// Each entry surfaces one lumen-server env var as a typed UI control.
// Render rules:
//   bool   → checkbox (stored as "1" / "0")
//   number → number input with min/max range
//   enum   → select dropdown from `options`
//   string → text input
//
// Values are still persisted as `Record<string, string>` (matches the
// existing PersistentConfig.env_overrides shape on the Rust side); the
// widgets only handle parse/format on the UI layer.

export type EnvControlKind = "bool" | "number" | "enum" | "string";

export type EnvEntry = {
  key: string;
  /** i18n key for the display label */
  labelKey: string;
  /** i18n key for the longer help text */
  helpKey?: string;
  kind: EnvControlKind;
  /** number range — inclusive */
  min?: number;
  max?: number;
  /** number step (default 1) */
  step?: number;
  /** enum options — first is treated as the "off" / "disabled" sentinel */
  options?: string[];
  /** stringified default — used as placeholder & "is set to default" check */
  defaultValue: string;
  /** UI section grouping */
  section: "thinking" | "sampling" | "safety" | "debug" | "advanced";
};

/**
 * Whitelist of env vars exposed through typed UI controls.
 *
 * Adding a new entry:
 *   1. Append here with appropriate kind / range / default.
 *   2. Add i18n strings `env.entry.<KEY>.label` and `.help`.
 *   3. Server-side reader already exists or is added in the same PR.
 */
export const ENV_SCHEMA: EnvEntry[] = [
  // ── Reasoning (per-request opt-in default) ────────────────────────
  {
    key: "LUMEN_BACKEND_THINKING_DEFAULT",
    labelKey: "env.entry.LUMEN_BACKEND_THINKING_DEFAULT.label",
    helpKey: "env.entry.LUMEN_BACKEND_THINKING_DEFAULT.help",
    kind: "enum",
    options: ["off", "on"],
    defaultValue: "off",
    section: "thinking",
  },

  // ── Sampling (Gemma 4 ↔ Ollama parity rails) ───────────────────────
  {
    key: "LUMEN_TEMPERATURE",
    labelKey: "env.entry.LUMEN_TEMPERATURE.label",
    helpKey: "env.entry.LUMEN_TEMPERATURE.help",
    kind: "number",
    min: 0,
    max: 2,
    step: 0.05,
    defaultValue: "1",
    section: "sampling",
  },
  {
    key: "LUMEN_TOP_P",
    labelKey: "env.entry.LUMEN_TOP_P.label",
    helpKey: "env.entry.LUMEN_TOP_P.help",
    kind: "number",
    min: 0,
    max: 1,
    step: 0.05,
    defaultValue: "0.95",
    section: "sampling",
  },
  {
    key: "LUMEN_TOP_K",
    labelKey: "env.entry.LUMEN_TOP_K.label",
    helpKey: "env.entry.LUMEN_TOP_K.help",
    kind: "number",
    min: 0,
    max: 200,
    step: 1,
    defaultValue: "64",
    section: "sampling",
  },
  {
    key: "REPEAT_PENALTY",
    labelKey: "env.entry.REPEAT_PENALTY.label",
    helpKey: "env.entry.REPEAT_PENALTY.help",
    kind: "number",
    min: 1,
    max: 2,
    step: 0.05,
    defaultValue: "1.1",
    section: "sampling",
  },
  {
    key: "LUMEN_DRY_MULTIPLIER",
    labelKey: "env.entry.LUMEN_DRY_MULTIPLIER.label",
    helpKey: "env.entry.LUMEN_DRY_MULTIPLIER.help",
    kind: "number",
    min: 0,
    max: 4,
    step: 0.1,
    defaultValue: "0",
    section: "sampling",
  },

  // ── Safety net (runaway / channel close) ───────────────────────────
  {
    key: "LUMEN_MAX_THINKING_TOKENS",
    labelKey: "env.entry.LUMEN_MAX_THINKING_TOKENS.label",
    helpKey: "env.entry.LUMEN_MAX_THINKING_TOKENS.help",
    kind: "number",
    min: 0,
    max: 4096,
    step: 50,
    defaultValue: "0",
    section: "safety",
  },
  {
    key: "LUMEN_MAX_FORCE_CLOSE_ATTEMPTS",
    labelKey: "env.entry.LUMEN_MAX_FORCE_CLOSE_ATTEMPTS.label",
    helpKey: "env.entry.LUMEN_MAX_FORCE_CLOSE_ATTEMPTS.help",
    kind: "number",
    min: 1,
    max: 3,
    step: 1,
    defaultValue: "1",
    section: "safety",
  },
  {
    key: "LUMEN_RUNAWAY_DETECT",
    labelKey: "env.entry.LUMEN_RUNAWAY_DETECT.label",
    helpKey: "env.entry.LUMEN_RUNAWAY_DETECT.help",
    kind: "bool",
    defaultValue: "1",
    section: "safety",
  },
  {
    key: "LUMEN_RUNAWAY_NGRAM",
    labelKey: "env.entry.LUMEN_RUNAWAY_NGRAM.label",
    helpKey: "env.entry.LUMEN_RUNAWAY_NGRAM.help",
    kind: "number",
    min: 2,
    max: 12,
    step: 1,
    defaultValue: "4",
    section: "safety",
  },
  {
    key: "LUMEN_RUNAWAY_NGRAM_MAX_REPEATS",
    labelKey: "env.entry.LUMEN_RUNAWAY_NGRAM_MAX_REPEATS.label",
    helpKey: "env.entry.LUMEN_RUNAWAY_NGRAM_MAX_REPEATS.help",
    kind: "number",
    min: 4,
    max: 64,
    step: 1,
    defaultValue: "8",
    section: "safety",
  },

  // ── Advanced (model-side correctness knobs) ────────────────────────
  {
    key: "LUMEN_GEMMA4_CRITICAL_LOGIT_CORRECTION",
    labelKey: "env.entry.LUMEN_GEMMA4_CRITICAL_LOGIT_CORRECTION.label",
    helpKey: "env.entry.LUMEN_GEMMA4_CRITICAL_LOGIT_CORRECTION.help",
    kind: "bool",
    defaultValue: "1",
    section: "advanced",
  },
  {
    key: "LUMEN_GEMMA4_GRAMMAR_LARK",
    labelKey: "env.entry.LUMEN_GEMMA4_GRAMMAR_LARK.label",
    helpKey: "env.entry.LUMEN_GEMMA4_GRAMMAR_LARK.help",
    kind: "bool",
    defaultValue: "1",
    section: "advanced",
  },
  {
    key: "LUMEN_TOOL_CHOICE_AUTO_AS_REQUIRED",
    labelKey: "env.entry.LUMEN_TOOL_CHOICE_AUTO_AS_REQUIRED.label",
    helpKey: "env.entry.LUMEN_TOOL_CHOICE_AUTO_AS_REQUIRED.help",
    kind: "bool",
    defaultValue: "0",
    section: "advanced",
  },
  {
    key: "LUMEN_GEMMA4_EMPTY_THOUGHT_ON_NOTHINK",
    labelKey: "env.entry.LUMEN_GEMMA4_EMPTY_THOUGHT_ON_NOTHINK.label",
    helpKey: "env.entry.LUMEN_GEMMA4_EMPTY_THOUGHT_ON_NOTHINK.help",
    kind: "bool",
    defaultValue: "0",
    section: "advanced",
  },
  {
    key: "LUMEN_USE_JINJA_RENDERER",
    labelKey: "env.entry.LUMEN_USE_JINJA_RENDERER.label",
    helpKey: "env.entry.LUMEN_USE_JINJA_RENDERER.help",
    kind: "bool",
    defaultValue: "0",
    section: "advanced",
  },
  {
    key: "LUMEN_QWEN35_FORCE_REQUIRED_PARAMS",
    labelKey: "env.entry.LUMEN_QWEN35_FORCE_REQUIRED_PARAMS.label",
    helpKey: "env.entry.LUMEN_QWEN35_FORCE_REQUIRED_PARAMS.help",
    kind: "bool",
    defaultValue: "0",
    section: "advanced",
  },
  {
    key: "LUMEN_QWEN35_TQ_KV",
    labelKey: "env.entry.LUMEN_QWEN35_TQ_KV.label",
    helpKey: "env.entry.LUMEN_QWEN35_TQ_KV.help",
    kind: "bool",
    defaultValue: "0",
    section: "advanced",
  },
  {
    key: "LUMEN_QWEN35_TQ_KV_BITS",
    labelKey: "env.entry.LUMEN_QWEN35_TQ_KV_BITS.label",
    helpKey: "env.entry.LUMEN_QWEN35_TQ_KV_BITS.help",
    kind: "number",
    min: 2,
    max: 8,
    step: 1,
    defaultValue: "8",
    section: "advanced",
  },

  // ── Debug (developer / triage) ─────────────────────────────────────
  {
    key: "LUMEN_DUMP_PROMPT",
    labelKey: "env.entry.LUMEN_DUMP_PROMPT.label",
    helpKey: "env.entry.LUMEN_DUMP_PROMPT.help",
    kind: "enum",
    options: ["off", "preview", "full"],
    defaultValue: "off",
    section: "debug",
  },
  {
    key: "LUMEN_LOG_REQUEST_BODY",
    labelKey: "env.entry.LUMEN_LOG_REQUEST_BODY.label",
    helpKey: "env.entry.LUMEN_LOG_REQUEST_BODY.help",
    kind: "bool",
    defaultValue: "0",
    section: "debug",
  },
  {
    key: "LUMEN_GEMMA4_TOKEN_TRACE",
    labelKey: "env.entry.LUMEN_GEMMA4_TOKEN_TRACE.label",
    helpKey: "env.entry.LUMEN_GEMMA4_TOKEN_TRACE.help",
    kind: "bool",
    defaultValue: "0",
    section: "debug",
  },
  {
    key: "LUMEN_EOS_GUARD_VERBOSE",
    labelKey: "env.entry.LUMEN_EOS_GUARD_VERBOSE.label",
    helpKey: "env.entry.LUMEN_EOS_GUARD_VERBOSE.help",
    kind: "bool",
    defaultValue: "0",
    section: "debug",
  },
  {
    key: "LUMEN_QWEN35_PREFILL_CHUNK",
    labelKey: "env.entry.LUMEN_QWEN35_PREFILL_CHUNK.label",
    helpKey: "env.entry.LUMEN_QWEN35_PREFILL_CHUNK.help",
    kind: "number",
    min: 256,
    max: 32768,
    step: 256,
    defaultValue: "2048",
    section: "advanced",
  },
  {
    key: "LUMEN_QWEN35_PREFILL_CHUNK_LOG",
    labelKey: "env.entry.LUMEN_QWEN35_PREFILL_CHUNK_LOG.label",
    helpKey: "env.entry.LUMEN_QWEN35_PREFILL_CHUNK_LOG.help",
    kind: "bool",
    defaultValue: "0",
    section: "debug",
  },
  {
    key: "LUMEN_NATIVE_TIMING",
    labelKey: "env.entry.LUMEN_NATIVE_TIMING.label",
    helpKey: "env.entry.LUMEN_NATIVE_TIMING.help",
    kind: "bool",
    defaultValue: "0",
    section: "debug",
  },
  {
    key: "LUMEN_QWEN35_TOOL_DEBUG",
    labelKey: "env.entry.LUMEN_QWEN35_TOOL_DEBUG.label",
    helpKey: "env.entry.LUMEN_QWEN35_TOOL_DEBUG.help",
    kind: "bool",
    defaultValue: "0",
    section: "debug",
  },
];

export const ENV_SCHEMA_KEYS = new Set(ENV_SCHEMA.map((e) => e.key));

export type EnvSection = EnvEntry["section"];

export const ENV_SECTIONS: { id: EnvSection; labelKey: string }[] = [
  { id: "thinking", labelKey: "env.section.thinking" },
  { id: "sampling", labelKey: "env.section.sampling" },
  { id: "safety", labelKey: "env.section.safety" },
  { id: "advanced", labelKey: "env.section.advanced" },
  { id: "debug", labelKey: "env.section.debug" },
];

/** Normalize a stored env value to a bool. Matches the server's parsing. */
export function envBoolOn(raw: string | undefined): boolean {
  if (raw == null) return false;
  const v = raw.trim().toLowerCase();
  return v === "1" || v === "on" || v === "true" || v === "yes";
}

/** Encode a bool back into the canonical "1"/"0" string. */
export function encodeBool(b: boolean): string {
  return b ? "1" : "0";
}
