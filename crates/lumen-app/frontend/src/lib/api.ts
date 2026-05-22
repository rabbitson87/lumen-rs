import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

// ── Types — mirror src/config.rs ─────────────────────────────────────

export type CorsMode = "off" | "localhost" | "all";

export interface ServerConfig {
  host: string;
  port: number;
  cors: CorsMode;
  api_key: string | null;
  wired_limit_gb: number | null;
  cache_limit_gb: number | null;
  memory_limit_gb: number | null;
  disable_wired_limit: boolean;
  embedding_model_id: string | null;
  tokenizer_id: string | null;
  local_model_dir: string | null;
  skip_warmup: boolean;
}

export interface QuantConfig {
  /** TurboQuant Lloyd-Max bits (2/3/4). Drives both legacy `TQ_BITS` and
   *  Gemma 4 native `LUMEN_GEMMA4_QUANT_KV_SLIDING_TURBOQUANT_BITS`. */
  bits: number;
  /** Legacy master switch for TurboQuant Stage 1 (v5 and earlier). v6+
   *  mirrors it from `turboquant_mode` so external tools still see a
   *  meaningful boolean; UI controls only `turboquant_mode`. */
  turboquant_enabled: boolean;
  /** QJL 1-bit residual correction layer (Stage 2). Only takes effect on
   *  requests where the per-request TQ resolution is ON. */
  turboquant_qjl_enabled: boolean;
  /** Three-way TurboQuant control (v6+).
   *   - `off`  — never apply TQ; sliding cache stays bf16 (fastest short-
   *              prompt decode, no memory savings).
   *   - `on`   — always apply TQ (max KV memory savings, ~20-56% slower).
   *   - `auto` — apply only when `prompt_tokens >= turboquant_auto_threshold_tokens`.
   *              Default. Short chats get bf16 decode speed; long context
   *              gets the memory savings. */
  turboquant_mode: "off" | "on" | "auto";
  /** Prompt-length threshold (tokens) at which `auto` mode flips TQ ON.
   *  Default 4096. Ignored unless mode = auto. */
  turboquant_auto_threshold_tokens: number;
}

export interface ContextConfig {
  max: number;
  sliding: number;
  prefill: number;
  /** Default `max_tokens` (generation budget) for OpenAI-compatible chat /
   *  completion requests that omit the field. Default 2048. Emitted as the
   *  `LUMEN_DEFAULT_MAX_TOKENS` env var; explicit `max_tokens` in the request
   *  body always wins. */
  default_max_tokens: number;
}

export type BackendMode = "auto" | "candle" | "mlx-native" | "mlx-pyo3";
export type SpecKind = "off" | "lookup" | "mtp";

export interface PagedConfig {
  enabled: boolean;
  layers: number | null;
  kv_heads: number | null;
  head_dim_sliding: number | null;
  head_dim_global: number | null;
  global_every: number | null;
  max_batch: number | null;
}

export interface AdvancedConfig {
  backend_mode: BackendMode;
  spec_kind: SpecKind;
  spec_draft_n_max: number | null;
  batched_engine: boolean;
  paged_attention: PagedConfig;
}

export interface PersistentConfig {
  server: ServerConfig;
  quant: QuantConfig;
  context: ContextConfig;
  advanced: AdvancedConfig;
  env_overrides: Record<string, string>;
  active_model: string | null;
  server_binary_path: string | null;
  models_dir: string;
}

export interface ModelEntry {
  id: string;
  path: string;
  size_bytes: number;
  ready: boolean;
  supported: boolean;
  label: string | null;
  /** HF Hub commit SHA recorded at download time. `null` for legacy installs
   * (pre-v0.1.3) and for local-only models. Used together with `UpdateStatus`
   * to detect "same id, new weights" rebuilds. */
  local_sha: string | null;
}

/** Result of an HF Hub revision check for one installed model. `needs_update`
 * is true iff both `local_sha` and `remote_sha` are known and differ.
 * Drives the "Update available" badge + Start-button gating in App.svelte. */
export interface UpdateStatus {
  repo_id: string;
  local_sha: string | null;
  remote_sha: string | null;
  needs_update: boolean;
}

export type ModelFamily = "qwen25" | "qwen35-dense" | "qwen35-moe" | "gemma4";

export interface FamilyInfo {
  family: ModelFamily;
  label: string;
  backend: string;
  notes: string;
}

export interface RecommendedModel {
  id: string;
  family: ModelFamily;
  label: string;
  approx_size_gb: number;
  min_ram_gb: number;
  notes: string;
}

export interface RecommendedEmbedding {
  id: string;
  label: string;
  approx_size_gb: number;
  min_ram_gb: number;
  notes: string;
}

export interface Catalog {
  families: FamilyInfo[];
  recommended: RecommendedModel[];
  embeddings: RecommendedEmbedding[];
}

export interface MemoryDefaults {
  wired_limit_gb: number;
  cache_limit_gb: number;
  memory_limit_gb: number;
}

export interface SystemInfo {
  ram_gb: number;
  arch: string;
  recommended: MemoryDefaults;
}

export interface MemoryUsage {
  used_bytes: number;
  total_bytes: number;
}

export type LifecycleState = "stopped" | "starting" | "running" | "stopping" | "crashed";

export interface ServerStatus {
  state: LifecycleState;
  pid: number | null;
  port: number;
  host: string;
  model_id: string | null;
  uptime_secs: number | null;
  last_error: string | null;
}

export interface ServerMetrics {
  tokens_per_sec: number | null;
  ms_per_step: number | null;
  kv_cache_mb: number | null;
  requests_per_min: number | null;
}

export interface LogLine {
  stream: "stdout" | "stderr";
  line: string;
  ts_unix_ms: number;
}

export interface DownloadProgress {
  repo_id: string;
  file: string;
  downloaded_bytes: number;
  total_bytes: number | null;
  done: boolean;
}

export type CheckStatus = "pass" | "warn" | "fail";
export type OverallHealth = "healthy" | "degraded" | "blocked" | "unknown";

export interface CheckResult {
  id: string;
  name: string;
  status: CheckStatus;
  message: string;
  detail: string | null;
  fix_hint: string | null;
  fix_command: string | null;
  fix_action: string | null;
}

export interface DoctorReport {
  overall: OverallHealth;
  checks: CheckResult[];
}

export interface UpdateInfo {
  available: boolean;
  current_version: string;
  latest_version: string | null;
  release_notes: string | null;
  published_at: string | null;
}

export interface UpdateProgress {
  downloaded_bytes: number;
  total_bytes: number | null;
  done: boolean;
}

// ── Commands ─────────────────────────────────────────────────────────

export const api = {
  getConfig: () => invoke<PersistentConfig>("get_config"),
  updateServerConfig: (server: ServerConfig) =>
    invoke<PersistentConfig>("update_server_config", { server }),
  updateQuantConfig: (quant: QuantConfig) =>
    invoke<PersistentConfig>("update_quant_config", { quant }),
  updateContextConfig: (context: ContextConfig) =>
    invoke<PersistentConfig>("update_context_config", { context }),
  updateAdvancedConfig: (advanced: AdvancedConfig) =>
    invoke<PersistentConfig>("update_advanced_config", { advanced }),
  updateEnvOverrides: (env_overrides: Record<string, string>) =>
    invoke<PersistentConfig>("update_env_overrides", { envOverrides: env_overrides }),
  typedEnvKeys: () => invoke<string[]>("typed_env_keys"),

  listModels: () => invoke<ModelEntry[]>("list_models"),
  getCatalog: () => invoke<Catalog>("get_catalog"),
  refreshCatalog: () => invoke<Catalog>("refresh_catalog"),
  setActiveModel: (model_id: string) =>
    invoke<PersistentConfig>("set_active_model", { modelId: model_id }),
  deleteModel: (model_id: string) => invoke<void>("delete_model", { modelId: model_id }),
  /** Query HF Hub for the current commit SHA of each provided repo id and
   * compare against the locally-recorded SHA. Side-effect: backend updates
   * its outdated-set so `startServer` will refuse to launch a stale model. */
  checkModelUpdates: (repo_ids: string[]) =>
    invoke<UpdateStatus[]>("check_model_updates", { repoIds: repo_ids }),
  downloadModel: (repo_id: string, files: string[] | null = null) =>
    invoke<void>("download_model", { repoId: repo_id, files }),

  startServer: () => invoke<ServerStatus>("start_server"),
  stopServer: () => invoke<ServerStatus>("stop_server"),
  serverStatus: () => invoke<ServerStatus>("server_status"),
  serverMetrics: () => invoke<ServerMetrics>("server_metrics"),

  openConfigDir: () => invoke<string>("open_config_dir"),

  getSystemInfo: () => invoke<SystemInfo>("get_system_info"),
  getMemoryUsage: () => invoke<MemoryUsage | null>("get_memory_usage"),
  resetMemoryCaps: () => invoke<PersistentConfig>("reset_memory_caps"),

  doctorRun: () => invoke<DoctorReport>("doctor_run"),
  doctorFix: (action: string) => invoke<string>("doctor_fix", { action }),

  currentVersion: () => invoke<string>("current_version"),
  checkForUpdates: () => invoke<UpdateInfo>("check_for_updates"),
  installUpdate: () => invoke<void>("install_update"),
};

export function onUpdateProgress(cb: (p: UpdateProgress) => void): Promise<UnlistenFn> {
  return listen<UpdateProgress>("lumen://update-progress", (e) => cb(e.payload));
}

// ── Events ───────────────────────────────────────────────────────────

export function onLog(cb: (l: LogLine) => void): Promise<UnlistenFn> {
  return listen<LogLine>("lumen://log", (e) => cb(e.payload));
}

export function onStatus(cb: (s: ServerStatus) => void): Promise<UnlistenFn> {
  return listen<ServerStatus>("lumen://status", (e) => cb(e.payload));
}

export function onDownload(cb: (p: DownloadProgress) => void): Promise<UnlistenFn> {
  return listen<DownloadProgress>("lumen://download", (e) => cb(e.payload));
}
