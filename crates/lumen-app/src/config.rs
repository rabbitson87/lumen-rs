use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};

/// Bump whenever the on-disk schema needs a migration step. The migration
/// chain in `migrate_in_place` is keyed by this number — incrementing it
/// without adding a corresponding migration step is a deserialization
/// landmine for anyone with an older config.toml.
// 8 -> 9 with the PagedAttention removal: `[advanced.paged_attention]` is gone
// and its one live field became `advanced.mlx_batch_max`.
//
// Earlier, 7 -> 8 alongside the Candle removal — a cautionary note worth
// keeping. The constant had been left at 7 while `migrate_in_place` already
// carried a `v7 -> v8` step, and migration only runs when `schema_version <
// CURRENT`, so a config sitting at exactly 7 never received that step. Bumping
// the constant is not optional bookkeeping; it is what makes the step run.
pub const CURRENT_SCHEMA_VERSION: u32 = 9;

/// On-disk persistent config. Lives at
/// `~/Library/Application Support/ai.lumen.app/config.toml` on macOS.
///
/// The schema mirrors the env vars that `lumen-server` reads at startup, so
/// every UI control corresponds to exactly one process env var. The mapping
/// lives in `server.rs::build_env`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistentConfig {
    /// Bumped by `migrate_in_place` as the chain is applied. Older configs
    /// that lack this field deserialize as 0 (default), triggering the full
    /// migration chain on first load.
    #[serde(default)]
    pub schema_version: u32,

    pub server: ServerConfig,
    pub quant: QuantConfig,
    pub context: ContextConfig,
    pub advanced: AdvancedConfig,
    /// Free-form `KEY=VALUE` overrides for anything not surfaced as a typed
    /// field. Passed verbatim to the subprocess after the typed fields, so a
    /// user can use this to override even fields that have dedicated UI (last
    /// write wins, with a warning displayed in the UI).
    #[serde(default)]
    pub env_overrides: BTreeMap<String, String>,
    /// Active chat / LLM model id. Independent of `active_image_model` so a
    /// chat model and an image model can be selected at once (→ hybrid serve).
    pub active_model: Option<String>,
    /// Active text-to-image diffusion model id (e.g. `flux2-dev`). Kept in a
    /// separate slot from `active_model` so selecting an image model no longer
    /// overwrites the chat model — with both set, the server launches in
    /// `LUMEN_SERVE=hybrid` (LLM + diffusion co-resident). `#[serde(default)]`
    /// keeps older configs (which lack the field) loadable.
    #[serde(default)]
    pub active_image_model: Option<String>,
    /// Optional override for the `lumen-server` binary path. When `None` the
    /// app searches PATH and (in bundled builds) the sidecar location.
    pub server_binary_path: Option<PathBuf>,
    /// Local directory holding downloaded model weights. Defaults to
    /// `~/Library/Application Support/ai.lumen.app/models`.
    pub models_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub cors: CorsMode,
    pub api_key: Option<String>,

    // ── Metal memory caps (mlx-native only) ──────────────────────────
    /// → `LUMEN_WIRED_LIMIT_GB` — page-locked ceiling (default 28 GB).
    pub wired_limit_gb: Option<usize>,
    /// → `LUMEN_CACHE_LIMIT_GB` — buffer pool cap (default 8 GB).
    pub cache_limit_gb: Option<usize>,
    /// → `LUMEN_MEMORY_LIMIT_GB` — soft total cap (default 32 GB).
    pub memory_limit_gb: Option<usize>,
    /// → `LUMEN_DISABLE_WIRED_LIMIT=1` — skip all three caps (mlx defaults).
    #[serde(default)]
    pub disable_wired_limit: bool,

    // ── Loading / warmup ────────────────────────────────────────────
    /// → `EMBEDDING_MODEL_ID` — optional embedding model spawned alongside.
    pub embedding_model_id: Option<String>,
    /// → `TOKENIZER_ID` — override tokenizer repo (rarely needed).
    pub tokenizer_id: Option<String>,
    /// → `LUMEN_GEMMA4_DIR` / `LUMEN_QWEN35_SHARDS` — local weights path.
    pub local_model_dir: Option<PathBuf>,
    /// → `SKIP_WARMUP=1`
    #[serde(default)]
    pub skip_warmup: bool,

    // ── Disk KV cache (persistent prefix-cache tier) ────────────────────
    /// → `LUMEN_KV_DISK=1` — persist prefix-cache KV snapshots to disk so they
    /// survive server restart / eviction (avoids re-paying cold prefill).
    #[serde(default)]
    pub kv_disk_enabled: bool,
    /// → `LUMEN_KV_DISK_MAX_GB` — max on-disk budget in GB (LRU-evicted).
    /// `0` = unlimited (default).
    #[serde(default)]
    pub kv_disk_max_gb: usize,
    /// → `LUMEN_KV_DISK_TTL_SECS` — drop disk entries idle longer than this.
    /// Default 86400 (1 day); `0` = never expire.
    #[serde(default = "default_kv_disk_ttl_secs")]
    pub kv_disk_ttl_secs: u32,
}

fn default_kv_disk_ttl_secs() -> u32 {
    86_400
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum CorsMode {
    Off,
    Localhost,
    All,
}

/// KV-cache quantization controls. The card drives the simple 4/8-bit
/// affine quantization path (`LUMEN_GEMMA4_QUANT_KV*` env vars). The
/// TurboQuant Stage-1/2 lever has been retired from the user-facing
/// surface — empirical sweeps on Apple Silicon (2026-05-26) showed it is
/// net-negative across all tested context sizes (4K → 32K, gap widens
/// from −9% to −20%). The TQ kernel/primitive code remains in-tree behind
/// dev-only env vars so future CUDA / PagedAttention work can re-evaluate
/// it. See `CLAUDE.md` Phase 2/3.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantConfig {
    /// KV-cache quantization bits (3 / 4 / 6 / 8). Default 4 (gives 4×
    /// compression). 8-bit gives 2× compression with closer-to-bf16
    /// quality. Emitted as `LUMEN_GEMMA4_QUANT_KV_BITS` (also kept as
    /// `TQ_BITS` for legacy Candle backends still on the
    /// turboquant-cache crate path).
    pub bits: u8,

    /// Three-way KV quantization mode.
    ///   * `Off`  — never quantize; KV stays in bf16 (fastest, most memory).
    ///   * `On`   — always quantize at `bits`; saves memory on every request.
    ///   * `Auto` — quantize only when a request's `prompt_tokens` reaches
    ///              `kv_auto_threshold_tokens`. Default — gives short chats
    ///              full bf16 speed and only pays the per-step quant
    ///              dispatch cost when the memory matters.
    ///
    /// Emits `LUMEN_GEMMA4_QUANT_KV_MODE`.
    #[serde(default = "default_kv_mode")]
    pub kv_mode: QuantKvMode,

    /// Prompt-length threshold (in tokens) at which `QuantKvMode::Auto`
    /// switches quantization ON for a request. Default 16384 (16K) — tuned
    /// for the 24 GB Mac mini target where bf16 KV pressure starts to bind;
    /// quantized-KV sliding-window wins are verified from ~16K up. A 128K
    /// threshold made Auto a near no-op (most prompts never crossed it).
    /// Big-memory machines (64 GB+) can raise this or use `Off`. Ignored
    /// unless `kv_mode = Auto`. Emits `LUMEN_GEMMA4_QUANT_KV_AUTO_THRESHOLD_TOKENS`.
    #[serde(default = "default_kv_auto_threshold_tokens")]
    pub kv_auto_threshold_tokens: u32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum QuantKvMode {
    /// Never quantize the KV cache. bf16 everywhere. Best for normal
    /// usage on machines with ample unified memory (M3 Max 36+ GB at
    /// ≤64K context bf16 KV ≈ 16 GB — no memory pressure).
    Off,
    /// Always quantize the KV cache at `bits`. Saves memory on every
    /// request at a small (<5%) decode cost.
    On,
    /// Quantize only when a request's `prompt_tokens` meets or exceeds
    /// `kv_auto_threshold_tokens`. Default — short chats stay at full
    /// bf16 speed; only long-context requests pay the quant trade-off
    /// in exchange for keeping memory bounded.
    Auto,
}

fn default_kv_mode() -> QuantKvMode {
    QuantKvMode::Off
}

fn default_kv_auto_threshold_tokens() -> u32 {
    16384
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    pub max: usize,
    pub sliding: usize,
    pub prefill: usize,
    /// Default `max_tokens` (generation budget) applied to chat / completion
    /// requests that omit the field. Emitted as `LUMEN_DEFAULT_MAX_TOKENS`;
    /// `lumen-server` reads it in `types.rs::default_max_tokens()`. `0` means
    /// "unbounded — generate until EOS / stop / context budget".
    #[serde(default = "default_default_max_tokens")]
    pub default_max_tokens: u32,
}

fn default_default_max_tokens() -> u32 {
    2048
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedConfig {
    /// → backend selection (`MLX_BACKEND` style). `"auto"` lets the server
    /// pick based on the model id.
    pub backend_mode: BackendMode,

    /// → `LUMEN_SPEC` — speculative decoding strategy.
    pub spec_kind: SpecKind,
    /// → `LUMEN_SPEC_K` — draft tokens per step.
    pub spec_draft_n_max: Option<u32>,

    /// → `LUMEN_MLX_BATCH_DECODE=1` — enables the MLX continuous-batching
    /// scheduler. Note it only admits greedy requests: anything with a
    /// non-zero temperature, tools, `response_format` or stop strings is
    /// routed to the sequential path regardless of this setting.
    #[serde(default)]
    pub batched_engine: bool,

    /// → `LUMEN_MLX_BATCH_MAX` — how many sequences the MLX scheduler above
    /// decodes in one step. `None` leaves the server's own default (8).
    ///
    /// Lived under `[advanced.paged_attention]` as `max_batch` until v9, which
    /// was always a misnomer: nothing paged ever read it, and the MLX scheduler
    /// always did.
    #[serde(default)]
    pub mlx_batch_max: Option<u32>,

    /// Retired in v9 along with the PagedAttention crate. Kept only so a config
    /// written by an older build still parses its `max_batch` — the v9
    /// migration lifts that value into `mlx_batch_max` and this is never
    /// written back (`skip_serializing`). Delete once no v8 configs remain in
    /// the wild. Same reasoning as the `candle` alias on [`BackendMode`].
    #[serde(default, skip_serializing)]
    paged_attention: Option<LegacyPagedConfig>,
}

/// v8 shape of `[advanced.paged_attention]`, read once by the v9 migration.
/// Every field except `max_batch` drove env vars that no code has read since
/// the Candle backend was removed.
#[derive(Debug, Clone, Deserialize)]
struct LegacyPagedConfig {
    #[serde(default)]
    max_batch: Option<u32>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum BackendMode {
    /// `candle` is aliased here rather than removed outright: a config saved
    /// before the Candle backend was deleted still says `backend-mode =
    /// "candle"`, and `toml::from_str` hard-errors on an unknown variant —
    /// which happens *before* `migrate_in_place` can rewrite anything. The
    /// alias lets those configs load, and the v8 migration re-saves them.
    #[serde(alias = "candle")]
    Auto,
    MlxNative,
    MlxPyo3,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SpecKind {
    Off,
    Lookup,
    Mtp,
}

impl Default for PersistentConfig {
    fn default() -> Self {
        // Use `~/models/` as default — matches the README convention for
        // `huggingface-cli download --local-dir ~/models/<name>` and the
        // existing developer workflow. Falls back to `./models` only if we
        // can't resolve HOME (extremely unlikely on macOS).
        let models_dir = default_models_dir();
        // Probe RAM and seed Metal memory caps to a sensible fraction of
        // total system memory (see `sysinfo::ram_defaults`). The caps are
        // Option<usize> so the user can still clear them — None means
        // "use lumen-server's built-in fallback".
        let mem = crate::sysinfo::probe().recommended;
        Self {
            schema_version: CURRENT_SCHEMA_VERSION,
            server: ServerConfig {
                host: "127.0.0.1".into(),
                // 41110 = lumen-server's new DEFAULT_PORT (away from 8080 collisions).
                port: 41110,
                cors: CorsMode::Localhost,
                api_key: None,
                wired_limit_gb: Some(mem.wired_limit_gb),
                cache_limit_gb: Some(mem.cache_limit_gb),
                memory_limit_gb: Some(mem.memory_limit_gb),
                disable_wired_limit: false,
                embedding_model_id: None,
                tokenizer_id: None,
                local_model_dir: None,
                skip_warmup: false,
                kv_disk_enabled: false,
                kv_disk_max_gb: 0,
                kv_disk_ttl_secs: 86_400,
            },
            quant: QuantConfig {
                bits: 4,
                kv_mode: QuantKvMode::Off,
                kv_auto_threshold_tokens: 16384,
            },
            context: ContextConfig {
                // Bumped 10× over the initial 8192 / 4096 defaults at
                // schema v4. Both Gemma 4 (128K native cap) and
                // Qwen 3.6 (256K with YaRN) comfortably handle ~80K
                // when TurboQuant is ON; the UI's "realistic ctx"
                // hint still clamps to the host's actual RAM budget.
                max: 81920,
                sliding: 1024,
                prefill: 40960,
                default_max_tokens: 2048,
            },
            advanced: AdvancedConfig {
                backend_mode: BackendMode::Auto,
                spec_kind: SpecKind::Off,
                spec_draft_n_max: None,
                batched_engine: false,
                mlx_batch_max: None,
                paged_attention: None,
            },
            env_overrides: BTreeMap::new(),
            active_model: None,
            active_image_model: None,
            server_binary_path: None,
            models_dir,
        }
    }
}

impl PersistentConfig {
    pub fn load_or_default() -> Result<Self> {
        let path = config_path()?;
        if !path.exists() {
            let cfg = Self::default();
            cfg.save()?;
            return Ok(cfg);
        }
        let raw =
            std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        let mut cfg: Self =
            toml::from_str(&raw).with_context(|| format!("parse {}", path.display()))?;
        if cfg.schema_version < CURRENT_SCHEMA_VERSION {
            let backup = path.with_extension("toml.bak");
            std::fs::copy(&path, &backup).with_context(|| {
                format!(
                    "backup {} → {} before migration",
                    path.display(),
                    backup.display()
                )
            })?;
            let from = cfg.schema_version;
            migrate_in_place(&mut cfg);
            eprintln!(
                "[config] migrated schema_version {} → {} (backup at {})",
                from,
                cfg.schema_version,
                backup.display()
            );
            cfg.save()?;
        }
        Ok(cfg)
    }

    pub fn save(&self) -> Result<()> {
        let path = config_path()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let raw = toml::to_string_pretty(self).context("serialize config")?;
        std::fs::write(&path, raw).with_context(|| format!("write {}", path.display()))?;
        Ok(())
    }
}

/// Forward-only schema migrations. Each step mutates `cfg` in place and bumps
/// `schema_version`. Steps must be idempotent so that re-running the chain on
/// a partially-migrated file is safe.
///
/// Add a new step when bumping `CURRENT_SCHEMA_VERSION`:
///
/// ```ignore
/// while cfg.schema_version < 2 {
///     // v1 -> v2: example — split server.api_key into api_keys: Vec<String>
///     cfg.schema_version = 2;
/// }
/// ```
fn migrate_in_place(cfg: &mut PersistentConfig) {
    // v0 -> v1: initial schema. New fields added in v1 are covered by
    // `#[serde(default)]` so the deserializer already filled them. We just
    // stamp the version so future loads don't re-trigger the migration path.
    if cfg.schema_version < 1 {
        cfg.schema_version = 1;
    }
    // v1 -> v2: DEFAULT_PORT moved from 8080 -> 41110 to avoid collisions
    // with common dev servers. Only rewrite if the saved port matches the
    // *old default* — if the user explicitly chose 8080, we don't touch it.
    if cfg.schema_version < 2 {
        if cfg.server.port == 8080 {
            cfg.server.port = 41110;
        }
        cfg.schema_version = 2;
    }
    // v2 -> v3: default models_dir moved from Application Support to ~/models/
    // (matches the README's `huggingface-cli download --local-dir` convention).
    // Only rewrite if the saved path matches the old default — user explicit
    // choices (e.g. external SSD) are preserved.
    if cfg.schema_version < 3 {
        if let Some(old) = legacy_app_support_models_dir() {
            if cfg.models_dir == old {
                cfg.models_dir = default_models_dir();
            }
        }
        cfg.schema_version = 3;
    }
    // v3 -> v4: context.max and context.prefill defaults bumped 10× to
    // 81920 / 40960 (TurboQuant + modern model native caps make the old
    // 8192 / 4096 a needlessly tight budget). Only rewrite when the
    // saved value matches the *old default exactly* — any explicit
    // user choice (smaller or larger) is preserved.
    if cfg.schema_version < 4 {
        if cfg.context.max == 8192 {
            cfg.context.max = 81920;
        }
        if cfg.context.prefill == 4096 {
            cfg.context.prefill = 40960;
        }
        cfg.schema_version = 4;
    }
    // v4 -> v5: dropped four debug-tab knobs that were either dead-path
    // (`candle_compute_per_buffer` — Candle backend, replaced by mlx-native
    // default) or misplaced (`repeat_penalty` is a per-request generation
    // param, not a debug switch), plus the TurboQuant `qjl_m` / `seed`
    // ablation knobs that belonged in benchmarks rather than UI. Existing
    // serialized values are silently dropped by serde on parse; this step
    // exists purely to stamp the version.
    if cfg.schema_version < 5 {
        cfg.schema_version = 5;
    }
    // v5 -> v6: introduced `quant.turboquant_mode` (Off / On / Auto). The
    // TurboQuant fields were retired in v8 — see below — so this step is
    // now a no-op apart from the version stamp.
    if cfg.schema_version < 6 {
        cfg.schema_version = 6;
    }
    // v6 -> v7: introduced `context.default_max_tokens` so the OpenAI-compat
    // server-side default for chat/completion `max_tokens` (when the client
    // omits the field) is user-configurable instead of hardcoded to 2048.
    // Existing configs that serialized without the field deserialize as 0
    // (serde's u32 default) — that would silently mean "unbounded" and is a
    // behaviour change from v6. Stamp the legacy 2048 default explicitly so
    // upgraders see no surprise.
    if cfg.schema_version < 7 {
        if cfg.context.default_max_tokens == 0 {
            cfg.context.default_max_tokens = 2048;
        }
        cfg.schema_version = 7;
    }
    // v7 -> v8: retired the TurboQuant UI surface. The `QuantConfig` schema
    // now holds simple-Q4 controls (`kv_mode`, `kv_auto_threshold_tokens`,
    // `bits`) instead of the TQ Stage-1/2 fields. Old fields
    // (`turboquant_enabled`, `turboquant_qjl_enabled`, `turboquant_mode`,
    // `turboquant_auto_threshold_tokens`) are dropped at deserialization
    // — serde silently ignores unknown fields. New fields land at their
    // safe defaults via `#[serde(default)]`. The TQ kernel/primitive
    // code path stays in-tree behind dev-only env vars; only the user-
    // facing config surface is reset. See `CLAUDE.md` Phase 2/3 plan.
    if cfg.schema_version < 8 {
        // Force the safe new defaults — older configs deserialized into
        // the new struct with serde defaults already, so this is a stamp
        // operation only. (If a v6 user opted into Auto + 4K threshold we
        // honour their intent indirectly: they opt back into Auto via UI
        // once they read the new help text. Past TQ measurements showed
        // Auto-at-4K was usually a perf regression anyway.)
        cfg.quant.kv_mode = QuantKvMode::Off;
        cfg.quant.kv_auto_threshold_tokens = 16384;
        if !(matches!(cfg.quant.bits, 3 | 4 | 6 | 8)) {
            cfg.quant.bits = 4;
        }
        cfg.schema_version = 8;
    }
    // v8 -> v9: PagedAttention was deleted, so `[advanced.paged_attention]`
    // went with it. Six of its seven fields drove env vars that nothing had
    // read since the Candle backend was removed; the seventh, `max_batch`, was
    // never a paged setting at all — it is the MLX scheduler's batch width, and
    // is lifted here into `mlx_batch_max` rather than dropped. `take()` clears
    // the shim so it is not carried forward once this has run.
    if cfg.schema_version < 9 {
        if let Some(legacy) = cfg.advanced.paged_attention.take() {
            if cfg.advanced.mlx_batch_max.is_none() {
                cfg.advanced.mlx_batch_max = legacy.max_batch;
            }
        }
        cfg.schema_version = 9;
    }
    // Future migrations append here.
}

pub fn project_dirs() -> Option<ProjectDirs> {
    ProjectDirs::from("ai", "lumen", "app")
}

/// Default location for weight downloads. `~/models/<repo-name>` matches the
/// README's `huggingface-cli download --local-dir ~/models/<name>` convention
/// and the existing dev workflow.
pub fn default_models_dir() -> PathBuf {
    directories::UserDirs::new()
        .and_then(|d| d.home_dir().to_path_buf().into())
        .map(|home: PathBuf| home.join("models"))
        .unwrap_or_else(|| PathBuf::from("./models"))
}

/// The old default that lived inside `~/Library/Application Support/ai.lumen.app/`.
/// Kept so the v2→v3 migration can detect "user never customised, was on old
/// default" and rewrite to `~/models/`.
fn legacy_app_support_models_dir() -> Option<PathBuf> {
    project_dirs().map(|d| d.data_dir().join("models"))
}

pub fn config_path() -> Result<PathBuf> {
    let dirs = project_dirs().context("locate application support dir")?;
    Ok(dirs.config_dir().join("config.toml"))
}

pub fn config_dir() -> Result<PathBuf> {
    let dirs = project_dirs().context("locate application support dir")?;
    Ok(dirs.config_dir().to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A v8 config, as the app wrote them before PagedAttention was removed.
    /// `max_batch` is the only field here that anything ever read.
    const V8_TOML: &str = r#"
schema_version = 8
models_dir = "/tmp/models"

[server]
host = "127.0.0.1"
port = 41110
cors = "localhost"

[quant]
kv_mode = "off"
kv_auto_threshold_tokens = 16384
bits = 4

[context]
max = 81920
sliding = 1024
prefill = 40960
default_max_tokens = 2048

[advanced]
backend_mode = "auto"
spec_kind = "off"
batched_engine = true

[advanced.paged_attention]
enabled = true
layers = 32
kv_heads = 4
head_dim_sliding = 128
head_dim_global = 256
global_every = 4
max_batch = 6

[env_overrides]
"#;

    fn migrated_v8() -> PersistentConfig {
        let mut cfg: PersistentConfig =
            toml::from_str(V8_TOML).expect("a v8 config must still deserialize");
        migrate_in_place(&mut cfg);
        cfg
    }

    /// The whole point of the shim: a batch width the user chose must survive
    /// the rename instead of being silently reset to the server default.
    #[test]
    fn v9_migration_carries_max_batch_into_mlx_batch_max() {
        let cfg = migrated_v8();
        assert_eq!(cfg.advanced.mlx_batch_max, Some(6));
        assert_eq!(cfg.schema_version, 9);
    }

    /// The six retired fields drove env vars nothing reads. They must be
    /// tolerated on the way in — serde has no `deny_unknown_fields` here, so
    /// this guards against someone adding it and breaking every old config.
    #[test]
    fn v9_migration_ignores_the_retired_paged_fields() {
        let cfg = migrated_v8();
        assert!(cfg.advanced.batched_engine, "unrelated fields survive");
        let out = toml::to_string_pretty(&cfg).expect("serialize");
        assert!(
            !out.contains("paged"),
            "the legacy table must not be written back out:\n{out}"
        );
    }

    /// `migrate_in_place` promises idempotence, and this step reads a field it
    /// then clears — exactly the shape that breaks when run twice.
    #[test]
    fn v9_migration_is_idempotent() {
        let mut cfg = migrated_v8();
        let once = cfg.advanced.mlx_batch_max;
        migrate_in_place(&mut cfg);
        assert_eq!(cfg.advanced.mlx_batch_max, once);
        assert_eq!(cfg.schema_version, 9);
    }

    /// A config already on v9 has no legacy table; the step must not clobber a
    /// value the user set under the new name.
    #[test]
    fn v9_migration_does_not_overwrite_an_existing_mlx_batch_max() {
        let mut cfg = PersistentConfig::default();
        cfg.schema_version = 8;
        cfg.advanced.mlx_batch_max = Some(2);
        cfg.advanced.paged_attention = Some(LegacyPagedConfig { max_batch: Some(6) });
        migrate_in_place(&mut cfg);
        assert_eq!(cfg.advanced.mlx_batch_max, Some(2));
    }

    /// Guards the 7 -> 8 class of bug: a migration step that exists but is
    /// unreachable because the constant was never bumped past it.
    #[test]
    fn current_schema_version_reaches_the_last_migration_step() {
        let mut cfg = PersistentConfig::default();
        cfg.schema_version = 0;
        migrate_in_place(&mut cfg);
        assert_eq!(
            cfg.schema_version, CURRENT_SCHEMA_VERSION,
            "migrate_in_place must land exactly on CURRENT_SCHEMA_VERSION"
        );
    }
}
