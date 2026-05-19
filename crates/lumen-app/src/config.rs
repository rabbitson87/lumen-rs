use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};

/// Bump whenever the on-disk schema needs a migration step. The migration
/// chain in `migrate_in_place` is keyed by this number — incrementing it
/// without adding a corresponding migration step is a deserialization
/// landmine for anyone with an older config.toml.
pub const CURRENT_SCHEMA_VERSION: u32 = 3;

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
    pub active_model: Option<String>,
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
    /// → `CANDLE_METAL_COMPUTE_PER_BUFFER` (default 10 — empirically +7% tok/s
    /// on M-series for Qwen3.6-27B Dense).
    pub candle_compute_per_buffer: Option<u32>,

    // ── Generation defaults ─────────────────────────────────────────
    /// → `REPEAT_PENALTY` (default 1.0).
    pub repeat_penalty: Option<f32>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum CorsMode {
    Off,
    Localhost,
    All,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantConfig {
    /// → `TQ_BITS` — TurboQuant scalar bits (2/3/4).
    pub bits: u8,
    pub qjl_m: usize,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    pub max: usize,
    pub sliding: usize,
    pub prefill: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedConfig {
    /// → backend selection (`MLX_BACKEND` style). `"auto"` lets the server
    /// pick based on the model id.
    pub backend_mode: BackendMode,

    /// → `LUMEN_SPEC` — speculative decoding strategy.
    pub spec_kind: SpecKind,
    /// → `LUMEN_SPEC_DRAFT_N_MAX` — max draft tokens per step.
    pub spec_draft_n_max: Option<u32>,

    /// → `BATCHED_ENGINE=1` — enables the batched scheduler.
    #[serde(default)]
    pub batched_engine: bool,

    /// → `PAGED_KV=1` (+ `PAGED_LAYERS` / `PAGED_KV_HEADS` / `PAGED_HEAD_DIM_*`
    /// / `PAGED_GLOBAL_EVERY` / `PAGED_MAX_BATCH`). Off by default; Phase 3.
    pub paged_attention: PagedConfig,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum BackendMode {
    Auto,
    Candle,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PagedConfig {
    #[serde(default)]
    pub enabled: bool,
    pub layers: Option<u32>,
    pub kv_heads: Option<u32>,
    pub head_dim_sliding: Option<u32>,
    pub head_dim_global: Option<u32>,
    pub global_every: Option<u32>,
    pub max_batch: Option<u32>,
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
                candle_compute_per_buffer: None,
                repeat_penalty: None,
            },
            quant: QuantConfig {
                bits: 3,
                qjl_m: 64,
                seed: 42,
            },
            context: ContextConfig {
                max: 8192,
                sliding: 1024,
                prefill: 4096,
            },
            advanced: AdvancedConfig {
                backend_mode: BackendMode::Auto,
                spec_kind: SpecKind::Off,
                spec_draft_n_max: None,
                batched_engine: false,
                paged_attention: PagedConfig {
                    enabled: false,
                    layers: None,
                    kv_heads: None,
                    head_dim_sliding: None,
                    head_dim_global: None,
                    global_every: None,
                    max_batch: None,
                },
            },
            env_overrides: BTreeMap::new(),
            active_model: None,
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
