use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};

/// Bump whenever the on-disk schema needs a migration step. The migration
/// chain in `migrate_in_place` is keyed by this number — incrementing it
/// without adding a corresponding migration step is a deserialization
/// landmine for anyone with an older config.toml.
// 11 -> 12 with the disk KV tier going on by default (10 GB, entries idle 30 days expire): a
// config still holding the old defaults as a whole takes the new ones (see `migrate_in_place`).
//
// 10 -> 11 with the Flash-Next draft head's default flipping from `auto` to `off`: a stored
// `auto` meant "the default", which is now `off`, so the migration rewrites it (see
// `migrate_in_place`).
//
// 8 -> 9 with the PagedAttention removal: `[advanced.paged_attention]` is gone
// and its one live field became `advanced.mlx_batch_max`.
//
// Earlier, 7 -> 8 alongside the Candle removal — a cautionary note worth
// keeping. The constant had been left at 7 while `migrate_in_place` already
// carried a `v7 -> v8` step, and migration only runs when `schema_version <
// CURRENT`, so a config sitting at exactly 7 never received that step. Bumping
// the constant is not optional bookkeeping; it is what makes the step run.
pub const CURRENT_SCHEMA_VERSION: u32 = 12;

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
    /// Settings only the Flash-Next model (the SSD-streamed MoE) reads. `#[serde(default)]`
    /// keeps configs written before the section existed loadable, so no schema bump.
    #[serde(default)]
    pub flash_next: FlashNextConfig,
    /// What this Mac measured of itself (memory bandwidth, SSD read speed), which the decode
    /// speed estimate needs. `#[serde(default)]`: older configs have none, so nothing is
    /// estimated until the first measurement.
    #[serde(default)]
    pub measured_hardware: MeasuredHardware,
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
    /// → `LUMEN_GEMMA4_DIR` — where Gemma 4 weights live when `MODEL_ID` is
    /// a Hub id the app has not downloaded. Every other model loads from
    /// `MODEL_ID` alone, which the app already sets to the local path.
    pub local_model_dir: Option<PathBuf>,
    /// → `SKIP_WARMUP=1`
    #[serde(default)]
    pub skip_warmup: bool,

    // ── Disk KV cache (persistent prefix-cache tier) ────────────────────
    /// → `LUMEN_KV_DISK` (`1` / `0`) — persist prefix-cache KV snapshots to
    /// disk so they survive server restart / eviction (avoids re-paying cold
    /// prefill). Default on, as in the server, which also stops writing while
    /// the store's volume has under 10 GB free (`LUMEN_KV_DISK_MIN_FREE_GB`).
    #[serde(default = "default_kv_disk_enabled")]
    pub kv_disk_enabled: bool,
    /// → `LUMEN_KV_DISK_MAX_GB` — max on-disk budget in GB (LRU-evicted).
    /// Default 10; `0` = unlimited.
    #[serde(default = "default_kv_disk_max_gb")]
    pub kv_disk_max_gb: usize,
    /// → `LUMEN_KV_DISK_TTL_SECS` — drop disk entries idle longer than this.
    /// Default 2592000 (30 days); `0` = never expire.
    #[serde(default = "default_kv_disk_ttl_secs")]
    pub kv_disk_ttl_secs: u32,

    // ── Launch behaviour ─────────────────────────────────────────────
    /// Start the server as soon as the app opens (what makes "reboot → app
    /// back at login → server serving" work). Whether the app itself opens
    /// at login is the OS login item, not stored here. `#[serde(default)]`
    /// keeps older configs loadable, so no schema bump.
    #[serde(default)]
    pub auto_start_server: bool,
}

/// The disk tier's defaults, the same as the server's own.
const KV_DISK_DEFAULT_MAX_GB: usize = 10;
const KV_DISK_DEFAULT_TTL_SECS: u32 = 30 * 24 * 60 * 60;

fn default_kv_disk_enabled() -> bool {
    true
}

fn default_kv_disk_max_gb() -> usize {
    KV_DISK_DEFAULT_MAX_GB
}

fn default_kv_disk_ttl_secs() -> u32 {
    KV_DISK_DEFAULT_TTL_SECS
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
    /// quality. Emitted as `LUMEN_GEMMA4_QUANT_KV_BITS`.
    pub bits: u8,

    /// Three-way KV quantization mode.
    ///   * `Off`  — never quantize; KV stays in bf16 (fastest, most memory).
    ///   * `On`   — always quantize at `bits`; saves memory on every request.
    ///   * `Auto` — quantize only when a request's `prompt_tokens` reaches
    ///     `kv_auto_threshold_tokens`. Default — gives short chats
    ///     full bf16 speed and only pays the per-step quant
    ///     dispatch cost when the memory matters.
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
    ///
    /// `mlx-pyo3` is aliased for the same reason: the PyO3 runner it selected
    /// was removed, and a config that chose it would otherwise fail to load.
    /// It lands on `Auto`, which the server serves on the native runner, and
    /// the next save writes `auto`.
    #[serde(alias = "candle", alias = "mlx-pyo3")]
    Auto,
    MlxNative,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SpecKind {
    Off,
    Lookup,
    Mtp,
}

/// Flash-Next settings: the expert pool, the longest prompt, and how the experts are read from
/// the SSD. Every default equals the server's own, so a field left at it (`None` / `Auto`) sends
/// nothing and the server decides; only a value the user chose reaches the server's environment
/// (see [`FlashNextConfig::env`]), and only when the model being launched is Flash-Next.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FlashNextConfig {
    /// → `LUMEN_MOE_EXPERT_CACHE_GB`, a plain number of GB. `None` = auto: the server sizes the
    /// pool from what the GPU's memory leaves.
    pub expert_cache_gb: Option<f64>,
    /// → `LUMEN_MAX_PROMPT_TOKENS`, the longest prompt the server plans its memory for. `None` =
    /// the machine's default, which the app works out and sends itself (see `server.rs`).
    pub max_prompt_tokens: Option<u32>,
    /// → `LUMEN_QWEN35_PREFILL_CHUNK`, tokens per prefill step. `None` = the machine's default.
    pub prefill_chunk: Option<u32>,
    /// → `LUMEN_MOE_PACK`.
    pub pack: FlashNextPack,
    /// → `LUMEN_QWEN4_MTP`.
    pub draft_head: FlashNextDraftHead,
    /// → `LUMEN_MOE_IO_MODE`.
    pub io_mode: FlashNextIoMode,
    /// → `LUMEN_MOE_IO_LANES`, reader threads (1 to 64). `None` = the server's 32.
    pub io_lanes: Option<u32>,
}

/// The expert pack: one contiguous file of the routed experts, built once from the weights.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FlashNextPack {
    /// Read the pack when it is complete.
    #[default]
    Auto,
    /// Never read it.
    Off,
    /// Build it when the server starts and it is missing.
    Build,
}

/// The draft head (`mtp.safetensors`) that lets the server guess a second token per step. Off
/// by default: measured, it gained at most 4% and often lost 1-4%, and its text is not
/// bit-identical to plain decoding.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FlashNextDraftHead {
    /// Never use it. The server's own default, so nothing is sent.
    #[default]
    Off,
    /// Use it when the file is installed, under the server's speed governor.
    Auto,
    /// Like `Auto`, but the server refuses to start without the file.
    On,
}

/// How expert reads go to the SSD.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FlashNextIoMode {
    /// The page cache while the Mac has memory to spare, direct reads otherwise.
    #[default]
    Auto,
    Direct,
    Buffered,
}

/// The server's own number of reader threads.
pub const FLASH_NEXT_IO_LANES_DEFAULT: u32 = 32;
/// The most reader threads the server accepts.
const FLASH_NEXT_IO_LANES_MAX: u32 = 64;

impl FlashNextConfig {
    /// The environment variables the typed fields stand for: only those set to something other
    /// than the server's own default, with the values as the server reads them (a plain number of
    /// GB, not `8g`). A value that could not be one (not a positive number, threads outside 1 to
    /// 64) counts as unset. The launch sends these to a Flash-Next server and to no other model.
    pub fn env(&self) -> Vec<(&'static str, String)> {
        let mut out = Vec::new();
        if let Some(gb) = self.expert_cache_gb.filter(|g| g.is_finite() && *g > 0.0) {
            // Two decimals, as the app's pickers write them; `{}` prints `8`, not `8.0`.
            out.push((
                "LUMEN_MOE_EXPERT_CACHE_GB",
                format!("{}", (gb * 100.0).round() / 100.0),
            ));
        }
        if let Some(n) = self.max_prompt_tokens.filter(|n| *n > 0) {
            out.push(("LUMEN_MAX_PROMPT_TOKENS", n.to_string()));
        }
        if let Some(n) = self.prefill_chunk.filter(|n| *n > 0) {
            out.push(("LUMEN_QWEN35_PREFILL_CHUNK", n.to_string()));
        }
        match self.pack {
            FlashNextPack::Auto => {}
            FlashNextPack::Off => out.push(("LUMEN_MOE_PACK", "off".into())),
            FlashNextPack::Build => out.push(("LUMEN_MOE_PACK", "build".into())),
        }
        match self.draft_head {
            FlashNextDraftHead::Off => {}
            FlashNextDraftHead::Auto => out.push(("LUMEN_QWEN4_MTP", "auto".into())),
            FlashNextDraftHead::On => out.push(("LUMEN_QWEN4_MTP", "on".into())),
        }
        match self.io_mode {
            FlashNextIoMode::Auto => {}
            FlashNextIoMode::Direct => out.push(("LUMEN_MOE_IO_MODE", "direct".into())),
            FlashNextIoMode::Buffered => out.push(("LUMEN_MOE_IO_MODE", "buffered".into())),
        }
        if let Some(n) = self.io_lanes.filter(|n| {
            (1..=FLASH_NEXT_IO_LANES_MAX).contains(n) && *n != FLASH_NEXT_IO_LANES_DEFAULT
        }) {
            out.push(("LUMEN_MOE_IO_LANES", n.to_string()));
        }
        out
    }
}

/// The numbers a decode-speed estimate for this Mac rests on, as last measured. Written by the
/// Flash-Next card's Measure action and by the Doctor's SSD test; logic in `hardware.rs`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MeasuredHardware {
    /// GPU memory read bandwidth in GB/s (`lumen-server --measure-memory-bandwidth`).
    pub memory_gbps: Option<f64>,
    /// When it was measured, in unix seconds.
    pub memory_measured_at: Option<u64>,
    /// Random-read speed of the SSD in GB/s (four threads, 1 MB and up, page cache bypassed).
    pub ssd_random_gbps: Option<f64>,
    /// Front-to-back read speed of the same test, when it reported one.
    pub ssd_sequential_gbps: Option<f64>,
    /// The Flash-Next model directory the SSD test read. A different or moved model folder is
    /// another volume as far as the number goes: it no longer counts.
    pub ssd_volume: Option<String>,
    pub ssd_measured_at: Option<u64>,
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
                local_model_dir: None,
                skip_warmup: false,
                kv_disk_enabled: default_kv_disk_enabled(),
                kv_disk_max_gb: KV_DISK_DEFAULT_MAX_GB,
                kv_disk_ttl_secs: KV_DISK_DEFAULT_TTL_SECS,
                auto_start_server: false,
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
            flash_next: FlashNextConfig::default(),
            measured_hardware: MeasuredHardware::default(),
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
    // v9 -> v10: dropped `server.tokenizer_id`. It became `TOKENIZER_ID`,
    // which nothing in the server has read since the Candle backend went (the
    // MLX loaders take the tokenizer from the model directory). serde drops a
    // stored value on parse; this step stamps the version.
    if cfg.schema_version < 10 {
        cfg.schema_version = 10;
    }
    // v10 -> v11: the Flash-Next draft head's default flipped from `auto` to `off` (measured: it
    // gained at most 4% and lost up to 4.4%, and its text is not bit-identical to plain decoding).
    // Before, `auto` was the default and the card's first chip, which sent nothing; the same word
    // now means the opt-in "use it when installed". A config written before this step that says
    // `auto` meant "the default", so it becomes `off`, the new default, rather than silently
    // turning the head on. `on` and `off` were explicit choices and stay.
    if cfg.schema_version < 11 {
        if cfg.flash_next.draft_head == FlashNextDraftHead::Auto {
            cfg.flash_next.draft_head = FlashNextDraftHead::Off;
        }
        cfg.schema_version = 11;
    }
    // v11 -> v12: the disk KV tier went on by default, bounded: 10 GB, entries idle 30 days
    // expire, and (server-side) no writes while the volume has under 10 GB free. A config still
    // holding the old defaults as a whole (off, unlimited, one day) never chose them, so it takes
    // the new ones. Any other combination was a choice and stays exactly, a stored unlimited
    // (`0`) with the tier on included.
    if cfg.schema_version < 12 {
        let s = &mut cfg.server;
        if (s.kv_disk_enabled, s.kv_disk_max_gb, s.kv_disk_ttl_secs) == (false, 0, 86_400) {
            s.kv_disk_enabled = default_kv_disk_enabled();
            s.kv_disk_max_gb = KV_DISK_DEFAULT_MAX_GB;
            s.kv_disk_ttl_secs = KV_DISK_DEFAULT_TTL_SECS;
        }
        cfg.schema_version = 12;
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
        assert_eq!(cfg.schema_version, CURRENT_SCHEMA_VERSION);
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
        assert_eq!(cfg.schema_version, CURRENT_SCHEMA_VERSION);
    }

    /// A config already on v9 has no legacy table; the step must not clobber a
    /// value the user set under the new name.
    #[test]
    fn v9_migration_does_not_overwrite_an_existing_mlx_batch_max() {
        let mut cfg = PersistentConfig {
            schema_version: 8,
            ..PersistentConfig::default()
        };
        cfg.advanced.mlx_batch_max = Some(2);
        cfg.advanced.paged_attention = Some(LegacyPagedConfig { max_batch: Some(6) });
        migrate_in_place(&mut cfg);
        assert_eq!(cfg.advanced.mlx_batch_max, Some(2));
    }

    /// `tokenizer_id` went in v10: `TOKENIZER_ID` had no reader. A config that
    /// still carries it must load, land on the current version, and stop
    /// writing the field back out.
    #[test]
    fn v10_migration_drops_the_tokenizer_override() {
        let stored = toml::to_string_pretty(&PersistentConfig::default())
            .expect("serialize")
            .replacen(
                "[server]\n",
                "[server]\ntokenizer_id = \"Qwen/Qwen3-8B\"\n",
                1,
            )
            .replace(
                &format!("schema_version = {CURRENT_SCHEMA_VERSION}"),
                "schema_version = 9",
            );
        assert!(
            stored.contains("tokenizer_id"),
            "the fixture must carry the old field"
        );
        let mut cfg: PersistentConfig =
            toml::from_str(&stored).expect("a v9 config must still deserialize");
        migrate_in_place(&mut cfg);
        assert_eq!(cfg.schema_version, CURRENT_SCHEMA_VERSION);
        let out = toml::to_string_pretty(&cfg).expect("serialize");
        assert!(!out.contains("tokenizer_id"), "written back out:\n{out}");
    }

    /// A config saved while the PyO3 runner existed must still load, and must
    /// not write the removed variant back out.
    #[test]
    fn removed_mlx_pyo3_backend_loads_as_auto() {
        let stored = toml::to_string_pretty(&PersistentConfig::default())
            .expect("serialize")
            .replace("backend_mode = \"auto\"", "backend_mode = \"mlx-pyo3\"");
        assert!(
            stored.contains("mlx-pyo3"),
            "the fixture must carry the old value"
        );
        let cfg: PersistentConfig =
            toml::from_str(&stored).expect("a config naming mlx-pyo3 must still deserialize");
        assert_eq!(cfg.advanced.backend_mode, BackendMode::Auto);
        let out = toml::to_string_pretty(&cfg).expect("serialize");
        assert!(!out.contains("pyo3"), "written back out:\n{out}");
    }

    /// Guards the 7 -> 8 class of bug: a migration step that exists but is
    /// unreachable because the constant was never bumped past it.
    #[test]
    fn current_schema_version_reaches_the_last_migration_step() {
        let mut cfg = PersistentConfig {
            schema_version: 0,
            ..PersistentConfig::default()
        };
        migrate_in_place(&mut cfg);
        assert_eq!(
            cfg.schema_version, CURRENT_SCHEMA_VERSION,
            "migrate_in_place must land exactly on CURRENT_SCHEMA_VERSION"
        );
    }

    // ── Flash-Next ─────────────────────────────────────────────────────

    fn every_flash_next_field_set() -> FlashNextConfig {
        FlashNextConfig {
            expert_cache_gb: Some(12.5),
            max_prompt_tokens: Some(32_768),
            prefill_chunk: Some(512),
            pack: FlashNextPack::Build,
            draft_head: FlashNextDraftHead::On,
            io_mode: FlashNextIoMode::Direct,
            io_lanes: Some(16),
        }
    }

    /// A config written before the Flash-Next section existed (every older file) loads with
    /// every option at its default, so nothing is sent and the server decides.
    #[test]
    fn an_old_config_without_the_flash_next_section_loads_with_defaults() {
        let cfg: PersistentConfig =
            toml::from_str(V8_TOML).expect("a config without [flash_next] must still deserialize");
        assert_eq!(cfg.flash_next, FlashNextConfig::default());
        assert!(cfg.flash_next.env().is_empty());
        // The same file with the section present but empty, or with a few keys.
        let partial = format!("{V8_TOML}\n[flash_next]\npack = \"off\"\n");
        let cfg: PersistentConfig = toml::from_str(&partial).expect("a partial section loads");
        assert_eq!(cfg.flash_next.pack, FlashNextPack::Off);
        assert_eq!(cfg.flash_next.expert_cache_gb, None);
        assert_eq!(cfg.flash_next.draft_head, FlashNextDraftHead::Off);
    }

    /// Every config written before the auto-start option existed keeps the old behaviour (the
    /// server waits for the Start button), and turning it on survives a restart.
    #[test]
    fn auto_start_server_defaults_off_for_old_configs_and_round_trips() {
        let cfg: PersistentConfig =
            toml::from_str(V8_TOML).expect("a config without auto_start_server must deserialize");
        assert!(!cfg.server.auto_start_server);

        let mut on = PersistentConfig::default();
        on.server.auto_start_server = true;
        let raw = toml::to_string_pretty(&on).expect("serialize");
        let back: PersistentConfig = toml::from_str(&raw).expect("reload");
        assert!(back.server.auto_start_server);
    }

    /// What the settings card saves is what comes back after a restart, and the file keeps
    /// the readable lowercase names.
    #[test]
    fn the_flash_next_section_round_trips_through_the_config_file() {
        let cfg = PersistentConfig {
            flash_next: every_flash_next_field_set(),
            ..PersistentConfig::default()
        };
        let raw = toml::to_string_pretty(&cfg).expect("serialize");
        assert!(raw.contains("[flash_next]"), "{raw}");
        assert!(raw.contains("pack = \"build\""), "{raw}");
        assert!(raw.contains("draft_head = \"on\""), "{raw}");
        assert!(raw.contains("io_mode = \"direct\""), "{raw}");
        assert!(raw.contains("expert_cache_gb = 12.5"), "{raw}");
        let back: PersistentConfig = toml::from_str(&raw).expect("parse");
        assert_eq!(back.flash_next, cfg.flash_next);
        // The defaults round trip too: nothing set stays nothing set.
        let raw = toml::to_string_pretty(&PersistentConfig::default()).expect("serialize");
        let back: PersistentConfig = toml::from_str(&raw).expect("parse");
        assert_eq!(back.flash_next, FlashNextConfig::default());
    }

    /// The same shape the frontend sends and receives (`null` for an unset option).
    #[test]
    fn the_flash_next_section_speaks_the_json_the_frontend_does() {
        let json = serde_json::to_value(FlashNextConfig::default()).expect("json");
        assert_eq!(
            json,
            serde_json::json!({
                "expert_cache_gb": null,
                "max_prompt_tokens": null,
                "prefill_chunk": null,
                "pack": "auto",
                "draft_head": "off",
                "io_mode": "auto",
                "io_lanes": null,
            })
        );
        let back: FlashNextConfig = serde_json::from_value(serde_json::json!({
            "expert_cache_gb": 8, "max_prompt_tokens": 16384, "prefill_chunk": null,
            "pack": "off", "draft_head": "auto", "io_mode": "buffered", "io_lanes": 64,
        }))
        .expect("from json");
        assert_eq!(back.expert_cache_gb, Some(8.0));
        assert_eq!(back.pack, FlashNextPack::Off);
        assert_eq!(back.io_lanes, Some(64));
    }

    #[test]
    fn nothing_set_sends_nothing() {
        assert!(FlashNextConfig::default().env().is_empty());
        // Choosing the server's own default is the same as not choosing: nothing is sent.
        let at_defaults = FlashNextConfig {
            io_lanes: Some(FLASH_NEXT_IO_LANES_DEFAULT),
            ..FlashNextConfig::default()
        };
        assert!(at_defaults.env().is_empty());
    }

    #[test]
    fn every_set_option_maps_to_its_variable_as_a_plain_value() {
        assert_eq!(
            every_flash_next_field_set().env(),
            vec![
                ("LUMEN_MOE_EXPERT_CACHE_GB", "12.5".to_string()),
                ("LUMEN_MAX_PROMPT_TOKENS", "32768".to_string()),
                ("LUMEN_QWEN35_PREFILL_CHUNK", "512".to_string()),
                ("LUMEN_MOE_PACK", "build".to_string()),
                ("LUMEN_QWEN4_MTP", "on".to_string()),
                ("LUMEN_MOE_IO_MODE", "direct".to_string()),
                ("LUMEN_MOE_IO_LANES", "16".to_string()),
            ]
        );
        let whole = FlashNextConfig {
            expert_cache_gb: Some(8.0),
            pack: FlashNextPack::Off,
            draft_head: FlashNextDraftHead::Auto,
            io_mode: FlashNextIoMode::Buffered,
            ..FlashNextConfig::default()
        };
        let env: std::collections::BTreeMap<_, _> = whole.env().into_iter().collect();
        // `8`, never `8.0` or `8g`; `off`, `auto` and `buffered` as the server spells them.
        assert_eq!(env["LUMEN_MOE_EXPERT_CACHE_GB"], "8");
        assert_eq!(env["LUMEN_MOE_PACK"], "off");
        assert_eq!(env["LUMEN_QWEN4_MTP"], "auto");
        assert_eq!(env["LUMEN_MOE_IO_MODE"], "buffered");
        assert_eq!(env.len(), 4);
    }

    /// The draft head is off unless chosen: the default sends nothing (the server's own default
    /// is off), and `auto` and `on` are the two ways to opt in.
    #[test]
    fn the_draft_head_is_off_by_default_and_sent_only_when_chosen() {
        assert_eq!(
            FlashNextConfig::default().draft_head,
            FlashNextDraftHead::Off
        );
        let head = |h: FlashNextDraftHead| {
            FlashNextConfig {
                draft_head: h,
                ..FlashNextConfig::default()
            }
            .env()
        };
        assert!(head(FlashNextDraftHead::Off).is_empty());
        assert_eq!(
            head(FlashNextDraftHead::Auto),
            vec![("LUMEN_QWEN4_MTP", "auto".to_string())]
        );
        assert_eq!(
            head(FlashNextDraftHead::On),
            vec![("LUMEN_QWEN4_MTP", "on".to_string())]
        );
    }

    /// A config saved before the default flipped says `draft_head = "auto"` for "the default"
    /// (the old card's first chip sent nothing). The migration turns that into `off`, the new
    /// default, instead of silently turning the head on; a config already on the new version keeps
    /// an `auto` it chose, and explicit `on` / `off` are never touched.
    #[test]
    fn the_v11_migration_turns_the_old_default_auto_into_off_and_leaves_choices_alone() {
        let stored = |version: u32, head: &str| {
            let raw = toml::to_string_pretty(&PersistentConfig::default())
                .expect("serialize")
                .replace(
                    &format!("schema_version = {CURRENT_SCHEMA_VERSION}"),
                    &format!("schema_version = {version}"),
                )
                .replace("draft_head = \"off\"", &format!("draft_head = \"{head}\""));
            assert!(raw.contains(&format!("draft_head = \"{head}\"")), "{raw}");
            let mut cfg: PersistentConfig = toml::from_str(&raw).expect("parse");
            migrate_in_place(&mut cfg);
            cfg
        };
        // Written by a build whose default was `auto`.
        let old_default = stored(10, "auto");
        assert_eq!(old_default.flash_next.draft_head, FlashNextDraftHead::Off);
        assert_eq!(old_default.schema_version, CURRENT_SCHEMA_VERSION);
        assert!(old_default.flash_next.env().is_empty());
        // Explicit choices of that build survive.
        assert_eq!(
            stored(10, "on").flash_next.draft_head,
            FlashNextDraftHead::On
        );
        assert_eq!(
            stored(10, "off").flash_next.draft_head,
            FlashNextDraftHead::Off
        );
        // From the new version on, `auto` is the opt-in and stays.
        assert_eq!(
            stored(CURRENT_SCHEMA_VERSION, "auto").flash_next.draft_head,
            FlashNextDraftHead::Auto
        );
        // Re-running the chain changes nothing more.
        let mut again = stored(10, "auto");
        migrate_in_place(&mut again);
        assert_eq!(again.flash_next.draft_head, FlashNextDraftHead::Off);
    }

    // ── Disk KV tier ───────────────────────────────────────────────────

    /// `(enabled, max GB, TTL seconds)` of the disk tier.
    fn disk_tier(cfg: &PersistentConfig) -> (bool, usize, u32) {
        (
            cfg.server.kv_disk_enabled,
            cfg.server.kv_disk_max_gb,
            cfg.server.kv_disk_ttl_secs,
        )
    }

    /// A config file at `version` holding `tier`, loaded and migrated as the app does it.
    fn load_stored_disk_tier(version: u32, tier: (bool, usize, u32)) -> PersistentConfig {
        let mut cfg = PersistentConfig {
            schema_version: version,
            ..PersistentConfig::default()
        };
        (
            cfg.server.kv_disk_enabled,
            cfg.server.kv_disk_max_gb,
            cfg.server.kv_disk_ttl_secs,
        ) = tier;
        let raw = toml::to_string_pretty(&cfg).expect("serialize");
        let mut cfg: PersistentConfig = toml::from_str(&raw).expect("parse");
        migrate_in_place(&mut cfg);
        cfg
    }

    const OLD_DISK_TIER_DEFAULTS: (bool, usize, u32) = (false, 0, 86_400);
    const NEW_DISK_TIER_DEFAULTS: (bool, usize, u32) = (true, 10, 30 * 24 * 60 * 60);

    /// A new config has the disk tier on, at 10 GB, with entries idle 30 days expiring; so does
    /// a file written before the three fields existed.
    #[test]
    fn a_new_config_has_the_disk_tier_on_with_its_limits() {
        assert_eq!(
            disk_tier(&PersistentConfig::default()),
            NEW_DISK_TIER_DEFAULTS
        );
        let mut cfg: PersistentConfig = toml::from_str(V8_TOML).expect("no disk-tier fields");
        migrate_in_place(&mut cfg);
        assert_eq!(disk_tier(&cfg), NEW_DISK_TIER_DEFAULTS);
    }

    /// A config still holding the old defaults as a whole (off, unlimited, one day) never chose
    /// them, so it takes the new ones, once.
    #[test]
    fn the_v12_migration_moves_the_untouched_old_disk_tier_defaults_to_the_new_ones() {
        let mut cfg = load_stored_disk_tier(11, OLD_DISK_TIER_DEFAULTS);
        assert_eq!(disk_tier(&cfg), NEW_DISK_TIER_DEFAULTS);
        assert_eq!(cfg.schema_version, CURRENT_SCHEMA_VERSION);
        migrate_in_place(&mut cfg);
        assert_eq!(disk_tier(&cfg), NEW_DISK_TIER_DEFAULTS);
    }

    /// Any other combination is a choice and is kept exactly: the owner's (on, 100 GB, never
    /// expire), an explicit unlimited with the tier on, and every partial edit of the old
    /// defaults. From v12 on the old triple is a choice too (turned off after the flip).
    #[test]
    fn the_v12_migration_keeps_every_chosen_disk_tier_setting() {
        for chosen in [
            (true, 100, 0),
            (true, 0, 86_400),
            (true, 0, 0),
            (false, 0, 3_600),
            (false, 5, 86_400),
            (false, 0, 0),
        ] {
            assert_eq!(
                disk_tier(&load_stored_disk_tier(11, chosen)),
                chosen,
                "{chosen:?}"
            );
        }
        assert_eq!(
            disk_tier(&load_stored_disk_tier(
                CURRENT_SCHEMA_VERSION,
                OLD_DISK_TIER_DEFAULTS
            )),
            OLD_DISK_TIER_DEFAULTS
        );
    }

    #[test]
    fn a_pool_is_written_to_two_decimals_and_a_value_that_cannot_be_one_is_dropped() {
        let pool = |gb: f64| FlashNextConfig {
            expert_cache_gb: Some(gb),
            ..FlashNextConfig::default()
        };
        assert_eq!(
            pool(6.349).env(),
            vec![("LUMEN_MOE_EXPERT_CACHE_GB", "6.35".into())]
        );
        for bad in [0.0, -3.0, f64::NAN, f64::INFINITY] {
            assert!(pool(bad).env().is_empty(), "{bad}");
        }
        for lanes in [0, 65, 1000] {
            let c = FlashNextConfig {
                io_lanes: Some(lanes),
                ..FlashNextConfig::default()
            };
            assert!(c.env().is_empty(), "{lanes} threads");
        }
        let one = FlashNextConfig {
            max_prompt_tokens: Some(0),
            prefill_chunk: Some(0),
            ..FlashNextConfig::default()
        };
        assert!(one.env().is_empty());
    }

    // ── What this Mac measured ─────────────────────────────────────────

    /// Every config written before the measurements existed loads with none, so nothing is
    /// estimated until the first measurement.
    #[test]
    fn an_old_config_without_measurements_loads_with_none() {
        let cfg: PersistentConfig =
            toml::from_str(V8_TOML).expect("a config without [measured_hardware] must load");
        assert_eq!(cfg.measured_hardware, MeasuredHardware::default());
        // And with only some of the fields (a partial measurement).
        let partial = format!("{V8_TOML}\n[measured_hardware]\nmemory_gbps = 231.5\n");
        let cfg: PersistentConfig = toml::from_str(&partial).expect("a partial table loads");
        assert_eq!(cfg.measured_hardware.memory_gbps, Some(231.5));
        assert_eq!(cfg.measured_hardware.ssd_random_gbps, None);
        assert_eq!(cfg.measured_hardware.ssd_volume, None);
    }

    #[test]
    fn the_measurements_round_trip_through_the_config_file() {
        let mut cfg = PersistentConfig::default();
        cfg.measured_hardware.record_memory(231.5, 1_790_000_000);
        cfg.measured_hardware.record_ssd(
            3.1,
            Some(5.2),
            "/Volumes/Fast/models/Flash-Next",
            1_790_000_100,
        );
        let raw = toml::to_string_pretty(&cfg).expect("serialize");
        assert!(raw.contains("[measured_hardware]"), "{raw}");
        let back: PersistentConfig = toml::from_str(&raw).expect("parse");
        assert_eq!(back.measured_hardware, cfg.measured_hardware);
        // Nothing measured stays nothing measured.
        let raw = toml::to_string_pretty(&PersistentConfig::default()).expect("serialize");
        let back: PersistentConfig = toml::from_str(&raw).expect("parse");
        assert_eq!(back.measured_hardware, MeasuredHardware::default());
    }
}
