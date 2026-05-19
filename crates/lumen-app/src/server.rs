use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::Serialize;
use tauri::{AppHandle, Emitter};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

use crate::config::{BackendMode, CorsMode, PersistentConfig, SpecKind};

/// Tauri event name for streaming server log lines.
pub const EVENT_LOG: &str = "lumen://log";
/// Tauri event name for status transitions (running ↔ stopped).
pub const EVENT_STATUS: &str = "lumen://status";

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum LifecycleState {
    Stopped,
    Starting,
    Running,
    Stopping,
    Crashed,
}

#[derive(Debug, Clone, Serialize)]
pub struct ServerStatus {
    pub state: LifecycleState,
    pub pid: Option<u32>,
    pub port: u16,
    pub host: String,
    pub model_id: Option<String>,
    pub uptime_secs: Option<u64>,
    pub last_error: Option<String>,
}

pub struct ServerSupervisor {
    inner: Mutex<Inner>,
}

struct Inner {
    child: Option<Child>,
    pid: Option<u32>,
    state: LifecycleState,
    started_at: Option<Instant>,
    host: String,
    port: u16,
    model_id: Option<String>,
    last_error: Option<String>,
}

impl ServerSupervisor {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(Inner {
                child: None,
                pid: None,
                state: LifecycleState::Stopped,
                started_at: None,
                host: "127.0.0.1".into(),
                port: 8080,
                model_id: None,
                last_error: None,
            }),
        })
    }

    pub async fn status(&self) -> ServerStatus {
        let g = self.inner.lock().await;
        ServerStatus {
            state: g.state.clone(),
            pid: g.pid,
            port: g.port,
            host: g.host.clone(),
            model_id: g.model_id.clone(),
            uptime_secs: g.started_at.map(|t| t.elapsed().as_secs()),
            last_error: g.last_error.clone(),
        }
    }

    pub async fn start(
        self: &Arc<Self>,
        app: AppHandle,
        cfg: &PersistentConfig,
        model_id: &str,
        active_model_bytes: Option<u64>,
    ) -> Result<ServerStatus> {
        {
            let g = self.inner.lock().await;
            if matches!(g.state, LifecycleState::Running | LifecycleState::Starting) {
                anyhow::bail!("server already {:?}", g.state);
            }
        }

        let bin = resolve_binary(cfg.server_binary_path.as_deref())
            .context("resolve lumen-server binary path")?;

        let mut cmd = Command::new(&bin);
        cmd.stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        apply_env(&mut cmd, cfg, model_id, active_model_bytes);

        let mut child = cmd
            .spawn()
            .with_context(|| format!("spawn lumen-server at {}", bin.display()))?;
        let pid = child.id();

        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        {
            let mut g = self.inner.lock().await;
            g.child = Some(child);
            g.pid = pid;
            g.state = LifecycleState::Starting;
            g.started_at = Some(Instant::now());
            g.host = cfg.server.host.clone();
            g.port = cfg.server.port;
            g.model_id = Some(model_id.to_string());
            g.last_error = None;
        }

        // Stream stdout/stderr → frontend.
        if let Some(stdout) = stdout {
            let app2 = app.clone();
            tokio::spawn(async move { pipe_to_event(app2, stdout, "stdout").await });
        }
        if let Some(stderr) = stderr {
            let app2 = app.clone();
            tokio::spawn(async move { pipe_to_event(app2, stderr, "stderr").await });
        }

        // Probe `/v1/models` to flip state → Running once the HTTP listener
        // is up. Times out after 60 s (cold weight load takes 10-40 s).
        let host = cfg.server.host.clone();
        let port = cfg.server.port;
        let app_probe = app.clone();
        let sup = Arc::clone(self);
        tokio::spawn(async move {
            let healthy = probe_until_ready(&host, port, Duration::from_secs(60)).await;
            let mut g = sup.inner.lock().await;
            if healthy {
                g.state = LifecycleState::Running;
            } else {
                g.state = LifecycleState::Crashed;
                g.last_error = Some("health check timed out".into());
            }
            let status = ServerStatus {
                state: g.state.clone(),
                pid: g.pid,
                port: g.port,
                host: g.host.clone(),
                model_id: g.model_id.clone(),
                uptime_secs: g.started_at.map(|t| t.elapsed().as_secs()),
                last_error: g.last_error.clone(),
            };
            let _ = app_probe.emit(EVENT_STATUS, status);
        });

        // Reap child + flip state on exit.
        let sup_reap = Arc::clone(self);
        let app_reap = app.clone();
        tokio::spawn(async move {
            // Take the child out of the lock so we can `.wait()` without
            // holding the mutex.
            let child_opt = {
                let mut g = sup_reap.inner.lock().await;
                g.child.take()
            };
            if let Some(mut child) = child_opt {
                let _ = child.wait().await;
                let mut g = sup_reap.inner.lock().await;
                if g.state != LifecycleState::Stopping {
                    g.state = LifecycleState::Crashed;
                    g.last_error = Some("server exited unexpectedly".into());
                } else {
                    g.state = LifecycleState::Stopped;
                }
                g.pid = None;
                g.started_at = None;
                let status = ServerStatus {
                    state: g.state.clone(),
                    pid: g.pid,
                    port: g.port,
                    host: g.host.clone(),
                    model_id: g.model_id.clone(),
                    uptime_secs: None,
                    last_error: g.last_error.clone(),
                };
                let _ = app_reap.emit(EVENT_STATUS, status);
            }
        });

        Ok(self.status().await)
    }

    pub async fn stop(&self, app: AppHandle) -> Result<ServerStatus> {
        let pid = {
            let mut g = self.inner.lock().await;
            if !matches!(g.state, LifecycleState::Running | LifecycleState::Starting) {
                return Ok(ServerStatus {
                    state: g.state.clone(),
                    pid: g.pid,
                    port: g.port,
                    host: g.host.clone(),
                    model_id: g.model_id.clone(),
                    uptime_secs: g.started_at.map(|t| t.elapsed().as_secs()),
                    last_error: g.last_error.clone(),
                });
            }
            g.state = LifecycleState::Stopping;
            g.pid
        };

        // Send SIGTERM via nix; fall back to SIGKILL via the Child handle
        // after 5 s if it hasn't exited.
        if let Some(pid) = pid {
            use nix::sys::signal::{Signal, kill};
            use nix::unistd::Pid;
            let _ = kill(Pid::from_raw(pid as i32), Signal::SIGTERM);
        }

        // Watchdog — escalate to SIGKILL if needed.
        let sup_kill = self.inner.lock().await;
        drop(sup_kill);
        tokio::time::sleep(Duration::from_secs(5)).await;
        {
            let mut g = self.inner.lock().await;
            if matches!(g.state, LifecycleState::Stopping) {
                if let Some(child) = g.child.as_mut() {
                    let _ = child.start_kill();
                }
            }
        }
        let _ = app.emit(EVENT_STATUS, self.status().await);
        Ok(self.status().await)
    }
}

async fn pipe_to_event<R: tokio::io::AsyncRead + Unpin>(app: AppHandle, reader: R, stream: &str) {
    let mut lines = BufReader::new(reader).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        let _ = app.emit(
            EVENT_LOG,
            LogLine {
                stream: stream.into(),
                line,
                ts_unix_ms: now_ms(),
            },
        );
    }
}

#[derive(Debug, Clone, Serialize)]
struct LogLine {
    stream: String,
    line: String,
    ts_unix_ms: u128,
}

fn now_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

async fn probe_until_ready(host: &str, port: u16, timeout: Duration) -> bool {
    let url = format!("http://{}:{}/v1/models", host, port);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(800))
        .build()
        .expect("reqwest client");
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if let Ok(r) = client.get(&url).send().await {
            if r.status().is_success() {
                return true;
            }
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    false
}

/// Map `PersistentConfig` → subprocess env. Typed fields are emitted first;
/// `env_overrides` is applied last so it wins on collision (the UI surfaces a
/// "shadowed by override" warning when this happens).
fn apply_env(
    cmd: &mut Command,
    cfg: &PersistentConfig,
    model_id: &str,
    active_model_bytes: Option<u64>,
) {
    // ── Core ────────────────────────────────────────────────────────
    cmd.env("MODEL_ID", model_id)
        .env("PORT", cfg.server.port.to_string())
        .env("LUMEN_HOST", &cfg.server.host);

    match cfg.server.cors {
        CorsMode::Off => cmd.env("LUMEN_CORS", "off"),
        CorsMode::Localhost => cmd.env("LUMEN_CORS", "localhost"),
        CorsMode::All => cmd.env("LUMEN_CORS", "all"),
    };
    if let Some(k) = &cfg.server.api_key {
        cmd.env("LUMEN_API_KEY", k);
    }

    // ── Metal memory caps ──────────────────────────────────────────
    if cfg.server.disable_wired_limit {
        cmd.env("LUMEN_DISABLE_WIRED_LIMIT", "1");
    } else {
        // Wired limit precedence:
        //   1. User-set `wired_limit_gb` (GB-rounded knob in the SERVER card)
        //   2. Active model byte size — exactly matches the weights file so a
        //      14.45 GB model isn't truncated to a 14 GB ceiling
        //   3. Server-side default (28 GB)
        if let Some(g) = cfg.server.wired_limit_gb {
            cmd.env("LUMEN_WIRED_LIMIT_GB", g.to_string());
        } else if let Some(b) = active_model_bytes {
            cmd.env("LUMEN_WIRED_LIMIT_BYTES", b.to_string());
        }
        if let Some(g) = cfg.server.cache_limit_gb {
            cmd.env("LUMEN_CACHE_LIMIT_GB", g.to_string());
        }
        if let Some(g) = cfg.server.memory_limit_gb {
            cmd.env("LUMEN_MEMORY_LIMIT_GB", g.to_string());
        }
    }

    // ── Loader / warmup ────────────────────────────────────────────
    if let Some(eid) = &cfg.server.embedding_model_id {
        if !eid.is_empty() {
            cmd.env("EMBEDDING_MODEL_ID", eid);
        }
    }
    if let Some(tid) = &cfg.server.tokenizer_id {
        if !tid.is_empty() {
            cmd.env("TOKENIZER_ID", tid);
        }
    }
    if let Some(p) = &cfg.server.local_model_dir {
        // The server reads either LUMEN_GEMMA4_DIR or LUMEN_QWEN35_SHARDS
        // depending on the detected architecture. Set both — the unused one
        // is harmless.
        let s = p.to_string_lossy();
        cmd.env("LUMEN_GEMMA4_DIR", s.as_ref());
        cmd.env("LUMEN_QWEN35_SHARDS", s.as_ref());
    }
    if cfg.server.skip_warmup {
        cmd.env("SKIP_WARMUP", "1");
    }
    if let Some(n) = cfg.server.candle_compute_per_buffer {
        cmd.env("CANDLE_METAL_COMPUTE_PER_BUFFER", n.to_string());
    }

    // ── Generation defaults ────────────────────────────────────────
    if let Some(rp) = cfg.server.repeat_penalty {
        cmd.env("REPEAT_PENALTY", format!("{rp}"));
    }

    // ── Quantization (TurboQuant) ──────────────────────────────────
    cmd.env("TQ_BITS", cfg.quant.bits.to_string());
    cmd.env("TQ_QJL_M", cfg.quant.qjl_m.to_string());
    cmd.env("TQ_SEED", cfg.quant.seed.to_string());

    // ── Context ────────────────────────────────────────────────────
    // Three knobs from the CONTEXT card; each maps to a single env var the
    // server reads at startup. `sliding=0` means "use the model's built-in
    // sliding window from config.json" (i.e. no override).
    cmd.env("LUMEN_MAX_CTX", cfg.context.max.to_string());
    cmd.env("LUMEN_PREFILL_CHUNK", cfg.context.prefill.to_string());
    if cfg.context.sliding > 0 {
        cmd.env("LUMEN_SLIDING_WINDOW", cfg.context.sliding.to_string());
    }

    // ── Advanced ───────────────────────────────────────────────────
    match cfg.advanced.backend_mode {
        BackendMode::Auto => {}
        BackendMode::Candle => {
            cmd.env("LUMEN_MLX_BACKEND", "candle");
        }
        BackendMode::MlxNative => {
            cmd.env("LUMEN_MLX_BACKEND", "mlx-native");
        }
        BackendMode::MlxPyo3 => {
            cmd.env("LUMEN_MLX_BACKEND", "pyo3");
        }
    }
    match cfg.advanced.spec_kind {
        SpecKind::Off => {}
        SpecKind::Lookup => {
            cmd.env("LUMEN_SPEC", "lookup");
        }
        SpecKind::Mtp => {
            cmd.env("LUMEN_SPEC", "mtp");
        }
    }
    if let Some(n) = cfg.advanced.spec_draft_n_max {
        cmd.env("LUMEN_SPEC_DRAFT_N_MAX", n.to_string());
    }
    if cfg.advanced.batched_engine {
        cmd.env("BATCHED_ENGINE", "1");
    }

    // ── Paged attention ────────────────────────────────────────────
    let p = &cfg.advanced.paged_attention;
    if p.enabled {
        cmd.env("PAGED_KV", "1");
        if let Some(n) = p.layers {
            cmd.env("PAGED_LAYERS", n.to_string());
        }
        if let Some(n) = p.kv_heads {
            cmd.env("PAGED_KV_HEADS", n.to_string());
        }
        if let Some(n) = p.head_dim_sliding {
            cmd.env("PAGED_HEAD_DIM_SLIDING", n.to_string());
        }
        if let Some(n) = p.head_dim_global {
            cmd.env("PAGED_HEAD_DIM_GLOBAL", n.to_string());
        }
        if let Some(n) = p.global_every {
            cmd.env("PAGED_GLOBAL_EVERY", n.to_string());
        }
        if let Some(n) = p.max_batch {
            cmd.env("PAGED_MAX_BATCH", n.to_string());
        }
    }

    // ── Free-form overrides ────────────────────────────────────────
    // Applied last so they override the typed fields above. The UI shows a
    // "shadowing" warning when a key set here is also a typed field.
    for (k, v) in &cfg.env_overrides {
        if k.is_empty() {
            continue;
        }
        cmd.env(k, v);
    }
}

/// The set of env var names that have dedicated typed UI controls. Used by
/// the UI to flag `env_overrides` entries that shadow a typed field.
pub const TYPED_ENV_KEYS: &[&str] = &[
    "MODEL_ID",
    "PORT",
    "LUMEN_HOST",
    "LUMEN_CORS",
    "LUMEN_API_KEY",
    "LUMEN_DISABLE_WIRED_LIMIT",
    "LUMEN_WIRED_LIMIT_GB",
    "LUMEN_CACHE_LIMIT_GB",
    "LUMEN_MEMORY_LIMIT_GB",
    "EMBEDDING_MODEL_ID",
    "TOKENIZER_ID",
    "LUMEN_GEMMA4_DIR",
    "LUMEN_QWEN35_SHARDS",
    "SKIP_WARMUP",
    "CANDLE_METAL_COMPUTE_PER_BUFFER",
    "REPEAT_PENALTY",
    "TQ_BITS",
    "TQ_QJL_M",
    "TQ_SEED",
    "LUMEN_MAX_CTX",
    "LUMEN_SLIDING_WINDOW",
    "LUMEN_PREFILL_CHUNK",
    "LUMEN_WIRED_LIMIT_BYTES",
    "LUMEN_MLX_BACKEND",
    "LUMEN_SPEC",
    "LUMEN_SPEC_DRAFT_N_MAX",
    "BATCHED_ENGINE",
    "PAGED_KV",
    "PAGED_LAYERS",
    "PAGED_KV_HEADS",
    "PAGED_HEAD_DIM_SLIDING",
    "PAGED_HEAD_DIM_GLOBAL",
    "PAGED_GLOBAL_EVERY",
    "PAGED_MAX_BATCH",
];

/// Public wrapper for `resolve_binary` — used by the doctor module so the
/// "binary located?" check uses the exact same resolution order as start().
pub fn resolve_binary_public(explicit: Option<&Path>) -> Result<PathBuf> {
    resolve_binary(explicit)
}

/// Resolution order:
/// 1. Explicit `cfg.server_binary_path` if set
/// 2. Sibling binary in the .app bundle (Resources/lumen-server)
/// 3. `LUMEN_SERVER_BIN` env var
/// 4. `lumen-server` on PATH
/// 5. Workspace target dir (dev fallback): `target/release/lumen-server`,
///    then `target/debug/lumen-server`
fn resolve_binary(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        if p.exists() {
            return Ok(p.to_path_buf());
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let sibling = dir.join("lumen-server");
            if sibling.exists() {
                return Ok(sibling);
            }
            // macOS .app: Resources/ sibling
            let resources = dir.join("../Resources/lumen-server");
            if resources.exists() {
                return Ok(resources);
            }
        }
    }
    if let Ok(p) = std::env::var("LUMEN_SERVER_BIN") {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return Ok(pb);
        }
    }
    if let Ok(p) = which::which("lumen-server") {
        return Ok(p);
    }
    // Dev fallback — walk up from CWD looking for target/{release,debug}.
    let mut cwd = std::env::current_dir()?;
    for _ in 0..5 {
        for profile in ["release", "debug"] {
            let candidate = cwd.join("target").join(profile).join("lumen-server");
            if candidate.exists() {
                return Ok(candidate);
            }
        }
        if !cwd.pop() {
            break;
        }
    }
    anyhow::bail!(
        "lumen-server binary not found — set `server_binary_path` in config.toml or build it via `cargo build -p lumen-server --release`"
    )
}
