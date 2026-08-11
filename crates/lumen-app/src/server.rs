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

use crate::config::{BackendMode, CorsMode, PersistentConfig, QuantKvMode, SpecKind};

/// What the spawned `lumen-server` loads. Mirrors the server-side `ServeMode`
/// and maps 1:1 onto `LUMEN_SERVE` (chat → unset/"chat", image → "image",
/// hybrid → "hybrid"). Hybrid co-resides the LLM + diffusion engines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServeKind {
    Chat,
    Image,
    Hybrid,
}

impl ServeKind {
    /// True when the diffusion backend is loaded (image or hybrid). Used to
    /// skip the wired-memory ceiling — the ~31 GB diffusion working set exceeds
    /// the LLM-tuned cap and pinning it would OOM the load.
    fn loads_diffusion(self) -> bool {
        matches!(self, ServeKind::Image | ServeKind::Hybrid)
    }
}

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
    metrics: MetricsAccumulator,
}

/// Live decode metrics aggregated from server stderr. We extract numbers from
/// the structured log lines lumen-server already emits (no extra HTTP endpoint
/// needed — keeps the engine side flag-clean). Two patterns are parsed:
///
///   1. `[stream-timing] sse: ... steady_rate_recv=23.45tok/s ...`
///      — emitted once per `/v1/chat/completions` request (chat.rs)
///   2. `seq N done: M tokens in T ms (X.Y tok/s)`
///      — emitted by the MlxQwen35Backend + Candle batched paths
///
/// Both signals feed an EMA-smoothed `tok_per_sec` + derived
/// `ms_per_step = 1000 / tok_per_sec`. `request_times` keeps a sliding
/// 60-second window of decode-finish timestamps for the requests/min
/// counter.
#[derive(Debug, Default)]
struct MetricsAccumulator {
    tok_per_sec_ema: Option<f64>,
    ms_per_step_ema: Option<f64>,
    request_times: std::collections::VecDeque<Instant>,
}

impl MetricsAccumulator {
    /// EMA smoothing factor — higher = more responsive, lower = more stable.
    const EMA_ALPHA: f64 = 0.3;

    fn observe(&mut self, line: &str) {
        if let Some(tps) = parse_tok_per_sec(line) {
            // Reject obviously bogus values (parser collision).
            if (0.1..=1000.0).contains(&tps) {
                let next = match self.tok_per_sec_ema {
                    Some(prev) => Self::EMA_ALPHA * tps + (1.0 - Self::EMA_ALPHA) * prev,
                    None => tps,
                };
                self.tok_per_sec_ema = Some(next);
                self.ms_per_step_ema = Some(1000.0 / next);

                // Each tok/s emission ≈ one finished decode request. Stamp it.
                let now = Instant::now();
                self.request_times.push_back(now);
                self.evict_old(now);
            }
        }
    }

    fn evict_old(&mut self, now: Instant) {
        let cutoff = Duration::from_secs(60);
        while let Some(&front) = self.request_times.front() {
            if now.duration_since(front) > cutoff {
                self.request_times.pop_front();
            } else {
                break;
            }
        }
    }

    fn snapshot(&mut self) -> ServerMetricsSnapshot {
        self.evict_old(Instant::now());
        ServerMetricsSnapshot {
            tokens_per_sec: self.tok_per_sec_ema,
            ms_per_step: self.ms_per_step_ema,
            requests_per_min: Some(self.request_times.len() as u32),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ServerMetricsSnapshot {
    pub tokens_per_sec: Option<f64>,
    pub ms_per_step: Option<f64>,
    pub requests_per_min: Option<u32>,
}

/// Extract a `tok/s` reading from a single log line, if present.
///
/// Recognized formats — see `docs/backend-metrics-convention.md` for the
/// rationale and the rules any new backend must follow:
///
/// 1. `... done: <N> tokens in <T>ms (<R> tok/s)` — decode-finalization
///    emission. The `done:` keyword distinguishes this from prefill /
///    per-step / EOS-mid-decode logs that also carry a `tok/s` reading
///    but are not the canonical end-of-request rate.
/// 2. `steady_rate_recv=<R>tok/s` — SSE stream-timing line (opt-in via
///    `LUMEN_STREAM_TIMING=1`).
///
/// Returns `None` for any other shape so prefill / per-step / aggregate
/// lines don't pollute the EMA.
fn parse_tok_per_sec(line: &str) -> Option<f64> {
    if let Some(rest) = line.split("steady_rate_recv=").nth(1) {
        // "23.45tok/s last_write..." — take up to "tok/s"
        let num = rest.split("tok/s").next()?;
        if let Ok(v) = num.trim().parse::<f64>() {
            return Some(v);
        }
    }
    // Require "done:" before the rate so we only sample the final
    // end-of-decode emission, not prefill / per-step diagnostics that
    // also format their rate as "(N.N tok/s)".
    if !line.contains("done:") {
        return None;
    }
    // Take the LAST " tok/s)" occurrence so a multi-rate done line
    // (`prefill ... (X tok/s) | decode ... (Y tok/s) | e2e ... (Z
    // tok/s)`) reports the end-to-end Z — what the user actually
    // perceives — rather than the prefill X they'd otherwise see.
    if let Some(idx) = line.rfind(" tok/s)") {
        let prefix = &line[..idx];
        let num: String = prefix
            .chars()
            .rev()
            .take_while(|c| c.is_ascii_digit() || *c == '.')
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        if let Ok(v) = num.parse::<f64>() {
            return Some(v);
        }
    }
    None
}

#[cfg(test)]
mod parse_tests {
    use super::parse_tok_per_sec;

    #[test]
    fn parses_stream_timing_sse() {
        let line = "[stream-timing] sse: n_deltas=42 first->last_recv=2150.4ms skip2->last_recv=1843.5ms steady_rate_recv=22.78tok/s last_write->DONE_flush=0.4ms";
        assert!((parse_tok_per_sec(line).unwrap() - 22.78).abs() < 1e-6);
    }

    #[test]
    fn parses_mlx_done() {
        let line = "[mlx] seq 7 done: 128 tokens in 2143ms (59.7 tok/s)";
        assert!((parse_tok_per_sec(line).unwrap() - 59.7).abs() < 1e-6);
    }

    #[test]
    fn parses_batched_engine_step() {
        let line = "[batched engine] step: N=4 latency=42.1ms aggregate=15.3 tok/s";
        // No closing paren — current parser only matches "(... tok/s)" form
        // and "steady_rate_recv=...". The aggregate-tok/s log is informational
        // and we don't pull from it. Confirm we don't false-positive.
        assert!(parse_tok_per_sec(line).is_none());
    }

    #[test]
    fn rejects_unrelated_lines() {
        assert!(parse_tok_per_sec("Loading model: foo").is_none());
        assert!(parse_tok_per_sec("Error: tok/s computed wrong").is_none());
    }

    #[test]
    fn rejects_prefill_line() {
        // Prefill lines look almost identical to done lines but represent
        // prompt-processing throughput, not decode rate — must be ignored.
        let line = "[mlx] seq 7 prefill: 4096 tokens in 1500ms (2730.7 tok/s) -> tok=42";
        assert!(parse_tok_per_sec(line).is_none());
    }

    #[test]
    fn rejects_eos_mid_decode() {
        // The mid-decode EOS log carries an instantaneous rate but isn't
        // the canonical end-of-request signal — the trailing "done:" line
        // is what we want to sample.
        let line = "[mlx] seq 7 EOS at step 42 (28.3 tok/s)";
        assert!(parse_tok_per_sec(line).is_none());
    }

    #[test]
    fn parses_gemma4_done() {
        // gemma4_backend emits this format from chat / completion paths.
        let line = "[gemma4] chat done: 42 tokens in 1530ms (27.5 tok/s)";
        assert!((parse_tok_per_sec(line).unwrap() - 27.5).abs() < 1e-6);
    }

    #[test]
    fn parses_gemma4_done_picks_e2e_from_multi_rate_line() {
        // New (post split-prefill-vs-decode) format. Three rates in one
        // line; the parser must surface the LAST one (end-to-end) so the
        // tok/s number the user sees in the UI matches their wall-clock
        // experience, not the prefill burst that always looks faster.
        let line = "[gemma4] chat done: prefill 12tok in 250ms (48.0 tok/s) | decode 100tok in 2000ms (50.0 tok/s) | e2e 112tok in 2250ms (49.8 tok/s)";
        assert!((parse_tok_per_sec(line).unwrap() - 49.8).abs() < 1e-6);
    }
}

#[cfg(test)]
mod resolve_tests {
    use super::prefer_release_over_dev_debug;
    use std::path::PathBuf;

    fn tmpdir(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("lumen-resolve-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn prefer_release_when_both_exist() {
        let root = tmpdir("both");
        let debug = root.join("target").join("debug");
        let release = root.join("target").join("release");
        std::fs::create_dir_all(&debug).unwrap();
        std::fs::create_dir_all(&release).unwrap();
        std::fs::write(debug.join("lumen-server"), b"").unwrap();
        std::fs::write(release.join("lumen-server"), b"").unwrap();
        let chosen = prefer_release_over_dev_debug(&debug.join("lumen-server"));
        assert_eq!(
            chosen.as_deref(),
            Some(release.join("lumen-server").as_path())
        );
    }

    #[test]
    fn fallback_to_debug_when_no_release() {
        let root = tmpdir("debug-only");
        let debug = root.join("target").join("debug");
        std::fs::create_dir_all(&debug).unwrap();
        std::fs::write(debug.join("lumen-server"), b"").unwrap();
        let chosen = prefer_release_over_dev_debug(&debug.join("lumen-server"));
        assert!(
            chosen.is_none(),
            "no release available -> return None so caller keeps debug"
        );
    }

    #[test]
    fn skip_override_for_non_target_paths() {
        // .app bundle sibling: dir != "debug" so override is a no-op.
        let root = tmpdir("bundle");
        let resources = root.join("Contents").join("Resources");
        std::fs::create_dir_all(&resources).unwrap();
        std::fs::write(resources.join("lumen-server"), b"").unwrap();
        let chosen = prefer_release_over_dev_debug(&resources.join("lumen-server"));
        assert!(chosen.is_none());
    }

    #[test]
    fn respect_lumen_use_debug_server_env() {
        let root = tmpdir("env-opt-out");
        let debug = root.join("target").join("debug");
        let release = root.join("target").join("release");
        std::fs::create_dir_all(&debug).unwrap();
        std::fs::create_dir_all(&release).unwrap();
        std::fs::write(debug.join("lumen-server"), b"").unwrap();
        std::fs::write(release.join("lumen-server"), b"").unwrap();
        // Safety: scoped to this test thread; tests in this module run
        // sequentially under cargo's default config.
        unsafe { std::env::set_var("LUMEN_USE_DEBUG_SERVER", "1") };
        let chosen = prefer_release_over_dev_debug(&debug.join("lumen-server"));
        unsafe { std::env::remove_var("LUMEN_USE_DEBUG_SERVER") };
        assert!(chosen.is_none(), "explicit opt-out must short-circuit");
    }
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
                metrics: MetricsAccumulator::default(),
            }),
        })
    }

    /// Live decode metrics aggregated from server stderr. Returns `None`-filled
    /// fields until the first chat request finishes; thereafter EMA-smoothed.
    pub async fn metrics(&self) -> ServerMetricsSnapshot {
        let mut g = self.inner.lock().await;
        g.metrics.snapshot()
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
        image_model_id: Option<&str>,
        serve: ServeKind,
    ) -> Result<ServerStatus> {
        {
            let g = self.inner.lock().await;
            if matches!(g.state, LifecycleState::Running | LifecycleState::Starting) {
                anyhow::bail!("server already {:?}", g.state);
            }
        }

        let bin = resolve_binary(cfg.server_binary_path.as_deref())
            .context("resolve lumen-server binary path")?;
        eprintln!("[lumen-app] using lumen-server: {}", bin.display());

        // If a previous app exit (or force-quit / crash) left an orphaned
        // lumen-server holding the configured port, reclaim it before
        // spawning the new one — otherwise bind() fails with EADDRINUSE.
        // Only kills processes whose argv0 name contains "lumen-server" to
        // avoid wiping unrelated services on the same port.
        reclaim_port_if_lumen_server(&cfg.server.host, cfg.server.port);

        let mut cmd = Command::new(&bin);
        cmd.stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        apply_env(
            &mut cmd,
            cfg,
            model_id,
            active_model_bytes,
            image_model_id,
            serve,
        );

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
            // Display id: the LLM when present, else the image model (image-only
            // launches pass an empty `model_id`).
            g.model_id = Some(if model_id.is_empty() {
                image_model_id.unwrap_or(model_id).to_string()
            } else {
                model_id.to_string()
            });
            g.last_error = None;
        }

        // Stream stdout/stderr → frontend. stderr is also tee'd into the
        // metrics accumulator since that's where the engine emits its
        // structured timing lines.
        if let Some(stdout) = stdout {
            let app2 = app.clone();
            let sup2 = Arc::clone(self);
            tokio::spawn(async move { pipe_to_event(app2, sup2, stdout, "stdout").await });
        }
        if let Some(stderr) = stderr {
            let app2 = app.clone();
            let sup2 = Arc::clone(self);
            tokio::spawn(async move { pipe_to_event(app2, sup2, stderr, "stderr").await });
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

    /// Synchronous best-effort kill for app-exit cleanup. Tauri's
    /// `RunEvent::ExitRequested` fires on the event-loop thread and may
    /// `std::process::exit()` before any tokio runtime drop, so we can't
    /// rely on `kill_on_drop` or the async `stop()` path — both need a live
    /// runtime. Instead grab the PID via `try_lock` and send signals
    /// directly via `nix::kill`. SIGTERM → 3 s grace → SIGKILL fallback.
    /// Returns silently if no server is running or the lock is contended
    /// (treat both as "nothing to clean up").
    pub fn shutdown_blocking(&self) {
        use nix::sys::signal::{Signal, kill};
        use nix::unistd::Pid;

        let pid = match self.inner.try_lock() {
            Ok(g) => g.pid,
            Err(_) => return,
        };
        let Some(pid) = pid else { return };
        let pid_t = Pid::from_raw(pid as i32);

        if kill(pid_t, Signal::SIGTERM).is_err() {
            return;
        }

        // Poll for exit every 100 ms up to 3 s. `kill(pid, 0)` returns Err
        // (ESRCH) once the process is reaped — that's our exit signal.
        for _ in 0..30 {
            std::thread::sleep(Duration::from_millis(100));
            if kill(pid_t, None).is_err() {
                return;
            }
        }

        let _ = kill(pid_t, Signal::SIGKILL);
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

/// Max lines coalesced into one `EVENT_LOG` emit, and the idle window after
/// which a partial batch is flushed.
const LOG_BATCH_MAX: usize = 512;
const LOG_FLUSH_INTERVAL: Duration = Duration::from_millis(50);

/// Drain the batch: observe stderr metrics under ONE lock, then emit the whole
/// `Vec<LogLine>` as a single IPC event. `batch` is left empty (capacity reset).
async fn flush_log_batch(
    app: &AppHandle,
    sup: &Arc<ServerSupervisor>,
    is_stderr: bool,
    batch: &mut Vec<LogLine>,
) {
    if batch.is_empty() {
        return;
    }
    if is_stderr {
        let mut g = sup.inner.lock().await;
        for l in batch.iter() {
            g.metrics.observe(&l.line);
        }
    }
    let _ = app.emit(EVENT_LOG, std::mem::take(batch));
}

/// Read the child's stdout/stderr line-by-line and forward it to the webview.
///
/// Lines are **batched** (size + time window) into a single `EVENT_LOG` carrying
/// a `Vec<LogLine>` rather than one IPC event per line. Under a log flood (an
/// agentic client with verbose env can emit 10k+ lines/s) per-line emits would
/// saturate the Tauri main loop (which also drives the UI) and, because the read
/// loop and emit shared a task, back-pressure could fill the OS pipe and stall
/// the server's `eprintln!` → freeze the inference engine. Batching cuts IPC
/// events ~512× and collapses the frontend's per-line O(n) state update to one
/// per batch, so the read side keeps draining the pipe and the engine never
/// blocks on logging.
async fn pipe_to_event<R: tokio::io::AsyncRead + Unpin>(
    app: AppHandle,
    sup: Arc<ServerSupervisor>,
    reader: R,
    stream: &str,
) {
    let mut lines = BufReader::new(reader).lines();
    let is_stderr = stream == "stderr";
    let mut batch: Vec<LogLine> = Vec::with_capacity(LOG_BATCH_MAX);
    loop {
        let mut flush = false;
        tokio::select! {
            res = lines.next_line() => match res {
                Ok(Some(line)) => {
                    batch.push(LogLine {
                        stream: stream.into(),
                        line,
                        ts_unix_ms: now_ms(),
                    });
                    if batch.len() >= LOG_BATCH_MAX {
                        flush = true;
                    }
                }
                // EOF or read error — flush what we have and exit.
                _ => {
                    flush_log_batch(&app, &sup, is_stderr, &mut batch).await;
                    return;
                }
            },
            // Idle flush: the sleep is recreated each iteration, so a sustained
            // flood keeps resetting it and only the size cap fires (max batching);
            // when input goes quiet a partial batch ships within the window.
            _ = tokio::time::sleep(LOG_FLUSH_INTERVAL), if !batch.is_empty() => {
                flush = true;
            }
        }
        if flush {
            flush_log_batch(&app, &sup, is_stderr, &mut batch).await;
        }
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
    image_model_id: Option<&str>,
    serve: ServeKind,
) {
    // ── Core ────────────────────────────────────────────────────────
    // MODEL_ID selects the LLM; omit it for image-only launches (empty id) so
    // the server falls back to its own default rather than trying to load "".
    if !model_id.is_empty() {
        cmd.env("MODEL_ID", model_id);
    }
    cmd.env("PORT", cfg.server.port.to_string())
        .env("LUMEN_HOST", &cfg.server.host);

    // IMAGE_MODEL_ID selects the diffusion pipeline (4-bit `flux2-dev` vs the
    // bf16 `black-forest-labs/FLUX.2-dev`). The server reads this — NOT MODEL_ID
    // — for the image backend, so it must be set whenever diffusion is loaded.
    if let Some(im) = image_model_id {
        cmd.env("IMAGE_MODEL_ID", im);
    }

    // Serve mode → LUMEN_SERVE. Chat leaves it unset (server defaults to chat);
    // image / hybrid select the diffusion backend (hybrid keeps the LLM too).
    // Crucially we must NOT pin a wired limit below when diffusion is loaded —
    // the working set (~31 GB) exceeds the device recommended set and pinning it
    // OOM-kills the load (see the memory-caps block).
    match serve {
        ServeKind::Chat => {}
        ServeKind::Image => {
            cmd.env("LUMEN_SERVE", "image");
        }
        ServeKind::Hybrid => {
            cmd.env("LUMEN_SERVE", "hybrid");
        }
    }

    match cfg.server.cors {
        CorsMode::Off => cmd.env("LUMEN_CORS", "off"),
        CorsMode::Localhost => cmd.env("LUMEN_CORS", "localhost"),
        CorsMode::All => cmd.env("LUMEN_CORS", "all"),
    };
    if let Some(k) = &cfg.server.api_key {
        cmd.env("LUMEN_API_KEY", k);
    }

    // ── Metal memory caps ──────────────────────────────────────────
    // Image / hybrid modes manage their own governors server-side (skip the
    // wired ceiling because the ~31 GB diffusion working set exceeds the device
    // recommended set). Pinning a wired/byte limit from here would re-introduce
    // the OOM, so leave all caps unset and let the server pick its defaults.
    if serve.loads_diffusion() {
        // no-op: server-side image/hybrid memory defaults apply.
    } else if cfg.server.disable_wired_limit {
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

    // ── Disk KV cache (persistent prefix-cache tier) ───────────────
    if cfg.server.kv_disk_enabled {
        cmd.env("LUMEN_KV_DISK", "1");
        cmd.env(
            "LUMEN_KV_DISK_MAX_GB",
            cfg.server.kv_disk_max_gb.to_string(),
        );
        cmd.env(
            "LUMEN_KV_DISK_TTL_SECS",
            cfg.server.kv_disk_ttl_secs.to_string(),
        );
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

    // ── Quantization (TurboQuant) ──────────────────────────────────
    // QJL Stage-2 projection dimension is fixed to D·4 = 1024 (Gemma 4
    // KV-cache simple quantization (Q3/Q4/Q6/Q8). The TurboQuant lever was
    // retired from the user-facing surface in schema v8 — empirical sweeps
    // (2026-05-26) showed TQ is net-negative on Apple Silicon batch=1 at
    // every context length tested. TQ env vars are no longer emitted; the
    // kernel code remains in-tree behind dev-only `LUMEN_GEMMA4_TQ_*` env
    // vars that users can still set manually through the Env Overrides tab
    // for ablation / CUDA Phase 2 work.
    let kv_mode_str = match cfg.quant.kv_mode {
        QuantKvMode::Off => "off",
        QuantKvMode::On => "on",
        QuantKvMode::Auto => "auto",
    };
    cmd.env("LUMEN_GEMMA4_QUANT_KV_MODE", kv_mode_str);
    cmd.env("LUMEN_GEMMA4_QUANT_KV_BITS", cfg.quant.bits.to_string());
    if matches!(cfg.quant.kv_mode, QuantKvMode::Auto) {
        cmd.env(
            "LUMEN_GEMMA4_QUANT_KV_AUTO_THRESHOLD_TOKENS",
            cfg.quant.kv_auto_threshold_tokens.to_string(),
        );
    }
    // Legacy `TQ_BITS` for the older turboquant-cache crate path used by
    // Candle backends / smaller MoE models. Mirror the same bit width so
    // a future re-enablement uses the same compression level the user
    // picked here. (No-op when those backends aren't loaded.)
    cmd.env("TQ_BITS", cfg.quant.bits.to_string());

    // ── Context ────────────────────────────────────────────────────
    // Three knobs from the CONTEXT card; each maps to a single env var the
    // server reads at startup. `sliding=0` means "use the model's built-in
    // sliding window from config.json" (i.e. no override).
    cmd.env("LUMEN_MAX_CTX", cfg.context.max.to_string());
    cmd.env("LUMEN_PREFILL_CHUNK", cfg.context.prefill.to_string());
    if cfg.context.sliding > 0 {
        cmd.env("LUMEN_SLIDING_WINDOW", cfg.context.sliding.to_string());
    }
    // Default `max_tokens` budget when the API client omits the field on
    // /v1/chat/completions or /v1/completions. `0` is forwarded as "unbounded
    // — generate until EOS / stop / context budget".
    //
    // The same value is also emitted as `LUMEN_MAX_TOKENS_CAP` so the
    // UI knob acts as a single ceiling: clients that send an explicit
    // `max_tokens` (e.g. Ayla hard-coding the full 256K ctx window) are
    // capped to the same value, instead of falling through to the
    // server's compiled-in 2048 default. `0` keeps the cap disabled,
    // matching the "unbounded" semantics of `LUMEN_DEFAULT_MAX_TOKENS=0`.
    cmd.env(
        "LUMEN_DEFAULT_MAX_TOKENS",
        cfg.context.default_max_tokens.to_string(),
    );
    cmd.env(
        "LUMEN_MAX_TOKENS_CAP",
        cfg.context.default_max_tokens.to_string(),
    );

    // ── Advanced ───────────────────────────────────────────────────
    match cfg.advanced.backend_mode {
        BackendMode::Auto => {}
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
        // `LUMEN_SPEC_DRAFT_N_MAX` was never read by anything; the runner's
        // knob is `LUMEN_SPEC_K`.
        cmd.env("LUMEN_SPEC_K", n.to_string());
    }
    if cfg.advanced.batched_engine {
        // `BATCHED_ENGINE` drove the Candle scheduler, which is gone. MLX has
        // its own, behind this flag.
        cmd.env("LUMEN_MLX_BATCH_DECODE", "1");
    }
    // Batch width for the MLX scheduler above. The five other `PAGED_*` vars
    // this used to emit went out with the PagedAttention crate — nothing had
    // read them since the Candle backend was removed, so toggling them in the
    // app silently did nothing.
    if let Some(n) = cfg.advanced.mlx_batch_max {
        cmd.env("LUMEN_MLX_BATCH_MAX", n.to_string());
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
    "LUMEN_GEMMA4_QUANT_KV_SLIDING_TURBOQUANT",
    "LUMEN_GEMMA4_QUANT_KV_SLIDING_TURBOQUANT_BITS",
    "LUMEN_GEMMA4_QUANT_KV_SLIDING_TURBOQUANT_QJL",
    "LUMEN_GEMMA4_QUANT_KV_SLIDING_TURBOQUANT_QJL_M",
    "LUMEN_KV_DISK",
    "LUMEN_KV_DISK_MAX_GB",
    "LUMEN_KV_DISK_TTL_SECS",
    "LUMEN_MAX_CTX",
    "LUMEN_SLIDING_WINDOW",
    "LUMEN_PREFILL_CHUNK",
    "LUMEN_DEFAULT_MAX_TOKENS",
    "LUMEN_MAX_TOKENS_CAP",
    "LUMEN_WIRED_LIMIT_BYTES",
    "LUMEN_MLX_BACKEND",
    "LUMEN_SPEC",
    "LUMEN_SPEC_K",
    "LUMEN_MLX_BATCH_DECODE",
    "LUMEN_MLX_BATCH_MAX",
];

/// Public wrapper for `resolve_binary` — used by the doctor module so the
/// "binary located?" check uses the exact same resolution order as start().
pub fn resolve_binary_public(explicit: Option<&Path>) -> Result<PathBuf> {
    resolve_binary(explicit)
}

/// If `host:port` is occupied by a process whose argv0 name contains
/// `lumen-server`, send it SIGTERM (then SIGKILL after a short grace) so
/// `start()` can bind cleanly. No-op when:
/// - nothing is listening on the port
/// - the listener is some other process (we leave it alone; the spawn will
///   fail loudly and surface as a port-collision error to the user)
///
/// macOS-only — uses `lsof` (always present on macOS) + `ps`. On other
/// platforms this would need a different probe; Lumen ships Apple Silicon
/// only so the cross-platform fork can wait.
fn reclaim_port_if_lumen_server(host: &str, port: u16) {
    use std::process::Command as StdCommand;

    let lsof_target = if host == "0.0.0.0" || host.is_empty() {
        format!("-iTCP:{}", port)
    } else {
        format!("-iTCP@{}:{}", host, port)
    };
    let out = match StdCommand::new("lsof")
        .args(["-nP", &lsof_target, "-sTCP:LISTEN", "-t"])
        .output()
    {
        Ok(o) if o.status.success() => o,
        _ => return,
    };
    let stdout = String::from_utf8_lossy(&out.stdout);
    let pids: Vec<i32> = stdout
        .lines()
        .filter_map(|l| l.trim().parse::<i32>().ok())
        .collect();
    if pids.is_empty() {
        return;
    }

    use nix::sys::signal::{Signal, kill};
    use nix::unistd::Pid;

    for pid in pids {
        // Confirm it's actually a lumen-server before killing.
        let ps_out = StdCommand::new("ps")
            .args(["-p", &pid.to_string(), "-o", "comm="])
            .output();
        let is_lumen = match ps_out {
            Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout)
                .trim()
                .rsplit('/')
                .next()
                .map(|name| name.contains("lumen-server"))
                .unwrap_or(false),
            _ => false,
        };
        if !is_lumen {
            continue;
        }

        let pid_t = Pid::from_raw(pid);
        let _ = kill(pid_t, Signal::SIGTERM);
        for _ in 0..20 {
            std::thread::sleep(Duration::from_millis(100));
            if kill(pid_t, None).is_err() {
                break;
            }
        }
        if kill(pid_t, None).is_ok() {
            let _ = kill(pid_t, Signal::SIGKILL);
            // brief settle so the kernel reclaims the socket before bind()
            std::thread::sleep(Duration::from_millis(200));
        }
    }
}

/// Resolution order:
/// 1. Explicit `cfg.server_binary_path` if set
/// 2. Sibling binary in the .app bundle (Resources/lumen-server)
///    — but if the sibling resolves to `target/debug/lumen-server`
///    (i.e. running `cargo tauri dev`), prefer the workspace
///    `target/release/lumen-server` when present, since debug builds
///    are 5-10× slower (~8 tok/s vs ~70 tok/s on Gemma 4 26B).
///    Set `LUMEN_USE_DEBUG_SERVER=1` to bypass this override.
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
                // `target/debug/lumen-server` is the `cargo tauri dev`
                // sidecar — fast iteration but 5-10× slower than release
                // (LTO + opt-level=3). Surface release silently if the
                // dev'r has built one, unless explicitly opted out.
                if let Some(release) = prefer_release_over_dev_debug(&sibling) {
                    return Ok(release);
                }
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

/// If `sibling` is `<…>/target/debug/lumen-server` AND its sibling
/// `<…>/target/release/lumen-server` exists, return the release path.
/// Returns `None` otherwise — i.e. when the sibling lives in a real
/// bundle (`.app/Contents/Resources/`), a user-customized dir, or
/// when no release build is available. Honors `LUMEN_USE_DEBUG_SERVER=1`
/// for the rare developer who wants symbols + assertions live.
fn prefer_release_over_dev_debug(sibling: &Path) -> Option<PathBuf> {
    if std::env::var("LUMEN_USE_DEBUG_SERVER")
        .ok()
        .filter(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .is_some()
    {
        return None;
    }
    let parent = sibling.parent()?;
    if parent.file_name()? != "debug" {
        return None;
    }
    let target_dir = parent.parent()?;
    if target_dir.file_name()? != "target" {
        return None;
    }
    let release = target_dir.join("release").join("lumen-server");
    if release.exists() {
        Some(release)
    } else {
        None
    }
}
