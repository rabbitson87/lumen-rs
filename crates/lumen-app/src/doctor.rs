//! Environment diagnostics — `flutter doctor`-style preflight.
//!
//! Each check produces a stable `id` so the UI can map status → row and the
//! `doctor_fix` command can dispatch to a typed auto-fix routine. Checks are
//! intentionally cheap (<1 s total) so they can run on every app launch and
//! after every "Re-check" press without locking up the window.

use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use serde::Serialize;

use crate::config::PersistentConfig;
use crate::models;
use crate::server;

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum CheckStatus {
    Pass,
    Warn,
    Fail,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum OverallHealth {
    Healthy,  // all pass
    Degraded, // some warn, no fail
    Blocked,  // any fail
    /// Returned by the frontend as the initial state before `doctor_run`
    /// completes. The Rust side never constructs this — present for the
    /// serde enum surface.
    #[allow(dead_code)]
    Unknown,
}

#[derive(Debug, Clone, Serialize)]
pub struct CheckResult {
    pub id: &'static str,
    pub name: &'static str,
    pub status: CheckStatus,
    pub message: String,
    /// Longer human-readable detail, shown on hover or expand.
    pub detail: Option<String>,
    /// User-facing instructions for fixing the issue.
    pub fix_hint: Option<String>,
    /// Optional shell command the user could run themselves. Display-only —
    /// we do NOT execute this automatically.
    pub fix_command: Option<String>,
    /// If `Some`, the UI shows a "Fix" button that calls
    /// `doctor_fix(check_id=this)`. Implementations live in `try_fix`.
    pub fix_action: Option<&'static str>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DoctorReport {
    pub overall: OverallHealth,
    pub checks: Vec<CheckResult>,
}

pub async fn run_all(cfg: &PersistentConfig) -> DoctorReport {
    let mut checks = Vec::new();
    checks.push(check_os_version());
    checks.push(check_architecture());
    checks.push(check_ram());
    checks.push(check_disk_free(&cfg.models_dir));
    checks.push(check_models_dir(&cfg.models_dir));
    checks.push(check_server_binary(cfg.server_binary_path.as_deref()));
    checks.push(check_port_free(&cfg.server.host, cfg.server.port));
    checks.push(check_active_model(cfg));
    checks.push(check_huggingface().await);

    let overall = aggregate(&checks);
    DoctorReport { overall, checks }
}

fn aggregate(checks: &[CheckResult]) -> OverallHealth {
    let any_fail = checks.iter().any(|c| c.status == CheckStatus::Fail);
    let any_warn = checks.iter().any(|c| c.status == CheckStatus::Warn);
    if any_fail {
        OverallHealth::Blocked
    } else if any_warn {
        OverallHealth::Degraded
    } else {
        OverallHealth::Healthy
    }
}

// ── Individual checks ───────────────────────────────────────────────

fn check_os_version() -> CheckResult {
    let raw = Command::new("sw_vers")
        .arg("-productVersion")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default();

    let major: u32 = raw
        .split('.')
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let (status, message, hint) = if major >= 14 {
        (CheckStatus::Pass, format!("macOS {raw}"), None)
    } else if major >= 11 {
        (
            CheckStatus::Warn,
            format!("macOS {raw} — supported, but 14+ recommended"),
            Some("Apple Silicon MPS performance improvements landed in macOS 14 (Sonoma). Earlier versions work but leave a few % on the table.".into()),
        )
    } else if major > 0 {
        (
            CheckStatus::Fail,
            format!("macOS {raw} — unsupported"),
            Some("lumen requires macOS 11 (Big Sur) or newer for the Metal stack. Update via System Settings → General → Software Update.".into()),
        )
    } else {
        (
            CheckStatus::Warn,
            "macOS version unknown".into(),
            Some("Could not run `sw_vers`. If you're on macOS this is unusual — please report the issue.".into()),
        )
    };

    CheckResult {
        id: "os_version",
        name: "macOS version",
        status,
        message,
        detail: None,
        fix_hint: hint,
        fix_command: None,
        fix_action: None,
    }
}

fn check_architecture() -> CheckResult {
    let arch = std::env::consts::ARCH;
    let (status, message, hint) = match arch {
        "aarch64" => (
            CheckStatus::Pass,
            "Apple Silicon (arm64)".into(),
            None,
        ),
        "x86_64" => (
            CheckStatus::Warn,
            "Intel Mac (x86_64)".into(),
            Some(
                "Metal works on Intel Macs but Apple Silicon is 5-20× faster for inference \
                 and is the supported development target. Consider running smaller (<2B) models, \
                 or build/run on an M-series machine."
                    .into(),
            ),
        ),
        other => (
            CheckStatus::Fail,
            format!("unsupported architecture: {other}"),
            Some("lumen targets macOS Apple Silicon (and limited Intel Mac). Other platforms are not yet supported.".into()),
        ),
    };
    CheckResult {
        id: "architecture",
        name: "CPU architecture",
        status,
        message,
        detail: None,
        fix_hint: hint,
        fix_command: None,
        fix_action: None,
    }
}

fn check_ram() -> CheckResult {
    let bytes = sysctl_u64("hw.memsize").unwrap_or(0);
    let gb = bytes / (1024 * 1024 * 1024);

    let (status, hint) = if gb >= 24 {
        (CheckStatus::Pass, None)
    } else if gb >= 16 {
        (
            CheckStatus::Pass,
            Some(
                "16 GB is enough for 1.5-7B models. For 13B+ or Mixture-of-Experts models, \
                 24 GB+ is recommended."
                    .into(),
            ),
        )
    } else if gb >= 8 {
        (
            CheckStatus::Warn,
            Some(
                "8-16 GB is tight. Stick to <2B parameter models. Use 3-bit TurboQuant and disable \
                 the wired memory caps (Server card → Disable caps) if you OOM."
                    .into(),
            ),
        )
    } else {
        (
            CheckStatus::Fail,
            Some("Less than 8 GB RAM — lumen will OOM on almost any model. Free RAM or use a larger machine.".into()),
        )
    };
    CheckResult {
        id: "ram",
        name: "Total RAM",
        status,
        message: format!("{} GB", gb),
        detail: None,
        fix_hint: hint,
        fix_command: None,
        fix_action: None,
    }
}

fn check_disk_free(models_dir: &std::path::Path) -> CheckResult {
    // walk up to a real ancestor if models_dir doesn't exist yet.
    let probe = {
        let mut p = models_dir.to_path_buf();
        while !p.exists() {
            if !p.pop() {
                p = PathBuf::from("/");
                break;
            }
        }
        p
    };
    let free_bytes = statvfs_free(&probe).unwrap_or(0);
    let gb = free_bytes / (1024 * 1024 * 1024);

    let (status, hint) = if gb >= 50 {
        (CheckStatus::Pass, None)
    } else if gb >= 20 {
        (
            CheckStatus::Warn,
            Some(
                "Most modern weight sets are 5-30 GB each. Keep an eye on free space — \
                 the MODELS card shows per-model sizes."
                    .into(),
            ),
        )
    } else {
        (
            CheckStatus::Fail,
            Some(
                "Less than 20 GB free. Model downloads will fail mid-stream. Free up disk \
                 (or change the weights path in SERVER → Weights dir)."
                    .into(),
            ),
        )
    };
    CheckResult {
        id: "disk_free",
        name: "Free disk space",
        status,
        message: format!("{} GB free at {}", gb, probe.display()),
        detail: None,
        fix_hint: hint,
        fix_command: None,
        fix_action: None,
    }
}

fn check_models_dir(models_dir: &std::path::Path) -> CheckResult {
    if models_dir.exists() {
        // Test writability with a temp file.
        let probe = models_dir.join(".lumen-write-test");
        match std::fs::write(&probe, b"ok") {
            Ok(_) => {
                let _ = std::fs::remove_file(&probe);
                CheckResult {
                    id: "models_dir",
                    name: "Models directory",
                    status: CheckStatus::Pass,
                    message: format!("writable: {}", models_dir.display()),
                    detail: None,
                    fix_hint: None,
                    fix_command: None,
                    fix_action: None,
                }
            }
            Err(e) => CheckResult {
                id: "models_dir",
                name: "Models directory",
                status: CheckStatus::Fail,
                message: format!("not writable: {}", models_dir.display()),
                detail: Some(e.to_string()),
                fix_hint: Some(
                    "Fix permissions on the models directory, or change it in SERVER → Weights dir.".into(),
                ),
                fix_command: Some(format!("chmod -R u+w {}", models_dir.display())),
                fix_action: None,
            },
        }
    } else {
        CheckResult {
            id: "models_dir",
            name: "Models directory",
            status: CheckStatus::Warn,
            message: format!("missing: {}", models_dir.display()),
            detail: None,
            fix_hint: Some("Click Fix to create the directory.".into()),
            fix_command: Some(format!("mkdir -p {}", models_dir.display())),
            fix_action: Some("create_models_dir"),
        }
    }
}

fn check_server_binary(explicit: Option<&std::path::Path>) -> CheckResult {
    match server::resolve_binary_public(explicit) {
        Ok(path) => {
            // Check executable bit.
            let executable = {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    std::fs::metadata(&path)
                        .map(|m| m.permissions().mode() & 0o111 != 0)
                        .unwrap_or(false)
                }
                #[cfg(not(unix))]
                { true }
            };
            if executable {
                CheckResult {
                    id: "server_binary",
                    name: "lumen-server binary",
                    status: CheckStatus::Pass,
                    message: format!("found: {}", path.display()),
                    detail: None,
                    fix_hint: None,
                    fix_command: None,
                    fix_action: None,
                }
            } else {
                CheckResult {
                    id: "server_binary",
                    name: "lumen-server binary",
                    status: CheckStatus::Fail,
                    message: format!("not executable: {}", path.display()),
                    detail: None,
                    fix_hint: Some("Set the executable bit on the binary, or rebuild it.".into()),
                    fix_command: Some(format!("chmod +x {}", path.display())),
                    fix_action: None,
                }
            }
        }
        Err(e) => CheckResult {
            id: "server_binary",
            name: "lumen-server binary",
            status: CheckStatus::Fail,
            message: "not found".into(),
            detail: Some(e.to_string()),
            fix_hint: Some(
                "Build the inference server from source, or set SERVER → server_binary_path in config.toml.".into(),
            ),
            fix_command: Some("cargo build -p lumen-server --release".into()),
            fix_action: None,
        },
    }
}

fn check_port_free(host: &str, port: u16) -> CheckResult {
    use std::net::TcpListener;
    let bind_addr = if host == "0.0.0.0" || host == "::" {
        format!("127.0.0.1:{port}")
    } else {
        format!("{host}:{port}")
    };
    match TcpListener::bind(&bind_addr) {
        Ok(listener) => {
            drop(listener);
            CheckResult {
                id: "port_free",
                name: "Server port",
                status: CheckStatus::Pass,
                message: format!("port {port} available"),
                detail: None,
                fix_hint: None,
                fix_command: None,
                fix_action: None,
            }
        }
        Err(e) => CheckResult {
            id: "port_free",
            name: "Server port",
            status: CheckStatus::Warn,
            message: format!("port {port} in use"),
            detail: Some(e.to_string()),
            fix_hint: Some(
                "Another process is bound to this port. Change PORT in the SERVER card, or stop the other process.".into(),
            ),
            fix_command: Some(format!("lsof -i :{port}")),
            fix_action: None,
        },
    }
}

fn check_active_model(cfg: &PersistentConfig) -> CheckResult {
    match &cfg.active_model {
        None => CheckResult {
            id: "active_model",
            name: "Active model",
            status: CheckStatus::Warn,
            message: "no model selected".into(),
            detail: None,
            fix_hint: Some(
                "Pick a model in the ACTIVE MODEL card, or download one from HF Hub via the MODELS card.".into(),
            ),
            fix_command: None,
            fix_action: None,
        },
        Some(id) => {
            // Doctor doesn't need supported-flag enrichment — pass an empty
            // catalog so we don't have to thread it through the report.
            let empty_cat = crate::catalog::Catalog::default();
            let entries = models::scan_local(&cfg.models_dir, &empty_cat).unwrap_or_default();
            let entry = entries.iter().find(|m| &m.id == id);
            match entry {
                Some(m) if m.ready => CheckResult {
                    id: "active_model",
                    name: "Active model",
                    status: CheckStatus::Pass,
                    message: format!("{} ready", id),
                    detail: None,
                    fix_hint: None,
                    fix_command: None,
                    fix_action: None,
                },
                Some(_) => CheckResult {
                    id: "active_model",
                    name: "Active model",
                    status: CheckStatus::Warn,
                    message: format!("{} on disk but incomplete", id),
                    detail: Some(
                        "Required files (config.json + at least one safetensors/gguf shard) are missing.".into(),
                    ),
                    fix_hint: Some("Re-download the model from the MODELS card, or remove + re-add.".into()),
                    fix_command: None,
                    fix_action: None,
                },
                None => CheckResult {
                    id: "active_model",
                    name: "Active model",
                    status: CheckStatus::Fail,
                    message: format!("{} not on disk", id),
                    detail: None,
                    fix_hint: Some("Download the model from the MODELS card, or pick a different active model.".into()),
                    fix_command: None,
                    fix_action: None,
                },
            }
        }
    }
}

async fn check_huggingface() -> CheckResult {
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(4))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            return CheckResult {
                id: "huggingface",
                name: "Hugging Face network",
                status: CheckStatus::Fail,
                message: "http client init failed".into(),
                detail: Some(e.to_string()),
                fix_hint: None,
                fix_command: None,
                fix_action: None,
            };
        }
    };
    match client.head("https://huggingface.co/").send().await {
        Ok(r) if r.status().is_success() || r.status().is_redirection() => CheckResult {
            id: "huggingface",
            name: "Hugging Face network",
            status: CheckStatus::Pass,
            message: format!("reachable ({})", r.status().as_u16()),
            detail: None,
            fix_hint: None,
            fix_command: None,
            fix_action: None,
        },
        Ok(r) => CheckResult {
            id: "huggingface",
            name: "Hugging Face network",
            status: CheckStatus::Warn,
            message: format!("unexpected status {}", r.status().as_u16()),
            detail: None,
            fix_hint: Some(
                "huggingface.co responded but not with 2xx/3xx. Service may be degraded — downloads may fail.".into(),
            ),
            fix_command: None,
            fix_action: Some("recheck"),
        },
        Err(e) => CheckResult {
            id: "huggingface",
            name: "Hugging Face network",
            status: CheckStatus::Fail,
            message: "unreachable".into(),
            detail: Some(e.to_string()),
            fix_hint: Some(
                "Model downloads via HF Hub will fail. Check your internet connection, VPN, or proxy.".into(),
            ),
            fix_command: None,
            fix_action: Some("recheck"),
        },
    }
}

// ── Auto-fix dispatch ───────────────────────────────────────────────

pub async fn try_fix(action: &str, cfg: &PersistentConfig) -> Result<String, String> {
    match action {
        "create_models_dir" => std::fs::create_dir_all(&cfg.models_dir)
            .map(|_| format!("created {}", cfg.models_dir.display()))
            .map_err(|e| e.to_string()),
        "recheck" => Ok("triggered re-check".into()),
        other => Err(format!("unknown fix action: {other}")),
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

fn sysctl_u64(key: &str) -> Option<u64> {
    let out = Command::new("sysctl").args(["-n", key]).output().ok()?;
    let s = String::from_utf8(out.stdout).ok()?;
    s.trim().parse().ok()
}

#[cfg(unix)]
fn statvfs_free(path: &std::path::Path) -> Option<u64> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let c = CString::new(path.as_os_str().as_bytes()).ok()?;
    unsafe {
        let mut buf: libc::statvfs = std::mem::zeroed();
        if libc::statvfs(c.as_ptr(), &mut buf) != 0 {
            return None;
        }
        // Available to non-root processes.
        Some(buf.f_bavail as u64 * buf.f_frsize as u64)
    }
}
#[cfg(not(unix))]
fn statvfs_free(_path: &std::path::Path) -> Option<u64> {
    None
}
