//! Environment diagnostics — `flutter doctor`-style preflight.
//!
//! Each check produces a stable `id` so the UI can map status → row and the
//! `doctor_fix` command can dispatch to a typed auto-fix routine. Checks are
//! intentionally cheap (<1 s total) so they can run on every app launch and
//! after every "Re-check" press without locking up the window.
//!
//! The one exception is Flash-Next's SSD read test (`probe_flash_next_ssd`):
//! it reads ~1 GB, so it never runs with the rest — only when the user asks.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use serde::Serialize;

use crate::config::PersistentConfig;
use crate::flash_next::{
    self, LaunchPlan, PackState, ProbeEnd, ProbePlan, ProbeResult, ReadVerdict,
};
use crate::models;
use crate::server;
use crate::sysinfo;

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
    let mut checks = vec![
        check_os_version(),
        check_architecture(),
        check_ram(),
        check_disk_free(&cfg.models_dir),
        check_models_dir(&cfg.models_dir),
        check_server_binary(cfg.server_binary_path.as_deref()),
        check_port_free(&cfg.server.host, cfg.server.port),
        check_active_model(cfg),
    ];
    checks.extend(flash_next_checks(cfg));
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

// ── Flash-Next ──────────────────────────────────────────────────────
//
// These rows exist only when a complete copy of Qwen 3.8 Flash-Next is installed. The cheap ones
// run with every other check; the SSD read test is separate (`probe_flash_next_ssd`).

/// The pack build writes 68 GB; leave a little over for the filesystem.
const DISK_MARGIN_BYTES: u64 = 2_000_000_000;
/// Expert pool assumed for the SSD verdict when the machine's RAM cannot be read.
const FALLBACK_POOL_GB: f64 = 10.0;

fn row(id: &'static str, name: &'static str, status: CheckStatus, message: String) -> CheckResult {
    CheckResult {
        id,
        name,
        status,
        message,
        detail: None,
        fix_hint: None,
        fix_command: None,
        fix_action: None,
    }
}

/// Decimal GB, the unit the model's sizes are quoted in (97 GB, 68 GB).
fn gb(bytes: u64) -> f64 {
    bytes as f64 / 1e9
}

/// `path` as one shell word.
fn sh_quote(path: &Path) -> String {
    let s = path.display().to_string();
    if s.chars()
        .all(|c| c.is_ascii_alphanumeric() || "/._-~".contains(c))
    {
        s
    } else {
        format!("'{}'", s.replace('\'', "'\\''"))
    }
}

fn fmt_tps(tps: f64) -> String {
    if tps >= 10.0 {
        format!("{tps:.0}")
    } else {
        format!("{tps:.1}")
    }
}

fn flash_next_checks(cfg: &PersistentConfig) -> Vec<CheckResult> {
    let Some(dir) = flash_next::find_model_dir(cfg) else {
        return Vec::new();
    };
    // What a launch carries: the Flash-Next settings, replaced by an env override of the same name.
    let launch_env = flash_next::effective_env(cfg);
    let env = |key: &str| launch_env.get(key).map(|v| v.trim().to_ascii_lowercase());
    let pack = flash_next::pack_state(&dir);
    let mut out = Vec::new();
    if let Some(ram) = sysinfo::total_ram_bytes() {
        out.push(flash_next_ram_check(
            ram,
            &flash_next::launch_plan(cfg, ram),
        ));
    }
    out.push(flash_next_disk_check(
        &dir,
        statvfs_free(&dir),
        pack,
        env("LUMEN_MOE_PACK").as_deref() == Some("build"),
    ));
    out.push(flash_next_pack_check(&dir, pack));
    out.push(flash_next_mtp_check(
        &dir,
        flash_next::has_mtp_head(&dir),
        env("LUMEN_QWEN4_MTP").as_deref(),
    ));
    out
}

/// Whether the memory plan the app's launch hands the server leaves room for the model. A bigger
/// Max context (the longest prompt the plan holds KV for) shrinks the expert pool, and past a
/// point leaves none.
fn flash_next_ram_check(ram_bytes: u64, launch: &LaunchPlan) -> CheckResult {
    const NAME: &str = "Flash-Next memory";
    let ram_gb = ram_bytes >> 30;
    let usable = flash_next::usable_bytes(ram_bytes, launch.memory_limit);
    let plan = flash_next::plan_memory(usable, launch.chunk, launch.ctx_tokens);
    if plan.fits() {
        let mut detail = String::from(
            "The expert pool is what the GPU's share of RAM leaves after the 3.6 GB of resident \
             weights and a prefill's reserve: buffers, the chunk's activations, the KV cache of \
             the longest prompt (60 KB per token) and attention scores. A longer prompt cap \
             shrinks it; a bigger pool means fewer SSD reads per token.",
        );
        if plan.pool < flash_next::POOL_PREFETCH_BYTES {
            detail.push_str(
                " With less than 4.2 GB the pool has room for one staging region only, so \
                 prefill cannot read the next layer while it computes the current one.",
            );
        }
        return CheckResult {
            detail: Some(detail),
            ..row(
                "flash_next_ram",
                NAME,
                CheckStatus::Pass,
                format!(
                    "{ram_gb} GB RAM: expert pool ~{:.1} GB at {} tokens",
                    gb(plan.pool),
                    plan.ctx_tokens
                ),
            )
        };
    }
    let longest = flash_next::max_fitting_context(usable, launch.chunk);
    if longest > 0 {
        return CheckResult {
            fix_hint: Some(
                "The KV cache of the longest prompt (60 KB per token) comes out of the GPU's \
                 share of RAM. The app sends LUMEN_MAX_PROMPT_TOKENS (16,384 tokens, 8,192 on a \
                 tight Mac) unless you set it, lowered to Max context. Lower it (the Flash-Next \
                 memory calculator's Apply writes it) or the prefill chunk, or raise the memory \
                 limit if it is below your RAM."
                    .into(),
            ),
            ..row(
                "flash_next_ram",
                NAME,
                CheckStatus::Warn,
                format!(
                    "{ram_gb} GB RAM: a {}-token context does not fit, up to {longest} do",
                    plan.ctx_tokens
                ),
            )
        };
    }
    CheckResult {
        fix_hint: Some(
            "Flash-Next needs its 3.6 GB of resident weights, an expert pool of at least 2.8 GB \
             and a prefill reserve inside the GPU's share of RAM (two thirds up to 32 GB, three \
             quarters above; the memory limit lowers it). Below that the model cannot load."
                .into(),
        ),
        ..row(
            "flash_next_ram",
            NAME,
            CheckStatus::Warn,
            format!(
                "{ram_gb} GB RAM, Flash-Next needs ~{} GB",
                flash_next::min_ram_gib()
            ),
        )
    }
}

fn flash_next_disk_check(
    dir: &Path,
    free: Option<u64>,
    pack: PackState,
    build_requested: bool,
) -> CheckResult {
    const NAME: &str = "Flash-Next disk";
    let Some(free) = free else {
        return row(
            "flash_next_disk",
            NAME,
            CheckStatus::Warn,
            "free space unknown".into(),
        );
    };
    if let PackState::Complete(_) = pack {
        return row(
            "flash_next_disk",
            NAME,
            CheckStatus::Pass,
            format!("{:.0} GB free, expert pack already built", gb(free)),
        );
    }
    // Rebuilding over an unfinished pack deletes it first.
    let usable = match pack {
        PackState::Incomplete(len) => free + len,
        _ => free,
    };
    let room = usable >= flash_next::PACK_BYTES + DISK_MARGIN_BYTES;
    let message = if room {
        format!(
            "{:.0} GB free, room for the {:.0} GB expert pack",
            gb(free),
            gb(flash_next::PACK_BYTES)
        )
    } else {
        format!(
            "{:.0} GB free, not enough for the {:.0} GB expert pack",
            gb(free),
            gb(flash_next::PACK_BYTES)
        )
    };
    // The pack is an optional speed-up, so having no room for it is not a fault, unless
    // the user already asked the server to build it.
    let status = if !room && build_requested {
        CheckStatus::Warn
    } else {
        CheckStatus::Pass
    };
    CheckResult {
        fix_hint: Some(if room {
            "An expert pack makes decode about 7-15% faster. Build it once with the command below, or \
             set LUMEN_MOE_PACK=build and the server builds it on its next start."
                .into()
        } else {
            "The expert pack is optional. To use it, free about 70 GB on this volume, then build \
             it with the command below."
                .into()
        }),
        fix_command: Some(format!("lumen-server --pack-experts {}", sh_quote(dir))),
        ..row("flash_next_disk", NAME, status, message)
    }
}

fn flash_next_pack_check(dir: &Path, pack: PackState) -> CheckResult {
    const NAME: &str = "Flash-Next expert pack";
    match pack {
        PackState::Complete(len) => row(
            "flash_next_pack",
            NAME,
            CheckStatus::Pass,
            format!("experts.lpack present ({:.0} GB)", gb(len)),
        ),
        PackState::Absent => CheckResult {
            detail: Some(
                "Without the pack the server reads each expert as nine scattered ranges of the \
                 weight shards. The pack stores the same bytes as one record per expert."
                    .into(),
            ),
            ..row(
                "flash_next_pack",
                NAME,
                CheckStatus::Pass,
                "no expert pack (optional)".into(),
            )
        },
        PackState::Incomplete(len) => CheckResult {
            fix_hint: Some(
                "The server ignores an unfinished pack and reads the shards instead. Delete it \
                 and build it again."
                    .into(),
            ),
            fix_command: Some(format!(
                "rm {} && lumen-server --pack-experts {}",
                sh_quote(&dir.join("experts.lpack")),
                sh_quote(dir)
            )),
            ..row(
                "flash_next_pack",
                NAME,
                CheckStatus::Warn,
                format!(
                    "experts.lpack incomplete ({:.0} GB of {:.0} GB)",
                    gb(len),
                    gb(flash_next::PACK_BYTES)
                ),
            )
        },
    }
}

/// `mode` is the lower-cased `LUMEN_QWEN4_MTP` a launch carries (the setting or an override), if
/// any. The draft head is off unless chosen (`auto`: use it when installed; `on`: the server
/// refuses to start without it), so an installed head that is not in use is the normal state, not
/// a warning.
fn flash_next_mtp_check(dir: &Path, present: bool, mode: Option<&str>) -> CheckResult {
    const NAME: &str = "Flash-Next draft head";
    let required = matches!(mode, Some("on" | "1" | "true"));
    let in_use = required || matches!(mode, Some("auto"));
    let fetch = format!("python3 scripts/qwen4_mtp_fetch.py {}", sh_quote(dir));
    match (present, in_use, required) {
        (true, false, _) => row(
            "flash_next_mtp",
            NAME,
            CheckStatus::Pass,
            "draft head installed, not used (off by default)".into(),
        ),
        (true, true, _) => row(
            "flash_next_mtp",
            NAME,
            CheckStatus::Pass,
            "draft head present (mtp.safetensors)".into(),
        ),
        (false, _, true) => CheckResult {
            fix_hint: Some(
                "The server refuses to start with LUMEN_QWEN4_MTP=on and no draft head. Download \
                 the head (1.5 GB) with the command below, or remove the override."
                    .into(),
            ),
            fix_command: Some(fetch),
            ..row(
                "flash_next_mtp",
                NAME,
                CheckStatus::Fail,
                "LUMEN_QWEN4_MTP=on but mtp.safetensors is missing".into(),
            )
        },
        (false, _, false) => CheckResult {
            fix_hint: Some(
                "The draft head (1.5 GB, downloaded once) is off by default: measured on this \
                 Mac it gained at most 4% and often lost 1-4%, and its text is not bit-identical \
                 to plain decoding. To try it, download the head with the command below and set \
                 the draft head to Auto in the Flash-Next card."
                    .into(),
            ),
            fix_command: Some(fetch),
            ..row(
                "flash_next_mtp",
                NAME,
                CheckStatus::Pass,
                "no draft head (optional)".into(),
            )
        },
    }
}

/// What the SSD read test measured, in the form the app stores (`MeasuredHardware`).
#[derive(Debug, Clone, PartialEq)]
pub struct SsdReading {
    pub random_gbps: f64,
    pub sequential_gbps: Option<f64>,
    /// The model folder the shard was read from.
    pub volume: String,
}

/// Flash-Next's SSD read test: ~1 GB read from one weight shard the way the server reads
/// experts, ending after a few seconds at the latest. Run only when the user asks. `None`
/// when `cancel` was set. Besides the row, it returns the numbers it measured (when it got a
/// random-read speed), for the caller to store. With this Mac's memory bandwidth measured, the
/// row also says what decode speed that buys.
pub fn probe_flash_next_ssd(
    cfg: &PersistentConfig,
    cancel: &AtomicBool,
) -> Option<(CheckResult, Option<SsdReading>)> {
    const NAME: &str = "Flash-Next SSD";
    let failed = |message: &str, detail: Option<String>| {
        Some((
            CheckResult {
                detail,
                ..row("flash_next_ssd", NAME, CheckStatus::Warn, message.into())
            },
            None,
        ))
    };
    let Some(dir) = flash_next::find_model_dir(cfg) else {
        return failed("Flash-Next is not installed", None);
    };
    let Some(shard) = flash_next::probe_shard(&dir) else {
        return failed("no weight shard to read", None);
    };
    let res = match flash_next::probe_read_speed(&shard, &ProbePlan::DEFAULT, cancel) {
        Ok(r) => r,
        Err(e) => return failed("read failed", Some(format!("{}: {e}", shard.display()))),
    };
    if res.end == ProbeEnd::Cancelled {
        return None;
    }
    let shard_name = shard
        .file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    // The pool this Mac's launch would get, and whether the server would read the pack.
    let pool_gb = sysinfo::total_ram_bytes().map_or(FALLBACK_POOL_GB, |ram| {
        let launch = flash_next::launch_plan(cfg, ram);
        let usable = flash_next::usable_bytes(ram, launch.memory_limit);
        gb(flash_next::plan_memory(usable, launch.chunk, launch.ctx_tokens).pool_bytes())
    });
    let pack_off = flash_next::effective_env(cfg)
        .get("LUMEN_MOE_PACK")
        .is_some_and(|v| v.trim().eq_ignore_ascii_case("off"));
    let pack = matches!(flash_next::pack_state(&dir), PackState::Complete(_)) && !pack_off;
    let check = flash_next_ssd_check(
        &shard_name,
        &res,
        pool_gb,
        pack,
        cfg.measured_hardware.memory(),
    );
    let reading = res.rand_gbps().map(|random_gbps| SsdReading {
        random_gbps,
        sequential_gbps: res.seq_gbps(),
        volume: dir.display().to_string(),
    });
    Some((check, reading))
}

/// What the test reports when the volume did not answer within the hard time limit.
pub fn flash_next_ssd_stalled() -> CheckResult {
    CheckResult {
        fix_hint: Some(
            "A read took longer than the test's time limit. The weights may sit on a network or \
             sleeping external drive; decoding reads from them constantly, so keep the model on \
             the Mac's internal SSD."
                .into(),
        ),
        ..row(
            "flash_next_ssd",
            "Flash-Next SSD",
            CheckStatus::Warn,
            "the volume did not answer in time".into(),
        )
    }
}

/// The SSD row. The verdict is the disk's alone (`slow` under 2 GB/s of random reads); a decode
/// speed is promised only with this Mac's measured memory bandwidth (`memory_gbps`), because
/// the speed depends on both.
fn flash_next_ssd_check(
    shard: &str,
    res: &ProbeResult,
    pool_gb: f64,
    pack: bool,
    memory_gbps: Option<f64>,
) -> CheckResult {
    const NAME: &str = "Flash-Next SSD";
    let Some(random) = res.rand_gbps() else {
        return row(
            "flash_next_ssd",
            NAME,
            CheckStatus::Warn,
            "could not measure random reads".into(),
        );
    };
    let sequential = res.seq_gbps().unwrap_or(0.0);
    let tps = memory_gbps
        .and_then(|memory| flash_next::decode_tps(pool_gb, memory, random, pack))
        .map(fmt_tps);
    let slow = flash_next::read_verdict(random) == ReadVerdict::Slow;
    let verdict = match (slow, &tps) {
        (true, Some(tps)) => format!("slow, ~{tps} tok/s"),
        (true, None) => "slow".to_string(),
        (false, Some(tps)) => format!("fast enough for ~{tps} tok/s"),
        (false, None) => "fast enough".to_string(),
    };
    let speed = match (memory_gbps, &tps) {
        (Some(memory), Some(_)) => format!(
            " Decode speed is estimated from this volume's random reads and this Mac's memory \
             bandwidth ({memory:.0} GB/s, measured), for an expert pool of ~{pool_gb:.1} GB, \
             what the memory plan leaves on this Mac.{}",
            if pack {
                " The expert pack's read efficiency is used."
            } else {
                ""
            }
        ),
        _ => " Measure this Mac's memory bandwidth in the Flash-Next card (Tuning tab) to get a \
               decode-speed estimate."
            .to_string(),
    };
    let early = if res.end == ProbeEnd::TimedOut {
        " The test hit its time limit and stopped early; the speeds come from what it had read."
    } else {
        ""
    };
    CheckResult {
        detail: Some(format!(
            "Read {:.0} MB of {shard} with the page cache bypassed: {:.0} MB front to back, then \
             {:.0} MB as {} random {:.1} MB reads on 4 threads, the way the server fetches \
             experts.{speed}{early}",
            (res.seq_bytes + res.rand_bytes) as f64 / 1e6,
            res.seq_bytes as f64 / 1e6,
            res.rand_bytes as f64 / 1e6,
            res.rand_reads,
            flash_next::EXPERT_BYTES as f64 / 1e6,
        )),
        fix_hint: slow.then(|| {
            "Decode speed is set by how fast the SSD serves expert reads. Keep the model on the \
             Mac's internal SSD (external and network drives are usually below 2 GB/s), or give \
             the expert pool more memory."
                .into()
        }),
        ..row(
            "flash_next_ssd",
            NAME,
            if slow {
                CheckStatus::Warn
            } else {
                CheckStatus::Pass
            },
            format!("random {random:.1} GB/s (sequential {sequential:.1} GB/s): {verdict}"),
        )
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
pub(crate) fn statvfs_free(path: &std::path::Path) -> Option<u64> {
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
pub(crate) fn statvfs_free(_path: &std::path::Path) -> Option<u64> {
    None
}

#[cfg(test)]
mod flash_next_tests {
    use super::*;
    use crate::flash_next::{PACK_BYTES, PACK_COMPLETE_MIN_BYTES};

    const GIB: u64 = 1 << 30;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("lumen-doctor-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A models directory holding a complete-looking Flash-Next of `shard_len` bytes.
    fn config_with_model(name: &str, shard_len: usize) -> (PersistentConfig, PathBuf) {
        let models = temp_dir(name);
        let dir = models.join("pipenetwork--Qwen3.8-Flash-Next-MLX-4bit");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        std::fs::write(dir.join("model-00001.safetensors"), vec![7u8; shard_len]).unwrap();
        let cfg = PersistentConfig {
            models_dir: models,
            ..PersistentConfig::default()
        };
        (cfg, dir)
    }

    fn result(seq_gbps: f64, rand_gbps: f64, end: ProbeEnd) -> ProbeResult {
        // One second of each, so the byte counts are the speeds.
        ProbeResult {
            seq_bytes: (seq_gbps * 1e9) as u64,
            seq_secs: 1.0,
            rand_bytes: (rand_gbps * 1e9) as u64,
            rand_secs: 1.0,
            rand_reads: 100,
            end,
        }
    }

    #[test]
    fn shell_words_are_quoted_only_when_they_need_to_be() {
        assert_eq!(sh_quote(Path::new("/a/b-c_d.e/~f")), "/a/b-c_d.e/~f");
        assert_eq!(sh_quote(Path::new("/a b/c")), "'/a b/c'");
        assert_eq!(sh_quote(Path::new("/it's")), "'/it'\\''s'");
    }

    /// What a launch with the memory plan's own defaults hands the server.
    fn default_launch() -> LaunchPlan {
        LaunchPlan {
            memory_limit: None,
            chunk: None,
            ctx_tokens: None,
        }
    }

    #[test]
    fn a_16_gb_mac_is_told_its_pool_and_an_8_gb_one_what_it_needs() {
        let ok = flash_next_ram_check(16 * GIB, &default_launch());
        assert_eq!(ok.status, CheckStatus::Pass);
        // 16 GB is a tight machine: 8K prompts, 512-token chunks, a ~4.6 GB pool.
        assert_eq!(ok.message, "16 GB RAM: expert pool ~4.6 GB at 8192 tokens");
        // 4.6 GB is above the 4.2 GB a second staging region needs.
        assert!(!ok.detail.unwrap().contains("one staging region only"));

        let short = flash_next_ram_check(8 * GIB, &default_launch());
        assert_eq!(short.status, CheckStatus::Warn);
        assert_eq!(short.message, "8 GB RAM, Flash-Next needs ~14 GB");
        assert!(short.fix_hint.is_some());
    }

    #[test]
    fn a_pool_under_4_2_gb_gets_the_one_staging_region_note_and_still_passes() {
        // A 12 GB Mac: pool ~2.1 GB is under the floor, so it does not fit at all ...
        assert_eq!(
            flash_next_ram_check(12 * GIB, &default_launch()).status,
            CheckStatus::Warn
        );
        // ... and one with a pool between the floor and 4.2 GB (an 8 GiB memory limit on a
        // big Mac: ~3.1 GB) runs, with the note.
        let launch = LaunchPlan {
            memory_limit: Some(8 * GIB),
            ..default_launch()
        };
        let c = flash_next_ram_check(36 * GIB, &launch);
        assert_eq!(c.status, CheckStatus::Pass, "{}", c.message);
        assert!(c.detail.unwrap().contains("one staging region only"));
    }

    #[test]
    fn a_long_prompt_cap_can_leave_no_room_for_the_pool() {
        // With LUMEN_MAX_PROMPT_TOKENS at 81,920 the KV for it alone is 4.9 GB, so a 16 GB
        // Mac no longer fits, though a shorter prompt would.
        let launch = LaunchPlan {
            ctx_tokens: Some(81_920),
            ..default_launch()
        };
        let c = flash_next_ram_check(16 * GIB, &launch);
        assert_eq!(c.status, CheckStatus::Warn);
        assert_eq!(
            c.message,
            "16 GB RAM: a 81920-token context does not fit, up to 34879 do"
        );
        let hint = c.fix_hint.unwrap();
        assert!(hint.contains("LUMEN_MAX_PROMPT_TOKENS"));
        assert!(hint.contains("comes out of the GPU's share of RAM"));
        // On a 36 GB Mac the same prompt fits, with a smaller pool than the default's.
        let big = flash_next_ram_check(36 * GIB, &launch);
        assert_eq!(
            big.message,
            "36 GB RAM: expert pool ~14.3 GB at 81920 tokens"
        );
    }

    /// The app sends a Flash-Next server its own `LUMEN_MAX_PROMPT_TOKENS` (the machine's
    /// default), so the Prefill card (40,960) and the Max context card (81,920) of a default
    /// config no longer decide the plan: a 16 GB Mac gets its 8K plan and fits.
    #[test]
    fn the_app_defaults_plan_the_machines_own_prompt_and_an_override_changes_it() {
        let mut cfg = PersistentConfig::default();
        cfg.context.max = 81_920;
        cfg.context.prefill = 40_960;
        cfg.server.memory_limit_gb = Some(64);
        cfg.server.disable_wired_limit = false;
        let ram = 16 * GIB;
        let c = flash_next_ram_check(ram, &flash_next::launch_plan(&cfg, ram));
        assert_eq!(c.status, CheckStatus::Pass);
        assert_eq!(c.message, "16 GB RAM: expert pool ~4.6 GB at 8192 tokens");
        // A user-set prompt cap wins, and this Mac cannot hold 40,960 tokens.
        cfg.env_overrides
            .insert("LUMEN_MAX_PROMPT_TOKENS".into(), "40960".into());
        let big = flash_next_ram_check(ram, &flash_next::launch_plan(&cfg, ram));
        assert_eq!(big.status, CheckStatus::Warn);
        assert_eq!(
            big.message,
            "16 GB RAM: a 40960-token context does not fit, up to 34879 do"
        );
        assert!(
            big.fix_hint
                .unwrap()
                .contains("The app sends LUMEN_MAX_PROMPT_TOKENS")
        );
    }

    #[test]
    fn room_for_the_pack_offers_the_build_command() {
        let dir = Path::new("/models/flash next");
        let c = flash_next_disk_check(dir, Some(200_000_000_000), PackState::Absent, false);
        assert_eq!(c.status, CheckStatus::Pass);
        assert_eq!(c.message, "200 GB free, room for the 68 GB expert pack");
        assert_eq!(
            c.fix_command.as_deref(),
            Some("lumen-server --pack-experts '/models/flash next'")
        );
    }

    #[test]
    fn no_room_is_a_fault_only_when_the_pack_was_asked_for() {
        let dir = Path::new("/m");
        let free = Some(40_000_000_000);
        let quiet = flash_next_disk_check(dir, free, PackState::Absent, false);
        assert_eq!(quiet.status, CheckStatus::Pass);
        assert_eq!(
            quiet.message,
            "40 GB free, not enough for the 68 GB expert pack"
        );
        let asked = flash_next_disk_check(dir, free, PackState::Absent, true);
        assert_eq!(asked.status, CheckStatus::Warn);
    }

    #[test]
    fn rebuilding_over_an_unfinished_pack_counts_the_space_it_frees() {
        let dir = Path::new("/m");
        let partial = PackState::Incomplete(60_000_000_000 - 1);
        // 20 GB free would not hold a new pack, 20 GB plus the 60 GB to be deleted does.
        let c = flash_next_disk_check(dir, Some(20_000_000_000), partial, true);
        assert_eq!(c.status, CheckStatus::Pass);
        assert!(c.message.contains("room for"), "{}", c.message);
    }

    #[test]
    fn a_finished_pack_needs_no_room() {
        let c = flash_next_disk_check(
            Path::new("/m"),
            Some(1_000_000_000),
            PackState::Complete(PACK_BYTES),
            true,
        );
        assert_eq!(c.status, CheckStatus::Pass);
        assert_eq!(c.message, "1 GB free, expert pack already built");
        assert!(
            flash_next_disk_check(Path::new("/m"), None, PackState::Absent, false).status
                == CheckStatus::Warn
        );
    }

    #[test]
    fn the_pack_row_tells_complete_absent_and_unfinished_apart() {
        let dir = Path::new("/m");
        let done = flash_next_pack_check(dir, PackState::Complete(PACK_BYTES));
        assert_eq!(done.status, CheckStatus::Pass);
        assert_eq!(done.message, "experts.lpack present (68 GB)");

        let none = flash_next_pack_check(dir, PackState::Absent);
        assert_eq!(
            (none.status, none.message.as_str()),
            (CheckStatus::Pass, "no expert pack (optional)")
        );

        let cut = flash_next_pack_check(dir, PackState::Incomplete(PACK_COMPLETE_MIN_BYTES - 1));
        assert_eq!(cut.status, CheckStatus::Warn);
        assert_eq!(cut.message, "experts.lpack incomplete (60 GB of 68 GB)");
        assert_eq!(
            cut.fix_command.as_deref(),
            Some("rm /m/experts.lpack && lumen-server --pack-experts /m")
        );
    }

    #[test]
    fn a_draft_head_is_optional_unless_the_user_demanded_it() {
        let dir = Path::new("/m");
        let status = |present, mode| flash_next_mtp_check(dir, present, mode).status;
        // Off by default: an installed head that is not used is no warning.
        assert_eq!(status(true, None), CheckStatus::Pass);
        assert_eq!(status(true, Some("off")), CheckStatus::Pass);
        assert_eq!(status(true, Some("auto")), CheckStatus::Pass);
        assert_eq!(status(false, None), CheckStatus::Pass);
        assert_eq!(status(false, Some("auto")), CheckStatus::Pass);
        assert_eq!(status(false, Some("off")), CheckStatus::Pass);
        // The server refuses to start in this one.
        let must = flash_next_mtp_check(dir, false, Some("on"));
        assert_eq!(must.status, CheckStatus::Fail);
        assert_eq!(
            must.fix_command.as_deref(),
            Some("python3 scripts/qwen4_mtp_fetch.py /m")
        );
        assert_eq!(status(false, Some("1")), CheckStatus::Fail);
    }

    /// The row's text is a contract with `DoctorPanel.svelte` (it matches the English). An
    /// installed head the default leaves unused reads as "not used", and the used cases keep the
    /// present/missing wording.
    #[test]
    fn the_draft_head_row_says_not_used_by_default_and_present_when_it_is() {
        let dir = Path::new("/m");
        let msg = |present, mode| flash_next_mtp_check(dir, present, mode).message;
        for mode in [None, Some("off"), Some("0")] {
            assert_eq!(
                msg(true, mode),
                "draft head installed, not used (off by default)",
                "{mode:?}"
            );
        }
        for mode in [Some("auto"), Some("on")] {
            assert_eq!(msg(true, mode), "draft head present (mtp.safetensors)");
        }
        assert_eq!(msg(false, None), "no draft head (optional)");
        let hint = flash_next_mtp_check(dir, false, None).fix_hint.unwrap();
        assert!(hint.starts_with("The draft head (1.5 GB"), "{hint}");
        assert!(hint.contains("off by default"), "{hint}");
    }

    const SHARD_6: &str = "model-00006.safetensors";

    /// The speed the row quotes for a 5.0 GB/s disk on a Mac that streams memory at 277 GB/s,
    /// for the pool the memory plan leaves on a 36 GB Mac.
    fn quoted(pack: bool) -> String {
        fmt_tps(flash_next::decode_tps(18.4, 277.0, 5.0, pack).unwrap())
    }

    #[test]
    fn a_fast_disk_passes_and_a_speed_is_promised_only_with_the_memory_measured() {
        let measured = flash_next_ssd_check(
            "model-00001.safetensors",
            &result(6.0, 5.0, ProbeEnd::Finished),
            18.4,
            false,
            Some(277.0),
        );
        assert_eq!(measured.status, CheckStatus::Pass);
        assert_eq!(
            measured.message,
            format!(
                "random 5.0 GB/s (sequential 6.0 GB/s): fast enough for ~{} tok/s",
                quoted(false)
            )
        );
        assert!(measured.fix_hint.is_none());
        assert!(
            measured
                .detail
                .as_deref()
                .unwrap()
                .contains("model-00001.safetensors")
        );
        // Memory not measured: the disk's verdict alone, no tok/s.
        let unmeasured = flash_next_ssd_check(
            "s",
            &result(6.0, 5.0, ProbeEnd::Finished),
            18.4,
            false,
            None,
        );
        assert_eq!(unmeasured.status, CheckStatus::Pass);
        assert_eq!(
            unmeasured.message,
            "random 5.0 GB/s (sequential 6.0 GB/s): fast enough"
        );
        assert!(!unmeasured.detail.unwrap().contains("tok/s"));
    }

    /// The same disk quotes more on a Mac with faster memory and the pack: the number is this
    /// Mac's, not a table's.
    #[test]
    fn the_quoted_speed_follows_this_macs_memory_and_the_pack() {
        let msg = |memory: f64, pack: bool| {
            flash_next_ssd_check(
                "s",
                &result(6.0, 5.0, ProbeEnd::Finished),
                18.4,
                pack,
                Some(memory),
            )
            .message
        };
        assert_ne!(msg(150.0, false), msg(400.0, false));
        assert_ne!(msg(277.0, false), msg(277.0, true));
        assert!(msg(277.0, true).ends_with(&format!("~{} tok/s", quoted(true))));
    }

    /// The panel translates the row by matching this text, so its shape is a contract with
    /// `DoctorPanel.svelte` (`translatedDetail`). Change both together.
    #[test]
    fn the_ssd_detail_keeps_the_shape_the_panel_translates() {
        let base = "Read 11000 MB of model-00006.safetensors with the page cache bypassed: 6000 MB \
                    front to back, then 5000 MB as 100 random 2.8 MB reads on 4 threads, the way \
                    the server fetches experts.";
        let c = flash_next_ssd_check(
            SHARD_6,
            &result(6.0, 5.0, ProbeEnd::Finished),
            18.4,
            false,
            Some(277.0),
        );
        assert_eq!(
            c.detail.as_deref(),
            Some(
                format!(
                    "{base} Decode speed is estimated from this volume's random reads and this \
                     Mac's memory bandwidth (277 GB/s, measured), for an expert pool of ~18.4 GB, \
                     what the memory plan leaves on this Mac."
                )
                .as_str()
            )
        );
        let none = flash_next_ssd_check(
            SHARD_6,
            &result(6.0, 5.0, ProbeEnd::Finished),
            18.4,
            false,
            None,
        );
        assert_eq!(
            none.detail.as_deref(),
            Some(
                format!(
                    "{base} Measure this Mac's memory bandwidth in the Flash-Next card (Tuning \
                     tab) to get a decode-speed estimate."
                )
                .as_str()
            )
        );
        let with_pack = flash_next_ssd_check(
            "s",
            &result(6.0, 5.0, ProbeEnd::Finished),
            10.0,
            true,
            Some(277.0),
        );
        assert!(
            with_pack
                .detail
                .unwrap()
                .ends_with("The expert pack's read efficiency is used.")
        );
    }

    #[test]
    fn under_two_gb_per_second_the_row_warns_that_decode_will_be_slow() {
        let c = flash_next_ssd_check(
            "s",
            &result(2.5, 1.5, ProbeEnd::Finished),
            18.4,
            false,
            Some(277.0),
        );
        assert_eq!(c.status, CheckStatus::Warn);
        assert!(
            c.message
                .starts_with("random 1.5 GB/s (sequential 2.5 GB/s): slow, ~"),
            "{}",
            c.message
        );
        assert!(c.fix_hint.is_some());
        // Without the memory number the warning stands without a tok/s.
        let c = flash_next_ssd_check(
            "s",
            &result(2.5, 1.5, ProbeEnd::Finished),
            18.4,
            false,
            None,
        );
        assert_eq!(c.status, CheckStatus::Warn);
        assert_eq!(c.message, "random 1.5 GB/s (sequential 2.5 GB/s): slow");
        // Exactly 2 GB/s is not slow.
        let edge = flash_next_ssd_check(
            "s",
            &result(3.0, 2.0, ProbeEnd::Finished),
            18.4,
            false,
            None,
        );
        assert_eq!(edge.status, CheckStatus::Pass);
    }

    #[test]
    fn a_probe_that_ran_out_of_time_says_so_and_still_reports() {
        let c = flash_next_ssd_check(
            "s",
            &result(0.1, 0.1, ProbeEnd::TimedOut),
            FALLBACK_POOL_GB,
            false,
            None,
        );
        assert_eq!(c.status, CheckStatus::Warn);
        assert!(c.detail.unwrap().contains("stopped early"));

        let mut none = result(1.0, 0.0, ProbeEnd::Finished);
        none.rand_bytes = 0;
        assert_eq!(
            flash_next_ssd_check("s", &none, FALLBACK_POOL_GB, false, Some(277.0)).message,
            "could not measure random reads"
        );
        assert_eq!(flash_next_ssd_stalled().status, CheckStatus::Warn);
    }

    #[test]
    fn nothing_flash_next_appears_without_the_weights() {
        let cfg = PersistentConfig {
            models_dir: temp_dir("no-model"),
            ..PersistentConfig::default()
        };
        assert!(flash_next_checks(&cfg).is_empty());
        let (c, reading) = probe_flash_next_ssd(&cfg, &AtomicBool::new(false)).unwrap();
        assert_eq!(c.message, "Flash-Next is not installed");
        assert_eq!(reading, None, "nothing was read, so nothing is stored");
        std::fs::remove_dir_all(&cfg.models_dir).ok();
    }

    #[test]
    fn an_installed_model_gets_its_rows_and_never_the_slow_probe() {
        let (cfg, dir) = config_with_model("rows", 1 << 20);
        let ids: Vec<_> = flash_next_checks(&cfg).iter().map(|c| c.id).collect();
        assert!(ids.contains(&"flash_next_disk"), "{ids:?}");
        assert!(ids.contains(&"flash_next_pack"), "{ids:?}");
        assert!(ids.contains(&"flash_next_mtp"), "{ids:?}");
        // Only the user may start the 1 GB read.
        assert!(!ids.contains(&"flash_next_ssd"), "{ids:?}");
        std::fs::remove_dir_all(dir.parent().unwrap()).ok();
    }

    #[test]
    fn the_probe_reads_a_real_shard_and_a_cancel_returns_nothing() {
        let (cfg, dir) = config_with_model("probe", 12 << 20);
        let (c, reading) = probe_flash_next_ssd(&cfg, &AtomicBool::new(false)).unwrap();
        assert_eq!(c.id, "flash_next_ssd");
        assert!(c.message.starts_with("random "), "{}", c.message);
        // The numbers come back for the app to store, with the folder they were read from.
        let reading = reading.expect("a reading");
        assert!(reading.random_gbps > 0.0);
        assert!(reading.sequential_gbps.is_some_and(|g| g > 0.0));
        assert_eq!(reading.volume, dir.display().to_string());
        // The memory is not measured yet, so the row promises no tok/s.
        assert!(!c.message.contains("tok/s"), "{}", c.message);
        assert!(probe_flash_next_ssd(&cfg, &AtomicBool::new(true)).is_none());
        std::fs::remove_dir_all(dir.parent().unwrap()).ok();
    }
}
