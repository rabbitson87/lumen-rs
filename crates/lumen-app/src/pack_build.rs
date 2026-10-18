//! Building Flash-Next's expert pack from the app: `lumen-server --pack-experts <model dir>`,
//! run in the background with its stderr progress turned into events.
//!
//! The pack is 68 GB of extra disk that makes decode about 7-15% faster (`docs/flash-next.md`). The
//! server writes `experts.lpack.tmp` and renames it at the end, so a stopped build leaves no pack
//! that looks complete, only a partial temp file, which this module deletes. Progress follows the
//! download's pattern: a long task whose lines become `lumen://pack` events, with a command to
//! cancel it.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use serde::Serialize;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::{Mutex, Notify};

use crate::config::PersistentConfig;
use crate::doctor;
use crate::flash_next::{self, PackState};
use crate::hardware::HardwareView;

/// Tauri event name for build progress.
pub const EVENT_PACK: &str = "lumen://pack";

/// Free space the build is refused below. It writes 68 GB and the server wants a spare gibibyte
/// on top; this rounds both up.
pub const MIN_FREE_BYTES: u64 = 70_000_000_000;

/// The partial file a build in progress (or one that was stopped) leaves.
const TMP_FILE: &str = "experts.lpack.tmp";

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum PackBuildState {
    Running,
    Done,
    Failed,
    Cancelled,
}

/// One progress report, emitted as `lumen://pack`.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct PackProgress {
    pub state: PackBuildState,
    /// Layers written so far, and in all (0 until the server says).
    pub layer: u32,
    pub layers: u32,
    /// The last lines the server printed, when it failed.
    pub error: Option<String>,
}

impl PackProgress {
    fn running(layer: u32, layers: u32) -> Self {
        Self {
            state: PackBuildState::Running,
            layer,
            layers,
            error: None,
        }
    }
}

/// `[offload] expert pack: layer 12 of 48` → `(12, 48)`. Any other line is not progress.
pub fn parse_progress(line: &str) -> Option<(u32, u32)> {
    let rest = line.trim().strip_prefix("[offload] expert pack: layer ")?;
    let (done, total) = rest.split_once(" of ")?;
    let (done, total): (u32, u32) = (done.trim().parse().ok()?, total.trim().parse().ok()?);
    (total > 0 && done <= total).then_some((done, total))
}

/// Why the build button is refused.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BuildBlock {
    NotInstalled,
    Building,
    AlreadyBuilt,
    ServerRunning,
    NotEnoughDisk,
}

/// The rules for starting a build, in the order they are reported. An unreadable free space does
/// not block: the server checks it again itself.
pub fn build_block(
    installed: bool,
    pack: PackState,
    server_running: bool,
    building: bool,
    free_bytes: Option<u64>,
) -> Option<BuildBlock> {
    if !installed {
        Some(BuildBlock::NotInstalled)
    } else if building {
        Some(BuildBlock::Building)
    } else if matches!(pack, PackState::Complete(_)) {
        Some(BuildBlock::AlreadyBuilt)
    } else if server_running {
        Some(BuildBlock::ServerRunning)
    } else if free_bytes.is_some_and(|free| free < MIN_FREE_BYTES) {
        Some(BuildBlock::NotEnoughDisk)
    } else {
        None
    }
}

/// The English reason for a refusal (the panel maps `BuildBlock` to its own translated text).
pub fn block_message(block: BuildBlock, free_bytes: Option<u64>) -> String {
    match block {
        BuildBlock::NotInstalled => "Flash-Next is not installed".into(),
        BuildBlock::Building => "a pack build is already running".into(),
        BuildBlock::AlreadyBuilt => "the expert pack is already built".into(),
        BuildBlock::ServerRunning => {
            "stop the server first: the build reads the whole model from the same disk".into()
        }
        BuildBlock::NotEnoughDisk => format!(
            "the pack needs about {:.0} GB free; {:.0} GB is free on this volume",
            MIN_FREE_BYTES as f64 / 1e9,
            free_bytes.unwrap_or(0) as f64 / 1e9
        ),
    }
}

/// What the Flash-Next settings need to know about the model on disk.
#[derive(Debug, Clone, Serialize)]
pub struct FlashNextStatus {
    pub installed: bool,
    pub dir: Option<String>,
    /// `absent`, `incomplete` (an interrupted copy the server ignores) or `complete`.
    pub pack: &'static str,
    pub pack_bytes: u64,
    pub mtp_present: bool,
    pub free_bytes: Option<u64>,
    /// What the pack build needs free.
    pub need_bytes: u64,
    pub server_running: bool,
    pub building: bool,
    pub build_block: Option<BuildBlock>,
    /// What this Mac measured of itself (decode speed rests on it), with the SSD number
    /// counted only if it was read from this model folder.
    pub hardware: HardwareView,
}

pub fn status(cfg: &PersistentConfig, server_running: bool, building: bool) -> FlashNextStatus {
    let Some(dir) = flash_next::find_model_dir(cfg) else {
        return FlashNextStatus {
            installed: false,
            dir: None,
            pack: "absent",
            pack_bytes: 0,
            mtp_present: false,
            free_bytes: None,
            need_bytes: MIN_FREE_BYTES,
            server_running,
            building,
            build_block: Some(BuildBlock::NotInstalled),
            hardware: cfg.measured_hardware.view(None),
        };
    };
    let pack = flash_next::pack_state(&dir);
    let free = doctor::statvfs_free(&dir);
    let (name, bytes) = match pack {
        PackState::Absent => ("absent", 0),
        PackState::Incomplete(n) => ("incomplete", n),
        PackState::Complete(n) => ("complete", n),
    };
    FlashNextStatus {
        installed: true,
        dir: Some(dir.display().to_string()),
        pack: name,
        pack_bytes: bytes,
        mtp_present: flash_next::has_mtp_head(&dir),
        free_bytes: free,
        need_bytes: MIN_FREE_BYTES,
        server_running,
        building,
        build_block: build_block(true, pack, server_running, building, free),
        hardware: cfg.measured_hardware.view(Some(&dir.display().to_string())),
    }
}

/// How progress reaches whoever listens (the app emits a Tauri event).
pub type Emit = Arc<dyn Fn(PackProgress) + Send + Sync>;

/// Runs at most one pack build at a time.
pub struct PackBuilder {
    /// Wakes the running build to stop it; `None` when nothing runs.
    running: Mutex<Option<Arc<Notify>>>,
}

impl PackBuilder {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            running: Mutex::new(None),
        })
    }

    pub async fn is_running(&self) -> bool {
        self.running.lock().await.is_some()
    }

    /// Start `bin --pack-experts dir` in the background. Returns once it is spawned; progress and
    /// the outcome arrive through `emit`.
    pub async fn start(self: &Arc<Self>, bin: PathBuf, dir: PathBuf, emit: Emit) -> Result<()> {
        let mut slot = self.running.lock().await;
        if slot.is_some() {
            bail!("a pack build is already running");
        }
        let child = spawn(&bin, &dir)?;
        let cancel = Arc::new(Notify::new());
        *slot = Some(Arc::clone(&cancel));
        drop(slot);

        emit(PackProgress::running(0, 0));
        let this = Arc::clone(self);
        tokio::spawn(async move {
            let last = run(child, &dir, &cancel, &emit).await;
            // Cleared first, so a listener that reacts to the outcome by asking for the status
            // sees no build running.
            *this.running.lock().await = None;
            emit(last);
        });
        Ok(())
    }

    /// Stop the running build. `false` when none runs.
    pub async fn cancel(&self) -> bool {
        match self.running.lock().await.as_ref() {
            Some(cancel) => {
                cancel.notify_one();
                true
            }
            None => false,
        }
    }
}

fn spawn(bin: &Path, dir: &Path) -> Result<tokio::process::Child> {
    Command::new(bin)
        .arg("--pack-experts")
        .arg(dir)
        // The server prints the pack's path there at the end; nobody needs it.
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("spawn {} --pack-experts", bin.display()))
}

/// Follow the child to its end, emitting a report per layer, and say how it ended.
async fn run(
    mut child: tokio::process::Child,
    dir: &Path,
    cancel: &Notify,
    emit: &Emit,
) -> PackProgress {
    let Some(stderr) = child.stderr.take() else {
        return finish(PackBuildState::Failed, dir, 0, 0, Some("no stderr".into()));
    };
    let mut lines = BufReader::new(stderr).lines();
    let (mut layer, mut layers) = (0, 0);
    let mut tail: VecDeque<String> = VecDeque::new();
    let mut cancelled = false;
    loop {
        tokio::select! {
            next = lines.next_line() => match next {
                Ok(Some(line)) => match parse_progress(&line) {
                    Some((done, total)) => {
                        (layer, layers) = (done, total);
                        emit(PackProgress::running(layer, layers));
                    }
                    None => {
                        if tail.len() == 4 {
                            tail.pop_front();
                        }
                        tail.push_back(line);
                    }
                },
                // End of the stream: the child has closed stderr, normally by exiting.
                _ => break,
            },
            () = cancel.notified() => {
                cancelled = true;
                let _ = child.start_kill();
                break;
            }
        }
    }
    let exit = child.wait().await;
    if cancelled {
        return finish(PackBuildState::Cancelled, dir, layer, layers, None);
    }
    let said = tail.into_iter().collect::<Vec<_>>().join("\n");
    match exit {
        Ok(status) if status.success() => {
            if matches!(flash_next::pack_state(dir), PackState::Complete(_)) {
                finish(PackBuildState::Done, dir, layers, layers, None)
            } else {
                let why = "the build ended but the pack is not complete".to_string();
                finish(PackBuildState::Failed, dir, layer, layers, Some(why))
            }
        }
        Ok(status) => {
            let why = if said.is_empty() {
                format!("lumen-server exited with {status}")
            } else {
                said
            };
            finish(PackBuildState::Failed, dir, layer, layers, Some(why))
        }
        Err(e) => finish(
            PackBuildState::Failed,
            dir,
            layer,
            layers,
            Some(e.to_string()),
        ),
    }
}

/// The last report of a build. A build that did not finish leaves its partial temp file, which
/// goes with it (68 GB would otherwise stay on disk).
fn finish(
    state: PackBuildState,
    dir: &Path,
    layer: u32,
    layers: u32,
    error: Option<String>,
) -> PackProgress {
    if state != PackBuildState::Done {
        let _ = std::fs::remove_file(dir.join(TMP_FILE));
    }
    PackProgress {
        state,
        layer,
        layers,
        error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::Mutex as StdMutex;
    use std::time::Duration;

    #[test]
    fn a_layer_line_is_progress() {
        assert_eq!(
            parse_progress("[offload] expert pack: layer 12 of 48"),
            Some((12, 48))
        );
        assert_eq!(
            parse_progress("  [offload] expert pack: layer 48 of 48\n"),
            Some((48, 48))
        );
        assert_eq!(
            parse_progress("[offload] expert pack: layer 0 of 48"),
            Some((0, 48))
        );
    }

    #[test]
    fn other_server_lines_are_not() {
        for line in [
            "",
            "[offload] building the expert pack in /m ...",
            "[offload] expert pack ready in 49 s: /m/experts.lpack",
            "[offload] ignoring the expert pack /m: bad magic",
            "[offload] expert pack: layer x of 48",
            "[offload] expert pack: layer 12 of",
            "[offload] expert pack: layer 12 of 0",
            // More layers done than there are.
            "[offload] expert pack: layer 49 of 48",
            "expert pack: layer 12 of 48",
        ] {
            assert_eq!(parse_progress(line), None, "{line:?}");
        }
    }

    const GB: u64 = 1_000_000_000;

    #[test]
    fn the_build_is_refused_until_everything_it_needs_is_there() {
        let go = |installed, pack, running, building, free| {
            build_block(installed, pack, running, building, free)
        };
        let absent = PackState::Absent;
        assert_eq!(go(true, absent, false, false, Some(100 * GB)), None);
        // An unfinished pack is rebuilt over.
        assert_eq!(
            go(
                true,
                PackState::Incomplete(5 * GB),
                false,
                false,
                Some(100 * GB)
            ),
            None
        );
        assert_eq!(
            go(false, absent, false, false, Some(100 * GB)),
            Some(BuildBlock::NotInstalled)
        );
        assert_eq!(
            go(true, absent, false, true, Some(100 * GB)),
            Some(BuildBlock::Building)
        );
        assert_eq!(
            go(
                true,
                PackState::Complete(68 * GB),
                false,
                false,
                Some(100 * GB)
            ),
            Some(BuildBlock::AlreadyBuilt)
        );
        assert_eq!(
            go(true, absent, true, false, Some(100 * GB)),
            Some(BuildBlock::ServerRunning)
        );
        assert_eq!(
            go(true, absent, false, false, Some(69 * GB)),
            Some(BuildBlock::NotEnoughDisk)
        );
    }

    #[test]
    fn seventy_gb_free_is_the_line_and_an_unreadable_volume_does_not_block() {
        let free = |n| build_block(true, PackState::Absent, false, false, n);
        assert_eq!(
            free(Some(MIN_FREE_BYTES - 1)),
            Some(BuildBlock::NotEnoughDisk)
        );
        assert_eq!(free(Some(MIN_FREE_BYTES)), None);
        // The server checks its own space again; a volume we cannot read is not a reason.
        assert_eq!(free(None), None);
    }

    #[test]
    fn the_reasons_are_reported_in_a_fixed_order() {
        // A running server and a full disk: the server is named first.
        assert_eq!(
            build_block(true, PackState::Absent, true, false, Some(GB)),
            Some(BuildBlock::ServerRunning)
        );
        // An already-built pack beats both.
        assert_eq!(
            build_block(true, PackState::Complete(68 * GB), true, false, Some(GB)),
            Some(BuildBlock::AlreadyBuilt)
        );
        assert_eq!(
            block_message(BuildBlock::NotEnoughDisk, Some(40 * GB)),
            "the pack needs about 70 GB free; 40 GB is free on this volume"
        );
    }

    // ── The runner, against a stand-in for `lumen-server --pack-experts` ──

    fn script(name: &str, body: &str) -> (PathBuf, PathBuf) {
        let root = std::env::temp_dir().join(format!("lumen-pack-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let model = root.join("model");
        std::fs::create_dir_all(&model).unwrap();
        let bin = root.join("fake-server");
        std::fs::write(
            &bin,
            format!("#!/bin/sh\n[ \"$1\" = --pack-experts ] || exit 9\n{body}\n"),
        )
        .unwrap();
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
        (bin, model)
    }

    fn collector() -> (Emit, Arc<StdMutex<Vec<PackProgress>>>) {
        let seen = Arc::new(StdMutex::new(Vec::new()));
        let sink = Arc::clone(&seen);
        (Arc::new(move |p| sink.lock().unwrap().push(p)), seen)
    }

    async fn finished(builder: &Arc<PackBuilder>) {
        for _ in 0..200 {
            if !builder.is_running().await {
                return;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        panic!("the build did not finish");
    }

    /// A sparse file of the size of a real pack, so the build ends "complete".
    const MAKE_PACK: &str =
        "dd if=/dev/null of=\"$2/experts.lpack\" bs=1 count=0 seek=68100000000 2>/dev/null";

    #[tokio::test]
    async fn a_build_reports_every_layer_and_ends_done_once_the_pack_is_complete() {
        let (bin, model) = script(
            "done",
            &format!(
                "echo '[offload] expert pack: layer 1 of 3' >&2\n\
                 echo 'unrelated noise' >&2\n\
                 echo '[offload] expert pack: layer 2 of 3' >&2\n\
                 echo '[offload] expert pack: layer 3 of 3' >&2\n\
                 {MAKE_PACK}"
            ),
        );
        let (emit, seen) = collector();
        let builder = PackBuilder::new();
        builder.start(bin, model.clone(), emit).await.unwrap();
        finished(&builder).await;
        let seen = seen.lock().unwrap();
        let layers: Vec<(PackBuildState, u32, u32)> =
            seen.iter().map(|p| (p.state, p.layer, p.layers)).collect();
        assert_eq!(layers.first(), Some(&(PackBuildState::Running, 0, 0)));
        assert!(layers.contains(&(PackBuildState::Running, 2, 3)));
        assert_eq!(layers.last(), Some(&(PackBuildState::Done, 3, 3)));
        std::fs::remove_dir_all(model.parent().unwrap()).ok();
    }

    #[tokio::test]
    async fn a_build_that_fails_says_what_the_server_said_and_removes_its_temp_file() {
        let (bin, model) = script(
            "fail",
            "touch \"$2/experts.lpack.tmp\"\n\
             echo 'pack: building it needs 69.1 GB and /m has 12.0 GB free' >&2\n\
             exit 1",
        );
        let (emit, seen) = collector();
        let builder = PackBuilder::new();
        builder.start(bin, model.clone(), emit).await.unwrap();
        finished(&builder).await;
        let last = seen.lock().unwrap().last().cloned().unwrap();
        assert_eq!(last.state, PackBuildState::Failed);
        assert!(last.error.unwrap().contains("69.1 GB"));
        assert!(!model.join(TMP_FILE).exists());
        std::fs::remove_dir_all(model.parent().unwrap()).ok();
    }

    #[tokio::test]
    async fn a_build_that_exits_without_a_complete_pack_has_not_succeeded() {
        let (bin, model) = script("empty", "echo '[offload] expert pack: layer 1 of 1' >&2");
        let (emit, seen) = collector();
        let builder = PackBuilder::new();
        builder.start(bin, model.clone(), emit).await.unwrap();
        finished(&builder).await;
        let last = seen.lock().unwrap().last().cloned().unwrap();
        assert_eq!(last.state, PackBuildState::Failed);
        std::fs::remove_dir_all(model.parent().unwrap()).ok();
    }

    #[tokio::test]
    async fn cancelling_stops_the_child_and_leaves_no_pack_and_no_temp_file() {
        let (bin, model) = script(
            "cancel",
            "touch \"$2/experts.lpack.tmp\"\n\
             echo '[offload] expert pack: layer 5 of 48' >&2\n\
             sleep 30",
        );
        let (emit, seen) = collector();
        let builder = PackBuilder::new();
        builder.start(bin, model.clone(), emit).await.unwrap();
        // Wait for the first layer, then stop it.
        for _ in 0..200 {
            if seen.lock().unwrap().iter().any(|p| p.layer == 5) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert!(builder.cancel().await);
        finished(&builder).await;
        let last = seen.lock().unwrap().last().cloned().unwrap();
        assert_eq!(
            (last.state, last.layer, last.layers),
            (PackBuildState::Cancelled, 5, 48)
        );
        assert!(!model.join(TMP_FILE).exists());
        assert!(!model.join("experts.lpack").exists());
        // Nothing is left to cancel.
        assert!(!builder.cancel().await);
        std::fs::remove_dir_all(model.parent().unwrap()).ok();
    }

    #[tokio::test]
    async fn a_second_build_is_refused_while_one_runs() {
        let (bin, model) = script("twice", "sleep 30");
        let (emit, _) = collector();
        let builder = PackBuilder::new();
        builder
            .start(bin.clone(), model.clone(), Arc::clone(&emit))
            .await
            .unwrap();
        assert!(builder.start(bin, model.clone(), emit).await.is_err());
        builder.cancel().await;
        finished(&builder).await;
        std::fs::remove_dir_all(model.parent().unwrap()).ok();
    }

    #[test]
    fn the_status_of_a_model_that_is_not_installed_says_so() {
        let cfg = PersistentConfig {
            models_dir: std::env::temp_dir()
                .join(format!("lumen-pack-none-{}", std::process::id())),
            ..PersistentConfig::default()
        };
        let s = status(&cfg, false, false);
        assert!(!s.installed);
        assert_eq!(s.build_block, Some(BuildBlock::NotInstalled));
        assert_eq!(s.need_bytes, MIN_FREE_BYTES);
    }

    #[test]
    fn the_status_of_an_installed_model_reports_pack_head_and_whether_it_can_be_built() {
        let models = std::env::temp_dir().join(format!("lumen-pack-status-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&models);
        let dir = models.join("pipenetwork--Qwen3.8-Flash-Next-MLX-4bit");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        std::fs::write(dir.join("model-00001.safetensors"), vec![0u8; 100]).unwrap();
        let cfg = PersistentConfig {
            models_dir: models.clone(),
            ..PersistentConfig::default()
        };
        let s = status(&cfg, false, false);
        assert!(s.installed);
        assert_eq!((s.pack, s.pack_bytes, s.mtp_present), ("absent", 0, false));
        // A running server blocks it whatever the disk says.
        assert_eq!(
            status(&cfg, true, false).build_block,
            Some(BuildBlock::ServerRunning)
        );
        std::fs::write(dir.join("mtp.safetensors"), b"x").unwrap();
        assert!(status(&cfg, false, false).mtp_present);
        std::fs::remove_dir_all(&models).ok();
    }

    /// The status carries what this Mac measured, and the SSD number counts only for the model
    /// folder it was read from: moved or deleted, it is stale and the memory number is kept.
    #[test]
    fn the_status_carries_the_measurements_and_marks_an_ssd_number_from_another_folder_stale() {
        let models = std::env::temp_dir().join(format!("lumen-pack-hw-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&models);
        let dir = models.join("pipenetwork--Qwen3.8-Flash-Next-MLX-4bit");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        std::fs::write(dir.join("model-00001.safetensors"), vec![0u8; 100]).unwrap();
        let mut cfg = PersistentConfig {
            models_dir: models.clone(),
            ..PersistentConfig::default()
        };
        // Nothing measured yet.
        let h = status(&cfg, false, false).hardware;
        assert_eq!(
            (h.memory_gbps, h.ssd_random_gbps, h.ssd_stale),
            (None, None, false)
        );
        // Measured on this folder.
        cfg.measured_hardware.record_memory(231.0, 10);
        cfg.measured_hardware
            .record_ssd(3.1, Some(5.0), &dir.display().to_string(), 20);
        let h = status(&cfg, false, false).hardware;
        assert_eq!(
            (h.memory_gbps, h.ssd_random_gbps, h.ssd_stale),
            (Some(231.0), Some(3.1), false)
        );
        // The SSD number was read from some other folder: stale, memory unaffected.
        cfg.measured_hardware
            .record_ssd(3.1, Some(5.0), "/Volumes/Elsewhere/Flash-Next", 20);
        let h = status(&cfg, false, false).hardware;
        assert_eq!(
            (h.memory_gbps, h.ssd_random_gbps, h.ssd_stale),
            (Some(231.0), None, true)
        );
        // The model is gone altogether: no folder to match, so the SSD is not measured.
        std::fs::remove_dir_all(&models).ok();
        let h = status(&cfg, false, false).hardware;
        assert_eq!(
            (h.memory_gbps, h.ssd_random_gbps, h.ssd_stale),
            (Some(231.0), None, true)
        );
    }
}
