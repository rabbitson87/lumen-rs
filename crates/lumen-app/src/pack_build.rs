//! Building Flash-Next's expert pack and its 3-bit copy from the app: `lumen-server
//! --pack-experts <model dir>` and `lumen-server --build-expert-copy <model dir> 3,64,top`, run in
//! the background with their stderr progress turned into events.
//!
//! The pack is 68 GB of extra disk that makes decode about 7-15% faster (`docs/flash-next.md`).
//! The copy is 54 GB that holds each expert in 2.15 MB instead of 2.76, so the pool holds more of
//! them; it is another model, which replies a little differently. The server writes either one
//! to a `.tmp` file and renames it at the end, so a stopped build leaves nothing that looks
//! complete, only a partial temp file, which this module deletes. Progress follows the download's
//! pattern: a long task whose lines become `lumen://pack` events, with a command to cancel it.
//! One build runs at a time: each reads the whole model from the same disk.

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

/// Free space the pack build is refused below. It writes 68 GB and the server wants a spare
/// gibibyte on top; this rounds both up.
pub const MIN_FREE_BYTES: u64 = 70_000_000_000;

/// The same for the copy, which writes 54 GB.
pub const COPY_MIN_FREE_BYTES: u64 = 56_000_000_000;

/// What a build makes from the weights.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum BuildKind {
    /// `experts.lpack`: the checkpoint's own experts, one record each.
    Pack,
    /// [`flash_next::COPY_FILE`]: the 3-bit copy of the experts.
    Copy,
}

impl BuildKind {
    fn args(self, dir: &Path) -> Vec<std::ffi::OsString> {
        match self {
            Self::Pack => vec!["--pack-experts".into(), dir.into()],
            Self::Copy => vec![
                "--build-expert-copy".into(),
                dir.into(),
                flash_next::COPY_SPEC.into(),
            ],
        }
    }

    /// What the server prints before `<done> of <total>` once per layer.
    fn progress_prefix(self) -> &'static str {
        match self {
            Self::Pack => "[offload] expert pack: layer ",
            Self::Copy => "[offload] low-bit pack: layer ",
        }
    }

    /// The partial file a build in progress (or one that was stopped) leaves.
    fn tmp_file(self) -> String {
        match self {
            Self::Pack => "experts.lpack.tmp".into(),
            Self::Copy => format!("{}.tmp", flash_next::COPY_FILE),
        }
    }

    pub fn min_free_bytes(self) -> u64 {
        match self {
            Self::Pack => MIN_FREE_BYTES,
            Self::Copy => COPY_MIN_FREE_BYTES,
        }
    }

    fn state(self, dir: &Path) -> PackState {
        match self {
            Self::Pack => flash_next::pack_state(dir),
            Self::Copy => flash_next::copy_state(dir),
        }
    }

    fn what(self) -> &'static str {
        match self {
            Self::Pack => "the expert pack",
            Self::Copy => "the 3-bit copy",
        }
    }
}

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
    /// Which build this is about.
    pub kind: BuildKind,
    pub state: PackBuildState,
    /// Layers written so far, and in all (0 until the server says).
    pub layer: u32,
    pub layers: u32,
    /// The last lines the server printed, when it failed.
    pub error: Option<String>,
}

impl PackProgress {
    fn running(kind: BuildKind, layer: u32, layers: u32) -> Self {
        Self {
            kind,
            state: PackBuildState::Running,
            layer,
            layers,
            error: None,
        }
    }
}

/// `[offload] expert pack: layer 12 of 48` → `(12, 48)` for the pack, `[offload] low-bit pack:
/// layer 12 of 49` for the copy. Any other line is not progress.
pub fn parse_progress(kind: BuildKind, line: &str) -> Option<(u32, u32)> {
    let rest = line.trim().strip_prefix(kind.progress_prefix())?;
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

/// The rules for starting a build of `kind`, whose file is in `state`, in the order they are
/// reported. `building` is any build: one runs at a time. An unreadable free space does not
/// block: the server checks it again itself.
pub fn build_block(
    kind: BuildKind,
    installed: bool,
    state: PackState,
    server_running: bool,
    building: bool,
    free_bytes: Option<u64>,
) -> Option<BuildBlock> {
    if !installed {
        Some(BuildBlock::NotInstalled)
    } else if building {
        Some(BuildBlock::Building)
    } else if matches!(state, PackState::Complete(_)) {
        Some(BuildBlock::AlreadyBuilt)
    } else if server_running {
        Some(BuildBlock::ServerRunning)
    } else if free_bytes.is_some_and(|free| free < kind.min_free_bytes()) {
        Some(BuildBlock::NotEnoughDisk)
    } else {
        None
    }
}

/// The English reason for a refusal (the panel maps `BuildBlock` to its own translated text).
pub fn block_message(kind: BuildKind, block: BuildBlock, free_bytes: Option<u64>) -> String {
    match block {
        BuildBlock::NotInstalled => "Flash-Next is not installed".into(),
        BuildBlock::Building => "a build is already running".into(),
        BuildBlock::AlreadyBuilt => format!("{} is already built", kind.what()),
        BuildBlock::ServerRunning => {
            "stop the server first: the build reads the whole model from the same disk".into()
        }
        BuildBlock::NotEnoughDisk => format!(
            "{} needs about {:.0} GB free; {:.0} GB is free on this volume",
            match kind {
                BuildKind::Pack => "the pack",
                BuildKind::Copy => "the copy",
            },
            kind.min_free_bytes() as f64 / 1e9,
            free_bytes.unwrap_or(0) as f64 / 1e9
        ),
    }
}

fn state_name(state: PackState) -> (&'static str, u64) {
    match state {
        PackState::Absent => ("absent", 0),
        PackState::Incomplete(n) => ("incomplete", n),
        PackState::Complete(n) => ("complete", n),
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
    /// The 3-bit copy, by the same rule.
    pub copy: &'static str,
    pub copy_bytes: u64,
    pub mtp_present: bool,
    pub free_bytes: Option<u64>,
    /// What the pack build needs free.
    pub need_bytes: u64,
    /// What the copy build needs free.
    pub copy_need_bytes: u64,
    pub server_running: bool,
    /// A build of either kind is running; `building_kind` says which.
    pub building: bool,
    pub building_kind: Option<BuildKind>,
    pub build_block: Option<BuildBlock>,
    pub copy_block: Option<BuildBlock>,
    /// What this Mac measured of itself (decode speed rests on it), with the SSD number
    /// counted only if it was read from this model folder.
    pub hardware: HardwareView,
}

pub fn status(
    cfg: &PersistentConfig,
    server_running: bool,
    building: Option<BuildKind>,
) -> FlashNextStatus {
    let Some(dir) = flash_next::find_model_dir(cfg) else {
        return FlashNextStatus {
            installed: false,
            dir: None,
            pack: "absent",
            pack_bytes: 0,
            copy: "absent",
            copy_bytes: 0,
            mtp_present: false,
            free_bytes: None,
            need_bytes: MIN_FREE_BYTES,
            copy_need_bytes: COPY_MIN_FREE_BYTES,
            server_running,
            building: building.is_some(),
            building_kind: building,
            build_block: Some(BuildBlock::NotInstalled),
            copy_block: Some(BuildBlock::NotInstalled),
            hardware: cfg.measured_hardware.view(None),
        };
    };
    let (pack, copy) = (flash_next::pack_state(&dir), flash_next::copy_state(&dir));
    let free = doctor::statvfs_free(&dir);
    let (pack_name, pack_bytes) = state_name(pack);
    let (copy_name, copy_bytes) = state_name(copy);
    let block =
        |kind, state| build_block(kind, true, state, server_running, building.is_some(), free);
    FlashNextStatus {
        installed: true,
        dir: Some(dir.display().to_string()),
        pack: pack_name,
        pack_bytes,
        copy: copy_name,
        copy_bytes,
        mtp_present: flash_next::has_mtp_head(&dir),
        free_bytes: free,
        need_bytes: MIN_FREE_BYTES,
        copy_need_bytes: COPY_MIN_FREE_BYTES,
        server_running,
        building: building.is_some(),
        building_kind: building,
        build_block: block(BuildKind::Pack, pack),
        copy_block: block(BuildKind::Copy, copy),
        hardware: cfg.measured_hardware.view(Some(&dir.display().to_string())),
    }
}

/// How progress reaches whoever listens (the app emits a Tauri event).
pub type Emit = Arc<dyn Fn(PackProgress) + Send + Sync>;

/// Runs at most one build at a time.
pub struct PackBuilder {
    /// The running build's kind, and what wakes it to stop it; `None` when nothing runs.
    running: Mutex<Option<(BuildKind, Arc<Notify>)>>,
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

    /// Which build runs, if one does.
    pub async fn running_kind(&self) -> Option<BuildKind> {
        self.running.lock().await.as_ref().map(|(kind, _)| *kind)
    }

    /// Start the server's build of `kind` for `dir` in the background. Returns once it is
    /// spawned; progress and the outcome arrive through `emit`.
    pub async fn start(
        self: &Arc<Self>,
        kind: BuildKind,
        bin: PathBuf,
        dir: PathBuf,
        emit: Emit,
    ) -> Result<()> {
        let mut slot = self.running.lock().await;
        if slot.is_some() {
            bail!("a build is already running");
        }
        let child = spawn(kind, &bin, &dir)?;
        let cancel = Arc::new(Notify::new());
        *slot = Some((kind, Arc::clone(&cancel)));
        drop(slot);

        emit(PackProgress::running(kind, 0, 0));
        let this = Arc::clone(self);
        tokio::spawn(async move {
            let last = run(kind, child, &dir, &cancel, &emit).await;
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
            Some((_, cancel)) => {
                cancel.notify_one();
                true
            }
            None => false,
        }
    }
}

fn spawn(kind: BuildKind, bin: &Path, dir: &Path) -> Result<tokio::process::Child> {
    let args = kind.args(dir);
    Command::new(bin)
        .args(&args)
        // The server prints the file's path there at the end; nobody needs it.
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("spawn {} {:?}", bin.display(), args[0]))
}

/// Follow the child to its end, emitting a report per layer, and say how it ended.
async fn run(
    kind: BuildKind,
    mut child: tokio::process::Child,
    dir: &Path,
    cancel: &Notify,
    emit: &Emit,
) -> PackProgress {
    let end = |state, layer, layers, error| finish(kind, state, dir, layer, layers, error);
    let Some(stderr) = child.stderr.take() else {
        return end(PackBuildState::Failed, 0, 0, Some("no stderr".into()));
    };
    let mut lines = BufReader::new(stderr).lines();
    let (mut layer, mut layers) = (0, 0);
    let mut tail: VecDeque<String> = VecDeque::new();
    let mut cancelled = false;
    loop {
        tokio::select! {
            next = lines.next_line() => match next {
                Ok(Some(line)) => match parse_progress(kind, &line) {
                    Some((done, total)) => {
                        (layer, layers) = (done, total);
                        emit(PackProgress::running(kind, layer, layers));
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
        return end(PackBuildState::Cancelled, layer, layers, None);
    }
    let said = tail.into_iter().collect::<Vec<_>>().join("\n");
    match exit {
        Ok(status) if status.success() => {
            if matches!(kind.state(dir), PackState::Complete(_)) {
                end(PackBuildState::Done, layers, layers, None)
            } else {
                let why = format!("the build ended but {} is not complete", kind.what());
                end(PackBuildState::Failed, layer, layers, Some(why))
            }
        }
        Ok(status) => {
            let why = if said.is_empty() {
                format!("lumen-server exited with {status}")
            } else {
                said
            };
            end(PackBuildState::Failed, layer, layers, Some(why))
        }
        Err(e) => end(PackBuildState::Failed, layer, layers, Some(e.to_string())),
    }
}

/// The last report of a build. A build that did not finish leaves its partial temp file, which
/// goes with it (54-68 GB would otherwise stay on disk).
fn finish(
    kind: BuildKind,
    state: PackBuildState,
    dir: &Path,
    layer: u32,
    layers: u32,
    error: Option<String>,
) -> PackProgress {
    if state != PackBuildState::Done {
        let _ = std::fs::remove_file(dir.join(kind.tmp_file()));
    }
    PackProgress {
        kind,
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
            parse_progress(BuildKind::Pack, "[offload] expert pack: layer 12 of 48"),
            Some((12, 48))
        );
        assert_eq!(
            parse_progress(BuildKind::Pack, "  [offload] expert pack: layer 48 of 48\n"),
            Some((48, 48))
        );
        assert_eq!(
            parse_progress(BuildKind::Pack, "[offload] expert pack: layer 0 of 48"),
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
            assert_eq!(parse_progress(BuildKind::Pack, line), None, "{line:?}");
        }
    }

    const GB: u64 = 1_000_000_000;

    #[test]
    fn the_build_is_refused_until_everything_it_needs_is_there() {
        let go = |installed, pack, running, building, free| {
            build_block(BuildKind::Pack, installed, pack, running, building, free)
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
        let free = |n| build_block(BuildKind::Pack, true, PackState::Absent, false, false, n);
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
            build_block(
                BuildKind::Pack,
                true,
                PackState::Absent,
                true,
                false,
                Some(GB)
            ),
            Some(BuildBlock::ServerRunning)
        );
        // An already-built pack beats both.
        assert_eq!(
            build_block(
                BuildKind::Pack,
                true,
                PackState::Complete(68 * GB),
                true,
                false,
                Some(GB)
            ),
            Some(BuildBlock::AlreadyBuilt)
        );
        assert_eq!(
            block_message(BuildKind::Pack, BuildBlock::NotEnoughDisk, Some(40 * GB)),
            "the pack needs about 70 GB free; 40 GB is free on this volume"
        );
    }

    #[test]
    fn the_copy_has_its_own_progress_line_and_its_own_disk_need() {
        let copy = |line| parse_progress(BuildKind::Copy, line);
        assert_eq!(
            copy("[offload] low-bit pack: layer 12 of 49"),
            Some((12, 49))
        );
        assert_eq!(
            copy("[offload] low-bit pack: layer 49 of 49"),
            Some((49, 49))
        );
        assert_eq!(copy("[offload] expert pack: layer 12 of 48"), None);
        assert_eq!(
            parse_progress(BuildKind::Pack, "[offload] low-bit pack: layer 12 of 49"),
            None
        );
        let free = |n| build_block(BuildKind::Copy, true, PackState::Absent, false, false, n);
        assert_eq!(
            free(Some(COPY_MIN_FREE_BYTES - 1)),
            Some(BuildBlock::NotEnoughDisk)
        );
        assert_eq!(free(Some(COPY_MIN_FREE_BYTES)), None);
        assert_eq!(
            block_message(BuildKind::Copy, BuildBlock::NotEnoughDisk, Some(40 * GB)),
            "the copy needs about 56 GB free; 40 GB is free on this volume"
        );
        assert_eq!(
            block_message(BuildKind::Copy, BuildBlock::AlreadyBuilt, None),
            "the 3-bit copy is already built"
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
        builder
            .start(BuildKind::Pack, bin, model.clone(), emit)
            .await
            .unwrap();
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
        builder
            .start(BuildKind::Pack, bin, model.clone(), emit)
            .await
            .unwrap();
        finished(&builder).await;
        let last = seen.lock().unwrap().last().cloned().unwrap();
        assert_eq!(last.state, PackBuildState::Failed);
        assert!(last.error.unwrap().contains("69.1 GB"));
        assert!(!model.join(BuildKind::Pack.tmp_file()).exists());
        std::fs::remove_dir_all(model.parent().unwrap()).ok();
    }

    #[tokio::test]
    async fn a_build_that_exits_without_a_complete_pack_has_not_succeeded() {
        let (bin, model) = script("empty", "echo '[offload] expert pack: layer 1 of 1' >&2");
        let (emit, seen) = collector();
        let builder = PackBuilder::new();
        builder
            .start(BuildKind::Pack, bin, model.clone(), emit)
            .await
            .unwrap();
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
        builder
            .start(BuildKind::Pack, bin, model.clone(), emit)
            .await
            .unwrap();
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
        assert!(!model.join(BuildKind::Pack.tmp_file()).exists());
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
            .start(
                BuildKind::Pack,
                bin.clone(),
                model.clone(),
                Arc::clone(&emit),
            )
            .await
            .unwrap();
        assert!(
            builder
                .start(BuildKind::Pack, bin, model.clone(), emit)
                .await
                .is_err()
        );
        builder.cancel().await;
        finished(&builder).await;
        std::fs::remove_dir_all(model.parent().unwrap()).ok();
    }

    /// A stand-in for `lumen-server --build-expert-copy <dir> 3,64,top`.
    fn copy_script(name: &str, body: &str) -> (PathBuf, PathBuf) {
        let (bin, model) = script(name, "");
        std::fs::write(
            &bin,
            format!(
                "#!/bin/sh\n[ \"$1\" = --build-expert-copy ] && [ \"$3\" = 3,64,top ] || exit 9\n{body}\n"
            ),
        )
        .unwrap();
        (bin, model)
    }

    #[tokio::test]
    async fn a_copy_build_runs_the_copy_command_and_ends_done_once_the_copy_is_complete() {
        let make_copy = format!(
            "dd if=/dev/null of=\"$2/{}\" bs=1 count=0 seek={} 2>/dev/null",
            flash_next::COPY_FILE,
            flash_next::COPY_BYTES
        );
        let (bin, model) = copy_script(
            "copy",
            &format!(
                "echo '[offload] low-bit pack: layer 1 of 2' >&2\n\
                 echo '[offload] low-bit pack: layer 2 of 2' >&2\n\
                 {make_copy}"
            ),
        );
        let (emit, seen) = collector();
        let builder = PackBuilder::new();
        builder
            .start(BuildKind::Copy, bin, model.clone(), emit)
            .await
            .unwrap();
        finished(&builder).await;
        let seen = seen.lock().unwrap();
        assert!(seen.iter().all(|p| p.kind == BuildKind::Copy));
        let layers: Vec<(PackBuildState, u32, u32)> =
            seen.iter().map(|p| (p.state, p.layer, p.layers)).collect();
        assert!(layers.contains(&(PackBuildState::Running, 1, 2)));
        assert_eq!(layers.last(), Some(&(PackBuildState::Done, 2, 2)));
        std::fs::remove_dir_all(model.parent().unwrap()).ok();
    }

    #[tokio::test]
    async fn a_failed_copy_build_removes_its_own_temp_file() {
        let tmp = BuildKind::Copy.tmp_file();
        let (bin, model) = copy_script(
            "copyfail",
            &format!("touch \"$2/{tmp}\"\necho 'pack: no room' >&2\nexit 1"),
        );
        let (emit, seen) = collector();
        let builder = PackBuilder::new();
        builder
            .start(BuildKind::Copy, bin, model.clone(), emit)
            .await
            .unwrap();
        finished(&builder).await;
        let last = seen.lock().unwrap().last().cloned().unwrap();
        assert_eq!(
            (last.kind, last.state),
            (BuildKind::Copy, PackBuildState::Failed)
        );
        assert!(!model.join(tmp).exists());
        std::fs::remove_dir_all(model.parent().unwrap()).ok();
    }

    #[tokio::test]
    async fn one_build_runs_at_a_time_whatever_its_kind() {
        let (bin, model) = script("onekind", "sleep 30");
        let (emit, _) = collector();
        let builder = PackBuilder::new();
        builder
            .start(
                BuildKind::Pack,
                bin.clone(),
                model.clone(),
                Arc::clone(&emit),
            )
            .await
            .unwrap();
        assert_eq!(builder.running_kind().await, Some(BuildKind::Pack));
        assert!(
            builder
                .start(BuildKind::Copy, bin, model.clone(), emit)
                .await
                .is_err()
        );
        builder.cancel().await;
        finished(&builder).await;
        assert_eq!(builder.running_kind().await, None);
        std::fs::remove_dir_all(model.parent().unwrap()).ok();
    }

    #[test]
    fn the_status_of_a_model_that_is_not_installed_says_so() {
        let cfg = PersistentConfig {
            models_dir: std::env::temp_dir()
                .join(format!("lumen-pack-none-{}", std::process::id())),
            ..PersistentConfig::default()
        };
        let s = status(&cfg, false, None);
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
        let s = status(&cfg, false, None);
        assert!(s.installed);
        assert_eq!((s.pack, s.pack_bytes, s.mtp_present), ("absent", 0, false));
        // Absent, so only this volume's free space can hold it back.
        let by_disk = |kind| build_block(kind, true, PackState::Absent, false, false, s.free_bytes);
        assert_eq!((s.copy, s.copy_bytes), ("absent", 0));
        assert_eq!(s.copy_block, by_disk(BuildKind::Copy));
        assert_eq!(s.copy_need_bytes, COPY_MIN_FREE_BYTES);
        // A built copy is reported and not built again; the pack can still be.
        std::fs::File::create(dir.join(flash_next::COPY_FILE))
            .unwrap()
            .set_len(flash_next::COPY_BYTES)
            .unwrap();
        let s = status(&cfg, false, None);
        assert_eq!((s.copy, s.copy_bytes), ("complete", flash_next::COPY_BYTES));
        assert_eq!(s.copy_block, Some(BuildBlock::AlreadyBuilt));
        assert_eq!(s.build_block, by_disk(BuildKind::Pack));
        // One build at a time: a running pack build blocks the copy's too.
        let s = status(&cfg, false, Some(BuildKind::Pack));
        assert_eq!((s.building, s.building_kind), (true, Some(BuildKind::Pack)));
        assert_eq!(s.build_block, Some(BuildBlock::Building));
        std::fs::remove_file(dir.join(flash_next::COPY_FILE)).unwrap();
        assert_eq!(
            status(&cfg, false, Some(BuildKind::Pack)).copy_block,
            Some(BuildBlock::Building)
        );
        // A running server blocks it whatever the disk says.
        assert_eq!(
            status(&cfg, true, None).build_block,
            Some(BuildBlock::ServerRunning)
        );
        std::fs::write(dir.join("mtp.safetensors"), b"x").unwrap();
        assert!(status(&cfg, false, None).mtp_present);
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
        let h = status(&cfg, false, None).hardware;
        assert_eq!(
            (h.memory_gbps, h.ssd_random_gbps, h.ssd_stale),
            (None, None, false)
        );
        // Measured on this folder.
        cfg.measured_hardware.record_memory(231.0, 10);
        cfg.measured_hardware
            .record_ssd(3.1, Some(5.0), &dir.display().to_string(), 20);
        let h = status(&cfg, false, None).hardware;
        assert_eq!(
            (h.memory_gbps, h.ssd_random_gbps, h.ssd_stale),
            (Some(231.0), Some(3.1), false)
        );
        // The SSD number was read from some other folder: stale, memory unaffected.
        cfg.measured_hardware
            .record_ssd(3.1, Some(5.0), "/Volumes/Elsewhere/Flash-Next", 20);
        let h = status(&cfg, false, None).hardware;
        assert_eq!(
            (h.memory_gbps, h.ssd_random_gbps, h.ssd_stale),
            (Some(231.0), None, true)
        );
        // The model is gone altogether: no folder to match, so the SSD is not measured.
        std::fs::remove_dir_all(&models).ok();
        let h = status(&cfg, false, None).hardware;
        assert_eq!(
            (h.memory_gbps, h.ssd_random_gbps, h.ssd_stale),
            (Some(231.0), None, true)
        );
    }
}
