//! What this Mac measured of itself for the Flash-Next decode-speed estimate.
//!
//! Decode speed is set by the Mac's memory bandwidth and its SSD's random-read speed
//! (`flash_next::decode_tps`), not by any one machine's table, so the app measures both:
//! the memory number with `lumen-server --measure-memory-bandwidth` (about a second, one JSON
//! line), the SSD number with `flash_next::probe_read_speed` on the model's own volume. They are
//! stored in the config (`MeasuredHardware`), with when and, for the SSD, which model folder they
//! were read from: a number measured on another folder no longer counts.

use std::path::Path;
use std::process::Stdio;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use tokio::process::Command;

use crate::config::MeasuredHardware;

/// Tauri event name for the Measure action's progress.
pub const EVENT_MEASURE: &str = "lumen://measure";

/// The memory test takes a second or two; this is how long the app waits before giving up on a
/// server binary that hangs (a stuck Metal device, say).
pub const MEMORY_PROBE_TIMEOUT: Duration = Duration::from_secs(30);

pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

// ── The memory-bandwidth test ─────────────────────────────────────────

/// The one JSON line `lumen-server --measure-memory-bandwidth` prints on stdout, e.g.
/// `{"gpu_read_gbps":277.0,"best_gbps":280.3,"bytes":1073741824,"passes":15}`.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct MemoryProbe {
    /// What the app uses: the GPU's sustained read speed.
    pub gpu_read_gbps: f64,
    #[serde(default)]
    pub best_gbps: Option<f64>,
    #[serde(default)]
    pub bytes: Option<u64>,
    #[serde(default)]
    pub passes: Option<u32>,
}

/// The measurement in the server's stdout: its last line that is a JSON object. Anything else it
/// printed is ignored; a malformed line, a missing `gpu_read_gbps` or one that is not a positive
/// number is an error, never a 0.
pub fn parse_memory_probe(stdout: &str) -> Result<MemoryProbe> {
    let line = stdout
        .lines()
        .rev()
        .map(str::trim)
        .find(|l| l.starts_with('{'))
        .ok_or_else(|| anyhow!("it printed no measurement"))?;
    let probe: MemoryProbe =
        serde_json::from_str(line).map_err(|e| anyhow!("its measurement is malformed ({e})"))?;
    if !(probe.gpu_read_gbps.is_finite() && probe.gpu_read_gbps > 0.0) {
        bail!(
            "its measurement is not a positive speed ({})",
            probe.gpu_read_gbps
        );
    }
    Ok(probe)
}

/// The last few non-empty lines of `text`, for an error message.
fn tail_lines(text: &str) -> String {
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    let tail = lines[lines.len().saturating_sub(3)..].join(" / ");
    if tail.chars().count() > 400 {
        tail.chars().skip(tail.chars().count() - 400).collect()
    } else {
        tail
    }
}

/// Runs `bin --measure-memory-bandwidth` and reads its measurement. A non-zero exit, no
/// measurement, a malformed one or running past `timeout` is an error carrying what the server
/// said on stderr.
pub async fn run_memory_probe(bin: &Path, timeout: Duration) -> Result<MemoryProbe> {
    let child = Command::new(bin)
        .arg("--measure-memory-bandwidth")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("start {}", bin.display()))?;
    // Dropping the future on a timeout drops the child, which kills it.
    let out = tokio::time::timeout(timeout, child.wait_with_output())
        .await
        .map_err(|_| anyhow!("the memory test did not finish in {} s", timeout.as_secs()))?
        .context("wait for the memory test")?;
    let stderr = String::from_utf8_lossy(&out.stderr);
    if !out.status.success() {
        let why = tail_lines(&stderr);
        bail!(
            "the memory test failed ({}){}{}",
            out.status,
            if why.is_empty() { "" } else { ": " },
            why
        );
    }
    parse_memory_probe(&String::from_utf8_lossy(&out.stdout)).map_err(|e| {
        let why = tail_lines(&stderr);
        if why.is_empty() {
            e
        } else {
            anyhow!("{e}: {why}")
        }
    })
}

// ── What is stored, and when it counts ────────────────────────────────

fn speed(g: Option<f64>) -> Option<f64> {
    g.filter(|g| g.is_finite() && *g > 0.0)
}

/// The stored numbers as the card shows them, with the SSD's staleness worked out.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct HardwareView {
    pub memory_gbps: Option<f64>,
    pub memory_measured_at: Option<u64>,
    /// `None` when the SSD was never measured, or was measured on another model folder.
    pub ssd_random_gbps: Option<f64>,
    pub ssd_sequential_gbps: Option<f64>,
    pub ssd_measured_at: Option<u64>,
    /// An SSD number is stored but not for the current model folder.
    pub ssd_stale: bool,
}

impl MeasuredHardware {
    /// The memory bandwidth, when one was measured.
    pub fn memory(&self) -> Option<f64> {
        speed(self.memory_gbps)
    }

    /// The SSD's random-read speed, when it was measured on `volume` (the current model folder).
    /// A number with no folder recorded, or recorded for another one, does not count.
    pub fn ssd_random_for(&self, volume: Option<&str>) -> Option<f64> {
        let g = speed(self.ssd_random_gbps)?;
        (volume.is_some() && self.ssd_volume.as_deref() == volume).then_some(g)
    }

    pub fn record_memory(&mut self, gbps: f64, at: u64) {
        self.memory_gbps = Some(gbps);
        self.memory_measured_at = Some(at);
    }

    pub fn record_ssd(
        &mut self,
        random_gbps: f64,
        sequential_gbps: Option<f64>,
        volume: &str,
        at: u64,
    ) {
        self.ssd_random_gbps = Some(random_gbps);
        self.ssd_sequential_gbps = sequential_gbps;
        self.ssd_volume = Some(volume.to_string());
        self.ssd_measured_at = Some(at);
    }

    pub fn view(&self, volume: Option<&str>) -> HardwareView {
        let ssd = self.ssd_random_for(volume);
        let fresh = ssd.is_some();
        HardwareView {
            memory_gbps: self.memory(),
            memory_measured_at: self.memory().and(self.memory_measured_at),
            ssd_random_gbps: ssd,
            ssd_sequential_gbps: if fresh {
                speed(self.ssd_sequential_gbps)
            } else {
                None
            },
            ssd_measured_at: if fresh { self.ssd_measured_at } else { None },
            ssd_stale: speed(self.ssd_random_gbps).is_some() && !fresh,
        }
    }
}

// ── The Measure action ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MeasureState {
    Running,
    Done,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MeasureStage {
    Memory,
    Ssd,
}

/// One report of the Measure action, emitted as `lumen://measure` and returned at the end.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct MeasureProgress {
    pub state: MeasureState,
    /// What is being measured now (while `Running`).
    pub stage: Option<MeasureStage>,
    pub memory_gbps: Option<f64>,
    pub ssd_random_gbps: Option<f64>,
    /// The SSD part did not run because the model is not installed.
    pub ssd_skipped: bool,
    pub memory_error: Option<String>,
    pub ssd_error: Option<String>,
}

impl MeasureProgress {
    pub fn running(stage: MeasureStage) -> Self {
        Self {
            state: MeasureState::Running,
            stage: Some(stage),
            memory_gbps: None,
            ssd_random_gbps: None,
            ssd_skipped: false,
            memory_error: None,
            ssd_error: None,
        }
    }
}

/// How the SSD part ended.
#[derive(Debug, Clone, PartialEq)]
pub enum SsdOutcome {
    /// The Flash-Next weights are not on disk: nothing to read.
    NotInstalled,
    Measured(f64),
    Failed(String),
    Cancelled,
}

/// The final report from how the two parts ended. Cancelling wins; otherwise the action is done
/// when either number was obtained and failed only when neither was. Numbers measured before a
/// cancel are in the report (and were already stored).
pub fn finish(memory: Result<f64, String>, ssd: SsdOutcome, cancelled: bool) -> MeasureProgress {
    let (memory_gbps, memory_error) = match memory {
        Ok(g) => (Some(g), None),
        Err(e) => (None, Some(e)),
    };
    let mut report = MeasureProgress {
        state: MeasureState::Done,
        stage: None,
        memory_gbps,
        ssd_random_gbps: None,
        ssd_skipped: false,
        memory_error,
        ssd_error: None,
    };
    let mut ssd_cancelled = false;
    match ssd {
        SsdOutcome::NotInstalled => report.ssd_skipped = true,
        SsdOutcome::Measured(g) => report.ssd_random_gbps = Some(g),
        SsdOutcome::Failed(e) => report.ssd_error = Some(e),
        SsdOutcome::Cancelled => ssd_cancelled = true,
    }
    report.state = if cancelled || ssd_cancelled {
        MeasureState::Cancelled
    } else if report.memory_gbps.is_some() || report.ssd_random_gbps.is_some() {
        MeasureState::Done
    } else {
        MeasureState::Failed
    };
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;

    const LINE: &str =
        r#"{"gpu_read_gbps":277.0,"best_gbps":280.3,"bytes":1073741824,"passes":15}"#;

    // ── Reading the server's line ───────────────────────────────────────

    #[test]
    fn the_servers_line_is_read() {
        let p = parse_memory_probe(&format!("{LINE}\n")).unwrap();
        assert_eq!(p.gpu_read_gbps, 277.0);
        assert_eq!(p.best_gbps, Some(280.3));
        assert_eq!(p.bytes, Some(1_073_741_824));
        assert_eq!(p.passes, Some(15));
        // A line the real `lumen-server --measure-memory-bandwidth` printed on a 36 GB Mac.
        let real =
            "{\"gpu_read_gbps\":256.5,\"best_gbps\":267.0,\"bytes\":1073741824,\"passes\":15}\n";
        assert_eq!(parse_memory_probe(real).unwrap().gpu_read_gbps, 256.5);
        // Only gpu_read_gbps is needed.
        assert_eq!(
            parse_memory_probe(r#"{"gpu_read_gbps":231.5}"#)
                .unwrap()
                .gpu_read_gbps,
            231.5
        );
    }

    #[test]
    fn other_output_around_the_line_is_ignored_and_the_last_object_wins() {
        let out = format!("loading metal\n{{\"gpu_read_gbps\":100.0}}\nwarming up\n{LINE}\ndone\n");
        assert_eq!(parse_memory_probe(&out).unwrap().gpu_read_gbps, 277.0);
        assert_eq!(
            parse_memory_probe(&format!("  {LINE}  "))
                .unwrap()
                .gpu_read_gbps,
            277.0
        );
    }

    #[test]
    fn a_malformed_missing_or_unusable_measurement_is_an_error_never_a_zero() {
        for bad in [
            "",
            "no json here\n",
            "{\"gpu_read_gbps\":",
            "{not json}",
            "{\"best_gbps\":280.3}",
            "{\"gpu_read_gbps\":null}",
            "{\"gpu_read_gbps\":\"fast\"}",
            "{\"gpu_read_gbps\":0}",
            "{\"gpu_read_gbps\":-5.0}",
        ] {
            assert!(parse_memory_probe(bad).is_err(), "{bad:?}");
        }
        let e = parse_memory_probe("{\"gpu_read_gbps\":")
            .unwrap_err()
            .to_string();
        assert!(e.contains("malformed"), "{e}");
        let e = parse_memory_probe("nothing\n").unwrap_err().to_string();
        assert!(e.contains("no measurement"), "{e}");
    }

    // ── Running the server binary, against a stand-in ───────────────────

    fn script(name: &str, body: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("lumen-mem-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let bin = root.join("fake-server");
        std::fs::write(
            &bin,
            format!("#!/bin/sh\n[ \"$1\" = --measure-memory-bandwidth ] || exit 9\n{body}\n"),
        )
        .unwrap();
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
        bin
    }

    fn cleanup(bin: &Path) {
        std::fs::remove_dir_all(bin.parent().unwrap()).ok();
    }

    #[tokio::test]
    async fn the_command_runs_the_server_with_the_flag_and_parses_its_line() {
        let bin = script("ok", &format!("echo 'warming up' >&2\necho '{LINE}'"));
        let p = run_memory_probe(&bin, Duration::from_secs(10))
            .await
            .unwrap();
        assert_eq!(p.gpu_read_gbps, 277.0);
        cleanup(&bin);
    }

    #[tokio::test]
    async fn a_malformed_line_is_an_error_that_says_so() {
        let bin = script(
            "bad",
            "echo '{\"gpu_read_gbps\":'\necho 'kernel compile failed' >&2",
        );
        let e = run_memory_probe(&bin, Duration::from_secs(10))
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains("malformed"), "{e}");
        assert!(
            e.contains("kernel compile failed"),
            "stderr is part of it: {e}"
        );
        cleanup(&bin);
        let bin = script("none", "echo 'hello'");
        let e = run_memory_probe(&bin, Duration::from_secs(10))
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains("no measurement"), "{e}");
        cleanup(&bin);
    }

    #[tokio::test]
    async fn a_failing_server_reports_its_exit_and_the_tail_of_stderr() {
        let bin = script(
            "fail",
            "echo 'line 1' >&2\necho 'line 2' >&2\necho 'Metal device not found' >&2\nexit 1",
        );
        let e = run_memory_probe(&bin, Duration::from_secs(10))
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains("failed"), "{e}");
        assert!(e.contains("Metal device not found"), "{e}");
        cleanup(&bin);
        // A line on stdout does not make a failed run a success.
        let bin = script("failjson", &format!("echo '{LINE}'\nexit 1"));
        assert!(
            run_memory_probe(&bin, Duration::from_secs(10))
                .await
                .is_err()
        );
        cleanup(&bin);
    }

    #[tokio::test]
    async fn a_missing_binary_and_a_hung_one_are_errors() {
        let missing = std::env::temp_dir().join("lumen-no-such-server-binary");
        assert!(
            run_memory_probe(&missing, Duration::from_secs(5))
                .await
                .is_err()
        );
        let bin = script("hang", "sleep 30");
        let started = std::time::Instant::now();
        let e = run_memory_probe(&bin, Duration::from_millis(300))
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains("did not finish"), "{e}");
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "it waited for the child"
        );
        cleanup(&bin);
    }

    // ── What is stored, and when it counts ──────────────────────────────

    const DIR: &str = "/Volumes/Fast/models/Flash-Next";

    fn measured() -> MeasuredHardware {
        let mut m = MeasuredHardware::default();
        m.record_memory(231.0, 1_000);
        m.record_ssd(3.1, Some(5.0), DIR, 2_000);
        m
    }

    #[test]
    fn nothing_measured_means_no_numbers_and_nothing_stale() {
        let v = MeasuredHardware::default().view(Some(DIR));
        assert_eq!(
            v,
            HardwareView {
                memory_gbps: None,
                memory_measured_at: None,
                ssd_random_gbps: None,
                ssd_sequential_gbps: None,
                ssd_measured_at: None,
                ssd_stale: false,
            }
        );
    }

    #[test]
    fn both_measured_on_this_folder_are_used_with_their_dates() {
        let v = measured().view(Some(DIR));
        assert_eq!(v.memory_gbps, Some(231.0));
        assert_eq!(v.memory_measured_at, Some(1_000));
        assert_eq!(v.ssd_random_gbps, Some(3.1));
        assert_eq!(v.ssd_sequential_gbps, Some(5.0));
        assert_eq!(v.ssd_measured_at, Some(2_000));
        assert!(!v.ssd_stale);
    }

    /// The SSD number belongs to the folder it read: another or moved folder (a different
    /// volume) makes it stale, while the memory number, which is the Mac's, stays.
    #[test]
    fn an_ssd_number_from_another_folder_is_stale_and_memory_is_unaffected() {
        let m = measured();
        for other in [
            Some("/Volumes/Slow/models/Flash-Next"),
            Some("/Volumes/Fast/models/Flash-Next-2"),
            None,
        ] {
            let v = m.view(other);
            assert_eq!(v.ssd_random_gbps, None, "{other:?}");
            assert_eq!(v.ssd_sequential_gbps, None);
            assert_eq!(v.ssd_measured_at, None);
            assert!(v.ssd_stale, "{other:?}");
            assert_eq!(v.memory_gbps, Some(231.0));
        }
        assert_eq!(
            m.ssd_random_for(Some("/Volumes/Slow/models/Flash-Next")),
            None
        );
        assert_eq!(m.ssd_random_for(Some(DIR)), Some(3.1));
    }

    #[test]
    fn a_partial_measurement_counts_only_what_exists() {
        let mut memory_only = MeasuredHardware::default();
        memory_only.record_memory(277.0, 5);
        let v = memory_only.view(Some(DIR));
        assert_eq!(
            (v.memory_gbps, v.ssd_random_gbps, v.ssd_stale),
            (Some(277.0), None, false)
        );

        let mut ssd_only = MeasuredHardware::default();
        ssd_only.record_ssd(6.5, None, DIR, 7);
        let v = ssd_only.view(Some(DIR));
        assert_eq!((v.memory_gbps, v.memory_measured_at), (None, None));
        assert_eq!(
            (v.ssd_random_gbps, v.ssd_sequential_gbps),
            (Some(6.5), None)
        );
    }

    #[test]
    fn a_number_that_is_not_a_positive_speed_does_not_count() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut m = measured();
            m.memory_gbps = Some(bad);
            m.ssd_random_gbps = Some(bad);
            let v = m.view(Some(DIR));
            assert_eq!((v.memory_gbps, v.memory_measured_at), (None, None), "{bad}");
            assert_eq!(v.ssd_random_gbps, None, "{bad}");
            assert!(
                !v.ssd_stale,
                "{bad}: nothing usable is stored, so nothing is stale"
            );
        }
    }

    #[test]
    fn recording_one_number_leaves_the_other_and_a_new_run_replaces_the_old() {
        let mut m = measured();
        m.record_memory(250.0, 3_000);
        assert_eq!(
            (m.memory_gbps, m.memory_measured_at),
            (Some(250.0), Some(3_000))
        );
        assert_eq!(
            (m.ssd_random_gbps, m.ssd_measured_at),
            (Some(3.1), Some(2_000))
        );
        // The SSD re-measured on another folder moves the number to that folder.
        m.record_ssd(5.5, None, "/Volumes/Slow/m", 4_000);
        assert_eq!(
            m.ssd_sequential_gbps, None,
            "an old sequential number does not linger"
        );
        assert_eq!(m.view(Some("/Volumes/Slow/m")).ssd_random_gbps, Some(5.5));
        assert!(m.view(Some(DIR)).ssd_stale);
    }

    // ── How the action ends ─────────────────────────────────────────────

    #[test]
    fn both_parts_measured_is_done() {
        let r = finish(Ok(231.0), SsdOutcome::Measured(3.1), false);
        assert_eq!(r.state, MeasureState::Done);
        assert_eq!((r.memory_gbps, r.ssd_random_gbps), (Some(231.0), Some(3.1)));
        assert!(!r.ssd_skipped && r.memory_error.is_none() && r.ssd_error.is_none());
    }

    #[test]
    fn without_the_model_only_the_memory_part_runs_and_that_is_enough() {
        let r = finish(Ok(231.0), SsdOutcome::NotInstalled, false);
        assert_eq!(r.state, MeasureState::Done);
        assert!(r.ssd_skipped);
        assert_eq!(r.ssd_random_gbps, None);
    }

    #[test]
    fn one_part_failing_keeps_the_other_and_says_what_failed() {
        let r = finish(
            Err("no server binary".into()),
            SsdOutcome::Measured(3.1),
            false,
        );
        assert_eq!(r.state, MeasureState::Done);
        assert_eq!(r.memory_error.as_deref(), Some("no server binary"));
        assert_eq!(r.ssd_random_gbps, Some(3.1));
        let r = finish(Ok(231.0), SsdOutcome::Failed("read failed".into()), false);
        assert_eq!(r.state, MeasureState::Done);
        assert_eq!(r.ssd_error.as_deref(), Some("read failed"));
        assert_eq!(r.memory_gbps, Some(231.0));
    }

    #[test]
    fn nothing_measured_is_a_failure_that_carries_the_errors() {
        let r = finish(
            Err("metal error".into()),
            SsdOutcome::Failed("read failed".into()),
            false,
        );
        assert_eq!(r.state, MeasureState::Failed);
        assert_eq!(r.memory_error.as_deref(), Some("metal error"));
        assert_eq!(r.ssd_error.as_deref(), Some("read failed"));
        // No model, and the memory test failed: nothing to show but the error.
        let r = finish(Err("metal error".into()), SsdOutcome::NotInstalled, false);
        assert_eq!(r.state, MeasureState::Failed);
        assert!(r.ssd_skipped);
    }

    #[test]
    fn cancelling_wins_and_keeps_what_was_already_measured() {
        let r = finish(Ok(231.0), SsdOutcome::Cancelled, true);
        assert_eq!(r.state, MeasureState::Cancelled);
        assert_eq!(r.memory_gbps, Some(231.0));
        // The SSD test saw the flag by itself.
        assert_eq!(
            finish(Ok(231.0), SsdOutcome::Cancelled, false).state,
            MeasureState::Cancelled
        );
        // Cancelled before the memory test could finish.
        assert_eq!(
            finish(Err("x".into()), SsdOutcome::Cancelled, true).state,
            MeasureState::Cancelled
        );
    }

    #[test]
    fn the_events_speak_the_json_the_frontend_reads() {
        let json = serde_json::to_value(MeasureProgress::running(MeasureStage::Ssd)).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "state": "running", "stage": "ssd", "memory_gbps": null, "ssd_random_gbps": null,
                "ssd_skipped": false, "memory_error": null, "ssd_error": null,
            })
        );
        let json = serde_json::to_value(measured().view(Some(DIR))).unwrap();
        assert_eq!(json["ssd_stale"], false);
        assert_eq!(json["memory_gbps"], 231.0);
    }
}
