//! Opt-in profile of one offload-served decode step (task 023, lever A6: CPU / launch overhead).
//!
//! Two diagnostics, both off by default. Off, a call site costs one relaxed atomic load.
//!
//! * `LUMEN_QWEN4_TOKEN_COUNTERS`: per decode step, how many Metal command buffers and kernel
//!   dispatches MLX made, how many primitives it encoded and how long the CPU spent encoding them
//!   (the fork's counters). No extra synchronisation, so the step runs at full speed.
//! * `LUMEN_QWEN4_STAGE_SYNC`: evaluates the step in stages (hyper-connection, mixer, router,
//!   experts, ...) and records each stage's graph-build time, evaluation wall time and counter
//!   deltas. The barriers cost time of their own, so the totals over-count: read the stage
//!   dispatch counts and the ratios, and subtract the `sync floor` line (an evaluation of a single
//!   trivial op after the same idle gap) from each stage to estimate its own cost.
//!
//! Only decode steps (at most [`DECODE_ROWS`] rows) are profiled; prefill runs unchanged.

use std::cell::Cell;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use mlx_rs::Array;

/// Passes of at most this many rows count as decode steps.
const DECODE_ROWS: usize = 4;
/// Print the cumulative report after this many profiled steps.
const REPORT_EVERY_STEPS: u64 = 100;

lumen_flags::flag! {
    /// Count the Metal command buffers, kernel dispatches and encoded primitives of every decode
    /// step, plus the CPU time spent encoding, and print them every 100 steps. Adds no sync.
    pub(crate) token_counters {
        env: "LUMEN_QWEN4_TOKEN_COUNTERS",
        default: false,
        kind: Diagnostic,
    }
}

lumen_flags::flag! {
    /// Evaluate every decode step stage by stage (hyper-connection, mixer, router, route, experts,
    /// shared expert, write-back) and print each stage's build time, evaluation time and
    /// dispatch counts. Costs a sync per stage; for diagnosis only.
    pub(crate) stage_sync {
        env: "LUMEN_QWEN4_STAGE_SYNC",
        default: false,
        kind: Diagnostic,
    }
}

/// The stages of a decode step, in execution order. `Route` is the offload's readback and expert
/// loading, timed rather than evaluated.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Stage {
    Embed,
    Ple,
    HcAttn,
    MixerGdn,
    MixerQsa,
    Wb1,
    HcMlp,
    Router,
    Route,
    Experts,
    Shared,
    Wb2,
    Tail,
}

const STAGES: [(Stage, &str); 13] = [
    (Stage::Embed, "embed"),
    (Stage::Ple, "n-gram injection"),
    (Stage::HcAttn, "attn hyper-connection"),
    (Stage::MixerGdn, "mixer GDN (+state)"),
    (Stage::MixerQsa, "mixer QSA (+kv)"),
    (Stage::Wb1, "write-back 1"),
    (Stage::HcMlp, "mlp hyper-connection"),
    (Stage::Router, "router gate+top-k"),
    (Stage::Route, "route (readback+SSD)"),
    (Stage::Experts, "routed experts"),
    (Stage::Shared, "shared expert"),
    (Stage::Wb2, "add+write-back 2"),
    (Stage::Tail, "tail (mix+lm_head)"),
];

/// What MLX counted between two points.
#[derive(Clone, Copy, Default)]
struct Counters {
    commits: u64,
    dispatches: u64,
    prims: u64,
    encode_ns: u64,
}

impl Counters {
    fn now() -> Self {
        let (commits, dispatches) = mlx_rs::metal::cmd_buffer_stats().unwrap_or_default();
        let (prims, encode_ns) = mlx_rs::metal::eval_gpu_stats().unwrap_or_default();
        Self {
            commits,
            dispatches,
            prims,
            encode_ns,
        }
    }

    fn since(self, earlier: Self) -> Self {
        Self {
            commits: self.commits.saturating_sub(earlier.commits),
            dispatches: self.dispatches.saturating_sub(earlier.dispatches),
            prims: self.prims.saturating_sub(earlier.prims),
            encode_ns: self.encode_ns.saturating_sub(earlier.encode_ns),
        }
    }

    fn add(&mut self, o: Self) {
        self.commits += o.commits;
        self.dispatches += o.dispatches;
        self.prims += o.prims;
        self.encode_ns += o.encode_ns;
    }
}

#[derive(Clone, Copy, Default)]
struct StageAcc {
    calls: u64,
    build_ns: u64,
    eval_ns: u64,
    counters: Counters,
}

#[derive(Default)]
struct Totals {
    steps: u64,
    wall_ns: u64,
    counters: Counters,
    stages: [StageAcc; STAGES.len()],
    /// Median wall time of evaluating one trivial op after a short idle gap.
    sync_floor_ns: Option<u64>,
}

static TOTALS: Mutex<Option<Totals>> = Mutex::new(None);

thread_local! {
    /// Whether the current pass is a profiled decode step.
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    /// When the stage in progress started building (the end of the previous stage).
    static LAST: Cell<Option<Instant>> = const { Cell::new(None) };
    static STEP_START: Cell<Option<(Instant, Counters)>> = const { Cell::new(None) };
}

fn with_totals<R>(f: impl FnOnce(&mut Totals) -> R) -> R {
    let mut guard = TOTALS.lock().unwrap_or_else(|p| p.into_inner());
    f(guard.get_or_insert_with(Totals::default))
}

/// Whether the pass in progress evaluates stage by stage.
#[inline]
pub(crate) fn sync_on() -> bool {
    stage_sync::get() && ACTIVE.with(Cell::get)
}

/// Start profiling a pass of `rows` rows (a no-op for prefill and when both flags are off).
pub(crate) fn begin_step(rows: usize) {
    let counting = token_counters::get();
    let staging = stage_sync::get();
    if !(counting || staging) || rows > DECODE_ROWS {
        ACTIVE.with(|a| a.set(false));
        return;
    }
    ACTIVE.with(|a| a.set(true));
    if staging {
        ensure_sync_floor();
    }
    let now = Instant::now();
    LAST.with(|l| l.set(Some(now)));
    STEP_START.with(|s| s.set(Some((now, Counters::now()))));
}

/// Finish the pass: fold its totals in and print the report now and then.
pub(crate) fn end_step() {
    if !ACTIVE.with(|a| a.replace(false)) {
        return;
    }
    let Some((t0, c0)) = STEP_START.with(Cell::take) else {
        return;
    };
    let (wall, delta) = (t0.elapsed(), Counters::now().since(c0));
    let report = with_totals(|t| {
        t.steps += 1;
        t.wall_ns += wall.as_nanos() as u64;
        t.counters.add(delta);
        (t.steps.is_multiple_of(REPORT_EVERY_STEPS)).then(|| render(t))
    });
    if let Some(r) = report {
        eprintln!("{r}");
    }
}

/// Close a stage: its graph-build time is what passed since the previous stage ended, then the
/// `arrays` are evaluated and the counters of that evaluation recorded. No-op unless
/// [`sync_on`].
pub(crate) fn stage(which: Stage, arrays: &[&Array]) -> Result<()> {
    if !sync_on() {
        return Ok(());
    }
    let entered = Instant::now();
    let build = LAST
        .with(Cell::get)
        .map_or(Duration::ZERO, |l| entered.duration_since(l));
    let before = Counters::now();
    mlx_rs::transforms::eval(arrays.iter().copied())
        .with_context(|| format!("stage profile: evaluate {which:?}"))?;
    let delta = Counters::now().since(before);
    let left = Instant::now();
    record(which, build, left.duration_since(entered), delta);
    LAST.with(|l| l.set(Some(left)));
    Ok(())
}

/// Record a stage that was timed from outside (the offload route), and restart the build clock.
pub(crate) fn timed(which: Stage, started: Instant) {
    let build = LAST
        .with(Cell::get)
        .map_or(Duration::ZERO, |l| started.duration_since(l));
    let left = Instant::now();
    record(
        which,
        build,
        left.duration_since(started),
        Counters::default(),
    );
    LAST.with(|l| l.set(Some(left)));
}

fn record(which: Stage, build: Duration, eval: Duration, delta: Counters) {
    let i = STAGES.iter().position(|(s, _)| *s == which).unwrap_or(0);
    with_totals(|t| {
        let a = &mut t.stages[i];
        a.calls += 1;
        a.build_ns += build.as_nanos() as u64;
        a.eval_ns += eval.as_nanos() as u64;
        a.counters.add(delta);
    });
}

/// Median wall time of evaluating one trivial op, after an idle gap about as long as a stage's
/// graph build, so the GPU is as asleep as it is between stages. Measured once.
fn ensure_sync_floor() {
    if with_totals(|t| t.sync_floor_ns.is_some()) {
        return;
    }
    let a = Array::from_slice(&[1.0f32; 64], &[1, 64]);
    if a.eval().is_err() {
        return;
    }
    let mut samples: Vec<u64> = Vec::with_capacity(300);
    for _ in 0..300 {
        let idle = Instant::now();
        while idle.elapsed() < Duration::from_micros(100) {
            std::hint::spin_loop();
        }
        let Ok(b) = mlx_rs::ops::add(&a, &a) else {
            return;
        };
        let t = Instant::now();
        if b.eval().is_err() {
            return;
        }
        samples.push(t.elapsed().as_nanos() as u64);
    }
    samples.sort_unstable();
    let median = samples[samples.len() / 2];
    with_totals(|t| t.sync_floor_ns = Some(median));
}

/// The cumulative report, `None` when nothing was profiled.
pub(crate) fn report() -> Option<String> {
    with_totals(|t| (t.steps > 0).then(|| render(t)))
}

fn render(t: &Totals) -> String {
    let steps = t.steps.max(1) as f64;
    let us = |ns: u64| ns as f64 / 1e3;
    let mut out = format!(
        "[decode-profile] {} steps: {:.2} ms/step wall; per step {:.0} command buffers, \
         {:.0} dispatches, {:.0} primitives encoded, {:.0} us CPU spent encoding",
        t.steps,
        t.wall_ns as f64 / steps / 1e6,
        t.counters.commits as f64 / steps,
        t.counters.dispatches as f64 / steps,
        t.counters.prims as f64 / steps,
        us(t.counters.encode_ns) / steps,
    );
    if t.stages.iter().all(|s| s.calls == 0) {
        return out;
    }
    let floor = t.sync_floor_ns.unwrap_or(0);
    out.push_str(&format!(
        "\n[decode-profile] sync floor (one trivial op, evaluated after a 100 us idle gap): {:.0} us\n\
         [decode-profile] {:<24} {:>6} {:>9} {:>9} {:>9} {:>7} {:>7} {:>7} {:>8} | per-step ms (build / eval-floor)",
        us(floor),
        "stage",
        "calls",
        "build us",
        "eval us",
        "-floor",
        "cbufs",
        "disp",
        "prims",
        "enc us",
    ));
    let (mut build_sum, mut eval_sum) = (0.0, 0.0);
    for ((_, name), s) in STAGES.iter().zip(&t.stages) {
        if s.calls == 0 {
            continue;
        }
        let c = s.calls as f64;
        let build = us(s.build_ns) / c;
        let eval = us(s.eval_ns) / c;
        let net = (eval - us(floor)).max(0.0);
        let (build_step, eval_step) = (build * c / steps / 1e3, net * c / steps / 1e3);
        build_sum += build_step;
        eval_sum += eval_step;
        out.push_str(&format!(
            "\n[decode-profile] {:<24} {:>6} {:>9.1} {:>9.1} {:>9.1} {:>7.2} {:>7.1} {:>7.1} {:>8.1} | {:.2} / {:.2}",
            name,
            s.calls,
            build,
            eval,
            net,
            s.counters.commits as f64 / c,
            s.counters.dispatches as f64 / c,
            s.counters.prims as f64 / c,
            us(s.counters.encode_ns) / c,
            build_step,
            eval_step,
        ));
    }
    out.push_str(&format!(
        "\n[decode-profile] sum over stages per step: build {:.2} ms, eval-floor {:.2} ms",
        build_sum, eval_sum
    ));
    out
}
