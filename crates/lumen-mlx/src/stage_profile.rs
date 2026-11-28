//! Opt-in profile of a decode step, stage by stage, for any model family.
//!
//! A model marks the end of each stage of its forward pass; with its stage flag on
//! (`LUMEN_GEMMA4_STAGE_SYNC`), a mark evaluates the stage's outputs and
//! charges what happened since the previous mark to the stage: the time spent building the lazy
//! graph, the time the evaluation took, and what MLX counted meanwhile (primitives encoded, kernel
//! dispatches, command buffers, CPU time spent encoding). The counts and the encode time are exact
//! whatever the barriers cost; the evaluation wall time includes a synchronisation per stage, so
//! read it as a ratio, not as an absolute. Off, a mark costs one thread-local read.
//!
//! The model decides which passes are profiled ([`begin`]); a prefill runs unchanged.
//! `examples/gemma4_profile.rs` prints the report.

use std::cell::Cell;
use std::fmt::Write as _;
use std::sync::Mutex;
use std::time::Instant;

use anyhow::Result;
use mlx_rs::Array;

thread_local! {
    /// Whether the pass being built on this thread is a profiled decode step.
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
}

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
}

struct Stage {
    name: &'static str,
    calls: u64,
    build_ns: u64,
    eval_ns: u64,
    counters: Counters,
}

struct State {
    steps: u64,
    last: Instant,
    stages: Vec<Stage>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

fn state() -> std::sync::MutexGuard<'static, Option<State>> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Starts a pass: profiled when `on` (the model's stage flag is on and the pass is a decode step).
pub(crate) fn begin(on: bool) {
    ACTIVE.with(|a| a.set(on));
    if on {
        let mut guard = state();
        let st = guard.get_or_insert_with(|| State {
            steps: 0,
            last: Instant::now(),
            stages: Vec::new(),
        });
        st.steps += 1;
        st.last = Instant::now();
    }
}

/// Ends a stage: evaluates `outs` and charges everything since the previous mark to `stage`.
pub(crate) fn mark(stage: &'static str, outs: &[&Array]) -> Result<()> {
    if !ACTIVE.with(Cell::get) {
        return Ok(());
    }
    let built = Instant::now();
    let before = Counters::now();
    mlx_rs::transforms::eval(outs.iter().copied())?;
    let evaluated = Instant::now();
    let counted = Counters::now().since(before);
    let mut guard = state();
    let Some(st) = guard.as_mut() else {
        return Ok(());
    };
    let at = match st.stages.iter().position(|s| s.name == stage) {
        Some(at) => at,
        None => {
            st.stages.push(Stage {
                name: stage,
                calls: 0,
                build_ns: 0,
                eval_ns: 0,
                counters: Counters::default(),
            });
            st.stages.len() - 1
        }
    };
    let acc = &mut st.stages[at];
    acc.calls += 1;
    acc.build_ns += built.duration_since(st.last).as_nanos() as u64;
    acc.eval_ns += evaluated.duration_since(built).as_nanos() as u64;
    acc.counters.commits += counted.commits;
    acc.counters.dispatches += counted.dispatches;
    acc.counters.prims += counted.prims;
    acc.counters.encode_ns += counted.encode_ns;
    st.last = Instant::now();
    Ok(())
}

/// Forgets what was recorded.
pub fn stage_reset() {
    *state() = None;
}

/// The recorded stages per decode step, in the order they first ran; `None` before any.
pub fn stage_report() -> Option<String> {
    let guard = state();
    let st = guard.as_ref()?;
    let steps = st.steps.max(1) as f64;
    let ms = |ns: u64| ns as f64 / 1e6 / steps;
    let total_wall: f64 = st.stages.iter().map(|s| ms(s.build_ns + s.eval_ns)).sum();
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{:<34} {:>6} {:>8} {:>8} {:>8} {:>9} {:>8} {:>6}",
        "stage (per decode step)",
        "calls",
        "build ms",
        "eval ms",
        "prims",
        "dispatch",
        "enc ms",
        "wall%"
    );
    let mut sum = Counters::default();
    for s in &st.stages {
        let wall = ms(s.build_ns + s.eval_ns);
        let _ = writeln!(
            out,
            "{:<34} {:>6.0} {:>8.2} {:>8.2} {:>8.0} {:>9.0} {:>8.2} {:>5.1}%",
            s.name,
            s.calls as f64 / steps,
            ms(s.build_ns),
            ms(s.eval_ns),
            s.counters.prims as f64 / steps,
            s.counters.dispatches as f64 / steps,
            ms(s.counters.encode_ns),
            wall / total_wall.max(1e-9) * 100.0
        );
        sum.prims += s.counters.prims;
        sum.dispatches += s.counters.dispatches;
        sum.encode_ns += s.counters.encode_ns;
    }
    let _ = writeln!(
        out,
        "{:<34} {:>6} {:>8.2} {:>8.2} {:>8.0} {:>9.0} {:>8.2}",
        "total",
        "",
        st.stages.iter().map(|s| ms(s.build_ns)).sum::<f64>(),
        st.stages.iter().map(|s| ms(s.eval_ns)).sum::<f64>(),
        sum.prims as f64 / steps,
        sum.dispatches as f64 / steps,
        ms(sum.encode_ns)
    );
    Some(out)
}
