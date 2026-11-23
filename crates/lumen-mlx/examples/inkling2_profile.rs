//! Where one plain decode step of Inkling 2 goes: how long the graph takes to build, how long
//! MLX takes to encode and run it, how many Metal command buffers, kernel dispatches and
//! primitives a step makes, and how busy the CPU is. No extra synchronisation: the step runs
//! as the server runs it (build, evaluate, read the argmax), and the counters are the MLX fork's.
//!
//! ```text
//! cargo run --release -p lumen-mlx --features mlx-native --example inkling2_profile -- \
//!     <model_dir> [--steps N] [--warm W] ["<prompt>"]
//! ```
//!
//! With `LUMEN_INKLING2_STAGE_SYNC=1` the step is also evaluated stage by stage (input mixing,
//! projections, attention, router, experts, ...) and each stage's primitives, dispatches and
//! encode time are listed; run it without the flag for the step's true wall time.
//!
//! Reading the numbers: `build` is the Rust and FFI work of making the lazy graph; `eval` is
//! MLX encoding every primitive (`encode` of it is CPU) and then waiting for the GPU. If the CPU
//! time per step is close to the wall time and `encode` is most of `eval`, the step is bound by
//! launching work, not by the weights it reads.

#![cfg(feature = "mlx-native")]

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use lumen_mlx::inkling2::{Inkling2Model, LoadOptions};
use mlx_rs::Array;
use tokenizers::Tokenizer;

fn host_f32(a: &Array) -> Result<Vec<f32>> {
    let a = a.as_dtype(mlx_rs::Dtype::Float32)?;
    a.eval()?;
    Ok(a.try_as_slice::<f32>()?.to_vec())
}

fn argmax(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// CPU time (user + system, every thread of the process) in microseconds.
fn cpu_micros() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: `getrusage` fills the struct it is given.
    let usage = unsafe {
        libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr());
        usage.assume_init()
    };
    let micros = |t: libc::timeval| t.tv_sec as u64 * 1_000_000 + t.tv_usec as u64;
    micros(usage.ru_utime) + micros(usage.ru_stime)
}

#[derive(Clone, Copy, Default)]
struct Step {
    build_ms: f64,
    eval_ms: f64,
    pick_ms: f64,
    commits: u64,
    dispatches: u64,
    prims: u64,
    encode_ms: f64,
}

fn mean(xs: impl Iterator<Item = f64> + Clone) -> f64 {
    let n = xs.clone().count().max(1) as f64;
    xs.sum::<f64>() / n
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(f64::total_cmp);
    xs[xs.len() / 2]
}

const PROMPT: &str = "Write a short story about a lighthouse keeper who finds a message in a bottle. \
The keeper lived alone on a rocky island far from the mainland, and every evening she climbed the \
spiral stairs to light the great lamp. One stormy night, while the waves crashed against the rocks, \
she noticed something glinting in the foam below. It was a green glass bottle, sealed with wax, and \
inside was a folded piece of paper. She";

/// What launching tiny dependent kernels costs on this machine, with no model: a chain of `n`
/// elementwise ops on one row of 2048 bf16 values, evaluated once. Divide the step's dispatch
/// count by it to see how much of a decode step is the launch floor.
fn launch_floor(n: usize) -> Result<()> {
    let one = Array::from_f32(1.0).as_dtype(mlx_rs::Dtype::Bfloat16)?;
    for round in 0..4 {
        let mut x = mlx_rs::ops::zeros::<f32>(&[1, 1, 2048])?.as_dtype(mlx_rs::Dtype::Bfloat16)?;
        x.eval()?;
        let (c0, d0) = mlx_rs::metal::cmd_buffer_stats().unwrap_or_default();
        let (p0, e0) = mlx_rs::metal::eval_gpu_stats().unwrap_or_default();
        let t0 = Instant::now();
        for _ in 0..n {
            x = mlx_rs::ops::add(&x, &one)?;
        }
        let t1 = Instant::now();
        x.eval()?;
        let t2 = Instant::now();
        let (c1, d1) = mlx_rs::metal::cmd_buffer_stats().unwrap_or_default();
        let (p1, e1) = mlx_rs::metal::eval_gpu_stats().unwrap_or_default();
        let (build, eval) = ((t1 - t0).as_secs_f64() * 1e3, (t2 - t1).as_secs_f64() * 1e3);
        let dispatches = d1.saturating_sub(d0).max(1) as f64;
        println!(
            "launch floor, round {round}: {n} dependent adds: build {build:.2} ms, eval {eval:.2} ms \
             ({:.2} us per dispatch), {} primitives, {} dispatches in {} command buffers, CPU encoding {:.2} ms ({:.2} us per primitive)",
            eval * 1e3 / dispatches,
            p1.saturating_sub(p0),
            d1.saturating_sub(d0),
            c1.saturating_sub(c0),
            e1.saturating_sub(e0) as f64 / 1e6,
            e1.saturating_sub(e0) as f64 / 1e3 / p1.saturating_sub(p0).max(1) as f64,
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(at) = args.iter().position(|a| a == "--floor") {
        args.remove(at);
        return launch_floor(11_600);
    }
    let mut flag_value = |name: &str| -> Option<String> {
        let at = args.iter().position(|a| a == name)?;
        let value = args.get(at + 1).cloned();
        args.drain(at..(at + 2).min(args.len()));
        value
    };
    let steps: usize = flag_value("--steps")
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    let warm: usize = flag_value("--warm")
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let dir = PathBuf::from(
        args.first()
            .ok_or_else(|| anyhow!("usage: <model_dir> [--steps N] [--warm W] [\"<prompt>\"]"))?,
    );
    let prompt = args.get(1).cloned().unwrap_or_else(|| PROMPT.into());
    let tok =
        Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("tokenizer: {e}"))?;
    let ids: Vec<u32> = tok
        .encode(prompt.as_str(), false)
        .map_err(|e| anyhow!("encode: {e}"))?
        .get_ids()
        .to_vec();

    let model = Inkling2Model::load(&dir, LoadOptions::default()).context("load the checkpoint")?;
    let live = model.args().unpadded_vocab_size;
    println!(
        "{} prompt tokens, {} layer visits per token, decoding {} steps after {} warm-up steps",
        ids.len(),
        model.num_visits(),
        steps,
        warm
    );

    let mut cache = model.make_cache();
    let logits = model.prefill(&ids, &mut cache, 2048)?;
    let row = host_f32(&logits)?;
    let mut next = argmax(&row[row.len() - model.args().vocab_size..][..live]);
    let mut generated = Vec::new();

    let mut recorded: Vec<Step> = Vec::new();
    let (mut cpu_start, mut wall_start) = (0u64, Instant::now());
    for step in 0..warm + steps {
        if step == warm {
            mlx_rs::metal::reset_prim_histogram_dynamic().ok();
            lumen_mlx::inkling2::stage_reset();
            cpu_start = cpu_micros();
            wall_start = Instant::now();
        }
        generated.push(next);
        let (c0, d0) = mlx_rs::metal::cmd_buffer_stats().unwrap_or_default();
        let (p0, e0) = mlx_rs::metal::eval_gpu_stats().unwrap_or_default();

        let t0 = Instant::now();
        let logits = model.forward(&[next], &mut cache, true)?;
        let t1 = Instant::now();
        logits.eval()?;
        let t2 = Instant::now();
        let host = logits.as_dtype(mlx_rs::Dtype::Float32)?;
        host.eval()?;
        next = argmax(&host.try_as_slice::<f32>()?[..live]);
        let t3 = Instant::now();

        let (c1, d1) = mlx_rs::metal::cmd_buffer_stats().unwrap_or_default();
        let (p1, e1) = mlx_rs::metal::eval_gpu_stats().unwrap_or_default();
        if step >= warm {
            recorded.push(Step {
                build_ms: (t1 - t0).as_secs_f64() * 1e3,
                eval_ms: (t2 - t1).as_secs_f64() * 1e3,
                pick_ms: (t3 - t2).as_secs_f64() * 1e3,
                commits: c1.saturating_sub(c0),
                dispatches: d1.saturating_sub(d0),
                prims: p1.saturating_sub(p0),
                encode_ms: e1.saturating_sub(e0) as f64 / 1e6,
            });
        }
    }
    let wall = wall_start.elapsed().as_secs_f64();
    let cpu = (cpu_micros() - cpu_start) as f64 / 1e6;
    println!(
        "continuation: {:?}",
        tok.decode(&generated, false).unwrap_or_default()
    );

    let n = recorded.len() as f64;
    let total: Vec<f64> = recorded
        .iter()
        .map(|s| s.build_ms + s.eval_ms + s.pick_ms)
        .collect();
    println!("\nper decode step (mean of {} steps):", recorded.len());
    println!(
        "  wall {:.2} ms (median {:.2})  =  build {:.2} + eval {:.2} + readback {:.2}   -> {:.1} tok/s",
        mean(total.iter().copied()),
        median(total.clone()),
        mean(recorded.iter().map(|s| s.build_ms)),
        mean(recorded.iter().map(|s| s.eval_ms)),
        mean(recorded.iter().map(|s| s.pick_ms)),
        1e3 / mean(total.iter().copied()),
    );
    println!(
        "  MLX per step: {:.0} primitives, {:.0} kernel dispatches in {:.1} command buffers; CPU encoding {:.2} ms",
        mean(recorded.iter().map(|s| s.prims as f64)),
        mean(recorded.iter().map(|s| s.dispatches as f64)),
        mean(recorded.iter().map(|s| s.commits as f64)),
        mean(recorded.iter().map(|s| s.encode_ms)),
    );
    println!(
        "  process CPU time {:.2} ms per step against {:.2} ms of wall ({:.0}% of one core, every thread)",
        cpu / n * 1e3,
        wall / n * 1e3,
        cpu / wall * 100.0
    );
    let per_visit = model.num_visits() as f64;
    println!(
        "  per layer visit: {:.0} primitives, {:.2} ms wall",
        mean(recorded.iter().map(|s| s.prims as f64)) / per_visit,
        mean(total.iter().copied()) / per_visit
    );

    // The primitive mix of the measured steps.
    let histogram = mlx_rs::metal::prim_histogram_dynamic().unwrap_or_default();
    let mut counts: Vec<(String, f64)> = histogram
        .lines()
        .filter_map(|l| {
            let (name, count) = l.split_once('=')?;
            Some((name.to_string(), count.trim().parse::<f64>().ok()? / n))
        })
        .collect();
    counts.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!("\nprimitives per step, by kind:");
    for (name, count) in counts.iter().take(40) {
        println!("  {count:>7.1}  {name}");
    }
    if let Some(report) = lumen_mlx::inkling2::stage_report() {
        println!(
            "\nstage by stage (LUMEN_INKLING2_STAGE_SYNC=1: a barrier after every stage, so the wall \
             times are inflated; the primitive, dispatch and encode numbers are exact):\n{report}"
        );
    }
    Ok(())
}
