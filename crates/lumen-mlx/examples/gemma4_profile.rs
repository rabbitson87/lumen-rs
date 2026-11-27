//! Where one plain decode step of Gemma 4 goes: how long the graph takes to build, how long MLX
//! takes to encode and run it, how many Metal command buffers, kernel dispatches and primitives a
//! step makes, and how busy the CPU is. The counters are the MLX fork's.
//!
//! ```text
//! cargo run --release -p lumen-mlx --features mlx-native --example gemma4_profile -- \
//!     <model_dir> [--steps N] [--warm W] ["<prompt>"]
//! cargo run --release -p lumen-mlx --features mlx-native --example gemma4_profile -- --floor 2500
//! cargo run --release -p lumen-mlx --features mlx-native --example gemma4_profile -- \
//!     <model_dir> --ab LUMEN_GEMMA4_NORM_KERNELS [--rounds N]
//! ```
//!
//! `--ab FLAG` loads the model once and times the pipelined loop in ABBA blocks with a decode
//! kernel's flag off and on (`set_decode_kernel_flag`), then prints the paired difference and
//! whether every run gave the same continuation.
//!
//! Two loops run over the same greedy continuation:
//! - `pipelined`: the server's loop (`generate_with_cache_and_images`): build step N+1 and
//!   `async_eval` it before reading step N's token. Its ms per token is the throughput a request sees.
//! - `serial`: build, evaluate, read, one step at a time, with MLX's counters read around each step.
//!   Its `build` is the Rust and FFI work of making the lazy graph; `eval` is MLX encoding every
//!   primitive (`encode` of it is CPU) and then waiting for the GPU.
//!
//! With `LUMEN_GEMMA4_STAGE_SYNC=1` the serial loop is also evaluated stage by stage and each
//! stage's primitives, dispatches and encode time are listed; its wall times are inflated by the
//! barriers, so read the pipelined number for time and the stage table for counts.
//!
//! `digest` is an FNV-1a hash of the generated ids: two runs that print the same digest produced the
//! same continuation.

#![cfg(feature = "mlx-native")]

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use lumen_mlx::gemma4::{NativeGemma4Model, NativeGemma4PromptCache};
use mlx_rs::Array;
use tokenizers::Tokenizer;

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

fn digest(ids: &[u32]) -> u64 {
    ids.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &id| {
        id.to_le_bytes()
            .iter()
            .fold(h, |h, &b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3))
    })
}

const PROMPT: &str = "Write a short story about a lighthouse keeper who finds a message in a bottle. \
The keeper lived alone on a rocky island far from the mainland, and every evening she climbed the \
spiral stairs to light the great lamp. One stormy night, while the waves crashed against the rocks, \
she noticed something glinting in the foam below. It was a green glass bottle, sealed with wax, and \
inside was a folded piece of paper. She";

/// What launching tiny dependent kernels costs on this machine, with no model: a chain of `n`
/// elementwise ops on one row of 2816 bf16 values (Gemma 4 26B's hidden width), evaluated once.
fn launch_floor(n: usize) -> Result<()> {
    let one = Array::from_f32(1.0).as_dtype(mlx_rs::Dtype::Bfloat16)?;
    for round in 0..4 {
        let mut x = mlx_rs::ops::zeros::<f32>(&[1, 1, 2816])?.as_dtype(mlx_rs::Dtype::Bfloat16)?;
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
             ({:.2} us per dispatch), {} primitives, {} dispatches in {} command buffers, CPU encoding {:.2} ms",
            eval * 1e3 / dispatches,
            p1.saturating_sub(p0),
            d1.saturating_sub(d0),
            c1.saturating_sub(c0),
            e1.saturating_sub(e0) as f64 / 1e6,
        );
    }
    Ok(())
}

/// Prefill `ids` into a fresh cache; returns the cache and the lazy first token.
fn prefill(model: &NativeGemma4Model, ids: &[u32]) -> Result<(NativeGemma4PromptCache, Array)> {
    let mut cache = model.make_cache();
    let logits = model.forward_last_token(ids, &mut cache)?;
    let first = model.argmax_last_token_lazy(&logits)?;
    first.eval()?;
    Ok((cache, first))
}

/// The server's loop: schedule step N+1 before reading step N. Returns ms per token over the
/// steps after `warm`, and every token read.
fn pipelined(
    model: &NativeGemma4Model,
    ids: &[u32],
    warm: usize,
    steps: usize,
) -> Result<(f64, Vec<u32>)> {
    let (mut cache, mut current) = prefill(model, ids)?;
    let mut generated = Vec::with_capacity(warm + steps);
    let mut started = Instant::now();
    for step in 0..warm + steps {
        if step == warm {
            started = Instant::now();
        }
        let logits = model.forward_array_last_token(&current, &mut cache)?;
        let next = model.argmax_last_token_lazy(&logits)?;
        mlx_rs::transforms::async_eval([&next])?;
        generated.push(model.read_token_u32(&current)?);
        current = next;
    }
    let ms = started.elapsed().as_secs_f64() * 1e3 / steps.max(1) as f64;
    generated.push(model.read_token_u32(&current)?);
    Ok((ms, generated))
}

/// Same-process ABBA blocks of the pipelined loop, `flag` off (A) and on (B), pinned through the
/// flag's own setter so a latched flag cannot measure one side twice. Prints each block, then the
/// mean of each arm, the paired difference with its standard error, and whether every run of both
/// arms produced the same continuation.
fn ab(
    model: &NativeGemma4Model,
    ids: &[u32],
    flag: &str,
    rounds: usize,
    warm: usize,
    steps: usize,
) -> Result<()> {
    println!("A/B {flag}: {rounds} ABBA blocks, {steps} timed steps a run after {warm} warm-up");
    let run = |on: bool| -> Result<(f64, u64)> {
        lumen_mlx::gemma4::set_decode_kernel_flag(flag, on)?;
        let (ms, tokens) = pipelined(model, ids, warm, steps)?;
        Ok((ms, digest(&tokens)))
    };
    // One unrecorded run of each arm: the first call of a kernel builds its pipeline.
    run(false)?;
    run(true)?;
    let (mut offs, mut ons, mut diffs, mut digests) = (vec![], vec![], vec![], vec![]);
    for block in 0..rounds {
        let (a1, da1) = run(false)?;
        let (b1, db1) = run(true)?;
        let (b2, db2) = run(true)?;
        let (a2, da2) = run(false)?;
        let (a, b) = ((a1 + a2) / 2.0, (b1 + b2) / 2.0);
        println!(
            "  block {block}: off {a:.3} ms, on {b:.3} ms, on - off {:+.3}",
            b - a
        );
        offs.push(a);
        ons.push(b);
        diffs.push(b - a);
        digests.extend([da1, db1, db2, da2]);
    }
    let n = diffs.len() as f64;
    let mean_diff = diffs.iter().sum::<f64>() / n;
    let var = diffs.iter().map(|d| (d - mean_diff).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    let se = (var / n).sqrt();
    let mean_off = mean(offs.iter().copied());
    println!(
        "off {mean_off:.3} ms, on {:.3} ms per token; on - off {mean_diff:+.3} ms ({:+.1}%), paired se {se:.3} ({:.1} se)",
        mean(ons.iter().copied()),
        mean_diff / mean_off * 100.0,
        mean_diff.abs() / se.max(1e-12)
    );
    let first = digests[0];
    let same = digests.iter().filter(|&&d| d == first).count();
    println!(
        "continuations: {same} of {} runs give digest {first:016x}",
        digests.len()
    );
    Ok(())
}

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut flag_value = |name: &str| -> Option<String> {
        let at = args.iter().position(|a| a == name)?;
        let value = args.get(at + 1).cloned();
        args.drain(at..(at + 2).min(args.len()));
        value
    };
    if let Some(n) = flag_value("--floor") {
        return launch_floor(n.parse().context("--floor takes a count")?);
    }
    let steps: usize = flag_value("--steps")
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let warm: usize = flag_value("--warm")
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let ab_flag = flag_value("--ab");
    let rounds: usize = flag_value("--rounds")
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
        .encode(prompt.as_str(), true)
        .map_err(|e| anyhow!("encode: {e}"))?
        .get_ids()
        .to_vec();

    let model = NativeGemma4Model::load(&dir).context("load the checkpoint")?;
    if let Some(flag) = ab_flag {
        return ab(&model, &ids, &flag, rounds, warm, steps);
    }
    let n_layers = model.num_layers();
    println!(
        "{} prompt tokens, {n_layers} layers, decoding {steps} steps after {warm} warm-up steps",
        ids.len()
    );

    // The pipelined loop first and last, so a drifting machine shows as a gap between the two.
    let (pipe_a, tokens_a) = pipelined(&model, &ids, warm, steps)?;

    let (mut cache, mut current) = prefill(&model, &ids)?;
    let mut generated = Vec::with_capacity(warm + steps);
    let mut recorded: Vec<Step> = Vec::new();
    let (mut cpu_start, mut wall_start) = (0u64, Instant::now());
    for step in 0..warm + steps {
        if step == warm {
            mlx_rs::metal::reset_prim_histogram_dynamic().ok();
            lumen_mlx::gemma4::stage_reset();
            cpu_start = cpu_micros();
            wall_start = Instant::now();
        }
        generated.push(model.read_token_u32(&current)?);
        let (c0, d0) = mlx_rs::metal::cmd_buffer_stats().unwrap_or_default();
        let (p0, e0) = mlx_rs::metal::eval_gpu_stats().unwrap_or_default();
        let t0 = Instant::now();
        let logits = model.forward_array_last_token(&current, &mut cache)?;
        let next = model.argmax_last_token_lazy(&logits)?;
        let t1 = Instant::now();
        next.eval()?;
        let t2 = Instant::now();
        let (c1, d1) = mlx_rs::metal::cmd_buffer_stats().unwrap_or_default();
        let (p1, e1) = mlx_rs::metal::eval_gpu_stats().unwrap_or_default();
        if step >= warm {
            recorded.push(Step {
                build_ms: (t1 - t0).as_secs_f64() * 1e3,
                eval_ms: (t2 - t1).as_secs_f64() * 1e3,
                commits: c1.saturating_sub(c0),
                dispatches: d1.saturating_sub(d0),
                prims: p1.saturating_sub(p0),
                encode_ms: e1.saturating_sub(e0) as f64 / 1e6,
            });
        }
        current = next;
    }
    generated.push(model.read_token_u32(&current)?);
    let wall = wall_start.elapsed().as_secs_f64();
    let cpu = (cpu_micros() - cpu_start) as f64 / 1e6;
    let histogram = mlx_rs::metal::prim_histogram_dynamic().unwrap_or_default();
    let stages = lumen_mlx::gemma4::stage_report();

    let (pipe_b, tokens_b) = pipelined(&model, &ids, warm, steps)?;

    println!(
        "continuation: {:?}",
        tok.decode(&generated, false).unwrap_or_default()
    );
    println!(
        "digest {:016x} (serial), {:016x} / {:016x} (pipelined, first / last)",
        digest(&generated),
        digest(&tokens_a),
        digest(&tokens_b)
    );

    println!(
        "\npipelined (the server's loop): {pipe_a:.2} ms per token first, {pipe_b:.2} ms last \
         -> {:.1} tok/s",
        2e3 / (pipe_a + pipe_b)
    );
    let n = recorded.len() as f64;
    let total: Vec<f64> = recorded.iter().map(|s| s.build_ms + s.eval_ms).collect();
    println!(
        "serial, per decode step (mean of {} steps):",
        recorded.len()
    );
    println!(
        "  wall {:.2} ms (median {:.2})  =  build {:.2} + eval {:.2}",
        mean(total.iter().copied()),
        median(total.clone()),
        mean(recorded.iter().map(|s| s.build_ms)),
        mean(recorded.iter().map(|s| s.eval_ms)),
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
    println!(
        "  per layer: {:.0} primitives, {:.0} dispatches",
        mean(recorded.iter().map(|s| s.prims as f64)) / n_layers as f64,
        mean(recorded.iter().map(|s| s.dispatches as f64)) / n_layers as f64,
    );

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
    if let Some(report) = stages {
        println!(
            "\nstage by stage (LUMEN_GEMMA4_STAGE_SYNC=1: a barrier after every stage, so the wall \
             times are inflated; the primitive, dispatch and encode numbers are exact):\n{report}"
        );
    }
    Ok(())
}
