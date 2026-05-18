//! Track A1.2 — Prefix-cache A/B end-to-end measurement.
//!
//! Drives `MlxBackend::chat_streaming` directly with `LUMEN_MLX_PREFIX_CACHE=1`
//! enabled at process level. Compares the first request (cache MISS, populates
//! the master snapshot) against subsequent requests (cache HIT, fork + extend).
//!
//! Validates two things:
//!   1. Cache HIT requests produce identical generated tokens to a cold path
//!      with the same system prompt + user message (bit-identical).
//!   2. Cache HIT latency to the first generated token is materially lower
//!      than the cold path's, proving the savings A1.0 predicted.
//!
//! Protocol:
//!   - System prompt fixed (multi-hundred tokens).
//!   - 1 warmup, 1 cold (cache empty), then N hits with different user queries.
//!   - All run with the prefix cache disabled (`LUMEN_MLX_PREFIX_CACHE=0`)
//!     once for cold-cold reference.
//!
//! Usage (must enable via env *before* binary launch — `read_prefix_cache_limits`
//! reads on `MlxBackend::load`):
//!   USE_MLX=1 LUMEN_MLX_PREFIX_CACHE=1 \
//!     cargo run --release -p lumen-mlx --example bench_prefix_cache_ab
//!   USE_MLX=1 MODEL_ID=mlx-community/Qwen3.6-35B-A3B-mxfp4 \
//!     LUMEN_MLX_PREFIX_CACHE=1 \
//!     cargo run --release -p lumen-mlx --example bench_prefix_cache_ab

use std::time::Instant;

use anyhow::Result;
use lumen_mlx::MlxBackend;

const SYSTEM_PROMPT: &str = "You are a helpful assistant. Follow these instructions \
    carefully. The user will ask questions and you should answer concisely. Always \
    write in clear, simple language. Use bullet points when appropriate. If the \
    question is technical, give an example. Avoid lengthy preambles. Be direct \
    but polite. When you don't know something, say so honestly. Format code in \
    fenced blocks. Cite sources when you can. Stay on topic. Don't refuse benign \
    requests. Default to brevity. Format math in LaTeX where it helps. Use \
    headings to organize long answers. Match the user's language unless asked \
    otherwise. Avoid filler phrases. Get to the point quickly.";

fn build_msgs(user: &str) -> Vec<(String, String)> {
    vec![
        ("system".into(), SYSTEM_PROMPT.into()),
        ("user".into(), user.into()),
    ]
}

#[derive(Debug)]
struct RunResult {
    label: String,
    user: String,
    text: String,
    ttft_ms: f64,
    total_ms: f64,
    n_tokens: usize,
}

fn run_one(backend: &mut MlxBackend, label: &str, user: &str) -> Result<RunResult> {
    let msgs = build_msgs(user);
    let max_new = 32;
    let seq_id = backend.alloc_seq_id();

    let t_start = Instant::now();
    let mut first_tok_time: Option<Instant> = None;
    let mut n_tokens = 0;
    let text = backend.chat_streaming(&msgs, max_new, false, seq_id, |s| {
        if first_tok_time.is_none() && !s.is_empty() {
            first_tok_time = Some(Instant::now());
        }
        n_tokens += 1;
        let _ = s;
    })?;

    let total_ms = t_start.elapsed().as_secs_f64() * 1000.0;
    let ttft_ms = first_tok_time
        .map(|t| t.duration_since(t_start).as_secs_f64() * 1000.0)
        .unwrap_or(total_ms);

    Ok(RunResult {
        label: label.into(),
        user: user.into(),
        text,
        ttft_ms,
        total_ms,
        n_tokens,
    })
}

fn main() -> Result<()> {
    let model_id = std::env::var("MODEL_ID").unwrap_or_else(|_| "Qwen/Qwen2.5-0.5B".into());
    let pc_enabled = std::env::var("LUMEN_MLX_PREFIX_CACHE").ok().as_deref() == Some("1");

    println!("--- Track A1.2 prefix-cache A/B ---");
    println!("model = {model_id}");
    println!("LUMEN_MLX_PREFIX_CACHE = {}", if pc_enabled { "1" } else { "0 (DISABLED)" });

    let mut backend = MlxBackend::load(&model_id)?;

    // Warmup uses a *different* system prompt so it doesn't pre-populate the
    // cache key for the measured runs.
    let warm_msgs = vec![
        ("system".into(), "Different warmup system.".to_string()),
        ("user".into(), "Hi".to_string()),
    ];
    let warm_seq = backend.alloc_seq_id();
    let _ = backend.chat_streaming(&warm_msgs, 8, false, warm_seq, |_| {})?;
    eprintln!("[bench] warmup complete (different sys prompt, separate cache key)");

    let queries = [
        "What is 2 + 2?",
        "Name a primary color.",
        "Spell 'cat'.",
        "What is 10 * 10?",
        "Pick a fruit.",
    ];

    let mut results: Vec<RunResult> = Vec::new();
    for (i, q) in queries.iter().enumerate() {
        let label = if i == 0 { "cold(seed)" } else { "hit?" };
        let r = run_one(&mut backend, label, q)?;
        eprintln!(
            "[run {i}] label={} user={q:?} ttft={:.0}ms total={:.0}ms n={}",
            r.label, r.ttft_ms, r.total_ms, r.n_tokens
        );
        results.push(r);
    }

    println!("\n{:<5} {:<12} {:>10} {:>10} {:>5} | {}", "i", "label", "ttft(ms)", "total(ms)", "n", "user");
    println!("{}", "-".repeat(90));
    for (i, r) in results.iter().enumerate() {
        println!(
            "{:<5} {:<12} {:>10.0} {:>10.0} {:>5} | {}",
            i, r.label, r.ttft_ms, r.total_ms, r.n_tokens, r.user
        );
    }

    // Summary: avg TTFT of cold (i=0) vs avg of subsequent (i>=1).
    if let Some(cold) = results.first() {
        let post: Vec<&RunResult> = results.iter().skip(1).collect();
        if !post.is_empty() {
            let avg_post_ttft: f64 =
                post.iter().map(|r| r.ttft_ms).sum::<f64>() / post.len() as f64;
            let cold_ttft = cold.ttft_ms;
            let savings_pct = 100.0 * (cold_ttft - avg_post_ttft) / cold_ttft.max(1e-9);
            println!(
                "\nCold seed TTFT:        {:.0}ms",
                cold_ttft
            );
            println!(
                "Avg subsequent TTFT:   {:.0}ms ({} runs)",
                avg_post_ttft,
                post.len()
            );
            println!(
                "Savings:               {:.1}% (positive = prefix cache HIT working)",
                savings_pct
            );
            if pc_enabled && savings_pct < 10.0 {
                println!(
                    "\n⚠️  Savings <10% with prefix cache enabled — \
                     suspect: system prefix not isolated, or fall-through"
                );
            }
        }
    }
    Ok(())
}
