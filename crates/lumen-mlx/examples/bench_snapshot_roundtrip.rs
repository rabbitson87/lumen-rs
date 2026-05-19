//! Track A2.1 snapshot/restore roundtrip test.
//!
//! Verifies that a snapshot taken right after prefill, then restored after
//! some decode_steps, produces the IDENTICAL decode trajectory as the path
//! that does the decode without snapshot/restore. This is the correctness
//! gate for using snapshot/restore as the rollback primitive in Path (c)
//! N-gram spec decode partial-accept handling.
//!
//! Protocol:
//!   1. Prefill prompt → state at offset P, predicted next = pred_after_prompt.
//!   2. Snapshot S0.
//!   3. Decode N steps via decode_step → reference tokens A[0..N].
//!   4. Restore S0 (state back to offset P).
//!   5. Decode N steps via decode_step → tokens B[0..N].
//!   6. Compare A vs B token-by-token.
//!
//! Usage:
//!   USE_MLX=1 cargo run --release -p lumen-mlx --example bench_snapshot_roundtrip -- \
//!     --model Qwen/Qwen2.5-0.5B --steps 10
//!
//! (Use a small model for fast iteration; 35B works but takes longer.)

use std::time::Instant;

use anyhow::{Result, anyhow};
use lumen_mlx::MlxBackend;

const DEFAULT_PROMPT: &str = "The quick brown fox jumps over the lazy dog. ";

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_id = std::env::var("MODEL_ID").unwrap_or_else(|_| "Qwen/Qwen2.5-0.5B".into());
    let mut steps: usize = 10;
    let mut prompt = DEFAULT_PROMPT.to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                model_id = args[i + 1].clone();
                i += 2;
            }
            "--steps" => {
                steps = args[i + 1].parse().unwrap_or(10);
                i += 2;
            }
            "--prompt" => {
                prompt = args[i + 1].clone();
                i += 2;
            }
            _ => i += 1,
        }
    }

    println!("--- Track A2.1 snapshot/restore roundtrip ---");
    println!("model = {model_id}");
    println!("steps = {steps}");

    let t0 = Instant::now();
    let mut backend = MlxBackend::load(&model_id)?;
    println!("loaded in {:.0}ms", t0.elapsed().as_secs_f64() * 1000.0);

    let prompt_ids = backend.encode(&prompt)?;
    println!(
        "prompt: {} chars → {} tokens",
        prompt.len(),
        prompt_ids.len()
    );

    // ── Phase 1: prefill + snapshot ──
    let seq = backend.alloc_seq_id();
    let (after_prompt_pred, mut pos_a) = backend.prefill(seq, &prompt_ids)?;
    let snap = backend.snapshot_state(seq)?;
    println!("prefilled: pos={pos_a}, pred_after_prompt={after_prompt_pred}, snapshot_id={snap}");

    // ── Phase 2: decode N steps (path A) ──
    let mut tokens_a: Vec<u32> = vec![after_prompt_pred];
    let mut last = after_prompt_pred;
    for _ in 1..steps {
        let (next, p) = backend.decode_step(seq, last, pos_a)?;
        tokens_a.push(next);
        last = next;
        pos_a = p;
    }
    println!("path A tokens (N={steps}): {tokens_a:?}");

    // ── Phase 3: restore snapshot ──
    let restored_pos = backend.restore_state(seq, snap)?;
    println!(
        "restored to pos={restored_pos} (expected {})",
        prompt_ids.len()
    );

    if restored_pos != prompt_ids.len() {
        return Err(anyhow!(
            "restore returned wrong position: got {restored_pos}, expected {}",
            prompt_ids.len()
        ));
    }

    // ── Phase 4: decode N steps again (path B) — must match path A ──
    let mut tokens_b: Vec<u32> = vec![after_prompt_pred];
    let mut last = after_prompt_pred;
    let mut pos_b = restored_pos;
    for _ in 1..steps {
        let (next, p) = backend.decode_step(seq, last, pos_b)?;
        tokens_b.push(next);
        last = next;
        pos_b = p;
    }
    println!("path B tokens (N={steps}): {tokens_b:?}");

    backend.remove_seq(seq).ok();

    // ── Compare ──
    let mismatches: Vec<(usize, u32, u32)> = tokens_a
        .iter()
        .zip(tokens_b.iter())
        .enumerate()
        .filter_map(|(i, (a, b))| if a == b { None } else { Some((i, *a, *b)) })
        .collect();

    println!("\n--- result ---");
    if mismatches.is_empty() {
        println!("✅ PASS: tokens_a == tokens_b (snapshot/restore preserves decode trajectory)");
    } else {
        println!("❌ FAIL: {} mismatches", mismatches.len());
        for (i, a, b) in &mismatches {
            println!("  step {i}: A={a} B={b}");
        }
        return Err(anyhow!(
            "snapshot/restore roundtrip broke decode determinism"
        ));
    }

    Ok(())
}
