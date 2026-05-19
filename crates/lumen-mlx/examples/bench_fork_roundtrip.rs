//! Track A1.1 — fork-from-snapshot roundtrip & independence test.
//!
//! Validates that `snapshot_state_deep` + `fork_from_snapshot` behave correctly
//! in three orthogonal axes:
//!
//!   1. **Bit-identical fork**: a forked seq's decode trajectory matches what
//!      a cold seq would produce from the same prefill (no drift).
//!   2. **Source isolation**: continuing decode on the source seq does NOT
//!      affect snapshots that were already taken from it.
//!   3. **Sibling isolation**: two seqs forked from the same master do NOT
//!      see each other's mutations (deep-copy worked).
//!   4. **Master reusability**: the master snapshot can be forked twice into
//!      two different destination seqs; both produce identical trajectories.
//!
//! Protocol:
//!   1. Prefill prefix on src seq → state at offset P, predicted next = T.
//!   2. snapshot_state_deep(src) → master M (independent of src).
//!   3. Fork master → dst1, decode N steps → tokens_dst1.
//!   4. Decode src N steps → tokens_src.
//!   5. Verify tokens_dst1 == tokens_src (axis 1 + 2: src untouched by snapshot,
//!      dst1 reproduces the same forward trajectory).
//!   6. Fork master → dst2, decode N steps → tokens_dst2.
//!      Verify tokens_dst2 == tokens_dst1 (axis 4: master reusable + axis 3:
//!      dst1 mutations didn't leak into dst2 because deep-copy at install).
//!
//! Default model is Qwen2.5-0.5B (KVCache only — fast iteration). For the
//! production hybrid model, set MODEL_ID=mlx-community/Qwen3.6-35B-A3B-mxfp4.
//!
//! Usage:
//!   USE_MLX=1 cargo run --release -p lumen-mlx --example bench_fork_roundtrip
//!   USE_MLX=1 MODEL_ID=mlx-community/Qwen3.6-35B-A3B-mxfp4 \
//!     cargo run --release -p lumen-mlx --example bench_fork_roundtrip

use std::time::Instant;

use anyhow::{Result, anyhow};
use lumen_mlx::MlxBackend;

const DEFAULT_PROMPT: &str = "The quick brown fox jumps over the lazy dog. \
                              The system prompt is detailed and lists rules.";

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut steps: usize = 10;
    let mut prompt = DEFAULT_PROMPT.to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
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

    let model_id = std::env::var("MODEL_ID").unwrap_or_else(|_| "Qwen/Qwen2.5-0.5B".into());

    println!("--- Track A1.1 fork-from-snapshot roundtrip ---");
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

    // ── Phase 1: prefill on src ──
    let src = backend.alloc_seq_id();
    let (after_prompt_pred, _pos_src0) = backend.prefill(src, &prompt_ids)?;
    println!("src prefilled: pred_after_prompt={after_prompt_pred}");

    // ── Phase 2: deep snapshot ──
    let t = Instant::now();
    let (master, master_pos) = backend.snapshot_state_deep(src)?;
    let snap_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "snapshot_state_deep: id={master} pos={master_pos} in {snap_ms:.1}ms (expected pos={})",
        prompt_ids.len()
    );

    if master_pos != prompt_ids.len() {
        return Err(anyhow!(
            "snapshot returned wrong position: got {master_pos}, expected {}",
            prompt_ids.len()
        ));
    }

    // ── Phase 3: fork master → dst1 ──
    let dst1 = backend.alloc_seq_id();
    let t = Instant::now();
    let dst1_pos = backend.fork_from_snapshot(master, dst1)?;
    let fork1_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("fork_from_snapshot(master → dst1={dst1}): pos={dst1_pos} in {fork1_ms:.1}ms");

    if dst1_pos != prompt_ids.len() {
        return Err(anyhow!(
            "fork returned wrong position: got {dst1_pos}, expected {}",
            prompt_ids.len()
        ));
    }

    // ── Phase 4: decode N steps on dst1 (path A) ──
    let mut tokens_dst1: Vec<u32> = vec![after_prompt_pred];
    let mut last = after_prompt_pred;
    let mut pos = dst1_pos;
    for _ in 1..steps {
        let (next, p) = backend.decode_step(dst1, last, pos)?;
        tokens_dst1.push(next);
        last = next;
        pos = p;
    }
    println!("path A (dst1)  tokens: {tokens_dst1:?}");

    // ── Phase 5: decode N steps on src (path B) ──
    // Source seq's cache should be untouched by snapshot+fork — its position
    // is still at master_pos; continue from after_prompt_pred.
    let mut tokens_src: Vec<u32> = vec![after_prompt_pred];
    let mut last = after_prompt_pred;
    let mut pos = master_pos;
    for _ in 1..steps {
        let (next, p) = backend.decode_step(src, last, pos)?;
        tokens_src.push(next);
        last = next;
        pos = p;
    }
    println!("path B (src)   tokens: {tokens_src:?}");

    // ── Phase 6: fork master → dst2, decode N steps ──
    // Master snapshot must still be valid (not consumed by first fork).
    let dst2 = backend.alloc_seq_id();
    let t = Instant::now();
    let dst2_pos = backend.fork_from_snapshot(master, dst2)?;
    let fork2_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("fork_from_snapshot(master → dst2={dst2}): pos={dst2_pos} in {fork2_ms:.1}ms");

    let mut tokens_dst2: Vec<u32> = vec![after_prompt_pred];
    let mut last = after_prompt_pred;
    let mut pos = dst2_pos;
    for _ in 1..steps {
        let (next, p) = backend.decode_step(dst2, last, pos)?;
        tokens_dst2.push(next);
        last = next;
        pos = p;
    }
    println!("path C (dst2)  tokens: {tokens_dst2:?}");

    // ── Cleanup ──
    backend.release_snapshot(master).ok();
    backend.remove_seq(src).ok();
    backend.remove_seq(dst1).ok();
    backend.remove_seq(dst2).ok();

    // ── Compare ──
    let cmp = |a: &[u32], b: &[u32]| -> Vec<(usize, u32, u32)> {
        a.iter()
            .zip(b.iter())
            .enumerate()
            .filter_map(|(i, (x, y))| if x == y { None } else { Some((i, *x, *y)) })
            .collect()
    };

    let mismatch_a_b = cmp(&tokens_dst1, &tokens_src);
    let mismatch_a_c = cmp(&tokens_dst1, &tokens_dst2);

    println!("\n--- result ---");
    let mut ok = true;
    if mismatch_a_b.is_empty() {
        println!(
            "✅ PASS axis 1+2 (dst1 == src) — fork reproduces source trajectory + src untouched"
        );
    } else {
        ok = false;
        println!(
            "❌ FAIL axis 1+2 (dst1 != src): {} mismatches",
            mismatch_a_b.len()
        );
        for (i, a, b) in mismatch_a_b.iter().take(5) {
            println!("  step {i}: dst1={a} src={b}");
        }
    }
    if mismatch_a_c.is_empty() {
        println!("✅ PASS axis 3+4 (dst1 == dst2) — master reusable + sibling isolation");
    } else {
        ok = false;
        println!(
            "❌ FAIL axis 3+4 (dst1 != dst2): {} mismatches",
            mismatch_a_c.len()
        );
        for (i, a, b) in mismatch_a_c.iter().take(5) {
            println!("  step {i}: dst1={a} dst2={b}");
        }
    }

    if !ok {
        return Err(anyhow!("fork roundtrip failed"));
    }

    println!("\nTimings: snap={snap_ms:.1}ms fork1={fork1_ms:.1}ms fork2={fork2_ms:.1}ms");
    Ok(())
}
