//! Track A2.0 drift baseline. For each checkpoint along a 30-step greedy
//! decode trajectory, compares the **batched S=3 forward** (row 0 argmax) to
//! the **single-step S=1 decode** (Lookahead L.3 lesson — MXFP4 v3 kernel can
//! produce different logits at different sequence dims, flipping argmax).
//!
//! Each checkpoint k:
//!   1. Reference: prefill prompt + 30 decode_step → ref_tokens[0..30].
//!   2. Probe: fresh seq, prefill(prompt + ref_tokens[0..k]) — state at offset
//!      P+k, "next predicted" token == ref_tokens[k] (sanity check).
//!   3. forward_probe([ref_tokens[k], ref_tokens[k+1], ref_tokens[k+2]]) →
//!      row 0 argmax should equal ref_tokens[k+1] (single-step match).
//!   4. Also record row 1 / row 2 matches and per-row max-abs-logit.
//!
//! Gate logic per the A2 plan:
//!   - All checkpoints within first 30 tokens match → safe for K=3
//!   - Diverges at ≥ 20 → safe with K=2 cap
//!   - Diverges < 10 → abandon (a) entirely, fall to (c) N-gram K=1
//!
//! Usage:
//!   USE_MLX=1 cargo run --release -p lumen-mlx --example bench_drift_S_vs_1 -- \
//!     --model mlx-community/Qwen3.6-35B-A3B-mxfp4 --max-decode 30 \
//!     --probes 0,5,10,15,20,25,27

use std::time::Instant;

use anyhow::{Result, anyhow};
use lumen_mlx::MlxBackend;

const DEFAULT_PROMPT: &str = "The history of artificial intelligence is a long and intricate journey \
that spans decades of academic curiosity, engineering breakthroughs, and societal ambition. \
Early pioneers asked whether machines could exhibit the kind of reasoning that humans take for \
granted, and their answers seeded entire research programs that we still inherit today. From \
symbolic systems to perceptrons, from expert systems to backpropagation, each generation \
rediscovered both the promise and the brittleness of computational intelligence. Modern systems \
build on these foundations while extending them in";

fn parse_probes(raw: &str, max: usize) -> Vec<usize> {
    raw.split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .filter(|&k| k + 3 <= max) // need 3 tokens past k
        .collect()
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_id =
        std::env::var("MODEL_ID").unwrap_or_else(|_| "mlx-community/Qwen3.6-35B-A3B-mxfp4".into());
    let mut max_decode: usize = 30;
    let mut probe_str = String::from("0,5,10,15,20,25,27");
    let mut prompt = DEFAULT_PROMPT.to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                model_id = args[i + 1].clone();
                i += 2;
            }
            "--max-decode" => {
                max_decode = args[i + 1].parse().unwrap_or(30);
                i += 2;
            }
            "--probes" => {
                probe_str = args[i + 1].clone();
                i += 2;
            }
            "--prompt" => {
                prompt = args[i + 1].clone();
                i += 2;
            }
            _ => i += 1,
        }
    }

    println!("--- Track A2.0 drift baseline (S=1 vs S=3 batched forward) ---");
    println!("model       = {model_id}");
    println!("max_decode  = {max_decode}");
    println!("probe steps = {probe_str}");

    let probes = parse_probes(&probe_str, max_decode);
    if probes.is_empty() {
        return Err(anyhow!("no valid probe steps after parsing"));
    }

    // ── Load model ──
    let t0 = Instant::now();
    let mut backend = MlxBackend::load(&model_id)?;
    println!("loaded in {:.0}ms", t0.elapsed().as_secs_f64() * 1000.0);

    let prompt_ids = backend.encode(&prompt)?;
    println!("prompt: {} chars → {} tokens", prompt.len(), prompt_ids.len());

    // ── Reference: 30 decode_steps via S=1 path ──
    println!("\n[reference] greedy {max_decode}-step decode via decode_step (S=1)...");
    let t0 = Instant::now();
    let ref_seq = backend.alloc_seq_id();
    let (mut last, mut pos) = backend.prefill(ref_seq, &prompt_ids)?;
    let mut ref_tokens: Vec<u32> = vec![last];
    for _step in 1..max_decode {
        let (next, new_pos) = backend.decode_step(ref_seq, last, pos)?;
        ref_tokens.push(next);
        last = next;
        pos = new_pos;
    }
    backend.remove_seq(ref_seq).ok();
    let ref_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "reference done: {} tokens in {ref_ms:.0}ms ({:.1} tok/s); first 8 = {:?}",
        ref_tokens.len(),
        ref_tokens.len() as f64 / (ref_ms / 1000.0),
        &ref_tokens[..8.min(ref_tokens.len())]
    );

    // ── Probes ──
    println!(
        "\n[probes] each entry: prefill(prompt + ref[0..k]) → forward_probe([ref[k], ref[k+1], ref[k+2]])"
    );
    println!(
        "{:>4}  {:>10}  {:>10}  {:>10}  {:>9}  {:>9}  {:>9}",
        "k", "row0_ok", "row1_ok", "row2_ok", "max_abs0", "max_abs1", "max_abs2",
    );

    let mut first_divergence: Option<usize> = None;
    let mut all_pass = true;

    for &k in &probes {
        let mut prefill_input = prompt_ids.clone();
        // prefill needs prompt + ref_tokens[0..k]
        prefill_input.extend_from_slice(&ref_tokens[..k]);

        let probe_seq = backend.alloc_seq_id();
        let (sanity, _pos2) = backend.prefill(probe_seq, &prefill_input)?;
        // Sanity check: prefill of (prompt + ref[0..k]) returns argmax that should equal ref_tokens[k]
        if sanity != ref_tokens[k] {
            eprintln!(
                "[warn] k={k} prefill sanity mismatch: prefill returned {sanity}, expected ref[{k}]={}",
                ref_tokens[k]
            );
        }

        let probe_tokens = [ref_tokens[k], ref_tokens[k + 1], ref_tokens[k + 2]];
        let probe = backend.forward_probe(probe_seq, &probe_tokens)?;
        backend.remove_seq(probe_seq).ok();

        let row0_ok = probe.row_argmaxes[0] == ref_tokens[k + 1];
        let row1_ok = if k + 2 < ref_tokens.len() {
            probe.row_argmaxes[1] == ref_tokens[k + 2]
        } else {
            true
        };
        let row2_ok = if k + 3 < ref_tokens.len() {
            probe.row_argmaxes[2] == ref_tokens[k + 3]
        } else {
            true
        };

        if !row0_ok && first_divergence.is_none() {
            first_divergence = Some(k);
        }
        if !row0_ok || !row1_ok || !row2_ok {
            all_pass = false;
        }

        println!(
            "{k:>4}  {:>10}  {:>10}  {:>10}  {:>9.2}  {:>9.2}  {:>9.2}",
            if row0_ok { "PASS" } else { "FAIL" },
            if row1_ok { "PASS" } else { "FAIL" },
            if row2_ok { "PASS" } else { "FAIL" },
            probe.row_max_abs[0],
            probe.row_max_abs[1],
            probe.row_max_abs[2],
        );

        if !row0_ok {
            println!(
                "      row0: got {} expected ref[{}]={}",
                probe.row_argmaxes[0],
                k + 1,
                ref_tokens[k + 1]
            );
        }
        if !row1_ok && k + 2 < ref_tokens.len() {
            println!(
                "      row1: got {} expected ref[{}]={}",
                probe.row_argmaxes[1],
                k + 2,
                ref_tokens[k + 2]
            );
        }
        if !row2_ok && k + 3 < ref_tokens.len() {
            println!(
                "      row2: got {} expected ref[{}]={}",
                probe.row_argmaxes[2],
                k + 3,
                ref_tokens[k + 3]
            );
        }
    }

    println!("\n--- summary ---");
    println!("probes_run     = {}", probes.len());
    println!("all_pass       = {all_pass}");
    if let Some(k) = first_divergence {
        println!("first_diverge  = k={k}");
    } else {
        println!("first_diverge  = none within probe range");
    }

    println!("\n--- gate ---");
    match first_divergence {
        None => println!("✅ S=3 batched verify is safe across all probed checkpoints. Path (a) K=3 viable."),
        Some(k) if k >= 20 => println!("⚠️  Diverges at k={k} ≥ 20 — proceed with K=2 cap only."),
        Some(k) if k >= 10 => println!("⚠️  Diverges at k={k} (10 ≤ k < 20) — K=2 risky, recommend abort verify on row-0 mismatch."),
        Some(k) => println!("❌ Diverges at k={k} < 10 — abandon path (a)/(c) batched verify. Fall to (c) N-gram K=1."),
    }

    Ok(())
}
