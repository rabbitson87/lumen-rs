//! Native-Rust port of `scripts/mlx_forward_state_equivalence.py`.
//!
//! Verifies the foundational assumption of speculative decoding at the native
//! Rust + MLX-rs binding level (i.e., the production `runner_native` path):
//!
//!     target.forward(input_seq_len=N, cache_in)
//!         ≡ N applications of target.forward(input_seq_len=1, cache_in_i)
//!         (in cache state evolution and per-position next-token argmax)
//!
//! If this equivalence is violated for the target on this backend, then ANY
//! greedy speculative decoding scheme that "verifies a block of N tokens in
//! one forward and rolls back the rejected suffix" will produce output that
//! differs from pure step-by-step greedy decoding — regardless of how clever
//! the rollback is. Pre-flight gate for spec-decode infrastructure work.
//!
//! Method (no DFlash, no draft, no GDN replay — just the target model):
//!   1. Encode PROMPT, run a reference greedy continuation to produce a
//!      deterministic test sequence `block_input` of length `N_max + 2` so
//!      every N value uses the same input prefix.
//!   2. For each N in the sweep:
//!        a. Allocate two seq ids A and B, prefill both with PROMPT (identical
//!           starting cache state).
//!        b. Path A: single `forward_probe(seq_A, block_input[..N])` — S=N.
//!        c. Path B: for i in 0..N: `forward_probe(seq_B, &[block_input[i]])`
//!           — N × S=1.
//!        d. Compare per-position argmax (A vs B) and per-position max-abs
//!           logit (proxy for logit-magnitude divergence; full element-wise
//!           diff would require exposing logits, but argmax + max-abs is the
//!           strongest signal the existing API surfaces).
//!
//! BAR.1 (max-abs proxy): per-position |max_abs_A[i] - max_abs_B[i]| < 0.5
//!     for all i in [0, N) AND all swept N. (Threshold rationale: if the two
//!     paths produce element-wise different logits, the per-position max abs
//!     value typically also drifts; ≥ 0.5 magnitude difference is far above
//!     legitimate fp accumulation noise for the same arithmetic.)
//! BAR.2 (argmax stability): argmax_A[i] == argmax_B[i] for all i and N.
//!
//! Both BARs must hold to declare spec-decode greedy bit-identical possible.
//!
//! Usage:
//!   LUMEN_MLX_BACKEND=native \
//!     cargo run --release -p lumen-mlx \
//!     --example bench_native_forward_equivalence -- \
//!     [--model mlx-community/Qwen3.6-35B-A3B-mxfp4] \
//!     [--n-list 2,3,4,5,6,7,8,12,16,24,32] \
//!     [--prompt "..."]
//!
//! The `--n-list` should NOT include any N > tokens-after-prompt-greedy. The
//! reference trajectory generates `max(N_list) + 2` tokens.

use std::time::Instant;

use anyhow::{Result, anyhow};
use lumen_mlx::MlxBackend;

const DEFAULT_PROMPT: &str = "<|im_start|>user\nWrite a Python function for binary search.<|im_end|>\n<|im_start|>assistant\n";

fn parse_n_list(raw: &str) -> Vec<usize> {
    let mut v: Vec<usize> = raw
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

struct PerN {
    n: usize,
    argmax_match_all: bool,
    max_abs_diff_max: f32,
    positions: Vec<PerPos>,
}

struct PerPos {
    i: usize,
    argmax_a: u32,
    argmax_b: u32,
    max_abs_a: f32,
    max_abs_b: f32,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_id =
        std::env::var("MODEL_ID").unwrap_or_else(|_| "mlx-community/Qwen3.6-35B-A3B-mxfp4".into());
    let mut n_list_raw = String::from("2,3,4,5,6,7,8,12,16,24,32");
    let mut prompt = DEFAULT_PROMPT.to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                model_id = args[i + 1].clone();
                i += 2;
            }
            "--n-list" => {
                n_list_raw = args[i + 1].clone();
                i += 2;
            }
            "--prompt" => {
                prompt = args[i + 1].clone();
                i += 2;
            }
            _ => i += 1,
        }
    }

    let n_list = parse_n_list(&n_list_raw);
    if n_list.is_empty() {
        return Err(anyhow!("no valid N values in --n-list={n_list_raw:?}"));
    }
    let n_max = *n_list.last().expect("non-empty after dedup");

    println!("--- native forward state-equivalence verifier ---");
    println!(
        "backend     = {}",
        std::env::var("LUMEN_MLX_BACKEND").unwrap_or_else(|_| "(default = pyo3)".into())
    );
    println!("model       = {model_id}");
    println!("n_list      = {n_list:?}");
    println!("prompt_chars= {}", prompt.len());

    let t0 = Instant::now();
    let mut backend = MlxBackend::load(&model_id)?;
    println!("loaded in {:.0}ms", t0.elapsed().as_secs_f64() * 1000.0);
    let backend = backend
        .as_qwen35_mut()
        .ok_or_else(|| anyhow!("bench requires Qwen35-family backend"))?;

    let prompt_ids = backend.encode(&prompt)?;
    println!(
        "prompt: {} chars → {} tokens",
        prompt.len(),
        prompt_ids.len()
    );

    // ── Reference: generate `n_max + 2` greedy tokens via S=1 path ──────────
    // This produces the deterministic block_input we compare on. The +2 gives
    // headroom past the largest N so we never index out of bounds.
    let n_ref = n_max + 2;
    println!("\n[reference] generating {n_ref} greedy tokens via decode_step (S=1)…");
    let ref_seq = backend.alloc_seq_id();
    let (mut last, mut pos) = backend.prefill(ref_seq, &prompt_ids)?;
    let mut ref_tokens: Vec<u32> = vec![last];
    for _step in 1..n_ref {
        let (next, new_pos) = backend.decode_step(ref_seq, last, pos)?;
        ref_tokens.push(next);
        last = next;
        pos = new_pos;
    }
    backend.remove_seq(ref_seq).ok();
    println!(
        "reference: first 8 = {:?} (total {})",
        &ref_tokens[..8.min(ref_tokens.len())],
        ref_tokens.len()
    );

    // The block_input for the equivalence test: the model's own greedy
    // continuation. Each block_input[i]'s "next predicted token" (under
    // sequential decode) is ref_tokens[i+1].
    //
    // Path A: prefill(prompt) → forward_probe(block_input[..N]) → row i argmax
    // Path B: prefill(prompt) → forward_probe([block_input[0]]) → argmax
    //                          → forward_probe([block_input[1]]) → argmax
    //                          → …
    //
    // Both paths START at "cache after prefill(prompt)". block_input[0] is the
    // post-prefill argmax (= ref_tokens[0]).
    let block_input: Vec<u32> = ref_tokens.clone();

    // ── Per-N comparison ─────────────────────────────────────────────────────
    let mut all_results: Vec<PerN> = Vec::with_capacity(n_list.len());
    for &n in &n_list {
        // ── Path A: single S=N forward ──
        let seq_a = backend.alloc_seq_id();
        let (_first_tok_a, _) = backend.prefill(seq_a, &prompt_ids)?;
        let probe_a = backend.forward_probe(seq_a, &block_input[..n])?;
        backend.remove_seq(seq_a).ok();

        // ── Path B: N × S=1 forwards ──
        let seq_b = backend.alloc_seq_id();
        let (_first_tok_b, _) = backend.prefill(seq_b, &prompt_ids)?;
        let mut argmaxes_b: Vec<u32> = Vec::with_capacity(n);
        let mut max_abs_b: Vec<f32> = Vec::with_capacity(n);
        for i in 0..n {
            let probe = backend.forward_probe(seq_b, &block_input[i..i + 1])?;
            // forward_probe with K=1 returns row vec of length 1
            argmaxes_b.push(probe.row_argmaxes[0]);
            max_abs_b.push(probe.row_max_abs[0]);
        }
        backend.remove_seq(seq_b).ok();

        // ── Compare ──
        let mut positions = Vec::with_capacity(n);
        let mut all_argmax_ok = true;
        let mut max_diff: f32 = 0.0;
        for i in 0..n {
            let am_a = probe_a.row_argmaxes[i];
            let am_b = argmaxes_b[i];
            let ma_a = probe_a.row_max_abs[i];
            let ma_b = max_abs_b[i];
            let diff = (ma_a - ma_b).abs();
            if am_a != am_b {
                all_argmax_ok = false;
            }
            if diff > max_diff {
                max_diff = diff;
            }
            positions.push(PerPos {
                i,
                argmax_a: am_a,
                argmax_b: am_b,
                max_abs_a: ma_a,
                max_abs_b: ma_b,
            });
        }

        println!(
            "\nN={n:2}: argmax_match_all={}  max|max_abs_A − max_abs_B|={:.4}",
            all_argmax_ok, max_diff
        );
        // Per-position detail: only print mismatches (or first 4 positions for
        // small N to confirm we're actually computing).
        let print_all = n <= 4;
        for p in &positions {
            let mismatch = p.argmax_a != p.argmax_b;
            let big_mag_diff = (p.max_abs_a - p.max_abs_b).abs() > 0.01;
            if print_all || mismatch || big_mag_diff {
                let tag = if mismatch { " ARGMAX MISMATCH" } else { "" };
                println!(
                    "   i={:2} argmax A={} B={}  max_abs A={:.3} B={:.3} (Δ={:.4}){}",
                    p.i,
                    p.argmax_a,
                    p.argmax_b,
                    p.max_abs_a,
                    p.max_abs_b,
                    (p.max_abs_a - p.max_abs_b).abs(),
                    tag,
                );
            }
        }
        all_results.push(PerN {
            n,
            argmax_match_all: all_argmax_ok,
            max_abs_diff_max: max_diff,
            positions,
        });
    }

    // ── Final verdict ────────────────────────────────────────────────────────
    println!("\n=== verdict ===");
    println!("  BAR.1: max|max_abs_A − max_abs_B| < 0.5 across all N (proxy for logit divergence)");
    println!("  BAR.2: all argmax match across all N and positions");
    println!("  results:");
    let bar1_threshold: f32 = 0.5;
    let mut overall_safe = true;
    for r in &all_results {
        let bar1 = r.max_abs_diff_max < bar1_threshold;
        let bar2 = r.argmax_match_all;
        let ok = bar1 && bar2;
        if !ok {
            overall_safe = false;
        }
        println!(
            "    N={:2}: max|Δmax_abs|={:.4} {}  argmax_match={} {}",
            r.n,
            r.max_abs_diff_max,
            if bar1 { "✓" } else { "✗" },
            r.argmax_match_all,
            if bar2 { "✓" } else { "✗" },
        );
    }
    println!(
        "  → spec-decode greedy bit-identical possible (native backend): {}",
        overall_safe
    );
    if !overall_safe {
        println!("\n  WARNING: even where argmax matches in tested positions, max-abs");
        println!(
            "  divergence ≥ {:.1} indicates the verify path's cache state evolution",
            bar1_threshold
        );
        println!("  differs from sequential decode — subsequent decode_steps WILL");
        println!("  eventually flip argmax. See `dflash_root_cause_diagnostic.md`.");
    }

    // Exit code: 0 if safe, 1 if any BAR fails (mirrors python verifier).
    if overall_safe {
        Ok(())
    } else {
        std::process::exit(1);
    }
}
