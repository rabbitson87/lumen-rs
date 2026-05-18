//! Lever A Step L.3 — Lookahead-decode A/B benchmark.
//!
//! Single-process protocol (one model load, mirrors `bench_lever_h_ab.rs`):
//!   - 60s cool-down before measurement
//!   - Run 1 (forward):  baseline → lookahead variants in declared order
//!   - 60s cool-down
//!   - Run 2 (reverse):  lookahead variants in reverse → baseline
//!   - Pool prompt-level avg_token_ms × 2 sub-runs for Welch's t between
//!     baseline and each lookahead variant.
//!
//! "Token-level" ms is computed per prompt as `sum(decode_step_ms) /
//! n_decode_tokens`, where n_decode_tokens excludes the prefill-emitted first
//! token. This gives a unit comparable across baseline (1 tok/iter) and
//! lookahead (1+jacobi_accept tok/iter).
//!
//! Bit-identical: under greedy (temp=0), lookahead with G=0 must reproduce the
//! exact baseline sequence — that's the lossless invariant. Cross-variant
//! token mismatch ⇒ correctness regression, fail.
//!
//! Defaults (full mode): 6 prompts × 20 tokens per sub-run; 60s cool-down.
//! Quick mode (`LD_BENCH_QUICK=1`): 2 prompts × 8 tokens, 5s cool-down.
//!
//! Variant set selection:
//!   - Default: baseline + W=4 G=0 (the L.2 bit-identical pair)
//!   - Sweep:   `LD_BENCH_SWEEP=1` adds W=4 G=2/3, W=5 G=0/2/3
//!   - Custom:  `LD_BENCH_WS=4,5` × `LD_BENCH_GS=0,2,3` (CSV)
//!
//! Usage:
//! ```sh
//! LUMEN_QWEN35_SHARDS="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-mxfp4/snapshots/<sha>" \
//! cargo run --release -p lumen-model --example bench_lookahead_decode_ab \
//!   --features turboquant-gpu
//! ```
//!
//! Environment:
//!   - `LD_BENCH_PROMPTS` (default 6): prompts per sub-run
//!   - `LD_BENCH_TOKENS`  (default 20): tokens per prompt
//!   - `LD_BENCH_COOLDOWN_S` (default 60): cool-down seconds
//!   - `LD_BENCH_QUICK=1`: 2 prompts × 8 tokens × 5s cool-down (sanity only)
//!   - `LD_BENCH_SWEEP=1`: enable full W={4,5} × G={0,2,3} sweep
//!   - `LD_BENCH_WS=4,5` / `LD_BENCH_GS=0,2,3`: custom W/G CSVs (overrides SWEEP)
//!   - `LD_BENCH_NGRAM` (default 3): pool n-gram length N
//!   - `LUMEN_QWEN35_SHARDS`: required, model snapshot dir
//!   - `MODEL_ID`: optional HF id override

#![cfg(feature = "turboquant-gpu")]

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use lumen_metal::mxfp4_gpu::MxFp4Context;
use lumen_model::qwen3_5_moe::backend::{LookaheadStats, Qwen35MoeBackend};

/// Reused from `bench_lever_h_ab` — varied prompts to exercise different
/// gate_up shapes and avoid repeated-context warm caches.
const PROMPTS: &[&str] = &[
    "<|im_start|>user\nHello, who are you?<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nList 3 colors.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nName a fruit.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nDescribe water.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nDefine gravity.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nWhat is light?<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nName a metal.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nWhat is air?<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nDefine love.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nName an animal.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nWhat is fire?<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nDefine peace.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nName a planet.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nWhat is wind?<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nDefine joy.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nName a tree.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nWhat is rain?<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nDefine hope.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nName a bird.<|im_end|>\n<|im_start|>assistant\n<think>\n",
];

/// Identifier for one variant in the sweep. `None` window/guesses ⇒ baseline
/// (lookahead OFF). Otherwise lookahead ON with given W/G; N is global.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Variant {
    label_w: usize, // 0 ⇒ baseline
    label_g: usize,
}

impl Variant {
    fn baseline() -> Self {
        Self { label_w: 0, label_g: 0 }
    }
    fn lookahead(w: usize, g: usize) -> Self {
        Self { label_w: w, label_g: g }
    }
    fn is_baseline(&self) -> bool {
        self.label_w == 0
    }
    fn label(&self) -> String {
        if self.is_baseline() {
            "BASE".to_string()
        } else {
            format!("W{}G{}", self.label_w, self.label_g)
        }
    }
}

/// Set env to activate the chosen variant, clearing the others. Lookahead +
/// ngram-spec are mutually exclusive in `backend.rs`; we only ever set
/// lookahead here (ngram-spec stays untouched/off).
fn apply_variant(v: Variant, ngram_n: usize) {
    unsafe {
        if v.is_baseline() {
            std::env::remove_var("LUMEN_LOOKAHEAD_DECODE");
            std::env::remove_var("LUMEN_LOOKAHEAD_W");
            std::env::remove_var("LUMEN_LOOKAHEAD_G");
            std::env::remove_var("LUMEN_LOOKAHEAD_N");
        } else {
            std::env::set_var("LUMEN_LOOKAHEAD_DECODE", "1");
            std::env::set_var("LUMEN_LOOKAHEAD_W", v.label_w.to_string());
            std::env::set_var("LUMEN_LOOKAHEAD_G", v.label_g.to_string());
            std::env::set_var("LUMEN_LOOKAHEAD_N", ngram_n.to_string());
        }
    }
}

/// Per-prompt sample for one sub-run.
struct PromptSample {
    /// `decode_total_ms / n_decode_tokens` — comparable across variants.
    avg_token_ms: f64,
    /// Full token sequence (for bit-identical check).
    tokens: Vec<u32>,
    /// avg_committed for this prompt: committed_total / attempts (lookahead only).
    /// `1.0` for baseline.
    avg_committed: f64,
    /// `committed_total` for this prompt (lookahead only); 0 for baseline.
    committed_total: usize,
    /// pool_len at end of this prompt's generation (lookahead only).
    pool_len: usize,
}

/// Run one sub-run: a sequence of prompts under a fixed variant.
fn run_subrun(
    backend: &mut Qwen35MoeBackend,
    label: &str,
    variant: Variant,
    n_prompts: usize,
    n_tokens: usize,
    ngram_n: usize,
) -> Result<Vec<PromptSample>> {
    apply_variant(variant, ngram_n);
    eprintln!(
        "\n=== sub-run [{label}] variant={} {n_prompts} prompts × {n_tokens} tokens ===",
        variant.label()
    );

    let mut samples = Vec::with_capacity(n_prompts);
    for (p_idx, prompt) in PROMPTS.iter().take(n_prompts).enumerate() {
        let ids = backend.encode(prompt)?;
        let t_total = Instant::now();
        let out = backend.generate_with_opts(&ids, n_tokens, 0.0, 1.0, 0, 1.0)?;
        let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;

        let n_gen = out.len();
        if n_gen <= 1 {
            eprintln!("[bench {label}] prompt {p_idx}: only {n_gen} tokens, skipping");
            continue;
        }
        let per_step = backend.last_decode_step_ms().to_vec();
        let decode_total_ms: f64 = per_step.iter().sum();
        let n_decode_tokens = (n_gen - 1) as f64; // first token came from prefill argmax
        let avg_token_ms = decode_total_ms / n_decode_tokens.max(1.0);

        let (avg_committed, committed_total, pool_len) = match backend.last_lookahead_stats() {
            Some(s) => {
                let avg_c = if s.attempts > 0 {
                    s.committed_total as f64 / s.attempts as f64
                } else {
                    1.0
                };
                (avg_c, s.committed_total, s.pool_len)
            }
            None => (1.0, 0, 0),
        };

        println!(
            "BENCH run={label} variant={} p={p_idx} n_gen={n_gen} \
             total_ms={total_ms:.2} decode_total_ms={decode_total_ms:.2} \
             avg_token_ms={avg_token_ms:.3} tok_per_s={:.2} \
             avg_committed={avg_committed:.2} pool_len={pool_len} \
             tokens={}",
            1000.0 / avg_token_ms.max(1e-9),
            variant.label(),
            out.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","),
        );

        samples.push(PromptSample {
            avg_token_ms,
            tokens: out,
            avg_committed,
            committed_total,
            pool_len,
        });
    }
    Ok(samples)
}

fn welch_t(a: &[f64], b: &[f64]) -> (f64, f64, f64, f64) {
    let na = a.len() as f64;
    let nb = b.len() as f64;
    let mean_a = a.iter().sum::<f64>() / na.max(1.0);
    let mean_b = b.iter().sum::<f64>() / nb.max(1.0);
    let var_a = a.iter().map(|v| (v - mean_a).powi(2)).sum::<f64>() / (na - 1.0).max(1.0);
    let var_b = b.iter().map(|v| (v - mean_b).powi(2)).sum::<f64>() / (nb - 1.0).max(1.0);
    let se = (var_a / na + var_b / nb).sqrt().max(1e-12);
    let t = (mean_a - mean_b) / se;
    (mean_a, mean_b, t, se)
}

fn cooldown(secs: u64) {
    if secs == 0 {
        return;
    }
    eprintln!("\n[cooldown] {secs}s ...");
    std::thread::sleep(std::time::Duration::from_secs(secs));
}

/// Build the variant set per env config. Always [BASE, W4G0, ...sweep].
fn build_variants() -> Vec<Variant> {
    let mut variants = vec![Variant::baseline(), Variant::lookahead(4, 0)];

    let custom_ws = std::env::var("LD_BENCH_WS").ok();
    let custom_gs = std::env::var("LD_BENCH_GS").ok();
    let sweep = std::env::var("LD_BENCH_SWEEP")
        .map(|v| v == "1")
        .unwrap_or(false);

    if custom_ws.is_some() || custom_gs.is_some() {
        let ws: Vec<usize> = custom_ws
            .as_deref()
            .unwrap_or("4")
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        let gs: Vec<usize> = custom_gs
            .as_deref()
            .unwrap_or("0")
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        variants.clear();
        variants.push(Variant::baseline());
        for &w in &ws {
            for &g in &gs {
                let v = Variant::lookahead(w, g);
                if !variants.contains(&v) {
                    variants.push(v);
                }
            }
        }
    } else if sweep {
        for &w in &[4usize, 5usize] {
            for &g in &[0usize, 2, 3] {
                let v = Variant::lookahead(w, g);
                if !variants.contains(&v) {
                    variants.push(v);
                }
            }
        }
    }
    variants
}

fn main() -> Result<()> {
    let shard_dir = std::env::var("LUMEN_QWEN35_SHARDS")
        .context("LUMEN_QWEN35_SHARDS required")?;
    let shard_dir = PathBuf::from(shard_dir);
    let model_id = std::env::var("MODEL_ID")
        .unwrap_or_else(|_| "mlx-community/Qwen3.6-35B-A3B-mxfp4".into());

    let quick = std::env::var("LD_BENCH_QUICK")
        .map(|v| v == "1")
        .unwrap_or(false);
    let n_prompts: usize = std::env::var("LD_BENCH_PROMPTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(if quick { 2 } else { 6 });
    let n_tokens: usize = std::env::var("LD_BENCH_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(if quick { 8 } else { 20 });
    let cool_s: u64 = std::env::var("LD_BENCH_COOLDOWN_S")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(if quick { 5 } else { 60 });
    let ngram_n: usize = std::env::var("LD_BENCH_NGRAM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    if n_prompts > PROMPTS.len() {
        anyhow::bail!(
            "LD_BENCH_PROMPTS {n_prompts} > PROMPTS.len() {}",
            PROMPTS.len()
        );
    }

    let variants = build_variants();
    eprintln!("=== Lookahead-decode A/B bench (single-process) ===");
    eprintln!("model: {model_id}");
    eprintln!(
        "n_prompts={n_prompts}, n_tokens={n_tokens}, cool-down={cool_s}s, quick={quick}"
    );
    eprintln!(
        "variants: {}",
        variants
            .iter()
            .map(|v| v.label())
            .collect::<Vec<_>>()
            .join(", ")
    );

    let gpu_ctx = std::sync::Arc::new(MxFp4Context::new()?);
    let mut backend = Qwen35MoeBackend::load(&model_id, &shard_dir, gpu_ctx)?;

    // Pre-warm each variant's path (1 prompt × 4 tokens) so JIT/cache costs
    // aren't billed to measurement.
    eprintln!("\n[warmup] pre-warming {} variant paths ...", variants.len());
    let warm_ids = backend.encode(PROMPTS[0])?;
    for v in &variants {
        apply_variant(*v, ngram_n);
        let _ = backend.generate_with_opts(&warm_ids, 4, 0.0, 1.0, 0, 1.0)?;
    }

    cooldown(cool_s);

    // ── Run 1 (forward) ───────────────────────────────────────────────────
    let mut run1: Vec<Vec<PromptSample>> = Vec::with_capacity(variants.len());
    for v in &variants {
        run1.push(run_subrun(
            &mut backend,
            &format!("R1_{}", v.label()),
            *v,
            n_prompts,
            n_tokens,
            ngram_n,
        )?);
    }

    cooldown(cool_s);

    // ── Run 2 (reverse) ───────────────────────────────────────────────────
    let mut run2: Vec<Vec<PromptSample>> = Vec::with_capacity(variants.len());
    for v in variants.iter().rev() {
        run2.push(run_subrun(
            &mut backend,
            &format!("R2_{}", v.label()),
            *v,
            n_prompts,
            n_tokens,
            ngram_n,
        )?);
    }
    run2.reverse(); // align indices with `variants` and `run1`

    // ── Pool prompt-level avg_token_ms per variant ────────────────────────
    let mut pooled_ms: Vec<Vec<f64>> = Vec::with_capacity(variants.len());
    let mut pooled_tokens: Vec<Vec<Vec<u32>>> = Vec::with_capacity(variants.len());
    let mut pooled_committed_attempts: Vec<(usize, usize)> = Vec::with_capacity(variants.len());
    let mut last_pool_len: Vec<usize> = Vec::with_capacity(variants.len());
    for vi in 0..variants.len() {
        let mut pool_ms: Vec<f64> = Vec::new();
        let mut pool_tok: Vec<Vec<u32>> = Vec::new();
        let mut sum_committed = 0usize;
        let mut sum_attempts = 0usize;
        let mut latest_pool_len = 0usize;
        for sub in [&run1[vi], &run2[vi]] {
            for s in sub.iter() {
                pool_ms.push(s.avg_token_ms);
                pool_tok.push(s.tokens.clone());
                sum_committed += s.committed_total;
                if !variants[vi].is_baseline() && s.avg_committed > 0.0 {
                    // attempts ≈ committed_total / avg_committed (recover from per-prompt stat)
                    let est = (s.committed_total as f64 / s.avg_committed).round() as usize;
                    sum_attempts += est;
                }
                latest_pool_len = s.pool_len;
            }
        }
        pooled_ms.push(pool_ms);
        pooled_tokens.push(pool_tok);
        pooled_committed_attempts.push((sum_committed, sum_attempts));
        last_pool_len.push(latest_pool_len);
    }

    // ── Per-variant summary ───────────────────────────────────────────────
    println!();
    println!("======================================================================");
    println!("Lookahead-decode A/B summary");
    println!(
        "  pooled n per variant: ~{}",
        pooled_ms.first().map(|p| p.len()).unwrap_or(0)
    );
    println!("  N (n-gram) = {ngram_n}");
    println!();
    println!(
        "  {:<8} {:>10} {:>10} {:>10} {:>10} {:>9}",
        "variant", "mean_ms", "tok/s", "avg_cmt", "Δ_vs_BASE", "pool_len"
    );
    let base_mean = if !pooled_ms.is_empty() && !pooled_ms[0].is_empty() {
        pooled_ms[0].iter().sum::<f64>() / pooled_ms[0].len() as f64
    } else {
        0.0
    };
    for vi in 0..variants.len() {
        let v = &variants[vi];
        let pool = &pooled_ms[vi];
        if pool.is_empty() {
            continue;
        }
        let mean = pool.iter().sum::<f64>() / pool.len() as f64;
        let tps = 1000.0 / mean.max(1e-9);
        let (committed, attempts) = pooled_committed_attempts[vi];
        let avg_cmt = if v.is_baseline() {
            1.0
        } else if attempts > 0 {
            committed as f64 / attempts as f64
        } else {
            0.0
        };
        let delta = if vi == 0 {
            0.0
        } else {
            100.0 * (mean - base_mean) / base_mean.max(1e-9)
        };
        println!(
            "  {:<8} {:>10.3} {:>10.2} {:>10.2} {:>9.2}% {:>9}",
            v.label(),
            mean,
            tps,
            avg_cmt,
            delta,
            last_pool_len[vi],
        );
    }
    println!("======================================================================");

    // ── Welch's t for each lookahead variant vs baseline ──────────────────
    if variants.len() >= 2 && !pooled_ms[0].is_empty() {
        println!();
        println!("Welch's t vs baseline (BASE):");
        for vi in 1..variants.len() {
            let v = &variants[vi];
            let pool_b = &pooled_ms[0];
            let pool_v = &pooled_ms[vi];
            if pool_v.is_empty() {
                continue;
            }
            let (mean_b, mean_v, t, se) = welch_t(pool_b, pool_v);
            let delta = mean_v - mean_b;
            let pct = 100.0 * delta / mean_b.max(1e-9);
            let signal = if t.abs() >= 5.0 {
                "STRONG"
            } else if t.abs() >= 2.0 {
                "MILD"
            } else {
                "WASH"
            };
            println!(
                "  {:<8} mean_b={:.3}  mean_v={:.3}  Δ={:+.3} ({:+.2}%)  σ={:+.2} se={:.3} [{}]",
                v.label(),
                mean_b,
                mean_v,
                delta,
                pct,
                t,
                se,
                signal
            );
        }
    }

    // ── Bit-identical check: baseline R1 vs each variant R1 ───────────────
    println!();
    println!("Bit-identical check (greedy temp=0; G=0 must match BASE; G>0 may diverge):");
    let base_run1 = &run1[0];
    for vi in 1..variants.len() {
        let v = &variants[vi];
        let var_run1 = &run1[vi];
        let n = base_run1.len().min(var_run1.len());
        let mut match_count = 0usize;
        let mut first_div: Option<(usize, usize, u32, u32)> = None;
        for p in 0..n {
            let bt = &base_run1[p].tokens;
            let vt = &var_run1[p].tokens;
            if bt == vt {
                match_count += 1;
            } else if first_div.is_none() {
                let div_idx = bt
                    .iter()
                    .zip(vt.iter())
                    .position(|(a, b)| a != b)
                    .unwrap_or(bt.len().min(vt.len()));
                let bt_v = bt.get(div_idx).copied().unwrap_or(u32::MAX);
                let vt_v = vt.get(div_idx).copied().unwrap_or(u32::MAX);
                first_div = Some((p, div_idx, bt_v, vt_v));
            }
        }
        let expected_match = v.label_g == 0;
        let verdict = if expected_match && match_count == n {
            "PASS (lossless)"
        } else if expected_match {
            "FAIL (lossy under G=0!)"
        } else {
            "INFO (G>0, divergence allowed)"
        };
        println!(
            "  {:<8} match={match_count}/{n}  [{verdict}]",
            v.label()
        );
        if let Some((p, idx, b_id, v_id)) = first_div {
            println!(
                "      first divergence: prompt {p} at token {idx}: BASE={} {}={}",
                b_id,
                v.label(),
                v_id
            );
        }
    }

    // ── Decision-gate summary for W4G0 vs BASE ────────────────────────────
    let w4g0_pos = variants
        .iter()
        .position(|v| v.label_w == 4 && v.label_g == 0);
    if let Some(vi) = w4g0_pos {
        let pool_v = &pooled_ms[vi];
        if !pool_v.is_empty() {
            let mean_v = pool_v.iter().sum::<f64>() / pool_v.len() as f64;
            let pct = if base_mean > 0.0 {
                100.0 * (mean_v - base_mean) / base_mean
            } else {
                0.0
            };
            let (committed, attempts) = pooled_committed_attempts[vi];
            let avg_cmt = if attempts > 0 {
                committed as f64 / attempts as f64
            } else {
                0.0
            };
            println!();
            println!("L.3 decision gate (W=4 G=0):");
            println!("  avg_committed = {avg_cmt:.2}");
            println!("  ms/token Δ vs BASE = {pct:+.2}%");
            let gate = if avg_cmt >= 1.5 {
                "PASS-STRONG (≥1.5) → flip default ON"
            } else if avg_cmt >= 1.0 {
                "MARGINAL (1.0-1.5) → consider G>0 sweep"
            } else {
                "FAIL (<1.0) → NEGATIVE memo, keep infra, retire after G>0 try"
            };
            println!("  gate verdict: {gate}");
        }
    }

    // Reset env to baseline before exiting (avoid surprising downstream).
    apply_variant(Variant::baseline(), ngram_n);
    let _ = LookaheadStats::default(); // tie struct into binary unconditionally
    Ok(())
}
