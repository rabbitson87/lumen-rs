//! R3 pre-flight: N-step batched forward ≡ N×1-step sequential forward.
//!
//! For greedy spec decode to be safe, the target model's batched forward over
//! N tokens must be numerically equivalent to N sequential forwards over 1
//! token each. If this breaks, accepted-prefix cache state diverges from a
//! pure sequential decode trajectory and downstream tokens drift.
//!
//! Protocol (per playbook_speculative_decoding.md):
//! 1. Pick a fixed prompt + greedy-decode K=16 reference tokens (block_seq).
//! 2. For each N in sweep:
//!    a. Path A on a fresh seq_id: prefill prompt → forward(block_seq[..N], offset)
//!       → logits_A shape [1, N, vocab]
//!    b. Path B on a different fresh seq_id: prefill prompt → for i in 0..N:
//!       forward([block_seq[i]], offset+i) → collect logits_B[i] shape [1, 1, vocab]
//!    c. Compare: max|logits_A[0,i] - logits_B[i][0,0]| for each position.
//! 3. PASS criteria: max_abs_diff < 1.0 across all N and positions
//!    AND argmax match across all N and positions.
//!
//! Run:
//!   LUMEN_QWEN35_SHARDS=/path/to/Qwen3.6-27B-4bit \
//!     cargo run --release --example preflight_n_step_eq

#![cfg(feature = "turboquant-gpu")]

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use candle_core::Tensor;
use lumen_metal::affine4_gpu::Affine4Context;
use lumen_metal::mxfp4_gpu::MxFp4Context;
use lumen_model::qwen3_5_moe::backend::Qwen35MoeBackend;

const MODEL_ID: &str = "mlx-community/Qwen3.6-27B-4bit";
const REF_BLOCK_LEN: usize = 16;
const PROMPT: &str = "Once upon a time, in a small village by the sea, there lived a curious cat. The cat";

fn build_backend() -> Result<Qwen35MoeBackend> {
    let shard_dir: PathBuf = std::env::var("LUMEN_QWEN35_SHARDS")
        .map(PathBuf::from)
        .map_err(|_| {
            anyhow::anyhow!("set LUMEN_QWEN35_SHARDS=<dir> to the 27B-4bit shards")
        })?;
    let gpu_ctx = Arc::new(MxFp4Context::new()?);
    let affine4_ctx = Arc::new(Affine4Context::new()?);
    Qwen35MoeBackend::load_with_affine4(MODEL_ID, &shard_dir, gpu_ctx, affine4_ctx)
}

/// Generate `n` greedy tokens after prefilling `prompt_ids` on `seq_id`.
/// Returns the generated tokens.
fn greedy_generate(
    backend: &mut Qwen35MoeBackend,
    seq_id: u64,
    prompt_ids: &[u32],
    n: usize,
) -> Result<Vec<u32>> {
    let (first_tok, mut pos) = backend.prefill_sequence(seq_id, prompt_ids)?;
    let mut out = Vec::with_capacity(n);
    out.push(first_tok);
    let mut last = first_tok;
    while out.len() < n {
        let next = backend.decode_step_single(seq_id, last, pos)?;
        pos += 1;
        out.push(next);
        last = next;
    }
    Ok(out)
}

/// `[1, N, vocab]` → `Vec<Vec<f32>>` of length N, each `vocab` long.
fn logits_to_per_pos(t: &Tensor) -> Result<Vec<Vec<f32>>> {
    let dims = t.dims();
    anyhow::ensure!(
        dims.len() == 3 && dims[0] == 1,
        "expected [1, N, vocab], got {:?}",
        dims
    );
    let n = dims[1];
    let vocab = dims[2];
    let flat = t.squeeze(0)?.to_vec2::<f32>()?;
    anyhow::ensure!(flat.len() == n);
    anyhow::ensure!(flat[0].len() == vocab);
    Ok(flat)
}

fn argmax(v: &[f32]) -> u32 {
    let mut max_v = f32::NEG_INFINITY;
    let mut idx = 0u32;
    for (i, &x) in v.iter().enumerate() {
        if x > max_v {
            max_v = x;
            idx = i as u32;
        }
    }
    idx
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn run_path_a(
    backend: &mut Qwen35MoeBackend,
    seq_id: u64,
    prompt: &[u32],
    block: &[u32],
) -> Result<Vec<Vec<f32>>> {
    // Re-prefill into fresh seq_id (no global reset → other seqs untouched).
    let (_first, prompt_len) = backend.prefill_sequence(seq_id, prompt)?;
    // One batched forward over the block at offset = prompt_len.
    let logits = backend.forward_logits_at_offset(seq_id, block, prompt_len)?;
    logits_to_per_pos(&logits)
}

fn run_path_b(
    backend: &mut Qwen35MoeBackend,
    seq_id: u64,
    prompt: &[u32],
    block: &[u32],
) -> Result<Vec<Vec<f32>>> {
    let (_first, prompt_len) = backend.prefill_sequence(seq_id, prompt)?;
    let mut out = Vec::with_capacity(block.len());
    for (i, &tok) in block.iter().enumerate() {
        let l = backend.forward_logits_at_offset(seq_id, &[tok], prompt_len + i)?;
        let per_pos = logits_to_per_pos(&l)?;
        anyhow::ensure!(per_pos.len() == 1);
        out.push(per_pos.into_iter().next().unwrap());
    }
    Ok(out)
}

fn main() -> Result<()> {
    eprintln!("[preflight] loading {MODEL_ID} ...");
    let mut backend = build_backend()?;
    eprintln!("[preflight] loaded.");

    // Tokenize prompt + generate reference block via greedy decode.
    let prompt_ids = backend.encode(PROMPT)?;
    eprintln!(
        "[preflight] prompt='{PROMPT}' → {} tokens",
        prompt_ids.len()
    );

    eprintln!("[preflight] greedy-decoding {REF_BLOCK_LEN} tokens for reference block ...");
    let block = greedy_generate(&mut backend, 9001, &prompt_ids, REF_BLOCK_LEN)?;
    eprintln!("[preflight] block tokens: {block:?}");
    let block_text = backend.decode(&block)?;
    eprintln!("[preflight] block text:   {block_text:?}");

    // Sweep N. Path B is run once at maximum N, then sliced for each smaller N.
    let n_sweep: Vec<usize> = vec![2, 3, 4, 5, 6, 8, REF_BLOCK_LEN];
    let n_max = *n_sweep.iter().max().unwrap();

    eprintln!("[preflight] running Path B (sequential, N={n_max}) ...");
    let logits_b_all = run_path_b(&mut backend, 9100, &prompt_ids, &block[..n_max])?;
    eprintln!("[preflight] Path B done.");

    eprintln!("[preflight] running Path A for each N in {:?} ...", n_sweep);
    eprintln!("\n  N |   max|Δ|     |   argmax matches?  |  PASS/FAIL");
    eprintln!("----+-----------------+--------------------+------------");

    let mut overall_pass = true;
    for &n in &n_sweep {
        let seq = 9200 + n as u64; // unique seq per N
        let logits_a = run_path_a(&mut backend, seq, &prompt_ids, &block[..n])?;
        anyhow::ensure!(logits_a.len() == n);

        let mut worst_diff = 0.0f32;
        let mut argmax_matches = 0usize;
        for i in 0..n {
            let a = &logits_a[i];
            let b = &logits_b_all[i];
            anyhow::ensure!(a.len() == b.len());
            let d = max_abs_diff(a, b);
            if d > worst_diff {
                worst_diff = d;
            }
            if argmax(a) == argmax(b) {
                argmax_matches += 1;
            }
        }

        let argmax_ok = argmax_matches == n;
        let bar1_ok = worst_diff < 1.0;
        let pass = argmax_ok && bar1_ok;
        overall_pass &= pass;

        eprintln!(
            " {n:>2} |  {worst_diff:>13.6e}  |    {argmax_matches:>2}/{n:<2}            | {}",
            if pass { "PASS" } else { "FAIL" }
        );
    }

    eprintln!("\n[preflight] OVERALL: {}", if overall_pass { "PASS — R3 spec decode is feasible" } else { "FAIL — R3 spec decode dead on this stack" });
    if overall_pass {
        Ok(())
    } else {
        anyhow::bail!("preflight FAILED");
    }
}
