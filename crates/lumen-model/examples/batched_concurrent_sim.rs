//! Stage 5c-α: continuous-batching simulator.
//!
//! Simulates N concurrent requests sharing one model instance via
//! `forward_batched_decode_v2`. Emits decoded text per seq just like a real
//! scheduler would. This is the numerical + end-to-end validation step
//! before wiring into the HTTP engine loop (Stage 5c-β).
//!
//! Run:
//!   MODEL_ID=.models/google_gemma-4-E4B-it-Q4_K_M.gguf \
//!   N=4 MAX_NEW=32 \
//!     cargo run --release --features paged-kv -p lumen-model \
//!     --example batched_concurrent_sim

#![cfg(feature = "paged-kv")]

use anyhow::Result;
use candle_core::{DType, Tensor};
use lumen_model::gemma_gguf::GemmaGgufModel;

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            best = i;
        }
    }
    best as u32
}

fn is_eos_gemma(tok: u32) -> bool {
    tok == 1 || tok == 106 // <eos> / end-of-turn
}

fn main() -> Result<()> {
    unsafe {
        std::env::set_var("TQ_THRESHOLD", "1");
    }
    let path = std::env::var("MODEL_ID")
        .unwrap_or_else(|_| ".models/google_gemma-4-E4B-it-Q4_K_M.gguf".into());
    let tok_id = std::env::var("TOKENIZER_ID").unwrap_or_else(|_| "google/gemma-4-E4B-it".into());
    let n: usize = std::env::var("N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let max_new: usize = std::env::var("MAX_NEW")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);

    let mut gem = GemmaGgufModel::load(&path, &tok_id)?;
    // Enable paged backend (uses env vars PAGED_KV_MB etc).
    unsafe {
        std::env::set_var("PAGED_KV_MB", "512");
    }
    #[cfg(feature = "paged-kv")]
    gem.enable_paged_kv(42, 2, 256, 512, 6)?;

    // N distinct user prompts.
    let prompts: Vec<String> = (0..n)
        .map(|i| {
            format!(
                "<start_of_turn>user\nCount to {} in words.<end_of_turn>\n<start_of_turn>model\n",
                3 + i
            )
        })
        .collect();

    // Tokenize each prompt.
    let prompt_ids: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| gem.encode(p))
        .collect::<Result<Vec<_>>>()?;

    let seq_ids: Vec<u64> = (1000..1000 + n as u64).collect();
    let device = gem.device().clone();

    // Prefill each seq under its seq_id.
    let t0 = std::time::Instant::now();
    for i in 0..n {
        gem.model_mut().set_current_seq_id(seq_ids[i]);
        let t = Tensor::new(prompt_ids[i].as_slice(), &device)?.unsqueeze(0)?;
        let logits = gem.model_mut().forward(&t, 0)?;
        let v: Vec<f32> = logits.squeeze(0)?.to_dtype(DType::F32)?.to_vec1()?;
        let next = argmax(&v);
        // stash first decode token for each seq
        println!(
            "[seq {}] prefilled {} tokens, first tok={}",
            seq_ids[i],
            prompt_ids[i].len(),
            next
        );
    }
    let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Initialize per-seq decode state.
    // "last_token" starts as the last prompt token (feed it at pos=prompt_len-1 is wrong;
    // we sampled next already above during prefill logits, so use that as the first new token).
    let mut last_tokens: Vec<u32> = Vec::with_capacity(n);
    let mut positions: Vec<usize> = Vec::with_capacity(n);
    let mut generated: Vec<Vec<u32>> = Vec::with_capacity(n);
    let mut active: Vec<bool> = vec![true; n];

    // Re-run prefill to capture first token cleanly (we already did it above; just use the logits).
    // Actually simpler: do an initial batched_decode with last prompt tokens to produce first output.
    for i in 0..n {
        last_tokens.push(*prompt_ids[i].last().unwrap());
        positions.push(prompt_ids[i].len() - 1);
        generated.push(Vec::new());
    }

    // Re-set the seq states cleanly: clear and re-prefill (for a clean starting position).
    gem.model_mut().clear_kv_cache();
    for i in 0..n {
        gem.model_mut().set_current_seq_id(seq_ids[i]);
        // Prefill without the last token (we'll feed it via batched decode).
        let prefix = &prompt_ids[i][..prompt_ids[i].len() - 1];
        if !prefix.is_empty() {
            let t = Tensor::new(prefix, &device)?.unsqueeze(0)?;
            let _ = gem.model_mut().forward(&t, 0)?;
        }
        positions[i] = prefix.len();
    }

    // Decode loop (batched).
    let t_decode = std::time::Instant::now();
    let mut step = 0usize;
    let mut total_new = 0usize;
    while step < max_new && active.iter().any(|&a| a) {
        // Gather only active seqs (simple version: if some seqs finished early, keep running all
        // to preserve batch shape; mark done seqs inactive but still feed last token to keep KV
        // aligned. Real scheduler would compact the batch.)
        let active_idx: Vec<usize> = (0..n).filter(|&i| active[i]).collect();
        if active_idx.is_empty() {
            break;
        }

        let tok_vec: Vec<u32> = active_idx.iter().map(|&i| last_tokens[i]).collect();
        let sids: Vec<u64> = active_idx.iter().map(|&i| seq_ids[i]).collect();
        let pos_vec: Vec<usize> = active_idx.iter().map(|&i| positions[i]).collect();

        let toks = Tensor::new(tok_vec.as_slice(), &device)?.reshape((active_idx.len(), 1))?;
        let logits = gem
            .model_mut()
            .forward_batched_decode_v2(&toks, &sids, &pos_vec)?;
        let flat: Vec<f32> = logits.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let vocab = flat.len() / active_idx.len();

        for (row, &i) in active_idx.iter().enumerate() {
            let next = argmax(&flat[row * vocab..(row + 1) * vocab]);
            generated[i].push(next);
            last_tokens[i] = next;
            positions[i] += 1;
            total_new += 1;
            if is_eos_gemma(next) || generated[i].len() >= max_new {
                active[i] = false;
                let text = gem.decode(&generated[i]).unwrap_or_default();
                println!(
                    "[seq {}] done ({} toks): {}",
                    seq_ids[i],
                    generated[i].len(),
                    text.trim()
                );
            }
        }

        step += 1;
    }
    let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;

    for i in 0..n {
        if active[i] {
            let text = gem.decode(&generated[i]).unwrap_or_default();
            println!(
                "[seq {}] reached max_new ({} toks): {}",
                seq_ids[i],
                generated[i].len(),
                text.trim()
            );
        }
    }

    println!(
        "\nPrefill: {:.0}ms ({} seqs sequentially, total {} tokens)",
        prefill_ms,
        n,
        prompt_ids.iter().map(|p| p.len()).sum::<usize>(),
    );
    println!(
        "Decode:  {:.0}ms for {} tokens across {} seqs ({:.1} tok/s aggregate, {:.1} tok/s per-seq)",
        decode_ms,
        total_new,
        n,
        total_new as f64 / (decode_ms / 1000.0),
        (total_new as f64 / n as f64) / (decode_ms / 1000.0),
    );
    Ok(())
}
