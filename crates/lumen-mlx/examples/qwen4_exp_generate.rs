//! Greedy generation from a Qwen3.8-Flash-Next checkpoint: the smoke test of the
//! `qwen4_exp` port on the real weights.
//!
//! ```text
//! cargo run --release -p lumen-mlx --features mlx-native --example qwen4_exp_generate -- \
//!     <model_dir> "<prompt>" [max_new_tokens=32] [expert_budget_gb=8] [--chat]
//! ```
//!
//! `--force "<text>"` teacher-forces a continuation instead of generating: it feeds the
//! prompt plus that text and reports how often the model's own next-token pick equals
//! the text's next token (the agreement against another engine's output).
//!
//! The routed experts are served from an SSD-backed slot pool of the given size
//! (0 = hold them all in memory, which needs ~80 GB) and the n-gram table is read
//! from disk. `--chat` wraps the prompt in the model's chat template with thinking
//! off; without it the prompt is continued as plain text.

#![cfg(feature = "mlx-native")]

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use lumen_mlx::qwen4_exp::Qwen4ExpModel;
use tokenizers::Tokenizer;

fn argmax(logits: &mlx_rs::Array) -> Result<u32> {
    let flat = mlx_rs::ops::reshape(logits, &[-1])?;
    let idx = mlx_rs::ops::indexing::argmax_axis(&flat, 0, false)?;
    idx.eval()?;
    Ok(idx.item::<u32>())
}

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let chat = args.iter().any(|a| a == "--chat");
    args.retain(|a| a != "--chat");
    let force = args.iter().position(|a| a == "--force").map(|i| {
        let text = args.get(i + 1).cloned().unwrap_or_default();
        args.drain(i..(i + 2).min(args.len()));
        text
    });
    let dir = PathBuf::from(
        args.first()
            .ok_or_else(|| anyhow!("usage: <model_dir> <prompt> ..."))?,
    );
    let prompt = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "The capital of France is".into());
    let max_new: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(32);
    let budget_gb: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8.0);

    let tok =
        Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("tokenizer: {e}"))?;
    let text = if chat {
        // Qwen chat format, thinking off (an empty think block, as the template renders it).
        format!(
            "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
    } else {
        prompt.clone()
    };
    let ids: Vec<u32> = tok
        .encode(text.as_str(), false)
        .map_err(|e| anyhow!("encode: {e}"))?
        .get_ids()
        .to_vec();

    let t0 = Instant::now();
    let budget = (budget_gb > 0.0).then_some((budget_gb * 1e9) as u64);
    let model = Qwen4ExpModel::load(&dir, budget).context("load")?;
    eprintln!(
        "loaded in {:.1}s; {} prompt tokens; MLX active {:.2} GB cache {:.2} GB",
        t0.elapsed().as_secs_f64(),
        ids.len(),
        lumen_mlx::metal_memory::get_active_memory().unwrap_or(0) as f64 / 1e9,
        lumen_mlx::metal_memory::get_cache_memory().unwrap_or(0) as f64 / 1e9,
    );

    let eos = model.config().eos_token_id;
    let mut cache = model.make_cache();
    if let Some(forced) = force {
        let cont: Vec<u32> = tok
            .encode(forced.as_str(), false)
            .map_err(|e| anyhow!("encode: {e}"))?
            .get_ids()
            .to_vec();
        let mut all = ids.clone();
        all.extend_from_slice(&cont);
        let vocab = model.config().vocab_size;
        let mut picks: Vec<u32> = Vec::new();
        for piece in all.chunks(512) {
            let logits = model.forward(piece, &mut cache, false)?;
            let flat = mlx_rs::ops::reshape(&logits, &[piece.len() as i32, vocab as i32])?;
            let am = mlx_rs::ops::indexing::argmax_axis(&flat, -1, false)?;
            am.eval()?;
            picks.extend_from_slice(am.as_slice::<u32>());
        }
        // picks[i] is the model's prediction for token i + 1 of `all`.
        let first = ids.len() - 1;
        let mut agree = 0;
        for (j, &t) in cont.iter().enumerate() {
            if picks[first + j] == t {
                agree += 1;
            } else {
                eprintln!(
                    "  position {j}: model {:?} vs text {:?}",
                    tok.decode(&[picks[first + j]], false).unwrap_or_default(),
                    tok.decode(&[t], false).unwrap_or_default()
                );
            }
        }
        println!(
            "teacher-forced agreement: {agree}/{} = {:.1}%",
            cont.len(),
            100.0 * agree as f64 / cont.len() as f64
        );
        return Ok(());
    }
    let t_prefill = Instant::now();
    let chunk: usize = std::env::var("QWEN4_CHUNK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    let mut last = None;
    if std::env::var("QWEN4_CHUNK_MAJOR").is_ok() {
        // Chunk by chunk: every chunk goes through all the layers before the next starts.
        for piece in ids.chunks(chunk) {
            let logits = model.forward(piece, &mut cache, true)?;
            logits.eval()?;
            eprintln!(
                "  after chunk of {}: MLX active {:.2} GB cache {:.2} GB peak {:.2} GB",
                piece.len(),
                lumen_mlx::metal_memory::get_active_memory().unwrap_or(0) as f64 / 1e9,
                lumen_mlx::metal_memory::get_cache_memory().unwrap_or(0) as f64 / 1e9,
                lumen_mlx::metal_memory::get_peak_memory().unwrap_or(0) as f64 / 1e9,
            );
            last = Some(logits);
        }
    } else {
        // Layer by layer: each layer's experts are read once for the whole prompt.
        let mut cuts: Vec<usize> = (1..)
            .map(|k| k * chunk)
            .take_while(|&c| c < ids.len())
            .collect();
        cuts.push(ids.len());
        let (logits, _) = model.forward_pieces(&ids, &mut cache, &cuts, None)?;
        logits.eval()?;
        last = Some(logits);
    }
    let mut next = argmax(&last.ok_or_else(|| anyhow!("empty prompt"))?)?;
    let prefill_s = t_prefill.elapsed().as_secs_f64();
    eprintln!(
        "prefill {:.2}s ({:.1} tok/s), chunk {chunk}, MLX peak {:.2} GB active {:.2} GB cache {:.2} GB",
        prefill_s,
        ids.len() as f64 / prefill_s,
        lumen_mlx::metal_memory::get_peak_memory().unwrap_or(0) as f64 / 1e9,
        lumen_mlx::metal_memory::get_active_memory().unwrap_or(0) as f64 / 1e9,
        lumen_mlx::metal_memory::get_cache_memory().unwrap_or(0) as f64 / 1e9,
    );

    if let Some(report) = model.offload_report() {
        eprintln!("[prefill] {report}");
    }
    model.offload_reset_timing();
    let mut out = Vec::new();
    let t_decode = Instant::now();
    for _ in 0..max_new {
        out.push(next);
        if next == eos || next == 248046 {
            break;
        }
        let logits = model.forward(&[next], &mut cache, true)?;
        next = argmax(&logits)?;
    }
    let decode_s = t_decode.elapsed().as_secs_f64();
    println!(
        "{}",
        tok.decode(&out, false)
            .map_err(|e| anyhow!("decode: {e}"))?
    );
    eprintln!("tokens: {out:?}");
    eprintln!(
        "decode {} tokens in {:.2}s = {:.2} tok/s",
        out.len(),
        decode_s,
        out.len() as f64 / decode_s
    );
    if let Some((hits, misses)) = model.offload_stats() {
        eprintln!(
            "experts: {hits} hits / {misses} misses (hit rate {:.3})",
            hits as f64 / (hits + misses).max(1) as f64
        );
    }
    if let Some(report) = model.offload_report() {
        eprintln!("{report}");
    }
    if let Some(s) = model.ngram_stats() {
        eprintln!(
            "n-gram rows: {} requested, {} read from disk",
            s.requested, s.read_from_disk
        );
    }
    Ok(())
}
