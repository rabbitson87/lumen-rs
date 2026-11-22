//! The `inkling2` text model on the real `Inkling-2-Mini` checkpoint: load, prefill, greedy decode,
//! and a comparison with the oMLX reference's logits for the same prompt.
//!
//! ```text
//! cargo run --release -p lumen-mlx --features mlx-native --example inkling2_forward -- \
//!     <model_dir> ["<prompt>"] [--new N] [--reference ref.safetensors] [--no-round]
//! ```
//!
//! `--reference` is a safetensors file with `ids` (int32 `[L]`) and `logits` (float32 `[L, vocab]`,
//! from the oMLX reference run over the same checkpoint) and optionally `greedy` (the reference's
//! greedy continuation): the prompt is then those ids, and the report says how closely the port's
//! logits follow the reference's. `--no-round` turns the loop controllers' bf16 rounding off.
//!
//! The prompt is continued as plain text: the chat template is not wired yet.

#![cfg(feature = "mlx-native")]

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use lumen_mlx::inkling2::{Inkling2Model, LoadOptions};
use mlx_rs::Array;
use tokenizers::Tokenizer;

fn host_f32(a: &Array) -> Result<Vec<f32>> {
    let a = a.as_dtype(mlx_rs::Dtype::Float32)?;
    a.eval()?;
    Ok(a.try_as_slice::<f32>()?.to_vec())
}

fn argmax(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Per-position comparison of two `[L, vocab]` logit buffers over the live ids: how many positions
/// pick the same top-1, the smallest cosine, and the largest |difference| (overall and per position).
fn report(label: &str, ours: &[f32], theirs: &[f32], vocab: usize, live: usize) {
    let (mut worst, mut min_cos, mut agree, mut rows) = (0.0f32, 1.0f32, 0usize, 0usize);
    let mut per_position = Vec::new();
    for (a, b) in ours.chunks(vocab).zip(theirs.chunks(vocab)) {
        let (a, b) = (&a[..live], &b[..live]);
        let diff = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max);
        per_position.push(format!("{diff:.2}"));
        worst = worst.max(diff);
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let (na, nb) = (
            a.iter().map(|x| x * x).sum::<f32>().sqrt(),
            b.iter().map(|x| x * x).sum::<f32>().sqrt(),
        );
        min_cos = min_cos.min(dot / (na * nb).max(1e-30));
        agree += usize::from(argmax(a) == argmax(b));
        rows += 1;
    }
    println!(
        "{label}: top-1 agrees at {agree}/{rows} positions, min cosine {min_cos:.6}, max |logit diff| {worst:.3e}\n  per position: {}",
        per_position.join(" ")
    );
}

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut flag_value = |name: &str| -> Option<String> {
        let at = args.iter().position(|a| a == name)?;
        let value = args.get(at + 1).cloned();
        args.drain(at..(at + 2).min(args.len()));
        value
    };
    let new_tokens: usize = flag_value("--new")
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let reference = flag_value("--reference");
    let chunk: Option<usize> = flag_value("--chunk").and_then(|s| s.parse().ok());
    let round = if let Some(at) = args.iter().position(|a| a == "--no-round") {
        args.remove(at);
        false
    } else {
        true
    };
    let dir =
        PathBuf::from(args.first().ok_or_else(|| {
            anyhow!("usage: <model_dir> [\"<prompt>\"] [--new N] [--reference f]")
        })?);
    let prompt = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "The capital of France is Paris. The capital of Japan is".into());
    let tok =
        Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("tokenizer: {e}"))?;

    let t = Instant::now();
    let model = Inkling2Model::load(
        &dir,
        LoadOptions {
            round_loop_mix: round,
        },
    )
    .context("load the checkpoint")?;
    println!(
        "loaded in {:.1}s: {} layers, {} layer visits per token, loop bf16 rounding {}",
        t.elapsed().as_secs_f64(),
        model.args().num_hidden_layers,
        model.num_visits(),
        if round { "on" } else { "off" }
    );

    let want = reference
        .as_ref()
        .map(|p| Array::load_safetensors(p).map_err(|e| anyhow!("reference {p}: {e}")))
        .transpose()?;
    let ids: Vec<u32> = match &want {
        Some(w) => w["ids"]
            .as_slice::<i32>()
            .iter()
            .map(|&t| t as u32)
            .collect(),
        None => tok
            .encode(prompt.as_str(), false)
            .map_err(|e| anyhow!("encode: {e}"))?
            .get_ids()
            .to_vec(),
    };
    println!(
        "prompt: {} tokens: {:?}",
        ids.len(),
        tok.decode(&ids, false).unwrap_or_default()
    );

    let mut cache = model.make_cache();
    let vocab = model.args().vocab_size;
    let live = model.args().unpadded_vocab_size;
    let t = Instant::now();
    let logits = model.forward(&ids, &mut cache, false)?;
    let rows = host_f32(&logits)?;
    println!(
        "prefill {} tokens: {:.2}s",
        ids.len(),
        t.elapsed().as_secs_f64()
    );

    if let Some(w) = &want {
        report(
            "against the reference",
            &rows,
            &host_f32(&w["logits"])?,
            vocab,
            live,
        );
    }
    if let Some(n) = chunk {
        // The same implementation with another chunking: MLX picks other kernels for other row
        // counts, so this is the arithmetic noise floor of the comparison above.
        let mut other = model.make_cache();
        let mut parts = Vec::new();
        for piece in ids.chunks(n.max(1)) {
            parts.extend(host_f32(&model.forward(piece, &mut other, false)?)?);
        }
        report(
            &format!("chunked prefill (chunks of {n}) against the single prefill"),
            &parts,
            &rows,
            vocab,
            live,
        );
    }

    let mut next = argmax(&rows[rows.len() - vocab..][..live]);
    let mut generated = Vec::new();
    let t = Instant::now();
    for _ in 0..new_tokens {
        generated.push(next);
        let step = host_f32(&model.forward(&[next], &mut cache, true)?)?;
        next = argmax(&step[..live]);
    }
    let secs = t.elapsed().as_secs_f64();
    println!(
        "decode {} tokens: {:.2}s ({:.2} tok/s)\ncontinuation: {:?}",
        new_tokens,
        secs,
        new_tokens as f64 / secs,
        tok.decode(&generated, false).unwrap_or_default()
    );
    if let Some(greedy) = want.as_ref().and_then(|w| w.get("greedy")) {
        let theirs: Vec<u32> = greedy.as_slice::<i32>().iter().map(|&t| t as u32).collect();
        let same = generated
            .iter()
            .zip(&theirs)
            .take_while(|(a, b)| a == b)
            .count();
        println!(
            "greedy continuation matches the reference for {same} of {} tokens",
            theirs.len().min(generated.len())
        );
    }
    let kv = cache.kv_bytes()?;
    println!(
        "KV cache: {} tokens, {:.2} MiB ({:.1} KiB per token)",
        cache.len(),
        kv as f64 / 1_048_576.0,
        kv as f64 / 1024.0 / cache.len().max(1) as f64
    );
    Ok(())
}
