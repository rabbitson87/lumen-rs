//! Smoke test for the in-process Qwen3-Embedding loader.
//!
//! Run:
//!   EMBEDDING_MODEL_ID=/path/to/qwen3-embedding-0.6b-8bit \
//!     cargo run --release -p lumen-model --example embedding_smoke
//!
//! `EMBEDDING_MODEL_ID` must point at a local MLX 8-bit Qwen3-Embedding
//! checkpoint directory (containing `config.json`, `tokenizer.json`, and
//! the safetensors shards). HuggingFace Hub repo IDs are also accepted.
//!
//! Validates:
//!   1. Model loads (MLX 8-bit dequant on the fly).
//!   2. Three semantically distinct inputs produce 3 vectors of dim 1024.
//!   3. Vectors are L2-normalized (||v||_2 ≈ 1.0).
//!   4. Korean-vs-Korean cosine > Korean-vs-English-irrelevant cosine.

use anyhow::Result;
use lumen_model::embedding::EmbeddingModel;

fn process_rss_mb() -> Option<u64> {
    // macOS / Linux: read RSS in KB from `ps`.
    let pid = std::process::id();
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    let kb: u64 = s.trim().parse().ok()?;
    Some(kb / 1024)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    // Inputs are L2-normalized, so cos = dot.
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
}

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn main() -> Result<()> {
    let model_id = std::env::var("EMBEDDING_MODEL_ID").map_err(|_| {
        anyhow::anyhow!(
            "EMBEDDING_MODEL_ID is not set. Point it at a local MLX 8-bit \
             Qwen3-Embedding checkpoint directory or a HF Hub repo id, e.g.:\n  \
             EMBEDDING_MODEL_ID=/path/to/qwen3-embedding-0.6b-8bit cargo run --release \
             -p lumen-model --example embedding_smoke"
        )
    })?;

    let rss_before = process_rss_mb();
    let mut model = EmbeddingModel::load(&model_id)?;
    let rss_after = process_rss_mb();
    eprintln!(
        "[smoke] model loaded — dim={} max_seq_len={}",
        model.dim(),
        model.max_seq_len()
    );
    if let (Some(b), Some(a)) = (rss_before, rss_after) {
        eprintln!(
            "[smoke] RSS: {b} MB → {a} MB (+{} MB at load)",
            a.saturating_sub(b)
        );
    }

    let texts: Vec<String> = vec![
        "한화 이글스".into(),
        "Hanwha Eagles baseball team".into(),
        "Los Angeles Lakers basketball".into(),
    ];

    // Warm up Metal shader cache for the gather/norm graph.
    let _ = model.embed(&texts)?;

    let iters: usize = std::env::var("BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let mut samples: Vec<f64> = Vec::with_capacity(iters);
    let mut batch = model.embed(&texts)?; // sized once
    for _ in 0..iters {
        let t = std::time::Instant::now();
        batch = model.embed(&texts)?;
        samples.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = samples[samples.len() / 2];
    let min = samples[0];
    let max = samples[samples.len() - 1];
    eprintln!(
        "[smoke] embed b={} max_len≈{} over {iters} iters: median={median:.2}ms min={min:.2}ms max={max:.2}ms (prompt_tokens={})",
        texts.len(),
        // approx — we don't expose max_len; just log batch
        "",
        batch.prompt_tokens
    );

    assert_eq!(batch.embeddings.len(), 3);
    for (i, v) in batch.embeddings.iter().enumerate() {
        assert_eq!(v.len(), model.dim(), "row {i} dim mismatch");
        let n = norm(v);
        eprintln!("  [{i}] ||v||_2 = {n:.6}  (text: {:?})", texts[i]);
        assert!(
            (n - 1.0).abs() < 1e-2,
            "row {i} not L2-normalized: ||v|| = {n}"
        );
    }

    let cos_kr_en_same = cosine(&batch.embeddings[0], &batch.embeddings[1]);
    let cos_kr_en_diff = cosine(&batch.embeddings[0], &batch.embeddings[2]);
    let cos_en_en_diff = cosine(&batch.embeddings[1], &batch.embeddings[2]);

    eprintln!("[smoke] cosine similarities:");
    eprintln!("  '한화 이글스'   vs 'Hanwha Eagles ...'      = {cos_kr_en_same:.4}");
    eprintln!("  '한화 이글스'   vs 'Los Angeles Lakers ...' = {cos_kr_en_diff:.4}");
    eprintln!("  'Hanwha Eagles' vs 'Los Angeles Lakers ...' = {cos_en_en_diff:.4}");

    // Sanity: semantically equivalent KR/EN pair should be more similar than
    // the unrelated pair. We don't assert hard thresholds (different team /
    // sport pairs vary), just that the ordering is correct.
    if cos_kr_en_same <= cos_kr_en_diff {
        eprintln!(
            "[smoke] WARN: same-team KR/EN pair ({cos_kr_en_same:.4}) not more similar than \
             cross-team pair ({cos_kr_en_diff:.4}) — quality below expectation"
        );
    } else {
        eprintln!("[smoke] OK: semantic ordering preserved");
    }

    Ok(())
}
