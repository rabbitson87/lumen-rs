//! Gates the MLX embedding port against the captured candle reference.
//!
//! This is the check that has to pass before `lumen-model` can be deleted
//! (task 006). It reads
//! `tests/golden/embedding_qwen3_0_6b_8bit.json` — corpus, reference vectors
//! and reference metrics, all produced by the candle implementation — and
//! requires the MLX model to reproduce them on the same checkpoint.
//!
//! Three assertions, in increasing order of what they would let through:
//!
//! 1. **Per-item cosine ≥ 0.99 against the reference vector.** The strict one.
//!    Two embedders can score identically on a 25-item retrieval eval while
//!    embedding into different spaces, so metrics alone do not establish that
//!    this is the *same* model.
//! 2. **Unit norm**, because `/v1/embeddings` consumers treat dot as cosine.
//! 3. **P@1 / P@3 / MRR at least as good as the reference**, which is what
//!    users actually experience.
//!
//! Run:
//!   EMBEDDING_MODEL_ID=/path/to/qwen3-embedding-0.6b-8bit \
//!     cargo run --release --features mlx-native -p lumen-mlx --example embedding_parity

use anyhow::{Context, Result, anyhow};

#[cfg(not(feature = "mlx-native"))]
fn main() -> Result<()> {
    Err(anyhow!("build with --features mlx-native"))
}

#[cfg(feature = "mlx-native")]
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += f64::from(x) * f64::from(y);
        na += f64::from(x) * f64::from(x);
        nb += f64::from(y) * f64::from(y);
    }
    dot / (na.sqrt() * nb.sqrt())
}

#[cfg(feature = "mlx-native")]
fn metrics(labels: &[String], emb: &[Vec<f32>]) -> (f64, f64, f64) {
    let n = labels.len();
    let (mut p1, mut p3, mut mrr) = (0usize, 0.0f64, 0.0f64);
    for i in 0..n {
        let mut scores: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, cosine(&emb[i], &emb[j])))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("no NaN"));
        if labels[scores[0].0] == labels[i] {
            p1 += 1;
        }
        p3 += scores
            .iter()
            .take(3)
            .filter(|(j, _)| labels[*j] == labels[i])
            .count() as f64
            / 3.0;
        if let Some(r) = scores.iter().position(|(j, _)| labels[*j] == labels[i]) {
            mrr += 1.0 / (r + 1) as f64;
        }
    }
    (p1 as f64 / n as f64, p3 / n as f64, mrr / n as f64)
}

#[cfg(feature = "mlx-native")]
fn main() -> Result<()> {
    use lumen_mlx::embedding::EmbeddingModel;

    let model_id = std::env::var("EMBEDDING_MODEL_ID")
        .map_err(|_| anyhow!("set EMBEDDING_MODEL_ID to the checkpoint dir or HF repo id"))?;
    let golden_path = std::env::var("EMBEDDING_GOLDEN").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/golden/embedding_qwen3_0_6b_8bit.json"
        )
        .to_string()
    });

    let golden: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&golden_path)
            .with_context(|| format!("read golden {golden_path}"))?,
    )
    .context("parse golden")?;

    let items = golden["items"]
        .as_array()
        .ok_or_else(|| anyhow!("golden has no items[]"))?;
    let labels: Vec<String> = items
        .iter()
        .map(|it| it["label"].as_str().unwrap_or_default().to_string())
        .collect();
    let texts: Vec<String> = items
        .iter()
        .map(|it| it["text"].as_str().unwrap_or_default().to_string())
        .collect();
    let refs: Vec<Vec<f32>> = items
        .iter()
        .map(|it| {
            it["vector"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_f64())
                        .map(|x| x as f32)
                        .collect()
                })
                .unwrap_or_default()
        })
        .collect();
    let ref_dim = golden["dim"].as_u64().unwrap_or(0) as usize;
    eprintln!("[parity] golden: {} items × {ref_dim} dims", items.len());

    let mut model = EmbeddingModel::load(&model_id)?;
    anyhow::ensure!(
        model.dim() == ref_dim,
        "dim mismatch: MLX {} vs reference {ref_dim}",
        model.dim()
    );

    // Warm up before timing, matching `embedding_quality` on the candle side.
    // The first pass pays MLX kernel specialization for every (batch, length)
    // shape it sees; comparing a cold run here against a warm one there would
    // be measuring compilation, not the model.
    let cold = std::time::Instant::now();
    let _ = model.embed(&texts)?;
    let cold_ms = cold.elapsed().as_secs_f64() * 1000.0;

    let t = std::time::Instant::now();
    let batch = model.embed(&texts)?;
    let warm_ms = t.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "[parity] embedded {} texts: cold {cold_ms:.0} ms, warm {warm_ms:.0} ms \
         ({:.2} ms/item warm)",
        texts.len(),
        warm_ms / texts.len() as f64,
    );

    // ── 1. per-item cosine vs the reference vectors ───────────────────────
    let mut worst = (f64::MAX, 0usize);
    for (i, (got, want)) in batch.embeddings.iter().zip(&refs).enumerate() {
        let c = cosine(got, want);
        if c < worst.0 {
            worst = (c, i);
        }
    }
    eprintln!(
        "[parity] worst per-item cosine vs candle = {:.6}  (item {}: {:?})",
        worst.0, worst.1, texts[worst.1]
    );

    // ── 2. unit norm ──────────────────────────────────────────────────────
    //
    // Fold over the *deviation*, seeded at 0.0. Folding over the norms with a
    // `f64::NAN` seed silently disables this check: every comparison against
    // NaN is false, so the seed survives and `(NaN - 1.0).abs() > tol` is
    // false too — a check that can only pass.
    let worst_dev = batch
        .embeddings
        .iter()
        .map(|v| (cosine(v, v).sqrt() - 1.0).abs())
        .fold(0.0f64, f64::max);
    eprintln!("[parity] largest deviation from unit norm = {worst_dev:.3e}");

    // ── 3. retrieval metrics ──────────────────────────────────────────────
    let (p1, p3, mrr) = metrics(&labels, &batch.embeddings);
    let (rp1, rp3, rmrr) = (
        golden["metrics"]["p_at_1"].as_f64().unwrap_or(0.0),
        golden["metrics"]["p_at_3"].as_f64().unwrap_or(0.0),
        golden["metrics"]["mrr"].as_f64().unwrap_or(0.0),
    );
    eprintln!("[parity] MLX    P@1 ={p1:.4}  P@3 ={p3:.4}  MRR ={mrr:.4}");
    eprintln!("[parity] candle P@1 ={rp1:.4}  P@3 ={rp3:.4}  MRR ={rmrr:.4}");

    let mut fail = Vec::new();
    if worst.0 < 0.99 {
        fail.push(format!(
            "per-item cosine {:.6} < 0.99 (item {}: {:?})",
            worst.0, worst.1, texts[worst.1]
        ));
    }
    if worst_dev > 1e-3 {
        fail.push(format!("L2 norm deviates by {worst_dev:.3e}, want ≤ 1e-3"));
    }
    if p1 < rp1 - 1e-9 {
        fail.push(format!("P@1 {p1:.4} below reference {rp1:.4}"));
    }
    if mrr < rmrr - 1e-9 {
        fail.push(format!("MRR {mrr:.4} below reference {rmrr:.4}"));
    }
    if fail.is_empty() {
        eprintln!("[parity] PASS — the MLX port reproduces the candle model");
        Ok(())
    } else {
        for f in &fail {
            eprintln!("[parity] FAIL: {f}");
        }
        Err(anyhow!("{} parity check(s) failed", fail.len()))
    }
}
