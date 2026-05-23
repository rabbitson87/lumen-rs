//! Side-by-side embedding comparison across multiple Qwen3-Embedding
//! variants. Built primarily to validate that the new MXFP8 4B path
//! produces semantically-equivalent embeddings to the already-shipped
//! 4B-4bit-DWQ baseline, while also reporting latency and memory.
//!
//! Compares up to **three** variants in one run (any subset of):
//!   - `EMBED_0` — first model id / local path     (e.g. 0.6B-8bit)
//!   - `EMBED_A` — second model id / local path    (e.g. 4B-4bit-DWQ)
//!   - `EMBED_B` — third model id / local path     (e.g. 4B-mxfp8)
//!
//! At least one of the three must be set. All metrics are computed
//! per-variant on the same labeled corpus, then a cross-variant cosine
//! summary tells you how closely the *new* model agrees with the
//! *reference* on the same input (1.0 = same embedding direction).
//!
//! Reported per-variant:
//!   - load time (model + tokenizer)
//!   - embed batch latency (warmup + measured)
//!   - P@1, P@3, MRR over the labeled corpus (same `corpus()` shape as
//!     `embedding_quality.rs` for cross-comparison)
//!
//! Reported cross-variant (when ≥ 2 set):
//!   - mean cosine between corresponding embeddings (variant_i vs variant_j)
//!   - min cosine + the worst-disagreeing prompt
//!   - Spearman rank correlation of intra-corpus nearest neighbors
//!     (proxy for "do both variants order the same retrieval results?")
//!
//! Run:
//!   ```bash
//!   EMBED_0=mlx-community/Qwen3-Embedding-0.6B-8bit \
//!   EMBED_A=mlx-community/Qwen3-Embedding-4B-4bit-DWQ \
//!   EMBED_B=mlx-community/Qwen3-Embedding-4B-mxfp8 \
//!     cargo run --release -p lumen-model --example embedding_compare
//!   ```
//!
//! Knobs:
//!   `LUMEN_EMBEDDING_TIMING=1` — per-stage tok/fwd/pool timing on stderr.
//!   `LUMEN_EMBEDDING_QUANT_KERNEL=0` — force CPU eager dequant fallback
//!                                      on any quantized variant (regression
//!                                      check; should produce ~identical
//!                                      results to the kernel path).

use anyhow::Result;
use lumen_model::embedding::EmbeddingModel;
use std::time::Instant;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    // Inputs from EmbeddingModel are L2-normalized → dot product == cosine.
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
}

fn corpus() -> Vec<(&'static str, &'static str)> {
    // Mirrors `embedding_quality.rs` so the same baseline numbers apply.
    vec![
        // 1. Korean baseball / KBO
        ("baseball_kbo", "한화 이글스 야구단"),
        ("baseball_kbo", "Hanwha Eagles baseball team"),
        ("baseball_kbo", "두산 베어스 KBO 리그"),
        ("baseball_kbo", "LG Twins KBO professional baseball"),
        ("baseball_kbo", "삼성 라이온즈 대구 홈구장"),
        // 2. NBA basketball
        ("basketball_nba", "Los Angeles Lakers basketball team"),
        ("basketball_nba", "LA 레이커스 NBA 농구"),
        ("basketball_nba", "Golden State Warriors NBA championship"),
        ("basketball_nba", "Chicago Bulls Michael Jordan era"),
        ("basketball_nba", "보스턴 셀틱스 NBA 농구팀"),
        // 3. Programming languages
        ("programming", "Python programming language general purpose"),
        ("programming", "러스트 시스템 프로그래밍 언어"),
        ("programming", "Rust memory-safe systems programming"),
        ("programming", "TypeScript static typing for JavaScript"),
        ("programming", "Go 동시성 프로그래밍 언어"),
        // 4. Korean cities
        ("city_korea", "서울 대한민국의 수도"),
        ("city_korea", "Seoul capital of South Korea"),
        ("city_korea", "부산 한국 제2의 도시 항구"),
        ("city_korea", "Busan major port city in Korea"),
        ("city_korea", "제주도 한국의 휴양 섬"),
        // 5. Korean food
        ("food_korean", "김치찌개 매콤한 한국 전통 요리"),
        ("food_korean", "Bibimbap Korean mixed rice bowl"),
        ("food_korean", "불고기 한국식 양념 소고기 구이"),
        (
            "food_korean",
            "Tteokbokki spicy rice cakes Korean street food",
        ),
        ("food_korean", "삼겹살 한국 돼지고기 구이"),
    ]
}

#[derive(Default)]
struct VariantResult {
    name: String,
    model_id: String,
    dim: usize,
    load_ms: f64,
    embed_ms: f64,
    embeddings: Vec<Vec<f32>>,
    p1: f64,
    p3: f64,
    mrr: f64,
}

fn metrics(labels: &[&str], embeddings: &[Vec<f32>]) -> (f64, f64, f64) {
    let n = labels.len();
    let mut p1_correct = 0usize;
    let mut p3_sum = 0.0f64;
    let mut mrr_sum = 0.0f64;

    for i in 0..n {
        let mut scores: Vec<(usize, f32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, cosine(&embeddings[i], &embeddings[j])))
            .collect();
        scores.sort_by(|a, b| b.1.total_cmp(&a.1));

        if labels[scores[0].0] == labels[i] {
            p1_correct += 1;
        }
        let same_top3 = scores
            .iter()
            .take(3)
            .filter(|(j, _)| labels[*j] == labels[i])
            .count();
        p3_sum += same_top3 as f64 / 3.0;
        if let Some(rank) = scores.iter().position(|(j, _)| labels[*j] == labels[i]) {
            mrr_sum += 1.0 / (rank + 1) as f64;
        }
    }
    (
        p1_correct as f64 / n as f64,
        p3_sum / n as f64,
        mrr_sum / n as f64,
    )
}

/// For each prompt, build the ranked list of every *other* prompt (sorted
/// by descending cosine). Returns `[prompt_idx][rank] = neighbor_idx`.
fn ranked_neighbors(embeddings: &[Vec<f32>]) -> Vec<Vec<usize>> {
    let n = embeddings.len();
    let mut out = vec![Vec::with_capacity(n - 1); n];
    for i in 0..n {
        let mut scores: Vec<(usize, f32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, cosine(&embeddings[i], &embeddings[j])))
            .collect();
        scores.sort_by(|a, b| b.1.total_cmp(&a.1));
        out[i] = scores.into_iter().map(|(j, _)| j).collect();
    }
    out
}

/// Spearman rank correlation of two ranked-neighbor lists for the same
/// query. Returns the mean over all queries — 1.0 means the two variants
/// agree perfectly on retrieval ordering, 0.0 means uncorrelated.
fn spearman_rank_agreement(a: &[Vec<usize>], b: &[Vec<usize>]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    for i in 0..n {
        // Map neighbor → rank in list a, and in list b.
        let m = a[i].len();
        let mut rank_a = vec![0usize; n];
        let mut rank_b = vec![0usize; n];
        for (r, &j) in a[i].iter().enumerate() {
            rank_a[j] = r + 1;
        }
        for (r, &j) in b[i].iter().enumerate() {
            rank_b[j] = r + 1;
        }
        // Spearman ρ = 1 − 6·Σd²/(m·(m²−1))
        let mut d2_sum = 0f64;
        for j in 0..n {
            if j == i {
                continue;
            }
            let d = rank_a[j] as i64 - rank_b[j] as i64;
            d2_sum += (d * d) as f64;
        }
        let mf = m as f64;
        let rho = 1.0 - 6.0 * d2_sum / (mf * (mf * mf - 1.0));
        sum += rho;
    }
    sum / n as f64
}

fn run_variant(
    label: &str,
    model_id: &str,
    texts: &[String],
    labels: &[&str],
) -> Result<VariantResult> {
    eprintln!("=== variant [{label}] ===");
    let t = Instant::now();
    let mut model = EmbeddingModel::load(model_id)?;
    let load_ms = t.elapsed().as_secs_f64() * 1000.0;
    let dim = model.dim();

    let t = Instant::now();
    let _ = model.embed(texts)?;
    let warmup_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let batch = model.embed(texts)?;
    let embed_ms = t.elapsed().as_secs_f64() * 1000.0;

    // Diagnostic: detect NaN/Inf in embeddings before metrics. Quantized
    // paths occasionally produce non-finite outputs on specific inputs;
    // surface the per-prompt damage instead of crashing in sort.
    let mut nan_prompts: Vec<usize> = Vec::new();
    for (i, e) in batch.embeddings.iter().enumerate() {
        if e.iter().any(|v| !v.is_finite()) {
            nan_prompts.push(i);
        }
    }
    if !nan_prompts.is_empty() {
        eprintln!(
            "  WARN: {} / {} embeddings contain NaN/Inf",
            nan_prompts.len(),
            batch.embeddings.len()
        );
        for &i in nan_prompts.iter().take(5) {
            eprintln!("    [{i}] {:?}", texts[i]);
        }
    }

    let (p1, p3, mrr) = metrics(labels, &batch.embeddings);
    eprintln!(
        "  load={load_ms:.0}ms  warmup={warmup_ms:.0}ms  embed_n={}={embed_ms:.1}ms  P@1={p1:.3} P@3={p3:.3} MRR={mrr:.3}",
        texts.len()
    );

    Ok(VariantResult {
        name: label.to_string(),
        model_id: model_id.to_string(),
        dim,
        load_ms,
        embed_ms,
        embeddings: batch.embeddings,
        p1,
        p3,
        mrr,
    })
}

fn cross_variant(a: &VariantResult, b: &VariantResult, texts: &[String]) {
    if a.dim != b.dim {
        eprintln!(
            "[cross] {} (dim={}) vs {} (dim={}) — dimensions differ, skipping cosine",
            a.name, a.dim, b.name, b.dim
        );
    } else {
        let n = a.embeddings.len();
        let mut sum = 0f32;
        let mut min_cos = f32::INFINITY;
        let mut min_idx = 0usize;
        let mut counted = 0usize;
        let mut skipped: Vec<usize> = Vec::new();
        for i in 0..n {
            let c = cosine(&a.embeddings[i], &b.embeddings[i]);
            if !c.is_finite() {
                skipped.push(i);
                continue;
            }
            sum += c;
            counted += 1;
            if c < min_cos {
                min_cos = c;
                min_idx = i;
            }
        }
        if counted == 0 {
            eprintln!(
                "[cross] {} ↔ {}  ALL pairs produced NaN/Inf — skipping",
                a.name, b.name
            );
        } else {
            let mean = sum / counted as f32;
            eprintln!(
                "[cross] {} ↔ {}  mean_cos={mean:.4}  min_cos={min_cos:.4}  worst=\"{}\"  (n={counted}/{n}{})",
                a.name,
                b.name,
                texts[min_idx],
                if skipped.is_empty() {
                    String::new()
                } else {
                    format!(", skipped {} NaN", skipped.len())
                }
            );
        }
    }
    // Spearman is dim-agnostic — works even at different dim, because the
    // ranks are over corpus indices, not embedding components.
    let rank_a = ranked_neighbors(&a.embeddings);
    let rank_b = ranked_neighbors(&b.embeddings);
    let rho = spearman_rank_agreement(&rank_a, &rank_b);
    eprintln!(
        "[cross] {} ↔ {}  Spearman ρ (retrieval-order agreement) = {rho:.4}",
        a.name, b.name
    );
}

fn main() -> Result<()> {
    let env_0 = std::env::var("EMBED_0").ok();
    let env_a = std::env::var("EMBED_A").ok();
    let env_b = std::env::var("EMBED_B").ok();
    let any = env_0.is_some() || env_a.is_some() || env_b.is_some();
    if !any {
        anyhow::bail!(
            "set at least one of EMBED_0 / EMBED_A / EMBED_B to a Qwen3-Embedding model id or local path"
        );
    }

    let data = corpus();
    let labels: Vec<&str> = data.iter().map(|(l, _)| *l).collect();
    let texts: Vec<String> = data.iter().map(|(_, t)| t.to_string()).collect();
    let n = texts.len();
    eprintln!(
        "[compare] corpus: {n} prompts across {} categories",
        labels
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    );

    let mut results: Vec<VariantResult> = Vec::new();
    if let Some(id) = env_0 {
        results.push(run_variant("0", &id, &texts, &labels)?);
    }
    if let Some(id) = env_a {
        results.push(run_variant("A", &id, &texts, &labels)?);
    }
    if let Some(id) = env_b {
        results.push(run_variant("B", &id, &texts, &labels)?);
    }

    // Summary table.
    println!();
    println!(
        "{:<6} {:<60} {:>6} {:>10} {:>10} {:>8} {:>8} {:>8}",
        "tag", "model_id", "dim", "load(ms)", "embed(ms)", "P@1", "P@3", "MRR"
    );
    println!("{}", "-".repeat(118));
    for r in &results {
        let truncated = if r.model_id.len() > 58 {
            format!("…{}", &r.model_id[r.model_id.len() - 57..])
        } else {
            r.model_id.clone()
        };
        println!(
            "{:<6} {:<60} {:>6} {:>10.0} {:>10.1} {:>8.3} {:>8.3} {:>8.3}",
            r.name, truncated, r.dim, r.load_ms, r.embed_ms, r.p1, r.p3, r.mrr
        );
    }

    // Cross-variant comparison (each unordered pair).
    if results.len() >= 2 {
        println!();
        for i in 0..results.len() {
            for j in (i + 1)..results.len() {
                cross_variant(&results[i], &results[j], &texts);
            }
        }
    }

    Ok(())
}
