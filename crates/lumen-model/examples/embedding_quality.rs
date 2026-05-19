//! Real-prompt quality eval for the in-process Qwen3-Embedding loader.
//!
//! Builds a small labeled multi-domain corpus (Korean + English + mixed) and
//! measures embedding retrieval quality with the standard IR metrics:
//!   - P@1  : top-1 nearest-neighbor has the same category as the query
//!   - P@3  : mean (same-category) fraction in the top-3 neighbors
//!   - MRR  : mean reciprocal rank of the first same-category neighbor
//!
//! Same input → same model → both forward passes (warmup + measured) follow
//! the kernel selection in `Affine8Context::encode_matmul_bf16_dispatch`
//! (qmv_fast by default; force naive with `LUMEN_AFFINE8_NAIVE=1`).
//!
//! Run:
//!   EMBEDDING_MODEL_ID=/path/to/qwen3-embedding-0.6b-8bit \
//!     cargo run --release -p lumen-model --example embedding_quality
//!
//! `EMBEDDING_MODEL_ID` must point at a local MLX 8-bit Qwen3-Embedding
//! checkpoint directory or a HuggingFace Hub repo id.

use anyhow::Result;
use lumen_model::embedding::EmbeddingModel;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
}

/// 5 categories × {Korean phrase, English phrase, mixed/longer phrase} variations.
/// Hand-curated so each category has 4-6 items and intra-category similarity
/// should clearly dominate inter-category similarity for a competent embedder.
fn corpus() -> Vec<(&'static str, &'static str)> {
    vec![
        // ── 1. Korean baseball / KBO ────────────────────────────────────────
        ("baseball_kbo", "한화 이글스 야구단"),
        ("baseball_kbo", "Hanwha Eagles baseball team"),
        ("baseball_kbo", "두산 베어스 KBO 리그"),
        ("baseball_kbo", "LG Twins KBO professional baseball"),
        ("baseball_kbo", "삼성 라이온즈 대구 홈구장"),
        // ── 2. NBA basketball ───────────────────────────────────────────────
        ("basketball_nba", "Los Angeles Lakers basketball team"),
        ("basketball_nba", "LA 레이커스 NBA 농구"),
        ("basketball_nba", "Golden State Warriors NBA championship"),
        ("basketball_nba", "Chicago Bulls Michael Jordan era"),
        ("basketball_nba", "보스턴 셀틱스 NBA 농구팀"),
        // ── 3. Programming languages ────────────────────────────────────────
        ("programming", "Python programming language general purpose"),
        ("programming", "러스트 시스템 프로그래밍 언어"),
        ("programming", "Rust memory-safe systems programming"),
        ("programming", "TypeScript static typing for JavaScript"),
        ("programming", "Go 동시성 프로그래밍 언어"),
        // ── 4. Korean cities ────────────────────────────────────────────────
        ("city_korea", "서울 대한민국의 수도"),
        ("city_korea", "Seoul capital of South Korea"),
        ("city_korea", "부산 한국 제2의 도시 항구"),
        ("city_korea", "Busan major port city in Korea"),
        ("city_korea", "제주도 한국의 휴양 섬"),
        // ── 5. Korean food ──────────────────────────────────────────────────
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

fn run_metrics(labels: &[&str], embeddings: &[Vec<f32>]) -> (f64, f64, f64) {
    let n = labels.len();
    let mut p1_correct = 0usize;
    let mut p3_sum = 0.0f64;
    let mut mrr_sum = 0.0f64;

    for i in 0..n {
        let mut scores: Vec<(usize, f32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, cosine(&embeddings[i], &embeddings[j])))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

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

fn main() -> Result<()> {
    let model_id = std::env::var("EMBEDDING_MODEL_ID").map_err(|_| {
        anyhow::anyhow!(
            "EMBEDDING_MODEL_ID is not set. Point it at a local MLX 8-bit \
             Qwen3-Embedding checkpoint directory or a HF Hub repo id."
        )
    })?;

    let t_load = std::time::Instant::now();
    let mut model = EmbeddingModel::load(&model_id)?;
    eprintln!(
        "[quality] loaded {} dim={} in {:.2}s",
        model_id,
        model.dim(),
        t_load.elapsed().as_secs_f32(),
    );

    let data = corpus();
    let labels: Vec<&str> = data.iter().map(|(l, _)| *l).collect();
    let texts: Vec<String> = data.iter().map(|(_, t)| t.to_string()).collect();
    let n = texts.len();
    eprintln!("[quality] corpus: {n} items across {} categories", {
        let mut set = std::collections::BTreeSet::new();
        for l in &labels {
            set.insert(*l);
        }
        set.len()
    });

    // Warm up + measure once.
    let _ = model.embed(&texts)?;
    let t_embed = std::time::Instant::now();
    let batch = model.embed(&texts)?;
    let embed_ms = t_embed.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "[quality] embed b={n} in {:.2}ms ({:.2}ms/item)",
        embed_ms,
        embed_ms / n as f64
    );

    let (p1, p3, mrr) = run_metrics(&labels, &batch.embeddings);
    eprintln!("[quality] P@1 = {p1:.3}   P@3 = {p3:.3}   MRR = {mrr:.3}");

    // Per-category breakdown.
    let categories: std::collections::BTreeSet<&str> = labels.iter().copied().collect();
    eprintln!("[quality] per-category MRR:");
    for cat in &categories {
        let mut cat_mrr_sum = 0.0f64;
        let mut count = 0usize;
        for i in 0..n {
            if labels[i] != *cat {
                continue;
            }
            count += 1;
            let mut scores: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, cosine(&batch.embeddings[i], &batch.embeddings[j])))
                .collect();
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            if let Some(rank) = scores.iter().position(|(j, _)| labels[*j] == labels[i]) {
                cat_mrr_sum += 1.0 / (rank + 1) as f64;
            }
        }
        eprintln!(
            "  {:<18} ({} items)  MRR = {:.3}",
            cat,
            count,
            cat_mrr_sum / count as f64
        );
    }

    // Diagnostic: print the wrong queries.
    let mut misses: Vec<(String, String, f32, String, f32)> = Vec::new();
    for i in 0..n {
        let mut scores: Vec<(usize, f32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, cosine(&batch.embeddings[i], &batch.embeddings[j])))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top = scores[0];
        if labels[top.0] != labels[i] {
            let first_same = scores
                .iter()
                .find(|(j, _)| labels[*j] == labels[i])
                .copied();
            misses.push((
                texts[i].clone(),
                texts[top.0].clone(),
                top.1,
                first_same
                    .map(|(j, _)| texts[j].clone())
                    .unwrap_or_default(),
                first_same.map(|(_, s)| s).unwrap_or(0.0),
            ));
        }
    }
    if misses.is_empty() {
        eprintln!("[quality] all top-1 neighbors were same-category");
    } else {
        eprintln!(
            "[quality] {} misses (top-1 different category):",
            misses.len()
        );
        for (q, top, ts, want, ws) in misses {
            eprintln!("  ?  {q:?}");
            eprintln!("     top  : {top:?}  cos={ts:.4}");
            eprintln!("     want : {want:?}  cos={ws:.4}");
        }
    }

    Ok(())
}
