//! End-to-end integration test: store → attention_scores → value_gather
//! through the GpuCompressor pool (simulates real model usage).

use rand::prelude::*;
use rand_distr::StandardNormal;

use lumen_core::compressor::TurboQuantCompressor;
use lumen_core::config::TurboQuantConfig;
use lumen_metal::GpuCompressor;

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-10)
}

/// Simulate a decode sequence: prefill 8 tokens, then decode 4 tokens one-by-one.
/// Verify attention scores and value gather produce correct results.
#[test]
fn e2e_prefill_then_decode() {
    let config = TurboQuantConfig {
        bits: 3,
        qjl_m: 64,
        seed: 42,
        lloyd_max_iter: 1000,
        head_dim: 128,
    };
    let dim = config.head_dim;
    let n_layers = 2;
    let n_kv_heads = 2;

    let mut gpu = GpuCompressor::new(config.clone(), n_layers, n_kv_heads, 256).unwrap();
    let cpu = TurboQuantCompressor::new(config);

    let mut rng = StdRng::seed_from_u64(42);
    let mut all_key_vecs: Vec<Vec<f32>> = Vec::new(); // per-token key vectors
    let mut all_val_vecs: Vec<Vec<f32>> = Vec::new();

    // === Prefill: 8 tokens at once ===
    let prefill_len = 8;
    let keys: Vec<f32> = (0..prefill_len * dim)
        .map(|_| rng.sample::<f32, _>(StandardNormal))
        .collect();
    let vals: Vec<f32> = (0..prefill_len * dim)
        .map(|_| rng.sample::<f32, _>(StandardNormal))
        .collect();

    // Store in GPU pool (layer 0, head 0)
    gpu.compress_and_store(0, 0, &keys, prefill_len, true)
        .unwrap();
    gpu.compress_and_store(0, 0, &vals, prefill_len, false)
        .unwrap();

    for i in 0..prefill_len {
        all_key_vecs.push(keys[i * dim..(i + 1) * dim].to_vec());
        all_val_vecs.push(vals[i * dim..(i + 1) * dim].to_vec());
    }

    assert_eq!(gpu.pool.head(0, 0).seq_len(), prefill_len);

    // === Decode: 4 tokens one at a time ===
    for _ in 0..4 {
        let k_new: Vec<f32> = (0..dim)
            .map(|_| rng.sample::<f32, _>(StandardNormal))
            .collect();
        let v_new: Vec<f32> = (0..dim)
            .map(|_| rng.sample::<f32, _>(StandardNormal))
            .collect();

        gpu.compress_and_store(0, 0, &k_new, 1, true).unwrap();
        gpu.compress_and_store(0, 0, &v_new, 1, false).unwrap();

        all_key_vecs.push(k_new);
        all_val_vecs.push(v_new);
    }

    let total_seq = prefill_len + 4;
    assert_eq!(gpu.pool.head(0, 0).seq_len(), total_seq);

    // === Query: compute attention scores ===
    let query: Vec<f32> = (0..dim)
        .map(|_| rng.sample::<f32, _>(StandardNormal))
        .collect();

    let gpu_scores = gpu.attention_scores(0, 0, &query).unwrap();
    assert_eq!(gpu_scores.len(), total_seq);

    // CPU reference: true inner products
    let true_scores: Vec<f32> = all_key_vecs
        .iter()
        .map(|k| query.iter().zip(k.iter()).map(|(&a, &b)| a * b).sum())
        .collect();

    // CPU compressed reference
    let all_keys_flat: Vec<f32> = all_key_vecs.iter().flatten().cloned().collect();
    let cpu_compressed = cpu.compress(&all_keys_flat, total_seq);
    let cpu_scores = cpu.estimate_inner_products(&query, &cpu_compressed);

    let gpu_vs_true = cosine_sim(&gpu_scores, &true_scores);
    let cpu_vs_true = cosine_sim(&cpu_scores, &true_scores);
    let gpu_vs_cpu = cosine_sim(&gpu_scores, &cpu_scores);

    eprintln!(
        "Scores — GPU vs true: {gpu_vs_true:.4}, CPU vs true: {cpu_vs_true:.4}, GPU vs CPU: {gpu_vs_cpu:.4}"
    );

    assert!(gpu_vs_true > 0.90, "GPU scores vs true: {gpu_vs_true}");
    assert!(gpu_vs_cpu > 0.90, "GPU scores vs CPU: {gpu_vs_cpu}");

    // === Softmax + value gather ===
    let max_s = gpu_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut weights: Vec<f32> = gpu_scores.iter().map(|&s| (s - max_s).exp()).collect();
    let sum: f32 = weights.iter().sum();
    for w in &mut weights {
        *w /= sum;
    }

    let gpu_output = gpu.value_gather(0, 0, &weights).unwrap();
    assert_eq!(gpu_output.len(), dim);

    // True weighted sum (uncompressed)
    let mut true_output = vec![0.0f32; dim];
    for (i, w) in weights.iter().enumerate() {
        for d in 0..dim {
            true_output[d] += w * all_val_vecs[i][d];
        }
    }

    let gather_cos = cosine_sim(&gpu_output, &true_output);
    eprintln!("Value gather — GPU vs true: {gather_cos:.4}");
    assert!(gather_cos > 0.90, "Value gather cosine: {gather_cos}");

    // === Clear and verify reset ===
    gpu.clear_cache();
    assert_eq!(gpu.pool.head(0, 0).seq_len(), 0);

    eprintln!("E2E integration test PASSED");
}

/// Test multi-layer, multi-head operation
#[test]
fn e2e_multi_layer_multi_head() {
    let config = TurboQuantConfig {
        bits: 3,
        qjl_m: 64,
        seed: 42,
        lloyd_max_iter: 1000,
        head_dim: 128,
    };
    let dim = config.head_dim;
    let n_layers = 4;
    let n_kv_heads = 2;

    let mut gpu = GpuCompressor::new(config, n_layers, n_kv_heads, 256).unwrap();
    let mut rng = StdRng::seed_from_u64(99);

    // Store 4 tokens across all layers and heads
    for layer in 0..n_layers {
        for head in 0..n_kv_heads {
            let keys: Vec<f32> = (0..4 * dim)
                .map(|_| rng.sample::<f32, _>(StandardNormal))
                .collect();
            let vals: Vec<f32> = (0..4 * dim)
                .map(|_| rng.sample::<f32, _>(StandardNormal))
                .collect();
            gpu.compress_and_store(layer, head, &keys, 4, true).unwrap();
            gpu.compress_and_store(layer, head, &vals, 4, false)
                .unwrap();
        }
    }

    // Verify all layers/heads have correct seq_len
    for layer in 0..n_layers {
        for head in 0..n_kv_heads {
            assert_eq!(gpu.pool.head(layer, head).seq_len(), 4);
        }
    }

    // Query each layer/head and verify non-empty scores
    for layer in 0..n_layers {
        for head in 0..n_kv_heads {
            let query: Vec<f32> = (0..dim)
                .map(|_| rng.sample::<f32, _>(StandardNormal))
                .collect();
            let scores = gpu.attention_scores(layer, head, &query).unwrap();
            assert_eq!(scores.len(), 4);
            // Scores should not all be zero
            let nonzero = scores.iter().any(|&s| s.abs() > 1e-6);
            assert!(nonzero, "all scores zero at layer={layer} head={head}");
        }
    }

    // Clear and verify
    gpu.clear_cache();
    for layer in 0..n_layers {
        for head in 0..n_kv_heads {
            assert_eq!(gpu.pool.head(layer, head).seq_len(), 0);
        }
    }

    eprintln!("Multi-layer multi-head test PASSED");
}
