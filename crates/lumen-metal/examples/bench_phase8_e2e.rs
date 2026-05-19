//! TurboQuant end-to-end attention benchmark.
//!
//! Measures the full optimized TurboQuant attention pipeline against an FP32
//! full-precision attention reference. Establishes the lumen-rs side of the
//! comparison vs MLX native 4-bit (which is run separately and compared later).
//!
//! Pipeline measured (matches lib.rs::compressed_attention production path):
//!   1. tq_qjl_project_query    — precomputed QJL projection
//!   2. tq_compressed_attention_scores_v6  — SIMD-group cooperative dot
//!   3. tq_softmax_parallel     — three-pass single-TG softmax (any n_kv)
//!   4. tq_compressed_value_gather_multi   — GQA fan-out value gather
//!
//! Reference (FP32 attention):
//!   - scores[i] = q · K[i]     (no quantization)
//!   - softmax over scores
//!   - output[d] = Σ_i softmax[i] · V[i][d]
//!
//! Reports per setting:
//!   - Memory: FP16 KV bytes vs TurboQuant compressed bytes (ratio)
//!   - Accuracy: cos(output_tq, output_ref), KL(softmax_tq || softmax_ref)
//!   - Speed: median wall time per attention call
//!
//! Usage:
//!     cargo run --release -p lumen-metal --example bench_phase8_e2e

use rand::prelude::*;
use rand_distr::StandardNormal;

use lumen_core::compressor::TurboQuantCompressor;
use lumen_core::config::TurboQuantConfig;
use lumen_metal::device::MetalContext;
use lumen_metal::kernels;
use lumen_metal::pipeline::ShaderPipelines;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-10)
}

fn kl_divergence(p: &[f32], q: &[f32]) -> f32 {
    // KL(p || q) = Σ p_i * log(p_i / q_i)
    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| {
            if pi < 1e-12 {
                0.0
            } else {
                pi * ((pi / qi.max(1e-12)).ln())
            }
        })
        .sum()
}

fn time_median<F: FnMut()>(iters: usize, mut f: F) -> std::time::Duration {
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = std::time::Instant::now();
        f();
        samples.push(t.elapsed());
    }
    samples.sort();
    samples[iters / 2]
}

#[derive(Clone)]
struct E2ESetting {
    label: &'static str,
    dim: usize,
    n_kv: usize,
    gqa_ratio: u32,
    bits: u32,
}

fn run_setting(s: &E2ESetting, iters: usize) {
    let qjl_m = s.dim / 2;
    let config = TurboQuantConfig {
        bits: s.bits,
        qjl_m,
        seed: 42,
        lloyd_max_iter: 1000,
        head_dim: s.dim,
    };
    let compressor = TurboQuantCompressor::new(config);

    // Synthetic data
    let mut rng = StdRng::seed_from_u64(99);
    let keys: Vec<f32> = (0..s.n_kv * s.dim)
        .map(|_| rng.sample::<f32, _>(StandardNormal))
        .collect();
    let values: Vec<f32> = (0..s.n_kv * s.dim)
        .map(|_| rng.sample::<f32, _>(StandardNormal))
        .collect();
    let query: Vec<f32> = (0..s.dim)
        .map(|_| rng.sample::<f32, _>(StandardNormal))
        .collect();

    // ── Reference: FP32 full-precision attention ────────────────────────
    let scale = 1.0f32 / (s.dim as f32).sqrt();
    let scores_ref: Vec<f32> = (0..s.n_kv)
        .map(|i| {
            let k = &keys[i * s.dim..(i + 1) * s.dim];
            query.iter().zip(k).map(|(&a, &b)| a * b).sum::<f32>() * scale
        })
        .collect();
    let max_ref = scores_ref.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_ref: Vec<f32> = scores_ref.iter().map(|&x| (x - max_ref).exp()).collect();
    let sum_ref: f32 = exp_ref.iter().sum();
    let softmax_ref: Vec<f32> = exp_ref.iter().map(|&e| e / sum_ref).collect();
    let mut output_ref = vec![0.0f32; s.dim];
    for i in 0..s.n_kv {
        let v = &values[i * s.dim..(i + 1) * s.dim];
        let w = softmax_ref[i];
        for d in 0..s.dim {
            output_ref[d] += w * v[d];
        }
    }

    // ── TurboQuant: compress K (for scores) and V (for value gather) ────
    let key_compressed = compressor.compress(&keys, s.n_kv);
    let val_compressed = compressor.compress(&values, s.n_kv);

    let n_packed = (s.dim + (64 / s.bits as usize) - 1) / (64 / s.bits as usize);
    let n_qjl_packed = (qjl_m + 63) / 64;
    let n_levels = 1u32 << s.bits;

    let mut k_packed: Vec<u64> = Vec::with_capacity(s.n_kv * n_packed);
    let mut k_scales: Vec<f32> = Vec::with_capacity(s.n_kv);
    let mut k_qjl: Vec<u64> = Vec::with_capacity(s.n_kv * n_qjl_packed);
    let mut k_resnorms: Vec<f32> = Vec::with_capacity(s.n_kv);
    for cv in &key_compressed {
        k_packed.extend_from_slice(&cv.stage1_packed);
        k_scales.push(cv.scale);
        k_qjl.extend_from_slice(&cv.stage2_bits);
        k_resnorms.push(cv.residual_norm);
    }
    let mut v_packed: Vec<u64> = Vec::with_capacity(s.n_kv * n_packed);
    let mut v_scales: Vec<f32> = Vec::with_capacity(s.n_kv);
    for cv in &val_compressed {
        v_packed.extend_from_slice(&cv.stage1_packed);
        v_scales.push(cv.scale);
    }

    // Compression ratio: FP16 KV bytes (K+V) vs TurboQuant bytes
    let fp16_bytes = (2 * s.n_kv * s.dim * 2) as f64; // K+V × dim × 2 bytes (fp16)
    let tq_bytes = (
        // K side
        k_packed.len() * 8       // codes (u64)
        + k_scales.len() * 4     // f32
        + k_qjl.len() * 8        // qjl bits (u64)
        + k_resnorms.len() * 4   // f32
        // V side (no qjl/resnorms — only stage 1 is used)
        + v_packed.len() * 8
        + v_scales.len() * 4
    ) as f64;
    let compress_ratio = fp16_bytes / tq_bytes;

    let centroids_f32: Vec<f32> = compressor
        .codebook()
        .centroids
        .iter()
        .map(|&x| x as f32)
        .collect();
    let rotation_flat: Vec<f32> = compressor.rotation().matrix.clone();
    let qjl_matrix_flat: Vec<f32> = {
        let qjl = lumen_core::qjl::QJLProjector::new(s.dim, qjl_m, 42u64.wrapping_add(1));
        qjl.proj_matrix
    };
    let rotated_query = compressor.rotation().apply(&query);

    // ── GPU buffers ──────────────────────────────────────────────────────
    let ctx = MetalContext::new().unwrap();
    let pipelines = ShaderPipelines::new(&ctx.device).unwrap();

    let q_buf = ctx.buffer_with_data(&query);
    let rq_buf = ctx.buffer_with_data(&rotated_query);
    let qjl_matrix_buf = ctx.buffer_with_data(&qjl_matrix_flat);
    let cent_buf = ctx.buffer_with_data(&centroids_f32);
    let rot_buf = ctx.buffer_with_data(&rotation_flat);

    let k_packed_buf = ctx.buffer_with_data(&k_packed);
    let k_scales_buf = ctx.buffer_with_data(&k_scales);
    let k_qjl_buf = ctx.buffer_with_data(&k_qjl);
    let k_resnorms_buf = ctx.buffer_with_data(&k_resnorms);

    let v_packed_buf = ctx.buffer_with_data(&v_packed);
    let v_scales_buf = ctx.buffer_with_data(&v_scales);

    let qjl_proj_buf = ctx.buffer_for::<f32>(qjl_m);
    let scores_buf = ctx.buffer_for::<f32>(s.n_kv);
    let output_buf = ctx.buffer_for::<f32>(s.gqa_ratio as usize * s.dim);

    // Bench: full pipeline (timing)
    let bench_pipeline = || {
        kernels::attention::qjl_project_query(
            &ctx,
            &pipelines,
            &q_buf,
            &qjl_matrix_buf,
            &qjl_proj_buf,
            s.dim as u32,
            qjl_m as u32,
        )
        .unwrap();
        kernels::attention::compressed_attention_scores_v6(
            &ctx,
            &pipelines,
            &rq_buf,
            &qjl_proj_buf,
            &k_packed_buf,
            &k_scales_buf,
            &cent_buf,
            &k_qjl_buf,
            &k_resnorms_buf,
            &scores_buf,
            s.dim as u32,
            s.n_kv as u32,
            s.bits,
            n_packed as u32,
            qjl_m as u32,
            n_qjl_packed as u32,
            n_levels,
        )
        .unwrap();
        kernels::attention::softmax_parallel(&ctx, &pipelines, &scores_buf, s.n_kv as u32).unwrap();
        kernels::attention::compressed_value_gather_multi(
            &ctx,
            &pipelines,
            &scores_buf,
            &v_packed_buf,
            &v_scales_buf,
            &cent_buf,
            &rot_buf,
            &output_buf,
            s.dim as u32,
            s.n_kv as u32,
            s.bits,
            n_packed as u32,
            n_levels,
            s.gqa_ratio,
        )
        .unwrap();
    };
    bench_pipeline(); // warmup
    let dur = time_median(iters, bench_pipeline);

    // Accuracy run: capture RAW scores BEFORE softmax (for primary accuracy metric)
    kernels::attention::qjl_project_query(
        &ctx,
        &pipelines,
        &q_buf,
        &qjl_matrix_buf,
        &qjl_proj_buf,
        s.dim as u32,
        qjl_m as u32,
    )
    .unwrap();
    kernels::attention::compressed_attention_scores_v6(
        &ctx,
        &pipelines,
        &rq_buf,
        &qjl_proj_buf,
        &k_packed_buf,
        &k_scales_buf,
        &cent_buf,
        &k_qjl_buf,
        &k_resnorms_buf,
        &scores_buf,
        s.dim as u32,
        s.n_kv as u32,
        s.bits,
        n_packed as u32,
        qjl_m as u32,
        n_qjl_packed as u32,
        n_levels,
    )
    .unwrap();
    let scores_tq_raw: Vec<f32> = ctx.read_buffer(&scores_buf, s.n_kv);

    // FP32 reference raw scores (no 1/sqrt(dim))
    let scores_ref_raw: Vec<f32> = (0..s.n_kv)
        .map(|i| {
            let k = &keys[i * s.dim..(i + 1) * s.dim];
            query.iter().zip(k).map(|(&a, &b)| a * b).sum::<f32>()
        })
        .collect();

    let cos_scores = cosine(&scores_tq_raw, &scores_ref_raw);
    let max_diff_scores = scores_tq_raw
        .iter()
        .zip(&scores_ref_raw)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // Apply realistic 1/sqrt(dim) attention scale on HOST for both, then compute
    // softmax + weighted V output and compare. This matches how production
    // models normalize attention; without it the softmax saturates and the
    // metric becomes meaningless for both methods.
    let attn_scale = 1.0f32 / (s.dim as f32).sqrt();

    let scaled_tq: Vec<f32> = scores_tq_raw.iter().map(|&x| x * attn_scale).collect();
    let scaled_ref: Vec<f32> = scores_ref_raw.iter().map(|&x| x * attn_scale).collect();

    let softmax_tq = host_softmax(&scaled_tq);
    let softmax_ref = host_softmax(&scaled_ref);

    let cos_softmax = cosine(&softmax_tq, &softmax_ref);
    let kl_tq_ref = kl_divergence(&softmax_tq, &softmax_ref);
    let topk_ref = top_indices(&softmax_ref, 5);
    let topk_tq = top_indices(&softmax_tq, 5);
    let top5_overlap = topk_ref.iter().filter(|i| topk_tq.contains(i)).count();
    let top1_match = topk_ref[0] == topk_tq[0];

    // Weighted V gather on host (using FP32 V)
    let mut output_ref = vec![0.0f32; s.dim];
    let mut output_tq_host = vec![0.0f32; s.dim];
    for i in 0..s.n_kv {
        let v = &values[i * s.dim..(i + 1) * s.dim];
        for d in 0..s.dim {
            output_ref[d] += softmax_ref[i] * v[d];
            output_tq_host[d] += softmax_tq[i] * v[d];
        }
    }
    // Note: output_tq_host uses FP32 V (so it isolates the attention-weight
    // error from the V-quantization error). This is the right metric for
    // "how well does TurboQuant reproduce attention weights".
    let cos_output_attn_only = cosine(&output_tq_host, &output_ref);

    println!(
        "[{}] dim={} n_kv={} gqa_ratio={} bits={}",
        s.label, s.dim, s.n_kv, s.gqa_ratio, s.bits
    );
    println!(
        "  memory: FP16={:.2}MB  TurboQuant={:.2}MB  ratio={:.2}× compression",
        fp16_bytes / 1e6,
        tq_bytes / 1e6,
        compress_ratio
    );
    println!(
        "  scores accuracy (raw inner products): cos={:.6}  max|Δ|={:.3e}",
        cos_scores, max_diff_scores
    );
    println!(
        "  attention dist (with 1/√d scale):     cos(softmax)={:.6}  KL={:.4e}  top-1={}  top-5={}/5",
        cos_softmax, kl_tq_ref, top1_match, top5_overlap
    );
    println!(
        "  attention output (FP32 V, isolates weights): cos={:.6}",
        cos_output_attn_only
    );
    println!(
        "  speed: pipeline median = {:.1}μs / attention call",
        dur.as_secs_f64() * 1e6
    );
    println!();
}

fn host_softmax(scores: &[f32]) -> Vec<f32> {
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = scores.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|&e| e / sum).collect()
}

fn top_indices(v: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap());
    idx.into_iter().take(k).collect()
}

fn main() {
    let iters = 50;
    let settings = [
        // Realistic head_dim (128 for Llama/Qwen) at production-relevant context lengths
        E2ESetting {
            label: "small-decode",
            dim: 128,
            n_kv: 1024,
            gqa_ratio: 8,
            bits: 3,
        },
        E2ESetting {
            label: "medium-decode",
            dim: 128,
            n_kv: 4096,
            gqa_ratio: 8,
            bits: 3,
        },
        E2ESetting {
            label: "long-context",
            dim: 128,
            n_kv: 8192,
            gqa_ratio: 8,
            bits: 3,
        },
        E2ESetting {
            label: "very-long-context",
            dim: 128,
            n_kv: 16384,
            gqa_ratio: 8,
            bits: 3,
        },
        // Larger head_dim (256-512 for some 35B-class models)
        E2ESetting {
            label: "large-head-medium-ctx",
            dim: 256,
            n_kv: 4096,
            gqa_ratio: 8,
            bits: 3,
        },
        E2ESetting {
            label: "large-head-long-ctx",
            dim: 256,
            n_kv: 8192,
            gqa_ratio: 8,
            bits: 3,
        },
        // Bits sweep at a fixed shape — accuracy vs compression tradeoff
        E2ESetting {
            label: "bits-sweep-2",
            dim: 128,
            n_kv: 4096,
            gqa_ratio: 8,
            bits: 2,
        },
        E2ESetting {
            label: "bits-sweep-3",
            dim: 128,
            n_kv: 4096,
            gqa_ratio: 8,
            bits: 3,
        },
        E2ESetting {
            label: "bits-sweep-4",
            dim: 128,
            n_kv: 4096,
            gqa_ratio: 8,
            bits: 4,
        },
    ];

    println!("=== Phase 8 (a) — TurboQuant E2E benchmark ===");
    println!("Pipeline: qjl_project_query → scores_v6 → softmax_parallel → value_gather_multi");
    println!(
        "Each setting runs {} iterations; reporting median wall time.",
        iters
    );
    println!();

    for s in &settings {
        run_setting(s, iters);
    }
}
