//! Microbench: TurboQuant compressed attention scores — old vs new (precomputed QJL).
//!
//! Compares three implementations producing identical mathematical results:
//!   1. CPU reference  (lumen_core::TurboQuantCompressor::estimate_inner_products)
//!   2. GPU legacy     (tq_compressed_attention_scores)
//!   3. GPU optimized  (tq_qjl_project_query + tq_compressed_attention_scores_v2)
//!
//! For each setting (dim, qjl_m, n_kv) reports:
//!   - Cosine similarity (CPU vs GPU-old vs GPU-new) — must be ~1.0
//!   - Wall time per call (median over N runs)
//!   - Speedup new vs old
//!
//! Usage:
//!     cargo run --release -p lumen-metal --example bench_qjl_precompute

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

struct Setting {
    dim: usize,
    qjl_m: usize,
    n_kv: usize,
    bits: u32,
}

fn run_setting(s: &Setting, iters: usize) {
    let config = TurboQuantConfig {
        bits: s.bits,
        qjl_m: s.qjl_m,
        seed: 42,
        lloyd_max_iter: 1000,
        head_dim: s.dim,
    };
    let compressor = TurboQuantCompressor::new(config);

    // Generate KV vectors and query
    let mut rng = StdRng::seed_from_u64(99);
    let vectors: Vec<f32> = (0..s.n_kv * s.dim)
        .map(|_| rng.sample::<f32, _>(StandardNormal))
        .collect();
    let query: Vec<f32> = (0..s.dim)
        .map(|_| rng.sample::<f32, _>(StandardNormal))
        .collect();

    // CPU reference scores
    let compressed = compressor.compress(&vectors, s.n_kv);
    let cpu_scores = compressor.estimate_inner_products(&query, &compressed);

    // True (uncompressed) scores
    let true_scores: Vec<f32> = (0..s.n_kv)
        .map(|i| {
            query
                .iter()
                .zip(&vectors[i * s.dim..(i + 1) * s.dim])
                .map(|(&a, &b)| a * b)
                .sum()
        })
        .collect();

    // ── GPU setup ─────────────────────────────────────────────────────────
    let ctx = MetalContext::new().unwrap();
    let pipelines = ShaderPipelines::new(&ctx.device).unwrap();

    let n_packed = (s.dim + (64 / s.bits as usize) - 1) / (64 / s.bits as usize);
    let n_qjl_packed = (s.qjl_m + 63) / 64;
    let n_levels = 1u32 << s.bits;

    // Pack compressed cache as flat buffers (matching what GPU kernels expect)
    let mut packed_codes_flat: Vec<u64> = Vec::with_capacity(s.n_kv * n_packed);
    let mut scales_flat: Vec<f32> = Vec::with_capacity(s.n_kv);
    let mut qjl_packed_flat: Vec<u64> = Vec::with_capacity(s.n_kv * n_qjl_packed);
    let mut res_norms_flat: Vec<f32> = Vec::with_capacity(s.n_kv);
    for cv in &compressed {
        packed_codes_flat.extend_from_slice(&cv.stage1_packed);
        scales_flat.push(cv.scale);
        qjl_packed_flat.extend_from_slice(&cv.stage2_bits);
        res_norms_flat.push(cv.residual_norm);
    }

    let centroids_f32: Vec<f32> = compressor
        .codebook()
        .centroids
        .iter()
        .map(|&x| x as f32)
        .collect();

    let rotated_query = compressor.rotation().apply(&query);

    // GPU buffers (constant)
    let rq_buf = ctx.buffer_with_data(&rotated_query);
    let q_buf = ctx.buffer_with_data(&query);
    let packed_buf = ctx.buffer_with_data(&packed_codes_flat);
    let scales_buf = ctx.buffer_with_data(&scales_flat);
    let cent_buf = ctx.buffer_with_data(&centroids_f32);
    let qjl_packed_buf = ctx.buffer_with_data(&qjl_packed_flat);
    let qjl_matrix_buf = ctx.buffer_with_data(&compressor_qjl_matrix(&compressor));
    let res_norms_buf = ctx.buffer_with_data(&res_norms_flat);
    let scores_old_buf = ctx.buffer_for::<f32>(s.n_kv);
    let scores_new_buf = ctx.buffer_for::<f32>(s.n_kv);
    let scores_v3_buf = ctx.buffer_for::<f32>(s.n_kv);
    let scores_v4_buf = ctx.buffer_for::<f32>(s.n_kv);
    let scores_v5_buf = ctx.buffer_for::<f32>(s.n_kv);
    let scores_v6_buf = ctx.buffer_for::<f32>(s.n_kv);
    let qjl_proj_buf = ctx.buffer_for::<f32>(s.qjl_m);

    // ── Old kernel (single call: scores) ─────────────────────────────────
    let bench_old = || {
        kernels::attention::compressed_attention_scores(
            &ctx,
            &pipelines,
            &rq_buf,
            &q_buf,
            &packed_buf,
            &scales_buf,
            &cent_buf,
            &qjl_packed_buf,
            &qjl_matrix_buf,
            &res_norms_buf,
            &scores_old_buf,
            s.dim as u32,
            s.n_kv as u32,
            s.bits,
            n_packed as u32,
            s.qjl_m as u32,
            n_qjl_packed as u32,
            n_levels,
        )
        .unwrap();
    };
    bench_old(); // warmup
    let dur_old = time_median(iters, bench_old);

    // ── New kernel (precompute + scores_v2) ──────────────────────────────
    let bench_new = || {
        kernels::attention::qjl_project_query(
            &ctx,
            &pipelines,
            &q_buf,
            &qjl_matrix_buf,
            &qjl_proj_buf,
            s.dim as u32,
            s.qjl_m as u32,
        )
        .unwrap();
        kernels::attention::compressed_attention_scores_v2(
            &ctx,
            &pipelines,
            &rq_buf,
            &qjl_proj_buf,
            &packed_buf,
            &scales_buf,
            &cent_buf,
            &qjl_packed_buf,
            &res_norms_buf,
            &scores_new_buf,
            s.dim as u32,
            s.n_kv as u32,
            s.bits,
            n_packed as u32,
            s.qjl_m as u32,
            n_qjl_packed as u32,
            n_levels,
        )
        .unwrap();
    };
    bench_new(); // warmup
    let dur_new = time_median(iters, bench_new);

    // ── V3 kernel (precompute + scores_v3 with float4 Stage 1) ───────────
    let bench_v3 = || {
        kernels::attention::qjl_project_query(
            &ctx,
            &pipelines,
            &q_buf,
            &qjl_matrix_buf,
            &qjl_proj_buf,
            s.dim as u32,
            s.qjl_m as u32,
        )
        .unwrap();
        kernels::attention::compressed_attention_scores_v3(
            &ctx,
            &pipelines,
            &rq_buf,
            &qjl_proj_buf,
            &packed_buf,
            &scales_buf,
            &cent_buf,
            &qjl_packed_buf,
            &res_norms_buf,
            &scores_v3_buf,
            s.dim as u32,
            s.n_kv as u32,
            s.bits,
            n_packed as u32,
            s.qjl_m as u32,
            n_qjl_packed as u32,
            n_levels,
        )
        .unwrap();
    };
    bench_v3(); // warmup
    let dur_v3 = time_median(iters, bench_v3);

    // ── V4 kernel (precompute + scores_v4 with centroids in threadgroup memory) ──
    let bench_v4 = || {
        kernels::attention::qjl_project_query(
            &ctx,
            &pipelines,
            &q_buf,
            &qjl_matrix_buf,
            &qjl_proj_buf,
            s.dim as u32,
            s.qjl_m as u32,
        )
        .unwrap();
        kernels::attention::compressed_attention_scores_v4(
            &ctx,
            &pipelines,
            &rq_buf,
            &qjl_proj_buf,
            &packed_buf,
            &scales_buf,
            &cent_buf,
            &qjl_packed_buf,
            &res_norms_buf,
            &scores_v4_buf,
            s.dim as u32,
            s.n_kv as u32,
            s.bits,
            n_packed as u32,
            s.qjl_m as u32,
            n_qjl_packed as u32,
            n_levels,
        )
        .unwrap();
    };
    bench_v4(); // warmup
    let dur_v4 = time_median(iters, bench_v4);

    // ── V5 kernel (precompute + scores_v5 with fp16 Stage 1) ─────────────
    let bench_v5 = || {
        kernels::attention::qjl_project_query(
            &ctx,
            &pipelines,
            &q_buf,
            &qjl_matrix_buf,
            &qjl_proj_buf,
            s.dim as u32,
            s.qjl_m as u32,
        )
        .unwrap();
        kernels::attention::compressed_attention_scores_v5(
            &ctx,
            &pipelines,
            &rq_buf,
            &qjl_proj_buf,
            &packed_buf,
            &scales_buf,
            &cent_buf,
            &qjl_packed_buf,
            &res_norms_buf,
            &scores_v5_buf,
            s.dim as u32,
            s.n_kv as u32,
            s.bits,
            n_packed as u32,
            s.qjl_m as u32,
            n_qjl_packed as u32,
            n_levels,
        )
        .unwrap();
    };
    bench_v5(); // warmup
    let dur_v5 = time_median(iters, bench_v5);

    // ── V6 kernel (precompute + scores_v6 with SIMD-group reduction) ─────
    let bench_v6 = || {
        kernels::attention::qjl_project_query(
            &ctx,
            &pipelines,
            &q_buf,
            &qjl_matrix_buf,
            &qjl_proj_buf,
            s.dim as u32,
            s.qjl_m as u32,
        )
        .unwrap();
        kernels::attention::compressed_attention_scores_v6(
            &ctx,
            &pipelines,
            &rq_buf,
            &qjl_proj_buf,
            &packed_buf,
            &scales_buf,
            &cent_buf,
            &qjl_packed_buf,
            &res_norms_buf,
            &scores_v6_buf,
            s.dim as u32,
            s.n_kv as u32,
            s.bits,
            n_packed as u32,
            s.qjl_m as u32,
            n_qjl_packed as u32,
            n_levels,
        )
        .unwrap();
    };
    bench_v6(); // warmup
    let dur_v6 = time_median(iters, bench_v6);

    // ── Production split path verification: v6 scores + parallel softmax ──
    // Compares against CPU softmax of cpu_scores (reference) — checks that the
    // production fused path migration produces a softmax distribution within
    // FP tolerance of the CPU reference.
    let split_softmax_buf = ctx.buffer_for::<f32>(s.n_kv);
    {
        // Run scores_v6 into a fresh buffer, then softmax_parallel in place.
        kernels::attention::qjl_project_query(
            &ctx, &pipelines, &q_buf, &qjl_matrix_buf, &qjl_proj_buf,
            s.dim as u32, s.qjl_m as u32,
        ).unwrap();
        kernels::attention::compressed_attention_scores_v6(
            &ctx, &pipelines, &rq_buf, &qjl_proj_buf, &packed_buf, &scales_buf,
            &cent_buf, &qjl_packed_buf, &res_norms_buf, &split_softmax_buf,
            s.dim as u32, s.n_kv as u32, s.bits, n_packed as u32,
            s.qjl_m as u32, n_qjl_packed as u32, n_levels,
        ).unwrap();
        kernels::attention::softmax_parallel(
            &ctx, &pipelines, &split_softmax_buf, s.n_kv as u32,
        ).unwrap();
    }
    let gpu_split_softmax: Vec<f32> = ctx.read_buffer(&split_softmax_buf, s.n_kv);

    // CPU softmax of cpu_scores (reference)
    let cpu_max = cpu_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let cpu_exp_sum: f32 = cpu_scores.iter().map(|&x| (x - cpu_max).exp()).sum();
    let cpu_softmax: Vec<f32> = cpu_scores.iter().map(|&x| (x - cpu_max).exp() / cpu_exp_sum).collect();
    let cos_split_cpu_sm = cosine(&gpu_split_softmax, &cpu_softmax);
    let max_diff_split = gpu_split_softmax.iter().zip(&cpu_softmax).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let split_sum: f32 = gpu_split_softmax.iter().sum();

    let gpu_old_scores: Vec<f32> = ctx.read_buffer(&scores_old_buf, s.n_kv);
    let gpu_new_scores: Vec<f32> = ctx.read_buffer(&scores_new_buf, s.n_kv);
    let gpu_v3_scores: Vec<f32> = ctx.read_buffer(&scores_v3_buf, s.n_kv);
    let gpu_v4_scores: Vec<f32> = ctx.read_buffer(&scores_v4_buf, s.n_kv);
    let gpu_v5_scores: Vec<f32> = ctx.read_buffer(&scores_v5_buf, s.n_kv);
    let gpu_v6_scores: Vec<f32> = ctx.read_buffer(&scores_v6_buf, s.n_kv);

    let cos_cpu_true = cosine(&cpu_scores, &true_scores);
    let cos_gpu_old_true = cosine(&gpu_old_scores, &true_scores);
    let cos_gpu_new_true = cosine(&gpu_new_scores, &true_scores);
    let cos_gpu_v3_true = cosine(&gpu_v3_scores, &true_scores);
    let cos_gpu_v4_true = cosine(&gpu_v4_scores, &true_scores);
    let cos_gpu_v5_true = cosine(&gpu_v5_scores, &true_scores);
    let cos_gpu_v6_true = cosine(&gpu_v6_scores, &true_scores);
    let cos_v2_v3 = cosine(&gpu_new_scores, &gpu_v3_scores);
    let cos_v3_v4 = cosine(&gpu_v3_scores, &gpu_v4_scores);
    let cos_v3_v5 = cosine(&gpu_v3_scores, &gpu_v5_scores);
    let cos_v3_v6 = cosine(&gpu_v3_scores, &gpu_v6_scores);

    let max_abs_diff_old_new = gpu_old_scores
        .iter()
        .zip(&gpu_new_scores)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_abs_diff_v2_v3 = gpu_new_scores
        .iter()
        .zip(&gpu_v3_scores)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_abs_diff_v3_v4 = gpu_v3_scores
        .iter()
        .zip(&gpu_v4_scores)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_abs_diff_v3_v5 = gpu_v3_scores
        .iter()
        .zip(&gpu_v5_scores)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_abs_diff_v3_v6 = gpu_v3_scores
        .iter()
        .zip(&gpu_v6_scores)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let speedup_v2 = dur_old.as_secs_f64() / dur_new.as_secs_f64();
    let speedup_v3 = dur_old.as_secs_f64() / dur_v3.as_secs_f64();
    let speedup_v4 = dur_old.as_secs_f64() / dur_v4.as_secs_f64();
    let speedup_v5 = dur_old.as_secs_f64() / dur_v5.as_secs_f64();
    let speedup_v6 = dur_old.as_secs_f64() / dur_v6.as_secs_f64();
    let speedup_v3_over_v2 = dur_new.as_secs_f64() / dur_v3.as_secs_f64();
    let speedup_v4_over_v3 = dur_v3.as_secs_f64() / dur_v4.as_secs_f64();
    let speedup_v5_over_v3 = dur_v3.as_secs_f64() / dur_v5.as_secs_f64();
    let speedup_v6_over_v3 = dur_v3.as_secs_f64() / dur_v6.as_secs_f64();

    println!(
        "dim={} qjl_m={} n_kv={} bits={}",
        s.dim, s.qjl_m, s.n_kv, s.bits
    );
    println!(
        "  cos(*,true): cpu={:.6} v1={:.6} v2={:.6} v3={:.6} v4={:.6} v5={:.6} v6={:.6}",
        cos_cpu_true, cos_gpu_old_true, cos_gpu_new_true, cos_gpu_v3_true, cos_gpu_v4_true, cos_gpu_v5_true, cos_gpu_v6_true
    );
    println!(
        "  max|Δ| v1-v2={:.3e}  v2-v3={:.3e}  v3-v4={:.3e}  v3-v5={:.3e}  v3-v6={:.3e}    cos(v2,v3)={:.6}  cos(v3,v4)={:.6}  cos(v3,v5)={:.6}  cos(v3,v6)={:.6}",
        max_abs_diff_old_new, max_abs_diff_v2_v3, max_abs_diff_v3_v4, max_abs_diff_v3_v5, max_abs_diff_v3_v6, cos_v2_v3, cos_v3_v4, cos_v3_v5, cos_v3_v6
    );
    println!(
        "  v1={:>7.1}μs v2={:>7.1}μs v3={:>7.1}μs v4={:>7.1}μs v5={:>7.1}μs v6={:>7.1}μs",
        dur_old.as_secs_f64() * 1e6,
        dur_new.as_secs_f64() * 1e6,
        dur_v3.as_secs_f64() * 1e6,
        dur_v4.as_secs_f64() * 1e6,
        dur_v5.as_secs_f64() * 1e6,
        dur_v6.as_secs_f64() * 1e6,
    );
    println!(
        "  speedup vs v1: v2={:.2}× v3={:.2}× v4={:.2}× v5={:.2}× v6={:.2}×    chain: v3/v2={:.2}× v4/v3={:.2}× v5/v3={:.2}× v6/v3={:.2}×",
        speedup_v2, speedup_v3, speedup_v4, speedup_v5, speedup_v6, speedup_v3_over_v2, speedup_v4_over_v3, speedup_v5_over_v3, speedup_v6_over_v3,
    );
    println!(
        "  prod split (v6+softmax_parallel): cos(split,cpu_sm)={:.6}  max|Δ|={:.3e}  Σ={:.6} (must be ~1.0)",
        cos_split_cpu_sm, max_diff_split, split_sum
    );
    println!();
}

fn compressor_qjl_matrix(c: &TurboQuantCompressor) -> Vec<f32> {
    // Reproduce the QJLProjector held inside the compressor by re-creating with
    // the same seed offset. This avoids exposing private fields.
    let qjl =
        lumen_core::qjl::QJLProjector::new(c.config.head_dim, c.config.qjl_m, c.config.seed.wrapping_add(1));
    qjl.proj_matrix
}

fn main() {
    let iters = 50;
    let settings = [
        // realistic head_dim across modern LLMs (Qwen=128, Llama=128, GPT=64-256)
        Setting { dim: 64,  qjl_m: 64,  n_kv: 1024,  bits: 3 },
        Setting { dim: 128, qjl_m: 64,  n_kv: 1024,  bits: 3 },
        Setting { dim: 128, qjl_m: 128, n_kv: 1024,  bits: 3 },
        Setting { dim: 128, qjl_m: 64,  n_kv: 4096,  bits: 3 },
        Setting { dim: 128, qjl_m: 64,  n_kv: 8192,  bits: 3 },
        Setting { dim: 256, qjl_m: 128, n_kv: 4096,  bits: 3 },
        Setting { dim: 256, qjl_m: 128, n_kv: 8192,  bits: 3 },
        // larger cases — beyond dispatch floor, expose true compute-bound win
        Setting { dim: 256, qjl_m: 128, n_kv: 16384, bits: 3 },
        Setting { dim: 512, qjl_m: 256, n_kv: 8192,  bits: 3 },
        Setting { dim: 512, qjl_m: 256, n_kv: 16384, bits: 3 },
    ];

    println!("=== TurboQuant QJL precompute optimization microbench ===");
    println!(
        "Each setting runs {} iterations; reporting median wall time.\n",
        iters
    );

    for s in &settings {
        run_setting(s, iters);
    }
}
