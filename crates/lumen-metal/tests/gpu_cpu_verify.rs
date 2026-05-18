//! GPU vs CPU verification tests for TurboQuant Metal kernels.
//!
//! Compresses vectors on both GPU (Metal kernels) and CPU (lumen-core),
//! then compares results to ensure bitwise/numerical equivalence.

use lumen_metal::metal::{BatchedEncoderExt, CommandBufferExt, ComputeEncoderCompat};
use rand::prelude::*;
use rand::RngExt;
use rand_distr::StandardNormal;

use lumen_core::bitpack;
use lumen_core::compressor::TurboQuantCompressor;
use lumen_core::config::TurboQuantConfig;
use lumen_core::lloyd_max::LloydMaxCodebook;
use lumen_core::qjl::QJLProjector;
use lumen_core::rotation::RotationMatrix;
use lumen_metal::device::MetalContext;
use lumen_metal::kernels;
use lumen_metal::metal;
use lumen_metal::mtl_size;
use lumen_metal::pipeline::ShaderPipelines;

fn make_config(bits: u32) -> TurboQuantConfig {
    TurboQuantConfig {
        bits,
        qjl_m: 64,
        seed: 42,
        lloyd_max_iter: 1000,
        head_dim: 128,
    }
}

fn random_vectors(rng: &mut StdRng, n: usize, dim: usize) -> Vec<f32> {
    (0..n * dim)
        .map(|_| rng.sample::<f32, _>(StandardNormal))
        .collect()
}

/// Test: GPU rotation + normalization matches CPU
#[test]
fn kernel1_rotate_and_normalize() {
    let config = make_config(3);
    let dim = config.head_dim;
    let n_vecs = 4;
    let mut rng = StdRng::seed_from_u64(99);
    let vectors = random_vectors(&mut rng, n_vecs, dim);

    // CPU
    let rotation = RotationMatrix::random(dim, config.seed);
    let mut cpu_rotated = vec![vec![0.0f32; dim]; n_vecs];
    let mut cpu_scales = vec![0.0f32; n_vecs];
    for v in 0..n_vecs {
        let vec_slice = &vectors[v * dim..(v + 1) * dim];
        let rotated = rotation.apply(vec_slice);
        let norm_sq: f32 = rotated.iter().map(|x| x * x).sum();
        let scale = (norm_sq / dim as f32).sqrt();
        cpu_scales[v] = scale;
        if scale > 1e-10 {
            let inv = 1.0 / scale;
            cpu_rotated[v] = rotated.iter().map(|&x| x * inv).collect();
        } else {
            cpu_rotated[v] = rotated;
        }
    }

    // GPU
    let ctx = MetalContext::new().unwrap();
    let pipelines = ShaderPipelines::new(&ctx.device).unwrap();
    let kv_buf = ctx.buffer_with_data(&vectors);
    let rot_buf = ctx.buffer_with_data(&rotation.matrix);
    let out_buf = ctx.buffer_for::<f32>(n_vecs * dim);
    let scales_buf = ctx.buffer_for::<f32>(n_vecs);

    let cmd = metal::new_command_buffer(&ctx.queue);
    {
        let enc = cmd.auto_compute_encoder();
        let pipeline = pipelines.get("tq_rotate_and_normalize").unwrap();
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(&kv_buf), 0);
        enc.set_buffer(1, Some(&rot_buf), 0);
        enc.set_buffer(2, Some(&out_buf), 0);
        enc.set_buffer(3, Some(&scales_buf), 0);
        set_u32(&enc, 4, dim as u32);
        set_u32(&enc, 5, n_vecs as u32);
        let grid = mtl_size!(dim, n_vecs, 1);
        let tg = mtl_size!(dim, 1, 1);
        enc.dispatch_threads(grid, tg);
        enc.end_encoding();
    }
    cmd.commit();
    cmd.wait_until_completed();

    let gpu_rotated: Vec<f32> = ctx.read_buffer(&out_buf, n_vecs * dim);
    let gpu_scales: Vec<f32> = ctx.read_buffer(&scales_buf, n_vecs);

    // Compare scales
    for v in 0..n_vecs {
        let err = (cpu_scales[v] - gpu_scales[v]).abs();
        assert!(
            err < 1e-4,
            "scale mismatch vec {v}: cpu={} gpu={} err={err}",
            cpu_scales[v],
            gpu_scales[v]
        );
    }

    // Compare rotated+normalized vectors
    let mut max_err = 0.0f32;
    for v in 0..n_vecs {
        for d in 0..dim {
            let cpu_val = cpu_rotated[v][d];
            let gpu_val = gpu_rotated[v * dim + d];
            max_err = max_err.max((cpu_val - gpu_val).abs());
        }
    }
    eprintln!("kernel1 max element error: {max_err:.2e}");
    assert!(
        max_err < 1e-3,
        "rotated values max error {max_err} too large"
    );
}

/// Test: Full compression pipeline (4 kernels) matches CPU compressor
#[test]
fn full_compression_pipeline_matches_cpu() {
    let config = make_config(3);
    let dim = config.head_dim;
    let n_vecs = 8;
    let mut rng = StdRng::seed_from_u64(99);
    let vectors = random_vectors(&mut rng, n_vecs, dim);

    // CPU compression
    let compressor = TurboQuantCompressor::new(config.clone());
    let cpu_compressed = compressor.compress(&vectors, n_vecs);

    // GPU compression
    let ctx = MetalContext::new().unwrap();
    let pipelines = ShaderPipelines::new(&ctx.device).unwrap();

    let rotation = RotationMatrix::random(dim, config.seed);
    let codebook = LloydMaxCodebook::compute(config.bits, config.lloyd_max_iter).unwrap();
    let qjl = QJLProjector::new(dim, config.qjl_m, config.seed.wrapping_add(1));

    let boundaries_f32: Vec<f32> = codebook.boundaries.iter().map(|&x| x as f32).collect();
    let centroids_f32: Vec<f32> = codebook.centroids.iter().map(|&x| x as f32).collect();

    let kv_buf = ctx.buffer_with_data(&vectors);
    let rot_buf = ctx.buffer_with_data(&rotation.matrix);
    let bound_buf = ctx.buffer_with_data(&boundaries_f32);
    let cent_buf = ctx.buffer_with_data(&centroids_f32);
    let qjl_buf = ctx.buffer_with_data(&qjl.proj_matrix);

    let codes_per_word = 64 / config.bits as usize;
    let n_packed = (dim + codes_per_word - 1) / codes_per_word;
    let n_qjl_packed = (config.qjl_m + 63) / 64;
    let n_levels = 1u32 << config.bits;

    let packed_buf = ctx.buffer_for::<u64>(n_vecs * n_packed);
    let scales_buf = ctx.buffer_for::<f32>(n_vecs);
    let res_norms_buf = ctx.buffer_for::<f32>(n_vecs);
    let qjl_packed_buf = ctx.buffer_for::<u64>(n_vecs * n_qjl_packed);

    kernels::compress::compress_vectors(
        &ctx,
        &pipelines,
        &kv_buf,
        &rot_buf,
        &bound_buf,
        &cent_buf,
        &qjl_buf,
        &packed_buf,
        &scales_buf,
        &res_norms_buf,
        &qjl_packed_buf,
        dim as u32,
        n_vecs as u32,
        config.bits,
        n_levels,
        n_packed as u32,
        config.qjl_m as u32,
        n_qjl_packed as u32,
    )
    .unwrap();

    let gpu_packed: Vec<u64> = ctx.read_buffer(&packed_buf, n_vecs * n_packed);
    let gpu_scales: Vec<f32> = ctx.read_buffer(&scales_buf, n_vecs);
    let gpu_res_norms: Vec<f32> = ctx.read_buffer(&res_norms_buf, n_vecs);
    let gpu_qjl_packed: Vec<u64> = ctx.read_buffer(&qjl_packed_buf, n_vecs * n_qjl_packed);

    // Compare scales
    let mut max_scale_err = 0.0f32;
    for v in 0..n_vecs {
        let err = (cpu_compressed[v].scale - gpu_scales[v]).abs();
        max_scale_err = max_scale_err.max(err);
    }
    eprintln!("max scale error: {max_scale_err:.2e}");
    assert!(
        max_scale_err < 1e-3,
        "scale error {max_scale_err} too large"
    );

    // Compare packed codes (should be identical if quantization matches)
    let mut codes_match = 0usize;
    let mut codes_total = 0usize;
    for v in 0..n_vecs {
        let gpu_slice = &gpu_packed[v * n_packed..(v + 1) * n_packed];
        let cpu_codes = bitpack::unpack_codes(&cpu_compressed[v].stage1_packed, dim, config.bits);
        let gpu_codes = bitpack::unpack_codes(gpu_slice, dim, config.bits);
        for d in 0..dim {
            codes_total += 1;
            if cpu_codes[d] == gpu_codes[d] {
                codes_match += 1;
            }
        }
    }
    let match_rate = codes_match as f64 / codes_total as f64;
    eprintln!(
        "code match rate: {codes_match}/{codes_total} ({:.1}%)",
        match_rate * 100.0
    );
    // Allow small mismatch from f32 vs f64 boundary precision
    assert!(match_rate > 0.95, "code match rate {match_rate:.3} too low");

    // Compare residual norms
    let mut max_rn_err = 0.0f32;
    for v in 0..n_vecs {
        let err = (cpu_compressed[v].residual_norm - gpu_res_norms[v]).abs()
            / (cpu_compressed[v].residual_norm.max(1e-6));
        max_rn_err = max_rn_err.max(err);
    }
    eprintln!("max residual norm relative error: {max_rn_err:.2e}");
    assert!(
        max_rn_err < 0.1,
        "residual norm error {max_rn_err} too large"
    );

    // Compare QJL packed bits
    let mut qjl_match = 0usize;
    let mut qjl_total = 0usize;
    for v in 0..n_vecs {
        let gpu_slice = &gpu_qjl_packed[v * n_qjl_packed..(v + 1) * n_qjl_packed];
        for w in 0..n_qjl_packed {
            let cpu_word = cpu_compressed[v].stage2_bits[w];
            let gpu_word = gpu_slice[w];
            // Count matching bits
            let xor = cpu_word ^ gpu_word;
            let matching = 64 - xor.count_ones() as usize;
            qjl_match += matching;
            qjl_total += 64;
        }
    }
    let qjl_rate = qjl_match as f64 / qjl_total as f64;
    eprintln!(
        "QJL bit match rate: {qjl_match}/{qjl_total} ({:.1}%)",
        qjl_rate * 100.0
    );
    // QJL bits depend on residual which depends on code accuracy, allow more slack
    assert!(qjl_rate > 0.8, "QJL match rate {qjl_rate:.3} too low");
}

/// Test: GPU inner product estimation matches CPU
#[test]
fn attention_scores_match_cpu() {
    let config = make_config(3);
    let dim = config.head_dim;
    let n_vecs = 16;
    let mut rng = StdRng::seed_from_u64(99);
    let vectors = random_vectors(&mut rng, n_vecs, dim);

    // CPU: compress and estimate
    let compressor = TurboQuantCompressor::new(config.clone());
    let cpu_compressed = compressor.compress(&vectors, n_vecs);

    let query: Vec<f32> = (0..dim)
        .map(|_| rng.sample::<f32, _>(StandardNormal))
        .collect();
    let cpu_scores = compressor.estimate_inner_products(&query, &cpu_compressed);

    // True scores for reference
    let true_scores: Vec<f32> = (0..n_vecs)
        .map(|i| {
            let v = &vectors[i * dim..(i + 1) * dim];
            query.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum()
        })
        .collect();

    // GPU: compress
    let ctx = MetalContext::new().unwrap();
    let pipelines = ShaderPipelines::new(&ctx.device).unwrap();

    let rotation = RotationMatrix::random(dim, config.seed);
    let codebook = LloydMaxCodebook::compute(config.bits, config.lloyd_max_iter).unwrap();
    let qjl = QJLProjector::new(dim, config.qjl_m, config.seed.wrapping_add(1));

    let boundaries_f32: Vec<f32> = codebook.boundaries.iter().map(|&x| x as f32).collect();
    let centroids_f32: Vec<f32> = codebook.centroids.iter().map(|&x| x as f32).collect();

    let kv_buf = ctx.buffer_with_data(&vectors);
    let rot_buf = ctx.buffer_with_data(&rotation.matrix);
    let bound_buf = ctx.buffer_with_data(&boundaries_f32);
    let cent_buf = ctx.buffer_with_data(&centroids_f32);
    let qjl_buf = ctx.buffer_with_data(&qjl.proj_matrix);

    let codes_per_word = 64 / config.bits as usize;
    let n_packed = (dim + codes_per_word - 1) / codes_per_word;
    let n_qjl_packed = (config.qjl_m + 63) / 64;
    let n_levels = 1u32 << config.bits;

    let packed_buf = ctx.buffer_for::<u64>(n_vecs * n_packed);
    let scales_buf = ctx.buffer_for::<f32>(n_vecs);
    let res_norms_buf = ctx.buffer_for::<f32>(n_vecs);
    let qjl_packed_buf = ctx.buffer_for::<u64>(n_vecs * n_qjl_packed);

    kernels::compress::compress_vectors(
        &ctx,
        &pipelines,
        &kv_buf,
        &rot_buf,
        &bound_buf,
        &cent_buf,
        &qjl_buf,
        &packed_buf,
        &scales_buf,
        &res_norms_buf,
        &qjl_packed_buf,
        dim as u32,
        n_vecs as u32,
        config.bits,
        n_levels,
        n_packed as u32,
        config.qjl_m as u32,
        n_qjl_packed as u32,
    )
    .unwrap();

    // GPU: compute attention scores
    let rotated_query = rotation.apply(&query);
    let rq_buf = ctx.buffer_with_data(&rotated_query);
    let q_buf = ctx.buffer_with_data(&query);
    let scores_buf = ctx.buffer_for::<f32>(n_vecs);

    kernels::attention::compressed_attention_scores(
        &ctx,
        &pipelines,
        &rq_buf,
        &q_buf,
        &packed_buf,
        &scales_buf,
        &cent_buf,
        &qjl_packed_buf,
        &qjl_buf,
        &res_norms_buf,
        &scores_buf,
        dim as u32,
        n_vecs as u32,
        config.bits,
        n_packed as u32,
        config.qjl_m as u32,
        n_qjl_packed as u32,
        n_levels,
    )
    .unwrap();

    let gpu_scores: Vec<f32> = ctx.read_buffer(&scores_buf, n_vecs);

    // Compare GPU scores with true scores (cosine similarity)
    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb).max(1e-10)
    }

    let gpu_vs_true = cosine_sim(&gpu_scores, &true_scores);
    let cpu_vs_true = cosine_sim(&cpu_scores, &true_scores);
    let gpu_vs_cpu = cosine_sim(&gpu_scores, &cpu_scores);

    eprintln!("GPU vs true: {gpu_vs_true:.4}");
    eprintln!("CPU vs true: {cpu_vs_true:.4}");
    eprintln!("GPU vs CPU:  {gpu_vs_cpu:.4}");

    assert!(
        gpu_vs_true > 0.95,
        "GPU scores vs true cosine {gpu_vs_true} too low"
    );
    assert!(gpu_vs_cpu > 0.95, "GPU vs CPU cosine {gpu_vs_cpu} too low");
}

/// Test: Value gather produces correct weighted sum
#[test]
fn value_gather_matches_cpu() {
    let config = make_config(3);
    let dim = config.head_dim;
    let n_vecs = 8;
    let mut rng = StdRng::seed_from_u64(99);
    let vectors = random_vectors(&mut rng, n_vecs, dim);

    // Create some attention weights (softmax-like)
    let raw_weights: Vec<f32> = (0..n_vecs).map(|_| rng.random::<f32>()).collect();
    let sum: f32 = raw_weights.iter().sum();
    let weights: Vec<f32> = raw_weights.iter().map(|&w| w / sum).collect();

    // CPU: compress values and compute weighted sum via full reconstruction
    let compressor = TurboQuantCompressor::new(config.clone());
    let cpu_compressed = compressor.compress(&vectors, n_vecs);

    // CPU weighted sum via reconstruction
    let rotation = RotationMatrix::random(dim, config.seed);
    let codebook = LloydMaxCodebook::compute(config.bits, config.lloyd_max_iter).unwrap();
    let mut cpu_output = vec![0.0f32; dim];
    for (i, cv) in cpu_compressed.iter().enumerate() {
        let codes = bitpack::unpack_codes(&cv.stage1_packed, dim, config.bits);
        let deq = codebook.dequantize_vec(&codes);
        let denorm: Vec<f32> = deq.iter().map(|&x| x * cv.scale).collect();
        let recon = rotation.apply_inverse(&denorm);
        for d in 0..dim {
            cpu_output[d] += weights[i] * recon[d];
        }
    }

    // GPU: compress then gather
    let ctx = MetalContext::new().unwrap();
    let pipelines = ShaderPipelines::new(&ctx.device).unwrap();

    let boundaries_f32: Vec<f32> = codebook.boundaries.iter().map(|&x| x as f32).collect();
    let centroids_f32: Vec<f32> = codebook.centroids.iter().map(|&x| x as f32).collect();
    let qjl = QJLProjector::new(dim, config.qjl_m, config.seed.wrapping_add(1));

    let kv_buf = ctx.buffer_with_data(&vectors);
    let rot_buf = ctx.buffer_with_data(&rotation.matrix);
    let bound_buf = ctx.buffer_with_data(&boundaries_f32);
    let cent_buf = ctx.buffer_with_data(&centroids_f32);
    let qjl_buf = ctx.buffer_with_data(&qjl.proj_matrix);

    let codes_per_word = 64 / config.bits as usize;
    let n_packed = (dim + codes_per_word - 1) / codes_per_word;
    let n_qjl_packed = (config.qjl_m + 63) / 64;
    let n_levels = 1u32 << config.bits;

    let packed_buf = ctx.buffer_for::<u64>(n_vecs * n_packed);
    let scales_buf = ctx.buffer_for::<f32>(n_vecs);
    let res_norms_buf = ctx.buffer_for::<f32>(n_vecs);
    let qjl_packed_buf = ctx.buffer_for::<u64>(n_vecs * n_qjl_packed);

    kernels::compress::compress_vectors(
        &ctx,
        &pipelines,
        &kv_buf,
        &rot_buf,
        &bound_buf,
        &cent_buf,
        &qjl_buf,
        &packed_buf,
        &scales_buf,
        &res_norms_buf,
        &qjl_packed_buf,
        dim as u32,
        n_vecs as u32,
        config.bits,
        n_levels,
        n_packed as u32,
        config.qjl_m as u32,
        n_qjl_packed as u32,
    )
    .unwrap();

    // Gather
    let weights_buf = ctx.buffer_with_data(&weights);
    let output_buf = ctx.buffer_for::<f32>(dim);

    kernels::attention::compressed_value_gather(
        &ctx,
        &pipelines,
        &weights_buf,
        &packed_buf,
        &scales_buf,
        &cent_buf,
        &rot_buf,
        &output_buf,
        dim as u32,
        n_vecs as u32,
        config.bits,
        n_packed as u32,
        n_levels,
    )
    .unwrap();

    let gpu_output: Vec<f32> = ctx.read_buffer(&output_buf, dim);

    // Compare with CPU
    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb).max(1e-10)
    }

    let cos = cosine_sim(&cpu_output, &gpu_output);
    eprintln!("value_gather GPU vs CPU cosine: {cos:.4}");

    // Also compare with true (uncompressed) weighted sum
    let mut true_output = vec![0.0f32; dim];
    for i in 0..n_vecs {
        for d in 0..dim {
            true_output[d] += weights[i] * vectors[i * dim + d];
        }
    }
    let gpu_vs_true = cosine_sim(&gpu_output, &true_output);
    eprintln!("value_gather GPU vs true: {gpu_vs_true:.4}");

    assert!(cos > 0.95, "GPU vs CPU cosine {cos} too low");
    assert!(
        gpu_vs_true > 0.95,
        "GPU vs true cosine {gpu_vs_true} too low"
    );
}

fn set_u32(encoder: &metal::ComputeCommandEncoderRef, index: usize, value: u32) {
    let bytes = value.to_ne_bytes();
    encoder.set_bytes_directly(
        index,
        std::mem::size_of::<u32>(),
        bytes.as_ptr() as *const _,
    );
}
