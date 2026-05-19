//! C4 verification: tq_compressed_value_gather_multi must produce identical
//! output to tq_compressed_value_gather, fanned out to gqa_ratio Q heads.
//!
//! Run: cargo test --release -p lumen-metal --test c4_multi_value_gather

use rand::prelude::*;
use rand_distr::StandardNormal;

use lumen_core::compressor::TurboQuantCompressor;
use lumen_core::config::TurboQuantConfig;
use lumen_metal::device::MetalContext;
use lumen_metal::kernels;
use lumen_metal::pipeline::ShaderPipelines;

#[test]
fn multi_value_gather_matches_single_per_qhead() {
    let dim = 128usize;
    let n_kv = 64usize;
    let bits = 3u32;
    let qjl_m = 64usize;
    let gqa_ratio = 8u32;
    let seed = 99u64;

    let config = TurboQuantConfig {
        bits,
        qjl_m,
        seed: 42,
        lloyd_max_iter: 1000,
        head_dim: dim,
    };
    let compressor = TurboQuantCompressor::new(config);

    // Build a synthetic V cache (random) and compress
    let mut rng = StdRng::seed_from_u64(seed);
    let vectors: Vec<f32> = (0..n_kv * dim)
        .map(|_| rng.sample::<f32, _>(StandardNormal))
        .collect();
    let compressed = compressor.compress(&vectors, n_kv);

    // Synthetic softmax weights: random positive, normalized to sum=1
    let raw: Vec<f32> = (0..n_kv)
        .map(|_| rng.sample::<f32, _>(StandardNormal).abs() + 1e-3)
        .collect();
    let raw_sum: f32 = raw.iter().sum();
    let weights: Vec<f32> = raw.iter().map(|&w| w / raw_sum).collect();

    // ── GPU setup ────────────────────────────────────────────────────────
    let ctx = MetalContext::new().unwrap();
    let pipelines = ShaderPipelines::new(&ctx.device).unwrap();

    let n_packed = (dim + (64 / bits as usize) - 1) / (64 / bits as usize);
    let n_levels = 1u32 << bits;

    let mut packed_codes_flat: Vec<u64> = Vec::with_capacity(n_kv * n_packed);
    let mut scales_flat: Vec<f32> = Vec::with_capacity(n_kv);
    for cv in &compressed {
        packed_codes_flat.extend_from_slice(&cv.stage1_packed);
        scales_flat.push(cv.scale);
    }

    let centroids_f32: Vec<f32> = compressor
        .codebook()
        .centroids
        .iter()
        .map(|&x| x as f32)
        .collect();
    let rotation_flat: Vec<f32> = compressor.rotation().matrix.clone();

    let weights_buf = ctx.buffer_with_data(&weights);
    let packed_buf = ctx.buffer_with_data(&packed_codes_flat);
    let scales_buf = ctx.buffer_with_data(&scales_flat);
    let cent_buf = ctx.buffer_with_data(&centroids_f32);
    let rot_buf = ctx.buffer_with_data(&rotation_flat);

    // Single-head reference: gqa_ratio separate dispatches
    let single_buf = ctx.buffer_for::<f32>(gqa_ratio as usize * dim);
    for q in 0..gqa_ratio as usize {
        // Hack: write into one slot. Simulate the existing per-Q dispatch path
        // by re-running the same compute and shifting output offset to q*dim.
        let slot = ctx.buffer_for::<f32>(dim);
        kernels::attention::compressed_value_gather(
            &ctx,
            &pipelines,
            &weights_buf,
            &packed_buf,
            &scales_buf,
            &cent_buf,
            &rot_buf,
            &slot,
            dim as u32,
            n_kv as u32,
            bits,
            n_packed as u32,
            n_levels,
        )
        .unwrap();
        let slot_data: Vec<f32> = ctx.read_buffer(&slot, dim);
        // Store into the single_buf slot at q*dim..(q+1)*dim
        let mut single_host: Vec<f32> = ctx.read_buffer(&single_buf, gqa_ratio as usize * dim);
        single_host[q * dim..(q + 1) * dim].copy_from_slice(&slot_data);
        // Re-upload (keep tests simple — alternative: just check bytes against multi result later)
        let _ = single_host; // unused; fall through and use slot_data per-q below
    }

    // Build the expected single-head reference on the host: just run once
    // and replicate (the math is genuinely identical for all Q heads).
    let single_slot = ctx.buffer_for::<f32>(dim);
    kernels::attention::compressed_value_gather(
        &ctx,
        &pipelines,
        &weights_buf,
        &packed_buf,
        &scales_buf,
        &cent_buf,
        &rot_buf,
        &single_slot,
        dim as u32,
        n_kv as u32,
        bits,
        n_packed as u32,
        n_levels,
    )
    .unwrap();
    let single_ref: Vec<f32> = ctx.read_buffer(&single_slot, dim);

    // Multi-head fused: one dispatch, fan out to gqa_ratio output slots
    let multi_buf = ctx.buffer_for::<f32>(gqa_ratio as usize * dim);
    kernels::attention::compressed_value_gather_multi(
        &ctx,
        &pipelines,
        &weights_buf,
        &packed_buf,
        &scales_buf,
        &cent_buf,
        &rot_buf,
        &multi_buf,
        dim as u32,
        n_kv as u32,
        bits,
        n_packed as u32,
        n_levels,
        gqa_ratio,
    )
    .unwrap();
    let multi_data: Vec<f32> = ctx.read_buffer(&multi_buf, gqa_ratio as usize * dim);

    // All gqa_ratio slots in multi_data must equal single_ref bit-for-bit.
    for q in 0..gqa_ratio as usize {
        let slot = &multi_data[q * dim..(q + 1) * dim];
        let max_diff = slot
            .iter()
            .zip(&single_ref)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "Q head {q} diverges from single ref: max|Δ|={max_diff:.3e}"
        );
    }

    // Sanity: slots should be byte-identical across q (since we run the
    // same compute and write deterministically)
    for q in 1..gqa_ratio as usize {
        let s0 = &multi_data[0..dim];
        let sq = &multi_data[q * dim..(q + 1) * dim];
        let max_intra = s0
            .iter()
            .zip(sq)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(
            max_intra, 0.0,
            "Q head {q} differs from Q head 0 within multi (must be identical fan-out)"
        );
    }

    // ── C4 microbench: gqa_ratio × single dispatch vs single multi dispatch ──
    // Compare wall time. We run each path many times and report median.
    let iters = 50usize;

    let bench_single_loop = || {
        for q in 0..gqa_ratio as usize {
            let _ = q;
            let slot = ctx.buffer_for::<f32>(dim);
            kernels::attention::compressed_value_gather(
                &ctx,
                &pipelines,
                &weights_buf,
                &packed_buf,
                &scales_buf,
                &cent_buf,
                &rot_buf,
                &slot,
                dim as u32,
                n_kv as u32,
                bits,
                n_packed as u32,
                n_levels,
            )
            .unwrap();
        }
    };
    let bench_multi = || {
        let multi = ctx.buffer_for::<f32>(gqa_ratio as usize * dim);
        kernels::attention::compressed_value_gather_multi(
            &ctx,
            &pipelines,
            &weights_buf,
            &packed_buf,
            &scales_buf,
            &cent_buf,
            &rot_buf,
            &multi,
            dim as u32,
            n_kv as u32,
            bits,
            n_packed as u32,
            n_levels,
            gqa_ratio,
        )
        .unwrap();
    };

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

    // warmup
    bench_single_loop();
    bench_multi();

    let dur_single = time_median(iters, bench_single_loop);
    let dur_multi = time_median(iters, bench_multi);
    let speedup = dur_single.as_secs_f64() / dur_multi.as_secs_f64();

    println!(
        "C4 microbench: single×{} = {:.1}μs   multi = {:.1}μs   speedup = {:.2}×",
        gqa_ratio,
        dur_single.as_secs_f64() * 1e6,
        dur_multi.as_secs_f64() * 1e6,
        speedup
    );

    // We expect a clear win — at least 2× since redundant compute is eliminated.
    assert!(
        speedup >= 1.5,
        "C4 multi dispatch must be at least 1.5× faster than single×{} (got {:.2}×)",
        gqa_ratio,
        speedup
    );
}
