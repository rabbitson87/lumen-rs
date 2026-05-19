//! Standalone microbench for `tq_flash_attn_bf16` vs `tq_flash_attn` (f32).
//!
//! Run with:
//!   LUMEN_FA_BF16_BENCH=1 cargo test --test flash_attn_bf16_microbench \
//!     -p lumen-metal --features model-integration --release -- --nocapture
//!
//! Decode shape (Qwen3.6-27B Dense self_attn): B=1, H=16, Sq=1, D=256,
//! H_kv=2, Skv ∈ {256, 1024}.
//!
//! Reports per-shape mean / median / Welch's t σ between f32 and bf16. The
//! σ ≥ +1 gate is the standalone WIN bar — bf16 kernel must be at least
//! marginally faster at kernel level for downstream B.3 integration to be
//! net-positive. σ < 0 = standalone NEGATIVE; B.3 can still recover it via
//! cast removal upstream, but kernel-level NEGATIVE is a red flag.

#![cfg(feature = "model-integration")]

use candle_core::backend::BackendDevice;
use candle_core::{DType, Device, Tensor};
use lumen_metal::flash_attn::{flash_attn_candle, set_disabled};
use std::time::Instant;

fn metal_device() -> Option<Device> {
    Device::new_metal(0).ok()
}

fn random_tensor(shape: &[usize], seed: u64, dev: &Device) -> Tensor {
    let n: usize = shape.iter().product();
    let mut state = seed
        .wrapping_mul(2862933555777941757)
        .wrapping_add(3037000493);
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = ((state >> 32) as u32) as f32 / u32::MAX as f32;
        data.push(bits - 0.5);
    }
    Tensor::from_vec(data, shape, dev).unwrap()
}

fn welchs_t(a: &[f64], b: &[f64]) -> f64 {
    let na = a.len() as f64;
    let nb = b.len() as f64;
    let ma = a.iter().sum::<f64>() / na;
    let mb = b.iter().sum::<f64>() / nb;
    let va = a.iter().map(|x| (x - ma).powi(2)).sum::<f64>() / (na - 1.0);
    let vb = b.iter().map(|x| (x - mb).powi(2)).sum::<f64>() / (nb - 1.0);
    let se = (va / na + vb / nb).sqrt();
    if se == 0.0 { 0.0 } else { (ma - mb) / se }
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|x, y| x.partial_cmp(y).unwrap());
    s[s.len() / 2]
}

fn bench_shape(label: &str, h: usize, h_kv: usize, sq: usize, skv: usize, iters: usize) {
    let dev = metal_device().expect("Metal device");
    let d = 256;
    let scale = (d as f64).powf(-0.5) as f32;

    let q_f32 = random_tensor(&[1, h, sq, d], 11, &dev);
    let k_f32 = random_tensor(&[1, h_kv, skv, d], 22, &dev);
    let v_f32 = random_tensor(&[1, h_kv, skv, d], 33, &dev);

    let q_b16 = q_f32.to_dtype(DType::BF16).unwrap().contiguous().unwrap();
    let k_b16 = k_f32.to_dtype(DType::BF16).unwrap().contiguous().unwrap();
    let v_b16 = v_f32.to_dtype(DType::BF16).unwrap().contiguous().unwrap();

    set_disabled(false);

    // Warm-up — compile pipelines, prime caches.
    for _ in 0..20 {
        let _ = flash_attn_candle(&q_f32, &k_f32, &v_f32, None, scale)
            .unwrap()
            .unwrap();
        let _ = flash_attn_candle(&q_b16, &k_b16, &v_b16, None, scale)
            .unwrap()
            .unwrap();
    }
    if let Device::Metal(md) = &dev {
        let _ = md.synchronize();
    }

    // Interleaved measurement to reduce thermal/scheduler bias.
    let mut t_f32: Vec<f64> = Vec::with_capacity(iters);
    let mut t_b16: Vec<f64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        let out = flash_attn_candle(&q_f32, &k_f32, &v_f32, None, scale)
            .unwrap()
            .unwrap();
        if let Device::Metal(md) = out.device() {
            let _ = md.synchronize();
        }
        t_f32.push(t0.elapsed().as_secs_f64() * 1e6); // µs

        let t1 = Instant::now();
        let out = flash_attn_candle(&q_b16, &k_b16, &v_b16, None, scale)
            .unwrap()
            .unwrap();
        if let Device::Metal(md) = out.device() {
            let _ = md.synchronize();
        }
        t_b16.push(t1.elapsed().as_secs_f64() * 1e6);
    }

    let mean_f32 = t_f32.iter().sum::<f64>() / iters as f64;
    let mean_b16 = t_b16.iter().sum::<f64>() / iters as f64;
    let med_f32 = median(&t_f32);
    let med_b16 = median(&t_b16);

    // σ measured as: how many standard errors is bf16 *faster* than f32.
    // Sign convention: positive σ = bf16 wins.
    let sigma = welchs_t(&t_f32, &t_b16);

    let pct = (med_b16 - med_f32) / med_f32 * 100.0;
    eprintln!(
        "  {label} (h={h} h_kv={h_kv} sq={sq} skv={skv}, n={iters}/variant)\n    \
         f32: mean={mean_f32:.2}µs median={med_f32:.2}µs\n    \
         bf16: mean={mean_b16:.2}µs median={med_b16:.2}µs\n    \
         Δ_median: {pct:+.2}%   Welch's σ (positive=bf16 win): {sigma:+.2}",
    );
}

#[test]
fn microbench_decode_and_prefill() {
    if std::env::var("LUMEN_FA_BF16_BENCH").as_deref() != Ok("1") {
        eprintln!("skip: set LUMEN_FA_BF16_BENCH=1 to run microbench");
        return;
    }

    eprintln!("\n=== flash_attn bf16 vs f32 microbench (Qwen3.6-27B Dense self_attn shapes) ===");

    // Decode (single-token) — most relevant for production decode tok/s.
    bench_shape("decode_skv256", 16, 2, 1, 256, 200);
    bench_shape("decode_skv1024", 16, 2, 1, 1024, 200);
    // Longer KV — measure scaling.
    bench_shape("decode_skv4096", 16, 2, 1, 4096, 100);
    // Prefill (multi-token) — less common path but validates kernel scaling.
    bench_shape("prefill_sq8_skv64", 16, 2, 8, 64, 200);
    bench_shape("prefill_sq16_skv16", 16, 2, 16, 16, 200);
}
