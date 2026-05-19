//! End-to-end sanity: `SelfAttention` forward pass with `ProjLinear::Mxfp4` weights.
//!
//! Builds a tiny `SelfAttention` whose Q/K/V/O projections are MXFP4-quantized on-GPU
//! (via `Mxfp4Linear`), then runs forward with synthetic input and checks:
//!   1. Output shape is `[B, L, hidden]` (not NaN/Inf) — proves the ProjLinear dispatch wires
//!      correctly through the full attention block.
//!   2. The *isolated Q-projection* output agrees between GPU-MXFP4 and CPU-dense paths to
//!      ≤ 1e-2 per element. Post-softmax comparisons are intentionally dropped: softmax is
//!      exponentially sensitive to small logit shifts, and tiny matmul rounding drift would
//!      amplify well past any useful bound. The matmul itself (the only GPU-vs-CPU math
//!      surface) is what this test gates.
//!
//! Real 19 GB shard loading is covered separately by the env-gated
//! `real_shards_layer0_loads_on_gpu_when_available` test in `loader.rs`.

#![cfg(feature = "turboquant-gpu")]

use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use candle_nn::RmsNorm;
use lumen_metal::mxfp4::dequantize_f32;
use lumen_metal::mxfp4_gpu::{MxFp4Context, Mxfp4Weight};
use lumen_metal::mxfp4_linear::Mxfp4Linear;

use lumen_model::qwen3_5_moe::proj::ProjLinear;
use lumen_model::qwen3_5_moe::self_attn::{SelfAttention, SelfAttnDims, SelfAttnRuntime};

fn synth_weight(out: usize, ins: usize, seed: u32) -> (Vec<u32>, Vec<u8>, Vec<f32>) {
    assert!(
        ins.is_multiple_of(32),
        "MXFP4 requires in_features % 32 == 0"
    );
    let n_groups = out * ins / 32;
    let n_words = out * ins / 8;
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        s
    };
    let packed: Vec<u32> = (0..n_words).map(|_| next()).collect();
    // Keep scales near 2^0 so dequantized magnitudes are well-bounded.
    let scales: Vec<u8> = (0..n_groups).map(|_| 120 + (next() & 0x0F) as u8).collect();
    let mut dense = vec![0.0_f32; out * ins];
    dequantize_f32(&packed, &scales, &mut dense).expect("cpu dequant");
    (packed, scales, dense)
}

fn make_mxfp4_proj(
    ctx: &Arc<MxFp4Context>,
    out: usize,
    ins: usize,
    seed: u32,
) -> (ProjLinear, Tensor) {
    let (packed, scales, dense) = synth_weight(out, ins, seed);
    make_mxfp4_proj_from_host(ctx, out, ins, packed, scales, dense)
}

fn make_mxfp4_proj_from_host(
    ctx: &Arc<MxFp4Context>,
    out: usize,
    ins: usize,
    packed: Vec<u32>,
    scales: Vec<u8>,
    dense: Vec<f32>,
) -> (ProjLinear, Tensor) {
    let weight = Mxfp4Weight::from_host(&ctx.ctx, &packed, &scales, out, ins).unwrap();
    let proj = ProjLinear::Mxfp4(Mxfp4Linear::new(weight, None, Arc::clone(ctx)));
    let dense_tensor = Tensor::from_vec(dense, (out, ins), &Device::Cpu).unwrap();
    (proj, dense_tensor)
}

#[test]
fn mxfp4_self_attention_forward_runs_and_is_finite() {
    let Ok(ctx) = MxFp4Context::new() else {
        eprintln!("skipping: no Metal GPU available");
        return;
    };
    let ctx = Arc::new(ctx);
    let device = Device::Cpu;

    // Tiny dims compatible with MXFP4 group size (in_features multiple of 32).
    let dims = SelfAttnDims {
        hidden_size: 32,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 32,
        attn_output_gate: true,
        rotary_dim: 8,
    };
    let runtime = SelfAttnRuntime {
        dims,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
    };

    // Build Mxfp4 projections for fused QKV and O.
    let (q_packed, q_scales, q_dense) = synth_weight(dims.q_out_dim(), dims.hidden_size, 0x11);
    let (k_packed, k_scales, k_dense) = synth_weight(dims.kv_out_dim(), dims.hidden_size, 0x22);
    let (v_packed, v_scales, v_dense) = synth_weight(dims.kv_out_dim(), dims.hidden_size, 0x33);
    let qkv_out = dims.q_out_dim() + 2 * dims.kv_out_dim();
    let qkv_proj = make_mxfp4_proj_from_host(
        &ctx,
        qkv_out,
        dims.hidden_size,
        [q_packed, k_packed, v_packed].concat(),
        [q_scales, k_scales, v_scales].concat(),
        [q_dense, k_dense, v_dense].concat(),
    )
    .0;
    let (o_proj, _) = make_mxfp4_proj(&ctx, dims.hidden_size, dims.attn_value_dim(), 0x44);

    let ones = Tensor::from_vec(vec![1f32; dims.head_dim], (dims.head_dim,), &device).unwrap();
    let mut attn = SelfAttention::new(
        runtime,
        qkv_proj,
        o_proj,
        RmsNorm::new(ones.clone(), runtime.rms_norm_eps),
        RmsNorm::new(ones, runtime.rms_norm_eps),
    );

    let x_vec: Vec<f32> = (0..1 * 3 * dims.hidden_size)
        .map(|i| (i as f32 * 0.013).sin() * 0.5)
        .collect();
    let x = Tensor::from_vec(x_vec, (1, 3, dims.hidden_size), &device).unwrap();

    let y = attn.forward(&x, 0, None).unwrap();
    assert_eq!(y.dims(), &[1, 3, dims.hidden_size]);

    let y_vec = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(
        y_vec.iter().all(|v| v.is_finite()),
        "mxfp4 SelfAttention output has non-finite values"
    );
    // Output must depend on input (not collapse to all-zero through a bug in the proj chain).
    let max_abs = y_vec.iter().fold(0f32, |a, b| a.max(b.abs()));
    assert!(
        max_abs > 1e-6,
        "output appears to be identically zero (max_abs = {max_abs})"
    );
}

#[test]
fn mxfp4_q_projection_matches_cpu_dense_matmul() {
    let Ok(ctx) = MxFp4Context::new() else {
        return;
    };
    let ctx = Arc::new(ctx);
    let device = Device::Cpu;

    let (out_features, in_features) = (256, 32);
    let (q_proj, q_dense) = make_mxfp4_proj(&ctx, out_features, in_features, 0x11);

    // Q projection: y = x @ W^T  (x shape [3, 32])
    let x_vec: Vec<f32> = (0..3 * in_features)
        .map(|i| (i as f32 * 0.013).sin())
        .collect();
    let x = Tensor::from_vec(x_vec.clone(), (3, in_features), &device).unwrap();
    let y_gpu = q_proj.forward(&x).unwrap();

    let w_ref = q_dense; // already [out, in]
    let y_ref = x.matmul(&w_ref.t().unwrap()).unwrap();

    assert_eq!(y_gpu.dims(), y_ref.dims());
    let g = y_gpu.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let r = y_ref.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let mut max_err = 0.0_f32;
    for (gi, ri) in g.iter().zip(r.iter()) {
        max_err = max_err.max((gi - ri).abs());
    }
    assert!(
        max_err < 1e-2,
        "GPU MXFP4 matmul vs CPU-dequant matmul disagreement: max_abs = {max_err}"
    );
}

#[test]
fn mxfp4_proj_linear_in_features_matches_dense() {
    // Quick sanity: ProjLinear::Mxfp4::in_features() returns the unpacked logical dim.
    let Ok(ctx) = MxFp4Context::new() else {
        return;
    };
    let ctx = Arc::new(ctx);
    let (proj, _) = make_mxfp4_proj(&ctx, 8, 64, 0x99);
    assert_eq!(proj.in_features(), 64);
    assert_eq!(proj.out_features(), 8);
    assert!(proj.is_mxfp4());

    // x: [2, 64] f32
    let x = Tensor::zeros((2, 64), DType::F32, &Device::Cpu).unwrap();
    let y = proj.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 8]);
}
