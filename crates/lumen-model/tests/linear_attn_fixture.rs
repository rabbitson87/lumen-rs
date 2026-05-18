//! Numerical parity test for `GatedDeltaNet::forward` against the MLX reference fixture.
//!
//! What we're proving: our Rust port of `mlx_lm.models.qwen3_5.GatedDeltaNet` (Qwen3-Next
//! gated-delta SSM with separate qkv / z / b / a input projections) produces the same
//! block output MLX produced for layer 0 of `mlx-community/Qwen3.6-35B-A3B-mxfp4` when
//! fed the same post-`input_layernorm` input.
//!
//! ## Fixture files
//!
//! - `layer0_linear_attn.safetensors` — golden `(x_post_ln, y)` pair from
//!   `scripts/generate_qwen3_5_moe_fixtures.py` (checked in, ~100 KB).
//! - `layer0_linear_attn_weights.safetensors` — dequantized fp32 projections/state for
//!   layer 0, written by `scripts/dump_qwen3_5_moe_layer_weights.py` (~130 MB; **not**
//!   checked in; test skips when absent).
//!
//! ## Tolerance
//!
//! MLX runs the full block in bf16 including the SSM sequential loop, which on layer 0
//! decays state aggressively (reference output std 0.024 per the fixture memo). The
//! bf16↔f32 gap on pure-attention blocks is ~5.5e-3 relative L2; the SSM path adds a
//! `softplus` + `exp` + `exp` stack whose f32 ↔ bf16 cast boundaries can drift a few more
//! ULPs. Since the recurrent loop has **no discrete branching** (unlike MoE top-k), we
//! don't need a dual max_abs/rel_L2 gate — a single `rel_L2 < 1e-2` should pass if the
//! forward is faithful. If it doesn't, `tests/linear_attn_trace.rs` localizes the stage.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor, safetensors as cst};
use candle_nn::Linear;
use lumen_model::qwen3_5_moe::linear_attn::{
    GatedDeltaNet, GatedDeltaNetRuntime, LinearAttnDims, conv1d_from_mlx_weight,
};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn take_f32(map: &HashMap<String, Tensor>, name: &str) -> Tensor {
    let t = map
        .get(name)
        .unwrap_or_else(|| panic!("missing tensor `{name}` in fixture"));
    assert_eq!(
        t.dtype(),
        DType::F32,
        "expected `{name}` to be f32 but got {:?}",
        t.dtype()
    );
    t.clone()
}

fn skip_if_missing(path: &Path) -> bool {
    if !path.exists() {
        eprintln!(
            "skipping: {} not found. Regenerate with\n    \
             python scripts/dump_qwen3_5_moe_layer_weights.py --layer 0 --block linear_attn",
            path.display()
        );
        return true;
    }
    false
}

fn layer0_dims() -> LinearAttnDims {
    LinearAttnDims {
        hidden_size: 2048,
        num_k_heads: 16,
        num_v_heads: 32,
        head_dim: 128,
        conv_kernel: 4,
    }
}

fn layer0_runtime() -> GatedDeltaNetRuntime {
    GatedDeltaNetRuntime {
        dims: layer0_dims(),
        rms_norm_eps: 1e-6,
    }
}

fn relative_l2(a: &Tensor, b: &Tensor) -> f32 {
    let diff = (a - b).unwrap();
    let num = diff
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
        .sqrt();
    let den = b
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
        .sqrt()
        .max(1e-12);
    num / den
}

#[test]
fn linear_attn_forward_matches_mlx_fixture_layer0() {
    let dir = fixtures_dir();
    let weights_path = dir.join("layer0_linear_attn_weights.safetensors");
    if skip_if_missing(&weights_path) {
        return;
    }
    let input_path = dir.join("layer0_linear_attn.safetensors");

    // CPU only: the SSM loop is a sequential reduction that has no Metal implementation
    // in Candle; pushing it to GPU for parity would just add another source of rounding.
    let device = Device::Cpu;
    let weights = cst::load(&weights_path, &device).unwrap();
    let fixture = cst::load(&input_path, &device).unwrap();

    let dims = layer0_dims();
    let rt = layer0_runtime();

    let x_post_ln = take_f32(&fixture, "x_post_ln"); // [S, hidden]
    let y_ref = take_f32(&fixture, "y");
    let x = x_post_ln.unsqueeze(0).unwrap();
    let y_ref = y_ref.unsqueeze(0).unwrap();

    let in_proj_qkv = Linear::new(take_f32(&weights, "in_proj_qkv.weight"), None);
    let in_proj_z = Linear::new(take_f32(&weights, "in_proj_z.weight"), None);
    let in_proj_b = Linear::new(take_f32(&weights, "in_proj_b.weight"), None);
    let in_proj_a = Linear::new(take_f32(&weights, "in_proj_a.weight"), None);
    let conv1d =
        conv1d_from_mlx_weight(take_f32(&weights, "conv1d.weight"), dims.conv_kernel).unwrap();
    let a_log = take_f32(&weights, "A_log");
    let dt_bias = take_f32(&weights, "dt_bias");
    let norm_w = take_f32(&weights, "norm.weight");
    let out_proj = Linear::new(take_f32(&weights, "out_proj.weight"), None);

    // Sanity-check shapes from the config against the dumped tensors so any drift in
    // dequantization surfaces here instead of deep inside the forward.
    let shapes = dims.shapes();
    assert_eq!(in_proj_qkv.weight().dims(), shapes.in_proj_qkv.as_slice());
    assert_eq!(in_proj_z.weight().dims(), shapes.in_proj_z.as_slice());
    assert_eq!(in_proj_b.weight().dims(), shapes.in_proj_b.as_slice());
    assert_eq!(in_proj_a.weight().dims(), shapes.in_proj_a.as_slice());
    assert_eq!(a_log.dims(), shapes.a_log.as_slice());
    assert_eq!(dt_bias.dims(), shapes.dt_bias.as_slice());
    assert_eq!(norm_w.dims(), shapes.norm.as_slice());
    assert_eq!(out_proj.weight().dims(), shapes.out_proj.as_slice());

    let in_proj_combined = Linear::new(
        Tensor::cat(
            &[
                in_proj_qkv.weight(),
                in_proj_z.weight(),
                in_proj_b.weight(),
                in_proj_a.weight(),
            ],
            0,
        )
        .unwrap(),
        None,
    );

    let mut net = GatedDeltaNet::new(
        rt,
        in_proj_combined.into(),
        conv1d,
        a_log,
        dt_bias,
        norm_w,
        out_proj.into(),
    );

    let y = net.forward(&x, None).unwrap();
    assert_eq!(y.dims(), y_ref.dims());

    let err = relative_l2(&y, &y_ref);
    let max_diff = (&y - &y_ref)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let y_sum = y.sum_all().unwrap().to_scalar::<f32>().unwrap();
    let y_ref_sum = y_ref.sum_all().unwrap().to_scalar::<f32>().unwrap();
    println!(
        "linear_attn layer0 parity: rel_L2 = {err:.4e}, max_abs = {max_diff:.4e}, \
         y_sum = {y_sum:.4} (ref {y_ref_sum:.4})"
    );

    assert!(
        err < 1e-2,
        "relative L2 error {err:.4e} exceeds 1e-2 bound; pure bf16↔f32 gap on this block \
         should stay below the self_attn 5.5e-3 floor by about 2× (extra softplus/exp stack)."
    );
}
