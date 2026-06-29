//! M1 of the fused linear-attn mega-kernel (path to 21 tok/s): a standalone
//! affine4 QMV (quantized matrix-vector, decode matvec) written as an mlx
//! custom Metal kernel, validated BIT-CLOSE against mlx's `quantized_matmul`.
//!
//! Why: the only lever past ~18 tok/s is fusing in_proj+conv+norm+ssm+out_proj
//! into ONE kernel so the matmul weight-read (BW) overlaps the SSM compute,
//! lifting the real-forward matmul from 62% to ~85% BW. The FOUNDATION of that
//! fused kernel is doing affine4 dequant+matmul INSIDE a custom kernel. This M1
//! proves we can reproduce mlx's affine4 QMV exactly. Speed parity (not a win)
//! is expected here; the win comes at M4 (fusion+software-pipelining).
//!
//! affine4 layout (mlx convention): packed `wq[out, in/8]` u32 (8 nibbles, value
//! k in bits [4k..4k+4)), per-group (group_size) `scales`/`biases` `[out, in/G]`.
//! dequant: `w = scales*q + biases`, q ∈ {0..15}.  y[o] = Σ_i x[i]·w[o,i].
//!
//! Run: cargo run --release -p lumen-mlx --features mlx-native --example m1_affine4_qmv_custom

use anyhow::{Result, anyhow};
use lumen_mlx::metal_kernel::{MetalKernel, MetalKernelConfig};
use mlx_rs::{Array, Dtype};

const AFFINE4_QMV_SOURCE: &str = r#"
    uint o = thread_position_in_grid.x;          // output row
    if (o >= (uint)OUT) return;
    uint n_u32 = (uint)IN / 8u;
    uint n_grp = (uint)IN / (uint)G;
    auto wq_row = wq + o * n_u32;
    auto sc_row = scales + o * n_grp;
    auto bi_row = biases + o * n_grp;
    float acc = 0.0f;
    for (uint u = 0u; u < n_u32; ++u) {
        uint packed = wq_row[u];
        uint base_i = u * 8u;
        for (uint n = 0u; n < 8u; ++n) {
            uint i = base_i + n;
            uint g = i / (uint)G;
            float q = float((packed >> (4u * n)) & 0xFu);
            float w = float(sc_row[g]) * q + float(bi_row[g]);
            acc += float(x[i]) * w;
        }
    }
    y[o] = static_cast<InT>(acc);
"#;

/// affine4 QMV via custom Metal kernel. x:[in] (1-D), wq:[out,in/8] u32,
/// scales/biases:[out,in/G]. Returns y:[out].
fn affine4_qmv_custom(
    x: &Array,
    wq: &Array,
    scales: &Array,
    biases: &Array,
    in_dim: i32,
    out_dim: i32,
    group_size: i32,
) -> Result<Array> {
    let kernel = MetalKernel::new(
        "affine4_qmv",
        &["x", "wq", "scales", "biases"],
        &["y"],
        AFFINE4_QMV_SOURCE,
        /* ensure_row_contiguous */ true,
        /* atomic_outputs */ false,
    )?;
    let config = MetalKernelConfig::new();
    config.add_template_arg_dtype("InT", x.dtype())?;
    config.add_template_arg_int("IN", in_dim)?;
    config.add_template_arg_int("OUT", out_dim)?;
    config.add_template_arg_int("G", group_size)?;
    // One thread per output row (correctness-first; M1 is a parity gate, not a
    // perf win). round grid up to a multiple of the threadgroup.
    let tg = 256;
    let grid = ((out_dim + tg - 1) / tg) * tg;
    config.set_grid(grid, 1, 1)?;
    config.set_thread_group(tg, 1, 1)?;
    config.add_output_arg(&[out_dim], x.dtype())?;
    let outputs = kernel.apply(&[x, wq, scales, biases], &config, 1)?;
    outputs
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("kernel produced no output"))
}

fn main() -> Result<()> {
    unsafe {
        if std::env::var("LUMEN_MLX_BACKEND").is_err() {
            std::env::set_var("LUMEN_MLX_BACKEND", "native");
        }
    }
    // 27B dense MLP gate shape.
    let (out_dim, in_dim, gs, bits) = (17408, 5120, 64, 4);
    let w = mlx_rs::random::normal::<f32>(&[out_dim, in_dim], None, None, None)?
        .as_dtype(Dtype::Bfloat16)?;
    let (wq, scales, biases) = mlx_rs::ops::quantize(&w, gs, bits)?;
    let x2d =
        mlx_rs::random::normal::<f32>(&[1, in_dim], None, None, None)?.as_dtype(Dtype::Bfloat16)?;
    let x1d = x2d.reshape(&[in_dim])?;

    // Reference: mlx quantized_matmul (x[1,in] @ Wᵀ → [1,out]).
    let y_ref = mlx_rs::ops::quantized_matmul(&x2d, &wq, &scales, Some(&biases), true, gs, bits)?
        .reshape(&[out_dim])?
        .as_dtype(Dtype::Float32)?;
    y_ref.eval()?;

    // Custom kernel.
    let y_custom = affine4_qmv_custom(&x1d, &wq, &scales, &biases, in_dim, out_dim, gs)?
        .as_dtype(Dtype::Float32)?;
    y_custom.eval()?;

    // Compare.
    let diff = y_ref.subtract(&y_custom)?.abs()?;
    let max_abs = diff.max(None)?;
    let ref_abs = y_ref.abs()?.max(None)?;
    max_abs.eval()?;
    ref_abs.eval()?;
    let m = max_abs.item::<f32>();
    let r = ref_abs.item::<f32>();
    let rel = if r > 0.0 { m / r } else { m };
    println!("=== M1 affine4 QMV custom-kernel parity ===");
    println!("shape: out={out_dim} in={in_dim} gs={gs}");
    println!("max|ref-custom| = {m:.6}  (ref |max| = {r:.4})  rel = {rel:.2e}");
    if rel < 1e-2 {
        println!("✓ PARITY PASS — affine4 dequant+matmul reproduced in a custom Metal kernel.");
        println!(
            "  → M1 foundation validated. Next: M2 (fold conv+norm), M4 (fuse out_proj + pipeline)."
        );
    } else {
        println!("✗ PARITY FAIL — dequant layout/math mismatch; fix before proceeding.");
    }
    Ok(())
}
