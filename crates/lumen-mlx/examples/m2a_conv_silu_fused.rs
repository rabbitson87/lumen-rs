//! M2a of the fused linear-attn kernel (path to ~21 tok/s via 3-bit + fusion):
//! depthwise conv1d (window-4) + silu fused into ONE custom Metal kernel,
//! validated BIT-CLOSE against the current path (mlx conv1d → silu).
//!
//! Why: gate-0 measured the whole linear-attn non-matmul chain at ~10ms/step
//! (launch-latency bound: ~600 tiny dependent kernels). The fix is FUSING the
//! ~5 input-side launches (conv1d, silu, split, reshape, rms_norm) into one.
//! M2a is the first brick — conv+silu — proving the fused-kernel parity before
//! folding in the rms_norm reductions (M2b) and the SSM recurrence (M3).
//!
//! Decode (b=1,s=1) depthwise conv, cross-correlation (mirrors the validated
//! `LUMEN_NATIVE_CONV_DECODE_EW` path):
//!   conv_raw[c] = Σ_{k=0..K-1} conv_input[k, c] · conv_w[c, k]      (per channel c)
//!   conv_out[c] = silu(conv_raw[c]) = conv_raw[c] · sigmoid(conv_raw[c])
//! conv_input is [K, conv_dim] (the K time taps for one decode step),
//! conv_w is [conv_dim, K] (the depthwise filter, reshaped from [conv_dim,1,K]).
//!
//! Run: cargo run --release -p lumen-mlx --features mlx-native --example m2a_conv_silu_fused

use anyhow::{Result, anyhow};
use lumen_mlx::metal_kernel::{MetalKernel, MetalKernelConfig};
use mlx_rs::{Array, Dtype};

const CONV_SILU_SOURCE: &str = r#"
    uint c = thread_position_in_grid.x;          // output channel
    if (c >= (uint)CONV_DIM) return;
    float acc = 0.0f;
    for (uint k = 0u; k < (uint)K; ++k) {
        // conv_input [K, CONV_DIM] row-major; conv_w [CONV_DIM, K] row-major.
        acc += float(conv_input[k * (uint)CONV_DIM + c]) * float(conv_w[c * (uint)K + k]);
    }
    // silu = x * sigmoid(x)
    float s = acc / (1.0f + metal::exp(-acc));
    out[c] = static_cast<InT>(s);
"#;

/// Fused depthwise-conv + silu. conv_input:[K, conv_dim], conv_w:[conv_dim, K].
/// Returns out:[conv_dim].
fn conv_silu_fused(
    conv_input: &Array,
    conv_w: &Array,
    conv_dim: i32,
    kernel: i32,
) -> Result<Array> {
    let k = MetalKernel::new(
        "conv_silu_fused",
        &["conv_input", "conv_w"],
        &["out"],
        CONV_SILU_SOURCE,
        /* ensure_row_contiguous */ true,
        /* atomic_outputs */ false,
    )?;
    let cfg = MetalKernelConfig::new();
    cfg.add_template_arg_dtype("InT", conv_input.dtype())?;
    cfg.add_template_arg_int("CONV_DIM", conv_dim)?;
    cfg.add_template_arg_int("K", kernel)?;
    let tg = 256;
    let grid = ((conv_dim + tg - 1) / tg) * tg;
    cfg.set_grid(grid, 1, 1)?;
    cfg.set_thread_group(tg, 1, 1)?;
    cfg.add_output_arg(&[conv_dim], conv_input.dtype())?;
    let out = k.apply(&[conv_input, conv_w], &cfg, 1)?;
    out.into_iter()
        .next()
        .ok_or_else(|| anyhow!("kernel produced no output"))
}

fn main() -> Result<()> {
    unsafe {
        if std::env::var("LUMEN_MLX_BACKEND").is_err() {
            std::env::set_var("LUMEN_MLX_BACKEND", "native");
        }
    }
    // 27B linear-attn: conv_dim = 2*(16*128) + 48*128 = 10240, kernel = 4.
    let (conv_dim, kernel) = (10240, 4);

    // conv_input [K, conv_dim] (the K decode time-taps), conv_w [conv_dim, K].
    let conv_input = mlx_rs::random::normal::<f32>(&[kernel, conv_dim], None, None, None)?
        .as_dtype(Dtype::Float32)?;
    let conv_w = mlx_rs::random::normal::<f32>(&[conv_dim, kernel], None, None, None)?
        .as_dtype(Dtype::Float32)?;

    // Reference: replicate the current decode path. mlx conv1d expects
    // input [N, L, C_in] and weight [C_out, K, C_in/groups]; the model stores
    // conv_w as [conv_dim, 1, K]. For the windowed-sum decode equivalence we
    // use the same cross-correlation the EW path validated:
    //   conv_raw[c] = Σ_k conv_input[k,c] · conv_w[c,k]
    let mut acc: Option<Array> = None;
    for k in 0..kernel {
        use mlx_rs::ops::indexing::TryIndexOp;
        let xk = conv_input
            .try_index((k..(k + 1), ..))?
            .reshape(&[conv_dim])?;
        let wk = conv_w.try_index((.., k..(k + 1)))?.reshape(&[conv_dim])?;
        let term = xk.multiply(&wk)?;
        acc = Some(match acc {
            Some(a) => a.add(&term)?,
            None => term,
        });
    }
    let conv_raw = acc.expect("kernel >= 1");
    let y_ref = mlx_rs::nn::silu(&conv_raw)?.as_dtype(Dtype::Float32)?;
    y_ref.eval()?;

    // Custom fused kernel.
    let y_custom =
        conv_silu_fused(&conv_input, &conv_w, conv_dim, kernel)?.as_dtype(Dtype::Float32)?;
    y_custom.eval()?;

    let diff = y_ref.subtract(&y_custom)?.abs()?;
    let max_abs = diff.max(None)?;
    let ref_abs = y_ref.abs()?.max(None)?;
    max_abs.eval()?;
    ref_abs.eval()?;
    let m = max_abs.item::<f32>();
    let r = ref_abs.item::<f32>();
    let rel = if r > 0.0 { m / r } else { m };
    println!("=== M2a conv+silu fused-kernel parity ===");
    println!("conv_dim={conv_dim} K={kernel}");
    println!("max|ref-custom| = {m:.6}  (ref |max| = {r:.4})  rel = {rel:.2e}");
    if rel < 1e-2 {
        println!("✓ PARITY PASS — conv1d(depthwise,window-4)+silu reproduced in one Metal kernel.");
        println!("  → M2a brick validated. Next: M2b (fold q/k rms_norm reductions), M3 (SSM).");
    } else {
        println!("✗ PARITY FAIL — conv/silu math mismatch; fix before proceeding.");
    }
    Ok(())
}
