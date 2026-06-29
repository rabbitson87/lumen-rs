//! M2b of the fused linear-attn kernel: per-head q/k RMSNorm (scale folded into
//! the weight) as a custom Metal kernel, validated BIT-CLOSE against
//! `mlx::fast::rms_norm`. Second fusion brick after M2a (conv+silu).
//!
//! Decode q/k path (mirrors `linear_attn_scale_fuse_enabled` LANDED path):
//!   q_scaled[h,d] = rms_norm(q_post[h,:], scaled_w_q)[d]
//!                 = q_post[h,d] * rsqrt(mean_d(q_post[h,:]^2) + eps) * scaled_w_q[d]
//! where scaled_w_q = ones(dn) * scale_q (the scale absorbed into the weight,
//! bit-identical by linearity of rms_norm). One threadgroup per head reduces
//! over dn; here correctness-first = one thread per head doing the dn-loop.
//!
//! Run: cargo run --release -p lumen-mlx --features mlx-native --example m2b_qk_rmsnorm_fused

use anyhow::{Result, anyhow};
use lumen_mlx::metal_kernel::{MetalKernel, MetalKernelConfig};
use mlx_rs::{Array, Dtype};

const QK_RMSNORM_SOURCE: &str = r#"
    uint h = thread_position_in_grid.x;          // head index
    if (h >= (uint)N_HEADS) return;
    uint base = h * (uint)DN;
    float ss = 0.0f;
    for (uint d = 0u; d < (uint)DN; ++d) {
        float v = float(x[base + d]);
        ss += v * v;
    }
    float inv = metal::rsqrt(ss / float(DN) + float(eps[0]));
    for (uint d = 0u; d < (uint)DN; ++d) {
        out[base + d] = static_cast<InT>(float(x[base + d]) * inv * float(w[d]));
    }
"#;

/// Per-head RMSNorm with folded scale. x:[n_heads, dn], w:[dn]. Returns [n_heads, dn].
fn qk_rmsnorm_fused(x: &Array, w: &Array, n_heads: i32, dn: i32, eps: f32) -> Result<Array> {
    let k = MetalKernel::new(
        "qk_rmsnorm_fused",
        &["x", "w", "eps"],
        &["out"],
        QK_RMSNORM_SOURCE,
        true,
        false,
    )?;
    let cfg = MetalKernelConfig::new();
    cfg.add_template_arg_dtype("InT", x.dtype())?;
    cfg.add_template_arg_int("N_HEADS", n_heads)?;
    cfg.add_template_arg_int("DN", dn)?;
    let tg = 64;
    let grid = ((n_heads + tg - 1) / tg) * tg;
    cfg.set_grid(grid, 1, 1)?;
    cfg.set_thread_group(tg, 1, 1)?;
    cfg.add_output_arg(&[n_heads, dn], x.dtype())?;
    let eps_arr = Array::from_slice(&[eps], &[1]);
    let out = k.apply(&[x, w, &eps_arr], &cfg, 1)?;
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
    // 27B: q has 16 key heads, k has 16 key heads, dn=128, eps=1e-6.
    let (n_heads, dn, eps) = (16, 128, 1e-6f32);
    let scale_q = (dn as f32).powf(-0.5) * (dn as f32).powf(-0.5); // illustrative folded scale
    let x = mlx_rs::random::normal::<f32>(&[n_heads, dn], None, None, None)?
        .as_dtype(Dtype::Float32)?;
    // scaled_w = ones(dn) * scale_q  (scale folded into the rms_norm weight)
    let w = mlx_rs::ops::ones::<f32>(&[dn])?.multiply(&Array::from_f32(scale_q))?;
    w.eval()?;

    // Reference: mlx::fast::rms_norm over the last axis with the folded weight.
    let y_ref = mlx_rs::fast::rms_norm(&x, &w, eps)?.as_dtype(Dtype::Float32)?;
    y_ref.eval()?;

    let y_custom = qk_rmsnorm_fused(&x, &w, n_heads, dn, eps)?.as_dtype(Dtype::Float32)?;
    y_custom.eval()?;

    let diff = y_ref.subtract(&y_custom)?.abs()?;
    let max_abs = diff.max(None)?;
    let ref_abs = y_ref.abs()?.max(None)?;
    max_abs.eval()?;
    ref_abs.eval()?;
    let m = max_abs.item::<f32>();
    let r = ref_abs.item::<f32>();
    let rel = if r > 0.0 { m / r } else { m };
    println!("=== M2b q/k RMSNorm fused-kernel parity ===");
    println!("n_heads={n_heads} dn={dn} eps={eps:e}");
    println!("max|ref-custom| = {m:.6}  (ref |max| = {r:.4})  rel = {rel:.2e}");
    if rel < 1e-2 {
        println!("✓ PARITY PASS — per-head folded-scale RMSNorm reproduced in a Metal kernel.");
        println!(
            "  → M2a+M2b bricks done. Next: M3 (SSM recurrence), M4 (wire input-side fused kernel)."
        );
    } else {
        println!("✗ PARITY FAIL — rms_norm math mismatch; fix before proceeding.");
    }
    Ok(())
}
