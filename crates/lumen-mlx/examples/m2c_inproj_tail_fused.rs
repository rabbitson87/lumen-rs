//! M2c: the WIREABLE input-side fused kernel — conv1d(window-4) + silu + split
//! + per-head q/k RMSNorm, ALL in one Metal dispatch, producing q_scaled,
//! k_scaled, v_post directly. Replaces ~5 launches/layer (conv1d, silu, split,
//! rms_norm×2) with 1. Validated bit-close against the current op-by-op path.
//! This is the unit M4 wires into `layer_linear_attn_forward`.
//!
//! Layout (27B decode b=1,s=1): conv_dim = 2*key_dim + value_dim,
//! key_dim = nk*dn (q and k each), value_dim = nv*dv.
//!   channels [0, key_dim)            → q  (nk heads × dn)  → RMSNorm(scaled_w_q)
//!   channels [key_dim, 2*key_dim)    → k  (nk heads × dn)  → RMSNorm(scaled_w_k)
//!   channels [2*key_dim, conv_dim)   → v  (nv heads × dv)  → conv+silu only
//! One threadgroup per head-row (nk + nk + nv rows), DN threads each, threadgroup
//! reduction for the q/k RMSNorm.
//!
//! Run: cargo run --release -p lumen-mlx --features mlx-native --example m2c_inproj_tail_fused

use anyhow::{Result, anyhow};
use lumen_mlx::metal_kernel::{MetalKernel, MetalKernelConfig};
use mlx_rs::{Array, Dtype};

// grid.x = DN (lane within head), grid.y = row (q heads | k heads | v heads).
// threadgroup = (DN, 1, 1). Each row is one head; lane d handles channel base+d.
const INPROJ_TAIL_SOURCE: &str = r#"
    uint d = thread_position_in_grid.x;          // lane within head (0..DN)
    uint row = thread_position_in_grid.y;        // 0..(NK+NK+NV)
    threadgroup float sh[1024];                  // ≥ DN/Dv; reduction scratch

    uint nk = (uint)NK, nv = (uint)NV;
    uint dn = (uint)DN, dv = (uint)DV;
    uint key_dim = nk * dn;
    uint conv_dim = 2u * key_dim + nv * dv;

    // region: 0=q, 1=k, 2=v
    uint region, head, width, ch_base;
    if (row < nk)            { region = 0u; head = row;        width = dn; ch_base = head * dn; }
    else if (row < 2u*nk)    { region = 1u; head = row - nk;   width = dn; ch_base = key_dim + head * dn; }
    else                     { region = 2u; head = row - 2u*nk; width = dv; ch_base = 2u*key_dim + head * dv; }

    if (d >= width) return;
    uint c = ch_base + d;

    // conv1d (window K, cross-correlation) + silu
    float acc = 0.0f;
    for (uint k = 0u; k < (uint)K; ++k) {
        acc += float(conv_input[k * conv_dim + c]) * float(conv_w[c * (uint)K + k]);
    }
    float s = acc / (1.0f + metal::exp(-acc));   // silu

    if (region == 2u) {
        // v: conv+silu only, no norm
        out_v[head * dv + d] = static_cast<InT>(s);
        return;
    }

    // q/k: RMSNorm over the head's DN lanes (threadgroup reduction)
    sh[d] = s * s;
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    // tree reduction over `width` (=dn, power of two)
    for (uint stride = width >> 1; stride > 0u; stride >>= 1) {
        if (d < stride) sh[d] += sh[d + stride];
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
    float inv = metal::rsqrt(sh[0] / float(width) + float(eps[0]));
    float wv = (region == 0u) ? float(wq[d]) : float(wk[d]);
    float y = s * inv * wv;
    if (region == 0u) out_q[head * dn + d] = static_cast<InT>(y);
    else              out_k[head * dn + d] = static_cast<InT>(y);
"#;

#[allow(clippy::too_many_arguments)]
fn inproj_tail_fused(
    conv_input: &Array,
    conv_w: &Array,
    wq: &Array,
    wk: &Array,
    eps: f32,
    nk: i32,
    nv: i32,
    dn: i32,
    dv: i32,
    k_taps: i32,
) -> Result<(Array, Array, Array)> {
    let kernel = MetalKernel::new(
        "inproj_tail_fused",
        &["conv_input", "conv_w", "wq", "wk", "eps"],
        &["out_q", "out_k", "out_v"],
        INPROJ_TAIL_SOURCE,
        true,
        false,
    )?;
    let cfg = MetalKernelConfig::new();
    cfg.add_template_arg_dtype("InT", conv_input.dtype())?;
    cfg.add_template_arg_int("NK", nk)?;
    cfg.add_template_arg_int("NV", nv)?;
    cfg.add_template_arg_int("DN", dn)?;
    cfg.add_template_arg_int("DV", dv)?;
    cfg.add_template_arg_int("K", k_taps)?;
    let rows = nk + nk + nv;
    let lane = dn.max(dv);
    cfg.set_grid(lane, rows, 1)?;
    cfg.set_thread_group(lane, 1, 1)?;
    cfg.add_output_arg(&[nk, dn], conv_input.dtype())?;
    cfg.add_output_arg(&[nk, dn], conv_input.dtype())?;
    cfg.add_output_arg(&[nv, dv], conv_input.dtype())?;
    let eps_arr = Array::from_slice(&[eps], &[1]);
    let outs = kernel.apply(&[conv_input, conv_w, wq, wk, &eps_arr], &cfg, 3)?;
    let mut it = outs.into_iter();
    let q = it.next().ok_or_else(|| anyhow!("no out_q"))?;
    let k = it.next().ok_or_else(|| anyhow!("no out_k"))?;
    let v = it.next().ok_or_else(|| anyhow!("no out_v"))?;
    Ok((q, k, v))
}

fn main() -> Result<()> {
    unsafe {
        if std::env::var("LUMEN_MLX_BACKEND").is_err() {
            std::env::set_var("LUMEN_MLX_BACKEND", "native");
        }
    }
    // 27B linear-attn dims.
    let (nk, nv, dn, dv, k_taps, eps) = (16, 48, 128, 128, 4, 1e-6f32);
    let key_dim = nk * dn;
    let value_dim = nv * dv;
    let conv_dim = 2 * key_dim + value_dim;

    let conv_input = mlx_rs::random::normal::<f32>(&[k_taps, conv_dim], None, None, None)?
        .as_dtype(Dtype::Float32)?;
    let conv_w = mlx_rs::random::normal::<f32>(&[conv_dim, k_taps], None, None, None)?
        .as_dtype(Dtype::Float32)?;
    let scale_q = (dn as f32).powf(-0.5) * (dn as f32).powf(-0.5);
    let scale_k = (dn as f32).powf(-0.5);
    let wq = mlx_rs::ops::ones::<f32>(&[dn])?.multiply(&Array::from_f32(scale_q))?;
    let wk = mlx_rs::ops::ones::<f32>(&[dn])?.multiply(&Array::from_f32(scale_k))?;
    wq.eval()?;
    wk.eval()?;

    // ---- Reference: the current op-by-op path ----
    let mut acc: Option<Array> = None;
    for k in 0..k_taps {
        use mlx_rs::ops::indexing::TryIndexOp;
        let xk = conv_input
            .try_index((k..(k + 1), ..))?
            .reshape(&[conv_dim])?;
        let wk_ = conv_w.try_index((.., k..(k + 1)))?.reshape(&[conv_dim])?;
        acc = Some(match acc {
            Some(a) => a.add(&xk.multiply(&wk_)?)?,
            None => xk.multiply(&wk_)?,
        });
    }
    let conv_out = mlx_rs::nn::silu(&acc.unwrap())?;
    let parts = mlx_rs::ops::split_sections(&conv_out, &[key_dim, 2 * key_dim], -1)?;
    let q_post = parts[0].reshape(&[nk, dn])?;
    let k_post = parts[1].reshape(&[nk, dn])?;
    let v_post = parts[2].reshape(&[nv, dv])?;
    let q_ref = mlx_rs::fast::rms_norm(&q_post, &wq, eps)?.as_dtype(Dtype::Float32)?;
    let k_ref = mlx_rs::fast::rms_norm(&k_post, &wk, eps)?.as_dtype(Dtype::Float32)?;
    let v_ref = v_post.as_dtype(Dtype::Float32)?;
    q_ref.eval()?;
    k_ref.eval()?;
    v_ref.eval()?;

    // ---- Custom fused kernel ----
    let (q_c, k_c, v_c) =
        inproj_tail_fused(&conv_input, &conv_w, &wq, &wk, eps, nk, nv, dn, dv, k_taps)?;
    let q_c = q_c.as_dtype(Dtype::Float32)?;
    let k_c = k_c.as_dtype(Dtype::Float32)?;
    let v_c = v_c.as_dtype(Dtype::Float32)?;
    q_c.eval()?;
    k_c.eval()?;
    v_c.eval()?;

    let rel = |a: &Array, b: &Array| -> Result<f32> {
        let m = a.subtract(b)?.abs()?.max(None)?;
        let r = a.abs()?.max(None)?;
        m.eval()?;
        r.eval()?;
        let mv = m.item::<f32>();
        let rv = r.item::<f32>();
        Ok(if rv > 0.0 { mv / rv } else { mv })
    };
    let rq = rel(&q_ref, &q_c)?;
    let rk = rel(&k_ref, &k_c)?;
    let rv = rel(&v_ref, &v_c)?;
    println!("=== M2c input-side fused kernel parity (conv+silu+split+rmsnorm → q/k/v) ===");
    println!("nk={nk} nv={nv} dn={dn} dv={dv} conv_dim={conv_dim}");
    println!("rel q={rq:.2e}  k={rk:.2e}  v={rv:.2e}");
    if rq < 1e-2 && rk < 1e-2 && rv < 1e-2 {
        println!("✓ PARITY PASS — entire input-side chain in ONE kernel (5 launches → 1).");
        println!(
            "  → M2c wireable unit validated. Next: M4 wire into layer_linear_attn_forward (default-OFF)."
        );
    } else {
        println!("✗ PARITY FAIL — fix before wiring.");
    }
    Ok(())
}
