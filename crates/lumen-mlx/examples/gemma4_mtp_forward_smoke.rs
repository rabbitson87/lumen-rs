//! drafter forward path with synthetic inputs.
//!
//! Loads the drafter, constructs random K/V/embed/hidden inputs of the
//! correct shapes, runs `draft_step` once, and verifies the output shape
//! and finiteness. Does NOT pair with the trunk — that's Phase 3.
//!
//! Run:
//!   MLX_LOCAL_SOURCE_DIR=/path/to/Documents/GitHub/mlx \
//!   DRAFTER_DIR=/path/to/models/gemma-4-26B-A4B-it-assistant-bf16 \
//!   cargo run --release -p lumen-mlx \
//!       --example gemma4_mtp_forward_smoke --features mlx-native

use std::path::Path;

use anyhow::{Context, Result};

#[cfg(feature = "mlx-native")]
fn main() -> Result<()> {
    use lumen_mlx::gemma4::mtp::load_drafter;
    use mlx_rs::Array;
    use mlx_rs::Dtype;

    let dir = std::env::var("DRAFTER_DIR").unwrap_or_else(|_| {
        "/path/to/models/gemma-4-26B-A4B-it-assistant-bf16".into()
    });
    eprintln!("[mtp-fwd-smoke] loading drafter from {dir}");
    let drafter = load_drafter(Path::new(&dir)).context("load_drafter")?;

    let tc = &drafter.config.text_config;
    let backbone = drafter.config.backbone_hidden_size as i32;
    let head_dim_sliding = tc.head_dim as i32;
    let n_kv_sliding = tc.num_key_value_heads as i32;
    let head_dim_full = tc.global_head_dim as i32;
    let n_kv_full = tc.num_global_key_value_heads as i32;

    // Synthetic cache state at position N=128 (arbitrary).
    let t_full = 128i32;
    let t_sliding = 128i32;
    let position = 128i32; // Q position for RoPE.

    let mk_random = |shape: &[i32]| -> Array {
        let n: usize = shape.iter().map(|d| *d as usize).product();
        let vals: Vec<f32> = (0..n).map(|i| 0.01_f32 * ((i % 100) as f32 - 50.0)).collect();
        Array::from_slice(&vals, shape).as_dtype(Dtype::Bfloat16).unwrap()
    };

    let trunk_embed = mk_random(&[1, 1, backbone]); // [1, 1, 2816]
    let last_hidden = mk_random(&[1, 1, backbone]); // [1, 1, 2816]
    let k_full = mk_random(&[1, n_kv_full, t_full, head_dim_full]); // [1, 2, T, 512]
    let v_full = mk_random(&[1, n_kv_full, t_full, head_dim_full]);
    let k_sliding = mk_random(&[1, n_kv_sliding, t_sliding, head_dim_sliding]); // [1, 8, T, 256]
    let v_sliding = mk_random(&[1, n_kv_sliding, t_sliding, head_dim_sliding]);

    eprintln!("[mtp-fwd-smoke] running draft_step (position={position}, t_full={t_full}, t_sliding={t_sliding})");
    let h_trunk = drafter
        .draft_step(
            &trunk_embed,
            &last_hidden,
            &k_full,
            &v_full,
            &k_sliding,
            &v_sliding,
            position,
        )
        .context("draft_step")?;

    let shape = h_trunk.shape();
    println!("=== draft_step output ===");
    println!("  shape  = {:?}", shape);
    println!("  dtype  = {:?}", h_trunk.dtype());

    let expected = [1, 1, backbone];
    if shape != expected {
        return Err(anyhow::anyhow!(
            "draft_step: shape mismatch {shape:?} != expected {expected:?}"
        ));
    }

    // Force eval and sniff finiteness via a few elementwise loads.
    h_trunk.eval().context("eval draft_step output")?;
    let casted = h_trunk.as_dtype(Dtype::Float32).context("cast to f32 for sniff")?;
    let flat = mlx_rs::ops::reshape(&casted, &[backbone]).context("reshape flat")?;
    flat.eval().context("eval flat")?;
    let vals: Vec<f32> = flat.as_slice::<f32>().to_vec();
    let n = vals.len().min(8);
    println!("  first {n} values = {:?}", &vals[..n]);
    let n_nan = vals.iter().filter(|v| v.is_nan()).count();
    let n_inf = vals.iter().filter(|v| v.is_infinite()).count();
    if n_nan > 0 || n_inf > 0 {
        return Err(anyhow::anyhow!(
            "draft_step: output contains {n_nan} NaN + {n_inf} Inf values"
        ));
    }

    println!("\n=== Phase 2 forward: PASS ===");
    Ok(())
}

#[cfg(not(feature = "mlx-native"))]
fn main() -> Result<()> {
    eprintln!("This example requires --features mlx-native");
    Ok(())
}
