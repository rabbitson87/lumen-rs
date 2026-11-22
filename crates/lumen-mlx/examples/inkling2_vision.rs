//! The `inkling2` vision tower on the real `Inkling-2-Mini` checkpoint: encode an image and compare
//! the features with the oMLX reference's.
//!
//! ```text
//! python scripts/golden/inkling2_vision_real.py <omlx ref dir> <model_dir> <out dir>
//! cargo run --release -p lumen-mlx --features mlx-native --example inkling2_vision -- \
//!     <model_dir> <image.png> [--reference <out dir>]
//! ```
//!
//! The script writes the reference tower's features on the checkpoint's bfloat16 weights and on the
//! same weights widened to float32. The port runs the weights as stored (bfloat16), so the number
//! to read the comparison against is how far the reference's own bfloat16 run is from its float32
//! one: the port should land in that distance, not beyond it.

#![cfg(feature = "mlx-native")]

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use lumen_mlx::inkling2::{VisionTower, prepare_image};
use mlx_rs::Array;

fn host_f32(a: &Array) -> Result<Vec<f32>> {
    let a = a.as_dtype(mlx_rs::Dtype::Float32)?;
    a.eval()?;
    Ok(a.try_as_slice::<f32>()?.to_vec())
}

/// (max |a-b|, min cosine over rows of `width`, largest |b|).
fn compare(a: &[f32], b: &[f32], width: usize) -> (f32, f32, f32) {
    let max_abs = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max);
    let scale = b.iter().map(|x| x.abs()).fold(0.0, f32::max);
    let mut min_cos = 1.0f32;
    for (ra, rb) in a.chunks(width).zip(b.chunks(width)) {
        let dot: f32 = ra.iter().zip(rb).map(|(x, y)| x * y).sum();
        let na: f32 = ra.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = rb.iter().map(|x| x * x).sum::<f32>().sqrt();
        min_cos = min_cos.min(dot / (na * nb).max(1e-30));
    }
    (max_abs, min_cos, scale)
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let (Some(dir), Some(image)) = (args.next(), args.next()) else {
        bail!("usage: inkling2_vision <model_dir> <image.png> [--reference <dir>]");
    };
    let mut reference: Option<PathBuf> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--reference" => reference = args.next().map(PathBuf::from),
            other => bail!("unknown argument {other}"),
        }
    }

    let started = Instant::now();
    let tower = VisionTower::load(&PathBuf::from(&dir)).context("load the tower")?;
    println!("tower loaded in {:.2?}", started.elapsed());

    let bytes = std::fs::read(&image).with_context(|| format!("read {image}"))?;
    let prepared = prepare_image(&bytes)?;
    println!(
        "{image}: {} x {} blocks = {} image tokens",
        prepared.rows,
        prepared.cols,
        prepared.tokens()
    );

    for run in 0..2 {
        let started = Instant::now();
        let features = tower.encode(&prepared)?;
        features.eval()?;
        println!(
            "encode #{run}: {:.1?}, features {:?} {:?}",
            started.elapsed(),
            features.shape(),
            features.dtype()
        );
        if run == 1 {
            if let Some(dir) = &reference {
                let width = *features.shape().last().unwrap() as usize;
                let got = host_f32(&features)?;
                for name in ["features_bf16", "features_f32"] {
                    let file = Array::load_safetensors(dir.join(format!("{name}.safetensors")))
                        .with_context(|| format!("read {name}"))?;
                    let (max_abs, min_cos, scale) =
                        compare(&got, &host_f32(&file["features"])?, width);
                    println!(
                        "port vs reference {name}: max |diff| {max_abs:.4} (largest feature {scale:.3}), min row cosine {min_cos:.6}"
                    );
                }
            }
        }
    }
    Ok(())
}
