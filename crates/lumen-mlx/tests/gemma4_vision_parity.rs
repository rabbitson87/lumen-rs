//! Numeric parity for the MLX Gemma 4 vision tower.
//!
//! Vision towers fail silently — a wrong RoPE split, norm convention, or
//! pooling order produces plausible-looking activations rather than an error —
//! so the only real check is a tensor-level comparison against known-good
//! values.
//!
//! ## Where the golden values come from
//!
//! `tests/fixtures/gemma4_vision_golden.safetensors` holds the soft tokens the
//! upstream reference produces for `tests/fixtures/gemma4_vision_probe.png`.
//! They were derived once from a hand-port of
//! `transformers/models/gemma4/modeling_gemma4.py` (main branch) run against
//! the `mlx-community/gemma-4-26b-a4b-it-4bit` weights, then frozen here. That
//! makes this an independent check — comparing lumen against lumen would prove
//! nothing — while keeping the repo and the test path free of Python.
//!
//! The probe is 912×672 on purpose: that is already an exact multiple of
//! `patch_size × pooling_kernel_size` for the shipped 280-soft-token budget, so
//! no resampling runs and the comparison isolates the tower from any
//! bicubic-filter difference between implementations.
//!
//! ## Running
//!
//! Needs the checkpoint for the ~1.1 GB of `vision_tower.*` weights; skipped
//! when it is absent so default `cargo test` runs stay green:
//!
//! ```sh
//! LUMEN_GEMMA4_MODEL_DIR=/path/to/gemma-4-26b-a4b \
//!   cargo test -p lumen-mlx --features mlx-native --test gemma4_vision_parity -- --nocapture
//! ```
//!
//! ## Regenerating
//!
//! Only needed if the vision architecture itself changes. Re-derive the values
//! from the upstream reference for the same probe image and overwrite the
//! fixture — do **not** regenerate them from this crate.

#![cfg(feature = "mlx-native")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use mlx_rs::Array;
use mlx_rs::Dtype;

use lumen_mlx::gemma4::{NativeGemma4Config, quant_params_for};
use lumen_mlx::gemma4_vision::{
    NativeGemma4VisionConfig, NativeGemma4VisionTower, VisionProjectionQuant, prepare_image,
};

const FIXTURE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn fixture(name: &str) -> PathBuf {
    Path::new(FIXTURE_DIR).join(name)
}

fn model_dir() -> Option<PathBuf> {
    std::env::var("LUMEN_GEMMA4_MODEL_DIR")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
}

/// Load just the vision weights. `load_safetensors` is lazy, so pulling every
/// key and dropping the text ones costs nothing beyond the header parse.
fn load_vision_weights(dir: &Path) -> HashMap<String, Array> {
    let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("read model dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    shards.sort();
    let mut out = HashMap::new();
    for shard in shards {
        for (k, v) in Array::load_safetensors(&shard).expect("load safetensors") {
            if k.starts_with("vision_tower.") || k.starts_with("embed_vision.") {
                out.insert(k, v);
            }
        }
    }
    out
}

fn to_vec_f32(a: &Array) -> Vec<f32> {
    a.as_dtype(Dtype::Float32)
        .expect("cast f32")
        .as_slice::<f32>()
        .to_vec()
}

#[test]
fn vision_tower_matches_reference_soft_tokens() {
    let Some(model_dir) = model_dir() else {
        eprintln!("skipping: set LUMEN_GEMMA4_MODEL_DIR to the Gemma 4 checkpoint");
        return;
    };

    // The reference ran in float32; match it so the comparison measures the
    // port's correctness rather than bf16 rounding.
    unsafe { std::env::set_var("LUMEN_VISION_F32", "1") };

    let cfg_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(model_dir.join("config.json")).unwrap())
            .expect("parse config.json");
    let vcfg: NativeGemma4VisionConfig =
        serde_json::from_value(cfg_json["vision_config"].clone()).expect("parse vision_config");
    let budget = cfg_json["vision_soft_tokens_per_image"]
        .as_u64()
        .unwrap_or(280) as usize;

    let golden = Array::load_safetensors(fixture("gemma4_vision_golden.safetensors"))
        .expect("load golden fixture");
    let grid = to_vec_f32(&golden["grid_hw"]);
    let (rows, cols) = (grid[0] as usize, grid[1] as usize);

    let png = std::fs::read(fixture("gemma4_vision_probe.png")).expect("read probe image");
    let prepared = prepare_image(&png, vcfg.patch_size, budget, vcfg.pooling_kernel_size)
        .expect("prepare probe image");
    assert_eq!(
        prepared.grid,
        (rows, cols),
        "probe resized differently than when the fixture was made"
    );
    println!("patch grid {rows}×{cols} ({} patches)", rows * cols);

    let weights = load_vision_weights(&model_dir);
    assert!(
        !weights.is_empty(),
        "no vision_tower.* weights in {} — this checkpoint was quantized without them",
        model_dir.display()
    );
    println!("loaded {} vision tensors", weights.len());

    // Resolve the projection's quantization exactly the way the model loader
    // does, so this test also covers "did we dispatch the right quant mode".
    // Guessing affine on the nvfp4 checkpoint yields plausible-looking garbage.
    let full_cfg = NativeGemma4Config::load(&model_dir.join("config.json"))
        .expect("parse config.json as NativeGemma4Config");
    let (group_size, bits, mode) = quant_params_for(&full_cfg, "embed_vision.embedding_projection")
        .expect("resolve projection quantization");
    println!("projection quant: group_size={group_size} bits={bits} mode={mode:?}");

    let tower = NativeGemma4VisionTower::load(
        &weights,
        vcfg,
        VisionProjectionQuant {
            group_size,
            bits,
            mode,
        },
    )
    .expect("build vision tower");
    let n = (prepared.grid.0 * prepared.grid.1) as i32;
    let width = (prepared.pixel_values.len() / (n as usize)) as i32;
    let px = Array::from_slice(&prepared.pixel_values, &[n, width]);

    let soft = tower.forward(&px, prepared.grid).expect("vision forward");
    let got = to_vec_f32(&soft);
    let want = to_vec_f32(&golden["soft_tokens"]);
    assert_eq!(
        got.len(),
        want.len(),
        "soft-token shape {:?} does not match the fixture",
        soft.shape()
    );

    let mut max_abs = 0.0f32;
    let (mut dot, mut ng, mut nw) = (0.0f64, 0.0f64, 0.0f64);
    for (g, w) in got.iter().zip(want.iter()) {
        max_abs = max_abs.max((g - w).abs());
        dot += (*g as f64) * (*w as f64);
        ng += (*g as f64) * (*g as f64);
        nw += (*w as f64) * (*w as f64);
    }
    let cos = (dot / (ng.sqrt() * nw.sqrt() + 1e-12)) as f32;
    println!("soft_tokens  max|Δ| = {max_abs:.4e}   cos = {cos:.8}");

    // The golden was produced from the affine-4bit/group-64 checkpoint, so it
    // only pins *that* quantization. Running any other one still exercises the
    // whole tower — a wrong quant mode makes the FFI reject the call outright,
    // which is how the nvfp4 dispatch bug surfaced — but its soft tokens
    // legitimately differ, so the tight bound cannot apply.
    let golden_quant = group_size == 64 && bits == 4 && mode == c"affine";
    if !golden_quant {
        assert!(
            cos > 0.98,
            "soft-token cosine {cos} is far below what a requantization explains \
             — this looks structural, not numeric"
        );
        println!(
            "note: {mode:?}/{bits}bit/group{group_size} is not the quantization the golden \
             was made from; ran the tower for coverage and checked a loose bound only"
        );
        return;
    }

    // Metal and torch reduce in different orders, so exact equality is not
    // expected. Cosine similarity is the meaningful gate: anything structurally
    // wrong drops it far below 0.99.
    assert!(
        cos > 0.9995,
        "soft-token cosine similarity {cos} is too low — the port diverges structurally"
    );
    let rms: f32 = (want.iter().map(|v| v * v).sum::<f32>() / want.len() as f32).sqrt();
    assert!(
        max_abs < 0.05 * rms.max(1e-6),
        "max|Δ| {max_abs} exceeds 5% of golden RMS {rms}"
    );
}

/// The patchify layout has no error signal if you get it wrong — the tower
/// happily consumes a transposed image and returns confident nonsense. This
/// pins the mapping directly against the source pixels, so it needs neither the
/// checkpoint nor a golden tensor and runs on every `cargo test`.
///
/// Upstream's `convert_image_to_patches` reshapes `(C, H, W)` to
/// `(C, ph, p, pw, p)` then permutes to `(ph, pw, p, p, C)`: patches in raster
/// order, and within a patch, raster pixels with the 3 channels innermost.
#[test]
fn preprocessing_uses_raster_patches_with_channels_innermost() {
    const PATCH: usize = 16;
    const POOL: usize = 3;
    const BUDGET: usize = 70;
    // 480×336 is a fixed point of the resize for this budget: it is exactly
    // `max_patches × patch²` pixels with both sides divisible by
    // `patch × pool`, so the scale factor is 1 and `prepare_image` passes the
    // pixels through untouched. (The resize does not only shrink — it scales
    // *up* to fill the patch budget, so a small image would be enlarged and
    // the comparison below would be against resampled pixels.)
    let (w, h) = (480usize, 336usize);
    let expect_grid = (h / PATCH, w / PATCH);

    // Every pixel gets a distinct RGB triple so a transposed or channel-swapped
    // read cannot coincidentally match.
    let mut rgb = image::RgbImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as u32;
            rgb.put_pixel(
                x as u32,
                y as u32,
                image::Rgb([
                    (i % 251) as u8,
                    ((i / 7) % 241) as u8,
                    ((i / 13) % 239) as u8,
                ]),
            );
        }
    }
    let mut png = Vec::new();
    image::DynamicImage::ImageRgb8(rgb.clone())
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .expect("encode probe png");

    let prepared = prepare_image(&png, PATCH, BUDGET, POOL).expect("prepare");
    assert_eq!(
        prepared.grid, expect_grid,
        "image was resampled — the layout comparison below would be meaningless"
    );
    let (rows, cols) = prepared.grid;
    assert_eq!(prepared.num_soft_tokens, (rows / POOL) * (cols / POOL));

    let per_patch = 3 * PATCH * PATCH;
    assert_eq!(prepared.pixel_values.len(), rows * cols * per_patch);

    for pr in 0..rows {
        for pc in 0..cols {
            let base = (pr * cols + pc) * per_patch;
            for py in 0..PATCH {
                for px in 0..PATCH {
                    let src = rgb.get_pixel((pc * PATCH + px) as u32, (pr * PATCH + py) as u32);
                    for c in 0..3 {
                        let got = prepared.pixel_values[base + (py * PATCH + px) * 3 + c];
                        let want = src[c] as f32 / 255.0;
                        assert!(
                            (got - want).abs() < 1e-6,
                            "patch ({pr},{pc}) pixel ({py},{px}) channel {c}: {got} != {want}"
                        );
                    }
                }
            }
        }
    }
}
