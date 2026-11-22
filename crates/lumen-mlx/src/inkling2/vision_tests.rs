//! Parity of the vision tower with the oMLX reference on a small random-weights replica
//! (`scripts/golden/inkling2_vision_tower_fixture.py` runs the reference and
//! `tests/fixtures/inkling2_vision/tower/` holds the result: three encoder layers, hidden 32, two
//! heads, patch 4, merge 2, a 5 x 5 position table, float32 throughout).
//!
//! Three images of different shapes (1 x 1, 2 x 3 and 3 x 1 blocks) put the patch grids at 2 x 2,
//! 4 x 6 and 6 x 2, so the table is resampled both up and down, the patches come in merge-block
//! order with more than one block per row and per column, and the rotary positions differ per
//! axis. The fixture script prints how far the reference's features move when each of those is
//! switched off.

use std::collections::HashMap;
use std::path::PathBuf;

use mlx_rs::Array;

use super::pixels::PreparedImage;
use super::tests::{compare, host_f32};
use super::vision::{VisionArgs, VisionTower};

const SHAPES: [(usize, usize); 3] = [(1, 1), (2, 3), (3, 1)];

fn tower_dir() -> PathBuf {
    let dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inkling2_vision/tower");
    assert!(
        dir.join("config.json").is_file(),
        "{} is missing; regenerate it with scripts/golden/inkling2_vision_tower_fixture.py",
        dir.display()
    );
    dir
}

fn reference(name: &str) -> HashMap<String, Array> {
    Array::load_safetensors(tower_dir().join("reference").join(name)).expect("reference file")
}

fn image(inputs: &HashMap<String, Array>, k: usize) -> PreparedImage {
    let (rows, cols) = SHAPES[k];
    let blocks = host_f32(&inputs[&format!("image_{k}")]);
    assert_eq!(blocks.len(), rows * cols * 8 * 8 * 3);
    PreparedImage { rows, cols, blocks }
}

#[test]
fn the_tower_matches_the_reference_at_every_stage() {
    let tower = VisionTower::load(&tower_dir()).expect("load the replica");
    let (inputs, want) = (
        reference("inputs.safetensors"),
        reference("outputs.safetensors"),
    );
    for k in 0..SHAPES.len() {
        let (features, stages) = tower.encode_captured(&image(&inputs, k)).expect("encode");
        assert_eq!(stages.len(), 2 + tower.args().num_hidden_layers);
        for (n, got) in stages.iter().enumerate() {
            let expected = &want[&format!("image_{k}.stage_{n}")];
            assert_eq!(got.shape(), expected.shape(), "image {k} stage {n}");
            let width = *got.shape().last().unwrap() as usize;
            let (max_abs, min_cos) = compare(&host_f32(got), &host_f32(expected), width);
            println!("image {k} stage {n}: max|diff| {max_abs:.2e}, min cosine {min_cos:.7}");
            assert!(max_abs < 1e-5 && min_cos > 0.999_999, "image {k} stage {n}");
        }
        let expected = &want[&format!("image_{k}.features")];
        assert_eq!(features.shape(), expected.shape(), "image {k} features");
        let width = *features.shape().last().unwrap() as usize;
        let (max_abs, min_cos) = compare(&host_f32(&features), &host_f32(expected), width);
        println!("image {k} features: max|diff| {max_abs:.2e}, min cosine {min_cos:.7}");
        assert!(max_abs < 1e-5 && min_cos > 0.999_999, "image {k} features");
    }
}

#[test]
fn encode_is_what_encode_captured_returns() {
    let tower = VisionTower::load(&tower_dir()).expect("load the replica");
    let inputs = reference("inputs.safetensors");
    let img = image(&inputs, 1);
    let plain = host_f32(&tower.encode(&img).unwrap());
    let (captured, _) = tower.encode_captured(&img).unwrap();
    assert_eq!(plain, host_f32(&captured));
}

#[test]
fn a_tower_is_refused_when_its_tensors_do_not_fit_its_config() {
    let config = std::fs::read_to_string(tower_dir().join("config.json")).unwrap();
    let args = VisionArgs::from_config_str(&config).unwrap();
    let tensors = Array::load_safetensors(tower_dir().join("weights.safetensors")).unwrap();

    VisionTower::from_tensors(args.clone(), &tensors).expect("the intact set builds");

    let mut missing = tensors.clone();
    missing.remove("vision_tower.backbone.layers.1.ls_2.gamma");
    let err = VisionTower::from_tensors(args.clone(), &missing)
        .err()
        .expect("refused");
    assert!(err.to_string().contains("ls_2.gamma"), "{err}");

    let mut reshaped = tensors.clone();
    reshaped.insert(
        "vision_tower.proj_out.weight".to_string(),
        Array::zeros::<f32>(&[20, 25]).unwrap(),
    );
    let err = VisionTower::from_tensors(args.clone(), &reshaped)
        .err()
        .expect("refused");
    assert!(err.to_string().contains("proj_out.weight"), "{err}");

    // A config that is not the tower this code implements.
    let biased = config.replace("\"projector_bias\": false", "\"projector_bias\": true");
    assert!(VisionArgs::from_config_str(&biased).is_err());
    let odd_heads = config.replace("\"num_attention_heads\": 2", "\"num_attention_heads\": 5");
    assert!(VisionArgs::from_config_str(&odd_heads).is_err());
    let no_block = config.replace("\"vision_config\"", "\"vision\"");
    assert!(VisionArgs::from_config_str(&no_block).is_err());
}

#[test]
fn an_image_that_does_not_match_the_tower_is_refused() {
    let tower = VisionTower::load(&tower_dir()).expect("load the replica");
    let inputs = reference("inputs.safetensors");
    let mut short = image(&inputs, 1);
    short.blocks.truncate(short.blocks.len() - 3);
    assert!(tower.encode(&short).is_err());
    let empty = PreparedImage {
        rows: 0,
        cols: 3,
        blocks: Vec::new(),
    };
    assert!(tower.encode(&empty).is_err());
}

/// The checkpoint's tower is bfloat16. A constant written as a float32 array promotes a bfloat16
/// activation (`mlx_rs::nn::gelu` does), after which the whole tower runs in float32 and no longer
/// follows the reference's rounding; the replica's weights cast to bfloat16 make that visible.
#[test]
fn a_bfloat16_tower_stays_bfloat16_to_the_end() {
    use mlx_rs::Dtype;
    let config = std::fs::read_to_string(tower_dir().join("config.json")).unwrap();
    let args = VisionArgs::from_config_str(&config).unwrap();
    let narrow: HashMap<String, Array> =
        Array::load_safetensors(tower_dir().join("weights.safetensors"))
            .unwrap()
            .into_iter()
            .map(|(k, v)| (k, v.as_dtype(Dtype::Bfloat16).unwrap()))
            .collect();
    let tower = VisionTower::from_tensors(args, &narrow).unwrap();
    let inputs = reference("inputs.safetensors");
    let (features, stages) = tower.encode_captured(&image(&inputs, 1)).unwrap();
    assert_eq!(features.dtype(), Dtype::Bfloat16);
    for (n, stage) in stages.iter().enumerate() {
        assert_eq!(stage.dtype(), Dtype::Bfloat16, "stage {n}");
    }
    // And it is still the reference's answer to bfloat16 rounding.
    let want = &reference("outputs.safetensors")["image_1.features"];
    let width = *features.shape().last().unwrap() as usize;
    let (max_abs, min_cos) = compare(&host_f32(&features), &host_f32(want), width);
    println!(
        "bfloat16 replica vs float32 reference: max|diff| {max_abs:.3e}, min cosine {min_cos:.6}"
    );
    assert!(min_cos > 0.99, "{min_cos}");
}
