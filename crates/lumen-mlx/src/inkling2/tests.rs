//! Parity of the port with the oMLX reference on a small random-weights replica of the checkpoint
//! (`scripts/golden/inkling2_tiny_fixture.py` builds it with the reference code and
//! `tests/fixtures/inkling2_tiny/` holds the result: 24 layers, loop core 6..18 with 3 passes
//! (48 layer visits), global layers 5/11/17/23, NVFP4 experts in layers 0-19 and affine 4-bit in
//! 20-23, the same tensor names, float32 scales so both sides run in float32).
//!
//! The replica exercises every mechanism of the shipped model: local and global attention, the
//! window and the relative band both binding against the 48-token prompt, log scaling, the L2
//! clips, sinks, the short convolutions, the attention residual, the loop with its controllers,
//! and top-4 routing over 16 experts. The fixture script prints how far the reference's logits
//! move when each is switched off.
//!
//! The loop controllers' bf16 rounding is off in the end-to-end runs (on both sides; see the
//! fixture script for why) and pinned on its own in `the_loop_controllers_match_the_reference`.

use std::collections::HashMap;
use std::path::PathBuf;

use mlx_rs::Array;

use super::model::{ImageRows, Inkling2Model, LoadOptions};
use super::ops::mix_messages;

/// The committed replica. A missing fixture fails the test: a skip here would read as a pass.
fn fixture_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inkling2_tiny");
    assert!(
        dir.join("config.json").is_file(),
        "{} is missing; regenerate it with scripts/golden/inkling2_tiny_fixture.py",
        dir.display()
    );
    dir
}

fn reference(name: &str) -> HashMap<String, Array> {
    Array::load_safetensors(fixture_dir().join("reference").join(name)).expect("reference file")
}

pub(super) fn host_f32(a: &Array) -> Vec<f32> {
    let a = a.as_dtype(mlx_rs::Dtype::Float32).unwrap();
    a.eval().unwrap();
    a.try_as_slice::<f32>().unwrap().to_vec()
}

/// (max |a-b|, min cosine over rows of `width`) between two flat buffers.
pub(super) fn compare(a: &[f32], b: &[f32], width: usize) -> (f32, f32) {
    assert_eq!(a.len(), b.len());
    let max_abs = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max);
    let mut min_cos = 1.0f32;
    for (ra, rb) in a.chunks(width).zip(b.chunks(width)) {
        let dot: f32 = ra.iter().zip(rb).map(|(x, y)| x * y).sum();
        let na: f32 = ra.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = rb.iter().map(|x| x * x).sum::<f32>().sqrt();
        min_cos = min_cos.min(dot / (na * nb).max(1e-30));
    }
    (max_abs, min_cos)
}

fn argmax_rows(logits: &[f32], width: usize) -> Vec<usize> {
    logits
        .chunks(width)
        .map(|r| {
            r.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .unwrap()
        })
        .collect()
}

fn load_replica() -> Inkling2Model {
    Inkling2Model::load(
        &fixture_dir(),
        LoadOptions {
            round_loop_mix: false,
        },
    )
    .expect("load the replica")
}

fn prompt_ids(want: &HashMap<String, Array>) -> Vec<u32> {
    want["ids"]
        .as_slice::<i32>()
        .iter()
        .map(|&t| t as u32)
        .collect()
}

/// Checks `got` (`[1, L, vocab]`) against the reference rows: the ids past the unpadded vocabulary
/// are `-inf` on both sides, and the rest meet the parity bar.
fn assert_logits_match(label: &str, got: &Array, want: &Array, model: &Inkling2Model) {
    let vocab = model.args().vocab_size;
    let live = model.args().unpadded_vocab_size;
    let (got, want) = (host_f32(got), host_f32(want));
    let strip = |all: &[f32]| -> Vec<f32> {
        all.chunks(vocab)
            .flat_map(|row| {
                assert!(
                    row[live..].iter().all(|&x| x == f32::NEG_INFINITY),
                    "{label}: a padded id has a finite logit"
                );
                row[..live].iter().copied()
            })
            .collect()
    };
    let (got, want) = (strip(&got), strip(&want));
    let (max_abs, min_cos) = compare(&got, &want, live);
    eprintln!("{label}: max |diff| {max_abs:.3e}, min cosine {min_cos:.7}");
    assert_eq!(
        argmax_rows(&got, live),
        argmax_rows(&want, live),
        "{label}: top-1 per position"
    );
    assert!(min_cos > 0.9999, "{label}: min cosine {min_cos}");
    assert!(max_abs < 5e-3, "{label}: max |diff| {max_abs}");
}

/// The loop controllers on the same arrays the reference ran them on, with the reference's bf16
/// rounding and without. Identical inputs, so no upstream difference can cross a rounding boundary.
#[test]
fn the_loop_controllers_match_the_reference() {
    let want = reference("mix_messages.safetensors");
    for n in [3, 1] {
        for (suffix, round) in [("bf16", true), ("plain", false)] {
            let got = mix_messages(
                &want[&format!("x{n}")],
                &want[&format!("m{n}")],
                &want[&format!("down{n}")],
                &want[&format!("gf{n}")],
                round,
            )
            .unwrap();
            let expected = &want[&format!("out{n}_{suffix}")];
            let (max_abs, _) = compare(&host_f32(&got), &host_f32(expected), 64);
            eprintln!("mix_messages N={n} {suffix}: max |diff| {max_abs:.3e}");
            assert!(max_abs < 1e-5, "N={n} {suffix}: {max_abs}");
        }
    }
}

#[test]
fn the_replica_loads_with_48_layer_visits() {
    let model = load_replica();
    assert_eq!(model.num_visits(), 48);
    assert_eq!(model.args().num_hidden_layers, 24);
}

/// The residual stream after every layer visit, so a mismatch names the visit it starts at
/// instead of surfacing only in the logits.
///
/// Today every one of the 49 tensors is bit-identical (the fixture's reference runs the same plain
/// MLX ops the port does). The bound is relative, 1e-3 of the tensor's largest value, for the day
/// MLX changes a kernel's reduction order: that moved the same quantity by 8e-5 when the
/// reference's Metal kernel replaced the plain ops. A wrong formula moves it by 1e-2 or more.
#[test]
fn every_layer_visit_matches_the_reference() {
    let model = load_replica();
    let want = reference("outputs.safetensors");
    let ids = prompt_ids(&want);
    let mut cache = model.make_cache();
    let (_, captured) = model.forward_captured(&ids, &mut cache).expect("forward");
    assert_eq!(captured.len(), model.num_visits() + 1);
    let hidden = model.args().hidden_size;
    let mut worst = 0.0f32;
    for (i, got) in captured.iter().enumerate() {
        let expected = host_f32(&want[&format!("visit_{i:02}")]);
        let (max_abs, min_cos) = compare(&host_f32(got), &expected, hidden);
        let scale = expected.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        worst = worst.max(max_abs / scale);
        eprintln!("visit {i:02}: max |diff| {max_abs:.3e} of {scale:.1}, min cosine {min_cos:.7}");
        assert!(
            max_abs <= 1e-3 * scale,
            "visit {i}: max |diff| {max_abs} against a largest value of {scale}"
        );
    }
    eprintln!("worst residual-stream difference: {worst:.3e} of the largest value");
}

#[test]
fn a_full_prefill_matches_the_reference_logits() {
    let model = load_replica();
    let want = reference("outputs.safetensors");
    let ids = prompt_ids(&want);
    let mut cache = model.make_cache();
    let logits = model.forward(&ids, &mut cache, false).expect("forward");
    assert_logits_match("prefill", &logits, &want["logits_full"], &model);
    assert_eq!(cache.len(), ids.len());
}

#[test]
fn prefill_then_single_token_decode_matches_the_reference() {
    let model = load_replica();
    let want = reference("outputs.safetensors");
    let ids = prompt_ids(&want);
    let mut cache = model.make_cache();
    let mut parts = vec![model.forward(&ids[..40], &mut cache, false).unwrap()];
    for t in 40..ids.len() {
        parts.push(model.forward(&ids[t..t + 1], &mut cache, false).unwrap());
    }
    let refs: Vec<&Array> = parts.iter().collect();
    let logits = mlx_rs::ops::concatenate_axis(&refs, 1).unwrap();
    assert_logits_match("prefill + decode", &logits, &want["logits_split"], &model);
    assert_eq!(cache.len(), ids.len());
}

/// The same prompt fed in uneven chunks must land on the same logits as one prefill: every carried
/// state (the KV caches of all 48 visits and the four convolution states of each) crosses the
/// boundaries, including a chunk shorter than the convolution and one that straddles the window.
#[test]
fn chunked_prefill_matches_a_single_prefill() {
    let model = load_replica();
    let want = reference("outputs.safetensors");
    let ids = prompt_ids(&want);
    let mut cache = model.make_cache();
    let mut parts = Vec::new();
    let mut at = 0;
    for len in [13, 1, 20, 7, 7] {
        parts.push(
            model
                .forward(&ids[at..at + len], &mut cache, false)
                .unwrap(),
        );
        at += len;
    }
    assert_eq!(at, ids.len());
    let refs: Vec<&Array> = parts.iter().collect();
    let logits = mlx_rs::ops::concatenate_axis(&refs, 1).unwrap();
    assert_logits_match("chunked prefill", &logits, &want["logits_full"], &model);
}

#[test]
fn last_only_returns_the_last_position_of_the_full_logits() {
    let model = load_replica();
    let want = reference("outputs.safetensors");
    let ids = prompt_ids(&want);
    let mut cache = model.make_cache();
    let last = model.forward(&ids, &mut cache, true).expect("forward");
    assert_eq!(last.shape(), &[1, 1, model.args().vocab_size as i32]);
    let full = host_f32(&want["logits_full"]);
    let vocab = model.args().vocab_size;
    let expected = &full[full.len() - vocab..];
    let got = host_f32(&last);
    let live = model.args().unpadded_vocab_size;
    let (max_abs, _) = compare(&got[..live], &expected[..live], live);
    assert!(max_abs < 5e-3, "last-only: {max_abs}");
}

/// The positions a spliced prompt marks as images in the splice tests: a run of 8 and a run of 3,
/// with text between and after, so a chunk boundary can fall inside a run, between runs and on a
/// run's edge.
const IMAGE_RUNS: [(usize, usize); 2] = [(10, 8), (30, 3)];

/// `ids` with an image-token stand-in at the marked positions, and the rows that must take the
/// place of the embeddings those positions had: the model's own embeddings of the original ids, so
/// the spliced forward pass has to land on exactly what the plain one computes.
fn marked_prompt(model: &Inkling2Model, ids: &[u32]) -> (Vec<u32>, ImageRows) {
    let stand_in = ids.iter().max().unwrap() + 1;
    let mut marked = ids.to_vec();
    let mut slots = vec![-1i32; ids.len()];
    let mut originals = Vec::new();
    for &(start, len) in &IMAGE_RUNS {
        for p in start..start + len {
            slots[p] = originals.len() as i32;
            originals.push(ids[p]);
            marked[p] = stand_in;
        }
    }
    let features =
        mlx_rs::ops::squeeze_axes(model.embed_tokens(&originals).unwrap(), &[0]).unwrap();
    (marked, ImageRows::new(slots, features).unwrap())
}

#[test]
fn spliced_rows_stand_in_for_the_embeddings_they_replace() {
    let model = load_replica();
    let ids = prompt_ids(&reference("outputs.safetensors"));
    let (marked, rows) = marked_prompt(&model, &ids);

    let plain = model.forward(&ids, &mut model.make_cache(), false).unwrap();
    let spliced = model
        .forward_with(&marked, &mut model.make_cache(), false, Some(&rows))
        .unwrap();
    assert_eq!(
        host_f32(&plain),
        host_f32(&spliced),
        "a splice of the same rows is the same pass"
    );

    // The test can tell: the stand-in tokens alone are a different prompt.
    let unspliced = model
        .forward(&marked, &mut model.make_cache(), false)
        .unwrap();
    let (max_abs, _) = compare(
        &host_f32(&unspliced),
        &host_f32(&plain),
        model.args().vocab_size,
    );
    assert!(
        max_abs > 1e-2 || max_abs.is_nan(),
        "the stand-ins did not change the logits: {max_abs}"
    );
}

#[test]
fn a_prompt_fed_in_chunks_splices_each_chunk_with_its_own_rows() {
    let model = load_replica();
    let ids = prompt_ids(&reference("outputs.safetensors"));
    let (marked, rows) = marked_prompt(&model, &ids);
    let plain = model.forward(&ids, &mut model.make_cache(), true).unwrap();
    // 7 cuts the first run, 4 leaves chunk edges on both runs' edges, 1 feeds a token at a time.
    for chunk in [7usize, 4, 1, 13] {
        let mut cache = model.make_cache();
        let last_start = (marked.len() - 1) / chunk * chunk;
        for piece in marked[..last_start].chunks(chunk) {
            model.advance_with(piece, &mut cache, Some(&rows)).unwrap();
        }
        let got = model
            .forward_with(&marked[last_start..], &mut cache, true, Some(&rows))
            .unwrap();
        assert_logits_match(&format!("chunk {chunk}"), &got, &plain, &model);
    }
}

#[test]
fn image_rows_that_do_not_fit_the_pass_are_refused() {
    let model = load_replica();
    let ids = prompt_ids(&reference("outputs.safetensors"));
    let (marked, rows) = marked_prompt(&model, &ids);
    let hidden = model.args().hidden_size as i32;
    let one_row = || Array::zeros::<f32>(&[1, hidden]).unwrap();

    // A slot past the rows, or below the plain marker.
    assert!(ImageRows::new(vec![-1, 1], one_row()).is_err());
    assert!(ImageRows::new(vec![-2], one_row()).is_err());
    // Features that are not a matrix, or not as wide as the embeddings.
    assert!(ImageRows::new(vec![0], Array::zeros::<f32>(&[hidden]).unwrap()).is_err());
    let narrow = ImageRows::new(
        vec![0; marked.len()],
        Array::zeros::<f32>(&[1, hidden - 1]).unwrap(),
    )
    .unwrap();
    assert!(
        model
            .forward_with(&marked, &mut model.make_cache(), true, Some(&narrow))
            .is_err()
    );
    // Slots for fewer tokens than the pass feeds.
    let short = ImageRows::new(vec![-1; marked.len() - 1], one_row()).unwrap();
    assert!(
        model
            .forward_with(&marked, &mut model.make_cache(), true, Some(&short))
            .is_err()
    );
    // The intact set still works.
    assert!(
        model
            .forward_with(&marked, &mut model.make_cache(), true, Some(&rows))
            .is_ok()
    );
}
