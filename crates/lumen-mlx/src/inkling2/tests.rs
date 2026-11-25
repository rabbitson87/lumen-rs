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

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use mlx_rs::Array;

use super::model::{ImageRows, Inkling2Model, LoadOptions};
use super::ops::mix_messages;

/// The committed replica. A missing fixture fails the test: a skip here would read as a pass.
pub(super) fn fixture_dir() -> PathBuf {
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
    // Flattened first: a transposed view read as a slice would come out in storage order.
    let a = mlx_rs::ops::reshape(a.as_dtype(mlx_rs::Dtype::Float32).unwrap(), &[-1]).unwrap();
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

pub(super) fn load_replica() -> Inkling2Model {
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

fn prefill_then_single_token_decode_against_the_reference(label: &str) {
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
    assert_logits_match(label, &logits, &want["logits_split"], &model);
    assert_eq!(cache.len(), ids.len());
}

#[test]
fn prefill_then_single_token_decode_matches_the_reference() {
    prefill_then_single_token_decode_against_the_reference("prefill + decode");
}

/// The same with the kernels off, so that every decode step runs the compiled regions (`fused.rs`)
/// the kernels replace by default: the guard of a defect in a region's arithmetic that only a decode
/// step with a convolution state shows (the first chunk of a prefill starts from a state of zeros).
#[test]
fn prefill_then_single_token_decode_matches_the_reference_through_the_compiled_regions() {
    without_kernels(|| {
        prefill_then_single_token_decode_against_the_reference("prefill + decode, compiled regions")
    });
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

/// The draft head reads the stream entering visits 5, 23 and 41 and the final-norm output. They
/// are the reference's own residual-stream tensors (the fixture keeps one per visit), side by
/// side on the feature axis in the order of the ids.
#[test]
fn taps_are_the_stream_entering_the_chosen_visits() {
    let model = load_replica();
    let want = reference("outputs.safetensors");
    let ids = prompt_ids(&want);
    let hidden = model.args().hidden_size;
    let tap_ids = [5usize, 23, 41, model.num_visits()];
    let taps = model
        .advance_taps(&ids, &mut model.make_cache(), &tap_ids, None)
        .expect("taps");
    assert_eq!(taps.shape(), &[1, ids.len() as i32, (4 * hidden) as i32]);
    let got = host_f32(&taps);
    for (k, &id) in tap_ids.iter().enumerate() {
        let expected = host_f32(&want[&format!("visit_{id:02}")]);
        let scale = expected.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let slice: Vec<f32> = got
            .chunks(4 * hidden)
            .flat_map(|row| row[k * hidden..(k + 1) * hidden].iter().copied())
            .collect();
        let (max_abs, _) = compare(&slice, &expected, hidden);
        assert!(max_abs <= 1e-3 * scale, "tap {id}: {max_abs} of {scale}");
    }
    // An id that names nothing is refused, and so is no id at all.
    let mut cache = model.make_cache();
    assert!(model.advance_taps(&ids, &mut cache, &[999], None).is_err());
    assert!(
        model
            .advance_taps(&ids, &mut model.make_cache(), &[], None)
            .is_err()
    );
}

/// A verify pass of 8 tokens rolled back to its first `kept` leaves the caches where a fresh
/// prefill of just those tokens does: the next logits agree. `kept` of 8 is a no-op, 1 keeps only
/// the anchor.
#[test]
fn a_rolled_back_verify_pass_leaves_the_caches_a_shorter_prefill_would() {
    let model = load_replica();
    let ids = prompt_ids(&reference("outputs.safetensors"));
    let (start, block) = (30usize, 8usize);
    for kept in [1usize, 3, 5, 8] {
        let mut cache = model.make_cache();
        model.advance(&ids[..start], &mut cache).unwrap();
        let (logits, _) = model
            .verify(&ids[start..start + block], &mut cache, &[5, 48])
            .unwrap();
        logits.eval().unwrap();
        cache.rollback(block - kept).expect("rollback");
        assert_eq!(cache.len(), start + kept);

        let mut fresh = model.make_cache();
        model.advance(&ids[..start + kept], &mut fresh).unwrap();
        // What follows is the next tokens of the same prompt after the kept ones; they differ from
        // the verified block's, which is the point: nothing of the rejected tokens may remain.
        let next: Vec<u32> = ids[start + block..start + block + 4].to_vec();
        let a = model.forward(&next, &mut cache, false).unwrap();
        let b = model.forward(&next, &mut fresh, false).unwrap();
        let vocab = model.args().unpadded_vocab_size;
        let live = |x: &Array| -> Vec<f32> {
            host_f32(x)
                .chunks(model.args().vocab_size)
                .flat_map(|r| r[..vocab].iter().copied())
                .collect()
        };
        let (max_abs, min_cos) = compare(&live(&a), &live(&b), vocab);
        eprintln!("kept {kept}: max |diff| {max_abs:.2e}, min cosine {min_cos:.7}");
        // A pass of 8 tokens and a pass of `kept` differ by the routing and reduction noise of the
        // replica (up to ~6e-4, as the chunking tests do); a wrong conv state or a KV row left
        // behind moves the logits by 1 or more (checked with planted faults).
        assert!(
            max_abs < 3e-3 && min_cos > 0.999_99,
            "kept {kept}: {max_abs}, {min_cos}"
        );
    }
}

#[test]
fn a_rollback_needs_the_windows_a_verify_pass_keeps() {
    let model = load_replica();
    let ids = prompt_ids(&reference("outputs.safetensors"));
    // A plain pass keeps none.
    let mut cache = model.make_cache();
    model.advance(&ids[..16], &mut cache).unwrap();
    assert!(cache.rollback(1).is_err());
    // A verify pass keeps them, and cannot give back the whole pass or more.
    let mut cache = model.make_cache();
    model
        .verify(&ids[..8], &mut cache, &[5])
        .unwrap()
        .0
        .eval()
        .unwrap();
    assert!(cache.rollback(8).is_err());
    assert!(cache.rollback(9).is_err());
    let mut cache = model.make_cache();
    model
        .verify(&ids[..8], &mut cache, &[5])
        .unwrap()
        .0
        .eval()
        .unwrap();
    cache.rollback(7).unwrap();
    assert_eq!(cache.len(), 1);
    // The windows are used up by the rollback: a second one has none to read.
    assert!(cache.rollback(1).is_err());
}

/// RED before: a refused rollback had already taken the first visit's windows, so the corrected
/// one that followed failed with "did not keep its convolution windows".
#[test]
fn a_refused_rollback_leaves_the_caches_as_they_were() {
    let model = load_replica();
    let ids = prompt_ids(&reference("outputs.safetensors"));
    let mut cache = model.make_cache();
    model
        .verify(&ids[..8], &mut cache, &[5])
        .unwrap()
        .0
        .eval()
        .unwrap();
    assert!(cache.rollback(8).is_err());
    assert!(cache.rollback(9).is_err());
    assert_eq!(cache.len(), 8, "the refusals changed nothing");
    cache.rollback(7).expect("the windows are still there");
    assert_eq!(cache.len(), 1);
}

/// RED before: a tap id past the final norm was found after the pass, which had advanced every
/// cache; a token id past the vocabulary was handed to a lookup that does not check it.
#[test]
fn a_bad_tap_or_token_id_is_refused_before_anything_runs() {
    let model = load_replica();
    let ids = prompt_ids(&reference("outputs.safetensors"));
    let mut cache = model.make_cache();
    assert!(model.verify(&ids[..4], &mut cache, &[999]).is_err());
    assert_eq!(cache.len(), 0, "the pass did not run");
    let mut cache = model.make_cache();
    assert!(model.advance(&[u32::MAX], &mut cache).is_err());
    assert!(model.advance(&[1 << 30], &mut cache).is_err());
    assert_eq!(cache.len(), 0);
}

/// A replica as serving loads it: the loop controllers round through bf16 (`round_loop_mix`), which
/// the float32 parity replica turns off because the rounding is a staircase two implementations
/// cross at different places.
fn load_replica_rounded() -> Inkling2Model {
    Inkling2Model::load(
        &fixture_dir(),
        LoadOptions {
            round_loop_mix: true,
        },
    )
    .expect("load the replica")
}

/// The same steps with the compiled regions on and off: token by token, then over a prefill, single
/// tokens and a verify block. Returns each step's logits.
fn steps_with(model: &Inkling2Model, fused_on: bool) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let ids = prompt_ids(&reference("outputs.safetensors"));
    super::fused::fuse::with(fused_on, || {
        let mut cache = model.make_cache();
        let by_token = ids[..10]
            .iter()
            .map(|&t| host_f32(&model.forward(&[t], &mut cache, true).unwrap()))
            .collect();
        let mut cache = model.make_cache();
        let mut in_blocks = vec![host_f32(
            &model.forward(&ids[..8], &mut cache, true).unwrap(),
        )];
        for &t in &ids[8..12] {
            in_blocks.push(host_f32(&model.forward(&[t], &mut cache, true).unwrap()));
        }
        let (logits, _) = model.verify(&ids[12..16], &mut cache, &[5]).unwrap();
        in_blocks.push(host_f32(&logits));
        (by_token, in_blocks)
    })
}

fn assert_same_logits(label: &str, plain: &[Vec<f32>], fused: &[Vec<f32>]) {
    assert_eq!(plain.len(), fused.len());
    for (step, (p, f)) in plain.iter().zip(fused).enumerate() {
        assert_eq!(p.len(), f.len());
        let (mut worst, mut scale) = (0.0f32, 0.0f32);
        for (a, b) in p.iter().zip(f) {
            assert_eq!(
                a.is_finite(),
                b.is_finite(),
                "{label}, step {step}: a masked id differs"
            );
            if a.is_finite() {
                worst = worst.max((a - b).abs());
                scale = scale.max(a.abs());
            }
        }
        assert!(
            worst <= 1e-6 * scale.max(1.0),
            "{label}, step {step}: the compiled regions differ from the plain ops by {worst} of {scale}"
        );
    }
}

/// The mix kernel (`mix_kernel.rs`) gives the logits the composed mix gives: the same steps with it
/// on and off, over single tokens and over blocks of rows. It sums in another order than the
/// composed matmuls, so one mix differs by float32 rounding (the unit test bounds that, at 6e-7) and
/// the 48 visits of the replica amplify it a thousandfold: the bound here, 2e-3 of the largest
/// logit, is where the reference-parity tests put theirs, and a wrong formula moves the logits by
/// 1e-2 or more. The replica as serving runs it rounds the loop controllers through bf16, a
/// staircase that float32 noise crosses at a few elements and the 48 visits turn into jumps of
/// order one (any change of numerics does, and the continuation of the real model can part at a near
/// tie for the same reason), so there no bound means anything: that replica only checks that the
/// kernel runs and the logits stay finite. A run with the kernel on launches it, one with it off does
/// not.
#[test]
fn the_mix_kernel_gives_the_logits_the_composed_mix_gives() {
    for (name, model, bound) in [
        ("replica", load_replica(), Some(2e-3f32)),
        ("rounded replica", load_replica_rounded(), None),
    ] {
        let before = super::mix_kernel::launches();
        let off = super::mix_kernel::kernel::with(false, || steps_with(&model, true));
        assert_eq!(
            super::mix_kernel::launches(),
            before,
            "{name}: the kernel ran while it was off"
        );
        // The fold off: the layer's mix then goes through `AttnResQuery::mix`, the choice between the
        // kernel and the compiled form that this test is about (the fold has its own test, and with
        // it on the kernel is launched from another place, which would hide a broken choice here).
        let on = super::mix_kernel::kernel::with(true, || {
            super::mix_kernel::fold::with(false, || steps_with(&model, true))
        });
        assert!(
            super::mix_kernel::launches() > before,
            "{name}: the kernel never ran while it was on"
        );
        for (label, a, b) in [("single tokens", &off.0, &on.0), ("blocks", &off.1, &on.1)] {
            assert_eq!(a.len(), b.len());
            for (step, (p, k)) in a.iter().zip(b).enumerate() {
                let (mut worst, mut scale) = (0.0f32, 0.0f32);
                for (x, y) in p.iter().zip(k) {
                    assert_eq!(
                        x.is_finite(),
                        y.is_finite(),
                        "{name}, {label}, step {step}: a masked id differs"
                    );
                    if x.is_finite() {
                        worst = worst.max((x - y).abs());
                        scale = scale.max(x.abs());
                    }
                }
                eprintln!(
                    "{name}, {label}, step {step}: mix kernel differs by {worst:.2e} of {scale:.1}"
                );
                if let Some(bound) = bound {
                    assert!(
                        worst <= bound * scale.max(1.0),
                        "{name}, {label}, step {step}: the mix kernel differs from the composed mix by {worst} of {scale}"
                    );
                }
            }
        }
    }
}

/// The router kernel (`route_kernel.rs`) gives the logits the composed router gives: the same steps
/// with it on and off, over single tokens and over blocks of rows. It picks the same experts with
/// the same weights to float32 rounding (the unit test bounds that) but returns them best first and
/// not in `argpartition`'s order, so the experts' weighted sum adds in another order: the float32
/// replica is bounded at the reference-parity tests' 2e-3 of the largest logit, and the replica as
/// serving runs it, whose bf16 loop rounding turns any change of numerics into jumps of order one,
/// only has to run and stay finite. A run with the kernel on launches it, one with it off does not.
#[test]
fn the_route_kernel_gives_the_logits_the_composed_router_gives() {
    for (name, model, bound) in [
        ("replica", load_replica(), Some(2e-3f32)),
        ("rounded replica", load_replica_rounded(), None),
    ] {
        let before = super::route_kernel::launches();
        let off = super::route_kernel::kernel::with(false, || steps_with(&model, true));
        assert_eq!(
            super::route_kernel::launches(),
            before,
            "{name}: the kernel ran while it was off"
        );
        let on = super::route_kernel::kernel::with(true, || steps_with(&model, true));
        assert!(
            super::route_kernel::launches() > before,
            "{name}: the kernel never ran while it was on"
        );
        // Only decode-sized passes take the kernel: a pass over more rows than that keeps the composed router.
        let launched = super::route_kernel::launches();
        super::route_kernel::kernel::with(true, || {
            let mut cache = model.make_cache();
            let ids = vec![1u32; super::ops::DECODE_ROWS + 1];
            host_f32(&model.forward(&ids, &mut cache, true).unwrap());
        });
        assert_eq!(
            super::route_kernel::launches(),
            launched,
            "{name}: the kernel took a pass of more than {} rows",
            super::ops::DECODE_ROWS
        );
        for (label, a, b) in [("single tokens", &off.0, &on.0), ("blocks", &off.1, &on.1)] {
            assert_eq!(a.len(), b.len());
            for (step, (p, k)) in a.iter().zip(b).enumerate() {
                let (mut worst, mut scale) = (0.0f32, 0.0f32);
                for (x, y) in p.iter().zip(k) {
                    assert_eq!(
                        x.is_finite(),
                        y.is_finite(),
                        "{name}, {label}, step {step}: a masked id differs"
                    );
                    if x.is_finite() {
                        worst = worst.max((x - y).abs());
                        scale = scale.max(x.abs());
                    }
                }
                eprintln!(
                    "{name}, {label}, step {step}: route kernel differs by {worst:.2e} of {scale:.1}"
                );
                if let Some(bound) = bound {
                    assert!(
                        worst <= bound * scale.max(1.0),
                        "{name}, {label}, step {step}: the route kernel differs from the composed router by {worst} of {scale}"
                    );
                }
            }
        }
    }
}

/// The fold (`mix_kernel::fold`) is an `Optimization`: the logits with it on are the bits they are
/// with it off (the mix kernel on both times), over single tokens and over blocks of rows, on the
/// float32 replica and on the replica as serving runs it (bf16 loop rounding, where any change of
/// numerics would show as jumps of order one). A run with the fold on launches the normalising
/// kernel, one with it off does not, and a pass of more rows than a decode takes never does.
#[test]
fn the_folded_mix_gives_the_logits_the_unfolded_kernel_gives() {
    for (name, model) in [
        ("replica", load_replica()),
        ("rounded replica", load_replica_rounded()),
    ] {
        let run = |fold: bool| {
            super::mix_kernel::kernel::with(true, || {
                super::mix_kernel::fold::with(fold, || steps_with(&model, true))
            })
        };
        let before = super::mix_kernel::folded_launches();
        let off = run(false);
        assert_eq!(
            super::mix_kernel::folded_launches(),
            before,
            "{name}: the fold ran while it was off"
        );
        let on = run(true);
        assert!(
            super::mix_kernel::folded_launches() > before,
            "{name}: the fold never ran while it was on"
        );
        let launched = super::mix_kernel::folded_launches();
        super::mix_kernel::kernel::with(true, || {
            super::mix_kernel::fold::with(true, || {
                let mut cache = model.make_cache();
                let ids = vec![1u32; super::ops::DECODE_ROWS + 1];
                host_f32(&model.forward(&ids, &mut cache, true).unwrap());
            })
        });
        assert_eq!(
            super::mix_kernel::folded_launches(),
            launched,
            "{name}: the fold took a pass of more than {} rows",
            super::ops::DECODE_ROWS
        );
        let bits = |steps: &[Vec<f32>]| -> Vec<Vec<u32>> {
            steps
                .iter()
                .map(|s| s.iter().map(|x| x.to_bits()).collect())
                .collect()
        };
        assert_eq!(
            bits(&off.0),
            bits(&on.0),
            "{name}: single tokens: the fold changed a logit"
        );
        assert_eq!(
            bits(&off.1),
            bits(&on.1),
            "{name}: blocks: the fold changed a logit"
        );
    }
}

/// The convolution kernel (`conv_kernel.rs`) is an `Optimization`: the logits with it on are the bits
/// they are with it off (the compiled regions run instead), over single tokens and over blocks of
/// rows, on the float32 replica and on the replica as serving runs it. A run with the kernel on
/// launches it after the attention and after the MLP, one with it off launches neither, and a pass of
/// more rows than a decode takes never does.
#[test]
fn the_conv_kernel_gives_the_logits_the_compiled_regions_give() {
    for (name, model) in [
        ("replica", load_replica()),
        ("rounded replica", load_replica_rounded()),
    ] {
        let run = |on: bool| super::conv_kernel::kernel::with(on, || steps_with(&model, true));
        let before = super::conv_kernel::launches();
        let off = run(false);
        assert_eq!(
            super::conv_kernel::launches(),
            before,
            "{name}: the kernel ran while it was off"
        );
        let on = run(true);
        let after = super::conv_kernel::launches();
        assert!(
            after[0] > before[0],
            "{name}: the kernel never ran after the attention while it was on"
        );
        assert!(
            after[1] > before[1],
            "{name}: the kernel never ran after the MLP while it was on"
        );
        let launched = super::conv_kernel::launches();
        super::conv_kernel::kernel::with(true, || {
            let mut cache = model.make_cache();
            let ids = vec![1u32; super::ops::DECODE_ROWS + 1];
            host_f32(&model.forward(&ids, &mut cache, true).unwrap());
        });
        assert_eq!(
            super::conv_kernel::launches(),
            launched,
            "{name}: the kernel took a pass of more than {} rows",
            super::ops::DECODE_ROWS
        );
        let bits = |steps: &[Vec<f32>]| -> Vec<Vec<u32>> {
            steps
                .iter()
                .map(|s| s.iter().map(|x| x.to_bits()).collect())
                .collect()
        };
        assert_eq!(
            bits(&off.0),
            bits(&on.0),
            "{name}: single tokens: the kernel changed a logit"
        );
        assert_eq!(
            bits(&off.1),
            bits(&on.1),
            "{name}: blocks: the kernel changed a logit"
        );
    }
}

/// The local projection kernel (`project_kernel.rs`) is an `Optimization`: the logits with it on are
/// the bits they are with it off (the compiled region runs instead), over single tokens and over
/// blocks of rows, on the float32 replica and on the replica as serving runs it (its local layers run
/// both the single-input projection and, in the loop's later passes, the stacked one). A run with the
/// kernel on launches it, one with it off does not, and a pass of more rows than a decode takes never
/// does.
#[test]
fn the_project_kernel_gives_the_logits_the_compiled_region_gives() {
    for (name, model) in [
        ("replica", load_replica()),
        ("rounded replica", load_replica_rounded()),
    ] {
        let run = |on: bool| super::project_kernel::kernel::with(on, || steps_with(&model, true));
        let before = super::project_kernel::launches();
        let off = run(false);
        assert_eq!(
            super::project_kernel::launches(),
            before,
            "{name}: the kernel ran while it was off"
        );
        let on = run(true);
        assert!(
            super::project_kernel::launches() > before,
            "{name}: the kernel never ran while it was on"
        );
        let launched = super::project_kernel::launches();
        super::project_kernel::kernel::with(true, || {
            let mut cache = model.make_cache();
            let ids = vec![1u32; super::ops::DECODE_ROWS + 1];
            host_f32(&model.forward(&ids, &mut cache, true).unwrap());
        });
        assert_eq!(
            super::project_kernel::launches(),
            launched,
            "{name}: the kernel took a pass of more than {} rows",
            super::ops::DECODE_ROWS
        );
        let bits = |steps: &[Vec<f32>]| -> Vec<Vec<u32>> {
            steps
                .iter()
                .map(|s| s.iter().map(|x| x.to_bits()).collect())
                .collect()
        };
        assert_eq!(
            bits(&off.0),
            bits(&on.0),
            "{name}: single tokens: the kernel changed a logit"
        );
        assert_eq!(
            bits(&off.1),
            bits(&on.1),
            "{name}: blocks: the kernel changed a logit"
        );
    }
}

/// A block added after a stack was built is in the next stack: the stack is kept between mixes, and
/// a stale one would drop the block a boundary layer just added from every later mix.
#[test]
fn a_block_added_after_a_stack_is_in_the_next_one() {
    let block = |v: f32| Array::from_slice(&[v; 16], &[1, 2, 8]);
    let mut blocks = super::model::Blocks::default();
    assert!(blocks.stacked().unwrap().is_none(), "no blocks, no stack");
    blocks.push(block(1.0));
    assert_eq!(blocks.stacked().unwrap().unwrap().shape(), &[2, 1, 8]);
    blocks.push(block(2.0));
    let stacked = blocks.stacked().unwrap().unwrap().clone();
    assert_eq!(stacked.shape(), &[2, 2, 8], "the new block is in the stack");
    assert_eq!(
        host_f32(&stacked)[8..16],
        [2.0; 8],
        "after the first, in order"
    );
    // A copy keeps its own list: the prelude's blocks are restored for each pass of the loop.
    let mut other = blocks.clone();
    other.push(block(3.0));
    assert_eq!(other.stacked().unwrap().unwrap().shape(), &[2, 3, 8]);
    assert_eq!(blocks.stacked().unwrap().unwrap().shape(), &[2, 2, 8]);
}

/// The regions `fused.rs` compiles, under the names it counts them by.
const REGIONS: [&str; 7] = [
    "swiglu",
    "local projection",
    "global projection",
    "conv residual",
    "conv next",
    "mix messages",
    "attention-residual mix",
];

/// Runs `f` with the kernels (the mix, the router, the convolution and the local projection) off. The
/// compiled regions are an `Optimization`: bit for bit the plain ops, and the compiled
/// attention-residual mix, convolutions and local projection are the forms a pass takes when their
/// kernels are off. The tests of that contract compare the composed forms, which the kernels replace
/// when they are on (as they are by default).
fn without_kernels<R>(f: impl FnOnce() -> R) -> R {
    super::mix_kernel::kernel::with(false, || {
        super::route_kernel::kernel::with(false, || {
            super::conv_kernel::kernel::with(false, || {
                super::project_kernel::kernel::with(false, f)
            })
        })
    })
}

/// How many times each region has been served since `before`.
fn served_since(before: &BTreeMap<&'static str, usize>) -> Vec<(&'static str, usize)> {
    let now = super::fused::served();
    REGIONS
        .into_iter()
        .map(|name| {
            let then = before.get(name).copied().unwrap_or(0);
            (name, now.get(name).copied().unwrap_or(0) - then)
        })
        .collect()
}

fn assert_every_region_served(label: &str, before: &BTreeMap<&'static str, usize>) {
    for (name, ran) in served_since(before) {
        assert!(ran > 0, "{label}: the {name} region was never served");
    }
}

/// The compiled regions (`fused.rs`) run the plain ops' arithmetic in the same order and dtypes:
/// the same steps with them on and off give the same logits (bit for bit on the replica; the bound
/// allows float noise on another GPU), over single tokens and over blocks of rows, with the loop
/// controllers' bf16 rounding off (the parity replica) and on (as serving runs them). Every region
/// is also served: a run with them on counts calls of each, one with them off counts none. Both
/// models load in one process, the second after the first has compiled its regions.
#[test]
fn the_compiled_regions_give_the_logits_the_plain_ops_give() {
    without_kernels(|| {
        for (label, model) in [
            ("replica", load_replica()),
            ("rounded replica", load_replica_rounded()),
        ] {
            let before = super::fused::served();
            let off = steps_with(&model, false);
            assert_eq!(
                super::fused::served(),
                before,
                "{label}: the plain run served a region"
            );
            let on = steps_with(&model, true);
            assert_every_region_served(label, &before);
            assert_same_logits(&format!("{label}, token by token"), &off.0, &on.0);
            assert_same_logits(&format!("{label}, in blocks"), &off.1, &on.1);
        }
    });
}

/// The replica with other clip thresholds in its config: the same shapes and weights, other
/// constants for the compiled projection regions. The directory lives as long as the returned guard.
fn load_replica_with_clips(qk: f32, rel: f32) -> (Inkling2Model, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let config = std::fs::read_to_string(fixture_dir().join("config.json")).expect("config.json");
    let edit = |text: &str, key: &str, value: f32| {
        let at = text.find(&format!("\"{key}\":")).expect("the key is there");
        let end = at + text[at..].find(',').expect("the value ends");
        format!("{}\"{key}\": {value:?}{}", &text[..at], &text[end..])
    };
    let config = edit(
        &edit(&config, "qk_l2_clip_threshold", qk),
        "rel_l2_clip_threshold",
        rel,
    );
    std::fs::write(dir.path().join("config.json"), config).unwrap();
    std::fs::copy(
        fixture_dir().join("model.safetensors"),
        dir.path().join("model.safetensors"),
    )
    .unwrap();
    let model = Inkling2Model::load(
        dir.path(),
        LoadOptions {
            round_loop_mix: true,
        },
    )
    .expect("load the replica with other clips");
    (model, dir)
}

/// The largest difference between two runs' logits (finite entries).
fn worst_difference(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| x.iter().zip(y))
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// A compiled region is keyed on the constants it was traced with, so a model whose layers have
/// other constants (here other clip thresholds, with the same shapes MLX also keys on) gets graphs
/// of its own in the same process, and the first model's are untouched. Run in the order A, B, A,
/// each region is served compiled for B, B's compiled logits equal its plain ones, and A's logits
/// are the same before and after B ran. (RED if the constants were left out of the key: B would
/// replay A's graphs, or A the ones B traced.) The configs do differ: the clip thresholds move the
/// logits, so the test cannot pass by the two models computing the same thing.
#[test]
fn a_model_with_other_constants_is_served_graphs_of_its_own() {
    without_kernels(|| {
        let a = load_replica_rounded();
        let (b, _dir) = load_replica_with_clips(0.4, 0.2);
        let a_first = steps_with(&a, true);
        let before_b = super::fused::served();
        let b_compiled = steps_with(&b, true);
        assert_every_region_served("the second model", &before_b);
        let a_again = steps_with(&a, true);
        let b_plain = steps_with(&b, false);
        assert_same_logits("the second model, compiled", &b_plain.0, &b_compiled.0);
        assert_same_logits(
            "the second model, compiled, in blocks",
            &b_plain.1,
            &b_compiled.1,
        );
        assert_same_logits(
            "the first model after the second ran",
            &a_first.0,
            &a_again.0,
        );
        assert!(
            worst_difference(&a_first.0, &b_compiled.0) > 1e-3,
            "the clip thresholds move nothing, so this test proves nothing"
        );
    });
}

/// A block's mask geometry is shared by the layer visits of a step (`ops::band_geometry`): a sliding
/// and a global layer on the same shapes, called alternately with the geometries kept, give what each
/// gives with none kept. (RED if the sharing keyed on the shapes alone: the second kind would read
/// the first's window.)
#[test]
fn a_blocks_mask_geometry_is_shared_only_by_layers_that_would_build_the_same() {
    use super::ops::{banded_sink_attention, forget_band_geometry};
    let filled = |n: usize, mul: usize, shape: &[i32]| {
        let data: Vec<f32> = (0..n)
            .map(|i| ((i * mul + 3) % 101) as f32 / 50.0 - 1.0)
            .collect();
        Array::from_slice(&data, shape)
    };
    let (b, hq, hkv, d, length, s, r) = (1, 4, 2, 8, 3, 11, 5);
    let q = filled((b * hq * length * d) as usize, 7919, &[b, hq, length, d]);
    let k = filled((b * hkv * s * d) as usize, 6073, &[b, hkv, s, d]);
    let v = filled((b * hkv * s * d) as usize, 4421, &[b, hkv, s, d]);
    let rel = filled((b * length * hq * r) as usize, 3331, &[b, length, hq, r]);
    let sink = filled((b * length * hq) as usize, 1229, &[b, length, hq]);
    let run = |window: usize| {
        let out = banded_sink_attention(&q, &k, &v, &rel, &sink, 0.35, window).unwrap();
        host_f32(&out)
    };
    forget_band_geometry();
    let sliding = run(r as usize);
    forget_band_geometry();
    let global = run(0);
    let alone = (sliding, global);
    forget_band_geometry();
    let interleaved = [run(r as usize), run(0), run(r as usize), run(0)];
    assert_eq!(interleaved[0], alone.0, "sliding, first");
    assert_eq!(interleaved[1], alone.1, "global after sliding");
    assert_eq!(interleaved[2], alone.0, "sliding after global");
    assert_eq!(interleaved[3], alone.1, "global, again");
    assert_ne!(alone.0, alone.1, "the two kinds are told apart by the data");
}

#[test]
#[ignore = "temporary probe"]
fn probe_projection_region_dispatches() {
    use mlx_rs::Dtype;
    let (nh, nkv, hd, dr, ext) = (24i32, 8i32, 128i32, 16i32, 1024i32);
    let cfg = super::regions::ProjCfg {
        heads: nh,
        kv_heads: nkv,
        head_dim: hd,
        d_rel: dr,
        qk_clip: 22.627417,
        rel_clip: 8.0,
        conv_scale: 0.4472136,
    };
    let (dq, dkv, drr) = (nh * hd, nkv * hd, nh * dr);
    let total = dq + 2 * dkv + drr + dq + nh;
    let mk = |shape: &[i32], dt: Dtype| {
        let n: i32 = shape.iter().product();
        let v: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
        Array::from_slice(&v, shape).as_dtype(dt).unwrap()
    };
    for (rows, dt) in [(1, Dtype::Bfloat16), (1, Dtype::Float32)] {
        let big = mk(&[1, rows, total], dt);
        use mlx_rs::ops::indexing::TryIndexOp;
        let (c0, c1, c2, c3) = (dq, dq + dkv, dq + 2 * dkv, dq + 2 * dkv + drr);
        let q = big.try_index((.., .., ..c0)).unwrap();
        let k = big.try_index((.., .., c0..c1)).unwrap();
        let v = big.try_index((.., .., c1..c2)).unwrap();
        let r = big.try_index((.., .., c2..c3)).unwrap();
        let ks = mk(&[1, 3, dkv], Dtype::Float32);
        let vs = mk(&[1, 3, dkv], Dtype::Float32);
        let kc = mk(&[dkv, 4], dt);
        let vc = mk(&[dkv, 4], dt);
        let qs = mk(&[nh, hd], dt);
        let rs = mk(&[nh, dr], dt);
        let rp = mk(&[dr, ext], dt);
        let args: Vec<&Array> = vec![&q, &k, &v, &r, &ks, &vs, &kc, &vc, &qs, &rs, &rp];
        type Run = fn(&super::regions::ProjCfg, &[&Array]) -> anyhow::Result<[Array; 6]>;
        let runs: [(&str, Run); 2] = [
            ("compiled", super::fused::project_local),
            ("plain", super::regions::project_local),
        ];
        for (name, f) in runs {
            let warm = f(&cfg, &args).unwrap();
            for a in &warm {
                a.eval().unwrap();
            }
            let (_, d0) = mlx_rs::metal::cmd_buffer_stats().unwrap_or_default();
            let n = 10;
            for _ in 0..n {
                let out = f(&cfg, &args).unwrap();
                for a in &out {
                    a.eval().unwrap();
                }
            }
            let (_, d1) = mlx_rs::metal::cmd_buffer_stats().unwrap_or_default();
            eprintln!(
                "PROBE {name} {dt:?} rows {rows}: {:.1} dispatches a call",
                (d1 - d0) as f64 / n as f64
            );
        }
    }
}
