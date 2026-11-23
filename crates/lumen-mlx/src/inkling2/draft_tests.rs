//! Parity of the draft head with the oMLX reference on a small random-weights replica
//! (`scripts/golden/inkling2_dspark_fixture.py` runs the reference's own `DFlashDrafter` forward and
//! `tests/fixtures/inkling2_dspark/` holds the result: three layers, hidden 64, 4 query and 2
//! key/value heads, a window of 12 (11 context rows), a block of 8, Markov rank 8, float32, bound to
//! the tiny text replica's embedding and head).
//!
//! Five cycles: the first hands over 30 rows (the ring keeps 11 of them and the positions still
//! count), the next ones 3, 1, 8 and 5, so the ring grows, wraps and takes a block's worth at once.
//! The fixture script prints how far the drafts move when each mechanism is switched off.

use std::collections::HashMap;
use std::path::PathBuf;

use mlx_rs::Array;

use super::draft::{DraftArgs, DraftModel, TapRing};
use super::model::Inkling2Model;
use super::tests::{compare, fixture_dir, host_f32, load_replica};

fn dspark_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inkling2_dspark");
    assert!(
        dir.join("config.json").is_file(),
        "{} is missing; regenerate it with scripts/golden/inkling2_dspark_fixture.py",
        dir.display()
    );
    dir
}

fn cycles() -> HashMap<String, Array> {
    Array::load_safetensors(dspark_dir().join("reference/cycles.safetensors")).expect("cycles")
}

fn load() -> (Inkling2Model, DraftModel) {
    let target = load_replica();
    let draft = DraftModel::load(&dspark_dir(), &target)
        .expect("load the draft")
        .expect("the replica declares a draft");
    (target, draft)
}

fn int(a: &Array) -> Vec<i64> {
    a.as_dtype(mlx_rs::Dtype::Int64)
        .unwrap()
        .as_slice::<i64>()
        .to_vec()
}

/// (max |a-b|, min row cosine) of two arrays of one shape, over the finite entries both have.
fn close(label: &str, got: &Array, want: &Array, tolerance: f32) {
    assert_eq!(got.shape(), want.shape(), "{label}: shape");
    let (g, w) = (host_f32(got), host_f32(want));
    assert_eq!(
        g.iter().map(|x| x.is_finite()).collect::<Vec<_>>(),
        w.iter().map(|x| x.is_finite()).collect::<Vec<_>>(),
        "{label}: where the entries are -inf"
    );
    let width = *got.shape().last().unwrap() as usize;
    let finite = |v: &[f32]| {
        v.iter()
            .map(|x| if x.is_finite() { *x } else { 0.0 })
            .collect::<Vec<_>>()
    };
    let (max_abs, min_cos) = compare(&finite(&g), &finite(&w), width);
    eprintln!("{label}: max |diff| {max_abs:.2e}, min cosine {min_cos:.7}");
    assert!(max_abs < tolerance, "{label}: max |diff| {max_abs}");
}

#[test]
fn the_draft_matches_the_reference_over_five_cycles() {
    let (target, draft) = load();
    let want = cycles();
    let n = int(&want["n_cycles"])[0] as usize;
    assert_eq!(n, 5);
    let mut ctx = draft.empty_context();
    for c in 0..n {
        let rows = &want[&format!("{c}.taps")];
        let anchor = int(&want[&format!("{c}.anchor")])[0] as u32;
        let out = draft
            .draft(&mut ctx, Some(rows), anchor, &target)
            .expect("draft");
        assert_eq!(
            ctx.total() as i64,
            int(&want[&format!("{c}.total")])[0],
            "cycle {c}: total"
        );
        for l in 0..draft.args().num_layers {
            let (k, v) = ctx.ring(l).expect("ring");
            close(
                &format!("cycle {c} layer {l} context keys"),
                k,
                &want[&format!("{c}.ring_k.{l}")],
                1e-4,
            );
            close(
                &format!("cycle {c} layer {l} context values"),
                v,
                &want[&format!("{c}.ring_v.{l}")],
                1e-4,
            );
        }
        close(
            &format!("cycle {c} block output"),
            &out.hidden,
            &want[&format!("{c}.hidden")],
            1e-3,
        );
        close(
            &format!("cycle {c} head logits"),
            &out.base_logits,
            &want[&format!("{c}.base_logits")],
            1e-3,
        );
        close(
            &format!("cycle {c} Markov logits"),
            &out.step_logits,
            &want[&format!("{c}.step_logits")],
            2e-3,
        );
        let tokens: Vec<i64> = out.tokens.iter().map(|&t| t as i64).collect();
        assert_eq!(
            tokens,
            int(&want[&format!("{c}.tokens")]),
            "cycle {c}: the drafts"
        );
    }
}

/// A context rebuilt from the raw taps a conversation kept is the context that grew a cycle at a
/// time: what a request that starts from retained caches relies on.
#[test]
fn a_context_rebuilt_from_kept_taps_is_the_one_that_grew() {
    let (_target, draft) = load();
    let want = cycles();
    let keep = draft.args().context_rows;
    let mut grown = draft.empty_context();
    let mut ring = TapRing::new();
    for c in 0..4 {
        let rows = &want[&format!("{c}.taps")];
        draft.extend(&mut grown, rows).unwrap();
        ring.push(rows, keep).unwrap();
    }
    assert_eq!(ring.end(), grown.total());
    assert_eq!(ring.len(), keep);
    let rebuilt = draft.context_from(&ring).expect("rebuild");
    assert_eq!(rebuilt.total(), grown.total());
    for l in 0..draft.args().num_layers {
        let ((gk, gv), (rk, rv)) = (grown.ring(l).unwrap(), rebuilt.ring(l).unwrap());
        close(&format!("layer {l} keys"), rk, gk, 1e-5);
        close(&format!("layer {l} values"), rv, gv, 1e-5);
    }
}

#[test]
fn a_tap_ring_keeps_the_newest_rows_and_can_be_cut_back() {
    let rows = |from: usize, to: usize| -> Array {
        let data: Vec<f32> = (from..to).flat_map(|p| [p as f32, -(p as f32)]).collect();
        Array::from_slice(&data, &[1, (to - from) as i32, 2])
    };
    let mut ring = TapRing::new();
    assert!(ring.is_empty() && ring.end() == 0);
    ring.push(&rows(0, 5), 4).unwrap();
    assert_eq!((ring.len(), ring.end()), (4, 5));
    ring.push(&rows(5, 7), 4).unwrap();
    assert_eq!((ring.len(), ring.end()), (4, 7));
    // Positions 3..7 are held; cutting to 5 leaves 3 and 4.
    ring.truncate_to(5).unwrap();
    assert_eq!((ring.len(), ring.end()), (2, 5));
    let held = host_f32(ring.rows_for_test().unwrap());
    assert_eq!(held, vec![3.0, -3.0, 4.0, -4.0]);
    // Cutting past what is held empties it but keeps the end; past the end is an error.
    ring.truncate_to(2).unwrap();
    assert_eq!((ring.len(), ring.end()), (0, 2));
    assert!(ring.truncate_to(3).is_err());
}

#[test]
fn a_draft_is_refused_when_its_files_do_not_fit_the_target() {
    let target = load_replica();
    let config = std::fs::read_to_string(dspark_dir().join("config.json")).unwrap();
    let args = |config: &str| DraftArgs::from_config_str(config, target.args());
    assert!(args(&config).unwrap().is_some());
    // No draft declared.
    assert!(
        args(r#"{"model_type": "inkling2_loop"}"#)
            .unwrap()
            .is_none()
    );
    // A head or a bias this port does not implement.
    let other_head = config.replace("\"rnn_hidden\"", "\"table\"");
    assert!(args(&other_head).is_err());
    let biased = config.replace("\"q_bias\": false", "\"q_bias\": true");
    assert!(args(&biased).is_err());
    // A width that is not the target's.
    let wide = config.replace("\"hidden_size\": 64", "\"hidden_size\": 96");
    assert!(args(&wide).is_err());
    // Tensors that are missing or the wrong shape.
    let a = args(&config).unwrap().unwrap();
    let tensors = Array::load_safetensors(dspark_dir().join("mtp.safetensors")).unwrap();
    assert!(DraftModel::from_tensors(a.clone(), tensors.clone(), &target).is_ok());
    let mut missing = tensors.clone();
    missing.remove("layers.1.self_attn.v_proj_ctx.weight");
    let err = DraftModel::from_tensors(a.clone(), missing, &target)
        .err()
        .expect("refused");
    assert!(err.to_string().contains("v_proj_ctx"), "{err}");
    let mut reshaped = tensors;
    reshaped.insert("mask_embed".into(), Array::zeros::<f32>(&[63]).unwrap());
    let err = DraftModel::from_tensors(a, reshaped, &target)
        .err()
        .expect("refused");
    assert!(err.to_string().contains("mask_embed"), "{err}");
    let _ = fixture_dir();
}

#[test]
fn a_draft_needs_context_and_rows_of_the_right_width() {
    let (target, draft) = load();
    let mut ctx = draft.empty_context();
    // No rows yet: nothing to attend to.
    assert!(draft.draft(&mut ctx, None, 1, &target).is_err());
    // Rows that are not [1, n, taps * hidden].
    let narrow = Array::zeros::<f32>(&[1, 3, 10]).unwrap();
    assert!(draft.extend(&mut ctx, &narrow).is_err());
    let batched = Array::zeros::<f32>(&[2, 3, 4 * 64]).unwrap();
    assert!(draft.extend(&mut ctx, &batched).is_err());
}

/// A draft asked for fewer tokens stops the Markov head early and proposes exactly the first
/// tokens of the full block (RED if the head's state or the previous token were not carried the
/// same way: the prefix would differ).
#[test]
fn a_draft_asked_for_fewer_tokens_proposes_the_start_of_the_full_block() {
    let (target, draft) = load();
    let want = cycles();
    let rows = want["0.taps"].clone();
    let anchor = int(&want["0.anchor"])[0] as u32;
    let mut ctx = draft.empty_context();
    draft.extend(&mut ctx, &rows).unwrap();
    let full = draft.propose(&ctx, anchor, &target).unwrap().tokens;
    assert_eq!(full.len(), draft.args().drafts_per_block());
    for count in 1..=full.len() {
        let some = draft
            .propose_up_to(&ctx, anchor, &target, count)
            .unwrap()
            .tokens;
        assert_eq!(some, full[..count], "{count} tokens");
    }
    // Asking for more than a block gives the block; asking for none is refused.
    let more = draft
        .propose_up_to(&ctx, anchor, &target, 99)
        .unwrap()
        .tokens;
    assert_eq!(more, full);
    assert!(draft.propose_up_to(&ctx, anchor, &target, 0).is_err());
}
