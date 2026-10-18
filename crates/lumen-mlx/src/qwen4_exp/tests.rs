//! Parity of the port with the reference implementation, on a small
//! random-weights replica of the checkpoint (`scripts/golden/qwen4exp_tiny_fixture.py`
//! builds it with the reference code and `tests/fixtures/qwen4exp_tiny/` holds
//! the result: 8 layers, 16 experts, the same tensor names, 4-bit affine
//! weights, and float32 scales so both sides run in f32).
//!
//! The replica exercises every mechanism of the shipped model: gated-DeltaNet
//! and attention layers, the QSA indexer (its budget is 16 keys; the prompt is
//! 40), hyper-connections, the n-gram injection with an EOS inside the prompt,
//! top-4 routing over 16 experts with biases, and the shared expert.

use std::collections::HashMap;
use std::path::PathBuf;

use mlx_rs::Array;

use super::config::Qwen4ExpConfig;
use super::model::{LoadOptions, NgramSource, Qwen4ExpModel};
use super::ngram::NgramHasher;

/// The committed replica. A missing fixture fails the test: a skip here would
/// read as a pass.
fn fixture_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4exp_tiny");
    assert!(
        dir.join("config.json").is_file(),
        "{} is missing; regenerate it with scripts/golden/qwen4exp_tiny_fixture.py",
        dir.display()
    );
    dir
}

fn reference(dir: &std::path::Path) -> HashMap<String, Array> {
    Array::load_safetensors(dir.join("reference/outputs.safetensors")).expect("reference outputs")
}

fn host_f32(a: &Array) -> Vec<f32> {
    let a = a.as_dtype(mlx_rs::Dtype::Float32).unwrap();
    a.eval().unwrap();
    a.try_as_slice::<f32>().unwrap().to_vec()
}

/// (max |a-b|, min cosine over rows of `width`) between two flat buffers.
fn compare(a: &[f32], b: &[f32], width: usize) -> (f32, f32) {
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

#[test]
fn the_ngram_rows_match_the_reference() {
    let dir = fixture_dir();
    let cfg = Qwen4ExpConfig::load(&dir).unwrap();
    let weights = Array::load_safetensors(dir.join("model.safetensors")).unwrap();
    let ints = |name: &str| -> Vec<u64> {
        let a = &weights[name];
        a.eval().unwrap();
        a.try_as_slice::<i64>()
            .unwrap()
            .iter()
            .map(|&v| v as u64)
            .collect()
    };
    let base = "language_model.model.layers.1.ple.ple_embedding";
    let hasher = NgramHasher::new(
        cfg.ngram_size,
        cfg.heads_per_ngram,
        cfg.eos_token_id,
        ints(&format!("{base}.layer_multipliers")),
        ints(&format!("{base}.ngram_heads_vocab_sizes")),
        ints(&format!("{base}.ngram_heads_offsets")),
    )
    .unwrap();

    let want = reference(&dir);
    let ids = want["ids"].as_slice::<i32>().to_vec();
    let full = &ids[..48];
    let gids = want["ngram_gids"].as_slice::<i64>().to_vec();
    let mut history = vec![cfg.eos_token_id; cfg.ngram_size - 1];
    history.extend(full.iter().map(|&t| t as u32));
    let got = hasher.row_ids(&history, full.len());
    assert_eq!(got.len(), gids.len());
    let bad = got.iter().zip(&gids).position(|(a, b)| *a as i64 != *b);
    assert!(bad.is_none(), "first mismatch at {bad:?}");
}

#[test]
fn a_full_prefill_matches_the_reference_logits() {
    let dir = fixture_dir();
    let model = Qwen4ExpModel::load(&dir, None).expect("load tiny replica");
    let vocab = model.config().vocab_size;
    let want = reference(&dir);
    let ids: Vec<u32> = want["ids"]
        .as_slice::<i32>()
        .iter()
        .map(|&t| t as u32)
        .collect();

    let mut cache = model.make_cache();
    let logits = model.forward(&ids, &mut cache, false).expect("forward");
    let got = host_f32(&logits);
    let expected = host_f32(&want["logits_full"]);
    let (max_abs, min_cos) = compare(&got, &expected, vocab);
    eprintln!("prefill: max |diff| {max_abs:.3e}, min cosine {min_cos:.6}");
    assert_eq!(
        argmax_rows(&got, vocab),
        argmax_rows(&expected, vocab),
        "top-1 per position"
    );
    assert!(min_cos > 0.9999, "min cosine {min_cos}");
    assert!(max_abs < 5e-3, "max |diff| {max_abs}");
}

#[test]
fn prefill_then_single_token_decode_matches_the_reference() {
    let dir = fixture_dir();
    let model = Qwen4ExpModel::load(&dir, None).expect("load tiny replica");
    let vocab = model.config().vocab_size;
    let want = reference(&dir);
    let ids: Vec<u32> = want["ids"]
        .as_slice::<i32>()
        .iter()
        .map(|&t| t as u32)
        .collect();
    let expected = host_f32(&want["logits_split"]);

    let mut cache = model.make_cache();
    let mut got = host_f32(&model.forward(&ids[..40], &mut cache, false).unwrap());
    for t in 40..ids.len() {
        let step = model.forward(&ids[t..t + 1], &mut cache, false).unwrap();
        got.extend(host_f32(&step));
    }
    let (max_abs, min_cos) = compare(&got, &expected, vocab);
    eprintln!("prefill + decode: max |diff| {max_abs:.3e}, min cosine {min_cos:.6}");
    assert_eq!(
        argmax_rows(&got, vocab),
        argmax_rows(&expected, vocab),
        "top-1 per position"
    );
    assert!(min_cos > 0.9999, "min cosine {min_cos}");
    assert!(max_abs < 5e-3, "max |diff| {max_abs}");
    assert_eq!(cache.len(), ids.len());
}

/// The same prompt fed in uneven chunks must land on the same logits as one
/// prefill: every carried state (convolution history, recurrent state, KV and
/// indexer keys, the n-gram tail and its convolution) crosses the boundaries.
#[test]
fn chunked_prefill_matches_a_single_prefill() {
    let dir = fixture_dir();
    let model = Qwen4ExpModel::load(&dir, None).expect("load tiny replica");
    let vocab = model.config().vocab_size;
    let want = reference(&dir);
    let ids: Vec<u32> = want["ids"]
        .as_slice::<i32>()
        .iter()
        .map(|&t| t as u32)
        .collect();

    let mut cache = model.make_cache();
    let mut got = Vec::new();
    let mut at = 0;
    for len in [13, 1, 20, 7, 7] {
        got.extend(host_f32(
            &model
                .forward(&ids[at..at + len], &mut cache, false)
                .unwrap(),
        ));
        at += len;
    }
    assert_eq!(at, ids.len());
    let expected = host_f32(&want["logits_full"]);
    let (max_abs, min_cos) = compare(&got, &expected, vocab);
    eprintln!("chunked prefill: max |diff| {max_abs:.3e}, min cosine {min_cos:.6}");
    assert_eq!(argmax_rows(&got, vocab), argmax_rows(&expected, vocab));
    assert!(min_cos > 0.9999 && max_abs < 5e-3);
}

/// The offload engine in front of the replica's experts.
///
/// While the pool has fewer than ~41 slots, as many as the replica's 16
/// experts per layer or a little more, logits are bit-identical to the resident
/// model. With more slots MLX's sorted `gather_qmm` picks a different tiling for
/// the prefill rows (measured in plain MLX, no offload: 16 vs 41+ experts give
/// logits that differ by ~1e-5), so the last bits move and only closeness can be
/// asserted. The shipped models are far above the threshold on both sides (256
/// and 512 experts per layer, and a pool is never narrower than one layer), where
/// the identity tests on the real checkpoint hold bit for bit.
#[test]
fn offloaded_experts_match_the_resident_model() {
    let dir = fixture_dir();
    let resident = Qwen4ExpModel::load(&dir, None).expect("load tiny replica");
    let vocab = resident.config().vocab_size;
    let want = reference(&dir);
    let ids: Vec<u32> = want["ids"]
        .as_slice::<i32>()
        .iter()
        .map(|&t| t as u32)
        .collect();

    let run = |model: &Qwen4ExpModel| -> Vec<f32> {
        let mut cache = model.make_cache();
        let mut out = host_f32(&model.forward(&ids[..40], &mut cache, false).unwrap());
        for t in 40..ids.len() {
            out.extend(host_f32(
                &model.forward(&ids[t..t + 1], &mut cache, false).unwrap(),
            ));
        }
        out
    };
    let baseline = run(&resident);
    drop(resident);
    // (budget in bytes, must be bit-identical). An expert of the replica is ~9.9 KB.
    for (budget, exact) in [
        (1u64, true),
        (20 * 9_900, true),
        (34 * 9_900, true),
        (u64::MAX / 2, false),
    ] {
        let model = Qwen4ExpModel::load(&dir, Some(budget)).expect("load with offload");
        let got = run(&model);
        let (hits, misses) = model.offload_stats().expect("offload on");
        let (max_abs, min_cos) = compare(&got, &baseline, vocab);
        eprintln!("budget {budget}: hits {hits} misses {misses}, max |diff| {max_abs:e}");
        if exact {
            assert!(
                got == baseline,
                "budget {budget}: logits differ from the resident model"
            );
        } else {
            assert!(
                max_abs < 5e-5 && min_cos > 0.999_999,
                "budget {budget}: {max_abs:e} / {min_cos}"
            );
            assert_eq!(argmax_rows(&got, vocab), argmax_rows(&baseline, vocab));
        }
    }
}

/// Rows read from the checkpoint's files one at a time give the logits the
/// resident table does, and a second pass over the same tokens reads nothing.
#[test]
fn serving_the_ngram_table_from_disk_matches_holding_it_in_memory() {
    let dir = fixture_dir();
    let want = reference(&dir);
    let ids: Vec<u32> = want["ids"]
        .as_slice::<i32>()
        .iter()
        .map(|&t| t as u32)
        .collect();
    let options = |ngram| LoadOptions {
        expert_budget_bytes: None,
        ngram,
    };
    let run = |model: &Qwen4ExpModel| -> Vec<f32> {
        let mut cache = model.make_cache();
        host_f32(&model.forward(&ids, &mut cache, false).unwrap())
    };
    let resident = Qwen4ExpModel::load_with(&dir, options(NgramSource::Resident)).unwrap();
    assert!(resident.ngram_stats().is_none());
    let baseline = run(&resident);
    drop(resident);

    let from_disk = Qwen4ExpModel::load_with(&dir, options(NgramSource::File)).unwrap();
    let first = run(&from_disk);
    assert!(
        first == baseline,
        "file-backed n-gram rows change the logits"
    );
    let after_first = from_disk.ngram_stats().expect("file-backed");
    assert_eq!(after_first.requested, 48 * 4);
    assert!(after_first.read_from_disk > 0 && after_first.read_from_disk <= after_first.requested);
    let again = run(&from_disk);
    assert!(again == baseline);
    let after_second = from_disk.ngram_stats().unwrap();
    assert_eq!(
        after_second.read_from_disk, after_first.read_from_disk,
        "the second pass should have been served from the row cache"
    );
}

/// The server loads the model on one thread and runs it on another. MLX streams
/// are per thread, so everything the forward pass touches must work from a thread
/// that did not build the model.
#[test]
fn the_forward_pass_runs_on_a_thread_that_did_not_load_the_model() {
    let dir = fixture_dir();
    let model = Qwen4ExpModel::load(&dir, None).expect("load tiny replica");
    let want = reference(&dir);
    let ids: Vec<u32> = want["ids"]
        .as_slice::<i32>()
        .iter()
        .map(|&t| t as u32)
        .collect();
    let expected = host_f32(&want["logits_full"]);
    let vocab = model.config().vocab_size;
    let got = std::thread::spawn(move || {
        let mut cache = model.make_cache();
        host_f32(
            &model
                .forward(&ids, &mut cache, false)
                .expect("forward on another thread"),
        )
    })
    .join()
    .expect("worker thread");
    let (max_abs, min_cos) = compare(&got, &expected, vocab);
    assert!(
        min_cos > 0.9999 && max_abs < 5e-3,
        "{max_abs:e} / {min_cos}"
    );
}

/// A state written to the disk format and read back continues exactly like the
/// state it was taken from: the convolution and recurrent states, the KV buffers,
/// the indexer's keys, the n-gram convolution history and the n-gram tail.
#[test]
fn a_state_round_tripped_through_the_disk_format_resumes_bit_for_bit() {
    use crate::kv_disk::{KvManifest, read_lkv, write_lkv};
    let dir = fixture_dir();
    let model = Qwen4ExpModel::load(&dir, None).expect("load tiny replica");
    let want = reference(&dir);
    let ids: Vec<u32> = want["ids"]
        .as_slice::<i32>()
        .iter()
        .map(|&t| t as u32)
        .collect();

    let mut original = model.make_cache();
    model.forward(&ids[..40], &mut original, true).unwrap();
    let (layers, records) = original.to_records().unwrap();
    let manifest = KvManifest {
        model_fingerprint: "test".into(),
        created_at_unix: 0,
        position: 40,
        prefix_tokens: ids[..40].to_vec(),
        last_token: None,
        is_deep: true,
        layers,
    };
    let mut bytes = Vec::new();
    write_lkv(&mut bytes, &manifest, &records).unwrap();
    let (manifest, records) = read_lkv(&mut bytes.as_slice()).unwrap();
    let mut restored = model
        .cache_from_records(&manifest.layers, &records, &ids[..40])
        .unwrap();
    assert_eq!(restored.len(), 40);

    // Continue both: a multi-token piece (sparse indexer, n-gram history), then decode steps.
    let mut a = host_f32(&model.forward(&ids[40..44], &mut original, false).unwrap());
    let mut b = host_f32(&model.forward(&ids[40..44], &mut restored, false).unwrap());
    for t in 44..ids.len() {
        a.extend(host_f32(
            &model.forward(&ids[t..t + 1], &mut original, false).unwrap(),
        ));
        b.extend(host_f32(
            &model.forward(&ids[t..t + 1], &mut restored, false).unwrap(),
        ));
    }
    assert!(a == b, "the restored state diverged from the original");

    // A state of the wrong shape is refused, not misread.
    let mut short = manifest.layers.clone();
    short.pop();
    assert!(
        model
            .cache_from_records(&short, &records, &ids[..40])
            .is_err()
    );
}

/// Feeding the prompt layer by layer (every chunk through layer 0, then every chunk through
/// layer 1, ...) is the same computation as feeding it chunk by chunk: same logits, same
/// state, and the rollback point it leaves mid-way is the state the chunked feed has there.
/// Run against the resident model and against pools of several sizes, where the layer-major
/// path takes the staging regions (a view of exactly one layer, so the kernels are the
/// resident model's whatever the pool holds).
#[test]
fn a_layer_major_prefill_equals_a_chunk_by_chunk_one() {
    let dir = fixture_dir();
    let want = reference(&dir);
    let ids: Vec<u32> = want["ids"]
        .as_slice::<i32>()
        .iter()
        .map(|&t| t as u32)
        .collect();
    let cuts = [13usize, 14, 34, 41, 48];
    let mark = 34;
    let tail = &ids[..48];

    // Chunk by chunk, resident: the logits after the whole prompt, the logits a few decode steps
    // later, and the continuation from the state at the mark.
    let continue_from = |model: &Qwen4ExpModel, cache: &mut super::model::Qwen4ExpCache| {
        let mut out = Vec::new();
        for t in [5u32, 9, 2, 17] {
            out.extend(host_f32(&model.forward(&[t], cache, true).unwrap()));
        }
        out
    };
    let resident = Qwen4ExpModel::load(&dir, None).expect("load tiny replica");
    let chunked = |model: &Qwen4ExpModel| {
        let mut cache = model.make_cache();
        let mut at = 0;
        let mut at_mark = None;
        let mut last = None;
        for &c in &cuts {
            last = Some(host_f32(
                &model.forward(&tail[at..c], &mut cache, true).unwrap(),
            ));
            at = c;
            if c == mark {
                at_mark = Some(cache.clone());
            }
        }
        (last.unwrap(), cache, at_mark.unwrap())
    };
    let (want_logits, mut want_cache, mut want_mark) = chunked(&resident);
    let want_after = continue_from(&resident, &mut want_cache);
    let want_from_mark = {
        let mut out = host_f32(
            &resident
                .forward(&tail[mark..], &mut want_mark, true)
                .unwrap(),
        );
        out.extend(continue_from(&resident, &mut want_mark));
        out
    };

    let layer_major = |model: &Qwen4ExpModel| {
        let mut cache = model.make_cache();
        let (logits, point) = model
            .forward_pieces(tail, &mut cache, &cuts, Some(mark))
            .unwrap();
        (host_f32(&logits), cache, point.expect("a rollback point"))
    };
    // (budget, state afterwards must match too): a pool wider than ~41 slots tiles the decode
    // gathers differently from the resident model, so only the prefill is compared there.
    let budgets: [(Option<u64>, bool); 4] = [
        (None, true),
        (Some(1), true),
        (Some(34 * 9_900), true),
        (Some(u64::MAX / 2), false),
    ];
    drop(resident);
    let resident = Qwen4ExpModel::load(&dir, None).unwrap();
    for (budget, decode_exact) in budgets {
        let model = match budget {
            None => Qwen4ExpModel::load(&dir, None).unwrap(),
            Some(b) => Qwen4ExpModel::load(&dir, Some(b)).unwrap(),
        };
        let (logits, mut cache, mut point) = layer_major(&model);
        if budget.is_some() {
            let report = model.offload_report().expect("offload on");
            assert!(
                report.contains("staged:"),
                "budget {budget:?}: the layer-major feed did not take the staging path: {report}"
            );
        }
        assert_eq!(cache.len(), 48);
        assert_eq!(point.len(), mark);
        assert!(
            logits == want_logits,
            "budget {budget:?}: layer-major logits differ from the chunked feed"
        );
        if decode_exact {
            assert!(
                continue_from(&model, &mut cache) == want_after,
                "budget {budget:?}: the state after a layer-major feed differs"
            );
            let mut got = host_f32(&model.forward(&tail[mark..], &mut point, true).unwrap());
            got.extend(continue_from(&model, &mut point));
            assert!(
                got == want_from_mark,
                "budget {budget:?}: the rollback point differs from the chunked feed's state"
            );
        }
    }
    drop(resident);

    // A pool that has served decode steps holds experts of its own: the staging loads then copy
    // those across instead of reading them, and the answer must not change.
    for budget in [34 * 9_900, u64::MAX / 2] {
        let model = Qwen4ExpModel::load(&dir, Some(budget)).unwrap();
        let mut warm = model.make_cache();
        model.forward(&tail[..20], &mut warm, true).unwrap();
        for t in &tail[20..30] {
            model.forward(&[*t], &mut warm, true).unwrap();
        }
        let (logits, _, _) = layer_major(&model);
        assert!(
            logits == want_logits,
            "budget {budget}: layer-major logits differ after the pool was warmed"
        );
        let report = model.offload_report().expect("offload on");
        assert!(
            report.contains("copied from the pool"),
            "budget {budget}: no expert was copied from the warm pool: {report}"
        );
    }
}

/// A layer-major feed over more tokens than one group holds goes through in groups, and the
/// groups (with the mark inside one) give the answer the chunk-by-chunk feed gives.
#[test]
fn a_prompt_longer_than_a_group_goes_through_in_groups() {
    let dir = fixture_dir();
    let want = reference(&dir);
    let base: Vec<u32> = want["ids"]
        .as_slice::<i32>()
        .iter()
        .map(|&t| t as u32)
        .collect();
    // 48 tokens repeated: the answer need not make sense, only agree.
    let ids: Vec<u32> = base.iter().cycle().take(48 * 3).copied().collect();
    let model = Qwen4ExpModel::load(&dir, Some(34 * 9_900)).unwrap();
    let cuts: Vec<usize> = (1..=ids.len() / 16).map(|k| k * 16).collect();
    let mark = 80;

    let mut want_cache = model.make_cache();
    let mut want_logits = Vec::new();
    let mut want_point = None;
    let mut at = 0;
    for &c in &cuts {
        want_logits = host_f32(&model.forward(&ids[at..c], &mut want_cache, true).unwrap());
        at = c;
        if c == mark {
            want_point = Some(want_cache.clone());
        }
    }
    let mut want_point = want_point.unwrap();
    let want_next = host_f32(
        &model
            .forward(&ids[mark..mark + 4], &mut want_point, true)
            .unwrap(),
    );

    // Two chunks to a group (40 tokens fit two 16-token chunks, not three).
    let mut cache = model.make_cache();
    let (logits, point) = model
        .forward_pieces_in_groups(&ids, &mut cache, &cuts, Some(mark), 40)
        .unwrap();
    assert!(
        host_f32(&logits) == want_logits,
        "grouped layer-major differs from chunked"
    );
    let mut point = point.expect("a rollback point inside a group");
    assert_eq!(point.len(), mark);
    let next = host_f32(
        &model
            .forward(&ids[mark..mark + 4], &mut point, true)
            .unwrap(),
    );
    assert!(next == want_next, "the rollback point differs");
}
