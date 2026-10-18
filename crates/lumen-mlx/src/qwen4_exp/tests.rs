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
use super::model::{LoadOptions, MtpMode, NgramSource, Qwen4ExpModel};
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

/// The offload engine in front of the replica's experts: logits bit-identical to the resident
/// model at every pool size, from the smallest pool (a region and a layer) to one holding every
/// expert.
///
/// MLX's sorted `gather_qmm` picks its kernel from the call's rows and the number of experts in
/// the stacked tensor it is given. The resident model gathers over one layer, the staging path
/// over a view of exactly one, so they pick alike; a pool-path call used to gather over the
/// whole pool and could pick the other kernel, whose last bits differ (logits ~1e-5 apart).
/// A call big enough to change kernels now takes the staging path whatever the pool size
/// (`offload::takes_staging_path`, with its own test on the real kernels in
/// `offload::exactness_tests`).
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
    // An expert of the replica is ~9.9 KB; a budget of 1 byte gives the smallest pool, 32 slots.
    for budget in [1u64, 20 * 9_900, 34 * 9_900, 100 * 9_900, u64::MAX / 2] {
        let model = Qwen4ExpModel::load(&dir, Some(budget)).expect("load with offload");
        let got = run(&model);
        let (hits, misses) = model.offload_stats().expect("offload on");
        let (max_abs, _) = compare(&got, &baseline, vocab);
        eprintln!("budget {budget}: hits {hits} misses {misses}, max |diff| {max_abs:e}");
        assert!(
            got == baseline,
            "budget {budget}: logits differ from the resident model (max |diff| {max_abs:e})"
        );
    }
}

/// Token streams cut into prefill pieces of many sizes: each piece is one layer call of
/// `rows x 4` index entries, and the sorted gather's kernel depends on that count against the
/// number of experts in what it gathers over, so a pool of any size must still give the
/// resident model's logits. The reference prompt routes to nearly every expert of a layer, so
/// every call is staged; a stream of one repeated token, or two alternating, routes to a few,
/// which is what sends a big call down the pool path.
#[test]
fn offloaded_experts_match_the_resident_model_for_every_call_size() {
    let dir = fixture_dir();
    let resident = Qwen4ExpModel::load(&dir, None).expect("load tiny replica");
    let want = reference(&dir);
    let prompt: Vec<u32> = want["ids"]
        .as_slice::<i32>()
        .iter()
        .map(|&t| t as u32)
        .collect();
    let streams: [(&str, Vec<u32>); 3] = [
        ("reference prompt", prompt.clone()),
        ("one repeated token", vec![prompt[3]; prompt.len()]),
        (
            "two alternating tokens",
            (0..prompt.len()).map(|i| prompt[i % 2]).collect(),
        ),
    ];
    let cuttings: [&[usize]; 6] = [
        &[48],
        &[16, 16, 16],
        &[24, 24],
        &[32, 16],
        &[20, 1, 7, 20],
        &[31, 1, 15, 1],
    ];
    let run = |model: &Qwen4ExpModel, ids: &[u32], pieces: &[usize]| -> Vec<f32> {
        let mut cache = model.make_cache();
        let mut out = Vec::new();
        let mut at = 0;
        for &n in pieces {
            out.extend(host_f32(
                &model.forward(&ids[at..at + n], &mut cache, false).unwrap(),
            ));
            at += n;
        }
        assert_eq!(at, ids.len());
        out
    };
    let baselines: Vec<Vec<Vec<f32>>> = streams
        .iter()
        .map(|(_, ids)| cuttings.iter().map(|p| run(&resident, ids, p)).collect())
        .collect();
    drop(resident);
    for budget in [1u64, 40 * 9_900, 100 * 9_900, u64::MAX / 2] {
        for ((name, ids), baselines) in streams.iter().zip(&baselines) {
            for (pieces, baseline) in cuttings.iter().zip(baselines) {
                // A fresh model per run: no run inherits another's pool.
                let model = Qwen4ExpModel::load(&dir, Some(budget)).expect("load with offload");
                assert!(
                    run(&model, ids, pieces) == *baseline,
                    "budget {budget}, {name}, pieces {pieces:?}: logits differ from the resident model"
                );
            }
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
        mtp: MtpMode::Off,
        scores_budget_bytes: 0,
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

// ---- the MTP draft head -------------------------------------------------------------------

fn mtp_outputs(dir: &std::path::Path) -> HashMap<String, Array> {
    Array::load_safetensors(dir.join("reference/mtp_outputs.safetensors")).expect("mtp outputs")
}

fn load_with_head(dir: &std::path::Path, budget: Option<u64>) -> Qwen4ExpModel {
    Qwen4ExpModel::load_with(
        dir,
        LoadOptions {
            expert_budget_bytes: budget,
            ngram: NgramSource::Auto,
            mtp: MtpMode::On,
            scores_budget_bytes: 0,
        },
    )
    .expect("load the replica with its draft head")
}

/// The head, fed the main model's own residual streams and embeddings, gives the reference's
/// logits: in one pass over the whole prompt, and as a chunk followed by single entries (the
/// cache and the sparse indexer, which only matters past 16 keys).
#[test]
fn the_draft_head_matches_the_reference() {
    // The head's experts resident, and served from the expert pool beside the trunk's.
    for budget in [None, Some(1u64), Some(u64::MAX / 2)] {
        check_the_draft_head_against_the_reference(budget);
    }
}

fn check_the_draft_head_against_the_reference(budget: Option<u64>) {
    let dir = fixture_dir();
    let model = load_with_head(&dir, budget);
    assert!(model.has_mtp());
    let want = mtp_outputs(&dir);
    let ids: Vec<u32> = want["mtp_ids"]
        .as_slice::<i32>()
        .iter()
        .map(|&t| t as u32)
        .collect();
    let rows = &ids[1..];
    let multi = &want["mtp_multi_in"];
    let vocab = model.config().vocab_size;

    let whole = model.mtp_logits_for_test(rows, multi, &[47]).unwrap();
    let got = host_f32(&whole[0]);
    let expected = host_f32(&want["mtp_logits_prefill"]);
    let (max_abs, min_cos) = compare(&got, &expected, vocab);
    eprintln!("head ({budget:?}), one pass: max |diff| {max_abs:.3e}, min cosine {min_cos:.6}");
    assert_eq!(argmax_rows(&got, vocab), argmax_rows(&expected, vocab));
    assert!(
        min_cos > 0.9999 && max_abs < 5e-3,
        "{max_abs:e} / {min_cos}"
    );

    let mut chunks = vec![39];
    chunks.extend([1; 8]);
    let parts = model.mtp_logits_for_test(rows, multi, &chunks).unwrap();
    let chunk = host_f32(&parts[0]);
    let (max_abs, min_cos) = compare(&chunk, &host_f32(&want["mtp_logits_chunk"]), vocab);
    eprintln!("head, 39-entry chunk: max |diff| {max_abs:.3e}, min cosine {min_cos:.6}");
    assert!(
        min_cos > 0.9999 && max_abs < 5e-3,
        "{max_abs:e} / {min_cos}"
    );
    let steps: Vec<f32> = parts[1..].iter().flat_map(host_f32).collect();
    let (max_abs, min_cos) = compare(&steps, &host_f32(&want["mtp_logits_steps"]), vocab);
    eprintln!("head, 8 single entries: max |diff| {max_abs:.3e}, min cosine {min_cos:.6}");
    assert!(
        min_cos > 0.9999 && max_abs < 5e-3,
        "{max_abs:e} / {min_cos}"
    );
}

fn fixture_ids(dir: &std::path::Path) -> Vec<u32> {
    reference(dir)["ids"]
        .as_slice::<i32>()
        .iter()
        .map(|&t| t as u32)
        .collect()
}

/// A prompt fed through the model leaves the head one entry per token but the first, and the
/// state a plain decode step leaves keeps it that way.
#[test]
fn the_head_keeps_one_entry_per_consumed_token_but_the_first() {
    let dir = fixture_dir();
    let model = load_with_head(&dir, None);
    let ids = fixture_ids(&dir);
    let mut cache = model.make_cache();
    model.forward(&ids[..1], &mut cache, true).unwrap();
    assert_eq!(model.mtp_entries(&cache), 0, "the first token has no entry");
    let cuts = [16usize, 40];
    model
        .forward_pieces(&ids[1..40], &mut cache, &cuts.map(|c| c - 1), None)
        .unwrap();
    assert_eq!(cache.len(), 40);
    assert_eq!(model.mtp_entries(&cache), 39);
    model.forward(&ids[40..41], &mut cache, true).unwrap();
    assert_eq!(model.mtp_entries(&cache), 40);
    assert!(model.mtp_ready(&cache));
    // A cache made without the head's blessing (restored from a state saved without it) is
    // not ready.
    let plain = Qwen4ExpModel::load(&dir, None).unwrap();
    assert!(!plain.mtp_ready(&plain.make_cache()));
}

/// Two rows through one verify pass give what two single steps give (to the usual batch
/// re-association), and the state it keeps for a rejected draft is the state after one step.
#[test]
fn a_verify_pass_agrees_with_two_single_steps() {
    let dir = fixture_dir();
    let model = load_with_head(&dir, None);
    let vocab = model.config().vocab_size;
    let ids = fixture_ids(&dir);
    let mut base = model.make_cache();
    model.forward(&ids[..40], &mut base, true).unwrap();

    let (p, d, x) = (ids[40], ids[41], ids[42]);
    let mut seq_cache = base.clone();
    let l0 = host_f32(&model.forward(&[p], &mut seq_cache, true).unwrap());
    let mut after_p = seq_cache.clone();
    let l1 = host_f32(&model.forward(&[d], &mut seq_cache, true).unwrap());

    let mut full = base.clone();
    let v = model.forward_verify(&[p, d], &mut full).unwrap();
    let got = host_f32(&v.logits);
    let (a0, c0) = compare(&got[..vocab], &l0, vocab);
    let (a1, c1) = compare(&got[vocab..], &l1, vocab);
    eprintln!("verify rows vs single steps: row 0 {a0:.2e}/{c0:.6}, row 1 {a1:.2e}/{c1:.6}");
    assert!(c0 > 0.9999 && a0 < 5e-3 && c1 > 0.9999 && a1 < 5e-3);
    assert_eq!(argmax_rows(&got[..vocab], vocab), argmax_rows(&l0, vocab));
    assert_eq!(argmax_rows(&got[vocab..], vocab), argmax_rows(&l1, vocab));

    // The kept first-row state continues like the state after one single step...
    let mut first = v.first;
    assert_eq!(first.len(), 41);
    assert_eq!(full.len(), 42);
    let from_first = host_f32(&model.forward(&[x], &mut first, true).unwrap());
    let from_step = host_f32(&model.forward(&[x], &mut after_p, true).unwrap());
    let (a, c) = compare(&from_first, &from_step, vocab);
    eprintln!("after a rejected draft vs after one step: {a:.2e}/{c:.6}");
    assert!(c > 0.9999 && a < 5e-3);
    // ...and the full state like the state after two.
    let from_full = host_f32(&model.forward(&[x], &mut full, true).unwrap());
    let from_two = host_f32(&model.forward(&[x], &mut seq_cache, true).unwrap());
    let (a, c) = compare(&from_full, &from_two, vocab);
    eprintln!("after an accepted draft vs after two steps: {a:.2e}/{c:.6}");
    assert!(c > 0.9999 && a < 5e-3);
}

/// Cost of a two-row verify pass against a single decode step, on the real checkpoint with its
/// experts warm (the same state is stepped repeatedly, so every expert it needs is resident).
/// Prints medians; asserts nothing. Run with `--release --ignored --nocapture` and
/// `LUMEN_QWEN4_MODEL_DIR` set; `LUMEN_QWEN4_POOL_GB` sizes the pool (default 19.8).
#[test]
#[ignore = "needs the real Flash-Next checkpoint; set LUMEN_QWEN4_MODEL_DIR"]
fn verify_cost_against_a_single_step() {
    let Ok(dir) = std::env::var("LUMEN_QWEN4_MODEL_DIR") else {
        return;
    };
    let pool_gb: f64 = std::env::var("LUMEN_QWEN4_POOL_GB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(19.8);
    let model = load_with_head(std::path::Path::new(&dir), Some((pool_gb * 1e9) as u64));
    let tok = tokenizers::Tokenizer::from_file(std::path::Path::new(&dir).join("tokenizer.json"))
        .expect("tokenizer");
    let text = "A hash table maps keys to values using a hash function. When two keys land in \
                the same bucket the table has a collision, which it resolves by chaining or by \
                probing for another slot. A table that is too full slows down, so it grows.";
    let ids: Vec<u32> = tok.encode(text, false).unwrap().get_ids().to_vec();
    let mut cache = model.make_cache();
    model.forward(&ids, &mut cache, true).unwrap();
    // Walk a few steps greedily so the pool holds what this text routes to.
    let argmax = |l: &Array| -> u32 {
        let flat = mlx_rs::ops::reshape(l, &[-1]).unwrap();
        let a = mlx_rs::ops::indexing::argmax_axis(&flat, 0, false).unwrap();
        a.eval().unwrap();
        a.item::<u32>()
    };
    let mut next = argmax(
        &model
            .forward(&ids[ids.len() - 1..], &mut cache.clone(), true)
            .unwrap(),
    );
    for _ in 0..24 {
        let l = model.forward(&[next], &mut cache, true).unwrap();
        next = argmax(&l);
    }
    let p = next;
    let d = argmax(&model.forward(&[p], &mut cache.clone(), true).unwrap());
    let median = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };
    let time = |f: &mut dyn FnMut()| {
        f();
        f();
        median(
            (0..9)
                .map(|_| {
                    let t = std::time::Instant::now();
                    f();
                    t.elapsed().as_secs_f64() * 1e3
                })
                .collect(),
        )
    };
    model.offload_reset_timing();
    let single = time(&mut || {
        let mut c = cache.clone();
        model.forward(&[p], &mut c, true).unwrap();
    });
    eprintln!(
        "single: {}",
        model
            .offload_report()
            .unwrap_or_default()
            .lines()
            .next()
            .unwrap_or("")
    );
    model.offload_reset_timing();
    let verify = time(&mut || {
        let mut c = cache.clone();
        model.forward_verify(&[p, d], &mut c).unwrap();
    });
    eprintln!(
        "verify: {}",
        model
            .offload_report()
            .unwrap_or_default()
            .lines()
            .next()
            .unwrap_or("")
    );
    let draft = time(&mut || {
        let mut c = cache.clone();
        model.draft_for(&mut c, p).unwrap();
    });
    eprintln!(
        "pool {pool_gb} GB, experts warm: single step {single:.1} ms, verify (2 rows) {verify:.1} ms \
         ({:.2}x), draft {draft:.1} ms",
        verify / single
    );
    if let Some(r) = model.offload_report() {
        eprintln!("{r}");
    }
}

/// Decode cost per token: plain steps against oracle-draft verify rounds, over different
/// stretches of one greedy continuation of the real checkpoint (so both pay for the experts
/// their tokens route to). Prints; asserts nothing.
#[test]
#[ignore = "needs the real Flash-Next checkpoint; set LUMEN_QWEN4_MODEL_DIR"]
fn speculative_rounds_against_plain_steps_per_token() {
    let Ok(dir) = std::env::var("LUMEN_QWEN4_MODEL_DIR") else {
        return;
    };
    let pool_gb: f64 = std::env::var("LUMEN_QWEN4_POOL_GB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(19.8);
    let model = load_with_head(std::path::Path::new(&dir), Some((pool_gb * 1e9) as u64));
    let tok = tokenizers::Tokenizer::from_file(std::path::Path::new(&dir).join("tokenizer.json"))
        .expect("tokenizer");
    let text = "Explain, in about 300 words, how a hash table handles collisions and when to resize it. \
                A hash table maps keys to values using a hash function.";
    let ids: Vec<u32> = tok.encode(text, false).unwrap().get_ids().to_vec();
    let argmax = |l: &Array| -> u32 {
        let flat = mlx_rs::ops::reshape(l, &[-1]).unwrap();
        let a = mlx_rs::ops::indexing::argmax_axis(&flat, 0, false).unwrap();
        a.eval().unwrap();
        a.item::<u32>()
    };
    let mut cache = model.make_cache();
    let logits = model.forward(&ids, &mut cache, true).unwrap();
    let mut toks = vec![argmax(&logits)];
    // The continuation: 120 greedy tokens, with snapshots every 40.
    let mut snaps = vec![cache.clone()];
    for i in 0..120 {
        let l = model.forward(&[toks[i]], &mut cache, true).unwrap();
        toks.push(argmax(&l));
        if (i + 1) % 40 == 0 {
            snaps.push(cache.clone());
        }
    }
    // The same stretch of the continuation (tokens 40..80, from snapshot 1) either one plain
    // step per token or oracle-draft verify rounds; which one comes from `QWEN4_MODE`, one
    // process each so neither inherits the other's page cache or expert pool.
    let n = 40;
    let mode = std::env::var("QWEN4_MODE").unwrap_or_else(|_| "plain".into());
    model.offload_reset_timing();
    let mut c = snaps[1].clone();
    let t = std::time::Instant::now();
    if mode == "plain" {
        for i in 40..40 + n {
            model.forward(&[toks[i]], &mut c, true).unwrap();
        }
    } else {
        let mut i = 40;
        while i + 1 < 40 + n {
            model.draft_for(&mut c, toks[i]).unwrap();
            let mut full = c.clone();
            let mut v = model
                .forward_verify(&[toks[i], toks[i + 1]], &mut full)
                .unwrap();
            model
                .mtp_finish_round(&mut v.first, Some(&mut full), &v.multi, toks[i + 1])
                .unwrap();
            c = full;
            i += 2;
        }
    }
    let per_token = t.elapsed().as_secs_f64() * 1e3 / n as f64;
    eprintln!(
        "{mode}: {per_token:.1} ms/token | {}",
        model
            .offload_report()
            .unwrap_or_default()
            .lines()
            .next()
            .unwrap_or("")
    );
}
