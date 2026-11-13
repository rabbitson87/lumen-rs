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
use std::path::{Path, PathBuf};

use mlx_rs::Array;

use super::config::Qwen4ExpConfig;
use super::model::{
    ExpertTier, LoadOptions, LowExpertOptions, MtpMode, NgramSource, Qwen4ExpModel,
};
use super::ngram::NgramHasher;
use crate::offload::lowbit::{LowbitSpec, Recipe};

/// `f` with the KV cache in f32 (`LUMEN_QWEN4_KV_BF16=0`): the reference logits were computed
/// with one, and these tests check the port's arithmetic against them at f32 tolerances.
fn f32_kv<R>(f: impl FnOnce() -> R) -> R {
    super::config::kv_bf16::with(false, f)
}

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
    f32_kv(|| {
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
    })
}

#[test]
fn prefill_then_single_token_decode_matches_the_reference() {
    f32_kv(|| {
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
    })
}

/// The same prompt fed in uneven chunks must land on the same logits as one
/// prefill: every carried state (convolution history, recurrent state, KV and
/// indexer keys, the n-gram tail and its convolution) crosses the boundaries.
#[test]
fn chunked_prefill_matches_a_single_prefill() {
    f32_kv(|| {
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
    })
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
    offloaded_matches_resident();
}

/// `LUMEN_MOE_OVERLAP`: the experts the pool holds are computed while the missing ones are read,
/// and put back in the routing's order before the weighted sum. The logits must stay the resident
/// model's bit for bit at every pool size, and the overlapped path must actually be taken (the
/// replica's reads are microseconds, so the read-time bar is lowered for the test).
#[test]
fn overlapping_reads_with_resident_experts_keeps_the_resident_models_logits() {
    // Build-ahead decode leaves the overlap out (its flag says so), and the flags matrix
    // switches both on: pin it off, since the overlapped path is what this test is for.
    let overlapped = super::model::build_ahead::with(false, || {
        crate::offload::overlap::with(true, || {
            crate::offload::test_overlap::with_any_read(offloaded_matches_resident)
        })
    });
    assert!(overlapped > 0, "no call took the overlapped path");
}

/// How many layer calls overlapped their reads, from the offload's report.
fn overlapped_calls(model: &Qwen4ExpModel) -> u64 {
    let report = model.offload_report().expect("offload on");
    report
        .split("overlapped ")
        .nth(1)
        .and_then(|rest| rest.split(' ').next())
        .and_then(|n| n.parse().ok())
        .unwrap_or(0)
}

/// The body of `offloaded_experts_match_the_resident_model`; returns the calls that overlapped.
fn offloaded_matches_resident() -> u64 {
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
    let mut overlapped = 0;
    for budget in [1u64, 20 * 9_900, 34 * 9_900, 100 * 9_900, u64::MAX / 2] {
        let model = Qwen4ExpModel::load(&dir, Some(budget)).expect("load with offload");
        let got = run(&model);
        let (hits, misses) = model.offload_stats().expect("offload on");
        let (max_abs, _) = compare(&got, &baseline, vocab);
        let split = overlapped_calls(&model);
        eprintln!(
            "budget {budget}: hits {hits} misses {misses}, {split} calls overlapped, max |diff| {max_abs:e}"
        );
        overlapped += split;
        assert!(
            got == baseline,
            "budget {budget}: logits differ from the resident model (max |diff| {max_abs:e})"
        );
    }
    overlapped
}

/// The pipelined decode path (`LUMEN_QWEN4_BUILD_AHEAD`) builds each layer's second half and the
/// next layer's graph before the routing is read back, with the gather reading its slot numbers
/// from a persistent array the CPU fills in afterwards. It must give the plain path's logits bit
/// for bit, whatever the pool holds: the smallest pool (every call evicts), a few layers' worth,
/// and every expert resident. A prefill piece in the same stream takes the plain path.
#[test]
fn building_decode_layers_ahead_gives_the_plain_paths_logits() {
    let dir = fixture_dir();
    let resident = Qwen4ExpModel::load(&dir, None).expect("load tiny replica");
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
    for budget in [1u64, 20 * 9_900, 34 * 9_900, 100 * 9_900, u64::MAX / 2] {
        let model = Qwen4ExpModel::load(&dir, Some(budget)).expect("load with offload");
        let got = super::model::build_ahead::with(true, || run(&model));
        let (hits, misses) = model.offload_stats().expect("offload on");
        eprintln!("budget {budget}: hits {hits} misses {misses}");
        assert!(
            got == baseline,
            "budget {budget}: the pipelined decode path changed the logits"
        );
    }
}

/// `LUMEN_MOE_TRACE_PREDICT` asks every layer's gate about the layer before it. It only looks: the
/// logits stay the plain path's. Each decode call past the first layer carries a prediction, the
/// first layer has nothing before it, and the prefill pieces record nothing.
#[test]
fn the_trace_predicts_each_later_layer_and_changes_nothing() {
    use crate::offload::trace::{PREDICT_TOP, moe_trace, moe_trace_predict};

    let dir = fixture_dir();
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
    let budget = Some(34 * 9_900);
    let baseline = run(&Qwen4ExpModel::load(&dir, budget).expect("load"));

    let path = std::env::temp_dir().join(format!("lumen-predicted-{}.bin", std::process::id()));
    // SAFETY: this crate's tests run on one thread, and this variable is only read at open.
    unsafe { std::env::set_var("LUMEN_MOE_TRACE", &path) };
    let model = moe_trace::with(true, || Qwen4ExpModel::load(&dir, budget).expect("load"));
    // SAFETY: as above.
    unsafe { std::env::remove_var("LUMEN_MOE_TRACE") };
    let got = moe_trace_predict::with(true, || {
        super::model::build_ahead::with(true, || run(&model))
    });
    drop(model);
    let (width, calls) = crate::offload::slot_replay::predictions(&path);
    std::fs::remove_file(&path).unwrap();

    assert!(got == baseline, "the prediction changed the logits");
    let want_len = (PREDICT_TOP as usize).min(width);
    let first = calls.iter().map(|c| c.0).min().unwrap();
    assert!(calls.len() >= 2, "decode calls recorded");
    for (layer, predicted) in &calls {
        if *layer == first {
            assert!(predicted.is_empty(), "nothing comes before layer {layer}");
            continue;
        }
        assert_eq!(predicted.len(), want_len, "layer {layer}");
        let mut seen = predicted.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), want_len, "distinct experts");
        assert!(seen.iter().all(|&e| usize::from(e) < width));
    }
}

/// `LUMEN_MOE_READ_AHEAD` reads the experts the next layer's gate ranks highest while the
/// current layer computes. It changes which experts are resident when, never what is computed:
/// a prompt, single steps and two-row verify passes give the same logits bit for bit with it on
/// and off, on the plain and the pipelined decode paths, at pool sizes from every-call-evicts to
/// all-resident. Where the pool is small it must actually read ahead, and some of it must hit.
#[test]
fn reading_ahead_changes_no_logit() {
    use crate::offload::read_ahead;

    let dir = fixture_dir();
    let ids = fixture_ids(&dir);
    let run = |model: &Qwen4ExpModel| -> Vec<f32> {
        let mut cache = model.make_cache();
        let mut out = host_f32(&model.forward(&ids[..40], &mut cache, false).unwrap());
        let mut t = 40;
        while t + 2 < ids.len() {
            out.extend(host_f32(
                &model.forward(&ids[t..t + 1], &mut cache, false).unwrap(),
            ));
            let v = model
                .forward_verify(&ids[t + 1..t + 3], &mut cache)
                .unwrap();
            out.extend(host_f32(&v.logits));
            t += 3;
        }
        out
    };
    let read_ahead_of = |model: &Qwen4ExpModel| -> (u64, u64) {
        let report = model.offload_report().expect("offload on");
        let Some(rest) = report.split("read ahead ").nth(1) else {
            return (0, 0);
        };
        // "read ahead {n} experts on {calls} calls, {pct}% of them used, ..."
        let n = |s: &str, end: char| {
            s.split(end)
                .next()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0)
        };
        let read: u64 = n(rest, ' ');
        let used_pct: u64 = n(rest.split("calls, ").nth(1).unwrap_or(""), '%');
        (read, read * used_pct / 100)
    };
    for pipelined in [false, true] {
        for budget in [1u64, 20 * 9_900, 34 * 9_900, u64::MAX / 2] {
            let load = || Qwen4ExpModel::load(&dir, Some(budget)).expect("load with offload");
            let plain = super::model::build_ahead::with(pipelined, || {
                read_ahead::with(false, || run(&load()))
            });
            let model = load();
            let got = super::model::build_ahead::with(pipelined, || {
                read_ahead::with(true, || run(&model))
            });
            let (read, used) = read_ahead_of(&model);
            eprintln!("pipelined {pipelined}, budget {budget}: read ahead {read}, ~{used} used");
            assert!(
                got == plain,
                "pipelined {pipelined}, budget {budget}: reading ahead changed the logits"
            );
            if budget < 100 * 9_900 {
                assert!(
                    read > 0 && used > 0,
                    "budget {budget}: nothing read ahead, or none of it used"
                );
            }
        }
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
    every_call_size_matches_resident();
}

/// The same streams and cuttings with `LUMEN_MOE_OVERLAP`: the one- and few-row pieces overlap
/// their reads, the rest take the plain path, and the logits stay the resident model's.
#[test]
fn overlapped_reads_match_the_resident_model_for_every_call_size() {
    // Build-ahead decode leaves the overlap out (its flag says so), and the flags matrix
    // switches both on: pin it off, since the overlapped path is what this test is for.
    let overlapped = super::model::build_ahead::with(false, || {
        crate::offload::overlap::with(true, || {
            crate::offload::test_overlap::with_any_read(every_call_size_matches_resident)
        })
    });
    assert!(overlapped > 0, "no call took the overlapped path");
}

/// The body of `offloaded_experts_match_the_resident_model_for_every_call_size`; returns the
/// calls that overlapped.
fn every_call_size_matches_resident() -> u64 {
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
    let mut overlapped = 0;
    for budget in [1u64, 40 * 9_900, 100 * 9_900, u64::MAX / 2] {
        for ((name, ids), baselines) in streams.iter().zip(&baselines) {
            for (pieces, baseline) in cuttings.iter().zip(baselines) {
                // A fresh model per run: no run inherits another's pool.
                let model = Qwen4ExpModel::load(&dir, Some(budget)).expect("load with offload");
                assert!(
                    run(&model, ids, pieces) == *baseline,
                    "budget {budget}, {name}, pieces {pieces:?}: logits differ from the resident model"
                );
                overlapped += overlapped_calls(&model);
            }
        }
    }
    overlapped
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
        low_experts: None,
        expert_copy: None,
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

/// The token embedding read from the checkpoint's files a row at a time
/// (`LUMEN_QWEN4_EMBED_DISK`) gives the logits the resident table does, whole and fed in
/// pieces, and a second pass over the same tokens reads nothing.
#[test]
fn serving_the_token_embedding_from_disk_matches_holding_it_in_memory() {
    use super::model::embed_from_disk;
    let dir = fixture_dir();
    let want = reference(&dir);
    let ids: Vec<u32> = want["ids"]
        .as_slice::<i32>()
        .iter()
        .map(|&t| t as u32)
        .collect();
    let whole = |model: &Qwen4ExpModel| -> Vec<f32> {
        let mut cache = model.make_cache();
        host_f32(&model.forward(&ids, &mut cache, false).unwrap())
    };
    // The prompt, then two more tokens one at a time, as decode feeds them.
    let stepped = |model: &Qwen4ExpModel| -> Vec<f32> {
        let mut cache = model.make_cache();
        let (head, tail) = ids.split_at(ids.len() - 2);
        let mut out = host_f32(&model.forward(head, &mut cache, true).unwrap());
        for t in tail {
            out.extend(host_f32(&model.forward(&[*t], &mut cache, true).unwrap()));
        }
        out
    };
    let resident = embed_from_disk::with(false, || Qwen4ExpModel::load(&dir, None)).unwrap();
    assert!(resident.embed_stats().is_none());
    let (baseline, baseline_stepped) = (whole(&resident), stepped(&resident));
    drop(resident);

    let from_disk = embed_from_disk::with(true, || Qwen4ExpModel::load(&dir, None)).unwrap();
    assert!(
        whole(&from_disk) == baseline,
        "an embedding read from disk changes the logits"
    );
    let first = from_disk.embed_stats().expect("read from disk");
    assert_eq!(first.requested, ids.len() as u64);
    assert!(first.read_from_disk > 0 && first.read_from_disk <= first.requested);
    assert!(stepped(&from_disk) == baseline_stepped);
    let second = from_disk.embed_stats().unwrap();
    assert_eq!(second.requested, 2 * ids.len() as u64);
    assert_eq!(
        second.read_from_disk, first.read_from_disk,
        "the second pass should have been served from the row cache"
    );
    // A token past the vocabulary is refused, not read from whatever follows the table.
    let vocab = from_disk.config().vocab_size as u32;
    assert!(
        from_disk
            .forward(&[vocab], &mut from_disk.make_cache(), false)
            .is_err()
    );
    // The pool the model would be given: the experts from disk as well.
    let both = embed_from_disk::with(true, || Qwen4ExpModel::load(&dir, Some(40 * 9_900))).unwrap();
    assert!(whole(&both) == baseline);
}

/// On the real checkpoint: embedding rows read from its files are the rows MLX's own lookup
/// gives, over a spread of the vocabulary and its last token, with a row cache smaller than the
/// request. Reads only the embedding (0.68 GB), not the model. Run with `--ignored` and
/// `LUMEN_QWEN4_MODEL_DIR` set.
#[test]
#[ignore = "needs the real Flash-Next checkpoint; set LUMEN_QWEN4_MODEL_DIR"]
fn real_embedding_rows_read_from_disk_are_the_rows_the_resident_lookup_gives() {
    use super::config::TOKEN_EMBEDDING;
    use super::ngram_table::FileTable;
    use crate::native_embedding::quantized_embedding_lookup_with_mode;
    use crate::native_quant::MODE_AFFINE;
    use crate::offload::IoMode;
    let Ok(dir) = std::env::var("LUMEN_QWEN4_MODEL_DIR") else {
        return;
    };
    let dir = Path::new(&dir);
    let cfg = Qwen4ExpConfig::load(dir).unwrap();
    let q = cfg.quant.for_base(TOKEN_EMBEDDING);
    let mut weights = crate::qwen3_5_moe::NativeWeights::load_dir(dir).unwrap();
    weights.sanitize().unwrap();
    let part = |name: &str| {
        weights
            .require(&format!("{TOKEN_EMBEDDING}.{name}"))
            .unwrap()
    };
    let last = cfg.vocab_size as u32 - 1;
    let ids: Vec<u32> = (0..last).step_by(97).chain([last, 0, last]).collect();
    let ids_i32: Vec<i32> = ids.iter().map(|&t| t as i32).collect();
    let resident = quantized_embedding_lookup_with_mode(
        part("weight"),
        part("scales"),
        Some(part("biases")),
        &Array::from_slice(&ids_i32, &[ids.len() as i32]),
        q.group_size,
        q.bits,
        MODE_AFFINE,
    )
    .unwrap();
    let table = FileTable::open(
        dir,
        1,
        &|_| TOKEN_EMBEDDING.to_string(),
        q.group_size,
        q.bits,
        64,
        IoMode::Buffered,
    )
    .unwrap();
    let rows: Vec<u64> = ids.iter().map(|&t| u64::from(t)).collect();
    let from_disk = table.gather(&rows).unwrap();
    assert_eq!(from_disk.shape(), resident.shape());
    assert_eq!(from_disk.dtype(), resident.dtype());
    assert_eq!(
        from_disk.shape(),
        &[ids.len() as i32, cfg.hidden_size as i32]
    );
    let (a, b) = (host_f32(&from_disk), host_f32(&resident));
    assert!(a.iter().any(|&v| v != 0.0));
    assert!(
        a == b,
        "rows read from disk differ from the resident lookup"
    );
    eprintln!(
        "{} rows of {} ({}-bit, groups of {}): identical; {} read from disk",
        ids.len(),
        cfg.hidden_size,
        q.bits,
        q.group_size,
        table.stats().read_from_disk
    );
}

/// A request for more rows than the row cache holds still returns every row: a call's rows
/// come from its own reads and from what the cache held when it started, whatever the cache
/// keeps afterwards. The table here is the fixture's token embedding, read as one shard.
#[test]
fn a_request_larger_than_the_row_cache_returns_every_row() {
    use super::ngram_table::FileTable;
    use crate::offload::IoMode;
    let dir = fixture_dir();
    let cfg = Qwen4ExpConfig::load(&dir).unwrap();
    let base = super::config::TOKEN_EMBEDDING;
    let q = cfg.quant.for_base(base);
    let open = |row_cache| {
        FileTable::open(
            &dir,
            1,
            &|_| base.to_string(),
            q.group_size,
            q.bits,
            row_cache,
            IoMode::Buffered,
        )
        .unwrap()
    };
    // Every row, then the first few again.
    let ids: Vec<u64> = (0..cfg.vocab_size as u64).chain(0..8).collect();
    let vocab = cfg.vocab_size as u64;
    let roomy = host_f32(&open(1 << 16).gather(&ids).unwrap());
    let tight = open(4);
    assert!(host_f32(&tight.gather(&ids).unwrap()) == roomy);
    // Each distinct row once, the repeats included.
    assert_eq!(tight.stats().read_from_disk, vocab);
    // Again, with a cache that holds a few of the rows and has to drop them on the way.
    assert!(host_f32(&tight.gather(&ids).unwrap()) == roomy);
    let stats = tight.stats();
    assert_eq!(stats.requested, 2 * ids.len() as u64);
    assert!((2 * vocab - 4..=2 * vocab).contains(&stats.read_from_disk));
    // A row past the table is refused.
    assert!(tight.gather(&[vocab]).is_err());
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
        f32_kv(|| {
            let mut cache = model.make_cache();
            host_f32(
                &model
                    .forward(&ids, &mut cache, false)
                    .expect("forward on another thread"),
            )
        })
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
            low_experts: None,
            expert_copy: None,
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

// ---- image prompts: 3-D rotary positions -----------------------------------------------------

/// A copy of the replica whose config declares 3-D rotary sections (its rotary dim is 8, so
/// `[2, 1, 1]`, interleaved as Flash-Next's) and an image token. It has no tower, so these tests
/// feed text through the image path and check its position handling.
fn mrope_replica() -> (tempfile::TempDir, Qwen4ExpModel) {
    let src = fixture_dir();
    let dir = tempfile::tempdir().expect("temp dir");
    for name in ["model.safetensors", "mtp.json", "mtp.safetensors"] {
        std::fs::copy(src.join(name), dir.path().join(name)).expect("copy the replica");
    }
    let mut cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(src.join("config.json")).unwrap()).unwrap();
    let rope = &mut cfg["text_config"]["rope_parameters"];
    rope["mrope_section"] = serde_json::json!([2, 1, 1]);
    rope["mrope_interleaved"] = serde_json::json!(true);
    cfg["image_token_id"] = serde_json::json!(77);
    std::fs::write(dir.path().join("config.json"), cfg.to_string()).unwrap();
    let model = Qwen4ExpModel::load(dir.path(), None).expect("load the replica copy");
    assert!(model.config().mrope.is_some(), "the sections are usable");
    (dir, model)
}

/// The prompt's logits and five greedy steps after it, fed through the image path at
/// `positions` with `delta` (no images), or through the text path when `positions` is `None`.
fn prompt_then_steps(
    model: &Qwen4ExpModel,
    ids: &[u32],
    positions: Option<(&[[i32; 3]], i32)>,
) -> (Vec<f32>, Vec<u32>) {
    let mut cache = model.make_cache();
    let cuts: Vec<usize> = (1..)
        .map(|k| k * 16)
        .take_while(|&c| c < ids.len())
        .chain([ids.len()])
        .collect();
    let logits = match positions {
        Some((pos, delta)) => model
            .forward_image_prompt(ids, &mut cache, 16, pos, delta, &[], &[])
            .expect("image path"),
        None => {
            model
                .forward_pieces(ids, &mut cache, &cuts, None)
                .expect("text path")
                .0
        }
    };
    let argmax = |a: &Array| -> u32 {
        let v = host_f32(a);
        (0..v.len()).max_by(|&x, &y| v[x].total_cmp(&v[y])).unwrap() as u32
    };
    let first = host_f32(&logits);
    let mut steps = vec![argmax(&logits)];
    for _ in 0..4 {
        let l = model
            .forward(&[*steps.last().unwrap()], &mut cache, true)
            .expect("decode");
        steps.push(argmax(&l));
    }
    if positions.is_some() {
        assert!(
            cache.to_records().is_err(),
            "an image sequence is not persisted"
        );
    }
    (first, steps)
}

/// Text fed through the image path at its own index (`[i, i, i]`, delta 0), or shifted by a
/// constant (RoPE sees only relative positions), predicts what the text path does: the prompt's
/// logits to rounding (the 3-D rotation is computed unfused), the same greedy steps after it
/// (decode continues at index + delta), past the indexer's budget so the sparse path and its
/// pooled block positions run too. Positions laid out as an image's grid do change them.
#[test]
fn text_through_the_image_path_matches_the_text_path() {
    // In f32 (`LUMEN_QWEN4_KV_BF16=0`): the check is the positions', at f32 tolerances.
    f32_kv(text_through_the_image_path);
}

fn text_through_the_image_path() {
    let (_dir, model) = mrope_replica();
    let ids = fixture_ids(&fixture_dir());
    assert!(ids.len() > 2 * 16, "past the replica's indexer budget");
    let vocab = model.config().vocab_size;
    let (want, want_steps) = prompt_then_steps(&model, &ids, None);
    for delta in [0i32, 7] {
        let pos: Vec<[i32; 3]> = (0..ids.len() as i32)
            .map(|i| [i + delta, i + delta, i + delta])
            .collect();
        let (got, steps) = prompt_then_steps(&model, &ids, Some((&pos, delta)));
        let (max_abs, min_cos) = compare(&got, &want, vocab);
        assert!(
            min_cos > 0.9999 && max_abs < 2e-3,
            "delta {delta}: {max_abs:e} / {min_cos}"
        );
        assert_eq!(steps, want_steps, "delta {delta}");
    }
    // Six tokens laid out as a 2 x 3 image after the first ten, as `mrope_positions` would.
    let mut pos: Vec<[i32; 3]> = Vec::new();
    for i in 0..10 {
        pos.push([i, i, i]);
    }
    for k in 0..6 {
        pos.push([10, 10 + k / 3, 10 + k % 3]);
    }
    let next = 13;
    for i in 0..(ids.len() as i32 - 16) {
        pos.push([next + i, next + i, next + i]);
    }
    let delta = next - 16;
    let (got, _) = prompt_then_steps(&model, &ids, Some((&pos, delta)));
    let (max_abs, _) = compare(&got, &want, vocab);
    assert!(
        max_abs > 1e-3,
        "image positions changed nothing ({max_abs:e})"
    );
}

/// An image's pixels are normalized as the checkpoint's `preprocessor_config.json` says (Flash-Next:
/// 0.5 / 0.5), with CLIP's constants only when it says nothing.
#[test]
fn image_normalization_comes_from_the_preprocessor_config() {
    use crate::qwen36_vision::ImageNorm;
    let (dir, model) = mrope_replica();
    assert_eq!(
        model.config().image_norm,
        ImageNorm::CLIP,
        "no preprocessor config"
    );
    std::fs::write(
        dir.path().join("preprocessor_config.json"),
        r#"{"image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "patch_size": 16}"#,
    )
    .unwrap();
    let cfg = super::config::Qwen4ExpConfig::load(dir.path()).unwrap();
    assert_eq!(
        cfg.image_norm,
        ImageNorm {
            mean: [0.5; 3],
            std: [0.5; 3]
        }
    );
}

// ---- low-bit experts ---------------------------------------------------------------------

/// The replica in a directory of its own, where a pack can be written.
fn replica_copy() -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    for name in ["config.json", "model.safetensors"] {
        std::fs::copy(fixture_dir().join(name), tmp.path().join(name)).unwrap();
    }
    tmp
}

/// [`replica_copy`], with the draft head.
fn replica_with_head() -> tempfile::TempDir {
    let tmp = replica_copy();
    add_head(tmp.path());
    tmp
}

fn add_head(dir: &Path) {
    for name in ["mtp.json", "mtp.safetensors"] {
        std::fs::copy(fixture_dir().join(name), dir.join(name)).unwrap();
    }
}

fn with_low_experts(dir: &Path, spec: LowbitSpec) -> Qwen4ExpModel {
    crate::offload::build_lowbit_pack(dir, 4, spec).expect("low-bit pack");
    let budget = 400 << 20;
    Qwen4ExpModel::load_with(
        dir,
        LoadOptions {
            expert_budget_bytes: Some(budget),
            ngram: NgramSource::Auto,
            mtp: MtpMode::Off,
            scores_budget_bytes: 0,
            low_experts: Some(LowExpertOptions {
                spec,
                budget_bytes: budget,
            }),
            expert_copy: None,
        },
    )
    .expect("load the replica with low-bit experts")
}

/// The last position's logits after feeding `tokens` one at a time.
fn last_logits(model: &Qwen4ExpModel, tokens: &[u32]) -> Vec<f32> {
    let mut cache = model.make_cache();
    let mut last = Vec::new();
    for &t in tokens {
        let logits = model.forward(&[t], &mut cache, true).unwrap();
        logits.eval().unwrap();
        last = logits.as_slice::<f32>().to_vec();
    }
    last
}

fn distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

#[test]
fn a_loaded_low_bit_copy_changes_nothing_until_it_is_chosen() {
    let tmp = replica_copy();
    let tokens = [3u32, 14, 15, 92, 65];
    let plain = last_logits(
        &Qwen4ExpModel::load(tmp.path(), Some(400 << 20)).unwrap(),
        &tokens,
    );
    let spec = LowbitSpec::new(2, 32, Recipe::LeastSquares).unwrap();
    let model = with_low_experts(tmp.path(), spec);
    assert_eq!(last_logits(&model, &tokens), plain);

    model.set_expert_tier(ExpertTier::Low).unwrap();
    let low = last_logits(&model, &tokens);
    assert!(low.iter().all(|x| x.is_finite()));
    assert_ne!(low, plain, "the low tier read the checkpoint's experts");
    assert_eq!(
        last_logits(&model, &tokens),
        low,
        "the low tier is deterministic"
    );

    model.set_expert_tier(ExpertTier::Full).unwrap();
    assert_eq!(last_logits(&model, &tokens), plain);
}

#[test]
fn more_bits_in_the_copy_stay_closer_to_the_checkpoint() {
    // The copy is fitted to the model: serving the right expert's bytes at the right bit width
    // shows as a distance that falls with the bits. Mixed-up experts or a wrong width do not.
    let tmp = replica_copy();
    let tokens = [3u32, 14, 15, 92, 65];
    let distance_at = |bits| {
        let spec = LowbitSpec::new(bits, 32, Recipe::LeastSquares).unwrap();
        let model = with_low_experts(tmp.path(), spec);
        let full = last_logits(&model, &tokens);
        model.set_expert_tier(ExpertTier::Low).unwrap();
        distance(&last_logits(&model, &tokens), &full)
    };
    let (two, four) = (distance_at(2), distance_at(4));
    assert!(four < two / 2.0, "4 bits {four}, 2 bits {two}");
}

#[test]
fn a_model_without_a_copy_refuses_the_low_tier() {
    let model = Qwen4ExpModel::load(&fixture_dir(), None).unwrap();
    assert!(model.set_expert_tier(ExpertTier::Low).is_err());
    assert!(model.set_expert_tier(ExpertTier::Full).is_ok());
}

#[test]
fn asking_for_a_copy_that_was_not_built_is_an_error() {
    let tmp = replica_copy();
    let spec = LowbitSpec::new(2, 32, Recipe::LeastSquares).unwrap();
    let r = Qwen4ExpModel::load_with(
        tmp.path(),
        LoadOptions {
            expert_budget_bytes: Some(400 << 20),
            ngram: NgramSource::Auto,
            mtp: MtpMode::Off,
            scores_budget_bytes: 0,
            low_experts: Some(LowExpertOptions {
                spec,
                budget_bytes: 400 << 20,
            }),
            expert_copy: None,
        },
    );
    assert!(r.is_err());
}

fn copy_options(spec: LowbitSpec, budget: Option<u64>) -> LoadOptions {
    LoadOptions {
        expert_budget_bytes: budget,
        ngram: NgramSource::Auto,
        mtp: MtpMode::Off,
        scores_budget_bytes: 0,
        low_experts: None,
        expert_copy: Some(spec),
    }
}

#[test]
fn a_model_served_from_a_copy_is_the_low_tier_of_one_that_holds_both() {
    let tmp = replica_copy();
    let tokens = [3u32, 14, 15, 92, 65];
    let spec = LowbitSpec::new(2, 32, Recipe::LeastSquares).unwrap();
    let both = with_low_experts(tmp.path(), spec);
    both.set_expert_tier(ExpertTier::Low).unwrap();
    let low = last_logits(&both, &tokens);
    drop(both);

    let budget = Some(400 << 20);
    let copy = Qwen4ExpModel::load_with(tmp.path(), copy_options(spec, budget)).unwrap();
    assert_eq!(last_logits(&copy, &tokens), low);
    // The plan reads the copy's size, not the checkpoint's.
    let own = Qwen4ExpConfig::load(tmp.path())
        .unwrap()
        .routed_expert_bytes();
    assert!(copy.config().routed_expert_bytes() < own);
}

#[test]
fn a_multi_token_feed_reads_the_copy_too() {
    // Prefill takes other paths through the pool than a decode step: the staging path has to be
    // served from the copy as well.
    let tmp = replica_copy();
    let tokens: Vec<u32> = (1..200).map(|t| t * 7 % 127 + 1).collect();
    let spec = LowbitSpec::new(3, 32, Recipe::LeastSquares).unwrap();
    let plain = Qwen4ExpModel::load(tmp.path(), Some(400 << 20)).unwrap();
    let mut cache = plain.make_cache();
    let full = plain.forward(&tokens, &mut cache, true).unwrap();
    full.eval().unwrap();
    let full = full.as_slice::<f32>().to_vec();
    drop(plain);

    let both = with_low_experts(tmp.path(), spec);
    both.set_expert_tier(ExpertTier::Low).unwrap();
    let mut cache = both.make_cache();
    let low = both.forward(&tokens, &mut cache, true).unwrap();
    low.eval().unwrap();
    let low = low.as_slice::<f32>().to_vec();
    drop(both);

    let copy = Qwen4ExpModel::load_with(tmp.path(), copy_options(spec, Some(400 << 20))).unwrap();
    let mut cache = copy.make_cache();
    let served = copy.forward(&tokens, &mut cache, true).unwrap();
    served.eval().unwrap();
    assert_eq!(served.as_slice::<f32>(), low);
    assert_ne!(low, full);
}

/// `lumen-server --build-expert-copy`: built from the checkpoint's own width, kept when it reads
/// back, built again when it does not.
#[test]
fn an_expert_copy_is_built_once_and_again_when_it_does_not_read_back() {
    use std::os::unix::fs::MetadataExt;
    let tmp = replica_with_head();
    let spec = LowbitSpec::new(2, 32, Recipe::Top).unwrap();
    let path = super::build_expert_copy(tmp.path(), spec).unwrap();
    assert_eq!(path, crate::offload::pack::lowbit_path(tmp.path(), spec));
    let built = std::fs::metadata(&path).unwrap().ino();
    assert_eq!(super::build_expert_copy(tmp.path(), spec).unwrap(), path);
    assert_eq!(std::fs::metadata(&path).unwrap().ino(), built, "kept");
    assert!(Qwen4ExpModel::load_with(tmp.path(), copy_options(spec, Some(400 << 20))).is_ok());

    std::fs::write(&path, b"not a pack").unwrap();
    assert_eq!(super::build_expert_copy(tmp.path(), spec).unwrap(), path);
    assert!(Qwen4ExpModel::load_with(tmp.path(), copy_options(spec, Some(400 << 20))).is_ok());
}

#[test]
fn a_copy_needs_a_pool_and_its_pack() {
    let tmp = replica_copy();
    let spec = LowbitSpec::new(2, 32, Recipe::LeastSquares).unwrap();
    assert!(Qwen4ExpModel::load_with(tmp.path(), copy_options(spec, Some(400 << 20))).is_err());
    crate::offload::build_lowbit_pack(tmp.path(), 4, spec).unwrap();
    assert!(Qwen4ExpModel::load_with(tmp.path(), copy_options(spec, None)).is_err());
}

/// The head's logits over the reference's rows, with a fixed input from the reference, so that
/// only the head's own experts can move them.
fn head_logits(model: &Qwen4ExpModel) -> Vec<f32> {
    let want = mtp_outputs(&fixture_dir());
    let ids: Vec<u32> = want["mtp_ids"]
        .as_slice::<i32>()
        .iter()
        .map(|&t| t as u32)
        .collect();
    let out = model
        .mtp_logits_for_test(&ids[1..], &want["mtp_multi_in"], &[47])
        .unwrap();
    host_f32(&out[0])
}

fn head_copy_options(spec: LowbitSpec) -> LoadOptions {
    LoadOptions {
        mtp: MtpMode::On,
        ..copy_options(spec, Some(400 << 20))
    }
}

#[test]
fn the_draft_head_reads_its_experts_at_the_width_of_the_copy() {
    let tmp = replica_with_head();
    let own = head_logits(&load_with_head(tmp.path(), Some(400 << 20)));
    // A copy at the checkpoint's own width and groups refits the same levels. Read at any other
    // width its records would be noise, or would not fit the kernel at all.
    let same = LowbitSpec::new(4, 32, Recipe::LeastSquares).unwrap();
    let coarse = LowbitSpec::new(2, 32, Recipe::LeastSquares).unwrap();
    let moved = |spec: LowbitSpec| {
        crate::offload::build_lowbit_pack(tmp.path(), 4, spec).unwrap();
        let model = Qwen4ExpModel::load_with(tmp.path(), head_copy_options(spec)).unwrap();
        assert!(model.has_mtp());
        distance(&head_logits(&model), &own)
    };
    let (near, far) = (moved(same), moved(coarse));
    assert!(near < 0.2 * far, "{near} against {far}");
    let size = own.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(far > 0.0 && far < 0.2 * size, "{far} of {size}");
}

#[test]
fn the_heads_experts_in_a_copy_leave_the_trunk_alone() {
    let tokens = [3u32, 14, 15, 92, 65];
    let spec = LowbitSpec::new(2, 32, Recipe::LeastSquares).unwrap();
    let logits = |dir: &Path| {
        crate::offload::build_lowbit_pack(dir, 4, spec).unwrap();
        let model = Qwen4ExpModel::load_with(dir, copy_options(spec, Some(400 << 20))).unwrap();
        last_logits(&model, &tokens)
    };
    assert_eq!(
        logits(replica_with_head().path()),
        logits(replica_copy().path())
    );
}

#[test]
fn a_copy_that_holds_no_head_serves_only_without_drafts() {
    let tmp = replica_with_head();
    let spec = LowbitSpec::new(2, 32, Recipe::LeastSquares).unwrap();
    let path = crate::offload::build_lowbit_pack(tmp.path(), 4, spec).unwrap();
    assert!(crate::offload::lowbit_holds_head(tmp.path(), spec).unwrap());
    // A pack from before copies held the head: the same bytes, without the header's word for it.
    let bytes = std::fs::read(&path).unwrap();
    let (key, old) = (b"draft_head", b"other_head");
    let at = bytes.windows(key.len()).position(|w| w == key).unwrap();
    let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    std::os::unix::fs::FileExt::write_all_at(&file, old, at as u64).unwrap();
    assert!(!crate::offload::lowbit_holds_head(tmp.path(), spec).unwrap());

    Qwen4ExpModel::load_with(tmp.path(), copy_options(spec, Some(400 << 20))).unwrap();
}

#[test]
fn a_copy_of_another_head_is_not_served() {
    let tmp = replica_with_head();
    let spec = LowbitSpec::new(2, 32, Recipe::LeastSquares).unwrap();
    crate::offload::build_lowbit_pack(tmp.path(), 4, spec).unwrap();
    Qwen4ExpModel::load_with(tmp.path(), head_copy_options(spec)).unwrap();
    let file = tmp.path().join("mtp.safetensors");
    let mut bytes = std::fs::read(&file).unwrap();
    bytes.push(0);
    std::fs::write(&file, bytes).unwrap();
    assert!(Qwen4ExpModel::load_with(tmp.path(), head_copy_options(spec)).is_err());
}
