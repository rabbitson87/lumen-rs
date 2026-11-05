//! Which MLX kernels compute a row the same whatever else is in its call.
//!
//! MLX picks a kernel, and how it splits its sums, from how many rows share a
//! call (the matmul and quantized-matmul dispatch, split-K, the attention
//! vector/full split). A row that goes through another kernel in a shorter
//! call comes out a few bits different. That is what decides where lumen may
//! cut a prefill and still match a cold one (see `session_feed`): MLX 0.32's
//! split-K made a cut inside a chunk inexact, and nothing failed until a
//! weight-gated rollback test was run by hand.
//!
//! This test pins the regimes on synthetic tensors, in seconds and without
//! weights, so the next MLX bump that moves one fails here. Each op is run on
//! the first `m` rows of an input and compared, bit for bit, with the same
//! rows of one 2048-row call; the row counts where they match are the regime.
//! The tables are measured, per device class: M5 parts take other kernels
//! (a wider vector limit, a head-dim-split attention), so a device without a
//! table prints what it measures and skips.
//!
//! What the M3 table says about lumen's cut rule:
//! - A 4-bit projection with few outputs (the linear-attention gates, 32)
//!   matches only inside a whole 2048-row call: split-K divides its sums by
//!   row count. So a cut inside a chunk is inexact on every hybrid model, and
//!   only the grid's own cuts are safe.
//! - Head-dim-256 attention (Qwen 3.5/3.6 full attention) matches no piece
//!   but a single row; head-dim 128 matches pieces of 16 rows or more.
//! - Sorted `gather_qmm` matches from 32 rows.
//! - An MTP verify of 2-8 rows equals their one-row decodes at some key
//!   lengths and not at 1024 keys (where the vector kernel changes, inferred).
#![cfg(feature = "mlx-native")]

use mlx_rs::fast::ScaledDotProductAttentionMask;
use mlx_rs::ops::indexing::{Ellipsis, IndexOp};
use mlx_rs::{Array, Dtype, random};

const MREF: i32 = 2048;
const MS: [i32; 13] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 600, 1024, 2048];
const K: i32 = 4096;

fn normal(shape: &[i32], std_dev: f32, seed: u64) -> Array {
    let key = random::key(seed).unwrap();
    random::normal::<f32>(shape, None, Some(std_dev), &key)
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap()
}

fn bits(a: &Array) -> Vec<u32> {
    let f = a.as_dtype(Dtype::Float32).unwrap();
    f.eval().unwrap();
    f.as_slice::<f32>().iter().map(|v| v.to_bits()).collect()
}

/// The row counts in [`MS`] at which the first rows of `op(x[..m])` equal the
/// same rows of `op(x[..MREF])`.
fn matching_row_counts(x: &Array, op: impl Fn(&Array) -> Array) -> Vec<i32> {
    let reference = op(&x.index((..MREF, Ellipsis)));
    MS.iter()
        .copied()
        .filter(|&m| {
            let rows = m.min(8);
            let piece = op(&x.index((..m, Ellipsis)));
            bits(&piece.index((..rows, Ellipsis))) == bits(&reference.index((..rows, Ellipsis)))
        })
        .collect()
}

/// Attention rows `[end - piece, end)` against keys `[0, end)` in one call,
/// compared with the same rows of one causal prefill over the first 2048
/// positions: the piece a resume after a cut at `end - piece` runs. Returns the
/// `(end, piece)` pairs that match.
fn matching_attention_pieces(head_dim: i32, seed: u64) -> Vec<(i32, i32)> {
    let heads = 8;
    let q = normal(&[1, heads, MREF, head_dim], 1.0, seed);
    let k = normal(&[1, heads, MREF, head_dim], 1.0, seed + 1);
    let v = normal(&[1, heads, MREF, head_dim], 1.0, seed + 2);
    let scale = (head_dim as f32).powf(-0.5);
    let attend = |q: &Array, keys: i32, causal: bool| {
        let mask = causal.then_some(ScaledDotProductAttentionMask::Causal);
        mlx_rs::fast::scaled_dot_product_attention(
            q,
            k.index((Ellipsis, ..keys, ..)),
            v.index((Ellipsis, ..keys, ..)),
            scale,
            mask,
            None,
        )
        .unwrap()
    };
    let reference = attend(&q, MREF, true);
    let mut out = Vec::new();
    for end in [600, 1024, 1500] {
        for piece in [1, 4, 8, 16, 64, 600] {
            if piece > end {
                continue;
            }
            let rows = q.index((Ellipsis, end - piece..end, ..));
            let got = attend(&rows, end, piece > 1);
            let want = reference.index((Ellipsis, end - piece..end, ..));
            if bits(&got) == bits(&want) {
                out.push((end, piece));
            }
        }
    }
    out
}

/// Attention rows `[end - rows, end)` in one causal call, compared with each
/// of them decoded alone (one query against the keys up to it): what an MTP
/// verify of `rows` tokens computes against what plain decode does. Returns
/// the `(end, rows)` pairs that match.
fn verify_rows_matching_decode(head_dim: i32, seed: u64) -> Vec<(i32, i32)> {
    let heads = 8;
    let q = normal(&[1, heads, MREF, head_dim], 1.0, seed);
    let k = normal(&[1, heads, MREF, head_dim], 1.0, seed + 1);
    let v = normal(&[1, heads, MREF, head_dim], 1.0, seed + 2);
    let scale = (head_dim as f32).powf(-0.5);
    let attend = |from: i32, to: i32| {
        let mask = (to - from > 1).then_some(ScaledDotProductAttentionMask::Causal);
        mlx_rs::fast::scaled_dot_product_attention(
            q.index((Ellipsis, from..to, ..)),
            k.index((Ellipsis, ..to, ..)),
            v.index((Ellipsis, ..to, ..)),
            scale,
            mask,
            None,
        )
        .unwrap()
    };
    let mut out = Vec::new();
    for end in [600, 1024, 1500] {
        for rows in [2, 3, 4, 8, 16] {
            let verify = attend(end - rows, end);
            let all = (0..rows).all(|r| {
                let at = end - rows + r;
                bits(&verify.index((Ellipsis, r..r + 1, ..))) == bits(&attend(at, at + 1))
            });
            if all {
                out.push((end, rows));
            }
        }
    }
    out
}

/// The regimes, measured on this machine and printed in the form the tables
/// below take.
fn measure() -> Vec<(String, String)> {
    let x = normal(&[MREF, K], 0.5, 0);
    let mut rows = Vec::new();
    for n in [32, 1024, 4096] {
        let w = normal(&[n, K], 0.02, 10 + n as u64);
        let (wq, s, b) = mlx_rs::ops::quantize(&w, 64, 4).unwrap();
        rows.push((
            format!("qmm affine4 N={n}"),
            format!(
                "{:?}",
                matching_row_counts(&x, |t| {
                    mlx_rs::ops::quantized_matmul(t, &wq, &s, &b, true, 64, 4).unwrap()
                })
            ),
        ));
        let wt = w.t();
        rows.push((
            format!("bf16 matmul N={n}"),
            format!(
                "{:?}",
                matching_row_counts(&x, |t| mlx_rs::ops::matmul(t, &wt).unwrap())
            ),
        ));
    }
    let ones = Array::ones::<f32>(&[K])
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    rows.push((
        "rms_norm".into(),
        format!(
            "{:?}",
            matching_row_counts(&x, |t| mlx_rs::fast::rms_norm(t, &ones, 1e-6).unwrap())
        ),
    ));
    // Routed experts as the MoE layers call them: each token's top-k routes
    // sorted by expert, one sorted `gather_qmm`.
    let (experts, top_k, n_out) = (64, 8, 512);
    let we = normal(&[experts, n_out, K], 0.02, 20);
    let (weq, wes, web) = mlx_rs::ops::quantize(&we, 64, 4).unwrap();
    let route =
        random::randint::<_, u32>(0, experts, &[MREF, top_k], &random::key(21).unwrap()).unwrap();
    let moe = |t: &Array| {
        let m = t.shape()[0];
        let idx = route.index((..m, ..)).flatten(None, None).unwrap();
        let order = mlx_rs::ops::argsort(&idx).unwrap();
        let inverse = mlx_rs::ops::argsort(&order).unwrap();
        let xs = mlx_rs::ops::expand_dims(t, -2).unwrap();
        let xs = mlx_rs::ops::repeat_axis::<f32>(xs, top_k, 0).unwrap();
        let xs = xs.take_axis(&order, 0).unwrap();
        let ids = idx.take_axis(&order, 0).unwrap();
        let y =
            mlx_rs::ops::gather_qmm(&xs, &weq, &wes, &web, None, &ids, true, 64, 4, true).unwrap();
        y.take_axis(&inverse, 0)
            .unwrap()
            .reshape(&[m, top_k, n_out])
            .unwrap()
    };
    rows.push((
        "gather_qmm sorted E=64 k=8".into(),
        format!("{:?}", matching_row_counts(&x, moe)),
    ));
    for head_dim in [128, 256] {
        rows.push((
            format!("sdpa d{head_dim} (end, piece)"),
            format!(
                "{:?}",
                matching_attention_pieces(head_dim, 30 + head_dim as u64)
            ),
        ));
        rows.push((
            format!("sdpa d{head_dim} verify = decode (end, rows)"),
            format!(
                "{:?}",
                verify_rows_matching_decode(head_dim, 40 + head_dim as u64)
            ),
        ));
    }
    rows
}

/// What [`measure`] reports on an Apple M3 Max with the MLX fork this crate
/// pins.
const M3: &[(&str, &str)] = &[
    ("qmm affine4 N=32", "[2048]"),
    (
        "bf16 matmul N=32",
        "[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 600, 1024, 2048]",
    ),
    ("qmm affine4 N=1024", "[512, 600, 1024, 2048]"),
    ("bf16 matmul N=1024", "[1, 600, 1024, 2048]"),
    ("qmm affine4 N=4096", "[128, 256, 512, 600, 1024, 2048]"),
    ("bf16 matmul N=4096", "[256, 512, 600, 1024, 2048]"),
    (
        "rms_norm",
        "[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 600, 1024, 2048]",
    ),
    (
        "gather_qmm sorted E=64 k=8",
        "[32, 64, 128, 256, 512, 600, 1024, 2048]",
    ),
    (
        "sdpa d128 (end, piece)",
        "[(600, 16), (600, 64), (600, 600), (1024, 16), (1024, 64), (1024, 600), (1500, 16), \
         (1500, 64), (1500, 600)]",
    ),
    (
        "sdpa d128 verify = decode (end, rows)",
        "[(600, 2), (600, 3), (600, 4), (600, 8), (1500, 2), (1500, 3), (1500, 4)]",
    ),
    ("sdpa d256 (end, piece)", "[(600, 1)]"),
    (
        "sdpa d256 verify = decode (end, rows)",
        "[(600, 2), (600, 3), (600, 4), (600, 8), (1500, 2), (1500, 3), (1500, 4)]",
    ),
];

/// The device class a table is recorded for, from the CPU brand string.
fn device_class() -> Option<&'static [(&'static str, &'static str)]> {
    let out = std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout)
        .contains("Apple M3")
        .then_some(M3)
}

#[test]
fn row_regimes_are_the_measured_ones() {
    let table = measure();
    let Some(expected) = device_class() else {
        eprintln!("no row-regime table for this device; it measures:");
        for (op, matches) in &table {
            eprintln!("    (\"{op}\", \"{matches}\"),");
        }
        return;
    };
    let moved: Vec<String> = expected
        .iter()
        .zip(&table)
        .filter(|((_, want), (_, got))| want != got)
        .map(|((op, want), (_, got))| format!("  {op}: was {want}, now {got}"))
        .collect();
    assert_eq!(
        expected.len(),
        table.len(),
        "the table lists other ops than measure"
    );
    assert!(
        moved.is_empty(),
        "MLX's row regimes moved. Check that session_feed's cut rule (only the prefill \
         grid's own cuts) and the MTP verify still hold, then record the new table:\n{}",
        moved.join("\n")
    );
}
