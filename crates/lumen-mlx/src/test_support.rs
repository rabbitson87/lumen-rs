//! Helpers the GPU test modules share: seeded bf16 inputs and the agreement
//! metric the attention parity tests report.

use mlx_rs::{Array, Dtype, random};

/// Standard-normal samples drawn in f32 from `random::key(seed)`, stored as
/// bf16 — the activation dtype the models run in.
pub(crate) fn bf16_normal(shape: &[i32], seed: u64) -> Array {
    let key = random::key(seed).unwrap();
    random::normal::<f32>(shape, None, None, &key)
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap()
}

/// [`bf16_normal`] with standard deviation `std_dev`.
pub(crate) fn bf16_normal_std(shape: &[i32], std_dev: f32, seed: u64) -> Array {
    let key = random::key(seed).unwrap();
    random::normal::<f32>(shape, None, Some(std_dev), &key)
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap()
}

/// Cosine over every element, and the largest absolute difference.
/// Flattened first: an attention output can be a strided view, and
/// `as_slice` reads memory order.
pub(crate) fn agreement(a: &Array, b: &Array) -> (f64, f32) {
    let a = a.as_dtype(Dtype::Float32).unwrap().reshape(&[-1]).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap().reshape(&[-1]).unwrap();
    a.eval().unwrap();
    b.eval().unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let (mut dot, mut na, mut nb, mut worst) = (0f64, 0f64, 0f64, 0f32);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
        worst = worst.max((x - y).abs());
    }
    (dot / (na.sqrt() * nb.sqrt()), worst)
}
