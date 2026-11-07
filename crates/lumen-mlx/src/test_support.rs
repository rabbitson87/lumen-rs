//! Helpers the GPU test modules share: seeded bf16 inputs and the agreement
//! metric the attention parity tests report.

use std::cell::RefCell;

use mlx_rs::random::RandomState;
use mlx_rs::{Array, Dtype, random};

thread_local! {
    /// This test's key sequence. The sequence `random::seed` and a keyless `random::normal` use
    /// is process-wide, and keeps its state as a lazy array on the stream of the thread that
    /// used it last. Every test runs on its own thread, so the next test to draw from it
    /// evaluates a graph on a stream that thread does not have: "There is no Stream(gpu, N) in
    /// current thread". Sequential runs got away with it only when the new thread recycled the
    /// id of the thread MLX took for its main thread, which timing decides.
    static KEYS: RefCell<RandomState> =
        RefCell::new(RandomState::with_seed(0).expect("a key from a seed"));
}

/// Start this test's key sequence over from `seed`.
pub(crate) fn seed(seed: u64) {
    KEYS.set(RandomState::with_seed(seed).expect("a key from a seed"));
}

/// The next key of this test's sequence: what `random::seed(n)` and keyless draws would give.
pub(crate) fn key() -> Array {
    KEYS.with_borrow_mut(|keys| keys.next_key().expect("the next key"))
}

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
