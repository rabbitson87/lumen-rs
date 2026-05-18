//! FFI overhead microbench — measures the cost components of an
//! `mlx-rs::try_from_op`-mediated mlx-c call.
//!
//! Goal: separate the per-call FFI overhead into:
//!   (a) `mlx_array_new` (allocator) + `mlx_array_free` (drop) — the "handle
//!       overhead" that an allocator pool would eliminate.
//!   (b) The actual op FFI dispatch (e.g. `mlx_add`).
//!   (c) Rust-side wrapper costs (`Array` struct construction, Drop).
//!
//! Hypothesis (from Phase 1.5 FFI overhead profile):
//!   3346 ops/step × ~6 μs/op pure FFI ≈ 20 ms/step overhead.
//!   If (a) is ≥1-2 μs/op, pooling saves 3-7 ms/step → ~+10-20% throughput.
//!
//! Reports per-iteration ns averaged over a tight loop. No GPU dispatch:
//! `Array::from_int` is a CPU constant, no Metal queue.
//!
//! Run:
//!   cargo run --release --features mlx-native --example bench_ffi_alloc_cost

#[cfg(feature = "mlx-native")]
fn main() {
    use mlx_rs::Array;
    use std::hint::black_box;
    use std::time::Instant;

    // Single-warmup iteration to settle thread-local init paths
    // (INIT_ERR_HANDLER's Once::call_once, atomic instrumentation reads).
    let _ = black_box(Array::from_int(1));

    // ── (1) Raw mlx_array_new + mlx_array_free, no try_from_op ─────────
    // The closest probe to the allocator's own per-call cost. Bypasses
    // try_from_op overhead.
    {
        const N: usize = 1_000_000;
        let t0 = Instant::now();
        unsafe {
            for _ in 0..N {
                let raw = mlx_sys::mlx_array_new();
                mlx_sys::mlx_array_free(raw);
            }
        }
        let elapsed = t0.elapsed();
        let ns = elapsed.as_nanos() as f64 / N as f64;
        println!("[bench-ffi] mlx_array_new + mlx_array_free      : {ns:.1} ns/iter  (N={N})");
    }

    // ── (2) Array::from_int (try_from_op + mlx_array_set_int + drop) ───
    // This is the closest approximation to "minimal op cost" — a single
    // CPU constant fill, going through the full try_from_op infrastructure.
    {
        const N: usize = 1_000_000;
        let t0 = Instant::now();
        for _ in 0..N {
            black_box(Array::from_int(1));
        }
        let elapsed = t0.elapsed();
        let ns = elapsed.as_nanos() as f64 / N as f64;
        println!("[bench-ffi] Array::from_int(1)                  : {ns:.1} ns/iter  (N={N})");
    }

    // ── (3) Single-op chain: from_int + add (no GPU work, all CPU) ────
    // Captures per-op overhead with a real op call (`mlx_add`) on small
    // CPU-resident scalars. Subtract (2) to estimate `mlx_add` FFI cost.
    {
        const N: usize = 100_000;
        let one = Array::from_int(1);
        let t0 = Instant::now();
        for _ in 0..N {
            let two = one.add(&one).expect("add");
            black_box(two);
        }
        let elapsed = t0.elapsed();
        let ns = elapsed.as_nanos() as f64 / N as f64;
        println!("[bench-ffi] one.add(&one) (no eval, lazy graph) : {ns:.1} ns/iter  (N={N})");
    }

    // ── (4) Same as (3) but force eval per iter — closer to decode path ─
    // Decode forces a sync read at end of step. Inside a step the ops
    // accumulate lazily, so this is a pessimistic estimate vs decode.
    {
        const N: usize = 50_000;
        let one = Array::from_int(1);
        let t0 = Instant::now();
        for _ in 0..N {
            let two = one.add(&one).expect("add");
            two.eval().expect("eval");
            black_box(two);
        }
        let elapsed = t0.elapsed();
        let ns = elapsed.as_nanos() as f64 / N as f64;
        println!("[bench-ffi] one.add(&one) + eval per iter       : {ns:.1} ns/iter  (N={N})");
    }

    // ── Summary ──
    println!();
    println!("─── Interpretation ───");
    println!("(1) is the raw allocator pair cost. If pooled, we save ~this per op.");
    println!("(2) is the floor for any try_from_op-mediated mlx-rs op (handle alloc + Rust wrappers).");
    println!("(3) - (2) ≈ pure mlx-c op FFI cost (lazy, no GPU dispatch).");
    println!("(4) - (3) ≈ GPU sync overhead per op.");
    println!();
    println!("Decode hot path measured: ~6 μs/op pure FFI estimate (33 ms / 3346 ops − GPU floor).");
    println!("If (1) > 1 μs (1000 ns), allocator pool is high-value lever.");
    println!("If (1) < 500 ns, pool is low-value; look elsewhere (batched dispatch, Array bypass).");
}

#[cfg(not(feature = "mlx-native"))]
fn main() {
    eprintln!("bench_ffi_alloc_cost requires the 'mlx-native' feature.");
}
