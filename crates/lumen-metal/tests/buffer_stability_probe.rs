//! MTLBuffer stability probe.
//!
//! Question: does Candle's Metal allocator return the same `MTLBuffer`
//! address across decode-style allocations of identically-shaped tensors?
//! Answer drives Phase 17.C ICB integration architecture:
//!
//!   stable    → record-once, replay-many (cheap fast path)
//!   unstable  → re-record per token (or scratch-pool + copy)
//!
//! Strategy: simulate a per-token decode "allocate intermediate, do op,
//! drop" sequence and count distinct backing buffer addresses across N
//! iterations. We probe two patterns:
//!
//!   * naive: allocate a fresh Tensor, drop, repeat → tests pure pool
//!     reuse.
//!   * forward-shaped: allocate input, run a real Affine4 dispatch (which
//!     issues a command buffer and triggers `drop_unused_buffers`), then
//!     repeat → tests whether the post-flush sweep evicts the pool.
//!
//! Run:
//!   cargo test --test buffer_stability_probe -p lumen-metal \
//!     --features model-integration --release -- --nocapture

#![cfg(feature = "model-integration")]

use candle_core::backend::BackendDevice;
use candle_core::{DType, Device, Tensor};
use std::collections::BTreeSet;

const HIDDEN: usize = 5120;
const ITERS: usize = 32;

/// Cast `Buffer` reference to a stable usize identity so we can hash/compare
/// across iterations.
fn buf_id(buf: &lumen_metal::metal::Buffer) -> usize {
    use objc2_metal::MTLBuffer as _;
    // Two reasonable identities:
    //  1. `as_ref()` Pointer → identifies the Retained protocol-object box.
    //  2. `contents()` → identifies the underlying allocation (CPU address).
    // (1) collapses if Candle returns the same Arc<Buffer> multiple times;
    // (2) tracks the underlying MTLBuffer allocation identity. We use (2)
    // because StorageModePrivate buffers report nullptr — we'll fall back
    // to the protocol-object pointer in that case.
    let contents = buf.as_ref().contents().as_ptr() as usize;
    if contents != 0 {
        contents
    } else {
        buf.as_ref() as *const _ as *const () as usize
    }
}

#[test]
fn probe_buffer_stability_naive_alloc() {
    let dev = match Device::new_metal(0) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("[skip] no Metal device");
            return;
        }
    };

    let mut ids: Vec<usize> = Vec::with_capacity(ITERS);

    for i in 0..ITERS {
        // Decode-shape BF16 hidden state.
        let t = Tensor::zeros(&[1, 1, HIDDEN], DType::BF16, &dev).unwrap();
        // Force a real allocation by touching the storage.
        let storage = t.storage_and_layout().0;
        let candle_storage = match &*storage {
            candle_core::Storage::Metal(s) => s,
            _ => panic!("expected Metal storage"),
        };
        let id = buf_id(candle_storage.buffer());
        ids.push(id);
        // `t` drops here → strong_count returns to pool.
        if i < 4 || i % 8 == 0 {
            eprintln!("  iter {i:2} id=0x{id:x}");
        }
    }

    let unique: BTreeSet<_> = ids.iter().copied().collect();
    eprintln!();
    eprintln!("=== Naive alloc probe (no command-buffer dispatch) ===");
    eprintln!("Iterations:       {ITERS}");
    eprintln!("Unique buf IDs:   {}", unique.len());
    eprintln!(
        "Stability ratio:  {:.1}%",
        100.0 * (ITERS - unique.len()) as f64 / (ITERS as f64).max(1.0)
    );
}

#[test]
fn probe_buffer_stability_with_dispatch() {
    let dev = match Device::new_metal(0) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("[skip] no Metal device");
            return;
        }
    };

    // Each iteration: allocate input → run a real elementwise op (forces a
    // command buffer dispatch + `drop_unused_buffers` sweep on the next
    // encoder) → drop → repeat. This is the pattern that decode actually
    // produces, and the sweep is the part that may evict our intermediates.
    let mut ids: Vec<usize> = Vec::with_capacity(ITERS);
    let mut out_ids: Vec<usize> = Vec::with_capacity(ITERS);

    for i in 0..ITERS {
        let x = Tensor::zeros(&[1, 1, HIDDEN], DType::BF16, &dev).unwrap();

        // Trigger a kernel dispatch on this Tensor — anything that creates
        // a fresh output tensor and ends an encoder counts.
        let y = x.affine(1.0, 0.0).unwrap();

        // Synchronize so the command buffer flushes and the next op's
        // `command_encoder()` triggers `drop_unused_buffers`.
        if let Device::Metal(md) = &dev {
            let _ = md.synchronize();
        }

        // Read identities BEFORE drop — we want to see what address Candle
        // actually returned.
        {
            let s = x.storage_and_layout().0;
            if let candle_core::Storage::Metal(ms) = &*s {
                ids.push(buf_id(ms.buffer()));
            }
        }
        {
            let s = y.storage_and_layout().0;
            if let candle_core::Storage::Metal(ms) = &*s {
                out_ids.push(buf_id(ms.buffer()));
            }
        }

        if i < 4 || i % 8 == 0 {
            let last_in = ids.last().copied().unwrap_or(0);
            let last_out = out_ids.last().copied().unwrap_or(0);
            eprintln!("  iter {i:2} x_id=0x{last_in:x}  y_id=0x{last_out:x}");
        }
        // x, y drop here.
    }

    let unique_in: BTreeSet<_> = ids.iter().copied().collect();
    let unique_out: BTreeSet<_> = out_ids.iter().copied().collect();
    eprintln!();
    eprintln!("=== Dispatch probe (op + synchronize + drop per iter) ===");
    eprintln!("Iterations:        {ITERS}");
    eprintln!("Unique x IDs:      {}", unique_in.len());
    eprintln!("Unique y IDs:      {}", unique_out.len());
    eprintln!(
        "x stability ratio: {:.1}%",
        100.0 * (ITERS - unique_in.len()) as f64 / (ITERS as f64).max(1.0)
    );
    eprintln!(
        "y stability ratio: {:.1}%",
        100.0 * (ITERS - unique_out.len()) as f64 / (ITERS as f64).max(1.0)
    );

    // Decision matrix:
    //   x stability ≥ 95% AND y stability ≥ 95% → record-once-replay-many viable
    //   either ratio < 95%  → must re-record per token OR allocate scratch
    eprintln!();
    if unique_in.len() <= ITERS / 20 + 1 && unique_out.len() <= ITERS / 20 + 1 {
        eprintln!("VERDICT: pool reuse stable → record-once architecture viable");
    } else {
        eprintln!("VERDICT: pool reuse NOT stable → must re-record or allocate scratch buffers");
    }
}
