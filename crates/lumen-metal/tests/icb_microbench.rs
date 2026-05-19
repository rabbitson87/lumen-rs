//! Indirect Command Buffer (ICB) PoC microbench.
//!
//! Compares per-iter CPU encoding cost of:
//!   A. Standard pattern: cb → encoder → setComputePipelineState
//!      → 7 × setBuffer → dispatchThreadgroups → endEncoding → commit → wait
//!   B. ICB pattern: ICB pre-recorded once (pipeline + 7 buffers + dispatch);
//!      per iter: cb → encoder → 7 × useResource → executeCommandsInBuffer
//!      → endEncoding → commit → wait
//!
//! Both paths run the IDENTICAL GPU dispatch (a no-op kernel that touches all
//! 7 buffers — same memory traffic, same threadgroup count). Difference is
//! purely the CPU-side encoding cost.
//!
//! Hypothesis: 27B Dense decode is dispatch-bound (Xcode profile shows GPU
//! ~27% active, 73% idle waiting for CPU encoding). Per-token: 64 layers ×
//! ~15 ops/layer = 960 dispatches × ~50 µs CPU encode = ~48 ms/token wasted.
//! If ICB encoding is significantly cheaper, layer-level integration could
//! recover this idle time.
//!
//! Acceptance gate: ICB execute (B) is at least 30% faster per iter than
//! standard encoding (A) on Apple Silicon M3 Max.

#![allow(unexpected_cfgs)]

use std::time::Instant;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineDescriptor, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLIndirectCommandBuffer,
    MTLIndirectCommandBufferDescriptor, MTLIndirectCommandType, MTLIndirectComputeCommand,
    MTLLibrary, MTLPipelineOption, MTLResource, MTLResourceOptions, MTLResourceUsage, MTLSize,
};

const NOOP_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// 7-buffer no-op kernel matching qmv_fast's buffer arity.
// Touches every buffer once so the GPU memory traffic is realistic but the
// arithmetic is trivial — isolates CPU encoding cost from GPU compute time.
kernel void noop_7bufs(
    device const uint* a [[buffer(0)]],
    device const uint* b [[buffer(1)]],
    device const uint* c [[buffer(2)]],
    device const uint* d [[buffer(3)]],
    device       uint* e [[buffer(4)]],
    device const uint* f [[buffer(5)]],
    device const uint* g [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid == 0) {
        e[0] = a[0] + b[0] + c[0] + d[0] + f[0] + g[0];
    }
}
"#;

fn make_device_queue() -> Option<(
    Retained<ProtocolObject<dyn MTLDevice>>,
    Retained<ProtocolObject<dyn MTLCommandQueue>>,
)> {
    let device = MTLCreateSystemDefaultDevice()?;
    let queue = device.newCommandQueue()?;
    Some((device, queue))
}

fn make_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
    let opts = MTLCompileOptions::new();
    let src = objc2_foundation::NSString::from_str(NOOP_SHADER);
    let library = unsafe {
        device
            .newLibraryWithSource_options_error(&src, Some(&opts))
            .expect("newLibraryWithSource failed")
    };
    let name = objc2_foundation::NSString::from_str("noop_7bufs");
    let function = library
        .newFunctionWithName(&name)
        .expect("function not found");

    // Pipeline must support ICB recording. Plain
    // `newComputePipelineStateWithFunction_error` produces a pipeline that
    // crashes when used in `MTLIndirectComputeCommand::setComputePipelineState`.
    let desc = MTLComputePipelineDescriptor::new();
    desc.setComputeFunction(Some(&function));
    desc.setSupportIndirectCommandBuffers(true);

    unsafe {
        device
            .newComputePipelineStateWithDescriptor_options_reflection_error(
                &desc,
                MTLPipelineOption::None,
                None,
            )
            .expect("newComputePipelineStateWithDescriptor failed")
    }
}

/// Submit an empty CB and block until done. Forces all previously committed
/// CBs in the queue to drain (Metal queue ordering guarantees in-order
/// completion).
fn drain_queue(queue: &ProtocolObject<dyn MTLCommandQueue>) {
    let cb = queue.commandBuffer().expect("commandBuffer nil");
    cb.commit();
    unsafe { cb.waitUntilCompleted() };
}

fn alloc_buf(
    device: &ProtocolObject<dyn MTLDevice>,
    n_u32: usize,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let bytes = (n_u32 * 4) as usize;
    device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .expect("newBufferWithLength failed")
}

/// One CB encoding N standard dispatches sequentially. Mimics Candle's
/// per-buffer batching where multiple ops accumulate into a single command
/// buffer (~10 ops/CB by default in lumen-rs).
fn run_standard_n(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    buffers: &[Retained<ProtocolObject<dyn MTLBuffer>>],
    grid: MTLSize,
    tg: MTLSize,
    n: usize,
    wait: bool,
) {
    let cb = queue.commandBuffer().expect("commandBuffer nil");
    let enc = cb
        .computeCommandEncoder()
        .expect("computeCommandEncoder nil");
    for _ in 0..n {
        enc.setComputePipelineState(pipeline);
        for (i, buf) in buffers.iter().enumerate() {
            unsafe {
                enc.setBuffer_offset_atIndex(Some(buf), 0, i);
            }
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    }
    enc.endEncoding();
    cb.commit();
    if wait {
        unsafe { cb.waitUntilCompleted() };
    }
}

/// One CB executing an ICB-recorded range of N commands via a single
/// `executeCommandsInBuffer` call. Replaces N × (set pipeline + 7 setBuffer +
/// dispatch) with one execute call + N × useResource (which is constant per
/// buffer regardless of N — ICB binds buffers internally at record time).
fn run_icb_n(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    icb: &ProtocolObject<dyn MTLIndirectCommandBuffer>,
    buffers: &[Retained<ProtocolObject<dyn MTLBuffer>>],
    n: usize,
    wait: bool,
) {
    let cb = queue.commandBuffer().expect("commandBuffer nil");
    let enc = cb
        .computeCommandEncoder()
        .expect("computeCommandEncoder nil");
    let usage = MTLResourceUsage(MTLResourceUsage::Read.0 | MTLResourceUsage::Write.0);
    for buf in buffers {
        let res: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(&**buf);
        enc.useResource_usage(res, usage);
    }
    unsafe {
        enc.executeCommandsInBuffer_withRange(
            icb,
            NSRange {
                location: 0,
                length: n,
            },
        );
    }
    enc.endEncoding();
    cb.commit();
    if wait {
        unsafe { cb.waitUntilCompleted() };
    }
}

/// Record `count` identical dispatches into an ICB. Real-world ICB use would
/// vary the buffer offsets per command (e.g., per-layer weight slices); here
/// every command points to the same buffers — the synthetic test isolates
/// pure encoding overhead, not ICB dynamic-args features.
fn record_icb(
    device: &ProtocolObject<dyn MTLDevice>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    buffers: &[Retained<ProtocolObject<dyn MTLBuffer>>],
    grid: MTLSize,
    tg: MTLSize,
    count: usize,
) -> Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>> {
    let desc = MTLIndirectCommandBufferDescriptor::new();
    desc.setCommandTypes(MTLIndirectCommandType::ConcurrentDispatch);
    desc.setInheritPipelineState(false);
    desc.setInheritBuffers(false);
    desc.setMaxKernelBufferBindCount(buffers.len());

    let icb = unsafe {
        device.newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
            &desc,
            count,
            MTLResourceOptions::StorageModePrivate,
        )
    }
    .expect("newIndirectCommandBufferWithDescriptor failed");

    for ci in 0..count {
        let icc = unsafe { icb.indirectComputeCommandAtIndex(ci) };
        icc.setComputePipelineState(pipeline);
        for (i, buf) in buffers.iter().enumerate() {
            unsafe {
                icc.setKernelBuffer_offset_atIndex(buf, 0, i);
            }
        }
        icc.concurrentDispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    }

    icb
}

#[test]
fn icb_vs_standard_per_iter_cost() {
    let (device, queue) = match make_device_queue() {
        Some(t) => t,
        None => {
            eprintln!("[skip] no Metal device");
            return;
        }
    };

    let pipeline = make_pipeline(&device);

    // 7 buffers, each 16 KB — realistic decode-stride buffer reads but trivial
    // memory traffic compared to the encoding overhead we're measuring.
    let buffers: Vec<_> = (0..7).map(|_| alloc_buf(&device, 4096)).collect();

    let grid = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: 64,
        height: 1,
        depth: 1,
    };

    // Sweep N (dispatches per CB): 1 (1 dispatch per CB), 10 (lumen-rs's
    // CANDLE_METAL_COMPUTE_PER_BUFFER), 64 (Apple A/B baseline), 128.
    // Higher N better amortizes the fixed CB-commit/wait latency, exposing
    // pure encoding cost differences between A and B.
    let n_sweep = [1usize, 10, 64, 128];
    let iters = 200usize;

    eprintln!();
    eprintln!(
        "=== ICB vs Standard encoding ({} iters/config, 7 buffers/op, wait per CB) ===",
        iters
    );
    eprintln!(
        "{:>6} | {:>10} | {:>10} | {:>10} | {:>10}",
        "N/CB", "STD µs", "ICB µs", "Δ µs/op", "Δ %/op"
    );
    eprintln!(
        "{:->6}-+-{:->10}-+-{:->10}-+-{:->10}-+-{:->10}",
        "", "", "", "", ""
    );

    let mut results: Vec<(usize, f64, f64, f64)> = Vec::new();
    for &n in &n_sweep {
        let icb = record_icb(&device, &pipeline, &buffers, grid, tg, n);

        for _ in 0..32 {
            run_standard_n(&queue, &pipeline, &buffers, grid, tg, n, true);
            run_icb_n(&queue, &icb, &buffers, n, true);
        }

        let t0 = Instant::now();
        for _ in 0..iters {
            run_standard_n(&queue, &pipeline, &buffers, grid, tg, n, true);
        }
        let mean_a_us = t0.elapsed().as_secs_f64() * 1.0e6 / iters as f64;

        let t1 = Instant::now();
        for _ in 0..iters {
            run_icb_n(&queue, &icb, &buffers, n, true);
        }
        let mean_b_us = t1.elapsed().as_secs_f64() * 1.0e6 / iters as f64;

        let t2 = Instant::now();
        for _ in 0..iters {
            run_standard_n(&queue, &pipeline, &buffers, grid, tg, n, true);
        }
        let mean_a2_us = t2.elapsed().as_secs_f64() * 1.0e6 / iters as f64;
        let mean_a_combined = (mean_a_us + mean_a2_us) / 2.0;

        let savings_us_per_op = (mean_a_combined - mean_b_us) / n as f64;
        let savings_pct_per_op = 100.0 * savings_us_per_op / (mean_a_combined / n as f64);

        eprintln!(
            "{:>6} | {:>10.2} | {:>10.2} | {:>+10.3} | {:>+10.2}",
            n, mean_a_combined, mean_b_us, savings_us_per_op, savings_pct_per_op
        );
        results.push((n, mean_a_combined, mean_b_us, savings_us_per_op));
    }

    eprintln!();
    eprintln!("Projection at 960 dispatches/token (27B Dense decode, 67 ms baseline):");
    for &(n, _, _, sav_us) in &results {
        if n >= 10 {
            let savings_ms = sav_us * 960.0 / 1000.0;
            let new_ms = (67.0 - savings_ms).max(1.0);
            eprintln!(
                "  N={n:>3}: Δ {sav_us:+.2} µs/op × 960 = {savings_ms:+.2} ms/token saved → ~{new_ms:.2} ms/token (~{:.1} tok/s)",
                1000.0 / new_ms
            );
        }
    }
}
