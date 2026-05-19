//! ICB PoC #2 — real `affine4_qmv_fast_bf16in_bf16out` kernel via ICB.
//!
//! Validates the trivial-kernel ICB savings projection (PoC #1 in
//! `icb_microbench.rs`) against an actual production-shape kernel:
//!   - 27B Dense `o_proj` shape: out=5120, in=5120, batch=1
//!   - qmv_fast architecture (NSG=2 RPS=4 VPT=16, 64 threads/TG, 5120/8=640
//!     row groups → grid 1×640×1)
//!   - All 7 buffers: packed (4-bit weights), bf16 scales/biases, bf16 x
//!     and y, plus dims/batch as ICB-compatible Buffer (replacing
//!     `set_bytes_directly`).
//!
//! Bench design:
//!   - **Bench A (standard)**: 64 dispatches per CB, each setting pipeline
//!     + 7 buffers + dispatchThreadgroups (mimics 64 decoder layers).
//!   - **Bench B (ICB)**: ICB pre-recorded with 64 commands; per iter,
//!     1 `executeCommandsInBuffer(icb, range=0..64)`. Same buffers.
//!
//! Output verification: GPU writes match between A and B (bit-identical).
//!
//! PoC #1 (no-op kernel) projected ~+2 µs/op savings → 1.92 ms/token at
//! 960 dispatches. Real qmv_fast may differ — set_bytes_directly is gone
//! in this test (we use Buffer for both A and B), and the larger grid +
//! pipeline state may add fixed encoding cost. This test answers: "how
//! much does ICB save on the real production kernel?"

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

const SHADER_SRC: &str = include_str!("../src/shaders/affine4.metal");

#[repr(C)]
#[derive(Clone, Copy)]
struct Affine4Dims {
    out_features: u32,
    in_features: u32,
}

fn make_device_queue() -> Option<(
    Retained<ProtocolObject<dyn MTLDevice>>,
    Retained<ProtocolObject<dyn MTLCommandQueue>>,
)> {
    let device = MTLCreateSystemDefaultDevice()?;
    let queue = device.newCommandQueue()?;
    Some((device, queue))
}

fn make_qmv_fast_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
    let opts = MTLCompileOptions::new();
    let src = objc2_foundation::NSString::from_str(SHADER_SRC);
    let library = device
        .newLibraryWithSource_options_error(&src, Some(&opts))
        .expect("affine4.metal compile failed");

    let name = objc2_foundation::NSString::from_str("affine4_qmv_fast_bf16in_bf16out");
    let function = library
        .newFunctionWithName(&name)
        .expect("kernel affine4_qmv_fast_bf16in_bf16out not found");

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
            .expect("pipeline failed")
    }
}

fn alloc_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: usize,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .expect("buffer alloc")
}

fn alloc_with_data<T: Copy>(
    device: &ProtocolObject<dyn MTLDevice>,
    data: &[T],
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let bytes = std::mem::size_of_val(data);
    let buf = alloc_buffer(device, bytes);
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr() as *const u8,
            buf.contents().as_ptr() as *mut u8,
            bytes,
        );
    }
    buf
}

fn synth_packed(out: usize, ins: usize, seed: u32) -> Vec<u32> {
    let n = out * ins / 8;
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            s
        })
        .collect()
}

fn synth_scales(out: usize, ins: usize, seed: u32) -> Vec<u16> {
    let n = out * ins / 64;
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            // Map to small bf16 around 1.0 to avoid overflow in result.
            let f = ((s >> 8) & 0xff) as f32 / 256.0 * 0.01 + 0.01;
            (f.to_bits() >> 16) as u16
        })
        .collect()
}

fn synth_biases(out: usize, ins: usize, seed: u32) -> Vec<u16> {
    let n = out * ins / 64;
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            let f = ((s >> 8) & 0xff) as f32 / 256.0 * 0.01 - 0.005;
            (f.to_bits() >> 16) as u16
        })
        .collect()
}

fn synth_x_bf16(n: usize, seed: u32) -> Vec<u16> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            let f = ((s >> 8) & 0xff) as f32 / 256.0 - 0.5;
            (f.to_bits() >> 16) as u16
        })
        .collect()
}

const N_PER_CB: usize = 64;

fn run_64_standard(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    weight_packed: &Retained<ProtocolObject<dyn MTLBuffer>>,
    weight_scales: &Retained<ProtocolObject<dyn MTLBuffer>>,
    weight_biases: &Retained<ProtocolObject<dyn MTLBuffer>>,
    x_buf: &Retained<ProtocolObject<dyn MTLBuffer>>,
    y_buf: &Retained<ProtocolObject<dyn MTLBuffer>>,
    dims_buf: &Retained<ProtocolObject<dyn MTLBuffer>>,
    batch_buf: &Retained<ProtocolObject<dyn MTLBuffer>>,
    grid: MTLSize,
    tg: MTLSize,
) {
    let cb = queue.commandBuffer().expect("commandBuffer");
    let enc = cb.computeCommandEncoder().expect("encoder");
    for _ in 0..N_PER_CB {
        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(weight_packed), 0, 0);
            enc.setBuffer_offset_atIndex(Some(weight_scales), 0, 1);
            enc.setBuffer_offset_atIndex(Some(weight_biases), 0, 2);
            enc.setBuffer_offset_atIndex(Some(x_buf), 0, 3);
            enc.setBuffer_offset_atIndex(Some(y_buf), 0, 4);
            enc.setBuffer_offset_atIndex(Some(dims_buf), 0, 5);
            enc.setBuffer_offset_atIndex(Some(batch_buf), 0, 6);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    }
    enc.endEncoding();
    cb.commit();
    unsafe { cb.waitUntilCompleted() };
}

fn run_64_icb(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    icb: &ProtocolObject<dyn MTLIndirectCommandBuffer>,
    use_resources: &[&Retained<ProtocolObject<dyn MTLBuffer>>],
) {
    let cb = queue.commandBuffer().expect("commandBuffer");
    let enc = cb.computeCommandEncoder().expect("encoder");
    let usage = MTLResourceUsage(MTLResourceUsage::Read.0 | MTLResourceUsage::Write.0);
    for buf in use_resources {
        let res: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(&***buf);
        enc.useResource_usage(res, usage);
    }
    unsafe {
        enc.executeCommandsInBuffer_withRange(
            icb,
            NSRange {
                location: 0,
                length: N_PER_CB,
            },
        );
    }
    enc.endEncoding();
    cb.commit();
    unsafe { cb.waitUntilCompleted() };
}

fn record_64_icb(
    device: &ProtocolObject<dyn MTLDevice>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    weight_packed: &Retained<ProtocolObject<dyn MTLBuffer>>,
    weight_scales: &Retained<ProtocolObject<dyn MTLBuffer>>,
    weight_biases: &Retained<ProtocolObject<dyn MTLBuffer>>,
    x_buf: &Retained<ProtocolObject<dyn MTLBuffer>>,
    y_buf: &Retained<ProtocolObject<dyn MTLBuffer>>,
    dims_buf: &Retained<ProtocolObject<dyn MTLBuffer>>,
    batch_buf: &Retained<ProtocolObject<dyn MTLBuffer>>,
    grid: MTLSize,
    tg: MTLSize,
) -> Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>> {
    let desc = MTLIndirectCommandBufferDescriptor::new();
    desc.setCommandTypes(MTLIndirectCommandType::ConcurrentDispatch);
    desc.setInheritPipelineState(false);
    desc.setInheritBuffers(false);
    desc.setMaxKernelBufferBindCount(7);

    let icb = unsafe {
        device.newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
            &desc,
            N_PER_CB,
            MTLResourceOptions::StorageModePrivate,
        )
    }
    .expect("ICB alloc");

    for ci in 0..N_PER_CB {
        let icc = unsafe { icb.indirectComputeCommandAtIndex(ci) };
        icc.setComputePipelineState(pipeline);
        unsafe {
            icc.setKernelBuffer_offset_atIndex(weight_packed, 0, 0);
            icc.setKernelBuffer_offset_atIndex(weight_scales, 0, 1);
            icc.setKernelBuffer_offset_atIndex(weight_biases, 0, 2);
            icc.setKernelBuffer_offset_atIndex(x_buf, 0, 3);
            icc.setKernelBuffer_offset_atIndex(y_buf, 0, 4);
            icc.setKernelBuffer_offset_atIndex(dims_buf, 0, 5);
            icc.setKernelBuffer_offset_atIndex(batch_buf, 0, 6);
        }
        icc.concurrentDispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    }

    icb
}

#[test]
fn icb_qmv_fast_real_kernel_microbench() {
    let (device, queue) = match make_device_queue() {
        Some(t) => t,
        None => {
            eprintln!("[skip] no Metal device");
            return;
        }
    };

    let pipeline = make_qmv_fast_pipeline(&device);

    // 27B Dense o_proj shape — most common decode matmul.
    let out: usize = 5120;
    let ins: usize = 5120;
    let batch: usize = 1;

    let packed = synth_packed(out, ins, 0xDEADBEEF);
    let scales = synth_scales(out, ins, 0xCAFEBABE);
    let biases = synth_biases(out, ins, 0x12345678);
    let x_bf16 = synth_x_bf16(batch * ins, 0xFADEFADE);

    let weight_packed = alloc_with_data(&device, &packed);
    let weight_scales = alloc_with_data(&device, &scales);
    let weight_biases = alloc_with_data(&device, &biases);
    let x_buf = alloc_with_data(&device, &x_bf16);
    let y_buf_a = alloc_buffer(&device, batch * out * 2); // bf16 = 2 bytes
    let y_buf_b = alloc_buffer(&device, batch * out * 2);

    let dims = Affine4Dims {
        out_features: out as u32,
        in_features: ins as u32,
    };
    let dims_buf = alloc_with_data(&device, &[dims]);
    let batch_u32 = batch as u32;
    let batch_buf = alloc_with_data(&device, &[batch_u32]);

    // qmv_fast dispatch geometry: NSG=2, RPS=4, 64 threads/TG.
    // grid.height = ceil(out / (NSG * RPS)) = out / 8.
    let grid = MTLSize {
        width: batch,
        height: out / 8,
        depth: 1,
    };
    let tg = MTLSize {
        width: 64,
        height: 1,
        depth: 1,
    };

    // ── Bit-identity check first ────────────────────────────────────────
    let icb = record_64_icb(
        &device,
        &pipeline,
        &weight_packed,
        &weight_scales,
        &weight_biases,
        &x_buf,
        &y_buf_b,
        &dims_buf,
        &batch_buf,
        grid,
        tg,
    );

    // Run A once into y_buf_a, B once into y_buf_b
    run_64_standard(
        &queue,
        &pipeline,
        &weight_packed,
        &weight_scales,
        &weight_biases,
        &x_buf,
        &y_buf_a,
        &dims_buf,
        &batch_buf,
        grid,
        tg,
    );
    let resources = [
        &weight_packed,
        &weight_scales,
        &weight_biases,
        &x_buf,
        &y_buf_b,
        &dims_buf,
        &batch_buf,
    ];
    run_64_icb(&queue, &icb, &resources);

    let y_a_ptr = y_buf_a.contents().as_ptr() as *const u16;
    let y_b_ptr = y_buf_b.contents().as_ptr() as *const u16;
    let mut diffs = 0usize;
    for i in 0..(out * batch) {
        let a = unsafe { *y_a_ptr.add(i) };
        let b = unsafe { *y_b_ptr.add(i) };
        if a != b {
            diffs += 1;
        }
    }
    eprintln!();
    eprintln!("=== ICB PoC #2 — real qmv_fast_bf16in_bf16out kernel ===");
    eprintln!("Shape: out={out} in={ins} batch={batch}, N_PER_CB={N_PER_CB}");
    eprintln!(
        "Bit-identical (A vs B output): {} / {} (diffs={})",
        out * batch - diffs,
        out * batch,
        diffs
    );

    if diffs > 0 {
        panic!("ICB output diverged from standard — kernel correctness broken");
    }

    // ── Bench ────────────────────────────────────────────────────────────
    let warmup = 16;
    for _ in 0..warmup {
        run_64_standard(
            &queue,
            &pipeline,
            &weight_packed,
            &weight_scales,
            &weight_biases,
            &x_buf,
            &y_buf_a,
            &dims_buf,
            &batch_buf,
            grid,
            tg,
        );
        run_64_icb(&queue, &icb, &resources);
    }

    let iters = 100;

    let t0 = Instant::now();
    for _ in 0..iters {
        run_64_standard(
            &queue,
            &pipeline,
            &weight_packed,
            &weight_scales,
            &weight_biases,
            &x_buf,
            &y_buf_a,
            &dims_buf,
            &batch_buf,
            grid,
            tg,
        );
    }
    let mean_a_us = t0.elapsed().as_secs_f64() * 1.0e6 / iters as f64;

    let t1 = Instant::now();
    for _ in 0..iters {
        run_64_icb(&queue, &icb, &resources);
    }
    let mean_b_us = t1.elapsed().as_secs_f64() * 1.0e6 / iters as f64;

    let t2 = Instant::now();
    for _ in 0..iters {
        run_64_standard(
            &queue,
            &pipeline,
            &weight_packed,
            &weight_scales,
            &weight_biases,
            &x_buf,
            &y_buf_a,
            &dims_buf,
            &batch_buf,
            grid,
            tg,
        );
    }
    let mean_a2_us = t2.elapsed().as_secs_f64() * 1.0e6 / iters as f64;
    let mean_a_combined = (mean_a_us + mean_a2_us) / 2.0;

    let savings_us_per_op = (mean_a_combined - mean_b_us) / N_PER_CB as f64;
    let savings_pct_total = 100.0 * (mean_a_combined - mean_b_us) / mean_a_combined;

    eprintln!();
    eprintln!("Per-CB timing (mean over {iters} iters, {N_PER_CB} dispatches/CB):");
    eprintln!("  Standard pass 1: {mean_a_us:.2} µs/CB");
    eprintln!("  ICB execute:     {mean_b_us:.2} µs/CB");
    eprintln!("  Standard pass 2: {mean_a2_us:.2} µs/CB");
    eprintln!("  Standard mean:   {mean_a_combined:.2} µs/CB");
    eprintln!();
    eprintln!(
        "Per-dispatch CPU savings: {savings_us_per_op:+.2} µs/op ({savings_pct_total:+.1}% per CB)"
    );
    eprintln!();

    // Project to 27B Dense decode (960 dispatches/token, 67 ms baseline).
    let savings_ms_per_token = savings_us_per_op * 960.0 / 1000.0;
    let new_ms = (67.0 - savings_ms_per_token).max(1.0);
    let new_tps = 1000.0 / new_ms;
    let pct_throughput = 100.0 * (new_tps - 1000.0 / 67.0) / (1000.0 / 67.0);
    eprintln!("Projected 27B Dense decode (real kernel basis):");
    eprintln!("  Baseline: 67.00 ms/token = 14.93 tok/s");
    eprintln!("  Post-ICB: {new_ms:.2} ms/token = {new_tps:.2} tok/s ({pct_throughput:+.1}%)");
    eprintln!();

    // Compare with PoC #1 trivial-kernel projection (~+2 µs/op = +1.92 ms).
    let poc1_us_per_op = 2.0;
    let real_vs_trivial_ratio = savings_us_per_op / poc1_us_per_op;
    eprintln!("Real kernel vs PoC #1 trivial: {real_vs_trivial_ratio:.2}× ICB savings",);
    if real_vs_trivial_ratio < 1.0 {
        eprintln!(
            "  → Real kernel ICB savings are LOWER than trivial — set_bytes_directly was likely cheaper"
        );
    } else if real_vs_trivial_ratio < 2.0 {
        eprintln!("  → Real kernel ICB savings ~match trivial");
    } else {
        eprintln!("  → Real kernel ICB savings significantly larger than trivial");
    }
}
