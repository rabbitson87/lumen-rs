//! ICB PoC #3 — multi-weight ICB validation.
//!
//! PoC #2 showed real qmv_fast saves +10.67 µs/op via ICB, but used the
//! SAME weight buffer for all 64 commands. Real decoder layers have
//! DIFFERENT weights per layer. This PoC validates ICB handles distinct
//! buffer references across commands correctly:
//!   - 4 distinct weight buffers (4 "layers" worth of data, same shape)
//!   - 4 distinct output buffers (one per weight)
//!   - 64 commands cycling through 4 (weight, output) pairs (16 each)
//!   - Bench A: 64 standard dispatches with cycling buffers
//!   - Bench B: 1 ICB execute (64 commands pre-bound to per-command buffers)
//!
//! Bit-identity: each of the 4 output buffers must match between A and B.
//! Per-output savings: should match PoC #2 (~10 µs/op) since ICB binds
//! distinct buffers at record time — the only difference vs PoC #2 is
//! per-command buffer variation, which costs nothing extra at execute time.
//!
//! Real integration analog: each decoder layer's per-op ICB binds that
//! layer's weight buffers; 64-layer decode → 64 ICB executes (or fewer if
//! ops grouped).

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
const N_LAYERS: usize = 4;
const COMMANDS_PER_LAYER: usize = 16;
const N_PER_CB: usize = N_LAYERS * COMMANDS_PER_LAYER; // = 64

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
        .expect("affine4.metal compile");
    let name = objc2_foundation::NSString::from_str("affine4_qmv_fast_bf16in_bf16out");
    let function = library
        .newFunctionWithName(&name)
        .expect("kernel not found");
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

struct LayerWeight {
    packed: Retained<ProtocolObject<dyn MTLBuffer>>,
    scales: Retained<ProtocolObject<dyn MTLBuffer>>,
    biases: Retained<ProtocolObject<dyn MTLBuffer>>,
    y: Retained<ProtocolObject<dyn MTLBuffer>>,
}

fn run_64_standard_cycle(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    layers: &[LayerWeight],
    x_buf: &Retained<ProtocolObject<dyn MTLBuffer>>,
    dims_buf: &Retained<ProtocolObject<dyn MTLBuffer>>,
    batch_buf: &Retained<ProtocolObject<dyn MTLBuffer>>,
    grid: MTLSize,
    tg: MTLSize,
) {
    let cb = queue.commandBuffer().expect("commandBuffer");
    let enc = cb.computeCommandEncoder().expect("encoder");
    for cmd in 0..N_PER_CB {
        let layer = &layers[cmd % N_LAYERS];
        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&layer.packed), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&layer.scales), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&layer.biases), 0, 2);
            enc.setBuffer_offset_atIndex(Some(x_buf), 0, 3);
            enc.setBuffer_offset_atIndex(Some(&layer.y), 0, 4);
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

fn record_icb_cycle(
    device: &ProtocolObject<dyn MTLDevice>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    layers: &[LayerWeight],
    x_buf: &Retained<ProtocolObject<dyn MTLBuffer>>,
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

    for cmd in 0..N_PER_CB {
        let layer = &layers[cmd % N_LAYERS];
        let icc = unsafe { icb.indirectComputeCommandAtIndex(cmd) };
        icc.setComputePipelineState(pipeline);
        unsafe {
            icc.setKernelBuffer_offset_atIndex(&layer.packed, 0, 0);
            icc.setKernelBuffer_offset_atIndex(&layer.scales, 0, 1);
            icc.setKernelBuffer_offset_atIndex(&layer.biases, 0, 2);
            icc.setKernelBuffer_offset_atIndex(x_buf, 0, 3);
            icc.setKernelBuffer_offset_atIndex(&layer.y, 0, 4);
            icc.setKernelBuffer_offset_atIndex(dims_buf, 0, 5);
            icc.setKernelBuffer_offset_atIndex(batch_buf, 0, 6);
        }
        icc.concurrentDispatchThreadgroups_threadsPerThreadgroup(grid, tg);
    }
    icb
}

#[test]
fn icb_multi_weight_real_kernel_microbench() {
    let (device, queue) = match make_device_queue() {
        Some(t) => t,
        None => {
            eprintln!("[skip] no Metal device");
            return;
        }
    };

    let pipeline = make_qmv_fast_pipeline(&device);

    let out: usize = 5120;
    let ins: usize = 5120;
    let batch: usize = 1;

    // 4 distinct layer weights (different seeds → different bit patterns).
    // Both A path and B path share the same 4 weights so we can A/B compare
    // each of the 4 output buffers between paths.
    let mut layers_a: Vec<LayerWeight> = Vec::new();
    let mut layers_b: Vec<LayerWeight> = Vec::new();
    for layer_idx in 0..N_LAYERS {
        let packed = synth_packed(out, ins, 0xDEADBEEF ^ (layer_idx as u32 * 0x100));
        let scales = synth_scales(out, ins, 0xCAFEBABE ^ (layer_idx as u32 * 0x100));
        let biases = synth_biases(out, ins, 0x12345678 ^ (layer_idx as u32 * 0x100));
        // Path A and B share weights but have separate output buffers so
        // bit-identity comparison is meaningful (same input + weights →
        // same output regardless of recording method).
        let p = alloc_with_data(&device, &packed);
        let s = alloc_with_data(&device, &scales);
        let b = alloc_with_data(&device, &biases);
        // For path A: keep the same buffers but allocate dedicated y_a per layer.
        let y_a = alloc_buffer(&device, batch * out * 2);
        let y_b = alloc_buffer(&device, batch * out * 2);
        layers_a.push(LayerWeight {
            packed: p.clone(),
            scales: s.clone(),
            biases: b.clone(),
            y: y_a,
        });
        layers_b.push(LayerWeight {
            packed: p,
            scales: s,
            biases: b,
            y: y_b,
        });
    }

    let x_bf16 = synth_x_bf16(batch * ins, 0xFADEFADE);
    let x_buf = alloc_with_data(&device, &x_bf16);

    let dims = Affine4Dims {
        out_features: out as u32,
        in_features: ins as u32,
    };
    let dims_buf = alloc_with_data(&device, &[dims]);
    let batch_u32 = batch as u32;
    let batch_buf = alloc_with_data(&device, &[batch_u32]);

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
    run_64_standard_cycle(
        &queue, &pipeline, &layers_a, &x_buf, &dims_buf, &batch_buf, grid, tg,
    );

    let icb = record_icb_cycle(
        &device, &pipeline, &layers_b, &x_buf, &dims_buf, &batch_buf, grid, tg,
    );
    let mut resources: Vec<&Retained<ProtocolObject<dyn MTLBuffer>>> = Vec::new();
    for layer in &layers_b {
        resources.push(&layer.packed);
        resources.push(&layer.scales);
        resources.push(&layer.biases);
        resources.push(&layer.y);
    }
    resources.push(&x_buf);
    resources.push(&dims_buf);
    resources.push(&batch_buf);
    run_64_icb(&queue, &icb, &resources);

    eprintln!();
    eprintln!("=== ICB PoC #3 — multi-weight real qmv_fast kernel ===");
    eprintln!(
        "Shape: out={out} in={ins} batch={batch}, N_LAYERS={N_LAYERS}, \
         commands/layer={COMMANDS_PER_LAYER}, total commands={N_PER_CB}"
    );

    let mut total_diffs = 0usize;
    for layer_idx in 0..N_LAYERS {
        let y_a = &layers_a[layer_idx].y;
        let y_b = &layers_b[layer_idx].y;
        let a_ptr = y_a.contents().as_ptr() as *const u16;
        let b_ptr = y_b.contents().as_ptr() as *const u16;
        let mut diffs = 0usize;
        for i in 0..(out * batch) {
            let a = unsafe { *a_ptr.add(i) };
            let b = unsafe { *b_ptr.add(i) };
            if a != b {
                diffs += 1;
            }
        }
        eprintln!(
            "  Layer {layer_idx}: bit-identical {} / {} (diffs={})",
            out * batch - diffs,
            out * batch,
            diffs
        );
        total_diffs += diffs;
    }

    if total_diffs > 0 {
        panic!("Multi-weight ICB diverged from standard — total diffs = {total_diffs}");
    }

    // ── Bench ────────────────────────────────────────────────────────────
    let warmup = 16;
    for _ in 0..warmup {
        run_64_standard_cycle(
            &queue, &pipeline, &layers_a, &x_buf, &dims_buf, &batch_buf, grid, tg,
        );
        run_64_icb(&queue, &icb, &resources);
    }

    let iters = 100;

    let t0 = Instant::now();
    for _ in 0..iters {
        run_64_standard_cycle(
            &queue, &pipeline, &layers_a, &x_buf, &dims_buf, &batch_buf, grid, tg,
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
        run_64_standard_cycle(
            &queue, &pipeline, &layers_a, &x_buf, &dims_buf, &batch_buf, grid, tg,
        );
    }
    let mean_a2_us = t2.elapsed().as_secs_f64() * 1.0e6 / iters as f64;
    let mean_a_combined = (mean_a_us + mean_a2_us) / 2.0;

    let savings_us_per_op = (mean_a_combined - mean_b_us) / N_PER_CB as f64;
    let savings_pct_total = 100.0 * (mean_a_combined - mean_b_us) / mean_a_combined;

    eprintln!();
    eprintln!("Per-CB timing (mean over {iters} iters, 64 dispatches across 4 layers):");
    eprintln!("  Standard pass 1: {mean_a_us:.2} µs/CB");
    eprintln!("  ICB execute:     {mean_b_us:.2} µs/CB");
    eprintln!("  Standard pass 2: {mean_a2_us:.2} µs/CB");
    eprintln!("  Standard mean:   {mean_a_combined:.2} µs/CB");
    eprintln!();
    eprintln!(
        "Per-dispatch CPU savings: {savings_us_per_op:+.2} µs/op ({savings_pct_total:+.1}% per CB)"
    );
    eprintln!();

    let savings_ms_per_token = savings_us_per_op * 960.0 / 1000.0;
    let new_ms = (67.0 - savings_ms_per_token).max(1.0);
    let new_tps = 1000.0 / new_ms;
    let pct_throughput = 100.0 * (new_tps - 1000.0 / 67.0) / (1000.0 / 67.0);
    eprintln!("Projected 27B Dense decode (multi-weight basis):");
    eprintln!("  Baseline: 67.00 ms/token = 14.93 tok/s");
    eprintln!("  Post-ICB: {new_ms:.2} ms/token = {new_tps:.2} tok/s ({pct_throughput:+.1}%)");

    // Compare with PoC #2 single-weight: should match closely (~10 µs/op)
    let poc2_us_per_op = 10.67;
    let ratio = savings_us_per_op / poc2_us_per_op;
    eprintln!();
    eprintln!(
        "Multi-weight vs single-weight (PoC #2): {ratio:.2}× ICB savings (1.0× = no overhead from per-command buffer variation)"
    );
    if ratio < 0.7 {
        eprintln!(
            "  ⚠ Multi-weight significantly LOWER — per-command buffer variation adds CPU cost"
        );
    } else if ratio > 1.3 {
        eprintln!(
            "  ✓ Multi-weight HIGHER (unexpected) — measurement noise or favorable scheduling"
        );
    } else {
        eprintln!("  ✓ Match within noise — ICB scales cleanly to per-layer weight binding");
    }
}
