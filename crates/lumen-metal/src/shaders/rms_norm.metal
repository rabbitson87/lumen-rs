// Standalone RmsNorm kernels — replaces MPSGraph-based bf16 output variant
// that was bit-non-deterministic on Apple Silicon (MPSGraph reduction
// order varies per call → 5119/5120 output bits flip across identical
// invocations, breaking the bf16 chain's R1↔R2 token determinism).
//
// Determinism contract: for identical (x, weight, eps) inputs, the kernel
// produces bit-identical output across invocations. Reduction order is
// fixed by simd_lid + simdgroup_index; fp32 arithmetic is deterministic
// per-lane on Apple Silicon GPUs.
//
// Input/output layout matches MpsRmsNormBf16Out:
//   x      : [m, hidden]  f32
//   weight : [hidden]     f32
//   y      : [m, hidden]  bf16
//
// Math: y[i, j] = bf16(x[i, j] · rsqrt(mean(x[i, :]²) + eps) · weight[j])

#include <metal_stdlib>
using namespace metal;

struct RmsNormBf16OutDims {
    uint  hidden;
    float eps;
};

// One threadgroup per row of x. Uses `THREADS=256` lanes split into
// `NSG=8` simdgroups for the cooperative Σx² reduction.
//
// Algorithm:
//   Phase A (Σx²): each thread reads `hidden / THREADS` (or up to)
//     elements, accumulates fp32 partial. simd_sum collapses to
//     32-lane partials (one per simdgroup), threadgroup memory
//     publishes them, simd_gid=0 sums NSG partials with simd_sum
//     (deterministic lane-ordered tree).
//   Phase B (write): each thread reads its slice again, computes
//     normalized + weighted, narrows to bf16 at store.
//
// Two reads of x per row (no shared memory cache) — cheap on
// unified-memory Apple Silicon: `hidden·4` bytes per row vs the
// MB-scale weight tensors that dominate decode bandwidth.
kernel void rms_norm_f32in_bf16out(
    device const float*           x        [[buffer(0)]],
    device const float*           weight   [[buffer(1)]],
    device bfloat*                y        [[buffer(2)]],
    constant RmsNormBf16OutDims&  dims     [[buffer(3)]],
    threadgroup float*            sg_partials [[threadgroup(0)]],
    uint  tg_pos    [[threadgroup_position_in_grid]],
    uint  tid       [[thread_index_in_threadgroup]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  simd_lid  [[thread_index_in_simdgroup]]
) {
    constexpr uint THREADS = 256;
    constexpr uint NSG = 8;  // THREADS / SIMD_SIZE(32)

    uint hidden = dims.hidden;
    float eps = dims.eps;
    uint row_off = tg_pos * hidden;

    // Phase A: per-thread partial Σx². Stride-THREADS access pattern
    // means each thread covers indices `tid, tid+THREADS, tid+2·THREADS, ...`.
    // For hidden not a multiple of THREADS, the tail naturally falls
    // out via the loop bound.
    float sumsq = 0.0f;
    for (uint i = tid; i < hidden; i += THREADS) {
        float v = x[row_off + i];
        sumsq = fma(v, v, sumsq);
    }

    // Reduce within simdgroup (32-lane fixed reduction tree).
    sumsq = simd_sum(sumsq);

    // Lane 0 of each simdgroup writes its partial.
    if (simd_lid == 0u) {
        sg_partials[simd_gid] = sumsq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // simd_gid=0 collects NSG partials and reduces deterministically.
    // Lanes 0..NSG-1 read the partials, others contribute 0 — final
    // simd_sum gives the full row Σx² in lane 0 of simd_gid=0.
    if (simd_gid == 0u) {
        float v = (simd_lid < NSG) ? sg_partials[simd_lid] : 0.0f;
        float total = simd_sum(v);
        if (simd_lid == 0u) {
            sg_partials[0] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_rms = rsqrt(sg_partials[0] / float(hidden) + eps);

    // Phase B: each thread normalizes its slice and stores bf16.
    for (uint i = tid; i < hidden; i += THREADS) {
        float v = x[row_off + i] * weight[i] * inv_rms;
        y[row_off + i] = bfloat(v);
    }
}

// Workstream B Phase 10 — bf16-in / bf16-out variant. Same algorithm as
// `rms_norm_f32in_bf16out` above, but reads bf16 input (widened to f32 on
// load) so the model-wide bf16 carrier stream avoids the bf16→f32 cast at
// every layernorm boundary.
//
// Determinism contract identical to the f32-in variant: reduction order
// pinned by simd_lid + simdgroup_index, all arithmetic in fp32.
//
// Input/output layout:
//   x      : [m, hidden]  bf16
//   weight : [hidden]     f32
//   y      : [m, hidden]  bf16
//
// Math (same as f32-in variant, just bf16 load):
//   y[i, j] = bf16(float(x[i, j]) · rsqrt(mean(float(x[i, :])²) + eps) · weight[j])
kernel void rms_norm_bf16in_bf16out(
    device const bfloat*          x        [[buffer(0)]],
    device const float*           weight   [[buffer(1)]],
    device bfloat*                y        [[buffer(2)]],
    constant RmsNormBf16OutDims&  dims     [[buffer(3)]],
    threadgroup float*            sg_partials [[threadgroup(0)]],
    uint  tg_pos    [[threadgroup_position_in_grid]],
    uint  tid       [[thread_index_in_threadgroup]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  simd_lid  [[thread_index_in_simdgroup]]
) {
    constexpr uint THREADS = 256;
    constexpr uint NSG = 8;

    uint hidden = dims.hidden;
    float eps = dims.eps;
    uint row_off = tg_pos * hidden;

    // Phase A: per-thread partial Σx². Bf16 input widened to f32 on load —
    // mantissa-narrow bf16 cannot accumulate hidden∈[2K..16K] reductions.
    float sumsq = 0.0f;
    for (uint i = tid; i < hidden; i += THREADS) {
        float v = float(x[row_off + i]);
        sumsq = fma(v, v, sumsq);
    }

    sumsq = simd_sum(sumsq);
    if (simd_lid == 0u) {
        sg_partials[simd_gid] = sumsq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 0u) {
        float v = (simd_lid < NSG) ? sg_partials[simd_lid] : 0.0f;
        float total = simd_sum(v);
        if (simd_lid == 0u) {
            sg_partials[0] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_rms = rsqrt(sg_partials[0] / float(hidden) + eps);

    // Phase B: each thread normalizes its slice and stores bf16. Re-reads
    // x (no shared cache) — same trade-off as the f32-in variant on
    // unified-memory Apple Silicon.
    for (uint i = tid; i < hidden; i += THREADS) {
        float v = float(x[row_off + i]) * weight[i] * inv_rms;
        y[row_off + i] = bfloat(v);
    }
}
