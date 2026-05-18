// MXFP4 fused dequant + matvec kernel.
//
// Layout matches `crate::mxfp4` (MLX storage convention):
//   packed: [out_features, in_features/8]  uint   -- 8 nibbles per word, LSB-first
//   scales: [out_features, in_features/32] uchar  -- one E8M0 exponent per 32-element group
//   x     : [in_features]                   float  -- dense activation
//   y     : [out_features]                  float  -- output
//
// One thread per output row; each thread iterates over all groups and accumulates the
// dot product on the fly, never materializing the dequantized weight tensor.

#include <metal_stdlib>
using namespace metal;

constant float E2M1_LUT[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

struct MxFp4Dims {
    uint out_features;
    uint in_features;
};

static inline float e8m0_scale_device(uchar byte) {
    if (byte == 0xFF) { return 0.0f; }
    uint bits = uint(byte) << 23;
    return as_type<float>(bits);
}

kernel void mxfp4_matvec_f32(
    device const uint*    packed  [[buffer(0)]],
    device const uchar*   scales  [[buffer(1)]],
    device const float*   x       [[buffer(2)]],
    device float*         y       [[buffer(3)]],
    constant MxFp4Dims&   dims    [[buffer(4)]],
    uint row                      [[thread_position_in_grid]]
) {
    if (row >= dims.out_features) { return; }

    uint groups         = dims.in_features / 32u;
    uint words_per_row  = dims.in_features / 8u;
    uint word_row_base  = row * words_per_row;
    uint scale_row_base = row * groups;

    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        float s = e8m0_scale_device(scales[scale_row_base + g]);
        if (s == 0.0f) { continue; }

        uint word_base = word_row_base + g * 4u;
        uint x_base    = g * 32u;
        for (uint w = 0; w < 4u; ++w) {
            uint word = packed[word_base + w];
            for (uint i = 0; i < 8u; ++i) {
                uint nib = (word >> (i * 4u)) & 0xFu;
                acc += E2M1_LUT[nib] * s * x[x_base + w * 8u + i];
            }
        }
    }
    y[row] = acc;
}

// Batched variant: grid is (out_features, batch). Each thread computes one output
// element y[b, row] = sum_k W[row, k] * x[b, k]. Weight is shared across the batch,
// so we read packed/scales once per thread and iterate in_features.
kernel void mxfp4_matmul_f32(
    device const uint*    packed  [[buffer(0)]],
    device const uchar*   scales  [[buffer(1)]],
    device const float*   x       [[buffer(2)]],
    device float*         y       [[buffer(3)]],
    constant MxFp4Dims&   dims    [[buffer(4)]],
    constant uint&        batch   [[buffer(5)]],
    uint2 gid                     [[thread_position_in_grid]]
) {
    uint row = gid.x;
    uint b   = gid.y;
    if (row >= dims.out_features || b >= batch) { return; }

    uint groups         = dims.in_features / 32u;
    uint words_per_row  = dims.in_features / 8u;
    uint word_row_base  = row * words_per_row;
    uint scale_row_base = row * groups;
    uint x_row_base     = b * dims.in_features;

    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        float s = e8m0_scale_device(scales[scale_row_base + g]);
        if (s == 0.0f) { continue; }

        uint word_base = word_row_base + g * 4u;
        uint x_base    = x_row_base + g * 32u;
        for (uint w = 0; w < 4u; ++w) {
            uint word = packed[word_base + w];
            for (uint i = 0; i < 8u; ++i) {
                uint nib = (word >> (i * 4u)) & 0xFu;
                acc += E2M1_LUT[nib] * s * x[x_base + w * 8u + i];
            }
        }
    }
    y[b * dims.out_features + row] = acc;
}

// MoE grouped matmul: one dispatch handles k expert slots, selecting the per-expert
// weight slab via an indirect `expert_indices` lookup. Grid = (out, batch, k).
//
// Buffer layout:
//   packed_all : [num_experts_total, out_features, in_features/8]  uint
//   scales_all : [num_experts_total, out_features, in_features/32] uchar
//   expert_indices : [k] uint     -- which expert each z-slot reads from
//   x          : f32
//                 if broadcast_x == 1 : [batch, in_features]        (shared across slots)
//                 if broadcast_x == 0 : [k, batch, in_features]     (per-slot band)
//   y          : [k, batch, out_features] f32
//
// Enables M3 Max to schedule all k experts in parallel instead of serializing as k
// separate command buffers; see `qwen3_5_moe_perf_plan.md` (Option I).
struct MxFp4MoeDims {
    uint out_features;
    uint in_features;
    uint batch;
    uint broadcast_x;
};

kernel void mxfp4_matmul_moe_f32(
    device const uint*     packed_all     [[buffer(0)]],
    device const uchar*    scales_all     [[buffer(1)]],
    device const uint*     expert_indices [[buffer(2)]],
    device const float*    x              [[buffer(3)]],
    device float*          y              [[buffer(4)]],
    constant MxFp4MoeDims& dims           [[buffer(5)]],
    uint3 gid                             [[thread_position_in_grid]]
) {
    uint row  = gid.x;
    uint b    = gid.y;
    uint slot = gid.z;
    if (row >= dims.out_features || b >= dims.batch) { return; }

    uint e = expert_indices[slot];

    uint groups         = dims.in_features / 32u;
    uint words_per_row  = dims.in_features / 8u;
    uint packed_expert_stride = dims.out_features * words_per_row;
    uint scale_expert_stride  = dims.out_features * groups;

    uint word_row_base  = e * packed_expert_stride + row * words_per_row;
    uint scale_row_base = e * scale_expert_stride  + row * groups;

    uint x_slot = (dims.broadcast_x != 0u) ? 0u : slot;
    uint x_row_base = x_slot * dims.batch * dims.in_features + b * dims.in_features;

    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        float s = e8m0_scale_device(scales_all[scale_row_base + g]);
        if (s == 0.0f) { continue; }
        uint word_base = word_row_base + g * 4u;
        uint x_base    = x_row_base + g * 32u;
        for (uint w = 0; w < 4u; ++w) {
            uint word = packed_all[word_base + w];
            for (uint i = 0; i < 8u; ++i) {
                uint nib = (word >> (i * 4u)) & 0xFu;
                acc += E2M1_LUT[nib] * s * x[x_base + w * 8u + i];
            }
        }
    }
    y[slot * dims.batch * dims.out_features + b * dims.out_features + row] = acc;
}

// ───────────────────────────────────────────────────────────────────────────
// V2 kernels (Phase A.1, Step 1B): uint4 + float4 vectorized loads.
//
// Same grid topology and same per-term arithmetic order as v1 (`acc += LUT * s * x`)
// — only the memory access is widened from scalar to 16-byte vector loads. The
// expectation is bit-identical output vs v1 for dense and effectively-identical
// output for MoE (parity-preserving by construction). Speedup comes from coalesced
// global-memory transactions and inner-loop unrolling enabled by knowing all 32
// nibbles up front.
//
// Kept separate from v1 so we can A/B via `LUMEN_MXFP4_KERNEL_VERSION=v1|v2` with
// zero risk of regressing the proven hot path. v1 will eventually be retired once
// v2 is verified across all in-flight shapes.
// ───────────────────────────────────────────────────────────────────────────

kernel void mxfp4_matmul_f32_v2(
    device const uint*    packed  [[buffer(0)]],
    device const uchar*   scales  [[buffer(1)]],
    device const float*   x       [[buffer(2)]],
    device float*         y       [[buffer(3)]],
    constant MxFp4Dims&   dims    [[buffer(4)]],
    constant uint&        batch   [[buffer(5)]],
    uint2 gid                     [[thread_position_in_grid]]
) {
    uint row = gid.x;
    uint b   = gid.y;
    if (row >= dims.out_features || b >= batch) { return; }

    uint groups         = dims.in_features / 32u;
    uint words_per_row  = dims.in_features / 8u;
    uint word_row_base  = row * words_per_row;
    uint scale_row_base = row * groups;
    uint x_row_base     = b * dims.in_features;

    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        float s = e8m0_scale_device(scales[scale_row_base + g]);
        if (s == 0.0f) { continue; }

        uint word_base = word_row_base + g * 4u;
        uint x_base    = x_row_base + g * 32u;

        // Vectorized loads: 4 packed words (16B) + 32 floats (128B) per group.
        // word_base is row*words_per_row + g*4, both row and g iterations leave a
        // multiple-of-4 base, so the uint4 cast is naturally aligned. x_base = b*in
        // + g*32; for batch=1 dim sizes that are multiples of 32, this is also
        // 16-byte aligned (8 float4s).
        uint4 ws = *((device const uint4*)(packed + word_base));
        device const float4* x4_ptr = (device const float4*)(x + x_base);
        float4 x0 = x4_ptr[0]; float4 x1 = x4_ptr[1];
        float4 x2 = x4_ptr[2]; float4 x3 = x4_ptr[3];
        float4 x4 = x4_ptr[4]; float4 x5 = x4_ptr[5];
        float4 x6 = x4_ptr[6]; float4 x7 = x4_ptr[7];

        // Per-term `acc += LUT[nib] * s * x[i]` preserved exactly (32 fmas per group)
        // so v2 is bit-identical with v1 modulo any compiler reordering of FMA.
        acc += E2M1_LUT[(ws.x      ) & 0xFu] * s * x0.x;
        acc += E2M1_LUT[(ws.x >>  4) & 0xFu] * s * x0.y;
        acc += E2M1_LUT[(ws.x >>  8) & 0xFu] * s * x0.z;
        acc += E2M1_LUT[(ws.x >> 12) & 0xFu] * s * x0.w;
        acc += E2M1_LUT[(ws.x >> 16) & 0xFu] * s * x1.x;
        acc += E2M1_LUT[(ws.x >> 20) & 0xFu] * s * x1.y;
        acc += E2M1_LUT[(ws.x >> 24) & 0xFu] * s * x1.z;
        acc += E2M1_LUT[(ws.x >> 28) & 0xFu] * s * x1.w;

        acc += E2M1_LUT[(ws.y      ) & 0xFu] * s * x2.x;
        acc += E2M1_LUT[(ws.y >>  4) & 0xFu] * s * x2.y;
        acc += E2M1_LUT[(ws.y >>  8) & 0xFu] * s * x2.z;
        acc += E2M1_LUT[(ws.y >> 12) & 0xFu] * s * x2.w;
        acc += E2M1_LUT[(ws.y >> 16) & 0xFu] * s * x3.x;
        acc += E2M1_LUT[(ws.y >> 20) & 0xFu] * s * x3.y;
        acc += E2M1_LUT[(ws.y >> 24) & 0xFu] * s * x3.z;
        acc += E2M1_LUT[(ws.y >> 28) & 0xFu] * s * x3.w;

        acc += E2M1_LUT[(ws.z      ) & 0xFu] * s * x4.x;
        acc += E2M1_LUT[(ws.z >>  4) & 0xFu] * s * x4.y;
        acc += E2M1_LUT[(ws.z >>  8) & 0xFu] * s * x4.z;
        acc += E2M1_LUT[(ws.z >> 12) & 0xFu] * s * x4.w;
        acc += E2M1_LUT[(ws.z >> 16) & 0xFu] * s * x5.x;
        acc += E2M1_LUT[(ws.z >> 20) & 0xFu] * s * x5.y;
        acc += E2M1_LUT[(ws.z >> 24) & 0xFu] * s * x5.z;
        acc += E2M1_LUT[(ws.z >> 28) & 0xFu] * s * x5.w;

        acc += E2M1_LUT[(ws.w      ) & 0xFu] * s * x6.x;
        acc += E2M1_LUT[(ws.w >>  4) & 0xFu] * s * x6.y;
        acc += E2M1_LUT[(ws.w >>  8) & 0xFu] * s * x6.z;
        acc += E2M1_LUT[(ws.w >> 12) & 0xFu] * s * x6.w;
        acc += E2M1_LUT[(ws.w >> 16) & 0xFu] * s * x7.x;
        acc += E2M1_LUT[(ws.w >> 20) & 0xFu] * s * x7.y;
        acc += E2M1_LUT[(ws.w >> 24) & 0xFu] * s * x7.z;
        acc += E2M1_LUT[(ws.w >> 28) & 0xFu] * s * x7.w;
    }
    y[b * dims.out_features + row] = acc;
}

// ───────────────────────────────────────────────────────────────────────────
// V3 kernels (Path B Phase B.1, 2026-04-26): simdgroup-cooperative reduction
// + threadgroup-shared activation cache.
//
// **What changed vs v2**
//   v2: 1 thread = 1 output row. Adjacent simdgroup lanes (e.g. lanes 0/1/...)
//   read consecutive ROWS of `packed`, jumping `words_per_row * 4` bytes apart.
//   For Qwen3.5 in_features=2048 that's 1 KB per lane → fully uncoalesced.
//   The microbench `bench_mxfp4_kernel` shows this caps shared gate_up
//   [1024×2048] at 9% of M3 Max peak bandwidth (36 GB/s of 400).
//
//   v3: 1 simdgroup (32 lanes) cooperates on ONE output row, partitioning the
//   `groups` axis across lanes. Adjacent lanes now read consecutive 16-byte
//   `uint4` chunks of the SAME row → fully coalesced. A `simd_sum` reduces
//   the 32 partial dot products at the end. 256 threads/threadgroup =
//   8 simdgroups → 8 rows per threadgroup. All 256 threads cooperatively
//   stage `x` into threadgroup memory once; subsequent loads are L1-resident.
//
// **What did NOT change**
//   The per-term arithmetic order `acc += LUT[nib] * s * x[i]` is preserved
//   (32 FMAs per group), so v3 is parity-equivalent with v1/v2 modulo
//   non-deterministic FMA reorderings the compiler may apply across lanes.
//
// **Constraints**
//   - `in_features * 4` bytes must fit in threadgroup memory (32 KB on
//     Apple M3 Max). For Qwen3.5 dims, max in_features is 8192 (self_attn o)
//     → 32 KB exactly. Fits.
//   - Threadgroup size hard-coded to 256 = 8 simdgroups × 32 lanes.
//   - `out_features` need not be a multiple of 8 — out-of-range simdgroups
//     return after the staging barrier without writing.
//
// **Routing**
//   `LUMEN_MXFP4_KERNEL_VERSION=v3` opts into v3 dispatch. v2 stays as the
//   default until v3 has end-to-end production parity verified across all
//   shapes the model actually exercises.
// ───────────────────────────────────────────────────────────────────────────

kernel void mxfp4_matmul_f32_v3(
    device const uint*    packed   [[buffer(0)]],
    device const uchar*   scales   [[buffer(1)]],
    device const float*   x        [[buffer(2)]],
    device float*         y        [[buffer(3)]],
    constant MxFp4Dims&   dims     [[buffer(4)]],
    constant uint&        batch    [[buffer(5)]],
    threadgroup float*    x_shared [[threadgroup(0)]],
    uint3 tg_pos          [[threadgroup_position_in_grid]],
    uint  tid_in_tg       [[thread_index_in_threadgroup]],
    uint  sg_id           [[simdgroup_index_in_threadgroup]],
    uint  sg_lane         [[thread_index_in_simdgroup]]
) {
    uint b = tg_pos.y;
    if (b >= batch) { return; }

    const uint ROWS_PER_TG = 8u;
    const uint THREADS_PER_TG = 256u;
    uint row = tg_pos.x * ROWS_PER_TG + sg_id;

    uint groups         = dims.in_features / 32u;
    uint words_per_row  = dims.in_features / 8u;
    uint x_row_base     = b * dims.in_features;

    // Cooperative stage of x[b, :] into threadgroup memory. All 256 threads
    // participate. Subsequent activation reads come from threadgroup memory
    // (effectively L1) instead of global, removing the per-group device
    // memory contention v2 paid.
    for (uint i = tid_in_tg; i < dims.in_features; i += THREADS_PER_TG) {
        x_shared[i] = x[x_row_base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Out-of-range simdgroups still helped stage above; bail out before doing
    // the dot product / write.
    if (row >= dims.out_features) { return; }

    uint word_row_base  = row * words_per_row;
    uint scale_row_base = row * groups;

    float acc = 0.0f;
    for (uint g = sg_lane; g < groups; g += 32u) {
        uchar scale_byte = scales[scale_row_base + g];
        if (scale_byte == 0xFFu) continue;
        uint sbits = uint(scale_byte) << 23;
        float s = as_type<float>(sbits);
        if (s == 0.0f) continue;

        uint word_base = word_row_base + g * 4u;
        uint x_base    = g * 32u;

        // Coalesced uint4 load: 32 lanes simultaneously hit 32 consecutive
        // group slots of the SAME row → adjacent 16-byte chunks.
        uint4 ws = *((device const uint4*)(packed + word_base));
        threadgroup const float4* x4 =
            (threadgroup const float4*)(x_shared + x_base);
        float4 x0 = x4[0]; float4 x1 = x4[1];
        float4 x2 = x4[2]; float4 x3 = x4[3];
        float4 x4_ = x4[4]; float4 x5 = x4[5];
        float4 x6 = x4[6]; float4 x7 = x4[7];

        acc += E2M1_LUT[(ws.x      ) & 0xFu] * s * x0.x;
        acc += E2M1_LUT[(ws.x >>  4) & 0xFu] * s * x0.y;
        acc += E2M1_LUT[(ws.x >>  8) & 0xFu] * s * x0.z;
        acc += E2M1_LUT[(ws.x >> 12) & 0xFu] * s * x0.w;
        acc += E2M1_LUT[(ws.x >> 16) & 0xFu] * s * x1.x;
        acc += E2M1_LUT[(ws.x >> 20) & 0xFu] * s * x1.y;
        acc += E2M1_LUT[(ws.x >> 24) & 0xFu] * s * x1.z;
        acc += E2M1_LUT[(ws.x >> 28) & 0xFu] * s * x1.w;

        acc += E2M1_LUT[(ws.y      ) & 0xFu] * s * x2.x;
        acc += E2M1_LUT[(ws.y >>  4) & 0xFu] * s * x2.y;
        acc += E2M1_LUT[(ws.y >>  8) & 0xFu] * s * x2.z;
        acc += E2M1_LUT[(ws.y >> 12) & 0xFu] * s * x2.w;
        acc += E2M1_LUT[(ws.y >> 16) & 0xFu] * s * x3.x;
        acc += E2M1_LUT[(ws.y >> 20) & 0xFu] * s * x3.y;
        acc += E2M1_LUT[(ws.y >> 24) & 0xFu] * s * x3.z;
        acc += E2M1_LUT[(ws.y >> 28) & 0xFu] * s * x3.w;

        acc += E2M1_LUT[(ws.z      ) & 0xFu] * s * x4_.x;
        acc += E2M1_LUT[(ws.z >>  4) & 0xFu] * s * x4_.y;
        acc += E2M1_LUT[(ws.z >>  8) & 0xFu] * s * x4_.z;
        acc += E2M1_LUT[(ws.z >> 12) & 0xFu] * s * x4_.w;
        acc += E2M1_LUT[(ws.z >> 16) & 0xFu] * s * x5.x;
        acc += E2M1_LUT[(ws.z >> 20) & 0xFu] * s * x5.y;
        acc += E2M1_LUT[(ws.z >> 24) & 0xFu] * s * x5.z;
        acc += E2M1_LUT[(ws.z >> 28) & 0xFu] * s * x5.w;

        acc += E2M1_LUT[(ws.w      ) & 0xFu] * s * x6.x;
        acc += E2M1_LUT[(ws.w >>  4) & 0xFu] * s * x6.y;
        acc += E2M1_LUT[(ws.w >>  8) & 0xFu] * s * x6.z;
        acc += E2M1_LUT[(ws.w >> 12) & 0xFu] * s * x6.w;
        acc += E2M1_LUT[(ws.w >> 16) & 0xFu] * s * x7.x;
        acc += E2M1_LUT[(ws.w >> 20) & 0xFu] * s * x7.y;
        acc += E2M1_LUT[(ws.w >> 24) & 0xFu] * s * x7.z;
        acc += E2M1_LUT[(ws.w >> 28) & 0xFu] * s * x7.w;
    }

    // Reduce 32 lanes' partial dot products → single value.
    acc = simd_sum(acc);

    // Lane 0 of each simdgroup writes the row's output. Other lanes' `acc`
    // is also the reduced value but we only need a single store.
    if (sg_lane == 0u) {
        y[b * dims.out_features + row] = acc;
    }
}

// ───────────────────────────────────────────────────────────────────────────
// `mxfp4_matmul_f32_v3_residual` — Lever L1 (residual fusion).
//
// Identical to `mxfp4_matmul_f32_v3` except the lane-0 store reads a residual
// element from buffer(6) and writes `acc + residual[idx]`. The residual must
// have the same `[batch, out_features]` layout as `y`. This collapses the
// downstream Tensor `+` add into the matmul kernel's tail, saving one
// element-wise dispatch per call site (≈ 50 µs CPU encoding / dispatch).
//
// Use cases:
//   - self_attn `o_proj`: residual = pre-attn input (so caller drops layer's
//     `(x + r)?` after the call).
//   - linear_attn `out_proj`: residual = pre-attn input (same idea).
//
// Numerical contract:
//   `acc + residual[idx]` is a single f32 add applied per output element. No
//   reduction/order ambiguity vs. the standalone `(x + r)` add — bit-identical
//   provided FMA reordering inside `acc` is preserved (it is — body unchanged).
// ───────────────────────────────────────────────────────────────────────────

kernel void mxfp4_matmul_f32_v3_residual(
    device const uint*    packed   [[buffer(0)]],
    device const uchar*   scales   [[buffer(1)]],
    device const float*   x        [[buffer(2)]],
    device float*         y        [[buffer(3)]],
    constant MxFp4Dims&   dims     [[buffer(4)]],
    constant uint&        batch    [[buffer(5)]],
    device const float*   residual [[buffer(6)]],
    threadgroup float*    x_shared [[threadgroup(0)]],
    uint3 tg_pos          [[threadgroup_position_in_grid]],
    uint  tid_in_tg       [[thread_index_in_threadgroup]],
    uint  sg_id           [[simdgroup_index_in_threadgroup]],
    uint  sg_lane         [[thread_index_in_simdgroup]]
) {
    uint b = tg_pos.y;
    if (b >= batch) { return; }

    const uint ROWS_PER_TG = 8u;
    const uint THREADS_PER_TG = 256u;
    uint row = tg_pos.x * ROWS_PER_TG + sg_id;

    uint groups         = dims.in_features / 32u;
    uint words_per_row  = dims.in_features / 8u;
    uint x_row_base     = b * dims.in_features;

    for (uint i = tid_in_tg; i < dims.in_features; i += THREADS_PER_TG) {
        x_shared[i] = x[x_row_base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= dims.out_features) { return; }

    uint word_row_base  = row * words_per_row;
    uint scale_row_base = row * groups;

    float acc = 0.0f;
    for (uint g = sg_lane; g < groups; g += 32u) {
        uchar scale_byte = scales[scale_row_base + g];
        if (scale_byte == 0xFFu) continue;
        uint sbits = uint(scale_byte) << 23;
        float s = as_type<float>(sbits);
        if (s == 0.0f) continue;

        uint word_base = word_row_base + g * 4u;
        uint x_base    = g * 32u;

        uint4 ws = *((device const uint4*)(packed + word_base));
        threadgroup const float4* x4 =
            (threadgroup const float4*)(x_shared + x_base);
        float4 x0 = x4[0]; float4 x1 = x4[1];
        float4 x2 = x4[2]; float4 x3 = x4[3];
        float4 x4_ = x4[4]; float4 x5 = x4[5];
        float4 x6 = x4[6]; float4 x7 = x4[7];

        acc += E2M1_LUT[(ws.x      ) & 0xFu] * s * x0.x;
        acc += E2M1_LUT[(ws.x >>  4) & 0xFu] * s * x0.y;
        acc += E2M1_LUT[(ws.x >>  8) & 0xFu] * s * x0.z;
        acc += E2M1_LUT[(ws.x >> 12) & 0xFu] * s * x0.w;
        acc += E2M1_LUT[(ws.x >> 16) & 0xFu] * s * x1.x;
        acc += E2M1_LUT[(ws.x >> 20) & 0xFu] * s * x1.y;
        acc += E2M1_LUT[(ws.x >> 24) & 0xFu] * s * x1.z;
        acc += E2M1_LUT[(ws.x >> 28) & 0xFu] * s * x1.w;

        acc += E2M1_LUT[(ws.y      ) & 0xFu] * s * x2.x;
        acc += E2M1_LUT[(ws.y >>  4) & 0xFu] * s * x2.y;
        acc += E2M1_LUT[(ws.y >>  8) & 0xFu] * s * x2.z;
        acc += E2M1_LUT[(ws.y >> 12) & 0xFu] * s * x2.w;
        acc += E2M1_LUT[(ws.y >> 16) & 0xFu] * s * x3.x;
        acc += E2M1_LUT[(ws.y >> 20) & 0xFu] * s * x3.y;
        acc += E2M1_LUT[(ws.y >> 24) & 0xFu] * s * x3.z;
        acc += E2M1_LUT[(ws.y >> 28) & 0xFu] * s * x3.w;

        acc += E2M1_LUT[(ws.z      ) & 0xFu] * s * x4_.x;
        acc += E2M1_LUT[(ws.z >>  4) & 0xFu] * s * x4_.y;
        acc += E2M1_LUT[(ws.z >>  8) & 0xFu] * s * x4_.z;
        acc += E2M1_LUT[(ws.z >> 12) & 0xFu] * s * x4_.w;
        acc += E2M1_LUT[(ws.z >> 16) & 0xFu] * s * x5.x;
        acc += E2M1_LUT[(ws.z >> 20) & 0xFu] * s * x5.y;
        acc += E2M1_LUT[(ws.z >> 24) & 0xFu] * s * x5.z;
        acc += E2M1_LUT[(ws.z >> 28) & 0xFu] * s * x5.w;

        acc += E2M1_LUT[(ws.w      ) & 0xFu] * s * x6.x;
        acc += E2M1_LUT[(ws.w >>  4) & 0xFu] * s * x6.y;
        acc += E2M1_LUT[(ws.w >>  8) & 0xFu] * s * x6.z;
        acc += E2M1_LUT[(ws.w >> 12) & 0xFu] * s * x6.w;
        acc += E2M1_LUT[(ws.w >> 16) & 0xFu] * s * x7.x;
        acc += E2M1_LUT[(ws.w >> 20) & 0xFu] * s * x7.y;
        acc += E2M1_LUT[(ws.w >> 24) & 0xFu] * s * x7.z;
        acc += E2M1_LUT[(ws.w >> 28) & 0xFu] * s * x7.w;
    }

    acc = simd_sum(acc);

    if (sg_lane == 0u) {
        uint out_idx = b * dims.out_features + row;
        y[out_idx] = acc + residual[out_idx];
    }
}

// ───────────────────────────────────────────────────────────────────────────
// `mxfp4_matmul_f32in_bf16out_v3`
//
// Phase A.0 (2026-04-27): identical math to `mxfp4_matmul_f32_v3` but writes
// the output as `bfloat` (16-bit) instead of `float` (32-bit). Accumulation
// stays in f32 inside the simdgroup — only the final store is narrowed.
//
// Why a sister kernel instead of a runtime dtype switch:
//   MSL is statically typed; `device float* y` and `device bfloat* y` are
//   distinct ABIs. Splitting into two kernels keeps each pipeline dispatch
//   monomorphic and avoids any conditional cost in the inner loop.
//
// Caller responsibility:
//   - `y` buffer must be sized for `bfloat` (2 bytes/elem) — half the f32
//     allocation.
//   - Activation `x` is still f32 (we only narrow the OUTPUT here). A
//     subsequent pass can add a `bf16in_bf16out` variant if the upstream
//     producer also emits bf16 (Phase A.1).
//
// Numerical contract:
//   bfloat16 = sign + 8-bit exponent + 7-bit mantissa. Round-to-nearest-even
//   on store. Activations in transformer FFN/attn live well within bf16
//   range (no overflow risk); the rounding error per element is ≤ 2^-7
//   relative, which RMSNorm + softmax absorb without measurable degradation
//   in MLX (~5e-3 cosine drift per block, validated downstream).
// ───────────────────────────────────────────────────────────────────────────

kernel void mxfp4_matmul_f32in_bf16out_v3(
    device const uint*    packed   [[buffer(0)]],
    device const uchar*   scales   [[buffer(1)]],
    device const float*   x        [[buffer(2)]],
    device bfloat*        y        [[buffer(3)]],
    constant MxFp4Dims&   dims     [[buffer(4)]],
    constant uint&        batch    [[buffer(5)]],
    threadgroup float*    x_shared [[threadgroup(0)]],
    uint3 tg_pos          [[threadgroup_position_in_grid]],
    uint  tid_in_tg       [[thread_index_in_threadgroup]],
    uint  sg_id           [[simdgroup_index_in_threadgroup]],
    uint  sg_lane         [[thread_index_in_simdgroup]]
) {
    uint b = tg_pos.y;
    if (b >= batch) { return; }

    const uint ROWS_PER_TG = 8u;
    const uint THREADS_PER_TG = 256u;
    uint row = tg_pos.x * ROWS_PER_TG + sg_id;

    uint groups         = dims.in_features / 32u;
    uint words_per_row  = dims.in_features / 8u;
    uint x_row_base     = b * dims.in_features;

    for (uint i = tid_in_tg; i < dims.in_features; i += THREADS_PER_TG) {
        x_shared[i] = x[x_row_base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= dims.out_features) { return; }

    uint word_row_base  = row * words_per_row;
    uint scale_row_base = row * groups;

    float acc = 0.0f;
    for (uint g = sg_lane; g < groups; g += 32u) {
        uchar scale_byte = scales[scale_row_base + g];
        if (scale_byte == 0xFFu) continue;
        uint sbits = uint(scale_byte) << 23;
        float s = as_type<float>(sbits);
        if (s == 0.0f) continue;

        uint word_base = word_row_base + g * 4u;
        uint x_base    = g * 32u;

        uint4 ws = *((device const uint4*)(packed + word_base));
        threadgroup const float4* x4 =
            (threadgroup const float4*)(x_shared + x_base);
        float4 x0 = x4[0]; float4 x1 = x4[1];
        float4 x2 = x4[2]; float4 x3 = x4[3];
        float4 x4_ = x4[4]; float4 x5 = x4[5];
        float4 x6 = x4[6]; float4 x7 = x4[7];

        acc += E2M1_LUT[(ws.x      ) & 0xFu] * s * x0.x;
        acc += E2M1_LUT[(ws.x >>  4) & 0xFu] * s * x0.y;
        acc += E2M1_LUT[(ws.x >>  8) & 0xFu] * s * x0.z;
        acc += E2M1_LUT[(ws.x >> 12) & 0xFu] * s * x0.w;
        acc += E2M1_LUT[(ws.x >> 16) & 0xFu] * s * x1.x;
        acc += E2M1_LUT[(ws.x >> 20) & 0xFu] * s * x1.y;
        acc += E2M1_LUT[(ws.x >> 24) & 0xFu] * s * x1.z;
        acc += E2M1_LUT[(ws.x >> 28) & 0xFu] * s * x1.w;

        acc += E2M1_LUT[(ws.y      ) & 0xFu] * s * x2.x;
        acc += E2M1_LUT[(ws.y >>  4) & 0xFu] * s * x2.y;
        acc += E2M1_LUT[(ws.y >>  8) & 0xFu] * s * x2.z;
        acc += E2M1_LUT[(ws.y >> 12) & 0xFu] * s * x2.w;
        acc += E2M1_LUT[(ws.y >> 16) & 0xFu] * s * x3.x;
        acc += E2M1_LUT[(ws.y >> 20) & 0xFu] * s * x3.y;
        acc += E2M1_LUT[(ws.y >> 24) & 0xFu] * s * x3.z;
        acc += E2M1_LUT[(ws.y >> 28) & 0xFu] * s * x3.w;

        acc += E2M1_LUT[(ws.z      ) & 0xFu] * s * x4_.x;
        acc += E2M1_LUT[(ws.z >>  4) & 0xFu] * s * x4_.y;
        acc += E2M1_LUT[(ws.z >>  8) & 0xFu] * s * x4_.z;
        acc += E2M1_LUT[(ws.z >> 12) & 0xFu] * s * x4_.w;
        acc += E2M1_LUT[(ws.z >> 16) & 0xFu] * s * x5.x;
        acc += E2M1_LUT[(ws.z >> 20) & 0xFu] * s * x5.y;
        acc += E2M1_LUT[(ws.z >> 24) & 0xFu] * s * x5.z;
        acc += E2M1_LUT[(ws.z >> 28) & 0xFu] * s * x5.w;

        acc += E2M1_LUT[(ws.w      ) & 0xFu] * s * x6.x;
        acc += E2M1_LUT[(ws.w >>  4) & 0xFu] * s * x6.y;
        acc += E2M1_LUT[(ws.w >>  8) & 0xFu] * s * x6.z;
        acc += E2M1_LUT[(ws.w >> 12) & 0xFu] * s * x6.w;
        acc += E2M1_LUT[(ws.w >> 16) & 0xFu] * s * x7.x;
        acc += E2M1_LUT[(ws.w >> 20) & 0xFu] * s * x7.y;
        acc += E2M1_LUT[(ws.w >> 24) & 0xFu] * s * x7.z;
        acc += E2M1_LUT[(ws.w >> 28) & 0xFu] * s * x7.w;
    }

    acc = simd_sum(acc);

    if (sg_lane == 0u) {
        // Round-to-nearest-even narrow happens in the bfloat() cast.
        y[b * dims.out_features + row] = bfloat(acc);
    }
}

// ───────────────────────────────────────────────────────────────────────────
// `mxfp4_matmul_bf16in_f32out_v3`
//
// Lever B L.2: identical math + dispatch topology to `mxfp4_matmul_f32_v3`
// but the activation `x` is `bfloat` (16-bit). Each thread converts to f32
// once during the cooperative threadgroup-memory staging step; the inner
// loop and accumulator stay in f32, matching f32_v3 byte-for-byte from the
// `threadgroup_barrier` onward. Output stays f32.
//
// Why this kernel exists:
//   The MoE counterpart (`mxfp4_matmul_moe_bf16in_f32out_v3`) carries an
//   `expert_indices` indirection that does not apply to dense projections
//   (qkv_proj / o_proj / lm_head). A non-MoE bf16-input variant is needed
//   so the upstream RmsNorm can emit bf16 (Lever B L.1) and the qkv matmul
//   reads it directly without a cast-back to f32 in the activation buffer.
//
// Numerical contract:
//   bf16 → f32 widening is exact (no rounding). Subsequent FMA / simd_sum
//   are bit-identical to the f32_v3 kernel for the same f32-equivalent
//   activation. Only difference is the bf16 → f32 mantissa truncation
//   already absorbed by the upstream RmsNorm output (≤ 6e-3 abs, validated
//   by L.1 acceptance tests).
// ───────────────────────────────────────────────────────────────────────────

kernel void mxfp4_matmul_bf16in_f32out_v3(
    device const uint*    packed   [[buffer(0)]],
    device const uchar*   scales   [[buffer(1)]],
    device const bfloat*  x        [[buffer(2)]],
    device float*         y        [[buffer(3)]],
    constant MxFp4Dims&   dims     [[buffer(4)]],
    constant uint&        batch    [[buffer(5)]],
    threadgroup float*    x_shared [[threadgroup(0)]],
    uint3 tg_pos          [[threadgroup_position_in_grid]],
    uint  tid_in_tg       [[thread_index_in_threadgroup]],
    uint  sg_id           [[simdgroup_index_in_threadgroup]],
    uint  sg_lane         [[thread_index_in_simdgroup]]
) {
    uint b = tg_pos.y;
    if (b >= batch) { return; }

    const uint ROWS_PER_TG = 8u;
    const uint THREADS_PER_TG = 256u;
    uint row = tg_pos.x * ROWS_PER_TG + sg_id;

    uint groups         = dims.in_features / 32u;
    uint words_per_row  = dims.in_features / 8u;
    uint x_row_base     = b * dims.in_features;

    // Convert bfloat → float once during staging; downstream reads from
    // x_shared as float (matches the f32-in v3 inner-loop layout exactly).
    for (uint i = tid_in_tg; i < dims.in_features; i += THREADS_PER_TG) {
        x_shared[i] = float(x[x_row_base + i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= dims.out_features) { return; }

    uint word_row_base  = row * words_per_row;
    uint scale_row_base = row * groups;

    float acc = 0.0f;
    for (uint g = sg_lane; g < groups; g += 32u) {
        uchar scale_byte = scales[scale_row_base + g];
        if (scale_byte == 0xFFu) continue;
        uint sbits = uint(scale_byte) << 23;
        float s = as_type<float>(sbits);
        if (s == 0.0f) continue;

        uint word_base = word_row_base + g * 4u;
        uint x_base    = g * 32u;

        uint4 ws = *((device const uint4*)(packed + word_base));
        threadgroup const float4* x4 =
            (threadgroup const float4*)(x_shared + x_base);
        float4 x0 = x4[0]; float4 x1 = x4[1];
        float4 x2 = x4[2]; float4 x3 = x4[3];
        float4 x4_ = x4[4]; float4 x5 = x4[5];
        float4 x6 = x4[6]; float4 x7 = x4[7];

        acc += E2M1_LUT[(ws.x      ) & 0xFu] * s * x0.x;
        acc += E2M1_LUT[(ws.x >>  4) & 0xFu] * s * x0.y;
        acc += E2M1_LUT[(ws.x >>  8) & 0xFu] * s * x0.z;
        acc += E2M1_LUT[(ws.x >> 12) & 0xFu] * s * x0.w;
        acc += E2M1_LUT[(ws.x >> 16) & 0xFu] * s * x1.x;
        acc += E2M1_LUT[(ws.x >> 20) & 0xFu] * s * x1.y;
        acc += E2M1_LUT[(ws.x >> 24) & 0xFu] * s * x1.z;
        acc += E2M1_LUT[(ws.x >> 28) & 0xFu] * s * x1.w;

        acc += E2M1_LUT[(ws.y      ) & 0xFu] * s * x2.x;
        acc += E2M1_LUT[(ws.y >>  4) & 0xFu] * s * x2.y;
        acc += E2M1_LUT[(ws.y >>  8) & 0xFu] * s * x2.z;
        acc += E2M1_LUT[(ws.y >> 12) & 0xFu] * s * x2.w;
        acc += E2M1_LUT[(ws.y >> 16) & 0xFu] * s * x3.x;
        acc += E2M1_LUT[(ws.y >> 20) & 0xFu] * s * x3.y;
        acc += E2M1_LUT[(ws.y >> 24) & 0xFu] * s * x3.z;
        acc += E2M1_LUT[(ws.y >> 28) & 0xFu] * s * x3.w;

        acc += E2M1_LUT[(ws.z      ) & 0xFu] * s * x4_.x;
        acc += E2M1_LUT[(ws.z >>  4) & 0xFu] * s * x4_.y;
        acc += E2M1_LUT[(ws.z >>  8) & 0xFu] * s * x4_.z;
        acc += E2M1_LUT[(ws.z >> 12) & 0xFu] * s * x4_.w;
        acc += E2M1_LUT[(ws.z >> 16) & 0xFu] * s * x5.x;
        acc += E2M1_LUT[(ws.z >> 20) & 0xFu] * s * x5.y;
        acc += E2M1_LUT[(ws.z >> 24) & 0xFu] * s * x5.z;
        acc += E2M1_LUT[(ws.z >> 28) & 0xFu] * s * x5.w;

        acc += E2M1_LUT[(ws.w      ) & 0xFu] * s * x6.x;
        acc += E2M1_LUT[(ws.w >>  4) & 0xFu] * s * x6.y;
        acc += E2M1_LUT[(ws.w >>  8) & 0xFu] * s * x6.z;
        acc += E2M1_LUT[(ws.w >> 12) & 0xFu] * s * x6.w;
        acc += E2M1_LUT[(ws.w >> 16) & 0xFu] * s * x7.x;
        acc += E2M1_LUT[(ws.w >> 20) & 0xFu] * s * x7.y;
        acc += E2M1_LUT[(ws.w >> 24) & 0xFu] * s * x7.z;
        acc += E2M1_LUT[(ws.w >> 28) & 0xFu] * s * x7.w;
    }

    acc = simd_sum(acc);

    if (sg_lane == 0u) {
        y[b * dims.out_features + row] = acc;
    }
}

kernel void mxfp4_matmul_moe_f32_v2(
    device const uint*     packed_all     [[buffer(0)]],
    device const uchar*    scales_all     [[buffer(1)]],
    device const uint*     expert_indices [[buffer(2)]],
    device const float*    x              [[buffer(3)]],
    device float*          y              [[buffer(4)]],
    constant MxFp4MoeDims& dims           [[buffer(5)]],
    uint3 gid                             [[thread_position_in_grid]]
) {
    uint row  = gid.x;
    uint b    = gid.y;
    uint slot = gid.z;
    if (row >= dims.out_features || b >= dims.batch) { return; }

    uint e = expert_indices[slot];

    uint groups         = dims.in_features / 32u;
    uint words_per_row  = dims.in_features / 8u;
    uint packed_expert_stride = dims.out_features * words_per_row;
    uint scale_expert_stride  = dims.out_features * groups;

    uint word_row_base  = e * packed_expert_stride + row * words_per_row;
    uint scale_row_base = e * scale_expert_stride  + row * groups;

    uint x_slot = (dims.broadcast_x != 0u) ? 0u : slot;
    uint x_row_base = x_slot * dims.batch * dims.in_features + b * dims.in_features;

    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        float s = e8m0_scale_device(scales_all[scale_row_base + g]);
        if (s == 0.0f) { continue; }
        uint word_base = word_row_base + g * 4u;
        uint x_base    = x_row_base + g * 32u;

        uint4 ws = *((device const uint4*)(packed_all + word_base));
        device const float4* x4_ptr = (device const float4*)(x + x_base);
        float4 x0 = x4_ptr[0]; float4 x1 = x4_ptr[1];
        float4 x2 = x4_ptr[2]; float4 x3 = x4_ptr[3];
        float4 x4 = x4_ptr[4]; float4 x5 = x4_ptr[5];
        float4 x6 = x4_ptr[6]; float4 x7 = x4_ptr[7];

        acc += E2M1_LUT[(ws.x      ) & 0xFu] * s * x0.x;
        acc += E2M1_LUT[(ws.x >>  4) & 0xFu] * s * x0.y;
        acc += E2M1_LUT[(ws.x >>  8) & 0xFu] * s * x0.z;
        acc += E2M1_LUT[(ws.x >> 12) & 0xFu] * s * x0.w;
        acc += E2M1_LUT[(ws.x >> 16) & 0xFu] * s * x1.x;
        acc += E2M1_LUT[(ws.x >> 20) & 0xFu] * s * x1.y;
        acc += E2M1_LUT[(ws.x >> 24) & 0xFu] * s * x1.z;
        acc += E2M1_LUT[(ws.x >> 28) & 0xFu] * s * x1.w;

        acc += E2M1_LUT[(ws.y      ) & 0xFu] * s * x2.x;
        acc += E2M1_LUT[(ws.y >>  4) & 0xFu] * s * x2.y;
        acc += E2M1_LUT[(ws.y >>  8) & 0xFu] * s * x2.z;
        acc += E2M1_LUT[(ws.y >> 12) & 0xFu] * s * x2.w;
        acc += E2M1_LUT[(ws.y >> 16) & 0xFu] * s * x3.x;
        acc += E2M1_LUT[(ws.y >> 20) & 0xFu] * s * x3.y;
        acc += E2M1_LUT[(ws.y >> 24) & 0xFu] * s * x3.z;
        acc += E2M1_LUT[(ws.y >> 28) & 0xFu] * s * x3.w;

        acc += E2M1_LUT[(ws.z      ) & 0xFu] * s * x4.x;
        acc += E2M1_LUT[(ws.z >>  4) & 0xFu] * s * x4.y;
        acc += E2M1_LUT[(ws.z >>  8) & 0xFu] * s * x4.z;
        acc += E2M1_LUT[(ws.z >> 12) & 0xFu] * s * x4.w;
        acc += E2M1_LUT[(ws.z >> 16) & 0xFu] * s * x5.x;
        acc += E2M1_LUT[(ws.z >> 20) & 0xFu] * s * x5.y;
        acc += E2M1_LUT[(ws.z >> 24) & 0xFu] * s * x5.z;
        acc += E2M1_LUT[(ws.z >> 28) & 0xFu] * s * x5.w;

        acc += E2M1_LUT[(ws.w      ) & 0xFu] * s * x6.x;
        acc += E2M1_LUT[(ws.w >>  4) & 0xFu] * s * x6.y;
        acc += E2M1_LUT[(ws.w >>  8) & 0xFu] * s * x6.z;
        acc += E2M1_LUT[(ws.w >> 12) & 0xFu] * s * x6.w;
        acc += E2M1_LUT[(ws.w >> 16) & 0xFu] * s * x7.x;
        acc += E2M1_LUT[(ws.w >> 20) & 0xFu] * s * x7.y;
        acc += E2M1_LUT[(ws.w >> 24) & 0xFu] * s * x7.z;
        acc += E2M1_LUT[(ws.w >> 28) & 0xFu] * s * x7.w;
    }
    y[slot * dims.batch * dims.out_features + b * dims.out_features + row] = acc;
}

// ───────────────────────────────────────────────────────────────────────────
// v3 MoE — simdgroup cooperative + threadgroup x cache (B.2)
// ───────────────────────────────────────────────────────────────────────────
// Mirrors `mxfp4_matmul_f32_v3` but with the v2-MoE expert-indices indirection
// fused into per-thread offsets. The grid is (out_tg_x, batch, k_slot) where
// `out_tg_x = ceil(out_features / 8)` (each tg fans out 8 rows). Threadgroup
// memory caches `x[b, :]` (or `x[slot, b, :]` when broadcast_x=0) once and
// reuses it across all 8 rows × 32 lanes of the same simdgroup.
//
// Why a separate kernel from v3 non-MoE: same E2M1 inner loop, but the
// expert-indices indirection touches every row's `packed`/`scales` base
// pointer and we want the address arithmetic outside the hot loop without
// per-iteration branches.
kernel void mxfp4_matmul_moe_f32_v3(
    device const uint*     packed_all     [[buffer(0)]],
    device const uchar*    scales_all     [[buffer(1)]],
    device const uint*     expert_indices [[buffer(2)]],
    device const float*    x              [[buffer(3)]],
    device float*          y              [[buffer(4)]],
    constant MxFp4MoeDims& dims           [[buffer(5)]],
    threadgroup float*     x_shared       [[threadgroup(0)]],
    uint3 tg_pos            [[threadgroup_position_in_grid]],
    uint  tid_in_tg         [[thread_index_in_threadgroup]],
    uint  sg_id             [[simdgroup_index_in_threadgroup]],
    uint  sg_lane           [[thread_index_in_simdgroup]]
) {
    uint b    = tg_pos.y;
    uint slot = tg_pos.z;
    if (b >= dims.batch) { return; }

    const uint ROWS_PER_TG = 8u;
    const uint THREADS_PER_TG = 256u;
    uint row = tg_pos.x * ROWS_PER_TG + sg_id;

    uint groups        = dims.in_features / 32u;
    uint words_per_row = dims.in_features / 8u;

    // Stage `x[b, :]` (or `x[slot, b, :]`) into threadgroup memory. All 256
    // threads cooperate; rows out-of-range still help with this stage and
    // bail before the dot product.
    uint x_slot     = (dims.broadcast_x != 0u) ? 0u : slot;
    uint x_row_base = x_slot * dims.batch * dims.in_features
                    + b * dims.in_features;
    for (uint i = tid_in_tg; i < dims.in_features; i += THREADS_PER_TG) {
        x_shared[i] = x[x_row_base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= dims.out_features) { return; }

    uint e = expert_indices[slot];
    uint packed_expert_stride = dims.out_features * words_per_row;
    uint scale_expert_stride  = dims.out_features * groups;
    uint word_row_base  = e * packed_expert_stride + row * words_per_row;
    uint scale_row_base = e * scale_expert_stride  + row * groups;

    float acc = 0.0f;
    for (uint g = sg_lane; g < groups; g += 32u) {
        uchar scale_byte = scales_all[scale_row_base + g];
        if (scale_byte == 0xFFu) continue;
        uint sbits = uint(scale_byte) << 23;
        float s = as_type<float>(sbits);
        if (s == 0.0f) continue;

        uint word_base = word_row_base + g * 4u;
        uint x_base    = g * 32u;

        uint4 ws = *((device const uint4*)(packed_all + word_base));
        threadgroup const float4* x4 =
            (threadgroup const float4*)(x_shared + x_base);
        float4 x0 = x4[0]; float4 x1 = x4[1];
        float4 x2 = x4[2]; float4 x3 = x4[3];
        float4 x4_ = x4[4]; float4 x5 = x4[5];
        float4 x6 = x4[6]; float4 x7 = x4[7];

        acc += E2M1_LUT[(ws.x      ) & 0xFu] * s * x0.x;
        acc += E2M1_LUT[(ws.x >>  4) & 0xFu] * s * x0.y;
        acc += E2M1_LUT[(ws.x >>  8) & 0xFu] * s * x0.z;
        acc += E2M1_LUT[(ws.x >> 12) & 0xFu] * s * x0.w;
        acc += E2M1_LUT[(ws.x >> 16) & 0xFu] * s * x1.x;
        acc += E2M1_LUT[(ws.x >> 20) & 0xFu] * s * x1.y;
        acc += E2M1_LUT[(ws.x >> 24) & 0xFu] * s * x1.z;
        acc += E2M1_LUT[(ws.x >> 28) & 0xFu] * s * x1.w;

        acc += E2M1_LUT[(ws.y      ) & 0xFu] * s * x2.x;
        acc += E2M1_LUT[(ws.y >>  4) & 0xFu] * s * x2.y;
        acc += E2M1_LUT[(ws.y >>  8) & 0xFu] * s * x2.z;
        acc += E2M1_LUT[(ws.y >> 12) & 0xFu] * s * x2.w;
        acc += E2M1_LUT[(ws.y >> 16) & 0xFu] * s * x3.x;
        acc += E2M1_LUT[(ws.y >> 20) & 0xFu] * s * x3.y;
        acc += E2M1_LUT[(ws.y >> 24) & 0xFu] * s * x3.z;
        acc += E2M1_LUT[(ws.y >> 28) & 0xFu] * s * x3.w;

        acc += E2M1_LUT[(ws.z      ) & 0xFu] * s * x4_.x;
        acc += E2M1_LUT[(ws.z >>  4) & 0xFu] * s * x4_.y;
        acc += E2M1_LUT[(ws.z >>  8) & 0xFu] * s * x4_.z;
        acc += E2M1_LUT[(ws.z >> 12) & 0xFu] * s * x4_.w;
        acc += E2M1_LUT[(ws.z >> 16) & 0xFu] * s * x5.x;
        acc += E2M1_LUT[(ws.z >> 20) & 0xFu] * s * x5.y;
        acc += E2M1_LUT[(ws.z >> 24) & 0xFu] * s * x5.z;
        acc += E2M1_LUT[(ws.z >> 28) & 0xFu] * s * x5.w;

        acc += E2M1_LUT[(ws.w      ) & 0xFu] * s * x6.x;
        acc += E2M1_LUT[(ws.w >>  4) & 0xFu] * s * x6.y;
        acc += E2M1_LUT[(ws.w >>  8) & 0xFu] * s * x6.z;
        acc += E2M1_LUT[(ws.w >> 12) & 0xFu] * s * x6.w;
        acc += E2M1_LUT[(ws.w >> 16) & 0xFu] * s * x7.x;
        acc += E2M1_LUT[(ws.w >> 20) & 0xFu] * s * x7.y;
        acc += E2M1_LUT[(ws.w >> 24) & 0xFu] * s * x7.z;
        acc += E2M1_LUT[(ws.w >> 28) & 0xFu] * s * x7.w;
    }

    acc = simd_sum(acc);
    if (sg_lane == 0u) {
        y[slot * dims.batch * dims.out_features + b * dims.out_features + row] = acc;
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Lever D (2026-04-27) — mxfp4_matmul_moe_bf16in_f32out_v3: bf16-input
// sister of `mxfp4_matmul_moe_f32_v3`. Reads `x: bfloat[k, batch, in]`,
// converts to f32 once during TG-shared staging, then runs the same
// f32 inner FMA loop. Pairs with `mxfp4_moe_gate_up_silu_mul_f32in_bf16out_v3`
// to form the chain: gate_up writes bf16 → down reads bf16 (no cast back
// through f32 device memory). Output stays f32.
// ───────────────────────────────────────────────────────────────────────────
kernel void mxfp4_matmul_moe_bf16in_f32out_v3(
    device const uint*     packed_all     [[buffer(0)]],
    device const uchar*    scales_all     [[buffer(1)]],
    device const uint*     expert_indices [[buffer(2)]],
    device const bfloat*   x              [[buffer(3)]],
    device float*          y              [[buffer(4)]],
    constant MxFp4MoeDims& dims           [[buffer(5)]],
    threadgroup float*     x_shared       [[threadgroup(0)]],
    uint3 tg_pos            [[threadgroup_position_in_grid]],
    uint  tid_in_tg         [[thread_index_in_threadgroup]],
    uint  sg_id             [[simdgroup_index_in_threadgroup]],
    uint  sg_lane           [[thread_index_in_simdgroup]]
) {
    uint b    = tg_pos.y;
    uint slot = tg_pos.z;
    if (b >= dims.batch) { return; }

    const uint ROWS_PER_TG = 8u;
    const uint THREADS_PER_TG = 256u;
    uint row = tg_pos.x * ROWS_PER_TG + sg_id;

    uint groups        = dims.in_features / 32u;
    uint words_per_row = dims.in_features / 8u;

    uint x_slot     = (dims.broadcast_x != 0u) ? 0u : slot;
    uint x_row_base = x_slot * dims.batch * dims.in_features
                    + b * dims.in_features;
    // Convert bfloat → float once during staging; downstream reads from
    // x_shared as float (matches the f32-in v3 inner-loop layout exactly).
    for (uint i = tid_in_tg; i < dims.in_features; i += THREADS_PER_TG) {
        x_shared[i] = float(x[x_row_base + i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= dims.out_features) { return; }

    uint e = expert_indices[slot];
    uint packed_expert_stride = dims.out_features * words_per_row;
    uint scale_expert_stride  = dims.out_features * groups;
    uint word_row_base  = e * packed_expert_stride + row * words_per_row;
    uint scale_row_base = e * scale_expert_stride  + row * groups;

    float acc = 0.0f;
    for (uint g = sg_lane; g < groups; g += 32u) {
        uchar scale_byte = scales_all[scale_row_base + g];
        if (scale_byte == 0xFFu) continue;
        uint sbits = uint(scale_byte) << 23;
        float s = as_type<float>(sbits);
        if (s == 0.0f) continue;

        uint word_base = word_row_base + g * 4u;
        uint x_base    = g * 32u;

        uint4 ws = *((device const uint4*)(packed_all + word_base));
        threadgroup const float4* x4 =
            (threadgroup const float4*)(x_shared + x_base);
        float4 x0 = x4[0]; float4 x1 = x4[1];
        float4 x2 = x4[2]; float4 x3 = x4[3];
        float4 x4_ = x4[4]; float4 x5 = x4[5];
        float4 x6 = x4[6]; float4 x7 = x4[7];

        acc += E2M1_LUT[(ws.x      ) & 0xFu] * s * x0.x;
        acc += E2M1_LUT[(ws.x >>  4) & 0xFu] * s * x0.y;
        acc += E2M1_LUT[(ws.x >>  8) & 0xFu] * s * x0.z;
        acc += E2M1_LUT[(ws.x >> 12) & 0xFu] * s * x0.w;
        acc += E2M1_LUT[(ws.x >> 16) & 0xFu] * s * x1.x;
        acc += E2M1_LUT[(ws.x >> 20) & 0xFu] * s * x1.y;
        acc += E2M1_LUT[(ws.x >> 24) & 0xFu] * s * x1.z;
        acc += E2M1_LUT[(ws.x >> 28) & 0xFu] * s * x1.w;

        acc += E2M1_LUT[(ws.y      ) & 0xFu] * s * x2.x;
        acc += E2M1_LUT[(ws.y >>  4) & 0xFu] * s * x2.y;
        acc += E2M1_LUT[(ws.y >>  8) & 0xFu] * s * x2.z;
        acc += E2M1_LUT[(ws.y >> 12) & 0xFu] * s * x2.w;
        acc += E2M1_LUT[(ws.y >> 16) & 0xFu] * s * x3.x;
        acc += E2M1_LUT[(ws.y >> 20) & 0xFu] * s * x3.y;
        acc += E2M1_LUT[(ws.y >> 24) & 0xFu] * s * x3.z;
        acc += E2M1_LUT[(ws.y >> 28) & 0xFu] * s * x3.w;

        acc += E2M1_LUT[(ws.z      ) & 0xFu] * s * x4_.x;
        acc += E2M1_LUT[(ws.z >>  4) & 0xFu] * s * x4_.y;
        acc += E2M1_LUT[(ws.z >>  8) & 0xFu] * s * x4_.z;
        acc += E2M1_LUT[(ws.z >> 12) & 0xFu] * s * x4_.w;
        acc += E2M1_LUT[(ws.z >> 16) & 0xFu] * s * x5.x;
        acc += E2M1_LUT[(ws.z >> 20) & 0xFu] * s * x5.y;
        acc += E2M1_LUT[(ws.z >> 24) & 0xFu] * s * x5.z;
        acc += E2M1_LUT[(ws.z >> 28) & 0xFu] * s * x5.w;

        acc += E2M1_LUT[(ws.w      ) & 0xFu] * s * x6.x;
        acc += E2M1_LUT[(ws.w >>  4) & 0xFu] * s * x6.y;
        acc += E2M1_LUT[(ws.w >>  8) & 0xFu] * s * x6.z;
        acc += E2M1_LUT[(ws.w >> 12) & 0xFu] * s * x6.w;
        acc += E2M1_LUT[(ws.w >> 16) & 0xFu] * s * x7.x;
        acc += E2M1_LUT[(ws.w >> 20) & 0xFu] * s * x7.y;
        acc += E2M1_LUT[(ws.w >> 24) & 0xFu] * s * x7.z;
        acc += E2M1_LUT[(ws.w >> 28) & 0xFu] * s * x7.w;
    }

    acc = simd_sum(acc);
    if (sg_lane == 0u) {
        y[slot * dims.batch * dims.out_features + b * dims.out_features + row] = acc;
    }
}

// ───────────────────────────────────────────────────────────────────────────
// mxfp4_moe_gate_up_silu_mul_f32_v3 — Lever A (2026-04-27): routed grouped
// fused gate+up matmul + SwiGLU. Combines the expert-indices indirection of
// `mxfp4_matmul_moe_f32_v3` with the SiLU(gate)*up fusion of
// `mxfp4_gate_up_silu_mul_f32_v3`. Output is [k, batch, inter] (half of the
// non-fused variant's [k, batch, 2*inter]); silu(acc_gate)*acc_up is computed
// in registers and stored once.
//
// Each expert weight slab is laid out [2*inter, in/8] (gate rows [0..inter),
// up rows [inter..2*inter)) — same as the v3 grouped path. We fold the
// gate/up split into per-row offset arithmetic so the dispatch grid retains
// the v3 topology over `inter` rows (not `2*inter`), giving 1024 TGs/decode
// instead of 2048 — well above the M.1 NEGATIVE TG-occupancy threshold.
// ───────────────────────────────────────────────────────────────────────────
struct MxFp4MoeGateUpSiluMulDims {
    uint inter;        // moe_inter (output rows per expert per slot)
    uint in_features;  // hidden
    uint batch;        // typically 1 in decode
};

kernel void mxfp4_moe_gate_up_silu_mul_f32_v3(
    device const uint*     packed_all     [[buffer(0)]],
    device const uchar*    scales_all     [[buffer(1)]],
    device const uint*     expert_indices [[buffer(2)]],
    device const float*    x              [[buffer(3)]],
    device float*          y              [[buffer(4)]],
    constant MxFp4MoeGateUpSiluMulDims& dims [[buffer(5)]],
    threadgroup float*     x_shared       [[threadgroup(0)]],
    uint3 tg_pos            [[threadgroup_position_in_grid]],
    uint  tid_in_tg         [[thread_index_in_threadgroup]],
    uint  sg_id             [[simdgroup_index_in_threadgroup]],
    uint  sg_lane           [[thread_index_in_simdgroup]]
) {
    uint b    = tg_pos.y;
    uint slot = tg_pos.z;
    if (b >= dims.batch) { return; }

    const uint ROWS_PER_TG = 8u;
    const uint THREADS_PER_TG = 256u;
    uint row = tg_pos.x * ROWS_PER_TG + sg_id;

    uint groups        = dims.in_features / 32u;
    uint words_per_row = dims.in_features / 8u;

    // Stage x[b, :] into threadgroup memory. The routed grouped path always
    // broadcasts x over slots (caller passes broadcast_x=true), so the slot
    // axis is irrelevant for the load.
    uint x_row_base = b * dims.in_features;
    for (uint i = tid_in_tg; i < dims.in_features; i += THREADS_PER_TG) {
        x_shared[i] = x[x_row_base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= dims.inter) { return; }

    uint e = expert_indices[slot];
    // Each expert has 2*inter rows: gate at [0..inter), up at [inter..2*inter).
    uint packed_expert_stride = 2u * dims.inter * words_per_row;
    uint scale_expert_stride  = 2u * dims.inter * groups;

    uint gate_word_base  = e * packed_expert_stride + row * words_per_row;
    uint gate_scale_base = e * scale_expert_stride  + row * groups;
    uint up_word_base    = e * packed_expert_stride + (row + dims.inter) * words_per_row;
    uint up_scale_base   = e * scale_expert_stride  + (row + dims.inter) * groups;

    float acc_gate = 0.0f;
    float acc_up   = 0.0f;

    for (uint g = sg_lane; g < groups; g += 32u) {
        uchar sg_byte = scales_all[gate_scale_base + g];
        uchar su_byte = scales_all[up_scale_base + g];

        float s_gate = 0.0f;
        float s_up   = 0.0f;
        if (sg_byte != 0xFFu) {
            uint sb = uint(sg_byte) << 23;
            s_gate = as_type<float>(sb);
        }
        if (su_byte != 0xFFu) {
            uint sb = uint(su_byte) << 23;
            s_up = as_type<float>(sb);
        }
        if (s_gate == 0.0f && s_up == 0.0f) { continue; }

        uint x_base = g * 32u;
        threadgroup const float4* x4 =
            (threadgroup const float4*)(x_shared + x_base);
        float4 x0 = x4[0]; float4 x1 = x4[1];
        float4 x2 = x4[2]; float4 x3 = x4[3];
        float4 x4_ = x4[4]; float4 x5 = x4[5];
        float4 x6 = x4[6]; float4 x7 = x4[7];

        if (s_gate != 0.0f) {
            uint4 ws = *((device const uint4*)(packed_all + gate_word_base + g * 4u));
            acc_gate += E2M1_LUT[(ws.x      ) & 0xFu] * s_gate * x0.x;
            acc_gate += E2M1_LUT[(ws.x >>  4) & 0xFu] * s_gate * x0.y;
            acc_gate += E2M1_LUT[(ws.x >>  8) & 0xFu] * s_gate * x0.z;
            acc_gate += E2M1_LUT[(ws.x >> 12) & 0xFu] * s_gate * x0.w;
            acc_gate += E2M1_LUT[(ws.x >> 16) & 0xFu] * s_gate * x1.x;
            acc_gate += E2M1_LUT[(ws.x >> 20) & 0xFu] * s_gate * x1.y;
            acc_gate += E2M1_LUT[(ws.x >> 24) & 0xFu] * s_gate * x1.z;
            acc_gate += E2M1_LUT[(ws.x >> 28) & 0xFu] * s_gate * x1.w;

            acc_gate += E2M1_LUT[(ws.y      ) & 0xFu] * s_gate * x2.x;
            acc_gate += E2M1_LUT[(ws.y >>  4) & 0xFu] * s_gate * x2.y;
            acc_gate += E2M1_LUT[(ws.y >>  8) & 0xFu] * s_gate * x2.z;
            acc_gate += E2M1_LUT[(ws.y >> 12) & 0xFu] * s_gate * x2.w;
            acc_gate += E2M1_LUT[(ws.y >> 16) & 0xFu] * s_gate * x3.x;
            acc_gate += E2M1_LUT[(ws.y >> 20) & 0xFu] * s_gate * x3.y;
            acc_gate += E2M1_LUT[(ws.y >> 24) & 0xFu] * s_gate * x3.z;
            acc_gate += E2M1_LUT[(ws.y >> 28) & 0xFu] * s_gate * x3.w;

            acc_gate += E2M1_LUT[(ws.z      ) & 0xFu] * s_gate * x4_.x;
            acc_gate += E2M1_LUT[(ws.z >>  4) & 0xFu] * s_gate * x4_.y;
            acc_gate += E2M1_LUT[(ws.z >>  8) & 0xFu] * s_gate * x4_.z;
            acc_gate += E2M1_LUT[(ws.z >> 12) & 0xFu] * s_gate * x4_.w;
            acc_gate += E2M1_LUT[(ws.z >> 16) & 0xFu] * s_gate * x5.x;
            acc_gate += E2M1_LUT[(ws.z >> 20) & 0xFu] * s_gate * x5.y;
            acc_gate += E2M1_LUT[(ws.z >> 24) & 0xFu] * s_gate * x5.z;
            acc_gate += E2M1_LUT[(ws.z >> 28) & 0xFu] * s_gate * x5.w;

            acc_gate += E2M1_LUT[(ws.w      ) & 0xFu] * s_gate * x6.x;
            acc_gate += E2M1_LUT[(ws.w >>  4) & 0xFu] * s_gate * x6.y;
            acc_gate += E2M1_LUT[(ws.w >>  8) & 0xFu] * s_gate * x6.z;
            acc_gate += E2M1_LUT[(ws.w >> 12) & 0xFu] * s_gate * x6.w;
            acc_gate += E2M1_LUT[(ws.w >> 16) & 0xFu] * s_gate * x7.x;
            acc_gate += E2M1_LUT[(ws.w >> 20) & 0xFu] * s_gate * x7.y;
            acc_gate += E2M1_LUT[(ws.w >> 24) & 0xFu] * s_gate * x7.z;
            acc_gate += E2M1_LUT[(ws.w >> 28) & 0xFu] * s_gate * x7.w;
        }

        if (s_up != 0.0f) {
            uint4 ws = *((device const uint4*)(packed_all + up_word_base + g * 4u));
            acc_up += E2M1_LUT[(ws.x      ) & 0xFu] * s_up * x0.x;
            acc_up += E2M1_LUT[(ws.x >>  4) & 0xFu] * s_up * x0.y;
            acc_up += E2M1_LUT[(ws.x >>  8) & 0xFu] * s_up * x0.z;
            acc_up += E2M1_LUT[(ws.x >> 12) & 0xFu] * s_up * x0.w;
            acc_up += E2M1_LUT[(ws.x >> 16) & 0xFu] * s_up * x1.x;
            acc_up += E2M1_LUT[(ws.x >> 20) & 0xFu] * s_up * x1.y;
            acc_up += E2M1_LUT[(ws.x >> 24) & 0xFu] * s_up * x1.z;
            acc_up += E2M1_LUT[(ws.x >> 28) & 0xFu] * s_up * x1.w;

            acc_up += E2M1_LUT[(ws.y      ) & 0xFu] * s_up * x2.x;
            acc_up += E2M1_LUT[(ws.y >>  4) & 0xFu] * s_up * x2.y;
            acc_up += E2M1_LUT[(ws.y >>  8) & 0xFu] * s_up * x2.z;
            acc_up += E2M1_LUT[(ws.y >> 12) & 0xFu] * s_up * x2.w;
            acc_up += E2M1_LUT[(ws.y >> 16) & 0xFu] * s_up * x3.x;
            acc_up += E2M1_LUT[(ws.y >> 20) & 0xFu] * s_up * x3.y;
            acc_up += E2M1_LUT[(ws.y >> 24) & 0xFu] * s_up * x3.z;
            acc_up += E2M1_LUT[(ws.y >> 28) & 0xFu] * s_up * x3.w;

            acc_up += E2M1_LUT[(ws.z      ) & 0xFu] * s_up * x4_.x;
            acc_up += E2M1_LUT[(ws.z >>  4) & 0xFu] * s_up * x4_.y;
            acc_up += E2M1_LUT[(ws.z >>  8) & 0xFu] * s_up * x4_.z;
            acc_up += E2M1_LUT[(ws.z >> 12) & 0xFu] * s_up * x4_.w;
            acc_up += E2M1_LUT[(ws.z >> 16) & 0xFu] * s_up * x5.x;
            acc_up += E2M1_LUT[(ws.z >> 20) & 0xFu] * s_up * x5.y;
            acc_up += E2M1_LUT[(ws.z >> 24) & 0xFu] * s_up * x5.z;
            acc_up += E2M1_LUT[(ws.z >> 28) & 0xFu] * s_up * x5.w;

            acc_up += E2M1_LUT[(ws.w      ) & 0xFu] * s_up * x6.x;
            acc_up += E2M1_LUT[(ws.w >>  4) & 0xFu] * s_up * x6.y;
            acc_up += E2M1_LUT[(ws.w >>  8) & 0xFu] * s_up * x6.z;
            acc_up += E2M1_LUT[(ws.w >> 12) & 0xFu] * s_up * x6.w;
            acc_up += E2M1_LUT[(ws.w >> 16) & 0xFu] * s_up * x7.x;
            acc_up += E2M1_LUT[(ws.w >> 20) & 0xFu] * s_up * x7.y;
            acc_up += E2M1_LUT[(ws.w >> 24) & 0xFu] * s_up * x7.z;
            acc_up += E2M1_LUT[(ws.w >> 28) & 0xFu] * s_up * x7.w;
        }
    }

    acc_gate = simd_sum(acc_gate);
    acc_up   = simd_sum(acc_up);

    if (sg_lane == 0u) {
        float silu_g = acc_gate / (1.0f + metal::exp(-acc_gate));
        y[slot * dims.batch * dims.inter + b * dims.inter + row] = silu_g * acc_up;
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Lever D (2026-04-27) — mxfp4_moe_gate_up_silu_mul_f32in_bf16out_v3:
// bf16-output sister of `mxfp4_moe_gate_up_silu_mul_f32_v3` (Lever A).
// Same compute and accumulation precision (f32 inner FMA loop, f32 simd_sum,
// f32 silu); only the device-memory store narrows to `bfloat`. Pairs with
// `mxfp4_matmul_moe_bf16in_f32out_v3` to form a chain that avoids cast-back.
// ───────────────────────────────────────────────────────────────────────────
kernel void mxfp4_moe_gate_up_silu_mul_f32in_bf16out_v3(
    device const uint*     packed_all     [[buffer(0)]],
    device const uchar*    scales_all     [[buffer(1)]],
    device const uint*     expert_indices [[buffer(2)]],
    device const float*    x              [[buffer(3)]],
    device bfloat*         y              [[buffer(4)]],
    constant MxFp4MoeGateUpSiluMulDims& dims [[buffer(5)]],
    threadgroup float*     x_shared       [[threadgroup(0)]],
    uint3 tg_pos            [[threadgroup_position_in_grid]],
    uint  tid_in_tg         [[thread_index_in_threadgroup]],
    uint  sg_id             [[simdgroup_index_in_threadgroup]],
    uint  sg_lane           [[thread_index_in_simdgroup]]
) {
    uint b    = tg_pos.y;
    uint slot = tg_pos.z;
    if (b >= dims.batch) { return; }

    const uint ROWS_PER_TG = 8u;
    const uint THREADS_PER_TG = 256u;
    uint row = tg_pos.x * ROWS_PER_TG + sg_id;

    uint groups        = dims.in_features / 32u;
    uint words_per_row = dims.in_features / 8u;

    uint x_row_base = b * dims.in_features;
    for (uint i = tid_in_tg; i < dims.in_features; i += THREADS_PER_TG) {
        x_shared[i] = x[x_row_base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= dims.inter) { return; }

    uint e = expert_indices[slot];
    uint packed_expert_stride = 2u * dims.inter * words_per_row;
    uint scale_expert_stride  = 2u * dims.inter * groups;

    uint gate_word_base  = e * packed_expert_stride + row * words_per_row;
    uint gate_scale_base = e * scale_expert_stride  + row * groups;
    uint up_word_base    = e * packed_expert_stride + (row + dims.inter) * words_per_row;
    uint up_scale_base   = e * scale_expert_stride  + (row + dims.inter) * groups;

    float acc_gate = 0.0f;
    float acc_up   = 0.0f;

    for (uint g = sg_lane; g < groups; g += 32u) {
        uchar sg_byte = scales_all[gate_scale_base + g];
        uchar su_byte = scales_all[up_scale_base + g];

        float s_gate = 0.0f;
        float s_up   = 0.0f;
        if (sg_byte != 0xFFu) {
            uint sb = uint(sg_byte) << 23;
            s_gate = as_type<float>(sb);
        }
        if (su_byte != 0xFFu) {
            uint sb = uint(su_byte) << 23;
            s_up = as_type<float>(sb);
        }
        if (s_gate == 0.0f && s_up == 0.0f) { continue; }

        uint x_base = g * 32u;
        threadgroup const float4* x4 =
            (threadgroup const float4*)(x_shared + x_base);
        float4 x0 = x4[0]; float4 x1 = x4[1];
        float4 x2 = x4[2]; float4 x3 = x4[3];
        float4 x4_ = x4[4]; float4 x5 = x4[5];
        float4 x6 = x4[6]; float4 x7 = x4[7];

        if (s_gate != 0.0f) {
            uint4 ws = *((device const uint4*)(packed_all + gate_word_base + g * 4u));
            acc_gate += E2M1_LUT[(ws.x      ) & 0xFu] * s_gate * x0.x;
            acc_gate += E2M1_LUT[(ws.x >>  4) & 0xFu] * s_gate * x0.y;
            acc_gate += E2M1_LUT[(ws.x >>  8) & 0xFu] * s_gate * x0.z;
            acc_gate += E2M1_LUT[(ws.x >> 12) & 0xFu] * s_gate * x0.w;
            acc_gate += E2M1_LUT[(ws.x >> 16) & 0xFu] * s_gate * x1.x;
            acc_gate += E2M1_LUT[(ws.x >> 20) & 0xFu] * s_gate * x1.y;
            acc_gate += E2M1_LUT[(ws.x >> 24) & 0xFu] * s_gate * x1.z;
            acc_gate += E2M1_LUT[(ws.x >> 28) & 0xFu] * s_gate * x1.w;

            acc_gate += E2M1_LUT[(ws.y      ) & 0xFu] * s_gate * x2.x;
            acc_gate += E2M1_LUT[(ws.y >>  4) & 0xFu] * s_gate * x2.y;
            acc_gate += E2M1_LUT[(ws.y >>  8) & 0xFu] * s_gate * x2.z;
            acc_gate += E2M1_LUT[(ws.y >> 12) & 0xFu] * s_gate * x2.w;
            acc_gate += E2M1_LUT[(ws.y >> 16) & 0xFu] * s_gate * x3.x;
            acc_gate += E2M1_LUT[(ws.y >> 20) & 0xFu] * s_gate * x3.y;
            acc_gate += E2M1_LUT[(ws.y >> 24) & 0xFu] * s_gate * x3.z;
            acc_gate += E2M1_LUT[(ws.y >> 28) & 0xFu] * s_gate * x3.w;

            acc_gate += E2M1_LUT[(ws.z      ) & 0xFu] * s_gate * x4_.x;
            acc_gate += E2M1_LUT[(ws.z >>  4) & 0xFu] * s_gate * x4_.y;
            acc_gate += E2M1_LUT[(ws.z >>  8) & 0xFu] * s_gate * x4_.z;
            acc_gate += E2M1_LUT[(ws.z >> 12) & 0xFu] * s_gate * x4_.w;
            acc_gate += E2M1_LUT[(ws.z >> 16) & 0xFu] * s_gate * x5.x;
            acc_gate += E2M1_LUT[(ws.z >> 20) & 0xFu] * s_gate * x5.y;
            acc_gate += E2M1_LUT[(ws.z >> 24) & 0xFu] * s_gate * x5.z;
            acc_gate += E2M1_LUT[(ws.z >> 28) & 0xFu] * s_gate * x5.w;

            acc_gate += E2M1_LUT[(ws.w      ) & 0xFu] * s_gate * x6.x;
            acc_gate += E2M1_LUT[(ws.w >>  4) & 0xFu] * s_gate * x6.y;
            acc_gate += E2M1_LUT[(ws.w >>  8) & 0xFu] * s_gate * x6.z;
            acc_gate += E2M1_LUT[(ws.w >> 12) & 0xFu] * s_gate * x6.w;
            acc_gate += E2M1_LUT[(ws.w >> 16) & 0xFu] * s_gate * x7.x;
            acc_gate += E2M1_LUT[(ws.w >> 20) & 0xFu] * s_gate * x7.y;
            acc_gate += E2M1_LUT[(ws.w >> 24) & 0xFu] * s_gate * x7.z;
            acc_gate += E2M1_LUT[(ws.w >> 28) & 0xFu] * s_gate * x7.w;
        }

        if (s_up != 0.0f) {
            uint4 ws = *((device const uint4*)(packed_all + up_word_base + g * 4u));
            acc_up += E2M1_LUT[(ws.x      ) & 0xFu] * s_up * x0.x;
            acc_up += E2M1_LUT[(ws.x >>  4) & 0xFu] * s_up * x0.y;
            acc_up += E2M1_LUT[(ws.x >>  8) & 0xFu] * s_up * x0.z;
            acc_up += E2M1_LUT[(ws.x >> 12) & 0xFu] * s_up * x0.w;
            acc_up += E2M1_LUT[(ws.x >> 16) & 0xFu] * s_up * x1.x;
            acc_up += E2M1_LUT[(ws.x >> 20) & 0xFu] * s_up * x1.y;
            acc_up += E2M1_LUT[(ws.x >> 24) & 0xFu] * s_up * x1.z;
            acc_up += E2M1_LUT[(ws.x >> 28) & 0xFu] * s_up * x1.w;

            acc_up += E2M1_LUT[(ws.y      ) & 0xFu] * s_up * x2.x;
            acc_up += E2M1_LUT[(ws.y >>  4) & 0xFu] * s_up * x2.y;
            acc_up += E2M1_LUT[(ws.y >>  8) & 0xFu] * s_up * x2.z;
            acc_up += E2M1_LUT[(ws.y >> 12) & 0xFu] * s_up * x2.w;
            acc_up += E2M1_LUT[(ws.y >> 16) & 0xFu] * s_up * x3.x;
            acc_up += E2M1_LUT[(ws.y >> 20) & 0xFu] * s_up * x3.y;
            acc_up += E2M1_LUT[(ws.y >> 24) & 0xFu] * s_up * x3.z;
            acc_up += E2M1_LUT[(ws.y >> 28) & 0xFu] * s_up * x3.w;

            acc_up += E2M1_LUT[(ws.z      ) & 0xFu] * s_up * x4_.x;
            acc_up += E2M1_LUT[(ws.z >>  4) & 0xFu] * s_up * x4_.y;
            acc_up += E2M1_LUT[(ws.z >>  8) & 0xFu] * s_up * x4_.z;
            acc_up += E2M1_LUT[(ws.z >> 12) & 0xFu] * s_up * x4_.w;
            acc_up += E2M1_LUT[(ws.z >> 16) & 0xFu] * s_up * x5.x;
            acc_up += E2M1_LUT[(ws.z >> 20) & 0xFu] * s_up * x5.y;
            acc_up += E2M1_LUT[(ws.z >> 24) & 0xFu] * s_up * x5.z;
            acc_up += E2M1_LUT[(ws.z >> 28) & 0xFu] * s_up * x5.w;

            acc_up += E2M1_LUT[(ws.w      ) & 0xFu] * s_up * x6.x;
            acc_up += E2M1_LUT[(ws.w >>  4) & 0xFu] * s_up * x6.y;
            acc_up += E2M1_LUT[(ws.w >>  8) & 0xFu] * s_up * x6.z;
            acc_up += E2M1_LUT[(ws.w >> 12) & 0xFu] * s_up * x6.w;
            acc_up += E2M1_LUT[(ws.w >> 16) & 0xFu] * s_up * x7.x;
            acc_up += E2M1_LUT[(ws.w >> 20) & 0xFu] * s_up * x7.y;
            acc_up += E2M1_LUT[(ws.w >> 24) & 0xFu] * s_up * x7.z;
            acc_up += E2M1_LUT[(ws.w >> 28) & 0xFu] * s_up * x7.w;
        }
    }

    acc_gate = simd_sum(acc_gate);
    acc_up   = simd_sum(acc_up);

    if (sg_lane == 0u) {
        float silu_g = acc_gate / (1.0f + metal::exp(-acc_gate));
        y[slot * dims.batch * dims.inter + b * dims.inter + row] =
            bfloat(silu_g * acc_up);
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Lever H POC (2026-04-27) — mxfp4_moe_gate_up_silu_mul_rmsnorm_f32_v3:
// Lever A's routed-grouped fused gate+up+silu*up kernel with internal
// RmsNorm. Reads raw `x` (un-normalized), computes
// `inv_rms = rsqrt(mean(x²) + eps)` cooperatively, applies
// `x_shared[i] = x[i] * rms_weight[i] * inv_rms` before the matmul loop.
//
// Designed to test whether per-kernel internal RmsNorm is cheap enough to
// justify multi-callsite migration that would eliminate the separate
// `post_attention_layernorm.forward` dispatch. Production wiring requires
// the routing gate + shared expert kernels to also gain `_rmsnorm` variants
// (multi-session work). This variant is the parity + cost-validation POC.
// ───────────────────────────────────────────────────────────────────────────
struct MxFp4MoeGateUpSiluMulRmsnormDims {
    uint inter;
    uint in_features;  // hidden
    uint batch;
    float rms_eps;
};

kernel void mxfp4_moe_gate_up_silu_mul_rmsnorm_f32_v3(
    device const uint*     packed_all     [[buffer(0)]],
    device const uchar*    scales_all     [[buffer(1)]],
    device const uint*     expert_indices [[buffer(2)]],
    device const float*    x              [[buffer(3)]],
    device const float*    rms_weight     [[buffer(4)]],
    device float*          y              [[buffer(5)]],
    constant MxFp4MoeGateUpSiluMulRmsnormDims& dims [[buffer(6)]],
    threadgroup float*     x_shared       [[threadgroup(0)]],
    threadgroup float*     reduce_buf     [[threadgroup(1)]],
    uint3 tg_pos            [[threadgroup_position_in_grid]],
    uint  tid_in_tg         [[thread_index_in_threadgroup]],
    uint  sg_id             [[simdgroup_index_in_threadgroup]],
    uint  sg_lane           [[thread_index_in_simdgroup]]
) {
    uint b    = tg_pos.y;
    uint slot = tg_pos.z;
    if (b >= dims.batch) { return; }

    const uint ROWS_PER_TG = 8u;
    const uint THREADS_PER_TG = 256u;
    uint row = tg_pos.x * ROWS_PER_TG + sg_id;

    uint groups        = dims.in_features / 32u;
    uint words_per_row = dims.in_features / 8u;

    // RmsNorm Phase 1: stage raw x + accumulate per-thread sum(x²).
    uint x_row_base = b * dims.in_features;
    float sum_sq = 0.0f;
    for (uint i = tid_in_tg; i < dims.in_features; i += THREADS_PER_TG) {
        float v = x[x_row_base + i];
        x_shared[i] = v;
        sum_sq = fma(v, v, sum_sq);
    }

    // SG-level partial reduction → 8 SGs each contribute one partial sum.
    sum_sq = simd_sum(sum_sq);
    if (sg_lane == 0u) {
        reduce_buf[sg_id] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // SG 0 reduces the 8 partials → final sum, then computes inv_rms.
    if (sg_id == 0u) {
        float v = (sg_lane < 8u) ? reduce_buf[sg_lane] : 0.0f;
        v = simd_sum(v);
        if (sg_lane == 0u) {
            float mean_sq = v / float(dims.in_features);
            reduce_buf[0] = metal::rsqrt(mean_sq + dims.rms_eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = reduce_buf[0];

    // RmsNorm Phase 2: apply weight × inv_rms in-place.
    for (uint i = tid_in_tg; i < dims.in_features; i += THREADS_PER_TG) {
        x_shared[i] = x_shared[i] * rms_weight[i] * inv_rms;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // === Lever A matmul body (identical from here) ===
    if (row >= dims.inter) { return; }

    uint e = expert_indices[slot];
    uint packed_expert_stride = 2u * dims.inter * words_per_row;
    uint scale_expert_stride  = 2u * dims.inter * groups;

    uint gate_word_base  = e * packed_expert_stride + row * words_per_row;
    uint gate_scale_base = e * scale_expert_stride  + row * groups;
    uint up_word_base    = e * packed_expert_stride + (row + dims.inter) * words_per_row;
    uint up_scale_base   = e * scale_expert_stride  + (row + dims.inter) * groups;

    float acc_gate = 0.0f;
    float acc_up   = 0.0f;

    for (uint g = sg_lane; g < groups; g += 32u) {
        uchar sg_byte = scales_all[gate_scale_base + g];
        uchar su_byte = scales_all[up_scale_base + g];

        float s_gate = 0.0f;
        float s_up   = 0.0f;
        if (sg_byte != 0xFFu) {
            uint sb = uint(sg_byte) << 23;
            s_gate = as_type<float>(sb);
        }
        if (su_byte != 0xFFu) {
            uint sb = uint(su_byte) << 23;
            s_up = as_type<float>(sb);
        }
        if (s_gate == 0.0f && s_up == 0.0f) { continue; }

        uint x_base = g * 32u;
        threadgroup const float4* x4 =
            (threadgroup const float4*)(x_shared + x_base);
        float4 x0 = x4[0]; float4 x1 = x4[1];
        float4 x2 = x4[2]; float4 x3 = x4[3];
        float4 x4_ = x4[4]; float4 x5 = x4[5];
        float4 x6 = x4[6]; float4 x7 = x4[7];

        if (s_gate != 0.0f) {
            uint4 ws = *((device const uint4*)(packed_all + gate_word_base + g * 4u));
            acc_gate += E2M1_LUT[(ws.x      ) & 0xFu] * s_gate * x0.x;
            acc_gate += E2M1_LUT[(ws.x >>  4) & 0xFu] * s_gate * x0.y;
            acc_gate += E2M1_LUT[(ws.x >>  8) & 0xFu] * s_gate * x0.z;
            acc_gate += E2M1_LUT[(ws.x >> 12) & 0xFu] * s_gate * x0.w;
            acc_gate += E2M1_LUT[(ws.x >> 16) & 0xFu] * s_gate * x1.x;
            acc_gate += E2M1_LUT[(ws.x >> 20) & 0xFu] * s_gate * x1.y;
            acc_gate += E2M1_LUT[(ws.x >> 24) & 0xFu] * s_gate * x1.z;
            acc_gate += E2M1_LUT[(ws.x >> 28) & 0xFu] * s_gate * x1.w;

            acc_gate += E2M1_LUT[(ws.y      ) & 0xFu] * s_gate * x2.x;
            acc_gate += E2M1_LUT[(ws.y >>  4) & 0xFu] * s_gate * x2.y;
            acc_gate += E2M1_LUT[(ws.y >>  8) & 0xFu] * s_gate * x2.z;
            acc_gate += E2M1_LUT[(ws.y >> 12) & 0xFu] * s_gate * x2.w;
            acc_gate += E2M1_LUT[(ws.y >> 16) & 0xFu] * s_gate * x3.x;
            acc_gate += E2M1_LUT[(ws.y >> 20) & 0xFu] * s_gate * x3.y;
            acc_gate += E2M1_LUT[(ws.y >> 24) & 0xFu] * s_gate * x3.z;
            acc_gate += E2M1_LUT[(ws.y >> 28) & 0xFu] * s_gate * x3.w;

            acc_gate += E2M1_LUT[(ws.z      ) & 0xFu] * s_gate * x4_.x;
            acc_gate += E2M1_LUT[(ws.z >>  4) & 0xFu] * s_gate * x4_.y;
            acc_gate += E2M1_LUT[(ws.z >>  8) & 0xFu] * s_gate * x4_.z;
            acc_gate += E2M1_LUT[(ws.z >> 12) & 0xFu] * s_gate * x4_.w;
            acc_gate += E2M1_LUT[(ws.z >> 16) & 0xFu] * s_gate * x5.x;
            acc_gate += E2M1_LUT[(ws.z >> 20) & 0xFu] * s_gate * x5.y;
            acc_gate += E2M1_LUT[(ws.z >> 24) & 0xFu] * s_gate * x5.z;
            acc_gate += E2M1_LUT[(ws.z >> 28) & 0xFu] * s_gate * x5.w;

            acc_gate += E2M1_LUT[(ws.w      ) & 0xFu] * s_gate * x6.x;
            acc_gate += E2M1_LUT[(ws.w >>  4) & 0xFu] * s_gate * x6.y;
            acc_gate += E2M1_LUT[(ws.w >>  8) & 0xFu] * s_gate * x6.z;
            acc_gate += E2M1_LUT[(ws.w >> 12) & 0xFu] * s_gate * x6.w;
            acc_gate += E2M1_LUT[(ws.w >> 16) & 0xFu] * s_gate * x7.x;
            acc_gate += E2M1_LUT[(ws.w >> 20) & 0xFu] * s_gate * x7.y;
            acc_gate += E2M1_LUT[(ws.w >> 24) & 0xFu] * s_gate * x7.z;
            acc_gate += E2M1_LUT[(ws.w >> 28) & 0xFu] * s_gate * x7.w;
        }

        if (s_up != 0.0f) {
            uint4 ws = *((device const uint4*)(packed_all + up_word_base + g * 4u));
            acc_up += E2M1_LUT[(ws.x      ) & 0xFu] * s_up * x0.x;
            acc_up += E2M1_LUT[(ws.x >>  4) & 0xFu] * s_up * x0.y;
            acc_up += E2M1_LUT[(ws.x >>  8) & 0xFu] * s_up * x0.z;
            acc_up += E2M1_LUT[(ws.x >> 12) & 0xFu] * s_up * x0.w;
            acc_up += E2M1_LUT[(ws.x >> 16) & 0xFu] * s_up * x1.x;
            acc_up += E2M1_LUT[(ws.x >> 20) & 0xFu] * s_up * x1.y;
            acc_up += E2M1_LUT[(ws.x >> 24) & 0xFu] * s_up * x1.z;
            acc_up += E2M1_LUT[(ws.x >> 28) & 0xFu] * s_up * x1.w;

            acc_up += E2M1_LUT[(ws.y      ) & 0xFu] * s_up * x2.x;
            acc_up += E2M1_LUT[(ws.y >>  4) & 0xFu] * s_up * x2.y;
            acc_up += E2M1_LUT[(ws.y >>  8) & 0xFu] * s_up * x2.z;
            acc_up += E2M1_LUT[(ws.y >> 12) & 0xFu] * s_up * x2.w;
            acc_up += E2M1_LUT[(ws.y >> 16) & 0xFu] * s_up * x3.x;
            acc_up += E2M1_LUT[(ws.y >> 20) & 0xFu] * s_up * x3.y;
            acc_up += E2M1_LUT[(ws.y >> 24) & 0xFu] * s_up * x3.z;
            acc_up += E2M1_LUT[(ws.y >> 28) & 0xFu] * s_up * x3.w;

            acc_up += E2M1_LUT[(ws.z      ) & 0xFu] * s_up * x4_.x;
            acc_up += E2M1_LUT[(ws.z >>  4) & 0xFu] * s_up * x4_.y;
            acc_up += E2M1_LUT[(ws.z >>  8) & 0xFu] * s_up * x4_.z;
            acc_up += E2M1_LUT[(ws.z >> 12) & 0xFu] * s_up * x4_.w;
            acc_up += E2M1_LUT[(ws.z >> 16) & 0xFu] * s_up * x5.x;
            acc_up += E2M1_LUT[(ws.z >> 20) & 0xFu] * s_up * x5.y;
            acc_up += E2M1_LUT[(ws.z >> 24) & 0xFu] * s_up * x5.z;
            acc_up += E2M1_LUT[(ws.z >> 28) & 0xFu] * s_up * x5.w;

            acc_up += E2M1_LUT[(ws.w      ) & 0xFu] * s_up * x6.x;
            acc_up += E2M1_LUT[(ws.w >>  4) & 0xFu] * s_up * x6.y;
            acc_up += E2M1_LUT[(ws.w >>  8) & 0xFu] * s_up * x6.z;
            acc_up += E2M1_LUT[(ws.w >> 12) & 0xFu] * s_up * x6.w;
            acc_up += E2M1_LUT[(ws.w >> 16) & 0xFu] * s_up * x7.x;
            acc_up += E2M1_LUT[(ws.w >> 20) & 0xFu] * s_up * x7.y;
            acc_up += E2M1_LUT[(ws.w >> 24) & 0xFu] * s_up * x7.z;
            acc_up += E2M1_LUT[(ws.w >> 28) & 0xFu] * s_up * x7.w;
        }
    }

    acc_gate = simd_sum(acc_gate);
    acc_up   = simd_sum(acc_up);

    if (sg_lane == 0u) {
        float silu_g = acc_gate / (1.0f + metal::exp(-acc_gate));
        y[slot * dims.batch * dims.inter + b * dims.inter + row] = silu_g * acc_up;
    }
}

// ───────────────────────────────────────────────────────────────────────────
// CB Phase 2 (2026-04-29) multi-token MoE kernels
//
// Three kernels with one-line behavioral change vs their single-token
// siblings: the expert lookup becomes per-batch:
//   single-token: e = expert_indices[slot]                (one expert per slot)
//   multi-token : e = expert_indices[b * k + slot]        (one expert per (b, slot))
//
// All other addressing (TG-shared x staging, output writes) was already
// batch-aware via `b = tg_pos.y` in the v3 kernels — we just unblock it
// for genuinely different per-token expert routing.
//
// Layout convention for multi-token (matches v3 batched output convention):
//   x       : [B, in]              (broadcast over slots)
//   inds    : [B, k]               (per-token expert IDs, flattened)
//   gate_up : writes [k, B, inter]
//   down    : reads  [k, B, inter] → writes [k, B, hidden]
//   wsum    : reads  [k, B, hidden] + [B, k] weights → writes [B, hidden]
//
// Used by `forward_with_rmsnorm` when `bl > 1` (continuous-batching decode).
// Single-token (bl == 1) keeps the v3 path — collapsing to k = 1 buffer
// indexing has no benefit and risks dispatch-overhead regression.
// ───────────────────────────────────────────────────────────────────────────

struct MxFp4MoeGateUpSiluMulRmsnormDimsMulti {
    uint inter;
    uint in_features;
    uint batch;
    uint k;
    float rms_eps;
};

kernel void mxfp4_moe_gate_up_silu_mul_rmsnorm_f32_v3_multi(
    device const uint*     packed_all     [[buffer(0)]],
    device const uchar*    scales_all     [[buffer(1)]],
    device const uint*     expert_indices [[buffer(2)]],   // [B, k]
    device const float*    x              [[buffer(3)]],   // [B, in_features]
    device const float*    rms_weight     [[buffer(4)]],   // [in_features]
    device float*          y              [[buffer(5)]],   // [k, B, inter]
    constant MxFp4MoeGateUpSiluMulRmsnormDimsMulti& dims [[buffer(6)]],
    threadgroup float*     x_shared       [[threadgroup(0)]],
    threadgroup float*     reduce_buf     [[threadgroup(1)]],
    uint3 tg_pos            [[threadgroup_position_in_grid]],
    uint  tid_in_tg         [[thread_index_in_threadgroup]],
    uint  sg_id             [[simdgroup_index_in_threadgroup]],
    uint  sg_lane           [[thread_index_in_simdgroup]]
) {
    uint b    = tg_pos.y;
    uint slot = tg_pos.z;
    if (b >= dims.batch) { return; }

    const uint ROWS_PER_TG = 8u;
    const uint THREADS_PER_TG = 256u;
    uint row = tg_pos.x * ROWS_PER_TG + sg_id;

    uint groups        = dims.in_features / 32u;
    uint words_per_row = dims.in_features / 8u;

    // RmsNorm Phase 1: stage x[b, :] + accumulate sum(x²) cooperatively.
    uint x_row_base = b * dims.in_features;
    float sum_sq = 0.0f;
    for (uint i = tid_in_tg; i < dims.in_features; i += THREADS_PER_TG) {
        float v = x[x_row_base + i];
        x_shared[i] = v;
        sum_sq = fma(v, v, sum_sq);
    }
    sum_sq = simd_sum(sum_sq);
    if (sg_lane == 0u) {
        reduce_buf[sg_id] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0u) {
        float v = (sg_lane < 8u) ? reduce_buf[sg_lane] : 0.0f;
        v = simd_sum(v);
        if (sg_lane == 0u) {
            float mean_sq = v / float(dims.in_features);
            reduce_buf[0] = metal::rsqrt(mean_sq + dims.rms_eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = reduce_buf[0];
    for (uint i = tid_in_tg; i < dims.in_features; i += THREADS_PER_TG) {
        x_shared[i] = x_shared[i] * rms_weight[i] * inv_rms;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= dims.inter) { return; }

    // Per-(b, slot) expert lookup — the only structural change vs v3.
    uint e = expert_indices[b * dims.k + slot];
    uint packed_expert_stride = 2u * dims.inter * words_per_row;
    uint scale_expert_stride  = 2u * dims.inter * groups;

    uint gate_word_base  = e * packed_expert_stride + row * words_per_row;
    uint gate_scale_base = e * scale_expert_stride  + row * groups;
    uint up_word_base    = e * packed_expert_stride + (row + dims.inter) * words_per_row;
    uint up_scale_base   = e * scale_expert_stride  + (row + dims.inter) * groups;

    float acc_gate = 0.0f;
    float acc_up   = 0.0f;

    for (uint g = sg_lane; g < groups; g += 32u) {
        uchar sg_byte = scales_all[gate_scale_base + g];
        uchar su_byte = scales_all[up_scale_base + g];

        float s_gate = 0.0f;
        float s_up   = 0.0f;
        if (sg_byte != 0xFFu) {
            uint sb = uint(sg_byte) << 23;
            s_gate = as_type<float>(sb);
        }
        if (su_byte != 0xFFu) {
            uint sb = uint(su_byte) << 23;
            s_up = as_type<float>(sb);
        }
        if (s_gate == 0.0f && s_up == 0.0f) { continue; }

        uint x_base = g * 32u;
        threadgroup const float4* x4 =
            (threadgroup const float4*)(x_shared + x_base);
        float4 x0 = x4[0]; float4 x1 = x4[1];
        float4 x2 = x4[2]; float4 x3 = x4[3];
        float4 x4_ = x4[4]; float4 x5 = x4[5];
        float4 x6 = x4[6]; float4 x7 = x4[7];

        if (s_gate != 0.0f) {
            uint4 ws = *((device const uint4*)(packed_all + gate_word_base + g * 4u));
            acc_gate += E2M1_LUT[(ws.x      ) & 0xFu] * s_gate * x0.x;
            acc_gate += E2M1_LUT[(ws.x >>  4) & 0xFu] * s_gate * x0.y;
            acc_gate += E2M1_LUT[(ws.x >>  8) & 0xFu] * s_gate * x0.z;
            acc_gate += E2M1_LUT[(ws.x >> 12) & 0xFu] * s_gate * x0.w;
            acc_gate += E2M1_LUT[(ws.x >> 16) & 0xFu] * s_gate * x1.x;
            acc_gate += E2M1_LUT[(ws.x >> 20) & 0xFu] * s_gate * x1.y;
            acc_gate += E2M1_LUT[(ws.x >> 24) & 0xFu] * s_gate * x1.z;
            acc_gate += E2M1_LUT[(ws.x >> 28) & 0xFu] * s_gate * x1.w;

            acc_gate += E2M1_LUT[(ws.y      ) & 0xFu] * s_gate * x2.x;
            acc_gate += E2M1_LUT[(ws.y >>  4) & 0xFu] * s_gate * x2.y;
            acc_gate += E2M1_LUT[(ws.y >>  8) & 0xFu] * s_gate * x2.z;
            acc_gate += E2M1_LUT[(ws.y >> 12) & 0xFu] * s_gate * x2.w;
            acc_gate += E2M1_LUT[(ws.y >> 16) & 0xFu] * s_gate * x3.x;
            acc_gate += E2M1_LUT[(ws.y >> 20) & 0xFu] * s_gate * x3.y;
            acc_gate += E2M1_LUT[(ws.y >> 24) & 0xFu] * s_gate * x3.z;
            acc_gate += E2M1_LUT[(ws.y >> 28) & 0xFu] * s_gate * x3.w;

            acc_gate += E2M1_LUT[(ws.z      ) & 0xFu] * s_gate * x4_.x;
            acc_gate += E2M1_LUT[(ws.z >>  4) & 0xFu] * s_gate * x4_.y;
            acc_gate += E2M1_LUT[(ws.z >>  8) & 0xFu] * s_gate * x4_.z;
            acc_gate += E2M1_LUT[(ws.z >> 12) & 0xFu] * s_gate * x4_.w;
            acc_gate += E2M1_LUT[(ws.z >> 16) & 0xFu] * s_gate * x5.x;
            acc_gate += E2M1_LUT[(ws.z >> 20) & 0xFu] * s_gate * x5.y;
            acc_gate += E2M1_LUT[(ws.z >> 24) & 0xFu] * s_gate * x5.z;
            acc_gate += E2M1_LUT[(ws.z >> 28) & 0xFu] * s_gate * x5.w;

            acc_gate += E2M1_LUT[(ws.w      ) & 0xFu] * s_gate * x6.x;
            acc_gate += E2M1_LUT[(ws.w >>  4) & 0xFu] * s_gate * x6.y;
            acc_gate += E2M1_LUT[(ws.w >>  8) & 0xFu] * s_gate * x6.z;
            acc_gate += E2M1_LUT[(ws.w >> 12) & 0xFu] * s_gate * x6.w;
            acc_gate += E2M1_LUT[(ws.w >> 16) & 0xFu] * s_gate * x7.x;
            acc_gate += E2M1_LUT[(ws.w >> 20) & 0xFu] * s_gate * x7.y;
            acc_gate += E2M1_LUT[(ws.w >> 24) & 0xFu] * s_gate * x7.z;
            acc_gate += E2M1_LUT[(ws.w >> 28) & 0xFu] * s_gate * x7.w;
        }

        if (s_up != 0.0f) {
            uint4 ws = *((device const uint4*)(packed_all + up_word_base + g * 4u));
            acc_up += E2M1_LUT[(ws.x      ) & 0xFu] * s_up * x0.x;
            acc_up += E2M1_LUT[(ws.x >>  4) & 0xFu] * s_up * x0.y;
            acc_up += E2M1_LUT[(ws.x >>  8) & 0xFu] * s_up * x0.z;
            acc_up += E2M1_LUT[(ws.x >> 12) & 0xFu] * s_up * x0.w;
            acc_up += E2M1_LUT[(ws.x >> 16) & 0xFu] * s_up * x1.x;
            acc_up += E2M1_LUT[(ws.x >> 20) & 0xFu] * s_up * x1.y;
            acc_up += E2M1_LUT[(ws.x >> 24) & 0xFu] * s_up * x1.z;
            acc_up += E2M1_LUT[(ws.x >> 28) & 0xFu] * s_up * x1.w;

            acc_up += E2M1_LUT[(ws.y      ) & 0xFu] * s_up * x2.x;
            acc_up += E2M1_LUT[(ws.y >>  4) & 0xFu] * s_up * x2.y;
            acc_up += E2M1_LUT[(ws.y >>  8) & 0xFu] * s_up * x2.z;
            acc_up += E2M1_LUT[(ws.y >> 12) & 0xFu] * s_up * x2.w;
            acc_up += E2M1_LUT[(ws.y >> 16) & 0xFu] * s_up * x3.x;
            acc_up += E2M1_LUT[(ws.y >> 20) & 0xFu] * s_up * x3.y;
            acc_up += E2M1_LUT[(ws.y >> 24) & 0xFu] * s_up * x3.z;
            acc_up += E2M1_LUT[(ws.y >> 28) & 0xFu] * s_up * x3.w;

            acc_up += E2M1_LUT[(ws.z      ) & 0xFu] * s_up * x4_.x;
            acc_up += E2M1_LUT[(ws.z >>  4) & 0xFu] * s_up * x4_.y;
            acc_up += E2M1_LUT[(ws.z >>  8) & 0xFu] * s_up * x4_.z;
            acc_up += E2M1_LUT[(ws.z >> 12) & 0xFu] * s_up * x4_.w;
            acc_up += E2M1_LUT[(ws.z >> 16) & 0xFu] * s_up * x5.x;
            acc_up += E2M1_LUT[(ws.z >> 20) & 0xFu] * s_up * x5.y;
            acc_up += E2M1_LUT[(ws.z >> 24) & 0xFu] * s_up * x5.z;
            acc_up += E2M1_LUT[(ws.z >> 28) & 0xFu] * s_up * x5.w;

            acc_up += E2M1_LUT[(ws.w      ) & 0xFu] * s_up * x6.x;
            acc_up += E2M1_LUT[(ws.w >>  4) & 0xFu] * s_up * x6.y;
            acc_up += E2M1_LUT[(ws.w >>  8) & 0xFu] * s_up * x6.z;
            acc_up += E2M1_LUT[(ws.w >> 12) & 0xFu] * s_up * x6.w;
            acc_up += E2M1_LUT[(ws.w >> 16) & 0xFu] * s_up * x7.x;
            acc_up += E2M1_LUT[(ws.w >> 20) & 0xFu] * s_up * x7.y;
            acc_up += E2M1_LUT[(ws.w >> 24) & 0xFu] * s_up * x7.z;
            acc_up += E2M1_LUT[(ws.w >> 28) & 0xFu] * s_up * x7.w;
        }
    }

    acc_gate = simd_sum(acc_gate);
    acc_up   = simd_sum(acc_up);

    if (sg_lane == 0u) {
        float silu_g = acc_gate / (1.0f + metal::exp(-acc_gate));
        y[slot * dims.batch * dims.inter + b * dims.inter + row] = silu_g * acc_up;
    }
}

struct MxFp4MoeDimsMulti {
    uint out_features;
    uint in_features;
    uint batch;
    uint k;
};

kernel void mxfp4_matmul_moe_f32_v3_multi(
    device const uint*     packed_all     [[buffer(0)]],
    device const uchar*    scales_all     [[buffer(1)]],
    device const uint*     expert_indices [[buffer(2)]],   // [B, k]
    device const float*    x              [[buffer(3)]],   // [k, B, in_features]
    device float*          y              [[buffer(4)]],   // [k, B, out_features]
    constant MxFp4MoeDimsMulti& dims      [[buffer(5)]],
    threadgroup float*     x_shared       [[threadgroup(0)]],
    uint3 tg_pos            [[threadgroup_position_in_grid]],
    uint  tid_in_tg         [[thread_index_in_threadgroup]],
    uint  sg_id             [[simdgroup_index_in_threadgroup]],
    uint  sg_lane           [[thread_index_in_simdgroup]]
) {
    uint b    = tg_pos.y;
    uint slot = tg_pos.z;
    if (b >= dims.batch) { return; }

    const uint ROWS_PER_TG = 8u;
    const uint THREADS_PER_TG = 256u;
    uint row = tg_pos.x * ROWS_PER_TG + sg_id;

    uint groups        = dims.in_features / 32u;
    uint words_per_row = dims.in_features / 8u;

    // x is per-(slot, b) here (down kernel input is gate_up output [k, B, in])
    uint x_row_base = slot * dims.batch * dims.in_features
                    + b * dims.in_features;
    for (uint i = tid_in_tg; i < dims.in_features; i += THREADS_PER_TG) {
        x_shared[i] = x[x_row_base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= dims.out_features) { return; }

    uint e = expert_indices[b * dims.k + slot];
    uint packed_expert_stride = dims.out_features * words_per_row;
    uint scale_expert_stride  = dims.out_features * groups;
    uint word_row_base  = e * packed_expert_stride + row * words_per_row;
    uint scale_row_base = e * scale_expert_stride  + row * groups;

    float acc = 0.0f;
    for (uint g = sg_lane; g < groups; g += 32u) {
        uchar scale_byte = scales_all[scale_row_base + g];
        if (scale_byte == 0xFFu) continue;
        uint sbits = uint(scale_byte) << 23;
        float s = as_type<float>(sbits);
        if (s == 0.0f) continue;

        uint word_base = word_row_base + g * 4u;
        uint x_base    = g * 32u;

        uint4 ws = *((device const uint4*)(packed_all + word_base));
        threadgroup const float4* x4 =
            (threadgroup const float4*)(x_shared + x_base);
        float4 x0 = x4[0]; float4 x1 = x4[1];
        float4 x2 = x4[2]; float4 x3 = x4[3];
        float4 x4_ = x4[4]; float4 x5 = x4[5];
        float4 x6 = x4[6]; float4 x7 = x4[7];

        acc += E2M1_LUT[(ws.x      ) & 0xFu] * s * x0.x;
        acc += E2M1_LUT[(ws.x >>  4) & 0xFu] * s * x0.y;
        acc += E2M1_LUT[(ws.x >>  8) & 0xFu] * s * x0.z;
        acc += E2M1_LUT[(ws.x >> 12) & 0xFu] * s * x0.w;
        acc += E2M1_LUT[(ws.x >> 16) & 0xFu] * s * x1.x;
        acc += E2M1_LUT[(ws.x >> 20) & 0xFu] * s * x1.y;
        acc += E2M1_LUT[(ws.x >> 24) & 0xFu] * s * x1.z;
        acc += E2M1_LUT[(ws.x >> 28) & 0xFu] * s * x1.w;

        acc += E2M1_LUT[(ws.y      ) & 0xFu] * s * x2.x;
        acc += E2M1_LUT[(ws.y >>  4) & 0xFu] * s * x2.y;
        acc += E2M1_LUT[(ws.y >>  8) & 0xFu] * s * x2.z;
        acc += E2M1_LUT[(ws.y >> 12) & 0xFu] * s * x2.w;
        acc += E2M1_LUT[(ws.y >> 16) & 0xFu] * s * x3.x;
        acc += E2M1_LUT[(ws.y >> 20) & 0xFu] * s * x3.y;
        acc += E2M1_LUT[(ws.y >> 24) & 0xFu] * s * x3.z;
        acc += E2M1_LUT[(ws.y >> 28) & 0xFu] * s * x3.w;

        acc += E2M1_LUT[(ws.z      ) & 0xFu] * s * x4_.x;
        acc += E2M1_LUT[(ws.z >>  4) & 0xFu] * s * x4_.y;
        acc += E2M1_LUT[(ws.z >>  8) & 0xFu] * s * x4_.z;
        acc += E2M1_LUT[(ws.z >> 12) & 0xFu] * s * x4_.w;
        acc += E2M1_LUT[(ws.z >> 16) & 0xFu] * s * x5.x;
        acc += E2M1_LUT[(ws.z >> 20) & 0xFu] * s * x5.y;
        acc += E2M1_LUT[(ws.z >> 24) & 0xFu] * s * x5.z;
        acc += E2M1_LUT[(ws.z >> 28) & 0xFu] * s * x5.w;

        acc += E2M1_LUT[(ws.w      ) & 0xFu] * s * x6.x;
        acc += E2M1_LUT[(ws.w >>  4) & 0xFu] * s * x6.y;
        acc += E2M1_LUT[(ws.w >>  8) & 0xFu] * s * x6.z;
        acc += E2M1_LUT[(ws.w >> 12) & 0xFu] * s * x6.w;
        acc += E2M1_LUT[(ws.w >> 16) & 0xFu] * s * x7.x;
        acc += E2M1_LUT[(ws.w >> 20) & 0xFu] * s * x7.y;
        acc += E2M1_LUT[(ws.w >> 24) & 0xFu] * s * x7.z;
        acc += E2M1_LUT[(ws.w >> 28) & 0xFu] * s * x7.w;
    }

    acc = simd_sum(acc);
    if (sg_lane == 0u) {
        y[slot * dims.batch * dims.out_features + b * dims.out_features + row] = acc;
    }
}

struct MoeWsumDimsMulti {
    uint k;
    uint batch;
    uint hidden;
};

kernel void moe_wsum_f32_multi(
    device const float* downs   [[buffer(0)]],   // [k, B, hidden]
    device const float* weights [[buffer(1)]],   // [B, k]
    device float*       out     [[buffer(2)]],   // [B, hidden]
    constant MoeWsumDimsMulti& dims [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint h = tid.x;
    uint b = tid.y;
    if (h >= dims.hidden || b >= dims.batch) { return; }
    float acc = 0.0f;
    uint slot_stride = dims.batch * dims.hidden;
    for (uint e = 0u; e < dims.k; e++) {
        acc += weights[b * dims.k + e]
             * downs[e * slot_stride + b * dims.hidden + h];
    }
    out[b * dims.hidden + h] = acc;
}

// ───────────────────────────────────────────────────────────────────────────
// mxfp4_matmul_f32_v3_rmsnorm — Lever H multi-callsite migration kernel.
// f32-output sister of `mxfp4_matmul_f32_v3` that internally computes RmsNorm
// on the input x before the matmul, eliminating the separate
// `post_attention_layernorm.forward` device dispatch when paired with the
// matching kernels for the routed gate_up (`*_moe_gate_up_silu_mul_rmsnorm_*`)
// and shared expert.
//
// This single kernel covers BOTH consumers in production hot path:
//   - Routing gate (small mxfp4 matmul producing [BL, num_experts])
//   - Shared expert gate_up (mxfp4 matmul producing [BL, 2*shared_inter])
// because both default to `mxfp4_matmul_f32_v3` topology when their respective
// opt-in flags (`LUMEN_ENABLE_SMALL_OUT_GATE`,
// `LUMEN_ENABLE_GATE_UP_SILU_MUL_FUSION`) are OFF.
//
// Buffer layout (mirrors `*_rmsnorm_*_v3` POC convention, with `expert_indices`
// removed since this is a non-routed matmul):
//   buffer(0): packed
//   buffer(1): scales
//   buffer(2): x          (raw post-attn residual, BEFORE rms norm)
//   buffer(3): rms_weight (the post_attention_layernorm weight, [in_features])
//   buffer(4): y          (output, [batch, out_features])
//   buffer(5): dims       (out_features, in_features, rms_eps)
//   buffer(6): batch
//
// Threadgroup memory:
//   tg(0): x_shared   [in_features]  — RmsNorm output cache
//   tg(1): reduce_buf [8]            — SG partial sums + inv_rms broadcast
//
// RmsNorm reduction order: Phase 1 stages raw x and per-thread accumulates
// sum(x²) (256 threads, each handling stride-256 chunks); SG-level simd_sum
// reduces 32 lanes to one partial; 8 SGs write their partials into
// reduce_buf[0..8]; SG 0 reduces those 8 partials with simd_sum; lane 0
// computes `inv_rms = rsqrt(mean_sq + eps)` and broadcasts via reduce_buf[0].
// Phase 2 applies `x_shared[i] = x_shared[i] * rms_weight[i] * inv_rms`.
// Two threadgroup_barriers vs. unfused.
//
// Cosine ≥ 0.999 vs CPU pre-RmsNorm + unfused v3 reference (parity test).
// ───────────────────────────────────────────────────────────────────────────
struct MxFp4MatmulRmsnormDims {
    uint out_features;
    uint in_features;
    float rms_eps;
};

kernel void mxfp4_matmul_f32_v3_rmsnorm(
    device const uint*    packed     [[buffer(0)]],
    device const uchar*   scales     [[buffer(1)]],
    device const float*   x          [[buffer(2)]],
    device const float*   rms_weight [[buffer(3)]],
    device float*         y          [[buffer(4)]],
    constant MxFp4MatmulRmsnormDims& dims [[buffer(5)]],
    constant uint&        batch      [[buffer(6)]],
    threadgroup float*    x_shared   [[threadgroup(0)]],
    threadgroup float*    reduce_buf [[threadgroup(1)]],
    uint3 tg_pos          [[threadgroup_position_in_grid]],
    uint  tid_in_tg       [[thread_index_in_threadgroup]],
    uint  sg_id           [[simdgroup_index_in_threadgroup]],
    uint  sg_lane         [[thread_index_in_simdgroup]]
) {
    uint b = tg_pos.y;
    if (b >= batch) { return; }

    const uint ROWS_PER_TG = 8u;
    const uint THREADS_PER_TG = 256u;
    uint row = tg_pos.x * ROWS_PER_TG + sg_id;

    uint groups         = dims.in_features / 32u;
    uint words_per_row  = dims.in_features / 8u;
    uint x_row_base     = b * dims.in_features;

    // RmsNorm Phase 1: stage raw x into x_shared + accumulate per-thread sum(x²).
    float sum_sq = 0.0f;
    for (uint i = tid_in_tg; i < dims.in_features; i += THREADS_PER_TG) {
        float v = x[x_row_base + i];
        x_shared[i] = v;
        sum_sq = fma(v, v, sum_sq);
    }

    // SG-level partial reduction → 8 SGs each contribute one partial sum.
    sum_sq = simd_sum(sum_sq);
    if (sg_lane == 0u) {
        reduce_buf[sg_id] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // SG 0 reduces the 8 partials → final sum, then computes inv_rms.
    if (sg_id == 0u) {
        float v = (sg_lane < 8u) ? reduce_buf[sg_lane] : 0.0f;
        v = simd_sum(v);
        if (sg_lane == 0u) {
            float mean_sq = v / float(dims.in_features);
            reduce_buf[0] = metal::rsqrt(mean_sq + dims.rms_eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = reduce_buf[0];

    // RmsNorm Phase 2: apply weight × inv_rms in-place on x_shared.
    for (uint i = tid_in_tg; i < dims.in_features; i += THREADS_PER_TG) {
        x_shared[i] = x_shared[i] * rms_weight[i] * inv_rms;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // === v3 matmul body (identical from here, reads x_shared) ===
    if (row >= dims.out_features) { return; }

    uint word_row_base  = row * words_per_row;
    uint scale_row_base = row * groups;

    float acc = 0.0f;
    for (uint g = sg_lane; g < groups; g += 32u) {
        uchar scale_byte = scales[scale_row_base + g];
        if (scale_byte == 0xFFu) continue;
        uint sbits = uint(scale_byte) << 23;
        float s = as_type<float>(sbits);
        if (s == 0.0f) continue;

        uint word_base = word_row_base + g * 4u;
        uint x_base    = g * 32u;

        uint4 ws = *((device const uint4*)(packed + word_base));
        threadgroup const float4* x4 =
            (threadgroup const float4*)(x_shared + x_base);
        float4 x0 = x4[0]; float4 x1 = x4[1];
        float4 x2 = x4[2]; float4 x3 = x4[3];
        float4 x4_ = x4[4]; float4 x5 = x4[5];
        float4 x6 = x4[6]; float4 x7 = x4[7];

        acc += E2M1_LUT[(ws.x      ) & 0xFu] * s * x0.x;
        acc += E2M1_LUT[(ws.x >>  4) & 0xFu] * s * x0.y;
        acc += E2M1_LUT[(ws.x >>  8) & 0xFu] * s * x0.z;
        acc += E2M1_LUT[(ws.x >> 12) & 0xFu] * s * x0.w;
        acc += E2M1_LUT[(ws.x >> 16) & 0xFu] * s * x1.x;
        acc += E2M1_LUT[(ws.x >> 20) & 0xFu] * s * x1.y;
        acc += E2M1_LUT[(ws.x >> 24) & 0xFu] * s * x1.z;
        acc += E2M1_LUT[(ws.x >> 28) & 0xFu] * s * x1.w;

        acc += E2M1_LUT[(ws.y      ) & 0xFu] * s * x2.x;
        acc += E2M1_LUT[(ws.y >>  4) & 0xFu] * s * x2.y;
        acc += E2M1_LUT[(ws.y >>  8) & 0xFu] * s * x2.z;
        acc += E2M1_LUT[(ws.y >> 12) & 0xFu] * s * x2.w;
        acc += E2M1_LUT[(ws.y >> 16) & 0xFu] * s * x3.x;
        acc += E2M1_LUT[(ws.y >> 20) & 0xFu] * s * x3.y;
        acc += E2M1_LUT[(ws.y >> 24) & 0xFu] * s * x3.z;
        acc += E2M1_LUT[(ws.y >> 28) & 0xFu] * s * x3.w;

        acc += E2M1_LUT[(ws.z      ) & 0xFu] * s * x4_.x;
        acc += E2M1_LUT[(ws.z >>  4) & 0xFu] * s * x4_.y;
        acc += E2M1_LUT[(ws.z >>  8) & 0xFu] * s * x4_.z;
        acc += E2M1_LUT[(ws.z >> 12) & 0xFu] * s * x4_.w;
        acc += E2M1_LUT[(ws.z >> 16) & 0xFu] * s * x5.x;
        acc += E2M1_LUT[(ws.z >> 20) & 0xFu] * s * x5.y;
        acc += E2M1_LUT[(ws.z >> 24) & 0xFu] * s * x5.z;
        acc += E2M1_LUT[(ws.z >> 28) & 0xFu] * s * x5.w;

        acc += E2M1_LUT[(ws.w      ) & 0xFu] * s * x6.x;
        acc += E2M1_LUT[(ws.w >>  4) & 0xFu] * s * x6.y;
        acc += E2M1_LUT[(ws.w >>  8) & 0xFu] * s * x6.z;
        acc += E2M1_LUT[(ws.w >> 12) & 0xFu] * s * x6.w;
        acc += E2M1_LUT[(ws.w >> 16) & 0xFu] * s * x7.x;
        acc += E2M1_LUT[(ws.w >> 20) & 0xFu] * s * x7.y;
        acc += E2M1_LUT[(ws.w >> 24) & 0xFu] * s * x7.z;
        acc += E2M1_LUT[(ws.w >> 28) & 0xFu] * s * x7.w;
    }

    acc = simd_sum(acc);

    if (sg_lane == 0u) {
        y[b * dims.out_features + row] = acc;
    }
}

// ───────────────────────────────────────────────────────────────────────────
// mxfp4_matmul_f32_v3_rmsnorm_large — Lever H Step 3 retry (out_features ≥ 8192).
//
// Identical RmsNorm + matmul logic as `mxfp4_matmul_f32_v3_rmsnorm` but with
// a doubled threadgroup topology (16 rows/TG × 512 threads = 16 SGs × 32 lanes).
// Halves the redundant RmsNorm work for large-out callsites by halving the TG
// count: out=9216 → 576 TGs (vs. 1152 with the small variant), out=12352 →
// 772 TGs (vs. 1544). Each TG redundantly recomputes the same RmsNorm of x
// (same input across the whole grid), so fewer TGs ≈ less wasted work.
//
// Sweet-spot rationale (see playbook_dispatch_fusion.md Pattern G):
//   out ≤ 4096 (post-attn MoE callsites) — the small variant (8 rows/TG, 256
//     threads) is optimal because TG count = out/8 ≤ 512 already saturates
//     concurrency without extra reduction overhead.
//   out ≥ 8192 (pre-attn qkv/in_proj callsites) — the large variant (this
//     kernel) reduces redundant reduction work which dominated the regression.
//
// Topology details vs small variant:
//   ROWS_PER_TG: 8 → 16
//   THREADS_PER_TG: 256 → 512  (still ≤ Apple Silicon 1024 max)
//   reduce_buf size: 8 → 16
//   SG 0 final reduce reads sg_lane < 16 partials (vs. < 8)
//   Threadgroup memory: x_shared[in_features] + reduce_buf[16] (~8KB + 64B)
//
// Cosine ≥ 0.999 vs CPU pre-RmsNorm + unfused v3 reference (parity test).
// ───────────────────────────────────────────────────────────────────────────
kernel void mxfp4_matmul_f32_v3_rmsnorm_large(
    device const uint*    packed     [[buffer(0)]],
    device const uchar*   scales     [[buffer(1)]],
    device const float*   x          [[buffer(2)]],
    device const float*   rms_weight [[buffer(3)]],
    device float*         y          [[buffer(4)]],
    constant MxFp4MatmulRmsnormDims& dims [[buffer(5)]],
    constant uint&        batch      [[buffer(6)]],
    threadgroup float*    x_shared   [[threadgroup(0)]],
    threadgroup float*    reduce_buf [[threadgroup(1)]],
    uint3 tg_pos          [[threadgroup_position_in_grid]],
    uint  tid_in_tg       [[thread_index_in_threadgroup]],
    uint  sg_id           [[simdgroup_index_in_threadgroup]],
    uint  sg_lane         [[thread_index_in_simdgroup]]
) {
    uint b = tg_pos.y;
    if (b >= batch) { return; }

    const uint ROWS_PER_TG = 16u;
    const uint THREADS_PER_TG = 512u;
    uint row = tg_pos.x * ROWS_PER_TG + sg_id;

    uint groups         = dims.in_features / 32u;
    uint words_per_row  = dims.in_features / 8u;
    uint x_row_base     = b * dims.in_features;

    // RmsNorm Phase 1: stage raw x into x_shared + accumulate per-thread sum(x²).
    float sum_sq = 0.0f;
    for (uint i = tid_in_tg; i < dims.in_features; i += THREADS_PER_TG) {
        float v = x[x_row_base + i];
        x_shared[i] = v;
        sum_sq = fma(v, v, sum_sq);
    }

    // SG-level partial reduction → 16 SGs each contribute one partial sum.
    sum_sq = simd_sum(sum_sq);
    if (sg_lane == 0u) {
        reduce_buf[sg_id] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // SG 0 reduces the 16 partials → final sum, then computes inv_rms.
    if (sg_id == 0u) {
        float v = (sg_lane < 16u) ? reduce_buf[sg_lane] : 0.0f;
        v = simd_sum(v);
        if (sg_lane == 0u) {
            float mean_sq = v / float(dims.in_features);
            reduce_buf[0] = metal::rsqrt(mean_sq + dims.rms_eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = reduce_buf[0];

    // RmsNorm Phase 2: apply weight × inv_rms in-place on x_shared.
    for (uint i = tid_in_tg; i < dims.in_features; i += THREADS_PER_TG) {
        x_shared[i] = x_shared[i] * rms_weight[i] * inv_rms;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // === v3 matmul body (identical from here, reads x_shared) ===
    if (row >= dims.out_features) { return; }

    uint word_row_base  = row * words_per_row;
    uint scale_row_base = row * groups;

    float acc = 0.0f;
    for (uint g = sg_lane; g < groups; g += 32u) {
        uchar scale_byte = scales[scale_row_base + g];
        if (scale_byte == 0xFFu) continue;
        uint sbits = uint(scale_byte) << 23;
        float s = as_type<float>(sbits);
        if (s == 0.0f) continue;

        uint word_base = word_row_base + g * 4u;
        uint x_base    = g * 32u;

        uint4 ws = *((device const uint4*)(packed + word_base));
        threadgroup const float4* x4 =
            (threadgroup const float4*)(x_shared + x_base);
        float4 x0 = x4[0]; float4 x1 = x4[1];
        float4 x2 = x4[2]; float4 x3 = x4[3];
        float4 x4_ = x4[4]; float4 x5 = x4[5];
        float4 x6 = x4[6]; float4 x7 = x4[7];

        acc += E2M1_LUT[(ws.x      ) & 0xFu] * s * x0.x;
        acc += E2M1_LUT[(ws.x >>  4) & 0xFu] * s * x0.y;
        acc += E2M1_LUT[(ws.x >>  8) & 0xFu] * s * x0.z;
        acc += E2M1_LUT[(ws.x >> 12) & 0xFu] * s * x0.w;
        acc += E2M1_LUT[(ws.x >> 16) & 0xFu] * s * x1.x;
        acc += E2M1_LUT[(ws.x >> 20) & 0xFu] * s * x1.y;
        acc += E2M1_LUT[(ws.x >> 24) & 0xFu] * s * x1.z;
        acc += E2M1_LUT[(ws.x >> 28) & 0xFu] * s * x1.w;

        acc += E2M1_LUT[(ws.y      ) & 0xFu] * s * x2.x;
        acc += E2M1_LUT[(ws.y >>  4) & 0xFu] * s * x2.y;
        acc += E2M1_LUT[(ws.y >>  8) & 0xFu] * s * x2.z;
        acc += E2M1_LUT[(ws.y >> 12) & 0xFu] * s * x2.w;
        acc += E2M1_LUT[(ws.y >> 16) & 0xFu] * s * x3.x;
        acc += E2M1_LUT[(ws.y >> 20) & 0xFu] * s * x3.y;
        acc += E2M1_LUT[(ws.y >> 24) & 0xFu] * s * x3.z;
        acc += E2M1_LUT[(ws.y >> 28) & 0xFu] * s * x3.w;

        acc += E2M1_LUT[(ws.z      ) & 0xFu] * s * x4_.x;
        acc += E2M1_LUT[(ws.z >>  4) & 0xFu] * s * x4_.y;
        acc += E2M1_LUT[(ws.z >>  8) & 0xFu] * s * x4_.z;
        acc += E2M1_LUT[(ws.z >> 12) & 0xFu] * s * x4_.w;
        acc += E2M1_LUT[(ws.z >> 16) & 0xFu] * s * x5.x;
        acc += E2M1_LUT[(ws.z >> 20) & 0xFu] * s * x5.y;
        acc += E2M1_LUT[(ws.z >> 24) & 0xFu] * s * x5.z;
        acc += E2M1_LUT[(ws.z >> 28) & 0xFu] * s * x5.w;

        acc += E2M1_LUT[(ws.w      ) & 0xFu] * s * x6.x;
        acc += E2M1_LUT[(ws.w >>  4) & 0xFu] * s * x6.y;
        acc += E2M1_LUT[(ws.w >>  8) & 0xFu] * s * x6.z;
        acc += E2M1_LUT[(ws.w >> 12) & 0xFu] * s * x6.w;
        acc += E2M1_LUT[(ws.w >> 16) & 0xFu] * s * x7.x;
        acc += E2M1_LUT[(ws.w >> 20) & 0xFu] * s * x7.y;
        acc += E2M1_LUT[(ws.w >> 24) & 0xFu] * s * x7.z;
        acc += E2M1_LUT[(ws.w >> 28) & 0xFu] * s * x7.w;
    }

    acc = simd_sum(acc);

    if (sg_lane == 0u) {
        y[b * dims.out_features + row] = acc;
    }
}

// ───────────────────────────────────────────────────────────────────────────
// mxfp4_matmul_f32_v3_rmsnorm_xlarge — Lever H Step 3 retry tier 2 (max TG).
//
// Same logic as `mxfp4_matmul_f32_v3_rmsnorm` but with the maximum Apple
// Silicon threadgroup topology (32 rows/TG × 1024 threads = 32 SGs × 32
// lanes). Quarters the redundant RmsNorm work vs the small variant for very
// large outputs: out=9216 → 288 TGs (vs 1152), out=12352 → 386 TGs.
// Each TG's Phase 1 (stage + sum-of-squares) is also 4× faster per-TG due
// to 4× more threads cooperating on the same in_features data.
//
// Threadgroup resources:
//   - x_shared[in_features] f32: ~8KB at in_features=2048
//   - reduce_buf[32] f32: 128B (one SG 0 simd_sum reduces all partials)
//   - threads: 1024 (Apple Silicon max)
//   - SGs: 32
//
// Cosine ≥ 0.999 vs CPU pre-RmsNorm + unfused v3 reference (parity test).
// ───────────────────────────────────────────────────────────────────────────
kernel void mxfp4_matmul_f32_v3_rmsnorm_xlarge(
    device const uint*    packed     [[buffer(0)]],
    device const uchar*   scales     [[buffer(1)]],
    device const float*   x          [[buffer(2)]],
    device const float*   rms_weight [[buffer(3)]],
    device float*         y          [[buffer(4)]],
    constant MxFp4MatmulRmsnormDims& dims [[buffer(5)]],
    constant uint&        batch      [[buffer(6)]],
    threadgroup float*    x_shared   [[threadgroup(0)]],
    threadgroup float*    reduce_buf [[threadgroup(1)]],
    uint3 tg_pos          [[threadgroup_position_in_grid]],
    uint  tid_in_tg       [[thread_index_in_threadgroup]],
    uint  sg_id           [[simdgroup_index_in_threadgroup]],
    uint  sg_lane         [[thread_index_in_simdgroup]]
) {
    uint b = tg_pos.y;
    if (b >= batch) { return; }

    const uint ROWS_PER_TG = 32u;
    const uint THREADS_PER_TG = 1024u;
    uint row = tg_pos.x * ROWS_PER_TG + sg_id;

    uint groups         = dims.in_features / 32u;
    uint words_per_row  = dims.in_features / 8u;
    uint x_row_base     = b * dims.in_features;

    // RmsNorm Phase 1: stage raw x into x_shared + accumulate per-thread sum(x²).
    float sum_sq = 0.0f;
    for (uint i = tid_in_tg; i < dims.in_features; i += THREADS_PER_TG) {
        float v = x[x_row_base + i];
        x_shared[i] = v;
        sum_sq = fma(v, v, sum_sq);
    }

    // SG-level partial reduction → 32 SGs each contribute one partial sum.
    sum_sq = simd_sum(sum_sq);
    if (sg_lane == 0u) {
        reduce_buf[sg_id] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // SG 0 reduces the 32 partials in a single simd_sum (sg_lane < 32 covers
    // all partials), then computes inv_rms.
    if (sg_id == 0u) {
        float v = reduce_buf[sg_lane];  // sg_lane in [0, 32) — full coverage
        v = simd_sum(v);
        if (sg_lane == 0u) {
            float mean_sq = v / float(dims.in_features);
            reduce_buf[0] = metal::rsqrt(mean_sq + dims.rms_eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = reduce_buf[0];

    // RmsNorm Phase 2: apply weight × inv_rms in-place on x_shared.
    for (uint i = tid_in_tg; i < dims.in_features; i += THREADS_PER_TG) {
        x_shared[i] = x_shared[i] * rms_weight[i] * inv_rms;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // === v3 matmul body (identical from here, reads x_shared) ===
    if (row >= dims.out_features) { return; }

    uint word_row_base  = row * words_per_row;
    uint scale_row_base = row * groups;

    float acc = 0.0f;
    for (uint g = sg_lane; g < groups; g += 32u) {
        uchar scale_byte = scales[scale_row_base + g];
        if (scale_byte == 0xFFu) continue;
        uint sbits = uint(scale_byte) << 23;
        float s = as_type<float>(sbits);
        if (s == 0.0f) continue;

        uint word_base = word_row_base + g * 4u;
        uint x_base    = g * 32u;

        uint4 ws = *((device const uint4*)(packed + word_base));
        threadgroup const float4* x4 =
            (threadgroup const float4*)(x_shared + x_base);
        float4 x0 = x4[0]; float4 x1 = x4[1];
        float4 x2 = x4[2]; float4 x3 = x4[3];
        float4 x4_ = x4[4]; float4 x5 = x4[5];
        float4 x6 = x4[6]; float4 x7 = x4[7];

        acc += E2M1_LUT[(ws.x      ) & 0xFu] * s * x0.x;
        acc += E2M1_LUT[(ws.x >>  4) & 0xFu] * s * x0.y;
        acc += E2M1_LUT[(ws.x >>  8) & 0xFu] * s * x0.z;
        acc += E2M1_LUT[(ws.x >> 12) & 0xFu] * s * x0.w;
        acc += E2M1_LUT[(ws.x >> 16) & 0xFu] * s * x1.x;
        acc += E2M1_LUT[(ws.x >> 20) & 0xFu] * s * x1.y;
        acc += E2M1_LUT[(ws.x >> 24) & 0xFu] * s * x1.z;
        acc += E2M1_LUT[(ws.x >> 28) & 0xFu] * s * x1.w;

        acc += E2M1_LUT[(ws.y      ) & 0xFu] * s * x2.x;
        acc += E2M1_LUT[(ws.y >>  4) & 0xFu] * s * x2.y;
        acc += E2M1_LUT[(ws.y >>  8) & 0xFu] * s * x2.z;
        acc += E2M1_LUT[(ws.y >> 12) & 0xFu] * s * x2.w;
        acc += E2M1_LUT[(ws.y >> 16) & 0xFu] * s * x3.x;
        acc += E2M1_LUT[(ws.y >> 20) & 0xFu] * s * x3.y;
        acc += E2M1_LUT[(ws.y >> 24) & 0xFu] * s * x3.z;
        acc += E2M1_LUT[(ws.y >> 28) & 0xFu] * s * x3.w;

        acc += E2M1_LUT[(ws.z      ) & 0xFu] * s * x4_.x;
        acc += E2M1_LUT[(ws.z >>  4) & 0xFu] * s * x4_.y;
        acc += E2M1_LUT[(ws.z >>  8) & 0xFu] * s * x4_.z;
        acc += E2M1_LUT[(ws.z >> 12) & 0xFu] * s * x4_.w;
        acc += E2M1_LUT[(ws.z >> 16) & 0xFu] * s * x5.x;
        acc += E2M1_LUT[(ws.z >> 20) & 0xFu] * s * x5.y;
        acc += E2M1_LUT[(ws.z >> 24) & 0xFu] * s * x5.z;
        acc += E2M1_LUT[(ws.z >> 28) & 0xFu] * s * x5.w;

        acc += E2M1_LUT[(ws.w      ) & 0xFu] * s * x6.x;
        acc += E2M1_LUT[(ws.w >>  4) & 0xFu] * s * x6.y;
        acc += E2M1_LUT[(ws.w >>  8) & 0xFu] * s * x6.z;
        acc += E2M1_LUT[(ws.w >> 12) & 0xFu] * s * x6.w;
        acc += E2M1_LUT[(ws.w >> 16) & 0xFu] * s * x7.x;
        acc += E2M1_LUT[(ws.w >> 20) & 0xFu] * s * x7.y;
        acc += E2M1_LUT[(ws.w >> 24) & 0xFu] * s * x7.z;
        acc += E2M1_LUT[(ws.w >> 28) & 0xFu] * s * x7.w;
    }

    acc = simd_sum(acc);

    if (sg_lane == 0u) {
        y[b * dims.out_features + row] = acc;
    }
}

// ───────────────────────────────────────────────────────────────────────────
// mxfp4_matmul_moe_f32in_bf16out_v3 — Phase A.1.5 (2026-04-27): bf16-output
// sister of `mxfp4_matmul_moe_f32_v3`. Same expert-indices indirection, same
// inner FMA loop in f32; only the device-memory store narrows to `bfloat`.
// ───────────────────────────────────────────────────────────────────────────
kernel void mxfp4_matmul_moe_f32in_bf16out_v3(
    device const uint*     packed_all     [[buffer(0)]],
    device const uchar*    scales_all     [[buffer(1)]],
    device const uint*     expert_indices [[buffer(2)]],
    device const float*    x              [[buffer(3)]],
    device bfloat*         y              [[buffer(4)]],
    constant MxFp4MoeDims& dims           [[buffer(5)]],
    threadgroup float*     x_shared       [[threadgroup(0)]],
    uint3 tg_pos            [[threadgroup_position_in_grid]],
    uint  tid_in_tg         [[thread_index_in_threadgroup]],
    uint  sg_id             [[simdgroup_index_in_threadgroup]],
    uint  sg_lane           [[thread_index_in_simdgroup]]
) {
    uint b    = tg_pos.y;
    uint slot = tg_pos.z;
    if (b >= dims.batch) { return; }

    const uint ROWS_PER_TG = 8u;
    const uint THREADS_PER_TG = 256u;
    uint row = tg_pos.x * ROWS_PER_TG + sg_id;

    uint groups        = dims.in_features / 32u;
    uint words_per_row = dims.in_features / 8u;

    uint x_slot     = (dims.broadcast_x != 0u) ? 0u : slot;
    uint x_row_base = x_slot * dims.batch * dims.in_features
                    + b * dims.in_features;
    for (uint i = tid_in_tg; i < dims.in_features; i += THREADS_PER_TG) {
        x_shared[i] = x[x_row_base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= dims.out_features) { return; }

    uint e = expert_indices[slot];
    uint packed_expert_stride = dims.out_features * words_per_row;
    uint scale_expert_stride  = dims.out_features * groups;
    uint word_row_base  = e * packed_expert_stride + row * words_per_row;
    uint scale_row_base = e * scale_expert_stride  + row * groups;

    float acc = 0.0f;
    for (uint g = sg_lane; g < groups; g += 32u) {
        uchar scale_byte = scales_all[scale_row_base + g];
        if (scale_byte == 0xFFu) continue;
        uint sbits = uint(scale_byte) << 23;
        float s = as_type<float>(sbits);
        if (s == 0.0f) continue;

        uint word_base = word_row_base + g * 4u;
        uint x_base    = g * 32u;

        uint4 ws = *((device const uint4*)(packed_all + word_base));
        threadgroup const float4* x4 =
            (threadgroup const float4*)(x_shared + x_base);
        float4 x0 = x4[0]; float4 x1 = x4[1];
        float4 x2 = x4[2]; float4 x3 = x4[3];
        float4 x4_ = x4[4]; float4 x5 = x4[5];
        float4 x6 = x4[6]; float4 x7 = x4[7];

        acc += E2M1_LUT[(ws.x      ) & 0xFu] * s * x0.x;
        acc += E2M1_LUT[(ws.x >>  4) & 0xFu] * s * x0.y;
        acc += E2M1_LUT[(ws.x >>  8) & 0xFu] * s * x0.z;
        acc += E2M1_LUT[(ws.x >> 12) & 0xFu] * s * x0.w;
        acc += E2M1_LUT[(ws.x >> 16) & 0xFu] * s * x1.x;
        acc += E2M1_LUT[(ws.x >> 20) & 0xFu] * s * x1.y;
        acc += E2M1_LUT[(ws.x >> 24) & 0xFu] * s * x1.z;
        acc += E2M1_LUT[(ws.x >> 28) & 0xFu] * s * x1.w;

        acc += E2M1_LUT[(ws.y      ) & 0xFu] * s * x2.x;
        acc += E2M1_LUT[(ws.y >>  4) & 0xFu] * s * x2.y;
        acc += E2M1_LUT[(ws.y >>  8) & 0xFu] * s * x2.z;
        acc += E2M1_LUT[(ws.y >> 12) & 0xFu] * s * x2.w;
        acc += E2M1_LUT[(ws.y >> 16) & 0xFu] * s * x3.x;
        acc += E2M1_LUT[(ws.y >> 20) & 0xFu] * s * x3.y;
        acc += E2M1_LUT[(ws.y >> 24) & 0xFu] * s * x3.z;
        acc += E2M1_LUT[(ws.y >> 28) & 0xFu] * s * x3.w;

        acc += E2M1_LUT[(ws.z      ) & 0xFu] * s * x4_.x;
        acc += E2M1_LUT[(ws.z >>  4) & 0xFu] * s * x4_.y;
        acc += E2M1_LUT[(ws.z >>  8) & 0xFu] * s * x4_.z;
        acc += E2M1_LUT[(ws.z >> 12) & 0xFu] * s * x4_.w;
        acc += E2M1_LUT[(ws.z >> 16) & 0xFu] * s * x5.x;
        acc += E2M1_LUT[(ws.z >> 20) & 0xFu] * s * x5.y;
        acc += E2M1_LUT[(ws.z >> 24) & 0xFu] * s * x5.z;
        acc += E2M1_LUT[(ws.z >> 28) & 0xFu] * s * x5.w;

        acc += E2M1_LUT[(ws.w      ) & 0xFu] * s * x6.x;
        acc += E2M1_LUT[(ws.w >>  4) & 0xFu] * s * x6.y;
        acc += E2M1_LUT[(ws.w >>  8) & 0xFu] * s * x6.z;
        acc += E2M1_LUT[(ws.w >> 12) & 0xFu] * s * x6.w;
        acc += E2M1_LUT[(ws.w >> 16) & 0xFu] * s * x7.x;
        acc += E2M1_LUT[(ws.w >> 20) & 0xFu] * s * x7.y;
        acc += E2M1_LUT[(ws.w >> 24) & 0xFu] * s * x7.z;
        acc += E2M1_LUT[(ws.w >> 28) & 0xFu] * s * x7.w;
    }

    acc = simd_sum(acc);
    if (sg_lane == 0u) {
        y[slot * dims.batch * dims.out_features + b * dims.out_features + row] = bfloat(acc);
    }
}

// ───────────────────────────────────────────────────────────────────────────
// mxfp4_gate_up_silu_mul_f32_v3 — Fused gate+up matmul with SwiGLU activation.
//
// Replaces 3 separate ops in SharedExpert::forward:
//   1. mxfp4_matmul_f32_v3 with W=[2*inter, hidden] → [batch, 2*inter]
//   2. element-wise silu(gate)         (Candle generic kernel)
//   3. element-wise mul gate * up      (Candle generic kernel)
//
// Output: [batch, inter] directly (half the writes; no 2*inter intermediate).
//
// Each simdgroup handles one fused output row r ∈ [0, inter):
//   acc_gate = dot(W[r,         :], x[b, :])
//   acc_up   = dot(W[r + inter, :], x[b, :])
//   y[b, r]  = silu(acc_gate) * acc_up
// Lanes share x_shared cache (same as v3); two accumulators per lane.
//
// Constraints (same as v3):
//   - in_features * 4 ≤ 32 KB threadgroup memory
//   - 256 threads/TG = 8 simdgroups × 32 lanes, 8 fused rows / TG
// ───────────────────────────────────────────────────────────────────────────

struct MxFp4GateUpSiluMulDims {
    uint inter;
    uint in_features;
    uint batch;
};

kernel void mxfp4_gate_up_silu_mul_f32_v3(
    device const uint*    packed   [[buffer(0)]],   // [2*inter, in/8]
    device const uchar*   scales   [[buffer(1)]],   // [2*inter, in/32]
    device const float*   x        [[buffer(2)]],   // [batch, in]
    device float*         y        [[buffer(3)]],   // [batch, inter]
    constant MxFp4GateUpSiluMulDims& dims [[buffer(4)]],
    threadgroup float*    x_shared [[threadgroup(0)]],
    uint3 tg_pos          [[threadgroup_position_in_grid]],
    uint  tid_in_tg       [[thread_index_in_threadgroup]],
    uint  sg_id           [[simdgroup_index_in_threadgroup]],
    uint  sg_lane         [[thread_index_in_simdgroup]]
) {
    uint b = tg_pos.y;
    if (b >= dims.batch) { return; }

    const uint ROWS_PER_TG = 8u;
    const uint THREADS_PER_TG = 256u;
    uint row = tg_pos.x * ROWS_PER_TG + sg_id;

    uint groups        = dims.in_features / 32u;
    uint words_per_row = dims.in_features / 8u;
    uint x_row_base    = b * dims.in_features;

    for (uint i = tid_in_tg; i < dims.in_features; i += THREADS_PER_TG) {
        x_shared[i] = x[x_row_base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= dims.inter) { return; }

    uint gate_word_base  = row * words_per_row;
    uint gate_scale_base = row * groups;
    uint up_word_base    = (row + dims.inter) * words_per_row;
    uint up_scale_base   = (row + dims.inter) * groups;

    float acc_gate = 0.0f;
    float acc_up   = 0.0f;

    for (uint g = sg_lane; g < groups; g += 32u) {
        uchar sg_byte = scales[gate_scale_base + g];
        uchar su_byte = scales[up_scale_base + g];

        float s_gate = 0.0f;
        float s_up   = 0.0f;
        if (sg_byte != 0xFFu) {
            uint sb = uint(sg_byte) << 23;
            s_gate = as_type<float>(sb);
        }
        if (su_byte != 0xFFu) {
            uint sb = uint(su_byte) << 23;
            s_up = as_type<float>(sb);
        }
        if (s_gate == 0.0f && s_up == 0.0f) { continue; }

        uint x_base = g * 32u;
        threadgroup const float4* x4 =
            (threadgroup const float4*)(x_shared + x_base);
        float4 x0 = x4[0]; float4 x1 = x4[1];
        float4 x2 = x4[2]; float4 x3 = x4[3];
        float4 x4_ = x4[4]; float4 x5 = x4[5];
        float4 x6 = x4[6]; float4 x7 = x4[7];

        if (s_gate != 0.0f) {
            uint4 ws = *((device const uint4*)(packed + gate_word_base + g * 4u));
            acc_gate += E2M1_LUT[(ws.x      ) & 0xFu] * s_gate * x0.x;
            acc_gate += E2M1_LUT[(ws.x >>  4) & 0xFu] * s_gate * x0.y;
            acc_gate += E2M1_LUT[(ws.x >>  8) & 0xFu] * s_gate * x0.z;
            acc_gate += E2M1_LUT[(ws.x >> 12) & 0xFu] * s_gate * x0.w;
            acc_gate += E2M1_LUT[(ws.x >> 16) & 0xFu] * s_gate * x1.x;
            acc_gate += E2M1_LUT[(ws.x >> 20) & 0xFu] * s_gate * x1.y;
            acc_gate += E2M1_LUT[(ws.x >> 24) & 0xFu] * s_gate * x1.z;
            acc_gate += E2M1_LUT[(ws.x >> 28) & 0xFu] * s_gate * x1.w;

            acc_gate += E2M1_LUT[(ws.y      ) & 0xFu] * s_gate * x2.x;
            acc_gate += E2M1_LUT[(ws.y >>  4) & 0xFu] * s_gate * x2.y;
            acc_gate += E2M1_LUT[(ws.y >>  8) & 0xFu] * s_gate * x2.z;
            acc_gate += E2M1_LUT[(ws.y >> 12) & 0xFu] * s_gate * x2.w;
            acc_gate += E2M1_LUT[(ws.y >> 16) & 0xFu] * s_gate * x3.x;
            acc_gate += E2M1_LUT[(ws.y >> 20) & 0xFu] * s_gate * x3.y;
            acc_gate += E2M1_LUT[(ws.y >> 24) & 0xFu] * s_gate * x3.z;
            acc_gate += E2M1_LUT[(ws.y >> 28) & 0xFu] * s_gate * x3.w;

            acc_gate += E2M1_LUT[(ws.z      ) & 0xFu] * s_gate * x4_.x;
            acc_gate += E2M1_LUT[(ws.z >>  4) & 0xFu] * s_gate * x4_.y;
            acc_gate += E2M1_LUT[(ws.z >>  8) & 0xFu] * s_gate * x4_.z;
            acc_gate += E2M1_LUT[(ws.z >> 12) & 0xFu] * s_gate * x4_.w;
            acc_gate += E2M1_LUT[(ws.z >> 16) & 0xFu] * s_gate * x5.x;
            acc_gate += E2M1_LUT[(ws.z >> 20) & 0xFu] * s_gate * x5.y;
            acc_gate += E2M1_LUT[(ws.z >> 24) & 0xFu] * s_gate * x5.z;
            acc_gate += E2M1_LUT[(ws.z >> 28) & 0xFu] * s_gate * x5.w;

            acc_gate += E2M1_LUT[(ws.w      ) & 0xFu] * s_gate * x6.x;
            acc_gate += E2M1_LUT[(ws.w >>  4) & 0xFu] * s_gate * x6.y;
            acc_gate += E2M1_LUT[(ws.w >>  8) & 0xFu] * s_gate * x6.z;
            acc_gate += E2M1_LUT[(ws.w >> 12) & 0xFu] * s_gate * x6.w;
            acc_gate += E2M1_LUT[(ws.w >> 16) & 0xFu] * s_gate * x7.x;
            acc_gate += E2M1_LUT[(ws.w >> 20) & 0xFu] * s_gate * x7.y;
            acc_gate += E2M1_LUT[(ws.w >> 24) & 0xFu] * s_gate * x7.z;
            acc_gate += E2M1_LUT[(ws.w >> 28) & 0xFu] * s_gate * x7.w;
        }

        if (s_up != 0.0f) {
            uint4 ws = *((device const uint4*)(packed + up_word_base + g * 4u));
            acc_up += E2M1_LUT[(ws.x      ) & 0xFu] * s_up * x0.x;
            acc_up += E2M1_LUT[(ws.x >>  4) & 0xFu] * s_up * x0.y;
            acc_up += E2M1_LUT[(ws.x >>  8) & 0xFu] * s_up * x0.z;
            acc_up += E2M1_LUT[(ws.x >> 12) & 0xFu] * s_up * x0.w;
            acc_up += E2M1_LUT[(ws.x >> 16) & 0xFu] * s_up * x1.x;
            acc_up += E2M1_LUT[(ws.x >> 20) & 0xFu] * s_up * x1.y;
            acc_up += E2M1_LUT[(ws.x >> 24) & 0xFu] * s_up * x1.z;
            acc_up += E2M1_LUT[(ws.x >> 28) & 0xFu] * s_up * x1.w;

            acc_up += E2M1_LUT[(ws.y      ) & 0xFu] * s_up * x2.x;
            acc_up += E2M1_LUT[(ws.y >>  4) & 0xFu] * s_up * x2.y;
            acc_up += E2M1_LUT[(ws.y >>  8) & 0xFu] * s_up * x2.z;
            acc_up += E2M1_LUT[(ws.y >> 12) & 0xFu] * s_up * x2.w;
            acc_up += E2M1_LUT[(ws.y >> 16) & 0xFu] * s_up * x3.x;
            acc_up += E2M1_LUT[(ws.y >> 20) & 0xFu] * s_up * x3.y;
            acc_up += E2M1_LUT[(ws.y >> 24) & 0xFu] * s_up * x3.z;
            acc_up += E2M1_LUT[(ws.y >> 28) & 0xFu] * s_up * x3.w;

            acc_up += E2M1_LUT[(ws.z      ) & 0xFu] * s_up * x4_.x;
            acc_up += E2M1_LUT[(ws.z >>  4) & 0xFu] * s_up * x4_.y;
            acc_up += E2M1_LUT[(ws.z >>  8) & 0xFu] * s_up * x4_.z;
            acc_up += E2M1_LUT[(ws.z >> 12) & 0xFu] * s_up * x4_.w;
            acc_up += E2M1_LUT[(ws.z >> 16) & 0xFu] * s_up * x5.x;
            acc_up += E2M1_LUT[(ws.z >> 20) & 0xFu] * s_up * x5.y;
            acc_up += E2M1_LUT[(ws.z >> 24) & 0xFu] * s_up * x5.z;
            acc_up += E2M1_LUT[(ws.z >> 28) & 0xFu] * s_up * x5.w;

            acc_up += E2M1_LUT[(ws.w      ) & 0xFu] * s_up * x6.x;
            acc_up += E2M1_LUT[(ws.w >>  4) & 0xFu] * s_up * x6.y;
            acc_up += E2M1_LUT[(ws.w >>  8) & 0xFu] * s_up * x6.z;
            acc_up += E2M1_LUT[(ws.w >> 12) & 0xFu] * s_up * x6.w;
            acc_up += E2M1_LUT[(ws.w >> 16) & 0xFu] * s_up * x7.x;
            acc_up += E2M1_LUT[(ws.w >> 20) & 0xFu] * s_up * x7.y;
            acc_up += E2M1_LUT[(ws.w >> 24) & 0xFu] * s_up * x7.z;
            acc_up += E2M1_LUT[(ws.w >> 28) & 0xFu] * s_up * x7.w;
        }
    }

    acc_gate = simd_sum(acc_gate);
    acc_up   = simd_sum(acc_up);

    if (sg_lane == 0u) {
        float silu_g = acc_gate / (1.0f + metal::exp(-acc_gate));
        y[b * dims.inter + row] = silu_g * acc_up;
    }
}

// ───────────────────────────────────────────────────────────────────────────
// mxfp4_gate_up_silu_mul_f32in_bf16out_v3 — Phase A.1.5 (2026-04-27): bf16
// output sister. Same fused gate+up matmul + SiLU(gate)*up math, store narrow.
// ───────────────────────────────────────────────────────────────────────────
kernel void mxfp4_gate_up_silu_mul_f32in_bf16out_v3(
    device const uint*    packed   [[buffer(0)]],
    device const uchar*   scales   [[buffer(1)]],
    device const float*   x        [[buffer(2)]],
    device bfloat*        y        [[buffer(3)]],
    constant MxFp4GateUpSiluMulDims& dims [[buffer(4)]],
    threadgroup float*    x_shared [[threadgroup(0)]],
    uint3 tg_pos          [[threadgroup_position_in_grid]],
    uint  tid_in_tg       [[thread_index_in_threadgroup]],
    uint  sg_id           [[simdgroup_index_in_threadgroup]],
    uint  sg_lane         [[thread_index_in_simdgroup]]
) {
    uint b = tg_pos.y;
    if (b >= dims.batch) { return; }

    const uint ROWS_PER_TG = 8u;
    const uint THREADS_PER_TG = 256u;
    uint row = tg_pos.x * ROWS_PER_TG + sg_id;

    uint groups        = dims.in_features / 32u;
    uint words_per_row = dims.in_features / 8u;
    uint x_row_base    = b * dims.in_features;

    for (uint i = tid_in_tg; i < dims.in_features; i += THREADS_PER_TG) {
        x_shared[i] = x[x_row_base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= dims.inter) { return; }

    uint gate_word_base  = row * words_per_row;
    uint gate_scale_base = row * groups;
    uint up_word_base    = (row + dims.inter) * words_per_row;
    uint up_scale_base   = (row + dims.inter) * groups;

    float acc_gate = 0.0f;
    float acc_up   = 0.0f;

    for (uint g = sg_lane; g < groups; g += 32u) {
        uchar sg_byte = scales[gate_scale_base + g];
        uchar su_byte = scales[up_scale_base + g];

        float s_gate = 0.0f;
        float s_up   = 0.0f;
        if (sg_byte != 0xFFu) {
            uint sb = uint(sg_byte) << 23;
            s_gate = as_type<float>(sb);
        }
        if (su_byte != 0xFFu) {
            uint sb = uint(su_byte) << 23;
            s_up = as_type<float>(sb);
        }
        if (s_gate == 0.0f && s_up == 0.0f) { continue; }

        uint x_base = g * 32u;
        threadgroup const float4* x4 =
            (threadgroup const float4*)(x_shared + x_base);
        float4 x0 = x4[0]; float4 x1 = x4[1];
        float4 x2 = x4[2]; float4 x3 = x4[3];
        float4 x4_ = x4[4]; float4 x5 = x4[5];
        float4 x6 = x4[6]; float4 x7 = x4[7];

        if (s_gate != 0.0f) {
            uint4 ws = *((device const uint4*)(packed + gate_word_base + g * 4u));
            acc_gate += E2M1_LUT[(ws.x      ) & 0xFu] * s_gate * x0.x;
            acc_gate += E2M1_LUT[(ws.x >>  4) & 0xFu] * s_gate * x0.y;
            acc_gate += E2M1_LUT[(ws.x >>  8) & 0xFu] * s_gate * x0.z;
            acc_gate += E2M1_LUT[(ws.x >> 12) & 0xFu] * s_gate * x0.w;
            acc_gate += E2M1_LUT[(ws.x >> 16) & 0xFu] * s_gate * x1.x;
            acc_gate += E2M1_LUT[(ws.x >> 20) & 0xFu] * s_gate * x1.y;
            acc_gate += E2M1_LUT[(ws.x >> 24) & 0xFu] * s_gate * x1.z;
            acc_gate += E2M1_LUT[(ws.x >> 28) & 0xFu] * s_gate * x1.w;

            acc_gate += E2M1_LUT[(ws.y      ) & 0xFu] * s_gate * x2.x;
            acc_gate += E2M1_LUT[(ws.y >>  4) & 0xFu] * s_gate * x2.y;
            acc_gate += E2M1_LUT[(ws.y >>  8) & 0xFu] * s_gate * x2.z;
            acc_gate += E2M1_LUT[(ws.y >> 12) & 0xFu] * s_gate * x2.w;
            acc_gate += E2M1_LUT[(ws.y >> 16) & 0xFu] * s_gate * x3.x;
            acc_gate += E2M1_LUT[(ws.y >> 20) & 0xFu] * s_gate * x3.y;
            acc_gate += E2M1_LUT[(ws.y >> 24) & 0xFu] * s_gate * x3.z;
            acc_gate += E2M1_LUT[(ws.y >> 28) & 0xFu] * s_gate * x3.w;

            acc_gate += E2M1_LUT[(ws.z      ) & 0xFu] * s_gate * x4_.x;
            acc_gate += E2M1_LUT[(ws.z >>  4) & 0xFu] * s_gate * x4_.y;
            acc_gate += E2M1_LUT[(ws.z >>  8) & 0xFu] * s_gate * x4_.z;
            acc_gate += E2M1_LUT[(ws.z >> 12) & 0xFu] * s_gate * x4_.w;
            acc_gate += E2M1_LUT[(ws.z >> 16) & 0xFu] * s_gate * x5.x;
            acc_gate += E2M1_LUT[(ws.z >> 20) & 0xFu] * s_gate * x5.y;
            acc_gate += E2M1_LUT[(ws.z >> 24) & 0xFu] * s_gate * x5.z;
            acc_gate += E2M1_LUT[(ws.z >> 28) & 0xFu] * s_gate * x5.w;

            acc_gate += E2M1_LUT[(ws.w      ) & 0xFu] * s_gate * x6.x;
            acc_gate += E2M1_LUT[(ws.w >>  4) & 0xFu] * s_gate * x6.y;
            acc_gate += E2M1_LUT[(ws.w >>  8) & 0xFu] * s_gate * x6.z;
            acc_gate += E2M1_LUT[(ws.w >> 12) & 0xFu] * s_gate * x6.w;
            acc_gate += E2M1_LUT[(ws.w >> 16) & 0xFu] * s_gate * x7.x;
            acc_gate += E2M1_LUT[(ws.w >> 20) & 0xFu] * s_gate * x7.y;
            acc_gate += E2M1_LUT[(ws.w >> 24) & 0xFu] * s_gate * x7.z;
            acc_gate += E2M1_LUT[(ws.w >> 28) & 0xFu] * s_gate * x7.w;
        }

        if (s_up != 0.0f) {
            uint4 ws = *((device const uint4*)(packed + up_word_base + g * 4u));
            acc_up += E2M1_LUT[(ws.x      ) & 0xFu] * s_up * x0.x;
            acc_up += E2M1_LUT[(ws.x >>  4) & 0xFu] * s_up * x0.y;
            acc_up += E2M1_LUT[(ws.x >>  8) & 0xFu] * s_up * x0.z;
            acc_up += E2M1_LUT[(ws.x >> 12) & 0xFu] * s_up * x0.w;
            acc_up += E2M1_LUT[(ws.x >> 16) & 0xFu] * s_up * x1.x;
            acc_up += E2M1_LUT[(ws.x >> 20) & 0xFu] * s_up * x1.y;
            acc_up += E2M1_LUT[(ws.x >> 24) & 0xFu] * s_up * x1.z;
            acc_up += E2M1_LUT[(ws.x >> 28) & 0xFu] * s_up * x1.w;

            acc_up += E2M1_LUT[(ws.y      ) & 0xFu] * s_up * x2.x;
            acc_up += E2M1_LUT[(ws.y >>  4) & 0xFu] * s_up * x2.y;
            acc_up += E2M1_LUT[(ws.y >>  8) & 0xFu] * s_up * x2.z;
            acc_up += E2M1_LUT[(ws.y >> 12) & 0xFu] * s_up * x2.w;
            acc_up += E2M1_LUT[(ws.y >> 16) & 0xFu] * s_up * x3.x;
            acc_up += E2M1_LUT[(ws.y >> 20) & 0xFu] * s_up * x3.y;
            acc_up += E2M1_LUT[(ws.y >> 24) & 0xFu] * s_up * x3.z;
            acc_up += E2M1_LUT[(ws.y >> 28) & 0xFu] * s_up * x3.w;

            acc_up += E2M1_LUT[(ws.z      ) & 0xFu] * s_up * x4_.x;
            acc_up += E2M1_LUT[(ws.z >>  4) & 0xFu] * s_up * x4_.y;
            acc_up += E2M1_LUT[(ws.z >>  8) & 0xFu] * s_up * x4_.z;
            acc_up += E2M1_LUT[(ws.z >> 12) & 0xFu] * s_up * x4_.w;
            acc_up += E2M1_LUT[(ws.z >> 16) & 0xFu] * s_up * x5.x;
            acc_up += E2M1_LUT[(ws.z >> 20) & 0xFu] * s_up * x5.y;
            acc_up += E2M1_LUT[(ws.z >> 24) & 0xFu] * s_up * x5.z;
            acc_up += E2M1_LUT[(ws.z >> 28) & 0xFu] * s_up * x5.w;

            acc_up += E2M1_LUT[(ws.w      ) & 0xFu] * s_up * x6.x;
            acc_up += E2M1_LUT[(ws.w >>  4) & 0xFu] * s_up * x6.y;
            acc_up += E2M1_LUT[(ws.w >>  8) & 0xFu] * s_up * x6.z;
            acc_up += E2M1_LUT[(ws.w >> 12) & 0xFu] * s_up * x6.w;
            acc_up += E2M1_LUT[(ws.w >> 16) & 0xFu] * s_up * x7.x;
            acc_up += E2M1_LUT[(ws.w >> 20) & 0xFu] * s_up * x7.y;
            acc_up += E2M1_LUT[(ws.w >> 24) & 0xFu] * s_up * x7.z;
            acc_up += E2M1_LUT[(ws.w >> 28) & 0xFu] * s_up * x7.w;
        }
    }

    acc_gate = simd_sum(acc_gate);
    acc_up   = simd_sum(acc_up);

    if (sg_lane == 0u) {
        float silu_g = acc_gate / (1.0f + metal::exp(-acc_gate));
        y[b * dims.inter + row] = bfloat(silu_g * acc_up);
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Small-out fast kernel (Phase M.2-A, 2026-04-26)
//
// Targets shapes where v3's `n_groups_x = ceil(out/8)` produces too few
// threadgroups for Apple GPU latency hiding. Concretely, the routing gate
// (out=256, in=2048) and any other small-out matmul: v3 schedules only
// 32 TGs over ~30 GPU cores → 1.07 TG/core, well below the 4-8 in-flight
// simdgroups/core Apple needs to hide ALU + memory latency.
//
// **Topology (1 TG = 1 row, 256 threads cooperate on the reduction):**
//   - Grid = (out_features, batch). 1 threadgroup per output element.
//   - 256 threads/TG = 8 simdgroups × 32 lanes.
//   - Each thread strides the row's `words_per_row` packed words (one word =
//     8 nibbles = 8 elements). For in=2048 → 256 words → 1 word/thread.
//   - Per-simdgroup reduce via `simd_sum`, then 8 partials → 1 via threadgroup
//     memory + a single `simd_sum` on simdgroup 0.
//
// **Why this beats v3 for small out:** Same total ALU work, but ~8× more
// TGs → enough in-flight simdgroups to mask global-memory latency. M.1's
// regression came from halving TGs (occupancy collapse); this kernel runs
// the inverse trade — more TGs at the cost of more cross-simdgroup reduction
// — which is the right side of the trade for tiny `out_features`.
//
// **Shape constraints:**
//   - in_features must be a multiple of 32 (MXFP4 group size)
//   - out_features and batch may be any positive value
//   - words_per_row = in_features / 8 may exceed 256; threads stride.
//
// **Memory:** No threadgroup x cache — small out + small batch means the
// row's x column gets read once across the TG. The HW L1 already coalesces
// the 256-thread broadcast access pattern, so an explicit cache costs more
// (load + barrier) than it saves.
// ───────────────────────────────────────────────────────────────────────────

kernel void mxfp4_matmul_small_out_f32_v1(
    device const uint*   packed     [[buffer(0)]],
    device const uchar*  scales     [[buffer(1)]],
    device const float*  x          [[buffer(2)]],
    device float*        y          [[buffer(3)]],
    constant MxFp4Dims&  dims       [[buffer(4)]],
    constant uint&       batch      [[buffer(5)]],
    threadgroup float*   sg_partial [[threadgroup(0)]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]],
    uint  tid_in_tg                 [[thread_index_in_threadgroup]],
    uint  sg_id                     [[simdgroup_index_in_threadgroup]],
    uint  sg_lane                   [[thread_index_in_simdgroup]]
) {
    uint row = tg_pos.x;
    uint b   = tg_pos.y;
    if (row >= dims.out_features || b >= batch) { return; }

    const uint THREADS_PER_TG = 256u;
    uint groups         = dims.in_features / 32u;
    uint words_per_row  = dims.in_features / 8u;
    uint scale_row_base = row * groups;
    uint word_row_base  = row * words_per_row;
    uint x_row_base     = b * dims.in_features;

    float acc = 0.0f;
    for (uint w = tid_in_tg; w < words_per_row; w += THREADS_PER_TG) {
        uint g      = w / 4u;
        uint w_in_g = w & 3u;
        uchar scale_byte = scales[scale_row_base + g];
        if (scale_byte == 0xFFu) { continue; }
        uint sbits = uint(scale_byte) << 23;
        float s = as_type<float>(sbits);
        if (s == 0.0f) { continue; }

        uint word   = packed[word_row_base + w];
        uint x_base = x_row_base + g * 32u + w_in_g * 8u;
        acc += E2M1_LUT[(word      ) & 0xFu] * s * x[x_base + 0u];
        acc += E2M1_LUT[(word >>  4) & 0xFu] * s * x[x_base + 1u];
        acc += E2M1_LUT[(word >>  8) & 0xFu] * s * x[x_base + 2u];
        acc += E2M1_LUT[(word >> 12) & 0xFu] * s * x[x_base + 3u];
        acc += E2M1_LUT[(word >> 16) & 0xFu] * s * x[x_base + 4u];
        acc += E2M1_LUT[(word >> 20) & 0xFu] * s * x[x_base + 5u];
        acc += E2M1_LUT[(word >> 24) & 0xFu] * s * x[x_base + 6u];
        acc += E2M1_LUT[(word >> 28) & 0xFu] * s * x[x_base + 7u];
    }

    acc = simd_sum(acc);
    if (sg_lane == 0u) { sg_partial[sg_id] = acc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_id == 0u) {
        float v = (sg_lane < 8u) ? sg_partial[sg_lane] : 0.0f;
        v = simd_sum(v);
        if (sg_lane == 0u) {
            y[b * dims.out_features + row] = v;
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// mxfp4_matmul_small_out_f32in_bf16out_v1 — Phase A.1.5 (2026-04-27): bf16
// output sister of `mxfp4_matmul_small_out_f32_v1`. Same 1-TG-per-row topology
// and 8-simdgroup partial reduction; only the device store narrows.
// ───────────────────────────────────────────────────────────────────────────
kernel void mxfp4_matmul_small_out_f32in_bf16out_v1(
    device const uint*   packed     [[buffer(0)]],
    device const uchar*  scales     [[buffer(1)]],
    device const float*  x          [[buffer(2)]],
    device bfloat*       y          [[buffer(3)]],
    constant MxFp4Dims&  dims       [[buffer(4)]],
    constant uint&       batch      [[buffer(5)]],
    threadgroup float*   sg_partial [[threadgroup(0)]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]],
    uint  tid_in_tg                 [[thread_index_in_threadgroup]],
    uint  sg_id                     [[simdgroup_index_in_threadgroup]],
    uint  sg_lane                   [[thread_index_in_simdgroup]]
) {
    uint row = tg_pos.x;
    uint b   = tg_pos.y;
    if (row >= dims.out_features || b >= batch) { return; }

    const uint THREADS_PER_TG = 256u;
    uint groups         = dims.in_features / 32u;
    uint words_per_row  = dims.in_features / 8u;
    uint scale_row_base = row * groups;
    uint word_row_base  = row * words_per_row;
    uint x_row_base     = b * dims.in_features;

    float acc = 0.0f;
    for (uint w = tid_in_tg; w < words_per_row; w += THREADS_PER_TG) {
        uint g      = w / 4u;
        uint w_in_g = w & 3u;
        uchar scale_byte = scales[scale_row_base + g];
        if (scale_byte == 0xFFu) { continue; }
        uint sbits = uint(scale_byte) << 23;
        float s = as_type<float>(sbits);
        if (s == 0.0f) { continue; }

        uint word   = packed[word_row_base + w];
        uint x_base = x_row_base + g * 32u + w_in_g * 8u;
        acc += E2M1_LUT[(word      ) & 0xFu] * s * x[x_base + 0u];
        acc += E2M1_LUT[(word >>  4) & 0xFu] * s * x[x_base + 1u];
        acc += E2M1_LUT[(word >>  8) & 0xFu] * s * x[x_base + 2u];
        acc += E2M1_LUT[(word >> 12) & 0xFu] * s * x[x_base + 3u];
        acc += E2M1_LUT[(word >> 16) & 0xFu] * s * x[x_base + 4u];
        acc += E2M1_LUT[(word >> 20) & 0xFu] * s * x[x_base + 5u];
        acc += E2M1_LUT[(word >> 24) & 0xFu] * s * x[x_base + 6u];
        acc += E2M1_LUT[(word >> 28) & 0xFu] * s * x[x_base + 7u];
    }

    acc = simd_sum(acc);
    if (sg_lane == 0u) { sg_partial[sg_id] = acc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_id == 0u) {
        float v = (sg_lane < 8u) ? sg_partial[sg_lane] : 0.0f;
        v = simd_sum(v);
        if (sg_lane == 0u) {
            y[b * dims.out_features + row] = bfloat(v);
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Lever B (2026-04-27) — moe_wsum_f32: weighted sum of expert outputs.
//
// Replaces the `downs.broadcast_mul(weights).sum_keepdim(0)` chain in
// MoE forward (moe.rs ~line 800). 2 Candle kernels + intermediate
// `weighted [k, hidden]` device tensor → 1 fused kernel writing
// `out[r] = sum_e w[e] * downs[e, r]` directly.
//
// Topology: 1D grid over `hidden`. 256 threads/TG, 1 thread = 1 output column.
// k is small (typically 8 for top-k MoE), so the inner reduction is a tight
// loop. Weights are read from device on every thread but the L1 cache
// coalesces the broadcast across the warp.
// ───────────────────────────────────────────────────────────────────────────
struct MoeWsumDims {
    uint k;
    uint hidden;
};

kernel void moe_wsum_f32(
    device const float* downs   [[buffer(0)]],   // [k, hidden]
    device const float* weights [[buffer(1)]],   // [k]
    device float*       out     [[buffer(2)]],   // [hidden]
    constant MoeWsumDims& dims  [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dims.hidden) { return; }
    float acc = 0.0f;
    for (uint e = 0; e < dims.k; e++) {
        acc += weights[e] * downs[e * dims.hidden + tid];
    }
    out[tid] = acc;
}

// ───────────────────────────────────────────────────────────────────────────
// Lever C (2026-04-27) — mxfp4_matmul_moe_wsum_f32_v3: fused down matmul +
// weighted sum.
//
// Replaces the chain
//   downs[slot, b, hr] = sum_m down[expert[slot], hr, m] * hiddens[slot, b, m]
//   out[b, hr]         = sum_slot weight[slot] * downs[slot, b, hr]
// with the slot reduction folded into the dot loop:
//   out[b, hr] = sum_slot weight[slot]
//                * sum_m down[expert[slot], hr, m] * hiddens[slot, b, m]
//
// Eliminates the `downs_big` intermediate (k × batch × hidden f32) and the
// separate `moe_wsum_f32` kernel launch + its sync. The slot axis moves
// from grid.z into an inner serial loop so that each TG sums over all k
// experts before writing once to `y` (no atomics, no contention).
//
// Topology:
//   Grid = (out_features / ROWS_PER_TG, batch, 1)   ROWS_PER_TG = 8
//   TG   = 256 threads = 8 SG × 32 lanes
//   TG-shared: hiddens slab `x_shared[in_features]` (re-staged per slot)
// ───────────────────────────────────────────────────────────────────────────
struct MxFp4MoeMatmulWsumDims {
    uint out_features;  // hidden_size (down out)
    uint in_features;   // moe_inter (down in)
    uint batch;         // typically 1 in decode
    uint k;             // top-k experts
};

kernel void mxfp4_matmul_moe_wsum_f32_v3(
    device const uint*     packed_all     [[buffer(0)]],
    device const uchar*    scales_all     [[buffer(1)]],
    device const uint*     expert_indices [[buffer(2)]],
    device const float*    weights        [[buffer(3)]],
    device const float*    x              [[buffer(4)]],
    device float*          y              [[buffer(5)]],
    constant MxFp4MoeMatmulWsumDims& dims [[buffer(6)]],
    threadgroup float*     x_shared       [[threadgroup(0)]],
    uint3 tg_pos            [[threadgroup_position_in_grid]],
    uint  tid_in_tg         [[thread_index_in_threadgroup]],
    uint  sg_id             [[simdgroup_index_in_threadgroup]],
    uint  sg_lane           [[thread_index_in_simdgroup]]
) {
    uint b = tg_pos.y;
    if (b >= dims.batch) { return; }

    const uint ROWS_PER_TG = 8u;
    const uint THREADS_PER_TG = 256u;
    uint row = tg_pos.x * ROWS_PER_TG + sg_id;

    uint groups        = dims.in_features / 32u;
    uint words_per_row = dims.in_features / 8u;
    uint packed_expert_stride = dims.out_features * words_per_row;
    uint scale_expert_stride  = dims.out_features * groups;

    float row_acc = 0.0f;

    for (uint slot = 0u; slot < dims.k; slot++) {
        uint h_row_base = slot * dims.batch * dims.in_features
                        + b * dims.in_features;
        for (uint i = tid_in_tg; i < dims.in_features; i += THREADS_PER_TG) {
            x_shared[i] = x[h_row_base + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row < dims.out_features) {
            uint e = expert_indices[slot];
            uint word_row_base  = e * packed_expert_stride + row * words_per_row;
            uint scale_row_base = e * scale_expert_stride  + row * groups;

            float acc = 0.0f;
            for (uint g = sg_lane; g < groups; g += 32u) {
                uchar scale_byte = scales_all[scale_row_base + g];
                if (scale_byte == 0xFFu) continue;
                uint sbits = uint(scale_byte) << 23;
                float s = as_type<float>(sbits);
                if (s == 0.0f) continue;

                uint word_base = word_row_base + g * 4u;
                uint x_base    = g * 32u;

                uint4 ws = *((device const uint4*)(packed_all + word_base));
                threadgroup const float4* x4 =
                    (threadgroup const float4*)(x_shared + x_base);
                float4 x0 = x4[0]; float4 x1 = x4[1];
                float4 x2 = x4[2]; float4 x3 = x4[3];
                float4 x4_ = x4[4]; float4 x5 = x4[5];
                float4 x6 = x4[6]; float4 x7 = x4[7];

                acc += E2M1_LUT[(ws.x      ) & 0xFu] * s * x0.x;
                acc += E2M1_LUT[(ws.x >>  4) & 0xFu] * s * x0.y;
                acc += E2M1_LUT[(ws.x >>  8) & 0xFu] * s * x0.z;
                acc += E2M1_LUT[(ws.x >> 12) & 0xFu] * s * x0.w;
                acc += E2M1_LUT[(ws.x >> 16) & 0xFu] * s * x1.x;
                acc += E2M1_LUT[(ws.x >> 20) & 0xFu] * s * x1.y;
                acc += E2M1_LUT[(ws.x >> 24) & 0xFu] * s * x1.z;
                acc += E2M1_LUT[(ws.x >> 28) & 0xFu] * s * x1.w;

                acc += E2M1_LUT[(ws.y      ) & 0xFu] * s * x2.x;
                acc += E2M1_LUT[(ws.y >>  4) & 0xFu] * s * x2.y;
                acc += E2M1_LUT[(ws.y >>  8) & 0xFu] * s * x2.z;
                acc += E2M1_LUT[(ws.y >> 12) & 0xFu] * s * x2.w;
                acc += E2M1_LUT[(ws.y >> 16) & 0xFu] * s * x3.x;
                acc += E2M1_LUT[(ws.y >> 20) & 0xFu] * s * x3.y;
                acc += E2M1_LUT[(ws.y >> 24) & 0xFu] * s * x3.z;
                acc += E2M1_LUT[(ws.y >> 28) & 0xFu] * s * x3.w;

                acc += E2M1_LUT[(ws.z      ) & 0xFu] * s * x4_.x;
                acc += E2M1_LUT[(ws.z >>  4) & 0xFu] * s * x4_.y;
                acc += E2M1_LUT[(ws.z >>  8) & 0xFu] * s * x4_.z;
                acc += E2M1_LUT[(ws.z >> 12) & 0xFu] * s * x4_.w;
                acc += E2M1_LUT[(ws.z >> 16) & 0xFu] * s * x5.x;
                acc += E2M1_LUT[(ws.z >> 20) & 0xFu] * s * x5.y;
                acc += E2M1_LUT[(ws.z >> 24) & 0xFu] * s * x5.z;
                acc += E2M1_LUT[(ws.z >> 28) & 0xFu] * s * x5.w;

                acc += E2M1_LUT[(ws.w      ) & 0xFu] * s * x6.x;
                acc += E2M1_LUT[(ws.w >>  4) & 0xFu] * s * x6.y;
                acc += E2M1_LUT[(ws.w >>  8) & 0xFu] * s * x6.z;
                acc += E2M1_LUT[(ws.w >> 12) & 0xFu] * s * x6.w;
                acc += E2M1_LUT[(ws.w >> 16) & 0xFu] * s * x7.x;
                acc += E2M1_LUT[(ws.w >> 20) & 0xFu] * s * x7.y;
                acc += E2M1_LUT[(ws.w >> 24) & 0xFu] * s * x7.z;
                acc += E2M1_LUT[(ws.w >> 28) & 0xFu] * s * x7.w;
            }

            acc = simd_sum(acc);
            if (sg_lane == 0u) {
                row_acc = fma(weights[slot], acc, row_acc);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < dims.out_features && sg_lane == 0u) {
        y[b * dims.out_features + row] = row_acc;
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Lever C-atomic (2026-04-27) — mxfp4_matmul_moe_wsum_atomic_f32_v3:
// grid-parallel fused MoE down matmul + weighted sum via atomic_float adds.
//
// Same math as `mxfp4_matmul_moe_wsum_f32_v3` but restores the slot axis to
// the dispatch grid (`grid.z = k`, like the original `mxfp4_matmul_moe_f32_v3`
// down kernel) so 2048 TGs (production: hidden/8 × batch × k) run in parallel
// instead of 256. The slot-fold serial loop variant lost inter-TG parallelism
// (σ=−10.70 regression, see `lever_c_moe_matmul_wsum_concluded.md`).
//
// The cost recovered here is k-way atomic contention per output element:
// for each `(b, hr)` slot, k=8 TGs (one per slot) atomic_fetch_add their
// weighted contribution. M3+ supports `atomic<float>` natively. Caller must
// **pre-zero the output buffer** before this kernel runs, since each TG only
// adds (does not assign).
//
// Topology:
//   Grid = (out/ROWS_PER_TG, batch, k)        ROWS_PER_TG = 8
//   TG   = 256 threads = 8 SG × 32 lanes
//   TG-shared: x_shared[in_features] for hiddens of one slot
// ───────────────────────────────────────────────────────────────────────────
kernel void mxfp4_matmul_moe_wsum_atomic_f32_v3(
    device const uint*     packed_all     [[buffer(0)]],
    device const uchar*    scales_all     [[buffer(1)]],
    device const uint*     expert_indices [[buffer(2)]],
    device const float*    weights        [[buffer(3)]],
    device const float*    x              [[buffer(4)]],
    device atomic<float>*  y              [[buffer(5)]],
    constant MxFp4MoeMatmulWsumDims& dims [[buffer(6)]],
    threadgroup float*     x_shared       [[threadgroup(0)]],
    uint3 tg_pos            [[threadgroup_position_in_grid]],
    uint  tid_in_tg         [[thread_index_in_threadgroup]],
    uint  sg_id             [[simdgroup_index_in_threadgroup]],
    uint  sg_lane           [[thread_index_in_simdgroup]]
) {
    uint b    = tg_pos.y;
    uint slot = tg_pos.z;
    if (b >= dims.batch) { return; }

    const uint ROWS_PER_TG = 8u;
    const uint THREADS_PER_TG = 256u;
    uint row = tg_pos.x * ROWS_PER_TG + sg_id;

    uint groups        = dims.in_features / 32u;
    uint words_per_row = dims.in_features / 8u;
    uint packed_expert_stride = dims.out_features * words_per_row;
    uint scale_expert_stride  = dims.out_features * groups;

    // Stage hiddens[slot, b, :] into TG-shared. All 256 threads cooperate.
    uint h_row_base = slot * dims.batch * dims.in_features
                    + b * dims.in_features;
    for (uint i = tid_in_tg; i < dims.in_features; i += THREADS_PER_TG) {
        x_shared[i] = x[h_row_base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= dims.out_features) { return; }

    uint e = expert_indices[slot];
    uint word_row_base  = e * packed_expert_stride + row * words_per_row;
    uint scale_row_base = e * scale_expert_stride  + row * groups;

    float acc = 0.0f;
    for (uint g = sg_lane; g < groups; g += 32u) {
        uchar scale_byte = scales_all[scale_row_base + g];
        if (scale_byte == 0xFFu) continue;
        uint sbits = uint(scale_byte) << 23;
        float s = as_type<float>(sbits);
        if (s == 0.0f) continue;

        uint word_base = word_row_base + g * 4u;
        uint x_base    = g * 32u;

        uint4 ws = *((device const uint4*)(packed_all + word_base));
        threadgroup const float4* x4 =
            (threadgroup const float4*)(x_shared + x_base);
        float4 x0 = x4[0]; float4 x1 = x4[1];
        float4 x2 = x4[2]; float4 x3 = x4[3];
        float4 x4_ = x4[4]; float4 x5 = x4[5];
        float4 x6 = x4[6]; float4 x7 = x4[7];

        acc += E2M1_LUT[(ws.x      ) & 0xFu] * s * x0.x;
        acc += E2M1_LUT[(ws.x >>  4) & 0xFu] * s * x0.y;
        acc += E2M1_LUT[(ws.x >>  8) & 0xFu] * s * x0.z;
        acc += E2M1_LUT[(ws.x >> 12) & 0xFu] * s * x0.w;
        acc += E2M1_LUT[(ws.x >> 16) & 0xFu] * s * x1.x;
        acc += E2M1_LUT[(ws.x >> 20) & 0xFu] * s * x1.y;
        acc += E2M1_LUT[(ws.x >> 24) & 0xFu] * s * x1.z;
        acc += E2M1_LUT[(ws.x >> 28) & 0xFu] * s * x1.w;

        acc += E2M1_LUT[(ws.y      ) & 0xFu] * s * x2.x;
        acc += E2M1_LUT[(ws.y >>  4) & 0xFu] * s * x2.y;
        acc += E2M1_LUT[(ws.y >>  8) & 0xFu] * s * x2.z;
        acc += E2M1_LUT[(ws.y >> 12) & 0xFu] * s * x2.w;
        acc += E2M1_LUT[(ws.y >> 16) & 0xFu] * s * x3.x;
        acc += E2M1_LUT[(ws.y >> 20) & 0xFu] * s * x3.y;
        acc += E2M1_LUT[(ws.y >> 24) & 0xFu] * s * x3.z;
        acc += E2M1_LUT[(ws.y >> 28) & 0xFu] * s * x3.w;

        acc += E2M1_LUT[(ws.z      ) & 0xFu] * s * x4_.x;
        acc += E2M1_LUT[(ws.z >>  4) & 0xFu] * s * x4_.y;
        acc += E2M1_LUT[(ws.z >>  8) & 0xFu] * s * x4_.z;
        acc += E2M1_LUT[(ws.z >> 12) & 0xFu] * s * x4_.w;
        acc += E2M1_LUT[(ws.z >> 16) & 0xFu] * s * x5.x;
        acc += E2M1_LUT[(ws.z >> 20) & 0xFu] * s * x5.y;
        acc += E2M1_LUT[(ws.z >> 24) & 0xFu] * s * x5.z;
        acc += E2M1_LUT[(ws.z >> 28) & 0xFu] * s * x5.w;

        acc += E2M1_LUT[(ws.w      ) & 0xFu] * s * x6.x;
        acc += E2M1_LUT[(ws.w >>  4) & 0xFu] * s * x6.y;
        acc += E2M1_LUT[(ws.w >>  8) & 0xFu] * s * x6.z;
        acc += E2M1_LUT[(ws.w >> 12) & 0xFu] * s * x6.w;
        acc += E2M1_LUT[(ws.w >> 16) & 0xFu] * s * x7.x;
        acc += E2M1_LUT[(ws.w >> 20) & 0xFu] * s * x7.y;
        acc += E2M1_LUT[(ws.w >> 24) & 0xFu] * s * x7.z;
        acc += E2M1_LUT[(ws.w >> 28) & 0xFu] * s * x7.w;
    }

    acc = simd_sum(acc);
    if (sg_lane == 0u) {
        float weight = weights[slot];
        atomic_fetch_add_explicit(
            &y[b * dims.out_features + row],
            weight * acc,
            memory_order_relaxed);
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Lever G (2026-04-27) — topk_partial_select_f32: routing top-k fusion.
//
// Replaces the Candle chain in MoE routing
//   sorted = probs.arg_sort_last_dim(false)?    // [BL, E] full descending sort
//   inds = sorted.narrow(D::Minus1, 0, k)?.contiguous()?
//   scores = probs.gather(&inds, D::Minus1)?    // [BL, k]
// with a single dispatch that picks the top-k directly via iterated argmax
// with a mask. For the production decode shape (E=256, k=8, BL=1) this saves
// the full 256-element sort (only 8 winners needed) plus the narrow→contiguous
// last-dim copy and the gather.
//
// Algorithm: each TG handles one row (BL). 256 threads, each holding one
// expert's (val, idx) at the start. For i in 0..k: TG-shared tree reduction
// finds the max, lane 0 writes (val, idx) to out, threads with idx==winner
// mask their slot to -INF. Tie-break favors the lower index (stable, matches
// Candle's stable arg_sort behavior).
//
// Topology:
//   Grid = (1, BL, 1)
//   TG   = 256 threads (assumes E ≤ 256; production E=256 fills exactly)
//   TG-shared: shared_v[256] f32 + shared_i[256] u32
// ───────────────────────────────────────────────────────────────────────────
struct TopkPartialDims {
    uint num_experts;  // E
    uint k;
};

kernel void topk_partial_select_f32(
    device const float* probs    [[buffer(0)]],
    device uint*        inds_out [[buffer(1)]],
    device float*       vals_out [[buffer(2)]],
    constant TopkPartialDims& dims [[buffer(3)]],
    threadgroup float*  shared_v [[threadgroup(0)]],
    threadgroup uint*   shared_i [[threadgroup(1)]],
    uint                bl       [[threadgroup_position_in_grid]],
    uint                tid      [[thread_index_in_threadgroup]],
    uint                tg_size  [[threads_per_threadgroup]]
) {
    uint base = bl * dims.num_experts;
    float my_v = (tid < dims.num_experts) ? probs[base + tid] : -INFINITY;
    uint  my_i = tid;

    for (uint i = 0; i < dims.k; i++) {
        shared_v[tid] = my_v;
        shared_i[tid] = my_i;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                float other_v = shared_v[tid + stride];
                uint  other_i = shared_i[tid + stride];
                float my_shared_v = shared_v[tid];
                uint  my_shared_i = shared_i[tid];
                if (other_v > my_shared_v ||
                    (other_v == my_shared_v && other_i < my_shared_i)) {
                    shared_v[tid] = other_v;
                    shared_i[tid] = other_i;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        uint winner_i = shared_i[0];
        float winner_v = shared_v[0];

        if (tid == 0u) {
            inds_out[bl * dims.k + i] = winner_i;
            vals_out[bl * dims.k + i] = winner_v;
        }

        if (my_i == winner_i) {
            my_v = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ───────────────────────────────────────────────────────────────────────────
// dense_f32_matmul_rmsnorm — Lever H Step 2 dense kernel.
//
// f32 weight + raw x + rms_weight → y, with RmsNorm fused before the matmul.
// Mirrors `mxfp4_matmul_f32_v3_rmsnorm`'s 2-phase RmsNorm cooperative pass
// (same reduction order, same threadgroup memory layout) but reads f32 weight
// directly instead of MXFP4-packed nibbles + E8M0 scales.
//
// **Why it exists**: the routing gate (`gate`, `[num_experts, hidden]` f32)
// and `shared_expert_gate` (`[1, hidden]` f32) projections in Qwen3.5-VL-MoE
// are int8-affine in the source checkpoint and **dequantized to dense f32 at
// load time** (per loader.rs:546). Without an in-kernel RmsNorm path for the
// dense f32 case, the external `post_attention_layernorm.forward` dispatch
// can't be eliminated even after all MXFP4 consumers are fused.
//
// **Topology**: same as v3 — `(ceil(out/8), batch, 1)` TGs, each with 256
// threads (8 simdgroups, ROWS_PER_TG=8). For `out=1` (shared_expert_gate)
// the kernel still works (single TG, only sg_id=0 emits a result, sg_id 1..7
// help with RmsNorm staging then bail out). For `out=256` (routing gate)
// 32 TGs fill the typical occupancy floor.
//
// **Constraints**:
//   - `in_features` must be a multiple of 4 (float4 vector loads).
//   - `out_features` and `batch` may be any positive value.
//
// **Buffer layout**:
//   buffer(0): weight     [out, in] f32 row-major
//   buffer(1): x          raw post-attn residual, [batch, in] f32
//   buffer(2): rms_weight [in] f32
//   buffer(3): y          [batch, out] f32
//   buffer(4): dims       (out, in, rms_eps)
//   buffer(5): batch
//
// **Threadgroup memory**:
//   tg(0): x_shared   [in_features]  — RmsNorm output cache
//   tg(1): reduce_buf [8]             — SG partial sums + inv_rms broadcast
// ───────────────────────────────────────────────────────────────────────────
struct DenseMatmulRmsnormDims {
    uint out_features;
    uint in_features;
    float rms_eps;
};

kernel void dense_f32_matmul_rmsnorm(
    device const float*   weight     [[buffer(0)]],
    device const float*   x          [[buffer(1)]],
    device const float*   rms_weight [[buffer(2)]],
    device float*         y          [[buffer(3)]],
    constant DenseMatmulRmsnormDims& dims [[buffer(4)]],
    constant uint&        batch      [[buffer(5)]],
    threadgroup float*    x_shared   [[threadgroup(0)]],
    threadgroup float*    reduce_buf [[threadgroup(1)]],
    uint3 tg_pos          [[threadgroup_position_in_grid]],
    uint  tid_in_tg       [[thread_index_in_threadgroup]],
    uint  sg_id           [[simdgroup_index_in_threadgroup]],
    uint  sg_lane         [[thread_index_in_simdgroup]]
) {
    uint b = tg_pos.y;
    if (b >= batch) { return; }

    const uint ROWS_PER_TG = 8u;
    const uint THREADS_PER_TG = 256u;
    uint row = tg_pos.x * ROWS_PER_TG + sg_id;

    uint x_row_base = b * dims.in_features;

    // RmsNorm Phase 1: stage raw x + accumulate per-thread sum(x²).
    float sum_sq = 0.0f;
    for (uint i = tid_in_tg; i < dims.in_features; i += THREADS_PER_TG) {
        float v = x[x_row_base + i];
        x_shared[i] = v;
        sum_sq = fma(v, v, sum_sq);
    }

    sum_sq = simd_sum(sum_sq);
    if (sg_lane == 0u) {
        reduce_buf[sg_id] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_id == 0u) {
        float v = (sg_lane < 8u) ? reduce_buf[sg_lane] : 0.0f;
        v = simd_sum(v);
        if (sg_lane == 0u) {
            float mean_sq = v / float(dims.in_features);
            reduce_buf[0] = metal::rsqrt(mean_sq + dims.rms_eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = reduce_buf[0];

    // RmsNorm Phase 2: apply weight × inv_rms in-place on x_shared.
    for (uint i = tid_in_tg; i < dims.in_features; i += THREADS_PER_TG) {
        x_shared[i] = x_shared[i] * rms_weight[i] * inv_rms;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // === Dense f32 matmul body ===
    if (row >= dims.out_features) { return; }

    uint w_row_base = row * dims.in_features;
    uint vec_count  = dims.in_features / 4u;

    // float4 vectorized dot: 32 lanes parallel, each handles in/4/32 vectors.
    device const float4*   w4 = (device const float4*)(weight + w_row_base);
    threadgroup const float4* x4 = (threadgroup const float4*)(x_shared);

    float acc = 0.0f;
    for (uint i = sg_lane; i < vec_count; i += 32u) {
        float4 ws = w4[i];
        float4 xs = x4[i];
        acc = fma(ws.x, xs.x, acc);
        acc = fma(ws.y, xs.y, acc);
        acc = fma(ws.z, xs.z, acc);
        acc = fma(ws.w, xs.w, acc);
    }

    acc = simd_sum(acc);
    if (sg_lane == 0u) {
        y[b * dims.out_features + row] = acc;
    }
}

// ───────────────────────────────────────────────────────────────────────────
// `tri_add_f32` — Lever L1 Step 2 (MoE-side residual fusion).
//
// Computes `y[i] = a[i] + b[i] + c[i]` for `i in [0, n)`. Single-pass,
// fully memory-bandwidth-bound. Replaces a 2-add chain (Candle binary_add
// twice) with one dispatch:
//
//   Before: summed = (y_routed + shared_y)?    // 1 dispatch
//           out    = (h + summed)?              // 1 dispatch
//
//   After:  out    = tri_add(y_routed, shared_y, h)   // 1 dispatch
//
// Saves 1 dispatch / layer × 40 layers ≈ 40 dispatches / decode step.
//
// Layout: all 4 buffers are flat f32 of length `n`. Caller is responsible
// for reshape/cat — kernel is shape-agnostic.
//
// Threadgroup: 256 threads/TG, one element per thread per iteration.
// `n` need not be a multiple of 256 — out-of-range threads return.
// ───────────────────────────────────────────────────────────────────────────

kernel void tri_add_f32(
    device const float* a   [[buffer(0)]],
    device const float* b   [[buffer(1)]],
    device const float* c   [[buffer(2)]],
    device       float* y   [[buffer(3)]],
    constant uint&      n   [[buffer(4)]],
    uint                gid [[thread_position_in_grid]]
) {
    if (gid >= n) { return; }
    y[gid] = a[gid] + b[gid] + c[gid];
}

// `gated_tri_add_f32` — REVERTED 2026-05-07 (Lever L1 Step 3 NEGATIVE).
// Metal `exp(-c)` drifted ≤1 ULP vs Candle host f32::exp, breaking
// bit-identical decode (0/20 runs identical). Sigmoid stays as a separate
// Candle dispatch in `SparseMoeBlock::forward_with_rmsnorm`. See
// `l1_residual_fusion_step3_concluded_negative.md`.

// ───────────────────────────────────────────────────────────────────────────
// `scalar_mul_tri_add_f32` — Lever L1 Step 3.5 (drift-safe partial Step 3).
//
// Computes `y[t, h] = a[t, h] + b[t, h] * coef[t] + d[t, h]` for `t in [0,
// bl)`, `h in [0, hidden)`. Coef is a per-token scalar already-computed by
// the caller (e.g. Candle's sigmoid output for the MoE shared-expert gate).
//
// No transcendentals — pure FMA + add. Bit-identical to scalar reference
// `a + b * coef + d` modulo the FMA contraction the compiler emits, which
// is identical for the host scalar reference at fp32 precision.
//
// Replaces the 2-op chain (assuming sigmoid stays in Candle):
//
//   Before:
//     shared_y = shared_out * shared_coef    // 1 dispatch (broadcast_mul)
//     y        = tri_add(y_routed, shared_y, residual)  // 1 dispatch
//
//   After:
//     y = scalar_mul_tri_add(y_routed, shared_out, shared_coef, residual)
//                                                       // 1 dispatch
//
// Saves 1 dispatch / layer × 40 layers = 40 dispatches / decode step ≈ 2ms.
// Step 3 NEGATIVE (sigmoid drift) is sidestepped — sigmoid stays Candle-side.
//
// Layout:
//   - a, b, d, y: flat f32 of length `bl * hidden`
//   - coef: f32 of length `bl` (per-token scalar, already-sigmoided)
//
// Dispatch grid: 2-D — width = ceil(hidden/256), height = bl.
// ───────────────────────────────────────────────────────────────────────────

// CRITICAL: `#pragma clang fp contract(off)` disables automatic FMA fusion
// across the `a + b * coef + d` expression. Without this, Metal's Clang
// compiler synthesizes `fma(b, coef, a) + d` (or `fma(b, coef, a + d)`),
// which is single-rounded and ≤1 ULP MORE accurate than the unfused chain.
// More accurate sounds good — but Candle computes `(b * coef) + a + d` as
// THREE separate Metal kernels (each round-to-fp32 between ops). Bit-
// identical decode requires matching Candle's exact rounding sequence.
// Without `fp contract(off)`, A/B run produced 0/10 bit-identical (text
// diverges at run 0 char 499 in the same way as Step 3 NEGATIVE).
#pragma clang fp contract(off)
kernel void scalar_mul_tri_add_f32(
    device const float* a       [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    device const float* coef    [[buffer(2)]],
    device const float* d       [[buffer(3)]],
    device       float* y       [[buffer(4)]],
    constant uint&      hidden  [[buffer(5)]],
    constant uint&      bl      [[buffer(6)]],
    uint2               gid2    [[thread_position_in_grid]]
) {
    uint t = gid2.y;
    uint h = gid2.x;
    if (t >= bl || h >= hidden) { return; }
    uint idx = t * hidden + h;
    float c = coef[t];
    // Force the compiler to round each intermediate to fp32 so the result
    // matches Candle's 3-dispatch chain `(b * coef).round() + a.round() + d.round()`.
    float prod = b[idx] * c;
    float partial = a[idx] + prod;
    y[idx] = partial + d[idx];
}

// ───────────────────────────────────────────────────────────────────────────
// `scalar_mul_tri_add_rmsnorm_f32` — Lever L4 (cross-layer megafusion).
//
// Per-token (1 TG = 1 token):
//   out[t,h]     = a[t,h] + b[t,h] * coef[t] + d[t,h]   // residual stream
//   attn_in[t,h] = out[t,h] * rms_weight[h] * inv_rms_t // next layer's pre-norm
// where inv_rms_t = rsqrt(mean(out[t,:]²) + rms_eps).
//
// Replaces the 2-dispatch chain at the layer boundary:
//   Layer i:    out = scalar_mul_tri_add(...)            // 1 dispatch
//   Layer i+1:  attn_in = input_layernorm(out)            // 1 dispatch
// with a single fused dispatch that produces BOTH `out` (residual stream)
// AND `attn_in` (pre-normalized for next layer's attention). Saves 1
// dispatch / layer × 39 layers (layer 0 remains separate) ≈ 2 ms / step.
//
// `#pragma clang fp contract(off)` blocks Metal Clang's automatic FMA
// contraction for the scalar_mul_tri_add part (matches Candle reference).
// The RmsNorm reduction is cooperative (256 threads per row) using the
// same SG-reduce pattern as Lever H Step 2's `matmul_f32_v3_rmsnorm`.
//
// Layout:
//   - a, b, d, out, attn_in: flat f32 of length `bl * hidden`
//   - coef: f32 of length `bl` (per-token scalar gate, already-sigmoided)
//   - rms_weight: f32 of length `hidden` (input_layernorm of NEXT layer)
//
// Threadgroup memory: `out_shared[hidden]` stages `out` for Phase 3
// re-use (avoids re-reading device memory). reduce_buf[8] for SG partial
// sums. hidden=2048 → 8 KB tg-mem, well within Apple Silicon's 32 KB cap.
// ───────────────────────────────────────────────────────────────────────────

#pragma clang fp contract(off)
kernel void scalar_mul_tri_add_rmsnorm_f32(
    device const float* a            [[buffer(0)]],
    device const float* b            [[buffer(1)]],
    device const float* coef         [[buffer(2)]],
    device const float* d            [[buffer(3)]],
    device const float* rms_weight   [[buffer(4)]],
    device       float* out          [[buffer(5)]],
    device       float* attn_in      [[buffer(6)]],
    constant uint&      hidden       [[buffer(7)]],
    constant uint&      bl           [[buffer(8)]],
    constant float&     rms_eps      [[buffer(9)]],
    threadgroup float*  out_shared   [[threadgroup(0)]],
    threadgroup float*  reduce_buf   [[threadgroup(1)]],
    uint                tg_t         [[threadgroup_position_in_grid]],
    uint                tid          [[thread_index_in_threadgroup]],
    uint                sg_id        [[simdgroup_index_in_threadgroup]],
    uint                sg_lane      [[thread_index_in_simdgroup]]
) {
    if (tg_t >= bl) { return; }
    const uint THREADS_PER_TG = 256u;
    uint row_base = tg_t * hidden;
    float c = coef[tg_t];

    // Phase 1: scalar_mul_tri_add per element + accumulate sum_sq. Stage
    // `out` into threadgroup memory for Phase 3 re-use.
    float sum_sq = 0.0f;
    for (uint i = tid; i < hidden; i += THREADS_PER_TG) {
        float prod = b[row_base + i] * c;
        float partial = a[row_base + i] + prod;
        float ov = partial + d[row_base + i];
        out[row_base + i] = ov;
        out_shared[i] = ov;
        sum_sq = fma(ov, ov, sum_sq);
    }

    // SG-level partial reduction → 8 SGs each contribute one partial sum.
    sum_sq = simd_sum(sum_sq);
    if (sg_lane == 0u) {
        reduce_buf[sg_id] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // SG 0 reduces the 8 partials → final sum, computes inv_rms.
    if (sg_id == 0u) {
        float v = (sg_lane < 8u) ? reduce_buf[sg_lane] : 0.0f;
        v = simd_sum(v);
        if (sg_lane == 0u) {
            float mean_sq = v / float(hidden);
            reduce_buf[0] = metal::rsqrt(mean_sq + rms_eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = reduce_buf[0];

    // Phase 3: write attn_in = out * rms_weight * inv_rms.
    for (uint i = tid; i < hidden; i += THREADS_PER_TG) {
        attn_in[row_base + i] = out_shared[i] * rms_weight[i] * inv_rms;
    }
}

// ───────────────────────────────────────────────────────────────────────────
// router_softmax_topk_renorm_f32 — Phase C.1.Y full routing-pipeline fusion
//
// Replaces the entire 6-dispatch routing chain in moe.rs:658-741 with a
// single Metal dispatch. The Candle reference path is:
//
//   probs   = softmax_last_dim(logits)       // [BL, E]      dispatch 1
//   sorted  = probs.arg_sort_last_dim(false) // [BL, E]      dispatch 2
//   inds    = sorted.narrow(0, k).contig     // [BL, K]      dispatch 3
//   scores  = probs.gather(inds)             // [BL, K]      dispatch 4
//   denom   = scores.sum_keepdim(-1)         // [BL, 1]      dispatch 5
//   weights = scores.broadcast_div(denom)    // [BL, K]      dispatch 6
//
// → 5 dispatches saved/layer × 40 MoE layers = 200 dispatches/decode step.
//
// Algorithm (single TG/row, 256 threads):
//   Phase 1 — softmax over E logits:
//             simd_max within SG → 8 partials → final simd_max → max_logit
//             exp_v = metal::fast::exp(logit - max_logit)  [matches Candle]
//             simd_sum partials → sum_exp; prob = exp_v / sum_exp.
//   Phase 2 — iterated argmax top-K (K ≤ 32) via TG-shared tree reduction
//             with lower-index tie-break (matches Candle stable arg_sort).
//             Mask winner = -INF after each iter; collect K winners in
//             `reduce_buf[0..K]` for the renorm phase.
//   Phase 3 — renormalize: SG 0 simd_sum over K winners → inv_sum →
//             write vals_out[bl*K..bl*K+K] = winner_v / sum_w.
//
// Topology:
//   Grid = (1, BL, 1)
//   TG   = 256 threads (E ≤ 256 = TG; K ≤ 32 fits one simdgroup for renorm)
//   TG-shared:
//     shared_v   [E]            f32  (top-K tree reduction)
//     shared_i   [E]            u32
//     reduce_buf [max(8, K)]    f32  (cross-SG max/sum + winner accumulator)
//
// Drift-class (vs Candle softmax_last_dim Metal kernel reduce.metal:910):
//   - `metal::fast::exp` is exactly Candle's exp (reduce.metal:840) — no
//     transcendental host-vs-Metal divergence (the L1 Step 3 sigmoid trap).
//   - `simd_max` + cross-SG reduction order may differ ≤1 ULP from Candle's
//     `el_per_block` partition. Top-K argpartition is stable when logit
//     gaps > 1 ULP (typical for E=256 routing). Weights cos ≥ 0.9999.
//   - End-to-end bit-identical greedy decode is the gold gate; parity test
//     (cos / abs) is necessary but not sufficient for autoregressive drift.
//
// Constraints:
//   - E ≤ 256 (TG size). K ≤ 32 (renorm uses single SG simd_sum).
//   - logits in F32. Caller must zero-fill or guard if E < 256 (the kernel
//     does this via `tid < E ? logit : -INF`).
// ───────────────────────────────────────────────────────────────────────────
struct RouterFusedDims {
    uint num_experts;  // E (≤ 256)
    uint k;            // top-K (≤ 32)
};

#pragma clang fp contract(off)
kernel void router_softmax_topk_renorm_f32(
    device const float*  logits     [[buffer(0)]],
    device       uint*   inds_out   [[buffer(1)]],
    device       float*  vals_out   [[buffer(2)]],
    constant RouterFusedDims& dims  [[buffer(3)]],
    threadgroup float*   shared_v   [[threadgroup(0)]],
    threadgroup uint*    shared_i   [[threadgroup(1)]],
    threadgroup float*   reduce_buf [[threadgroup(2)]],
    uint                 bl         [[threadgroup_position_in_grid]],
    uint                 tid        [[thread_index_in_threadgroup]],
    uint                 sg_id      [[simdgroup_index_in_threadgroup]],
    uint                 sg_lane    [[thread_index_in_simdgroup]],
    uint                 tg_size    [[threads_per_threadgroup]]
) {
    const uint E = dims.num_experts;
    const uint K = dims.k;
    uint base = bl * E;

    // Phase 1a: load logit; cross-SG simd_max → max_logit.
    float my_logit = (tid < E) ? logits[base + tid] : -INFINITY;

    float sg_max = simd_max(my_logit);
    if (sg_lane == 0u) {
        reduce_buf[sg_id] = sg_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_id == 0u) {
        float v = (sg_lane < 8u) ? reduce_buf[sg_lane] : -INFINITY;
        v = simd_max(v);
        if (sg_lane == 0u) {
            reduce_buf[0] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float max_logit = reduce_buf[0];

    // Phase 1b: exp(logit - max), cross-SG simd_sum → sum_exp.
    float exp_v = (tid < E) ? metal::fast::exp(my_logit - max_logit) : 0.0f;

    float sg_sum = simd_sum(exp_v);
    if (sg_lane == 0u) {
        reduce_buf[sg_id] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_id == 0u) {
        float v = (sg_lane < 8u) ? reduce_buf[sg_lane] : 0.0f;
        v = simd_sum(v);
        if (sg_lane == 0u) {
            reduce_buf[0] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum_exp = reduce_buf[0];

    // probs[tid] = exp_v / sum_exp (only meaningful for tid < E; else -INF).
    float prob = (tid < E) ? (exp_v / sum_exp) : -INFINITY;

    // Phase 2: iterated argmax top-K via TG-shared tree reduction.
    float my_v = prob;
    uint  my_i = tid;

    for (uint i = 0u; i < K; i++) {
        shared_v[tid] = my_v;
        shared_i[tid] = my_i;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                float other_v = shared_v[tid + stride];
                uint  other_i = shared_i[tid + stride];
                float my_shared_v = shared_v[tid];
                uint  my_shared_i = shared_i[tid];
                if (other_v > my_shared_v ||
                    (other_v == my_shared_v && other_i < my_shared_i)) {
                    shared_v[tid] = other_v;
                    shared_i[tid] = other_i;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        uint  winner_i = shared_i[0];
        float winner_v = shared_v[0];

        if (tid == 0u) {
            inds_out[bl * K + i] = winner_i;   // device write — final
            reduce_buf[i]        = winner_v;   // tg memory — read in Phase 3
        }

        if (my_i == winner_i) {
            my_v = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Phase 3: renormalize top-K weights.
    // K ≤ 32 fits one simdgroup; SG 0 reads K winners from reduce_buf,
    // simd_sum to sum_w, then writes vals_out = winner_v / sum_w.
    if (sg_id == 0u) {
        float w     = (sg_lane < K) ? reduce_buf[sg_lane] : 0.0f;
        float sum_w = simd_sum(w);
        if (sg_lane < K) {
            vals_out[bl * K + sg_lane] = w / sum_w;
        }
    }
}
