// Affine 4-bit fused dequant + matmul kernel.
//
// Layout matches `crate::affine4` (MLX 4-bit affine quant convention used by
// e.g. `Qwen3.6-27B-MLX-4bit` whose `quantization_config.mode == "affine"` and
// `quantization_config.bits == 4`):
//
//   packed: [out_features, in_features/8]   uint    -- 8 nibbles per word, LSB-first
//   scales: [out_features, in_features/64]  ushort  -- one bf16 scale per 64-element group
//   biases: [out_features, in_features/64]  ushort  -- one bf16 bias  per 64-element group
//   x     : [batch, in_features]            float
//   y     : [batch, out_features]           float
//
// Per-element dequant: `w_real = u4_value * scale + bias`
// where `u4_value` is the raw 0..15 nibble (unsigned, no signed lookup unlike MXFP4).
//
// One thread per output row × batch element. Each thread iterates all groups and
// accumulates the dot product on the fly, never materializing the dequantized weight
// tensor — same access pattern as `mxfp4_matmul_f32`.

#include <metal_stdlib>
using namespace metal;

struct Affine4Dims {
    uint out_features;
    uint in_features;
};

struct Affine4MatmulRmsnormDims {
    uint out_features;
    uint in_features;
    float rms_eps;
};

// Decode a bf16 half stored as a `ushort` into `float`. MLX writes bf16 as the
// upper 16 bits of an IEEE f32, so left-shifting by 16 reconstructs the f32.
static inline float bf16_to_f32_device(ushort b) {
    uint bits = uint(b) << 16;
    return as_type<float>(bits);
}

// Helper: dequant + dot-product accumulation for ONE 64-element group.
// Reads 8 packed words (= 64 nibbles) + 16 float4s from threadgroup-staged x.
// `acc += sum_{i=0..63} (nib[i] * s + bi) * x[i]`. Inlined into each kernel
// so unrolling unfolds at the call-site (no function-call overhead).
static inline float affine4_group_dot_tg(
    uint4 ws0, uint4 ws1,
    float s, float bi,
    threadgroup const float4* x4
) {
    float4 xa0 = x4[ 0]; float4 xa1 = x4[ 1];
    float4 xa2 = x4[ 2]; float4 xa3 = x4[ 3];
    float4 xa4 = x4[ 4]; float4 xa5 = x4[ 5];
    float4 xa6 = x4[ 6]; float4 xa7 = x4[ 7];
    float4 xa8 = x4[ 8]; float4 xa9 = x4[ 9];
    float4 xaA = x4[10]; float4 xaB = x4[11];
    float4 xaC = x4[12]; float4 xaD = x4[13];
    float4 xaE = x4[14]; float4 xaF = x4[15];
    float acc = 0.0f;
    acc += (float((ws0.x      ) & 0xFu) * s + bi) * xa0.x;
    acc += (float((ws0.x >>  4) & 0xFu) * s + bi) * xa0.y;
    acc += (float((ws0.x >>  8) & 0xFu) * s + bi) * xa0.z;
    acc += (float((ws0.x >> 12) & 0xFu) * s + bi) * xa0.w;
    acc += (float((ws0.x >> 16) & 0xFu) * s + bi) * xa1.x;
    acc += (float((ws0.x >> 20) & 0xFu) * s + bi) * xa1.y;
    acc += (float((ws0.x >> 24) & 0xFu) * s + bi) * xa1.z;
    acc += (float((ws0.x >> 28) & 0xFu) * s + bi) * xa1.w;
    acc += (float((ws0.y      ) & 0xFu) * s + bi) * xa2.x;
    acc += (float((ws0.y >>  4) & 0xFu) * s + bi) * xa2.y;
    acc += (float((ws0.y >>  8) & 0xFu) * s + bi) * xa2.z;
    acc += (float((ws0.y >> 12) & 0xFu) * s + bi) * xa2.w;
    acc += (float((ws0.y >> 16) & 0xFu) * s + bi) * xa3.x;
    acc += (float((ws0.y >> 20) & 0xFu) * s + bi) * xa3.y;
    acc += (float((ws0.y >> 24) & 0xFu) * s + bi) * xa3.z;
    acc += (float((ws0.y >> 28) & 0xFu) * s + bi) * xa3.w;
    acc += (float((ws0.z      ) & 0xFu) * s + bi) * xa4.x;
    acc += (float((ws0.z >>  4) & 0xFu) * s + bi) * xa4.y;
    acc += (float((ws0.z >>  8) & 0xFu) * s + bi) * xa4.z;
    acc += (float((ws0.z >> 12) & 0xFu) * s + bi) * xa4.w;
    acc += (float((ws0.z >> 16) & 0xFu) * s + bi) * xa5.x;
    acc += (float((ws0.z >> 20) & 0xFu) * s + bi) * xa5.y;
    acc += (float((ws0.z >> 24) & 0xFu) * s + bi) * xa5.z;
    acc += (float((ws0.z >> 28) & 0xFu) * s + bi) * xa5.w;
    acc += (float((ws0.w      ) & 0xFu) * s + bi) * xa6.x;
    acc += (float((ws0.w >>  4) & 0xFu) * s + bi) * xa6.y;
    acc += (float((ws0.w >>  8) & 0xFu) * s + bi) * xa6.z;
    acc += (float((ws0.w >> 12) & 0xFu) * s + bi) * xa6.w;
    acc += (float((ws0.w >> 16) & 0xFu) * s + bi) * xa7.x;
    acc += (float((ws0.w >> 20) & 0xFu) * s + bi) * xa7.y;
    acc += (float((ws0.w >> 24) & 0xFu) * s + bi) * xa7.z;
    acc += (float((ws0.w >> 28) & 0xFu) * s + bi) * xa7.w;
    acc += (float((ws1.x      ) & 0xFu) * s + bi) * xa8.x;
    acc += (float((ws1.x >>  4) & 0xFu) * s + bi) * xa8.y;
    acc += (float((ws1.x >>  8) & 0xFu) * s + bi) * xa8.z;
    acc += (float((ws1.x >> 12) & 0xFu) * s + bi) * xa8.w;
    acc += (float((ws1.x >> 16) & 0xFu) * s + bi) * xa9.x;
    acc += (float((ws1.x >> 20) & 0xFu) * s + bi) * xa9.y;
    acc += (float((ws1.x >> 24) & 0xFu) * s + bi) * xa9.z;
    acc += (float((ws1.x >> 28) & 0xFu) * s + bi) * xa9.w;
    acc += (float((ws1.y      ) & 0xFu) * s + bi) * xaA.x;
    acc += (float((ws1.y >>  4) & 0xFu) * s + bi) * xaA.y;
    acc += (float((ws1.y >>  8) & 0xFu) * s + bi) * xaA.z;
    acc += (float((ws1.y >> 12) & 0xFu) * s + bi) * xaA.w;
    acc += (float((ws1.y >> 16) & 0xFu) * s + bi) * xaB.x;
    acc += (float((ws1.y >> 20) & 0xFu) * s + bi) * xaB.y;
    acc += (float((ws1.y >> 24) & 0xFu) * s + bi) * xaB.z;
    acc += (float((ws1.y >> 28) & 0xFu) * s + bi) * xaB.w;
    acc += (float((ws1.z      ) & 0xFu) * s + bi) * xaC.x;
    acc += (float((ws1.z >>  4) & 0xFu) * s + bi) * xaC.y;
    acc += (float((ws1.z >>  8) & 0xFu) * s + bi) * xaC.z;
    acc += (float((ws1.z >> 12) & 0xFu) * s + bi) * xaC.w;
    acc += (float((ws1.z >> 16) & 0xFu) * s + bi) * xaD.x;
    acc += (float((ws1.z >> 20) & 0xFu) * s + bi) * xaD.y;
    acc += (float((ws1.z >> 24) & 0xFu) * s + bi) * xaD.z;
    acc += (float((ws1.z >> 28) & 0xFu) * s + bi) * xaD.w;
    acc += (float((ws1.w      ) & 0xFu) * s + bi) * xaE.x;
    acc += (float((ws1.w >>  4) & 0xFu) * s + bi) * xaE.y;
    acc += (float((ws1.w >>  8) & 0xFu) * s + bi) * xaE.z;
    acc += (float((ws1.w >> 12) & 0xFu) * s + bi) * xaE.w;
    acc += (float((ws1.w >> 16) & 0xFu) * s + bi) * xaF.x;
    acc += (float((ws1.w >> 20) & 0xFu) * s + bi) * xaF.y;
    acc += (float((ws1.w >> 24) & 0xFu) * s + bi) * xaF.z;
    acc += (float((ws1.w >> 28) & 0xFu) * s + bi) * xaF.w;
    return acc;
}

// Group size for the 4-bit affine encoding. Fixed at 64 by the MLX quant recipe.
constant uint AFFINE4_GROUP_SIZE = 64u;
// Words (u32) per group. 64 nibbles / 8 nibbles per word = 8.
constant uint AFFINE4_WORDS_PER_GROUP = 8u;

kernel void affine4_matvec_f32(
    device const uint*    packed  [[buffer(0)]],
    device const ushort*  scales  [[buffer(1)]],
    device const ushort*  biases  [[buffer(2)]],
    device const float*   x       [[buffer(3)]],
    device float*         y       [[buffer(4)]],
    constant Affine4Dims& dims    [[buffer(5)]],
    uint row                      [[thread_position_in_grid]]
) {
    if (row >= dims.out_features) { return; }

    uint groups         = dims.in_features / AFFINE4_GROUP_SIZE;
    uint words_per_row  = dims.in_features / 8u;
    uint word_row_base  = row * words_per_row;
    uint scale_row_base = row * groups;

    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        float s = bf16_to_f32_device(scales[scale_row_base + g]);
        float b = bf16_to_f32_device(biases[scale_row_base + g]);

        uint word_base = word_row_base + g * AFFINE4_WORDS_PER_GROUP;
        uint x_base    = g * AFFINE4_GROUP_SIZE;
        for (uint w = 0; w < AFFINE4_WORDS_PER_GROUP; ++w) {
            uint word = packed[word_base + w];
            for (uint i = 0; i < 8u; ++i) {
                float v = float((word >> (i * 4u)) & 0xFu);
                float dequant = v * s + b;
                acc += dequant * x[x_base + w * 8u + i];
            }
        }
    }
    y[row] = acc;
}

// Batched variant: grid is (out_features, batch). One thread = one output element
// y[b, row] = sum_k W[row, k] * x[b, k]. Weight is shared across the batch.
// V2 kernel: vectorized uint4/float4 loads, 1 thread per (row, batch_elem).
// No threadgroup memory → handles ANY `in_features` (including 27B-Dense
// down_proj with in=22528). Halves load instructions vs v1. Used as the
// fallback when v3's 32 KB TG memory budget is exceeded.
kernel void affine4_matmul_f32_v2(
    device const uint*    packed   [[buffer(0)]],
    device const ushort*  scales   [[buffer(1)]],
    device const ushort*  biases   [[buffer(2)]],
    device const float*   x        [[buffer(3)]],
    device float*         y        [[buffer(4)]],
    constant Affine4Dims& dims     [[buffer(5)]],
    constant uint&        batch    [[buffer(6)]],
    uint2 gid                      [[thread_position_in_grid]]
) {
    uint row = gid.x;
    uint b   = gid.y;
    if (row >= dims.out_features || b >= batch) { return; }

    uint groups         = dims.in_features / AFFINE4_GROUP_SIZE;
    uint words_per_row  = dims.in_features / 8u;
    uint word_row_base  = row * words_per_row;
    uint scale_row_base = row * groups;
    uint x_row_base     = b * dims.in_features;

    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        uint sb = uint(scales[scale_row_base + g]) << 16;
        uint bb = uint(biases[scale_row_base + g]) << 16;
        float s = as_type<float>(sb);
        float bi = as_type<float>(bb);

        uint word_base = word_row_base + g * AFFINE4_WORDS_PER_GROUP;
        uint x_base    = x_row_base + g * AFFINE4_GROUP_SIZE;

        // 2 × uint4 = 8 words = 64 nibbles per group.
        uint4 ws0 = *((device const uint4*)(packed + word_base));
        uint4 ws1 = *((device const uint4*)(packed + word_base + 4u));

        // 16 × float4 = 64 floats. x is global memory but vectorized loads are still better
        // than per-element scalar loads.
        device const float4* x4 = (device const float4*)(x + x_base);
        float4 xa0 = x4[ 0]; float4 xa1 = x4[ 1];
        float4 xa2 = x4[ 2]; float4 xa3 = x4[ 3];
        float4 xa4 = x4[ 4]; float4 xa5 = x4[ 5];
        float4 xa6 = x4[ 6]; float4 xa7 = x4[ 7];
        float4 xa8 = x4[ 8]; float4 xa9 = x4[ 9];
        float4 xaA = x4[10]; float4 xaB = x4[11];
        float4 xaC = x4[12]; float4 xaD = x4[13];
        float4 xaE = x4[14]; float4 xaF = x4[15];

        acc += (float((ws0.x      ) & 0xFu) * s + bi) * xa0.x;
        acc += (float((ws0.x >>  4) & 0xFu) * s + bi) * xa0.y;
        acc += (float((ws0.x >>  8) & 0xFu) * s + bi) * xa0.z;
        acc += (float((ws0.x >> 12) & 0xFu) * s + bi) * xa0.w;
        acc += (float((ws0.x >> 16) & 0xFu) * s + bi) * xa1.x;
        acc += (float((ws0.x >> 20) & 0xFu) * s + bi) * xa1.y;
        acc += (float((ws0.x >> 24) & 0xFu) * s + bi) * xa1.z;
        acc += (float((ws0.x >> 28) & 0xFu) * s + bi) * xa1.w;
        acc += (float((ws0.y      ) & 0xFu) * s + bi) * xa2.x;
        acc += (float((ws0.y >>  4) & 0xFu) * s + bi) * xa2.y;
        acc += (float((ws0.y >>  8) & 0xFu) * s + bi) * xa2.z;
        acc += (float((ws0.y >> 12) & 0xFu) * s + bi) * xa2.w;
        acc += (float((ws0.y >> 16) & 0xFu) * s + bi) * xa3.x;
        acc += (float((ws0.y >> 20) & 0xFu) * s + bi) * xa3.y;
        acc += (float((ws0.y >> 24) & 0xFu) * s + bi) * xa3.z;
        acc += (float((ws0.y >> 28) & 0xFu) * s + bi) * xa3.w;
        acc += (float((ws0.z      ) & 0xFu) * s + bi) * xa4.x;
        acc += (float((ws0.z >>  4) & 0xFu) * s + bi) * xa4.y;
        acc += (float((ws0.z >>  8) & 0xFu) * s + bi) * xa4.z;
        acc += (float((ws0.z >> 12) & 0xFu) * s + bi) * xa4.w;
        acc += (float((ws0.z >> 16) & 0xFu) * s + bi) * xa5.x;
        acc += (float((ws0.z >> 20) & 0xFu) * s + bi) * xa5.y;
        acc += (float((ws0.z >> 24) & 0xFu) * s + bi) * xa5.z;
        acc += (float((ws0.z >> 28) & 0xFu) * s + bi) * xa5.w;
        acc += (float((ws0.w      ) & 0xFu) * s + bi) * xa6.x;
        acc += (float((ws0.w >>  4) & 0xFu) * s + bi) * xa6.y;
        acc += (float((ws0.w >>  8) & 0xFu) * s + bi) * xa6.z;
        acc += (float((ws0.w >> 12) & 0xFu) * s + bi) * xa6.w;
        acc += (float((ws0.w >> 16) & 0xFu) * s + bi) * xa7.x;
        acc += (float((ws0.w >> 20) & 0xFu) * s + bi) * xa7.y;
        acc += (float((ws0.w >> 24) & 0xFu) * s + bi) * xa7.z;
        acc += (float((ws0.w >> 28) & 0xFu) * s + bi) * xa7.w;
        acc += (float((ws1.x      ) & 0xFu) * s + bi) * xa8.x;
        acc += (float((ws1.x >>  4) & 0xFu) * s + bi) * xa8.y;
        acc += (float((ws1.x >>  8) & 0xFu) * s + bi) * xa8.z;
        acc += (float((ws1.x >> 12) & 0xFu) * s + bi) * xa8.w;
        acc += (float((ws1.x >> 16) & 0xFu) * s + bi) * xa9.x;
        acc += (float((ws1.x >> 20) & 0xFu) * s + bi) * xa9.y;
        acc += (float((ws1.x >> 24) & 0xFu) * s + bi) * xa9.z;
        acc += (float((ws1.x >> 28) & 0xFu) * s + bi) * xa9.w;
        acc += (float((ws1.y      ) & 0xFu) * s + bi) * xaA.x;
        acc += (float((ws1.y >>  4) & 0xFu) * s + bi) * xaA.y;
        acc += (float((ws1.y >>  8) & 0xFu) * s + bi) * xaA.z;
        acc += (float((ws1.y >> 12) & 0xFu) * s + bi) * xaA.w;
        acc += (float((ws1.y >> 16) & 0xFu) * s + bi) * xaB.x;
        acc += (float((ws1.y >> 20) & 0xFu) * s + bi) * xaB.y;
        acc += (float((ws1.y >> 24) & 0xFu) * s + bi) * xaB.z;
        acc += (float((ws1.y >> 28) & 0xFu) * s + bi) * xaB.w;
        acc += (float((ws1.z      ) & 0xFu) * s + bi) * xaC.x;
        acc += (float((ws1.z >>  4) & 0xFu) * s + bi) * xaC.y;
        acc += (float((ws1.z >>  8) & 0xFu) * s + bi) * xaC.z;
        acc += (float((ws1.z >> 12) & 0xFu) * s + bi) * xaC.w;
        acc += (float((ws1.z >> 16) & 0xFu) * s + bi) * xaD.x;
        acc += (float((ws1.z >> 20) & 0xFu) * s + bi) * xaD.y;
        acc += (float((ws1.z >> 24) & 0xFu) * s + bi) * xaD.z;
        acc += (float((ws1.z >> 28) & 0xFu) * s + bi) * xaD.w;
        acc += (float((ws1.w      ) & 0xFu) * s + bi) * xaE.x;
        acc += (float((ws1.w >>  4) & 0xFu) * s + bi) * xaE.y;
        acc += (float((ws1.w >>  8) & 0xFu) * s + bi) * xaE.z;
        acc += (float((ws1.w >> 12) & 0xFu) * s + bi) * xaE.w;
        acc += (float((ws1.w >> 16) & 0xFu) * s + bi) * xaF.x;
        acc += (float((ws1.w >> 20) & 0xFu) * s + bi) * xaF.y;
        acc += (float((ws1.w >> 24) & 0xFu) * s + bi) * xaF.z;
        acc += (float((ws1.w >> 28) & 0xFu) * s + bi) * xaF.w;
    }
    y[b * dims.out_features + row] = acc;
}

// ────────────────────────────────────────────────────────────────────────────
// V3-tiled kernel: same simdgroup-cooperative pattern as v3, but the activation
// is staged in **chunks** along the in_features axis so we fit inside the 32 KB
// threadgroup-memory budget for shapes where the full activation does not fit
// (e.g. 27B-Dense `down_proj` with in_features=22528).
//
// Each threadgroup processes ROWS_PER_TG=8 output rows × ONE chunk × ONE batch
// element. Per-chunk partial sums are written to a scratch buffer
//   scratch[chunk, batch, out]  (n_chunks × batch × out_features f32)
// and a tiny follow-up `affine4_reduce_chunks_f32` kernel sums over the chunk
// axis to produce the final `y`.
//
// Tile size invariants (caller-enforced):
//   tile_in <= AFFINE4_V3_MAX_IN_FEATURES (8192)
//   tile_in % AFFINE4_GROUP_SIZE == 0
//   in_features % tile_in == 0  (equal-sized chunks)
// ────────────────────────────────────────────────────────────────────────────
struct Affine4TiledDims {
    uint out_features;
    uint in_features;
    uint tile_in;     // chunk size along in axis (multiple of 64)
    uint n_chunks;    // in_features / tile_in
};

kernel void affine4_matmul_f32_v3_tiled(
    device const uint*    packed   [[buffer(0)]],
    device const ushort*  scales   [[buffer(1)]],
    device const ushort*  biases   [[buffer(2)]],
    device const float*   x        [[buffer(3)]],   // [batch, in_features]
    device float*         scratch  [[buffer(4)]],   // [n_chunks, batch, out_features]
    constant Affine4TiledDims& dims [[buffer(5)]],
    constant uint&        batch    [[buffer(6)]],
    threadgroup float*    x_shared [[threadgroup(0)]],
    uint3 tg_pos          [[threadgroup_position_in_grid]],
    uint  tid_in_tg       [[thread_index_in_threadgroup]],
    uint  sg_id           [[simdgroup_index_in_threadgroup]],
    uint  sg_lane         [[thread_index_in_simdgroup]]
) {
    uint b     = tg_pos.y;
    uint chunk = tg_pos.z;
    if (b >= batch || chunk >= dims.n_chunks) { return; }

    const uint ROWS_PER_TG = 8u;
    const uint THREADS_PER_TG = 256u;
    uint row = tg_pos.x * ROWS_PER_TG + sg_id;

    uint chunk_in_start = chunk * dims.tile_in;
    uint x_row_base     = b * dims.in_features;

    // Stage `tile_in` floats from x[b, chunk_in_start : chunk_in_start + tile_in].
    for (uint i = tid_in_tg; i < dims.tile_in; i += THREADS_PER_TG) {
        x_shared[i] = x[x_row_base + chunk_in_start + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= dims.out_features) { return; }

    uint groups_total   = dims.in_features / AFFINE4_GROUP_SIZE;
    uint groups_per_chunk = dims.tile_in / AFFINE4_GROUP_SIZE;
    uint groups_chunk_start = chunk * groups_per_chunk;
    uint words_per_row  = dims.in_features / 8u;
    uint word_row_base  = row * words_per_row;
    uint scale_row_base = row * groups_total;

    float acc = 0.0f;
    // Lanes partition this chunk's groups (adjacent lanes → adjacent groups for coalesced reads).
    for (uint g_local = sg_lane; g_local < groups_per_chunk; g_local += 32u) {
        uint g = groups_chunk_start + g_local;

        uint sb = uint(scales[scale_row_base + g]) << 16;
        uint bb = uint(biases[scale_row_base + g]) << 16;
        float s  = as_type<float>(sb);
        float bi = as_type<float>(bb);

        uint word_base = word_row_base + g * AFFINE4_WORDS_PER_GROUP;
        // x_shared offset: chunk-local position (group `g_local`).
        uint x_local_base = g_local * AFFINE4_GROUP_SIZE;

        uint4 ws0 = *((device const uint4*)(packed + word_base));
        uint4 ws1 = *((device const uint4*)(packed + word_base + 4u));

        threadgroup const float4* x4 =
            (threadgroup const float4*)(x_shared + x_local_base);
        acc += affine4_group_dot_tg(ws0, ws1, s, bi, x4);
    }

    acc = simd_sum(acc);

    if (sg_lane == 0u) {
        // scratch[chunk, b, row]
        uint scratch_idx =
            ((chunk * batch) + b) * dims.out_features + row;
        scratch[scratch_idx] = acc;
    }
}

struct Affine4ReduceDims {
    uint out_features;
    uint n_chunks;
};

// Companion reduction: sum scratch over the chunk axis to produce the final y.
// Grid: (out_features, batch). One thread per output element. n_chunks is small
// (≤ 8 for current shapes), so a serial loop is fine.
kernel void affine4_reduce_chunks_f32(
    device const float* scratch     [[buffer(0)]],   // [n_chunks, batch, out]
    device float*       y           [[buffer(1)]],   // [batch, out]
    constant Affine4ReduceDims& dims [[buffer(2)]],
    constant uint&      batch       [[buffer(3)]],
    uint2 gid                       [[thread_position_in_grid]]
) {
    uint row = gid.x;
    uint b   = gid.y;
    if (row >= dims.out_features || b >= batch) { return; }

    float sum = 0.0f;
    for (uint c = 0; c < dims.n_chunks; ++c) {
        uint idx = ((c * batch) + b) * dims.out_features + row;
        sum += scratch[idx];
    }
    y[b * dims.out_features + row] = sum;
}

// Fused-residual reduction: same as `affine4_reduce_chunks_f32` but adds a
// per-element residual to the output. Saves a downstream broadcast_add
// dispatch when the matmul output feeds a residual addition (e.g.
// `out = h + down_proj(x)` in 27B-Dense MLP block tail).
kernel void affine4_reduce_chunks_f32_residual(
    device const float* scratch     [[buffer(0)]],   // [n_chunks, batch, out]
    device const float* residual    [[buffer(1)]],   // [batch, out]
    device float*       y           [[buffer(2)]],   // [batch, out]
    constant Affine4ReduceDims& dims [[buffer(3)]],
    constant uint&      batch       [[buffer(4)]],
    uint2 gid                       [[thread_position_in_grid]]
) {
    uint row = gid.x;
    uint b   = gid.y;
    if (row >= dims.out_features || b >= batch) { return; }

    float sum = 0.0f;
    for (uint c = 0; c < dims.n_chunks; ++c) {
        uint idx = ((c * batch) + b) * dims.out_features + row;
        sum += scratch[idx];
    }
    uint y_idx = b * dims.out_features + row;
    y[y_idx] = sum + residual[y_idx];
}

// ────────────────────────────────────────────────────────────────────────────
// QMV_FAST: MLX-pattern decode kernel for affine 4-bit (bits=4, group_size=64).
//
// Differs from v3 in three structural ways that, in benchmark, account for the
// gap to the MLX reference implementation:
//   1. Smaller TG (64 threads vs v3's 256) — more concurrent TGs per
//      execution unit on Apple Silicon → better GPU utilization.
//   2. NO threadgroup memory cache for x — when weight read is the dominant
//      bandwidth term (e.g. 27B Dense decode), caching activations does not
//      pay; freeing the TG budget enables more concurrent dispatch.
//   3. Reformulated dot product: pre-scale x_thread[] by powers of 16 at load
//      time, then dot against UNSHIFTED nibble masks. Per-group cost goes
//      from 192 ops (v3 per-element `(nib*s+b)*x`) to ~65 ops
//      (`s * Σ(nib*x') + b * Σ(x)`). Saves ~3× FMA work per group.
//
// Constants tuned for our affine 4-bit format (bits=4, group_size=64,
// scales+biases bf16-as-u16):
//   pack_factor=8 (8 nibbles per uint32), packs_per_thread=2 → values_per_thread=16
//   num_simdgroups=2, results_per_simdgroup=4 → 8 output rows per TG
//   block_size=512 (values_per_thread * SIMD_SIZE)
//   scale_step_per_thread = group_size / values_per_thread = 4
//
// Constraint: in_features must be a multiple of 512 (block_size). All 27B-Dense
// projection shapes (5120, 22528) satisfy this.
// ────────────────────────────────────────────────────────────────────────────

// Pre-scale x by 1/16, 1/256, 1/4096 so a later 4-nibble dot can use raw
// bitmasked nibbles without per-element shifts. Returns Σ(x) for the bias term.
static inline float qmv_fast_load_x(
    device const float* x,
    thread float* x_thread
) {
    float sum = 0.0f;
    for (int i = 0; i < 16; i += 4) {
        float a0 = x[i + 0];
        float a1 = x[i + 1];
        float a2 = x[i + 2];
        float a3 = x[i + 3];
        sum += a0 + a1 + a2 + a3;
        x_thread[i + 0] = a0;
        x_thread[i + 1] = a1 * (1.0f / 16.0f);
        x_thread[i + 2] = a2 * (1.0f / 256.0f);
        x_thread[i + 3] = a3 * (1.0f / 4096.0f);
    }
    return sum;
}

// Inner dot for one thread's slice of one row (16 nibbles read as 4 ushorts).
// Computes `scale * Σ(unshifted_nib * x_thread) + bias * sum_x` in 16 MACs +
// final 2-op recombine. The unshifted nibble values pair with the pre-scaled
// `x_thread` to reconstruct the correct dequantized product.
static inline float qmv_fast_dot(
    device const uint8_t* w,
    thread const float* x_thread,
    float scale,
    float bias,
    float sum_x
) {
    device const ushort* ws = (device const ushort*)w;
    float accum = 0.0f;
    // 4 iterations × 4 nibbles per ushort = 16 nibbles processed.
    //
    // (C') 2026-05-09 — MLX-aligned tree-reduction pattern. Each iteration's
    // 4 muls fold into a single sub-expression (`a + b + c + d`) before
    // accumulating, cutting the dependency chain on `accum` from 16 sequential
    // adds to 4. Compiler can ILP the muls and inner adds independently.
    // Mirrors `qdot<U, values_per_thread, bits=4>` in MLX 0.31.x
    // `mlx/backend/metal/kernels/quantized.h:235-244`.
    ushort w0 = ws[0];
    ushort w1 = ws[1];
    ushort w2 = ws[2];
    ushort w3 = ws[3];

    accum += (x_thread[ 0] * float(w0 & 0x000fu)
            + x_thread[ 1] * float(w0 & 0x00f0u)
            + x_thread[ 2] * float(w0 & 0x0f00u)
            + x_thread[ 3] * float(w0 & 0xf000u));
    accum += (x_thread[ 4] * float(w1 & 0x000fu)
            + x_thread[ 5] * float(w1 & 0x00f0u)
            + x_thread[ 6] * float(w1 & 0x0f00u)
            + x_thread[ 7] * float(w1 & 0xf000u));
    accum += (x_thread[ 8] * float(w2 & 0x000fu)
            + x_thread[ 9] * float(w2 & 0x00f0u)
            + x_thread[10] * float(w2 & 0x0f00u)
            + x_thread[11] * float(w2 & 0xf000u));
    accum += (x_thread[12] * float(w3 & 0x000fu)
            + x_thread[13] * float(w3 & 0x00f0u)
            + x_thread[14] * float(w3 & 0x0f00u)
            + x_thread[15] * float(w3 & 0xf000u));

    return scale * accum + bias * sum_x;
}

kernel void affine4_qmv_fast(
    device const uint*    packed   [[buffer(0)]],   // [out, in/8]   uint32
    device const ushort*  scales   [[buffer(1)]],   // [out, in/64]  bf16-as-u16
    device const ushort*  biases   [[buffer(2)]],   // [out, in/64]  bf16-as-u16
    device const float*   x        [[buffer(3)]],   // [batch, in]
    device float*         y        [[buffer(4)]],   // [batch, out]
    constant Affine4Dims& dims     [[buffer(5)]],
    constant uint&        batch    [[buffer(6)]],
    uint3 tid          [[threadgroup_position_in_grid]],
    uint  simd_gid     [[simdgroup_index_in_threadgroup]],
    uint  simd_lid     [[thread_index_in_simdgroup]]
) {
    constexpr int VPT = 16;             // values_per_thread
    constexpr int RPS = 4;              // results_per_simdgroup
    constexpr int NSG = 2;              // num_simdgroups
    constexpr int BLK = 512;            // block_size (VPT * SIMD_SIZE=32)
    constexpr int GS  = 64;             // group_size
    constexpr int SCALE_STEP = GS / VPT; // = 4
    constexpr int BYTES_PER_PACK = 4;   // sizeof(uint32)
    constexpr int PACK_FACTOR  = 8;     // 8 nibbles per uint32

    int b_idx = int(tid.x);
    int out_row = int(tid.y) * (NSG * RPS) + int(simd_gid) * RPS;
    if (b_idx >= int(batch) || out_row >= int(dims.out_features)) return;

    int in_vec   = int(dims.in_features);
    int row_w_bytes = in_vec * BYTES_PER_PACK / PACK_FACTOR;  // bytes per row in `packed`
    int row_g       = in_vec / GS;                            // groups per row

    // Per-thread starting pointers. Each thread owns:
    //   - 16 input values (simd_lid * VPT .. +VPT)
    //   - 8 bytes of weights per row (simd_lid * 8 byte slice, since
    //     packs_per_thread*bytes_per_pack = 2*4 = 8)
    //   - one (scale, bias) pair per row, indexed by simd_lid / SCALE_STEP
    device const uint8_t* ws =
        (device const uint8_t*)packed
        + size_t(out_row) * row_w_bytes
        + size_t(simd_lid) * 8;
    int sb_offset = int(simd_lid) / SCALE_STEP;
    device const ushort* scl = scales + size_t(out_row) * row_g + sb_offset;
    device const ushort* bse = biases + size_t(out_row) * row_g + sb_offset;
    device const float*  xp  = x + size_t(b_idx) * in_vec + size_t(simd_lid) * VPT;

    thread float x_thread[VPT];
    thread float result[RPS] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Outer loop: BLK input elements per iteration, advancing all pointers.
    for (int k = 0; k < in_vec; k += BLK) {
        float sum_x = qmv_fast_load_x(xp, x_thread);
        for (int row = 0; row < RPS; row++) {
            device const uint8_t* wl = ws + size_t(row) * row_w_bytes;
            float s = as_type<float>(uint(scl[row * row_g]) << 16);
            float b = as_type<float>(uint(bse[row * row_g]) << 16);
            result[row] += qmv_fast_dot(wl, x_thread, s, b, sum_x);
        }
        ws  += BLK * BYTES_PER_PACK / PACK_FACTOR;  // advance row pointer (256 bytes)
        scl += BLK / GS;                             // advance group pointer (8 groups)
        bse += BLK / GS;
        xp  += BLK;
    }

    // Reduce 32-lane partial dot products per row.
    for (int row = 0; row < RPS; row++) {
        result[row] = simd_sum(result[row]);
        if (simd_lid == 0u) {
            int out_idx = b_idx * int(dims.out_features) + out_row + row;
            y[out_idx] = result[row];
        }
    }
}

// bf16-in/bf16-out variant of `affine4_qmv_fast`. Reads activation as bf16
// (half the BW of f32) and emits result as bf16 directly — eliminates the
// f32→bf16 cast that `Affine4Linear::forward_bf16_out` would otherwise need.
//
// Internal arithmetic stays in f32: each lane upcasts its 16 bf16 inputs to
// f32 in `qmv_fast_load_x_bf16`, the dot product accumulates in f32, and the
// final per-row result is downcast to bf16 only at store time. This keeps
// numerical parity with the f32 kernel — bf16 mantissa loss only on the I/O
// boundaries, never inside the accumulation chain.
//
// Activation savings per call: in_features × 2 bytes (vs 4) on read,
// out_features × 2 bytes on write. Weights are unchanged (already 4-bit).
// Real win shows up when the entire pipeline is bf16: no upstream f32→bf16
// cast feeding `x`, no downstream bf16→f32 cast consuming `y`.
// (C) 2026-05-09 Phase 16 — Vectorized bf16 load + SIMD cast.
// Replaces 16 scalar bf16 loads + 16 scalar BF→F casts with 4 × bfloat4
// vector loads + 4 × float4(bfloat4) SIMD casts. The Apple Silicon GPU
// can issue 64-bit aligned reads + vectorized BF→F conversions in fewer
// instructions, reducing memory transactions and ALU pressure.
//
// **Bit-identical preserved**: addition order is the same as the prior
// scalar loop (4 iterations × `a0 + a1 + a2 + a3` with results
// folded into `sum` left-to-right), and per-element BF→F cast values
// are identical (same hardware conversion). The pre-scaling layout of
// `x_thread` (positions 0,4,8,12 unscaled; 1,5,9,13 / 16; etc.) is
// preserved so downstream `qmv_fast_dot` packs nibbles correctly.
//
// Pointer alignment: caller invokes with `xp = x + b_idx*in + simd_lid*16`.
// `simd_lid*16` × 2 bytes/bf16 = 32-byte multiple → always 8-byte (bfloat4)
// aligned. `in*2` mod 8 == 0 for all decode shapes (in_features ≥ 4).
static inline float qmv_fast_load_x_bf16(
    device const bfloat* x,
    thread float* x_thread
) {
    bfloat4 v0 = *(device const bfloat4*)(x + 0);
    bfloat4 v1 = *(device const bfloat4*)(x + 4);
    bfloat4 v2 = *(device const bfloat4*)(x + 8);
    bfloat4 v3 = *(device const bfloat4*)(x + 12);

    float4 f0 = float4(v0);
    float4 f1 = float4(v1);
    float4 f2 = float4(v2);
    float4 f3 = float4(v3);

    float sum = 0.0f;
    sum += f0.x + f0.y + f0.z + f0.w;
    sum += f1.x + f1.y + f1.z + f1.w;
    sum += f2.x + f2.y + f2.z + f2.w;
    sum += f3.x + f3.y + f3.z + f3.w;

    x_thread[ 0] = f0.x;
    x_thread[ 1] = f0.y * (1.0f / 16.0f);
    x_thread[ 2] = f0.z * (1.0f / 256.0f);
    x_thread[ 3] = f0.w * (1.0f / 4096.0f);
    x_thread[ 4] = f1.x;
    x_thread[ 5] = f1.y * (1.0f / 16.0f);
    x_thread[ 6] = f1.z * (1.0f / 256.0f);
    x_thread[ 7] = f1.w * (1.0f / 4096.0f);
    x_thread[ 8] = f2.x;
    x_thread[ 9] = f2.y * (1.0f / 16.0f);
    x_thread[10] = f2.z * (1.0f / 256.0f);
    x_thread[11] = f2.w * (1.0f / 4096.0f);
    x_thread[12] = f3.x;
    x_thread[13] = f3.y * (1.0f / 16.0f);
    x_thread[14] = f3.z * (1.0f / 256.0f);
    x_thread[15] = f3.w * (1.0f / 4096.0f);

    return sum;
}

kernel void affine4_qmv_fast_bf16in_bf16out(
    device const uint*    packed   [[buffer(0)]],
    device const ushort*  scales   [[buffer(1)]],
    device const ushort*  biases   [[buffer(2)]],
    device const bfloat*  x        [[buffer(3)]],   // [batch, in]  bf16
    device bfloat*        y        [[buffer(4)]],   // [batch, out] bf16
    constant Affine4Dims& dims     [[buffer(5)]],
    constant uint&        batch    [[buffer(6)]],
    uint3 tid          [[threadgroup_position_in_grid]],
    uint  simd_gid     [[simdgroup_index_in_threadgroup]],
    uint  simd_lid     [[thread_index_in_simdgroup]]
) {
    constexpr int VPT = 16;
    constexpr int RPS = 4;
    constexpr int NSG = 2;
    constexpr int BLK = 512;
    constexpr int GS  = 64;
    constexpr int SCALE_STEP = GS / VPT;
    constexpr int BYTES_PER_PACK = 4;
    constexpr int PACK_FACTOR  = 8;

    int b_idx = int(tid.x);
    int out_row = int(tid.y) * (NSG * RPS) + int(simd_gid) * RPS;
    if (b_idx >= int(batch) || out_row >= int(dims.out_features)) return;

    int in_vec   = int(dims.in_features);
    int row_w_bytes = in_vec * BYTES_PER_PACK / PACK_FACTOR;
    int row_g       = in_vec / GS;

    device const uint8_t* ws =
        (device const uint8_t*)packed
        + size_t(out_row) * row_w_bytes
        + size_t(simd_lid) * 8;
    int sb_offset = int(simd_lid) / SCALE_STEP;
    device const ushort* scl = scales + size_t(out_row) * row_g + sb_offset;
    device const ushort* bse = biases + size_t(out_row) * row_g + sb_offset;
    device const bfloat* xp  = x + size_t(b_idx) * in_vec + size_t(simd_lid) * VPT;

    thread float x_thread[VPT];
    thread float result[RPS] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int k = 0; k < in_vec; k += BLK) {
        float sum_x = qmv_fast_load_x_bf16(xp, x_thread);
        for (int row = 0; row < RPS; row++) {
            device const uint8_t* wl = ws + size_t(row) * row_w_bytes;
            float s = as_type<float>(uint(scl[row * row_g]) << 16);
            float b = as_type<float>(uint(bse[row * row_g]) << 16);
            result[row] += qmv_fast_dot(wl, x_thread, s, b, sum_x);
        }
        ws  += BLK * BYTES_PER_PACK / PACK_FACTOR;
        scl += BLK / GS;
        bse += BLK / GS;
        xp  += BLK;
    }

    for (int row = 0; row < RPS; row++) {
        result[row] = simd_sum(result[row]);
        if (simd_lid == 0u) {
            int out_idx = b_idx * int(dims.out_features) + out_row + row;
            y[out_idx] = bfloat(result[row]);
        }
    }
}

// bf16-in/bf16-out fused-residual variant: same as `affine4_qmv_fast_residual`
// but with bf16 activation/residual/output. Closes the B.10 σ-NEGATIVE root cause
// — under bf16 carrier, the f32 fused-residual fast-path can't fire, splitting
// the dispatch into non-fused matmul + separate broadcast_add. This variant
// restores fusion on the bf16 chain.
kernel void affine4_qmv_fast_bf16in_bf16out_residual(
    device const uint*    packed   [[buffer(0)]],
    device const ushort*  scales   [[buffer(1)]],
    device const ushort*  biases   [[buffer(2)]],
    device const bfloat*  x        [[buffer(3)]],   // [batch, in]  bf16
    device const bfloat*  residual [[buffer(4)]],   // [batch, out] bf16
    device bfloat*        y        [[buffer(5)]],   // [batch, out] bf16
    constant Affine4Dims& dims     [[buffer(6)]],
    constant uint&        batch    [[buffer(7)]],
    uint3 tid          [[threadgroup_position_in_grid]],
    uint  simd_gid     [[simdgroup_index_in_threadgroup]],
    uint  simd_lid     [[thread_index_in_simdgroup]]
) {
    constexpr int VPT = 16;
    constexpr int RPS = 4;
    constexpr int NSG = 2;
    constexpr int BLK = 512;
    constexpr int GS  = 64;
    constexpr int SCALE_STEP = GS / VPT;
    constexpr int BYTES_PER_PACK = 4;
    constexpr int PACK_FACTOR  = 8;

    int b_idx = int(tid.x);
    int out_row = int(tid.y) * (NSG * RPS) + int(simd_gid) * RPS;
    if (b_idx >= int(batch) || out_row >= int(dims.out_features)) return;

    int in_vec   = int(dims.in_features);
    int row_w_bytes = in_vec * BYTES_PER_PACK / PACK_FACTOR;
    int row_g       = in_vec / GS;

    device const uint8_t* ws =
        (device const uint8_t*)packed
        + size_t(out_row) * row_w_bytes
        + size_t(simd_lid) * 8;
    int sb_offset = int(simd_lid) / SCALE_STEP;
    device const ushort* scl = scales + size_t(out_row) * row_g + sb_offset;
    device const ushort* bse = biases + size_t(out_row) * row_g + sb_offset;
    device const bfloat* xp  = x + size_t(b_idx) * in_vec + size_t(simd_lid) * VPT;

    thread float x_thread[VPT];
    thread float result[RPS] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int k = 0; k < in_vec; k += BLK) {
        float sum_x = qmv_fast_load_x_bf16(xp, x_thread);
        for (int row = 0; row < RPS; row++) {
            device const uint8_t* wl = ws + size_t(row) * row_w_bytes;
            float s = as_type<float>(uint(scl[row * row_g]) << 16);
            float b = as_type<float>(uint(bse[row * row_g]) << 16);
            result[row] += qmv_fast_dot(wl, x_thread, s, b, sum_x);
        }
        ws  += BLK * BYTES_PER_PACK / PACK_FACTOR;
        scl += BLK / GS;
        bse += BLK / GS;
        xp  += BLK;
    }

    for (int row = 0; row < RPS; row++) {
        result[row] = simd_sum(result[row]);
        if (simd_lid == 0u) {
            int out_idx = b_idx * int(dims.out_features) + out_row + row;
            y[out_idx] = bfloat(result[row] + float(residual[out_idx]));
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Dims for fused gate+up SiLU mul kernel.  Mirrors `MxFp4GateUpSiluMulDims`.
// `inter` = intermediate width (= half of the gate_up weight rows since gate
// is concatenated above up). `batch` = activation batch.
// ────────────────────────────────────────────────────────────────────────────
struct Affine4GateUpSiluMulDims {
    uint inter;
    uint in_features;
    uint batch;
};

// Fused RmsNorm + qmv_fast: reads RAW x and the input-layernorm weight,
// computes inv_rms = 1/sqrt(mean(x^2) + eps) cooperatively across the TG,
// then runs the qmv_fast inner loop with `x' = x * inv_rms * rms_w` folded
// into the pre-scaling. Saves the entire Candle RmsNorm dispatch chain
// (sqr → mean → sqrt → recip → 2× broadcast_mul) per qkv/gate_up call.
//
// 64 layers × 2 fusion sites = 128 RmsNorm dispatches absorbed per token.
//
// Two passes over x (read twice from device) — cheap (~20 KB of activation
// vs many MB of weights), avoids any TG memory cache.
kernel void affine4_qmv_fast_rmsnorm(
    device const uint*    packed   [[buffer(0)]],
    device const ushort*  scales   [[buffer(1)]],
    device const ushort*  biases   [[buffer(2)]],
    device const float*   x_raw    [[buffer(3)]],   // [batch, in]
    device const float*   rms_w    [[buffer(4)]],   // [in]
    device float*         y        [[buffer(5)]],
    constant Affine4MatmulRmsnormDims& dims [[buffer(6)]],
    constant uint&        batch    [[buffer(7)]],
    threadgroup float*    sg_partials [[threadgroup(0)]],   // [NSG] = 2 floats
    uint3 tid          [[threadgroup_position_in_grid]],
    uint  simd_gid     [[simdgroup_index_in_threadgroup]],
    uint  simd_lid     [[thread_index_in_simdgroup]]
) {
    constexpr int VPT = 16;
    constexpr int RPS = 4;
    constexpr int NSG = 2;
    constexpr int BLK = 512;
    constexpr int GS  = 64;
    constexpr int SCALE_STEP = GS / VPT;
    constexpr int BYTES_PER_PACK = 4;
    constexpr int PACK_FACTOR  = 8;

    int b_idx = int(tid.x);
    int out_row = int(tid.y) * (NSG * RPS) + int(simd_gid) * RPS;
    if (b_idx >= int(batch) || out_row >= int(dims.out_features)) return;

    int in_vec = int(dims.in_features);

    // ── Phase A: cooperative Σ(x^2) across all TG threads ──
    // Each thread covers `simd_lid * VPT .. (simd_lid+1) * VPT` per outer iter,
    // 32 threads × VPT = 512 elements per outer = block_size; full input
    // covered after `in_vec / BLK` outer iters. Both simdgroups iterate the
    // same data in parallel — they will combine to produce 2× the true sum
    // (each lane reads independently); divide once at the end.
    device const float* xp_a = x_raw + size_t(b_idx) * in_vec + size_t(simd_lid) * VPT;
    float sumsq = 0.0f;
    for (int k = 0; k < in_vec; k += BLK) {
        for (int i = 0; i < VPT; i++) {
            float v = xp_a[i];
            sumsq = fma(v, v, sumsq);
        }
        xp_a += BLK;
    }
    sumsq = simd_sum(sumsq);
    if (simd_lid == 0u) sg_partials[simd_gid] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Both simdgroups computed identical Σx^2 (their inputs are the SAME).
    // Take simdgroup 0's partial only — no doubling.
    float total_sumsq = sg_partials[0];
    float inv_rms = rsqrt(total_sumsq / float(in_vec) + dims.rms_eps);

    // ── Phase B: qmv_fast with x_thread folded with inv_rms * rms_w ──
    int row_w_bytes = in_vec * BYTES_PER_PACK / PACK_FACTOR;
    int row_g = in_vec / GS;
    device const uint8_t* ws =
        (device const uint8_t*)packed
        + size_t(out_row) * row_w_bytes
        + size_t(simd_lid) * 8;
    int sb_offset = int(simd_lid) / SCALE_STEP;
    device const ushort* scl = scales + size_t(out_row) * row_g + sb_offset;
    device const ushort* bse = biases + size_t(out_row) * row_g + sb_offset;
    device const float*  xp  = x_raw + size_t(b_idx) * in_vec + size_t(simd_lid) * VPT;
    device const float*  rwp = rms_w + size_t(simd_lid) * VPT;

    thread float x_thread[VPT];
    thread float result[RPS] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int k = 0; k < in_vec; k += BLK) {
        // Compute normalized + pre-scaled x_thread, plus sum of normalized x.
        float sum_xn = 0.0f;
        for (int i = 0; i < VPT; i += 4) {
            float v0 = xp[i + 0] * rwp[i + 0] * inv_rms;
            float v1 = xp[i + 1] * rwp[i + 1] * inv_rms;
            float v2 = xp[i + 2] * rwp[i + 2] * inv_rms;
            float v3 = xp[i + 3] * rwp[i + 3] * inv_rms;
            sum_xn += v0 + v1 + v2 + v3;
            x_thread[i + 0] = v0;
            x_thread[i + 1] = v1 * (1.0f / 16.0f);
            x_thread[i + 2] = v2 * (1.0f / 256.0f);
            x_thread[i + 3] = v3 * (1.0f / 4096.0f);
        }
        for (int row = 0; row < RPS; row++) {
            device const uint8_t* wl = ws + size_t(row) * row_w_bytes;
            float s = as_type<float>(uint(scl[row * row_g]) << 16);
            float b = as_type<float>(uint(bse[row * row_g]) << 16);
            result[row] += qmv_fast_dot(wl, x_thread, s, b, sum_xn);
        }
        ws  += BLK * BYTES_PER_PACK / PACK_FACTOR;
        scl += BLK / GS;
        bse += BLK / GS;
        xp  += BLK;
        rwp += BLK;
    }

    for (int row = 0; row < RPS; row++) {
        result[row] = simd_sum(result[row]);
        if (simd_lid == 0u) {
            int out_idx = b_idx * int(dims.out_features) + out_row + row;
            y[out_idx] = result[row];
        }
    }
}

// Fused residual: same as qmv_fast but adds `residual[batch, out]` element-wise
// to the output. Saves one downstream broadcast_add dispatch per call when the
// matmul output feeds a residual addition (e.g. `out = h + down_proj(x)` in
// the 27B-Dense MLP block tail; or `out = h + o_proj(...)` for full-attn).
kernel void affine4_qmv_fast_residual(
    device const uint*    packed   [[buffer(0)]],
    device const ushort*  scales   [[buffer(1)]],
    device const ushort*  biases   [[buffer(2)]],
    device const float*   x        [[buffer(3)]],
    device const float*   residual [[buffer(4)]],   // [batch, out]
    device float*         y        [[buffer(5)]],
    constant Affine4Dims& dims     [[buffer(6)]],
    constant uint&        batch    [[buffer(7)]],
    uint3 tid          [[threadgroup_position_in_grid]],
    uint  simd_gid     [[simdgroup_index_in_threadgroup]],
    uint  simd_lid     [[thread_index_in_simdgroup]]
) {
    constexpr int VPT = 16;
    constexpr int RPS = 4;
    constexpr int NSG = 2;
    constexpr int BLK = 512;
    constexpr int GS  = 64;
    constexpr int SCALE_STEP = GS / VPT;
    constexpr int BYTES_PER_PACK = 4;
    constexpr int PACK_FACTOR  = 8;

    int b_idx = int(tid.x);
    int out_row = int(tid.y) * (NSG * RPS) + int(simd_gid) * RPS;
    if (b_idx >= int(batch) || out_row >= int(dims.out_features)) return;

    int in_vec   = int(dims.in_features);
    int row_w_bytes = in_vec * BYTES_PER_PACK / PACK_FACTOR;
    int row_g       = in_vec / GS;

    device const uint8_t* ws =
        (device const uint8_t*)packed
        + size_t(out_row) * row_w_bytes
        + size_t(simd_lid) * 8;
    int sb_offset = int(simd_lid) / SCALE_STEP;
    device const ushort* scl = scales + size_t(out_row) * row_g + sb_offset;
    device const ushort* bse = biases + size_t(out_row) * row_g + sb_offset;
    device const float*  xp  = x + size_t(b_idx) * in_vec + size_t(simd_lid) * VPT;

    thread float x_thread[VPT];
    thread float result[RPS] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int k = 0; k < in_vec; k += BLK) {
        float sum_x = qmv_fast_load_x(xp, x_thread);
        for (int row = 0; row < RPS; row++) {
            device const uint8_t* wl = ws + size_t(row) * row_w_bytes;
            float s = as_type<float>(uint(scl[row * row_g]) << 16);
            float b = as_type<float>(uint(bse[row * row_g]) << 16);
            result[row] += qmv_fast_dot(wl, x_thread, s, b, sum_x);
        }
        ws  += BLK * BYTES_PER_PACK / PACK_FACTOR;
        scl += BLK / GS;
        bse += BLK / GS;
        xp  += BLK;
    }

    for (int row = 0; row < RPS; row++) {
        result[row] = simd_sum(result[row]);
        if (simd_lid == 0u) {
            int out_idx = b_idx * int(dims.out_features) + out_row + row;
            y[out_idx] = result[row] + residual[out_idx];
        }
    }
}

// V3 kernel: simdgroup-cooperative reduction + threadgroup-shared activation cache.
// Mirror of `mxfp4_matmul_f32_v3`. ROWS_PER_TG=8 (8 simdgroups × 32 lanes = 256 threads).
// Each simdgroup cooperates on ONE output row; lanes partition the `groups` axis,
// adjacent lanes read consecutive 16-byte uint4 chunks (coalesced). `simd_sum`
// reduces 32-lane partial dot products. Activation `x` is staged into threadgroup
// memory once per threadgroup → effectively L1-resident for the inner loop.
//
// Constraint: `in_features * 4` bytes ≤ 32 KB TG memory. Dispatcher must check
// `in_features <= 8192` before binding this kernel; for larger `in_features`
// (e.g. 27B-Dense's `down_proj` with in=intermediate=~22K) fall back to v1.
kernel void affine4_matmul_f32_v3(
    device const uint*    packed   [[buffer(0)]],
    device const ushort*  scales   [[buffer(1)]],
    device const ushort*  biases   [[buffer(2)]],
    device const float*   x        [[buffer(3)]],
    device float*         y        [[buffer(4)]],
    constant Affine4Dims& dims     [[buffer(5)]],
    constant uint&        batch    [[buffer(6)]],
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

    uint groups         = dims.in_features / AFFINE4_GROUP_SIZE;
    uint words_per_row  = dims.in_features / 8u;
    uint x_row_base     = b * dims.in_features;

    // Cooperative stage of x[b, :] into threadgroup memory.
    for (uint i = tid_in_tg; i < dims.in_features; i += THREADS_PER_TG) {
        x_shared[i] = x[x_row_base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= dims.out_features) { return; }

    uint word_row_base  = row * words_per_row;
    uint scale_row_base = row * groups;

    float acc = 0.0f;
    for (uint g = sg_lane; g < groups; g += 32u) {
        uint sb = uint(scales[scale_row_base + g]) << 16;
        uint bb = uint(biases[scale_row_base + g]) << 16;
        float s = as_type<float>(sb);
        float bi = as_type<float>(bb);

        uint word_base = word_row_base + g * AFFINE4_WORDS_PER_GROUP;
        uint x_base    = g * AFFINE4_GROUP_SIZE;

        // 8 words per group = 2 × uint4 loads.
        uint4 ws0 = *((device const uint4*)(packed + word_base));
        uint4 ws1 = *((device const uint4*)(packed + word_base + 4u));

        // 64 floats per group = 16 × float4 loads from threadgroup memory.
        threadgroup const float4* x4 =
            (threadgroup const float4*)(x_shared + x_base);
        float4 xa0 = x4[ 0]; float4 xa1 = x4[ 1];
        float4 xa2 = x4[ 2]; float4 xa3 = x4[ 3];
        float4 xa4 = x4[ 4]; float4 xa5 = x4[ 5];
        float4 xa6 = x4[ 6]; float4 xa7 = x4[ 7];
        float4 xa8 = x4[ 8]; float4 xa9 = x4[ 9];
        float4 xaA = x4[10]; float4 xaB = x4[11];
        float4 xaC = x4[12]; float4 xaD = x4[13];
        float4 xaE = x4[14]; float4 xaF = x4[15];

        // Per-element dequant: w = nib * s + bi.  Per-element accumulate: acc += w * x.
        // 64 FMAs per group, fully unrolled.
        acc += (float((ws0.x      ) & 0xFu) * s + bi) * xa0.x;
        acc += (float((ws0.x >>  4) & 0xFu) * s + bi) * xa0.y;
        acc += (float((ws0.x >>  8) & 0xFu) * s + bi) * xa0.z;
        acc += (float((ws0.x >> 12) & 0xFu) * s + bi) * xa0.w;
        acc += (float((ws0.x >> 16) & 0xFu) * s + bi) * xa1.x;
        acc += (float((ws0.x >> 20) & 0xFu) * s + bi) * xa1.y;
        acc += (float((ws0.x >> 24) & 0xFu) * s + bi) * xa1.z;
        acc += (float((ws0.x >> 28) & 0xFu) * s + bi) * xa1.w;

        acc += (float((ws0.y      ) & 0xFu) * s + bi) * xa2.x;
        acc += (float((ws0.y >>  4) & 0xFu) * s + bi) * xa2.y;
        acc += (float((ws0.y >>  8) & 0xFu) * s + bi) * xa2.z;
        acc += (float((ws0.y >> 12) & 0xFu) * s + bi) * xa2.w;
        acc += (float((ws0.y >> 16) & 0xFu) * s + bi) * xa3.x;
        acc += (float((ws0.y >> 20) & 0xFu) * s + bi) * xa3.y;
        acc += (float((ws0.y >> 24) & 0xFu) * s + bi) * xa3.z;
        acc += (float((ws0.y >> 28) & 0xFu) * s + bi) * xa3.w;

        acc += (float((ws0.z      ) & 0xFu) * s + bi) * xa4.x;
        acc += (float((ws0.z >>  4) & 0xFu) * s + bi) * xa4.y;
        acc += (float((ws0.z >>  8) & 0xFu) * s + bi) * xa4.z;
        acc += (float((ws0.z >> 12) & 0xFu) * s + bi) * xa4.w;
        acc += (float((ws0.z >> 16) & 0xFu) * s + bi) * xa5.x;
        acc += (float((ws0.z >> 20) & 0xFu) * s + bi) * xa5.y;
        acc += (float((ws0.z >> 24) & 0xFu) * s + bi) * xa5.z;
        acc += (float((ws0.z >> 28) & 0xFu) * s + bi) * xa5.w;

        acc += (float((ws0.w      ) & 0xFu) * s + bi) * xa6.x;
        acc += (float((ws0.w >>  4) & 0xFu) * s + bi) * xa6.y;
        acc += (float((ws0.w >>  8) & 0xFu) * s + bi) * xa6.z;
        acc += (float((ws0.w >> 12) & 0xFu) * s + bi) * xa6.w;
        acc += (float((ws0.w >> 16) & 0xFu) * s + bi) * xa7.x;
        acc += (float((ws0.w >> 20) & 0xFu) * s + bi) * xa7.y;
        acc += (float((ws0.w >> 24) & 0xFu) * s + bi) * xa7.z;
        acc += (float((ws0.w >> 28) & 0xFu) * s + bi) * xa7.w;

        acc += (float((ws1.x      ) & 0xFu) * s + bi) * xa8.x;
        acc += (float((ws1.x >>  4) & 0xFu) * s + bi) * xa8.y;
        acc += (float((ws1.x >>  8) & 0xFu) * s + bi) * xa8.z;
        acc += (float((ws1.x >> 12) & 0xFu) * s + bi) * xa8.w;
        acc += (float((ws1.x >> 16) & 0xFu) * s + bi) * xa9.x;
        acc += (float((ws1.x >> 20) & 0xFu) * s + bi) * xa9.y;
        acc += (float((ws1.x >> 24) & 0xFu) * s + bi) * xa9.z;
        acc += (float((ws1.x >> 28) & 0xFu) * s + bi) * xa9.w;

        acc += (float((ws1.y      ) & 0xFu) * s + bi) * xaA.x;
        acc += (float((ws1.y >>  4) & 0xFu) * s + bi) * xaA.y;
        acc += (float((ws1.y >>  8) & 0xFu) * s + bi) * xaA.z;
        acc += (float((ws1.y >> 12) & 0xFu) * s + bi) * xaA.w;
        acc += (float((ws1.y >> 16) & 0xFu) * s + bi) * xaB.x;
        acc += (float((ws1.y >> 20) & 0xFu) * s + bi) * xaB.y;
        acc += (float((ws1.y >> 24) & 0xFu) * s + bi) * xaB.z;
        acc += (float((ws1.y >> 28) & 0xFu) * s + bi) * xaB.w;

        acc += (float((ws1.z      ) & 0xFu) * s + bi) * xaC.x;
        acc += (float((ws1.z >>  4) & 0xFu) * s + bi) * xaC.y;
        acc += (float((ws1.z >>  8) & 0xFu) * s + bi) * xaC.z;
        acc += (float((ws1.z >> 12) & 0xFu) * s + bi) * xaC.w;
        acc += (float((ws1.z >> 16) & 0xFu) * s + bi) * xaD.x;
        acc += (float((ws1.z >> 20) & 0xFu) * s + bi) * xaD.y;
        acc += (float((ws1.z >> 24) & 0xFu) * s + bi) * xaD.z;
        acc += (float((ws1.z >> 28) & 0xFu) * s + bi) * xaD.w;

        acc += (float((ws1.w      ) & 0xFu) * s + bi) * xaE.x;
        acc += (float((ws1.w >>  4) & 0xFu) * s + bi) * xaE.y;
        acc += (float((ws1.w >>  8) & 0xFu) * s + bi) * xaE.z;
        acc += (float((ws1.w >> 12) & 0xFu) * s + bi) * xaE.w;
        acc += (float((ws1.w >> 16) & 0xFu) * s + bi) * xaF.x;
        acc += (float((ws1.w >> 20) & 0xFu) * s + bi) * xaF.y;
        acc += (float((ws1.w >> 24) & 0xFu) * s + bi) * xaF.z;
        acc += (float((ws1.w >> 28) & 0xFu) * s + bi) * xaF.w;
    }

    // Reduce 32-lane partial sums to a single value per simdgroup.
    acc = simd_sum(acc);

    if (sg_lane == 0u) {
        y[b * dims.out_features + row] = acc;
    }
}

kernel void affine4_matmul_f32(
    device const uint*    packed  [[buffer(0)]],
    device const ushort*  scales  [[buffer(1)]],
    device const ushort*  biases  [[buffer(2)]],
    device const float*   x       [[buffer(3)]],
    device float*         y       [[buffer(4)]],
    constant Affine4Dims& dims    [[buffer(5)]],
    constant uint&        batch   [[buffer(6)]],
    uint2 gid                     [[thread_position_in_grid]]
) {
    uint row = gid.x;
    uint b   = gid.y;
    if (row >= dims.out_features || b >= batch) { return; }

    uint groups         = dims.in_features / AFFINE4_GROUP_SIZE;
    uint words_per_row  = dims.in_features / 8u;
    uint word_row_base  = row * words_per_row;
    uint scale_row_base = row * groups;
    uint x_row_base     = b * dims.in_features;

    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        float s = bf16_to_f32_device(scales[scale_row_base + g]);
        float bi = bf16_to_f32_device(biases[scale_row_base + g]);

        uint word_base = word_row_base + g * AFFINE4_WORDS_PER_GROUP;
        uint x_base    = x_row_base + g * AFFINE4_GROUP_SIZE;
        for (uint w = 0; w < AFFINE4_WORDS_PER_GROUP; ++w) {
            uint word = packed[word_base + w];
            for (uint i = 0; i < 8u; ++i) {
                float v = float((word >> (i * 4u)) & 0xFu);
                float dequant = v * s + bi;
                acc += dequant * x[x_base + w * 8u + i];
            }
        }
    }
    y[b * dims.out_features + row] = acc;
}

// ────────────────────────────────────────────────────────────────────────────
// Fused variant: residual add. y[row] = sum + residual[row]. Lane-0 store
// reads the residual value directly so we avoid a separate elementwise pass.
// ────────────────────────────────────────────────────────────────────────────
kernel void affine4_matmul_f32_v3_residual(
    device const uint*    packed   [[buffer(0)]],
    device const ushort*  scales   [[buffer(1)]],
    device const ushort*  biases   [[buffer(2)]],
    device const float*   x        [[buffer(3)]],
    device const float*   residual [[buffer(4)]],
    device float*         y        [[buffer(5)]],
    constant Affine4Dims& dims     [[buffer(6)]],
    constant uint&        batch    [[buffer(7)]],
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
    uint groups = dims.in_features / AFFINE4_GROUP_SIZE;
    uint words_per_row = dims.in_features / 8u;
    uint x_row_base = b * dims.in_features;
    for (uint i = tid_in_tg; i < dims.in_features; i += THREADS_PER_TG) {
        x_shared[i] = x[x_row_base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= dims.out_features) { return; }
    uint word_row_base = row * words_per_row;
    uint scale_row_base = row * groups;
    float acc = 0.0f;
    for (uint g = sg_lane; g < groups; g += 32u) {
        uint sb = uint(scales[scale_row_base + g]) << 16;
        uint bb = uint(biases[scale_row_base + g]) << 16;
        float s = as_type<float>(sb);
        float bi = as_type<float>(bb);
        uint word_base = word_row_base + g * AFFINE4_WORDS_PER_GROUP;
        uint x_base = g * AFFINE4_GROUP_SIZE;
        uint4 ws0 = *((device const uint4*)(packed + word_base));
        uint4 ws1 = *((device const uint4*)(packed + word_base + 4u));
        threadgroup const float4* x4 =
            (threadgroup const float4*)(x_shared + x_base);
        acc += affine4_group_dot_tg(ws0, ws1, s, bi, x4);
    }
    acc = simd_sum(acc);
    if (sg_lane == 0u) {
        uint y_idx = b * dims.out_features + row;
        y[y_idx] = acc + residual[y_idx];
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Fused variant: f32 input, bf16 output. Same compute, narrow store.
// ────────────────────────────────────────────────────────────────────────────
kernel void affine4_matmul_f32in_bf16out_v3(
    device const uint*    packed   [[buffer(0)]],
    device const ushort*  scales   [[buffer(1)]],
    device const ushort*  biases   [[buffer(2)]],
    device const float*   x        [[buffer(3)]],
    device bfloat*        y        [[buffer(4)]],
    constant Affine4Dims& dims     [[buffer(5)]],
    constant uint&        batch    [[buffer(6)]],
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
    uint groups = dims.in_features / AFFINE4_GROUP_SIZE;
    uint words_per_row = dims.in_features / 8u;
    uint x_row_base = b * dims.in_features;
    for (uint i = tid_in_tg; i < dims.in_features; i += THREADS_PER_TG) {
        x_shared[i] = x[x_row_base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= dims.out_features) { return; }
    uint word_row_base = row * words_per_row;
    uint scale_row_base = row * groups;
    float acc = 0.0f;
    for (uint g = sg_lane; g < groups; g += 32u) {
        uint sb = uint(scales[scale_row_base + g]) << 16;
        uint bb = uint(biases[scale_row_base + g]) << 16;
        float s = as_type<float>(sb);
        float bi = as_type<float>(bb);
        uint word_base = word_row_base + g * AFFINE4_WORDS_PER_GROUP;
        uint x_base = g * AFFINE4_GROUP_SIZE;
        uint4 ws0 = *((device const uint4*)(packed + word_base));
        uint4 ws1 = *((device const uint4*)(packed + word_base + 4u));
        threadgroup const float4* x4 =
            (threadgroup const float4*)(x_shared + x_base);
        acc += affine4_group_dot_tg(ws0, ws1, s, bi, x4);
    }
    acc = simd_sum(acc);
    if (sg_lane == 0u) {
        y[b * dims.out_features + row] = bfloat(acc);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Fused variant: bf16 input, f32 output. Activation read narrows to bf16.
// ────────────────────────────────────────────────────────────────────────────
kernel void affine4_matmul_bf16in_f32out_v3(
    device const uint*    packed   [[buffer(0)]],
    device const ushort*  scales   [[buffer(1)]],
    device const ushort*  biases   [[buffer(2)]],
    device const bfloat*  x        [[buffer(3)]],
    device float*         y        [[buffer(4)]],
    constant Affine4Dims& dims     [[buffer(5)]],
    constant uint&        batch    [[buffer(6)]],
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
    uint groups = dims.in_features / AFFINE4_GROUP_SIZE;
    uint words_per_row = dims.in_features / 8u;
    uint x_row_base = b * dims.in_features;
    // Stage bf16 → f32 widen into TG memory (same as MXFP4 bf16-in v3).
    for (uint i = tid_in_tg; i < dims.in_features; i += THREADS_PER_TG) {
        x_shared[i] = float(x[x_row_base + i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= dims.out_features) { return; }
    uint word_row_base = row * words_per_row;
    uint scale_row_base = row * groups;
    float acc = 0.0f;
    for (uint g = sg_lane; g < groups; g += 32u) {
        uint sb = uint(scales[scale_row_base + g]) << 16;
        uint bb = uint(biases[scale_row_base + g]) << 16;
        float s = as_type<float>(sb);
        float bi = as_type<float>(bb);
        uint word_base = word_row_base + g * AFFINE4_WORDS_PER_GROUP;
        uint x_base = g * AFFINE4_GROUP_SIZE;
        uint4 ws0 = *((device const uint4*)(packed + word_base));
        uint4 ws1 = *((device const uint4*)(packed + word_base + 4u));
        threadgroup const float4* x4 =
            (threadgroup const float4*)(x_shared + x_base);
        acc += affine4_group_dot_tg(ws0, ws1, s, bi, x4);
    }
    acc = simd_sum(acc);
    if (sg_lane == 0u) {
        y[b * dims.out_features + row] = acc;
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Fused gate+up matmul + SiLU(gate)*up. Single dispatch replaces:
//   gate = x @ W_gate^T (matmul)
//   up   = x @ W_up^T   (matmul)
//   silu(gate) * up     (2 elemwise ops)
// → 4 dispatches collapsed to 1. Critical hot-path for Dense MLP since it's
// invoked once per layer × 64 layers × every decode step.
// ────────────────────────────────────────────────────────────────────────────
kernel void affine4_gate_up_silu_mul_f32_v3(
    device const uint*    packed   [[buffer(0)]],   // [2*inter, in/8]
    device const ushort*  scales   [[buffer(1)]],   // [2*inter, in/64]
    device const ushort*  biases   [[buffer(2)]],   // [2*inter, in/64]
    device const float*   x        [[buffer(3)]],   // [batch, in]
    device float*         y        [[buffer(4)]],   // [batch, inter]
    constant Affine4GateUpSiluMulDims& dims [[buffer(5)]],
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
    uint groups = dims.in_features / AFFINE4_GROUP_SIZE;
    uint words_per_row = dims.in_features / 8u;
    uint x_row_base = b * dims.in_features;
    for (uint i = tid_in_tg; i < dims.in_features; i += THREADS_PER_TG) {
        x_shared[i] = x[x_row_base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= dims.inter) { return; }

    // Gate row at index `row`; up row at index `row + inter` (axis-0 concat).
    uint gate_word_base = row * words_per_row;
    uint gate_scale_base = row * groups;
    uint up_word_base = (row + dims.inter) * words_per_row;
    uint up_scale_base = (row + dims.inter) * groups;

    float acc_gate = 0.0f;
    float acc_up = 0.0f;
    for (uint g = sg_lane; g < groups; g += 32u) {
        uint x_base = g * AFFINE4_GROUP_SIZE;
        threadgroup const float4* x4 =
            (threadgroup const float4*)(x_shared + x_base);

        // Gate dot
        {
            uint sb = uint(scales[gate_scale_base + g]) << 16;
            uint bb = uint(biases[gate_scale_base + g]) << 16;
            float s = as_type<float>(sb);
            float bi = as_type<float>(bb);
            uint4 ws0 = *((device const uint4*)(packed + gate_word_base + g * AFFINE4_WORDS_PER_GROUP));
            uint4 ws1 = *((device const uint4*)(packed + gate_word_base + g * AFFINE4_WORDS_PER_GROUP + 4u));
            acc_gate += affine4_group_dot_tg(ws0, ws1, s, bi, x4);
        }
        // Up dot
        {
            uint sb = uint(scales[up_scale_base + g]) << 16;
            uint bb = uint(biases[up_scale_base + g]) << 16;
            float s = as_type<float>(sb);
            float bi = as_type<float>(bb);
            uint4 ws0 = *((device const uint4*)(packed + up_word_base + g * AFFINE4_WORDS_PER_GROUP));
            uint4 ws1 = *((device const uint4*)(packed + up_word_base + g * AFFINE4_WORDS_PER_GROUP + 4u));
            acc_up += affine4_group_dot_tg(ws0, ws1, s, bi, x4);
        }
    }
    acc_gate = simd_sum(acc_gate);
    acc_up = simd_sum(acc_up);
    if (sg_lane == 0u) {
        float silu_g = acc_gate / (1.0f + metal::exp(-acc_gate));
        y[b * dims.inter + row] = silu_g * acc_up;
    }
}

// ────────────────────────────────────────────────────────────────────────────
// RmsNorm-fused matmul (Lever R1).
//
// Collapses `RmsNorm(x_raw) → matmul(W)` into one dispatch:
//   Phase 1: stage x_raw into TG memory + accumulate per-thread sum(x²)
//   Phase 2: hierarchical reduction → inv_rms = rsqrt(mean_sq + eps)
//   Phase 3: apply x_normed = x_raw * rms_weight * inv_rms in-place on TG mem
//   Phase 4: standard v3 matmul body reading from x_shared
//
// Saves one full RmsNorm dispatch + the cross-op cast/store/reload per layer.
// 64 layers × 2 RmsNorms = 128 dispatches/token saved → ~+15-20% throughput.
// (Affine4MatmulRmsnormDims declared near top of file.)
// ────────────────────────────────────────────────────────────────────────────

kernel void affine4_matmul_f32_v3_rmsnorm(
    device const uint*    packed     [[buffer(0)]],
    device const ushort*  scales     [[buffer(1)]],
    device const ushort*  biases     [[buffer(2)]],
    device const float*   x          [[buffer(3)]],
    device const float*   rms_weight [[buffer(4)]],
    device float*         y          [[buffer(5)]],
    constant Affine4MatmulRmsnormDims& dims [[buffer(6)]],
    constant uint&        batch      [[buffer(7)]],
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

    uint groups        = dims.in_features / AFFINE4_GROUP_SIZE;
    uint words_per_row = dims.in_features / 8u;
    uint x_row_base    = b * dims.in_features;

    // Phase 1: stage raw x + accumulate per-thread sum(x²).
    float sum_sq = 0.0f;
    for (uint i = tid_in_tg; i < dims.in_features; i += THREADS_PER_TG) {
        float v = x[x_row_base + i];
        x_shared[i] = v;
        sum_sq = fma(v, v, sum_sq);
    }
    sum_sq = simd_sum(sum_sq);
    if (sg_lane == 0u) { reduce_buf[sg_id] = sum_sq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: SG 0 reduces 8 partials → inv_rms.
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

    // Phase 3: apply weight × inv_rms in-place on x_shared.
    for (uint i = tid_in_tg; i < dims.in_features; i += THREADS_PER_TG) {
        x_shared[i] = x_shared[i] * rms_weight[i] * inv_rms;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: standard v3 matmul body, reading from x_shared.
    if (row >= dims.out_features) { return; }
    uint word_row_base = row * words_per_row;
    uint scale_row_base = row * groups;
    float acc = 0.0f;
    for (uint g = sg_lane; g < groups; g += 32u) {
        uint sb = uint(scales[scale_row_base + g]) << 16;
        uint bb = uint(biases[scale_row_base + g]) << 16;
        float s = as_type<float>(sb);
        float bi = as_type<float>(bb);
        uint word_base = word_row_base + g * AFFINE4_WORDS_PER_GROUP;
        uint x_base = g * AFFINE4_GROUP_SIZE;
        uint4 ws0 = *((device const uint4*)(packed + word_base));
        uint4 ws1 = *((device const uint4*)(packed + word_base + 4u));
        threadgroup const float4* x4 =
            (threadgroup const float4*)(x_shared + x_base);
        acc += affine4_group_dot_tg(ws0, ws1, s, bi, x4);
    }
    acc = simd_sum(acc);
    if (sg_lane == 0u) {
        y[b * dims.out_features + row] = acc;
    }
}

// Same fused gate+up SiLU mul, bf16 output store.
kernel void affine4_gate_up_silu_mul_f32in_bf16out_v3(
    device const uint*    packed   [[buffer(0)]],
    device const ushort*  scales   [[buffer(1)]],
    device const ushort*  biases   [[buffer(2)]],
    device const float*   x        [[buffer(3)]],
    device bfloat*        y        [[buffer(4)]],
    constant Affine4GateUpSiluMulDims& dims [[buffer(5)]],
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
    uint groups = dims.in_features / AFFINE4_GROUP_SIZE;
    uint words_per_row = dims.in_features / 8u;
    uint x_row_base = b * dims.in_features;
    for (uint i = tid_in_tg; i < dims.in_features; i += THREADS_PER_TG) {
        x_shared[i] = x[x_row_base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= dims.inter) { return; }
    uint gate_word_base = row * words_per_row;
    uint gate_scale_base = row * groups;
    uint up_word_base = (row + dims.inter) * words_per_row;
    uint up_scale_base = (row + dims.inter) * groups;
    float acc_gate = 0.0f;
    float acc_up = 0.0f;
    for (uint g = sg_lane; g < groups; g += 32u) {
        uint x_base = g * AFFINE4_GROUP_SIZE;
        threadgroup const float4* x4 = (threadgroup const float4*)(x_shared + x_base);
        {
            uint sb = uint(scales[gate_scale_base + g]) << 16;
            uint bb = uint(biases[gate_scale_base + g]) << 16;
            float s = as_type<float>(sb);
            float bi = as_type<float>(bb);
            uint4 ws0 = *((device const uint4*)(packed + gate_word_base + g * AFFINE4_WORDS_PER_GROUP));
            uint4 ws1 = *((device const uint4*)(packed + gate_word_base + g * AFFINE4_WORDS_PER_GROUP + 4u));
            acc_gate += affine4_group_dot_tg(ws0, ws1, s, bi, x4);
        }
        {
            uint sb = uint(scales[up_scale_base + g]) << 16;
            uint bb = uint(biases[up_scale_base + g]) << 16;
            float s = as_type<float>(sb);
            float bi = as_type<float>(bb);
            uint4 ws0 = *((device const uint4*)(packed + up_word_base + g * AFFINE4_WORDS_PER_GROUP));
            uint4 ws1 = *((device const uint4*)(packed + up_word_base + g * AFFINE4_WORDS_PER_GROUP + 4u));
            acc_up += affine4_group_dot_tg(ws0, ws1, s, bi, x4);
        }
    }
    acc_gate = simd_sum(acc_gate);
    acc_up = simd_sum(acc_up);
    if (sg_lane == 0u) {
        float silu_g = acc_gate / (1.0f + metal::exp(-acc_gate));
        y[b * dims.inter + row] = bfloat(silu_g * acc_up);
    }
}
