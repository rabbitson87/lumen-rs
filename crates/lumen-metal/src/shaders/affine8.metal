// Affine 8-bit fused dequant + matmul kernel.
//
// Layout matches mlx.quantize(bits=8, group_size=64) checkpoints
// (e.g. Qwen3-Embedding-0.6B-8bit). Per-element dequant:
//
//   w_real = uint8_value * scale + bias
//
// where `uint8_value` is the raw 0..255 byte (unsigned) extracted from
// `packed` (uint32 holds 4 consecutive byte-values, LSB-first).
//
// Memory layout:
//   packed: [out_features, in_features/4]   uint    -- 4 bytes per word, LSB-first
//   scales: [out_features, in_features/64]  ushort  -- one bf16 scale per 64-element group
//   biases: [out_features, in_features/64]  ushort  -- one bf16 bias  per 64-element group
//   x     : [batch, in_features]            ushort  (bf16)
//   y     : [batch, out_features]           ushort  (bf16)
//
// Grid:  (out_features, batch). One thread per (output row, batch row).
// Each thread iterates all 64-element groups of its row and accumulates the
// dot product on the fly. Mirrors `affine4_matmul_f32_v2` but with 8-bit
// byte unpack (4 bytes/u32) instead of 4-bit nibble (8 nibbles/u32).

#include <metal_stdlib>
using namespace metal;

constant constexpr uint AFFINE8_GROUP_SIZE = 64u;
// 64 input elements per group / 4 bytes per uint = 16 uints per group.
constant constexpr uint AFFINE8_WORDS_PER_GROUP = 16u;

struct Affine8Dims {
    uint out_features;
    uint in_features;
};

// Decode bf16 stored as `ushort` (= upper 16 bits of an IEEE f32) to float.
static inline float bf16_to_f32_dev(ushort b) {
    uint bits = uint(b) << 16;
    return as_type<float>(bits);
}

// Truncating bf16 from f32 (drop low 16 bits). Lossy but matches MLX
// quantization conventions; sufficient for the output accumulation here.
static inline ushort f32_to_bf16_dev(float f) {
    uint bits = as_type<uint>(f);
    return ushort((bits + 0x8000u) >> 16);
}

kernel void affine8_matmul_bf16(
    device const uint*    packed   [[buffer(0)]],
    device const ushort*  scales   [[buffer(1)]],
    device const ushort*  biases   [[buffer(2)]],
    device const ushort*  x        [[buffer(3)]],
    device ushort*        y        [[buffer(4)]],
    constant Affine8Dims& dims     [[buffer(5)]],
    constant uint&        batch    [[buffer(6)]],
    uint2 gid                      [[thread_position_in_grid]]
) {
    uint row = gid.x;
    uint b   = gid.y;
    if (row >= dims.out_features || b >= batch) { return; }

    uint groups         = dims.in_features / AFFINE8_GROUP_SIZE;
    uint words_per_row  = dims.in_features / 4u;
    uint word_row_base  = row * words_per_row;
    uint scale_row_base = row * groups;
    uint x_row_base     = b * dims.in_features;

    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        float s  = bf16_to_f32_dev(scales[scale_row_base + g]);
        float bi = bf16_to_f32_dev(biases[scale_row_base + g]);

        uint word_base = word_row_base + g * AFFINE8_WORDS_PER_GROUP;
        uint x_base    = x_row_base + g * AFFINE8_GROUP_SIZE;

        for (uint w = 0; w < AFFINE8_WORDS_PER_GROUP; ++w) {
            uint word = packed[word_base + w];
            uint xb   = x_base + w * 4u;
            // Unroll the 4-byte unpack — compiler benefits from the
            // sequential shifts being visible.
            float v0 = float((word      ) & 0xFFu);
            float v1 = float((word >>  8) & 0xFFu);
            float v2 = float((word >> 16) & 0xFFu);
            float v3 = float((word >> 24) & 0xFFu);
            acc += (v0 * s + bi) * bf16_to_f32_dev(x[xb    ]);
            acc += (v1 * s + bi) * bf16_to_f32_dev(x[xb + 1]);
            acc += (v2 * s + bi) * bf16_to_f32_dev(x[xb + 2]);
            acc += (v3 * s + bi) * bf16_to_f32_dev(x[xb + 3]);
        }
    }
    y[b * dims.out_features + row] = f32_to_bf16_dev(acc);
}

// ─────────────────────────────────────────────────────────────────────────────
// affine8_qmv_fast_bf16in_bf16out — cooperative simdgroup kernel.
//
// Mirrors the affine4 qmv_fast layout (32-lane simdgroup splits the K
// dimension; one simdgroup produces RPS contiguous output rows; NSG
// simdgroups per threadgroup) but for the MLX 8-bit packed format
// (PACK_FACTOR=4 bytes / u32, no nibble shift trick needed since each
// byte is already isolated).
//
// Constants for 8-bit:
//   VPT  = 16    values_per_thread (16 bytes / 16 weight elements / 16 bf16 inputs)
//   RPS  = 4     results_per_simdgroup (output rows per simdgroup)
//   NSG  = 2     simdgroups_per_threadgroup
//   BLK  = 512   = VPT * SIMD_SIZE (input columns processed per simdgroup iter)
//   GS   = 64    group_size (one scale/bias per 64 consecutive weights)
//   SCALE_STEP = GS / VPT = 4 (so 4 consecutive threads share one (scale,bias))
//
// Constraints:
//   - in_features  % 512 == 0 (block alignment)
//   - out_features % 8   == 0 (NSG*RPS rows per TG)
//
// Both hold for Qwen3-Embedding-0.6B (in ∈ {1024,3072}, out ∈ {512,1024,3072,vocab}).
//
// Internal math: f32 accumulation. bf16 narrows only on the I/O boundary
// (16 bf16 inputs upcast to f32 via bfloat4 vector loads; final per-row
// result downcast to bf16 at store time). simd_sum reduces the 32 partial
// dots across the simdgroup.
// ─────────────────────────────────────────────────────────────────────────────

// Vectorized bf16→f32 load for 16 consecutive activations + scalar sum_x.
// Mirrors `qmv_fast_load_x_bf16` in affine4.metal but WITHOUT the nibble
// pre-scaling (1/16, 1/256, 1/4096) — 8-bit bytes don't need it.
static inline float qmv8_load_x_bf16(
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

    float sum = (f0.x + f0.y + f0.z + f0.w)
              + (f1.x + f1.y + f1.z + f1.w)
              + (f2.x + f2.y + f2.z + f2.w)
              + (f3.x + f3.y + f3.z + f3.w);

    x_thread[ 0] = f0.x; x_thread[ 1] = f0.y; x_thread[ 2] = f0.z; x_thread[ 3] = f0.w;
    x_thread[ 4] = f1.x; x_thread[ 5] = f1.y; x_thread[ 6] = f1.z; x_thread[ 7] = f1.w;
    x_thread[ 8] = f2.x; x_thread[ 9] = f2.y; x_thread[10] = f2.z; x_thread[11] = f2.w;
    x_thread[12] = f3.x; x_thread[13] = f3.y; x_thread[14] = f3.z; x_thread[15] = f3.w;

    return sum;
}

// Inner dot for one thread's 16 contiguous weight bytes against 16 bf16
// activations (already upcast to x_thread[]). All 16 weights belong to the
// SAME group g (since VPT=16 ≤ GS=64 and the thread offset aligns to a
// 16-byte boundary inside the group), so we apply
//   accum = scale_g * Σ(x[i] * byte[i]) + bias_g * Σ(x[i])
// which is mathematically identical to dequantizing each byte to
// `(byte * scale_g + bias_g)` and computing the dot.
//
// 4 uchar4 vector loads (16 bytes total) → 4 × 4 = 16 MACs structured as
// tree-reduced sub-expressions (mirrors affine4's pattern: ILP-friendly,
// 4 chains of 4 adds vs 16 sequential adds on the accumulator).
static inline float qmv8_dot(
    device const uint8_t* w,
    thread const float* x_thread,
    float scale,
    float bias,
    float sum_x
) {
    device const uchar4* w4 = (device const uchar4*)w;
    uchar4 b0 = w4[0];
    uchar4 b1 = w4[1];
    uchar4 b2 = w4[2];
    uchar4 b3 = w4[3];

    float accum = 0.0f;
    accum += (x_thread[ 0] * float(b0.x)
            + x_thread[ 1] * float(b0.y)
            + x_thread[ 2] * float(b0.z)
            + x_thread[ 3] * float(b0.w));
    accum += (x_thread[ 4] * float(b1.x)
            + x_thread[ 5] * float(b1.y)
            + x_thread[ 6] * float(b1.z)
            + x_thread[ 7] * float(b1.w));
    accum += (x_thread[ 8] * float(b2.x)
            + x_thread[ 9] * float(b2.y)
            + x_thread[10] * float(b2.z)
            + x_thread[11] * float(b2.w));
    accum += (x_thread[12] * float(b3.x)
            + x_thread[13] * float(b3.y)
            + x_thread[14] * float(b3.z)
            + x_thread[15] * float(b3.w));

    return scale * accum + bias * sum_x;
}

kernel void affine8_qmv_fast_bf16(
    device const uint*    packed   [[buffer(0)]],   // [out, in/4]    u32 (4 bytes / word)
    device const ushort*  scales   [[buffer(1)]],   // [out, in/64]   bf16-as-u16
    device const ushort*  biases   [[buffer(2)]],   // [out, in/64]   bf16-as-u16
    device const bfloat*  x        [[buffer(3)]],   // [batch, in]    bf16
    device bfloat*        y        [[buffer(4)]],   // [batch, out]   bf16
    constant Affine8Dims& dims     [[buffer(5)]],
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
    constexpr int SCALE_STEP   = GS / VPT;   // = 4
    constexpr int BYTES_PER_PACK = 4;
    constexpr int PACK_FACTOR    = 4;        // 4 bytes per uint32 (8-bit)

    int b_idx   = int(tid.x);
    int out_row = int(tid.y) * (NSG * RPS) + int(simd_gid) * RPS;
    if (b_idx >= int(batch) || out_row >= int(dims.out_features)) return;

    int in_vec      = int(dims.in_features);
    int row_w_bytes = in_vec * BYTES_PER_PACK / PACK_FACTOR;  // = in_vec (8-bit: 1 byte/weight)
    int row_g       = in_vec / GS;

    // Per-thread starting pointers.
    //   - 16 bytes of weights/row at offset simd_lid * VPT = simd_lid * 16
    //   - 1 (scale,bias) per row, indexed by simd_lid / SCALE_STEP
    //   - 16 bf16 activations at offset simd_lid * VPT
    device const uint8_t* ws =
        (device const uint8_t*)packed
        + size_t(out_row) * row_w_bytes
        + size_t(simd_lid) * VPT;
    int sb_offset = int(simd_lid) / SCALE_STEP;
    device const ushort* scl = scales + size_t(out_row) * row_g + sb_offset;
    device const ushort* bse = biases + size_t(out_row) * row_g + sb_offset;
    device const bfloat* xp  = x + size_t(b_idx) * in_vec + size_t(simd_lid) * VPT;

    thread float x_thread[VPT];
    thread float result[RPS] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Outer loop: BLK input elements per iteration, advancing all pointers.
    for (int k = 0; k < in_vec; k += BLK) {
        float sum_x = qmv8_load_x_bf16(xp, x_thread);
        for (int row = 0; row < RPS; row++) {
            device const uint8_t* wl = ws + size_t(row) * row_w_bytes;
            float s = bf16_to_f32_dev(scl[row * row_g]);
            float b = bf16_to_f32_dev(bse[row * row_g]);
            result[row] += qmv8_dot(wl, x_thread, s, b, sum_x);
        }
        ws  += BLK;          // advance row pointer (BLK bytes, 1 byte/weight)
        scl += BLK / GS;     // advance group pointer (8 groups)
        bse += BLK / GS;
        xp  += BLK;
    }

    // Reduce 32-lane partial dot products per row.
    for (int row = 0; row < RPS; row++) {
        result[row] = simd_sum(result[row]);
        if (simd_lid == 0u) {
            int out_idx = b_idx * int(dims.out_features) + out_row + row;
            y[out_idx] = bfloat(result[row]);
        }
    }
}
