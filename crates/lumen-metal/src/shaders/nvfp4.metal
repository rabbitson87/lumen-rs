// NVFP4 fused dequant + matvec kernel (Phase 2 first unit).
//
// Layout matches `crate::nvfp4` (MLX nvfp4 storage convention, group_size=16):
//   packed: [out_features, in_features/8]  uint   -- 8 nibbles per word, LSB-first (same as MXFP4)
//   scales: [out_features, in_features/16] uchar  -- one E4M3 FP8 scale per 16-element group
//   x     : [in_features]                   float  -- dense activation
//   y     : [out_features]                  float  -- output
//
// Differs from `mxfp4.metal` only by group size (16 vs 32 → 2 words/group, /16 indexing)
// and the per-group scale decode (E4M3 vs E8M0). The E2M1 nibble value table and packing
// are identical. CPU parity reference: `crate::nvfp4::dequantize_f32` (bit-identical decode).

#include <metal_stdlib>
using namespace metal;

constant float E2M1_LUT[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

struct NvFp4Dims {
    uint out_features;
    uint in_features;
};

// E4M3 FP8 scale decode (`s eeee mmm`, exponent bias 7). Mirrors `crate::nvfp4::e4m3_to_f32`.
//   normal   (exp != 0): (-1)^s * (1 + mant/8) * 2^(exp - 7)
//   subnormal (exp == 0): (-1)^s * (mant/8) * 2^-6
//   byte & 0x7F == 0x7F → NaN encoding, mapped to 0.
static inline float e4m3_scale_device(uchar byte) {
    if ((byte & 0x7F) == 0x7F) { return 0.0f; }
    float sign = (byte & 0x80) ? -1.0f : 1.0f;
    int   exp  = int((byte >> 3) & 0x0F);
    float mant = float(byte & 0x07);
    float mag  = (exp == 0) ? (mant * 0.125f) * 0.015625f
                            : (1.0f + mant * 0.125f) * exp2(float(exp - 7));
    return sign * mag;
}

kernel void nvfp4_matvec_f32(
    device const uint*    packed  [[buffer(0)]],
    device const uchar*   scales  [[buffer(1)]],
    device const float*   x       [[buffer(2)]],
    device float*         y       [[buffer(3)]],
    constant NvFp4Dims&   dims    [[buffer(4)]],
    uint row                      [[thread_position_in_grid]]
) {
    if (row >= dims.out_features) { return; }

    uint groups         = dims.in_features / 16u;
    uint words_per_row  = dims.in_features / 8u;
    uint word_row_base  = row * words_per_row;
    uint scale_row_base = row * groups;

    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        float s = e4m3_scale_device(scales[scale_row_base + g]);
        if (s == 0.0f) { continue; }

        uint word_base = word_row_base + g * 2u; // 16 elems / 8 nibbles = 2 words per group
        uint x_base    = g * 16u;
        for (uint w = 0; w < 2u; ++w) {
            uint word = packed[word_base + w];
            for (uint i = 0; i < 8u; ++i) {
                uint nib = (word >> (i * 4u)) & 0xFu;
                acc += E2M1_LUT[nib] * s * x[x_base + w * 8u + i];
            }
        }
    }
    y[row] = acc;
}

// Batched variant: grid is (out_features, batch). Each thread computes
// y[b, row] = sum_k W[row, k] * x[b, k]. Ported from `mxfp4_matmul_f32` (5-edit).
kernel void nvfp4_matmul_f32(
    device const uint*    packed  [[buffer(0)]],
    device const uchar*   scales  [[buffer(1)]],
    device const float*   x       [[buffer(2)]],
    device float*         y       [[buffer(3)]],
    constant NvFp4Dims&   dims    [[buffer(4)]],
    constant uint&        batch   [[buffer(5)]],
    uint2 gid                     [[thread_position_in_grid]]
) {
    uint row = gid.x;
    uint b   = gid.y;
    if (row >= dims.out_features || b >= batch) { return; }

    uint groups         = dims.in_features / 16u;
    uint words_per_row  = dims.in_features / 8u;
    uint word_row_base  = row * words_per_row;
    uint scale_row_base = row * groups;
    uint x_row_base     = b * dims.in_features;

    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        float s = e4m3_scale_device(scales[scale_row_base + g]);
        if (s == 0.0f) { continue; }

        uint word_base = word_row_base + g * 2u;
        uint x_base    = x_row_base + g * 16u;
        for (uint w = 0; w < 2u; ++w) {
            uint word = packed[word_base + w];
            for (uint i = 0; i < 8u; ++i) {
                uint nib = (word >> (i * 4u)) & 0xFu;
                acc += E2M1_LUT[nib] * s * x[x_base + w * 8u + i];
            }
        }
    }
    y[b * dims.out_features + row] = acc;
}

// MoE grouped matmul: one dispatch handles k expert slots via `expert_indices`.
// Grid = (out, batch, k). Layout (per-expert slabs):
//   packed_all : [num_experts, out_features, in_features/8]  uint
//   scales_all : [num_experts, out_features, in_features/16] uchar
//   x          : broadcast_x==1 → [batch, in]; else [k, batch, in]
//   y          : [k, batch, out_features] f32
// Ported from `mxfp4_matmul_moe_f32` (5-edit).
struct NvFp4MoeDims {
    uint out_features;
    uint in_features;
    uint batch;
    uint broadcast_x;
};

kernel void nvfp4_matmul_moe_f32(
    device const uint*     packed_all     [[buffer(0)]],
    device const uchar*    scales_all     [[buffer(1)]],
    device const uint*     expert_indices [[buffer(2)]],
    device const float*    x              [[buffer(3)]],
    device float*          y              [[buffer(4)]],
    constant NvFp4MoeDims& dims           [[buffer(5)]],
    uint3 gid                             [[thread_position_in_grid]]
) {
    uint row  = gid.x;
    uint b    = gid.y;
    uint slot = gid.z;
    if (row >= dims.out_features || b >= dims.batch) { return; }

    uint e = expert_indices[slot];

    uint groups         = dims.in_features / 16u;
    uint words_per_row  = dims.in_features / 8u;
    uint packed_expert_stride = dims.out_features * words_per_row;
    uint scale_expert_stride  = dims.out_features * groups;

    uint word_row_base  = e * packed_expert_stride + row * words_per_row;
    uint scale_row_base = e * scale_expert_stride  + row * groups;

    uint x_slot = (dims.broadcast_x != 0u) ? 0u : slot;
    uint x_row_base = x_slot * dims.batch * dims.in_features + b * dims.in_features;

    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        float s = e4m3_scale_device(scales_all[scale_row_base + g]);
        if (s == 0.0f) { continue; }
        uint word_base = word_row_base + g * 2u;
        uint x_base    = x_row_base + g * 16u;
        for (uint w = 0; w < 2u; ++w) {
            uint word = packed_all[word_base + w];
            for (uint i = 0; i < 8u; ++i) {
                uint nib = (word >> (i * 4u)) & 0xFu;
                acc += E2M1_LUT[nib] * s * x[x_base + w * 8u + i];
            }
        }
    }
    y[slot * dims.batch * dims.out_features + b * dims.out_features + row] = acc;
}
