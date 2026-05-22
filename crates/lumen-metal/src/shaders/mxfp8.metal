// MXFP8 (OCP) fused dequant + matmul kernel.
//
// Layout matches `mlx.quantize(bits=8, group_size=32, mode="mxfp8")`
// checkpoints (e.g. `mlx-community/Qwen3-Embedding-4B-mxfp8`).
//
// Element format: OCP E4M3 (1 sign + 4 exp + 3 mantissa, bias=7, finite-only).
// Group format:   OCP E8M0 (1 byte = biased exponent, bias=127), one per
//                 group of 32 consecutive weight elements.
//
// Per-element dequant:
//   w_real = e4m3_to_float(byte) * 2^(e8m0_scale - 127)
//
// E4M3 decode (no infinities; NaN at S=0 E=15 M=111 and S=1 E=15 M=111):
//   - exp == 0       → subnormal: (-1)^S * mant * 2^-9      = (-1)^S * mant / 512
//   - exp == 15 & m=7→ NaN sentinel (treat as 0)
//   - otherwise      → normal: (-1)^S * (1 + mant/8) * 2^(exp-7)
//
// E8M0 decode:
//   - byte == 0xFF   → NaN sentinel (treat as 0)
//   - otherwise      → 2^(byte - 127)
//
// Memory layout (matches on-disk MLX format):
//   packed: [out_features, in_features/4]   uint    -- 4 E4M3 bytes per uint, LSB-first
//   scales: [out_features, in_features/32]  uchar   -- one E8M0 byte per 32-element group
//   x     : [batch, in_features]            ushort  (bf16)
//   y     : [batch, out_features]           ushort  (bf16)
//
// Grid:  (out_features, batch). One thread per (output row, batch row).

#include <metal_stdlib>
using namespace metal;

constant constexpr uint MXFP8_GROUP_SIZE = 32u;
// 32 input elements per group / 4 bytes per uint = 8 uints per group.
constant constexpr uint MXFP8_WORDS_PER_GROUP = 8u;

struct Mxfp8Dims {
    uint out_features;
    uint in_features;
};

// Decode bf16 stored as `ushort` (= upper 16 bits of an IEEE f32) to float.
static inline float bf16_to_f32_dev(ushort b) {
    uint bits = uint(b) << 16;
    return as_type<float>(bits);
}

// Truncating bf16 from f32 (round-to-nearest-even via +0x8000 bias on the
// dropped low bits). Lossy but matches MLX quantization conventions; the
// dynamic range fits since the dequant accumulation is in f32.
static inline ushort f32_to_bf16_dev(float f) {
    uint bits = as_type<uint>(f);
    return ushort((bits + 0x8000u) >> 16);
}

// Decode one E4M3 byte to f32. NaN encodings collapse to 0 (matches MLX
// host-side dequant: NaN scales propagate as zero contribution rather than
// poisoning the entire output row).
static inline float e4m3_to_f32_dev(uchar b) {
    uint v = uint(b);
    uint sign = (v >> 7) & 0x1u;
    uint exp  = (v >> 3) & 0xFu;
    uint mant = v & 0x7u;
    float f;
    if (exp == 0u) {
        // Subnormal: value = mant * 2^-9 = mant / 512
        f = float(mant) * (1.0f / 512.0f);
    } else if (exp == 15u && mant == 7u) {
        // E4M3 NaN — collapse to 0.
        f = 0.0f;
    } else {
        // Normal: (1 + mant/8) * 2^(exp - 7)
        int e = int(exp) - 7;
        float scale = ldexp(1.0f, e);
        f = scale * (1.0f + float(mant) * (1.0f / 8.0f));
    }
    return (sign != 0u) ? -f : f;
}

// Decode one E8M0 byte to its block-scale multiplier. The 0xFF NaN encoding
// collapses to 0 (whole group contributes zero — matches MLX semantics).
static inline float e8m0_to_f32_dev(uchar b) {
    if (b == 0xFFu) {
        return 0.0f;
    }
    return ldexp(1.0f, int(b) - 127);
}

// ─────────────────────────────────────────────────────────────────────────
// Naive 1-thread-per-output kernel. Always usable; BW-bound on small shapes.
// Mirrors the affine8_matmul_bf16 layout.
// ─────────────────────────────────────────────────────────────────────────
kernel void mxfp8_matmul_bf16(
    device const uint*    packed   [[buffer(0)]],
    device const uchar*   scales   [[buffer(1)]],
    device const ushort*  x        [[buffer(2)]],
    device ushort*        y        [[buffer(3)]],
    constant Mxfp8Dims&   dims     [[buffer(4)]],
    constant uint&        batch    [[buffer(5)]],
    uint2 gid                      [[thread_position_in_grid]]
) {
    uint row = gid.x;
    uint b   = gid.y;
    if (row >= dims.out_features || b >= batch) { return; }

    uint groups         = dims.in_features / MXFP8_GROUP_SIZE;
    uint words_per_row  = dims.in_features / 4u;
    uint word_row_base  = row * words_per_row;
    uint scale_row_base = row * groups;
    uint x_row_base     = b * dims.in_features;

    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        float s = e8m0_to_f32_dev(scales[scale_row_base + g]);

        uint word_base = word_row_base + g * MXFP8_WORDS_PER_GROUP;
        uint x_base    = x_row_base + g * MXFP8_GROUP_SIZE;

        for (uint w = 0; w < MXFP8_WORDS_PER_GROUP; ++w) {
            uint word = packed[word_base + w];
            uint xb   = x_base + w * 4u;
            uchar b0 = uchar((word      ) & 0xFFu);
            uchar b1 = uchar((word >>  8) & 0xFFu);
            uchar b2 = uchar((word >> 16) & 0xFFu);
            uchar b3 = uchar((word >> 24) & 0xFFu);
            float v0 = e4m3_to_f32_dev(b0) * s;
            float v1 = e4m3_to_f32_dev(b1) * s;
            float v2 = e4m3_to_f32_dev(b2) * s;
            float v3 = e4m3_to_f32_dev(b3) * s;
            acc += v0 * bf16_to_f32_dev(x[xb    ]);
            acc += v1 * bf16_to_f32_dev(x[xb + 1]);
            acc += v2 * bf16_to_f32_dev(x[xb + 2]);
            acc += v3 * bf16_to_f32_dev(x[xb + 3]);
        }
    }
    y[b * dims.out_features + row] = f32_to_bf16_dev(acc);
}

// ─────────────────────────────────────────────────────────────────────────
// Cooperative simdgroup qmv_fast variant. 32 lanes split the K dimension;
// one simdgroup produces 4 contiguous output rows; 2 simdgroups per TG.
//
// Constants (mirrors affine8_qmv_fast):
//   VPT  = 16    values_per_thread (16 E4M3 bytes / 16 bf16 inputs)
//   RPS  = 4     results_per_simdgroup
//   NSG  = 2     simdgroups_per_threadgroup
//   BLK  = 512   = VPT * SIMD_SIZE (input columns processed per simdgroup iter)
//   GS   = 32    group_size (one E8M0 scale per 32 consecutive weights)
//   SCALE_STEP = GS / VPT = 2 (2 consecutive threads share one scale)
//
// Constraints:
//   - in_features  % 512 == 0 (block alignment)
//   - out_features % 8   == 0 (NSG*RPS rows per TG)
// Both hold for Qwen3-Embedding-4B (in ∈ {1024, 2560, 9728}, out ∈ {1024, 2560, 4096, 9728, vocab}).
// ─────────────────────────────────────────────────────────────────────────
constant constexpr uint MXFP8_VPT = 16u;
constant constexpr uint MXFP8_RPS = 4u;
constant constexpr uint MXFP8_NSG = 2u;
constant constexpr uint MXFP8_SIMD = 32u;
constant constexpr uint MXFP8_BLK = MXFP8_VPT * MXFP8_SIMD;   // 512
constant constexpr uint MXFP8_SCALE_STEP = MXFP8_GROUP_SIZE / MXFP8_VPT; // 2

kernel void mxfp8_qmv_fast_bf16(
    device const uint*    packed   [[buffer(0)]],
    device const uchar*   scales   [[buffer(1)]],
    device const ushort*  x        [[buffer(2)]],
    device ushort*        y        [[buffer(3)]],
    constant Mxfp8Dims&   dims     [[buffer(4)]],
    constant uint&        batch    [[buffer(5)]],
    uint3 gid                      [[threadgroup_position_in_grid]],
    uint  tid_in_tg                [[thread_index_in_threadgroup]],
    uint  simd_lane                [[thread_index_in_simdgroup]],
    uint  simd_id                  [[simdgroup_index_in_threadgroup]]
) {
    uint b           = gid.x;
    uint row_group   = gid.y;
    if (b >= batch) { return; }

    uint row_base = row_group * (MXFP8_NSG * MXFP8_RPS) + simd_id * MXFP8_RPS;
    if (row_base >= dims.out_features) { return; }

    uint in_features   = dims.in_features;
    uint groups_per_row = in_features / MXFP8_GROUP_SIZE;
    uint words_per_row  = in_features / 4u;

    // Each simdgroup walks K in BLK-sized blocks. Within a block, lane L
    // owns VPT consecutive weight elements starting at L*VPT.
    uint k_start = simd_lane * MXFP8_VPT;
    uint n_blocks = in_features / MXFP8_BLK;
    uint x_base_b = b * in_features;

    // Per-row accumulators in this simdgroup.
    float acc[MXFP8_RPS] = { 0.0f, 0.0f, 0.0f, 0.0f };

    for (uint blk = 0; blk < n_blocks; ++blk) {
        uint k0 = blk * MXFP8_BLK + k_start;

        // Load 16 bf16 inputs (4 bfloat4 each = 4 lots of 4 elements).
        // Use vector loads — bf16 in Metal is `bfloat`/`bfloat4` (Metal 3+).
        ushort x_raw[MXFP8_VPT];
        for (uint i = 0; i < MXFP8_VPT; ++i) {
            x_raw[i] = x[x_base_b + k0 + i];
        }

        // For each of the RPS rows the simdgroup is responsible for,
        // accumulate this lane's 16-element dot product.
        for (uint r = 0; r < MXFP8_RPS; ++r) {
            uint row = row_base + r;
            if (row >= dims.out_features) continue;

            // VPT = 16 = 4 uint32 words per lane.
            uint word_base = row * words_per_row + (k0 / 4u);
            uint scale_base = row * groups_per_row + (k0 / MXFP8_GROUP_SIZE);

            // Two scales cover the 16 lane-local elements (since GS=32, VPT=16,
            // so 2 lanes share each scale; the lane covers half a group).
            // k0 may start at half-group boundary (k0 % 32 == 0 or 16).
            // We compute the local scale index for each of the 16 elems:
            //   elem_global = k0 + e   (e in 0..16)
            //   scale_idx_global = elem_global / GS  = (k0 + e) >> 5
            // Within the lane, e ranges over 16 consecutive elements, so the
            // boundary at most crosses once.
            uint s_lo = (k0) / MXFP8_GROUP_SIZE;
            uint s_hi = (k0 + MXFP8_VPT - 1u) / MXFP8_GROUP_SIZE;
            float scale_lo = e8m0_to_f32_dev(scales[scale_base + (s_lo - (row * groups_per_row + (k0 / MXFP8_GROUP_SIZE)))]);
            // Re-derive offsets correctly: scale_base IS row*groups + (k0/GS).
            // So scale at s_lo = scales[scale_base] always; scale at s_hi =
            // scales[scale_base + (s_hi - s_lo)] which is +0 or +1.
            scale_lo = e8m0_to_f32_dev(scales[scale_base + 0u]);
            float scale_hi = (s_hi == s_lo)
                ? scale_lo
                : e8m0_to_f32_dev(scales[scale_base + 1u]);

            // 4 uint32 words = 16 E4M3 bytes.
            float partial = 0.0f;
            for (uint w = 0; w < 4u; ++w) {
                uint word = packed[word_base + w];
                uint elem_base = w * 4u;
                // Compute per-element scale: split point at e where
                // (k0 + e) % GS == 0 → e = (GS - (k0 % GS)) % GS.
                // Practically: if entire word is within one scale group, use scale_lo;
                // if it crosses, handle byte-by-byte.
                uint k_word_lo = k0 + elem_base;
                uint k_word_hi = k_word_lo + 3u;
                bool same_group = (k_word_lo / MXFP8_GROUP_SIZE) == (k_word_hi / MXFP8_GROUP_SIZE);
                if (same_group) {
                    float s = ((k_word_lo / MXFP8_GROUP_SIZE) == s_lo) ? scale_lo : scale_hi;
                    uchar b0 = uchar((word      ) & 0xFFu);
                    uchar b1 = uchar((word >>  8) & 0xFFu);
                    uchar b2 = uchar((word >> 16) & 0xFFu);
                    uchar b3 = uchar((word >> 24) & 0xFFu);
                    float v0 = e4m3_to_f32_dev(b0) * s;
                    float v1 = e4m3_to_f32_dev(b1) * s;
                    float v2 = e4m3_to_f32_dev(b2) * s;
                    float v3 = e4m3_to_f32_dev(b3) * s;
                    partial += v0 * bf16_to_f32_dev(x_raw[elem_base    ]);
                    partial += v1 * bf16_to_f32_dev(x_raw[elem_base + 1]);
                    partial += v2 * bf16_to_f32_dev(x_raw[elem_base + 2]);
                    partial += v3 * bf16_to_f32_dev(x_raw[elem_base + 3]);
                } else {
                    // Per-byte scale lookup (rare — only one word per lane
                    // straddles when k0 lands at a half-group).
                    for (uint i = 0; i < 4u; ++i) {
                        uint k_g = (k_word_lo + i) / MXFP8_GROUP_SIZE;
                        float s = (k_g == s_lo) ? scale_lo : scale_hi;
                        uchar bv = uchar((word >> (8u * i)) & 0xFFu);
                        partial += e4m3_to_f32_dev(bv) * s
                                 * bf16_to_f32_dev(x_raw[elem_base + i]);
                    }
                }
            }
            acc[r] += partial;
        }
    }

    // Reduce across the 32 lanes of the simdgroup; lane 0 writes the result.
    for (uint r = 0; r < MXFP8_RPS; ++r) {
        float total = simd_sum(acc[r]);
        if (simd_lane == 0u) {
            uint row = row_base + r;
            if (row < dims.out_features) {
                y[b * dims.out_features + row] = f32_to_bf16_dev(total);
            }
        }
    }
}
