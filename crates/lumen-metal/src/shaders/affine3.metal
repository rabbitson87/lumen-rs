// Affine 3-bit fused dequant + matvec kernel — Phase 18.A POC.
//
// Bit-plane packing (Option A from phase18a_affine3_design.md):
//   For each chunk of 32 elements:
//     u32_lo  = bit 0 of elem[0..32], one bit per element  (32 LSBs)
//     u32_mid = bit 1 of elem[0..32]
//     u32_hi  = bit 2 of elem[0..32]
//   3 u32 = 96 bits = 32 codes × 3 bits ⟶ 0.375 bytes/element
//   (vs Affine4's 0.5 bytes/element — 25% packed-byte saving)
//
// Group size = 64 (matches Affine4) → 2 bit-plane chunks per group.
// Per-group meta: 1 bf16 scale + 1 bf16 bias (unchanged from Affine4).
//
// Decode per element (~9 ops):
//   lo = (u_lo  >> i) & 1
//   md = (u_mid >> i) & 1
//   hi = (u_hi  >> i) & 1
//   code = (hi << 2) | (md << 1) | lo                  (range [0..7])
//   w_real = code * scale + bias
//
// vs Affine4 (~4 ops):
//   nibble = (word >> (i*4)) & 0xF                     (range [0..15])
//   w_real = nibble * scale + bias
//
// POC goal: validate that BW saving (25%) materializes despite 2.25× compute
// overhead. Risk = compute exhausts the 30% BW headroom we measured at
// MLP gate_up (BW utilization 69.8%).

#include <metal_stdlib>
using namespace metal;

struct Affine3Dims {
    uint out_features;
    uint in_features;
};

constant uint AFFINE3_GROUP_SIZE     = 64u;
constant uint AFFINE3_CHUNK_SIZE     = 32u;
constant uint AFFINE3_WORDS_PER_CHUNK = 3u;
constant uint AFFINE3_CHUNKS_PER_GROUP = 2u;

static inline float bf16_to_f32_device(ushort b) {
    return as_type<float>(uint(b) << 16);
}

// ────────────────────────────────────────────────────────────────────────
// POC kernel: 1 thread per output row, no threadgroup memory.
// Mirrors `affine4_matmul_f32_v2` topology (the simplest variant).
// ────────────────────────────────────────────────────────────────────────
kernel void affine3_matvec_bf16in_bf16out(
    device const uint*    packed   [[buffer(0)]],   // [out, in/32 * 3]
    device const ushort*  scales   [[buffer(1)]],   // [out, in/64]
    device const ushort*  biases   [[buffer(2)]],   // [out, in/64]
    device const bfloat*  x        [[buffer(3)]],   // [in]
    device bfloat*        y        [[buffer(4)]],   // [out]
    constant Affine3Dims& dims     [[buffer(5)]],
    uint row                       [[thread_position_in_grid]]
) {
    if (row >= dims.out_features) return;

    uint in_vec        = dims.in_features;
    uint groups        = in_vec / AFFINE3_GROUP_SIZE;
    uint chunks_per_row = in_vec / AFFINE3_CHUNK_SIZE;
    uint words_per_row = chunks_per_row * AFFINE3_WORDS_PER_CHUNK;

    uint word_row_base  = row * words_per_row;
    uint scale_row_base = row * groups;

    float acc = 0.0f;
    for (uint chunk_idx = 0; chunk_idx < chunks_per_row; ++chunk_idx) {
        uint group_idx = chunk_idx / AFFINE3_CHUNKS_PER_GROUP;
        float s  = bf16_to_f32_device(scales[scale_row_base + group_idx]);
        float bi = bf16_to_f32_device(biases[scale_row_base + group_idx]);

        uint word_base = word_row_base + chunk_idx * AFFINE3_WORDS_PER_CHUNK;
        uint x_base    = chunk_idx * AFFINE3_CHUNK_SIZE;

        // Vectorized load: 3 u32 = uint3 = 12 bytes (single 12-byte load).
        device const uint3* p3 = (device const uint3*)(packed + word_base);
        uint3 ws = *p3;
        uint u_lo  = ws.x;
        uint u_mid = ws.y;
        uint u_hi  = ws.z;

        // 32 × float = 32 elements (vectorized as 8 × float4).
        device const float4* x4 = (device const float4*)((device const float*)x + x_base);
        // Hmm: x is bfloat, can't reinterpret as float4. Read bf16 → float scalar.
        // (Future opt: bfloat4 vector load like affine4_qmv_fast.)

        // Manually unroll 32 elements.
        for (uint i = 0; i < 32; ++i) {
            uint lo = (u_lo  >> i) & 1u;
            uint md = (u_mid >> i) & 1u;
            uint hi = (u_hi  >> i) & 1u;
            float v = float((hi << 2) | (md << 1) | lo);
            float dequant = v * s + bi;
            acc += dequant * float(x[x_base + i]);
        }
    }
    y[row] = bfloat(acc);
}

// ────────────────────────────────────────────────────────────────────────
// Phase 18.A POC.1 — qmv_fast topology Affine3 kernel
// Mirrors `affine4_qmv_fast_bf16in_bf16out` (NSG=2, RPS=4, VPT=16, BLK=512).
//
// Each thread owns:
//   - 16 input values (x_thread, bf16-loaded then upcast to f32)
//   - Half of one bit-plane chunk (32 elements / 3 u32). Two adjacent
//     threads share one chunk; each thread extracts its own 16-bit mask
//     by shifting the 32-bit bit-plane word by 0 (lower half) or 16 (upper).
//   - Per row in RPS=4: one (scale, bias) pair (indexed by simd_lid / SCALE_STEP)
//
// Decode trick (analog of Affine4's pre-scaled x_thread):
//   code_i = 4 * hi_i + 2 * mid_i + lo_i  where (hi_i, mid_i, lo_i) ∈ {0,1}
//   Σ_i (code_i * scale * x_i) = scale * (4 * Σ(hi_i * x_i) + 2 * Σ(mid_i * x_i) + Σ(lo_i * x_i))
//   Each partial Σ is a select-sum: include x_i iff the bit is set.
//   Implementation: float((mask >> i) & 1) * x_i — branch-free, 1 shift + 1 mask + 1 cast + 1 FMA.
// ────────────────────────────────────────────────────────────────────────

// Load 16 bf16 activations as 4 × bfloat4 vector loads, upcast to f32.
// Returns Σ(x_i) for the bias term. Mirrors `qmv_fast_load_x_bf16` in affine4.metal
// but WITHOUT pre-scaling (Affine3 needs raw values for bit-plane select-sum).
static inline float affine3_load_x_bf16(
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

    x_thread[ 0] = f0.x; x_thread[ 1] = f0.y; x_thread[ 2] = f0.z; x_thread[ 3] = f0.w;
    x_thread[ 4] = f1.x; x_thread[ 5] = f1.y; x_thread[ 6] = f1.z; x_thread[ 7] = f1.w;
    x_thread[ 8] = f2.x; x_thread[ 9] = f2.y; x_thread[10] = f2.z; x_thread[11] = f2.w;
    x_thread[12] = f3.x; x_thread[13] = f3.y; x_thread[14] = f3.z; x_thread[15] = f3.w;

    return sum;
}

// Inner dot for one thread's slice. Reads 3 × u32 (one bit-plane chunk = 32 elements),
// extracts its 16-bit half (lower or upper) for the 16 codes this thread owns,
// and computes scale * (4*hi_dot + 2*mid_dot + lo_dot) + bias * sum_x.
//
// `chunk_words[0..3]` = (u_lo, u_mid, u_hi) for the 32-element chunk.
// `half` = 0 (lower 16 elements) or 1 (upper 16). Selected via `simd_lid % 2`.
static inline float affine3_qmv_fast_dot(
    uint u_lo, uint u_mid, uint u_hi,
    uint half_idx,
    thread const float* x_thread,
    float scale,
    float bias,
    float sum_x
) {
    uint shift = half_idx * 16u;
    uint mask_lo  = (u_lo  >> shift) & 0xFFFFu;
    uint mask_mid = (u_mid >> shift) & 0xFFFFu;
    uint mask_hi  = (u_hi  >> shift) & 0xFFFFu;

    // Three partial select-sums. Branch-free: bit-extract → cast → FMA.
    float lo_dot = 0.0f;
    float mid_dot = 0.0f;
    float hi_dot = 0.0f;

    // Unrolled 16 iterations. Tree-reduction pattern mirrors affine4_qmv_fast_dot
    // to keep dependency chains short.
    #pragma unroll
    for (uint i = 0; i < 16; ++i) {
        float bit_lo  = float((mask_lo  >> i) & 1u);
        float bit_mid = float((mask_mid >> i) & 1u);
        float bit_hi  = float((mask_hi  >> i) & 1u);
        lo_dot  += bit_lo  * x_thread[i];
        mid_dot += bit_mid * x_thread[i];
        hi_dot  += bit_hi  * x_thread[i];
    }

    float weighted_dot = 4.0f * hi_dot + 2.0f * mid_dot + lo_dot;
    return scale * weighted_dot + bias * sum_x;
}

kernel void affine3_qmv_fast_bf16in_bf16out(
    device const uint*    packed   [[buffer(0)]],   // [out, in/32 * 3]   uint32, bit-plane
    device const ushort*  scales   [[buffer(1)]],   // [out, in/64]       bf16-as-u16
    device const ushort*  biases   [[buffer(2)]],   // [out, in/64]       bf16-as-u16
    device const bfloat*  x        [[buffer(3)]],   // [batch, in]        bf16
    device bfloat*        y        [[buffer(4)]],   // [batch, out]       bf16
    constant Affine3Dims& dims     [[buffer(5)]],
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
    constexpr int SCALE_STEP = GS / VPT; // = 4

    int b_idx = int(tid.x);
    int out_row = int(tid.y) * (NSG * RPS) + int(simd_gid) * RPS;
    if (b_idx >= int(batch) || out_row >= int(dims.out_features)) return;

    int in_vec  = int(dims.in_features);
    int row_g   = in_vec / GS;
    // Words per row (3 u32 per 32-element chunk).
    int chunks_per_row = in_vec / 32;
    int words_per_row  = chunks_per_row * 3;

    // Each thread owns 16 elements within one chunk (lower or upper half).
    // simd_lid 0..31 → element span [simd_lid * 16 .. simd_lid * 16 + 16].
    // chunk_idx = simd_lid / 2; half = simd_lid % 2.
    uint chunk_idx_thread = uint(simd_lid) / 2u;
    uint half_idx_thread  = uint(simd_lid) & 1u;

    device const uint* base_words = packed + size_t(out_row) * size_t(words_per_row)
                                           + size_t(chunk_idx_thread) * 3u;
    int sb_offset = int(simd_lid) / SCALE_STEP;
    device const ushort* scl = scales + size_t(out_row) * row_g + sb_offset;
    device const ushort* bse = biases + size_t(out_row) * row_g + sb_offset;
    device const bfloat* xp  = x + size_t(b_idx) * in_vec + size_t(simd_lid) * VPT;

    thread float x_thread[VPT];
    thread float result[RPS] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Outer loop: BLK input elements per iteration.
    // BLK = 512 elements = 16 chunks. Per simdgroup (32 threads), each thread
    // covers 16 elements ⟹ 32 × 16 = 512 elements per BLK, matches.
    for (int k = 0; k < in_vec; k += BLK) {
        float sum_x = affine3_load_x_bf16(xp, x_thread);

        // Read the 3 u32s for THIS thread's chunk (12 bytes).
        // Vectorized as uint3 if alignment permits (always — 3 × 4 = 12 bytes,
        // base aligned to 4 since chunk_idx_thread × 3 stays u32-aligned).
        uint3 ws = uint3(base_words[0], base_words[1], base_words[2]);
        uint u_lo  = ws.x;
        uint u_mid = ws.y;
        uint u_hi  = ws.z;

        for (int row = 0; row < RPS; row++) {
            // Each row in RPS has its own packed words at row_offset.
            // row_words_per_row stride = words_per_row.
            // Look up scale/bias for row.
            float s = as_type<float>(uint(scl[row * row_g]) << 16);
            float b = as_type<float>(uint(bse[row * row_g]) << 16);

            // For row > 0, fetch the chunk for that row's offset.
            // Compute row's base_words pointer.
            device const uint* row_words = base_words + size_t(row) * size_t(words_per_row);
            uint3 ws_row = uint3(row_words[0], row_words[1], row_words[2]);
            float partial = affine3_qmv_fast_dot(
                ws_row.x, ws_row.y, ws_row.z,
                half_idx_thread,
                x_thread, s, b, sum_x
            );
            result[row] += partial;
        }

        // Advance pointers for next BLK chunk.
        // Each BLK = 16 chunks per row. Per thread we step 1 chunk per BLK (since
        // the simd group covers 16 chunks via 32 threads × 16 elements / 32 elements/chunk = 16).
        // Wait: 32 threads × 16 elements = 512, and 512/32=16 chunks per BLK. Each thread's
        // chunk advances by 16 (the simdgroup covers 16 chunks per BLK, then next BLK starts
        // 16 chunks later).
        base_words += 16u * 3u; // 16 chunks × 3 u32 per chunk
        scl += BLK / GS;        // 8 groups
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
