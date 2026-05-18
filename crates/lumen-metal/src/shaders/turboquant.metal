#include <metal_stdlib>
using namespace metal;

// ============================================================================
// TurboQuant Metal Kernels
// ============================================================================
//
// 6 kernels for GPU-native KV cache compression and compressed attention.
//
// Bitpacking format (matches CPU turboquant-core::bitpack):
//   codes_per_word = 64 / bits  (no cross-word-boundary packing)
//   word_idx = code_index / codes_per_word
//   bit_pos  = (code_index % codes_per_word) * bits
//   n_packed = ceil(dim / codes_per_word)

// ============================================================================
// Kernel 1: Rotate and Normalize
// ============================================================================

kernel void tq_rotate_and_normalize(
    device const float* kv_vectors  [[buffer(0)]],
    device const float* rotation    [[buffer(1)]],
    device float*       rotated_out [[buffer(2)]],
    device float*       scales_out  [[buffer(3)]],
    constant uint&      dim         [[buffer(4)]],
    constant uint&      n_vecs      [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint vec_idx = tid.y;
    uint elem_idx = tid.x;
    if (vec_idx >= n_vecs || elem_idx >= dim) return;

    // rotated[vec_idx][elem_idx] = sum_j R[elem_idx][j] * kv[vec_idx][j]
    float sum = 0.0f;
    for (uint j = 0; j < dim; j++) {
        sum += rotation[elem_idx * dim + j] * kv_vectors[vec_idx * dim + j];
    }

    uint out_idx = vec_idx * dim + elem_idx;
    rotated_out[out_idx] = sum;

    // Reduction: each thread contributes sum^2, thread 0 reduces
    threadgroup float partial_sq[256];
    partial_sq[elem_idx] = sum * sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (elem_idx == 0) {
        float norm_sq = 0.0f;
        for (uint j = 0; j < dim; j++) {
            norm_sq += partial_sq[j];
        }
        float scale = sqrt(norm_sq / float(dim));
        scales_out[vec_idx] = scale;

        if (scale > 1e-10f) {
            float inv_scale = 1.0f / scale;
            for (uint j = 0; j < dim; j++) {
                rotated_out[vec_idx * dim + j] *= inv_scale;
            }
        }
    }
}

// ============================================================================
// Kernel 2: Lloyd-Max Quantize (kept for tests)
// ============================================================================

kernel void tq_lloyd_max_quantize(
    device const float* normalized   [[buffer(0)]],
    device const float* boundaries   [[buffer(1)]],
    device uchar*       codes_out    [[buffer(2)]],
    constant uint&      dim          [[buffer(3)]],
    constant uint&      n_vecs       [[buffer(4)]],
    constant uint&      n_levels     [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint vec_idx = tid.y;
    uint elem_idx = tid.x;
    if (vec_idx >= n_vecs || elem_idx >= dim) return;

    float x = normalized[vec_idx * dim + elem_idx];

    uint code = n_levels - 1;
    for (uint i = 1; i < n_levels; i++) {
        if (x < boundaries[i]) {
            code = i - 1;
            break;
        }
    }

    codes_out[vec_idx * dim + elem_idx] = uchar(code);
}

// ============================================================================
// Kernel 1+2 Fused: Rotate, Normalize, and Quantize
// ============================================================================
// Eliminates intermediate rotated_out buffer round-trip.
// Thread structure: [dim, n_vecs] — one thread per element.

kernel void tq_rotate_normalize_quantize(
    device const float* kv_vectors  [[buffer(0)]],
    device const float* rotation    [[buffer(1)]],
    device const float* boundaries  [[buffer(2)]],
    device float*       scales_out  [[buffer(3)]],
    device uchar*       codes_out   [[buffer(4)]],
    device float*       rotated_out [[buffer(5)]],  // still needed for residual in kernel 3
    constant uint&      dim         [[buffer(6)]],
    constant uint&      n_vecs      [[buffer(7)]],
    constant uint&      n_levels    [[buffer(8)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint vec_idx = tid.y;
    uint elem_idx = tid.x;
    if (vec_idx >= n_vecs || elem_idx >= dim) return;

    // --- Rotate ---
    float sum = 0.0f;
    for (uint j = 0; j < dim; j++) {
        sum += rotation[elem_idx * dim + j] * kv_vectors[vec_idx * dim + j];
    }

    uint out_idx = vec_idx * dim + elem_idx;

    // --- Normalize (reduction for scale) ---
    // Phase 18.B-RM.12 (2026-05-11): scale is broadcast via threadgroup memory,
    // not via device round-trip. The legacy `scales_out[vec_idx] = scale; ...
    // float scale = scales_out[vec_idx];` pattern relied on a threadgroup-only
    // barrier to make a *device* write visible to peers — Metal's threadgroup
    // barrier does not order device-memory writes, so non-thread-0 lanes could
    // read a stale (often zero) `scales_out[vec_idx]` and compute
    // `normalized = sum / 0`. Symptom: huge stage-1 scores at decode (score
    // kernel multiplies by `scales[kv_idx]`), softmax overflow, NaN attention
    // output. Synthetic bench couldn't catch this because it uses CPU compress
    // (`TurboQuantCompressor::compress`) and only exercises GPU attention.
    threadgroup float partial_sq[256];
    threadgroup float shared_scale;
    partial_sq[elem_idx] = sum * sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (elem_idx == 0) {
        float norm_sq = 0.0f;
        for (uint j = 0; j < dim; j++) {
            norm_sq += partial_sq[j];
        }
        float scale = sqrt(norm_sq / float(dim));
        shared_scale = scale;
        scales_out[vec_idx] = scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = shared_scale;
    float normalized = (scale > 1e-10f) ? (sum / scale) : sum;

    // Write rotated+normalized for kernel 3 residual computation
    rotated_out[out_idx] = normalized;

    // --- Quantize (inline Lloyd-Max) ---
    uint code = n_levels - 1;
    for (uint i = 1; i < n_levels; i++) {
        if (normalized < boundaries[i]) {
            code = i - 1;
            break;
        }
    }
    codes_out[out_idx] = uchar(code);
}

// ============================================================================
// Kernel 3: Bitpack Codes and Compute Residual
// ============================================================================
// Packing matches CPU: codes_per_word = 64/bits, no cross-word boundaries.

kernel void tq_bitpack_and_residual(
    device const uchar*    codes       [[buffer(0)]],
    device const float*    centroids   [[buffer(1)]],
    device const float*    scales      [[buffer(2)]],
    device const float*    rotation    [[buffer(3)]],
    device const float*    kv_orig     [[buffer(4)]],
    device ulong*          packed_out  [[buffer(5)]],
    device float*          residuals   [[buffer(6)]],
    device float*          res_norms   [[buffer(7)]],
    constant uint&         dim         [[buffer(8)]],
    constant uint&         n_vecs      [[buffer(9)]],
    constant uint&         bits        [[buffer(10)]],
    constant uint&         n_packed    [[buffer(11)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint vec_idx = tid.y;
    if (vec_idx >= n_vecs) return;
    uint elem_idx = tid.x;
    if (elem_idx >= dim) return;

    uint code_base = vec_idx * dim;
    float scale = scales[vec_idx];
    uint codes_per_word = 64 / bits;

    // --- Bitpack (thread 0 only) ---
    if (elem_idx == 0) {
        uint pack_base = vec_idx * n_packed;
        for (uint p = 0; p < n_packed; p++) {
            packed_out[pack_base + p] = 0;
        }

        for (uint d = 0; d < dim; d++) {
            uchar code = codes[code_base + d];
            uint word_idx = d / codes_per_word;
            uint bit_pos = (d % codes_per_word) * bits;
            packed_out[pack_base + word_idx] |= (ulong(code) << bit_pos);
        }
    }

    // --- Residual: reconstructed[elem_idx] = sum_j R^T[elem_idx][j] * centroids[codes[j]] * scale ---
    float recon = 0.0f;
    for (uint j = 0; j < dim; j++) {
        float deq = centroids[codes[code_base + j]] * scale;
        recon += rotation[j * dim + elem_idx] * deq;
    }

    float orig = kv_orig[vec_idx * dim + elem_idx];
    float res = orig - recon;
    residuals[vec_idx * dim + elem_idx] = res;

    // Reduce residual norm
    threadgroup float partial_sq[256];
    partial_sq[elem_idx] = res * res;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (elem_idx == 0) {
        float norm_sq = 0.0f;
        for (uint j = 0; j < dim; j++) {
            norm_sq += partial_sq[j];
        }
        res_norms[vec_idx] = sqrt(norm_sq);
    }
}

// ============================================================================
// Kernel 4: QJL Project and Pack Signs
// ============================================================================

// One thread per vector — each thread computes all qjl_m projections serially.
// Avoids 64-bit atomics (unsupported on Metal).
// For qjl_m=64, dim=128: 8192 MADs per thread — fast on Apple Silicon.

kernel void tq_qjl_project_signs(
    device const float* residuals    [[buffer(0)]],
    device const float* qjl_matrix   [[buffer(1)]],
    device ulong*       qjl_packed   [[buffer(2)]],
    constant uint&      dim          [[buffer(3)]],
    constant uint&      n_vecs       [[buffer(4)]],
    constant uint&      qjl_m        [[buffer(5)]],
    constant uint&      n_qjl_packed [[buffer(6)]],
    uint tid_x [[thread_position_in_grid]]
) {
    uint vec_idx = tid_x;
    if (vec_idx >= n_vecs) return;

    uint pack_base = vec_idx * n_qjl_packed;
    for (uint p = 0; p < n_qjl_packed; p++) {
        qjl_packed[pack_base + p] = 0;
    }

    for (uint j = 0; j < qjl_m; j++) {
        float dot = 0.0f;
        for (uint d = 0; d < dim; d++) {
            dot += qjl_matrix[j * dim + d] * residuals[vec_idx * dim + d];
        }
        if (dot >= 0.0f) {
            uint word_idx = j / 64;
            uint bit_pos = j % 64;
            qjl_packed[pack_base + word_idx] |= (ulong(1) << bit_pos);
        }
    }
}

// ============================================================================
// Kernel 5: Compressed Attention Scores (kept for tests)
// ============================================================================
// Unpacking matches CPU bitpack format. QJL correction matches CPU formula.

kernel void tq_compressed_attention_scores(
    device const float* rotated_query [[buffer(0)]],
    device const float* query_orig    [[buffer(1)]],
    device const ulong* packed_codes  [[buffer(2)]],
    device const float* scales        [[buffer(3)]],
    device const float* centroids     [[buffer(4)]],
    device const ulong* qjl_packed    [[buffer(5)]],
    device const float* qjl_matrix    [[buffer(6)]],
    device const float* res_norms     [[buffer(7)]],
    device float*       scores_out    [[buffer(8)]],
    constant uint&      dim           [[buffer(9)]],
    constant uint&      n_kv          [[buffer(10)]],
    constant uint&      bits          [[buffer(11)]],
    constant uint&      n_packed      [[buffer(12)]],
    constant uint&      qjl_m         [[buffer(13)]],
    constant uint&      n_qjl_packed  [[buffer(14)]],
    constant uint&      n_levels      [[buffer(15)]],
    uint tid_x [[thread_position_in_grid]]
) {
    uint kv_idx = tid_x;
    if (kv_idx >= n_kv) return;

    uint codes_per_word = 64 / bits;
    ulong mask = (ulong(1) << bits) - 1;

    // --- Stage 1: dot product in rotated + scaled space ---
    uint pack_base = kv_idx * n_packed;
    float stage1 = 0.0f;

    for (uint d = 0; d < dim; d++) {
        uint word_idx = d / codes_per_word;
        uint bit_pos = (d % codes_per_word) * bits;
        uint code = uint((packed_codes[pack_base + word_idx] >> bit_pos) & mask);
        if (code >= n_levels) code = n_levels - 1;
        stage1 += rotated_query[d] * centroids[code];
    }
    stage1 *= scales[kv_idx];

    // --- Stage 2: QJL correction ---
    // Matches CPU: correction = residual_norm * sqrt(pi/2) / sqrt(m) * Σ sign_r * proj_q
    // where proj_q = qjl_matrix[j] . query (matrix has 1/sqrt(m) baked in)
    float correction = 0.0f;
    uint qjl_base = kv_idx * n_qjl_packed;

    for (uint j = 0; j < qjl_m; j++) {
        float qp = 0.0f;
        for (uint d = 0; d < dim; d++) {
            qp += qjl_matrix[j * dim + d] * query_orig[d];
        }
        uint word = j / 64;
        uint bpos = j % 64;
        bool sign_positive = (qjl_packed[qjl_base + word] >> bpos) & 1;
        // sign_r * proj_q (no fabs — matches CPU exactly)
        correction += sign_positive ? qp : -qp;
    }
    // sqrt(pi/2) = 1.2533141373
    float sqrt_pi_over_2 = 1.2533141373f;
    correction *= res_norms[kv_idx] * sqrt_pi_over_2 / sqrt(float(qjl_m));

    scores_out[kv_idx] = stage1 + correction;
}

// ============================================================================
// Kernel 5b: Fused Scores + Softmax
// ============================================================================
// Combines attention scores and softmax in a single kernel.
// All threads in one threadgroup for parallel softmax reduction.
// REQUIRES: threadgroup_size >= n_kv.

kernel void tq_scores_and_softmax(
    device const float* rotated_query [[buffer(0)]],
    device const float* query_orig    [[buffer(1)]],
    device const ulong* packed_codes  [[buffer(2)]],
    device const float* scales        [[buffer(3)]],
    device const float* centroids     [[buffer(4)]],
    device const ulong* qjl_packed    [[buffer(5)]],
    device const float* qjl_matrix    [[buffer(6)]],
    device const float* res_norms     [[buffer(7)]],
    device float*       scores_out    [[buffer(8)]],
    constant uint&      dim           [[buffer(9)]],
    constant uint&      n_kv          [[buffer(10)]],
    constant uint&      bits          [[buffer(11)]],
    constant uint&      n_packed      [[buffer(12)]],
    constant uint&      qjl_m         [[buffer(13)]],
    constant uint&      n_qjl_packed  [[buffer(14)]],
    constant uint&      n_levels      [[buffer(15)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    // Use device memory for scores (threadgroup memory limited to 32KB)
    // Phase 1: compute scores directly to output buffer
    float score = -INFINITY;
    if (tid < n_kv) {
        uint codes_per_word = 64 / bits;
        ulong mask = (ulong(1) << bits) - 1;
        uint pack_base = tid * n_packed;
        float stage1 = 0.0f;

        for (uint d = 0; d < dim; d++) {
            uint word_idx = d / codes_per_word;
            uint bit_pos = (d % codes_per_word) * bits;
            uint code = uint((packed_codes[pack_base + word_idx] >> bit_pos) & mask);
            if (code >= n_levels) code = n_levels - 1;
            stage1 += rotated_query[d] * centroids[code];
        }
        stage1 *= scales[tid];

        float correction = 0.0f;
        uint qjl_base = tid * n_qjl_packed;
        for (uint j = 0; j < qjl_m; j++) {
            float qp = 0.0f;
            for (uint d = 0; d < dim; d++) {
                qp += qjl_matrix[j * dim + d] * query_orig[d];
            }
            uint word = j / 64;
            uint bpos = j % 64;
            bool sign_positive = (qjl_packed[qjl_base + word] >> bpos) & 1;
            correction += sign_positive ? qp : -qp;
        }
        correction *= res_norms[tid] * 1.2533141373f / sqrt(float(qjl_m));
        score = stage1 + correction;
    }
    scores_out[tid] = score;
    threadgroup_barrier(mem_flags::mem_device);

    // Phase 2: Parallel softmax using threadgroup shared memory for reduction
    threadgroup float shared_reduce[1024];
    shared_reduce[tid] = (tid < n_kv) ? score : -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Find max
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < tg_size) {
            shared_reduce[tid] = max(shared_reduce[tid], shared_reduce[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shared_reduce[0];

    // Exp and parallel sum
    float exp_val = (tid < n_kv) ? exp(score - max_val) : 0.0f;
    shared_reduce[tid] = exp_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < tg_size) {
            shared_reduce[tid] += shared_reduce[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_val = shared_reduce[0];

    // Normalize and write
    if (tid < n_kv) {
        scores_out[tid] = exp_val / sum_val;
    }
}

// ============================================================================
// Kernel 6: Compressed Value Gather (Two-Pass Algorithm)
// ============================================================================
// Swapped summation order for ~dim× speedup (128-256×):
//   Pass 1: inner[j] = Σ_i weights[i] * scales[i] * centroids[code_i[j]]  — O(dim × n_kv)
//   Pass 2: output[d] = Σ_j R^T[d][j] * inner[j]                          — O(dim²) once
// Total: O(dim × n_kv + dim²) vs old O(dim² × n_kv)
// REQUIRES: all threads in a single threadgroup (threadgroup_size = dim).

kernel void tq_compressed_value_gather(
    device const float* weights      [[buffer(0)]],
    device const ulong* packed_codes [[buffer(1)]],
    device const float* scales       [[buffer(2)]],
    device const float* centroids    [[buffer(3)]],
    device const float* rotation     [[buffer(4)]],
    device float*       output       [[buffer(5)]],
    constant uint&      dim          [[buffer(6)]],
    constant uint&      n_kv         [[buffer(7)]],
    constant uint&      bits         [[buffer(8)]],
    constant uint&      n_packed     [[buffer(9)]],
    constant uint&      n_levels     [[buffer(10)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    // Threadgroup shared memory for intermediate accumulation
    threadgroup float inner[512];  // max head_dim = 512

    uint j = tid;
    if (j >= dim) return;

    uint codes_per_word = 64 / bits;
    ulong code_mask = (ulong(1) << bits) - 1;

    // Pass 1: For position j, accumulate weighted centroids across all KV vectors
    float acc = 0.0f;
    for (uint i = 0; i < n_kv; i++) {
        float w = weights[i];
        if (fabs(w) < 1e-8f) continue;

        float scale = scales[i];
        uint pack_base = i * n_packed;

        uint word_idx = j / codes_per_word;
        uint bit_pos = (j % codes_per_word) * bits;
        uint code = uint((packed_codes[pack_base + word_idx] >> bit_pos) & code_mask);
        if (code >= n_levels) code = n_levels - 1;

        acc += w * scale * centroids[code];
    }
    inner[j] = acc;

    // Synchronize: all threads must finish pass 1 before pass 2
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: Inverse rotation — output[d] = Σ_j R^T[d][j] * inner[j]
    uint out_d = tid;
    float result = 0.0f;
    for (uint k = 0; k < dim; k++) {
        result += rotation[k * dim + out_d] * inner[k];
    }
    output[out_d] = result;
}

// ============================================================================
// Kernel 6b: Compressed value gather, fan-out to multiple Q heads (GQA)
// ============================================================================
// Identical math to tq_compressed_value_gather, but writes the result to
// `gqa_ratio` consecutive output slices (one per Q head sharing this KV head).
// In current production GQA pipeline, all Q heads in one KV group share the
// same softmax weights and the same V cache → identical compute. This kernel
// runs that compute ONCE and fans out the result to all `gqa_ratio` slots,
// saving (gqa_ratio - 1) redundant kernel invocations per KV head.
// Single-threadgroup, dim threads. Output layout: output_base[qh*dim + d]
// for qh in [first_qh, first_qh + gqa_ratio).

kernel void tq_compressed_value_gather_multi(
    device const float* weights      [[buffer(0)]],
    device const ulong* packed_codes [[buffer(1)]],
    device const float* scales       [[buffer(2)]],
    device const float* centroids    [[buffer(3)]],
    device const float* rotation     [[buffer(4)]],
    device float*       output_base  [[buffer(5)]],
    constant uint&      dim          [[buffer(6)]],
    constant uint&      n_kv         [[buffer(7)]],
    constant uint&      bits         [[buffer(8)]],
    constant uint&      n_packed     [[buffer(9)]],
    constant uint&      n_levels     [[buffer(10)]],
    constant uint&      gqa_ratio    [[buffer(11)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    threadgroup float inner[512];

    uint j = tid;
    if (j >= dim) return;

    uint codes_per_word = 64 / bits;
    ulong code_mask = (ulong(1) << bits) - 1;

    // Pass 1: weighted centroid accumulation
    float acc = 0.0f;
    for (uint i = 0; i < n_kv; i++) {
        float w = weights[i];
        if (fabs(w) < 1e-8f) continue;

        float scale = scales[i];
        uint pack_base = i * n_packed;

        uint word_idx = j / codes_per_word;
        uint bit_pos = (j % codes_per_word) * bits;
        uint code = uint((packed_codes[pack_base + word_idx] >> bit_pos) & code_mask);
        if (code >= n_levels) code = n_levels - 1;

        acc += w * scale * centroids[code];
    }
    inner[j] = acc;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: inverse rotation, then fan out
    uint out_d = tid;
    float result = 0.0f;
    for (uint k = 0; k < dim; k++) {
        result += rotation[k * dim + out_d] * inner[k];
    }
    for (uint q = 0; q < gqa_ratio; q++) {
        output_base[q * dim + out_d] = result;
    }
}

// ============================================================================
// Kernel 7.5: Rotate query (no normalization)
// ============================================================================
// y[i] = sum_j R[i][j] * q[j]
// One thread per output element. No scale/normalize.

kernel void tq_rotate_query(
    device const float* query    [[buffer(0)]],   // [dim] or [n_queries * dim]
    device const float* rotation [[buffer(1)]],   // [dim, dim]
    device float*       out      [[buffer(2)]],   // [dim] or [n_queries * dim]
    constant uint&      dim      [[buffer(3)]],
    constant uint&      n_queries [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint q_idx = tid.y;
    uint elem = tid.x;
    if (q_idx >= n_queries || elem >= dim) return;

    float sum = 0.0f;
    for (uint j = 0; j < dim; j++) {
        sum += rotation[elem * dim + j] * query[q_idx * dim + j];
    }
    out[q_idx * dim + elem] = sum;
}

// ============================================================================
// Kernel 4b: QJL Project Query (precompute)
// ============================================================================
// qjl_proj[j] = sum_d qjl_matrix[j*dim + d] * query[d]   for j in [0, qjl_m)
//
// Computed ONCE per attention call, replacing per-kv_idx recomputation in
// kernels 5 and 5b. Eliminates O(qjl_m * dim * n_kv) redundancy.
// One thread per j.

kernel void tq_qjl_project_query(
    device const float* query        [[buffer(0)]],
    device const float* qjl_matrix   [[buffer(1)]],
    device float*       qjl_proj_out [[buffer(2)]],
    constant uint&      dim          [[buffer(3)]],
    constant uint&      qjl_m        [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint j = tid;
    if (j >= qjl_m) return;

    uint base = j * dim;
    float dot = 0.0f;
    for (uint d = 0; d < dim; d++) {
        dot += qjl_matrix[base + d] * query[d];
    }
    qjl_proj_out[j] = dot;
}

// ============================================================================
// Kernel 5_v2: Compressed attention scores (precomputed QJL projection)
// ============================================================================
// Identical math to tq_compressed_attention_scores but consumes pre-projected
// qjl_proj[qjl_m] instead of recomputing qjl_matrix . query per kv_idx.

kernel void tq_compressed_attention_scores_v2(
    device const float* rotated_query [[buffer(0)]],
    device const float* qjl_proj      [[buffer(1)]],
    device const ulong* packed_codes  [[buffer(2)]],
    device const float* scales        [[buffer(3)]],
    device const float* centroids     [[buffer(4)]],
    device const ulong* qjl_packed    [[buffer(5)]],
    device const float* res_norms     [[buffer(6)]],
    device float*       scores_out    [[buffer(7)]],
    constant uint&      dim           [[buffer(8)]],
    constant uint&      n_kv          [[buffer(9)]],
    constant uint&      bits          [[buffer(10)]],
    constant uint&      n_packed      [[buffer(11)]],
    constant uint&      qjl_m         [[buffer(12)]],
    constant uint&      n_qjl_packed  [[buffer(13)]],
    constant uint&      n_levels      [[buffer(14)]],
    uint tid_x [[thread_position_in_grid]]
) {
    uint kv_idx = tid_x;
    if (kv_idx >= n_kv) return;

    uint codes_per_word = 64 / bits;
    ulong mask = (ulong(1) << bits) - 1;

    // Stage 1
    uint pack_base = kv_idx * n_packed;
    float stage1 = 0.0f;
    for (uint d = 0; d < dim; d++) {
        uint word_idx = d / codes_per_word;
        uint bit_pos = (d % codes_per_word) * bits;
        uint code = uint((packed_codes[pack_base + word_idx] >> bit_pos) & mask);
        if (code >= n_levels) code = n_levels - 1;
        stage1 += rotated_query[d] * centroids[code];
    }
    stage1 *= scales[kv_idx];

    // Stage 2: QJL correction using precomputed projection
    float correction = 0.0f;
    uint qjl_base = kv_idx * n_qjl_packed;
    for (uint j = 0; j < qjl_m; j++) {
        float qp = qjl_proj[j];
        uint word = j / 64;
        uint bpos = j % 64;
        bool sign_positive = (qjl_packed[qjl_base + word] >> bpos) & 1;
        correction += sign_positive ? qp : -qp;
    }
    correction *= res_norms[kv_idx] * 1.2533141373f / sqrt(float(qjl_m));

    scores_out[kv_idx] = stage1 + correction;
}

// ============================================================================
// Kernel 5b_v2: Fused scores + softmax (precomputed QJL projection)
// ============================================================================

kernel void tq_scores_and_softmax_v2(
    device const float* rotated_query [[buffer(0)]],
    device const float* qjl_proj      [[buffer(1)]],
    device const ulong* packed_codes  [[buffer(2)]],
    device const float* scales        [[buffer(3)]],
    device const float* centroids     [[buffer(4)]],
    device const ulong* qjl_packed    [[buffer(5)]],
    device const float* res_norms     [[buffer(6)]],
    device float*       scores_out    [[buffer(7)]],
    constant uint&      dim           [[buffer(8)]],
    constant uint&      n_kv          [[buffer(9)]],
    constant uint&      bits          [[buffer(10)]],
    constant uint&      n_packed      [[buffer(11)]],
    constant uint&      qjl_m         [[buffer(12)]],
    constant uint&      n_qjl_packed  [[buffer(13)]],
    constant uint&      n_levels      [[buffer(14)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    float score = -INFINITY;
    if (tid < n_kv) {
        uint codes_per_word = 64 / bits;
        ulong mask = (ulong(1) << bits) - 1;
        uint pack_base = tid * n_packed;
        float stage1 = 0.0f;
        for (uint d = 0; d < dim; d++) {
            uint word_idx = d / codes_per_word;
            uint bit_pos = (d % codes_per_word) * bits;
            uint code = uint((packed_codes[pack_base + word_idx] >> bit_pos) & mask);
            if (code >= n_levels) code = n_levels - 1;
            stage1 += rotated_query[d] * centroids[code];
        }
        stage1 *= scales[tid];

        float correction = 0.0f;
        uint qjl_base = tid * n_qjl_packed;
        for (uint j = 0; j < qjl_m; j++) {
            float qp = qjl_proj[j];
            uint word = j / 64;
            uint bpos = j % 64;
            bool sign_positive = (qjl_packed[qjl_base + word] >> bpos) & 1;
            correction += sign_positive ? qp : -qp;
        }
        correction *= res_norms[tid] * 1.2533141373f / sqrt(float(qjl_m));
        score = stage1 + correction;
    }
    scores_out[tid] = score;
    threadgroup_barrier(mem_flags::mem_device);

    threadgroup float shared_reduce[1024];
    shared_reduce[tid] = (tid < n_kv) ? score : -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < tg_size) {
            shared_reduce[tid] = max(shared_reduce[tid], shared_reduce[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shared_reduce[0];

    float exp_val = (tid < n_kv) ? exp(score - max_val) : 0.0f;
    shared_reduce[tid] = exp_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < tg_size) {
            shared_reduce[tid] += shared_reduce[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_val = shared_reduce[0];

    if (tid < n_kv) {
        scores_out[tid] = exp_val / sum_val;
    }
}

// ============================================================================
// Kernel 5_v3: Compressed attention scores (precomputed QJL + float4 Stage 1)
// ============================================================================
// Same as v2 but Stage 1 inner loop processes 4 elements per iteration.
// rotated_query[d:d+4] loaded as float4; 4 codes unpacked to a float4 of
// centroids; single dot4 + FMA. Reduces inner loop trip count by 4×.
// dim must be a multiple of 4 (asserted by Rust dispatch).

inline float tq_dequant_one(
    device const ulong* packed_codes,
    device const float* centroids,
    uint pack_base,
    uint d,
    uint codes_per_word,
    uint bits,
    ulong mask,
    uint n_levels
) {
    uint word_idx = d / codes_per_word;
    uint bit_pos = (d % codes_per_word) * bits;
    uint code = uint((packed_codes[pack_base + word_idx] >> bit_pos) & mask);
    if (code >= n_levels) code = n_levels - 1;
    return centroids[code];
}

kernel void tq_compressed_attention_scores_v3(
    device const float* rotated_query [[buffer(0)]],
    device const float* qjl_proj      [[buffer(1)]],
    device const ulong* packed_codes  [[buffer(2)]],
    device const float* scales        [[buffer(3)]],
    device const float* centroids     [[buffer(4)]],
    device const ulong* qjl_packed    [[buffer(5)]],
    device const float* res_norms     [[buffer(6)]],
    device float*       scores_out    [[buffer(7)]],
    constant uint&      dim           [[buffer(8)]],
    constant uint&      n_kv          [[buffer(9)]],
    constant uint&      bits          [[buffer(10)]],
    constant uint&      n_packed      [[buffer(11)]],
    constant uint&      qjl_m         [[buffer(12)]],
    constant uint&      n_qjl_packed  [[buffer(13)]],
    constant uint&      n_levels      [[buffer(14)]],
    uint tid_x [[thread_position_in_grid]]
) {
    uint kv_idx = tid_x;
    if (kv_idx >= n_kv) return;

    uint codes_per_word = 64 / bits;
    ulong mask = (ulong(1) << bits) - 1;
    uint pack_base = kv_idx * n_packed;

    // Stage 1 — float4 vectorized
    device const float4* rq4 = (device const float4*)rotated_query;
    uint dim4 = dim / 4;
    float stage1 = 0.0f;
    for (uint d4 = 0; d4 < dim4; d4++) {
        uint d = d4 * 4;
        float4 cents = float4(
            tq_dequant_one(packed_codes, centroids, pack_base, d,     codes_per_word, bits, mask, n_levels),
            tq_dequant_one(packed_codes, centroids, pack_base, d + 1, codes_per_word, bits, mask, n_levels),
            tq_dequant_one(packed_codes, centroids, pack_base, d + 2, codes_per_word, bits, mask, n_levels),
            tq_dequant_one(packed_codes, centroids, pack_base, d + 3, codes_per_word, bits, mask, n_levels)
        );
        stage1 += dot(rq4[d4], cents);
    }
    stage1 *= scales[kv_idx];

    // Stage 2 — same as v2 (precomputed projection)
    float correction = 0.0f;
    uint qjl_base = kv_idx * n_qjl_packed;
    for (uint j = 0; j < qjl_m; j++) {
        float qp = qjl_proj[j];
        uint word = j / 64;
        uint bpos = j % 64;
        bool sign_positive = (qjl_packed[qjl_base + word] >> bpos) & 1;
        correction += sign_positive ? qp : -qp;
    }
    correction *= res_norms[kv_idx] * 1.2533141373f / sqrt(float(qjl_m));

    scores_out[kv_idx] = stage1 + correction;
}

kernel void tq_scores_and_softmax_v3(
    device const float* rotated_query [[buffer(0)]],
    device const float* qjl_proj      [[buffer(1)]],
    device const ulong* packed_codes  [[buffer(2)]],
    device const float* scales        [[buffer(3)]],
    device const float* centroids     [[buffer(4)]],
    device const ulong* qjl_packed    [[buffer(5)]],
    device const float* res_norms     [[buffer(6)]],
    device float*       scores_out    [[buffer(7)]],
    constant uint&      dim           [[buffer(8)]],
    constant uint&      n_kv          [[buffer(9)]],
    constant uint&      bits          [[buffer(10)]],
    constant uint&      n_packed      [[buffer(11)]],
    constant uint&      qjl_m         [[buffer(12)]],
    constant uint&      n_qjl_packed  [[buffer(13)]],
    constant uint&      n_levels      [[buffer(14)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    float score = -INFINITY;
    if (tid < n_kv) {
        uint codes_per_word = 64 / bits;
        ulong mask = (ulong(1) << bits) - 1;
        uint pack_base = tid * n_packed;

        device const float4* rq4 = (device const float4*)rotated_query;
        uint dim4 = dim / 4;
        float stage1 = 0.0f;
        for (uint d4 = 0; d4 < dim4; d4++) {
            uint d = d4 * 4;
            float4 cents = float4(
                tq_dequant_one(packed_codes, centroids, pack_base, d,     codes_per_word, bits, mask, n_levels),
                tq_dequant_one(packed_codes, centroids, pack_base, d + 1, codes_per_word, bits, mask, n_levels),
                tq_dequant_one(packed_codes, centroids, pack_base, d + 2, codes_per_word, bits, mask, n_levels),
                tq_dequant_one(packed_codes, centroids, pack_base, d + 3, codes_per_word, bits, mask, n_levels)
            );
            stage1 += dot(rq4[d4], cents);
        }
        stage1 *= scales[tid];

        float correction = 0.0f;
        uint qjl_base = tid * n_qjl_packed;
        for (uint j = 0; j < qjl_m; j++) {
            float qp = qjl_proj[j];
            uint word = j / 64;
            uint bpos = j % 64;
            bool sign_positive = (qjl_packed[qjl_base + word] >> bpos) & 1;
            correction += sign_positive ? qp : -qp;
        }
        correction *= res_norms[tid] * 1.2533141373f / sqrt(float(qjl_m));
        score = stage1 + correction;
    }
    scores_out[tid] = score;
    threadgroup_barrier(mem_flags::mem_device);

    threadgroup float shared_reduce[1024];
    shared_reduce[tid] = (tid < n_kv) ? score : -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < tg_size) {
            shared_reduce[tid] = max(shared_reduce[tid], shared_reduce[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shared_reduce[0];

    float exp_val = (tid < n_kv) ? exp(score - max_val) : 0.0f;
    shared_reduce[tid] = exp_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < tg_size) {
            shared_reduce[tid] += shared_reduce[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_val = shared_reduce[0];

    if (tid < n_kv) {
        scores_out[tid] = exp_val / sum_val;
    }
}

// ============================================================================
// Kernel 5b_v4: Fused scores + softmax (v3 + centroids in threadgroup memory)
// ============================================================================
// Caches centroids[0..n_levels] in threadgroup shared memory once per
// threadgroup. Removes O(dim) global-memory loads of centroids per thread.
// Same numeric output as v3 (just different load path).
// REQUIRES: all threads in one threadgroup (single-threadgroup dispatch).

kernel void tq_scores_and_softmax_v4(
    device const float* rotated_query [[buffer(0)]],
    device const float* qjl_proj      [[buffer(1)]],
    device const ulong* packed_codes  [[buffer(2)]],
    device const float* scales        [[buffer(3)]],
    device const float* centroids     [[buffer(4)]],
    device const ulong* qjl_packed    [[buffer(5)]],
    device const float* res_norms     [[buffer(6)]],
    device float*       scores_out    [[buffer(7)]],
    constant uint&      dim           [[buffer(8)]],
    constant uint&      n_kv          [[buffer(9)]],
    constant uint&      bits          [[buffer(10)]],
    constant uint&      n_packed      [[buffer(11)]],
    constant uint&      qjl_m         [[buffer(12)]],
    constant uint&      n_qjl_packed  [[buffer(13)]],
    constant uint&      n_levels      [[buffer(14)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    // Cache centroids (max 256 levels for 8-bit; typically 8 for 3-bit)
    threadgroup float shared_centroids[256];
    if (tid < n_levels) {
        shared_centroids[tid] = centroids[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float score = -INFINITY;
    if (tid < n_kv) {
        uint codes_per_word = 64 / bits;
        ulong mask = (ulong(1) << bits) - 1;
        uint pack_base = tid * n_packed;

        device const float4* rq4 = (device const float4*)rotated_query;
        uint dim4 = dim / 4;
        float stage1 = 0.0f;
        for (uint d4 = 0; d4 < dim4; d4++) {
            uint d = d4 * 4;
            // Inline 4 dequant lookups via shared centroids
            uint w0 = (d    ) / codes_per_word, b0 = ((d    ) % codes_per_word) * bits;
            uint w1 = (d + 1) / codes_per_word, b1 = ((d + 1) % codes_per_word) * bits;
            uint w2 = (d + 2) / codes_per_word, b2 = ((d + 2) % codes_per_word) * bits;
            uint w3 = (d + 3) / codes_per_word, b3 = ((d + 3) % codes_per_word) * bits;
            uint c0 = uint((packed_codes[pack_base + w0] >> b0) & mask);
            uint c1 = uint((packed_codes[pack_base + w1] >> b1) & mask);
            uint c2 = uint((packed_codes[pack_base + w2] >> b2) & mask);
            uint c3 = uint((packed_codes[pack_base + w3] >> b3) & mask);
            if (c0 >= n_levels) c0 = n_levels - 1;
            if (c1 >= n_levels) c1 = n_levels - 1;
            if (c2 >= n_levels) c2 = n_levels - 1;
            if (c3 >= n_levels) c3 = n_levels - 1;
            float4 cents = float4(
                shared_centroids[c0],
                shared_centroids[c1],
                shared_centroids[c2],
                shared_centroids[c3]
            );
            stage1 += dot(rq4[d4], cents);
        }
        stage1 *= scales[tid];

        float correction = 0.0f;
        uint qjl_base = tid * n_qjl_packed;
        for (uint j = 0; j < qjl_m; j++) {
            float qp = qjl_proj[j];
            uint word = j / 64;
            uint bpos = j % 64;
            bool sign_positive = (qjl_packed[qjl_base + word] >> bpos) & 1;
            correction += sign_positive ? qp : -qp;
        }
        correction *= res_norms[tid] * 1.2533141373f / sqrt(float(qjl_m));
        score = stage1 + correction;
    }
    scores_out[tid] = score;
    threadgroup_barrier(mem_flags::mem_device);

    threadgroup float shared_reduce[1024];
    shared_reduce[tid] = (tid < n_kv) ? score : -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < tg_size) {
            shared_reduce[tid] = max(shared_reduce[tid], shared_reduce[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shared_reduce[0];

    float exp_val = (tid < n_kv) ? exp(score - max_val) : 0.0f;
    shared_reduce[tid] = exp_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < tg_size) {
            shared_reduce[tid] += shared_reduce[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_val = shared_reduce[0];

    if (tid < n_kv) {
        scores_out[tid] = exp_val / sum_val;
    }
}

// ============================================================================
// Kernel 5_v4 (non-fused): Compressed scores with shared-centroids cache.
// ============================================================================
// Same as v3 but threadgroup memory caches centroids[0..n_levels].
// Multi-threadgroup dispatch — caller passes [grid=n_kv, threadgroup=any].
// Each threadgroup loads centroids once into shared memory, then all threads
// in that group reuse the cache for the full Stage 1 inner loop.

kernel void tq_compressed_attention_scores_v4(
    device const float* rotated_query [[buffer(0)]],
    device const float* qjl_proj      [[buffer(1)]],
    device const ulong* packed_codes  [[buffer(2)]],
    device const float* scales        [[buffer(3)]],
    device const float* centroids     [[buffer(4)]],
    device const ulong* qjl_packed    [[buffer(5)]],
    device const float* res_norms     [[buffer(6)]],
    device float*       scores_out    [[buffer(7)]],
    constant uint&      dim           [[buffer(8)]],
    constant uint&      n_kv          [[buffer(9)]],
    constant uint&      bits          [[buffer(10)]],
    constant uint&      n_packed      [[buffer(11)]],
    constant uint&      qjl_m         [[buffer(12)]],
    constant uint&      n_qjl_packed  [[buffer(13)]],
    constant uint&      n_levels      [[buffer(14)]],
    uint kv_idx [[thread_position_in_grid]],
    uint tg_local [[thread_position_in_threadgroup]]
) {
    threadgroup float shared_centroids[256];
    if (tg_local < n_levels) {
        shared_centroids[tg_local] = centroids[tg_local];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (kv_idx >= n_kv) return;

    uint codes_per_word = 64 / bits;
    ulong mask = (ulong(1) << bits) - 1;
    uint pack_base = kv_idx * n_packed;

    device const float4* rq4 = (device const float4*)rotated_query;
    uint dim4 = dim / 4;
    float stage1 = 0.0f;
    for (uint d4 = 0; d4 < dim4; d4++) {
        uint d = d4 * 4;
        uint w0 = (d    ) / codes_per_word, b0 = ((d    ) % codes_per_word) * bits;
        uint w1 = (d + 1) / codes_per_word, b1 = ((d + 1) % codes_per_word) * bits;
        uint w2 = (d + 2) / codes_per_word, b2 = ((d + 2) % codes_per_word) * bits;
        uint w3 = (d + 3) / codes_per_word, b3 = ((d + 3) % codes_per_word) * bits;
        uint c0 = uint((packed_codes[pack_base + w0] >> b0) & mask);
        uint c1 = uint((packed_codes[pack_base + w1] >> b1) & mask);
        uint c2 = uint((packed_codes[pack_base + w2] >> b2) & mask);
        uint c3 = uint((packed_codes[pack_base + w3] >> b3) & mask);
        if (c0 >= n_levels) c0 = n_levels - 1;
        if (c1 >= n_levels) c1 = n_levels - 1;
        if (c2 >= n_levels) c2 = n_levels - 1;
        if (c3 >= n_levels) c3 = n_levels - 1;
        float4 cents = float4(
            shared_centroids[c0],
            shared_centroids[c1],
            shared_centroids[c2],
            shared_centroids[c3]
        );
        stage1 += dot(rq4[d4], cents);
    }
    stage1 *= scales[kv_idx];

    float correction = 0.0f;
    uint qjl_base = kv_idx * n_qjl_packed;
    for (uint j = 0; j < qjl_m; j++) {
        float qp = qjl_proj[j];
        uint word = j / 64;
        uint bpos = j % 64;
        bool sign_positive = (qjl_packed[qjl_base + word] >> bpos) & 1;
        correction += sign_positive ? qp : -qp;
    }
    correction *= res_norms[kv_idx] * 1.2533141373f / sqrt(float(qjl_m));
    scores_out[kv_idx] = stage1 + correction;
}

// ============================================================================
// Kernel 5_v5: Compressed scores with fp16 Stage 1 (V3 + half precision FMA)
// ============================================================================
// Centroids cached in threadgroup memory as half (2× smaller than v4).
// rotated_query loaded as float4 then cast to half4 inline.
// Stage 1 inner loop: half4 dot-product, f32 accumulation.
// Stage 2 unchanged (f32) — qjl loop is dominated by sign-bit branch, no win
// from half precision there.
// Multi-threadgroup dispatch (correctly handles n_kv > max_threadgroup).

kernel void tq_compressed_attention_scores_v5(
    device const float* rotated_query [[buffer(0)]],
    device const float* qjl_proj      [[buffer(1)]],
    device const ulong* packed_codes  [[buffer(2)]],
    device const float* scales        [[buffer(3)]],
    device const float* centroids     [[buffer(4)]],
    device const ulong* qjl_packed    [[buffer(5)]],
    device const float* res_norms     [[buffer(6)]],
    device float*       scores_out    [[buffer(7)]],
    constant uint&      dim           [[buffer(8)]],
    constant uint&      n_kv          [[buffer(9)]],
    constant uint&      bits          [[buffer(10)]],
    constant uint&      n_packed      [[buffer(11)]],
    constant uint&      qjl_m         [[buffer(12)]],
    constant uint&      n_qjl_packed  [[buffer(13)]],
    constant uint&      n_levels      [[buffer(14)]],
    uint kv_idx [[thread_position_in_grid]],
    uint tg_local [[thread_position_in_threadgroup]]
) {
    threadgroup half shared_centroids_h[256];
    if (tg_local < n_levels) {
        shared_centroids_h[tg_local] = half(centroids[tg_local]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (kv_idx >= n_kv) return;

    uint codes_per_word = 64 / bits;
    ulong mask = (ulong(1) << bits) - 1;
    uint pack_base = kv_idx * n_packed;

    device const float4* rq4 = (device const float4*)rotated_query;
    uint dim4 = dim / 4;
    float stage1 = 0.0f;
    for (uint d4 = 0; d4 < dim4; d4++) {
        uint d = d4 * 4;
        uint w0 = (d    ) / codes_per_word, b0 = ((d    ) % codes_per_word) * bits;
        uint w1 = (d + 1) / codes_per_word, b1 = ((d + 1) % codes_per_word) * bits;
        uint w2 = (d + 2) / codes_per_word, b2 = ((d + 2) % codes_per_word) * bits;
        uint w3 = (d + 3) / codes_per_word, b3 = ((d + 3) % codes_per_word) * bits;
        uint c0 = uint((packed_codes[pack_base + w0] >> b0) & mask);
        uint c1 = uint((packed_codes[pack_base + w1] >> b1) & mask);
        uint c2 = uint((packed_codes[pack_base + w2] >> b2) & mask);
        uint c3 = uint((packed_codes[pack_base + w3] >> b3) & mask);
        if (c0 >= n_levels) c0 = n_levels - 1;
        if (c1 >= n_levels) c1 = n_levels - 1;
        if (c2 >= n_levels) c2 = n_levels - 1;
        if (c3 >= n_levels) c3 = n_levels - 1;
        half4 cents_h = half4(
            shared_centroids_h[c0],
            shared_centroids_h[c1],
            shared_centroids_h[c2],
            shared_centroids_h[c3]
        );
        half4 rq_h = half4(rq4[d4]);
        stage1 += float(dot(rq_h, cents_h));
    }
    stage1 *= scales[kv_idx];

    float correction = 0.0f;
    uint qjl_base = kv_idx * n_qjl_packed;
    for (uint j = 0; j < qjl_m; j++) {
        float qp = qjl_proj[j];
        uint word = j / 64;
        uint bpos = j % 64;
        bool sign_positive = (qjl_packed[qjl_base + word] >> bpos) & 1;
        correction += sign_positive ? qp : -qp;
    }
    correction *= res_norms[kv_idx] * 1.2533141373f / sqrt(float(qjl_m));
    scores_out[kv_idx] = stage1 + correction;
}

// ============================================================================
// Kernel 5b_v5: Fused scores + softmax (v5 + parallel softmax reduction)
// ============================================================================
// Same Stage 1 as v5 (half precision compute) plus inline parallel softmax.
// REQUIRES single-threadgroup dispatch (parallel reduce uses tg shared memory).
// Caller must ensure tg_size = next_pow2(n_kv).min(max_tg) and n_kv ≤ max_tg.

kernel void tq_scores_and_softmax_v5(
    device const float* rotated_query [[buffer(0)]],
    device const float* qjl_proj      [[buffer(1)]],
    device const ulong* packed_codes  [[buffer(2)]],
    device const float* scales        [[buffer(3)]],
    device const float* centroids     [[buffer(4)]],
    device const ulong* qjl_packed    [[buffer(5)]],
    device const float* res_norms     [[buffer(6)]],
    device float*       scores_out    [[buffer(7)]],
    constant uint&      dim           [[buffer(8)]],
    constant uint&      n_kv          [[buffer(9)]],
    constant uint&      bits          [[buffer(10)]],
    constant uint&      n_packed      [[buffer(11)]],
    constant uint&      qjl_m         [[buffer(12)]],
    constant uint&      n_qjl_packed  [[buffer(13)]],
    constant uint&      n_levels      [[buffer(14)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup half shared_centroids_h[256];
    if (tid < n_levels) {
        shared_centroids_h[tid] = half(centroids[tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float score = -INFINITY;
    if (tid < n_kv) {
        uint codes_per_word = 64 / bits;
        ulong mask = (ulong(1) << bits) - 1;
        uint pack_base = tid * n_packed;

        device const float4* rq4 = (device const float4*)rotated_query;
        uint dim4 = dim / 4;
        float stage1 = 0.0f;
        for (uint d4 = 0; d4 < dim4; d4++) {
            uint d = d4 * 4;
            uint w0 = (d    ) / codes_per_word, b0 = ((d    ) % codes_per_word) * bits;
            uint w1 = (d + 1) / codes_per_word, b1 = ((d + 1) % codes_per_word) * bits;
            uint w2 = (d + 2) / codes_per_word, b2 = ((d + 2) % codes_per_word) * bits;
            uint w3 = (d + 3) / codes_per_word, b3 = ((d + 3) % codes_per_word) * bits;
            uint c0 = uint((packed_codes[pack_base + w0] >> b0) & mask);
            uint c1 = uint((packed_codes[pack_base + w1] >> b1) & mask);
            uint c2 = uint((packed_codes[pack_base + w2] >> b2) & mask);
            uint c3 = uint((packed_codes[pack_base + w3] >> b3) & mask);
            if (c0 >= n_levels) c0 = n_levels - 1;
            if (c1 >= n_levels) c1 = n_levels - 1;
            if (c2 >= n_levels) c2 = n_levels - 1;
            if (c3 >= n_levels) c3 = n_levels - 1;
            half4 cents_h = half4(
                shared_centroids_h[c0],
                shared_centroids_h[c1],
                shared_centroids_h[c2],
                shared_centroids_h[c3]
            );
            half4 rq_h = half4(rq4[d4]);
            stage1 += float(dot(rq_h, cents_h));
        }
        stage1 *= scales[tid];

        float correction = 0.0f;
        uint qjl_base = tid * n_qjl_packed;
        for (uint j = 0; j < qjl_m; j++) {
            float qp = qjl_proj[j];
            uint word = j / 64;
            uint bpos = j % 64;
            bool sign_positive = (qjl_packed[qjl_base + word] >> bpos) & 1;
            correction += sign_positive ? qp : -qp;
        }
        correction *= res_norms[tid] * 1.2533141373f / sqrt(float(qjl_m));
        score = stage1 + correction;
    }
    scores_out[tid] = score;
    threadgroup_barrier(mem_flags::mem_device);

    threadgroup float shared_reduce[1024];
    shared_reduce[tid] = (tid < n_kv) ? score : -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < tg_size) {
            shared_reduce[tid] = max(shared_reduce[tid], shared_reduce[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shared_reduce[0];

    float exp_val = (tid < n_kv) ? exp(score - max_val) : 0.0f;
    shared_reduce[tid] = exp_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < tg_size) {
            shared_reduce[tid] += shared_reduce[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_val = shared_reduce[0];

    if (tid < n_kv) {
        scores_out[tid] = exp_val / sum_val;
    }
}

// ============================================================================
// Kernel 5_v6: Compressed scores with SIMD-group cooperative reduction over dim
// ============================================================================
// Each kv_idx is processed by ONE simd-group (32 threads cooperatively).
// Each lane handles dim/32 chunks of Stage 1 inner loop, then simd_sum reduces.
// At dim=512, inner loop becomes 16 iters/lane (vs v3's 128 iters/thread) —
// targets 4-16× compute parallelism in the dot-product.
// Stage 2 (qjl_m loop) also uses cooperative simd-group reduction.
// Threadgroup = 8 simd-groups (256 threads) → 8 kv_idx per TG.
// Multi-threadgroup dispatch: grid = ceil(n_kv/8) TGs.

kernel void tq_compressed_attention_scores_v6(
    device const float* rotated_query [[buffer(0)]],
    device const float* qjl_proj      [[buffer(1)]],
    device const ulong* packed_codes  [[buffer(2)]],
    device const float* scales        [[buffer(3)]],
    device const float* centroids     [[buffer(4)]],
    device const ulong* qjl_packed    [[buffer(5)]],
    device const float* res_norms     [[buffer(6)]],
    device float*       scores_out    [[buffer(7)]],
    constant uint&      dim           [[buffer(8)]],
    constant uint&      n_kv          [[buffer(9)]],
    constant uint&      bits          [[buffer(10)]],
    constant uint&      n_packed      [[buffer(11)]],
    constant uint&      qjl_m         [[buffer(12)]],
    constant uint&      n_qjl_packed  [[buffer(13)]],
    constant uint&      n_levels      [[buffer(14)]],
    uint tid_in_tg [[thread_position_in_threadgroup]],
    uint tg_size   [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]],
    uint tg_pos    [[threadgroup_position_in_grid]]
) {
    const uint SIMD_SIZE = 32;
    uint kv_per_tg = tg_size / SIMD_SIZE;
    uint kv_idx = tg_pos * kv_per_tg + simd_id;

    // Cooperative load of centroids cache (all threads in TG help)
    threadgroup float shared_centroids[256];
    if (tid_in_tg < n_levels) {
        shared_centroids[tid_in_tg] = centroids[tid_in_tg];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (kv_idx >= n_kv) return;

    uint codes_per_word = 64 / bits;
    ulong mask = (ulong(1) << bits) - 1;
    uint pack_base = kv_idx * n_packed;

    // Stage 1 — simd-group cooperative dot product
    float partial1 = 0.0f;
    for (uint d = simd_lane; d < dim; d += SIMD_SIZE) {
        uint word_idx = d / codes_per_word;
        uint bit_pos = (d % codes_per_word) * bits;
        uint code = uint((packed_codes[pack_base + word_idx] >> bit_pos) & mask);
        if (code >= n_levels) code = n_levels - 1;
        partial1 += rotated_query[d] * shared_centroids[code];
    }
    float stage1 = simd_sum(partial1) * scales[kv_idx];

    // Phase 18.B-RM.12 diagnostic: clamp stage1 to a sane range before softmax
    // to expose whether overflow lives in stage1 vs QJL correction. Stage1
    // shape ≈ Q·K (raw, no 1/sqrt(d_k) scale). Typical range ±10 after qknorm;
    // anything beyond ±50 means upstream scales[] / centroids[] / packed_codes
    // produced a poisoned value.
    if (stage1 != stage1) stage1 = 0.0f;
    if (stage1 > 1e6f) stage1 = 1e6f;
    if (stage1 < -1e6f) stage1 = -1e6f;

    // Stage 2 — simd-group cooperative QJL correction
    float partial2 = 0.0f;
    uint qjl_base = kv_idx * n_qjl_packed;
    for (uint j = simd_lane; j < qjl_m; j += SIMD_SIZE) {
        float qp = qjl_proj[j];
        uint word = j / 64;
        uint bpos = j % 64;
        bool sign_positive = (qjl_packed[qjl_base + word] >> bpos) & 1;
        partial2 += sign_positive ? qp : -qp;
    }
    float correction = simd_sum(partial2) * res_norms[kv_idx] * 1.2533141373f / sqrt(float(qjl_m));
    if (correction != correction) correction = 0.0f;
    if (correction > 1e6f) correction = 1e6f;
    if (correction < -1e6f) correction = -1e6f;

    if (simd_lane == 0) {
        float raw = stage1 + correction;
        // Phase 18.B-RM.12 attention scale: production callers never apply the
        // standard 1/sqrt(d_k) factor (host_softmax in `bench_phase8_e2e` did,
        // production didn't → softmax saturated). Apply it here so the scale
        // is consistent regardless of caller wiring.
        scores_out[kv_idx] = raw / sqrt(float(dim));
    }
}

// ============================================================================
// Kernel 7: Softmax (in-place, single vector)
// ============================================================================
// Single-threadgroup softmax for attention scores.
// Thread 0 computes max, then all threads compute exp/sum.

kernel void tq_softmax(
    device float* scores     [[buffer(0)]],
    constant uint& n         [[buffer(1)]],
    uint tid_x [[thread_position_in_grid]]
) {
    // Single thread does the whole softmax (n is typically < 8192)
    if (tid_x != 0) return;

    float max_val = -INFINITY;
    for (uint i = 0; i < n; i++) {
        max_val = max(max_val, scores[i]);
    }
    float sum = 0.0f;
    for (uint i = 0; i < n; i++) {
        scores[i] = exp(scores[i] - max_val);
        sum += scores[i];
    }
    float inv_sum = 1.0f / sum;
    for (uint i = 0; i < n; i++) {
        scores[i] *= inv_sum;
    }
}

// ============================================================================
// Kernel 7b: Parallel softmax (single threadgroup, arbitrary n_kv up to ~1M)
// ============================================================================
// Three-pass within one threadgroup:
//   1) Each thread strides over n/tg_size scores, computes partial max →
//      reduce within tg via shared memory tree.
//   2) Each thread exp(scores - max) writeback + accumulates partial sum →
//      reduce.
//   3) Each thread strides again to multiply by 1/sum.
//
// Threadgroup size MUST be a power of 2 and ≥ 1 — caller enforces.
// Fixes the silent-corruption bug of single-threadgroup fused softmax when
// n_kv > max_threads_per_threadgroup.

kernel void tq_softmax_parallel(
    device float* scores  [[buffer(0)]],
    constant uint& n      [[buffer(1)]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float shared_reduce[1024];

    // Pass 1: partial max
    float local_max = -INFINITY;
    for (uint i = tid; i < n; i += tg_size) {
        local_max = max(local_max, scores[i]);
    }
    shared_reduce[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_reduce[tid] = max(shared_reduce[tid], shared_reduce[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shared_reduce[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: exp + partial sum
    float local_sum = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        float e = exp(scores[i] - max_val);
        scores[i] = e;
        local_sum += e;
    }
    shared_reduce[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_reduce[tid] += shared_reduce[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_val = shared_reduce[0];
    float inv_sum = 1.0f / sum_val;

    // Pass 3: normalize
    for (uint i = tid; i < n; i += tg_size) {
        scores[i] *= inv_sum;
    }
}
