#include <metal_stdlib>
using namespace metal;

// ============================================================================
// PagedAttention Metal Kernels
// ============================================================================
//
// Three kernels for vLLM-style paged KV cache attention on Metal GPU.
//
// Per-layer buffer layout (each block):
//   K: [block_size, n_kv_heads, head_dim] as f16
//   V: [block_size, n_kv_heads, head_dim] as f16
//   (V starts at offset = block_size * n_kv_heads * head_dim * sizeof(half))
//
// Block table: block_table[logical_block_idx] = physical_block_id
// Physical address: buffer_base + physical_block_id * block_bytes + kv_offset + token_offset

// ============================================================================
// Kernel 1: paged_attention_decode
// ============================================================================
//
// Single-query (decode) attention over paged KV cache.
// Each threadgroup handles one (batch_idx, head_idx) pair.
//
// Algorithm:
//   1. Load q vector for this head
//   2. Iterate over all KV blocks, compute q·k scores
//   3. Online softmax (numerically stable)
//   4. Accumulate weighted v sum
//
// GQA: Multiple Q heads share the same KV head.
//   kv_head = q_head / gqa_ratio
//
// Supports heterogeneous head_dim per layer (passed as parameter).

kernel void paged_attention_decode(
    device const float* q           [[buffer(0)]],   // [batch, n_q_heads, head_dim]
    device const half*  kv_pool     [[buffer(1)]],   // layer's KV pool buffer
    device const int*   block_table [[buffer(2)]],   // [batch, max_num_blocks]
    device const int*   context_lens[[buffer(3)]],   // [batch]
    device float*       output      [[buffer(4)]],   // [batch, n_q_heads, head_dim]
    constant uint&      head_dim    [[buffer(5)]],
    constant uint&      block_size  [[buffer(6)]],
    constant uint&      n_q_heads   [[buffer(7)]],
    constant uint&      n_kv_heads  [[buffer(8)]],
    constant float&     scale       [[buffer(9)]],
    constant uint&      max_num_blocks [[buffer(10)]],
    uint3 group_id  [[threadgroup_position_in_grid]],   // (batch, q_head, 0)
    uint3 tid3      [[thread_position_in_threadgroup]],
    uint3 tg_size3  [[threads_per_threadgroup]]
) {
    const uint tid     = tid3.x;
    const uint tg_size = tg_size3.x;
    const uint batch_idx = group_id.x;
    const uint q_head    = group_id.y;
    const uint ctx_len   = (uint)context_lens[batch_idx];

    if (ctx_len == 0) return;

    // GQA mapping: which KV head does this Q head use?
    const uint gqa_ratio = n_q_heads / n_kv_heads;
    const uint kv_head   = q_head / gqa_ratio;

    // Block layout constants
    const uint kv_stride = n_kv_heads * head_dim;  // one token's KV width
    const uint block_kv_size = block_size * kv_stride;  // K (or V) section per block
    // V offset within a block = block_kv_size (K comes first, then V)

    // Pointer to this batch's block table
    device const int* my_block_table = block_table + batch_idx * max_num_blocks;

    // Load Q vector into registers (each thread loads a subset)
    device const float* q_ptr = q + (batch_idx * n_q_heads + q_head) * head_dim;

    // Phase 1: Compute attention scores with online softmax
    // Each thread processes a subset of tokens
    float max_score = -INFINITY;
    float sum_exp = 0.0f;

    // Temporary storage for weighted V accumulation
    // We accumulate in float for precision
    // Each thread maintains a partial V sum
    threadgroup float shared_max[1];
    threadgroup float shared_sum[1];

    // First pass: find max score across all tokens
    float thread_max = -INFINITY;
    for (uint token_idx = tid; token_idx < ctx_len; token_idx += tg_size) {
        uint block_idx = token_idx / block_size;
        uint block_offset = token_idx % block_size;
        int phys_block = my_block_table[block_idx];

        // K address: kv_pool + phys_block * (2 * block_kv_size) + block_offset * kv_stride + kv_head * head_dim
        device const half* k_ptr = kv_pool
            + (uint)phys_block * (2 * block_kv_size)  // block start
            + block_offset * kv_stride                 // token within block
            + kv_head * head_dim;                      // head offset

        // Dot product: q · k
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += q_ptr[d] * float(k_ptr[d]);
        }
        dot *= scale;
        thread_max = max(thread_max, dot);
    }

    // Reduce max across threadgroup
    threadgroup float shared_maxes[256];
    shared_maxes[tid] = thread_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float global_max = -INFINITY;
        for (uint i = 0; i < tg_size; i++) {
            global_max = max(global_max, shared_maxes[i]);
        }
        shared_max[0] = global_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_score = shared_max[0];

    // Second pass: compute exp(score - max) and sum
    float thread_sum = 0.0f;
    for (uint token_idx = tid; token_idx < ctx_len; token_idx += tg_size) {
        uint block_idx = token_idx / block_size;
        uint block_offset = token_idx % block_size;
        int phys_block = my_block_table[block_idx];

        device const half* k_ptr = kv_pool
            + (uint)phys_block * (2 * block_kv_size)
            + block_offset * kv_stride
            + kv_head * head_dim;

        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += q_ptr[d] * float(k_ptr[d]);
        }
        dot *= scale;
        thread_sum += exp(dot - max_score);
    }

    // Reduce sum across threadgroup
    threadgroup float shared_sums[256];
    shared_sums[tid] = thread_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < tg_size; i++) {
            total += shared_sums[i];
        }
        shared_sum[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum_exp = shared_sum[0];

    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;

    // Third pass: compute weighted V sum
    // Each thread accumulates partial output for all dims
    device float* out_ptr = output + (batch_idx * n_q_heads + q_head) * head_dim;

    // Each thread computes output for a subset of head_dim dimensions
    for (uint d = tid; d < head_dim; d += tg_size) {
        float acc = 0.0f;
        for (uint token_idx = 0; token_idx < ctx_len; token_idx++) {
            uint block_idx = token_idx / block_size;
            uint block_offset = token_idx % block_size;
            int phys_block = my_block_table[block_idx];

            // K address for score recomputation
            device const half* k_ptr = kv_pool
                + (uint)phys_block * (2 * block_kv_size)
                + block_offset * kv_stride
                + kv_head * head_dim;

            // Recompute score (q·k)
            float dot = 0.0f;
            for (uint dd = 0; dd < head_dim; dd++) {
                dot += q_ptr[dd] * float(k_ptr[dd]);
            }
            float w = exp(dot * scale - max_score) * inv_sum;

            // V address: after K section
            device const half* v_ptr = kv_pool
                + (uint)phys_block * (2 * block_kv_size)
                + block_kv_size                        // skip K section
                + block_offset * kv_stride
                + kv_head * head_dim;

            acc += w * float(v_ptr[d]);
        }
        out_ptr[d] = acc;
    }
}


// ============================================================================
// Kernel 1b: paged_attention_decode_v2 (FlashAttention-style, 1-pass)
// ============================================================================
//
// Online softmax avoids re-reading K: a single pass over KV blocks
// maintains running (m, l, o) state and rescales on each block.
//
// For each block:
//   1. Compute scores = q · k_block (parallel over tokens)
//   2. Find block_max = max(scores)
//   3. new_m = max(m_i, block_max), alpha = exp(m_i - new_m)
//   4. probs = exp(scores - new_m), block_sum = sum(probs)
//   5. l_i ← l_i * alpha + block_sum
//   6. o ← o * alpha + sum(probs * v_block)
// Final: output = o / l_i
//
// Threadgroup memory layout (head_dim up to 512):
//   q_shared:  head_dim * 4 bytes
//   o_shared:  head_dim * 4 bytes
//   scores:    block_size * 4 bytes
//   probs:     block_size * 4 bytes
// Total ≈ 4 KB for head_dim=512, block_size=16 (well under 32KB limit)

kernel void paged_attention_decode_v2(
    device const float* q           [[buffer(0)]],
    device const half*  kv_pool     [[buffer(1)]],
    device const int*   block_table [[buffer(2)]],
    device const int*   context_lens[[buffer(3)]],
    device float*       output      [[buffer(4)]],
    constant uint&      head_dim    [[buffer(5)]],
    constant uint&      block_size  [[buffer(6)]],
    constant uint&      n_q_heads   [[buffer(7)]],
    constant uint&      n_kv_heads  [[buffer(8)]],
    constant float&     scale       [[buffer(9)]],
    constant uint&      max_num_blocks [[buffer(10)]],
    uint3 group_id  [[threadgroup_position_in_grid]],
    uint3 tid3      [[thread_position_in_threadgroup]],
    uint3 tg_size3  [[threads_per_threadgroup]]
) {
    const uint tid = tid3.x;
    const uint tg_size = tg_size3.x;
    const uint batch_idx = group_id.x;
    const uint q_head    = group_id.y;
    const uint ctx_len   = (uint)context_lens[batch_idx];
    if (ctx_len == 0) return;

    const uint gqa_ratio = n_q_heads / n_kv_heads;
    const uint kv_head   = q_head / gqa_ratio;

    const uint kv_stride = n_kv_heads * head_dim;
    const uint block_kv_size = block_size * kv_stride;

    device const int* my_block_table = block_table + batch_idx * max_num_blocks;
    device const float* q_ptr = q + (batch_idx * n_q_heads + q_head) * head_dim;
    device float* out_ptr = output + (batch_idx * n_q_heads + q_head) * head_dim;

    // Threadgroup memory (sized for max head_dim=512)
    threadgroup float q_shared[512];
    threadgroup float o_shared[512];
    threadgroup float scores[64];      // block_size up to 64
    threadgroup float probs[64];
    threadgroup float scratch[4];       // block_max, new_m, alpha, block_sum

    // Load Q into shared memory once
    for (uint d = tid; d < head_dim; d += tg_size) {
        q_shared[d] = q_ptr[d];
        o_shared[d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Running softmax state (each thread keeps identical copy)
    float m_i = -INFINITY;
    float l_i = 0.0f;

    const uint num_blocks = (ctx_len + block_size - 1) / block_size;

    for (uint block_idx = 0; block_idx < num_blocks; block_idx++) {
        const uint block_start = block_idx * block_size;
        const uint block_end = min(block_start + block_size, ctx_len);
        const uint block_tokens = block_end - block_start;
        const int phys_block = my_block_table[block_idx];

        device const half* k_block = kv_pool
            + (uint)phys_block * (2 * block_kv_size)
            + kv_head * head_dim;
        device const half* v_block = k_block + block_kv_size;

        // Step 1: compute scores for this block (parallel over tokens)
        for (uint t = tid; t < block_tokens; t += tg_size) {
            device const half* k = k_block + t * kv_stride;
            float dot = 0.0f;
            for (uint d = 0; d < head_dim; d++) {
                dot += q_shared[d] * float(k[d]);
            }
            scores[t] = dot * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 2: block_max, new_m, alpha, block_sum (thread 0 reduces)
        if (tid == 0) {
            float mx = -INFINITY;
            for (uint t = 0; t < block_tokens; t++) {
                mx = max(mx, scores[t]);
            }
            scratch[0] = mx;                      // block_max
            scratch[1] = max(m_i, mx);            // new_m
            scratch[2] = exp(m_i - scratch[1]);   // alpha (0 if m_i was -INF → exp(-INF)=0)
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float new_m = scratch[1];
        const float alpha = scratch[2];

        // Step 3: compute probs = exp(scores - new_m) (parallel)
        for (uint t = tid; t < block_tokens; t += tg_size) {
            probs[t] = exp(scores[t] - new_m);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 4: block_sum
        if (tid == 0) {
            float s = 0.0f;
            for (uint t = 0; t < block_tokens; t++) {
                s += probs[t];
            }
            scratch[3] = s;  // block_sum
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float block_sum = scratch[3];

        // Step 5: update o_shared (rescale + weighted V sum, parallel over head_dim)
        for (uint d = tid; d < head_dim; d += tg_size) {
            float acc = o_shared[d] * alpha;
            for (uint t = 0; t < block_tokens; t++) {
                device const half* v = v_block + t * kv_stride;
                acc += probs[t] * float(v[d]);
            }
            o_shared[d] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Update running state (all threads compute same value from shared scratch)
        m_i = new_m;
        l_i = l_i * alpha + block_sum;
    }

    // Final normalization
    const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
    for (uint d = tid; d < head_dim; d += tg_size) {
        out_ptr[d] = o_shared[d] * inv_l;
    }
}


// ============================================================================
// Kernel 2: write_kv_to_blocks
// ============================================================================
//
// Scatter K and V tensors from model projection into paged block locations.
// Called after q/k/v projection during both prefill and decode.
//
// Each thread writes one (token, kv_head, dim_element) to the correct block.

kernel void write_kv_to_blocks(
    device const half*  k_src       [[buffer(0)]],   // [seq_len, n_kv_heads, head_dim] f16
    device const half*  v_src       [[buffer(1)]],   // [seq_len, n_kv_heads, head_dim] f16
    device half*        kv_pool     [[buffer(2)]],   // layer's KV pool buffer
    device const int*   block_table [[buffer(3)]],   // [max_num_blocks]
    constant uint&      seq_len     [[buffer(4)]],
    constant uint&      start_pos   [[buffer(5)]],   // context_len before this write
    constant uint&      block_size  [[buffer(6)]],
    constant uint&      n_kv_heads  [[buffer(7)]],
    constant uint&      head_dim    [[buffer(8)]],
    uint3 tid [[thread_position_in_grid]]   // (dim_idx, token_idx, kv_head_idx)
) {
    const uint d_idx  = tid.x;
    const uint t_idx  = tid.y;
    const uint h_idx  = tid.z;

    if (d_idx >= head_dim || t_idx >= seq_len || h_idx >= n_kv_heads) return;

    // Source offset: [t_idx, h_idx, d_idx]
    const uint src_offset = (t_idx * n_kv_heads + h_idx) * head_dim + d_idx;

    // Destination: find which block and position within block
    const uint global_pos = start_pos + t_idx;
    const uint block_idx  = global_pos / block_size;
    const uint block_off  = global_pos % block_size;
    const int  phys_block = block_table[block_idx];

    // Layout: [phys_block][K_section | V_section]
    // K section: [block_size, n_kv_heads, head_dim]
    // V section: [block_size, n_kv_heads, head_dim]
    const uint kv_stride = n_kv_heads * head_dim;
    const uint block_kv_size = block_size * kv_stride;
    const uint block_start = (uint)phys_block * (2 * block_kv_size);

    // K destination
    const uint k_dst = block_start + block_off * kv_stride + h_idx * head_dim + d_idx;
    kv_pool[k_dst] = k_src[src_offset];

    // V destination (after K section)
    const uint v_dst = block_start + block_kv_size + block_off * kv_stride + h_idx * head_dim + d_idx;
    kv_pool[v_dst] = v_src[src_offset];
}


// ============================================================================
// Kernel 2b: write_kv_to_blocks_f32_candle
// ============================================================================
//
// Zero-copy variant that accepts F32 tensors in Candle's native layout
// `[1, n_kv_heads, seq_len, head_dim]` (contiguous). Converts to F16 and
// writes to paged blocks in our pool layout `[blocks][K|V][seq][kv][dim]`.
//
// Source indexing: src[t, h, d] = k_src[(h * seq_len + t) * head_dim + d]

kernel void write_kv_to_blocks_f32_candle(
    device const float* k_src       [[buffer(0)]],
    device const float* v_src       [[buffer(1)]],
    device half*        kv_pool     [[buffer(2)]],
    device const int*   block_table [[buffer(3)]],
    constant uint&      seq_len     [[buffer(4)]],
    constant uint&      start_pos   [[buffer(5)]],
    constant uint&      block_size  [[buffer(6)]],
    constant uint&      n_kv_heads  [[buffer(7)]],
    constant uint&      head_dim    [[buffer(8)]],
    uint3 tid [[thread_position_in_grid]]
) {
    const uint d_idx = tid.x;
    const uint t_idx = tid.y;
    const uint h_idx = tid.z;
    if (d_idx >= head_dim || t_idx >= seq_len || h_idx >= n_kv_heads) return;

    // Candle layout: [1, n_kv, seq, dim] contiguous
    const uint src_offset = (h_idx * seq_len + t_idx) * head_dim + d_idx;

    const uint global_pos = start_pos + t_idx;
    const uint block_idx  = global_pos / block_size;
    const uint block_off  = global_pos % block_size;
    const int  phys_block = block_table[block_idx];

    const uint kv_stride = n_kv_heads * head_dim;
    const uint block_kv_size = block_size * kv_stride;
    const uint block_start = (uint)phys_block * (2 * block_kv_size);

    const uint k_dst = block_start + block_off * kv_stride + h_idx * head_dim + d_idx;
    kv_pool[k_dst] = (half)k_src[src_offset];

    const uint v_dst = block_start + block_kv_size + block_off * kv_stride + h_idx * head_dim + d_idx;
    kv_pool[v_dst] = (half)v_src[src_offset];
}


// ============================================================================
// Kernel 3: copy_blocks
// ============================================================================
//
// Copy physical blocks for beam search forking or COW (copy-on-write).
// Each thread copies one element from source block to destination block.

kernel void copy_blocks(
    device half*        kv_pool       [[buffer(0)]],
    device const int2*  block_mapping [[buffer(1)]],   // [num_pairs] (src_block, dst_block)
    constant uint&      num_pairs     [[buffer(2)]],
    constant uint&      block_elems   [[buffer(3)]],   // total elements per block (2 * block_size * n_kv_heads * head_dim)
    uint2 tid [[thread_position_in_grid]]   // (elem_idx, pair_idx)
) {
    const uint elem_idx = tid.x;
    const uint pair_idx = tid.y;

    if (elem_idx >= block_elems || pair_idx >= num_pairs) return;

    const int src_block = block_mapping[pair_idx].x;
    const int dst_block = block_mapping[pair_idx].y;

    const uint src_offset = (uint)src_block * block_elems + elem_idx;
    const uint dst_offset = (uint)dst_block * block_elems + elem_idx;

    kv_pool[dst_offset] = kv_pool[src_offset];
}
