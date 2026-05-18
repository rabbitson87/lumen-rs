#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Flash Attention 2 — fused Q @ K^T * scale → softmax → @ V
// ============================================================================
//
// Implements the Flash Attention 2 algorithm (Dao 2023) for the Qwen3.5-MoE
// attention module (head_dim = 256, GQA already resolved by the caller).
//
// Fixed configuration (tuned for head_dim = 256):
//   TFA_D     = 256  (head dimension — one thread per element)
//   TFA_BLOCK = 8    (KV tile width — chosen so TG memory ≈ 18 KB < 32 KB limit)
//   SG_WIDTH  = 32   (Apple Silicon simdgroup width)
//   N_SG      = 8    (TFA_D / SG_WIDTH)
//
// Grid:    (B * H * Sq, 1, 1) threadgroups
// Threads: TFA_D = 256 threads per threadgroup
//          Thread d "owns" element d of the head dimension.
//
// Threadgroup memory layout (~18 KB total, < 32 KB Apple Silicon limit):
//   tg_q      [TFA_D]             1 KB   — query row loaded once
//   tg_k      [TFA_BLOCK][TFA_D]  8 KB   — current K tile
//   tg_v      [TFA_BLOCK][TFA_D]  8 KB   — current V tile
//   tg_s      [TFA_BLOCK]         32 B   — scores for current tile
//   tg_sg     [N_SG][TFA_BLOCK]  256 B   — simdgroup dot-product partials
//   tg_stats  [3]                 12 B   — [running_m, running_l, correction]
//   tg_o      [TFA_D]              1 KB   — fp32 output accumulator
//
// NOTE: BLOCK=16 + V-direct-read variant was measured 2026-05-06 — WASH
// (-0.20σ, 20/20 bit-identical). Barrier reduction (16 j's per tile vs 8)
// was offset by loss of V load/dot-product overlap. See
// `fa_block16_concluded.md`. Kernel reverted to BLOCK=8 + both-staged.
//
// Online softmax per tile (Flash Attention 2):
//   m_new = max(m_old, max_j s_j)
//   corr  = exp(m_old − m_new)
//   l    ← l * corr + Σ_j exp(s_j − m_new)
//   O    ← O * corr + Σ_j exp(s_j − m_new) * V_j
//   Final: O /= l
//
// Buffers:
//   [0] Q      [B, H,    Sq,  D]  f32 contiguous
//   [1] K      [B, H_kv, Skv, D]  f32 contiguous (H_kv = H / group)
//   [2] V      [B, H_kv, Skv, D]  f32 contiguous (H_kv = H / group)
//   [3] O      [B, H,    Sq,  D]  f32 output
//   [4] mask   [Sq, Skv]          f32 additive bias (ignored when has_mask==0)
//   [5] B_val  uint
//   [6] H_val  uint   (number of query heads)
//   [7] Sq_val uint
//   [8] Skv_val uint
//   [9] scale  float  (typically 1/sqrt(head_dim))
//  [10] has_mask uint  (0 = no mask, 1 = apply mask buffer)
//  [11] group  uint   (GQA replication factor; H = H_kv * group; group=1 = MHA)
//
// GQA in-kernel: query head h reads from kv head h_kv = h / group.
// Replaces external `repeat_kv_heads` (expand + reshape contiguous) with a
// pointer-arithmetic adjustment: same loads, ⅛× resident memory at group=8.

constant uint TFA_D     = 256;
constant uint TFA_BLOCK = 8;
constant uint SG_WIDTH  = 32;
constant uint N_SG      = TFA_D / SG_WIDTH;   // 8

kernel void tq_flash_attn(
    device const float* __restrict__ Q    [[buffer(0)]],
    device const float* __restrict__ K    [[buffer(1)]],
    device const float* __restrict__ V    [[buffer(2)]],
    device       float*              O    [[buffer(3)]],
    device const float* __restrict__ mask [[buffer(4)]],
    constant uint&   B_val    [[buffer(5)]],
    constant uint&   H_val    [[buffer(6)]],
    constant uint&   Sq_val   [[buffer(7)]],
    constant uint&   Skv_val  [[buffer(8)]],
    constant float&  scale    [[buffer(9)]],
    constant uint&   has_mask [[buffer(10)]],
    constant uint&   group    [[buffer(11)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint d    [[thread_position_in_threadgroup]]
) {
    // ── Decode (b, h, qi) from flat grid index ────────────────────────────────
    uint linear = tgid;
    uint qi = linear % Sq_val;
    uint h  = (linear / Sq_val) % H_val;
    uint b  = linear / (Sq_val * H_val);
    if (b >= B_val) return;

    // GQA: query head h shares K/V with replication group h_kv = h / group.
    uint H_kv  = H_val / group;
    uint h_kv  = h / group;

    // ── Tensor base offsets (contiguous row-major) ────────────────────────────
    uint q_off   = (b * H_val * Sq_val  + h    * Sq_val  + qi) * TFA_D;
    uint kv_base = (b * H_kv  * Skv_val + h_kv * Skv_val      ) * TFA_D;

    // ── Threadgroup memory ────────────────────────────────────────────────────
    threadgroup float tg_q   [256];      // TFA_D
    threadgroup float tg_k   [8][256];   // TFA_BLOCK × TFA_D
    threadgroup float tg_v   [8][256];
    threadgroup float tg_s   [8];        // scores for current tile
    threadgroup float tg_sg  [8][8];     // N_SG × TFA_BLOCK simdgroup partials
    threadgroup float tg_stats[3];       // [m, l, correction]
    threadgroup float tg_o   [256];      // TFA_D output accumulator

    // ── Initialise ────────────────────────────────────────────────────────────
    tg_q[d] = Q[q_off + d];
    tg_o[d] = 0.0f;
    if (d == 0) {
        tg_stats[0] = -HUGE_VALF;  // running max  m = -∞
        tg_stats[1] = 0.0f;        // running sum  l = 0
        tg_stats[2] = 1.0f;        // correction (unused until first tile)
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sg_id = d / SG_WIDTH;   // simdgroup index 0..7
    uint sg_d  = d % SG_WIDTH;   // lane within simdgroup 0..31

    // ── Tile loop ─────────────────────────────────────────────────────────────
    for (uint kv_start = 0; kv_start < Skv_val; kv_start += TFA_BLOCK) {
        uint blk = min(TFA_BLOCK, Skv_val - kv_start);

        // 1. Load K tile: each thread d loads K[kv_start+j][d] for all j.
        for (uint j = 0; j < TFA_BLOCK; j++) {
            tg_k[j][d] = (j < blk)
                ? K[kv_base + (kv_start + j) * TFA_D + d]
                : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. Dot products Q·K_j for each j in [0, TFA_BLOCK).
        //    Thread d computes q[d] * k_j[d]; simd_sum reduces across 32 lanes.
        //    Simdgroup lane 0 writes partial result to tg_sg[sg_id][j].
        for (uint j = 0; j < TFA_BLOCK; j++) {
            float contrib = tg_q[d] * tg_k[j][d];
            float sg_sum  = simd_sum(contrib);
            if (sg_d == 0) tg_sg[sg_id][j] = sg_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. Threads 0..TFA_BLOCK-1 collect the 8 simdgroup partials → score[j].
        if (d < TFA_BLOCK) {
            float s = 0.0f;
            for (uint sg = 0; sg < N_SG; sg++) s += tg_sg[sg][d];
            s *= scale;
            if (has_mask && d < blk) {
                // mask shape [Sq, Skv]: element [qi, kv_start + d]
                s += mask[qi * Skv_val + kv_start + d];
            }
            tg_s[d] = (d < blk) ? s : (-HUGE_VALF);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 4. Thread 0 only: online softmax update (serial over tile, 16 ops).
        //    Rewrites tg_s[j] → exp(s_j - m_new)  and updates tg_stats.
        if (d == 0) {
            float m_old = tg_stats[0];
            float m_new = m_old;
            for (uint j = 0; j < blk; j++) m_new = max(m_new, tg_s[j]);

            float corr    = exp(m_old - m_new);
            float l_delta = 0.0f;
            for (uint j = 0; j < blk; j++) {
                float p   = exp(tg_s[j] - m_new);
                tg_s[j]   = p;
                l_delta  += p;
            }
            tg_stats[0] = m_new;
            tg_stats[1] = tg_stats[1] * corr + l_delta;
            tg_stats[2] = corr;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 5. All threads rescale current output accumulator.
        tg_o[d] *= tg_stats[2];

        // 6. Load V tile.
        for (uint j = 0; j < TFA_BLOCK; j++) {
            tg_v[j][d] = (j < blk)
                ? V[kv_base + (kv_start + j) * TFA_D + d]
                : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 7. Accumulate: O[d] += Σ_j softmax_weight[j] * V[j][d].
        float acc = 0.0f;
        for (uint j = 0; j < blk; j++) acc += tg_s[j] * tg_v[j][d];
        tg_o[d] += acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Normalise and write output ────────────────────────────────────────────
    // tg_stats[1] = l (final running sum-of-softmax-weights).
    // Safe to read without barrier: the last loop iteration's final
    // threadgroup_barrier ensures all threads see the up-to-date value.
    O[q_off + d] = tg_o[d] / tg_stats[1];
}

// ============================================================================
// SDPA Vector — port of MLX `sdpa_vector` for batch=1 / Sq=1 decode.
// ============================================================================
//
// Different layout from `tq_flash_attn`:
//   - 1024 threads/TG (32 simdgroups × 32 lanes), each thread holds 8 q + 8 o
//     elements directly in registers (no Q/K/V threadgroup cache).
//   - 32-way KV parallelism: simdgroup `g` processes KV indices `g, g+32, g+64,
//     ...`. Each simdgroup runs its own online softmax independently; cross-sg
//     merge happens once at the end.
//   - TG memory ~4.5 KB (vs 18 KB for FA2): more concurrent TGs per Apple
//     Silicon execution unit.
//
// Trade-off vs FA2:
//   - WIN when KV is the bottleneck and sequence is wide enough that 32-way
//     parallelism saturates better than 8-way.
//   - WIN: no serial softmax bottleneck on thread 0 (each sg is independent).
//   - LOSS when batch×head TG count is already saturating the GPU (overhead of
//     cross-sg reduction not amortized).
//
// Layout matches `tq_flash_attn`:
//   Q [B, H,    Sq,  D]   K/V [B, H_kv, Skv, D]   O [B, H, Sq, D]
//   GQA: query head h reads kv head h_kv = h / group.
//
// Constraint: D=256 hardcoded (matches our Qwen3.6 head_dim).

constant uint TSDPA_D    = 256;
constant uint TSDPA_BD   = 32;                 // simdgroup width
constant uint TSDPA_BN   = 32;                 // simdgroups per TG
constant uint TSDPA_QK_PER_THREAD = TSDPA_D / TSDPA_BD;  // 8
constant uint TSDPA_V_PER_THREAD  = TSDPA_D / TSDPA_BD;  // 8

kernel void tq_sdpa_vector(
    device const float* __restrict__ Q    [[buffer(0)]],
    device const float* __restrict__ K    [[buffer(1)]],
    device const float* __restrict__ V    [[buffer(2)]],
    device       float*              O    [[buffer(3)]],
    device const float* __restrict__ mask [[buffer(4)]],
    constant uint&   B_val    [[buffer(5)]],
    constant uint&   H_val    [[buffer(6)]],
    constant uint&   Sq_val   [[buffer(7)]],
    constant uint&   Skv_val  [[buffer(8)]],
    constant float&  scale    [[buffer(9)]],
    constant uint&   has_mask [[buffer(10)]],
    constant uint&   group    [[buffer(11)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  simd_lid  [[thread_index_in_simdgroup]]
) {
    // Decode (b, h, qi) from grid: tid.x = b*H + h, tid.y = qi.
    uint linear = tid.x;
    uint qi     = tid.y;
    uint h      = linear % H_val;
    uint b      = linear / H_val;
    if (b >= B_val) return;

    uint H_kv = H_val / group;
    uint h_kv = h / group;

    uint q_offset = (b * H_val * Sq_val  + h    * Sq_val  + qi) * TSDPA_D;
    uint kv_base  = (b * H_kv  * Skv_val + h_kv * Skv_val      ) * TSDPA_D;
    uint o_offset = q_offset;

    device const float* q_ptr = Q + q_offset + simd_lid * TSDPA_QK_PER_THREAD;
    device const float* k_ptr = K + kv_base + simd_gid * TSDPA_D + simd_lid * TSDPA_QK_PER_THREAD;
    device const float* v_ptr = V + kv_base + simd_gid * TSDPA_D + simd_lid * TSDPA_V_PER_THREAD;
    device       float* o_ptr = O + o_offset + simd_gid * TSDPA_V_PER_THREAD;
    device const float* m_ptr = mask;
    if (has_mask != 0) {
        m_ptr = mask + qi * Skv_val + simd_gid;
    }

    // Pointer increment per outer step (advance by BN KV rows).
    uint inner_kv_stride   = TSDPA_BN * TSDPA_D;   // 32 * 256
    uint inner_mask_stride = TSDPA_BN;             // mask is [Sq, Skv]

    // Per-thread Q (pre-scaled), K, output accumulator.
    float q[TSDPA_QK_PER_THREAD];
    float k[TSDPA_QK_PER_THREAD];
    float o[TSDPA_V_PER_THREAD];

    for (uint i = 0; i < TSDPA_QK_PER_THREAD; ++i) {
        q[i] = scale * q_ptr[i];
    }
    for (uint i = 0; i < TSDPA_V_PER_THREAD; ++i) {
        o[i] = 0.0f;
    }

    float max_score     = -INFINITY;
    float sum_exp_score = 0.0f;

    threadgroup float outputs       [TSDPA_BN * TSDPA_BD];   // 32*32 = 1024 fp32 = 4 KB
    threadgroup float max_scores    [TSDPA_BN];              // 32 fp32
    threadgroup float sum_exp_scores[TSDPA_BN];              // 32 fp32

    // KV loop: simdgroup g processes KVs at i = g, g+32, g+64, ...
    for (uint i = simd_gid; i < Skv_val; i += TSDPA_BN) {
        // Load K row for this simdgroup's current KV.
        for (uint j = 0; j < TSDPA_QK_PER_THREAD; ++j) {
            k[j] = k_ptr[j];
        }

        // Q · K dot product, reduced across the 32 lanes of this simdgroup.
        float score = 0.0f;
        for (uint j = 0; j < TSDPA_QK_PER_THREAD; ++j) {
            score += q[j] * k[j];
        }
        score = simd_sum(score);

        if (has_mask != 0) {
            score += m_ptr[0];
        }

        // Online softmax update (per-simdgroup, per-thread state).
        float new_max = max(max_score, score);
        float factor  = exp(max_score - new_max);
        float exp_s   = exp(score - new_max);
        max_score     = new_max;
        sum_exp_score = sum_exp_score * factor + exp_s;

        // Accumulate into output: o[j] = o[j] * factor + exp_s * V[j].
        for (uint j = 0; j < TSDPA_V_PER_THREAD; ++j) {
            o[j] = o[j] * factor + exp_s * v_ptr[j];
        }

        // Advance pointers to next KV row processed by this simdgroup.
        k_ptr += inner_kv_stride;
        v_ptr += inner_kv_stride;
        if (has_mask != 0) {
            m_ptr += inner_mask_stride;
        }
    }

    // ── Cross-simdgroup combine: max + sum_exp ────────────────────────────────
    if (simd_lid == 0) {
        max_scores    [simd_gid] = max_score;
        sum_exp_scores[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each lane reads a different sg's value; simd_max reduces across sgs.
    max_score = max_scores[simd_lid];
    float new_max = simd_max(max_score);
    float factor  = exp(max_score - new_max);
    sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

    // ── Cross-simdgroup combine: per-element output (transpose pattern) ───────
    // For each output element i (per-thread):
    //   1) Thread (g, l) writes its o[i] to outputs[l*BD + g].
    //   2) After barrier, thread (g, l) reads outputs[g*BD + l] = o[i] from
    //      thread (l, g) — i.e., the contribution of sg `l` to column g*8+i.
    //   3) simd_sum across the 32 lanes of sg g sums contributions from all
    //      32 sgs, scaled by their respective factors.
    for (uint i = 0; i < TSDPA_V_PER_THREAD; ++i) {
        outputs[simd_lid * TSDPA_BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_gid * TSDPA_BD + simd_lid] * factor);
        o[i] = (sum_exp_score == 0.0f) ? o[i] : (o[i] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Lane 0 of each simdgroup writes its 8 output columns.
    if (simd_lid == 0) {
        for (uint i = 0; i < TSDPA_V_PER_THREAD; ++i) {
            o_ptr[i] = o[i];
        }
    }
}

// ============================================================================
// Flash Attention 2 — bf16 I/O variant
// ============================================================================
//
// Same algorithm as `tq_flash_attn` (f32 version above) — only the I/O dtype
// changes. Q/K/V/O and mask buffers are bf16; all internal accumulation,
// softmax statistics, and threadgroup memory remain f32 (correctness:
// bf16 has 7-bit mantissa, accumulating 256+ products in bf16 would lose
// precision catastrophically).
//
// MLX reference (qwen3_next.py:97-141, mlx-lm 0.31.3): MLX's
// `Qwen3NextAttention.__call__` runs `scaled_dot_product_attention` with
// activation-dtype inputs (bf16 for the bf16 checkpoint). MLX's MPSGraph SDPA
// internally widens to f32 for the same numerical reasons. This kernel mirrors
// that policy: bf16 I/O, f32 compute.
//
// Buffer layout (matches f32 version, only dtype differs):
//   [0] Q      [B, H,    Sq,  D]  bf16 contiguous
//   [1] K      [B, H_kv, Skv, D]  bf16 contiguous
//   [2] V      [B, H_kv, Skv, D]  bf16 contiguous
//   [3] O      [B, H,    Sq,  D]  bf16 output
//   [4] mask   [Sq, Skv]          bf16 additive bias (ignored when has_mask==0)
//   [5..11]    same as f32 kernel

kernel void tq_flash_attn_bf16(
    device const bfloat* __restrict__ Q    [[buffer(0)]],
    device const bfloat* __restrict__ K    [[buffer(1)]],
    device const bfloat* __restrict__ V    [[buffer(2)]],
    device       bfloat*              O    [[buffer(3)]],
    device const bfloat* __restrict__ mask [[buffer(4)]],
    constant uint&   B_val    [[buffer(5)]],
    constant uint&   H_val    [[buffer(6)]],
    constant uint&   Sq_val   [[buffer(7)]],
    constant uint&   Skv_val  [[buffer(8)]],
    constant float&  scale    [[buffer(9)]],
    constant uint&   has_mask [[buffer(10)]],
    constant uint&   group    [[buffer(11)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint d    [[thread_position_in_threadgroup]]
) {
    // ── Decode (b, h, qi) from flat grid index ────────────────────────────────
    uint linear = tgid;
    uint qi = linear % Sq_val;
    uint h  = (linear / Sq_val) % H_val;
    uint b  = linear / (Sq_val * H_val);
    if (b >= B_val) return;

    uint H_kv  = H_val / group;
    uint h_kv  = h / group;

    uint q_off   = (b * H_val * Sq_val  + h    * Sq_val  + qi) * TFA_D;
    uint kv_base = (b * H_kv  * Skv_val + h_kv * Skv_val      ) * TFA_D;

    // ── Threadgroup memory (f32 throughout for accumulation correctness) ─────
    threadgroup float tg_q   [256];
    threadgroup float tg_k   [8][256];
    threadgroup float tg_v   [8][256];
    threadgroup float tg_s   [8];
    threadgroup float tg_sg  [8][8];
    threadgroup float tg_stats[3];
    threadgroup float tg_o   [256];

    // ── Initialise ────────────────────────────────────────────────────────────
    tg_q[d] = float(Q[q_off + d]);   // bf16 → f32 widen on load
    tg_o[d] = 0.0f;
    if (d == 0) {
        tg_stats[0] = -HUGE_VALF;
        tg_stats[1] = 0.0f;
        tg_stats[2] = 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sg_id = d / SG_WIDTH;
    uint sg_d  = d % SG_WIDTH;

    // ── Tile loop ─────────────────────────────────────────────────────────────
    for (uint kv_start = 0; kv_start < Skv_val; kv_start += TFA_BLOCK) {
        uint blk = min(TFA_BLOCK, Skv_val - kv_start);

        // 1. Load K tile (bf16 → f32 widen on load).
        for (uint j = 0; j < TFA_BLOCK; j++) {
            tg_k[j][d] = (j < blk)
                ? float(K[kv_base + (kv_start + j) * TFA_D + d])
                : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. Dot products Q·K_j.
        for (uint j = 0; j < TFA_BLOCK; j++) {
            float contrib = tg_q[d] * tg_k[j][d];
            float sg_sum  = simd_sum(contrib);
            if (sg_d == 0) tg_sg[sg_id][j] = sg_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. Collect simdgroup partials → score[j].
        if (d < TFA_BLOCK) {
            float s = 0.0f;
            for (uint sg = 0; sg < N_SG; sg++) s += tg_sg[sg][d];
            s *= scale;
            if (has_mask && d < blk) {
                // mask is bf16 — widen on load.
                s += float(mask[qi * Skv_val + kv_start + d]);
            }
            tg_s[d] = (d < blk) ? s : (-HUGE_VALF);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 4. Online softmax update (serial on thread 0).
        if (d == 0) {
            float m_old = tg_stats[0];
            float m_new = m_old;
            for (uint j = 0; j < blk; j++) m_new = max(m_new, tg_s[j]);

            float corr    = exp(m_old - m_new);
            float l_delta = 0.0f;
            for (uint j = 0; j < blk; j++) {
                float p   = exp(tg_s[j] - m_new);
                tg_s[j]   = p;
                l_delta  += p;
            }
            tg_stats[0] = m_new;
            tg_stats[1] = tg_stats[1] * corr + l_delta;
            tg_stats[2] = corr;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 5. Rescale current output accumulator.
        tg_o[d] *= tg_stats[2];

        // 6. Load V tile (bf16 → f32 widen on load).
        for (uint j = 0; j < TFA_BLOCK; j++) {
            tg_v[j][d] = (j < blk)
                ? float(V[kv_base + (kv_start + j) * TFA_D + d])
                : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 7. Accumulate output.
        float acc = 0.0f;
        for (uint j = 0; j < blk; j++) acc += tg_s[j] * tg_v[j][d];
        tg_o[d] += acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Normalise and write output (f32 → bf16 cast on store) ─────────────────
    O[q_off + d] = bfloat(tg_o[d] / tg_stats[1]);
}
