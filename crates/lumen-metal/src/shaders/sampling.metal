#include <metal_stdlib>
using namespace metal;

// Single-threadgroup cooperative argmax over a 1-D F32 vector.
//
// Each thread strides over `n` elements collecting a private (best_v, best_i),
// then a tree reduction in threadgroup memory finds the global winner. One
// thread writes the resulting index to `out_idx[0]`.
//
// Buffers:
//   0: x         — [n] f32
//   1: out_idx   — [1] u32 (writes argmax index)
//   2: n         — uint (length)
// Threadgroup memory:
//   0: shared_v  — tg_size f32
//   1: shared_i  — tg_size u32
kernel void argmax_f32(
    device const float* x        [[buffer(0)]],
    device uint*        out_idx  [[buffer(1)]],
    constant uint&      n        [[buffer(2)]],
    threadgroup float*  shared_v [[threadgroup(0)]],
    threadgroup uint*   shared_i [[threadgroup(1)]],
    uint                tid      [[thread_index_in_threadgroup]],
    uint                tg_size  [[threads_per_threadgroup]]
) {
    float best_v = -INFINITY;
    uint  best_i = 0u;
    for (uint i = tid; i < n; i += tg_size) {
        float v = x[i];
        if (v > best_v) {
            best_v = v;
            best_i = i;
        }
    }
    shared_v[tid] = best_v;
    shared_i[tid] = best_i;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            float other_v = shared_v[tid + stride];
            uint  other_i = shared_i[tid + stride];
            float my_v    = shared_v[tid];
            // Stable: prefer earlier index on tie (matches CPU std::max behavior used in
            // sample_token_cpu_full for greedy=temperature=0 path).
            if (other_v > my_v || (other_v == my_v && other_i < shared_i[tid])) {
                shared_v[tid] = other_v;
                shared_i[tid] = other_i;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        out_idx[0] = shared_i[0];
    }
}

// Apply per-token multiplicative penalties to a logits buffer in place.
//
// Each input pair `(token_idx[t], multiplier[t])` updates `logits[token_idx[t]]`
// using the same sign-aware rule the CPU sampler used:
//   logit > 0  → logit /= mul   (penalize: shrink positive logit)
//   logit ≤ 0  → logit *= mul   (penalize: stretch negative logit further negative)
//
// Caller pre-aggregates penalties so each `token_idx` appears AT MOST ONCE in
// the input list (combined `rep_penalty^count * ngram_penalty^matches`). With
// uniqueness, parallel writes never race.
//
// Buffers:
//   0: logits       — [vocab] f32, modified in place
//   1: token_idx    — [n_pairs] u32, target token positions
//   2: multipliers  — [n_pairs] f32, per-pair multiplier (>= 1.0 expected)
//   3: n_pairs      — uint
kernel void apply_token_penalties_f32(
    device float*        logits      [[buffer(0)]],
    device const uint*   token_idx   [[buffer(1)]],
    device const float*  multipliers [[buffer(2)]],
    constant uint&       n_pairs     [[buffer(3)]],
    uint                 tid         [[thread_position_in_grid]]
) {
    if (tid >= n_pairs) { return; }
    uint  idx = token_idx[tid];
    float mul = multipliers[tid];
    if (mul == 1.0f) { return; }
    float v = logits[idx];
    if (v > 0.0f) {
        logits[idx] = v / mul;
    } else {
        logits[idx] = v * mul;
    }
}

// Phase C #4 (2026-05-02) — Fused single-threadgroup top-k + top-p + Gumbel-max
// sampler. Replaces the CPU softmax/sort/cumsum path for the (penalty-applied)
// logits buffer when the caller wants nucleus / top-k filtering on the GPU.
//
// Pipeline (one threadgroup, K_MAX cap = 32, recommended tg_size = 64):
//   1. Phase 1 — each thread strides through `n_logits` and maintains a local
//      top-K sorted-descending list. Insertion via linear scan + upward shift
//      (K is small, branch-free is unnecessary).
//   2. Phase 2 — tree reduction in threadgroup memory: 2-way descending merge
//      of two top-K lists at each level, halving active threads per round.
//   3. Phase 3 (thread 0 only) — softmax over the top-K (with stability
//      subtract = tg_vals[0], the global max), cumulative-sum top-p mask,
//      Gumbel-max sample over the kept prefix, write 1 u32 token.
//
// Notes:
//   - Input `x` is already penalty-applied; this kernel does NOT touch logits
//     outside reading.
//   - `inv_temp = 1/T`; caller validates `T > 0` so `inv_temp` is finite.
//   - `top_k` clamped to [1, K_MAX]; `top_p` in (0, 1].
//   - Tie behavior in argmax: prefer earlier sorted-position (i.e. higher
//     prob token) on equal perturbed scores — matches the local-list
//     descending insertion's tie rule (later equal value does not displace).
//   - seed_hi reserved for future 64-bit RNG state expansion.
//
// Buffers:
//   0: x          — [n_logits] f32 (read-only)
//   1: out_idx    — [1] u32 (sampled token index, written by tid 0)
//   2: n_logits   — uint
//   3: top_k      — uint (1..=K_MAX)
//   4: top_p      — float (0..=1.0)
//   5: inv_temp   — float (1/T)
//   6: seed_lo    — uint (RNG seed low half)
//   7: seed_hi    — uint (reserved)
// Threadgroup memory:
//   0: tg_vals    — tg_size * K_MAX f32
//   1: tg_idxs    — tg_size * K_MAX u32
constant uint K_MAX = 32;

inline uint mix32(uint a, uint b) {
    uint h = a ^ (b * 0x9e3779b9u);
    h ^= h >> 16; h *= 0x85ebca6bu;
    h ^= h >> 13; h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

inline float u01_from_u32(uint x) {
    // Map u32 → (0, 1] with a small floor so log(u) stays finite.
    return max(1e-7f, float(x) * (1.0f / 4294967296.0f));
}

kernel void topk_topp_gumbel_argmax_f32(
    device const float* x          [[buffer(0)]],
    device uint*        out_idx    [[buffer(1)]],
    constant uint&      n_logits   [[buffer(2)]],
    constant uint&      top_k      [[buffer(3)]],
    constant float&     top_p      [[buffer(4)]],
    constant float&     inv_temp   [[buffer(5)]],
    constant uint&      seed_lo    [[buffer(6)]],
    constant uint&      seed_hi    [[buffer(7)]],
    threadgroup float*  tg_vals    [[threadgroup(0)]],
    threadgroup uint*   tg_idxs    [[threadgroup(1)]],
    uint                tid        [[thread_index_in_threadgroup]],
    uint                tg_size    [[threads_per_threadgroup]]
) {
    // Effective K: clamp [1, K_MAX]. Caller is expected to pass this in
    // range; clamp here as defense-in-depth (avoids runtime UB if a 0 / >32
    // sneaks through validation).
    uint K = top_k;
    if (K == 0u) { K = 1u; }
    if (K > K_MAX) { K = K_MAX; }

    // ── Phase 1 — per-thread local top-K (sorted descending) ────────────
    float local_v[K_MAX];
    uint  local_i[K_MAX];
    for (uint k = 0u; k < K_MAX; ++k) {
        local_v[k] = -INFINITY;
        local_i[k] = 0u;
    }

    for (uint i = tid; i < n_logits; i += tg_size) {
        float v = x[i];
        // Only consider if it can beat the smallest kept (slot K-1).
        if (v > local_v[K - 1u]) {
            // Find insertion position by scanning from the top.
            // pos = first index where local_v[pos] < v (strict <, so equal
            // earlier values keep their slot — earlier-index preference on tie).
            uint pos = K - 1u;
            for (uint p = 0u; p < K; ++p) {
                if (v > local_v[p]) { pos = p; break; }
            }
            // Shift right [pos .. K-2] → [pos+1 .. K-1] (drop the smallest).
            for (uint p = K - 1u; p > pos; --p) {
                local_v[p] = local_v[p - 1u];
                local_i[p] = local_i[p - 1u];
            }
            local_v[pos] = v;
            local_i[pos] = i;
        }
    }

    // Spill local list into threadgroup memory at slot tid * K_MAX.
    uint base = tid * K_MAX;
    for (uint k = 0u; k < K_MAX; ++k) {
        tg_vals[base + k] = local_v[k];
        tg_idxs[base + k] = local_i[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2 — tree reduction: 2-way descending merge of K-lists ─────
    // At each round, threads in [0, stride) merge their list with [stride, 2*stride).
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            uint a_off = tid * K_MAX;
            uint b_off = (tid + stride) * K_MAX;
            // Two-way merge into a temporary, then write back to a_off.
            float merged_v[K_MAX];
            uint  merged_i[K_MAX];
            uint ia = 0u, ib = 0u;
            for (uint k = 0u; k < K; ++k) {
                float va = tg_vals[a_off + ia];
                float vb = tg_vals[b_off + ib];
                // Prefer A on equal value (stable; earlier slot already won).
                if (va >= vb) {
                    merged_v[k] = va;
                    merged_i[k] = tg_idxs[a_off + ia];
                    ia += 1u;
                } else {
                    merged_v[k] = vb;
                    merged_i[k] = tg_idxs[b_off + ib];
                    ib += 1u;
                }
            }
            for (uint k = 0u; k < K; ++k) {
                tg_vals[a_off + k] = merged_v[k];
                tg_idxs[a_off + k] = merged_i[k];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Phase 3 — thread 0 does softmax + top-p + Gumbel-max ─────────────
    if (tid == 0u) {
        // Scale and stability-subtract using vmax = scaled[0] (descending order).
        float scaled[K_MAX];
        for (uint k = 0u; k < K; ++k) {
            scaled[k] = tg_vals[k] * inv_temp;
        }
        float vmax = scaled[0];

        float exps[K_MAX];
        float sum_exp = 0.0f;
        for (uint k = 0u; k < K; ++k) {
            float e = exp(scaled[k] - vmax);
            exps[k] = e;
            sum_exp += e;
        }
        // sum_exp >= 1.0 by construction (exps[0] == 1.0); safe to divide.
        float inv_sum = 1.0f / sum_exp;

        // Top-p mask: smallest prefix whose cumulative prob ≥ top_p.
        // Always keep at least 1.
        uint kept = K;
        float cumsum = 0.0f;
        bool truncated = false;
        for (uint k = 0u; k < K; ++k) {
            float p = exps[k] * inv_sum;
            cumsum += p;
            if (!truncated && cumsum >= top_p) {
                kept = k + 1u;
                truncated = true;
            }
        }
        if (kept < 1u) { kept = 1u; }

        // Gumbel-max sample over [0, kept). gumbel = -log(-log(u01)).
        float best_score = -INFINITY;
        uint best_k = 0u;
        for (uint k = 0u; k < kept; ++k) {
            uint r = mix32(seed_lo, k);
            // seed_hi reserved — currently unused; mix in cheaply to suppress
            // unused-buffer warnings without changing behavior when seed_hi=0.
            r = mix32(r, seed_hi);
            float u = u01_from_u32(r);
            float g = -log(-log(u));
            float perturbed = scaled[k] + g;
            // Tie: prefer earlier k (higher-prob token).
            if (perturbed > best_score) {
                best_score = perturbed;
                best_k = k;
            }
        }

        out_idx[0] = tg_idxs[best_k];
    }
}
