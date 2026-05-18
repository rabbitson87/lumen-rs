// Native forward kernels for Qwen3.5-VL-MoE (Phase A.2+).
//
// Each kernel takes flat Metal buffers + inline dim constants — no Candle in
// the call path. Dispatched directly from `NativeContext::queue` so the entire
// forward stays on one queue and avoids cross-queue waits.

#include <metal_stdlib>
using namespace metal;

// ─── Embedding lookup ────────────────────────────────────────────────────────
//
// out[t, h] = embed[token_ids[t], h].
// Grid: (hidden, seq_len). One thread per (token, hidden_dim) cell.
kernel void embedding_lookup_f32(
    device const uint*  token_ids [[buffer(0)]],
    device const float* embed     [[buffer(1)]],
    device float*       out       [[buffer(2)]],
    constant uint&      hidden    [[buffer(3)]],
    uint2 gid                     [[thread_position_in_grid]]
) {
    uint h = gid.x;
    uint t = gid.y;
    if (h >= hidden) { return; }
    uint id = token_ids[t];
    out[t * hidden + h] = embed[id * hidden + h];
}

// ─── RMSNorm ────────────────────────────────────────────────────────────────
//
// y[r, h] = (x[r, h] * rsqrt(mean(x[r, :]^2) + eps)) * gamma[h]
//
// One threadgroup per row. Threads cooperate to compute sum-of-squares via a
// shared-memory tree reduction, then write the normalized output.
//
// Threadgroup size must be a power of 2 ≤ 1024. The launcher picks min(hidden,
// 256, max_threads). `hidden` is expected to be a multiple of the tg size on
// the hot path (≥ 1024 for Qwen3.5-MoE), but the kernel handles partial tails
// via the strided loop.
// Dispatched as `dispatch_thread_groups(grid_groups=(rows,1,1), tg=(tg_size,1,1))`
// — 1D so all positional attributes resolve to scalars (Metal forbids mixing
// scalar and vector attributes in a single kernel signature).
kernel void rms_norm_f32(
    device const float* x      [[buffer(0)]],
    device const float* gamma  [[buffer(1)]],
    device float*       y      [[buffer(2)]],
    constant uint&      hidden [[buffer(3)]],
    constant float&     eps    [[buffer(4)]],
    threadgroup float*  sdata  [[threadgroup(0)]],
    uint row                   [[threadgroup_position_in_grid]],
    uint lane                  [[thread_index_in_threadgroup]],
    uint tg_size               [[threads_per_threadgroup]]
) {
    uint base = row * hidden;

    // Stage 1: per-thread partial sum-of-squares over a strided slice.
    float local_sq = 0.0f;
    for (uint i = lane; i < hidden; i += tg_size) {
        float v = x[base + i];
        local_sq += v * v;
    }
    sdata[lane] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 2: tree reduction across the threadgroup.
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            sdata[lane] += sdata[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean_sq = sdata[0] / float(hidden);
    float scale   = rsqrt(mean_sq + eps);

    // Stage 3: normalize + apply gamma in a single strided pass.
    for (uint i = lane; i < hidden; i += tg_size) {
        y[base + i] = (x[base + i] * scale) * gamma[i];
    }
}

// ─── Partial rotary embedding (GPT-NeoX split form) ─────────────────────────
//
// Applies RoPE to the first `2*half_d` (= rotary_dim) components of each head;
// remaining `head_dim - rotary_dim` components pass through unchanged.
//
// Layout:
//   x, y : [batch, seq, heads, head_dim]  (row-major contiguous)
//   cos, sin : [seq, half_d]
//
// Pair `(x[i], x[i+half_d])` for i in [0, half_d) is rotated by `(cos[s,i], sin[s,i])`:
//   out[i]        = x[i] * cos[i] - x[i+half] * sin[i]
//   out[i+half]   = x[i] * sin[i] + x[i+half] * cos[i]
//
// Each thread writes one (b, s, h, i) element. Caller must not alias `x` and `y`
// (each thread reads from a partner offset that another thread writes).
//
// Grid: (head_dim, heads, batch*seq). Threadgroup: (≤ head_dim, 1, 1).
kernel void rope_partial_f32(
    device const float* x        [[buffer(0)]],
    device const float* cos_t    [[buffer(1)]],
    device const float* sin_t    [[buffer(2)]],
    device float*       y        [[buffer(3)]],
    constant uint&      seq      [[buffer(4)]],
    constant uint&      heads    [[buffer(5)]],
    constant uint&      head_dim [[buffer(6)]],
    constant uint&      half_d   [[buffer(7)]],
    uint3 gid                    [[thread_position_in_grid]]
) {
    uint i  = gid.x;
    uint h  = gid.y;
    uint bs = gid.z;
    if (i >= head_dim || h >= heads) { return; }

    uint s    = bs % seq;
    uint base = (bs * heads + h) * head_dim;

    float v;
    if (i < half_d) {
        float c  = cos_t[s * half_d + i];
        float si = sin_t[s * half_d + i];
        float a  = x[base + i];
        float b_ = x[base + i + half_d];
        v = a * c - b_ * si;
    } else if (i < 2u * half_d) {
        uint k  = i - half_d;
        float c  = cos_t[s * half_d + k];
        float si = sin_t[s * half_d + k];
        float a  = x[base + k];
        float b_ = x[base + i];
        v = a * si + b_ * c;
    } else {
        v = x[base + i];
    }
    y[base + i] = v;
}

// ─── Causal masked attention (single-tile baseline) ─────────────────────────
//
// Computes `out = softmax(Q · K^T / sqrt(D) + causal_mask) · V`.
//
// Layout (post-GQA expansion — caller broadcasts KV heads to match Q):
//   Q   : [B, H, L_q,  D]
//   K, V: [B, H, L_kv, D]
//   out : [B, H, L_q,  D]
//
// One threadgroup per `(b, h, q_idx)`. Threads cooperate via threadgroup memory:
//   sscores[k] holds the per-k score during the softmax phases.
// Sequential max/sum reductions inside lane 0 — fine for parity correctness;
// later phases swap to a tree-reduction or FlashAttention-style tile loop.
//
// Causal mask: q_idx → q_pos = `pos_offset + q_idx`. Score at `k` is `-INFINITY`
// if `k > q_pos`. `pos_offset` lets decode reuse the kernel by pointing past
// the prompt.
//
// Constraint: `L_kv` must fit in the threadgroup memory provided by the
// launcher (`L_kv * 4` bytes). For Qwen3.5 with 256K context this needs
// FlashAttention tiling — added in step 2c.
// `tg_size` is passed as a constant buffer (not the `[[threads_per_threadgroup]]`
// attribute) because Metal forces the latter to match the dispatch dimensionality
// (uint3 for our 3D group dispatch), which can't be mixed with the scalar
// `thread_index_in_threadgroup` in the same kernel signature.
//
// GQA-aware: each Q head `q_h` shares K/V head `q_h / (q_heads / kv_heads)`
// (PyTorch `repeat_interleave` semantics). When `kv_heads == q_heads` this
// degenerates to plain MHA.
kernel void attention_causal_f32(
    device const float* Q                 [[buffer(0)]],
    device const float* K                 [[buffer(1)]],
    device const float* V                 [[buffer(2)]],
    device float*       out               [[buffer(3)]],
    constant uint&      q_heads           [[buffer(4)]],
    constant uint&      kv_heads          [[buffer(5)]],
    constant uint&      l_q               [[buffer(6)]],
    constant uint&      l_kv              [[buffer(7)]],
    constant uint&      head_dim          [[buffer(8)]],
    constant uint&      pos_offset        [[buffer(9)]],
    constant float&     scale             [[buffer(10)]],
    constant uint&      tg_size           [[buffer(11)]],
    constant uint&      apply_causal      [[buffer(12)]],
    constant uint&      kv_layout_stride  [[buffer(13)]],
    threadgroup float*  sscores           [[threadgroup(0)]],
    uint3 tg_pos                          [[threadgroup_position_in_grid]],
    uint  lane                            [[thread_index_in_threadgroup]]
) {
    uint q_idx = tg_pos.x;
    uint q_h   = tg_pos.y;
    uint b     = tg_pos.z;
    if (q_idx >= l_q || q_h >= q_heads) { return; }

    uint group = q_heads / kv_heads;
    uint kv_h  = q_h / group;

    // K/V indexing uses `kv_layout_stride` for the inter-head step. When the
    // tensor is contiguous post-RoPE this equals `l_kv` and the kernel behaves
    // as before. When K/V come from an `NativeKvCache` buffer of physical shape
    // `[B, kv_heads, max_seq_len, head_dim]`, callers pass `l_kv = current_seq_len`
    // (the active range) and `kv_layout_stride = max_seq_len` (the capacity), so
    // `kv_h+1` skips the entire reserved row regardless of fill state.
    uint q_base   = ((b * q_heads  + q_h)  * l_q              + q_idx) * head_dim;
    uint kv_base  =  (b * kv_heads + kv_h) * kv_layout_stride * head_dim;
    uint out_base = q_base;

    uint q_pos = pos_offset + q_idx;

    // Phase 1: scores[k] = (Q · K[k]) * scale, with optional causal mask.
    // Each thread computes the full D-dim dot product for its strided k slice.
    // `apply_causal == 0` skips the upper-triangular -inf substitution, giving
    // bidirectional attention — matches the `mask=None` path Candle's production
    // self_attn takes.
    for (uint k = lane; k < l_kv; k += tg_size) {
        if (apply_causal != 0u && k > q_pos) {
            sscores[k] = -INFINITY;
        } else {
            float dot = 0.0f;
            for (uint d = 0; d < head_dim; ++d) {
                dot += Q[q_base + d] * K[kv_base + k * head_dim + d];
            }
            sscores[k] = dot * scale;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: max reduction (sequential — single lane).
    threadgroup float s_max;
    threadgroup float s_sum;
    if (lane == 0) {
        float m = -INFINITY;
        for (uint k = 0; k < l_kv; ++k) {
            float v = sscores[k];
            if (v > m) { m = v; }
        }
        s_max = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: exp(scores - max), in place.
    for (uint k = lane; k < l_kv; k += tg_size) {
        sscores[k] = exp(sscores[k] - s_max);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: sum reduction (sequential).
    if (lane == 0) {
        float s = 0.0f;
        for (uint k = 0; k < l_kv; ++k) { s += sscores[k]; }
        s_sum = s == 0.0f ? 1.0f : s;  // avoid div-by-zero on fully-masked rows
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_sum = 1.0f / s_sum;

    // Phase 5: out[d] = sum_k sscores[k] * V[k, d] / sum.
    // Threads parallelize over d.
    for (uint d = lane; d < head_dim; d += tg_size) {
        float o = 0.0f;
        for (uint k = 0; k < l_kv; ++k) {
            o += sscores[k] * V[kv_base + k * head_dim + d];
        }
        out[out_base + d] = o * inv_sum;
    }
}

// ─── Transpose [B, L, H, D] ↔ [B, H, L, D] ──────────────────────────────────
//
// `direction == 0`: input  [B, L, H, D] → output [B, H, L, D]
// `direction == 1`: input  [B, H, L, D] → output [B, L, H, D]
//
// Each thread copies one (b, *, *, d) cell. Grid: (D, L*H, B). With both
// orientations supported in one kernel the caller just flips the direction
// flag to undo the transpose after attention.
kernel void transpose_blhd_f32(
    device const float* x         [[buffer(0)]],
    device float*       y         [[buffer(1)]],
    constant uint&      l         [[buffer(2)]],
    constant uint&      h         [[buffer(3)]],
    constant uint&      d         [[buffer(4)]],
    constant uint&      direction [[buffer(5)]],
    uint3 gid                     [[thread_position_in_grid]]
) {
    uint di = gid.x;
    uint lh = gid.y;
    uint b  = gid.z;
    if (di >= d) { return; }

    // Always interpret gid.y as the BLHD-style (l_idx, h_idx) pair so the
    // indexing is symmetric between the two directions; only the src/dst
    // role flips with `direction`.
    uint l_idx = lh / h;
    uint h_idx = lh - l_idx * h;

    uint blhd_off = ((b * l + l_idx) * h + h_idx) * d + di;
    uint bhld_off = ((b * h + h_idx) * l + l_idx) * d + di;

    if (direction == 0u) {
        // BLHD → BHLD
        y[bhld_off] = x[blhd_off];
    } else {
        // BHLD → BLHD
        y[blhd_off] = x[bhld_off];
    }
}

// ─── GatedDeltaNet SSM single-step update (Phase A.4) ───────────────────────
//
// Performs one timestep of the recurrent state update used by Qwen3-Next's
// linear attention layer. The Candle reference (qwen3_5_moe::linear_attn::forward)
// does this via a Python-style loop:
//
//   state *= g[..., None, None]
//   kv_mem = sum(state * k[..., None, :], dim=-1)
//   delta  = (v - kv_mem) * beta[..., None]
//   state += k[..., None, :] * delta[..., None]
//   y      = sum(state * q[..., None, :], dim=-1)
//
// Per call (single timestep):
//   state : [B, Hv, Dv, Dk] f32 (in/out)
//   q, k  : [B, Hv, Dk] f32
//   v     : [B, Hv, Dv] f32
//   beta  : [B, Hv] f32
//   g     : [B, Hv] f32
//   y     : [B, Hv, Dv] f32 (out)
//
// One threadgroup per `(b, hv, dv)` cell. Threads cooperate over the `Dk`
// inner axis: parallel decay + kv_mem reduction → broadcast delta → parallel
// state update + y reduction. Two tree reductions per step.
//
// `tg_size` must be a power of 2 ≤ Dk. Threadgroup memory: `tg_size * 4`
// bytes (one scratch row for the reductions).
kernel void ssm_step_f32(
    device float*       state    [[buffer(0)]],
    device const float* q        [[buffer(1)]],
    device const float* k        [[buffer(2)]],
    device const float* v        [[buffer(3)]],
    device const float* beta     [[buffer(4)]],
    device const float* g        [[buffer(5)]],
    device float*       y        [[buffer(6)]],
    constant uint&      Hv       [[buffer(7)]],
    constant uint&      Dv       [[buffer(8)]],
    constant uint&      Dk       [[buffer(9)]],
    constant uint&      tg_size  [[buffer(10)]],
    threadgroup float*  sbuf     [[threadgroup(0)]],
    uint3 tg_pos                 [[threadgroup_position_in_grid]],
    uint  lane                   [[thread_index_in_threadgroup]]
) {
    uint dv = tg_pos.x;
    uint hv = tg_pos.y;
    uint b  = tg_pos.z;
    if (dv >= Dv || hv >= Hv) { return; }

    uint state_base  = ((b * Hv + hv) * Dv + dv) * Dk;
    uint kv_vec_base = (b * Hv + hv) * Dk;
    uint v_base      = (b * Hv + hv) * Dv;
    uint scalar_base =  b * Hv + hv;

    float beta_v = beta[scalar_base];
    float g_v    = g[scalar_base];
    float v_val  = v[v_base + dv];

    // Phase 1: state[dv, dk] *= g, accumulate kv_mem = sum_dk state * k.
    float local_kv = 0.0f;
    for (uint dk = lane; dk < Dk; dk += tg_size) {
        float st = state[state_base + dk] * g_v;
        state[state_base + dk] = st;
        local_kv += st * k[kv_vec_base + dk];
    }
    sbuf[lane] = local_kv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (lane < stride) { sbuf[lane] += sbuf[lane + stride]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float kv_mem = sbuf[0];

    // Phase 2: delta = (v - kv_mem) * beta. Scalar — every thread computes the
    // same value (cheap).
    float delta = (v_val - kv_mem) * beta_v;

    // Phase 3: state[dv, dk] += k[dk] * delta, accumulate y = sum_dk state * q.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_y = 0.0f;
    for (uint dk = lane; dk < Dk; dk += tg_size) {
        float st = state[state_base + dk] + k[kv_vec_base + dk] * delta;
        state[state_base + dk] = st;
        local_y += st * q[kv_vec_base + dk];
    }
    sbuf[lane] = local_y;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (lane < stride) { sbuf[lane] += sbuf[lane + stride]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0u) {
        y[v_base + dv] = sbuf[0];
    }
}

// ─── Helpers used by linear_attn (GatedDeltaNet) ────────────────────────────

// Weightless RMSNorm on the last axis: `y = x / sqrt(mean(x^2) + eps)`.
// Same dispatch convention as `rms_norm_f32`: 1D thread-group dispatch with
// one threadgroup per row, all positional attributes scalar uint.
kernel void rms_norm_weightless_f32(
    device const float* x      [[buffer(0)]],
    device float*       y      [[buffer(1)]],
    constant uint&      hidden [[buffer(2)]],
    constant float&     eps    [[buffer(3)]],
    threadgroup float*  sdata  [[threadgroup(0)]],
    uint row                   [[threadgroup_position_in_grid]],
    uint lane                  [[thread_index_in_threadgroup]],
    uint tg_size               [[threads_per_threadgroup]]
) {
    uint base = row * hidden;

    float local_sq = 0.0f;
    for (uint i = lane; i < hidden; i += tg_size) {
        float v = x[base + i];
        local_sq += v * v;
    }
    sdata[lane] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (lane < stride) { sdata[lane] += sdata[lane + stride]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean_sq = sdata[0] / float(hidden);
    float scale   = rsqrt(mean_sq + eps);

    for (uint i = lane; i < hidden; i += tg_size) {
        y[base + i] = x[base + i] * scale;
    }
}

// Numerically stable softplus: `ln(1 + exp(x))` computed via
// `max(x, 0) + ln1p(exp(-|x|))`. Element-wise; one thread per element.
kernel void softplus_f32(
    device const float* x [[buffer(0)]],
    device float*       y [[buffer(1)]],
    constant uint&      n [[buffer(2)]],
    uint gid              [[thread_position_in_grid]]
) {
    if (gid >= n) { return; }
    float v = x[gid];
    float pos = max(v, 0.0f);
    float log1p_exp = log(1.0f + exp(-fabs(v)));
    y[gid] = pos + log1p_exp;
}

// Element-wise SiLU: `x * sigmoid(x)`.
kernel void silu_f32(
    device const float* x [[buffer(0)]],
    device float*       y [[buffer(1)]],
    constant uint&      n [[buffer(2)]],
    uint gid              [[thread_position_in_grid]]
) {
    if (gid >= n) { return; }
    float v = x[gid];
    y[gid] = v / (1.0f + exp(-v));
}

// Element-wise sigmoid: `1 / (1 + exp(-x))`.
kernel void sigmoid_f32(
    device const float* x [[buffer(0)]],
    device float*       y [[buffer(1)]],
    constant uint&      n [[buffer(2)]],
    uint gid              [[thread_position_in_grid]]
) {
    if (gid >= n) { return; }
    y[gid] = 1.0f / (1.0f + exp(-x[gid]));
}

// ─── Linear-attn helpers (Phase A.8-C.3) ────────────────────────────────────
//
// All five kernels below back the host-side `forward_linear_attn` rewrite.
// They share a flat (rows × per_head) layout where the *fastest* axis is the
// per-head dim (`hv` for the gated chain, `head_dim` for the head repeat).

// y[i] = x[i] + bias[i % hv].  Used to add `dt_bias` onto `a_flat` along the
// trailing head axis: x:[B,S,Hv], bias:[Hv].
kernel void broadcast_add_per_head_f32(
    device const float* x    [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device float*       y    [[buffer(2)]],
    constant uint&      n    [[buffer(3)]],
    constant uint&      hv   [[buffer(4)]],
    uint gid                 [[thread_position_in_grid]]
) {
    if (gid >= n) { return; }
    uint h = gid % hv;
    y[gid] = x[gid] + bias[h];
}

// y[i] = x[i] * scale[i % hv].  Used to fold `softplus(a + dt_bias) * exp(a_log)`
// per head: x:[B,S,Hv], scale:[Hv].
kernel void mul_broadcast_per_head_f32(
    device const float* x     [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device float*       y     [[buffer(2)]],
    constant uint&      n     [[buffer(3)]],
    constant uint&      hv    [[buffer(4)]],
    uint gid                  [[thread_position_in_grid]]
) {
    if (gid >= n) { return; }
    uint h = gid % hv;
    y[gid] = x[gid] * scale[h];
}

// y[i] = exp(-x[i]).  Final step of the gated_decay chain
// (`g = exp(-softplus(a + dt) * exp(a_log))`).
kernel void neg_exp_f32(
    device const float* x [[buffer(0)]],
    device float*       y [[buffer(1)]],
    constant uint&      n [[buffer(2)]],
    uint gid              [[thread_position_in_grid]]
) {
    if (gid >= n) { return; }
    y[gid] = exp(-x[gid]);
}

// Phase 19.A.4 — fused compute_g + beta. Replaces 5 element-wise dispatches:
//   beta      = sigmoid(b)
//   a_plus_dt = a + dt_bias[h]
//   softplus  = softplus(a_plus_dt)
//   g_pre     = softplus * exp_a_log[h]
//   g         = exp(-g_pre)
// Layout: b/a/beta_out/g_out flat [B*S*Hv]; dt_bias/exp_a_log [Hv].
// Mirrors MLX's `@partial(mx.compile) compute_g` (gated_delta.py:8-11).
kernel void compute_g_full_f32(
    device const float* b         [[buffer(0)]],
    device const float* a         [[buffer(1)]],
    device const float* dt_bias   [[buffer(2)]],
    device const float* exp_a_log [[buffer(3)]],
    device float*       beta_out  [[buffer(4)]],
    device float*       g_out     [[buffer(5)]],
    constant uint&      n         [[buffer(6)]],
    constant uint&      hv        [[buffer(7)]],
    uint gid                      [[thread_position_in_grid]]
) {
    if (gid >= n) { return; }
    uint h = gid % hv;
    float bv = b[gid];
    beta_out[gid] = 1.0f / (1.0f + exp(-bv));
    float a_dt = a[gid] + dt_bias[h];
    float pos = max(a_dt, 0.0f);
    float log1p_exp = log(1.0f + exp(-fabs(a_dt)));
    float sp = pos + log1p_exp;
    float g_pre = sp * exp_a_log[h];
    g_out[gid] = exp(-g_pre);
}

// Repeat heads along axis 2 for `[B, S, Hk, D] → [B, S, Hk*repeats, D]`
// with `repeat_interleave` semantics: output head index `hv` reads from
// source head `hv / repeats`.
//
// Grid: (head_dim, hv_out, B*S).  One thread per output (b, s, hv, d).
kernel void repeat_heads_blhd_f32(
    device const float* x          [[buffer(0)]],
    device float*       y          [[buffer(1)]],
    constant uint&      hk         [[buffer(2)]],
    constant uint&      repeats    [[buffer(3)]],
    constant uint&      head_dim   [[buffer(4)]],
    uint3 gid                      [[thread_position_in_grid]]
) {
    uint d  = gid.x;
    uint hv = gid.y;
    uint bs = gid.z;
    if (d >= head_dim) { return; }
    uint hv_total = hk * repeats;
    if (hv >= hv_total) { return; }
    uint hk_src = hv / repeats;
    uint x_off = bs * (hk * head_dim) + hk_src * head_dim + d;
    uint y_off = bs * (hv_total * head_dim) + hv * head_dim + d;
    y[y_off] = x[x_off];
}

// y[i] = scale * x[i] + bias.  Replaces Candle `Tensor::affine` for
// the two `inv_scale` rescales after the weightless RMSNorm.
kernel void affine_scalar_f32(
    device const float* x     [[buffer(0)]],
    device float*       y     [[buffer(1)]],
    constant uint&      n     [[buffer(2)]],
    constant float&     scale [[buffer(3)]],
    constant float&     bias  [[buffer(4)]],
    uint gid                  [[thread_position_in_grid]]
) {
    if (gid >= n) { return; }
    y[gid] = x[gid] * scale + bias;
}

// ─── Depthwise causal conv1d + SiLU (Phase A.8-C.7-c) ───────────────────────
//
// Replaces the GatedDeltaNet 8-op Candle path
//   (4 narrows → stack → broadcast_mul → sum → silu)
// with a single fused dispatch: per-channel causal kernel reduction over a
// fixed `kernel_size` window with a fused SiLU activation on the output.
//
// Layouts (all f32 row-major, contiguous):
//   x:      [B, kernel_size - 1 + S, C]   (prev_conv_state ++ qkv_flat)
//   weight: [C, kernel_size]              (depthwise: per-channel weights)
//   y:      [B, S, C]                     (post-SiLU output)
//
// Grid: (C, S, B). One thread per output element; the inner loop reduces
// over `kernel_size` (4 in production). Fused SiLU avoids a second kernel.
kernel void depthwise_conv1d_silu_f32(
    device const float* x      [[buffer(0)]],
    device const float* w      [[buffer(1)]],
    device float*       y      [[buffer(2)]],
    constant uint&      batch  [[buffer(3)]],
    constant uint&      seq    [[buffer(4)]],
    constant uint&      ksize  [[buffer(5)]],
    constant uint&      chan   [[buffer(6)]],
    uint3 gid                  [[thread_position_in_grid]]
) {
    uint c = gid.x;
    uint t = gid.y;
    uint b = gid.z;
    if (c >= chan || t >= seq || b >= batch) { return; }

    uint x_total = ksize - 1u + seq;
    uint x_b_base = b * x_total * chan;

    float acc = 0.0f;
    for (uint k = 0u; k < ksize; ++k) {
        float xv = x[x_b_base + (t + k) * chan + c];
        float wv = w[c * ksize + k];
        acc += xv * wv;
    }
    float sig = 1.0f / (1.0f + exp(-acc));
    y[b * seq * chan + t * chan + c] = acc * sig;
}

// ─── Linear-attn output fusion (Phase A.8-D) ────────────────────────────────
//
// Fused `silu(z) * x` to keep the RMSNormGated tail of the linear-attention
// post-conv pipeline inside the same Metal command buffer as the SSM loop,
// so the host can encode out_proj on top without committing first.
//
// All three buffers share the same flat layout (`[B, S, V]` viewed as length
// `n`); the host computes `n = batch * seq * v_dim` once.
kernel void silu_mul_f32(
    device const float* z [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float*       y [[buffer(2)]],
    constant uint&      n [[buffer(3)]],
    uint gid              [[thread_position_in_grid]]
) {
    if (gid >= n) { return; }
    float zv = z[gid];
    float silu_z = zv / (1.0f + exp(-zv));
    y[gid] = silu_z * x[gid];
}

// Fused `y = sigmoid(g) * x`. Used by the self-attention output gate to
// keep the gating + o_proj tail inside the post-attention command buffer
// (Phase A.8-D). Same flat-shape contract as `silu_mul_f32`.
kernel void sigmoid_mul_f32(
    device const float* g [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float*       y [[buffer(2)]],
    constant uint&      n [[buffer(3)]],
    uint gid              [[thread_position_in_grid]]
) {
    if (gid >= n) { return; }
    float gv = g[gid];
    float sig = 1.0f / (1.0f + exp(-gv));
    y[gid] = sig * x[gid];
}

// ─── Workstream B Phase 6 — bf16 SSM subgraph kernels ───────────────────────
//
// bf16 variants of the post-conv SSM subgraph kernels, mirroring the f32
// variants above. Compute happens in float (bfloat read → float convert →
// float compute → bfloat write) so the only difference is I/O bandwidth and
// dtype contract. State buffers in `ssm_step_bf16` stay float32 (Escape #3 —
// the recurrent state is the lone f32 island, same as MLX's gated_delta and
// the `tq_gated_delta_step_bf16` kernel in turboquant-metal).
//
// Used by `forward_post_conv_fused_*` when `conv_out.dtype() == BF16`. q/k/v
// flow as bfloat through QK-norm → affine → repeat_heads → ssm_step. State
// remains f32 across timesteps. Output `y_n` is bfloat; the rms_norm tail
// casts back to f32 (norm_weight is f32; cheap one-shot cast).

// Element-wise f32 → bf16 cast. Used to convert beta/g (computed in f32 by
// sigmoid/softplus chain) to bf16 just before `ssm_step_bf16`. Trivial.
kernel void cast_f32_to_bf16(
    device const float*  x [[buffer(0)]],
    device bfloat*       y [[buffer(1)]],
    constant uint&       n [[buffer(2)]],
    uint gid               [[thread_position_in_grid]]
) {
    if (gid >= n) { return; }
    y[gid] = bfloat(x[gid]);
}

// Element-wise bf16 → f32 cast. Used to bridge `y_n` (bf16 output of
// `ssm_step_bf16`) into the f32 RMSNormGated tail in `run_post_conv_fused`.
kernel void cast_bf16_to_f32(
    device const bfloat* x [[buffer(0)]],
    device float*        y [[buffer(1)]],
    constant uint&       n [[buffer(2)]],
    uint gid               [[thread_position_in_grid]]
) {
    if (gid >= n) { return; }
    y[gid] = float(x[gid]);
}

kernel void rms_norm_weightless_bf16(
    device const bfloat* x      [[buffer(0)]],
    device bfloat*       y      [[buffer(1)]],
    constant uint&       hidden [[buffer(2)]],
    constant float&      eps    [[buffer(3)]],
    threadgroup float*   sdata  [[threadgroup(0)]],
    uint row                    [[threadgroup_position_in_grid]],
    uint lane                   [[thread_index_in_threadgroup]],
    uint tg_size                [[threads_per_threadgroup]]
) {
    uint base = row * hidden;

    float local_sq = 0.0f;
    for (uint i = lane; i < hidden; i += tg_size) {
        float v = float(x[base + i]);
        local_sq += v * v;
    }
    sdata[lane] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (lane < stride) { sdata[lane] += sdata[lane + stride]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean_sq = sdata[0] / float(hidden);
    float scale   = rsqrt(mean_sq + eps);

    for (uint i = lane; i < hidden; i += tg_size) {
        y[base + i] = bfloat(float(x[base + i]) * scale);
    }
}

kernel void affine_scalar_bf16(
    device const bfloat* x     [[buffer(0)]],
    device bfloat*       y     [[buffer(1)]],
    constant uint&       n     [[buffer(2)]],
    constant float&      scale [[buffer(3)]],
    constant float&      bias  [[buffer(4)]],
    uint gid                   [[thread_position_in_grid]]
) {
    if (gid >= n) { return; }
    y[gid] = bfloat(float(x[gid]) * scale + bias);
}

kernel void repeat_heads_blhd_bf16(
    device const bfloat* x        [[buffer(0)]],
    device bfloat*       y        [[buffer(1)]],
    constant uint&       hk       [[buffer(2)]],
    constant uint&       repeats  [[buffer(3)]],
    constant uint&       head_dim [[buffer(4)]],
    uint3 gid                     [[thread_position_in_grid]]
) {
    uint d  = gid.x;
    uint hv = gid.y;
    uint bs = gid.z;
    if (d >= head_dim) { return; }
    uint hv_total = hk * repeats;
    if (hv >= hv_total) { return; }
    uint hk_src = hv / repeats;
    uint x_off = bs * (hk * head_dim) + hk_src * head_dim + d;
    uint y_off = bs * (hv_total * head_dim) + hv * head_dim + d;
    y[y_off] = x[x_off];
}

// bf16 variant of `ssm_step_f32`. Same SIMD/threadgroup geometry; q/k/v/beta/g
// read as bfloat (converted to float at use), y written as bfloat. State
// stays float32 — Escape #3 (recurrent state cannot live in bf16 without
// drift across long sequences; matches MLX `_gated_delta_step_ops` and
// `tq_gated_delta_step_bf16` in turboquant-metal).
kernel void ssm_step_bf16(
    device float*        state    [[buffer(0)]],
    device const bfloat* q        [[buffer(1)]],
    device const bfloat* k        [[buffer(2)]],
    device const bfloat* v        [[buffer(3)]],
    device const bfloat* beta     [[buffer(4)]],
    device const bfloat* g        [[buffer(5)]],
    device bfloat*       y        [[buffer(6)]],
    constant uint&       Hv       [[buffer(7)]],
    constant uint&       Dv       [[buffer(8)]],
    constant uint&       Dk       [[buffer(9)]],
    constant uint&       tg_size  [[buffer(10)]],
    threadgroup float*   sbuf     [[threadgroup(0)]],
    uint3 tg_pos                  [[threadgroup_position_in_grid]],
    uint  lane                    [[thread_index_in_threadgroup]]
) {
    uint dv = tg_pos.x;
    uint hv = tg_pos.y;
    uint b  = tg_pos.z;
    if (dv >= Dv || hv >= Hv) { return; }

    uint state_base  = ((b * Hv + hv) * Dv + dv) * Dk;
    uint kv_vec_base = (b * Hv + hv) * Dk;
    uint v_base      = (b * Hv + hv) * Dv;
    uint scalar_base =  b * Hv + hv;

    float beta_v = float(beta[scalar_base]);
    float g_v    = float(g[scalar_base]);
    float v_val  = float(v[v_base + dv]);

    // Phase 1: state[dv, dk] *= g, accumulate kv_mem = sum_dk state * k.
    float local_kv = 0.0f;
    for (uint dk = lane; dk < Dk; dk += tg_size) {
        float st = state[state_base + dk] * g_v;
        state[state_base + dk] = st;
        local_kv += st * float(k[kv_vec_base + dk]);
    }
    sbuf[lane] = local_kv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (lane < stride) { sbuf[lane] += sbuf[lane + stride]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float kv_mem = sbuf[0];

    // Phase 2: delta = (v - kv_mem) * beta. Scalar — every thread same value.
    float delta = (v_val - kv_mem) * beta_v;

    // Phase 3: state[dv, dk] += k[dk] * delta, accumulate y = sum_dk state * q.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_y = 0.0f;
    for (uint dk = lane; dk < Dk; dk += tg_size) {
        float st = state[state_base + dk] + float(k[kv_vec_base + dk]) * delta;
        state[state_base + dk] = st;
        local_y += st * float(q[kv_vec_base + dk]);
    }
    sbuf[lane] = local_y;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (lane < stride) { sbuf[lane] += sbuf[lane + stride]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0u) {
        y[v_base + dv] = bfloat(sbuf[0]);
    }
}
