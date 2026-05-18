#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Gated Delta-Net SSM step kernel (port of MLX `gated_delta_step`)
// ============================================================================
//
// Replaces our Candle-based ops loop in `qwen3_5_moe::linear_attn::forward_inner`
// (8+ Candle dispatches per timestep × 48 linear-attn layers ≈ 380 dispatches
// per decode step) with a single Metal kernel.
//
// Algorithm (per timestep, identical to mlx_lm.models.gated_delta._gated_delta_step_ops):
//   state *= g          // decay
//   kv_mem = state · k  // simdgroup-reduced dot over Dk
//   delta  = (v - kv_mem) * beta
//   state += k ⊗ delta  // outer product update
//   y      = state · q  // simdgroup-reduced dot over Dk
//
// Shapes (f32 throughout — Candle widens to f32 inside the SSM loop):
//   q, k       : [B, T, Hk, Dk]  contiguous
//   v          : [B, T, Hv, Dv]  contiguous
//   g, beta    : [B, T, Hv]      contiguous (scalar gating per head)
//   state_in   : [B, Hv, Dv, Dk] contiguous (zeros on cold start)
//   y          : [B, T, Hv, Dv]  output
//   state_out  : [B, Hv, Dv, Dk] output
//
// Configuration passed via buffer args (uniform constants):
//   T_val, Dk, Dv, Hk, Hv  — runtime dims (Dk must be multiple of 32)
//
// Grid: (32, Dv, B*Hv)    Threadgroup: (32, 4, 1) — one simdgroup × 4 dv per TG.
// Each thread holds n_per_t = Dk/32 state floats in registers; simd_sum reduces
// across the 32 lanes of the simdgroup over the Dk dimension.

kernel void tq_gated_delta_step(
    device const float* __restrict__ q        [[buffer(0)]],   // [B,T,Hk,Dk]
    device const float* __restrict__ k        [[buffer(1)]],   // [B,T,Hk,Dk]
    device const float* __restrict__ v        [[buffer(2)]],   // [B,T,Hv,Dv]
    device const float* __restrict__ g        [[buffer(3)]],   // [B,T,Hv]
    device const float* __restrict__ beta     [[buffer(4)]],   // [B,T,Hv]
    device const float* __restrict__ state_in [[buffer(5)]],   // [B,Hv,Dv,Dk]
    device       float*              y        [[buffer(6)]],   // [B,T,Hv,Dv]
    device       float*              state_out[[buffer(7)]],   // [B,Hv,Dv,Dk]
    constant uint&                   T_val    [[buffer(8)]],
    constant uint&                   Dk       [[buffer(9)]],
    constant uint&                   Dv       [[buffer(10)]],
    constant uint&                   Hk       [[buffer(11)]],
    constant uint&                   Hv       [[buffer(12)]],
    uint3 tpig  [[thread_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]]
) {
    uint n_per_t = Dk / 32u;
    uint h_ratio = Hv / Hk;

    uint n      = tpig.z;          // 0 .. B*Hv-1
    uint b_idx  = n / Hv;
    uint hv_idx = n % Hv;
    uint hk_idx = hv_idx / h_ratio;

    uint dk_lane = tpitg.x;        // 0..31, lane within simdgroup over Dk tile
    uint dv_idx  = tpig.y;         // 0 .. Dv-1

    // q, k: [B, T, Hk, Dk]
    uint qk_base = b_idx * T_val * Hk * Dk + hk_idx * Dk;
    // v, y:   [B, T, Hv, Dv]
    uint v_base  = b_idx * T_val * Hv * Dv + hv_idx * Dv;
    // g, beta: [B, T, Hv]
    uint g_base  = b_idx * T_val * Hv;

    device const float* q_     = q       + qk_base;
    device const float* k_     = k       + qk_base;
    device const float* v_     = v       + v_base;
    device       float* y_     = y       + v_base;
    device const float* g_     = g       + g_base;
    device const float* beta_  = beta    + g_base;

    // state_in / state_out: [B, Hv, Dv, Dk] — index = ((b*Hv + hv)*Dv + dv)*Dk
    uint state_off = (n * Dv + dv_idx) * Dk;
    device const float* i_state = state_in  + state_off;
    device       float* o_state = state_out + state_off;

    // Per-thread state slice in registers — n_per_t (e.g., 4 for Dk=128) elems.
    float state[16];               // upper bound; actual usage = n_per_t ≤ Dk/32 ≤ 16 (Dk ≤ 512)
    for (uint i = 0; i < n_per_t; ++i) {
        uint s_idx = n_per_t * dk_lane + i;
        state[i] = i_state[s_idx];
    }

    for (uint t = 0; t < T_val; ++t) {
        // Phase 1: decay + kv_mem = state · k
        float kv_mem = 0.0f;
        float gv = g_[hv_idx];
        for (uint i = 0; i < n_per_t; ++i) {
            uint s_idx = n_per_t * dk_lane + i;
            state[i] = state[i] * gv;
            kv_mem += state[i] * k_[s_idx];
        }
        kv_mem = simd_sum(kv_mem);

        // Phase 2: delta = (v - kv_mem) * beta — broadcast across all lanes
        float delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

        // Phase 3: state update + y = state · q
        float out = 0.0f;
        for (uint i = 0; i < n_per_t; ++i) {
            uint s_idx = n_per_t * dk_lane + i;
            state[i] = state[i] + k_[s_idx] * delta;
            out += state[i] * q_[s_idx];
        }
        out = simd_sum(out);
        if (simd_lid == 0) {
            y_[dv_idx] = out;
        }

        // Advance pointers to next timestep.
        q_     += Hk * Dk;
        k_     += Hk * Dk;
        v_     += Hv * Dv;
        y_     += Hv * Dv;
        g_     += Hv;
        beta_  += Hv;
    }

    // Persist final state.
    for (uint i = 0; i < n_per_t; ++i) {
        uint s_idx = n_per_t * dk_lane + i;
        o_state[s_idx] = state[i];
    }
}

// ============================================================================
// tq_gated_delta_step_bf16 — bf16 I/O variant (Workstream B Phase 4)
// ============================================================================
//
// Same algorithm as tq_gated_delta_step. Differences:
//   - q, k, v, g, beta : bfloat (cast to float on read)
//   - y                : bfloat (cast from float on write)
//   - state_in/_out    : float  (UNCHANGED — Escape #3 in MLX gated_delta.py)
//   - All internal compute stays float (preserves accumulation precision over Dk)
//
// Mirrors MLX `gated_delta_kernel` template `(InT=bf16, StT=f32)` in
// mlx-lm gated_delta.py, verified line-by-line on 2026-05-08.
// I/O bandwidth halved on q/k/v/g/beta/y (the bulk of the per-token traffic);
// state I/O unchanged because state ranking is f32 to match MLX numerics.

kernel void tq_gated_delta_step_bf16(
    device const bfloat* __restrict__ q        [[buffer(0)]],   // [B,T,Hk,Dk]
    device const bfloat* __restrict__ k        [[buffer(1)]],   // [B,T,Hk,Dk]
    device const bfloat* __restrict__ v        [[buffer(2)]],   // [B,T,Hv,Dv]
    device const bfloat* __restrict__ g        [[buffer(3)]],   // [B,T,Hv]
    device const bfloat* __restrict__ beta     [[buffer(4)]],   // [B,T,Hv]
    device const float*  __restrict__ state_in [[buffer(5)]],   // [B,Hv,Dv,Dk] f32
    device       bfloat*              y        [[buffer(6)]],   // [B,T,Hv,Dv]
    device       float*               state_out[[buffer(7)]],   // [B,Hv,Dv,Dk] f32
    constant uint&                    T_val    [[buffer(8)]],
    constant uint&                    Dk       [[buffer(9)]],
    constant uint&                    Dv       [[buffer(10)]],
    constant uint&                    Hk       [[buffer(11)]],
    constant uint&                    Hv       [[buffer(12)]],
    uint3 tpig  [[thread_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]]
) {
    uint n_per_t = Dk / 32u;
    uint h_ratio = Hv / Hk;

    uint n      = tpig.z;
    uint b_idx  = n / Hv;
    uint hv_idx = n % Hv;
    uint hk_idx = hv_idx / h_ratio;

    uint dk_lane = tpitg.x;
    uint dv_idx  = tpig.y;

    uint qk_base = b_idx * T_val * Hk * Dk + hk_idx * Dk;
    uint v_base  = b_idx * T_val * Hv * Dv + hv_idx * Dv;
    uint g_base  = b_idx * T_val * Hv;

    device const bfloat* q_     = q       + qk_base;
    device const bfloat* k_     = k       + qk_base;
    device const bfloat* v_     = v       + v_base;
    device       bfloat* y_     = y       + v_base;
    device const bfloat* g_     = g       + g_base;
    device const bfloat* beta_  = beta    + g_base;

    uint state_off = (n * Dv + dv_idx) * Dk;
    device const float* i_state = state_in  + state_off;
    device       float* o_state = state_out + state_off;

    float state[16];
    for (uint i = 0; i < n_per_t; ++i) {
        uint s_idx = n_per_t * dk_lane + i;
        state[i] = i_state[s_idx];     // f32 read — state is f32
    }

    for (uint t = 0; t < T_val; ++t) {
        float kv_mem = 0.0f;
        float gv = float(g_[hv_idx]);
        for (uint i = 0; i < n_per_t; ++i) {
            uint s_idx = n_per_t * dk_lane + i;
            state[i] = state[i] * gv;
            kv_mem += state[i] * float(k_[s_idx]);
        }
        kv_mem = simd_sum(kv_mem);

        float delta = (float(v_[dv_idx]) - kv_mem) * float(beta_[hv_idx]);

        float out = 0.0f;
        for (uint i = 0; i < n_per_t; ++i) {
            uint s_idx = n_per_t * dk_lane + i;
            state[i] = state[i] + float(k_[s_idx]) * delta;
            out += state[i] * float(q_[s_idx]);
        }
        out = simd_sum(out);
        if (simd_lid == 0) {
            y_[dv_idx] = bfloat(out);  // narrow on write
        }

        q_     += Hk * Dk;
        k_     += Hk * Dk;
        v_     += Hv * Dv;
        y_     += Hv * Dv;
        g_     += Hv;
        beta_  += Hv;
    }

    for (uint i = 0; i < n_per_t; ++i) {
        uint s_idx = n_per_t * dk_lane + i;
        o_state[s_idx] = state[i];     // f32 write — state stays f32
    }
}
