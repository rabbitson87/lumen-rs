//! Two kernels for the routed and shared experts of a single-token MoE decode step.
//!
//! The composed block runs the experts as about twenty-five MLX ops: the gate/up gather, the
//! SwiGLU, the down gather, the weighting and sum over the routed experts, then the shared expert's
//! three matmuls, its SwiGLU and gate, and the final add. Decode is GPU-bound and each op costs
//! about a microsecond of GPU time on its own, on top of the bytes it reads.
//!
//! * [`GATE_UP_SOURCE`]: one simdgroup per (slot, intermediate row), where slot 0..7 are the routed
//!   experts and slot 8 the shared expert. Gate and up are dotted against the activation and
//!   SwiGLU'd in the same kernel.
//! * [`DOWN_COMBINE_SOURCE`]: one simdgroup per output row. Each slot's down projection is
//!   weighted by its score (the shared expert by its sigmoid gate) and summed.
//!
//! Routed slots round where the composed block rounds: the activation to bf16, the gate/up and down
//! dot products to bf16 (the down output before it is weighted), and SwiGLU op by op as MLX's bf16
//! `sigmoid`, `multiply` do. The shared expert stays in f32 as it does there.
//!
//! The router has two kernels of its own ([`ROUTER_LOGITS_SOURCE`], [`ROUTER_SELECT_SOURCE`]): the
//! 256 expert logits and the shared expert's gate logit, then softmax, top-8, renormalisation and
//! the shared sigmoid in one simdgroup. They replace the gate matmul, softmax, argpartition,
//! take-along, sum, divide, the shared gate matmul and its sigmoid (twelve dispatches).
//!
//! One rounding is easy to miss and decides whether the result matches. MLX's quantized matvec
//! sums the activation four elements at a time in the activation's own type (`load_vector` adds
//! `x[i]`, `x[i+1]`, `x[i+2]` and `x[i+3]` left to right before adding the total to its f32
//! sum). For bf16 activations that sum, which `bias` multiplies, rounds to bf16 after every add.
//! Dotting in exact f32 instead is the more accurate result and moves every gate/up activation by
//! bf16 noise: 12.9% of the activations then equal the composed block's, against 99.97% with the
//! same quad sums. The remaining difference is the order of the f32 accumulation inside a dot
//! product.

use std::cell::OnceCell;

use anyhow::{Context, Result, anyhow};
use mlx_rs::{Array, Dtype};

use crate::metal_kernel::{MetalKernel, MetalKernelConfig};
use crate::native_moe::{
    AffineGateWeights, GateProj, GateUpWeights, MoeBlockWeights, SharedMlpWeights, SwitchGluWeights,
};
use crate::native_quant::MODE_AFFINE;

lumen_flags::flag! {
    /// Route a single-token step with two kernels (the expert and shared-gate logits; then softmax,
    /// top-8, renormalisation and the shared sigmoid) instead of the twelve MLX ops, when
    /// `LUMEN_NATIVE_FUSED_MOE_DECODE` runs the experts in kernels and the router is an 8-bit
    /// group-64 gate over a multiple of 32 experts.
    ///
    /// Softmax and the top-8 are computed in f32 as the composed ops do; the logits' dot products
    /// accumulate in a different order, so two experts whose probabilities differ by float noise
    /// can swap places. A `Behavior` flag. Default on; `=0` keeps the composed router.
    ///
    /// Qwen3.6-35B-A3B affine, M3 Max, pipelined greedy decode, ABBA over six server runs: 11.9 ms
    /// composed, 11.1 ms with the expert kernels only, 10.5 ms with the router kernels too (-0.6
    /// ms; the router alone is 31 us a layer, 1.24 ms a token, in the composed ops). Teacher-forced
    /// top-1 agreement with the composed ops over 4,050 positions: 98.79%, against 98.62% between
    /// the composed model's prefill and decode kernels.
    pub(crate) fused_router {
        env: "LUMEN_NATIVE_FUSED_MOE_ROUTER",
        default: true,
        kind: Behavior,
    }
}

/// Routed experts per token. The kernels are written for Qwen's top-8 routing.
const TOP_K: i32 = 8;
/// Simdgroups per threadgroup of the expert kernels.
const ROWS_PER_GROUP: i32 = 4;
/// MLX hands a kernel input of fewer than eight elements over in the `constant` address space.
/// The router kernel picks the shared gate's scales and biases (`[1, hidden / 64]`) or the
/// experts' (`device`) with one expression, which does not compile across address spaces, so a
/// router needs at least this many groups per row to run on the kernels.
const MIN_GATE_GROUPS: i32 = 8;
/// Rows one simdgroup of each expert kernel owns, so the activation it loads is used for all of
/// them (MLX's own quantized matvec does four). More rows means fewer simdgroups to hide memory
/// latency with, so the best count is each kernel's own, found on the benchmark
/// (`moe_routed_probe`).
const ROWS_GATE_UP: i32 = 2;
const ROWS_DOWN: i32 = 1;

const GATE_UP_SOURCE: &str = r#"
    #define RB(v) static_cast<float>(static_cast<bfloat16_t>(v))
    // MLX sums the activation four at a time in the activation's own type (`sum += x[i] + x[i + 1]
    // + x[i + 2] + x[i + 3]`), so bf16 activations round after every add; f32 ones do not.
    #define QSUM(a, b, c, d) (routed ? RB(RB(RB((a) + (b)) + (c)) + (d)) : (((a) + (b)) + (c)) + (d))
    const int lane = thread_index_in_simdgroup;
    const int rg = thread_position_in_grid.y;
    const int slot = thread_position_in_grid.z;
    constexpr int G = H / 64;
    constexpr int WPR = H / 8;
    const bool routed = slot < K;
    const uint e = routed ? inds[slot] : 0u;

    float g_part[R];
    float u_part[R];
    for (int r = 0; r < R; ++r) {
        g_part[r] = 0.0f;
        u_part[r] = 0.0f;
    }
    for (int grp = lane; grp < G; grp += 32) {
        // The group's activation, once for the R rows this simdgroup owns.
        float xq[64];
        float xs = 0.0f;
        for (int j = 0; j < 8; ++j) {
            for (int i = 0; i < 8; ++i) {
                const float raw = x[grp * 64 + j * 8 + i];
                xq[j * 8 + i] = routed ? RB(raw) : raw;
            }
            xs += QSUM(xq[j * 8], xq[j * 8 + 1], xq[j * 8 + 2], xq[j * 8 + 3]);
            xs += QSUM(xq[j * 8 + 4], xq[j * 8 + 5], xq[j * 8 + 6], xq[j * 8 + 7]);
            // Nibble i is read where it sits (masked, not shifted), so its activation is divided
            // by 16^i once here for all R rows. Both factors are exact powers of two times a
            // 4-bit integer, so the product is bit for bit x * nibble.
            for (int i = 0; i < 8; ++i) {
                xq[j * 8 + i] *= 1.0f / static_cast<float>(1u << (4 * i));
            }
        }
        for (int r = 0; r < R; ++r) {
            const int row = rg * R + r;
            const size_t rgate = (size_t)e * (2 * I) + row;
            const size_t rup = rgate + I;
            auto wg = routed ? (w + rgate * WPR) : (gw + (size_t)row * WPR);
            auto wu = routed ? (w + rup * WPR) : (uw + (size_t)row * WPR);
            auto sg = routed ? (sc + rgate * G) : (gs + (size_t)row * G);
            auto su = routed ? (sc + rup * G) : (us + (size_t)row * G);
            auto bg = routed ? (bi + rgate * G) : (gb + (size_t)row * G);
            auto bu = routed ? (bi + rup * G) : (ub + (size_t)row * G);
            float ag = 0.0f;
            float au = 0.0f;
            for (int j = 0; j < 8; ++j) {
                const uint32_t pg = wg[grp * 8 + j];
                const uint32_t pu = wu[grp * 8 + j];
                for (int i = 0; i < 8; ++i) {
                    ag += xq[j * 8 + i] * static_cast<float>(pg & (0xFu << (4 * i)));
                    au += xq[j * 8 + i] * static_cast<float>(pu & (0xFu << (4 * i)));
                }
            }
            g_part[r] += static_cast<float>(sg[grp]) * ag + static_cast<float>(bg[grp]) * xs;
            u_part[r] += static_cast<float>(su[grp]) * au + static_cast<float>(bu[grp]) * xs;
        }
    }
    for (int r = 0; r < R; ++r) {
        const float g = simd_sum(g_part[r]);
        const float u = simd_sum(u_part[r]);
        if (lane == 0) {
            float a;
            if (routed) {
                const float gr = RB(g);
                const float ur = RB(u);
                // MLX's sigmoid, op by op: y = 1 / (1 + exp(|x|)); x < 0 ? y : 1 - y.
                const float ex = RB(metal::exp(metal::abs(gr)));
                const float d = RB(1.0f + ex);
                const float y = RB(1.0f / d);
                const float sig = gr < 0.0f ? y : RB(1.0f - y);
                a = RB(RB(gr * sig) * ur);
            } else {
                a = (g / (1.0f + metal::exp(-g))) * u;
            }
            act[slot * I + rg * R + r] = a;
        }
    }
"#;

const DOWN_COMBINE_SOURCE: &str = r#"
    #define RB(v) static_cast<float>(static_cast<bfloat16_t>(v))
    #define QSUM(a, b, c, d) (routed ? RB(RB(RB((a) + (b)) + (c)) + (d)) : (((a) + (b)) + (c)) + (d))
    const int lane = thread_index_in_simdgroup;
    const int rg = thread_position_in_grid.y;
    constexpr int G = I / 64;
    constexpr int WPR = I / 8;
    constexpr int CHUNKS = I / 16;

    float total[R];
    for (int r = 0; r < R; ++r) {
        total[r] = 0.0f;
    }
    for (int s = 0; s <= K; ++s) {
        const bool routed = s < K;
        const uint e = routed ? inds[s] : 0u;
        float part[R];
        for (int r = 0; r < R; ++r) {
            part[r] = 0.0f;
        }
        for (int c = lane; c < CHUNKS; c += 32) {
            // This lane's 16 activations, once for the R rows.
            float av[16];
            for (int i = 0; i < 16; ++i) {
                av[i] = act[s * I + c * 16 + i];
            }
            float xs = 0.0f;
            xs += QSUM(av[0], av[1], av[2], av[3]);
            xs += QSUM(av[4], av[5], av[6], av[7]);
            xs += QSUM(av[8], av[9], av[10], av[11]);
            xs += QSUM(av[12], av[13], av[14], av[15]);
            // As in the gate/up kernel: nibble i is masked in place, its activation pre-divided.
            for (int i = 0; i < 8; ++i) {
                const float inv = 1.0f / static_cast<float>(1u << (4 * i));
                av[i] *= inv;
                av[8 + i] *= inv;
            }
            for (int r = 0; r < R; ++r) {
                const int row = rg * R + r;
                const size_t rr = (size_t)e * H + row;
                auto wp = routed ? (w + rr * WPR) : (dw + (size_t)row * WPR);
                auto sp = routed ? (sc + rr * G) : (ds + (size_t)row * G);
                auto bp = routed ? (bi + rr * G) : (db + (size_t)row * G);
                const uint32_t p0 = wp[c * 2];
                const uint32_t p1 = wp[c * 2 + 1];
                float acc = 0.0f;
                for (int i = 0; i < 8; ++i) {
                    acc += av[i] * static_cast<float>(p0 & (0xFu << (4 * i)));
                    acc += av[8 + i] * static_cast<float>(p1 & (0xFu << (4 * i)));
                }
                part[r] += static_cast<float>(sp[c / 4]) * acc + static_cast<float>(bp[c / 4]) * xs;
            }
        }
        for (int r = 0; r < R; ++r) {
            const float y = simd_sum(part[r]);
            total[r] += routed ? scores[s] * RB(y) : sscore[0] * y;
        }
    }
    if (lane == 0) {
        for (int r = 0; r < R; ++r) {
            out[rg * R + r] = total[r];
        }
    }
"#;

/// One simdgroup per gate row: rows `0..E` are the experts' router logits, row `E` the shared
/// expert's gate. 8-bit affine group 64, activation in f32 (so MLX sums it in f32 as well).
const ROUTER_LOGITS_SOURCE: &str = r#"
    const int lane = thread_index_in_simdgroup;
    const int row = thread_position_in_grid.y;
    constexpr int G = H / 64;
    constexpr int WPR = H / 4;
    const bool shared_row = row == E;
    auto wr = shared_row ? sw : (gw + (size_t)row * WPR);
    auto sr = shared_row ? ss : (gs + (size_t)row * G);
    auto br = shared_row ? sb : (gb + (size_t)row * G);

    float part = 0.0f;
    for (int grp = lane; grp < G; grp += 32) {
        float acc = 0.0f;
        float xs = 0.0f;
        for (int j = 0; j < 16; ++j) {
            const uint32_t p = wr[grp * 16 + j];
            for (int i = 0; i < 4; ++i) {
                const float xv = x[grp * 64 + j * 4 + i];
                acc += xv * static_cast<float>((p >> (8 * i)) & 0xFFu);
                xs += xv;
            }
        }
        part += static_cast<float>(sr[grp]) * acc + static_cast<float>(br[grp]) * xs;
    }
    const float v = simd_sum(part);
    if (lane == 0) {
        logits[row] = v;
    }
"#;

/// One simdgroup (E / 32 logits a lane, so no barrier anywhere): softmax over the `E` expert
/// logits (`1 / sum` multiplied in, as MLX does), the eight largest probabilities by repeated
/// argmax (the lowest index wins a tie), their renormalisation, and the shared expert's sigmoid
/// gate.
const ROUTER_SELECT_SOURCE: &str = r#"
    constexpr int PER = E / 32;
    const uint lane = thread_index_in_simdgroup;

    float p[PER];
    float m = -INFINITY;
    for (int k = 0; k < PER; ++k) {
        p[k] = logits[lane + k * 32];
        m = metal::max(m, p[k]);
    }
    m = simd_max(m);
    float sum = 0.0f;
    for (int k = 0; k < PER; ++k) {
        p[k] = metal::exp(p[k] - m);
        sum += p[k];
    }
    sum = simd_sum(sum);
    const float inv = 1.0f / sum;
    for (int k = 0; k < PER; ++k) {
        p[k] = p[k] * inv;
    }

    float sel_p[8];
    uint sel_i[8];
    for (int r = 0; r < 8; ++r) {
        float bv = -INFINITY;
        uint bi = 0xFFFFFFFFu;
        for (int k = 0; k < PER; ++k) {
            if (p[k] > bv) {
                bv = p[k];
                bi = lane + k * 32;
            }
        }
        const float gv = simd_max(bv);
        const uint gi = simd_min(bv == gv ? bi : 0xFFFFFFFFu);
        for (int k = 0; k < PER; ++k) {
            if (lane + k * 32 == gi) {
                p[k] = -INFINITY;
            }
        }
        sel_p[r] = gv;
        sel_i[r] = gi;
    }

    if (lane == 0) {
        float denom = 0.0f;
        for (int r = 0; r < 8; ++r) {
            denom += sel_p[r];
        }
        for (int r = 0; r < 8; ++r) {
            inds[r] = sel_i[r];
            scores[r] = sel_p[r] / denom;
        }
        const float z = logits[E];
        const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(z)));
        sscore[0] = z < 0.0f ? y : 1.0f - y;
    }
"#;

struct Kernels {
    gate_up: MetalKernel,
    down: MetalKernel,
    router_logits: MetalKernel,
    router_select: MetalKernel,
}

impl Kernels {
    fn new() -> Result<Self> {
        // Each kernel is constructed with its name as a literal: `cargo xtask test` reads the
        // names off `MetalKernel::new` calls to check that every kernel ran in its Metal 3.1 pass.
        Ok(Self {
            gate_up: MetalKernel::new(
                "moe_decode_gate_up_swiglu",
                &[
                    "x", "w", "sc", "bi", "gw", "gs", "gb", "uw", "us", "ub", "inds",
                ],
                &["act"],
                GATE_UP_SOURCE,
                /* ensure_row_contiguous */ true,
                /* atomic_outputs */ false,
            )?,
            down: MetalKernel::new(
                "moe_decode_down_combine",
                &[
                    "act", "w", "sc", "bi", "dw", "ds", "db", "inds", "scores", "sscore",
                ],
                &["out"],
                DOWN_COMBINE_SOURCE,
                /* ensure_row_contiguous */ true,
                /* atomic_outputs */ false,
            )?,
            router_logits: MetalKernel::new(
                "moe_decode_router_logits",
                &["x", "gw", "gs", "gb", "sw", "ss", "sb"],
                &["logits"],
                ROUTER_LOGITS_SOURCE,
                /* ensure_row_contiguous */ true,
                /* atomic_outputs */ false,
            )?,
            router_select: MetalKernel::new(
                "moe_decode_router_select",
                &["logits"],
                &["inds", "scores", "sscore"],
                ROUTER_SELECT_SOURCE,
                /* ensure_row_contiguous */ true,
                /* atomic_outputs */ false,
            )?,
        })
    }
}

thread_local! {
    /// MLX keeps one stream per thread and the engine decodes on one thread, so the kernels are
    /// built once per thread rather than per layer per step.
    static KERNELS: OnceCell<Kernels> = const { OnceCell::new() };
}

fn with_kernels<R>(f: impl FnOnce(&Kernels) -> Result<R>) -> Result<R> {
    KERNELS.with(|cell| {
        if cell.get().is_none() {
            let _ = cell.set(Kernels::new()?);
        }
        f(cell
            .get()
            .ok_or_else(|| anyhow!("moe decode kernels missing"))?)
    })
}

/// The sizes a supported layer has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Dims {
    pub hidden: i32,
    pub inter: i32,
}

/// `Some` when the kernels can run this layer's decode step: one token, affine 4-bit group-64
/// experts stacked gate-then-up, a shared expert of the same width, biases and float scales of
/// one dtype per role.
pub(crate) fn plan(
    x: &Array,
    glu: &SwitchGluWeights<'_>,
    shared: &SharedMlpWeights<'_>,
    top_k: i32,
) -> Option<Dims> {
    let [1, 1, hidden] = *x.shape() else {
        return None;
    };
    let GateUpWeights::Fused(gate_up) = &glu.gate_up else {
        return None;
    };
    if top_k != TOP_K
        || glu.mode != MODE_AFFINE
        || shared.mode != MODE_AFFINE
        || (glu.group_size, glu.bits, shared.group_size, shared.bits) != (64, 4, 64, 4)
    {
        return None;
    }
    let experts = *gate_up.weight.shape().first()?;
    let inter = *gate_up.weight.shape().get(1)? / 2;
    if hidden % 64 != 0 || inter % 16 != 0 || inter % ROWS_GATE_UP != 0 || hidden % ROWS_DOWN != 0 {
        return None;
    }
    let shape_is = |a: &Array, want: &[i32]| a.shape() == want;
    let float = |d: Dtype| matches!(d, Dtype::Bfloat16 | Dtype::Float16 | Dtype::Float32);
    let (gu_b, down_b) = (gate_up.biases?, glu.down_proj.biases?);
    let (sg_b, su_b, sd_b) = (shared.gate_proj_b?, shared.up_proj_b?, shared.down_proj_b?);
    let shapes_ok = shape_is(gate_up.weight, &[experts, 2 * inter, hidden / 8])
        && shape_is(gate_up.scales, &[experts, 2 * inter, hidden / 64])
        && shape_is(gu_b, &[experts, 2 * inter, hidden / 64])
        && shape_is(glu.down_proj.weight, &[experts, hidden, inter / 8])
        && shape_is(glu.down_proj.scales, &[experts, hidden, inter / 64])
        && shape_is(down_b, &[experts, hidden, inter / 64])
        && shape_is(shared.gate_proj_w, &[inter, hidden / 8])
        && shape_is(shared.up_proj_w, &[inter, hidden / 8])
        && shape_is(shared.down_proj_w, &[hidden, inter / 8])
        && shape_is(shared.gate_proj_s, &[inter, hidden / 64])
        && shape_is(shared.up_proj_s, &[inter, hidden / 64])
        && shape_is(shared.down_proj_s, &[hidden, inter / 64])
        && shape_is(sg_b, &[inter, hidden / 64])
        && shape_is(su_b, &[inter, hidden / 64])
        && shape_is(sd_b, &[hidden, inter / 64]);
    // The kernels pick routed or shared pointers with one expression, so each role's two tensors
    // must share a type.
    let gate_up_dtype = gate_up.scales.dtype();
    let down_dtype = glu.down_proj.scales.dtype();
    let dtypes_ok = float(gate_up_dtype)
        && float(down_dtype)
        && [
            gu_b.dtype(),
            shared.gate_proj_s.dtype(),
            shared.up_proj_s.dtype(),
            sg_b.dtype(),
            su_b.dtype(),
        ]
        .iter()
        .all(|d| *d == gate_up_dtype)
        && [down_b.dtype(), shared.down_proj_s.dtype(), sd_b.dtype()]
            .iter()
            .all(|d| *d == down_dtype);
    (shapes_ok && dtypes_ok).then_some(Dims { hidden, inter })
}

fn need<'a>(b: Option<&'a Array>, what: &str) -> Result<&'a Array> {
    b.ok_or_else(|| anyhow!("missing {what} biases"))
}

/// Gate, up and SwiGLU of the eight routed experts and the shared expert: `[9 * inter]` f32, the
/// shared expert's row last. `x` is `[hidden]` f32, `inds` `[8]` u32.
pub(crate) fn gate_up_swiglu(
    x: &Array,
    inds: &Array,
    glu: &SwitchGluWeights<'_>,
    shared: &SharedMlpWeights<'_>,
    dims: Dims,
) -> Result<Array> {
    let GateUpWeights::Fused(gate_up) = &glu.gate_up else {
        return Err(anyhow!(
            "moe decode kernels need the stacked gate/up experts"
        ));
    };
    let Dims { hidden, inter } = dims;
    let slots = TOP_K + 1;
    with_kernels(|k| {
        let config = MetalKernelConfig::new();
        config.add_template_arg_int("H", hidden)?;
        config.add_template_arg_int("I", inter)?;
        config.add_template_arg_int("K", TOP_K)?;
        config.add_template_arg_int("R", ROWS_GATE_UP)?;
        config.set_grid(32, inter / ROWS_GATE_UP, slots)?;
        config.set_thread_group(32, ROWS_PER_GROUP, 1)?;
        config.add_output_arg(&[slots * inter], Dtype::Float32)?;
        Ok(k.gate_up
            .apply(
                &[
                    x,
                    gate_up.weight,
                    gate_up.scales,
                    need(gate_up.biases, "gate/up")?,
                    shared.gate_proj_w,
                    shared.gate_proj_s,
                    need(shared.gate_proj_b, "shared gate")?,
                    shared.up_proj_w,
                    shared.up_proj_s,
                    need(shared.up_proj_b, "shared up")?,
                    inds,
                ],
                &config,
                1,
            )
            .context("moe decode: gate/up/swiglu kernel failed")?
            .swap_remove(0))
    })
}

/// Down projections of the nine slots, weighted and summed: `[hidden]` f32.
pub(crate) fn down_combine(
    act: &Array,
    inds: &Array,
    scores: &Array,
    shared_score: &Array,
    glu: &SwitchGluWeights<'_>,
    shared: &SharedMlpWeights<'_>,
    dims: Dims,
) -> Result<Array> {
    let Dims { hidden, inter } = dims;
    with_kernels(|k| {
        let config = MetalKernelConfig::new();
        config.add_template_arg_int("H", hidden)?;
        config.add_template_arg_int("I", inter)?;
        config.add_template_arg_int("K", TOP_K)?;
        config.add_template_arg_int("R", ROWS_DOWN)?;
        config.set_grid(32, hidden / ROWS_DOWN, 1)?;
        config.set_thread_group(32, ROWS_PER_GROUP, 1)?;
        config.add_output_arg(&[hidden], Dtype::Float32)?;
        Ok(k.down
            .apply(
                &[
                    act,
                    glu.down_proj.weight,
                    glu.down_proj.scales,
                    need(glu.down_proj.biases, "down")?,
                    shared.down_proj_w,
                    shared.down_proj_s,
                    need(shared.down_proj_b, "shared down")?,
                    inds,
                    scores,
                    shared_score,
                ],
                &config,
                1,
            )
            .context("moe decode: down/combine kernel failed")?
            .swap_remove(0))
    })
}

/// The gate tensors of a router the kernels can run.
struct RouterGates<'a> {
    experts: i32,
    gate: &'a AffineGateWeights<'a>,
    shared: &'a AffineGateWeights<'a>,
}

fn router_gates<'a>(w: &'a MoeBlockWeights<'a>, dims: Dims) -> Option<RouterGates<'a>> {
    let (GateProj::Affine(gate), GateProj::Affine(shared)) = (&w.gate, &w.shared_gate) else {
        return None;
    };
    let experts = *gate.weight.shape().first()?;
    let rows = |a: &Array, n: i32, cols: i32| a.shape() == [n, cols];
    let ok = w.norm_topk_prob
        && w.top_k == TOP_K
        && experts % 32 == 0
        && dims.hidden / 64 >= MIN_GATE_GROUPS
        && (gate.bits, gate.group_size, shared.bits, shared.group_size) == (8, 64, 8, 64)
        && rows(gate.weight, experts, dims.hidden / 4)
        && rows(gate.scales, experts, dims.hidden / 64)
        && rows(gate.biases, experts, dims.hidden / 64)
        && rows(shared.weight, 1, dims.hidden / 4)
        && rows(shared.scales, 1, dims.hidden / 64)
        && rows(shared.biases, 1, dims.hidden / 64)
        && [
            gate.biases.dtype(),
            shared.scales.dtype(),
            shared.biases.dtype(),
        ]
        .iter()
        .all(|d| *d == gate.scales.dtype())
        && matches!(
            gate.scales.dtype(),
            Dtype::Bfloat16 | Dtype::Float16 | Dtype::Float32
        );
    ok.then_some(RouterGates {
        experts,
        gate,
        shared,
    })
}

/// Whether [`route`] can replace the composed router of this layer.
pub(crate) fn router_supported(w: &MoeBlockWeights<'_>, dims: Dims) -> bool {
    fused_router::get() && router_gates(w, dims).is_some()
}

/// The routing of `x` (`[1, 1, hidden]`): the eight experts' ids (`[8]` u32), their renormalised
/// probabilities (`[8]` f32) and the shared expert's sigmoid gate (`[1]` f32).
pub(crate) fn route(
    x: &Array,
    w: &MoeBlockWeights<'_>,
    dims: Dims,
) -> Result<(Array, Array, Array)> {
    let logits = route_logits(x, w, dims)?;
    route_select(&logits, w, dims)
}

/// The router's first kernel: the `[experts + 1]` f32 logits, the shared expert's gate last.
pub(crate) fn route_logits(x: &Array, w: &MoeBlockWeights<'_>, dims: Dims) -> Result<Array> {
    let gates = router_gates(w, dims).ok_or_else(|| anyhow!("moe decode: router not supported"))?;
    let x = mlx_rs::ops::reshape(&x.as_dtype(Dtype::Float32)?, &[dims.hidden])?;
    with_kernels(|k| {
        let logits_config = MetalKernelConfig::new();
        logits_config.add_template_arg_int("H", dims.hidden)?;
        logits_config.add_template_arg_int("E", gates.experts)?;
        logits_config.set_grid(32, gates.experts + 1, 1)?;
        logits_config.set_thread_group(32, 1, 1)?;
        logits_config.add_output_arg(&[gates.experts + 1], Dtype::Float32)?;
        let logits = k
            .router_logits
            .apply(
                &[
                    &x,
                    gates.gate.weight,
                    gates.gate.scales,
                    gates.gate.biases,
                    gates.shared.weight,
                    gates.shared.scales,
                    gates.shared.biases,
                ],
                &logits_config,
                1,
            )
            .context("moe decode: router logits kernel failed")?
            .swap_remove(0);
        Ok(logits)
    })
}

/// The router's second kernel: softmax, top-8, renormalisation and the shared sigmoid over the
/// logits of [`route_logits`].
pub(crate) fn route_select(
    logits: &Array,
    w: &MoeBlockWeights<'_>,
    dims: Dims,
) -> Result<(Array, Array, Array)> {
    let gates = router_gates(w, dims).ok_or_else(|| anyhow!("moe decode: router not supported"))?;
    with_kernels(|k| {
        let select_config = MetalKernelConfig::new();
        select_config.add_template_arg_int("E", gates.experts)?;
        select_config.set_grid(32, 1, 1)?;
        select_config.set_thread_group(32, 1, 1)?;
        select_config.add_output_arg(&[TOP_K], Dtype::Uint32)?;
        select_config.add_output_arg(&[TOP_K], Dtype::Float32)?;
        select_config.add_output_arg(&[1], Dtype::Float32)?;
        let mut out = k
            .router_select
            .apply(&[logits], &select_config, 3)
            .context("moe decode: router select kernel failed")?;
        let shared_score = out
            .pop()
            .ok_or_else(|| anyhow!("router select: no shared score"))?;
        let scores = out
            .pop()
            .ok_or_else(|| anyhow!("router select: no scores"))?;
        let inds = out
            .pop()
            .ok_or_else(|| anyhow!("router select: no indices"))?;
        Ok((inds, scores, shared_score))
    })
}

/// The routed and shared experts' contribution to the block, `[1, 1, hidden]` f32: routed experts
/// `inds` weighted by `scores`, plus the shared expert weighted by `shared_score`.
pub(crate) fn run(
    x: &Array,
    inds: &Array,
    scores: &Array,
    shared_score: &Array,
    glu: &SwitchGluWeights<'_>,
    shared: &SharedMlpWeights<'_>,
    dims: Dims,
) -> Result<Array> {
    let hidden = dims.hidden;
    let x = mlx_rs::ops::reshape(&x.as_dtype(Dtype::Float32)?, &[hidden])?;
    let inds = mlx_rs::ops::reshape(&inds.as_dtype(Dtype::Uint32)?, &[TOP_K])?;
    let scores = mlx_rs::ops::reshape(&scores.as_dtype(Dtype::Float32)?, &[TOP_K])?;
    let shared_score = mlx_rs::ops::reshape(&shared_score.as_dtype(Dtype::Float32)?, &[1])?;
    let act = gate_up_swiglu(&x, &inds, glu, shared, dims)?;
    let out = down_combine(&act, &inds, &scores, &shared_score, glu, shared, dims)?;
    Ok(mlx_rs::ops::reshape(&out, &[1, 1, hidden])?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moe_routed_probe::Layer;
    use crate::native_moe::moe_block_forward;

    fn host(a: &Array) -> Vec<f32> {
        let a = a.as_dtype(Dtype::Float32).expect("f32");
        a.eval().expect("eval");
        a.as_slice::<f32>().to_vec()
    }

    /// `(cosine, largest gap / largest value)` of two outputs.
    fn closeness(a: &[f32], b: &[f32]) -> (f64, f64) {
        assert_eq!(a.len(), b.len());
        let dot: f64 = a
            .iter()
            .zip(b)
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum();
        let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
        let gap = a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        let top = a.iter().fold(0f32, |m, x| m.max(x.abs()));
        (dot / (norm(a) * norm(b)), f64::from(gap) / f64::from(top))
    }

    /// The decode kernels give the composed block's output (router, routed experts, shared
    /// expert, their sum) for the same weights and activation, up to f32 accumulation order.
    fn kernels_match_the_composed_block(hidden: i32, inter: i32, experts: i32) {
        let layer = Layer::build(hidden, inter, experts).expect("layer");
        let composed = layer.view(true, false);
        let kernels = layer.view(true, true);
        for seed in 0..6u64 {
            let x = layer.activation(100 + seed).expect("x");
            assert!(
                plan(
                    &x,
                    &kernels.switch_glu,
                    &kernels.shared_expert,
                    kernels.top_k
                )
                .is_some(),
                "the kernels must take this layer, or the comparison compares the composed block with itself"
            );
            let want = host(&moe_block_forward(&x, &composed).expect("composed block"));
            let got = host(&moe_block_forward(&x, &kernels).expect("kernel block"));
            let (cosine, gap) = closeness(&want, &got);
            assert!(
                cosine > 0.99999 && gap < 1e-2,
                "{hidden}x{inter}x{experts} seed {seed}: cosine {cosine:.6}, largest gap {gap:.3e} of the largest value"
            );
        }
    }

    #[test]
    fn kernels_match_the_composed_block_on_a_small_layer() {
        // Two input groups, so only two lanes of a simdgroup carry gate/up work.
        kernels_match_the_composed_block(128, 64, 16);
    }

    #[test]
    fn kernels_match_the_composed_block_at_the_a3b_shape() {
        kernels_match_the_composed_block(2048, 512, 256);
    }

    #[test]
    fn kernels_match_the_composed_block_when_the_rows_outnumber_the_lanes() {
        // 4096 hidden = 64 input groups (two per lane); 1024 intermediate = 64 chunks (two per lane).
        kernels_match_the_composed_block(4096, 1024, 16);
    }

    #[test]
    fn layers_the_kernels_cannot_run_stay_on_the_composed_block() {
        let layer = Layer::build(128, 64, 16).expect("layer");
        let x = layer.activation(1).expect("x");
        let stacked = layer.view(true, true);
        let top_k = stacked.top_k;
        assert!(plan(&x, &stacked.switch_glu, &stacked.shared_expert, top_k).is_some());

        // Separate gate and up: nothing stacked for the kernel to read.
        let split = layer.view(false, true);
        assert!(plan(&x, &split.switch_glu, &split.shared_expert, top_k).is_none());
        // Two tokens: prefill shape.
        let two = mlx_rs::ops::concatenate_axis(&[&x, &x], 1).expect("two tokens");
        assert!(plan(&two, &stacked.switch_glu, &stacked.shared_expert, top_k).is_none());
        // Not a top-8 router.
        assert!(plan(&x, &stacked.switch_glu, &stacked.shared_expert, 4).is_none());
    }

    /// The kernel's routed activations are the composed SwiGLU's, element for element but for the
    /// rare element whose f32 accumulation order tips a bf16 rounding. This is what the quad-sum
    /// emulation in the kernel buys: without it 13% match.
    #[test]
    fn routed_activations_are_the_composed_swiglu() {
        use crate::native_moe::{moe_router, swiglu_fused};
        use crate::native_quant::gather_qmm_with_mode;

        let layer = Layer::build(2048, 512, 256).expect("layer");
        let w = layer.view(true, true);
        let GateUpWeights::Fused(gate_up) = &w.switch_glu.gate_up else {
            panic!("a stacked layer");
        };
        let (mut equal, mut total) = (0usize, 0usize);
        for seed in 0..8u64 {
            let x = layer.activation(900 + seed).expect("x");
            let dims = plan(&x, &w.switch_glu, &w.shared_expert, w.top_k).expect("supported");
            let (inds, _) = moe_router(&x, &w).expect("router");
            let x32 =
                mlx_rs::ops::reshape(x.as_dtype(Dtype::Float32).unwrap(), &[dims.hidden]).unwrap();
            let inds32 =
                mlx_rs::ops::reshape(inds.as_dtype(Dtype::Uint32).unwrap(), &[TOP_K]).unwrap();
            let kernel = host(
                &gate_up_swiglu(&x32, &inds32, &w.switch_glu, &w.shared_expert, dims)
                    .expect("kernel"),
            );

            // The composed decode step: bf16 activation, one stacked gather, split, SwiGLU.
            let xb = x.as_dtype(Dtype::Bfloat16).unwrap();
            let x5 = mlx_rs::ops::expand_dims_axes(&xb, &[-2, -3]).unwrap();
            let both = gather_qmm_with_mode(
                &x5,
                gate_up.weight,
                gate_up.scales,
                gate_up.biases,
                None,
                Some(&inds),
                true,
                64,
                4,
                MODE_AFFINE,
                false,
            )
            .expect("gather");
            let mut halves = mlx_rs::ops::split(&both, 2, -1).unwrap().into_iter();
            let (gate, up) = (halves.next().unwrap(), halves.next().unwrap());
            let composed = host(&swiglu_fused(&gate, &up).expect("swiglu"));
            // The kernel's shared-expert row comes last; the composed routed rows are all of this.
            for (a, b) in composed.iter().zip(&kernel) {
                total += 1;
                equal += usize::from(a.to_bits() == b.to_bits());
            }
        }
        let share = equal as f64 / total as f64;
        assert!(
            share > 0.995,
            "only {:.2}% of the routed activations equal the composed block's",
            share * 100.0
        );
    }

    /// The router kernels pick the composed router's experts with its probabilities, and the
    /// shared gate's sigmoid, for many activations.
    fn router_matches_the_composed_router(hidden: i32, inter: i32, experts: i32, inputs: u64) {
        use crate::native_moe::moe_router;
        use crate::native_quant::quantized_matmul_with_mode;

        let layer = Layer::build(hidden, inter, experts).expect("layer");
        let w = layer.view(true, true);
        let GateProj::Affine(shared_gate) = &w.shared_gate else {
            panic!("an affine shared gate");
        };
        for seed in 0..inputs {
            let x = layer.activation(2000 + seed).expect("x");
            let dims = plan(&x, &w.switch_glu, &w.shared_expert, w.top_k).expect("supported");
            assert!(
                router_supported(&w, dims),
                "the kernels must take this router, or the comparison compares the composed one with itself"
            );
            let (want_inds, want_scores) = moe_router(&x, &w).expect("composed router");
            let (inds, scores, shared_score) = route(&x, &w, dims).expect("kernel router");

            let by_expert = |inds: &Array, scores: &Array| {
                let ids = host(&inds.as_dtype(Dtype::Float32).unwrap());
                let sc = host(scores);
                let mut pairs: Vec<(u32, f32)> = ids.iter().map(|i| *i as u32).zip(sc).collect();
                pairs.sort_by_key(|(id, _)| *id);
                pairs
            };
            let (want, got) = (
                by_expert(&want_inds, &want_scores),
                by_expert(&inds, &scores),
            );
            let ids = |v: &[(u32, f32)]| v.iter().map(|(i, _)| *i).collect::<Vec<_>>();
            assert_eq!(ids(&got), ids(&want), "seed {seed}: different experts");
            for ((_, a), (_, b)) in want.iter().zip(&got) {
                assert!((a - b).abs() < 1e-5, "seed {seed}: score {a} against {b}");
            }

            let logit = quantized_matmul_with_mode(
                &x,
                shared_gate.weight,
                shared_gate.scales,
                Some(shared_gate.biases),
                true,
                64,
                8,
                MODE_AFFINE,
            )
            .expect("shared gate");
            let want_shared = host(&mlx_rs::ops::sigmoid(&logit).unwrap())[0];
            let got_shared = host(&shared_score)[0];
            assert!(
                (want_shared - got_shared).abs() < 1e-6,
                "seed {seed}: shared gate {want_shared} against {got_shared}"
            );
        }
    }

    #[test]
    fn router_kernels_match_the_composed_router_at_the_a3b_shape() {
        router_matches_the_composed_router(2048, 512, 256, 120);
    }

    #[test]
    fn router_kernels_match_the_composed_router_with_512_experts() {
        router_matches_the_composed_router(1024, 256, 512, 60);
    }

    #[test]
    fn kernels_match_the_composed_block_with_a_512_expert_router() {
        kernels_match_the_composed_block(1024, 256, 512);
    }

    #[test]
    fn a_router_over_a_narrow_hidden_state_stays_on_the_composed_ops() {
        // 256 / 64 = 4 groups: the shared gate's scales hold four elements, which MLX would pass in
        // the constant address space and the router kernel could not compile.
        let layer = Layer::build(256, 64, 256).expect("layer");
        let w = layer.view(true, true);
        let x = layer.activation(3).expect("x");
        let dims = plan(&x, &w.switch_glu, &w.shared_expert, w.top_k).expect("experts supported");
        assert!(!router_supported(&w, dims));
        // The experts still run on the kernels, behind the composed router.
        kernels_match_the_composed_block(256, 64, 256);
    }

    #[test]
    fn a_router_over_a_few_experts_stays_on_the_composed_ops() {
        let layer = Layer::build(128, 64, 16).expect("layer");
        let w = layer.view(true, true);
        let x = layer.activation(3).expect("x");
        let dims = plan(&x, &w.switch_glu, &w.shared_expert, w.top_k).expect("experts supported");
        assert!(
            !router_supported(&w, dims),
            "a 16-expert router is not a multiple of 32"
        );
    }
}
