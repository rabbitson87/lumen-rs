//! Two kernels for the routed and shared experts of a single-token MoE decode step, and one for
//! the router's logits.
//!
//! The composed block runs the experts as about twenty-five MLX ops: the gate/up gather, the
//! SwiGLU, the down gather, the weighting and sum over the routed experts, then the shared expert's
//! three matmuls, its SwiGLU and gate, and the final add. Decode is GPU-bound and each op costs
//! about a microsecond of GPU time on its own, on top of the bytes it reads.
//!
//! * [`GATE_UP_SOURCE`]: one simdgroup per (slot, intermediate row), where slot 0..7 are the routed
//!   experts and slot 8 the shared expert. Gate and up are dotted against the activation and
//!   SwiGLU'd in the same kernel. With a router in front it also does the router's select (see
//!   below), so no kernel sits between the logits and the experts.
//! * [`DOWN_COMBINE_SOURCE`]: one simdgroup per slot, nine to a threadgroup, for a few output
//!   rows. Each slot's down projection is weighted by its score (the shared expert by its sigmoid
//!   gate) and summed.
//!
//! Routed slots round where the composed block rounds: the activation to bf16, the gate/up and down
//! dot products to bf16 (the down output before it is weighted), and SwiGLU op by op as MLX's bf16
//! `sigmoid`, `multiply` do. The shared expert stays in f32 as it does there.
//!
//! The router is one kernel of its own ([`ROUTER_LOGITS_SOURCE`], the 256 expert logits and the
//! shared expert's gate logit) plus a prologue of the gate/up kernel: every routed simdgroup
//! repeats softmax and an argmax per rank up to its own slot's, and one of them stores the
//! renormalised top-8 and the shared sigmoid for the down kernel. That replaces the gate matmul,
//! softmax, argpartition, take-along, sum, divide, the shared gate matmul and its sigmoid (twelve
//! dispatches), and a separate select kernel (one simdgroup, ~8 us, plus the dependency it
//! chained) that sat between the logits and the experts before.
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
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::{Array, Dtype};

use crate::metal_kernel::{MetalKernel, MetalKernelConfig};
use crate::native_moe::{
    AffineGateWeights, GateProj, GateUpWeights, MoeBlockWeights, SharedMlpWeights,
    SwitchExpertWeights, SwitchGluWeights,
};
use crate::native_qmv::{QMV_8BIT, QMV_BF16, QMV_F32};
use crate::native_quant::MODE_AFFINE;

lumen_flags::flag! {
    /// Route a single-token step with a logits kernel (the expert and shared-gate logits) and the
    /// select (softmax, top-8, renormalisation, the shared sigmoid) inside the experts' gate/up
    /// kernel, instead of the twelve MLX ops, when `LUMEN_NATIVE_FUSED_MOE_DECODE` runs the experts
    /// in kernels and the router is an 8-bit group-64 gate over a multiple of 128 experts.
    ///
    /// Bit-identical to the composed router: the logits run MLX's matvec arithmetic, the softmax
    /// adds as MLX's kernel does, and the eight experts come in the order the composed sort leaves
    /// them (ascending probability, equal ones by index), so the routing, the scores and the
    /// block's output equal the composed ones in every bit. An `Optimization` flag. Default on;
    /// `=0` keeps the composed router.
    ///
    /// Qwen3.6-35B-A3B affine, M3 Max, pipelined greedy decode, ABBA: 11.9 ms composed, 11.1 ms
    /// with the expert kernels only, 10.5 ms with the router kernels too (-0.6 ms; the router alone
    /// is 31 us a layer, 1.24 ms a token, in the composed ops); folding the select into the
    /// gate/up kernel took another 0.1 ms.
    pub(crate) fused_router {
        env: "LUMEN_NATIVE_FUSED_MOE_ROUTER",
        default: true,
        kind: Optimization,
    }
}

/// Routed experts per token. The kernels are written for Qwen's top-8 routing.
const TOP_K: i32 = 8;
/// Simdgroups per threadgroup of the expert kernels.
const ROWS_PER_GROUP: i32 = 4;
/// Rows one simdgroup of each expert kernel owns, so the activation it loads is used for all of
/// them (MLX's own quantized matvec does four). More rows means fewer simdgroups to hide memory
/// latency with, so the best count is each kernel's own, found end to end (ABBA on the 35B-A3B
/// decode, `moe_routed_probe` is too noisy to tell them apart). The down kernel repeats the
/// activation's setup for every row it owns, so it gained from two rows once the nibble masking
/// had made the dot products cheaper; four rows and four gate/up rows were no different.
const ROWS_GATE_UP: i32 = 2;
const ROWS_DOWN: i32 = 2;

/// MLX's Sigmoid on f32 and on bf16 (every operation rounds to bf16; `x` is already bf16), for the
/// gate/up kernel's SwiGLU and the router's shared gate. Beside it the expert kernels take MLX's
/// matvec helpers: the routed slots follow the composed block's bf16 activation ([`QMV_BF16`]),
/// the shared slot its f32 one ([`QMV_F32`]), the router its 8-bit gate ([`QMV_8BIT`]). The Metal
/// text carries no comments: MLX copies and hashes a run-time kernel's whole source on every call.
const SIGMOID_HEADER: &str = r#"
inline float sigmoid_f32(float x) {
    const float y = 1 / (1 + metal::precise::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
}
inline float sigmoid_bf16(float x) {
    const float e = RB(metal::precise::exp(metal::abs(x)));
    const float d = RB(1.0f + e);
    const float y = RB(1.0f / d);
    return (x < 0.0f) ? y : RB(1.0f - y);
}
"#;

/// Gate, up and SwiGLU of one slot's rows: `R` gate rows and `R` up rows a simdgroup, as MLX's
/// `qmv_fast` takes four, each summed in its order. A routed slot rounds as the composed block
/// does: the gather returns bf16, and the compiled SwiGLU is bf16 op by op.
const GATE_UP_SOURCE: &str = r#"
    const int lane = thread_index_in_simdgroup;
    const int rg = thread_position_in_grid.y;
    const int slot = thread_position_in_grid.z;
    constexpr int G = H / 64;
    constexpr int WPR = H / 8;
    constexpr int BLOCKS = H / 512;
    const bool routed = slot < K;
    /*EXPERT*/

    float g_acc[R];
    float u_acc[R];
    for (int r = 0; r < R; ++r) {
        g_acc[r] = 0.0f;
        u_acc[r] = 0.0f;
    }
    for (int blk = 0; blk < BLOCKS; ++blk) {
        float x_thread[16];
        const device float* xl = x + blk * 512 + lane * 16;
        const float sum = routed ? load_vector_bf16(xl, x_thread) : load_vector_f32(xl, x_thread);
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
            const int scale_at = blk * 8 + lane / 4;
            const device uint16_t* pg = (const device uint16_t*)(wg + blk * 64 + lane * 2);
            const device uint16_t* pu = (const device uint16_t*)(wu + blk * 64 + lane * 2);
            const float s_g = sg[scale_at];
            const float b_g = bg[scale_at];
            const float s_u = su[scale_at];
            const float b_u = bu[scale_at];
            g_acc[r] += qdot4(pg, x_thread, s_g, b_g, sum);
            u_acc[r] += qdot4(pu, x_thread, s_u, b_u, sum);
        }
    }
    for (int r = 0; r < R; ++r) {
        const float g = simd_sum(g_acc[r]);
        const float u = simd_sum(u_acc[r]);
        if (lane == 0) {
            float a;
            if (routed) {
                const float gr = RB(g);
                const float ur = RB(u);
                a = RB(RB(gr * sigmoid_bf16(gr)) * ur);
            } else {
                a = (g * sigmoid_f32(g)) * u;
            }
            act[slot * I + rg * R + r] = a;
        }
    }
"#;

/// Where [`GATE_UP_SOURCE`] gets a slot's expert from: the routing the caller already has.
const EXPERT_FROM_INDICES: &str = "const uint e = routed ? inds[slot] : 0u;";

/// Where it gets it from when the router's select is folded in: every routed simdgroup repeats the
/// composed router's select over the router logits for its own slot, and the simdgroup of slot 0
/// and row group 0 runs it whole and stores the routing the down kernel reads. No kernel sits
/// between the router logits and the experts.
///
/// The select is MLX's, op for op. Softmax: each of `E / 4` threads of its kernel takes four
/// consecutive logits (a lane here plays thread `lane + 32 g` of simdgroup `g`), sums their
/// `fast::exp` in order, then the simdgroup sums (`simd_sum`), then the simdgroups' sums are
/// `simd_sum`med, and each probability is `exp * (1 / sum)`. `argpartition` is a full ascending
/// sort on Metal, so the experts come in ascending order of probability (equal probabilities in
/// index order, hence the highest index is the top one): slot `s` is the `(7 - s)`-th largest. The
/// eight probabilities are added in that order, starting from the smallest, for the renormalising
/// sum, and each is divided by it.
const EXPERT_FROM_LOGITS: &str = r#"
    uint e = 0u;
    {
        constexpr int GROUPS = E / 128;
        const bool writer = slot == 0 && rg == 0;
        const int ranks = writer ? 8 : (routed ? 8 - slot : 0);
        if (ranks > 0) {
            float p[GROUPS][4];
            float top = -INFINITY;
            for (int g = 0; g < GROUPS; ++g) {
                for (int i = 0; i < 4; ++i) {
                    p[g][i] = logits[(lane + 32 * g) * 4 + i];
                    top = (top < p[g][i]) ? p[g][i] : top;
                }
            }
            top = simd_max(top);
            float group_sum[GROUPS];
            for (int g = 0; g < GROUPS; ++g) {
                float partial = 0.0f;
                for (int i = 0; i < 4; ++i) {
                    p[g][i] = metal::fast::exp(p[g][i] - top);
                    partial += p[g][i];
                }
                group_sum[g] = simd_sum(partial);
            }
            float sum_in = 0.0f;
            for (int g = 0; g < GROUPS; ++g) {
                if (lane == g) {
                    sum_in = group_sum[g];
                }
            }
            const float inv = 1 / simd_sum(sum_in);
            for (int g = 0; g < GROUPS; ++g) {
                for (int i = 0; i < 4; ++i) {
                    p[g][i] = p[g][i] * inv;
                }
            }

            float largest[8];
            for (int r = 0; r < 8; ++r) {
                largest[r] = 0.0f;
                if (r < ranks) {
                    float best = -INFINITY;
                    uint at = 0u;
                    for (int g = 0; g < GROUPS; ++g) {
                        for (int i = 0; i < 4; ++i) {
                            if (p[g][i] >= best) {
                                best = p[g][i];
                                at = (uint)((lane + 32 * g) * 4 + i);
                            }
                        }
                    }
                    const float top_p = simd_max(best);
                    const uint top_i = simd_max(best == top_p ? at : 0u);
                    for (int g = 0; g < GROUPS; ++g) {
                        for (int i = 0; i < 4; ++i) {
                            if ((uint)((lane + 32 * g) * 4 + i) == top_i) {
                                p[g][i] = -INFINITY;
                            }
                        }
                    }
                    largest[r] = top_p;
                    if (r == 7 - slot) {
                        e = top_i;
                    }
                    if (writer && lane == 0) {
                        inds[7 - r] = top_i;
                    }
                }
            }
            if (writer && lane == 0) {
                float denom = 0.0f;
                for (int j = 0; j < 8; ++j) {
                    denom = largest[7 - j] + denom;
                }
                for (int j = 0; j < 8; ++j) {
                    scores[j] = largest[7 - j] / denom;
                }
                sscore[0] = sigmoid_f32(logits[E]);
            }
        }
    }
"#;

/// The down projections and the combine: a threadgroup of nine simdgroups, one per slot (the eight
/// routed experts, then the shared one), computes `R` output rows of every slot at once; simdgroup
/// 0 then adds them up as the composed block does. It casts each expert's output to f32 (after
/// the bf16 rounding of the gather), multiplies it by its score and sums the eight products in
/// order (MLX's col_reduce_small: product 0, then each next one added to the running sum), then
/// adds the shared expert's gated output. One simdgroup walking the nine slots in turn added the
/// same way, but chained nine matvecs in one simdgroup: 0.2 ms a token slower on the 35B-A3B
/// decode.
const DOWN_COMBINE_SOURCE: &str = r#"
    const int lane = thread_index_in_simdgroup;
    const int s = simdgroup_index_in_threadgroup;
    const int rg = threadgroup_position_in_grid.y;
    constexpr int G = I / 64;
    constexpr int WPR = I / 8;
    constexpr int BLOCKS = I / 512;
    threadgroup float part[(K + 1) * R];

    const bool routed = s < K;
    const uint e = routed ? inds[s] : 0u;
    float acc[R];
    for (int r = 0; r < R; ++r) {
        acc[r] = 0.0f;
    }
    for (int blk = 0; blk < BLOCKS; ++blk) {
        float x_thread[16];
        const device float* xl = act + s * I + blk * 512 + lane * 16;
        const float sum = routed ? load_vector_bf16(xl, x_thread) : load_vector_f32(xl, x_thread);
        for (int r = 0; r < R; ++r) {
            const int row = rg * R + r;
            const size_t rr = (size_t)e * H + row;
            auto wp = routed ? (w + rr * WPR) : (dw + (size_t)row * WPR);
            auto sp = routed ? (sc + rr * G) : (ds + (size_t)row * G);
            auto bp = routed ? (bi + rr * G) : (db + (size_t)row * G);
            const int scale_at = blk * 8 + lane / 4;
            const device uint16_t* pw = (const device uint16_t*)(wp + blk * 64 + lane * 2);
            const float sc_v = sp[scale_at];
            const float bi_v = bp[scale_at];
            acc[r] += qdot4(pw, x_thread, sc_v, bi_v, sum);
        }
    }
    for (int r = 0; r < R; ++r) {
        const float y = simd_sum(acc[r]);
        if (lane == 0) {
            part[s * R + r] = routed ? RB(y) : y;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (s == 0 && lane < R) {
        float total = 0.0f;
        for (int j = 0; j < K; ++j) {
            const float product = part[j * R + lane] * scores[j];
            total = (j == 0) ? product : (product + total);
        }
        const float gated = sscore[0] * part[K * R + lane];
        total = total + gated;
        out[rg * R + lane] = total;
    }
"#;

/// One simdgroup per gate row: rows `0..E` are the experts' router logits, row `E` the shared
/// expert's gate. 8-bit affine group 64 over an f32 activation, in the order MLX's matvec adds: the
/// experts' rows (256 of them or more, a multiple of 8) run as `qmv_fast`, eight values a lane in
/// blocks of 256; the shared gate has one row, which MLX runs as `qmv`, four values a lane in
/// blocks of 128.
const ROUTER_LOGITS_SOURCE: &str = r#"
    const int lane = thread_index_in_simdgroup;
    const int row = thread_position_in_grid.y;
    constexpr int G = H / 64;
    float result = 0.0f;
    if (row < E) {
        constexpr int BLOCKS = H / 256;
        const device uint8_t* wr = (const device uint8_t*)gw + (size_t)row * H;
        for (int blk = 0; blk < BLOCKS; ++blk) {
            float x_thread[8];
            const float sum = load_vector8<8>(x + blk * 256 + lane * 8, x_thread);
            const int scale_at = row * G + blk * 4 + lane / 8;
            const float s = gs[scale_at];
            const float b = gb[scale_at];
            result += qdot8<8>(wr + blk * 256 + lane * 8, x_thread, s, b, sum);
        }
    } else {
        constexpr int BLOCKS = H / 128;
        const device uint8_t* wr = (const device uint8_t*)sw;
        for (int blk = 0; blk < BLOCKS; ++blk) {
            float x_thread[4];
            const float sum = load_vector8<4>(x + blk * 128 + lane * 4, x_thread);
            const int scale_at = blk * 2 + lane / 16;
            const float s = ss[scale_at];
            const float b = sb[scale_at];
            result += qdot8<4>(wr + blk * 128 + lane * 4, x_thread, s, b, sum);
        }
    }
    result = simd_sum(result);
    if (lane == 0) {
        logits[row] = result;
    }
"#;

/// PROBE (timing only, NOT correct — `LUMEN_PROBE_ABLATE=moe_logits,moe_select,moe_gateup,moe_down`,
/// any subset): empties one piece of the MoE decode kernels while every launch, and every
/// dependency between launches, stays, so the drop in ms/token against an unablated run is that
/// piece's work in the pipeline. `moe_logits`: the router logits kernel writes zeros. `moe_select`:
/// the gate/up kernel takes experts 0..7 instead of running the folded select (it still binds the
/// logits, so it still waits for them). `moe_gateup`, `moe_down`: the matvec loops of the gate/up
/// or down kernel run no blocks.
#[derive(Clone, Copy, Default)]
struct Probe {
    logits: bool,
    select: bool,
    gateup: bool,
    down: bool,
}

fn probe() -> Probe {
    let v = std::env::var("LUMEN_PROBE_ABLATE").unwrap_or_default();
    let has = |k: &str| v.split(',').any(|p| p.trim() == k);
    Probe {
        logits: has("moe_logits"),
        select: has("moe_select"),
        gateup: has("moe_gateup"),
        down: has("moe_down"),
    }
}

/// The matvec block loop of the expert kernels, and the same loop running no blocks (probe).
const MATVEC_LOOP: &str = "for (int blk = 0; blk < BLOCKS; ++blk) {";
const NO_MATVEC_LOOP: &str = "for (int blk = 0; blk < 0; ++blk) {";

/// `moe_select` probe: experts 0..7 with equal scores, written where the select would write them.
const EXPERT_FIXED: &str = r#"
    const uint e = routed ? (uint)slot : 0u;
    if (slot == 0 && rg == 0 && lane < 8) {
        inds[lane] = (uint)lane;
        scores[lane] = 0.125f;
        if (lane == 0) {
            sscore[0] = 0.5f;
        }
    }
"#;

/// `moe_logits` probe: the router logits kernel's launch with none of its work.
const ROUTER_LOGITS_ZERO: &str = r#"
    if (thread_index_in_simdgroup == 0) {
        logits[thread_position_in_grid.y] = 0.0f;
    }
"#;

struct Kernels {
    gate_up: MetalKernel,
    gate_up_routed: MetalKernel,
    down: MetalKernel,
    router_logits: MetalKernel,
}

impl Kernels {
    fn new() -> Result<Self> {
        let probe = probe();
        let matvec = |source: String, empty: bool| {
            if empty {
                source.replace(MATVEC_LOOP, NO_MATVEC_LOOP)
            } else {
                source
            }
        };
        let routed_expert = if probe.select {
            EXPERT_FIXED
        } else {
            EXPERT_FROM_LOGITS
        };
        let gate_up_header = format!("{QMV_BF16}{QMV_F32}{SIGMOID_HEADER}");
        let down_header = format!("{QMV_BF16}{QMV_F32}");
        // Each kernel is constructed with its name as a literal: `cargo xtask test` reads the
        // names off `MetalKernel::new` calls to check that every kernel ran in its Metal 3.1 pass.
        Ok(Self {
            gate_up: MetalKernel::new_with_header(
                "moe_decode_gate_up_swiglu",
                &[
                    "x", "w", "sc", "bi", "gw", "gs", "gb", "uw", "us", "ub", "inds",
                ],
                &["act"],
                &gate_up_header,
                &matvec(
                    GATE_UP_SOURCE.replace("/*EXPERT*/", EXPERT_FROM_INDICES),
                    probe.gateup,
                ),
                /* ensure_row_contiguous */ true,
                /* atomic_outputs */ false,
            )?,
            gate_up_routed: MetalKernel::new_with_header(
                "moe_decode_gate_up_swiglu_routed",
                &[
                    "x", "w", "sc", "bi", "gw", "gs", "gb", "uw", "us", "ub", "logits",
                ],
                &["act", "inds", "scores", "sscore"],
                &gate_up_header,
                &matvec(
                    GATE_UP_SOURCE.replace("/*EXPERT*/", routed_expert),
                    probe.gateup,
                ),
                /* ensure_row_contiguous */ true,
                /* atomic_outputs */ false,
            )?,
            down: MetalKernel::new_with_header(
                "moe_decode_down_combine",
                &[
                    "act", "w", "sc", "bi", "dw", "ds", "db", "inds", "scores", "sscore",
                ],
                &["out"],
                &down_header,
                &matvec(DOWN_COMBINE_SOURCE.to_owned(), probe.down),
                /* ensure_row_contiguous */ true,
                /* atomic_outputs */ false,
            )?,
            router_logits: MetalKernel::new_with_header(
                "moe_decode_router_logits",
                &["x", "gw", "gs", "gb", "sw", "ss", "sb"],
                &["logits"],
                QMV_8BIT,
                if probe.logits {
                    ROUTER_LOGITS_ZERO
                } else {
                    ROUTER_LOGITS_SOURCE
                },
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
    experts_in_scale_dtype: bool,
) -> Option<Dims> {
    let [1, 1, hidden] = *x.shape() else {
        return None;
    };
    // The kernels reproduce the composed block bit for bit, which fixes what that block does: an
    // f32 activation, cast to bf16 for the routed experts (whose bf16 scales it follows) and left
    // in f32 for the shared expert.
    if x.dtype() != Dtype::Float32 || !experts_in_scale_dtype {
        return None;
    }
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
    // MLX runs a one-row matvec as `qmv_fast`, whose lane partition the kernels copy, only when the
    // input length is a multiple of 512 (4-bit weights).
    if hidden % 512 != 0 || inter % 512 != 0 || inter % ROWS_GATE_UP != 0 || hidden % ROWS_DOWN != 0
    {
        return None;
    }
    let shape_is = |a: &Array, want: &[i32]| a.shape() == want;
    let bf16 = |d: Dtype| d == Dtype::Bfloat16;
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
    // must share a type; the routed one is bf16, which the composed block's cast follows.
    let gate_up_dtype = gate_up.scales.dtype();
    let down_dtype = glu.down_proj.scales.dtype();
    let dtypes_ok = bf16(gate_up_dtype)
        && bf16(down_dtype)
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

/// One-expert views of a layer's stacked routed-expert tensors, for the kernels to bind.
///
/// MLX commits the open command buffer once the inputs bound since the last commit add up to more
/// than ~50 M elements (`CommandEncoder::needs_commit` adds each buffer's `data_size()`), and a
/// kernel bound to a whole expert stack crosses that by itself: a commit per MoE layer, about 43
/// command buffers a token on the 35B-A3B. MLX also lets the host run at most 10 command buffers
/// ahead of the GPU (`MAX_ACTIVE_TASKS`), so small buffers cap how far ahead the next step can be
/// built; building it took longer than that lead, and the GPU went idle at every step boundary. A
/// view of expert 0 starts at the same offset of the same buffer, so a kernel indexes every expert
/// through it exactly as through the stack, and it counts as one expert.
pub(crate) struct ExpertViews {
    /// Gate/up weight, scales and biases.
    gate_up: [Array; 3],
    /// Down weight, scales and biases.
    down: [Array; 3],
}

impl ExpertViews {
    /// The views of the `gate_up` and `down` stacks; `None` when the experts carry no biases, which
    /// the kernels need.
    ///
    /// Left unevaluated: the first step that binds a view evaluates it once, as a slice that shares
    /// the stack's buffer. Evaluating it here would load the down stack while the loader still holds
    /// the unstacked gate and up tensors, which raised the 35B-A3B's load peak by several GB (3.5 GB
    /// swapped out, load 4.7 -> 14 s on a 36 GB Mac).
    pub(crate) fn new(
        gate_up: &SwitchExpertWeights<'_>,
        down: &SwitchExpertWeights<'_>,
    ) -> Result<Option<Self>> {
        let (Some(gate_up_biases), Some(down_biases)) = (gate_up.biases, down.biases) else {
            return Ok(None);
        };
        let first = |a: &Array| -> Result<Array> {
            a.try_index(0..1)
                .context("moe decode: view of the first expert")
        };
        Ok(Some(Self {
            gate_up: [
                first(gate_up.weight)?,
                first(gate_up.scales)?,
                first(gate_up_biases)?,
            ],
            down: [
                first(down.weight)?,
                first(down.scales)?,
                first(down_biases)?,
            ],
        }))
    }
}

/// What the kernels bind for one stacked projection: its views when the layer has them, else the
/// stack itself.
fn bound<'a>(
    views: Option<&'a [Array; 3]>,
    stack: &SwitchExpertWeights<'a>,
    what: &str,
) -> Result<[&'a Array; 3]> {
    Ok(match views {
        Some([weight, scales, biases]) => [weight, scales, biases],
        None => [stack.weight, stack.scales, need(stack.biases, what)?],
    })
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
    let [w, sc, bi] = bound(glu.decode_views.map(|v| &v.gate_up), gate_up, "gate/up")?;
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
                    w,
                    sc,
                    bi,
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

/// [`gate_up_swiglu`] with the router's select folded in: takes the `[experts + 1]` logits of
/// [`route_logits`] instead of the chosen experts, and also returns the routing it chose (the
/// experts' ids `[8]` u32, their renormalised probabilities `[8]` f32, the shared sigmoid gate
/// `[1]` f32) for the down kernel.
fn gate_up_swiglu_routed(
    x: &Array,
    logits: &Array,
    experts: i32,
    glu: &SwitchGluWeights<'_>,
    shared: &SharedMlpWeights<'_>,
    dims: Dims,
) -> Result<(Array, Array, Array, Array)> {
    let GateUpWeights::Fused(gate_up) = &glu.gate_up else {
        return Err(anyhow!(
            "moe decode kernels need the stacked gate/up experts"
        ));
    };
    let Dims { hidden, inter } = dims;
    let slots = TOP_K + 1;
    let [w, sc, bi] = bound(glu.decode_views.map(|v| &v.gate_up), gate_up, "gate/up")?;
    with_kernels(|k| {
        let config = MetalKernelConfig::new();
        config.add_template_arg_int("H", hidden)?;
        config.add_template_arg_int("I", inter)?;
        config.add_template_arg_int("K", TOP_K)?;
        config.add_template_arg_int("R", ROWS_GATE_UP)?;
        config.add_template_arg_int("E", experts)?;
        config.set_grid(32, inter / ROWS_GATE_UP, slots)?;
        config.set_thread_group(32, ROWS_PER_GROUP, 1)?;
        config.add_output_arg(&[slots * inter], Dtype::Float32)?;
        config.add_output_arg(&[TOP_K], Dtype::Uint32)?;
        config.add_output_arg(&[TOP_K], Dtype::Float32)?;
        config.add_output_arg(&[1], Dtype::Float32)?;
        let mut out = k
            .gate_up_routed
            .apply(
                &[
                    x,
                    w,
                    sc,
                    bi,
                    shared.gate_proj_w,
                    shared.gate_proj_s,
                    need(shared.gate_proj_b, "shared gate")?,
                    shared.up_proj_w,
                    shared.up_proj_s,
                    need(shared.up_proj_b, "shared up")?,
                    logits,
                ],
                &config,
                4,
            )
            .context("moe decode: gate/up/swiglu kernel with the router select failed")?;
        let sscore = out.pop().ok_or_else(|| anyhow!("no shared score"))?;
        let scores = out.pop().ok_or_else(|| anyhow!("no scores"))?;
        let inds = out.pop().ok_or_else(|| anyhow!("no indices"))?;
        let act = out.pop().ok_or_else(|| anyhow!("no activations"))?;
        Ok((act, inds, scores, sscore))
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
    let [w, sc, bi] = bound(glu.decode_views.map(|v| &v.down), &glu.down_proj, "down")?;
    with_kernels(|k| {
        let config = MetalKernelConfig::new();
        config.add_template_arg_int("H", hidden)?;
        config.add_template_arg_int("I", inter)?;
        config.add_template_arg_int("K", TOP_K)?;
        config.add_template_arg_int("R", ROWS_DOWN)?;
        // One threadgroup of nine simdgroups (a slot each) per `ROWS_DOWN` output rows.
        config.set_grid(32, (TOP_K + 1) * (hidden / ROWS_DOWN), 1)?;
        config.set_thread_group(32, TOP_K + 1, 1)?;
        config.add_output_arg(&[hidden], Dtype::Float32)?;
        Ok(k.down
            .apply(
                &[
                    act,
                    w,
                    sc,
                    bi,
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
        && experts % 128 == 0
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

/// Whether [`run_routed`] can replace the composed router of this layer.
pub(crate) fn router_supported(w: &MoeBlockWeights<'_>, dims: Dims) -> bool {
    fused_router::get() && router_gates(w, dims).is_some()
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

/// [`route_logits`], then the experts with the router's select folded into the gate/up kernel:
/// the routed and shared experts' contribution to the block, `[1, 1, hidden]` f32.
pub(crate) fn run_routed(x: &Array, w: &MoeBlockWeights<'_>, dims: Dims) -> Result<Array> {
    Ok(run_routed_with_routing(x, w, dims)?.0)
}

/// [`run_routed`], and the routing the gate/up kernel chose: the experts' ids (`[8]` u32), their
/// renormalised probabilities (`[8]` f32) and the shared expert's sigmoid gate (`[1]` f32).
fn run_routed_with_routing(
    x: &Array,
    w: &MoeBlockWeights<'_>,
    dims: Dims,
) -> Result<(Array, [Array; 3])> {
    let gates = router_gates(w, dims).ok_or_else(|| anyhow!("moe decode: router not supported"))?;
    let logits = route_logits(x, w, dims)?;
    let hidden = dims.hidden;
    let x = mlx_rs::ops::reshape(&x.as_dtype(Dtype::Float32)?, &[hidden])?;
    let (act, inds, scores, shared_score) = gate_up_swiglu_routed(
        &x,
        &logits,
        gates.experts,
        &w.switch_glu,
        &w.shared_expert,
        dims,
    )?;
    let out = down_combine(
        &act,
        &inds,
        &scores,
        &shared_score,
        &w.switch_glu,
        &w.shared_expert,
        dims,
    )?;
    Ok((
        mlx_rs::ops::reshape(&out, &[1, 1, hidden])?,
        [inds, scores, shared_score],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moe_routed_probe::Layer;
    use crate::native_moe::moe_block_forward;

    /// [`super::plan`] for the synthetic layers, whose experts are bf16 behind an f32 activation.
    fn plan(
        x: &Array,
        glu: &SwitchGluWeights<'_>,
        shared: &SharedMlpWeights<'_>,
        top_k: i32,
    ) -> Option<Dims> {
        super::plan(x, glu, shared, top_k, true)
    }

    fn host(a: &Array) -> Vec<f32> {
        let a = a.as_dtype(Dtype::Float32).expect("f32");
        a.eval().expect("eval");
        a.as_slice::<f32>().to_vec()
    }

    /// The decode kernels give the composed block's output (router, routed experts, shared
    /// expert, their sum) for the same weights and activation, bit for bit.
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
            let off = want
                .iter()
                .zip(&got)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                off,
                0,
                "{hidden}x{inter}x{experts} seed {seed}: {off} of {} elements differ",
                want.len()
            );
        }
    }

    #[test]
    fn kernels_match_the_composed_block_on_a_small_layer() {
        // One block of 512 inputs, and 16 experts: the router stays composed.
        kernels_match_the_composed_block(512, 512, 16);
    }

    #[test]
    fn kernels_match_the_composed_block_at_the_a3b_shape() {
        kernels_match_the_composed_block(2048, 512, 256);
    }

    #[test]
    fn kernels_match_the_composed_block_when_the_rows_outnumber_the_lanes() {
        // 4096 hidden = 8 blocks of 512 inputs; 1024 intermediate = 2.
        kernels_match_the_composed_block(4096, 1024, 16);
    }

    /// The kernels read every expert through one-expert views as through the whole stacks: the
    /// block's output is the same in every bit, so a view starts at its stack's buffer (a copy of
    /// expert 0 alone would hand every other expert's reads other memory). Views of the same stacks
    /// with the experts in reverse order change the output, so the views are what the kernels bind.
    #[test]
    fn the_kernels_read_the_experts_through_one_expert_views() {
        use mlx_rs::ops::indexing::take_axis;
        let layer = Layer::build(2048, 512, 256).expect("layer");
        let viewed = layer.view(true, true);
        assert!(
            viewed.switch_glu.decode_views.is_some(),
            "a stacked affine layer binds views"
        );
        let mut stacked = layer.view(true, true);
        stacked.switch_glu.decode_views = None;

        let reverse = |a: &Array| -> Array {
            let n = a.shape()[0];
            let order: Vec<i32> = (0..n).rev().collect();
            let r = take_axis(a, Array::from_slice(&order, &[n]), 0).expect("reverse the experts");
            r.eval().expect("eval");
            r
        };
        let GateUpWeights::Fused(gate_up) = &stacked.switch_glu.gate_up else {
            panic!("the layer is stacked");
        };
        let down = &stacked.switch_glu.down_proj;
        let gate_up = [
            gate_up.weight,
            gate_up.scales,
            gate_up.biases.expect("biases"),
        ]
        .map(reverse);
        let down = [down.weight, down.scales, down.biases.expect("biases")].map(reverse);
        let reversed_views = ExpertViews::new(
            &SwitchExpertWeights {
                weight: &gate_up[0],
                scales: &gate_up[1],
                biases: Some(&gate_up[2]),
            },
            &SwitchExpertWeights {
                weight: &down[0],
                scales: &down[1],
                biases: Some(&down[2]),
            },
        )
        .expect("views")
        .expect("the experts carry biases");
        let mut reversed = layer.view(true, true);
        reversed.switch_glu.decode_views = Some(&reversed_views);

        for seed in 0..4u64 {
            let x = layer.activation(300 + seed).expect("x");
            assert!(
                plan(&x, &viewed.switch_glu, &viewed.shared_expert, viewed.top_k).is_some(),
                "the kernels must take this layer"
            );
            let want = host(&moe_block_forward(&x, &stacked).expect("stacks bound"));
            let got = host(&moe_block_forward(&x, &viewed).expect("views bound"));
            let off = want
                .iter()
                .zip(&got)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                off,
                0,
                "seed {seed}: {off} of {} elements differ",
                want.len()
            );
            let other = host(&moe_block_forward(&x, &reversed).expect("reversed views bound"));
            assert!(
                want.iter()
                    .zip(&other)
                    .any(|(a, b)| a.to_bits() != b.to_bits()),
                "seed {seed}: views of other experts left the output unchanged, so the kernels did not read through the views"
            );
        }
    }

    #[test]
    fn layers_the_kernels_cannot_run_stay_on_the_composed_block() {
        let layer = Layer::build(512, 512, 16).expect("layer");
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
        // The composed block would not cast to bf16 here, or not run the shared expert in f32.
        assert!(
            super::plan(
                &x,
                &stacked.switch_glu,
                &stacked.shared_expert,
                top_k,
                false
            )
            .is_none()
        );
        let bf16 = x.as_dtype(Dtype::Bfloat16).expect("bf16");
        assert!(plan(&bf16, &stacked.switch_glu, &stacked.shared_expert, top_k).is_none());
        // MLX runs these as `qmv`, not `qmv_fast`: an input length that is not a multiple of 512.
        for (hidden, inter) in [(640, 512), (512, 640)] {
            let odd = Layer::build(hidden, inter, 16).expect("layer");
            let v = odd.view(true, true);
            let x = odd.activation(1).expect("x");
            assert!(
                plan(&x, &v.switch_glu, &v.shared_expert, top_k).is_none(),
                "{hidden}x{inter}"
            );
        }
    }

    /// The gate's ranking agrees with the routing it comes from: the same experts on top, in the
    /// gate's order, however many are asked for.
    #[test]
    fn the_gates_ranking_ends_in_the_routed_experts() {
        use crate::native_moe::{moe_ranked, moe_router};

        let layer = Layer::build(2048, 512, 256).expect("layer");
        let w = layer.view(true, true);
        for seed in 0..4u64 {
            let x = layer.activation(700 + seed).expect("x");
            let (inds, _) = moe_router(&x, &w).expect("router");
            let inds = inds.as_dtype(Dtype::Int32).unwrap();
            inds.eval().unwrap();
            let mut routed = inds.as_slice::<i32>().to_vec();
            routed.sort_unstable();

            let wide = moe_ranked(&x, &w, TOP_K + 12)
                .unwrap()
                .as_dtype(Dtype::Int32)
                .unwrap();
            wide.eval().unwrap();
            assert_eq!(wide.size(), (TOP_K + 12) as usize);
            let mut top = wide.as_slice::<i32>()[12..].to_vec();
            top.sort_unstable();
            assert_eq!(top, routed, "seed {seed}");
        }
    }

    /// The kernel's routed activations are the composed SwiGLU's, every one of them: the matvec
    /// follows MLX's lane partition and add order, the activation is rounded to bf16 and summed
    /// four at a time in bf16 as `load_vector` does, and the SwiGLU rounds op by op.
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
        assert_eq!(
            equal,
            total,
            "{} of {total} routed activations differ from the composed block's",
            total - equal
        );
    }

    #[test]
    fn kernels_match_the_composed_block_with_a_512_expert_router() {
        kernels_match_the_composed_block(1024, 512, 512);
    }

    /// The experts' contribution for the composed router's routing, kernels against composed ops:
    /// how many of the block's elements differ in any bit, and the largest relative gap.
    fn expert_bits(hidden: i32, inter: i32, experts: i32, inputs: u64) -> (usize, usize, f64) {
        use crate::native_moe::moe_router;
        use crate::native_quant::quantized_matmul_with_mode;

        let layer = Layer::build(hidden, inter, experts).expect("layer");
        let composed = layer.view(true, false);
        let kernels = layer.view(true, true);
        let GateProj::Affine(shared_gate) = &kernels.shared_gate else {
            panic!("an affine shared gate");
        };
        let (mut differ, mut total, mut worst) = (0usize, 0usize, 0f64);
        for seed in 0..inputs {
            let x = layer.activation(3000 + seed).expect("x");
            let dims = plan(
                &x,
                &kernels.switch_glu,
                &kernels.shared_expert,
                kernels.top_k,
            )
            .expect("plan");
            let (inds, scores) = moe_router(&x, &composed).expect("composed router");
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
            let shared_score = mlx_rs::ops::sigmoid(&logit).expect("sigmoid");
            let got = host(
                &run(
                    &x,
                    &inds,
                    &scores,
                    &shared_score,
                    &kernels.switch_glu,
                    &kernels.shared_expert,
                    dims,
                )
                .expect("kernel experts"),
            );
            let want = host(&moe_block_forward(&x, &composed).expect("composed block"));
            for (a, b) in want.iter().zip(&got) {
                total += 1;
                if a.to_bits() != b.to_bits() {
                    differ += 1;
                    worst = worst.max(f64::from((a - b).abs()) / f64::from(a.abs().max(1e-30)));
                }
            }
        }
        (differ, total, worst)
    }

    #[test]
    fn the_expert_kernels_equal_the_composed_experts_bit_for_bit() {
        // 4 blocks of 512 inputs to the gate/up matvec and one to the down; 2 blocks to the down;
        // 8 and 2 blocks, with 512 experts.
        for (hidden, inter, experts) in [(2048, 512, 256), (1024, 512, 512), (4096, 1024, 32)] {
            let (differ, total, worst) = expert_bits(hidden, inter, experts, 24);
            eprintln!(
                "EXPERT-BITS {hidden}x{inter}x{experts}: {differ}/{total} differ, worst relative gap {worst:.2e}"
            );
            assert_eq!(
                differ, 0,
                "{hidden}x{inter}x{experts}: {differ} of {total} block elements differ from the composed ops (worst relative gap {worst:.2e})"
            );
        }
    }

    /// The router's routing and the whole block with the router in kernels, against the composed
    /// router and block: (routing positions that differ, score or gate bits that differ, block
    /// elements that differ, block elements), with the largest relative gap of the last.
    fn router_bits(layer: &Layer, inputs: u64) -> (usize, usize, usize, usize, f64) {
        use crate::native_moe::moe_router;
        use crate::native_quant::quantized_matmul_with_mode;

        let composed = layer.view(true, false);
        let kernels = layer.view(true, true);
        let GateProj::Affine(shared_gate) = &kernels.shared_gate else {
            panic!("an affine shared gate");
        };
        let (mut routing, mut scored, mut differ, mut total, mut worst) = (0, 0, 0, 0, 0f64);
        for seed in 0..inputs {
            let x = layer.activation(5000 + seed).expect("x");
            let dims = plan(
                &x,
                &kernels.switch_glu,
                &kernels.shared_expert,
                kernels.top_k,
            )
            .expect("plan");
            assert!(
                router_supported(&kernels, dims),
                "the kernels must take this router, or the comparison compares the composed one with itself"
            );
            let (want_inds, want_scores) = moe_router(&x, &composed).expect("composed router");
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
            let want_gate = host(&mlx_rs::ops::sigmoid(&logit).expect("sigmoid"));
            let (got_block, [inds, scores, gate]) =
                run_routed_with_routing(&x, &kernels, dims).expect("kernel router");

            let ids = |a: &Array| {
                let a = a.as_dtype(Dtype::Float32).expect("f32");
                host(&a)
            };
            routing += ids(&want_inds)
                .iter()
                .zip(ids(&inds))
                .filter(|(a, b)| **a != *b)
                .count();
            scored += host(&want_scores)
                .iter()
                .zip(host(&scores))
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            scored += usize::from(want_gate[0].to_bits() != host(&gate)[0].to_bits());

            let want = host(&moe_block_forward(&x, &composed).expect("composed block"));
            for (a, b) in want.iter().zip(host(&got_block)) {
                total += 1;
                if a.to_bits() != b.to_bits() {
                    differ += 1;
                    worst = worst.max(f64::from((a - b).abs()) / f64::from(a.abs().max(1e-30)));
                }
            }
        }
        (routing, scored, differ, total, worst)
    }

    fn assert_router_is_bit_exact(label: &str, layer: &Layer, inputs: u64) {
        // The comparison is of the kernel router, so it holds the flag on when the equivalence
        // matrix has flipped it off.
        let (routing, scored, differ, total, worst) =
            fused_router::with(true, || router_bits(layer, inputs));
        eprintln!(
            "ROUTER-BITS {label}: routing positions off {routing}, score/gate bits off {scored}, block {differ}/{total} differ (worst {worst:.2e})"
        );
        assert_eq!(
            (routing, scored, differ),
            (0, 0, 0),
            "{label}: routing positions off {routing}, score/gate bits off {scored}, block elements off {differ} of {total}"
        );
    }

    #[test]
    fn the_router_and_block_equal_the_composed_ones_bit_for_bit() {
        // 256 experts: two simdgroups of MLX's softmax; 512: four.
        assert_router_is_bit_exact(
            "2048x512x256",
            &Layer::build(2048, 512, 256).expect("layer"),
            60,
        );
        assert_router_is_bit_exact(
            "1024x512x512",
            &Layer::build(1024, 512, 512).expect("layer"),
            40,
        );
    }

    #[test]
    fn equal_probabilities_are_ordered_as_the_composed_sort_orders_them() {
        // Runs of four identical gate rows: the top eight are two runs of four equal probabilities.
        let tied = Layer::build(2048, 512, 256)
            .expect("layer")
            .with_tied_router(4)
            .expect("tied router");
        assert_router_is_bit_exact("tied 2048x512x256", &tied, 30);
    }

    #[test]
    fn a_router_over_a_few_experts_stays_on_the_composed_ops() {
        let layer = Layer::build(512, 512, 16).expect("layer");
        let w = layer.view(true, true);
        let x = layer.activation(3).expect("x");
        let dims = plan(&x, &w.switch_glu, &w.shared_expert, w.top_k).expect("experts supported");
        assert!(
            !router_supported(&w, dims),
            "a 16-expert router is not a multiple of 32"
        );
    }
}
