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
use crate::native_moe::{GateUpWeights, SharedMlpWeights, SwitchGluWeights};
use crate::native_quant::MODE_AFFINE;

/// Routed experts per token. The kernels are written for Qwen's top-8 routing.
const TOP_K: i32 = 8;
/// Output rows per threadgroup (the simdgroup count of a threadgroup).
const ROWS_PER_GROUP: i32 = 4;

const GATE_UP_SOURCE: &str = r#"
    #define RB(v) static_cast<float>(static_cast<bfloat16_t>(v))
    // MLX sums the activation four at a time in the activation's own type (`sum += x[i] + x[i + 1]
    // + x[i + 2] + x[i + 3]`), so bf16 activations round after every add; f32 ones do not.
    #define QSUM(a, b, c, d) (routed ? RB(RB(RB((a) + (b)) + (c)) + (d)) : (((a) + (b)) + (c)) + (d))
    const int lane = thread_index_in_simdgroup;
    const int row = thread_position_in_grid.y;
    const int slot = thread_position_in_grid.z;
    constexpr int G = H / 64;
    constexpr int WPR = H / 8;
    const bool routed = slot < K;
    const uint e = routed ? inds[slot] : 0u;
    const size_t rg = (size_t)e * (2 * I) + row;
    const size_t ru = rg + I;

    auto wg = routed ? (w + rg * WPR) : (gw + (size_t)row * WPR);
    auto wu = routed ? (w + ru * WPR) : (uw + (size_t)row * WPR);
    auto sg = routed ? (sc + rg * G) : (gs + (size_t)row * G);
    auto su = routed ? (sc + ru * G) : (us + (size_t)row * G);
    auto bg = routed ? (bi + rg * G) : (gb + (size_t)row * G);
    auto bu = routed ? (bi + ru * G) : (ub + (size_t)row * G);

    float g_part = 0.0f;
    float u_part = 0.0f;
    for (int grp = lane; grp < G; grp += 32) {
        float ag = 0.0f;
        float au = 0.0f;
        float xs = 0.0f;
        for (int j = 0; j < 8; ++j) {
            const uint32_t pg = wg[grp * 8 + j];
            const uint32_t pu = wu[grp * 8 + j];
            float xq[8];
            for (int i = 0; i < 8; ++i) {
                const float raw = x[grp * 64 + j * 8 + i];
                xq[i] = routed ? RB(raw) : raw;
                ag += xq[i] * static_cast<float>((pg >> (4 * i)) & 0xFu);
                au += xq[i] * static_cast<float>((pu >> (4 * i)) & 0xFu);
            }
            xs += QSUM(xq[0], xq[1], xq[2], xq[3]);
            xs += QSUM(xq[4], xq[5], xq[6], xq[7]);
        }
        g_part += static_cast<float>(sg[grp]) * ag + static_cast<float>(bg[grp]) * xs;
        u_part += static_cast<float>(su[grp]) * au + static_cast<float>(bu[grp]) * xs;
    }
    const float g = simd_sum(g_part);
    const float u = simd_sum(u_part);
    if (lane == 0) {
        float a;
        if (routed) {
            const float gr = RB(g);
            const float ur = RB(u);
            // MLX's sigmoid, op by op: y = 1 / (1 + exp(|x|)); x < 0 ? y : 1 - y.
            const float e = RB(metal::exp(metal::abs(gr)));
            const float d = RB(1.0f + e);
            const float y = RB(1.0f / d);
            const float sig = gr < 0.0f ? y : RB(1.0f - y);
            a = RB(RB(gr * sig) * ur);
        } else {
            a = (g / (1.0f + metal::exp(-g))) * u;
        }
        act[slot * I + row] = a;
    }
"#;

const DOWN_COMBINE_SOURCE: &str = r#"
    #define RB(v) static_cast<float>(static_cast<bfloat16_t>(v))
    #define QSUM(a, b, c, d) (routed ? RB(RB(RB((a) + (b)) + (c)) + (d)) : (((a) + (b)) + (c)) + (d))
    const int lane = thread_index_in_simdgroup;
    const int row = thread_position_in_grid.y;
    constexpr int G = I / 64;
    constexpr int WPR = I / 8;
    constexpr int CHUNKS = I / 16;

    float total = 0.0f;
    for (int s = 0; s <= K; ++s) {
        const bool routed = s < K;
        const uint e = routed ? inds[s] : 0u;
        const size_t r = (size_t)e * H + row;
        auto wp = routed ? (w + r * WPR) : (dw + (size_t)row * WPR);
        auto sp = routed ? (sc + r * G) : (ds + (size_t)row * G);
        auto bp = routed ? (bi + r * G) : (db + (size_t)row * G);
        float part = 0.0f;
        for (int c = lane; c < CHUNKS; c += 32) {
            const uint32_t p0 = wp[c * 2];
            const uint32_t p1 = wp[c * 2 + 1];
            float acc = 0.0f;
            float xs = 0.0f;
            float av[16];
            for (int i = 0; i < 8; ++i) {
                av[i] = act[s * I + c * 16 + i];
                av[8 + i] = act[s * I + c * 16 + 8 + i];
                acc += av[i] * static_cast<float>((p0 >> (4 * i)) & 0xFu);
                acc += av[8 + i] * static_cast<float>((p1 >> (4 * i)) & 0xFu);
            }
            xs += QSUM(av[0], av[1], av[2], av[3]);
            xs += QSUM(av[4], av[5], av[6], av[7]);
            xs += QSUM(av[8], av[9], av[10], av[11]);
            xs += QSUM(av[12], av[13], av[14], av[15]);
            part += static_cast<float>(sp[c / 4]) * acc + static_cast<float>(bp[c / 4]) * xs;
        }
        const float y = simd_sum(part);
        total += routed ? scores[s] * RB(y) : sscore[0] * y;
    }
    if (lane == 0) {
        out[row] = total;
    }
"#;

struct Kernels {
    gate_up: MetalKernel,
    down: MetalKernel,
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
    if hidden % 64 != 0 || inter % 16 != 0 || hidden % ROWS_PER_GROUP != 0 {
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
fn gate_up_swiglu(
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
        config.set_grid(32, inter, slots)?;
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
fn down_combine(
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
        config.set_grid(32, hidden, 1)?;
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
}
