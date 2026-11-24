//! The Inkling 2 text model: weights, the layer visit and the forward pass over the loop.
//!
//! Derived from `omlx/patches/inkling2/model.py` of oMLX 0.8.2 (Apache-2.0); the structure
//! follows its `Inkling2TextModel`, `Inkling2Layer`, `Inkling2Attention` and `Inkling2MoE`. The
//! contract is `SPEC-step2-text.md` in the task folder and the parity tests in `tests.rs`.
//!
//! Batch size is 1. Every layer visit owns a cache entry (a KV cache and four convolution
//! states), so the looped core keeps one per pass.

use std::ffi::CStr;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail, ensure};
use mlx_rs::ops::indexing::{TryIndexOp, take_axis};
use mlx_rs::{Array, Dtype};

use super::fused;
use super::mix_kernel;
use super::ops::{
    attn_res_mix, banded_sink_attention, dynamic_conv_scale, mix_messages, next_conv_state, owned,
    route_topk, scalar, softcap, softcapped_swiglu, static_conv_scale, unit_rms,
};
use super::profile;
use super::quant::{QuantMode, QuantTable};
use super::regions::{self, ProjCfg};
use super::route_kernel;
use super::text_config::{AttnDims, TextArgs};
use crate::native_cache::NativeKvCache;
use crate::native_embedding::quantized_embedding_lookup_with_mode;
use crate::native_quant::{
    MODE_AFFINE, MODE_NVFP4, gather_qmm_with_mode, quantized_matmul_with_mode,
};
use crate::qwen3_5_moe::NativeWeights;

/// MLX's NVFP4 tensor scale is the expert's amax; the source scale is `global_scale / (6 * 448)`,
/// applied after the matmul so the packed codes and E4M3 block scales stay as stored.
const NVFP4_DENOMINATOR: f32 = 448.0 * 6.0;

#[derive(Debug, Clone, Copy)]
pub struct LoadOptions {
    /// Round the loop controllers' projection through bf16 as the reference does. Always on in
    /// serving. A float32 parity replica turns it off: there the rounding is a staircase that two
    /// implementations cross at different places and the 48 layer visits amplify (see
    /// `scripts/golden/inkling2_tiny_fixture.py`).
    pub round_loop_mix: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            round_loop_mix: true,
        }
    }
}

// ---- weights ---------------------------------------------------------------------------------

/// The tensors of a checkpoint directory, read by name. Every array handed out is remembered so
/// the loader can evaluate them all on the loading thread (the lazy file-backed arrays are bound
/// to that thread's stream, and the server runs the model on another).
struct Bag {
    weights: NativeWeights,
    used: Vec<Array>,
}

impl Bag {
    fn get(&mut self, name: &str) -> Result<Array> {
        let a = self.weights.require(name)?.clone();
        self.used.push(a.clone());
        Ok(a)
    }

    fn get_shaped(&mut self, name: &str, shape: &[i32]) -> Result<Array> {
        let a = self.get(name)?;
        ensure!(
            a.shape() == shape,
            "{name}: shape {:?}, expected {shape:?}",
            a.shape()
        );
        Ok(a)
    }

    /// `name` widened to float32 (the loop controllers and the attention-residual queries are
    /// float32 in the source checkpoint and stay so). Evaluated here, on the loading thread.
    fn get_f32(&mut self, name: &str, shape: &[i32]) -> Result<Array> {
        let a = self.weights.require(name)?;
        ensure!(
            a.shape() == shape,
            "{name}: shape {:?}, expected {shape:?}",
            a.shape()
        );
        let wide = a.as_dtype(Dtype::Float32)?;
        self.used.push(wide.clone());
        Ok(wide)
    }

    fn materialize(&self) -> Result<()> {
        for batch in self.used.chunks(256) {
            mlx_rs::transforms::eval(batch.iter())?;
        }
        Ok(())
    }
}

fn mode_of(mode: QuantMode) -> &'static CStr {
    match mode {
        QuantMode::Affine => MODE_AFFINE,
        QuantMode::Nvfp4 => MODE_NVFP4,
    }
}

/// An affine-quantized projection `y = x @ W^T`.
struct QLinear {
    weight: Array,
    scales: Array,
    biases: Array,
    group_size: i32,
    bits: i32,
}

impl QLinear {
    fn load(bag: &mut Bag, quant: &QuantTable, module: &str, out: i32, inp: i32) -> Result<Self> {
        let spec = quant.resolve(module);
        ensure!(
            spec.mode == QuantMode::Affine,
            "{module}: projections are affine-quantized, the config says {}",
            spec.mode.name()
        );
        let (bits, group_size) = (spec.bits as i32, spec.group_size as i32);
        ensure!(
            inp % group_size == 0 && (inp * bits) % 32 == 0,
            "{module}: {inp} inputs do not pack at {bits} bits, group {group_size}"
        );
        let groups = inp / group_size;
        Ok(Self {
            weight: bag.get_shaped(&format!("{module}.weight"), &[out, inp * bits / 32])?,
            scales: bag.get_shaped(&format!("{module}.scales"), &[out, groups])?,
            biases: bag.get_shaped(&format!("{module}.biases"), &[out, groups])?,
            group_size,
            bits,
        })
    }

    fn apply(&self, x: &Array) -> Result<Array> {
        quantized_matmul_with_mode(
            x,
            &self.weight,
            &self.scales,
            Some(&self.biases),
            true,
            self.group_size,
            self.bits,
            MODE_AFFINE,
        )
    }
}

/// The 8-bit affine token embedding: rows are looked up, then dequantized.
struct QEmbedding {
    weight: Array,
    scales: Array,
    biases: Array,
    group_size: i32,
    bits: i32,
}

impl QEmbedding {
    fn lookup(&self, ids: &Array) -> Result<Array> {
        quantized_embedding_lookup_with_mode(
            &self.weight,
            &self.scales,
            Some(&self.biases),
            ids,
            self.group_size,
            self.bits,
            MODE_AFFINE,
        )
    }
}

/// A stacked expert projection on an MLX quantized carrier.
struct ExpertProj {
    weight: Array,
    scales: Array,
    /// Affine only.
    biases: Option<Array>,
    /// NVFP4 only: `[experts]` float32.
    global_scale: Option<Array>,
    mode: &'static CStr,
    group_size: i32,
    bits: i32,
}

impl ExpertProj {
    fn load(
        bag: &mut Bag,
        quant: &QuantTable,
        module: &str,
        experts: i32,
        out: i32,
        inp: i32,
    ) -> Result<Self> {
        let spec = quant.resolve(module);
        let (bits, group_size) = (spec.bits as i32, spec.group_size as i32);
        ensure!(
            inp % group_size == 0 && (inp * bits) % 32 == 0,
            "{module}: {inp} inputs do not pack at {bits} bits, group {group_size}"
        );
        let groups = inp / group_size;
        let weight = bag.get_shaped(
            &format!("{module}.weight"),
            &[experts, out, inp * bits / 32],
        )?;
        ensure!(
            weight.dtype() == Dtype::Uint32,
            "{module}.weight is not uint32"
        );
        let scales = bag.get_shaped(&format!("{module}.scales"), &[experts, out, groups])?;
        let (biases, global_scale) = match spec.mode {
            QuantMode::Affine => (
                Some(bag.get_shaped(&format!("{module}.biases"), &[experts, out, groups])?),
                None,
            ),
            QuantMode::Nvfp4 => {
                ensure!(
                    scales.dtype() == Dtype::Uint8,
                    "{module}.scales is not uint8"
                );
                (
                    None,
                    Some(bag.get_f32(&format!("{module}.global_scale"), &[experts])?),
                )
            }
        };
        Ok(Self {
            weight,
            scales,
            biases,
            global_scale,
            mode: mode_of(spec.mode),
            group_size,
            bits,
        })
    }

    /// `global_scale[idx] / (6 * 448)`, shaped like `idx`; `None` off NVFP4.
    fn post_scale(&self, idx: &Array) -> Result<Option<Array>> {
        let Some(global) = &self.global_scale else {
            return Ok(None);
        };
        let picked = take_axis(global, idx, 0)?;
        Ok(Some(mlx_rs::ops::multiply(
            &picked,
            scalar((1.0f64 / f64::from(NVFP4_DENOMINATOR)) as f32),
        )?))
    }

    fn apply(&self, x: &Array, idx: &Array, sorted: bool) -> Result<Array> {
        gather_qmm_with_mode(
            x,
            &self.weight,
            &self.scales,
            self.biases.as_ref(),
            None,
            Some(idx),
            true,
            self.group_size,
            self.bits,
            self.mode,
            sorted,
        )
    }
}

/// A float32 `[Q, hidden]` query table that mixes the depth sources (the attention residual).
struct AttnResQuery {
    query: Array,
    scale: f32,
}

impl AttnResQuery {
    fn load(bag: &mut Bag, name: &str, queries: i32, hidden: i32) -> Result<Self> {
        Ok(Self {
            query: bag.get_f32(name, &[queries, hidden])?,
            scale: (f64::from(hidden).powf(-0.5)) as f32,
        })
    }

    /// The mix of the depth sources: one Metal kernel when `fuse` (decode-sized) and the kernel is
    /// on (`mix_kernel.rs`), compiled when only `fuse` (`fused.rs`), the composed ops otherwise.
    fn mix(&self, blocks: &[Array], partial_n: &Array, fuse: bool) -> Result<Array> {
        let run = match (fuse, mix_kernel::enabled()) {
            (true, true) => mix_kernel::attn_res_mix,
            (true, false) => fused::attn_res_mix,
            (false, _) => attn_res_mix,
        };
        run(blocks, partial_n, &self.query, self.scale)
    }
}

/// The loop controllers of a core layer: message queries and the gain/forget projections for the
/// q, k, v and MLP inputs, in that order.
struct LoopCtl {
    message: AttnResQuery,
    down: Array,
    gain_forget: Array,
}

// ---- attention -------------------------------------------------------------------------------

enum AttnProj {
    /// One fused projection: q | k | v | r | gate | sink, with static short convolutions on k, v.
    Local {
        in_proj: QLinear,
        k_conv: Array,
        v_conv: Array,
    },
    /// q | r | gate | sink from the q input; k and v (and their dynamic conv taps) from their own
    /// attention-residual inputs, plus static taps.
    Global {
        q_proj: QLinear,
        k_proj: QLinear,
        v_proj: QLinear,
        k_static: Array,
        v_static: Array,
    },
}

struct Attention {
    dims: AttnDims,
    proj: AttnProj,
    o_proj: QLinear,
    q_scale: Array,
    r_scale: Array,
    rel_proj: Array,
    /// `heads * head_dim`, `kv_heads * head_dim`, `heads * d_rel`.
    dq: i32,
    dkv: i32,
    d_rel: i32,
    /// Where the fused projection splits.
    cuts: Vec<i32>,
    scale: f32,
    qk_clip: f32,
    rel_clip: f32,
    /// Global layers scale q and the relative state for long positions.
    log_floor: Option<f32>,
    log_alpha: f32,
    sconv_scale: f32,
    dconv_scale: f32,
}

/// What one attention call needs after the projections.
struct Projected {
    q: Array,
    k: Array,
    v: Array,
    rel: Array,
    gate: Array,
    sink: Array,
    /// The padded convolution inputs; their last `taps - 1` rows are the next states.
    k_window: Array,
    v_window: Array,
}

impl Attention {
    fn load(
        bag: &mut Bag,
        quant: &QuantTable,
        args: &TextArgs,
        layer: usize,
        sliding: bool,
    ) -> Result<Self> {
        let dims = args.dims(sliding);
        let (h, nh, nkv, hd) = (
            args.hidden_size as i32,
            dims.heads as i32,
            dims.kv_heads as i32,
            dims.head_dim as i32,
        );
        let (d_rel, taps, rel_extent) = (
            args.d_rel as i32,
            args.sconv_kernel_size as i32,
            dims.rel_extent as i32,
        );
        let (dq, dkv, dr) = (nh * hd, nkv * hd, nh * d_rel);
        let pre = format!("model.layers.{layer}.attn");
        let (proj, cuts) = if sliding {
            let rows = dq + 2 * dkv + dr + dq + nh;
            (
                AttnProj::Local {
                    in_proj: QLinear::load(bag, quant, &format!("{pre}.in_proj"), rows, h)?,
                    k_conv: bag.get_shaped(&format!("{pre}.k_conv"), &[dkv, taps])?,
                    v_conv: bag.get_shaped(&format!("{pre}.v_conv"), &[dkv, taps])?,
                },
                vec![
                    dq,
                    dq + dkv,
                    dq + 2 * dkv,
                    dq + 2 * dkv + dr,
                    dq + 2 * dkv + dr + dq,
                ],
            )
        } else {
            let rows = dq + dr + dq + nh;
            let kv_rows = dkv + nkv * taps;
            (
                AttnProj::Global {
                    q_proj: QLinear::load(bag, quant, &format!("{pre}.q_proj"), rows, h)?,
                    k_proj: QLinear::load(bag, quant, &format!("{pre}.k_proj"), kv_rows, h)?,
                    v_proj: QLinear::load(bag, quant, &format!("{pre}.v_proj"), kv_rows, h)?,
                    k_static: bag.get_shaped(&format!("{pre}.k_static"), &[taps, dkv])?,
                    v_static: bag.get_shaped(&format!("{pre}.v_static"), &[taps, dkv])?,
                },
                vec![dq, dq + dr, dq + dr + dq],
            )
        };
        Ok(Self {
            dims,
            proj,
            o_proj: QLinear::load(bag, quant, &format!("{pre}.o_proj"), h, dq)?,
            q_scale: bag.get_shaped(&format!("{pre}.q_scale"), &[nh, hd])?,
            r_scale: bag.get_shaped(&format!("{pre}.r_scale"), &[nh, d_rel])?,
            rel_proj: bag.get_shaped(&format!("{pre}.rel_proj"), &[d_rel, rel_extent])?,
            dq,
            dkv,
            d_rel,
            cuts,
            scale: (f64::from(hd).powf(-0.5)) as f32,
            qk_clip: args.qk_l2_clip_threshold,
            rel_clip: args.rel_l2_clip_threshold,
            log_floor: (!sliding).then_some(args.log_scaling_n_floor as f32),
            log_alpha: args.log_scaling_alpha,
            sconv_scale: static_conv_scale(args.sconv_kernel_size),
            dconv_scale: dynamic_conv_scale(args.sconv_kernel_size),
        })
    }

    /// `log_tau`: `1 + alpha * ln(max((pos + 1) / floor, 1))` for the positions of this chunk;
    /// `None` on a local layer.
    fn log_tau(&self, offset: usize, length: usize) -> Result<Option<Array>> {
        let Some(floor) = self.log_floor else {
            return Ok(None);
        };
        let tau: Vec<f32> = (0..length)
            .map(|i| 1.0 + self.log_alpha * (((offset + i + 1) as f32) / floor).max(1.0).ln())
            .collect();
        Ok(Some(Array::from_slice(&tau, &[1, length as i32, 1, 1])))
    }

    /// Projections, short convolutions and clips. `xk` and `xv` are the separate k and v inputs
    /// of a global layer or of a local layer in a later loop pass; `None` on a local layer that
    /// feeds one input to the fused projection. `fuse` runs the region after the matmuls compiled.
    fn project(
        &self,
        xq: &Array,
        xk: Option<&Array>,
        xv: Option<&Array>,
        k_state: &Array,
        v_state: &Array,
        tau: Option<&Array>,
        fuse: bool,
    ) -> Result<Projected> {
        let (out, gate, sink) = match &self.proj {
            AttnProj::Local {
                in_proj,
                k_conv,
                v_conv,
            } => {
                let (q, k, v, r, gate, sink) = match (xk, xv) {
                    (None, None) => {
                        let mut p =
                            mlx_rs::ops::split_sections(&in_proj.apply(xq)?, &self.cuts, -1)?
                                .into_iter();
                        let mut next = || {
                            p.next()
                                .ok_or_else(|| anyhow!("split yielded too few parts"))
                        };
                        (next()?, next()?, next()?, next()?, next()?, next()?)
                    }
                    (Some(xk), Some(xv)) => {
                        // Looped passes give q, k and v their own inputs; one stacked projection
                        // serves all three.
                        let stacked = mlx_rs::ops::concatenate_axis(&[xq, xk, xv], 0)?;
                        let mut thirds =
                            mlx_rs::ops::split(&in_proj.apply(&stacked)?, 3, 0)?.into_iter();
                        let (pq, pk, pv) = (
                            thirds
                                .next()
                                .ok_or_else(|| anyhow!("split yielded too few parts"))?,
                            thirds
                                .next()
                                .ok_or_else(|| anyhow!("split yielded too few parts"))?,
                            thirds
                                .next()
                                .ok_or_else(|| anyhow!("split yielded too few parts"))?,
                        );
                        let mut p = mlx_rs::ops::split_sections(&pq, &self.cuts, -1)?.into_iter();
                        let mut next = || {
                            p.next()
                                .ok_or_else(|| anyhow!("split yielded too few parts"))
                        };
                        let (q, _, _, r, gate, sink) =
                            (next()?, next()?, next()?, next()?, next()?, next()?);
                        let k = pk.try_index((.., .., self.cuts[0]..self.cuts[1]))?;
                        let v = pv.try_index((.., .., self.cuts[1]..self.cuts[2]))?;
                        (q, k, v, r, gate, sink)
                    }
                    _ => bail!("a local layer takes either one input or all three"),
                };
                let lead = [&q, &k, &v, &r, k_state, v_state, k_conv, v_conv];
                (self.region(&lead, tau, fuse)?, gate, sink)
            }
            AttnProj::Global {
                q_proj,
                k_proj,
                v_proj,
                k_static,
                v_static,
            } => {
                let (xk, xv) = xk
                    .zip(xv)
                    .ok_or_else(|| anyhow!("a global layer needs separate k and v inputs"))?;
                let mut p =
                    mlx_rs::ops::split_sections(&q_proj.apply(xq)?, &self.cuts, -1)?.into_iter();
                let mut next = || {
                    p.next()
                        .ok_or_else(|| anyhow!("split yielded too few parts"))
                };
                let (q, r, gate, sink) = (next()?, next()?, next()?, next()?);
                let kp = k_proj.apply(xk)?;
                let vp = v_proj.apply(xv)?;
                let lead = [&q, &r, &kp, &vp, k_state, v_state, k_static, v_static];
                (self.region(&lead, tau, fuse)?, gate, sink)
            }
        };
        let [q, k, v, rel, k_window, v_window] = out;
        Ok(Projected {
            q,
            k,
            v,
            rel,
            gate,
            sink,
            k_window,
            v_window,
        })
    }

    /// The projection region of this layer kind (`regions.rs`), compiled when `fuse` (`fused.rs`):
    /// `lead` are the pieces it starts from (q, k, v, r with their states and taps, or the global
    /// layer's), after which come the layer's scales and relative projection and the position scaling.
    fn region(&self, lead: &[&Array], tau: Option<&Array>, fuse: bool) -> Result<[Array; 6]> {
        let local = matches!(self.proj, AttnProj::Local { .. });
        let cfg = ProjCfg {
            heads: self.dims.heads as i32,
            kv_heads: self.dims.kv_heads as i32,
            head_dim: self.dims.head_dim as i32,
            d_rel: self.d_rel,
            qk_clip: self.qk_clip,
            rel_clip: self.rel_clip,
            conv_scale: if local {
                self.sconv_scale
            } else {
                self.dconv_scale
            },
        };
        let mut args: Vec<&Array> = lead.to_vec();
        args.extend([&self.q_scale, &self.r_scale, &self.rel_proj]);
        args.extend(tau);
        let run: fn(&ProjCfg, &[&Array]) -> Result<[Array; 6]> = match (local, fuse) {
            (true, true) => fused::project_local,
            (true, false) => regions::project_local,
            (false, true) => fused::project_global,
            (false, false) => regions::project_global,
        };
        run(&cfg, &args)
    }

    fn attend(&self, p: &Projected, kv: &mut NativeKvCache) -> Result<Array> {
        let (k, v) = kv.update_and_fetch(&p.k, &p.v)?;
        banded_sink_attention(&p.q, &k, &v, &p.rel, &p.sink, self.scale, self.dims.window)
    }

    fn output(&self, out: &Array, gate: &Array) -> Result<Array> {
        let (b, length) = (out.shape()[0], out.shape()[2]);
        let merged = mlx_rs::ops::reshape(
            mlx_rs::ops::transpose_axes(out, &[0, 2, 1, 3])?,
            &[b, length, self.dq],
        )?;
        let gated = mlx_rs::ops::multiply(
            merged.as_dtype(Dtype::Float32)?,
            mlx_rs::ops::sigmoid(gate.as_dtype(Dtype::Float32)?)?,
        )?
        .as_dtype(gate.dtype())?;
        self.o_proj.apply(&gated)
    }
}

// ---- mixture of experts ----------------------------------------------------------------------

struct Moe {
    router_weight: Array,
    router_bias: Array,
    w13: ExpertProj,
    w2: ExpertProj,
    top_k: i32,
    route_scale: f32,
    gate_cap: f32,
    up_cap: f32,
}

impl Moe {
    fn load(bag: &mut Bag, quant: &QuantTable, args: &TextArgs, layer: usize) -> Result<Self> {
        let (h, e, i) = (
            args.hidden_size as i32,
            args.n_routed_experts as i32,
            args.intermediate_size as i32,
        );
        let pre = format!("model.layers.{layer}.mlp");
        Ok(Self {
            router_weight: bag.get_f32(&format!("{pre}.router_weight"), &[e, h])?,
            router_bias: bag.get_f32(&format!("{pre}.router_bias"), &[e])?,
            // The rows of `w13` interleave gate and up: [gate0, up0, gate1, up1, ...].
            w13: ExpertProj::load(bag, quant, &format!("{pre}.experts.w13"), e, 2 * i, h)?,
            w2: ExpertProj::load(bag, quant, &format!("{pre}.experts.w2"), e, h, i)?,
            top_k: args.num_experts_per_tok as i32,
            route_scale: args.route_scale,
            gate_cap: args.mlp_gate_softcap,
            up_cap: args.mlp_up_softcap,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let shape = x.shape().to_vec();
        let hidden = *shape.last().expect("rank >= 1");
        let flat = mlx_rs::ops::reshape(x, &[-1, hidden])?;
        let logits = mlx_rs::ops::matmul(
            flat.as_dtype(Dtype::Float32)?,
            mlx_rs::ops::transpose_axes(&self.router_weight, &[1, 0])?,
        )?;
        // Decode-sized passes only (a verify block too), as the mix kernel: prefill keeps the composed ops.
        let rows = flat.shape()[0] as usize;
        let route = if fused::serves(rows) && route_kernel::enabled() {
            route_kernel::route_topk
        } else {
            route_topk
        };
        let (inds, weights) = route(&logits, &self.router_bias, self.top_k, self.route_scale)?;
        profile::mark("moe norm + router + top-k", &[&inds, &weights])?;
        let out = self.experts(&flat, &inds, &weights)?;
        profile::mark("moe combine", &[&out])?;
        Ok(mlx_rs::ops::reshape(&out, &shape)?)
    }

    /// `Inkling2Experts.__call__`: `x` `[N, H]`, `inds` `[N, K]`, `weights` `[N, K]`; `[N, H]`.
    fn experts(&self, x: &Array, inds: &Array, weights: &Array) -> Result<Array> {
        let k = inds.shape()[1];
        let xe = mlx_rs::ops::expand_dims_axes(x, &[-2, -3])?; // [N, 1, 1, H]
        // The reference sorts the (token, expert) pairs by expert once there are 64 or more of
        // them, so each expert's rows are contiguous for the sorted gather kernel.
        let do_sort = inds.size() >= 64;
        let (xe, idx, inverse) = if do_sort {
            let flat_inds = mlx_rs::ops::flatten(inds, 0, -1)?;
            let order = mlx_rs::ops::argsort(&flat_inds)?;
            let inverse = mlx_rs::ops::argsort(&order)?;
            let rows = mlx_rs::ops::flatten(&xe, 0, -3)?;
            let source = mlx_rs::ops::floor_divide(&order, Array::from_int(k))?;
            (
                take_axis(&rows, &source, 0)?,
                take_axis(&flat_inds, &order, 0)?,
                Some(inverse),
            )
        } else {
            (xe, inds.clone(), None)
        };
        let mut y = self.w13.apply(&xe, &idx, do_sort)?;
        if let Some(scale) = self.w13.post_scale(&idx)? {
            let mut shape = scale.shape().to_vec();
            shape.extend([1, 1]);
            y = mlx_rs::ops::multiply(
                &y,
                mlx_rs::ops::reshape(&scale, &shape)?.as_dtype(y.dtype())?,
            )?;
        }
        profile::mark("moe w13 (gate+up experts)", &[&y])?;
        // y[..., 0::2] is the gate, y[..., 1::2] the up projection.
        let mut pair_shape = y.shape().to_vec();
        let last = pair_shape.pop().expect("rank >= 1");
        pair_shape.extend([last / 2, 2]);
        let pairs = mlx_rs::ops::reshape(&y, &pair_shape)?;
        let mut halves = mlx_rs::ops::split(&pairs, 2, -1)?.into_iter();
        let (gate, up) = (
            mlx_rs::ops::squeeze_axes(
                &halves
                    .next()
                    .ok_or_else(|| anyhow!("split yielded too few parts"))?,
                &[-1],
            )?,
            mlx_rs::ops::squeeze_axes(
                &halves
                    .next()
                    .ok_or_else(|| anyhow!("split yielded too few parts"))?,
                &[-1],
            )?,
        );
        // Elementwise and compiled shapeless: every row count takes the compiled form.
        let swiglu = if fused::enabled() {
            fused::softcapped_swiglu
        } else {
            softcapped_swiglu
        };
        let act = swiglu(&gate, &up, self.gate_cap, self.up_cap)?;
        profile::mark("moe softcapped swiglu", &[&act])?;
        let mut z = self.w2.apply(&act, &idx, do_sort)?;
        if let Some(inverse) = inverse {
            let n = inds.shape()[0];
            z = mlx_rs::ops::unflatten(&take_axis(&z, &inverse, 0)?, 0, &[n, k])?;
        }
        profile::mark("moe w2 (down experts)", &[&z])?;
        let z = mlx_rs::ops::squeeze_axes(&z, &[-2])?; // [N, K, H]
        let weights = match self.w2.post_scale(inds)? {
            Some(s2) => mlx_rs::ops::multiply(weights, &s2)?,
            None => weights.clone(),
        };
        let weighted = mlx_rs::ops::multiply(
            z.as_dtype(Dtype::Float32)?,
            mlx_rs::ops::expand_dims(&weights, -1)?,
        )?;
        Ok(mlx_rs::ops::sum_axis(&weighted, -2, false)?.as_dtype(x.dtype())?)
    }
}

// ---- layers ----------------------------------------------------------------------------------

enum Norms {
    Local { attn_norm: Array },
    Global { q: Array, k: Array, v: Array },
}

struct Layer {
    idx: usize,
    sliding: bool,
    /// Every `n_layers_per_attn_res_block` layers the partial sum is completed into a block.
    is_boundary: bool,
    /// Absent on layer 0.
    attn_input: Option<AttnResQuery>,
    mlp_input: AttnResQuery,
    loop_ctl: Option<LoopCtl>,
    attn: Attention,
    mlp: Moe,
    norms: Norms,
    mlp_norm: Array,
    attn_conv: Array,
    mlp_conv: Array,
}

/// The previous pass's message bank: its blocks and its final partial sum, unit-RMS float32.
struct Bank {
    blocks: Vec<Array>,
    partial_n: Array,
}

struct Ctx<'a> {
    args: &'a TextArgs,
    round_loop_mix: bool,
    /// Keep each visit's convolution windows, so a pass can later be rolled back to any of its
    /// positions ([`Inkling2Cache::rollback`]).
    keep_windows: bool,
}

impl Layer {
    fn load(bag: &mut Bag, quant: &QuantTable, args: &TextArgs, idx: usize) -> Result<Self> {
        let sliding = args.is_sliding(idx);
        let (h, taps) = (args.hidden_size as i32, args.sconv_kernel_size as i32);
        let pre = format!("model.layers.{idx}");
        let loop_ctl = if args.in_core(idx) {
            let rank = args.loop_config.gate_rank as i32;
            Some(LoopCtl {
                message: AttnResQuery::load(bag, &format!("{pre}.loop_message.query"), 4, h)?,
                down: bag.get_f32(&format!("{pre}.loop_down"), &[4, 2, h, rank])?,
                gain_forget: bag.get_f32(&format!("{pre}.loop_gain_forget"), &[4, rank, 2 * h])?,
            })
        } else {
            None
        };
        let norms = if sliding {
            Norms::Local {
                attn_norm: bag.get_shaped(&format!("{pre}.attn_norm.weight"), &[h])?,
            }
        } else {
            Norms::Global {
                q: bag.get_shaped(&format!("{pre}.q_input_norm.weight"), &[h])?,
                k: bag.get_shaped(&format!("{pre}.k_input_norm.weight"), &[h])?,
                v: bag.get_shaped(&format!("{pre}.v_input_norm.weight"), &[h])?,
            }
        };
        Ok(Self {
            idx,
            sliding,
            is_boundary: idx.is_multiple_of(args.n_layers_per_attn_res_block),
            attn_input: if idx > 0 {
                let queries = if sliding { 1 } else { 3 };
                Some(AttnResQuery::load(
                    bag,
                    &format!("{pre}.attn_input.query"),
                    queries,
                    h,
                )?)
            } else {
                None
            },
            mlp_input: AttnResQuery::load(bag, &format!("{pre}.mlp_input.query"), 1, h)?,
            loop_ctl,
            attn: Attention::load(bag, quant, args, idx, sliding)?,
            mlp: Moe::load(bag, quant, args, idx)?,
            norms,
            mlp_norm: bag.get_shaped(&format!("{pre}.mlp_norm.weight"), &[h])?,
            attn_conv: bag.get_shaped(&format!("{pre}.attn_conv"), &[h, taps])?,
            mlp_conv: bag.get_shaped(&format!("{pre}.mlp_conv"), &[h, taps])?,
        })
    }

    /// The convolution of an attention output and the residual it joins, in float32: `h` plus the
    /// output, or the output alone at a block boundary (`h` is `None`). Returns the residual and the
    /// convolution's padded input window.
    fn conv_residual(
        &self,
        fuse: bool,
        x: &Array,
        w: &Array,
        state: &Array,
        h: Option<&Array>,
    ) -> Result<(Array, Array)> {
        let mut args: Vec<&Array> = vec![x, w, state];
        args.extend(h);
        let run = if fuse {
            fused::conv_residual
        } else {
            regions::conv_residual
        };
        let [residual, window] = run(self.attn.sconv_scale, &args)?;
        Ok((residual, window))
    }

    /// The MLP output's convolution added to `residual`, in the layer's `dtype`: the visit's
    /// output and the convolution's padded input window.
    fn conv_next(
        &self,
        fuse: bool,
        m: &Array,
        w: &Array,
        state: &Array,
        residual: &Array,
        dtype: Dtype,
    ) -> Result<(Array, Array)> {
        let run = if fuse {
            fused::conv_next
        } else {
            regions::conv_next
        };
        let [next, window] = run(self.attn.sconv_scale, dtype, &[m, w, state, residual])?;
        Ok((next, window))
    }

    /// One visit: `h` is the partial block sum `[B, L, H]`; `blocks` the completed blocks, extended
    /// at a block boundary; `bank` the previous loop pass's messages.
    fn forward(
        &self,
        cx: &Ctx<'_>,
        h: &Array,
        blocks: &mut Vec<Array>,
        cache: &mut VisitCache,
        bank: Option<&Bank>,
    ) -> Result<Array> {
        let args = cx.args;
        let dtype = h.dtype();
        let (b, length, hidden) = (h.shape()[0], h.shape()[1], h.shape()[2]);
        let taps = args.sconv_kernel_size;
        // The compiled regions (`fused.rs`) serve decode-sized inputs.
        let fuse = fused::serves(length as usize);
        let mix_msgs = |x: &Array, message: &Array, down: &Array, gain_forget: &Array| {
            let run = if fuse {
                fused::mix_messages
            } else {
                mix_messages
            };
            run(x, message, down, gain_forget, cx.round_loop_mix)
        };
        let widths = [self.attn.dkv, self.attn.dkv, hidden, hidden];
        let mut states = Vec::with_capacity(4);
        for (slot, width) in widths.into_iter().enumerate() {
            states.push(match &cache.conv[slot] {
                Some(s) => s.clone(),
                None => mlx_rs::ops::zeros::<f32>(&[b, taps as i32 - 1, width])?,
            });
        }
        let tau = self.attn.log_tau(cache.kv.offset(), length as usize)?;
        let messages = match (bank, &self.loop_ctl) {
            (Some(bk), Some(ctl)) => Some(ctl.message.mix(&bk.blocks, &bk.partial_n, fuse)?),
            (Some(_), None) => bail!(
                "layer {} got a message bank but is not in the loop core",
                self.idx
            ),
            (None, _) => None,
        };
        if let Some(m) = &messages {
            profile::mark("loop message bank mix", &[m])?;
        }
        let norm = |x: &Array, w: &Array| -> Result<Array> {
            Ok(mlx_rs::fast::rms_norm(x, w, args.rms_norm_eps)?.as_dtype(dtype)?)
        };

        // -- the inputs of attention --
        let h_n = unit_rms(&h.as_dtype(Dtype::Float32)?)?;
        let (mut xq, mut xk, mut xv);
        if self.idx == 0 {
            (xq, xk, xv) = (h.clone(), h.clone(), h.clone());
        } else {
            let mixed = self
                .attn_input
                .as_ref()
                .ok_or_else(|| anyhow!("layer {} has no attn_input", self.idx))?
                .mix(blocks, &h_n, fuse)?;
            xq = mixed.try_index((.., .., 0))?;
            if self.sliding {
                (xk, xv) = (xq.clone(), xq.clone());
            } else {
                xk = mixed.try_index((.., .., 1))?;
                xv = mixed.try_index((.., .., 2))?;
            }
        }
        profile::mark("attn_input mix", &[&xq, &xk, &xv])?;
        if let (Some(m), Some(ctl)) = (&messages, &self.loop_ctl) {
            let x = mlx_rs::ops::stack_axis(&[&xq, &xk, &xv], 0)?;
            let message =
                mlx_rs::ops::transpose_axes(&m.try_index((.., .., ..3, ..))?, &[2, 0, 1, 3])?;
            let mixed = mix_msgs(
                &x,
                &message,
                &ctl.down.try_index(..3)?,
                &ctl.gain_forget.try_index(..3)?,
            )?;
            (xq, xk, xv) = (
                mixed.try_index(0)?,
                mixed.try_index(1)?,
                mixed.try_index(2)?,
            );
            profile::mark("loop message mix (attn)", &[&xq, &xk, &xv])?;
        }
        let (nq, nk, nv) = match &self.norms {
            Norms::Local { attn_norm } if messages.is_none() => (norm(&xq, attn_norm)?, None, None),
            Norms::Local { attn_norm } => (
                norm(&xq, attn_norm)?,
                Some(norm(&xk, attn_norm)?),
                Some(norm(&xv, attn_norm)?),
            ),
            Norms::Global { q, k, v } => (norm(&xq, q)?, Some(norm(&xk, k)?), Some(norm(&xv, v)?)),
        };
        if profile::active() {
            let outs: Vec<&Array> = [Some(&nq), nk.as_ref(), nv.as_ref()]
                .into_iter()
                .flatten()
                .collect();
            profile::mark("attn input norms", &outs)?;
        }

        // -- attention --
        let p = self.attn.project(
            &nq,
            nk.as_ref(),
            nv.as_ref(),
            &states[0],
            &states[1],
            tau.as_ref(),
            fuse,
        )?;
        profile::mark(
            "qkv projection + conv + clip",
            &[&p.q, &p.k, &p.v, &p.rel, &p.gate, &p.sink],
        )?;
        let out = self.attn.attend(&p, &mut cache.kv)?;
        profile::mark("attention (kv + sdpa)", &[&out])?;
        if self.is_boundary {
            blocks.push(h_n.clone());
        }

        // -- the residual and the MLP --
        let a = self.attn.output(&out, &p.gate)?;
        profile::mark("attn gate + o_proj", &[&a])?;
        let (residual, attn_window) = self.conv_residual(
            fuse,
            &a,
            &self.attn_conv,
            &states[2],
            (!self.is_boundary).then_some(h),
        )?;
        profile::mark("attn conv + residual", &[&residual, &attn_window])?;
        let mut y = self
            .mlp_input
            .mix(blocks, &unit_rms(&residual)?, fuse)?
            .try_index((.., .., 0))?;
        profile::mark("mlp_input mix", &[&y])?;
        if let (Some(m), Some(ctl)) = (&messages, &self.loop_ctl) {
            let message = mlx_rs::ops::expand_dims(&m.try_index((.., .., 3))?, 0)?;
            let mixed = mix_msgs(
                &mlx_rs::ops::expand_dims(&y, 0)?,
                &message,
                &ctl.down.try_index(3..)?,
                &ctl.gain_forget.try_index(3..)?,
            )?;
            y = mixed.try_index(0)?;
            profile::mark("loop message mix (mlp)", &[&y])?;
        }
        let m = self.mlp.forward(&norm(&y, &self.mlp_norm)?)?;
        let (next, mlp_window) =
            self.conv_next(fuse, &m, &self.mlp_conv, &states[3], &residual, dtype)?;
        profile::mark("mlp conv + residual", &[&next, &mlp_window])?;

        cache.conv = [
            Some(next_conv_state(&p.k_window, taps)?),
            Some(next_conv_state(&p.v_window, taps)?),
            Some(next_conv_state(&attn_window, taps)?),
            Some(next_conv_state(&mlp_window, taps)?),
        ];
        if profile::active() {
            let outs: Vec<&Array> = cache.conv.iter().flatten().collect();
            profile::mark("conv state cuts", &outs)?;
        }
        cache.windows = cx.keep_windows.then(|| ConvWindows {
            arrays: [p.k_window, p.v_window, attn_window, mlp_window],
            rows: length as usize,
            keep: taps - 1,
        });
        Ok(next)
    }
}

// ---- cache -----------------------------------------------------------------------------------

/// What one layer visit keeps between calls.
struct VisitCache {
    kv: NativeKvCache,
    /// The `taps - 1` rows before the next chunk: k, v, the attention output and the MLP output.
    /// `None` is zeros.
    conv: [Option<Array>; 4],
    /// The last pass's whole convolution inputs, when it was asked to keep them.
    windows: Option<ConvWindows>,
}

/// The padded inputs of a visit's four short convolutions for one pass of `rows` tokens: the
/// `keep` rows of state before the pass, then the pass's own rows. The state after the first `n`
/// of the pass's tokens is `keep` rows from row `n`.
struct ConvWindows {
    arrays: [Array; 4],
    rows: usize,
    keep: usize,
}

/// The convolution states of every layer visit at one point of a sequence ([`Inkling2Cache::conv_snapshot`]).
/// The KV caches only append, so they can be cut back to any length; these cannot, which is why
/// a point to go back to has to be recorded when the sequence is there.
pub struct ConvSnapshot(Vec<[Option<Array>; 4]>);

/// Image features to put in place of some of a prompt's token embeddings (`embed_inputs` of the
/// reference: `embeds[mask] = features[cumsum(mask) - 1]`, the features cast to the embedding's
/// dtype and not normalized).
///
/// `slots` is indexed by a token's position in the prompt the cache is being filled with, so a
/// prompt fed in chunks passes the same value with each of them: a slot of `-1` is a plain token,
/// any other is the row of `features` (`[rows, hidden]`) that stands in for that position.
pub struct ImageRows {
    slots: Vec<i32>,
    features: Array,
}

impl ImageRows {
    pub fn new(slots: Vec<i32>, features: Array) -> Result<Self> {
        ensure!(
            features.ndim() == 2,
            "image features are {:?}, expected [rows, hidden]",
            features.shape()
        );
        let rows = features.shape()[0];
        if let Some(bad) = slots.iter().find(|&&s| s < -1 || s >= rows) {
            bail!("an image slot of {bad} with {rows} feature rows");
        }
        Ok(Self { slots, features })
    }

    /// `h` (`[1, L, hidden]`, the embeddings of the tokens at `start..start + L`) with the image
    /// positions replaced.
    fn splice(&self, h: &Array, start: usize) -> Result<Array> {
        let len = h.shape()[1] as usize;
        ensure!(
            start + len <= self.slots.len(),
            "the image slots cover {} tokens, the pass feeds positions {start}..{}",
            self.slots.len(),
            start + len
        );
        ensure!(
            self.features.shape()[1] == h.shape()[2],
            "image features are {} wide, the embeddings {}",
            self.features.shape()[1],
            h.shape()[2]
        );
        let window = &self.slots[start..start + len];
        if window.iter().all(|&s| s < 0) {
            return Ok(h.clone());
        }
        let rows: Vec<i32> = window.iter().map(|&s| s.max(0)).collect();
        let mask: Vec<bool> = window.iter().map(|&s| s >= 0).collect();
        let picked = mlx_rs::ops::expand_dims(
            take_axis(&self.features, Array::from_slice(&rows, &[len as i32]), 0)?
                .as_dtype(h.dtype())?,
            0,
        )?;
        let mask = Array::from_slice(&mask, &[1, len as i32, 1]);
        Ok(mlx_rs::ops::which(&mask, &picked, h)?)
    }
}

/// One entry per layer visit (`TextArgs::layer_schedule`): a layer in the loop core has one for
/// each pass.
pub struct Inkling2Cache {
    visits: Vec<VisitCache>,
}

impl Inkling2Cache {
    /// Tokens held.
    pub fn len(&self) -> usize {
        self.visits.first().map_or(0, |v| v.kv.offset())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes the KV caches of all layer visits hold for the tokens seen so far (keys and values,
    /// not the padding the cache keeps for growth; the convolution states are a constant few KB
    /// per visit and not counted).
    pub fn kv_bytes(&self) -> Result<usize> {
        let mut total = 0;
        for visit in &self.visits {
            for view in [visit.kv.keys_view()?, visit.kv.values_view()?]
                .into_iter()
                .flatten()
            {
                total += view.nbytes();
            }
        }
        Ok(total)
    }

    /// The convolution states of every layer visit, to put back with [`Self::rewind`]. Evaluated
    /// here, so what is kept is the small state and not the graph that made it.
    pub fn conv_snapshot(&self) -> Result<ConvSnapshot> {
        let snapshot: Vec<[Option<Array>; 4]> =
            self.visits.iter().map(|v| v.conv.clone()).collect();
        mlx_rs::transforms::eval(snapshot.iter().flatten().flatten())?;
        Ok(ConvSnapshot(snapshot))
    }

    /// Goes back to `len` tokens: the KV caches drop what is past it (they only ever append) and
    /// the convolution states are the ones `conv` was taken with, which must have been taken at
    /// exactly `len` tokens.
    pub fn rewind(&mut self, len: usize, conv: &ConvSnapshot) -> Result<()> {
        ensure!(
            conv.0.len() == self.visits.len(),
            "the snapshot has {} entries, the cache {}",
            conv.0.len(),
            self.visits.len()
        );
        for (visit, states) in self.visits.iter_mut().zip(&conv.0) {
            visit.kv.truncate_to(len)?;
            visit.conv = states.clone();
            visit.windows = None;
        }
        Ok(())
    }

    /// Takes back the last `rejected` tokens of the last pass, which must have kept its windows
    /// (`keep_windows`, as [`Inkling2Model::verify`] does): the KV caches drop them and each
    /// convolution state becomes the one after the pass's first `rows - rejected` tokens, cut
    /// from the windows. Leaves the caches as if only the kept tokens had been fed (the same
    /// states, to rounding: the windows are the very values the pass computed).
    pub fn rollback(&mut self, rejected: usize) -> Result<()> {
        // Every visit is checked before any is changed: a refused rollback leaves the caches as
        // they were, windows included.
        for visit in &self.visits {
            let windows = visit
                .windows
                .as_ref()
                .context("the last pass did not keep its convolution windows")?;
            ensure!(
                rejected < windows.rows,
                "rolling back {rejected} of a pass of {} tokens leaves nothing",
                windows.rows
            );
            ensure!(
                visit.kv.offset() >= rejected,
                "the cache holds {} tokens, fewer than the {rejected} to take back",
                visit.kv.offset()
            );
        }
        for visit in &mut self.visits {
            let windows = visit
                .windows
                .take()
                .context("the last pass did not keep its convolution windows")?;
            let kept = windows.rows - rejected;
            let (from, to) = (kept as i32, (kept + windows.keep) as i32);
            let mut states: [Option<Array>; 4] = [None, None, None, None];
            for (slot, window) in windows.arrays.iter().enumerate() {
                // A small copy, so the state does not pin the pass's windows.
                states[slot] = Some(owned(&window.try_index((.., from..to, ..))?)?);
            }
            let len = visit.kv.offset();
            visit.kv.truncate_to(len - rejected)?;
            visit.conv = states;
        }
        Ok(())
    }

    /// Evaluates the caches' arrays: the KV buffers and the convolution states. A prefill chunk
    /// ends with this, so the lazy graph of one chunk is not carried into the next.
    pub fn eval_state(&self) -> Result<()> {
        let mut pending: Vec<&Array> = Vec::new();
        for visit in &self.visits {
            pending.extend(visit.kv.keys());
            pending.extend(visit.kv.values());
            pending.extend(visit.conv.iter().flatten());
        }
        mlx_rs::transforms::eval(pending)?;
        Ok(())
    }
}

// ---- model -----------------------------------------------------------------------------------

/// What one forward pass records besides the logits, and what it is given besides the ids.
#[derive(Default)]
struct Pass<'a> {
    /// The stream entering every visit and the final-norm output (the parity tests).
    capture: Option<&'a mut Vec<Array>>,
    /// `(ids, slots)`: for each position `k` of `ids`, the stream entering visit `ids[k]` (the
    /// final-norm output for `ids[k] == num_visits()`) goes to `slots[k]`.
    taps: Option<(&'a [usize], &'a mut Vec<Option<Array>>)>,
    images: Option<&'a ImageRows>,
    keep_windows: bool,
}

pub struct Inkling2Model {
    args: TextArgs,
    options: LoadOptions,
    embed: QEmbedding,
    embed_norm: Array,
    layers: Vec<Layer>,
    final_input: AttnResQuery,
    norm: Array,
    lm_head: QLinear,
    /// `-inf` for the ids past `unpadded_vocab_size`.
    vocab_bias: Option<Array>,
    schedule: Vec<(usize, usize)>,
}

impl Inkling2Model {
    /// Loads the text model of the checkpoint in `dir`: every weight it uses is read into memory,
    /// on this thread.
    pub fn load(dir: &Path, options: LoadOptions) -> Result<Self> {
        let args = TextArgs::load(dir)?;
        let config: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("config.json")).context("read config.json")?,
        )?;
        let block = config
            .get("quantization")
            .and_then(serde_json::Value::as_object)
            .context("config.json has no quantization block")?;
        let block: std::collections::BTreeMap<String, serde_json::Value> =
            block.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        let quant = QuantTable::from_block(&block)?;
        let schedule = args.layer_schedule()?;

        let mut bag = Bag {
            weights: NativeWeights::load_dir(dir)?,
            used: Vec::new(),
        };
        let (h, vocab) = (args.hidden_size as i32, args.vocab_size as i32);
        let embed_spec = quant.resolve("model.embed");
        ensure!(
            embed_spec.mode == QuantMode::Affine,
            "model.embed must be affine-quantized"
        );
        let (bits, group_size) = (embed_spec.bits as i32, embed_spec.group_size as i32);
        let groups = h / group_size;
        let embed = QEmbedding {
            weight: bag.get_shaped("model.embed.weight", &[vocab, h * bits / 32])?,
            scales: bag.get_shaped("model.embed.scales", &[vocab, groups])?,
            biases: bag.get_shaped("model.embed.biases", &[vocab, groups])?,
            group_size,
            bits,
        };
        let embed_norm = bag.get_shaped("model.embed_norm.weight", &[h])?;
        let layers = (0..args.num_hidden_layers)
            .map(|i| Layer::load(&mut bag, &quant, &args, i).with_context(|| format!("layer {i}")))
            .collect::<Result<Vec<_>>>()?;
        let final_input = AttnResQuery::load(&mut bag, "model.final_input.query", 1, h)?;
        let norm = bag.get_shaped("model.norm.weight", &[h])?;
        let lm_head = QLinear::load(&mut bag, &quant, "lm_head", vocab, h)?;
        // 0 is "not set", as in the reference.
        let vocab_bias =
            if args.unpadded_vocab_size > 0 && args.unpadded_vocab_size < args.vocab_size {
                let mut bias = vec![0.0f32; args.unpadded_vocab_size];
                bias.resize(args.vocab_size, f32::NEG_INFINITY);
                let a = Array::from_slice(&bias, &[vocab]);
                bag.used.push(a.clone());
                Some(a)
            } else {
                None
            };
        bag.materialize().context("read the weights into memory")?;
        Ok(Self {
            args,
            options,
            embed,
            embed_norm,
            layers,
            final_input,
            norm,
            lm_head,
            vocab_bias,
            schedule,
        })
    }

    pub fn args(&self) -> &TextArgs {
        &self.args
    }

    /// Layer visits per token: the length of the schedule.
    pub fn num_visits(&self) -> usize {
        self.schedule.len()
    }

    pub fn make_cache(&self) -> Inkling2Cache {
        Inkling2Cache {
            visits: self
                .schedule
                .iter()
                .map(|_| VisitCache {
                    kv: NativeKvCache::new(),
                    conv: [None, None, None, None],
                    windows: None,
                })
                .collect(),
        }
    }

    /// Feeds `ids` to the caches without computing logits, and evaluates the result.
    pub fn advance(&self, ids: &[u32], cache: &mut Inkling2Cache) -> Result<()> {
        self.advance_with(ids, cache, None)
    }

    /// [`Self::advance`] with image features standing in for some positions.
    pub fn advance_with(
        &self,
        ids: &[u32],
        cache: &mut Inkling2Cache,
        images: Option<&ImageRows>,
    ) -> Result<()> {
        self.hidden(
            ids,
            cache,
            Pass {
                images,
                ..Pass::default()
            },
        )?
        .eval()?;
        cache.eval_state()
    }

    /// [`Self::advance_with`] that also returns the stream at the `tap_ids` for each fed token,
    /// `[1, L, tap_ids.len() * H]` (see [`Pass`] for what an id names), evaluated. The caches
    /// are evaluated too.
    pub fn advance_taps(
        &self,
        ids: &[u32],
        cache: &mut Inkling2Cache,
        tap_ids: &[usize],
        images: Option<&ImageRows>,
    ) -> Result<Array> {
        let (hidden, taps) = self.run_tapped(ids, cache, tap_ids, images, false)?;
        mlx_rs::transforms::eval([&hidden, &taps])?;
        cache.eval_state()?;
        Ok(taps)
    }

    /// [`Self::forward_with`] that also returns the stream at the `tap_ids` for each fed token,
    /// `[1, L, tap_ids.len() * H]` (the taps are not cut to the last position with `last_only`).
    /// With `keep_windows` the pass keeps its convolution windows, so
    /// [`Inkling2Cache::rollback`] can take back positions of it. The graph is lazy.
    pub fn forward_taps(
        &self,
        ids: &[u32],
        cache: &mut Inkling2Cache,
        last_only: bool,
        tap_ids: &[usize],
        images: Option<&ImageRows>,
        keep_windows: bool,
    ) -> Result<(Array, Array)> {
        let (mut hidden, taps) = self.run_tapped(ids, cache, tap_ids, images, keep_windows)?;
        if last_only {
            let len = hidden.shape()[1];
            hidden = hidden.try_index((.., (len - 1).., ..))?;
        }
        Ok((self.logits_from_hidden(&hidden)?, taps))
    }

    /// The pass that checks a block of drafted tokens: logits `[1, L, vocab]` for every position
    /// and the taps `[1, L, tap_ids.len() * H]`, with the convolution windows kept.
    pub fn verify(
        &self,
        ids: &[u32],
        cache: &mut Inkling2Cache,
        tap_ids: &[usize],
    ) -> Result<(Array, Array)> {
        self.forward_taps(ids, cache, false, tap_ids, None, true)
    }

    fn run_tapped(
        &self,
        ids: &[u32],
        cache: &mut Inkling2Cache,
        tap_ids: &[usize],
        images: Option<&ImageRows>,
        keep_windows: bool,
    ) -> Result<(Array, Array)> {
        ensure!(!tap_ids.is_empty(), "no tap ids");
        // Before the pass: one that fails after it has run leaves every cache advanced.
        if let Some(id) = tap_ids.iter().find(|&&id| id > self.schedule.len()) {
            bail!(
                "tap id {id} names no layer visit (0..{}) or the final norm ({})",
                self.schedule.len(),
                self.schedule.len()
            );
        }
        let mut slots: Vec<Option<Array>> = vec![None; tap_ids.len()];
        let hidden = self.hidden(
            ids,
            cache,
            Pass {
                taps: Some((tap_ids, &mut slots)),
                images,
                keep_windows,
                ..Pass::default()
            },
        )?;
        let mut rows = Vec::with_capacity(slots.len());
        for (slot, id) in slots.into_iter().zip(tap_ids) {
            rows.push(slot.with_context(|| {
                format!(
                    "tap id {id} names no layer visit (0..{}) or the final norm ({})",
                    self.schedule.len(),
                    self.schedule.len()
                )
            })?);
        }
        let taps = if rows.len() == 1 {
            rows.remove(0)
        } else {
            let refs: Vec<&Array> = rows.iter().collect();
            mlx_rs::ops::concatenate_axis(&refs, -1)?
        };
        Ok((hidden, taps))
    }

    /// The head's output for `hidden` before the softcap and the vocabulary mask: what the draft
    /// head's logits are made of.
    pub(super) fn head_logits_raw(&self, hidden: &Array) -> Result<Array> {
        self.lm_head.apply(hidden)
    }

    /// Rows `start..end` of the LM head's weight, dequantized to float32, `[end - start, H]`.
    pub(super) fn head_rows_f32(&self, start: i32, end: i32) -> Result<Array> {
        let h = &self.lm_head;
        let rows = |a: &Array| -> Result<Array> { Ok(a.try_index(start..end)?) };
        Ok(mlx_rs::ops::dequantize(
            &rows(&h.weight)?,
            &rows(&h.scales)?,
            &rows(&h.biases)?,
            h.group_size,
            h.bits,
        )?
        .as_dtype(Dtype::Float32)?)
    }

    /// The embedding of each of `ids` as the stack sees it, `[1, L, H]`: looked up and
    /// normalized.
    pub(super) fn embed_tokens(&self, ids: &[u32]) -> Result<Array> {
        // The lookup does not check bounds on the GPU: an id past the table reads other memory.
        let vocab = self.args.vocab_size;
        ensure!(
            ids.iter().all(|&t| (t as usize) < vocab),
            "a token id is past the vocabulary of {vocab}"
        );
        let ids_i32: Vec<i32> = ids.iter().map(|&t| t as i32).collect();
        let ids = Array::from_slice(&ids_i32, &[1, ids_i32.len() as i32]);
        let embedded = self.embed.lookup(&ids)?;
        Ok(mlx_rs::fast::rms_norm(
            &embedded,
            &self.embed_norm,
            self.args.rms_norm_eps,
        )?)
    }

    /// Feeds `ids` in chunks of at most `chunk` tokens and returns the logits of the last one,
    /// `[1, 1, vocab]`, evaluated. Only the last chunk pays for the output layer: a chunk's
    /// activations are bounded by its length, the whole prompt's would not be.
    pub fn prefill(&self, ids: &[u32], cache: &mut Inkling2Cache, chunk: usize) -> Result<Array> {
        ensure!(!ids.is_empty(), "no token ids");
        let chunk = chunk.max(1);
        let last_start = (ids.len() - 1) / chunk * chunk;
        for piece in ids[..last_start].chunks(chunk) {
            self.advance(piece, cache)?;
        }
        let logits = self.forward(&ids[last_start..], cache, true)?;
        logits.eval()?;
        cache.eval_state()?;
        Ok(logits)
    }

    /// Logits `[1, L, vocab]` float32 for `ids`, the next tokens after `cache.len()` (or `[1, 1,
    /// vocab]` with `last_only`). The ids past `unpadded_vocab_size` are `-inf`. The graph is lazy.
    pub fn forward(
        &self,
        ids: &[u32],
        cache: &mut Inkling2Cache,
        last_only: bool,
    ) -> Result<Array> {
        self.forward_with(ids, cache, last_only, None)
    }

    /// [`Self::forward`] with image features standing in for some positions.
    pub fn forward_with(
        &self,
        ids: &[u32],
        cache: &mut Inkling2Cache,
        last_only: bool,
        images: Option<&ImageRows>,
    ) -> Result<Array> {
        let mut hidden = self.hidden(
            ids,
            cache,
            Pass {
                images,
                ..Pass::default()
            },
        )?;
        if last_only {
            let len = hidden.shape()[1];
            hidden = hidden.try_index((.., (len - 1).., ..))?;
        }
        self.logits_from_hidden(&hidden)
    }

    /// [`Self::forward`] with the partial block sum entering every layer visit, in order, and the
    /// final-norm output last (`num_visits() + 1` arrays of `[1, L, H]`): for the parity tests.
    pub fn forward_captured(
        &self,
        ids: &[u32],
        cache: &mut Inkling2Cache,
    ) -> Result<(Array, Vec<Array>)> {
        let mut captured = Vec::with_capacity(self.schedule.len() + 1);
        let hidden = self.hidden(
            ids,
            cache,
            Pass {
                capture: Some(&mut captured),
                ..Pass::default()
            },
        )?;
        Ok((self.logits_from_hidden(&hidden)?, captured))
    }

    fn logits_from_hidden(&self, hidden: &Array) -> Result<Array> {
        let logits = self.lm_head.apply(hidden)?.as_dtype(Dtype::Float32)?;
        // A cap of 0 is "no cap", as for the draft head; dividing by it would give NaN.
        let logits = if self.args.final_logit_softcapping > 0.0 {
            softcap(&logits, self.args.final_logit_softcapping)?
        } else {
            logits
        };
        let logits = match &self.vocab_bias {
            Some(bias) => mlx_rs::ops::add(&logits, bias)?,
            None => logits,
        };
        profile::mark("lm_head + softcap", &[&logits])?;
        Ok(logits)
    }

    fn hidden(&self, ids: &[u32], cache: &mut Inkling2Cache, mut rec: Pass<'_>) -> Result<Array> {
        ensure!(!ids.is_empty(), "no token ids");
        ensure!(
            cache.visits.len() == self.schedule.len(),
            "the cache has {} entries, the model needs one per layer visit ({})",
            cache.visits.len(),
            self.schedule.len()
        );
        profile::begin(ids.len());
        let mut h = self.embed_tokens(ids)?;
        if let Some(images) = rec.images {
            h = images.splice(&h, cache.len())?;
        }
        profile::mark("embed", &[&h])?;
        let cx = Ctx {
            args: &self.args,
            round_loop_mix: self.options.round_loop_mix,
            keep_windows: rec.keep_windows,
        };
        let core_start = self.args.loop_config.core_start;
        let mut blocks: Vec<Array> = Vec::new();
        let mut prelude: Option<(Array, Vec<Array>)> = None;
        let mut bank: Option<Bank> = None;
        for (visit, &(idx, pass)) in self.schedule.iter().enumerate() {
            if idx == core_start {
                match &prelude {
                    None => prelude = Some((h.clone(), blocks.clone())),
                    Some((prelude_h, prelude_blocks)) => {
                        // The next pass restarts from the prelude's output; the blocks the last
                        // pass wrote (all but the core's first, which every pass rebuilds the
                        // same) become its message bank.
                        bank = Some(Bank {
                            blocks: blocks[prelude_blocks.len() + 1..].to_vec(),
                            partial_n: unit_rms(&h.as_dtype(Dtype::Float32)?)?,
                        });
                        if let Some(bank) = &bank {
                            profile::mark("loop bank", &[&bank.partial_n])?;
                        }
                        h = prelude_h.clone();
                        blocks = prelude_blocks.clone();
                    }
                }
            }
            if let Some(captured) = rec.capture.as_deref_mut() {
                captured.push(h.clone());
            }
            if let Some((tap_ids, slots)) = rec.taps.as_mut() {
                for (k, id) in tap_ids.iter().enumerate() {
                    if *id == visit {
                        slots[k] = Some(h.clone());
                    }
                }
            }
            let active = if pass > 0 { bank.as_ref() } else { None };
            h = self.layers[idx]
                .forward(&cx, &h, &mut blocks, &mut cache.visits[visit], active)
                .with_context(|| format!("layer visit {visit} (layer {idx}, pass {pass})"))?;
        }
        let x = self
            .final_input
            .mix(&blocks, &unit_rms(&h.as_dtype(Dtype::Float32)?)?, false)?
            .try_index((.., .., 0))?;
        let out =
            mlx_rs::fast::rms_norm(&x, &self.norm, self.args.rms_norm_eps)?.as_dtype(h.dtype())?;
        profile::mark("final mix + norm", &[&out])?;
        if let Some(captured) = rec.capture {
            captured.push(out.clone());
        }
        if let Some((tap_ids, slots)) = rec.taps.as_mut() {
            for (k, id) in tap_ids.iter().enumerate() {
                if *id == self.schedule.len() {
                    slots[k] = Some(out.clone());
                }
            }
        }
        Ok(out)
    }
}
