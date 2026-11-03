//! A synthetic MoE layer for tests and a decode-block benchmark of `LUMEN_NATIVE_FUSED_MOE_DECODE`.
//!
//! [`Layer`] holds one Qwen-style layer's quantized tensors (affine 4-bit group 64 with bf16
//! scales for the experts; 8-bit for the router and shared-expert gates) at any size, and views
//! them as the `MoeBlockWeights` the composed ops or the decode kernels read.
//!
//! The ignored benchmark chains `STEPS` blocks of the Qwen3.6-35B-A3B shape (hidden 2048,
//! intermediate 512, 256 experts, top-8) in one graph and prints wall time per block, the CPU time
//! MLX spent encoding it and the dispatches it made: a chain whose encode time reaches its wall
//! time measures the CPU, not the GPU.
//!
//! `cargo test --release -p lumen-mlx --features mlx-native --lib moe_routed_probe -- --ignored
//! --nocapture --test-threads=1`

use std::time::Instant;

use anyhow::{Context, Result};
use mlx_rs::{Array, Dtype, random};

use crate::native_moe::{
    AffineGateWeights, GateProj, GateUpWeights, MoeBlockWeights, SharedMlpWeights,
    SwitchExpertWeights, SwitchGluWeights, moe_block_forward, moe_router,
};
use crate::native_norm::rms_norm;
use crate::native_quant::{MODE_AFFINE, quantize_with_mode};

pub(crate) const TOP_K: i32 = 8;
const GROUP: i32 = 64;
const LAYERS_PER_TOKEN: f64 = 40.0;
const STEPS: usize = 200;
const RUNS: usize = 7;

pub(crate) struct Quant {
    pub w: Array,
    pub s: Array,
    pub b: Array,
}

fn quant(shape: &[i32], bits: i32) -> Result<Quant> {
    let w = random::normal::<f32>(shape, None, None, None)?;
    let w = mlx_rs::ops::multiply(&w, Array::from_f32(0.02))?.as_dtype(Dtype::Bfloat16)?;
    let (w, s, b) = quantize_with_mode(&w, GROUP, bits, MODE_AFFINE)?;
    let b = b.context("affine quantize returned no biases")?;
    for a in [&w, &s, &b] {
        a.eval()?;
    }
    Ok(Quant { w, s, b })
}

fn stack(parts: &[&Array], axis: i32) -> Result<Array> {
    let a = mlx_rs::ops::concatenate_axis(parts, axis)?;
    a.eval()?;
    Ok(a)
}

fn expert(q: &Quant) -> SwitchExpertWeights<'_> {
    SwitchExpertWeights {
        weight: &q.w,
        scales: &q.s,
        biases: Some(&q.b),
    }
}

fn affine(q: &Quant, bits: i32) -> GateProj<'_> {
    GateProj::Affine(AffineGateWeights {
        weight: &q.w,
        scales: &q.s,
        biases: &q.b,
        group_size: GROUP,
        bits,
    })
}

pub(crate) struct Layer {
    pub hidden: i32,
    gate: Quant,
    sw_gate: Quant,
    sw_up: Quant,
    sw_down: Quant,
    se_gate: Quant,
    se_up: Quant,
    se_down: Quant,
    shared_gate: Quant,
    /// `[experts, 2 * inter, ..]`: gate rows then up rows of each expert.
    gate_up: Quant,
}

impl Layer {
    pub(crate) fn build(hidden: i32, inter: i32, experts: i32) -> Result<Self> {
        random::seed(11)?;
        let sw_gate = quant(&[experts, inter, hidden], 4)?;
        let sw_up = quant(&[experts, inter, hidden], 4)?;
        let gate_up = Quant {
            w: stack(&[&sw_gate.w, &sw_up.w], 1)?,
            s: stack(&[&sw_gate.s, &sw_up.s], 1)?,
            b: stack(&[&sw_gate.b, &sw_up.b], 1)?,
        };
        Ok(Self {
            hidden,
            gate: quant(&[experts, hidden], 8)?,
            sw_down: quant(&[experts, hidden, inter], 4)?,
            se_gate: quant(&[inter, hidden], 4)?,
            se_up: quant(&[inter, hidden], 4)?,
            se_down: quant(&[hidden, inter], 4)?,
            shared_gate: quant(&[1, hidden], 8)?,
            sw_gate,
            sw_up,
            gate_up,
        })
    }

    /// The layer as the block reads it: experts stacked or not, decode kernels allowed or not.
    pub(crate) fn view(&self, stacked: bool, fused_decode: bool) -> MoeBlockWeights<'_> {
        MoeBlockWeights {
            gate: affine(&self.gate, 8),
            switch_glu: SwitchGluWeights {
                gate_up: if stacked {
                    GateUpWeights::Fused(expert(&self.gate_up))
                } else {
                    GateUpWeights::Split {
                        gate: expert(&self.sw_gate),
                        up: expert(&self.sw_up),
                    }
                },
                down_proj: expert(&self.sw_down),
                group_size: GROUP,
                bits: 4,
                mode: MODE_AFFINE,
            },
            shared_expert: SharedMlpWeights {
                gate_proj_w: &self.se_gate.w,
                gate_proj_s: &self.se_gate.s,
                gate_proj_b: Some(&self.se_gate.b),
                up_proj_w: &self.se_up.w,
                up_proj_s: &self.se_up.s,
                up_proj_b: Some(&self.se_up.b),
                down_proj_w: &self.se_down.w,
                down_proj_s: &self.se_down.s,
                down_proj_b: Some(&self.se_down.b),
                group_size: GROUP,
                bits: 4,
                mode: MODE_AFFINE,
            },
            shared_gate: affine(&self.shared_gate, 8),
            top_k: TOP_K,
            norm_topk_prob: true,
            experts_in_scale_dtype: true,
            fused_decode,
            offload: None,
        }
    }

    /// A unit-RMS `[1, 1, hidden]` f32 activation.
    pub(crate) fn activation(&self, seed: u64) -> Result<Array> {
        random::seed(seed)?;
        let x = random::normal::<f32>(&[1, 1, self.hidden], None, None, None)?;
        let x = rms_norm(&x, &Array::ones::<f32>(&[self.hidden])?, 1e-6)?;
        x.eval()?;
        Ok(x)
    }

    /// Bytes one block reads: nine experts' gate, up and down tensors (packed, scales, biases).
    fn bytes_per_block(&self) -> f64 {
        let per_expert = |q: &Quant| {
            let n = |a: &Array| a.shape()[1..].iter().product::<i32>() as f64;
            n(&q.w) * 4.0 + n(&q.s) * 2.0 + n(&q.b) * 2.0
        };
        (TOP_K + 1) as f64 * (per_expert(&self.gate_up) + per_expert(&self.sw_down))
    }
}

fn chain(
    step: &dyn Fn(&Array) -> Result<Array>,
    x0: &Array,
    norm_w: &Array,
    steps: usize,
) -> Result<Array> {
    let mut x = x0.clone();
    for _ in 0..steps {
        let y = step(&x)?.as_dtype(Dtype::Float32)?;
        x = rms_norm(&y, norm_w, 1e-6)?;
    }
    Ok(x)
}

struct Timing {
    wall_us: f64,
    encode_us: f64,
    dispatches: f64,
}

/// Best-of-`RUNS` per-block wall time with the encode time and dispatches of that run.
fn time_chain(
    step: &dyn Fn(&Array) -> Result<Array>,
    x0: &Array,
    norm_w: &Array,
) -> Result<Timing> {
    let eval = || -> Result<Timing> {
        let (_, d0) = mlx_rs::metal::cmd_buffer_stats().unwrap_or_default();
        let (_, e0) = mlx_rs::metal::eval_gpu_stats().unwrap_or_default();
        let t = Instant::now();
        chain(step, x0, norm_w, STEPS)?.eval()?;
        let wall = t.elapsed().as_secs_f64();
        let (_, d1) = mlx_rs::metal::cmd_buffer_stats().unwrap_or_default();
        let (_, e1) = mlx_rs::metal::eval_gpu_stats().unwrap_or_default();
        Ok(Timing {
            wall_us: wall * 1e6 / STEPS as f64,
            encode_us: (e1 - e0) as f64 / 1e3 / STEPS as f64,
            dispatches: (d1 - d0) as f64 / STEPS as f64,
        })
    };
    for _ in 0..3 {
        eval()?;
    }
    let mut runs = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        runs.push(eval()?);
    }
    runs.sort_by(|a, b| a.wall_us.total_cmp(&b.wall_us));
    Ok(runs.swap_remove(0))
}

fn report(name: &str, t: &Timing) {
    println!(
        "{name:<40} {:>7.1} us/block  encode {:>6.1} us  {:>5.1} dispatches   -> {:>5.2} ms/token",
        t.wall_us,
        t.encode_us,
        t.dispatches,
        t.wall_us * LAYERS_PER_TOKEN / 1e3
    );
}

#[test]
#[ignore = "benchmark: prints timings"]
fn decode_kernels_against_composed_block() {
    let layer = Layer::build(2048, 512, 256).expect("layer");
    let norm_w = Array::ones::<f32>(&[2048]).expect("ones");
    let x0 = layer.activation(5).expect("x");

    let bw = crate::measure_memory_bandwidth().expect("bandwidth");
    let bytes = layer.bytes_per_block();
    println!(
        "\nread bandwidth {:.0} GB/s (best {:.0}); a block reads {:.1} MB -> floor {:.1} us/block = {:.2} ms/token",
        bw.median_gbps,
        bw.best_gbps,
        bytes / 1e6,
        bytes / (bw.best_gbps * 1e9) * 1e6,
        bytes / (bw.best_gbps * 1e9) * LAYERS_PER_TOKEN * 1e3
    );

    let identity = |x: &Array| Ok(x.clone());
    let norm_t = time_chain(&identity, &x0, &norm_w).expect("norm timing");
    let split = layer.view(false, false);
    let router_only = |x: &Array| {
        let (_, scores) = moe_router(x, &split)?;
        Ok(mlx_rs::ops::multiply(x, &scores.sum(None)?)?)
    };
    let router_t = time_chain(&router_only, &x0, &norm_w).expect("router timing");
    println!();
    report("chain overhead (cast + norm)", &norm_t);
    report("router only", &router_t);

    let mut timings = Vec::new();
    for (name, stacked, fused) in [
        ("composed, gate/up separate", false, false),
        ("composed, gate/up stacked", true, false),
        ("decode kernels", true, true),
    ] {
        let w = layer.view(stacked, fused);
        let step = |x: &Array| moe_block_forward(x, &w);
        let t = time_chain(&step, &x0, &norm_w).expect("block timing");
        report(name, &t);
        timings.push(t);
    }
    println!(
        "\ndecode kernels vs composed: {:+.2} ms/token over {LAYERS_PER_TOKEN} layers (positive = kernels faster)",
        (timings[0].wall_us - timings[2].wall_us) * LAYERS_PER_TOKEN / 1e3
    );
}
