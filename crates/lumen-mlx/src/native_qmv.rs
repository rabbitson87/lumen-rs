//! MLX's quantized matvec, transcribed for the decode kernels that must add what MLX adds in the
//! order MLX adds it, and the projection kernels built on it.
//!
//! MLX runs `quantized_matmul` in the common dtype of the activation and the scales. The Qwen
//! trunk's activation is f32 and an affine checkpoint's scales and biases bf16, so MLX either casts
//! them on every call or reads copies widened to f32 at load (`LUMEN_NATIVE_WIDEN_SCALES`): 8 bytes
//! per group of 64 weights (32 bytes) instead of 4. [`matvec`] reads the bf16 scales and widens them
//! in registers, which is exact, and otherwise runs the matvec MLX would run for that many vectors
//! on this GPU, in its arithmetic, so its output equals `quantized_matmul` on the widened scales in
//! every bit. With the kernels serving a projection's decode steps, its scales need no f32 copy
//! ([`serves`]).
//!
//! A run-time kernel costs CPU on every call that MLX's own matmul does not: `fast::metal_kernel`
//! rebuilds the kernel's whole source text and hashes it each time it is called and encoded. A
//! decode step that is close to CPU-bound (the 35B-A3B's) feels that, so projections that share an
//! activation run in one launch ([`matvec_shared`]) and every source carries only the helpers it
//! calls, with no comments inside the Metal text.

use std::cell::{OnceCell, RefCell};
use std::collections::HashMap;
use std::ffi::{CStr, c_char};
use std::rc::Rc;
use std::sync::OnceLock;

use anyhow::{Context, Result, anyhow};
use mlx_rs::{Array, Dtype};

use crate::metal_kernel::{MetalKernel, MetalKernelConfig};
use crate::native_quant::MODE_AFFINE;

lumen_flags::flag! {
    /// Run the matvecs of an affine 4-bit group-64 projection whose activation is f32 and whose
    /// scales and biases are bf16 or f16 in kernels that read those scales as they are
    /// (`native_qmv::matvec`), instead of MLX's `quantized_matmul`, which casts them to f32 on every
    /// call, and keep such projections' scales in bf16 under `LUMEN_NATIVE_WIDEN_SCALES`. A
    /// single-token step runs the projections of one activation (linear attention's four input
    /// projections, attention's q/k/v, a dense MLP's gate and up) in one launch.
    ///
    /// A matvec is up to MLX's batch limit of rows (6 to 33, by GPU and size; 13-15 on an M3 Max):
    /// a decode step, a batch of decode steps, an MTP verify. Larger matmuls (prefill) stay on MLX,
    /// which casts the bf16 scales per call there; against a matmul that size the cast is noise.
    ///
    /// Bit-identical: the kernels run MLX's own matvec for that many rows on this GPU (`qmv_fast`
    /// for one, `qmv_wide` or `qmv_fast` for more), its lane partition and order of adds, and
    /// widening a bf16 or f16 scale to f32 is exact. An `Optimization` flag. Default on; `=0` keeps
    /// MLX's matmul (and widens every affine projection's scales again).
    ///
    /// M3 Max, one matvec over projections larger than the system cache, best of 12: 17408x5120
    /// 218 -> 200 us, 5120x17408 213 -> 200 us, 248320x2048 (lm_head) 1187 -> 1063 us against MLX
    /// on widened scales (MLX on bf16 activations and scales, the bound: 195, 196, 1066 us); a small
    /// projection (32x2048) is slower alone, 3.6 -> 6.1 us, which the shared launch removes.
    pub(crate) decode_qmv {
        env: "LUMEN_NATIVE_DECODE_QMV",
        default: true,
        kind: Optimization,
    }
}

/// Output rows one simdgroup owns, and simdgroups to a threadgroup: MLX's `qmv_fast` takes 4 and 2.
const ROWS: i32 = 4;
const SIMDGROUPS: i32 = 2;
/// Projections one shared launch takes at most: linear attention's four input projections.
const MAX_SHARED: usize = 4;

/// MLX's `qmv_fast` (`qmv_fast_impl<float, 64, 4>`), one simdgroup per `R` output rows of vector
/// `z`: lane `l` takes values `16 l .. 16 l + 16` of every block of 512, the rows' sums run block
/// by block in `qdot`'s order, then `simd_sum` adds the lanes. [`fast_source`] puts in front of it
/// the choice of the projection (`w`, `sc`, `bi`, `out`, `N`) and the row group `rg` within it.
const FAST_BODY: &str = r#"
    const int lane = thread_index_in_simdgroup;
    const int vec = thread_position_in_grid.z;
    constexpr int G = K / 64;
    constexpr int WPR = K / 8;
    constexpr int BLOCKS = K / 512;
    const device float* xv = x + (size_t)vec * K;
    float acc[R];
    for (int r = 0; r < R; ++r) {
        acc[r] = 0.0f;
    }
    for (int blk = 0; blk < BLOCKS; ++blk) {
        float x_thread[16];
        const float sum = load_vector_f32(xv + blk * 512 + lane * 16, x_thread);
        const int scale_at = blk * 8 + lane / 4;
        for (int r = 0; r < R; ++r) {
            const size_t row = (size_t)rg * R + r;
            const device uint16_t* pw = (const device uint16_t*)(w + row * WPR + blk * 64 + lane * 2);
            const float s = sc[row * G + scale_at];
            const float b = bi[row * G + scale_at];
            acc[r] += qdot4(pw, x_thread, s, b, sum);
        }
    }
    for (int r = 0; r < R; ++r) {
        const float y = simd_sum(acc[r]);
        if (lane == 0) {
            out[(size_t)vec * N + rg * R + r] = y;
        }
    }
"#;

/// [`FAST_BODY`] for `projections` projections of one activation (`w0`, `sc0`, `bi0`, `out0`,
/// `N0`, then `w1`, ...): projection `p` owns the `Np / R` row groups after those of the ones
/// before it, along the grid's y axis.
fn fast_source(projections: usize) -> String {
    let mut s = String::from("    int rg = thread_position_in_grid.y;\n    int p = 0;\n");
    for p in 1..projections {
        let q = p - 1;
        s += &format!(
            "    if (p == {q} && rg >= N{q} / R) {{\n        rg -= N{q} / R;\n        p = {p};\n    }}\n"
        );
    }
    let pick = |name: &str| {
        (0..projections - 1)
            .rev()
            .fold(format!("{name}{}", projections - 1), |rest, p| {
                format!("p == {p} ? {name}{p} : {rest}")
            })
    };
    s += &format!("    const device uint32_t* w = {};\n", pick("w"));
    s += &format!("    const auto sc = {};\n", pick("sc"));
    s += &format!("    const auto bi = {};\n", pick("bi"));
    s += &format!("    device float* out = {};\n", pick("out"));
    s += &format!("    const int N = {};\n", pick("N"));
    s + FAST_BODY
}

/// MLX's `qmv_wide` (`qmv_wide_impl<float, 64, 4, V, 8>`), `M` vectors in tiles of `V`: eight lanes
/// to a row, lane `k` taking groups `k, k + 8, ...` of the row, each decoded eight values at a time
/// and dotted with every vector of the tile; a shuffle ladder then adds the eight lanes.
const WIDE_SOURCE: &str = r#"
    const int lane = thread_index_in_simdgroup;
    const int sg = simdgroup_index_in_threadgroup;
    const int tile = threadgroup_position_in_grid.x;
    constexpr int KL = 8;
    constexpr int RPS = 32 / KL;
    constexpr int G = K / 64;
    const int k_lane = lane % KL;
    const int sg_row = lane / KL;
    const int out_row = threadgroup_position_in_grid.y * (RPS * 2) + RPS * sg + sg_row;
    const int vec0 = tile * V;
    const int row = min(out_row, N - 1);
    const device uint8_t* wrow = (const device uint8_t*)w + (size_t)row * (K / 2);
    const size_t srow = (size_t)row * G;
    const device float* xv[V];
    for (int v = 0; v < V; v++) {
        xv[v] = x + (size_t)min(vec0 + v, M - 1) * K;
    }
    float result[V];
    for (int v = 0; v < V; v++) {
        result[v] = 0.0f;
    }
    for (int g = k_lane; g < G; g += KL) {
        const float scale = sc[srow + g];
        const float bias = bi[srow + g];
        for (int c = 0; c < 8; c++) {
            const int k0 = g * 64 + c * 8;
            float w_dq[8];
            dequantize4x8(wrow + k0 / 2, scale, bias, w_dq);
            for (int v = 0; v < V; v++) {
                const device float* xc = xv[v] + k0;
                float acc = 0;
                for (int i = 0; i < 8; i++) {
                    acc += xc[i] * w_dq[i];
                }
                result[v] += acc;
            }
        }
    }
    for (int v = 0; v < V; v++) {
        result[v] += simd_shuffle_down(result[v], 4);
        result[v] += simd_shuffle_down(result[v], 2);
        result[v] += simd_shuffle_down(result[v], 1);
    }
    if (k_lane == 0 && out_row < N) {
        for (int v = 0; v < V; v++) {
            if (vec0 + v < M) {
                out[(size_t)(vec0 + v) * N + out_row] = result[v];
            }
        }
    }
"#;

/// The matvec MLX runs for some number of vectors, which the kernels reproduce.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Kind {
    /// `qmv_fast`, each vector on its own.
    Fast,
    /// `qmv_wide` in tiles of `vecs` vectors.
    Wide { vecs: i32 },
}

/// The GPU as MLX's dispatch reads it: the generation and size letter of its architecture name
/// (`applegpu_g15s` is generation 15, size `s`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Gpu {
    generation: i32,
    size: u8,
}

impl Gpu {
    /// `metal::Device`'s parse: the two digits before the last character, each 0 when it is not a
    /// digit, and the last character.
    fn parse(architecture: &str) -> Option<Self> {
        let b = architecture.as_bytes();
        let [.., tens, ones, size] = *b else {
            return None;
        };
        let digit = |c: u8| {
            if c.is_ascii_digit() {
                i32::from(c - b'0')
            } else {
                0
            }
        };
        Some(Self {
            generation: digit(tens) * 10 + digit(ones),
            size,
        })
    }

    /// MLX's `get_qmv_batch_limit`: from this many vectors on, a `[outputs, inputs]` projection runs
    /// as a matrix kernel.
    fn batch_limit(self, inputs: i32, outputs: i32) -> i32 {
        let pick = |small: i32, medium: i32, large: i32| {
            if inputs <= 2048 && outputs <= 2048 {
                small
            } else if inputs <= 4096 && outputs <= 4096 {
                medium
            } else {
                large
            }
        };
        let desktop = self.size == b'd';
        match self.generation {
            g if g >= 17 && !desktop => pick(33, 25, 13),
            g if g >= 15 && !desktop => pick(13, 15, 13),
            _ if desktop => pick(32, 18, 12),
            g if g >= 13 => pick(14, 10, 6),
            _ => pick(18, 12, 10),
        }
    }

    /// The matvec MLX runs for `vectors` rows of a `[outputs, inputs]` affine projection whose
    /// lengths are `qmv_fast`'s, or `None` when it runs a matrix kernel.
    fn kind(self, vectors: i32, inputs: i32, outputs: i32) -> Option<Kind> {
        if vectors >= self.batch_limit(inputs, outputs) {
            return None;
        }
        // `use_qmv_wide`: affine `qmv_wide` only from generation 15. Tiles of at most five vectors,
        // as few tiles as that allows, as even as they can be.
        Some(if vectors >= 2 && self.generation >= 15 {
            let tiles = (vectors + 4) / 5;
            Kind::Wide {
                vecs: (vectors + tiles - 1) / tiles,
            }
        } else {
            Kind::Fast
        })
    }
}

/// The default GPU's architecture as MLX reports it (`device_info`), which is the name its dispatch
/// reads, `MLX_METAL_GPU_ARCH` included. `None` if it cannot be read.
fn gpu() -> Option<Gpu> {
    static GPU: OnceLock<Option<Gpu>> = OnceLock::new();
    *GPU.get_or_init(|| {
        // SAFETY: plain mlx-c calls on objects created and freed here; the architecture string
        // belongs to `info` and is copied before `info` is freed.
        unsafe {
            let device = mlx_sys::mlx_device_new_type(mlx_sys::mlx_device_type__MLX_GPU, 0);
            let mut info = mlx_sys::mlx_device_info_new();
            let mut name: *const c_char = std::ptr::null();
            let ok = mlx_sys::mlx_device_info_get(&mut info, device) == 0
                && mlx_sys::mlx_device_info_get_string(&mut name, info, c"architecture".as_ptr())
                    == 0
                && !name.is_null();
            let architecture = ok.then(|| CStr::from_ptr(name).to_string_lossy().into_owned());
            mlx_sys::mlx_device_info_free(info);
            mlx_sys::mlx_device_free(device);
            architecture.and_then(|a| Gpu::parse(&a))
        }
    })
}

/// One quantized projection's tensors and parameters, as the kernels read them.
#[derive(Clone, Copy)]
pub(crate) struct Weights<'a> {
    pub weight: &'a Array,
    pub scales: &'a Array,
    pub biases: Option<&'a Array>,
    pub group_size: i32,
    pub bits: i32,
    pub mode: &'a CStr,
}

/// A matvec the kernels run: `vectors` rows of `inputs` against `[outputs, inputs]` weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Plan {
    vectors: i32,
    inputs: i32,
    outputs: i32,
    kind: Kind,
}

/// `[outputs, inputs]` and the biases when the kernels read this projection: `decode_qmv` is on,
/// the weights are affine 4-bit group 64 in the lengths MLX runs as `qmv_fast` (inputs a multiple
/// of 512, outputs a multiple of 8), and the scales and biases are bf16 or f16, which
/// `quantized_matmul` would cast.
fn layout<'a>(w: &Weights<'a>) -> Option<(i32, i32, &'a Array)> {
    if !decode_qmv::get() || w.mode != MODE_AFFINE || (w.group_size, w.bits) != (64, 4) {
        return None;
    }
    let (Some(biases), &[outputs, packed]) = (w.biases, w.weight.shape()) else {
        return None;
    };
    let inputs = packed * 8;
    let ok = inputs % 512 == 0
        && outputs % (ROWS * SIMDGROUPS) == 0
        && w.weight.dtype() == Dtype::Uint32
        && w.scales.shape() == [outputs, inputs / 64]
        && biases.shape() == [outputs, inputs / 64]
        && matches!(w.scales.dtype(), Dtype::Bfloat16 | Dtype::Float16)
        && biases.dtype() == w.scales.dtype();
    ok.then_some((outputs, inputs, biases))
}

/// Whether the kernels run every matvec of this projection that MLX would run as one, so a load
/// may keep its scales in bf16: the projection has their layout and the GPU is known (a single
/// vector needs no GPU rule; several do).
pub(crate) fn serves(w: &Weights<'_>) -> bool {
    gpu().is_some() && layout(w).is_some()
}

/// The matvec [`matvec`] runs for `x` against this projection, or `None` when it is not the
/// kernels' (see [`layout`]), `x` is not f32, or `x` has as many rows as MLX runs as a matrix.
///
/// Rows count as MLX counts them for a contiguous `x`: every vector of the leading axes. MLX counts
/// a strided `x` by its second-to-last axis instead, so a strided `[n, 1, k]` would run elsewhere;
/// the trunk's activations come out of norms and are contiguous.
pub(crate) fn plan(x: &Array, w: &Weights<'_>) -> Option<Plan> {
    let (outputs, inputs, _) = layout(w)?;
    if x.dtype() != Dtype::Float32 || x.shape().last() != Some(&inputs) {
        return None;
    }
    let vectors = i32::try_from(x.size() / inputs as usize).ok()?;
    let kind = match vectors {
        0 => return None,
        // One vector runs as `qmv_fast` on every GPU.
        1 => Kind::Fast,
        _ => gpu()?.kind(vectors, inputs, outputs)?,
    };
    Some(Plan {
        vectors,
        inputs,
        outputs,
        kind,
    })
}

struct Kernels {
    /// `fast[p - 1]` runs `p` projections of one activation.
    fast: [MetalKernel; MAX_SHARED],
    wide: MetalKernel,
}

thread_local! {
    /// MLX keeps one stream per thread and the engine decodes on one thread, so the kernels are
    /// built once per thread rather than per call.
    static KERNELS: OnceCell<Kernels> = const { OnceCell::new() };
    /// Launch configs by shape, built once: a config is a dozen FFI calls, a decode step runs a
    /// few shapes many times.
    static CONFIGS: RefCell<HashMap<ConfigKey, Rc<MetalKernelConfig>>> =
        RefCell::new(HashMap::new());
}

/// What a launch's config depends on: the kernel, the row count, the input length and each
/// projection's output rows (unused slots 0).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct ConfigKey {
    kind: Kind,
    vectors: i32,
    inputs: i32,
    outputs: [i32; MAX_SHARED],
}

/// The config for `key`, built by `build` the first time.
fn config(
    key: ConfigKey,
    build: impl FnOnce(&MetalKernelConfig) -> Result<()>,
) -> Result<Rc<MetalKernelConfig>> {
    if let Some(c) = CONFIGS.with(|m| m.borrow().get(&key).cloned()) {
        return Ok(c);
    }
    let c = MetalKernelConfig::new();
    build(&c)?;
    let c = Rc::new(c);
    CONFIGS.with(|m| m.borrow_mut().insert(key, Rc::clone(&c)));
    Ok(c)
}

/// The input names, output names and source of the fast kernel over `projections` projections of
/// one activation: `x`, then `w`, `sc`, `bi` of each; `out` of each.
struct FastSignature {
    inputs: Vec<String>,
    outputs: Vec<String>,
    source: String,
}

impl FastSignature {
    fn new(projections: usize) -> Self {
        let mut inputs = vec!["x".to_string()];
        for p in 0..projections {
            inputs.extend([format!("w{p}"), format!("sc{p}"), format!("bi{p}")]);
        }
        Self {
            inputs,
            outputs: (0..projections).map(|p| format!("out{p}")).collect(),
            source: fast_source(projections),
        }
    }

    fn inputs(&self) -> Vec<&str> {
        self.inputs.iter().map(String::as_str).collect()
    }

    fn outputs(&self) -> Vec<&str> {
        self.outputs.iter().map(String::as_str).collect()
    }
}

fn with_kernels<R>(f: impl FnOnce(&Kernels) -> Result<R>) -> Result<R> {
    KERNELS.with(|cell| {
        if cell.get().is_none() {
            // Each kernel is constructed with its name as a literal: `cargo xtask test` reads the
            // names off the constructor calls to check that every kernel ran in its Metal 3.1 pass.
            let [one, two, three, four] = [1, 2, 3, 4].map(FastSignature::new);
            let kernels = Kernels {
                fast: [
                    MetalKernel::new_with_header(
                        "decode_qmv_affine4",
                        &one.inputs(),
                        &one.outputs(),
                        QMV_F32,
                        &one.source,
                        /* ensure_row_contiguous */ true,
                        /* atomic_outputs */ false,
                    )?,
                    MetalKernel::new_with_header(
                        "decode_qmv_affine4_x2",
                        &two.inputs(),
                        &two.outputs(),
                        QMV_F32,
                        &two.source,
                        /* ensure_row_contiguous */ true,
                        /* atomic_outputs */ false,
                    )?,
                    MetalKernel::new_with_header(
                        "decode_qmv_affine4_x3",
                        &three.inputs(),
                        &three.outputs(),
                        QMV_F32,
                        &three.source,
                        /* ensure_row_contiguous */ true,
                        /* atomic_outputs */ false,
                    )?,
                    MetalKernel::new_with_header(
                        "decode_qmv_affine4_x4",
                        &four.inputs(),
                        &four.outputs(),
                        QMV_F32,
                        &four.source,
                        /* ensure_row_contiguous */ true,
                        /* atomic_outputs */ false,
                    )?,
                ],
                wide: MetalKernel::new_with_header(
                    "decode_qmv_wide_affine4",
                    &["x", "w", "sc", "bi"],
                    &["out"],
                    QMV_WIDE,
                    WIDE_SOURCE,
                    /* ensure_row_contiguous */ true,
                    /* atomic_outputs */ false,
                )?,
            };
            let _ = cell.set(kernels);
        }
        f(cell
            .get()
            .ok_or_else(|| anyhow!("decode matvec kernels missing"))?)
    })
}

/// `x`'s shape with the last axis `outputs`, for an output the kernels wrote flat.
fn reshape_out(y: &Array, x: &Array, outputs: i32) -> Result<Array> {
    let mut shape = x.shape().to_vec();
    if let Some(last) = shape.last_mut() {
        *last = outputs;
    }
    Ok(mlx_rs::ops::reshape(y, &shape)?)
}

/// `qmv_fast` for every plan, one launch: the plans share `x` and its row count.
fn run_fast(x: &Array, plans: &[Plan], weights: &[Weights<'_>]) -> Result<Vec<Array>> {
    let (vectors, inputs) = (plans[0].vectors, plans[0].inputs);
    let flat = mlx_rs::ops::reshape(x, &[vectors * inputs])?;
    let mut tensors: Vec<&Array> = vec![&flat];
    for w in weights {
        let biases = w
            .biases
            .context("decode matvec: projection without biases")?;
        tensors.extend([w.weight, w.scales, biases]);
    }
    let mut outputs = [0; MAX_SHARED];
    for (o, plan) in outputs.iter_mut().zip(plans) {
        *o = plan.outputs;
    }
    let key = ConfigKey {
        kind: Kind::Fast,
        vectors,
        inputs,
        outputs,
    };
    let config = config(key, |config| {
        config.add_template_arg_int("K", inputs)?;
        config.add_template_arg_int("R", ROWS)?;
        for (p, plan) in plans.iter().enumerate() {
            config.add_template_arg_int(&format!("N{p}"), plan.outputs)?;
            config.add_output_arg(&[vectors * plan.outputs], Dtype::Float32)?;
        }
        let groups: i32 = plans.iter().map(|p| p.outputs / ROWS).sum();
        config.set_grid(32, groups, vectors)?;
        config.set_thread_group(32, SIMDGROUPS, 1)
    })?;
    let ys = with_kernels(|k| {
        k.fast[plans.len() - 1]
            .apply(&tensors, &config, plans.len())
            .context("decode matvec kernel failed")
    })?;
    ys.iter()
        .zip(plans)
        .map(|(y, plan)| reshape_out(y, x, plan.outputs))
        .collect()
}

/// `x @ deq(weight).T` as [`plan`] planned it: `x`'s shape with the last axis the weight's rows,
/// f32.
pub(crate) fn matvec(plan: &Plan, x: &Array, w: &Weights<'_>) -> Result<Array> {
    let Kind::Wide { vecs } = plan.kind else {
        return Ok(run_fast(x, &[*plan], &[*w])?.swap_remove(0));
    };
    let Plan {
        vectors,
        inputs,
        outputs,
        ..
    } = *plan;
    let biases = w
        .biases
        .context("decode matvec: projection without biases")?;
    let flat = mlx_rs::ops::reshape(x, &[vectors * inputs])?;
    let key = ConfigKey {
        kind: plan.kind,
        vectors,
        inputs,
        outputs: [outputs, 0, 0, 0],
    };
    let config = config(key, |config| {
        config.add_template_arg_int("K", inputs)?;
        config.add_template_arg_int("N", outputs)?;
        config.add_template_arg_int("M", vectors)?;
        config.add_template_arg_int("V", vecs)?;
        config.add_output_arg(&[vectors * outputs], Dtype::Float32)?;
        let tiles = (vectors + vecs - 1) / vecs;
        config.set_grid(32 * tiles, SIMDGROUPS * (outputs / (ROWS * SIMDGROUPS)), 1)?;
        config.set_thread_group(32, SIMDGROUPS, 1)
    })?;
    let y = with_kernels(|k| {
        Ok(k.wide
            .apply(&[&flat, w.weight, w.scales, biases], &config, 1)
            .context("decode matvec kernel failed")?
            .swap_remove(0))
    })?;
    reshape_out(&y, x, outputs)
}

/// `x @ deq(W).T` for each of up to four projections of one activation, in one launch, or `None`
/// when that launch cannot run them: one is not the kernels' (see [`plan`]) or MLX would not run it
/// as `qmv_fast` (several vectors on a GPU that runs them as `qmv_wide`). The caller then runs the
/// projections one at a time. Each output equals [`matvec`]'s for its projection in every bit: the
/// launch gives every projection's rows their own simdgroups, as its own launch would.
pub(crate) fn matvec_shared(x: &Array, weights: &[Weights<'_>]) -> Result<Option<Vec<Array>>> {
    if weights.is_empty() || weights.len() > MAX_SHARED {
        return Ok(None);
    }
    let Some(plans) = weights
        .iter()
        .map(|w| plan(x, w).filter(|p| p.kind == Kind::Fast))
        .collect::<Option<Vec<Plan>>>()
    else {
        return Ok(None);
    };
    run_fast(x, &plans, weights).map(Some)
}

/// MLX's own quantized matvec, transcribed (`kernels/quantized.h`, MLX 0.32): the helpers its
/// `qmv_fast` and `qmv_wide` run, so a kernel built on them adds in the order `quantized_matmul`
/// and `gather_qmm` do. A `qmv_fast` row is one simdgroup's lanes, each 16 values of every 512
/// (4-bit weights), summed per lane in `qdot`'s order and then across lanes with `simd_sum`;
/// changing that partition or order changes the low bits.
///
/// Each kernel takes only the pieces it calls, and the Metal text carries no comments: MLX copies
/// and hashes a run-time kernel's whole source on every call.
///
/// `QMV_F32`: `load_vector` over f32 activations (each quad summed left to right, the lanes'
/// values pre-divided by 16, 256, 4096), and `qdot` for 4-bit weights, 16 values.
pub(crate) const QMV_F32: &str = r#"
inline float load_vector_f32(const device float* x, thread float* x_thread) {
    float sum = 0;
    for (int i = 0; i < 16; i += 4) {
        sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
        x_thread[i] = x[i];
        x_thread[i + 1] = x[i + 1] / 16.0f;
        x_thread[i + 2] = x[i + 2] / 256.0f;
        x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
    return sum;
}
inline float qdot4(const device uint16_t* ws, const thread float* x_thread, float scale, float bias, float sum) {
    float accum = 0;
    for (int i = 0; i < 4; i++) {
        accum +=
            (x_thread[4 * i] * (ws[i] & 0x000f) +
             x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
             x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
             x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
    return scale * accum + sum * bias;
}
"#;

/// `load_vector` over bf16 activations. The composed path casts the activation to bf16 before the
/// gather, so `x` (f32 here) is rounded first; the quad sum then rounds to bf16 after every add, as
/// `sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3]` does on bf16 operands. `RB` rounds through bf16.
pub(crate) const QMV_BF16: &str = r#"
#define RB(v) static_cast<float>(static_cast<bfloat16_t>(v))
inline float load_vector_bf16(const device float* x, thread float* x_thread) {
    float sum = 0;
    for (int i = 0; i < 16; i += 4) {
        const float a = RB(x[i]);
        const float b = RB(x[i + 1]);
        const float c = RB(x[i + 2]);
        const float d = RB(x[i + 3]);
        sum += RB(RB(RB(a + b) + c) + d);
        x_thread[i] = a;
        x_thread[i + 1] = b / 16.0f;
        x_thread[i + 2] = c / 256.0f;
        x_thread[i + 3] = d / 4096.0f;
    }
    return sum;
}
"#;

/// `load_vector` and `qdot` for 8-bit weights over f32 activations, `V` values a lane (`V` is 8
/// under `qmv_fast`, 4 under `qmv`).
pub(crate) const QMV_8BIT: &str = r#"
template <int V>
inline float load_vector8(const device float* x, thread float* x_thread) {
    float sum = 0;
    for (int i = 0; i < V; i++) {
        sum += x[i];
        x_thread[i] = x[i];
    }
    return sum;
}
template <int V>
inline float qdot8(const device uint8_t* w, const thread float* x_thread, float scale, float bias, float sum) {
    float accum = 0;
    for (int i = 0; i < V; i++) {
        accum += x_thread[i] * w[i];
    }
    return scale * accum + sum * bias;
}
"#;

/// `dequantize<float, 8, 4>` (`qmv_wide`'s): eight 4-bit values from four bytes, `scale * q + bias`.
const QMV_WIDE: &str = r#"
inline void dequantize4x8(const device uint8_t* w, float scale, float bias, thread float* w_local) {
    const float s = float(scale);
    const float b = float(bias);
    float sc[2] = {s, s / 16.0f};
    for (int i = 0; i < 4; i++) {
        w_local[2 * i] = static_cast<float>(sc[0] * (w[i] & 0x0f) + b);
        w_local[2 * i + 1] = static_cast<float>(sc[1] * (w[i] & 0xf0) + b);
    }
}
"#;

#[cfg(test)]
mod tests {
    use mlx_rs::random;

    use super::*;
    use crate::native_quant::{quantize_with_mode, quantized_matmul_with_mode};

    struct Projection {
        w: Array,
        s: Array,
        b: Array,
    }

    impl Projection {
        fn weights(&self) -> Weights<'_> {
            self.weights_with(&self.s, &self.b)
        }

        fn weights_with<'a>(&'a self, s: &'a Array, b: &'a Array) -> Weights<'a> {
            Weights {
                weight: &self.w,
                scales: s,
                biases: Some(b),
                group_size: 64,
                bits: 4,
                mode: MODE_AFFINE,
            }
        }
    }

    /// An affine 4-bit group-64 `[rows, inputs]` projection quantized from bf16 weights, so its
    /// scales and biases are bf16 as a checkpoint's are.
    fn projection(rows: i32, inputs: i32, seed: u64) -> Projection {
        random::seed(seed).expect("seed");
        let w = random::normal::<f32>(&[rows, inputs], None, None, None).expect("weights");
        let w = mlx_rs::ops::multiply(&w, Array::from_f32(0.02))
            .and_then(|w| w.as_dtype(Dtype::Bfloat16))
            .expect("bf16 weights");
        let (w, s, b) = quantize_with_mode(&w, 64, 4, MODE_AFFINE).expect("quantize");
        Projection {
            w,
            s,
            b: b.expect("affine biases"),
        }
    }

    fn activation(shape: &[i32], seed: u64) -> Array {
        random::seed(seed).expect("seed");
        random::normal::<f32>(shape, None, None, None).expect("activation")
    }

    fn wide(a: &Array) -> Array {
        a.as_dtype(Dtype::Float32).expect("f32")
    }

    fn host(a: &Array) -> Vec<f32> {
        a.eval().expect("eval");
        a.as_slice::<f32>().to_vec()
    }

    fn mlx_on_widened_scales(x: &Array, p: &Projection) -> Array {
        quantized_matmul_with_mode(
            x,
            &p.w,
            &wide(&p.s),
            Some(&wide(&p.b)),
            /* transpose */ true,
            64,
            4,
            MODE_AFFINE,
        )
        .expect("quantized_matmul")
    }

    fn differing(want: &Array, got: &Array) -> usize {
        assert_eq!(got.shape(), want.shape());
        let (want, got) = (host(want), host(got));
        want.iter()
            .zip(&got)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count()
    }

    #[test]
    fn the_kernels_equal_quantized_matmul_on_widened_scales_bit_for_bit() {
        let generation = gpu()
            .expect("MLX must report this GPU's architecture")
            .generation;
        let mut wide_runs = 0;
        // One block of 512 inputs, then the 35B-A3B's linear-attention and attention projections.
        for (rows, inputs) in [
            (8, 512),
            (8192, 2048),
            (2048, 4096),
            (32, 2048),
            (512, 2048),
        ] {
            let p = projection(rows, inputs, (rows * 7 + inputs) as u64);
            assert_eq!(p.s.dtype(), Dtype::Bfloat16);
            // A decode step, an MTP verify, batches up to past MLX's batch limit (13-15 here).
            for vectors in [1, 2, 3, 5, 6, 7, 11, 12, 14] {
                let x = activation(&[1, vectors, inputs], 1000 + vectors as u64);
                let Some(plan) = plan(&x, &p.weights()) else {
                    let limit = gpu().map(|g| g.batch_limit(inputs, rows));
                    assert!(
                        Some(vectors) >= limit,
                        "{rows}x{inputs}: the kernels must take {vectors} vectors (limit {limit:?})"
                    );
                    continue;
                };
                wide_runs += usize::from(matches!(plan.kind, Kind::Wide { .. }));
                let got = matvec(&plan, &x, &p.weights()).expect("kernel");
                let off = differing(&mlx_on_widened_scales(&x, &p), &got);
                assert_eq!(
                    off, 0,
                    "{rows}x{inputs}, {vectors} vectors ({:?}): {off} outputs differ",
                    plan.kind
                );
            }
        }
        // From generation 15 MLX runs several vectors as `qmv_wide`, so the wide kernel ran here.
        assert_eq!(wide_runs > 0, generation >= 15);
    }

    /// One launch over the projections of one activation gives each projection's own output: the
    /// 35B-A3B's linear-attention input projections (four), attention's q/k/v (three), a dense
    /// MLP's gate and up (two), and one alone.
    #[test]
    fn a_shared_launch_equals_each_projection_alone_bit_for_bit() {
        let groups: [&[(i32, i32)]; 4] = [
            &[(8192, 2048), (4096, 2048), (32, 2048), (32, 2048)],
            &[(8192, 2048), (512, 2048), (512, 2048)],
            &[(1024, 1024), (1024, 1024)],
            &[(64, 1536)],
        ];
        for (g, shapes) in groups.iter().enumerate() {
            let ps: Vec<Projection> = shapes
                .iter()
                .enumerate()
                .map(|(i, &(rows, inputs))| projection(rows, inputs, (10 * g + i) as u64))
                .collect();
            let weights: Vec<Weights<'_>> = ps.iter().map(Projection::weights).collect();
            let x = activation(&[1, 1, shapes[0].1], 50 + g as u64);
            let got = matvec_shared(&x, &weights)
                .expect("kernel")
                .expect("the shared launch must take one vector of these projections");
            assert_eq!(got.len(), ps.len());
            for (p, (y, &(rows, inputs))) in ps.iter().zip(got.iter().zip(shapes.iter())) {
                let off = differing(&mlx_on_widened_scales(&x, p), y);
                assert_eq!(off, 0, "group {g}, {rows}x{inputs}: {off} outputs differ");
            }
        }
        // Several vectors on a GPU that runs them as `qmv_wide` are not the shared launch's.
        let ps = [projection(512, 2048, 1), projection(32, 2048, 2)];
        let weights = [ps[0].weights(), ps[1].weights()];
        let x = activation(&[1, 3, 2048], 3);
        let shared = matvec_shared(&x, &weights).expect("kernel");
        assert_eq!(
            shared.is_some(),
            gpu().is_some_and(|g| g.kind(3, 2048, 512) == Some(Kind::Fast))
        );
        // Neither are projections of different input lengths, nor more than four.
        let mixed = [projection(64, 512, 4), projection(64, 1024, 5)];
        let x = activation(&[1, 1, 512], 6);
        assert!(
            matvec_shared(&x, &[mixed[0].weights(), mixed[1].weights()])
                .expect("kernel")
                .is_none()
        );
        let five: Vec<Projection> = (0..5).map(|i| projection(8, 512, 20 + i)).collect();
        let five: Vec<Weights<'_>> = five.iter().map(Projection::weights).collect();
        assert!(matvec_shared(&x, &five).expect("kernel").is_none());
    }

    /// Several vectors as `qmv_fast` (MLX's choice below generation 15) equal MLX's single-vector
    /// matvec of each, so that path holds on GPUs this machine does not have.
    #[test]
    fn several_vectors_through_qmv_fast_equal_one_vector_at_a_time() {
        let (rows, inputs, vectors) = (512, 2048, 6);
        let p = projection(rows, inputs, 11);
        let x = activation(&[1, vectors, inputs], 12);
        let plan = Plan {
            vectors,
            inputs,
            outputs: rows,
            kind: Kind::Fast,
        };
        let got = matvec(&plan, &x, &p.weights()).expect("kernel");
        let rows_of_x = mlx_rs::ops::split(&x, vectors, 1).expect("split");
        let one_by_one: Vec<Array> = rows_of_x
            .iter()
            .map(|xi| mlx_on_widened_scales(xi, &p))
            .collect();
        let want = mlx_rs::ops::concatenate_axis(&one_by_one, 1).expect("concatenate");
        assert_eq!(differing(&want, &got), 0);
    }

    #[test]
    fn the_gpu_rule_is_mlx_dispatch() {
        let g = |a: &str| Gpu::parse(a).expect("architecture");
        assert_eq!(
            g("applegpu_g15s"),
            Gpu {
                generation: 15,
                size: b's'
            }
        );
        assert_eq!(g("applegpu_g13p").generation, 13);
        assert_eq!(g("gpu_gxys").generation, 0);
        assert_eq!(Gpu::parse("ab"), None);
        // get_qmv_batch_limit's table: small, medium and large projections.
        let limits = |a: &str| {
            let gpu = g(a);
            [
                gpu.batch_limit(2048, 2048),
                gpu.batch_limit(4096, 2048),
                gpu.batch_limit(5120, 17408),
            ]
        };
        assert_eq!(limits("applegpu_g17s"), [33, 25, 13]);
        assert_eq!(limits("applegpu_g15s"), [13, 15, 13]);
        assert_eq!(limits("applegpu_g16d"), [32, 18, 12]);
        assert_eq!(limits("applegpu_g14s"), [14, 10, 6]);
        assert_eq!(limits("applegpu_g12d"), [32, 18, 12]);
        assert_eq!(limits("applegpu_g12p"), [18, 12, 10]);
        // qmv_wide from generation 15, in tiles of at most five, as even as they can be.
        let m3 = g("applegpu_g15s");
        let kinds: Vec<Option<Kind>> = [1, 2, 5, 6, 7, 11, 12, 13]
            .iter()
            .map(|&m| m3.kind(m, 2048, 2048))
            .collect();
        let wide = |vecs| Some(Kind::Wide { vecs });
        assert_eq!(
            kinds,
            [
                Some(Kind::Fast),
                wide(2),
                wide(5),
                wide(3),
                wide(4),
                wide(4),
                wide(4),
                None
            ]
        );
        assert_eq!(g("applegpu_g14s").kind(3, 2048, 2048), Some(Kind::Fast));
    }

    #[test]
    fn projections_the_kernels_cannot_run_stay_on_quantized_matmul() {
        let p = projection(16, 1024, 1);
        let x = activation(&[1, 1, 1024], 2);
        let (ws, wb) = (wide(&p.s), wide(&p.b));
        assert!(plan(&x, &p.weights()).is_some());
        assert!(serves(&p.weights()));
        // As many vectors as MLX runs as a matrix (prefill).
        assert!(plan(&activation(&[1, 64, 1024], 3), &p.weights()).is_none());
        // A bf16 activation meets bf16 scales without a cast.
        let x_bf16 = x.as_dtype(Dtype::Bfloat16).expect("bf16");
        assert!(plan(&x_bf16, &p.weights()).is_none());
        // Widened scales: MLX reads them without a cast too.
        assert!(plan(&x, &p.weights_with(&ws, &wb)).is_none());
        assert!(!serves(&p.weights_with(&ws, &wb)));
        let no_biases = Weights {
            biases: None,
            ..p.weights()
        };
        assert!(plan(&x, &no_biases).is_none());
        // An input length MLX does not run as `qmv_fast`, and rows not a multiple of 8.
        let q = projection(16, 768, 4);
        assert!(plan(&activation(&[1, 1, 768], 5), &q.weights()).is_none());
        assert!(!serves(&q.weights()));
        let q = projection(12, 1024, 6);
        assert!(plan(&x, &q.weights()).is_none());
        // Other quantizations.
        for (group_size, bits) in [(32, 4), (64, 8)] {
            let other = Weights {
                group_size,
                bits,
                ..p.weights()
            };
            assert!(plan(&x, &other).is_none());
        }
    }

    /// Bench: one matvec's time and GB/s, MLX's against the kernel's, over `layers` distinct
    /// projections (more bytes than the system cache), the variants interleaved round by round.
    /// `QMV_BENCH_SHAPES=rowsxinputsxlayers,...`; run with `--release --ignored --nocapture`.
    #[test]
    #[ignore = "bench"]
    fn decode_matvec_bandwidth() {
        use std::time::Instant;
        let shapes: Vec<(i32, i32, usize)> = std::env::var("QMV_BENCH_SHAPES")
            .unwrap_or_else(|_| "17408x5120x8,5120x17408x8,10240x5120x12,8192x2048x32".into())
            .split(',')
            .map(|s| {
                let v: Vec<i64> = s.split('x').map(|t| t.parse().expect("num")).collect();
                (v[0] as i32, v[1] as i32, v[2] as usize)
            })
            .collect();
        for (rows, inputs, layers) in shapes {
            let ps: Vec<Projection> = (0..layers)
                .map(|i| projection(rows, inputs, i as u64))
                .collect();
            let wides: Vec<(Array, Array)> = ps
                .iter()
                .map(|p| {
                    let (s, b) = (wide(&p.s), wide(&p.b));
                    s.eval().expect("eval");
                    b.eval().expect("eval");
                    (s, b)
                })
                .collect();
            let x = activation(&[1, 1, inputs], 7);
            let x_bf16 = x.as_dtype(Dtype::Bfloat16).expect("bf16");
            x.eval().expect("eval");
            x_bf16.eval().expect("eval");
            let one = Plan {
                vectors: 1,
                inputs,
                outputs: rows,
                kind: Kind::Fast,
            };
            let packed = (rows as f64) * (inputs as f64) / 2.0;
            let groups = (rows as f64) * (inputs as f64) / 64.0;
            let names = [
                "mlx f32 scales",
                "kernel bf16 scales",
                "kernel f32 scales",
                "mlx bf16 x+scales",
            ];
            let bytes = [
                packed + groups * 8.0,
                packed + groups * 4.0,
                packed + groups * 8.0,
                packed + groups * 4.0,
            ];
            let mut best = [f64::MAX; 4];
            for _round in 0..12 {
                for (v, best_v) in best.iter_mut().enumerate() {
                    let t = Instant::now();
                    let mut outs = Vec::with_capacity(layers);
                    for (p, (s, b)) in ps.iter().zip(&wides) {
                        let y = match v {
                            0 => quantized_matmul_with_mode(
                                &x,
                                &p.w,
                                s,
                                Some(b),
                                true,
                                64,
                                4,
                                MODE_AFFINE,
                            ),
                            1 => matvec(&one, &x, &p.weights()),
                            2 => matvec(&one, &x, &p.weights_with(s, b)),
                            _ => quantized_matmul_with_mode(
                                &x_bf16,
                                &p.w,
                                &p.s,
                                Some(&p.b),
                                true,
                                64,
                                4,
                                MODE_AFFINE,
                            ),
                        }
                        .expect("matvec");
                        outs.push(y);
                    }
                    mlx_rs::transforms::eval(outs.iter()).expect("eval");
                    *best_v = best_v.min(t.elapsed().as_secs_f64() / layers as f64);
                }
            }
            for v in 0..4 {
                eprintln!(
                    "[qmv-bench] {rows}x{inputs} {:<20} {:7.1} us  {:6.1} GB/s",
                    names[v],
                    best[v] * 1e6,
                    bytes[v] / best[v] / 1e9
                );
            }
        }
    }
}
