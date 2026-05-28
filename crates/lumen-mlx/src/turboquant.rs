//! TurboQuant — KV cache pre-quantization rotation lever.
//!
//! Implements the rotation half of the TurboQuant pipeline (Google ICLR 2026
//! paper). Apply a Haar-random orthogonal D×D rotation to K (and Q symmetrically)
//! before quantization.
//!
//! Math (orthogonality preserves inner products):
//!   <Q, K> = <Q @ R, K @ R> = <Q_r, K_r>
//! so caching K_r (quantized) + dequant + Q_r in SDPA gives the same scores
//! as Q · K (modulo quant noise).
//!
//! **2026-05-17 finding (rotation + mlx affine = no-go)**:
//! TurboQuant's rotation gain assumes a **fixed-codebook quantizer**
//! (Lloyd-Max for N(0,1)). Rotating non-Gaussian K → Gaussian K, then
//! quantizing against a fixed N(0,1) codebook, reduces per-element error.
//!
//! mlx's affine quant is the opposite — it's **adaptive per-group min/max**.
//! Structured K (per-coordinate magnitude variation) is *easier* for affine
//! quant because each 32-element group adapts to its narrow value range.
//! Rotation Gaussianizes (= homogenizes) groups → every group has the same
//! wide range → per-element precision drops → quality worse, not better.
//!
//! Smoke test 2026-05-17 (prompt=128, 4-bit gs=32):
//!   no-rotation: 11 clean arithmetic tokens, mild drift after
//!   bf16 rotation: degenerates at token 5 (orthogonality ε~1/128 × 25 layers)
//!   f32 rotation: 11 tokens match no-rotation, then degenerates to 144×N
//!     attractor (proving orthogonality is fine; affine adaptivity is what's
//!     being broken)
//!
//! **To make rotation pay off, swap mlx affine for Lloyd-Max**:
//!   - Custom Metal kernel: per-element binary search against
//!     codebook boundaries [n_levels+1] → uint8/uint4 code.
//!   - Custom Metal kernel: gather centroids[codes] * per-token σ
//!     for dequant.
//!   - Optional Stage 2: QJL 1-bit residual + popcount correction kernel.
//!
//! That's multi-session work. The rotation matrix builder + apply helper
//! below are kept as scaffolding for that future implementation.
//!
//! Env: `LUMEN_GEMMA4_QUANT_KV_SLIDING_ROTATE=1` (default OFF — currently
//! NEGATIVE due to the affine/Lloyd-Max mismatch above).
//!
//! ──────────────────────────────────────────────────────────────────────────
//! TurboQuant Stage-1 cache path (`LUMEN_GEMMA4_QUANT_KV_SLIDING_TURBOQUANT=1`)
//! ──────────────────────────────────────────────────────────────────────────
//!
//! Status (2026-05-17): **infrastructure complete, perf/quality not yet
//! competitive; multi-session kernel work required.**
//!
//! Architecture LANDED in this session:
//!   - `rotation_matrix_f32()` — Haar orthogonal R [D,D] from lumen-core,
//!     cached OnceLock per (dim, seed)
//!   - `lloyd_max_centroids()` — fixed N(0,1) codebook from lumen-core
//!     1000-iter EM, cached OnceLock per bit width
//!   - `lloyd_max_quantize_stage1(x, centroids)` → `(codes uint8, σ f32)`
//!     via per-vector σ + argmin-broadcast against centroids
//!   - `lloyd_max_dequantize_scaled(codes, σ, centroids)` → bf16 via
//!     `take(centroids, codes) * σ`
//!   - `NativeRotatingKvCacheTurboQuant` — 4-buffer ring (K codes+σ,
//!     V codes+σ), in-place slice_update fast path mirroring the bf16 cache
//!   - `SlidingTurboquant` enum variant + SDPA inline branch (rotate Q,
//!     dequant K/V, bf16 SDPA, no inverse rotation since V stays in original)
//!
//! Smoke test 2026-05-17 (prompt=128, 20 decode steps):
//!   bits=8 → clean tokens, matches affine baseline (563, 570, 577, …)
//!   bits=6 → clean tokens, mild periodicity
//!   bits=5 → degenerates to short cycle (108, 236811, 640, 647, 661, …)
//!   bits=4 → total degeneration ([236743, 107] alternation)
//!   bits=3 → total degeneration
//!
//! 8K perf test 2026-05-17 (bits=6, lowest "clean" precision):
//!   prefill: 916 → 149 tok/s  = −84% (argmin-broadcast catastrophic)
//!   decode : 56.6 → 19.7 tok/s = −65%
//!
//! ── Why Stage 1 alone falls short ──
//!
//! 1. **No QJL Stage 2 unbiased correction**: Lloyd-Max stage 1 is biased
//!    per-element. Bias compounds across 25 sliding layers and pushes
//!    attention scores off-distribution. The paper's 0.9945 cos sim @ 3-bit
//!    figure includes the Stage-2 QJL correction; Stage 1 alone is
//!    insufficient at low bits with deep stacks.
//!
//! 2. **Argmin-broadcast is compute-prohibitive**: encode at bits=6 builds
//!    a [B, n_kv, T, D, n_levels=64] intermediate per layer per step. At
//!    8K prefill that's ~1 GB per layer — argmin walks the whole thing.
//!    Slow even after f32→bf16 plumbing optimizations.
//!
//! 3. **V quantization no rotation = non-Gaussian → poor Lloyd-Max fit**:
//!    V's per-coordinate distribution is generally NOT N(0,1) without
//!    rotation. Lloyd-Max codebook is optimal for N(0,1), so V quant has
//!    higher error than affine quant would for the same V.
//!
//! ── Path to production ──
//!
//! For TurboQuant to beat mlx affine 4-bit gs=32, all four are required
//! (~1 week focused work):
//!
//!   (a) **Metal kernel: Lloyd-Max binary-search encode** — replaces
//!       argmin-broadcast. Per element ~4 compares for 4-bit (log2 16).
//!       Memory: O(B·n_kv·T·D) instead of O(B·n_kv·T·D·n_levels).
//!       Compute: ~16× less than argmin path.
//!
//!   (b) **Metal kernel: QJL Stage-2 (1-bit projection + popcount
//!       correction)** — adds unbiased estimator on top of Stage 1. This
//!       is what makes 3-bit / 2-bit operationally viable.
//!
//!   (c) **4-bit packing in storage** — currently codes are uint8
//!       unpacked (2× compression). Packing to uint4 gives ~3.5×.
//!
//!   (d) **Fused dequant + SDPA kernel** — avoid materializing the
//!       intermediate bf16 K_dq buffer (saves BW + memory). flash-attn
//!       style: dequant on the fly within the kernel.
//!
//! Without these, the current SlidingTurboquant path is a reference impl
//! useful for verifying the math and Stage 1 quality, not for production.
//!
//! ── Recommended user-facing posture ──
//!
//! - **Production**: `LUMEN_GEMMA4_QUANT_KV_SLIDING=1` (mlx affine path)
//!   → NEUTRAL @ 8K, +8% @ 12K, +68~199% @ 16K verified.
//! - **Experimental**: `LUMEN_GEMMA4_QUANT_KV_SLIDING_TURBOQUANT=1` only
//!   for those who want to validate the math. Default bits=4 will
//!   produce garbage; use bits=8 for sanity, bits=6 for "lowest clean"
//!   but with severe perf regression.
//!
//! ──────────────────────────────────────────────────────────────────────────
//! 2026-05-17 update — Lloyd-Max encode Metal kernel LANDED
//! ──────────────────────────────────────────────────────────────────────────
//!
//! `lumen_tq_encode` Metal kernel (mlx fork + mlx-c + mlx-rs + lumen-mlx
//! end-to-end wiring) replaces the argmin-broadcast encode. Per-thread
//! linear scan over inner boundaries loaded into threadgroup memory.
//!
//! Perf 8K decode 2026-05-17:
//!   argmin path (last session) : 19.7 tok/s  (−66% vs OFF 58.2)
//!   kernel  path (this session): 40.5 tok/s  (−30% vs OFF 58.2)
//!
//! Speedup of the kernel itself: 2.1× decode / 4.4× prefill vs argmin.
//!
//! Remaining gap to OFF (≈ −30% decode) is dispatch overhead in the
//! surrounding op chain — per quantize call we still pay:
//!   cast bf16→f32 → square → sum_axis → divide(D) → sqrt → divide(σ)
//!   → cast bf16 → kernel → (...) dequant: take → multiply → cast bf16
//! That's ~8 dispatches per K and per V per layer per step = 400/step.
//! At ~50 μs each, ~20 ms decode overhead on top of compute.
//!
//! ── Next levers (decreasing ROI) ──
//!
//!   (1) **Fuse encode** — LANDED, NEUTRAL-positive. `lumen_tq_encode_fused`
//!       Metal kernel collapses (cast → square → sum → divide → sqrt →
//!       divide → cast → encode = 8 dispatches) into one threadgroup-per-row
//!       kernel that computes σ via simdgroup reduce, normalizes in-register,
//!       and emits codes + σ. Env gate `LUMEN_GEMMA4_TQ_FUSED_ENCODE=1`
//!       (default ON), `=0` for A/B vs non-fused.
//!
//!       Predicted: ~12.5 ms decode savings (8 dispatches × ~50 µs × 25
//!       layers × 2 K+V).
//!       Actual (8K, bits=6, STEPS=64, 2 trials):
//!           fused    39.2 ± 0.1 tok/s  (25.50 ms/step)
//!           non-fused 38.4 ± 0.1 tok/s  (26.05 ms/step)
//!           Δ = +2.1% decode (~0.55 ms saved); prefill +1.2%
//!
//!       Why the predicted savings didn't materialize: mlx already
//!       async-pipelines simple elementwise op chains (cast/multiply/sum/
//!       sqrt) across the compute encoder, hiding most of the dispatch
//!       overhead behind concurrent layer work. Matches the Tier-2C
//!       pattern (dense_mlp norm fuse NEUTRAL — async absorbs trivial
//!       fusions). The kernel is still landed because (a) it's the right
//!       structural primitive for follow-up work (QJL Stage 2, packed
//!       codes, fused quant-SDPA all want the σ + codes pair as an atomic
//!       unit), and (b) the small win is real and consistent.
//!
//!   (2) **Fuse dequant** into a single kernel that gathers centroids[codes]
//!       and multiplies by σ, emitting bf16 K. Replaces 3 dispatches with 1.
//!       Expected: same ~NEUTRAL outcome as (1) per Tier-2C — mlx already
//!       async-pipelines take + multiply + cast. ROI now downgraded;
//!       defer to (5) which subsumes dequant.
//!
//!   (3) **QJL Stage-2 kernel** (1-bit projection + popcount correction).
//!       Unlocks clean output at 3-bit / 2-bit → 5×/8× compression vs bf16.
//!       The 4-bit / 3-bit degeneration we see is Stage-1 bias accumulating
//!       over 25 layers; Stage 2 is unbiased.
//!
//!   (4) **4-bit code packing** kernel (uint8 → packed uint4 in storage).
//!       2× memory reduction (~10 MB saved on sliding KV).
//!
//!   (5) **Fused dequant + windowed SDPA** for prefill. The −22% decode
//!       gap remaining vs OFF is now traced to dequant K/V buffer
//!       materialization + bf16 SDPA bandwidth (not encode dispatch).
//!       Building a quant-aware variant of `lumen_sdpa_windowed` that
//!       reads codes + σ in the kernel and dequantizes on the fly within
//!       register tiles would (a) avoid materializing the dequant
//!       buffers and (b) recover the (L−W)/L block skip win for the
//!       quant cache path. **Now top ROI item.**
//!
//! Updated combined impact estimate after (1) LANDED: NEUTRAL gain from
//! kernel fusion alone (Tier-2C). To reach parity with the OFF path, (5)
//! is required — it's the real lever for the remaining 22% gap.

#[cfg(feature = "mlx-native")]
use anyhow::{Context, Result, anyhow};
#[cfg(feature = "mlx-native")]
use mlx_rs::{Array, Dtype};
#[cfg(feature = "mlx-native")]
use std::collections::HashMap;
#[cfg(feature = "mlx-native")]
use std::sync::Mutex;
#[cfg(feature = "mlx-native")]
use std::sync::OnceLock;

#[cfg(feature = "mlx-native")]
static ROTATION_CACHE_F32: OnceLock<Mutex<HashMap<(usize, u64), Array>>> = OnceLock::new();

/// Get or build a Haar-distributed `[dim, dim]` orthogonal matrix as an f32
/// `Array`. Cached per `(dim, seed)` so repeated calls cost a HashMap lookup.
///
/// **Stored as f32, not bf16**: orthogonality `R R^T = I` only holds to the
/// matrix dtype's precision. bf16 has 7-bit mantissa → ||R R^T − I|| ≈ 1/128.
/// 25 sliding layers compound this into ~20% score perturbation. f32 reduces
/// the orthogonality residual to ~1e-7, negligible after 25 layers.
#[cfg(feature = "mlx-native")]
pub fn rotation_matrix_f32(dim: usize, seed: u64) -> Result<Array> {
    let cache = ROTATION_CACHE_F32.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache
        .lock()
        .map_err(|e| anyhow!("turboquant: rotation cache mutex poisoned: {e}"))?;
    if let Some(arr) = guard.get(&(dim, seed)) {
        return Ok(arr.clone());
    }
    let r = lumen_core::rotation::RotationMatrix::random(dim, seed);
    if r.dim != dim {
        return Err(anyhow!(
            "turboquant: RotationMatrix returned dim={} but expected {}",
            r.dim,
            dim
        ));
    }
    let arr = Array::from_slice(&r.matrix, &[dim as i32, dim as i32]);
    guard.insert((dim, seed), arr.clone());
    Ok(arr)
}

/// Apply rotation: `x @ R`. Last axis of `x` must equal `R.shape[0]`.
/// Casts `x` (bf16) → f32 for the matmul to preserve orthogonality precision,
/// then back to bf16 for downstream consumers (cache + SDPA).
///
/// NEGATIVE (2026-05-22): tried removing the bf16→f32 input cast and
/// passing bf16 directly into `matmul(bf16, f32)` — MLX accepts mixed
/// dtype and the output is bit-identical, but the decode-step wall time
/// REGRESSED ~7-16% with variance jumping 14×. Hypothesis: Apple MPS /
/// Metal GEMM has fast kernels for matching-dtype inputs (f32×f32 here);
/// mixed-dtype falls back to a slower generic path. The explicit cast is
/// the "setup for fast kernel" pattern, NOT a redundant dispatch. Keep
/// the cast; do not retry without a fused custom Metal kernel.
#[cfg(feature = "mlx-native")]
pub fn rotate_last_axis(x: &Array, rotation_f32: &Array) -> Result<Array> {
    let x_f32 = x
        .as_dtype(Dtype::Float32)
        .context("turboquant: cast input to f32 for rotation")?;
    let rotated =
        mlx_rs::ops::matmul(&x_f32, rotation_f32).context("turboquant: rotate (matmul with R)")?;
    rotated
        .as_dtype(Dtype::Bfloat16)
        .context("turboquant: cast rotated output back to bf16")
}

/// Default seed for TurboQuant rotation — matches lumen-core's
/// `TurboQuantConfig::default().seed`. Constant so K-rotation and Q-rotation
/// across SDPA use the same matrix.
pub const TURBOQUANT_SEED: u64 = 42;

// ───────────────────────── Lloyd-Max codebook (fixed N(0,1)) ─────────────────────────

#[cfg(feature = "mlx-native")]
static CENTROIDS_CACHE: OnceLock<Mutex<HashMap<u32, Array>>> = OnceLock::new();

#[cfg(feature = "mlx-native")]
static BOUNDARIES_CACHE: OnceLock<Mutex<HashMap<u32, Array>>> = OnceLock::new();

/// Get or build the Lloyd-Max centroids `[n_levels]` (f32 Array) for the
/// given bit width. Centroids are the conditional means E[X | bin_i] under
/// N(0,1), precomputed by lumen-core via 1000-iter EM.
///
/// `bits` ∈ [2, 8]; n_levels = 2^bits ≤ 256.
#[cfg(feature = "mlx-native")]
pub fn lloyd_max_centroids(bits: u32) -> Result<Array> {
    let cache = CENTROIDS_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache
        .lock()
        .map_err(|e| anyhow!("turboquant: centroids cache mutex poisoned: {e}"))?;
    if let Some(arr) = guard.get(&bits) {
        return Ok(arr.clone());
    }
    let codebook = lumen_core::lloyd_max::LloydMaxCodebook::compute(bits, 1000)
        .map_err(|e| anyhow!("turboquant: lloyd-max codebook compute (bits={bits}): {e}"))?;
    let centroids_f32: Vec<f32> = codebook.centroids.iter().map(|&x| x as f32).collect();
    let arr = Array::from_slice(&centroids_f32, &[codebook.centroids.len() as i32]);
    guard.insert(bits, arr.clone());
    Ok(arr)
}

/// Get or build the Lloyd-Max inner boundaries `[n_levels - 1]` (f32 Array)
/// for the given bit width. Drops the `-INF` and `+INF` endpoints that
/// `lumen_core::LloydMaxCodebook.boundaries` includes (length n_levels+1).
///
/// Used by the `lumen_tq_encode` Metal kernel for per-element binary scan.
#[cfg(feature = "mlx-native")]
pub fn lloyd_max_boundaries_inner(bits: u32) -> Result<Array> {
    let cache = BOUNDARIES_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache
        .lock()
        .map_err(|e| anyhow!("turboquant: boundaries cache mutex poisoned: {e}"))?;
    if let Some(arr) = guard.get(&bits) {
        return Ok(arr.clone());
    }
    let codebook = lumen_core::lloyd_max::LloydMaxCodebook::compute(bits, 1000)
        .map_err(|e| anyhow!("turboquant: lloyd-max codebook compute (bits={bits}): {e}"))?;
    // Drop endpoints (boundaries[0] = -INF, boundaries[n_levels] = +INF).
    let inner: Vec<f32> = codebook
        .boundaries
        .iter()
        .skip(1)
        .take(codebook.boundaries.len() - 2)
        .map(|&x| x as f32)
        .collect();
    let arr = Array::from_slice(&inner, &[inner.len() as i32]);
    guard.insert(bits, arr.clone());
    Ok(arr)
}

/// Encode `x` (f32) into Lloyd-Max codes via nearest-centroid argmin.
/// Returns codes as uint8 (assumes n_levels ≤ 256 → bits ≤ 8).
///
/// Implementation: broadcast subtract → abs → argmin along centroid axis.
/// Memory intermediate is `numel(x) × n_levels × 4 bytes`. Caller must chunk
/// large inputs (e.g. prefill of 8K tokens) to avoid GPU OOM.
#[cfg(feature = "mlx-native")]
pub fn lloyd_max_encode(x: &Array, centroids: &Array) -> Result<Array> {
    let n_levels = centroids.shape()[0];
    // x [..., D] → [..., D, 1]; centroids [n_levels] reshape → [..., 1, n_levels]
    let x_expanded = mlx_rs::ops::expand_dims(x, -1)
        .context("turboquant: lloyd_max_encode expand_dims(x, -1)")?;
    // Build a shape for centroids matching x's rank + 1.
    let mut c_shape = vec![1i32; x.ndim() + 1];
    *c_shape.last_mut().unwrap() = n_levels;
    let c_reshaped = mlx_rs::ops::reshape(centroids, &c_shape)
        .context("turboquant: lloyd_max_encode reshape centroids")?;
    let diff = mlx_rs::ops::subtract(&x_expanded, &c_reshaped)
        .context("turboquant: lloyd_max_encode subtract")?;
    let abs_diff = mlx_rs::ops::abs(&diff).context("turboquant: lloyd_max_encode abs")?;
    let last_axis = (abs_diff.ndim() as i32) - 1;
    let codes_i32 =
        mlx_rs::ops::indexing::argmin_axis(&abs_diff, last_axis, /* keepdims */ false)
            .context("turboquant: lloyd_max_encode argmin")?;
    codes_i32
        .as_dtype(Dtype::Uint8)
        .context("turboquant: lloyd_max_encode cast to uint8")
}

/// Decode `codes` (uint8) → reconstructed f32 values via `take(centroids, codes)`.
#[cfg(feature = "mlx-native")]
pub fn lloyd_max_decode(codes: &Array, centroids: &Array) -> Result<Array> {
    // mlx::take(values_1d, indices) gathers from flattened values. centroids
    // is 1D [n_levels]; codes has shape [..., D]. Result mirrors codes' shape.
    mlx_rs::ops::indexing::take(centroids, codes)
        .context("turboquant: lloyd_max_decode take(centroids, codes)")
}

/// Per-vector scale σ = ||x|| / √D along the last axis (operating in f32).
/// Returns shape `[..., 1]` f32 so it broadcasts back for normalize / denormalize.
/// Caller is responsible for casting `x` to f32 first if needed (we don't
/// cast here to avoid a redundant copy when callers already have f32).
#[cfg(feature = "mlx-native")]
pub fn per_vector_sigma(x_f32: &Array) -> Result<Array> {
    let last_axis = (x_f32.ndim() as i32) - 1;
    let d = x_f32.shape().last().copied().unwrap_or(1) as f32;
    let x_sq = mlx_rs::ops::multiply(x_f32, x_f32).context("turboquant: sigma square")?;
    let sum_sq = mlx_rs::ops::sum_axis(&x_sq, last_axis, /* keepdims */ true)
        .context("turboquant: sigma sum")?;
    let mean_sq =
        mlx_rs::ops::divide(&sum_sq, &Array::from_f32(d)).context("turboquant: sigma divide")?;
    mlx_rs::ops::sqrt(&mean_sq).context("turboquant: sigma sqrt")
}

/// One-shot TurboQuant encode (Stage 1 only — no QJL): for a bf16 tensor
/// `x_bf16` of shape `[..., D]`, compute `(codes, sigma)` where
/// - `codes` is uint8 `[..., D]` Lloyd-Max nearest-centroid indices,
/// - `sigma` is f32 `[..., 1]` per-vector scale σ = ||x|| / √D.
///
/// Steps:
///   1. x_f32 = x_bf16.astype(f32)
///   2. σ = ||x_f32|| / √D  per row
///   3. x_norm_bf16 = (x_f32 / σ).astype(bf16)
///   4. codes = lumen_turboquant_encode(x_norm_bf16, boundaries_inner)
///      (GPU kernel: per-thread linear scan; no argmin broadcast)
///
/// Use `lloyd_max_dequantize_scaled` to reverse: `centroids[codes] * σ`.
#[cfg(feature = "mlx-native")]
pub fn lloyd_max_quantize_stage1(x_bf16: &Array, centroids: &Array) -> Result<(Array, Array)> {
    let x_f32 = x_bf16
        .as_dtype(Dtype::Float32)
        .context("turboquant: quantize cast x → f32")?;
    let sigma = per_vector_sigma(&x_f32)?;
    let x_norm_f32 =
        mlx_rs::ops::divide(&x_f32, &sigma).context("turboquant: quantize normalize x / σ")?;
    let x_norm_bf16 = x_norm_f32
        .as_dtype(Dtype::Bfloat16)
        .context("turboquant: cast x_norm to bf16 for kernel")?;
    // Derive bits from centroids length: n_levels = centroids.shape[0].
    let n_levels = centroids.shape()[0] as u32;
    let bits = (n_levels as f32).log2() as u32;
    let boundaries_inner = lloyd_max_boundaries_inner(bits)?;
    let stream = mlx_rs::Stream::gpu();
    let codes = mlx_rs::metal::lumen_turboquant_encode(&x_norm_bf16, &boundaries_inner, &stream)
        .map_err(|e| anyhow!("turboquant: lumen_tq_encode kernel: {e}"))?;
    Ok((codes, sigma))
}

/// Reverse of `lloyd_max_quantize_stage1`: gather centroids by codes, multiply
/// by per-row σ, cast back to bf16 for downstream SDPA.
#[cfg(feature = "mlx-native")]
pub fn lloyd_max_dequantize_scaled(
    codes: &Array,
    sigma_f32: &Array,
    centroids: &Array,
) -> Result<Array> {
    let dequant_f32 = lloyd_max_decode(codes, centroids)?;
    let scaled =
        mlx_rs::ops::multiply(&dequant_f32, sigma_f32).context("turboquant: dequant multiply σ")?;
    scaled
        .as_dtype(Dtype::Bfloat16)
        .context("turboquant: dequant cast back to bf16")
}

// ─────────────────── QJL Stage-2 (1-bit residual correction) ───────────────────
//
// Math (K-side, GQA-aware). Lloyd-Max Stage-1 gives biased K_dq; the residual
//   r_k = K_rot - K_dq
// is encoded as 1-bit signs of a Johnson-Lindenstrauss projection:
//   b_k[i] = sign(g_iᵀ r_k)   with g_i ~ N(0, I_D)  packaged row-i of Φ_unit.
// The QJL unbiased estimator gives:
//   <q, r_k> ≈ ‖r_k‖ · √(π/2)/√m · Σ_i (Φq)_i · b_k[i]
// By bilinearity of the inner product this is equivalent to inserting a
// virtual K-side correction (no Q dependence in the cached term):
//   c_k[d] = ‖r_k‖ · √(π/2)/√m · Σ_i Φ[i,d] · b_k[i]
//          = ‖r_k‖ · √(π/2)/√m · (Φᵀ b_k)[d]
//   K_eff[k] = K_dq[k] + c_k
//   <q, K_eff[k]> = <q, K_dq[k]> + <q, c_k> = <q, K_dq[k]> + <q, r_k>
// So we can recover the unbiased correction with *no* manual SDPA — just
// add a per-K-vector delta to K_dq and call standard SDPA on K_eff.
//
// We store `signs ∈ {-1, +1}` as bf16 directly (no bitpacking) so the
// correction is one matmul `signs @ Φ`. Memory: m=128 → 256 B per K vector
// (vs 16 B packed). A later pass replaces this with a uint8/u64 packed
// layout + dedicated kernel; for now we want the math validated first.
//
// In `QJLProjector` (lumen-core), entries are ~ N(0, 1/m). The estimator
// scale absorbs the 1/m → 1/√m correctly (see qjl.rs comments). Here we
// reuse that matrix and the same scale constant.

#[cfg(feature = "mlx-native")]
static QJL_PROJECTION_CACHE: OnceLock<Mutex<HashMap<(usize, usize, u64), Array>>> = OnceLock::new();

/// QJL projection matrix Φ as a f32 mlx Array shaped `[m, dim]`. Entries
/// ~ N(0, 1/m) — matches `lumen_core::qjl::QJLProjector::new`. Cached per
/// `(dim, m, seed)`.
#[cfg(feature = "mlx-native")]
pub fn qjl_projection_matrix_f32(dim: usize, m: usize, seed: u64) -> Result<Array> {
    let cache = QJL_PROJECTION_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache
        .lock()
        .map_err(|e| anyhow!("turboquant: qjl projection cache mutex poisoned: {e}"))?;
    if let Some(arr) = guard.get(&(dim, m, seed)) {
        return Ok(arr.clone());
    }
    let proj = lumen_core::qjl::QJLProjector::new(dim, m, seed);
    if proj.dim != dim || proj.m != m {
        return Err(anyhow!(
            "turboquant: QJLProjector returned (dim={}, m={}) but expected ({}, {})",
            proj.dim,
            proj.m,
            dim,
            m
        ));
    }
    let arr = Array::from_slice(&proj.proj_matrix, &[m as i32, dim as i32]);
    guard.insert((dim, m, seed), arr.clone());
    Ok(arr)
}

/// QJL correction scale factor: `√(π/2) / √m`. Multiplied into K's residual
/// correction so the unbiased estimator absorbs the projection-matrix scale.
#[inline]
pub fn qjl_correction_scale(m: usize) -> f32 {
    (std::f32::consts::FRAC_PI_2.sqrt()) / (m as f32).sqrt()
}

#[cfg(feature = "mlx-native")]
static QJL_SCALE_F32_CACHE: OnceLock<Mutex<HashMap<usize, Array>>> = OnceLock::new();

/// Cached 0-D f32 `Array` holding `qjl_correction_scale(m)`. Avoids the
/// per-decode-step `Array::from_f32(scale)` FFI allocation inside
/// `qjl_apply_correction_to_k_dq` (called per layer per step under QJL ON).
/// Keyed by `m`; in production there is exactly one entry per process.
#[cfg(feature = "mlx-native")]
fn qjl_correction_scale_array_f32(m: usize) -> Result<Array> {
    let cache = QJL_SCALE_F32_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache
        .lock()
        .map_err(|e| anyhow!("turboquant: qjl scale cache mutex poisoned: {e}"))?;
    if let Some(arr) = guard.get(&m) {
        return Ok(arr.clone());
    }
    let arr = Array::from_f32(qjl_correction_scale(m));
    guard.insert(m, arr.clone());
    Ok(arr)
}

/// QJL Stage-2 encode. Given rotated K and its Stage-1 dequantization,
/// produces:
///   - `signs` bf16 `[..., m]` with values in {-1, +1}, the sign of each
///     projected residual.
///   - `r_norm` f32 `[..., 1]` per-vector residual L2 norm `‖r‖`.
///
/// `projection` is the cached Φ from `qjl_projection_matrix_f32`, shape
/// `[m, D]`. Computation is done in f32 for projection-matmul precision,
/// then signs cast to bf16 ±1.
#[cfg(feature = "mlx-native")]
pub fn qjl_encode_stage2(
    k_rot_bf16: &Array,
    k_dq_bf16: &Array,
    projection: &Array,
) -> Result<(Array, Array)> {
    let k_rot = k_rot_bf16
        .as_dtype(Dtype::Float32)
        .context("qjl_encode_stage2: cast K_rot to f32")?;
    let k_dq = k_dq_bf16
        .as_dtype(Dtype::Float32)
        .context("qjl_encode_stage2: cast K_dq to f32")?;
    let residual = mlx_rs::ops::subtract(&k_rot, &k_dq)
        .context("qjl_encode_stage2: residual = K_rot - K_dq")?;

    // r_norm = sqrt(Σ_d r²) along the last axis (keepdims).
    let last_axis = (residual.ndim() as i32) - 1;
    let r_sq = mlx_rs::ops::multiply(&residual, &residual)
        .context("qjl_encode_stage2: residual square")?;
    let r_sumsq = mlx_rs::ops::sum_axis(&r_sq, last_axis, /* keepdims */ true)
        .context("qjl_encode_stage2: residual sumsq")?;
    let r_norm = mlx_rs::ops::sqrt(&r_sumsq).context("qjl_encode_stage2: residual L2 norm")?;

    // r_proj = residual @ Φᵀ   →   shape [..., m]
    // projection is [m, D] → use transposed in matmul to get @ Φᵀ.
    let proj_t = mlx_rs::ops::transpose_axes(projection, &[1, 0])
        .context("qjl_encode_stage2: projection transpose for matmul")?;
    let r_proj =
        mlx_rs::ops::matmul(&residual, &proj_t).context("qjl_encode_stage2: residual @ Φᵀ")?;

    // signs = where(r_proj >= 0, +1, -1) — kept as bf16 ±1 so downstream
    // correction is one matmul against Φ. We avoid `sign(x)` because it
    // returns 0 for x == 0 (rare with Gaussian residuals but conceptually
    // wrong for an unbiased ±1 estimator).
    let zero = Array::from_f32(0.0);
    let positive = mlx_rs::ops::ge(&r_proj, &zero).context("qjl_encode_stage2: r_proj >= 0")?;
    let plus_one_f32 = Array::from_f32(1.0);
    let minus_one_f32 = Array::from_f32(-1.0);
    let signs_f32 = mlx_rs::ops::r#where(&positive, &plus_one_f32, &minus_one_f32)
        .context("qjl_encode_stage2: where → ±1")?;
    let signs_bf16 = signs_f32
        .as_dtype(Dtype::Bfloat16)
        .context("qjl_encode_stage2: cast signs to bf16")?;

    Ok((signs_bf16, r_norm))
}

/// Number of u32 words required to pack `m` sign bits.
#[inline]
pub fn qjl_packed_words(m: usize) -> usize {
    (m + 31) / 32
}

/// QJL Stage-2 encode (packed). Same math as `qjl_encode_stage2` but
/// returns sign bits packed into u32 words `[..., ceil(m/32)]` instead of
/// bf16 ±1 `[..., m]`. 16× smaller cache footprint.
///
/// Returns `(packed_signs u32 [..., ceil(m/32)], r_norm f32 [..., 1])`.
#[cfg(feature = "mlx-native")]
pub fn qjl_encode_stage2_packed(
    k_rot_bf16: &Array,
    k_dq_bf16: &Array,
    projection: &Array,
) -> Result<(Array, Array)> {
    let k_rot = k_rot_bf16
        .as_dtype(Dtype::Float32)
        .context("qjl_encode_stage2_packed: cast K_rot → f32")?;
    let k_dq = k_dq_bf16
        .as_dtype(Dtype::Float32)
        .context("qjl_encode_stage2_packed: cast K_dq → f32")?;
    let residual = mlx_rs::ops::subtract(&k_rot, &k_dq)
        .context("qjl_encode_stage2_packed: residual = K_rot - K_dq")?;

    let last_axis = (residual.ndim() as i32) - 1;
    let r_sq = mlx_rs::ops::multiply(&residual, &residual)
        .context("qjl_encode_stage2_packed: residual square")?;
    let r_sumsq = mlx_rs::ops::sum_axis(&r_sq, last_axis, true)
        .context("qjl_encode_stage2_packed: residual sumsq")?;
    let r_norm =
        mlx_rs::ops::sqrt(&r_sumsq).context("qjl_encode_stage2_packed: residual L2 norm")?;

    let proj_t = mlx_rs::ops::transpose_axes(projection, &[1, 0])
        .context("qjl_encode_stage2_packed: projection transpose")?;
    let r_proj = mlx_rs::ops::matmul(&residual, &proj_t)
        .context("qjl_encode_stage2_packed: residual @ Φᵀ")?;

    // r_proj shape: [..., m]. Pack into [..., ceil(m/32)] u32 words.
    let m = projection.shape()[0];
    let stream = mlx_rs::Stream::gpu();
    let packed = mlx_rs::metal::lumen_qjl_pack_signs(&r_proj, m, &stream)
        .map_err(|e| anyhow!("turboquant: lumen_qjl_pack_signs kernel: {e}"))?;

    Ok((packed, r_norm))
}

/// Apply QJL Stage-2 correction from packed u32 signs. Internally unpacks
/// to bf16 ±1 (single Metal kernel) and reuses the bf16-path matmul +
/// scale + add. Mathematically identical to `qjl_apply_correction_to_k_dq`
/// — only the storage layout differs.
#[cfg(feature = "mlx-native")]
pub fn qjl_apply_correction_packed(
    k_dq_bf16: &Array,
    packed_signs: &Array,
    r_norm_f32: &Array,
    projection: &Array,
    qjl_m: usize,
) -> Result<Array> {
    let stream = mlx_rs::Stream::gpu();
    let signs_bf16 = mlx_rs::metal::lumen_qjl_unpack_signs(packed_signs, qjl_m as i32, &stream)
        .map_err(|e| anyhow!("turboquant: lumen_qjl_unpack_signs kernel: {e}"))?;
    qjl_apply_correction_to_k_dq(k_dq_bf16, &signs_bf16, r_norm_f32, projection, qjl_m)
}

/// Apply the QJL Stage-2 correction to Stage-1's `K_dq` so the result is an
/// unbiased K reconstruction (in the sense that `<Q, K_eff> ≈ <Q, K>`).
///
///   c_k[d] = ‖r_k‖ · √(π/2)/√m · (Φᵀ b_k)[d]
///   K_eff  = K_dq + c_k
///
/// `signs_bf16_pm1` is `[..., m]` with entries in {-1, +1} (as produced by
/// `qjl_encode_stage2`). `r_norm_f32` is `[..., 1]`. `projection` is `[m, D]`.
#[cfg(feature = "mlx-native")]
pub fn qjl_apply_correction_to_k_dq(
    k_dq_bf16: &Array,
    signs_bf16_pm1: &Array,
    r_norm_f32: &Array,
    projection: &Array,
    qjl_m: usize,
) -> Result<Array> {
    // signs @ Φ   →   [..., D]
    let signs_f32 = signs_bf16_pm1
        .as_dtype(Dtype::Float32)
        .context("qjl_apply_correction: cast signs to f32")?;
    let phi_inv_matmul =
        mlx_rs::ops::matmul(&signs_f32, projection).context("qjl_apply_correction: signs @ Φ")?;

    // Scale by ‖r‖_k × √(π/2)/√m. The scale Array is cached (see
    // `qjl_correction_scale_array_f32`) so we don't pay an `Array::from_f32`
    // host allocation per decode step per layer.
    let scale_arr = qjl_correction_scale_array_f32(qjl_m)?;
    let scaled_norm = mlx_rs::ops::multiply(r_norm_f32, &scale_arr)
        .context("qjl_apply_correction: scale ‖r‖ × √(π/2)/√m")?;
    let correction_f32 = mlx_rs::ops::multiply(&phi_inv_matmul, &scaled_norm)
        .context("qjl_apply_correction: scaled correction")?;
    let correction_bf16 = correction_f32
        .as_dtype(Dtype::Bfloat16)
        .context("qjl_apply_correction: cast correction back to bf16")?;
    mlx_rs::ops::add(k_dq_bf16, &correction_bf16)
        .context("qjl_apply_correction: K_eff = K_dq + correction")
}

/// Fused TurboQuant Stage-1 encode: σ + normalize + Lloyd-Max in one kernel.
///
/// Replaces `lloyd_max_quantize_stage1`'s 8-op chain (`cast → square → sum →
/// divide → sqrt → divide → cast → encode`) with a single dispatch. The Metal
/// kernel runs one threadgroup per row of `[..., D]`, computes σ via
/// simdgroup reduction, normalizes in-register, then emits codes + σ.
///
/// Returns `(codes uint8 [..., D], sigma f32 [..., 1])`. `sigma` matches the
/// keepdims-true shape of `per_vector_sigma` so it broadcasts cleanly in
/// `lloyd_max_dequantize_scaled`.
///
/// Constraints (validated in the C++ factory):
///   - `x_bf16` last-axis D must be a positive multiple of 32, ≤ 1024
///   - `n_inner = n_levels - 1` ∈ [1, 255]
#[cfg(feature = "mlx-native")]
pub fn lloyd_max_quantize_stage1_fused(
    x_bf16: &Array,
    centroids: &Array,
) -> Result<(Array, Array)> {
    let n_levels = centroids.shape()[0] as u32;
    let bits = (n_levels as f32).log2() as u32;
    let boundaries_inner = lloyd_max_boundaries_inner(bits)?;
    let stream = mlx_rs::Stream::gpu();
    mlx_rs::metal::lumen_turboquant_encode_fused(x_bf16, &boundaries_inner, &stream)
        .map_err(|e| anyhow!("turboquant: lumen_tq_encode_fused kernel: {e}"))
}

/// Rotate + Stage-1 quantize in a single Metal kernel. Equivalent to
/// `(rotate_last_axis; lloyd_max_quantize_stage1_fused)` but skips the bf16
/// intermediate rotation tensor plus the separate matmul/cast dispatches.
///
/// Use when the rotated tensor is only consumed by the encode (e.g. V side
/// of the SlidingTurboquant SDPA branch). For K, prefer the unfused chain
/// since the rotated K is also consumed by the QJL Stage-2 residual.
///
/// Returns `(codes uint8 [..., D], sigma f32 [..., 1])`. Constraints
/// (validated in the C++ factory):
///   - `x_bf16` last-axis D ∈ multiples of 32, ≤ 1024
///   - `r_f32` shape `[D, D]`
///   - centroids drive `n_inner = n_levels - 1` ∈ [1, 255]
#[cfg(feature = "mlx-native")]
pub fn rotate_and_lloyd_max_quantize_stage1_fused(
    x_bf16: &Array,
    r_f32: &Array,
    centroids: &Array,
) -> Result<(Array, Array)> {
    let n_levels = centroids.shape()[0] as u32;
    let bits = (n_levels as f32).log2() as u32;
    let boundaries_inner = lloyd_max_boundaries_inner(bits)?;
    let stream = mlx_rs::Stream::gpu();
    mlx_rs::metal::lumen_turboquant_rot_encode_fused(x_bf16, r_f32, &boundaries_inner, &stream)
        .map_err(|e| anyhow!("turboquant: lumen_tq_rot_encode_fused kernel: {e}"))
}

/// Q @ K_codes inline matmul: computes `scores = Q · K_dq^T` without ever
/// materializing `K_dq`. Inline Lloyd-Max dequant via the centroids LUT and
/// per-K-vector σ. Saves the K_dq DRAM round-trip in the TQ-Stage-1 decode
/// path (Stage 2 of the 4-stage roadmap).
///
/// Inputs (shape conventions match the SlidingTurboquant cache):
///   - `q`        : bf16  `[B, H, T, D]`        (T = 1 for decode)
///   - `k_codes`  : uint8 `[B, H_kv, N, D]`
///   - `k_sigma`  : f32   `[B, H_kv, N, 1]` (cache layout) — the trailing-1
///                   axis is squeezed before the kernel call. May also be
///                   passed as `[B, H_kv, N]`.
///   - `centroids`: f32   `[n_levels]`
///
/// Returns `scores` bf16 `[B, H, T, N]`. First-iteration constraints
/// (enforced in the C++ factory): D == 256, n_levels ≤ 16, H % H_kv == 0.
#[cfg(feature = "mlx-native")]
pub fn turboquant_qk_inline(
    q: &Array,
    k_codes: &Array,
    k_sigma: &Array,
    centroids: &Array,
) -> Result<Array> {
    // The encode kernel emits sigma with a trailing-1 axis. Squeeze before
    // dispatching so the kernel sees a clean rank-3 `[B, H_kv, N]`.
    let k_sigma_3d = if k_sigma.ndim() == 4 && k_sigma.shape()[3] == 1 {
        let s = k_sigma.shape();
        mlx_rs::ops::reshape(k_sigma, &[s[0], s[1], s[2]])
            .context("turboquant_qk_inline: squeeze sigma trailing-1")?
    } else {
        k_sigma.clone()
    };
    let stream = mlx_rs::Stream::gpu();
    mlx_rs::metal::lumen_turboquant_qk_inline(q, k_codes, &k_sigma_3d, centroids, &stream)
        .map_err(|e| anyhow!("turboquant: lumen_tq_qk_inline kernel: {e}"))
}

/// Stage-1 fused encode with **packed 4-bit output**. Emits codes as uint32
/// with 8 codes per word — halves K/V cache storage and the downstream
/// inline-kernel DRAM read bandwidth. 4-bit only (`centroids` length must
/// be 16).
#[cfg(feature = "mlx-native")]
pub fn lloyd_max_quantize_stage1_packed4(
    x_bf16: &Array,
    centroids: &Array,
) -> Result<(Array, Array)> {
    let n_levels = centroids.shape()[0] as u32;
    if n_levels != 16 {
        return Err(anyhow!(
            "lloyd_max_quantize_stage1_packed4: only 4-bit supported \
             (n_levels=16), got {}",
            n_levels
        ));
    }
    let bits = 4u32;
    let boundaries_inner = lloyd_max_boundaries_inner(bits)?;
    let stream = mlx_rs::Stream::gpu();
    mlx_rs::metal::lumen_turboquant_encode_fused_packed4(x_bf16, &boundaries_inner, &stream)
        .map_err(|e| anyhow!("turboquant: lumen_tq_encode_fused_packed4 kernel: {e}"))
}

/// Q @ K_codes_packed inline matmul (4-bit packed). Symmetric to
/// `turboquant_qk_inline` but consumes packed uint32 codes (8 codes per
/// word). Used by the bit-packed cache path in the SlidingTurboquant
/// decode branch.
#[cfg(feature = "mlx-native")]
pub fn turboquant_qk_inline_packed4(
    q: &Array,
    k_codes_pkd: &Array,
    k_sigma: &Array,
    centroids: &Array,
) -> Result<Array> {
    let k_sigma_3d = if k_sigma.ndim() == 4 && k_sigma.shape()[3] == 1 {
        let s = k_sigma.shape();
        mlx_rs::ops::reshape(k_sigma, &[s[0], s[1], s[2]])
            .context("turboquant_qk_inline_packed4: squeeze sigma trailing-1")?
    } else {
        k_sigma.clone()
    };
    let stream = mlx_rs::Stream::gpu();
    mlx_rs::metal::lumen_turboquant_qk_inline_packed4(
        q,
        k_codes_pkd,
        &k_sigma_3d,
        centroids,
        &stream,
    )
    .map_err(|e| anyhow!("turboquant: lumen_tq_qk_inline_packed4 kernel: {e}"))
}

/// softmax_scores @ V_codes inline matmul: computes `O = S · V_dq` without
/// ever materializing `V_dq`. Symmetric V-side counterpart to
/// `turboquant_qk_inline` (Stage 3 of the 4-stage roadmap). Inline Lloyd-Max
/// dequant via centroids LUT and per-V-vector σ. Saves the V_dq DRAM
/// round-trip in the TQ-Stage-1 decode path.
///
/// Inputs (shape conventions match the SlidingTurboquant cache):
///   - `s`        : bf16  `[B, H, T, N]`        (softmax-normalized scores)
///   - `v_codes`  : uint8 `[B, H_kv, N, D]`
///   - `v_sigma`  : f32   `[B, H_kv, N, 1]` (cache layout) — the trailing-1
///                   axis is squeezed before the kernel call. May also be
///                   passed as `[B, H_kv, N]`.
///   - `centroids`: f32   `[n_levels]`
///
/// Returns `attn_out` bf16 `[B, H, T, D]`. First-iteration constraints
/// (enforced in the C++ factory): D == 256, n_levels ≤ 16, H % H_kv == 0.
#[cfg(feature = "mlx-native")]
pub fn turboquant_sv_inline(
    s: &Array,
    v_codes: &Array,
    v_sigma: &Array,
    centroids: &Array,
) -> Result<Array> {
    let v_sigma_3d = if v_sigma.ndim() == 4 && v_sigma.shape()[3] == 1 {
        let sh = v_sigma.shape();
        mlx_rs::ops::reshape(v_sigma, &[sh[0], sh[1], sh[2]])
            .context("turboquant_sv_inline: squeeze sigma trailing-1")?
    } else {
        v_sigma.clone()
    };
    let stream = mlx_rs::Stream::gpu();
    mlx_rs::metal::lumen_turboquant_sv_inline(s, v_codes, &v_sigma_3d, centroids, &stream)
        .map_err(|e| anyhow!("turboquant: lumen_tq_sv_inline kernel: {e}"))
}

/// Fused TurboQuant attention: `O = softmax(Q · K_dq^T · scale) · V_dq`
/// in a single Metal dispatch with inline Lloyd-Max K/V dequant. Replaces
/// the (qk_inline + softmax + sv_inline) 3-dispatch chain at decode (T=1).
///
/// Constraints (enforced in C++ factory): `T == 1`, `D ∈ {256, 512}`,
/// `n_levels ≤ 16`, `H % H_kv == 0`.
#[cfg(feature = "mlx-native")]
#[allow(clippy::too_many_arguments)]
pub fn turboquant_fused_attn(
    q: &Array,
    k_codes: &Array,
    k_sigma: &Array,
    v_codes: &Array,
    v_sigma: &Array,
    centroids: &Array,
    scale: f32,
) -> Result<Array> {
    // Encode emits sigma with a trailing-1 axis; squeeze for the kernel.
    let k_sigma_3d = if k_sigma.ndim() == 4 && k_sigma.shape()[3] == 1 {
        let sh = k_sigma.shape();
        mlx_rs::ops::reshape(k_sigma, &[sh[0], sh[1], sh[2]])
            .context("turboquant_fused_attn: squeeze k_sigma trailing-1")?
    } else {
        k_sigma.clone()
    };
    let v_sigma_3d = if v_sigma.ndim() == 4 && v_sigma.shape()[3] == 1 {
        let sh = v_sigma.shape();
        mlx_rs::ops::reshape(v_sigma, &[sh[0], sh[1], sh[2]])
            .context("turboquant_fused_attn: squeeze v_sigma trailing-1")?
    } else {
        v_sigma.clone()
    };
    let stream = mlx_rs::Stream::gpu();
    mlx_rs::metal::lumen_turboquant_fused_attn(
        q,
        k_codes,
        &k_sigma_3d,
        v_codes,
        &v_sigma_3d,
        centroids,
        scale,
        &stream,
    )
    .map_err(|e| anyhow!("turboquant: lumen_tq_fused_attn kernel: {e}"))
}

// ────────────────────── QJL Stage-2 correctness tests ──────────────────────
//
// Pure-math tests of the QJL Stage-2 implementation against synthetic K
// (Gaussian after rotation, the regime our K_rot occupies). We check:
//
//   1. cos(K_eff, K_true) > cos(K_dq, K_true)              — QJL improves
//                                                            K reconstruction
//   2. MSE(K_eff, K_true) < MSE(K_dq, K_true)
//   3. inner-product RMSE: < q, K_eff > closer to < q, K > than
//      < q, K_dq > is to < q, K >, averaged over random q
//
// All assertions use `>=` with a tolerance because at very high bits Stage 1
// already reconstructs near-perfectly and Stage 2 only adds variance noise
// (the corrected estimator is unbiased but introduces O(‖r‖/√m) per-element
// jitter). The test sweeps bits and confirms QJL helps where Stage 1 has
// room to improve.

#[cfg(all(test, feature = "mlx-native"))]
mod qjl_correctness_tests {
    use super::*;
    use mlx_rs::{Array, Dtype, random};

    /// One-sided cosine similarity ⟨x, y⟩ / (‖x‖ ‖y‖) computed in f32 and
    /// reduced to a Rust scalar.
    fn cosine_sim(x: &Array, y: &Array) -> Result<f32> {
        let xf = x.as_dtype(Dtype::Float32)?;
        let yf = y.as_dtype(Dtype::Float32)?;
        let dot = mlx_rs::ops::sum(&mlx_rs::ops::multiply(&xf, &yf)?, false)?;
        let nx = mlx_rs::ops::sqrt(&mlx_rs::ops::sum(&mlx_rs::ops::multiply(&xf, &xf)?, false)?)?;
        let ny = mlx_rs::ops::sqrt(&mlx_rs::ops::sum(&mlx_rs::ops::multiply(&yf, &yf)?, false)?)?;
        let denom = mlx_rs::ops::multiply(&nx, &ny)?;
        let cos = mlx_rs::ops::divide(&dot, &denom)?;
        Ok(cos.item::<f32>())
    }

    /// Per-element MSE in f32, reduced to a scalar.
    fn mse(a: &Array, b: &Array) -> Result<f32> {
        let af = a.as_dtype(Dtype::Float32)?;
        let bf = b.as_dtype(Dtype::Float32)?;
        let d = mlx_rs::ops::subtract(&af, &bf)?;
        let sq = mlx_rs::ops::multiply(&d, &d)?;
        let m = mlx_rs::ops::mean(&sq, false)?;
        Ok(m.item::<f32>())
    }

    /// Inner-product RMSE across random q. For each row of K we compute
    /// (q · k_row)² averaged then sqrt'd; we apply this to both
    /// `(q · K_eff - q · K_true)` and `(q · K_dq - q · K_true)`. The
    /// estimator improves iff the K_eff RMSE is smaller.
    fn inner_product_rmse(k_test: &Array, k_true: &Array, q: &Array) -> Result<f32> {
        // K shapes: [..., D]. Q shape: [..., D]. Compute (K_test - K_true) · qᵀ.
        let kf = k_test.as_dtype(Dtype::Float32)?;
        let tf = k_true.as_dtype(Dtype::Float32)?;
        let qf = q.as_dtype(Dtype::Float32)?;
        let diff = mlx_rs::ops::subtract(&kf, &tf)?;
        // diff shape [..., D]; q shape [..., D]. Their elementwise product summed
        // over the last axis gives <q, diff> per row.
        let prod = mlx_rs::ops::multiply(&diff, &qf)?;
        let last = (prod.ndim() as i32) - 1;
        let dot = mlx_rs::ops::sum_axis(&prod, last, false)?;
        let sq = mlx_rs::ops::multiply(&dot, &dot)?;
        let m = mlx_rs::ops::mean(&sq, false)?;
        Ok(m.item::<f32>().sqrt())
    }

    /// End-to-end QJL pipeline on a synthetic Gaussian K. Returns a table
    /// `[(bits, cos_stage1, cos_qjl, mse_stage1, mse_qjl, ip_rmse_stage1,
    ///   ip_rmse_qjl)]`.
    fn run_pipeline(
        shape: &[i32],
        m: usize,
        bits_sweep: &[u32],
    ) -> Result<Vec<(u32, f32, f32, f32, f32, f32, f32)>> {
        // Synthesize K ≈ N(0,1) (what K_rot looks like after Haar rotation).
        let key = random::key(0xDEADBEEFu64)?;
        let k_true_f32 = random::normal::<f32>(shape, None, None, &key)?;
        let k_true_bf16 = k_true_f32.as_dtype(Dtype::Bfloat16)?;

        // Random Q for inner-product accuracy probe — independent key so Q
        // and K aren't trivially correlated.
        let key_q = random::key(0xBADCAFE0u64)?;
        let q_f32 = random::normal::<f32>(shape, None, None, &key_q)?;
        let q_bf16 = q_f32.as_dtype(Dtype::Bfloat16)?;

        let d = *shape.last().unwrap() as usize;
        let proj = qjl_projection_matrix_f32(d, m, TURBOQUANT_SEED)?;

        let mut rows = Vec::new();
        for &bits in bits_sweep {
            let centroids = lloyd_max_centroids(bits)?;
            let (codes, sigma) = lloyd_max_quantize_stage1_fused(&k_true_bf16, &centroids)?;
            let k_dq = lloyd_max_dequantize_scaled(&codes, &sigma, &centroids)?;

            let (signs, r_norm) = qjl_encode_stage2(&k_true_bf16, &k_dq, &proj)?;
            let k_eff = qjl_apply_correction_to_k_dq(&k_dq, &signs, &r_norm, &proj, m)?;

            let cos_s1 = cosine_sim(&k_dq, &k_true_bf16)?;
            let cos_q = cosine_sim(&k_eff, &k_true_bf16)?;
            let mse_s1 = mse(&k_dq, &k_true_bf16)?;
            let mse_q = mse(&k_eff, &k_true_bf16)?;
            let ip_s1 = inner_product_rmse(&k_dq, &k_true_bf16, &q_bf16)?;
            let ip_q = inner_product_rmse(&k_eff, &k_true_bf16, &q_bf16)?;

            rows.push((bits, cos_s1, cos_q, mse_s1, mse_q, ip_s1, ip_q));
        }
        Ok(rows)
    }

    /// Sweep across m at fixed D to map the QJL variance/bias trade-off.
    /// Theory: for the per-element K reconstruction the QJL unbiased
    /// estimator has variance ≈ ‖r‖² · (π/2)/m, while Stage-1's per-element
    /// MSE is ‖r‖² / D. So QJL beats Stage 1 only once `m > D · π/2`.
    /// For D=256 that threshold is ~402; we expect m=128 to lose, m=512
    /// to draw or marginally win, m=1024 to win clearly.
    #[test]
    fn qjl_m_sweep_traces_variance_threshold() {
        let shape = [1, 4, 64, 256];
        let d = 256;
        let bits = 4u32;
        let m_sweep = [64usize, 128, 256, 512, 1024];

        eprintln!(
            "\nQJL m sweep at D={} bits={} (threshold m* = D·π/2 ≈ {:.0})",
            d,
            bits,
            (d as f32) * std::f32::consts::FRAC_PI_2
        );
        eprintln!("m    | cos_dq      | cos_eff     | Δcos    | MSE_dq    | MSE_eff   | MSE ratio");
        let mut results = Vec::new();
        for &m in &m_sweep {
            let row = run_pipeline(&shape, m, &[bits]).expect("QJL pipeline ran");
            let (_b, cs1, cq, ms1, mq, _, _) = row[0];
            let ratio = mq / ms1;
            eprintln!(
                "{:>4} | {:>11.5} | {:>11.5} | {:>7.5} | {:>9.5} | {:>9.5} | {:>9.3}",
                m,
                cs1,
                cq,
                cq - cs1,
                ms1,
                mq,
                ratio
            );
            results.push((m, cs1, cq, ms1, mq, ratio));
        }

        // Verify variance-dominated regime: at m=64,128 the MSE ratio is
        // approximately D·π/(2m). Allow ±25% tolerance for finite-batch
        // estimator noise (the analytic formula is asymptotic).
        for &(m, _, _, _, _, ratio) in results.iter().filter(|r| r.0 < 256) {
            let theoretical = (d as f32) * std::f32::consts::FRAC_PI_2 / (m as f32);
            assert!(
                (ratio / theoretical - 1.0).abs() < 0.30,
                "m={}: measured MSE ratio {:.3} far from theoretical {:.3}",
                m,
                ratio,
                theoretical
            );
        }

        // Verify bias-corrected regime: at m=1024 (well past threshold),
        // QJL should strictly beat Stage 1 in MSE.
        let m1024 = results.iter().find(|r| r.0 == 1024).unwrap();
        assert!(
            m1024.4 < m1024.3,
            "m=1024: QJL MSE {:.5} should beat Stage 1 MSE {:.5}",
            m1024.4,
            m1024.3
        );
    }

    /// Packed-bit path equivalence: encode/apply via u32 packed signs must
    /// produce bit-identical K_eff vs the bf16-±1 path. Validates that the
    /// 16× storage savings come for free (modulo bf16 rounding inside the
    /// matmul, which both paths share equally).
    #[test]
    fn qjl_packed_path_equivalent_to_bf16_path() {
        let shape = [1, 4, 16, 256];
        let d = 256;
        let m = 128;
        let bits = 4u32;

        let key = random::key(0x1234_5678u64).unwrap();
        let k_true_f32 = random::normal::<f32>(&shape[..], None, None, &key).unwrap();
        let k_true_bf16 = k_true_f32.as_dtype(Dtype::Bfloat16).unwrap();
        let proj = qjl_projection_matrix_f32(d, m, TURBOQUANT_SEED).unwrap();
        let centroids = lloyd_max_centroids(bits).unwrap();

        let (codes, sigma) = lloyd_max_quantize_stage1_fused(&k_true_bf16, &centroids).unwrap();
        let k_dq = lloyd_max_dequantize_scaled(&codes, &sigma, &centroids).unwrap();

        // ── bf16 reference path ──
        let (signs_bf16, r_norm_a) = qjl_encode_stage2(&k_true_bf16, &k_dq, &proj).unwrap();
        let k_eff_a =
            qjl_apply_correction_to_k_dq(&k_dq, &signs_bf16, &r_norm_a, &proj, m).unwrap();

        // ── packed path ──
        let (signs_packed, r_norm_b) =
            qjl_encode_stage2_packed(&k_true_bf16, &k_dq, &proj).unwrap();
        let k_eff_b =
            qjl_apply_correction_packed(&k_dq, &signs_packed, &r_norm_b, &proj, m).unwrap();

        // r_norm must be bit-identical (same op chain).
        let rn_diff = mse(&r_norm_a, &r_norm_b).unwrap();
        assert!(
            rn_diff < 1e-10,
            "r_norm differs between paths: MSE={}",
            rn_diff
        );

        // K_eff: both paths share the same matmul + scale + add, so the
        // only divergence is the encode-side sign extraction (the pack
        // kernel takes sign of the f32 r_proj directly; the bf16 path
        // does the same via where(>= 0)). Result should be bit-identical
        // up to bf16 rounding.
        let diff = mse(&k_eff_a, &k_eff_b).unwrap();
        eprintln!(
            "\nQJL packed-path equivalence: MSE(K_eff_bf16, K_eff_packed) = {:.2e}",
            diff
        );
        assert!(
            diff < 1e-6,
            "Packed and bf16 paths produced divergent K_eff: MSE={}",
            diff
        );

        // Storage savings sanity: packed should be 16× smaller in elements.
        let bf16_elems = signs_bf16.shape().iter().product::<i32>() as f32;
        let packed_elems = signs_packed.shape().iter().product::<i32>() as f32;
        let ratio = bf16_elems / packed_elems;
        eprintln!(
            "QJL signs storage: bf16={} elems × 2 B = {} B; packed={} elems × 4 B = {} B; \
             ratio = {:.1}× smaller (raw bytes)",
            bf16_elems,
            bf16_elems * 2.0,
            packed_elems,
            packed_elems * 4.0,
            (bf16_elems * 2.0) / (packed_elems * 4.0)
        );
        assert!(
            ratio >= 31.0 && ratio <= 33.0,
            "expected packed elements 32× fewer than bf16 (m=128 → n_words=4); got {}×",
            ratio
        );
    }

    /// Default-config Gemma 4 test (D=256, m=128). Documents the regression:
    /// at m=128 QJL hurts reconstruction. Acts as a regression guard and
    /// surfaces the variance-vs-bias trade-off if anyone tunes m back down.
    #[test]
    fn qjl_at_default_m128_is_variance_dominated() {
        let shape = [1, 4, 64, 256];
        let m = 128;
        let table = run_pipeline(&shape, m, &[8, 6, 4, 3, 2]).expect("QJL pipeline ran");

        eprintln!("\nQJL Stage-2 at D=256 m=128 (DEFAULT — variance-dominated)");
        eprintln!(
            "bits | cos_dq    | cos_eff   | MSE_dq    | MSE_eff   | IP_rmse_dq | IP_rmse_eff"
        );
        for (bits, cs1, cq, ms1, mq, ips1, ipq) in &table {
            eprintln!(
                "{:>4} | {:>9.5} | {:>9.5} | {:>9.5} | {:>9.5} | {:>10.4} | {:>11.4}",
                bits, cs1, cq, ms1, mq, ips1, ipq
            );
        }

        // Document the negative finding: at m=128, MSE_eff is consistently
        // worse than MSE_dq. Test asserts this regime so a future "QJL
        // works at m=128 now!" bug is caught loudly. Once a packed-kernel
        // implementation pushes m to >D·π/2 cheaply (or QJL variance is
        // reduced by other means), update this assertion.
        for (bits, _, _, ms1, mq, _, _) in &table {
            if *bits <= 4 {
                assert!(
                    *mq > *ms1,
                    "bits={}: at m=128 QJL is expected to be worse than Stage 1 \
                     (variance-dominated); got MSE_dq={:.5} MSE_eff={:.5}",
                    bits,
                    ms1,
                    mq
                );
            }
        }
    }

    /// Verify that mlx-affine 4-bit roundtrip through a Haar-rotated
    /// weight matrix recovers the dense rotated weight to within typical
    /// quant noise. If this MSE is much larger than the original Wv's
    /// quant MSE, the rotation is interacting poorly with mlx-affine
    /// quantization (suspected when baked V-projection produces garbage
    /// despite the dense math being exact).
    #[test]
    fn bake_r_affine_4bit_roundtrip_check() {
        use crate::native_quant::{MODE_AFFINE, dequantize_with_mode, quantize_with_mode};

        let d = 256i32;
        let h_kv = 8i32;
        let hidden = 2816i32;
        let group_size = 64i32;
        let bits = 4i32;

        // Simulate a trained weight: small-magnitude trained-like values.
        let key_w = random::key(0xFACE0001u64).unwrap();
        let wv_f32 = random::normal::<f32>(&[h_kv * d, hidden], None, None, &key_w).unwrap();
        // Trained weights are ~N(0, 0.02²) typical. Scale down.
        let scale = Array::from_f32(0.02f32);
        let wv_f32 = mlx_rs::ops::multiply(&wv_f32, &scale).unwrap();
        let wv_bf16 = wv_f32.as_dtype(Dtype::Bfloat16).unwrap();

        let r = rotation_matrix_f32(d as usize, TURBOQUANT_SEED).unwrap();

        // Reference: dense bf16 Wv @ R (no quant noise on Wv_rot).
        let wv_3d = mlx_rs::ops::reshape(&wv_bf16, &[h_kv, d, hidden]).unwrap();
        let wv_t = mlx_rs::ops::transpose_axes(&wv_3d, &[0, 2, 1]).unwrap();
        let r_bf16 = r.as_dtype(Dtype::Bfloat16).unwrap();
        let wv_t_rot = mlx_rs::ops::matmul(&wv_t, &r_bf16).unwrap();
        let wv_rot_3d = mlx_rs::ops::transpose_axes(&wv_t_rot, &[0, 2, 1]).unwrap();
        let wv_rot_ref = mlx_rs::ops::reshape(&wv_rot_3d, &[h_kv * d, hidden]).unwrap();

        // Path A: quant Wv first, then dequant, matmul R (path that
        // works at runtime — produces V then V @ R).
        let (wv_q, wv_q_scales, wv_q_biases) =
            quantize_with_mode(&wv_bf16, group_size, bits, MODE_AFFINE).unwrap();
        let wv_dq = dequantize_with_mode(
            &wv_q,
            &wv_q_scales,
            wv_q_biases.as_ref(),
            group_size,
            bits,
            MODE_AFFINE,
        )
        .unwrap();
        // Apply rotation to the dequantized Wv (this is what runtime does
        // post-projection, modulo per-vector scope).
        let wv_dq_3d = mlx_rs::ops::reshape(&wv_dq, &[h_kv, d, hidden]).unwrap();
        let wv_dq_t = mlx_rs::ops::transpose_axes(&wv_dq_3d, &[0, 2, 1]).unwrap();
        let wv_dq_rot_t = mlx_rs::ops::matmul(&wv_dq_t, &r_bf16).unwrap();
        let wv_dq_rot_3d = mlx_rs::ops::transpose_axes(&wv_dq_rot_t, &[0, 2, 1]).unwrap();
        let wv_dq_rot = mlx_rs::ops::reshape(&wv_dq_rot_3d, &[h_kv * d, hidden]).unwrap();

        // Path B: bake — rotate dense Wv, requant, dequant.
        let wv_rot_for_bake = wv_rot_ref.clone();
        let (wv_b_q, wv_b_scales, wv_b_biases) =
            quantize_with_mode(&wv_rot_for_bake, group_size, bits, MODE_AFFINE).unwrap();
        let wv_b_dq = dequantize_with_mode(
            &wv_b_q,
            &wv_b_scales,
            wv_b_biases.as_ref(),
            group_size,
            bits,
            MODE_AFFINE,
        )
        .unwrap();

        let mse_runtime = mse(&wv_rot_ref, &wv_dq_rot).unwrap();
        let mse_baked = mse(&wv_rot_ref, &wv_b_dq).unwrap();
        eprintln!("\nbake_r affine 4-bit roundtrip:");
        eprintln!(
            "  MSE(runtime: dequant-then-rotate vs dense_rot) = {:.5e}",
            mse_runtime
        );
        eprintln!(
            "  MSE(baked:   rotate-then-requant-then-dequant)  = {:.5e}",
            mse_baked
        );
        eprintln!(
            "  ratio baked/runtime = {:.2}",
            mse_baked / mse_runtime.max(1e-12)
        );
        // Baked-path error shouldn't be wildly larger than runtime-path
        // error (which is already 4-bit-quant-limited). A 2× or so is
        // expected (two rounds of small noise vs one), but 10× would
        // indicate the bake is incompatible with affine grouping.
        assert!(
            mse_baked < 10.0 * mse_runtime,
            "baked MSE {:.3e} >> runtime MSE {:.3e}",
            mse_baked,
            mse_runtime
        );
    }

    /// Sanity check that `V @ R == X @ (Wv @ R)^T` and that
    /// `(V_rot @ Rᵀ) @ Wo^T == V_rot @ (Wo @ R)^T` (the two matmul
    /// identities that justify the bake-R-into-weights lever).
    /// Done in dense fp32 first to confirm the math, then with mlx-affine
    /// 4-bit quantize/dequantize to confirm the lossy path also lands
    /// in the same neighborhood as the runtime rotation chain.
    #[test]
    fn bake_r_identity_in_dense_and_affine() {
        let d = 256i32;
        let h_kv = 8i32;
        let h_q = 16i32;
        let hidden = 2816i32;

        let key_x = random::key(0xA5A5A5A5u64).unwrap();
        let key_w = random::key(0xC0FFEE00u64).unwrap();
        let key_o = random::key(0xBEEFu64).unwrap();
        let x = random::normal::<f32>(&[2, hidden], None, None, &key_x).unwrap();
        let wv = random::normal::<f32>(&[h_kv * d, hidden], None, None, &key_w).unwrap();
        let wo = random::normal::<f32>(&[hidden, h_q * d], None, None, &key_o).unwrap();
        let r = rotation_matrix_f32(d as usize, TURBOQUANT_SEED).unwrap();

        // Path A (un-baked): V = X @ Wv^T; V_rot = V @ R (per-head).
        let wv_t = mlx_rs::ops::transpose_axes(&wv, &[1, 0]).unwrap();
        let v = mlx_rs::ops::matmul(&x, &wv_t).unwrap(); // [2, h_kv*d]
        let v_4d = mlx_rs::ops::reshape(&v, &[2, h_kv, d]).unwrap(); // [B, h_kv, d]
        let v_rot = mlx_rs::ops::matmul(&v_4d, &r).unwrap(); // [2, h_kv, d]

        // Path B (baked Wv): V_baked = X @ Wv_rot^T directly.
        let wv_3d = mlx_rs::ops::reshape(&wv, &[h_kv, d, hidden]).unwrap();
        let wv_t3 = mlx_rs::ops::transpose_axes(&wv_3d, &[0, 2, 1]).unwrap();
        let wv_rot3_t = mlx_rs::ops::matmul(&wv_t3, &r).unwrap();
        let wv_rot3 = mlx_rs::ops::transpose_axes(&wv_rot3_t, &[0, 2, 1]).unwrap();
        let wv_rot = mlx_rs::ops::reshape(&wv_rot3, &[h_kv * d, hidden]).unwrap();
        let wv_rot_t = mlx_rs::ops::transpose_axes(&wv_rot, &[1, 0]).unwrap();
        let v_baked = mlx_rs::ops::matmul(&x, &wv_rot_t).unwrap();
        let v_baked_4d = mlx_rs::ops::reshape(&v_baked, &[2, h_kv, d]).unwrap();

        let v_diff = mse(&v_rot, &v_baked_4d).unwrap();
        eprintln!(
            "\nbake_r v_proj identity: MSE(V_rot, V_baked) = {:.3e}",
            v_diff
        );
        assert!(v_diff < 1e-4, "V_rot vs V_baked: MSE too large {v_diff}");

        // Wo bake identity: with attn in rotated space, baked Wo should
        // give the same output as (un-rotation @ original Wo).
        let attn_rot = random::normal::<f32>(&[2, h_q, d], None, None, &key_x).unwrap();
        // Path A (un-baked): un-rotate then matmul with original Wo.
        let r_t = mlx_rs::ops::transpose_axes(&r, &[1, 0]).unwrap();
        let attn_unrot = mlx_rs::ops::matmul(&attn_rot, &r_t).unwrap();
        let attn_flat = mlx_rs::ops::reshape(&attn_unrot, &[2, h_q * d]).unwrap();
        let wo_t = mlx_rs::ops::transpose_axes(&wo, &[1, 0]).unwrap();
        let out_a = mlx_rs::ops::matmul(&attn_flat, &wo_t).unwrap();

        // Path B (baked Wo): use rotated attn directly with baked Wo.
        let wo_3d = mlx_rs::ops::reshape(&wo, &[hidden, h_q, d]).unwrap();
        let wo_rot3 = mlx_rs::ops::matmul(&wo_3d, &r).unwrap();
        let wo_rot = mlx_rs::ops::reshape(&wo_rot3, &[hidden, h_q * d]).unwrap();
        let wo_rot_t = mlx_rs::ops::transpose_axes(&wo_rot, &[1, 0]).unwrap();
        let attn_rot_flat = mlx_rs::ops::reshape(&attn_rot, &[2, h_q * d]).unwrap();
        let out_b = mlx_rs::ops::matmul(&attn_rot_flat, &wo_rot_t).unwrap();

        let o_diff = mse(&out_a, &out_b).unwrap();
        eprintln!(
            "bake_r o_proj identity: MSE(out_unbaked, out_baked) = {:.3e}",
            o_diff
        );
        assert!(o_diff < 1e-3, "Wo bake identity MSE too large {o_diff}");
    }

    /// Fused rotate+encode kernel must match the (rotate_last_axis;
    /// encode_fused) chain modulo bf16 truncation of the rotated
    /// intermediate. The fused path skips that bf16 round-trip, so its
    /// codes/σ are *strictly more* accurate vs the f32 ground truth — but
    /// the differences vs the reference chain should be tiny (≤ bf16 ULP
    /// on rotated y; ≥ 99% of codes must agree at 4-bit).
    #[test]
    fn rot_encode_fused_matches_rotate_then_encode() {
        let shape = [1, 8, 64, 256];
        let d = 256usize;
        let bits = 4u32;

        let key = random::key(0xABCDEF01u64).unwrap();
        let v_f32 = random::normal::<f32>(&shape[..], None, None, &key).unwrap();
        let v_bf16 = v_f32.as_dtype(Dtype::Bfloat16).unwrap();

        let r_arr = rotation_matrix_f32(d, TURBOQUANT_SEED).unwrap();
        let centroids = lloyd_max_centroids(bits).unwrap();

        // Reference: rotate (bf16→f32 matmul→bf16) → encode_fused.
        let v_rot = rotate_last_axis(&v_bf16, &r_arr).unwrap();
        let (codes_ref, sigma_ref) = lloyd_max_quantize_stage1_fused(&v_rot, &centroids).unwrap();

        // Fused: rotate+encode in one kernel.
        let (codes_f, sigma_f) =
            rotate_and_lloyd_max_quantize_stage1_fused(&v_bf16, &r_arr, &centroids).unwrap();

        // σ comparison in f32 — both kernels write f32 σ directly; the only
        // numeric difference is the bf16 round-trip on y in the reference
        // path.
        let sigma_mse = mse(&sigma_ref, &sigma_f).unwrap();
        eprintln!("\nrot+encode fused vs reference: σ MSE = {:.3e}", sigma_mse);
        assert!(
            sigma_mse < 1e-3,
            "σ MSE too large between fused and reference: {}",
            sigma_mse
        );

        // Code agreement rate — the only divergence is at bin boundaries
        // where bf16 noise on y/σ can flip a code by ±1.
        let codes_ref_i32 = codes_ref.as_dtype(Dtype::Int32).unwrap();
        let codes_f_i32 = codes_f.as_dtype(Dtype::Int32).unwrap();
        let eq_i32 = mlx_rs::ops::eq(&codes_ref_i32, &codes_f_i32)
            .unwrap()
            .as_dtype(Dtype::Int32)
            .unwrap();
        let agree = mlx_rs::ops::mean(&eq_i32, false)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap()
            .item::<f32>();
        eprintln!(
            "rot+encode fused vs reference: code agreement = {:.4}",
            agree
        );
        assert!(
            agree > 0.99,
            "code agreement {:.4} below 0.99 — kernel divergence likely",
            agree
        );

        // End-to-end V_dq reconstruction parity. Build V_dq via both paths
        // (dequant uses the SAME centroids; only codes/σ differ slightly)
        // and assert MSE against V_rot is comparable.
        let v_dq_ref = lloyd_max_dequantize_scaled(&codes_ref, &sigma_ref, &centroids).unwrap();
        let v_dq_f = lloyd_max_dequantize_scaled(&codes_f, &sigma_f, &centroids).unwrap();
        let dq_mse = mse(&v_dq_ref, &v_dq_f).unwrap();
        eprintln!("rot+encode fused vs reference: V_dq MSE = {:.3e}", dq_mse);
        assert!(
            dq_mse < 1e-2,
            "V_dq MSE too large between fused and reference: {}",
            dq_mse
        );
    }

    /// `turboquant_qk_inline` must match the reference path
    ///   `lloyd_max_dequantize_scaled(K_codes, K_sigma) → bf16 matmul(Q, K_dq^T)`
    /// up to bf16 fma-ordering noise. Both paths accumulate in f32 inside the
    /// kernel/qmatmul but the reduction order differs, so we use a tight
    /// relative-MSE bound rather than bit-identical equality.
    ///
    /// Test uses H == H_kv (no GQA broadcast) to keep the reference reduction
    /// trivial. GQA correctness is exercised separately by the kernel
    /// constraint `H % H_kv == 0` (validated in the factory) and end-to-end
    /// via the integration A/B at the call site.
    #[test]
    fn qk_inline_matches_dequant_then_matmul() {
        use mlx_rs::random;
        let bits = 4u32;
        let b: i32 = 1;
        let h: i32 = 8;
        let h_kv: i32 = 8; // no GQA in this unit test
        let t: i32 = 1;
        let n: i32 = 128;
        let d: i32 = 256;
        let centroids = lloyd_max_centroids(bits).unwrap();

        let key_q = random::key(0xA11CE001u64).unwrap();
        let key_k = random::key(0xA11CE002u64).unwrap();
        let q_bf16 = random::normal::<f32>(&[b, h, t, d], None, None, &key_q)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let k_rot_bf16 = random::normal::<f32>(&[b, h_kv, n, d], None, None, &key_k)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();

        let (k_codes, k_sigma) = lloyd_max_quantize_stage1_fused(&k_rot_bf16, &centroids).unwrap();
        // Reference: materialize K_dq, transpose last two axes, then matmul.
        let k_dq = lloyd_max_dequantize_scaled(&k_codes, &k_sigma, &centroids).unwrap();
        let k_dq_t = mlx_rs::ops::transpose_axes(&k_dq, &[0, 1, 3, 2]).unwrap();
        let scores_ref = mlx_rs::ops::matmul(&q_bf16, &k_dq_t).unwrap();

        let scores_fused =
            super::turboquant_qk_inline(&q_bf16, &k_codes, &k_sigma, &centroids).unwrap();

        let r = scores_ref.as_dtype(Dtype::Float32).unwrap();
        let f = scores_fused.as_dtype(Dtype::Float32).unwrap();
        let diff = mlx_rs::ops::subtract(&r, &f).unwrap();
        let sq = mlx_rs::ops::multiply(&diff, &diff).unwrap();
        let nelems = (b * h * t * n) as f32;
        let sum_arr = mlx_rs::ops::sum(&sq, false).unwrap();
        let mse_val = sum_arr.item::<f32>() / nelems;

        let r_sq = mlx_rs::ops::multiply(&r, &r).unwrap();
        let r_sum_arr = mlx_rs::ops::sum(&r_sq, false).unwrap();
        let r_var = r_sum_arr.item::<f32>() / nelems;

        let rel = mse_val / (r_var + 1e-12);
        eprintln!(
            "qk_inline vs reference: MSE={:.3e}, ref_var={:.3e}, rel={:.3e}",
            mse_val, r_var, rel
        );
        // Both paths f32-accumulate identical math; difference comes from
        // bf16 store rounding at the end. Tight bound.
        assert!(
            rel < 1e-3,
            "qk_inline output relative MSE too large vs reference: rel={}",
            rel
        );
    }

    /// D=512 variant of `qk_inline_matches_dequant_then_matmul`. Exercises
    /// the `lumen_tq_qk_inline_d512` Metal kernel (VPT=16) — the kernel that
    /// crashes in full-attn end-to-end runs. Same correctness contract: rel
    /// MSE < 1e-3 vs the dequant + bf16 matmul reference. Failure mode this
    /// test isolates: if mlx-rs swallows the crash error in the full forward
    /// path, here the kernel is dispatched alone so the underlying mlx
    /// exception surfaces directly via `.unwrap()`.
    #[test]
    fn qk_inline_d512_matches_dequant_then_matmul() {
        use mlx_rs::random;
        let bits = 4u32;
        let b: i32 = 1;
        let h: i32 = 32;  // Gemma 4 full-attn H
        let h_kv: i32 = 2; // Gemma 4 full-attn H_kv (GQA group=16)
        let t: i32 = 1;
        let n: i32 = 64;
        let d: i32 = 512; // global_head_dim
        let centroids = lloyd_max_centroids(bits).unwrap();

        let key_q = random::key(0xD512001u64).unwrap();
        let key_k = random::key(0xD512002u64).unwrap();
        let q_bf16 = random::normal::<f32>(&[b, h, t, d], None, None, &key_q)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let k_rot_bf16 = random::normal::<f32>(&[b, h_kv, n, d], None, None, &key_k)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();

        let (k_codes, k_sigma) = lloyd_max_quantize_stage1_fused(&k_rot_bf16, &centroids).unwrap();

        // Reference: dequant + GQA-broadcast K + matmul. With GQA the K head
        // axis must be expanded to match Q's H. Easiest reference: replicate
        // K along the head dim via reshape + broadcast, then matmul with Q.
        let k_dq = lloyd_max_dequantize_scaled(&k_codes, &k_sigma, &centroids).unwrap();
        let group = (h / h_kv) as i32;
        // [B, H_kv, N, D] → [B, H_kv, 1, N, D] → broadcast to [B, H_kv, group, N, D]
        //                → [B, H, N, D]
        let k_dq_5d = mlx_rs::ops::reshape(&k_dq, &[b, h_kv, 1, n, d]).unwrap();
        let k_dq_b = mlx_rs::ops::broadcast_to(&k_dq_5d, &[b, h_kv, group, n, d]).unwrap();
        let k_dq_full = mlx_rs::ops::reshape(&k_dq_b, &[b, h, n, d]).unwrap();
        let k_dq_t = mlx_rs::ops::transpose_axes(&k_dq_full, &[0, 1, 3, 2]).unwrap();
        let scores_ref = mlx_rs::ops::matmul(&q_bf16, &k_dq_t).unwrap();

        let scores_fused =
            super::turboquant_qk_inline(&q_bf16, &k_codes, &k_sigma, &centroids).unwrap();

        let r = scores_ref.as_dtype(Dtype::Float32).unwrap();
        let f = scores_fused.as_dtype(Dtype::Float32).unwrap();
        let diff = mlx_rs::ops::subtract(&r, &f).unwrap();
        let sq = mlx_rs::ops::multiply(&diff, &diff).unwrap();
        let nelems = (b * h * t * n) as f32;
        let sum_arr = mlx_rs::ops::sum(&sq, false).unwrap();
        let mse_val = sum_arr.item::<f32>() / nelems;

        let r_sq = mlx_rs::ops::multiply(&r, &r).unwrap();
        let r_sum_arr = mlx_rs::ops::sum(&r_sq, false).unwrap();
        let r_var = r_sum_arr.item::<f32>() / nelems;

        let rel = mse_val / (r_var + 1e-12);
        eprintln!(
            "qk_inline_d512 vs reference: MSE={:.3e}, ref_var={:.3e}, rel={:.3e}",
            mse_val, r_var, rel
        );
        assert!(
            rel < 1e-3,
            "qk_inline_d512 output relative MSE too large vs reference: rel={}",
            rel
        );
    }

    /// `turboquant_sv_inline` must match the reference path
    ///   `lloyd_max_dequantize_scaled(V_codes, V_sigma) → bf16 matmul(S, V_dq)`
    /// up to bf16 fma-ordering noise. Both paths accumulate in f32; the
    /// kernel store rounds to bf16 at the end.
    ///
    /// Test uses H == H_kv (no GQA broadcast) to keep the reference matmul
    /// trivial. GQA correctness is exercised separately by the kernel
    /// constraint `H % H_kv == 0` and end-to-end via the integration A/B
    /// at the decode call site.
    #[test]
    fn sv_inline_matches_dequant_then_matmul() {
        use mlx_rs::random;
        let bits = 4u32;
        let b: i32 = 1;
        let h: i32 = 8;
        let h_kv: i32 = 8; // no GQA in this unit test
        let t: i32 = 1;
        let n: i32 = 128;
        let d: i32 = 256;
        let centroids = lloyd_max_centroids(bits).unwrap();

        let key_s = random::key(0xB055ED01u64).unwrap();
        let key_v = random::key(0xB055ED02u64).unwrap();
        let scores_bf16 = random::normal::<f32>(&[b, h, t, n], None, None, &key_s)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let v_rot_bf16 = random::normal::<f32>(&[b, h_kv, n, d], None, None, &key_v)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();

        let (v_codes, v_sigma) = lloyd_max_quantize_stage1_fused(&v_rot_bf16, &centroids).unwrap();
        // Reference: materialize V_dq, then S · V_dq.
        let v_dq = lloyd_max_dequantize_scaled(&v_codes, &v_sigma, &centroids).unwrap();
        let out_ref = mlx_rs::ops::matmul(&scores_bf16, &v_dq).unwrap();

        let out_fused =
            super::turboquant_sv_inline(&scores_bf16, &v_codes, &v_sigma, &centroids).unwrap();

        let r = out_ref.as_dtype(Dtype::Float32).unwrap();
        let f = out_fused.as_dtype(Dtype::Float32).unwrap();
        let diff = mlx_rs::ops::subtract(&r, &f).unwrap();
        let sq = mlx_rs::ops::multiply(&diff, &diff).unwrap();
        let nelems = (b * h * t * d) as f32;
        let mse_val = mlx_rs::ops::sum(&sq, false).unwrap().item::<f32>() / nelems;

        let r_sq = mlx_rs::ops::multiply(&r, &r).unwrap();
        let r_var = mlx_rs::ops::sum(&r_sq, false).unwrap().item::<f32>() / nelems;

        let rel = mse_val / (r_var + 1e-12);
        eprintln!(
            "sv_inline vs reference: MSE={:.3e}, ref_var={:.3e}, rel={:.3e}",
            mse_val, r_var, rel
        );
        assert!(
            rel < 1e-3,
            "sv_inline output relative MSE too large vs reference: rel={}",
            rel
        );
    }

    /// `turboquant_fused_attn` must match the 3-stage reference
    ///   `qk_inline → softmax → sv_inline` up to f32 accumulation noise.
    /// This is the correctness oracle for the fused kernel: same math, just
    /// one Metal dispatch instead of three.
    ///
    /// Test uses H == H_kv (no GQA) and small N to keep the reference
    /// computation simple and the softmax numerically benign. Failures fall
    /// into two buckets: kernel arithmetic (per-thread loops, exp / sum
    /// updates) or kernel layout (cross-SG aggregation, output write
    /// offset). Both surface as rel MSE > 1e-3 here.
    #[test]
    fn fused_attn_matches_qk_softmax_sv_reference() {
        use mlx_rs::random;
        let bits = 4u32;
        let b: i32 = 1;
        let h: i32 = 8;
        let h_kv: i32 = 8; // no GQA
        let t: i32 = 1;
        let n: i32 = 128;
        let d: i32 = 256;
        let centroids = lloyd_max_centroids(bits).unwrap();

        let key_q = random::key(0xFA5ED001u64).unwrap();
        let key_k = random::key(0xFA5ED002u64).unwrap();
        let key_v = random::key(0xFA5ED003u64).unwrap();
        let q_bf16 = random::normal::<f32>(&[b, h, t, d], None, None, &key_q)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let k_rot_bf16 = random::normal::<f32>(&[b, h_kv, n, d], None, None, &key_k)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let v_rot_bf16 = random::normal::<f32>(&[b, h_kv, n, d], None, None, &key_v)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();

        let (k_codes, k_sigma) = lloyd_max_quantize_stage1_fused(&k_rot_bf16, &centroids).unwrap();
        let (v_codes, v_sigma) = lloyd_max_quantize_stage1_fused(&v_rot_bf16, &centroids).unwrap();

        // Reference: same 3-stage chain the existing TQ inline path uses.
        let scores_ref =
            super::turboquant_qk_inline(&q_bf16, &k_codes, &k_sigma, &centroids).unwrap();
        let last_axis = (scores_ref.ndim() as i32) - 1;
        let attn_w_ref =
            mlx_rs::ops::softmax_axis(&scores_ref, last_axis, Some(true)).unwrap();
        let out_ref =
            super::turboquant_sv_inline(&attn_w_ref, &v_codes, &v_sigma, &centroids).unwrap();

        // Fused: scale=1.0 — q_norm already normalized Q so no extra 1/sqrt(D).
        let out_fused = super::turboquant_fused_attn(
            &q_bf16, &k_codes, &k_sigma, &v_codes, &v_sigma, &centroids, 1.0,
        )
        .unwrap();

        let r = out_ref.as_dtype(Dtype::Float32).unwrap();
        let f = out_fused.as_dtype(Dtype::Float32).unwrap();
        let diff = mlx_rs::ops::subtract(&r, &f).unwrap();
        let sq = mlx_rs::ops::multiply(&diff, &diff).unwrap();
        let nelems = (b * h * t * d) as f32;
        let mse_val = mlx_rs::ops::sum(&sq, false).unwrap().item::<f32>() / nelems;

        let r_sq = mlx_rs::ops::multiply(&r, &r).unwrap();
        let r_var = mlx_rs::ops::sum(&r_sq, false).unwrap().item::<f32>() / nelems;

        let rel = mse_val / (r_var + 1e-12);
        eprintln!(
            "fused_attn vs (qk+softmax+sv) reference: MSE={:.3e}, ref_var={:.3e}, rel={:.3e}",
            mse_val, r_var, rel
        );
        assert!(
            rel < 1e-3,
            "fused_attn output relative MSE too large vs reference: rel={}",
            rel
        );
    }

    /// GQA variant of `fused_attn_matches_qk_softmax_sv_reference` — the
    /// no-GQA test confirms kernel arithmetic; this one validates the GQA
    /// indexing (`h_kv = h * H_kv / H`). Mismatch in this test pins the bug
    /// to the per-head dispatch inside the kernel.
    #[test]
    fn fused_attn_gqa_matches_reference() {
        use mlx_rs::random;
        let bits = 4u32;
        let b: i32 = 1;
        let h: i32 = 32;  // Gemma 4 sliding H
        let h_kv: i32 = 8; // sliding H_kv (group=4)
        let t: i32 = 1;
        let n: i32 = 128;
        let d: i32 = 256;
        let centroids = lloyd_max_centroids(bits).unwrap();

        let key_q = random::key(0xFA5EDA01u64).unwrap();
        let key_k = random::key(0xFA5EDA02u64).unwrap();
        let key_v = random::key(0xFA5EDA03u64).unwrap();
        let q_bf16 = random::normal::<f32>(&[b, h, t, d], None, None, &key_q)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let k_rot_bf16 = random::normal::<f32>(&[b, h_kv, n, d], None, None, &key_k)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let v_rot_bf16 = random::normal::<f32>(&[b, h_kv, n, d], None, None, &key_v)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();

        let (k_codes, k_sigma) = lloyd_max_quantize_stage1_fused(&k_rot_bf16, &centroids).unwrap();
        let (v_codes, v_sigma) = lloyd_max_quantize_stage1_fused(&v_rot_bf16, &centroids).unwrap();

        // Reference: same 3-stage chain. qk_inline + sv_inline handle GQA
        // internally (`h_kv = h * H_kv / H`); softmax is shape-only.
        let scores_ref =
            super::turboquant_qk_inline(&q_bf16, &k_codes, &k_sigma, &centroids).unwrap();
        let last_axis = (scores_ref.ndim() as i32) - 1;
        let attn_w_ref =
            mlx_rs::ops::softmax_axis(&scores_ref, last_axis, Some(true)).unwrap();
        let out_ref =
            super::turboquant_sv_inline(&attn_w_ref, &v_codes, &v_sigma, &centroids).unwrap();

        let out_fused = super::turboquant_fused_attn(
            &q_bf16, &k_codes, &k_sigma, &v_codes, &v_sigma, &centroids, 1.0,
        )
        .unwrap();

        let r = out_ref.as_dtype(Dtype::Float32).unwrap();
        let f = out_fused.as_dtype(Dtype::Float32).unwrap();
        let diff = mlx_rs::ops::subtract(&r, &f).unwrap();
        let sq = mlx_rs::ops::multiply(&diff, &diff).unwrap();
        let nelems = (b * h * t * d) as f32;
        let mse_val = mlx_rs::ops::sum(&sq, false).unwrap().item::<f32>() / nelems;

        let r_sq = mlx_rs::ops::multiply(&r, &r).unwrap();
        let r_var = mlx_rs::ops::sum(&r_sq, false).unwrap().item::<f32>() / nelems;

        let rel = mse_val / (r_var + 1e-12);
        eprintln!(
            "fused_attn (GQA H=32 H_kv=8): MSE={:.3e}, ref_var={:.3e}, rel={:.3e}",
            mse_val, r_var, rel
        );
        assert!(rel < 1e-3, "fused_attn GQA rel MSE too large: rel={}", rel);
    }

    /// Full-attn shape variant: H=32, H_kv=2 (group=16), D=512 (global_head_dim).
    /// Tests the `_d512` source-duplicated kernel under tight GQA.
    #[test]
    fn fused_attn_d512_gqa_matches_reference() {
        use mlx_rs::random;
        let bits = 4u32;
        let b: i32 = 1;
        let h: i32 = 32;  // Gemma 4 full-attn H
        let h_kv: i32 = 2; // full-attn H_kv (group=16)
        let t: i32 = 1;
        let n: i32 = 64;
        let d: i32 = 512;
        let centroids = lloyd_max_centroids(bits).unwrap();

        let key_q = random::key(0xFAD512A01u64).unwrap();
        let key_k = random::key(0xFAD512A02u64).unwrap();
        let key_v = random::key(0xFAD512A03u64).unwrap();
        let q_bf16 = random::normal::<f32>(&[b, h, t, d], None, None, &key_q)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let k_rot_bf16 = random::normal::<f32>(&[b, h_kv, n, d], None, None, &key_k)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let v_rot_bf16 = random::normal::<f32>(&[b, h_kv, n, d], None, None, &key_v)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();

        let (k_codes, k_sigma) = lloyd_max_quantize_stage1_fused(&k_rot_bf16, &centroids).unwrap();
        let (v_codes, v_sigma) = lloyd_max_quantize_stage1_fused(&v_rot_bf16, &centroids).unwrap();

        let scores_ref =
            super::turboquant_qk_inline(&q_bf16, &k_codes, &k_sigma, &centroids).unwrap();
        let last_axis = (scores_ref.ndim() as i32) - 1;
        let attn_w_ref =
            mlx_rs::ops::softmax_axis(&scores_ref, last_axis, Some(true)).unwrap();
        let out_ref =
            super::turboquant_sv_inline(&attn_w_ref, &v_codes, &v_sigma, &centroids).unwrap();

        let out_fused = super::turboquant_fused_attn(
            &q_bf16, &k_codes, &k_sigma, &v_codes, &v_sigma, &centroids, 1.0,
        )
        .unwrap();

        let r = out_ref.as_dtype(Dtype::Float32).unwrap();
        let f = out_fused.as_dtype(Dtype::Float32).unwrap();
        let diff = mlx_rs::ops::subtract(&r, &f).unwrap();
        let sq = mlx_rs::ops::multiply(&diff, &diff).unwrap();
        let nelems = (b * h * t * d) as f32;
        let mse_val = mlx_rs::ops::sum(&sq, false).unwrap().item::<f32>() / nelems;

        let r_sq = mlx_rs::ops::multiply(&r, &r).unwrap();
        let r_var = mlx_rs::ops::sum(&r_sq, false).unwrap().item::<f32>() / nelems;

        let rel = mse_val / (r_var + 1e-12);
        eprintln!(
            "fused_attn (D=512 GQA H=32 H_kv=2): MSE={:.3e}, ref_var={:.3e}, rel={:.3e}",
            mse_val, r_var, rel
        );
        assert!(rel < 1e-3, "fused_attn D=512 GQA rel MSE too large: rel={}", rel);
    }

    /// Bit-packed 4-bit encode + packed qk_inline must produce results
    /// bit-identical (modulo bf16 store rounding) to the unpacked path.
    /// Validates both the encode kernel's packing and the inline kernel's
    /// unpacking are consistent.
    #[test]
    fn packed4_path_matches_unpacked_path() {
        use mlx_rs::random;
        let bits = 4u32;
        let b: i32 = 1;
        let h: i32 = 8;
        let h_kv: i32 = 8; // no GQA for this test
        let t: i32 = 1;
        let n: i32 = 128;
        let d: i32 = 256;
        let centroids = lloyd_max_centroids(bits).unwrap();

        let key_q = random::key(0xDEAD0001u64).unwrap();
        let key_k = random::key(0xBEEF0002u64).unwrap();
        let q_bf16 = random::normal::<f32>(&[b, h, t, d], None, None, &key_q)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let k_rot_bf16 = random::normal::<f32>(&[b, h_kv, n, d], None, None, &key_k)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();

        // Unpacked path (reference).
        let (k_codes, k_sigma) = lloyd_max_quantize_stage1_fused(&k_rot_bf16, &centroids).unwrap();
        let scores_unpacked =
            super::turboquant_qk_inline(&q_bf16, &k_codes, &k_sigma, &centroids).unwrap();

        // Packed path.
        let (k_codes_pkd, k_sigma_pkd) =
            super::lloyd_max_quantize_stage1_packed4(&k_rot_bf16, &centroids).unwrap();
        // Sanity: packed shape must be [..., D/8] uint32.
        assert_eq!(
            k_codes_pkd.shape(),
            &[b, h_kv, n, d / 8],
            "packed codes shape: expected [B, H_kv, N, D/8]"
        );
        assert_eq!(k_codes_pkd.dtype(), Dtype::Uint32);

        let scores_packed =
            super::turboquant_qk_inline_packed4(&q_bf16, &k_codes_pkd, &k_sigma_pkd, &centroids)
                .unwrap();

        // Compare f32 versions.
        let r = scores_unpacked.as_dtype(Dtype::Float32).unwrap();
        let f = scores_packed.as_dtype(Dtype::Float32).unwrap();
        let diff = mlx_rs::ops::subtract(&r, &f).unwrap();
        let sq = mlx_rs::ops::multiply(&diff, &diff).unwrap();
        let nelems = (b * h * t * n) as f32;
        let mse_val = mlx_rs::ops::sum(&sq, false).unwrap().item::<f32>() / nelems;
        let r_sq = mlx_rs::ops::multiply(&r, &r).unwrap();
        let r_var = mlx_rs::ops::sum(&r_sq, false).unwrap().item::<f32>() / nelems;
        let rel = mse_val / (r_var + 1e-12);
        eprintln!(
            "packed4 path vs unpacked: MSE={:.3e}, ref_var={:.3e}, rel={:.3e}",
            mse_val, r_var, rel
        );
        // Same math, just packed storage. Both paths produce bit-identical
        // scores up to the bf16 store rounding.
        assert!(
            rel < 1e-4,
            "packed4 vs unpacked relative MSE too large: rel={}",
            rel
        );
    }

    /// Microbenchmark: compare wall-clock of the packed vs unpacked qk_inline
    /// kernel at production-like shape (decode T=1, N=1024 sliding-window).
    /// Forces eval after each call to defeat lazy-graph batching. Prints
    /// the ratio; not a unit-test assertion (`#[ignore]` by default — opt in
    /// via `cargo test --release -- --ignored bench_packed4_vs_unpacked_qk`).
    #[test]
    #[ignore]
    fn bench_packed4_vs_unpacked_qk() {
        use mlx_rs::random;
        use std::time::Instant;

        let bits = 4u32;
        let b: i32 = 1;
        let h: i32 = 16; // production GQA: H=16 query heads
        let h_kv: i32 = 8; // 8 KV heads
        let t: i32 = 1;
        let n: i32 = 1024; // sliding-window size
        let d: i32 = 256;
        let centroids = lloyd_max_centroids(bits).unwrap();
        let key_q = random::key(0xC0FFEE01u64).unwrap();
        let key_k = random::key(0xC0FFEE02u64).unwrap();

        let q_bf16 = random::normal::<f32>(&[b, h, t, d], None, None, &key_q)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let k_rot_bf16 = random::normal::<f32>(&[b, h_kv, n, d], None, None, &key_k)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();

        // Pre-encode once for each path. eval to materialize.
        let (k_codes_u8, k_sigma_u8) =
            lloyd_max_quantize_stage1_fused(&k_rot_bf16, &centroids).unwrap();
        let (k_codes_pkd, k_sigma_pkd) =
            super::lloyd_max_quantize_stage1_packed4(&k_rot_bf16, &centroids).unwrap();
        mlx_rs::transforms::eval([&k_codes_u8, &k_sigma_u8, &k_codes_pkd, &k_sigma_pkd]).unwrap();

        // Warmup.
        let warmup_iters = 16;
        for _ in 0..warmup_iters {
            let s_u =
                super::turboquant_qk_inline(&q_bf16, &k_codes_u8, &k_sigma_u8, &centroids).unwrap();
            let s_p = super::turboquant_qk_inline_packed4(
                &q_bf16,
                &k_codes_pkd,
                &k_sigma_pkd,
                &centroids,
            )
            .unwrap();
            mlx_rs::transforms::eval([&s_u, &s_p]).unwrap();
        }

        // Timed loop. eval after each call so we measure GPU kernel + sync.
        let bench_iters = 500;
        let t0 = Instant::now();
        for _ in 0..bench_iters {
            let s =
                super::turboquant_qk_inline(&q_bf16, &k_codes_u8, &k_sigma_u8, &centroids).unwrap();
            mlx_rs::transforms::eval([&s]).unwrap();
        }
        let t_unpacked = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..bench_iters {
            let s = super::turboquant_qk_inline_packed4(
                &q_bf16,
                &k_codes_pkd,
                &k_sigma_pkd,
                &centroids,
            )
            .unwrap();
            mlx_rs::transforms::eval([&s]).unwrap();
        }
        let t_packed = t1.elapsed();

        let us_u = t_unpacked.as_micros() as f64 / bench_iters as f64;
        let us_p = t_packed.as_micros() as f64 / bench_iters as f64;
        let ratio = us_p / us_u;
        eprintln!(
            "qk_inline bench (N={}): unpacked={:.2} us/iter, packed4={:.2} us/iter, packed/unpacked={:.3}",
            n, us_u, us_p, ratio
        );
        eprintln!(
            "  -> packed4 is {:.1}% {} than unpacked",
            (1.0 - ratio).abs() * 100.0,
            if ratio < 1.0 { "FASTER" } else { "SLOWER" }
        );
    }

    /// Probe: does MLX accept `matmul(bf16, f32)` directly?
    /// If yes — we can skip the bf16→f32 input cast in `rotate_last_axis`
    /// since R is already f32 (orthogonality precision). MLX should auto-
    /// upcast internally with f32 accumulator (same path the explicit cast
    /// triggers, minus one host-side dispatch).
    ///
    /// Logs the output dtype + max-abs-error vs the cast-then-matmul path.
    #[test]
    fn probe_matmul_mixed_dtype_bf16_x_f32() {
        let d = 256usize;
        // bf16 input vector
        let x_bf16 = random::normal::<f32>(&[1, d as i32], None, None, None)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        // f32 rotation
        let r_f32 = rotation_matrix_f32(d, TURBOQUANT_SEED).unwrap();

        // Path A (current): cast → matmul(f32, f32) → cast back
        let xa_f32 = x_bf16.as_dtype(Dtype::Float32).unwrap();
        let ya_f32 = mlx_rs::ops::matmul(&xa_f32, &r_f32).unwrap();
        let ya_bf16 = ya_f32.as_dtype(Dtype::Bfloat16).unwrap();

        // Path B (mixed): matmul(bf16, f32) directly — what dtype comes out?
        let yb_attempt = mlx_rs::ops::matmul(&x_bf16, &r_f32);
        match yb_attempt {
            Ok(yb) => {
                let yb_dtype = yb.dtype();
                eprintln!(
                    "[probe] matmul(bf16, f32) OK, output dtype = {:?}",
                    yb_dtype
                );
                // Compare with Path A (cast to common dtype first for comparison).
                let yb_f32 = yb.as_dtype(Dtype::Float32).unwrap();
                let diff = mlx_rs::ops::subtract(&ya_f32, &yb_f32).unwrap();
                let abs = mlx_rs::ops::abs(&diff).unwrap();
                let max_err = mlx_rs::ops::max(&abs, false).unwrap().item::<f32>();
                let ya_max = mlx_rs::ops::max(&mlx_rs::ops::abs(&ya_f32).unwrap(), false)
                    .unwrap()
                    .item::<f32>();
                let rel = if ya_max > 0.0 {
                    max_err / ya_max
                } else {
                    max_err
                };
                eprintln!(
                    "[probe] |Path_A - Path_B|_max = {:.3e}, relative {:.3e}",
                    max_err, rel
                );
                // If bit-identical: rel ≈ 0. If f32 accum + bf16 output truncation: ~1/128.
                // If bf16 accum (precision loss): potentially larger.
                let _ = ya_bf16; // unused in B's existence is informational only
            }
            Err(e) => {
                eprintln!("[probe] matmul(bf16, f32) NOT SUPPORTED: {e}");
                eprintln!(
                    "[probe] -> B-1 lever blocked, must use explicit cast or write fused kernel"
                );
            }
        }
    }
}
