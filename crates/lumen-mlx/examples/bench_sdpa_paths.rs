//! Attention paths at the full-attention shapes of the models lumen serves
//! (task 021, P0): what each costs, how much memory it peaks at, and whether
//! it computes the same thing.
//!
//! Prefill runs one chunk of queries at the bottom-right of `KV` keys, with the
//! chunk the production scores clamp would pick there. Decode runs one query.
//! Keys and values are a slice of a longer buffer, as the KV cache returns
//! them.
//!
//! Arms:
//!   sdpa        `mlx::fast::scaled_dot_product_attention`, as production calls
//!               it. Which kernel that is depends on the process latches below:
//!               by default D=256/512 causal prefill takes MLX's unfused
//!               fallback, which materializes the scores.
//!   fa2         the fork's `lumen_flash_attn_prefill_bf16` (prefill only).
//!   fa2-qview   the same, with Q as the transposed view the real call site
//!               holds (`[B, L, H, D]` → `[B, H, L, D]`, not row-contiguous).
//!   q8-3op /    decode over 8- / 4-bit affine-quantized K/V (group 64) as
//!   q4-3op      Gemma 4's quantized branches run it: two `quantized_matmul`s
//!               around a softmax (decode only).
//!   q8-fused /  the same caches through `native_quant_attention`'s fused
//!   q4-fused    two-pass kernel (decode only).
//!
//! Each arm reports mean / p50 / min / sd over `ITERS` synchronized calls, the
//! peak MLX memory one call adds, cosine against `sdpa`, and Welch's t
//! against `sdpa` (negative = faster).
//!
//! Env:
//!   MODELS  comma list of qwen9b, qwen27b, qwen35b, gemma-full (default all)
//!   KV      comma list of key counts (default 4096,16384,32768,65536)
//!   PHASE   prefill | decode | both (default both)
//!   ITERS   timed calls per arm (default 27)
//!   ARMS    comma list of fa2, fa2-qview, q8-3op, q4-3op, q8-fused, q4-fused
//!           to run besides sdpa (default all)
//!
//! Process latches (read once by the MLX fork; set them for a separate run):
//!   LUMEN_GEMMA4_PREFILL_FAST_BD256=1  `sdpa` takes steel attention at D=256
//!   LUMEN_SDPA_VECTOR_D512=1           `sdpa` takes sdpa_vector at D=512 decode
//!
//! Run:
//!   cargo run --release -p lumen-mlx --features mlx-native \
//!       --example bench_sdpa_paths

#[cfg(not(feature = "mlx-native"))]
fn main() {
    eprintln!("rebuild with --features mlx-native");
}

#[cfg(feature = "mlx-native")]
fn main() -> anyhow::Result<()> {
    imp::run()
}

#[cfg(feature = "mlx-native")]
mod imp {
    use anyhow::{Context, Result, anyhow};
    use lumen_mlx::metal_memory;
    use lumen_mlx::native_quant_attention::quantized_sdpa_decode;
    use lumen_mlx::prefill_budget::{max_safe_chunk, scores_budget_from_env};
    use mlx_rs::fast::ScaledDotProductAttentionMask;
    use mlx_rs::ops::indexing::{Ellipsis, IndexOp};
    use mlx_rs::{Array, Dtype, Stream, random};
    use std::time::Instant;

    /// A model's full-attention layer: query heads, KV heads, head dim, and the
    /// env var that sets its prefill scores budget.
    struct Shape {
        name: &'static str,
        heads: i32,
        kv_heads: i32,
        head_dim: i32,
        budget_env: &'static str,
    }

    const SHAPES: &[Shape] = &[
        Shape {
            name: "qwen9b",
            heads: 16,
            kv_heads: 4,
            head_dim: 256,
            budget_env: "LUMEN_QWEN35_PREFILL_SCORES_GB",
        },
        Shape {
            name: "qwen27b",
            heads: 24,
            kv_heads: 4,
            head_dim: 256,
            budget_env: "LUMEN_QWEN35_PREFILL_SCORES_GB",
        },
        Shape {
            name: "qwen35b",
            heads: 16,
            kv_heads: 2,
            head_dim: 256,
            budget_env: "LUMEN_QWEN35_PREFILL_SCORES_GB",
        },
        Shape {
            name: "gemma-full",
            heads: 16,
            kv_heads: 2,
            head_dim: 512,
            budget_env: "LUMEN_GEMMA4_PREFILL_SCORES_GB",
        },
    ];

    /// Production prefill chunk before the scores clamp.
    const CHUNK: usize = 2048;
    const WARMUP: usize = 3;
    /// Affine KV quantization group, as the quantized caches use by default.
    const GROUP_SIZE: i32 = 64;

    /// `(packed, scales, biases)`.
    type Triple = (Array, Array, Array);

    /// Decode over an affine-quantized cache the way Gemma 4's quantized
    /// branches do it: query heads folded per KV head, `quantized_matmul` for
    /// `Q·Kᵀ`, softmax, `quantized_matmul` for the weighted values. `q` must
    /// already carry the softmax scale.
    fn quantized_decode(q: &Array, k: &Triple, v: &Triple, bits: i32) -> Result<Array> {
        let s = q.shape();
        let (b, h, l, d) = (s[0], s[1], s[2], s[3]);
        let h_kv = k.0.shape()[1];
        let q = mlx_rs::ops::reshape(q, &[b, h_kv, h / h_kv, l, d])?;
        let exp = |a: &Array| mlx_rs::ops::expand_dims(a, -3);
        let (k, v) = (
            (exp(&k.0)?, exp(&k.1)?, exp(&k.2)?),
            (exp(&v.0)?, exp(&v.1)?, exp(&v.2)?),
        );
        let scores =
            mlx_rs::ops::quantized_matmul(&q, &k.0, &k.1, Some(&k.2), true, GROUP_SIZE, bits)?;
        let probs = mlx_rs::ops::softmax_axis(&scores, -1, Some(true))?;
        let out =
            mlx_rs::ops::quantized_matmul(&probs, &v.0, &v.1, Some(&v.2), false, GROUP_SIZE, bits)?;
        Ok(mlx_rs::ops::reshape(&out, &[b, h, l, d])?)
    }

    struct Stats {
        mean: f64,
        p50: f64,
        min: f64,
        sd: f64,
        n: usize,
    }

    impl Stats {
        fn of(mut ms: Vec<f64>) -> Self {
            ms.sort_by(f64::total_cmp);
            let n = ms.len();
            let mean = ms.iter().sum::<f64>() / n as f64;
            let var = ms.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
            Stats {
                mean,
                p50: ms[n / 2],
                min: ms[0],
                sd: var.sqrt(),
                n,
            }
        }

        /// Welch's t of `self` against `base`: negative when `self` is faster.
        fn welch(&self, base: &Stats) -> f64 {
            let se = (self.sd.powi(2) / self.n as f64 + base.sd.powi(2) / base.n as f64).sqrt();
            (self.mean - base.mean) / se.max(1e-12)
        }
    }

    fn env_list(name: &str) -> Option<Vec<String>> {
        std::env::var(name)
            .ok()
            .map(|v| v.split(',').map(|s| s.trim().to_string()).collect())
    }

    fn normal_bf16(shape: &[i32], seed: u64) -> Result<Array> {
        let key = random::key(seed)?;
        Ok(random::normal::<f32>(shape, None, None, &key)?.as_dtype(Dtype::Bfloat16)?)
    }

    fn cosine(a: &Array, b: &Array) -> Result<f64> {
        let flat = |x: &Array| -> Result<Vec<f32>> {
            let x = x.as_dtype(Dtype::Float32)?.reshape(&[-1])?;
            x.eval()?;
            Ok(x.as_slice::<f32>().to_vec())
        };
        let (a, b) = (flat(a)?, flat(b)?);
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (&x, &y) in a.iter().zip(&b) {
            dot += x as f64 * y as f64;
            na += x as f64 * x as f64;
            nb += y as f64 * y as f64;
        }
        Ok(dot / (na.sqrt() * nb.sqrt()))
    }

    /// Wait until the GPU is done with `stream`. `eval` returns when the
    /// result is ready, but MLX frees a graph's temporaries in the command
    /// buffer's completion handler, so memory read right after `eval` still
    /// counts the previous call's scores.
    fn synchronize(stream: &Stream) {
        // SAFETY: a valid stream handle; MLX blocks until its queue drains.
        unsafe {
            mlx_sys::mlx_synchronize(stream.as_ptr());
        }
    }

    /// Time `iters` synchronized calls of `f`, after a warmup, and measure the
    /// peak memory one call adds on top of what is live before it.
    fn measure(
        stream: &Stream,
        iters: usize,
        f: &dyn Fn() -> Result<Array>,
    ) -> Result<(Array, Stats, f64)> {
        for _ in 0..WARMUP {
            f()?.eval()?;
        }
        synchronize(stream);
        metal_memory::clear_cache()?;
        let before = metal_memory::get_active_memory()?;
        metal_memory::reset_peak_memory();
        let out = f()?;
        out.eval()?;
        synchronize(stream);
        let peak = metal_memory::get_peak_memory()?;
        let peak_gb = (peak.saturating_sub(before)) as f64 / 1e9;
        let mut ms = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            f()?.eval()?;
            ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        Ok((out, Stats::of(ms), peak_gb))
    }

    fn report(label: &str, arm: &str, stats: &Stats, peak_gb: f64, cos: f64, base: Option<&Stats>) {
        let t = base.map_or(String::from("-"), |b| format!("{:+.1}", stats.welch(b)));
        println!(
            "{label} {arm:<10} mean={:9.3}ms p50={:9.3}ms min={:9.3}ms sd={:7.3} \
             peak+={:6.2}GB cos={cos:.6} t={t}",
            stats.mean, stats.p50, stats.min, stats.sd, peak_gb
        );
    }

    pub fn run() -> Result<()> {
        let models = env_list("MODELS");
        let arms = env_list("ARMS");
        let kvs: Vec<usize> = env_list("KV")
            .map(|v| v.iter().map(|s| s.parse()).collect::<Result<_, _>>())
            .transpose()
            .context("KV must be a comma list of integers")?
            .unwrap_or_else(|| vec![4096, 16384, 32768, 65536]);
        let phase = std::env::var("PHASE").unwrap_or_else(|_| "both".into());
        let iters: usize = std::env::var("ITERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n >= 2)
            .unwrap_or(27);
        let latch = |k: &str| std::env::var(k).map(|v| v == "1").unwrap_or(false);
        println!(
            "bench_sdpa_paths: iters={iters} phase={phase} BD256={} VECTOR_D512={}",
            latch("LUMEN_GEMMA4_PREFILL_FAST_BD256"),
            latch("LUMEN_SDPA_VECTOR_D512"),
        );
        let stream = Stream::gpu();
        for shape in SHAPES {
            if models
                .as_ref()
                .is_some_and(|m| !m.iter().any(|n| n == shape.name))
            {
                continue;
            }
            let scale = 1.0 / (shape.head_dim as f32).sqrt();
            for &kv in &kvs {
                // Keys come from a longer buffer, as the KV cache's step
                // prealloc hands them out: a strided view, not a fresh array.
                let cap = kv as i32 + 256;
                let seed = (shape.heads * 1000 + kv as i32) as u64;
                let k_buf = normal_bf16(&[1, shape.kv_heads, cap, shape.head_dim], seed + 1)?;
                let v_buf = normal_bf16(&[1, shape.kv_heads, cap, shape.head_dim], seed + 2)?;
                let k = k_buf.index((Ellipsis, 0..kv as i32, ..));
                let v = v_buf.index((Ellipsis, 0..kv as i32, ..));

                if phase != "decode" {
                    let budget = scores_budget_from_env(shape.budget_env);
                    let lq = CHUNK
                        .min(max_safe_chunk(budget, shape.heads as usize, kv))
                        .min(kv);
                    let label = format!(
                        "[prefill {} q={lq} k={kv} h={}/{} d={}]",
                        shape.name, shape.heads, shape.kv_heads, shape.head_dim
                    );
                    let q = normal_bf16(&[1, shape.heads, lq as i32, shape.head_dim], seed)?;
                    // Same values laid out [B, L, H, D] and viewed back.
                    let q_blhd = mlx_rs::ops::add(
                        &mlx_rs::ops::transpose_axes(&q, &[0, 2, 1, 3])?,
                        &Array::from_f32(0.0).as_dtype(Dtype::Bfloat16)?,
                    )?;
                    let q_view = mlx_rs::ops::transpose_axes(&q_blhd, &[0, 2, 1, 3])?;
                    let offset = (kv - lq) as u32;
                    let sdpa = |q: &Array| {
                        mlx_rs::fast::scaled_dot_product_attention(
                            q,
                            &k,
                            &v,
                            scale,
                            Some(ScaledDotProductAttentionMask::Causal),
                            None,
                        )
                        .map_err(|e| anyhow!("sdpa: {e}"))
                    };
                    let fa2 = |q: &Array| {
                        mlx_rs::metal::lumen_flash_attn_prefill_bf16(
                            q, &k, &v, scale, 0, offset, &stream,
                        )
                        .map_err(|e| anyhow!("fa2: {e}"))
                    };
                    let (base_out, base, base_peak) = measure(&stream, iters, &|| sdpa(&q_view))?;
                    report(&label, "sdpa", &base, base_peak, 1.0, None);
                    for (arm, q_arm) in [("fa2", &q), ("fa2-qview", &q_view)] {
                        if arms.as_ref().is_some_and(|a| !a.iter().any(|n| n == arm)) {
                            continue;
                        }
                        let (out, stats, peak) = measure(&stream, iters, &|| fa2(q_arm))?;
                        let cos = cosine(&out, &base_out)?;
                        report(&label, arm, &stats, peak, cos, Some(&base));
                    }
                }

                if phase != "prefill" {
                    let label = format!(
                        "[decode  {} q=1 k={kv} h={}/{} d={}]",
                        shape.name, shape.heads, shape.kv_heads, shape.head_dim
                    );
                    let q = normal_bf16(&[1, shape.heads, 1, shape.head_dim], seed + 3)?;
                    let (base_out, base, peak) = measure(&stream, iters, &|| {
                        mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, scale, None, None)
                            .map_err(|e| anyhow!("sdpa decode: {e}"))
                    })?;
                    report(&label, "sdpa", &base, peak, 1.0, None);
                    let q_scaled = mlx_rs::ops::multiply(
                        &q,
                        &Array::from_f32(scale).as_dtype(Dtype::Bfloat16)?,
                    )?;
                    let wanted =
                        |arm: &str| arms.as_ref().is_none_or(|a| a.iter().any(|n| n == arm));
                    for bits in [8, 4] {
                        let (three_op, fused) = (format!("q{bits}-3op"), format!("q{bits}-fused"));
                        if !wanted(&three_op) && !wanted(&fused) {
                            continue;
                        }
                        // Quantize the whole buffer and slice, as the
                        // quantized cache's prealloc hands its triples out.
                        let quantize = |buf: &Array| -> Result<Triple> {
                            let (w, s, b) = mlx_rs::ops::quantize(buf, GROUP_SIZE, bits)?;
                            let cut = |a: Array| a.index((Ellipsis, 0..kv as i32, ..));
                            Ok((cut(w), cut(s), cut(b)))
                        };
                        let (kq, vq) = (quantize(&k_buf)?, quantize(&v_buf)?);
                        if wanted(&three_op) {
                            let (out, stats, peak) = measure(&stream, iters, &|| {
                                quantized_decode(&q_scaled, &kq, &vq, bits)
                            })?;
                            let cos = cosine(&out, &base_out)?;
                            report(&label, &three_op, &stats, peak, cos, Some(&base));
                        }
                        if wanted(&fused) {
                            let (out, stats, peak) = measure(&stream, iters, &|| {
                                quantized_sdpa_decode(&q_scaled, &kq, &vq, GROUP_SIZE, bits)
                            })?;
                            let cos = cosine(&out, &base_out)?;
                            report(&label, &fused, &stats, peak, cos, Some(&base));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
