//! Single-query attention straight off an affine-quantized KV cache.
//!
//! Decode over a quantized cache otherwise runs three MLX ops: a
//! `quantized_matmul` for `Q·Kᵀ`, a softmax, and a `quantized_matmul` for the
//! weighted values. Measured against bf16 `sdpa_vector` on the full-attention
//! shapes the served models use (`examples/bench_sdpa_paths.rs`, task 021),
//! that path is 1.2-2x *slower* at 4K-64K keys even though it reads 2-4x fewer
//! bytes. This is MLX's two-pass vector kernel with the dequantization moved
//! into its inner loop, so the cache is read once, packed.
//!
//! Layout follows `sdpa_vector_2pass`: pass 1 runs one threadgroup per (KV
//! head, key block), one simdgroup per query head of the GQA group, each lane
//! holding `D / 32` contiguous elements of its query; keys are dealt to the
//! blocks round-robin and reduced with an online softmax. Pass 2 merges the
//! blocks. A lane's elements always fall in one quantization group, so a
//! key's score is `scale · Σ q·c + bias · Σ q` over the lane's codes `c`.
//!
//! The kernels are JIT-compiled through `mlx::fast::metal_kernel`, so they are
//! ordinary graph nodes on the caller's stream — no command buffer of their
//! own, no eval. Inputs are read through their strides: the quantized cache
//! hands out views into a preallocated buffer, and copying them row-contiguous
//! every step would cost more than the attention.

#[cfg(feature = "mlx-native")]
mod imp {
    use crate::metal_kernel::{MetalKernel, MetalKernelConfig};
    use anyhow::{Result, anyhow};
    use mlx_rs::{Array, Dtype};

    // Each lane's `D / 32` elements are handled as float4 quads. At 8 bits a
    // packed word is one quad, elements in order. At 4 bits a word holds eight
    // elements, byte k carrying elements 2k (low nibble) and 2k+1 (high), so
    // its low nibbles are quad (0, 2, 4, 6) and its high nibbles (1, 3, 5, 7);
    // the query and the accumulator are kept in that interleaved order and
    // unscrambled on the way out.
    const PASS1_SOURCE: &str = r#"
        constexpr int BD = 32;
        constexpr int QPT = D / BD;
        constexpr int NQ = QPT / 4;
        constexpr int PPT = QPT * BITS / 32;
        constexpr int GRP = GS;

        const uint lane = thread_position_in_threadgroup.x;
        const int g = thread_position_in_threadgroup.y;
        const int n_kv = kw_shape[1];
        const int z = threadgroup_position_in_grid.z;
        const int b = z / (n_kv * BLOCKS);
        const int kvh = (z / BLOCKS) % n_kv;
        const int blk = z % BLOCKS;
        const int n_heads = n_kv * GQA;
        const int qh = kvh * GQA + g;
        const int n = kw_shape[2];

        auto qp = q + b * q_strides[0] + qh * q_strides[1] + lane * QPT;
        float4 qv[NQ];
        float qsum = 0.0f;
        for (int i = 0; i < NQ; i++) {
            if (BITS == 8) {
                qv[i] = float4(qp[4 * i], qp[4 * i + 1], qp[4 * i + 2], qp[4 * i + 3]);
            } else {
                const int base = 8 * (i / 2) + (i % 2);
                qv[i] = float4(qp[base], qp[base + 2], qp[base + 4], qp[base + 6]);
            }
            qsum += qv[i].x + qv[i].y + qv[i].z + qv[i].w;
        }

        const int grp = (lane * QPT) / GRP;
        // Pointers start at this block's first key and advance BLOCKS rows a
        // step. Indexing `t * stride` instead costs a 64-bit multiply per
        // array per key, which the GPU emulates.
        auto kwp = kw + b * kw_strides[0] + kvh * kw_strides[1] + blk * kw_strides[2] + lane * PPT;
        auto ksp = ks + b * ks_strides[0] + kvh * ks_strides[1] + blk * ks_strides[2] + grp;
        auto kbp = kb + b * kb_strides[0] + kvh * kb_strides[1] + blk * kb_strides[2] + grp;
        auto vwp = vw + b * vw_strides[0] + kvh * vw_strides[1] + blk * vw_strides[2] + lane * PPT;
        auto vsp = vs + b * vs_strides[0] + kvh * vs_strides[1] + blk * vs_strides[2] + grp;
        auto vbp = vb + b * vb_strides[0] + kvh * vb_strides[1] + blk * vb_strides[2] + grp;
        const int kw_t = BLOCKS * int(kw_strides[2]);
        const int ks_t = BLOCKS * int(ks_strides[2]);
        const int kb_t = BLOCKS * int(kb_strides[2]);
        const int vw_t = BLOCKS * int(vw_strides[2]);
        const int vs_t = BLOCKS * int(vs_strides[2]);
        const int vb_t = BLOCKS * int(vb_strides[2]);

        float m = -INFINITY;
        float s = 0.0f;
        float4 o[NQ];
        for (int i = 0; i < NQ; i++) {
            o[i] = float4(0.0f);
        }

        // One key's dot product with the query (codes only; scale and bias
        // are applied by the caller), and one key's weighted values added to
        // the accumulator.
        #define LUMEN_QK(WP, DOT)                                                   \
            DOT = 0.0f;                                                             \
            for (int p = 0; p < PPT; p++) {                                         \
                const uint w = (WP)[p];                                             \
                if (BITS == 8) {                                                    \
                    DOT += metal::dot(qv[p], float4(as_type<uchar4>(w)));           \
                } else {                                                            \
                    DOT += metal::dot(qv[2 * p], float4(as_type<uchar4>(w & 0x0f0f0f0fu))) \
                        + metal::dot(qv[2 * p + 1],                                 \
                                     float4(as_type<uchar4>((w >> 4) & 0x0f0f0f0fu))); \
                }                                                                   \
            }
        #define LUMEN_PV(WP, ES, EB)                                                \
            for (int p = 0; p < PPT; p++) {                                         \
                const uint w = (WP)[p];                                             \
                if (BITS == 8) {                                                    \
                    o[p] += (ES) * float4(as_type<uchar4>(w)) + (EB);               \
                } else {                                                            \
                    o[2 * p] += (ES) * float4(as_type<uchar4>(w & 0x0f0f0f0fu)) + (EB); \
                    o[2 * p + 1] +=                                                 \
                        (ES) * float4(as_type<uchar4>((w >> 4) & 0x0f0f0f0fu)) + (EB); \
                }                                                                   \
            }

        // Two keys per step, so their loads and reductions overlap; the online
        // softmax folds both in with one rescale.
        int t = blk;
        for (; t + BLOCKS < n; t += 2 * BLOCKS) {
            float d0, d1;
            LUMEN_QK(kwp, d0);
            LUMEN_QK(kwp + kw_t, d1);
            float s0 = float(ksp[0]) * d0 + float(kbp[0]) * qsum;
            float s1 = float(ksp[ks_t]) * d1 + float(kbp[kb_t]) * qsum;
            s0 = simd_sum(s0);
            s1 = simd_sum(s1);
            const float new_m = max(m, max(s0, s1));
            const float factor = metal::fast::exp(m - new_m);
            const float e0 = metal::fast::exp(s0 - new_m);
            const float e1 = metal::fast::exp(s1 - new_m);
            m = new_m;
            s = s * factor + e0 + e1;
            for (int i = 0; i < NQ; i++) {
                o[i] *= factor;
            }
            LUMEN_PV(vwp, e0 * float(vsp[0]), e0 * float(vbp[0]));
            LUMEN_PV(vwp + vw_t, e1 * float(vsp[vs_t]), e1 * float(vbp[vb_t]));
            kwp += 2 * kw_t;
            ksp += 2 * ks_t;
            kbp += 2 * kb_t;
            vwp += 2 * vw_t;
            vsp += 2 * vs_t;
            vbp += 2 * vb_t;
        }
        if (t < n) {
            float d0;
            LUMEN_QK(kwp, d0);
            float s0 = simd_sum(float(ksp[0]) * d0 + float(kbp[0]) * qsum);
            const float new_m = max(m, s0);
            const float factor = metal::fast::exp(m - new_m);
            const float e0 = metal::fast::exp(s0 - new_m);
            m = new_m;
            s = s * factor + e0;
            for (int i = 0; i < NQ; i++) {
                o[i] *= factor;
            }
            LUMEN_PV(vwp, e0 * float(vsp[0]), e0 * float(vbp[0]));
        }
        #undef LUMEN_QK
        #undef LUMEN_PV

        const int row = (b * n_heads + qh) * BLOCKS + blk;
        auto dst = partials + row * D + lane * QPT;
        for (int i = 0; i < NQ; i++) {
            if (BITS == 8) {
                for (int c = 0; c < 4; c++) {
                    dst[4 * i + c] = o[i][c];
                }
            } else {
                const int base = 8 * (i / 2) + (i % 2);
                for (int c = 0; c < 4; c++) {
                    dst[base + 2 * c] = o[i][c];
                }
            }
        }
        if (lane == 0) {
            sums[row] = s;
            maxs[row] = m;
        }
    "#;

    const PASS2_SOURCE: &str = r#"
        const uint d = thread_position_in_grid.x;
        const uint row = thread_position_in_grid.y;
        const device float* mx = maxs + row * BLOCKS;
        const device float* sm = sums + row * BLOCKS;
        float top = -INFINITY;
        for (int i = 0; i < BLOCKS; i++) {
            top = max(top, mx[i]);
        }
        float total = 0.0f;
        float acc = 0.0f;
        for (int i = 0; i < BLOCKS; i++) {
            const float w = metal::fast::exp(mx[i] - top);
            total += sm[i] * w;
            acc += partials[(row * BLOCKS + i) * D + d] * w;
        }
        out[row * D + d] = static_cast<OutT>(acc / total);
    "#;

    /// Key blocks pass 1 deals `n` keys to: the counts `sdpa_vector_2pass`
    /// picks on M3-class devices, by key count and GQA group size.
    fn blocks_for(n: i32, gqa: i32) -> i32 {
        if n <= 1024 || gqa <= 4 {
            64
        } else if n <= 8192 {
            128
        } else if n <= 32768 {
            256
        } else if n <= 65536 {
            512
        } else {
            1024
        }
    }

    /// `softmax(q·Kᵀ)·V` for a single query position over an affine-quantized
    /// cache.
    ///
    /// * `queries`: `[B, H, 1, D]`, already carrying the softmax scale.
    /// * `keys` / `values`: `(packed u32, scales, biases)` as `mlx::quantize`
    ///   returns them, `[B, H_kv, S, ·]`, possibly strided views.
    /// * Needs `D / 32` elements per lane to fill whole packed words
    ///   (`D / 32 · bits` divisible by 32) and to sit inside one quantization
    ///   group (`group_size` divisible by `D / 32`). Errors otherwise, so the
    ///   caller can fall back to the three-op path.
    ///
    /// Returns `[B, H, 1, D]` in the queries' dtype.
    pub fn quantized_sdpa_decode(
        queries: &Array,
        keys: &(Array, Array, Array),
        values: &(Array, Array, Array),
        group_size: i32,
        bits: i32,
    ) -> Result<Array> {
        let qs = queries.shape();
        if qs.len() != 4 || qs[2] != 1 {
            return Err(anyhow!(
                "quantized_sdpa_decode: queries must be [B, H, 1, D], got {qs:?}"
            ));
        }
        let (b, h, d) = (qs[0], qs[1], qs[3]);
        let ks = keys.0.shape();
        let (h_kv, n) = (ks[1], ks[2]);
        if h_kv == 0 || h % h_kv != 0 || h / h_kv > 32 {
            return Err(anyhow!(
                "quantized_sdpa_decode: {h} query heads over {h_kv} KV heads"
            ));
        }
        let per_lane = d / 32;
        if d % 32 != 0
            || per_lane % 4 != 0
            || !(bits == 4 || bits == 8)
            || (per_lane * bits) % 32 != 0
            || group_size % per_lane != 0
        {
            return Err(anyhow!(
                "quantized_sdpa_decode: unsupported D={d} bits={bits} group_size={group_size}"
            ));
        }
        let gqa = h / h_kv;
        let blocks = blocks_for(n, gqa);

        let pass1 = MetalKernel::new(
            "lumen_quantized_sdpa_decode_pass1",
            &["q", "kw", "ks", "kb", "vw", "vs", "vb"],
            &["partials", "sums", "maxs"],
            PASS1_SOURCE,
            /* ensure_row_contiguous */ false,
            /* atomic_outputs */ false,
        )?;
        let config = MetalKernelConfig::new();
        config.add_template_arg_int("D", d)?;
        config.add_template_arg_int("BITS", bits)?;
        config.add_template_arg_int("GS", group_size)?;
        config.add_template_arg_int("BLOCKS", blocks)?;
        config.add_template_arg_int("GQA", gqa)?;
        config.set_grid(32, gqa, b * h_kv * blocks)?;
        config.set_thread_group(32, gqa, 1)?;
        let rows = b * h * blocks;
        config.add_output_arg(&[rows * d], Dtype::Float32)?;
        config.add_output_arg(&[rows], Dtype::Float32)?;
        config.add_output_arg(&[rows], Dtype::Float32)?;
        let mut pass1_out = pass1
            .apply(
                &[
                    queries, &keys.0, &keys.1, &keys.2, &values.0, &values.1, &values.2,
                ],
                &config,
                3,
            )?
            .into_iter();
        let (partials, sums, maxs) = (
            pass1_out.next().expect("pass 1 partials"),
            pass1_out.next().expect("pass 1 sums"),
            pass1_out.next().expect("pass 1 maxs"),
        );

        let pass2 = MetalKernel::new(
            "lumen_quantized_sdpa_decode_pass2",
            &["partials", "sums", "maxs"],
            &["out"],
            PASS2_SOURCE,
            /* ensure_row_contiguous */ true,
            /* atomic_outputs */ false,
        )?;
        let config = MetalKernelConfig::new();
        config.add_template_arg_int("D", d)?;
        config.add_template_arg_int("BLOCKS", blocks)?;
        config.add_template_arg_dtype("OutT", queries.dtype())?;
        config.set_grid(d, b * h, 1)?;
        config.set_thread_group(d.min(1024), 1, 1)?;
        config.add_output_arg(&[b, h, 1, d], queries.dtype())?;
        let out = pass2
            .apply(&[&partials, &sums, &maxs], &config, 1)?
            .into_iter()
            .next()
            .expect("pass 2 output");
        Ok(out)
    }
}

#[cfg(feature = "mlx-native")]
pub use imp::quantized_sdpa_decode;

// The fused kernel against SDPA on the same cache dequantized: the exact
// computation it replaces, minus the packing.
#[cfg(all(test, feature = "mlx-native"))]
mod tests {
    use super::imp::quantized_sdpa_decode;
    use mlx_rs::ops::indexing::{Ellipsis, IndexOp};
    use mlx_rs::{Array, Dtype, random};

    fn normal(shape: &[i32], std_dev: f32, seed: u64) -> Array {
        let key = random::key(seed).unwrap();
        random::normal::<f32>(shape, None, Some(std_dev), &key)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap()
    }

    fn cosine(a: &Array, b: &Array) -> f64 {
        let flat = |x: &Array| {
            let x = x.as_dtype(Dtype::Float32).unwrap().reshape(&[-1]).unwrap();
            x.eval().unwrap();
            x.as_slice::<f32>().to_vec()
        };
        let (a, b) = (flat(a), flat(b));
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (&x, &y) in a.iter().zip(&b) {
            dot += x as f64 * y as f64;
            na += x as f64 * x as f64;
            nb += y as f64 * y as f64;
        }
        dot / (na.sqrt() * nb.sqrt())
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn fused_quantized_decode_matches_sdpa_on_the_dequantized_cache() {
        const GROUP_SIZE: i32 = 64;
        // (heads, kv_heads, head_dim, keys): the served full-attention shapes,
        // a key count below the block count, and one far above it.
        let cases = [
            (16, 4, 256, 4100),
            (24, 4, 256, 777),
            (16, 2, 256, 20),
            (16, 2, 512, 9000),
            (16, 8, 256, 40000),
        ];
        let mut failures = Vec::new();
        for bits in [8, 4] {
            for (i, &(h, h_kv, d, n)) in cases.iter().enumerate() {
                let seed = 0xf5ed + 100 * i as u64 + bits as u64;
                let q = normal(&[1, h, 1, d], 1.0 / (d as f32).sqrt(), seed);
                // A view into a longer buffer, as the cache hands it out.
                let quantize = |s: u64| {
                    let buf = normal(&[1, h_kv, n + 256, d], 1.0, s);
                    let (w, sc, bi) = mlx_rs::ops::quantize(&buf, GROUP_SIZE, bits).unwrap();
                    let cut = |a: Array| a.index((Ellipsis, 0..n, ..));
                    (cut(w), cut(sc), cut(bi))
                };
                let (k, v) = (quantize(seed + 1), quantize(seed + 2));
                let dequantize = |t: &(Array, Array, Array)| {
                    mlx_rs::ops::dequantize(&t.0, &t.1, &t.2, GROUP_SIZE, bits).unwrap()
                };
                let want = mlx_rs::fast::scaled_dot_product_attention(
                    &q,
                    dequantize(&k),
                    dequantize(&v),
                    1.0,
                    None,
                    None,
                )
                .unwrap();
                let got = quantized_sdpa_decode(&q, &k, &v, GROUP_SIZE, bits).expect("kernel");
                assert_eq!(got.shape(), want.shape());
                let cos = cosine(&got, &want);
                eprintln!("[quantized-sdpa] bits={bits} h={h}/{h_kv} d={d} n={n} cos={cos:.6}");
                if cos < 0.9999 {
                    failures.push(format!(
                        "bits={bits} h={h}/{h_kv} d={d} n={n}: cos {cos:.6}"
                    ));
                }
            }
        }
        assert!(failures.is_empty(), "fused kernel disagrees: {failures:#?}");
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn fused_quantized_decode_rejects_layouts_it_cannot_read() {
        let q = normal(&[1, 16, 1, 128], 1.0, 1);
        let buf = normal(&[1, 2, 64, 128], 1.0, 2);
        let t = mlx_rs::ops::quantize(&buf, 64, 4).unwrap();
        // 4 elements per lane at 4 bits is half a packed word.
        assert!(quantized_sdpa_decode(&q, &t, &t, 64, 4).is_err());
        let q2 = normal(&[1, 16, 2, 256], 1.0, 3);
        let buf2 = normal(&[1, 2, 64, 256], 1.0, 4);
        let t2 = mlx_rs::ops::quantize(&buf2, 64, 8).unwrap();
        assert!(
            quantized_sdpa_decode(&q2, &t2, &t2, 64, 8).is_err(),
            "two query rows"
        );
    }
}
