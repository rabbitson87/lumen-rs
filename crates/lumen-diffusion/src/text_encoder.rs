//! Native FLUX.2 text encoder: Mistral-Small-3.2-24B, text-only (Phase 1).
//!
//! A faithful mechanical port of the Mistral decoder stack used by FLUX.2-dev
//! to embed the prompt. FLUX does **not** run the LM head; instead it captures
//! the raw hidden states *after* decoder layers 10, 20 and 30, stacks them, and
//! flattens to `[1, T, 3 * hidden]`. We reproduce exactly that.
//!
//! Config (Mistral-Small-3.2-24B): `hidden = 5120`, `n_layers = 40`,
//! `n_heads = 32`, `n_kv_heads = 8` (GQA), `head_dim = 128`,
//! `rope_theta = 1e9`, `rms_norm_eps = 1e-5`, `intermediate = 32768`,
//! `vocab = 131072`, no sliding window.
//!
//! Weights come in two forms, auto-detected per tensor by the presence of a
//! `.scales` sibling (exactly like [`crate::dit`]):
//!
//! - **mlx-lm affine 4-bit** (`group_size = 64`, `bits = 4`): every projection
//!   stays in its packed `{weight(U32),scales,biases}` form and is applied via
//!   [`crate::ops::qlinear`]; `embed_tokens` is dequantized and gathered.
//!   Source: `mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit`.
//! - **dense BF16** (the official `black-forest-labs/FLUX.2-dev/text_encoder/`):
//!   each projection is a single BF16 `.weight` `[out,in]` applied as
//!   `x @ Wᵀ`; `embed_tokens.weight` is a dense BF16 `[vocab,hidden]` table
//!   gathered directly by token id (no dequant). For 128 GB machines.
//!
//! Both forms share identical key names and architecture, spread across
//! safetensors shards keyed by `model.safetensors.index.json`. The 4-bit vs
//! bf16 choice is purely `.scales`-presence — the same loader handles both.
//!
//! ## Mask (causal + left-pad)
//!
//! The 512-token prompt is **left-padded** (pad id 11) with the real tokens at
//! `first_real..T`. The additive attention mask allows position `i` to attend
//! to `j` iff `first_real <= j <= i`, else `-1e9`. A plain causal (or
//! bidirectional) mask does *not* match the validated Swift reference.

#[cfg(feature = "mlx-native")]
mod imp {
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};

    use anyhow::{Context, Result, anyhow, bail};
    use mlx_rs::Array;

    use crate::ops::linear_nobias::linear_nobias;
    use crate::ops::qlinear::{BITS, GROUP_SIZE, QuantLinear, qlinear};
    use crate::ops::silu::silu;

    // ── linear (mlx affine-quantized 4-bit OR dense BF16) ─────────────────────
    //
    // Mirrors `crate::dit::Linear`: the 4-bit encoder stores each projection as
    // a packed `{weight,scales,biases}` triple; the bf16 encoder stores a single
    // dense `[out,in]` BF16 weight (already transposed to `[in,out]` at load).
    // A single [`linear`] dispatch hides the difference so the forward pass is
    // byte-identical for the 4-bit path and correct for the bf16 path.

    /// One Mistral projection: mlx affine 4-bit packed, or dense BF16.
    enum Linear {
        /// mlx affine 4-bit packed `{weight,scales,biases}` (applied via `qlinear`).
        Quant(QuantLinear),
        /// Dense `[in,out]` BF16 weight (already transposed) — `matmul(x, w)`.
        Dense(Array),
    }

    /// Apply a [`Linear`]: quantized → `qlinear` (unchanged 4-bit path), dense →
    /// bias-free `x @ Wᵀ`.
    #[inline]
    fn linear(x: &Array, l: &Linear) -> Result<Array> {
        match l {
            Linear::Quant(q) => qlinear(x, q),
            Linear::Dense(w) => linear_nobias(x, w),
        }
    }

    // ── model config ──────────────────────────────────────────────────────────
    pub const HIDDEN: i32 = 5120;
    pub const N_HEADS: i32 = 32;
    pub const N_KV_HEADS: i32 = 8;
    pub const HEAD_DIM: i32 = 128;
    pub const ROPE_THETA: f32 = 1e9;
    pub const RMS_EPS: f32 = 1e-5;
    /// Hidden states captured after these (1-based) decoder layers.
    pub const CAPTURE_LAYERS: [usize; 3] = [10, 20, 30];
    /// Only the layers up to the deepest capture point are needed — FLUX.2
    /// extracts hidden states after layer 30, so decoder layers 31..=40, the
    /// final `norm`, and `lm_head` are never used and are skipped at load. This
    /// trims the encoder's resident footprint (~13 GB → ~10 GB) and its forward
    /// (40 → 30 layers), enabling encoder + DiT co-residence on a 36 GB Mac.
    /// = `max(CAPTURE_LAYERS)`.
    pub const LOAD_LAYERS: usize = 30;

    // ── sharded safetensors loader ──────────────────────────────────────────────

    struct TensorMeta {
        dtype: String,
        shape: Vec<i32>,
        start: usize,
        end: usize,
    }

    /// One safetensors shard: parsed JSON header + raw tensor bytes.
    struct Shard {
        metas: HashMap<String, TensorMeta>,
        data_start: usize,
        blob: Vec<u8>,
    }

    impl Shard {
        fn load(path: impl AsRef<Path>) -> Result<Self> {
            let blob = std::fs::read(path.as_ref())
                .with_context(|| format!("reading shard {:?}", path.as_ref()))?;
            if blob.len() < 8 {
                bail!("safetensors shard too short");
            }
            let hlen = u64::from_le_bytes(blob[0..8].try_into().unwrap()) as usize;
            let header_end = 8 + hlen;
            if blob.len() < header_end {
                bail!("safetensors header length {hlen} exceeds shard size");
            }
            let header: serde_json::Value = serde_json::from_slice(&blob[8..header_end])
                .context("parsing safetensors JSON header")?;
            let obj = header
                .as_object()
                .ok_or_else(|| anyhow!("safetensors header is not an object"))?;
            let mut metas = HashMap::new();
            for (name, v) in obj {
                if name == "__metadata__" {
                    continue;
                }
                let dtype = v
                    .get("dtype")
                    .and_then(|d| d.as_str())
                    .ok_or_else(|| anyhow!("tensor {name} missing dtype"))?
                    .to_string();
                let shape: Vec<i32> = v
                    .get("shape")
                    .and_then(|s| s.as_array())
                    .ok_or_else(|| anyhow!("tensor {name} missing shape"))?
                    .iter()
                    .map(|d| d.as_i64().map(|x| x as i32))
                    .collect::<Option<_>>()
                    .ok_or_else(|| anyhow!("tensor {name} shape not all ints"))?;
                let offs = v
                    .get("data_offsets")
                    .and_then(|o| o.as_array())
                    .ok_or_else(|| anyhow!("tensor {name} missing data_offsets"))?;
                let start = offs[0].as_u64().unwrap() as usize;
                let end = offs[1].as_u64().unwrap() as usize;
                metas.insert(
                    name.clone(),
                    TensorMeta {
                        dtype,
                        shape,
                        start,
                        end,
                    },
                );
            }
            Ok(Self {
                metas,
                data_start: header_end,
                blob,
            })
        }

        fn bytes(&self, m: &TensorMeta) -> &[u8] {
            &self.blob[self.data_start + m.start..self.data_start + m.end]
        }
    }

    /// All shards of a model + the key→shard map from `index.json`.
    pub struct ShardedModel {
        shards: Vec<Shard>,
        /// tensor name → index into `shards`.
        key_shard: HashMap<String, usize>,
    }

    impl ShardedModel {
        /// Load every shard referenced by `<dir>/model.safetensors.index.json`.
        pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
            let dir = dir.as_ref();
            let index_path = dir.join("model.safetensors.index.json");
            let index: serde_json::Value = serde_json::from_slice(
                &std::fs::read(&index_path)
                    .with_context(|| format!("reading index {index_path:?}"))?,
            )
            .context("parsing model.safetensors.index.json")?;
            let weight_map = index
                .get("weight_map")
                .and_then(|m| m.as_object())
                .ok_or_else(|| anyhow!("index.json missing weight_map"))?;

            // Distinct shard filenames in deterministic order.
            let mut shard_files: Vec<String> = weight_map
                .values()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect();
            shard_files.sort();
            shard_files.dedup();

            let mut file_index: HashMap<String, usize> = HashMap::new();
            let mut shards = Vec::with_capacity(shard_files.len());
            for (i, f) in shard_files.iter().enumerate() {
                shards.push(Shard::load(dir.join(f))?);
                file_index.insert(f.clone(), i);
            }

            let mut key_shard = HashMap::new();
            for (k, v) in weight_map {
                if let Some(f) = v.as_str() {
                    if let Some(&si) = file_index.get(f) {
                        key_shard.insert(k.clone(), si);
                    }
                }
            }
            Ok(Self { shards, key_shard })
        }

        /// True iff a tensor with this exact name exists in any shard.
        fn has(&self, name: &str) -> bool {
            self.key_shard.contains_key(name)
        }

        fn meta(&self, name: &str) -> Result<(&Shard, &TensorMeta)> {
            let &si = self
                .key_shard
                .get(name)
                .ok_or_else(|| anyhow!("tensor not in index: {name}"))?;
            let shard = &self.shards[si];
            let m = shard
                .metas
                .get(name)
                .ok_or_else(|| anyhow!("tensor not found in shard: {name}"))?;
            Ok((shard, m))
        }

        /// Load a `U32` packed tensor as-is (shape preserved). Used for the
        /// packed 4-bit quantized weight matrices.
        fn u32_tensor(&self, name: &str) -> Result<Array> {
            let (shard, m) = self.meta(name)?;
            if m.dtype != "U32" {
                bail!("expected U32 for {name}, got {}", m.dtype);
            }
            let bytes = shard.bytes(m);
            let data: Vec<u32> = bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Ok(Array::from_slice(&data, &m.shape))
        }

        /// Load a `BF16` (or `F16`/`F32`) tensor as an f32 [`Array`].
        fn f32_tensor(&self, name: &str) -> Result<Array> {
            let (shard, m) = self.meta(name)?;
            let bytes = shard.bytes(m);
            let data: Vec<f32> = match m.dtype.as_str() {
                "BF16" => bytes
                    .chunks_exact(2)
                    .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                    .collect(),
                "F16" => bytes
                    .chunks_exact(2)
                    .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                    .collect(),
                "F32" => bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                other => bail!("unsupported dtype {other} for {name}"),
            };
            Ok(Array::from_slice(&data, &m.shape))
        }

        /// Load a quantized linear's `{weight,scales,biases}` triple.
        fn qlinear(&self, prefix: &str) -> Result<QuantLinear> {
            Ok(QuantLinear {
                weight: self.u32_tensor(&format!("{prefix}.weight"))?,
                scales: self.f32_tensor(&format!("{prefix}.scales"))?,
                biases: self.f32_tensor(&format!("{prefix}.biases"))?,
            })
        }

        /// Load a dense BF16 weight `[out, in]` and transpose once to `[in, out]`
        /// so it can be applied as `matmul(x, w)` (bias-free `x @ Wᵀ`). Evaluated
        /// to bound the lazy graph at load (mirrors `dit::SafeTensors::linw`).
        fn dense_weight(&self, prefix: &str) -> Result<Array> {
            let w = self
                .f32_tensor(&format!("{prefix}.weight"))?
                .transpose_axes(&[1, 0])
                .with_context(|| format!("transpose dense weight {prefix}"))?;
            w.eval()
                .with_context(|| format!("eval dense weight {prefix}"))?;
            Ok(w)
        }

        /// A projection: mlx affine 4-bit (iff a `<prefix>.scales` sibling
        /// exists) else dense BF16. The auto-detect makes `MistralEncoder::load`
        /// work for BOTH the 4-bit and bf16 encoder dirs.
        fn linear(&self, prefix: &str) -> Result<Linear> {
            if self.has(&format!("{prefix}.scales")) {
                Ok(Linear::Quant(self.qlinear(prefix)?))
            } else {
                Ok(Linear::Dense(self.dense_weight(prefix)?))
            }
        }
    }

    // ── weight bundles ──────────────────────────────────────────────────────────

    struct LayerW {
        input_layernorm: Array, // [hidden]
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        post_attention_layernorm: Array, // [hidden]
        gate_proj: Linear,
        up_proj: Linear,
        down_proj: Linear,
    }

    /// All weights for the Mistral text encoder.
    pub struct MistralEncoder {
        /// Token embedding table `[vocab, hidden]` (f32, eval'd). For the 4-bit
        /// encoder this is the dequantized table; for bf16 it is the dense table
        /// loaded directly. Either way `embed` just gathers rows by token id.
        embed_table: Array,
        layers: Vec<LayerW>,
    }

    impl MistralEncoder {
        /// Load the encoder from a snapshot directory. Works for BOTH the
        /// mlx-lm 4-bit dir (`mlx-community/...-4bit`, sharded U32 + `.scales`)
        /// and the official bf16 dir
        /// (`black-forest-labs/FLUX.2-dev/text_encoder/`, sharded BF16) — the
        /// quantized/dense choice is per-tensor by `.scales` presence.
        pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
            let m = ShardedModel::load(dir)?;
            let pfx = "language_model.model";

            // Embedding table → [vocab, hidden] f32. 4-bit: dequantize the
            // packed `{weight,scales,biases}`. bf16: the dense `.weight` is
            // already the [vocab,hidden] table — load it as-is (no dequant).
            // `embed` gathers rows by token id from whichever form.
            let embed_table = if m.has(&format!("{pfx}.embed_tokens.scales")) {
                let embed_w = m.u32_tensor(&format!("{pfx}.embed_tokens.weight"))?;
                let embed_s = m.f32_tensor(&format!("{pfx}.embed_tokens.scales"))?;
                let embed_b = m.f32_tensor(&format!("{pfx}.embed_tokens.biases"))?;
                mlx_rs::ops::dequantize(&embed_w, &embed_s, &embed_b, GROUP_SIZE, BITS)
                    .context("dequantize embed_tokens")?
            } else {
                m.f32_tensor(&format!("{pfx}.embed_tokens.weight"))
                    .context("load dense bf16 embed_tokens")?
            };
            embed_table.eval().context("eval embed_table")?;

            // Load only layers 0..LOAD_LAYERS (deepest capture = layer 30); the
            // remaining layers / final norm / lm_head are never read. The
            // `linear()` loader auto-detects 4-bit vs dense bf16 per projection.
            let mut layers = Vec::with_capacity(LOAD_LAYERS);
            for li in 0..LOAD_LAYERS {
                let lp = format!("{pfx}.layers.{li}");
                layers.push(LayerW {
                    input_layernorm: m.f32_tensor(&format!("{lp}.input_layernorm.weight"))?,
                    q_proj: m.linear(&format!("{lp}.self_attn.q_proj"))?,
                    k_proj: m.linear(&format!("{lp}.self_attn.k_proj"))?,
                    v_proj: m.linear(&format!("{lp}.self_attn.v_proj"))?,
                    o_proj: m.linear(&format!("{lp}.self_attn.o_proj"))?,
                    post_attention_layernorm: m
                        .f32_tensor(&format!("{lp}.post_attention_layernorm.weight"))?,
                    gate_proj: m.linear(&format!("{lp}.mlp.gate_proj"))?,
                    up_proj: m.linear(&format!("{lp}.mlp.up_proj"))?,
                    down_proj: m.linear(&format!("{lp}.mlp.down_proj"))?,
                });
            }
            Ok(Self {
                embed_table,
                layers,
            })
        }

        /// Embedding lookup: gather rows of the dequantized table by token id.
        /// `ids`: `&[i32]` length `T` → `[1, T, hidden]`.
        fn embed(&self, ids: &[i32]) -> Result<Array> {
            let idx = Array::from_slice(ids, &[ids.len() as i32]);
            // take(table, idx, axis=0) -> [T, hidden]
            let rows = mlx_rs::ops::indexing::take_axis(&self.embed_table, &idx, 0)
                .context("embed: take rows")?;
            rows.reshape(&[1, ids.len() as i32, HIDDEN])
                .context("embed: reshape to [1,T,hidden]")
        }

        /// Build the additive causal + left-pad mask `[1, 1, T, T]`:
        /// 0.0 where `first_real <= j <= i`, else `-1e9`.
        fn build_mask(t: usize, first_real: usize) -> Array {
            let neg = -1e9f32;
            let mut data = vec![neg; t * t];
            for i in 0..t {
                for j in first_real..=i {
                    data[i * t + j] = 0.0;
                }
            }
            Array::from_slice(&data, &[1, 1, t as i32, t as i32])
        }

        /// Full encode. `ids`: length-`T` token ids; `first_real`: first
        /// real-token index for the left-pad mask. Returns the FLUX text
        /// embedding `[1, T, 3 * hidden]` (layers 10/20/30 stacked & flattened),
        /// and (for staged debugging) the hidden state after layer 10.
        pub fn encode_with_stage(&self, ids: &[i32], first_real: usize) -> Result<(Array, Array)> {
            let t = ids.len();
            let ti = t as i32;
            let mask = Self::build_mask(t, first_real);
            mask.eval().context("eval mask")?;
            let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();

            let mut h = self.embed(ids)?; // [1, T, hidden]
            // hidden_states[0] = embeddings; hidden_states[k] = after layer k.
            let mut captured: HashMap<usize, Array> = HashMap::new();
            let mut h10: Option<Array> = None;

            for (li, lw) in self.layers.iter().enumerate() {
                let layer_no = li + 1; // 1-based: output AFTER this layer

                // ── self-attention ──
                let r = h.clone();
                let x = mlx_rs::fast::rms_norm(&h, &lw.input_layernorm, RMS_EPS)
                    .context("attn rms_norm")?;

                let q = linear(&x, &lw.q_proj)?; // [1,T,4096]
                let k = linear(&x, &lw.k_proj)?; // [1,T,1024]
                let v = linear(&x, &lw.v_proj)?; // [1,T,1024]

                // [1,T,H,D] -> [1,H,T,D]
                let q = q
                    .reshape(&[1, ti, N_HEADS, HEAD_DIM])?
                    .transpose_axes(&[0, 2, 1, 3])?;
                let k = k
                    .reshape(&[1, ti, N_KV_HEADS, HEAD_DIM])?
                    .transpose_axes(&[0, 2, 1, 3])?;
                let v = v
                    .reshape(&[1, ti, N_KV_HEADS, HEAD_DIM])?
                    .transpose_axes(&[0, 2, 1, 3])?;

                // RoPE (non-traditional, base 1e9, offset 0, scale 1).
                let q = mlx_rs::fast::rope(&q, HEAD_DIM, false, Some(ROPE_THETA), 1.0, 0, None)
                    .context("rope q")?;
                let k = mlx_rs::fast::rope(&k, HEAD_DIM, false, Some(ROPE_THETA), 1.0, 0, None)
                    .context("rope k")?;

                // GQA: 32 q heads, 8 kv heads — mlx sdpa broadcasts kv.
                let attn =
                    mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, scale, &mask, None)
                        .context("sdpa")?;

                // [1,H,T,D] -> [1,T,H,D] -> [1,T,4096]
                let attn =
                    attn.transpose_axes(&[0, 2, 1, 3])?
                        .reshape(&[1, ti, N_HEADS * HEAD_DIM])?;
                let attn = linear(&attn, &lw.o_proj)?; // [1,T,hidden]
                h = r.add(&attn).context("attn residual add")?;

                // ── SwiGLU MLP ──
                let r2 = h.clone();
                let x = mlx_rs::fast::rms_norm(&h, &lw.post_attention_layernorm, RMS_EPS)
                    .context("mlp rms_norm")?;
                let g = silu(&linear(&x, &lw.gate_proj)?)?;
                let u = linear(&x, &lw.up_proj)?;
                let gu = g.multiply(&u).context("swiglu gate*up")?;
                let down = linear(&gu, &lw.down_proj)?;
                h = r2.add(&down).context("mlp residual add")?;

                // Eval each layer's hidden to bound the lazy graph / memory.
                h.eval().context("eval layer hidden")?;

                if layer_no == 10 {
                    h10 = Some(h.clone());
                }
                if CAPTURE_LAYERS.contains(&layer_no) {
                    captured.insert(layer_no, h.clone());
                }
            }

            // stack [10],[20],[30] on axis=1 -> [1,3,T,hidden]
            let stacked = mlx_rs::ops::stack_axis(
                &[
                    captured.remove(&10).context("missing layer 10")?,
                    captured.remove(&20).context("missing layer 20")?,
                    captured.remove(&30).context("missing layer 30")?,
                ],
                1,
            )
            .context("stack captured layers")?;
            // transpose (0,2,1,3) -> [1,T,3,hidden] -> reshape [1,T,3*hidden]
            let out = stacked
                .transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[1, ti, 3 * HIDDEN])?;
            out.eval().context("eval encoder output")?;

            Ok((out, h10.context("layer 10 not captured")?))
        }

        /// Convenience: encode and return only the `[1, T, 3*hidden]` embedding.
        pub fn encode(&self, ids: &[i32], first_real: usize) -> Result<Array> {
            Ok(self.encode_with_stage(ids, first_real)?.0)
        }
    }

    /// Default text-encoder dir: the 4-bit MLX Mistral-Small-3.2-24B snapshot
    /// resolved from the local HuggingFace cache. Falls back to the repo id (so
    /// load fails with a clear message) if it isn't downloaded.
    pub fn default_weights_dir() -> PathBuf {
        crate::hf_cache::snapshot_dir_or_id(crate::repos::ENCODER_4BIT)
    }
}

#[cfg(feature = "mlx-native")]
pub use imp::{MistralEncoder, default_weights_dir};

// ── parity test against the validated golden reference (/tmp/mistral_*.bin) ───
#[cfg(all(test, feature = "mlx-native"))]
mod parity_tests {
    use super::imp::MistralEncoder;

    fn read_i32_bin(path: &str) -> Vec<i32> {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        assert_eq!(bytes.len() % 4, 0, "{path} not i32-aligned");
        bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn read_f32_bin(path: &str) -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        assert_eq!(bytes.len() % 4, 0, "{path} not f32-aligned");
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// max-abs, mean-abs, cosine over the slice of `[T, D]` rows in
    /// `[row_lo, row_hi)` (real-token positions only).
    fn compare_rows(
        label: &str,
        obs: &[f32],
        exp: &[f32],
        t: usize,
        d: usize,
        row_lo: usize,
        row_hi: usize,
    ) -> (f32, f32, f64) {
        assert_eq!(obs.len(), t * d, "{label}: obs size");
        assert_eq!(exp.len(), t * d, "{label}: exp size");
        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f64;
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        let mut n = 0usize;
        for row in row_lo..row_hi {
            for c in 0..d {
                let g = obs[row * d + c];
                let e = exp[row * d + c];
                let diff = (g - e).abs();
                if diff > max_abs {
                    max_abs = diff;
                }
                sum_abs += diff as f64;
                dot += (g as f64) * (e as f64);
                na += (g as f64) * (g as f64);
                nb += (e as f64) * (e as f64);
                n += 1;
            }
        }
        let mean_abs = (sum_abs / n as f64) as f32;
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
        println!(
            "{label}: rows[{row_lo}..{row_hi}) max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} cosine={cos:.6}"
        );
        (max_abs, mean_abs, cos)
    }

    #[test]
    #[ignore = "needs a Metal device AND the /tmp/mistral_*.bin dump set (tokens, \
                firstreal, h10, embed_ref); no committed script produces them — \
                see crates/lumen-mlx/tests/golden/README.md"]
    fn mistral_encoder_matches_reference() {
        const T: usize = 512;
        let ids = read_i32_bin("/tmp/mistral_tokens.bin");
        assert_eq!(ids.len(), T, "token count");
        let first_real = read_i32_bin("/tmp/mistral_firstreal.bin")[0] as usize;
        assert_eq!(first_real, 465, "first_real");

        let enc =
            MistralEncoder::load(super::imp::default_weights_dir()).expect("load encoder weights");
        let (out, h10) = enc.encode_with_stage(&ids, first_real).expect("encode");
        assert_eq!(out.shape(), &[1, T as i32, 3 * 5120], "output shape");

        // ── staged: hidden after layer 10 vs /tmp/mistral_h10.bin ──
        h10.eval().expect("eval h10");
        let h10_obs: &[f32] = h10.as_slice();
        let h10_exp = read_f32_bin("/tmp/mistral_h10.bin");
        let (_, _, h10_cos) = compare_rows("h10", h10_obs, &h10_exp, T, 5120, first_real, T);
        assert!(
            h10_cos > 0.999,
            "staged layer-10 cosine {h10_cos:.6} <= 0.999 — divergence before/at layer 10"
        );

        // ── final: full embedding vs /tmp/mistral_embed_ref.bin ──
        out.eval().expect("eval out");
        let obs: &[f32] = out.as_slice();
        let exp = read_f32_bin("/tmp/mistral_embed_ref.bin");
        let (max_abs, mean_abs, cos) = compare_rows("embed", obs, &exp, T, 3 * 5120, first_real, T);

        // outlier-dim max-abs may be a few units (cross-impl bf16/quant noise);
        // do NOT gate on it. Gate on cosine + mean-abs.
        assert!(cos > 0.999, "embedding cosine {cos:.6} <= 0.999");
        assert!(
            mean_abs < 0.05,
            "embedding mean-abs {mean_abs:.6e} >= 0.05 (max_abs {max_abs:.6e})"
        );
        println!(
            "Phase 1 gate PASSED: cosine={cos:.6} mean_abs={mean_abs:.6e} max_abs={max_abs:.6e}"
        );
    }
}
