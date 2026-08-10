//! In-process Qwen3-Embedding model on mlx-rs.
//!
//! Replaces the candle implementation that used to live in `lumen-model`. The
//! surface `lumen-server` consumes is unchanged — `load` / `model_id` / `dim` /
//! `max_seq_len` / `embed` — and so are the two semantics that make the output
//! usable:
//!
//! - **Last-token pooling.** Qwen3-Embedding is a causal LM adapted for
//!   retrieval; the sentence vector is the final hidden state at the *last*
//!   real token, not a mean. With causal attention that position is the only
//!   one that has seen the whole input.
//! - **L2-normalized f32 output**, so a dot product *is* cosine similarity.
//!   `/v1/embeddings` consumers rely on this.
//!
//! ## Shape
//!
//! Plain Qwen3 (`model_type: "qwen3"`), which is a simpler layer than the
//! Qwen3.5/3.6 blocks elsewhere in this crate: no attention gate, no linear
//! attention, no MoE. Per layer:
//!
//! ```text
//!   h += o_proj(attn(q_norm(q), k_norm(k), v))     ← RoPE on q,k; GQA; causal
//!   h += down_proj(silu(gate_proj(h)) * up_proj(h))
//! ```
//!
//! `head_dim` is read from the config and is **not** `hidden_size /
//! num_attention_heads` — Qwen3-Embedding-0.6B is 1024/16 with `head_dim: 128`,
//! so deriving it would silently build a model with the wrong `q_proj` shape.
//!
//! ## Batching
//!
//! `embed` sorts inputs by token length, pads each chunk of up to
//! [`imp::DEFAULT_MAX_BATCH_ROWS`] to that chunk's longest row, and restores the
//! caller's order afterwards.
//!
//! The padding side is load-bearing and wrong-in-silence if reversed. Rows are
//! **right**-padded so that the pooled position, `len - 1`, attends only to
//! real tokens under the causal mask; left-padding would shift every position
//! and pull pad tokens inside that window. The diffusion tokenizer in this
//! workspace needs exactly the opposite — its encoder reads the *tail* of the
//! window — and getting that backwards is a recorded defect
//! (`flux-left-padding` in `xtask/src/red_green.rs`). Same class, opposite
//! answer, which is why the reasoning is written down on both sides.
//!
//! Sorting by length matters more than it looks: one long document in a batch
//! of short queries would otherwise pad every row out to its length and spend
//! nearly all the compute on positions that are discarded.

/// Result of embedding a batch of texts.
///
/// Deliberately outside the `mlx-native` gate: `lumen-server` names this type
/// in its worker-channel signatures regardless of which features the binary was
/// built with, the same reason [`crate::chat_io`] exists.
#[derive(Debug)]
pub struct EmbeddingBatch {
    /// One L2-normalized vector per input text, in input order.
    pub embeddings: Vec<Vec<f32>>,
    /// Total tokens consumed across the batch, for usage accounting.
    pub prompt_tokens: u32,
}

#[cfg(feature = "mlx-native")]
pub(crate) mod imp {
    use std::ffi::CStr;
    use std::path::{Path, PathBuf};

    use anyhow::{Context, Result, anyhow};
    use mlx_rs::Array;
    use mlx_rs::ops::indexing::TryIndexOp;
    use serde::Deserialize;
    use tokenizers::Tokenizer;

    use super::EmbeddingBatch;
    use crate::native_attention::sdpa;
    use crate::native_embedding::quantized_embedding_lookup_with_mode;
    use crate::native_norm::rms_norm;
    use crate::native_quant::quantized_matmul_with_mode;
    use crate::native_rope::rope;

    /// `quantized_matmul` mode string for MLX affine quantization.
    const MODE_AFFINE: &CStr = c"affine";

    /// Rows per padded forward pass.
    ///
    /// Bounds peak activation memory, which grows as `rows × max_len`: a
    /// caller is free to POST a thousand strings to `/v1/embeddings`, and
    /// padding all of them into one array is how that turns into an OOM
    /// instead of a slightly slower response. 32 keeps the GPU busy on the
    /// 0.6B model while capping the worst case.
    const DEFAULT_MAX_BATCH_ROWS: usize = 32;

    /// `LUMEN_EMBEDDING_BATCH_ROWS` overrides [`DEFAULT_MAX_BATCH_ROWS`].
    /// Setting it to 1 gives the unbatched, one-sequence-at-a-time path, which
    /// is how the throughput contribution of batching was measured. Output is
    /// unaffected — each row pools at its own last real token regardless of how
    /// many rows share the forward pass.
    fn max_batch_rows() -> usize {
        static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *ROWS.get_or_init(|| {
            std::env::var("LUMEN_EMBEDDING_BATCH_ROWS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&n| n > 0)
                .unwrap_or(DEFAULT_MAX_BATCH_ROWS)
        })
    }

    #[derive(Debug, Deserialize)]
    struct QuantConfig {
        group_size: i32,
        bits: i32,
    }

    #[derive(Debug, Deserialize)]
    struct RawConfig {
        hidden_size: usize,
        num_hidden_layers: usize,
        num_attention_heads: i32,
        num_key_value_heads: i32,
        /// Explicit in every Qwen3 checkpoint, and not derivable — see the
        /// module docs.
        head_dim: i32,
        max_position_embeddings: usize,
        rms_norm_eps: f32,
        rope_theta: f32,
        quantization: Option<QuantConfig>,
        #[serde(default)]
        model_type: String,
    }

    /// One quantized projection: MLX stores packed weights beside per-group
    /// scales and biases.
    struct QuantLinear {
        weight: Array,
        scales: Array,
        biases: Option<Array>,
        group_size: i32,
        bits: i32,
    }

    impl QuantLinear {
        fn forward(&self, x: &Array) -> Result<Array> {
            quantized_matmul_with_mode(
                x,
                &self.weight,
                &self.scales,
                self.biases.as_ref(),
                /* transpose */ true,
                self.group_size,
                self.bits,
                MODE_AFFINE,
            )
        }
    }

    struct Layer {
        input_layernorm: Array,
        q_proj: QuantLinear,
        k_proj: QuantLinear,
        v_proj: QuantLinear,
        o_proj: QuantLinear,
        q_norm: Array,
        k_norm: Array,
        post_attention_layernorm: Array,
        gate_proj: QuantLinear,
        up_proj: QuantLinear,
        down_proj: QuantLinear,
    }

    pub struct EmbeddingModel {
        tokenizer: Tokenizer,
        embed_weight: Array,
        embed_scales: Array,
        embed_biases: Option<Array>,
        embed_group_size: i32,
        embed_bits: i32,
        layers: Vec<Layer>,
        norm: Array,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        eps: f32,
        rope_theta: f32,
        dim: usize,
        max_seq_len: usize,
        model_id: String,
    }

    /// Weight bag for a single-file or sharded checkpoint.
    struct Weights(std::collections::HashMap<String, Array>);

    impl Weights {
        fn load(dir: &Path) -> Result<Self> {
            let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)
                .with_context(|| format!("read model dir {}", dir.display()))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.is_file() && p.extension().is_some_and(|x| x == "safetensors"))
                .collect();
            if shards.is_empty() {
                return Err(anyhow!("no *.safetensors under {}", dir.display()));
            }
            // Deterministic order so a duplicate key across shards always
            // resolves the same way rather than by directory-iteration luck.
            shards.sort();
            let mut all = std::collections::HashMap::new();
            for shard in &shards {
                let map = Array::load_safetensors(shard)
                    .map_err(|e| anyhow!("load {}: {e}", shard.display()))?;
                all.extend(map);
            }
            Ok(Self(all))
        }

        fn take(&mut self, key: &str) -> Result<Array> {
            self.0
                .remove(key)
                .ok_or_else(|| anyhow!("checkpoint is missing tensor {key:?}"))
        }

        fn take_opt(&mut self, key: &str) -> Option<Array> {
            self.0.remove(key)
        }

        fn take_linear(&mut self, base: &str, group_size: i32, bits: i32) -> Result<QuantLinear> {
            Ok(QuantLinear {
                weight: self.take(&format!("{base}.weight"))?,
                scales: self.take(&format!("{base}.scales"))?,
                biases: self.take_opt(&format!("{base}.biases")),
                group_size,
                bits,
            })
        }
    }

    impl EmbeddingModel {
        /// Load from a local checkpoint directory, or a HuggingFace repo id
        /// resolved through the shared model-directory resolver.
        pub fn load(model_id_or_path: &str) -> Result<Self> {
            let dir = crate::runner_native::resolve_model_dir(model_id_or_path)
                .with_context(|| format!("resolve embedding model {model_id_or_path:?}"))?;

            let cfg_text = std::fs::read_to_string(dir.join("config.json"))
                .with_context(|| format!("read {}/config.json", dir.display()))?;
            let cfg: RawConfig =
                serde_json::from_str(&cfg_text).context("parse embedding config.json")?;

            if !cfg.model_type.is_empty() && cfg.model_type != "qwen3" {
                return Err(anyhow!(
                    "embedding model_type {:?} is not supported — this loader implements plain \
                     Qwen3 (`model_type: \"qwen3\"`). A different architecture needs its own \
                     layer body, not a config tweak.",
                    cfg.model_type
                ));
            }
            let quant = cfg.quantization.as_ref().ok_or_else(|| {
                anyhow!(
                    "embedding checkpoint has no `quantization` block. Only MLX-quantized \
                     (affine) checkpoints are supported; convert with `mlx_lm.convert -q`."
                )
            })?;
            let (group_size, bits) = (quant.group_size, quant.bits);

            let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
                .map_err(|e| anyhow!("load tokenizer.json: {e}"))?;

            let mut w = Weights::load(&dir)?;

            let embed_weight = w.take("model.embed_tokens.weight")?;
            let embed_scales = w.take("model.embed_tokens.scales")?;
            let embed_biases = w.take_opt("model.embed_tokens.biases");

            let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
            for i in 0..cfg.num_hidden_layers {
                let p = format!("model.layers.{i}");
                layers.push(Layer {
                    input_layernorm: w.take(&format!("{p}.input_layernorm.weight"))?,
                    q_proj: w.take_linear(&format!("{p}.self_attn.q_proj"), group_size, bits)?,
                    k_proj: w.take_linear(&format!("{p}.self_attn.k_proj"), group_size, bits)?,
                    v_proj: w.take_linear(&format!("{p}.self_attn.v_proj"), group_size, bits)?,
                    o_proj: w.take_linear(&format!("{p}.self_attn.o_proj"), group_size, bits)?,
                    q_norm: w.take(&format!("{p}.self_attn.q_norm.weight"))?,
                    k_norm: w.take(&format!("{p}.self_attn.k_norm.weight"))?,
                    post_attention_layernorm: w
                        .take(&format!("{p}.post_attention_layernorm.weight"))?,
                    gate_proj: w.take_linear(&format!("{p}.mlp.gate_proj"), group_size, bits)?,
                    up_proj: w.take_linear(&format!("{p}.mlp.up_proj"), group_size, bits)?,
                    down_proj: w.take_linear(&format!("{p}.mlp.down_proj"), group_size, bits)?,
                });
            }
            let norm = w.take("model.norm.weight")?;

            eprintln!(
                "[embedding] loaded {} dim={} layers={} heads={}/{} head_dim={} ({bits}-bit affine)",
                dir.display(),
                cfg.hidden_size,
                cfg.num_hidden_layers,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
            );

            Ok(Self {
                tokenizer,
                embed_weight,
                embed_scales,
                embed_biases,
                embed_group_size: group_size,
                embed_bits: bits,
                layers,
                norm,
                num_heads: cfg.num_attention_heads,
                num_kv_heads: cfg.num_key_value_heads,
                head_dim: cfg.head_dim,
                eps: cfg.rms_norm_eps,
                rope_theta: cfg.rope_theta,
                dim: cfg.hidden_size,
                max_seq_len: cfg.max_position_embeddings,
                model_id: model_id_or_path.to_string(),
            })
        }

        pub fn model_id(&self) -> &str {
            &self.model_id
        }

        pub fn dim(&self) -> usize {
            self.dim
        }

        pub fn max_seq_len(&self) -> usize {
            self.max_seq_len
        }

        /// Encode `texts`, returning L2-normalized vectors in input order.
        pub fn embed(&mut self, texts: &[String]) -> Result<EmbeddingBatch> {
            if texts.is_empty() {
                return Ok(EmbeddingBatch {
                    embeddings: Vec::new(),
                    prompt_tokens: 0,
                });
            }

            let mut rows: Vec<Vec<i32>> = Vec::with_capacity(texts.len());
            let mut prompt_tokens = 0u32;
            for text in texts {
                let enc = self
                    .tokenizer
                    .encode(text.as_str(), /* add_special_tokens */ true)
                    .map_err(|e| anyhow!("tokenize {text:?}: {e}"))?;
                let mut ids: Vec<i32> = enc.get_ids().iter().map(|&t| t as i32).collect();
                if ids.is_empty() {
                    return Err(anyhow!("input {text:?} tokenized to nothing"));
                }
                // Truncate rather than fail: a caller pasting a long document
                // wants an embedding of what fits, and the alternative is a
                // 400 on input the model can partially handle. Positions past
                // the RoPE table have no defined meaning.
                ids.truncate(self.max_seq_len);
                prompt_tokens = prompt_tokens.saturating_add(ids.len() as u32);
                rows.push(ids);
            }

            // Bucket by length before padding. One 4k-token document in a batch
            // of short queries would otherwise pad every other row out to 4k and
            // spend ~all the compute on positions that are thrown away. Sorting
            // by length keeps each padded batch close to uniform; `order` carries
            // the mapping so results come back in the caller's order.
            let mut order: Vec<usize> = (0..rows.len()).collect();
            order.sort_by_key(|&i| rows[i].len());

            let mut embeddings: Vec<Option<Vec<f32>>> = vec![None; rows.len()];
            for chunk in order.chunks(max_batch_rows()) {
                let group: Vec<&[i32]> = chunk.iter().map(|&i| rows[i].as_slice()).collect();
                for (&idx, vec) in chunk.iter().zip(self.forward_batch(&group)?) {
                    embeddings[idx] = Some(vec);
                }
            }

            Ok(EmbeddingBatch {
                embeddings: embeddings
                    .into_iter()
                    .map(|v| v.expect("every row was assigned exactly once"))
                    .collect(),
                prompt_tokens,
            })
        }

        /// Forward one padded batch; returns L2-normalized pooled vectors in
        /// the order given.
        ///
        /// Rows are **right**-padded. With causal attention, position
        /// `len - 1` — the pooled one — attends only to `0..len-1`, which are
        /// all real tokens, so padding cannot reach the output. That is what
        /// makes the padding value irrelevant, and it is also why the padding
        /// must go on the right: left-padding would shift every real token's
        /// position and put pad tokens *inside* the pooled row's causal window.
        fn forward_batch(&self, rows: &[&[i32]]) -> Result<Vec<Vec<f32>>> {
            let b = rows.len() as i32;
            let max_len = rows.iter().map(|r| r.len()).max().unwrap_or(0);
            if max_len == 0 {
                return Err(anyhow!("forward_batch: every row is empty"));
            }
            let l = max_len as i32;

            // Pad with the first real token of each row rather than a constant:
            // it is guaranteed in-vocabulary for this checkpoint, and the value
            // is provably unobservable (see above), so this only has to be safe.
            let mut flat = Vec::with_capacity(rows.len() * max_len);
            for r in rows {
                flat.extend_from_slice(r);
                flat.resize(flat.len() + (max_len - r.len()), r[0]);
            }
            let token_ids = Array::from_slice(&flat, &[b, l]);

            let mut h = quantized_embedding_lookup_with_mode(
                &self.embed_weight,
                &self.embed_scales,
                self.embed_biases.as_ref(),
                &token_ids,
                self.embed_group_size,
                self.embed_bits,
                MODE_AFFINE,
            )
            .context("embedding lookup")?
            .as_dtype(mlx_rs::Dtype::Float32)
            .context("cast embeddings to f32")?;

            for (i, layer) in self.layers.iter().enumerate() {
                h = self
                    .layer_forward(layer, &h, b, l)
                    .with_context(|| format!("layer {i}"))?;
            }

            let h = rms_norm(&h, &self.norm, self.eps).context("final norm")?;

            let mut out = Vec::with_capacity(rows.len());
            for (i, r) in rows.iter().enumerate() {
                // Each row pools at its own last *real* token, not at `l - 1`.
                let pooled = h
                    .try_index((i as i32, r.len() as i32 - 1, ..))
                    .context("pool last token")?
                    .as_dtype(mlx_rs::Dtype::Float32)
                    .context("cast pooled vector to f32")?;
                let v: Vec<f32> = pooled.as_slice::<f32>().to_vec();
                let l2: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if !l2.is_finite() || l2 <= 0.0 {
                    return Err(anyhow!(
                        "embedding row {i} has L2 norm {l2} — cannot normalize; the forward pass \
                         produced a degenerate vector"
                    ));
                }
                out.push(v.into_iter().map(|x| x / l2).collect());
            }
            Ok(out)
        }

        fn layer_forward(&self, layer: &Layer, x: &Array, b: i32, l: i32) -> Result<Array> {
            let (nh, nkv, hd) = (self.num_heads, self.num_kv_heads, self.head_dim);

            // ── self-attention ────────────────────────────────────────────
            let normed = rms_norm(x, &layer.input_layernorm, self.eps)?;

            let q = layer.q_proj.forward(&normed)?;
            let k = layer.k_proj.forward(&normed)?;
            let v = layer.v_proj.forward(&normed)?;

            // Per-head RMSNorm happens on [B, L, heads, head_dim] so the
            // `[head_dim]` weight broadcasts along the last axis, then the
            // transpose puts heads in front for attention.
            let q = mlx_rs::ops::reshape(&q, &[b, l, nh, hd]).context("reshape q")?;
            let q = rms_norm(&q, &layer.q_norm, self.eps)?;
            let q = mlx_rs::ops::transpose_axes(&q, &[0, 2, 1, 3]).context("transpose q")?;

            let k = mlx_rs::ops::reshape(&k, &[b, l, nkv, hd]).context("reshape k")?;
            let k = rms_norm(&k, &layer.k_norm, self.eps)?;
            let k = mlx_rs::ops::transpose_axes(&k, &[0, 2, 1, 3]).context("transpose k")?;

            let v = mlx_rs::ops::reshape(&v, &[b, l, nkv, hd]).context("reshape v")?;
            let v = mlx_rs::ops::transpose_axes(&v, &[0, 2, 1, 3]).context("transpose v")?;

            // Offset 0: every call encodes a whole sequence from scratch, so
            // there is no KV cache and no carried position.
            let q = rope(
                &q,
                hd,
                /* traditional */ false,
                self.rope_theta,
                1.0,
                0,
            )?;
            let k = rope(&k, hd, false, self.rope_theta, 1.0, 0)?;

            let scale = 1.0f32 / (hd as f32).sqrt();
            // `causal` matters even though we pool the last position: every
            // earlier position feeding it must not have seen the future.
            let attn = sdpa(&q, &k, &v, scale, /* causal */ true)?;

            let attn =
                mlx_rs::ops::transpose_axes(&attn, &[0, 2, 1, 3]).context("transpose attn out")?;
            let attn = mlx_rs::ops::reshape(&attn, &[b, l, nh * hd]).context("reshape attn out")?;
            let attn_out = layer.o_proj.forward(&attn)?;
            let h = mlx_rs::ops::add(x, &attn_out).context("attn residual")?;

            // ── MLP ───────────────────────────────────────────────────────
            let normed = rms_norm(&h, &layer.post_attention_layernorm, self.eps)?;
            let gate = layer.gate_proj.forward(&normed)?;
            let up = layer.up_proj.forward(&normed)?;
            let act =
                mlx_rs::ops::multiply(&mlx_rs::nn::silu(&gate).context("silu(gate_proj)")?, &up)
                    .context("gate * up")?;
            let mlp_out = layer.down_proj.forward(&act)?;
            mlx_rs::ops::add(&h, &mlp_out).context("mlp residual")
        }
    }
}

#[cfg(feature = "mlx-native")]
pub use imp::EmbeddingModel;
