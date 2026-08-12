//! Native FLUX.2 DiT (diffusion transformer), Phase 2 of the diffusion port.
//!
//! A faithful mechanical port of the mflux `Flux2Transformer` for
//! FLUX.2-klein-4B. The math mirrors `flux2_transformer/*.py` exactly; the
//! config is parameterized ([`DitConfig`]) so the larger dev-32B checkpoint
//! (same architecture, larger dims/layer-counts) is a config + weight swap.
//!
//! Forward (no KV cache, no guidance — the klein config has `guidance_embeds`
//! false):
//! ```text
//! temb = time_embed(timestep*1000)                      # [B, inner]
//! hidden = x_embedder(latent)      [B, S_img, inner]
//! enc    = context_embedder(text)  [B, S_txt, inner]
//! (cos,sin) = pos_embed([txt_ids ; img_ids])            # joint [txt;img]
//! mod_img/mod_txt/mod_single = shared modulation(temb)  # SAME for all layers
//! for b in double_blocks:  enc,hidden = double(b)
//! hidden = concat([enc, hidden], seq)                   # [txt;img]
//! for b in single_blocks:  hidden = single(b)
//! hidden = hidden[:, S_txt:, :]                         # image part
//! hidden = norm_out(hidden, temb)                       # AdaLN-continuous
//! out = proj_out(hidden)           [B, S_img, out_ch]
//! ```
//!
//! All Linear weights are bias-free (`x @ Wᵀ`); LayerNorms are affine-false; q/k
//! RMSNorm is per-head over `head_dim`. BF16 weights are upcast to f32 on load.
//!
//! Design docs: `.ai/memory/active/flux2-dev-diffusion-port/`.

#[cfg(feature = "mlx-native")]
mod imp {
    use std::collections::HashMap;
    use std::path::Path;

    use anyhow::{Context, Result, anyhow, bail};
    use mlx_rs::Array;
    use mlx_rs::ops::{concatenate_axis, split, split_sections};

    use crate::ops::layernorm::layer_norm_no_affine;
    use crate::ops::linear_nobias::linear_nobias;
    use crate::ops::qlinear::{QuantLinear, qlinear};
    use crate::ops::rope4d::{apply_rope_bshd, concat_seq, pos_embed};
    use crate::ops::silu::silu;
    use crate::ops::timestep_embed::timestep_embedding;

    // ── linear (dense BF16 or mlx affine-quantized) ───────────────────────────

    /// A DiT linear weight. klein checkpoints are dense BF16 (`x @ Wᵀ`, stored
    /// pre-transposed `[in,out]`); the dev-32B checkpoint is mlx affine 4-bit
    /// (`{weight,scales,biases}` packed). A single [`Linear::apply`] hides the
    /// difference so the forward pass is identical for both.
    pub enum Linear {
        /// Dense `[in, out]` (already transposed at load) — `matmul(x, w)`.
        Dense(Array),
        /// mlx affine 4-bit packed `{weight,scales,biases}`.
        Quant(QuantLinear),
    }

    impl Linear {
        #[inline]
        fn apply(&self, x: &Array) -> Result<Array> {
            match self {
                Linear::Dense(w) => linear_nobias(x, w),
                Linear::Quant(q) => qlinear(x, q),
            }
        }
    }

    // ── config ────────────────────────────────────────────────────────────────

    /// DiT geometry. Defaults are klein-4B; swap fields for dev-32B.
    #[derive(Clone, Debug)]
    pub struct DitConfig {
        pub num_layers: usize,        // double-stream blocks
        pub num_single_layers: usize, // single-stream blocks
        pub num_heads: usize,
        pub head_dim: usize,
        pub joint_dim: usize, // text-embed dim (context_embedder in)
        pub in_channels: usize,
        pub out_channels: usize,
        pub patch_size: usize,
        pub axes_dim: Vec<usize>, // 4D rope per-axis dims
        pub rope_theta: f32,
        pub mlp_ratio: f32,
        pub ts_channels: usize,    // timestep sinusoid dim
        pub guidance_embeds: bool, // dev uses a guidance MLP branch; klein does not
        pub eps_ln: f32,
        pub eps_rms: f32,
    }

    impl DitConfig {
        pub fn klein_4b() -> Self {
            Self {
                num_layers: 5,
                num_single_layers: 20,
                num_heads: 24,
                head_dim: 128,
                joint_dim: 7680,
                in_channels: 128,
                out_channels: 128,
                patch_size: 1,
                axes_dim: vec![32, 32, 32, 32],
                rope_theta: 2000.0,
                mlp_ratio: 3.0,
                ts_channels: 256,
                guidance_embeds: false,
                eps_ln: 1e-6,
                eps_rms: 1e-5,
            }
        }

        /// FLUX.2-dev-32B geometry (mlx affine 4-bit, guidance-embedded).
        pub fn dev_32b() -> Self {
            Self {
                num_layers: 8,
                num_single_layers: 48,
                num_heads: 48,
                head_dim: 128,
                joint_dim: 15360,
                in_channels: 128,
                out_channels: 128,
                patch_size: 1,
                axes_dim: vec![32, 32, 32, 32],
                rope_theta: 2000.0,
                mlp_ratio: 3.0,
                ts_channels: 256,
                guidance_embeds: true,
                eps_ln: 1e-6,
                eps_rms: 1e-5,
            }
        }

        #[inline]
        pub fn inner_dim(&self) -> usize {
            self.num_heads * self.head_dim
        }
        #[inline]
        pub fn mlp_hidden(&self) -> usize {
            (self.inner_dim() as f32 * self.mlp_ratio) as usize
        }
    }

    // ── safetensors loader (self-contained, mirrors vae.rs) ───────────────────

    struct TensorMeta {
        dtype: String,
        shape: Vec<i32>,
        start: usize,
        end: usize,
    }

    struct SafeTensors {
        metas: HashMap<String, TensorMeta>,
        data_start: usize,
        blob: Vec<u8>,
    }

    impl SafeTensors {
        fn load(path: impl AsRef<Path>) -> Result<Self> {
            let blob = std::fs::read(path.as_ref())
                .with_context(|| format!("reading safetensors {:?}", path.as_ref()))?;
            if blob.len() < 8 {
                bail!("safetensors file too short");
            }
            let hlen = u64::from_le_bytes(blob[0..8].try_into().unwrap()) as usize;
            let header_end = 8 + hlen;
            let header: serde_json::Value = serde_json::from_slice(&blob[8..header_end])
                .context("parsing safetensors JSON header")?;
            let obj = header
                .as_object()
                .ok_or_else(|| anyhow!("safetensors header not an object"))?;
            let mut metas = HashMap::new();
            for (name, v) in obj {
                if name == "__metadata__" {
                    continue;
                }
                let dtype = v["dtype"].as_str().unwrap().to_string();
                let shape: Vec<i32> = v["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|d| d.as_i64().unwrap() as i32)
                    .collect();
                let offs = v["data_offsets"].as_array().unwrap();
                metas.insert(
                    name.clone(),
                    TensorMeta {
                        dtype,
                        shape,
                        start: offs[0].as_u64().unwrap() as usize,
                        end: offs[1].as_u64().unwrap() as usize,
                    },
                );
            }
            Ok(Self {
                metas,
                data_start: header_end,
                blob,
            })
        }

        /// Load a tensor as f32 in its native (PyTorch) shape.
        fn raw(&self, name: &str) -> Result<Array> {
            let m = self
                .metas
                .get(name)
                .ok_or_else(|| anyhow!("tensor not found: {name}"))?;
            let bytes = &self.blob[self.data_start + m.start..self.data_start + m.end];
            let data: Vec<f32> = match m.dtype.as_str() {
                "BF16" => bytes
                    .chunks_exact(2)
                    .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                    .collect(),
                "F32" => bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                "F16" => bytes
                    .chunks_exact(2)
                    .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                    .collect(),
                other => bail!("unsupported dtype {other} for {name}"),
            };
            Ok(Array::from_slice(&data, &m.shape))
        }

        /// Linear weight: diffusers `[out,in]` → mlx `[in,out]` for `matmul`.
        fn linw(&self, name: &str) -> Result<Array> {
            let w = self
                .raw(name)?
                .transpose_axes(&[1, 0])
                .with_context(|| format!("linear weight transpose {name}"))?;
            w.eval().with_context(|| format!("eval linw {name}"))?;
            Ok(w)
        }

        /// 1-D vector (RMSNorm weight) as-is.
        fn vec(&self, name: &str) -> Result<Array> {
            let v = self.raw(name)?;
            v.eval().with_context(|| format!("eval vec {name}"))?;
            Ok(v)
        }

        fn has(&self, name: &str) -> bool {
            self.metas.contains_key(name)
        }

        /// Raw U32 packed tensor (no transpose), evaluated.
        fn u32_raw(&self, name: &str) -> Result<Array> {
            let m = self
                .metas
                .get(name)
                .ok_or_else(|| anyhow!("tensor not found: {name}"))?;
            if m.dtype != "U32" {
                bail!("expected U32 for {name}, got {}", m.dtype);
            }
            let bytes = &self.blob[self.data_start + m.start..self.data_start + m.end];
            let data: Vec<u32> = bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let a = Array::from_slice(&data, &m.shape);
            a.eval().with_context(|| format!("eval u32 {name}"))?;
            Ok(a)
        }
    }

    // ── unified weight source: single-file (klein) or sharded (dev) ───────────

    /// Loads DiT tensors from either a single safetensors file (klein BF16) or
    /// a directory of shards keyed by `model.safetensors.index.json` (dev 4-bit).
    pub struct DitWeights {
        shards: Vec<SafeTensors>,
        /// tensor name → shard index (empty for the single-file case).
        key_shard: HashMap<String, usize>,
    }

    impl DitWeights {
        /// Open a single `.safetensors` file (klein) or a directory of shards
        /// (dev — must contain `model.safetensors.index.json`).
        pub fn open(path: impl AsRef<Path>) -> Result<Self> {
            let path = path.as_ref();
            if path.is_dir() {
                let index_path = path.join("model.safetensors.index.json");
                let index: serde_json::Value = serde_json::from_slice(
                    &std::fs::read(&index_path)
                        .with_context(|| format!("reading index {index_path:?}"))?,
                )
                .context("parsing model.safetensors.index.json")?;
                let weight_map = index
                    .get("weight_map")
                    .and_then(|m| m.as_object())
                    .ok_or_else(|| anyhow!("index.json missing weight_map"))?;
                let mut shard_files: Vec<String> = weight_map
                    .values()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect();
                shard_files.sort();
                shard_files.dedup();
                let mut file_index: HashMap<String, usize> = HashMap::new();
                let mut shards = Vec::with_capacity(shard_files.len());
                for (i, f) in shard_files.iter().enumerate() {
                    shards.push(SafeTensors::load(path.join(f))?);
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
            } else {
                Ok(Self {
                    shards: vec![SafeTensors::load(path)?],
                    key_shard: HashMap::new(),
                })
            }
        }

        #[inline]
        fn shard_for(&self, name: &str) -> &SafeTensors {
            match self.key_shard.get(name) {
                Some(&si) => &self.shards[si],
                None => &self.shards[0],
            }
        }

        fn has(&self, name: &str) -> bool {
            if self.key_shard.is_empty() {
                self.shards[0].has(name)
            } else {
                self.key_shard.contains_key(name)
            }
        }

        /// 1-D vector (RMSNorm weight) as f32.
        fn vec(&self, name: &str) -> Result<Array> {
            self.shard_for(name).vec(name)
        }

        /// A linear weight: quantized (mlx affine 4-bit) iff a `.scales` sibling
        /// exists, else dense BF16 (`[out,in]` → transposed to `[in,out]`).
        fn linear(&self, name: &str) -> Result<Linear> {
            let wname = format!("{name}.weight");
            let sname = format!("{name}.scales");
            let s = self.shard_for(&sname);
            if self.has(&sname) {
                let q = QuantLinear {
                    weight: s.u32_raw(&wname)?,
                    scales: s.vec(&sname)?,
                    biases: s.vec(&format!("{name}.biases"))?,
                };
                Ok(Linear::Quant(q))
            } else {
                Ok(Linear::Dense(self.shard_for(&wname).linw(&wname)?))
            }
        }
    }

    // ── weight bundles ────────────────────────────────────────────────────────

    struct DoubleAttnW {
        to_q: Linear,
        to_k: Linear,
        to_v: Linear,
        to_out: Linear,
        norm_q: Array,
        norm_k: Array,
        add_q: Linear,
        add_k: Linear,
        add_v: Linear,
        to_add_out: Linear,
        norm_added_q: Array,
        norm_added_k: Array,
    }

    struct FeedForwardW {
        linear_in: Linear,
        linear_out: Linear,
    }

    struct DoubleBlockW {
        attn: DoubleAttnW,
        ff: FeedForwardW,
        ff_context: FeedForwardW,
    }

    struct SingleBlockW {
        to_qkv_mlp: Linear,
        to_out: Linear,
        norm_q: Array,
        norm_k: Array,
    }

    /// All DiT weights for one checkpoint.
    pub struct Dit {
        cfg: DitConfig,
        x_embedder: Linear,
        context_embedder: Linear,
        ts_linear_1: Linear,
        ts_linear_2: Linear,
        /// dev guidance MLP (None for klein, which has no guidance branch).
        guidance_linear_1: Option<Linear>,
        guidance_linear_2: Option<Linear>,
        mod_img: Linear,
        mod_txt: Linear,
        mod_single: Linear,
        double_blocks: Vec<DoubleBlockW>,
        single_blocks: Vec<SingleBlockW>,
        norm_out: Linear,
        proj_out: Linear,
    }

    impl Dit {
        /// Load all DiT weights from either a single diffusers BF16 safetensors
        /// file (klein) or a directory of mlx affine 4-bit shards (dev). The
        /// quantized/dense choice is per-linear (`.scales` presence).
        pub fn load(path: impl AsRef<Path>, cfg: DitConfig) -> Result<Self> {
            let st = DitWeights::open(path)?;

            // Key-name mapping differs between the klein (diffusers) and dev
            // (mflux) checkpoints: klein nests the timestep MLP under
            // `timestep_embedder` and names the attn output `to_out.0`; dev
            // flattens both. Detect which the checkpoint uses.
            let ts_prefix = if st.has("time_guidance_embed.timestep_embedder.linear_1.weight")
                || st.has("time_guidance_embed.timestep_embedder.linear_1.scales")
            {
                "time_guidance_embed.timestep_embedder"
            } else {
                "time_guidance_embed"
            };

            let mut double_blocks = Vec::with_capacity(cfg.num_layers);
            for i in 0..cfg.num_layers {
                let p = format!("transformer_blocks.{i}");
                let ap = format!("{p}.attn");
                // klein: `to_out.0`; dev: `to_out`.
                let to_out_name = if st.has(&format!("{ap}.to_out.0.weight")) {
                    format!("{ap}.to_out.0")
                } else {
                    format!("{ap}.to_out")
                };
                double_blocks.push(DoubleBlockW {
                    attn: DoubleAttnW {
                        to_q: st.linear(&format!("{ap}.to_q"))?,
                        to_k: st.linear(&format!("{ap}.to_k"))?,
                        to_v: st.linear(&format!("{ap}.to_v"))?,
                        to_out: st.linear(&to_out_name)?,
                        norm_q: st.vec(&format!("{ap}.norm_q.weight"))?,
                        norm_k: st.vec(&format!("{ap}.norm_k.weight"))?,
                        add_q: st.linear(&format!("{ap}.add_q_proj"))?,
                        add_k: st.linear(&format!("{ap}.add_k_proj"))?,
                        add_v: st.linear(&format!("{ap}.add_v_proj"))?,
                        to_add_out: st.linear(&format!("{ap}.to_add_out"))?,
                        norm_added_q: st.vec(&format!("{ap}.norm_added_q.weight"))?,
                        norm_added_k: st.vec(&format!("{ap}.norm_added_k.weight"))?,
                    },
                    ff: FeedForwardW {
                        linear_in: st.linear(&format!("{p}.ff.linear_in"))?,
                        linear_out: st.linear(&format!("{p}.ff.linear_out"))?,
                    },
                    ff_context: FeedForwardW {
                        linear_in: st.linear(&format!("{p}.ff_context.linear_in"))?,
                        linear_out: st.linear(&format!("{p}.ff_context.linear_out"))?,
                    },
                });
            }

            let mut single_blocks = Vec::with_capacity(cfg.num_single_layers);
            for i in 0..cfg.num_single_layers {
                let ap = format!("single_transformer_blocks.{i}.attn");
                single_blocks.push(SingleBlockW {
                    to_qkv_mlp: st.linear(&format!("{ap}.to_qkv_mlp_proj"))?,
                    to_out: st.linear(&format!("{ap}.to_out"))?,
                    norm_q: st.vec(&format!("{ap}.norm_q.weight"))?,
                    norm_k: st.vec(&format!("{ap}.norm_k.weight"))?,
                });
            }

            // Guidance MLP (dev only). Mirrors the timestep MLP shapes.
            let (guidance_linear_1, guidance_linear_2) =
                if cfg.guidance_embeds && st.has("time_guidance_embed.guidance_linear_1.weight") {
                    (
                        Some(st.linear("time_guidance_embed.guidance_linear_1")?),
                        Some(st.linear("time_guidance_embed.guidance_linear_2")?),
                    )
                } else {
                    (None, None)
                };

            Ok(Self {
                x_embedder: st.linear("x_embedder")?,
                context_embedder: st.linear("context_embedder")?,
                ts_linear_1: st.linear(&format!("{ts_prefix}.linear_1"))?,
                ts_linear_2: st.linear(&format!("{ts_prefix}.linear_2"))?,
                guidance_linear_1,
                guidance_linear_2,
                mod_img: st.linear("double_stream_modulation_img.linear")?,
                mod_txt: st.linear("double_stream_modulation_txt.linear")?,
                mod_single: st.linear("single_stream_modulation.linear")?,
                double_blocks,
                single_blocks,
                norm_out: st.linear("norm_out.linear")?,
                proj_out: st.linear("proj_out")?,
                cfg,
            })
        }

        // ── small math helpers ────────────────────────────────────────────────

        /// Per-head RMSNorm over the last (head_dim) axis, applied in f32.
        fn rmsnorm(&self, x: &Array, w: &Array) -> Result<Array> {
            mlx_rs::fast::rms_norm(x, w, self.cfg.eps_rms).context("rms_norm failed")
        }

        /// Reshape `[B,S,inner]` -> `[B,H,S,D]`.
        fn heads(&self, x: &Array) -> Result<Array> {
            let b = x.dim(0);
            let s = x.dim(1);
            let (h, d) = (self.cfg.num_heads as i32, self.cfg.head_dim as i32);
            x.reshape(&[b, s, h, d])
                .and_then(|r| r.transpose_axes(&[0, 2, 1, 3]))
                .context("heads reshape failed")
        }

        /// SDPA over `[B,H,S,D]` -> `[B,S,inner]`.
        fn sdpa(&self, q: &Array, k: &Array, v: &Array) -> Result<Array> {
            let scale = 1.0f32 / (self.cfg.head_dim as f32).sqrt();
            let out = mlx_rs::fast::scaled_dot_product_attention(q, k, v, scale, None, None)
                .context("sdpa failed")?;
            let b = out.dim(0);
            let inner = self.cfg.inner_dim() as i32;
            out.transpose_axes(&[0, 2, 1, 3])
                .and_then(|t| t.reshape(&[b, -1, inner]))
                .context("sdpa reshape failed")
        }

        /// SwiGLU feed-forward: `linear_out(silu(a) * b)` where `[a,b] =
        /// split(linear_in(x), 2)`.
        fn feedforward(&self, x: &Array, w: &FeedForwardW) -> Result<Array> {
            let h = w.linear_in.apply(x)?;
            let parts = split(&h, 2, -1).context("ff split failed")?;
            let g = silu(&parts[0])?;
            let gated = g.multiply(&parts[1]).context("ff gate mul failed")?;
            w.linear_out.apply(&gated)
        }

        /// Shared modulation: returns `sets` groups of (shift, scale, gate),
        /// each `[B,1,inner]`.
        fn modulation(&self, temb: &Array, w: &Linear, sets: usize) -> Result<Vec<[Array; 3]>> {
            let m = silu(temb)?;
            let m = w.apply(&m)?; // [B, inner*3*sets]
            let b = m.dim(0);
            let inner = self.cfg.inner_dim() as i32;
            // expand axis 1 -> [B,1,inner*3*sets]
            let m = m
                .reshape(&[b, 1, inner * 3 * (sets as i32)])
                .context("mod expand")?;
            let parts = split(&m, 3 * sets as i32, -1).context("mod split failed")?;
            let mut out = Vec::with_capacity(sets);
            for i in 0..sets {
                out.push([
                    parts[3 * i].clone(),
                    parts[3 * i + 1].clone(),
                    parts[3 * i + 2].clone(),
                ]);
            }
            Ok(out)
        }

        /// `(1+scale)*ln(x) + shift`.
        fn modulate(&self, x: &Array, scale: &Array, shift: &Array) -> Result<Array> {
            let n = layer_norm_no_affine(x, self.cfg.eps_ln)?;
            let one = Array::from_slice(&[1.0f32], &[1]);
            let one_plus = scale.add(&one).context("1+scale failed")?;
            n.multiply(&one_plus)
                .and_then(|m| m.add(shift))
                .context("modulate failed")
        }

        // ── blocks ────────────────────────────────────────────────────────────

        #[allow(clippy::too_many_arguments)]
        fn double_attn(
            &self,
            nh: &Array,
            ne: &Array,
            w: &DoubleAttnW,
            cos: &Array,
            sin: &Array,
        ) -> Result<(Array, Array)> {
            let st = ne.dim(1); // txt seq len
            // image stream
            let mut q = self.heads(&w.to_q.apply(nh)?)?;
            let mut k = self.heads(&w.to_k.apply(nh)?)?;
            let v = self.heads(&w.to_v.apply(nh)?)?;
            q = self.rmsnorm(&q, &w.norm_q)?;
            k = self.rmsnorm(&k, &w.norm_k)?;
            // text stream
            let mut eq = self.heads(&w.add_q.apply(ne)?)?;
            let mut ek = self.heads(&w.add_k.apply(ne)?)?;
            let ev = self.heads(&w.add_v.apply(ne)?)?;
            eq = self.rmsnorm(&eq, &w.norm_added_q)?;
            ek = self.rmsnorm(&ek, &w.norm_added_k)?;
            // concat [enc, img] along seq axis (axis 2 in BHSD)
            let q = concatenate_axis(&[&eq, &q], 2).context("double q concat")?;
            let k = concatenate_axis(&[&ek, &k], 2).context("double k concat")?;
            let v = concatenate_axis(&[&ev, &v], 2).context("double v concat")?;
            let (q, k) = apply_rope_bshd(&q, &k, cos, sin)?;
            let out = self.sdpa(&q, &k, &v)?; // [B, st+si, inner]
            // split back [enc | img]
            let parts = split_sections(&out, &[st], 1).context("double out split")?;
            let enc_out = w.to_add_out.apply(&parts[0])?;
            let img_out = w.to_out.apply(&parts[1])?;
            Ok((img_out, enc_out))
        }

        #[allow(clippy::too_many_arguments)]
        fn double_block(
            &self,
            hidden: &Array,
            enc: &Array,
            w: &DoubleBlockW,
            mod_img: &[[Array; 3]],
            mod_txt: &[[Array; 3]],
            cos: &Array,
            sin: &Array,
        ) -> Result<(Array, Array)> {
            let [shift_msa, scale_msa, gate_msa] = &mod_img[0];
            let [shift_mlp, scale_mlp, gate_mlp] = &mod_img[1];
            let [c_shift_msa, c_scale_msa, c_gate_msa] = &mod_txt[0];
            let [c_shift_mlp, c_scale_mlp, c_gate_mlp] = &mod_txt[1];

            let nh = self.modulate(hidden, scale_msa, shift_msa)?;
            let ne = self.modulate(enc, c_scale_msa, c_shift_msa)?;
            let (attn_out, enc_attn_out) = self.double_attn(&nh, &ne, &w.attn, cos, sin)?;
            let hidden = hidden
                .add(&gate_msa.multiply(&attn_out).context("gate_msa mul")?)
                .context("hidden + gate*attn")?;
            let enc = enc
                .add(
                    &c_gate_msa
                        .multiply(&enc_attn_out)
                        .context("c_gate_msa mul")?,
                )
                .context("enc + c_gate*attn")?;

            let nh = self.modulate(&hidden, scale_mlp, shift_mlp)?;
            let ff = self.feedforward(&nh, &w.ff)?;
            let hidden = hidden
                .add(&gate_mlp.multiply(&ff).context("gate_mlp mul")?)
                .context("hidden + gate*ff")?;

            let ne = self.modulate(&enc, c_scale_mlp, c_shift_mlp)?;
            let ffc = self.feedforward(&ne, &w.ff_context)?;
            let enc = enc
                .add(&c_gate_mlp.multiply(&ffc).context("c_gate_mlp mul")?)
                .context("enc + c_gate*ffc")?;
            Ok((enc, hidden))
        }

        fn single_block(
            &self,
            hidden: &Array,
            w: &SingleBlockW,
            mod_single: &[Array; 3],
            cos: &Array,
            sin: &Array,
        ) -> Result<Array> {
            let [shift, scale, gate] = mod_single;
            let nh = self.modulate(hidden, scale, shift)?;
            let proj = w.to_qkv_mlp.apply(&nh)?;
            let inner = self.cfg.inner_dim() as i32;
            // split [qkv (inner*3) | mlp (mlp_hidden*2)]
            let split1 = split_sections(&proj, &[inner * 3], -1).context("single qkv|mlp split")?;
            let qkv = &split1[0];
            let mlp_hidden = &split1[1];
            let qkv_parts = split(qkv, 3, -1).context("single qkv split")?;
            let q = self.heads(&qkv_parts[0])?;
            let k = self.heads(&qkv_parts[1])?;
            let v = self.heads(&qkv_parts[2])?;
            let q = self.rmsnorm(&q, &w.norm_q)?;
            let k = self.rmsnorm(&k, &w.norm_k)?;
            let (q, k) = apply_rope_bshd(&q, &k, cos, sin)?;
            let attn = self.sdpa(&q, &k, &v)?; // [B,S,inner]
            // SwiGLU on mlp half
            let mlp_parts = split(mlp_hidden, 2, -1).context("single mlp split")?;
            let mlp = silu(&mlp_parts[0])?
                .multiply(&mlp_parts[1])
                .context("single mlp gate")?;
            let cat = concatenate_axis(&[&attn, &mlp], -1).context("single concat attn|mlp")?;
            let out = w.to_out.apply(&cat)?;
            hidden
                .add(&gate.multiply(&out).context("single gate mul")?)
                .context("single residual")
        }

        // ── forward ───────────────────────────────────────────────────────────

        /// Full DiT forward.
        ///
        /// - `latent`: `[B, S_img, in_channels]`
        /// - `text`:   `[B, S_txt, joint_dim]`
        /// - `timestep`: scalar in [0,1] (or already scaled) — scaled ×1000 if ≤1.
        /// - `img_ids`/`txt_ids`: row-major `[S, 4]` i32 position ids.
        /// - returns `[B, S_img, out_channels]`.
        pub fn forward(
            &self,
            latent: &Array,
            text: &Array,
            timestep: f32,
            guidance: Option<f32>,
            img_ids: &[i32],
            txt_ids: &[i32],
        ) -> Result<Array> {
            let (out, _d0, _sg) =
                self.forward_inner(latent, text, timestep, guidance, img_ids, txt_ids, false)?;
            Ok(out)
        }

        /// Forward + staged intermediates for parity debugging.
        /// Returns (output, hidden after first double block (image stream),
        /// joint hidden after all single blocks).
        #[cfg(test)]
        pub fn forward_debug(
            &self,
            latent: &Array,
            text: &Array,
            timestep: f32,
            img_ids: &[i32],
            txt_ids: &[i32],
        ) -> Result<(Array, Array, Array)> {
            let (out, d0, sg) =
                self.forward_inner(latent, text, timestep, None, img_ids, txt_ids, true)?;
            Ok((out, d0, sg))
        }

        #[allow(clippy::too_many_arguments)]
        fn forward_inner(
            &self,
            latent: &Array,
            text: &Array,
            timestep: f32,
            guidance: Option<f32>,
            img_ids: &[i32],
            txt_ids: &[i32],
            capture: bool,
        ) -> Result<(Array, Array, Array)> {
            let b = latent.dim(0);
            let s_img = latent.dim(1) as usize;
            let s_txt = text.dim(1) as usize;

            // timestep: scale ×1000 if ≤1, then sinusoid + MLP.
            let ts = if timestep <= 1.0 {
                timestep * 1000.0
            } else {
                timestep
            };
            let ts_arr = Array::from_slice(&vec![ts; b as usize], &[b]);
            let temb = timestep_embedding(&ts_arr, self.cfg.ts_channels)?;
            let temb = self.ts_linear_1.apply(&temb)?;
            let temb = silu(&temb)?;
            let mut temb = self.ts_linear_2.apply(&temb)?; // [B, inner]

            // dev guidance branch: temb += guidance_mlp(sinusoid(guidance)).
            // mflux scales the guidance scalar ×1000 only if it is ≤1 (a raw
            // guidance like 4.0 is left as-is).
            if let (Some(g1), Some(g2), Some(g)) =
                (&self.guidance_linear_1, &self.guidance_linear_2, guidance)
            {
                let gs = if g <= 1.0 { g * 1000.0 } else { g };
                let g_arr = Array::from_slice(&vec![gs; b as usize], &[b]);
                let gemb = timestep_embedding(&g_arr, self.cfg.ts_channels)?;
                let gemb = g1.apply(&gemb)?;
                let gemb = silu(&gemb)?;
                let gemb = g2.apply(&gemb)?;
                temb = temb.add(&gemb).context("temb + guidance_emb failed")?;
            }
            let temb = temb;

            let mut hidden = self.x_embedder.apply(latent)?;
            let mut enc = self.context_embedder.apply(text)?;

            // 4D RoPE tables, joint [txt;img].
            let (img_cos, img_sin) =
                pos_embed(img_ids, s_img, &self.cfg.axes_dim, self.cfg.rope_theta)?;
            let (txt_cos, txt_sin) =
                pos_embed(txt_ids, s_txt, &self.cfg.axes_dim, self.cfg.rope_theta)?;
            let cos = concat_seq(&txt_cos, &img_cos)?;
            let sin = concat_seq(&txt_sin, &img_sin)?;

            let mod_img = self.modulation(&temb, &self.mod_img, 2)?;
            let mod_txt = self.modulation(&temb, &self.mod_txt, 2)?;
            let mod_single = self.modulation(&temb, &self.mod_single, 1)?;
            let mod_single = &mod_single[0];

            let mut after_double0 = None;
            for (i, blk) in self.double_blocks.iter().enumerate() {
                let (e, h) =
                    self.double_block(&hidden, &enc, blk, &mod_img, &mod_txt, &cos, &sin)?;
                enc = e;
                hidden = h;
                if capture && i == 0 {
                    after_double0 = Some(hidden.clone());
                }
            }

            // joint [txt;img]
            let mut joint = concatenate_axis(&[&enc, &hidden], 1).context("joint concat")?;
            for blk in &self.single_blocks {
                joint = self.single_block(&joint, blk, mod_single, &cos, &sin)?;
            }
            let after_single = if capture { Some(joint.clone()) } else { None };

            // image part, norm_out (AdaLN-continuous), proj_out.
            let img_part =
                split_sections(&joint, &[s_txt as i32], 1).context("split image part failed")?;
            let img = &img_part[1];

            // AdaLN-continuous: silu(temb) -> linear -> [scale|shift].
            let t = self.norm_out.apply(&silu(&temb)?)?; // [B, inner*2]
            let inner = self.cfg.inner_dim() as i32;
            let sc_sh = split_sections(&t, &[inner], -1).context("norm_out split")?;
            let scale = sc_sh[0]
                .reshape(&[b, 1, inner])
                .context("norm_out scale reshape")?;
            let shift = sc_sh[1]
                .reshape(&[b, 1, inner])
                .context("norm_out shift reshape")?;
            let normed = layer_norm_no_affine(img, self.cfg.eps_ln)?;
            let one = Array::from_slice(&[1.0f32], &[1]);
            let one_plus = scale.add(&one).context("norm_out 1+scale")?;
            let out = normed
                .multiply(&one_plus)
                .and_then(|m| m.add(&shift))
                .context("norm_out affine")?;
            let out = self.proj_out.apply(&out)?;
            out.eval().context("dit forward: final eval")?;

            let _ = s_img;
            Ok((
                out,
                after_double0.unwrap_or_else(|| hidden.clone()),
                after_single.unwrap_or_else(|| joint.clone()),
            ))
        }
    }
}

#[cfg(feature = "mlx-native")]
pub use imp::{Dit, DitConfig};

// ── parity test against the golden reference (/tmp/dit_*.bin) ─────────────────
#[cfg(all(test, feature = "mlx-native"))]
mod parity_tests {
    use super::imp::{Dit, DitConfig};
    use mlx_rs::Array;

    // Klein DiT (bf16), resolved from the local HF cache.
    fn dit_path() -> std::path::PathBuf {
        crate::hf_cache::snapshot_path_or_rel(
            crate::repos::VAE,
            "transformer/diffusion_pytorch_model.safetensors",
        )
    }

    fn read_f32_bin(path: &str) -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        assert_eq!(bytes.len() % 4, 0, "{path} not f32-aligned");
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn read_i32_bin(path: &str) -> Vec<i32> {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        assert_eq!(bytes.len() % 4, 0, "{path} not i32-aligned");
        bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// cosine + mean-abs over flat buffers.
    fn metrics(label: &str, obs: &[f32], exp: &[f32]) -> (f64, f64) {
        assert_eq!(obs.len(), exp.len(), "{label}: size mismatch");
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        let mut sad = 0.0f64;
        for (&g, &e) in obs.iter().zip(exp.iter()) {
            dot += g as f64 * e as f64;
            na += g as f64 * g as f64;
            nb += e as f64 * e as f64;
            sad += (g - e).abs() as f64;
        }
        let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
        let mean_abs = sad / obs.len() as f64;
        println!("{label}: cosine={cos:.6} mean_abs={mean_abs:.6e}");
        (cos, mean_abs)
    }

    #[test]
    #[ignore = "needs a Metal device AND the /tmp/dit_*.bin dump set (latent, \
                textemb, imgids, txtids, after_double0, after_single, out); no \
                committed script produces them — see \
                crates/lumen-mlx/tests/golden/README.md"]
    fn dit_matches_reference() {
        let cfg = DitConfig::klein_4b();
        let s_img = 64usize;
        let s_txt = 16usize;

        let latent = Array::from_slice(
            &read_f32_bin("/tmp/dit_latent.bin"),
            &[1, s_img as i32, 128],
        );
        let text = Array::from_slice(
            &read_f32_bin("/tmp/dit_textemb.bin"),
            &[1, s_txt as i32, 7680],
        );
        let img_ids = read_i32_bin("/tmp/dit_imgids.bin");
        let txt_ids = read_i32_bin("/tmp/dit_txtids.bin");
        latent.eval().unwrap();
        text.eval().unwrap();

        let dit = Dit::load(dit_path(), cfg).expect("load DiT weights");
        let (out, d0, sg) = dit
            .forward_debug(&latent, &text, 0.5, &img_ids, &txt_ids)
            .expect("forward");
        assert_eq!(out.shape(), &[1, s_img as i32, 128], "output shape");
        out.eval().unwrap();
        d0.eval().unwrap();
        sg.eval().unwrap();

        // staged: after first double block (image stream) [1,64,3072]
        let exp_d0 = read_f32_bin("/tmp/dit_after_double0.bin");
        let (cos_d0, _) = metrics("after_double0", d0.as_slice(), &exp_d0);
        assert!(cos_d0 > 0.999, "after_double0 cosine {cos_d0:.6} <= 0.999");

        // staged: after all single blocks (joint [txt;img]) [1,80,3072]
        let exp_sg = read_f32_bin("/tmp/dit_after_single.bin");
        let (cos_sg, _) = metrics("after_single", sg.as_slice(), &exp_sg);
        assert!(cos_sg > 0.999, "after_single cosine {cos_sg:.6} <= 0.999");

        // final output [1,64,128]
        let exp_out = read_f32_bin("/tmp/dit_out.bin");
        let (cos_out, mae_out) = metrics("output", out.as_slice(), &exp_out);
        assert!(cos_out > 0.999, "output cosine {cos_out:.6} <= 0.999");
        assert!(mae_out < 5e-3, "output mean_abs {mae_out:.6e} >= 5e-3");
    }
}
